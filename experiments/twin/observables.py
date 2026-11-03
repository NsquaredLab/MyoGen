"""Observables for the motor-neuron-pool digital twin.

Two tiers:

* **Differentiable** (JAX, operate on the surrogate/rate spike matrices coming out
  of ``run_jax``): smoothed firing rates, population rate, force features. These
  are what inverse-problem losses are built from (Phase 3+).
* **Analysis** (NumPy, hard spikes): recruitment/derecruitment thresholds and
  order, firing-rate hysteresis, ΔF via the paired-motor-unit (Gorassini) method.
  These are for validation (Phase 1) and for scoring fits — not for gradients.

Conventions
-----------
``spikes``: array ``(n_steps, n_mn)``, 1.0 at spike samples (hard mode) or a
surrogate/rate value in optimization modes. ``dt_ms``: timestep in milliseconds.
All times returned in seconds.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

__all__ = [
    "smoothed_rates",
    "population_rate",
    "spike_times",
    "recruitment_times",
    "derecruitment_times",
    "recruitment_order",
    "delta_f_pairs",
    "DeltaFResult",
    "hysteresis_index",
]


# --------------------------------------------------------------------------- #
# Differentiable tier (JAX)
# --------------------------------------------------------------------------- #

def smoothed_rates(spikes, dt_ms: float, tau_ms: float = 250.0):
    """Per-MN instantaneous firing rate (Hz) via causal exponential smoothing.

    Differentiable in ``spikes`` (use surrogate/rate spike mode upstream).
    Implemented as a first-order IIR scanned over time:
    ``r[t] = a * r[t-1] + (1 - a) * s[t] / dt_s`` with ``a = exp(-dt/tau)``.

    Parameters
    ----------
    spikes : (n_steps, n_mn)
    tau_ms : smoothing time constant. 250 ms ≈ the smoothing used for ΔF-style
        rate profiles; use smaller values for transient-sensitive losses.
    """
    dt_s = dt_ms * 1e-3
    a = jnp.exp(-dt_ms / tau_ms)
    inst = spikes / dt_s  # spike-count density → Hz

    import jax

    def step(r, x):
        r = a * r + (1.0 - a) * x
        return r, r

    r0 = jnp.zeros(spikes.shape[1], dtype=spikes.dtype)
    _, rates = jax.lax.scan(step, r0, inst)
    return rates


def population_rate(spikes, dt_ms: float, tau_ms: float = 100.0):
    """Mean smoothed firing rate across the pool, shape ``(n_steps,)``."""
    return smoothed_rates(spikes, dt_ms, tau_ms).mean(axis=1)


# --------------------------------------------------------------------------- #
# Analysis tier (NumPy, hard spikes)
# --------------------------------------------------------------------------- #

def spike_times(spikes, dt_ms: float) -> list[np.ndarray]:
    """Per-MN spike times in seconds from a hard 0/1 spike matrix."""
    sp = np.asarray(spikes) > 0.5
    dt_s = dt_ms * 1e-3
    return [np.flatnonzero(sp[:, i]) * dt_s for i in range(sp.shape[1])]


def _first_sustained(train_s: np.ndarray, min_spikes: int, max_isi_s: float):
    """First spike time that begins a run of ``min_spikes`` spikes with all
    ISIs below ``max_isi_s`` — filters out isolated spikes near threshold."""
    if len(train_s) < min_spikes:
        return np.nan
    for k in range(len(train_s) - min_spikes + 1):
        isis = np.diff(train_s[k : k + min_spikes])
        if np.all(isis <= max_isi_s):
            return train_s[k]
    return np.nan


def recruitment_times(
    spikes, dt_ms: float, min_spikes: int = 3, max_isi_s: float = 0.5
) -> np.ndarray:
    """Per-MN recruitment time (s): onset of first sustained firing; NaN if silent."""
    return np.array(
        [_first_sustained(t, min_spikes, max_isi_s) for t in spike_times(spikes, dt_ms)]
    )


def derecruitment_times(
    spikes, dt_ms: float, min_spikes: int = 3, max_isi_s: float = 0.5
) -> np.ndarray:
    """Per-MN derecruitment time (s): end of the last sustained firing run."""
    out = []
    for t in spike_times(spikes, dt_ms):
        rev = _first_sustained(t[::-1] * -1.0, min_spikes, max_isi_s)
        out.append(np.nan if np.isnan(rev) else -rev)
    return np.array(out)


def recruitment_order(spikes, dt_ms: float) -> np.ndarray:
    """Indices of recruited MNs sorted by recruitment time (silent MNs excluded)."""
    rt = recruitment_times(spikes, dt_ms)
    active = np.flatnonzero(~np.isnan(rt))
    return active[np.argsort(rt[active])]


def _rate_profile(train_s: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Smoothed instantaneous rate (Hz) on ``t_grid`` from spike times:
    1/ISI held between spikes, then 500 ms moving average (standard for ΔF)."""
    r = np.zeros_like(t_grid)
    if len(train_s) >= 2:
        inst = 1.0 / np.diff(train_s)
        idx = np.searchsorted(train_s, t_grid, side="right") - 1
        valid = (idx >= 0) & (idx < len(inst))
        r[valid] = inst[np.clip(idx, 0, len(inst) - 1)][valid]
        # zero outside firing span
        r[(t_grid < train_s[0]) | (t_grid > train_s[-1])] = 0.0
    if len(t_grid) > 1:
        dt = t_grid[1] - t_grid[0]
        w = max(1, int(round(0.5 / dt)))
        kernel = np.ones(w) / w
        r = np.convolve(r, kernel, mode="same")
    return r


@dataclass
class DeltaFResult:
    """ΔF estimates for control/test MN pairs on a ramp protocol.

    ``delta_f`` holds every accepted pair (NOT statistically independent —
    pairs share control and test units). ``delta_f_per_test`` averages the
    controls within each test unit first; ``mean`` is the mean of those
    per-test values, so every test unit carries equal weight.
    """

    pairs: np.ndarray  # (n_pairs, 2) [control_idx, test_idx]
    delta_f: np.ndarray  # (n_pairs,) Hz
    t_recruit_test: np.ndarray
    t_derecruit_test: np.ndarray
    test_units: np.ndarray  # unique test MN indices, aligned with delta_f_per_test
    delta_f_per_test: np.ndarray  # (n_test_units,) controls averaged within test

    @property
    def mean(self) -> float:
        return (float(np.nanmean(self.delta_f_per_test))
                if len(self.delta_f_per_test) else np.nan)


def delta_f_pairs(
    spikes,
    dt_ms: float,
    min_lead_s: float = 1.0,
    min_ctrl_rate_hz: float = 0.5,
    max_pairs_per_test: int = 5,
    min_control_outlast_s: float = 1.5,
    min_rate_correlation: float = 0.7,
    min_ctrl_modulation_hz: float = 0.5,
) -> DeltaFResult:
    """ΔF via the paired-motor-unit method (Gorassini et al. 2002).

    For each accepted (control, test) pair:

        ΔF = f_control(t_recruit_test) − f_control(t_derecruit_test)

    i.e. the drop in synaptic drive (proxied by the control MN's rate) between
    when the test MN turned on and when it turned off. A positive ΔF on a
    symmetric ramp indicates sustained depolarization from PICs. Control rate
    profiles use the 1/ISI + 500 ms smoothing convention.

    Pair-acceptance safeguards (cf. Gorassini 2002; Afsharipour et al. 2020):

    * control recruited ≥ ``min_lead_s`` before the test MN;
    * control derecruited ≥ ``min_control_outlast_s`` AFTER the test MN — its
      rate must remain a meaningful drive proxy at test derecruitment;
    * rate–rate correlation between control and test profiles over the test
      MN's firing span ≥ ``min_rate_correlation`` (common-modulation check);
    * control rate modulation (max − min over the test window) ≥
      ``min_ctrl_modulation_hz`` (saturation / sensitivity check);
    * control firing (≥ ``min_ctrl_rate_hz``) at both endpoints.

    Controls are averaged within each test unit before the population mean
    (``DeltaFResult.mean``), so test units with many valid controls are not
    over-weighted. Raw pair values remain in ``delta_f``.
    """
    trains = spike_times(spikes, dt_ms)
    n_steps = np.asarray(spikes).shape[0]
    t_grid = np.arange(n_steps) * dt_ms * 1e-3
    rt = recruitment_times(spikes, dt_ms)
    dt_de = derecruitment_times(spikes, dt_ms)

    pairs, dfs, trs, tds = [], [], [], []
    active = np.flatnonzero(~np.isnan(rt))
    profiles: dict[int, np.ndarray] = {}

    def prof(i: int) -> np.ndarray:
        if i not in profiles:
            profiles[i] = _rate_profile(trains[i], t_grid)
        return profiles[i]

    for test in active:
        if np.isnan(dt_de[test]):
            continue
        ctrls = [
            c
            for c in active
            if c != test
            and rt[c] <= rt[test] - min_lead_s
            and not np.isnan(dt_de[c])
            and dt_de[c] >= dt_de[test] + min_control_outlast_s
        ]
        # prefer the controls recruited closest before the test MN
        ctrls = sorted(ctrls, key=lambda c: rt[test] - rt[c])[:max_pairs_per_test]
        i_r = min(int(np.searchsorted(t_grid, rt[test])), n_steps - 1)
        i_d = min(int(np.searchsorted(t_grid, dt_de[test])), n_steps - 1)
        if i_d <= i_r:
            continue
        test_prof = prof(test)
        for c in ctrls:
            p = prof(c)
            if p[i_r] < min_ctrl_rate_hz or p[i_d] < min_ctrl_rate_hz:
                continue
            seg_c, seg_t = p[i_r:i_d], test_prof[i_r:i_d]
            if seg_c.max() - seg_c.min() < min_ctrl_modulation_hz:
                continue  # control saturated / insensitive over the window
            if seg_c.std() == 0 or seg_t.std() == 0:
                continue  # flat profile → common modulation undefined
            r = float(np.corrcoef(seg_c, seg_t)[0, 1])
            if not np.isfinite(r) or r < min_rate_correlation:
                continue  # no (or undefined) common modulation
            pairs.append((c, test))
            dfs.append(p[i_r] - p[i_d])
            trs.append(rt[test])
            tds.append(dt_de[test])

    pairs_arr = np.array(pairs).reshape(-1, 2)
    dfs_arr = np.array(dfs)
    if len(pairs_arr):
        test_units = np.unique(pairs_arr[:, 1])
        per_test = np.array(
            [dfs_arr[pairs_arr[:, 1] == tu].mean() for tu in test_units]
        )
    else:
        test_units = np.array([], dtype=int)
        per_test = np.array([])

    return DeltaFResult(
        pairs=pairs_arr,
        delta_f=dfs_arr,
        t_recruit_test=np.array(trs),
        t_derecruit_test=np.array(tds),
        test_units=test_units,
        delta_f_per_test=per_test,
    )


def hysteresis_index(
    drive: np.ndarray, spikes, dt_ms: float
) -> np.ndarray:
    """Per-MN derecruitment−recruitment drive difference on a ramp.

    ``drive`` is the scalar command time-series ``(n_steps,)`` (e.g. DD rate).
    Negative values (derecruited at *lower* drive than recruitment) indicate
    sustained depolarization / PIC hysteresis.
    """
    drive = np.asarray(drive)
    t = np.arange(len(drive)) * dt_ms * 1e-3
    rt = recruitment_times(spikes, dt_ms)
    dtd = derecruitment_times(spikes, dt_ms)
    out = np.full(len(rt), np.nan)
    for i, (a, b) in enumerate(zip(rt, dtd)):
        if not (np.isnan(a) or np.isnan(b)):
            out[i] = drive[np.searchsorted(t, b) - 1] - drive[np.searchsorted(t, a)]
    return out
