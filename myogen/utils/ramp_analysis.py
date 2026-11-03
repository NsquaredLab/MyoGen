# myogen/utils/ramp_analysis.py
"""
Extract recruitment curves from trapezoidal ramp simulations.

Analyzes spike trains produced during a time-varying drive signal
to count active MUs and compute mean firing rates at %MVC checkpoints.
"""

import numpy as np
from neo import Block


def extract_ramp_recruitment(
    spike_block: Block,
    drive_signal: np.ndarray,
    threshold_hz: float,
    max_drive_hz: float,
    timestep_ms: float = 0.1,
    checkpoints_pct: list[float] | None = None,
    window_ms: float = 500.0,
    min_spikes: int = 3,
) -> dict[float, dict]:
    """Extract recruitment data at %MVC checkpoints during ramp.

    Parameters
    ----------
    spike_block : neo.Block
        Block with one segment containing spike trains from the ramp sim.
    drive_signal : ndarray
        The time-varying drive signal (Hz) used in the simulation.
    threshold_hz : float
        DD recruitment threshold (Hz).
    max_drive_hz : float
        Maximum DD drive (Hz) corresponding to 100% MVC.
    timestep_ms : float
        Simulation timestep in ms.
    checkpoints_pct : list of float
        %MVC levels to extract data at. Default [5, 10, 20, 30].
    window_ms : float
        Sliding window width for counting spikes (ms).
    min_spikes : int
        Minimum spikes in window to count a MU as "active".

    Returns
    -------
    dict
        Mapping mvc_pct -> {"n_active": int, "mean_fr_hz": float}
    """
    if checkpoints_pct is None:
        checkpoints_pct = [5, 10, 20, 30]

    spiketrains = spike_block.segments[0].spiketrains
    window_s = window_ms / 1000.0

    def mvc_to_hz(pct):
        return threshold_hz + (pct / 100.0) * (max_drive_hz - threshold_hz)

    results = {}
    for mvc_pct in checkpoints_pct:
        target_hz = mvc_to_hz(mvc_pct)

        # Find the time window where drive is closest to target_hz
        # (during the ramp-up phase — first occurrence)
        drive_diff = np.abs(drive_signal - target_hz)
        midpoint = len(drive_signal) // 2
        ramp_up_diff = drive_diff[:midpoint]
        best_step = int(np.argmin(ramp_up_diff))
        center_s = best_step * timestep_ms / 1000.0

        win_start = center_s - window_s / 2
        win_end = center_s + window_s / 2

        n_active = 0
        firing_rates = []
        for st in spiketrains:
            spike_times = np.asarray(st.magnitude)
            in_window = spike_times[(spike_times >= win_start) & (spike_times < win_end)]
            if len(in_window) >= min_spikes:
                n_active += 1
                if len(in_window) >= 2:
                    isis = np.diff(in_window)
                    fr = 1.0 / np.mean(isis)
                    firing_rates.append(fr)

        mean_fr = float(np.mean(firing_rates)) if firing_rates else 0.0
        results[mvc_pct] = {"n_active": n_active, "mean_fr_hz": mean_fr}

    return results


def recruitment_force_pct_mvc(
    spike_times_s,
    force_pct_mvc: np.ndarray,
    force_fs_hz: float,
    window_s: float = 0.5,
    min_spikes: int = 3,
) -> np.ndarray:
    """Recruitment force of each motor unit on an MVC ramp, in % MVC.

    A unit is recruited at its first spike that starts ``min_spikes`` spikes
    within ``window_s`` (the rule of ``extract_ramp_recruitment``); its
    recruitment force is the force at that spike. Units that never meet the
    rule get NaN.

    Parameters
    ----------
    spike_times_s : sequence of array_like
        Spike times (s) of each unit, in pool order.
    force_pct_mvc : ndarray
        Force during the ramp in % MVC (e.g. ``ForceModel`` output divided by
        its MVC), sampled at ``force_fs_hz`` from t = 0.
    force_fs_hz : float
        Sampling rate of ``force_pct_mvc`` (Hz).
    """
    force = np.asarray(force_pct_mvc, dtype=float)
    out = np.full(len(spike_times_s), np.nan)
    for unit, times in enumerate(spike_times_s):
        t = np.sort(np.asarray(times, dtype=float))
        n = min_spikes - 1
        if len(t) <= n:
            continue
        sustained = np.flatnonzero(t[n:] - t[:-n] <= window_s)
        if len(sustained):
            index = min(int(round(t[sustained[0]] * force_fs_hz)), len(force) - 1)
            out[unit] = force[index]
    return out


def cv_from_recruitment_force(
    recruitment_pct_mvc: np.ndarray,
    low_band_pct: tuple[float, float] = (0.0, 30.0),
    high_band_pct: tuple[float, float] = (50.0, 70.0),
    cv_low__m_s: float = 3.89,
    cv_high__m_s: float = 5.12,
    unrecruited_pct_mvc: float = 100.0,
    residual_sd__m_s: float = 0.39,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Motor-unit conduction velocity linear in recruitment force.

    Del Vecchio et al. (2018) measured CV rising linearly with recruitment
    threshold in tibialis anterior with high-density surface EMG: 3.89 m/s for
    units recruited at 0-30 % MVC and 5.12 m/s at 50-70 % MVC
    [delvecchio2018 p. 2]. The line ``cv = a + b * rt`` passes through those
    values at the mean recruitment force of this pool's units in each band
    (a derivation: the paper gives no band-mean thresholds). Units with NaN
    recruitment force (not recruited on the ramp) are placed at
    ``unrecruited_pct_mvc``.

    Each unit then scatters about the line by ``N(0, residual_sd__m_s)``. The
    default 0.39 m/s is derived, not measured: CV is linear in threshold with
    R² 0.70 ± 0.09 and SD 0.71 m/s over units [delvecchio2018 p. 2], so
    sqrt(1 - 0.70) * 0.71 of it is left unexplained. Pass 0 for the line alone.
    ``rng`` defaults to myogen's global generator (``set_random_seed``).

    Returns
    -------
    cv : ndarray
        CV of each unit (m/s), e.g. for ``SurfaceFiberDispersion(fiber_cv_per_unit__m_s=...)``.
    (a, b) : tuple of float
        Intercept (m/s) and slope (m/s per % MVC).
    """
    rt = np.asarray(recruitment_pct_mvc, dtype=float)
    rt = np.where(np.isnan(rt), unrecruited_pct_mvc, rt)
    low = (rt >= low_band_pct[0]) & (rt < low_band_pct[1])
    high = (rt >= high_band_pct[0]) & (rt <= high_band_pct[1])
    if not low.any() or not high.any():
        raise ValueError("cv_from_recruitment_force needs recruited units in both bands")
    x_low, x_high = rt[low].mean(), rt[high].mean()
    b = (cv_high__m_s - cv_low__m_s) / (x_high - x_low)
    a = cv_low__m_s - b * x_low
    cv = a + b * rt
    if residual_sd__m_s > 0:
        if rng is None:
            from myogen import get_random_generator

            rng = get_random_generator()
        cv = cv + rng.normal(0.0, residual_sd__m_s, len(cv))
    return cv, (float(a), float(b))
