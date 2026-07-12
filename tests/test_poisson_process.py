"""Statistical validation that the descending drive emits a *true* Poisson process.

Regression guard for the bug where ``poisson_batch_size`` / ``N > 1`` silently turned the
"Poisson" generator into a ``Gamma(N, N)`` renewal process (ISI CV = ``1/sqrt(N)`` instead
of 1). The generator now draws a single ``Exp(1)`` inter-spike threshold, so at a constant
drive the inter-spike intervals (ISIs) must be exponentially distributed with CV = 1.

The tests exercise the process end-to-end through the public API — a real
``DescendingDrive__Pool`` whose cells are driven with ``cell.integrate(rate)`` exactly as the
simulation loop does — and check every defining property of a homogeneous Poisson process:

* the ISIs are exponentially distributed (KS goodness-of-fit),
* CV(ISI) = 1,
* spike counts in fixed windows are Poisson-distributed (Fano factor = 1),
* the ISIs are serially independent (memoryless / renewal),
* the mean rate matches the requested drive.

The Gamma process (``process_type="gamma"``) is validated symmetrically: its ISIs are shown to
follow a *genuine* Gamma(shape) distribution (shape recovered, KS against Gamma not rejected, KS
against a wrong shape rejected) and to be rejected as Exponential — both as a check in its own
right and to prove the Poisson tests have discriminating power. A kernel-level layer localises any
failure to the public path vs the Cython generator.
"""

from __future__ import annotations

import numpy as np
import pytest
import quantities as pq
from scipy import stats

import myogen
from myogen.simulator.neuron import cells
from myogen.simulator.neuron._cython._poisson_process_generator import (
    _PoissonProcessGenerator__Cython,
)
from myogen.simulator.neuron.populations.descending_drive import DescendingDrive__Pool

DT_MS = 0.1
SEED = 42
RATE_HZ = 25.0
DURATION_MS = 30_000.0
N_CELLS = 20


def _drive_pool(pool, rate_hz, duration_ms):
    """Drive every cell in ``pool`` at a constant rate, as the simulation loop does.

    Returns per-cell lists of spike step-indices. ``cell.integrate`` advances the cell's own
    generator by its internal ``dt``, so a plain Python loop reproduces the real spike-generation
    path without needing NEURON's ``fadvance`` (which only steps the postsynaptic dynamics).
    """
    n = len(list(pool))
    n_steps = int(duration_ms / DT_MS)
    spike_steps = [[] for _ in range(n)]
    for step in range(n_steps):
        for cell in pool:
            if cell.integrate(rate_hz):
                spike_steps[cell.pool__ID].append(step)
    return spike_steps


def _pooled_isis(spike_steps):
    """Concatenate per-cell ISIs (seconds); each cell's ISIs are i.i.d. so pooling is valid."""
    dt_s = DT_MS * 1e-3
    per_cell = [np.diff(np.asarray(s, dtype=float) * dt_s) for s in spike_steps]
    return np.concatenate([d for d in per_cell if d.size > 2])


# --------------------------------------------------------------------------------------------
# End-to-end: a real DescendingDrive__Pool with process_type="poisson"
# --------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def poisson_pool_spikes():
    myogen.set_random_seed(SEED)
    pool = DescendingDrive__Pool(
        n=N_CELLS, timestep__ms=DT_MS * pq.ms, process_type="poisson"
    )
    return _drive_pool(pool, RATE_HZ, DURATION_MS)


def test_pool_mean_rate_matches_drive(poisson_pool_spikes):
    isis = _pooled_isis(poisson_pool_spikes)
    assert 1.0 / isis.mean() == pytest.approx(RATE_HZ, rel=0.05)


def test_pool_isi_cv_is_one(poisson_pool_spikes):
    """Exponential ISIs have CV = 1. The old Gamma(N, N) bug gave CV = 1/sqrt(N) (0.45 at N=5)."""
    isis = _pooled_isis(poisson_pool_spikes)
    assert isis.std() / isis.mean() == pytest.approx(1.0, abs=0.05)


def test_pool_isis_are_exponential(poisson_pool_spikes):
    """KS goodness-of-fit: pooled DD ISIs must not be distinguishable from Exponential(mean)."""
    isis = _pooled_isis(poisson_pool_spikes)
    p_value = stats.kstest(isis, "expon", args=(0.0, isis.mean())).pvalue
    assert p_value > 0.01, f"pooled DD ISIs reject the Exponential hypothesis (KS p={p_value:.4f})"


def test_pool_spike_counts_are_poisson(poisson_pool_spikes):
    """Counts in fixed 100 ms windows are Poisson-distributed: mean Fano factor (var/mean) ~ 1."""
    window = int(100.0 / DT_MS)
    n_steps = int(DURATION_MS / DT_MS)
    fanos = []
    for steps in poisson_pool_spikes:
        if len(steps) < 50:
            continue
        train = np.zeros(n_steps, dtype=np.int64)
        train[np.asarray(steps, dtype=int)] = 1
        counts = train[: (n_steps // window) * window].reshape(-1, window).sum(axis=1)
        fanos.append(counts.var() / counts.mean())
    assert np.mean(fanos) == pytest.approx(1.0, abs=0.1)


def test_pool_spike_counts_follow_poisson_distribution(poisson_pool_spikes):
    """Chi-square goodness-of-fit: the full distribution of spike counts in 100 ms windows must
    not be distinguishable from a Poisson(lambda) distribution (lambda = mean count). This is the
    count-domain counterpart to the KS-on-ISIs test and the direct definition of "Poisson".
    """
    window = int(100.0 / DT_MS)
    n_steps = int(DURATION_MS / DT_MS)
    n_windows = n_steps // window
    per_cell_counts = []
    for steps in poisson_pool_spikes:
        train = np.zeros(n_steps, dtype=np.int64)
        if steps:
            train[np.asarray(steps, dtype=int)] = 1
        per_cell_counts.append(train[: n_windows * window].reshape(-1, window).sum(axis=1))
    counts = np.concatenate(per_cell_counts)

    lam = counts.mean()
    kmax = int(counts.max())
    observed = np.bincount(counts, minlength=kmax + 1).astype(float)
    # Expected Poisson(lambda) frequencies. The final (k = kmax) bin carries the ENTIRE upper
    # tail P(X > kmax) via the survival function, so the expected probabilities sum to 1 — an
    # unconditional goodness-of-fit, not one conditioned away above the observed maximum.
    expected = stats.poisson.pmf(np.arange(kmax + 1), lam)
    expected[-1] += stats.poisson.sf(kmax, lam)
    expected = expected * counts.size

    # Merge bins from the bottom so every expected frequency >= 5 (chi-square validity condition).
    obs_binned, exp_binned, acc_o, acc_e = [], [], 0.0, 0.0
    for o, e in zip(observed, expected):
        acc_o += o
        acc_e += e
        if acc_e >= 5:
            obs_binned.append(acc_o)
            exp_binned.append(acc_e)
            acc_o = acc_e = 0.0
    if acc_e > 0:  # fold any remainder into the last bin
        obs_binned[-1] += acc_o
        exp_binned[-1] += acc_e
    obs_binned = np.asarray(obs_binned)
    exp_binned = np.asarray(exp_binned)
    exp_binned *= obs_binned.sum() / exp_binned.sum()  # float-safety only; totals already match

    # ddof=1 because lambda was estimated from the data.
    p_value = stats.chisquare(obs_binned, exp_binned, ddof=1).pvalue
    assert p_value > 0.01, f"spike counts reject the Poisson distribution (chi2 p={p_value:.4f})"


def test_pool_isis_are_serially_independent(poisson_pool_spikes):
    """A memoryless (renewal) Poisson process has independent ISIs: lag-1 autocorrelation ~ 0."""
    isis = _pooled_isis(poisson_pool_spikes)
    x = isis - isis.mean()
    lag1 = float((x[:-1] * x[1:]).sum() / (x * x).sum())
    assert abs(lag1) < 0.05


def test_gamma_pool_follows_gamma_distribution():
    """End-to-end Gamma validation through the public pool: process_type="gamma", shape=5 must
    produce *genuine* Gamma(5) ISIs — CV = 1/sqrt(5) and a KS test against Gamma(5) that does not
    reject — while being emphatically rejected as Exponential (i.e. not Poisson). This is exactly
    the old poisson_batch_size=5 behaviour, now only reachable explicitly, and it doubles as proof
    that the Poisson checks above have discriminating power.
    """
    myogen.set_random_seed(SEED)
    shape = 5.0
    pool = DescendingDrive__Pool(
        n=N_CELLS, timestep__ms=DT_MS * pq.ms, process_type="gamma", shape=shape
    )
    isis = _pooled_isis(_drive_pool(pool, RATE_HZ, DURATION_MS))
    mean = isis.mean()
    assert isis.std() / mean == pytest.approx(1.0 / np.sqrt(shape), abs=0.05)
    assert stats.kstest(isis, "gamma", args=(shape, 0.0, mean / shape)).pvalue > 0.01  # fits Gamma(5)
    assert stats.kstest(isis, "expon", args=(0.0, mean)).pvalue < 0.01  # not Exponential/Poisson


# --------------------------------------------------------------------------------------------
# Afferents: the renamed `shape` parameter must set the Gamma shape (CV = 1/sqrt(shape))
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [4, 9])
def test_afferent_shape_produces_gamma_isis(shape):
    """The renamed `shape` param on afferents must set the Gamma shape: an AffIa cell's ISIs
    follow Gamma(shape) (KS not rejected) with CV = 1/sqrt(shape)."""
    myogen.set_random_seed(SEED)
    cell = cells.AffIa(RT=0.0, shape=shape, timestep__ms=DT_MS * pq.ms)
    n_steps = int(DURATION_MS / DT_MS)
    steps = [i for i in range(n_steps) if cell.integrate(RATE_HZ)]
    isis = np.diff(np.asarray(steps, dtype=float) * DT_MS * 1e-3)
    mean = isis.mean()
    assert isis.std() / mean == pytest.approx(1.0 / np.sqrt(shape), abs=0.05)
    assert stats.kstest(isis, "gamma", args=(shape, 0.0, mean / shape)).pvalue > 0.01


# --------------------------------------------------------------------------------------------
# Kernel-level precision checks on the raw Cython generator
# --------------------------------------------------------------------------------------------


def _generator_isis(generator, rate_hz, duration_ms=200_000.0):
    n_steps = int(duration_ms / DT_MS)
    spikes = np.fromiter(
        (generator.compute(rate_hz) for _ in range(n_steps)), dtype=np.int8, count=n_steps
    )
    return np.diff(np.flatnonzero(spikes) * DT_MS * 1e-3)


@pytest.mark.parametrize("rate_hz", [10.0, 50.0])
def test_generator_is_true_poisson(rate_hz):
    isis = _generator_isis(_PoissonProcessGenerator__Cython(20260710, DT_MS), rate_hz)
    assert 1.0 / isis.mean() == pytest.approx(rate_hz, rel=0.02)
    assert isis.std() / isis.mean() == pytest.approx(1.0, abs=0.03)
    assert stats.kstest(isis, "expon", args=(0.0, isis.mean())).pvalue > 0.01


@pytest.mark.parametrize("shape", [2.0, 5.0, 10.0])
def test_gamma_generator_follows_gamma_distribution(shape):
    """Positive goodness-of-fit that the Gamma generator produces *genuine* Gamma(shape) ISIs:
    the mean rate is preserved, the shape is recovered (method of moments, shape = 1/CV^2), a KS
    test against Gamma(shape) does NOT reject, a KS test against a wrong shape DOES, and it is not
    Exponential. This is the Gamma counterpart of ``test_generator_is_true_poisson``.
    """
    from myogen.simulator.neuron._cython._gamma_process_generator import (
        _GammaProcessGenerator__Cython,
    )

    isis = _generator_isis(_GammaProcessGenerator__Cython(20260710, shape, DT_MS), RATE_HZ)
    mean = isis.mean()

    assert 1.0 / mean == pytest.approx(RATE_HZ, rel=0.02)  # mean rate preserved
    assert 1.0 / (isis.std() / mean) ** 2 == pytest.approx(shape, rel=0.1)  # shape recovered
    assert stats.kstest(isis, "gamma", args=(shape, 0.0, mean / shape)).pvalue > 0.01  # fits Gamma(shape)
    assert stats.kstest(isis, "gamma", args=(1.5 * shape, 0.0, mean / (1.5 * shape))).pvalue < 1e-3  # not a wrong shape
    assert stats.kstest(isis, "expon", args=(0.0, mean)).pvalue < 1e-3  # not Exponential
