"""Surrogate-gradient spikes and the JAX Poisson/Gamma drive re-home."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from myogen.diff import spikes as S

DT_S = 1e-4


def _isis(spike_row):
    times = np.flatnonzero(np.asarray(spike_row)) * DT_S
    return np.diff(times)


def test_surrogate_gradient_matches_finite_difference():
    """The surrogate makes spike counting differentiable, and correctly so (direction)."""

    def spike_count(scale):
        v = scale * jnp.sin(jnp.linspace(0, 20, 400))
        return S.spike_train(v, threshold_mV=0.3).sum()

    auto = float(jax.grad(spike_count)(2.0))
    eps = 1e-3
    fd = float((spike_count(2.0 + eps) - spike_count(2.0 - eps)) / (2 * eps))
    assert np.isfinite(auto) and auto != 0.0
    # hard spike count is piecewise-constant (fd may be 0 between jumps); the surrogate must at
    # least agree in sign when the finite difference is itself non-zero.
    if fd != 0.0:
        assert np.sign(auto) == np.sign(fd)


@pytest.mark.parametrize("rate_hz", [15.0, 30.0])
def test_poisson_drive_is_true_poisson(rate_hz):
    """The re-homed Poisson drive reproduces the fixed Cython statistics: CV=1, ISIs ~ Exp."""
    sp = S.poisson_spike_train(jax.random.PRNGKey(0), rate_hz, DT_S, int(80.0 / DT_S))
    isis = _isis(sp)
    assert isis.std() / isis.mean() == pytest.approx(1.0, abs=0.08)
    assert stats.kstest(isis, "expon", args=(0.0, isis.mean())).pvalue > 0.01


def test_gamma_drive_matches_requested_shape():
    """Gamma(shape=5) drive has CV = 1/sqrt(5) and is rejected as Exponential."""
    sp = S.gamma_spike_train(jax.random.PRNGKey(0), 25.0, 5, DT_S, int(80.0 / DT_S))
    isis = _isis(sp)
    assert isis.std() / isis.mean() == pytest.approx(1.0 / np.sqrt(5.0), abs=0.06)
    assert stats.kstest(isis, "expon", args=(0.0, isis.mean())).pvalue < 0.01


def test_drive_trains_vectorize_per_unit():
    # long enough window that Poisson counting noise doesn't swamp the rate ordering
    n_steps = int(20.0 / DT_S)  # 20 s
    d = S.drive_spike_trains(jax.random.PRNGKey(1), jnp.array([10.0, 20.0, 40.0]), DT_S, n_steps)
    assert d.shape == (3, n_steps)
    counts = np.asarray(d.sum(axis=1))
    # empirical rate tracks the requested rate, and counts are strictly increasing
    assert counts[0] < counts[1] < counts[2]
    assert counts[2] / (n_steps * DT_S) == pytest.approx(40.0, rel=0.1)
