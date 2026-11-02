"""Fibre types by motor unit: one diameter ladder across the pool (spec 2026-11-02-fdi-fibre-typing-design)."""

import numpy as np
import pytest

from myogen.simulator.core.muscle.fiber_types import draw_fiber_diameters__um, fiber_type_ladder

n = np.round(np.geomspace(40, 1600, 120)).astype(int)
thresholds = np.geomspace(1, 30, 120)
assignment = np.repeat(np.arange(120), n)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_types_follow_threshold_and_hit_the_fraction():
    lad = fiber_type_ladder(n, thresholds)
    order = np.argsort(thresholds, kind="stable")
    t = lad.unit_types[order]
    assert set(t) == {1, 2} and np.all(np.diff(t) >= 0)  # no type 2 below a type 1
    assert abs(lad.type_i_fiber_fraction - 0.574) <= n.max() / n.sum()


def test_ladder_rises_across_the_type_boundary():
    lad = fiber_type_ladder(n, thresholds)
    assert np.all(np.diff(lad.unit_z[np.argsort(thresholds, kind="stable")]) >= 0)


def test_diameters_keep_polgar_means_and_spread(rng):
    lad = fiber_type_ladder(n, thresholds)
    d = draw_fiber_diameters__um(lad, assignment, rng)
    t = lad.unit_types[assignment]
    assert d[t == 1].mean() == pytest.approx(50.0, abs=0.5)
    assert d[t == 2].mean() == pytest.approx(67.0, abs=0.5)
    assert d.mean() == pytest.approx(0.574 * 50.0 + 0.426 * 67.0, abs=0.05)
    assert np.log(d).std() == pytest.approx(np.sqrt(0.04 + 0.574 * 0.426 * np.log(67 / 50) ** 2), abs=0.01)


def test_tails_are_cut_at_three_sd(rng):
    lad = fiber_type_ladder(n, thresholds)
    d = draw_fiber_diameters__um(lad, assignment, rng)
    S = np.hypot(lad.sigma_between, lad.sigma_within)
    assert np.log(d).max() - np.log(d).min() <= 6 * S + 1e-9


def test_empty_unit_gets_a_type_and_position():
    m = n.copy()
    m[5] = 0
    lad = fiber_type_ladder(m, thresholds)
    assert lad.unit_types[5] in (1, 2) and np.isfinite(lad.unit_z).all()


def test_tied_thresholds_order_by_index():
    th = thresholds.copy()
    th[10] = th[11]
    a, b = fiber_type_ladder(n, th), fiber_type_ladder(n, th)
    np.testing.assert_array_equal(a.unit_z, b.unit_z)
    assert a.unit_z[10] <= a.unit_z[11]


@pytest.mark.parametrize("counts, ths, f", [(np.array([100]), np.array([1.0]), 0.574), (n, thresholds, 1.0)])
def test_degenerate_typing_raises(counts, ths, f):
    with pytest.raises(ValueError, match="type II"):
        fiber_type_ladder(counts, ths, type_i_fiber_fraction=f)


def test_too_small_type_spread_raises():
    with pytest.raises(ValueError, match="sigma_within"):
        fiber_type_ladder(n, thresholds, type_log_sd=0.01)
