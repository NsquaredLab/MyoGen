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
    f = lad.type_i_fiber_fraction
    assert d.mean() == pytest.approx(f * 50.0 + (1 - f) * 67.0, abs=0.05)
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


@pytest.mark.parametrize("counts, ths, f", [(np.array([100]), np.array([1.0]), 0.574), (n, thresholds, 0.99)])
def test_degenerate_typing_raises(counts, ths, f):
    with pytest.raises(ValueError, match="type II"):
        fiber_type_ladder(counts, ths, type_i_fiber_fraction=f)


def test_too_small_type_spread_raises():
    with pytest.raises(ValueError, match="sigma_within"):
        fiber_type_ladder(n, thresholds, type_log_sd=0.01)


def test_type_means_hold_when_the_realised_fraction_is_coarse(rng):
    """With few units the type I fraction lands a unit's worth off 57.4 %; the type means must
    still be 50.0 / 67.0 um, so the pool mean follows the realised fraction."""
    coarse = np.round(np.geomspace(40, 1600, 12)).astype(int)
    lad = fiber_type_ladder(coarse, np.geomspace(1, 30, 12))
    assert abs(lad.type_i_fiber_fraction - 0.574) > 0.01
    d = draw_fiber_diameters__um(lad, np.repeat(np.arange(12), coarse), rng)
    t = lad.unit_types[np.repeat(np.arange(12), coarse)]
    assert d[t == 1].mean() == pytest.approx(50.0, abs=0.5)
    assert d[t == 2].mean() == pytest.approx(67.0, abs=0.5)


def test_pool_spread_follows_the_realised_fraction(rng):
    """The pool log-variance is that of Polgar's two-type mixture at the realised type fraction, as
    the pool mean is. The drawn check is coarse (nominal vs realised differ by ~0.001 in log-SD here,
    below sampling noise and the 3 SD truncation); the designed variance is checked exactly."""
    coarse = np.round(np.geomspace(40, 1600, 12)).astype(int)
    lad = fiber_type_ladder(coarse, np.geomspace(1, 30, 12))
    r = lad.type_i_fiber_fraction
    mixture_var = 0.04 + r * (1 - r) * np.log(67 / 50) ** 2
    d = draw_fiber_diameters__um(lad, np.repeat(np.arange(12), coarse), rng)
    assert np.log(d).std() == pytest.approx(np.sqrt(mixture_var), abs=0.01)
    designed_var = lad.sigma_between**2 * np.average(lad.unit_z**2, weights=coarse) + lad.sigma_within**2
    assert designed_var == pytest.approx(mixture_var, abs=1e-9)


@pytest.mark.parametrize("kwargs, message", [
    ({"type_mean_diameters__um": (67.0, 50.0)}, "type_mean_diameters__um"),
    ({"type_i_fiber_fraction": 0.0}, "type_i_fiber_fraction"),
    ({"type_i_fiber_fraction": -0.1}, "type_i_fiber_fraction"),
    ({"type_log_sd": -0.2}, "type_log_sd"),
    ({"recruitment_thresholds": thresholds[:-1]}, "same length"),
])
def test_bad_inputs_raise_clear_errors(kwargs, message):
    args = {"fibers_per_unit": n, "recruitment_thresholds": thresholds, **kwargs}
    with pytest.raises(ValueError, match=message):
        fiber_type_ladder(**args)


def test_a_subset_draw_keeps_each_units_own_mean(rng):
    """The diameter scale is a property of the ladder, not of the fibres passed in: drawing only
    the top unit's fibres gives that unit's mean, not the pool mean."""
    lad = fiber_type_ladder(n, thresholds)
    full = draw_fiber_diameters__um(lad, assignment, rng)
    top = int(np.argmax(thresholds))
    subset = draw_fiber_diameters__um(lad, np.full(5000, top), rng)
    assert subset.mean() == pytest.approx(full[assignment == top].mean(), rel=0.02)
    assert subset.mean() > 1.2 * lad.target_mean__um


def test_within_unit_spread_is_set_per_type(rng):
    """Within-unit diameter log-SD by type: 0.08 (type I) and 0.13 (type II), half the cat TA
    within-unit fibre-area CoV of slow and fast units [martin1988 p. C46 Table 1]."""
    lad = fiber_type_ladder(n, thresholds, within_unit_log_sd=(0.08, 0.13))
    np.testing.assert_allclose(lad.unit_sigma_within, np.where(lad.unit_types == 1, 0.08, 0.13))
    d = draw_fiber_diameters__um(lad, assignment, rng)
    for t, s in ((1, 0.08), (2, 0.13)):
        sds = [np.log(d[assignment == u]).std() for u in range(len(n)) if lad.unit_types[u] == t and n[u] >= 100]
        assert len(sds) >= 3
        assert np.median(sds) == pytest.approx(s, rel=0.1)


def test_per_type_spread_keeps_polgar_means(rng):
    lad = fiber_type_ladder(n, thresholds, within_unit_log_sd=(0.08, 0.13))
    d = draw_fiber_diameters__um(lad, assignment, rng)
    t = lad.unit_types[assignment]
    assert d[t == 1].mean() == pytest.approx(50.0, abs=0.5)
    assert d[t == 2].mean() == pytest.approx(67.0, abs=0.5)
    f = lad.type_i_fiber_fraction
    assert d.mean() == pytest.approx(f * 50.0 + (1 - f) * 67.0, abs=0.1)


def test_residual_ladder_has_one_within_unit_spread():
    lad = fiber_type_ladder(n, thresholds)
    assert np.all(lad.unit_sigma_within == lad.sigma_within)


@pytest.mark.parametrize("value", [(0.0, 0.1), (0.1, -0.1), (0.1,)])
def test_bad_within_unit_spread_raises(value):
    with pytest.raises(ValueError, match="within_unit_log_sd"):
        fiber_type_ladder(n, thresholds, within_unit_log_sd=value)
