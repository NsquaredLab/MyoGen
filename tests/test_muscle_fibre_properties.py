"""Per-fibre conduction velocities and the fibre layout of the circular ``Muscle``."""

import numpy as np
import pytest
import quantities as pq

from myogen import set_random_seed
from myogen.simulator import Muscle, RecruitmentThresholds


def _muscle(radius__mm, density=None, **kw):
    thr, _ = RecruitmentThresholds(N=20, recruitment_range__ratio=50, mode="fuglevand")
    if density is not None:
        kw["fiber_density__fibers_per_mm2"] = density * pq.mm**-2
    return Muscle(recruitment_thresholds=thr, radius__mm=radius__mm * pq.mm, **kw)


def _typed(radius__mm=3, **kw):
    """A small muscle with fibres assigned to units and fibre properties generated (typed model by default)."""
    kw.setdefault("fiber_model", "typed")
    set_random_seed(1)
    m = _muscle(radius__mm, **kw)
    m.distribute_innervation_centers()
    m.generate_muscle_fiber_centers(verbose=False)
    m.assign_mfs2mns(verbose=False, n_jobs=1)
    m._generate_fiber_properties()
    return m


@pytest.fixture(scope="module")
def typed_muscle():
    return _typed()


def test_typed_fibre_properties_need_the_assignment():
    set_random_seed(1)
    m = _muscle(3, fiber_model="typed")
    m.generate_muscle_fiber_centers(verbose=False)
    with pytest.raises(ValueError, match="assign_mfs2mns"):
        m._generate_fiber_properties()


def test_each_unit_has_one_type_ordered_by_threshold(typed_muscle):
    m = typed_muscle
    a = m._assignment.astype(int)
    np.testing.assert_array_equal(m.muscle_fiber_types, m.motor_unit_fiber_types[a])
    t = m.motor_unit_fiber_types[np.argsort(m.recruitment_thresholds, kind="stable")]
    assert set(t) == {1, 2} and np.all(np.diff(t) >= 0)


def test_designed_unit_diameter_rises_with_threshold(typed_muscle):
    lad = typed_muscle.fiber_type_ladder
    assert np.all(np.diff(lad.unit_z[np.argsort(typed_muscle.recruitment_thresholds, kind="stable")]) >= 0)


def test_type_means_are_polgar(typed_muscle):
    """Type I 50.0 um, type II 67.0 um [polgar1973 p. 310 Table 2]."""
    d = np.asarray(typed_muscle.muscle_fiber_diameters__mm.rescale(pq.mm)) * 1e3
    t = typed_muscle.muscle_fiber_types
    assert d[t == 1].mean() == pytest.approx(50.0, abs=0.5)
    assert d[t == 2].mean() == pytest.approx(67.0, abs=0.5)


@pytest.mark.parametrize("type_means", [(50.0, 67.0) * pq.um, (0.050, 0.067) * pq.mm])
def test_type_mean_diameters_accept_any_length_unit(typed_muscle, type_means):
    """Type means given as a Quantity are rescaled to um; plain numbers are read as um."""
    m = _typed(type_mean_fiber_diameters__um=type_means)
    assert m.fiber_type_ladder.target_mean__um == pytest.approx(
        typed_muscle.fiber_type_ladder.target_mean__um, rel=1e-12
    )


def test_reassignment_clears_fibre_properties():
    """Fibre types and diameters belong to one assignment; a new assignment invalidates them."""
    m = _typed()
    m.assign_mfs2mns(verbose=False, n_jobs=1)
    with pytest.raises(ValueError, match="_generate_fiber_properties"):
        m.muscle_fiber_types
    with pytest.raises(ValueError, match="_generate_fiber_properties"):
        m.muscle_fiber_diameters__mm
    with pytest.raises(ValueError, match="_generate_fiber_properties"):
        m.muscle_fiber_conduction_velocities__mm_per_s


def test_fibre_properties_are_reproducible_under_a_seed(typed_muscle):
    m = _typed()
    for name in ("muscle_fiber_diameters__mm", "muscle_fiber_conduction_velocities__mm_per_s"):
        np.testing.assert_array_equal(
            np.asarray(getattr(m, name).magnitude), np.asarray(getattr(typed_muscle, name).magnitude)
        )
    np.testing.assert_array_equal(m.muscle_fiber_types, typed_muscle.muscle_fiber_types)


@pytest.mark.parametrize("kw, mean_cv, sd_cv", [
    ({}, 4.32, 0.71),
    ({"mean_fiber_conduction_velocity__m_per_s": 4.2 * pq.m / pq.s,
      "sd_fiber_conduction_velocity__m_per_s": 0.4 * pq.m / pq.s}, 4.2, 0.4),
])
def test_fibre_cv_has_the_requested_distribution_ordered_by_diameter(kw, mean_cv, sd_cv):
    """Typed model without within-unit spread (``within_unit_fiber_diameter_log_sd=None``): fibre CVs are 4.32 +/- 0.71 m/s over fibres, the surface (HD-sEMG) CV of all
    tibialis anterior units in Del Vecchio et al. (2018), ordered by fibre diameter through the
    Nandedkar & Stalberg line; the 2-6 m/s sanity clip trims the top. The muscle mean CV used by
    surface EMG is separate. Unit-mean CV rises with recruitment threshold.

    With diameters in mm, the old slope of 5.5 (m/s)/mm gave ~1.1 m/s and the 2 m/s clip
    set every fibre to exactly 2.0 m/s.
    """
    from scipy.stats import spearmanr

    m = _typed(mean_conduction_velocity__m_per_s=5.0 * pq.m / pq.s, within_unit_fiber_diameter_log_sd=None, **kw)
    d_um = np.asarray(m.muscle_fiber_diameters__mm.rescale(pq.mm)) * 1e3
    cv = np.asarray(m.muscle_fiber_conduction_velocities__mm_per_s.rescale(pq.m / pq.s))

    assert np.mean(np.isclose(cv, 2.0) | np.isclose(cv, 6.0)) < 0.03
    assert np.mean(cv) == pytest.approx(mean_cv, abs=0.03)
    assert np.std(cv) == pytest.approx(sd_cv, abs=0.03)
    assert np.corrcoef(d_um, cv)[0, 1] > 0.9
    assert float(m.mean_conduction_velocity__m_s.rescale(pq.m / pq.s)) == 5.0
    a = m._assignment.astype(int)
    unit_cv = np.bincount(a, weights=cv) / np.maximum(np.bincount(a), 1)
    assert spearmanr(m.recruitment_thresholds, unit_cv)[0] > 0


def _cv_and_units(m):
    cv = np.asarray(m.muscle_fiber_conduction_velocities__mm_per_s.rescale(pq.m / pq.s))
    d_um = np.asarray(m.muscle_fiber_diameters__mm.rescale(pq.mm)) * 1e3
    return cv, d_um, m._assignment.astype(int)


def test_default_within_unit_spread_is_martin(typed_muscle):
    """Within-unit diameter log-SD 0.08 (type I) / 0.13 (type II) [martin1988 p. C46 Table 1]."""
    lad = typed_muscle.fiber_type_ladder
    np.testing.assert_allclose(lad.unit_sigma_within, np.where(lad.unit_types == 1, 0.08, 0.13))


@pytest.mark.parametrize("mean_cv, sd_units", [(4.0, 0.5), (4.2, 0.4)])
def test_cv_is_over_unit_means(mean_cv, sd_units):
    """Without per-unit CVs the mean and SD are those of motor-unit CVs [delvecchio2018 p. 2]
    [andreassen1987 p. 567], so they hold over unit means, each unit once. (At the 4.32 +/- 0.71
    default the 6 m/s bound trims the fastest units, so values are chosen clear of it.)"""
    kw = {"mean_fiber_conduction_velocity__m_per_s": mean_cv * pq.m / pq.s,
          "sd_fiber_conduction_velocity__m_per_s": sd_units * pq.m / pq.s}
    m = _typed(**kw)
    cv, _, a = _cv_and_units(m)
    counts = np.bincount(a)
    unit_cv = (np.bincount(a, weights=cv) / np.maximum(counts, 1))[counts > 0]
    assert np.mean(unit_cv) == pytest.approx(mean_cv, abs=0.03)
    assert np.std(unit_cv) == pytest.approx(sd_units, abs=0.03)


def test_given_unit_cvs_set_each_units_mean():
    """Per-unit CVs (e.g. from ``cv_from_recruitment_force``) set each unit's mean CV; within a unit CV
    follows diameter on the Nandedkar & Stalberg slope, 0.05 m/s per um [nandedkar1983 p. 163]."""
    target = np.linspace(3.5, 5.0, 20)
    m = _typed(motor_unit_conduction_velocities__m_per_s=target * pq.m / pq.s)
    cv, d_um, a = _cv_and_units(m)
    counts = np.bincount(a, minlength=20)
    unit_cv = np.bincount(a, weights=cv, minlength=20) / np.maximum(counts, 1)
    clipped = np.bincount(a, weights=(np.isclose(cv, 2.0) | np.isclose(cv, 6.0)).astype(float), minlength=20)
    has = (counts > 0) & (clipped == 0)  # the 2-6 m/s bound moves a clipped unit's mean
    assert has.sum() >= 15
    np.testing.assert_allclose(unit_cv[has], target[has], atol=1e-9)
    for u in np.flatnonzero(has & (counts >= 3)):
        sel = a == u
        assert np.polyfit(d_um[sel], cv[sel], 1)[0] == pytest.approx(0.05, rel=1e-9)


@pytest.mark.parametrize("value, message", [
    (np.full(19, 4.0), "motor_unit_conduction_velocities__m_per_s"),
    (np.r_[np.full(19, 4.0), -1.0], "motor_unit_conduction_velocities__m_per_s"),
])
def test_bad_unit_cvs_raise(value, message):
    with pytest.raises(ValueError, match=message):
        _typed(motor_unit_conduction_velocities__m_per_s=value * pq.m / pq.s)


def test_cv_follows_diameter_within_each_unit(typed_muscle):
    """No extra CV scatter: inside a unit every unclipped fibre sits on the same CV-diameter line."""
    cv, d_um, a = _cv_and_units(typed_muscle)
    free = (cv > 2.0 + 1e-9) & (cv < 6.0 - 1e-9)
    slopes = []
    for u in np.unique(a):
        sel = free & (a == u)
        if sel.sum() >= 3:
            assert np.corrcoef(d_um[sel], cv[sel])[0, 1] > 0.9999
            slopes.append(np.polyfit(d_um[sel], cv[sel], 1)[0])
    assert np.ptp(slopes) < 1e-9


# Fibres of a small muscle (radius 3 mm, 20 units, seed 1) at evenly spaced fibre indices, stored as values
# rather than a hash of float bytes so that last-bit differences between platforms do not fail the test.
REFERENCE_INDICES = [0, 1413, 2827, 4240, 5654, 7067, 8481, 9895]
REFERENCE_FIBRES = {
    # Default model: independent log-normal diameters, Nandedkar & Stalberg line plus 0.2 m/s, as in 0.13.0
    # with the mm/um unit fix.
    "independent": dict(
        kw={},
        d_um=[56.34238382660647, 51.58253367532592, 46.3707025489695, 45.46069783475108,
              33.19107866716076, 44.573230829782865, 80.00000000000001, 42.65000327592993],
        cv=[3.734535319916328, 3.804904588371462, 3.422200610932065, 3.550286142413758,
            2.595026007526017, 3.512092815395755, 4.978012609466153, 2.936726492241513],
        sums=[509371.7252799411, 34845.050353542174, 1891535.0309199751],
    ),
    # Typed model without within-unit spread.
    "typed_no_spread": dict(
        kw={"fiber_model": "typed", "within_unit_fiber_diameter_log_sd": None},
        d_um=[66.77219954698421, 58.743157829196456, 63.52655760550686, 61.226787523119604,
              37.86102310760331, 62.22181655685925, 29.286352645372745, 43.177565794727656],
        cv=[4.56143978382481, 4.415072227167105, 4.727232242590925, 4.487681901091421,
            3.4003698534009, 4.828684861117164, 2.871166973145793, 3.372563964823911],
        sums=[558345.4393773584, 42691.145676895205, 2497788.7535994113],
    ),
}


@pytest.mark.parametrize("model", sorted(REFERENCE_FIBRES))
def test_fibre_models_are_reproducible(model):
    ref = REFERENCE_FIBRES[model]
    set_random_seed(1)
    m = _muscle(3, **ref["kw"])
    m.distribute_innervation_centers()
    m.generate_muscle_fiber_centers(verbose=False)
    m.assign_mfs2mns(verbose=False, n_jobs=1)
    m._generate_fiber_properties()
    d_um = np.asarray(m.muscle_fiber_diameters__mm.rescale(pq.um).magnitude)
    cv = np.asarray(m.muscle_fiber_conduction_velocities__mm_per_s.rescale(pq.m / pq.s).magnitude)
    assert len(d_um) == 9896
    np.testing.assert_allclose(d_um[REFERENCE_INDICES], ref["d_um"], rtol=1e-9)
    np.testing.assert_allclose(cv[REFERENCE_INDICES], ref["cv"], rtol=1e-9)
    np.testing.assert_allclose([d_um.sum(), cv.sum(), (d_um * cv).sum()], ref["sums"], rtol=1e-9)


def test_default_fibre_model_does_not_warn():
    import warnings

    set_random_seed(1)
    m = _muscle(3)
    m.distribute_innervation_centers()
    m.generate_muscle_fiber_centers(verbose=False)
    m.assign_mfs2mns(verbose=False, n_jobs=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m._generate_fiber_properties()


@pytest.mark.parametrize("kw", [
    {"mean_fiber_conduction_velocity__m_per_s": 4.32 * pq.m / pq.s},
    {"sd_fiber_conduction_velocity__m_per_s": 0.71 * pq.m / pq.s},
    {"motor_unit_conduction_velocities__m_per_s": np.full(20, 4.0)},
    {"type_i_fiber_fraction": 0.574},
    {"type_mean_fiber_diameters__um": (50.0, 67.0)},
    {"type_fiber_diameter_log_sd": 0.2},
    {"within_unit_fiber_diameter_log_sd": (0.08, 0.13)},
])
def test_typed_arguments_need_the_typed_model(kw):
    with pytest.raises(ValueError, match="fiber_model='typed'"):
        _muscle(3, **kw)


def test_unknown_fibre_model_raises():
    with pytest.raises(ValueError, match="fiber_model"):
        _muscle(3, fiber_model="mixed")


def test_fibre_types_need_the_typed_model():
    set_random_seed(1)
    m = _muscle(3)
    m.distribute_innervation_centers()
    m.generate_muscle_fiber_centers(verbose=False)
    m.assign_mfs2mns(verbose=False, n_jobs=1)
    m._generate_fiber_properties()
    for name in ("motor_unit_fiber_types", "muscle_fiber_types", "fiber_type_ladder"):
        with pytest.raises(ValueError, match="fiber_model='typed'"):
            getattr(m, name)


def test_surface_emg_fibre_diameter_default_is_fdi_rms():
    """sqrt(0.574 * 50.0**2 + 0.426 * 67.0**2) = 57.9 um [polgar1973 p. 310 Table 2] [polgar1973 p. 312 Table 4]."""
    import inspect

    from myogen.simulator import SurfaceEMG

    d = inspect.signature(SurfaceEMG.__init__).parameters["fiber_diameter__um"].default
    assert d == pytest.approx(np.sqrt(0.574 * 50.0**2 + 0.426 * 67.0**2), abs=0.05)


def _layout(radius__mm, density):
    set_random_seed(1)
    m = _muscle(radius__mm, density)
    m.generate_muscle_fiber_centers(verbose=False)
    return m, np.asarray(m._muscle_fiber_centers__mm.rescale(pq.mm))


def test_large_fibre_counts_are_not_capped():
    m, xy = _layout(20, 400)  # 502 655 fibres requested, more than the 399 499-point table
    expected = int(np.rint(np.pi * 20**2 * 400))
    assert len(xy) == m._number_of_muscle_fibers == expected


def test_large_fibre_layouts_stay_circular():
    """Fibres in the outer ring must be spread around the whole circle, not only the diagonals."""
    _, xy = _layout(20, 300)  # 376 991 fibres: beyond the table's 313 k inscribed-disc points
    r = np.hypot(*xy.T)
    assert r.max() <= 20 + 1e-9
    ring = xy[r > 0.97 * 20]
    counts, _ = np.histogram(np.arctan2(ring[:, 1], ring[:, 0]), bins=16, range=(-np.pi, np.pi))
    assert counts.min() > 0.5 * counts.mean()


def test_many_fibres_at_the_upper_cv_bound_warn():
    """More than 5 % of fibres clipped to 6 m/s warns and points to per-unit CVs."""
    with pytest.warns(UserWarning, match="motor_unit_conduction_velocities__m_per_s"):
        _typed(mean_fiber_conduction_velocity__m_per_s=5.5 * pq.m / pq.s)


def test_few_fibres_at_the_upper_cv_bound_do_not_warn():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _typed(mean_fiber_conduction_velocity__m_per_s=4.0 * pq.m / pq.s,
               sd_fiber_conduction_velocity__m_per_s=0.4 * pq.m / pq.s)


@pytest.mark.parametrize("given", [
    np.linspace(3.5, 5.0, 20),                          # plain array, as cv_from_recruitment_force returns: m/s
    np.linspace(3500.0, 5000.0, 20) * pq.mm / pq.s,     # another velocity unit
])
def test_unit_cvs_accept_plain_arrays_and_other_velocity_units(given):
    reference = _cv_and_units(_typed(motor_unit_conduction_velocities__m_per_s=np.linspace(3.5, 5.0, 20) * pq.m / pq.s))[0]
    np.testing.assert_allclose(_cv_and_units(_typed(motor_unit_conduction_velocities__m_per_s=given))[0], reference)


def test_unit_cvs_with_a_non_velocity_unit_raise():
    with pytest.raises(ValueError):
        _typed(motor_unit_conduction_velocities__m_per_s=np.full(20, 4.0) * pq.mm)
