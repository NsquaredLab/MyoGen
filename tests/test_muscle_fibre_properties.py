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
    """A small muscle with fibres assigned to units and fibre properties generated."""
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


def test_fibre_properties_need_the_assignment():
    set_random_seed(1)
    m = _muscle(3)
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
    """Fibre CVs (intramuscular EMG) default to 4.32 +/- 0.71 m/s, the surface (HD-sEMG) CV of all
    tibialis anterior units in Del Vecchio et al. (2018), ordered by fibre diameter through the
    Nandedkar & Stalberg line; the 2-6 m/s sanity clip trims the top. The muscle mean CV used by
    surface EMG is separate. Unit-mean CV rises with recruitment threshold.

    With diameters in mm, the old slope of 5.5 (m/s)/mm gave ~1.1 m/s and the 2 m/s clip
    set every fibre to exactly 2.0 m/s.
    """
    from scipy.stats import spearmanr

    m = _typed(mean_conduction_velocity__m_per_s=5.0 * pq.m / pq.s, **kw)
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
