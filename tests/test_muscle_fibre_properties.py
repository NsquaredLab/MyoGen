"""Per-fibre conduction velocities and the fibre layout of the circular ``Muscle``."""

import numpy as np
import pytest
import quantities as pq

from myogen import set_random_seed
from myogen.simulator import Muscle, RecruitmentThresholds


def _muscle(radius__mm, density=None):
    thr, _ = RecruitmentThresholds(N=20, recruitment_range__ratio=50, mode="fuglevand")
    kw = {} if density is None else {"fiber_density__fibers_per_mm2": density * pq.mm**-2}
    return Muscle(recruitment_thresholds=thr, radius__mm=radius__mm * pq.mm, **kw)


def test_fibre_cv_follows_the_diameter_line():
    """CV = 0.95 m/s + 0.05 m/s per um of diameter (Nandedkar & Stalberg 1983): ~3.45 m/s at 50 um.

    With diameters in mm, the old slope of 5.5 (m/s)/mm gave ~1.1 m/s and the 2 m/s clip
    set every fibre to exactly 2.0 m/s.
    """
    set_random_seed(1)
    m = _muscle(4)
    m.generate_muscle_fiber_centers(verbose=False)
    m._generate_fiber_properties()
    d_um = np.asarray(m._muscle_fiber_diameters__mm.rescale(pq.mm)) * 1e3
    cv = np.asarray(m._muscle_fiber_conduction_velocities__mm_per_s.rescale(pq.m / pq.s))

    assert np.mean(np.isclose(cv, 2.0)) < 0.05
    assert np.median(cv) == pytest.approx(0.95 + 0.05 * np.median(d_um), abs=0.05)
    assert np.corrcoef(d_um, cv)[0, 1] > 0.9


def test_fibre_diameters_are_the_fdi_type_mixture():
    """57.4 % type I fibres of mean 50.0 um, type II of mean 67.0 um [polgar1973 p. 310 Table 2]
    [polgar1973 p. 312 Table 4]: mean 57.2 um, and no fibres piled up at a clip bound."""
    set_random_seed(1)
    m = _muscle(4)
    m.generate_muscle_fiber_centers(verbose=False)
    m._generate_fiber_properties()
    d_um = np.asarray(m._muscle_fiber_diameters__mm.rescale(pq.mm)) * 1e3
    assert np.mean(d_um) == pytest.approx(0.574 * 50.0 + 0.426 * 67.0, abs=0.6)
    assert np.mean(np.isclose(d_um, d_um.max())) < 1e-3
    assert np.mean(np.isclose(d_um, d_um.min())) < 1e-3
    # Two components: the lower half is mostly type I, the upper half mostly type II
    assert np.median(d_um[d_um < np.median(d_um)]) < 50.0 < np.median(d_um) < 67.0


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
