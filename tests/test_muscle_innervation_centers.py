"""Innervation-centre placement must not depend on last-bit differences between platforms.

The fast-marching field has exact ties (two cells at the same distance, e.g. the mirror pair at step 2). A plain
``argmax`` then takes whichever cell a platform's floating point puts a last bit higher, and every later centre and
fibre assignment follows that choice.
"""
import numpy as np
import pytest
import quantities as pq

from myogen import set_random_seed
from myogen.simulator import Muscle, RecruitmentThresholds
import myogen.simulator.core.muscle.muscle as muscle_module


def test_farthest_cell_takes_the_lowest_index_among_ties():
    from myogen.simulator.core.muscle.muscle import _farthest_cell

    d = np.array([1.0, 5.0, 2.0, 5.0])
    assert _farthest_cell(d) == 1
    d[3] = 5.0 * (1 + 1e-14)  # the tie broken by a last-bit difference
    assert _farthest_cell(d) == 1
    d[3] = 5.0 * (1 + 1e-6)  # a real difference
    assert _farthest_cell(d) == 3


def _centres(noise_seed=None):
    set_random_seed(1)
    thr, _ = RecruitmentThresholds(N=20, recruitment_range__ratio=50, mode="fuglevand")
    m = Muscle(recruitment_thresholds=thr, radius__mm=3 * pq.mm)
    fmm = muscle_module._perform_fast_marching
    if noise_seed is not None:
        rng = np.random.default_rng(noise_seed)
        muscle_module._perform_fast_marching = lambda s, p: (lambda d: d * (1 + 1e-13 * rng.standard_normal(d.shape)))(
            fmm(s, p))
    try:
        m.distribute_innervation_centers()
    finally:
        muscle_module._perform_fast_marching = fmm
    return np.asarray(m._innervation_center_positions__mm.magnitude)


@pytest.mark.parametrize("noise_seed", range(4))
def test_centres_survive_last_bit_noise(noise_seed):
    np.testing.assert_array_equal(_centres(noise_seed), _centres())
