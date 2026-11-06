"""The ``execution=`` iEMG route reproduces the default route's physics.

``runtime/density.py`` is a numba port of ``bioelectric.get_current_density``
(the reference); it must follow every change to the reference.
"""
import numpy as np
import pytest
import quantities as pq

from myogen import set_random_seed
from myogen.simulator import (
    ExecutionPolicy,
    IntramuscularElectrodeArray,
    IntramuscularEMG,
    Muscle,
    RecruitmentThresholds,
)
from myogen.simulator.core.emg.intramuscular import bioelectric
from myogen.simulator.core.emg.runtime import density


@pytest.mark.parametrize("L1, L2", [(12.0, 10.0), (10.0, 12.0), (11.0, 11.0)])
@pytest.mark.parametrize("suppress", [True, False])
def test_kernel_matches_the_reference_current_density(L1, L2, suppress):
    t = np.arange(0.0, 20.0 + 1e-9, 0.1)[:, None]
    z = np.arange(-15.0, 15.0 + 1e-9, 0.05)[:, None]
    args = (t, z, 0.0, L1, L2, 4.0, 0.055)
    expected = bioelectric.get_current_density(*args, suppress_endplate_density=suppress)
    actual = density.current_density(*args, suppress_endplate_density=suppress)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12 * np.max(np.abs(expected)))


@pytest.mark.usefixtures("ample_session_memory")
def test_execution_route_muaps_match_the_default_route(monkeypatch):
    from myogen.simulator.core.emg.intramuscular import intramuscular_emg

    monkeypatch.setattr(intramuscular_emg, "HAS_CUPY", False)
    set_random_seed(5)
    thresholds, _ = RecruitmentThresholds(N=4, recruitment_range__ratio=30)
    muscle = Muscle(recruitment_thresholds=thresholds, radius__mm=2.0 * pq.mm,
                    radius_bone__mm=0.5 * pq.mm, fiber_density__fibers_per_mm2=50 * pq.mm**-2,
                    grid_resolution=32, autorun=True)
    electrode = IntramuscularElectrodeArray(num_electrodes=2, inter_electrode_distance__mm=0.5 * pq.mm,
                                            position__mm=(0.0 * pq.mm, 0.0 * pq.mm, 15.0 * pq.mm))
    muaps = []
    for execution in (None, ExecutionPolicy()):
        iemg = IntramuscularEMG(muscle_model=muscle, electrode_array=electrode, MUs_to_simulate=[0, 3],
                                sampling_frequency__Hz=10240.0 * pq.Hz)
        set_random_seed(11)
        muaps.append(iemg.simulate_muaps(n_jobs=1, verbose=False, execution=execution))
    for reference, optimized in zip(muaps[0].segments, muaps[1].segments):
        x = np.asarray(reference.analogsignals[0].magnitude)
        y = np.asarray(optimized.analogsignals[0].magnitude)
        assert x.shape == y.shape
        np.testing.assert_allclose(y, x, rtol=1e-7, atol=1e-9 * np.max(np.abs(x)))
