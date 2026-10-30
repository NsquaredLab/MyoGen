"""Surface-EMG consistency with the Farina et al. (2004) model.

Each test checks the analytical pipeline against an independent reference:
linearity of the detection system, the conduction velocity that was set, an
independent time-domain Rosenfalck path, or the documented source geometry.
"""
import numpy as np
import pytest
import quantities as pq
from scipy.signal import correlate, resample

from myogen import set_random_seed
from myogen.simulator import (
    ExecutionPolicy,
    Muscle,
    RecruitmentThresholds,
    SurfaceElectrodeArray,
    SurfaceEMG,
)
from myogen.simulator.core.emg import fiber_simulation
from myogen.simulator.core.emg.surface import simulate_fiber
from myogen.utils.neo import signal_to_grid

GEOMETRY = dict(r=8.5, r_bone=0.0, th_fat=0.3, th_skin=1.29, sig_muscle_rho=0.09,
                sig_muscle_z=0.4, sig_fat=4.07e-2, sig_skin=4.88e-4)
V = 4.0          # mm/ms
N = 512
KERNEL = 79.25   # mm
DT = KERNEL / (V * (N - 1))  # ms per internal sample


def _array(rows, cols, ied, z0, mode="monopolar"):
    return SurfaceElectrodeArray(rows, cols, ied * pq.mm, 0.5 * pq.mm,
                                 center_point__mm_deg=(z0 * pq.mm, 0 * pq.deg),
                                 bending_radius__mm=GEOMETRY["r"] * pq.mm,
                                 differentiation_mode=mode)


def _fibre(array, *, R=5.5, theta=0.15, L1=16.0, L2=16.0, zi=0.0, kernel=KERNEL, n=N):
    phi, _, _ = simulate_fiber._simulate_fiber_v2_python(
        Fs=10.0, v=V, N=n, M=32, R=R, L1=L1, L2=L2, zi=zi, electrode_array=array,
        fiber_length__mm=kernel, use_gpu=False, theta_offset=-theta, **GEOMETRY)
    return phi


def _upsampled_lag_ms(a, b, dt_ms, up=20):
    a2, b2 = resample(a, len(a) * up), resample(b, len(b) * up)
    return (np.argmax(correlate(b2, a2, "full")) - (len(a2) - 1)) * dt_ms / up


@pytest.fixture(params=["numba", "numpy"])
def b_kz_backend(request, monkeypatch):
    if request.param == "numpy":
        monkeypatch.setattr(simulate_fiber, "_numba_B_kz_func", None)
        monkeypatch.setattr(simulate_fiber, "_numba_phi_func", None)
    return request.param


def test_longitudinal_bipolar_equals_difference_of_monopolar_pair(b_kz_backend):
    bipolar = _fibre(_array(1, 1, 10, 8, "bipolar_longitudinal"))[0, 0]
    pair = _fibre(_array(2, 1, 10, 8))
    expected = pair[1, 0] - pair[0, 0]
    np.testing.assert_allclose(bipolar, expected, atol=1e-9 * np.ptp(expected))


def test_transversal_bipolar_equals_difference_of_monopolar_pair(b_kz_backend):
    bipolar = _fibre(_array(1, 1, 3, 8, "bipolar_transversal"))[0, 0]
    pair = _fibre(_array(1, 2, 3, 8))
    expected = pair[0, 1] - pair[0, 0]
    np.testing.assert_allclose(bipolar, expected, atol=1e-9 * np.ptp(expected))


def test_separately_simulated_channels_keep_their_propagation_delay():
    # One call per electrode: a per-call re-centring would erase the delay.
    near = _fibre(_array(1, 1, 5, 6), R=6.5, theta=0.0)[0, 0]
    far = _fibre(_array(1, 1, 5, 11), R=6.5, theta=0.0)[0, 0]
    assert _upsampled_lag_ms(near, far, DT) == pytest.approx(5.0 / V, rel=0.03)


def test_firing_instant_is_the_window_centre():
    # Monopolar negative peaks along a long fibre fall on t = t0 + z/v; the
    # intercept t0 is the IAP depolarisation peak (z = 3 mm for Rosenfalck)
    # leaving the end-plate, measured from the window centre.
    n, kernel = 1024, 300.0
    dt = kernel / (V * (n - 1))
    z_values = np.array([10.0, 20.0, 30.0, 40.0])
    peaks = [np.argmin(_fibre(_array(1, 1, 5, z), R=6.7, theta=0.0, L1=60.0, L2=60.0,
                              kernel=kernel, n=n)[0, 0]) for z in z_values]
    slope, intercept = np.polyfit(z_values, peaks, 1)
    assert slope * dt == pytest.approx(1.0 / V, rel=0.01)
    assert (intercept - n // 2) * dt * V == pytest.approx(3.0, abs=0.5)


def test_source_matches_independent_time_domain_rosenfalck():
    # simulate_fiber_hybrid builds the published Rosenfalck IAP (lambda = 1 mm)
    # in the time domain; only the amplitude constant differs.
    array = _array(1, 1, 5, 8)
    kwargs = dict(Fs=10.0, v=V, N=N, M=32, R=6.5, L1=15.85, L2=15.85, zi=0.0,
                  electrode_array=array, fiber_length__mm=KERNEL, use_gpu=False, **GEOMETRY)
    frequency = simulate_fiber._simulate_fiber_v2_python(**kwargs)[0][0, 0]
    reference = fiber_simulation.simulate_fiber_hybrid(**kwargs, D1=96.0)[0][0, 0]
    similarity = np.max(np.abs(correlate(frequency, reference, "full")))
    similarity /= np.linalg.norm(frequency) * np.linalg.norm(reference)
    assert similarity > 0.99


def test_l1_is_the_positive_z_semi_length():
    # Fibre spans [zi - L2, zi + L1]: pairs inside it see a travelling wave,
    # pairs beyond its end see only a standing end-of-fibre component.
    kernel = 100.0
    dt = kernel / (V * (N - 1))
    inside = _fibre(_array(2, 1, 5, 12.5), R=6.4, theta=0.0, L1=30.0, L2=8.0, kernel=kernel)
    beyond = _fibre(_array(2, 1, 5, -12.5), R=6.4, theta=0.0, L1=30.0, L2=8.0, kernel=kernel)
    assert _upsampled_lag_ms(inside[0, 0], inside[1, 0], dt) == pytest.approx(5.0 / V, rel=0.05)
    assert abs(_upsampled_lag_ms(beyond[1, 0], beyond[0, 0], dt)) < 0.5 * 5.0 / V


def _muscle(**kwargs):
    set_random_seed(42)
    thresholds, _ = RecruitmentThresholds(N=20, recruitment_range__ratio=50, mode="konstantin")
    muscle = Muscle(thresholds, fiber_density__fibers_per_mm2=10 * pq.mm**-2, **kwargs)
    muscle.distribute_innervation_centers()
    muscle.generate_muscle_fiber_centers(verbose=False)
    muscle.assign_mfs2mns(verbose=False, n_jobs=1)
    return muscle


@pytest.fixture(scope="module")
def muscle():
    return _muscle()


@pytest.fixture(scope="module")
def long_muscle():
    return _muscle(mean_fiber_length__mm=120 * pq.mm)


def _simulate(semg, route):
    if route == "default":
        return semg.simulate_muaps(n_jobs=1, verbose=False, use_gpu=False)
    return semg.simulate_muaps(n_jobs=1, verbose=False, execution=ExecutionPolicy())


@pytest.mark.parametrize("route", ["default", "execution"])
@pytest.mark.usefixtures("ample_session_memory")
def test_muap_sources_span_the_fibre_symmetrically(route, muscle, monkeypatch):
    from myogen.simulator.core.emg.runtime import semg as runtime_semg
    from myogen.simulator.core.emg.surface import surface_emg
    calls = []

    def spy(**kwargs):
        calls.append((kwargs["L1"], kwargs["L2"], kwargs["zi"]))
        array = kwargs["electrode_array"]
        return np.zeros((array.num_rows, array.num_cols, kwargs["N"])), None, None

    monkeypatch.setattr(surface_emg, "_simulate_fiber_v2_python", spy)
    monkeypatch.setattr(runtime_semg, "_simulate_fiber_v2_python", spy)
    _simulate(SurfaceEMG(muscle, [_array(1, 1, 5, 8)], MUs_to_simulate=[1, 2]), route)
    L1, L2, zi = np.array(calls).T
    assert np.all(zi != 0)
    # Fibre of length L1 + L2 centred on z = 0 with its end-plate at zi.
    np.testing.assert_allclose(zi + L1, (L1 + L2) / 2)
    np.testing.assert_allclose(zi - L2, -(L1 + L2) / 2)


@pytest.mark.parametrize("route", ["default", "execution"])
@pytest.mark.usefixtures("ample_session_memory")
def test_muap_time_axis_recovers_the_conduction_velocity(route, long_muscle):
    muscle = long_muscle
    mu = 1
    centre = muscle.innervation_center_positions__mm.magnitude[mu]
    column = SurfaceElectrodeArray(5, 1, 5 * pq.mm, 0.5 * pq.mm, center_point__mm_deg=(
        30 * pq.mm, np.degrees(np.arctan2(centre[1], centre[0])) * pq.deg))
    signal = _simulate(SurfaceEMG(muscle, [column], MUs_to_simulate=[mu]), route
                       ).groups[0].segments[mu].analogsignals[0]
    single_differential = np.diff(signal_to_grid(signal)[:, :, 0], axis=1)
    dt_ms = 1e3 / float(signal.sampling_rate)
    lags = [_upsampled_lag_ms(single_differential[:, i], single_differential[:, i + 1], dt_ms)
            for i in range(single_differential.shape[1] - 1)]
    velocity = float(muscle.mean_conduction_velocity__m_s.magnitude)
    assert 5.0 / np.median(lags) == pytest.approx(velocity, rel=0.03)


def test_mixed_electrode_array_shapes_keep_their_own_shape(muscle):
    semg = SurfaceEMG(muscle, [_array(2, 1, 10, 8), _array(1, 1, 10, 8, "bipolar_longitudinal")],
                      MUs_to_simulate=[1])
    block = semg.simulate_muaps(n_jobs=1, verbose=False, use_gpu=False)
    shapes = [signal_to_grid(group.segments[1].analogsignals[0]).shape[1:] for group in block.groups]
    assert shapes == [(2, 1), (1, 1)]
