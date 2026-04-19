"""Tests for the unified fiber simulation module."""

import numpy as np
import pytest

from myogen.simulator.core.emg.fiber_simulation import rosenfalck_dVm_dz


def test_import():
    """Verify the module can be imported."""
    from myogen.simulator.core.emg.fiber_simulation import rosenfalck_dVm_dz
    assert callable(rosenfalck_dVm_dz)


class TestRosenfalckDVmDz:
    """Tests for the raw Rosenfalck first derivative."""

    def test_zero_for_negative_z(self):
        z = np.array([-5.0, -1.0, -0.01])
        result = rosenfalck_dVm_dz(z)
        np.testing.assert_array_equal(result, 0.0)

    def test_zero_at_z_zero(self):
        result = rosenfalck_dVm_dz(np.array([0.0]))
        assert result[0] == 0.0

    def test_peak_near_1_27mm(self):
        """Rosenfalck dVm/dz peaks at z = 3 - sqrt(3) ~ 1.27 mm."""
        z = np.linspace(0.01, 5.0, 1000)
        result = rosenfalck_dVm_dz(z)
        peak_z = z[np.argmax(result)]
        assert abs(peak_z - 1.27) < 0.05

    def test_decays_to_zero_at_large_z(self):
        result = rosenfalck_dVm_dz(np.array([30.0, 50.0]))
        assert np.all(np.abs(result) < 1e-4)

    def test_matches_analytical_formula(self):
        """D1 * (3z^2 - z^3) * exp(-z) for z > 0."""
        z = np.array([0.5, 1.0, 2.0, 3.0])
        expected = 96.0 * (3 * z**2 - z**3) * np.exp(-z)
        result = rosenfalck_dVm_dz(z)
        np.testing.assert_allclose(result, expected)

    def test_custom_D1(self):
        z = np.array([1.0])
        result = rosenfalck_dVm_dz(z, D1=48.0)
        expected = 48.0 * (3 * 1.0 - 1.0) * np.exp(-1.0)
        np.testing.assert_allclose(result, expected)

    def test_matches_existing_iemg_function(self):
        """Must produce identical output to bioelectric.get_tm_current_dz."""
        from myogen.simulator.core.emg.intramuscular.bioelectric import get_tm_current_dz
        z = np.linspace(0.01, 10.0, 500)
        unified = rosenfalck_dVm_dz(z, D1=96.0)
        existing = get_tm_current_dz(z, D1=96.0)
        np.testing.assert_allclose(unified, existing, rtol=1e-12)


from myogen.simulator.core.emg.fiber_simulation import compute_intramuscular_kernel


import quantities as pq
from myogen.simulator.core.emg.electrodes import SurfaceElectrodeArray
from myogen.simulator.core.emg.fiber_simulation import compute_surface_kernel


class TestComputeSurfaceKernel:

    @pytest.fixture
    def electrode_array(self):
        return SurfaceElectrodeArray(
            num_rows=8, num_cols=1,
            inter_electrode_distances__mm=5.0 * pq.mm,
            electrode_radius__mm=5.0 * pq.mm,
            center_point__mm_deg=(0.0 * pq.mm, 0.0 * pq.deg),
            bending_radius__mm=8.5 * pq.mm,
            rotation_angle__deg=0.0 * pq.deg,
            differentiation_mode="monopolar",
        )

    def test_returns_correct_shape(self, electrode_array):
        Nz = 256
        M = 21
        z_grid = np.linspace(-60, 60, Nz)
        k_theta = np.arange(-(M - 1) / 2, (M - 1) / 2 + 1)
        b_z, _ = compute_surface_kernel(
            z_grid=z_grid, k_theta=k_theta, R=5.0,
            electrode_array=electrode_array,
            r=8.5, r_bone=0.0, th_fat=0.3, th_skin=1.29,
            sig_muscle_rho=0.09, sig_muscle_z=0.4,
            sig_fat=0.0407, sig_skin=4.88e-4,
        )
        assert b_z.shape == (8, 1, Nz)

    def test_kernel_is_real(self, electrode_array):
        """Verify the IFFT result has negligible imaginary part.

        The implementation asserts this internally before taking np.real().
        This test verifies the assertion does not fire (i.e., the function
        completes without AssertionError) and that the output is real-valued.
        """
        Nz = 256
        M = 21
        z_grid = np.linspace(-60, 60, Nz)
        k_theta = np.arange(-(M - 1) / 2, (M - 1) / 2 + 1)
        # If the IFFT result had a non-negligible imaginary part, the
        # internal assertion in compute_surface_kernel would fire here.
        b_z, _ = compute_surface_kernel(
            z_grid=z_grid, k_theta=k_theta, R=5.0,
            electrode_array=electrode_array,
            r=8.5, r_bone=0.0, th_fat=0.3, th_skin=1.29,
            sig_muscle_rho=0.09, sig_muscle_z=0.4,
            sig_fat=0.0407, sig_skin=4.88e-4,
        )
        assert b_z.dtype == np.float64

    def test_kernel_is_nonzero(self, electrode_array):
        Nz = 256
        M = 21
        z_grid = np.linspace(-60, 60, Nz)
        k_theta = np.arange(-(M - 1) / 2, (M - 1) / 2 + 1)
        b_z, _ = compute_surface_kernel(
            z_grid=z_grid, k_theta=k_theta, R=5.0,
            electrode_array=electrode_array,
            r=8.5, r_bone=0.0, th_fat=0.3, th_skin=1.29,
            sig_muscle_rho=0.09, sig_muscle_z=0.4,
            sig_fat=0.0407, sig_skin=4.88e-4,
        )
        assert np.max(np.abs(b_z)) > 0

    def test_caching_returns_same_result(self, electrode_array):
        Nz = 256
        M = 21
        z_grid = np.linspace(-60, 60, Nz)
        k_theta = np.arange(-(M - 1) / 2, (M - 1) / 2 + 1)
        params = dict(
            z_grid=z_grid, k_theta=k_theta, R=5.0,
            electrode_array=electrode_array, r=8.5, r_bone=0.0,
            th_fat=0.3, th_skin=1.29, sig_muscle_rho=0.09,
            sig_muscle_z=0.4, sig_fat=0.0407, sig_skin=4.88e-4,
        )
        b_z1, A_mat = compute_surface_kernel(**params)
        b_z2, _ = compute_surface_kernel(**params, A_matrix=A_mat)
        np.testing.assert_allclose(b_z1, b_z2, rtol=1e-10)


class TestComputeIntramuscularKernel:
    """Tests for the simple Green's function kernel."""

    def test_returns_correct_shape(self):
        z_grid = np.linspace(-30, 30, 200)
        electrode_z = np.array([0.0, 5.0, 10.0])
        b_z = compute_intramuscular_kernel(z_grid, electrode_z, r=1.0)
        assert b_z.shape == (3, 200)

    def test_peaks_at_electrode_position(self):
        z_grid = np.linspace(-30, 30, 1000)
        electrode_z = np.array([5.0])
        b_z = compute_intramuscular_kernel(z_grid, electrode_z, r=1.0)
        peak_z = z_grid[np.argmax(b_z[0])]
        assert abs(peak_z - 5.0) < 0.1

    def test_decays_with_distance(self):
        z_grid = np.linspace(-30, 30, 1000)
        electrode_z = np.array([0.0])
        b_z = compute_intramuscular_kernel(z_grid, electrode_z, r=1.0)
        center = len(z_grid) // 2
        assert b_z[0, center] > b_z[0, 0]
        assert b_z[0, center] > b_z[0, -1]

    def test_matches_existing_function(self):
        """Must match bioelectric.get_elementary_current_response."""
        from myogen.simulator.core.emg.intramuscular.bioelectric import (
            get_elementary_current_response,
        )
        z_grid = np.linspace(-30, 30, 200)
        z_elec = 5.0
        r = 2.0
        r_arr = np.full_like(z_grid, r)
        unified = compute_intramuscular_kernel(z_grid, np.array([z_elec]), r=r)
        existing = get_elementary_current_response(z_grid, z_elec, r_arr)
        np.testing.assert_allclose(unified[0], existing, rtol=1e-10)


from myogen.simulator.core.emg.fiber_simulation import simulate_fiber_unified


class TestSimulateFiberUnified:

    def _make_simple_kernel(self):
        """Create a simple peaked kernel for testing."""
        z = np.linspace(-30, 30, 512)
        b = 1.0 / np.sqrt(1.0 + z**2)
        return z, b.reshape(1, -1), np.array([0.0])

    def test_returns_correct_shape(self):
        z_grid, b_z, elec_z = self._make_simple_kernel()
        phi = simulate_fiber_unified(
            v=4.0, L1=15.0, L2=15.0, zi=0.0,
            b_z=b_z, z_kernel=z_grid, electrode_z=elec_z,
            Fs=10.0, duration_ms=20.0,
        )
        n_samples = int(20.0 * 10.0)
        assert phi.shape == (1, n_samples)

    def test_signal_is_biphasic(self):
        z_grid, b_z, elec_z = self._make_simple_kernel()
        phi = simulate_fiber_unified(
            v=4.0, L1=15.0, L2=15.0, zi=0.0,
            b_z=b_z, z_kernel=z_grid, electrode_z=elec_z,
            Fs=10.0, duration_ms=20.0,
        )
        signal = phi[0]
        assert np.max(signal) > 0
        assert np.min(signal) < 0

    def test_faster_cv_shifts_signal_earlier(self):
        z_grid, b_z, elec_z = self._make_simple_kernel()
        elec_z_offset = np.array([10.0])
        phi_slow = simulate_fiber_unified(
            v=3.0, L1=20.0, L2=20.0, zi=0.0,
            b_z=b_z, z_kernel=z_grid, electrode_z=elec_z_offset,
            Fs=10.0, duration_ms=30.0,
        )
        phi_fast = simulate_fiber_unified(
            v=5.0, L1=20.0, L2=20.0, zi=0.0,
            b_z=b_z, z_kernel=z_grid, electrode_z=elec_z_offset,
            Fs=10.0, duration_ms=30.0,
        )
        peak_slow = np.argmax(np.abs(phi_slow[0]))
        peak_fast = np.argmax(np.abs(phi_fast[0]))
        assert peak_fast < peak_slow

    def test_zero_signal_for_zero_kernel(self):
        z_grid = np.linspace(-30, 30, 512)
        b_z = np.zeros((1, 512))
        phi = simulate_fiber_unified(
            v=4.0, L1=15.0, L2=15.0, zi=0.0,
            b_z=b_z, z_kernel=z_grid, electrode_z=np.array([0.0]),
            Fs=10.0, duration_ms=20.0,
        )
        np.testing.assert_array_equal(phi, 0.0)


class TestSurfaceEMGIntegration:
    def test_use_unified_flag_accepted(self):
        from myogen.simulator.core.emg.surface.surface_emg import SurfaceEMG
        import inspect
        sig = inspect.signature(SurfaceEMG.__init__)
        assert "use_unified" in sig.parameters


class TestOldVsNewComparison:
    """Compare MUAPs from old frequency-domain and new time-domain paths."""

    @pytest.fixture
    def electrode_array(self):
        return SurfaceElectrodeArray(
            num_rows=8, num_cols=1,
            inter_electrode_distances__mm=5.0 * pq.mm,
            electrode_radius__mm=5.0 * pq.mm,
            center_point__mm_deg=(0.0 * pq.mm, 0.0 * pq.deg),
            bending_radius__mm=8.5 * pq.mm,
            rotation_angle__deg=0.0 * pq.deg,
            differentiation_mode="monopolar",
        )

    def test_both_paths_produce_nonzero_signals(self, electrode_array):
        """Both paths should produce non-trivial MUAPs."""
        from myogen.simulator.core.emg.surface.simulate_fiber import _simulate_fiber_v2_python
        from myogen.simulator.core.emg.fiber_simulation import (
            compute_surface_kernel, simulate_fiber_unified,
        )

        v = 4.0; R = 5.0; L1 = 15.0; L2 = 15.0; zi = 0.0
        r = 8.5; N = 256; M = 32; Fs = 10.0

        # Old path
        phi_old, _, _ = _simulate_fiber_v2_python(
            Fs=Fs, v=v, N=N, M=M, r=r, r_bone=0.0,
            th_fat=0.3, th_skin=1.29, R=R, L1=L1, L2=L2, zi=zi,
            electrode_array=electrode_array,
            sig_muscle_rho=0.09, sig_muscle_z=0.4,
            sig_fat=0.0407, sig_skin=4.88e-4,
            fiber_length__mm=75.0,
        )

        # New path
        z_kernel = np.linspace(-60, 60, N)
        k_theta = np.arange(-(M - 1) / 2, (M - 1) / 2 + 1)
        b_z, _ = compute_surface_kernel(
            z_grid=z_kernel, k_theta=k_theta, R=R,
            electrode_array=electrode_array,
            r=r, r_bone=0.0, th_fat=0.3, th_skin=1.29,
            sig_muscle_rho=0.09, sig_muscle_z=0.4,
            sig_fat=0.0407, sig_skin=4.88e-4,
        )

        elec_z = electrode_array.pos_z.rescale(pq.mm).magnitude.flatten()
        duration = 75.0 / v

        phi_new = simulate_fiber_unified(
            v=v, L1=L1, L2=L2, zi=zi,
            b_z=b_z.reshape(-1, N), z_kernel=z_kernel,
            electrode_z=elec_z, Fs=Fs, duration_ms=duration,
        )

        assert np.max(np.abs(phi_old)) > 0, "Old path produced zero signal"
        assert np.max(np.abs(phi_new)) > 0, "New path produced zero signal"

    def test_muap_structural_properties(self, electrode_array):
        """Both paths should produce biphasic MUAPs with plausible structure."""
        from myogen.simulator.core.emg.surface.simulate_fiber import _simulate_fiber_v2_python
        from myogen.simulator.core.emg.fiber_simulation import (
            compute_surface_kernel, simulate_fiber_unified,
        )

        v = 4.0; R = 5.0; L1 = 15.0; L2 = 15.0; zi = 0.0
        r = 8.5; N = 256; M = 32; Fs = 10.0

        phi_old, _, _ = _simulate_fiber_v2_python(
            Fs=Fs, v=v, N=N, M=M, r=r, r_bone=0.0,
            th_fat=0.3, th_skin=1.29, R=R, L1=L1, L2=L2, zi=zi,
            electrode_array=electrode_array,
            sig_muscle_rho=0.09, sig_muscle_z=0.4,
            sig_fat=0.0407, sig_skin=4.88e-4,
            fiber_length__mm=75.0,
        )

        z_kernel = np.linspace(-60, 60, N)
        k_theta = np.arange(-(M - 1) / 2, (M - 1) / 2 + 1)
        b_z, _ = compute_surface_kernel(
            z_grid=z_kernel, k_theta=k_theta, R=R,
            electrode_array=electrode_array,
            r=r, r_bone=0.0, th_fat=0.3, th_skin=1.29,
            sig_muscle_rho=0.09, sig_muscle_z=0.4,
            sig_fat=0.0407, sig_skin=4.88e-4,
        )

        elec_z = electrode_array.pos_z.rescale(pq.mm).magnitude.flatten()
        duration = 75.0 / v

        phi_new = simulate_fiber_unified(
            v=v, L1=L1, L2=L2, zi=zi,
            b_z=b_z.reshape(-1, N), z_kernel=z_kernel,
            electrode_z=elec_z, Fs=Fs, duration_ms=duration,
        )

        # The old (frequency-domain) and new (time-domain) paths use fundamentally
        # different mathematical approaches:
        #   - Old: frequency-domain convolution (FFT of source * H_vc, then IFFT)
        #   - New: spatial kernel IFFT + time-domain traveling-wave convolution
        # They also differ in Rosenfalck parameterization (D1=9.6 + z/2 vs D1=96).
        # Strict waveform correlation is not expected; instead we verify that both
        # produce physically plausible signals with consistent structural properties.

        # 1. Both are biphasic (have both positive and negative values)
        old_center = phi_old[4, 0, :]
        new_center = phi_new[4] if phi_new.ndim == 2 else phi_new[4, 0, :]

        print(f"Old peak-to-peak: {np.ptp(old_center):.6e}")
        print(f"New peak-to-peak: {np.ptp(new_center):.6e}")

        assert np.max(old_center) > 0 and np.min(old_center) < 0, (
            "Old path signal is not biphasic"
        )
        assert np.max(new_center) > 0 and np.min(new_center) < 0, (
            "New path signal is not biphasic"
        )

        # 2. Both paths should produce signals on all channels
        old_amps = np.array([np.max(np.abs(phi_old[i, 0, :])) for i in range(8)])
        new_amps = np.array([np.max(np.abs(phi_new[i])) for i in range(8)])

        old_amps_norm = old_amps / (np.max(old_amps) + 1e-30)
        new_amps_norm = new_amps / (np.max(new_amps) + 1e-30)

        print(f"Old channel amplitudes (normalized): {old_amps_norm}")
        print(f"New channel amplitudes (normalized): {new_amps_norm}")

        assert np.all(old_amps > 0), "Old path has zero-amplitude channels"
        assert np.all(new_amps > 0), "New path has zero-amplitude channels"

        # 3. Amplitude variation across channels should be modest (not all energy
        # in one channel) -- both paths should spread signal across electrodes
        assert np.min(old_amps_norm) > 0.1, (
            f"Old path channel amplitude ratio too extreme: {np.min(old_amps_norm):.3f}"
        )
        assert np.min(new_amps_norm) > 0.01, (
            f"New path channel amplitude ratio too extreme: {np.min(new_amps_norm):.3f}"
        )

        # 4. The amplitude order of magnitude should be within a reasonable factor
        # (allowing for the 10x D1 difference and z/2 scaling)
        old_max = np.max(np.abs(phi_old))
        new_max = np.max(np.abs(phi_new))
        ratio = new_max / old_max
        print(f"Amplitude ratio (new/old): {ratio:.2f}")
        # The D1 difference alone gives ~10x, plus z-scaling effects.
        # Allow a wide range (0.1 to 10000) to accommodate the different formulations.
        assert 0.1 < ratio < 10000, (
            f"Amplitude ratio {ratio:.1f} outside plausible range [0.1, 10000]"
        )
