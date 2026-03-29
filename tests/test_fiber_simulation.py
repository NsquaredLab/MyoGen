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
