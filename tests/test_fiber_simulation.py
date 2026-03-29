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
