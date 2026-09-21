"""Regression tests for the Farina (2004) layered-cylinder volume conductor.

The transfer function H_vc(k_z, k_theta) is even in both arguments, so every
implementation computes it on the non-negative quadrant and reconstructs the
full grid by symmetry.  Commit ``b666a3b3`` moved the ``k_theta`` / ``k_t``
grids to FFT bin centres (``-N/2 .. N/2-1``, asymmetric for even N) but kept a
mirror that is only valid for the previous symmetric ``linspace`` grid.  On the
new grid the mirror shifts the whole negative half by one bin (DC counted
twice, the Nyquist bin dropped), which roughly doubles the surface potential.

These tests pin the behaviour that a correct reconstruction must have without
relying on any particular grid convention:

* an odd ``M`` (symmetric truncation) must agree with an even ``M``,
* rotating the electrode by a full turn must not change the signal
  (integer angular orders only),
* the Cython port's reconstruction must satisfy the same invariants,
* the ``r_bone = 0`` reduced system must be continuous with a small bone.
"""

from __future__ import annotations

import numpy as np
import pytest
import quantities as pq

from myogen.simulator.core.emg.electrodes import SurfaceElectrodeArray
from myogen.simulator.core.emg.fiber_simulation import simulate_fiber_hybrid
from myogen.simulator.core.emg.surface.simulate_fiber import (
    HAS_CYTHON_FIBER,
    _simulate_fiber_v2_python,
    simulate_fiber_v2,
)

# Standard validation cylinder (Farina 2004): bone 10, muscle to 35, fat to 38,
# skin to 40 mm.  One monopolar row of three electrodes 10 mm apart around the
# circumference, centred over the fibre.
R_TOTAL = 40.0
GEOMETRY = dict(
    r=R_TOTAL,
    r_bone=10.0,
    th_fat=2.0,
    th_skin=1.0,
    L1=60.0,
    L2=60.0,
    zi=0.0,
    sig_muscle_rho=0.1,
    sig_muscle_z=0.5,
    sig_fat=0.04,
    sig_skin=1.0,
    sig_bone=0.02,
)
SIGNAL = dict(Fs=4.096, v=4.0, N=512)


def _electrode_array() -> SurfaceElectrodeArray:
    return SurfaceElectrodeArray(
        num_rows=1,
        num_cols=3,
        inter_electrode_distance__mm=10.0 * pq.mm,
        electrode_radius__mm=1.0 * pq.mm,
        bending_radius__mm=R_TOTAL * pq.mm,
        differentiation_mode="monopolar",
    )


def _centre(phi: np.ndarray) -> np.ndarray:
    return phi[0, 1, :]


def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


@pytest.fixture(scope="module")
def electrode_array() -> SurfaceElectrodeArray:
    return _electrode_array()


class TestAngularOrderTruncation:
    """Even and odd ``M`` truncate the same integer-order series."""

    @pytest.mark.parametrize("R", [33.0, 25.0])
    def test_python_even_and_odd_M_agree(self, electrode_array, R):
        common = dict(
            **SIGNAL,
            **GEOMETRY,
            R=R,
            electrode_array=electrode_array,
            fiber_length__mm=150.0,
            use_gpu=False,
        )
        phi_even, _, _ = _simulate_fiber_v2_python(M=32, **common)
        phi_odd, _, _ = _simulate_fiber_v2_python(M=33, **common)
        # Odd M adds exactly one further angular order (|k| = 16), whose
        # contribution is negligible for a source >= 7 mm below the skin.
        assert _rel_l2(_centre(phi_even), _centre(phi_odd)) < 0.02

    @pytest.mark.parametrize("R", [33.0, 25.0])
    def test_hybrid_even_and_odd_M_agree(self, electrode_array, R):
        common = dict(
            **SIGNAL,
            **GEOMETRY,
            R=R,
            electrode_array=electrode_array,
            fiber_length__mm=150.0,
            use_gpu=False,
        )
        phi_even, _, _ = simulate_fiber_hybrid(M=32, **common)
        phi_odd, _, _ = simulate_fiber_hybrid(M=33, **common)
        assert _rel_l2(_centre(phi_even), _centre(phi_odd)) < 0.02


class TestAngularPeriodicity:
    """The potential on a cylinder is 2*pi-periodic in theta.

    Half-integer angular orders (``linspace(-(M-1)/2, (M-1)/2, M)`` for even
    M) build a series with period 4*pi, so a full turn flips the sign.
    """

    def test_python_full_turn_is_identity(self, electrode_array):
        common = dict(
            **SIGNAL,
            **GEOMETRY,
            M=32,
            R=30.0,
            electrode_array=electrode_array,
            fiber_length__mm=150.0,
            use_gpu=False,
        )
        phi_0, _, _ = _simulate_fiber_v2_python(theta_offset=0.0, **common)
        phi_2pi, _, _ = _simulate_fiber_v2_python(theta_offset=2.0 * np.pi, **common)
        np.testing.assert_allclose(phi_2pi, phi_0, rtol=1e-9, atol=1e-12)

    def test_hybrid_full_turn_is_identity(self, electrode_array):
        common = dict(
            **SIGNAL,
            **GEOMETRY,
            M=32,
            R=30.0,
            electrode_array=electrode_array,
            fiber_length__mm=150.0,
            use_gpu=False,
        )
        phi_0, _, _ = simulate_fiber_hybrid(theta_offset=0.0, **common)
        phi_2pi, _, _ = simulate_fiber_hybrid(theta_offset=2.0 * np.pi, **common)
        np.testing.assert_allclose(phi_2pi, phi_0, rtol=1e-9, atol=1e-12)


@pytest.mark.skipif(not HAS_CYTHON_FIBER, reason="Cython extension not built")
class TestCythonVolumeConductor:
    """The Cython port carries its own copy of the volume-conductor assembly.

    Its source model diverged from the Python reference (fixed +-6 mm IAP
    grid, different amplitude, real-valued angular sum), so waveform parity
    with ``_simulate_fiber_v2_python`` is not expected; only the
    volume-conductor reconstruction is pinned here.
    """

    @pytest.mark.parametrize("R", [33.0, 25.0])
    def test_cython_even_and_odd_M_agree(self, electrode_array, R):
        common = dict(
            **SIGNAL,
            **GEOMETRY,
            R=R,
            electrode_array=electrode_array,
            fiber_length__mm=None,  # Cython supports the legacy z-grid only
            use_gpu=False,
        )
        phi_even, _, _ = simulate_fiber_v2(use_cython=True, M=32, **common)
        phi_odd, _, _ = simulate_fiber_v2(use_cython=True, M=33, **common)
        assert _rel_l2(_centre(phi_even), _centre(phi_odd)) < 0.02

    def test_cython_r_bone_zero_is_continuous_with_small_bone(self, electrode_array):
        """r_bone = 0 selects a reduced 5x5 system; it must stay close to a
        thin bone, not collapse to round-off (the pre-087caf33 column bug)."""
        common = dict(
            **SIGNAL,
            **{k: v for k, v in GEOMETRY.items() if k != "r_bone"},
            M=32,
            R=33.0,
            electrode_array=electrode_array,
            fiber_length__mm=None,
            use_gpu=False,
        )
        phi_none, _, _ = simulate_fiber_v2(use_cython=True, r_bone=0.0, **common)
        phi_thin, _, _ = simulate_fiber_v2(use_cython=True, r_bone=8.0, **common)
        assert np.max(np.abs(phi_none)) > 1e-6
        assert _rel_l2(_centre(phi_none), _centre(phi_thin)) < 0.05


class TestPythonReducedSystem:
    def test_python_r_bone_zero_is_continuous_with_small_bone(self, electrode_array):
        common = dict(
            **SIGNAL,
            **{k: v for k, v in GEOMETRY.items() if k != "r_bone"},
            M=32,
            R=33.0,
            electrode_array=electrode_array,
            fiber_length__mm=150.0,
            use_gpu=False,
        )
        phi_none, _, _ = _simulate_fiber_v2_python(r_bone=0.0, **common)
        phi_thin, _, _ = _simulate_fiber_v2_python(r_bone=8.0, **common)
        assert np.max(np.abs(phi_none)) > 1e-6
        assert _rel_l2(_centre(phi_none), _centre(phi_thin)) < 0.05
