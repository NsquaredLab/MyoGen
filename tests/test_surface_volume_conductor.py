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
from scipy.special import iv

from myogen.simulator.core.emg.electrodes import SurfaceElectrodeArray
from myogen.simulator.core.emg.fiber_simulation import simulate_fiber_hybrid
from myogen.simulator.core.emg.surface.simulate_fiber import (
    HAS_CYTHON_FIBER,
    _simulate_fiber_v2_python,
    simulate_fiber_v2,
)
from myogen.simulator.core.emg.surface.volume_conductor import (
    layered_cylinder_transfer,
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
# Transfer-function-only view of the same cylinder (no fibre extent).
GEOMETRY_TRANSFER = {
    k: v for k, v in GEOMETRY.items() if k not in ("L1", "L2", "zi")
} | {"R": 30.0}


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


def _fft_bin_grids(N: int, M: int, Fs_kHz: float, v_m_s: float):
    """k_z (rad/mm) and integer k_theta as built by ``_simulate_fiber_v2_python``."""
    k_t = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=1.0 / Fs_kHz))
    k_t = np.where(k_t == 0.0, 0.5 * (2 * np.pi * Fs_kHz / N), k_t)
    k_theta = np.fft.fftshift(np.fft.fftfreq(M, d=1.0 / M))
    return k_t / v_m_s, k_theta


class TestLayeredCylinderTransfer:
    """Direct checks on the shared transfer-function authority."""

    ISO = dict(
        r=40.0,
        r_bone=10.0,
        th_fat=2.0,
        th_skin=1.0,
        R=30.0,
        sig_muscle_rho=0.2,
        sig_muscle_z=0.2,
        sig_fat=0.2,
        sig_skin=0.2,
        sig_bone=0.2,
    )

    def test_matches_homogeneous_cylinder_closed_form_on_full_grid(self):
        """All layers at one isotropic sigma reduce the four-layer system to
        a homogeneous cylinder of radius d, insulated at r = d, with a unit
        line source at radius R.  The Wronskian I_n K_n' - I_n' K_n = -1/x
        collapses the skin potential to

            H_vc(k, n) = I_n(kR) / (sigma * k d * I_n'(kd)),
            I_n'(x) = (I_{n-1}(x) + I_{n+1}(x)) / 2.

        Checked at every (k_z, k_theta) slot, negative halves included, with
        no free scale factor.
        """
        k_z, k_theta = _fft_bin_grids(N=512, M=32, Fs_kHz=4.096, v_m_s=4.0)
        H, _, _ = layered_cylinder_transfer(k_z, k_theta, **self.ISO)

        R, d, sigma = self.ISO["R"], self.ISO["r"], self.ISO["sig_muscle_rho"]
        K, NN = np.meshgrid(np.abs(k_z), np.abs(k_theta), indexing="ij")
        expected = iv(NN, K * R) / (
            sigma * K * d * 0.5 * (iv(NN - 1, K * d) + iv(NN + 1, K * d))
        )
        # Beyond k_z ~ 1.5 rad/mm the Bessel ratios under/overflow in double
        # precision in both the solver and the closed form; the source
        # spectrum is negligible there.
        sel = np.abs(k_z) < 1.5
        rel_err = np.linalg.norm(H[sel] - expected[sel]) / np.linalg.norm(expected[sel])
        assert rel_err < 1e-12

    def test_even_in_both_arguments(self):
        k_z, k_theta = _fft_bin_grids(N=512, M=32, Fs_kHz=4.096, v_m_s=4.0)
        H, _, _ = layered_cylinder_transfer(k_z, k_theta, **GEOMETRY_TRANSFER)
        for j, n in enumerate(k_theta):
            j_mirror = np.flatnonzero(k_theta == -n)
            if j_mirror.size:
                np.testing.assert_array_equal(H[:, j], H[:, j_mirror[0]])
        for i, k in enumerate(k_z):
            i_mirror = np.flatnonzero(k_z == -k)
            if i_mirror.size:
                np.testing.assert_array_equal(H[i, :], H[i_mirror[0], :])

    def test_even_and_odd_grids_agree_on_shared_bins(self):
        """Odd M/N give symmetric grids; the transfer at a given |k| pair must
        not depend on which grid it was requested from."""
        # Same k_z spacing (Fs scales with N) so the two grids share bins.
        k_z_e, k_theta_e = _fft_bin_grids(N=512, M=32, Fs_kHz=4.096, v_m_s=4.0)
        k_z_o, k_theta_o = _fft_bin_grids(
            N=513, M=33, Fs_kHz=4.096 * 513 / 512, v_m_s=4.0
        )
        H_e, _, _ = layered_cylinder_transfer(k_z_e, k_theta_e, **GEOMETRY_TRANSFER)
        H_o, _, _ = layered_cylinder_transfer(k_z_o, k_theta_o, **GEOMETRY_TRANSFER)
        assert H_e.shape == (512, 32) and H_o.shape == (513, 33)
        # Every bin of the even grid exists on the odd grid (the nudged DC
        # bin included, since the spacing is identical).
        shared_z = [
            (ie, io)
            for ie, k in enumerate(k_z_e)
            for io in np.flatnonzero(np.isclose(k_z_o, k, rtol=1e-12, atol=0.0))
        ]
        assert len(shared_z) == 512
        shared_theta = [
            (je, np.flatnonzero(k_theta_o == n)[0]) for je, n in enumerate(k_theta_e)
        ]
        ie, io = map(np.array, zip(*shared_z))
        je, jo = map(np.array, zip(*shared_theta))
        np.testing.assert_allclose(
            H_e[np.ix_(ie, je)], H_o[np.ix_(io, jo)], rtol=1e-12, atol=0.0
        )

    @pytest.mark.parametrize("N", [512, 2048])
    def test_thin_bone_is_continuous_with_no_bone(self, N):
        """Normalising the muscle Kn coefficient at the outer radius made the
        bone rows overflow past (bm - am) k ~ 700: r_bone < ~8 mm raised
        LinAlgError at the default grid and every r_bone did at N >= 1024,
        with garbage of ~1e20 just below the threshold.  The skin potential
        must vary smoothly with the bone radius and be grid-converged."""
        # Grid as built by _simulate_fiber_v2_python for fiber_length 150 mm.
        Fs_eff = SIGNAL["v"] * (N - 1) / 150.0
        k_z, k_theta = _fft_bin_grids(N=N, M=32, Fs_kHz=Fs_eff, v_m_s=SIGNAL["v"])
        geometry = {k: v for k, v in GEOMETRY_TRANSFER.items() if k != "r_bone"}
        H = {
            rb: layered_cylinder_transfer(k_z, k_theta, r_bone=rb, **geometry)[0]
            for rb in (0.0, 1.0, 3.0, 5.0, 10.0)
        }
        j0 = np.flatnonzero(k_theta == 0)[0]
        # Mode 0 decays monotonically in |k_z| for every bone radius.
        for rb, H_rb in H.items():
            order = np.argsort(np.abs(k_z))
            mode0 = np.abs(H_rb[order, j0])
            assert np.all(np.diff(mode0[::2]) <= 0), rb
            assert np.all(np.isfinite(H_rb))
        # A 1 mm bone in a 40 mm cylinder is indistinguishable from none.
        assert _rel_l2(H[1.0], H[0.0]) < 1e-3
        # ...and the dependence on the bone radius is monotone.
        peaks = [np.max(np.abs(H[rb])) for rb in (0.0, 1.0, 3.0, 5.0, 10.0)]
        assert np.all(np.diff(peaks) > 0)

    def test_dc_column_is_zero_and_cache_is_reused(self):
        k_z, k_theta = _fft_bin_grids(N=512, M=32, Fs_kHz=4.096, v_m_s=4.0)
        k_z_with_dc = np.where(np.abs(k_z) == np.min(np.abs(k_z)), 0.0, k_z)
        H, A, B = layered_cylinder_transfer(k_z_with_dc, k_theta, **GEOMETRY_TRANSFER)
        assert np.all(H[k_z_with_dc == 0.0, :] == 0.0)
        H2, _, _ = layered_cylinder_transfer(
            k_z_with_dc, k_theta, A_matrix=A, B_incomplete=B, **GEOMETRY_TRANSFER
        )
        np.testing.assert_array_equal(H, H2)


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
