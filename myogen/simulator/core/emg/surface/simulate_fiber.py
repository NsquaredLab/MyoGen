#######################################################################################################
##################################### Initial Explanations ############################################
#######################################################################################################

# This script simulates a single fiber: it must be runned in the script simulate_muscle_XXX.py, which simulates
# this script several times and generates the MUAPs for each MU dectected at each electrode channel.
# Length parameters are in mm, frequencies in kHz, time in ms, conductivities in S/m


#######################################################################################################
##################################### Input Parameters ################################################
#######################################################################################################

# Fs             -> Sampling frequency
# mean_conduction_velocity__mm_s              -> Conduction velocity
# N              -> Number of points in t and z domains
# M              -> Number of points in theta domain
# r              -> Model total radius
# r_bone         -> Bone radius
# th_fat         -> Fat thickness
# th_skin        -> Skin thickness
# R              -> Source position in rho coordinate
# L1             -> Semifiber length (z > 0)
# L2             -> Semifiber length (z < 0)
# zi             -> Innervation zone (z = 0 for the mean innervation zone of the M.U)
# alpha          -> Inclination angle (in degrees) between the eletrode matrix and the muscle fibers
# channels       -> Matrix of electrodes (tuple). The columns are aligned with the muscle fibers if alpha = 0
# electrode_grid_center -> Matrix of electrodes centers (tuple -> (z electrode_grid_center in mm, theta electrode_grid_center in degrees))
# d_ele          -> Distance between neighboring electrodes
# rele           -> Electrode radius (circular electrodes)
# sig_bone       -> Bone conductivity
# sig_muscle_rho -> Muscle conductivity in rho direction
# sig_muscle_z   -> Muscle conductivity in z direction
# sig_fat        -> Fat conductivity
# sig_skin       -> Skin conductivity


#######################################################################################################
##################################### Model ###########################################################
#######################################################################################################

import math

import numpy as np
from scipy.special import iv as In
from scipy.special import ive as In_scaled  # Exponentially scaled: ive(n,z) = iv(n,z)*exp(-|z|)
from scipy.special import jv as Jn
from scipy.special import kv as Kn
from scipy.special import kve as Kn_scaled  # Exponentially scaled: kve(n,z) = kv(n,z)*exp(z)

from myogen.utils.decorators import beartowertype

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

try:
    import numba
    from numba import njit, prange

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

from myogen.simulator.core.emg.electrodes import SurfaceElectrodeArray

# Try to import Cython-optimized version
try:
    from myogen.simulator.neuron._cython._simulate_fiber import (
        simulate_fiber_v2 as _simulate_fiber_v2_cython,
    )

    HAS_CYTHON_FIBER = True
except ImportError:
    HAS_CYTHON_FIBER = False
    _simulate_fiber_v2_cython = None  # type: ignore


def In_tilde(K_THETA, x):
    return (In(K_THETA + 1, x) + In(K_THETA - 1, x)) / 2


def In_tilde_scaled(K_THETA, x):
    """Scaled version of In_tilde to avoid overflow."""
    return (In_scaled(K_THETA + 1, x) + In_scaled(K_THETA - 1, x)) / 2


def Kn_tilde(K_THETA, x):
    return (Kn(K_THETA + 1, x) + Kn(K_THETA - 1, x)) / 2


def Kn_tilde_scaled(K_THETA, x):
    """Scaled version of Kn_tilde to avoid overflow."""
    return (Kn_scaled(K_THETA + 1, x) + Kn_scaled(K_THETA - 1, x)) / 2


def log_In(K_THETA, x):
    """Compute log(In(n,z)) in a numerically stable way."""
    # log(In(n,z)) = log(In_scaled(n,z)) + |z|
    In_s = In_scaled(K_THETA, x)
    # Handle zeros and negative values
    result = np.where(In_s > 0, np.log(In_s) + np.abs(x), -np.inf)
    return result


def log_Kn(K_THETA, x):
    """Compute log(Kn(n,z)) in a numerically stable way."""
    # log(Kn(n,z)) = log(Kn_scaled(n,z)) - z
    Kn_s = Kn_scaled(K_THETA, x)
    # Handle zeros and negative values
    result = np.where(Kn_s > 0, np.log(Kn_s) - x, -np.inf)
    return result


def f_minus_t(y):
    y_new = np.zeros(len(y))
    for i in range(0, len(y)):
        y_new[i] = y[-i]
    return y_new


# Numba-optimized helper functions for simulate_fiber_v3
def _get_numba_functions():
    """Get Numba-optimized functions if available, otherwise return None."""
    if not HAS_NUMBA:
        return None, None

    @njit(parallel=True, cache=True)
    def _compute_B_kz_fast(
        H_glo_real,
        H_glo_imag,
        pos_theta,
        k_theta_diff,
        ktheta_mesh_kzktheta,
        channels_0,
        channels_1,
        len_kz,
        len_ktheta,
    ):
        """Numba-optimized computation of B_kz matrix."""
        B_kz = np.zeros((channels_0, channels_1, len_kz))

        for channel_z in prange(channels_0):
            for channel_theta in prange(channels_1):
                pos_theta_val = pos_theta[channel_z, channel_theta]
                for i in range(len_kz):
                    sum_real = 0.0
                    for j in range(len_ktheta):
                        phase = pos_theta_val * ktheta_mesh_kzktheta[i, j]
                        cos_phase = math.cos(phase)
                        sin_phase = math.sin(phase)

                        # Complex multiplication: H_glo * exp(1j * phase)
                        result_real = H_glo_real[i, j] * cos_phase - H_glo_imag[i, j] * sin_phase
                        sum_real += result_real

                    B_kz[channel_z, channel_theta, i] = sum_real * k_theta_diff / (2 * math.pi)

        return B_kz

    @njit(parallel=True, cache=True)
    def _compute_phi_fast(
        I_kzkt_real,
        I_kzkt_imag,
        B_kz,
        pos_z,
        kz_mesh_kzkt,
        channels_0,
        channels_1,
        len_kz,
        len_kt,
        k_z_diff,
    ):
        """Numba-optimized computation of phi signals."""
        PHI_complex = np.zeros((channels_0, channels_1, len_kt), dtype=np.complex128)

        for channel_z in prange(channels_0):
            for channel_theta in prange(channels_1):
                pos_z_val = pos_z[channel_z, channel_theta]

                for j in range(len_kt):
                    sum_real = 0.0
                    sum_imag = 0.0

                    for i in range(len_kz):
                        # I_kzkt[i, j] * B_kz[channel_z, channel_theta, i]
                        B_val = B_kz[channel_z, channel_theta, i]
                        arg_real = I_kzkt_real[i, j] * B_val
                        arg_imag = I_kzkt_imag[i, j] * B_val

                        # arg * exp(1j * pos_z * kz_mesh_kzkt[i, j])
                        phase = pos_z_val * kz_mesh_kzkt[i, j]
                        cos_phase = math.cos(phase)
                        sin_phase = math.sin(phase)

                        result_real = arg_real * cos_phase - arg_imag * sin_phase
                        result_imag = arg_real * sin_phase + arg_imag * cos_phase

                        sum_real += result_real
                        sum_imag += result_imag

                    PHI_complex[channel_z, channel_theta, j] = complex(
                        sum_real * k_z_diff / (2 * math.pi),
                        sum_imag * k_z_diff / (2 * math.pi),
                    )

        return PHI_complex

    return _compute_B_kz_fast, _compute_phi_fast


# Get the optimized functions
_numba_B_kz_func, _numba_phi_func = _get_numba_functions()


@beartowertype
def _simulate_fiber_v2_python(
    Fs: float,
    v: float,
    N: int,
    M: int,
    r: float,
    r_bone: float,
    th_fat: float,
    th_skin: float,
    R: float,
    L1: float,
    L2: float,
    zi: float,
    electrode_array: SurfaceElectrodeArray,
    sig_muscle_rho: float,
    sig_muscle_z: float,
    sig_fat: float,
    sig_skin: float,
    sig_bone: float = 0,
    A_matrix: np.ndarray | None = None,
    B_incomplete: np.ndarray | None = None,
    use_gpu: bool = True,
    fiber_length__mm: float | None = None,
):
    """
    Simulate a single fiber (Python implementation).

    This is the original Python implementation, kept for fallback and validation.

    Parameters
    ----------
    Fs : float
        Sampling frequency in kHz.
    v : float
        Conduction velocity in m/s.
    N : int
        Number of points in t and z domains (controls resolution).
    M : int
        Number of points in theta domain.
    r : float
        Total radius of the muscle model in mm.
    r_bone : float
        Bone radius in mm.
    th_fat : float
        Fat thickness in mm.
    th_skin : float
        Skin thickness in mm.
    R : float
        Source position in rho coordinate (mm).
    L1 : float
        Semifiber length (z > 0) in mm.
    L2 : float
        Semifiber length (z < 0) in mm
    zi : float
        Innervation zone position in mm.
    electrode_array : SurfaceElectrodeArray
        Electrode array configuration object.
    sig_muscle_rho : float
        Muscle conductivity in rho direction (S/m).
    sig_muscle_z : float
        Muscle conductivity in z direction (S/m).
    sig_fat : float
        Fat conductivity (S/m).
    sig_skin : float
        Skin conductivity (S/m).
    sig_bone : float, optional
        Bone conductivity (S/m), default=0
    A_matrix : np.ndarray, optional
        Pre-computed A matrix for optimization
    B_incomplete : np.ndarray, optional
        Pre-computed B matrix for optimization
    use_gpu : bool, optional
        If True, may use GPU acceleration if available, by default True.
    fiber_length__mm : float, optional
        Physical fiber length for IAP kernel evaluation in mm. This defines the spatial
        extent over which the intracellular action potential kernel is computed.
        If None (default), uses legacy behavior: (N-1) * (v/Fs) mm, which incorrectly
        ties MUAP duration to sampling resolution.

        **Recommended**: Set this explicitly (e.g., 50-100 mm for typical motor units)
        to ensure MUAP duration is physically accurate and independent of N.

        The actual MUAP duration will be approximately: fiber_length__mm / (2 * v) ms
        (factor of 2 due to internal scaling for Rosenfalck model).

        Example: fiber_length__mm=100 with v=4 m/s → ~12.5 ms MUAP duration.

    Returns
    -------
    phi : np.ndarray
        Generated surface EMG signal for each electrode
    A_matrix : np.ndarray
        A matrix for reuse in subsequent calls
    B_incomplete : np.ndarray
        B matrix for reuse in subsequent calls
    """
    # Get electrode configuration from the array
    channels = [electrode_array.num_rows, electrode_array.num_cols]

    # Extract magnitudes from Quantity objects for numerical operations
    import quantities as pq

    rele = float(electrode_array.electrode_radius__mm.rescale(pq.mm).magnitude)
    pos_z_mm = electrode_array.pos_z.rescale(pq.mm).magnitude  # Extract as plain array in mm
    pos_theta_rad = electrode_array.pos_theta.rescale(
        pq.rad
    ).magnitude  # Extract as plain array in rad

    ###################################################################################################
    ## 1. Constants

    ## Model angular frequencies
    k_theta = np.linspace(-(M - 1) / 2, (M - 1) / 2, M)

    # Calculate effective Fs based on grid spacing
    # When fiber_length__mm is specified, we must use the corresponding Fs for FFT consistency
    if fiber_length__mm is not None:
        # z-grid spans fiber_length__mm with N points (before z /= 2 scaling)
        # Grid spacing (before scaling): dz = fiber_length__mm / (N-1)
        # For FFT: dz = v / Fs_eff → Fs_eff = v * (N-1) / fiber_length__mm
        # Note: The z /= 2 scaling affects the IAP kernel evaluation but the FFT
        # operates on the array structure, so we use the pre-scaling grid spacing
        Fs_effective = v * (N - 1) / fiber_length__mm  # kHz
    else:
        Fs_effective = Fs

    k_t = 2 * math.pi * np.linspace(-Fs_effective / 2, Fs_effective / 2, N)
    k_z = k_t / v
    (kt_mesh_kzkt, kz_mesh_kzkt) = np.meshgrid(k_t, k_z)

    ## Model radii -> (Farina, 2004), Figure 1 - b)
    th_muscle = r - th_fat - th_skin - r_bone
    a = r_bone
    b = r_bone + th_muscle
    c = r_bone + th_muscle + th_fat
    d = r_bone + th_muscle + th_fat + th_skin

    ###################################################################################################
    ## 2. I(k_t, k_z)
    # Rosenfalck (1969) intracellular action potential spatial distribution
    # As used in Farina & Merletti (2001), equation 16
    # IMPORTANT: This matches the original Farina implementation exactly (no normalization)
    A = 96 // 10  # mV/mm^3 (amplitude parameter from Farina 2001, eq 16)

    # Define spatial grid for IAP kernel evaluation
    if fiber_length__mm is None:
        # Legacy behavior: z-range depends on N (INCORRECT - causes MUAP duration to scale with N)
        z = np.linspace(-(N - 1) * (v / Fs) / 2, (N - 1) * (v / Fs) / 2, N, dtype=np.longdouble)
    else:
        # Correct behavior: z-range defined by physical fiber length (independent of N)
        # N controls only the resolution, not the physical extent
        z = np.linspace(-fiber_length__mm / 2, fiber_length__mm / 2, N, dtype=np.longdouble)

    z /= 2  # mm - scaling for Rosenfalck model

    # Rosenfalck action potential: A * exp(-z) * (3z² - z³) for z ≥ 0
    # Uses raw z values in mm (NOT normalized) to match original implementation
    aux = np.zeros_like(z)
    for i in range(len(z)):
        if z[i] >= 0:
            aux[i] = A * np.exp(-z[i]) * (3 * z[i] ** 2 - z[i] ** 3)

    # print(f"DEBUG: z range = [{np.min(z):.2f}, {np.max(z):.2f}] mm")
    # print(f"DEBUG: aux range = [{np.min(aux):.6e}, {np.max(aux):.6e}]")
    # print(f"DEBUG: aux nonzero count = {np.count_nonzero(aux)}/{len(aux)}")

    psi = -f_minus_t(aux)
    # print(f"DEBUG: psi range = [{np.min(psi):.6e}, {np.max(psi):.6e}]")
    PSI = np.fft.fftshift(np.fft.fft(psi)) / len(psi)
    PSI_conj = np.conj(PSI)
    PSI_conj = PSI_conj.reshape(-1, len(PSI_conj))
    ones = np.ones((len(PSI_conj[0, :]), 1))
    PSI_mesh_conj = np.dot(ones, PSI_conj)
    I_kzkt = 1j * np.multiply(
        kz_mesh_kzkt / v, np.multiply(PSI_mesh_conj, np.exp(-1j * kz_mesh_kzkt * zi))
    )
    k_eps = kz_mesh_kzkt + kt_mesh_kzkt / v
    k_beta = kz_mesh_kzkt - kt_mesh_kzkt / v
    aux1 = np.multiply(np.exp(-1j * k_eps * L1 / 2), np.sinc(k_eps * L1 / 2 / math.pi) * L1)
    aux2 = np.multiply(np.exp(1j * k_beta * L2 / 2), np.sinc(k_beta * L2 / 2 / math.pi) * L2)
    I_kzkt = np.multiply(I_kzkt, (aux1 - aux2))
    # print(f"DEBUG: I_kzkt range = [{np.min(np.abs(I_kzkt)):.6e}, {np.max(np.abs(I_kzkt)):.6e}]")

    # Time grid: use the effective Fs (matches grid spacing)
    t = np.linspace(0, (N - 1) / Fs_effective, N)

    ###################################################################################################
    ## 3. H_vc(k_z, k_theta)
    am = a * math.sqrt(sig_muscle_z / sig_muscle_rho)
    bm = b * math.sqrt(sig_muscle_z / sig_muscle_rho)
    Rm = R * math.sqrt(sig_muscle_z / sig_muscle_rho)

    i_start, i_end = int(len(k_z) / 2), len(k_z)
    j_start, j_end = int(len(k_theta) / 2), len(k_theta)

    # Create sub-arrays for positive frequencies
    k_z_pos = k_z[i_start:i_end]
    k_theta_pos = k_theta[j_start:j_end]

    K_THETA, K_Z = np.meshgrid(k_theta_pos, k_z_pos, indexing="ij")

    n_theta, n_z = K_THETA.shape

    A_mat = np.zeros((n_theta, n_z, 7, 7))
    B = np.zeros((n_theta, n_z, 7, 1))

    """ print(
        f"DEBUG: In_Rm_scaled range = [{np.min(In_Rm_scaled):.6e}, {np.max(In_Rm_scaled):.6e}], has inf: {np.any(np.isinf(In_Rm_scaled))}, has nan: {np.any(np.isnan(In_Rm_scaled))}"
    )
    print(
        f"DEBUG: Kn_Rm_scaled range = [{np.min(Kn_Rm_scaled):.6e}, {np.max(Kn_Rm_scaled):.6e}], has inf: {np.any(np.isinf(Kn_Rm_scaled))}, has nan: {np.any(np.isnan(Kn_Rm_scaled))}"
    ) """

    if A_matrix is not None and B_incomplete is not None:
        # print(f"DEBUG: Using CACHED A_matrix and B_incomplete (am={am:.6f}, bm={bm:.6f})")
        A_mat = A_matrix
        B = B_incomplete.copy()
    else:
        # print(f"DEBUG: Computing NEW A_matrix and B_incomplete (am={am:.6f}, bm={bm:.6f})")
        # Compute log-space Bessel functions for all radii to avoid overflow
        # For ratios like In_am/In_bm, we use: exp(log_In(am) - log_In(bm))

        log_In_a = log_In(K_THETA, a * K_Z)
        log_In_am = log_In(K_THETA, am * K_Z)
        log_In_b = log_In(K_THETA, b * K_Z)
        log_In_bm = log_In(K_THETA, bm * K_Z)
        log_In_c = log_In(K_THETA, c * K_Z)
        log_In_d = log_In(K_THETA, d * K_Z)

        log_Kn_am = log_Kn(K_THETA, am * K_Z)
        log_Kn_b = log_Kn(K_THETA, b * K_Z)
        log_Kn_bm = log_Kn(K_THETA, bm * K_Z)
        log_Kn_c = log_Kn(K_THETA, c * K_Z)
        log_Kn_d = log_Kn(K_THETA, d * K_Z)

        # Compute log-space tilde functions: In_tilde = (In(n+1) + In(n-1)) / 2
        log_In_tilde_a = np.logaddexp(
            log_In(K_THETA + 1, a * K_Z), log_In(K_THETA - 1, a * K_Z)
        ) - np.log(2)
        log_In_tilde_am = np.logaddexp(
            log_In(K_THETA + 1, am * K_Z), log_In(K_THETA - 1, am * K_Z)
        ) - np.log(2)
        log_In_tilde_b = np.logaddexp(
            log_In(K_THETA + 1, b * K_Z), log_In(K_THETA - 1, b * K_Z)
        ) - np.log(2)
        log_In_tilde_bm = np.logaddexp(
            log_In(K_THETA + 1, bm * K_Z), log_In(K_THETA - 1, bm * K_Z)
        ) - np.log(2)
        log_In_tilde_c = np.logaddexp(
            log_In(K_THETA + 1, c * K_Z), log_In(K_THETA - 1, c * K_Z)
        ) - np.log(2)
        log_In_tilde_d = np.logaddexp(
            log_In(K_THETA + 1, d * K_Z), log_In(K_THETA - 1, d * K_Z)
        ) - np.log(2)

        log_Kn_tilde_am = np.logaddexp(
            log_Kn(K_THETA + 1, am * K_Z), log_Kn(K_THETA - 1, am * K_Z)
        ) - np.log(2)
        log_Kn_tilde_bm = np.logaddexp(
            log_Kn(K_THETA + 1, bm * K_Z), log_Kn(K_THETA - 1, bm * K_Z)
        ) - np.log(2)
        log_Kn_tilde_c = np.logaddexp(
            log_Kn(K_THETA + 1, c * K_Z), log_Kn(K_THETA - 1, c * K_Z)
        ) - np.log(2)
        log_Kn_tilde_d = np.logaddexp(
            log_Kn(K_THETA + 1, d * K_Z), log_Kn(K_THETA - 1, d * K_Z)
        ) - np.log(2)
        log_Kn_tilde_b = np.logaddexp(
            log_Kn(K_THETA + 1, b * K_Z), log_Kn(K_THETA - 1, b * K_Z)
        ) - np.log(2)

        # Debug: Check K_THETA shape and which values cause underflow
        """  
        print(f"DEBUG: K_THETA shape = {K_THETA.shape}, unique values = {np.unique(K_THETA)}")
        print(
            f"DEBUG: a = {a:.6f}, am = {am:.6f}, b = {b:.6f}, bm = {bm:.6f}, c = {c:.6f}, d = {d:.6f}"
        )
        print(f"DEBUG: sig_muscle_z = {sig_muscle_z:.6f}, sig_muscle_rho = {sig_muscle_rho:.6f}")
        print(f"DEBUG: K_Z range = [{np.min(K_Z):.6e}, {np.max(K_Z):.6e}]")
        print(f"DEBUG: a*K_Z range = [{np.min(a * K_Z):.6e}, {np.max(a * K_Z):.6e}]")
        print(f"DEBUG: am*K_Z range = [{np.min(am * K_Z):.6e}, {np.max(am * K_Z):.6e}]") 
        """

        # Check which (K_THETA, K_Z) pairs cause -inf in log_In_am
        is_neginf = np.isneginf(log_In_am)
        if np.any(is_neginf):
            bad_theta = K_THETA[is_neginf]
            bad_kz = K_Z[is_neginf]
            bad_arg = am * bad_kz
            print(f"DEBUG: log_In_am -inf at {np.sum(is_neginf)} locations")
            print(f"DEBUG:   K_THETA (order n) causing -inf: {np.unique(bad_theta)}")
            print(f"DEBUG:   am*K_Z argument range: [{np.min(bad_arg):.6e}, {np.max(bad_arg):.6e}]")

        # Build the A matrix using LOG-SPACE arithmetic to avoid overflow
        # For ratios: In_x / In_y = exp(log_In_x - log_In_y)
        # For products with scalars: scalar / In_x * In_tilde_y = scalar * exp(log_In_tilde_y - log_In_x)
        # IMPORTANT: When log_In_y = -inf (denominator is zero), the ratio should be inf (which we'll handle)

        # First row
        A_mat[..., 0, 0] = 1
        A_mat[..., 0, 1] = -np.exp(log_In_am - log_In_bm)  # -In_am / In_bm
        A_mat[..., 0, 2] = -np.exp(log_Kn_am - log_Kn_bm)  # -Kn_am / Kn_bm

        # Second row
        A_mat[..., 1, 0] = sig_bone * np.exp(
            log_In_tilde_a - log_In_a
        )  # sig_bone / In_a * In_tilde_a
        A_mat[..., 1, 1] = -math.sqrt(sig_muscle_rho * sig_muscle_z) * np.exp(
            log_In_tilde_am - log_In_bm
        )
        A_mat[..., 1, 2] = math.sqrt(sig_muscle_rho * sig_muscle_z) * np.exp(
            log_Kn_tilde_am - log_Kn_bm
        )  # Note: (-1)*(-1) = +1

        # Third row
        A_mat[..., 2, 1] = 1
        A_mat[..., 2, 2] = 1
        A_mat[..., 2, 3] = -np.exp(log_In_b - log_In_c)  # -In_b / In_c
        A_mat[..., 2, 4] = -np.exp(log_Kn_b - log_Kn_c)  # -Kn_b / Kn_c

        # Fourth row
        A_mat[..., 3, 1] = math.sqrt(sig_muscle_rho * sig_muscle_z) * np.exp(
            log_In_tilde_bm - log_In_bm
        )
        A_mat[..., 3, 2] = -math.sqrt(sig_muscle_rho * sig_muscle_z) * np.exp(
            log_Kn_tilde_bm - log_Kn_bm
        )
        A_mat[..., 3, 3] = -sig_fat * np.exp(
            log_In_tilde_b - log_In_c
        )  # -sig_fat / In_c * In_tilde_b
        A_mat[..., 3, 4] = sig_fat * np.exp(
            log_Kn_tilde_b - log_Kn_c
        )  # -sig_fat / Kn_c * (-1) * Kn_tilde_b = +sig_fat * Kn_tilde_b / Kn_c

        # Fifth row
        A_mat[..., 4, 3] = 1
        A_mat[..., 4, 4] = 1
        A_mat[..., 4, 5] = -np.exp(log_In_c - log_In_d)  # -In_c / In_d
        A_mat[..., 4, 6] = -np.exp(log_Kn_c - log_Kn_d)  # -Kn_c / Kn_d

        # Sixth row
        A_mat[..., 5, 3] = sig_fat * np.exp(
            log_In_tilde_c - log_In_c
        )  # sig_fat / In_c * In_tilde_c
        A_mat[..., 5, 4] = -sig_fat * np.exp(
            log_Kn_tilde_c - log_Kn_c
        )  # sig_fat / Kn_c * (-1) * Kn_tilde_c
        A_mat[..., 5, 5] = -sig_skin * np.exp(
            log_In_tilde_c - log_In_d
        )  # -sig_skin / In_d * In_tilde_c
        A_mat[..., 5, 6] = sig_skin * np.exp(
            log_Kn_tilde_c - log_Kn_d
        )  # -sig_skin / Kn_d * (-1) * Kn_tilde_c = +sig_skin * Kn_tilde_c / Kn_d

        # Seventh row
        # Debug row 6 computation
        log_diff_In_d = log_In_tilde_d - log_In_d
        log_diff_Kn_d = log_Kn_tilde_d - log_Kn_d
        """ 
        print(
            f"DEBUG: Row 6 - log_In_tilde_d range: [{np.min(log_In_tilde_d):.6e}, {np.max(log_In_tilde_d):.6e}]"
        )
        print(f"DEBUG: Row 6 - log_In_d range: [{np.min(log_In_d):.6e}, {np.max(log_In_d):.6e}]")
        print(
            f"DEBUG: Row 6 - log_diff (In) range: [{np.min(log_diff_In_d):.6e}, {np.max(log_diff_In_d):.6e}]"
        )
        print(
            f"DEBUG: Row 6 - exp(log_diff_In) range: [{np.min(np.exp(log_diff_In_d)):.6e}, {np.max(np.exp(log_diff_In_d)):.6e}]"
        ) 
        """

        A_mat[..., 6, 5] = sig_skin * np.exp(log_diff_In_d)  # sig_skin / In_d * In_tilde_d
        A_mat[..., 6, 6] = -sig_skin * np.exp(log_diff_Kn_d)  # sig_skin / Kn_d * (-1) * Kn_tilde_d

        """ 
        print(
            f"DEBUG: A[6,5] range: [{np.min(A_mat[..., 6, 5]):.6e}, {np.max(A_mat[..., 6, 5]):.6e}]"
        )
        print(
            f"DEBUG: A[6,6] range: [{np.min(A_mat[..., 6, 6]):.6e}, {np.max(A_mat[..., 6, 6]):.6e}]"
        ) 
        """

        # Check A matrix for numerical issues
        has_inf = np.isinf(A_mat)
        has_nan = np.isnan(A_mat)
        """ 
        print(
            f"DEBUG: A_mat (log-space) - inf count: {np.sum(has_inf)}, nan count: {np.sum(has_nan)}, total elements: {A_mat.size}"
        ) 
        """

        # Check if inf/nan are only in rows/columns 0-1 (bone-related, which will be discarded)
        # has_inf_23456 = np.isinf(A_mat[..., 2:, 2:])
        # has_nan_23456 = np.isnan(A_mat[..., 2:, 2:])
        """ 
        print(
            f"DEBUG: A_mat[2:,2:] (kept part) - inf count: {np.sum(has_inf_23456)}, nan count: {np.sum(has_nan_23456)}"
        ) 
        """

        if np.any(has_inf) or np.any(has_nan):
            # print("WARNING: A_mat still has inf/nan after log-space reformulation!")
            # Replace only to avoid solver crashes, but this shouldn't happen now
            A_mat[np.isinf(A_mat)] = 0
            A_mat[np.isnan(A_mat)] = 0

        A_matrix = A_mat.copy()

        """ 
        print(
            f"DEBUG: A_mat range (non-zero): [{np.min(np.abs(A_mat[A_mat != 0])):.6e}, {np.max(np.abs(A_mat)):.6e}]"
        ) 
        """

        # B vector is updated later with log-space arithmetic
        B_incomplete = B.copy()

    # Update B vector using LOG-SPACE arithmetic to avoid overflow
    # Compute products as: In*Kn = exp(log(In) + log(Kn))

    # Compute log-space Bessel values
    log_In_am = log_In(K_THETA, am * K_Z)
    log_Kn_Rm = log_Kn(K_THETA, Rm * K_Z)
    log_Kn_bm = log_Kn(K_THETA, bm * K_Z)
    log_In_Rm = log_In(K_THETA, Rm * K_Z)

    # Debug B vector computation
    """  
    print(f"DEBUG: B vector - am={am:.6f}, Rm={Rm:.6f}")
    print(
        f"DEBUG: B vector - log_In_am has -inf: {np.any(np.isneginf(log_In_am))}, count: {np.sum(np.isneginf(log_In_am))}"
    )
    print(
        f"DEBUG: B vector - log_In_Rm has -inf: {np.any(np.isneginf(log_In_Rm))}, count: {np.sum(np.isneginf(log_In_Rm))}"
    )
    if not np.all(np.isneginf(log_In_Rm)):
        print(
            f"DEBUG: B vector - log_In_Rm range (non -inf): [{np.nanmin(log_In_Rm[~np.isneginf(log_In_Rm)]):.6e}, {np.nanmax(log_In_Rm):.6e}]"
        )
    print(
        f"DEBUG: B vector - log_Kn_bm range: [{np.nanmin(log_Kn_bm):.6e}, {np.nanmax(log_Kn_bm):.6e}]"
    ) 
    """

    # For In_tilde, we need log(In(n+1) + In(n-1)) which is trickier
    # Use log-sum-exp trick: log(a+b) = log(a) + log(1 + exp(log(b)-log(a)))
    log_In_p1 = log_In(K_THETA + 1, am * K_Z)
    log_In_m1 = log_In(K_THETA - 1, am * K_Z)
    log_In_tilde_am = np.logaddexp(log_In_p1, log_In_m1) - np.log(2)

    log_Kn_p1 = log_Kn(K_THETA + 1, bm * K_Z)
    log_Kn_m1 = log_Kn(K_THETA - 1, bm * K_Z)
    log_Kn_tilde_bm = np.logaddexp(log_Kn_p1, log_Kn_m1) - np.log(2)

    # Compute products in log space, then exponentiate
    # Use np.where to handle potential overflow: if log value is too large, set to 0
    # This prevents inf but maintains physics better than clipping
    MAX_LOG_SAFE = 700.0  # exp(700) ≈ 1e304 (near float64 max)

    # B[0,0] = In_am * Kn_Rm / sig_muscle_rho
    log_val_00 = log_In_am + log_Kn_Rm
    B[..., 0, 0] = np.where(log_val_00 < MAX_LOG_SAFE, np.exp(log_val_00), 0) / sig_muscle_rho

    # B[1,0] = sqrt(sig_z/sig_rho) * In_tilde_am * Kn_Rm
    log_val_10 = log_In_tilde_am + log_Kn_Rm
    B[..., 1, 0] = math.sqrt(sig_muscle_z / sig_muscle_rho) * np.where(
        log_val_10 < MAX_LOG_SAFE, np.exp(log_val_10), 0
    )

    # B[2,0] = -Kn_bm * In_Rm / sig_muscle_rho
    log_val_20 = log_Kn_bm + log_In_Rm
    B[..., 2, 0] = -np.where(log_val_20 < MAX_LOG_SAFE, np.exp(log_val_20), 0) / sig_muscle_rho

    # B[3,0] = sqrt(sig_z/sig_rho) * Kn_tilde_bm * In_Rm
    log_val_30 = log_Kn_tilde_bm + log_In_Rm
    B[..., 3, 0] = math.sqrt(sig_muscle_z / sig_muscle_rho) * np.where(
        log_val_30 < MAX_LOG_SAFE, np.exp(log_val_30), 0
    )

    """ 
    print(f"DEBUG: B (log-space) has NaN: {np.any(np.isnan(B))}, has inf: {np.any(np.isinf(B))}")
    print(
        f"DEBUG: B (log-space) range: [{np.nanmin(np.abs(B[B != 0])):.6e}, {np.nanmax(np.abs(B)):.6e}]"
    ) 
    """

    # Replace inf/nan in B to avoid solver issues
    B[np.isinf(B)] = 0
    B[np.isnan(B)] = 0

    A_flat = A_mat.reshape(-1, 7, 7)
    B_flat = B.reshape(-1, 7, 1)

    """ 
    print(
        f"DEBUG: Before solve - B_flat range = [{np.min(np.abs(B_flat)):.6e}, {np.max(np.abs(B_flat)):.6e}]"
    )
    print(
        f"DEBUG: Before solve - A_flat range = [{np.min(np.abs(A_flat[A_flat != 0])):.6e}, {np.max(np.abs(A_flat)):.6e}]"
    ) 
    """

    # Solve the linear system
    if r_bone == 0:
        A_flat = A_flat[..., 2:, 2:]
        B_flat = B_flat[..., 2:, :]

        # Check condition number of a few matrices
        cond_numbers = []
        for idx in range(min(10, A_flat.shape[0])):
            cond = np.linalg.cond(A_flat[idx])
            cond_numbers.append(cond)
        # print(f"DEBUG: A_flat (5x5) condition numbers (first 10): {np.array(cond_numbers)}")
        # print(f"DEBUG: A_flat (5x5) mean condition number: {np.mean(cond_numbers):.6e}")
        # print(f"DEBUG: A_flat (5x5) max condition number: {np.max(cond_numbers):.6e}")

        # Check B_flat values - how many are non-zero?
        # b_nonzero_per_row = [np.sum(np.abs(B_flat[:, i, 0]) > 1e-10) for i in range(5)]
        # print(f"DEBUG: B_flat non-zero counts per row: {b_nonzero_per_row}")
        # print(f"DEBUG: B_flat[0] (first matrix):\n{B_flat[0]}")
        # print(f"DEBUG: A_flat[0] (first matrix):\n{A_flat[0]}")

        if HAS_CUPY and use_gpu:
            A_gpu = cp.asarray(A_flat)
            B_gpu = cp.asarray(B_flat)
            X = cp.linalg.solve(A_gpu, B_gpu)
            X = cp.asnumpy(X)
            del A_gpu, B_gpu
        else:
            X = np.linalg.solve(A_flat, B_flat)

        X = X.reshape(n_theta, n_z, 5, 1)
        """
        print(
            f"DEBUG: After solve (r_bone=0) - X range = [{np.min(np.abs(X)):.6e}, {np.max(np.abs(X)):.6e}]"
        )
        """

        # for i in range(5):
        #   print(f"DEBUG: X[{i}] range = [{np.min(X[..., i, 0]):.6e}, {np.max(X[..., i, 0]):.6e}]")
        H_vc = X[..., 3, 0] + X[..., 4, 0]
        """
        print(
            f"DEBUG: After sum (r_bone=0) - H_vc range = [{np.min(np.abs(H_vc)):.6e}, {np.max(np.abs(H_vc)):.6e}]"
        )
        """
    else:
        if HAS_CUPY and use_gpu:
            A_gpu = cp.asarray(A_flat)
            B_gpu = cp.asarray(B_flat)
            X = cp.linalg.solve(A_gpu, B_gpu)
            X = cp.asnumpy(X)
            del A_gpu, B_gpu
        else:
            X = np.linalg.solve(A_flat, B_flat)

        X = X.reshape(n_theta, n_z, 7, 1)
        H_vc = X[..., 5, 0] + X[..., 6, 0]

    # Reconstruct full H_vc using symmetry
    temp = np.zeros((len(k_z), len(k_theta)))
    temp[i_start:i_end, j_start:j_end] = H_vc.T
    H_vc = temp

    H_vc_pos_section = H_vc[i_start:i_end, j_start:j_end]
    H_vc[i_start:i_end, :j_start] = np.fliplr(H_vc_pos_section)
    H_vc[:i_start, j_start:j_end] = np.flipud(H_vc_pos_section)
    H_vc[:i_start, :j_start] = np.flipud(np.fliplr(H_vc_pos_section))

    ###################################################################################################
    ## 4. H_ele(k_z, k_theta)

    # Use the SurfaceElectrodeArray's spatial filtering capabilities
    ktheta_mesh_kzktheta, kz_mesh_kzktheta = np.meshgrid(k_theta, k_z)

    # Get spatial filter from electrode array
    H_sf = electrode_array.get_H_sf(ktheta_mesh_kzktheta, kz_mesh_kzktheta)

    # Electrode size effect
    arg = np.sqrt((rele * ktheta_mesh_kzktheta / r) ** 2 + (rele * kz_mesh_kzktheta) ** 2)
    H_size = 2 * np.divide(Jn(1, arg), arg)
    auxxx = np.ones(H_size.shape)
    H_size[np.isnan(H_size)] = auxxx[np.isnan(H_size)]

    # Combined electrode response
    H_ele = np.multiply(H_sf, H_size)

    # print(f"DEBUG: H_vc range = [{np.min(np.abs(H_vc)):.6e}, {np.max(np.abs(H_vc)):.6e}]")
    # print(f"DEBUG: H_ele range = [{np.min(np.abs(H_ele)):.6e}, {np.max(np.abs(H_ele)):.6e}]")

    ###################################################################################################
    ## 5. H_glo(k_z, k_theta) - Use electrode array's positions

    # Use the electrode array's pre-computed positions
    H_glo = np.multiply(H_vc, H_ele)
    # print(f"DEBUG: H_glo range = [{np.min(np.abs(H_glo)):.6e}, {np.max(np.abs(H_glo)):.6e}]")
    B_kz = np.zeros((channels[0], channels[1], len(k_z)))

    for channel_z in range(channels[0]):
        for channel_theta in range(channels[1]):
            arg = np.multiply(
                H_glo,
                np.exp(1j * pos_theta_rad[channel_z, channel_theta] * ktheta_mesh_kzktheta)
                * (k_theta[1] - k_theta[0]),
            )
            B_kz[channel_z, channel_theta, :] = sum(np.transpose(arg)) / 2 / math.pi

    # print(f"DEBUG: B_kz range = [{np.min(np.abs(B_kz)):.6e}, {np.max(np.abs(B_kz)):.6e}]")

    ###################################################################################################
    ## 6. phi(t) for each channel

    phi = np.zeros((channels[0], channels[1], len(t)))
    for channel_z in range(channels[0]):
        for channel_theta in range(channels[1]):
            auxiliar = np.dot(
                np.ones((len(I_kzkt[1, :]), 1)),
                B_kz[channel_z, channel_theta, :].reshape(1, -1),
            )
            auxiliar = np.transpose(auxiliar)
            arg = np.multiply(I_kzkt, auxiliar)
            arg2 = np.multiply(
                arg,
                np.exp(1j * pos_z_mm[channel_z, channel_theta] * kz_mesh_kzkt) * (k_z[1] - k_z[0]),
            )
            PHI = sum(arg2)
            phi[channel_z, channel_theta, :] = np.real(
                (
                    np.fft.ifft(
                        np.fft.fftshift(PHI / 2 / math.pi * len(psi))
                    )  # Matches original line 239
                )
            )

    # Center the MUAP signal in the time window by finding the peak and shifting
    # For each electrode channel, find the peak and center it
    N_time = phi.shape[2]
    center_idx = N_time // 2

    # Find the peak location (use center electrode as reference)
    center_electrode_row = phi.shape[0] // 2
    center_electrode_col = phi.shape[1] // 2
    center_signal = phi[center_electrode_row, center_electrode_col, :]

    # Find peak (max absolute value)
    peak_idx = np.argmax(np.abs(center_signal))

    # Calculate shift needed to center the peak
    shift = center_idx - peak_idx

    # Apply circular shift to all channels
    phi = np.roll(phi, shift, axis=2)

    # print(f"DEBUG: Final phi range = [{np.min(phi):.6e}, {np.max(phi):.6e}]")
    # print(f"DEBUG: Final phi peak-to-peak = {np.ptp(phi):.6e}")

    return phi, A_matrix, B_incomplete


@beartowertype
def simulate_fiber_v2(
    Fs: float,
    v: float,
    N: int,
    M: int,
    r: float,
    r_bone: float,
    th_fat: float,
    th_skin: float,
    R: float,
    L1: float,
    L2: float,
    zi: float,
    electrode_array: SurfaceElectrodeArray,
    sig_muscle_rho: float,
    sig_muscle_z: float,
    sig_fat: float,
    sig_skin: float,
    sig_bone: float = 0,
    A_matrix: np.ndarray | None = None,
    B_incomplete: np.ndarray | None = None,
    use_cython: bool = True,
    use_gpu: bool = True,
    fiber_length__mm: float | None = None,
):
    """
    Simulate a single fiber (dispatcher to Cython or Python implementation).

    This function automatically dispatches to the Cython-optimized implementation
    if available, otherwise falls back to the Python version.

    Parameters
    ----------
    Fs : float
        Sampling frequency in kHz.
    v : float
        Conduction velocity in m/s.
    N : int
        Number of points in t and z domains (controls resolution).
    M : int
        Number of points in theta domain.
    r : float
        Total radius of the muscle model in mm.
    r_bone : float
        Bone radius in mm.
    th_fat : float
        Fat thickness in mm.
    th_skin : float
        Skin thickness in mm.
    R : float
        Source position in rho coordinate (mm).
    L1 : float
        Semifiber length (z > 0) in mm.
    L2 : float
        Semifiber length (z < 0) in mm.
    zi : float
        Innervation zone position in mm.
    electrode_array : SurfaceElectrodeArray
        Electrode array configuration object.
    sig_muscle_rho : float
        Muscle conductivity in rho direction (S/m).
    sig_muscle_z : float
        Muscle conductivity in z direction (S/m).
    sig_fat : float
        Fat conductivity (S/m).
    sig_skin : float
        Skin conductivity (S/m).
    sig_bone : float, optional
        Bone conductivity (S/m), default=0.
    A_matrix : np.ndarray, optional
        Pre-computed A matrix for optimization.
    B_incomplete : np.ndarray, optional
        Pre-computed B matrix for optimization.
    use_cython : bool, optional
        If True (default), use optimized Cython CPU implementation.
        If False, use Python implementation. Default is True (recommended).
    use_gpu : bool, optional
        If True (default), Python version may use GPU (CuPy) if available.
        Cython version always uses CPU. For typical EMG simulations, CPU is faster
        due to GPU memory transfer overhead. Default is True.
    fiber_length__mm : float, optional
        Physical fiber length for IAP kernel evaluation in mm. This defines the spatial
        extent over which the intracellular action potential kernel is computed.
        If None (default), uses legacy behavior: (N-1) * (v/Fs) mm, which incorrectly
        ties MUAP duration to sampling resolution.

        **Recommended**: Set this explicitly (e.g., 50-100 mm for typical motor units)
        to ensure MUAP duration is physically accurate and independent of N.

        The actual MUAP duration will be approximately: fiber_length__mm / (2 * v) ms
        (factor of 2 due to internal scaling for Rosenfalck model).

        Example: fiber_length__mm=100 with v=4 m/s → ~12.5 ms MUAP duration.

        **Note**: Currently only supported in Python implementation. Cython version
        does not yet support this parameter and will use legacy behavior.

    Returns
    -------
    phi : np.ndarray
        Generated surface EMG signal for each electrode [num_rows, num_cols, num_timepoints].
    A_matrix : np.ndarray
        A matrix for reuse in subsequent calls [n_theta, n_z, 7, 7].
    B_incomplete : np.ndarray
        B matrix for reuse in subsequent calls [n_theta, n_z, 7, 1].

    Notes
    -----
    **Performance (Cython CPU vs Python+GPU):**

    The Cython implementation outperforms GPU acceleration for typical EMG simulations:
    - Small problems (64×16): **20x faster** than GPU (avoids memory transfer overhead)
    - Medium problems (128×32): **1.5x faster** than GPU
    - Large problems (256×64): Comparable to GPU
    - Average across sizes: **7.5x faster** than GPU by eliminating memory transfer overhead

    **Why CPU beats GPU:**

    - No GPU memory transfers (eliminates 40% overhead)
    - No kernel launch overhead
    - Direct NumPy integration
    - Optimized compiled C code
    - OpenMP parallelization for multi-core CPUs

    **Recommendation:** Use default settings (use_cython=True) for best performance.
    """
    if use_cython and HAS_CYTHON_FIBER:
        # Cython version doesn't support fiber_length__mm yet
        if fiber_length__mm is not None:
            import warnings

            warnings.warn(
                "fiber_length__mm parameter is not supported by the Cython implementation. "
                "Falling back to Python implementation. To use Cython, either set "
                "fiber_length__mm=None (legacy behavior) or set use_cython=False.",
                UserWarning,
                stacklevel=2,
            )
            # Fall through to Python implementation
        else:
            assert _simulate_fiber_v2_cython is not None  # Type checker hint
            return _simulate_fiber_v2_cython(
                Fs,
                v,
                N,
                M,
                r,
                r_bone,
                th_fat,
                th_skin,
                R,
                L1,
                L2,
                zi,
                electrode_array,
                sig_muscle_rho,
                sig_muscle_z,
                sig_fat,
                sig_skin,
                sig_bone,
                A_matrix,
                B_incomplete,
            )

    # Use Python implementation (either explicitly requested or Cython unavailable/incompatible)
    return _simulate_fiber_v2_python(
        Fs,
        v,
        N,
        M,
        r,
        r_bone,
        th_fat,
        th_skin,
        R,
        L1,
        L2,
        zi,
        electrode_array,
        sig_muscle_rho,
        sig_muscle_z,
        sig_fat,
        sig_skin,
        sig_bone,
        A_matrix,
        B_incomplete,
        use_gpu,
        fiber_length__mm,
    )
