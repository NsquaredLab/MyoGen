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
# electrode_grid_center         -> Matrix of electrodes electrode_grid_center (tuple -> (z electrode_grid_center in mm, theta electrode_grid_center in degrees))
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
from scipy.special import jv as Jn
from scipy.special import kv as Kn

from myogen.utils.types import beartowertype

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


def In_tilde(K_THETA, x):
    return (In(K_THETA + 1, x) + In(K_THETA - 1, x)) / 2


def Kn_tilde(K_THETA, x):
    return (Kn(K_THETA + 1, x) + Kn(K_THETA - 1, x)) / 2


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
                        result_real = (
                            H_glo_real[i, j] * cos_phase - H_glo_imag[i, j] * sin_phase
                        )
                        sum_real += result_real

                    B_kz[channel_z, channel_theta, i] = (
                        sum_real * k_theta_diff / (2 * math.pi)
                    )

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
):
    """
    Simulate a single fiber.

    Parameters
    ----------
    Fs : float
        Sampling frequency in kHz.
    v : float
        Conduction velocity in m/s.
    N : int
        Number of points in t and z domains.
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
    rele = electrode_array.electrode_radius__mm

    ###################################################################################################
    ## 1. Constants

    ## Model angular frequencies
    k_theta = np.linspace(-(M - 1) / 2, (M - 1) / 2, M)
    k_t = 2 * math.pi * np.linspace(-Fs / 2, Fs / 2, N)
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
    A = 96  ## mV/mm^3 -> (Farina, 2001), eq (16)
    z = np.linspace(
        -(N - 1) * (v / Fs) / 2, (N - 1) * (v / Fs) / 2, N, dtype=np.longdouble
    )

    aux = np.zeros_like(z)
    positive_mask = z >= 0
    aux[positive_mask] = (
        A
        * np.exp(-z[positive_mask])
        * (3 * z[positive_mask] ** 2 - z[positive_mask] ** 3)
    )

    psi = -f_minus_t(aux)
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
    aux1 = np.multiply(
        np.exp(-1j * k_eps * L1 / 2), np.sinc(k_eps * L1 / 2 / math.pi) * L1
    )
    aux2 = np.multiply(
        np.exp(1j * k_beta * L2 / 2), np.sinc(k_beta * L2 / 2 / math.pi) * L2
    )
    I_kzkt = np.multiply(I_kzkt, (aux1 - aux2))
    t = np.linspace(0, (N - 1) / Fs, N)

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

    Kn_Rm = Kn(K_THETA, Rm * K_Z)
    In_Rm = In(K_THETA, Rm * K_Z)

    if A_matrix is not None and B_incomplete is not None:
        A_mat = A_matrix
        B = B_incomplete.copy()
    else:
        # Compute all the modified Bessel functions
        In_a = In(K_THETA, a * K_Z)
        In_bm = In(K_THETA, bm * K_Z)
        In_c = In(K_THETA, c * K_Z)
        In_d = In(K_THETA, d * K_Z)
        In_b = In(K_THETA, b * K_Z)
        In_am = In(K_THETA, am * K_Z)
        Kn_am = Kn(K_THETA, am * K_Z)
        In_tilde_am = In_tilde(K_THETA, am * K_Z)
        Kn_c = Kn(K_THETA, c * K_Z)
        Kn_d = Kn(K_THETA, d * K_Z)
        Kn_b = Kn(K_THETA, b * K_Z)
        In_tilde_a = In_tilde(K_THETA, a * K_Z)
        Kn_bm = Kn(K_THETA, bm * K_Z)
        In_tilde_bm = In_tilde(K_THETA, bm * K_Z)
        In_tilde_c = In_tilde(K_THETA, c * K_Z)
        In_tilde_d = In_tilde(K_THETA, d * K_Z)
        In_tilde_b = In_tilde(K_THETA, b * K_Z)
        Kn_tilde_am = Kn_tilde(K_THETA, am * K_Z)
        Kn_tilde_bm = Kn_tilde(K_THETA, bm * K_Z)
        Kn_tilde_c = Kn_tilde(K_THETA, c * K_Z)
        Kn_tilde_d = Kn_tilde(K_THETA, d * K_Z)
        Kn_tilde_b = Kn_tilde(K_THETA, b * K_Z)

        # Build the A matrix
        A_mat[..., 0, 0] = 1
        A_mat[..., 0, 1] = -In_am / In_bm
        A_mat[..., 0, 2] = -Kn_am / Kn_bm
        # Second row
        A_mat[..., 1, 0] = sig_bone / In_a * In_tilde_a
        A_mat[..., 1, 1] = (
            -math.sqrt(sig_muscle_rho * sig_muscle_z) / In_bm * In_tilde_am
        )
        A_mat[..., 1, 2] = (
            -math.sqrt(sig_muscle_rho * sig_muscle_z) / Kn_bm * (-1) * Kn_tilde_am
        )
        # Third row
        A_mat[..., 2, 1] = 1
        A_mat[..., 2, 2] = 1
        A_mat[..., 2, 3] = -In_b / In_c
        A_mat[..., 2, 4] = -Kn_b / Kn_c
        # Fourth row
        A_mat[..., 3, 1] = (
            math.sqrt(sig_muscle_rho * sig_muscle_z) / In_bm * In_tilde_bm
        )
        A_mat[..., 3, 2] = (
            math.sqrt(sig_muscle_rho * sig_muscle_z) / Kn_bm * (-1) * Kn_tilde_bm
        )
        A_mat[..., 3, 3] = -sig_fat / In_c * In_tilde_b
        A_mat[..., 3, 4] = -sig_fat / Kn_c * (-1) * Kn_tilde_b
        # Fifth row
        A_mat[..., 4, 3] = 1
        A_mat[..., 4, 4] = 1
        A_mat[..., 4, 5] = -In_c / In_d
        A_mat[..., 4, 6] = -Kn_c / Kn_d
        # Sixth row
        A_mat[..., 5, 3] = sig_fat / In_c * In_tilde_c
        A_mat[..., 5, 4] = sig_fat / Kn_c * (-1) * Kn_tilde_c
        A_mat[..., 5, 5] = -sig_skin / In_d * In_tilde_c
        A_mat[..., 5, 6] = -sig_skin / Kn_d * (-1) * Kn_tilde_c
        # Seventh row
        A_mat[..., 6, 5] = sig_skin / In_d * In_tilde_d
        A_mat[..., 6, 6] = sig_skin / Kn_d * (-1) * Kn_tilde_d

        A_matrix = A_mat.copy()

        # Build the B vector
        B[..., 0, 0] = In_am / sig_muscle_rho
        B[..., 1, 0] = math.sqrt(sig_muscle_z / sig_muscle_rho) * In_tilde_am
        B[..., 2, 0] = Kn_bm / sig_muscle_rho
        B[..., 3, 0] = -math.sqrt(sig_muscle_z / sig_muscle_rho) * (-1) * Kn_tilde_bm

        B_incomplete = B.copy()

    # Update B vector with fiber-specific terms
    B[..., 0, 0] *= Kn_Rm
    B[..., 1, 0] *= Kn_Rm
    B[..., 2, 0] *= -In_Rm
    B[..., 3, 0] *= In_Rm

    A_flat = A_mat.reshape(-1, 7, 7)
    B_flat = B.reshape(-1, 7, 1)

    # Solve the linear system
    if r_bone == 0:
        A_flat = A_flat[..., 2:, 2:]
        B_flat = B_flat[..., 2:, :]

        if HAS_CUPY:
            A_gpu = cp.asarray(A_flat)
            B_gpu = cp.asarray(B_flat)
            X = cp.linalg.solve(A_gpu, B_gpu)
            X = cp.asnumpy(X)
            del A_gpu, B_gpu
        else:
            X = np.linalg.solve(A_flat, B_flat)

        X = X.reshape(n_theta, n_z, 5, 1)
        H_vc = X[..., 3, 0] + X[..., 4, 0]
    else:
        if HAS_CUPY:
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
    arg = np.sqrt(
        (rele * ktheta_mesh_kzktheta / r) ** 2 + (rele * kz_mesh_kzktheta) ** 2
    )
    H_size = 2 * np.divide(Jn(1, arg), arg)
    auxxx = np.ones(H_size.shape)
    H_size[np.isnan(H_size)] = auxxx[np.isnan(H_size)]

    # Combined electrode response
    H_ele = np.multiply(H_sf, H_size)

    ###################################################################################################
    ## 5. H_glo(k_z, k_theta) - Use electrode array's positions

    # Use the electrode array's pre-computed positions
    H_glo = np.multiply(H_vc, H_ele)
    B_kz = np.zeros((channels[0], channels[1], len(k_z)))

    for channel_z in range(channels[0]):
        for channel_theta in range(channels[1]):
            arg = np.multiply(
                H_glo,
                np.exp(
                    1j
                    * electrode_array.pos_theta[channel_z, channel_theta]
                    * ktheta_mesh_kzktheta
                )
                * (k_theta[1] - k_theta[0]),
            )
            B_kz[channel_z, channel_theta, :] = sum(np.transpose(arg)) / 2 / math.pi

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
                np.exp(
                    1j * electrode_array.pos_z[channel_z, channel_theta] * kz_mesh_kzkt
                )
                * (k_z[1] - k_z[0]),
            )
            PHI = sum(arg2)
            phi[channel_z, channel_theta, :] = np.real(
                (np.fft.ifft(np.fft.fftshift(PHI / 2 / math.pi * len(psi))))
            )

    return phi, A_matrix, B_incomplete


@beartowertype
def simulate_fiber(
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
    use_numba: bool = True,
):
    """
    High-performance simulate fiber function using Numba JIT compilation.

    This is an optimized version of simulate_fiber_v2 that uses Numba for JIT compilation
    of the most computationally intensive parts. It provides significant speed improvements
    for large electrode arrays and multiple fiber simulations.

    Parameters
    ----------
    Fs : float
        Sampling frequency in kHz
    v : float
        Conduction velocity in m/s
    N : int
        Number of points in t and z domains
    M : int
        Number of points in theta domain
    r : float
        Total radius of the muscle model in mm
    r_bone : float
        Bone radius in mm
    th_fat : float
        Fat thickness in mm
    th_skin : float
        Skin thickness in mm
    R : float
        Source position in rho coordinate (mm)
    L1 : float
        Semifiber length (z > 0) in mm
    L2 : float
        Semifiber length (z < 0) in mm
    zi : float
        Innervation zone position in mm
    electrode_array : SurfaceElectrodeArray
        Electrode array configuration object
    sig_muscle_rho : float
        Muscle conductivity in rho direction (S/m)
    sig_muscle_z : float
        Muscle conductivity in z direction (S/m)
    sig_fat : float
        Fat conductivity (S/m)
    sig_skin : float
        Skin conductivity (S/m)
    sig_bone : float, optional
        Bone conductivity (S/m), default=0
    A_matrix : np.ndarray, optional
        Pre-computed A matrix for optimization
    B_incomplete : np.ndarray, optional
        Pre-computed B matrix for optimization
    use_numba : bool, optional
        Whether to use Numba acceleration, default=True

    Returns
    -------
    phi : np.ndarray
        Generated surface EMG signal for each electrode
    A_matrix : np.ndarray
        A matrix for reuse in subsequent calls
    B_incomplete : np.ndarray
        B matrix for reuse in subsequent calls

    Notes
    -----
    This function requires numba to be installed for optimal performance. If numba is not
    available and use_numba=True, it will fall back to the regular implementation.
    For maximum performance, install numba: `pip install numba`
    """

    # Check if Numba acceleration is available and requested
    if use_numba and not HAS_NUMBA:
        print("Warning: Numba not available. Falling back to simulate_fiber_v2.")
        return simulate_fiber_v2(
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

    if not use_numba:
        # Use regular implementation if Numba is disabled
        return simulate_fiber_v2(
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

    # Get electrode configuration from the array
    channels = [electrode_array.num_rows, electrode_array.num_cols]
    rele = electrode_array.electrode_radius__mm

    ###################################################################################################
    ## 1. Constants (same as v2)
    k_theta = np.linspace(-(M - 1) / 2, (M - 1) / 2, M)
    k_t = 2 * math.pi * np.linspace(-Fs / 2, Fs / 2, N)
    k_z = k_t / v
    (kt_mesh_kzkt, kz_mesh_kzkt) = np.meshgrid(k_t, k_z)

    ## Model radii
    th_muscle = r - th_fat - th_skin - r_bone
    a = r_bone
    b = r_bone + th_muscle
    c = r_bone + th_muscle + th_fat
    d = r_bone + th_muscle + th_fat + th_skin

    ###################################################################################################
    ## 2. I(k_t, k_z) (same as v2)
    A = 96  ## mV/mm^3
    z = np.linspace(
        -(N - 1) * (v / Fs) / 2, (N - 1) * (v / Fs) / 2, N, dtype=np.longdouble
    )

    aux = np.zeros_like(z)
    positive_mask = z >= 0
    aux[positive_mask] = (
        A
        * np.exp(-z[positive_mask])
        * (3 * z[positive_mask] ** 2 - z[positive_mask] ** 3)
    )

    psi = -f_minus_t(aux)
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
    aux1 = np.multiply(
        np.exp(-1j * k_eps * L1 / 2), np.sinc(k_eps * L1 / 2 / math.pi) * L1
    )
    aux2 = np.multiply(
        np.exp(1j * k_beta * L2 / 2), np.sinc(k_beta * L2 / 2 / math.pi) * L2
    )
    I_kzkt = np.multiply(I_kzkt, (aux1 - aux2))
    t = np.linspace(0, (N - 1) / Fs, N)

    ###################################################################################################
    ## 3. H_vc(k_z, k_theta) (optimized but same logic as v2)
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

    Kn_Rm = Kn(K_THETA, Rm * K_Z)
    In_Rm = In(K_THETA, Rm * K_Z)

    if A_matrix is not None and B_incomplete is not None:
        A_mat = A_matrix
        B = B_incomplete.copy()
    else:
        # Compute Bessel functions (same as v2)
        In_a = In(K_THETA, a * K_Z)
        In_bm = In(K_THETA, bm * K_Z)
        In_c = In(K_THETA, c * K_Z)
        In_d = In(K_THETA, d * K_Z)
        In_b = In(K_THETA, b * K_Z)
        In_am = In(K_THETA, am * K_Z)
        Kn_am = Kn(K_THETA, am * K_Z)
        In_tilde_am = In_tilde(K_THETA, am * K_Z)
        Kn_c = Kn(K_THETA, c * K_Z)
        Kn_d = Kn(K_THETA, d * K_Z)
        Kn_b = Kn(K_THETA, b * K_Z)
        In_tilde_a = In_tilde(K_THETA, a * K_Z)
        Kn_bm = Kn(K_THETA, bm * K_Z)
        In_tilde_bm = In_tilde(K_THETA, bm * K_Z)
        In_tilde_c = In_tilde(K_THETA, c * K_Z)
        In_tilde_d = In_tilde(K_THETA, d * K_Z)
        In_tilde_b = In_tilde(K_THETA, b * K_Z)
        Kn_tilde_am = Kn_tilde(K_THETA, am * K_Z)
        Kn_tilde_bm = Kn_tilde(K_THETA, bm * K_Z)
        Kn_tilde_c = Kn_tilde(K_THETA, c * K_Z)
        Kn_tilde_d = Kn_tilde(K_THETA, d * K_Z)
        Kn_tilde_b = Kn_tilde(K_THETA, b * K_Z)

        # Build the A matrix (same as v2)
        A_mat[..., 0, 0] = 1
        A_mat[..., 0, 1] = -In_am / In_bm
        A_mat[..., 0, 2] = -Kn_am / Kn_bm
        A_mat[..., 1, 0] = sig_bone / In_a * In_tilde_a
        A_mat[..., 1, 1] = (
            -math.sqrt(sig_muscle_rho * sig_muscle_z) / In_bm * In_tilde_am
        )
        A_mat[..., 1, 2] = (
            -math.sqrt(sig_muscle_rho * sig_muscle_z) / Kn_bm * (-1) * Kn_tilde_am
        )
        A_mat[..., 2, 1] = 1
        A_mat[..., 2, 2] = 1
        A_mat[..., 2, 3] = -In_b / In_c
        A_mat[..., 2, 4] = -Kn_b / Kn_c
        A_mat[..., 3, 1] = (
            math.sqrt(sig_muscle_rho * sig_muscle_z) / In_bm * In_tilde_bm
        )
        A_mat[..., 3, 2] = (
            math.sqrt(sig_muscle_rho * sig_muscle_z) / Kn_bm * (-1) * Kn_tilde_bm
        )
        A_mat[..., 3, 3] = -sig_fat / In_c * In_tilde_b
        A_mat[..., 3, 4] = -sig_fat / Kn_c * (-1) * Kn_tilde_b
        A_mat[..., 4, 3] = 1
        A_mat[..., 4, 4] = 1
        A_mat[..., 4, 5] = -In_c / In_d
        A_mat[..., 4, 6] = -Kn_c / Kn_d
        A_mat[..., 5, 3] = sig_fat / In_c * In_tilde_c
        A_mat[..., 5, 4] = sig_fat / Kn_c * (-1) * Kn_tilde_c
        A_mat[..., 5, 5] = -sig_skin / In_d * In_tilde_c
        A_mat[..., 5, 6] = -sig_skin / Kn_d * (-1) * Kn_tilde_c
        A_mat[..., 6, 5] = sig_skin / In_d * In_tilde_d
        A_mat[..., 6, 6] = sig_skin / Kn_d * (-1) * Kn_tilde_d

        A_matrix = A_mat.copy()

        # Build the B vector
        B[..., 0, 0] = In_am / sig_muscle_rho
        B[..., 1, 0] = math.sqrt(sig_muscle_z / sig_muscle_rho) * In_tilde_am
        B[..., 2, 0] = Kn_bm / sig_muscle_rho
        B[..., 3, 0] = -math.sqrt(sig_muscle_z / sig_muscle_rho) * (-1) * Kn_tilde_bm

        B_incomplete = B.copy()

    # Update B vector with fiber-specific terms
    B[..., 0, 0] *= Kn_Rm
    B[..., 1, 0] *= Kn_Rm
    B[..., 2, 0] *= -In_Rm
    B[..., 3, 0] *= In_Rm

    A_flat = A_mat.reshape(-1, 7, 7)
    B_flat = B.reshape(-1, 7, 1)

    # Solve the linear system (same as v2 but with potential GPU acceleration)
    if r_bone == 0:
        A_flat = A_flat[..., 2:, 2:]
        B_flat = B_flat[..., 2:, :]

        if HAS_CUPY:
            A_gpu = cp.asarray(A_flat)
            B_gpu = cp.asarray(B_flat)
            X = cp.linalg.solve(A_gpu, B_gpu)
            X = cp.asnumpy(X)
            del A_gpu, B_gpu
        else:
            X = np.linalg.solve(A_flat, B_flat)

        X = X.reshape(n_theta, n_z, 5, 1)
        H_vc = X[..., 3, 0] + X[..., 4, 0]
    else:
        if HAS_CUPY:
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
    ## 4. H_ele(k_z, k_theta) (same as v2)
    ktheta_mesh_kzktheta, kz_mesh_kzktheta = np.meshgrid(k_theta, k_z)

    # Get spatial filter from electrode array
    H_sf = electrode_array.get_H_sf(ktheta_mesh_kzktheta, kz_mesh_kzktheta)

    # Electrode size effect
    arg = np.sqrt(
        (rele * ktheta_mesh_kzktheta / r) ** 2 + (rele * kz_mesh_kzktheta) ** 2
    )
    H_size = 2 * np.divide(Jn(1, arg), arg)
    auxxx = np.ones(H_size.shape)
    H_size[np.isnan(H_size)] = auxxx[np.isnan(H_size)]

    # Combined electrode response
    H_ele = np.multiply(H_sf, H_size)

    ###################################################################################################
    ## 5. H_glo(k_z, k_theta) - OPTIMIZED WITH NUMBA
    H_glo = np.multiply(H_vc, H_ele)

    # Use Numba-optimized computation if available
    if _numba_B_kz_func is not None:
        # Prepare data for Numba function
        H_glo_real = np.real(H_glo)
        H_glo_imag = np.imag(H_glo)
        k_theta_diff = k_theta[1] - k_theta[0]

        B_kz = _numba_B_kz_func(
            H_glo_real,
            H_glo_imag,
            electrode_array.pos_theta,
            k_theta_diff,
            ktheta_mesh_kzktheta,
            channels[0],
            channels[1],
            len(k_z),
            len(k_theta),
        )
    else:
        # Fallback to regular computation
        B_kz = np.zeros((channels[0], channels[1], len(k_z)))
        for channel_z in range(channels[0]):
            for channel_theta in range(channels[1]):
                arg = np.multiply(
                    H_glo,
                    np.exp(
                        1j
                        * electrode_array.pos_theta[channel_z, channel_theta]
                        * ktheta_mesh_kzktheta
                    )
                    * (k_theta[1] - k_theta[0]),
                )
                B_kz[channel_z, channel_theta, :] = sum(np.transpose(arg)) / 2 / math.pi

    ###################################################################################################
    ## 6. phi(t) for each channel - OPTIMIZED WITH NUMBA

    if _numba_phi_func is not None:
        # Use Numba-optimized computation
        I_kzkt_real = np.real(I_kzkt)
        I_kzkt_imag = np.imag(I_kzkt)
        k_z_diff = k_z[1] - k_z[0]

        PHI_complex = _numba_phi_func(
            I_kzkt_real,
            I_kzkt_imag,
            B_kz,
            electrode_array.pos_z,
            kz_mesh_kzkt,
            channels[0],
            channels[1],
            len(k_z),
            len(k_t),
            k_z_diff,
        )

        # Apply IFFT to get time domain signals
        phi = np.zeros((channels[0], channels[1], len(t)))
        for channel_z in range(channels[0]):
            for channel_theta in range(channels[1]):
                phi[channel_z, channel_theta, :] = np.real(
                    np.fft.ifft(
                        np.fft.fftshift(
                            PHI_complex[channel_z, channel_theta, :] / len(psi)
                        )
                    )
                )
    else:
        # Fallback to regular computation
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
                    np.exp(
                        1j
                        * electrode_array.pos_z[channel_z, channel_theta]
                        * kz_mesh_kzkt
                    )
                    * (k_z[1] - k_z[0]),
                )
                PHI = sum(arg2)
                phi[channel_z, channel_theta, :] = np.real(
                    (np.fft.ifft(np.fft.fftshift(PHI / 2 / math.pi * len(psi))))
                )

    return phi, A_matrix, B_incomplete


#######################################################################################################
###################################### References #####################################################
#######################################################################################################

# FARINA, D.; MERLETTI, R. A novel approach for precise simulation of the EMG signal detected by surface electrodes.
# IEEE Transactions on Biomedical Engineering, mean_conduction_velocity__mm_s. 48, n. 6, p. 637–646, 2001. DOI: 10.1109/10.923782.

# FARINA, D.; MESIN, L.; MARTINA, S.; MERLETTI, R. A Surface EMG Generation Model With Multilayer Cylindrical
# Description of the Volume Conductor. IEEE Transactions on Biomedical Engineering, mean_conduction_velocity__mm_s. 51, n. 3, p. 415–426,
# 2004. DOI: 10.1109/TBME.2003.820998.
