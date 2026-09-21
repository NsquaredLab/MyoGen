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

from myogen.simulator.core.emg.surface.volume_conductor import (  # noqa: F401 (re-exported)
    HAS_CUPY,
    layered_cylinder_transfer,
    log_In,
    log_Kn,
)
from myogen.utils.decorators import beartowertype

try:
    import numba
    from numba import njit, prange

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

from myogen.simulator.core.emg.electrodes import SurfaceElectrodeArray

# The Cython transcription of this module (``_cython/_simulate_fiber.pyx``)
# was removed in 0.12: it had drifted from the Python reference (its own IAP
# grid and amplitude, real-valued angular sum, raw-Bessel assembly) and was
# ~5x slower than the Numba-jitted Python path.  ``HAS_CYTHON_FIBER`` is kept
# for importers that tested it; it is always False.
HAS_CYTHON_FIBER = False


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


def f_minus_t(y):
    # Spatial reflection psi(z) -> psi(-z) on the SYMMETRIC grid
    # z = linspace(-L/2, L/2, N): the sample at z[i] maps to z[N-1-i] = -z[i],
    # i.e. a full array reversal. The previous `y[-i]` was a one-sample-off bug
    # (for i=0 it returned y[0] instead of y[-1], keeping z=-L/2 fixed), which
    # shifted the SFAP kernel by one sample.
    return y[::-1].copy()


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


# NOTE: beartype disabled for performance — this is called 100s-1000s of times per MU
# Type validation happens at the simulate_fiber_v2 dispatcher level instead
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
    theta_offset: float = 0.0,
    pos_z_precomputed: np.ndarray | None = None,
    pos_theta_precomputed: np.ndarray | None = None,
    rele_precomputed: float | None = None,
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

    # Use pre-computed electrode positions if available (avoids per-fiber grid recomputation)
    import quantities as pq
    if pos_z_precomputed is not None and pos_theta_precomputed is not None:
        rele = rele_precomputed if rele_precomputed is not None else float(electrode_array.electrode_radius__mm.rescale(pq.mm).magnitude)
        pos_z_mm = pos_z_precomputed
        pos_theta_rad = pos_theta_precomputed + theta_offset  # Apply angular offset as simple addition
    else:
        rele = float(electrode_array.electrode_radius__mm.rescale(pq.mm).magnitude)
        pos_z_mm = electrode_array.pos_z.rescale(pq.mm).magnitude
        pos_theta_rad = electrode_array.pos_theta.rescale(pq.rad).magnitude + theta_offset

    ###################################################################################################
    ## 1. Constants

    ## Model angular frequencies
    # FFT-consistent bin centers (matches np.fft.fft + fftshift): spacing = 1, bins at
    # (-M/2, ..., M/2 - 1) for even M (and the analogous range for odd M).
    k_theta = np.fft.fftshift(np.fft.fftfreq(M, d=1.0 / M))

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

    # FFT-consistent angular-frequency grid (matches np.fft.fft + fftshift):
    # bin spacing = Fs/N (not Fs/(N-1)). Using fftfreq guarantees the grid
    # aligns with how the IFFT later interprets the spectrum, eliminating an
    # off-by-half-bin error that affected the previous np.linspace-based grid.
    k_t = 2 * math.pi * np.fft.fftshift(np.fft.fftfreq(N, d=1.0 / Fs_effective))
    # The DC bin (k_t = 0 ⇒ k_z = 0) makes the Bessel-function arguments
    # vanish and turns the 7×7 volume-conductor A-matrix singular. The
    # previous np.linspace grid for even N happened to avoid k = 0 because
    # of its (N-1) spacing; the FFT-correct grid lands exactly on it.
    # Replace exact zeros with a tiny offset (half a bin) so the DC bin
    # remains finite without measurably perturbing the rest of the spectrum.
    if N > 1:
        dk_t = 2 * math.pi * Fs_effective / N
        k_t = np.where(k_t == 0.0, 0.5 * dk_t, k_t)
    k_z = k_t / v
    (kt_mesh_kzkt, kz_mesh_kzkt) = np.meshgrid(k_t, k_z)

    ###################################################################################################
    ## 2. I(k_t, k_z)
    # Rosenfalck (1969) intracellular action potential spatial distribution
    # As used in Farina & Merletti (2001), equation 16
    # IMPORTANT: This matches the original Farina implementation exactly (no normalization)
    A = 96 / 10  # mV/mm^3 (amplitude parameter from Farina 2001, eq 16)

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
    ## 3. H_vc(k_z, k_theta) — single authority in volume_conductor.py
    H_vc, A_matrix, B_incomplete = layered_cylinder_transfer(
        k_z,
        k_theta,
        r=r,
        r_bone=r_bone,
        th_fat=th_fat,
        th_skin=th_skin,
        R=R,
        sig_muscle_rho=sig_muscle_rho,
        sig_muscle_z=sig_muscle_z,
        sig_fat=sig_fat,
        sig_skin=sig_skin,
        sig_bone=sig_bone,
        A_matrix=A_matrix,
        B_incomplete=B_incomplete,
        use_gpu=use_gpu,
    )

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

    k_theta_diff = k_theta[1] - k_theta[0]
    k_z_diff = k_z[1] - k_z[0]

    if _numba_B_kz_func is not None:
        # Use Numba-optimized parallel computation
        H_glo_f64 = np.ascontiguousarray(H_glo.real, dtype=np.float64)
        H_glo_i64 = np.ascontiguousarray(H_glo.imag, dtype=np.float64)
        pt_f64 = np.ascontiguousarray(pos_theta_rad, dtype=np.float64)
        km_f64 = np.ascontiguousarray(ktheta_mesh_kzktheta, dtype=np.float64)
        B_kz = _numba_B_kz_func(
            H_glo_f64, H_glo_i64, pt_f64,
            float(k_theta_diff), km_f64,
            channels[0], channels[1], len(k_z), len(k_theta),
        )
    else:
        B_kz = np.zeros((channels[0], channels[1], len(k_z)))
        for channel_z in range(channels[0]):
            for channel_theta in range(channels[1]):
                arg = np.multiply(
                    H_glo,
                    np.exp(1j * pos_theta_rad[channel_z, channel_theta] * ktheta_mesh_kzktheta)
                    * k_theta_diff,
                )
                B_kz[channel_z, channel_theta, :] = sum(np.transpose(arg)) / 2 / math.pi

    ###################################################################################################
    ## 6. phi(t) for each channel

    if _numba_phi_func is not None:
        # Use Numba-optimized parallel computation
        PHI_complex = _numba_phi_func(
            np.ascontiguousarray(I_kzkt.real, dtype=np.float64),
            np.ascontiguousarray(I_kzkt.imag, dtype=np.float64),
            np.ascontiguousarray(B_kz, dtype=np.float64),
            np.ascontiguousarray(pos_z_mm, dtype=np.float64),
            np.ascontiguousarray(kz_mesh_kzkt, dtype=np.float64),
            channels[0], channels[1], len(k_z), len(k_t),
            float(k_z_diff),
        )
        # Apply IFFT to get time-domain signal
        phi = np.zeros((channels[0], channels[1], len(t)))
        for channel_z in range(channels[0]):
            for channel_theta in range(channels[1]):
                # Use ifftshift (not fftshift) as the inverse of the fftshift
                # applied when building the centered spectrum. They are
                # equivalent for even N but differ by a one-sample phase ramp
                # for odd N — ifftshift is the correct inverse here.
                phi[channel_z, channel_theta, :] = np.real(
                    np.fft.ifft(
                        np.fft.ifftshift(PHI_complex[channel_z, channel_theta, :] * len(psi))
                    )
                )
    else:
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
                    np.exp(1j * pos_z_mm[channel_z, channel_theta] * kz_mesh_kzkt) * k_z_diff,
                )
                PHI = sum(arg2)
                # Use ifftshift (not fftshift) as the inverse of the fftshift
                # applied when building the centered spectrum. They are
                # equivalent for even N but differ by a one-sample phase ramp
                # for odd N — ifftshift is the correct inverse here.
                phi[channel_z, channel_theta, :] = np.real(
                    np.fft.ifft(
                        np.fft.ifftshift(PHI / 2 / math.pi * len(psi))
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


# NOTE: beartype disabled — called 100s-1000s of times per MU in the inner fiber loop
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
    use_cython: bool = False,
    use_gpu: bool = True,
    fiber_length__mm: float | None = None,
    theta_offset: float = 0.0,
    pos_z_precomputed: np.ndarray | None = None,
    pos_theta_precomputed: np.ndarray | None = None,
    rele_precomputed: float | None = None,
):
    """
    Simulate a single fiber in the layered-cylinder volume conductor.

    Thin wrapper around :func:`_simulate_fiber_v2_python`, kept for API
    stability.

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
        Deprecated and ignored (the Cython implementation was removed in
        0.12).  Passing ``True`` emits a ``DeprecationWarning``.
    use_gpu : bool, optional
        If True (default), the batched Bessel system may be solved on the GPU
        (CuPy) if available. For typical EMG simulations, CPU is faster due
        to GPU memory transfer overhead. Default is True.
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
        Generated surface EMG signal for each electrode [num_rows, num_cols, num_timepoints].
    A_matrix : np.ndarray
        A matrix for reuse in subsequent calls [n_theta, n_z, 7, 7].
    B_incomplete : np.ndarray
        B matrix for reuse in subsequent calls [n_theta, n_z, 7, 1].

    Notes
    -----
    The volume-conductor system matrix does not depend on the source radius
    ``R``; pass the returned ``A_matrix`` / ``B_incomplete`` back in when
    iterating over the fibres of a motor unit.
    """
    if use_cython:
        import warnings

        warnings.warn(
            "use_cython is deprecated and ignored: the Cython simulate_fiber "
            "implementation was removed in MyoGen 0.12 (it had diverged from "
            "the Python reference and was slower). Pass use_cython=False to "
            "silence this warning.",
            DeprecationWarning,
            stacklevel=2,
        )

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
        theta_offset,
        pos_z_precomputed,
        pos_theta_precomputed,
        rele_precomputed,
    )
