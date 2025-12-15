# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: warn.maybe_uninitialized=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

"""
Cython-optimized implementation of simulate_fiber_v2.

This module provides a high-performance version of the single fiber EMG simulation
using the 4-layer cylindrical volume conductor model (Farina et al., 2004).

Features:
- Log-space Bessel arithmetic to prevent numerical overflow
- Exponentially scaled Bessel functions (ive, kve)
- Overflow protection with MAX_LOG_SAFE threshold
- Parallelized Bessel function computation

Expected performance: 5-10x speedup over Python implementation.

TODO: Add support for fiber_length__mm parameter
----------------------------------------------
The Python implementation in simulate_fiber.py has been updated to accept an optional
fiber_length__mm parameter that defines the physical spatial extent for IAP kernel
evaluation, making MUAP duration independent of sampling resolution N.

This Cython implementation needs to be updated with the same parameter:
1. Add fiber_length__mm parameter to simulate_fiber_v2() signature
2. Update z-grid calculation (search for "z = np.linspace" in this file)
3. Replace: z = np.linspace(-(N-1)*(v/Fs)/2, (N-1)*(v/Fs)/2, N)
   With: if fiber_length__mm is None: [legacy] else: z = np.linspace(-fiber_length__mm/2, fiber_length__mm/2, N)

Until this is implemented, passing fiber_length__mm will trigger a fallback to the
Python implementation with a user warning.
"""

import numpy as np
cimport numpy as cnp
cimport cython
from cython.parallel import prange
from libc.math cimport exp, sqrt, sin, cos, M_PI as pi, fabs, log as clog, INFINITY
from scipy.special.cython_special cimport iv as In_cython, kv as Kn_cython, jv as Jn_cython
from scipy.special.cython_special cimport ive as In_scaled_cython, kve as Kn_scaled_cython

# Import Python versions for arrays
from scipy.special import iv as In, kv as Kn, jv as Jn, ive as In_scaled, kve as Kn_scaled

# Initialize NumPy C API
cnp.import_array()

# Pure CPU implementation - no GPU/CuPy
# This Cython version is optimized for CPU-only execution with parallelization


#######################################################################################################
##################################### Helper Functions #################################################
#######################################################################################################

cdef inline double In_tilde_scalar(double n, double x) nogil:
    """
    Compute (I_{n+1}(x) + I_{n-1}(x)) / 2.

    This is related to the derivative of the modified Bessel function of the first kind.
    """
    return (In_cython(n + 1.0, x) + In_cython(n - 1.0, x)) / 2.0


cdef inline double Kn_tilde_scalar(double n, double x) nogil:
    """
    Compute (K_{n+1}(x) + K_{n-1}(x)) / 2.

    This is related to the derivative of the modified Bessel function of the second kind.
    """
    return (Kn_cython(n + 1.0, x) + Kn_cython(n - 1.0, x)) / 2.0


cdef inline double log_In_scalar(double n, double x) nogil:
    """
    Compute log(In(n,x)) in a numerically stable way using scaled Bessel.

    Uses: log(In(n,z)) = log(In_scaled(n,z)) + |z|
    where In_scaled(n,z) = In(n,z)*exp(-|z|)
    """
    cdef double In_s = In_scaled_cython(n, x)
    if In_s > 0:
        return clog(In_s) + fabs(x)
    else:
        return -INFINITY


cdef inline double log_Kn_scalar(double n, double x) nogil:
    """
    Compute log(Kn(n,x)) in a numerically stable way using scaled Bessel.

    Uses: log(Kn(n,z)) = log(Kn_scaled(n,z)) - z
    where Kn_scaled(n,z) = Kn(n,z)*exp(z)
    """
    cdef double Kn_s = Kn_scaled_cython(n, x)
    if Kn_s > 0:
        return clog(Kn_s) - x
    else:
        return -INFINITY


cdef inline double logaddexp_scalar(double a, double b) nogil:
    """
    Compute log(exp(a) + exp(b)) in a numerically stable way.

    Implements the log-sum-exp trick.
    """
    cdef double max_val
    if a == -INFINITY and b == -INFINITY:
        return -INFINITY
    elif a > b:
        max_val = a
        return a + clog(1.0 + exp(b - a))
    else:
        max_val = b
        return b + clog(1.0 + exp(a - b))


cdef inline double log_In_tilde_scalar(double n, double x) nogil:
    """
    Compute log(In_tilde(n,x)) = log((In(n+1,x) + In(n-1,x))/2) using log-space.

    Uses log-sum-exp trick to avoid overflow.
    """
    cdef double log_In_p1 = log_In_scalar(n + 1.0, x)
    cdef double log_In_m1 = log_In_scalar(n - 1.0, x)
    return logaddexp_scalar(log_In_p1, log_In_m1) - clog(2.0)


cdef inline double log_Kn_tilde_scalar(double n, double x) nogil:
    """
    Compute log(Kn_tilde(n,x)) = log((Kn(n+1,x) + Kn(n-1,x))/2) using log-space.

    Uses log-sum-exp trick to avoid overflow.
    """
    cdef double log_Kn_p1 = log_Kn_scalar(n + 1.0, x)
    cdef double log_Kn_m1 = log_Kn_scalar(n - 1.0, x)
    return logaddexp_scalar(log_Kn_p1, log_Kn_m1) - clog(2.0)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef cnp.ndarray[cnp.float64_t, ndim=1] f_minus_t(double[::1] y):
    """
    Reverse array - equivalent to y[::-1].

    Note: Original Python had bug (y[-i] when i=0 gives last element).
    This version correctly reverses the array.
    """
    cdef Py_ssize_t n = y.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] y_new = np.zeros(n, dtype=np.float64)
    cdef Py_ssize_t i

    for i in range(n):
        y_new[i] = y[n - 1 - i]

    return y_new


#######################################################################################################
############################ Performance-Critical Subroutines ##########################################
#######################################################################################################

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void compute_bessel_arrays(
    double[:, ::1] K_THETA,
    double[:, ::1] K_Z,
    double a, double b, double c, double d,
    double am, double bm, double Rm,
    double[:, ::1] In_a,
    double[:, ::1] In_b,
    double[:, ::1] In_c,
    double[:, ::1] In_d,
    double[:, ::1] In_am,
    double[:, ::1] In_bm,
    double[:, ::1] In_Rm,
    double[:, ::1] Kn_am,
    double[:, ::1] Kn_bm,
    double[:, ::1] Kn_b,
    double[:, ::1] Kn_c,
    double[:, ::1] Kn_d,
    double[:, ::1] Kn_Rm,
    double[:, ::1] In_tilde_a,
    double[:, ::1] In_tilde_b,
    double[:, ::1] In_tilde_c,
    double[:, ::1] In_tilde_d,
    double[:, ::1] In_tilde_am,
    double[:, ::1] In_tilde_bm,
    double[:, ::1] Kn_tilde_am,
    double[:, ::1] Kn_tilde_bm,
    double[:, ::1] Kn_tilde_b,
    double[:, ::1] Kn_tilde_c,
    double[:, ::1] Kn_tilde_d
) noexcept nogil:
    """
    Compute all required Bessel function values in parallel.

    This function computes all modified Bessel functions (I, K)
    and their tilde versions needed for the A matrix and B vector construction.
    Uses parallel loops for performance.
    """
    cdef Py_ssize_t i, j
    cdef Py_ssize_t n_i = K_THETA.shape[0]
    cdef Py_ssize_t n_j = K_THETA.shape[1]
    cdef double n, x

    for i in prange(n_i):
        for j in range(n_j):
            n = K_THETA[i, j]

            # Regular Bessel functions (NO unit conversion - matches original Farina implementation)
            In_a[i, j] = In_cython(n, a * K_Z[i, j])
            In_b[i, j] = In_cython(n, b * K_Z[i, j])
            In_c[i, j] = In_cython(n, c * K_Z[i, j])
            In_d[i, j] = In_cython(n, d * K_Z[i, j])
            In_am[i, j] = In_cython(n, am * K_Z[i, j])
            In_bm[i, j] = In_cython(n, bm * K_Z[i, j])
            In_Rm[i, j] = In_cython(n, Rm * K_Z[i, j])

            Kn_am[i, j] = Kn_cython(n, am * K_Z[i, j])
            Kn_bm[i, j] = Kn_cython(n, bm * K_Z[i, j])
            Kn_b[i, j] = Kn_cython(n, b * K_Z[i, j])
            Kn_c[i, j] = Kn_cython(n, c * K_Z[i, j])
            Kn_d[i, j] = Kn_cython(n, d * K_Z[i, j])
            Kn_Rm[i, j] = Kn_cython(n, Rm * K_Z[i, j])

            # Tilde versions (derivatives)
            In_tilde_a[i, j] = In_tilde_scalar(n, a * K_Z[i, j])
            In_tilde_b[i, j] = In_tilde_scalar(n, b * K_Z[i, j])
            In_tilde_c[i, j] = In_tilde_scalar(n, c * K_Z[i, j])
            In_tilde_d[i, j] = In_tilde_scalar(n, d * K_Z[i, j])
            In_tilde_am[i, j] = In_tilde_scalar(n, am * K_Z[i, j])
            In_tilde_bm[i, j] = In_tilde_scalar(n, bm * K_Z[i, j])

            Kn_tilde_am[i, j] = Kn_tilde_scalar(n, am * K_Z[i, j])
            Kn_tilde_bm[i, j] = Kn_tilde_scalar(n, bm * K_Z[i, j])
            Kn_tilde_b[i, j] = Kn_tilde_scalar(n, b * K_Z[i, j])
            Kn_tilde_c[i, j] = Kn_tilde_scalar(n, c * K_Z[i, j])
            Kn_tilde_d[i, j] = Kn_tilde_scalar(n, d * K_Z[i, j])


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void build_A_matrix(
    double[:, :, :, ::1] A_mat,
    double[:, ::1] In_a, double[:, ::1] In_b, double[:, ::1] In_c, double[:, ::1] In_d,
    double[:, ::1] In_am, double[:, ::1] In_bm,
    double[:, ::1] Kn_am, double[:, ::1] Kn_bm, double[:, ::1] Kn_b,
    double[:, ::1] Kn_c, double[:, ::1] Kn_d,
    double[:, ::1] In_tilde_a, double[:, ::1] In_tilde_b,
    double[:, ::1] In_tilde_c, double[:, ::1] In_tilde_d,
    double[:, ::1] In_tilde_am, double[:, ::1] In_tilde_bm,
    double[:, ::1] Kn_tilde_am, double[:, ::1] Kn_tilde_bm,
    double[:, ::1] Kn_tilde_b, double[:, ::1] Kn_tilde_c, double[:, ::1] Kn_tilde_d,
    double sig_bone, double sig_muscle_rho, double sig_muscle_z,
    double sig_fat, double sig_skin
) noexcept nogil:
    """
    Build the 4D A matrix for the volume conductor model.

    A_mat has shape (n_theta, n_z, 7, 7) where 7x7 is the linear system
    for the 4-layer cylindrical model (bone, muscle, fat, skin).

    Parallelized with prange for performance.
    """
    cdef Py_ssize_t i, j
    cdef Py_ssize_t n_theta = A_mat.shape[0]
    cdef Py_ssize_t n_z = A_mat.shape[1]
    cdef double sqrt_sig_prod = sqrt(sig_muscle_rho * sig_muscle_z)

    for i in prange(n_theta):
        for j in range(n_z):
            # Row 0
            A_mat[i, j, 0, 0] = 1.0
            A_mat[i, j, 0, 1] = -In_am[i, j] / In_bm[i, j]
            A_mat[i, j, 0, 2] = -Kn_am[i, j] / Kn_bm[i, j]

            # Row 1
            A_mat[i, j, 1, 0] = sig_bone / In_a[i, j] * In_tilde_a[i, j]
            A_mat[i, j, 1, 1] = -sqrt_sig_prod / In_bm[i, j] * In_tilde_am[i, j]
            A_mat[i, j, 1, 2] = -sqrt_sig_prod / Kn_bm[i, j] * (-1.0) * Kn_tilde_am[i, j]

            # Row 2
            A_mat[i, j, 2, 1] = 1.0
            A_mat[i, j, 2, 2] = 1.0
            A_mat[i, j, 2, 3] = -In_b[i, j] / In_c[i, j]
            A_mat[i, j, 2, 4] = -Kn_b[i, j] / Kn_c[i, j]

            # Row 3
            A_mat[i, j, 3, 1] = sqrt_sig_prod / In_bm[i, j] * In_tilde_bm[i, j]
            A_mat[i, j, 3, 2] = sqrt_sig_prod / Kn_bm[i, j] * (-1.0) * Kn_tilde_bm[i, j]
            A_mat[i, j, 3, 3] = -sig_fat / In_c[i, j] * In_tilde_b[i, j]
            A_mat[i, j, 3, 4] = -sig_fat / Kn_c[i, j] * (-1.0) * Kn_tilde_b[i, j]

            # Row 4
            A_mat[i, j, 4, 3] = 1.0
            A_mat[i, j, 4, 4] = 1.0
            A_mat[i, j, 4, 5] = -In_c[i, j] / In_d[i, j]
            A_mat[i, j, 4, 6] = -Kn_c[i, j] / Kn_d[i, j]

            # Row 5
            A_mat[i, j, 5, 3] = sig_fat / In_c[i, j] * In_tilde_c[i, j]
            A_mat[i, j, 5, 4] = sig_fat / Kn_c[i, j] * (-1.0) * Kn_tilde_c[i, j]
            A_mat[i, j, 5, 5] = -sig_skin / In_d[i, j] * In_tilde_c[i, j]
            A_mat[i, j, 5, 6] = -sig_skin / Kn_d[i, j] * (-1.0) * Kn_tilde_c[i, j]

            # Row 6
            A_mat[i, j, 6, 5] = sig_skin / In_d[i, j] * In_tilde_d[i, j]
            A_mat[i, j, 6, 6] = sig_skin / Kn_d[i, j] * (-1.0) * Kn_tilde_d[i, j]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void compute_B_kz(
    double[:, ::1] H_glo_real,
    double[:, ::1] H_glo_imag,
    double[:, ::1] pos_theta,
    double[:, ::1] ktheta_mesh,
    double k_theta_diff,
    double[:, :, ::1] B_kz
) noexcept nogil:
    """
    Compute B_kz array for all channels.

    Parallelized channel loop with manual complex arithmetic for performance.
    """
    cdef Py_ssize_t ch_z, ch_theta, i, j
    cdef Py_ssize_t n_kz = H_glo_real.shape[0]
    cdef Py_ssize_t n_ktheta = H_glo_real.shape[1]
    cdef double sum_real, phase, cos_phase, sin_phase, factor

    factor = k_theta_diff / (2.0 * pi)

    for ch_z in prange(B_kz.shape[0], nogil=True):
        for ch_theta in range(B_kz.shape[1]):
            for i in range(n_kz):
                sum_real = 0.0
                for j in range(n_ktheta):
                    phase = pos_theta[ch_z, ch_theta] * ktheta_mesh[i, j]
                    cos_phase = cos(phase)
                    sin_phase = sin(phase)
                    # Complex multiplication: H_glo * exp(1j * phase)
                    sum_real = sum_real + (H_glo_real[i, j] * cos_phase - H_glo_imag[i, j] * sin_phase)
                B_kz[ch_z, ch_theta, i] = sum_real * factor

#######################################################################################################
##################################### Main Function ####################################################
#######################################################################################################

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef tuple simulate_fiber_v2(
    double Fs,
    double v,
    Py_ssize_t N,
    Py_ssize_t M,
    double r,
    double r_bone,
    double th_fat,
    double th_skin,
    double R,
    double L1,
    double L2,
    double zi,
    object electrode_array,  # SurfaceElectrodeArray - Python object
    double sig_muscle_rho,
    double sig_muscle_z,
    double sig_fat,
    double sig_skin,
    double sig_bone = 0.0,
    cnp.ndarray A_matrix = None,
    cnp.ndarray B_incomplete = None,
):
    """
    Simulate a single fiber using Cython-optimized volume conductor model.

    This function computes the EMG signal generated by a single muscle fiber
    using a 4-layer cylindrical volume conductor model (Farina et al., 2004).

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

    Returns
    -------
    phi : np.ndarray
        Generated surface EMG signal for each electrode [num_rows, num_cols, num_timepoints].
    A_matrix : np.ndarray
        A matrix for reuse in subsequent calls [n_theta, n_z, 7, 7].
    B_incomplete : np.ndarray
        B matrix for reuse in subsequent calls [n_theta, n_z, 7, 1].
    """
    # Extract electrode configuration
    cdef Py_ssize_t channels_0 = electrode_array.num_rows
    cdef Py_ssize_t channels_1 = electrode_array.num_cols

    # Extract magnitude from Quantity objects
    import quantities as pq
    cdef double rele = float(electrode_array.electrode_radius__mm.rescale(pq.mm).magnitude)

    ###################################################################################################
    ## 1. Constants
    ###################################################################################################

    # Model angular frequencies
    k_theta = np.linspace(-(M - 1) / 2, (M - 1) / 2, M)
    k_t = 2 * pi * np.linspace(-Fs / 2, Fs / 2, N)
    k_z = k_t / v
    kt_mesh_kzkt, kz_mesh_kzkt = np.meshgrid(k_t, k_z)

    # Model radii (Farina, 2004, Figure 1-b)
    cdef double th_muscle = r - th_fat - th_skin - r_bone
    cdef double a = r_bone
    cdef double b = r_bone + th_muscle
    cdef double c = r_bone + th_muscle + th_fat
    cdef double d = r_bone + th_muscle + th_fat + th_skin

    ###################################################################################################
    ## 2. I(k_t, k_z)
    ###################################################################################################

    cdef double lambda_z = 1.0  # mm, typical for FDI (you can tune)

    cdef double A_coef = 96.0  # mV/mm^3 from Farina 2001 eq. 16

    # correct spatial window
    z_max = 6 * lambda_z  
    z = np.linspace(-z_max, z_max, N)

    aux = np.zeros_like(z)
    positive_mask = z >= 0

    z_norm = z[positive_mask] / lambda_z
    z_norm = np.clip(z_norm, 0, 6)  # enforce support used in Farina's model

    aux[positive_mask] = (
        A_coef
        * np.exp(-z_norm)
        * (3*z_norm**2 - z_norm**3)
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
        np.exp(-1j * k_eps * L1 / 2), np.sinc(k_eps * L1 / 2 / pi) * L1
    )
    aux2 = np.multiply(
        np.exp(1j * k_beta * L2 / 2), np.sinc(k_beta * L2 / 2 / pi) * L2
    )
    I_kzkt = np.multiply(I_kzkt, (aux1 - aux2))

    t = np.linspace(0, (N - 1) / Fs, N)

    ###################################################################################################
    ## 3. H_vc(k_z, k_theta)
    ###################################################################################################

    cdef double am = a * sqrt(sig_muscle_z / sig_muscle_rho)
    cdef double bm = b * sqrt(sig_muscle_z / sig_muscle_rho)
    cdef double Rm = R * sqrt(sig_muscle_z / sig_muscle_rho)

    cdef Py_ssize_t i_start = <Py_ssize_t>(len(k_z) / 2)
    cdef Py_ssize_t i_end = len(k_z)
    cdef Py_ssize_t j_start = <Py_ssize_t>(len(k_theta) / 2)
    cdef Py_ssize_t j_end = len(k_theta)

    # Create sub-arrays for positive frequencies
    k_z_pos = k_z[i_start:i_end]
    k_theta_pos = k_theta[j_start:j_end]

    K_THETA, K_Z = np.meshgrid(k_theta_pos, k_z_pos, indexing="ij")

    cdef Py_ssize_t n_theta = K_THETA.shape[0]
    cdef Py_ssize_t n_z = K_THETA.shape[1]

    # Declare arrays
    cdef cnp.ndarray[cnp.float64_t, ndim=4] A_mat_np
    cdef cnp.ndarray[cnp.float64_t, ndim=4] B_np
    cdef double[:, :, :, ::1] A_mat
    cdef double[:, :, :, ::1] B

    # Memoryviews for Bessel functions
    cdef double[:, ::1] K_THETA_view = K_THETA
    cdef double[:, ::1] K_Z_view = K_Z

    # Allocate Bessel function arrays
    cdef cnp.ndarray[cnp.float64_t, ndim=2] In_a_np, In_b_np, In_c_np, In_d_np
    cdef cnp.ndarray[cnp.float64_t, ndim=2] In_am_np, In_bm_np, In_Rm_np
    cdef cnp.ndarray[cnp.float64_t, ndim=2] Kn_am_np, Kn_bm_np, Kn_b_np, Kn_c_np, Kn_d_np, Kn_Rm_np
    cdef cnp.ndarray[cnp.float64_t, ndim=2] In_tilde_a_np, In_tilde_b_np, In_tilde_c_np, In_tilde_d_np
    cdef cnp.ndarray[cnp.float64_t, ndim=2] In_tilde_am_np, In_tilde_bm_np
    cdef cnp.ndarray[cnp.float64_t, ndim=2] Kn_tilde_am_np, Kn_tilde_bm_np, Kn_tilde_b_np, Kn_tilde_c_np, Kn_tilde_d_np

    # Additional loop variables and temporary arrays
    cdef Py_ssize_t i_b, j_b, i_update, channel_z, channel_theta
    cdef double sqrt_sig_ratio, k_z_diff, k_theta_diff
    cdef double[:, ::1] Kn_Rm_view, In_Rm_view, pos_theta, ktheta_mesh_view
    cdef double[:, :, ::1] B_kz_view
    cdef cnp.ndarray[cnp.float64_t, ndim=3] B_kz_np, phi_np

    # Check if we need to compute matrices
    cdef bint need_compute = (A_matrix is None or B_incomplete is None)

    if need_compute:
        # Allocate arrays
        A_mat_np = np.zeros((n_theta, n_z, 7, 7), dtype=np.float64)
        B_np = np.zeros((n_theta, n_z, 7, 1), dtype=np.float64)

        # Bessel arrays
        In_a_np = np.empty((n_theta, n_z), dtype=np.float64)
        In_b_np = np.empty((n_theta, n_z), dtype=np.float64)
        In_c_np = np.empty((n_theta, n_z), dtype=np.float64)
        In_d_np = np.empty((n_theta, n_z), dtype=np.float64)
        In_am_np = np.empty((n_theta, n_z), dtype=np.float64)
        In_bm_np = np.empty((n_theta, n_z), dtype=np.float64)
        In_Rm_np = np.empty((n_theta, n_z), dtype=np.float64)

        Kn_am_np = np.empty((n_theta, n_z), dtype=np.float64)
        Kn_bm_np = np.empty((n_theta, n_z), dtype=np.float64)
        Kn_b_np = np.empty((n_theta, n_z), dtype=np.float64)
        Kn_c_np = np.empty((n_theta, n_z), dtype=np.float64)
        Kn_d_np = np.empty((n_theta, n_z), dtype=np.float64)
        Kn_Rm_np = np.empty((n_theta, n_z), dtype=np.float64)

        In_tilde_a_np = np.empty((n_theta, n_z), dtype=np.float64)
        In_tilde_b_np = np.empty((n_theta, n_z), dtype=np.float64)
        In_tilde_c_np = np.empty((n_theta, n_z), dtype=np.float64)
        In_tilde_d_np = np.empty((n_theta, n_z), dtype=np.float64)
        In_tilde_am_np = np.empty((n_theta, n_z), dtype=np.float64)
        In_tilde_bm_np = np.empty((n_theta, n_z), dtype=np.float64)

        Kn_tilde_am_np = np.empty((n_theta, n_z), dtype=np.float64)
        Kn_tilde_bm_np = np.empty((n_theta, n_z), dtype=np.float64)
        Kn_tilde_b_np = np.empty((n_theta, n_z), dtype=np.float64)
        Kn_tilde_c_np = np.empty((n_theta, n_z), dtype=np.float64)
        Kn_tilde_d_np = np.empty((n_theta, n_z), dtype=np.float64)

        # Compute all Bessel functions in parallel
        compute_bessel_arrays(
            K_THETA_view, K_Z_view,
            a, b, c, d, am, bm, Rm,
            In_a_np, In_b_np, In_c_np, In_d_np,
            In_am_np, In_bm_np, In_Rm_np,
            Kn_am_np, Kn_bm_np, Kn_b_np, Kn_c_np, Kn_d_np, Kn_Rm_np,
            In_tilde_a_np, In_tilde_b_np, In_tilde_c_np, In_tilde_d_np,
            In_tilde_am_np, In_tilde_bm_np,
            Kn_tilde_am_np, Kn_tilde_bm_np, Kn_tilde_b_np, Kn_tilde_c_np, Kn_tilde_d_np
        )

        # Build A matrix
        A_mat = A_mat_np
        build_A_matrix(
            A_mat,
            In_a_np, In_b_np, In_c_np, In_d_np,
            In_am_np, In_bm_np,
            Kn_am_np, Kn_bm_np, Kn_b_np, Kn_c_np, Kn_d_np,
            In_tilde_a_np, In_tilde_b_np, In_tilde_c_np, In_tilde_d_np,
            In_tilde_am_np, In_tilde_bm_np,
            Kn_tilde_am_np, Kn_tilde_bm_np, Kn_tilde_b_np, Kn_tilde_c_np, Kn_tilde_d_np,
            sig_bone, sig_muscle_rho, sig_muscle_z, sig_fat, sig_skin
        )

        A_matrix = A_mat_np.copy()
        B_incomplete = B_np.copy()
    else:
        A_mat_np = A_matrix.copy()
        B_np = B_incomplete.copy()
        A_mat = A_mat_np
        B = B_np

    # Build B vector using LOG-SPACE arithmetic to prevent overflow
    # Compute products as: In*Kn = exp(log(In) + log(Kn))
    # Safety threshold: exp(700) ≈ 1e304 (near float64 max)
    cdef double MAX_LOG_SAFE = 700.0
    cdef double log_val, n, k_z_val

    B = B_np  # Get view
    sqrt_sig_ratio = sqrt(sig_muscle_z / sig_muscle_rho)

    for i_b in range(n_theta):
        for j_b in range(n_z):
            n = K_THETA_view[i_b, j_b]
            k_z_val = K_Z_view[i_b, j_b]

            # B[0,0] = In_am * Kn_Rm / sig_muscle_rho
            log_val = log_In_scalar(n, am * k_z_val) + log_Kn_scalar(n, Rm * k_z_val)
            if log_val < MAX_LOG_SAFE:
                B[i_b, j_b, 0, 0] = exp(log_val) / sig_muscle_rho
            else:
                B[i_b, j_b, 0, 0] = 0.0

            # B[1,0] = sqrt(sig_z/sig_rho) * In_tilde_am * Kn_Rm
            log_val = log_In_tilde_scalar(n, am * k_z_val) + log_Kn_scalar(n, Rm * k_z_val)
            if log_val < MAX_LOG_SAFE:
                B[i_b, j_b, 1, 0] = sqrt_sig_ratio * exp(log_val)
            else:
                B[i_b, j_b, 1, 0] = 0.0

            # B[2,0] = -Kn_bm * In_Rm / sig_muscle_rho
            log_val = log_Kn_scalar(n, bm * k_z_val) + log_In_scalar(n, Rm * k_z_val)
            if log_val < MAX_LOG_SAFE:
                B[i_b, j_b, 2, 0] = -exp(log_val) / sig_muscle_rho
            else:
                B[i_b, j_b, 2, 0] = 0.0

            # B[3,0] = sqrt(sig_z/sig_rho) * Kn_tilde_bm * In_Rm
            log_val = log_Kn_tilde_scalar(n, bm * k_z_val) + log_In_scalar(n, Rm * k_z_val)
            if log_val < MAX_LOG_SAFE:
                B[i_b, j_b, 3, 0] = sqrt_sig_ratio * exp(log_val)
            else:
                B[i_b, j_b, 3, 0] = 0.0

    # Solve linear system
    A_flat = np.asarray(A_mat).reshape(-1, 7, 7)
    B_flat = np.asarray(B).reshape(-1, 7, 1)

    if r_bone == 0:
        A_flat = A_flat[..., 2:, 2:]
        B_flat = B_flat[..., 2:, :]

        # Pure CPU solve (optimized for small-to-medium problems)
        X = np.linalg.solve(A_flat, B_flat)

        X = X.reshape(n_theta, n_z, 5, 1)
        H_vc = X[..., 3, 0] + X[..., 4, 0]
    else:
        # Pure CPU solve (optimized for small-to-medium problems)
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
    ###################################################################################################

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
    ## 5. H_glo(k_z, k_theta) and B_kz
    ###################################################################################################

    H_glo = np.multiply(H_vc, H_ele)

    # Split complex array into real and imaginary (ensure writable copies)
    H_glo_real = np.array(np.real(H_glo), dtype=np.float64, copy=True, order='C')
    H_glo_imag = np.array(np.imag(H_glo), dtype=np.float64, copy=True, order='C')

    # Extract electrode positions (ensure writable, contiguous)
    # Extract magnitude from Quantity objects
    import quantities as pq
    pos_theta = np.array(electrode_array.pos_theta.rescale(pq.rad).magnitude, dtype=np.float64, copy=True, order='C')
    pos_z = np.array(electrode_array.pos_z.rescale(pq.mm).magnitude, dtype=np.float64, copy=True, order='C')
    ktheta_mesh_view = np.array(ktheta_mesh_kzktheta, dtype=np.float64, copy=True, order='C')

    # Allocate B_kz
    B_kz_np = np.zeros(
        (channels_0, channels_1, len(k_z)), dtype=np.float64
    )
    B_kz_view = B_kz_np

    # Compute B_kz with Cython
    k_theta_diff = k_theta[1] - k_theta[0]
    compute_B_kz(
        H_glo_real, H_glo_imag,
        pos_theta, ktheta_mesh_view,
        k_theta_diff, B_kz_view
    )

    ###################################################################################################
    ## 6. phi(t) for each channel
    ###################################################################################################

    # Allocate phi
    phi_np = np.zeros(
        (channels_0, channels_1, len(t)), dtype=np.float64
    )

    # Compute phi - channel iteration
    k_z_diff = k_z[1] - k_z[0]

    for channel_z in range(channels_0):
        for channel_theta in range(channels_1):
            # Compute in frequency domain
            auxiliar = np.dot(
                np.ones((len(I_kzkt[1, :]), 1)),
                B_kz_np[channel_z, channel_theta, :].reshape(1, -1),
            )
            auxiliar = np.transpose(auxiliar)
            arg = np.multiply(I_kzkt, auxiliar)
            arg2 = np.multiply(
                arg,
                np.exp(1j * pos_z[channel_z, channel_theta] * kz_mesh_kzkt) * k_z_diff,
            )
            PHI = np.sum(arg2, axis=0)
            phi_np[channel_z, channel_theta, :] = np.real(
                (np.fft.ifft(np.fft.fftshift(PHI / 2 / pi * len(psi))))
            )

    return (phi_np, A_matrix, B_incomplete)
