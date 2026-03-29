"""
Unified fiber simulation module.

Combines iEMG and sEMG fiber simulation under a single Rosenfalck source
model (D1=96, z in physical mm) with volume conductor kernels computed
as 1D spatial impulse responses.

References
----------
.. [1] Rosenfalck, P., 1969. Intra- and extracellular potential fields of active
       nerve and muscle fibres. Acta Physiol. Scand. Suppl. 321, 1-168.
.. [2] Farina, D. et al., 2004. A surface EMG generation model with multilayer
       cylindrical description of the volume conductor. IEEE TBME 51(3), 415-426.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import jv as Jn

from myogen.simulator.core.emg.electrodes import SurfaceElectrodeArray
from myogen.simulator.core.emg.surface.simulate_fiber import log_In, log_Kn


def rosenfalck_dVm_dz(z: np.ndarray, D1: float = 96.0) -> np.ndarray:
    """
    First spatial derivative of the Rosenfalck transmembrane potential.

    Computes dVm/dz = D1 * (3z^2 - z^3) * exp(-z) for z > 0, else 0.
    z is in physical millimeters with no artificial scaling.

    Parameters
    ----------
    z : np.ndarray
        Spatial coordinates along fiber in mm.
    D1 : float, default=96.0
        Amplitude parameter in mV/mm^3 (Rosenfalck 1969).

    Returns
    -------
    np.ndarray
        First derivative of transmembrane potential (mV/mm).
    """
    result = np.zeros_like(z, dtype=np.float64)
    pos = z > 0
    zp = z[pos]
    result[pos] = D1 * (3.0 * zp**2 - zp**3) * np.exp(-zp)
    return result


def compute_intramuscular_kernel(
    z_grid: np.ndarray,
    electrode_z: np.ndarray,
    r: float,
    sigma_r: float = 63.0,
    sigma_z: float = 330.0,
) -> np.ndarray:
    """
    Compute the intramuscular volume conductor kernel using the Green's function.

    For an anisotropic infinite medium (Nandedkar & Stalberg 1983):
    b(z) = 1/(4*pi*sigma_r) / sqrt(sigma_z/sigma_r * r^2 + (z - z_elec)^2)

    Parameters
    ----------
    z_grid : np.ndarray
        Spatial grid in mm.
    electrode_z : np.ndarray
        Electrode z-positions in mm, shape (n_electrodes,).
    r : float
        Radial distance from fiber to electrode in mm.
    sigma_r : float, default=63.0
        Radial conductivity in S/m.
    sigma_z : float, default=330.0
        Longitudinal conductivity in S/m.

    Returns
    -------
    np.ndarray
        Kernel b(z), shape (n_electrodes, len(z_grid)).
    """
    sigma_r_mm = sigma_r / 1000.0  # S/m -> S/mm
    sigma_z_mm = sigma_z / 1000.0

    b_z = np.zeros((len(electrode_z), len(z_grid)))
    for i, z_e in enumerate(electrode_z):
        b_z[i] = (1.0 / (4.0 * np.pi * sigma_r_mm)) / np.sqrt(
            sigma_z_mm / sigma_r_mm * r**2 + (z_grid - z_e) ** 2
        )
    return b_z


def compute_surface_kernel(
    z_grid: np.ndarray,
    k_theta: np.ndarray,
    R: float,
    electrode_array: SurfaceElectrodeArray,
    r: float,
    r_bone: float,
    th_fat: float,
    th_skin: float,
    sig_muscle_rho: float,
    sig_muscle_z: float,
    sig_fat: float,
    sig_skin: float,
    sig_bone: float = 0.0,
    A_matrix: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the surface volume conductor spatial kernel b(z).

    Extracts the Bessel volume conductor computation (Farina 2004) from the
    existing sEMG pipeline and adds an IFFT step to produce the spatial kernel.

    Steps:
      1. Derive kz from the z_grid via FFT frequencies.
      2. Compute H_vc(kz, k_theta) using the Bessel log-space A-matrix solver.
      3. Compute H_ele(kz, k_theta) from the electrode array (spatial filter + size).
      4. Compute H_glo = H_vc * H_ele.
      5. Integrate over k_theta: B(kz) = (1/2pi) * sum_ktheta H_glo * exp(j*theta_elec*ktheta) * dk_theta.
      6. IFFT B(kz) to obtain the spatial kernel b(z).

    Parameters
    ----------
    z_grid : np.ndarray
        Spatial grid along the fiber axis in mm, length Nz.
    k_theta : np.ndarray
        Angular frequency indices, e.g. arange(-(M-1)/2, (M-1)/2+1).
    R : float
        Source radial position in mm (fiber depth).
    electrode_array : SurfaceElectrodeArray
        Electrode array configuration.
    r : float
        Total model radius in mm.
    r_bone : float
        Bone radius in mm.
    th_fat : float
        Fat layer thickness in mm.
    th_skin : float
        Skin layer thickness in mm.
    sig_muscle_rho : float
        Muscle conductivity in radial direction (S/m).
    sig_muscle_z : float
        Muscle conductivity in longitudinal direction (S/m).
    sig_fat : float
        Fat conductivity (S/m).
    sig_skin : float
        Skin conductivity (S/m).
    sig_bone : float, optional
        Bone conductivity (S/m), default 0.0.
    A_matrix : np.ndarray or None, optional
        Cached A matrix from a previous call with identical geometry.
        Only the A matrix is cacheable because the B vector depends on R
        (fiber depth) which changes per fiber.

    Returns
    -------
    b_z : np.ndarray
        Spatial kernel, shape (n_rows, n_cols, len(z_grid)). Real-valued.
    A_matrix : np.ndarray
        A matrix for caching.
    """
    import quantities as pq

    Nz = len(z_grid)

    # --- Derive kz from z_grid via FFT frequencies ---
    dz = z_grid[1] - z_grid[0]
    k_z = 2.0 * np.pi * np.fft.fftfreq(Nz, d=dz)
    # Shift to match the convention used by the existing code (centered)
    k_z = np.fft.fftshift(k_z)

    # --- Tissue geometry (Farina 2004 Fig 1b) ---
    th_muscle = r - th_fat - th_skin - r_bone
    a = r_bone                              # bone outer radius
    b = r_bone + th_muscle                  # muscle outer radius
    c = r_bone + th_muscle + th_fat         # fat outer radius
    d = r_bone + th_muscle + th_fat + th_skin  # skin outer radius

    am = a * math.sqrt(sig_muscle_z / sig_muscle_rho)
    bm = b * math.sqrt(sig_muscle_z / sig_muscle_rho)
    Rm = R * math.sqrt(sig_muscle_z / sig_muscle_rho)

    # --- Electrode properties ---
    rele = float(electrode_array.electrode_radius__mm.rescale(pq.mm).magnitude)
    pos_z_mm = electrode_array.pos_z.rescale(pq.mm).magnitude
    pos_theta_rad = electrode_array.pos_theta.rescale(pq.rad).magnitude
    channels = [electrode_array.num_rows, electrode_array.num_cols]

    # =========================================================================
    # Section 3: H_vc(kz, k_theta)  — Bessel volume conductor transfer function
    # =========================================================================
    # H_vc is even in both kz and k_theta, so we compute it for unique
    # non-negative |kz| and |k_theta| values, then map back to the full grid.
    kz_abs = np.abs(k_z)
    kt_abs = np.abs(k_theta)

    # Unique non-negative values (sorted) for computation
    kz_unique = np.unique(kz_abs)
    kt_unique = np.unique(kt_abs)

    # Exclude kz=0 from computation (singular Bessel system; H_vc=0 at DC)
    kz_unique_nz = kz_unique[kz_unique > 0]

    K_THETA_u, K_Z_u = np.meshgrid(kt_unique, kz_unique_nz, indexing="ij")
    n_theta_u, n_z_u = K_THETA_u.shape

    A_mat = np.zeros((n_theta_u, n_z_u, 7, 7))
    B = np.zeros((n_theta_u, n_z_u, 7, 1))

    if A_matrix is not None:
        A_mat = A_matrix
    else:
        # Log-space Bessel functions for all radii
        log_In_a = log_In(K_THETA_u, a * K_Z_u)
        log_In_am = log_In(K_THETA_u, am * K_Z_u)
        log_In_b = log_In(K_THETA_u, b * K_Z_u)
        log_In_bm = log_In(K_THETA_u, bm * K_Z_u)
        log_In_c = log_In(K_THETA_u, c * K_Z_u)
        log_In_d = log_In(K_THETA_u, d * K_Z_u)

        log_Kn_am = log_Kn(K_THETA_u, am * K_Z_u)
        log_Kn_b = log_Kn(K_THETA_u, b * K_Z_u)
        log_Kn_bm = log_Kn(K_THETA_u, bm * K_Z_u)
        log_Kn_c = log_Kn(K_THETA_u, c * K_Z_u)
        log_Kn_d = log_Kn(K_THETA_u, d * K_Z_u)

        # Tilde functions: In_tilde = (In(n+1) + In(n-1)) / 2
        log_In_tilde_a = np.logaddexp(
            log_In(K_THETA_u + 1, a * K_Z_u), log_In(K_THETA_u - 1, a * K_Z_u)
        ) - np.log(2)
        log_In_tilde_am = np.logaddexp(
            log_In(K_THETA_u + 1, am * K_Z_u), log_In(K_THETA_u - 1, am * K_Z_u)
        ) - np.log(2)
        log_In_tilde_b = np.logaddexp(
            log_In(K_THETA_u + 1, b * K_Z_u), log_In(K_THETA_u - 1, b * K_Z_u)
        ) - np.log(2)
        log_In_tilde_bm = np.logaddexp(
            log_In(K_THETA_u + 1, bm * K_Z_u), log_In(K_THETA_u - 1, bm * K_Z_u)
        ) - np.log(2)
        log_In_tilde_c = np.logaddexp(
            log_In(K_THETA_u + 1, c * K_Z_u), log_In(K_THETA_u - 1, c * K_Z_u)
        ) - np.log(2)
        log_In_tilde_d = np.logaddexp(
            log_In(K_THETA_u + 1, d * K_Z_u), log_In(K_THETA_u - 1, d * K_Z_u)
        ) - np.log(2)

        log_Kn_tilde_am = np.logaddexp(
            log_Kn(K_THETA_u + 1, am * K_Z_u), log_Kn(K_THETA_u - 1, am * K_Z_u)
        ) - np.log(2)
        log_Kn_tilde_bm = np.logaddexp(
            log_Kn(K_THETA_u + 1, bm * K_Z_u), log_Kn(K_THETA_u - 1, bm * K_Z_u)
        ) - np.log(2)
        log_Kn_tilde_c = np.logaddexp(
            log_Kn(K_THETA_u + 1, c * K_Z_u), log_Kn(K_THETA_u - 1, c * K_Z_u)
        ) - np.log(2)
        log_Kn_tilde_d = np.logaddexp(
            log_Kn(K_THETA_u + 1, d * K_Z_u), log_Kn(K_THETA_u - 1, d * K_Z_u)
        ) - np.log(2)
        log_Kn_tilde_b = np.logaddexp(
            log_Kn(K_THETA_u + 1, b * K_Z_u), log_Kn(K_THETA_u - 1, b * K_Z_u)
        ) - np.log(2)

        # Build A matrix (7x7 linear system per (kz, ktheta) pair)
        # Row 0
        A_mat[..., 0, 0] = 1
        A_mat[..., 0, 1] = -np.exp(log_In_am - log_In_bm)
        A_mat[..., 0, 2] = -np.exp(log_Kn_am - log_Kn_bm)

        # Row 1
        A_mat[..., 1, 0] = sig_bone * np.exp(log_In_tilde_a - log_In_a)
        A_mat[..., 1, 1] = -math.sqrt(sig_muscle_rho * sig_muscle_z) * np.exp(
            log_In_tilde_am - log_In_bm
        )
        A_mat[..., 1, 2] = math.sqrt(sig_muscle_rho * sig_muscle_z) * np.exp(
            log_Kn_tilde_am - log_Kn_bm
        )

        # Row 2
        A_mat[..., 2, 1] = 1
        A_mat[..., 2, 2] = 1
        A_mat[..., 2, 3] = -np.exp(log_In_b - log_In_c)
        A_mat[..., 2, 4] = -np.exp(log_Kn_b - log_Kn_c)

        # Row 3
        A_mat[..., 3, 1] = math.sqrt(sig_muscle_rho * sig_muscle_z) * np.exp(
            log_In_tilde_bm - log_In_bm
        )
        A_mat[..., 3, 2] = -math.sqrt(sig_muscle_rho * sig_muscle_z) * np.exp(
            log_Kn_tilde_bm - log_Kn_bm
        )
        A_mat[..., 3, 3] = -sig_fat * np.exp(log_In_tilde_b - log_In_c)
        A_mat[..., 3, 4] = sig_fat * np.exp(log_Kn_tilde_b - log_Kn_c)

        # Row 4
        A_mat[..., 4, 3] = 1
        A_mat[..., 4, 4] = 1
        A_mat[..., 4, 5] = -np.exp(log_In_c - log_In_d)
        A_mat[..., 4, 6] = -np.exp(log_Kn_c - log_Kn_d)

        # Row 5
        A_mat[..., 5, 3] = sig_fat * np.exp(log_In_tilde_c - log_In_c)
        A_mat[..., 5, 4] = -sig_fat * np.exp(log_Kn_tilde_c - log_Kn_c)
        A_mat[..., 5, 5] = -sig_skin * np.exp(log_In_tilde_c - log_In_d)
        A_mat[..., 5, 6] = sig_skin * np.exp(log_Kn_tilde_c - log_Kn_d)

        # Row 6
        log_diff_In_d = log_In_tilde_d - log_In_d
        log_diff_Kn_d = log_Kn_tilde_d - log_Kn_d
        A_mat[..., 6, 5] = sig_skin * np.exp(log_diff_In_d)
        A_mat[..., 6, 6] = -sig_skin * np.exp(log_diff_Kn_d)

        # Clean up numerical issues
        A_mat[np.isinf(A_mat)] = 0
        A_mat[np.isnan(A_mat)] = 0

        A_matrix = A_mat.copy()

    # --- Update B vector (depends on R, so not fully cacheable) ---
    log_In_am_b = log_In(K_THETA_u, am * K_Z_u)
    log_Kn_Rm = log_Kn(K_THETA_u, Rm * K_Z_u)
    log_Kn_bm_b = log_Kn(K_THETA_u, bm * K_Z_u)
    log_In_Rm = log_In(K_THETA_u, Rm * K_Z_u)

    log_In_tilde_am_b = np.logaddexp(
        log_In(K_THETA_u + 1, am * K_Z_u), log_In(K_THETA_u - 1, am * K_Z_u)
    ) - np.log(2)
    log_Kn_tilde_bm_b = np.logaddexp(
        log_Kn(K_THETA_u + 1, bm * K_Z_u), log_Kn(K_THETA_u - 1, bm * K_Z_u)
    ) - np.log(2)

    MAX_LOG_SAFE = 700.0

    # B[0,0] = In_am * Kn_Rm / sig_muscle_rho
    log_val_00 = log_In_am_b + log_Kn_Rm
    B[..., 0, 0] = np.where(
        log_val_00 < MAX_LOG_SAFE, np.exp(log_val_00), 0
    ) / sig_muscle_rho

    # B[1,0] = sqrt(sig_z/sig_rho) * In_tilde_am * Kn_Rm
    log_val_10 = log_In_tilde_am_b + log_Kn_Rm
    B[..., 1, 0] = math.sqrt(sig_muscle_z / sig_muscle_rho) * np.where(
        log_val_10 < MAX_LOG_SAFE, np.exp(log_val_10), 0
    )

    # B[2,0] = -Kn_bm * In_Rm / sig_muscle_rho
    log_val_20 = log_Kn_bm_b + log_In_Rm
    B[..., 2, 0] = -np.where(
        log_val_20 < MAX_LOG_SAFE, np.exp(log_val_20), 0
    ) / sig_muscle_rho

    # B[3,0] = sqrt(sig_z/sig_rho) * Kn_tilde_bm * In_Rm
    log_val_30 = log_Kn_tilde_bm_b + log_In_Rm
    B[..., 3, 0] = math.sqrt(sig_muscle_z / sig_muscle_rho) * np.where(
        log_val_30 < MAX_LOG_SAFE, np.exp(log_val_30), 0
    )

    B[np.isinf(B)] = 0
    B[np.isnan(B)] = 0

    # --- Solve the linear system ---
    A_flat = A_mat.reshape(-1, 7, 7)
    B_flat = B.reshape(-1, 7, 1)

    if r_bone == 0:
        # When no bone, muscle is innermost layer. Kn diverges at rho=0,
        # so B2 (col 2) must be zero. Keep cols [1,3,4,5,6].
        keep_cols = [1, 3, 4, 5, 6]
        A_flat = A_flat[..., 2:, keep_cols]
        B_flat = B_flat[..., 2:, :]
        X = np.linalg.solve(A_flat, B_flat)
        X = X.reshape(n_theta_u, n_z_u, 5, 1)
        H_vc_unique = X[..., 3, 0] + X[..., 4, 0]  # shape: (n_kt_unique, n_kz_nz)
    else:
        X = np.linalg.solve(A_flat, B_flat)
        X = X.reshape(n_theta_u, n_z_u, 7, 1)
        H_vc_unique = X[..., 5, 0] + X[..., 6, 0]

    # Map unique H_vc back to the full (kz, ktheta) grid.
    # H_vc is even in both kz and ktheta, so H_vc(kz, ktheta) = H_vc(|kz|, |ktheta|).
    # Build lookup: for each (i_kz, j_kt) in the full grid, find the index into
    # the unique arrays.
    kz_unique_all = np.concatenate(([0.0], kz_unique_nz))  # prepend DC
    # H_vc at DC is 0
    H_vc_with_dc = np.zeros((len(kt_unique), len(kz_unique_all)))
    H_vc_with_dc[:, 1:] = H_vc_unique  # DC column stays 0

    # Index maps: for each entry in k_z, find position in kz_unique_all
    kz_idx = np.searchsorted(kz_unique_all, kz_abs)
    kt_idx = np.searchsorted(kt_unique, kt_abs)

    # Build full H_vc array: shape (len(k_z), len(k_theta))
    H_vc = H_vc_with_dc[kt_idx, :][:, kz_idx].T  # transpose to (n_kz, n_kt)

    # =========================================================================
    # Section 4: H_ele(kz, k_theta)  — Electrode transfer function
    # =========================================================================
    ktheta_mesh, kz_mesh = np.meshgrid(k_theta, k_z)

    H_sf = electrode_array.get_H_sf(ktheta_mesh, kz_mesh)

    # Electrode size effect (circular electrode)
    arg = np.sqrt((rele * ktheta_mesh / r) ** 2 + (rele * kz_mesh) ** 2)
    H_size = 2 * np.divide(Jn(1, arg), arg)
    H_size[np.isnan(H_size)] = 1.0  # Jn(1,0)/0 -> 0.5, so 2*0.5 = 1

    H_ele = np.multiply(H_sf, H_size)

    # =========================================================================
    # Section 5: B(kz) — Angular integration
    # =========================================================================
    H_glo = np.multiply(H_vc, H_ele)

    k_theta_diff = k_theta[1] - k_theta[0]

    B_kz = np.zeros((channels[0], channels[1], len(k_z)))
    for ch_z in range(channels[0]):
        for ch_theta in range(channels[1]):
            integrand = np.multiply(
                H_glo,
                np.exp(1j * pos_theta_rad[ch_z, ch_theta] * ktheta_mesh)
                * k_theta_diff,
            )
            B_kz[ch_z, ch_theta, :] = np.sum(integrand, axis=1).real / (2 * math.pi)

    # =========================================================================
    # Section 6: IFFT B(kz) -> b(z)
    # =========================================================================
    # B_kz is in the fftshift convention (DC in center). Un-shift before IFFT.
    b_z = np.zeros((channels[0], channels[1], Nz))
    for ch_z in range(channels[0]):
        for ch_theta in range(channels[1]):
            B_kz_unshifted = np.fft.ifftshift(B_kz[ch_z, ch_theta, :])
            ifft_result = np.fft.ifft(B_kz_unshifted)
            assert np.allclose(ifft_result.imag, 0, atol=1e-10), (
                f"IFFT result has non-negligible imaginary part "
                f"(max |imag| = {np.max(np.abs(ifft_result.imag)):.2e})"
            )
            b_z[ch_z, ch_theta, :] = ifft_result.real

    return b_z, A_matrix


def simulate_fiber_unified(
    v: float,
    L1: float,
    L2: float,
    zi: float,
    b_z: np.ndarray,
    z_kernel: np.ndarray,
    electrode_z: np.ndarray,
    Fs: float,
    duration_ms: float,
    D1: float = 96.0,
) -> np.ndarray:
    """
    Simulate electrode potentials from a single fiber using traveling-wave convolution.

    The IAP is modeled as a traveling wave dVm/dz(z - vt) propagating in both
    directions from the endplate.  The electrode potential is computed via
    cross-correlation of the source current with the volume conductor kernel.

    Parameters
    ----------
    v : float
        Conduction velocity in mm/ms.
    L1 : float
        Semi-length from endplate to the right tendon in mm.
    L2 : float
        Semi-length from endplate to the left tendon in mm.
    zi : float
        Endplate offset from electrode center in mm.
    b_z : np.ndarray
        Volume conductor kernel, shape (n_channels, Nz).
    z_kernel : np.ndarray
        Spatial grid for the kernel in mm, shape (Nz,).
    electrode_z : np.ndarray
        Electrode z-positions in mm, shape (n_channels,).
    Fs : float
        Output sampling frequency in kHz.
    duration_ms : float
        Signal duration in ms.
    D1 : float, optional
        Rosenfalck amplitude parameter, default 96.0.

    Returns
    -------
    np.ndarray
        Electrode potentials, shape (n_channels, n_timepoints).
    """
    from scipy.interpolate import interp1d

    n_channels = b_z.shape[0]
    Nz = len(z_kernel)
    dz = z_kernel[1] - z_kernel[0]
    n_samples = int(duration_ms * Fs)
    t = np.arange(n_samples) / Fs  # time in ms

    # Rosenfalck spatial extent (effectively zero beyond ~10 mm)
    W = 10.0

    # Compute the source: dVm/dz on the kernel grid
    source = rosenfalck_dVm_dz(z_kernel, D1)

    # Output array
    phi = np.zeros((n_channels, n_samples))

    for ch in range(n_channels):
        kernel_ch = b_z[ch]

        # Cross-correlate source with kernel: h(s) = sum_u source(u) * kernel(u + s)
        # np.correlate(source, kernel, 'full') gives cross-correlation
        # Result length: 2*Nz - 1
        h = np.correlate(source, kernel_ch, mode='full') * dz
        # The s-axis for the correlate result spans [-(Nz-1)*dz, (Nz-1)*dz]
        n_h = len(h)
        s_axis = np.linspace(-(Nz - 1) * dz, (Nz - 1) * dz, n_h)

        # Build interpolator for h(s), zero outside range
        h_interp = interp1d(s_axis, h, kind='linear', bounds_error=False, fill_value=0.0)

        z0 = electrode_z[ch]

        # ---- Rightward wave (propagates toward right tendon) ----
        # s_right(t) = zi + v*t - z0
        # Propagating phase: 0 < v*t < L1 - W  (source fully inside fiber)
        # Extinction phase:  L1 - W < v*t < L1 + W  (source partially outside)
        # Post-extinction:   v*t > L1 + W  (source fully exited)
        phi_right = np.zeros(n_samples)

        vt = v * t
        s_right = zi + vt - z0

        # Phase boundaries for rightward wave
        prop_end_right = max(L1 - W, 0.0)
        ext_end_right = L1 + W

        # Masks for each phase
        mask_prop_right = vt <= prop_end_right
        mask_ext_right = (vt > prop_end_right) & (vt <= ext_end_right)
        # post-extinction: phi = 0, already initialized

        # Propagating phase: just sample h(s)
        if np.any(mask_prop_right):
            phi_right[mask_prop_right] = h_interp(s_right[mask_prop_right])

        # Extinction phase: truncated spatial integral
        # The source extends from position (vt) to (vt + W) relative to endplate,
        # but the fiber ends at L1. So we integrate from 0 to (L1 - vt) in source coords.
        if np.any(mask_ext_right):
            ext_indices = np.where(mask_ext_right)[0]
            for idx in ext_indices:
                # Truncation: fiber ends at L1, source starts at vt from endplate
                # In source coordinate u, the valid range is [0, L1 - vt[idx]]
                # but source is nonzero only for u in [0, W]
                trunc_len = L1 - vt[idx]
                if trunc_len <= 0:
                    continue
                # Integration grid in source coordinate u
                u_max = min(trunc_len, W)
                n_int = max(int(u_max / dz), 2)
                u_int = np.linspace(0, u_max, n_int)
                du = u_int[1] - u_int[0]
                src_vals = rosenfalck_dVm_dz(u_int, D1)
                # kernel evaluated at (u + zi + vt[idx] - z0) relative to z_kernel
                kernel_pos = u_int + zi + vt[idx] - z0
                # Interpolate kernel from z_kernel
                kernel_interp = interp1d(z_kernel, kernel_ch, kind='linear',
                                         bounds_error=False, fill_value=0.0)
                kernel_vals = kernel_interp(kernel_pos)
                phi_right[idx] = np.trapz(src_vals * kernel_vals, dx=du)

        # ---- Leftward wave (propagates toward left tendon) ----
        # s_left(t) = -zi - v*t + z0  (flipped direction)
        phi_left = np.zeros(n_samples)

        s_left = -zi - vt + z0

        # Phase boundaries for leftward wave
        prop_end_left = max(L2 - W, 0.0)
        ext_end_left = L2 + W

        mask_prop_left = vt <= prop_end_left
        mask_ext_left = (vt > prop_end_left) & (vt <= ext_end_left)

        # Propagating phase
        if np.any(mask_prop_left):
            phi_left[mask_prop_left] = h_interp(s_left[mask_prop_left])

        # Extinction phase for leftward wave
        if np.any(mask_ext_left):
            ext_indices = np.where(mask_ext_left)[0]
            for idx in ext_indices:
                trunc_len = L2 - vt[idx]
                if trunc_len <= 0:
                    continue
                u_max = min(trunc_len, W)
                n_int = max(int(u_max / dz), 2)
                u_int = np.linspace(0, u_max, n_int)
                du = u_int[1] - u_int[0]
                src_vals = rosenfalck_dVm_dz(u_int, D1)
                # For leftward wave, kernel position is flipped:
                # kernel at (-u - zi - vt[idx] + z0) = -(u + zi + vt[idx]) + z0
                kernel_pos = -u_int - zi - vt[idx] + z0
                kernel_interp = interp1d(z_kernel, kernel_ch, kind='linear',
                                         bounds_error=False, fill_value=0.0)
                kernel_vals = kernel_interp(kernel_pos)
                phi_left[idx] = np.trapz(src_vals * kernel_vals, dx=du)

        # Total: rightward minus leftward
        phi[ch] = phi_right - phi_left

    return phi
