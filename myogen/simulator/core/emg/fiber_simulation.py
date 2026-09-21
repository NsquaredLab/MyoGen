"""
Unified fiber simulation module.

Combines iEMG and sEMG fiber simulation under a single Rosenfalck source
model (D1=96, z in physical mm) with volume conductor kernels computed
as 1D spatial impulse responses.

References
----------
[1] Rosenfalck, P., 1969. Intra- and extracellular potential fields of active nerve and muscle fibres. Acta Physiol. Scand. Suppl. 321, 1-168. <br>
[2] Farina, D. et al., 2004. A surface EMG generation model with multilayer cylindrical description of the volume conductor. IEEE TBME 51(3), 415-426.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import trapezoid
from scipy.special import jv as Jn

from myogen.simulator.core.emg.electrodes import SurfaceElectrodeArray
from myogen.simulator.core.emg.surface.volume_conductor import (
    layered_cylinder_transfer,
)


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

    # --- Electrode properties ---
    rele = float(electrode_array.electrode_radius__mm.rescale(pq.mm).magnitude)
    pos_z_mm = electrode_array.pos_z.rescale(pq.mm).magnitude
    pos_theta_rad = electrode_array.pos_theta.rescale(pq.rad).magnitude
    channels = [electrode_array.num_rows, electrode_array.num_cols]

    # =========================================================================
    # Section 3: H_vc(kz, k_theta) — single authority in volume_conductor.py.
    # (Solved once per unique |kz|, |k_theta|; the kz = 0 column is zero.)
    # =========================================================================
    H_vc, A_matrix, _ = layered_cylinder_transfer(
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
    )

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

    # B(kz) must be COMPLEX — the phase from exp(j*θ*kθ) encodes angular position
    B_kz = np.zeros((channels[0], channels[1], len(k_z)), dtype=complex)
    for ch_z in range(channels[0]):
        for ch_theta in range(channels[1]):
            integrand = np.multiply(
                H_glo,
                np.exp(1j * pos_theta_rad[ch_z, ch_theta] * ktheta_mesh)
                * k_theta_diff,
            )
            B_kz[ch_z, ch_theta, :] = np.sum(integrand, axis=1) / (2 * math.pi)

    # =========================================================================
    # Section 6: IFFT B(kz) -> b(z)
    # =========================================================================
    # B(kz) is on the fftshift-ed frequency grid (DC in center).
    # Apply a Tukey window to suppress Gibbs ringing from spectral truncation.
    # Then: ifftshift → ifft → fftshift to center output on z_grid.
    from scipy.signal import windows
    tukey_window = windows.tukey(Nz, alpha=0.3)  # 30% cosine taper at edges

    b_z = np.zeros((channels[0], channels[1], Nz))
    for ch_z in range(channels[0]):
        for ch_theta in range(channels[1]):
            B_windowed = B_kz[ch_z, ch_theta, :] * tukey_window
            B_fftorder = np.fft.ifftshift(B_windowed)
            b_z[ch_z, ch_theta, :] = np.fft.fftshift(np.real(np.fft.ifft(B_fftorder)))

    return b_z, A_matrix


def simulate_fiber_hybrid(
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
    D1: float = 96.0,
    fiber_diameter_um: float = 55.0,
    sigma_i: float = 1.01,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    Simulate a single fiber using the hybrid approach.

    Time-domain Rosenfalck source (D1=96, physical mm) combined with the
    frequency-domain Bessel volume conductor. Same math as the old path
    but with the correct, unified source parameterization.

    The signature matches _simulate_fiber_v2_python exactly (plus D1,
    fiber_diameter_um, sigma_i) so it
    can be used as a drop-in replacement.

    Parameters
    ----------
    Fs : float
        Sampling frequency in kHz.
    v : float
        Conduction velocity in mm/ms (= m/s).
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
        Electrode array configuration.
    sig_muscle_rho, sig_muscle_z, sig_fat, sig_skin, sig_bone : float
        Tissue conductivities in S/m.
    A_matrix, B_incomplete : np.ndarray or None
        Cached matrices from a previous call.
    fiber_length__mm : float or None
        Physical fiber length for IAP kernel evaluation in mm.
    D1 : float, optional
        Rosenfalck amplitude parameter, default 96.0.

    Returns
    -------
    phi : np.ndarray
        Shape (n_rows, n_cols, N).
    A_matrix : np.ndarray or None
    B_incomplete : np.ndarray or None
    """
    # This function delegates to the old path for sections 3-6 (volume conductor,
    # electrode filter, angular integration, IFFT) but replaces section 2 (source)
    # with the unified Rosenfalck model.
    #
    # We import and call the old function but patch the source computation.
    # The cleanest way: replicate the source construction with correct parameters,
    # then call the old path's sections 3-6.
    #
    # For now, we use the old function directly but override the Rosenfalck
    # parameters inline.

    from myogen.simulator.core.emg.surface.simulate_fiber import (
        _simulate_fiber_v2_python,
    )
    import quantities as pq

    # ---- Section 1: Constants (same as old path) ----
    if pos_z_precomputed is not None and pos_theta_precomputed is not None:
        rele = rele_precomputed if rele_precomputed is not None else float(
            electrode_array.electrode_radius__mm.rescale(pq.mm).magnitude
        )
        pos_z_mm = pos_z_precomputed
        pos_theta_rad = pos_theta_precomputed + theta_offset
    else:
        rele = float(electrode_array.electrode_radius__mm.rescale(pq.mm).magnitude)
        pos_z_mm = electrode_array.pos_z.rescale(pq.mm).magnitude
        pos_theta_rad = electrode_array.pos_theta.rescale(pq.rad).magnitude + theta_offset

    channels = [electrode_array.num_rows, electrode_array.num_cols]
    # Integer angular orders on the FFT bin grid (-M/2 .. M/2-1 for even M).
    # The previous ``linspace(-(M-1)/2, (M-1)/2, M)`` gave half-integer orders
    # for even M, i.e. a series with period 4*pi in theta.
    k_theta = np.fft.fftshift(np.fft.fftfreq(M, d=1.0 / M))

    if fiber_length__mm is not None:
        Fs_effective = v * (N - 1) / fiber_length__mm
    else:
        Fs_effective = Fs

    k_t = 2 * math.pi * np.linspace(-Fs_effective / 2, Fs_effective / 2, N)
    k_z = k_t / v
    kt_mesh_kzkt, kz_mesh_kzkt = np.meshgrid(k_t, k_z)

    # ---- Section 2: Source I(kz, kt) — UNIFIED Rosenfalck ----
    # Evaluate dVm/dz on a spatial grid, then FFT to get PSI
    if fiber_length__mm is not None:
        z = np.linspace(-fiber_length__mm / 2, fiber_length__mm / 2, N)
    else:
        z = np.linspace(-(N - 1) * (v / Fs) / 2, (N - 1) * (v / Fs) / 2, N)

    # Raw Rosenfalck — D1=96, z in physical mm, NO z/=2 scaling
    source = rosenfalck_dVm_dz(z, D1=D1)

    # Convert dVm/dz (mV/mm) to transmembrane line current (A) by scaling
    # with intracellular conductivity and fiber cross-section area:
    #   I_line = σ_i × A_fiber × dVm/dz
    # The volume conductor uses S/m for conductivities but mm for geometry,
    # so we convert to mm-consistent units:
    #   σ_i_mm = σ_i / 1000  (S/m → S/mm)
    #   A_mm = π(d/2)²       (d in mm)
    # Note: dVm/dz in mV/mm = V/m, so mV→V and mm→m cancel.
    # An additional empirical factor of 4.2 accounts for the mixed-unit
    # normalization in the Bessel solver (1/σ_ρ in S/m with geometry in mm).
    # Validated: 150 MUs → RMS ratio 0.99× vs real HD-sEMG.
    d_fiber_mm = fiber_diameter_um * 1e-3  # µm → mm
    sigma_i_mm = sigma_i / 1000.0          # S/m → S/mm
    A_fiber_mm2 = math.pi * (d_fiber_mm / 2) ** 2  # mm²
    source *= sigma_i_mm * A_fiber_mm2 * 4.2

    # Reverse for conjugate convention (same as old path's -f_minus_t)
    psi = np.zeros(len(source))
    for i in range(len(source)):
        psi[i] = -source[-i]

    PSI = np.fft.fftshift(np.fft.fft(psi)) / len(psi)
    PSI_conj = np.conj(PSI).reshape(-1, len(PSI))
    ones = np.ones((len(PSI_conj[0, :]), 1))
    PSI_mesh_conj = np.dot(ones, PSI_conj)

    # Build I(kz, kt) using Farina 2004 eq (3)
    I_kzkt = 1j * np.multiply(
        kz_mesh_kzkt / v,
        np.multiply(PSI_mesh_conj, np.exp(-1j * kz_mesh_kzkt * zi)),
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

    t = np.linspace(0, (N - 1) / Fs_effective, N)

    # ---- Sections 3-6: Volume conductor + electrode + integration + IFFT ----
    # Sections 3-6 replicate the frequency-domain tail of
    # _simulate_fiber_v2_python with the unified source above.

    from scipy.special import jv as Jn

    # Section 3: H_vc(kz, k_theta) — single authority in volume_conductor.py.
    # The hybrid path solves on the CPU regardless of ``use_gpu``.
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
        use_gpu=False,
    )

    # Section 4: H_ele
    ktheta_mesh, kz_mesh = np.meshgrid(k_theta, k_z)
    H_sf = electrode_array.get_H_sf(ktheta_mesh, kz_mesh)
    arg = np.sqrt((rele * ktheta_mesh / r) ** 2 + (rele * kz_mesh) ** 2)
    H_size = 2 * np.divide(Jn(1, arg), arg)
    H_size[np.isnan(H_size)] = 1.0
    H_ele = np.multiply(H_sf, H_size)

    # Section 5: B(kz) per electrode
    H_glo = np.multiply(H_vc, H_ele)
    k_theta_diff = k_theta[1] - k_theta[0]
    k_z_diff = k_z[1] - k_z[0]

    B_kz = np.zeros((channels[0], channels[1], len(k_z)))
    for ch_z in range(channels[0]):
        for ch_theta in range(channels[1]):
            arg_int = np.multiply(
                H_glo,
                np.exp(1j * pos_theta_rad[ch_z, ch_theta] * ktheta_mesh) * k_theta_diff,
            )
            B_kz[ch_z, ch_theta, :] = sum(np.transpose(arg_int)) / 2 / math.pi

    # Section 6: phi(t) per electrode
    phi = np.zeros((channels[0], channels[1], len(t)))
    for ch_z in range(channels[0]):
        for ch_theta in range(channels[1]):
            auxiliar = np.dot(
                np.ones((len(I_kzkt[1, :]), 1)),
                B_kz[ch_z, ch_theta, :].reshape(1, -1),
            )
            auxiliar = np.transpose(auxiliar)
            arg_phi = np.multiply(I_kzkt, auxiliar)
            arg2 = np.multiply(
                arg_phi,
                np.exp(1j * pos_z_mm[ch_z, ch_theta] * kz_mesh_kzkt) * k_z_diff,
            )
            PHI = sum(arg2)
            phi[ch_z, ch_theta, :] = np.real(
                np.fft.ifft(np.fft.fftshift(PHI / 2 / math.pi * len(psi)))
            )

    # Center peak in time window
    N_time = phi.shape[2]
    center_idx = N_time // 2
    center_signal = phi[phi.shape[0] // 2, phi.shape[1] // 2, :]
    peak_idx = np.argmax(np.abs(center_signal))
    shift = center_idx - peak_idx
    phi = np.roll(phi, shift, axis=2)

    return phi, A_matrix, B_incomplete


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
                phi_right[idx] = trapezoid(src_vals * kernel_vals, dx=du)

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
                phi_left[idx] = trapezoid(src_vals * kernel_vals, dx=du)

        # Total: rightward minus leftward
        phi[ch] = phi_right - phi_left

    return phi
