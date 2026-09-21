"""Layered-cylinder volume conductor transfer function (Farina & Merletti 2004).

Single authority for the spatial transfer function ``H_vc(k_z, k_theta)`` of
the four-layer (bone / anisotropic muscle / fat / skin) cylindrical volume
conductor used by every surface-EMG fibre model in MyoGen:

* :func:`myogen.simulator.core.emg.surface.simulate_fiber._simulate_fiber_v2_python`
* :func:`myogen.simulator.core.emg.fiber_simulation.simulate_fiber_hybrid`
* :func:`myogen.simulator.core.emg.fiber_simulation.compute_surface_kernel`

The boundary-condition system is assembled in log-space to keep the modified
Bessel functions finite at large ``k_z * radius``; see the ``log_In`` /
``log_Kn`` helpers.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import ive as In_scaled  # ive(n, z) = iv(n, z) * exp(-|z|)
from scipy.special import kve as Kn_scaled  # kve(n, z) = kv(n, z) * exp(z)

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None  # type: ignore

# exp(700) ~ 1e304, just below the float64 maximum.
_MAX_LOG_SAFE = 700.0


def log_In(K_THETA, x):
    """Compute log(In(n, z)) in a numerically stable way."""
    # log(In(n,z)) = log(In_scaled(n,z)) + |z|
    In_s = In_scaled(K_THETA, x)
    # Handle zeros and negative values
    result = np.where(In_s > 0, np.log(In_s) + np.abs(x), -np.inf)
    return result


def log_Kn(K_THETA, x):
    """Compute log(Kn(n, z)) in a numerically stable way."""
    # log(Kn(n,z)) = log(Kn_scaled(n,z)) - z
    Kn_s = Kn_scaled(K_THETA, x)
    # Handle zeros and negative values
    result = np.where(Kn_s > 0, np.log(Kn_s) - x, -np.inf)
    return result


def _log_tilde(log_fn, K_THETA, x):
    """log of the derivative-like average (F_{n+1}(x) + F_{n-1}(x)) / 2."""
    return np.logaddexp(log_fn(K_THETA + 1, x), log_fn(K_THETA - 1, x)) - np.log(2)


def layered_cylinder_transfer(
    k_z: np.ndarray,
    k_theta: np.ndarray,
    *,
    r: float,
    r_bone: float,
    th_fat: float,
    th_skin: float,
    R: float,
    sig_muscle_rho: float,
    sig_muscle_z: float,
    sig_fat: float,
    sig_skin: float,
    sig_bone: float = 0.0,
    A_matrix: np.ndarray | None = None,
    B_incomplete: np.ndarray | None = None,
    use_gpu: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Surface potential transfer function of a fibre in a layered cylinder.

    Solves the Farina 2004 boundary-condition system for a line source at
    radius ``R`` inside the anisotropic muscle layer and returns the skin
    potential per unit source, ``H_vc(k_z, k_theta)``, on the full 2-D
    frequency grid.

    ``H_vc`` is even in both ``k_z`` and ``k_theta``, so the system is solved
    once per unique ``(|k_z|, |k_theta|)`` pair and mapped back onto the full
    grid.  This is exact for symmetric and asymmetric (FFT bin) grids alike
    and for odd or even grid lengths.  ``k_z = 0`` is never solved (the
    Bessel system is singular there); its column is returned as zero, which
    is consistent with a zero-mean line source.

    Parameters
    ----------
    k_z : np.ndarray
        Longitudinal spatial frequencies (rad/mm), any ordering.
    k_theta : np.ndarray
        Integer angular orders, any ordering.
    r, r_bone, th_fat, th_skin : float
        Outer radius, bone radius, fat and skin thickness (mm).  Muscle
        thickness is ``r - th_fat - th_skin - r_bone``.  ``r_bone = 0``
        selects the reduced five-unknown system without a bone layer.
    R : float
        Radial source position (mm).
    sig_muscle_rho, sig_muscle_z, sig_fat, sig_skin, sig_bone : float
        Conductivities (S/m).
    A_matrix, B_incomplete : np.ndarray, optional
        Cache returned by a previous call with the same grids and geometry.
        The system matrix does not depend on ``R``; pass both back in when
        iterating over fibres.
    use_gpu : bool
        Solve the batched linear systems with CuPy when available.

    Returns
    -------
    H_vc : np.ndarray
        Shape ``(len(k_z), len(k_theta))``.
    A_matrix, B_incomplete : np.ndarray
        Cache for subsequent calls.
    """
    k_z = np.asarray(k_z, dtype=np.float64)
    k_theta = np.asarray(k_theta, dtype=np.float64)

    # Model radii -> (Farina, 2004), Figure 1 - b)
    th_muscle = r - th_fat - th_skin - r_bone
    a = r_bone
    b = r_bone + th_muscle
    c = r_bone + th_muscle + th_fat
    d = r_bone + th_muscle + th_fat + th_skin

    anisotropy = math.sqrt(sig_muscle_z / sig_muscle_rho)
    am = a * anisotropy
    bm = b * anisotropy
    Rm = R * anisotropy

    # Unique |k| values.  The inverse indices map back onto the full grid.
    k_z_abs, k_z_inverse = np.unique(np.abs(k_z), return_inverse=True)
    k_theta_abs, k_theta_inverse = np.unique(np.abs(k_theta), return_inverse=True)
    has_dc = bool(k_z_abs.size) and k_z_abs[0] == 0.0
    k_z_solve = k_z_abs[1:] if has_dc else k_z_abs

    K_THETA, K_Z = np.meshgrid(k_theta_abs, k_z_solve, indexing="ij")
    n_theta, n_z = K_THETA.shape

    if A_matrix is not None:
        A_mat = A_matrix
        B = (
            B_incomplete.copy()
            if B_incomplete is not None
            else np.zeros((n_theta, n_z, 7, 1))
        )
    else:
        A_mat = np.zeros((n_theta, n_z, 7, 7))
        B = np.zeros((n_theta, n_z, 7, 1))

        # Log-space Bessel functions at every interface radius.  Ratios such as
        # In_am / In_bm become exp(log_In(am) - log_In(bm)).
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

        log_In_tilde_a = _log_tilde(log_In, K_THETA, a * K_Z)
        log_In_tilde_am = _log_tilde(log_In, K_THETA, am * K_Z)
        log_In_tilde_b = _log_tilde(log_In, K_THETA, b * K_Z)
        log_In_tilde_bm = _log_tilde(log_In, K_THETA, bm * K_Z)
        log_In_tilde_c = _log_tilde(log_In, K_THETA, c * K_Z)
        log_In_tilde_d = _log_tilde(log_In, K_THETA, d * K_Z)

        log_Kn_tilde_am = _log_tilde(log_Kn, K_THETA, am * K_Z)
        log_Kn_tilde_bm = _log_tilde(log_Kn, K_THETA, bm * K_Z)
        log_Kn_tilde_c = _log_tilde(log_Kn, K_THETA, c * K_Z)
        log_Kn_tilde_d = _log_tilde(log_Kn, K_THETA, d * K_Z)
        log_Kn_tilde_b = _log_tilde(log_Kn, K_THETA, b * K_Z)

        sig_muscle_geo = math.sqrt(sig_muscle_rho * sig_muscle_z)

        # Unknown 1 (muscle In coefficient) is normalised at the outer muscle
        # radius bm, unknown 2 (muscle Kn coefficient) at the inner radius am.
        # Kn grows inward, so normalising it at bm would put
        # Kn(am k)/Kn(bm k) ~ exp((bm - am) k) in the bone rows; past
        # (bm - am) k ~ 700 that overflows while In(am k)/In(bm k)
        # underflows, the two bone rows become proportional and the system is
        # singular (in practice: any bone radius below ~8 mm at the default
        # grid, and every bone radius at N >= 1024).  With the inner
        # normalisation the bone rows stay O(1) and the muscle/fat rows carry
        # the harmless underflow instead.  The skin unknowns are unaffected,
        # so H_vc is unchanged up to round-off.

        # Row 0/1: potential and radial current continuity at the bone.
        A_mat[..., 0, 0] = 1
        A_mat[..., 0, 1] = -np.exp(log_In_am - log_In_bm)
        A_mat[..., 0, 2] = -1
        A_mat[..., 1, 0] = sig_bone * np.exp(log_In_tilde_a - log_In_a)
        A_mat[..., 1, 1] = -sig_muscle_geo * np.exp(log_In_tilde_am - log_In_bm)
        A_mat[..., 1, 2] = sig_muscle_geo * np.exp(log_Kn_tilde_am - log_Kn_am)

        # Row 2/3: muscle / fat interface.
        A_mat[..., 2, 1] = 1
        A_mat[..., 2, 2] = np.exp(log_Kn_bm - log_Kn_am)
        A_mat[..., 2, 3] = -np.exp(log_In_b - log_In_c)
        A_mat[..., 2, 4] = -np.exp(log_Kn_b - log_Kn_c)
        A_mat[..., 3, 1] = sig_muscle_geo * np.exp(log_In_tilde_bm - log_In_bm)
        A_mat[..., 3, 2] = -sig_muscle_geo * np.exp(log_Kn_tilde_bm - log_Kn_am)
        A_mat[..., 3, 3] = -sig_fat * np.exp(log_In_tilde_b - log_In_c)
        A_mat[..., 3, 4] = sig_fat * np.exp(log_Kn_tilde_b - log_Kn_c)

        # Row 4/5: fat / skin interface.
        A_mat[..., 4, 3] = 1
        A_mat[..., 4, 4] = 1
        A_mat[..., 4, 5] = -np.exp(log_In_c - log_In_d)
        A_mat[..., 4, 6] = -np.exp(log_Kn_c - log_Kn_d)
        A_mat[..., 5, 3] = sig_fat * np.exp(log_In_tilde_c - log_In_c)
        A_mat[..., 5, 4] = -sig_fat * np.exp(log_Kn_tilde_c - log_Kn_c)
        A_mat[..., 5, 5] = -sig_skin * np.exp(log_In_tilde_c - log_In_d)
        A_mat[..., 5, 6] = sig_skin * np.exp(log_Kn_tilde_c - log_Kn_d)

        # Row 6: insulating outer boundary.
        A_mat[..., 6, 5] = sig_skin * np.exp(log_In_tilde_d - log_In_d)
        A_mat[..., 6, 6] = -sig_skin * np.exp(log_Kn_tilde_d - log_Kn_d)

        A_mat[np.isinf(A_mat)] = 0
        A_mat[np.isnan(A_mat)] = 0

        A_matrix = A_mat.copy()
        B_incomplete = B.copy()

    # Source vector (depends on R, so never cached).
    log_In_am = log_In(K_THETA, am * K_Z)
    log_Kn_Rm = log_Kn(K_THETA, Rm * K_Z)
    log_Kn_bm = log_Kn(K_THETA, bm * K_Z)
    log_In_Rm = log_In(K_THETA, Rm * K_Z)
    log_In_tilde_am = _log_tilde(log_In, K_THETA, am * K_Z)
    log_Kn_tilde_bm = _log_tilde(log_Kn, K_THETA, bm * K_Z)

    def _safe_exp(log_val):
        return np.where(log_val < _MAX_LOG_SAFE, np.exp(log_val), 0)

    B[..., 0, 0] = _safe_exp(log_In_am + log_Kn_Rm) / sig_muscle_rho
    B[..., 1, 0] = anisotropy * _safe_exp(log_In_tilde_am + log_Kn_Rm)
    B[..., 2, 0] = -_safe_exp(log_Kn_bm + log_In_Rm) / sig_muscle_rho
    B[..., 3, 0] = anisotropy * _safe_exp(log_Kn_tilde_bm + log_In_Rm)

    B[np.isinf(B)] = 0
    B[np.isnan(B)] = 0

    A_flat = A_mat.reshape(-1, 7, 7)
    B_flat = B.reshape(-1, 7, 1)

    if r_bone == 0:
        # No bone: the muscle is the innermost layer and Kn diverges at
        # rho = 0, so the Kn_muscle coefficient (column 2) must vanish.  Drop
        # the two bone rows and keep [In_muscle, In_fat, Kn_fat, In_skin,
        # Kn_skin].
        keep_cols = [1, 3, 4, 5, 6]
        A_flat = A_flat[..., 2:, keep_cols]
        B_flat = B_flat[..., 2:, :]
        n_unknowns = 5
    else:
        n_unknowns = 7

    if HAS_CUPY and use_gpu:
        A_gpu = cp.asarray(A_flat)
        B_gpu = cp.asarray(B_flat)
        X = cp.asnumpy(cp.linalg.solve(A_gpu, B_gpu))
        del A_gpu, B_gpu
    else:
        X = np.linalg.solve(A_flat, B_flat)

    X = X.reshape(n_theta, n_z, n_unknowns, 1)
    # Skin potential = In_skin + Kn_skin coefficients (last two unknowns).
    H_vc_unique = X[..., n_unknowns - 2, 0] + X[..., n_unknowns - 1, 0]

    if has_dc:
        H_vc_unique = np.concatenate((np.zeros((n_theta, 1)), H_vc_unique), axis=1)

    # Map H_vc(|k_z|, |k_theta|) back onto the full (k_z, k_theta) grid.
    H_vc = H_vc_unique.T[np.ix_(k_z_inverse, k_theta_inverse)]
    return H_vc, A_matrix, B_incomplete
