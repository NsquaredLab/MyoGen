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

import numpy as np


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
