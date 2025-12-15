"""
Bioelectric functions for intramuscular EMG simulation.

This module contains the core bioelectric modeling functions for simulating
single fiber action potentials (SFAPs) and motor unit action potentials (MUAPs)
in intramuscular EMG. The functions implement the volume conductor models
from Farina et al. 2004 and the transmembrane current models from
Rosenfalck 1969.

References
----------
.. [1] Farina, D., Merletti, R., 2001. A novel approach for precise simulation of
       the EMG signal detected by surface electrodes. IEEE Transactions on
       Biomedical Engineering 48, 637–646.
.. [2] Rosenfalck, P., 1969. Intra- and extracellular potential fields of active
       nerve and muscle fibres. Acta Physiologica Scandinavica Supplementum 321, 1–168.
.. [3] Nandedkar, S.D., Stålberg, E., 1983. Simulation of single muscle fibre
       action potentials. Medical & Biological Engineering & Computing 21, 158–165.
"""

import numpy as np


def get_tm_current(z: np.ndarray, D1: float = 96.0, D2: float = -90.0) -> np.ndarray:
    """
    Calculate transmembrane current using Rosenfalck's model.

    This function implements the transmembrane current model from:
    P. Rosenfalck "Intra and extracellular fields of active nerve and muscle fibers" (1969)

    Parameters
    ----------
    z : np.ndarray
        Spatial coordinates along fiber in mm
    D1 : float, default=96.0
        Current amplitude parameter in mV/mm³
    D2 : float, default=-90.0
        Baseline potential in mV

    Returns
    -------
    np.ndarray
        Transmembrane potential in mV
    """
    Vm = np.full(z.shape, D2, dtype=np.float64)
    Vm[z > 0] = D1 * (z[z > 0] ** 3) * np.exp(-z[z > 0]) + D2
    return Vm


def get_tm_current_dz(z: np.ndarray, D1: float = 96.0) -> np.ndarray:
    """
    Calculate first derivative of transmembrane current (Rosenfalck model).

    This is the spatial derivative of the transmembrane current model used
    for action potential propagation simulation.

    Parameters
    ----------
    z : np.ndarray
        Spatial coordinates along fiber in mm
    D1 : float, default=96.0
        Current amplitude parameter in mV/mm³

    Returns
    -------
    np.ndarray
        First derivative of transmembrane current
    """
    Vm = np.zeros_like(z, dtype=np.float64)
    pos_mask = z > 0
    z_pos = z[pos_mask]
    Vm[pos_mask] = D1 * (3 * z_pos**2 - z_pos**3) * np.exp(-z_pos)
    return Vm


def get_tm_current_ddz(z: np.ndarray, D1: float = 96.0) -> np.ndarray:
    """
    Calculate second derivative of transmembrane current (Rosenfalck model).

    Parameters
    ----------
    z : np.ndarray
        Spatial coordinates along fiber in mm
    D1 : float, default=96.0
        Current amplitude parameter in mV/mm³

    Returns
    -------
    np.ndarray
        Second derivative of transmembrane current
    """
    Vm = np.zeros_like(z, dtype=np.float64)
    pos_mask = z > 0
    z_pos = z[pos_mask]
    Vm[pos_mask] = D1 * ((6 * z_pos - 3 * z_pos**2) - (3 * z_pos**2 - z_pos**3)) * np.exp(-z_pos)
    return Vm


def get_elementary_current_response(
    z: np.ndarray,
    z_electrode: float,
    r: np.ndarray,
    sigma_r: float = 63.0,  # S/m
    sigma_z: float = 330.0,  # S/m
) -> np.ndarray:
    """
    Calculate elementary current response for volume conductor.

    This function calculates the potential response at electrode location
    due to a unit current source at different positions along the muscle fiber.
    Based on Nandedkar & Stålberg 1983.

    Parameters
    ----------
    z : np.ndarray
        Longitudinal coordinates along fiber in mm
    z_electrode : float
        Electrode position along z-axis in mm
    r : np.ndarray
        Radial distance from fiber to electrode in mm
    sigma_r : float, default=63.0
        Radial conductivity in S/m (from Andreassen & Rosenfalck 1980)
    sigma_z : float, default=330.0
        Longitudinal conductivity in S/m (from Andreassen & Rosenfalck 1980)

    Returns
    -------
    np.ndarray
        Elementary current response (transfer function)
    """
    # ---- FIXED UNIT CONVERSIONS ----
    # Convert conductivities from S/m to S/mm to match spatial units (mm)
    sigma_r_S_per_mm = sigma_r / 1000.0  # CORRECTED: convert S/m → S/mm
    sigma_z_S_per_mm = sigma_z / 1000.0  # CORRECTED: convert S/m → S/mm

    return np.divide(
        1 / 4 / np.pi / sigma_r_S_per_mm,
        np.sqrt(sigma_z_S_per_mm / sigma_r_S_per_mm * r**2 + (z - z_electrode) ** 2),
    )


def shift_padding(vec, sh, axis):
    """
    Circularly shifts 'vec' by 'sh' positions along the specified 'axis'
    and then pads the shifted region with zeros.

    Parameters
    ----------
    vec : ndarray
        Input array to shift.
    sh : int
        Shift amount (positive means downward/rightward like MATLAB).
    axis : int
        Axis along which to shift.

    Returns
    -------
    ndarray
        Shifted and zero-padded array.
    """
    vec = np.roll(vec, sh, axis=axis)

    n = len(vec)

    # Equivalent of vec(1:sh) = 0
    if sh > 0:
        vec[:sh] = 0

    # Equivalent of vec(end+sh+1:end) = 0
    if sh < 0:
        start = n + sh  # because end+sh+1 in MATLAB is 1-based
        if start < n:
            vec[start:] = 0
    elif sh > 0:
        vec[-sh:] = 0

    return vec


def hr_shift_template(x, delay):
    """
    Shifts waveform x by a subsample step 'delay' using FFT-based phase shifting.

    Parameters:
        x (array-like): Input signal.
        delay (float): Fraction of the sampling period to delay (e.g. 0.1 means 1/10th).

    Returns:
        shifted (np.ndarray): Fractionally shifted signal.
    """
    x = np.asarray(x).flatten()

    # Pad if even length
    padded = False
    if len(x) % 2 == 0:
        x = np.append(x, 0)
        padded = True

    N = len(x)

    X = np.fft.fft(x)
    X0 = X[0]
    Xk = X[1 : int(np.ceil(N / 2))]

    k = np.arange(1, len(Xk) + 1)
    Sk = Xk * np.exp(1j * (2 * np.pi * delay) * k / N)
    S = np.concatenate(([X0], Sk, np.conj(Sk[::-1])))

    shifted = np.fft.ifft(S).real  # same as MATLAB, assumes real signal

    if padded:
        shifted = shifted[:-1]

    return shifted


def get_current_density(
    t, z, zi, L1, L2, v, d=55e-3, suppress_endplate_density=True, endplate_width=0.5
):
    """
    Model the individual action potential (IAP) or single fiber action potential (SFAP) in space and time.
    Translated from Farina & Merletti (2001) and Nandedkar & Stålberg (1998).

    Parameters
    ----------
    t : array
        Time vector
    z : array
        Spatial coordinates along the muscle fiber (in mm)
    zi : float
        Position of endplate (in mm)
    L1 : float
        Length of fiber from zi to positive end (mm)
    L2 : float
        Length of fiber from zi to negative end (mm)
    v : float
        Conduction speed in mm/s
    d : float, optional
        Fiber diameter in mm (default: 55e-3 mm = 55 um)
    suppress_endplate_density : bool, optional
        Whether to suppress density at endplate region (default: True)
    endplate_width : float, optional
        Width around endplate where density is suppressed (mm)
    """

    dz = np.mean(np.diff(z, axis=0))
    z = np.concatenate([z, z[[-1]] + dz], axis=0)

    T, Z = np.meshgrid(t, z)

    # Tendon terminator function
    def tendon_terminator(z_inline, L_inline):
        return (z_inline <= L_inline / 2) & (z_inline >= -L_inline / 2)

    # Compute psi (transmembrane current derivative)
    if L1 >= L2:
        psi = -4 * get_tm_current_dz(-2 * (Z - zi - v * T))
        longest_wave = np.diff(psi, axis=0) / dz
        longest_wave *= tendon_terminator(Z[:-1, :] - zi - L1 / 2, L1)
        longest_wave *= (Z[:-1, :] - zi) / v > 0  # negative time suppression
    else:
        psi = 4 * get_tm_current_dz(-2 * (-Z + zi - v * T))
        longest_wave = np.diff(psi, axis=0) / dz
        longest_wave *= tendon_terminator(Z[:-1, :] - zi + L2 / 2, L2)
        longest_wave *= (-Z[:-1, :] + zi) / v > 0

    # Shortest wave (reversed)
    shortest_wave = longest_wave[::-1].copy()
    shift_amount = int(np.round((L1 + L2 - max(z) + L2 - L1) / dz))
    shortest_wave = shift_padding(shortest_wave, shift_amount, 0)

    if L1 >= L2:
        shortest_wave *= tendon_terminator(Z[:-1, :] - zi + L2 / 2, L2)
        iap = longest_wave - shortest_wave
    else:
        shortest_wave *= tendon_terminator(Z[:-1, :] - zi - L1 / 2, L1)
        iap = shortest_wave - longest_wave

    # Suppress endplate density if required
    if suppress_endplate_density:

        def endplate_terminator(z_inline):
            return (z_inline <= (zi - endplate_width)) | (z_inline >= (zi + endplate_width))

        iap *= endplate_terminator(Z[:-1, :])

    # ---- FIXED UNIT CONVERSIONS ----
    # Intracellular conductivity: 1.01 S/m → convert to S/mm
    sigma_i_S_per_m = 1.01
    sigma_i = sigma_i_S_per_m / 1000.0  # S/mm (CORRECTED: was *1000, now /1000)

    # Fiber diameter is already in mm (default d=55e-3 mm = 55 um)
    # Compute cross-sectional area in mm²
    area_mm2 = np.pi * (d / 2) ** 2  # CORRECTED: removed extra /4

    # Scale current density by intracellular conductivity and fiber cross-section area
    iap *= sigma_i * area_mm2

    return iap


def get_current_density_fast(
    precalculated: np.ndarray,
    t: np.ndarray,
    z: np.ndarray,
    zi: float,
    L1: float,
    L2: float,
    v: float,
    d: float = 55e-6,
    suppress_endplate_density: bool = True,
    endplate_width: float = 0.5,
) -> np.ndarray:
    """
    Fast version of current density calculation using precalculated lookup table.

    This is an optimized version that uses a precalculated transmembrane current
    derivative lookup table to speed up computation for multiple fibers.

    Parameters
    ----------
    precalculated : np.ndarray
        Precalculated lookup table for get_tm_current_dz
    t : np.ndarray
        Time vector in seconds
    z : np.ndarray
        Spatial coordinates along muscle fiber in mm
    zi : float
        Position of endplate in mm
    L1 : float
        Length from endplate to positive tendon in mm
    L2 : float
        Length from endplate to negative tendon in mm
    v : float
        Conduction velocity in mm/s
    d : float, default=55e-6
        Fiber diameter in mm
    suppress_endplate_density : bool, default=True
        Whether to suppress endplate region
    endplate_width : float, default=0.5
        Endplate suppression width in mm

    Returns
    -------
    np.ndarray
        Current density matrix (space × time)
    """
    # This is a simplified version - full implementation would require
    # proper lookup table indexing and bounds checking
    # For now, fall back to the regular version
    return get_current_density(t, z, zi, L1, L2, v, d, suppress_endplate_density, endplate_width)


def calculate_sfap(
    electrode_position: np.ndarray,
    fiber_positions: np.ndarray,
    fiber_lengths: tuple[float, float],
    endplate_position: float,
    conduction_velocity: float,
    fiber_diameter: float,
    time_vector: np.ndarray,
    spatial_resolution: float = 0.5,
) -> np.ndarray:
    """
    Calculate Single Fiber Action Potential (SFAP) at electrode location.

    This is a high-level function that combines current density calculation
    with volume conductor modeling to compute the SFAP detected by an electrode.

    Parameters
    ----------
    electrode_position : np.ndarray
        3D position of electrode [x, y, z] in mm
    fiber_positions : np.ndarray
        3D positions along fiber [x, y, z] in mm (N × 3)
    fiber_lengths : tuple[float, float]
        Lengths (L1, L2) from endplate to each tendon in mm
    endplate_position : float
        Z-coordinate of endplate in mm
    conduction_velocity : float
        Fiber conduction velocity in mm/s
    fiber_diameter : float
        Fiber diameter in mm
    time_vector : np.ndarray
        Time points for simulation in seconds
    spatial_resolution : float, default=0.5
        Spatial sampling resolution in mm

    Returns
    -------
    np.ndarray
        SFAP signal at electrode location
    """
    L1, L2 = fiber_lengths

    # Create spatial grid along fiber
    z_min = min(fiber_positions[:, 2])
    z_max = max(fiber_positions[:, 2])
    z_fiber = np.arange(z_min, z_max + spatial_resolution, spatial_resolution)

    # Calculate current density along fiber
    current_density = get_current_density(
        time_vector,
        z_fiber,
        endplate_position,
        L1,
        L2,
        conduction_velocity,
        fiber_diameter,
    )

    # Calculate volume conductor response for each point along fiber
    sfap_signal = np.zeros(len(time_vector))

    for i, z_point in enumerate(z_fiber):
        # Find closest fiber position point
        distances = np.sqrt(
            np.sum(
                (fiber_positions - [electrode_position[0], electrode_position[1], z_point]) ** 2,
                axis=1,
            )
        )
        min_idx = np.argmin(distances)
        r_distance = distances[min_idx]

        # Calculate elementary response
        h_response = get_elementary_current_response(
            np.array([z_point]), electrode_position[2], np.array([r_distance])
        )

        # Convolve current with volume conductor response
        sfap_signal += current_density[i, :] * h_response[0] * spatial_resolution

    return sfap_signal
