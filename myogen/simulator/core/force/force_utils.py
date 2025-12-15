from typing import Optional

import numpy as np
import scipy.sparse as sp


def get_gain_vectorized(ipi: np.ndarray, T__samples: np.ndarray) -> np.ndarray:
    """
    ipi: shape (L, N) - all neurons
    T__samples: shape (N,) - one value per neuron
    """
    Sf_const = (1 - np.exp(-2 * 0.4 ** 3)) / 0.4  # Pre-compute constant

    inst_dr = T__samples[None, :] / ipi  # Broadcast: (1, N) / (L, N) -> (L, N)
    gain = np.ones_like(inst_dr)

    mask = inst_dr > 0.4
    Sf_values = 1 - np.exp(-2 * inst_dr[mask] ** 3)
    gain[mask] = (Sf_values / inst_dr[mask]) / Sf_const

    return gain


def get_gain(ipi: np.ndarray, T__samples: float) -> np.ndarray:
    """
    Returns the gain value for the force output for a motor unit with current
    inter-pulse-interval ipi and T-parameter of the twitch. This function
    corresponds to Fuglevand's nonlinear gain model for the force output,
    see Fuglevand - Models of Rate Coding..., eq. 17.

    Parameters
    ----------
    ipi : np.ndarray
        Inter-pulse interval vector in samples.
    T__samples : float
        T-parameter of the twitch in samples (contraction time).

    Returns
    -------
    np.ndarray
        Gain vector with nonlinear modulation based on discharge rate.
    """
    Sf = lambda x: 1 - np.exp(-2 * x**3)

    inst_dr = T__samples / ipi  # Instantaneous discharge rate
    gain = np.ones_like(inst_dr)  # Gain

    mask = inst_dr > 0.4
    gain[mask] = (Sf(inst_dr[mask]) / inst_dr[mask]) / (Sf(0.4) / 0.4)

    return gain


def spikes2sawtooth(
    spikes, initial_values: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Convert spikes to sawtooth signal.

    The sawtooth signal represents inter-pulse intervals (IPIs) where each sawtooth
    tooth height equals the number of time steps between consecutive spikes. This is
    used to calculate instantaneous discharge rates for force generation.

    **Sawtooth Sizing:**
        - Each time step without a spike: sawtooth value increments by 1
        - When a spike occurs: sawtooth resets to 0
        - Sawtooth height = IPI in time samples (at recording_frequency__Hz)
        - Example: height of 50 at 10kHz = 5ms IPI

    Parameters:
    -----------
    spikes: np.ndarray
        Spike train matrix (time x neurons). Binary values where 1 indicates a spike.
    initial_values: np.ndarray, optional
        Initial values for each neuron. Default is ones. Will be reset to 0 if
        there's a spike at t=0.

    Returns:
    --------
    np.ndarray
        Sawtooth sequence where each value represents the time since the last spike
        in samples. Used for calculating inter-pulse intervals and gain modulation.
    """
    # Convert sparse to dense if needed
    if sp.issparse(spikes):
        spikes = spikes.toarray()

    if initial_values is None:
        initial_values = np.ones((1, spikes.shape[1]))

    l, w = spikes.shape
    seq = np.zeros((l, w), dtype=np.uint32)

    # Set initial values, but reset to 0 if there's a spike at t=0
    initial_values = initial_values * (spikes[0] != 1)
    seq[0] = initial_values

    # Vectorized approach using cumulative operations
    # Create cumulative counter that increments by 1 at each timestep
    counter = np.arange(l, dtype=np.uint32)[:, None]

    # Find indices where spikes occur and get the counter value at those positions
    # Broadcast spike positions to get reset points
    # Convert spikes to uint32 to prevent dtype promotion to float64
    spike_positions = spikes.astype(np.uint32) * counter

    # Get the maximum counter value up to each point where a spike occurred
    # This gives us the "reset" value for the cumulative counter
    reset_values = np.maximum.accumulate(spike_positions, axis=0)

    # Calculate time since last spike by subtracting reset value from current counter
    seq[1:] = counter[1:] - reset_values[:-1]

    # Set spike positions explicitly to 0
    seq[spikes.astype(bool)] = 0

    return seq


def sawtooth2spikes(sawtooth: np.ndarray) -> np.ndarray:
    """
    Convert sawtooth signal to spikes.

    A spike occurs when the sawtooth resets to 0, indicating the end of an
    inter-pulse interval. This is the inverse operation of spikes2sawtooth.

    **Spike Detection:**
        - Spike at t=0 if sawtooth starts at 0
        - Spike when sawtooth decreases (resets from peak back to 0)
        - Each spike marks the end of one IPI and start of the next

    Parameters:
    -----------
    sawtooth: np.ndarray
        Sawtooth signal matrix where values represent time since last spike.

    Returns:
    --------
    np.ndarray
        Spike train matrix with binary values (1 = spike, 0 = no spike).
    """
    spikes = np.zeros_like(sawtooth, dtype=bool)

    spikes[sawtooth == 0] = 1

    return spikes.astype(int)


def sawtooth2ipi(
    sawtooth: np.ndarray, spikes, ipi_saturation: float = np.inf
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert sawtooth signal to inter-pulse intervals.

    This method converts sawtooth heights (representing time since last spike) into
    inter-pulse intervals (IPIs) by combining forward and backward spike information.
    The IPI values are used to calculate instantaneous discharge rates for the
    Fuglevand gain model.

    **IPI Calculation:**
    - Forward sawtooth: time since last spike
    - Backward sawtooth: time until next spike
    - IPI = forward + backward sawtooth values
    - IPI represents the full inter-pulse interval in time samples
    - Convert to physiological units: IPI_ms = IPI_samples / (recording_frequency__Hz / 1000)

    Parameters:
    -----------
    sawtooth: np.ndarray
        Sawtooth signal matrix where values represent time since last spike.
    ipi_saturation: float, optional
        Maximum IPI value (saturation) to prevent infinite values when spikes
        are far apart. Default is infinity.

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        ipi_filled: IPI signal with zeros where no full IPI can be calculated.
        ipi_filled_seamless: IPI signal with zeros filled by preceding values
        for smooth gain calculations.
    """
    i_sawtooth = spikes2sawtooth(spikes[::-1], sawtooth[0])
    i_sawtooth = i_sawtooth[::-1]
    ipi_filled = sawtooth + i_sawtooth
    ipi_filled = np.minimum(ipi_filled, ipi_saturation)

    ipi_filled_seamless = ipi_filled.copy()
    ipi_filled_seamless[ipi_filled == 0] = ipi_filled_seamless[ipi_filled == 0] - 1
    ipi_filled_seamless = np.minimum(ipi_filled_seamless, ipi_saturation)

    return ipi_filled, ipi_filled_seamless
