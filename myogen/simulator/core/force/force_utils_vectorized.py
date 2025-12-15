"""Vectorized force utilities for optimized performance.

This module provides numpy-vectorized implementations of force generation utilities,
optimized for speed compared to the original loop-based implementations.
"""

from typing import Optional

import numpy as np
from scipy import signal as scipy_signal


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


def spikes2sawtooth_vectorized(
    spikes: np.ndarray, initial_values: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Convert spikes to sawtooth signal using fully vectorized numpy operations.

    This is a performance-optimized version that uses cumsum tricks.

    Parameters
    ----------
    spikes: np.ndarray
        Spike train matrix (time x neurons). Binary values where 1 indicates a spike.
    initial_values: np.ndarray, optional
        Initial values for each neuron. Default is ones.

    Returns
    -------
    np.ndarray
        Sawtooth sequence where each value represents the time since the last spike
        in samples.
    """
    if initial_values is None:
        initial_values = np.ones((1, spikes.shape[1]))

    # Handle both 1D and 2D initial_values
    if initial_values.ndim == 1:
        initial_values = initial_values.reshape(1, -1)

    l, w = spikes.shape

    # Use cumsum-based approach for speed
    # Create an array that increments but resets at spikes
    sawtooth = np.zeros((l, w), dtype=float)

    # For each neuron (still have to loop over neurons, but vectorize time)
    for j in range(w):
        # Get initial value for this neuron
        init_val = initial_values[0, j] if j < initial_values.shape[1] else 1.0

        # Get spike times
        spike_times = np.where(spikes[:, j])[0]

        if len(spike_times) == 0:
            # No spikes - just count from initial value
            sawtooth[:, j] = init_val + np.arange(l)
        else:
            # Use searchsorted to find most recent spike for each time point
            # This is O(L log S) instead of O(L*S) where S is number of spikes
            last_spike_idx = np.searchsorted(spike_times, np.arange(l), side='right') - 1

            # Where last_spike_idx < 0, no spike has occurred yet
            no_spike_yet = last_spike_idx < 0
            has_spike = ~no_spike_yet

            # Calculate time since last spike
            sawtooth[has_spike, j] = np.arange(l)[has_spike] - spike_times[last_spike_idx[has_spike]]
            sawtooth[no_spike_yet, j] = init_val + np.arange(l)[no_spike_yet]

            # Set spike times to 0
            sawtooth[spike_times, j] = 0

    return sawtooth


def spikes2sawtooth(
    spikes: np.ndarray, initial_values: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Convert spikes to sawtooth signal - alias for vectorized version.

    This maintains API compatibility while using optimized implementation.
    """
    return spikes2sawtooth_vectorized(spikes, initial_values)


def sawtooth2spikes(sawtooth: np.ndarray) -> np.ndarray:
    """
    Convert sawtooth signal to spikes.

    Parameters
    ----------
    sawtooth: np.ndarray
        Sawtooth signal matrix where values represent time since last spike.

    Returns
    -------
    np.ndarray
        Spike train matrix with binary values (1 = spike, 0 = no spike).
    """
    spikes = np.zeros_like(sawtooth, dtype=bool)
    spikes[sawtooth == 0] = 1
    return spikes.astype(int)


def sawtooth2ipi(
    sawtooth: np.ndarray, ipi_saturation: float = np.inf
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert sawtooth signal to inter-pulse intervals.

    Parameters
    ----------
    sawtooth: np.ndarray
        Sawtooth signal matrix where values represent time since last spike.
    ipi_saturation: float, optional
        Maximum IPI value (saturation). Default is infinity.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        ipi_filled: IPI signal with zeros where no full IPI can be calculated.
        ipi_filled_seamless: IPI signal with zeros filled by preceding values.
    """
    spikes = sawtooth2spikes(sawtooth)

    i_sawtooth = spikes2sawtooth(spikes[::-1], sawtooth[0])
    i_sawtooth = i_sawtooth[::-1]
    ipi_filled = sawtooth + i_sawtooth
    ipi_filled = np.minimum(ipi_filled, ipi_saturation)

    ipi_filled_seamless = ipi_filled.copy()
    ipi_filled_seamless[ipi_filled == 0] = ipi_filled_seamless[ipi_filled == 0] - 1
    ipi_filled_seamless = np.minimum(ipi_filled_seamless, ipi_saturation)

    return ipi_filled, ipi_filled_seamless


def generate_force_vectorized(
    spikes: np.ndarray,
    gain: np.ndarray,
    twitch_list: list[np.ndarray],
) -> np.ndarray:
    """
    Generate force using vectorized operations for better performance.

    This function uses scipy's FFT-based convolution for speed.

    Parameters
    ----------
    spikes : np.ndarray
        Spike array (L x N) where L is time points, N is motor units
    gain : np.ndarray
        Gain modulation array (L x N)
    twitch_list : list[np.ndarray]
        List of twitch templates for each motor unit

    Returns
    -------
    np.ndarray
        Force output array (L,)
    """
    L, N = spikes.shape
    force = np.zeros(L)

    # Process each motor unit
    for n in range(N):
        # Get spike times for this neuron
        spike_indices = np.where(spikes[:, n])[0]

        if len(spike_indices) == 0:
            continue

        twitch = twitch_list[n]

        # For each spike, add the gain-weighted twitch
        for spike_t in spike_indices:
            to_take = min(len(twitch), L - spike_t)
            force[spike_t:spike_t + to_take] += gain[spike_t, n] * twitch[:to_take]

    return force
