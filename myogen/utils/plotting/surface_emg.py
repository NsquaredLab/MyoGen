from typing import Any, Optional, Union

import numpy as np
from matplotlib.axes import Axes
from tqdm import tqdm

from myogen.utils.types import SURFACE_EMG__Block, SURFACE_MUAP__Block
from myogen.utils.decorators import beartowertype


def _get_axis(axes, row_idx: int, col_idx: int, n_rows: int, n_cols: int):
    """Helper function to safely get axis from matplotlib subplots."""
    if n_rows == 1 and n_cols == 1:
        return axes
    elif n_rows == 1:
        return axes[col_idx]
    elif n_cols == 1:
        return axes[row_idx]
    else:
        return axes[row_idx, col_idx]


def _auto_zoom_muaps(
    muap_data: np.ndarray,
    threshold: float = 0.01,
    padding: float = 0.1,
    center: bool = True,
) -> np.ndarray:
    """
    Auto-zoom MUAPs by centering them and cropping to significant regions.

    Based on the algorithm from check_generated_surface_muaps.py.

    Parameters
    ----------
    muap_data : np.ndarray
        MUAP data with shape (n_muaps, n_rows, n_cols, n_time)
    threshold : float
        Threshold as fraction of max amplitude for significant regions
    padding : float
        Padding as fraction of signal length to add around significant regions
    center : bool
        Whether to center MUAPs temporally before cropping

    Returns
    -------
    np.ndarray
        Processed MUAP data with same shape but potentially fewer time samples
    """
    n_muaps, n_rows, n_cols, n_time = muap_data.shape
    processed_data = muap_data.copy()

    if center:
        # Center each MUAP around its centroid
        for muap_idx in range(n_muaps):
            # Find the center of mass for this MUAP across all electrodes
            muap_abs = np.abs(processed_data[muap_idx])

            # Calculate center of mass for each electrode
            centers = np.zeros((n_rows, n_cols))
            for row_idx in range(n_rows):
                for col_idx in range(n_cols):
                    signal = muap_abs[row_idx, col_idx]
                    if np.sum(signal) > 0:
                        # Center of mass calculation
                        time_indices = np.arange(n_time)
                        centers[row_idx, col_idx] = np.average(
                            time_indices, weights=signal
                        )
                    else:
                        centers[row_idx, col_idx] = n_time // 2

            # Use mean center across all electrodes
            mean_center = int(np.mean(centers))
            shift = n_time // 2 - mean_center

            # Apply shift to center the MUAP
            if shift != 0:
                shifted_muap = np.zeros_like(processed_data[muap_idx])
                for row_idx in range(n_rows):
                    for col_idx in range(n_cols):
                        if shift > 0:
                            # Shift right
                            shifted_muap[row_idx, col_idx, shift:] = processed_data[
                                muap_idx, row_idx, col_idx, :-shift
                            ]
                        elif shift < 0:
                            # Shift left
                            shifted_muap[row_idx, col_idx, :shift] = processed_data[
                                muap_idx, row_idx, col_idx, -shift:
                            ]
                        else:
                            shifted_muap[row_idx, col_idx] = processed_data[
                                muap_idx, row_idx, col_idx
                            ]

                processed_data[muap_idx] = shifted_muap

    # Find significant regions across all MUAPs and crop
    max_amplitude = np.max(np.abs(processed_data))
    threshold_value = threshold * max_amplitude

    # Find where any MUAP exceeds threshold
    significant_mask = np.abs(processed_data) > threshold_value
    significant_indices = np.where(significant_mask)[-1]  # Get time dimension indices

    if len(significant_indices) > 0:
        start_idx = int(np.min(significant_indices))
        end_idx = int(np.max(significant_indices))

        # Add padding
        padding_samples = int(padding * n_time)
        start_idx = max(0, start_idx - padding_samples)
        end_idx = min(n_time, end_idx + padding_samples)

        # Crop all MUAPs to this region
        processed_data = processed_data[..., start_idx:end_idx]

    return processed_data


@beartowertype
def plot_surface_emg(
    surface_emg__tensor: SURFACE_EMG__Block,
    axs: list[Union[Axes, np.ndarray]],
    apply_default_formatting: bool = True,
    **kwargs: Any,
) -> list[Union[Axes, np.ndarray]]:
    """
    Plot the EMG signal across electrode grids.

    Parameters
    ----------
    surface_emg__tensor : SURFACE_EMG__TENSOR
        Tensor of shape (n_pools, n_rows, n_cols, n_time) containing EMG signals
    axs : list[Union[Axes, np.ndarray]]
        Matplotlib axes to plot on. Should provide one set of axes per pool.
        Each set can be a 2D array of axes (from plt.subplots), a single axis, or a 1D array.
        Expected structure: list of axes configurations, one per pool.
    apply_default_formatting : bool, default=True
        Whether to apply default formatting to the plot
    **kwargs : dict
        Additional keyword arguments to pass to the plot function. Only used if apply_default_formatting is False.

    Returns
    -------
    list[Union[Axes, np.ndarray]]
        The axes that were plotted on

    Raises
    ------
    ValueError
        If the number of axes does not match the number of pools
    """
    axs_list = list(axs)
    n_pools = surface_emg__tensor.shape[0]

    if len(axs_list) != n_pools:
        raise ValueError(
            f"Number of axes must match number of pools. Got {len(axs_list)} axes, but {n_pools} pools."
        )

    n_rows = surface_emg__tensor.shape[1]
    n_cols = surface_emg__tensor.shape[2]

    for pool_idx, pool_axes in enumerate(axs_list):
        # Handle the case where pool_axes is a single axis or array of axes
        if hasattr(pool_axes, "flat") and not isinstance(pool_axes, Axes):
            # pool_axes is a 2D array of axes
            axes_flat: Any = pool_axes.flat
        elif isinstance(pool_axes, Axes):
            # pool_axes is a single axis
            axes_flat: Any = [pool_axes]
        else:
            # pool_axes is a 1D array or other iterable
            axes_flat: Any = pool_axes

        for row_idx in range(n_rows):
            for col_idx in range(n_cols):
                electrode_idx = row_idx * n_cols + col_idx
                if electrode_idx < len(axes_flat):
                    ax = axes_flat[electrode_idx]

                    plot_kwargs = kwargs.copy() if not apply_default_formatting else {}

                    if apply_default_formatting:
                        ax.plot(surface_emg__tensor[pool_idx, row_idx, col_idx])
                        ax.set_title(f"Pool {pool_idx + 1} - R{row_idx} C{col_idx}")
                        ax.set_xlabel("Time (samples)")
                        ax.set_ylabel("Amplitude")
                    else:
                        ax.plot(
                            surface_emg__tensor[pool_idx, row_idx, col_idx],
                            **plot_kwargs,
                        )

    return axs


@beartowertype
def plot_muap_grid(
    surface_muap__Block: SURFACE_MUAP__Block,
    axs: list[Union[Axes, np.ndarray]],
    muap_indices: Optional[list[int]] = None,
    time_slice: Optional[slice] = None,
    apply_default_formatting: bool = True,
    **kwargs: Any,
) -> list[Union[Axes, np.ndarray]]:
    """
    Plot Motor Unit Action Potentials (MUAPs) in electrode grid format.

    This function visualizes MUAPs as they appear across a grid of electrodes,
    with each subplot showing the MUAP waveform at a specific electrode position.
    The layout matches the physical electrode arrangement.

    Parameters
    ----------
    surface_muap__Block : SURFACE_MUAP__Block
        Neo Block object returned by SurfaceEMG.simulate_muaps() containing MUAP data.
    axs : list[Union[Axes, np.ndarray]]
        Matplotlib axes to plot on. Should provide one set of axes per MUAP.
        Each set can be a 2D array of axes (from plt.subplots), a single axis, or a 1D array.
        Expected structure: list of axes configurations, one per MUAP to plot.
    muap_indices : list[int], optional
        List of MUAP indices to plot. If None, plots all MUAPs.
    time_slice : slice, optional
        Time slice to apply to the MUAP data (e.g., slice(100, -100) to remove first
        and last 100 samples). If None, uses the full time series.
    apply_default_formatting : bool, default=True
        Whether to apply default formatting to the plot
    **kwargs : dict
        Additional keyword arguments to pass to the plot function. Only used if apply_default_formatting is False.

    Returns
    -------
    list[Union[Axes, np.ndarray]]
        The axes that were plotted on

    Raises
    ------
    ValueError
        If the number of axes does not match the number of MUAPs to plot
    """
    # Convert Block object to numpy array
    muap_arrays = []
    for group in surface_muap__Block.groups:
        for segment in group.segments:
            for analogsignal in segment.analogsignals:
                # Convert GridAnalogSignal to numpy array in grid format
                grid_data = analogsignal.magnitude  # This gives (time, rows, cols)
                muap_arrays.append(grid_data)

    if not muap_arrays:
        raise ValueError("No MUAP data found in the provided Block object")

    # Stack all MUAPs: shape becomes (n_muaps, time, rows, cols)
    muap_data_array = np.stack(muap_arrays, axis=0)
    # Reorder to match expected format: (n_muaps, rows, cols, time)
    muap_array = np.transpose(muap_data_array, (0, 2, 3, 1))

    # Apply time slicing if specified
    if time_slice is not None:
        muap_array = muap_array[..., time_slice]

    # Handle single MUAP case by adding a dimension
    if muap_array.ndim == 3:
        muap_array = muap_array[np.newaxis, ...]

    # Validate input dimensions
    if muap_array.ndim != 4:
        raise ValueError(
            f"muap_data must have 3 or 4 dimensions, got {muap_array.ndim}. "
            f"Expected shape: (n_muaps, n_rows, n_cols, n_time) or (n_rows, n_cols, n_time)"
        )

    n_muaps, n_rows, n_cols, n_time = muap_array.shape

    # Set default MUAP indices if not provided
    if muap_indices is None:
        muap_indices = list(range(n_muaps))

    # Validate MUAP indices
    invalid_indices = [idx for idx in muap_indices if idx >= n_muaps or idx < 0]
    if invalid_indices:
        raise ValueError(
            f"Invalid MUAP indices: {invalid_indices}. Must be in range [0, {n_muaps - 1}]"
        )

    axs_list = list(axs)
    if len(axs_list) != len(muap_indices):
        raise ValueError(
            f"Number of axes must match number of MUAPs to plot. Got {len(axs_list)} axes, but {len(muap_indices)} MUAPs."
        )

    # Plot each requested MUAP
    for i, muap_idx in enumerate(
        tqdm(muap_indices, desc="Plotting MUAPs", unit="MUAP")
    ):
        muap_axes = axs_list[i]

        # Handle the case where muap_axes is a single axis or array of axes
        if hasattr(muap_axes, "flat") and not isinstance(muap_axes, Axes):
            # muap_axes is a 2D array of axes
            axes_flat: Any = muap_axes.flat
        elif isinstance(muap_axes, Axes):
            # muap_axes is a single axis
            axes_flat: Any = [muap_axes]
        else:
            # muap_axes is a 1D array or other iterable
            axes_flat: Any = muap_axes

        # Calculate global y-limits for consistent scaling across electrodes
        # Use nanmin/nanmax to handle NaN values robustly
        muap_min = np.nanmin(muap_array[muap_idx])
        muap_max = np.nanmax(muap_array[muap_idx])

        # Handle edge cases where all values are NaN or min/max are invalid
        if (
            np.isnan(muap_min)
            or np.isnan(muap_max)
            or np.isinf(muap_min)
            or np.isinf(muap_max)
        ):
            # Fallback to reasonable defaults
            muap_min, muap_max = -1.0, 1.0
        elif muap_min == muap_max:
            # Handle case where all values are identical (avoid zero range)
            if muap_min == 0:
                muap_min, muap_max = -0.1, 0.1
            else:
                margin = abs(muap_min) * 0.1
                muap_min -= margin
                muap_max += margin

        # Plot MUAP at each electrode position
        for row_idx in range(n_rows):
            for col_idx in range(n_cols):
                electrode_idx = row_idx * n_cols + col_idx
                if electrode_idx < len(axes_flat):
                    ax = axes_flat[electrode_idx]

                    plot_kwargs = kwargs.copy() if not apply_default_formatting else {}

                    if apply_default_formatting:
                        # Plot MUAP waveform
                        ax.plot(muap_array[muap_idx, row_idx, col_idx])

                        # Set consistent y-limits across all electrodes
                        ax.set_ylim(muap_min, muap_max)

                        # Clean up axes for better visualization
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.spines["top"].set_visible(False)
                        ax.spines["right"].set_visible(False)
                        ax.spines["left"].set_visible(False)
                        ax.spines["bottom"].set_visible(False)
                    else:
                        ax.plot(muap_array[muap_idx, row_idx, col_idx], **plot_kwargs)

    return axs
