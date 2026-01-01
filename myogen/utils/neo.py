"""
Neo utilities for enhanced signal handling.

This module provides utilities for working with Neo signals,
including grid-based electrode array handling via annotations.
"""

from typing import Literal

import numpy as np
import quantities as pq
from neo import AnalogSignal

from .decorators import beartowertype


@beartowertype
def create_grid_signal(
    signal: np.ndarray,
    grid_shape: tuple[int, int],
    sampling_rate: pq.Quantity,
    units: pq.Quantity | str = pq.mV,
    t_start: pq.Quantity = 0 * pq.s,
    electrode_positions: list[tuple[float, float]] | None = None,
    ied: float | None = None,
    **kwargs,
) -> AnalogSignal:
    """
    Create an AnalogSignal with grid metadata annotations for electrode arrays.

    This function creates a standard Neo AnalogSignal with grid structure stored
    in annotations, making it NWB-compatible while preserving spatial information.

    Parameters
    ----------
    signal : np.ndarray
        Signal data with shape (time, rows, cols) or (time, n_electrodes).
        If 3D, will be flattened to (time, n_electrodes) for storage.
    grid_shape : tuple[int, int]
        Shape of the electrode grid as (rows, cols).
    sampling_rate : pq.Quantity
        Sampling rate of the signal.
    units : pq.Quantity or str, default=pq.mV
        Units of the signal.
    t_start : pq.Quantity, default=0*pq.s
        Start time of the signal.
    electrode_positions : list[tuple[float, float]], optional
        Physical (x, y) positions of each electrode in mm. If None,
        positions are computed from grid_shape and ied.
    ied : float, optional
        Inter-electrode distance in mm. Used to compute electrode_positions
        if not provided directly.
    **kwargs
        Additional arguments passed to AnalogSignal constructor or stored
        as annotations.

    Returns
    -------
    AnalogSignal
        Neo AnalogSignal with grid metadata in annotations:
        - 'grid_shape': (rows, cols) tuple
        - 'electrode_positions': list of (x, y) tuples in mm
        - 'ied': inter-electrode distance in mm (if provided)

    Examples
    --------
    >>> import numpy as np
    >>> import quantities as pq
    >>> from myogen.utils.neo import create_grid_signal, signal_to_grid
    >>>
    >>> # Create grid data (time, rows, cols)
    >>> data = np.random.rand(1000, 8, 8)
    >>> signal = create_grid_signal(
    ...     data,
    ...     grid_shape=(8, 8),
    ...     sampling_rate=2048 * pq.Hz,
    ...     ied=8.0,  # 8mm inter-electrode distance
    ... )
    >>>
    >>> # Access as grid
    >>> grid = signal_to_grid(signal)  # shape: (1000, 8, 8)
    >>>
    >>> # Access single electrode
    >>> row, col = 2, 3
    >>> electrode_idx = row * 8 + col
    >>> single = signal[:, electrode_idx]
    """
    signal_array = np.asarray(signal)

    # Handle units from input data
    if hasattr(signal, "units") and units == pq.mV:
        units = signal.units

    # Flatten 3D to 2D if necessary
    if signal_array.ndim == 3:
        time_points, rows, cols = signal_array.shape
        if (rows, cols) != grid_shape:
            raise ValueError(
                f"Signal shape {(rows, cols)} does not match grid_shape {grid_shape}"
            )
        signal_2d = signal_array.reshape(time_points, rows * cols)
    elif signal_array.ndim == 2:
        signal_2d = signal_array
        rows, cols = grid_shape
        expected_channels = rows * cols
        if signal_2d.shape[1] != expected_channels:
            raise ValueError(
                f"Signal has {signal_2d.shape[1]} channels but grid_shape {grid_shape} "
                f"expects {expected_channels} channels"
            )
    else:
        raise ValueError(
            f"Signal must be 2D or 3D array, got {signal_array.ndim}D with shape {signal_array.shape}"
        )

    # Compute electrode positions if not provided
    if electrode_positions is None and ied is not None:
        rows, cols = grid_shape
        electrode_positions = []
        for r in range(rows):
            for c in range(cols):
                x = c * ied
                y = r * ied
                electrode_positions.append((x, y))

    # Separate Neo kwargs from annotation kwargs
    neo_kwargs = {}
    annotation_kwargs = {}
    neo_params = {"name", "description", "file_origin", "array_annotations", "copy"}

    for key, value in kwargs.items():
        if key in neo_params:
            neo_kwargs[key] = value
        else:
            annotation_kwargs[key] = value

    # Create the AnalogSignal
    analog_signal = AnalogSignal(
        signal_2d * units if not hasattr(signal_2d, "units") else signal_2d,
        sampling_rate=sampling_rate,
        t_start=t_start,
        **neo_kwargs,
    )

    # Add grid annotations
    analog_signal.annotate(
        grid_shape=grid_shape,
        electrode_positions=electrode_positions,
        ied=ied,
        **annotation_kwargs,
    )

    return analog_signal


@beartowertype
def signal_to_grid(signal: AnalogSignal, time_slice: slice | None = None) -> np.ndarray:
    """
    Convert a grid-annotated AnalogSignal back to 3D grid format.

    Parameters
    ----------
    signal : AnalogSignal
        AnalogSignal with 'grid_shape' annotation.
    time_slice : slice, optional
        Time slice to extract. If None, returns all time points.

    Returns
    -------
    np.ndarray
        Data in grid format with shape (time, rows, cols).

    Raises
    ------
    ValueError
        If the signal doesn't have grid_shape annotation.

    Examples
    --------
    >>> grid = signal_to_grid(signal)
    >>> grid.shape  # (time, rows, cols)
    (1000, 8, 8)
    >>>
    >>> # Get single time point
    >>> frame = signal_to_grid(signal, time_slice=slice(100, 101))
    >>> frame.shape
    (1, 8, 8)
    """
    if "grid_shape" not in signal.annotations:
        raise ValueError(
            "Signal does not have 'grid_shape' annotation. "
            "Use create_grid_signal() to create grid-annotated signals."
        )

    grid_shape = signal.annotations["grid_shape"]
    rows, cols = grid_shape

    data = signal.magnitude
    if time_slice is not None:
        data = data[time_slice]

    if data.ndim == 1:
        # Single time point
        return data.reshape(1, rows, cols)
    else:
        return data.reshape(-1, rows, cols)


@beartowertype
def get_electrode(
    signal: AnalogSignal,
    row: int,
    col: int,
) -> AnalogSignal:
    """
    Extract a single electrode's signal from a grid-annotated AnalogSignal.

    Parameters
    ----------
    signal : AnalogSignal
        AnalogSignal with 'grid_shape' annotation.
    row : int
        Row index of the electrode.
    col : int
        Column index of the electrode.

    Returns
    -------
    AnalogSignal
        Single-channel AnalogSignal for the specified electrode.

    Examples
    --------
    >>> electrode_signal = get_electrode(signal, row=2, col=3)
    >>> electrode_signal.shape
    (1000, 1)
    """
    if "grid_shape" not in signal.annotations:
        raise ValueError("Signal does not have 'grid_shape' annotation.")

    rows, cols = signal.annotations["grid_shape"]
    if row < 0 or row >= rows:
        raise ValueError(f"Row {row} out of bounds for grid with {rows} rows")
    if col < 0 or col >= cols:
        raise ValueError(f"Column {col} out of bounds for grid with {cols} columns")

    channel_idx = row * cols + col
    return signal[:, channel_idx]


@beartowertype
def get_row(
    signal: AnalogSignal,
    row: int,
) -> AnalogSignal:
    """
    Extract all electrodes from a specific row.

    Parameters
    ----------
    signal : AnalogSignal
        AnalogSignal with 'grid_shape' annotation.
    row : int
        Row index to extract.

    Returns
    -------
    AnalogSignal
        Multi-channel AnalogSignal for all electrodes in the row.

    Examples
    --------
    >>> row_signal = get_row(signal, row=2)
    >>> row_signal.shape  # (time, n_cols)
    (1000, 8)
    """
    if "grid_shape" not in signal.annotations:
        raise ValueError("Signal does not have 'grid_shape' annotation.")

    rows, cols = signal.annotations["grid_shape"]
    if row < 0 or row >= rows:
        raise ValueError(f"Row {row} out of bounds for grid with {rows} rows")

    start_idx = row * cols
    end_idx = start_idx + cols
    return signal[:, start_idx:end_idx]


@beartowertype
def get_column(
    signal: AnalogSignal,
    col: int,
) -> AnalogSignal:
    """
    Extract all electrodes from a specific column.

    Parameters
    ----------
    signal : AnalogSignal
        AnalogSignal with 'grid_shape' annotation.
    col : int
        Column index to extract.

    Returns
    -------
    AnalogSignal
        Multi-channel AnalogSignal for all electrodes in the column.

    Examples
    --------
    >>> col_signal = get_column(signal, col=3)
    >>> col_signal.shape  # (time, n_rows)
    (1000, 8)
    """
    if "grid_shape" not in signal.annotations:
        raise ValueError("Signal does not have 'grid_shape' annotation.")

    rows, cols = signal.annotations["grid_shape"]
    if col < 0 or col >= cols:
        raise ValueError(f"Column {col} out of bounds for grid with {cols} columns")

    channel_indices = [r * cols + col for r in range(rows)]
    return signal[:, channel_indices]


# Backwards compatibility: Keep GridAnalogSignal as deprecated alias
# TODO: Remove in future version
class GridAnalogSignal(AnalogSignal):
    """
    DEPRECATED: Use create_grid_signal() instead.

    This class is maintained for backwards compatibility with existing
    pickled data. New code should use create_grid_signal() and the
    associated helper functions (signal_to_grid, get_electrode, etc.).
    """

    def __new__(cls, signal, **kwargs):
        """Create new GridAnalogSignal instance."""
        import warnings

        warnings.warn(
            "GridAnalogSignal is deprecated. Use create_grid_signal() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Handle units from input data
        if hasattr(signal, "units") and "units" not in kwargs:
            kwargs["units"] = signal.units

        # Convert 3D grid data to 2D for AnalogSignal storage
        signal_array = np.asarray(signal)

        if signal_array.ndim == 3:
            time_points, rows, cols = signal_array.shape
            signal_2d = signal_array.reshape(time_points, rows * cols)
        elif signal_array.ndim == 2:
            signal_2d = signal_array
        else:
            raise ValueError(
                f"Signal must be a 2D or 3D array, got {signal_array.ndim}D"
            )

        obj = AnalogSignal.__new__(cls, signal_2d, **kwargs)
        return obj

    def __init__(self, signal, **kwargs):
        """Initialize the GridAnalogSignal."""
        super().__init__(signal, **kwargs)

        signal_array = np.asarray(signal)
        if signal_array.ndim == 3:
            _, rows, cols = signal_array.shape
            self.grid_size = (rows, cols)
            # Also store in annotations for NWB compatibility
            self.annotate(grid_shape=(rows, cols))
        elif hasattr(self, "grid_size"):
            pass
        else:
            time_points, channels = signal_array.shape
            import math

            cols = int(math.sqrt(channels))
            if cols * cols == channels:
                rows = cols
            else:
                for c in range(int(math.sqrt(channels)), 0, -1):
                    if channels % c == 0:
                        cols = c
                        rows = channels // c
                        break
            self.grid_size = (rows, cols)
            self.annotate(grid_shape=(rows, cols))

    @property
    def magnitude(self):
        """Return the signal magnitude in 3D grid format (time, rows, cols)."""
        magnitude_2d = super().magnitude
        if hasattr(self, "grid_size"):
            rows, cols = self.grid_size
            if magnitude_2d.ndim == 1:
                return magnitude_2d.reshape(1, rows, cols)
            else:
                return magnitude_2d.reshape(-1, rows, cols)
        else:
            return magnitude_2d

    @property
    def shape(self):
        """Return the 3D grid shape (time, rows, cols)."""
        if hasattr(self, "grid_size"):
            rows, cols = self.grid_size
            return (super().shape[0], rows, cols)
        else:
            return super().shape

    def as_grid(self, time_slice=None) -> np.ndarray:
        """Return data in grid format (time, rows, cols)."""
        return signal_to_grid(self, time_slice)

    def __reduce_ex__(self, protocol):
        """Custom pickle support to preserve grid_size."""
        parent_pickle = super().__reduce_ex__(protocol)
        if len(parent_pickle) >= 3:
            constructor, args, state = parent_pickle[:3]
            if state is None:
                state = {}
            if hasattr(self, "grid_size"):
                state["grid_size"] = self.grid_size
            return (constructor, args, state) + parent_pickle[3:]
        else:
            return parent_pickle

    def __setstate__(self, state):
        """Custom unpickle support to restore grid_size."""
        grid_size = state.pop("grid_size", None)
        if hasattr(super(), "__setstate__"):
            super().__setstate__(state)
        if grid_size is not None:
            self.grid_size = grid_size
            self.annotate(grid_shape=grid_size)


__all__ = [
    "create_grid_signal",
    "signal_to_grid",
    "get_electrode",
    "get_row",
    "get_column",
    "GridAnalogSignal",  # Deprecated, kept for backwards compatibility
]
