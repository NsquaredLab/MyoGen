"""
Neo utilities for enhanced signal handling.

This module provides utilities for working with Neo signals,
including grid-based electrode array handling.
"""

import numpy as np
from neo import AnalogSignal

from .decorators import beartowertype


@beartowertype
class GridAnalogSignal(AnalogSignal):
    """
    Grid-based AnalogSignal wrapper for electrode arrays.

    This class extends Neo's AnalogSignal to handle 2D electrode grids,
    storing the data in standard AnalogSignal format (time x channels)
    but providing grid-based indexing (time, row, col).

    Parameters
    ----------
    signal : array-like
        Signal data with shape (time, rows, cols). Grid size is automatically
        inferred from the array dimensions.
    **kwargs
        Additional arguments passed to AnalogSignal constructor.
        See https://neo.readthedocs.io/en/latest/api_reference.html#neo.core.AnalogSignal

    Examples
    --------
    >>> import numpy as np
    >>> import quantities as pq
    >>> from myogen.utils.neo import GridAnalogSignal
    >>>
    >>> # Create electrode grid data (time, rows, cols)
    >>> data = np.random.rand(1000, 4, 6) * pq.mV
    >>> grid_signal = GridAnalogSignal(data, sampling_rate=2000*pq.Hz)
    >>>
    >>> # Grid-based indexing
    >>> electrode = grid_signal[:, 1, 2]      # Single electrode -> AnalogSignal
    >>> row = grid_signal[:, 1, :]            # Row data -> AnalogSignal
    >>> subgrid = grid_signal[:, 1:3, 2:5]    # Subgrid -> GridAnalogSignal
    """

    def __new__(cls, signal, **kwargs):
        """Create new GridAnalogSignal instance."""

        # Handle units from input data
        if hasattr(signal, "units") and "units" not in kwargs:
            kwargs["units"] = signal.units

        # Convert 3D grid data to 2D for AnalogSignal storage
        signal_array = np.asarray(signal)

        if signal_array.ndim == 3:
            # Normal creation: Reshape (time, rows, cols) -> (time, channels)
            time_points, rows, cols = signal_array.shape
            signal_2d = signal_array.reshape(time_points, rows * cols)
        elif signal_array.ndim == 2:
            # This might be during unpickling where data is already flattened
            # We'll handle the grid size reconstruction in __init__
            signal_2d = signal_array
        else:
            raise ValueError(
                f"Signal must be a 2D or 3D array, got {signal_array.ndim}D array with shape {signal_array.shape}"
            )

        # Create the AnalogSignal with data
        obj = AnalogSignal.__new__(cls, signal_2d, **kwargs)
        return obj

    def __init__(self, signal, **kwargs):
        """Initialize the GridAnalogSignal."""
        # The actual initialization is handled by __new__ and AnalogSignal.__init__
        # Just call the parent __init__ to ensure proper initialization
        # Note: signal processing already done in __new__
        super().__init__(signal, **kwargs)

        # Handle grid_size setting for both creation and unpickling
        signal_array = np.asarray(signal)
        if signal_array.ndim == 3:
            # During creation: signal is 3D (time, rows, cols)
            _, rows, cols = signal_array.shape
            self.grid_size = (rows, cols)
        elif hasattr(self, "grid_size"):
            # During unpickling: grid_size should already be set
            pass
        else:
            # Fallback: try to infer from 2D signal if we can determine original grid dimensions
            # This is a best effort - ideally grid_size should be preserved during pickling
            time_points, channels = signal_array.shape
            # Try to find reasonable grid dimensions (this is heuristic)
            import math

            cols = int(math.sqrt(channels))
            if cols * cols == channels:
                rows = cols
            else:
                # Find best factorization
                for c in range(int(math.sqrt(channels)), 0, -1):
                    if channels % c == 0:
                        cols = c
                        rows = channels // c
                        break
            self.grid_size = (rows, cols)

    def __getitem__(self, key):
        """Enhanced indexing supporting grid coordinates."""
        rows, cols = self.grid_size

        # Handle different indexing patterns
        if isinstance(key, tuple):
            if len(key) == 3:
                # 3D grid indexing: (time_slice, row_idx, col_idx)
                time_slice, row_idx, col_idx = key

                # Count how many rows/cols we'll get
                if isinstance(row_idx, slice):
                    if row_idx == slice(None):
                        row_count = rows
                    else:
                        row_count = len(np.arange(*row_idx.indices(rows)))
                else:
                    row_count = len(np.atleast_1d(row_idx))

                if isinstance(col_idx, slice):
                    if col_idx == slice(None):
                        col_count = cols
                    else:
                        col_count = len(np.arange(*col_idx.indices(cols)))
                else:
                    col_count = len(np.atleast_1d(col_idx))

                # Convert row/col indices to channel indices
                if isinstance(row_idx, slice):
                    if row_idx == slice(None):
                        row_indices = np.arange(rows)
                    else:
                        row_indices = np.arange(*row_idx.indices(rows))
                elif hasattr(row_idx, "__iter__") and not isinstance(row_idx, (str, np.integer)):
                    row_indices = np.asarray(row_idx)
                else:
                    row_indices = np.array([row_idx])

                if isinstance(col_idx, slice):
                    if col_idx == slice(None):
                        col_indices = np.arange(cols)
                    else:
                        col_indices = np.arange(*col_idx.indices(cols))
                elif hasattr(col_idx, "__iter__") and not isinstance(col_idx, (str, np.integer)):
                    col_indices = np.asarray(col_idx)
                else:
                    col_indices = np.array([col_idx])

                # Create channel indices for all row/col combinations
                channel_indices = []
                for r in row_indices:
                    for c in col_indices:
                        channel_indices.append(r * cols + c)

                result = super().__getitem__((time_slice, channel_indices))

                # If we're slicing both rows and cols, return GridAnalogSignal
                # Otherwise return regular AnalogSignal
                if row_count > 1 and col_count > 1:
                    # Create new GridAnalogSignal with the sliced data
                    new_grid_size = (len(row_indices), len(col_indices))

                    # Safely get attributes from result
                    kwargs = {}
                    if hasattr(result, "sampling_rate"):
                        kwargs["sampling_rate"] = result.sampling_rate
                    if hasattr(result, "units"):
                        kwargs["units"] = result.units
                    if hasattr(result, "t_start"):
                        kwargs["t_start"] = result.t_start
                    if hasattr(result, "annotations") and result.annotations:
                        kwargs.update(result.annotations)

                    return GridAnalogSignal(result.reshape(-1, *new_grid_size), **kwargs)
                else:
                    # For single time point or very small results, just return the Neo result
                    if result.magnitude.ndim == 0 or (
                        result.magnitude.ndim == 1 and len(result.magnitude) == 1
                    ):
                        # Single value - return as is (will be a quantity with units)
                        return result

                    # Convert to regular AnalogSignal
                    kwargs = {}
                    if hasattr(result, "sampling_rate"):
                        kwargs["sampling_rate"] = result.sampling_rate
                    if hasattr(result, "units"):
                        kwargs["units"] = result.units
                    if hasattr(result, "t_start"):
                        kwargs["t_start"] = result.t_start
                    if hasattr(result, "annotations") and result.annotations:
                        kwargs.update(result.annotations)

                    return AnalogSignal(result, **kwargs)

            elif len(key) == 2:
                # Check for ellipsis patterns: [..., col] means all time, all rows, specific column
                if key[0] is ...:
                    col_idx = key[1]
                    if isinstance(col_idx, (int, np.integer)) and 0 <= col_idx < cols:
                        # [..., col] - get all rows for this column (returns regular AnalogSignal)
                        channel_indices = [r * cols + col_idx for r in range(rows)]
                        result = super().__getitem__((slice(None), channel_indices))

                        # Convert to regular AnalogSignal (similar to above)
                        if result.magnitude.ndim == 0 or (
                            result.magnitude.ndim == 1 and len(result.magnitude) == 1
                        ):
                            return result

                        kwargs = {}
                        if hasattr(result, "sampling_rate"):
                            kwargs["sampling_rate"] = result.sampling_rate
                        if hasattr(result, "units"):
                            kwargs["units"] = result.units
                        if hasattr(result, "t_start"):
                            kwargs["t_start"] = result.t_start
                        if hasattr(result, "annotations") and result.annotations:
                            kwargs.update(result.annotations)

                        return AnalogSignal(result, **kwargs)

                return super().__getitem__(key)

        # Handle single values or other patterns
        return super().__getitem__(key)

    def as_grid(self, time_slice=None) -> np.ndarray:
        """Return data in grid format (time, rows, cols)."""
        data = self.magnitude
        rows, cols = self.grid_size

        if time_slice is not None:
            data = data[time_slice]

        if data.ndim == 1:
            # Single time point, reshape to (rows, cols)
            return data.reshape(rows, cols)
        else:
            # Multiple time points, reshape to (time, rows, cols)
            return data.reshape(-1, rows, cols)

    @property
    def magnitude(self):
        """Return the signal magnitude in 3D grid format (time, rows, cols)."""
        # Get the 2D magnitude from parent class
        magnitude_2d = super().magnitude

        # Reshape to 3D grid format
        if hasattr(self, "grid_size"):
            rows, cols = self.grid_size
            if magnitude_2d.ndim == 1:
                # Single time point, reshape to (rows, cols)
                return magnitude_2d.reshape(1, rows, cols)
            else:
                # Multiple time points, reshape to (time, rows, cols)
                return magnitude_2d.reshape(-1, rows, cols)
        else:
            # Fall back to parent magnitude during initialization
            return magnitude_2d

    @property
    def shape(self):
        """Return the 3D grid shape (time, rows, cols)."""
        # During initialization, grid_size might not be set yet
        if hasattr(self, "grid_size"):
            rows, cols = self.grid_size
            return (super().shape[0], rows, cols)
        else:
            # Fall back to parent shape during initialization
            return super().shape

    def __reduce_ex__(self, protocol):
        """Custom pickle support to preserve grid_size."""
        # Get the standard AnalogSignal pickle data
        parent_pickle = super().__reduce_ex__(protocol)

        # The parent returns (constructor, args, state)
        if len(parent_pickle) >= 3:
            constructor, args, state = parent_pickle[:3]
            # Add grid_size to the state
            if state is None:
                state = {}
            if hasattr(self, "grid_size"):
                state["grid_size"] = self.grid_size
            return (constructor, args, state) + parent_pickle[3:]
        else:
            return parent_pickle

    def __setstate__(self, state):
        """Custom unpickle support to restore grid_size."""
        # Extract grid_size before calling parent
        grid_size = state.pop("grid_size", None)

        # Call parent setstate
        if hasattr(super(), "__setstate__"):
            super().__setstate__(state)

        # Restore grid_size
        if grid_size is not None:
            self.grid_size = grid_size


__all__ = ["GridAnalogSignal"]
