from typing import Any

import numpy as np
from beartype.cave import IterableType
from matplotlib.axes import Axes

from myogen.utils.types import INPUT_CURRENT__MATRIX, beartowertype


@beartowertype
def plot_input_current__matrix(
    input_current__matrix: INPUT_CURRENT__MATRIX,
    timestep__ms: float,
    axs: IterableType[Axes],
    apply_default_formatting: bool = True,
    **kwargs: Any,
) -> IterableType[Axes]:
    """
    Plot the input current.

    Parameters
    ----------
    input_current__matrix: INPUT_CURRENT__MATRIX
        Matrix of shape (n_pools, t_points) containing current values
        Each row represents the current for one pool
    timestep__ms: float
        Time step in milliseconds
    axs: IterableType[Axes]
        Matplotlib axes to plot on. This could be the same axis for all pools, or a separate axis for each pool.
    apply_default_formatting: bool
        Whether to apply default formatting to the plot
    **kwargs: dict
        Additional keyword arguments to pass to the plot function. Only used if apply_default_formatting is False.

    Returns
    -------
    IterableType[Axes]
        The axes that were plotted on

    Raises
    ------
    ValueError
        If the number of axes does not match the number of pools
    """

    t = np.arange(0, input_current__matrix.shape[-1] * timestep__ms, timestep__ms)

    if len(list(axs)) != input_current__matrix.shape[0]:
        raise ValueError(
            f"Number of axes must match number of pools. Got {len(list(axs))} axes, but {input_current__matrix.shape[0]} pools."
        )

    for i, (current, ax) in enumerate(zip(input_current__matrix, list(axs))):
        ax.plot(t, current, **kwargs)
        if apply_default_formatting:
            ax.set_title(f"Pool {i + 1} Input Current")
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Current (nA)")

    return axs
