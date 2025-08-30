from typing import Any

from beartype.cave import IterableType
from matplotlib.axes import Axes

from myogen.utils.decorators import beartowertype
from myogen.utils.types import CURRENT__AnalogSignal


@beartowertype
def plot_input_current__matrix(
    input_current__matrix: CURRENT__AnalogSignal,
    axs: IterableType[Axes],
    apply_default_formatting: bool = True,
    **kwargs: Any,
) -> IterableType[Axes]:
    """
    Plot the input current.

    Parameters
    ----------
    input_current__matrix: INPUT_CURRENT__MATRIX
        AnalogSignal of shape (t_points, n_pools) containing current values
        Each column represents the current for one pool
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

    # Extract time array and signal data from AnalogSignal
    t = input_current__matrix.times.magnitude  # Time in milliseconds
    signal_data = input_current__matrix.magnitude  # Signal data without units

    n_pools = signal_data.shape[1]  # Number of pools is second dimension

    if len(list(axs)) != n_pools:
        raise ValueError(
            f"Number of axes must match number of pools. Got {len(list(axs))} axes, but {n_pools} pools."
        )

    for i, ax in enumerate(list(axs)):
        ax.plot(t, signal_data[:, i], **kwargs)
        if apply_default_formatting:
            ax.set_title(f"Pool {i + 1} Input Current")
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Current (nA)")

    return axs
