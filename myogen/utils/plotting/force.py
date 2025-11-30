from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes

from myogen.simulator import ForceModel
from myogen.utils.decorators import beartowertype


@beartowertype
def plot_twitch_parameter_assignment(
    force_model: ForceModel,
    ax: Axes,
    MU_to_highlight: list[int] | None = None,
    flip_x: bool = True,
    apply_default_formatting: bool = True,
    **kwargs: Any,
) -> Axes:
    """
    Plot the twitch parameter assignment.

    Parameters
    ----------
    force_model : ForceModel
        The force model to plot.
    ax : Axes
        The axes to plot on.
    MU_to_highlight : list[int], optional
        The MUs to highlight. The highlight is done by arrows pointing to the MUs with the label of the MU N where N is the index of the MU in the list.
    flip_x : bool, optional
        Whether to flip the x-axis. By default, the x-axis is flipped. In the original paper, the x-axis is not flipped.
    apply_default_formatting : bool, optional
        Whether to apply default formatting to the plot, by default True
    **kwargs : Any
        Additional keyword arguments to pass to the plot function. Only used if apply_default_formatting is False.
    Returns
    -------
    Axes
        The axes with the plot.
    """

    # Convert data to plotting coordinates
    x_data = (
        force_model.contraction_times__samples / float(force_model.recording_frequency__Hz) * 1000
    )
    y_data = force_model.peak_twitch_forces__unitless

    if apply_default_formatting:
        ax.plot(x_data, y_data, "-o")
    else:
        ax.plot(x_data, y_data, **kwargs)

    if MU_to_highlight is not None:
        # Convert to 0-indexed numpy array for easier manipulation
        mu_indices = np.array(MU_to_highlight) - 1

        # Get default matplotlib color cycle
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"][1:]  # Skip first color to avoid black

        # Calculate normal vectors for label positioning
        def calculate_normal_vector(
            idx: int, x_coords: np.ndarray, y_coords: np.ndarray
        ) -> tuple[float, float]:
            """Calculate the normal vector at a given point using neighboring points."""
            n_points = len(x_coords)

            if n_points == 1:
                # Single point case - use vertical normal
                return (0.0, 1.0)
            elif idx == 0:
                # First point - use forward difference
                dx = x_coords[1] - x_coords[0]
                dy = y_coords[1] - y_coords[0]
            elif idx == n_points - 1:
                # Last point - use backward difference
                dx = x_coords[idx] - x_coords[idx - 1]
                dy = y_coords[idx] - y_coords[idx - 1]
            else:
                # Middle points - use central difference
                dx = x_coords[idx + 1] - x_coords[idx - 1]
                dy = y_coords[idx + 1] - y_coords[idx - 1]

            # Normalize the tangent vector
            tangent_length = np.sqrt(dx**2 + dy**2)
            if tangent_length == 0:
                return (0.0, 1.0)

            dx_norm = dx / tangent_length
            dy_norm = dy / tangent_length

            # Rotate 90 degrees counterclockwise to get normal vector
            # (dx, dy) -> (-dy, dx)
            normal_x = -dy_norm
            normal_y = dx_norm

            return (normal_x, normal_y)

        # Calculate offset distance (5% of the y-axis range)
        y_range = np.max(y_data) - np.min(y_data)
        offset_distance = 0.1 * y_range

        for i, MU in enumerate(mu_indices):
            # Get color from cycle
            color = colors[i % len(colors)]

            # Calculate normal vector at this point
            normal_x, normal_y = calculate_normal_vector(MU, x_data, y_data)

            # Position label using normal vector
            label_x = x_data[MU] - normal_x * offset_distance
            label_y = y_data[MU] - normal_y * offset_distance

            ax.text(
                label_x,
                label_y,
                f"MU {MU + 1}",
                color=color,
                ha="center",
                va="center",
            )

            # Highlight the selected MU with colored dot
            ax.plot(
                x_data[MU],
                y_data[MU],
                "o",
                color=color,
            )

    if flip_x:
        ax.invert_xaxis()

    if apply_default_formatting:
        ax.set_xlabel("Contraction Time (ms)")
        ax.set_ylabel("Twitch Force (a. u.)")

        sns.despine(ax=ax, top=True, right=True, trim=True, offset=0)

    return ax


@beartowertype
def plot_twitches(
    force_model: ForceModel,
    ax: Axes,
    MU_to_highlight: list[int] | None = None,
    apply_default_formatting: bool = True,
    **kwargs: Any,
) -> Axes:
    """
    Plot the twitches.

    Parameters
    ----------
    force_model : ForceModel
        The force model to plot.
    ax : Axes
        The axes to plot on.
    MU_to_highlight : list[int], optional
        The MUs to highlight with matplotlib default colors. If provided, these MUs will be
        plotted with colors from the default color cycle, while all other MUs will be black.
    apply_default_formatting : bool, optional
        Whether to apply default formatting to the plot, by default True
    **kwargs : Any
        Additional keyword arguments to pass to the plot function. Only used if apply_default_formatting is False.

    Returns
    -------
    Axes
        The axes with the plot.
    """

    # Calculate timeline for all twitches
    max_twitch_length = force_model.twitch_mat.shape[0]
    timeline = (
        np.arange(max_twitch_length)
        / float(force_model.recording_frequency__Hz.rescale("Hz").magnitude)
        * 1000
    )

    # Get default matplotlib color cycle
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]  # Skip first color to avoid black

    if MU_to_highlight is not None:
        # Convert to 0-indexed numpy array
        mu_indices = np.array(MU_to_highlight) - 1

        # Plot all MUs
        for i in range(force_model.twitch_mat.shape[1]):
            # Get timeline for this specific twitch (may be shorter than max)
            twitch = force_model.twitch_list[i]
            twitch_timeline = timeline[: len(twitch)]

            if i in mu_indices:
                # Highlighted MU - use color from cycle
                color_idx = np.where(mu_indices == i)[0][0]
                color = colors[1:][color_idx % len(colors)]
                ax.plot(twitch_timeline, twitch, color=color, linewidth=2, zorder=10)
            else:
                # Non-highlighted MU - use black
                ax.plot(twitch_timeline, twitch, color=colors[0], linewidth=1.0)
    else:
        # No highlighting - plot all in black as before
        if apply_default_formatting:
            ax.plot(timeline, force_model.twitch_mat, color=colors[0], linewidth=1.0)
        else:
            ax.plot(timeline, force_model.twitch_mat, color=colors[0], **kwargs)

    if apply_default_formatting:
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Twitch Force (a. u.)")
        sns.despine(ax=ax, top=True, right=True, trim=True, offset=0)

    return ax
