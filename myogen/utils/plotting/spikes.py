from pathlib import Path
import logging
import os
import warnings
from typing import Any

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from beartype import beartype
from beartype.cave import IterableType

from myogen.utils.types import (
    INPUT_CURRENT__MATRIX,
    SPIKE_TRAIN__MATRIX,
    CORTICAL_INPUT__MATRIX,
)

# Configure multiple sources to suppress font warnings
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
logging.getLogger("libNeuroML").setLevel(logging.ERROR)

# Set environment variable to suppress matplotlib font cache warnings
os.environ["MPLCONFIGDIR"] = "/tmp"


@beartype
def plot_spike_trains(
    spike_trains__matrix: SPIKE_TRAIN__MATRIX,
    timestep__ms: float,
    axs: IterableType[Axes],
    pool_current__matrix: INPUT_CURRENT__MATRIX | None = None,
    cortical_input__matrix: CORTICAL_INPUT__MATRIX | None = None,
    pool_to_plot: list[int] | None = None,
    apply_default_formatting: bool = True,
    **kwargs: Any,
) -> IterableType[Axes]:
    """
    Plot spike trains for each motor neuron pool.

    Parameters
    ----------
    spike_trains__matrix : SPIKE_TRAIN__MATRIX
        The spike trains matrix to plot.
    timestep__ms : float
        Simulation timestep in ms.
    axs : IterableType[Axes]
        Matplotlib axes to plot on. This could be the same axis for all pools, or a separate axis for each pool.
    pool_current__matrix : INPUT_CURRENT__MATRIX | None, optional
        The input current matrix to plot, by default None.
    pool_to_plot : list[int] | None, optional
        The pools to plot if not all pools should be plotted, by default None (all pools are plotted).
    apply_default_formatting : bool, optional
        Whether to apply default formatting to the plot, by default True.
    **kwargs : Any
        Additional keyword arguments to pass to the plot function. Only used if apply_default_formatting is False.

    Returns
    -------
    IterableType[Axes]
        The axes that were plotted on.

    Raises
    ------
    ValueError
        If the number of axes does not match the number of pools to plot.
    """

    if pool_to_plot is None:
        _pool_to_plot = np.arange(spike_trains__matrix.shape[0])
    else:
        _pool_to_plot = np.array(pool_to_plot)

    if len(list(axs)) != len(_pool_to_plot):
        raise ValueError(
            f"Number of axes must match number of pools to plot. Got {len(list(axs))} axes, but {len(_pool_to_plot)} pools to plot."
        )

    # Global warning filter that catches all font-related warnings
    warnings.filterwarnings("ignore", message=".*Font family.*not found.*")
    warnings.filterwarnings("ignore", message=".*findfont.*")

    for spike_pool_idx, (spike_pool, ax) in enumerate(
        zip(spike_trains__matrix[_pool_to_plot], list(axs))
    ):
        # Get spike times for each neuron that fires
        spike_times_list = []
        neuron_indices = []

        for neuron in range(spike_pool.shape[0]):
            spike_times = (
                np.where(spike_pool[neuron] == 1)[0] * timestep__ms / 1000
            )  # convert to seconds
            if len(spike_times) > 0:
                spike_times_list.append(spike_times)
                neuron_indices.append(neuron)

        # Sort by first spike time for consistent ordering
        if spike_times_list:
            first_spike_times = [
                times[0] if len(times) > 0 else float("inf")
                for times in spike_times_list
            ]
            sorted_indices = np.argsort(first_spike_times)
            spike_times_list = [spike_times_list[i] for i in sorted_indices]
            neuron_indices = [neuron_indices[i] for i in sorted_indices]

        # Define alternating colors
        colors = ["#90b8e0", "#af8bff"]
        plot_colors = [colors[i % 2] for i in range(len(spike_times_list))]

        # Use scatter dots for cleaner spike visualization
        if spike_times_list:
            for i, spike_times in enumerate(spike_times_list):
                ax.scatter(
                    spike_times,
                    np.full(len(spike_times), i + 1),
                    color=plot_colors[i],
                    s=10,  # dot size
                    alpha=0.8,
                    zorder=1,
                    edgecolors="none",
                    **kwargs,
                )

        index = len(spike_times_list) if spike_times_list else 0

        if apply_default_formatting:
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Motor Neuron Index")
            ax.set_title(f"Pool {_pool_to_plot[spike_pool_idx] + 1} Spike Trains")

        # Initialize current variables to avoid undefined variable errors
        pc_min = 0
        pc_max = 1

        if cortical_input__matrix is not None:
            pc = cortical_input__matrix[_pool_to_plot[spike_pool_idx]]

            pc_min = np.min(pc)
            pc_max = np.max(pc)
            print(pc_min, pc_max)
            pc_normalized = (pc - pc_min) / (pc_max - pc_min)
            pc_normalized = pc_normalized * (index)

            ax.plot(
                np.arange(0, len(pc)) * timestep__ms / 1000,
                pc_normalized,
                linestyle="--",
                linewidth=1,
                alpha=1,
                zorder=0,
                color="black",
                label=f"Cortical\nInput Firing Rate",
            )

            if apply_default_formatting:
                ax.legend(frameon=False)

                ax2 = ax.twinx()
                ax2.spines["right"].set_color("black")
                print("index", index)
                ax2.set_ylim(0, index + 1)
                ax2.set_yticks(np.linspace(0, index + 1, 10))
                ax2.set_yticklabels(
                    np.round(
                        np.linspace(0, index + 1, 10) * (pc_max - pc_min) / (index + 1)
                        + pc_min
                    )
                )
                ax2.set_ylabel("Firing rate (pps)")

                ax2.tick_params(axis="y", colors="black")
                ax2.yaxis.label.set_color("black")

        if pool_current__matrix is not None:
            pc = pool_current__matrix[_pool_to_plot[spike_pool_idx]]

            pc_min = np.min(pc)
            pc_max = np.max(pc)

            pc_normalized = (pc - pc_min) / (pc_max - pc_min)
            pc_normalized = pc_normalized * (index + 1)

            ax.plot(
                np.arange(0, len(pc)) * timestep__ms / 1000,
                pc_normalized,
                linestyle="--",
                linewidth=1,
                alpha=1,
                zorder=0,
                color="black",
                label=f"Input\nCurrent",
            )

            if apply_default_formatting:
                ax.legend(frameon=False)

                ax2 = ax.twinx()
                ax2.spines["right"].set_color("black")
                ax2.set_ylim(0, index + 1)
                ax2.set_yticks(
                    np.linspace(0, index + 1, 10) * (pc_max - pc_min) / (index + 1)
                    + pc_min
                )
                ax2.set_ylabel("Input Current (nA)")

                ax2.tick_params(axis="y", colors="black")
                ax2.yaxis.label.set_color("black")

        if apply_default_formatting:
            sns.despine(
                ax=ax,
                offset=10,
                trim=True,
                right=False if pool_current__matrix is not None else True,
            )

    return axs


# Legacy function for backward compatibility
def plot_spike_trains_legacy(
    spike_trains__matrix: SPIKE_TRAIN__MATRIX,
    pool_current__matrix: INPUT_CURRENT__MATRIX | None = None,
    save__path: Path | None = None,
    pool_to_plot: list[int] | None = None,
    return_figs: bool = False,
    show_figs: bool = True,
):
    """
    Legacy function for backward compatibility.

    This function maintains the old API for existing code that depends on it.
    For new code, use the new plot_spike_trains function.
    """

    if pool_to_plot is None:
        _pool_to_plot = np.arange(spike_trains__matrix.shape[0])
    else:
        _pool_to_plot = np.array(pool_to_plot)

    # Global warning filter that catches all font-related warnings
    warnings.filterwarnings("ignore", message=".*Font family.*not found.*")
    warnings.filterwarnings("ignore", message=".*findfont.*")

    figs = []
    for spike_pool_idx, spike_pool in enumerate(spike_trains__matrix[_pool_to_plot]):
        index = 0

        with plt.xkcd():
            plt.rcParams.update({"font.size": 24})
            plt.rcParams.update({"axes.titlesize": 24})
            plt.rcParams.update({"axes.labelsize": 24})
            plt.rcParams.update({"xtick.labelsize": 24})
            plt.rcParams.update({"ytick.labelsize": 24})

            fig = plt.figure(num=f"Pool {spike_pool_idx + 1}", figsize=(10, 6))
            if return_figs:
                figs.append(fig)

            # Get spike times for each neuron that fires
            spike_times_list = []
            neuron_indices = []

            for neuron in range(spike_pool.shape[0]):
                spike_times = (
                    np.where(spike_pool[neuron] == 1)[0] * 0.05 / 1000
                )  # convert to seconds
                if len(spike_times) > 0:
                    spike_times_list.append(spike_times)
                    neuron_indices.append(neuron)

            # Sort by first spike time for consistent ordering
            if spike_times_list:
                first_spike_times = [
                    times[0] if len(times) > 0 else float("inf")
                    for times in spike_times_list
                ]
                sorted_indices = np.argsort(first_spike_times)
                spike_times_list = [spike_times_list[i] for i in sorted_indices]
                neuron_indices = [neuron_indices[i] for i in sorted_indices]

            # Define alternating colors
            colors = ["#90b8e0", "#af8bff"]
            plot_colors = [colors[i % 2] for i in range(len(spike_times_list))]

            # Use scatter dots for cleaner spike visualization
            if spike_times_list:
                for i, spike_times in enumerate(spike_times_list):
                    plt.scatter(
                        spike_times,
                        np.full(len(spike_times), i + 1),
                        color=plot_colors[i],
                        s=10,  # dot size
                        alpha=0.8,
                        zorder=1,
                        edgecolors="none",
                    )
                index = len(spike_times_list)

            plt.xlabel("Time (s)")
            plt.ylabel("Motor Neuron Index")

            # Initialize current variables to avoid undefined variable errors
            pc_min = 0
            pc_max = 1

            if pool_current__matrix is not None:
                pc = pool_current__matrix[spike_pool_idx]

                pc_min = np.min(pc)
                pc_max = np.max(pc)

                pc_normalized = (pc - pc_min) / (pc_max - pc_min)
                pc_normalized = pc_normalized * (index + 1)

                plt.plot(
                    np.arange(0, len(pc)) * 0.05 / 1000,
                    pc_normalized,
                    linestyle="--",
                    linewidth=1,
                    alpha=1,
                    zorder=0,
                    color="black",
                    label=f"Input\nCurrent",
                )

                plt.legend(frameon=False)

                ax2 = plt.gca().twinx()
                ax2.spines["right"].set_color("black")
                ax2.set_ylim(0, index + 1)
                ax2.set_yticks(
                    np.linspace(0, index + 1, 10) * (pc_max - pc_min) / (index + 1)
                    + pc_min
                )
                ax2.set_ylabel("Input Current (nA)")

                ax2.tick_params(axis="y", colors="black")
                ax2.yaxis.label.set_color("black")

            sns.despine(
                offset=10,
                trim=True,
                right=False if pool_current__matrix is not None else True,
            )

            plt.tight_layout()

            if save__path is not None:
                plt.savefig(save__path / f"spike_trains_{spike_pool_idx + 1}.png")
            if show_figs:
                plt.show()

    if return_figs:
        return figs
    return None
