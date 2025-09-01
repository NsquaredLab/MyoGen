import logging
import os
import warnings
from typing import Any

import numpy as np
import seaborn as sns
from beartype.cave import IterableType
from matplotlib.axes import Axes

from myogen.utils.decorators import beartowertype
from myogen.utils.types import (
    CORTICAL_INPUT__MATRIX,
    CURRENT__AnalogSignal,
    SPIKE_TRAIN__Block,
)

# Configure multiple sources to suppress font warnings
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
logging.getLogger("libNeuroML").setLevel(logging.ERROR)

# Set environment variable to suppress matplotlib font cache warnings
os.environ["MPLCONFIGDIR"] = "/tmp"


@beartowertype
def plot_spike_trains(
    spike_trains__Block: SPIKE_TRAIN__Block,
    axs: IterableType[Axes],
    pool_current__AnalogSignal: CURRENT__AnalogSignal | None = None,
    cortical_input__matrix: CORTICAL_INPUT__MATRIX | None = None,
    pool_to_plot: list[int] | None = None,
    apply_default_formatting: bool = True,
    **kwargs: Any,
) -> IterableType[Axes]:
    """
    Plot spike trains for each motor neuron pool.

    Parameters
    ----------
    spike_trains__Block : SPIKE_TRAIN__Block
        A neo Block containing at least one segment with spike trains to plot.
    axs : IterableType[Axes]
        Matplotlib axes to plot on.
        This could be the same axis for all pools, or a separate axis for each pool.
        If a separate axis is used, the number of axes must match the number of pools.
    pool_current__AnalogSignal : CURRENT__AnalogSignal, optional
        The input current signal to plot, by default None.
    cortical_input__matrix : CORTICAL_INPUT__MATRIX, optional
        The cortical input matrix to plot, by default None.
    pool_to_plot : list[int], optional
        The pools to plot if not all pools should be plotted, by default None (all pools are plotted).
    apply_default_formatting : bool, default True
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
        _pool_to_plot = np.arange(len(spike_trains__Block.segments))
    else:
        _pool_to_plot = np.array(pool_to_plot)

    if len(list(axs)) != len(_pool_to_plot):
        raise ValueError(
            f"Number of axes must match number of pools to plot. Got {len(list(axs))} axes, but {len(_pool_to_plot)} pools to plot."
        )

    colors = ["#90b8e0", "#af8bff"]

    # Global warning filter that catches all font-related warnings
    warnings.filterwarnings("ignore", message=".*Font family.*not found.*")
    warnings.filterwarnings("ignore", message=".*findfont.*")

    for ax_idx, pool_idx in enumerate(_pool_to_plot):
        segment = spike_trains__Block.segments[pool_idx]
        ax = list(axs)[ax_idx]

        spike_trains__SpikeTrainList = segment.spiketrains

        # Sort spike trains by their minimum spike time
        spike_trains__SpikeTrainList.sort(
            key=lambda x: x.min() if len(x) > 0 else np.inf
        )

        i = 0
        # Use scatter dots for cleaner spike visualization
        for spike_times in spike_trains__SpikeTrainList:
            if len(spike_times) > 0:
                ax.scatter(
                    spike_times.rescale("s"),
                    np.full(len(spike_times), i + 1),
                    color=colors[i % 2],
                    s=10,  # dot size
                    alpha=0.8,
                    zorder=1,
                    edgecolors="none",
                    **kwargs,
                )

                i += 1

        if apply_default_formatting:
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Motor Neuron Index")
            ax.set_title(f"Pool {pool_idx + 1} Spike Trains")

        # Initialize current variables to avoid undefined variable errors
        pc_min = 0
        pc_max = 1

        # if cortical_input__matrix is not None:
        #     pc = cortical_input__matrix[_pool_to_plot[spike_pool_idx]]

        #     pc_min = np.min(pc)
        #     pc_max = np.max(pc)
        #     print(pc_min, pc_max)
        #     pc_normalized = (pc - pc_min) / (pc_max - pc_min)
        #     pc_normalized = pc_normalized * i

        #     ax.plot(
        #         np.arange(0, len(pc)) * timestep__ms / 1000,
        #         pc_normalized,
        #         linestyle="--",
        #         linewidth=1,
        #         alpha=1,
        #         zorder=0,
        #         color="black",
        #         label=f"Cortical\nInput Firing Rate",
        #     )

        #     if apply_default_formatting:
        #         ax.legend(frameon=False)

        #         ax2 = ax.twinx()
        #         ax2.spines["right"].set_color("black")
        #         print("index", index)
        #         ax2.set_ylim(0, index + 1)
        #         ax2.set_yticks(np.linspace(0, index + 1, 10))
        #         ax2.set_yticklabels(
        #             np.round(
        #                 np.linspace(0, index + 1, 10) * (pc_max - pc_min) / (index + 1)
        #                 + pc_min
        #             )
        #         )
        #         ax2.set_ylabel("Firing rate (pps)")

        #         ax2.tick_params(axis="y", colors="black")
        #         ax2.yaxis.label.set_color("black")

        if pool_current__AnalogSignal is not None:
            pc = pool_current__AnalogSignal[:, pool_idx]

            pc_min = np.min(pc)
            pc_max = np.max(pc)

            pc_normalized = (pc - pc_min) / (pc_max - pc_min)
            pc_normalized = pc_normalized * (i + 1)

            ax.plot(
                pc.times.rescale("s"),
                np.squeeze(pc_normalized.magnitude),
                linestyle="--",
                linewidth=1,
                alpha=1,
                zorder=0,
                color="black",
                label="Input\nCurrent",
            )

            if apply_default_formatting:
                ax.legend(frameon=False)

                ax2 = ax.twinx()
                ax2.spines["right"].set_color("black")
                ax2.set_ylim(0, i + 1)

                # Align twin axis ticks with main axis
                main_ticks = ax.get_yticks()
                ax2.set_yticks(main_ticks)
                ax2.set_yticklabels(
                    [
                        f"{tick * (pc_max - pc_min) / (i + 1) + pc_min:.1f}"
                        for tick in main_ticks
                    ]
                )
                ax2.set_ylabel("Input Current (nA)")

                ax2.tick_params(axis="y", colors="black")
                ax2.yaxis.label.set_color("black")

        if apply_default_formatting:
            try:
                # Apply despine to main axis
                sns.despine(
                    ax=ax,
                    offset=10,
                    trim=True,
                    top=True,
                    right=True,  # Keep right spine for twin axis
                )

                sns.despine(
                    ax=ax2,
                    offset=10,
                    trim=False,
                    top=True,
                    bottom=True,
                    left=True,
                    right=False,  # Keep right spine for twin axis
                )

            except NameError:
                pass

    return axs
