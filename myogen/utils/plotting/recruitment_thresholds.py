import logging
import warnings
from typing import Any, Union, Dict, Optional

import numpy as np
import seaborn as sns
from beartype.cave import IterableType
from matplotlib.axes import Axes

from myogen.utils.types import beartowertype

# Configure multiple sources to suppress font warnings
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
logging.getLogger("libNeuroML").setLevel(logging.ERROR)


@beartowertype
def plot_recruitment_thresholds(
    thresholds: Union[Dict[Union[str, int, float], np.ndarray], np.ndarray],
    axs: IterableType[Axes],
    model_name: Optional[str] = None,
    y_max: Optional[float] = None,
    colors: Optional[Union[str, list]] = None,
    markers: Optional[Union[str, list]] = None,
    linestyles: Optional[Union[str, list]] = None,
    apply_default_formatting: bool = True,
    **kwargs: Any,
) -> IterableType[Axes]:
    """
    Plot recruitment thresholds for one or multiple parameter sets.

    Parameters
    ----------
    thresholds : Dict[str | int | float, np.ndarray] | np.ndarray
        If dict: {parameter_value: rt_array} for multiple lines
        If array: single rt_array for single line plot
    axs : IterableType[Axes]
        Matplotlib axes to plot on. This could be the same axis for all datasets, or separate axes for each dataset.
    model_name : str, optional
        Name of the model for the plot title (only used if apply_default_formatting is True)
    y_max : float, optional
        Maximum y-axis value. If None, determined from data
    colors : str or list, optional
        Colors for plot lines
    markers : str or list, optional
        Markers for plot lines
    linestyles : str or list, optional
        Line styles for plot lines
    apply_default_formatting : bool, optional
        Whether to apply default formatting to the plot
    **kwargs : dict
        Additional keyword arguments to pass to the plot function. Only used if apply_default_formatting is False.

    Returns
    -------
    IterableType[Axes]
        The axes that were plotted on

    Raises
    ------
    ValueError
        If the number of axes does not match the expected number of plots
    """
    if model_name is not None:
        print(f"Creating {model_name} model visualization...")

    # Global warning filter that catches all font-related warnings
    warnings.filterwarnings("ignore", message=".*Font family.*not found.*")
    warnings.filterwarnings("ignore", message=".*findfont.*")

    axs_list = list(axs)

    # Determine if we're plotting single or multiple datasets
    if isinstance(thresholds, dict):
        # Multiple lines plot - use first axis or require one axis per dataset
        if len(axs_list) == 1:
            # Plot all datasets on the same axis
            ax = axs_list[0]
            _plot_multiple_datasets(
                ax,
                thresholds,
                colors,
                markers,
                linestyles,
                y_max,
                apply_default_formatting,
                model_name,
                **kwargs,
            )
        else:
            # Plot each dataset on separate axes
            if len(axs_list) != len(thresholds):
                raise ValueError(
                    f"Number of axes must match number of datasets. Got {len(axs_list)} axes, but {len(thresholds)} datasets."
                )
            for ax, (param, dataset) in zip(axs_list, thresholds.items()):
                single_data = {param: dataset}
                _plot_multiple_datasets(
                    ax,
                    single_data,
                    colors,
                    markers,
                    linestyles,
                    y_max,
                    apply_default_formatting,
                    f"{model_name} - {param}" if model_name else str(param),
                    **kwargs,
                )
    else:
        # Single line plot - use first axis
        if len(axs_list) == 0:
            raise ValueError("At least one axis must be provided")
        ax = axs_list[0]
        _plot_single_dataset(
            ax,
            thresholds,
            colors,
            markers,
            linestyles,
            y_max,
            apply_default_formatting,
            model_name,
            **kwargs,
        )

    return axs


def _plot_multiple_datasets(
    ax: Axes,
    data: Dict[Union[str, int, float], np.ndarray],
    colors,
    markers,
    linestyles,
    y_max,
    apply_default_formatting,
    model_name,
    **kwargs,
):
    """Helper function to plot multiple datasets on a single axis."""
    if colors is None:
        colors = [
            "blue",
            "navy",
            "royalblue",
            "steelblue",
            "green",
            "darkgreen",
            "forestgreen",
        ]
    if markers is None:
        markers = ["^", "s", "D", "o", "v", "<", ">"]
    if linestyles is None:
        linestyles = ["-", "--", "-.", ":", "-", "--", "-."]

    y_values = []
    for i, (param, rt) in enumerate(reversed(data.items())):
        times = np.arange(len(rt))
        rt = np.concatenate([rt[::2], rt[-1:]])
        times = np.concatenate([times[::2], times[-1:]])

        plot_kwargs = kwargs.copy() if not apply_default_formatting else {}

        if apply_default_formatting:
            ax.plot(
                times,
                rt,
                color=colors[i % len(colors)],
                linewidth=2,
                zorder=0,
            )
            ax.scatter(
                times,
                rt,
                color=colors[i % len(colors)],
                label=f"slope={param}",
                marker=markers[i % len(markers)],
                zorder=1,
            )
        else:
            ax.plot(times, rt, **plot_kwargs)
            ax.scatter(times, rt, label=f"slope={param}", **plot_kwargs)

        y_values.extend(rt)

    if y_max is None:
        y_max = max(y_values)

    if apply_default_formatting:
        _apply_default_formatting(ax, data, y_max, model_name, is_multiple=True)


def _plot_single_dataset(
    ax: Axes,
    data: np.ndarray,
    colors,
    markers,
    linestyles,
    y_max,
    apply_default_formatting,
    model_name,
    **kwargs,
):
    """Helper function to plot a single dataset."""
    rt = data

    if apply_default_formatting:
        color = colors or "red"
        marker = markers or "o"
        linestyle = linestyles or "-"

        ax.plot(
            rt,
            color=color,
            linewidth=2,
            linestyle=linestyle,
            label="Recruitment Thresholds",
            marker=marker,
            markersize=4,
        )
    else:
        ax.plot(rt, **kwargs)

    if y_max is None:
        y_max = np.max(rt)

    if apply_default_formatting:
        _apply_default_formatting(ax, data, y_max, model_name, is_multiple=False)


def _apply_default_formatting(ax: Axes, data, y_max, model_name, is_multiple: bool):
    """Helper function to apply default formatting to the plot."""
    ax.set_xlabel("Motor Unit Index")
    ax.set_ylabel("Recruitment\nThreshold (%)")
    if model_name is not None:
        ax.set_title(model_name)

    # Set y-axis limits and ticks
    ax.set_ylim(0, y_max * 1.1)
    if is_multiple and isinstance(data, dict):
        y_min = min([np.min(rt) for rt in data.values()])
    else:
        y_min = (
            np.min(data)
            if not is_multiple
            else min([np.min(rt) for rt in data.values()])
        )

    ax.set_yticks(
        [y_min, y_max / 2, y_max],
    )
    ax.set_yticklabels([f"min={y_min:.3f}", f"mid={y_max / 2:.2f}", f"max={y_max:.2f}"])

    # Remove legend box
    legend = ax.legend()
    if legend:
        legend.set_frame_on(False)

    # Apply seaborn despine to the specific axis
    sns.despine(ax=ax, offset=10, trim=True)
