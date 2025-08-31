"""
Modular plotting functions for neuron simulation results.

This module provides specialized plotting functions for different aspects of
neuromuscular simulation results from MyoGen's neuron simulator, including
raster plots, membrane traces, and physiological model dynamics.
"""

import warnings
from typing import Any

import numpy as np
from beartype.cave import IterableType
from matplotlib.axes import Axes
from neo import Block

from myogen.utils.decorators import beartowertype


@beartowertype
def plot_raster_spikes(
    results: Block,
    axs: IterableType[Axes],
    populations: list[str] = ["aMN", "Ia", "II", "gII", "gIb", "Ib", "DD"],
    time_range: tuple[float, float] | None = None,
    dot_size: float = 0.8,
    title: str = "Raster Plot",
    xlabel: str = "Time [ms]",
    ylabel: str = "Neuron ID",
    apply_default_formatting: bool = True,
    **kwargs: Any,
) -> IterableType[Axes]:
    """
    Plot spike raster plots for neural populations.

    Parameters
    ----------
    results : Block
        NEO Block containing spike train segments for each population.
    axs : IterableType[Axes]
        Matplotlib axes to plot on.
    populations : list[str], optional
        List of population names to plot, by default ["aMN", "Ia", "II", "gII", "gIb", "Ib", "DD"].
    time_range : tuple[float, float], optional
        Time range to plot (start, end) in milliseconds, by default None (full range).
    dot_size : float, optional
        Size of spike markers, by default 0.8.
    title : str, optional
        Plot title, by default "Raster Plot".
    xlabel : str, optional
        X-axis label, by default "Time [ms]".
    ylabel : str, optional
        Y-axis label, by default "Neuron ID".
    apply_default_formatting : bool, optional
        Whether to apply default formatting, by default True.
    **kwargs : Any
        Additional keyword arguments passed to matplotlib plot functions.

    Returns
    -------
    IterableType[Axes]
        The axes that were plotted on.
    """
    ax_list = list(axs)
    if len(ax_list) != 1:
        raise ValueError(
            f"plot_raster_spikes requires exactly 1 axis, got {len(ax_list)}"
        )

    ax = ax_list[0]

    # Extract spike data from NEO Block segments
    all_spike_times = []
    all_spike_ids = []
    population_offsets = {}
    current_offset = 0

    for pop_name in populations:
        # Find segment for this population
        segment = None
        for seg in results.segments:
            if seg.name == pop_name:
                segment = seg
                break

        if segment is None or not segment.spiketrains:
            # No spikes for this population, skip
            population_offsets[pop_name] = current_offset
            continue

        population_offsets[pop_name] = current_offset

        # Extract spike times and IDs from spike trains
        for spiketrain in segment.spiketrains:
            neuron_id = int(spiketrain.name)
            for spike_time in spiketrain.times:
                all_spike_times.append(float(spike_time.magnitude))
                all_spike_ids.append(current_offset + neuron_id)

        # Update offset for next population
        max_id = (
            max([int(st.name) for st in segment.spiketrains])
            if segment.spiketrains
            else -1
        )
        current_offset += max_id + 1

    if all_spike_times:
        all_spike_times = np.array(all_spike_times)
        all_spike_ids = np.array(all_spike_ids)

        # Apply time range filter if specified
        if time_range is not None:
            time_mask = (all_spike_times >= time_range[0]) & (
                all_spike_times <= time_range[1]
            )
            all_spike_times = all_spike_times[time_mask]
            all_spike_ids = all_spike_ids[time_mask]

        # Plot spikes
        ax.plot(all_spike_times, all_spike_ids, ".", ms=dot_size, **kwargs)

    if apply_default_formatting:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        if time_range is not None:
            ax.set_xlim(time_range)

        # Add population labels on y-axis
        yticks = []
        yticklabels = []
        for pop_name, offset in population_offsets.items():
            yticks.append(offset)
            yticklabels.append(pop_name)

        if yticks:
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels)

    return axs


@beartowertype
def plot_membrane_traces(
    results: Block,
    axs: IterableType[Axes],
    population: str = "aMN",
    cell_indices: list[int] = [0, 10, 20, 30, 40],
    time_range: tuple[float, float] | None = None,
    title: str = "Membrane Potential",
    xlabel: str = "Time [ms]",
    ylabel: str = "Voltage [mV]",
    apply_default_formatting: bool = True,
    **kwargs: Any,
) -> IterableType[Axes]:
    """
    Plot membrane potential traces for selected cells.

    Parameters
    ----------
    results : Block
        NEO Block containing analog signal segments for membrane potentials.
    axs : IterableType[Axes]
        Matplotlib axes to plot on.
    population : str, optional
        Population name to plot, by default "aMN".
    cell_indices : list[int], optional
        List of cell indices to plot, by default [0, 10, 20, 30, 40].
    time_range : tuple[float, float], optional
        Time range to plot (start, end) in milliseconds, by default None (full range).
    title : str, optional
        Plot title, by default "Membrane Potential".
    xlabel : str, optional
        X-axis label, by default "Time [ms]".
    ylabel : str, optional
        Y-axis label, by default "Voltage [mV]".
    apply_default_formatting : bool, optional
        Whether to apply default formatting, by default True.
    **kwargs : Any
        Additional keyword arguments passed to matplotlib plot functions.

    Returns
    -------
    IterableType[Axes]
        The axes that were plotted on.
    """
    ax_list = list(axs)
    if len(ax_list) != 1:
        raise ValueError(
            f"plot_membrane_traces requires exactly 1 axis, got {len(ax_list)}"
        )

    ax = ax_list[0]

    # Find segment for this population
    segment = None
    for seg in results.segments:
        if seg.name == population:
            segment = seg
            break

    if segment is None or not segment.analogsignals:
        warnings.warn(f"No membrane potential data found for population '{population}'")
        return axs

    # Plot traces for requested cell indices
    for signal in segment.analogsignals:
        cell_id = int(signal.name)
        if cell_id in cell_indices:
            times = signal.times.rescale("ms").magnitude
            voltage = signal.magnitude.flatten()

            # Apply time range filter if specified
            if time_range is not None:
                time_mask = (times >= time_range[0]) & (times <= time_range[1])
                times = times[time_mask]
                voltage = voltage[time_mask]

            ax.plot(times, voltage, label=f"{population}[{cell_id}]", **kwargs)

    if apply_default_formatting:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="upper right")

        if time_range is not None:
            ax.set_xlim(time_range)

    return axs


@beartowertype
def plot_muscle_dynamics(
    results: Block,
    joint_angle: np.ndarray,
    time: np.ndarray,
    axs: IterableType[Axes],
    muscle_name: str = "hill",
    include_signals: list[str] = ["artAng", "L", "force", "torque"],
    include_activations: list[str] = ["TypeI", "TypeII"],
    normalize: bool = True,
    time_range: tuple[float, float] | None = None,
    title: str = "Muscle Hill Model Dynamics",
    xlabel: str = "Time [ms]",
    apply_default_formatting: bool = True,
    **kwargs: Any,
) -> IterableType[Axes]:
    """
    Plot muscle dynamics from Hill model.

    Parameters
    ----------
    results : Block
        NEO Block containing muscle model segment.
    joint_angle : np.ndarray
        Joint angle time series data.
    time : np.ndarray
        Time vector in milliseconds.
    axs : IterableType[Axes]
        Matplotlib axes to plot on (one per signal).
    muscle_name : str, optional
        Name of muscle segment to plot, by default "hill".
    include_signals : list[str], optional
        Signals to plot, by default ["artAng", "L", "force", "torque"].
    include_activations : list[str], optional
        Activation types to plot, by default ["TypeI", "TypeII"].
    normalize : bool, optional
        Whether to normalize signals, by default True.
    time_range : tuple[float, float], optional
        Time range to plot, by default None.
    title : str, optional
        Plot title, by default "Muscle Hill Model Dynamics".
    xlabel : str, optional
        X-axis label, by default "Time [ms]".
    apply_default_formatting : bool, optional
        Whether to apply default formatting, by default True.
    **kwargs : Any
        Additional keyword arguments passed to matplotlib plot functions.

    Returns
    -------
    IterableType[Axes]
        The axes that were plotted on.
    """
    ax_list = list(axs)

    # Find muscle segment by name
    muscle_segment = None
    for seg in results.segments:
        if seg.name == muscle_name:
            muscle_segment = seg
            break

    if muscle_segment is None:
        warnings.warn(f"No muscle dynamics data found for '{muscle_name}' in results")
        return axs

    # Create signal data dictionary
    signal_data = {}

    # Add joint angle if requested
    if "artAng" in include_signals:
        signal_data["artAng"] = (joint_angle, "Joint Angle [deg]")

    # Extract muscle signals from analog signals
    for signal in muscle_segment.analogsignals:
        signal_name = signal.name
        signal_array = signal.magnitude.flatten()

        if signal_name == "muscle_length" and "L" in include_signals:
            signal_data["L"] = (signal_array, "Length [L0]")
        elif signal_name == "muscle_force" and "force" in include_signals:
            signal_data["force"] = (signal_array, "Force [F0]")
        elif signal_name == "muscle_torque" and "torque" in include_signals:
            signal_data["torque"] = (signal_array, "Torque [F0·cm]")
        elif signal_name == "type1_activation" and "TypeI" in include_activations:
            if "act" not in signal_data:
                signal_data["act"] = {}
            signal_data["act"]["TypeI"] = signal_array
        elif signal_name == "type2_activation" and "TypeII" in include_activations:
            if "act" not in signal_data:
                signal_data["act"] = {}
            signal_data["act"]["TypeII"] = signal_array

    # Apply time range filter
    plot_time = time
    if time_range is not None:
        time_mask = (time >= time_range[0]) & (time <= time_range[1])
        plot_time = time[time_mask]
        # Apply same mask to all signal data
        for key, value in signal_data.items():
            if key != "act" and isinstance(value, tuple):
                if len(value) >= 2:
                    # Ensure signal array and time mask have compatible lengths
                    signal_array = value[0]
                    if len(signal_array) != len(time_mask):
                        # Trim signal array to match time array length
                        min_length = min(len(signal_array), len(time_mask))
                        signal_array = signal_array[:min_length]
                        time_mask_trimmed = time_mask[:min_length]
                        signal_data[key] = (signal_array[time_mask_trimmed], value[1])
                    else:
                        signal_data[key] = (signal_array[time_mask], value[1])
                else:
                    print(f"Warning: signal_data['{key}'] is a tuple but has length {len(value)}: {value}")
            elif key == "act":
                for act_type, act_data in value.items():
                    # Handle activation data length mismatch
                    if len(act_data) != len(time_mask):
                        min_length = min(len(act_data), len(time_mask))
                        act_data_trimmed = act_data[:min_length]
                        time_mask_trimmed = time_mask[:min_length]
                        signal_data[key][act_type] = act_data_trimmed[time_mask_trimmed]
                    else:
                        signal_data[key][act_type] = act_data[time_mask]
            else:
                print(f"Warning: signal_data['{key}'] is not a tuple or 'act': type={type(value)}, value={value}")

    # Plot signals
    plot_idx = 0

    # Regular signals
    for signal_name in include_signals:
        if signal_name in signal_data and plot_idx < len(ax_list):
            ax = ax_list[plot_idx]
            data, ylabel = signal_data[signal_name]
            ax.plot(plot_time, data, **kwargs)

            if apply_default_formatting:
                ax.set_ylabel(ylabel)
                if plot_idx == 0:
                    ax.set_title(title)
                if plot_idx == len(ax_list) - 1:
                    ax.set_xlabel(xlabel)
                if time_range is not None:
                    ax.set_xlim(time_range)

            plot_idx += 1

    # Activations (if any to plot)
    if include_activations and "act" in signal_data and plot_idx < len(ax_list):
        ax = ax_list[plot_idx]
        for act_type in include_activations:
            if act_type in signal_data["act"]:
                ax.plot(
                    plot_time, signal_data["act"][act_type], label=act_type, **kwargs
                )

        if apply_default_formatting:
            ax.set_ylabel("Activation [a.u.]")
            ax.legend()
            if plot_idx == 0:
                ax.set_title(title)
            if plot_idx == len(ax_list) - 1:
                ax.set_xlabel(xlabel)
            if time_range is not None:
                ax.set_xlim(time_range)

    return axs

@beartowertype
def plot_antagonist_muscle_comparison(
    results: Block,
    joint_angle: np.ndarray,
    time: np.ndarray,
    axs: IterableType[Axes],
    flexor_name: str = "hill_flexor",
    extensor_name: str = "hill_extensor",
    include_signals: list[str] = ["artAng", "force", "torque"],
    time_range: tuple[float, float] | None = None,
    title: str = "Antagonist Muscle Comparison",
    xlabel: str = "Time [ms]",
    apply_default_formatting: bool = True,
    **kwargs: Any,
) -> IterableType[Axes]:
    """
    Plot comparison of antagonist muscle dynamics.

    Parameters
    ----------
    results : Block
        NEO Block containing both muscle model segments.
    joint_angle : np.ndarray
        Joint angle time series data.
    time : np.ndarray
        Time vector in milliseconds.
    axs : IterableType[Axes]
        Matplotlib axes to plot on (one per signal).
    flexor_name : str, optional
        Name of flexor muscle segment, by default "hill_flexor".
    extensor_name : str, optional
        Name of extensor muscle segment, by default "hill_extensor".
    include_signals : list[str], optional
        Signals to compare, by default ["artAng", "force", "torque"].
    time_range : tuple[float, float], optional
        Time range to plot, by default None.
    title : str, optional
        Plot title, by default "Antagonist Muscle Comparison".
    xlabel : str, optional
        X-axis label, by default "Time [ms]".
    apply_default_formatting : bool, optional
        Whether to apply default formatting, by default True.
    **kwargs : Any
        Additional keyword arguments passed to matplotlib plot functions.

    Returns
    -------
    IterableType[Axes]
        The axes that were plotted on.
    """
    ax_list = list(axs)

    # Find muscle segments
    flexor_segment = None
    extensor_segment = None
    for seg in results.segments:
        if seg.name == flexor_name:
            flexor_segment = seg
        elif seg.name == extensor_name:
            extensor_segment = seg

    if flexor_segment is None or extensor_segment is None:
        warnings.warn("Could not find both flexor and extensor muscle segments")
        return axs

    # Extract data from both muscles
    flexor_data = {}
    extensor_data = {}

    for signal in flexor_segment.analogsignals:
        flexor_data[signal.name] = signal.magnitude.flatten()

    for signal in extensor_segment.analogsignals:
        extensor_data[signal.name] = signal.magnitude.flatten()

    # Apply time range filter
    plot_time = time
    if time_range is not None:
        time_mask = (time >= time_range[0]) & (time <= time_range[1])
        plot_time = time[time_mask]
        
        # Handle length mismatches for joint angle
        if len(joint_angle) != len(time_mask):
            min_length = min(len(joint_angle), len(time_mask))
            joint_angle = joint_angle[:min_length]
            time_mask_trimmed = time_mask[:min_length]
            joint_angle = joint_angle[time_mask_trimmed]
        else:
            joint_angle = joint_angle[time_mask]
            
        # Apply mask to muscle data with length checks
        for key in flexor_data:
            if len(flexor_data[key]) != len(time_mask):
                min_length = min(len(flexor_data[key]), len(time_mask))
                data_trimmed = flexor_data[key][:min_length]
                time_mask_trimmed = time_mask[:min_length]
                flexor_data[key] = data_trimmed[time_mask_trimmed]
            else:
                flexor_data[key] = flexor_data[key][time_mask]
                
        for key in extensor_data:
            if len(extensor_data[key]) != len(time_mask):
                min_length = min(len(extensor_data[key]), len(time_mask))
                data_trimmed = extensor_data[key][:min_length]
                time_mask_trimmed = time_mask[:min_length]
                extensor_data[key] = data_trimmed[time_mask_trimmed]
            else:
                extensor_data[key] = extensor_data[key][time_mask]

    # Plot signals
    plot_idx = 0

    # Joint angle
    if "artAng" in include_signals and plot_idx < len(ax_list):
        ax = ax_list[plot_idx]
        ax.plot(plot_time, joint_angle, label="Joint Angle", **kwargs)
        
        if apply_default_formatting:
            ax.set_ylabel("Joint Angle [deg]")
            ax.legend()
            if plot_idx == 0:
                ax.set_title(title)
            if plot_idx == len(ax_list) - 1:
                ax.set_xlabel(xlabel)
            if time_range is not None:
                ax.set_xlim(time_range)
        plot_idx += 1

    # Force comparison
    if "force" in include_signals and plot_idx < len(ax_list):
        ax = ax_list[plot_idx]
        if "muscle_force" in flexor_data:
            ax.plot(plot_time, flexor_data["muscle_force"], label="Flexor", **kwargs)
        if "muscle_force" in extensor_data:
            ax.plot(plot_time, extensor_data["muscle_force"], label="Extensor", **kwargs)
        
        if apply_default_formatting:
            ax.set_ylabel("Force [F0]")
            ax.legend()
            if plot_idx == 0:
                ax.set_title(title)
            if plot_idx == len(ax_list) - 1:
                ax.set_xlabel(xlabel)
            if time_range is not None:
                ax.set_xlim(time_range)
        plot_idx += 1

    # Torque comparison (including net torque)
    if "torque" in include_signals and plot_idx < len(ax_list):
        ax = ax_list[plot_idx]
        if "muscle_torque" in flexor_data and "muscle_torque" in extensor_data:
            flex_torque = flexor_data["muscle_torque"]
            ext_torque = -extensor_data["muscle_torque"]  # Negative for extensor
            net_torque = flex_torque + ext_torque
            
            ax.plot(plot_time, flex_torque, label="Flexor", **kwargs)
            ax.plot(plot_time, ext_torque, label="Extensor", **kwargs)
            ax.plot(plot_time, net_torque, label="Net", linewidth=2, **kwargs)
        
        if apply_default_formatting:
            ax.set_ylabel("Torque [F0·cm]")
            ax.legend()
            if plot_idx == 0:
                ax.set_title(title)
            if plot_idx == len(ax_list) - 1:
                ax.set_xlabel(xlabel)
            if time_range is not None:
                ax.set_xlim(time_range)

    return axs


@beartowertype
def plot_spindle_dynamics(
    results: Block,
    axs: IterableType[Axes],
    muscle_name: str = "hill_flexor",
    include_signals: list[str] = ["L"],
    include_activations: list[str] = ["Bag1", "Bag2", "Chain"],
    include_tensions: list[str] = ["Bag1", "Bag2", "Chain"],
    include_afferents: list[str] = ["Ia", "II"],
    time_range: tuple[float, float] | None = None,
    title: str = "Spindle Model Dynamics",
    xlabel: str = "Time [ms]",
    apply_default_formatting: bool = True,
    **kwargs: Any,
) -> IterableType[Axes]:
    """
    Plot spindle model dynamics.

    Parameters
    ----------
    results : Block
        NEO Block containing spindle model segment.
    axs : IterableType[Axes]
        Matplotlib axes to plot on (one per signal group).
    muscle_name : str, optional
        Name of muscle segment to use for length and time data, by default "hill_flexor".
    include_signals : list[str], optional
        Basic signals to plot, by default ["L"].
    include_activations : list[str], optional
        Intrafusal activations to plot, by default ["Bag1", "Bag2", "Chain"].
    include_tensions : list[str], optional
        Intrafusal tensions to plot, by default ["Bag1", "Bag2", "Chain"].
    include_afferents : list[str], optional
        Afferent firing rates to plot, by default ["Ia", "II"].
    time_range : tuple[float, float], optional
        Time range to plot, by default None.
    title : str, optional
        Plot title, by default "Spindle Model Dynamics".
    xlabel : str, optional
        X-axis label, by default "Time [ms]".
    apply_default_formatting : bool, optional
        Whether to apply default formatting, by default True.
    **kwargs : Any
        Additional keyword arguments passed to matplotlib plot functions.

    Returns
    -------
    IterableType[Axes]
        The axes that were plotted on.
    """
    ax_list = list(axs)

    # Find spindle segment
    spindle_segment = None
    for seg in results.segments:
        if seg.name == "spin":
            spindle_segment = seg
            break

    if spindle_segment is None:
        warnings.warn("No spindle dynamics data found in results")
        return axs

    # Extract spindle data
    spindle_data = {}

    # Get muscle length from specified muscle segment for "L" signal
    if "L" in include_signals:
        for seg in results.segments:
            if seg.name == muscle_name:
                for signal in seg.analogsignals:
                    if signal.name == "muscle_length":
                        spindle_data["L"] = (signal.magnitude.flatten(), "Length [L0]")
                        break
                break

    # Extract spindle-specific signals
    for signal in spindle_segment.analogsignals:
        signal_name = signal.name
        signal_array = signal.magnitude

        if signal_name == "bag1_activation":
            if "act" not in spindle_data:
                spindle_data["act"] = {}
            spindle_data["act"]["Bag1"] = signal_array.flatten()
        elif signal_name == "bag2_activation":
            if "act" not in spindle_data:
                spindle_data["act"] = {}
            spindle_data["act"]["Bag2"] = signal_array.flatten()
        elif signal_name == "chain_activation":
            if "act" not in spindle_data:
                spindle_data["act"] = {}
            spindle_data["act"]["Chain"] = signal_array.flatten()
        elif signal_name == "intrafusal_tensions":
            # Handle 2D tensions array (3 x time_points) or (time_points x 3)
            if signal_array.ndim == 2:
                if signal_array.shape[0] == 3:  # (3, time_points)
                    spindle_data["tension"] = {
                        "Bag1": signal_array[0, :],
                        "Bag2": signal_array[1, :],
                        "Chain": signal_array[2, :],
                    }
                elif signal_array.shape[1] == 3:  # (time_points, 3)
                    spindle_data["tension"] = {
                        "Bag1": signal_array[:, 0],
                        "Bag2": signal_array[:, 1],
                        "Chain": signal_array[:, 2],
                    }
        elif signal_name == "primary_afferent_firing__Hz":
            if "aff" not in spindle_data:
                spindle_data["aff"] = {}
            spindle_data["aff"]["Ia"] = signal_array.flatten()
        elif signal_name == "secondary_afferent_firing__Hz":
            if "aff" not in spindle_data:
                spindle_data["aff"] = {}
            spindle_data["aff"]["II"] = signal_array.flatten()

    # Get time vector from specified muscle segment
    time_vector = None
    for seg in results.segments:
        if seg.name == muscle_name:
            for signal in seg.analogsignals:
                time_vector = signal.times.rescale("ms").magnitude
                break
            break

    if time_vector is None:
        warnings.warn("Could not find time vector for spindle plotting")
        return axs

    # Apply time range filter
    plot_time = time_vector
    if time_range is not None:
        time_mask = (time_vector >= time_range[0]) & (time_vector <= time_range[1])
        plot_time = time_vector[time_mask]
        # Apply same mask to all signal data
        for key, value in spindle_data.items():
            if key in ["L"] and isinstance(value, tuple):
                spindle_data[key] = (value[0][time_mask], value[1])
            elif key in ["act", "tension", "aff"]:
                for subkey, subvalue in value.items():
                    spindle_data[key][subkey] = subvalue[time_mask]

    # Plot signals
    plot_idx = 0

    # Basic signals (like length)
    for signal_name in include_signals:
        if signal_name in spindle_data and plot_idx < len(ax_list):
            ax = ax_list[plot_idx]
            data, ylabel = spindle_data[signal_name]
            ax.plot(plot_time, data, **kwargs)

            if apply_default_formatting:
                ax.set_ylabel(ylabel)
                if plot_idx == 0:
                    ax.set_title(title)
                if plot_idx == len(ax_list) - 1:
                    ax.set_xlabel(xlabel)
                if time_range is not None:
                    ax.set_xlim(time_range)

            plot_idx += 1

    # Activations
    if include_activations and "act" in spindle_data and plot_idx < len(ax_list):
        ax = ax_list[plot_idx]
        for act_type in include_activations:
            if act_type in spindle_data["act"]:
                ax.plot(
                    plot_time, spindle_data["act"][act_type], label=act_type, **kwargs
                )

        if apply_default_formatting:
            ax.set_ylabel("Activation [a.u.]")
            ax.legend()
            if plot_idx == 0:
                ax.set_title(title)
            if plot_idx == len(ax_list) - 1:
                ax.set_xlabel(xlabel)
            if time_range is not None:
                ax.set_xlim(time_range)

        plot_idx += 1

    # Tensions
    if include_tensions and "tension" in spindle_data and plot_idx < len(ax_list):
        ax = ax_list[plot_idx]
        for tension_type in include_tensions:
            if tension_type in spindle_data["tension"]:
                ax.plot(
                    plot_time,
                    spindle_data["tension"][tension_type],
                    label=tension_type,
                    **kwargs,
                )

        if apply_default_formatting:
            ax.set_ylabel("Tension [F0]")
            ax.legend()
            if plot_idx == 0:
                ax.set_title(title)
            if plot_idx == len(ax_list) - 1:
                ax.set_xlabel(xlabel)
            if time_range is not None:
                ax.set_xlim(time_range)

        plot_idx += 1

    # Afferents
    if include_afferents and "aff" in spindle_data and plot_idx < len(ax_list):
        ax = ax_list[plot_idx]
        for aff_type in include_afferents:
            if aff_type in spindle_data["aff"]:
                ax.plot(
                    plot_time, spindle_data["aff"][aff_type], label=aff_type, **kwargs
                )

        if apply_default_formatting:
            ax.set_ylabel("Firing Rate [Hz]")
            ax.legend()
            if plot_idx == 0:
                ax.set_title(title)
            if plot_idx == len(ax_list) - 1:
                ax.set_xlabel(xlabel)
            if time_range is not None:
                ax.set_xlim(time_range)

    return axs


@beartowertype
def plot_gto_dynamics(
    results: Block,
    axs: IterableType[Axes],
    muscle_name: str = "hill_flexor",
    include_signals: list[str] = ["force", "Ib"],
    time_range: tuple[float, float] | None = None,
    title: str = "GTO Model Dynamics",
    xlabel: str = "Time [ms]",
    apply_default_formatting: bool = True,
    **kwargs: Any,
) -> IterableType[Axes]:
    """
    Plot Golgi tendon organ (GTO) dynamics.

    Parameters
    ----------
    results : Block
        NEO Block containing GTO model segment.
    axs : IterableType[Axes]
        Matplotlib axes to plot on (one per signal).
    muscle_name : str, optional
        Name of muscle segment to use for force and time data, by default "hill_flexor".
    include_signals : list[str], optional
        Signals to plot, by default ["force", "Ib"].
    time_range : tuple[float, float], optional
        Time range to plot, by default None.
    title : str, optional
        Plot title, by default "GTO Model Dynamics".
    xlabel : str, optional
        X-axis label, by default "Time [ms]".
    apply_default_formatting : bool, optional
        Whether to apply default formatting, by default True.
    **kwargs : Any
        Additional keyword arguments passed to matplotlib plot functions.

    Returns
    -------
    IterableType[Axes]
        The axes that were plotted on.
    """
    ax_list = list(axs)

    # Find GTO segment
    gto_segment = None
    for seg in results.segments:
        if seg.name == "gto":
            gto_segment = seg
            break

    if gto_segment is None:
        warnings.warn("No GTO dynamics data found in results")
        return axs

    # Extract GTO data
    gto_data = {}

    # Get muscle force from specified muscle segment for "force" signal
    if "force" in include_signals:
        for seg in results.segments:
            if seg.name == muscle_name:
                for signal in seg.analogsignals:
                    if signal.name == "muscle_force":
                        gto_data["force"] = (signal.magnitude.flatten(), "Force [F0]")
                        break
                break

    # Extract GTO-specific signals
    for signal in gto_segment.analogsignals:
        if signal.name == "ib_afferent_firing__Hz" and "Ib" in include_signals:
            gto_data["Ib"] = (signal.magnitude.flatten(), "Firing Rate [Hz]")

    # Get time vector from specified muscle segment
    time_vector = None
    for seg in results.segments:
        if seg.name == muscle_name:
            for signal in seg.analogsignals:
                time_vector = signal.times.rescale("ms").magnitude
                break
            break

    if time_vector is None:
        warnings.warn("Could not find time vector for GTO plotting")
        return axs

    # Apply time range filter
    plot_time = time_vector
    if time_range is not None:
        time_mask = (time_vector >= time_range[0]) & (time_vector <= time_range[1])
        plot_time = time_vector[time_mask]
        # Apply same mask to all signal data
        for key, value in gto_data.items():
            if isinstance(value, tuple):
                gto_data[key] = (value[0][time_mask], value[1])

    # Plot signals
    plot_idx = 0

    for signal_name in include_signals:
        if signal_name in gto_data and plot_idx < len(ax_list):
            ax = ax_list[plot_idx]
            data, ylabel = gto_data[signal_name]
            ax.plot(plot_time, data, **kwargs)

            if apply_default_formatting:
                ax.set_ylabel(ylabel)
                if plot_idx == 0:
                    ax.set_title(title)
                if plot_idx == len(ax_list) - 1:
                    ax.set_xlabel(xlabel)
                if time_range is not None:
                    ax.set_xlim(time_range)

            plot_idx += 1

    return axs
