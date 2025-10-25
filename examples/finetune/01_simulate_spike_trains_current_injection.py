"""
Spike Train Generation with Current Injection
=======================================

This example demonstrates how to simulate spike trains of cell populations (here alpha motor neurons) using current injection.

The example shows two approaches:
1. **Manual step-by-step approach** - demonstrates each phase of the NEURON simulation pipeline for educational purposes
2. **Utility function approach** - uses the convenient inject_currents_and_simulate_spike_trains function for routine use

Both approaches produce identical results, but the manual approach helps you understand the underlying mechanisms.
"""

# %%

##############################################################################
# Import Libraries
# ----------------
#
# .. important::
#    In **MyoGen** all **random number generation** is handled by the ``RANDOM_GENERATOR`` object.
#
#    This object is a wrapper around the ``numpy.random`` module and is used to generate random numbers.
#
#    It is intended to be used with the following API:
#
#    .. code-block:: python
#
#       from myogen import simulator, RANDOM_GENERATOR
#
#    To change the default seed, use ``set_random_seed``:
#
#    .. code-block:: python
#
#       from myogen import set_random_seed
#       set_random_seed(42)

from pathlib import Path

import elephant
import joblib
import neuron
import numpy as np
import quantities as pq
from matplotlib import pyplot as plt
import scienceplots  # noqa
import seaborn as sns
from neo import Block, Segment, SpikeTrain
from neuron import h

from myogen import RANDOM_GENERATOR
from myogen.simulator.neuron.populations import AlphaMN__Pool
from myogen.utils.currents import create_trapezoid_current
from myogen.utils.neuron.inject_currents_into_populations import (
    inject_currents_and_simulate_spike_trains,
    inject_currents_into_populations,
)
from myogen.utils.nmodl import load_nmodl_mechanisms

# Configure matplotlib style
plt.style.use(["science", "nature"])
sns.set_context("paper", font_scale=2)

# Disable LaTeX rendering (not required)
plt.rcParams["text.usetex"] = False
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.figsize"] = (10, 4.5)

# Keep text editable in SVG/PDF exports (for Illustrator)
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42

# Set font to Liberation Sans or Roboto
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Liberation Sans", "Roboto", "DejaVu Sans"]

# Remove top and right spines and ticks
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["xtick.top"] = False
plt.rcParams["ytick.right"] = False

# Make ticks and axis lines thicker
plt.rcParams["axes.linewidth"] = 2.0
plt.rcParams["xtick.major.width"] = 2.0
plt.rcParams["ytick.major.width"] = 2.0

# Remove minor ticks
plt.rcParams["xtick.minor.visible"] = False
plt.rcParams["ytick.minor.visible"] = False

# Adjust subplot spacing to prevent label overlap
plt.rcParams["figure.subplot.left"] = 0.2
plt.rcParams["figure.subplot.bottom"] = 0.15

##############################################################################
# Create Motor Neuron Populations (Pools)
# ---------------------------------------
#
# In MyoGen a population of cells (e.g. motor neurons) is represented by a
# **Population** class and available in the `myogen.simulator.neuron.populations` module.
#
# A population can easily be created by specifying the number of cells. Plausible default parameters are already set.
#
# For a motor neuron population (refferred to as **motor pool**), we can use the **AlphaMN__Pool** class.
# This class can also use the recruitment thresholds generated in the previous example to distribute the motor units properties in a physiologically plausible manner.
#
# .. important::
#    These **Population** classes are custom build and use therefore custom NMODL mechanisms.
#    To use them, the NMODL mechanisms need to be loaded first using the ``load_nmodl_mechanisms`` function.
#
# To showcase MyoGen's capabilities, we will create two different motor neuron pools with identical properties but different input currents.
load_nmodl_mechanisms()

save_path = Path("./results")
save_path.mkdir(exist_ok=True)

recruitment_thresholds = joblib.load(save_path / "thresholds.pkl")

n_pools = 2
motor_neuron_pools = [
    AlphaMN__Pool(
        recruitment_thresholds__array=recruitment_thresholds,
        config_file="alpha_mn_FDI.yaml",
    )
    for _ in range(n_pools)
]

##############################################################################
# Create Input Currents
# ---------------------
#
# To drive the motor units, we use a **common input current profile**.
#
# In this example, we use a **trapezoid-shaped input current** which is generated using the ``create_trapezoid_current`` function.
#
# .. note::
#    More convenient functions for generating input current profiles are available in the ``myogen.utils.currents`` module.
#
# .. note::
#    The generated input current is an instance of the ``AnalogSignal`` class from the ``neo`` package.

timestep = 0.1  # ms
simulation_time = 3000  # ms
h.secondorder = 2  # Crank-Nicolson method (second-order accurate)

rise_time_ms = list(RANDOM_GENERATOR.uniform(700, 900, size=n_pools))
plateau_time_ms = list(RANDOM_GENERATOR.uniform(800, 1000, size=n_pools))
fall_time_ms = rise_time_ms  # Make fall time equal to rise time for symmetry

# Calculate delays to center trapezoid at 1.5s (1500ms)
center_time_ms = 1500.0
delays_ms = [
    center_time_ms - rise - plateau / 2
    for rise, plateau in zip(rise_time_ms, plateau_time_ms)
]

input_current__AnalogSignal = create_trapezoid_current(
    n_pools,
    int(simulation_time / timestep),
    timestep,
    amplitudes__nA=[10.0] * n_pools,
    rise_times__ms=rise_time_ms,
    plateau_times__ms=plateau_time_ms,
    fall_times__ms=fall_time_ms,
    delays__ms=delays_ms,
)

print(
    f"Input current signal shape: {input_current__AnalogSignal.shape}\nClass: {input_current__AnalogSignal.__class__}"
)

# Save input current signal for later analysis
joblib.dump(input_current__AnalogSignal, save_path / "input_current__AnalogSignal.pkl")

##############################################################################
# Manual Simulation Approach - Step by Step
# -------------------------------------------
#
# Before showing the convenient utility function, let's understand what happens
# under the hood by implementing the simulation pipeline manually.
# This approach gives you full control and helps understand NEURON's mechanisms.

# Step 1: Set up current injection manually
# =========================================
# We need to inject time-varying currents into each motor neuron.
# This uses NEURON's ``IClamp`` (current clamp) mechanism with ``Vector.play()``.

inject_currents_into_populations(motor_neuron_pools, input_current__AnalogSignal)

# Step 2: Set up spike recording manually
# =======================================
# For each neuron, we create a ``NetCon`` (network connection) object that detects
# spikes when the membrane voltage crosses a threshold, and records spike times.

spike_detection_threshold__mV = 50.0
simulation_time__ms = input_current__AnalogSignal.t_stop.rescale(pq.ms).magnitude

spike_recorders = []
voltage_recorders = []  # Record membrane potential for all neurons
time_recorder = h.Vector()  # Record time points
time_recorder.record(h._ref_t)

for pool_idx, pool in enumerate(motor_neuron_pools):
    pool_spike_recorders = []
    pool_voltage_recorders = []

    for cell_idx, cell in enumerate(pool):
        # Create a vector to record spike times
        spike_recorder = h.Vector()

        # Create NetCon object: monitors voltage at soma(0.5) and records spikes
        # NetCon(source, target, threshold, delay, weight)
        # source: cell.soma(0.5)._ref_v (membrane voltage reference)
        # target: None (no post-synaptic target, just recording)
        nc = h.NetCon(cell.soma(0.5)._ref_v, None, sec=cell.soma)
        nc.threshold = spike_detection_threshold__mV  # Spike detection threshold
        nc.record(spike_recorder)  # Record spike times into vector

        pool_spike_recorders.append(spike_recorder)

        # Record membrane potential for all neurons
        voltage_recorder = h.Vector()
        voltage_recorder.record(cell.soma(0.5)._ref_v)
        pool_voltage_recorders.append(voltage_recorder)

    spike_recorders.append(pool_spike_recorders)
    voltage_recorders.append(pool_voltage_recorders)

# Step 3: Initialize voltages and run simulation
# ==============================================
# Before running, we need to initialize membrane voltages to physiological values.
#
# .. note:: For this MyoGen populations provide the ``get_initialization_data()`` method.
# This returns the sections and their initial voltages.

# Initialize each neuron's membrane voltage to its resting potential
for pool in motor_neuron_pools:
    for section, voltage in zip(*pool.get_initialization_data()):
        section.v = voltage

# Initialize NEURON's internal state and run the simulation
h.finitialize()  # Initialize all mechanisms and variables
neuron.run(simulation_time__ms)


# Step 4: Convert recorded data to neo.Block format
# =================================================
# The spike times are now stored in NEURON vectors. We convert them to
# the standardized neo.Block format for analysis and compatibility.

spike_train__Block_manual = Block(name="Manual Simulation Results")

for pool_idx, pool_spike_recorders in enumerate(spike_recorders):
    # Create a segment for this motor unit pool
    segment = Segment(name=f"Pool {pool_idx}")

    # Convert each neuron's spike times to a SpikeTrain object
    segment.spiketrains = []
    for neuron_idx, spike_recorder in enumerate(pool_spike_recorders):
        # Convert NEURON vector to numpy array and add units
        spike_times = spike_recorder.as_numpy() * pq.ms

        # Create SpikeTrain object with metadata
        spiketrain = SpikeTrain(
            spike_times,
            t_stop=simulation_time__ms * pq.ms,
            sampling_rate=(1 / h.dt * (1 / pq.ms)),  # Based on NEURON's dt
            sampling_period=h.dt * pq.ms,
            name=str(neuron_idx),
            description=f"Pool {pool_idx}, Neuron {neuron_idx}",
        )
        segment.spiketrains.append(spiketrain)

    spike_train__Block_manual.segments.append(segment)

joblib.dump(spike_train__Block_manual, save_path / "spike_train__Block_manual.pkl")

##############################################################################
# Convenient Utility Function Approach
# ------------------------------------
#
# The manual approach above shows you exactly what happens during simulation.
# However, since this is a common task, MyoGen provides the ``inject_currents_and_simulate_spike_trains``
# utility function that encapsulates all these steps in a single call.
#
# This is the recommended approach for routine simulations, while the manual
# approach is useful when you need custom spike detection, specialized recording,
# or want to understand the underlying mechanisms.

# Run the same simulation using the utility function
spike_train__Block = inject_currents_and_simulate_spike_trains(
    populations=motor_neuron_pools,
    input_current__AnalogSignal=input_current__AnalogSignal,
    spike_detection_thresholds__mV=50,
)

joblib.dump(spike_train__Block, save_path / "spike_train__Block_utility.pkl")

# Compare the two approaches
print("\nComparison of results:")
print(f"Manual approach: {len(spike_train__Block_manual.segments)} segments")
print(f"Utility approach: {len(spike_train__Block.segments)} segments")

# Verify they produce similar results (spike counts should be identical)
for i, (manual_seg, utility_seg) in enumerate(
    zip(spike_train__Block_manual.segments, spike_train__Block.segments)
):
    manual_spikes = sum(len(st) for st in manual_seg.spiketrains)
    utility_spikes = sum(len(st) for st in utility_seg.spiketrains)
    print(f"Pool {i}: Manual={manual_spikes} spikes, Utility={utility_spikes} spikes")

##############################################################################
# Calculate and Display Statistics
# ---------------------------------
#
# It might be of interest to calculate the **firing rates** of the motor units.
#
# .. note::
#    The **firing rates** are calculated as the number of spikes divided by the time in which each MU was active.
#    The simulation time is in milliseconds, so we need to convert it to seconds.

firing_rates = [
    np.array(
        [
            elephant.statistics.mean_firing_rate(
                st__s.time_slice(st__s.min(), st__s.max())
            )
            for st__ms in spike_train__segment.spiketrains
            if len(st__s := st__ms.rescale(pq.s)) > 0
        ]
    )
    for spike_train__segment in spike_train__Block.segments
]

print("Firing rate statistics:")
for pool_idx, firing_rates_per_pool in enumerate(firing_rates):
    active_neurons = np.sum(firing_rates_per_pool > 0)
    if len(firing_rates_per_pool) > 0 and np.sum(firing_rates_per_pool > 0) > 0:
        mean_rate = np.mean(firing_rates_per_pool[firing_rates_per_pool > 0])
        max_rate = np.max(firing_rates_per_pool)
    else:
        mean_rate = 0.0
        max_rate = 0.0

    print(
        f"  Pool {pool_idx + 1}: {active_neurons}/{len(recruitment_thresholds)} active neurons, "
        f"mean rate: {mean_rate:.1f} Hz, max rate: {max_rate:.1f} Hz"
    )

##############################################################################
# Visualize Spike Trains
# ----------------------
#
# The **spike trains** can be visualized with publication-quality styling.

# Plot each pool separately - three separate plots
colors = ["#90b8e0", "#af8bff"]
for pool_idx, segment in enumerate(spike_train__Block.segments):
    # Find last active motor unit
    active_neurons = [i for i, st in enumerate(segment.spiketrains) if len(st) > 0]
    last_active_mu = max(active_neurons) if active_neurons else len(segment.spiketrains)
    n_neurons = len(segment.spiketrains)

    # 1. Create separate figure for raster plot
    fig_raster, ax_raster = plt.subplots()

    # Plot raster with alternating colors
    for i, spiketrain in enumerate(segment.spiketrains):
        if len(spiketrain) > 0:
            spike_times = spiketrain.rescale(pq.s).magnitude
            ax_raster.scatter(
                spike_times,
                [i] * len(spike_times),
                color=colors[i % 2],  # Alternate blue and purple for each neuron
                s=1.0,
                alpha=0.8,
            )

    ax_raster.set_xlabel("Time (s)")
    ax_raster.set_ylabel("Motor Neuron ID")
    ax_raster.set_ylim(-1, n_neurons)
    ax_raster.set_xlim(0, simulation_time / 1000.0)
    ax_raster.set_xticks(range(int(simulation_time / 1000.0) + 1))

    # Set y-axis ticks to show 1, mid, and max
    ax_raster.set_yticks([1, n_neurons // 2, n_neurons])
    ax_raster.set_yticklabels(["1", str(n_neurons // 2), str(n_neurons)])

    sns.despine(ax=ax_raster, offset=10, trim=True)
    plt.tight_layout()
    plt.savefig(
        save_path / f"spike_train_raster_pool_{pool_idx + 1}.svg", transparent=True
    )
    plt.show()

    # 2. Create separate figure for current plot
    fig_current, ax_current = plt.subplots()

    time_s = input_current__AnalogSignal.times.rescale(pq.s).magnitude
    current_nA = input_current__AnalogSignal[:, pool_idx]

    ax_current.plot(
        time_s,
        current_nA,
        color="black",
        linewidth=2,
    )
    ax_current.set_xlabel("Time (s)")
    ax_current.set_ylabel("Current (nA)")
    ax_current.set_xlim(0, simulation_time / 1000.0)
    ax_current.set_xticks(range(int(simulation_time / 1000.0) + 1))

    # Set y-axis ticks for current
    current_max = np.max(current_nA)
    ax_current.set_yticks([0, current_max / 2, current_max])
    ax_current.set_yticklabels(
        [f"{0:.0f}", f"{current_max / 2:.1f}", f"{current_max:.1f}"]
    )

    sns.despine(ax=ax_current, offset=10, trim=True)
    plt.tight_layout()
    plt.savefig(
        save_path / f"spike_train_current_pool_{pool_idx + 1}.svg", transparent=True
    )
    plt.show()

    # 3. Create separate figure for membrane potential plot
    fig_voltage, ax_voltage = plt.subplots()

    # Plot membrane potential for 3rd to last active neuron
    time_voltage = time_recorder.as_numpy() / 1000.0  # Convert ms to s

    # Find the 3rd to last neuron that actually spiked
    if len(active_neurons) >= 3:
        third_last_active_idx = active_neurons[-3]
    elif len(active_neurons) > 0:
        third_last_active_idx = active_neurons[0]  # If fewer than 3 active, use first
    else:
        third_last_active_idx = 0  # Fallback if no neurons spiked

    voltage = voltage_recorders[pool_idx][third_last_active_idx].as_numpy()

    # Plot membrane potential in black
    ax_voltage.plot(time_voltage, voltage, color="black", linewidth=1.5)

    # Find and mark threshold crossings with red dots
    # Detect upward crossings: voltage goes from below to above threshold
    threshold_crossings = (
        np.where(
            (voltage[:-1] < spike_detection_threshold__mV)
            & (voltage[1:] >= spike_detection_threshold__mV)
        )[0]
        + 1
    )  # +1 because we compare with next point

    if len(threshold_crossings) > 0:
        ax_voltage.scatter(
            time_voltage[threshold_crossings],
            [spike_detection_threshold__mV] * len(threshold_crossings),
            color="red",
            s=20,
            zorder=5,
            marker="o",
        )

    ax_voltage.set_xlabel("Time (s)")
    ax_voltage.set_ylabel("Membrane\nPotential (mV)")
    ax_voltage.set_xlim(0, simulation_time / 1000.0)
    ax_voltage.set_xticks(range(int(simulation_time / 1000.0) + 1))

    # Set voltage y-axis limits
    v_min = np.min(voltage)
    v_max = np.max(voltage)
    ax_voltage.set_ylim(v_min - 5, v_max + 5)
    ax_voltage.set_yticks([v_min, (v_min + v_max) / 2, v_max])
    ax_voltage.set_yticklabels(
        [f"{v_min:.0f}", f"{(v_min + v_max) / 2:.0f}", f"{v_max:.0f}"]
    )

    sns.despine(ax=ax_voltage, offset=10, trim=True)
    plt.tight_layout()
    plt.savefig(
        save_path / f"spike_train_voltage_pool_{pool_idx + 1}.svg", transparent=True
    )
    plt.show()

    # 4. Create separate figure for summed and filtered spike train plot
    fig_summed, ax_summed = plt.subplots()

    # Use elephant to convert spike trains to binary representation
    # Use timestep as bin size (one sample)
    bin_size_ms = timestep
    binned_st = elephant.conversion.BinnedSpikeTrain(
        segment.spiketrains,
        bin_size=bin_size_ms * pq.ms,
        t_start=0 * pq.ms,
        t_stop=simulation_time * pq.ms,
    )

    # Sum all binary spike trains across neurons
    summed_activity = np.sum(binned_st.to_array(), axis=0)

    # Apply Hanning window filter (400 ms window)
    window_size_ms = 75.0
    window_size_bins = int(window_size_ms / bin_size_ms)
    hanning_window = np.hanning(window_size_bins)
    hanning_window = hanning_window / hanning_window.sum()  # Normalize

    # Apply convolution for filtering
    filtered_activity = np.convolve(summed_activity, hanning_window, mode="same")

    # Convert from spikes/bin to spikes/s (Hz)
    # bin_size is in ms, so multiply by 1000/bin_size to get Hz
    filtered_activity_hz = filtered_activity * (1000.0 / bin_size_ms)

    # Divide by the number of active motor units
    n_active_mus = len(active_neurons)
    if n_active_mus > 0:
        filtered_activity_hz = filtered_activity_hz / n_active_mus

    # Create time axis
    time_bins_s = binned_st.bin_centers.rescale(pq.s).magnitude

    # Plot filtered summed spike train
    ax_summed.plot(
        time_bins_s,
        filtered_activity_hz,
        color="black",
        linewidth=2,
    )

    ax_summed.set_xlabel("Time (s)")
    ax_summed.set_ylabel("Population Activity (pps)")
    ax_summed.set_xlim(0, simulation_time / 1000.0)
    ax_summed.set_xticks(range(int(simulation_time / 1000.0) + 1))

    # Set y-axis limits and ticks
    y_max_pps = 25.0
    ax_summed.set_ylim(0, y_max_pps)
    ax_summed.set_yticks([0, y_max_pps / 2, y_max_pps])
    ax_summed.set_yticklabels(
        [f"{0:.0f}", f"{y_max_pps / 2:.0f}", f"{y_max_pps:.0f}"]
    )

    sns.despine(ax=ax_summed, offset=10, trim=True)
    plt.tight_layout()
    plt.savefig(
        save_path / f"spike_train_summed_filtered_pool_{pool_idx + 1}.svg",
        transparent=True,
    )
    plt.show()
