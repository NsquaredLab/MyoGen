"""
Motor Unit Spike Trains with Sinusoidal Descending Drive
======================================================

This example demonstrates **realistic spike train simulation** using **sinusoidal descending drive (DD)**
instead of direct current injection. This approach provides more physiologically accurate motor control
patterns by modeling cortical input through descending drive populations.

.. note::
    This example bridges the gap between simple current injection (example 01) and full spinal network
    simulation (network_config.py). It uses:

    - **DescendingDrive__Pool**: Poisson process neurons modeling cortical input
    - **AlphaMN__Pool**: Biophysically detailed motor neurons (Powers2017 model)
    - **Network**: Synaptic connections between DD and motor neuron populations
    - **Sinusoidal patterns**: Smooth, physiologically relevant input at 0.5-2 Hz

.. important::
    **Descending Drive (DD)** refers to the cortical and subcortical neural pathways that provide
    voluntary motor commands to spinal motor neurons. This is more realistic than direct current
    injection because it models the actual synaptic input patterns from upper motor neurons.
"""

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
import numpy as np
import quantities as pq
from matplotlib import pyplot as plt
from neuron import h
from neo import Block, Segment, SpikeTrain

from myogen import RANDOM_GENERATOR
from myogen.simulator.neuron.network import Network
from myogen.simulator.neuron.populations import AlphaMN__Pool, DescendingDrive__Pool
from myogen.utils.nmodl import load_nmodl_mechanisms

##############################################################################
# Define Parameters
# -----------------
# This example simulates a **motor pool** driven by **sinusoidal descending drive** patterns.
# The DD populations receive smooth sinusoidal input at physiologically relevant frequencies,
# which then drive the motor neuron pools through realistic synaptic connections.
#
# Key parameters:
#
# - ``n_dd_neurons``: Number of descending drive neurons per pool
# - ``dd_frequency__Hz``: Frequency of sinusoidal drive pattern
# - ``dd_amplitude__Hz``: Amplitude of drive modulation
# - ``dd_baseline__Hz``: Baseline drive level
# - ``timestep``: Simulation timestep in ms (high resolution)
# - ``simulation_time``: Total simulation duration in ms

n_dd_neurons = 400  # Number of DD neurons (cortical input)
dd_frequency__Hz = 1  # Sinusoidal drive frequency (1 Hz)
dd_amplitude__Hz = 60.0  # Drive amplitude modulation
dd_baseline__Hz = 30.0  # Baseline drive level
timestep = 0.05  # ms (high resolution)
simulation_time = 6000  # ms (6 seconds for clear oscillation pattern)
dd_poisson_order = 1  # Poisson process order for DD neurons

# Connection parameters
dd_to_mn_probability = 0.8  # Connection probability DD -> MN
dd_to_mn_weight__μS = 0.05  # Synaptic weight DD -> MN (increased for motor activation)

##############################################################################
# Create Motor Neuron Pool
# ------------------------
#
# Load the **recruitment thresholds** from the previous example and create a motor neuron pool
# using the **AlphaMN__Pool** class with biophysically detailed Powers2017 neurons.

save_path = Path("./results")
save_path.mkdir(exist_ok=True)

print("Loading NMODL mechanisms...")
load_nmodl_mechanisms()

# Load recruitment thresholds from example 00
recruitment_thresholds = joblib.load(save_path / "thresholds.pkl")
n_motor_neurons = len(recruitment_thresholds)

print(f"Creating motor neuron pool with {n_motor_neurons} neurons...")
motor_neuron_pool = AlphaMN__Pool(recruitment_thresholds__array=recruitment_thresholds)

##############################################################################
# Create Descending Drive Pool
# -----------------------------
#
# Create a **DescendingDrive__Pool** that will receive the sinusoidal input pattern.
# These neurons use Poisson processes to convert the smooth input signal into realistic
# spike patterns that represent cortical input to the spinal cord.

print(f"Creating descending drive pool with {n_dd_neurons} neurons...")
descending_drive_pool = DescendingDrive__Pool(
    n=n_dd_neurons, poisson_random_process_order=dd_poisson_order, timestep__ms=timestep
)

##############################################################################
# Generate Sinusoidal Drive Pattern
# ----------------------------------
#
# Create a **smooth sinusoidal drive pattern** that represents realistic cortical motor commands.
# This pattern combines:
# - **Baseline activity**: Continuous low-level drive
# - **Sinusoidal modulation**: Smooth oscillation at physiological frequency
# - **Noise**: Small random variations for realism

# Time vector
time_points = int(simulation_time / timestep)
time_array = np.linspace(0, simulation_time, time_points)

# Generate sinusoidal pattern
sinusoidal_drive = (
    dd_baseline__Hz
    + dd_amplitude__Hz * np.sin(2 * np.pi * dd_frequency__Hz * time_array / 1000.0)
    + RANDOM_GENERATOR.normal(0, 2.0, size=time_points)  # Add small noise
)

# Ensure drive is never negative
sinusoidal_drive = np.maximum(sinusoidal_drive, 0.0)

print("Generated sinusoidal drive pattern:")
print(f"  Frequency: {dd_frequency__Hz} Hz")
print(f"  Baseline: {dd_baseline__Hz} Hz")
print(f"  Amplitude: {dd_amplitude__Hz} Hz")
print(f"  Duration: {simulation_time} ms")

# Save drive pattern
joblib.dump(sinusoidal_drive, save_path / "sinusoidal_drive_pattern.pkl")

##############################################################################
# Create Network and Connections
# -------------------------------
#
# Use the **Network** class to create synaptic connections between the descending drive
# population and the motor neuron pool. This creates realistic synaptic transmission
# with appropriate delays and weights.

print("Setting up neural network...")
network = Network(
    {
        "DD": descending_drive_pool,
        "aMN": motor_neuron_pool,
    }
)

# Connect DD neurons to motor neurons with realistic synaptic parameters
print(
    f"Connecting DD -> MN with probability {dd_to_mn_probability:.1f}, weight {dd_to_mn_weight__μS:.3f} μS"
)
network.connect(
    "DD", "aMN", probability=dd_to_mn_probability, weight__μS=dd_to_mn_weight__μS
)

# Set up external input to DD population
network.connect_from_external("cortical_input", "DD", weight__μS=1.0)

# Get NetCons for manual DD stimulation
dd_netcons = network.get_netcons("cortical_input", "DD")

##############################################################################
# Setup Spike Recording
# ---------------------
#
# Set up spike recording for motor neurons and manual tracking for DD neurons.
# DD neurons use Poisson processes and don't have traditional spike detection.

print("Setting up spike recording...")

# Manual spike tracking for DD neurons (they use Poisson processes)
dd_spike_times = [[] for _ in range(len(descending_drive_pool))]

# Record spikes from motor neurons
mn_spike_recorders = []
for cell in motor_neuron_pool:
    spike_recorder = h.Vector()
    nc = h.NetCon(cell.soma(0.5)._ref_v, None, sec=cell.soma)
    nc.threshold = 50  # Standard threshold for motor neurons
    nc.record(spike_recorder)
    mn_spike_recorders.append(spike_recorder)

##############################################################################
# Run Simulation with Sinusoidal Drive
# ------------------------------------
#
# Execute the NEURON simulation with real-time injection of the sinusoidal drive pattern.
# The DD neurons receive time-varying input that drives their Poisson processes.

print("Running simulation...")

# Set NEURON simulation parameters
h.load_file("stdrun.hoc")
h.dt = timestep
h.tstop = simulation_time

# Initialize voltages
for section, voltage in zip(*motor_neuron_pool.get_initialization_data()):
    section.v = voltage

for section, voltage in zip(*descending_drive_pool.get_initialization_data()):
    section.v = voltage


# Custom simulation loop to inject time-varying drive
def run_with_dd_input():
    """Run simulation with time-varying DD input."""
    h.finitialize()

    step_counter = 0
    while h.t < h.tstop:
        # Calculate current drive value
        current_drive = sinusoidal_drive[min(step_counter, len(sinusoidal_drive) - 1)]

        # Drive DD neurons with current input level
        for dd_cell in descending_drive_pool:
            if dd_cell.integrate(current_drive):
                # Record spike time for DD neuron
                dd_spike_times[dd_cell.pool_ID].append(h.t)
                # Generate spike in DD neuron
                spike_time = h.t + np.clip(RANDOM_GENERATOR.normal(0, 10), 0, None)
                if spike_time < h.tstop:
                    dd_netcons[dd_cell.pool_ID].event(spike_time)

        # Progress simulation
        h.fadvance()
        step_counter += 1

        # Progress indicator
        if int(h.t) % 1000 == 0 and h.t > 0:
            print(f"  Simulation time: {h.t:.0f} ms")


# Run the simulation
run_with_dd_input()

print("Simulation completed!")

##############################################################################
# Convert Spike Data to Neo Format
# ---------------------------------
#
# Convert the recorded spike data to neo.Block format for analysis and visualization.
# This creates separate segments for DD and motor neuron populations.

print("Converting spike data to neo format...")

# Create neo Block
spike_train_block = Block(name="Sinusoidal DD Spike Trains")

# Create segment for DD neurons
dd_segment = Segment(name="Descending Drive")
dd_segment.spiketrains = [
    SpikeTrain(
        spike_times * pq.ms,
        t_stop=simulation_time * pq.ms,
        sampling_rate=(1 / h.dt * (1 / pq.ms)),
        sampling_period=h.dt * pq.ms,
        name=f"DD_{i}",
        description=f"Descending Drive Neuron {i}",
    )
    for i, spike_times in enumerate(dd_spike_times)
]

# Create segment for motor neurons
mn_segment = Segment(name="Motor Neurons")
mn_segment.spiketrains = [
    SpikeTrain(
        recorder.as_numpy() * pq.ms,
        t_stop=simulation_time * pq.ms,
        sampling_rate=(1 / h.dt * (1 / pq.ms)),
        sampling_period=h.dt * pq.ms,
        name=f"MN_{i}",
        description=f"Motor Neuron {i}",
    )
    for i, recorder in enumerate(mn_spike_recorders)
]

# Add segments to block
# spike_train_block.segments.append(dd_segment)
spike_train_block.segments.append(mn_segment)

# Save results
joblib.dump(spike_train_block, save_path / "sinusoidal_dd_spike_trains.pkl")

##############################################################################
# Calculate Firing Rate Statistics
# ---------------------------------
#
# Analyze the firing patterns of both DD and motor neuron populations to understand
# how the sinusoidal input is transformed into motor output.

print("\nFiring rate analysis:")

# Calculate DD firing rates
dd_firing_rates = np.array(
    [
        elephant.statistics.mean_firing_rate(st.time_slice(st.t_start, st.t_stop))
        for st in dd_segment.spiketrains
        if len(st) > 0
    ]
)

# Calculate MN firing rates
mn_firing_rates = np.array(
    [
        elephant.statistics.mean_firing_rate(st.time_slice(st.t_start, st.t_stop))
        for st in mn_segment.spiketrains
        if len(st) > 0
    ]
)

print("Descending Drive neurons:")
print(f"  Active neurons: {len(dd_firing_rates)}/{n_dd_neurons}")
if len(dd_firing_rates) > 0:
    print(
        f"  Mean firing rate: {np.mean(dd_firing_rates):.1f} ± {np.std(dd_firing_rates):.1f} Hz"
    )
    print(
        f"  Rate range: {np.min(dd_firing_rates):.1f} - {np.max(dd_firing_rates):.1f} Hz"
    )

print("Motor neurons:")
print(f"  Active neurons: {len(mn_firing_rates)}/{n_motor_neurons}")
if len(mn_firing_rates) > 0:
    print(
        f"  Mean firing rate: {np.mean(mn_firing_rates):.1f} ± {np.std(mn_firing_rates):.1f} Hz"
    )
    print(
        f"  Rate range: {np.min(mn_firing_rates):.1f} - {np.max(mn_firing_rates):.1f} Hz"
    )

##############################################################################
# Advanced Visualization
# -----------------------
#
# Create comprehensive visualizations showing:
# 1. Sinusoidal drive input pattern
# 2. DD population raster plot with drive overlay
# 3. Motor neuron raster plot showing recruitment
# 4. Population firing rates over time

print("Creating visualizations...")

# Create figure with subplots
fig, axes = plt.subplots(4, 1, figsize=(15, 12))

# 1. Plot sinusoidal drive pattern
time_s = time_array / 1000.0  # Convert to seconds
axes[0].plot(time_s, sinusoidal_drive, "b-", linewidth=2, label="DD Input")
axes[0].axhline(dd_baseline__Hz, color="r", linestyle="--", alpha=0.7, label="Baseline")
axes[0].set_ylabel("Drive (Hz)")
axes[0].set_title(f"Sinusoidal Descending Drive Pattern ({dd_frequency__Hz} Hz)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2. DD population raster plot
dd_colors = plt.cm.get_cmap("Blues")(np.linspace(0.3, 0.8, len(dd_segment.spiketrains)))
for i, (spiketrain, color) in enumerate(zip(dd_segment.spiketrains, dd_colors)):
    if len(spiketrain) > 0:
        spike_times = spiketrain.rescale(pq.s).magnitude
        axes[1].scatter(
            spike_times, [i] * len(spike_times), c=[color], s=0.8, alpha=0.8
        )

axes[1].set_ylabel("DD Neuron ID")
axes[1].set_title(f"Descending Drive Population Activity (n={n_dd_neurons})")
axes[1].set_ylim(-1, n_dd_neurons)
axes[1].grid(True, alpha=0.3)

# 3. Motor neuron raster plot (recruitment ordered)
mn_colors = plt.cm.get_cmap("Reds")(np.linspace(0.3, 0.9, len(mn_segment.spiketrains)))
active_mn_count = 0
for i, (spiketrain, color) in enumerate(zip(mn_segment.spiketrains, mn_colors)):
    if len(spiketrain) > 0:
        spike_times = spiketrain.rescale(pq.s).magnitude
        axes[2].scatter(
            spike_times, [i] * len(spike_times), c=[color], s=1.0, alpha=0.8
        )
        active_mn_count += 1

axes[2].set_ylabel("Motor Neuron ID\n(Recruitment Order)")
axes[2].set_title(
    f"Motor Neuron Population Activity (n={active_mn_count}/{n_motor_neurons} active)"
)
axes[2].set_ylim(-1, n_motor_neurons)
axes[2].grid(True, alpha=0.3)

# 4. Population firing rates over time (binned)
bin_size_ms = 200  # 200ms bins
bins = np.arange(0, simulation_time + bin_size_ms, bin_size_ms)
bin_centers = bins[:-1] + bin_size_ms / 2

# Calculate binned firing rates for DD
dd_rates_binned = []
for bin_start, bin_end in zip(bins[:-1], bins[1:]):
    bin_spikes = []
    for spiketrain in dd_segment.spiketrains:
        spikes_in_bin = (
            spiketrain.time_slice(bin_start * pq.ms, bin_end * pq.ms)
            .times.rescale(pq.ms)
            .magnitude
        )
        bin_spikes.extend(spikes_in_bin)

    rate_hz = len(bin_spikes) / (bin_size_ms / 1000.0) / n_dd_neurons
    dd_rates_binned.append(rate_hz)

# Calculate binned firing rates for MN
mn_rates_binned = []
for bin_start, bin_end in zip(bins[:-1], bins[1:]):
    bin_spikes = []
    for spiketrain in mn_segment.spiketrains:
        spikes_in_bin = (
            spiketrain.time_slice(bin_start * pq.ms, bin_end * pq.ms)
            .times.rescale(pq.ms)
            .magnitude
        )
        bin_spikes.extend(spikes_in_bin)

    rate_hz = len(bin_spikes) / (bin_size_ms / 1000.0) / n_motor_neurons
    mn_rates_binned.append(rate_hz)

bin_centers_s = bin_centers / 1000.0
axes[3].plot(
    bin_centers_s, dd_rates_binned, "b-", linewidth=2, label="DD Population", alpha=0.8
)
axes[3].plot(
    bin_centers_s, mn_rates_binned, "r-", linewidth=2, label="MN Population", alpha=0.8
)

axes[3].set_xlabel("Time (s)")
axes[3].set_ylabel("Population Rate (Hz)")
axes[3].set_title("Population Firing Rates Over Time")
axes[3].legend()
axes[3].grid(True, alpha=0.3)

# Format all axes
for ax in axes:
    ax.set_xlim(0, simulation_time / 1000.0)

plt.tight_layout()
plt.savefig(save_path / "sinusoidal_dd_analysis.png", dpi=300, bbox_inches="tight")
plt.show()

##############################################################################
# Summary and Results
# -------------------
#
# This example demonstrates how **sinusoidal descending drive** creates more realistic
# motor neuron activation patterns compared to direct current injection.

print(f"\n{'=' * 60}")
print("SIMULATION SUMMARY")
print(f"{'=' * 60}")
print(f"Simulation duration: {simulation_time} ms")
print(f"DD frequency: {dd_frequency__Hz} Hz")
print(f"DD neurons: {n_dd_neurons} (Poisson processes)")
print(f"Motor neurons: {n_motor_neurons} (Powers2017 model)")
print(
    f"Connection strength: {dd_to_mn_weight__μS:.3f} μS (probability: {dd_to_mn_probability:.1f})"
)
print(f"Results saved to: {save_path}")
print(f"{'=' * 60}")

print("\nThis example shows how cortical oscillations at 1 Hz can drive")
print("realistic motor unit recruitment patterns through descending")
print("pathways, providing a more physiologically accurate simulation")
print("of voluntary motor control than direct current injection.")
