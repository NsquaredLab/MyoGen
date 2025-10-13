"""
Spike Train Generation with Descending Drive
============================================

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

import itertools
from pathlib import Path

import elephant
import joblib
import numpy as np
import quantities as pq
from matplotlib import pyplot as plt
from neo import AnalogSignal, Block, Segment, SpikeTrain
from neuron import h
from tqdm import tqdm

from myogen import RANDOM_GENERATOR
from myogen.simulator.neuron import Network
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

# Connection parameters

##############################################################################
# Create Populations
# ------------------------
#
# Like the previous example, we create a **motor neuron pool** using the **AlphaMN__Pool** class.
#
# We also create a **DescendingDrive__Pool** to represent the cortical input.
#
# .. note:: These neurons are modeled as Poisson point processes to convert the smooth input signal into realistic
# spike patterns that represent cortical input to the spinal cord.
#

load_nmodl_mechanisms()

save_path = Path("./results")
save_path.mkdir(exist_ok=True)

recruitment_thresholds = joblib.load(save_path / "thresholds.pkl")

motor_neuron_pool = AlphaMN__Pool(recruitment_thresholds__array=recruitment_thresholds, config_file="alpha_mn_VLVM.yaml")

timestep = 0.1  # ms
h.secondorder = 2  # Crank-Nicolson method (second-order accurate)
descending_drive_pool = DescendingDrive__Pool(
    n=400, poisson_batch_size=16, timestep__ms=timestep
)

##############################################################################
# Generate Drive Pattern
# ----------------------------------
#
# The descending drive neuron population needs a time-varying input pattern to drive their Poisson processes.
# This
# Create a **smooth sinusoidal drive pattern** that represents realistic cortical motor commands.
# This pattern combines:
# - **Baseline activity**: Continuous low-level drive
# - **Sinusoidal modulation**: Smooth oscillation at physiological frequency
# - **Noise**: Small random variations for realism

simulation_time = 3000  # ms

time_points = int(simulation_time / timestep)

dd_frequency__Hz = 0.25
dd_amplitude__Hz = 60.0
dd_baseline__Hz = 10.0

sinusoidal_drive = AnalogSignal(
    signal=(
        np.maximum(
            dd_baseline__Hz
            + (
                (dd_amplitude__Hz - dd_baseline__Hz)
                * np.sin(
                    2
                    * np.pi
                    * dd_frequency__Hz
                    * np.linspace(0, simulation_time, time_points)
                    / 1000.0
                )
            ),
            dd_baseline__Hz,
        )
        + np.clip(RANDOM_GENERATOR.normal(0, 1.0, size=time_points), 0, None)
    ),
    units=pq.Hz,
    sampling_period=(timestep * pq.ms).rescale(pq.s),
)

joblib.dump(sinusoidal_drive, save_path / "sinusoidal_drive_pattern.pkl")

##############################################################################
# Create Network and Connections
# -------------------------------
#
# In MyoGen, populations can be connected using the **Network** class from the
# `myogen.simulator.neuron` module.
#
# The **Network** class provides a high-level interface for creating and managing
# connections between neuron populations.

# Use the **Network** class to create synaptic connections between the descending drive
# population and the motor neuron pool. This creates realistic synaptic transmission
# with appropriate delays and weights.
#

network = Network({"DD": descending_drive_pool, "aMN": motor_neuron_pool})

# Connect DD neurons to motor neurons with realistic synaptic parameters
network.connect(source="DD", target="aMN", probability=0.5, weight__μS=0.1)

# Set up external input to DD population
network.connect_from_external(source="cortical_input", target="DD", weight__μS=1.0)

# Get NetCons for manual DD stimulation
dd_netcons = network.get_netcons("cortical_input", "DD")

##############################################################################
# Setup Spike Recording
# ---------------------
#
# To record spikes, we need to manually set up spike detection for the motor neurons
# and track spike times for the DD neurons.
#

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
# Run Simulation
# ------------------------------------
#
# Execute the NEURON simulation with real-time injection of the sinusoidal drive pattern.
# The DD neurons receive time-varying input that drives their Poisson processes.

h.load_file("stdrun.hoc")  # Load standard run library for NEURON
h.dt = timestep
h.tstop = simulation_time

# Initialize voltages for all pools
for section, voltage in itertools.chain.from_iterable(
    zip(*pool.get_initialization_data())
    for pool in [motor_neuron_pool, descending_drive_pool]
):
    section.v = voltage


h.finitialize()

# Calculate total simulation steps for progress bar
total_steps = int(simulation_time / timestep)

step_counter = 0
with tqdm(
    total=simulation_time,
    desc="Running simulation",
    unit="ms",
    bar_format="{l_bar}{bar}| {n:.2f}/{total:.2f} ms [{elapsed}<{remaining}, {rate_fmt}]",
) as pbar:
    while h.t < h.tstop:
        current_drive = sinusoidal_drive[min(step_counter, len(sinusoidal_drive) - 1)]

        # Drive DD neurons with current input level
        for dd_cell in descending_drive_pool:
            if dd_cell.integrate(current_drive):
                # Record spike time for DD neuron
                dd_spike_times[dd_cell.pool__ID].append(h.t)
                # Generate spike in DD neuron
                spike_time = h.t + np.clip(RANDOM_GENERATOR.normal(0, 1), 0, None)
                if spike_time < h.tstop:  # Avoid scheduling beyond simulation end
                    dd_netcons[dd_cell.pool__ID].event(spike_time)

        # Progress simulation
        h.fadvance()
        step_counter += 1
        pbar.update(timestep)


##############################################################################
# Convert Spike Data to Neo Format
# ---------------------------------
#

spike_train_block = Block(name="Sinusoidal DD Spike Trains")

dd_segment = Segment(name="Descending Drive")
dd_segment.spiketrains = [
    SpikeTrain(
        spike_times * pq.ms,
        t_stop=simulation_time * pq.ms,
        sampling_rate=(1 / h.dt * (1 / pq.ms)),
        sampling_period=h.dt * pq.ms,
        name=f"DD_{i}",
    )
    for i, spike_times in enumerate(dd_spike_times)
]

mn_segment = Segment(name="Motor Neurons")
mn_segment.spiketrains = [
    SpikeTrain(
        recorder.as_numpy() * pq.ms,
        t_stop=simulation_time * pq.ms,
        sampling_rate=(1 / h.dt * (1 / pq.ms)),
        sampling_period=h.dt * pq.ms,
        name=f"MN_{i}",
    )
    for i, recorder in enumerate(mn_spike_recorders)
]

# We only save the motor neuron spikes  segment
spike_train_block.segments.append(mn_segment)

joblib.dump(spike_train_block, save_path / "sinusoidal_dd_spike_trains.pkl")

##############################################################################
# Calculate Firing Rate Statistics
# ---------------------------------
#

print("\nFiring rate analysis:")

# Calculate DD firing rates
dd_firing_rates = np.array(
    [
        elephant.statistics.mean_firing_rate(st__s.time_slice(st__s.min(), st__s.max()))
        for st__ms in dd_segment.spiketrains
        if len(st__s := st__ms.rescale(pq.s)) > 0
    ]
)

# Calculate MN firing rates
mn_firing_rates = np.array(
    [
        elephant.statistics.mean_firing_rate(st__s.time_slice(st__s.min(), st__s.max()))
        for st__ms in mn_segment.spiketrains
        if len(st__s := st__ms.rescale(pq.s)) > 0
    ]
)

print("Descending Drive neurons:")
print(f"\tActive neurons: {len(dd_firing_rates)}/{descending_drive_pool.n}")
if len(dd_firing_rates) > 0:
    print(
        f"\tMean firing rate: {np.mean(dd_firing_rates):.1f} ± {np.std(dd_firing_rates):.1f} Hz"
    )
    print(
        f"\tRate range: {np.min(dd_firing_rates):.1f} - {np.max(dd_firing_rates):.1f} Hz"
    )

print("Motor neurons:")
print(f"\tActive neurons: {len(mn_firing_rates)}/{motor_neuron_pool.n}")
if len(mn_firing_rates) > 0:
    print(
        f"\tMean firing rate: {np.mean(mn_firing_rates):.1f} ± {np.std(mn_firing_rates):.1f} Hz"
    )
    print(
        f"\tRate range: {np.min(mn_firing_rates):.1f} - {np.max(mn_firing_rates):.1f} Hz"
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

# Create figure with subplots
fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

# 1. Plot sinusoidal drive pattern
time_s = sinusoidal_drive.times.rescale(pq.s).magnitude
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
axes[1].set_title(f"Descending Drive Population Activity (n={descending_drive_pool.n})")
axes[1].set_ylim(-1, descending_drive_pool.n)
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
    f"Motor Neuron Population Activity (n={active_mn_count}/{motor_neuron_pool.n} active)"
)
axes[2].set_ylim(-1, motor_neuron_pool.n)
axes[2].grid(True, alpha=0.3)

# 4. Population firing rates over time (binned)
bin_size_ms = 100

dd_psth = elephant.statistics.time_histogram(
    dd_segment.spiketrains, bin_size_ms * pq.ms
)
dd_rates_binned = (
    (dd_psth / (bin_size_ms * pq.ms) / descending_drive_pool.n).rescale(pq.Hz).magnitude
)

mn_psth = elephant.statistics.time_histogram(
    mn_segment.spiketrains, bin_size_ms * pq.ms
)
mn_rates_binned = (
    (mn_psth / (bin_size_ms * pq.ms) / motor_neuron_pool.n).rescale(pq.Hz).magnitude
)

bin_centers_s = dd_psth.times.rescale(pq.s).magnitude
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
plt.show()
