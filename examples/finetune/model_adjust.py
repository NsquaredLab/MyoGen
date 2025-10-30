"""
Motor Neuron Model Adjustment and Validation
=============================================

This script performs model parameter adjustment and validation by comparing simulated motor neuron
behavior with experimental data. It uses descending drive populations to generate realistic spike
trains and analyzes the resulting firing rate and ISI statistics against empirical measurements.

.. note::
    This script uses the following components for model adjustment:

    - **DescendingDrive__Pool**: Poisson process neurons modeling cortical input
    - **AlphaMN__Pool**: Biophysically detailed motor neurons (Powers2017 model)
    - **Network**: Synaptic connections between descending drive and motor neuron populations
    - **ForceModel**: Conversion of spike trains to force output
    - **ISI Analysis**: Comparison with experimental data from ISI_statistics.csv

.. important::
    **Purpose**: This script validates that the motor neuron model parameters (recruitment thresholds,
    synaptic weights, neural dynamics) produce physiologically realistic firing patterns that match
    experimental observations from multiple muscles (VL, VM, TA, FDI).
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

# %%

import itertools
from pathlib import Path

import elephant
import joblib
import numpy as np
import pandas as pd
import quantities as pq
import scienceplots  # noqa
import seaborn as sns
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from neo import AnalogSignal, Block, Segment, SpikeTrain
from neuron import h
from tqdm import tqdm

from myogen import simulator, RANDOM_GENERATOR
from myogen.simulator.core.force.force_model import ForceModel
from myogen.simulator.neuron import Network
from myogen.simulator.neuron.populations import AlphaMN__Pool, DescendingDrive__Pool
from myogen.utils.nmodl import load_nmodl_mechanisms

# Configure matplotlib style
plt.style.use(["science", "nature"])
sns.set_context("paper", font_scale=2)

# Disable LaTeX rendering (not required)
plt.rcParams["text.usetex"] = False
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.figsize"] = (10, 4)

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
# Load NEURON Mechanisms
# ----------------------
#
# Compile and load the NMODL mechanisms required for biophysically detailed neuron simulations.
# This must be done before creating any neuron populations.
#
load_nmodl_mechanisms()

h.secondorder = 2  # Crank-Nicolson method (second-order accurate)

save_path = Path("./results")
save_path.mkdir(exist_ok=True)


##############################################################################
# Generate Recruitment Thresholds
# --------------------------------
#
# Generate recruitment thresholds using the combined model from example 0.
# This creates physiologically realistic recruitment patterns.
#
# The Combined model merges De Luca's shape control with Konstantin's scaling,
# offering the most flexibility for custom recruitment patterns.

n_motor_units = 100  # Number of motor units in the pool (matches example 00)
recruitment_range = 100  # Recruitment range (max_threshold / min_threshold)
combined_max_threshold = 1.0  # Maximum threshold for combined model

# Generate thresholds using combined model with slope=5 (same as example 00)
recruitment_thresholds, _ = simulator.RecruitmentThresholds(
    N=n_motor_units,
    recruitment_range__ratio=recruitment_range,
    deluca__slope=5,
    konstantin__max_threshold__ratio=combined_max_threshold,
    mode="combined",
)

# Save the generated thresholds
joblib.dump(recruitment_thresholds, save_path / "thresholds.pkl")

##############################################################################
# Create Neuron Populations
# --------------------------
#
# Create the motor neuron pool and descending drive population with the following
# simulation parameters:
#
# - ``timestep``: 0.1 ms (high temporal resolution for accurate spike timing)
# - ``n_dd_neurons``: 400 descending drive neurons (Poisson processes)
# - ``motor_neuron_pool``: AlphaMN pool using VLVM configuration
#

timestep = 0.1  # ms
h.secondorder = 2  # Crank-Nicolson method (second-order accurate)
# Create three motor neuron pools for different muscles
motor_neuron_pool_VLVM = AlphaMN__Pool(
    recruitment_thresholds__array=recruitment_thresholds,
)

motor_neuron_pool_TA = AlphaMN__Pool(
    recruitment_thresholds__array=recruitment_thresholds,
)

motor_neuron_pool_FDI = AlphaMN__Pool(
    recruitment_thresholds__array=recruitment_thresholds,
)

descending_drive_pool = DescendingDrive__Pool(
    n=460, poisson_batch_size=16, timestep__ms=timestep
)

##############################################################################
# Generate Drive Pattern
# -----------------------
#
# Create a time-varying input pattern to drive the descending drive population's Poisson processes.
# This pattern represents cortical motor commands with the following characteristics:
#
# - **Baseline activity**: Continuous low-level drive at 20 Hz
# - **Sinusoidal modulation**: 0.25 Hz oscillation (disabled in current configuration)
# - **Gaussian noise**: Small random fluctuations (std=1.0 Hz) for realism
#
# The pattern is generated at the simulation timestep resolution and provided as an AnalogSignal.
#


simulation_time = 15000  # ms
time_points = int(simulation_time / timestep)

# Trapezoidal parameters
dd_baseline__Hz = 0.0  # Baseline drive during rest
dd_peak__Hz = 87.03744066199334  # Peak drive during plateau

# Phase durations (ms) - Total trapezoid duration: 13000ms
ramp_up_duration = 500  # 2s ramp up
plateau_duration = 10000  # 9s hold
ramp_down_duration = 500  # 2s ramp down

# Add rest periods before and after
rest_before = 1000  # 1s rest before trapezoid
rest_after = 1000  # 1s rest after trapezoid

# Center the trapezoid at 7.5s (middle of 15s simulation)
# Calculate phase boundaries with rest period before
trapezoid_start = rest_before  # Start at 1s
ramp_up_end = trapezoid_start + ramp_up_duration  # 3s
plateau_end = ramp_up_end + plateau_duration  # 12s
ramp_down_end = plateau_end + ramp_down_duration  # 14s

# Create time array
time_array = np.linspace(0, simulation_time, time_points)

# Initialize drive signal (all baseline)
trapezoid_drive = np.ones(time_points) * dd_baseline__Hz

for i, t in enumerate(time_array):
    if t < trapezoid_start:
        # Phase 0: Rest before
        trapezoid_drive[i] = dd_baseline__Hz
    elif t < ramp_up_end:
        # Phase 1: Ramp up
        elapsed = t - trapezoid_start
        trapezoid_drive[i] = dd_baseline__Hz + (dd_peak__Hz - dd_baseline__Hz) * (
            elapsed / ramp_up_duration
        )
    elif t < plateau_end:
        # Phase 2: Plateau
        trapezoid_drive[i] = dd_peak__Hz
    elif t < ramp_down_end:
        # Phase 3: Ramp down
        elapsed = t - plateau_end
        trapezoid_drive[i] = dd_peak__Hz - (dd_peak__Hz - dd_baseline__Hz) * (
            elapsed / ramp_down_duration
        )
    else:
        # Phase 4: Rest after
        trapezoid_drive[i] = dd_baseline__Hz

# Add small noise for realism
trapezoid_drive = trapezoid_drive + np.clip(
    RANDOM_GENERATOR.normal(0, 1.0, size=time_points), 0, None
)

# Create AnalogSignal
sinusoidal_drive = AnalogSignal(
    signal=trapezoid_drive,
    units=pq.Hz,
    sampling_period=(timestep * pq.ms).rescale(pq.s),
)

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
# population and the three motor neuron pools. Synaptic weights are tuned individually
# per muscle to match experimental ISI statistics:
#
# **Tuning Rationale:**
# All muscles share the same recruitment thresholds and descending drive pattern, but
# experimental data shows different firing rate ranges:
# - VLVM (VL+VM): 7.6 Hz mean firing rate, CV 0.10
# - TA: 13.1 Hz mean firing rate (+70% vs VLVM), CV 0.116
# - FDI: 12.2 Hz mean firing rate (+60% vs VLVM), CV 0.188
#
# Synaptic weights from DD→MN are adjusted proportionally to achieve target firing rates:
# - VLVM: 0.1 μS (baseline reference)
# - TA: 0.17 μS (1.7× multiplier)
# - FDI: 0.16 μS (1.6× multiplier)
#

network = Network(
    {
        "DD": descending_drive_pool,
        "aMN_VLVM": motor_neuron_pool_VLVM,
        "aMN_TA": motor_neuron_pool_TA,
        "aMN_FDI": motor_neuron_pool_FDI,
    }
)

# Connect DD neurons to all three motor neuron pools with muscle-specific synaptic weights
# Synaptic weights tuned to match experimental firing rate targets:
# - VLVM: 0.1 μS (baseline, target FR ~7.6 Hz)
# - TA: 0.17 μS (1.7× increase, target FR ~13.1 Hz, +70%)
# - FDI: 0.16 μS (1.6× increase, target FR ~12.2 Hz, +60%)
network.connect(
    source="DD", target="aMN_VLVM", probability=0.5367791765777158, weight__μS=0.05
)
network.connect(
    source="DD", target="aMN_TA", probability=0.5367791765777158, weight__μS=0.05
)
network.connect(
    source="DD", target="aMN_FDI", probability=0.5367791765777158, weight__μS=0.05
)

# Set up external input to DD population
network.connect_from_external(source="cortical_input", target="DD", weight__μS=1.0)

# Get NetCons for manual DD stimulation
dd_netcons = network.get_netcons("cortical_input", "DD")

##############################################################################
# Setup Spike Recording
# ----------------------
#
# Configure spike recording for both populations:
# - Motor neurons: Use NetCon objects with voltage threshold detection (50 mV)
# - Descending drive neurons: Track spike times manually during simulation loop
#

# Manual spike tracking for DD neurons (they use Poisson processes)
dd_spike_times = [[] for _ in range(len(descending_drive_pool))]

# Record spikes from all three motor neuron pools
mn_spike_recorders_VLVM = []
for cell in motor_neuron_pool_VLVM:
    spike_recorder = h.Vector()
    nc = h.NetCon(cell.soma(0.5)._ref_v, None, sec=cell.soma)
    nc.threshold = 50  # Standard threshold for motor neurons
    nc.record(spike_recorder)
    mn_spike_recorders_VLVM.append(spike_recorder)

mn_spike_recorders_TA = []
for cell in motor_neuron_pool_TA:
    spike_recorder = h.Vector()
    nc = h.NetCon(cell.soma(0.5)._ref_v, None, sec=cell.soma)
    nc.threshold = 50  # Standard threshold for motor neurons
    nc.record(spike_recorder)
    mn_spike_recorders_TA.append(spike_recorder)

mn_spike_recorders_FDI = []
for cell in motor_neuron_pool_FDI:
    spike_recorder = h.Vector()
    nc = h.NetCon(cell.soma(0.5)._ref_v, None, sec=cell.soma)
    nc.threshold = 50  # Standard threshold for motor neurons
    nc.record(spike_recorder)
    mn_spike_recorders_FDI.append(spike_recorder)

##############################################################################
# Run Simulation
# --------------
#
# Execute the NEURON simulation with real-time injection of the drive pattern into descending
# drive neurons. At each timestep, the DD Poisson processes integrate the current drive level
# and may generate spikes, which then propagate through the synaptic network to motor neurons.

h.load_file("stdrun.hoc")  # Load standard run library for NEURON
h.dt = timestep
h.tstop = simulation_time

# Initialize voltages for all pools
for section, voltage in itertools.chain.from_iterable(
    zip(*pool.get_initialization_data())
    for pool in [
        motor_neuron_pool_VLVM,
        motor_neuron_pool_TA,
        motor_neuron_pool_FDI,
        descending_drive_pool,
    ]
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
                spike_time = h.t + np.clip(RANDOM_GENERATOR.normal(0, 10), 0, None)
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
# Convert recorded spike times from all populations into Neo SpikeTrain objects.
# Create separate segments for each muscle type (VLVM, TA, FDI) and separate blocks
# for subsequent force generation and analysis.
#

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

# Create separate segments for each muscle type
mn_segment_VLVM = Segment(name="Motor Neurons VLVM")
mn_segment_VLVM.spiketrains = [
    SpikeTrain(
        recorder.as_numpy() * pq.ms,
        t_stop=simulation_time * pq.ms,
        sampling_rate=(1 / h.dt * (1 / pq.ms)),
        sampling_period=h.dt * pq.ms,
        name=f"MN_VLVM_{i}",
    )
    for i, recorder in enumerate(mn_spike_recorders_VLVM)
]

mn_segment_TA = Segment(name="Motor Neurons TA")
mn_segment_TA.spiketrains = [
    SpikeTrain(
        recorder.as_numpy() * pq.ms,
        t_stop=simulation_time * pq.ms,
        sampling_rate=(1 / h.dt * (1 / pq.ms)),
        sampling_period=h.dt * pq.ms,
        name=f"MN_TA_{i}",
    )
    for i, recorder in enumerate(mn_spike_recorders_TA)
]

mn_segment_FDI = Segment(name="Motor Neurons FDI")
mn_segment_FDI.spiketrains = [
    SpikeTrain(
        recorder.as_numpy() * pq.ms,
        t_stop=simulation_time * pq.ms,
        sampling_rate=(1 / h.dt * (1 / pq.ms)),
        sampling_period=h.dt * pq.ms,
        name=f"MN_FDI_{i}",
    )
    for i, recorder in enumerate(mn_spike_recorders_FDI)
]

# Create separate blocks for each muscle
spike_train_block_VLVM = Block(name="Motor Neuron Spike Trains VLVM")
spike_train_block_VLVM.segments.append(mn_segment_VLVM)

spike_train_block_TA = Block(name="Motor Neuron Spike Trains TA")
spike_train_block_TA.segments.append(mn_segment_TA)

spike_train_block_FDI = Block(name="Motor Neuron Spike Trains FDI")
spike_train_block_FDI.segments.append(mn_segment_FDI)

# Save spike trains for each muscle
joblib.dump(spike_train_block_VLVM, save_path / "VLVM_spike_trains.pkl")
joblib.dump(spike_train_block_TA, save_path / "TA_spike_trains.pkl")
joblib.dump(spike_train_block_FDI, save_path / "FDI_spike_trains.pkl")

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

# Calculate MN firing rates for each muscle
mn_firing_rates_VLVM = np.array(
    [
        elephant.statistics.mean_firing_rate(st__s.time_slice(st__s.min(), st__s.max()))
        for st__ms in mn_segment_VLVM.spiketrains
        if len(st__s := st__ms.rescale(pq.s)) > 0
    ]
)

mn_firing_rates_TA = np.array(
    [
        elephant.statistics.mean_firing_rate(st__s.time_slice(st__s.min(), st__s.max()))
        for st__ms in mn_segment_TA.spiketrains
        if len(st__s := st__ms.rescale(pq.s)) > 0
    ]
)

mn_firing_rates_FDI = np.array(
    [
        elephant.statistics.mean_firing_rate(st__s.time_slice(st__s.min(), st__s.max()))
        for st__ms in mn_segment_FDI.spiketrains
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

print("\nMotor neurons (VLVM):")
print(f"\tActive neurons: {len(mn_firing_rates_VLVM)}/{motor_neuron_pool_VLVM.n}")
if len(mn_firing_rates_VLVM) > 0:
    print(
        f"\tMean firing rate: {np.mean(mn_firing_rates_VLVM):.1f} ± {np.std(mn_firing_rates_VLVM):.1f} Hz"
    )
    print(
        f"\tRate range: {np.min(mn_firing_rates_VLVM):.1f} - {np.max(mn_firing_rates_VLVM):.1f} Hz"
    )

print("\nMotor neurons (TA):")
print(f"\tActive neurons: {len(mn_firing_rates_TA)}/{motor_neuron_pool_TA.n}")
if len(mn_firing_rates_TA) > 0:
    print(
        f"\tMean firing rate: {np.mean(mn_firing_rates_TA):.1f} ± {np.std(mn_firing_rates_TA):.1f} Hz"
    )
    print(
        f"\tRate range: {np.min(mn_firing_rates_TA):.1f} - {np.max(mn_firing_rates_TA):.1f} Hz"
    )

print("\nMotor neurons (FDI):")
print(f"\tActive neurons: {len(mn_firing_rates_FDI)}/{motor_neuron_pool_FDI.n}")
if len(mn_firing_rates_FDI) > 0:
    print(
        f"\tMean firing rate: {np.mean(mn_firing_rates_FDI):.1f} ± {np.std(mn_firing_rates_FDI):.1f} Hz"
    )
    print(
        f"\tRate range: {np.min(mn_firing_rates_FDI):.1f} - {np.max(mn_firing_rates_FDI):.1f} Hz"
    )

##############################################################################
# Network Activity Visualization
# -------------------------------
#
# Streamlined visualization for model adjustment showing:
# 1. Drive input pattern (baseline + noise)
# 2. Motor neuron raster plot (recruitment order)
# 3. Population firing rates over time (binned)

time_s = sinusoidal_drive.times.rescale(pq.s).magnitude

# Plot 1: Drive pattern
fig1, ax1 = plt.subplots()
ax1.plot(time_s, sinusoidal_drive, color="#90b8e0", linewidth=2, label="DD Input")
ax1.axhline(
    dd_baseline__Hz,
    color="#af8bff",
    linestyle="--",
    linewidth=2,
    alpha=0.7,
    label="Baseline",
)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Drive (Hz)")
ax1.legend(frameon=False)
ax1.set_xlim(0, simulation_time / 1000.0)
ax1.set_xticks(range(int(simulation_time / 1000.0) + 1))
sns.despine(ax=ax1, offset=10, trim=True)
plt.tight_layout()
plt.savefig(save_path / "model_adjust_drive_pattern.svg", transparent=True)
plt.show()

# Plot 2: Motor neuron raster plots (recruitment ordered) for each muscle
fig2, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# VLVM
for i, spiketrain in enumerate(mn_segment_VLVM.spiketrains):
    if len(spiketrain) > 0:
        spike_times = spiketrain.rescale(pq.s).magnitude
        axes[0].scatter(
            spike_times,
            [i] * len(spike_times),
            color="#E63946",
            s=1.0,
            alpha=0.6,
        )

axes[0].set_ylabel("MN ID (VLVM)")
axes[0].set_ylim(-1, motor_neuron_pool_VLVM.n)
axes[0].set_yticks([1, motor_neuron_pool_VLVM.n // 2, motor_neuron_pool_VLVM.n])
sns.despine(ax=axes[0], offset=10, trim=True)

# TA
for i, spiketrain in enumerate(mn_segment_TA.spiketrains):
    if len(spiketrain) > 0:
        spike_times = spiketrain.rescale(pq.s).magnitude
        axes[1].scatter(
            spike_times,
            [i] * len(spike_times),
            color="#06A77D",
            s=1.0,
            alpha=0.6,
        )

axes[1].set_ylabel("MN ID (TA)")
axes[1].set_ylim(-1, motor_neuron_pool_TA.n)
axes[1].set_yticks([1, motor_neuron_pool_TA.n // 2, motor_neuron_pool_TA.n])
sns.despine(ax=axes[1], offset=10, trim=True)

# FDI
for i, spiketrain in enumerate(mn_segment_FDI.spiketrains):
    if len(spiketrain) > 0:
        spike_times = spiketrain.rescale(pq.s).magnitude
        axes[2].scatter(
            spike_times,
            [i] * len(spike_times),
            color="#4361EE",
            s=1.0,
            alpha=0.6,
        )

axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("MN ID (FDI)")
axes[2].set_ylim(-1, motor_neuron_pool_FDI.n)
axes[2].set_xlim(0, simulation_time / 1000.0)
axes[2].set_xticks(range(int(simulation_time / 1000.0) + 1))
axes[2].set_yticks([1, motor_neuron_pool_FDI.n // 2, motor_neuron_pool_FDI.n])
sns.despine(ax=axes[2], offset=10, trim=True)

plt.tight_layout()
plt.savefig(save_path / "model_adjust_motor_neuron_rasters.svg", transparent=True)
plt.show()

# Plot 3: Population firing rates over time
bin_size_ms = 100
dd_psth = elephant.statistics.time_histogram(
    dd_segment.spiketrains, bin_size_ms * pq.ms
)
mn_psth_VLVM = elephant.statistics.time_histogram(
    mn_segment_VLVM.spiketrains, bin_size_ms * pq.ms
)
mn_psth_TA = elephant.statistics.time_histogram(
    mn_segment_TA.spiketrains, bin_size_ms * pq.ms
)
mn_psth_FDI = elephant.statistics.time_histogram(
    mn_segment_FDI.spiketrains, bin_size_ms * pq.ms
)

dd_rates_binned = (
    (dd_psth / (bin_size_ms * pq.ms) / descending_drive_pool.n).rescale(pq.Hz).magnitude
)
mn_rates_binned_VLVM = (
    (mn_psth_VLVM / (bin_size_ms * pq.ms) / motor_neuron_pool_VLVM.n)
    .rescale(pq.Hz)
    .magnitude
)
mn_rates_binned_TA = (
    (mn_psth_TA / (bin_size_ms * pq.ms) / motor_neuron_pool_TA.n)
    .rescale(pq.Hz)
    .magnitude
)
mn_rates_binned_FDI = (
    (mn_psth_FDI / (bin_size_ms * pq.ms) / motor_neuron_pool_FDI.n)
    .rescale(pq.Hz)
    .magnitude
)
bin_centers_s = dd_psth.times.rescale(pq.s).magnitude

fig3, ax3 = plt.subplots()
ax3.plot(
    bin_centers_s,
    dd_rates_binned,
    color="#90b8e0",
    linewidth=2,
    label="DD Population",
    alpha=0.8,
)
ax3.plot(
    bin_centers_s,
    mn_rates_binned_VLVM,
    color="#E63946",
    linewidth=2,
    label="MN VLVM",
    alpha=0.8,
)
ax3.plot(
    bin_centers_s,
    mn_rates_binned_TA,
    color="#06A77D",
    linewidth=2,
    label="MN TA",
    alpha=0.8,
)
ax3.plot(
    bin_centers_s,
    mn_rates_binned_FDI,
    color="#4361EE",
    linewidth=2,
    label="MN FDI",
    alpha=0.8,
)
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Population Rate (Hz)")
ax3.legend(frameon=False)
ax3.set_xlim(0, simulation_time / 1000.0)
ax3.set_xticks(range(int(simulation_time / 1000.0) + 1))
y_max_rate = max(
    np.max(dd_rates_binned),
    np.max(mn_rates_binned_VLVM),
    np.max(mn_rates_binned_TA),
    np.max(mn_rates_binned_FDI),
)
ax3.set_yticks([0, y_max_rate / 2, y_max_rate])
ax3.set_yticklabels([f"{0:.0f}", f"{y_max_rate / 2:.1f}", f"{y_max_rate:.1f}"])
sns.despine(ax=ax3, offset=10, trim=True)
plt.tight_layout()
plt.savefig(save_path / "model_adjust_population_firing_rates.svg", transparent=True)
plt.show()

##############################################################################
# Helper Function: Calculate ISI Statistics
# ------------------------------------------
#


def calculate_isi_statistics(spiketrains, simulation_time_ms, plateau_start_ms=None, plateau_end_ms=None):
    """
    Calculate inter-spike interval (ISI) statistics from spike trains.

    IMPORTANT: ISI and CV should only be computed during the plateau phase
    where firing is stable, not during ramp-up/down where rates change by design.

    Parameters
    ----------
    spiketrains : list of neo.SpikeTrain
        List of spike train objects to analyze.
    simulation_time_ms : float
        Total simulation duration in milliseconds.
    plateau_start_ms : float, optional
        Start time of plateau phase in milliseconds.
        If provided along with plateau_end_ms, only spikes within
        the plateau phase will be used for ISI/CV calculation.
    plateau_end_ms : float, optional
        End time of plateau phase in milliseconds.

    Returns
    -------
    tuple
        Arrays of (firing_rates, cv_values, neuron_indices) for neurons with valid ISIs.
    """
    simulated_fr = []
    simulated_cv = []
    simulated_neuron_idx = []

    duration_s = simulation_time_ms / 1000.0

    for idx, spiketrain in enumerate(spiketrains):
        # Filter to plateau phase if boundaries are provided
        if plateau_start_ms is not None and plateau_end_ms is not None:
            # Use time_slice to extract only plateau spikes
            plateau_spiketrain = spiketrain.time_slice(
                plateau_start_ms * pq.ms, plateau_end_ms * pq.ms
            )
            # Recalculate duration for plateau phase only
            duration_s = (plateau_end_ms - plateau_start_ms) / 1000.0
        else:
            plateau_spiketrain = spiketrain

        if len(plateau_spiketrain) > 2:  # Need at least 3 spikes for meaningful CV
            # Compute mean firing rate (Hz) from spike count over duration
            mean_rate = len(plateau_spiketrain) / duration_s

            # Compute CV of inter-spike intervals (only from plateau phase)
            spike_times_s = plateau_spiketrain.rescale(pq.s).magnitude
            isis = np.diff(spike_times_s)
            cv = np.std(isis) / np.mean(isis) if len(isis) > 1 else 0.0

            # Only include neurons with meaningful firing rates
            if mean_rate >= 0.01:
                simulated_fr.append(mean_rate)
                simulated_cv.append(cv)
                simulated_neuron_idx.append(idx)

    return (
        np.array(simulated_fr),
        np.array(simulated_cv),
        np.array(simulated_neuron_idx),
    )


##############################################################################
# Plot ISI Statistics with Recruitment Thresholds
# ------------------------------------------------
#
# This function creates a comprehensive visualization combining:
# 1. Generated recruitment thresholds from the simulation
# 2. Experimental ISI statistics from ISI_statistics.csv
#
# The plot shows how the model's recruitment thresholds compare with
# real experimental data from different muscles and force levels.


def plot_isi_statistics_with_thresholds():
    """
    Plot ISI statistics from CSV file alongside generated recruitment thresholds.

    This function loads experimental ISI data and plots it together with the
    recruitment thresholds generated using the combined model.
    """
    # Load ISI statistics from CSV
    csv_path = Path(__file__).parent / "ISI_statistics.csv"
    isi_data = pd.read_csv(csv_path)

    # Calculate ISI statistics from simulated spike trains for each muscle
    # IMPORTANT: Only analyze plateau phase where firing is stable
    simulated_fr_VLVM, simulated_cv_VLVM, simulated_neuron_idx_VLVM = (
        calculate_isi_statistics(
            mn_segment_VLVM.spiketrains,
            simulation_time,
            plateau_start_ms=ramp_up_end,
            plateau_end_ms=plateau_end
        )
    )
    simulated_fr_TA, simulated_cv_TA, simulated_neuron_idx_TA = (
        calculate_isi_statistics(
            mn_segment_TA.spiketrains,
            simulation_time,
            plateau_start_ms=ramp_up_end,
            plateau_end_ms=plateau_end
        )
    )
    simulated_fr_FDI, simulated_cv_FDI, simulated_neuron_idx_FDI = (
        calculate_isi_statistics(
            mn_segment_FDI.spiketrains,
            simulation_time,
            plateau_start_ms=ramp_up_end,
            plateau_end_ms=plateau_end
        )
    )

    # Color palette for muscles (map experimental VM/VL to simulated VLVM)
    muscles = isi_data["Muscle"].unique()
    colors = {"VM": "#E63946", "VL": "#F77F00", "TA": "#06A77D", "FDI": "#4361EE"}

    # Simulated muscle colors
    sim_colors = {"VLVM": "#E63946", "TA": "#06A77D", "FDI": "#4361EE"}

    # Plot 1: Recruitment Thresholds
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(
        range(1, len(recruitment_thresholds) + 1),
        recruitment_thresholds,
        "o-",
        linewidth=2,
        markersize=6,
        color="#2E86AB",
        label="Generated Thresholds",
    )
    ax1.set_xlabel("Motor Unit Index", fontsize=12)
    ax1.set_ylabel("Recruitment Threshold", fontsize=12)
    ax1.tick_params(axis="both", labelsize=10)
    ax1.legend(frameon=False, fontsize=9)
    sns.despine(ax=ax1, offset=10, trim=True)
    plt.tight_layout()
    plt.savefig(save_path / "isi_recruitment_thresholds.svg", transparent=True)
    plt.show()

    # Plot 2: Mean Firing Rate by Muscle and Force Level
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    for muscle in muscles:
        muscle_data = isi_data[isi_data["Muscle"] == muscle]
        force_levels = muscle_data["Force Level"].unique()

        for force in sorted(force_levels):
            force_data = muscle_data[muscle_data["Force Level"] == force]
            mean_fr = force_data["FR mean"].mean()
            std_fr = force_data["FR mean"].std()

            ax2.errorbar(
                force,
                mean_fr,
                yerr=std_fr,
                marker="o",
                markersize=6,
                color=colors.get(muscle, "#000000"),
                label=f"{muscle}" if force == sorted(force_levels)[0] else "",
                alpha=0.7,
            )

    ax2.set_xlabel("Force Level (%)", fontsize=12)
    ax2.set_ylabel("Mean Firing Rate (pps)", fontsize=12)
    ax2.tick_params(axis="both", labelsize=10)
    ax2.legend(frameon=False, loc="best", fontsize=9)
    sns.despine(ax=ax2, offset=10, trim=True)
    plt.tight_layout()
    plt.savefig(save_path / "isi_firing_rates_by_force.svg", transparent=True)
    plt.show()

    # Plot 3: ISI Coefficient of Variation (CV) Distribution (Ridge Plot)
    fig3, ax3 = plt.subplots(figsize=(12, 4))

    from scipy import stats

    # Group muscles: combine VL and VM into VLVM
    grouped_muscles = ["VLVM", "TA", "FDI"]
    grouped_colors = {"VLVM": "#E63946", "TA": "#06A77D", "FDI": "#4361EE"}

    # Create ridge plot with vertical stacking
    y_offset = 0
    y_spacing = 0.8
    muscle_positions = {}

    for muscle_group in reversed(
        grouped_muscles
    ):  # Reverse to have first muscle on top
        # Combine data for VL and VM into VLVM
        if muscle_group == "VLVM":
            vlvm_data = isi_data[isi_data["Muscle"].isin(["VL", "VM"])]
            experimental_cv_data = vlvm_data["ISI CV"].values
        else:
            muscle_data = isi_data[isi_data["Muscle"] == muscle_group]
            experimental_cv_data = muscle_data["ISI CV"].values

        # Calculate KDE for experimental data
        kde = stats.gaussian_kde(experimental_cv_data, bw_method=0.25)
        x_range = np.linspace(
            experimental_cv_data.min() - 0.1, experimental_cv_data.max() + 0.1, 200
        )
        density = kde(x_range)

        # Normalize and offset
        density_normalized = density / density.max() * 0.6  # Scale to 0.6 height
        density_offset = density_normalized + y_offset

        # Plot filled KDE
        ax3.fill_between(
            x_range,
            y_offset,
            density_offset,
            color=grouped_colors[muscle_group],
            alpha=0.6,
            linewidth=0,
        )

        # Plot outline
        ax3.plot(
            x_range,
            density_offset,
            color=grouped_colors[muscle_group],
            linewidth=2.5,
            label=f"{muscle_group} (Exp)",
        )

        # Store position for y-tick
        muscle_positions[muscle_group] = y_offset + 0.4

        y_offset += y_spacing

    # Add simulated data overlays in gray (using CV from ISI calculations)
    sim_y_offset = 0
    sim_cv_data_dict = {
        "VLVM": simulated_cv_VLVM,
        "TA": simulated_cv_TA,
        "FDI": simulated_cv_FDI,
    }

    for muscle_group in reversed(grouped_muscles):
        simulated_cv_data = sim_cv_data_dict.get(muscle_group)
        if simulated_cv_data is not None and len(simulated_cv_data) > 0:
            # Calculate KDE for simulated CV data
            sim_kde = stats.gaussian_kde(simulated_cv_data, bw_method=0.25)
            sim_x_range = np.linspace(
                max(0, simulated_cv_data.min() - 0.1),
                simulated_cv_data.max() + 0.1,
                200,
            )
            sim_density = sim_kde(sim_x_range)

            # Normalize and offset
            sim_density_normalized = sim_density / sim_density.max() * 0.6
            sim_density_offset = sim_density_normalized + sim_y_offset

            # Plot simulated data in gray with transparency
            ax3.plot(
                sim_x_range,
                sim_density_offset,
                color="gray",
                linewidth=2.0,
                linestyle="--",
                alpha=0.7,
                label=f"{muscle_group} (Sim)"
                if muscle_group == grouped_muscles[-1]
                else "",
            )

        sim_y_offset += y_spacing

    ax3.set_xlabel("Coefficient of Variation (CV)", fontsize=12)
    ax3.set_ylabel("")
    ax3.set_xlim(left=0)
    ax3.set_yticks(list(muscle_positions.values()))
    ax3.set_yticklabels(list(muscle_positions.keys()), fontsize=10)
    ax3.tick_params(axis="x", labelsize=10)
    ax3.legend(frameon=False, loc="upper right", fontsize=9)
    sns.despine(ax=ax3, left=True, offset=10, trim=True)
    plt.tight_layout()
    plt.savefig(save_path / "isi_cv_distribution.svg", transparent=True)
    plt.show()

    # Plot 4: CV vs Firing Rate - Create separate plots for each muscle
    from scipy.spatial import ConvexHull
    from matplotlib.patches import Polygon

    # Helper function to create individual muscle plots
    def plot_muscle_cv_vs_fr(muscle_name, sim_cv, sim_fr, sim_idx, save_name):
        """Create CV vs FR plot for a specific muscle, showing all experimental muscles."""
        fig, ax = plt.subplots(figsize=(4, 4))

        # Plot experimental data for ALL muscles
        for muscle in muscles:
            muscle_data = isi_data[isi_data["Muscle"] == muscle]
            cv_data = muscle_data["ISI CV"].values
            fr_data = muscle_data["FR mean"].values

            if len(cv_data) > 2:  # Need at least 3 points for convex hull
                points = np.column_stack([cv_data, fr_data])
                try:
                    hull = ConvexHull(points)
                    hull_points = points[hull.vertices]
                    polygon = Polygon(
                        hull_points,
                        facecolor=colors.get(muscle, "#000000"),
                        alpha=0.25,
                        edgecolor=colors.get(muscle, "#000000"),
                        linewidth=1.5,
                        linestyle="-",
                        zorder=0,
                    )
                    ax.add_patch(polygon)
                except Exception:
                    pass

            # Plot scatter points
            ax.scatter(
                cv_data,
                fr_data,
                s=20,
                alpha=1.0,
                color=colors.get(muscle, "#000000"),
                label=f"{muscle} (Exp)",
                edgecolors="white",
                linewidth=0.5,
                marker="x",
                zorder=1,
            )

        # Plot simulated data
        if len(sim_fr) > 0:
            if len(sim_idx) > 0:
                vmin_active = np.min(sim_idx)
                vmax_active = np.max(sim_idx)
            else:
                vmin_active = 0
                vmax_active = 1

            scatter_sim = ax.scatter(
                sim_cv,
                sim_fr,
                c=sim_idx,
                cmap="rainbow",
                s=25,
                alpha=0.9,
                vmin=vmin_active,
                vmax=vmax_active,
                edgecolors="black",
                linewidth=0.6,
                label=f"{muscle_name} (Sim)",
                zorder=2,
            )

            # Add colorbar
            cax = inset_axes(
                ax,
                width="5%",
                height="50%",
                loc="upper right",
                bbox_to_anchor=(0.05, 0, 1, 1),
                bbox_transform=ax.transAxes,
            )
            cbar = plt.colorbar(scatter_sim, cax=cax)
            cbar.ax.tick_params(labelsize=8)
            cbar.set_label("Motor Neuron ID", fontsize=10)

        ax.set_xlabel("Coefficient of Variation (CV)", fontsize=12)
        ax.set_ylabel("Mean Firing Rate (pps)", fontsize=12)
        ax.set_xlim(0, 0.5)
        ax.set_ylim(4, 25)
        ax.set_title(f"{muscle_name}", fontsize=14)
        ax.tick_params(axis="both", labelsize=10)
        ax.legend(frameon=False, fontsize=9, loc="upper left")
        sns.despine(ax=ax, offset=10, trim=True)
        plt.tight_layout()
        plt.savefig(save_path / save_name, transparent=True)
        plt.show()

    # Create three separate plots (each showing all experimental muscles)
    plot_muscle_cv_vs_fr(
        "VLVM",
        simulated_cv_VLVM,
        simulated_fr_VLVM,
        simulated_neuron_idx_VLVM,
        "isi_cv_vs_fr_VLVM.svg",
    )

    plot_muscle_cv_vs_fr(
        "TA",
        simulated_cv_TA,
        simulated_fr_TA,
        simulated_neuron_idx_TA,
        "isi_cv_vs_fr_TA.svg",
    )

    plot_muscle_cv_vs_fr(
        "FDI",
        simulated_cv_FDI,
        simulated_fr_FDI,
        simulated_neuron_idx_FDI,
        "isi_cv_vs_fr_FDI.svg",
    )

    print(f"\nPlots saved to: {save_path}")


# Call the plotting function
plot_isi_statistics_with_thresholds()
