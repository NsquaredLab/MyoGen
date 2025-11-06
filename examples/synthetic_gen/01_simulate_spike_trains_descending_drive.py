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

# %%

import itertools
from pathlib import Path

import elephant
import joblib
import numpy as np
import quantities as pq
from matplotlib import pyplot as plt
from neo import AnalogSignal, Block, Segment, SpikeTrain
from neuron import h
from noise import pnoise1
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
save_path = Path("/home/oj98yqyk/code/simulators/MyoGen/results/synthetic_gen")
save_path.mkdir(exist_ok=True)

recruitment_thresholds = joblib.load(save_path / "thresholds.pkl")

motor_neuron_pool = AlphaMN__Pool(
    recruitment_thresholds__array=recruitment_thresholds,
    config_file="alpha_mn_default.yaml",
)

##############################################################################
# Add Individual Noise to Each Motor Neuron
# ------------------------------------------
#
# Use the **Gfluctdv** mechanism to add stochastic conductance fluctuations
# to each motor neuron individually. This simulates synaptic background noise
# using Ornstein-Uhlenbeck processes for realistic temporal correlations.
#
# The noise parameters can be customized per neuron to create heterogeneous
# variability across the motor pool.

""" print("\nAdding noise to motor neurons...")
for i, cell in enumerate(motor_neuron_pool):
    # Insert Gfluctdv mechanism into soma
    cell.soma.insert("Gfluctdv")

    # Noise scaling based on neuron order
    noise_scale = 1.0 + 0.05 * (i / len(motor_neuron_pool))

    # SOMA: excitatory
    cell.soma.g_e0_Gfluctdv = 1e-6  # baseline mean conductance
    cell.soma.std_e_Gfluctdv = 2e-6 * noise_scale  # OU noise amplitude
    cell.soma.tau_e_Gfluctdv = 5.0  # correlation time (ms)
    cell.soma.E_e_Gfluctdv = 0.0

    # SOMA: inhibitory
    cell.soma.g_i0_Gfluctdv = 2e-6
    cell.soma.std_i_Gfluctdv = 4e-6 * noise_scale
    cell.soma.tau_i_Gfluctdv = 10.0
    cell.soma.E_i_Gfluctdv = -75.0

    # DENDRITES (optional, lower amplitude)
    if hasattr(cell, "dend") and len(cell.dend) > 0:
        for dendrite in cell.dend:
            dendrite.insert("Gfluctdv")
            dendrite.g_e0_Gfluctdv = 5e-7
            dendrite.std_e_Gfluctdv = 1e-6 * noise_scale
            dendrite.tau_e_Gfluctdv = 5.0
            dendrite.E_e_Gfluctdv = 0.0

            dendrite.g_i0_Gfluctdv = 1e-6
            dendrite.std_i_Gfluctdv = 2e-6 * noise_scale
            dendrite.tau_i_Gfluctdv = 10.0
            dendrite.E_i_Gfluctdv = -75.0


# Enable noise globally (GLOBAL variables - affect all Gfluctdv mechanisms)
# Note: multex and multin are GLOBAL parameters, not per-segment
h.multex_Gfluctdv = 1.0  # Enable excitatory noise (1.0 = on, 0.0 = off)
h.multin_Gfluctdv = 1.0  # Enable inhibitory noise (1.0 = on, 0.0 = off) """

timestep = 0.1  # ms
h.secondorder = 2  # Crank-Nicolson method (second-order accurate)
descending_drive_pool = DescendingDrive__Pool(
    n=400, poisson_batch_size=5, timestep__ms=timestep
)
##############################################################################
# Generate Trapezoidal Drive Pattern
# -----------------------------------
#
# Create a **trapezoidal ramp contraction pattern** that represents realistic
# voluntary isometric contractions. This pattern has 4 phases:
# 1. **Ramp-up**: Linear increase from baseline to peak
# 2. **Plateau**: Sustained peak drive level
# 3. **Ramp-down**: Linear decrease from peak to baseline
# 4. **Rest**: Baseline activity
#
# This is a common experimental paradigm used in motor control studies.

simulation_time = int(5000 * 2 + 10e3 + 2e3)  # ms
time_points = int(simulation_time / timestep)

# Trapezoidal parameters
dd_baseline__Hz = 0.0  # Baseline drive during rest
dd_peak__Hz = 65  # Peak drive during plateau

# Phase durations (ms) - Total trapezoid duration: 13000ms
ramp_up_duration = 5000  # 2s ramp up
plateau_duration = 10000  # 9s hold
ramp_down_duration = 5000  # 2s ramp down

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

# Add Perlin noise for realistic smooth temporal variations
# Generate Perlin noise with parameters tuned for biological variability
perlin_seed = int(RANDOM_GENERATOR.integers(0, 2**31 - 1))
perlin_noise = np.array(
    [
        pnoise1(
            i * 0.000025,  # scale: controls frequency (smaller = slower variations)
            octaves=10,  # number of noise layers for detail
            persistence=0.3,  # amplitude decay per octave
            base=perlin_seed,
        )
        for i in range(time_points)
    ]
)

plt.plot(perlin_noise)
plt.show()

perlin_noise /= np.max(np.abs(perlin_noise))  # Normalize to ±1

# shift and scale the noise to 0 - 30 Hz
perlin_noise = (perlin_noise + 1) / 2 * 20

# Scale Perlin noise to desired amplitude (±30 Hz) and subtract only positive values
trapezoid_drive = trapezoid_drive - perlin_noise

# Create AnalogSignal
sinusoidal_drive = AnalogSignal(
    signal=trapezoid_drive,
    units=pq.Hz,
    sampling_period=(timestep * pq.ms).rescale(pq.s),
)

joblib.dump(sinusoidal_drive, save_path / "trapezoid_drive_pattern.pkl")
print(
    f"\n Trapezoidal drive pattern (1000ms trapezoid centered in {simulation_time}ms simulation):"
)
print(f"  Rest before: 0 - {trapezoid_start} ms ({dd_baseline__Hz} Hz)")
print(
    f"  Ramp up: {trapezoid_start} - {ramp_up_end} ms ({dd_baseline__Hz} → {dd_peak__Hz} Hz)"
)
print(
    f"  Plateau: {ramp_up_end} - {plateau_end} ms ({dd_peak__Hz} Hz, center at {(ramp_up_end + plateau_end) / 2:.0f}ms)"
)
print(
    f"  Ramp down: {plateau_end} - {ramp_down_end} ms ({dd_peak__Hz} → {dd_baseline__Hz} Hz)"
)
print(f"  Rest after: {ramp_down_end} - {simulation_time} ms ({dd_baseline__Hz} Hz)")

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
network.connect(source="DD", target="aMN", probability=0.5, weight__μS=0.05)

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
                spike_time = h.t + 1
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
axes[0].set_title("Trapezoidal Descending Drive Pattern (Ramp Contraction)")
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

##############################################################################
# Individual Motor Neuron Discharge Rates
# ----------------------------------------
#
# Compute smoothed instantaneous firing rates for each motor neuron
# using Elephant's kernel-based rate estimation.

print("\nComputing smoothed discharge rates per neuron...")

# Parameters
window_ms = 400  # 400 ms Hanning window
dt_s = timestep / 1000.0  # simulation timestep in seconds
window_samples = int(window_ms / 1000.0 / dt_s)

# Hanning window normalized to preserve rate
hanning_window = np.hanning(window_samples)
hanning_window = hanning_window / (hanning_window.sum() * dt_s)  # convert to Hz

mn_instantaneous_rates = []
active_neuron_ids = []

mean_firing_rates = []
cv_isi = []

for i, spiketrain in enumerate(mn_segment.spiketrains):
    if len(spiketrain) > 2:
        # Convert spike times to a binary spike train
        t = np.arange(0, simulation_time / 1000.0, dt_s)
        spikes = np.zeros_like(t)
        spike_indices = np.searchsorted(t, spiketrain.rescale(pq.s).magnitude)
        spikes[spike_indices[spike_indices < len(t)]] = 1

        # Convolve with Hanning window
        rate = np.convolve(spikes, hanning_window, mode="same")
        mn_instantaneous_rates.append(rate)
        active_neuron_ids.append(i)

        # IMPORTANT: Compute ISI/CV only during plateau phase where firing is stable
        # Filter spike train to plateau phase
        plateau_spiketrain = spiketrain.time_slice(
            ramp_up_end * pq.ms, plateau_end * pq.ms
        )

        # Compute mean firing rate (Hz) from plateau spikes only
        plateau_duration_s = (plateau_end - ramp_up_end) / 1000.0
        mean_rate = (
            len(plateau_spiketrain) / plateau_duration_s
            if len(plateau_spiketrain) > 0
            else 0.0
        )
        mean_firing_rates.append(mean_rate)

        # Compute CV of inter-spike intervals (plateau phase only)
        if len(plateau_spiketrain) > 1:
            spike_times = plateau_spiketrain.rescale(pq.s).magnitude
            isis = np.diff(spike_times)
            cv = np.std(isis) / np.mean(isis) if len(isis) > 1 else 0.0
        else:
            cv = 0.0
        cv_isi.append(cv)

        print(
            f"Neuron {i}: Mean firing rate (plateau) = {mean_rate:.2f} Hz, CV = {cv:.2f}"
        )

# Population averages
pop_mean_rate = np.mean(mean_firing_rates)
pop_mean_cv = np.mean(cv_isi)
print(
    f"\nPopulation: Mean firing rate = {pop_mean_rate:.2f} Hz, CV = {pop_mean_cv:.2f}"
)

print(f"  Computed rates for {len(active_neuron_ids)} active motor neurons")

# Create new figure for individual discharge rates
fig2, axes2 = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

# 1. Heatmap of instantaneous firing rates
if len(mn_instantaneous_rates) > 0:
    # Stack rates into 2D array (neurons x time)
    rates_array = np.array(mn_instantaneous_rates)
    time_points = np.linspace(0, simulation_time / 1000.0, rates_array.shape[1])

    # Plot heatmap (with origin='lower' so MU 0 is at TOP)
    im = axes2[0].imshow(
        rates_array,
        aspect="auto",
        cmap="hot",
        interpolation="bilinear",
        extent=[0, simulation_time / 1000.0, 0, len(active_neuron_ids)],
        origin="lower",  # Puts first row (MU 0) at bottom, but we'll flip with extent
        vmin=0,
        vmax=np.percentile(
            rates_array, 95
        ),  # Cap at 95th percentile for better contrast
    )

    axes2[0].set_ylabel("Motor Neuron ID\n(Recruitment Order, MU 0 at top)")
    axes2[0].set_title(
        "Individual Motor Neuron Discharge Rates (Smoothed with 50ms Gaussian)"
    )
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes2[0])
    cbar.set_label("Firing Rate (Hz)")
    axes2[0].grid(False)

    # 2. Individual traces (show ALL active neurons)
    n_to_plot = len(active_neuron_ids)  # Plot ALL active neurons

    # Use colormap for lines (gradient from blue to red showing recruitment order)
    colors = plt.cm.get_cmap("rainbow")(np.linspace(0, 1, n_to_plot))

    for neuron_idx in range(n_to_plot):
        axes2[1].plot(
            time_points,
            mn_instantaneous_rates[neuron_idx],
            linewidth=0.8,
            color=colors[neuron_idx],
            label=f"MN {active_neuron_ids[neuron_idx]}" if n_to_plot <= 20 else None,
        )

    axes2[1].set_xlabel("Time (s)")
    axes2[1].set_ylabel("Firing Rate (Hz)")
    axes2[1].set_title(f"All Motor Neuron Discharge Rates (n={n_to_plot})")

    # Only show legend if there are few neurons
    if n_to_plot <= 20:
        axes2[1].legend(loc="upper right", ncol=3, fontsize=6)

    axes2[1].grid(True, alpha=0.3)
    axes2[1].set_xlim(0, simulation_time / 1000.0)
    axes2[1].set_ylim(0, np.max(rates_array) * 1.1)

plt.tight_layout()
plt.show()

print(
    "\n✅ Simulation complete with individual neuron noise and discharge rate analysis!"
)
