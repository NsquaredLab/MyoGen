"""
Spike Train Generation with Descending Drive
============================================

This example demonstrates **realistic spike train simulation** using **sinusoidal descending drive (DD)**
instead of direct current injection. This approach provides more physiologically accurate motor control
patterns by modeling cortical input through descending drive populations.

!!! note
    This example bridges the gap between simple current injection (example 02) and full spinal network
    simulation (11_simulate_spinal_network.py). It uses:

    - **DescendingDrive__Pool**: Poisson process neurons modeling cortical input
    - **AlphaMN__Pool**: Biophysically detailed motor neurons (Powers2017 model)
    - **Network**: Synaptic connections between DD and motor neuron populations
    - **Sinusoidal patterns**: Smooth, physiologically relevant input at 0.5-2 Hz

!!! important
    **Descending Drive (DD)** refers to the cortical and subcortical neural pathways that provide
    voluntary motor commands to spinal motor neurons. This is more realistic than direct current
    injection because it models the actual synaptic input patterns from upper motor neurons.
"""

# %%

##############################################################################
# Import Libraries
# ----------------
#
# !!! important
#     In **MyoGen** all **random number generation** is handled by the RNG returned from
#     `get_random_generator()`, a thin wrapper around `numpy.random`.
#
#     Always fetch the generator at the call site so the current seed is honored:
#
#     ```python
#     from myogen import simulator, get_random_generator
#     get_random_generator().normal(0, 1)
#     ```
#
#     To change the seed, use `set_random_seed`:
#
#     ```python
#     from myogen import set_random_seed
#     set_random_seed(42)
#     ```

# %%

import itertools
from pathlib import Path

import joblib
import numpy as np
import quantities as pq
from matplotlib import pyplot as plt
from neo import AnalogSignal, Block, Segment, SpikeTrain
from neuron import h
from tqdm import tqdm

from myogen import get_random_generator
from myogen.simulator.neuron import Network
from myogen.simulator.neuron.populations import AlphaMN__Pool, DescendingDrive__Pool
from myogen.utils.nmodl import load_nmodl_mechanisms
from myogen.utils.types import pps

plt.style.use("fivethirtyeight")


def mean_firing_rate(spiketrain):
    """Mean firing rate of a neo.SpikeTrain (replaces elephant.statistics.mean_firing_rate)."""
    return (len(spiketrain) / (spiketrain.t_stop - spiketrain.t_start)).rescale(pq.Hz)


def population_psth(spiketrains, bin_size):
    """Total spike counts per bin across spiketrains, plus bin left edges (s).

    Native replacement for elephant.statistics.time_histogram (output="counts").
    """
    t_start = min(st.t_start for st in spiketrains).rescale(pq.s).magnitude
    t_stop = max(st.t_stop for st in spiketrains).rescale(pq.s).magnitude
    bs = (bin_size).rescale(pq.s).magnitude
    n_bins = int((t_stop - t_start) / bs)
    edges = t_start + np.arange(n_bins + 1) * bs
    spikes = np.concatenate([st.rescale(pq.s).magnitude for st in spiketrains])
    spikes = spikes[(spikes >= edges[0]) & (spikes < edges[-1])]  # drop right-edge spikes
    counts, _ = np.histogram(spikes, bins=edges)
    return counts, edges[:-1]

##############################################################################
# Create Populations
# ------------------------
#
# Like the previous example, we create a **motor neuron pool** using the **AlphaMN__Pool** class.
#
# We also create a **DescendingDrive__Pool** to represent the cortical input.
#
# !!! note
#     These neurons are modeled as Poisson point processes to convert the smooth input signal into realistic
#     spike patterns that represent cortical input to the spinal cord.
#

load_nmodl_mechanisms()

save_path = Path("./results")
save_path.mkdir(exist_ok=True)

recruitment_thresholds = joblib.load(save_path / "thresholds.pkl")

motor_neuron_pool = AlphaMN__Pool(
    recruitment_thresholds__array=recruitment_thresholds,
    config_file="alpha_mn_default.yaml",
)

timestep = 0.1 * pq.ms
h.secondorder = 2  # Crank-Nicolson method (second-order accurate)
descending_drive_pool = DescendingDrive__Pool(n=100, poisson_batch_size=5, timestep__ms=timestep)
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

simulation_time = 15000 * pq.ms
time_points = int(simulation_time / timestep)

# Trapezoidal parameters
dd_baseline__pps = 0.0 * pps  # Baseline drive during rest
dd_peak__pps = 65 * pps  # Peak drive during plateau

# Phase durations (ms) - Total trapezoid duration: 11000ms
ramp_up_duration = 500 * pq.ms  # 500ms ramp up
plateau_duration = 10000 * pq.ms  # 10s hold
ramp_down_duration = 500 * pq.ms  # 500ms ramp down

# Add rest periods before and after
rest_before = 1000 * pq.ms  # 1s rest before trapezoid
rest_after = 1000 * pq.ms  # 1s rest after trapezoid

# Center the trapezoid at 7.5s (middle of 15s simulation)
# Calculate phase boundaries with rest period before
trapezoid_start = rest_before  # Start at 1s
ramp_up_end = trapezoid_start + ramp_up_duration  # 1.5s
plateau_end = ramp_up_end + plateau_duration  # 11.5s
ramp_down_end = plateau_end + ramp_down_duration  # 12s

# Create time array
time_array = np.linspace(0, simulation_time.magnitude, time_points) * pq.ms

# Initialize drive signal (all baseline)
trapezoid_drive = np.ones(time_points) * dd_baseline__pps

for i, t in enumerate(time_array):
    if t < trapezoid_start:
        # Phase 0: Rest before
        trapezoid_drive[i] = dd_baseline__pps
    elif t < ramp_up_end:
        # Phase 1: Ramp up
        elapsed = t - trapezoid_start
        trapezoid_drive[i] = dd_baseline__pps + (dd_peak__pps - dd_baseline__pps) * (
            elapsed / ramp_up_duration
        )
    elif t < plateau_end:
        # Phase 2: Plateau
        trapezoid_drive[i] = dd_peak__pps
    elif t < ramp_down_end:
        # Phase 3: Ramp down
        elapsed = t - plateau_end
        trapezoid_drive[i] = dd_peak__pps - (dd_peak__pps - dd_baseline__pps) * (
            elapsed / ramp_down_duration
        )
    else:
        # Phase 4: Rest after
        trapezoid_drive[i] = dd_baseline__pps

# Add small noise for realism
trapezoid_drive = (
    trapezoid_drive + np.clip(get_random_generator().normal(0, 1.0, size=time_points), 0, None) * pps
)

# Create AnalogSignal
trapezoid_drive_signal = AnalogSignal(
    signal=trapezoid_drive, sampling_period=timestep.rescale(pq.s)
)

joblib.dump(trapezoid_drive_signal, save_path / "trapezoid_drive_pattern.pkl")
print(
    f"\n Trapezoidal drive pattern (1000ms trapezoid centered in {simulation_time}ms simulation):"
)
print(f"\tRest before: 0 - {trapezoid_start} ms ({dd_baseline__pps} pps)")
print(f"\tRamp up: {trapezoid_start} - {ramp_up_end} ms ({dd_baseline__pps} → {dd_peak__pps} pps)")
print(
    f"\tPlateau: {ramp_up_end} - {plateau_end} ms ({dd_peak__pps} pps, center at {(ramp_up_end + plateau_end) / 2:.0f}ms)"
)
print(f"\tRamp down: {plateau_end} - {ramp_down_end} ms ({dd_peak__pps} → {dd_baseline__pps} pps)")
print(f"\tRest after: {ramp_down_end} - {simulation_time} ms ({dd_baseline__pps} pps)")

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
network.connect(source="DD", target="aMN", probability=0.5, weight__uS=0.15 * pq.uS)

# Set up external input to DD population
network.connect_from_external(source="cortical_input", target="DD", weight__uS=1.0 * pq.uS)

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
    zip(*pool.get_initialization_data()) for pool in [motor_neuron_pool, descending_drive_pool]
):
    section.v = voltage


h.finitialize()

# Calculate total simulation steps for progress bar
total_steps = int(simulation_time / timestep)

step_counter = 0
with tqdm(
    total=float(simulation_time),
    desc="Running simulation",
    unit="ms",
    bar_format="{l_bar}{bar}| {n:.2f}/{total:.2f} ms [{elapsed}<{remaining}, {rate_fmt}]",
) as pbar:
    while h.t < h.tstop:
        current_drive = trapezoid_drive_signal[min(step_counter, len(trapezoid_drive_signal) - 1)]

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
        pbar.update(float(timestep))


##############################################################################
# Convert Spike Data to Neo Format
# ---------------------------------
#

spike_train_block = Block(name="Trapezoidal DD Spike Trains")

dd_segment = Segment(name="Descending Drive")
dd_segment.spiketrains = [
    SpikeTrain(
        (spike_times * pq.ms).rescale(pq.s),  # type: ignore
        t_stop=simulation_time.rescale(pq.s),
        sampling_rate=(1 / (h.dt * pq.ms)).rescale(pq.Hz),
        sampling_period=h.dt * pq.ms,
        name=f"DD_{i}",
    )
    for i, spike_times in enumerate(dd_spike_times)
]

mn_segment = Segment(name="Motor Neurons")
mn_segment.spiketrains = [
    SpikeTrain(
        (recorder.as_numpy() * pq.ms).rescale(pq.s),  # type: ignore
        t_stop=simulation_time.rescale(pq.s),
        sampling_rate=(1 / (h.dt * pq.ms)).rescale(pq.Hz),
        sampling_period=h.dt * pq.ms,
        name=f"MN_{i}",
    )
    for i, recorder in enumerate(mn_spike_recorders)
]

# We only save the motor neuron spikes  segment
spike_train_block.segments.append(mn_segment)

joblib.dump(spike_train_block, save_path / "trapezoid_dd_spike_trains.pkl")

##############################################################################
# Calculate Firing Rate Statistics
# ---------------------------------
#

print("\nFiring rate analysis:")

# Calculate DD firing rates
dd_firing_rates = np.array(
    [
        mean_firing_rate(st__s.time_slice(st__s.min(), st__s.max()))
        for st__s in dd_segment.spiketrains
        if len(st__s) > 0
    ]
)

# Calculate MN firing rates
mn_firing_rates = np.array(
    [
        mean_firing_rate(st__s.time_slice(st__s.min(), st__s.max()))
        for st__s in mn_segment.spiketrains
        if len(st__s) > 0
    ]
)

print("Descending Drive neurons:")
print(f"\tActive neurons: {len(dd_firing_rates)}/{descending_drive_pool.n}")
if len(dd_firing_rates) > 0:
    print(f"\tMean firing rate: {np.mean(dd_firing_rates):.1f} ± {np.std(dd_firing_rates):.1f} pps")
    print(f"\tRate range: {np.min(dd_firing_rates):.1f} - {np.max(dd_firing_rates):.1f} pps")

print("Motor neurons:")
print(f"\tActive neurons: {len(mn_firing_rates)}/{motor_neuron_pool.n}")
if len(mn_firing_rates) > 0:
    print(f"\tMean firing rate: {np.mean(mn_firing_rates):.1f} ± {np.std(mn_firing_rates):.1f} pps")
    print(f"\tRate range: {np.min(mn_firing_rates):.1f} - {np.max(mn_firing_rates):.1f} pps")

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

# 1. Plot trapezoidal drive pattern
time_s = trapezoid_drive_signal.times.rescale(pq.s).magnitude
axes[0].plot(time_s, trapezoid_drive_signal, "b-", linewidth=2, label="DD Input")
axes[0].axhline(dd_baseline__pps, color="r", linestyle="--", alpha=0.7, label="Baseline")
axes[0].set_ylabel("Drive (Hz)")
axes[0].set_title("Trapezoidal Descending Drive Pattern (Ramp Contraction)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2. DD population raster plot
dd_colors = plt.cm.get_cmap("Blues")(np.linspace(0.3, 0.8, len(dd_segment.spiketrains)))
for i, (spiketrain, color) in enumerate(zip(dd_segment.spiketrains, dd_colors)):
    if len(spiketrain) > 0:
        axes[1].scatter(spiketrain.magnitude, [i] * len(spiketrain), c=[color], s=0.8, alpha=0.8)

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
        axes[2].scatter(spike_times, [i] * len(spike_times), c=[color], s=1.0, alpha=0.8)
        active_mn_count += 1

axes[2].set_ylabel("Motor Neuron ID\n(Recruitment Order)")
axes[2].set_title(
    f"Motor Neuron Population Activity (n={active_mn_count}/{motor_neuron_pool.n} active)"
)
axes[2].set_ylim(-1, motor_neuron_pool.n)
axes[2].grid(True, alpha=0.3)

# 4. Population firing rates over time (binned)
bin_size_ms = 100

dd_counts, bin_centers_s = population_psth(dd_segment.spiketrains, bin_size_ms * pq.ms)
dd_rates_binned = dd_counts / (bin_size_ms / 1000.0) / descending_drive_pool.n

mn_counts, _ = population_psth(mn_segment.spiketrains, bin_size_ms * pq.ms)
mn_rates_binned = mn_counts / (bin_size_ms / 1000.0) / motor_neuron_pool.n
axes[3].plot(bin_centers_s, dd_rates_binned, "b-", linewidth=2, label="DD Population", alpha=0.8)
axes[3].plot(bin_centers_s, mn_rates_binned, "r-", linewidth=2, label="MN Population", alpha=0.8)

axes[3].set_xlabel("Time (s)")
axes[3].set_ylabel("Population Rate (Hz)")
axes[3].set_title("Population Firing Rates Over Time")
axes[3].legend()
axes[3].grid(True, alpha=0.3)

# Format all axes
for ax in axes:
    ax.set_xlim(0, simulation_time.rescale(pq.s).magnitude)

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
window_ms = 400 * pq.ms  # 400 ms Hanning window
dt_s = timestep.rescale(pq.s)  # simulation timestep in seconds
window_samples = int(window_ms.rescale(pq.s) / dt_s)

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
        t = np.arange(0, simulation_time.rescale(pq.s).magnitude, dt_s.magnitude)
        spikes = np.zeros_like(t)
        spike_indices = np.searchsorted(t, spiketrain.magnitude)
        spikes[spike_indices[spike_indices < len(t)]] = 1

        # Convolve with Hanning window
        rate = np.convolve(spikes, hanning_window, mode="same")
        mn_instantaneous_rates.append(rate)
        active_neuron_ids.append(i)

        # IMPORTANT: Compute ISI/CV only during plateau phase where firing is stable
        # Filter spike train to plateau phase
        plateau_spiketrain = spiketrain.time_slice(ramp_up_end, plateau_end)

        # Compute mean firing rate (Hz) from plateau spikes only
        plateau_duration_s = (plateau_end - ramp_up_end) / 1000.0
        mean_rate = (
            len(plateau_spiketrain) / plateau_duration_s if len(plateau_spiketrain) > 0 else 0.0
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

# Population averages
pop_mean_rate = np.mean(mean_firing_rates)
pop_mean_cv = np.mean(cv_isi)
print(f"\nPopulation: Mean firing rate = {pop_mean_rate:.2f} Hz, CV = {pop_mean_cv:.2f}")

print(f"  Computed rates for {len(active_neuron_ids)} active motor neurons")

# Create new figure for individual discharge rates
fig2, axes2 = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

# 1. Heatmap of instantaneous firing rates
if len(mn_instantaneous_rates) > 0:
    # Stack rates into 2D array (neurons x time)
    rates_array = np.array(mn_instantaneous_rates)
    time_points = np.linspace(0, simulation_time.rescale(pq.s).magnitude, rates_array.shape[1])

    # Plot heatmap (with origin='lower' so MU 0 is at TOP)
    im = axes2[0].imshow(
        rates_array,
        aspect="auto",
        cmap="hot",
        interpolation="bilinear",
        extent=[0, simulation_time.rescale(pq.s).magnitude, 0, len(active_neuron_ids)],
        origin="lower",  # Puts first row (MU 0) at bottom, but we'll flip with extent
        vmin=0,
        vmax=np.percentile(rates_array, 95),
    )

    axes2[0].set_ylabel("Motor Neuron ID\n(Recruitment Order, MU 0 at top)")
    axes2[0].set_title(
        "Individual Motor Neuron Discharge Rates (Smoothed with 400ms Hanning Window)"
    )
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes2[0])
    cbar.set_label("Firing Rate (Hz)")
    axes2[0].grid(False)

    # 2. Individual traces
    n_to_plot = len(active_neuron_ids)

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
    axes2[1].set_xlim(0, simulation_time.rescale(pq.s).magnitude)
    axes2[1].set_ylim(0, np.max(rates_array) * 1.1)

plt.tight_layout()
plt.show()

print("\n[DONE] Simulation complete with individual neuron noise and discharge rate analysis!")

# mkdocs_gallery_thumbnail_path = "gallery_thumbs/03_simulate_spike_trains_descending_drive.png"
