"""
Spike Train Generation with Optimized Descending Drive Parameters
===================================================================

This script extracts ISI (inter-spike interval) and CV (coefficient of variation) data from
motor neuron spike trains driven by **optimized descending drive (DD) parameters**.

The script automatically loads DD parameters from optimization results to ensure consistency
with the firing rate optimization performed in `optimize_dd_for_target_firing_rate.py`.

Prerequisites
-------------
You must run `optimize_dd_for_target_firing_rate.py` first to generate
the optimized DD parameters that this script will load and use.
"""

import os

os.environ["MPLBACKEND"] = "Agg"
if "DISPLAY" in os.environ:
    del os.environ["DISPLAY"]

import argparse
import itertools
import json
from pathlib import Path

import elephant
import joblib
import numpy as np
import quantities as pq
from matplotlib import pyplot as plt
from neo import AnalogSignal, Block, Segment, SpikeTrain
from neuron import h
from tqdm import tqdm

from examples.finetune.helper import calculate_firing_rate_statistics
from myogen import RANDOM_GENERATOR, set_random_seed
from myogen.simulator.core.physiological_distribution import RecruitmentThresholds
from myogen.simulator.neuron import Network
from myogen.simulator.neuron.populations import AlphaMN__Pool, DescendingDrive__Pool
from myogen.utils.nmodl import load_nmodl_mechanisms

##############################################################################
# Command-Line Arguments
#############################################################################

parser = argparse.ArgumentParser(description="Extract ISI/CV data from spike trains")
parser.add_argument("--muscle", type=str, default="VLVM", help="Muscle type")
parser.add_argument("--mvc-level", type=int, default=30, help="MVC level %%")
parser.add_argument("--study-prefix", type=str, default="VLVM_", help="Study prefix")
parser.add_argument(
    "--use-baseline", action="store_true", help="Use baseline optimization"
)
parser.add_argument("--seed", type=int, default=42, help="Random seed")

args = parser.parse_args()

set_random_seed(args.seed)

USE_BASELINE_OPTIMIZATION = args.use_baseline
TARGET_FORCE_PCT = args.mvc_level
STUDY_PREFIX = args.study_prefix
MUSCLE_TYPE = args.muscle
MVC_LEVEL = args.mvc_level

##############################################################################

print(f"\n{'=' * 80}")
print("LOADING OPTIMIZED DD PARAMETERS")
print(f"{'=' * 80}\n")

if USE_BASELINE_OPTIMIZATION:
    # Load from baseline firing rate optimization
    OPTIMIZATION_RESULTS_DIR = Path("../../../results/dd_optimization")
    PARAMS_FILE = OPTIMIZATION_RESULTS_DIR / f"{STUDY_PREFIX}dd_optimized_params.json"

    print("Loading from BASELINE firing rate optimization...")

    if not PARAMS_FILE.exists():
        raise FileNotFoundError(
            f"Baseline optimization results not found at {PARAMS_FILE}.\n"
            "Please run optimize_dd_for_target_firing_rate.py first."
        )

    with open(PARAMS_FILE, "r") as f:
        optimization_results = json.load(f)

    # Extract DD parameters from best firing rate match
    dd_params = optimization_results["best_firing_rate_match"]
    source_description = "baseline FR optimization"

else:
    # Load from force-specific optimization
    OPTIMIZATION_RESULTS_DIR = Path(
        "/home/oj98yqyk/code/simulators/MyoGen/results/force_optimization"
    )

    # List available force optimization files
    if OPTIMIZATION_RESULTS_DIR.exists():
        available_files = sorted(
            OPTIMIZATION_RESULTS_DIR.glob(
                f"{STUDY_PREFIX}dd_optimized_params_force_*pct.json"
            )
        )
        if available_files:
            print("Available force optimization results:")
            for i, file in enumerate(available_files):
                # Extract percentage from filename
                pct = file.stem.split("_")[-1].replace("pct", "")
                print(f"  {i + 1}. {pct}% MVC - {file.name}")
            print()

    PARAMS_FILE = (
        OPTIMIZATION_RESULTS_DIR
        / f"{STUDY_PREFIX}dd_optimized_params_force_{int(TARGET_FORCE_PCT)}pct.json"
    )

    print(f"Loading from FORCE-SPECIFIC optimization ({TARGET_FORCE_PCT}% MVC)...")

    if not PARAMS_FILE.exists():
        raise FileNotFoundError(
            f"Force optimization results not found at {PARAMS_FILE}.\n"
            f"Please run: python optimize_dd_for_target_force.py --target-force-pct {TARGET_FORCE_PCT}"
        )

    with open(PARAMS_FILE, "r") as f:
        optimization_results = json.load(f)

    # Extract DD parameters from force optimization results
    dd_params = optimization_results["dd_parameters"]
    source_description = f"{TARGET_FORCE_PCT}% MVC force optimization"

# Extract common DD parameters
N_DD_NEURONS = dd_params["dd_neurons"]
DD_CONNECTION_PROBABILITY = dd_params["conn_probability"]
DD_PEAK__Hz = dd_params["dd_drive__Hz"]
DD_SHAPE_PARAMETER = dd_params["gamma_shape"]
DD_SYNAPTIC_WEIGHT = dd_params.get("synaptic_weight", 0.05)

# Load Gfluctdv settings if present
GFLUCTDV_ENABLED = optimization_results.get("gfluctdv_enabled", False)
GFLUCTDV_NOISE_AMPLITUDE = dd_params.get("gfluctdv_noise_amplitude", None)

print(f"\n✓ Loaded from: {source_description}")
print(f"  Source file: {PARAMS_FILE.name}")
print("\nOptimized DD parameters:")
print(f"  DD neurons:       {N_DD_NEURONS}")
print(f"  Conn probability: {DD_CONNECTION_PROBABILITY:.3f}")
print(f"  Synaptic weight:  {DD_SYNAPTIC_WEIGHT:.4f} μS")
print(f"  DD drive level:   {DD_PEAK__Hz:.2f} Hz")
print(
    f"  Gamma shape:      {DD_SHAPE_PARAMETER:.2f} (CV={1 / DD_SHAPE_PARAMETER**0.5:.3f})"
)
if GFLUCTDV_ENABLED:
    print(f"  Gfluctdv:         ENABLED (noise={GFLUCTDV_NOISE_AMPLITUDE:.2e} S/cm²)")

##############################################################################
# Configuration
##############################################################################

CONFIG_FILE = "alpha_mn_default.yaml"
RAMP_UP_DURATION__ms = 1e3
PLATEAU_DURATION__ms = 2e3
RAMP_DOWN_DURATION__ms = 1e3
REST_BEFORE__ms = 1e3
REST_AFTER__ms = 1e3
TIMESTEP__ms = 0.1

##############################################################################
# Create Populations
##############################################################################

load_nmodl_mechanisms()

save_path = Path("./results")
save_path.mkdir(exist_ok=True)

recruitment_thresholds, _ = RecruitmentThresholds(
    N=100,
    recruitment_range__ratio=100,
    deluca__slope=5,
    konstantin__max_threshold__ratio=1,
    mode="combined",
)

motor_neuron_pool = AlphaMN__Pool(
    recruitment_thresholds__array=recruitment_thresholds,
    config_file=CONFIG_FILE,
)

##############################################################################
# Add Individual Noise to Each Motor Neuron
# ------------------------------------------
#
# Apply Gfluctdv mechanism if it was enabled during DD optimization.
# This ensures consistency with the optimization results.
# Uses the optimized noise amplitude from the DD optimization.

if GFLUCTDV_ENABLED and GFLUCTDV_NOISE_AMPLITUDE is not None:
    print("\nApplying Gfluctdv to motor neurons (matching DD optimization)...")
    print(f"  Noise amplitude: {GFLUCTDV_NOISE_AMPLITUDE:.2e} S/cm²")
    for cell in motor_neuron_pool:
        cell.insert_Gfluctdv()
        for d in cell.dend:
            d.std_e_Gfluctdv = GFLUCTDV_NOISE_AMPLITUDE
            d.std_i_Gfluctdv = GFLUCTDV_NOISE_AMPLITUDE
else:
    print("\nGfluctdv NOT enabled (following DD optimization settings)")

timestep = TIMESTEP__ms  # ms
h.secondorder = 2  # Crank-Nicolson method (second-order accurate)

# Create descending drive pool using Gamma process for physiologically realistic variability
descending_drive_pool = DescendingDrive__Pool(
    n=N_DD_NEURONS,
    timestep__ms=timestep,
    process_type="gamma",
    shape=DD_SHAPE_PARAMETER,
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

# Calculate simulation time from configured durations
simulation_time = (
    REST_BEFORE__ms
    + RAMP_UP_DURATION__ms
    + PLATEAU_DURATION__ms
    + RAMP_DOWN_DURATION__ms
    + REST_AFTER__ms
)  # ms
time_points = int(simulation_time / timestep)

# Trapezoidal parameters (from configuration)
dd_baseline__Hz = 0.0  # Baseline drive during rest
dd_peak__Hz = DD_PEAK__Hz  # Peak drive during plateau

# Phase durations from configuration
ramp_up_duration = RAMP_UP_DURATION__ms
plateau_duration = PLATEAU_DURATION__ms
ramp_down_duration = RAMP_DOWN_DURATION__ms
rest_before = REST_BEFORE__ms
rest_after = REST_AFTER__ms

# Calculate phase boundaries
trapezoid_start = rest_before
ramp_up_end = trapezoid_start + ramp_up_duration
plateau_end = ramp_up_end + plateau_duration
ramp_down_end = plateau_end + ramp_down_duration

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

# Create AnalogSignal
sinusoidal_drive = AnalogSignal(
    signal=trapezoid_drive,
    units=pq.Hz,
    sampling_period=(timestep * pq.ms).rescale(pq.s),
)

joblib.dump(sinusoidal_drive, save_path / f"{STUDY_PREFIX}trapezoid_drive_pattern.pkl")
print(
    f"\n📊 Trapezoidal drive pattern (using optimized DD drive of {dd_peak__Hz:.2f} Hz):"
)
print(f"  Rest before: 0 - {trapezoid_start} ms ({dd_baseline__Hz} Hz)")
print(
    f"  Ramp up: {trapezoid_start} - {ramp_up_end} ms ({dd_baseline__Hz} → {dd_peak__Hz:.2f} Hz)"
)
print(
    f"  Plateau: {ramp_up_end} - {plateau_end} ms ({dd_peak__Hz:.2f} Hz, center at {(ramp_up_end + plateau_end) / 2:.0f}ms)"
)
print(
    f"  Ramp down: {plateau_end} - {ramp_down_end} ms ({dd_peak__Hz:.2f} → {dd_baseline__Hz} Hz)"
)
print(f"  Rest after: {ramp_down_end} - {simulation_time} ms ({dd_baseline__Hz} Hz)")
print("\nUsing optimized DD parameters:")
print(f"  Gamma shape: {DD_SHAPE_PARAMETER:.2f} (CV={1 / DD_SHAPE_PARAMETER**0.5:.3f})")
print(f"  Connection probability: {DD_CONNECTION_PROBABILITY:.3f}")

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

# Connect DD neurons to motor neurons with optimized synaptic parameters
network.connect(
    source="DD",
    target="aMN",
    probability=DD_CONNECTION_PROBABILITY,
    weight__μS=DD_SYNAPTIC_WEIGHT,
)

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
                dd_spike_times[dd_cell.pool__ID].append(h.t + 1)
                # Generate spike in DD neuron
                spike_time = h.t
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

joblib.dump(
    spike_train_block, save_path / f"{STUDY_PREFIX}sinusoidal_dd_spike_trains.pkl"
)

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
# Calculate and Save ISI/CV Statistics
# -------------------------------------
#
# Calculate ISI and CV for each motor unit and save to CSV file


# Calculate ISI/CV statistics for motor neurons
print("\n📊 Calculating ISI and CV statistics...")
print(f"  Analyzing only plateau phase: {ramp_up_end:.1f} - {plateau_end:.1f} ms")
isi_cv_df = calculate_firing_rate_statistics(
    mn_segment.spiketrains,
    plateau_start_ms=ramp_up_end,
    plateau_end_ms=plateau_end,
    return_per_neuron=True,
    min_spikes_for_cv=3,
)

# Save to CSV with MVC level suffix
output_file = save_path / f"{STUDY_PREFIX}isi_cv_data_{MUSCLE_TYPE}_{MVC_LEVEL}.csv"
isi_cv_df.to_csv(output_file, index=False)

print(f"✅ Saved ISI/CV data to: {output_file}")
print("   (Generated using optimized DD parameters)")
print(f"   Total motor units analyzed: {len(isi_cv_df)}")
if len(isi_cv_df) > 0:
    print(f"   Mean firing rate: {isi_cv_df['mean_firing_rate_Hz'].mean():.2f} Hz")
    print(f"   Mean CV: {isi_cv_df['CV_ISI'].mean():.3f}")

##############################################################################
# Minimal Visualization
# ----------------------
#
# Create simple 2-panel plot for verification:
# 1. Drive input pattern
# 2. Motor neuron raster plot

print("\n📊 Creating verification plots...")

# Create figure with 2 subplots
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# 1. Plot drive pattern
time_s = sinusoidal_drive.times.rescale(pq.s).magnitude
axes[0].plot(time_s, sinusoidal_drive, "b-", linewidth=2, label="DD Input")
axes[0].axhline(dd_baseline__Hz, color="r", linestyle="--", alpha=0.7, label="Baseline")
axes[0].set_ylabel("Drive (Hz)")
axes[0].set_title(
    f"Trapezoidal Drive Pattern (Optimized DD: {DD_PEAK__Hz:.1f} Hz, "
    f"Gamma shape: {DD_SHAPE_PARAMETER:.2f})"
)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2. Motor neuron raster plot (recruitment ordered)
mn_colors = plt.cm.get_cmap("Reds")(np.linspace(0.3, 0.9, len(mn_segment.spiketrains)))
active_mn_count = 0
for i, (spiketrain, color) in enumerate(zip(mn_segment.spiketrains, mn_colors)):
    if len(spiketrain) > 0:
        spike_times = spiketrain.rescale(pq.s).magnitude
        axes[1].scatter(
            spike_times, [i] * len(spike_times), c=[color], s=1.0, alpha=0.8
        )
        active_mn_count += 1

axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Motor Neuron ID\n(Recruitment Order)")
axes[1].set_title(
    f"Motor Neuron Activity (n={active_mn_count}/{motor_neuron_pool.n} active)"
)
axes[1].set_ylim(-1, motor_neuron_pool.n)
axes[1].grid(True, alpha=0.3)

# Format all axes
for ax in axes:
    ax.set_xlim(0, simulation_time / 1000.0)

plt.tight_layout()
plt.savefig(
    save_path / f"{STUDY_PREFIX}simulation_verification_{MUSCLE_TYPE}_{MVC_LEVEL}.png",
    dpi=150,
)
plt.close(fig)

##############################################################################
# Individual Motor Neuron Discharge Rates
# ----------------------------------------
#
# Compute smoothed instantaneous firing rates for each motor neuron
# using a Hanning window (similar to 01_simulate_spike_trains_descending_drive.py)

print("\n📊 Computing smoothed discharge rates per neuron...")

# Parameters
window_ms = 400  # 400 ms Hanning window
dt_s = timestep / 1000.0  # simulation timestep in seconds
window_samples = int(window_ms / 1000.0 / dt_s)

# Hanning window normalized to preserve rate
hanning_window = np.hanning(window_samples)
hanning_window = hanning_window / (hanning_window.sum() * dt_s)  # convert to Hz

mn_instantaneous_rates = []
active_neuron_ids = []

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

print(f"  Computed rates for {len(active_neuron_ids)} active motor neurons")

# Create figure for discharge rate visualizations
fig2, axes2 = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

# 1. Heatmap of instantaneous firing rates
if len(mn_instantaneous_rates) > 0:
    # Stack rates into 2D array (neurons x time)
    rates_array = np.array(mn_instantaneous_rates)
    time_points_arr = np.linspace(0, simulation_time / 1000.0, rates_array.shape[1])

    # Plot heatmap
    im = axes2[0].imshow(
        rates_array,
        aspect="auto",
        cmap="hot",
        interpolation="bilinear",
        extent=[0, simulation_time / 1000.0, 0, len(active_neuron_ids)],
        origin="lower",
        vmin=0,
        vmax=np.percentile(rates_array, 95),  # Cap at 95th percentile
    )

    axes2[0].set_ylabel("Motor Neuron ID\n(Recruitment Order)")
    axes2[0].set_title(
        f"Individual Motor Neuron Discharge Rates - {MUSCLE_TYPE} @ {MVC_LEVEL}% MVC\n"
        f"(Smoothed with {window_ms}ms Hanning Window)"
    )
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes2[0])
    cbar.set_label("Firing Rate (Hz)")
    axes2[0].grid(False)

    # 2. Individual traces (show all active neurons)
    n_to_plot = len(active_neuron_ids)

    # Use colormap for lines (gradient showing recruitment order)
    colors = plt.cm.get_cmap("rainbow")(np.linspace(0, 1, n_to_plot))

    for neuron_idx in range(n_to_plot):
        axes2[1].plot(
            time_points_arr,
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
plt.savefig(
    save_path / f"{STUDY_PREFIX}discharge_rates_{MUSCLE_TYPE}_{MVC_LEVEL}.png",
    dpi=300,
)
plt.close(fig2)

print(
    f"✅ Saved discharge rate plot to: {save_path / f'{STUDY_PREFIX}discharge_rates_{MUSCLE_TYPE}_{MVC_LEVEL}.png'}"
)

print("\n✅ Simulation complete! ISI/CV data extracted and saved.")
