"""
Direct Current Injection - ISI/CV Extraction
==============================================

This script drives motor neurons with **direct current injection** rather than
through a descending drive network. This provides precise control over:
- Current amplitude (controls firing rate)
- Current noise (controls CoV independently)
- Recruitment-based current distribution

This approach eliminates gamma process variability from DD neurons, giving you
the cleanest control over motor neuron firing patterns.
"""

import os

os.environ["MPLBACKEND"] = "Agg"
if "DISPLAY" in os.environ:
    del os.environ["DISPLAY"]

import argparse
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
from myogen.simulator.neuron.populations import AlphaMN__Pool
from myogen.utils.nmodl import load_nmodl_mechanisms

##############################################################################
# Command-Line Arguments
##############################################################################

parser = argparse.ArgumentParser(
    description="Extract ISI/CV data using direct current injection"
)
parser.add_argument("--muscle", type=str, default="VLVM", help="Muscle type")
parser.add_argument("--mvc-level", type=int, default=30, help="MVC level %%")
parser.add_argument("--study-prefix", type=str, default="VLVM_", help="Study prefix")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument(
    "--target-fr-mean", type=float, default=16.8, help="Target mean firing rate (Hz)"
)
parser.add_argument(
    "--target-fr-std", type=float, default=2.5, help="Target FR std (Hz)"
)
parser.add_argument(
    "--current-noise-std",
    type=float,
    default=0.1,
    help="Current noise std (nA) - controls CoV",
)
parser.add_argument(
    "--base-current",
    type=float,
    default=5.0,
    help="Base current amplitude (nA) - controls firing rate",
)
parser.add_argument(
    "--current-range",
    type=float,
    default=10.0,
    help="Current range across MN pool (nA) - recruitment gradient",
)

args = parser.parse_args()

set_random_seed(args.seed)

STUDY_PREFIX = args.study_prefix
MUSCLE_TYPE = args.muscle
MVC_LEVEL = args.mvc_level
TARGET_FR_MEAN = args.target_fr_mean
TARGET_FR_STD = args.target_fr_std
CURRENT_NOISE_STD = args.current_noise_std
BASE_CURRENT = args.base_current
CURRENT_RANGE = args.current_range

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
N_MOTOR_UNITS = 100

print(f"\n{'=' * 80}")
print("DIRECT CURRENT INJECTION - ISI/CV EXTRACTION")
print(f"{'=' * 80}\n")
print(f"Muscle: {MUSCLE_TYPE} @ {MVC_LEVEL}% MVC")
print(f"Target FR: {TARGET_FR_MEAN:.1f} ± {TARGET_FR_STD:.1f} Hz")
print(f"\nCurrent injection parameters:")
print(f"  Base current:     {BASE_CURRENT:.2f} nA")
print(f"  Current range:    {CURRENT_RANGE:.2f} nA (across pool)")
print(f"  Noise std:        {CURRENT_NOISE_STD:.3f} nA (controls CoV)")
print(f"  Seed:             {args.seed}")

##############################################################################
# Create Motor Neuron Pool
##############################################################################

load_nmodl_mechanisms()

save_path = Path("./results")
save_path.mkdir(exist_ok=True)

# Create recruitment thresholds (will use for current scaling)
recruitment_thresholds, _ = RecruitmentThresholds(
    N=N_MOTOR_UNITS,
    recruitment_range__ratio=100,
    deluca__slope=5,
    konstantin__max_threshold__ratio=1,
    mode="combined",
)

motor_neuron_pool = AlphaMN__Pool(
    recruitment_thresholds__array=recruitment_thresholds,
    config_file=CONFIG_FILE,
)

print(f"\n✓ Created motor neuron pool (N={N_MOTOR_UNITS})")
print("  NO Gfluctdv (clean biophysics)")
print("  NO DD network (direct current control)")

h.secondorder = 2  # Crank-Nicolson method

##############################################################################
# Generate Trapezoidal Current Pattern
##############################################################################

# Calculate simulation time
simulation_time = (
    REST_BEFORE__ms
    + RAMP_UP_DURATION__ms
    + PLATEAU_DURATION__ms
    + RAMP_DOWN_DURATION__ms
    + REST_AFTER__ms
)
time_points = int(simulation_time / TIMESTEP__ms)

# Phase boundaries
trapezoid_start = REST_BEFORE__ms
ramp_up_end = trapezoid_start + RAMP_UP_DURATION__ms
plateau_end = ramp_up_end + PLATEAU_DURATION__ms
ramp_down_end = plateau_end + RAMP_DOWN_DURATION__ms

# Create time array
time_array = np.linspace(0, simulation_time, time_points)

# Normalize recruitment thresholds (0 to 1, where 0 = lowest threshold)
norm_thresholds = (recruitment_thresholds - recruitment_thresholds.min()) / (
    recruitment_thresholds.max() - recruitment_thresholds.min()
)

# Create current waveforms for each motor neuron
# Early-recruited (low threshold) get MORE current than late-recruited
current_waveforms = np.zeros((N_MOTOR_UNITS, time_points))

for mu_idx in range(N_MOTOR_UNITS):
    # Invert threshold: low threshold MU → high current multiplier
    current_multiplier = 1.0 - norm_thresholds[mu_idx]

    # Current amplitude for this MU (early MUs get more current)
    mu_current_amplitude = BASE_CURRENT + CURRENT_RANGE * current_multiplier

    # Build trapezoidal waveform
    trapezoid_current = np.zeros(time_points)

    for i, t in enumerate(time_array):
        if t < trapezoid_start:
            # Phase 0: Rest before
            trapezoid_current[i] = 0.0
        elif t < ramp_up_end:
            # Phase 1: Ramp up
            elapsed = t - trapezoid_start
            trapezoid_current[i] = mu_current_amplitude * (elapsed / RAMP_UP_DURATION__ms)
        elif t < plateau_end:
            # Phase 2: Plateau
            trapezoid_current[i] = mu_current_amplitude
        elif t < ramp_down_end:
            # Phase 3: Ramp down
            elapsed = t - plateau_end
            trapezoid_current[i] = mu_current_amplitude * (
                1.0 - elapsed / RAMP_DOWN_DURATION__ms
            )
        else:
            # Phase 4: Rest after
            trapezoid_current[i] = 0.0

    # Add noise (this controls CoV!)
    if CURRENT_NOISE_STD > 0:
        noise = RANDOM_GENERATOR.normal(0, CURRENT_NOISE_STD, size=time_points)
        trapezoid_current = np.clip(trapezoid_current + noise, 0, None)

    current_waveforms[mu_idx, :] = trapezoid_current

print(f"\n📊 Trapezoidal current injection pattern:")
print(f"  Rest before:  0 - {trapezoid_start:.0f} ms (0 nA)")
print(
    f"  Ramp up:      {trapezoid_start:.0f} - {ramp_up_end:.0f} ms (0 → {BASE_CURRENT:.1f}-{BASE_CURRENT + CURRENT_RANGE:.1f} nA)"
)
print(
    f"  Plateau:      {ramp_up_end:.0f} - {plateau_end:.0f} ms ({BASE_CURRENT:.1f}-{BASE_CURRENT + CURRENT_RANGE:.1f} nA)"
)
print(
    f"  Ramp down:    {plateau_end:.0f} - {ramp_down_end:.0f} ms ({BASE_CURRENT:.1f}-{BASE_CURRENT + CURRENT_RANGE:.1f} nA → 0)"
)
print(f"  Rest after:   {ramp_down_end:.0f} - {simulation_time:.0f} ms (0 nA)")
print(f"\nCurrent distribution:")
print(f"  Early MUs (low threshold): {BASE_CURRENT + CURRENT_RANGE:.2f} nA")
print(f"  Late MUs (high threshold): {BASE_CURRENT:.2f} nA")
print(f"  Noise std: {CURRENT_NOISE_STD:.3f} nA")

##############################################################################
# Setup Current Injection and Spike Recording
##############################################################################

# Create IClamp objects for each motor neuron (inject at soma)
iclamps = []
for cell in motor_neuron_pool:
    iclamp = h.IClamp(cell.soma(0.5))
    iclamp.delay = 0  # Start immediately
    iclamp.dur = 1e9  # Very long duration (we'll control via .amp)
    iclamp.amp = 0  # Initial amplitude (will update each timestep)
    iclamps.append(iclamp)

# Setup spike recording
mn_spike_recorders = []
for cell in motor_neuron_pool:
    spike_recorder = h.Vector()
    nc = h.NetCon(cell.soma(0.5)._ref_v, None, sec=cell.soma)
    nc.threshold = 50
    nc.record(spike_recorder)
    mn_spike_recorders.append(spike_recorder)

##############################################################################
# Run Simulation
##############################################################################

h.load_file("stdrun.hoc")
h.dt = TIMESTEP__ms
h.tstop = simulation_time

# Initialize voltages
for section, voltage in zip(*motor_neuron_pool.get_initialization_data()):
    section.v = voltage

h.finitialize()

step_counter = 0
with tqdm(
    total=simulation_time,
    desc="Running simulation",
    unit="ms",
    bar_format="{l_bar}{bar}| {n:.2f}/{total:.2f} ms [{elapsed}<{remaining}, {rate_fmt}]",
) as pbar:
    while h.t < h.tstop:
        # Update current injection for each motor neuron
        for mu_idx, iclamp in enumerate(iclamps):
            iclamp.amp = current_waveforms[mu_idx, min(step_counter, time_points - 1)]

        # Advance simulation
        h.fadvance()
        step_counter += 1
        pbar.update(TIMESTEP__ms)

##############################################################################
# Convert Spike Data to Neo Format
##############################################################################

spike_train_block = Block(name="Direct Current Injection Spike Trains")

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

spike_train_block.segments.append(mn_segment)

joblib.dump(
    spike_train_block,
    save_path / f"{STUDY_PREFIX}direct_current_spike_trains_{MUSCLE_TYPE}_{MVC_LEVEL}.pkl",
)

##############################################################################
# Calculate Firing Rate Statistics
##############################################################################

print("\n" + "=" * 80)
print("FIRING RATE ANALYSIS")
print("=" * 80)

mn_firing_rates = np.array(
    [
        elephant.statistics.mean_firing_rate(st__s.time_slice(st__s.min(), st__s.max()))
        for st__ms in mn_segment.spiketrains
        if len(st__s := st__ms.rescale(pq.s)) > 0
    ]
)

print(f"\nMotor neurons:")
print(f"  Active neurons: {len(mn_firing_rates)}/{N_MOTOR_UNITS}")
if len(mn_firing_rates) > 0:
    print(
        f"  Mean firing rate: {np.mean(mn_firing_rates):.1f} ± {np.std(mn_firing_rates):.1f} Hz"
    )
    print(f"  Rate range: {np.min(mn_firing_rates):.1f} - {np.max(mn_firing_rates):.1f} Hz")
    print(f"\nTarget comparison:")
    print(f"  Target FR:  {TARGET_FR_MEAN:.1f} ± {TARGET_FR_STD:.1f} Hz")
    print(f"  Achieved:   {np.mean(mn_firing_rates):.1f} ± {np.std(mn_firing_rates):.1f} Hz")
    print(f"  Error:      {abs(np.mean(mn_firing_rates) - TARGET_FR_MEAN):.1f} Hz")

##############################################################################
# Calculate and Save ISI/CV Statistics
##############################################################################

print("\n" + "=" * 80)
print("ISI/CV STATISTICS")
print("=" * 80)

print(f"\nAnalyzing plateau phase: {ramp_up_end:.1f} - {plateau_end:.1f} ms")
isi_cv_df = calculate_firing_rate_statistics(
    mn_segment.spiketrains,
    plateau_start_ms=ramp_up_end,
    plateau_end_ms=plateau_end,
    return_per_neuron=True,
    min_spikes_for_cv=3,
)

# Save to CSV
output_file = (
    save_path / f"{STUDY_PREFIX}isi_cv_direct_{MUSCLE_TYPE}_{MVC_LEVEL}.csv"
)
isi_cv_df.to_csv(output_file, index=False)

print(f"\n✅ Saved ISI/CV data to: {output_file}")
print(f"   Total motor units analyzed: {len(isi_cv_df)}")
if len(isi_cv_df) > 0:
    mean_cv = isi_cv_df["CV_ISI"].mean()
    print(f"   Mean firing rate: {isi_cv_df['mean_firing_rate_Hz'].mean():.2f} Hz")
    print(f"   Mean CV:          {mean_cv:.3f}")
    print(f"   CV range:         {isi_cv_df['CV_ISI'].min():.3f} - {isi_cv_df['CV_ISI'].max():.3f}")

##############################################################################
# Visualization
##############################################################################

print("\n📊 Creating verification plots...")

# Create figure with 3 subplots
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# 1. Current injection patterns (show subset of MUs)
time_s = time_array / 1000.0
n_to_show = min(10, N_MOTOR_UNITS)
colors_current = plt.cm.get_cmap("viridis")(
    np.linspace(0, 1, n_to_show)
)
for i in range(n_to_show):
    mu_idx = int(i * N_MOTOR_UNITS / n_to_show)
    axes[0].plot(
        time_s,
        current_waveforms[mu_idx, :],
        color=colors_current[i],
        linewidth=1.5,
        label=f"MN {mu_idx}" if n_to_show <= 5 else None,
        alpha=0.7,
    )
axes[0].set_ylabel("Current (nA)")
axes[0].set_title(
    f"Current Injection Waveforms (showing {n_to_show} of {N_MOTOR_UNITS} MUs)\n"
    f"Noise std: {CURRENT_NOISE_STD:.3f} nA"
)
if n_to_show <= 5:
    axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2. Motor neuron raster plot
mn_colors = plt.cm.get_cmap("Reds")(np.linspace(0.3, 0.9, len(mn_segment.spiketrains)))
active_mn_count = 0
for i, (spiketrain, color) in enumerate(zip(mn_segment.spiketrains, mn_colors)):
    if len(spiketrain) > 0:
        spike_times = spiketrain.rescale(pq.s).magnitude
        axes[1].scatter(spike_times, [i] * len(spike_times), c=[color], s=1.5, alpha=0.8)
        active_mn_count += 1

axes[1].set_ylabel("Motor Neuron ID\n(Recruitment Order)")
axes[1].set_title(f"Motor Neuron Activity (n={active_mn_count}/{N_MOTOR_UNITS} active)")
axes[1].set_ylim(-1, N_MOTOR_UNITS)
axes[1].grid(True, alpha=0.3)

# 3. Firing rate and CV distribution
if len(isi_cv_df) > 0:
    # Create twin axis
    ax3_fr = axes[2]
    ax3_cv = ax3_fr.twinx()

    mu_indices = isi_cv_df.index.values
    firing_rates = isi_cv_df["mean_firing_rate_Hz"].values
    cv_values = isi_cv_df["CV_ISI"].values

    ax3_fr.scatter(mu_indices, firing_rates, c="blue", s=30, alpha=0.6, label="Firing Rate")
    ax3_cv.scatter(mu_indices, cv_values, c="red", s=30, alpha=0.6, label="CV ISI")

    ax3_fr.axhline(TARGET_FR_MEAN, color="blue", linestyle="--", alpha=0.5, label="Target FR")

    ax3_fr.set_xlabel("Motor Neuron ID")
    ax3_fr.set_ylabel("Firing Rate (Hz)", color="blue")
    ax3_cv.set_ylabel("CV ISI", color="red")
    ax3_fr.tick_params(axis="y", labelcolor="blue")
    ax3_cv.tick_params(axis="y", labelcolor="red")
    ax3_fr.set_title(
        f"Per-Neuron Statistics (Plateau Phase)\n"
        f"Mean FR: {firing_rates.mean():.1f} Hz | Mean CV: {cv_values.mean():.3f}"
    )
    ax3_fr.grid(True, alpha=0.3)
    ax3_fr.legend(loc="upper left")
    ax3_cv.legend(loc="upper right")

plt.tight_layout()
plt.savefig(
    save_path / f"{STUDY_PREFIX}direct_current_verification_{MUSCLE_TYPE}_{MVC_LEVEL}.png",
    dpi=150,
)
plt.close(fig)

print(f"✅ Saved plot to: {save_path / f'{STUDY_PREFIX}direct_current_verification_{MUSCLE_TYPE}_{MVC_LEVEL}.png'}")

##############################################################################
# Summary
##############################################################################

print("\n" + "=" * 80)
print("SIMULATION COMPLETE")
print("=" * 80)
print(f"\nFiles saved:")
print(f"  • Spike trains: {STUDY_PREFIX}direct_current_spike_trains_{MUSCLE_TYPE}_{MVC_LEVEL}.pkl")
print(f"  • ISI/CV data:  {STUDY_PREFIX}isi_cv_direct_{MUSCLE_TYPE}_{MVC_LEVEL}.csv")
print(f"  • Verification: {STUDY_PREFIX}direct_current_verification_{MUSCLE_TYPE}_{MVC_LEVEL}.png")

if len(isi_cv_df) > 0:
    print(f"\nKey Results:")
    print(f"  Mean FR:  {isi_cv_df['mean_firing_rate_Hz'].mean():.2f} Hz (target: {TARGET_FR_MEAN:.1f})")
    print(f"  Mean CV:  {isi_cv_df['CV_ISI'].mean():.3f}")
    print(f"\nTo reduce CV further:")
    print(f"  → Decrease --current-noise-std (currently {CURRENT_NOISE_STD:.3f})")
    print(f"  → Try: --current-noise-std 0.05 or 0.01")
    print(f"\nTo adjust firing rate:")
    print(f"  → Adjust --base-current (currently {BASE_CURRENT:.2f})")
    print(f"  → Adjust --current-range (currently {CURRENT_RANGE:.2f})")
