"""
Optimize Oscillating DC Offset to Match Reference Force (Watanabe Network)
===========================================================================

This example optimizes the DC offset component of an oscillating descending drive to
match the reference force produced by constant drive (from script 01).
The drive pattern is: `DC_offset + 20*sin(2π*20*t)`.

This finds the equivalent of Watanabe's 58 pps value (Phase 3) that produces the same
force as constant drive (Phase 1) when combined with 20 Hz oscillation.

.. note::
    **Purpose**: Find DC offset for Phase 3 that matches Phase 1 force

    - Reference: Constant drive force from script 01 (configurable, default 40 Hz)
    - Optimize: DC offset + 20 Hz oscillation to match that force
    - Result: DC offset < constant drive (oscillation adds activation)

.. important::
    **Prerequisites**: Run ``01_compute_baseline_force.py`` first

**Output**: ``results/watanabe_optimization/dc_offset_optimized.json``

**MyoGen Components Used**:

- :class:`~myogen.simulator.neuron.populations.AlphaMN__Pool`:
  Same motor neuron pool as script 01. Created fresh for each optimization trial
  to ensure independent network realizations.

- :class:`~myogen.simulator.neuron.populations.DescendingDrive__Pool`:
  Poisson spike generators driven by time-varying rate: ``DC_offset + amplitude*sin(2πft)``.
  The ``integrate()`` method advances the internal Poisson process by one timestep.

- :class:`~myogen.simulator.neuron.Network`:
  Rebuilds the network for each trial with same connectivity parameters (30%).

- :class:`~myogen.simulator.core.force.force_model.ForceModel`:
  Computes steady-state force from spike trains to evaluate objective function.

- **External dependency**: ``optuna`` for Bayesian optimization (TPE sampler).

**Key Concept**: The oscillation itself contributes to motor neuron activation, so
the DC offset in Phase 3 must be *lower* than the constant drive in Phase 1 to
produce the same mean force. Watanabe found 58 Hz offset vs 65 Hz constant.

**Workflow Position**: Step 2 of 6

**Next Step**: Run ``03_10pct_mvc_simulation.py`` to run the full 3-phase simulation.
"""

# %%

##############################################################################
# Import Libraries
# ----------------

import json
import os

os.environ["MPLBACKEND"] = "Agg"
if "DISPLAY" in os.environ:
    del os.environ["DISPLAY"]

import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import quantities as pq
from neo import Block, Segment, SpikeTrain
from neuron import h

from myogen import RANDOM_GENERATOR, set_random_seed
from myogen.simulator import RecruitmentThresholds
from myogen.simulator.core.force.force_model import ForceModel
from myogen.simulator.neuron import Network
from myogen.simulator.neuron.populations import AlphaMN__Pool, DescendingDrive__Pool
from myogen.utils.helper import calculate_firing_rate_statistics
from myogen.utils.nmodl import load_nmodl_mechanisms

warnings.filterwarnings("ignore")
plt.style.use("fivethirtyeight")

##############################################################################
# Configuration
# -------------

# Simulation parameters
SIMULATION_TIME_MS = 5000.0  # Shorter simulation for efficient optimization
TIMESTEP_MS = 0.1
N_MOTOR_UNITS = 800  # Watanabe specification

# Force model parameters
RECORDING_FREQUENCY__HZ = 2048
LONGEST_DURATION_RISE_TIME__MS = 90.0
CONTRACTION_TIME_RANGE = 3

# Oscillation parameters (Watanabe specification)
OSC_FREQUENCY__HZ = 20.0  # 20 Hz physiological tremor
OSC_AMPLITUDE__HZ = 20.0  # Amplitude of oscillation

# Optimization settings
N_TRIALS = 25  # Increase for production
TIMEOUT_SECONDS = 3600

# Network parameters (Watanabe specification)
N_DD_NEURONS = 400
DD_CONNECTIVITY = 0.3
SYNAPTIC_WEIGHT = 0.05

# Directories
try:
    _script_dir = Path(__file__).parent
except NameError:
    _script_dir = Path.cwd()
BASELINE_DIR = _script_dir / "results" / "watanabe_optimization"
RESULTS_DIR = _script_dir / "results" / "watanabe_optimization"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

##############################################################################
# Load Reference Force Data
# --------------------------

print("\nLoading Reference Force (Constant Drive)")
print("=" * 50)

reference_file = BASELINE_DIR / "force_reference.json"

if not reference_file.exists():
    raise FileNotFoundError(
        f"Reference force not found: {reference_file}\n"
        f"Run 01_compute_baseline_force.py first!"
    )

with open(reference_file, "r") as f:
    reference_results = json.load(f)

# Extract reference force and scaling
REFERENCE_FORCE__N = reference_results["force"]["mean__N"]
REFERENCE_DRIVE__HZ = reference_results["network_parameters"]["dd_drive__Hz"]
TARGET_FORCE__N = REFERENCE_FORCE__N  # Match the reference force
MAX_FORCE_N = reference_results["force_scaling"]["max_force__N"]

print(f"Reference force: {REFERENCE_FORCE__N:.2f} N ({REFERENCE_DRIVE__HZ:.1f} Hz constant)")
print(f"Target force: {TARGET_FORCE__N:.2f} N (match with oscillation)")
print(f"Force scaling: {MAX_FORCE_N:.0f} N maximum")
print(f"Oscillation: {OSC_FREQUENCY__HZ} Hz, amplitude {OSC_AMPLITUDE__HZ} Hz")
print(f"DD neurons: {N_DD_NEURONS}")
print(f"Connection probability: {DD_CONNECTIVITY:.1%}")
print("=" * 50 + "\n")

##############################################################################
# Initialize NEURON
# -----------------

set_random_seed(42)
load_nmodl_mechanisms()
h.secondorder = 2

##############################################################################
# Define Simulation Function
# ---------------------------


def run_simulation_with_oscillating_drive(dc_offset, recruitment_thresholds):
    """
    Run network simulation with oscillating drive and compute resulting force.

    Drive pattern: dc_offset + OSC_AMPLITUDE * sin(2*pi*OSC_FREQUENCY*t)

    Parameters
    ----------
    dc_offset : float
        DC offset component of oscillating drive (Hz)
    recruitment_thresholds : np.ndarray
        Motor unit recruitment thresholds

    Returns
    -------
    tuple
        (force_mean, n_active, fr_mean, fr_std)
    """
    # Create motor neuron pool
    motor_neuron_pool = AlphaMN__Pool(
        recruitment_thresholds__array=recruitment_thresholds,
        config_file="alpha_mn_default.yaml",
    )

    # Create descending drive pool (Poisson - Watanabe specification)
    descending_drive_pool = DescendingDrive__Pool(
        n=N_DD_NEURONS,
        timestep__ms=TIMESTEP_MS * pq.ms,
        process_type="poisson",
        poisson_batch_size=1,  # Order 1 Poisson (Watanabe)
    )

    # Build network
    network = Network({"DD": descending_drive_pool, "aMN": motor_neuron_pool})
    network.connect(
        source="DD",
        target="aMN",
        probability=DD_CONNECTIVITY,
        weight__uS=SYNAPTIC_WEIGHT * pq.uS,
    )
    network.connect_from_external(source="cortical_input", target="DD", weight__uS=1.0 * pq.uS)
    dd_netcons = network.get_netcons("cortical_input", "DD")

    # Setup spike recording
    mn_spike_recorders = []
    for cell in motor_neuron_pool:
        spike_recorder = h.Vector()
        nc = h.NetCon(cell.soma(0.5)._ref_v, None, sec=cell.soma)
        nc.threshold = 50
        nc.record(spike_recorder)
        mn_spike_recorders.append(spike_recorder)

    # Create oscillating drive signal: DC + amplitude * sin(2*pi*freq*t)
    time_points = int(SIMULATION_TIME_MS / TIMESTEP_MS)
    time_s = np.arange(time_points) * TIMESTEP_MS / 1000.0

    # Oscillating component
    drive_signal = dc_offset + OSC_AMPLITUDE__HZ * np.sin(2 * np.pi * OSC_FREQUENCY__HZ * time_s)

    # Clip to prevent negative firing rates
    drive_signal = np.clip(drive_signal, 0, None)

    # Add small noise
    drive_signal += np.clip(RANDOM_GENERATOR.normal(0, 1.0, size=time_points), 0, None)

    # Initialize simulation
    h.load_file("stdrun.hoc")
    h.dt = TIMESTEP_MS
    h.tstop = SIMULATION_TIME_MS

    for section, voltage in zip(*motor_neuron_pool.get_initialization_data()):
        section.v = voltage
    for section, voltage in zip(*descending_drive_pool.get_initialization_data()):
        section.v = voltage

    h.finitialize()

    # Run simulation
    step_counter = 0
    while h.t < h.tstop:
        current_drive = drive_signal[min(step_counter, len(drive_signal) - 1)]
        for dd_cell in descending_drive_pool:
            if dd_cell.integrate(current_drive):
                if h.t < h.tstop:
                    dd_netcons[dd_cell.pool__ID].event(h.t + 1)
        h.fadvance()
        step_counter += 1

    # Convert to Neo format
    dt_s = h.dt / 1000.0
    mn_segment = Segment(name="Motor Neurons")
    mn_segment.spiketrains = [
        SpikeTrain(
            recorder.as_numpy() / 1000 * pq.s,
            t_stop=SIMULATION_TIME_MS / 1000 * pq.s,
            sampling_rate=(1 / dt_s * pq.Hz),
            sampling_period=dt_s * pq.s,
            name=f"MN_{i}",
        )
        for i, recorder in enumerate(mn_spike_recorders)
    ]

    spike_train__Block = Block(name="Motor Unit Pool")
    spike_train__Block.segments = [mn_segment]

    # Calculate statistics
    n_active = sum(1 for st in mn_segment.spiketrains if len(st) > 1)
    stats = calculate_firing_rate_statistics(mn_segment.spiketrains)
    fr_mean = float(stats["FR_mean"])
    fr_std = float(stats["FR_std"])

    # Generate force
    force_model = ForceModel(
        recruitment_thresholds=recruitment_thresholds,
        recording_frequency__Hz=RECORDING_FREQUENCY__HZ * pq.Hz,
        longest_duration_rise_time__ms=LONGEST_DURATION_RISE_TIME__MS * pq.ms,
        contraction_time_range_factor=CONTRACTION_TIME_RANGE,
    )

    force_output = force_model.generate_force(spike_train__Block=spike_train__Block)
    force_raw = force_output.magnitude[:, 0]  # Arbitrary units (sum of MU twitches)

    # Normalize force to 0-1 range, then scale to Newtons
    # (ForceModel outputs sum of MU twitch forces, not normalized values)
    force_max_raw = np.max(force_raw)
    force_signal = (force_raw / force_max_raw) * MAX_FORCE_N  # Scale to Newtons

    # Calculate steady-state force
    steady_idx = len(force_signal) // 2
    force_mean = np.mean(force_signal[steady_idx:])

    return force_mean, n_active, fr_mean, fr_std


##############################################################################
# Define Objective Function
# --------------------------

# Generate recruitment thresholds
recruitment_thresholds, _ = RecruitmentThresholds(
    N=N_MOTOR_UNITS,
    recruitment_range__ratio=100,
    deluca__slope=5,
    konstantin__max_threshold__ratio=1.0,
    mode="combined",
)


def objective(trial):
    """
    Optimize DC offset to match reference force with oscillation.

    Parameters
    ----------
    trial : optuna.Trial
        Optimization trial

    Returns
    -------
    float
        Relative force error (minimize)
    """
    try:
        # Optimize DC offset (will be lower than constant drive)
        dc_offset = trial.suggest_float("dc_offset", 1.0, 100.0)

        # Run simulation with oscillating drive
        force_mean, n_active, fr_mean, fr_std = run_simulation_with_oscillating_drive(
            dc_offset, recruitment_thresholds
        )

        # Check minimum recruitment
        if n_active < 10:  # At least 1.25% of neurons
            return 1000.0 + (10 - n_active) * 100.0

        # Calculate relative error
        force_error = abs(force_mean - TARGET_FORCE__N) / TARGET_FORCE__N

        # Store metadata
        trial.set_user_attr("force_achieved", float(force_mean))
        trial.set_user_attr("force_error", float(force_error))
        trial.set_user_attr("n_active", n_active)
        trial.set_user_attr("dc_offset__Hz", float(dc_offset))
        trial.set_user_attr("FR_mean", float(fr_mean))
        trial.set_user_attr("FR_std", float(fr_std))

        if trial.number % 5 == 0:
            print(
                f"Trial {trial.number}: "
                f"Force={force_mean:.2f}N (target={TARGET_FORCE__N:.2f}N), "
                f"Error={force_error:.1%}, "
                f"DC={dc_offset:.1f}Hz, "
                f"Active={n_active}/{N_MOTOR_UNITS}"
            )

        return force_error

    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return 1000.0


##############################################################################
# Run Optimization
# ----------------

print(f"\nOptimizing DC Offset to Match Reference Force (Oscillating Drive)")
print("=" * 50)
print(f"Target force: {TARGET_FORCE__N:.2f} N ({REFERENCE_DRIVE__HZ:.1f} Hz reference)")
print(f"Oscillation: {OSC_FREQUENCY__HZ} Hz (amplitude {OSC_AMPLITUDE__HZ} Hz)")
print(f"Optimizing: DC offset component")
print(f"Trials: {N_TRIALS}\n")

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

storage_name = f"sqlite:///{RESULTS_DIR}/optuna_oscillating_dc.db"
study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42),
    study_name="dc_offset_oscillating_match",
    storage=storage_name,
    load_if_exists=True,
)

study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT_SECONDS, show_progress_bar=True)

##############################################################################
# Analyze Results
# ---------------

best_trial = study.best_trial

print("\nOptimization Complete")
print("=" * 50)
print(f"Best trial: {best_trial.number}")
print(f"Force error: {best_trial.value:.1%}")
print("\nForce Results:")
print(f"  Target:   {TARGET_FORCE__N:.2f} N ({REFERENCE_DRIVE__HZ:.1f} Hz reference)")
print(f"  Achieved: {best_trial.user_attrs['force_achieved']:.2f} N")
print(f"  Error:    {best_trial.user_attrs['force_error']:.1%}")
print("\nOptimized Parameters:")
print(f"  DC offset: {best_trial.user_attrs['dc_offset__Hz']:.2f} Hz")
print(f"  Oscillation: {OSC_FREQUENCY__HZ} Hz (amplitude {OSC_AMPLITUDE__HZ} Hz)")
print(f"  Active MUs: {best_trial.user_attrs['n_active']}/{N_MOTOR_UNITS}")
print(
    f"  Firing rate: {best_trial.user_attrs['FR_mean']:.1f}±{best_trial.user_attrs['FR_std']:.1f} Hz"
)

##############################################################################
# Save Results
# ------------

dd_parameters = {
    "dd_neurons": N_DD_NEURONS,
    "dd_connectivity": DD_CONNECTIVITY,
    "synaptic_weight__uS": SYNAPTIC_WEIGHT,
    "dc_offset__Hz": best_trial.user_attrs["dc_offset__Hz"],
    "process_type": "poisson",
    "poisson_batch_size": 1,
}

results = {
    "reference_drive__Hz": REFERENCE_DRIVE__HZ,
    "reference_force__N": REFERENCE_FORCE__N,
    "target_force__N": TARGET_FORCE__N,
    "achieved_force__N": best_trial.user_attrs["force_achieved"],
    "force_error": best_trial.user_attrs["force_error"],
    "force_scaling": {
        "max_force__N": MAX_FORCE_N,
    },
    "oscillation": {
        "frequency__Hz": OSC_FREQUENCY__HZ,
        "amplitude__Hz": OSC_AMPLITUDE__HZ,
    },
    "dd_parameters": dd_parameters,
}

json_path = RESULTS_DIR / "dc_offset_optimized.json"
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)

joblib.dump(study, RESULTS_DIR / "study_oscillating_dc.pkl")

print(f"\nSaved results: {json_path}")
print(f"\nNext step: Run 03_10pct_mvc_simulation.py to run the full 3-phase simulation")

##############################################################################
# Visualize Optimization History
# -------------------------------

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Extract trial data
trial_numbers = [t.number for t in study.trials]
force_achieved = [t.user_attrs.get("force_achieved", None) for t in study.trials]
force_errors = [t.value for t in study.trials]
dc_offset_vals = [t.params.get("dc_offset", None) for t in study.trials]

# 1. Optimization progress
ax_error = fig.add_subplot(gs[0, 0])
ax_error.plot(trial_numbers, force_errors, "o-", alpha=0.6, markersize=4, color=colors[0])
ax_error.axhline(
    best_trial.value, linestyle="--", color=colors[1], label=f"Best: {best_trial.value:.2%}"
)
ax_error.set_ylabel("Relative Error")
ax_error.set_title(f"Optimization Progress (DC Offset for {REFERENCE_DRIVE__HZ:.0f} Hz Match)")
ax_error.set_yscale("log")
ax_error.grid(True, alpha=0.3)
ax_error.legend(framealpha=1.0, edgecolor="none")
ax_error.set_xlabel("Trial")

# 2. Force convergence
ax_force = fig.add_subplot(gs[1, 0])
ax_force.plot(trial_numbers, force_achieved, "o-", alpha=0.6, markersize=4, color=colors[0])
ax_force.axhline(
    TARGET_FORCE__N, linestyle="--", color=colors[1], label=f"Target: {TARGET_FORCE__N:.2f} N"
)
ax_force.set_xlabel("Trial")
ax_force.set_ylabel("Force (N)")
ax_force.set_title("Force Convergence")
ax_force.legend(framealpha=1.0, edgecolor="none")

# 3. DC offset distribution
ax_hist = fig.add_subplot(gs[0, 1])
ax_hist.hist(dc_offset_vals, bins=20, alpha=0.7, color=colors[2])
ax_hist.axvline(
    best_trial.params["dc_offset"],
    linestyle="--",
    color=colors[1],
    label=f"Best: {best_trial.params['dc_offset']:.1f}Hz",
)
ax_hist.set_xlabel("DC Offset (Hz)")
ax_hist.set_ylabel("Number of Trials")
ax_hist.set_title("DC Offset Distribution")
ax_hist.legend(framealpha=1.0, edgecolor="none")

# 4. Force vs DC offset relationship
ax_scatter = fig.add_subplot(gs[1, 1])
ax_scatter.scatter(dc_offset_vals, force_achieved, c=force_errors, s=50, alpha=0.6)
ax_scatter.scatter(
    best_trial.params["dc_offset"],
    best_trial.user_attrs["force_achieved"],
    s=200,
    marker="*",
    color=colors[3],
    label="Best Trial",
)
ax_scatter.axhline(
    TARGET_FORCE__N, linestyle="--", color=colors[1], alpha=0.5, label="Target Force"
)
ax_scatter.set_xlabel("DC Offset (Hz)")
ax_scatter.set_ylabel("Force (N)")
ax_scatter.set_title("Force-DC Offset Relationship")
ax_scatter.legend(framealpha=1.0, edgecolor="none")

plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "optimization_history_dc_offset.png",
    dpi=150,
    bbox_inches="tight",
)
plt.show()

print("\n[DONE] DC offset optimization complete!")
print(f"Best DC offset: {best_trial.user_attrs['dc_offset__Hz']:.2f} Hz")
print(f"Original Watanabe: 65 Hz constant → 58 Hz + oscillation (ratio: {58/65:.3f})")
print(f"This implementation: {REFERENCE_DRIVE__HZ:.1f} Hz constant → {best_trial.user_attrs['dc_offset__Hz']:.1f} Hz + oscillation (ratio: {best_trial.user_attrs['dc_offset__Hz']/REFERENCE_DRIVE__HZ:.3f})")
