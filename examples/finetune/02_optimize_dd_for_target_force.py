"""
Optimizing Descending Drive for Target Force Levels
====================================================

This example demonstrates **force-level optimization** by tuning descending drive parameters
to achieve specific muscle force outputs. Unlike firing rate optimization (example 00), this
approach targets biomechanically relevant force production.

.. note::
    **Force-based optimization workflow**:

    1. Load baseline force from previous optimization (example 01)
    2. Specify target force as percentage of baseline (e.g., 30% MVC)
    3. Optimize DD drive frequency to match target force
    4. Keep network structure fixed (neurons, connectivity from baseline)

.. important::
    **Prerequisites**: This example requires results from previous examples:

    - Run ``00_optimize_dd_for_target_firing_rate.py`` first
    - Run ``01_compute_force_from_optimized_dd.py`` second
    - Generates baseline force measurements needed here

**Optimization Strategy**: We optimize only the DD drive frequency while keeping the network
architecture fixed. This simulates modulating cortical drive to achieve different force levels
while maintaining the same motor unit pool and connectivity.
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
from myogen.utils.helper import calculate_firing_rate_statistics, get_gamma_shape_for_mvc
from myogen.utils.nmodl import load_nmodl_mechanisms

warnings.filterwarnings("ignore")
plt.style.use("fivethirtyeight")

##############################################################################
# Configuration
# -------------
#
# Set simulation and optimization parameters.

# Simulation parameters
SIMULATION_TIME_MS = 5000.0  # Shorter simulation for efficient optimization
TIMESTEP_MS = 0.1
N_MOTOR_UNITS = 100

# Force model parameters
RECORDING_FREQUENCY__HZ = 2048
LONGEST_DURATION_RISE_TIME__MS = 90.0
CONTRACTION_TIME_RANGE = 3

# Optimization settings
N_TRIALS = 25  # Reduced for example (increase for production)
TIMEOUT_SECONDS = 3600

# Target force level
TARGET_FORCE_PCT = 30.0  # 30% of baseline force (adjustable)
MVC_PERCENT = 30.0  # MVC percentage for gamma shape calculation

# Directories - use project root to share across examples
# Handle both manual execution and Sphinx Gallery (where __file__ is not defined)
try:
    _script_dir = Path(__file__).parent
except NameError:
    _script_dir = Path.cwd()
BASELINE_DIR = _script_dir.parent.parent / "results" / "force_validation"
RESULTS_DIR = _script_dir.parent.parent / "results" / "force_optimization"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

##############################################################################
# Load Baseline Force Data
# -------------------------
#
# Load the baseline force measurements from example 01.

print("\nLoading Baseline Force Data")
print("=" * 50)

baseline_file = BASELINE_DIR / "force_results.json"

if not baseline_file.exists():
    raise FileNotFoundError(
        f"Baseline force results not found: {baseline_file}\n"
        f"Run 01_compute_force_from_optimized_dd.py first!"
    )

with open(baseline_file, "r") as f:
    baseline_results = json.load(f)

# Extract baseline parameters
BASELINE_FORCE__AU = baseline_results["force"]["mean__au"]
DD_NEURONS = baseline_results["dd_parameters"]["dd_neurons"]
CONN_PROBABILITY = baseline_results["dd_parameters"]["conn_probability"]
SYNAPTIC_WEIGHT = baseline_results["dd_parameters"].get("synaptic_weight", 0.05)
MVC_SHAPE_VALUE = baseline_results["dd_parameters"]["gamma_shape"]

# Gfluctdv settings from baseline
GFLUCTDV_ENABLED = baseline_results.get("gfluctdv_enabled", False)
GFLUCTDV_NOISE_AMPLITUDE = baseline_results["dd_parameters"].get("gfluctdv_noise_amplitude", None)

print(f"Baseline force: {BASELINE_FORCE__AU:.4f} a.u.")
print(f"DD neurons: {DD_NEURONS}")
print(f"Connection probability: {CONN_PROBABILITY:.3f}")
print(f"Synaptic weight: {SYNAPTIC_WEIGHT:.4f} µS")
if GFLUCTDV_ENABLED:
    print(f"Gfluctdv: ENABLED ({GFLUCTDV_NOISE_AMPLITUDE:.2e} S/cm²)")
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
#
# Helper function to run a single simulation and compute force.


def run_simulation_and_compute_force(dd_drive__Hz, gamma_shape, recruitment_thresholds):
    """
    Run network simulation and compute resulting force.

    Parameters
    ----------
    dd_drive__Hz : float
        DD drive frequency
    gamma_shape : float
        Gamma distribution shape parameter
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

    # Apply Gfluctdv if enabled
    if GFLUCTDV_ENABLED and GFLUCTDV_NOISE_AMPLITUDE is not None:
        for cell in motor_neuron_pool:
            cell.insert_Gfluctdv()
            for d in cell.dend:
                d.std_e_Gfluctdv = GFLUCTDV_NOISE_AMPLITUDE
                d.std_i_Gfluctdv = GFLUCTDV_NOISE_AMPLITUDE

    # Create descending drive pool
    descending_drive_pool = DescendingDrive__Pool(
        n=DD_NEURONS,
        timestep__ms=TIMESTEP_MS * pq.ms,
        process_type="gamma",
        shape=gamma_shape,
    )

    # Build network
    network = Network({"DD": descending_drive_pool, "aMN": motor_neuron_pool})
    network.connect(
        source="DD",
        target="aMN",
        probability=CONN_PROBABILITY,
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

    # Create constant drive signal
    time_points = int(SIMULATION_TIME_MS / TIMESTEP_MS)
    drive_signal = np.ones(time_points) * dd_drive__Hz + np.clip(
        RANDOM_GENERATOR.normal(0, 1.0, size=time_points), 0, None
    )

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
    force_signal = force_output.magnitude[:, 0]

    # Calculate steady-state force
    steady_idx = len(force_signal) // 2
    force_mean = np.mean(force_signal[steady_idx:])

    return force_mean, n_active, fr_mean, fr_std


##############################################################################
# Define Objective Function
# --------------------------
#
# Optuna objective function that minimizes force error.

# Calculate target force
TARGET_FORCE__AU = BASELINE_FORCE__AU * (TARGET_FORCE_PCT / 100.0)

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
    Optimize DD drive frequency to match target force.

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
        # Optimize only DD drive frequency
        dd_drive__Hz = trial.suggest_float("dd_drive", 1.0, 100.0)
        gamma_shape = get_gamma_shape_for_mvc(MVC_PERCENT, MVC_SHAPE_VALUE)

        # Run simulation
        force_mean, n_active, fr_mean, fr_std = run_simulation_and_compute_force(
            dd_drive__Hz, gamma_shape, recruitment_thresholds
        )

        # Check minimum recruitment
        if n_active < 5:
            return 1000.0 + (5 - n_active) * 100.0

        # Calculate relative error
        force_error = abs(force_mean - TARGET_FORCE__AU) / TARGET_FORCE__AU

        # Store metadata
        trial.set_user_attr("force_achieved", float(force_mean))
        trial.set_user_attr("force_error", float(force_error))
        trial.set_user_attr("n_active", n_active)
        trial.set_user_attr("dd_drive__Hz", float(dd_drive__Hz))
        trial.set_user_attr("FR_mean", float(fr_mean))
        trial.set_user_attr("FR_std", float(fr_std))

        if trial.number % 5 == 0:
            print(
                f"Trial {trial.number}: "
                f"Force={force_mean:.4f} (target={TARGET_FORCE__AU:.4f}), "
                f"Error={force_error:.1%}, "
                f"Drive={dd_drive__Hz:.1f}Hz"
            )

        return force_error

    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return 1000.0


##############################################################################
# Run Optimization
# ----------------
#
# Use Optuna TPE sampler to find optimal DD drive frequency.

print(f"\nOptimizing for {TARGET_FORCE_PCT}% of Baseline Force")
print("=" * 50)
print(f"Target force: {TARGET_FORCE__AU:.4f} a.u.")
print(f"Trials: {N_TRIALS}\n")

# Get colors from style
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

storage_name = f"sqlite:///{RESULTS_DIR}/optuna_force_{int(TARGET_FORCE_PCT)}pct.db"
study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42),
    study_name=f"force_{int(TARGET_FORCE_PCT)}pct",
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
print(f"  Target:   {TARGET_FORCE__AU:.4f} a.u. ({TARGET_FORCE_PCT}% of baseline)")
print(f"  Achieved: {best_trial.user_attrs['force_achieved']:.4f} a.u.")
print(f"  Error:    {best_trial.user_attrs['force_error']:.1%}")
print("\nOptimized Parameters:")
print(f"  DD drive: {best_trial.user_attrs['dd_drive__Hz']:.2f} Hz")
print(f"  Active MUs: {best_trial.user_attrs['n_active']}/{N_MOTOR_UNITS}")
print(
    f"  Firing rate: {best_trial.user_attrs['FR_mean']:.1f}±{best_trial.user_attrs['FR_std']:.1f} Hz"
)

##############################################################################
# Save Results
# ------------

dd_parameters = {
    "dd_neurons": DD_NEURONS,
    "conn_probability": CONN_PROBABILITY,
    "synaptic_weight": SYNAPTIC_WEIGHT,
    "dd_drive__Hz": best_trial.user_attrs["dd_drive__Hz"],
    "mvc_percent": MVC_PERCENT,
    "gamma_shape": get_gamma_shape_for_mvc(MVC_PERCENT, MVC_SHAPE_VALUE),
}

if GFLUCTDV_ENABLED:
    dd_parameters["gfluctdv_noise_amplitude"] = GFLUCTDV_NOISE_AMPLITUDE

results = {
    "target_force_pct": TARGET_FORCE_PCT,
    "baseline_force__au": BASELINE_FORCE__AU,
    "target_force__au": TARGET_FORCE__AU,
    "achieved_force__au": best_trial.user_attrs["force_achieved"],
    "force_error": best_trial.user_attrs["force_error"],
    "gfluctdv_enabled": GFLUCTDV_ENABLED,
    "dd_parameters": dd_parameters,
}

json_path = RESULTS_DIR / f"dd_optimized_params_force_{int(TARGET_FORCE_PCT)}pct.json"
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)

joblib.dump(study, RESULTS_DIR / f"study_force_{int(TARGET_FORCE_PCT)}pct.pkl")

print(f"\nSaved results: {json_path}")

##############################################################################
# Visualize Optimization History
# -------------------------------

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Extract trial data
trial_numbers = [t.number for t in study.trials]
force_achieved = [t.user_attrs.get("force_achieved", None) for t in study.trials]
force_errors = [t.value for t in study.trials]
dd_drive_vals = [t.params.get("dd_drive", None) for t in study.trials]

# Left column: shared x-axis for optimization progress and force convergence
ax_error = fig.add_subplot(gs[0, 0])
ax_force = fig.add_subplot(gs[1, 0], sharex=ax_error)

# 1. Optimization progress (force error) - top left
ax_error.plot(trial_numbers, force_errors, "o-", alpha=0.6, markersize=4, color=colors[0])
ax_error.axhline(
    best_trial.value, linestyle="--", color=colors[1], label=f"Best: {best_trial.value:.2%}"
)
ax_error.set_ylabel("Relative Error")
ax_error.set_title("Optimization Progress")
ax_error.set_yscale("log")
ax_error.grid(True, alpha=0.3)
ax_error.legend(framealpha=1.0, edgecolor="none")
ax_error.tick_params(labelbottom=False)

# 2. Force convergence - bottom left
ax_force.plot(trial_numbers, force_achieved, "o-", alpha=0.6, markersize=4, color=colors[0])
ax_force.axhline(
    TARGET_FORCE__AU, linestyle="--", color=colors[1], label=f"Target: {TARGET_FORCE__AU:.4f}"
)
ax_force.set_xlabel("Trial")
ax_force.set_ylabel("Force (a.u.)")
ax_force.set_title("Force Convergence")
ax_force.legend(framealpha=1.0, edgecolor="none")

# Right column plots
ax_hist = fig.add_subplot(gs[0, 1])
ax_scatter = fig.add_subplot(gs[1, 1])

# 3. DD drive distribution - top right
ax_hist.hist(dd_drive_vals, bins=20, alpha=0.7, color=colors[2])
ax_hist.axvline(
    best_trial.params["dd_drive"],
    linestyle="--",
    color=colors[1],
    label=f"Best: {best_trial.params['dd_drive']:.1f}Hz",
)
ax_hist.set_xlabel("DD Drive (Hz)")
ax_hist.set_ylabel("Number of Trials")
ax_hist.set_title("Drive Frequency Distribution")
ax_hist.legend(framealpha=1.0, edgecolor="none")

# 4. Force vs Drive relationship - bottom right
ax_scatter.scatter(dd_drive_vals, force_achieved, c=force_errors, s=50, alpha=0.6)
ax_scatter.scatter(
    best_trial.params["dd_drive"],
    best_trial.user_attrs["force_achieved"],
    s=200,
    marker="*",
    color=colors[3],
    label="Best Trial",
)
ax_scatter.axhline(
    TARGET_FORCE__AU, linestyle="--", color=colors[1], alpha=0.5, label="Target Force"
)
ax_scatter.set_xlabel("DD Drive (Hz)")
ax_scatter.set_ylabel("Force (a.u.)")
ax_scatter.set_title("Force-Drive Relationship")
ax_scatter.legend(framealpha=1.0, edgecolor="none")

plt.tight_layout()
plt.savefig(
    RESULTS_DIR / f"optimization_history_{int(TARGET_FORCE_PCT)}pct.png",
    dpi=150,
    bbox_inches="tight",
)
plt.show()
