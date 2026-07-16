"""
Optimizing Descending Drive Parameters for Target Firing Rates
===============================================================

This example demonstrates **parameter optimization** for descending drive networks using **Optuna**.
The goal is to find network parameters (number of DD neurons, connection probability, drive frequency)
that produce motor neuron firing patterns matching target physiological characteristics.

!!! note
    **Multi-objective optimization** is essential for tuning complex neural network models because:

    - Multiple parameters interact non-linearly (DD neurons, connectivity, synaptic weights)
    - Target outputs have multiple constraints (mean firing rate, variability, recruitment)
    - Manual parameter tuning is time-consuming and may miss optimal combinations
    - Systematic search ensures reproducible, well-documented parameter choices

!!! important
    **Descending Drive (DD) Networks** represent cortical input to spinal motor neurons. Key parameters:

    - `dd_neurons`: Population size (affects input diversity and convergence)
    - `conn_probability`: Synaptic connectivity (affects drive strength and correlation)
    - `dd_drive__Hz`: Input frequency (controls baseline excitation level)
    - `gamma_shape`: Variability of Poisson processes (affects temporal patterns)

**Optimization Objective**: Match target motor neuron firing rate statistics while maintaining
biologically plausible network parameters.
"""

# %%

##############################################################################
# Import Libraries
# ----------------
#
# For optimization, we need:
#
# - **Optuna**: Bayesian optimization framework with TPE sampler
# - **NEURON**: Biophysical simulation engine
# - **MyoGen**: Motor neuron pools and network connectivity

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
from neo import Segment, SpikeTrain
from neuron import h

from myogen import get_random_generator
from myogen.simulator import RecruitmentThresholds
from myogen.simulator.neuron import Network
from myogen.simulator.neuron.populations import AlphaMN__Pool, DescendingDrive__Pool
from myogen.utils.helper import calculate_firing_rate_statistics, get_gamma_shape_for_mvc
from myogen.utils.nmodl import load_nmodl_mechanisms

warnings.filterwarnings("ignore")
plt.style.use("fivethirtyeight")

##############################################################################
# Define Optimization Parameters
# -------------------------------
#
# Set target firing rate characteristics and optimization constraints.
# These values are based on physiological measurements from motor control studies.

# Target motor neuron firing statistics (from experimental data)
TARGET_FR_MEAN__HZ = 16.82  # Mean firing rate during isometric contraction
TARGET_FR_STD__HZ = 2.5  # Standard deviation across motor unit pool

# Network parameter search space
N_DD_NEURONS_MIN = 100  # Minimum descending drive population size
N_DD_NEURONS_MAX = 1000  # Maximum descending drive population size

# Fixed simulation parameters
N_MOTOR_UNITS = 100  # Motor neuron pool size
SIMULATION_TIME_MS = 3000.0  # Simulation duration (ms)
TIMESTEP_MS = 0.1  # Integration timestep (ms)
GAMMA_SHAPE = 3.0  # Fixed gamma shape for DD variability
SYNAPTIC_WEIGHT = 0.05  # DD→MN synaptic weight (µS)

# Optimization settings
N_TRIALS = 25  # Number of optimization trials (increase for production use)
TIMEOUT_SECONDS = 36000  # Maximum optimization time (10 hours)

# Gfluctdv (fluctuating conductance noise) parameters
ENABLE_GFLUCTDV = False  # Enable membrane noise in motor neurons
GFLUCTDV_NOISE_MIN = 5e-6  # Minimum noise amplitude (S/cm²)
GFLUCTDV_NOISE_MAX = 3e-5  # Maximum noise amplitude (S/cm²)

# Results directory - use project root to share across examples
# Handle both manual execution and Sphinx Gallery (where __file__ is not defined)
try:
    _script_dir = Path(__file__).parent
except NameError:
    _script_dir = Path.cwd()
RESULTS_DIR = _script_dir.parent.parent / "results" / "dd_optimization"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

##############################################################################
# Load NEURON Mechanisms
# ----------------------
#
# Compile and load the custom NMODL mechanisms needed for biophysical simulation.

load_nmodl_mechanisms()
h.secondorder = 2  # Use Crank-Nicolson integration (second-order accurate)

##############################################################################
# Define Objective Function
# --------------------------
#
# The objective function evaluates network performance for a given parameter set.
# Optuna will minimize this function to find optimal parameters.
#
# **Optimization variables** (suggested by Optuna each trial):
#
# 1. ``dd_neurons``: Number of descending drive neurons [100-1000]
# 2. ``conn_probability``: DD→MN connection probability [0.1-1.0]
# 3. ``dd_drive__Hz``: Input drive frequency [5-1000 Hz]
# 4. ``gfluctdv_noise_amplitude``: Membrane noise level (if enabled)
#
# **Objective components**:
#
# - ``mean_error``: Normalized deviation from target mean firing rate
# - ``std_error``: Normalized deviation from target firing rate std
# - ``plausibility_penalty``: Soft constraints on biologically unrealistic parameters


def objective(trial):
    """
    Optuna single-objective optimization function.

    Optimizes network parameters to match target firing rate statistics
    while preferring biologically plausible configurations.

    Parameters
    ----------
    trial : optuna.Trial
        Current optimization trial with parameter suggestions

    Returns
    -------
    float
        Objective value (lower is better)
    """
    try:
        # Suggest parameters for this trial
        dd_neurons = trial.suggest_int("dd_neurons", N_DD_NEURONS_MIN, N_DD_NEURONS_MAX)
        conn_probability = trial.suggest_float("conn_prob", 0.1, 1.0)
        dd_drive__Hz = trial.suggest_float("dd_drive", 5.0, 1000.0)

        # Use fixed gamma shape (not optimized)
        gamma_shape = get_gamma_shape_for_mvc(100, mvc_shape_value=GAMMA_SHAPE)

        # Optional: Gfluctdv noise amplitude
        if ENABLE_GFLUCTDV:
            gfluctdv_noise_amplitude = trial.suggest_float(
                "gfluctdv_noise_amplitude", GFLUCTDV_NOISE_MIN, GFLUCTDV_NOISE_MAX
            )
        else:
            gfluctdv_noise_amplitude = 0.0

        # Create recruitment thresholds for motor unit pool
        recruitment_thresholds, _ = RecruitmentThresholds(
            N=N_MOTOR_UNITS,
            recruitment_range__ratio=100,
            deluca__slope=5,
            konstantin__max_threshold__ratio=1.0,
            mode="combined",
        )

        # Create motor neuron pool
        motor_neuron_pool = AlphaMN__Pool(
            recruitment_thresholds__array=recruitment_thresholds,
            config_file="alpha_mn_default.yaml",
        )

        # Apply Gfluctdv noise if enabled
        if ENABLE_GFLUCTDV:
            for cell in motor_neuron_pool:
                cell.insert_Gfluctdv()
                for d in cell.dend:
                    d.std_e_Gfluctdv = gfluctdv_noise_amplitude
                    d.std_i_Gfluctdv = gfluctdv_noise_amplitude

        # Create descending drive pool
        descending_drive_pool = DescendingDrive__Pool(
            n=dd_neurons,
            timestep__ms=TIMESTEP_MS * pq.ms,
            process_type="gamma",
            shape=gamma_shape,  # type: ignore
        )

        # Build network with synaptic connections
        network = Network({"DD": descending_drive_pool, "aMN": motor_neuron_pool})
        network.connect(
            source="DD",
            target="aMN",
            probability=conn_probability,
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

        # Generate constant drive signal with small noise
        time_points = int(SIMULATION_TIME_MS / TIMESTEP_MS)
        drive_signal = np.ones(time_points) * dd_drive__Hz + np.clip(
            get_random_generator().normal(0, 1.0, size=time_points), 0, None
        )

        # Run NEURON simulation
        h.load_file("stdrun.hoc")
        h.dt = TIMESTEP_MS
        h.tstop = SIMULATION_TIME_MS

        # Initialize voltages
        for section, voltage in zip(*motor_neuron_pool.get_initialization_data()):
            section.v = voltage
        for section, voltage in zip(*descending_drive_pool.get_initialization_data()):
            section.v = voltage

        h.finitialize()

        # Simulation loop with DD drive injection
        step_counter = 0
        while h.t < h.tstop:
            current_drive = drive_signal[min(step_counter, len(drive_signal) - 1)]
            for dd_cell in descending_drive_pool:
                if dd_cell.integrate(current_drive):
                    if h.t < h.tstop:
                        dd_netcons[dd_cell.pool__ID].event(h.t + 1)
            h.fadvance()
            step_counter += 1

        # Convert spike data to Neo format
        mn_segment = Segment(name="Motor Neurons")
        dt_s = h.dt / 1000.0
        mn_segment.spiketrains = [
            SpikeTrain(
                recorder.as_numpy() / 1000 * pq.s,
                t_stop=SIMULATION_TIME_MS / 1000 * pq.s,
                sampling_rate=(1 / dt_s * (pq.Hz)),
                sampling_period=dt_s * pq.s,
                name=f"MN_{i}",
            )
            for i, recorder in enumerate(mn_spike_recorders)
        ]

        # Calculate firing rate statistics
        stats = calculate_firing_rate_statistics(mn_segment.spiketrains)
        n_active: int = int(stats["n_active"])

        # Penalty for insufficient recruitment
        if n_active < 10:
            return 1000.0

        fr_mean: float = float(stats["FR_mean"])
        fr_std: float = float(stats["FR_std"])

        # Normalized errors
        mean_error = abs(fr_mean - TARGET_FR_MEAN__HZ) / TARGET_FR_MEAN__HZ
        std_error = abs(fr_std - TARGET_FR_STD__HZ) / TARGET_FR_STD__HZ

        # Biological plausibility penalties
        plausibility_penalty = 0.0
        if conn_probability > 0.7:  # Very high connectivity is less realistic
            plausibility_penalty += 0.1 * (conn_probability - 0.7)
        if dd_neurons > 800:  # Excessive neurons are less common
            plausibility_penalty += 0.1 * ((dd_neurons - 800) / 200)

        # Combined objective (minimize)
        objective_value = mean_error + std_error + plausibility_penalty

        # Store trial metadata
        trial.set_user_attr("n_active", n_active)
        trial.set_user_attr("FR_mean", fr_mean)
        trial.set_user_attr("FR_std", fr_std)
        trial.set_user_attr("mean_error", float(mean_error))
        trial.set_user_attr("std_error", float(std_error))
        trial.set_user_attr("plausibility_penalty", float(plausibility_penalty))
        trial.set_user_attr("objective_value", float(objective_value))
        trial.set_user_attr("dd_neurons", dd_neurons)
        trial.set_user_attr("conn_probability", float(conn_probability))
        trial.set_user_attr("synaptic_weight", float(SYNAPTIC_WEIGHT))
        trial.set_user_attr("dd_drive__Hz", float(dd_drive__Hz))
        trial.set_user_attr("gamma_shape", float(gamma_shape))
        trial.set_user_attr("gfluctdv_enabled", ENABLE_GFLUCTDV)
        trial.set_user_attr("gfluctdv_noise_amplitude", float(gfluctdv_noise_amplitude))

        # Progress reporting
        if trial.number % 1 == 0:
            print(
                f"Trial {trial.number}: FR={fr_mean:.1f}±{fr_std:.1f}Hz, "
                f"obj={objective_value:.3f} (mean_err={mean_error:.3f}, std_err={std_error:.3f})"
            )

        return objective_value

    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        return 1000.0


##############################################################################
# Run Optimization Study
# ----------------------
#
# Create an Optuna study and optimize parameters using Tree-structured Parzen Estimator (TPE).
# TPE is a Bayesian optimization algorithm that builds probabilistic models of the objective function.

print(
    f"\nOptimizing network parameters | "
    f"Target: {TARGET_FR_MEAN__HZ:.1f}±{TARGET_FR_STD__HZ:.1f}Hz | "
    f"Trials: {N_TRIALS}\n"
)

# Create or load existing study (allows resuming interrupted optimization)
storage_name = f"sqlite:///{RESULTS_DIR}/optuna_dd_optimization.db"
study = optuna.create_study(
    direction="minimize",  # Minimize objective value
    sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
    study_name="dd_optimization",
    storage=storage_name,
    load_if_exists=True,
)

# Run optimization
study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT_SECONDS, show_progress_bar=True)

##############################################################################
# Analyze Results
# ---------------
#
# Extract the best trial and display optimized parameters.

best_trial = study.best_trial
print(f"\nCompleted {len(study.trials)} trials")
print(
    f"\nBest Trial {best_trial.number}: "
    f"FR={best_trial.user_attrs.get('FR_mean'):.1f}±{best_trial.user_attrs.get('FR_std'):.1f}Hz | "
    f"Objective={best_trial.value:.3f} | "
    f"Drive={best_trial.user_attrs.get('dd_drive__Hz'):.1f}Hz | "
    f"Neurons={best_trial.user_attrs.get('dd_neurons')} | "
    f"Conn={best_trial.user_attrs.get('conn_probability'):.2f}"
)

##############################################################################
# Save Optimized Parameters
# --------------------------
#
# Store the best parameters in JSON format for use in production simulations.


def trial_to_dict(t):
    """Convert trial to dictionary of parameters."""
    base_params = {
        k: t.user_attrs.get(k)
        for k in [
            "FR_mean",
            "FR_std",
            "dd_neurons",
            "conn_probability",
            "dd_drive__Hz",
            "gamma_shape",
            "mean_error",
            "std_error",
            "plausibility_penalty",
            "objective_value",
        ]
    }
    if ENABLE_GFLUCTDV:
        base_params["gfluctdv_noise_amplitude"] = t.user_attrs.get("gfluctdv_noise_amplitude")
    return base_params


results = {
    "target": {
        "FR_mean__Hz": TARGET_FR_MEAN__HZ,
        "FR_std__Hz": TARGET_FR_STD__HZ,
    },
    "input_parameters": {
        "gamma_shape": GAMMA_SHAPE,
        "gfluctdv_enabled": ENABLE_GFLUCTDV,
    },
    "best_trial": trial_to_dict(best_trial),
}

json_path = RESULTS_DIR / "dd_optimized_params.json"
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)

joblib.dump(study, RESULTS_DIR / "study.pkl")
print(f"Saved optimized parameters: {json_path}\n")

##############################################################################
# Visualize Optimization History
# -------------------------------
#
# Plot the optimization trajectory showing how the objective value improved over trials.

# Get colors from the style
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Left column: Optimization progress and FR convergence (shared x-axis)
ax_obj = fig.add_subplot(gs[0, 0])
ax_fr = fig.add_subplot(gs[1, 0], sharex=ax_obj)

# Right column: Parameter exploration and drive distribution
ax_param = fig.add_subplot(gs[0, 1])
ax_drive = fig.add_subplot(gs[1, 1])

# 1. Objective value over trials (log scale)
trial_numbers = [t.number for t in study.trials]
objective_values = [t.value for t in study.trials]
ax_obj.plot(trial_numbers, objective_values, "o-", alpha=0.6, markersize=4, color=colors[0])
ax_obj.axhline(best_trial.value, linestyle="--", color=colors[1], label=f"Best: {best_trial.value:.3f}")
ax_obj.set_ylabel("Objective Value (log scale)")
ax_obj.set_title("Optimization Progress")
ax_obj.set_yscale("log")
ax_obj.legend(framealpha=1.0, edgecolor="none")
ax_obj.tick_params(labelbottom=False)

# 2. Firing rate convergence
fr_means = [t.user_attrs.get("FR_mean", None) for t in study.trials]
fr_stds = [t.user_attrs.get("FR_std", None) for t in study.trials]
ax_fr.plot(trial_numbers, fr_means, "o-", alpha=0.6, markersize=4, color=colors[0], label="Mean FR")
ax_fr.axhline(TARGET_FR_MEAN__HZ, linestyle="--", color=colors[1], label=f"Target: {TARGET_FR_MEAN__HZ:.1f}Hz")
ax_fr.fill_between(
    trial_numbers,
    TARGET_FR_MEAN__HZ - TARGET_FR_STD__HZ,
    TARGET_FR_MEAN__HZ + TARGET_FR_STD__HZ,
    alpha=0.2,
    color=colors[1],
    label="Target Range",
)
ax_fr.set_xlabel("Trial")
ax_fr.set_ylabel("Firing Rate (Hz)")
ax_fr.set_title("Firing Rate Convergence")
ax_fr.legend(framealpha=1.0, edgecolor="none")

# 3. Parameter exploration: DD neurons vs Connection probability
dd_neurons_vals = [t.params.get("dd_neurons", None) for t in study.trials]
conn_prob_vals = [t.params.get("conn_prob", None) for t in study.trials]
scatter = ax_param.scatter(dd_neurons_vals, conn_prob_vals, c=objective_values, s=50, alpha=0.6)
ax_param.scatter(
    best_trial.params["dd_neurons"],
    best_trial.params["conn_prob"],
    s=200,
    marker="*",
    color=colors[3],
    label="Best Trial",
)
ax_param.set_xlabel("DD Neurons")
ax_param.set_ylabel("Connection Probability")
ax_param.set_title("Parameter Space Exploration")
ax_param.legend(framealpha=1.0, edgecolor="none")
plt.colorbar(scatter, ax=ax_param, label="Objective Value")

# 4. Drive frequency distribution
dd_drive_vals = [t.params.get("dd_drive", None) for t in study.trials]
ax_drive.hist(dd_drive_vals, bins=30, alpha=0.7, color=colors[2])
ax_drive.axvline(
    best_trial.params["dd_drive"],
    linestyle="--",
    color=colors[1],
    label=f'Best: {best_trial.params["dd_drive"]:.1f}Hz',
)
ax_drive.set_xlabel("DD Drive Frequency (Hz)")
ax_drive.set_ylabel("Number of Trials")
ax_drive.set_title("Drive Frequency Distribution")
ax_drive.legend(framealpha=1.0, edgecolor="none")

plt.tight_layout()
plt.savefig(RESULTS_DIR / "optimization_history.png", dpi=150, bbox_inches="tight")
plt.show()

##############################################################################
# Compare Target vs Achieved
# ---------------------------
#
# Validate that the optimization successfully matched the target firing rate statistics.

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Dumbbell plot comparing target vs achieved (two scalar endpoints per
# category — not a distribution, so Editor 8's "show all data points"
# rule is honoured by marking each value as an explicit point).
categories = ["Mean FR (Hz)", "Std FR (Hz)"]
targets = [TARGET_FR_MEAN__HZ, TARGET_FR_STD__HZ]
achieved = [
    best_trial.user_attrs.get("FR_mean"),
    best_trial.user_attrs.get("FR_std"),
]

x = np.arange(len(categories))

for xi, t, a in zip(x, targets, achieved):
    ax.plot([xi, xi], [t, a], color="gray", linestyle="--", alpha=0.6, zorder=1)

ax.scatter(x, targets, s=120, marker="o", label="Target", zorder=3)
ax.scatter(x, achieved, s=120, marker="D", label="Achieved", zorder=3)

for xi, t, a in zip(x, targets, achieved):
    ax.text(xi + 0.05, t, f"{t:.2f}", va="center", ha="left")
    ax.text(xi + 0.05, a, f"{a:.2f}", va="center", ha="left")

# Calculate percent errors
mean_error = abs(achieved[0] - targets[0]) / targets[0] * 100
std_error = abs(achieved[1] - targets[1]) / targets[1] * 100

ax.set_ylabel("Value")
ax.set_title(
    f"Firing Rate Optimization: Target vs Achieved\n"
    f"(Mean error: {mean_error:.1f}%, Std error: {std_error:.1f}%)"
)
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend(framealpha=1.0, edgecolor="none")

plt.tight_layout()
plt.savefig(RESULTS_DIR / "target_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

##############################################################################
# Using Optimized Parameters
# ---------------------------
#
# !!! important
#     The optimized parameters can be loaded and used in production simulations:
#
#     ```python
#     import json
#     from pathlib import Path
#
#     # Load optimized parameters
#     with open("results/dd_optimization/dd_optimized_params.json") as f:
#         params = json.load(f)
#
#     # Extract best values
#     dd_neurons = int(params["best_trial"]["dd_neurons"])
#     conn_prob = params["best_trial"]["conn_probability"]
#     dd_drive_Hz = params["best_trial"]["dd_drive__Hz"]
#
#     # Use in your simulation
#     descending_drive_pool = DescendingDrive__Pool(n=dd_neurons, ...)
#     network.connect("DD", "aMN", probability=conn_prob, ...)
#     ```

print("\n[DONE] Optimization complete!")
print(f"Best parameters saved to: {json_path}")
print(f"Full study saved to: {RESULTS_DIR / 'study.pkl'}")

# mkdocs_gallery_thumbnail_path = "gallery_thumbs/01_optimize_dd_for_target_firing_rate.png"
