"""
Optimize Descending Drive Parameters for Target Force Level
=============================================================

This script optimizes descending drive (DD) pool parameters to achieve a target
muscle force level specified as a percentage of the baseline steady-state force.

The baseline force is loaded from force_results.json (computed by compute_force_from_optimized_dd.py).
The user specifies a target percentage (e.g., 50 for 50% of baseline force).

The optimization tunes:
- DD constant drive level (1-100 Hz)

Fixed parameters (from baseline):
- DD neuron count (from baseline optimization)
- DD-to-MN connection probability (from baseline)
- DD-to-MN synaptic weight: 0.05 μS
- DD process type: Gamma distribution
- Gamma rate scale: 32
- Gamma shape parameter: Derived from MVC percentage via get_gamma_shape_for_mvc()
  (adjustable via MVC_PERCENT configuration parameter)

Firing rate statistics are allowed to vary - only force level is constrained.
"""

import os

os.environ["MPLBACKEND"] = "Agg"
if "DISPLAY" in os.environ:
    del os.environ["DISPLAY"]

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import optuna
from neo import SpikeTrain, Segment, Block
import quantities as pq
from neuron import h
from scipy.stats import wasserstein_distance

from examples.finetune.helper import (
    calculate_firing_rate_statistics,
    get_gamma_shape_for_mvc,
)
from myogen import RANDOM_GENERATOR, set_random_seed
from myogen.simulator import RecruitmentThresholds
from myogen.simulator.core.force.force_model import ForceModel
from myogen.simulator.neuron import Network
from myogen.simulator.neuron.populations import AlphaMN__Pool, DescendingDrive__Pool
from myogen.utils.nmodl import load_nmodl_mechanisms

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

##############################################################################
# Configuration
# -------------

# Simulation parameters
SIMULATION_TIME_MS = 5000.0  # 5 seconds for efficient optimization
TIMESTEP_MS = 0.1
N_MOTOR_UNITS = 100

# Force model parameters
RECORDING_FREQUENCY__HZ = 2048
LONGEST_DURATION_RISE_TIME__MS = 90.0
CONTRACTION_TIME_RANGE = 3

# Optimization parameters
N_TRIALS = 100
TIMEOUT_SECONDS = 3600  # 1 hour

# Use the same prefix as in optimization
STUDY_PREFIX = "VLVM_"

# Descending Drive parameters
MVC_PERCENT = 30.0  # MVC percentage for gamma shape parameter (adjustable)

# Input/output directories
BASELINE_DIR = Path("./results/force_validation")
RESULTS_DIR = Path("./results/force_optimization")

##############################################################################
# Helper Functions
# ----------------


def run_simulation_and_compute_force(
    dd_neurons,
    conn_probability,
    synaptic_weight,
    dd_drive__Hz,
    gamma_shape,
    recruitment_thresholds,
):
    """
    Run simulation and compute resulting force.

    Parameters
    ----------
    dd_neurons : int
        Number of DD neurons
    conn_probability : float
        DD-to-MN connection probability
    synaptic_weight : float
        DD-to-MN synaptic weight (μS)
    dd_drive__Hz : float
        DD constant drive level (Hz)
    gamma_shape : float
        Gamma distribution shape parameter (controls spike regularity)
    recruitment_thresholds : np.ndarray
        Motor unit recruitment thresholds

    Returns
    -------
    float
        Mean steady-state force (a.u.)
    int
        Number of active motor units
    float
        Mean firing rate (Hz)
    float
        Standard deviation of firing rate (Hz)
    float
        Wasserstein distance between actual and theoretical normal distribution
    """
    # Create motor neuron pool with DEFAULT config
    motor_neuron_pool = AlphaMN__Pool(
        recruitment_thresholds__array=recruitment_thresholds,
        config_file="alpha_mn_default.yaml",
    )

    # Apply Gfluctdv if it was enabled during baseline DD optimization
    if GFLUCTDV_ENABLED and GFLUCTDV_NOISE_AMPLITUDE is not None:
        for cell in motor_neuron_pool:
            cell.insert_Gfluctdv()
            for d in cell.dend:
                d.std_e_Gfluctdv = GFLUCTDV_NOISE_AMPLITUDE
                d.std_i_Gfluctdv = GFLUCTDV_NOISE_AMPLITUDE

    # Create descending drive pool (using Gamma distribution)
    descending_drive_pool = DescendingDrive__Pool(
        n=dd_neurons,
        timestep__ms=TIMESTEP_MS,
        process_type="gamma",
        shape=gamma_shape,
    )

    # Create network and connect
    network = Network(
        {
            "DD": descending_drive_pool,
            "aMN": motor_neuron_pool,
        }
    )

    network.connect(
        source="DD",
        target="aMN",
        probability=conn_probability,
        weight__μS=synaptic_weight,
    )

    network.connect_from_external(source="cortical_input", target="DD", weight__μS=1.0)
    dd_netcons = network.get_netcons("cortical_input", "DD")

    # Setup spike recording
    dd_spike_times = [[] for _ in range(len(descending_drive_pool))]

    mn_spike_recorders = []
    for cell in motor_neuron_pool:
        spike_recorder = h.Vector()
        nc = h.NetCon(cell.soma(0.5)._ref_v, None, sec=cell.soma)
        nc.threshold = 50
        nc.record(spike_recorder)
        mn_spike_recorders.append(spike_recorder)

    # Create CONSTANT drive pattern
    time_points = int(SIMULATION_TIME_MS / TIMESTEP_MS)
    drive_signal = np.ones(time_points) * dd_drive__Hz + np.clip(
        RANDOM_GENERATOR.normal(0, 1.0, size=time_points), 0, None
    )

    # Initialize simulation
    h.load_file("stdrun.hoc")
    h.dt = TIMESTEP_MS
    h.tstop = SIMULATION_TIME_MS

    # Initialize voltages
    for section, voltage in zip(*motor_neuron_pool.get_initialization_data()):
        section.v = voltage
    for section, voltage in zip(*descending_drive_pool.get_initialization_data()):
        section.v = voltage

    h.finitialize()

    # Simulation loop
    step_counter = 0
    while h.t < h.tstop:
        current_drive = drive_signal[min(step_counter, len(drive_signal) - 1)]

        for dd_cell in descending_drive_pool:
            if dd_cell.integrate(current_drive):
                dd_spike_times[dd_cell.pool__ID].append(h.t + 1)
                spike_time = h.t
                if spike_time < h.tstop:
                    dd_netcons[dd_cell.pool__ID].event(spike_time)

        h.fadvance()
        step_counter += 1

    # Convert to Neo spike trains
    dt_s = h.dt / 1000.0
    mn_segment = Segment(name="Motor Neurons")
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

    # Create Block for force model
    spike_train__Block = Block(name="Motor Unit Pool")
    spike_train__Block.segments = [mn_segment]

    # Count active neurons
    n_active = sum(1 for st in mn_segment.spiketrains if len(st) > 1)

    # Calculate firing rate statistics
    stats = calculate_firing_rate_statistics(mn_segment.spiketrains)
    fr_mean = stats["FR_mean"]
    fr_std = stats["FR_std"]
    data = stats["firing_rates"]

    # Calculate Wasserstein distance between actual and theoretical normal distribution
    if len(data) > 0:
        wdist = wasserstein_distance(data, np.random.normal(fr_mean, fr_std, len(data)))
    else:
        wdist = 0.0

    # Create force model
    force_model = ForceModel(
        recruitment_thresholds=recruitment_thresholds,
        recording_frequency__Hz=RECORDING_FREQUENCY__HZ,
        longest_duration_rise_time__ms=LONGEST_DURATION_RISE_TIME__MS,
        contraction_time_range__unitless=CONTRACTION_TIME_RANGE,
    )

    # Generate force
    force_output = force_model.generate_force(spike_train__Block=spike_train__Block)
    force_signal = force_output.magnitude[:, 0]

    # Calculate steady-state force (last 50%)
    steady_state_start_idx = len(force_signal) // 2
    steady_state_force = force_signal[steady_state_start_idx:]
    force_mean_steady = np.mean(steady_state_force)

    return force_mean_steady, n_active, fr_mean, fr_std, wdist


##############################################################################
# Objective Function
# ------------------


def objective(trial, target_force__au, recruitment_thresholds):
    """
    Optuna objective function for DD parameter optimization.

    Parameters
    ----------
    trial : optuna.Trial
        Current optimization trial
    target_force__au : float
        Target force level (a.u.)
    recruitment_thresholds : np.ndarray
        Motor unit recruitment thresholds

    Returns
    -------
    float
        Relative force error (lower is better)
    """

    try:
        # TUNABLE DESCENDING DRIVE PARAMETERS
        synaptic_weight = 0.05  # FIXED
        dd_neurons = DD_NEURONS
        conn_probability = CONN_PROBABILITY
        dd_drive__Hz = trial.suggest_float("dd_drive", 1.0, 100.0)
        gamma_shape = get_gamma_shape_for_mvc(MVC_PERCENT, MVC_SHAPE_VALUE)

        # Diagnostic: Print DD parameters every 10 trials
        if trial.number % 10 == 0:
            print(f"\n  Trial {trial.number} DD parameters:")
            print(f"    DD neurons: {dd_neurons}")
            print(f"    Conn prob: {conn_probability:.3f}")
            print(f"    Weight: {synaptic_weight:.4f} μS (fixed)")
            print(f"    Drive: {dd_drive__Hz:.1f} Hz")
            print(f"    MVC: {MVC_PERCENT:.1f}%")
            print(f"    Gamma shape: {gamma_shape:.2f} (CV={1 / gamma_shape**0.5:.3f})")

        # Run simulation and compute force
        force_mean_steady, n_active, fr_mean, fr_std, wdist = (
            run_simulation_and_compute_force(
                dd_neurons,
                conn_probability,
                synaptic_weight,
                dd_drive__Hz,
                gamma_shape,
                recruitment_thresholds,
            )
        )

        # Check if we have enough active neurons
        if n_active < 5:  # Need at least 5 active neurons for realistic force
            penalty = 1000.0 + (5 - n_active) * 100.0
            return penalty

        # Calculate relative force error
        force_error = abs(force_mean_steady - target_force__au) / target_force__au

        # Store attributes for analysis
        trial.set_user_attr("force_achieved", float(force_mean_steady))
        trial.set_user_attr("force_target", float(target_force__au))
        trial.set_user_attr("force_error", float(force_error))
        trial.set_user_attr("n_active", n_active)
        trial.set_user_attr("dd_neurons", dd_neurons)
        trial.set_user_attr("conn_probability", float(conn_probability))
        trial.set_user_attr("synaptic_weight", float(synaptic_weight))
        trial.set_user_attr("dd_drive__Hz", float(dd_drive__Hz))
        trial.set_user_attr("mvc_percent", float(MVC_PERCENT))
        trial.set_user_attr("gamma_shape", float(gamma_shape))
        trial.set_user_attr("FR_mean", float(fr_mean))
        trial.set_user_attr("FR_std", float(fr_std))
        trial.set_user_attr("wasserstein_distance", float(wdist))

        # Print diagnostic info every 10 trials
        if trial.number % 10 == 0:
            print(
                f"  Trial {trial.number} results:\n"
                f"    Force: {force_mean_steady:.4f} a.u. (target: {target_force__au:.4f} a.u.)\n"
                f"    Error: {force_error:.2%}\n"
                f"    FR: {fr_mean:.1f}±{fr_std:.1f} Hz (Wasserstein: {wdist:.3f})\n"
                f"    MVC: {MVC_PERCENT:.1f}%\n"
                f"    Gamma shape: {gamma_shape:.2f} (CV={1 / gamma_shape**0.5:.3f})\n"
                f"    Active neurons: {n_active}/{N_MOTOR_UNITS}"
            )

        return force_error

    except Exception as e:
        print(f"Trial failed with error: {e}")
        return 1000.0  # High penalty for failed trials


##############################################################################
# Optimization
# ------------


def optimize_for_force_level(target_force_pct, n_trials=N_TRIALS, reset_study=False):
    """
    Run Optuna optimization for a specific force level.

    Parameters
    ----------
    target_force_pct : float
        Target force as percentage of baseline (e.g., 50 for 50%)
    n_trials : int
        Number of optimization trials
    reset_study : bool, optional
        If True, delete existing study database, by default False

    Returns
    -------
    optuna.Study
        Completed optimization study
    """
    print(f"\n{'=' * 80}")
    print(f"OPTIMIZING FOR {target_force_pct}% OF BASELINE FORCE")
    print(f"{'=' * 80}\n")

    # Calculate target force
    target_force__au = BASELINE_FORCE__AU * (target_force_pct / 100.0)
    gamma_shape = get_gamma_shape_for_mvc(MVC_PERCENT, MVC_SHAPE_VALUE)

    print(f"Baseline force: {BASELINE_FORCE__AU:.4f} a.u.")
    print(f"Target force:   {target_force__au:.4f} a.u. ({target_force_pct}%)")
    print("\nDD configuration:")
    print(f"  MVC:         {MVC_PERCENT:.1f}%")
    print(f"  Gamma shape: {gamma_shape:.2f} (CV={1 / gamma_shape**0.5:.3f})")

    # Generate recruitment thresholds (same as baseline)
    max_threshold = 1.0
    recruitment_range = 100
    deluca_slope = 5

    recruitment_thresholds, _ = RecruitmentThresholds(
        N=N_MOTOR_UNITS,
        recruitment_range__ratio=recruitment_range,
        deluca__slope=deluca_slope,
        konstantin__max_threshold__ratio=max_threshold,
        mode="combined",
    )

    # Create study
    storage_name = f"sqlite:///{RESULTS_DIR}/{STUDY_PREFIX}optuna_force_{int(target_force_pct)}pct.db"

    # Delete existing database if reset requested
    if reset_study:
        db_path = (
            RESULTS_DIR / f"{STUDY_PREFIX}optuna_force_{int(target_force_pct)}pct.db"
        )
        if db_path.exists():
            db_path.unlink()
            print(f"✓ Deleted old study database: {db_path}\n")

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name=f"{STUDY_PREFIX}force_{int(target_force_pct)}pct_optimization",
        storage=storage_name,
        load_if_exists=True,
    )

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, target_force__au, recruitment_thresholds),
        n_trials=n_trials,
        timeout=TIMEOUT_SECONDS,
        show_progress_bar=True,
    )

    # Print results
    print(f"\n{'=' * 80}")
    print(f"OPTIMIZATION COMPLETE FOR {target_force_pct}%")
    print(f"{'=' * 80}\n")

    best_trial = study.best_trial

    print(f"Best trial: {best_trial.number}")
    print(f"  Force error: {best_trial.value:.2%}")
    print(
        f"  Active neurons: {best_trial.user_attrs.get('n_active', 'N/A')}/{N_MOTOR_UNITS}"
    )
    print("\nForce results:")
    print(f"  Target:   {target_force__au:.4f} a.u. ({target_force_pct}% of baseline)")
    print(
        f"  Achieved: {best_trial.user_attrs.get('force_achieved', 'N/A'):.4f} a.u. "
        f"(error: {best_trial.user_attrs.get('force_error', 'N/A'):.1%})"
    )

    print("\nOptimized DD parameters:")
    print(f"  DD neurons:       {best_trial.user_attrs.get('dd_neurons')}")
    print(f"  Conn probability: {best_trial.user_attrs.get('conn_probability'):.3f}")
    print(f"  Synaptic weight:  {best_trial.user_attrs.get('synaptic_weight'):.4f} μS")
    print(f"  DD drive level:   {best_trial.user_attrs.get('dd_drive__Hz'):.2f} Hz")
    mvc_val = best_trial.user_attrs.get("mvc_percent", MVC_PERCENT)
    gamma_shape_val = best_trial.user_attrs.get(
        "gamma_shape", get_gamma_shape_for_mvc(MVC_PERCENT)
    )
    print(f"  MVC:              {mvc_val:.1f}%")
    print(
        f"  Gamma shape:      {gamma_shape_val:.2f} (CV={1 / gamma_shape_val**0.5:.3f})"
    )

    return study


##############################################################################
# Results Export
# --------------


def export_results(study, target_force_pct):
    """
    Export optimization results to JSON files.

    Parameters
    ----------
    study : optuna.Study
        Completed study
    target_force_pct : float
        Target force percentage
    """
    best_trial = study.best_trial

    # Prepare results dictionary
    dd_parameters = {
        "dd_neurons": best_trial.user_attrs.get("dd_neurons"),
        "conn_probability": best_trial.user_attrs.get("conn_probability"),
        "synaptic_weight": best_trial.user_attrs.get("synaptic_weight"),
        "dd_drive__Hz": best_trial.user_attrs.get("dd_drive__Hz"),
        "mvc_percent": best_trial.user_attrs.get("mvc_percent"),
        "gamma_shape": best_trial.user_attrs.get("gamma_shape"),
    }

    # Include Gfluctdv parameters if they were used
    if GFLUCTDV_ENABLED and GFLUCTDV_NOISE_AMPLITUDE is not None:
        dd_parameters["gfluctdv_noise_amplitude"] = GFLUCTDV_NOISE_AMPLITUDE

    results = {
        "target_force_pct": target_force_pct,
        "baseline_force__au": BASELINE_FORCE__AU,
        "target_force__au": best_trial.user_attrs.get("force_target"),
        "achieved_force__au": best_trial.user_attrs.get("force_achieved"),
        "force_error": best_trial.user_attrs.get("force_error"),
        "gfluctdv_enabled": GFLUCTDV_ENABLED,
        "dd_parameters": dd_parameters,
        "optimization": {
            "trial_number": best_trial.number,
            "n_trials": len(study.trials),
            "n_active": best_trial.user_attrs.get("n_active"),
        },
    }

    # Save to JSON
    json_path = (
        RESULTS_DIR
        / f"{STUDY_PREFIX}dd_optimized_params_force_{int(target_force_pct)}pct.json"
    )
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved parameters to: {json_path}")

    # Save study
    study_path = (
        RESULTS_DIR / f"{STUDY_PREFIX}study_force_{int(target_force_pct)}pct.pkl"
    )
    joblib.dump(study, study_path)
    print(f"Saved study to: {study_path}")

    return json_path


##############################################################################
# Main Execution
# --------------


def main():
    """Main execution function."""
    global \
        BASELINE_FORCE__AU, \
        DD_NEURONS, \
        CONN_PROBABILITY, \
        MVC_SHAPE_VALUE, \
        STUDY_PREFIX, \
        GFLUCTDV_ENABLED, \
        GFLUCTDV_NOISE_AMPLITUDE

    import argparse

    parser = argparse.ArgumentParser(
        description="Optimize DD parameters for target force level"
    )
    parser.add_argument(
        "--target-force-pct",
        type=float,
        default=MVC_PERCENT,
        required=False,
        help="Target force as percentage of baseline (e.g., 50 for 50%%)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=N_TRIALS,
        help=f"Number of optimization trials (default: {N_TRIALS})",
    )
    parser.add_argument(
        "--reset-study",
        action="store_true",
        help="Delete existing Optuna database and start fresh",
    )
    parser.add_argument(
        "--study-prefix",
        type=str,
        default=STUDY_PREFIX,
        help=f"Study prefix for file naming (default: {STUDY_PREFIX})",
    )

    args = parser.parse_args()

    # Update global STUDY_PREFIX with command-line argument
    STUDY_PREFIX = args.study_prefix

    # Initialize environment
    set_random_seed(42)
    load_nmodl_mechanisms()
    h.secondorder = 2

    # Create results directory
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)

    # Load baseline force data
    print(f"\n{'=' * 80}")
    print("LOADING BASELINE FORCE DATA")
    print(f"{'=' * 80}\n")

    baseline_file = BASELINE_DIR / f"{args.study_prefix}force_results.json"

    if not baseline_file.exists():
        raise FileNotFoundError(
            f"Baseline force results not found at {baseline_file}.\n"
            "Please run the following scripts first:\n"
            "  1. python optimize_dd_for_target_firing_rate.py\n"
            "  2. python compute_force_from_optimized_dd.py"
        )

    with open(baseline_file, "r") as f:
        baseline_results = json.load(f)

    # Extract baseline parameters and set globals
    # Handle both old and new JSON formats
    if "force_stats" in baseline_results:
        BASELINE_FORCE__AU = baseline_results["force_stats"]["steady_state_mean__au"]
    elif "force" in baseline_results:
        BASELINE_FORCE__AU = baseline_results["force"]["mean__au"]
    else:
        raise KeyError(
            "Cannot find force data in baseline results (expected 'force_stats' or 'force' key)"
        )

    DD_NEURONS = baseline_results["dd_parameters"]["dd_neurons"]
    CONN_PROBABILITY = baseline_results["dd_parameters"]["conn_probability"]
    MVC_SHAPE_VALUE = baseline_results["dd_parameters"]["mvc_shape_value"]

    # Load Gfluctdv settings if present in baseline optimization
    GFLUCTDV_ENABLED = baseline_results.get("gfluctdv_enabled", False)
    GFLUCTDV_NOISE_AMPLITUDE = baseline_results["dd_parameters"].get(
        "gfluctdv_noise_amplitude", None
    )

    # Print configuration
    print(f"Baseline steady-state force: {BASELINE_FORCE__AU:.4f} a.u.")
    print("Configuration:")
    print(f"  DD neurons:       {DD_NEURONS}")
    print(f"  Conn probability: {CONN_PROBABILITY:.3f}")
    print(f"  MVC:              {MVC_PERCENT:.1f}%")
    if GFLUCTDV_ENABLED:
        print(
            f"  Gfluctdv:         ENABLED (noise={GFLUCTDV_NOISE_AMPLITUDE:.2e} S/cm²)"
        )
    print(
        f"  Gamma shape:      {get_gamma_shape_for_mvc(MVC_PERCENT, MVC_SHAPE_VALUE):.2f} "
        f"(CV={1 / get_gamma_shape_for_mvc(MVC_PERCENT, MVC_SHAPE_VALUE) ** 0.5:.3f})"
    )

    print("\n" + "=" * 80)
    print("DD PARAMETER OPTIMIZATION FOR TARGET FORCE LEVEL")
    print("=" * 80)
    print(f"Target force: {args.target_force_pct}% of baseline")
    print(f"Trials: {args.n_trials}")
    print(f"Reset study: {args.reset_study}")

    # Run optimization
    study = optimize_for_force_level(
        target_force_pct=args.target_force_pct,
        n_trials=args.n_trials,
        reset_study=args.reset_study,
    )

    # Export results
    export_results(study, args.target_force_pct)


if __name__ == "__main__":
    main()
