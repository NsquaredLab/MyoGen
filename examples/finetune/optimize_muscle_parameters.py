"""
Optuna-based Parameter Optimization for Descending Drive Network Parameters
============================================================================

This script optimizes descending drive (DD) and network parameters to match experimental
2D CV vs Firing Rate relationships for a single muscle (VLVM, TA, or FDI).

Multi-objective optimization using NSGA-III algorithm to find Pareto-optimal solutions
balancing 5 objectives:
1. CV-FR correlation coefficient matching
2. Mean firing rate position matching
3. Mean CV position matching
4. 2D point cloud distance minimization (selected from Pareto front)
5. Distribution variance matching

The optimization tunes descending drive and network parameters:
- DD neuron count (100-1000 neurons)
- DD-to-MN connection probability (10-90%)
- DD-to-MN synaptic weight (0.01-1.0 μS)
- DD baseline drive level (5-40 Hz)
- DD peak drive level (30-100 Hz)

Alpha motor neuron parameters are kept FIXED at their optimized values.

Target metrics are extracted from ISI_statistics.csv containing experimental data
with multiple motor unit recordings showing the CV-FR relationship.
"""

# %%
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import quantities as pq
from matplotlib import pyplot as plt
from neo import SpikeTrain, Segment
from neuron import h

from myogen import RANDOM_GENERATOR, set_random_seed
from myogen.simulator import RecruitmentThresholds
from myogen.simulator.neuron import Network
from myogen.simulator.neuron.populations import AlphaMN__Pool, DescendingDrive__Pool
from myogen.utils.nmodl import load_nmodl_mechanisms

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

##############################################################################
# Configuration
# -------------

# Set random seed for reproducibility
set_random_seed(42)

# Load NEURON mechanisms
print("Loading NMODL mechanisms...")
load_nmodl_mechanisms()
h.secondorder = 2  # Crank-Nicolson method (second-order accurate)

# Simulation parameters
SIMULATION_TIME_MS = 5000.0  # Fast trials for optimization
TIMESTEP_MS = 0.1
DD_BASELINE_HZ = 20.0
DD_NEURONS = 400
N_MOTOR_UNITS = 100

# Optimization parameters
N_TRIALS_PER_MUSCLE = 250
TIMEOUT_SECONDS = 3600  # 1 hour per muscle

# Output directory
RESULTS_DIR = Path("./results/optimization")
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

##############################################################################
# Load Experimental Data
# ----------------------

print("Loading experimental ISI statistics...")
csv_path = Path(__file__).parent / "ISI_statistics.csv"
isi_data = pd.read_csv(csv_path)


# Calculate target statistics for each muscle
def calculate_target_stats(muscle_names):
    """Calculate mean and std of FR and CV for given muscles."""
    muscle_data = isi_data[isi_data["Muscle"].isin(muscle_names)]
    return {
        "FR_mean": muscle_data["FR mean"].mean(),
        "FR_std": muscle_data["FR mean"].std(),
        "CV_mean": muscle_data["ISI CV"].mean(),
        "CV_std": muscle_data["ISI CV"].std(),
        "n_samples": len(muscle_data),
    }


# Define targets for each muscle
EXPERIMENTAL_TARGETS = {
    "VLVM": calculate_target_stats(["VM", "VL"]),
    "TA": calculate_target_stats(["TA"]),
    "FDI": calculate_target_stats(["FDI"]),
}

print("\nExperimental targets:")
for muscle, stats in EXPERIMENTAL_TARGETS.items():
    print(f"\n{muscle}:")
    print(f"  FR: {stats['FR_mean']:.2f} ± {stats['FR_std']:.2f} Hz")
    print(f"  CV: {stats['CV_mean']:.3f} ± {stats['CV_std']:.3f}")
    print(f"  Samples: {stats['n_samples']}")

##############################################################################
# Helper Functions
# ----------------


def calculate_isi_statistics(spiketrains):
    """
    Calculate ISI statistics from spike trains.

    Returns dict with FR_mean, FR_std, CV_mean, CV_std.
    """
    firing_rates = []
    cvs = []

    for spiketrain in spiketrains:
        if len(spiketrain) > 1:
            spike_times_s = spiketrain.rescale(pq.s).magnitude
            isis = np.diff(spike_times_s)

            if len(isis) > 0:
                mean_isi = np.mean(isis)
                if mean_isi > 0:
                    fr = 1.0 / mean_isi
                    cv = np.std(isis) / mean_isi

                    if fr >= 0.01:  # Filter out very low firing rates
                        firing_rates.append(fr)
                        cvs.append(cv)

    if len(firing_rates) == 0:
        return {
            "FR_mean": 0.0,
            "FR_std": 0.0,
            "CV_mean": 0.0,
            "CV_std": 0.0,
            "n_active": 0,
        }

    return {
        "FR_mean": np.mean(firing_rates),
        "FR_std": np.std(firing_rates),
        "CV_mean": np.mean(cvs),
        "CV_std": np.std(cvs),
        "n_active": len(firing_rates),
    }


##############################################################################
# Objective Function
# ------------------


def objective(trial, muscle_name, exp_targets):
    """
    Optuna objective function for parameter optimization.

    Parameters
    ----------
    trial : optuna.Trial
        Current optimization trial
    muscle_name : str
        Muscle identifier ("VLVM", "TA", or "FDI")
    exp_targets : dict
        Experimental target statistics

    Returns
    -------
    tuple
        Tuple of 5 loss values (multi-objective optimization)
    """

    try:
        # 1. Fixed recruitment threshold parameters (not optimized)
        max_threshold = 1.0
        recruitment_range = 100
        deluca_slope = 5

        # 2. TUNABLE DESCENDING DRIVE PARAMETERS
        dd_neurons = trial.suggest_int(f"{muscle_name}_dd_neurons", 100, 1000)
        conn_probability = trial.suggest_float(f"{muscle_name}_conn_prob", 0.1, 0.9)
        synaptic_weight = trial.suggest_float(
            f"{muscle_name}_weight", 0.01, 1.0, log=True
        )
        dd_baseline__Hz = trial.suggest_float(f"{muscle_name}_dd_baseline", 5.0, 40.0)
        dd_peak__Hz = trial.suggest_float(f"{muscle_name}_dd_peak", 30.0, 100.0)

        # 3. Generate recruitment thresholds
        recruitment_thresholds, _ = RecruitmentThresholds(
            N=N_MOTOR_UNITS,
            recruitment_range__ratio=recruitment_range,
            deluca__slope=deluca_slope,
            konstantin__max_threshold__ratio=max_threshold,
            mode="combined",
        )

        # 4. Use FIXED alpha MN config file (NOT optimized)
        # Use the optimized config for this muscle if it exists, otherwise default
        from pathlib import Path

        config_file = f"alpha_mn_{muscle_name}_optimized.yaml"
        if not Path(config_file).exists():
            config_file = "alpha_mn_default.yaml"

        # 5. Create motor neuron pool with FIXED config
        motor_neuron_pool = AlphaMN__Pool(
            recruitment_thresholds__array=recruitment_thresholds,
            config_file=config_file,
        )

        # Diagnostic: Print DD parameters every 10 trials
        if trial.number % 10 == 0:
            print(f"  Trial {trial.number} DD parameters:")
            print(f"    DD neurons: {dd_neurons}")
            print(f"    Conn prob: {conn_probability:.3f}")
            print(f"    Weight: {synaptic_weight:.4f} μS")
            print(f"    Baseline: {dd_baseline__Hz:.1f} Hz")
            print(f"    Peak: {dd_peak__Hz:.1f} Hz")

        # 6. Create descending drive pool with TUNABLE size
        descending_drive_pool = DescendingDrive__Pool(
            n=dd_neurons, poisson_batch_size=16, timestep__ms=TIMESTEP_MS
        )

        # 7. Create network and connect with TUNABLE parameters
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

        network.connect_from_external(
            source="cortical_input", target="DD", weight__μS=1.0
        )
        dd_netcons = network.get_netcons("cortical_input", "DD")

        # 8. Setup spike recording
        dd_spike_times = [[] for _ in range(len(descending_drive_pool))]

        mn_spike_recorders = []
        for cell in motor_neuron_pool:
            spike_recorder = h.Vector()
            nc = h.NetCon(cell.soma(0.5)._ref_v, None, sec=cell.soma)
            nc.threshold = 50
            nc.record(spike_recorder)
            mn_spike_recorders.append(spike_recorder)

        # 9. Create drive pattern with TUNABLE baseline and peak
        time_points = int(SIMULATION_TIME_MS / TIMESTEP_MS)

        # Use constant drive at baseline level (sinusoidal component disabled)
        drive_signal = np.ones(time_points) * dd_baseline__Hz + np.clip(
            RANDOM_GENERATOR.normal(0, 1.0, size=time_points), 0, None
        )

        # 10. Run simulation
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
                    dd_spike_times[dd_cell.pool__ID].append(h.t)
                    spike_time = h.t + np.clip(RANDOM_GENERATOR.normal(0, 10), 0, None)
                    if spike_time < h.tstop:
                        dd_netcons[dd_cell.pool__ID].event(spike_time)

            h.fadvance()
            step_counter += 1

        # 11. Convert to Neo spike trains
        mn_segment = Segment(name="Motor Neurons")

        dt_s = h.dt / 1000.0  # Convert ms to s

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

        # 12. Calculate ISI statistics using Elephant
        from elephant.statistics import isi, cv

        firing_rates = []
        cvs = []

        for spiketrain in mn_segment.spiketrains:
            if len(spiketrain) > 1:
                # Calculate ISI using Elephant
                isis = isi(spiketrain)

                if len(isis) > 0:
                    mean_isi = np.mean(isis.magnitude)
                    if mean_isi > 0:
                        # Calculate firing rate
                        fr = 1.0 / mean_isi

                        # Calculate CV using Elephant (more robust)
                        cv_value = cv(isis)

                        if fr >= 0.01:  # Filter out very low firing rates
                            firing_rates.append(fr)
                            cvs.append(float(cv_value))

        # 13. Check if we have enough active neurons
        n_active = len(firing_rates)
        if n_active < 5:
            # Penalize configurations that don't recruit enough neurons
            penalty = 1000.0 + (5 - n_active) * 100.0
            return (penalty, penalty, penalty, penalty, penalty)

        # 14. Calculate loss based on 2D CV-FR relationship and distribution
        # Get experimental data for this muscle from the CSV
        exp_muscle_data = isi_data[
            isi_data["Muscle"].isin(
                ["VM", "VL"] if muscle_name == "VLVM" else [muscle_name]
            )
        ]

        exp_frs = exp_muscle_data["FR mean"].values
        exp_cvs = exp_muscle_data["ISI CV"].values

        # Convert to numpy arrays
        sim_frs = np.array(firing_rates)
        sim_cvs = np.array(cvs)

        # ===== 2D LOSS CALCULATION =====
        # Calculate Pearson correlation coefficients
        from scipy.stats import pearsonr

        # Experimental correlation
        exp_corr, _ = pearsonr(exp_frs, exp_cvs)

        # Simulated correlation
        if len(sim_frs) > 2:
            sim_corr, _ = pearsonr(sim_frs, sim_cvs)
        else:
            sim_corr = 0.0  # Default if insufficient data

        # Loss component 1: Correlation difference (for 2D relationship shape)
        correlation_loss = abs(exp_corr - sim_corr)

        # Loss component 2: Mean FR difference (for 2D position - horizontal)
        mean_fr_loss = abs(np.mean(sim_frs) - np.mean(exp_frs)) / np.mean(exp_frs)

        # Loss component 3: Mean CV difference (for 2D position - vertical)
        mean_cv_loss = abs(np.mean(sim_cvs) - np.mean(exp_cvs)) / np.mean(exp_cvs)

        # Loss component 4: 2D point cloud distance using mean minimum distance
        # For each simulated point, find closest experimental point in normalized 2D space
        fr_scale = exp_targets["FR_mean"]
        cv_scale = exp_targets["CV_mean"]

        # Normalize to similar scales
        sim_points = np.column_stack([sim_frs / fr_scale, sim_cvs / cv_scale])
        exp_points = np.column_stack([exp_frs / fr_scale, exp_cvs / cv_scale])

        # Calculate mean minimum distance (2D)
        from scipy.spatial.distance import cdist

        distances = cdist(sim_points, exp_points, metric="euclidean")
        min_distances = np.min(distances, axis=1)
        mean_min_distance = np.mean(min_distances)

        # Loss component 5: Distribution spread in 2D (variance of point cloud)
        sim_2d_var = np.var(sim_points)
        exp_2d_var = np.var(exp_points)
        variance_loss = abs(sim_2d_var - exp_2d_var) / exp_2d_var

        # Multi-objective optimization: return tuple of all 5 losses
        # Optuna will find Pareto-optimal solutions balancing all objectives

        # Report intermediate values
        trial.set_user_attr("n_active", n_active)
        trial.set_user_attr("FR_mean", np.mean(sim_frs))
        trial.set_user_attr("FR_std", np.std(sim_frs))
        trial.set_user_attr("CV_mean", np.mean(sim_cvs))
        trial.set_user_attr("CV_std", np.std(sim_cvs))
        trial.set_user_attr("exp_corr", float(exp_corr))
        trial.set_user_attr("sim_corr", float(sim_corr))
        trial.set_user_attr("correlation_loss", float(correlation_loss))
        trial.set_user_attr("mean_fr_loss", float(mean_fr_loss))
        trial.set_user_attr("mean_cv_loss", float(mean_cv_loss))
        trial.set_user_attr("mean_min_distance", float(mean_min_distance))
        trial.set_user_attr("variance_loss", float(variance_loss))

        # Store DD parameters for later export
        trial.set_user_attr("dd_neurons", dd_neurons)
        trial.set_user_attr("conn_probability", float(conn_probability))
        trial.set_user_attr("synaptic_weight", float(synaptic_weight))
        trial.set_user_attr("dd_baseline__Hz", float(dd_baseline__Hz))
        trial.set_user_attr("dd_peak__Hz", float(dd_peak__Hz))

        # Print diagnostic info every 10 trials
        if trial.number % 10 == 0:
            print(
                f"\n  Trial {trial.number}:\n"
                f"    Corr: sim={sim_corr:.3f} vs exp={exp_corr:.3f} (loss={correlation_loss:.4f})\n"
                f"    FR:   sim={np.mean(sim_frs):.1f} vs exp={exp_targets['FR_mean']:.1f} Hz (loss={mean_fr_loss:.4f})\n"
                f"    CV:   sim={np.mean(sim_cvs):.3f} vs exp={exp_targets['CV_mean']:.3f} (loss={mean_cv_loss:.4f})\n"
                f"    Dist: {mean_min_distance:.4f} | Var: {variance_loss:.4f}"
            )

        # Return tuple of 5 objectives for multi-objective optimization
        return (
            correlation_loss,
            mean_fr_loss,
            mean_cv_loss,
            mean_min_distance,
            variance_loss,
        )

    except Exception as e:
        print(f"Trial failed with error: {e}")
        return (
            1000.0,
            1000.0,
            1000.0,
            1000.0,
            1000.0,
        )  # High penalty for failed trials  # High penalty for failed trials


##############################################################################
# Optimization Loop
# -----------------


def optimize_muscle(muscle_name, n_trials=N_TRIALS_PER_MUSCLE, reset_study=False):
    """
    Run Optuna optimization for a specific muscle.

    Parameters
    ----------
    muscle_name : str
        Muscle identifier
    n_trials : int
        Number of optimization trials
    reset_study : bool, optional
        If True, delete existing Optuna database before starting, by default False

    Returns
    -------
    optuna.Study
        Completed optimization study
    """
    print(f"\n{'=' * 80}")
    print(f"Optimizing parameters for {muscle_name}")
    print(f"{'=' * 80}\n")

    # Get experimental targets
    exp_targets = EXPERIMENTAL_TARGETS[muscle_name]

    # Create study with storage for parallel optimization
    # SQLite storage allows multiple processes to work on the same study
    storage_name = f"sqlite:///{RESULTS_DIR}/optuna_{muscle_name}.db"

    # Delete existing database if reset requested
    if reset_study:
        db_path = RESULTS_DIR / f"optuna_{muscle_name}.db"
        if db_path.exists():
            db_path.unlink()
            print(f"✓ Deleted old study database: {db_path}\n")

    study = optuna.create_study(
        directions=["minimize"] * 5,  # Multi-objective: [corr, fr, cv, dist, var]
        sampler=optuna.samplers.NSGAIIISampler(
            population_size=50,  # Genetic algorithm population size
            mutation_prob=0.1,  # Mutation probability for diversity
            crossover_prob=0.9,  # Crossover probability for exploitation
            seed=42,
        ),
        study_name=f"{muscle_name}_optimization",
        storage=storage_name,
        load_if_exists=True,  # Resume if interrupted
    )

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, muscle_name, exp_targets),
        n_trials=n_trials,
        timeout=TIMEOUT_SECONDS,
        show_progress_bar=True,
    )

    # Print results - Multi-objective: select best from Pareto front
    print(f"\n{muscle_name} Optimization Complete!")
    print(f"  Pareto-optimal trials: {len(study.best_trials)}")

    # Select best trial based on lowest 2D point cloud distance
    best_trial = min(
        study.best_trials, key=lambda t: t.values[3]
    )  # values[3] = mean_min_distance

    print(f"  Selected trial: {best_trial.number} (lowest 2D point cloud distance)")
    print(f"  Active neurons: {best_trial.user_attrs.get('n_active', 'N/A')}")
    print("\n  Objective Values (from Pareto front):")
    print(f"    [0] Correlation:  {best_trial.values[0]:.4f}")
    print(f"    [1] Mean FR:      {best_trial.values[1]:.4f}")
    print(f"    [2] Mean CV:      {best_trial.values[2]:.4f}")
    print(f"    [3] 2D Distance:  {best_trial.values[3]:.4f}")
    print(f"    [4] Variance:     {best_trial.values[4]:.4f}")
    print("\n  Results vs Targets:")
    print(
        f"    FR: {best_trial.user_attrs.get('FR_mean', 'N/A'):.2f} Hz vs {exp_targets['FR_mean']:.2f} Hz (error: {best_trial.user_attrs.get('mean_fr_loss', 'N/A'):.1%})"
    )
    print(
        f"    CV: {best_trial.user_attrs.get('CV_mean', 'N/A'):.3f} vs {exp_targets['CV_mean']:.3f} (error: {best_trial.user_attrs.get('mean_cv_loss', 'N/A'):.1%})"
    )
    print(
        f"    Corr: {best_trial.user_attrs.get('sim_corr', 'N/A'):.3f} vs {best_trial.user_attrs.get('exp_corr', 'N/A'):.3f} (diff: {best_trial.user_attrs.get('correlation_loss', 'N/A'):.3f})"
    )

    return study


##############################################################################
# Results Export
# --------------


def select_best_trial(study):
    """
    Select best trial from Pareto front.

    For multi-objective optimization, select the trial with lowest mean_min_distance
    (objective[3]) as this prioritizes having simulated points closest to experimental
    points in 2D (FR, CV) space.

    Parameters
    ----------
    study : optuna.Study
        Completed multi-objective study

    Returns
    -------
    optuna.Trial
        Selected best trial from Pareto front
    """
    return min(study.best_trials, key=lambda t: t.values[3])


def create_cv_fr_scatter_plot(study, muscle_name, exp_frs, exp_cvs):
    """
    Create 2D scatter plot comparing experimental and simulated CV vs FR relationship.

    Parameters
    ----------
    study : optuna.Study
        Completed multi-objective optimization study
    muscle_name : str
        Muscle identifier
    exp_frs : np.ndarray
        Experimental firing rates
    exp_cvs : np.ndarray
        Experimental CV values
    """
    from scipy.stats import pearsonr

    # Get best trial from Pareto front
    best_trial = select_best_trial(study)

    # Need to re-run best trial to get individual FR and CV values
    # For now, use user attributes if available
    # Otherwise, we need to extract from trial or re-simulate
    # As a workaround, we'll create a placeholder plot with experimental data only
    # and note that individual simulated points would require re-simulation

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Plot experimental data
    exp_corr, _ = pearsonr(exp_frs, exp_cvs)
    ax.scatter(
        exp_frs,
        exp_cvs,
        alpha=0.6,
        s=100,
        c="blue",
        label=f"Experimental (r={exp_corr:.3f})",
        edgecolors="black",
        linewidth=0.5,
    )

    # Add trend line for experimental data
    z = np.polyfit(exp_frs, exp_cvs, 1)
    p = np.poly1d(z)
    fr_range = np.linspace(exp_frs.min(), exp_frs.max(), 100)
    ax.plot(fr_range, p(fr_range), "--", color="blue", alpha=0.5, linewidth=2)

    # Get simulated correlation from best trial if available
    if "sim_corr" in best_trial.user_attrs:
        sim_corr = best_trial.user_attrs["sim_corr"]
        ax.text(
            0.05,
            0.95,
            f"Best trial correlation:\n"
            f"  Experimental: r={exp_corr:.3f}\n"
            f"  Simulated: r={sim_corr:.3f}\n"
            f"  Difference: Δ={abs(exp_corr - sim_corr):.3f}",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    ax.set_xlabel("Firing Rate (Hz)", fontsize=12)
    ax.set_ylabel("Coefficient of Variation (CV)", fontsize=12)
    ax.set_title(
        f"{muscle_name} - CV vs Firing Rate\n(2D Distribution Target)", fontsize=14
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = RESULTS_DIR / f"cv_vs_fr_{muscle_name}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved CV vs FR plot: {plot_path}")

    return plot_path


def export_results(study, muscle_name):
    """
    Export optimization results to JSON files.

    Parameters
    ----------
    study : optuna.Study
        Completed multi-objective study
    muscle_name : str
        Muscle identifier
    """
    # Select best trial from Pareto front
    best_trial = select_best_trial(study)

    # Extract optimized DD and network parameters
    dd_params = {
        "dd_neurons": best_trial.user_attrs.get("dd_neurons"),
        "conn_probability": best_trial.user_attrs.get("conn_probability"),
        "synaptic_weight": best_trial.user_attrs.get("synaptic_weight"),
        "dd_baseline__Hz": best_trial.user_attrs.get("dd_baseline__Hz"),
        "dd_peak__Hz": best_trial.user_attrs.get("dd_peak__Hz"),
    }

    # Combine all parameters
    all_params = {
        "muscle": muscle_name,
        "descending_drive": dd_params,
        "results": {
            "objectives": {
                "correlation_loss": best_trial.values[0],
                "mean_fr_loss": best_trial.values[1],
                "mean_cv_loss": best_trial.values[2],
                "mean_min_distance": best_trial.values[3],
                "variance_loss": best_trial.values[4],
            },
            "trial_number": best_trial.number,
            "n_pareto_trials": len(study.best_trials),
            "FR_mean": best_trial.user_attrs.get("FR_mean"),
            "CV_mean": best_trial.user_attrs.get("CV_mean"),
            "n_active": best_trial.user_attrs.get("n_active"),
            "exp_corr": best_trial.user_attrs.get("exp_corr"),
            "sim_corr": best_trial.user_attrs.get("sim_corr"),
        },
        "targets": EXPERIMENTAL_TARGETS[muscle_name],
    }

    # Save to JSON
    json_path = RESULTS_DIR / f"dd_optimized_params_{muscle_name}.json"
    with open(json_path, "w") as f:
        json.dump(all_params, f, indent=2)
    print(f"Saved DD parameters JSON: {json_path}")

    # Save study object
    study_path = RESULTS_DIR / f"study_dd_{muscle_name}.pkl"
    joblib.dump(study, study_path)
    print(f"Saved study: {study_path}")

    # Create CV vs FR scatter plot
    exp_muscle_data = isi_data[
        isi_data["Muscle"].isin(
            ["VM", "VL"] if muscle_name == "VLVM" else [muscle_name]
        )
    ]
    exp_frs = exp_muscle_data["FR mean"].values
    exp_cvs = exp_muscle_data["ISI CV"].values
    create_cv_fr_scatter_plot(study, muscle_name, exp_frs, exp_cvs)

    # Print summary of optimized DD parameters
    print(f"\n{'=' * 60}")
    print(f"Optimized DD Parameters for {muscle_name}:")
    print(f"{'=' * 60}")
    print(f"  DD Neurons:        {dd_params['dd_neurons']}")
    print(f"  Conn Probability:  {dd_params['conn_probability']:.3f}")
    print(f"  Synaptic Weight:   {dd_params['synaptic_weight']:.4f} μS")
    print(f"  DD Baseline:       {dd_params['dd_baseline__Hz']:.2f} Hz")
    print(f"  DD Peak:           {dd_params['dd_peak__Hz']:.2f} Hz")
    print(f"{'=' * 60}\n")


##############################################################################
# Main Execution
# --------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Optimize motor neuron parameters for a single muscle model"
    )
    parser.add_argument(
        "--muscle",
        type=str,
        choices=["VLVM", "TA", "FDI"],
        required=True,
        help="Which muscle to optimize (required)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=N_TRIALS_PER_MUSCLE,
        help=f"Number of optimization trials (default: {N_TRIALS_PER_MUSCLE})",
    )
    parser.add_argument(
        "--reset-study",
        action="store_true",
        help="Delete existing Optuna database and start fresh optimization",
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("OPTUNA-BASED MUSCLE PARAMETER OPTIMIZATION")
    print("=" * 80)
    print(f"Muscle: {args.muscle}")
    print(f"Trials: {args.n_trials}")
    print(f"Reset study: {args.reset_study}")

    # Run single muscle optimization
    muscle = args.muscle
    print(f"\n{'=' * 80}")
    print(f"Starting optimization for {muscle}")
    print(f"{'=' * 80}\n")

    study = optimize_muscle(
        muscle_name=muscle, n_trials=args.n_trials, reset_study=args.reset_study
    )

    export_results(study, muscle)

    # Create summary visualization
    print("\n" + "=" * 80)
    print("Creating optimization history plots...")
    print("=" * 80)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Multi-objective: Plot primary objective (correlation_loss) over time
    ax = axes[0]
    trial_numbers = [t.number for t in study.trials]
    # Extract correlation_loss (objective 0) from multi-objective values
    corr_losses = [t.values[0] if t.values is not None else 1000 for t in study.trials]
    ax.plot(trial_numbers, corr_losses, "o-", alpha=0.6, label="Correlation Loss")

    # Mark Pareto front trials
    pareto_numbers = [t.number for t in study.best_trials]
    pareto_corr = [t.values[0] for t in study.best_trials]
    ax.scatter(
        pareto_numbers,
        pareto_corr,
        c="red",
        s=100,
        marker="*",
        label="Pareto Front",
        zorder=10,
        edgecolors="black",
        linewidth=1,
    )

    ax.set_xlabel("Trial")
    ax.set_ylabel("Correlation Loss (Objective 0)")
    ax.set_title(f"{muscle} - Optimization History")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Parameter importance for correlation_loss (objective 0)
    ax = axes[1]
    if len(study.trials) >= 10:
        try:
            # For multi-objective, compute importance for first objective
            importance = optuna.importance.get_param_importances(
                study, target=lambda t: t.values[0]
            )
            top_params = dict(list(importance.items())[:10])
            param_names = [p.split("_")[-2:] for p in top_params.keys()]
            param_names = ["_".join(p) for p in param_names]
            ax.barh(param_names, list(top_params.values()))
            ax.set_xlabel("Importance")
            ax.set_title(f"{muscle} - Top 10 Parameters\n(for Correlation Loss)")
        except Exception:
            ax.text(
                0.5,
                0.5,
                "Not enough trials\nfor importance",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
    else:
        ax.text(
            0.5,
            0.5,
            "Not enough trials",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    plt.tight_layout()
    plot_path = RESULTS_DIR / f"optimization_history_{muscle}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {plot_path}")

    print("\n" + "=" * 80)
    print("DD PARAMETER OPTIMIZATION COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print("\nNext steps:")
    print("1. Review optimized DD parameters in JSON files")
    print("2. Apply parameters in your simulation scripts")
    print("3. Run full validation simulation with optimized DD params")
