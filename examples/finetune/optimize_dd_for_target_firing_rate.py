"""Multi-objective optimization of descending drive parameters."""

import os

os.environ["MPLBACKEND"] = "Agg"
if "DISPLAY" in os.environ:
    del os.environ["DISPLAY"]

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import optuna
import quantities as pq
from neo import Segment, SpikeTrain
from neuron import h
from scipy.stats import wasserstein_distance

from examples.finetune.helper import (
    calculate_firing_rate_statistics,
    get_gamma_shape_for_mvc,
)
from myogen import RANDOM_GENERATOR
from myogen.simulator import RecruitmentThresholds
from myogen.simulator.neuron import Network
from myogen.simulator.neuron.populations import AlphaMN__Pool, DescendingDrive__Pool
from myogen.utils.nmodl import load_nmodl_mechanisms

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--study-prefix", type=str, default="VLVM_")
    p.add_argument("--target-fr-mean", type=float, default=(16.31 + 17.33) / 2)
    p.add_argument("--target-fr-std", type=float, default=2.5)
    p.add_argument("--target-conn-prob", type=float, default=0.30)
    p.add_argument("--target-n-dd-neurons", type=int, default=400)
    p.add_argument("--n-trials", type=int, default=100)
    p.add_argument("--n-dd-neurons-min", type=int, default=100)
    p.add_argument("--n-dd-neurons-max", type=int, default=1000)
    p.add_argument("--n-motor-units", type=int, default=100)
    p.add_argument(
        "--gamma-shape",
        type=float,
        default=3.0,
        help="Fixed gamma shape parameter for descending drive variability",
    )
    # Gfluctdv (fluctuating conductance noise) parameters
    p.add_argument(
        "--enable-gfluctdv",
        action="store_true",
        help="Enable Gfluctdv (fluctuating conductance) noise mechanism for motor neurons",
    )
    p.add_argument(
        "--gfluctdv-noise-min",
        type=float,
        default=5e-6,
        help="Minimum Gfluctdv noise amplitude (S/cm²) for optimization search",
    )
    p.add_argument(
        "--gfluctdv-noise-max",
        type=float,
        default=3e-5,
        help="Maximum Gfluctdv noise amplitude (S/cm²) for optimization search",
    )
    return p.parse_args()


SIMULATION_TIME_MS = 3000.0
TIMESTEP_MS = 0.1
N_MOTOR_UNITS = 100
TARGET_FR_MEAN__HZ = (16.31 + 17.33) / 2
TARGET_FR_STD__HZ = 2.5
TARGET_CONN_PROB = 0.30
TARGET_N_DD_NEURONS = 400
N_TRIALS = 100
TIMEOUT_SECONDS = 3600
STUDY_PREFIX = "VLVM_"
N_DD_NEURONS_MIN = 100
N_DD_NEURONS_MAX = 1000
GAMMA_SHAPE = 3.0
SYNAPTIC_WEIGHT = 0.05
RESULTS_DIR = Path("./results/dd_optimization")
# Gfluctdv parameters
ENABLE_GFLUCTDV = False
GFLUCTDV_NOISE_MIN = 5e-6
GFLUCTDV_NOISE_MAX = 3e-5


def objective(trial):
    """Optuna single-objective optimization function.

    Optimizes network parameters (dd_neurons, conn_prob, dd_drive) to match
    target firing rate statistics while using fixed gamma shape input.
    """
    try:
        dd_neurons = trial.suggest_int("dd_neurons", N_DD_NEURONS_MIN, N_DD_NEURONS_MAX)
        conn_probability = trial.suggest_float("conn_prob", 0.1, 1.0)
        dd_drive__Hz = trial.suggest_float("dd_drive", 5.0, 1000.0)

        # Use fixed gamma shape (input parameter, not optimized)
        gamma_shape = get_gamma_shape_for_mvc(100, mvc_shape_value=GAMMA_SHAPE)

        # Gfluctdv noise amplitude (if enabled)
        if ENABLE_GFLUCTDV:
            gfluctdv_noise_amplitude = trial.suggest_float(
                "gfluctdv_noise_amplitude", GFLUCTDV_NOISE_MIN, GFLUCTDV_NOISE_MAX
            )
        else:
            gfluctdv_noise_amplitude = 0.0

        recruitment_thresholds, _ = RecruitmentThresholds(
            N=N_MOTOR_UNITS,
            recruitment_range__ratio=100,
            deluca__slope=5,
            konstantin__max_threshold__ratio=1.0,
            mode="combined",
        )

        motor_neuron_pool = AlphaMN__Pool(
            recruitment_thresholds__array=recruitment_thresholds,
            config_file="alpha_mn_default.yaml",
        )

        # Apply Gfluctdv noise if enabled
        if ENABLE_GFLUCTDV:
            for cell in motor_neuron_pool:
                cell.insert_Gfluctdv()
                for d in cell.dend:
                    # Set noise amplitude (std_e and std_i)
                    d.std_e_Gfluctdv = gfluctdv_noise_amplitude
                    d.std_i_Gfluctdv = gfluctdv_noise_amplitude
                    # Keep default correlation times (tau_e = tau_i = 20 ms)

        descending_drive_pool = DescendingDrive__Pool(
            n=dd_neurons,
            timestep__ms=TIMESTEP_MS,
            process_type="gamma",
            shape=gamma_shape,  # type: ignore
        )

        network = Network({"DD": descending_drive_pool, "aMN": motor_neuron_pool})
        network.connect(
            source="DD",
            target="aMN",
            probability=conn_probability,
            weight__μS=SYNAPTIC_WEIGHT,
        )
        network.connect_from_external(
            source="cortical_input", target="DD", weight__μS=1.0
        )
        dd_netcons = network.get_netcons("cortical_input", "DD")
        mn_spike_recorders = []
        for cell in motor_neuron_pool:
            spike_recorder = h.Vector()
            nc = h.NetCon(cell.soma(0.5)._ref_v, None, sec=cell.soma)
            nc.threshold = 50
            nc.record(spike_recorder)
            mn_spike_recorders.append(spike_recorder)

        time_points = int(SIMULATION_TIME_MS / TIMESTEP_MS)
        drive_signal = np.ones(time_points) * dd_drive__Hz + np.clip(
            RANDOM_GENERATOR.normal(0, 1.0, size=time_points), 0, None
        )

        h.load_file("stdrun.hoc")
        h.dt = TIMESTEP_MS
        h.tstop = SIMULATION_TIME_MS

        for section, voltage in zip(*motor_neuron_pool.get_initialization_data()):
            section.v = voltage
        for section, voltage in zip(*descending_drive_pool.get_initialization_data()):
            section.v = voltage

        h.finitialize()

        step_counter = 0
        while h.t < h.tstop:
            current_drive = drive_signal[min(step_counter, len(drive_signal) - 1)]
            for dd_cell in descending_drive_pool:
                if dd_cell.integrate(current_drive):
                    if h.t < h.tstop:
                        dd_netcons[dd_cell.pool__ID].event(h.t + 1)
            h.fadvance()
            step_counter += 1

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

        stats = calculate_firing_rate_statistics(mn_segment.spiketrains)
        n_active = stats["n_active"]
        if n_active < 10:
            return 1000.0

        fr_mean, fr_std = stats["FR_mean"], stats["FR_std"]

        # Simple normalized errors (just mean and std)
        mean_error = abs(fr_mean - TARGET_FR_MEAN__HZ) / TARGET_FR_MEAN__HZ
        std_error = abs(fr_std - TARGET_FR_STD__HZ) / TARGET_FR_STD__HZ

        # Soft biological plausibility penalties
        plausibility_penalty = 0.0
        # Penalize very high connectivity (> 0.7 is less realistic)
        if conn_probability > 0.7:
            plausibility_penalty += 0.1 * (conn_probability - 0.7)
        # Penalize excessive neurons (> 800 is less common)
        if dd_neurons > 800:
            plausibility_penalty += 0.1 * ((dd_neurons - 800) / 200)

        # Single objective: match target FR stats + prefer realistic parameters
        objective_value = mean_error + std_error + plausibility_penalty

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

        if trial.number % 1 == 0:
            print(
                f"Trial {trial.number}: FR={fr_mean:.1f}±{fr_std:.1f}Hz, "
                f"obj={objective_value:.3f} (mean_err={mean_error:.3f}, std_err={std_error:.3f})"
            )

        return objective_value

    except Exception:
        return 1000.0


def main():
    """Main execution function."""
    global N_MOTOR_UNITS, TARGET_FR_MEAN__HZ, TARGET_FR_STD__HZ, TARGET_CONN_PROB
    global \
        TARGET_N_DD_NEURONS, \
        N_TRIALS, \
        STUDY_PREFIX, \
        N_DD_NEURONS_MIN, \
        N_DD_NEURONS_MAX, \
        GAMMA_SHAPE, \
        RESULTS_DIR, \
        ENABLE_GFLUCTDV, \
        GFLUCTDV_NOISE_MIN, \
        GFLUCTDV_NOISE_MAX

    args = parse_args()
    load_nmodl_mechanisms()
    h.secondorder = 2

    N_MOTOR_UNITS = args.n_motor_units
    TARGET_FR_MEAN__HZ = args.target_fr_mean
    TARGET_FR_STD__HZ = args.target_fr_std
    TARGET_CONN_PROB = args.target_conn_prob
    TARGET_N_DD_NEURONS = args.target_n_dd_neurons
    N_TRIALS = args.n_trials
    STUDY_PREFIX = args.study_prefix
    N_DD_NEURONS_MIN = args.n_dd_neurons_min
    N_DD_NEURONS_MAX = args.n_dd_neurons_max
    GAMMA_SHAPE = args.gamma_shape
    ENABLE_GFLUCTDV = args.enable_gfluctdv
    GFLUCTDV_NOISE_MIN = args.gfluctdv_noise_min
    GFLUCTDV_NOISE_MAX = args.gfluctdv_noise_max
    RESULTS_DIR = Path("./results/dd_optimization")
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)

    gfluctdv_status = "ENABLED" if ENABLE_GFLUCTDV else "DISABLED"
    print(
        f"\nOptimizing: {STUDY_PREFIX} | Target FR: {TARGET_FR_MEAN__HZ:.1f}±{TARGET_FR_STD__HZ:.1f}Hz | "
        f"Trials: {N_TRIALS} | Gfluctdv: {gfluctdv_status}\n"
    )

    storage_name = f"sqlite:///{RESULTS_DIR}/{STUDY_PREFIX}optuna_dd_optimization.db"
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
        study_name=f"{STUDY_PREFIX}dd_optimization",
        storage=storage_name,
        load_if_exists=True,
    )

    study.optimize(
        objective, n_trials=N_TRIALS, timeout=TIMEOUT_SECONDS, show_progress_bar=True
    )

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

    def trial_to_dict(t):
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
        # Add Gfluctdv parameters if enabled
        if ENABLE_GFLUCTDV:
            base_params["gfluctdv_noise_amplitude"] = t.user_attrs.get(
                "gfluctdv_noise_amplitude"
            )
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

    json_path = RESULTS_DIR / f"{STUDY_PREFIX}dd_optimized_params.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    joblib.dump(study, RESULTS_DIR / f"{STUDY_PREFIX}study.pkl")
    print(f"Saved: {json_path}\n")


if __name__ == "__main__":
    main()
