"""Compute force from optimized descending drive parameters."""

import os

os.environ["MPLBACKEND"] = "Agg"
if "DISPLAY" in os.environ:
    del os.environ["DISPLAY"]

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from neo import SpikeTrain, Segment, Block
import quantities as pq
from neuron import h

from examples.finetune.helper import calculate_firing_rate_statistics
from myogen import RANDOM_GENERATOR
from myogen.simulator import RecruitmentThresholds
from myogen.simulator.core.force.force_model import ForceModel
from myogen.simulator.neuron import Network
from myogen.simulator.neuron.populations import AlphaMN__Pool, DescendingDrive__Pool
from myogen.utils.nmodl import load_nmodl_mechanisms

warnings.filterwarnings("ignore")


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--study-prefix", type=str, default="VLVM_")
    p.add_argument("--simulation-time", type=float, default=10000.0)
    p.add_argument("--n-motor-units", type=int, default=100)
    return p.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    load_nmodl_mechanisms()
    h.secondorder = 2

    SIMULATION_TIME_MS = args.simulation_time
    TIMESTEP_MS = 0.1
    N_MOTOR_UNITS = args.n_motor_units
    STUDY_PREFIX = args.study_prefix
    SYNAPTIC_WEIGHT = 0.05

    RESULTS_DIR = Path("./results/dd_optimization")
    OUTPUT_DIR = Path("./results/force_validation")
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # Load optimized parameters
    params_file = RESULTS_DIR / f"{STUDY_PREFIX}dd_optimized_params.json"
    if not params_file.exists():
        raise FileNotFoundError(f"Optimized parameters not found: {params_file}")

    with open(params_file, "r") as f:
        results = json.load(f)

    dd_params = results["best_trial"]
    dd_neurons = dd_params["dd_neurons"]
    conn_probability = dd_params["conn_probability"]
    dd_drive__Hz = dd_params["dd_drive__Hz"]
    gamma_shape = dd_params["gamma_shape"]

    # Load Gfluctdv settings if present
    gfluctdv_enabled = results.get("input_parameters", {}).get("gfluctdv_enabled", False)
    gfluctdv_noise_amplitude = dd_params.get("gfluctdv_noise_amplitude", None)

    print(f"\n{STUDY_PREFIX}Force Validation")
    print(f"DD: {dd_neurons} neurons, conn_prob={conn_probability:.3f}, drive={dd_drive__Hz:.1f}Hz")
    if gfluctdv_enabled:
        print(f"Gfluctdv: ENABLED (noise amplitude={gfluctdv_noise_amplitude:.2e} S/cm²)")

    # Setup simulation
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

    # Apply Gfluctdv if it was enabled during DD optimization
    if gfluctdv_enabled and gfluctdv_noise_amplitude is not None:
        print("Applying Gfluctdv to motor neurons (matching DD optimization)...")
        for cell in motor_neuron_pool:
            cell.insert_Gfluctdv()
            for d in cell.dend:
                d.std_e_Gfluctdv = gfluctdv_noise_amplitude
                d.std_i_Gfluctdv = gfluctdv_noise_amplitude

    descending_drive_pool = DescendingDrive__Pool(
        n=dd_neurons,
        timestep__ms=TIMESTEP_MS,
        process_type="gamma",
        shape=float(gamma_shape),
    )

    network = Network({"DD": descending_drive_pool, "aMN": motor_neuron_pool})
    network.connect(
        source="DD",
        target="aMN",
        probability=conn_probability,
        weight__μS=SYNAPTIC_WEIGHT,
    )
    network.connect_from_external(source="cortical_input", target="DD", weight__μS=1.0)
    dd_netcons = network.get_netcons("cortical_input", "DD")

    # Setup recording
    mn_spike_recorders = []
    for cell in motor_neuron_pool:
        spike_recorder = h.Vector()
        nc = h.NetCon(cell.soma(0.5)._ref_v, None, sec=cell.soma)
        nc.threshold = 50
        nc.record(spike_recorder)
        mn_spike_recorders.append(spike_recorder)

    # Create drive signal
    time_points = int(SIMULATION_TIME_MS / TIMESTEP_MS)
    drive_signal = np.ones(time_points) * dd_drive__Hz + np.clip(
        RANDOM_GENERATOR.normal(0, 1.0, size=time_points), 0, None
    )

    # Initialize and run simulation
    h.load_file("stdrun.hoc")
    h.dt = TIMESTEP_MS
    h.tstop = SIMULATION_TIME_MS

    for section, voltage in zip(*motor_neuron_pool.get_initialization_data()):
        section.v = voltage
    for section, voltage in zip(*descending_drive_pool.get_initialization_data()):
        section.v = voltage

    h.finitialize()

    print("Running simulation...")
    step_counter = 0
    while h.t < h.tstop:
        current_drive = drive_signal[min(step_counter, len(drive_signal) - 1)]
        for dd_cell in descending_drive_pool:
            if dd_cell.integrate(current_drive):
                if h.t < h.tstop:
                    dd_netcons[dd_cell.pool__ID].event(h.t + 1)
        h.fadvance()
        step_counter += 1

    # Process spike trains
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

    spike_train__Block = Block(name="Motor Unit Pool")
    spike_train__Block.segments = [mn_segment]

    # Calculate firing rate statistics
    stats = calculate_firing_rate_statistics(mn_segment.spiketrains)
    fr_mean = stats["FR_mean"]
    fr_std = stats["FR_std"]
    n_active = stats["n_active"]

    print(f"Firing rate: {fr_mean:.1f}±{fr_std:.1f} Hz ({n_active}/{N_MOTOR_UNITS} active)")

    # Generate force
    force_model = ForceModel(
        recruitment_thresholds=recruitment_thresholds,
        recording_frequency__Hz=2048,
        longest_duration_rise_time__ms=90.0,
        contraction_time_range_factor=3,
    )

    force_output = force_model.generate_force(spike_train__Block=spike_train__Block)
    force_signal = force_output.magnitude[:, 0]

    # Calculate force statistics (steady-state: last 50%)
    steady_idx = len(force_signal) // 2
    force_mean = np.mean(force_signal[steady_idx:])
    force_std = np.std(force_signal[steady_idx:])

    print(
        f"Force (steady-state): {force_mean:.4f}±{force_std:.4f} a.u. (CoV={force_std / force_mean:.3f})"
    )

    # Save results
    force_results = {
        "dd_parameters": dd_params,
        "gfluctdv_enabled": gfluctdv_enabled,
        "firing_rate": {
            "mean__Hz": float(fr_mean),
            "std__Hz": float(fr_std),
            "n_active": n_active,
        },
        "force": {
            "mean__au": float(force_mean),
            "std__au": float(force_std),
            "cov": float(force_std / force_mean),
        },
    }

    results_file = OUTPUT_DIR / f"{STUDY_PREFIX}force_results.json"
    with open(results_file, "w") as f:
        json.dump(force_results, f, indent=2)

    print(f"Saved: {results_file}\n")


if __name__ == "__main__":
    main()
