"""
Multiprocessing-Parallelized Motor Unit Simulations
===================================================

Demonstrates using Python's multiprocessing to run independent motor neuron
pool simulations in parallel. Each process simulates a different motor unit
subset or experimental condition.

This approach is ideal for:
- Parameter sweeps (different recruitment thresholds, input patterns, etc.)
- Multiple muscle simulations
- Independent trials for statistics
- Embarrassingly parallel workloads

Note: This uses process-based parallelism where each process has its own
NEURON instance, avoiding serialization issues.
"""

from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
from neo import Block, Segment, SpikeTrain
import quantities as pq

from myogen import load_nmodl_mechanisms, set_random_seed
from myogen.simulator import generate_mu_recruitment_thresholds
from myogen.simulator.neuron.populations import AlphaMN__Pool
from myogen.utils.currents import create_trapezoid_current
from myogen.utils.neuron.inject_currents_into_populations import (
    inject_currents_and_simulate_spike_trains,
)


def simulate_motor_pool_worker(args: Tuple[int, dict]) -> dict:
    """
    Worker function that runs in separate process.

    Each process creates its own NEURON instance and runs an independent simulation.
    This avoids serialization issues with NEURON's hoc objects.

    Parameters
    ----------
    args : Tuple[int, dict]
        (worker_id, simulation_params) where simulation_params contains:
        - n_motor_units: Number of motor units
        - recruitment_method: Method for recruitment thresholds
        - input_amplitude__nA: Current amplitude
        - duration__ms: Simulation duration
        - seed_offset: Random seed offset for this worker

    Returns
    -------
    dict
        Results containing spike trains, firing rates, and metadata
    """
    worker_id, params = args

    # Set unique random seed for this worker
    set_random_seed(42 + params.get("seed_offset", worker_id))

    # Load NMODL mechanisms (each process needs this)
    load_nmodl_mechanisms(quiet=True)

    print(
        f"Worker {worker_id}: Starting simulation with {params['n_motor_units']} motor units"
    )

    # Generate recruitment thresholds
    recruitment_thresholds = generate_mu_recruitment_thresholds(
        n_motor_units=params["n_motor_units"],
        method=params.get("recruitment_method", "fuglevand"),
        rr=params.get("rr", 30.0),
    )

    # Create motor neuron pool
    motor_pool = AlphaMN__Pool(
        recruitment_thresholds__array=recruitment_thresholds,
        config_file=params.get("config_file", "alpha_mn_FDI.yaml"),
    )

    # Create input current
    timestep__ms = params.get("timestep__ms", 0.025)
    duration__ms = params.get("duration__ms", 5000.0)

    input_current__matrix = create_trapezoid_current(
        n_motor_units=params["n_motor_units"],
        timestep__ms=timestep__ms,
        duration__ms=duration__ms,
        amplitude__nA=params.get("input_amplitude__nA", 10.0),
        rise_time__ms=params.get("rise_time__ms", 500.0),
        plateau_time__ms=params.get("plateau_time__ms", 3000.0),
        fall_time__ms=params.get("fall_time__ms", 500.0),
    )

    # Run simulation using utility function
    spike_trains_neo = inject_currents_and_simulate_spike_trains(
        populations=[motor_pool],
        input_current__matrix=input_current__matrix,
        timestep__ms=timestep__ms,
        duration__ms=duration__ms,
    )

    # Convert Neo spike trains to simple numpy arrays for serialization
    spike_trains_serializable = [
        {
            "times": np.array(st.times.magnitude),
            "units": str(st.times.units),
            "t_start": float(st.t_start.magnitude),
            "t_stop": float(st.t_stop.magnitude),
        }
        for st in spike_trains_neo[0]  # First (and only) segment
    ]

    # Calculate firing rates
    firing_rates = []
    for st_dict in spike_trains_serializable:
        if len(st_dict["times"]) > 0:
            duration_s = (st_dict["t_stop"] - st_dict["t_start"]) / 1000.0
            firing_rate = len(st_dict["times"]) / duration_s
        else:
            firing_rate = 0.0
        firing_rates.append(firing_rate)

    # Calculate statistics
    active_units = sum(1 for fr in firing_rates if fr > 0)
    mean_fr = (
        np.mean([fr for fr in firing_rates if fr > 0]) if active_units > 0 else 0.0
    )

    print(
        f"Worker {worker_id}: Complete! Active units: {active_units}/{params['n_motor_units']}, "
        f"Mean FR: {mean_fr:.2f} Hz"
    )

    return {
        "worker_id": worker_id,
        "spike_trains": spike_trains_serializable,
        "firing_rates": firing_rates,
        "n_active": active_units,
        "mean_firing_rate__Hz": mean_fr,
        "std_firing_rate__Hz": np.std([fr for fr in firing_rates if fr > 0]),
        "params": params,
    }


def run_parallel_parameter_sweep():
    """
    Example 1: Parameter sweep across different current amplitudes.

    Simulates motor pools with varying input currents to explore
    recruitment and firing rate relationships.
    """
    print("=" * 80)
    print("Example 1: Parameter Sweep - Current Amplitudes")
    print("=" * 80)

    # Define parameter sweep
    current_amplitudes = [5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0]

    # Create worker arguments
    worker_args = [
        (
            i,
            {
                "n_motor_units": 60,
                "recruitment_method": "fuglevand",
                "rr": 30.0,
                "input_amplitude__nA": amplitude,
                "duration__ms": 3000.0,
                "rise_time__ms": 500.0,
                "plateau_time__ms": 2000.0,
                "fall_time__ms": 500.0,
                "timestep__ms": 0.025,
                "seed_offset": i * 1000,
            },
        )
        for i, amplitude in enumerate(current_amplitudes)
    ]

    # Run in parallel
    n_processes = min(cpu_count(), len(worker_args))
    print(f"\nRunning {len(worker_args)} simulations on {n_processes} processes...")

    with Pool(processes=n_processes) as pool:
        results = pool.map(simulate_motor_pool_worker, worker_args)

    # Save results
    save_path = Path("./results/parallel")
    save_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(results, save_path / "parameter_sweep_results.pkl")

    print(f"\n✓ Results saved to {save_path / 'parameter_sweep_results.pkl'}")

    # Print summary
    print("\nSummary:")
    print(f"{'Current (nA)':<15} {'Active Units':<15} {'Mean FR (Hz)':<15}")
    print("-" * 45)
    for result in results:
        amplitude = result["params"]["input_amplitude__nA"]
        print(
            f"{amplitude:<15.1f} {result['n_active']:<15d} {result['mean_firing_rate__Hz']:<15.2f}"
        )

    return results


def run_parallel_multiple_trials():
    """
    Example 2: Multiple independent trials with same parameters.

    Runs multiple trials to gather statistics on motor pool behavior
    with stochastic dynamics.
    """
    print("\n" + "=" * 80)
    print("Example 2: Multiple Trials - Statistical Analysis")
    print("=" * 80)

    n_trials = 8

    # Same parameters, different seeds
    worker_args = [
        (
            i,
            {
                "n_motor_units": 80,
                "recruitment_method": "fuglevand",
                "rr": 30.0,
                "input_amplitude__nA": 12.0,
                "duration__ms": 4000.0,
                "rise_time__ms": 500.0,
                "plateau_time__ms": 3000.0,
                "fall_time__ms": 500.0,
                "timestep__ms": 0.025,
                "seed_offset": i * 1000,
            },
        )
        for i in range(n_trials)
    ]

    # Run in parallel
    n_processes = min(cpu_count(), n_trials)
    print(f"\nRunning {n_trials} trials on {n_processes} processes...")

    with Pool(processes=n_processes) as pool:
        results = pool.map(simulate_motor_pool_worker, worker_args)

    # Calculate cross-trial statistics
    mean_frs = [r["mean_firing_rate__Hz"] for r in results]
    n_actives = [r["n_active"] for r in results]

    print(f"\nCross-Trial Statistics:")
    print(f"Mean firing rate: {np.mean(mean_frs):.2f} ± {np.std(mean_frs):.2f} Hz")
    print(f"Active units: {np.mean(n_actives):.1f} ± {np.std(n_actives):.1f}")

    # Save results
    save_path = Path("./results/parallel")
    save_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(results, save_path / "multiple_trials_results.pkl")

    print(f"\n✓ Results saved to {save_path / 'multiple_trials_results.pkl'}")

    return results


def run_parallel_different_muscles():
    """
    Example 3: Simulate different muscle types in parallel.

    Each worker simulates a different muscle configuration,
    useful for comparative studies.
    """
    print("\n" + "=" * 80)
    print("Example 3: Multiple Muscles - Comparative Analysis")
    print("=" * 80)

    # Define different muscle configurations
    muscle_configs = [
        {
            "name": "FDI_small",
            "n_motor_units": 120,
            "config_file": "alpha_mn_FDI.yaml",
            "rr": 30.0,
        },
        {
            "name": "FDI_medium",
            "n_motor_units": 120,
            "config_file": "alpha_mn_FDI.yaml",
            "rr": 50.0,
        },
        {
            "name": "TA_small",
            "n_motor_units": 200,
            "config_file": "alpha_mn_TA.yaml",
            "rr": 40.0,
        },
    ]

    # Create worker arguments
    worker_args = [
        (
            i,
            {
                "n_motor_units": config["n_motor_units"],
                "recruitment_method": "fuglevand",
                "rr": config["rr"],
                "config_file": config["config_file"],
                "input_amplitude__nA": 15.0,
                "duration__ms": 3000.0,
                "rise_time__ms": 500.0,
                "plateau_time__ms": 2000.0,
                "fall_time__ms": 500.0,
                "timestep__ms": 0.025,
                "seed_offset": i * 1000,
                "muscle_name": config["name"],
            },
        )
        for i, config in enumerate(muscle_configs)
    ]

    # Run in parallel
    n_processes = min(cpu_count(), len(worker_args))
    print(
        f"\nSimulating {len(muscle_configs)} muscle configurations on {n_processes} processes..."
    )

    with Pool(processes=n_processes) as pool:
        results = pool.map(simulate_motor_pool_worker, worker_args)

    # Print comparison
    print("\nMuscle Comparison:")
    print(f"{'Muscle':<20} {'Total MUs':<15} {'Active MUs':<15} {'Mean FR (Hz)':<15}")
    print("-" * 65)
    for result in results:
        muscle_name = result["params"].get("muscle_name", "Unknown")
        total_mus = result["params"]["n_motor_units"]
        print(
            f"{muscle_name:<20} {total_mus:<15d} {result['n_active']:<15d} "
            f"{result['mean_firing_rate__Hz']:<15.2f}"
        )

    # Save results
    save_path = Path("./results/parallel")
    save_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(results, save_path / "muscle_comparison_results.pkl")

    print(f"\n✓ Results saved to {save_path / 'muscle_comparison_results.pkl'}")

    return results


if __name__ == "__main__":
    print(f"System has {cpu_count()} CPU cores available")
    print()

    # Run examples
    results_sweep = run_parallel_parameter_sweep()
    results_trials = run_parallel_multiple_trials()
    results_muscles = run_parallel_different_muscles()

    print("\n" + "=" * 80)
    print("All parallel simulations complete!")
    print("=" * 80)
