"""
Performance benchmark for Cython simulate_fiber_v2 implementation.

Compares execution time between Python and Cython versions across
different problem sizes to measure actual speedup.
"""

import time
import numpy as np
from myogen.simulator.core.emg.surface.simulate_fiber import simulate_fiber_v2
from myogen.simulator.core.emg.electrodes import SurfaceElectrodeArray

print("=" * 70)
print("Cython simulate_fiber_v2 Performance Benchmark")
print("=" * 70)

# Test configurations (increasing complexity)
test_configs = [
    {"name": "Small (Quick)", "N": 64, "M": 16, "grid": (2, 2)},
    {"name": "Medium (Typical)", "N": 128, "M": 32, "grid": (4, 4)},
    {"name": "Large (Realistic)", "N": 256, "M": 64, "grid": (8, 8)},
]

# Fixed parameters
Fs = 2.048  # kHz
v = 4.0  # m/s
r = 40.0  # mm
r_bone = 0.0  # mm
th_fat = 5.0  # mm
th_skin = 2.0  # mm
R = 10.0  # mm
L1 = 50.0  # mm
L2 = 50.0  # mm
zi = 0.0  # mm

# Conductivity parameters
sig_muscle_rho = 0.5  # S/m
sig_muscle_z = 0.1  # S/m
sig_fat = 0.04  # S/m
sig_skin = 0.2  # S/m
sig_bone = 0.0  # S/m

def benchmark_config(config, n_runs=3):
    """Benchmark a single configuration."""
    N = config["N"]
    M = config["M"]
    grid_rows, grid_cols = config["grid"]

    print(f"\n{config['name']} Configuration:")
    print(f"  N={N}, M={M}, Electrode Grid={grid_rows}x{grid_cols}")
    print(f"  Running {n_runs} iterations...")

    # Create electrode array
    electrode_array = SurfaceElectrodeArray(
        num_rows=grid_rows,
        num_cols=grid_cols,
        inter_electrode_distances__mm=5.0,
        electrode_radius__mm=1.0,
    )

    # Warm-up run (not timed)
    print("  Warming up...", end="", flush=True)
    _ = simulate_fiber_v2(
        Fs, v, N, M, r, r_bone, th_fat, th_skin, R, L1, L2, zi,
        electrode_array, sig_muscle_rho, sig_muscle_z, sig_fat, sig_skin,
        sig_bone, use_cython=True
    )
    print(" done")

    # Benchmark Python version
    print(f"  Benchmarking Python version...", end="", flush=True)
    python_times = []
    for i in range(n_runs):
        start = time.perf_counter()
        phi_py, A_py, B_py = simulate_fiber_v2(
            Fs, v, N, M, r, r_bone, th_fat, th_skin, R, L1, L2, zi,
            electrode_array, sig_muscle_rho, sig_muscle_z, sig_fat, sig_skin,
            sig_bone, use_cython=False
        )
        end = time.perf_counter()
        python_times.append(end - start)
        print(".", end="", flush=True)
    print(" done")

    # Benchmark Cython version
    print(f"  Benchmarking Cython version...", end="", flush=True)
    cython_times = []
    for i in range(n_runs):
        start = time.perf_counter()
        phi_cy, A_cy, B_cy = simulate_fiber_v2(
            Fs, v, N, M, r, r_bone, th_fat, th_skin, R, L1, L2, zi,
            electrode_array, sig_muscle_rho, sig_muscle_z, sig_fat, sig_skin,
            sig_bone, use_cython=True
        )
        end = time.perf_counter()
        cython_times.append(end - start)
        print(".", end="", flush=True)
    print(" done")

    # Calculate statistics
    py_mean = np.mean(python_times)
    py_std = np.std(python_times)
    cy_mean = np.mean(cython_times)
    cy_std = np.std(cython_times)
    speedup = py_mean / cy_mean

    # Verify numerical equivalence
    phi_diff = np.max(np.abs(phi_py - phi_cy))

    print(f"\n  Results:")
    print(f"    Python:  {py_mean:.3f}s ± {py_std:.3f}s")
    print(f"    Cython:  {cy_mean:.3f}s ± {cy_std:.3f}s")
    print(f"    Speedup: {speedup:.2f}x")
    print(f"    PHI max diff: {phi_diff:.2e}")

    return {
        "name": config["name"],
        "N": N,
        "M": M,
        "python_time": py_mean,
        "cython_time": cy_mean,
        "speedup": speedup,
        "phi_diff": phi_diff,
    }

# Run benchmarks
results = []
for config in test_configs:
    result = benchmark_config(config, n_runs=3)
    results.append(result)

# Print summary
print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print(f"{'Configuration':<20} {'Python (s)':<12} {'Cython (s)':<12} {'Speedup':<10}")
print("-" * 70)
for r in results:
    print(f"{r['name']:<20} {r['python_time']:>10.3f}  {r['cython_time']:>10.3f}  {r['speedup']:>8.2f}x")

avg_speedup = np.mean([r['speedup'] for r in results])
print("-" * 70)
print(f"{'Average Speedup:':<20} {avg_speedup:>42.2f}x")
print("=" * 70)

# Performance tier assessment
print("\nPerformance Assessment:")
if avg_speedup >= 10:
    print("  🎯 EXCELLENT: Exceeded 10x target speedup!")
elif avg_speedup >= 5:
    print("  ✅ SUCCESS: Met 5-10x target speedup range!")
elif avg_speedup >= 3:
    print("  ⚠️  ACCEPTABLE: Above minimum 3x threshold")
else:
    print("  ❌ BELOW TARGET: Less than 3x speedup")

print(f"\nFor a typical 10,000-fiber simulation:")
print(f"  Python:  ~{results[1]['python_time'] * 10000 / 3600:.1f} hours")
print(f"  Cython:  ~{results[1]['cython_time'] * 10000 / 3600:.1f} hours")
print(f"  Savings: ~{(results[1]['python_time'] - results[1]['cython_time']) * 10000 / 3600:.1f} hours")
print("=" * 70)
