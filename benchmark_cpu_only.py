"""
Benchmark CPU-only mode (no CuPy) to see if Cython helps when GPU isn't available.
"""

import time
import sys
import numpy as np

# Disable CuPy by removing it from sys.modules
if 'cupy' in sys.modules:
    del sys.modules['cupy']

# Mock cupy to force CPU mode
class MockCupy:
    def __getattr__(self, name):
        raise ImportError("CuPy mocked as unavailable")

sys.modules['cupy'] = MockCupy()

# Now import after cupy is mocked
from myogen.simulator.core.emg.surface.simulate_fiber import simulate_fiber_v2
from myogen.simulator.core.emg.electrodes import SurfaceElectrodeArray

print("=" * 70)
print("CPU-Only Performance Benchmark (No CuPy)")
print("=" * 70)

# Medium configuration
N, M = 128, 32
Fs, v = 2.048, 4.0
r, r_bone = 40.0, 0.0
th_fat, th_skin = 5.0, 2.0
R, L1, L2, zi = 10.0, 50.0, 50.0, 0.0

electrode_array = SurfaceElectrodeArray(
    num_rows=4,
    num_cols=4,
    inter_electrode_distances__mm=5.0,
    electrode_radius__mm=1.0,
)

sig_muscle_rho, sig_muscle_z = 0.5, 0.1
sig_fat, sig_skin, sig_bone = 0.04, 0.2, 0.0

print(f"\nConfiguration: N={N}, M={M}, Electrode Grid=4x4")
print("Running 5 iterations per version...")

# Warm-up
print("\nWarming up...", end="", flush=True)
try:
    _ = simulate_fiber_v2(
        Fs, v, N, M, r, r_bone, th_fat, th_skin, R, L1, L2, zi,
        electrode_array, sig_muscle_rho, sig_muscle_z, sig_fat, sig_skin,
        sig_bone, use_cython=False
    )
    print(" done")
except Exception as e:
    print(f" failed: {e}")
    sys.exit(1)

# Benchmark Python
print("\nBenchmarking Python (CPU-only)...", end="", flush=True)
python_times = []
for i in range(5):
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

# Benchmark Cython
print("Benchmarking Cython (CPU-only)...", end="", flush=True)
cython_times = []
for i in range(5):
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

py_mean = np.mean(python_times)
py_std = np.std(python_times)
cy_mean = np.mean(cython_times)
cy_std = np.std(cython_times)
speedup = py_mean / cy_mean

print("\n" + "=" * 70)
print("Results (CPU-Only Mode)")
print("=" * 70)
print(f"Python:  {py_mean:.3f}s ± {py_std:.3f}s")
print(f"Cython:  {cy_mean:.3f}s ± {cy_std:.3f}s")
print(f"Speedup: {speedup:.2f}x")
print("=" * 70)

if speedup >= 5:
    print("✅ Met 5x+ target in CPU-only mode!")
elif speedup >= 3:
    print("⚠️  Above 3x threshold")
elif speedup >= 1.5:
    print("⚠️  Modest speedup")
else:
    print("❌ No significant speedup")
