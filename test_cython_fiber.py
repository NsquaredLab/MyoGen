"""
Quick validation test for Cython simulate_fiber_v2 implementation.

This script compares the Cython and Python implementations to ensure
they produce identical numerical results.
"""

import numpy as np
from myogen.simulator.core.emg.surface.simulate_fiber import simulate_fiber_v2
from myogen.simulator.core.emg.electrodes import SurfaceElectrodeArray

print("=" * 70)
print("Cython simulate_fiber_v2 Validation Test")
print("=" * 70)

# Set up test parameters (small problem for quick testing)
Fs = 2.048  # kHz
v = 4.0  # m/s
N = 128  # Reduced for faster testing
M = 32   # Reduced for faster testing
r = 40.0  # mm
r_bone = 0.0  # mm (no bone for simplicity)
th_fat = 5.0  # mm
th_skin = 2.0  # mm
R = 10.0  # mm
L1 = 50.0  # mm
L2 = 50.0  # mm
zi = 0.0  # mm

# Create electrode array
electrode_array = SurfaceElectrodeArray(
    num_rows=4,
    num_cols=4,
    inter_electrode_distances__mm=5.0,
    electrode_radius__mm=1.0,
)

# Conductivity parameters
sig_muscle_rho = 0.5  # S/m
sig_muscle_z = 0.1  # S/m
sig_fat = 0.04  # S/m
sig_skin = 0.2  # S/m
sig_bone = 0.0  # S/m

print(f"\nTest Parameters:")
print(f"  N={N}, M={M}, Electrodes={electrode_array.num_rows}x{electrode_array.num_cols}")
print(f"  Fs={Fs} kHz, v={v} m/s")
print(f"  r={r} mm, th_fat={th_fat} mm, th_skin={th_skin} mm")

# Run Python version
print(f"\n{'Running Python version...':<40}", end="", flush=True)
phi_py, A_py, B_py = simulate_fiber_v2(
    Fs, v, N, M, r, r_bone, th_fat, th_skin, R, L1, L2, zi,
    electrode_array, sig_muscle_rho, sig_muscle_z, sig_fat, sig_skin,
    sig_bone, use_cython=False
)
print("✓")

# Run Cython version
print(f"{'Running Cython version...':<40}", end="", flush=True)
phi_cy, A_cy, B_cy = simulate_fiber_v2(
    Fs, v, N, M, r, r_bone, th_fat, th_skin, R, L1, L2, zi,
    electrode_array, sig_muscle_rho, sig_muscle_z, sig_fat, sig_skin,
    sig_bone, use_cython=True
)
print("✓")

# Compare outputs
print(f"\n{'Output Comparison:':<40}")
print(f"{'='*70}")

# PHI comparison
phi_diff = np.max(np.abs(phi_py - phi_cy))
phi_rel_diff = phi_diff / (np.max(np.abs(phi_py)) + 1e-12)
print(f"  PHI shape: {phi_py.shape}")
print(f"  PHI max abs difference: {phi_diff:.2e}")
print(f"  PHI max rel difference: {phi_rel_diff:.2e}")

# A_matrix comparison
A_diff = np.max(np.abs(A_py - A_cy))
A_rel_diff = A_diff / (np.max(np.abs(A_py)) + 1e-12)
print(f"  A_matrix shape: {A_py.shape}")
print(f"  A_matrix max abs difference: {A_diff:.2e}")
print(f"  A_matrix max rel difference: {A_rel_diff:.2e}")

# B_incomplete comparison
B_diff = np.max(np.abs(B_py - B_cy))
B_rel_diff = B_diff / (np.max(np.abs(B_py)) + 1e-12)
print(f"  B_incomplete shape: {B_py.shape}")
print(f"  B_incomplete max abs difference: {B_diff:.2e}")
print(f"  B_incomplete max rel difference: {B_rel_diff:.2e}")

# Verdict
print(f"\n{'Validation Result:':<40}")
print(f"{'='*70}")

rtol = 1e-8
atol = 1e-10

phi_match = np.allclose(phi_py, phi_cy, rtol=rtol, atol=atol)
A_match = np.allclose(A_py, A_cy, rtol=rtol, atol=atol)
B_match = np.allclose(B_py, B_cy, rtol=rtol, atol=atol)

all_match = phi_match and A_match and B_match

if all_match:
    print("  ✅ SUCCESS: All outputs match within tolerance!")
    print(f"     (rtol={rtol}, atol={atol})")
else:
    print("  ❌ FAILURE: Outputs differ beyond tolerance!")
    if not phi_match:
        print("     - PHI mismatch")
    if not A_match:
        print("     - A_matrix mismatch")
    if not B_match:
        print("     - B_incomplete mismatch")

print(f"\n{'='*70}")
