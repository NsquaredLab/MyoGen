"""Diagnostic: Verify NaPS and LCaInact channels are active in Jaxley."""
import sys
import numpy as np
import jax.numpy as jnp
import jaxley as jx
from io import StringIO
from contextlib import contextmanager

@contextmanager
def suppress_stdout():
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout

# Create a pool with use_jaxley_mech=True
from myogen import simulator
from myogen.simulator.jaxley.populations import AlphaMN__Pool

rt, _ = simulator.RecruitmentThresholds(N=3, recruitment_range__ratio=100, mode="fuglevand")

with suppress_stdout():
    pool = AlphaMN__Pool(
        recruitment_thresholds__array=rt,
        mode="active",
        use_jaxley_mech=True,
    )

cell_wrapper = pool[0]
cell = cell_wrapper.cell

# Check 1: What channels are registered on the cell?
print("=" * 60)
print("DIAGNOSTIC: Channel Verification")
print("=" * 60)

# Check cell's channel list
if hasattr(cell, 'channels'):
    print(f"\ncell.channels: {cell.channels}")
elif hasattr(cell, '_channels'):
    print(f"\ncell._channels: {cell._channels}")

# Check the branch's channels
branch0 = cell.branch(0)
if hasattr(branch0, 'channels'):
    print(f"branch(0).channels: {branch0.channels}")

# Check cell's parameters for NaPS and LCa
print("\n--- Cell parameters containing 'NaPS' ---")
if hasattr(cell, 'nodes'):
    for col in cell.nodes.columns:
        if 'NaPS' in col or 'Naps' in col or 'naps' in col:
            print(f"  {col}: {cell.nodes[col].values}")

print("\n--- Cell parameters containing 'LCa' ---")
if hasattr(cell, 'nodes'):
    for col in cell.nodes.columns:
        if 'LCa' in col or 'lca' in col:
            print(f"  {col}: {cell.nodes[col].values}")

print("\n--- Cell parameters containing 'MAHP' ---")
if hasattr(cell, 'nodes'):
    for col in cell.nodes.columns:
        if 'MAHP' in col or 'mahp' in col:
            print(f"  {col}: {cell.nodes[col].values}")

print("\n--- Cell parameters containing 'Na' (fast) ---")
if hasattr(cell, 'nodes'):
    for col in cell.nodes.columns:
        if col.startswith('Na_'):
            print(f"  {col}: {cell.nodes[col].values}")

print("\n--- ALL column names in cell.nodes ---")
if hasattr(cell, 'nodes'):
    print(f"  columns: {list(cell.nodes.columns)}")

# Check 2: Run a simulation and see if NaPS contributes current
print("\n" + "=" * 60)
print("DIAGNOSTIC: Simulation Test")
print("=" * 60)

# Run simulation with a subthreshold current
dt_ms = 0.025
t_max_ms = 100.0
n_points = int(t_max_ms / dt_ms)

# Test with moderate current
current = np.ones(n_points) * 10.0  # 10 nA constant

with suppress_stdout():
    cell.delete_recordings()
    cell.delete_stimuli()
    cell.set("v", -65.0)
    cell.init_states()
    cell.record("v")
    cell.stimulate(jnp.array(current))
    voltages = jx.integrate(cell, delta_t=dt_ms, t_max=t_max_ms)

v = np.array(voltages[0] if voltages.ndim == 2 else voltages)
print(f"\nWith 10 nA injection:")
print(f"  V min: {v.min():.2f} mV")
print(f"  V max: {v.max():.2f} mV")
print(f"  V final: {v[-1]:.2f} mV")
print(f"  Spike count: {np.sum((v[:-1] < 0) & (v[1:] >= 0))}")

# Check 3: Try to record NaPS current directly
print("\n--- Attempting to record NaPS current ---")
try:
    with suppress_stdout():
        cell.delete_recordings()
        cell.delete_stimuli()
        cell.set("v", -65.0)
        cell.init_states()
        cell.record("v")
        # Try recording NaPS states
        try:
            cell.record("NaPS_m")
            print("  Successfully registered recording for NaPS_m")
        except Exception as e:
            print(f"  Failed to record NaPS_m: {e}")
        try:
            cell.record("NaPS_s")
            print("  Successfully registered recording for NaPS_s")
        except Exception as e:
            print(f"  Failed to record NaPS_s: {e}")
        try:
            cell.record("LCa_m")
            print("  Successfully registered recording for LCa_m")
        except Exception as e:
            print(f"  Failed to record LCa_m: {e}")
        try:
            cell.record("MAHP_mca")
            print("  Successfully registered recording for MAHP_mca")
        except Exception as e:
            print(f"  Failed to record MAHP_mca: {e}")

        cell.stimulate(jnp.array(current))
        voltages = jx.integrate(cell, delta_t=dt_ms, t_max=t_max_ms)

    print(f"\n  Recorded traces shape: {voltages.shape}")
    if voltages.ndim == 2:
        for i in range(voltages.shape[0]):
            trace = np.array(voltages[i])
            print(f"  Trace {i}: min={trace.min():.6f}, max={trace.max():.6f}, final={trace[-1]:.6f}")
except Exception as e:
    print(f"  Recording failed: {e}")

# Check 4: Compare with a fresh cell WITHOUT NaPS/LCa
print("\n" + "=" * 60)
print("DIAGNOSTIC: Comparison — manually remove NaPS to see effect")
print("=" * 60)

# Set NaPS gbar to 0 and re-run
with suppress_stdout():
    cell.delete_recordings()
    cell.delete_stimuli()

    # Zero out NaPS and LCa
    try:
        cell.set("NaPS_gbar", 0.0)
        print("  Set NaPS_gbar = 0.0")
    except Exception as e:
        print(f"  Failed to set NaPS_gbar = 0: {e}")
    try:
        cell.set("LCa_gbar", 0.0)
        print("  Set LCa_gbar = 0.0")
    except Exception as e:
        print(f"  Failed to set LCa_gbar = 0: {e}")

    cell.set("v", -65.0)
    cell.init_states()
    cell.record("v")
    cell.stimulate(jnp.array(current))
    voltages_no_pic = jx.integrate(cell, delta_t=dt_ms, t_max=t_max_ms)

v_no_pic = np.array(voltages_no_pic[0] if voltages_no_pic.ndim == 2 else voltages_no_pic)
print(f"\nWithout NaPS/LCa (gbar=0):")
print(f"  V min: {v_no_pic.min():.2f} mV")
print(f"  V max: {v_no_pic.max():.2f} mV")
print(f"  V final: {v_no_pic[-1]:.2f} mV")
print(f"  Spike count: {np.sum((v_no_pic[:-1] < 0) & (v_no_pic[1:] >= 0))}")

print(f"\n  DIFFERENCE (with PIC - without PIC):")
print(f"  V max diff: {v.max() - v_no_pic.max():.4f} mV")
print(f"  V final diff: {v[-1] - v_no_pic[-1]:.4f} mV")

if abs(v.max() - v_no_pic.max()) < 0.001:
    print("\n  *** WARNING: NaPS/LCa channels have ZERO effect! ***")
    print("  Channels are likely NOT being simulated by Jaxley.")
else:
    print("\n  NaPS/LCa channels ARE having an effect.")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
