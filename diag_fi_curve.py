"""Diagnostic: trace the smallest neuron's F-I curve point by point."""
import numpy as np
import jax.numpy as jnp
import jaxley as jx
from myogen.simulator.jaxley.populations import AlphaMN__Pool
from myogen import simulator

# Create a small pool — just need the smallest neuron (cell 0)
thresholds, _ = simulator.RecruitmentThresholds(
    N=5, recruitment_range__ratio=50, mode='combined', deluca__slope=5
)
pool = AlphaMN__Pool(
    recruitment_thresholds__array=thresholds, mode='active', use_jaxley_mech=True
)

cell_wrapper = pool[0]  # smallest neuron
cell = cell_wrapper.cell

dt_ms = 0.025
tstop_ms = 3000.0
t_max_ms = tstop_ms
n_steps = int(tstop_ms / dt_ms)
settle_samples = int(500.0 / dt_ms)

print(f"Smallest neuron (cell 0), threshold={thresholds[0]:.4f}")
print(f"Sweeping current from 0 to 20 nA in 1 nA steps...")
print(f"{'Current (nA)':>14} {'Spikes':>8} {'Rate (Hz)':>10} {'Max V (mV)':>12} {'Min V (mV)':>12}")
print("-" * 60)

for current_nA in np.arange(0.0, 21.0, 1.0):
    current = jnp.ones(n_steps) * current_nA

    cell.delete_recordings()
    cell.delete_stimuli()
    cell.set("v", -65.0)
    cell.init_states()
    cell.branch(0).loc(0.5).record("v")
    cell.branch(0).loc(0.5).stimulate(current)

    voltages = jx.integrate(cell, delta_t=dt_ms, t_max=t_max_ms)
    v = np.array(voltages[0] if voltages.ndim == 2 else voltages)

    # Analysis window (after 500ms settling)
    v_analysis = v[settle_samples:]

    # Count spikes
    crossings = np.where((v_analysis[:-1] < 0) & (v_analysis[1:] >= 0))[0]
    n_spikes = len(crossings)
    rate = (n_spikes / 2500.0) * 1000.0

    print(f"{current_nA:>14.1f} {n_spikes:>8d} {rate:>10.1f} {v_analysis.max():>12.1f} {v_analysis.min():>12.1f}")
