"""
Spike Train Generation with Descending Drive - Jaxley Backend
==============================================================

This example demonstrates **realistic spike train simulation** using **trapezoidal descending drive (DD)**
with biophysically realistic motor neurons using the Jaxley (JAX-based) simulator.

This is the Jaxley equivalent of ``03_simulate_spike_trains_descending_drive.py``
which uses NEURON. Both examples now use the SAME ``model="NERLab"`` motor-neuron
model — the production NEURON setup.

    - **DescendingDrive__Pool**: Poisson process neurons modeling cortical input
    - **AlphaMN__Pool**: NERLab motor neurons (soma + 1 isopotential dendrite,
      ``napp`` + ``caL`` channels), matching the production NEURON model
    - **Network**: Synaptic connections between DD and motor neuron populations
    - **Trapezoidal patterns**: Smooth, physiologically relevant input

.. important::
    **Descending Drive (DD)** refers to the cortical and subcortical neural pathways that provide
    voluntary motor commands to spinal motor neurons. This is more realistic than direct current
    injection because it models the actual synaptic input patterns from upper motor neurons.

.. note::
    **Voltage convention.** NERLab cells live in the *original 1952 HH frame*:
    V_rest ≈ 0 mV, ENa = +120 mV, EK = -10 mV, spike peaks ≈ +90 mV. Synaptic
    constants below (``e_syn``, ``v_rest``) and the spike-detection threshold
    are written in this frame.

.. note::
    **Biophysical Simulation**: This example uses actual Jaxley biophysical simulation with
    ``napp`` (Na fast + Na persistent + Kfast + Kslow + leak) on the soma and ``caL``
    (L-type Ca, no inactivation + leak) on the dendrite.

    Recruitment emerges naturally from biophysics (Henneman size principle):
    - Low threshold MNs: smaller soma, higher input resistance → easier to activate
    - High threshold MNs: larger soma, lower input resistance → harder to activate

    Each MN receives input from its own random subset of DD neurons (50% connection
    probability, matching the NEURON example), so input currents differ between MNs.
    Recruitment still emerges from biophysics.

.. note::
    **Implementation**: Uses ``build_init_and_step_fn`` + ``jax.lax.scan`` to
    compile the entire time loop into a single XLA kernel — the same architecture
    as ``11_simulate_spinal_network.py``.  DD spike times are pre-generated
    (open-loop), then fed as a sparse binary matrix into the scan.  Synaptic
    conductances are updated with an IIR recurrence inside the scan body; no dense
    current array is pre-allocated.

    **Approximations vs NEURON**: Driving force is evaluated at a fixed resting
    potential rather than the live membrane voltage.  For fully causal closed-loop
    simulation with real synaptic conductances see ``11_simulate_spinal_network.py``.
"""

# %%
##############################################################################
# Import Libraries
# ----------------

from pathlib import Path

import jax
import jax.numpy as jnp
import jaxley as jx
from jaxley.integrate import build_init_and_step_fn
import joblib
import numpy as np
import quantities as pq
from matplotlib import pyplot as plt
from neo import AnalogSignal, Block, Segment, SpikeTrain
from tqdm import tqdm

from myogen import RANDOM_GENERATOR
from myogen.simulator.jaxley.populations import AlphaMN__Pool, DescendingDrive__Pool
from myogen.utils.types import pps

plt.style.use("fivethirtyeight")


def mean_firing_rate(spiketrain):
    """Mean firing rate of a neo.SpikeTrain (replaces elephant.statistics.mean_firing_rate)."""
    return (len(spiketrain) / (spiketrain.t_stop - spiketrain.t_start)).rescale(pq.Hz)


def population_psth(spiketrains, bin_size):
    """Total spike counts per bin across spiketrains, plus bin left edges (s).

    Native replacement for elephant.statistics.time_histogram (output="counts").
    """
    t_start = min(st.t_start for st in spiketrains).rescale(pq.s).magnitude
    t_stop = max(st.t_stop for st in spiketrains).rescale(pq.s).magnitude
    bs = bin_size.rescale(pq.s).magnitude
    n_bins = int((t_stop - t_start) / bs)
    edges = t_start + np.arange(n_bins + 1) * bs
    spikes = np.concatenate([st.rescale(pq.s).magnitude for st in spiketrains])
    spikes = spikes[(spikes >= edges[0]) & (spikes < edges[-1])]  # drop right-edge spikes
    counts, _ = np.histogram(spikes, bins=edges)
    return counts, edges[:-1]


##############################################################################
# Create Populations
# ------------------
#
# Like the NEURON example, we create a **motor neuron pool** using the **AlphaMN__Pool** class
# and a **DescendingDrive__Pool** to represent the cortical input.
#
# .. note::
#     ``model="NERLab"`` is the default and matches the production NEURON model
#     exactly (soma + 1 isopotential dendrite, ``napp`` + ``caL`` channels).
#     ``use_jaxley_mech`` only affects the Powers2017 path and is ignored here.

save_path = Path("./results")
save_path.mkdir(exist_ok=True)

# Load recruitment thresholds from previous example
recruitment_thresholds = joblib.load(save_path / "thresholds.pkl")

# Create motor neuron pool — matches NEURON production (model="NERLab" default,
# gamma=0.2 default matches the NEURON config). The inverted recruitment we
# saw earlier wasn't a gamma issue — it was the IIR synaptic weight delivering
# ~6× too much current vs. NEURON's Exp2Syn. See base_synaptic_weight below.
motor_neuron_pool = AlphaMN__Pool(
    recruitment_thresholds__array=recruitment_thresholds,
    model="NERLab",
    mode="active",
)

timestep = 0.1 * pq.ms  # matches NEURON ex03; finer dt (0.025ms) only needed for ex02/ex10 F-I characterization.
dt_ms = float(timestep.rescale(pq.ms).magnitude)

# Create descending drive pool using MyoGen's Jaxley DescendingDrive__Pool
descending_drive_pool = DescendingDrive__Pool(
    n=100,
    poisson_batch_size=5,
    timestep__ms=timestep,
)

print(f"Created motor neuron pool with {motor_neuron_pool.n} neurons")
print(f"  Model: {motor_neuron_pool.model}  (soma + 1 isopotential dendrite, napp + caL)")
print(f"Created descending drive pool with {descending_drive_pool.n} neurons")
print(f"Recruitment thresholds range: {recruitment_thresholds.min():.2f} - {recruitment_thresholds.max():.2f} nA")
print(f"  Recruitment from biophysics (Henneman principle) - NO manual current scaling")

##############################################################################
# Generate Trapezoidal Drive Pattern
# -----------------------------------
#
# Create a **trapezoidal ramp contraction pattern** that represents realistic
# voluntary isometric contractions.
#
# Match the NEURON version timing to capture the full trapezoidal pattern
# including ramp-down and rest-after phases.

simulation_time = 15000 * pq.ms  # 15 seconds to match NEURON version
time_points = int(simulation_time / timestep)

# Trapezoidal parameters
dd_baseline__pps = 0.0 * pps
dd_peak__pps = 65 * pps

# Phase durations
ramp_up_duration = 500 * pq.ms
plateau_duration = 10000 * pq.ms
ramp_down_duration = 500 * pq.ms

# Rest periods
rest_before = 1000 * pq.ms
rest_after = 1000 * pq.ms

# Phase boundaries
trapezoid_start = rest_before
ramp_up_end = trapezoid_start + ramp_up_duration
plateau_end = ramp_up_end + plateau_duration
ramp_down_end = plateau_end + ramp_down_duration

# Create time array
time_array = np.linspace(0, simulation_time.magnitude, time_points) * pq.ms

# Initialize drive signal
trapezoid_drive = np.ones(time_points) * dd_baseline__pps

for i, t in enumerate(time_array):
    if t < trapezoid_start:
        trapezoid_drive[i] = dd_baseline__pps
    elif t < ramp_up_end:
        elapsed = t - trapezoid_start
        trapezoid_drive[i] = dd_baseline__pps + (dd_peak__pps - dd_baseline__pps) * (
            elapsed / ramp_up_duration
        )
    elif t < plateau_end:
        trapezoid_drive[i] = dd_peak__pps
    elif t < ramp_down_end:
        elapsed = t - plateau_end
        trapezoid_drive[i] = dd_peak__pps - (dd_peak__pps - dd_baseline__pps) * (
            elapsed / ramp_down_duration
        )
    else:
        trapezoid_drive[i] = dd_baseline__pps

# Add small noise for realism
trapezoid_drive = (
    trapezoid_drive + np.clip(RANDOM_GENERATOR.normal(0, 1.0, size=time_points), 0, None) * pps
)

# Create AnalogSignal
trapezoid_drive_signal = AnalogSignal(
    signal=trapezoid_drive, sampling_period=timestep.rescale(pq.s)
)

joblib.dump(trapezoid_drive_signal, save_path / "trapezoid_drive_pattern_jaxley.pkl")

print(f"\nTrapezoidal drive pattern ({simulation_time} simulation):")
print(f"\tRest before: 0 - {trapezoid_start} ({dd_baseline__pps})")
print(f"\tRamp up: {trapezoid_start} - {ramp_up_end}")
print(f"\tPlateau: {ramp_up_end} - {plateau_end} ({dd_peak__pps})")
print(f"\tRamp down: {plateau_end} - {ramp_down_end}")
print(f"\tRest after: {ramp_down_end} - {simulation_time}")

##############################################################################
# Setup Spike Recording
# ---------------------

# Manual spike tracking for DD neurons
dd_spike_times = [[] for _ in range(len(descending_drive_pool))]

# Store DD spike events for later MN processing
# Each entry: (dd_idx, spike_time_ms)
all_dd_spike_events = []

##############################################################################
# Run Simulation
# --------------
#
# Execute the Jaxley simulation with real-time injection of the trapezoidal drive pattern.
# Similar to the NEURON example, we:
# 1. Run DD neurons as Poisson processes driven by the trapezoidal signal
# 2. Collect DD spikes
# 3. Use DD spikes to drive MN activity through synaptic integration

dt_ms = float(timestep.rescale(pq.ms).magnitude)
simulation_time_ms = float(simulation_time.rescale(pq.ms).magnitude)

print(f"\nRunning simulation ({simulation_time_ms} ms, dt={dt_ms} ms)...")

# Simulation loop - DD neurons (Poisson processes)
step_counter = 0
current_time_ms = 0.0

with tqdm(
    total=float(simulation_time),
    desc="Running DD simulation",
    unit="ms",
    bar_format="{l_bar}{bar}| {n:.2f}/{total:.2f} ms [{elapsed}<{remaining}, {rate_fmt}]",
) as pbar:
    while current_time_ms < simulation_time_ms:
        current_drive = trapezoid_drive_signal[min(step_counter, len(trapezoid_drive_signal) - 1)]

        # Drive DD neurons with current input level
        for dd_cell in descending_drive_pool:
            if dd_cell.integrate(current_drive):
                # Record spike time for DD neuron
                dd_spike_times[dd_cell.pool__ID].append(current_time_ms)
                all_dd_spike_events.append((dd_cell.pool__ID, current_time_ms))

        # Progress simulation
        current_time_ms += dt_ms
        step_counter += 1
        pbar.update(float(timestep))

##############################################################################
# Simulate Motor Neuron Responses — lax.scan with Inline Conductance Updates
# ---------------------------------------------------------------------------
#
# Architecture mirrors ex11 (``build_init_and_step_fn`` + ``lax.scan``):
#
# 1. Convert pre-generated DD spike times → sparse binary matrix (n_steps, n_dd)
# 2. Build DD→MN connectivity matrix (n_mns, n_dd)
# 3. Build jx.Network, compile a single-step function via build_init_and_step_fn
# 4. lax.scan over all timesteps:
#      g[t] = alpha * g[t-1] + (conn @ dd_spikes[t]) * weight   [IIR update]
#      I[t] = g[t] * (E_rev - V_rest)
#      states[t+1] = step_fn(states[t], I[t])
# 5. Post-hoc spike detection from voltage output
#
# Why this is faster than jx.integrate with precomputed currents:
# - No all_i_syn array (n_mns × n_steps = 100 × 150k floats allocated/transferred)
# - lax.scan compiles the entire loop into one XLA kernel; no Python interpreter
#   overhead per step; JAX can fuse conductance update + step_fn across cells

n_dd  = len(descending_drive_pool)
n_mns = len(motor_neuron_pool)
n_steps = int(simulation_time_ms / dt_ms)

# Synaptic parameters — matched to the NEURON ex03 setup so the comparison at
# the bottom of this script is apples-to-apples.
#   NEURON: ``network.connect(source="DD", target="aMN", probability=0.5,
#                             weight__uS=0.15 * pq.uS)``
#   (see 03_simulate_spike_trains_descending_drive.py:202)
# Voltage constants are written in the NERLab (1952-HH) frame: V_rest ≈ 0 mV.
# The driving force magnitude (~70 mV) is preserved across frames, so the
# per-spike synaptic current is unchanged in physical units.
# IIR-vs-Exp2Syn equivalence factor — DO NOT match NEURON's NetCon weight directly.
#
# NEURON ex03 uses ``network.connect(..., weight__uS=0.15)`` with an Exp2Syn
# synapse: each presynaptic spike triggers a conductance that RISES with τ1
# (~0.5 ms) and DECAYS with τ2 (~2-5 ms). The peak conductance per spike is
# only ``weight × peak_factor``, where peak_factor ≈ 0.1-0.2 depending on
# τ1/τ2 ratio. So the effective per-cell steady current at the production drive
# (3250 Hz/cell × 0.15 × 0.005 × peak_factor × 70 mV) lands at ~15-25 nA —
# right inside the NERLab rheobase distribution.
#
# Our IIR ``g_new = α·g_old + weight × spikes`` has NO rise time: every spike
# contributes its full weight instantly, so without the peak-shaping factor
# we deliver ~6× more current than NEURON for the same weight. At weight=0.08
# the IIR delivered ~91 nA per cell, pushing every small/mid MN into a
# persistent-Na sub-threshold plateau at +39 mV (verified via the
# results/diag_v_traces.png diagnostic).
#
# Match NEURON's *effective* drive instead — set the IIR weight to
# ``NEURON_weight × peak_factor`` ≈ 0.15 × 0.085 ≈ 0.013 µS. With this the
# scan delivers ex02-equivalent ~15 nA per cell, restoring Henneman recruitment.
base_synaptic_weight = 0.009  # µS — IIR/Exp2Syn equivalent of NEURON's 0.15 µS
                              # (refined down from 0.013: w=0.013 gave 98/100 active
                              # at 27 Hz; NEURON ref is 77/100 at ~18 Hz, ~1.5× lower)
tau_syn = 5.0                 # ms — synaptic time constant (decay)
e_syn   = 70.0                # mV — excitatory reversal in NERLab frame (modern 0 mV + 70)

# Pre-compute random DD→MN connectivity (50% probability, matches NEURON ex03)
DD_MN_CONNECTION_PROBABILITY = 0.5
dd_to_mn_connections = {
    mn_idx: [dd_idx for dd_idx in range(n_dd)
             if RANDOM_GENERATOR.random() < DD_MN_CONNECTION_PROBABILITY]
    for mn_idx in range(n_mns)
}

# Connectivity matrix (n_mns, n_dd) — binary, JAX float32
conn_mat_np = np.zeros((n_mns, n_dd), dtype=np.float32)
for mn_idx, dd_indices in dd_to_mn_connections.items():
    conn_mat_np[mn_idx, dd_indices] = 1.0
conn_mat = jnp.array(conn_mat_np)

# DD spike matrix (n_steps, n_dd) — binary float32
print("\nBuilding DD spike matrix...")
dd_spike_mat_np = np.zeros((n_steps, n_dd), dtype=np.float32)
for dd_idx, spikes in enumerate(dd_spike_times):
    if len(spikes) == 0:
        continue
    idxs = np.floor(np.array(spikes) / dt_ms).astype(int)
    idxs = idxs[idxs < n_steps]
    dd_spike_mat_np[idxs, dd_idx] = 1.0
dd_spike_mat = jnp.array(dd_spike_mat_np)  # scanned as xs in lax.scan

for mn_idx in range(min(3, n_mns)):
    print(f"  MN {mn_idx}: threshold={recruitment_thresholds[mn_idx]:.3f} nA, "
          f"n_DD={len(dd_to_mn_connections[mn_idx])}")

# Build jx.Network + step function
print(f"\nBuilding jx.Network from {n_mns} MN cells and compiling step function...")
mn_cells = []
for cw in motor_neuron_pool:
    cell = cw.cell
    cell.delete_recordings()
    cell.delete_stimuli()
    mn_cells.append(cell)

net = jx.Network(mn_cells)

# Register one recording + one stimulus slot per MN (establishes external_inds order)
placeholder = jnp.zeros(n_steps)
for mn_idx in range(n_mns):
    net.cell(mn_idx).branch(0).loc(0.5).record("v")
    net.cell(mn_idx).branch(0).loc(0.5).stimulate(placeholder)

# Explicitly initialise the WHOLE network at the NERLab resting voltage.
# AlphaMN__Pool already calls .set("v", 0)+init_states on each underlying
# cell at construction time, but wrapping cells in a fresh jx.Network can
# silently revert v to Jaxley's -70 mV default — see codex note. Set it
# here defensively so the next lines (to_jax + init_fn) capture the correct
# resting state. Verified by the print on the next line.
net.set("v", 0.0)
net.init_states()
print(f"  net resting V (first 5 nodes): {net.nodes['v'].head(5).tolist()}")

net.to_jax()
init_fn, step_fn = build_init_and_step_fn(net)
params        = net.get_parameters()
external_inds = net.external_inds.copy()
rec_inds      = jnp.array(net.recordings.rec_index.to_numpy(), dtype=jnp.int32)
states, params = init_fn(params)

# Scalar constants closed over in the scan body
alpha_syn = jnp.float32(np.exp(-dt_ms / tau_syn))
# Resting potential for the fixed-driving-force approximation, NERLab frame.
v_rest = 0.0
driving_force = jnp.float32(e_syn - v_rest)   # 70 mV in the NERLab frame

def scan_body(carry, dd_spikes_t):
    """Single timestep: update conductances, inject current at fixed driving force,
    advance network.

    Approximation: synaptic current is computed as ``g_syn × (e_syn - v_rest)``
    rather than ``g_syn × (e_syn - V_soma)``. Live-V evaluation looked attractive
    on paper (closer to a real conductance synapse) but, with the IIR
    conductance model used here, it produced a brief onset burst followed by
    silent depolarisation block: once V depolarised past ~+50 mV the driving
    force clamped to 0 and the cell never recovered. Fixed driving keeps the
    synaptic current flowing through the plateau, which is the regime we want
    to match NEURON ex03 in (sustained ~18 Hz firing across ~77 MNs).
    """
    jax_states, g_syn = carry

    # IIR conductance update for all MNs simultaneously
    g_new = alpha_syn * g_syn + (conn_mat @ dd_spikes_t) * base_synaptic_weight

    # Fixed driving force per MN: I [nA] = g [µS] × ΔV [mV]
    i_stim = g_new * driving_force  # shape (n_mns,)

    # Advance Jaxley network one step
    new_states = step_fn(jax_states, params, {"i": i_stim}, external_inds, delta_t=dt_ms)

    # Extract recorded voltages (shape: n_mns)
    v_t = new_states["v"][rec_inds]

    return (new_states, g_new), v_t

print(f"Running lax.scan over {n_steps} steps ({simulation_time_ms:.0f} ms at dt={dt_ms} ms)...")
print("  (First run triggers XLA compilation — subsequent runs are fast)")
init_carry = (states, jnp.zeros(n_mns, dtype=jnp.float32))
_, all_voltages = jax.jit(
    lambda c, xs: jax.lax.scan(scan_body, c, xs)
)(init_carry, dd_spike_mat)
jax.block_until_ready(all_voltages)
print("Scan complete.")
# all_voltages shape: (n_steps, n_mns)

# Post-hoc spike detection. NERLab APs cross +50 mV reliably on their way to
# the +90 mV peak; using +50 mV as the threshold (instead of 0 mV) avoids the
# false positives that the resting-state drift through 0 mV would produce in
# the Powers2017 frame.
SPIKE_DETECTION_THRESHOLD_MV = 50.0
v_np = np.array(all_voltages)  # (n_steps, n_mns)
mn_spike_times = [[] for _ in range(n_mns)]
for mn_idx in range(n_mns):
    v = v_np[:, mn_idx]
    spike_indices = np.where(
        (v[:-1] < SPIKE_DETECTION_THRESHOLD_MV) & (v[1:] >= SPIKE_DETECTION_THRESHOLD_MV)
    )[0]
    mn_spike_times[mn_idx] = list(spike_indices * dt_ms)
    if mn_idx < 3:
        print(f"  MN {mn_idx}: {len(mn_spike_times[mn_idx])} spikes detected")

# --- diagnostic: save full V(t) for 6 sampled cells across the size range
# (codex suggestion). Use this to spot whether large cells sit at a depolarised
# plateau (PIC runaway) or actually cycle through APs.
_diag_cells = [0, 25, 50, 75, 95, 99]
_diag_traces = {f"MN_{i}": v_np[:, i].astype(np.float32) for i in _diag_cells}
_diag_traces["__dt_ms"] = np.float32(dt_ms)
_diag_traces["__sim_ms"] = np.float32(simulation_time_ms)
joblib.dump(_diag_traces, save_path / "trapezoid_dd_voltage_traces_jaxley.pkl")
print(f"\nSaved V(t) diagnostic traces for cells {_diag_cells} "
      f"to results/trapezoid_dd_voltage_traces_jaxley.pkl")

active_mn_count = sum(1 for spikes in mn_spike_times if len(spikes) > 0)
print(f"\nAny-spike active count: {active_mn_count}/{n_mns}  "
      f"(misleading — counts cells that fired only at onset)")

# Plateau-based recruitment — only count cells with sustained firing during
# the constant-drive window (matches the NEURON comparison; suppresses
# transient-only spikes and lets us spot depolarisation-block silently).
_plateau_t0_ms = float(ramp_up_end.rescale(pq.ms).magnitude)
_plateau_t1_ms = float(plateau_end.rescale(pq.ms).magnitude)
_plateau_dur_s = (_plateau_t1_ms - _plateau_t0_ms) / 1000.0
plateau_spike_counts = [
    sum(1 for t in spikes if _plateau_t0_ms <= t < _plateau_t1_ms)
    for spikes in mn_spike_times
]
plateau_active_any   = sum(1 for c in plateau_spike_counts if c >= 1)
plateau_active_susta = sum(1 for c in plateau_spike_counts if c >= 5)   # sustained ≥ 0.5 Hz over plateau
plateau_total_spikes = sum(plateau_spike_counts)
print(f"Plateau recruitment ({_plateau_t0_ms:.0f}-{_plateau_t1_ms:.0f} ms):")
print(f"  any-spike-in-plateau  : {plateau_active_any}/{n_mns} MNs")
print(f"  ≥5 spikes in plateau  : {plateau_active_susta}/{n_mns} MNs  (target: ~77)")
print(f"  total plateau spikes  : {plateau_total_spikes}  (target: ~13800)")
print(f"  mean rate (sustained) : "
      f"{(plateau_total_spikes / max(plateau_active_susta, 1) / _plateau_dur_s):.1f} Hz/MN")

##############################################################################
# Convert Spike Data to Neo Format
# ---------------------------------

spike_train_block = Block(name="Trapezoidal DD Spike Trains - Jaxley Biophysical")

dd_segment = Segment(name="Descending Drive")
dd_segment.spiketrains = [
    SpikeTrain(
        (np.array(spike_times) * pq.ms).rescale(pq.s),
        t_stop=simulation_time.rescale(pq.s),
        sampling_rate=(1 / timestep).rescale(pq.Hz),
        sampling_period=timestep.rescale(pq.s),
        name=f"DD_{i}",
    )
    for i, spike_times in enumerate(dd_spike_times)
]

mn_segment = Segment(name="Motor Neurons")
mn_segment.spiketrains = [
    SpikeTrain(
        (np.array(spike_times) * pq.ms).rescale(pq.s),
        t_stop=simulation_time.rescale(pq.s),
        sampling_rate=(1 / timestep).rescale(pq.Hz),
        sampling_period=timestep.rescale(pq.s),
        name=f"MN_{i}",
    )
    for i, spike_times in enumerate(mn_spike_times)
]

spike_train_block.segments.append(mn_segment)
joblib.dump(spike_train_block, save_path / "trapezoid_dd_spike_trains_jaxley.pkl")

##############################################################################
# Calculate Firing Rate Statistics
# ---------------------------------

print("\nFiring rate analysis:")

# DD firing rates
dd_firing_rates = np.array(
    [
        mean_firing_rate(st__s.time_slice(st__s.min(), st__s.max()))
        for st__s in dd_segment.spiketrains
        if len(st__s) > 1
    ]
)

# MN firing rates
mn_firing_rates = np.array(
    [
        mean_firing_rate(st__s.time_slice(st__s.min(), st__s.max()))
        for st__s in mn_segment.spiketrains
        if len(st__s) > 1
    ]
)

print("Descending Drive neurons:")
print(f"\tActive neurons: {len(dd_firing_rates)}/{descending_drive_pool.n}")
if len(dd_firing_rates) > 0:
    print(f"\tMean firing rate: {np.mean(dd_firing_rates):.1f} ± {np.std(dd_firing_rates):.1f} pps")
    print(f"\tRate range: {np.min(dd_firing_rates):.1f} - {np.max(dd_firing_rates):.1f} pps")

print("Motor neurons:")
print(f"\tActive neurons: {len(mn_firing_rates)}/{motor_neuron_pool.n}")
if len(mn_firing_rates) > 0:
    print(f"\tMean firing rate: {np.mean(mn_firing_rates):.1f} ± {np.std(mn_firing_rates):.1f} pps")
    print(f"\tRate range: {np.min(mn_firing_rates):.1f} - {np.max(mn_firing_rates):.1f} pps")

##############################################################################
# Advanced Visualization
# -----------------------

fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

# 1. Plot trapezoidal drive pattern
time_s = trapezoid_drive_signal.times.rescale(pq.s).magnitude
axes[0].plot(time_s, trapezoid_drive_signal, "b-", linewidth=2, label="DD Input")
axes[0].axhline(float(dd_baseline__pps), color="r", linestyle="--", alpha=0.7, label="Baseline")
axes[0].set_ylabel("Drive (Hz)")
axes[0].set_title("Trapezoidal Descending Drive Pattern (Ramp Contraction)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2. DD population raster plot
dd_colors = plt.cm.Blues(np.linspace(0.3, 0.8, len(dd_segment.spiketrains)))
for i, (spiketrain, color) in enumerate(zip(dd_segment.spiketrains, dd_colors)):
    if len(spiketrain) > 0:
        axes[1].scatter(spiketrain.magnitude, [i] * len(spiketrain), c=[color], s=0.8, alpha=0.8)

axes[1].set_ylabel("DD Neuron ID")
axes[1].set_title(f"Descending Drive Population Activity (n={descending_drive_pool.n})")
axes[1].set_ylim(-1, descending_drive_pool.n)
axes[1].grid(True, alpha=0.3)

# 3. Motor neuron raster plot
mn_colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(mn_segment.spiketrains)))
active_mn_count = 0
for i, (spiketrain, color) in enumerate(zip(mn_segment.spiketrains, mn_colors)):
    if len(spiketrain) > 0:
        spike_times = spiketrain.rescale(pq.s).magnitude
        axes[2].scatter(spike_times, [i] * len(spike_times), c=[color], s=1.0, alpha=0.8)
        active_mn_count += 1

axes[2].set_ylabel("Motor Neuron ID\n(Recruitment Order)")
axes[2].set_title(f"Motor Neuron Population Activity (n={active_mn_count}/{motor_neuron_pool.n} active)")
axes[2].set_ylim(-1, motor_neuron_pool.n)
axes[2].grid(True, alpha=0.3)

# 4. Population firing rates over time
bin_size_ms = 100

bin_size_s = (bin_size_ms * pq.ms).rescale(pq.s).magnitude

dd_counts, dd_edges_s = population_psth(dd_segment.spiketrains, bin_size_ms * pq.ms)
dd_rates_binned = dd_counts / bin_size_s / descending_drive_pool.n  # Hz

mn_counts, _ = population_psth(mn_segment.spiketrains, bin_size_ms * pq.ms)
mn_rates_binned = mn_counts / bin_size_s / motor_neuron_pool.n  # Hz

bin_centers_s = dd_edges_s + bin_size_s / 2
axes[3].plot(bin_centers_s, dd_rates_binned, "b-", linewidth=2, label="DD Population", alpha=0.8)
axes[3].plot(bin_centers_s, mn_rates_binned, "r-", linewidth=2, label="MN Population", alpha=0.8)

axes[3].set_xlabel("Time (s)")
axes[3].set_ylabel("Population Rate (Hz)")
axes[3].set_title("Population Firing Rates Over Time")
axes[3].legend()
axes[3].grid(True, alpha=0.3)

for ax in axes:
    ax.set_xlim(0, simulation_time.rescale(pq.s).magnitude)

plt.tight_layout()
plt.savefig(save_path / "trapezoid_dd_4panel_jaxley.png", dpi=150)
print(f"\nSaved 4-panel figure to {save_path / 'trapezoid_dd_4panel_jaxley.png'}")
plt.show()

##############################################################################
# Individual Motor Neuron Discharge Rates
# ----------------------------------------

print("\nComputing smoothed discharge rates per neuron...")

# Parameters
window_ms = 400 * pq.ms
dt_s = timestep.rescale(pq.s)
window_samples = int(window_ms.rescale(pq.s) / dt_s)

# Hanning window
hanning_window = np.hanning(window_samples)
hanning_window = hanning_window / (hanning_window.sum() * dt_s)

mn_instantaneous_rates = []
active_neuron_ids = []
mean_firing_rates = []
cv_isi = []

for i, spiketrain in enumerate(mn_segment.spiketrains):
    if len(spiketrain) > 2:
        t = np.arange(0, simulation_time.rescale(pq.s).magnitude, dt_s.magnitude)
        spikes = np.zeros_like(t)
        spike_indices = np.searchsorted(t, spiketrain.magnitude)
        spikes[spike_indices[spike_indices < len(t)]] = 1

        rate = np.convolve(spikes, hanning_window, mode="same")
        mn_instantaneous_rates.append(rate)
        active_neuron_ids.append(i)

        # ISI/CV during plateau
        plateau_spiketrain = spiketrain.time_slice(
            ramp_up_end.rescale(pq.s), plateau_end.rescale(pq.s)
        )

        plateau_duration_s = float((plateau_end - ramp_up_end).rescale(pq.s).magnitude)
        mean_rate = len(plateau_spiketrain) / plateau_duration_s if len(plateau_spiketrain) > 0 else 0.0
        mean_firing_rates.append(mean_rate)

        if len(plateau_spiketrain) > 1:
            spike_times_arr = plateau_spiketrain.rescale(pq.s).magnitude
            isis = np.diff(spike_times_arr)
            cv = np.std(isis) / np.mean(isis) if len(isis) > 1 else 0.0
        else:
            cv = 0.0
        cv_isi.append(cv)

if len(mean_firing_rates) > 0:
    pop_mean_rate = np.mean(mean_firing_rates)
    pop_mean_cv = np.mean(cv_isi)
    print(f"\nPopulation: Mean firing rate = {pop_mean_rate:.2f} Hz, CV = {pop_mean_cv:.2f}")
print(f"Computed rates for {len(active_neuron_ids)} active motor neurons")

# Discharge rates figure
fig2, axes2 = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

if len(mn_instantaneous_rates) > 0:
    rates_array = np.array(mn_instantaneous_rates)
    time_points_plot = np.linspace(0, simulation_time.rescale(pq.s).magnitude, rates_array.shape[1])

    im = axes2[0].imshow(
        rates_array,
        aspect="auto",
        cmap="hot",
        interpolation="bilinear",
        extent=[0, simulation_time.rescale(pq.s).magnitude, 0, len(active_neuron_ids)],
        origin="lower",
        vmin=0,
        vmax=np.percentile(rates_array, 95),
    )

    axes2[0].set_ylabel("Motor Neuron ID\n(Recruitment Order)")
    axes2[0].set_title("Individual Motor Neuron Discharge Rates (Smoothed with 400ms Hanning Window)")
    cbar = plt.colorbar(im, ax=axes2[0])
    cbar.set_label("Firing Rate (Hz)")
    axes2[0].grid(False)

    n_to_plot = len(active_neuron_ids)
    colors = plt.cm.rainbow(np.linspace(0, 1, n_to_plot))

    for neuron_idx in range(n_to_plot):
        axes2[1].plot(
            time_points_plot,
            mn_instantaneous_rates[neuron_idx],
            linewidth=0.8,
            color=colors[neuron_idx],
            label=f"MN {active_neuron_ids[neuron_idx]}" if n_to_plot <= 20 else None,
        )

    axes2[1].set_xlabel("Time (s)")
    axes2[1].set_ylabel("Firing Rate (Hz)")
    axes2[1].set_title(f"All Motor Neuron Discharge Rates (n={n_to_plot})")

    if n_to_plot <= 20:
        axes2[1].legend(loc="upper right", ncol=3, fontsize=6)

    axes2[1].grid(True, alpha=0.3)
    axes2[1].set_xlim(0, simulation_time.rescale(pq.s).magnitude)
    axes2[1].set_ylim(0, np.max(rates_array) * 1.1)

plt.tight_layout()
plt.savefig(save_path / "trapezoid_dd_discharge_rates_jaxley.png", dpi=150)
print(f"Saved discharge rates figure to {save_path / 'trapezoid_dd_discharge_rates_jaxley.png'}")
plt.show()

print("\n" + "=" * 60)
print("[DONE] Jaxley biophysical simulation complete with descending drive!")
print("       jx.integrate() + NERLab channels (soma napp; dendrite caL)")
print("       Per-MN random DD connectivity (50% prob); current injection with fixed driving force.")
print("=" * 60)
