"""
Spinal Network Simulation with Systematic Tendon Tap Protocol - Jaxley Backend
===============================================================================

This example demonstrates **complete spinal reflex network modeling** with a **comprehensive tendon tap
protocol** using the Jaxley (JAX-based) simulator with biophysical motor neuron channels.

This is the Jaxley equivalent of ``11_simulate_spinal_network.py`` which uses NEURON.
Both examples implement the same neuromuscular system:

    - **Motor neuron pool**: α-motoneurons with biophysical HH channels
    - **Afferent populations**: Ia, II (spindle), Ib (GTO) feedback
    - **Interneuron populations**: gII, gIb for reflex modulation
    - **Proprioceptive models**: Muscle spindles and Golgi tendon organs
    - **Hill-type muscle model**: Realistic force generation
    - **Joint dynamics**: Closed-loop biomechanical control
    - **Descending drive**: Cortical control via Poisson spike trains

Learning Objectives
------------------

1. **Reflex Gain Modulation**: How fusimotor (gamma) drive amplifies stretch reflex sensitivity
2. **Sensory Pathway Timing**: Different feedback pathways activate at different times
3. **Reflex-Voluntary Interaction**: Cortical commands interact with spinal reflexes
4. **Closed-loop Biomechanics**: Muscle forces create movement that feeds back to sensors

**Experimental Protocol (5-second, 2-phase design):**

**Phase 1 (0-2.5s): Reflex Gain Modulation**
- Triangular tendon taps every 0.5s with increasing amplitude (50%, 60%, 70%, 80%)
- Stepwise increasing gamma drive (0→25→50→75→100 pps)
- No cortical drive (isolated reflex testing)

**Phase 2 (2.5-5s): Reflex-Voluntary Interaction**
- Repeat triangular tap pattern
- Repeat gamma drive pattern
- Sinusoidal cortical drive (40±1 Hz at 1 Hz)

.. note::
    **Motor-neuron model.** ``AlphaMN__Pool`` defaults to ``model="NERLab"``
    (the production NEURON model). ex11 is a *mixed-frame* network: the
    gII / gIb interneurons defined in
    ``myogen/simulator/jaxley/populations/interneurons.py`` use Na3rp-style
    channels in the modern absolute voltage convention (V_rest ≈ -70 mV),
    while NERLab MNs live in the original 1952-HH frame (V_rest ≈ 0 mV).
    This is fine here because **every synapse in the network targets only
    the MNs** (gII → MN, gIb → MN), so we just need one inhibitory reversal
    set for the NERLab-frame target — see ``IonotropicSynapse_e_syn`` below.
    The interneurons act only as presynaptic sources, and the synapse spike
    threshold ``IonotropicSynapse_v_th = -35 mV`` is correct in their
    (modern) frame. NEURON ex11 handles the same situation by setting
    ``syn.e`` per-postsynaptic-cell — see ``myogen.simulator.neuron.cells.create_synapses``.
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
import matplotlib.pyplot as plt
import numpy as np
import quantities as pq
from neo import AnalogSignal, Block, Segment, SpikeTrain
from tqdm import tqdm

import myogen
from jaxley.connect import sparse_connect
from jaxley.synapses import IonotropicSynapse
from myogen.simulator.jaxley.joint_dynamics import JointDynamics
from myogen.simulator.jaxley.muscle import HillModel
from myogen.simulator.jaxley.populations import (
    AffIa__Pool,
    AffIb__Pool,
    AffII__Pool,
    AlphaMN__Pool,
    DescendingDrive__Pool,
    GIb__Pool,
    GII__Pool,
)
from myogen.simulator.jaxley.proprioception import (
    GolgiTendonOrganModel,
    SpindleModel,
)
from myogen.utils.nwb import export_to_nwb
from myogen.utils.plotting import (
    plot_gto_dynamics,
    plot_membrane_potentials,
    plot_muscle_dynamics,
    plot_raster_spikes,
    plot_spindle_dynamics,
)
from myogen.utils.types import pps
from myogen.simulator.jaxley.jax_models import (
    gamma_init,
    gto_init, gto_params_from_dict, gto_step,
    hill_init_params, hill_init_state, hill_step,
    joint_init, joint_step,
    make_connectivity_matrix,
    make_scan_step,
    poisson_init,
    spindle_init, spindle_params_from_dict, spindle_step,
)

##############################################################################
# Setup Results Directory and Load Previous Data
# -----------------------------------------------

save_path = Path(r"./results")
save_path.mkdir(exist_ok=True)

recruitment_thresholds = joblib.load(save_path / "thresholds.pkl")
print("(OK) Loaded recruitment thresholds from example 00")

##############################################################################
# Define Simulation Parameters
# ---------------------------

# Temporal parameters
dt = 0.025 * pq.ms  # Integration timestep (reduced from 0.1 ms for Hill/spindle ODE accuracy)
tstop = 5e3  # ms - Total simulation duration (5s for comprehensive protocol)
time = np.arange(0, tstop, dt.magnitude)  # Time vector

print("Simulation parameters:")
print(f"\tDuration: {tstop} ms")
print(f"\tTimestep: {dt} ms")
print(f"\tTime samples: {len(time)}")

##############################################################################
# Define Neural Population Sizes
# -----------------------------

# Afferent populations (sensory input)
nIa = 73  # Group Ia afferents from muscle spindles (velocity-sensitive)
nII = 80  # Group II afferents from muscle spindles (length-sensitive)
nIb = 58  # Group Ib afferents from Golgi tendon organs (force-sensitive)

# Interneuron populations (spinal processing)
ngII = 120  # Group II interneurons
ngIb = 145  # Group Ib interneurons

# Motor neurons (output to muscles)
# Motor unit type composition (needed for Hill muscle model)
# NEURON uses 102/18 (85%/15%) for 120 MNs — scale proportionally to loaded pool size
naMN = len(recruitment_thresholds)  # Total α-motoneurons from thresholds file
nType1 = int(round(naMN * 102 / 120))  # Type I motor units (slow, fatigue-resistant)
nType2 = naMN - nType1                  # Type II motor units (fast, fatigue-prone)

# Descending drive (cortical input)
nDD = 400  # Total descending drive neurons
DDorder = 5  # Poisson process batch size

print("Neural population sizes:")
print(f"\t- α-Motoneurons: {naMN} ({nType1} Type I + {nType2} Type II)")
print(f"\t- Ia afferents: {nIa}")
print(f"\t- II afferents: {nII}")
print(f"\t- Ib afferents: {nIb}")
print(f"\t- Interneurons: {ngII + ngIb}")
print(f"\t- Descending drive: {nDD}")

##############################################################################
# Define Descending Drive Pattern
# -------------------------------

# Phase 1: No cortical drive
# Phase 2: 1 Hz sinusoidal drive
DDdrive = np.zeros_like(time)
cortical_start_time = 2500  # ms

# Parameters for sinusoidal drive
bias_hz = 40  # Bias for motor neuron recruitment
amplitude_hz = 1  # Oscillation amplitude
frequency_hz = 1.0  # 1 Hz oscillation

# Apply sinusoidal drive from 2.5s onwards
mask = time >= cortical_start_time
DDdrive[mask] = bias_hz + amplitude_hz * np.sin(
    2 * np.pi * frequency_hz * (time[mask] - cortical_start_time) / 1000.0
)

print("Descending drive pattern:")
print("\t- Phase 1 (0-2.5s): No cortical drive")
print(f"\t- Phase 2 (2.5-5s): Sinusoidal drive ({bias_hz}±{amplitude_hz} Hz at {frequency_hz} Hz)")

##############################################################################
# Define Fusimotor Drive
# --------------------

# Stepwise gamma drive pattern
gamma_times = np.array([0, 750, 1250, 1750, 2250, 2500, 3250, 3750, 4250, 4750, 5000])
gamma_values = np.array([0, 25, 50, 75, 100, 0, 25, 50, 75, 100, 100])

# Create stepwise gamma drive arrays
gDyn = np.zeros_like(time)
gStat = np.zeros_like(time)

# Phase 1: Steps every 0.5s starting at 0.75s
gDyn[(time >= 750) & (time < 1250)] = 25
gDyn[(time >= 1250) & (time < 1750)] = 50
gDyn[(time >= 1750) & (time < 2250)] = 75
gDyn[(time >= 2250) & (time < 2500)] = 100

# Phase 2: Reset and repeat pattern
gDyn[(time >= 3250) & (time < 3750)] = 25
gDyn[(time >= 3750) & (time < 4250)] = 50
gDyn[(time >= 4250) & (time < 4750)] = 75
gDyn[(time >= 4750) & (time <= 5000)] = 100

# Static gamma follows same pattern
gStat = gDyn.copy()

# Add small physiological variability
gDyn = gDyn + myogen.get_random_generator().normal(0, 1, len(time))
gStat = gStat + myogen.get_random_generator().normal(0, 1, len(time))

# Ensure non-negative
gDyn = np.maximum(gDyn, 0)
gStat = np.maximum(gStat, 0)

print("Fusimotor drive parameters:")
print("\t- Stepwise increases: 0 → 25 → 50 → 75 → 100 pps")

##############################################################################
# Initialize Joint Dynamics
# ------------------------

joint_dynamics = JointDynamics(
    inertia__kg_m2=0.001,
    damping__Nm_s_per_rad=0.002,
)

# Initialize joint angle array
artAng = np.zeros_like(time)
artAng[0] = 0.0

print("Joint dynamics parameters:")
print(f"\t- Inertia: {joint_dynamics.inertia__kg_m2} kg⋅m²")
print(f"\t- Damping: {joint_dynamics.damping__Nm_s_per_rad} N⋅m⋅s/rad")

##############################################################################
# Create Neural Populations
# ------------------------

# Create motor neuron pool — default model="NERLab" matches the production
# NEURON model (soma napp + dendrite caL). The mixed-frame issue with
# modern-frame interneurons is resolved by setting the synapse reversal
# below for the NERLab-frame target (see IonotropicSynapse_e_syn).
aMN = AlphaMN__Pool(
    recruitment_thresholds__array=recruitment_thresholds,
    mode="active",
)

# Create descending drive population
DD = DescendingDrive__Pool(n=nDD, poisson_batch_size=DDorder, timestep__ms=dt)

# Create afferent populations
Ia = AffIa__Pool(n=nIa, timestep__ms=dt)
II = AffII__Pool(n=nII, timestep__ms=dt)
Ib = AffIb__Pool(n=nIb, timestep__ms=dt)

# Create interneuron populations
gII = GII__Pool(n=ngII)
gIb = GIb__Pool(n=ngIb)

print(f"(OK) Created neural populations (MNs: {aMN.model} — napp + caL; gII/gIb: modern frame)")

##############################################################################
# Create Proprioceptive Models
# ---------------------------

# Golgi Tendon Organ - monitors muscle force/tension
gto = GolgiTendonOrganModel(
    simulation_time__ms=tstop * pq.ms,
    time_step__ms=dt,
    gto_parameters=GolgiTendonOrganModel.create_default_gto_parameters(),
)

# Muscle Spindle - monitors muscle length and velocity
# Uses the full Mileusnic et al. (2006) 2nd-order ODE model matching NEURON.
spindle_params = SpindleModel.create_default_spindle_parameters()
spin = SpindleModel(
    simulation_time__ms=tstop * pq.ms,
    time_step__ms=dt,
    spindle_parameters=spindle_params,
)

print("(OK) Initialized proprioceptive models (spindle, GTO)")

##############################################################################
# Create Muscle Model
# ------------------

hill_muscle = HillModel(
    simulation_time__ms=tstop * pq.ms,
    time_step__ms=dt,
    muscle_parameters=HillModel.create_default_muscle_parameters(),
    n_motor_units_type1=nType1,
    n_motor_units_type2=nType2,
    initial_joint_angle__deg=artAng[0],
    muscle_role="flexor",
)

print("(OK) Created muscle model")
print(f"\t- Motor units: {naMN} ({nType1} Type I + {nType2} Type II)")
print(f"\t- Force capacity: ~{hill_muscle.F0:.1f} N")

##############################################################################
# Create Arrays for Force Tracking
# --------------------------------

musculotendon_force__N = np.zeros_like(time)

##############################################################################
# Tendon Tap Schedule
# -------------------

TENDON_TAP_DURATION = 100  # ms
TENDON_TAP_SCHEDULE = [
    # Phase 1: Ascending amplitudes without cortical drive
    (500, 50.0),
    (1000, 60.0),
    (1500, 70.0),
    (2000, 80.0),
    # Phase 2: Repeat pattern with cortical drive
    (3000, 50.0),
    (3500, 60.0),
    (4000, 70.0),
    (4500, 80.0),
]

print("\nTendon tap schedule:")
print(f"\t- Duration per tap: {TENDON_TAP_DURATION} ms")
print(f"\t- Phase 1: {len([t for t, m in TENDON_TAP_SCHEDULE if t < 2500])} taps")
print(f"\t- Phase 2: {len([t for t, m in TENDON_TAP_SCHEDULE if t >= 2500])} taps")

##############################################################################
# Causally-Correct Closed-Loop Simulation Setup
# ----------------------------------------------
#
# A single Python for-loop advances all components together each timestep:
#   - Jaxley neural network (gII + gIb + MN): stepped with build_init_and_step_fn
#   - DD/Ia/II/Ib afferent cells: stepped via their existing integrate() API
#   - Hill muscle, spindle, GTO, joint dynamics: stepped via integrate()
#
# Afferent feedback at step t affects neural input at step t+1 (1-step delay,
# ~0.1 ms at dt=0.1 ms — physiologically negligible).

print("\nStarting causally-correct closed-loop simulation (Jaxley)...")
print(f"\tDuration: {tstop} ms")
print(f"\tTimestep: {dt} ms")

dt_ms = float(dt.rescale(pq.ms).magnitude)
dt_s = float(dt.rescale(pq.s).magnitude)
n_steps = len(time)

# Synaptic parameters — matched to NEURON ex11 as starting point.
# TODO: validate EPSP/IPSP amplitudes; cable-dendrite Jaxley MNs may need
# retuning since membrane impedance differs from point-neuron NEURON model.
# IIR / Exp2Syn equivalence factor (~0.06) for the EXCITATORY drives onto MNs.
# NEURON uses live-conductance Exp2Syn synapses with shaped onset; this script
# uses an IIR + fixed driving force ("g × (e_exc_mn - V_mn)") approximation
# which delivers ~10× more effective current at the same nominal weight, so
# the DD and Ia weights here are scaled down to keep MN recruitment in the
# physiological window.  See ex03 for the same equivalence factor analysis.
# in_weight (postsynaptic = modern-frame interneurons) and base_inh_weight
# (postsynaptic = NERLab MN, but via a real IonotropicSynapse with live V)
# don't share the factor and stay at the NEURON values.
base_dd_weight = 0.01    # µS — excitatory DD → MN  (NEURON: 0.05 µS, ×0.2 for IIR)
base_ia_weight = 0.01    # µS — excitatory Ia → MN  (NEURON: 0.05 µS, ×0.2 for IIR)
in_weight = 0.025        # µS — excitatory II → gII, Ib → gIb  (NEURON: 0.025 µS)
base_inh_weight = 0.05   # µS — inhibitory gII/gIb → MN  (NEURON: 0.05 µS)
tau_syn = 5.0            # ms — synaptic exponential decay
e_exc = 0.0              # mV — excitatory reversal potential
v_rest = -70.0           # mV — resting potential

# Per-MN Henneman (size-principle) current scaling
threshold_min = recruitment_thresholds.min()
threshold_max = recruitment_thresholds.max()
normalized_thresholds = (recruitment_thresholds - threshold_min) / (threshold_max - threshold_min)
mn_current_scale = np.exp(-1.0 * normalized_thresholds)

# Spike recording arrays
dd_spike_times = [[] for _ in range(nDD)]
ia_spike_times = [[] for _ in range(nIa)]
ii_spike_times = [[] for _ in range(nII)]
ib_spike_times = [[] for _ in range(nIb)]
gii_spike_times = [[] for _ in range(ngII)]
gib_spike_times = [[] for _ in range(ngIb)]
mn_spike_times = [[] for _ in range(naMN)]

# Muscle/spindle/GTO time series for plotting
muscle_length_series = []
muscle_velocity_series = []
muscle_force_series = []
ia_firing_series = []
ii_firing_series = []
ib_firing_series = []

# Membrane trace storage for selected MNs
vm_traces = {}
vm_cell_indices = [0, 10, 20, 30, 40]
for _ci in vm_cell_indices:
    if _ci < naMN:
        vm_traces[_ci] = np.zeros(n_steps)

##############################################################################
# Connectivity Pre-computation
# ----------------------------

print("\nPre-computing neural connectivity...")

# DD → MN (forward: mn_idx → dd_list; reverse: dd_idx → mn_list)
dd_to_mn_connections = {
    mn_idx: [j for j in range(nDD) if myogen.get_random_generator().random() < 0.3]
    for mn_idx in range(naMN)
}
dd_to_mn_rev = {dd: [] for dd in range(nDD)}
for mn_idx, dd_list in dd_to_mn_connections.items():
    for dd_idx in dd_list:
        dd_to_mn_rev[dd_idx].append(mn_idx)

# Ia → MN (forward and reverse)
ia_to_mn_connections = {
    mn_idx: [j for j in range(nIa) if myogen.get_random_generator().random() < 0.8]
    for mn_idx in range(naMN)
}
ia_to_mn_rev = {ia: [] for ia in range(nIa)}
for mn_idx, ia_list in ia_to_mn_connections.items():
    for ia_idx in ia_list:
        ia_to_mn_rev[ia_idx].append(mn_idx)

# II → gII (forward and reverse)
ii_to_gii_connections = {
    gii_idx: [j for j in range(nII) if myogen.get_random_generator().random() < 0.3]
    for gii_idx in range(ngII)
}
ii_to_gii_rev = {ii: [] for ii in range(nII)}
for gii_idx, ii_list in ii_to_gii_connections.items():
    for ii_idx in ii_list:
        ii_to_gii_rev[ii_idx].append(gii_idx)

# Ib → gIb (forward and reverse)
ib_to_gib_connections = {
    gib_idx: [j for j in range(nIb) if myogen.get_random_generator().random() < 0.3]
    for gib_idx in range(ngIb)
}
ib_to_gib_rev = {ib: [] for ib in range(nIb)}
for gib_idx, ib_list in ib_to_gib_connections.items():
    for ib_idx in ib_list:
        ib_to_gib_rev[ib_idx].append(gib_idx)

# Pre-compute MN axonal delays (ms) — used when scheduling spikes into hill model
axonal_delays_ms = np.array([
    1.0 + (mn.axon_delay__ms.magnitude if hasattr(mn.axon_delay__ms, 'magnitude')
           else float(mn.axon_delay__ms))
    for mn in aMN
])

##############################################################################
# JAX Physiology Setup
# --------------------
# Build param dicts and initial states for JAX functional step functions.
# hill_init_params calls ForceSatParams (scipy) once — never called again.

hillD_default = HillModel.create_default_muscle_parameters()
hill_p = hill_init_params(hillD_default, nType1, nType2, dt_ms)
spindle_p = spindle_params_from_dict(spindle_params)
gto_p = gto_params_from_dict(GolgiTendonOrganModel.create_default_gto_parameters())
joint_p = {
    "inertia":   joint_dynamics.inertia__kg_m2,
    "damping":   joint_dynamics.damping__Nm_s_per_rad,
    "stiffness": joint_dynamics.stiffness__Nm_per_rad,
}

# Initial muscle length from HillModel's auto-computed rest length
L0_init = float(hill_muscle._hill_model.L[0])
# MN spike buffer depth: max axonal delay + safety margin
_max_mn_delay_steps = int(np.ceil(axonal_delays_ms.max() / dt_ms)) + 4
hill_state    = hill_init_state(L0_init, naMN, max_delay_steps=_max_mn_delay_steps)
spindle_state = spindle_init()
gto_state     = gto_init()
joint_state   = joint_init(angle_deg=float(np.degrees(artAng[0])))

# Per-MU axonal delay in timesteps (for hill_step spike buffer)
delay_steps_arr = np.maximum(1, np.round(axonal_delays_ms / dt_ms)).astype(np.int32)

print("(OK) JAX physiology states initialised")
print(f"\t- Hill L0_init = {L0_init:.4f}, max_delay_steps = {_max_mn_delay_steps}")

##############################################################################
# Build Combined Jaxley Network
# ----------------------------

print("Building combined jx.Network (gII + gIb + MN)...")

# Collect and reset all cells
gii_cells = []
for gii_cell in gII:
    cell = gii_cell.cell
    cell.delete_recordings()
    cell.delete_stimuli()
    cell.set("v", -70.0)
    cell.init_states()
    gii_cells.append(cell)

gib_cells = []
for gib_cell in gIb:
    cell = gib_cell.cell
    cell.delete_recordings()
    cell.delete_stimuli()
    cell.set("v", -70.0)
    cell.init_states()
    gib_cells.append(cell)

mn_cells = []
for mn in aMN:
    cell = mn.cell
    cell.delete_recordings()
    cell.delete_stimuli()
    cell.set("v", 0.0)            # NERLab resting potential (1952-HH frame)
    cell.init_states()
    mn_cells.append(cell)

n_gii = len(gii_cells)
n_gib = len(gib_cells)
n_mn = len(mn_cells)

combined_net = jx.Network(gii_cells + gib_cells + mn_cells)

# Register recordings in order: gII soma, gIb soma, MN soma
for i in range(n_gii):
    combined_net.cell(i).branch(0).loc(0.5).record("v")
for i in range(n_gib):
    combined_net.cell(n_gii + i).branch(0).loc(0.5).record("v")
for mn_idx in range(n_mn):
    combined_net.cell(n_gii + n_gib + mn_idx).branch(0).loc(0.5).record("v")

# Register stimulus sites with placeholder arrays (establishes external_inds order)
# Must match recording order: gII, gIb, MN
placeholder = jnp.zeros(n_steps)
for i in range(n_gii):
    combined_net.cell(i).branch(0).loc(0.5).stimulate(placeholder)
for i in range(n_gib):
    combined_net.cell(n_gii + i).branch(0).loc(0.5).stimulate(placeholder)
for mn_idx in range(n_mn):
    combined_net.cell(n_gii + n_gib + mn_idx).branch(0).loc(0.5).stimulate(placeholder)

# Inhibitory synapses: gII → MN and gIb → MN (handled internally by Jaxley step_fn)
print("  Adding inhibitory synapses (gII → MN, gIb → MN, p=0.3)...")
sparse_connect(
    combined_net.cell(list(range(n_gii))),
    combined_net.cell(list(range(n_gii + n_gib, n_gii + n_gib + n_mn))),
    IonotropicSynapse(),
    p=0.3,
)
sparse_connect(
    combined_net.cell(list(range(n_gii, n_gii + n_gib))),
    combined_net.cell(list(range(n_gii + n_gib, n_gii + n_gib + n_mn))),
    IonotropicSynapse(),
    p=0.3,
)
combined_net.set("IonotropicSynapse_gS", base_inh_weight)
# Inhibitory reversal on NERLab MN targets — set in the NERLab voltage frame.
#
# We want the same *physical* inhibitory driving force the original Powers2017
# setup delivered:
#     Powers2017 frame:   e_syn = -80 mV,  V_rest = -65 mV
#                         →  driving = e_syn - V_rest = -15 mV  (mild GABA-like)
# In the NERLab frame (V_rest = 0 mV) the analogous value is
#     NERLab frame:       e_syn = -15 mV,  V_rest =  0 mV
#                         →  driving = -15 mV  (same physical effect)
#
# NEURON ex11 uses syn.e = -75 mV literally without a frame shift (see
# myogen.simulator.neuron.cells:776), which on a NERLab MN gives ~75 mV of
# hyperpolarising driving force. That's far stronger than the Powers2017
# equivalent and would over-inhibit MNs into silence here. We match the
# physical driving force instead of the literal mV value, so the recruitment
# behaviour is comparable across the two backends.
combined_net.set("IonotropicSynapse_e_syn", -15.0)
combined_net.set("IonotropicSynapse_k_minus", 0.1)   # 10 ms IPSP decay
# Spike activation threshold — applies to the PRESYNAPTIC cell V crossing.
# Sources are gII/gIb interneurons (modern frame, V_rest ≈ -70 mV), so the
# -35 mV threshold below is correct in their frame, not the MN's NERLab frame.
combined_net.set("IonotropicSynapse_v_th", -35.0)
combined_net.set("IonotropicSynapse_delta", 10.0)    # voltage sensitivity (mV)

# Compile Jaxley single-step function
print("  Compiling Jaxley step function...")
combined_net.to_jax()
init_fn, step_fn = build_init_and_step_fn(combined_net)
params = combined_net.get_parameters()
external_inds = combined_net.external_inds.copy()
rec_inds = combined_net.recordings.rec_index.to_numpy()
states, params = init_fn(params)
step_fn_jit = jax.jit(step_fn)

print(f"  Network: {n_gii + n_gib + n_mn} cells, "
      f"{n_gii + n_gib + n_mn} stimulus sites, {len(rec_inds)} recording sites")

##############################################################################
# Run Closed-Loop Simulation with lax.scan
# -----------------------------------------
#
# lax.scan compiles the entire loop into a single GPU/XLA kernel, eliminating
# Python interpreter overhead at each of the 200k timesteps (dt=0.025ms, 5s).
#
# All Python state (afferent cells, DD cells) is replaced by JAX-native
# generators (poisson_step, gamma_step) inside the compiled scan body.
# Per-cell axonal delays are approximated with per-population mean FIFO queues.

print("\n[Simulation] Building lax.scan closed-loop...")

decay = np.exp(-dt_ms / tau_syn)

# --- Pre-compute tendon tap coefficient arrays (normalised, no L dependence) ---
# update_physiology applies: L_sp = L*(1+tap_dL),  V_sp = V + L*tap_dV
tap_dL = np.zeros(n_steps, dtype=np.float32)
tap_dV = np.zeros(n_steps, dtype=np.float32)
_half_tap = TENDON_TAP_DURATION / 2.0           # ms
_dV_rate  = 1.0 / (_half_tap * 1e-3)            # 1/s normalised by L inside update_physiology
for _tap_t, _tap_mag in TENDON_TAP_SCHEDULE:
    _tap_end = _tap_t + TENDON_TAP_DURATION
    _frac    = _tap_mag / 100.0
    for _ii, _t in enumerate(time):
        if _tap_t <= _t < _tap_end:
            if _t < _tap_t + _half_tap:
                _prog = (_t - _tap_t) / _half_tap
                tap_dL[_ii] = _frac * _prog
                tap_dV[_ii] = _frac * _dV_rate
            else:
                _prog = (_tap_end - _t) / _half_tap
                tap_dL[_ii] = _frac * _prog
                tap_dV[_ii] = -_frac * _dV_rate

# --- Dense connectivity matrices (pre_idx → post_idx) ---
print("  Building connectivity matrices...")
# dd_to_mn_rev: {dd_idx: [mn_idx]}  == forward map for make_connectivity_matrix
dd_to_mn_mat  = make_connectivity_matrix(dd_to_mn_rev,  nDD,  naMN)   # (nDD,  naMN)
ia_to_mn_mat  = make_connectivity_matrix(ia_to_mn_rev,  nIa,  naMN)   # (nIa,  naMN)
ii_to_gii_mat = make_connectivity_matrix(ii_to_gii_rev, nII,  ngII)   # (nII,  ngII)
ib_to_gib_mat = make_connectivity_matrix(ib_to_gib_rev, nIb,  ngIb)   # (nIb,  ngIb)

# --- Afferent response thresholds and gamma-ISI shape parameters ---
ia_rts   = jnp.array([c.RT for c in Ia], dtype=jnp.float32)
ii_rts   = jnp.array([c.RT for c in II], dtype=jnp.float32)
ib_rts   = jnp.array([c.RT for c in Ib], dtype=jnp.float32)
ia_shape = 1.0   # exponential ISI (pure Poisson); increase for more regular firing
ii_shape = 1.0
ib_shape = 1.0

# --- Per-cell axonal delay steps (FIFO queue depth per afferent cell) ---
def _cell_ds(cell):
    d = cell.axon_delay__ms.magnitude if hasattr(cell.axon_delay__ms, 'magnitude') else float(cell.axon_delay__ms)
    return max(1, int(round(d / dt_ms)))

# Per-cell delay arrays (sorted by pool__ID so index matches gamma generator order)
ia_delay_steps_arr = np.array([_cell_ds(c) for c in sorted(Ia, key=lambda c: c.pool__ID)], dtype=np.int32)
ii_delay_steps_arr = np.array([_cell_ds(c) for c in sorted(II, key=lambda c: c.pool__ID)], dtype=np.int32)
ib_delay_steps_arr = np.array([_cell_ds(c) for c in sorted(Ib, key=lambda c: c.pool__ID)], dtype=np.int32)
max_ia_delay_steps = int(ia_delay_steps_arr.max())
max_ii_delay_steps = int(ii_delay_steps_arr.max())
max_ib_delay_steps = int(ib_delay_steps_arr.max())
print(f"  Per-cell afferent delays — Ia: {ia_delay_steps_arr.min()}–{ia_delay_steps_arr.max()} steps "
      f"({ia_delay_steps_arr.min()*dt_ms:.2f}–{ia_delay_steps_arr.max()*dt_ms:.2f} ms), "
      f"II: {ii_delay_steps_arr.min()}–{ii_delay_steps_arr.max()} steps, "
      f"Ib: {ib_delay_steps_arr.min()}–{ib_delay_steps_arr.max()} steps")

# --- Build scan_step closure (closes over all static params) ---
print("  Building scan step function...")
scan_step = make_scan_step(
    jaxley_step_fn      = step_fn,
    jaxley_params       = params,
    external_inds       = external_inds,
    rec_inds            = rec_inds,
    n_gii               = n_gii,
    n_gib               = n_gib,
    n_mn                = n_mn,
    ia_rts              = ia_rts,
    ii_rts              = ii_rts,
    ib_rts              = ib_rts,
    ia_shape            = ia_shape,
    ii_shape            = ii_shape,
    ib_shape            = ib_shape,
    dd_N_batch          = DDorder,
    dd_to_mn_mat        = dd_to_mn_mat,
    ia_to_mn_mat        = ia_to_mn_mat,
    ii_to_gii_mat       = ii_to_gii_mat,
    ib_to_gib_mat       = ib_to_gib_mat,
    ia_delay_steps_arr  = ia_delay_steps_arr,
    ii_delay_steps_arr  = ii_delay_steps_arr,
    ib_delay_steps_arr  = ib_delay_steps_arr,
    delay_steps         = delay_steps_arr,
    hill_p              = hill_p,
    spindle_p           = spindle_p,
    gto_p               = gto_p,
    joint_p             = joint_p,
    base_dd_weight      = base_dd_weight,
    base_ia_weight      = base_ia_weight,
    in_weight           = in_weight,
    e_exc               = e_exc,                # AMPA reversal for interneurons (modern frame)
    v_rest              = v_rest,
    e_exc_mn            = 70.0,                 # AMPA reversal for NERLab MNs (1952-HH frame)
    mn_spike_threshold_mV = 50.0,               # NERLab AP detection threshold
    mn_current_scale    = mn_current_scale,
    tau_syn_decay       = float(decay),
    dt_ms               = dt_ms,
    dt_s                = dt_s,
)

# --- Initial carry ---
init_carry = {
    "neural":   states,    # Jaxley neural states from init_fn
    "phys": {
        "hill":    hill_state,
        "spindle": spindle_state,
        "gto":     gto_state,
        "joint":   joint_state,
    },
    "g_dd":     jnp.zeros(n_mn,  dtype=jnp.float32),
    "g_ia":     jnp.zeros(n_mn,  dtype=jnp.float32),
    "g_ii":     jnp.zeros(n_gii, dtype=jnp.float32),
    "g_ib":     jnp.zeros(n_gib, dtype=jnp.float32),
    # prev_v is just a "below every threshold" sentinel for first-iteration spike
    # detection; -70 sits below both the gII/gIb spike threshold (-35 mV, modern
    # frame) and the NERLab MN threshold (+50 mV), so the first real V crossing
    # is detected correctly for every cell regardless of which frame it lives in.
    "prev_v":   jnp.full(n_gii + n_gib + n_mn, -70.0, dtype=jnp.float32),
    "dd_st":    poisson_init(nDD, DDorder, seed=42),
    "ia_st":    gamma_init(nIa, ia_shape, seed=43),
    "ii_st":    gamma_init(nII, ii_shape, seed=44),
    "ib_st":    gamma_init(nIb, ib_shape, seed=45),
    "prev_Iay": jnp.float32(0.0),
    "prev_IIy": jnp.float32(0.0),
    "prev_Iby": jnp.float32(0.0),
    "ia_delay_buf": jnp.zeros((nIa, max_ia_delay_steps), dtype=jnp.float32),
    "ii_delay_buf": jnp.zeros((nII, max_ii_delay_steps), dtype=jnp.float32),
    "ib_delay_buf": jnp.zeros((nIb, max_ib_delay_steps), dtype=jnp.float32),
}

# --- Per-step inputs stacked as (n_steps, ...) ---
scan_inputs = {
    "DDdrive": jnp.array(DDdrive, dtype=jnp.float32),
    "gDyn":    jnp.array(gDyn,    dtype=jnp.float32),
    "gStat":   jnp.array(gStat,   dtype=jnp.float32),
    "tap_dL":  jnp.array(tap_dL,  dtype=jnp.float32),
    "tap_dV":  jnp.array(tap_dV,  dtype=jnp.float32),
}

# --- Execute lax.scan ---
# First call triggers XLA compilation (~30-60s); subsequent calls are fast.
print(f"\n[Simulation] Running lax.scan over {n_steps} steps "
      f"({tstop:.0f} ms at dt={dt_ms} ms)...")
_, scan_out = jax.jit(lambda c, xs: jax.lax.scan(scan_step, c, xs))(init_carry, scan_inputs)
jax.block_until_ready(scan_out)
print("Scan complete. Extracting results...")

# --- Convert boolean spike arrays → spike-time lists ---
# scan_out["mn_spikes"] shape: (n_steps, n_mn), dtype bool
dd_spike_arr  = np.array(scan_out["dd_spikes"])    # (n_steps, nDD)
mn_spike_arr  = np.array(scan_out["mn_spikes"])    # (n_steps, n_mn)
gii_spike_arr = np.array(scan_out["gii_spikes"])   # (n_steps, n_gii)
gib_spike_arr = np.array(scan_out["gib_spikes"])   # (n_steps, n_gib)
ia_spike_arr  = np.array(scan_out["ia_spikes"])    # (n_steps, nIa)
ii_spike_arr  = np.array(scan_out["ii_spikes"])    # (n_steps, nII)
ib_spike_arr  = np.array(scan_out["ib_spikes"])    # (n_steps, nIb)

dd_spike_times  = [time[dd_spike_arr[:,  dd]].tolist()     for dd     in range(nDD)]
mn_spike_times  = [time[mn_spike_arr[:,  mn_idx]].tolist() for mn_idx in range(n_mn)]
gii_spike_times = [time[gii_spike_arr[:, gi]].tolist()     for gi     in range(n_gii)]
gib_spike_times = [time[gib_spike_arr[:, gi]].tolist()     for gi     in range(n_gib)]
ia_spike_times  = [time[ia_spike_arr[:,  ia]].tolist()     for ia     in range(nIa)]
ii_spike_times  = [time[ii_spike_arr[:,  ii]].tolist()     for ii     in range(nII)]
ib_spike_times  = [time[ib_spike_arr[:,  ib]].tolist()     for ib     in range(nIb)]

# --- Time series (physiology outputs) ---
_force_norm_arr   = np.array(scan_out["force"])    # normalised
_torque_norm_arr  = np.array(scan_out["torque"])   # normalised
muscle_length_series  = np.array(scan_out["L"]).tolist()
muscle_velocity_series = np.gradient(np.array(scan_out["L"]), dt_s).tolist()
muscle_force_series   = (hill_p["F0"] * _force_norm_arr).tolist()
muscle_torque_series  = _torque_norm_arr.tolist()
ia_firing_series      = np.array(scan_out["Iay"]).tolist()
ii_firing_series      = np.array(scan_out["IIy"]).tolist()
ib_firing_series      = np.array(scan_out["Iby"]).tolist()

# --- Force and joint angle arrays (overwrite initialised zeros) ---
musculotendon_force__N = hill_p["F0"] * _force_norm_arr
artAng = np.radians(np.array(scan_out["angle_deg"]))

# --- Membrane potential traces for selected MNs ---
vm_cell_indices = [0, 10, 20, 30, 40]
_v_mn_series = np.array(scan_out["v_mn"])          # (n_steps, n_mn)
vm_traces = {
    _ci: _v_mn_series[:, _ci]
    for _ci in vm_cell_indices
    if _ci < n_mn
}

# Summary
active_mn_count  = sum(1 for s in mn_spike_times  if s)
active_gii_count = sum(1 for s in gii_spike_times if s)
active_gib_count = sum(1 for s in gib_spike_times if s)
print(f"\nSimulation complete!")
print(f"  MNs active : {active_mn_count}/{naMN}")
print(f"  gII active : {active_gii_count}/{ngII}")
print(f"  gIb active : {active_gib_count}/{ngIb}")
print(f"  Peak force : {max(muscle_force_series) if muscle_force_series else 0:.4f} F0")
print(f"  Peak angle : {np.degrees(np.max(artAng)):.2f} deg")
print(f"  Peak force (norm)       : {max(muscle_force_series) / hill_p['F0']:.4f} F0" if muscle_force_series else "  Peak force: (none)")

##############################################################################
# Convert Results to Neo Format
# -----------------------------
#
# Store all simulation data in a Neo Block matching NEURON's structure:
# - 7 population segments (aMN, Ia, II, Ib, gII, gIb, DD) with spike trains
# - Muscle segment with dynamics AnalogSignals
# - Spindle segment with proprioceptive AnalogSignals
# - GTO segment with force feedback AnalogSignal

results = Block(name="Spinal Network - Jaxley")
t_stop_s = (tstop * pq.ms).rescale(pq.s)
sampling_period_s = dt.rescale(pq.s)
n_steps = len(time)

def _make_spiketrain(spikes_ms, name):
    """Create Neo SpikeTrain, filtering out any spikes beyond t_stop (e.g. from axon delays)."""
    arr = np.array(spikes_ms)
    if len(arr) > 0:
        arr = arr[arr < tstop]
    times_s = (arr * pq.ms).rescale(pq.s) if len(arr) > 0 else np.array([]) * pq.s
    return SpikeTrain(times_s, t_stop=t_stop_s, name=name)

# --- Population segments (matching NEURON order) ---

# 1. aMN segment (spike trains + membrane potential traces)
mn_segment = Segment(name="aMN")
mn_segment.spiketrains = [_make_spiketrain(spikes, f"aMN_{i}") for i, spikes in enumerate(mn_spike_times)]
for cell_idx, trace in vm_traces.items():
    mn_segment.analogsignals.append(
        AnalogSignal(
            np.array(trace).reshape(-1, 1) * pq.mV,
            sampling_period=sampling_period_s,
            name=f"aMN_cell{cell_idx}_Vm",
            cell_idx=cell_idx,
        )
    )
results.segments.append(mn_segment)

# 2. Ia segment
ia_segment = Segment(name="Ia")
ia_segment.spiketrains = [_make_spiketrain(spikes, f"Ia_{i}") for i, spikes in enumerate(ia_spike_times)]
results.segments.append(ia_segment)

# 3. II segment
ii_segment = Segment(name="II")
ii_segment.spiketrains = [_make_spiketrain(spikes, f"II_{i}") for i, spikes in enumerate(ii_spike_times)]
results.segments.append(ii_segment)

# 4. Ib segment
ib_segment = Segment(name="Ib")
ib_segment.spiketrains = [_make_spiketrain(spikes, f"Ib_{i}") for i, spikes in enumerate(ib_spike_times)]
results.segments.append(ib_segment)

# 5. gII segment
gii_segment = Segment(name="gII")
gii_segment.spiketrains = [_make_spiketrain(spikes, f"gII_{i}") for i, spikes in enumerate(gii_spike_times)]
results.segments.append(gii_segment)

# 6. gIb segment
gib_segment = Segment(name="gIb")
gib_segment.spiketrains = [_make_spiketrain(spikes, f"gIb_{i}") for i, spikes in enumerate(gib_spike_times)]
results.segments.append(gib_segment)

# 7. DD segment
dd_segment = Segment(name="DD")
dd_segment.spiketrains = [_make_spiketrain(spikes, f"DD_{i}") for i, spikes in enumerate(dd_spike_times)]
results.segments.append(dd_segment)

# --- Model segments (muscle, spindle, GTO) ---

# 8. Hill muscle segment
muscle_segment = Segment(name="hill_muscle")

# Muscle length — always from scan output (hill_muscle.muscle_length is zero-initialized)
muscle_length_arr = np.array(muscle_length_series[:n_steps])
muscle_segment.analogsignals.append(
    AnalogSignal(
        muscle_length_arr.reshape(-1, 1) * pq.dimensionless,
        sampling_period=sampling_period_s,
        name="hill_muscle_muscle_length",
        attr_name="muscle_length",
    )
)

# Muscle force (from JAX hill_step series collected during loop)
muscle_force_arr = np.array(muscle_force_series[:n_steps])
muscle_segment.analogsignals.append(
    AnalogSignal(
        muscle_force_arr.reshape(-1, 1) * pq.dimensionless,
        sampling_period=sampling_period_s,
        name="hill_muscle_muscle_force",
        attr_name="muscle_force",
    )
)

# Muscle torque (normalised, from JAX hill_step series collected during loop)
muscle_torque_arr = np.array(muscle_torque_series[:n_steps]) if muscle_torque_series else np.zeros(n_steps)
muscle_segment.analogsignals.append(
    AnalogSignal(
        muscle_torque_arr.reshape(-1, 1) * pq.dimensionless,
        sampling_period=sampling_period_s,
        name="hill_muscle_muscle_torque",
        attr_name="muscle_torque",
    )
)

# TypeI/TypeII summed activations — from scan outputs
for _scan_key, _attr in [("type1_act", "type1_activation"), ("type2_act", "type2_activation")]:
    _act_arr = np.array(scan_out[_scan_key])[:n_steps]
    muscle_segment.analogsignals.append(
        AnalogSignal(
            _act_arr.reshape(-1, 1) * pq.dimensionless,
            sampling_period=sampling_period_s,
            name=f"hill_muscle_{_attr}",
            attr_name=_attr,
        )
    )

results.segments.append(muscle_segment)

# 9. Spindle segment
spin_segment = Segment(name="spin")

# Primary afferent (Ia) firing rate — use capped series (matches what afferents received)
ia_data = np.array(ia_firing_series[:n_steps])
spin_segment.analogsignals.append(
    AnalogSignal(
        ia_data.reshape(-1, 1) * pq.dimensionless,
        sampling_period=sampling_period_s,
        name="spin_primary_afferent_firing__Hz",
        attr_name="primary_afferent_firing__Hz",
    )
)

# Secondary afferent (II) firing rate — use capped series
ii_data = np.array(ii_firing_series[:n_steps])
spin_segment.analogsignals.append(
    AnalogSignal(
        ii_data.reshape(-1, 1) * pq.dimensionless,
        sampling_period=sampling_period_s,
        name="spin_secondary_afferent_firing__Hz",
        attr_name="secondary_afferent_firing__Hz",
    )
)

# Intrafusal fiber activations — from scan outputs (bag1, bag2 only; chain is algebraic)
for _spin_key, _spin_attr in [("bag1_act", "bag1_activation"), ("bag2_act", "bag2_activation")]:
    _act_arr = np.array(scan_out[_spin_key])[:n_steps]
    spin_segment.analogsignals.append(
        AnalogSignal(
            _act_arr.reshape(-1, 1) * pq.dimensionless,
            sampling_period=sampling_period_s,
            name=f"spin_{_spin_attr}",
            attr_name=_spin_attr,
        )
    )

# Intrafusal tensions — shape (n_steps, 3): [Bag1, Bag2, Chain] per row
_tensions_arr = np.array(scan_out["spin_T"])[:n_steps]   # (n_steps, 3)
spin_segment.analogsignals.append(
    AnalogSignal(
        _tensions_arr * pq.dimensionless,
        sampling_period=sampling_period_s,
        name="spin_intrafusal_tensions",
        attr_name="intrafusal_tensions",
    )
)

# Gamma fusimotor drive — step-like reference signal for activations subplot
spin_segment.analogsignals.append(
    AnalogSignal(
        gDyn[:n_steps].reshape(-1, 1) * pq.dimensionless,
        sampling_period=sampling_period_s,
        name="spin_gamma_dynamic",
        attr_name="gamma_dynamic",
    )
)

results.segments.append(spin_segment)

# 10. GTO segment — always from scan output (gto.ib_afferent_firing__Hz is zero-initialized)
gto_segment = Segment(name="gto")
ib_data = np.array(ib_firing_series[:n_steps])
gto_segment.analogsignals.append(
    AnalogSignal(
        ib_data.reshape(-1, 1) * pq.dimensionless,
        sampling_period=sampling_period_s,
        name="gto_ib_afferent_firing__Hz",
        attr_name="ib_afferent_firing__Hz",
    )
)
results.segments.append(gto_segment)

##############################################################################
# Save Results
# ------------

joblib.dump(results, save_path / "spinal_network_results_jaxley.pkl")
joblib.dump(artAng, save_path / "joint_angles_jaxley.pkl")
joblib.dump(musculotendon_force__N, save_path / "musculotendon_force__N_jaxley.pkl")

# Save drive signals
drive_signals = {
    "time": time,
    "descending_drive": DDdrive,
    "gamma_dynamic": gDyn,
    "gamma_static": gStat,
    "tendon_tap_schedule": TENDON_TAP_SCHEDULE,
    "tendon_tap_duration": TENDON_TAP_DURATION,
}
joblib.dump(drive_signals, save_path / "drive_signals_jaxley.pkl")

print(f"\nResults saved to {save_path}")
print("\t- spinal_network_results_jaxley.pkl")
print("\t- joint_angles_jaxley.pkl")
print("\t- musculotendon_force__N_jaxley.pkl")
print("\t- drive_signals_jaxley.pkl")

##############################################################################
# Export to NWB Format
# --------------------

try:
    nwb_filepath = export_to_nwb(
        results,
        save_path / "spinal_network_results_jaxley.nwb",
        session_description=(
            "MyoGen spinal network simulation (Jaxley backend) with systematic tendon tap protocol. "
            "Uses NERLab motor neurons (soma napp; dendrite caL) — matches production NEURON."
        ),
        experimenter="MyoGen Simulation",
        institution="MyoGen Framework",
        lab="Neuromuscular Simulation",
        experiment_description=(
            f"5-second simulation with {naMN} motor neurons (NERLab), "
            f"{nIa} Ia afferents, {nII} II afferents, {nIb} Ib afferents, "
            f"and {ngII + ngIb} spinal interneurons."
        ),
        keywords=["MyoGen", "Jaxley", "spinal network", "motor neuron", "stretch reflex"],
        subject_id="simulated_subject_001",
        species="Homo sapiens",
        subject_description="Simulated human motor neuron pool - Jaxley backend",
    )
    print(f"\n(OK) Exported to NWB format: {nwb_filepath}")
except Exception as e:
    print(f"\n(Warning) NWB export failed: {e}")

##############################################################################
# Comprehensive Results Visualization
# ---------------------------------
#
# Create a series of plots that tell the complete story of spinal network
# function, from neural activity to mechanical output.
# Matches NEURON Example 11 visualization structure exactly.

print("\nGenerating comprehensive visualizations...")

# 1. NEURAL ACTIVITY: Raster plot showing all population spike patterns
populations_list = [
    "aMN",   # Motor output
    "Ia",
    "II",
    "Ib",    # Sensory input
    "gII",
    "gIb",   # Interneurons
    "DD",    # Descending drive
]
fig1, axes1 = plt.subplots(len(populations_list), 1, figsize=(15, 12))
plot_raster_spikes(
    results,
    axes1,
    populations=populations_list,
    time_range=(0, tstop),
    title="Spinal Network Activity (Single Muscle)",
)
plt.tight_layout()
plt.savefig(save_path / "neural_raster_plot_jaxley.png", dpi=150, bbox_inches="tight")
plt.show()

# 2. MOTOR NEURON DYNAMICS: Membrane potentials showing integration
fig2, ax2 = plt.subplots(1, 1, figsize=(15, 6))
plot_membrane_potentials(
    results,
    [ax2],
    populations=["aMN"],
    cell_indices=[0, 10, 20, 30, 40],
    time_range=(0, tstop),
    title="Motor Neuron Membrane Potentials",
)
plt.tight_layout()
plt.savefig(save_path / "membrane_potentials_jaxley.png", dpi=150, bbox_inches="tight")
plt.show()

# 3. MUSCLE MECHANICS: Muscle dynamics
fig3, axes3 = plt.subplots(5, 1, figsize=(15, 20))
plot_muscle_dynamics(
    results,
    artAng,
    time,
    axes3,
    muscle_name="hill_muscle",
    include_signals=["artAng", "L", "force", "torque"],
    include_activations=["TypeI", "TypeII"],
    normalize=True,
    time_range=(0, tstop),
    title="Muscle Dynamics - Length, Force, and Activation",
)
plt.tight_layout()
plt.savefig(save_path / "muscle_dynamics_jaxley.png", dpi=150, bbox_inches="tight")
plt.show()

# 4. PROPRIOCEPTIVE FEEDBACK: Spindle dynamics and sensory encoding
fig4, axes4 = plt.subplots(4, 1, figsize=(15, 16))
plot_spindle_dynamics(
    results,
    axes4,
    muscle_name="hill_muscle",
    include_signals=["L"],
    include_activations=["Bag1", "Bag2"],   # Chain is algebraic; not stored in scan state
    include_tensions=["Bag1", "Bag2", "Chain"],
    include_afferents=["Ia", "II"],
    time_range=(0, tstop),
    title="Muscle Spindle Dynamics - Proprioceptive Feedback System",
)
plt.tight_layout()
plt.savefig(save_path / "spindle_dynamics_jaxley.png", dpi=150, bbox_inches="tight")
plt.show()

# 5. FORCE FEEDBACK: GTO dynamics and protective reflexes
fig5, axes5 = plt.subplots(2, 1, figsize=(15, 8))
plot_gto_dynamics(
    results,
    axes5,
    muscle_name="hill_muscle",
    include_signals=["force", "Ib"],
    time_range=(0, tstop),
    title="Golgi Tendon Organ Dynamics - Force Feedback System",
)
plt.tight_layout()
plt.savefig(save_path / "gto_dynamics_jaxley.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n" + "=" * 60)
print("[DONE] Jaxley spinal network simulation complete!")
print("       Using NERLab motor neurons (napp + caL) — matches production NEURON.")
print("=" * 60)
