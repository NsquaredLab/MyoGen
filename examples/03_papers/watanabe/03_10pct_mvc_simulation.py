"""
Spinal Network Simulation with Optimized DC Offset
============================================================

This example runs the Watanabe and Kohn (2015) spinal network simulation using:
- Constant drive (Phase 1) - baseline from script 01
- Constant + oscillation (Phase 2) - oscillation added to baseline
- Optimized DC + oscillation (Phase 3) - DC offset matches Phase 1 force

!!! note
    **Simulation Phases** (5 seconds each):

    - **Phase 1**: Constant drive (baseline, e.g., 40-65 Hz)
    - **Phase 2**: Constant + 20*sin(20Hz) (oscillation added to baseline)
    - **Phase 3**: Optimized DC + 20*sin(20Hz) (DC offset matches Phase 1 force)

!!! important
    **Prerequisites**:

    1. `01_compute_baseline_force.py` - Compute reference force
    2. `02_optimize_oscillating_dc.py` - Optimize DC offset

**Scientific Context**: Demonstrates how shared descending drive creates motor unit
synchronization, while independent noise reduces it. Phase 3 uses lower DC offset than
Phase 1 (e.g., 58 < 65) because oscillation contributes additional activation.

**MyoGen Components Used**:

- [`AlphaMN__Pool`][myogen.simulator.neuron.populations.AlphaMN__Pool]:
  Pool of 800 motor neurons. Created from recruitment thresholds that determine
  each unit's excitability and force contribution.

- [`DescendingDrive__Pool`][myogen.simulator.neuron.populations.DescendingDrive__Pool]:
  Two separate pools are used here:

  1. **DD (Descending Drive)**: 400 neurons with *shared* input to motor neurons
     (30% connectivity). This creates correlated activity across the motor pool.
  2. **IN (Independent Noise)**: 800 neurons with *one-to-one* connectivity.
     This decorrelates motor neuron activity. Driven at constant 125 Hz.

- [`Network`][myogen.simulator.Network]:
  Manages multiple populations and different connection patterns:

  - ``connect(prob=0.3)``: Random connectivity for shared drive
  - ``connect_one_to_one()``: Private inputs for independent noise
  - ``connect_from_external()``: External command signals to drive populations

- [`SimulationRunner`][myogen.simulator.SimulationRunner]:
  High-level interface for running NEURON simulations with callbacks.
  The ``step_callback`` is called every timestep to update drives and record data.

- [`ContinuousSaver`][myogen.utils.continuous_saver.ContinuousSaver]:
  Memory-efficient recording that saves data in chunks during long simulations.
  Essential for 15-second simulations with 800+ neurons.

**Key Concept**: The ratio of shared vs independent input determines motor unit
synchronization.  Watanabe and Kohn (2015) showed that 20 Hz cortical oscillations are
transmitted through the spinal network and can be detected in EMG coherence spectra.

**Workflow Position**: Step 3 of 6

**Next Step**: Run ``04_load_and_analyze_results.py`` to analyze the simulation output.
"""

# %%

##############################################################################
# Import Libraries
# ----------------

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import quantities as pq
from neuron import h

from myogen import get_random_generator, load_nmodl_mechanisms
from myogen.simulator import RecruitmentThresholds
from myogen.simulator.neuron.network import Network
from myogen.simulator.neuron.populations import AlphaMN__Pool, DescendingDrive__Pool
from myogen.simulator.neuron.simulation_runner import SimulationRunner
from myogen.utils.continuous_saver import ContinuousSaver

plt.style.use("fivethirtyeight")

##############################################################################
# Load NEURON Mechanisms
# ----------------------

load_nmodl_mechanisms()

# Setup results directory (use __file__ for directory-independent paths)
try:
    _script_dir = Path(__file__).parent
except NameError:
    _script_dir = Path.cwd()

save_path = _script_dir / "results"
save_path.mkdir(exist_ok=True)

# Setup continuous saving directory
chunks_path = save_path / "watanabe_chunks"
chunks_path.mkdir(exist_ok=True)

##############################################################################
# Load Optimized DC Offset
# -------------------------

dc_offset_file = _script_dir / "results" / "watanabe_optimization" / "dc_offset_optimized.json"

if not dc_offset_file.exists():
    raise FileNotFoundError(
        f"Optimized DC offset not found: {dc_offset_file}\nRun 02_optimize_oscillating_dc.py first!"
    )

with open(dc_offset_file) as f:
    dc_results = json.load(f)

# Extract drive values from optimization results
DD_DRIVE_CONSTANT = dc_results["reference_drive__Hz"]  # Constant drive from Phase 1
DC_OFFSET_OPTIMIZED = dc_results["dd_parameters"]["dc_offset__Hz"]  # Optimized for Phase 3

print("\n" + "=" * 70)
print("WATANABE SPINAL NETWORK SIMULATION")
print("=" * 70)
print(f"\nPhase 1 (Constant):     {DD_DRIVE_CONSTANT:.2f} Hz")
print(f"Phase 2 (Oscillating):  {DD_DRIVE_CONSTANT:.2f} + 20*sin(20Hz) Hz")
print(f"Phase 3 (Oscillating):  {DC_OFFSET_OPTIMIZED:.2f} + 20*sin(20Hz) Hz (optimized)")
print("\nOriginal Watanabe values:")
print("  Phase 1: 65 pps")
print("  Phase 2: 65 + 20*sin(20Hz) pps")
print("  Phase 3: 58 + 20*sin(20Hz) pps")
print(
    f"\nOptimization ratio: {DC_OFFSET_OPTIMIZED / DD_DRIVE_CONSTANT:.3f} (Watanabe: {58 / 65:.3f})"
)
print(f"Current setup uses {DD_DRIVE_CONSTANT:.1f} Hz baseline (configurable in script 11)")
print("=" * 70 + "\n")

##############################################################################
# Define Simulation Parameters
# ----------------------------

# Temporal parameters
dt = 0.025  # ms - Integration timestep
segment_duration__s = 5  # seconds - Duration of each phase
tstop = segment_duration__s * 3 * 1e3  # ms - Total simulation (3 phases)
n_steps = int(tstop / dt)
time = np.linspace(0, tstop, n_steps)

print("Simulation parameters:")
print(f"\tSegment duration: {segment_duration__s} s")
print(f"\tTotal duration: {tstop / 1e3} s ({tstop} ms)")
print(f"\tTimestep: {dt} ms")
print(f"\tTime samples: {len(time)}")

##############################################################################
# Define Neural Population Sizes
# -----------------------------

# Motor neurons
naMN = 800  # Watanabe specification

# Descending drive (shared)
nDD = 400  # Watanabe specification
DDorder = 16  # Poisson batch size

# Independent noise (one per MN)
nIN = naMN  # One per motor neuron
INorder = 16  # Poisson batch size

# Generate recruitment thresholds
rt, _ = RecruitmentThresholds(
    N=naMN, recruitment_range__ratio=100, mode="combined", deluca__slope=10
)

plt.plot(rt, "-o")
plt.xlabel("Motor Unit Index")
plt.ylabel("Recruitment Threshold (a.u.)")
plt.title("Recruitment Thresholds (800 Motor Units)")
plt.tight_layout()
plt.show()

##############################################################################
# Define Descending Drive Patterns (10% MVC Calibrated)
# -----------------------------------------------------

time_s = time / 1000.0  # Convert to seconds

# Descending drive (DD) - Watanabe baseline + optimized DC
DDdrive = np.full_like(time, DD_DRIVE_CONSTANT)

# Warm-up period (first 1 second at zero)
DDdrive[time_s < 1] = 0.0

# Phase 2: 20 Hz sinusoid with baseline DC (65 Hz)
phase2_mask = (time_s >= segment_duration__s) & (time_s < 2 * segment_duration__s)
DDdrive[phase2_mask] = DD_DRIVE_CONSTANT + 20 * np.sin(2 * np.pi * 20 * time_s[phase2_mask])

# Phase 3: 20 Hz sinusoid with optimized DC offset (matches Phase 1 force)
phase3_mask = time_s >= 2 * segment_duration__s
DDdrive[phase3_mask] = DC_OFFSET_OPTIMIZED + 20 * np.sin(2 * np.pi * 20 * time_s[phase3_mask])

# Independent noise (IN) - 125 Hz constant (Watanabe specification)
INdrive = 125.0 + get_random_generator().normal(0, 5.0, len(time))

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(time_s, DDdrive)
plt.xlabel("Time (s)")
plt.ylabel("DD Drive (Hz)")
plt.title("Descending Drive Pattern (Optimized)")
plt.axvline(segment_duration__s, color="r", linestyle="--", alpha=0.5, label="Phase boundary")
plt.axvline(2 * segment_duration__s, color="r", linestyle="--", alpha=0.5)
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(time_s, INdrive)
plt.xlabel("Time (s)")
plt.ylabel("IN Drive (Hz)")
plt.title("Independent Noise Drive (Constant 125 Hz)")
plt.grid()
plt.tight_layout()
plt.show()

##############################################################################
# Create Neural Populations
# -------------------------
#
# MyoGen provides several population types. Here we use:
#   - AlphaMN__Pool: Biophysically detailed motor neurons (Hodgkin-Huxley style)
#   - DescendingDrive__Pool: Poisson spike generators (no biophysics, just spikes)
#
# The key difference: AlphaMN neurons have membrane dynamics and integrate
# synaptic inputs, while DescendingDrive neurons just generate spikes at a
# specified rate - they're "virtual" neurons that provide input to the network.

# Alpha motor neurons - the actual motor neuron pool
# Each neuron's properties (input resistance, rheobase) depend on its recruitment
# threshold. Low-threshold units are more excitable and fire first.
aMN = AlphaMN__Pool(recruitment_thresholds__array=rt)

# Descending drive (DD) - SHARED input to motor neurons
# These represent corticospinal neurons that project to multiple motor neurons.
# poisson_batch_size=1 creates renewal (order-1) Poisson processes.
# Higher batch sizes would create more regular spike trains (order-k gamma).
DD = DescendingDrive__Pool(
    n=nDD,
    poisson_batch_size=1,  # Order 1 Poisson (Watanabe specification)
    timestep__ms=dt * pq.ms,
)

# Independent noise (IN) - PRIVATE input to each motor neuron
# These decorrelate motor neuron activity. Each IN neuron connects to exactly
# one motor neuron (one-to-one), providing uncorrelated background input.
IN = DescendingDrive__Pool(
    n=nIN,
    poisson_batch_size=1,  # Order 1 Poisson (Watanabe specification)
    timestep__ms=dt * pq.ms,
)

##############################################################################
# Define Callback Functions
# -------------------------


def eachStep(
    popD,
    ncD,
    DDdrive,
    INdrive,
    step_counter,
    tstop,
    continuous_saver,
    timestep__ms,
):
    """
    Step-wise integration callback - called every simulation timestep.

    This is where we:
    1. Update the Poisson generators with current drive rate
    2. Deliver spikes to motor neurons via NetCons
    3. Record membrane potentials and spike times

    The key method is cell.integrate(rate_Hz):
        - Advances the internal Poisson process by one timestep
        - Returns True if a spike occurred, False otherwise
        - The spike probability depends on rate_Hz and timestep

    When a spike occurs, we schedule an event via NetCon.event(time).
    This triggers synaptic conductance in the target motor neuron.
    """
    if h.t >= tstop:
        return

    i = next(step_counter)

    # Drive descending drive (DD) and independent noise (IN)
    if i < len(DDdrive):
        # DD: Time-varying drive (constant, or DC + sinusoid depending on phase)
        # Each DD neuron independently generates spikes at rate DDdrive[i]
        for DDcell in popD["DD"]:
            if DDcell.integrate(DDdrive[i]):  # Returns True if spike occurred
                spike_time = h.t + 1  # Schedule 1ms in future (synaptic delay)
                if spike_time < tstop:
                    # Deliver spike to all motor neurons connected to this DD cell
                    ncD["cmd->DD"][DDcell.pool__ID].event(spike_time)

        # IN: Constant 125 Hz independent noise
        # Each IN neuron connects to exactly one motor neuron (one-to-one)
        for INcell in popD["IN"]:
            if INcell.integrate(INdrive[i]):
                spike_time = h.t + 1
                if spike_time < tstop:
                    ncD["cmd->IN"][INcell.pool__ID].event(spike_time)

    # Record membrane potentials and spikes for this timestep
    continuous_saver.record_step(timestep__ms)


##############################################################################
# Create Neural Network
# ---------------------
#
# The Network class is a container that manages:
#   1. Multiple neural populations (referenced by string keys)
#   2. Synaptic connections between populations
#   3. External inputs (e.g., command signals)
#
# The dictionary keys ("DD", "IN", "aMN") become the identifiers used
# when creating connections.

network = Network({"DD": DD, "IN": IN, "aMN": aMN})

##############################################################################
# Configure Neural Connections
# ----------------------------
#
# MyoGen supports several connection patterns:
#
#   network.connect(src, tgt, probability=p)
#       Random connectivity: each target neuron receives input from ~p*N_src
#       source neurons. Used for SHARED inputs (creates correlation).
#
#   network.connect_one_to_one(src, tgt)
#       Each source connects to exactly one target (requires N_src == N_tgt).
#       Used for PRIVATE inputs (decorrelates activity).
#
#   network.connect_from_external(name, tgt)
#       Creates NetCon objects for external event delivery (e.g., commands).
#
# The ratio of shared (DD) to private (IN) input determines the degree of
# motor unit synchronization - a key parameter in the original analysis.

# DD connectivity: 30% probability (original specification)
# This means each motor neuron receives input from ~30% of DD neurons
DD_connectivity = 0.3
expected_DD_per_MN = int(DD_connectivity * nDD)  # ~120 connections per MN

# Target conductances - total synaptic drive to each motor neuron
target_total_DD_conductance__uS = 0.1 * pq.uS
target_IN_conductance__uS = 0.05 * pq.uS

# Scale DD weight by convergence (each synapse contributes 1/N of total)
weight_DD__uS = target_total_DD_conductance__uS / expected_DD_per_MN

print("\nSynaptic Weight Scaling:")
print(f"\tDD connections per MN: {expected_DD_per_MN}")
print(f"\tDD weight per connection: {weight_DD__uS:.6f} uS")
print(f"\tTotal DD conductance per MN: {weight_DD__uS * expected_DD_per_MN:.3f} uS")
print(f"\tIN conductance per MN: {target_IN_conductance__uS:.3f} uS")

# SHARED descending drive: DD → aMN (30% random connectivity)
# Creates correlated input across motor neurons - the source of synchronization
network.connect(
    "DD",
    "aMN",
    probability=DD_connectivity,
    weight__uS=target_total_DD_conductance__uS,
)

# PRIVATE independent noise: IN → aMN (one-to-one connectivity)
# Each motor neuron gets input from exactly one IN neuron - decorrelates activity
network.connect_one_to_one("IN", "aMN", probability=1.0, weight__uS=target_IN_conductance__uS)

##############################################################################
# Configure External Inputs
# -------------------------
#
# External inputs are how we deliver "command" events to drive the network.
# connect_from_external() creates NetCon objects that can deliver events to
# the target population. We then retrieve these NetCons and use them in the
# simulation loop to trigger spikes in DD and IN populations.
#
# Note: The weight here (1.0 uS) doesn't matter for DescendingDrive neurons
# since they just use the event as a trigger, not as synaptic input.

network.connect_from_external("cmd", "DD", weight__uS=1.0 * pq.uS)
network.connect_from_external("cmd", "IN", weight__uS=1.0 * pq.uS)

# Get NetCon dictionaries for event delivery in the simulation loop
# These map pool_ID → NetCon for each neuron in the population
ncD = {
    "cmd->DD": network.get_netcons("cmd", "DD"),
    "cmd->IN": network.get_netcons("cmd", "IN"),
}

##############################################################################
# Setup Continuous Saving
# -----------------------

# Record sample of neurons (every 5th neuron)
recording_neurons = list(range(0, naMN, 5))
chunk_duration_ms = 5000.0 * pq.ms

continuous_saver = ContinuousSaver(
    save_path=chunks_path,
    chunk_duration__ms=chunk_duration_ms,
    populations=network.populations,
    recording_config={"aMN": recording_neurons},
)

print("\nContinuous saving configured:")
print(f"\tRecording {len(recording_neurons)} neurons (sampled)")
print(f"\tChunk duration: {chunk_duration_ms / 1000:.1f} seconds")

##############################################################################
# Prepare Simulation
# ------------------


def step_callback(step_counter):
    return eachStep(
        popD=network.populations,
        ncD=ncD,
        DDdrive=DDdrive,
        INdrive=INdrive,
        step_counter=step_counter,
        tstop=tstop,
        continuous_saver=continuous_saver,
        timestep__ms=dt,
    )


##############################################################################
# Integration Method
# ------------------

# Use Crank-Nicolson integration (Watanabe used RK4, this is second-order)
h.secondorder = 2

print("\nUsing Crank-Nicolson integration (secondorder=2)")
print(f"Fixed timestep: {dt} ms")

##############################################################################
# Run Simulation
# --------------

print("\nStarting 10% MVC calibrated spinal network simulation...")
print(f"\tDuration: {tstop} ms")
print(f"\tTimestep: {dt} ms")
print(f"\tPopulations: {len(network.populations)}")

runner = SimulationRunner(
    network=network,
    models={},
    step_callback=step_callback,
)

print("\nRunning simulation with continuous saving...")

results = runner.run(
    duration__ms=tstop * pq.ms,
    timestep__ms=dt * pq.ms,
    membrane_recording=None,  # Continuous saver handles recording
)
print("Simulation completed successfully!")

# Finalize continuous saving
print("\nFinalizing continuous data saving...")
continuous_saver.finalize(timestep__ms=dt * pq.ms, spike_results=results)

# Save simulation parameters
print("\nSaving simulation parameters...")
simulation_params = {
    "segment_duration__s": segment_duration__s,
    "tstop__ms": tstop,
    "dt__ms": dt,
    "n_steps": n_steps,
    "naMN": naMN,
    "nDD": nDD,
    "nIN": nIN,
    "DD_drive_constant__Hz": DD_DRIVE_CONSTANT,
    "DC_offset_optimized__Hz": DC_OFFSET_OPTIMIZED,
    "optimization_ratio": DC_OFFSET_OPTIMIZED / DD_DRIVE_CONSTANT,
}
joblib.dump(simulation_params, save_path / "watanabe_simulation_params.pkl")
print(f"Simulation parameters saved to {save_path / 'watanabe_simulation_params.pkl'}")

# Save spike results
if results is not None:
    print("\nSaving spike data...")
    joblib.dump(results, save_path / "watanabe_spikes.pkl")
    print(f"Spike data saved to {save_path / 'watanabe_spikes.pkl'}")

print("\n" + "=" * 70)
print("SIMULATION COMPLETE")
print("=" * 70)
print(f"\nData saved to:")
print(f"  Chunks: {chunks_path}/")
print(f"  Parameters: {save_path / 'watanabe_simulation_params.pkl'}")
print(f"  Spikes: {save_path / 'watanabe_spikes.pkl'}")
print(f"\nPhase 1 constant drive: {DD_DRIVE_CONSTANT:.2f} Hz")
print(f"Phase 3 DC offset: {DC_OFFSET_OPTIMIZED:.2f} Hz (optimized to match Phase 1 force)")
print(f"Ratio: {DC_OFFSET_OPTIMIZED / DD_DRIVE_CONSTANT:.3f} (Watanabe: {58 / 65:.3f})")

# mkdocs_gallery_thumbnail_path = "gallery_thumbs/03_10pct_mvc_simulation.png"
