"""
Paper Watanabe
=====================================================
"""
##############################################################################
# Import Libraries
# ----------------
#

# %%

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from neuron import h

from myogen import load_nmodl_mechanisms, RANDOM_GENERATOR
from myogen.simulator.neuron.network import Network
from myogen import simulator
from myogen.simulator.neuron.populations import AlphaMN__Pool, DescendingDrive__Pool
from myogen.simulator.neuron.simulation_runner import SimulationRunner

# Import continuous saving utilities
from continuous_saver import (
    ContinuousSaver,
    load_and_combine_chunks,
    convert_chunks_to_neo,
)

##############################################################################
# Load NEURON Mechanisms and Dependencies
# ---------------------------------------
#
# Load the compiled NMODL mechanisms required for biophysical neuron modeling
# and load results from previous examples that serve as inputs to this simulation.

# Load NEURON mechanisms
load_nmodl_mechanisms()

# Setup results directory
save_path = Path("./results")
save_path.mkdir(exist_ok=True)

# Setup continuous saving directory (subdirectory for chunks)
chunks_path = save_path / "watanabe_chunks"
chunks_path.mkdir(exist_ok=True)

##############################################################################
# Define Simulation Parameters
# ---------------------------
#
# These parameters control the temporal and spatial resolution of the simulation,
# as well as the physiological characteristics of the neural populations and
# mechanical system.

# Temporal parameters - high resolution for accurate neural integration
dt = 0.025  # ms - Integration timestep
tstop = 180 * 1e3  # ms - Total simulation duration
n_steps = int(tstop / dt)
# Add margin for NEURON's potential overstep
time = np.linspace(0, tstop, n_steps)

print("Simulation parameters:")
print(f"  - Duration: {tstop} ms")
print(f"  - Timestep: {dt} ms")
print(f"  - Time samples: {len(time)}")

##############################################################################
# Define Neural Population Sizes
# -----------------------------
#
# Population sizes are based on physiological estimates from cat and human studies.
# These numbers represent typical motor pool compositions for a single muscle.

# Motor neurons (output to muscles)
naMN = 100

# Descending drive (cortical input) - shared across motor neurons
nDD = 400  # Total descending drive neurons (30% connectivity to each MN)
DDorder = 16  # Poisson batch size for threshold generation (higher = better statistics)

# Independent noise (IN) - one per motor neuron
nIN = naMN  # One independent Poisson noise source per motor neuron
INorder = 16  # Poisson batch size for threshold generation

# Watanabe paper specifies: mean ISI = 8 ms → firing rate = 125 Hz
# This firing rate is controlled by the DDdrive signal (see eachStep callback below)

rt, _ = simulator.RecruitmentThresholds(
    N=naMN, recruitment_range__ratio=100, mode="combined", deluca__slope=10
)


plt.plot(rt, "-o")
plt.xlabel("Motor Unit Index")
plt.ylabel("Recruitment Threshold (a.u.)")
plt.title("Combined Model Recruitment Thresholds")
plt.grid()
plt.show()

##############################################################################
# Define Descending Drive Patterns
# -------------------------------
#
# Create DD drive pattern as specified:
# - Constant 65 until 60s
# - 20 Hz sinusoid with DC 65, amplitude 20 from 60-120s
# - Same 20 Hz sinusoid from 120-180s but DC reduced to 58
time_s = time / 1000.0  # Convert to seconds for easier calculation

# Descending drive (DD) - modulated signal
DDdrive = np.full_like(time, 65.0)

# Phase 2: 20 Hz sinusoid with DC=65, amplitude=20 from 60-120s
phase2_mask = (time_s >= 60) & (time_s < 120)
DDdrive[phase2_mask] = 65 + 20 * np.sin(2 * np.pi * 20 * time_s[phase2_mask])

# Phase 3: Same sinusoid from 120-180s but DC reduced to 58
phase3_mask = (time_s >= 120) & (time_s <= 180)
DDdrive[phase3_mask] = 58 + 20 * np.sin(2 * np.pi * 20 * time_s[phase3_mask])

# Independent noise (IN) - 125 Hz with random variation
# ±1 ms ISI variation: 1/(8-1)=143 Hz to 1/(8+1)=111 Hz, so ±16 Hz range
# Using smaller random noise for smoother variation
INdrive = 125.0 + RANDOM_GENERATOR.normal(0, 5.0, len(time))  # 125 Hz ± ~5 Hz (σ)

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(time_s, DDdrive)
plt.xlabel("Time (s)")
plt.ylabel("DD Drive (Hz)")
plt.title("Descending Drive Pattern (Modulated)")
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(time_s, INdrive)
plt.xlabel("Time (s)")
plt.ylabel("IN Drive (Hz)")
plt.title("Independent Noise Drive (Constant)")
plt.grid()
plt.tight_layout()
plt.show()

##############################################################################
# Create Neural Populations
# ------------------------
#
# Create three neural populations:
# 1. Alpha motor neurons (aMN) - the output neurons controlling muscle
# 2. Descending drive (DD) - 400 axons with 30% connectivity to each MN
# 3. Independent noise (IN) - one-to-one Poisson noise source per MN

aMN = AlphaMN__Pool(recruitment_thresholds__array=rt)
DD = DescendingDrive__Pool(n=nDD, poisson_batch_size=DDorder, timestep__ms=dt)
IN = DescendingDrive__Pool(n=nIN, poisson_batch_size=INorder, timestep__ms=dt)

##############################################################################
# Define Callback Functions
# ------------------------
#
# These functions handle the integration of different system components during
# the simulation, including spike events and step-wise updates.


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
    Step-wise integration function called at each simulation timestep.

    Implements the Watanabe paper protocol by driving descending drive (DD)
    with a modulated signal and independent noise (IN) with a constant signal.
    The Poisson process nature of these populations ensures independent spike trains.

    Parameters
    ----------
    popD : dict
        Dictionary of neural populations ("DD", "IN", "aMN")
    ncD : dict
        Dictionary of network connections ("cmd->DD", "cmd->IN")
    DDdrive : array
        Time-varying modulated drive signal for descending drive (DD) population (Hz)
    INdrive : array
        Constant drive signal for independent noise (IN) population (Hz)
    step_counter : iterator
        Simulation step counter
    tstop : float
        Simulation stop time in ms
    continuous_saver : ContinuousSaver
        Continuous data saver instance
    timestep__ms : float
        Integration timestep in milliseconds
    """
    # Check if simulation time has exceeded the limit
    if h.t >= tstop:
        return  # Stop processing when simulation time limit reached

    i = next(step_counter)

    # DESCENDING DRIVE PROCESSING: Convert cortical signals to spikes
    # DD receives modulated drive, IN receives constant drive (both generate
    # independent spike trains due to their Poisson process nature)
    if i < len(DDdrive):
        # Drive descending axons (DD) - shared across motor neurons with modulated signal
        for DDcell in popD["DD"]:
            if DDcell.integrate(DDdrive[i]):
                spike_time = h.t + 1
                if spike_time < tstop:
                    ncD["cmd->DD"][DDcell.pool__ID].event(spike_time)

        # Drive independent noise sources (IN) - one per motor neuron with constant signal
        for INcell in popD["IN"]:
            if INcell.integrate(INdrive[i]):
                spike_time = h.t + 1
                if spike_time < tstop:
                    ncD["cmd->IN"][INcell.pool__ID].event(spike_time)

    # CONTINUOUS SAVING: Record data for this timestep
    continuous_saver.record_step(timestep__ms)


##############################################################################
# Create Neural Network
# --------------------
#
# Assemble all neural populations into a connected network that implements
# the Watanabe paper architecture:
# - DD (400) → aMN (100) with 30% probability (shared descending drive)
# - IN (100) → aMN (100) one-to-one (independent noise per MN)

network = Network({"DD": DD, "IN": IN, "aMN": aMN})

##############################################################################
# Configure Neural Connections with Scaled Synaptic Weights
# ---------------------------------------------------------
#
# Two types of inputs to motor neurons (per Watanabe paper):
# 1. Shared descending drive with sparse connectivity (30%)
# 2. Independent noise with one-to-one mapping
#
# CRITICAL: Synaptic weights must be scaled by the number of converging connections
# to avoid over-excitation. With 30% connectivity, each MN receives ~120 DD inputs.

# Calculate scaled weights
DD_connectivity = 0.3
expected_DD_per_MN = int(DD_connectivity * nDD)  # 120 connections per MN

# Target total conductances per MN (physiologically realistic)
target_total_DD_conductance__μS = 0.2  # Total DD conductance per MN
target_IN_conductance__μS = 0.1

# Scale DD weight by number of converging connections
weight_DD__μS = target_total_DD_conductance__μS / expected_DD_per_MN

print(f"\nSynaptic Weight Scaling:")
print(f"  DD connections per MN: {expected_DD_per_MN}")
print(f"  DD weight per connection: {weight_DD__μS:.6f} μS")
print(f"  Total DD conductance per MN: {weight_DD__μS * expected_DD_per_MN:.3f} μS")
print(f"  IN conductance per MN: {target_IN_conductance__μS:.3f} μS")
print(
    f"  DD/IN ratio: {(weight_DD__μS * expected_DD_per_MN) / target_IN_conductance__μS:.1f}x"
)

network.connect(
    "DD",
    "aMN",
    probability=DD_connectivity,
    weight__μS=target_total_DD_conductance__μS,
)

# Independent noise: IN → aMN (one-to-one)
network.connect_one_to_one(
    "IN", "aMN", probability=1.0, weight__μS=target_IN_conductance__μS
)


##############################################################################
# Configure External Inputs
# ------------------------
#
# Setup external input pathways for both descending drive and independent noise.
# Both populations receive the same external drive signal but generate independent
# spike trains due to their Poisson process nature.

network.connect_from_external("cmd", "DD", weight__μS=1.0)
network.connect_from_external("cmd", "IN", weight__μS=1.0)

ncD = {
    "cmd->DD": network.get_netcons("cmd", "DD"),
    "cmd->IN": network.get_netcons("cmd", "IN"),
}

##############################################################################
# Setup Continuous Saving
# ----------------------
#
# Initialize continuous saver to prevent memory overflow during long simulation.
# Data will be saved in chunks every 10 seconds of simulation time.

# Record ALL motor neurons (full resolution)
recording_neurons = list(range(naMN))  # All 400 neurons

# Use smaller chunks (50s) to keep peak RAM manageable with full recording
# 50 seconds × 400 neurons × 200,000 timesteps × 8 bytes ≈ 640 MB per chunk
chunk_duration_ms = 5000.0

continuous_saver = ContinuousSaver(
    save_path=chunks_path,
    chunk_duration__ms=chunk_duration_ms,
    populations=network.populations,
    recording_config={"aMN": recording_neurons},
)

print(f"\nContinuous saving configured:")
print(f"  Recording {len(recording_neurons)} neurons (ALL)")
print(f"  Chunk duration: {chunk_duration_ms / 1000:.1f} seconds")
print(f"  Estimated max RAM per chunk: ~640 MB")
print(f"  Total chunks expected: {int(tstop / chunk_duration_ms)}")

##############################################################################
# Prepare Simulation Models
# ------------------------
#


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
# Enable Adaptive Integration for Speed
# -------------------------------------
#
# CVode provides variable timestep integration that automatically reduces dt
# during fast events (spikes) and increases it during slow periods. This provides
# 3-5x speedup while maintaining numerical stability and accuracy guarantees.
#
# How it works:
# - During action potentials: dt drops to ~0.001 ms (fine resolution)
# - During subthreshold integration: dt increases to ~0.05-0.1 ms (coarse resolution)
# - Error tolerances ensure accuracy is maintained throughout
#
# Tolerances:
# - atol (absolute): Maximum allowed absolute error in voltage (mV)
# - rtol (relative): Maximum allowed relative error (fraction)
# - Lower values = more accurate but slower; current values are conservative

# CVode adaptive integration - DISABLED for this simulation
#
# Reason: The step callback loops through 200 neurons (DD + IN) which gets
# called at every adaptive timestep. With CVode taking very small steps during
# spikes (dt ~ 0.0001 ms), the callback overhead dominates and slows everything down.
#
# Alternative speedup strategies:
# 1. Use larger fixed dt (0.01-0.025 ms) with secondorder=2 (Crank-Nicolson)
# 2. Reduce callback frequency by only updating drive every N steps
# 3. Move Poisson integration into NEURON mechanisms instead of Python loops

# cv = h.CVode()
# cv.active(1)
# cv.atol(1e-4)
# cv.rtol(1e-3)

# Use second-order Crank-Nicolson integration (closest to Watanabe paper's RK4)
# Note: NEURON does not have built-in RK4 for main simulation loop
# Watanabe paper used fourth-order Runge-Kutta, but NEURON's secondorder=2
# (Crank-Nicolson) provides second-order accuracy which is a reasonable approximation
h.secondorder = 2  # Crank-Nicolson method (second-order accurate)

print("\nUsing Crank-Nicolson integration (secondorder=2)")
print(f"Fixed timestep: {dt} ms")
print(
    "   Note: Watanabe paper used RK4; NEURON's Crank-Nicolson provides 2nd-order accuracy"
)

##############################################################################
# Run Spinal Network Simulation
# ----------------------------
#
# Execute the complete simulation with all integrated components.
# With CVode enabled, the 'dt' parameter becomes the maximum timestep -
# NEURON will use smaller steps automatically during fast dynamics.

print("\nStarting spinal network simulation...")
print(f"   Duration: {tstop} ms")
print(f"   Max timestep (CVode): {dt} ms")
print(f"   Populations: {len(network.populations)}")

runner = SimulationRunner(
    network=network,
    models={},
    step_callback=step_callback,
)

# Motor neuron spike recording thresholds are now fixed in the Network class

# CONTINUOUS SAVING MODE: No membrane recording via SimulationRunner
# Data is recorded continuously via the ContinuousSaver in the step callback
# This keeps peak RAM usage below 100 MB regardless of simulation duration
print("\nRunning simulation with continuous saving (low memory mode)...")
print("  Recording is handled by ContinuousSaver")
print("  Peak RAM usage: < 100 MB")

results = runner.run(
    duration__ms=tstop,
    timestep__ms=dt,
    membrane_recording=None,  # Continuous saver handles recording
)
print("Simulation completed successfully!")

# Finalize continuous saving (save last chunk and ALL spike data from SimulationRunner)
print("\nFinalizing continuous data saving...")
continuous_saver.finalize(timestep__ms=dt, spike_results=results)

# Also save spike results separately for backup/compatibility
if results is not None:
    print("\nSaving backup spike data...")
    joblib.dump(results, save_path / "watanabe__spikes_only.pkl")
    print(f"✓ Backup spike data saved to {save_path / 'watanabe__spikes_only.pkl'}")
else:
    print("✗ No spike results from SimulationRunner")

print(f"\n" + "=" * 60)
print("SIMULATION COMPLETE - Data saved in chunks")
print("=" * 60)
print(f"Chunks directory: {chunks_path}")
print(f"\nTo load as NEO Block (SELF-CONTAINED - all data in chunks):")
print(f"  from continuous_saver import convert_chunks_to_neo")
print(f"  from pathlib import Path")
print(f"  results = convert_chunks_to_neo(Path('{chunks_path}'))")
print(f"  # Now 'results' contains ALL data from chunks:")
print(f"  #   - Spike trains: aMN, DD, IN")
print(f"  #   - Membrane potentials: aMN (all 400 neurons)")
print(f"\nAlternatively, to load raw chunks:")
print(f"  from continuous_saver import load_and_combine_chunks")
print(f"  data = load_and_combine_chunks(Path('{chunks_path}'))")
print("=" * 60)
