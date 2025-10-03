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

from myogen import load_nmodl_mechanisms
from myogen.simulator.neuron.network import Network
from myogen import simulator
from myogen.simulator.neuron.populations import (
    AlphaMN__Pool,
    DescendingDrive__Pool,
)
from myogen.simulator.neuron.simulation_runner import SimulationRunner

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
time = np.linspace(0, tstop, n_steps + 100)

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
DDorder = 1  # Poisson process order for realistic spike patterns

# Independent noise (IN) - one per motor neuron
nIN = naMN  # One independent Poisson noise source per motor neuron
INorder = 1  # Poisson process order for independent noise

rt, _ = simulator.RecruitmentThresholds(
    N=naMN,
    recruitment_range__ratio=100,
    mode="combined",
    deluca__slope=10,
    konstantin__max_threshold__ratio=1.0,
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

# Initialize with constant value
DDdrive = np.full_like(time, 65.0)

# Phase 2: 20 Hz sinusoid with DC=65, amplitude=20 from 60-120s
phase2_mask = (time_s >= 60) & (time_s < 120)
DDdrive[phase2_mask] = 65 + 20 * np.sin(2 * np.pi * 20 * time_s[phase2_mask])

# Phase 3: Same sinusoid from 120-180s but DC=58
phase3_mask = (time_s >= 120) & (time_s <= 180)
DDdrive[phase3_mask] = 58 + 20 * np.sin(2 * np.pi * 20 * time_s[phase3_mask])

# The time array now includes margin, so DDdrive is automatically padded
# Set extra points beyond 180s to maintain the last phase value (DC=58)
beyond_180_mask = time_s > 180
DDdrive[beyond_180_mask] = 58

plt.plot(time_s, DDdrive)
plt.xlabel("Time (s)")
plt.ylabel("Descending Drive (a.u.)")
plt.title("Descending Drive Pattern Over Time")
plt.grid()
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
DD = DescendingDrive__Pool(n=nDD, poisson_random_process_order=DDorder, timestep__ms=dt)
IN = DescendingDrive__Pool(n=nIN, poisson_random_process_order=INorder, timestep__ms=dt)

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
    step_counter,
    tstop,
):
    """
    Step-wise integration function called at each simulation timestep.

    Implements the Watanabe paper protocol by driving both descending drive (DD)
    and independent noise (IN) populations with the same external signal. The
    Poisson process nature of these populations ensures independent spike trains
    despite sharing the same input drive.

    Parameters
    ----------
    popD : dict
        Dictionary of neural populations ("DD", "IN", "aMN")
    ncD : dict
        Dictionary of network connections ("cmd->DD", "cmd->IN")
    DDdrive : array
        Time-varying drive signal (same for both DD and IN populations)
    step_counter : iterator
        Simulation step counter
    tstop : float
        Simulation stop time in ms
    """
    # Check if simulation time has exceeded the limit
    if h.t >= tstop:
        return  # Stop processing when simulation time limit reached

    i = next(step_counter)

    # DESCENDING DRIVE PROCESSING: Convert cortical signals to spikes
    # Both DD and IN populations receive the same drive signal but generate
    # independent spike trains due to their Poisson process nature
    if i < len(DDdrive):
        # Drive descending axons (DD) - shared across motor neurons
        for DDcell in popD["DD"]:
            if DDcell.integrate(DDdrive[i]):
                spike_time = h.t + 1
                if spike_time < tstop:
                    ncD["cmd->DD"][DDcell.pool__ID].event(spike_time)

        # Drive independent noise sources (IN) - one per motor neuron
        for INcell in popD["IN"]:
            if INcell.integrate(DDdrive[i]):
                spike_time = h.t + 1
                if spike_time < tstop:
                    ncD["cmd->IN"][INcell.pool__ID].event(spike_time)


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
# Configure Neural Connections
# ---------------------------
#
# Two types of inputs to motor neurons (per Watanabe paper):
# 1. Shared descending drive with sparse connectivity (30%)
# 2. Independent noise with one-to-one mapping
#
# Synaptic weight rationale:
# - DD→aMN: 0.05 μS × ~30 synapses/MN = ~1.5 μS total (shared drive)
# - IN→aMN: 0.025 μS × 1 synapse/MN = 0.025 μS (independent noise)
# - Total conductance per MN: ~1.5 μS (physiologically realistic range)

# Shared descending drive: DD → aMN (30% connectivity)
network.connect("DD", "aMN", probability=0.3, weight__μS=0.5)

# Independent noise: IN → aMN (one-to-one)
network.connect_one_to_one("IN", "aMN", probability=1.0, weight__μS=0.075)


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
# Prepare Simulation Models
# ------------------------
#


# Create step callback function with access to step counter
def step_callback(step_counter):
    return eachStep(
        popD=network.populations,
        ncD=ncD,
        DDdrive=DDdrive,
        step_counter=step_counter,
        tstop=tstop,
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

# Use implicit integration for better stability at larger dt
h.secondorder = 2  # Crank-Nicolson method (more stable than default)

print("\n✓ Using Crank-Nicolson integration (secondorder=2)")
print(f"   Fixed timestep: {dt} ms")
print(
    "   More stable than default forward Euler - allows larger dt without instability"
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

results = runner.run(
    duration__ms=tstop,
    timestep__ms=dt,
    membrane_recording={
        "aMN": list(range(0, naMN, 1)),
    },
)

print("Simulation completed successfully!")

# Save simulation results if we have them
if "results" in locals() and results is not None:
    joblib.dump(results, save_path / "spinal_network_results.pkl")
    print(f"✓ Results saved to {save_path}")
else:
    print("✗ No results to save")

##############################################################################
# Comprehensive Results Visualization
# ---------------------------------
#
# Create a series of plots that tell the complete story of spinal network
# function, from neural activity to mechanical output.

if "results" in locals() and results is not None:
    print(f"📊 Generating comprehensive visualizations...")

    # Import plotting utilities
    from myogen.utils.plotting import (
        plot_membrane_potentials,
        plot_raster_spikes,
    )

    # 1. NEURAL ACTIVITY: Raster plot showing all population spike patterns
    populations_list = ["aMN", "DD", "IN"]
    fig1, axes1 = plt.subplots(len(populations_list), 1, figsize=(15, 10))
    plot_raster_spikes(
        results,
        axes1,
        populations=populations_list,
        time_range=(0, tstop),
        title="Watanabe Paper Simulation - Motor Neurons with Shared DD and Independent Noise",
    )
    plt.tight_layout()
    plt.savefig(save_path / "watanabe_raster_plot.png", dpi=150, bbox_inches="tight")
    plt.show()

    # 2. MOTOR NEURON DYNAMICS: Membrane potentials showing integration
    fig2, axes2 = plt.subplots(1, 1, figsize=(15, 8))
    plot_membrane_potentials(
        results,
        [axes2],
        populations=["aMN"],
        cell_indices=[0, 10, 20, 30, 40, 50, 60, 70],
        time_range=(0, tstop),
        title="Motor Neuron Membrane Potentials - Watanabe Paper",
    )
    plt.tight_layout()
    plt.savefig(
        save_path / "watanabe_membrane_potentials.png", dpi=150, bbox_inches="tight"
    )
    plt.show()

    # 3. DESCENDING DRIVE PATTERN: Show the applied drive signal
    fig3, ax3 = plt.subplots(1, 1, figsize=(15, 6))
    ax3.plot(time_s, DDdrive, "b-", linewidth=2)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Descending Drive (Hz)")
    ax3.set_title("Descending Drive Pattern - Watanabe Paper Protocol")
    ax3.grid(True, alpha=0.3)
    ax3.axvline(x=60, color="r", linestyle="--", alpha=0.7, label="Phase 2 start")
    ax3.axvline(x=120, color="r", linestyle="--", alpha=0.7, label="Phase 3 start")
    ax3.legend()
    plt.tight_layout()
    plt.savefig(save_path / "watanabe_drive_pattern.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"✓ All visualizations completed and saved!")
else:
    print("⚠ No results available for visualization")

# %%
