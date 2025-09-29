"""
Paper Watanabe
=====================================================
"""
##############################################################################
# Import Libraries
# ----------------
#

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from neuron import h

from myogen import load_nmodl_mechanisms
from myogen.simulator.neuron.network import Network
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
dt = 0.1  # ms - Integration timestep
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
naMN = 11

# Descending drive (cortical input)
nDD = 400  # Total descending drive neurons
DDorder = 1  # Poisson process order for realistic spike patterns

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

aMN = AlphaMN__Pool(n=naMN)
DD = DescendingDrive__Pool(n=nDD, poisson_random_process_order=DDorder, timestep__ms=dt)

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
    tstop,  # Add tstop as parameter
):
    """
    Step-wise integration function called at each simulation timestep.

    This function orchestrates the complex interactions between:
    - Muscle mechanics and force generation
    - Proprioceptive feedback from spindles and GTOs
    - Joint dynamics and closed-loop control
    - Neural population dynamics and spike generation

    Parameters
    ----------
    muscle_flex, muscle_ext : HillModel
        Flexor and extensor muscle models
    spin : SpindleModel
        Muscle spindle model
    golgi : GolgiTendonOrganModel
        Golgi tendon organ model
    popD : dict
        Dictionary of neural populations
    ncD : dict
        Dictionary of network connections
    gMN : dict
        Fusimotor drive parameters
    joint_dyn : JointDynamics
        Joint biomechanics model
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
    if i < len(DDdrive):
        for DDcell in popD["DD"]:
            if DDcell.integrate(DDdrive[i]):
                spike_time = h.t + 1
                if spike_time < tstop:
                    ncD["cmd->DD"][DDcell.pool__ID].event(spike_time)


##############################################################################
# Create Neural Network
# --------------------
#
# Assemble all neural populations into a connected network that implements
# the spinal reflex circuitry for antagonist muscle control.

network = Network({"DD": DD, "aMN": aMN})

##############################################################################
# Configure Neural Connections
# ---------------------------
#

network.connect("DD", "aMN", probability=0.3, weight__μS=1.0)


##############################################################################
# Configure External Inputs
# ------------------------
#
# Setup external input pathways for sensory feedback and descending commands.

network.connect_from_external("cmd", "DD", weight__μS=1.0)
ncD = {"cmd->DD": network.get_netcons("cmd", "DD")}

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
# Run Spinal Network Simulation
# ----------------------------
#
# Execute the complete simulation with all integrated components.

print("\nStarting spinal network simulation...")
print(f"   Duration: {tstop} ms")
print(f"   Timestep: {dt} ms")
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
        "aMN": [0, 5, 10],
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
    print(f"\n📊 Generating comprehensive visualizations...")

    # Import plotting utilities
    from myogen.utils.plotting import (
        plot_membrane_potentials,
        plot_raster_spikes,
    )

    # 1. NEURAL ACTIVITY: Raster plot showing motor neuron spike patterns
    populations_list = ["aMN", "DD"]
    fig1, axes1 = plt.subplots(len(populations_list), 1, figsize=(15, 8))
    plot_raster_spikes(
        results,
        axes1,
        populations=populations_list,
        time_range=(0, tstop),
        title="Motor Neuron Pool Activity - Watanabe Paper Simulation",
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
