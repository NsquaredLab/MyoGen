"""
Spinal Network Simulation with Proprioceptive Feedback
=====================================================

This example demonstrates **complete spinal reflex network modeling** with **antagonist muscle control**,
**proprioceptive feedback**, and **closed-loop joint dynamics**. This represents the culmination of the
MyoGen simulation pipeline, integrating all previous components into a biologically realistic motor control system.

.. note::
    This example builds upon all previous examples and demonstrates how the complete neuromuscular system
    functions as an integrated network:

    - **Motor neuron pools**: Flexor and extensor α-motoneurons (from examples 01-02)
    - **Muscle models**: Hill-type muscle models with realistic force generation (from examples 02, 06)
    - **Proprioceptive feedback**: Muscle spindles and Golgi tendon organs (from examples)
    - **Spinal interneurons**: Group II and Group Ib interneurons for reflex modulation
    - **Joint dynamics**: Closed-loop biomechanical control with realistic inertia and damping
    - **Descending drive**: Cortical control signals for voluntary movement (from example 01)

.. important::
    **Spinal Reflex Networks** are the fundamental control circuits that coordinate muscle activity.
    Key physiological concepts demonstrated:

    - **Reciprocal inhibition**: Flexor activation inhibits extensors and vice versa
    - **Stretch reflex**: Muscle spindles provide length/velocity feedback for posture control
    - **Force feedback**: Golgi tendon organs monitor muscle tension for protective reflexes
    - **Closed-loop control**: Joint mechanics influence neural activity through proprioception
    - **Antagonist coordination**: Balanced control of opposing muscle groups

Scientific Background
-------------------

The spinal cord contains intricate neural circuits that process sensory information and generate appropriate
motor responses. This example models several key components:

**Muscle Spindles**: Specialized sensory organs that detect muscle length and velocity changes, providing
Group Ia (velocity-sensitive) and Group II (length-sensitive) afferent signals.

**Golgi Tendon Organs**: Force-sensitive receptors that monitor muscle tension and provide Group Ib
afferent feedback for protective reflexes.

**Spinal Interneurons**: Group II interneurons process spindle feedback, while Group Ib interneurons
handle force feedback, both contributing to reflex modulation.

**Joint Dynamics**: Realistic biomechanical modeling of joint inertia, damping, and muscle-generated torques
creates the closed-loop system where neural activity affects movement, which in turn affects sensory feedback.
"""

# %%

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
from myogen.simulator.neuron.joint_dynamics import JointDynamics
from myogen.simulator.neuron.muscle import HillModel
from myogen.simulator.neuron.network import Network
from myogen.simulator.neuron.populations import (
    AffIa__Pool,
    AffIb__Pool,
    AffII__Pool,
    AlphaMN__Pool,
    DescendingDrive__Pool,
    GIb__Pool,
    GII__Pool,
)
from myogen.simulator.neuron.proprioception import (
    GolgiTendonOrganModel,
    SpindleModel,
)
from myogen.simulator.neuron.simulation_runner import SimulationRunner
from myogen.utils.plotting import (
    plot_antagonist_muscle_comparison,
    plot_gto_dynamics,
    plot_membrane_potentials,
    plot_muscle_dynamics,
    plot_raster_spikes,
    plot_spindle_dynamics,
)

import quantities as pq

##############################################################################
# Load NEURON Mechanisms and Dependencies
# ---------------------------------------
#
# Load the compiled NMODL mechanisms required for biophysical neuron modeling
# and load results from previous examples that serve as inputs to this simulation.

# Load NEURON mechanisms
load_nmodl_mechanisms()

# Setup results directory
save_path = Path(r"/home/oj98yqyk/code/simulators/MyoGen/examples/basic/results")
save_path.mkdir(exist_ok=True)

recruitment_thresholds = joblib.load(save_path / "thresholds.pkl")
print("✓ Loaded recruitment thresholds from example 00")

##############################################################################
# Define Simulation Parameters
# ---------------------------
#
# These parameters control the temporal and spatial resolution of the simulation,
# as well as the physiological characteristics of the neural populations and
# mechanical system.

# Temporal parameters - high resolution for accurate neural integration
dt = 0.005 * pq.ms  # Integration timestep with units
tstop = 2e3  # ms - Total simulation duration
time = np.arange(0, tstop, dt.magnitude)  # Time vector (use magnitude for numpy)

print("Simulation parameters:")
print(f"\tDuration: {tstop} ms")
print(f"\tTimestep: {dt} ms")
print(f"\tTime samples: {len(time)}")

##############################################################################
# Define Neural Population Sizes
# -----------------------------
#
# Population sizes are based on physiological estimates from cat and human studies.
# These numbers represent typical motor pool compositions for a single muscle.

# Afferent populations (sensory input)
nIa = 100  # Group Ia afferents from muscle spindles (velocity-sensitive)
nII = 25  # Group II afferents from muscle spindles (length-sensitive)
nIb = 25  # Group Ib afferents from Golgi tendon organs (force-sensitive)

# Interneuron populations (spinal processing)
ngII = 25  # Group II interneurons (process spindle feedback)
ngIb = 25  # Group Ib interneurons (process force feedback)

# Motor neurons (output to muscles)
nType1 = 102  # Type I motor neurons (slow, fatigue-resistant)
nType2 = 18  # Type II motor neurons (fast, fatigue-prone)
naMN = nType1 + nType2  # Total α-motoneurons per muscle

# Descending drive (cortical input)
nDD = 400  # Total descending drive neurons
DDorder = 1  # Poisson process order for realistic spike patterns

print("Neural population sizes:")
print(f"  - α-Motoneurons: {naMN} ({nType1} Type I, {nType2} Type II)")
print(f"  - Ia afferents: {nIa}")
print(f"  - II afferents: {nII}")
print(f"  - Ib afferents: {nIb}")
print(f"  - Interneurons: {ngII + ngIb}")
print(f"  - Descending drive: {nDD}")

##############################################################################
# Define Descending Drive Patterns
# -------------------------------
#
# Create antagonist drive patterns that demonstrate reciprocal activation.
# Flexor and extensor muscles receive complementary activation patterns to
# simulate realistic voluntary movement commands from the motor cortex.

# Flexor drive: smooth ramp up and sustained activation
DDdrive_flexor = np.interp(time, [0, tstop // 4, tstop // 2, tstop], [20, 60, 20, 20])

# Extensor drive: complementary pattern (low when flexor high)
DDdrive_extensor = np.interp(time, [0, tstop // 4, tstop // 2, tstop], [20, 20, 60, 20])

print("Descending drive patterns:")
print(f"  - Flexor peak: {np.max(DDdrive_flexor):.1f} Hz")
print(f"  - Extensor peak: {np.max(DDdrive_extensor):.1f} Hz")
print("  - Pattern: Reciprocal activation (anti-phase)")

##############################################################################
# Define Fusimotor Drive
# --------------------
#
# Fusimotor neurons control the sensitivity of muscle spindles by adjusting
# the tension in intrafusal muscle fibers. This allows the nervous system to
# tune proprioceptive feedback based on behavioral demands.

gDynDrive = 70  # Hz - Dynamic fusimotor drive (affects velocity sensitivity)
gStatDrive = 70  # Hz - Static fusimotor drive (affects length sensitivity)

# Add physiological variability to fusimotor drives
gDyn = np.interp(time, [0, tstop], [gDynDrive, gDynDrive]) + np.random.normal(0, 3, len(time))
gStat = np.interp(time, [0, tstop], [gStatDrive, gStatDrive]) + np.random.normal(0, 3, len(time))

# Package fusimotor drives for the simulation
gMN = {
    "name": "gMN",
    "gDyn": gDyn,  # Dynamic fusimotor drive array
    "gStat": gStat,  # Static fusimotor drive array
}

print("Fusimotor drive parameters:")
print(f"  - Dynamic drive: {gDynDrive} ± 3 Hz")
print(f"  - Static drive: {gStatDrive} ± 3 Hz")

##############################################################################
# Initialize Joint Dynamics
# ------------------------
#
# The joint model provides realistic biomechanical constraints and creates the
# closed-loop system where muscle forces influence joint motion, which in turn
# affects proprioceptive feedback.

joint_dynamics = JointDynamics(
    inertia__kg_m2=0.001,  # Realistic joint inertia
    damping__Nm_s_per_rad=0.002,  # Light damping for stability
)

# Initialize joint angle array for closed-loop integration
artAng = np.zeros_like(time)
artAng[0] = 0.0  # Initial joint angle

print("Joint dynamics parameters:")
print(f"  - Inertia: {joint_dynamics.inertia__kg_m2} kg⋅m²")
print(f"  - Damping: {joint_dynamics.damping__Nm_s_per_rad} N⋅m⋅s/rad")
print(f"  - Initial angle: {artAng[0]}°")

##############################################################################
# Create Neural Populations
# ------------------------
#
# Instantiate all neural population objects with physiologically appropriate
# parameters. Each population type has specialized properties reflecting their
# biological counterparts.

# Create separate motor neuron pools for antagonist muscles
# Motor neuron pools (default spike_threshold__mV=50.0 for proper spike detection)
aMN_flex = AlphaMN__Pool(
    recruitment_thresholds__array=recruitment_thresholds
)  # Flexor α-motoneurons
aMN_ext = AlphaMN__Pool(
    recruitment_thresholds__array=recruitment_thresholds
)  # Extensor α-motoneurons

# Create separate descending drive populations for proper antagonist control
DD_flex = DescendingDrive__Pool(n=nDD // 2, poisson_batch_size=DDorder, timestep__ms=dt)
DD_ext = DescendingDrive__Pool(n=nDD // 2, poisson_batch_size=DDorder, timestep__ms=dt)

# Create afferent populations (shared between muscles for this example)
Ia = AffIa__Pool(n=nIa, timestep__ms=dt)  # Primary spindle afferents
II = AffII__Pool(n=nII, timestep__ms=dt)  # Lower thresholds for Group II
Ib = AffIb__Pool(n=nIb, timestep__ms=dt)  # Golgi tendon organ afferents

# Create interneuron populations for reflex modulation
gII = GII__Pool(n=ngII)  # Group II interneurons
gIb = GIb__Pool(n=ngIb)  # Group Ib interneurons

print(
    f"✓ Created {len([aMN_flex, aMN_ext, DD_flex, DD_ext, Ia, II, Ib, gII, gIb])} neural populations"
)

##############################################################################
# Create Proprioceptive Models
# ---------------------------
#
# Initialize the sensory models that convert mechanical variables (muscle length,
# velocity, force) into neural signals that drive the afferent populations.

# Golgi Tendon Organ - monitors muscle force/tension
gto = GolgiTendonOrganModel(
    simulation_time__ms=tstop * pq.ms,
    time_step__ms=dt,
    gto_parameters=GolgiTendonOrganModel.create_default_gto_parameters(),
)

# Muscle Spindle - monitors muscle length and velocity
spin = SpindleModel(
    simulation_time__ms=tstop * pq.ms,
    time_step__ms=dt,
    spindle_parameters=SpindleModel.create_default_spindle_parameters(),
)

print("✓ Initialized proprioceptive models (spindle, GTO)")

##############################################################################
# Create Muscle Models
# ------------------
#
# Initialize Hill-type muscle models for both flexor and extensor muscles.
# These models convert motor neuron activation into realistic force generation
# with appropriate biomechanical properties.

# Flexor muscle model
hill_flexor = HillModel(
    simulation_time__ms=tstop * pq.ms,
    time_step__ms=dt,
    muscle_parameters=HillModel.create_default_muscle_parameters(),
    n_motor_units_type1=nType1,
    n_motor_units_type2=nType2,
    initial_joint_angle__deg=artAng[0],
    muscle_role="flexor",
)

# Extensor muscle model
hill_extensor = HillModel(
    simulation_time__ms=tstop * pq.ms,
    time_step__ms=dt,
    muscle_parameters=HillModel.create_default_muscle_parameters(),
    n_motor_units_type1=nType1,
    n_motor_units_type2=nType2,
    initial_joint_angle__deg=artAng[0],
    muscle_role="extensor",
)

print("✓ Created antagonist muscle models (flexor, extensor)")
print(f"  - Motor units per muscle: {nType1 + nType2}")
print(f"  - Force capacity: ~{hill_flexor.F0:.1f} N per muscle")

##############################################################################
# Define Callback Functions
# ------------------------
#
# These functions handle the integration of different system components during
# the simulation, including spike events and step-wise updates.


def spkEvent(i, muscle, delay):
    muscle.add_spike(i, delay)


def eachStep(
    muscle_flex,
    muscle_ext,
    spin,
    golgi,
    popD,
    ncD,
    gMN,
    joint_dyn,
    step_counter,
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
    """
    i = next(step_counter)

    current_angle = artAng[i]

    # MUSCLE MECHANICS: Integrate both muscles with current joint angle
    L_flex, V_flex, A_flex = muscle_flex.integrate(current_angle)
    _, _, _ = muscle_ext.integrate(current_angle)

    # PROPRIOCEPTIVE FEEDBACK: Use flexor kinematics for spindle feedback
    Iay, IIy = spin.integrate(L_flex, V_flex, A_flex, gMN["gDyn"][i], gMN["gStat"][i])

    # FORCE FEEDBACK: Combined force from both muscles for GTO
    f_flex = muscle_flex.F0 * muscle_flex.muscle_force[i]
    f_ext = muscle_ext.F0 * muscle_ext.muscle_force[i]
    Iby = golgi.integrate(f_flex + f_ext)

    # CLOSED-LOOP DYNAMICS: Update joint angle based on net muscle torque
    new_angle, _ = joint_dyn.integrate(
        torque__Nm=muscle_flex.signed_muscle_torque[i] + muscle_ext.signed_muscle_torque[i],
        dt__s=float(dt.rescale(pq.s).magnitude),
    )
    # Update joint angle array for next timestep
    if i < len(artAng) - 1:
        artAng[i + 1] = new_angle

    # DESCENDING DRIVE PROCESSING: Convert cortical signals to spikes
    for DDcell in popD["DD_flex"]:
        if DDcell.integrate(DDdrive_flexor[i]):
            spike_time = h.t + 1
            if spike_time < tstop:
                ncD["cmd->DD_flex"][DDcell.pool__ID].event(spike_time)

    for DDcell in popD["DD_ext"]:
        if DDcell.integrate(DDdrive_extensor[i]):
            spike_time = h.t + 1
            if spike_time < tstop:
                ncD["cmd->DD_ext"][DDcell.pool__ID].event(spike_time)

    # AFFERENT PROCESSING: Convert sensory signals to neural spikes
    for Ia in popD["Ia"]:
        if Iay >= Ia.RT:
            if Ia.integrate(Iay):
                # Ensure h.t is converted to a quantities object with units of ms
                spike_time = h.t + float(Ia.axon_delay__ms)
                if spike_time < tstop:
                    ncD["Spindle->Ia"][Ia.pool__ID].event(spike_time)

    ii_spikes = 0
    for II in popD["II"]:
        if IIy >= II.RT:
            if II.integrate(IIy):
                spike_time = h.t + float(II.axon_delay__ms)
                if spike_time < tstop:
                    ncD["Spindle->II"][II.pool__ID].event(spike_time)
                    ii_spikes += 1

    ib_spikes = 0
    for Ib in popD["Ib"]:
        if Iby >= Ib.RT:
            if Ib.integrate(Iby):
                spike_time = h.t + float(Ib.axon_delay__ms)
                if spike_time < tstop:
                    ncD["GTO->Ib"][Ib.pool__ID].event(spike_time)
                    ib_spikes += 1


##############################################################################
# Create Neural Network
# --------------------
#
# Assemble all neural populations into a connected network that implements
# the spinal reflex circuitry for antagonist muscle control.

network = Network(
    {
        "DD_flex": DD_flex,  # Flexor descending drive
        "DD_ext": DD_ext,  # Extensor descending drive
        "aMN_flex": aMN_flex,  # Flexor α-motoneurons
        "aMN_ext": aMN_ext,  # Extensor α-motoneurons
        "Ia": Ia,  # Group Ia afferents
        "II": II,  # Group II afferents
        "Ib": Ib,  # Group Ib afferents
        "gII": gII,  # Group II interneurons
        "gIb": gIb,  # Group Ib interneurons
        "gMN": gMN,  # Fusimotor parameters
    }
)

print(f"✓ Created network with {len(network.populations)} populations")

##############################################################################
# Configure Neural Connections
# ---------------------------
#
# Establish synaptic connections that implement physiologically realistic
# spinal reflex pathways, including both excitatory and inhibitory connections.

# DESCENDING CONTROL: Direct cortical drive to motor neurons
network.connect("DD_flex", "aMN_flex", probability=0.3, weight__uS=0.05)
network.connect("DD_ext", "aMN_ext", probability=0.3, weight__uS=0.05)

# MONOSYNAPTIC STRETCH REFLEX: Ia afferents excite homonymous motor neurons
network.connect("Ia", "aMN_flex", probability=0.3, weight__uS=0.05)
network.connect("Ia", "aMN_ext", probability=0.3, weight__uS=0.05)

# POLYSYNAPTIC PATHWAYS: Secondary afferents through interneurons
network.connect("II", "gII", probability=0.6, weight__uS=0.0073)
network.connect("gII", "aMN_flex", probability=0.2, weight__uS=0.05)
network.connect("gII", "aMN_ext", probability=0.2, weight__uS=0.05)

# INHIBITORY REFLEXES: Force feedback through Ib interneurons
network.connect("Ib", "gIb", probability=0.5, weight__uS=0.0073)
network.connect("gIb", "aMN_flex", probability=0.1, weight__uS=-0.05)  # Inhibitory
network.connect("gIb", "aMN_ext", probability=0.1, weight__uS=-0.05)  # Inhibitory

print("✓ Configured synaptic connections")
print("  - Excitatory: DD→MN, Ia→MN, II→gII, gII→MN")
print("  - Inhibitory: gIb→MN (autogenic inhibition)")

##############################################################################
# Connect Motors to Muscles
# ------------------------
#
# Establish the neuromuscular junctions that convert motor neuron spikes
# into muscle fiber activation.

# Connect motor neurons to muscles (spike threshold automatically uses population default of 50.0 mV)
network.connect_to_muscle(
    "aMN_flex", muscle=hill_flexor, activation_callback=spkEvent, weight__uS=1.0
)
network.connect_to_muscle(
    "aMN_ext", muscle=hill_extensor, activation_callback=spkEvent, weight__uS=1.0
)

print("✓ Connected motor neurons to muscles")

##############################################################################
# Configure External Inputs
# ------------------------
#
# Setup external input pathways for sensory feedback and descending commands.

network.connect_from_external("cmd_flex", "DD_flex", weight__uS=1.0)
network.connect_from_external("cmd_ext", "DD_ext", weight__uS=1.0)
network.connect_from_external("spindle", "Ia", weight__uS=1.0)
network.connect_from_external("spindle", "II", weight__uS=1.0)
network.connect_from_external("gto", "Ib", weight__uS=1.0)
# Add test drive for Group II interneurons (commented out due to INgII missing .ns attribute)
# network.connect_from_external("test_gII", "gII", weight__uS=1.0)

# Get NetCons for manual triggering during simulation
ncD = {
    "cmd->DD_flex": network.get_netcons("cmd_flex", "DD_flex"),
    "cmd->DD_ext": network.get_netcons("cmd_ext", "DD_ext"),
    "Spindle->Ia": network.get_netcons("spindle", "Ia"),
    "Spindle->II": network.get_netcons("spindle", "II"),
    "GTO->Ib": network.get_netcons("gto", "Ib"),
}

print("✓ Configured external input pathways")

##############################################################################
# Prepare Simulation Models
# ------------------------

# Package all models for the simulation runner
models = {
    "hill_flexor": hill_flexor,
    "hill_extensor": hill_extensor,
    "spin": spin,
    "gto": gto,
    "joint": joint_dynamics,
}


# Create step callback function with access to step counter
def step_callback(step_counter):
    return eachStep(
        muscle_flex=hill_flexor,
        muscle_ext=hill_extensor,
        spin=spin,
        golgi=gto,
        popD=network.populations,
        ncD=ncD,
        gMN=gMN,
        joint_dyn=joint_dynamics,
        step_counter=step_counter,
    )


##############################################################################
# Run Spinal Network Simulation
# ----------------------------
#
# Execute the complete simulation with all integrated components.

print("\nStarting spinal network simulation...")
print(f"\tDuration: {tstop} ms")
print(f"\tTimestep: {dt} ms")
print(f"\tPopulations: {len(network.populations)}")

runner = SimulationRunner(
    network=network,
    models=models,
    step_callback=step_callback,
)

# Motor neuron spike recording thresholds are now fixed in the Network class

results = runner.run(
    duration__ms=tstop * pq.ms,
    timestep__ms=dt,
    membrane_recording={
        "aMN_flex": [0, 5, 10, 15, 20, 30, 40, 50, 60, 70],
        "aMN_ext": [0, 5, 10, 15, 20, 30, 40, 50, 60, 70],
    },
)

print("Simulation completed successfully!")

# Save simulation results
joblib.dump(results, save_path / "spinal_network_results.pkl")
joblib.dump(artAng, save_path / "joint_angles.pkl")
print("Results saved to {save_path}")

##############################################################################
# Comprehensive Results Visualization
# ---------------------------------
#
# Create a series of plots that tell the complete story of spinal network
# function, from neural activity to mechanical output.

print("\nGenerating comprehensive visualizations...")

# 1. NEURAL ACTIVITY: Raster plot showing all population spike patterns
populations_list = [
    "aMN_flex",
    "aMN_ext",  # Motor output
    "Ia",
    "II",
    "Ib",  # Sensory input
    "gII",
    "gIb",  # Interneurons
    "DD_flex",
    "DD_ext",  # Descending drive
]
fig1, axes1 = plt.subplots(len(populations_list), 1, figsize=(15, 15))
plot_raster_spikes(
    results,
    axes1,
    populations=populations_list,
    time_range=(0, tstop),
    title="Spinal Network Activity",
)
plt.tight_layout()
plt.savefig(save_path / "neural_raster_plot.png", dpi=150, bbox_inches="tight")
plt.show()

# 2. MOTOR NEURON DYNAMICS: Membrane potentials showing integration
motor_populations = ["aMN_flex", "aMN_ext"]
fig2, axes2 = plt.subplots(len(motor_populations), 1, figsize=(15, 10))

plot_membrane_potentials(
    results,
    axes2,
    populations=motor_populations,
    cell_indices=[0, 10, 20, 30, 40],
    time_range=(0, tstop),
    title="Motor Neuron Membrane Potentials",
)

plt.tight_layout()
plt.show()

# 3. MUSCLE MECHANICS: Individual muscle dynamics
fig3, axes3 = plt.subplots(5, 1, figsize=(15, 20))
plot_muscle_dynamics(
    results,
    artAng,
    time,
    axes3,
    muscle_name="hill_flexor",
    include_signals=["artAng", "L", "force", "torque"],
    include_activations=["TypeI", "TypeII"],
    normalize=True,
    time_range=(0, tstop),
    title="Flexor Muscle Dynamics - Length, Force, and Activation",
)
plt.tight_layout()
plt.savefig(save_path / "flexor_muscle_dynamics.png", dpi=150, bbox_inches="tight")
plt.show()

fig4, axes4 = plt.subplots(5, 1, figsize=(15, 20))
plot_muscle_dynamics(
    results,
    artAng,
    time,
    axes4,
    muscle_name="hill_extensor",
    include_signals=["artAng", "L", "force", "torque"],
    include_activations=["TypeI", "TypeII"],
    normalize=True,
    time_range=(0, tstop),
    title="Extensor Muscle Dynamics - Length, Force, and Activation",
)
plt.tight_layout()
plt.savefig(save_path / "extensor_muscle_dynamics.png", dpi=150, bbox_inches="tight")
plt.show()

# 4. ANTAGONIST COMPARISON: Coordinated muscle function
fig5, axes5 = plt.subplots(3, 1, figsize=(15, 12))
plot_antagonist_muscle_comparison(
    results,
    artAng,
    time,
    axes5,
    flexor_name="hill_flexor",
    extensor_name="hill_extensor",
    include_signals=["artAng", "force", "torque"],
    time_range=(0, tstop),
    title="Antagonist Muscle Coordination - Flexor vs Extensor Control",
)
plt.tight_layout()
plt.savefig(save_path / "antagonist_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# 5. PROPRIOCEPTIVE FEEDBACK: Spindle dynamics and sensory encoding
fig6, axes6 = plt.subplots(4, 1, figsize=(15, 16))
plot_spindle_dynamics(
    results,
    axes6,
    muscle_name="hill_flexor",
    include_signals=["L"],
    include_activations=["Bag1", "Bag2", "Chain"],
    include_tensions=["Bag1", "Bag2", "Chain"],
    include_afferents=["Ia", "II"],
    time_range=(0, tstop),
    title="Muscle Spindle Dynamics - Proprioceptive Feedback System",
)
plt.tight_layout()
plt.savefig(save_path / "spindle_dynamics.png", dpi=150, bbox_inches="tight")
plt.show()

# 6. FORCE FEEDBACK: GTO dynamics and protective reflexes
fig7, axes7 = plt.subplots(2, 1, figsize=(15, 8))
plot_gto_dynamics(
    results,
    axes7,
    muscle_name="hill_flexor",
    include_signals=["force", "Ib"],
    time_range=(0, tstop),
    title="Golgi Tendon Organ Dynamics - Force Feedback System",
)
plt.tight_layout()
plt.savefig(save_path / "gto_dynamics.png", dpi=150, bbox_inches="tight")
plt.show()
