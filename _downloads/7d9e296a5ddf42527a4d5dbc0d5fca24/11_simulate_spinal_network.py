"""
Spinal Network Simulation with Systematic Tendon Tap Protocol
==============================================================

This example demonstrates **complete spinal reflex network modeling** with a **comprehensive tendon tap
protocol** that systematically varies mechanical perturbations, fusimotor drive, and cortical activity.
This sophisticated experimental design reveals how the nervous system modulates stretch reflex gain.

Learning Objectives
------------------

This example teaches the following key concepts in neuromuscular control:

1. **Reflex Gain Modulation**: How fusimotor (gamma) drive amplifies stretch reflex sensitivity
   - Compare motor neuron responses across different gamma levels (0→100 pps)
   - Observe how higher gamma drive increases spindle sensitivity and reflex output

2. **Sensory Pathway Timing**: Why different feedback pathways activate at different times
   - Spindle-driven pathways (Ia, II → gII) respond immediately to muscle stretch
   - Force-driven pathways (Ib → gIb) respond later after muscle contraction builds force

3. **Reflex-Voluntary Interaction**: How cortical commands interact with spinal reflexes
   - Phase 1: Isolated reflex responses (no cortical drive)
   - Phase 2: Reflex modulation during voluntary contraction (with cortical drive)
   - Observe how background muscle activity affects reflex sensitivity

4. **Closed-loop Biomechanics**: How muscle forces create movement that feeds back to sensors
   - Joint dynamics convert muscle torque into motion
   - Motion changes muscle length/velocity, affecting spindle feedback
   - Creates realistic proprioceptive-motor coupling

**Experimental Protocol (5-second, 2-phase design):**

**Phase 1 (0-2.5s): Reflex Gain Modulation**
- Triangular tendon taps every 0.5s with increasing amplitude (50%, 60%, 70%, 80%)

  * Triangular waveform: Linear ramp up to peak (50ms) → linear ramp down (50ms)
  * Total duration: 100ms per tap
  * Mimics realistic tendon tap dynamics with stretch and release phases

- Stepwise increasing gamma drive starting at 0.75s (0→25→50→75→100 pps)
- No cortical drive (isolated reflex testing)

**Phase 2 (2.5-5s): Reflex-Voluntary Interaction**
- Repeat triangular tap pattern (50%, 60%, 70%, 80%)
- Repeat gamma drive pattern (0→25→50→75→100 pps)
- Sinusoidal cortical drive (38±1 Hz at 1 Hz) recruiting ~75% of motor neurons

.. note::
    This example builds upon all previous examples and demonstrates how the complete neuromuscular system
    functions as an integrated network:

    - **Motor neuron pool**: α-motoneurons controlling a single muscle (from examples 01-02)
    - **Muscle model**: Hill-type muscle model with realistic force generation (from examples 02, 06)
    - **Proprioceptive feedback**: Muscle spindles and Golgi tendon organs (from examples)
    - **Spinal interneurons**: Group II and Group Ib interneurons for reflex modulation
    - **Joint dynamics**: Closed-loop biomechanical control with realistic inertia and damping
    - **Descending drive**: Cortical control signals for voluntary movement (from example 01)

.. important::
    **Spinal Reflex Networks** are the fundamental control circuits that coordinate muscle activity.
    Key physiological concepts demonstrated:

    - **Stretch reflex**: Muscle spindles provide length/velocity feedback for posture control
    - **Force feedback**: Golgi tendon organs monitor muscle tension for protective reflexes
    - **Closed-loop control**: Joint mechanics influence neural activity through proprioception
    - **Motor control**: Integration of descending commands with sensory feedback

Scientific Background
-------------------

The spinal cord contains intricate neural circuits that process sensory information and generate appropriate
motor responses. This example models several key components:

**Muscle Spindles**: Specialized sensory organs that detect muscle length and velocity changes, providing
Group Ia (velocity-sensitive) and Group II (length-sensitive) afferent signals.

**Golgi Tendon Organs**: Force-sensitive receptors that monitor muscle tension and provide Group Ib
afferent feedback for protective reflexes.

**Spinal Interneurons**: The model includes two key inhibitory interneuron populations:

- **Group II interneurons (gII)**: Process Group II spindle afferents (length-sensitive). In this model,
  they provide inhibitory feedback to motor neurons, representing one component of the complex Group II
  pathway (which can be excitatory or inhibitory depending on context).

- **Group Ib interneurons (gIb)**: Receive input from Golgi tendon organs and provide **autogenic
  inhibition** - force-dependent negative feedback that protects against excessive muscle tension.
  Classic Ib inhibitory pathway.

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
import quantities as pq
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
    plot_gto_dynamics,
    plot_membrane_potentials,
    plot_muscle_dynamics,
    plot_raster_spikes,
    plot_spindle_dynamics,
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
save_path = Path(r"./results")
save_path.mkdir(exist_ok=True)

recruitment_thresholds = joblib.load(save_path / "thresholds.pkl")
print("(OK) Loaded recruitment thresholds from example 00")

##############################################################################
# Define Simulation Parameters
# ---------------------------
#
# These parameters control the temporal and spatial resolution of the simulation,
# as well as the physiological characteristics of the neural populations and
# mechanical system.

# Temporal parameters - high resolution for accurate neural integration
dt = 0.005 * pq.ms  # Integration timestep with units
tstop = 5e3  # 5e3  # ms - Total simulation duration (5s for comprehensive protocol)
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
# Based on Banks et al. (2006, 2015) for distal upper-limb muscles
nIa = 73  # Group Ia afferents from muscle spindles (velocity-sensitive)
nII = 80  # Group II afferents from muscle spindles (length-sensitive) - 110% of Ia
nIb = 58  # Group Ib afferents from Golgi tendon organs (force-sensitive) - 80% of Ia

# Interneuron populations (spinal processing)
# Note: Interneuron:MN ratio is ~10:1 in spinal cord. Group II/Ib INs are a subset.
# Using ~50% of afferent count as a reasonable estimate for convergent interneurons.
ngII = 120  # Group II interneurons (process spindle feedback from 80 II afferents)
ngIb = 145  # Group Ib interneurons (process force feedback from 58 Ib afferents)

# Motor neurons (output to muscles)
nType1 = 102  # Type I motor neurons (slow, fatigue-resistant)
nType2 = 18  # Type II motor neurons (fast, fatigue-prone)
naMN = nType1 + nType2  # Total α-motoneurons per muscle

# Descending drive (cortical input)
nDD = 400  # Total descending drive neurons (increased for more drive)
DDorder = (
    5  # Poisson process order for realistic spike patterns (higher order = more regular spiking)
)

print("Neural population sizes:")
print(f"\t- α-Motoneurons: {naMN} ({nType1} Type I, {nType2} Type II)")
print(f"\t- Ia afferents: {nIa}")
print(f"\t- II afferents: {nII}")
print(f"\t- Ib afferents: {nIb}")
print(f"\t- Interneurons: {ngII + ngIb}")
print(f"\t- Descending drive: {nDD}")

##############################################################################
# Define Descending Drive Pattern
# -------------------------------
#
# Two-phase pattern:
# Phase 1 (0-2500ms): No cortical drive (reflex testing with varying gamma)
# Phase 2 (2500-5000ms): Sinusoidal cortical drive (reflex modulation during voluntary contraction)

# Phase 1: No cortical drive
# Phase 2: 1 Hz sinusoidal drive with bias to recruit ~75% of MN population
DDdrive = np.zeros_like(time)
cortical_start_time = 2500  # ms

# Parameters for sinusoidal drive
bias_hz = 40  # Bias for robust motor neuron recruitment (increased from 75 for ~20 pps firing)
amplitude_hz = 1  # Oscillation amplitude (±5 Hz around bias)
frequency_hz = 1.0  # 1 Hz oscillation

# Apply sinusoidal drive from 2.5s onwards
mask = time >= cortical_start_time
DDdrive[mask] = bias_hz + amplitude_hz * np.sin(
    2 * np.pi * frequency_hz * (time[mask] - cortical_start_time) / 1000.0
)

print("Descending drive pattern:")
print("\t- Phase 1 (0-2.5s): No cortical drive")
print(f"\t- Phase 2 (2.5-5s): Sinusoidal drive ({bias_hz}±{amplitude_hz} Hz at {frequency_hz} Hz)")
print(f"\t  (Drive range: {bias_hz - amplitude_hz}-{bias_hz + amplitude_hz} Hz)")

##############################################################################
# Define Fusimotor Drive
# --------------------
#
# Fusimotor neurons control the sensitivity of muscle spindles by adjusting
# the tension in intrafusal muscle fibers. Stepwise increasing gamma drive
# demonstrates how fusimotor activity modulates stretch reflex gain.
#
# Pattern: Increases every 0.5s starting at 0.75s, resets at 2.5s
# Steps: 0 → 25 → 50 → 75 → 100 pps (then repeat)

# Time points for stepwise gamma drive
# Phase 1: 0→0.75s (wait)→1.25s→1.75s→2.25s (steps every 0.5s)
# Phase 2: 2.5s (reset)→3.25s (wait 0.75s)→3.75s→4.25s→4.75s (steps every 0.5s)
gamma_times = np.array([0, 750, 1250, 1750, 2250, 2500, 3250, 3750, 4250, 4750, 5000])
gamma_values = np.array([0, 25, 50, 75, 100, 0, 25, 50, 75, 100, 100])

# Create stepwise gamma drive arrays using proper step function (not interpolation!)
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
gDyn = gDyn + np.random.normal(0, 1, len(time))
gStat = gStat + np.random.normal(0, 1, len(time))

# Ensure non-negative
gDyn = np.maximum(gDyn, 0)
gStat = np.maximum(gStat, 0)

# Package fusimotor drives for the simulation
gMN = {
    "name": "gMN",
    "gDyn": gDyn,  # Dynamic fusimotor drive array
    "gStat": gStat,  # Static fusimotor drive array
}

print("Fusimotor drive parameters:")
print("\t- Stepwise increases: 0 → 25 → 50 → 75 → 100 pps")
print("\t- Starting at 0.75s, stepping every 0.5s")
print("\t- Resets at 2.5s and repeats pattern")

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
print(f"\t- Inertia: {joint_dynamics.inertia__kg_m2} kg⋅m²")
print(f"\t- Damping: {joint_dynamics.damping__Nm_s_per_rad} N⋅m⋅s/rad")
print(f"\t- Initial angle: {artAng[0]}°")

##############################################################################
# Create Neural Populations
# ------------------------
#
# Instantiate all neural population objects with physiologically appropriate
# parameters. Each population type has specialized properties reflecting their
# biological counterparts.

# Create motor neuron pool
# Motor neuron pool (default spike_threshold__mV=50.0 for proper spike detection)
aMN = AlphaMN__Pool(recruitment_thresholds__array=recruitment_thresholds)  # α-motoneurons

# Create descending drive population
DD = DescendingDrive__Pool(n=nDD, poisson_batch_size=DDorder, timestep__ms=dt)

# Create afferent populations
Ia = AffIa__Pool(n=nIa, timestep__ms=dt)  # Primary spindle afferents
II = AffII__Pool(n=nII, timestep__ms=dt)  # Lower thresholds for Group II
Ib = AffIb__Pool(n=nIb, timestep__ms=dt)  # Golgi tendon organ afferents

# Create interneuron populations for reflex modulation
gII = GII__Pool(n=ngII)  # Group II interneurons
gIb = GIb__Pool(n=ngIb)  # Group Ib interneurons

print(f"(OK) Created {len([aMN, DD, Ia, II, Ib, gII, gIb])} neural populations")

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

print("(OK) Initialized proprioceptive models (spindle, GTO)")

##############################################################################
# Create Muscle Model
# ------------------
#
# Initialize Hill-type muscle model. This model converts motor neuron
# activation into realistic force generation with appropriate biomechanical properties.

# Muscle model
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
print(f"\t- Motor units: {nType1 + nType2}")
print(f"\t- Force capacity: ~{hill_muscle.F0:.1f} N")

##############################################################################
# Create Arrays for Real-Unit Forces
# ----------------------------------
#
# Store musculotendon force in real units [N] for analysis

# Musculotendon force array [N]
musculotendon_force__N = np.zeros_like(time)

print("(OK) Initialized force tracking arrays")

##############################################################################
# Define Callback Functions
# ------------------------
#
# These functions handle the integration of different system components during
# the simulation, including spike events and step-wise updates.


def spkEvent(i, muscle, delay):
    muscle.add_spike(i, delay)


def eachStep(
    muscle,
    spin,
    golgi,
    popD,
    ncD,
    gMN,
    joint_dyn,
    step_counter,
    tendon_tap_schedule,  # List of (time_ms, magnitude_percent) tuples
    tendon_tap_duration=100,  # ms
):
    """
    Step-wise integration function called at each simulation timestep.

    This function orchestrates the complex interactions between:
    - Muscle mechanics and force generation
    - Proprioceptive feedback from spindles and GTOs
    - Joint dynamics and closed-loop control
    - Neural population dynamics and spike generation
    - Multiple tendon tap perturbations with varying amplitudes

    Parameters
    ----------
    muscle : HillModel
        Muscle model
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
    tendon_tap_schedule : list of tuples
        List of (time_ms, magnitude_percent) defining tap timing and amplitudes
    tendon_tap_duration : float
        Duration (ms) of each tendon tap stretch
    """
    i = next(step_counter)

    current_angle = artAng[i]

    # MUSCLE MECHANICS: Integrate muscle with current joint angle
    L, V, A = muscle.integrate(current_angle)

    # TENDON TAP PERTURBATION: Triangular pulse (ramp up, then down)
    current_time = h.t

    for tap_time, tap_magnitude in tendon_tap_schedule:
        tap_end_time = tap_time + tendon_tap_duration
        tap_midpoint = tap_time + (tendon_tap_duration / 2.0)

        if tap_time <= current_time < tap_end_time:
            # Triangular pulse: ramp up to peak at midpoint, then ramp down
            max_perturbation = L * (tap_magnitude / 100.0)

            if current_time < tap_midpoint:
                # Rising edge: 0 → peak (first half)
                progress = (current_time - tap_time) / (tendon_tap_duration / 2.0)
                length_perturbation = max_perturbation * progress
            else:
                # Falling edge: peak → 0 (second half)
                progress = (tap_end_time - current_time) / (tendon_tap_duration / 2.0)
                length_perturbation = max_perturbation * progress

            L_perturbed = L + length_perturbation

            # Velocity reflects the triangle slope
            if current_time < tap_midpoint:
                # Rising: positive velocity
                V_perturbed = V + (max_perturbation / (tendon_tap_duration / 2000.0))
            else:
                # Falling: negative velocity
                V_perturbed = V - (max_perturbation / (tendon_tap_duration / 2000.0))

            # Acceleration spikes at tap onset and reverses at midpoint
            if abs(current_time - tap_time) < dt.magnitude:
                A_perturbed = A + (max_perturbation / ((tendon_tap_duration / 2000.0) ** 2))
                print(
                    f"\t>> TENDON TAP (triangular) at t={current_time:.1f} ms: "
                    f"{tap_magnitude}% peak stretch ({max_perturbation:.4f} L0), "
                    f"duration={tendon_tap_duration} ms"
                )
                print(f"\t   Baseline length: {L:.4f} L0 → Peak: {L + max_perturbation:.4f} L0")
            elif abs(current_time - tap_midpoint) < dt.magnitude:
                A_perturbed = A - 2 * (max_perturbation / ((tendon_tap_duration / 2000.0) ** 2))
            else:
                A_perturbed = A

            # Use perturbed values for spindle feedback
            L, V, A = L_perturbed, V_perturbed, A_perturbed
            break  # Only one tap active at a time

    # PROPRIOCEPTIVE FEEDBACK: Use muscle kinematics for spindle feedback
    Iay, IIy = spin.integrate(L, V, A, gMN["gDyn"][i], gMN["gStat"][i])

    # FORCE FEEDBACK: Muscle force for GTO
    f = muscle.F0 * muscle.muscle_force[i]  # Musculotendon force [N]
    musculotendon_force__N[i] = f  # Store in real units
    Iby = golgi.integrate(f)

    # CLOSED-LOOP DYNAMICS: Update joint angle based on muscle torque
    new_angle, _ = joint_dyn.integrate(
        torque__Nm=muscle.signed_muscle_torque[i],
        dt__s=float(dt.rescale(pq.s).magnitude),
    )
    # Update joint angle array for next timestep
    if i < len(artAng) - 1:
        artAng[i + 1] = new_angle

    # DESCENDING DRIVE PROCESSING: Convert cortical signals to spikes
    for DDcell in popD["DD"]:
        if DDcell.integrate(DDdrive[i]):
            spike_time = h.t + 1
            if spike_time < tstop:
                ncD["cmd->DD"][DDcell.pool__ID].event(spike_time)

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
# the spinal reflex circuitry for single muscle control.

network = Network(
    {
        "DD": DD,  # Descending drive
        "aMN": aMN,  # α-motoneurons
        "Ia": Ia,  # Group Ia afferents
        "II": II,  # Group II afferents
        "Ib": Ib,  # Group Ib afferents
        "gII": gII,  # Group II interneurons
        "gIb": gIb,  # Group Ib interneurons
        "gMN": gMN,  # Fusimotor parameters
    }
)

print(f"(OK) Created network with {len(network.populations)} populations")

##############################################################################
# Configure Neural Connections
# ---------------------------
#
# Establish synaptic connections that implement physiologically realistic
# spinal reflex pathways, including both excitatory and inhibitory connections.
#
# Neural Circuit Diagram:
# ----------------------
#
#   Spindle (length/velocity)          GTO (force)           Cortex
#        |          |                       |                  |
#        v          v                       v                  v
#       Ia         II                      Ib                 DD
#        |          |                       |                  |
#        |          v                       v                  |
#        |        gII (-)                 gIb (-)              |
#        |          |                       |                  |
#        +----------+--------(+)------------+----------(+)-----+
#                              |
#                              v
#                            α-MN -----> Muscle
#                                          |
#                                          v
#                                    Joint Motion
#                                          |
#         +--------------------------------+
#         |                                |
#         v                                v
#      Spindle                           GTO
#    (feedback)                      (feedback)
#
# Legend: (+) = excitatory, (-) = inhibitory
#
# Key pathways:
# 1. Ia → MN: Monosynaptic stretch reflex (EXCITATORY, fast)
# 2. II → gII → MN: Polysynaptic length pathway (INHIBITORY, moderate speed)
# 3. Ib → gIb → MN: Autogenic inhibition (INHIBITORY, slower - force-dependent)
# 4. DD → MN: Descending cortical drive (EXCITATORY, voluntary control)

# DESCENDING CONTROL: Direct cortical drive to motor neurons (EXCITATORY)
network.connect("DD", "aMN", probability=0.3, weight__uS=0.05 * pq.uS)

# MONOSYNAPTIC STRETCH REFLEX: Ia afferents excite motor neurons (EXCITATORY)
# High connection probability (80%) reflects homogeneous excitatory distribution
network.connect("Ia", "aMN", probability=0.8, weight__uS=0.05 * pq.uS)

# POLYSYNAPTIC PATHWAYS: Group II pathway (afferents → interneurons → motor neurons)
# Lower connection probability (30%) captures predominantly indirect, disynaptic influence
network.connect(
    "II", "gII", probability=0.3, weight__uS=0.025 * pq.uS
)  # Afferent → Interneuron (EXCITATORY)
network.connect(
    "gII", "aMN", probability=0.3, weight__uS=0.05 * pq.uS, inhibitory=True
)  # Interneuron → MN (INHIBITORY, weak modulatory effect)

# INHIBITORY REFLEXES: Force feedback through Ib interneurons (autogenic inhibition)
# Lower connection probability (30%) captures predominantly indirect, disynaptic influence
network.connect(
    "Ib", "gIb", probability=0.3, weight__uS=0.025 * pq.uS
)  # Afferent → Interneuron (EXCITATORY)
network.connect(
    "gIb", "aMN", probability=0.3, weight__uS=0.05 * pq.uS, inhibitory=True
)  # Interneuron → MN (INHIBITORY, weak modulatory effect)

print("(OK) Configured synaptic connections")
print("\t- Excitatory pathways:")
print("\t\t• DD→MN (descending drive)")
print("\t\t• Ia→MN (monosynaptic stretch reflex)")
print("\t\t• Afferents→Interneurons: II→gII, Ib→gIb")
print("\t- Inhibitory pathways:")
print("\t\t• gII→MN (Group II interneuron inhibition)")
print("\t\t• gIb→MN (autogenic inhibition from force feedback)")

##############################################################################
# Connect Motors to Muscle
# ------------------------
#
# Establish the neuromuscular junctions that convert motor neuron spikes
# into muscle fiber activation.

# Connect motor neurons to muscle (spike threshold automatically uses population default of 50.0 mV)
network.connect_to_muscle(
    "aMN", muscle=hill_muscle, activation_callback=spkEvent, weight__uS=1.0 * pq.uS
)

print("(OK) Connected motor neurons to muscle")

##############################################################################
# Configure External Inputs
# ------------------------
#
# Setup external input pathways for sensory feedback and descending commands.

network.connect_from_external("cmd", "DD", weight__uS=1.0 * pq.uS)
network.connect_from_external("spindle", "Ia", weight__uS=1.0 * pq.uS)
network.connect_from_external("spindle", "II", weight__uS=1.0 * pq.uS)
network.connect_from_external("gto", "Ib", weight__uS=1.0 * pq.uS)

# Get NetCons for manual triggering during simulation
ncD = {
    "cmd->DD": network.get_netcons("cmd", "DD"),
    "Spindle->Ia": network.get_netcons("spindle", "Ia"),
    "Spindle->II": network.get_netcons("spindle", "II"),
    "GTO->Ib": network.get_netcons("gto", "Ib"),
}

print("(OK) Configured external input pathways")

##############################################################################
# Prepare Simulation Models
# ------------------------

# Package all models for the simulation runner
models = {
    "hill_muscle": hill_muscle,
    "spin": spin,
    "gto": gto,
    "joint": joint_dynamics,
}


# Tendon tap parameters - multiple taps with increasing amplitudes
# Pattern: Taps every 0.5s with increasing amplitude, reset at 2.5s
# High amplitudes to elicit strong reflex responses
# Phase 1 (0-2.5s): 50%, 60%, 70%, 80%
# Phase 2 (2.5-5s): 50%, 60%, 70%, 80%

TENDON_TAP_DURATION = 100  # ms - duration of each stretch
TENDON_TAP_SCHEDULE = [
    # Phase 1: Ascending amplitudes without cortical drive
    (500, 50.0),  # 0.5s: 50%
    (1000, 60.0),  # 1.0s: 60%
    (1500, 70.0),  # 1.5s: 70%
    (2000, 80.0),  # 2.0s: 80%
    # Phase 2: Repeat pattern with cortical drive
    (3000, 50.0),  # 3.0s: 50%
    (3500, 60.0),  # 3.5s: 60%
    (4000, 70.0),  # 4.0s: 70%
    (4500, 80.0),  # 4.5s: 80%
]

print("\nTendon tap schedule:")
print(f"\t- Duration per tap: {TENDON_TAP_DURATION} ms")
print(f"\t- Phase 1 (0-2.5s): {len([t for t, m in TENDON_TAP_SCHEDULE if t < 2500])} taps")
print(f"\t- Phase 2 (2.5-5s): {len([t for t, m in TENDON_TAP_SCHEDULE if t >= 2500])} taps")
print(f"\t- Amplitudes: {sorted(set([m for t, m in TENDON_TAP_SCHEDULE]))}%")


# Create step callback function with access to step counter
def step_callback(step_counter):
    return eachStep(
        muscle=hill_muscle,
        spin=spin,
        golgi=gto,
        popD=network.populations,
        ncD=ncD,
        gMN=gMN,
        joint_dyn=joint_dynamics,
        step_counter=step_counter,
        tendon_tap_schedule=TENDON_TAP_SCHEDULE,
        tendon_tap_duration=TENDON_TAP_DURATION,
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
        "aMN": [0, 5, 10, 15, 20, 30, 40, 50, 60, 70],
    },
)

print("Simulation completed successfully!")

# Save simulation results
joblib.dump(results, save_path / "spinal_network_results.pkl")
joblib.dump(artAng, save_path / "joint_angles.pkl")
joblib.dump(musculotendon_force__N, save_path / "musculotendon_force__N.pkl")

# Save drive signals for analysis and plotting
drive_signals = {
    "time": time,
    "descending_drive": DDdrive,
    "gamma_dynamic": gDyn,
    "gamma_static": gStat,
    "tendon_tap_schedule": TENDON_TAP_SCHEDULE,
    "tendon_tap_duration": TENDON_TAP_DURATION,
}
joblib.dump(drive_signals, save_path / "drive_signals.pkl")

# Save event timeline for plot annotations
events = {
    # Tendon tap events: (peak_time_ms, amplitude_%, duration_ms)
    # Note: peak_time is at the midpoint of the triangular pulse
    "tendon_taps": [
        (t + TENDON_TAP_DURATION / 2.0, m, TENDON_TAP_DURATION) for t, m in TENDON_TAP_SCHEDULE
    ],
    # Gamma drive transitions: (time_ms, level_pps)
    "gamma_transitions": list(zip(gamma_times, gamma_values)),
    # Gamma drive periods: (start_time_ms, end_time_ms, level_pps)
    "gamma_periods": [
        (gamma_times[i], gamma_times[i + 1], gamma_values[i]) for i in range(len(gamma_times) - 1)
    ],
    # Phase markers
    "phase_boundary": 2500,  # ms - transition from phase 1 to phase 2
    "cortical_drive_onset": cortical_start_time,  # ms - when sinusoidal drive starts
    # Experimental phases with descriptions
    "phases": [
        {"start": 0, "end": 2500, "name": "Phase 1", "description": "Reflex gain modulation"},
        {
            "start": 2500,
            "end": 5000,
            "name": "Phase 2",
            "description": "Reflex-voluntary interaction",
        },
    ],
}
joblib.dump(events, save_path / "event_timeline.pkl")

print(f"Results saved to {save_path}")
print("\t- spinal_network_results.pkl (neural activity, muscle, spindle, GTO)")
print("\t- joint_angles.pkl (joint kinematics)")
print("\t- musculotendon_force__N.pkl (real-unit musculotendon force [N])")
print("\t- drive_signals.pkl (cortical drive, gamma drive, tap schedule)")
print("\t- event_timeline.pkl (event markers for plot annotations)")
print("\nNote: Intrafusal fiber tensions [FU] are in spinal_network_results.pkl")
print("\tAccess via: results.segments[0].analogsignals (look for 'intrafusal_tensions')")
print("\tConvert to real units: T[N] ≈ T[FU] * scaling_factor (no standard conversion)")
print(f"\tMusculotendon force capacity: {hill_muscle.F0:.1f} N")

##############################################################################
# Comprehensive Results Visualization
# ---------------------------------
#
# Create a series of plots that tell the complete story of spinal network
# function, from neural activity to mechanical output.

print("\nGenerating comprehensive visualizations...")

# 1. NEURAL ACTIVITY: Raster plot showing all population spike patterns
populations_list = [
    "aMN",  # Motor output
    "Ia",
    "II",
    "Ib",  # Sensory input
    "gII",
    "gIb",  # Interneurons
    "DD",  # Descending drive
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
plt.savefig(save_path / "neural_raster_plot.png", dpi=150, bbox_inches="tight")
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
plt.savefig(save_path / "muscle_dynamics.png", dpi=150, bbox_inches="tight")
plt.show()

# 4. PROPRIOCEPTIVE FEEDBACK: Spindle dynamics and sensory encoding
fig4, axes4 = plt.subplots(4, 1, figsize=(15, 16))
plot_spindle_dynamics(
    results,
    axes4,
    muscle_name="hill_muscle",
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
plt.savefig(save_path / "gto_dynamics.png", dpi=150, bbox_inches="tight")
plt.show()
