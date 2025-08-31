import numpy as np

from myogen import setup_myogen

setup_myogen()

import matplotlib.pyplot as plt
from neuron import h

from myogen.simulator.neuron.joint_dynamics import JointDynamics
from myogen.simulator.neuron.muscle import HillModel
from myogen.simulator.neuron.network import Network
from myogen.simulator.neuron.pops import *
from myogen.simulator.neuron.proprioception import (
    GolgiTendonOrganModel,
    SpindleModel,
)
from myogen.simulator.neuron.simulation_runner import SimulationRunner
from myogen.utils.plotting import (
    plot_antagonist_muscle_comparison,
    plot_gto_dynamics,
    plot_membrane_traces,
    plot_muscle_dynamics,
    plot_raster_spikes,
    plot_spindle_dynamics,
)

# Simulation time - use more stable timestep
dt = 0.005  # ms
tstop = 1e3  # ms
time = np.arange(0, tstop, dt)

# Descending Drive - Reciprocal activation for antagonist muscles

# Flexor drive: smooth ramp up and sustained activation
DDdrive_flexor = np.interp(time, [0, tstop // 2, tstop], [0, 60, 0])
# Extensor drive: low baseline activity
DDdrive_extensor = np.interp(time, [0, tstop // 2, tstop], [60, 0, 60])
DDorder = 16

# Fusimotor drive dynamic gDyn and static gStat
gDynDrive = 70
gStatDrive = 70
gDyn = np.interp(time, [0, tstop], [gDynDrive, gDynDrive]) + np.random.normal(
    0, 3, len(time)
)
gStat = np.interp(time, [0, tstop], [gStatDrive, gStatDrive]) + np.random.normal(
    0, 3, len(time)
)

# Initialize joint dynamics for closed-loop control
joint_dynamics = JointDynamics(
    inertia__kg_m2=0.001,  # Reasonable inertia for small joint
    damping__Nm_s_per_rad=0.002,  # Small amount of damping for stability
    stiffness__Nm_per_rad=0.0,  # No spring stiffness
    initial_angle__deg=0,  # Starting angle
    initial_velocity__deg_per_s=0.0,
)

# Initialize angle array - will be computed dynamically
artAng = np.zeros_like(time)
artAng[0] = 0.0  # Initial condition

# Number of neurons
nIa = 100  # 210 # 315 # 420

nII = 25  # 210 # 315 # 420
ngII = 25  # 210 # 315 # 420

nIb = 25  # 220
ngIb = 25  # 220

nDD = 400  # 400 	# 400 Desceding Drive number

nType1 = 102  # 102 Alpha MN fiber type I
nType2 = 18  # 18 alpha MN fiber type II

naMN = nType1 + nType2

# Create separate motor neuron pools for flexor and extensor muscles
aMN_flex = AlphaMN__Pool(n=naMN)
aMN_ext = AlphaMN__Pool(n=naMN)
# Separate descending drive populations for proper antagonist control
DD_flex = DescendingDrive__Pool(
    n=nDD // 2, poisson_random_process_order=DDorder, timestep__ms=dt
)
DD_ext = DescendingDrive__Pool(
    n=nDD // 2, poisson_random_process_order=DDorder, timestep__ms=dt
)
Ia = AffIa__Pool(n=nIa, timestep__ms=dt)
II = AffII__Pool(n=nII, timestep__ms=dt)
Ib = AffIb__Pool(n=nIb, timestep__ms=dt)
gII = GII__Pool(n=ngII)
gIb = GIb__Pool(n=ngIb)
gMN = {
    "name": "gMN",
    "gDyn": gDyn,  # [Hz] Fusimotor Dynamic drive
    "gStat": gStat,
}  # [Hz] Fusimotor Static drive


gto = GolgiTendonOrganModel(
    simulation_time__ms=tstop,
    time_step__ms=dt,
    gto_parameters=GolgiTendonOrganModel.create_default_gto_parameters(),
)

# Create flexor and extensor muscle models
hill_flexor = HillModel(
    simulation_time__ms=tstop,
    time_step__ms=dt,
    muscle_parameters=HillModel.create_default_muscle_parameters(),
    n_motor_units_type1=nType1,
    n_motor_units_type2=nType2,
    initial_joint_angle__deg=artAng[0],
    muscle_role="flexor",
)

hill_extensor = HillModel(
    simulation_time__ms=tstop,
    time_step__ms=dt,
    muscle_parameters=HillModel.create_default_muscle_parameters(),
    n_motor_units_type1=nType1,
    n_motor_units_type2=nType2,
    initial_joint_angle__deg=artAng[0],
    muscle_role="extensor",
)

spin = SpindleModel(
    simulation_time__ms=tstop,
    time_step__ms=dt,
    spindle_parameters=SpindleModel.create_default_spindle_parameters(),
)


def spkEvent(i, muscle, delay):
    """
    Spike event callback for motor neuron activation.

    Parameters
    ----------
    i : int
        Global motor neuron ID from the network (can exceed muscle capacity)
    muscle : HillModel
        The target muscle model (either flexor or extensor)
    delay : float
        Spike delay in ms
    """
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
    i = next(step_counter)

    if round(h.t, 3) % 100 == 0:
        print("Simulation time: {} ms".format(round(h.t, 3)))

    current_angle = artAng[i]

    # Integrate both muscles with same joint angle
    L_flex, V_flex, A_flex = muscle_flex.integrate(current_angle)
    L_ext, V_ext, A_ext = muscle_ext.integrate(current_angle)

    # Use flexor muscle kinematics for spindle feedback (primary muscle)
    Iay, IIy = spin.integrate(L_flex, V_flex, A_flex, gMN["gDyn"][i], gMN["gStat"][i])

    # Combined force feedback for GTO
    f_flex = muscle_flex.F0 * muscle_flex.muscle_force[i]
    f_ext = muscle_ext.F0 * muscle_ext.muscle_force[i]
    Iby = golgi.integrate(f_flex + f_ext)

    # CLOSED-LOOP: Compute net torque from both muscles and update joint angle
    new_angle, _ = joint_dyn.integrate(
        torque__Nm=muscle_flex.signed_muscle_torque[i]
        + muscle_ext.signed_muscle_torque[i],
        dt__s=dt * 0.001,
    )
    # Update artAng array if we haven't reached the end
    if i < len(artAng) - 1:
        artAng[i + 1] = new_angle

    # Descending drive processing - separate populations for each muscle
    for DDcell in popD["DD_flex"]:
        if DDcell.integrate(DDdrive_flexor[i]):
            spike_time = h.t + 1
            if spike_time < tstop:  # Avoid spikes at or after tstop
                ncD["cmd->DD_flex"][DDcell.pool_ID].event(spike_time)

    for DDcell in popD["DD_ext"]:
        if DDcell.integrate(DDdrive_extensor[i]):
            spike_time = h.t + 1
            if spike_time < tstop:  # Avoid spikes at or after tstop
                ncD["cmd->DD_ext"][DDcell.pool_ID].event(spike_time)

    for Ia in popD["Ia"]:
        if Iay >= Ia.RT:
            if Ia.integrate(Iay):
                spike_time = h.t + Ia.axonDelay
                if spike_time < tstop:  # Avoid spikes at or after tstop
                    ncD["Spindle->Ia"][Ia.pool_ID].event(spike_time)
    for II in popD["II"]:
        if IIy >= II.RT:
            if II.integrate(IIy):
                spike_time = h.t + II.axonDelay
                if spike_time < tstop:  # Avoid spikes at or after tstop
                    ncD["Spindle->II"][II.pool_ID].event(spike_time)
    for Ib in popD["Ib"]:
        if Iby >= Ib.RT:
            if Ib.integrate(Iby):
                spike_time = h.t + Ib.axonDelay
                if spike_time < tstop:  # Avoid spikes at or after tstop
                    ncD["GTO->Ib"][Ib.pool_ID].event(spike_time)


# ========================================
# Create neural network using new Network API
network = Network(
    {
        "DD_flex": DD_flex,  # Flexor descending drive
        "DD_ext": DD_ext,  # Extensor descending drive
        "aMN_flex": aMN_flex,  # Flexor alpha motoneurones
        "aMN_ext": aMN_ext,  # Extensor alpha motoneurones
        "Ia": Ia,  # Afferent Ia
        "II": II,  # Afferent II
        "Ib": Ib,  # Afferent Ib
        "gII": gII,  # Group II interneurones
        "gIb": gIb,  # Group Ib interneurones
        "gMN": gMN,
    }
)

# Add neural connections - separate DD populations for each muscle
network.connect("DD_flex", "aMN_flex", probability=0.3, weight__μS=0.05)
network.connect("DD_ext", "aMN_ext", probability=0.3, weight__μS=0.05)
network.connect("Ia", "aMN_flex", probability=0.3, weight__μS=0.05)
network.connect("Ia", "aMN_ext", probability=0.3, weight__μS=0.05)
network.connect("II", "gII", probability=0.6, weight__μS=0.0073)
network.connect("gII", "aMN_flex", probability=0.2, weight__μS=0.05)
network.connect("gII", "aMN_ext", probability=0.2, weight__μS=0.05)
network.connect("Ib", "gIb", probability=0.5, weight__μS=0.0073)
network.connect("gIb", "aMN_flex", probability=0.1, weight__μS=-0.05)
network.connect("gIb", "aMN_ext", probability=0.1, weight__μS=-0.05)

# Motor neuron to muscle connections for both muscles
network.connect_to_muscle(
    "aMN_flex", muscle=hill_flexor, activation_callback=spkEvent, weight__μS=1.0
)
network.connect_to_muscle(
    "aMN_ext", muscle=hill_extensor, activation_callback=spkEvent, weight__μS=1.0
)

# External input connections using new Network API - separate for each muscle
network.connect_from_external("cmd_flex", "DD_flex", weight__μS=1.0)
network.connect_from_external("cmd_ext", "DD_ext", weight__μS=1.0)
network.connect_from_external("spindle", "Ia", weight__μS=1.0)
network.connect_from_external("spindle", "II", weight__μS=1.0)
network.connect_from_external("gto", "Ib", weight__μS=1.0)

# Get individual NetCons for manual triggering (needed by eachStep function)
ncD = {
    "cmd->DD_flex": network.get_netcons("cmd_flex", "DD_flex"),
    "cmd->DD_ext": network.get_netcons("cmd_ext", "DD_ext"),
    "Spindle->Ia": network.get_netcons("spindle", "Ia"),
    "Spindle->II": network.get_netcons("spindle", "II"),
    "GTO->Ib": network.get_netcons("gto", "Ib"),
}

# Models dictionary for SimulationRunner
models = {
    "hill_flexor": hill_flexor,
    "hill_extensor": hill_extensor,
    "spin": spin,
    "gto": gto,
    "joint": joint_dynamics,
}


# Create step callback function with step counter access
def step_callback(step_counter):
    return eachStep(
        muscle_flex=hill_flexor,
        muscle_ext=hill_extensor,
        spin=spin,
        golgi=gto,
        popD=network.populations,  # Get populations from network
        ncD=ncD,
        gMN=gMN,
        joint_dyn=joint_dynamics,
        step_counter=step_counter,
    )


# Run simulation using SimulationRunner
runner = SimulationRunner(
    network=network,
    models=models,
    step_callback=step_callback,
)

results = runner.run(
    duration__ms=tstop,
    timestep__ms=dt,
    membrane_recording={
        "aMN_flex": [0, 5, 10, 15, 20, 30, 40, 50, 60, 70],
        "aMN_ext": [0, 5, 10, 15, 20, 30, 40, 50, 60, 70],
    },
)

# 1. Raster plot of all populations
fig1, ax1 = plt.subplots(1, 1, figsize=(12, 8))
plot_raster_spikes(
    results,
    [ax1],
    populations=[
        "aMN_flex",
        "aMN_ext",
        "Ia",
        "II",
        "gII",
        "gIb",
        "Ib",
        "DD_flex",
        "DD_ext",
    ],
    time_range=(0, tstop),
    title="Raster Plot - Antagonist Muscles",
)
plt.show()

# 2. Membrane potential traces for flexor motor neurons
fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8))
plot_membrane_traces(
    results,
    [ax2],
    population="aMN_flex",
    cell_indices=[0, 10, 20, 30, 40],
    time_range=(0, tstop),
    title="Flexor Motor Neuron Membrane Potential",
)
plt.show()

# 3. Membrane potential traces for extensor motor neurons
fig3, ax3 = plt.subplots(1, 1, figsize=(12, 8))
plot_membrane_traces(
    results,
    [ax3],
    population="aMN_ext",
    cell_indices=[0, 10, 20, 30, 40],
    time_range=(0, tstop),
    title="Extensor Motor Neuron Membrane Potential",
)
plt.show()

# 4. Flexor muscle dynamics
fig4, axes4 = plt.subplots(5, 1, figsize=(12, 16))
plot_muscle_dynamics(
    results,
    artAng,
    time,
    axes4,
    muscle_name="hill_flexor",
    include_signals=["artAng", "L", "force", "torque"],
    include_activations=["TypeI", "TypeII"],
    normalize=True,
    time_range=(0, tstop),
    title="Flexor Muscle Dynamics",
)
plt.tight_layout()
plt.show()

# 5. Extensor muscle dynamics
fig5, axes5 = plt.subplots(5, 1, figsize=(12, 16))
plot_muscle_dynamics(
    results,
    artAng,
    time,
    axes5,
    muscle_name="hill_extensor",
    include_signals=["artAng", "L", "force", "torque"],
    include_activations=["TypeI", "TypeII"],
    normalize=True,
    time_range=(0, tstop),
    title="Extensor Muscle Dynamics",
)
plt.tight_layout()
plt.show()

# 6. Antagonist muscle comparison (net torque analysis)
fig6, axes6 = plt.subplots(3, 1, figsize=(12, 12))
plot_antagonist_muscle_comparison(
    results,
    artAng,
    time,
    axes6,
    flexor_name="hill_flexor",
    extensor_name="hill_extensor",
    include_signals=["artAng", "force", "torque"],
    time_range=(0, tstop),
    title="Antagonist Muscle Comparison - Flexor vs Extensor",
)
plt.tight_layout()
plt.show()

# 7. Spindle dynamics - multiple subplots for different signal groups
fig7, axes7 = plt.subplots(4, 1, figsize=(12, 16))
plot_spindle_dynamics(
    results,
    axes7,
    muscle_name="hill_flexor",
    include_signals=["L"],
    include_activations=["Bag1", "Bag2", "Chain"],
    include_tensions=["Bag1", "Bag2", "Chain"],
    include_afferents=["Ia", "II"],
    time_range=(0, tstop),
    title="Spindle Model Dynamics",
)
plt.tight_layout()
plt.show()

# 8. GTO dynamics - multiple subplots for force and firing rate
fig8, axes8 = plt.subplots(2, 1, figsize=(12, 8))
plot_gto_dynamics(
    results,
    axes8,
    muscle_name="hill_flexor",
    include_signals=["force", "Ib"],
    time_range=(0, tstop),
    title="GTO Model Dynamics",
)
plt.tight_layout()
plt.show()
