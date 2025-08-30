from itertools import count

import numpy as np

from myogen import setup_myogen

setup_myogen()

from neuron import h

from myogen.simulator.neuron._cython._emg import _sEMG__Cython
from myogen.simulator.neuron.muscle import HillModel
from myogen.simulator.neuron.network import create_network
from myogen.simulator.neuron.pops import *
from myogen.simulator.neuron.proprioception import (
    GolgiTendonOrganModel,
    SpindleModel,
)
from myogen.simulator.neuron.joint_dynamics import JointDynamics

# Simulation time
dt = 0.0125
tstop = 10e3
time = np.arange(0, tstop + dt, dt)

# Descending Drive

DDdrive = np.interp(time, [0, tstop // 2, tstop], [0, 80, 0])
DDorder = 16

# Fusimotor drive dynamic gDyn and static gStat
gDynDrive = 70
gStatDrive = 70
gDyn = np.interp(
    time, [0, 250, 251, tstop], [0, 0] + [gDynDrive] * 2
) + np.random.normal(0, 3, len(time))
gStat = np.interp(
    time, [0, 250, 251, tstop], [0, 0] + [gStatDrive] * 2
) + np.random.normal(0, 3, len(time))

# Spin parameters
deaffII = True
deaffIa = True
species = "human"

# fileName = 'modals_nIa'+ str(nIa) + '_DDo' + str(DDorder) # .pickle is pos added
fileName = "plato2_nIa100_DDf20_DDo1"
folderName = "mvc"  #'reflex'


# Initialize joint dynamics for closed-loop control
joint_dynamics = JointDynamics(
    inertia__kg_m2=0.01,  # Finger/hand inertia
    damping__Nm_s_per_rad=0.005,  # Joint damping
    stiffness__Nm_per_rad=0.0,  # No spring stiffness
    initial_angle__deg=0,  # Starting angle
    initial_velocity__deg_per_s=0.0,
)

# Initialize angle array - will be computed dynamically
artAng = np.zeros_like(time)
artAng[0] = 20.0  # Initial condition

# Number of neurons
nIa = 100  # 210 # 315 # 420

nII = 25  # 210 # 315 # 420
ngII = 25  # 210 # 315 # 420

nIb = 25  # 220
ngIb = 25  # 220

nDD = 400  # 400 	# 400 Desceding Drive number

nType1 = 102  # 102 Alpha MN fiber type I
nType2 = 18  # 18 alpha MN fiber type II

# save Parameters
save = {
    "fileName": fileName,
    "folder": folderName,
    "time": time,
    "hill": ["L", "force", "torque"],
    "raster": ["aMN", "Ia", "II", "gII", "gIb", "Ib", "DD"],
    "traces": [("aMN", [0, 5, 10, 15, 20, 30, 40, 50, 60, 70])],
    "emg": True,
    "spin": True,
    "golgi": True,
}

# Simulation Parameters
sim = {
    "tstop": tstop,  # [ms] Simlation time
    "dt": dt,  # [ms] Integration time step
    "time": time,
    "artAng": artAng,
    "celsius": 36,  # [^oC]
    "getEMG": False,
    "pltResults": True,
    "saveResults": True,
}

######################
### PLOT CONFIGURATION
######################

# 'include'	: ['aMN','Ia','II','gII','gIb','Ib'],
pltRaster = {
    "include": ["aMN", "Ia", "II", "gII", "gIb", "Ib", "DD"],
    "timeRange": (0, sim["tstop"]),
    "figsize": (12, 8),
    "rKey": "raster",
    "title": "Rater plot",
    "saveFig": None,
}

# 'include' 	: [('aMN',[0,10,20,30,40]),
# ('gII',[0,50,100,150,200])],
pltTraces = {
    "include": [("aMN", [0, 10, 20, 30, 40])],
    "timeRange": (0, sim["tstop"]),
    "figsize": (12, 8),
    "title": "Transmebrane potential",
    "rKey": "traces",
    "saveFig": None,
}

pltHill = {
    "include": ["artAng", "L", ("act", ["TypeI", "TypeII"]), "force", "torque"],
    "norm": True,
    "timeRange": (0, sim["tstop"]),
    "figsize": (12, 8),
    "title": "Muscle hill model dynamics",
    "rKey": "hill",
    "saveFig": None,
}

pltSpin = {
    "include": [
        "L",
        ("act", ["Bag1", "Bag2", "Chain"]),
        ("tension", ["Bag1", "Bag2", "Chain"]),
        ("aff", ["Ia", "II"]),
    ],
    "timeRange": (0, sim["tstop"]),
    "figsize": (12, 8),
    "title": "Spindle model dynamics",
    "rKey": "spin",
    "saveFig": None,
}

pltGTO = {
    "include": ["force", "Ib"],
    "timeRange": (0, sim["tstop"]),
    "figsize": (12, 8),
    "title": "GTO model dynamics",
    "rKey": "golgi",
    "saveFig": None,
}

pltISIHist = {}
pltConn = {}

labels = {
    "artAng": "art.Ang.[deg]",
    "L": "Length [L0]",
    "V": "Strech Vel. [L0/s]",
    "A": "Strech Acc. [L0/s^2]",
    "act": "Activation [a.u.]",
    "force": "Force [F0]",
    "tension": "Tension [F0]",
    "torque": "Torque [F0.cm]",
    "aff": "Firing Rate [Imps/s]",
    "affIb": "Firing Rate [Imps/s]",
    "time": "Time [ms]",
}

analysis = {
    "pltRaster": pltRaster,
    "pltTraces": pltTraces,
    "pltHill": pltHill,
    "pltSpin": pltSpin,
    "pltGTO": pltGTO,
    "labels": labels,
}

naMN = nType1 + nType2

aMN = AlphaMN__Pool(n=naMN, initial_voltage__mV=-67)

DD = DescendingDrive__Pool(n=nDD, poisson_random_process_order=DDorder, timestep__ms=dt)

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

pop = {
    "DD": DD,  # Descending drive
    "aMN": aMN,  # Alpha motoneurones
    "Ia": Ia,  # Afferent Ia
    "II": II,  # Afferent II
    "Ib": Ib,  # Afferent Ib
    "gII": gII,  # Group II interneurones
    "gIb": gIb,  # Group Ib interneurones
    "gMN": gMN,
}  # Gamma motoneurones


"""
Connection Parameters
"""
# Descending drive -> aMN and -> gII connections
DD_aMN = {
    "connP": 0.30,  # Connection probability
    "source": "DD",
    "target": "aMN",
    "w": 0.05,
}  # Synaptic weigth

# alpha MN -> Muscle connection
aMN_mus = {
    "connP": 1,  # Connection probability
    "source": "aMN",
    "target": None,  # Muscle
    "w": None,
}  # Synaptic weigth

cmd_DD = {
    "connP": 1,  # Connection probability
    "source": None,
    "target": "DD",
    "w": None,
}  # [uS]  Synaptic weigth

# Spindle -> Afferent Ia -> aMN connection
spin_Ia = {
    "connP": 1,  # Connection probability
    "source": None,
    "target": "Ia",
    "w": None,
}  # [uS]  Synaptic weigth

Ia_aMN = {
    "connP": 0.3,  # Connection probability
    "source": "Ia",
    "target": "aMN",
    "w": 0.05,
}  # [uS] Synaptic weigth 0.05 for Powers2017

# Spindle -> Afferent II -> gII -> aMN Connections
spin_II = {
    "connP": 1,  # Connection probability
    "source": None,
    "target": "II",
    "w": None,
}  # [uS]  Synaptic weigth

II_gII = {
    "connP": 0.6,  # Connection probability
    "source": "II",
    "target": "gII",
    "w": 0.0073,
}  # [uS]  Synaptic weigth

gII_aMN = {
    "connP": 0.2,  # Connection probability
    "source": "gII",
    "target": "aMN",
    "w": 0.05,
}  # [uS]  Synaptic weigth

# GTO -> Afferent Ib -> gIb -> aMN Connections
GTO_Ib = {
    "connP": 1,  # Connection probability
    "source": None,
    "target": "Ib",
    "w": None,
}  # [uS]  Synaptic weigth

# was using 0.6 connectiviy
Ib_gIb = {
    "connP": 0.5,  # Connection probability
    "source": "Ib",
    "target": "gIb",
    "w": 0.0073,
}  # [uS]  Synaptic weigth

# was using 0.6 connectivy
gIb_aMN = {
    "connP": 0.1,  # Connection probability
    "source": "gIb",
    "target": "aMN",
    "w": -0.05,
}  # [uS]  Synaptic weigth

conn = {
    "cmd->DD": cmd_DD,
    "DD->aMN": DD_aMN,
    # 'DD->gII' 		: DD_gII,
    # 'DD->gIb' 		: DD_gIb,
    "aMN->Muscle": aMN_mus,
    "Spindle->Ia": spin_Ia,
    "Ia->aMN": Ia_aMN,
    "Spindle->II": spin_II,
    "II->gII": II_gII,
    "gII->aMN": gII_aMN,
    "GTO->Ib": GTO_Ib,
    "Ib->gIb": Ib_gIb,
    "gIb->aMN": gIb_aMN,
}


gto = GolgiTendonOrganModel(
    simulation_time__ms=tstop,
    time_step__ms=dt,
    gto_parameters=GolgiTendonOrganModel.create_default_gto_parameters(),
)

hill = HillModel(
    simulation_time__ms=tstop,
    time_step__ms=dt,
    muscle_parameters=HillModel.create_default_muscle_parameters(),
    n_motor_units_type1=nType1,
    n_motor_units_type2=nType2,
    initial_joint_angle__deg=artAng[0],
)

spin = SpindleModel(
    simulation_time__ms=tstop,
    time_step__ms=dt,
    spindle_parameters=SpindleModel.create_default_spindle_parameters(),
)

sEMG = {
    "morpho": 2,  # [1-4] map to [circle,ring,pizza,ellipse]
    "csa": 150e-6,
    "fat": 0.2e-3,
    "skin": 0.1e-3,
    "theta": 0.9,
    "prop": 0.4,
    "first": 21,
    "ratio": 84,
    "t1m": 0.7,
    "t1dp": 0.5,
    "t2m": 1,
    "t2dp": 0.25,
    "v1": 1,
    "v2": 11.4,
    "d1": 1.8,
    "d2": 1.4,
    "ampk": 5e-3,
    "durak": 100,
    "noise": 0.05,  # [mV]
    "filt": True,
    "lc": 10,  # [Hz]
    "hc": 500,  # [Hz]
    "rmsWin": 8,
}  # [ms]


def spkEvent(i, muscle, delay):
    muscle.add_spike(i, delay)


def eachStep(muscle, spin, golgi, popD, ncD, gMN, DD, joint_dyn):
    global step
    i = next(step)
    if round(h.t, 3) % 100 == 0:
        print("Simulation time: {} ms".format(round(h.t, 3)))

    # Use current joint angle (computed from previous step)
    current_angle = artAng[i] if i < len(artAng) else joint_dyn.angle__deg
    L, V, A = muscle.integrate(current_angle)
    Iay, IIy = spin.integrate(L, V, A, gMN["gDyn"][i], gMN["gStat"][i])
    f = muscle.F0 * muscle.muscle_force[i]
    Iby = golgi.integrate(f)

    # CLOSED-LOOP: Compute next joint angle from muscle torque
    if i < len(artAng) - 1:
        new_angle, _ = joint_dyn.integrate(
            torque__Nm=muscle.muscle_torque[i],
            dt__s=dt * 0.001,  # Convert ms to seconds
        )
        artAng[i + 1] = new_angle
    for DDcell in popD["DD"]:
        spk = DDcell.integrate(DDdrive[i])
        if spk == 1:
            ncD["cmd->DD"][DDcell.class_ID].event(h.t + 1)

    for Ia in popD["Ia"]:
        if Iay >= Ia.RT:
            spk = Ia.integrate(Iay)
            if spk == 1:
                ncD["Spindle->Ia"][Ia.class_ID].event(h.t + Ia.axonDelay)
                # print('{} Spk: {}'.format(Ia,round(h.t,1)))
    for II in popD["II"]:
        if IIy >= II.RT:
            spk = II.integrate(IIy)
            if spk == 1:
                # print('{} Spk: {}'.format(II,round(h.t,1)))
                ncD["Spindle->II"][II.class_ID].event(h.t + II.axonDelay)
    for Ib in popD["Ib"]:
        if Iby >= Ib.RT:
            spk = Ib.integrate(Iby)
            if spk == 1:
                # print('{} Spk: {}'.format(Ib,round(h.t,1)))
                ncD["GTO->Ib"][Ib.class_ID].event(h.t + Ib.axonDelay)


def setVinit(secL, vHold):
    for sec, v in zip(secL, vHold):
        sec.v = v


def pltResults(rslt, cfg):
    #######################
    print("Start Plotting...")
    #######################

    import matplotlib.pyplot as plt

    flag = 0
    for value in analysis.values():
        try:
            if value["include"] is not None:
                flag = 1
        except KeyError:
            flag = 1
    if flag == 0:
        return -1

    # vName = ['L','V','A','force','torque']
    # if cfg['analysis']['pltHill']['norm']:
    # 	nVal = [muscle.L0]*3 + [muscle.F0]*2
    # 	nlab = ['Length [mm]','Strech Vel. [mm/s]',
    # 	'Strech acc. [mm/s^2]','Force[N]','Torque[N.cm]']
    # 	nFactor = [1e3]*3 + [1] + [1e2]
    # else:
    # 	nVal = [1]*5
    # nVar = [muscle.L,muscle.V,muscle.A,muscle.force,
    # 												muscle.torque]
    # for name,var,value,factor,l in zip(vName,nVar,nVal,nFactor,nlab):
    # 	muscleD[name] = np.asarray(var)*value*factor
    # 	cfg['analysis']['labels'][name] = l

    for key, value in analysis.items():
        if "include" not in value.keys():
            continue
        if value["include"] is not None:
            if value["figsize"] is not None:
                plt.figure(figsize=value["figsize"])
            else:
                plt.figure()
            nPlots = len(value["include"])
            for i, var in enumerate(value["include"]):
                if i == 0:
                    ax = plt.subplot(nPlots, 1, i + 1)
                    plt.xlim(value["timeRange"])
                    plt.title(value["title"])
                elif i == (nPlots - 1):
                    plt.subplot(nPlots, 1, i + 1, sharex=ax)
                    plt.xlabel(analysis["labels"]["time"])
                else:
                    plt.subplot(nPlots, 1, i + 1, sharex=ax)
                if len(var) > 1 and not isinstance(var[1], str):
                    for cell in var[1]:
                        plt.plot(
                            sim["time"],
                            rslt[value["rKey"]][var[0]][cell],
                            label=var[0] + "[" + str(cell) + "]",
                        )
                    plt.legend(loc="upper left")
                elif key == "pltRaster":
                    if len(rslt[value["rKey"]]["idvec"][var]) >= 1:
                        minV = min(rslt[value["rKey"]]["idvec"][var])
                        plt.plot(
                            rslt[value["rKey"]]["spkvec"][var],
                            rslt[value["rKey"]]["idvec"][var] - minV,
                            ".",
                            ms=0.8,
                            label=var,
                        )
                        # plt.ylim(-1, pop[var].n)
                        plt.legend(loc="upper left")
                else:
                    if var == "artAng":
                        plt.plot(sim["time"], sim["artAng"])
                    else:
                        plt.plot(sim["time"], rslt[value["rKey"]][var])
                if isinstance(var, str) and var in analysis["labels"]:
                    plt.ylabel(analysis["labels"][var])
                else:
                    plt.ylabel("#")
    #######################
    print("End Plotting.")
    #######################
    plt.show()
    return 1


# ========================================
h.load_file("stdrun.hoc")

step = count(0)

# NEURON POOLS CONFIG

popD = pop

idvec = {}  # Vectors for each type of neuron
spkvec = {}  # Vectors for each type of neuron
for pop in popD.keys():
    idvec[pop] = h.Vector()
    spkvec[pop] = h.Vector()
rasterD = {"idvec": idvec, "spkvec": spkvec}
ncD = create_network(
    populations=popD,
    connections_config=conn,
    id_vector=idvec,
    spike_vector=spkvec,
    muscle_callback=spkEvent,
    muscle=hill,
)

pltL = analysis["pltTraces"]["include"]
if pltL is None:
    pltL = []
saveL = save["traces"]
if saveL is None:
    saveL = []
toTrace = {}
for pName, pList in pltL + saveL:
    if pName in toTrace:
        toTrace[pName] = set(toTrace[pName] + pList)
    else:
        toTrace[pName] = pList
tracesD = {}
for popN, popL in zip(toTrace.keys(), toTrace.values()):
    popTraceD = {}
    for cell in popL:
        v = h.Vector()
        v.record(popD[popN][cell].soma(0.5)._ref_v)
        popTraceD[cell] = v
    tracesD[popN] = popTraceD

# NEURON SIM CONFIG
lamb1 = lambda: eachStep(
    muscle=hill,
    spin=spin,
    golgi=gto,
    popD=popD,
    ncD=ncD,
    gMN=gMN,
    DD=DD,
    joint_dyn=joint_dynamics,
)
esg = h.CVode().extra_scatter_gather(0, lamb1)

## SIMULATION CONFIG
sections = []
voltages = []
for pop in popD.values():
    try:
        secL, vHold = pop.get_initialization_data()
        sections += secL
        voltages += vHold
    except Exception as e:
        continue

lamb2 = lambda: setVinit(sections, voltages)
fih = h.FInitializeHandler(0, lamb2)
h.celsius = sim["celsius"]
h.tstop = sim["tstop"]
h.dt = sim["dt"]
print("Starting Simulation")
h.run()
print("Simulation end")

# Muscle Hill model output variables
muscleD = {}
vName = ["L", "force", "torque"]
nVar = [hill.muscle_length, hill.muscle_force, hill.muscle_torque]
for name, var in zip(vName, nVar):
    muscleD[name] = np.asarray(var)
muscleD["act"] = {
    "TypeI": np.asarray(hill.type1_activation),
    "TypeII": np.asarray(hill.type2_activation),
}

# Fusimotor Spindle model output variables
spinD = {
    "L": muscleD["L"],
    "act": {
        "Bag1": np.asarray(spin.bag1_activation),
        "Bag2": np.asarray(spin.bag2_activation),
        "Chain": np.asarray(spin.chain_activation),
    },
    "tension": {
        "Bag1": np.asarray(spin.intrafusal_tensions[0]),
        "Bag2": np.asarray(spin.intrafusal_tensions[1]),
        "Chain": np.asarray(spin.intrafusal_tensions[2]),
    },
    "aff": {
        "Ia": np.asarray(spin.primary_afferent_firing__Hz),
        "II": np.asarray(spin.secondary_afferent_firing__Hz),
    },
}

# Golgi tendon organ model output variables
golgiD = {"force": muscleD["force"], "Ib": np.asarray(gto.ib_afferent_firing__Hz)}

rslt = {
    "raster": rasterD,
    "traces": tracesD,
    "hill": muscleD,
    "spin": spinD,
    "golgi": golgiD,
}


# EMG generation
if sim["getEMG"] and len(rslt["raster"]["idvec"]["aMN"]) > 1:
    print("Generating EMG...")
    emgOb = _sEMG__Cython({"sEMG": sEMG})
    emgEMG = emgOb.sEMG(
        np.array(rslt["raster"]["idvec"]["aMN"]),
        np.array(rslt["raster"]["spkvec"]["aMN"]),
    )
    emgRMS = emgOb.movingAverage(emgEMG, sEMG["rmsWin"])
    rslt["emg"] = {"sEMG": np.array(emgEMG), "sRMS": np.array(emgRMS)}
    print("Done")

if sim["pltResults"]:
    pltResults(rslt, None)

simD = {"pop": popD, "muscle": hill, "nc": ncD, "results": rslt}
