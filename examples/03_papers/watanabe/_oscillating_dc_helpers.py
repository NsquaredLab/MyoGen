"""
Internal helpers for 02_optimize_oscillating_dc.py.

Not a gallery example — imported by 02_optimize_oscillating_dc.py and
_optimize_dc_worker.py. Contains all simulation logic, configuration
constants, and Optuna storage helpers so that both the gallery script and
the worker subprocesses share exactly one copy of the code.
"""

import json
import os

os.environ["MPLBACKEND"] = "Agg"
if "DISPLAY" in os.environ:
    del os.environ["DISPLAY"]

import warnings
from pathlib import Path

import numpy as np
import optuna
import quantities as pq
from neo import Block, Segment, SpikeTrain
from neuron import h

from myogen import get_random_generator, set_random_seed
from myogen.simulator import RecruitmentThresholds
from myogen.simulator.core.force.force_model import ForceModel
from myogen.simulator.neuron import Network
from myogen.simulator.neuron.populations import AlphaMN__Pool, DescendingDrive__Pool
from myogen.utils.helper import calculate_firing_rate_statistics
from myogen.utils.nmodl import load_nmodl_mechanisms

warnings.filterwarnings("ignore")

##############################################################################
# Configuration
# -------------

# Simulation parameters
SIMULATION_TIME_MS = 5000.0  # Shorter simulation for efficient optimization
TIMESTEP_MS = 0.1
N_MOTOR_UNITS = 800  # Watanabe specification

# Force model parameters
RECORDING_FREQUENCY__HZ = 2048
LONGEST_DURATION_RISE_TIME__MS = 90.0
CONTRACTION_TIME_RANGE = 3

# Oscillation parameters (Watanabe specification)
OSC_FREQUENCY__HZ = 20.0  # 20 Hz physiological tremor
OSC_AMPLITUDE__HZ = 20.0  # Amplitude of oscillation

# Optimization settings
N_TRIALS = 25  # Increase for production
TIMEOUT_SECONDS = 3600

# Network parameters (Watanabe specification)
N_DD_NEURONS = 400
DD_CONNECTIVITY = 0.3
SYNAPTIC_WEIGHT = 0.05

# Directories
RESULTS_DIR = Path(__file__).parent / "results" / "watanabe_optimization"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

##############################################################################
# Load Reference Force Data
# --------------------------

_reference_file = RESULTS_DIR / "force_reference.json"

if not _reference_file.exists():
    raise FileNotFoundError(
        f"Reference force not found: {_reference_file}\nRun 01_compute_baseline_force.py first!"
    )

with open(_reference_file, "r") as _f:
    _reference_results = json.load(_f)

REFERENCE_FORCE__N = _reference_results["force"]["mean__N"]
REFERENCE_DRIVE__HZ = _reference_results["network_parameters"]["dd_drive__Hz"]
TARGET_FORCE__N = REFERENCE_FORCE__N  # Match the reference force
MAX_FORCE_N = _reference_results["force_scaling"]["max_force__N"]

##############################################################################
# Initialize NEURON
# -----------------

set_random_seed(42)
load_nmodl_mechanisms()
h.secondorder = 2

##############################################################################
# Simulation Function
# -------------------


def run_simulation_with_oscillating_drive(dc_offset, recruitment_thresholds):
    """
    Run network simulation with oscillating drive and compute resulting force.

    Drive pattern: dc_offset + OSC_AMPLITUDE * sin(2*pi*OSC_FREQUENCY*t)

    Parameters
    ----------
    dc_offset : float
        DC offset component of oscillating drive (Hz)
    recruitment_thresholds : np.ndarray
        Motor unit recruitment thresholds

    Returns
    -------
    tuple
        (force_mean, n_active, fr_mean, fr_std)
    """
    # Create motor neuron pool
    motor_neuron_pool = AlphaMN__Pool(
        recruitment_thresholds__array=recruitment_thresholds,
        config_file="alpha_mn_default.yaml",
    )

    # Create descending drive pool (Poisson - Watanabe specification)
    descending_drive_pool = DescendingDrive__Pool(
        n=N_DD_NEURONS,
        timestep__ms=TIMESTEP_MS * pq.ms,
        process_type="poisson",
    )

    # Build network
    network = Network({"DD": descending_drive_pool, "aMN": motor_neuron_pool})
    network.connect(
        source="DD",
        target="aMN",
        probability=DD_CONNECTIVITY,
        weight__uS=SYNAPTIC_WEIGHT * pq.uS,
    )
    network.connect_from_external(source="cortical_input", target="DD", weight__uS=1.0 * pq.uS)
    dd_netcons = network.get_netcons("cortical_input", "DD")

    # Setup spike recording
    mn_spike_recorders = []
    for cell in motor_neuron_pool:
        spike_recorder = h.Vector()
        nc = h.NetCon(cell.soma(0.5)._ref_v, None, sec=cell.soma)
        nc.threshold = 50
        nc.record(spike_recorder)
        mn_spike_recorders.append(spike_recorder)

    # Create oscillating drive signal: DC + amplitude * sin(2*pi*freq*t)
    time_points = int(SIMULATION_TIME_MS / TIMESTEP_MS)
    time_s = np.arange(time_points) * TIMESTEP_MS / 1000.0

    # Oscillating component
    drive_signal = dc_offset + OSC_AMPLITUDE__HZ * np.sin(2 * np.pi * OSC_FREQUENCY__HZ * time_s)

    # Clip to prevent negative firing rates
    drive_signal = np.clip(drive_signal, 0, None)

    # Add small noise
    drive_signal += np.clip(get_random_generator().normal(0, 1.0, size=time_points), 0, None)

    # Initialize simulation
    h.load_file("stdrun.hoc")
    h.dt = TIMESTEP_MS
    h.tstop = SIMULATION_TIME_MS

    for section, voltage in zip(*motor_neuron_pool.get_initialization_data()):
        section.v = voltage
    for section, voltage in zip(*descending_drive_pool.get_initialization_data()):
        section.v = voltage

    h.finitialize()

    # Run simulation
    step_counter = 0
    while h.t < h.tstop:
        current_drive = drive_signal[min(step_counter, len(drive_signal) - 1)]
        for dd_cell in descending_drive_pool:
            if dd_cell.integrate(current_drive):
                if h.t < h.tstop:
                    dd_netcons[dd_cell.pool__ID].event(h.t + 1)
        h.fadvance()
        step_counter += 1

    # Convert to Neo format
    dt_s = h.dt / 1000.0
    mn_segment = Segment(name="Motor Neurons")
    mn_segment.spiketrains = [
        SpikeTrain(
            recorder.as_numpy() / 1000 * pq.s,
            t_stop=SIMULATION_TIME_MS / 1000 * pq.s,
            sampling_rate=(1 / dt_s * pq.Hz),
            sampling_period=dt_s * pq.s,
            name=f"MN_{i}",
        )
        for i, recorder in enumerate(mn_spike_recorders)
    ]

    spike_train__Block = Block(name="Motor Unit Pool")
    spike_train__Block.segments = [mn_segment]

    # Calculate statistics
    n_active = sum(1 for st in mn_segment.spiketrains if len(st) > 1)
    stats = calculate_firing_rate_statistics(mn_segment.spiketrains)
    fr_mean = float(stats["FR_mean"])
    fr_std = float(stats["FR_std"])

    # Generate force
    force_model = ForceModel(
        recruitment_thresholds=recruitment_thresholds,
        recording_frequency__Hz=RECORDING_FREQUENCY__HZ * pq.Hz,
        longest_duration_rise_time__ms=LONGEST_DURATION_RISE_TIME__MS * pq.ms,
        contraction_time_range_factor=CONTRACTION_TIME_RANGE,
    )

    force_output = force_model.generate_force(spike_train__Block=spike_train__Block)
    force_raw = force_output.magnitude[:, 0]  # Arbitrary units (sum of MU twitches)

    # Normalize force to 0-1 range, then scale to Newtons
    # (ForceModel outputs sum of MU twitch forces, not normalized values)
    force_max_raw = np.max(force_raw)
    force_signal = (force_raw / force_max_raw) * MAX_FORCE_N  # Scale to Newtons

    # Calculate steady-state force
    steady_idx = len(force_signal) // 2
    force_mean = np.mean(force_signal[steady_idx:])

    return force_mean, n_active, fr_mean, fr_std


##############################################################################
# Recruitment Thresholds (module global — built once under seed 42)
# -----------------------------------------------------------------

recruitment_thresholds, _ = RecruitmentThresholds(
    N=N_MOTOR_UNITS,
    recruitment_range__ratio=100,
    deluca__slope=5,
    konstantin__max_threshold__ratio=1.0,
    mode="combined",
)

##############################################################################
# Objective Function
# ------------------


def objective(trial):
    """
    Optimize DC offset to match reference force with oscillation.

    Parameters
    ----------
    trial : optuna.Trial
        Optimization trial

    Returns
    -------
    float
        Relative force error (minimize)
    """
    try:
        # Optimize DC offset (will be lower than constant drive)
        dc_offset = trial.suggest_float("dc_offset", 1.0, 100.0)

        # Run simulation with oscillating drive
        force_mean, n_active, fr_mean, fr_std = run_simulation_with_oscillating_drive(
            dc_offset, recruitment_thresholds
        )

        # Check minimum recruitment
        if n_active < 10:  # At least 1.25% of neurons
            return 1000.0 + (10 - n_active) * 100.0

        # Calculate relative error
        force_error = abs(force_mean - TARGET_FORCE__N) / TARGET_FORCE__N

        # Store metadata
        trial.set_user_attr("force_achieved", float(force_mean))
        trial.set_user_attr("force_error", float(force_error))
        trial.set_user_attr("n_active", n_active)
        trial.set_user_attr("dc_offset__Hz", float(dc_offset))
        trial.set_user_attr("FR_mean", float(fr_mean))
        trial.set_user_attr("FR_std", float(fr_std))

        if trial.number % 5 == 0:
            print(
                f"Trial {trial.number}: "
                f"Force={force_mean:.2f}N (target={TARGET_FORCE__N:.2f}N), "
                f"Error={force_error:.1%}, "
                f"DC={dc_offset:.1f}Hz, "
                f"Active={n_active}/{N_MOTOR_UNITS}"
            )

        return force_error

    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return 1000.0


##############################################################################
# Optuna Study Config + Storage
# ------------------------------

STUDY_NAME = "dc_offset_oscillating_match"
STORAGE_URL = f"sqlite:///{RESULTS_DIR}/optuna_oscillating_dc.db"


def make_storage():
    """Return an RDBStorage with a generous lock timeout for concurrent workers."""
    # connect timeout reduces "database is locked" under concurrent workers
    return optuna.storages.RDBStorage(STORAGE_URL, engine_kwargs={"connect_args": {"timeout": 60}})


##############################################################################
# Public API
# ----------

__all__ = [
    "objective",
    "recruitment_thresholds",
    "STUDY_NAME",
    "STORAGE_URL",
    "make_storage",
    "RESULTS_DIR",
    "N_TRIALS",
    "TIMEOUT_SECONDS",
    "REFERENCE_FORCE__N",
    "REFERENCE_DRIVE__HZ",
    "TARGET_FORCE__N",
    "MAX_FORCE_N",
    "N_MOTOR_UNITS",
    "OSC_FREQUENCY__HZ",
    "OSC_AMPLITUDE__HZ",
    "N_DD_NEURONS",
    "DD_CONNECTIVITY",
    "SYNAPTIC_WEIGHT",
]
