"""
Intramuscular EMG Signals
========================

This example demonstrates how to simulate **intramuscular EMG signals** using
needle electrodes. It shows the complete pipeline from muscle model creation
to EMG signal generation with realistic noise and motor unit detectability.

.. note::
    **Intramuscular EMG** (iEMG) is recorded using needle electrodes inserted
    directly into the muscle tissue. This provides high spatial resolution
    and allows for the detection of individual motor unit action potentials
    (MUAPs). Unlike surface EMG, intramuscular recordings can capture the
    activity of deeper motor units and provide better selectivity.

Key Features:
    - **High spatial resolution**: Needle electrodes can detect individual MUAPs
    - **Deep muscle access**: Can record from muscles not accessible by surface electrodes
    - **Motor unit discrimination**: Individual motor units can be identified and tracked
    - **Realistic noise modeling**: Includes physiological noise and recording artifacts
"""
# sphinx_gallery_thumbnail_number = -1

##############################################################################
# Import Libraries
# ----------------

from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from myogen import simulator
from myogen.utils.types import CURRENT__AnalogSignal, SPIKE_TRAIN__Block

##############################################################################
# Define Parameters
# -----------------
#

# Motor Unit Selection
# ~~~~~~~~~~~~~~~~~~~~
# For decomposition testing or computational efficiency, you can simulate
# only a subset of motor units using the MUs_to_simulate parameter.
#
# Motor units are indexed in **recruitment order** (0 = recruited first).
#
# Options:
# - None: Simulate all motor units (default)
# - List of indices: Simulate specific MUs, e.g., [0, 5, 10, 15, 20]
# - Range: Simulate first N MUs, e.g., list(range(20))
# - Every Nth: Evenly spaced MUs, e.g., list(range(0, 100, 10))

# Example configurations:
MUs_to_simulate = [0, 5, 10, 15, 20, 25, 30]  # Simulate all 100 MUs
# MUs_to_simulate = [0, 5, 10, 15, 20, 25, 30]  # 7 specific MUs
# MUs_to_simulate = list(range(20))  # First 20 MUs only
# MUs_to_simulate = list(range(0, 100, 10))  # Every 10th MU (10 total)

# Electrode parameters
inter_electrode_distance = 0.5  # mm
electrode_position = (0.0, 0.0, 0.0)  # mm (start of muscle)


##############################################################################
# Load Muscle Model
# -------------------
#
# Load the **muscle model** with the generated recruitment thresholds.

muscle: simulator.Muscle = joblib.load("results/synthetic_gen/muscle_model.pkl")

##############################################################################
# Create Intramuscular Electrode Array
# ------------------------------------
#
# Set up a **differential needle electrode** for intramuscular recordings.

electrode = simulator.IntramuscularElectrodeArray(
    num_electrodes=4,
    inter_electrode_distance__mm=inter_electrode_distance,
    differentiation_mode="consecutive",
    position__mm=electrode_position,
    orientation__rad=(-np.pi / 2, 0, -np.pi / 2),  # perpendicular to muscle
    trajectory_distance__mm=0.125,  # mm
    trajectory_steps=1,  # number of steps
)

##############################################################################
# Initialize Intramuscular EMG Simulator
# --------------------------------------
#
# Create the **intramuscular EMG simulator** with the muscle model and electrode.
#
# The ``MUs_to_simulate`` parameter allows selective simulation of specific motor
# units for computational efficiency or decomposition testing. Only the selected
# MUs will have MUAPs computed and convolved with spike trains.

print("Initializing iEMG simulator...")
if MUs_to_simulate is not None:
    print(f"  → Simulating {len(MUs_to_simulate)} selected MUs: {MUs_to_simulate}")
else:
    n_mus = len(muscle.resulting_number_of_innervated_fibers)
    print(f"  → Simulating all {n_mus} MUs")

iemg_sim = simulator.IntramuscularEMG(
    muscle_model=muscle,
    electrode_array=electrode,
    MUs_to_simulate=MUs_to_simulate,
)

##############################################################################
# Calculate Motor Unit Action Potentials
# --------------------------------------
#
# Compute the **MUAPs** for each motor unit at the electrode positions.


print("Computing motor unit action potentials...")
iemg_sim.simulate_muaps()

# iemg_sim = joblib.load("./results/iemg_simulator.pkl")

joblib.dump(iemg_sim, "./results/synthetic_gen/iemg_simulator.pkl")


##############################################################################
# Load Input Currents and Spike Trains
# -----------------------
#

save_path = Path("./results/synthetic_gen")

spike_train__Block: SPIKE_TRAIN__Block = joblib.load(
    save_path / "sinusoidal_dd_spike_trains.pkl"
)
input_current__AnalogSignal: CURRENT__AnalogSignal = joblib.load(
    save_path / "trapezoid_drive_pattern.pkl"
)

##############################################################################
# Simulate Intramuscular EMG
# -------------------------
#
# Generate the final **intramuscular EMG signals** by convolving spike trains with MUAPs.

print("Simulating intramuscular EMG signals...")
emg_signals = iemg_sim.simulate_intramuscular_emg(spike_train__Block=spike_train__Block)

print("Intramuscular EMG simulation completed!")
# Access the first segment (pool) analogsignal
first_emg_signal = emg_signals.segments[0].analogsignals[0]
print(f"Generated EMG shape: {first_emg_signal.shape}")
print(f"  - {first_emg_signal.shape[0]} time samples")
print(f"  - {first_emg_signal.shape[1]} electrode channels")
print(
    f"  - Signal RMS (before noise): {np.sqrt(np.mean(first_emg_signal.magnitude**2)):.3f}"
)

print("Adding realistic noise (SNR = 20 dB)...")
noisy_emg_signals__Block = iemg_sim.add_noise(snr__dB=20)
first_noisy_signal = noisy_emg_signals__Block.segments[0].analogsignals[0]
print(
    f"  - Signal RMS (after noise): {np.sqrt(np.mean(first_noisy_signal.magnitude**2)):.3f}"
)

##############################################################################
# Save EMG Signals
# ----------------
#
# Save the noisy EMG signals for later visualization and analysis.
# This allows you to use the visualization script (09_visualize_emg_channels.py)
# without re-running the entire simulation.

joblib.dump(noisy_emg_signals__Block, save_path / "emg_signals.pkl")
print(f"\nSaved EMG signals to: {save_path / 'emg_signals.pkl'}")

##############################################################################
# Visualize Intramuscular EMG Results
# ----------------------------------
#
# Create an **xkcd-style plot** comparing the **intramuscular EMG** signal
# with the input current, similar to the surface EMG example.
#
# .. note::
#   The intramuscular EMG provides **high spatial resolution** and can detect
#   individual motor unit action potentials (MUAPs) with excellent signal quality.

# Clear matplotlib cache and set up xkcd style
matplotlib.get_cachedir()
with plt.xkcd():
    plt.rcParams.update({"font.size": 24})

    # Create single plot with normalized signals
    fig, ax = plt.subplots(figsize=(12, 6))

    # Get the signals - use first electrode from the Block
    iemg_signal = noisy_emg_signals__Block.segments[0].analogsignals[0][:, 0]
    current_signal = input_current__AnalogSignal[:, 0]  # First current pool

    # Create time axes
    emg_time = iemg_signal.times.rescale("s").magnitude
    current_time = input_current__AnalogSignal.times.rescale("s").magnitude

    # Normalize iEMG signal by dividing by maximum absolute value
    iemg_normalized = iemg_signal / np.max(np.abs(iemg_signal))

    # Normalize current between 0 and 1
    current_normalized = (current_signal - np.min(current_signal)) / (
        np.max(current_signal) - np.min(current_signal)
    )

# Plot both normalized signals on same axis
ax.plot(
    emg_time,
    iemg_normalized,
    linewidth=2,
    label="Intramuscular EMG",
)

with plt.xkcd():
    ax.plot(
        current_time,
        current_normalized,
        linewidth=2,
        label="Input Current",
        alpha=0.7,
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized Amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend()

    sns.despine(trim=True, left=False, bottom=False, right=True, top=True, offset=5)

    plt.title("Normalized Intramuscular EMG and Input Current")

plt.tight_layout()

##############################################################################
# Package Data for Decomposition
# -------------------------------
#
# Create a comprehensive data package containing everything needed for
# EMG decomposition algorithm development and validation.
#
# This package includes:
# - **EMG signal**: The noisy mixed signal to decompose
# - **MUAP templates**: Ground truth motor unit waveforms
# - **Spike trains**: Ground truth firing times for validation
# - **Metadata**: MU indices, sampling rate, electrode configuration, etc.
#
# .. note::
#    This single pkl file contains all data needed for decomposition:
#
#    .. code-block:: python
#
#       import joblib
#       data = joblib.load("decomposition_package.pkl")
#       emg = data['emg_signal']  # Shape: (time_samples, n_electrodes)
#       templates = data['muap_templates']  # List of MUAP waveforms
#       spikes = data['spike_trains']  # Ground truth spike times

print("\n" + "=" * 70)
print("PACKAGING DATA FOR DECOMPOSITION")
print("=" * 70)

# Get motor unit indices first
if MUs_to_simulate is not None:
    mu_indices = MUs_to_simulate
else:
    mu_indices = list(range(len(muscle.resulting_number_of_innervated_fibers)))

# Extract EMG signal data
emg_signal = noisy_emg_signals__Block.segments[0].analogsignals[0]
emg_data = emg_signal.magnitude

# Extract MUAP templates (one per motor unit) - already filtered by MUs_to_simulate
muap_templates = []
muap_durations_ms = []
for segment in iemg_sim.muaps__Block.segments:
    muap = segment.analogsignals[0].magnitude
    muap_templates.append(muap)
    duration_ms = muap.shape[0] / iemg_sim.sampling_frequency__Hz * 1000
    muap_durations_ms.append(duration_ms)

# Extract spike trains (ground truth) - only for selected MUs
spike_trains_list = []
all_spike_trains = spike_train__Block.segments[0].spiketrains
for mu_idx in mu_indices:
    spike_times_ms = all_spike_trains[mu_idx].magnitude  # Already in ms
    spike_trains_list.append(spike_times_ms)

# Create decomposition package
decomposition_package = {
    # ========== SIGNALS ==========
    "emg_signal": emg_data,  # Shape: (time_samples, n_electrodes)
    "muap_templates": muap_templates,  # List of (muap_samples, n_electrodes)
    "spike_trains": spike_trains_list,  # List of spike times in ms

    # ========== MOTOR UNIT INFO ==========
    "mu_indices": mu_indices,  # Global MU indices
    "n_motor_units": len(mu_indices),
    "recruitment_thresholds": muscle.recruitment_thresholds[mu_indices],
    "muap_durations_ms": np.array(muap_durations_ms),

    # ========== TIMING ==========
    "sampling_rate_hz": float(iemg_sim.sampling_frequency__Hz),
    "time_duration_s": float(emg_signal.t_stop.rescale("s").magnitude),
    "time_start_s": float(emg_signal.t_start.rescale("s").magnitude),
    "n_samples": emg_data.shape[0],

    # ========== ELECTRODE CONFIG ==========
    "electrode_positions_mm": electrode.pts,  # Shape: (n_electrodes, 3)
    "n_electrodes": electrode.num_electrodes,
    "inter_electrode_distance_mm": electrode.inter_electrode_distance__mm,

    # ========== NOISE ==========
    "snr_db": 20.0,

    # ========== MUSCLE PROPERTIES ==========
    "muscle_radius_mm": muscle.radius__mm,
    "muscle_length_mm": muscle.length__mm,
    "endplate_center_percent": iemg_sim.endplate_center__percent,
    "endplate_center_mm": iemg_sim.endplate_center__mm,
}

# Save the package
decomposition_file = save_path / "decomposition_package.pkl"
joblib.dump(decomposition_package, decomposition_file)

# Print summary
print("\n✓ Created decomposition package:")
print(f"  - EMG signal shape: {emg_data.shape} (time × electrodes)")
print(f"  - Number of MUs: {len(mu_indices)}")
print(f"  - MU indices: {mu_indices}")
print(f"  - MUAP templates: {len(muap_templates)} waveforms")
print(f"  - Spike trains: {len(spike_trains_list)} ground truth trains")
print(f"  - Sampling rate: {iemg_sim.sampling_frequency__Hz} Hz")
print(f"  - Duration: {emg_signal.t_stop.rescale('s').magnitude:.2f} seconds")
print("  - SNR: 20 dB")
print(f"\n✓ Saved to: {decomposition_file}")

print("\nPackage contents:")
for key in decomposition_package.keys():
    value = decomposition_package[key]
    if isinstance(value, np.ndarray):
        print(f"  '{key}': ndarray {value.shape} ({value.dtype})")
    elif isinstance(value, list):
        if len(value) > 0 and isinstance(value[0], np.ndarray):
            # Show all shapes for MUAP templates and spike trains
            shapes = [v.shape for v in value]
            print(f"  '{key}': list of {len(value)} ndarrays")
            for i, shape in enumerate(shapes):
                mu_id = mu_indices[i]
                print(f"      MU {mu_id}: shape {shape}")
        else:
            print(f"  '{key}': list of {len(value)} items")
            if key == "mu_indices":
                print(f"      {value}")
    else:
        print(f"  '{key}': {type(value).__name__} = {value}")

print("\n" + "=" * 70)
print("READY FOR DECOMPOSITION!")
print("=" * 70 + "\n")

plt.show()
