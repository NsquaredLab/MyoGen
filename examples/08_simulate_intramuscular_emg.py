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

# Simulation parameters

# Electrode parameters
inter_electrode_distance = 0.5  # mm
electrode_position = (0.0, 0.0, 0.0)  # mm (start of muscle)


##############################################################################
# Load Muscle Model
# -------------------
#
# Load the **muscle model** with the generated recruitment thresholds.

muscle: simulator.Muscle = joblib.load("results/muscle_model.pkl")

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

print("Initializing iEMG simulator...")
iemg_sim = simulator.IntramuscularEMG(
    muscle_model=muscle,
    electrode_array=electrode,
)

##############################################################################
# Calculate Motor Unit Action Potentials
# --------------------------------------
#
# Compute the **MUAPs** for each motor unit at the electrode positions.


print("Computing motor unit action potentials...")
iemg_sim.simulate_muaps()


# iemg_sim = joblib.load("./results/iemg_simulator.pkl")

joblib.dump(iemg_sim, "./results/iemg_simulator.pkl")


##############################################################################
# Load Input Currents and Spike Trains
# -----------------------
#

save_path = Path("./results")

spike_train__Block: SPIKE_TRAIN__Block = joblib.load(
    save_path / "sinusoidal_dd_spike_trains.pkl"
)
input_current__AnalogSignal: CURRENT__AnalogSignal = joblib.load(
    save_path / "sinusoidal_drive_pattern.pkl"
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
plt.show()
