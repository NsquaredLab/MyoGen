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

# %%

##############################################################################
# Import Libraries
# ----------------

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import quantities as pq
import seaborn as sns

from myogen import simulator
from myogen.utils.types import CURRENT__AnalogSignal, SPIKE_TRAIN__Block

plt.style.use("fivethirtyeight")

##############################################################################
# Define Parameters
# -----------------
#
# All parameters now use quantities for type safety and unit validation.

# Electrode parameters
inter_electrode_distance = 2.0 * pq.mm  # Electrode spacing (typical: 1-5 mm)
electrode_position = (
    0.0 * pq.mm,  # Center of muscle (x)
    0.0 * pq.mm,  # Center of muscle (y)
    15.0 * pq.mm,  # Middle of muscle (z) - near endplate zone
)


##############################################################################
# Load Muscle Model
# -------------------
#
# Load the **muscle model** with the generated recruitment thresholds.

muscle: simulator.Muscle = joblib.load("results/muscle_model.pkl")

#######################################
# Create Intramuscular Electrode Array
# ------------------------------------
#
# Set up a **differential needle electrode** for intramuscular recordings.
# All distance and angle parameters use quantities for type-safe unit validation.

electrode = simulator.IntramuscularElectrodeArray(
    num_electrodes=4,
    inter_electrode_distance__mm=inter_electrode_distance,
    differentiation_mode="consecutive",
    position__mm=electrode_position,
)

##############################################################################
# Initialize Intramuscular EMG Simulator
# --------------------------------------
#
# Create the **intramuscular EMG simulator** with the muscle model and electrode.
# All simulation parameters now use quantities for type safety and unit validation.

MUs_to_simulate = [0, 25, 50, 90]

print("Initializing iEMG simulator...")
iemg_sim = simulator.IntramuscularEMG(
    muscle_model=muscle, electrode_array=electrode, MUs_to_simulate=MUs_to_simulate
)

##############################################################################
# Calculate Motor Unit Action Potentials
# --------------------------------------
#
# Compute the **MUAPs** for the selected motor units at the electrode positions.
# Using parallel processing for faster computation.

print("Computing motor unit action potentials...")
muaps__Block = iemg_sim.simulate_muaps(n_jobs=2)  # Use 2 parallel jobs

# Display MUAP amplitudes for the selected MUs
print("\n=== MUAP Amplitudes ===")
for mu_idx in MUs_to_simulate:
    segment = muaps__Block.segments[mu_idx]
    if segment.name != "MUAP_None":
        signal = segment.analogsignals[0].magnitude
        ptp_values = np.ptp(signal, axis=0)
        max_ptp = np.max(ptp_values)
        print(f"MU {mu_idx:2d}): Max PtP = {max_ptp * 1000:6.1f} µV")

joblib.dump(iemg_sim, "./results/iemg_simulator.pkl")

##############################################################################
# Visualize Individual MUAP Shapes
# ---------------------------------
#
# Display the **MUAP waveforms** for the three selected motor units to
# demonstrate how motor unit size affects MUAP amplitude and shape.

fig, axes = plt.subplots(len(MUs_to_simulate), 1, figsize=(12, 8), sharex=True)

for plot_idx, mu_idx in enumerate(MUs_to_simulate):
    segment = muaps__Block.segments[mu_idx]
    if segment.name != "MUAP_None":
        muap_signal = segment.analogsignals[0][:, 0]  # First electrode channel
        time_axis = muap_signal.times.rescale("ms").magnitude
        amplitude_uV = muap_signal.magnitude * 1000  # Convert mV to µV

        axes[plot_idx].plot(time_axis, amplitude_uV, linewidth=2)
        axes[plot_idx].set_ylabel("Amplitude (µV)")
        axes[plot_idx].grid(True, alpha=0.3)
        axes[plot_idx].set_title(f"MU {mu_idx}: PtP: {np.ptp(amplitude_uV):.1f} µV")

axes[-1].set_xlabel("Time (ms)")
sns.despine(trim=True, left=False, bottom=False, right=True, top=True, offset=5)
plt.tight_layout()
plt.show()

##############################################################################
# Load Input Currents and Spike Trains
# -----------------------
#

save_path = Path("./results")

spike_train__Block: SPIKE_TRAIN__Block = joblib.load(save_path / "trapezoid_dd_spike_trains.pkl")
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
print(f"\t{first_emg_signal.shape[0]} time samples")
print(f"\t{first_emg_signal.shape[1]} electrode channels")
print(f"\tSignal RMS (before noise): {np.sqrt(np.mean(first_emg_signal.magnitude**2)):.3f}")

print("Adding realistic noise (SNR = 20 dB)...")
noisy_emg_signals__Block = iemg_sim.add_noise(snr__dB=20)
first_noisy_signal = noisy_emg_signals__Block.segments[0].analogsignals[0]
print(f"\tSignal RMS (after noise): {np.sqrt(np.mean(first_noisy_signal.magnitude**2)):.3f}")

##############################################################################
# Visualize Intramuscular EMG Results
# ----------------------------------
#
# Compare the **intramuscular EMG** signal with the input current.
#
# .. note::
#   In this example, only **three motor units** (small, medium, large) contribute
#   to the EMG signal. This demonstrates how individual MUAPs sum to create the
#   composite EMG signal, and how different motor unit sizes affect signal amplitude.

plt.figure(figsize=(12, 6))

# Get the signals - use first electrode from the Block
iemg_signal = noisy_emg_signals__Block.segments[0].analogsignals[0][:, 0]
current_signal = input_current__AnalogSignal[:, 0]  # First current pool

# Create time axes
emg_time = iemg_signal.times.rescale("s").magnitude
current_time = input_current__AnalogSignal.times.rescale("s").magnitude

# Normalize current between 0 and 1
current_normalized = (
    (current_signal - np.min(current_signal))
    / (np.max(current_signal) - np.min(current_signal))
    * np.max(iemg_signal.magnitude)
)

plt.plot(emg_time, iemg_signal.magnitude, label="Intramuscular EMG", linewidth=1)
plt.plot(current_time, current_normalized, label="Input Current", linewidth=1)

plt.xlabel("Time (s)")
plt.ylabel("Amplitude (mV)")
plt.legend()

plt.tight_layout()
plt.show()
