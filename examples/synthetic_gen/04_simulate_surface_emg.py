"""
Surface EMG Signals
=============================

After having created the **MUAPs**, we can finally simulate the **surface EMG** by creating a **surface EMG model**.

.. note::
    The **surface EMG** signals are the **summation** of the **MUAPs** at the surface of the skin.

    In **Myogen**, we can simulate the **surface EMG** by convolving the **MUAPs** with the **spike trains** of the **motor units**.
"""

##############################################################################
# Import Libraries
# -----------------
from pathlib import Path

import joblib
import numpy as np
import quantities as pq
import seaborn as sns
from matplotlib import pyplot as plt

from myogen import simulator
from myogen.utils.types import CURRENT__AnalogSignal, SPIKE_TRAIN__Block

##############################################################################
# Load Necessary Models
# ----------------------------
#

save_path = Path("./results/synthetic_gen")

spike_train__Block: SPIKE_TRAIN__Block = joblib.load(
    save_path / "sinusoidal_dd_spike_trains.pkl"
)
input_current__AnalogSignal: CURRENT__AnalogSignal = joblib.load(
    save_path / "trapezoid_drive_pattern.pkl"
)
surface_emg: simulator.SurfaceEMG = joblib.load(save_path / "surface_emg.pkl")

##############################################################################
# Generate Surface EMG
# -----------------------------------------------
#
# To simulate the **surface EMG**, we need to run the ``simulate_surface_emg`` method of the **SurfaceEMG** object.

surface_emg_signals = surface_emg.simulate_surface_emg(
    spike_train__Block=spike_train__Block
)

print("Surface EMG simulation completed!")
# Access the first group (electrode array) and first segment (pool)
first_emg_signal = surface_emg_signals.groups[0].segments[0].analogsignals[0]
print(f"Generated EMG shape: {first_emg_signal.shape}")
print(f"  - {first_emg_signal.shape[0]} time samples")
print(f"  - {first_emg_signal.shape[1]} electrode rows")
print(f"  - {first_emg_signal.shape[2]} electrode columns")

# Save the surface EMG results
joblib.dump(surface_emg_signals, save_path / "surface_emg_signals.pkl")

##############################################################################
# Visualize Surface EMG Results
# -----------------------------
#
# .. note::
#   Since **MyoGen** is a simulator, the results will have no real-world noise.
#
#   We can add noise to the **surface EMG** signals to make them more realistic.
#
#   For this the method ``add_noise`` is used.

print("\nAdding noise to surface EMG (SNR = 5 dB)...")
noisy_surface_emg__Block = surface_emg.add_noise(snr__dB=1.0)

# Save the noisy surface EMG
joblib.dump(noisy_surface_emg__Block, save_path / "noisy_surface_emg_signals.pkl")
print(f"✓ Saved noisy surface EMG to: {save_path / 'noisy_surface_emg_signals.pkl'}")

# Load input current as AnalogSignal

with plt.xkcd():
    plt.rcParams.update({"font.size": 24})
    # Create single plot with normalized signals
    fig, ax = plt.subplots(figsize=(12, 6))

    # Get EMG signal from noisy surface EMG (first pool, electrode at row 2, col 2)
    emg_signal = (
        noisy_surface_emg__Block.groups[0]
        .segments[0]
        .analogsignals[0][:, 2, 2]
        .magnitude
    )
    current_signal = input_current__AnalogSignal[:, 0].magnitude

    # Normalize EMG by dividing by maximum
    emg_normalized = emg_signal / np.max(np.abs(emg_signal))

    # Normalize current between 0 and 1
    current_normalized = (current_signal - np.min(current_signal)) / (
        np.max(current_signal) - np.min(current_signal)
    )

# Plot both normalized signals on same axis
ax.plot(
    np.arange(len(emg_normalized)) / surface_emg.sampling_frequency__Hz,
    emg_normalized,
    linewidth=2,
    label="Surface EMG",
)

with plt.xkcd():
    ax.plot(
        input_current__AnalogSignal.times.rescale(pq.s).magnitude,
        current_normalized,
        linewidth=2,
        label="Input Current",
        alpha=0.7,
        zorder=-1,
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized Amplitude")
    ax.legend()

    sns.despine(trim=True, left=False, bottom=False, right=True, top=True, offset=5)

    plt.title("Normalized Surface EMG and Input Current")

plt.tight_layout()

##############################################################################
# Package Data for Decomposition
# -------------------------------
#
# Create a comprehensive data package containing everything needed for
# surface EMG decomposition algorithm development and validation.
#
# This package includes:
# - **EMG signal**: The noisy mixed signal to decompose (2D grid)
# - **MUAP templates**: Ground truth motor unit waveforms (2D spatial patterns)
# - **Spike trains**: Ground truth firing times for validation
# - **Metadata**: MU indices, sampling rate, electrode configuration, etc.

print("\n" + "=" * 70)
print("PACKAGING DATA FOR SURFACE EMG DECOMPOSITION")
print("=" * 70)

# Get motor unit indices
if hasattr(surface_emg, 'MUs_to_simulate') and surface_emg.MUs_to_simulate is not None:
    mu_indices = surface_emg.MUs_to_simulate
else:
    # All MUs were simulated
    n_mus = len(surface_emg_signals.groups[0].segments)
    mu_indices = list(range(n_mus))

# Extract surface EMG signal data (first electrode array, first pool)
emg_signal = noisy_surface_emg__Block.groups[0].segments[0].analogsignals[0]
emg_data = emg_signal.magnitude  # Shape: (time, rows, cols)

# Extract MUAP templates (one per motor unit) from surface EMG simulator
# Surface MUAPs are stored in groups[array_idx].segments[mu_idx]
muap_templates = []
muap_durations_ms = []

# Access the MUAPs from the first electrode array
muap_block = surface_emg._muaps__Block
for segment in muap_block.groups[0].segments:
    muap = segment.analogsignals[0].magnitude  # Shape: (time, rows, cols)
    muap_templates.append(muap)
    duration_ms = muap.shape[0] / surface_emg.sampling_frequency__Hz * 1000
    muap_durations_ms.append(duration_ms)

# Extract spike trains (ground truth) - only for selected MUs
spike_trains_list = []
all_spike_trains = spike_train__Block.segments[0].spiketrains
for mu_idx in mu_indices:
    spike_times_ms = all_spike_trains[mu_idx].magnitude  # Already in ms
    spike_trains_list.append(spike_times_ms)

# Get electrode array configuration
electrode_array = surface_emg.electrode_arrays[0]

# Get muscle model
muscle = surface_emg.muscle_model

# Create decomposition package
decomposition_package = {
    # ========== SIGNALS ==========
    "emg_signal": emg_data,  # Shape: (time, rows, cols)
    "muap_templates": muap_templates,  # List of (time, rows, cols)
    "spike_trains": spike_trains_list,  # List of spike times in ms

    # ========== MOTOR UNIT INFO ==========
    "mu_indices": mu_indices,  # Global MU indices
    "n_motor_units": len(mu_indices),
    "recruitment_thresholds": muscle.recruitment_thresholds[mu_indices],
    "muap_durations_ms": np.array(muap_durations_ms),

    # ========== TIMING ==========
    "sampling_rate_hz": float(surface_emg.sampling_frequency__Hz),
    "time_duration_s": float(emg_signal.t_stop.rescale("s").magnitude),
    "time_start_s": float(emg_signal.t_start.rescale("s").magnitude),
    "n_samples": emg_data.shape[0],

    # ========== ELECTRODE CONFIG ==========
    "electrode_grid_shape": (electrode_array.num_rows, electrode_array.num_cols),
    "num_rows": electrode_array.num_rows,
    "num_cols": electrode_array.num_cols,
    "inter_electrode_distance_mm": electrode_array.inter_electrode_distances__mm,
    "electrode_radius_mm": electrode_array.electrode_radius__mm,
    "differentiation_mode": electrode_array.differentiation_mode,

    # ========== NOISE ==========
    "snr_db": 1.0,

    # ========== MUSCLE PROPERTIES ==========
    "muscle_radius_mm": muscle.radius__mm,
    "muscle_length_mm": muscle.length__mm,
    "fat_thickness_mm": muscle.fat_thickness__mm,
    "skin_thickness_mm": muscle.skin_thickness__mm,
}

# Save the package
decomposition_file = save_path / "surface_emg_decomposition_package.pkl"
joblib.dump(decomposition_package, decomposition_file)

# Print summary
print("\n✓ Created surface EMG decomposition package:")
print(f"  - EMG signal shape: {emg_data.shape} (time × rows × cols)")
print(f"  - Number of MUs: {len(mu_indices)}")
print(f"  - MU indices: {mu_indices}")
print(f"  - MUAP templates: {len(muap_templates)} waveforms")
print(f"  - Spike trains: {len(spike_trains_list)} ground truth trains")
print(f"  - Sampling rate: {surface_emg.sampling_frequency__Hz} Hz")
print(f"  - Duration: {emg_signal.t_stop.rescale('s').magnitude:.2f} seconds")
print(f"  - Electrode grid: {electrode_array.num_rows}×{electrode_array.num_cols}")
print("  - SNR: 1.0 dB")
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
    elif isinstance(value, tuple):
        print(f"  '{key}': tuple = {value}")
    else:
        print(f"  '{key}': {type(value).__name__} = {value}")

print("\n" + "=" * 70)
print("READY FOR SURFACE EMG DECOMPOSITION!")
print("=" * 70 + "\n")

plt.show()

