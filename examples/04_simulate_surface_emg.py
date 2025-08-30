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

save_path = Path("./results")

spike_train__Block: SPIKE_TRAIN__Block = joblib.load(
    save_path / "spike_train__Block.pkl"
)
input_current__AnalogSignal: CURRENT__AnalogSignal = joblib.load(
    save_path / "input_current__AnalogSignal.pkl"
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

noisy_surface_emg__Block = surface_emg.add_noise(snr__dB=5.0)

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
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized Amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend()

    sns.despine(trim=True, left=False, bottom=False, right=True, top=True, offset=5)

    plt.title("Normalized Surface EMG and Input Current")

plt.tight_layout()
plt.show()
