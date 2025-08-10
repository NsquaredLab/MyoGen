"""
Surface EMG Signals
=============================

After having created the **MUAPs**, we can finally simulate the **surface EMG** by creating a **surface EMG model**.

.. note::
    The **surface EMG** signals are the **summation** of the **MUAPs** at the surface of the skin.

    In **Myogen**, we can simulate the **surface EMG** by convolving the **MUAPs** with the **spike trains** of the **motor units**.
"""

import shutil
##############################################################################
# Import Libraries
# -----------------

from pathlib import Path

import joblib
import matplotlib
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

##############################################################################
# Load Muscle Model
# ----------------------------
#

save_path = Path("./results")
muscle = joblib.load(save_path / "muscle_model.pkl")

##############################################################################
# Load Motor Neuron Pool
# ----------------------------
#

motor_neuron_pool = joblib.load(save_path / "motor_neuron_pool.pkl")

##############################################################################
# Load Surface EMG Model
# ----------------------------
#
# .. note::
#   Because of computation time, we pre-computed the **surface EMG** model for 100 motor units.

surface_emg = joblib.load(save_path / "surface_emg.pkl")


##############################################################################
# Generate Surface EMG
# -----------------------------------------------
#
# To simulate the **surface EMG**, we need to run the ``simulate_surface_emg`` method of the **SurfaceEMG** object.

surface_emg_signals = surface_emg.simulate_surface_emg(
    motor_neuron_pool=motor_neuron_pool
)

print(f"Surface EMG simulation completed!")
print(f"Generated EMG shape: {surface_emg_signals[0].shape}")
print(f"  - {surface_emg_signals[0].shape[0]} pools")
print(f"  - {surface_emg_signals[0].shape[1]} electrode rows")
print(f"  - {surface_emg_signals[0].shape[2]} electrode columns")
print(f"  - {surface_emg_signals[0].shape[3]} time samples")

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

surface_emg.add_noise(snr_db=10.0)

# Load input current matrix
input_current_matrix = joblib.load(save_path / "input_current_matrix.pkl")
shutil.rmtree(matplotlib.get_cachedir())
with plt.xkcd():
    plt.rcParams.update({"font.size": 24})
    # Create single plot with normalized signals
    fig, ax = plt.subplots(figsize=(12, 6))
    # Get the signals
    emg_signal = surface_emg.noisy_surface_emg__tensors[0][0, 3, 3]
    current_signal = input_current_matrix[0]  # Only first current

    # Normalize EMG by dividing by maximum
    emg_normalized = emg_signal / np.max(emg_signal)

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
        motor_neuron_pool.times / 1000.0,
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
