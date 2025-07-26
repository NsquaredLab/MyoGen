"""
Surface Motor Unit Action Potentials
====================================

After having created the **muscle model**, we can simulate the **surface EMG** by creating a **surface EMG model**.

First step is to create **MUAPs** from the **muscle model**.

.. note::
    The **MUAPs** are the **action potentials** of the **motor units** at the surface of the skin.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np

from myogen import simulator
from myogen.utils.plotting import plot_muap_grid

##############################################################################
# Define Parameters
# -----------------
#
# The **surface EMG** is created using the **SurfaceEMG** object.
#
# The **SurfaceEMG** object takes the following parameters:
#
# - ``muscle_model``: Muscle model
# - ``sampling_frequency``: Sampling frequency
# - ``electrode_grid_dimensions``: Electrode grid dimensions
# - ``inter_electrode_distance``: Inter-electrode distance
# - ``fat_thickness``: Fat thickness
# - ``skin_thickness``: Skin thickness

# Define simulation parameters
sampling_frequency = 2048.0  # Hz - standard for surface EMG

##############################################################################
# Load Muscle Model
# ----------------------------
#
# Load muscle model from previous example

save_path = Path("./results")
muscle = joblib.load(save_path / "muscle_model.pkl")

##############################################################################
# Create Surface EMG Model
# -------------------------
#
# The **SurfaceEMG** object is initialized with the **muscle model**, the **electrode array**, and the **simulation parameters**.
#
# .. note::
#    For simplicity, we only simulate the first motor unit.
#    This can be changed by modifying the ``MUs_to_simulate`` parameter.
#
#   This is to simulate the **surface EMG** from two different directions.
#

electrode_array_monopolar = simulator.SurfaceElectrodeArray(
    num_rows=13,
    num_cols=5,
    inter_electrode_distances__mm=2,
    electrode_radius__mm=1,
    differentiation_mode="monopolar",
    bending_radius__mm=muscle.radius__mm
    + muscle.skin_thickness__mm
    + muscle.fat_thickness__mm,
)

surface_emg = simulator.SurfaceEMG(
    muscle_model=muscle,
    electrode_arrays=[electrode_array_monopolar],
    sampling_frequency__Hz=sampling_frequency,
    MUs_to_simulate=[0],
)

##############################################################################
# Simulate MUAPs
# --------------
#
# To generate the **MUAPs**, we need to run the ``simulate_muaps`` method of the **SurfaceEMG** object.


# Run simulation with progress output
muaps = surface_emg.simulate_muaps()

print(f"\nMUAP simulation completed!")
print(f"Generated MUAPs shape: {muaps[0].shape}")
print(f"  - {muaps[0].shape[0]} motor units")
print(f"  - {muaps[0].shape[1]}×{muaps[0].shape[2]} electrode grid")
print(f"  - {muaps[0].shape[3]} time samples")

# Save results
joblib.dump(surface_emg, save_path / "surface_emg.pkl")

##############################################################################
# Plot MUAPs
# ------------------------------------
#
# The MUAPs can be plotted using the ``plot_muap_grid`` function.
#
# .. note::
#   **Plotting helper functions** are available in the ``myogen.utils.plotting`` module.
#   The new API requires creating matplotlib axes and passing them to the plotting function.

# Concatenate MUAPs from all electrode positions and motor units
muaps_concatenated = np.concatenate(muaps)
print(f"Concatenated MUAPs shape: {muaps_concatenated.shape}")

# Create subplot grid for each MUAP (matches electrode grid layout)
n_muaps = muaps_concatenated.shape[0]
electrode_rows = muaps_concatenated.shape[1]
electrode_cols = muaps_concatenated.shape[2]

# Create axes for each MUAP - one subplot grid per MUAP
axes_list = []
for muap_idx in range(n_muaps):
    fig, axes = plt.subplots(
        electrode_rows,
        electrode_cols,
        figsize=(electrode_cols * 2, electrode_rows * 2),
        sharex=True,
        sharey=True,
    )
    fig.suptitle(f"MUAP {muap_idx}")
    axes_list.append(axes)

# Plot MUAPs using the new API
plot_muap_grid(
    muaps_concatenated[:, :, :, 100:-100], axes_list, apply_default_formatting=True
)
plt.tight_layout()
# Show all plots
plt.show()
