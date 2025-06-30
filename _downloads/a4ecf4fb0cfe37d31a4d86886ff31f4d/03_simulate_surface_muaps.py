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
electrode_grid_size = (5, 5)
inter_electrode_distance = 4.0  # mm - standard spacing

# Define volume conductor parameters
fat_thickness = 1.0  # mm
skin_thickness = 1.0  # mm

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
# The **SurfaceEMG** object is initialized with the **muscle model** and the **simulation parameters**.
#
# .. note::
#    For simplicity, we only simulate the first motor unit.
#    This can be changed by modifying the ``MUs_to_simulate`` parameter.
#
#   This is to simulate the **surface EMG** from two different directions.
#

surface_emg = simulator.SurfaceEMG(
    muscle_model=muscle,
    sampling_frequency__Hz=sampling_frequency,
    electrode_grid_dimensions__rows_cols=electrode_grid_size,
    inter_electrode_distance__mm=inter_electrode_distance,
    MUs_to_simulate=[0],
    electrode_grid_center_positions=[(0, 0)],
    fat_thickness__mm=fat_thickness,
    skin_thickness__mm=skin_thickness,
)

##############################################################################
# Simulate MUAPs
# --------------
#
# To generate the **MUAPs**, we need to run the ``simulate_muaps`` method of the **SurfaceEMG** object.


# Run simulation with progress output
muaps = surface_emg.simulate_muaps(show_plots=False, verbose=False)

print(f"\nMUAP simulation completed!")
print(f"Generated MUAPs shape: {muaps.shape}")
print(f"  - {muaps.shape[0]} electrode position(s)")
print(f"  - {muaps.shape[1]} motor units")
print(f"  - {muaps.shape[2]}×{muaps.shape[3]} electrode grid")
print(f"  - {muaps.shape[4]} time samples")

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
        sharey=True
    )
    fig.suptitle(f"MUAP {muap_idx}")
    axes_list.append(axes)

# Plot MUAPs using the new API
plot_muap_grid(muaps_concatenated[ :, :, :, 100:-100], axes_list, apply_default_formatting=True)

# Show all plots
plt.show()
