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
muscle: simulator.Muscle = joblib.load(save_path / "muscle_model.pkl")

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
    num_rows=5,
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
)

##############################################################################
# Simulate MUAPs
# --------------
#
# To generate the **MUAPs**, we need to run the ``simulate_muaps`` method of the **SurfaceEMG** object.


# Run simulation with progress output
muaps = surface_emg.simulate_muaps()


print("MUAP simulation completed!")
print(f"Generated MUAPs shape: {muaps.groups[0].segments[0].analogsignals[0].shape}")
print(f"  - {len(muaps.groups[0].segments)} motor units")
print(
    "  - {} rows × {} columns electrode grid".format(
        muaps.groups[0].segments[0].analogsignals[0].shape[1],
        muaps.groups[0].segments[0].analogsignals[0].shape[2],
    )
)
print(f"  - {muaps.groups[0].segments[0].analogsignals[0].shape[0]} time samples")

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

# Create axes for the first MUAP
fig, ax = plt.subplots(
    electrode_array_monopolar.num_rows,
    electrode_array_monopolar.num_cols,
    figsize=(
        electrode_array_monopolar.num_cols * 2,
        electrode_array_monopolar.num_rows * 2,
    ),
    sharex=True,
    sharey=True,
)
fig.suptitle("MUAP 0")

plot_muap_grid(
    surface_muap__Block=muaps,
    axs=[ax],
    muap_indices=[0],
    time_slice=slice(100, -100),
    apply_default_formatting=True,
)
plt.tight_layout()
plt.show()
