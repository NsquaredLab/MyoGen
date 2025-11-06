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

# Motor Unit Selection
# ~~~~~~~~~~~~~~~~~~~~
# For computational efficiency or focused analysis, you can simulate
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
MUs_to_simulate = [0, 5, 10, 15, 20, 25, 30, 50]  # Simulate all 100 MUs
# MUs_to_simulate = [0, 5, 10, 15, 20, 25, 30]  # 7 specific MUs
# MUs_to_simulate = list(range(20))  # First 20 MUs only
# MUs_to_simulate = list(range(0, 100, 10))  # Every 10th MU (10 total)

##############################################################################
# Load Muscle Model
# ----------------------------
#
# Load muscle model from previous example

save_path = Path("./results/synthetic_gen")
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

print("Initializing Surface EMG simulator...")
if MUs_to_simulate is not None:
    print(f"  → Simulating {len(MUs_to_simulate)} selected MUs: {MUs_to_simulate}")
else:
    n_mus = len(muscle.resulting_number_of_innervated_fibers)
    print(f"  → Simulating all {n_mus} MUs")

surface_emg = simulator.SurfaceEMG(
    muscle_model=muscle,
    electrode_arrays=[electrode_array_monopolar],
    sampling_frequency__Hz=sampling_frequency,
    MUs_to_simulate=MUs_to_simulate,
)

##############################################################################
# Simulate MUAPs
# --------------
#
# To generate the **MUAPs**, we need to run the ``simulate_muaps`` method of the **SurfaceEMG** object.

print("\nSimulating surface MUAPs...")
# Run simulation with progress output
muaps = surface_emg.simulate_muaps()


print("\nMUAP simulation completed!")
n_simulated_mus = len(muaps.groups[0].segments)
muap_shape = muaps.groups[0].segments[0].analogsignals[0].shape
print(f"Generated MUAPs for {n_simulated_mus} motor units")
print(f"  - Shape per MU: {muap_shape}")
print(f"  - {muap_shape[0]} time samples")
print(f"  - {muap_shape[1]} rows × {muap_shape[2]} columns electrode grid")
if MUs_to_simulate is not None:
    print(f"  - MU indices: {MUs_to_simulate}")

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
