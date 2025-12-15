"""
Muscle Model
===============================

The **spike trains** alone are not enough to create **EMG signals**.

To create **EMG signals**, we need to create a **muscle model** that will distribute the **motor units** and **fibers** within the muscle volume.
"""

# %%

##############################################################################
# Import Libraries
# -----------------

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import quantities as pq

from myogen import simulator
from myogen.utils.plotting.muscle import plot_innervation_areas_2d, plot_mf_centers

plt.style.use("fivethirtyeight")

##############################################################################
# Define Parameters
# -----------------
#
# The **muscle model** is created using the **Muscle** object.
#
# The **Muscle** object takes the following parameters:
#
# - ``recruitment_thresholds``: Recruitment thresholds of the motor units
# - ``radius``: Radius of the muscle in mm
# - ``fiber_density``: Fiber density per mm²
# - ``max_innervation_area_to_total_muscle_area__ratio``: Maximum innervation area to total muscle area ratio
# - ``grid_resolution``: Spatial resolution for muscle discretization
#
# Since the **recruitment thresholds** are already generated, we can load them from the previous example using ``joblib``.

# Load recruitment thresholds
save_path = Path("./results")

recruitment_thresholds = joblib.load(save_path / "thresholds.pkl")

# Define muscle parameters
muscle_radius = 4.9 * pq.mm  # Muscle radius in mm
mean_fiber_length = 32 * pq.mm  # Mean fiber length in mm
fiber_length_variation = 3 * pq.mm  # Fiber length variation in mm
fiber_density = 400 * pq.mm**-2  # Fiber density in fibers per mm²

# Define simulation parameters
max_innervation_ratio = 1 / 4  # Maximum motor unit territory size
grid_resolution = 256  # Spatial resolution for muscle discretization

##############################################################################
# Create Muscle Model
# -------------------
#
# .. note::
#    Depending on the parameters, the simulation can take a few minutes to run.
#
#    To **avoid running the simulation every time**, we can save the muscle model using ``joblib``.

# Create muscle model
muscle = simulator.Muscle(
    recruitment_thresholds=recruitment_thresholds,
    radius_bone__mm=1.0 * pq.mm,  # Non-zero bone radius required for proper MUAP amplitudes
    fiber_density__fibers_per_mm2=fiber_density,
    fat_thickness__mm=10 * pq.mm,
    autorun=True,
)

# Save muscle model for future use
joblib.dump(muscle, save_path / "muscle_model.pkl")

# Display muscle statistics
total_fibers = sum(muscle.resulting_number_of_innervated_fibers)

print("Muscle model statistics:")
print(f"\tTotal muscle fibers: {total_fibers}")
print(f"\tMean fibers per MU: {total_fibers / len(recruitment_thresholds):.1f}")
print(f"\tMuscle cross-sectional area: {np.pi * muscle_radius**2:.1f}")

##############################################################################
# Visualize Muscle Fiber Centers
# ------------------------------
#
# The **fiber centers** are the centers of the fibers that are innervated by the motor units.
#
# .. note::
#    The **fiber centers** have been precalculated from a Voronoi tessellation of the muscle volume.
#    Then depending on the **fiber density**, the **fiber centers** are distributed within the muscle volume.

plt.figure(figsize=(6, 6))

plot_mf_centers(muscle, ax=plt.gca())
plt.xlabel("X Position (mm)")
plt.ylabel("Y Position (mm)")
plt.axis("equal")
plt.grid(False)

plt.tight_layout()
plt.show()

##############################################################################
# Visualize Motor Unit Innervation Areas
# -------------------------------------
#
# Display the spatial organization of motor units and their innervation areas.
#
# .. note::
#    The **innervation areas** are the areas where the motor units are innervated.
#    They are calculated using the **recruitment thresholds** and the **fiber density**.

plt.figure(figsize=(6, 6))

plot_innervation_areas_2d(muscle, ax=plt.gca())
plt.xlabel("X Position (mm)")
plt.ylabel("Y Position (mm)")
plt.axis("equal")
plt.grid(False)

plt.tight_layout()
plt.show()
