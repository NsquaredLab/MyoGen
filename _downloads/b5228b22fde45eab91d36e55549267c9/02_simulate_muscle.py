"""
Muscle Model
===============================

The **spike trains** alone are not enough to create **EMG signals**.

To create **EMG signals**, we need to create a **muscle model** that will distribute the **motor units** and **fibers** within the muscle volume.
"""

##############################################################################
# Import Libraries
# -----------------

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import joblib

from myogen import simulator
from myogen.utils.plotting.muscle import plot_mf_centers, plot_innervation_areas_2d

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

# Define muscle parameters for an FDI muscle
muscle_radius = 4.9  # Muscle radius in mm
mean_fiber_length = 32  # Mean fiber length in mm
fiber_length_variation = 3  # Fiber length variation (±) in mm
fiber_density = 50  # Fiber density per mm²

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
    recruitment_thresholds=recruitment_thresholds[::4],  # For faster simulation
    radius__mm=muscle_radius,
    fiber_density__fibers_per_mm2=fiber_density,
    max_innervation_area_to_total_muscle_area__ratio=max_innervation_ratio,
    grid_resolution=grid_resolution,
    autorun=True,
)

# Save muscle model for future use
joblib.dump(muscle, save_path / "muscle_model.pkl")

# Display muscle statistics
total_fibers = sum(muscle.resulting_number_of_innervated_fibers)

print(f"\nMuscle model statistics:")
print(f"  - Total muscle fibers: {total_fibers}")
print(f"  - Mean fibers per MU: {total_fibers / len(recruitment_thresholds):.1f}")
print(f"  - Muscle cross-sectional area: {np.pi * muscle_radius**2:.1f} mm²")

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

with plt.xkcd():
    plot_mf_centers(muscle, ax=plt.gca())
plt.title("Motor Unit Fiber Centers")
plt.xlabel("X Position (mm)")
plt.ylabel("Y Position (mm)")
plt.axis("equal")

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

selected_indices = np.arange(25)[::-1]
with plt.xkcd():
    plot_innervation_areas_2d(muscle, indices_to_plot=selected_indices, ax=plt.gca())
plt.title("Motor Unit Innervation Areas")
plt.xlabel("X Position (mm)")
plt.ylabel("Y Position (mm)")
plt.axis("equal")

plt.tight_layout()
plt.show()
