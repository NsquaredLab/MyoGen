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

from myogen import load_profile
from myogen.utils.plotting.muscle import plot_innervation_areas_2d, plot_mf_centers

plt.style.use("fivethirtyeight")

##############################################################################
# Define Parameters
# -----------------
#
# The **muscle model** is a [`Muscle`][myogen.simulator.Muscle] object. Its main parameters are:
#
# - ``recruitment_thresholds``: Recruitment thresholds of the motor units
# - ``radius__mm``: Radius of the muscle
# - ``fiber_density__fibers_per_mm2``: Fiber density
# - ``skin_thickness__mm``, ``fat_thickness__mm``: Layers between the muscle and the electrodes
#
# The ``fdi_semg`` **profile card** holds the FDI values with their sources (radius 6.91 mm, 267 fibres/mm²,
# skin 1.2 mm, fat 1.3 mm, per-unit conduction velocities from recruitment force, ...). ``card.muscle()`` builds
# the muscle with them and the card's recruitment thresholds (regenerated from its seed: those of example 01). Keywords override a card value, e.g.
# ``card.muscle(fat_thickness__mm=3 * pq.mm)``; the card then prints a note on what else that affects.

save_path = Path("./results")
card = load_profile()  # fdi_semg

##############################################################################
# Create Muscle Model
# -------------------
#
# !!! note
#     Depending on the parameters, the simulation can take a few minutes to run.
#
#     To **avoid running the simulation every time**, we can save the muscle model using `joblib`.

# Create muscle model (innervation areas, fibres, fibre-to-unit assignment, fibre properties)
muscle = card.muscle()

# Save muscle model for future use
joblib.dump(muscle, save_path / "muscle_model.pkl")

# Display muscle statistics
total_fibers = sum(muscle.resulting_number_of_innervated_fibers)

print("Muscle model statistics:")
print(f"\tTotal muscle fibers: {total_fibers}")
print(f"\tMean fibers per MU: {total_fibers / len(muscle.resulting_number_of_innervated_fibers):.1f}")
print(f"\tMuscle cross-sectional area: {np.pi * muscle.radius__mm**2:.1f}")

##############################################################################
# Visualize Muscle Fiber Centers
# ------------------------------
#
# The **fiber centers** are the centers of the fibers that are innervated by the motor units.
#
# !!! note
#     The **fiber centers** have been precalculated from a Voronoi tessellation of the muscle volume.
#     Then depending on the **fiber density**, the **fiber centers** are distributed within the muscle volume.

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
# !!! note
#     The **innervation areas** are the areas where the motor units are innervated.
#     They are calculated using the **recruitment thresholds** and the **fiber density**.

plt.figure(figsize=(6, 6))

plot_innervation_areas_2d(muscle, ax=plt.gca())
plt.xlabel("X Position (mm)")
plt.ylabel("Y Position (mm)")
plt.axis("equal")
plt.grid(False)

plt.tight_layout()
plt.show()

# mkdocs_gallery_thumbnail_path = "gallery_thumbs/04_simulate_muscle.png"
