"""
Current Generation
==================

**MyoGen** provides various functions to generate different types of input currents for motor neuron pool simulations.
These currents can be used to drive motor unit recruitment and firing patterns.

.. note::
    **Input currents** are the electrical stimuli applied to motor neuron pools to simulate muscle activation.
    Different current shapes can produce different recruitment and firing patterns.

**MyoGen** offers **5 different current waveform types**:

* **Sinusoidal**: Smooth oscillatory currents for rhythmic activation
* **Sawtooth**: Asymmetric ramp currents with adjustable rise/fall characteristics
* **Step**: Constant amplitude pulses for simple on/off activation
* **Ramp**: Linear increase/decrease for gradual force changes
* **Trapezoid**: Complex waveforms with rise, plateau, and fall phases
"""

##############################################################################
# Import Libraries
# ----------------

from pathlib import Path

import joblib
import numpy as np
from matplotlib import pyplot as plt

from myogen.utils.currents import (
    create_sinusoidal_current,
    create_sawtooth_current,
    create_step_current,
    create_ramp_current,
    create_trapezoid_current,
)
from myogen.utils.plotting.currents import plot_input_current__matrix

##############################################################################
# Define Parameters
# -----------------
# Each current simulation is defined by common timing parameters:
#
# - ``n_pools``: Number of motor neuron pools
# - ``simulation_duration__ms``: Total simulation time in milliseconds
# - ``timestep__ms``: Time step for simulation in milliseconds
#
# Additional parameters are specific to each current type and control amplitude,
# frequency, timing, and shape characteristics.

# Common simulation parameters
n_pools = 3
simulation_duration__ms = 2000.0  # 2 seconds
timestep__ms = 0.1  # 0.1 ms time step
t_points = int(simulation_duration__ms / timestep__ms)

# Create results directory
save_path = Path("./results")
save_path.mkdir(exist_ok=True)

##############################################################################
# Generate Sinusoidal Currents
# -----------------------------
#
# Sinusoidal currents are useful for simulating rhythmic muscle activations,
# such as those occurring during locomotion or tremor.

print("Generating sinusoidal currents...")

# Parameters for sinusoidal currents
sin_amplitudes = [50.0, 75.0, 100.0]  # Different amplitudes for each pool
sin_frequencies = [1.0, 2.0, 3.0]  # Different frequencies (Hz)
sin_offsets = [120.0, 130.0, 140.0]  # DC offsets (µV)
sin_phases = [0.0, np.pi / 3, 2 * np.pi / 3]  # Phase shifts (rad)

sinusoidal_currents = create_sinusoidal_current(
    n_pools=n_pools,
    t_points=t_points,
    timestep__ms=timestep__ms,
    amplitudes__muV=sin_amplitudes,
    frequencies__Hz=sin_frequencies,
    offsets__muV=sin_offsets,
    phases__rad=sin_phases,
)

##############################################################################
# Generate Sawtooth Currents
# ---------------------------
#
# Sawtooth currents provide asymmetric ramps useful for simulating
# force-variable contractions with different rise and fall characteristics.

print("Generating sawtooth currents...")

# Parameters for sawtooth currents
saw_amplitudes = [80.0, 60.0, 100.0]
saw_frequencies = [0.5, 1.0, 1.5]  # Slow sawtooth waves
saw_offsets = [120.0, 125.0, 130.0]
saw_widths = [0.3, 0.5, 0.7]  # Width of rising edge (0-1)

sawtooth_currents = create_sawtooth_current(
    n_pools=n_pools,
    t_points=t_points,
    timestep_ms=timestep__ms,
    amplitudes__muV=saw_amplitudes,
    frequencies__Hz=saw_frequencies,
    offsets__muV=saw_offsets,
    widths=saw_widths,
)

##############################################################################
# Generate Step Currents
# -----------------------
#
# Step currents are useful for simulating sudden muscle activations,
# such as those occurring during ballistic movements.

print("Generating step currents...")

# Parameters for step currents
step_heights = [100.0, 75.0, 125.0]  # Step amplitudes
step_durations = [500.0, 800.0, 300.0]  # Step durations (ms)
step_offsets = [120.0, 115.0, 125.0]  # Baseline currents

step_currents = create_step_current(
    n_pools=n_pools,
    t_points=t_points,
    timestep_ms=timestep__ms,
    step_heights__muV=step_heights,
    step_durations__ms=step_durations,
    offsets__muV=step_offsets,
)

##############################################################################
# Generate Ramp Currents
# -----------------------
#
# Ramp currents simulate gradual force changes, useful for modeling
# slow force development or fatigue scenarios.

print("Generating ramp currents...")

# Parameters for ramp currents
ramp_start = [0.0, 25.0, 50.0]  # Starting current levels
ramp_end = [150.0, 100.0, 200.0]  # Ending current levels
ramp_offsets = [120.0, 130.0, 110.0]  # DC offsets

ramp_currents = create_ramp_current(
    n_pools=n_pools,
    t_points=t_points,
    start_currents__muV=ramp_start,
    end_currents__muV=ramp_end,
    offsets__muV=ramp_offsets,
)

##############################################################################
# Generate Trapezoid Currents
# ----------------------------
#
# Trapezoid currents provide complex activation patterns with distinct
# phases: rise, plateau, and fall. These are useful for simulating
# controlled muscle contractions.

print("Generating trapezoid currents...")

# Parameters for trapezoid currents
trap_amplitudes = [100.0, 80.0, 120.0]  # Peak amplitudes
trap_rise_times = [200.0, 150.0, 300.0]  # Rise durations (ms)
trap_plateau_times = [800.0, 1000.0, 600.0]  # Plateau durations (ms)
trap_fall_times = [300.0, 200.0, 400.0]  # Fall durations (ms)
trap_offsets = [120.0, 125.0, 115.0]  # Baseline currents
trap_delays = [100.0, 200.0, 50.0]  # Initial delays (ms)

trapezoid_currents = create_trapezoid_current(
    n_pools=n_pools,
    t_points=t_points,
    timestep_ms=timestep__ms,
    amplitudes__muV=trap_amplitudes,
    rise_times__ms=trap_rise_times,
    plateau_times__ms=trap_plateau_times,
    fall_times__ms=trap_fall_times,
    offsets__muV=trap_offsets,
    delays__ms=trap_delays,
)

##############################################################################
# Save Current Matrices
# ----------------------
#
# .. note::
#    All **MyoGen** current matrices can be saved to files using ``joblib``.
#    This is useful to **avoid re-generating currents** if you need to use
#    the same parameters in multiple simulations.

print("Saving current matrices...")

joblib.dump(sinusoidal_currents, save_path / "sinusoidal_currents.pkl")
joblib.dump(sawtooth_currents, save_path / "sawtooth_currents.pkl")
joblib.dump(step_currents, save_path / "step_currents.pkl")
joblib.dump(ramp_currents, save_path / "ramp_currents.pkl")
joblib.dump(trapezoid_currents, save_path / "trapezoid_currents.pkl")

##############################################################################
# Plot Current Waveforms
# -----------------------
#
# The current waveforms can be visualized to understand their characteristics
# and verify they match the intended stimulation patterns.
#
# .. note::
#   **Plotting helper functions** are available in the ``myogen.utils.plotting`` module.
#
#   .. code-block:: python
#
#       from myogen.utils.plotting.currents import plot_input_currents

# Suppress font warnings to keep output clean
import warnings
import logging

warnings.filterwarnings("ignore", message=".*Font family.*not found.*")
warnings.filterwarnings("ignore", message=".*findfont.*")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

##############################################################################
# Sinusoidal Currents Visualization
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Sinusoidal currents show smooth oscillations with different frequencies
# and phase relationships between pools.

print("Plotting sinusoidal currents...")
with plt.xkcd():
    _, axs = plt.subplots(figsize=(10, 6), nrows=n_pools, sharex=True)
    plot_input_current__matrix(sinusoidal_currents, timestep__ms, axs, color="#90b8e0")
plt.tight_layout()
plt.show()

##############################################################################
# Sawtooth Currents Visualization
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Sawtooth currents demonstrate asymmetric ramps with different widths
# controlling the rise/fall characteristics.

print("Plotting sawtooth currents...")
with plt.xkcd():
    _, axs = plt.subplots(figsize=(10, 6), nrows=n_pools, sharex=True)
    plot_input_current__matrix(sawtooth_currents, timestep__ms, axs, color="#90b8e0")
plt.tight_layout()
plt.show()

##############################################################################
# Step Currents Visualization
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Step currents show sudden changes in amplitude with different durations
# for each pool.

print("Plotting step currents...")
with plt.xkcd():
    _, axs = plt.subplots(figsize=(10, 6), nrows=n_pools, sharex=True)
    plot_input_current__matrix(step_currents, timestep__ms, axs, color="#90b8e0")
plt.tight_layout()
plt.show()

##############################################################################
# Ramp Currents Visualization
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Ramp currents demonstrate linear changes from start to end values
# with different slopes for each pool.

print("Plotting ramp currents...")
with plt.xkcd():
    _, axs = plt.subplots(figsize=(10, 6), nrows=n_pools, sharex=True)
    plot_input_current__matrix(ramp_currents, timestep__ms, axs, color="#90b8e0")
plt.tight_layout()
plt.show()

##############################################################################
# Trapezoid Currents Visualization
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Trapezoid currents show complex multi-phase patterns with distinct
# rise, plateau, and fall periods.

print("Plotting trapezoid currents...")
with plt.xkcd():
    _, axs = plt.subplots(figsize=(10, 6), nrows=n_pools, sharex=True)
    plot_input_current__matrix(trapezoid_currents, timestep__ms, axs, color="#90b8e0")
plt.tight_layout()
plt.show()
