"""
Motor Unit Spike Trains
==================================

After generating the **recruitment thresholds**, we can simulate the **spike trains** of the motor units.

.. note::
    The spike trains are simulated using the **NEURON simulator** wrapped by **PyNN**.
    This way we can simulate accurately the biophysical properties of the motor units.
"""

##############################################################################
# Import Libraries
# ----------------
#
# .. important::
#    In **MyoGen** all **random number generation** is handled by the ``RANDOM_GENERATOR`` object.
#
#    This object is a wrapper around the ``numpy.random`` module and is used to generate random numbers.
#
#    It is intended to be used with the following API:
#
#    .. code-block:: python
#
#       from myogen import simulator, RANDOM_GENERATOR
#
#    To change the default seed, use ``set_random_seed``:
#
#    .. code-block:: python
#
#       from myogen import set_random_seed
#       set_random_seed(42)

from pathlib import Path

import joblib
import numpy as np
from matplotlib import pyplot as plt

from myogen import simulator, RANDOM_GENERATOR
from myogen.utils import load_nmodl_files
from myogen.utils.currents import create_trapezoid_current
from myogen.utils.plotting import plot_spike_trains

##############################################################################
# Define Parameters
# -----------------
# In this example we will simulate a **motor pool** using the **recruitment thresholds** generated in the previous example.
#
# This motor pool will have **two different randomly generated trapezoidal ramp currents** injected into the motor units.
#
# The parameters of the input current are:
#
# - ``n_pools``: Number of distinct motor neuron pools
# - ``timestep``: Simulation timestep in ms (high resolution)
# - ``simulation_time``: Total simulation duration in ms
#
# To simulate realistic spike trains, we will also add a **common noise current source** to each neuron.
# The parameters of the noise current are:
#
# - ``noise_mean``: Mean noise current in nA
# - ``noise_stdev``: Standard deviation of noise current in nA

n_pools = 2  # Number of distinct motor neuron pools

timestep = 0.05  # Simulation timestep in ms (high resolution)
simulation_time = 2000  # Total simulation duration in ms

noise_mean = 26  # Mean noise current in nA
noise_stdev = 20  # Standard deviation of noise current in nA

##############################################################################
# Create Input Currents
# ------------------------------
#
# To drive the motor units, we use a **common input current profile**.
#
# In this example, we use a **trapezoid-shaped input current** which is generated using the ``create_trapezoid_current`` function.
#
# .. note::
#    More convenient functions for generating input current profiles are available in the ``myogen.utils.currents`` module.

# Calculate number of time points
t_points = int(simulation_time / timestep)

# Generate random parameters for each pool's input current
amplitude_range = list(RANDOM_GENERATOR.uniform(100, 150, size=n_pools))
rise_time_ms = list(RANDOM_GENERATOR.uniform(100, 500, size=n_pools))
plateau_time_ms = list(RANDOM_GENERATOR.uniform(100, 500, size=n_pools))
fall_time_ms = list(RANDOM_GENERATOR.uniform(100, 500, size=n_pools))

print(f"\nInput current parameters:")
for i in range(n_pools):
    print(
        f"  Pool {i + 1}: amplitude={amplitude_range[i]:.1f} nA, "
        f"rise={rise_time_ms[i]:.0f} ms, "
        f"plateau={plateau_time_ms[i]:.0f} ms, "
        f"fall={fall_time_ms[i]:.0f} ms"
    )

# Create the input current matrix
input_current_matrix = create_trapezoid_current(
    n_pools,
    t_points,
    timestep,
    amplitudes__muV=amplitude_range,
    rise_times__ms=rise_time_ms,
    plateau_times__ms=plateau_time_ms,
    fall_times__ms=fall_time_ms,
    delays__ms=500.0,
)

print(f"\nInput current matrix shape: {input_current_matrix.shape}")

##############################################################################
# Create Motor Neuron Pools
# -------------------------
#
# Since the **recruitment thresholds** are already generated, we can load them from the previous example using ``joblib``.
#
# Afterwards the custom files made for the ``NEURON`` simulator must be loaded.
#
# .. note::
#   This step is required as ``NEURON`` does not support the simulation of motor units directly.
#
#   This is done using the ``load_nmodl_files`` function.
#
# Finally, the **motor neuron pools** are created using the ``MotorNeuronPool`` object.
#
# .. note::
#    The **MotorNeuronPool** object handles the simulation of the motor units.
#

save_path = Path("./results")

# Load recruitment thresholds
recruitment_thresholds = joblib.load(save_path / "thresholds.pkl")

# Save input current matrix for later analysis
joblib.dump(input_current_matrix, save_path / "input_current_matrix.pkl")

# Load NEURON mechanism
load_nmodl_files()

# Create motor neuron pool
motor_neuron_pool = simulator.MotorNeuronPool(recruitment_thresholds)

##############################################################################
# Simulate Motor Unit Spike Trains
# ---------------------------------
#
# The **motor unit spike trains** are simulated using the ``generate_spike_trains`` method of the ``MotorNeuronPool`` object.

spike_trains_matrix, active_neuron_indices, data = motor_neuron_pool.generate_spike_trains(
    input_current__matrix=input_current_matrix,
    timestep__ms=timestep,
    noise_mean__nA=noise_mean,
    noise_stdev__nA=noise_stdev,
)

# Save motor neuron pool for later analysis
joblib.dump(motor_neuron_pool, save_path / "motor_neuron_pool.pkl")

print(f"\nSpike trains shape: {spike_trains_matrix.shape}")
print(f"  - {spike_trains_matrix.shape[0]} pools")
print(f"  - {spike_trains_matrix.shape[1]} neurons per pool")
print(f"  - {spike_trains_matrix.shape[2]} time points\n")

##############################################################################
# Calculate and Display Statistics
# ---------------------------------
#
# It might be of interest to calculate the **firing rates** of the motor units.
#
# .. note::
#    The **firing rates** are calculated as the number of spikes divided by the simulation time.
#    The simulation time is in milliseconds, so we need to convert it to seconds.

# Calculate firing rates for each pool
firing_rates = np.zeros((n_pools, len(motor_neuron_pool.recruitment_thresholds)))
for pool_idx in range(n_pools):
    for neuron_idx in range(len(motor_neuron_pool.recruitment_thresholds)):
        spike_count = np.sum(spike_trains_matrix[pool_idx, neuron_idx, :])
        firing_rates[pool_idx, neuron_idx] = spike_count / (simulation_time / 1000.0)

print(f"\nFiring rate statistics:")
for pool_idx in range(n_pools):
    active_neurons = np.sum(firing_rates[pool_idx, :] > 0)
    mean_rate = np.mean(firing_rates[pool_idx, firing_rates[pool_idx, :] > 0])
    max_rate = np.max(firing_rates[pool_idx, :])

    print(
        f"  Pool {pool_idx + 1}: {active_neurons}/{len(motor_neuron_pool.recruitment_thresholds)} active neurons, "
        f"mean rate: {mean_rate:.1f} Hz, max rate: {max_rate:.1f} Hz"
    )

##############################################################################
# Visualize Spike Trains
# ----------------------
#
# The **spike trains** can be visualized using the ``plot_spike_trains`` function.
#
# .. note::
#    **Plotting helper functions** are available in the ``myogen.utils.plotting`` module.
#
#    .. code-block:: python
#
#       from myogen.utils.plotting import plot_spike_trains

# Suppress font warnings to keep output clean
import warnings
import logging

warnings.filterwarnings("ignore", message=".*Font family.*not found.*")
warnings.filterwarnings("ignore", message=".*findfont.*")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

print("Plotting spike trains...")
with plt.xkcd():
    _, ax = plt.subplots(figsize=(10, 6))
    plot_spike_trains(
        spike_trains__matrix=spike_trains_matrix,
        timestep__ms=timestep,
        axs=[ax],
        pool_current__matrix=input_current_matrix,
        pool_to_plot=[0],
    )
plt.tight_layout()
plt.show()
