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

import elephant
import joblib
import numpy as np
import pyNN.neuron as sim
import quantities as pq
from matplotlib import pyplot as plt
from neo import Block

from myogen import RANDOM_GENERATOR, simulator
from myogen.utils.currents import create_trapezoid_current
from myogen.utils.plotting import plot_spike_trains
from myogen.utils.pyNN import (
    compute_maximum_activation_current_thresholds,
    inject_currents_into_populations,
)

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

timestep = 0.01  # Simulation timestep in ms (high resolution)
simulation_time = 2000  # Total simulation duration in ms

##############################################################################
# Create Motor Neuron Pools
# -------------------------
#
# Since the **recruitment thresholds** are already generated, we can load them from the previous example using ``joblib``.
#
# In pyNN a motor neuron **pool** is refered to as a motor neuron **population**.
# To avoid confusion we christen the class that can create a motor neuron pool as
# ``MotorNeuronPopulation``.
#

save_path = Path("./results")

# Load recruitment thresholds
recruitment_thresholds = joblib.load(save_path / "thresholds.pkl")

# Create motor neuron pools
motor_neuron_pools = [
    simulator.pyNN.neurons.MotorNeuron__Population(recruitment_thresholds)
    for _ in range(n_pools)
]

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

# get maximum activation threshold for the pool
max_activation_thresholds = compute_maximum_activation_current_thresholds(
    motor_neuron_pools
)

# Generate random parameters for each pool's input current
amplitude_range = max_activation_thresholds
rise_time_ms = list(RANDOM_GENERATOR.uniform(100, 500, size=n_pools))
plateau_time_ms = list(RANDOM_GENERATOR.uniform(100, 500, size=n_pools))
fall_time_ms = list(RANDOM_GENERATOR.uniform(100, 500, size=n_pools))

print("Input current parameters:")
for i in range(n_pools):
    print(
        f"  Pool {i + 1}: amplitude={amplitude_range[i]:.1f} nA, "
        f"rise={rise_time_ms[i]:.0f} ms, "
        f"plateau={plateau_time_ms[i]:.0f} ms, "
        f"fall={fall_time_ms[i]:.0f} ms"
    )

# Create the input current signal
input_current__AnalogSignal = create_trapezoid_current(
    n_pools,
    t_points,
    timestep,
    amplitudes__nA=amplitude_range,
    rise_times__ms=rise_time_ms,
    plateau_times__ms=plateau_time_ms,
    fall_times__ms=fall_time_ms,
    delays__ms=500.0,
)

print(
    f"Input current signal shape: {input_current__AnalogSignal.shape}\nClass: {input_current__AnalogSignal.__class__}"
)

# Save input current signal for later analysis
joblib.dump(input_current__AnalogSignal, save_path / "input_current__AnalogSignal.pkl")

##############################################################################
# Simulate Motor Unit Spike Trains
# ---------------------------------
#
# The **motor unit spike trains** are simulated using the ``generate_spike_trains`` method of the ``MotorNeuronPool`` object.

# Setup simulation
sim.setup(timestep=timestep)

# Inject currents into populations
inject_currents_into_populations(
    input_current__AnalogSignal=input_current__AnalogSignal,
    populations=motor_neuron_pools,
)

# Tell pyNN to record spikes of each pool
for pool in motor_neuron_pools:
    pool.record("spikes")

# Run the simulation and clean up after end
sim.run(simulation_time)
sim.end()

# Store the spike trains in a Neo Block for later use
# see https://neuralensemble.org/docs/PyNN/data_handling.html and
# https://neo.readthedocs.io/en/latest/api_reference.html#neo.core.Block

spike_train__Block = Block()
spike_train__Block.segments.extend(
    [pool.get_data().segments[0] for pool in motor_neuron_pools]
)

# This shouldn't be necessary if pyNN were correctly handling this. Apparently it is not.
for segment in spike_train__Block.segments:
    for spike_train in segment.spiketrains:
        spike_train.sampling_period = input_current__AnalogSignal.sampling_period
        spike_train.sampling_rate = input_current__AnalogSignal.sampling_rate

joblib.dump(spike_train__Block, save_path / "spike_train__Block.pkl")

##############################################################################
# Calculate and Display Statistics
# ---------------------------------
#
# It might be of interest to calculate the **firing rates** of the motor units.
#
# .. note::
#    The **firing rates** are calculated as the number of spikes divided by the time in which each MU was active.
#    The simulation time is in milliseconds, so we need to convert it to seconds.

firing_rates = [
    np.array(
        [
            elephant.statistics.mean_firing_rate(
                st__s.time_slice(st__s.min(), st__s.max())
            )
            for st__ms in spike_train__segment.spiketrains
            if len(st__s := st__ms.rescale(pq.s)) > 0
        ]
    )
    for spike_train__segment in spike_train__Block.segments
]

print("Firing rate statistics:")
for pool_idx, firing_rates_per_pool in enumerate(firing_rates):
    active_neurons = np.sum(firing_rates_per_pool > 0)
    mean_rate = np.mean(firing_rates_per_pool[firing_rates_per_pool > 0])
    max_rate = np.max(firing_rates_per_pool)

    print(
        f"  Pool {pool_idx + 1}: {active_neurons}/{len(recruitment_thresholds)} active neurons, "
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

with plt.xkcd():
    _, ax = plt.subplots(figsize=(10, 6))
    plot_spike_trains(
        spike_trains__Block=spike_train__Block,
        axs=[ax],
        pool_current__AnalogSignal=input_current__AnalogSignal,
        pool_to_plot=[0],
    )
plt.tight_layout()
plt.show()
