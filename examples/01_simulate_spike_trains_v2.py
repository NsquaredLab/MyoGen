"""
Motor Unit Spike Trains (NEURON-based)
=======================================

After generating the **recruitment thresholds**, we can simulate the **spike trains** of the motor units.

.. note::
    This example uses the **direct NEURON simulator** with biophysically detailed alpha motor neurons.
    This provides more accurate modeling of motor neuron biophysics compared to the PyNN wrapper approach.

    The AlphaMN__Pool class creates populations of detailed motor neurons with realistic morphology
    and ion channel distributions based on the Powers2017 model.
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
import quantities as pq
from matplotlib import pyplot as plt
from neuron import h

from myogen import RANDOM_GENERATOR
from myogen.simulator.neuron.pops import AlphaMN__Pool
from myogen.utils.currents import create_trapezoid_current
from myogen.utils.neuron.inject_currents_into_populations import (
    inject_currents_and_simulate_spike_trains,
)
from myogen.utils.nmodl import load_nmodl_files
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

timestep = 0.01  # Simulation timestep in ms (reasonable resolution)
simulation_time = 4000  # Total simulation duration in ms

##############################################################################
# Create Motor Neuron Pools
# -------------------------
#
# Since the **recruitment thresholds** are already generated, we can load them from the previous example using ``joblib``.
#
# In the NEURON approach, we use the **AlphaMN__Pool** class which creates biophysically detailed
# alpha motor neurons with realistic morphology and ion channel distributions.
#
# The AlphaMN__Pool uses the Powers2017 model by default, which includes:
# - Detailed soma and dendritic morphology
# - Multiple ion channel types (Na, K, Ca, h-current)
# - Calcium-dependent potassium channels for afterhyperpolarization
# - L-type calcium channels in dendrites for persistent inward currents

save_path = Path("./results")

# Load NMODL mechanisms first
print("Loading NMODL mechanisms...")
load_nmodl_files(force_reload=False, quiet=True)

# Load recruitment thresholds
recruitment_thresholds = joblib.load(save_path / "thresholds.pkl")

# Create motor neuron pools using AlphaMN__Pool
motor_neuron_pools = [
    AlphaMN__Pool(recruitment_thresholds__array=recruitment_thresholds)
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

rise_time_ms = list(RANDOM_GENERATOR.uniform(100, 500, size=n_pools))
plateau_time_ms = list(RANDOM_GENERATOR.uniform(1000, 2000, size=n_pools))
fall_time_ms = list(RANDOM_GENERATOR.uniform(1000, 2000, size=n_pools))

# Create the input current signal
input_current__AnalogSignal = create_trapezoid_current(
    n_pools,
    int(simulation_time / timestep),
    timestep,
    amplitudes__nA=[15.0] * n_pools,
    rise_times__ms=rise_time_ms,
    plateau_times__ms=plateau_time_ms,
    fall_times__ms=fall_time_ms,
    delays__ms=500.0,
)

print(
    f"Input current signal shape: {input_current__AnalogSignal.shape}\\nClass: {input_current__AnalogSignal.__class__}"
)

# Save input current signal for later analysis
joblib.dump(
    input_current__AnalogSignal, save_path / "input_current__AnalogSignal_v2.pkl"
)

##############################################################################
# Setup and Run Complete Simulation Pipeline
# -------------------------------------------
#
# We use the inject_currents_and_simulate_spike_trains function which provides
# a complete end-to-end pipeline: current injection, spike recording, simulation
# execution, and conversion to neo.Block format.

# Set NEURON simulation parameters
h.dt = timestep
h.tstop = simulation_time

# Run complete simulation pipeline: inject currents, record spikes, and return neo.Block
spike_train__Block = inject_currents_and_simulate_spike_trains(
    populations=motor_neuron_pools,
    input_current__AnalogSignal=input_current__AnalogSignal,
)

# Save spike trains
joblib.dump(spike_train__Block, save_path / "spike_train__Block_v2.pkl")

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
    if len(firing_rates_per_pool) > 0 and np.sum(firing_rates_per_pool > 0) > 0:
        mean_rate = np.mean(firing_rates_per_pool[firing_rates_per_pool > 0])
        max_rate = np.max(firing_rates_per_pool)
    else:
        mean_rate = 0.0
        max_rate = 0.0

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
    _, axs = plt.subplots(nrows=2, figsize=(10, 6))
    plot_spike_trains(
        spike_trains__Block=spike_train__Block,
        axs=axs,
        pool_current__AnalogSignal=input_current__AnalogSignal,
    )
plt.tight_layout()
plt.show()
