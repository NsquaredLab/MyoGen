"""
Force Generation - Jaxley Backend
=================================

After generating **spike trains** from motor neuron pools, the next step is to simulate the **force output**
produced by the muscle. **MyoGen** provides a comprehensive force model based on the classic
Fuglevand et al. (1993) approach.

This example uses the Jaxley backend to generate spike trains with the NERLab motor-neuron
model (``napp`` Na/K/leak on soma; ``caL`` L-type Ca + leak on dendrite — identical
channel set to the NEURON ex08 production setup). It then feeds the spike trains into the
shared ``ForceModel`` which is simulator-agnostic and identical to the NEURON version.

.. note::
    The **force model** is shared between NEURON and Jaxley backends.  It converts spike trains
    into force by simulating individual motor unit twitches and their summation.  The Jaxley-specific
    part is only the spike train generation.

.. note::
    **Voltage convention.** NERLab cells live in the *original 1952 HH frame*: V_rest ≈ 0 mV,
    ENa = +120 mV, EK = -10 mV, spike peaks ≈ +90 mV. The pool sets ``initial_voltage__mV = 0.0``
    and ``spike_threshold__mV = 50.0`` automatically; the spike-detection threshold below is
    set in this frame too.

.. note::
    **Current amplitude**: 15 nA, matching the NEURON ex08 setup exactly (see ex02_jaxley
    and ex02 NEURON: parity within 0.3%). The earlier 18 nA was Powers2017-era tuning and
    is no longer relevant for NERLab cells.

The force model includes:

* **Motor unit twitch characteristics**: Peak force and contraction time based on recruitment thresholds
* **Nonlinear gain modulation**: Force scaling based on inter-pulse intervals (discharge rate)
* **Temporal summation**: Realistic force buildup from overlapping twitches
* **Physiological scaling**: Realistic force ranges and dynamics

References
----------
.. [1] Fuglevand, A. J., Winter, D. A., & Patla, A. E. (1993). Models of recruitment
    and rate coding organization in motor-unit pools. J Neurophysiol, 70(6), 2470–2488.
    https://doi.org/10.1152/jn.1993.70.6.2470.
"""

##############################################################################
# Import Libraries
# ----------------

# %%
import os
import jax.numpy as jnp
import jaxley as jx
import matplotlib.pyplot as plt
import numpy as np
import quantities as pq

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
from neo import Block, Segment, SpikeTrain

from myogen.simulator import RecruitmentThresholds
from myogen.simulator.core.force.force_model import ForceModel
from myogen.simulator.jaxley.populations import AlphaMN__Pool
from myogen.utils.currents import create_trapezoid_current
from myogen.utils.plotting.force import plot_twitch_parameter_assignment, plot_twitches

plt.style.use("fivethirtyeight")

##############################################################################
# Define Parameters
# -----------------
#
# The **force model** requires several key parameters:
#
# - ``recruitment_thresholds``: Motor unit recruitment thresholds
# - ``recording_frequency__Hz``: Sampling frequency for force simulation
# - ``longest_duration_rise_time__ms``: Maximum twitch rise time
# - ``contraction_time_range``: Range of contraction times across motor units
#
# For demonstration, we'll also create input currents and simulate the complete
# pipeline from current input to force output.

# Motor unit pool parameters
n_motor_units = 50
recruitment_range = 50  # Recruitment range (max_threshold / min_threshold)

# Force model parameters (same as NEURON version — ForceModel is simulator-agnostic)
recording_frequency__Hz = 2048 * pq.Hz  # 2048 Hz sampling rate
longest_duration_rise_time__ms = 90.0 * pq.ms  # Maximum twitch rise time
contraction_time_range = 3  # Contraction time range factor

# Simulation parameters
simulation_duration__ms = 10000.0  # 10 seconds
timestep__ms = 0.05 * pq.ms  # 0.05 ms time step
t_points = int(simulation_duration__ms / timestep__ms.magnitude)

##############################################################################
# Generate Recruitment Thresholds
# --------------------------------
#
# First, we need to generate recruitment thresholds for our motor unit pool.
# These will determine both the activation order and the force characteristics
# of each motor unit.

recruitment_thresholds, _ = RecruitmentThresholds(
    N=n_motor_units,
    recruitment_range__ratio=recruitment_range,
    mode="combined",
    deluca__slope=5,
)

##############################################################################
# Create Force Model
# ------------------
#
# The force model calculates individual motor unit twitch properties based on
# recruitment thresholds and physiological scaling rules.

force_model = ForceModel(
    recruitment_thresholds=recruitment_thresholds,
    recording_frequency__Hz=recording_frequency__Hz,
    longest_duration_rise_time__ms=longest_duration_rise_time__ms,
    contraction_time_range_factor=contraction_time_range,
)

# Display force model statistics
print("Force model statistics:")
print(f"\tNumber of motor units: {force_model._number_of_neurons}")
print(f"\tRecruitment ratio: {force_model._recruitment_ratio:.1f}")
print(
    f"\tPeak force range: {force_model.peak_twitch_forces__unitless[0]:.3f} - {force_model.peak_twitch_forces__unitless[-1]:.3f}"
)
print(
    f"\tContraction time range: {force_model.contraction_times__samples[0]:.1f} - {force_model.contraction_times__samples[-1]:.1f} samples"
)

##############################################################################
# Visualize Twitch Parameter Assignment
# --------------------------------------
#
# The force model assigns twitch parameters (peak force and contraction time)
# to each motor unit based on its recruitment threshold. Let's visualize this
# relationship for a subset of motor units.

plt.figure(figsize=(8, 12))

ax1 = plt.subplot(2, 1, 1)
plot_twitch_parameter_assignment(
    force_model, ax1, [10, 20, 40], flip_x=True, apply_default_formatting=True
)
ax1.set_title("Twitch Parameter Assignment")

ax2 = plt.subplot(2, 1, 2)
plot_twitches(force_model, ax2, [10, 20, 40], apply_default_formatting=True)
ax2.set_title("Motor Unit Twitches")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "ex08_jaxley_twitches.png"), dpi=150, bbox_inches="tight")
plt.show()

##############################################################################
# Generate Input Current
# ----------------------
#
# To demonstrate force generation, we'll create a trapezoid current that will
# drive motor unit recruitment and firing patterns.
# Parameters match NEURON Example 8 exactly.

# Parameters for trapezoid current — matches NEURON ex08 exactly.
trap_amplitude = 15.0 * pq.nA
trap_rise_time = 5000.0 * pq.ms  # Rise duration (ms)
trap_plateau_time = 8000.0 * pq.ms  # Plateau duration (ms)
trap_fall_time = 3000.0 * pq.ms  # Fall duration (ms)
trap_offset = 5.0 * pq.nA  # Baseline current (same as NEURON)
trap_delay = 0.0 * pq.ms  # Initial delay (ms)

input_current__AnalogSignal = create_trapezoid_current(
    n_pools=1,
    t_points=t_points,
    timestep__ms=timestep__ms,
    amplitudes__nA=[trap_amplitude],
    rise_times__ms=[trap_rise_time],
    plateau_times__ms=[trap_plateau_time],
    fall_times__ms=[trap_fall_time],
    offsets__nA=[trap_offset],
    delays__ms=[trap_delay],
)

##############################################################################
# Create Motor Neuron Pool and Generate Spike Trains (Batched)
# ------------------------------------------------------------
#
# Creates a motor neuron pool with NERLab channels (napp + caL), matching the
# production NEURON model.  All neurons receive the SAME current; recruitment
# emerges from biophysics (Henneman size principle: smaller soma → higher Rin
# → lower threshold).
#
# All cells are assembled into a single jx.Network and integrated in one
# jx.integrate() call instead of 50 serial calls.

motor_neuron_pool = AlphaMN__Pool(
    recruitment_thresholds__array=recruitment_thresholds,
    mode="active",
    # model defaults to "NERLab"; pool auto-sets V_init = 0 mV and
    # spike_threshold = 50 mV for the NERLab voltage frame.
)

dt_ms   = float(timestep__ms.rescale(pq.ms).magnitude)
t_max_ms = float(simulation_duration__ms)
pool_current = jnp.array(np.array(input_current__AnalogSignal.magnitude)[:, 0])

# NERLab APs peak at ~+90 mV from V_rest = 0 mV; threshold at +50 mV avoids the
# resting-state drift false positives a 0 mV threshold would produce.
spike_detection_threshold__mV = 50.0

print("\n" + "=" * 60)
print("Running batched biophysical simulation with jx.integrate()")
print(f"Model: {motor_neuron_pool.model}  (soma napp; dendrite caL)")
print("=" * 60)

# Assemble all MN cells into a single Network; integrate once.
mn_cells = []
for cw in motor_neuron_pool:
    cell = cw.cell
    cell.delete_recordings()
    cell.delete_stimuli()
    mn_cells.append(cell)

net = jx.Network(mn_cells)
for mn_idx in range(n_motor_units):
    net.cell(mn_idx).branch(0).loc(0.5).record("v")
    net.cell(mn_idx).branch(0).loc(0.5).stimulate(pool_current)  # same current to all

print(f"Integrating {n_motor_units}-cell network (1 call instead of {n_motor_units})...")
all_voltages = jx.integrate(net, delta_t=dt_ms, t_max=t_max_ms)
# all_voltages shape: (n_motor_units, n_timesteps)

# Build Neo Block from voltage matrix
spike_train__Block = Block(name="Jaxley Motor Neuron Simulation")
segment = Segment(name="Pool 0")
segment.spiketrains = []

active_neurons = 0
total_spikes   = 0

for neuron_idx in range(n_motor_units):
    v = np.array(all_voltages[neuron_idx])
    spike_indices = np.where(
        (v[:-1] < spike_detection_threshold__mV) &
        (v[1:] >= spike_detection_threshold__mV)
    )[0]
    spike_times_ms = spike_indices * dt_ms

    if len(spike_times_ms) > 0:
        active_neurons += 1
        total_spikes   += len(spike_times_ms)

    spiketrain = SpikeTrain(
        (spike_times_ms / 1000.0) * pq.s,
        t_stop=(simulation_duration__ms / 1000.0) * pq.s,
        sampling_rate=(1 / timestep__ms).rescale(pq.Hz),
        sampling_period=timestep__ms.rescale(pq.s),
        name=str(neuron_idx),
        description=f"Pool 0, Neuron {neuron_idx}",
    )
    segment.spiketrains.append(spiketrain)

spike_train__Block.segments.append(segment)

print(f"\nActive neurons: {active_neurons}/{n_motor_units}")
print(f"Total spikes:   {total_spikes}")

# Calculate per-neuron firing rates
firing_rates = []
for st in segment.spiketrains:
    if len(st) > 0:
        rate = len(st) / (simulation_duration__ms / 1000.0)
        firing_rates.append(rate)

if len(firing_rates) > 0:
    firing_rates = np.array(firing_rates)
    print(f"\nFiring rate statistics:")
    print(f"  Mean: {np.mean(firing_rates):.1f} Hz")
    print(f"  Min:  {np.min(firing_rates):.1f} Hz")
    print(f"  Max:  {np.max(firing_rates):.1f} Hz")
    print(f"  Std:  {np.std(firing_rates):.1f} Hz")

    # Check for unreasonably high rates
    high_rate_neurons = np.sum(firing_rates > 50)
    if high_rate_neurons > 0:
        print(f"\n  WARNING: {high_rate_neurons} neurons firing > 50 Hz (may cause jagged force)")

##############################################################################
# Generate Force Output
# ---------------------
#
# Now we can use the force model to convert the spike trains into force output.
# The model will simulate individual motor unit twitches and their temporal
# summation to produce the total muscle force.

# Generate force from spike trains
force_output = force_model.generate_force(spike_train__Block=spike_train__Block)

# Add some realistic noise to the force signal
noise_level = 0.015  # 0.15% of mean force
noisy_force = force_output.magnitude[:, 0] + np.random.randn(
    len(force_output.magnitude[:, 0])
) * noise_level * np.mean(force_output.magnitude[:, 0])

##############################################################################
# Visualize Force Output
# ----------------------
#
# Let's plot the generated force alongside the input current to see how
# the muscle responds to the electrical stimulation.

plt.figure(figsize=(8, 10))

ax1 = plt.subplot(2, 1, 1)
ax1.plot(
    force_output.times.rescale("s"),
    force_output[:, 0],
    linewidth=2,
    label="Clean Force",
)
ax1.set_ylabel("Force (a.u.)")
ax1.set_title("Simulated Force Output")
ax1.grid(True, alpha=0.3)

# Plot noisy force (more realistic)
ax2 = plt.subplot(2, 1, 2)
ax2.plot(
    force_output.times.rescale("s"),
    noisy_force,
    linewidth=1,
    alpha=0.8,
    label="Noisy Force",
)
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Force (a.u.)")
ax2.set_title("Realistic Force Output (with noise)")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "ex08_jaxley_force.png"), dpi=150, bbox_inches="tight")
plt.show()
