"""
Spike Train Generation with Current Injection - Jaxley Backend (NERLab)
=======================================================================

This example demonstrates how to simulate spike trains in a population of alpha motor neurons
using current injection with the **Jaxley** (JAX-based) simulator.

This is designed to produce output comparable to the NEURON version
(``02_simulate_spike_trains_current_injection.py``), which uses the SAME ``model="NERLab"``
motor-neuron model.

Key features:
    - Uses the NERLab architecture (soma + 1 isopotential dendrite), matching the
      production NEURON model
    - NERLab channels: ``napp`` (Na fast + Na persistent + Kfast + Kslow + leak) on the
      soma; ``caL`` (L-type Ca, no inactivation + leak) on the dendrite
    - Same current to ALL neurons — recruitment from biophysics (Henneman size principle)
    - Runs full biophysical simulation with ``jx.integrate()``

.. note::
    **Voltage convention.** NERLab cells live in the *original 1952 HH frame*:
    V_rest ≈ 0 mV, ENa = +120 mV, EK = -10 mV, spike peaks ≈ +90 mV. This is set
    automatically by ``AlphaMN__Pool`` when ``model="NERLab"`` (the default), but if
    you reinitialise ``v`` manually anywhere in your pipeline you must use 0 mV, not
    -65 mV, and detect spikes at ~+50 mV (not 0 mV).

.. note::
    Jaxley uses JAX for accelerated computation. On first run, JAX will compile
    the simulation which may take a few seconds, but subsequent runs are faster.
"""

# %%

##############################################################################
# Import Libraries
# ----------------

from pathlib import Path

import jax.numpy as jnp
import jaxley as jx
import joblib
import numpy as np
import quantities as pq
import seaborn as sns
from matplotlib import pyplot as plt
from neo import Block, Segment, SpikeTrain
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from myogen import get_random_generator, simulator
from myogen.simulator.jaxley.populations import AlphaMN__Pool
from myogen.utils.currents import create_trapezoid_current

plt.style.use("fivethirtyeight")


def mean_firing_rate(spiketrain):
    """Mean firing rate of a neo.SpikeTrain (replaces elephant.statistics.mean_firing_rate)."""
    return (len(spiketrain) / (spiketrain.t_stop - spiketrain.t_start)).rescale(pq.Hz)


def rasterplot_rates(spiketrains, filter_function=None):
    """Native spike raster with top/right marginal axes.

    Lightweight stand-in for ``viziphant.rasterplot.rasterplot_rates``: draws a
    spike raster (one row per train), a right marginal with each train's mean
    firing rate, and an (initially empty) top marginal that the caller fills with
    the smoothed population rate. Returns ``(ax, axhistx, axhisty)``.
    """
    if filter_function is not None:
        spiketrains = [st for st in spiketrains if filter_function(st)]

    fig = plt.figure()
    ax = fig.add_axes((0.10, 0.10, 0.62, 0.62))
    axhistx = fig.add_axes((0.10, 0.74, 0.62, 0.16), sharex=ax)
    axhisty = fig.add_axes((0.74, 0.10, 0.16, 0.62), sharey=ax)

    ax.eventplot(
        [st.rescale(pq.s).magnitude for st in spiketrains],
        lineoffsets=np.arange(len(spiketrains)),
        colors="black",
        linelengths=0.8,
        linewidths=0.7,
    )
    rates = [float(mean_firing_rate(st).magnitude) for st in spiketrains]
    axhisty.barh(np.arange(len(spiketrains)), rates, height=0.85, color="C0")
    ax.set_ylim(-1, max(len(spiketrains), 1))
    return ax, axhistx, axhisty


##############################################################################
# Create Motor Neuron Populations (Pools)
# ---------------------------------------
#
# We create motor neuron pools using the :class:`~myogen.simulator.jaxley.populations.AlphaMN__Pool`
# class. The default ``model="NERLab"`` builds the same architecture as the production
# NEURON model: soma + 1 isopotential dendrite, ``napp`` + ``caL`` channels.
#
# Recruitment emerges naturally from biophysics (Henneman size principle):
# - Low threshold MNs: smaller soma, higher input resistance -> easier to activate
# - High threshold MNs: larger soma, lower input resistance -> harder to activate

save_path = Path("./results")
save_path.mkdir(exist_ok=True)

# Load recruitment thresholds from Example 1 (or generate defaults)
try:
    recruitment_thresholds = joblib.load(save_path / "thresholds.pkl")
    print(f"Loaded recruitment thresholds: {len(recruitment_thresholds)} values")
except FileNotFoundError:
    print("Recruitment thresholds not found. Generating defaults matching ex01...")
    n_motor_units = 100
    recruitment_thresholds, _ = simulator.RecruitmentThresholds(
        N=n_motor_units,
        recruitment_range__ratio=100,
        deluca__slope=5,
        konstantin__max_threshold__ratio=1.0,
        mode="combined",
    )
    joblib.dump(recruitment_thresholds, save_path / "thresholds.pkl")

n_pools = 2

# Create motor neuron pools — defaults to model="NERLab" (matches the production
# NEURON model: soma + 1 isopotential dendrite, napp + caL channels).
motor_neuron_pools = [
    AlphaMN__Pool(
        recruitment_thresholds__array=recruitment_thresholds,
        mode="active",
        # model defaults to "NERLab"; use_jaxley_mech is ignored for NERLab.
    )
    for _ in range(n_pools)
]

print(f"Created {n_pools} motor neuron pools with {len(recruitment_thresholds)} neurons each")
print(f"  Model: {motor_neuron_pools[0].model}  (soma + 1 isopotential dendrite, napp + caL)")

##############################################################################
# Create Input Currents
# ---------------------
#
# Same current to ALL neurons, matching the NEURON approach.
# Recruitment determined by biophysics (cell size, conductances) — no manual cutoff.

timestep = 0.025 * pq.ms  # 0.025 ms = 25 µs, typical for HH simulations
simulation_time = 4000 * pq.ms

# Current amplitude — same to ALL neurons; biophysics determines recruitment.
# 15 nA matches the NEURON example exactly (see
# 02_simulate_spike_trains_current_injection.py:118). Using the same drive
# level lets the comparison block at the bottom of this script be
# apples-to-apples (active count, total spikes, mean rate).
current_amplitude = 15.0 * pq.nA

# NO max_recruitment cutoff - simulate ALL neurons like NEURON does
# Recruitment emerges from biophysics (cell size, conductances)

rise_time_ms = list(get_random_generator().uniform(100, 500, size=n_pools)) * pq.ms
plateau_time_ms = list(get_random_generator().uniform(1000, 2000, size=n_pools)) * pq.ms
fall_time_ms = list(get_random_generator().uniform(1000, 2000, size=n_pools)) * pq.ms

input_current__AnalogSignal = create_trapezoid_current(
    n_pools,
    int(simulation_time / timestep) + 1,
    timestep,
    amplitudes__nA=[current_amplitude] * n_pools,
    rise_times__ms=rise_time_ms,
    plateau_times__ms=plateau_time_ms,
    fall_times__ms=fall_time_ms,
    delays__ms=500.0 * pq.ms,
)

print(f"\nInput current signal shape: {input_current__AnalogSignal.shape}")
print(f"Timestep: {timestep}, Total time: {simulation_time}")
print(f"Current amplitude: {current_amplitude} (same for ALL neurons - NEURON approach)")
print("Recruitment determined by biophysics (cell size, conductances) - no manual cutoff")

# Save input current signal
joblib.dump(input_current__AnalogSignal, save_path / "input_current__AnalogSignal_jaxley.pkl")

##############################################################################
# Manual Simulation Approach - Step by Step (Jaxley Biophysics)
# -------------------------------------------------------------
#
# We walk through each stage of the Jaxley biophysical simulation:
#
# 1. Extract simulation parameters
# 2. For each neuron: inject current, run jx.integrate(), detect spikes
# 3. Convert to Neo format
#
# The key function is ``jx.integrate()`` which solves the HH differential
# equations to compute membrane voltage over time.

# Step 1: Extract simulation parameters
dt_ms = float(timestep.rescale(pq.ms).magnitude)
t_max_ms = float(simulation_time.rescale(pq.ms).magnitude)

# Spike detection threshold — NERLab cells live in the 1952-HH frame
# (V_rest ≈ 0 mV, AP peaks ≈ +90 mV), so a positive threshold avoids
# false-positives from the resting-state membrane drift.
spike_detection_threshold__mV = 50.0

# Convert Neo AnalogSignal to numpy array
current_data = np.array(input_current__AnalogSignal.magnitude)
print(f"\nCurrent data shape: {current_data.shape}")

# Step 2: Simulate each pool and collect spike trains
spike_train__Block_manual = Block(name="Manual Jaxley NERLab Biophysical Simulation")

print("\n" + "=" * 60)
print("Running biophysical simulations with jx.integrate()")
print("Using NERLab channels (napp + caL; 1 soma + 1 isopotential dendrite)")
print("=" * 60)

for pool_idx, pool in enumerate(motor_neuron_pools):
    print(f"\nSimulating Pool {pool_idx}...")

    # Get current waveform for this pool
    pool_current = jnp.array(current_data[:, pool_idx])

    # Create a segment for this pool's spike trains
    segment = Segment(name=f"Pool {pool_idx}")
    segment.spiketrains = []

    # Simulate each neuron in the pool
    # NEURON approach: simulate ALL neurons with SAME current
    # Recruitment emerges from biophysics (cell size, conductances)
    for neuron_idx, cell_wrapper in enumerate(tqdm(pool, desc=f"  Pool {pool_idx}", leave=False)):
        spike_times_ms = np.array([])

        if hasattr(cell_wrapper, 'cell'):
            cell = cell_wrapper.cell

            try:
                # === BIOPHYSICAL SIMULATION WITH MAHP FOR AHP ===
                cell.delete_recordings()
                cell.delete_stimuli()

                # Reset V to NERLab resting potential (≈ 0 mV in the 1952-HH frame)
                # and reinitialise channel states at that V.
                cell.set("v", 0.0)
                cell.init_states()

                # Set up voltage recording on soma
                cell.branch(0).loc(0.5).record("v")

                # Inject SAME current to ALL neurons on soma (NEURON approach)
                # Recruitment determined by biophysics, not current scaling
                cell.branch(0).loc(0.5).stimulate(pool_current)

                # Run biophysical simulation
                voltages = jx.integrate(cell, delta_t=dt_ms, t_max=t_max_ms)

                # Extract voltage trace
                if voltages.ndim == 2:
                    v = np.array(voltages[0])
                else:
                    v = np.array(voltages)

                # Detect spikes (threshold crossings)
                spike_indices = np.where(
                    (v[:-1] < spike_detection_threshold__mV) &
                    (v[1:] >= spike_detection_threshold__mV)
                )[0]
                spike_times_ms = spike_indices * dt_ms

            except Exception as e:
                if neuron_idx == 0:
                    print(f"  Warning: Neuron {neuron_idx} simulation failed: {e}")

        # Convert to Neo SpikeTrain
        spike_times_s = spike_times_ms / 1000.0

        spiketrain = SpikeTrain(
            spike_times_s * pq.s,
            t_stop=simulation_time.rescale(pq.s),
            sampling_rate=(1 / timestep).rescale(pq.Hz),
            sampling_period=timestep.rescale(pq.s),
            name=str(neuron_idx),
            description=f"Pool {pool_idx}, Neuron {neuron_idx}",
        )
        segment.spiketrains.append(spiketrain)

    spike_train__Block_manual.segments.append(segment)

    # Report statistics for this pool
    active_count = sum(1 for st in segment.spiketrains if len(st) > 0)
    total_spikes = sum(len(st) for st in segment.spiketrains)
    print(f"  Pool {pool_idx}: {active_count}/{len(pool)} active neurons, {total_spikes} total spikes")

# Save results — single file; no separate utility implementation exists for Jaxley
spike_train__Block = spike_train__Block_manual
joblib.dump(spike_train__Block, save_path / "spike_train__Block_utility_jaxley.pkl")

##############################################################################
# Calculate and Display Statistics
# ---------------------------------

firing_rates = [
    np.array(
        [
            mean_firing_rate(st__s.time_slice(st__s.min(), st__s.max()))
            for st__s in spike_train__segment.spiketrains
            if len(st__s) > 0
        ]
    )
    for spike_train__segment in spike_train__Block.segments
]

print("\n" + "=" * 60)
print("Firing Rate Statistics")
print("=" * 60)

for pool_idx, firing_rates_per_pool in enumerate(firing_rates):
    if len(firing_rates_per_pool) > 0:
        active_neurons = np.sum(firing_rates_per_pool > 0)
        mean_rate = np.mean(firing_rates_per_pool[firing_rates_per_pool > 0]) if active_neurons > 0 else 0.0
        max_rate = np.max(firing_rates_per_pool) if len(firing_rates_per_pool) > 0 else 0.0
    else:
        active_neurons = 0
        mean_rate = 0.0
        max_rate = 0.0

    print(
        f"  Pool {pool_idx + 1}: {active_neurons}/{len(recruitment_thresholds)} active neurons, "
        f"mean rate: {mean_rate:.1f} Hz, max rate: {max_rate:.1f} Hz"
    )

##############################################################################
# Visualize Spike Trains
# ----------------------

spike_train_list = list(spike_train__Block.segments[0].spiketrains)
active_spiketrains = [st for st in spike_train_list if len(st) > 0]

if len(active_spiketrains) > 0:
    ax, axhistx, axhisty = rasterplot_rates(spike_train_list, filter_function=lambda st: len(st) > 0)

    # Overlay scaled input current
    ax.plot(
        input_current__AnalogSignal.times,
        input_current__AnalogSignal.magnitude.T[0]
        / input_current__AnalogSignal.magnitude.T[0].max()
        * len(active_spiketrains),
        color="black",
        linewidth=2,
        label="Input Current (scaled)",
    )

    axhisty.set_xlabel("FR (pps)")

    # Add smoothed population firing rate over time (Gaussian, sigma = 15 ms).
    # Native replacement for elephant.statistics.instantaneous_rate.
    axhistx.clear()

    sampling_period_s = timestep.rescale(pq.s).magnitude
    t_start = min(st.t_start for st in active_spiketrains).rescale(pq.s).magnitude
    t_stop = max(st.t_stop for st in active_spiketrains).rescale(pq.s).magnitude
    n_bins = int(round((t_stop - t_start) / sampling_period_s))
    edges = t_start + np.arange(n_bins + 1) * sampling_period_s
    all_spikes = np.concatenate(
        [st.rescale(pq.s).magnitude for st in active_spiketrains]
    )
    all_spikes = all_spikes[(all_spikes >= edges[0]) & (all_spikes < edges[-1])]
    counts, _ = np.histogram(all_spikes, bins=edges)
    rate_hz = counts / sampling_period_s / len(active_spiketrains)
    rate_hz = gaussian_filter1d(rate_hz, sigma=(15e-3) / sampling_period_s, mode="constant")

    axhistx.plot(
        edges[:-1] + sampling_period_s / 2,
        rate_hz,
        linewidth=2,
        color="blue",
    )
    axhistx.set_ylabel("FR (pps)")
    axhistx.set_xlim(ax.get_xlim())

    ax.set_ylabel("Neuron Index (#)")
    ax.set_xlabel("Time (s)")


    sns.despine(ax=ax)

    fig = plt.gcf()
    fig.set_size_inches(12, 6)

    # Adjust positioning for better layout
    gap = 0.025
    bottom_margin = 0.03

    ax_pos = ax.get_position()
    axhistx_pos = axhistx.get_position()
    axhisty_pos = axhisty.get_position()

    ax.set_position([ax_pos.x0, ax_pos.y0 + bottom_margin, ax_pos.width, ax_pos.height])
    axhistx.set_position(
        [
            axhistx_pos.x0,
            axhistx_pos.y0 + gap + bottom_margin,
            axhistx_pos.width,
            axhistx_pos.height,
        ]
    )
    axhisty.set_position(
        [
            axhisty_pos.x0 + gap,
            axhisty_pos.y0 + bottom_margin,
            axhisty_pos.width,
            axhisty_pos.height,
        ]
    )

    plt.savefig(save_path / "spike_trains_jaxley_nerlab.png", dpi=150, bbox_inches="tight")
    plt.savefig(save_path / "spike_trains_jaxley_nerlab.svg", bbox_inches="tight")
    print(f"\nSaved figure to {save_path / 'spike_trains_jaxley_nerlab.png'} (and .svg)")
    plt.show()
else:
    print("\nNo active spike trains to plot.")
    print("This may indicate the current amplitude needs adjustment.")

##############################################################################
# Compare with NEURON Results (if available)
# ------------------------------------------

print("\n" + "=" * 60)
print("Comparison with NEURON Results")
print("=" * 60)

try:
    neuron_block = joblib.load(save_path / "spike_train__Block_utility.pkl")

    print("\nFound NEURON results for comparison:")
    for i, (neuron_seg, jaxley_seg) in enumerate(
        zip(neuron_block.segments, spike_train__Block.segments)
    ):
        neuron_spikes = sum(len(st) for st in neuron_seg.spiketrains)
        jaxley_spikes = sum(len(st) for st in jaxley_seg.spiketrains)

        neuron_active = sum(1 for st in neuron_seg.spiketrains if len(st) > 0)
        jaxley_active = sum(1 for st in jaxley_seg.spiketrains if len(st) > 0)

        print(f"\nPool {i}:")
        print(f"  NEURON:       {neuron_active} active neurons, {neuron_spikes} total spikes")
        print(f"  Jaxley:       {jaxley_active} active neurons, {jaxley_spikes} total spikes")

except FileNotFoundError:
    print("\nNEURON results not found.")
    print("Run 02_simulate_spike_trains_current_injection.py first for comparison.")

print("\n" + "=" * 60)
print("[DONE] Jaxley biophysical simulation complete!")
print("       Using jx.integrate() with NERLab channels (napp + caL)")
print("=" * 60)
