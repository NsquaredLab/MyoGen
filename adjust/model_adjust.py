"""
Spike Train Generation with Descending Drive
============================================

This example demonstrates **realistic spike train simulation** using **sinusoidal descending drive (DD)**
instead of direct current injection. This approach provides more physiologically accurate motor control
patterns by modeling cortical input through descending drive populations.

.. note::
    This example bridges the gap between simple current injection (example 01) and full spinal network
    simulation (network_config.py). It uses:

    - **DescendingDrive__Pool**: Poisson process neurons modeling cortical input
    - **AlphaMN__Pool**: Biophysically detailed motor neurons (Powers2017 model)
    - **Network**: Synaptic connections between DD and motor neuron populations
    - **Sinusoidal patterns**: Smooth, physiologically relevant input at 0.5-2 Hz

.. important::
    **Descending Drive (DD)** refers to the cortical and subcortical neural pathways that provide
    voluntary motor commands to spinal motor neurons. This is more realistic than direct current
    injection because it models the actual synaptic input patterns from upper motor neurons.
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

import itertools
from pathlib import Path

import elephant
import joblib
import numpy as np
import pandas as pd
import quantities as pq
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from neo import AnalogSignal, Block, Segment, SpikeTrain
from neuron import h
from tqdm import tqdm

from myogen import simulator
from myogen import RANDOM_GENERATOR, simulator
from myogen.simulator.core.force.force_model import ForceModel
from myogen.simulator.neuron import Network
from myogen.simulator.neuron.populations import AlphaMN__Pool, DescendingDrive__Pool
from myogen.utils.nmodl import load_nmodl_mechanisms, compile_nmodl_files

##############################################################################
# Define Parameters
# -----------------
# This example simulates a **motor pool** driven by **sinusoidal descending drive** patterns.
# The DD populations receive smooth sinusoidal input at physiologically relevant frequencies,
# which then drive the motor neuron pools through realistic synaptic connections.
#
# Key parameters:
#
# - ``n_dd_neurons``: Number of descending drive neurons per pool
# - ``dd_frequency__Hz``: Frequency of sinusoidal drive pattern
# - ``dd_amplitude__Hz``: Amplitude of drive modulation
# - ``dd_baseline__Hz``: Baseline drive level
# - ``timestep``: Simulation timestep in ms (high resolution)
# - ``simulation_time``: Total simulation duration in ms

# Connection parameters

##############################################################################
# Create Populations
# ------------------------
#
# Like the previous example, we create a **motor neuron pool** using the **AlphaMN__Pool** class.
#
# We also create a **DescendingDrive__Pool** to represent the cortical input.
#
# .. note:: These neurons are modeled as Poisson point processes to convert the smooth input signal into realistic
# spike patterns that represent cortical input to the spinal cord.
#
compile_nmodl_files()
load_nmodl_mechanisms()

save_path = Path("./results")
save_path.mkdir(exist_ok=True)


##############################################################################
# Generate Recruitment Thresholds
# --------------------------------
#
# Generate recruitment thresholds using the combined model from example 0.
# This creates physiologically realistic recruitment patterns.

n_motor_units = 100  # Number of motor units in the pool (increased for better comparison)
recruitment_range = 100  # Recruitment range (max_threshold / min_threshold)
combined_max_threshold = 1.0  # Maximum threshold for combined model

# Generate thresholds using combined model with slope=5 (same as example 0)
recruitment_thresholds, _ = simulator.RecruitmentThresholds(
    N=n_motor_units,
    recruitment_range__ratio=recruitment_range,
    deluca__slope=5,
    konstantin__max_threshold__ratio=combined_max_threshold,
    mode="combined",
)

# Save the generated thresholds
joblib.dump(recruitment_thresholds, save_path / "thresholds.pkl")

motor_neuron_pool = AlphaMN__Pool(recruitment_thresholds__array=recruitment_thresholds, config_file="alpha_mn_VLVM.yaml")

timestep = 0.1  # ms
descending_drive_pool = DescendingDrive__Pool(
    n=400, poisson_batch_size=16, timestep__ms=timestep
)

##############################################################################
# Generate Drive Pattern
# ----------------------------------
#
# The descending drive neuron population needs a time-varying input pattern to drive their Poisson processes.
# This
# Create a **smooth sinusoidal drive pattern** that represents realistic cortical motor commands.
# This pattern combines:
# - **Baseline activity**: Continuous low-level drive
# - **Sinusoidal modulation**: Smooth oscillation at physiological frequency
# - **Noise**: Small random variations for realism

simulation_time = 3000  # ms

time_points = int(simulation_time / timestep)

dd_frequency__Hz = 0.25
dd_amplitude__Hz = 60.0
dd_baseline__Hz = 20.0

sinusoidal_drive = AnalogSignal(
    signal=(
        np.maximum(
            dd_baseline__Hz
            + (
                0*(dd_amplitude__Hz - dd_baseline__Hz)
                * np.sin(
                    2
                    * np.pi
                    * dd_frequency__Hz
                    * np.linspace(0, simulation_time, time_points)
                    / 1000.0
                )
            ),
            dd_baseline__Hz,
        )
        + np.clip(RANDOM_GENERATOR.normal(0, 1.0, size=time_points), 0, None)
    ),
    units=pq.Hz,
    sampling_period=(timestep * pq.ms).rescale(pq.s),
)

joblib.dump(sinusoidal_drive, save_path / "sinusoidal_drive_pattern.pkl")

##############################################################################
# Create Network and Connections
# -------------------------------
#
# In MyoGen, populations can be connected using the **Network** class from the
# `myogen.simulator.neuron` module.
#
# The **Network** class provides a high-level interface for creating and managing
# connections between neuron populations.

# Use the **Network** class to create synaptic connections between the descending drive
# population and the motor neuron pool. This creates realistic synaptic transmission
# with appropriate delays and weights.
#

network = Network({"DD": descending_drive_pool, "aMN": motor_neuron_pool})

# Connect DD neurons to motor neurons with realistic synaptic parameters
network.connect(source="DD", target="aMN", probability=0.5, weight__μS=0.1)

# Set up external input to DD population
network.connect_from_external(source="cortical_input", target="DD", weight__μS=1.0)

# Get NetCons for manual DD stimulation
dd_netcons = network.get_netcons("cortical_input", "DD")

##############################################################################
# Setup Spike Recording
# ---------------------
#
# To record spikes, we need to manually set up spike detection for the motor neurons
# and track spike times for the DD neurons.
#

# Manual spike tracking for DD neurons (they use Poisson processes)
dd_spike_times = [[] for _ in range(len(descending_drive_pool))]

# Record spikes from motor neurons
mn_spike_recorders = []
for cell in motor_neuron_pool:
    spike_recorder = h.Vector()
    nc = h.NetCon(cell.soma(0.5)._ref_v, None, sec=cell.soma)
    nc.threshold = 50  # Standard threshold for motor neurons
    nc.record(spike_recorder)
    mn_spike_recorders.append(spike_recorder)

##############################################################################
# Run Simulation
# ------------------------------------
#
# Execute the NEURON simulation with real-time injection of the sinusoidal drive pattern.
# The DD neurons receive time-varying input that drives their Poisson processes.

h.load_file("stdrun.hoc")  # Load standard run library for NEURON
h.dt = timestep
h.tstop = simulation_time

# Initialize voltages for all pools
for section, voltage in itertools.chain.from_iterable(
    zip(*pool.get_initialization_data())
    for pool in [motor_neuron_pool, descending_drive_pool]
):
    section.v = voltage


h.finitialize()

# Calculate total simulation steps for progress bar
total_steps = int(simulation_time / timestep)

step_counter = 0
with tqdm(
    total=simulation_time,
    desc="Running simulation",
    unit="ms",
    bar_format="{l_bar}{bar}| {n:.2f}/{total:.2f} ms [{elapsed}<{remaining}, {rate_fmt}]",
) as pbar:
    while h.t < h.tstop:
        current_drive = sinusoidal_drive[min(step_counter, len(sinusoidal_drive) - 1)]

        # Drive DD neurons with current input level
        for dd_cell in descending_drive_pool:
            if dd_cell.integrate(current_drive):
                # Record spike time for DD neuron
                dd_spike_times[dd_cell.pool__ID].append(h.t)
                # Generate spike in DD neuron
                spike_time = h.t + np.clip(RANDOM_GENERATOR.normal(0, 10), 0, None)
                if spike_time < h.tstop:  # Avoid scheduling beyond simulation end
                    dd_netcons[dd_cell.pool__ID].event(spike_time)

        # Progress simulation
        h.fadvance()
        step_counter += 1
        pbar.update(timestep)


##############################################################################
# Convert Spike Data to Neo Format
# ---------------------------------
#

spike_train_block = Block(name="Sinusoidal DD Spike Trains")

dd_segment = Segment(name="Descending Drive")
dd_segment.spiketrains = [
    SpikeTrain(
        spike_times * pq.ms,
        t_stop=simulation_time * pq.ms,
        sampling_rate=(1 / h.dt * (1 / pq.ms)),
        sampling_period=h.dt * pq.ms,
        name=f"DD_{i}",
    )
    for i, spike_times in enumerate(dd_spike_times)
]

mn_segment = Segment(name="Motor Neurons")
mn_segment.spiketrains = [
    SpikeTrain(
        recorder.as_numpy() * pq.ms,
        t_stop=simulation_time * pq.ms,
        sampling_rate=(1 / h.dt * (1 / pq.ms)),
        sampling_period=h.dt * pq.ms,
        name=f"MN_{i}",
    )
    for i, recorder in enumerate(mn_spike_recorders)
]

# We only save the motor neuron spikes  segment
spike_train_block.segments.append(mn_segment)

joblib.dump(spike_train_block, save_path / "sinusoidal_dd_spike_trains.pkl")

##############################################################################
# Generate Force Output
# ---------------------
#
# Convert spike trains to force using the ForceModel.
# This simulates individual motor unit twitches and their temporal summation.

# Force model parameters
recording_frequency__Hz = 2048  # 2048 Hz sampling rate
longest_duration_rise_time__ms = 90.0  # Maximum twitch rise time
contraction_time_range = 3  # Contraction time range factor

# Create force model
force_model = ForceModel(
    recruitment_thresholds=recruitment_thresholds,
    recording_frequency__Hz=recording_frequency__Hz,
    longest_duration_rise_time__ms=longest_duration_rise_time__ms,
    contraction_time_range__unitless=contraction_time_range,
)

print("\nForce model statistics:")
print(f"  Number of motor units: {force_model._number_of_neurons}")
print(f"  Recruitment ratio: {force_model._recruitment_ratio:.1f}")
print(
    f"  Peak force range: {force_model.peak_twitch_forces__unitless[0]:.3f} - {force_model.peak_twitch_forces__unitless[-1]:.3f}"
)
print(
    f"  Contraction time range: {force_model.contraction_times__samples[0]:.1f} - {force_model.contraction_times__samples[-1]:.1f} samples"
)

# Generate force from spike trains
force_output = force_model.generate_force(spike_train__Block=spike_train_block)

# Add realistic noise to the force signal
noise_level = 0.015  # 1.5% of mean force
noisy_force = force_output.magnitude[:, 0] + RANDOM_GENERATOR.normal(
    0, noise_level * np.mean(force_output.magnitude[:, 0]), size=len(force_output.magnitude[:, 0])
)

# Save force output
joblib.dump(force_output, save_path / "force_output.pkl")

print(f"\nForce output statistics:")
print(f"  Peak force: {np.max(force_output.magnitude[:, 0]):.3f} a.u.")
print(f"  Mean force: {np.mean(force_output.magnitude[:, 0]):.3f} a.u.")
print(f"  Force duration: {force_output.times[-1]:.2f}")

##############################################################################
# Calculate Firing Rate Statistics
# ---------------------------------
#

print("\nFiring rate analysis:")

# Calculate DD firing rates
dd_firing_rates = np.array(
    [
        elephant.statistics.mean_firing_rate(st__s.time_slice(st__s.min(), st__s.max()))
        for st__ms in dd_segment.spiketrains
        if len(st__s := st__ms.rescale(pq.s)) > 0
    ]
)

# Calculate MN firing rates
mn_firing_rates = np.array(
    [
        elephant.statistics.mean_firing_rate(st__s.time_slice(st__s.min(), st__s.max()))
        for st__ms in mn_segment.spiketrains
        if len(st__s := st__ms.rescale(pq.s)) > 0
    ]
)

print("Descending Drive neurons:")
print(f"\tActive neurons: {len(dd_firing_rates)}/{descending_drive_pool.n}")
if len(dd_firing_rates) > 0:
    print(
        f"\tMean firing rate: {np.mean(dd_firing_rates):.1f} ± {np.std(dd_firing_rates):.1f} Hz"
    )
    print(
        f"\tRate range: {np.min(dd_firing_rates):.1f} - {np.max(dd_firing_rates):.1f} Hz"
    )

print("Motor neurons:")
print(f"\tActive neurons: {len(mn_firing_rates)}/{motor_neuron_pool.n}")
if len(mn_firing_rates) > 0:
    print(
        f"\tMean firing rate: {np.mean(mn_firing_rates):.1f} ± {np.std(mn_firing_rates):.1f} Hz"
    )
    print(
        f"\tRate range: {np.min(mn_firing_rates):.1f} - {np.max(mn_firing_rates):.1f} Hz"
    )

##############################################################################
# Advanced Visualization
# -----------------------
#
# Create comprehensive visualizations showing:
# 1. Sinusoidal drive input pattern
# 2. DD population raster plot with drive overlay
# 3. Motor neuron raster plot showing recruitment
# 4. Population firing rates over time

# Create figure with subplots
fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

# 1. Plot sinusoidal drive pattern
time_s = sinusoidal_drive.times.rescale(pq.s).magnitude
axes[0].plot(time_s, sinusoidal_drive, "b-", linewidth=2, label="DD Input")
axes[0].axhline(dd_baseline__Hz, color="r", linestyle="--", alpha=0.7, label="Baseline")
axes[0].set_ylabel("Drive (Hz)")
axes[0].set_title(f"Sinusoidal Descending Drive Pattern ({dd_frequency__Hz} Hz)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2. DD population raster plot
dd_colors = plt.cm.get_cmap("Blues")(np.linspace(0.3, 0.8, len(dd_segment.spiketrains)))
for i, (spiketrain, color) in enumerate(zip(dd_segment.spiketrains, dd_colors)):
    if len(spiketrain) > 0:
        spike_times = spiketrain.rescale(pq.s).magnitude
        axes[1].scatter(
            spike_times, [i] * len(spike_times), c=[color], s=0.8, alpha=0.8
        )

axes[1].set_ylabel("DD Neuron ID")
axes[1].set_title(f"Descending Drive Population Activity (n={descending_drive_pool.n})")
axes[1].set_ylim(-1, descending_drive_pool.n)
axes[1].grid(True, alpha=0.3)

# 3. Motor neuron raster plot (recruitment ordered)
mn_colors = plt.cm.get_cmap("Reds")(np.linspace(0.3, 0.9, len(mn_segment.spiketrains)))
active_mn_count = 0
for i, (spiketrain, color) in enumerate(zip(mn_segment.spiketrains, mn_colors)):
    if len(spiketrain) > 0:
        spike_times = spiketrain.rescale(pq.s).magnitude
        axes[2].scatter(
            spike_times, [i] * len(spike_times), c=[color], s=1.0, alpha=0.8
        )
        active_mn_count += 1

axes[2].set_ylabel("Motor Neuron ID\n(Recruitment Order)")
axes[2].set_title(
    f"Motor Neuron Population Activity (n={active_mn_count}/{motor_neuron_pool.n} active)"
)
axes[2].set_ylim(-1, motor_neuron_pool.n)
axes[2].grid(True, alpha=0.3)

# 4. Population firing rates over time (binned)
bin_size_ms = 100

dd_psth = elephant.statistics.time_histogram(
    dd_segment.spiketrains, bin_size_ms * pq.ms
)
dd_rates_binned = (
    (dd_psth / (bin_size_ms * pq.ms) / descending_drive_pool.n).rescale(pq.Hz).magnitude
)

mn_psth = elephant.statistics.time_histogram(
    mn_segment.spiketrains, bin_size_ms * pq.ms
)
mn_rates_binned = (
    (mn_psth / (bin_size_ms * pq.ms) / motor_neuron_pool.n).rescale(pq.Hz).magnitude
)

bin_centers_s = dd_psth.times.rescale(pq.s).magnitude
axes[3].plot(
    bin_centers_s, dd_rates_binned, "b-", linewidth=2, label="DD Population", alpha=0.8
)
axes[3].plot(
    bin_centers_s, mn_rates_binned, "r-", linewidth=2, label="MN Population", alpha=0.8
)

axes[3].set_xlabel("Time (s)")
axes[3].set_ylabel("Population Rate (Hz)")
axes[3].set_title("Population Firing Rates Over Time")
axes[3].legend()
axes[3].grid(True, alpha=0.3)

# Format all axes
for ax in axes:
    ax.set_xlim(0, simulation_time / 1000.0)

plt.tight_layout()
plt.show()

##############################################################################
# Visualize Force Output
# ----------------------
#
# Create a comprehensive visualization of the force generation process,
# showing the relationship between descending drive, spike activity, and force.

fig_force, axes_force = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

# 1. Descending drive input
time_s = sinusoidal_drive.times.rescale(pq.s).magnitude
axes_force[0].plot(time_s, sinusoidal_drive, "b-", linewidth=2, label="DD Input")
axes_force[0].axhline(dd_baseline__Hz, color="r", linestyle="--", alpha=0.7, label="Baseline")
axes_force[0].set_ylabel("Drive (Hz)", fontsize=12)
axes_force[0].set_title("Descending Drive Input", fontsize=14, fontweight='bold')
axes_force[0].legend()
axes_force[0].grid(True, alpha=0.3)

# 2. Motor neuron raster plot
mn_colors = plt.cm.get_cmap("Reds")(np.linspace(0.3, 0.9, len(mn_segment.spiketrains)))
active_mn_count = 0
for i, (spiketrain, color) in enumerate(zip(mn_segment.spiketrains, mn_colors)):
    if len(spiketrain) > 0:
        spike_times = spiketrain.rescale(pq.s).magnitude
        axes_force[1].scatter(
            spike_times, [i] * len(spike_times), c=[color], s=1.0, alpha=0.8
        )
        active_mn_count += 1

axes_force[1].set_ylabel("Motor Neuron ID", fontsize=12)
axes_force[1].set_title(
    f"Motor Neuron Spike Trains (n={active_mn_count}/{motor_neuron_pool.n} active)",
    fontsize=14, fontweight='bold'
)
axes_force[1].set_ylim(-1, motor_neuron_pool.n)
axes_force[1].grid(True, alpha=0.3)

# 3. Clean force output
force_time_s = force_output.times.rescale(pq.s).magnitude
axes_force[2].plot(
    force_time_s,
    force_output[:, 0],
    "b-",
    linewidth=2,
    label="Clean Force"
)
axes_force[2].set_ylabel("Force (a.u.)", fontsize=12)
axes_force[2].set_title("Simulated Force Output (Clean)", fontsize=14, fontweight='bold')
axes_force[2].grid(True, alpha=0.3)
axes_force[2].legend()

# 4. Noisy force output (more realistic)
axes_force[3].plot(
    force_time_s,
    noisy_force,
    "r-",
    linewidth=1,
    alpha=0.8,
    label="Noisy Force"
)
axes_force[3].set_xlabel("Time (s)", fontsize=12)
axes_force[3].set_ylabel("Force (a.u.)", fontsize=12)
axes_force[3].set_title("Realistic Force Output (with noise)", fontsize=14, fontweight='bold')
axes_force[3].grid(True, alpha=0.3)
axes_force[3].legend()

# Format all axes
for ax in axes_force:
    ax.set_xlim(0, simulation_time / 1000.0)

plt.tight_layout()

# Save the figure
output_path_force = save_path / "force_output_visualization.png"
fig_force.savefig(str(output_path_force), dpi=300, bbox_inches='tight')
print(f"\nForce visualization saved to: {output_path_force}")

plt.show()


##############################################################################
# Plot ISI Statistics with Recruitment Thresholds
# ------------------------------------------------
#
# This function creates a comprehensive visualization combining:
# 1. Generated recruitment thresholds from the simulation
# 2. Experimental ISI statistics from ISI_statistics.csv
#
# The plot shows how the model's recruitment thresholds compare with
# real experimental data from different muscles and force levels.


def plot_isi_statistics_with_thresholds():
    """
    Plot ISI statistics from CSV file alongside generated recruitment thresholds.

    This function loads experimental ISI data and plots it together with the
    recruitment thresholds generated using the combined model.
    """
    # Load ISI statistics from CSV
    csv_path = Path(__file__).parent / "ISI_statistics.csv"
    isi_data = pd.read_csv(csv_path)

    # Calculate ISI statistics from simulated spike trains
    simulated_fr = []
    simulated_cv = []
    simulated_neuron_idx = []

    for idx, spiketrain in enumerate(mn_segment.spiketrains):
        if len(spiketrain) > 1:  # Need at least 2 spikes to calculate ISI
            # Calculate ISIs (inter-spike intervals)
            spike_times_s = spiketrain.rescale(pq.s).magnitude
            isis = np.diff(spike_times_s)

            if len(isis) > 0:
                # Calculate firing rate (1 / mean ISI)
                mean_isi = np.mean(isis)
                fr = 1.0 / mean_isi if mean_isi > 0 else 0

                # Calculate coefficient of variation (CV = std / mean)
                cv = np.std(isis) / mean_isi if mean_isi > 0 else 0

                # Only include neurons with meaningful firing rates
                if fr >= 0.01:
                    simulated_fr.append(fr)
                    simulated_cv.append(cv)
                    simulated_neuron_idx.append(idx)

    simulated_fr = np.array(simulated_fr)
    simulated_cv = np.array(simulated_cv)
    simulated_neuron_idx = np.array(simulated_neuron_idx)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Subplot 1: Recruitment Thresholds
    ax1 = axes[0, 0]
    ax1.plot(
        range(1, len(recruitment_thresholds) + 1),
        recruitment_thresholds,
        'o-',
        linewidth=2,
        markersize=8,
        color='#2E86AB',
        label='Generated Thresholds'
    )
    ax1.set_xlabel('Motor Unit Index', fontsize=12)
    ax1.set_ylabel('Recruitment Threshold', fontsize=12)
    ax1.set_title('Generated Recruitment Thresholds (Combined Model)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # Subplot 2: Mean Firing Rate by Muscle and Force Level
    ax2 = axes[0, 1]
    muscles = isi_data['Muscle'].unique()
    colors = {'VM': '#E63946', 'VL': '#F77F00', 'TA': '#06A77D', 'FDI': '#4361EE'}

    for muscle in muscles:
        muscle_data = isi_data[isi_data['Muscle'] == muscle]
        force_levels = muscle_data['Force Level'].unique()

        for force in sorted(force_levels):
            force_data = muscle_data[muscle_data['Force Level'] == force]
            mean_fr = force_data['FR mean'].mean()
            std_fr = force_data['FR mean'].std()

            ax2.errorbar(
                force, mean_fr, yerr=std_fr,
                marker='o', markersize=8,
                color=colors.get(muscle, '#000000'),
                label=f'{muscle} - {force}%' if force == sorted(force_levels)[0] else '',
                alpha=0.7
            )

    ax2.set_xlabel('Force Level (%)', fontsize=12)
    ax2.set_ylabel('Mean Firing Rate (Hz)', fontsize=12)
    ax2.set_title('Experimental Firing Rates by Muscle and Force', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9, loc='best')

    # Subplot 3: ISI Coefficient of Variation (CV) Distribution
    ax3 = axes[1, 0]

    for muscle in muscles:
        muscle_data = isi_data[isi_data['Muscle'] == muscle]
        ax3.hist(
            muscle_data['ISI CV'],
            bins=20,
            alpha=0.5,
            color=colors.get(muscle, '#000000'),
            label=muscle,
            edgecolor='black'
        )

    ax3.set_xlabel('ISI Coefficient of Variation', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('ISI Variability Distribution by Muscle', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend(fontsize=10)

    # Subplot 4: CV vs Firing Rate (Simulated + Experimental) - Styled like fr_cv function
    ax4 = axes[1, 1]

    # Plot simulated data with colormap based on neuron index (recruitment order)
    if len(simulated_fr) > 0:
        scatter_sim = ax4.scatter(
            simulated_cv,
            simulated_fr,
            c=simulated_neuron_idx,
            cmap='Blues',
            s=80,
            alpha=0.7,
            vmin=0,
            vmax=len(mn_segment.spiketrains),
            edgecolors='black',
            linewidth=0.5,
            label='Simulated MN'
        )

        # Add colorbar for simulated data
        cax = inset_axes(ax4, width="3%", height="30%", loc='upper right',
                        bbox_to_anchor=(0.15, 0, 1, 1), bbox_transform=ax4.transAxes)
        cbar = plt.colorbar(scatter_sim, cax=cax, label='MN Index')
        cbar.ax.tick_params(labelsize=8)

    # Plot experimental data from CSV
    for muscle in muscles:
        muscle_data = isi_data[isi_data['Muscle'] == muscle]
        ax4.scatter(
            muscle_data['ISI CV'],
            muscle_data['FR mean'],
            s=60,
            alpha=0.5,
            color=colors.get(muscle, '#000000'),
            label=f'{muscle} (Exp)',
            edgecolors='black',
            linewidth=0.3,
            marker='x'
        )

    ax4.set_xlabel('CV (Coefficient of Variation)', fontsize=12)
    ax4.set_ylabel('Firing Rate (Hz)', fontsize=12)
    ax4.set_title('CV vs Firing Rate (Simulated + Experimental)', fontsize=14, fontweight='bold')
    ax4.grid(True, linestyle='--', alpha=0.7)
    ax4.legend(fontsize=8, loc='upper left')

    # Add overall title
    fig.suptitle(
        'Recruitment Thresholds and ISI Statistics Analysis',
        fontsize=16,
        fontweight='bold',
        y=0.995
    )

    plt.tight_layout()

    # Save the figure
    output_path = save_path / "isi_statistics_with_thresholds.png"
    fig.savefig(str(output_path), dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    plt.show()


# Call the plotting function
plot_isi_statistics_with_thresholds()


##############################################################################
# FR vs CV Plot (Standalone - Matching Original Style)
# -----------------------------------------------------
#
# This creates a standalone plot matching the exact style of the fr_cv function
# with inset zoom plots for detailed regions.


def plot_fr_cv_standalone():
    """
    Create a standalone FR vs CV plot matching the original fr_cv function style.

    This plot includes:
    - Main scatter plot with CV on x-axis, FR on y-axis
    - Color-coded by neuron index (recruitment order)
    - Two inset zoom plots for detailed regions
    - Experimental data overlay
    """
    # Load ISI statistics from CSV
    csv_path = Path(__file__).parent / "ISI_statistics.csv"
    isi_data = pd.read_csv(csv_path)

    # Calculate ISI statistics from simulated spike trains
    simulated_fr = []
    simulated_cv = []
    simulated_neuron_idx = []

    for idx, spiketrain in enumerate(mn_segment.spiketrains):
        if len(spiketrain) > 1:
            spike_times_s = spiketrain.rescale(pq.s).magnitude
            isis = np.diff(spike_times_s)

            if len(isis) > 0:
                mean_isi = np.mean(isis)
                fr = 1.0 / mean_isi if mean_isi > 0 else 0
                cv = np.std(isis) / mean_isi if mean_isi > 0 else 0

                if fr >= 0.01:
                    simulated_fr.append(fr)
                    simulated_cv.append(cv)
                    simulated_neuron_idx.append(idx)

    simulated_fr = np.array(simulated_fr)
    simulated_cv = np.array(simulated_cv)
    simulated_neuron_idx = np.array(simulated_neuron_idx)

    # Create main figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot simulated data with colormap (Blues like in original)
    if len(simulated_fr) > 0:
        scatter_sim = ax.scatter(
            simulated_cv,
            simulated_fr,
            c=simulated_neuron_idx,
            cmap='Blues',
            s=50,
            alpha=0.7,
            vmin=0,
            vmax=len(mn_segment.spiketrains),
            edgecolors='black',
            linewidth=0.3,
            label='Simulated'
        )

    # Plot experimental data - Include VL, VM, TA, and FDI muscles
    exp_colors = {'VL': 'orange', 'VM': 'red', 'TA': 'green', 'FDI': 'magenta'}
    exp_muscles = ['VL', 'VM', 'TA', 'FDI']

    for muscle in exp_muscles:
        data_muscle = isi_data.query(f'Muscle == "{muscle}"')
        ax.scatter(
            data_muscle['ISI CV'],
            data_muscle['FR mean'],
            color=exp_colors[muscle],
            s=30,
            alpha=0.5,
            marker='x',
            label=f'Experimental ({muscle})'
        )

    ax.set_xlabel('CV', fontsize=12)
    ax.set_ylabel('Firing rate mean (pps)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)

    # Create legend with both simulated and experimental data
    ax.legend(loc='upper left', fontsize=10)

    # Print statistics to verify data is present
    print(f"\nFR vs CV Plot Statistics:")
    print(f"  Simulated neurons: {len(simulated_fr)}")
    if len(simulated_fr) > 0:
        print(f"  Simulated FR range: {simulated_fr.min():.2f} - {simulated_fr.max():.2f} Hz")
        print(f"  Simulated CV range: {simulated_cv.min():.3f} - {simulated_cv.max():.3f}")

    print(f"\n  Experimental data:")
    for muscle in exp_muscles:
        data_muscle_stats = isi_data.query(f'Muscle == "{muscle}"')
        if len(data_muscle_stats) > 0:
            print(f"    {muscle}: {len(data_muscle_stats)} points")
            print(f"      FR range: {data_muscle_stats['FR mean'].min():.2f} - {data_muscle_stats['FR mean'].max():.2f} Hz")
            print(f"      CV range: {data_muscle_stats['ISI CV'].min():.3f} - {data_muscle_stats['ISI CV'].max():.3f}")

    
    plt.tight_layout()

    # Save the figure
    output_path = save_path / "fr_cv_scatter_full.png"
    fig.savefig(str(output_path), dpi=300, bbox_inches='tight')
    print(f"\nFR vs CV plot saved to: {output_path}")

    plt.show()


# Call the standalone FR vs CV plotting function
plot_fr_cv_standalone()
