"""
Load and Analyze Watanabe Simulation Results
=============================================

This script demonstrates loading the chunked simulation data as a NEO Block,
which is compatible with all existing analysis code that expects SimulationRunner output.
"""

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np

from continuous_saver import convert_chunks_to_neo

##############################################################################
# Load Data as NEO Block
# ----------------------
#
# This is the RECOMMENDED approach - converts chunks to a NEO Block that's
# identical in structure to what SimulationRunner.run() returns

chunks_path = Path("./results/watanabe_chunks")

print("=" * 70)
print("LOADING SIMULATION DATA AS NEO BLOCK")
print("=" * 70)
print(f"Chunks directory: {chunks_path}\n")

# Convert chunks to NEO Block format (compatible with SimulationRunner output)
# All data (spikes + membrane potentials) is self-contained in the chunks
results = convert_chunks_to_neo(chunks_path)

print("\n✓ NEO Block loaded successfully!")
print(f"  Type: {type(results)}")
print(f"  Segments: {len(results.segments)}")

for seg in results.segments:
    print(
        f"    - {seg.name}: {len(seg.spiketrains)} spike trains, "
        f"{len(seg.analogsignals)} analog signals"
    )

##############################################################################
# Analyze Membrane Potentials
# ---------------------------
#
# Access membrane potentials from NEO AnalogSignals

print("\n" + "=" * 70)
print("ANALYZING MEMBRANE POTENTIALS")
print("=" * 70)

# Find the aMN segment
aMN_segment = None
for seg in results.segments:
    if seg.name == "aMN":
        aMN_segment = seg
        break

if aMN_segment and len(aMN_segment.analogsignals) > 0:
    # Extract analog signals
    analog_signals = aMN_segment.analogsignals
    n_neurons = len(analog_signals)

    # Get time vector from first signal
    first_signal = analog_signals[0]
    times_ms = first_signal.times.rescale("ms").magnitude
    sampling_period = float(first_signal.sampling_period.rescale("ms"))

    print(f"\nMembrane potential recording:")
    print(f"  Population: aMN (alpha motor neurons)")
    print(f"  Recorded neurons: {n_neurons}")
    print(f"  Time range: {times_ms[0]:.1f} - {times_ms[-1]:.1f} ms")
    print(f"  Duration: {(times_ms[-1] - times_ms[0]) / 1000:.1f} seconds")
    print(f"  Sampling period: {sampling_period} ms")
    print(f"  Samples per neuron: {len(times_ms)}")

    # Plot membrane potentials for early, middle, and late recruited neurons
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Select neurons to plot
    indices_to_plot = [0, n_neurons // 2, n_neurons - 1]

    for ax, idx in zip(axes, indices_to_plot):
        signal = analog_signals[idx]
        neuron_id = signal.name  # Actual neuron ID
        voltage = signal.rescale("mV").magnitude.flatten()

        ax.plot(times_ms / 1000.0, voltage, linewidth=0.5)
        ax.set_ylabel(f"aMN[{neuron_id}]\nVoltage (mV)")
        ax.set_title(f"Motor Neuron {neuron_id} Membrane Potential")
        ax.grid(True, alpha=0.3)

        # Highlight the three experimental phases
        ax.axvspan(0, 60, alpha=0.1, color="gray", label="Phase 1: Constant")
        ax.axvspan(60, 120, alpha=0.1, color="blue", label="Phase 2: Sinusoid DC=65")
        ax.axvspan(120, 180, alpha=0.1, color="red", label="Phase 3: Sinusoid DC=58")

        if ax == axes[0]:
            ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(chunks_path.parent / "watanabe_membrane_potentials.png", dpi=150)
    print(f"\n✓ Saved plot: {chunks_path.parent / 'watanabe_membrane_potentials.png'}")
    plt.show()

##############################################################################
# Analyze Spike Data
# ------------------
#
# Access spike trains from NEO SpikeTrain objects

print("\n" + "=" * 70)
print("ANALYZING SPIKE DATA")
print("=" * 70)

# Print summary for all populations
for seg in results.segments:
    if len(seg.spiketrains) > 0:
        n_units = len(seg.spiketrains)
        n_spikes = sum(len(st) for st in seg.spiketrains)
        duration_s = float(seg.spiketrains[0].t_stop.rescale("s"))

        print(f"\n  {seg.name}:")
        print(f"    Total units: {n_units}")
        print(f"    Total spikes: {n_spikes}")
        if duration_s > 0 and n_units > 0:
            print(f"    Mean firing rate: {n_spikes / n_units / duration_s:.2f} Hz")

# Plot raster plot for aMN population
if aMN_segment and len(aMN_segment.spiketrains) > 0:
    print(f"\nGenerating spike raster plot...")

    fig, ax = plt.subplots(figsize=(14, 6))

    # Extract spike times and IDs from NEO spike trains
    for st in aMN_segment.spiketrains:
        neuron_id = int(st.name)
        spike_times_s = st.times.rescale("s").magnitude
        if len(spike_times_s) > 0:
            ax.scatter(
                spike_times_s,
                [neuron_id] * len(spike_times_s),
                s=0.5,
                c="black",
                alpha=0.5,
            )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Motor Neuron ID")
    ax.set_title("Alpha Motor Neuron Spike Raster")
    ax.grid(True, alpha=0.3)

    # Highlight the three experimental phases
    ax.axvspan(0, 60, alpha=0.1, color="gray", label="Phase 1: Constant")
    ax.axvspan(60, 120, alpha=0.1, color="blue", label="Phase 2: Sinusoid DC=65")
    ax.axvspan(120, 180, alpha=0.1, color="red", label="Phase 3: Sinusoid DC=58")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(chunks_path.parent / "watanabe_spike_raster.png", dpi=150)
    print(f"✓ Saved plot: {chunks_path.parent / 'watanabe_spike_raster.png'}")
    plt.show()

    # Plot population firing rate over time
    print(f"\nGenerating population firing rate plot...")

    fig, ax = plt.subplots(figsize=(14, 4))

    # Collect all spike times
    all_spike_times = []
    for st in aMN_segment.spiketrains:
        all_spike_times.extend(st.times.rescale("ms").magnitude)
    all_spike_times = np.array(all_spike_times)

    # Calculate population firing rate in 100 ms bins
    bin_size_ms = 100
    duration_ms = float(aMN_segment.spiketrains[0].t_stop.rescale("ms"))
    bins = np.arange(0, duration_ms + bin_size_ms, bin_size_ms)
    counts, _ = np.histogram(all_spike_times, bins=bins)

    # Convert to firing rate (spikes/sec/neuron)
    n_neurons = len(aMN_segment.spiketrains)
    firing_rate = counts / (bin_size_ms / 1000.0) / n_neurons
    bin_centers = (bins[:-1] + bins[1:]) / 2

    ax.plot(bin_centers / 1000.0, firing_rate, linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Population Firing Rate (Hz)")
    ax.set_title(f"Alpha Motor Neuron Population Firing Rate ({bin_size_ms} ms bins)")
    ax.grid(True, alpha=0.3)

    # Highlight the three experimental phases
    ax.axvspan(0, 60, alpha=0.1, color="gray", label="Phase 1")
    ax.axvspan(60, 120, alpha=0.1, color="blue", label="Phase 2")
    ax.axvspan(120, 180, alpha=0.1, color="red", label="Phase 3")
    ax.legend()

    plt.tight_layout()
    plt.savefig(chunks_path.parent / "watanabe_firing_rate.png", dpi=150)
    print(f"✓ Saved plot: {chunks_path.parent / 'watanabe_firing_rate.png'}")
    plt.show()

##############################################################################
# Save NEO Block for Future Use (OPTIONAL)
# ----------------------------------------

# NOTE: Saving the full NEO Block is slow (~23 GB of data) and unnecessary
# since we can quickly regenerate it from chunks using convert_chunks_to_neo()
#
# If you really want to save it, uncomment below and use compress=0 for speed:
#
print(f"\nSaving NEO Block for future use...")
import joblib

neo_output_path = chunks_path.parent / "watanabe_results_neo.pkl"
joblib.dump(results, neo_output_path, compress=0)  # No compression = faster
print(f"✓ NEO Block saved to: {neo_output_path}")

print(f"\nNOTE: NEO Block NOT saved (regenerate from chunks when needed)")
print(f"To reload later, simply run:")
print(f"  results = convert_chunks_to_neo(")
print(f"      Path('{chunks_path}'),")
print(
    f"      spike_data_file=Path('{chunks_path.parent / 'watanabe__spikes_only.pkl'}')"
)
print(f"  )")

##############################################################################
# Summary
# -------

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"NEO Block loaded from: {chunks_path}")
if neo_output_path:
    print(f"NEO Block saved to: {neo_output_path}")
else:
    print(f"NEO Block not saved (regenerate from chunks as needed)")
print(f"Plots saved to: {chunks_path.parent}")
print("\n✓ The NEO Block 'results' can be used with ANY existing analysis code")
print("  that expects SimulationRunner output!")
print("\nExample usage:")
print("  # Access spike trains")
print("  for seg in results.segments:")
print("      if seg.name == 'aMN':")
print("          for st in seg.spiketrains:")
print("              print(f'Neuron {st.name}: {len(st)} spikes')")
print("\n  # Access membrane potentials")
print("  for seg in results.segments:")
print("      if seg.name == 'aMN':")
print("          for signal in seg.analogsignals:")
print("              voltage = signal.magnitude  # NEO Quantity")
print("              times = signal.times        # NEO Quantity")
print("=" * 70)
