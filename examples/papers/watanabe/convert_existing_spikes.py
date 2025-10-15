"""
Convert Existing Spike Data to Chunks Format
==============================================

This script extracts spike data from the existing SimulationRunner results
and saves it into the chunks directory, making the chunks self-contained.

Use this if you already ran the simulation before the chunk format was updated.
"""

from pathlib import Path
import joblib
import numpy as np

# Paths
chunks_path = Path("./results/watanabe_chunks")
spike_results_path = Path("./results/watanabe__spikes_only.pkl")

print("="*70)
print("CONVERTING EXISTING SPIKE DATA TO CHUNK FORMAT")
print("="*70)

# Load the spike results from SimulationRunner
print(f"\nLoading spike data from: {spike_results_path}")
spike_results = joblib.load(spike_results_path)

# Extract spike data from NEO Block
print(f"Extracting spike data from NEO Block...")
spike_data_arrays = {}

for seg in spike_results.segments:
    if len(seg.spiketrains) > 0:
        pop_name = seg.name
        times_list = []
        ids_list = []

        for st in seg.spiketrains:
            neuron_id = int(st.name)
            spike_times = st.times.rescale('ms').magnitude
            times_list.extend(spike_times)
            ids_list.extend([neuron_id] * len(spike_times))

        spike_data_arrays[pop_name] = {
            "times": np.array(times_list),
            "ids": np.array(ids_list),
        }

        print(f"  {pop_name}: {len(times_list)} spikes from {len(seg.spiketrains)} neurons")

# Save to chunks directory
spike_filename = chunks_path / "spikes.pkl"
print(f"\nSaving spike data to: {spike_filename}")
joblib.dump(spike_data_arrays, spike_filename, compress=3)

print(f"\n✓ Conversion complete!")
print(f"  Spike data saved to: {spike_filename}")
print(f"  Populations: {list(spike_data_arrays.keys())}")
print(f"\nNow you can load everything from chunks:")
print(f"  from continuous_saver import convert_chunks_to_neo")
print(f"  results = convert_chunks_to_neo(Path('{chunks_path}'))")
print("="*70)
