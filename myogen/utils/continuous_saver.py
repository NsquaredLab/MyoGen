"""
Continuous Saving Utilities for Long Simulations
=================================================

Provides memory-efficient continuous data saving for very long NEURON simulations
by periodically flushing data chunks to disk instead of keeping everything in RAM.
"""

from collections import defaultdict
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

# NEO imports for standard output format
import quantities as pq
from neo import AnalogSignal, Block, Segment, SpikeTrain
from neuron import h
from tqdm import tqdm

from myogen.utils.types import Quantity__ms


class ContinuousSaver:
    """
    Manages continuous saving of simulation data in chunks to prevent memory overflow.

    Instead of accumulating all data in RAM, this class periodically saves chunks
    to disk and clears memory. Data can be loaded and combined afterward.

    Parameters
    ----------
    save_path : Path
        Directory where chunks will be saved
    chunk_duration__ms : float
        Duration of each chunk in milliseconds (default: 10000 ms = 10 seconds)
    populations : dict
        Dictionary of populations to record from
    recording_config : dict
        Configuration like {"aMN": [0, 10, 20, ...]} for which cells to record
    """

    def __init__(
        self,
        save_path: Path,
        chunk_duration__ms: Quantity__ms = 10000.0 * pq.ms,
        populations: Optional[dict] = None,
        recording_config: Optional[dict] = None,
    ):
        self.save_path = Path(save_path)
        self.save_path.mkdir(exist_ok=True, parents=True)
        self.chunk_duration__ms = chunk_duration__ms
        self.populations = populations or {}
        self.recording_config = recording_config or {}

        # Tracking state
        self.chunk_id = 0
        self.last_save_time = 0.0
        self.current_chunk_data = defaultdict(lambda: defaultdict(list))
        self.current_chunk_times = []

        # Spike recording
        self.spike_data = defaultdict(lambda: {"times": [], "ids": []})

        print("ContinuousSaver initialized:")
        print(f"\tSave path: {self.save_path}")
        print(f"\tChunk duration: {chunk_duration__ms} ms")
        print(f"\tRecording config: {recording_config}")

    def record_step(self, timestep__ms: float) -> None:
        """
        Record data for current simulation timestep.

        Call this from your step callback at each timestep.

        Parameters
        ----------
        timestep__ms : float
            Integration timestep in milliseconds
        """
        current_time = h.t

        # Record current time
        self.current_chunk_times.append(current_time)

        # Record membrane potentials for specified cells
        for pop_name, cell_indices in self.recording_config.items():
            if pop_name not in self.populations:
                continue

            population = self.populations[pop_name]

            for cell_idx in cell_indices:
                if cell_idx < len(population):
                    # Read voltage from soma
                    voltage = population[cell_idx].soma(0.5).v
                    self.current_chunk_data[pop_name][cell_idx].append(voltage)

        # Check if it's time to save this chunk
        if current_time - self.last_save_time >= self.chunk_duration__ms:
            self._save_current_chunk(timestep__ms)

    def record_spike(self, pop_name: str, cell_id: int, spike_time: float) -> None:
        """
        Record a spike event.

        Parameters
        ----------
        pop_name : str
            Population name
        cell_id : int
            Cell ID within population
        spike_time : float
            Time of spike in milliseconds
        """
        self.spike_data[pop_name]["times"].append(spike_time)
        self.spike_data[pop_name]["ids"].append(cell_id)

    def _save_current_chunk(self, timestep__ms: float) -> None:
        """Save current chunk to disk and clear memory."""
        if len(self.current_chunk_times) == 0:
            return  # Nothing to save

        chunk_data = {
            "chunk_id": self.chunk_id,
            "time_start": self.current_chunk_times[0],
            "time_end": self.current_chunk_times[-1],
            "times": np.array(self.current_chunk_times),
            "timestep__ms": timestep__ms,
            "membrane_data": {},
        }

        # Convert lists to numpy arrays for efficient storage
        for pop_name, cells_dict in self.current_chunk_data.items():
            chunk_data["membrane_data"][pop_name] = {}
            for cell_idx, voltages in cells_dict.items():
                chunk_data["membrane_data"][pop_name][cell_idx] = np.array(voltages)

        # Save to disk
        chunk_filename = self.save_path / f"chunk_{self.chunk_id:04d}.pkl"
        joblib.dump(chunk_data, chunk_filename, compress=3)

        # Calculate chunk size for logging
        n_timepoints = len(self.current_chunk_times)
        n_neurons = sum(len(cells) for cells in self.current_chunk_data.values())
        chunk_size_mb = (n_timepoints * n_neurons * 8) / (1024**2)  # 8 bytes per float

        print(
            f"Saved chunk {self.chunk_id}: {self.current_chunk_times[0]:.1f}-{self.current_chunk_times[-1]:.1f} ms "
            f"({n_timepoints} steps, {n_neurons} neurons, ~{chunk_size_mb:.1f} MB)"
        )

        # Clear memory
        self.current_chunk_data.clear()
        self.current_chunk_times.clear()
        self.chunk_id += 1
        self.last_save_time = h.t

    def finalize(self, timestep__ms: Quantity__ms, spike_results=None) -> None:
        """
        Save final chunk and spike data.

        Call this after simulation completes.

        Parameters
        ----------
        timestep__ms : Quantity__ms
            Integration timestep in milliseconds
        spike_results : NEO Block, optional
            NEO Block containing spike trains from SimulationRunner.
            If provided, spike data will be extracted and saved to chunks.
        """
        # Save any remaining data
        if len(self.current_chunk_times) > 0:
            self._save_current_chunk(timestep__ms)

        # Save spike data
        spike_filename = self.save_path / "spikes.pkl"
        spike_data_arrays = {}

        if spike_results is not None:
            # Extract spike data from NEO Block (from SimulationRunner)
            print("\nExtracting spike data from SimulationRunner results...")
            from neo import Block

            if isinstance(spike_results, Block):
                for seg in spike_results.segments:
                    if len(seg.spiketrains) > 0:
                        pop_name = seg.name
                        times_list = []
                        ids_list = []

                        for st in seg.spiketrains:
                            neuron_id = int(st.name)
                            spike_times = st.times.rescale("ms").magnitude
                            times_list.extend(spike_times)
                            ids_list.extend([neuron_id] * len(spike_times))

                        spike_data_arrays[pop_name] = {
                            "times": np.array(times_list),
                            "ids": np.array(ids_list),
                        }

                        print(
                            f"{pop_name}: {len(times_list)} spikes from {len(seg.spiketrains)} neurons"
                        )
        else:
            # Use manually recorded spike data (legacy)
            for pop_name, data in self.spike_data.items():
                spike_data_arrays[pop_name] = {
                    "times": np.array(data["times"]),
                    "ids": np.array(data["ids"]),
                }

        joblib.dump(spike_data_arrays, spike_filename, compress=3)

        # Save metadata
        metadata = {
            "total_chunks": self.chunk_id,
            "chunk_duration__ms": self.chunk_duration__ms,
            "recording_config": self.recording_config,
        }
        joblib.dump(metadata, self.save_path / "metadata.pkl")

        print("\nContinuous saving complete:")
        print(f"\tTotal chunks saved: {self.chunk_id}")
        print(f"\tSpike data saved: {spike_filename}")
        print(f"\tPopulations with spikes: {list(spike_data_arrays.keys())}")
        print(f"\tAll data in: {self.save_path}")


def load_and_combine_chunks(save_path: Path, output_filename: Optional[str] = None):
    """
    Load all chunks from disk and combine into a single dataset.

    Parameters
    ----------
    save_path : Path
        Directory where chunks were saved
    output_filename : str, optional
        If provided, save combined data to this file

    Returns
    -------
    dict
        Combined dataset with all chunks merged
    """
    save_path = Path(save_path)

    # Load metadata
    metadata = joblib.load(save_path / "metadata.pkl")
    total_chunks = metadata["total_chunks"]

    print(f"Loading {total_chunks} chunks from {save_path}...")

    # Load all chunks
    chunks = []
    for chunk_id in range(total_chunks):
        chunk_filename = save_path / f"chunk_{chunk_id:04d}.pkl"
        if chunk_filename.exists():
            chunks.append(joblib.load(chunk_filename))
        else:
            print(f"Warning: Missing chunk {chunk_id}")

    # Combine chunks
    combined = {
        "times": np.concatenate([c["times"] for c in chunks]),
        "timestep__ms": chunks[0]["timestep__ms"],
        "membrane_data": defaultdict(dict),
    }

    # Combine membrane data for each population and cell
    for chunk in chunks:
        for pop_name, cells_dict in chunk["membrane_data"].items():
            for cell_idx, voltages in cells_dict.items():
                if cell_idx not in combined["membrane_data"][pop_name]:
                    combined["membrane_data"][pop_name][cell_idx] = []
                combined["membrane_data"][pop_name][cell_idx].append(voltages)

    # Convert concatenated lists to arrays
    for pop_name in combined["membrane_data"]:
        for cell_idx in combined["membrane_data"][pop_name]:
            combined["membrane_data"][pop_name][cell_idx] = np.concatenate(
                combined["membrane_data"][pop_name][cell_idx]
            )

    # Load spike data
    spike_filename = save_path / "spikes.pkl"
    if spike_filename.exists():
        combined["spikes"] = joblib.load(spike_filename)

    combined["metadata"] = metadata

    print(f"Combined {total_chunks} chunks:")
    print(f"\tTotal time points: {len(combined['times'])}")
    print(f"\tTime range: {combined['times'][0]:.1f} - {combined['times'][-1]:.1f} ms")
    print(f"\tDuration: {(combined['times'][-1] - combined['times'][0]) / 1000:.1f} seconds")

    # Optionally save combined data
    if output_filename:
        output_path = save_path / output_filename
        joblib.dump(combined, output_path, compress=3)
        print(f"\tSaved combined data to: {output_path}")

    return combined


def convert_chunks_to_neo(
    save_path: Path, duration__ms: Optional[float] = None, spike_data_file: Optional[Path] = None
) -> Block:
    """
    Load chunks and convert to NEO Block format (compatible with SimulationRunner output).

    This function creates a NEO Block that's identical in structure to what
    SimulationRunner.run() would return, making it compatible with existing
    analysis code.

    Parameters
    ----------
    save_path : Path
        Directory where chunks were saved
    duration__ms : float, optional
        Total simulation duration in ms (if None, inferred from data)
    spike_data_file : Path, optional
        Path to SimulationRunner spike results file (e.g., 'watanabe__spikes_only.pkl')
        If provided, spike data will be loaded from this NEO Block instead of chunks

    Returns
    -------
    Block
        NEO Block containing spike trains and analog signals
    """
    save_path = Path(save_path)

    print("Converting chunks to NEO Block format...")

    # Load metadata
    metadata = joblib.load(save_path / "metadata.pkl")
    total_chunks = metadata["total_chunks"]
    timestep__ms = None

    # Load spike data - either from external file or from chunks
    if spike_data_file is not None:
        # Load spike data from SimulationRunner results (NEO Block)
        print("\tLoading spike data from: {spike_data_file}")
        spike_results = joblib.load(spike_data_file)
        use_neo_spikes = True
    else:
        # Load spike data from chunks (legacy format)
        spike_filename = save_path / "spikes.pkl"
        if spike_filename.exists():
            print(f"  Loading spike data from: {spike_filename}")
            spike_data = joblib.load(spike_filename)
            use_neo_spikes = False
        else:
            print("\tWarning: No spike data found")
            spike_data = {}
            use_neo_spikes = False

    # Load first chunk to get timestep info
    first_chunk = joblib.load(save_path / "chunk_0000.pkl")
    timestep__ms = first_chunk["timestep__ms"]

    # Infer duration if not provided
    if duration__ms is None:
        last_chunk = joblib.load(save_path / f"chunk_{total_chunks - 1:04d}.pkl")
        duration__ms = last_chunk["time_end"]

    print(f"\tDuration: {duration__ms} ms")
    print(f"\tTimestep: {timestep__ms} ms")
    print(f"\tTotal chunks: {total_chunks}")

    # Create NEO Block
    block = Block()

    # Add spike data for each population
    print("\tAdding spike trains...")

    if use_neo_spikes:
        # Use spike data from SimulationRunner NEO Block
        for seg in spike_results.segments:
            if len(seg.spiketrains) > 0:
                # Create new segment with spike trains
                new_segment = Segment(name=seg.name)
                for st in seg.spiketrains:
                    new_segment.spiketrains.append(st)
                block.segments.append(new_segment)
                print(f"\t{seg.name}: {len(new_segment.spiketrains)} spike trains")
    else:
        # Use spike data from chunks (legacy format)
        for pop_name, spikes in spike_data.items():
            segment = Segment(name=pop_name)

            spike_times = spikes["times"]
            spike_ids = spikes["ids"]

            # Create spike trains for each neuron
            unique_ids = sorted(np.unique(spike_ids))
            for spike_id in tqdm(
                unique_ids, desc=f"\tCreating {pop_name} spike trains", leave=False
            ):
                times_for_id = spike_times[spike_ids == spike_id]

                # Filter out spike times that exceed duration due to floating-point precision
                # Keep only spikes strictly less than t_stop
                times_for_id = times_for_id[times_for_id < duration__ms]

                if len(times_for_id) > 0:
                    segment.spiketrains.append(
                        SpikeTrain(
                            name=str(int(spike_id)),
                            times=(times_for_id * pq.ms).rescale(pq.s),
                            t_start=0.0 * pq.s,
                            t_stop=(duration__ms * pq.ms).rescale(pq.s),
                            sampling_rate=(1.0 / (timestep__ms * pq.ms)).rescale(pq.Hz),
                        )
                    )

            block.segments.append(segment)
            print(f"\t{pop_name}: {len(segment.spiketrains)} spike trains")

    # Add membrane potential data by loading and combining chunks
    print(f"\tLoading and combining membrane data from {total_chunks} chunks...")

    # Determine which populations have membrane recordings
    first_chunk_membrane = first_chunk["membrane_data"]

    for pop_name in first_chunk_membrane.keys():
        # Find or create segment for this population
        segment = None
        for seg in block.segments:
            if seg.name == pop_name:
                segment = seg
                break

        if segment is None:
            segment = Segment(name=pop_name)
            block.segments.append(segment)

        # Get all cell indices for this population
        cell_indices = sorted(first_chunk_membrane[pop_name].keys())

        print(f"\t{pop_name}: Combining {len(cell_indices)} neurons from {total_chunks} chunks...")

        # OPTIMIZED: Load each chunk once and extract all neurons
        # This reduces file reads from (neurons × chunks) to just (chunks)
        # For 400 neurons × 36 chunks: 14,400 reads → 36 reads (400x faster!)

        # Initialize storage for each neuron
        neuron_data = {cell_idx: [] for cell_idx in cell_indices}

        # Load chunks once and distribute data to neurons
        for chunk_id in tqdm(
            range(total_chunks), desc=f"\tLoading {pop_name} chunks", unit="chunk"
        ):
            chunk = joblib.load(save_path / f"chunk_{chunk_id:04d}.pkl")

            if pop_name in chunk["membrane_data"]:
                for cell_idx in cell_indices:
                    if cell_idx in chunk["membrane_data"][pop_name]:
                        neuron_data[cell_idx].append(chunk["membrane_data"][pop_name][cell_idx])

        # Concatenate data for each neuron and create AnalogSignals
        print(f"\t{pop_name}: Creating analog signals...")
        for cell_idx in tqdm(
            cell_indices, desc=f"\tCreating {pop_name} signals", leave=False, unit="signal"
        ):
            if neuron_data[cell_idx]:
                combined_voltage = np.concatenate(neuron_data[cell_idx])

                segment.analogsignals.append(
                    AnalogSignal(
                        name=str(cell_idx),
                        sampling_period=(timestep__ms * pq.ms).rescale(pq.s),
                        signal=combined_voltage * pq.mV,
                    )
                )

        print(f"\t{pop_name}: {len(segment.analogsignals)} analog signals created")

    # Add metadata annotations
    block.annotations["time__ms"] = duration__ms
    block.annotations["timestep__ms"] = timestep__ms
    block.annotations["temperature__celsius"] = 36.0  # Default from SimulationRunner

    print("\nNEO Block created successfully")
    print(f"\tTotal segments: {len(block.segments)}")

    return block
