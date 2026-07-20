"""
Current Injection and Spike Train Simulation Utilities for Jaxley.

This module provides high-level functions for simulating populations of
neurons with current injection, mirroring the NEURON workflow.

Functions
---------
inject_currents_and_simulate_spike_trains
    High-level function to inject currents and record spike trains.
inject_currents_into_populations
    Set up current injection for populations.
simulation_result_to_neo_block
    Convert simulation results to Neo Block format.
"""

from typing import List, Optional, Sequence, Union

import numpy as np
import jax.numpy as jnp
import quantities as pq
from neo import Block, Segment, SpikeTrain, AnalogSignal

import jaxley as jx

from myogen.simulator.jaxley.populations.base import _Pool
from myogen.simulator.jaxley.integrate import (
    detect_spikes,
    compute_firing_rate_from_spikes,
    compute_isi_cv,
    custom_current,
)


# Type aliases for clarity
CURRENT__AnalogSignal = AnalogSignal
SPIKE_TRAIN__Block = Block


def inject_currents_into_populations(
    populations: Sequence[_Pool],
    input_current__AnalogSignal: CURRENT__AnalogSignal,
    stim_branch: int = 0,
    stim_loc: float = 0.5,
) -> None:
    """
    Set up current injection for populations of Jaxley cells.
    
    Parameters
    ----------
    populations : Sequence[_Pool]
        List of population objects (e.g., AlphaMN__Pool).
    input_current__AnalogSignal : neo.AnalogSignal
        Input current signal with shape (n_samples, n_pools).
        Each column corresponds to a population.
    stim_branch : int
        Branch index for current injection (default: 0, soma).
    stim_loc : float
        Location on branch for injection (0.0 to 1.0).
    """
    # Extract timestep and convert current to numpy
    dt = float(input_current__AnalogSignal.sampling_period.rescale(pq.ms).magnitude)
    current_data = np.array(input_current__AnalogSignal.magnitude)
    
    # Handle 1D vs 2D current arrays
    if current_data.ndim == 1:
        current_data = current_data[:, np.newaxis]
    
    n_pools = len(populations)
    
    for pool_idx, pool in enumerate(populations):
        # Get current for this pool (use modulo to handle fewer columns than pools)
        pool_current = current_data[:, pool_idx % current_data.shape[1]]
        pool_current_jnp = jnp.array(pool_current)
        
        # Inject current into each cell in the pool
        for cell_wrapper in pool:
            if hasattr(cell_wrapper, 'cell'):
                cell = cell_wrapper.cell
                # Clear existing stimuli
                cell.delete_stimuli()
                # Set up stimulation
                cell.branch(stim_branch).loc(stim_loc).stimulate(pool_current_jnp)


def inject_currents_and_simulate_spike_trains(
    populations: Sequence[_Pool],
    input_current__AnalogSignal: CURRENT__AnalogSignal,
    spike_detection_thresholds__mV: Union[pq.Quantity, float] = 50.0 * pq.mV,
    stim_branch: int = 0,
    stim_loc: float = 0.5,
    record_branch: int = 0,
    record_loc: float = 0.5,
) -> SPIKE_TRAIN__Block:
    """
    Inject currents into populations and simulate to produce spike trains.
    
    This is the main high-level function for running Jaxley simulations.
    It mirrors the NEURON utility function for API compatibility.
    
    Parameters
    ----------
    populations : Sequence[_Pool]
        List of population objects (e.g., AlphaMN__Pool).
    input_current__AnalogSignal : neo.AnalogSignal
        Input current signal with shape (n_samples, n_pools).
    spike_detection_thresholds__mV : pq.Quantity or float
        Threshold for spike detection. Can be a single value or per-population.
    stim_branch : int
        Branch index for current injection (default: 0).
    stim_loc : float
        Location on branch for injection (0.0 to 1.0).
    record_branch : int
        Branch index for voltage recording (default: 0).
    record_loc : float
        Location on branch for recording (0.0 to 1.0).
    
    Returns
    -------
    neo.Block
        Block containing spike trains for each population (as segments).
    """
    # Extract simulation parameters
    dt = float(input_current__AnalogSignal.sampling_period.rescale(pq.ms).magnitude)
    t_max = float(input_current__AnalogSignal.t_stop.rescale(pq.ms).magnitude)
    n_samples = len(input_current__AnalogSignal)
    
    # Handle threshold
    if hasattr(spike_detection_thresholds__mV, 'magnitude'):
        threshold = float(spike_detection_thresholds__mV.rescale(pq.mV).magnitude)
    else:
        threshold = float(spike_detection_thresholds__mV)
    
    # Get current data
    current_data = np.array(input_current__AnalogSignal.magnitude)
    if current_data.ndim == 1:
        current_data = current_data[:, np.newaxis]
    
    # Create output Block
    spike_train__Block = Block(name="Jaxley Simulation Results")
    
    # Simulate each population
    for pool_idx, pool in enumerate(populations):
        # Get current for this pool
        pool_current = current_data[:, pool_idx % current_data.shape[1]]
        pool_current_jnp = jnp.array(pool_current)
        
        # Create segment for this pool
        segment = Segment(name=f"Pool {pool_idx}")
        segment.spiketrains = []
        
        # Simulate each cell in the pool
        for neuron_idx, cell_wrapper in enumerate(pool):
            spike_times_ms = _simulate_single_cell(
                cell_wrapper,
                pool_current_jnp,
                dt,
                t_max,
                threshold,
                stim_branch,
                stim_loc,
                record_branch,
                record_loc,
            )
            
            # Convert to Neo SpikeTrain
            spike_times_s = spike_times_ms / 1000.0  # ms to s
            
            spiketrain = SpikeTrain(
                spike_times_s * pq.s,
                t_stop=(t_max / 1000.0) * pq.s,
                sampling_rate=(1000.0 / dt) * pq.Hz,
                name=str(neuron_idx),
                description=f"Pool {pool_idx}, Neuron {neuron_idx}",
            )
            segment.spiketrains.append(spiketrain)
        
        spike_train__Block.segments.append(segment)
    
    return spike_train__Block


def _simulate_single_cell(
    cell_wrapper,
    current: jnp.ndarray,
    dt: float,
    t_max: float,
    threshold: float,
    stim_branch: int,
    stim_loc: float,
    record_branch: int,
    record_loc: float,
) -> np.ndarray:
    """
    Simulate a single cell and return spike times.
    
    Parameters
    ----------
    cell_wrapper : cell object
        Cell wrapper with .cell attribute containing jx.Cell.
    current : jnp.ndarray
        Current waveform to inject.
    dt : float
        Time step (ms).
    t_max : float
        Total simulation time (ms).
    threshold : float
        Spike detection threshold (mV).
    stim_branch : int
        Branch for stimulation.
    stim_loc : float
        Location for stimulation.
    record_branch : int
        Branch for recording.
    record_loc : float
        Location for recording.
    
    Returns
    -------
    np.ndarray
        Spike times in ms.
    """
    if not hasattr(cell_wrapper, 'cell'):
        return np.array([])
    
    cell = cell_wrapper.cell
    
    try:
        # Clear previous recordings and stimuli
        cell.delete_recordings()
        cell.delete_stimuli()
        
        # Set up recording
        cell.branch(record_branch).loc(record_loc).record("v")
        
        # Set up stimulation
        cell.branch(stim_branch).loc(stim_loc).stimulate(current)
        
        # Run simulation
        voltages = jx.integrate(cell, delta_t=dt, t_max=t_max)
        
        # Extract voltage trace for the single recorded compartment.
        # jx.integrate() returns shape (n_timepoints, n_recorded_compartments).
        # [:, 0] selects all timepoints for compartment index 0.
        if voltages.ndim == 2:
            v = np.array(voltages[:, 0])
        else:
            v = np.array(voltages)
        
        # Detect spikes
        spike_indices = np.where((v[:-1] < threshold) & (v[1:] >= threshold))[0]
        spike_times = spike_indices * dt
        
        return spike_times
        
    except Exception as e:
        # If simulation fails, return empty spike train
        print(f"Warning: Simulation failed for cell: {e}")
        return np.array([])


def simulation_result_to_neo_block(
    voltages: np.ndarray,
    spike_times_list: List[List[float]],
    dt: float,
    t_max: float,
    pool_names: Optional[List[str]] = None,
) -> Block:
    """
    Convert simulation results to Neo Block format.
    
    Parameters
    ----------
    voltages : np.ndarray
        Voltage traces, shape (n_cells, n_timesteps) or (n_timesteps,).
    spike_times_list : List[List[float]]
        List of spike times for each cell (in ms).
    dt : float
        Time step (ms).
    t_max : float
        Total simulation time (ms).
    pool_names : List[str], optional
        Names for each pool/segment.
    
    Returns
    -------
    neo.Block
        Block with spike trains.
    """
    block = Block(name="Jaxley Simulation Results")
    
    # Handle different input formats
    if isinstance(spike_times_list[0], (list, np.ndarray)):
        # Multiple cells
        n_cells = len(spike_times_list)
    else:
        # Single cell - wrap in list
        spike_times_list = [spike_times_list]
        n_cells = 1
    
    # Create single segment with all spike trains
    segment = Segment(name=pool_names[0] if pool_names else "Pool 0")
    segment.spiketrains = []
    
    for i, spike_times_ms in enumerate(spike_times_list):
        spike_times_s = np.array(spike_times_ms) / 1000.0
        
        spiketrain = SpikeTrain(
            spike_times_s * pq.s,
            t_stop=(t_max / 1000.0) * pq.s,
            sampling_rate=(1000.0 / dt) * pq.Hz,
            name=str(i),
        )
        segment.spiketrains.append(spiketrain)
    
    block.segments.append(segment)
    
    return block


def create_input_current_from_waveform(
    waveform: np.ndarray,
    dt_ms: float,
    n_pools: int = 1,
) -> AnalogSignal:
    """
    Create Neo AnalogSignal from a current waveform array.
    
    Parameters
    ----------
    waveform : np.ndarray
        Current waveform in nA, shape (n_samples,) or (n_samples, n_pools).
    dt_ms : float
        Time step in ms.
    n_pools : int
        Number of pools (columns) to create if waveform is 1D.
    
    Returns
    -------
    neo.AnalogSignal
        Current signal compatible with inject_currents_and_simulate_spike_trains.
    """
    if waveform.ndim == 1:
        # Replicate for n_pools
        waveform = np.tile(waveform[:, np.newaxis], (1, n_pools))
    
    signal = AnalogSignal(
        waveform * pq.nA,
        sampling_period=dt_ms * pq.ms,
        t_start=0 * pq.ms,
    )
    
    return signal
