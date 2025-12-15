import neuron
import numpy as np
import quantities as pq
from beartype.typing import Sequence
from neuron import h

from myogen.simulator.neuron.populations import _Pool
from myogen.utils.decorators import beartowertype
from myogen.utils.types import CURRENT__AnalogSignal, Quantity__mV, SPIKE_TRAIN__Block


@beartowertype
def inject_currents_into_populations(
    populations: Sequence[_Pool],
    input_current__AnalogSignal: CURRENT__AnalogSignal,
) -> None:
    """
    Injects input currents into the specified populations.

    Sets up time-varying current injection using NEURON's IClamp and Vector.play()
    mechanisms. This function only sets up the current injection - spike recording
    and simulation execution must be handled separately by the user.

    Parameters
    ----------
    populations : Sequence[_Pool]
        The populations of neurons to inject current into.
    input_current__AnalogSignal : CURRENT__AnalogSignal
        The analog signal of input currents to inject into the population.
        Shape should be (time_points, n_pools) where n_pools matches len(populations).

    Returns
    -------
    None
        Current injection mechanisms are attached to the neurons as side effects.

    Raises
    ------
    ValueError
        If the number of populations does not match the number of current channels.

    Notes
    -----
    - Current injection vectors and IClamp objects are stored on each cell as
      `cell._stim_vectors` to prevent garbage collection
    - The user is responsible for setting up spike recording and calling h.run()
    """
    # Validate input dimensions
    n_pools = len(populations)
    n_current_channels = input_current__AnalogSignal.shape[1]

    if n_pools != n_current_channels:
        raise ValueError(
            f"Number of populations ({n_pools}) must match number of current channels "
            f"({n_current_channels})"
        )

    simulation_time__ms = input_current__AnalogSignal.t_stop.rescale(pq.ms).magnitude

    for pool_idx, pool in enumerate(populations):
        for cell in pool:
            # Create IClamp with time-varying amplitude using Vector.play()
            stim = h.IClamp(cell.soma(0.5))
            stim.delay = 0
            stim.dur = simulation_time__ms  # Keep it "always on"
            stim.amp = 0  # Will be controlled by vector

            # Create time and current vectors
            vec_t = h.Vector(input_current__AnalogSignal.times.rescale(pq.ms).magnitude)
            vec_i = h.Vector(
                input_current__AnalogSignal[:, pool_idx].rescale(pq.nA).magnitude
            )

            # Play waveform into stim.amp
            vec_i.play(stim._ref_amp, vec_t, 1)  # 1 = continuous update

            # CRITICAL: Keep references to vectors to prevent garbage collection
            # Store them as attributes on the cell to keep them alive
            cell._stim_vectors = (stim, vec_t, vec_i)  # Prevent GC cleanup


@beartowertype
def inject_currents_and_simulate_spike_trains(
    populations: Sequence[_Pool],
    input_current__AnalogSignal: CURRENT__AnalogSignal,
    spike_detection_thresholds__mV: Quantity__mV | Sequence[Quantity__mV] = -10.0
    * pq.mV,
) -> SPIKE_TRAIN__Block:
    """
    Injects input currents into populations and returns recorded spike trains.

    This is a complete pipeline function that sets up current injection, spike recording,
    runs the NEURON simulation, and returns the results as a properly formatted neo.Block.

    Parameters
    ----------
    populations : Sequence[_Pool]
        The populations of neurons to inject current into.
    input_current__AnalogSignal : CURRENT__AnalogSignal
        The analog signal of input currents to inject into the population.
        Shape should be (time_points, n_pools) where n_pools matches len(populations).
    spike_detection_thresholds__mV : float | Sequence[float], optional
        Thresholds for spike detection in millivolts, by default -10.0. If a sequence is provided, it must match the number of populations.


    Returns
    -------
    SPIKE_TRAIN__Block
        Neo Block containing spike trains organized as segments (pools) with spiketrains (neurons).
        Each segment represents a motor unit pool, each spiketrain represents a neuron.

    Raises
    ------
    ValueError
        If the number of populations does not match the number of current channels.

    Notes
    -----
    This function performs the complete simulation pipeline:
    1. Sets up current injection (same as inject_currents_into_populations)
    2. Sets up spike recording for all neurons
    3. Runs the NEURON simulation via h.run()
    4. Converts recorded spikes to neo.Block format
    """
    from neo import Block, Segment, SpikeTrain

    # First, set up current injection using the existing function
    inject_currents_into_populations(populations, input_current__AnalogSignal)

    simulation_time__ms = input_current__AnalogSignal.t_stop.rescale(pq.ms)

    # Validate input dimensions
    n_pools = len(populations)
    n_current_channels = input_current__AnalogSignal.shape[1]

    if n_pools != n_current_channels:
        raise ValueError(
            f"Number of populations ({n_pools}) must match number of current channels "
            f"({n_current_channels})"
        )

    if not isinstance(spike_detection_thresholds__mV, Sequence):
        spike_detection_thresholds__mV = [spike_detection_thresholds__mV] * n_pools

    # Set up spike recording for all neurons
    spike_recorders = []

    for pool_idx, pool in enumerate(populations):
        pool_spike_recorders = []

        for cell in pool:
            # Setup spike recording
            spike_recorder = h.Vector()
            nc = h.NetCon(cell.soma(0.5)._ref_v, None, sec=cell.soma)
            nc.threshold = spike_detection_thresholds__mV[pool_idx]
            nc.record(spike_recorder)

            pool_spike_recorders.append(spike_recorder)

        spike_recorders.append(pool_spike_recorders)

    # Initialize sections with proper voltages for each population
    for pool in populations:
        for section, voltage in zip(*pool.get_initialization_data()):
            section.v = voltage

    # Initialize and run the NEURON simulation
    h.finitialize()  # Use default initialization, voltages already set above
    neuron.run(simulation_time__ms)

    # Convert spike data to neo.Block format
    block = Block()

    for pool_idx, pool_spike_recorders in enumerate(spike_recorders):
        # Create segment for this motor unit pool
        segment = Segment(name=f"Pool {pool_idx}")

        segment.spiketrains = [
            SpikeTrain(
                (spike_recorder.as_numpy() * pq.ms).rescale(pq.s),
                t_stop=simulation_time__ms.rescale(pq.s),
                sampling_rate=(1 / (h.dt * pq.ms)).rescale(pq.Hz),
                sampling_period=(h.dt * pq.ms).rescale(pq.s),
                name=str(int(neuron_idx)),
                description=f"Pool {pool_idx}, Neuron {neuron_idx}",
            )
            for neuron_idx, spike_recorder in enumerate(pool_spike_recorders)
        ]

        # Only add segment if it has spiketrains (which it always will now)
        block.segments.append(segment)

    return block
