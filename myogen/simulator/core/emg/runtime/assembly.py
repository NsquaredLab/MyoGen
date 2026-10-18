"""Modality adapters preserving the public padding, time grids, and Neo layout."""
import numpy as np
import quantities as pq
from scipy.fft import next_fast_len
from neo import AnalogSignal, Block, Group, Segment

from myogen.utils.binning import bin_spike_trains, _round_binning_errors
from myogen.utils.neo import create_grid_signal, signal_to_grid
from .session import ExecutionSession
from .synthesis import convolve_add
from .cache import fingerprint


def occupancy(spikes, session):
    first = spikes.segments[0].spiketrains[0]
    dt = first.sampling_period.rescale(pq.s)
    counts = []
    max_events = 0
    for pool in spikes.segments:
        train = pool.spiketrains[0]
        counts.append(max(0, _round_binning_errors(float(
            ((train.t_stop - train.t_start) / dt).simplified))))
        max_events = max(max_events, sum(len(train) for train in pool.spiketrains))
    if len(set(counts)) != 1 or len({len(p.spiketrains) for p in spikes.segments}) != 1:
        raise ValueError("Pools must have matching neuron and bin counts")
    shape = (len(spikes.segments), len(spikes.segments[0].spiketrains), counts[0])
    session.memory.require(int(np.prod(shape)) + shape[1] * shape[2] + 64 * max_events)
    result = session.zeros(shape, bool)
    for index, pool in enumerate(spikes.segments):
        with session.memory.wait_reserve(shape[1] * shape[2] + 64 * max_events):
            result[index] = bin_spike_trains(pool.spiketrains, bin_size=dt)
    return result, float(dt.magnitude)


def assemble(owner, spikes, policy, *, surface):
    if owner._muaps__Block is None:
        raise ValueError("MUAP templates have not been generated. Call simulate_muaps() first.")
    # This stage uses serial NumPy kernels and explicitly one-worker SciPy FFTs;
    # it has no BLAS/Numba work requiring process-wide native-pool discovery.
    with ExecutionSession(owner, policy, "surface_emg" if surface else "intramuscular_emg",
                          native_pools=False) as session:
        groups = owner._muaps__Block.groups if surface else [owner._muaps__Block]
        signals = [seg.analogsignals[0] for group in groups for seg in group.segments]
        session.track([signal.magnitude for signal in signals])
        bins, dt = occupancy(spikes, session)
        chosen = set(range(bins.shape[1])) if owner._MUs_to_simulate is None else set(owner._MUs_to_simulate)
        selected = chosen.intersection(set(range(bins.shape[1])))
        block = Block()
        # The reference uses the first surface template's native rate for all arrays.
        native = (float(signals[0].sampling_rate) if surface else owner._sampling_frequency__Hz)
        for array_index, group in enumerate(groups):
            data = [signal_to_grid(seg.analogsignals[0]) if surface else seg.analogsignals[0].magnitude
                    for seg in group.segments]
            longest = max(value.shape[0] for value in data)
            channel_shape = data[0].shape[1:]
            channels = int(np.prod(channel_shape))
            # Preserve the reference conversion and operation order: an
            # algebraically equivalent division can round a half sample down.
            period = ((float((1 / native) * 1000) * pq.ms).rescale(pq.s).magnitude
                      if surface else 1 / native)
            duration = longest * period if surface else longest / native
            target = int(np.round(duration / dt))
            if surface:
                target = max(1, target)
                x = session.array(lambda: np.arange(target) * dt, target * 16)
                xp = session.array(lambda: np.arange(longest) * period, longest * 16)
            else:
                x = session.array(lambda: np.linspace(0, duration, target, endpoint=False), (target + 2) * 16)
                xp = session.array(lambda: np.arange(0, duration, 1 / native), (longest + 2) * 16)
            # Surface amplitude has no population normalization dependency.
            # Prepare only configured contributors; retain every source MUAP.
            # iEMG normalization still uses the complete template population.
            indices = (tuple(sorted(mu for mu in selected if mu < len(data))) if surface
                       else tuple(range(len(data))))
            prepared_data = [data[mu] for mu in indices]
            slots = {mu: slot for slot, mu in enumerate(indices)}
            key = fingerprint("resampled-templates", surface, indices, channel_shape,
                              prepared_data, native, dt, x, xp)
            templates = session.cache.get(key)
            if templates is None:
                templates = session.zeros((len(indices), channels, target))
                # Reuse one useful padding workspace across all public float64
                # templates. Its pending reservation covers the whole bounded
                # preparation loop, including interpolation's temporary copies.
                with session.memory.wait_reserve((2 * longest * channels + target) * 8):
                    workspace = (np.empty((longest, channels), dtype=np.float64)
                                 if prepared_data and all(value.dtype == np.float64 for value in prepared_data)
                                 else None)
                    for slot, value in enumerate(prepared_data):
                        padding = longest - value.shape[0]
                        # Reference iEMG centres shorter MUAPs before interpolation.
                        if workspace is None:
                            padded = np.pad(value.reshape(value.shape[0], channels),
                                            ((padding // 2, padding - padding // 2), (0, 0)))
                        else:
                            workspace.fill(0)
                            workspace[padding // 2:padding // 2 + value.shape[0]] = value.reshape(
                                value.shape[0], channels)
                            padded = workspace
                        for channel in range(channels):
                            templates[slot, channel] = np.interp(x, xp, padded[:, channel])
                        del padded
                    del workspace
                if not surface:
                    # Avoid a full-template abs allocation; extrema give the
                    # same scale, including NaN/inf behavior, without a copy.
                    amplitude = max(abs(np.min(templates)), abs(np.max(templates)))
                    if amplitude > 0:
                        templates /= amplitude
                session.cache.put(key, templates)
            else:
                session.memory.events["cache_hits"] += 1
            length = max(bins.shape[-1], target)
            n_output = int(length * dt * owner._sampling_frequency__Hz)
            # Retain only the required outputs and one native-rate channel.
            # Neo accepts a numerical array and explicit units without the
            # recording-sized Quantity multiplication used by the legacy path.
            convolution_minimum = min((bins.shape[-1] + target + length) * 8,
                                      next_fast_len(target, real=True) * 8 * 8)
            interpolation_peak = (length + n_output + max(length, n_output)) * 8
            session.memory.require((bins.shape[0] * channels * n_output + length) * 8 +
                                   max(convolution_minimum, interpolation_peak))
            summed = session.zeros((length,))
            if surface:
                target_time = session.array(lambda: np.arange(n_output) / owner._sampling_frequency__Hz,
                                            n_output * 16)
                source_time = session.array(lambda: np.arange(length) * dt, length * 16)
            else:
                target_time = session.array(lambda: np.arange(0, length * dt, 1 / owner._sampling_frequency__Hz),
                                            (n_output + 2) * 16)
                source_time = session.array(lambda: np.arange(0, length * dt, dt), (length + 2) * 16)
            destination = block
            if surface:
                destination = Group(name=f"ElectrodeArray_{array_index}")
                block.groups.append(destination)
            for pool in range(bins.shape[0]):
                output = session.zeros((n_output, channels))
                for channel in range(channels):
                    summed.fill(0)
                    for mu in selected:
                        slot = slots.get(mu)
                        if slot is not None:
                            convolve_add(summed, bins[pool, mu], templates[slot, channel], session)
                    with session.memory.wait_reserve(n_output * 8):
                        output[:, channel] = np.interp(target_time, source_time, summed)
                segment = Segment(name=f"Pool_{pool}")
                destination.segments.append(segment)
                signal = AnalogSignal(output, units=pq.mV if surface else pq.dimensionless,
                                      t_start=0 * (pq.s if surface else pq.ms),
                                      sampling_rate=owner._sampling_frequency__Hz * pq.Hz)
                if surface:
                    # Same public create_grid_signal defaults and ordering.
                    signal.annotate(grid_shape=channel_shape, electrode_positions=None, ied=None)
                session.track([signal.magnitude])
                segment.analogsignals.append(signal)
            del target_time, source_time, summed, output, templates, x, xp, data, prepared_data
    owner._spike_train__Block = spikes
    if surface:
        owner._surface_emg__Block = block
    else:
        owner._intramuscular_emg__Block = block
    return block
