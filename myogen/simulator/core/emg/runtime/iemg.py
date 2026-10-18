"""Stream the public zero-jitter fibre calculation on the full MU time grid."""
import numpy as np
import quantities as pq
from neo import AnalogSignal, Block, Segment

from ..intramuscular.bioelectric import (
    get_elementary_current_response, shift_padding, hr_shift_template,
)
from .density import current_density as get_current_density
from .session import ExecutionSession
from .cache import DiskCache, fingerprint

IEMG_CACHE_NAMESPACE = "intramuscular-muap-density-v2"


def stream_muap(mu, dt, dz, electrodes, session, *, batch_size=8, label="fibres"):
    # Owned snapshots protect realized inputs; no random work occurs below.
    centers, diameters, velocities, nmj, left, right, paths, electrodes = [
        session.snapshot(value) for value in (
            mu.muscle_fiber_centers__mm, mu.muscle_fiber_diameters__mm,
            mu.muscle_fiber_conduction_velocity__mm_per_s,
            mu._neuromuscular_z_coordinates__mm,
            mu._muscle_fiber_left_ends__mm, mu._muscle_fiber_right_ends__mm,
            mu.nerve_paths, electrodes,
        )
    ]
    t = np.arange(0, 2 * np.max([
        np.divide(nmj - left, velocities), np.divide(right - nmj, velocities)
    ]) + dt, dt)[..., None]
    session.track([t])
    output = session.zeros((len(t), len(electrodes)))
    minimum_radius = float(np.mean(diameters))
    delays = np.divide(paths, np.asarray(
        mu.neuromuscular_junction_conduction_velocities__mm_per_s)[None, :]).sum(axis=1)
    session.track([delays])
    max_z = int(np.ceil(np.max((right - left) / dz))) + 5
    # Retain the conservative reference workspace reservation, including native
    # preparation overhead. The compiled kernel uses only two dense arrays;
    # a reservation does not allocate or fill the unused allowance.
    workspace = max_z * len(t) * 8 * 20 + max_z * len(electrodes) * 8 * 3
    per_fibre = output.nbytes

    def compute(start, stop):
        values = np.empty((stop - start, *output.shape), dtype=np.float64)
        for local, index in enumerate(range(start, stop)):
            zi = nmj[index].item()
            a, b = left[index].item(), right[index].item()
            z_left = np.arange(start=zi, step=-dz, stop=a - dz)
            z_right = np.arange(start=zi, step=dz, stop=b + dz)
            z = np.concatenate((z_left[::-1], z_right[1:]))[:, None]
            current = get_current_density(t, z, zi, b - zi, zi - a,
                                          velocities[index].item(), diameters[index].item())
            responses = []
            for electrode in electrodes:
                radius = np.sqrt(np.sum((electrode[:2] - centers[index])**2, keepdims=True))
                if radius < minimum_radius:
                    radius = minimum_radius
                responses.append(get_elementary_current_response(z, electrode[2], radius))
            values[local] = current.T @ np.column_stack(responses)
            delay = float(delays[index] / dt)
            integer = int(np.floor(delay))
            fraction = delay - integer
            for point in range(len(electrodes)):
                values[local, :, point] = hr_shift_template(
                    shift_padding(values[local, :, point], integer, axis=0), -fraction)
            del current, responses
        return values

    def commit(values):
        # Stable fibre order is independent of changing batch boundaries.
        for value in values:
            output[:] += value

    session.batches(len(centers), batch_size, per_fibre, compute, commit,
                    label=label, fixed_bytes=workspace)
    return output


def simulate_muaps(owner, policy, n_jobs, verbose):
    with ExecutionSession(owner, policy, "intramuscular_muaps", n_jobs=n_jobs) as session:
        model_arrays = [value for value in vars(owner._muscle_model).values()
                        if isinstance(value, np.ndarray)]
        session.track(model_arrays)
        # Realization is deterministic, but still needs admission before its
        # arrays and NMJ branch bookkeeping are constructed.
        count = int(np.sum(owner._muscle_model.resulting_number_of_innervated_fibers))
        realization_bytes = count * 1024 + owner._n_motor_units * 4096
        session.memory.require(realization_bytes)
        # Exactly the reference realization order, including unselected MUs.
        with session.memory.wait_reserve(realization_bytes):
            owner._initialize_motor_units(verbose=verbose)
            owner._simulate_neuromuscular_junctions(verbose=verbose)
        session.track([value for mu in owner._motor_units if mu is not None
                       for value in vars(mu).values() if isinstance(value, np.ndarray)])
        disk = None if policy.cache_dir is None else DiskCache(policy.cache_dir)
        results = {}
        output_bytes = []
        def compute_mu(index):
            mu = owner._motor_units[index]
            if mu is None:
                return index, None
            key = fingerprint(
                IEMG_CACHE_NAMESPACE, owner._dt, owner._dz, owner._electrode_array.pts,
                mu.muscle_fiber_centers__mm, mu.muscle_fiber_diameters__mm,
                mu.muscle_fiber_conduction_velocity__mm_per_s,
                mu._neuromuscular_z_coordinates__mm, mu._muscle_fiber_left_ends__mm,
                mu._muscle_fiber_right_ends__mm, mu.nerve_paths,
                mu.neuromuscular_junction_conduction_velocities__mm_per_s,
            )
            result = None if disk is None else disk.get(key, session)
            if result is None:
                result = stream_muap(mu, owner._dt, owner._dz,
                                     owner._electrode_array.pts, session, label=f"MU_{index}")
                if disk is not None:
                    with session.memory.wait_reserve(result.nbytes + 65536):
                        disk.put(key, result)
            return index, result
        def minimum(index):
            mu = owner._motor_units[index]
            if mu is None:
                return 0
            time_samples = int(np.ceil(2 * np.max([
                (mu._neuromuscular_z_coordinates__mm - mu._muscle_fiber_left_ends__mm) /
                mu.muscle_fiber_conduction_velocity__mm_per_s,
                (mu._muscle_fiber_right_ends__mm - mu._neuromuscular_z_coordinates__mm) /
                mu.muscle_fiber_conduction_velocity__mm_per_s,
            ]) / owner._dt)) + 2
            space_samples = int(np.ceil(np.max(
                mu._muscle_fiber_right_ends__mm - mu._muscle_fiber_left_ends__mm
            ) / owner._dz)) + 5
            return time_samples * space_samples * 8 * 24 + 1024 * len(mu.nerve_paths)
        for mu in owner._motor_units:
            if mu is None:
                samples = 1
            else:
                samples = int(np.ceil(2 * np.max([
                    (mu._neuromuscular_z_coordinates__mm - mu._muscle_fiber_left_ends__mm) /
                    mu.muscle_fiber_conduction_velocity__mm_per_s,
                    (mu._muscle_fiber_right_ends__mm - mu._neuromuscular_z_coordinates__mm) /
                    mu.muscle_fiber_conduction_velocity__mm_per_s,
                ]) / owner._dt)) + 2
            output_bytes.append(samples * len(owner._electrode_array.pts) * 8)
        minimum_job = max((minimum(index) for index in owner._MUs_to_simulate), default=0)
        required = 3 * sum(output_bytes)
        session.memory.require(required + minimum_job)
        if required + 2 * minimum_job > session.memory.allowance() and session.workers > 1:
            session.workers = 1
            session.worker_changes.append(1)
        for index, value in session.jobs(owner._MUs_to_simulate, compute_mu, minimum_bytes=minimum):
            results[index] = value
        block = Block()
        for index in range(len(owner._motor_units)):
            value = results.get(index)
            if value is None:
                value = session.zeros((1, len(owner._electrode_array.pts)))
            segment = Segment(name=f"MUAP_{index}")
            with session.memory.wait_reserve(value.nbytes * 2):
                signal = AnalogSignal(value * pq.dimensionless,
                                      sampling_rate=owner._sampling_frequency__Hz * pq.Hz)
            session.track([signal.magnitude])
            segment.analogsignals.append(signal)
            block.segments.append(segment)
    owner._muaps__Block = block
    return block
