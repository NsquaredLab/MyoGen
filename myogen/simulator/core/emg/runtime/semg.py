"""Bounded evaluation of the unchanged public frequency-domain source model."""
from copy import deepcopy
import numpy as np
import quantities as pq
from neo import Block, Group, Segment
from scipy.signal import resample

from myogen import get_random_generator
from myogen.utils.neo import create_grid_signal
from ..electrodes import SurfaceElectrodeArray
from ..surface.simulate_fiber import (
    _simulate_fiber_v2_python,
    fiber_semi_lengths,
    muap_sampling_rate__Hz,
)
from .session import ExecutionSession
from .surface_preparation import SurfacePreparation
from .cache import DiskCache, fingerprint


def simulate_muaps(owner, policy, n_jobs, verbose, use_gpu):
    with ExecutionSession(owner, policy, "surface_muaps", n_jobs=n_jobs) as session:
        # A single ordered worker amortizes conductor preparations across MUs.
        # The policy is an upper bound, not a request to create idle workers.
        session.workers = 1
        session.worker_changes[:] = [1]
        if use_gpu is True:
            raise ValueError("execution=ExecutionPolicy(backend='cpu') contradicts use_gpu=True")
        if owner._use_unified:
            raise ValueError("use_unified=True requires the existing route (execution=None)")
        if any(type(array) is not SurfaceElectrodeArray for array in owner._electrode_arrays):
            raise ValueError("Custom electrodes require the existing route (execution=None)")
        shapes = {(a.num_rows, a.num_cols) for a in owner._electrode_arrays}
        if len(shapes) != 1:
            raise ValueError("Optimized sEMG requires matching electrode grid dimensions")
        counts = np.array(owner._muscle_model.resulting_number_of_innervated_fibers, dtype=int)
        session.track([value for value in vars(owner._muscle_model).values()
                       if isinstance(value, np.ndarray)])
        n_mu = len(counts)
        selected = list(range(n_mu)) if owner._MUs_to_simulate is None else owner._MUs_to_simulate
        # Preserve reference draws for ALL MUs, including unselected MUs and
        # cache hits. A second call always obtains its new realization first.
        session.memory.require(int(sum(counts) + n_mu) * 8 * 4)
        zones = get_random_generator().uniform(
            -owner._mean_fiber_length__mm * .1 / 2,
            owner._mean_fiber_length__mm * .1 / 2, size=n_mu)
        variations = [get_random_generator().uniform(
            -owner._var_fiber_length__mm, owner._var_fiber_length__mm, size=int(count))
            for count in counts]
        session.track([zones, *variations])
        for value in (zones, *variations):
            value.flags.writeable = False
        kernel_length = (owner._iap_kernel_length__mm if owner._iap_kernel_length__mm is not None
                         else owner._mean_fiber_length__mm * 2.5)
        N, M = int(owner._internal_sampling_points), int(owner._sampling_points_in_theta_domain)
        rows, cols = next(iter(shapes))
        shape = (rows, cols, N)
        final_samples = owner._sampling_points_in_t_and_z_domains
        rate = muap_sampling_rate__Hz(owner._mean_conduction_velocity__m_s, N, final_samples,
                                      kernel_length)
        # Source-grid arithmetic retains the original precision and complete
        # N x N grids. Estimate all live complex/native temporaries generously.
        workspace = N * N * 16 * 24 + N * M * 8 * 180 + rows * cols * N * 16 * 4
        session.memory.require(workspace + n_mu * rows * cols * final_samples * 8 * 3)
        block = Block()
        disk = None if policy.cache_dir is None else DiskCache(policy.cache_dir)
        for array_index, original in enumerate(owner._electrode_arrays):
            electrode = deepcopy(original)
            base_z = electrode.pos_z.rescale(pq.mm).magnitude.copy()
            base_theta = electrode.pos_theta.rescale(pq.rad).magnitude.copy()
            radius = float(electrode.electrode_radius__mm.rescale(pq.mm).magnitude)
            preparation = SurfacePreparation()
            dependencies = (
                "surface-muap-gitlab-fc077-v2", N, M, final_samples, kernel_length, rate,
                owner._internal_sampling_frequency__Hz, owner._mean_conduction_velocity__m_s,
                owner._radius_total, owner._radius_bone__mm,
                owner._fat_thickness__mm, owner._skin_thickness__mm,
                owner._muscle_conductivity_radial__S_m, owner._muscle_conductivity_longitudinal__S_m,
                owner._fat_conductivity__S_m, owner._skin_conductivity__S_m,
                vars(electrode), base_z, base_theta, radius,
            )
            results = {}
            group = Group(name=f"ElectrodeArray_{array_index}")
            block.groups.append(group)
            for mu_index in selected:
                positions = session.snapshot(owner._muscle_model.resulting_fiber_assignment(mu_index))
                radial = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
                angular = np.arctan2(positions[:, 1], positions[:, 0])
                lengths = owner._mean_fiber_length__mm + variations[mu_index]
                zone = zones[mu_index]
                session.track([radial, angular, lengths])
                key = fingerprint(dependencies, positions, radial, angular, lengths, zone)
                cached = None if disk is None else disk.get(key, session)
                if cached is not None:
                    results[mu_index] = cached
                    continue
                accumulated = session.zeros(shape)
                def compute(start, stop):
                    values = np.empty((stop - start, *shape))
                    for local, fibre in enumerate(range(start, stop)):
                        L1, L2 = fiber_semi_lengths(zone, lengths[fibre])
                        values[local], _, _ = _simulate_fiber_v2_python(
                            Fs=float(owner._internal_sampling_frequency__Hz * 1e-3),
                            v=float(owner._mean_conduction_velocity__m_s),
                            N=N, M=M, r=float(owner._radius_total),
                            r_bone=float(owner._radius_bone__mm),
                            th_fat=float(owner._fat_thickness__mm),
                            th_skin=float(owner._skin_thickness__mm),
                            R=radial[fibre], L1=L1, L2=L2, zi=zone,
                            electrode_array=electrode,
                            sig_muscle_rho=float(owner._muscle_conductivity_radial__S_m),
                            sig_muscle_z=float(owner._muscle_conductivity_longitudinal__S_m),
                            sig_fat=float(owner._fat_conductivity__S_m),
                            sig_skin=float(owner._skin_conductivity__S_m),
                            fiber_length__mm=kernel_length, use_gpu=False,
                            theta_offset=-angular[fibre], pos_z_precomputed=base_z,
                            pos_theta_precomputed=base_theta, rele_precomputed=radius,
                            _prepared=preparation,
                        )
                        if not np.all(np.isfinite(values[local])):
                            raise FloatingPointError(f"Nonfinite surface fibre in MU {mu_index}")
                        session.track(preparation.arrays())
                    return values
                def commit(values):
                    for value in values:
                        accumulated[:] += value
                if counts[mu_index]:
                    session.batches(int(counts[mu_index]), 4, accumulated.nbytes,
                                    compute, commit, fixed_bytes=workspace,
                                    label=f"array_{array_index}_MU_{mu_index}")
                with session.memory.wait_reserve(accumulated.nbytes * 4):
                    results[mu_index] = resample(accumulated, final_samples, axis=2)
                session.track([results[mu_index]])
                if disk is not None:
                    with session.memory.wait_reserve(results[mu_index].nbytes + 65536):
                        disk.put(key, results[mu_index])
            for mu_index in range(n_mu):
                result = results.get(mu_index)
                if result is None:
                    result = session.zeros((rows, cols, final_samples))
                with session.memory.wait_reserve(result.nbytes * 2):
                    signal = create_grid_signal(
                        np.transpose(result, (2, 0, 1)) * pq.mV,
                        grid_shape=(rows, cols), sampling_rate=rate * pq.Hz, t_start=0 * pq.s)
                session.track([signal.magnitude])
                segment = Segment(name=f"MUAP_{mu_index}")
                segment.analogsignals.append(signal)
                group.segments.append(segment)
    owner._muaps__Block = block
    return block
