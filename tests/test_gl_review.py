"""Synthetic contract regressions from the GitLab CPU review; no biology claims."""
from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace

import numpy as np
import quantities as pq
import pytest
from neo import Block, Group, Segment, SpikeTrain

from myogen.simulator import ExecutionPolicy, SurfaceEMG
from myogen.simulator.core.emg.runtime.cache import DiskCache, fingerprint
from myogen.simulator.core.emg.runtime.memory import MemorySample
from myogen.simulator.core.emg.runtime.session import ExecutionSession
from myogen.utils.neo import create_grid_signal


@pytest.mark.parametrize("samples,rate,expected_length", [
    (256, 5885.057471264368, 90),
    (185, 10000., 38),
])
@pytest.mark.usefixtures("ample_session_memory")
def test_surface_half_sample_rounding_preserves_shape_and_time(samples, rate, expected_length, monkeypatch):
    from myogen.simulator.core.emg.surface import surface_emg
    monkeypatch.setattr(surface_emg, "HAS_CUPY", False)
    sim = object.__new__(SurfaceEMG)
    sim._MUs_to_simulate = [0]
    sim._sampling_frequency__Hz = 2048.
    sim._muaps__Block = Block()
    group, segment = Group(name="ElectrodeArray_0"), Segment(name="MUAP_0")
    sim._muaps__Block.groups.append(group)
    group.segments.append(segment)
    segment.analogsignals.append(create_grid_signal(np.zeros((samples, 1, 1)) * pq.mV,
        grid_shape=(1, 1), sampling_rate=rate * pq.Hz))
    spikes, pool = Block(), Segment(name="Pool_0")
    spikes.segments.append(pool)
    train = SpikeTrain([] * pq.s, t_stop=.01 * pq.s)
    train.sampling_period = 1 * pq.ms
    pool.spiketrains.append(train)
    reference = sim.simulate_surface_emg(spikes, verbose=False).groups[0].segments[0].analogsignals[0]
    actual = sim.simulate_surface_emg(spikes, verbose=False,
        execution=ExecutionPolicy()).groups[0].segments[0].analogsignals[0]
    assert reference.shape == (expected_length, 1)
    assert actual.shape == reference.shape
    assert actual.t_start == reference.t_start
    assert actual.t_stop == reference.t_stop
    assert actual.sampling_rate == reference.sampling_rate
    assert actual.units == reference.units
    assert actual.annotations == reference.annotations


@pytest.mark.parametrize("physical_pressure", [False, True])
def test_jobs_keep_minimum_capacity_until_the_worker_finishes(physical_pressure):
    mib = 1024**2
    started = [threading.Event(), threading.Event()]
    ready = [threading.Event(), threading.Event()]
    owner = SimpleNamespace()
    with ExecutionSession(owner, ExecutionPolicy(memory_budget_mib=None if physical_pressure else 10),
            "contract", n_jobs=2, probe=lambda: MemorySample(16*1024**3, 12*1024**3, 0)) as session:
        if physical_pressure:
            # Plenty of RAM at preflight, then only one minimum fits after
            # another application consumes RAM. Resident buffers reduce it.
            session.memory.require(18*mib)
            session.memory.probe = lambda: MemorySample(16*1024**3,
                int(4.5*1024**3) + 10*mib - session.memory.managed, 0)
            now = [0.]
            session.memory.clock = lambda: now[0]
            session.memory.sleep = lambda seconds: now.__setitem__(0, now[0] + seconds)
        def compute(index):
            started[index].set()
            started[1-index].wait(.5)
            held = session.zeros((2*mib,), dtype=np.uint8)
            ready[index].set()
            # With sufficient reservation protection only one worker starts.
            # Without it both retain 2 MiB and neither can obtain its 7 MiB batch.
            ready[1-index].wait(.5)
            session.batches(2, 1, 7*mib, lambda start, stop: np.zeros(1), lambda result: None)
            return index
        assert list(session.jobs([0, 1], compute, minimum_bytes=9*mib)) == [0, 1]
    assert owner._execution_report["completed_work"] == 4
    assert session.memory.pending == session.memory.managed == 0


def test_concurrent_same_key_writers_leave_no_unreferenced_payloads(tmp_path, monkeypatch):
    caches = [DiskCache(tmp_path, limit_bytes=1000) for _ in range(2)]
    rendezvous = threading.Barrier(2)
    first_done = threading.Event()
    first_read = threading.Event()
    for index, cache in enumerate(caches):
        original = cache._metadata
        state = [False]
        def metadata(key, original=original, state=state, index=index):
            result = original(key)
            if not state[0]:
                state[0] = True
                if index == 0:
                    first_read.set()
                # Hold the first manifest read to expose stale publication.
                # Serialized writers cannot rendezvous and continue on timeout.
                try:
                    rendezvous.wait(.5)
                except threading.BrokenBarrierError:
                    pass
                if index == 1:
                    assert first_done.wait(3)
            return result
        monkeypatch.setattr(cache, "_metadata", metadata)
    value = np.arange(64.)
    key = fingerprint("concurrent cache contract")
    def publish(index):
        if index == 1:
            assert first_read.wait(3)
        try:
            caches[index].put(key, value)
        finally:
            if index == 0:
                first_done.set()
    with ThreadPoolExecutor(2) as workers:
        list(workers.map(publish, range(2)))
    caches[0].evict()
    np.testing.assert_array_equal(caches[0].get(key), value)
    payloads = list(tmp_path.glob("*.npy"))
    assert len(payloads) == 1
    assert sum(path.stat().st_size for path in payloads) <= 1000


@pytest.mark.parametrize("stage", [".npy", ".json"])
def test_interrupted_cache_writer_recovers_only_owned_files(tmp_path, stage):
    import subprocess
    import sys
    cache = DiskCache(tmp_path)
    key = fingerprint("crash recovery contract")
    before, after = np.arange(12.), np.arange(12.) + 1
    cache.put(key, before)
    foreign = tmp_path / ("a"*32 + ".npy")
    foreign.write_bytes(b"unrelated, preserve")
    code = r"""
import os, sys
import numpy as np
from myogen.simulator.core.emg.runtime.cache import DiskCache
cache = DiskCache(sys.argv[1])
original = os.replace
def crash(source, destination):
    original(source, destination)
    if str(destination).endswith(sys.argv[3]):
        os._exit(23)
os.replace = crash
cache.put(sys.argv[2], np.arange(12.) + 1)
"""
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path), key, stage],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 23, result.stderr
    # A fresh instance must acquire the abandoned writer's lock and recover.
    restarted = DiskCache(tmp_path)
    restarted.evict()
    np.testing.assert_array_equal(restarted.get(key), after if stage == ".json" else before)
    assert foreign.read_bytes() == b"unrelated, preserve"
    assert len([p for p in tmp_path.glob("*.npy") if p != foreign]) == 1


def test_cache_writers_in_separate_processes_share_publication_lock(tmp_path):
    import subprocess
    import sys
    key = fingerprint("process publication contract")
    code = r"""
import sys
import numpy as np
from myogen.simulator.core.emg.runtime.cache import DiskCache
cache = DiskCache(sys.argv[1], limit_bytes=1000)
for _ in range(10):
    cache.put(sys.argv[2], np.arange(64.))
"""
    processes = [subprocess.Popen([sys.executable, "-c", code, str(tmp_path), key],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(2)]
    try:
        for process in processes:
            output, error = process.communicate(timeout=30)
            assert process.returncode == 0, error
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate()
    cache = DiskCache(tmp_path, limit_bytes=1000)
    cache.evict()
    np.testing.assert_array_equal(cache.get(key), np.arange(64.))
    assert len(list(tmp_path.glob("*.npy"))) == 1
    assert not list(tmp_path.glob(".myogen-*.json"))


def test_admission_credit_recycles_without_retaining_extra_batch_capacity():
    """Ledger ownership contract: retained results outlive their worker lease."""
    from myogen.simulator.core.emg.runtime.memory import MemoryController
    memory = MemoryController(ExecutionPolicy(memory_budget_mib=1),
        probe=lambda: MemorySample(16*1024**3, 12*1024**3, 0))
    root = memory.try_reserve(100)
    with memory.use_admission(root):
        batch = memory.wait_reserve(160)
        result = batch.allocate_child(40)
        assert (memory.pending, memory.managed) == (120, 40)
        batch.close()
        # Only the original 100-byte job promise survives; the extra 60 is free.
        assert (memory.pending, memory.managed) == (60, 40)
        retry = memory.wait_reserve(60)
        retry.close()
        assert (memory.pending, memory.managed) == (60, 40)
    assert (memory.pending, memory.managed) == (0, 40)
    result.close()
    assert (memory.pending, memory.managed) == (0, 0)


def test_resident_input_consumes_admission_and_returns_credit_when_released():
    from myogen.simulator.core.emg.runtime.memory import MemoryController
    memory = MemoryController(ExecutionPolicy(memory_budget_mib=1),
        probe=lambda: MemorySample(16*1024**3, 12*1024**3, 0))
    with memory.use_admission(memory.try_reserve(100)):
        resident = memory.track_resident(60)
        assert (memory.pending, memory.managed) == (40, 60)
        resident.close()
        assert (memory.pending, memory.managed) == (100, 0)
    assert (memory.pending, memory.managed) == (0, 0)
