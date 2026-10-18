"""Synthetic contract-only tests of retry and reporting."""
import numpy as np
import pytest

from myogen.simulator.core.emg.runtime import ExecutionPolicy, MemoryBudgetError
from myogen.simulator.core.emg.runtime.session import ExecutionSession
from myogen.simulator.core.emg.runtime.memory import MemorySample


class Owner:
    pass


def test_retry_does_not_commit_twice_or_draw_random_numbers():
    owner = Owner()
    values = np.arange(9.0)
    attempts = []
    committed = []
    rng = np.random.default_rng(123)
    before = rng.bit_generator.state
    def compute(start, stop):
        attempts.append((start, stop))
        if stop - start > 2:
            raise MemoryError("injected allocator failure")
        return values[start:stop].copy()
    with ExecutionSession(owner, ExecutionPolicy(), "contract", n_jobs=1) as session:
        session.batches(9, 8, 128, compute, lambda result: committed.extend(result))
    assert committed == list(values)
    assert attempts[:3] == [(0, 8), (0, 4), (0, 2)]
    assert rng.bit_generator.state == before
    assert owner._execution_report["allocation_retries"] == 2
    assert owner._execution_report["status"] == "complete"
    with pytest.raises(TypeError):
        owner._execution_report["status"] = "failed"


def test_smallest_batch_failure_is_reported_without_commit():
    owner = Owner()
    committed = []
    def compute(*_):
        raise MemoryError("cannot fit")
    with pytest.raises(MemoryBudgetError, match="smallest"):
        with ExecutionSession(owner, ExecutionPolicy(), "contract") as session:
            session.batches(2, 2, 128, compute, committed.append)
    assert committed == []
    assert owner._execution_report["status"] == "failed"


def test_failure_in_probe_still_publishes_report():
    owner = Owner()
    def broken():
        raise OSError("test probe")
    with pytest.raises(MemoryBudgetError):
        with ExecutionSession(owner, ExecutionPolicy(), "contract", probe=broken):
            pass
    assert owner._execution_report["status"] == "failed"


def test_output_rejected_before_allocation():
    owner = Owner()
    with pytest.raises(MemoryBudgetError):
        with ExecutionSession(owner, ExecutionPolicy(memory_budget_mib=1), "contract") as session:
            session.zeros((2**20,))
    assert owner._execution_report["completed_work"] == 0


def test_dispatch_is_bounded_and_results_keep_input_order():
    import threading
    import time
    owner = Owner()
    active = [0, 0]
    lock = threading.Lock()
    def work(value):
        with lock:
            active[0] += 1
            active[1] = max(active[0], active[1])
        time.sleep(0.02 * (5 - value))
        with lock:
            active[0] -= 1
        return value * 3
    with ExecutionSession(owner, ExecutionPolicy(), "contract", n_jobs=-1) as session:
        results = list(session.jobs(range(5), work, minimum_bytes=1024))
    assert results == [0, 3, 6, 9, 12]
    assert active[1] == 2


def test_alias_tracking_does_not_charge_shared_buffer_twice():
    owner = Owner()
    with ExecutionSession(owner, ExecutionPolicy(), "contract") as session:
        array = session.zeros((64,))
        charged = session.memory.managed
        session.track([array, array[4:]])
        assert session.memory.managed == charged


def test_monitor_failure_at_last_boundary_prevents_success_report():
    owner = Owner()
    with pytest.raises(MemoryBudgetError, match="late failure"):
        with ExecutionSession(owner, ExecutionPolicy(), "contract") as session:
            session.memory.error = "late failure"
    assert owner._execution_report["status"] == "failed"


def test_native_thread_limit_survives_nested_sessions():
    from threadpoolctl import threadpool_info
    outer, inner = Owner(), Owner()
    with ExecutionSession(outer, ExecutionPolicy(), "contract"):
        with ExecutionSession(inner, ExecutionPolicy(), "contract"):
            np.ones((2, 2)) @ np.ones((2, 2))
        assert all(pool["num_threads"] == 1 for pool in threadpool_info())


def test_failing_worker_stops_sibling_at_next_batch_boundary():
    import threading
    import time
    owner = Owner()
    ready = threading.Event()
    committed = []
    with pytest.raises(ValueError, match="failed worker"):
        with ExecutionSession(owner, ExecutionPolicy(), "contract", n_jobs=2) as session:
            def compute(item):
                if item == 0:
                    assert ready.wait(3)
                    raise ValueError("failed worker")
                def batch(start, stop):
                    ready.set()
                    time.sleep(.02)
                    return np.arange(start, stop)
                session.batches(20, 1, 8, batch, lambda result: committed.extend(result))
            list(session.jobs([0, 1], compute, minimum_bytes=8))
    assert len(committed) <= 1


def test_unused_warm_cache_is_evicted_for_tighter_ceiling():
    from myogen.simulator.core.emg.runtime.cache import ArrayCache
    owner = Owner()
    owner._basic_preparation_cache = ArrayCache()
    owner._basic_preparation_cache.put("unused", np.zeros(2 * 1024**2, dtype=np.uint8))
    with ExecutionSession(owner, ExecutionPolicy(memory_budget_mib=1), "contract") as session:
        assert session.zeros((1024,), dtype=np.uint8).nbytes == 1024
    assert not owner._basic_preparation_cache.entries


def test_unused_cache_is_evicted_when_new_output_needs_its_space():
    from myogen.simulator.core.emg.runtime.cache import ArrayCache
    owner = Owner()
    owner._basic_preparation_cache = ArrayCache()
    owner._basic_preparation_cache.put("unused", np.zeros(1000 * 1024, dtype=np.uint8))
    with ExecutionSession(owner, ExecutionPolicy(memory_budget_mib=1), "contract") as session:
        assert session.zeros((128 * 1024,), dtype=np.uint8).nbytes == 128 * 1024
    assert not owner._basic_preparation_cache.entries


def test_later_worker_failure_keeps_original_error_after_sibling_cancellation():
    import threading
    import time
    ready = threading.Event()
    with pytest.raises(ValueError, match="later worker failed"):
        with ExecutionSession(Owner(), ExecutionPolicy(), "contract", n_jobs=2) as session:
            def compute(item):
                if item == 1:
                    assert ready.wait(3)
                    raise ValueError("later worker failed")
                def batch(start, stop):
                    ready.set()
                    time.sleep(.02)
                    return np.arange(start, stop)
                session.batches(20, 1, 8, batch, lambda result: None)
            list(session.jobs([0, 1], compute, minimum_bytes=8))


def test_zero_buffer_is_touched_before_resident_accounting():
    """Bounded 64-MiB allocation mechanics in a fresh process, no model."""
    import json
    import subprocess
    import sys
    code = """
import json, psutil, numpy as np
from types import SimpleNamespace
from myogen.simulator import ExecutionPolicy
from myogen.simulator.core.emg.runtime.session import ExecutionSession
with ExecutionSession(SimpleNamespace(), ExecutionPolicy(), 'resident_contract') as session:
    process = psutil.Process()
    before = process.memory_info().rss
    value = session.zeros((64*1024**2,), dtype=np.uint8)
    resident = process.memory_info().rss - before
    print(json.dumps({'resident': resident, 'size': value.nbytes, 'pending': session.memory.pending}))
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    values = json.loads(result.stdout.strip().splitlines()[-1])
    assert values["resident"] >= .75 * values["size"]
    assert values["pending"] == 0
