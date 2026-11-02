"""Synthetic contract-only tests: admission/accounting, never model evidence."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError

import pytest

from myogen.simulator.core.emg.runtime import ExecutionPolicy, MemoryBudgetError
from myogen.simulator.core.emg.runtime.memory import MemoryController, MemorySample

GIB = 1024**3


class Probe:
    def __init__(self, available=12 * GIB):
        self.available = available
    def __call__(self):
        return MemorySample(16 * GIB, self.available, GIB)


def test_two_reservations_cannot_claim_same_headroom():
    controller = MemoryController(ExecutionPolicy(), probe=Probe())
    def admit(_):
        return controller.try_reserve(4 * GIB)
    with ThreadPoolExecutor(2) as workers:
        leases = list(workers.map(admit, range(2)))
    assert sum(lease is not None for lease in leases) == 1
    for lease in leases:
        if lease:
            lease.close()
    assert controller.pending == 0


def test_resident_memory_is_not_counted_again_as_pending():
    probe = Probe()
    controller = MemoryController(ExecutionPolicy(memory_budget_mib=8192), probe=probe)
    lease = controller.try_reserve(2 * GIB)
    lease.materialize(2 * GIB)
    probe.available -= 2 * GIB
    assert controller.pending == 0
    assert controller.managed == 2 * GIB
    assert controller.allowance() == pytest.approx(5.4 * GIB, abs=2)
    lease.close()
    assert controller.managed == 0


def test_ceiling_includes_required_resident_inputs():
    controller = MemoryController(ExecutionPolicy(memory_budget_mib=1024), probe=Probe())
    lease = controller.track_resident(900 * 1024**2)
    assert controller.try_reserve(200 * 1024**2) is None
    with pytest.raises(MemoryBudgetError, match="minimum"):
        controller.require(200 * 1024**2)
    lease.close()


def test_pressure_zero_admission_timeout_and_recovery_hysteresis():
    probe = Probe(4 * GIB)
    now = [0.0]
    controller = MemoryController(ExecutionPolicy(), probe=probe,
                                  clock=lambda: now[0],
                                  sleep=lambda seconds: now.__setitem__(0, now[0] + seconds))
    assert controller.try_reserve(1) is None
    with pytest.raises(MemoryBudgetError, match="60"):
        controller.wait_reserve(1024)
    assert now[0] == 60
    assert controller.events["wait_seconds"] == 60
    probe.available = 12 * GIB
    assert controller.next_batch(8, 1024, 100) <= 8
    for _ in range(5):
        now[0] += 1
        controller.sample()
    assert controller.next_batch(8, 1024, 100) == 10


def test_cancelled_pressure_wait_is_included_in_report():
    from concurrent.futures import CancelledError
    now = [0.]
    def sleep(seconds):
        now[0] += seconds
        controller.cancelled.set()
    controller = MemoryController(ExecutionPolicy(), probe=Probe(4 * GIB),
                                  clock=lambda: now[0], sleep=sleep)
    with pytest.raises(CancelledError):
        controller.wait_reserve(1024)
    assert controller.events["wait_seconds"] == 1


def test_monitor_failure_stops_admissions():
    def failed():
        raise OSError("probe unavailable")
    controller = MemoryController(ExecutionPolicy(), probe=failed)
    with pytest.raises(MemoryBudgetError, match="monitor"):
        controller.try_reserve(1)


@pytest.mark.parametrize("kwargs", [
    {"backend": "cuda"}, {"max_workers": 0}, {"reserve_fraction": -1},
    {"memory_budget_mib": 0}, {"memory_budget_mib": float("nan")},
])
def test_invalid_policy_rejected(kwargs):
    with pytest.raises(ValueError):
        ExecutionPolicy(**kwargs)


def test_policy_is_immutable():
    policy = ExecutionPolicy()
    with pytest.raises(FrozenInstanceError):
        policy.max_workers = 20


def test_windows_physical_probe_uses_available_bytes_and_fails_closed():
    from myogen.simulator.core.emg.runtime.memory import windows_physical_memory
    def success(pointer):
        assert pointer.contents.dwLength == 64
        pointer.contents.ullTotalPhys = 16 * GIB
        pointer.contents.ullAvailPhys = 7 * GIB
        pointer.contents.ullAvailPageFile = 80 * GIB
        return 1
    assert windows_physical_memory(success) == (16 * GIB, 7 * GIB)
    with pytest.raises(OSError):
        windows_physical_memory(lambda _: 0)


def test_thread_workers_share_parent_rss_and_exclude_unrelated_child_processes(monkeypatch):
    from types import SimpleNamespace
    from myogen.simulator.core.emg.runtime import memory
    class Process:
        def memory_info(self):
            return SimpleNamespace(rss=12345)
        def children(self, **kwargs):
            raise AssertionError("Other child applications are not CPU thread workers")
    monkeypatch.setattr(memory.psutil, "Process", Process)
    monkeypatch.setattr(memory, "_rss_sample", (0., 0))
    assert memory.physical_memory().rss == 12345


@pytest.mark.parametrize("fixed", [0, 256 * 1024**2])
def test_partial_pressure_restarts_growth_hysteresis(fixed):
    probe, now = Probe(), [0.]
    controller = MemoryController(ExecutionPolicy(), probe=probe, clock=lambda: now[0])
    for second in range(6):
        now[0] = float(second)
        controller.sample()
    probe.available = 5 * GIB  # Positive headroom, but the previous batch no longer fits.
    smaller = controller.next_batch(8, 128 * 1024**2, 100, fixed_bytes=fixed)
    assert smaller < 8
    probe.available = 12 * GIB
    assert controller.next_batch(smaller, 128 * 1024**2, 100, fixed_bytes=fixed) == smaller
