"""Central admission ledger with physical-memory monitoring and bounded waits.

Reservations represent not-yet-resident growth. Call materialize immediately
after an owned allocation; the physical probe then accounts for that memory.
The managed ledger separately enforces an optional numerical-buffer ceiling.
"""
from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import CancelledError
import ctypes
import math
import os
import threading
import time

import psutil

MIB = 1024**2
_rss_lock = threading.Lock()
_rss_sample = (0.0, 0)
_memory_status_ex = None


class _MemoryStatus(ctypes.Structure):
    # Microsoft MEMORYSTATUSEX ABI, including all seven DWORDLONG members.
    _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32)] + [
        (name, ctypes.c_uint64) for name in (
            "ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile",
            "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")]


def windows_physical_memory(call=None):
    """Fresh physical bytes, avoiding the unused Windows commit/swap query.

    https://learn.microsoft.com/windows/win32/api/sysinfoapi/ns-sysinfoapi-memorystatusex
    ullAvailPhys includes standby, free, and zero-page lists, not swap.
    """
    global _memory_status_ex
    if call is None:
        if _memory_status_ex is None:
            function = ctypes.WinDLL("kernel32", use_last_error=True).GlobalMemoryStatusEx
            function.argtypes = [ctypes.POINTER(_MemoryStatus)]
            function.restype = ctypes.c_int
            _memory_status_ex = function
        call = _memory_status_ex
    value = _MemoryStatus()
    value.dwLength = ctypes.sizeof(value)
    if not call(ctypes.pointer(value)):
        raise OSError("GlobalMemoryStatusEx failed")
    return int(value.ullTotalPhys), int(value.ullAvailPhys)


class MemoryBudgetError(MemoryError):
    """Required storage or monitored host headroom prevents safe progress."""


@dataclass(frozen=True)
class MemorySample:
    total: int
    available: int
    rss: int


def physical_memory():
    global _rss_sample
    if os.name == "nt":
        total, available = windows_physical_memory()
    else:
        memory = psutil.virtual_memory()
        total, available = memory.total, memory.available
    # Available RAM is always fresh. Optimized workers are threads in this
    # process. Other child applications are not simulation workers and must
    # not be included (or enumerated on every short call).
    with _rss_lock:
        now = time.monotonic()
        if now - _rss_sample[0] >= 1:
            process = psutil.Process()
            rss = process.memory_info().rss
            _rss_sample = (now, rss)
        rss = _rss_sample[1]
    return MemorySample(total, available, rss)


class Reservation:
    def __init__(self, controller, amount, resident=0):
        self.controller = controller
        self.remaining = amount
        self.resident = resident
        self.closed = False

    def materialize(self, amount):
        with self.controller.lock:
            if self.closed or amount < 0 or amount > self.remaining:
                raise ValueError("Invalid allocation against reservation")
            self.remaining -= amount
            self.resident += amount
            self.controller.pending -= amount
            self.controller.managed += amount

    def allocate_child(self, amount):
        """Transfer part of this pending promise to an owned resident buffer."""
        with self.controller.lock:
            if self.closed or amount < 0 or amount > self.remaining:
                raise ValueError("Allocation exceeds remaining reservation")
            self.remaining -= amount
            self.controller.pending -= amount
            self.controller.managed += amount
            return Reservation(self.controller, 0, amount)

    def close(self):
        with self.controller.lock:
            if not self.closed:
                self.controller.pending -= self.remaining
                self.controller.managed -= self.resident
                self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class MemoryController:
    def __init__(self, policy, *, probe=physical_memory, clock=time.monotonic,
                 sleep=time.sleep):
        self.policy = policy
        self.probe = probe
        self.clock = clock
        self.sleep = sleep
        self.lock = threading.RLock()
        self.pending = 0
        self.managed = 0
        self.healthy = 0
        self.last_healthy_sample = -math.inf
        self.last_sample = None
        self.error = None
        self.evictors = []
        self.events = dict(pressure_pauses=0, allocation_retries=0, cache_hits=0,
                           cache_evictions=0, completed_work=0, wait_seconds=0.0,
                           peak_managed_bytes=0, peak_pending_bytes=0,
                           minimum_available_bytes=None, peak_process_rss_bytes=0)
        self._stop = threading.Event()
        self.cancelled = threading.Event()
        self._thread = None

    @property
    def ceiling(self):
        return (math.inf if self.policy.memory_budget_mib is None
                else int(self.policy.memory_budget_mib * MIB))

    def sample(self):
        with self.lock:
            if self.error is not None:
                raise MemoryBudgetError(f"Memory monitor failed: {self.error}")
            try:
                value = self.probe()
                if value.total <= 0 or not 0 <= value.available <= value.total or value.rss < 0:
                    raise ValueError("invalid physical memory measurement")
            except Exception as exc:
                self.error = str(exc)
                raise MemoryBudgetError(f"Memory monitor failed: {exc}") from exc
            self.last_sample = value
            self.events["physical_total_bytes"] = value.total
            reserve, margin = self.reserve_and_margin(value)
            self.events["reserve_bytes"] = reserve
            self.events["allocation_margin_bytes"] = margin
            previous = self.events["minimum_available_bytes"]
            self.events["minimum_available_bytes"] = value.available if previous is None else min(previous, value.available)
            self.events["peak_process_rss_bytes"] = max(value.rss, self.events["peak_process_rss_bytes"])
            self.events["peak_managed_bytes"] = max(self.managed, self.events["peak_managed_bytes"])
            self.events["peak_pending_bytes"] = max(self.pending, self.events["peak_pending_bytes"])
            now = self.clock()
            if self._allowance(value) > 0:
                if now - self.last_healthy_sample >= 1:
                    self.healthy += 1
                    self.last_healthy_sample = now
            else:
                self.healthy = 0
                self.last_healthy_sample = now
            return value

    def reserve_and_margin(self, value):
        reserve = max(self.policy.reserve_min_mib * MIB, self.policy.reserve_fraction * value.total)
        headroom = max(0, value.available - reserve)
        return reserve, max(512 * MIB, 0.1 * headroom)

    def _allowance(self, value):
        reserve, margin = self.reserve_and_margin(value)
        return max(0, int(min(value.available - reserve - margin - self.pending,
                              self.ceiling - self.managed - self.pending)))

    def allowance(self):
        with self.lock:
            return self._allowance(self.sample())

    def track_resident(self, amount):
        """Account for already resident, uniquely owned numerical buffers."""
        with self.lock:
            if amount < 0 or self.managed + self.pending + amount > self.ceiling:
                raise self.failure(amount)
            self.managed += amount
            self.events["peak_managed_bytes"] = max(self.managed, self.events["peak_managed_bytes"])
            return Reservation(self, 0, amount)

    def try_reserve(self, amount):
        if amount < 0:
            raise ValueError("Reservation must be nonnegative")
        with self.lock:
            if amount > self.allowance():
                return None
            self.pending += amount
            self.events["peak_pending_bytes"] = max(self.pending, self.events["peak_pending_bytes"])
            return Reservation(self, amount)

    def failure(self, amount, reason="required minimum working set cannot fit"):
        value = self.last_sample
        reserve = None if value is None else self.reserve_and_margin(value)[0]
        return MemoryBudgetError(
            f"{reason}; minimum additional bytes={amount}, managed={self.managed}, "
            f"available={None if value is None else value.available}, reserve={reserve}, "
            f"ceiling={self.policy.memory_budget_mib} MiB, "
            f"completed={self.events['completed_work']}, cache_dir={self.policy.cache_dir}"
        )

    def require(self, amount):
        """Fail before a stage starts if its smallest working set cannot fit."""
        with self.lock:
            if amount > self.allowance():
                self.evict()
                if amount > self.allowance():
                    raise self.failure(amount)

    def evict(self):
        # Eviction changes managed accounting, but NEVER fabricates an increase
        # in available physical memory. A fresh probe decides admission.
        for evictor in tuple(self.evictors):
            self.events["cache_evictions"] += evictor()

    def wait_reserve(self, amount):
        if amount + self.managed > self.ceiling:
            self.evict()
            if amount + self.managed > self.ceiling:
                raise self.failure(amount)
        start = self.clock()
        paused = False
        try:
            while True:
                if self.cancelled.is_set():
                    raise CancelledError("Execution stopped at a safe memory boundary")
                reservation = self.try_reserve(amount)
                if reservation is not None:
                    return reservation
                if not paused:
                    with self.lock:
                        self.events["pressure_pauses"] += 1
                    paused = True
                self.evict()
                if self.clock() - start >= 60:
                    raise self.failure(amount, "memory pressure exceeded 60 seconds")
                self.sleep(min(1, 60 - (self.clock() - start)))
        finally:
            if paused:
                with self.lock:
                    self.events["wait_seconds"] += self.clock() - start

    def next_batch(self, current, bytes_per_item, maximum, *, successful=True, fixed_bytes=0):
        with self.lock:
            allowance = max(0, self.allowance() - fixed_bytes)
            fits = int(allowance // max(1, bytes_per_item))
            if fits < current:
                # A brief recovery after partial pressure must not reuse the
                # healthy samples accumulated before the batch had to shrink.
                self.healthy = 0
                self.last_healthy_sample = self.clock()
                return max(0, min(fits, max(1, current // 2)))
            if successful and self.healthy >= 5:
                self.healthy = 0
                return min(maximum, fits, max(current, int(current * 1.25)))
            return min(current, fits, maximum)

    def start(self):
        self.sample()
        def monitor():
            while not self._stop.wait(1):
                try:
                    self.sample()
                except MemoryBudgetError:
                    return
        self._thread = threading.Thread(target=monitor, daemon=True, name="myogen-memory")
        self._thread.start()

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
