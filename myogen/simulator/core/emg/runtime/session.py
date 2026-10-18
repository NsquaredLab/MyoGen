"""Per-call lifecycle, transactional batches, and immutable execution reports."""
from __future__ import annotations

from dataclasses import asdict
from collections import deque
from concurrent.futures import CancelledError, ThreadPoolExecutor
import os
import sys
import threading
import time
import weakref
from types import MappingProxyType

import numpy as np
from threadpoolctl import ThreadpoolController

from .memory import MemoryBudgetError, MemoryController
from .policy import ExecutionPolicy

_native_lock = threading.RLock()
_native_users = 0
_native_limits = None
_native_controller = None
_native_module_count = 0


def enter_native_limit():
    global _native_users, _native_limits, _native_controller, _native_module_count
    with _native_lock:
        if _native_users == 0:
            # Re-enumerate DLLs when imports may have introduced a numerical
            # backend, not on every short recording with the same modules.
            if _native_controller is None or len(sys.modules) != _native_module_count:
                _native_controller = ThreadpoolController()
                _native_module_count = len(sys.modules)
            _native_limits = _native_controller.limit(limits=1)
        _native_users += 1


def leave_native_limit():
    global _native_users, _native_limits
    with _native_lock:
        _native_users -= 1
        if _native_users == 0:
            _native_limits.restore_original_limits()
            _native_limits = None


def readonly(value):
    if isinstance(value, dict):
        return MappingProxyType({key: readonly(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(readonly(item) for item in value)
    return value


class ExecutionSession:
    def __init__(self, owner, policy, stage, *, n_jobs=1, probe=None, native_pools=True):
        if not isinstance(policy, ExecutionPolicy):
            raise TypeError("execution must be an ExecutionPolicy or None")
        self.owner, self.policy, self.stage = owner, policy, stage
        self.native_pools = native_pools
        self.memory = MemoryController(policy, **({} if probe is None else {"probe": probe}))
        if n_jobs == 0:
            raise ValueError("n_jobs=0 is invalid")
        requested = n_jobs if n_jobs > 0 else max(1, (os.cpu_count() or 1) + 1 + n_jobs)
        self.workers = min(2, policy.max_workers, requested)
        self.worker_changes = [self.workers]
        self.batch_sizes = {}
        self.convolution_methods = {}
        self.leases = []
        self._tracked = {}
        self._local = threading.local()
        self._failure = None
        self.started = time.perf_counter()
        self.limits = None
        self.numba_threads = None

    def __enter__(self):
        try:
            self.memory.start()
            if self.native_pools:
                import numba
                self.numba_threads = numba.get_num_threads()
                numba.set_num_threads(1)
                enter_native_limit()
                self.limits = True
            from .cache import ArrayCache
            cache = getattr(self.owner, "_basic_preparation_cache", None)
            if cache is None:
                cache = ArrayCache()
                self.owner._basic_preparation_cache = cache
            self.cache = cache
            self.memory.evictors.append(cache.clear)
            if sum(value.nbytes for value in cache.entries.values()) > self.memory.ceiling:
                self.memory.evict()
            self.track(list(cache.entries.values()))
            return self
        except BaseException as exc:
            self.__exit__(type(exc), exc, exc.__traceback__)
            raise

    def __exit__(self, kind, exc, traceback):
        self.memory.close()
        final_error = None
        if exc is None:
            try:
                self.memory.sample()
            except MemoryBudgetError as error:
                final_error = exc = error
        if self.numba_threads is not None:
            import numba
            numba.set_num_threads(self.numba_threads)
        if self.limits is not None:
            leave_native_limit()
        for lease in self.leases:
            lease.close()
        self.owner._execution_report = readonly({
            "policy": asdict(self.policy), "backend": "cpu", "stage": self.stage,
            "stage_seconds": {self.stage: time.perf_counter() - self.started},
            "status": "complete" if exc is None else "failed",
            "error": None if exc is None else str(exc),
            "batch_sizes": self.batch_sizes, "worker_count_changes": self.worker_changes,
            "rss_note": "CPU workers are threads; parent-process RSS counts their shared memory once",
            **self.memory.events,
        })
        if final_error is not None:
            raise final_error

    def zeros(self, shape, dtype=np.float64):
        amount = int(np.prod(shape)) * np.dtype(dtype).itemsize
        def create():
            # calloc/np.zeros may reserve virtual pages without making them
            # resident. Touch the useful buffer while its growth promise is
            # still pending, before the physical probe takes over accounting.
            result = np.empty(shape, dtype=dtype)
            result.fill(0)
            return result
        return self.array(create, amount)

    def array(self, create, amount):
        """Reserve a known peak and retain accounting for the owning array."""
        lease = self.memory.wait_reserve(amount)
        try:
            result = create()
            allocation = result
            while isinstance(allocation.base, np.ndarray):
                allocation = allocation.base
            child = lease.allocate_child(allocation.nbytes)
            self.leases.append(child)
            self._register(allocation, child)
            return result
        finally:
            lease.close()

    def check_cancelled(self):
        if self.memory.cancelled.is_set():
            raise CancelledError("Execution stopped at a safe batch boundary")

    def _register(self, array, lease):
        self._tracked[id(array)] = weakref.ref(array)
        weakref.finalize(array, lease.close)

    def snapshot(self, array):
        array = np.asarray(array)
        result = self.zeros(array.shape, array.dtype)
        np.copyto(result, array)
        result.flags.writeable = False
        return result

    def track(self, arrays):
        """Count each owning allocation only once, including views."""
        with self.memory.lock:
            for array in arrays:
                value = np.asarray(array)
                while isinstance(value.base, np.ndarray):
                    value = value.base
                previous = self._tracked.get(id(value))
                if previous is None or previous() is not value:
                    pending = getattr(self._local, "reservation", None)
                    if pending is not None and not pending.closed and pending.remaining >= value.nbytes:
                        lease = pending.allocate_child(value.nbytes)
                    else:
                        lease = self.memory.track_resident(value.nbytes)
                    self.leases.append(lease)
                    self._register(value, lease)

    def jobs(self, items, compute, *, minimum_bytes):
        """Admit at most two outer jobs, with no unbounded prefetch queue.

        Threads share immutable inputs and the central memory controller.
        The admitted minimum stays protected until completion; allocations
        consume that promise instead of charging the same capacity twice.
        """
        def run(item, admission):
            with self.memory.use_admission(admission):
                import numba
                previous = numba.get_num_threads()
                numba.set_num_threads(1)
                try:
                    self.check_cancelled()
                    return compute(item)
                except BaseException as error:
                    with self.memory.lock:
                        if self._failure is None and not isinstance(error, CancelledError):
                            self._failure = error
                        self.memory.cancelled.set()
                    raise
                finally:
                    numba.set_num_threads(previous)
        iterator = iter(items)
        queue = deque()
        marker = object()
        item = next(iterator, marker)
        with ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="myogen-cpu") as executor:
            try:
                while item is not marker or queue:
                    while item is not marker and len(queue) < self.workers:
                        self.check_cancelled()
                        amount = minimum_bytes(item) if callable(minimum_bytes) else minimum_bytes
                        admission = self.memory.try_reserve(amount)
                        if admission is None:
                            self.memory.evict()
                            if queue:
                                break
                            admission = self.memory.wait_reserve(amount)
                        try:
                            future = executor.submit(run, item, admission)
                            future.add_done_callback(lambda future, token=admission:
                                                     token.close() if future.cancelled() else None)
                            queue.append(future)
                        except BaseException:
                            admission.close()
                            raise
                        item = next(iterator, marker)
                    if queue:
                        yield queue.popleft().result()
            except BaseException as error:
                self.memory.cancelled.set()
                if isinstance(error, CancelledError) and self._failure is not None:
                    raise self._failure from error
                raise
            finally:
                for future in queue:
                    future.cancel()

    def batches(self, count, initial, bytes_per_item, compute, commit, *, label="fibres", fixed_bytes=0):
        batch = min(count, max(1, initial))
        if bytes_per_item + fixed_bytes + self.memory.managed > self.memory.ceiling:
            self.memory.evict()
            if bytes_per_item + fixed_bytes + self.memory.managed > self.memory.ceiling:
                raise self.memory.failure(bytes_per_item + fixed_bytes)
        self.batch_sizes[label] = {"initial": batch, "final": batch}
        offset = 0
        while offset < count:
            self.check_cancelled()
            proposed = self.memory.next_batch(batch, bytes_per_item, count - offset,
                                              fixed_bytes=fixed_bytes)
            if proposed == 0:
                self.memory.evict()
                self.workers = 1
                if self.worker_changes[-1] != 1:
                    self.worker_changes.append(1)
                proposed = 1
            batch = min(proposed, count - offset)
            lease = self.memory.wait_reserve(fixed_bytes + batch * bytes_per_item)
            try:
                self._local.reservation = lease
                result = compute(offset, offset + batch)
                if isinstance(result, np.ndarray):
                    self.track([result])
            except MemoryBudgetError:
                raise
            except MemoryError as exc:
                self.memory.events["allocation_retries"] += 1
                if batch == 1:
                    raise self.memory.failure(bytes_per_item, "allocation failed at smallest batch") from exc
                batch = max(1, batch // 2)
                self.memory.healthy = 0
                continue
            finally:
                self._local.reservation = None
                lease.close()
            # A retry is possible only before this boundary. Exceptions in
            # commit propagate; repeating a partially committed result is unsafe.
            self.check_cancelled()
            commit(result)
            del result
            offset += batch
            self.memory.events["completed_work"] += batch
            self.batch_sizes[label]["final"] = batch
