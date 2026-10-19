"""Synthetic contract-only convolution boundaries; not biological evidence."""
import numpy as np
import pytest

from myogen.simulator.core.emg.runtime.synthesis import convolve_add
from myogen.simulator.core.emg.runtime.session import ExecutionSession
from myogen.simulator.core.emg.runtime import ExecutionPolicy, MemoryBudgetError
from tests._memory import ample_memory


def test_float_method_estimate_reused_by_shape_but_integer_values_rechecked(monkeypatch):
    """Dispatch metadata contract only; no model or waveform evidence."""
    from types import SimpleNamespace
    import myogen.simulator.core.emg.runtime.synthesis as module
    calls = []
    monkeypatch.setattr(module, "choose_conv_method", lambda *args, **kwargs: calls.append(1) or "direct")
    with ExecutionSession(SimpleNamespace(), ExecutionPolicy(), "contract", probe=ample_memory) as session:
        output, spikes = session.zeros((64,)), np.ones(64, bool)
        for amplitude in (1., 2.):
            convolve_add(output, spikes, np.full(7, amplitude), session)
        assert len(calls) == 1
        convolve_add(output, spikes, np.ones(8), session)
        assert len(calls) == 2
        for amplitude in (1, 2):
            convolve_add(output, spikes, np.full(7, amplitude, dtype=np.int64), session)
        assert len(calls) == 4


def test_auto_direct_admission_uses_one_fresh_physical_probe(monkeypatch):
    """Atomic admission contract; no physiological model."""
    from types import SimpleNamespace
    from myogen.simulator.core.emg.runtime.memory import MemorySample
    import myogen.simulator.core.emg.runtime.synthesis as module
    calls = []
    def probe():
        calls.append(1)
        return MemorySample(16 * 1024**3, 12 * 1024**3, 1024**3)
    monkeypatch.setattr(module, "choose_conv_method", lambda *args, **kwargs: "direct")
    with ExecutionSession(SimpleNamespace(), ExecutionPolicy(), "contract", probe=probe) as session:
        output = session.zeros((64,))
        session.memory.close()
        calls.clear()
        convolve_add(output, np.ones(64, bool), np.ones(7), session)
        assert len(calls) == 1


def test_fft_uses_one_worker_even_with_external_parallel_default(monkeypatch):
    """Numerical-worker dispatch contract, independent of any model."""
    from types import SimpleNamespace
    from scipy.fft import set_workers
    import myogen.simulator.core.emg.runtime.synthesis as module
    workers = []
    for name in ("rfft", "irfft"):
        original = getattr(module, name)
        def measured(*args, _original=original, **kwargs):
            workers.append(kwargs.get("workers"))
            return _original(*args, **kwargs)
        monkeypatch.setattr(module, name, measured)
    with set_workers(4), ExecutionSession(SimpleNamespace(), ExecutionPolicy(), "contract", probe=ample_memory) as session:
        output = session.zeros((64,))
        convolve_add(output, np.ones(64, bool), np.ones(7), session, method="fft")
    assert workers and set(workers) == {1}


@pytest.mark.parametrize("length,kernel_length", [(1, 1), (2, 9), (9, 2), (8, 4), (9, 5), (37, 12), (64, 31)])
@pytest.mark.parametrize("method", ["direct", "fft"])
def test_same_crop_overlap_and_first_last_events(length, kernel_length, method):
    spikes = np.zeros(length, dtype=bool)
    spikes[[0, -1]] = True
    kernel = np.random.default_rng(123).normal(size=kernel_length)
    expected = np.convolve(spikes, kernel, mode="same")
    class Owner:
        pass
    with ExecutionSession(Owner(), ExecutionPolicy(), "contract", probe=ample_memory) as session:
        out = session.zeros((max(length, kernel_length),))
        convolve_add(out, spikes, kernel, session, method=method, block_size=3)
        np.testing.assert_allclose(out, expected, atol=1e-12, rtol=1e-9)


def test_fft_zero_activity_stays_exactly_zero():
    class Owner:
        pass
    with ExecutionSession(Owner(), ExecutionPolicy(), "contract", probe=ample_memory) as session:
        out = session.zeros((40,))
        convolve_add(out, np.zeros(40, bool), np.ones(7), session, method="fft", block_size=4)
        np.testing.assert_array_equal(out, 0)


def test_direct_boolean_cast_and_result_must_both_fit_ceiling():
    """Allocation mechanics only, independent of physiological templates."""
    class Owner:
        pass
    spikes = np.ones(1_000_000, dtype=bool)
    kernel = np.ones(3)
    with ExecutionSession(Owner(), ExecutionPolicy(memory_budget_mib=17), "contract", probe=ample_memory) as session:
        session.track([spikes, kernel])
        output = session.zeros(spikes.shape)
        with pytest.raises(MemoryBudgetError):
            convolve_add(output, spikes, kernel, session, method="direct")
        assert not np.any(output)


def test_auto_uses_bounded_fft_when_direct_promotion_cannot_fit():
    class Owner:
        pass
    spikes = np.ones(1_000_000, dtype=bool)
    kernel = np.ones(3)
    with ExecutionSession(Owner(), ExecutionPolicy(memory_budget_mib=17), "contract", probe=ample_memory) as session:
        session.track([spikes, kernel])
        output = session.zeros(spikes.shape)
        convolve_add(output, spikes, kernel, session)
        np.testing.assert_allclose(output, np.convolve(spikes, kernel, "same"), atol=1e-12, rtol=1e-9)


def test_fft_retry_after_committed_block_never_duplicates_contribution(monkeypatch):
    import myogen.simulator.core.emg.runtime.synthesis as module
    class Owner:
        pass
    original = module.irfft
    attempts = []
    def fail_second(*args, **kwargs):
        attempts.append(1)
        if len(attempts) == 2:
            raise MemoryError("injected second block allocation failure")
        return original(*args, **kwargs)
    monkeypatch.setattr(module, "irfft", fail_second)
    spikes = np.ones(31, bool)
    kernel = np.arange(7.0)
    owner = Owner()
    with ExecutionSession(owner, ExecutionPolicy(), "contract", probe=ample_memory) as session:
        output = session.zeros(spikes.shape)
        convolve_add(output, spikes, kernel, session, method="fft", block_size=8)
        np.testing.assert_allclose(output, np.convolve(spikes, kernel, "same"), atol=1e-12, rtol=1e-9)
    assert owner._execution_report["allocation_retries"] == 1


def test_failed_fft_releases_local_transform_before_tighter_retry(monkeypatch):
    """Allocator/cache ownership contract, with no physiological templates."""
    import myogen.simulator.core.emg.runtime.synthesis as module
    from types import SimpleNamespace
    original = module.irfft
    attempted = []
    def fail_first(*args, **kwargs):
        if not attempted:
            attempted.append(True)
            raise MemoryError("injected failure after preparing kernel")
        return original(*args, **kwargs)
    monkeypatch.setattr(module, "irfft", fail_first)
    spikes, kernel = np.ones(32, bool), np.ones(1000)
    with ExecutionSession(SimpleNamespace(), ExecutionPolicy(memory_budget_mib=.078), "contract", probe=ample_memory) as session:
        session.track([spikes, kernel])
        output = session.zeros((1000,))
        convolve_add(output, spikes, kernel, session, method="fft", block_size=2)
        np.testing.assert_allclose(output, np.convolve(spikes, kernel, "same"), atol=1e-12, rtol=1e-9)

def test_fft_key_reused_within_call_and_rebuilt_after_input_change(monkeypatch):
    """Hash/cache contract only; blocks still perform fresh memory admission."""
    from types import SimpleNamespace
    import myogen.simulator.core.emg.runtime.synthesis as module
    calls = []
    original = module.fingerprint
    def measured(*args):
        calls.append(args[2])
        return original(*args)
    monkeypatch.setattr(module, "fingerprint", measured)
    spikes, kernel = np.ones(71, bool), np.arange(9.)
    with ExecutionSession(SimpleNamespace(), ExecutionPolicy(), "contract", probe=ample_memory) as session:
        output = session.zeros(spikes.shape)
        for change in (0., .25):
            kernel += change
            output.fill(0)
            calls.clear()
            convolve_add(output, spikes, kernel, session, method="fft", block_size=8)
            assert len(calls) == 1
            np.testing.assert_allclose(output, np.convolve(spikes, kernel, "same"),
                                       rtol=1e-12, atol=1e-12)


def test_fft_key_changes_when_retry_shrinks_transform(monkeypatch):
    """Retry/cache contract only; no model waveform claim."""
    from types import SimpleNamespace
    import myogen.simulator.core.emg.runtime.synthesis as module
    hashes, attempts = [], []
    fingerprint, irfft = module.fingerprint, module.irfft
    def measured(*args):
        hashes.append(args[2])
        return fingerprint(*args)
    def fail_second(*args, **kwargs):
        attempts.append(1)
        if len(attempts) == 2:
            raise MemoryError("injected FFT retry")
        return irfft(*args, **kwargs)
    monkeypatch.setattr(module, "fingerprint", measured)
    monkeypatch.setattr(module, "irfft", fail_second)
    spikes, kernel = np.ones(71, bool), np.arange(9.)
    with ExecutionSession(SimpleNamespace(), ExecutionPolicy(), "contract", probe=ample_memory) as session:
        output = session.zeros(spikes.shape)
        convolve_add(output, spikes, kernel, session, method="fft", block_size=16)
        assert len(hashes) == 2 and hashes[0] != hashes[1]
        np.testing.assert_allclose(output, np.convolve(spikes, kernel, "same"),
                                   rtol=1e-12, atol=1e-12)
