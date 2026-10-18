"""Synthetic contract checks only; waveform evidence uses the full population."""
import numpy as np
import pytest

from myogen import get_random_generator
from myogen.simulator.core.emg.runtime import density


def inputs():
    t = np.arange(8, dtype=np.float64)[:, None] / 10000
    z = np.arange(-1, 1.01, .1)[:, None]
    t.flags.writeable = z.flags.writeable = False
    return t, z, 0., 1., 1., 4000., .055


def test_kernel_keeps_readonly_inputs_rng_shape_and_precision():
    args = inputs()
    copies = [value.copy() for value in args[:2]]
    state = get_random_generator().bit_generator.state
    result = density.current_density(*args)
    assert result.shape == (len(args[1]), len(args[0]))
    assert result.dtype == np.float64 and result.flags.c_contiguous
    for value, snapshot in zip(args, copies):
        np.testing.assert_array_equal(value, snapshot)
        assert not value.flags.writeable
    assert get_random_generator().bit_generator.state == state


def test_kernel_uses_serial_strict_arithmetic_and_releases_gil():
    assert density._density.targetoptions["fastmath"] is False
    assert not density._density.targetoptions.get("parallel", False)
    assert density._density.targetoptions["nogil"] is True


def test_allocation_failure_reaches_transactional_batch_owner(monkeypatch):
    def fail(*args):
        raise MemoryError("injected kernel allocation failure")
    monkeypatch.setattr(density, "_density", fail)
    with pytest.raises(MemoryError, match="injected kernel"):
        density.current_density(*inputs())


def test_optimized_path_uses_kernel_without_replacing_public_diagnostics():
    from myogen.simulator.core.emg.runtime import iemg
    from myogen.simulator.core.emg.intramuscular import bioelectric
    assert iemg.get_current_density is density.current_density
    assert bioelectric.get_current_density is not density.current_density
