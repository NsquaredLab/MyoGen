"""Public import compatibility and lazy-loading contracts."""
import subprocess
import sys


def test_policy_import_does_not_eagerly_load_simulator_kernels():
    result = subprocess.run([sys.executable, "-c",
        "import sys; from myogen.simulator import ExecutionPolicy; "
        "assert 'myogen.simulator.neuron.muscle' not in sys.modules; "
        "assert 'myogen.simulator.core.emg.surface.simulate_fiber' not in sys.modules"
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_existing_public_import_names_remain_available():
    from myogen import simulator
    for name in simulator.__all__:
        assert getattr(simulator, name) is not None


def test_cpu_numerical_contract_works_with_cupy_unavailable():
    # Optional-dependency contract only; this is not a physiological fixture.
    code = """
import sys
sys.modules['cupy'] = None
import numpy as np
from types import SimpleNamespace
from myogen.simulator import ExecutionPolicy
from myogen.simulator.core.emg.runtime.session import ExecutionSession
from myogen.simulator.core.emg.runtime.synthesis import convolve_add
from tests._memory import ample_memory
with ExecutionSession(SimpleNamespace(), ExecutionPolicy(), 'optional_gpu_contract', probe=ample_memory) as session:
    spikes, kernel = np.ones(101, bool), np.arange(7.)
    output = session.zeros((101,))
    convolve_add(output, spikes, kernel, session, method='fft', block_size=16)
    np.testing.assert_allclose(output, np.convolve(spikes, kernel, 'same'), atol=1e-12, rtol=1e-9)
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
