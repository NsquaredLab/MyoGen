"""
GPU parity test: verify CuPy backend produces bit-identical SFAPs to NumPy.

Skipped automatically when CuPy is not installed or no CUDA GPU is available.
"""

import numpy as np
import pytest

from myogen.simulator.core.emg.intramuscular.bioelectric import (
    get_current_density,
    get_elementary_current_response,
    shift_padding,
)

# Graceful skip: pytest.importorskip handles missing CuPy, then we
# additionally check that a real CUDA device is present (e.g. Docker
# images may have CuPy installed without a GPU).
cupy = pytest.importorskip("cupy", reason="CuPy not installed")


@pytest.fixture(autouse=True)
def _require_cuda():
    """Skip all tests in this module if no CUDA device is available."""
    if not cupy.cuda.is_available():
        pytest.skip("CUDA device not available")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_inputs():
    """Return canonical test inputs matching calc_sfaps conventions."""
    t = np.linspace(0, 0.01, 200)[:, None]  # (Nt, 1)
    z = np.linspace(0, 120, 500)[:, None]    # (Nz, 1)
    zi = 60.0      # neuromuscular junction position (mm)
    L1 = 60.0      # right half-fiber length (mm)
    L2 = 60.0      # left half-fiber length (mm)
    v  = 4000.0    # conduction velocity (mm/s)
    d  = 0.055     # fiber diameter (mm)
    z_elec = 60.0  # electrode z-position (mm)
    r  = np.array([2.0])  # radial distance (mm)
    return t, z, zi, L1, L2, v, d, z_elec, r


# ── GPU Parity Tests ─────────────────────────────────────────────────────────

class TestGPUParity:
    """Verify that CuPy-accelerated bioelectric kernels match CPU output."""

    def test_current_density_parity(self):
        """get_current_density: GPU vs CPU xcorr ≈ 1.0."""
        cp = cupy

        t, z, zi, L1, L2, v, d, _, _ = _make_inputs()

        cd_cpu = get_current_density(t, z, zi, L1, L2, v, d)
        cd_gpu = get_current_density(
            cp.asarray(t), cp.asarray(z), zi, L1, L2, v, d, xp=cp,
        )
        cd_back = cp.asnumpy(cd_gpu)

        assert cd_cpu.shape == cd_back.shape
        xcorr = np.corrcoef(cd_cpu.ravel(), cd_back.ravel())[0, 1]
        rmse = float(np.sqrt(np.mean((cd_cpu - cd_back) ** 2)))
        assert xcorr > 0.999999, f"xcorr={xcorr}"
        assert rmse < 1e-12, f"RMSE={rmse}"

    def test_elementary_response_parity(self):
        """get_elementary_current_response: GPU vs CPU."""
        cp = cupy

        _, z, _, _, _, _, _, z_elec, r = _make_inputs()

        ecr_cpu = get_elementary_current_response(z, z_elec, r)
        ecr_gpu = get_elementary_current_response(
            cp.asarray(z), z_elec, cp.asarray(r), xp=cp,
        )
        ecr_back = cp.asnumpy(ecr_gpu)

        assert ecr_cpu.shape == ecr_back.shape
        xcorr = np.corrcoef(ecr_cpu.ravel(), ecr_back.ravel())[0, 1]
        assert xcorr > 0.999999, f"xcorr={xcorr}"

    def test_full_sfap_pipeline_parity(self):
        """Full SFAP chain: current_density.T @ response → same on CPU and GPU."""
        cp = cupy

        t, z, zi, L1, L2, v, d, z_elec, r = _make_inputs()

        # CPU
        cd_cpu = get_current_density(t, z, zi, L1, L2, v, d)
        ecr_cpu = get_elementary_current_response(z, z_elec, r)
        sfap_cpu = (cd_cpu.T @ ecr_cpu)[:, 0]

        # GPU
        t_g, z_g, r_g = cp.asarray(t), cp.asarray(z), cp.asarray(r)
        cd_gpu = get_current_density(t_g, z_g, zi, L1, L2, v, d, xp=cp)
        ecr_gpu = get_elementary_current_response(z_g, z_elec, r_g, xp=cp)
        sfap_gpu = cp.asnumpy((cd_gpu.T @ ecr_gpu)[:, 0])

        xcorr = np.corrcoef(sfap_cpu, sfap_gpu)[0, 1]
        rmse = float(np.sqrt(np.mean((sfap_cpu - sfap_gpu) ** 2)))
        assert xcorr > 0.999999, f"xcorr={xcorr}"
        assert rmse < 1e-12, f"RMSE={rmse}"


# ── shift_padding contract tests ─────────────────────────────────────────────

class TestShiftPadding:
    """Verify shift_padding semantics for positive, negative, and zero shifts."""

    def test_positive_shift(self):
        """Positive shift: head and tail zeroed (original MATLAB semantics)."""
        vec = np.arange(1, 11, dtype=float)  # [1..10]
        result = shift_padding(vec.copy(), 3, axis=0)
        # roll(3) wraps [8,9,10,1,2,3,4,5,6,7], then [:3]=0 and [-3:]=0
        assert np.all(result[:3] == 0), f"head not zeroed: {result[:3]}"
        assert np.all(result[-3:] == 0), f"tail not zeroed: {result[-3:]}"
        # middle values preserved
        np.testing.assert_array_equal(result[3:7], [1, 2, 3, 4])

    def test_negative_shift(self):
        """Negative shift: only tail zeroed."""
        vec = np.arange(1, 11, dtype=float)
        result = shift_padding(vec.copy(), -3, axis=0)
        # roll(-3) wraps [4,5,6,7,8,9,10,1,2,3], then [7:]=0
        assert np.all(result[-3:] == 0), f"tail not zeroed: {result[-3:]}"
        np.testing.assert_array_equal(result[:7], [4, 5, 6, 7, 8, 9, 10])

    def test_zero_shift(self):
        """Zero shift: array unchanged."""
        vec = np.arange(1, 11, dtype=float)
        result = shift_padding(vec.copy(), 0, axis=0)
        np.testing.assert_array_equal(result, vec)

    def test_gpu_parity(self):
        """shift_padding: GPU vs CPU produce identical output."""
        cp = cupy
        vec = np.random.default_rng(42).random(100)
        for sh in [-5, 0, 5, 20]:
            cpu_result = shift_padding(vec.copy(), sh, axis=0)
            gpu_result = cp.asnumpy(
                shift_padding(cp.asarray(vec.copy()), sh, axis=0, xp=cp)
            )
            np.testing.assert_array_equal(cpu_result, gpu_result, err_msg=f"sh={sh}")
