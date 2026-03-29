# Unified Fiber Simulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fragile frequency-domain sEMG source model with a time-domain traveling-wave approach, unifying iEMG and sEMG under one Rosenfalck parameterization.

**Architecture:** New module `fiber_simulation.py` with three layers: (1) shared Rosenfalck source, (2) volume conductor kernel constructors (surface Bessel or intramuscular Green's function), (3) traveling-wave convolution. Coexists with old code via `use_unified` flag.

**Tech Stack:** NumPy, SciPy (Bessel functions, signal processing), existing MyoGen electrode/muscle infrastructure.

**Spec:** `docs/superpowers/specs/2026-03-29-unified-fiber-simulation-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `myogen/simulator/core/emg/fiber_simulation.py` | Create | All three layers: source, kernels, signal computation |
| `tests/test_fiber_simulation.py` | Create | Unit and integration tests for the new module |
| `myogen/simulator/core/emg/surface/surface_emg.py` | Modify | Add `use_unified` flag, wire new path into `simulate_muaps()` |
| `pyproject.toml` | Modify | Add pytest to dev dependencies |

---

### Task 1: Test infrastructure

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/test_fiber_simulation.py`

- [ ] **Step 1: Add pytest dependency**

In `pyproject.toml`, add pytest to the dev dependency group:

```toml
[dependency-groups]
dev = [
    "pandas-stubs",
    "poethepoet",
    "scipy-stubs",
    "pytest>=7.0",
]
```

- [ ] **Step 2: Install and verify pytest**

Run:
```bash
cd /Users/oj98yqyk/code/MyoGen && uv sync --group dev
.venv/bin/python -m pytest --version
```
Expected: pytest version prints successfully.

- [ ] **Step 3: Create test file with a smoke test**

Create `tests/test_fiber_simulation.py`:

```python
"""Tests for the unified fiber simulation module."""

import numpy as np
import pytest


def test_import():
    """Verify the module can be imported."""
    from myogen.simulator.core.emg.fiber_simulation import rosenfalck_dVm_dz
    assert callable(rosenfalck_dVm_dz)
```

- [ ] **Step 4: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_fiber_simulation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'myogen.simulator.core.emg.fiber_simulation'`

- [ ] **Step 5: Create empty module**

Create `myogen/simulator/core/emg/fiber_simulation.py`:

```python
"""
Unified fiber simulation module.

Combines iEMG and sEMG fiber simulation under a single Rosenfalck source
model (D1=96, z in physical mm) with volume conductor kernels computed
as 1D spatial impulse responses.

References
----------
.. [1] Rosenfalck, P., 1969. Intra- and extracellular potential fields of active
       nerve and muscle fibres. Acta Physiol. Scand. Suppl. 321, 1-168.
.. [2] Farina, D. et al., 2004. A surface EMG generation model with multilayer
       cylindrical description of the volume conductor. IEEE TBME 51(3), 415-426.
"""

import numpy as np


def rosenfalck_dVm_dz(z: np.ndarray, D1: float = 96.0) -> np.ndarray:
    """Placeholder."""
    raise NotImplementedError
```

- [ ] **Step 6: Run test to verify it passes import but fails on call**

Run: `.venv/bin/python -m pytest tests/test_fiber_simulation.py -v`
Expected: PASS (test only checks `callable`, not invocation).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml tests/test_fiber_simulation.py myogen/simulator/core/emg/fiber_simulation.py
git commit -m "chore: add pytest, create test and module scaffolds for unified fiber simulation"
```

---

### Task 2: Layer 1 — Rosenfalck source model

**Files:**
- Modify: `myogen/simulator/core/emg/fiber_simulation.py`
- Modify: `tests/test_fiber_simulation.py`

- [ ] **Step 1: Write tests for rosenfalck_dVm_dz**

Add to `tests/test_fiber_simulation.py`:

```python
from myogen.simulator.core.emg.fiber_simulation import rosenfalck_dVm_dz


class TestRosenfalckDVmDz:
    """Tests for the raw Rosenfalck first derivative."""

    def test_zero_for_negative_z(self):
        z = np.array([-5.0, -1.0, -0.01])
        result = rosenfalck_dVm_dz(z)
        np.testing.assert_array_equal(result, 0.0)

    def test_zero_at_z_zero(self):
        result = rosenfalck_dVm_dz(np.array([0.0]))
        assert result[0] == 0.0

    def test_peak_near_1_27mm(self):
        """Rosenfalck dVm/dz peaks at z ≈ 3 - sqrt(3) ≈ 1.27 mm."""
        z = np.linspace(0.01, 5.0, 1000)
        result = rosenfalck_dVm_dz(z)
        peak_z = z[np.argmax(result)]
        assert abs(peak_z - 1.27) < 0.05

    def test_decays_to_zero_at_large_z(self):
        result = rosenfalck_dVm_dz(np.array([20.0, 50.0]))
        assert np.all(np.abs(result) < 1e-4)

    def test_matches_analytical_formula(self):
        """D1 * (3z^2 - z^3) * exp(-z) for z > 0."""
        z = np.array([0.5, 1.0, 2.0, 3.0])
        expected = 96.0 * (3 * z**2 - z**3) * np.exp(-z)
        result = rosenfalck_dVm_dz(z)
        np.testing.assert_allclose(result, expected)

    def test_custom_D1(self):
        z = np.array([1.0])
        result = rosenfalck_dVm_dz(z, D1=48.0)
        expected = 48.0 * (3 * 1.0 - 1.0) * np.exp(-1.0)
        np.testing.assert_allclose(result, expected)

    def test_matches_existing_iemg_function(self):
        """Must produce identical output to bioelectric.get_tm_current_dz."""
        from myogen.simulator.core.emg.intramuscular.bioelectric import get_tm_current_dz

        z = np.linspace(0.01, 10.0, 500)
        unified = rosenfalck_dVm_dz(z, D1=96.0)
        existing = get_tm_current_dz(z, D1=96.0)
        np.testing.assert_allclose(unified, existing, rtol=1e-12)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_fiber_simulation.py::TestRosenfalckDVmDz -v`
Expected: All FAIL with `NotImplementedError`.

- [ ] **Step 3: Implement rosenfalck_dVm_dz**

Replace the placeholder in `myogen/simulator/core/emg/fiber_simulation.py`:

```python
def rosenfalck_dVm_dz(z: np.ndarray, D1: float = 96.0) -> np.ndarray:
    """
    First spatial derivative of the Rosenfalck transmembrane potential.

    Computes dVm/dz = D1 * (3z^2 - z^3) * exp(-z) for z > 0, else 0.
    z is in physical millimeters with no artificial scaling.

    Parameters
    ----------
    z : np.ndarray
        Spatial coordinates along fiber in mm.
    D1 : float, default=96.0
        Amplitude parameter in mV/mm^3 (Rosenfalck 1969).

    Returns
    -------
    np.ndarray
        First derivative of transmembrane potential (mV/mm).
    """
    result = np.zeros_like(z, dtype=np.float64)
    pos = z > 0
    zp = z[pos]
    result[pos] = D1 * (3.0 * zp**2 - zp**3) * np.exp(-zp)
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_fiber_simulation.py::TestRosenfalckDVmDz -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add myogen/simulator/core/emg/fiber_simulation.py tests/test_fiber_simulation.py
git commit -m "feat: add rosenfalck_dVm_dz — unified Rosenfalck source model (D1=96, physical mm)"
```

---

### Task 3: Layer 2a — Surface volume conductor kernel

**Files:**
- Modify: `myogen/simulator/core/emg/fiber_simulation.py`
- Modify: `tests/test_fiber_simulation.py`

This task extracts the Bessel volume conductor code from `simulate_fiber.py` (sections 3-5: H_vc, H_ele, H_glo, B(kz)) and adds an IFFT step to produce the spatial kernel b(z).

- [ ] **Step 1: Write test for compute_surface_kernel**

Add to `tests/test_fiber_simulation.py`:

```python
import quantities as pq
from myogen.simulator.core.emg.electrodes import SurfaceElectrodeArray
from myogen.simulator.core.emg.fiber_simulation import compute_surface_kernel


class TestComputeSurfaceKernel:
    """Tests for the multilayer volume conductor spatial kernel."""

    @pytest.fixture
    def electrode_array(self):
        """Standard 8x1 monopolar electrode array."""
        return SurfaceElectrodeArray(
            num_rows=8,
            num_cols=1,
            inter_electrode_distances__mm=5.0 * pq.mm,
            electrode_radius__mm=5.0 * pq.mm,
            center_point__mm_deg=(0.0 * pq.mm, 0.0 * pq.deg),
            bending_radius__mm=8.5 * pq.mm,
            rotation_angle__deg=0.0 * pq.deg,
            differentiation_mode="monopolar",
        )

    def test_returns_correct_shape(self, electrode_array):
        """Kernel shape must be (n_rows, n_cols, len(z_grid))."""
        Nz = 256
        M = 21
        z_grid = np.linspace(-60, 60, Nz)
        k_theta = np.arange(-(M - 1) / 2, (M - 1) / 2 + 1)

        b_z, _, _ = compute_surface_kernel(
            z_grid=z_grid,
            k_theta=k_theta,
            R=5.0,
            electrode_array=electrode_array,
            r=8.5,
            r_bone=0.0,
            th_fat=0.3,
            th_skin=1.29,
            sig_muscle_rho=0.09,
            sig_muscle_z=0.4,
            sig_fat=0.0407,
            sig_skin=4.88e-4,
        )
        assert b_z.shape == (8, 1, Nz)

    def test_kernel_peaks_near_electrode(self, electrode_array):
        """b(z) should peak near the electrode z-position."""
        Nz = 256
        M = 21
        z_grid = np.linspace(-60, 60, Nz)
        k_theta = np.arange(-(M - 1) / 2, (M - 1) / 2 + 1)

        b_z, _, _ = compute_surface_kernel(
            z_grid=z_grid,
            k_theta=k_theta,
            R=5.0,
            electrode_array=electrode_array,
            r=8.5,
            r_bone=0.0,
            th_fat=0.3,
            th_skin=1.29,
            sig_muscle_rho=0.09,
            sig_muscle_z=0.4,
            sig_fat=0.0407,
            sig_skin=4.88e-4,
        )
        # Center electrode (index 4 for 8-row array)
        center_kernel = b_z[4, 0, :]
        peak_idx = np.argmax(np.abs(center_kernel))
        peak_z = z_grid[peak_idx]
        # Electrode center is at z=0, peak should be near 0
        assert abs(peak_z) < 5.0, f"Kernel peak at z={peak_z}, expected near 0"

    def test_kernel_is_real(self, electrode_array):
        """Spatial impulse response must be real-valued."""
        Nz = 256
        M = 21
        z_grid = np.linspace(-60, 60, Nz)
        k_theta = np.arange(-(M - 1) / 2, (M - 1) / 2 + 1)

        b_z, _, _ = compute_surface_kernel(
            z_grid=z_grid,
            k_theta=k_theta,
            R=5.0,
            electrode_array=electrode_array,
            r=8.5,
            r_bone=0.0,
            th_fat=0.3,
            th_skin=1.29,
            sig_muscle_rho=0.09,
            sig_muscle_z=0.4,
            sig_fat=0.0407,
            sig_skin=4.88e-4,
        )
        assert np.allclose(b_z.imag, 0, atol=1e-10)

    def test_caching_returns_same_result(self, electrode_array):
        """Passing cached A_matrix/B_incomplete should give same kernel."""
        Nz = 256
        M = 21
        z_grid = np.linspace(-60, 60, Nz)
        k_theta = np.arange(-(M - 1) / 2, (M - 1) / 2 + 1)

        params = dict(
            z_grid=z_grid, k_theta=k_theta, R=5.0,
            electrode_array=electrode_array, r=8.5, r_bone=0.0,
            th_fat=0.3, th_skin=1.29, sig_muscle_rho=0.09,
            sig_muscle_z=0.4, sig_fat=0.0407, sig_skin=4.88e-4,
        )

        b_z1, A_mat, B_inc = compute_surface_kernel(**params)
        b_z2, _, _ = compute_surface_kernel(**params, A_matrix=A_mat, B_incomplete=B_inc)
        np.testing.assert_allclose(b_z1, b_z2, rtol=1e-10)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_fiber_simulation.py::TestComputeSurfaceKernel -v`
Expected: FAIL — `ImportError: cannot import name 'compute_surface_kernel'`

- [ ] **Step 3: Implement compute_surface_kernel**

Add to `myogen/simulator/core/emg/fiber_simulation.py`. This function extracts sections 3-5 from `_simulate_fiber_v2_python` (H_vc, H_ele, H_glo → B(kz)) and adds an IFFT to produce b(z).

```python
import math
from scipy.special import jv as Jn

from myogen.simulator.core.emg.electrodes import SurfaceElectrodeArray
from myogen.simulator.core.emg.surface.simulate_fiber import (
    log_In, log_Kn,
)


def compute_surface_kernel(
    z_grid: np.ndarray,
    k_theta: np.ndarray,
    R: float,
    electrode_array: SurfaceElectrodeArray,
    r: float,
    r_bone: float,
    th_fat: float,
    th_skin: float,
    sig_muscle_rho: float,
    sig_muscle_z: float,
    sig_fat: float,
    sig_skin: float,
    sig_bone: float = 0.0,
    A_matrix: np.ndarray | None = None,
    B_incomplete: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the multilayer cylindrical volume conductor spatial kernel b(z).

    Solves the Bessel-based boundary value problem (Farina 2004) to get
    B(kz) per electrode, then IFFTs to produce the spatial impulse response.

    Parameters
    ----------
    z_grid : np.ndarray
        Spatial grid for kernel evaluation in mm.
    k_theta : np.ndarray
        Angular frequency axis (integer wavenumbers).
    R : float
        Fiber radial position in mm.
    electrode_array : SurfaceElectrodeArray
        Electrode configuration.
    r, r_bone, th_fat, th_skin : float
        Tissue geometry in mm.
    sig_muscle_rho, sig_muscle_z, sig_fat, sig_skin, sig_bone : float
        Tissue conductivities in S/m.
    A_matrix, B_incomplete : np.ndarray or None
        Cached matrices from a previous call (same geometry).

    Returns
    -------
    b_z : np.ndarray
        Spatial kernel, shape (n_rows, n_cols, len(z_grid)). Real-valued.
    A_matrix : np.ndarray
        Cached A matrix for reuse.
    B_incomplete : np.ndarray
        Cached B matrix for reuse.
    """
    import quantities as pq

    Nz = len(z_grid)
    dz = z_grid[1] - z_grid[0]

    # kz axis from the z_grid via FFT conventions
    k_z = 2 * np.pi * np.fft.fftfreq(Nz, d=dz)

    # Electrode properties
    rele = float(electrode_array.electrode_radius__mm.rescale(pq.mm).magnitude)
    pos_z_mm = electrode_array.pos_z.rescale(pq.mm).magnitude
    pos_theta_rad = electrode_array.pos_theta.rescale(pq.rad).magnitude
    n_rows = electrode_array.num_rows
    n_cols = electrode_array.num_cols

    # --- Section 3: H_vc(kz, kθ) — Bessel volume conductor transfer function ---
    th_muscle = r - th_fat - th_skin - r_bone
    a = r_bone
    b = r_bone + th_muscle
    c = r_bone + th_muscle + th_fat
    d = r_bone + th_muscle + th_fat + th_skin

    am = a * math.sqrt(sig_muscle_z / sig_muscle_rho)
    bm = b * math.sqrt(sig_muscle_z / sig_muscle_rho)
    Rm = R * math.sqrt(sig_muscle_z / sig_muscle_rho)

    # Use only positive frequencies (symmetry), like the existing code
    k_z_pos = np.abs(k_z[k_z >= 0])
    # Avoid kz=0 (causes division issues in Bessel functions)
    k_z_pos = np.maximum(k_z_pos, 1e-10)
    k_theta_pos = k_theta[k_theta >= 0]

    K_THETA, K_Z = np.meshgrid(k_theta_pos, k_z_pos, indexing="ij")
    n_theta, n_z_pos = K_THETA.shape

    # Build A matrix and B vector using the EXISTING log-space Bessel code
    # This is extracted directly from _simulate_fiber_v2_python sections 3-5.
    # Rather than duplicating ~300 lines, we call a helper that wraps the
    # existing Bessel solver and returns H_vc and B(kz) per electrode.
    H_vc, A_matrix, B_incomplete = _solve_bessel_system(
        K_THETA, K_Z, a, b, c, d, am, bm, Rm,
        sig_bone, sig_muscle_rho, sig_muscle_z, sig_fat, sig_skin, r_bone,
        A_matrix, B_incomplete,
    )

    # --- Section 4: H_ele(kz, kθ) ---
    ktheta_mesh, kz_mesh = np.meshgrid(k_theta_pos, k_z_pos, indexing="ij")
    H_sf = electrode_array.get_H_sf(ktheta_mesh, kz_mesh)
    arg = np.sqrt((rele * ktheta_mesh / r) ** 2 + (rele * kz_mesh) ** 2)
    H_size = np.where(arg == 0, 1.0, 2 * Jn(1, arg) / arg)
    H_ele = H_sf * H_size

    # --- Section 5: H_glo → B(kz) per electrode ---
    H_glo = H_vc * H_ele
    k_theta_diff = k_theta_pos[1] - k_theta_pos[0] if len(k_theta_pos) > 1 else 1.0

    # Reconstruct full H_vc/H_glo using symmetry (both kz and ktheta)
    # For the angular integration, we need the full ktheta range
    H_glo_full = _reconstruct_full_spectrum(H_glo, k_theta, k_z)

    ktheta_full_mesh, _ = np.meshgrid(k_theta, k_z_pos, indexing="ij")
    k_theta_diff_full = k_theta[1] - k_theta[0] if len(k_theta) > 1 else 1.0

    B_kz = np.zeros((n_rows, n_cols, len(k_z_pos)), dtype=complex)
    for row in range(n_rows):
        for col in range(n_cols):
            theta_e = pos_theta_rad[row, col]
            integrand = H_glo_full * np.exp(1j * theta_e * ktheta_full_mesh) * k_theta_diff_full
            B_kz[row, col, :] = np.sum(integrand, axis=0) / (2 * np.pi)

    # --- IFFT B(kz) → b(z) ---
    # B_kz is defined on k_z_pos (positive freqs only). Reconstruct full spectrum.
    B_kz_full = np.zeros((n_rows, n_cols, Nz), dtype=complex)
    # Map positive kz indices to FFT frequency order
    freqs = np.fft.fftfreq(Nz, d=dz)
    for i, f in enumerate(freqs):
        # Find closest k_z_pos
        idx = np.argmin(np.abs(k_z_pos - abs(2 * np.pi * f)))
        B_kz_full[:, :, i] = B_kz[:, :, idx]

    b_z = np.zeros((n_rows, n_cols, Nz))
    for row in range(n_rows):
        for col in range(n_cols):
            z_shift = pos_z_mm[row, col]
            # Apply position phase shift and IFFT
            phase = np.exp(1j * 2 * np.pi * freqs * z_shift)
            b_z[row, col, :] = np.real(np.fft.ifft(B_kz_full[row, col, :] * phase)) * Nz

    return b_z, A_matrix, B_incomplete
```

Note: `_solve_bessel_system` and `_reconstruct_full_spectrum` are private helpers that wrap the existing Bessel code from `simulate_fiber.py`. The implementation should reuse the existing log-space Bessel functions (`log_In`, `log_Kn`) and A-matrix construction verbatim — extracted, not rewritten.

The full implementation of these helpers (~200 lines) copies sections 3 and the symmetry reconstruction from `_simulate_fiber_v2_python` lines 414-795. The key difference: they no longer depend on the source term `I(kz,kt)` or the FFT parameters `N`, `Fs`, or `v`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_fiber_simulation.py::TestComputeSurfaceKernel -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add myogen/simulator/core/emg/fiber_simulation.py tests/test_fiber_simulation.py
git commit -m "feat: add compute_surface_kernel — multilayer volume conductor as spatial kernel b(z)"
```

---

### Task 4: Layer 2b — Intramuscular kernel

**Files:**
- Modify: `myogen/simulator/core/emg/fiber_simulation.py`
- Modify: `tests/test_fiber_simulation.py`

- [ ] **Step 1: Write test for compute_intramuscular_kernel**

Add to `tests/test_fiber_simulation.py`:

```python
from myogen.simulator.core.emg.fiber_simulation import compute_intramuscular_kernel


class TestComputeIntramuscularKernel:
    """Tests for the simple Green's function kernel."""

    def test_returns_correct_shape(self):
        z_grid = np.linspace(-30, 30, 200)
        electrode_z = np.array([0.0, 5.0, 10.0])
        b_z = compute_intramuscular_kernel(z_grid, electrode_z, r=1.0)
        assert b_z.shape == (3, 200)

    def test_peaks_at_electrode_position(self):
        z_grid = np.linspace(-30, 30, 1000)
        electrode_z = np.array([5.0])
        b_z = compute_intramuscular_kernel(z_grid, electrode_z, r=1.0)
        peak_z = z_grid[np.argmax(b_z[0])]
        assert abs(peak_z - 5.0) < 0.1

    def test_decays_with_distance(self):
        z_grid = np.linspace(-30, 30, 1000)
        electrode_z = np.array([0.0])
        b_z = compute_intramuscular_kernel(z_grid, electrode_z, r=1.0)
        center = len(z_grid) // 2
        # Value at center should be larger than at edges
        assert b_z[0, center] > b_z[0, 0]
        assert b_z[0, center] > b_z[0, -1]

    def test_matches_existing_function(self):
        """Must match bioelectric.get_elementary_current_response."""
        from myogen.simulator.core.emg.intramuscular.bioelectric import (
            get_elementary_current_response,
        )

        z_grid = np.linspace(-30, 30, 200)
        z_elec = 5.0
        r = 2.0
        r_arr = np.full_like(z_grid, r)

        unified = compute_intramuscular_kernel(z_grid, np.array([z_elec]), r=r)
        existing = get_elementary_current_response(z_grid, z_elec, r_arr)
        np.testing.assert_allclose(unified[0], existing, rtol=1e-10)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_fiber_simulation.py::TestComputeIntramuscularKernel -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement compute_intramuscular_kernel**

Add to `myogen/simulator/core/emg/fiber_simulation.py`:

```python
def compute_intramuscular_kernel(
    z_grid: np.ndarray,
    electrode_z: np.ndarray,
    r: float,
    sigma_r: float = 63.0,
    sigma_z: float = 330.0,
) -> np.ndarray:
    """
    Compute the intramuscular volume conductor kernel using the Green's function.

    For an anisotropic infinite medium (Nandedkar & Stalberg 1983):
    b(z) = 1/(4π·σ_r) / sqrt(σ_z/σ_r · r² + (z - z_elec)²)

    Parameters
    ----------
    z_grid : np.ndarray
        Spatial grid in mm.
    electrode_z : np.ndarray
        Electrode z-positions in mm, shape (n_electrodes,).
    r : float
        Radial distance from fiber to electrode in mm.
    sigma_r : float, default=63.0
        Radial conductivity in S/m.
    sigma_z : float, default=330.0
        Longitudinal conductivity in S/m.

    Returns
    -------
    np.ndarray
        Kernel b(z), shape (n_electrodes, len(z_grid)).
    """
    sigma_r_mm = sigma_r / 1000.0  # S/m → S/mm
    sigma_z_mm = sigma_z / 1000.0

    b_z = np.zeros((len(electrode_z), len(z_grid)))
    for i, z_e in enumerate(electrode_z):
        b_z[i] = (1.0 / (4.0 * np.pi * sigma_r_mm)) / np.sqrt(
            sigma_z_mm / sigma_r_mm * r**2 + (z_grid - z_e) ** 2
        )
    return b_z
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_fiber_simulation.py::TestComputeIntramuscularKernel -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add myogen/simulator/core/emg/fiber_simulation.py tests/test_fiber_simulation.py
git commit -m "feat: add compute_intramuscular_kernel — Green's function volume conductor kernel"
```

---

### Task 5: Layer 3 — Traveling-wave signal computation

**Files:**
- Modify: `myogen/simulator/core/emg/fiber_simulation.py`
- Modify: `tests/test_fiber_simulation.py`

- [ ] **Step 1: Write tests for simulate_fiber_unified**

Add to `tests/test_fiber_simulation.py`:

```python
from myogen.simulator.core.emg.fiber_simulation import simulate_fiber_unified


class TestSimulateFiberUnified:
    """Tests for the traveling-wave fiber signal computation."""

    def _make_simple_kernel(self):
        """Create a simple peaked kernel for testing (Gaussian-like)."""
        z = np.linspace(-30, 30, 512)
        # Simple Green's function-like kernel peaked at z=0
        b = 1.0 / np.sqrt(1.0 + z**2)
        return z, b.reshape(1, -1), np.array([0.0])  # z_grid, b_z, electrode_z

    def test_returns_correct_shape(self):
        z_grid, b_z, elec_z = self._make_simple_kernel()
        phi = simulate_fiber_unified(
            v=4.0, L1=15.0, L2=15.0, zi=0.0,
            b_z=b_z, z_kernel=z_grid, electrode_z=elec_z,
            Fs=10.0, duration_ms=20.0,
        )
        n_samples = int(20.0 * 10.0)  # duration_ms * Fs
        assert phi.shape == (1, n_samples)

    def test_signal_is_biphasic(self):
        """A propagating + extinguishing wave should produce a biphasic signal."""
        z_grid, b_z, elec_z = self._make_simple_kernel()
        phi = simulate_fiber_unified(
            v=4.0, L1=15.0, L2=15.0, zi=0.0,
            b_z=b_z, z_kernel=z_grid, electrode_z=elec_z,
            Fs=10.0, duration_ms=20.0,
        )
        signal = phi[0]
        # Should have both positive and negative phases
        assert np.max(signal) > 0
        assert np.min(signal) < 0

    def test_faster_cv_shifts_signal_earlier(self):
        """Higher conduction velocity should shift the MUAP peak to earlier time."""
        z_grid, b_z, elec_z = self._make_simple_kernel()
        # Electrode at z=10mm, endplate at z=0
        elec_z_offset = np.array([10.0])

        phi_slow = simulate_fiber_unified(
            v=3.0, L1=20.0, L2=20.0, zi=0.0,
            b_z=b_z, z_kernel=z_grid, electrode_z=elec_z_offset,
            Fs=10.0, duration_ms=30.0,
        )
        phi_fast = simulate_fiber_unified(
            v=5.0, L1=20.0, L2=20.0, zi=0.0,
            b_z=b_z, z_kernel=z_grid, electrode_z=elec_z_offset,
            Fs=10.0, duration_ms=30.0,
        )
        peak_slow = np.argmax(np.abs(phi_slow[0]))
        peak_fast = np.argmax(np.abs(phi_fast[0]))
        assert peak_fast < peak_slow

    def test_symmetric_fiber_gives_symmetric_signal(self):
        """Equal semi-lengths and centered electrode → approximately symmetric."""
        z_grid, b_z, elec_z = self._make_simple_kernel()
        phi = simulate_fiber_unified(
            v=4.0, L1=15.0, L2=15.0, zi=0.0,
            b_z=b_z, z_kernel=z_grid, electrode_z=elec_z,
            Fs=10.0, duration_ms=20.0,
        )
        signal = phi[0]
        # Find the peak and check rough symmetry around it
        peak = np.argmax(np.abs(signal))
        # Not exact symmetry due to two-wave interaction, but peak should be near center
        assert abs(peak - len(signal) // 2) < len(signal) // 4

    def test_zero_signal_for_zero_kernel(self):
        z_grid = np.linspace(-30, 30, 512)
        b_z = np.zeros((1, 512))
        phi = simulate_fiber_unified(
            v=4.0, L1=15.0, L2=15.0, zi=0.0,
            b_z=b_z, z_kernel=z_grid, electrode_z=np.array([0.0]),
            Fs=10.0, duration_ms=20.0,
        )
        np.testing.assert_array_equal(phi, 0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_fiber_simulation.py::TestSimulateFiberUnified -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement simulate_fiber_unified**

Add to `myogen/simulator/core/emg/fiber_simulation.py`:

```python
from scipy.interpolate import interp1d


def simulate_fiber_unified(
    v: float,
    L1: float,
    L2: float,
    zi: float,
    b_z: np.ndarray,
    z_kernel: np.ndarray,
    electrode_z: np.ndarray,
    Fs: float,
    duration_ms: float,
    D1: float = 96.0,
) -> np.ndarray:
    """
    Compute electrode potential from a single fiber using traveling-wave convolution.

    Algorithm:
    1. Evaluate dVm/dz on z_kernel grid
    2. Cross-correlate with b_z per channel → h(s)
    3. Sample h at s(t) = zi + v*t - z0 (rightward) and s(t) = -zi - v*t + z0 (leftward)
    4. Apply exact endpoint handling for tendon extinction

    Parameters
    ----------
    v : float
        Conduction velocity in mm/ms.
    L1 : float
        Semi-length from endplate to right tendon in mm.
    L2 : float
        Semi-length from endplate to left tendon in mm.
    zi : float
        Endplate offset from electrode array center in mm.
    b_z : np.ndarray
        Volume conductor kernel, shape (n_channels, Nz).
    z_kernel : np.ndarray
        Spatial grid for the kernel in mm.
    electrode_z : np.ndarray
        Electrode z-positions in mm, shape (n_channels,).
    Fs : float
        Output sampling frequency in kHz.
    duration_ms : float
        Signal duration in ms.
    D1 : float, default=96.0
        Rosenfalck amplitude parameter.

    Returns
    -------
    np.ndarray
        Electrode signals, shape (n_channels, n_timepoints).
    """
    n_channels = b_z.shape[0]
    n_t = int(duration_ms * Fs)
    t = np.arange(n_t) / Fs  # time in ms

    dz = z_kernel[1] - z_kernel[0]
    source = rosenfalck_dVm_dz(z_kernel, D1=D1)

    # Rosenfalck spatial extent (where it's > 0.1% of peak)
    W = 10.0  # mm, conservative bound for Rosenfalck support

    phi = np.zeros((n_channels, n_t))

    for ch in range(n_channels):
        z0 = electrode_z[ch]
        kernel = b_z[ch]

        # Cross-correlate source with kernel to get h(s)
        # h(s) = Σ_i source(z_i) * kernel(z_i + s) ≈ correlate(source, kernel)
        h = np.correlate(source, kernel, mode="full") * dz
        # s-axis for the cross-correlation result
        s_axis = np.arange(-(len(kernel) - 1), len(source)) * dz

        # Build interpolator for h(s)
        h_interp = interp1d(s_axis, h, kind="linear", bounds_error=False, fill_value=0.0)

        # --- Rightward wave ---
        # Propagating phase: φ_right(t) = h(zi + v*t - z0) for 0 ≤ vt ≤ L1-W
        s_right = zi + v * t - z0
        phi_right = h_interp(s_right)

        # Tendon extinction phase: truncated convolution for L1-W < vt < L1+W
        for ti in range(n_t):
            vt = v * t[ti]
            if vt > L1 - W and vt < L1 + W:
                # The wave extends from (vt - W) to vt in fiber-relative coords
                # But fiber ends at L1, so truncate at L1
                z_lo = max(0.0, vt - W)
                z_hi = min(L1, vt + W)
                if z_hi <= z_lo:
                    phi_right[ti] = 0.0
                    continue
                # Truncated convolution on the fiber extent
                z_local = np.arange(z_lo, z_hi, dz)
                if len(z_local) == 0:
                    phi_right[ti] = 0.0
                    continue
                src_local = rosenfalck_dVm_dz(z_local - vt, D1=D1)
                kern_local = h_interp(z_local + zi - z0 - (z_local - vt + zi + vt - z0))
                # Simpler: direct spatial integration
                z_fiber = np.arange(z_lo, z_hi, dz)
                src_vals = rosenfalck_dVm_dz(z_fiber - vt, D1=D1)
                # b_z is on z_kernel grid, interpolate to fiber positions
                b_interp = interp1d(z_kernel, kernel, bounds_error=False, fill_value=0.0)
                kern_vals = b_interp(z_fiber + zi - z0)
                phi_right[ti] = np.sum(src_vals * kern_vals) * dz
            elif vt >= L1 + W:
                phi_right[ti] = 0.0

        # --- Leftward wave (same logic, reversed direction) ---
        s_left = -zi - v * t + z0
        phi_left = h_interp(s_left)

        for ti in range(n_t):
            vt = v * t[ti]
            if vt > L2 - W and vt < L2 + W:
                z_lo = max(0.0, vt - W)
                z_hi = min(L2, vt + W)
                if z_hi <= z_lo:
                    phi_left[ti] = 0.0
                    continue
                z_fiber = np.arange(z_lo, z_hi, dz)
                src_vals = rosenfalck_dVm_dz(z_fiber - vt, D1=D1)
                b_interp = interp1d(z_kernel, kernel, bounds_error=False, fill_value=0.0)
                kern_vals = b_interp(-z_fiber - zi + z0)
                phi_left[ti] = np.sum(src_vals * kern_vals) * dz
            elif vt >= L2 + W:
                phi_left[ti] = 0.0

        phi[ch] = phi_right - phi_left

    return phi
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_fiber_simulation.py::TestSimulateFiberUnified -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add myogen/simulator/core/emg/fiber_simulation.py tests/test_fiber_simulation.py
git commit -m "feat: add simulate_fiber_unified — traveling-wave convolution for fiber signals"
```

---

### Task 6: Integration — wire into surface_emg.py

**Files:**
- Modify: `myogen/simulator/core/emg/surface/surface_emg.py`
- Modify: `tests/test_fiber_simulation.py`

- [ ] **Step 1: Write integration test**

Add to `tests/test_fiber_simulation.py`:

```python
class TestSurfaceEMGIntegration:
    """Test that the unified path can be invoked through SurfaceEMG."""

    def test_use_unified_flag_accepted(self):
        """SurfaceEMG constructor should accept use_unified parameter."""
        from myogen.simulator.core.emg.surface.surface_emg import SurfaceEMG
        import inspect

        sig = inspect.signature(SurfaceEMG.__init__)
        assert "use_unified" in sig.parameters
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_fiber_simulation.py::TestSurfaceEMGIntegration -v`
Expected: FAIL — `use_unified` not in parameters.

- [ ] **Step 3: Add use_unified parameter to SurfaceEMG**

In `myogen/simulator/core/emg/surface/surface_emg.py`, find the `__init__` method and add the parameter. Then in `simulate_muaps`, add the branching logic.

Find the `__init__` signature (around line 70-120) and add `use_unified: bool = False` as a parameter. Store it as `self._use_unified = use_unified`.

Then in the fiber loop (around line 400-428), add a branch:

```python
if self._use_unified:
    from myogen.simulator.core.emg.fiber_simulation import (
        compute_surface_kernel,
        simulate_fiber_unified,
    )

    # Compute kernel (cached per MU via A_matrix/B_incomplete)
    z_kernel = np.linspace(-60, 60, N_internal)
    k_theta_unified = np.arange(-(M_theta - 1) / 2, (M_theta - 1) / 2 + 1)

    if fiber_number == 0:
        b_z, A_matrix_unified, B_incomplete_unified = compute_surface_kernel(
            z_grid=z_kernel, k_theta=k_theta_unified, R=R,
            electrode_array=electrode_array,
            r=r_total, r_bone=r_bone, th_fat=th_fat, th_skin=th_skin,
            sig_muscle_rho=sig_rho, sig_muscle_z=sig_z,
            sig_fat=sig_fat_val, sig_skin=sig_skin_val,
        )
    else:
        b_z, A_matrix_unified, B_incomplete_unified = compute_surface_kernel(
            z_grid=z_kernel, k_theta=k_theta_unified, R=R,
            electrode_array=electrode_array,
            r=r_total, r_bone=r_bone, th_fat=th_fat, th_skin=th_skin,
            sig_muscle_rho=sig_rho, sig_muscle_z=sig_z,
            sig_fat=sig_fat_val, sig_skin=sig_skin_val,
            A_matrix=A_matrix_unified, B_incomplete=B_incomplete_unified,
        )

    elec_z = base_pos_z.flatten()
    duration = kernel_length / v_conduction  # ms

    phi_temp = simulate_fiber_unified(
        v=v_conduction, L1=L1, L2=L2, zi=innervation_zone,
        b_z=b_z.reshape(-1, len(z_kernel)),
        z_kernel=z_kernel, electrode_z=elec_z,
        Fs=Fs_internal, duration_ms=duration,
    )
    # Reshape back to (n_rows, n_cols, n_t)
    phi_temp = phi_temp.reshape(n_rows, n_cols, -1)
else:
    # Existing frequency-domain path
    phi_temp, A_matrix, B_incomplete = _simulate_fiber_v2_python(...)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_fiber_simulation.py::TestSurfaceEMGIntegration -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add myogen/simulator/core/emg/surface/surface_emg.py tests/test_fiber_simulation.py
git commit -m "feat: add use_unified flag to SurfaceEMG for new fiber simulation path"
```

---

### Task 7: Validation — compare old vs new

**Files:**
- Modify: `tests/test_fiber_simulation.py`

- [ ] **Step 1: Write comparison test**

Add to `tests/test_fiber_simulation.py`:

```python
class TestOldVsNewComparison:
    """Compare MUAPs from old frequency-domain and new time-domain paths."""

    @pytest.fixture
    def electrode_array(self):
        return SurfaceElectrodeArray(
            num_rows=8, num_cols=1,
            inter_electrode_distances__mm=5.0 * pq.mm,
            electrode_radius__mm=5.0 * pq.mm,
            center_point__mm_deg=(0.0 * pq.mm, 0.0 * pq.deg),
            bending_radius__mm=8.5 * pq.mm,
            rotation_angle__deg=0.0 * pq.deg,
            differentiation_mode="monopolar",
        )

    def test_muap_shapes_are_similar(self, electrode_array):
        """Both paths should produce MUAPs with similar temporal shape."""
        from myogen.simulator.core.emg.surface.simulate_fiber import (
            _simulate_fiber_v2_python,
        )
        from myogen.simulator.core.emg.fiber_simulation import (
            compute_surface_kernel,
            simulate_fiber_unified,
        )

        # Common parameters
        v = 4.0  # mm/ms
        R = 5.0  # mm depth
        L1 = 15.0
        L2 = 15.0
        zi = 0.0
        r = 8.5
        N = 256
        M = 21
        Fs = 10.0  # kHz (internal)

        # --- Old path ---
        phi_old, _, _ = _simulate_fiber_v2_python(
            Fs=Fs, v=v, N=N, M=M, r=r, r_bone=0.0,
            th_fat=0.3, th_skin=1.29, R=R, L1=L1, L2=L2, zi=zi,
            electrode_array=electrode_array,
            sig_muscle_rho=0.09, sig_muscle_z=0.4,
            sig_fat=0.0407, sig_skin=4.88e-4,
            fiber_length__mm=75.0,
        )

        # --- New path ---
        z_kernel = np.linspace(-60, 60, N)
        k_theta = np.arange(-(M - 1) / 2, (M - 1) / 2 + 1)
        b_z, _, _ = compute_surface_kernel(
            z_grid=z_kernel, k_theta=k_theta, R=R,
            electrode_array=electrode_array,
            r=r, r_bone=0.0, th_fat=0.3, th_skin=1.29,
            sig_muscle_rho=0.09, sig_muscle_z=0.4,
            sig_fat=0.0407, sig_skin=4.88e-4,
        )

        elec_z = electrode_array.pos_z.rescale(pq.mm).magnitude.flatten()
        duration = 75.0 / v  # ms, same as fiber_length / v

        phi_new = simulate_fiber_unified(
            v=v, L1=L1, L2=L2, zi=zi,
            b_z=b_z.reshape(-1, N), z_kernel=z_kernel,
            electrode_z=elec_z, Fs=Fs, duration_ms=duration,
        )

        # Both should be non-zero
        assert np.max(np.abs(phi_old)) > 0, "Old path produced zero signal"
        assert np.max(np.abs(phi_new)) > 0, "New path produced zero signal"

        # Normalize both to unit peak and compare shape correlation
        old_center = phi_old[4, 0, :]  # center electrode
        new_center = phi_new[4, :] if phi_new.ndim == 2 else phi_new[4, 0, :]

        # Resample to same length if needed
        from scipy.signal import resample
        if len(old_center) != len(new_center):
            new_center = resample(new_center, len(old_center))

        old_norm = old_center / np.max(np.abs(old_center))
        new_norm = new_center / np.max(np.abs(new_center))

        # Cross-correlation should show high similarity (> 0.7)
        correlation = np.corrcoef(old_norm, new_norm)[0, 1]
        assert correlation > 0.7, (
            f"MUAP shape correlation {correlation:.3f} < 0.7 — "
            f"old and new paths produce dissimilar waveforms"
        )
```

- [ ] **Step 2: Run comparison test**

Run: `.venv/bin/python -m pytest tests/test_fiber_simulation.py::TestOldVsNewComparison -v -s`
Expected: PASS with correlation > 0.7. Print the correlation value for verification.

- [ ] **Step 3: Commit**

```bash
git add tests/test_fiber_simulation.py
git commit -m "test: add old-vs-new MUAP comparison test for validation"
```
