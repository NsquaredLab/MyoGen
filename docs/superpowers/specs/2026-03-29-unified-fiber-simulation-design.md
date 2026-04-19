# Unified Fiber Simulation Module

## Problem

The sEMG and iEMG simulations use incompatible source models for the same physics (Rosenfalck 1969 transmembrane current). The sEMG code (`simulate_fiber.py`) operates entirely in the frequency domain with fragile parameter coupling:

- `z /= 2` ad-hoc spatial scaling
- `A = 96/10 = 9.6` vs iEMG's `D1 = 96.0`
- N simultaneously controls temporal resolution, spatial resolution, and Rosenfalck evaluation domain
- `Fs_effective` workaround to decouple fiber length from sampling rate
- `IAP_SCALE_FACTOR = 2.5` heuristic for kernel extent
- FFT of a one-sided kernel on a large grid (mostly zeros, spectral leakage)

The iEMG code (`bioelectric.py`) is stable because it separates source from volume conductor cleanly: time-domain IAP propagation convolved with a simple Green's function.

## Solution

Unify both paths into a single module using a **traveling-wave convolution** approach:

1. One Rosenfalck source model (D1=96, z in physical mm, no artificial scaling)
2. Volume conductor expressed as a 1D spatial kernel b(z) — either the simple Green's function (iEMG) or IFFT of the multilayer Bessel transfer function (sEMG)
3. Signal computed as a cross-correlation of the source with b(z), sampled along the wave trajectory

## Architecture

### New file: `myogen/simulator/core/emg/fiber_simulation.py`

Three layers:

#### Layer 1 — Source model

```python
def rosenfalck_dVm_dz(z: np.ndarray, D1: float = 96.0) -> np.ndarray:
    """Raw Rosenfalck first derivative. z in physical mm. No scaling."""
```

Shared by iEMG and sEMG. Replaces both `get_tm_current_dz()` (with its ad-hoc `D1` usage) and the sEMG's `A * exp(-z) * (3z^2 - z^3)` block.

#### Layer 2 — Volume conductor kernels

Two kernel constructors, same interface:

```python
def compute_surface_kernel(
    z_grid: np.ndarray,       # spatial grid for kernel evaluation
    k_theta: np.ndarray,      # angular frequency axis
    R: float,                 # fiber radial position (mm)
    electrode_array: SurfaceElectrodeArray,
    tissue_params: TissueParams,  # r, r_bone, th_fat, th_skin, conductivities
    A_matrix: np.ndarray | None = None,
    B_incomplete: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute multilayer cylindrical volume conductor kernel b(z) per electrode.

    Uses existing Bessel-based code to compute B(kz) for each electrode,
    then IFFTs to get the spatial impulse response.

    Returns: (b_z, A_matrix, B_incomplete)
        b_z: shape (n_rows, n_cols, len(z_grid))
    """
```

```python
def compute_intramuscular_kernel(
    z_grid: np.ndarray,
    electrode_positions: np.ndarray,
    r: float,                 # radial distance fiber-to-electrode
    sigma_r: float = 63.0,
    sigma_z: float = 330.0,
) -> np.ndarray:
    """
    Simple Green's function kernel for intramuscular electrodes.
    Wraps get_elementary_current_response().

    Returns: b_z, shape (n_electrodes, len(z_grid))
    """
```

#### Layer 3 — Traveling-wave signal computation

```python
def simulate_fiber_unified(
    v: float,                 # conduction velocity (mm/ms)
    L1: float,                # semi-length endplate to right tendon (mm)
    L2: float,                # semi-length endplate to left tendon (mm)
    zi: float,                # endplate offset from electrode center (mm)
    b_z: np.ndarray,          # volume conductor kernel, shape (n_channels, Nz)
    z_kernel: np.ndarray,     # spatial grid for kernel (mm)
    electrode_z: np.ndarray,  # electrode z-positions (mm), shape (n_channels,)
    Fs: float,                # output sampling frequency (kHz)
    duration_ms: float,       # signal duration (ms)
    D1: float = 96.0,         # Rosenfalck parameter
) -> np.ndarray:
    """
    Compute surface or intramuscular potential from a single fiber.

    Algorithm:
    1. Evaluate dVm/dz on z_kernel grid
    2. Cross-correlate with b_z to get h(s) per channel
    3. For each channel, sample h(zi + v*t - z0) with endpoint windowing
    4. Combine rightward and leftward waves

    Returns: phi, shape (n_channels, n_timepoints)
    """
```

### Signal computation detail

The IAP is a traveling wave: `dVm/dz(z - zi - vt)` for the rightward component. The electrode potential is:

```
φ(t) = ∫ dVm/dz(z - zi - vt) · b(z - z0) dz
```

Substituting `u = z - zi - vt`:

```
φ(t) = ∫ dVm/dz(u) · b(u + zi + vt - z0) du = h(zi + vt - z0)
```

Where `h(s) = (dVm/dz ⋆ b)(s)` is the cross-correlation, computed once.

For the full fiber signal:
- **Rightward wave**: `h_right(s)` sampled at `s = zi + v*t - z0`, active for `0 ≤ v*t ≤ L1`
- **Leftward wave**: `h_left(s)` sampled at `s = -zi - v*t + z0`, active for `0 ≤ v*t ≤ L2`
- **Total**: `φ(t) = φ_right(t) - φ_left(t)`

The progressive appearing/disappearing at endplate and tendons is handled automatically: the Rosenfalck waveform has finite spatial support (~10mm), so as it enters/exits the fiber extent, the convolution integral smoothly ramps up/down. This replaces the frequency-domain sinc windowing from Farina 2004 eq (3).

### Endpoint windowing detail

The fiber occupies `[zi - L2, zi + L1]` in absolute z. The rightward wave propagates from zi toward zi + L1.

The key insight: `h(s)` represents the signal from an *infinite* traveling wave. For a finite fiber, the signal has three phases:

1. **Propagating phase** (`0 < vt < L1 - W`, where W ≈ 10mm is the Rosenfalck spatial width): The entire Rosenfalck waveform is inside the fiber. Signal is exactly `h(zi + vt - z0)`. This is the main phase — most of the MUAP duration.

2. **Tendon extinction** (`L1 - W < vt < L1 + W`): The waveform reaches the tendon and is progressively truncated. The signal deviates from `h(s)` because part of the Rosenfalck waveform extends beyond the fiber boundary. This produces the characteristic end-of-fiber (non-propagating) component.

3. **Post-extinction** (`vt > L1 + W`): Wave has fully exited. Signal → 0.

For the propagating phase (1), the traveling-wave optimization applies directly. For the extinction phase (2), a short truncated convolution over ~W mm is needed — this covers only a few ms of signal and is not expensive. Implementation options:
- **Exact**: compute truncated spatial convolution for the ~W/v ms extinction window
- **Approximate**: evaluate `h(s)` with hard cutoff at `vt = L1` (loses end-of-fiber component but simpler)

The implementation should use the exact approach, since the end-of-fiber component is physiologically important for bipolar recordings and fiber length estimation.

### Caching strategy

| What | Key | Reuse scope |
|------|-----|-------------|
| A_matrix (7x7 Bessel system) | tissue geometry (a, b, c, d, conductivities) | All fibers in muscle |
| B_incomplete → b(z) | A_matrix + fiber depth R | All fibers at same depth (bin to 0.1mm) |
| h(s) cross-correlation | b(z) + conduction velocity v | Fibers with same depth and CV |

### Rosenfalck standardization

| Parameter | Old iEMG | Old sEMG | Unified |
|-----------|----------|----------|---------|
| D1/A | 96.0 | 9.6 | 96.0 |
| z scaling | -2z (compress) | z/2 (stretch) | none (physical mm) |
| IAP width | ~1.7mm | ~7mm | ~3.5mm |
| Derivative | time-domain diff | FFT(psi) | `rosenfalck_dVm_dz()` |
| Fiber endpoints | tendon_terminator rect window | sinc in frequency domain | implicit in convolution limits |

### Integration with existing code

The new module coexists with the old code for validation:

- `surface_emg.py` gets a `use_unified: bool = False` parameter
- When `True`, calls `simulate_fiber_unified()` instead of `simulate_fiber_v2()`
- The MUAP output format is identical (shape: `n_rows x n_cols x n_timepoints`)
- After validation, the old `simulate_fiber_v2` code path can be removed

### What this eliminates

- `z /= 2` ad-hoc spatial scaling
- `A = 96/10` separate parameterization
- `f_minus_t()` array reversal
- FFT of Rosenfalck kernel (PSI, PSI_conj, PSI_mesh_conj)
- N/Fs/fiber_length three-way coupling
- `Fs_effective` workaround
- `IAP_SCALE_FACTOR = 2.5` heuristic
- `kz_mesh_kzkt` / `kt_mesh_kzkt` 2D frequency meshgrids for source
- Source-coupled I(kz,kt) computation (~50 lines)

### What stays

- Bessel volume conductor code (A matrix, B vector, log-space arithmetic)
- Electrode H_sf and H_size computation
- Angular integration to get B(kz)
- Motor unit loop structure in `surface_emg.py`
- MUAP → spike train convolution pipeline
- `get_current_density()` in bioelectric.py (kept for backward compat, not used by new path)

## Verification plan

1. **Unit test**: `rosenfalck_dVm_dz(z)` matches `get_tm_current_dz(z)` exactly (same function, same params)
2. **Kernel test**: `compute_surface_kernel()` b(z) is the IFFT of B(kz) from existing code — compare numerically
3. **MUAP comparison**: Run both old and new paths on the same motor unit, compare monopolar MUAPs. Expected: similar shape, different amplitude (due to D1=96 vs A=9.6 and scaling differences). The ratio should be consistent across electrodes.
4. **Bipolar/Laplacian**: Verify that spatial differentiation modes work correctly with the new kernel approach
5. **Full sEMG**: Generate interference EMG with both paths, compare spectral characteristics (power spectrum shape, mean/median frequency)
6. **Performance**: New path should be faster (no 2D FFT meshgrids, no per-fiber matrix solve for B vector if caching works)
