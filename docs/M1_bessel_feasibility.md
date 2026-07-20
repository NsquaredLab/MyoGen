# M1 — Differentiable Bessel Feasibility: GO

**Decision: GO.** Differentiating the EMG volume-conductor model w.r.t. its
generating parameters (tissue conductivities, conduction velocity, geometry) is
feasible. The full port (M7) is therefore unblocked.

## Question

The surface-EMG volume-conductor model (`core/emg/surface/simulate_fiber.py`)
evaluates modified Bessel functions `iv`/`kv` at integer orders up to ~16. JAX's
`jax.scipy.special` provides only `i0/i1/i0e/i1e` — **no `k0/k1`, no arbitrary-order
`iv/kv/jv`**. Without differentiable Bessel functions, EMG cannot be made
differentiable w.r.t. `sig_*`/CV/geometry (Tier B / M7). This spike answered:
can we implement `iv`/`kv` natively in JAX with accurate values **and gradients**?

## Result

Implemented in `myogen/simulator/jaxley/bessel.py`:
- `k0`, `k1` via Abramowitz & Stegun 9.8.5–9.8.8 polynomial approximations (seeded
  by JAX `i0`/`i1`).
- `kv_int(n, x)` via stable **upward** recurrence `K_{m+1}=K_{m-1}+(2m/x)K_m`.
- `iv_int(n, x)` via Miller **downward** recurrence (stable direction for I_n)
  normalised against JAX `i0`, blended with an ascending series `Σ (x/2)^(2k+n)/(k!(n+k)!)`
  below `x=1` where the recurrence's `(2k/x)` factors would degrade the derivative.

**Key design choice:** the forward pass uses only smooth ops (JAX `i0`/`i1`,
polynomials, logs, recurrences), so `jax.grad` yields the exact derivative
automatically — no `custom_jvp` needed.

## Validation

`scripts/prototype_bessel_jax.py` checks values and gradients against scipy and the
analytic identities `I_n'=(I_{n-1}+I_{n+1})/2`, `K_n'=-(K_{n-1}+K_{n+1})/2`, over
orders {0,1,2,3,5,8,12,16} and x ∈ {0.05, 0.3, 1, 2.5, 6, 15, 40}:

- **Worst value relative error: 1.67e-7**
- **Worst gradient relative error: 1.67e-7**

Both sit at the accuracy floor of JAX's `i0`/`i1` polynomial approximation — more
than sufficient for the float32 EMG pipeline.

## Implications for M7

- `iv`/`kv` are solved. The remaining pieces of the volume-conductor port are
  mechanical JAX swaps: `np.linalg.solve` → `jnp.linalg.solve`, `np.fft` → `jnp.fft`,
  and freezing the `argmax`/`roll` peak-centring as a constant shift.
- **Follow-up:** the ordinary Bessel `jv` (used once for electrode size,
  `simulate_fiber.py:801`) still needs an analogous native implementation
  (downward recurrence seeded by `j0`/`j1`; JAX lacks these too). Same technique;
  small additional effort.
- For iEMG, hard step masks in `bioelectric.py` must become temperature-controlled
  sigmoids to get gradients w.r.t. CV / fiber length (as noted in the plan).
