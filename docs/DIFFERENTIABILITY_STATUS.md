# MyoGen Differentiability — Implementation Status

Goal: make the MyoGen simulation differentiable end-to-end in JAX (force, joint
kinematics, and EMG) w.r.t. neural, muscle, and joint parameters, with a hard-spike
scientific mode and surrogate/rate modes for optimization.

**Key reframing (verified in the codebase):** most of the compute core was already
pure-JAX. Retiring NEURON is orthogonal to differentiability. The blockers were
narrow: hard spike thresholds, stochastic generators, a scipy muscle-setup step,
and the numpy EMG summation. The EMG splits into an easy tier (differentiable
w.r.t. neural/muscle/joint params — a convolution against frozen MUAP templates)
and a hard-but-optional tier (differentiable w.r.t. tissue conductivity / geometry —
needs a volume-conductor port).

## Delivered (validated)

| Milestone | What | Files | Validation |
|---|---|---|---|
| **M0** | `run_jax` closed-loop entry point; differentiable-PyTree vs static-config split; explicit PRNG keys; `spike_mode` + metadata; `value_and_grad_run` (float-leaf partition) | `jaxley/closed_loop.py` | end-to-end run in 3 modes; finite non-zero grads; int leaves excluded |
| **M2** | Surrogate-gradient spike primitive (`spike_detect`, custom-JVP straight-through); hard/surrogate/rate modes wired into the scan | `jaxley/jax_models.py` | hard≡surrogate forward (bit-identical); hard grad=0, surrogate/rate grad≠0 |
| **M3** | Differentiable stochastic generators: `rate` (exact expected-value) and `pathwise` (frozen-sample + surrogate) modes | `jaxley/jax_models.py` | hard≡pathwise forward; rate grad = exact `T·N·dt` |
| **M4** | `differentiable_twitch_params` — closed-form Fuglevand path (P, T, twiAmp, IIR) in JAX; gradients w.r.t. RP/Tl/RT/fP (saturation `tetF` frozen) | `jaxley/jax_models.py` | matches scipy path to 6e-8; finite grads |
| **M5** | Differentiable EMG: `surface_emg_jax`, `intramuscular_emg_jax`, `resample_muaps` (spikes ⋆ frozen MUAP) | `jaxley/emg.py` | matches numpy `correlate` to 2e-7; spike→EMG grad finite |
| **M1** | Differentiable modified Bessel `iv_int`/`kv_int` (Miller/upward recurrence + small-x series; auto-diff, no custom_jvp) — **GO** | `jaxley/bessel.py` | values & grads vs scipy < 1.7e-7 |
| **M6** | New differentiable API exported from `myogen.simulator.jaxley`; backward-compat preserved (defaults unchanged) | `jaxley/__init__.py` | package imports; hard-mode defaults return bool as before |
| **M8** | Per-stage FD gradient checks (spindle/gto/joint/hill/twitch) + end-to-end | `scripts/test_gradient_checks.py` | all stages autodiff ≈ FD |

**Headline result:** gradient of a surface-EMG loss flows through spikes to a neural
weight (`d(EMG loss)/d(base_dd)` finite and non-zero). Force + joint + EMG are
differentiable w.r.t. neural, muscle, and joint parameters today.

### Validation scripts
- `scripts/test_differentiable_pipeline.py` — end-to-end (M0/M2/M3/M5) on a tiny network.
- `scripts/test_gradient_checks.py` — per-stage autodiff-vs-FD (M8).
- `scripts/prototype_bessel_jax.py` — Bessel values/gradients vs scipy (M1).
- `scripts/test_old_vs_new_equivalence.py` — **regression**: OLD (git HEAD) vs NEW
  full-scan output, bit-identical (0.0) across all channels in `hard` mode. Proves
  the refactor changed no numbers.
- `scripts/test_nerlab_differentiable.py` — **runs on the REAL NERLab motor neurons**
  (napp+caL, V_rest≈0, spike peak ≈+90 mV). Confirms MNs spike, hard≡surrogate
  forward, and surrogate gradients flow to the descending-drive weight AND to the
  actual channel conductances (`napp_gnabar`, `napp_gkfbar`, `caL_gcaLbar`).

### Differentiating w.r.t. Jaxley channel parameters — required step
`net.get_parameters()` returns **only** parameters marked with
`net.make_trainable("napp_gnabar")` (etc.). Without `make_trainable`, the neural
parameter subtree is empty and gradients flow only to the muscle/joint/weight
params. Mark the conductances (or gates) you want to optimize trainable **before**
`build_init_and_step_fn`, then they appear in `params["jaxley"]` and receive
gradients.

## Remaining

**M7 — full volume-conductor JAX port (optional, de-risked, not yet done).** Only
needed for gradients w.r.t. tissue conductivities / conduction velocity / geometry.
M1 proved the hard enabler (differentiable arbitrary-order Bessel) is feasible; the
rest is mechanical (`np.linalg.solve`→`jnp`, `np.fft`→`jnp.fft`, freeze the argmax
peak-centring, smooth the iEMG step masks). Also needs a native `jv` (same technique
as `iv`/`kv`; `j0`/`j1` absent from JAX). Est. 700–1300 LOC + validation. See
`docs/M1_bessel_feasibility.md`.

**Deeper M6 rewire (optional).** The legacy fake-JAX `jaxley/muscle.py` class and the
NumPy `jaxley/proprioception/*` wrappers can be rewired to delegate to the
`jax_models.py` compute path so the *class* API also exposes gradients. The
functional path (`run_jax`) already does; this is ergonomics.

## JIT usage
`run_jax` cannot be handed to `jax.jit` directly: `ClosedLoopConfig` is intentionally
unhashable (holds arrays), and integer Hill fields (`N`/`Ntype1`) are used in shape
contexts (`jnp.arange(N)`) so must stay static. Use the wrappers, which do the
static/dynamic split: `value_and_grad_run` for gradients (primary path), and
`compile_run(config, params_template)` for a `jax.jit`-compiled forward pass.

## Notes / caveats
- Surrogate spike gradients are biased (inherent to differentiating spiking models);
  `rate` mode gives cleaner gradients at the cost of a continuous forward output.
- Pathwise generator gradients are pathwise-only (fixed PRNG sample).
- M4 leaves the tetanic-saturation constant `tetF` frozen; gradients w.r.t. the
  saturation-shape constants are the low-value tail and are not propagated.
