# MyoGen API Redesign — a pufferlib-inspired, backend-swappable architecture

**Status:** Design proposal (for team alignment — Ricardo, Devon). No implementation yet.
**Branch:** `feat/puffer-api`
**Date:** 2026-06-23

---

## 1. Goal

Replace MyoGen's current NEURON-hardwired API with a clean, **backend-swappable, vectorized** API in the spirit of [PufferLib](https://github.com/PufferAI/PufferLib). The new API becomes *the* API (clean break; the current `simulator.AlphaMN__Pool` / `SurfaceEMG` surface is deprecated and removed over a transition window).

Two concrete future integrations drive the design and must "just slot in":

- **jaxley replaces NEURON** as the neural-dynamics engine (JAX, GPU, `vmap`/`jit`, differentiable).
- **An FEM muscle model (Devon)** upgrades the muscle/force/volume-conductor stage.

These are **two distinct backend seams in different pipeline stages** — the architecture must support both without leaking either backend into the public API.

## 2. What we are borrowing from pufferlib (and what we are not)

Studied from the pufferlib 3.x source and Codex-verified. Transferable ideas:

1. **A flat, contiguous buffer is the interface, not objects.** Stages write results *in place* into shared arrays; consumers get zero-copy views. (pufferlib `src/vecenv.h` `StaticVec`; `torch_pufferl.py` zero-copy tensor wrapping.)
2. **One uniform driver over a thin surface.** A single `Simulation` object owns the time loop; everything else is small and uniform. (pufferlib `make()` + uniform `step()`.)
3. **Config-as-data.** Reproducible, diffable, sweepable experiment specs. (pufferlib INI configs merged at load.)
4. **A backend escalation path behind a fixed interface** — the API does not change when the engine does.

**Not** borrowing (RL-specific or made unnecessary by JAX): hand-rolled OMP busy-wait threading, CUDA graphs, bf16 obs, the `reset()/step(action)` RL env framing. jaxley/XLA give us GPU vectorization and fusion "for free"; we keep the *lessons* (batch, don't allocate per step, overlap) without the bespoke C runtime.

> Correction folded in from verification: pufferlib's bf16 is a *network* default, not a uniform observation dtype; and MyoGen's current force path is dense-convert + per-MU JIT loops, **not** sparse matmul. Neither changes this design.

## 3. Architecture — seven layers

The system is **not** a feed-forward chain. It is a closed loop (afferent feedback through the biomechanics) expressed as a stage graph stepped in lockstep over one shared buffer.

```
┌─ 7. Conveniences ──────────────────────────────────────────────┐
│   make() one-liner   ·   Spec/TOML config                       │
├─ 6. Data & IO ─────────────────────────────────────────────────┤
│   buffer⇄Neo/quantities adapters · grid annotations             │
│   persistence: ContinuousSaver, chunked save, NWB export        │
├─ 5. Specs (config-as-data, units at the boundary) ─────────────┤
│   RecruitmentThresholds(4 modes) · Drive/Stimuli(ramp…trapezoid)│
│   Connectivity · ElectrodeArray(surface/intramuscular)          │
├─ 4. Stage graph (closed loop over shared buffer) ──────────────┤
│   Network → Muscle → Joint → Afferents ↺   + SurfaceEMG/iEMG    │
│   seam#1 NEURON│jaxley        seam#2 analytic│Hill│FEM          │
├─ 3. Core ──────────────────────────────────────────────────────┤
│   SimState buffer (plain SI float arrays) · Simulation driver   │
│   · Stage protocol · Backend protocol · SimResult facade        │
├─ 2. Cross-cutting services ────────────────────────────────────┤
│   reproducibility (seed, derive_subseed, id reset)              │
│   backend lifecycle (NMODL/Cython build·load·teardown, params)  │
│   units & validation (quantities, beartype)                     │
├─ 1. Analysis / viz (out of hot path) ──────────────────────────┤
│   plotting.* · firing-rate/CV stats · noise calibration         │
└────────────────────────────────────────────────────────────────┘
```

### Closed-loop data flow

```
        descending drive ─────────────┐
                                       ▼
   ┌──────────────────────────────────────────────┐
   │  Network  (dynamics seam: NEURON │ jaxley)    │
   │   α-MN pool · Renshaw · Ia-IN · afferent in   │◀───────┐
   └───────────────┬──────────────────────────────┘        │
            MN spikes                                       │ afferent
                   ▼                                        │ firing
            ┌──────────┐   ┌───────────────┐   ┌────────────┴─────┐
            │  Muscle  │──▶│ Joint /       │──▶│   Afferents      │
            │ (force)  │   │ biomechanics  │   │ spindle Ia/II,   │
            └────┬─────┘   │ (len,vel,ang) │   │ GTO Ib           │
                 ▼ force   └───────────────┘   └──────────────────┘
            ┌────────────┐
            │ SurfaceEMG │──▶ SimResult
            └────────────┘
```

At step *t*, the `Afferents` stage reads muscle/joint state from *t−1* and injects input for *t*. This **one-tick feedback delay** is where conduction/axonal delays live; it makes the loop steppable without a global implicit solve. Open-loop runs simply do not wire the feedback edge.

## 4. Core protocols

Plain-array, in-place, backend-agnostic. Sketches (final signatures to be settled in the implementation plan):

```python
class SimState:
    """The shared buffer. Single source of truth. SI floats, no units, no neo."""
    dynamics: Array          # [n_units, state_dim]  (xp = numpy | jax | torch)
    spikes:   Array          # event/boolean or per-unit spike-time store
    force:    Array          # [n_t, n_pools]
    emg:      Array | None    # [n_t, n_rows, n_cols]
    xp: ModuleType           # the array backend (np/jnp/torch) — one place
    def view(self, name) -> Array: ...      # zero-copy slice for a stage

class Stage(Protocol):
    """Thin. Owns API + units + validation; delegates physics to a Backend."""
    def setup(self, state: SimState) -> None: ...    # claim buffer slices
    def step(self, state: SimState, t: int) -> None: ...  # write IN PLACE
    def render(self, state: SimState) -> Array: ...   # optional derived output

class Backend(Protocol):
    """Behind a seam. The thing that actually integrates physics."""
    def init(self, spec, state: SimState) -> None: ...
    def advance(self, state: SimState, dt: float) -> None: ...
    def teardown(self) -> None: ...

class Simulation:
    """Backend-agnostic driver. Owns dt + the lockstep tick over the graph."""
    def __init__(self, *stages: Stage, dt, edges=None): ...
    def run(self, drive, dt=None, until=None) -> "SimResult": ...
    def step_callback(self, fn): ...   # per-step closed-loop hook (raw slices)
```

### `SimResult` — buffer-first data facade (the data-format decision)

Core + results are **plain SI float arrays**. Neo/NWB are opt-in **methods** (not properties — so neo cannot silently re-enter helper code) built on demand.

```python
@dataclass(slots=True)
class SimResult:
    spike_times_s: list[Array]          # per unit, seconds
    force_N:       Array                 # [n_t, n_pools], Newtons
    surface_emg_V: Array | None           # [n_t, n_rows, n_cols], Volts
    dt_s: float; t_start_s: float; n_units: int; grid_shape: tuple | None
    def to_neo(self) -> "neo.Block": ...   # zero-copy for C-contig float64; 1 copy for float32/GPU
    def to_nwb(self, path, **kw): ...      # delegates to to_neo() → existing export_to_nwb
    def to_torch(self): ...                # zero-copy if CPU
    def to_jax(self): ...
```

**Where units live:** `quantities` only at the spec-input boundary (existing `types.py` `Quantity__Hz`/`Quantity__ms` + `beartype` pattern) and inside the adapter methods. SI is implied in `SimResult` by field-name suffix (`_V`, `_N`, `_s`). The hot-path modules import neither `neo` nor `quantities`.

**Gotchas (documented, from verification):** `neo.AnalogSignal(arr*pq.V, copy=False)` is zero-copy only for C-contiguous float64 numpy; float32 upcasts (one copy). JAX arrays are immutable/on-device → `__array__` forces a host copy; this is fundamental to JAX, not a MyoGen bug. Per-step closed-loop callbacks **must** use raw buffer slices, never `to_neo()` (constructing a Block per tick defeats the design).

## 5. The two backend seams

### Seam #1 — neural dynamics (`Network`): NEURON now → jaxley later

`Network` holds **populations** (α-MN, Renshaw, Ia interneurons, afferent input terminals, descending-drive terminals) and a **backend-neutral connectivity spec** (source pop, target pop, weight, delay, sign), replacing direct `h.NetCon`/`h.Section`/synapse objects in the public API.

- **`NEURONBackend` (today):** keeps its internal `Section`/`NetCon` graph but **syncs to/from `SimState` at the step boundary** (read input slice → advance NEURON → write spike/voltage slice). NEURON globals (`h.run`, `h.dt`, `FInitializeHandler`) stay *inside* this backend; the public `Simulation` never sees them.
- **`JaxleyBackend` (future):** the population state *is* the `SimState.dynamics` array; `advance` is `vmap`/`jit` over it on GPU. This is the case the flat-buffer core exists for.

### Seam #2 — muscle / force / volume conductor (`Muscle`): analytic → FEM later

`Muscle` owns more than force: **spatial geometry, MU territories, innervation areas, fiber layout** *and* a pluggable force model.

- **Force models (today):** `Fuglevand` (`ForceModel`), `ForceModelVectorized`, and dynamic `HillModel` (with flexor/extensor role).
- **`FEMBackend` (future, Devon):** replaces the cylindrical (Farina) volume-conductor / analytic geometry with a finite-element muscle model, feeding better length/force signals into the afferents.

## 6. Mapping the current public surface (nothing dropped)

From the Codex completeness audit. Every current public item has a home:

| Current surface | New home |
| --- | --- |
| `AlphaMN__Pool`, `*__Pool` (afferents, interneurons, descending drive), `Network`, connectivity | **Network** stage + Connectivity spec |
| `SimulationRunner` | **Simulation** driver (backend-agnostic) |
| `RecruitmentThresholds` (fuglevand/deluca/konstantin/combined) | **Recruitment** spec (shared Network + Muscle) |
| `Muscle` (geometry, territories, innervation, fibers) | **Muscle** stage (spatial) |
| `ForceModel`, `ForceModelVectorized`, `HillModel` | **Muscle** force backends (seam #2) |
| `JointDynamics`, `JointBiomechanics`, `*Geometry` | **Joint/biomechanics** stage (multi-muscle, roles/signs explicit) |
| `SpindleModel`, `GolgiTendonOrganModel` | **Afferents** stage (transduction) |
| `SurfaceEMG`, `IntramuscularEMG` | **EMG output** stages |
| `SurfaceElectrodeArray`, `IntramuscularElectrodeArray` | **ElectrodeArray** spec |
| `create_grid_signal`, `signal_to_grid`, `get_electrode/row/column`, `GridAnalogSignal` | **Data & IO** (grid annotations) |
| `create_*_current` (sinusoid/sawtooth/step/ramp/trapezoid), IClamp injection helpers | **Drive/Stimuli** spec |
| `set_random_seed`, `derive_subseed`, `reset_cell_id_counters`, `get_random_*` | **Reproducibility** service |
| `_setup_myogen`, NMODL compile/load, `get/validate/set_mechanism_param`, `BuildWithNMODL` | **Backend lifecycle** service |
| `bin_spike_trains`, `types.py`, `beartowertype` | **Core data adapters / units & validation** |
| `ContinuousSaver`, chunk combine, `export_to_nwb`, `validate_nwb` | **Data & IO** (persistence/export) |
| `emg_noise.*` (generation/calibration) | **EMG output** stage (attached) |
| `plotting.*`, `calculate_SD_FR`, `calculate_firing_rate_statistics`, `get_gamma_shape_for_mvc` | **Analysis/viz** layer (out of hot path) |

## 7. Net-new work this implies (not in the codebase today)

- The `SimState` shared buffer + zero-copy `numpy`/`torch`/`jax` views with explicit shape/dtype/device/ownership rules.
- Backend-agnostic `Stage` / `Backend` protocols (write-in-place, not return-value).
- Backend-agnostic `Simulation` driver + graph scheduler with feedback edges and one-tick-delay semantics (today's `SimulationRunner` is NEURON-specific).
- Backend-neutral population/connectivity spec (replacing direct `h.Section`/`h.NetCon`/synapse objects).
- `SimResult` facade + neo/NWB adapters; the `run().to_neo()` migration shim.
- FEM-ready muscle/force abstraction; multi-muscle/antagonist torque model with explicit roles/signs.
- `make()` one-liner + TOML/`Spec` config (current config is YAML + partial).
- The `JaxleyBackend` and `FEMBackend` themselves (separate, later efforts — this proposal only guarantees the seams exist for them).

## 8. Migration / clean-break plan

1. Build the new layered API alongside the current one on `feat/puffer-api`.
2. `NEURONBackend` reuses today's NEURON/NMODL/Cython machinery internally — no physics rewrite to ship v1.
3. Provide `run().to_neo()` ≡ old `run()` (identical Block structure) + thin shims so existing scripts/examples keep working during transition.
4. Port the examples and paper-reproduction scripts to the new API.
5. Deprecate the old surface over two releases, then remove.

## 9. Testing strategy (for the eventual implementation)

- **Parity tests:** new `NEURONBackend` path reproduces current `SimulationRunner` outputs (spike times, force, EMG) within tolerance on the existing examples — the safety net for the clean break.
- **Protocol/contract tests:** a `FakeBackend` and `FakeStage` exercise `SimState`/`Stage`/`Backend`/`Simulation` without NEURON, so the core is testable in CI without compiled deps.
- **Closed-loop tests:** a minimal reflex loop (drive → Network → Muscle → Afferents → Network) verifies one-tick-delay semantics and stability.
- **Data-format tests:** `SimResult.to_neo()`/`to_nwb()` round-trips equal the legacy Block; assert no `neo`/`quantities` import on the hot-path modules; assert zero-copy where promised (shared base pointer for C-contig float64).
- **Backend-swap smoke test:** the same `Spec` runs through `NEURONBackend`; a stub `JaxleyBackend` satisfies the protocol (interface-level, no physics) to prove the seam holds.

## 10. Open questions (for the team)

- **jaxley scope:** does it become the *default* batched backend with NEURON as the reference, or an opt-in alternative? (Affects how hard we push float32/GPU defaults.)
- **FEM coupling:** is the FEM muscle a per-step stage in the closed loop, or a higher-fidelity offline pass that precomputes MUAPs/territories? (Affects whether seam #2 is a step-backend or a geometry-backend.)
- **Analysis/viz:** keep `plotting.*` in-tree as `myogen.viz`, or split to a companion package?
- **Spec format:** TOML vs. dataclasses-as-primary (with TOML as one loader). Current config is YAML.

---

*This is a proposal to agree on shape before building. The next step after sign-off is an implementation plan (per-task, TDD) for the core layer + `NEURONBackend`, behind which jaxley and FEM land as separate efforts.*
