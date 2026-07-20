# NEURON Decoupling — Jaxley as Default Backend

Goal: make Jaxley the default/only *required* simulation backend and keep NEURON as
an *optional* extra (option A — retain NEURON as a validation reference, do not
delete it). Phased so each step is independently shippable.

## Phase 1 — Import decoupling (DONE)

`import myogen` and `import myogen.simulator` no longer require the NEURON runtime.
Verified by blocking `import neuron` (simulating a Jaxley-only install): `import
myogen`, `import myogen.simulator`, and a differentiable `run_jax` simulation all
succeed; the public classes resolve to their Jaxley implementations.

Changes:
- `myogen/simulator/__init__.py` — the eager `from myogen.simulator.neuron.* import`
  lines (the only thing forcing NEURON at import) now import the Jaxley equivalents
  (`HillModel`, `JointDynamics`, `Network`, `SpindleModel`, `GolgiTendonOrganModel`,
  `SimulationRunner`). Added a lazy `__getattr__` so `simulator.neuron` still works
  on demand when NEURON is installed.
- `myogen/utils/continuous_saver.py` — moved `from neuron import h` from module level
  into the two methods that use the NEURON clock (it's a NEURON step-callback util).
- `myogen/utils/nmodl.py` — `load_nmodl_mechanisms` now stays silent (instead of
  warning) when NEURON is simply absent and `quiet=True`; still raises under `strict`.

Backward compatibility (NEURON installed): fully preserved. All NEURON usage goes
through the explicit namespace — `from myogen.simulator.neuron.<sub> import ...` and
`from myogen.simulator.neuron import Network` (lazy `__getattr__`), plus
`simulator.neuron.*`. No real code depended on top-level `simulator.Network`/
`HillModel`/etc. being the NEURON version (only docstring references existed). The
full differentiable-pipeline + regression test suite remains green.

Already-favourable preconditions found (no change needed):
- `neuron/__init__.py` already lazy-loads via `__getattr__`.
- `_setup_myogen` is defined but not auto-called at import.
- The `core/` muscle/force/EMG pipeline is simulator-agnostic; the one cross-import
  (`simulate_fiber.py` → `neuron/_cython/_simulate_fiber`) is inside a try/except with
  a numpy fallback, so it degrades gracefully.

## Phase 2 — Packaging (DONE, with one CI-only verification pending)

Jaxley-only installs no longer pull in NEURON/MPI, and the build no longer hard-
requires NEURON.

Changes:
- `pyproject.toml` — moved `neuron`, `mpi4py`, `impi-rt` out of core `dependencies`
  into a new optional extra: `pip install "myogen[neuron]"`. (`mpi4py`/`impi-rt` are
  MPI companions used only by NEURON — confirmed no Python source imports `mpi4py`.)
- `setup.py` — `BuildWithNMODL.compile_nmodl` previously **raised** a RuntimeError on
  Windows when NEURON was absent, hard-requiring it at build time. Now it warns and
  continues on every platform (NMODL is only needed for the optional NEURON backend);
  users install `myogen[neuron]` then run the `setup_myogen` task to compile mechanisms.

Verified: `pyproject.toml` parses and core deps contain no NEURON/MPI; `setup.py`
parses and single-threaded `cythonize` of the extensions works; `import myogen` + a
Jaxley `run_jax` simulation runs with `neuron` blocked.

### Closing the "installs without NEURON" verification

The definitive test is that the built wheel installs and runs on a machine with **no
NEURON, MPI, or compiler**. This is now covered three ways:

1. **CI job (definitive).** `.github/workflows/build-wheels.yml` gains a
   `test_wheels_jaxley_only` job that downloads the built wheel, installs it with **no**
   NEURON/MPI (and asserts `import neuron` fails), then imports `myogen`, the
   differentiable API, and builds real NERLab motor neurons on the Jaxley backend.
   `publish` now depends on it, so a release cannot ship unless the Jaxley-only install
   passes. (The pre-existing `test_wheels` job still covers the NEURON-installed path.)
2. **Build robustness.** `setup.py` reads `MYOGEN_CYTHONIZE_NTHREADS` (default 4) so
   process-pool-constrained environments can build fully serially with
   `MYOGEN_CYTHONIZE_NTHREADS=0` (+ `build_ext -j1`), avoiding the pool entirely.
3. **Local proof.** With that fallback, a full fresh Cython build of all six extensions
   was completed in the dev sandbox, and the freshly-built artifacts import and run a
   Jaxley simulation with `neuron` blocked. (GitHub runners handle the default
   parallel build; the serial fallback is only for constrained sandboxes.)

### Descoped from Phase 2: Cython relocation

Moving the misfiled `neuron/_cython/` kernels to `core/` is **not required** under
option A: the `neuron/` *subpackage* still ships and is importable without the NEURON
*runtime*, so the kernels (which are pure Cython, no NEURON) compile and load fine
where they are, and the one `core/` consumer (`simulate_fiber.py`) already has a numpy
fallback. Relocation only becomes necessary if the `neuron/` subpackage is ever
deleted outright — deferred with that deletion.

## Phase 3 — Examples (DONE, including the canonical rename)

Classified all 29 examples by backend and documented the split in `examples/README.md`:
**17 default (Jaxley / backend-agnostic) examples** run on a plain `pip install myogen`;
**12 NEURON-only examples** require `pip install "myogen[neuron]"`.

- **Verified**: a representative set of default examples (01, 04–07, 09, 12, plus the
  heavy `08_..._jaxley` and `11_..._jaxley`) resolve **all** their top-level imports with
  `neuron` blocked — they get past imports with no NEURON. (Examples 04–07/09 are `core/`
  computation; they only transitively imported NEURON before Phase 1.)
- **Quarantine documented**: the NEURON-only examples (`02/03/08/10/11` base,
  `02_finetune/01–04`, `watanabe/01–03`) use explicit NEURON imports and still run once
  the extra is installed. `examples/README.md` maps each to its `_jaxley` counterpart and
  states the `myogen[neuron]` requirement. The docs gallery's `reset_neuron` already
  no-ops gracefully when NEURON is absent (`docs/source/conf.py`).

Not yet ported to Jaxley: `02_finetune/01–04` and `watanabe/01–03` (tracked as follow-up;
runnable via the NEURON extra meanwhile).

**Canonical rename (DONE).** The five paired examples were renamed so the **Jaxley**
version is canonical: `XX_..._jaxley.py` → `XX_....py`, and the NEURON version moved to
`XX_..._neuron.py` (examples 02, 03, 08, 10, 11 in `01_basic`). Verified each canonical
`.py` is Jaxley-backed and resolves imports with `neuron` blocked, and each `_neuron.py`
is NEURON-backed. Stale in-docstring command/cross-references were updated; output
artifact filenames (`*_jaxley.pkl/png/csv`) were intentionally left unchanged because
downstream examples read those exact files. No `.rst`/gallery config hard-codes example
filenames (auto-discovered via `filename_pattern`), so no docs config change was needed;
`examples/README.md` documents the canonical/NEURON pairing. Note: generated gallery URLs
change (e.g. `auto_examples/.../11_simulate_spinal_network.html` now shows the Jaxley
version) — expected with a rename.

## Not doing (option A)

Deleting the `neuron/` tree, NMODL files, and MPI dependency (the old plan's M7) is
deferred indefinitely: NEURON stays as an optional validation reference until frozen
reference datasets exist and the example surface is fully ported.
