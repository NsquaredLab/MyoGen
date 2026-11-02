# Surface EMG reads the fibres of `Muscle` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `SurfaceFiberDispersion(fiber_properties="muscle")` makes surface EMG use every fibre's CV and diameter from `Muscle`; then refit λ on the Botelho FDI benchmark.

**Architecture:** One new config field plus validation and a preset (`fiber_dispersion.py`); `realize_fiber_dispersion` takes per-unit per-fibre CV and diameter arrays; `SurfaceEMG.simulate_muaps` splits `Muscle`'s per-fibre arrays by unit and passes them down. Benchmark harness gains the mode; eight λ arms run sequentially.

**Tech Stack:** Python 3.12, numpy, quantities, pytest.

**Spec:** `docs/superpowers/specs/2026-11-03-surface-emg-muscle-fibres-design.md`

## Global Constraints

- No default changes: `SurfaceFiberDispersion()` and `fiber_properties="config"` stay bit-identical to today.
- Old CV options mutually exclusive with `"muscle"` (ValueError naming the field).
- `"muscle"` + `diameter_scales_source=True`: `scale_i = (d_i / fiber_diameter__um)²`, not normalised.
- Commits: `git -c user.name="Devon Rohlf" -c user.email="devon.rohlf@fau.de" commit`, no Co-Authored-By trailer, explicit pathspecs; no push; never `git stash`.
- Sourced constants cite register keys with locators read; unsourced values are labelled modelling assumptions.
- Env: `export PYTHONPATH="$(pwd -W);c:\\Neuron\\lib\\python"`; `PY=/c/Users/DRRohlf/PycharmProjects/MyoGen/.venv/Scripts/python.exe`.

## Review Focus

1. Fibre order: `Muscle` per-fibre arrays split by `assignment == mu` must match `resulting_fiber_assignment(mu)` order (test: CV per fibre equals the `Muscle` value at the same position).
2. A `Muscle` built without `_generate_fiber_properties()` (e.g. manual `assign_mfs2mns` only) must raise a clear error, not use stale or None arrays.
3. `fiber_diameter__um` must have no effect in `"muscle"` + scaling (test: two values → identical MUAPs).
4. Units with zero fibres (empty arrays) must not break realisation.
5. `"config"` mode output unchanged (existing bit-identical test plus the literature preset test).

---

### Task 1: Config field, validation, preset

**Files:** Modify `myogen/simulator/core/emg/surface/fiber_dispersion.py`; Test `tests/test_surface_dispersion.py`.

**Interfaces — Produces:** `SurfaceFiberDispersion.fiber_properties: Literal["config", "muscle"] = "config"`; `SurfaceFiberDispersion.muscle_candidate(seed: int = 0) -> SurfaceFiberDispersion`; module constant `FIBER_PROPERTY_SOURCES = ("config", "muscle")`.

- [ ] Step 1: tests
  - `test_muscle_fibre_source_is_active`: `SurfaceFiberDispersion(fiber_properties="muscle").active` is True.
  - `test_muscle_fibre_source_rejects_surface_cv_options` parametrised over `fiber_cv_threshold_slope=0.63`, `fiber_cv_inter_mu_spread=0.3`, `fiber_cv_within_mu_cov=0.05`, `fiber_cv_per_unit__m_s=(4.0,)`: `pytest.raises(ValueError, match=<field name>)`.
  - `test_fibre_property_source_must_be_known`: `fiber_properties="x"` raises ValueError matching `fiber_properties`.
  - `test_muscle_candidate_preset`: equals `literature_candidate()` with `fiber_properties="muscle"`, slope 0, within-cov 0, `diameter_scales_source=True` (compare via `dataclasses.replace`); seed passes through.
- [ ] Step 2: run, expect FAIL (unexpected keyword / no attribute).
- [ ] Step 3: implement field, `__post_init__` checks, `active`, preset, docstring (`fiber_properties` paragraph; note on `diameter_scales_source` in `"muscle"` mode).
- [ ] Step 4: `$PY -m pytest tests/test_surface_dispersion.py -q -p no:cacheprovider` → all pass.
- [ ] Step 5: commit `feat(semg): fibre property source option for surface dispersion`.

### Task 2: Realisation from `Muscle` arrays

**Files:** Modify `fiber_dispersion.py` (`realize_fiber_dispersion`); Test `tests/test_surface_dispersion.py`.

**Interfaces — Consumes:** Task 1 field. **Produces:** `realize_fiber_dispersion(..., fiber_cv__m_s: Optional[Sequence[np.ndarray]] = None, fiber_diameters__um: Optional[Sequence[np.ndarray]] = None)`.

- [ ] Step 1: tests (use the module `muscle` fixture; build synthetic per-unit arrays `cvs = [np.linspace(3, 5, n) for n in counts]`, `ds = [np.linspace(40, 80, n) for n in counts]`):
  - `test_muscle_source_uses_the_given_fibre_cvs`: realised `cv__m_s[mu]` equals `cvs[mu]` exactly; `report["cv_mean__m_s"]` their means.
  - `test_muscle_source_scales_by_absolute_cross_section`: with `diameter_scales_source=True`, `fiber_diameter__um=50.0`: `amplitude_scale[mu] == (ds[mu]/50)**2` (allclose, rtol 1e-12) — no normalisation; `report["diameter_mean__um"][mu] == ds[mu].mean()`; `line_diameter_mean__um` NaN.
  - `test_muscle_source_without_scaling_keeps_unit_scale`: `diameter_scales_source=False` → all scales 1.
  - `test_muscle_source_keeps_the_other_draws`: same seed, `nmj_model="two_layer_intra"`, `nerve_delays=True`, tendon band 5: `zi__mm`, `delay__ms`, `end_plus__mm` identical between `"config"` and `"muscle"` realisations.
  - `test_muscle_source_needs_the_arrays`: missing `fiber_cv__m_s` → ValueError matching `fiber_cv__m_s`; scaling on with missing diameters → ValueError matching `fiber_diameters__um`; a unit with wrong length → ValueError matching `motor unit`.
- [ ] Step 2: run, expect FAIL (unexpected keyword).
- [ ] Step 3: implement. In the unit loop, `"muscle"` replaces the `cv_mu`/cov block with `cv = np.asarray(fiber_cv__m_s[mu], float)`; scale `(d / fiber_diameter__um)**2` (requires `fiber_diameter__um`), skip the pool normalisation; report as spec §5.2. CV RNG not consumed in `"muscle"` mode (other generators unchanged).
- [ ] Step 4: run file → pass.
- [ ] Step 5: commit `feat(semg): realise dispersion from Muscle fibre CVs and diameters`.

### Task 3: `SurfaceEMG` wiring

**Files:** Modify `myogen/simulator/core/emg/surface/surface_emg.py` (dispersion block ~362-385; `fiber_diameter__um` / `fiber_dispersion` docstrings); Test `tests/test_surface_dispersion.py`.

**Interfaces — Consumes:** Task 2 signature; `Muscle.muscle_fiber_conduction_velocities__mm_per_s`, `Muscle.muscle_fiber_diameters__mm`, `Muscle.assignment`.

- [ ] Step 1: tests (module fixture `typed`: the existing `_muscle()` plus `muscle._generate_fiber_properties()`; helper `_semg(...)` gains `**kw` passed to `SurfaceEMG`):
  - `test_surface_emg_takes_each_fibre_from_muscle`: `semg = _semg(typed, SurfaceFiberDispersion(fiber_properties="muscle"))`, `_muaps(semg)`; report `cv_mean__m_s[mu]` equals the mean of `Muscle` CVs (m/s) over `assignment == mu` for simulated units (rel 1e-12), `diameter_mean__um[mu]` equals the mean `Muscle` diameter (µm).
  - `test_muscle_fibres_make_fibre_diameter_irrelevant`: `fiber_properties="muscle", diameter_scales_source=True` with `fiber_diameter__um=40.0` vs `70.0` → MUAPs allclose (rtol 1e-10).
  - `test_muscle_fibres_change_the_muap`: `"muscle"` MUAPs differ from undispersed ones.
  - `test_muscle_fibres_need_generated_properties`: `_muscle()` (no properties) with `"muscle"` → `simulate_muaps` raises ValueError matching `_generate_fiber_properties`.
- [ ] Step 2: run, expect FAIL (report CVs equal 4.2-based values / no error).
- [ ] Step 3: implement: in `"muscle"` mode, read the two `Muscle` arrays (catch the property's ValueError and re-raise with the required-call message), convert to m/s and µm, split with `assignment == mu`, pass to `realize_fiber_dispersion`. Docstrings: `fiber_diameter__um` no effect under `"muscle"` + scaling; mean CV is then only the grid velocity.
- [ ] Step 4: `$PY -m pytest tests/test_surface_dispersion.py -q -p no:cacheprovider` → pass.
- [ ] Step 5: commit `feat(semg): SurfaceEMG reads Muscle fibres for dispersion`.

### Task 4: Benchmark arms, docs

**Files:** Modify `benchmarks/semg_dispersion/gen_arm.py`, `benchmarks/semg_dispersion/arms.json`, `benchmarks/semg_dispersion/README.md` (if it lists arms), `CHANGELOG.md`, `docs/pr/2026-11-01-upstream-changes.md`, `docs/backlog.md`.

- [ ] Step 1: `gen_arm.py`: if `config.get("fiber_properties") == "muscle"`: `state = get_random_generator().bit_generator.state`; `muscle._generate_fiber_properties()`; restore state; `property_hash` = sha1 of diameters + CVs bytes, saved and printed. Docstring line for the mode.
- [ ] Step 2: `arms.json` fdi: `delsys5r_wexp_muscle_lam{0.6,0.65,0.7,0.75,0.8,0.85,0.9,1}` = `delsys5r_wexp_lam0.7_cvrtforce` minus `_fiber_cv_per_unit_file`, `_mean_cv__m_s`, with `fiber_properties: "muscle"`, `fiber_cv_within_mu_cov: 0.0`, `diameter_scales_source: true`, `_iap_lambda__per_mm` per arm.
- [ ] Step 3: smoke: build the first arm's config and run `gen_arm.py` with `_delsys5` reduced to one sensor and `_fs__Hz` 2048 into the scratchpad (not committed) — expect a finished npz and a property hash. Check the global-state restore: fibre-length draws equal with/without the property call (quick python check in scratchpad).
- [ ] Step 4: CHANGELOG [Unreleased] entry; PR doc item; backlog "Other open items" update.
- [ ] Step 5: commit `bench(semg): Muscle-fibre arms for the λ refit; docs`.

### After the tasks

Full suite (`$PY -m pytest -q -p no:cacheprovider` to a file), final whole-branch review, fix pass; then launch the eight arms sequentially in the background (~40 min each), score with the Delsys cc/adj scorer (MDF 20 Hz–2 kHz, medians over sensors × pairs), pick λ per spec §6, update the `iap_lambda__per_mm` docstring, PR doc, handover; report to Devon.
