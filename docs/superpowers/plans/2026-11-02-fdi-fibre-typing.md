# Fibre Typing by Motor Unit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every motor unit of `Muscle` one fibre type and a mean fibre diameter that rises with recruitment threshold across the whole pool, so intramuscular fibre CVs rise with threshold.

**Architecture:** A pure module `fiber_types.py` builds the unit ladder (types, quantile positions, derived spreads) from fibre counts and thresholds, and draws fibre diameters from it. `Muscle._generate_fiber_properties` calls it after `assign_mfs2mns`, then derives CVs as today with new defaults 4.32 ± 0.71 m/s.

**Tech Stack:** Python ≥3.12, NumPy, SciPy (`scipy.stats.norm`, `truncnorm`, `scipy.optimize.brentq`), quantities, beartype (`beartowertype`), pytest.

**Spec:** `docs/superpowers/specs/2026-11-02-fdi-fibre-typing-design.md` (Approved 2026-11-02).

## Global Constraints

- Branch `calib/fdi-dd-main`, worktree `C:\Users\DRRohlf\PycharmProjects\MyoGen\.claude\worktrees\fdi-dd-main`; tests run with `PYTHONPATH="$(pwd -W);c:\Neuron\lib\python"` and `/c/Users/DRRohlf/PycharmProjects/MyoGen/.venv/Scripts/python.exe`.
- Commits: `git -c user.name="Devon Rohlf" -c user.email="devon.rohlf@fau.de" commit`, no Co-Authored-By trailers. No push.
- Every sourced constant carries an inline register key with locator, e.g. `[polgar1973 p. 310 Table 2]`; unsourced values are labelled "modelling assumption". After adding citations run `scripts/build_literature.py --nexus "C:/Users/DRRohlf/Documents/FAU/Syncthing/Obsidian/Claude/projects/MyoGen_sEMG/Literature register.md"`.
- Values: type I fibre fraction 0.574 [polgar1973 p. 312 Table 4]; type means 50.0 / 67.0 µm [polgar1973 p. 310 Table 2]; per-type log-SD 0.2 (modelling assumption); fibre CV defaults 4.32 ± 0.71 m/s [delvecchio2018 p. 2]; CV sanity clip 2–6 m/s (modelling assumption); truncation ±3 SD.
- No change to fibre placement, `assign_mfs2mns`, `SurfaceEMG`, `HillModel`, `ForceModel` (spec §3, §4).
- Never use bare `git stash`. Don't kill python processes you didn't start.

## Review Focus

1. A unit left with zero fibres by `assign_mfs2mns` must still get a type and a ladder position, with no division by zero (Task 1 test `test_empty_unit_gets_a_type_and_position`).
2. Tied recruitment thresholds must give a deterministic order (stable sort by unit index) (Task 1 test `test_tied_thresholds_order_by_index`).
3. Inputs with no type II fibres (one unit, or `type_i_fiber_fraction` 1.0) must raise a clear `ValueError`, not a `brentq` sign error (Task 1 test `test_degenerate_typing_raises`).
4. Calling `_generate_fiber_properties()` before `assign_mfs2mns()` must raise `ValueError` naming `assign_mfs2mns` (Task 2 test `test_fibre_properties_need_the_assignment`).
5. A per-type log-SD so small that `sigma_within² ≤ 0` must raise `ValueError` (Task 1 test `test_too_small_type_spread_raises`).

---

### Task 1: Unit ladder and diameter draw (pure functions)

**Files:**
- Create: `myogen/simulator/core/muscle/fiber_types.py`
- Test: `tests/test_fiber_types.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class FiberTypeLadder` with fields `unit_types: np.ndarray` (int, 1 or 2, per unit), `unit_z: np.ndarray` (float, `clip(Φ⁻¹(q_u), −3, 3)` per unit), `sigma_between: float`, `sigma_within: float`, `type_i_fiber_fraction: float` (realised), `target_mean__um: float`.
  - `fiber_type_ladder(fibers_per_unit: np.ndarray, recruitment_thresholds: np.ndarray, type_i_fiber_fraction: float = 0.574, type_mean_diameters__um: tuple[float, float] = (50.0, 67.0), type_log_sd: float = 0.2) -> FiberTypeLadder`
  - `draw_fiber_diameters__um(ladder: FiberTypeLadder, assignment: np.ndarray, rng: np.random.Generator) -> np.ndarray` (per-fibre diameters in µm, aligned with `assignment`).

Algorithm (spec §5.1–5.3), which the tests do not fully determine:

```text
order    = argsort(thresholds, kind="stable")
cum      = cumsum(n[order]);  q[order] = (cum - n[order]/2) / cum[-1]
k        = argmin |cum/cum[-1] - f|;  units order[:k+1] are type 1, rest type 2
unit_z   = clip(norm.ppf(q), -3, 3)
sigma_b  = brentq(ratio(s) - m1/m2, 1e-3, 2.0), ratio(s) = fibre-weighted mean of exp(s*unit_z) over type 1 / over type 2
mix      = type_log_sd**2 + f*(1-f)*ln(m2/m1)**2        # f, m1, m2 nominal
sigma_w  = sqrt(mix - sigma_b**2 * average(unit_z**2, weights=n))
draw:    S = hypot(sigma_b, sigma_w); per fibre z ~ truncnorm((-3S - sigma_b*unit_z[u])/sigma_w, (3S - sigma_b*unit_z[u])/sigma_w)
         d = exp(sigma_b*unit_z[u] + sigma_w*z); d *= target_mean / mean(d),  target_mean = f*m1 + (1-f)*m2
```

- [ ] **Step 1: Write the failing tests** in `tests/test_fiber_types.py`. Use a synthetic pool: 120 units, `n = np.round(np.geomspace(40, 1600, 120)).astype(int)`, `thresholds = np.geomspace(1, 30, 120)`, `assignment = np.repeat(np.arange(120), n)`, `rng = np.random.default_rng(0)`.

```python
def test_types_follow_threshold_and_hit_the_fraction():
    lad = fiber_type_ladder(n, thresholds)
    order = np.argsort(thresholds, kind="stable")
    t = lad.unit_types[order]
    assert set(t) == {1, 2} and np.all(np.diff(t) >= 0)          # no type 2 below a type 1
    assert abs(lad.type_i_fiber_fraction - 0.574) <= n.max() / n.sum()

def test_ladder_rises_across_the_type_boundary():
    lad = fiber_type_ladder(n, thresholds)
    assert np.all(np.diff(lad.unit_z[np.argsort(thresholds, kind="stable")]) >= 0)

def test_diameters_keep_polgar_means_and_spread():
    lad = fiber_type_ladder(n, thresholds)
    d = draw_fiber_diameters__um(lad, assignment, rng)
    t = lad.unit_types[assignment]
    assert d[t == 1].mean() == pytest.approx(50.0, abs=0.5)
    assert d[t == 2].mean() == pytest.approx(67.0, abs=0.5)
    assert d.mean() == pytest.approx(0.574 * 50.0 + 0.426 * 67.0, abs=0.05)
    assert np.log(d).std() == pytest.approx(np.sqrt(0.04 + 0.574 * 0.426 * np.log(67 / 50) ** 2), abs=0.01)

def test_tails_are_cut_at_three_sd():
    lad = fiber_type_ladder(n, thresholds)
    d = draw_fiber_diameters__um(lad, assignment, rng)
    S = np.hypot(lad.sigma_between, lad.sigma_within)
    assert np.log(d).max() - np.log(d).min() <= 6 * S + 1e-9
```

Plus the Review Focus tests:

```python
def test_empty_unit_gets_a_type_and_position():
    m = n.copy(); m[5] = 0
    lad = fiber_type_ladder(m, thresholds)
    assert lad.unit_types[5] in (1, 2) and np.isfinite(lad.unit_z).all()

def test_tied_thresholds_order_by_index():
    th = thresholds.copy(); th[10] = th[11]
    a, b = fiber_type_ladder(n, th), fiber_type_ladder(n, th)
    np.testing.assert_array_equal(a.unit_z, b.unit_z)
    assert a.unit_z[10] <= a.unit_z[11]

@pytest.mark.parametrize("counts, ths, f", [(np.array([100]), np.array([1.0]), 0.574), (n, thresholds, 1.0)])
def test_degenerate_typing_raises(counts, ths, f):
    with pytest.raises(ValueError, match="type II"):
        fiber_type_ladder(counts, ths, type_i_fiber_fraction=f)

def test_too_small_type_spread_raises():
    with pytest.raises(ValueError, match="sigma_within"):
        fiber_type_ladder(n, thresholds, type_log_sd=0.01)
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_fiber_types.py -q`
Expected: collection error, `ModuleNotFoundError: myogen.simulator.core.muscle.fiber_types`.

- [ ] **Step 3: Implement** `FiberTypeLadder`, `fiber_type_ladder`, `draw_fiber_diameters__um` in `fiber_types.py` per the algorithm above. Module docstring cites the spec and the sources: [polgar1973 p. 310 Table 2], [polgar1973 p. 312 Table 4], [garnett1979 p. 42] (strict order is a simplification), [delvecchio2018 p. 2]; label type log-SD 0.2 and the ±3 SD truncation as modelling assumptions. `ValueError` messages: "no type II fibres …" when the type 2 set is empty; "sigma_within² ≤ 0 …" for the spread case.

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_fiber_types.py -q`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add myogen/simulator/core/muscle/fiber_types.py tests/test_fiber_types.py
git commit -m "feat(muscle): fibre-type ladder across the motor-unit pool"
```

### Task 2: Wire the ladder into `Muscle`

**Files:**
- Modify: `myogen/simulator/core/muscle/muscle.py` (constructor ~227–290 and docstring ~130–140; `_generate_fiber_properties` ~337–395; new properties after `muscle_fiber_conduction_velocities__mm_per_s` ~1050)
- Test: `tests/test_muscle_fibre_properties.py` (rebuild the two `_generate_fiber_properties` tests; delete `test_fibre_diameters_are_the_fdi_type_mixture`)

**Interfaces:**
- Consumes: `fiber_type_ladder`, `draw_fiber_diameters__um`, `FiberTypeLadder` (Task 1).
- Produces:
  - Constructor args: `type_i_fiber_fraction: float = 0.574`, `type_mean_fiber_diameters__um: tuple[float, float] = (50.0, 67.0)`, `type_fiber_diameter_log_sd: float = 0.2`; defaults changed: `mean_fiber_conduction_velocity__m_per_s = 4.32 * pq.m / pq.s`, `sd_fiber_conduction_velocity__m_per_s = 0.71 * pq.m / pq.s`.
  - Properties: `motor_unit_fiber_types -> np.ndarray` (per unit, 1/2), `muscle_fiber_types -> np.ndarray` (per fibre), `fiber_type_ladder -> FiberTypeLadder`; each raises `ValueError` like `muscle_fiber_diameters__mm` when not yet computed.

- [ ] **Step 1: Write the failing tests.** Add a module-scoped fixture that builds one small assigned muscle: `set_random_seed(1)`, `_muscle(3)` (20 units), `distribute_innervation_centers()`, `generate_muscle_fiber_centers(verbose=False)`, `assign_mfs2mns(verbose=False, n_jobs=1)`, `_generate_fiber_properties()`. Tests:

```python
def test_fibre_properties_need_the_assignment():
    set_random_seed(1); m = _muscle(3)
    m.generate_muscle_fiber_centers(verbose=False)
    with pytest.raises(ValueError, match="assign_mfs2mns"):
        m._generate_fiber_properties()

def test_each_unit_has_one_type_ordered_by_threshold(typed_muscle):
    m = typed_muscle
    a = m._assignment.astype(int)
    np.testing.assert_array_equal(m.muscle_fiber_types, m.motor_unit_fiber_types[a])
    t = m.motor_unit_fiber_types[np.argsort(m.recruitment_thresholds, kind="stable")]
    assert np.all(np.diff(t) >= 0)

def test_designed_unit_diameter_rises_with_threshold(typed_muscle):
    lad = typed_muscle.fiber_type_ladder
    assert np.all(np.diff(lad.unit_z[np.argsort(typed_muscle.recruitment_thresholds, kind="stable")]) >= 0)

def test_type_means_are_polgar(typed_muscle):
    d = np.asarray(typed_muscle.muscle_fiber_diameters__mm.rescale(pq.mm)) * 1e3
    t = typed_muscle.muscle_fiber_types
    assert d[t == 1].mean() == pytest.approx(50.0, abs=0.5) and d[t == 2].mean() == pytest.approx(67.0, abs=0.5)
```

Rebuild `test_fibre_cv_has_the_requested_distribution_ordered_by_diameter` on the same build path (per-parameter build, not the fixture), parametrised `({}, 4.32, 0.71)` and the custom `(4.2, 0.4)` case; tolerances `abs=0.03` on mean and SD (the 6 m/s clip trims the top); keep `corrcoef(d, cv) > 0.9`, keep the surface `mean_conduction_velocity` = 5.0 check; replace the 2.0 pile-up check with `np.mean(np.isclose(cv, 2.0) | np.isclose(cv, 6.0)) < 0.03`. Add a unit-level check: unit-mean CV Spearman with threshold > 0 (reported value, not gated harder).

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_muscle_fibre_properties.py -q`
Expected: failures on missing `motor_unit_fiber_types` / no `ValueError` / CV mean 3.89.

- [ ] **Step 3: Implement.** In `_generate_fiber_properties`: raise `ValueError("... Call assign_mfs2mns() before generating fiber properties.")` if `self._assignment is None`; build the ladder from `np.bincount(self._assignment.astype(int), minlength=self._number_of_neurons)` and `np.asarray(self._recruitment_thresholds)`; draw with `get_random_generator()`; store `self._fiber_type_ladder`, set diameters in mm. CV block unchanged except comment and defaults (4.32 ± 0.71 m/s, all-unit HD-sEMG CV, range 2.78–6.21 [delvecchio2018 p. 2]; anchored over fibres, see spec §5.2). Replace the Polgar-mixture comment with one that cites the ladder and points to `fiber_types.py`. Update the class docstring for the three new arguments, the new CV defaults, and the consequence "~86 % of FDI units are type I because fibres concentrate in large units" (spec §5.1). Initialise `self._fiber_type_ladder = None` in `__init__`.

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_muscle_fibre_properties.py tests/test_fiber_types.py tests/test_determinism.py -q`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add myogen/simulator/core/muscle/muscle.py tests/test_muscle_fibre_properties.py
git commit -m "feat(muscle)!: type fibres by motor unit; fibre CV 4.32 +/- 0.71 m/s"
```

### Task 3: FDI check, docs, register, full suite

**Files:**
- Modify: `CHANGELOG.md`, `docs/pr/2026-11-01-upstream-changes.md`, `scripts/build_literature.py` (+ regenerated `docs/literature.md`), `docs/superpowers/specs/2026-11-02-fdi-fibre-typing-design.md` (§8 untouched; add "Implemented" line)
- Test: full suite

- [ ] **Step 1: Run the FDI check** on the default FDI muscle (`RecruitmentThresholds(N=120, recruitment_range__ratio=30, mode="fuglevand")`, `Muscle(..., autorun=True)`, seed 1) from a scratchpad script; record: type I units / realised fraction, `sigma_between` / `sigma_within`, type means, per-type and pool log-SD, fibre range, fibre CV mean ± SD, % clipped at 2 / 6 m/s, unit-mean CV range, R² and Spearman of unit-mean CV vs threshold.
Expected (spec §5.2): ~102 type I units, 0.572, σ 0.184 / 0.166, means 50.0 / ~67.0, pool log-SD ~0.246, range ~26–117 µm, CV ~4.31 ± 0.69, ~2 % at 6 m/s, R² ~0.95. A value off by more than the spec tables' rounding: stop and report.

- [ ] **Step 2: Docs.** CHANGELOG entry (breaking for iEMG: diameters and CVs move between units; fibre CV default 3.89 ± 0.50 → 4.32 ± 0.71; `_generate_fiber_properties` now requires `assign_mfs2mns`; upgrade note: pass the old CV values explicitly to reproduce previous CV moments). PR doc: tick "Fibre typing by motor unit" with commits and the Step 1 table. Register: `polgar1973` notes (ladder replaces the per-fibre mixture), `delvecchio2018` notes (4.32 ± 0.71 sets the Muscle fibre CV defaults since 2026-11-02), `garnett1979` notes (cited in code). Rebuild the register.

- [ ] **Step 3: Full suite**

Run: `python -m pytest tests -q -p no:cacheprovider > <scratchpad>/full_suite_typing.txt` (background; ~10 min), then read the summary line.
Expected: `452+ passed, 2 skipped`, no failures. Any failure: report by name with output.

- [ ] **Step 4: Commit**

```bash
git add CHANGELOG.md docs/pr/2026-11-01-upstream-changes.md scripts/build_literature.py docs/literature.md docs/superpowers/specs/2026-11-02-fdi-fibre-typing-design.md
git commit -m "docs: fibre typing by motor unit (changelog, PR doc, register)"
```

## Open item for Devon (does not block Tasks 1–3)

- **6 m/s CV clip.** With 4.32 ± 0.71 m/s about 2 % of fibres pile at 6 m/s, below Del Vecchio's maximum unit CV 6.21 m/s [delvecchio2018 p. 2]. The plan keeps 2–6 m/s (a modelling assumption) and tests the pile-up < 3 %. Raising the upper bound would be a default change.
