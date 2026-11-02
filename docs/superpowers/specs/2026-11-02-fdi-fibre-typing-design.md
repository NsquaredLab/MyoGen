# Fibre typing by motor unit (FDI) — design

Status: **Draft** (2026-11-02). Branch `calib/fdi-dd-main`. Requested by Devon ("that's an important step I'd like to do").

## 1. Why

`Muscle._generate_fiber_properties` draws each fibre's type independently (57.4 % type I, Polgar 1973), so every motor unit holds a mix of type I and type II fibres. Physiologically all fibres of a unit share one type, and unit properties are ordered by recruitment threshold. Because types and diameters are mixed within units, the intramuscular path has no CV–recruitment-threshold relation, although one is measured (Del Vecchio 2018: CV linear in threshold, R² 0.70 ± 0.09 [delvecchio2018 p. 2]; Nishizono 1990, FDI: CV vs threshold force r = 0.52 [nishizono1990 p. 79]).

## 2. Goals

1. Each motor unit has one fibre type; all its fibres carry it.
2. Type I units are the lowest-threshold units, and type I fibres make up 57.4 % of all fibres [polgar1973 p. 312 Table 4].
3. A unit's mean fibre diameter rises with its recruitment threshold across the whole pool, through the type boundary (option A, Devon 2026-11-02; §5.2); fibres within a unit scatter about that mean.
4. Each type's fibre-weighted mean diameter stays at Polgar's 50.0 / 67.0 µm [polgar1973 p. 310 Table 2]; the overall mean stays 57.2 µm.
5. Fibre CVs keep the current authority: ordered by diameter through the CV line plus 0.2 m/s scatter, rescaled to `mean_fiber_conduction_velocity__m_per_s` / `sd_fiber_conduction_velocity__m_per_s` (3.89 ± 0.50 m/s). With goal 3 this yields CV rising with threshold across the whole pool.
6. Expose per-unit and per-fibre types on `Muscle`.

## 3. Non-goals

- Surface EMG: `SurfaceEMG` does not read `Muscle` fibre diameters or CVs. Making the surface dispersion (per-unit CV, diameter-scaled source) read typed `Muscle` units is a follow-up (§8), not part of this change.
- `HillModel(n_motor_units_type1, n_motor_units_type2)` keeps its own arguments; deriving them from `Muscle` is a follow-up.
- `ForceModel` twitch properties stay as they are (Fuglevand 1993 ordering by threshold).
- Type IIa/IIx distinction: Polgar reports type I / II only [polgar1973 p. 308].

## 4. Anti-goals

- No reduced motor-unit counts in validation (FDI 120 units).
- No change to fibre placement or unit territories (`assign_mfs2mns` untouched; it runs before typing).
- No new unsourced constant without an "assumption" label in code and the literature register.

## 5. Design

### 5.1 Unit typing (decided by Devon: strict order)

Sort units by recruitment threshold. Walk up the order accumulating fibre counts; units are type I until the cumulative count reaches 57.4 % of fibres. The unit that crosses the boundary is type I if that leaves the type I fraction closer to 57.4 %, else type II. Report the realised type I fibre fraction.

Consequence (checked on the default FDI muscle, 120 units, 52 502 fibres, seed 1): 103 of 120 units (86 %) are type I, because fibres concentrate in the large, late-recruited units. This is stated in the docstring.

Sources and limits of strict order:
- Size principle in human units: recruitment in order of increasing size (Milner-Brown et al. 1973, FDI) [milnerbrown1973b].
- Garnett et al. 1979 (human medial gastrocnemius, 57 units, microstimulation + glycogen depletion + biopsy; read in full):
  - Type S units were small, slow and fatigue resistant; FR fast, intermediate, resistant; FF fast, large, fatigable [garnett1979 p. 33].
  - Glycogen depletion: S units were made of type 1 fibres, FF units of type 2b [garnett1979 p. 39–41].
  - Ranking units by size: "at low force levels both type S and type FR unit would be recruited. Above about 30 % of the total cumulative twitch tension only type FF units would be additionally recruited" [garnett1979 p. 42]. So type I and type IIa units interleave at low thresholds; only the type IIb (FF) units sit strictly at the top.
  - Limits: medial gastrocnemius, not FDI; recruitment order deduced from size, not measured; sampling biased against the largest and smallest units [garnett1979 p. 41]; one small FF unit (5.3 g wt) was type 2b [garnett1979 p. 39–40].
- Therefore strict order is a **modelling simplification**: it is right for the large late units (type II at the top) and ignores the type I / IIa interleaving at low thresholds that Garnett describes. Polgar gives only type I / II for FDI [polgar1973 p. 308], so the IIa share cannot be placed from FDI data. Overlap is follow-up §8.3.

### 5.2 Diameters: one ladder across the pool (option A)

Decided by Devon 2026-11-02 ("let's try A, but let's make sure to document this and the reason for it").

**Reason.** The first draft ranked units within each type separately. The last type I unit then sat near 70 µm and the first type II unit near 46 µm, so CV against threshold climbed through type I, dropped at the boundary and climbed again through type II. That contradicts the measured linear CV–threshold relation (Del Vecchio 2018 [delvecchio2018 p. 2]; Nishizono 1990, FDI [nishizono1990 p. 79]) and Casolo 2023's finding that CV follows fibre size, not type [casolo2023 p. 1845]. Garnett 1979 also gives no basis for a size drop at the type boundary: FF (type 2b) units were the large ones [garnett1979 p. 37].

**Design.** Units in threshold order; `q_u` = midpoint of the unit's share of the pool's cumulative fibre count (fibre-weighted, so fibre-level spreads match the targets).

- Unit log-mean: `mu_u = c + sigma_b · clip(Φ⁻¹(q_u), −3, 3)`. Deterministic, strictly increasing with threshold across the whole pool. Types are labels on this ladder (lower 57.4 % of fibres type I).
- Fibre: `d = exp(mu_u + sigma_w · z)`, `z` standard normal truncated at ±3.
- `sigma_b` is solved so the type I : type II fibre-weighted mean ratio is 50.0 : 67.0 [polgar1973 p. 310 Table 2] on the realised units.
- `sigma_w` is solved so the pool's fibre log-SD equals that of Polgar's two-type mixture (each type log-SD 0.2, today's assumption): `sqrt(0.2² + 0.574·0.426·ln(67/50)²) = 0.247`.
- `c` is set so the overall fibre-weighted mean is 57.2 µm; then the type means are 50.0 / 67.0 µm.

No free split is left: both spreads follow from Polgar's type means and fractions plus the existing 0.2 assumption.

**Check on the default FDI muscle** (120 units, 52 502 fibres, seed 1; scratch run 2026-11-02):

| Quantity | Option A | Target / today |
|---|---|---|
| Type I units / fibre fraction | 102 / 0.572 | 57.4 % |
| sigma_b / sigma_w | 0.184 / 0.166 | derived |
| Type means | 50.0 / 67.0 µm | 50.0 / 67.0 |
| Type log-SD | 0.202 / 0.192 | 0.2 / 0.2 |
| Pool log-SD | 0.246 | 0.247 |
| Fibre range | 18.6–134.8 µm (ladder not yet clipped) | today 26.9–119.7 |
| Unit-mean CV | 3.00–4.79 m/s | — |
| Unit-mean CV vs threshold | R² 0.94, Spearman 0.998 | Del Vecchio R² 0.70 ± 0.09 |

**Costs, stated plainly.**
- Each type's histogram is no longer a log-normal: type I and type II overlap only through within-unit scatter, not across units. Their means and log-SDs still match Polgar.
- The tails widen (smallest units' fibres reach ~19 µm before clipping the ladder at ±3; the clip is new in this revision and narrows that).
- The model's CV–threshold relation is tighter (R² 0.94) than Del Vecchio's measurement (0.70 ± 0.09). Their R² includes HD-sEMG estimation error and between-unit variability the ladder lacks. Reported, not fitted.

### 5.3 Keeping the type means

Handled by solving `sigma_b` and `c` on the realised units (§5.2); no per-type rescaling (that would reintroduce a step at the boundary). Report the realised type means; test to 0.5 µm.

### 5.4 Interface

- `Muscle.motor_unit_fiber_types` → `np.ndarray[int]` (1 or 2) per unit.
- `Muscle.muscle_fiber_types` → per fibre, aligned with `muscle_fiber_diameters__mm`.
- New constructor arguments with defaults: `type_i_fiber_fraction=0.574`, `type_mean_fiber_diameters__um=(50.0, 67.0)`; `type_fiber_diameter_log_sd=0.2` (the per-type spread from which `sigma_w` is derived; an assumption). `sigma_b`, `sigma_w` and `c` are derived, stored and reported, not arguments.
- `_generate_fiber_properties` requires `assign_mfs2mns` to have run (it raises otherwise; autorun already orders them so).

### 5.5 Tests (written first)

1. Every unit has one type; per-fibre type equals its unit's type.
2. Type I units are exactly the lowest-threshold ones (no type II unit below a type I unit).
3. Type I fibre fraction within one unit's worth of 57.4 %.
4. Type fibre-weighted means 50.0 / 67.0 µm (to 0.5 µm); overall ≈ 57.2 µm; pool log-SD ≈ 0.247.
5. The designed unit mean `exp(mu_u)` is non-decreasing in threshold across the whole pool, including across the type boundary (my call: designed, not realised — small units' realised means wander by ~0.02 in log against ~0.003 between neighbours).
6. Fibre CV mean/SD 3.89 / 0.50 (unchanged). Realised unit-mean CV vs threshold: Spearman and R² reported (not gated), compared with Del Vecchio's R² 0.70 ± 0.09.
7. Existing iEMG tests pass; full suite green. The two existing tests in `tests/test_muscle_fibre_properties.py` call `_generate_fiber_properties()` without `assign_mfs2mns`; they are rebuilt with a full assignment (my call).

## 6. Constraints

- Literature register entries for every constant (keys + locators actually read). Garnett added once read.
- `beartowertype` signatures, `Quantity` conventions as in `Muscle`.
- Breaking for iEMG outputs (diameters and CVs move between units); CHANGELOG + upgrade note.

## 7. Left to the builder

Exact boundary-unit rule tie-breaks; vectorisation; parameter names if the file's conventions suggest better ones.

## 8. Follow-ups (not in this change)

1. Surface dispersion reads typed `Muscle` units: per-unit CV and diameter-scaled source from `Muscle` instead of the threshold slope / per-unit CV file (one authority).
2. `HillModel` type counts from `Muscle`.
3. Optional overlap of types around the boundary if a source gives its width.

## 9. Sources

Polgar et al. 1973 (FT, read); Del Vecchio et al. 2018 (FT); Nishizono et al. 1990 (FT); Milner-Brown et al. 1973b (FT); Casolo et al. 2023 (FT; CV tracks size, not type, r = 0.20 [casolo2023 p. 1845]); Garnett et al. 1979 (FT, read; doi:10.1113/jphysiol.1979.sp012643).
