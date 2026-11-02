# Fibre typing by motor unit (FDI) — design

Status: **Draft** (2026-11-02). Branch `calib/fdi-dd-main`. Requested by Devon ("that's an important step I'd like to do").

## 1. Why

`Muscle._generate_fiber_properties` draws each fibre's type independently (57.4 % type I, Polgar 1973), so every motor unit holds a mix of type I and type II fibres. Physiologically all fibres of a unit share one type, and unit properties are ordered by recruitment threshold. Because types and diameters are mixed within units, the intramuscular path has no CV–recruitment-threshold relation, although one is measured (Del Vecchio 2018: CV linear in threshold, R² 0.70 ± 0.09 [delvecchio2018 p. 2]; Nishizono 1990, FDI: CV vs threshold force r = 0.52 [nishizono1990 p. 79]).

## 2. Goals

1. Each motor unit has one fibre type; all its fibres carry it.
2. Type I units are the lowest-threshold units, and type I fibres make up 57.4 % of all fibres [polgar1973 p. 312 Table 4].
3. Within each type, a unit's mean fibre diameter rises with its recruitment threshold; fibres within a unit scatter about that mean.
4. Each type's fibre-weighted mean diameter stays at Polgar's 50.0 / 67.0 µm [polgar1973 p. 310 Table 2]; the overall mean stays 57.2 µm.
5. Fibre CVs keep the current authority: ordered by diameter through the CV line plus 0.2 m/s scatter, rescaled to `mean_fiber_conduction_velocity__m_per_s` / `sd_fiber_conduction_velocity__m_per_s` (3.89 ± 0.50 m/s). With goal 3 this yields CV rising with threshold.
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

Source for strict order: the size principle in human units — Milner-Brown et al. 1973 (FDI: twitch force rises with threshold) [milnerbrown1973b]; Garnett et al. 1979 (human medial gastrocnemius, glycogen-depletion typing of single units; **to be read**, doi:10.1113/jphysiol.1979.sp012643). Until Garnett is read, the ordering-by-type is cited as a modelling assumption resting on the size principle.

### 5.2 Diameters

Per type T (median `m_T` chosen so the type's fibre-weighted mean is `mean_T`, §5.3):

- Unit log-mean: `mu_u = log(m_T) + sigma_b · Φ⁻¹(q_u)`, where `q_u` is the midpoint of the unit's share of its type's cumulative fibre count, units in threshold order (my call: fibre-weighted, so the type's fibre histogram keeps log-SD 0.2 and the §5.3 factor stays near 1; a unit-rank quantile would shrink the fibre-level spread to ~0.16 because fibres sit in the top-ranked units). Deterministic, strictly ordered.
- Fibre: `d = exp(mu_u + sigma_w · z)`, `z` standard normal truncated at ±3 (as now).
- `sigma_b² + sigma_w² = 0.2²` keeps today's total within-type spread (log-SD 0.2, itself an assumption).

**Open parameter — split of the 0.2.** Default proposal `sigma_b = sigma_w = 0.2/√2 ≈ 0.141` (equal split), labelled an assumption, unless Garnett (or another source on within-unit fibre-size variability) gives a basis. Diagnostic to report: the realised unit-mean-CV vs threshold R², to compare with Del Vecchio's 0.70 ± 0.09 [delvecchio2018 p. 2]. The split is the knob that moves that R²; it is reported, not fitted.

### 5.3 Keeping the type means

Large units hold more fibres and sit at the top of each type's rank order, so the fibre-weighted mean exceeds the unit-weighted mean. After drawing, multiply each type's diameters by `mean_T / (fibre-weighted realised mean)` so the type means are exactly 50.0 / 67.0 µm. Truncation bounds move with the scale factor; report the factor.

### 5.4 Interface

- `Muscle.motor_unit_fiber_types` → `np.ndarray[int]` (1 or 2) per unit.
- `Muscle.muscle_fiber_types` → per fibre, aligned with `muscle_fiber_diameters__mm`.
- New constructor arguments with defaults: `type_i_fiber_fraction=0.574`, `type_mean_fiber_diameters__um=(50.0, 67.0)`; the log-SD split as `fiber_diameter_log_sd_between_units=0.141`, `fiber_diameter_log_sd_within_unit=0.141` (names to match the file's conventions).
- `_generate_fiber_properties` requires `assign_mfs2mns` to have run (it raises otherwise; autorun already orders them so).

### 5.5 Tests (written first)

1. Every unit has one type; per-fibre type equals its unit's type.
2. Type I units are exactly the lowest-threshold ones (no type II unit below a type I unit).
3. Type I fibre fraction within one unit's worth of 57.4 %.
4. Type fibre-weighted means 50.0 / 67.0 µm (to 0.1 µm); overall ≈ 57.2 µm.
5. Within each type, the designed unit mean `exp(mu_u)` is non-decreasing in threshold (my call: designed, not realised — small units' realised means wander by ~0.02 in log against ~0.003 between neighbours).
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

Polgar et al. 1973 (FT, read); Del Vecchio et al. 2018 (FT); Nishizono et al. 1990 (FT); Milner-Brown et al. 1973b (FT); Casolo et al. 2023 (FT; CV tracks size, not type, r = 0.20 [casolo2023 p. 1845]); Garnett et al. 1979 (pending, PMC1281479).
