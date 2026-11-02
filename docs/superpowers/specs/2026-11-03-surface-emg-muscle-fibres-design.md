# Surface EMG reads the fibres of `Muscle` — design

Status: Approved in conversation 2026-11-03 (Devon: "use the same fibres for both, then refit λ"; answers to four design questions). Implemented overnight 2026-11-03.

## 1. Why

Since fibre typing (2026-11-02) `Muscle` gives every fibre a diameter and a conduction velocity (CV), ordered by motor unit and threshold, and intramuscular EMG uses them. Surface EMG builds its own fibre CVs (muscle mean 4.2 m/s times a threshold factor, or a per-unit CV file, plus an assumed 5 % within-unit spread) and reads the diameter back from each CV through the CV–diameter line. Two authorities describe the same fibres, and they disagree: on the default FDI muscle the active units at 25 % MVC are ~4 % slower in `Muscle` (3.72 vs 3.87 m/s) and twice as spread within a unit (11 % vs 5 %). One set of fibres for both signals removes the duplicate.

## 2. Goals

- Surface EMG can take every fibre's CV and diameter from `Muscle` — the same fibres intramuscular EMG uses.
- With diameter scaling on, each fibre's source scales with its own absolute cross-section (`d²`), not a pool-normalised one.
- Refit the IAP length scale λ on the Botelho et al. (2019) FDI benchmark with this fibre source.

## 3. Non-goals and anti-goals

- No change of defaults. Off stays off; `SurfaceFiberDispersion()` stays the undispersed model, bit-identical.
- No on/off/manual switch on `SurfaceEMG` and no profile system: those belong to the named-profiles design (docs/backlog.md item 1).
- Existing surface CV options (threshold slope, inter-unit spread, per-unit CV file, within-unit spread, line diameter) stay for ablations and reproducing earlier runs. Anti-goal: silently combining them with the `Muscle` source.
- Intramuscular EMG unchanged.
- λ is not fitted to RMS; RMS is reported, not tuned.

## 4. Decisions (Devon, 2026-11-03)

1. Opt-in, not a default.
2. Old CV options kept, mutually exclusive with the `Muscle` source (error when combined).
3. Absolute `Muscle` diameters set the source amplitude: `SurfaceEMG.fiber_diameter__um` no longer sets the level in this mode. `Muscle`'s pool RMS diameter is ~58.9 µm vs the 57.9 µm default, so amplitude rises ~3.5 % before any λ change.
4. λ refit on median frequency only, λ 0.60–0.90 in steps of 0.05.

## 5. Design

### 5.1 Configuration

New field `SurfaceFiberDispersion.fiber_properties: Literal["config", "muscle"] = "config"`.

- `"config"`: today's behaviour, unchanged.
- `"muscle"`: each fibre's CV is `Muscle.muscle_fiber_conduction_velocities__mm_per_s` (in m/s) for that fibre. Validation raises `ValueError` if any of `fiber_cv_threshold_slope`, `fiber_cv_inter_mu_spread`, `fiber_cv_within_mu_cov` is non-zero or `fiber_cv_per_unit__m_s` is set, naming the conflicting field. `cv_per_diameter__m_s_per_mm` / `cv_intercept__m_s` are not used.
- `active` is true when `fiber_properties == "muscle"`.

`diameter_scales_source` keeps its meaning (source scales with cross-section) and stays independent, so the diameter effect can still be ablated:
- with `"muscle"` and `True`: `scale_i = (d_i / fiber_diameter__um)²`, **not** normalised over the pool. The source's own `d²` factor uses `fiber_diameter__um`, so the product is exactly each fibre's own `d_i²`: `fiber_diameter__um` cancels and has no effect.
- with `"muscle"` and `False`: scale 1; every fibre's source uses `fiber_diameter__um` (CVs still from `Muscle`).
- with `"config"`: unchanged (line diameter, normalised to mean 1).

### 5.2 Realisation

`realize_fiber_dispersion(..., fiber_cv__m_s=None, fiber_diameters__um=None)`: two new optional arguments, per unit lists of per-fibre arrays in the same order as `fiber_positions__mm`. Required (ValueError) when `fiber_properties == "muscle"`; their per-unit lengths must equal the position counts (ValueError naming the unit). Ignored otherwise. End-plates, nerve delays and tendon ends are drawn exactly as now, from the same seeded generators, so switching the CV source leaves those draws unchanged.

Report in `"muscle"` mode: `cv_mean__m_s`, `cv_cov` from the `Muscle` CVs; `diameter_mean__um` = the mean diameter the source uses (the `Muscle` diameter with scaling on, `fiber_diameter__um` with it off; existing meaning kept, builder check Q5); `line_diameter_mean__um` NaN; `amplitude_scale_mean` as defined above.

### 5.3 `SurfaceEMG`

When the dispersion is active with `"muscle"`, `simulate_muaps` reads `muscle_fiber_conduction_velocities__mm_per_s` and `muscle_fiber_diameters__mm` and splits them per unit with the boolean mask `assignment == mu`, which has the order of `resulting_fiber_assignment(mu)` (both are `np.where` order). A `Muscle` without generated fibre properties raises `ValueError` saying to run `_generate_fiber_properties()` after `assign_mfs2mns()` (or build with `autorun=True`).

`mean_conduction_velocity__m_per_s` keeps one role in this mode: the velocity of the shared simulation grid (`v_grid`, window `kernel_length / v`), which does not change the physical MUAP. The existing window check (slow fibres must finish inside the window) applies; slow `Muscle` fibres (≥ 2 m/s) may need a longer `iap_kernel_length__mm`, and the error says so.

### 5.4 Preset

`SurfaceFiberDispersion.muscle_candidate(seed=0)`: `literature_candidate` with `fiber_properties="muscle"`, `fiber_cv_threshold_slope=0`, `fiber_cv_within_mu_cov=0`, `diameter_scales_source=True` (end-plates, nerve delays, tendon ends as in `literature_candidate`). This is the bundle the future `"physiological"` profile would use for surface EMG.

## 6. Benchmark and λ refit

`benchmarks/semg_dispersion/gen_arm.py`: when the arm sets `fiber_properties: "muscle"`, call `muscle._generate_fiber_properties()` after `assign_mfs2mns`, with the global generator's state saved before and restored after, so every other draw (fibre lengths, surface draws) is identical to the earlier `_wexp` arms; record a hash of the fibre properties. Arms `delsys5r_wexp_muscle_lam{0.60,…,0.90}` plus `…_lam1.0` as the published reference: same spike file (25 % MVC, `_wexp`), skin/fat/conductivity/density/fs/Delsys geometry as `delsys5r_wexp_lam0.7_cvrtforce`, dispersion = `muscle_candidate` fields. Eight arms, sequential, ~40 min each.

λ choice: the grid value whose MDF, averaged over the corner–centre and adjacent pin pairs, is closest to Botelho's ~165 Hz [botelho2019]; also the linearly interpolated λ. RMS reported against the Botelho IQR 0.19–0.26 mV. The `iap_lambda__per_mm` docstring and the PR doc then cite the new fit; the default stays 1.

## 7. Left to the builder

Test names and helper structure; exact error wording (must name the field or the missing call); where the `Muscle` arrays are split (in `SurfaceEMG`, passed down).
