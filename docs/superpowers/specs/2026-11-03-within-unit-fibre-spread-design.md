# Within-unit fibre spread in `Muscle` — design

Status: Approved in conversation 2026-11-03 (Devon: "Do the proposed fix you suggested"; Stålberg 1966 unavailable, proceed without it).

## 1. Why

Surface EMG with `Muscle` fibres (`fiber_properties="muscle"`) came out below Botelho's RMS range. An ablation at λ 0.7 (same fibres and spikes) traced the whole drop to the spread of fibre CV within each unit: CoV 0 / 0.05 / 0.11 gave corner–centre RMS 0.199 / 0.193 / 0.173 mV and mean MDF 171.9 / 170.0 / 163.1 Hz. Nothing in `Muscle` sets that spread directly:

- the within-unit diameter log-SD is the residual of a pool spread built from `type_fiber_diameter_log_sd` 0.2 (an assumption): 0.166 for every unit in the FDI default;
- 0.2 m/s of CV scatter is added (an assumption);
- all fibre CVs are rescaled to 4.32 ± 0.71 m/s, but 0.71 is Del Vecchio's spread of **motor-unit** CVs [delvecchio2018 p. 2], matched by Andreassen & Arendt-Nielsen's 3.7 ± 0.7 m/s over 33 human TA units [andreassen1987 p. 567]. Over fibres it leaves the unit means too close (0.32 m/s over the active units).

The only within-unit fibre-size data found: cat TA, fibre cross-section CoV within a unit 14–21 % in slow units and 10–34 % (mean ≈ 25 %) in fast units, against 46 % across units [martin1988 p. C46 Table 1]. Diameter CoV ≈ area CoV / 2: type S ≈ 0.08, fast ≈ 0.13.

## 2. Goals

- Within-unit fibre diameter spread set directly per fibre type from Martin et al. 1988.
- The 0.71 m/s CV spread applies to unit-mean CVs (as measured), not to fibres.
- Then refit λ on the Botelho benchmark with the new fibres.

## 3. Non-goals and anti-goals

- Type means (50.0 / 67.0 µm), type fraction, the ladder and its ordering, the 2–6 m/s clip: unchanged.
- Surface EMG and intramuscular EMG code: unchanged (they read `Muscle`'s fibres).
- The old behaviour stays reachable, bit-identical, for reproducing earlier runs.
- No fit of the within-unit spread to the EMG benchmark: it comes from the literature.

## 4. Decisions

Devon (2026-11-03): the proposed fix — within-unit diameter spread from Martin 1988 per type; 0.71 m/s over unit means; drop the extra 0.2 m/s scatter; refit λ.

My calls (rulings, reported to Devon):
1. **Switch.** New `Muscle(within_unit_fiber_diameter_log_sd=(0.08, 0.13))`, (type I, type II), default on. `None` restores the old model exactly (residual within-unit spread from `type_fiber_diameter_log_sd`, 0.2 m/s scatter, rescale over fibres). `type_fiber_diameter_log_sd` is used only when this is `None`. The scatter is dropped from the new default and kept only in the old path (answers the open question without a new parameter).
2. **Values.** 0.08 = 16.5 % / 2 (mean of the four type S units); 0.13 = 25.5 % / 2 (mean of the ten FR, FI, FF units), [martin1988 p. C46 Table 1]. Cat, single sections of 8–10 fibres per unit; labelled as such. Log-SD ≈ CoV at these sizes.
3. **CV map.** One affine map of the Nandedkar–Stålberg line [nandedkar1983 p. 163], `cv = a + b · k · d` (k = 0.05 m/s per µm), applied to every fibre, so within a unit CV follows diameter on the same line as between units. `b` is set so the SD of unit-mean CVs over units (each unit with fibres counted once) is `sd_fiber_conduction_velocity__m_per_s` (0.71); `a` so the mean over fibres is `mean_fiber_conduction_velocity__m_per_s` (4.32), keeping Devon's 2026-11-02 decision to anchor the mean over fibres. The SD weighting changes, the mean weighting does not.
4. **Type means stay exact.** With a per-type within-unit spread the log-normal mean factor differs by type, so the ladder solve includes each type's factor; the existing tests of the 50.0 / 67.0 µm type means must still pass.

## 5. Checks (reported, not fitted)

- Within-unit CV CoV per type in the FDI default (expected ≈ 0.05–0.08 for type I).
- Unit-mean CV by recruitment-threshold band against Del Vecchio: 3.89 ± 0.50 m/s at RT 0–30 % MVC, 5.12 ± 0.45 at 50–70 % [delvecchio2018 p. 2].
- Fraction of fibres at the 2 / 6 m/s clip.

## 6. Then

Rerun the λ sweep (`delsys5r_wexp_muscle_lam*`, kernel 200 mm, n_jobs ≤ 16, Idle priority, one core free), pick λ as before (MDF closest to ~165 Hz, interpolated), report RMS against Botelho's IQR. Update the `iap_lambda__per_mm` docstring, PR doc, CHANGELOG, backlog, register.

## 7. Check results (FDI benchmark muscle, 120 units, seed 42; 2026-11-03)

| | new default | old model (`None`) | reference |
|---|---|---|---|
| within-unit CV CoV, type I units (median) | 0.075 | 0.110 | ≈ 0.06 from [martin1988 p. C46 Table 1] via the line |
| within-unit CV CoV, type II units | 0.117 | 0.114 | ≈ 0.10 |
| unit-mean CV, each unit once | 3.59 ± 0.69 | 3.84 ± 0.47 | 3.7 ± 0.7 [andreassen1987 p. 567]; 4.32 ± 0.71 [delvecchio2018 p. 2] |
| units RT 0–30 % MVC (approx.) | 3.30 ± 0.40 | 3.64 ± 0.26 | 3.89 ± 0.50 [delvecchio2018 p. 2] |
| units RT 50–70 % MVC (approx.) | 5.08 ± 0.34 | 4.85 ± 0.27 | 5.12 ± 0.45 [delvecchio2018 p. 2] |
| fibres clipped at 6 m/s | 4.9 % | 2.1 % | — |

RT bands map thresholds linearly to 0–70 % MVC (approximate). Open (Devon): with the mean anchored over fibres, the many small low-threshold units come out ~0.6 m/s slower than Del Vecchio's RT 0–30 % group (and than Nishizono's FDI units, 4.2 ± 0.5 m/s, recruited 5–42 % MVC [nishizono1990 p. 77]). The λ refit waits for that decision.

## 8. Decision on unit-mean CVs (Devon, 2026-11-03) and results

Devon chose option 3 with option 2 as fallback:
- `Muscle(motor_unit_conduction_velocities__m_per_s=...)` sets each unit's mean fibre CV (e.g. from `cv_from_recruitment_force`, Del Vecchio band means 3.89 / 5.12 m/s [delvecchio2018 p. 2]); within a unit, CV follows diameter on the Nandedkar–Stålberg slope, 0.05 m/s per µm [nandedkar1983 p. 163] (my call: the slope of the sourced line, since nothing is fitted in this mode).
- Without it, 4.32 ± 0.71 m/s holds over unit means, each unit once (was: mean over fibres).

FDI benchmark muscle, per-unit CVs from the `_wexp` recruitment forces (`rt_force_cv.json`, 3.75–5.34 m/s):

| | per-unit CVs (option 3) | fallback (option 2) | reference |
|---|---|---|---|
| within-unit CV CoV, type I / II (median) | 0.046 / 0.087 | 0.062 / 0.084 | ≈ 0.06 / 0.10 [martin1988 p. C46 Table 1] |
| unit-mean CV, each unit once | 4.02 ± 0.38 | 4.29 ± 0.65 | 4.32 ± 0.71 [delvecchio2018 p. 2] |
| units RT 0–30 % MVC (approx.) | 3.85 ± 0.12 | 4.02 ± 0.40 | 3.89 ± 0.50 |
| units RT 50–70 % MVC (approx.) | 4.98 ± 0.18 | 5.62 ± 0.20 | 5.12 ± 0.45 |
| fibres at the 6 m/s bound | 1.0 % | 14.6 % | — |

Option 3 matches the band means but has little spread inside a band (the helper maps recruitment force to CV deterministically). The fallback puts 14.6 % of fibres at the 6 m/s bound in this pool, because Fuglevand-distributed pools are mostly small units and the SD over units then stretches the large ones.
