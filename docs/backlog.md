# MyoGen backlog (running task list)

Started 2026-11-03 (Devon). Future work we must not lose sight of. Newest decisions in the linked docs; this list says what is open and why. Move items to "Done" with the commit or issue that closed them.

## Priority: future work

### 1. Named simulation profiles

**Why.** MyoGen has many interconnected settings; signal-quality ones are easy to miss because they are off by default and buried (Gfluctdv noise was silently off; the human force-frequency curve, surface fibre dispersion, Muscle-fibre surface CVs and λ are all opt-in). A user who does not know they exist gets a worse signal without warning.

**Idea (to be designed).** A small set of named profiles, each filling every interconnected setting with consistent defaults across the neuron pool, `Muscle`, `ForceModel`, surface EMG and intramuscular EMG. Explicit arguments always override the profile.
- `"physiological"`: every validated mechanism on, with literature values (Gfluctdv on and calibrated, human force-frequency curve, fibre typing by unit, surface EMG reading `Muscle` fibres with the FDI λ ≈ 0.63 fit, two-layer end-plates with nerve delays, tendon-end stagger, …). λ is fitted per muscle and per fibre source (0.7 with surface CVs, ≈ 0.63 with `Muscle` fibres and per-unit CVs from recruitment force), so the profile must carry it with the fibre source.
- `"published"`: each component exactly as its source paper describes it (e.g. Fuglevand force-frequency, Rosenfalck λ = 1, no dispersion).
- `"manual"`: nothing filled in; the user sets everything.

**Open questions for the design.** Which profile is the default (a default change: needs Devon's yes). Per-muscle values (FDI vs VL calibrations) inside a profile. How the active profile and every setting it filled are recorded in outputs (manifest), so a reader can see what was on. Whether to warn when a quality-relevant setting is off under `"physiological"`.

**Input.** The settings inventory (overnight 2026-11-03, `docs/superpowers/specs/2026-11-03-settings-inventory.md`): every quality-relevant setting, where it lives, default, physiological value with source, interactions, off-by-default flag.

### 2. GitHub issues for the recent changes

**Why.** Trackable history for future sanity checks and backtracking: each key change since ~2026-10-30 gets an issue, and the fixes reach `main` through those issues.

**Plan (to be drafted, not opened yet).** One issue per major change, each listing its commits on `calib/fdi-dd-main` (and earlier branches), the evidence, and the decision record (PR doc / design notes). Candidates:
- Surface EMG corrections against Farina 2004 (B(k_z) complex part, MUAP time axis, Rosenfalck λ / `z/2`, fibre semi-lengths, fibre timing, mixed arrays, absolute amplitude, skin/fat defaults, failed units raise).
- Intramuscular SFAP units and polarity; end-plate spreads (Akhmadeev published coefficients).
- Gfluctdv noise silent; large circular fibre layouts; fibre CV unit bug.
- Surface fibre dispersion (end-plates, nerve delays, CV, d² source normalisation, tendon ends).
- FDI fibre diameters (Polgar), fibre typing by motor unit, fibre CV 4.32 ± 0.71 (Del Vecchio 2018).
- IAP λ option and the FDI fit to Botelho 2019.
- Descending-drive optimiser and FDI calibration (`_human59_wexp`), human force-frequency option, % MVC / CV helpers.
- Literature register and inline citation convention.
Then decide one PR or several, routed through the issues. Source of truth for the list: `docs/pr/2026-11-01-upstream-changes.md`, `CHANGELOG.md`.

## Other open items

- **Non-zero Gfluctdv default**: separate PR; generic current-injection calibration; recheck the primary source of the 15–22 % ISI CV target. Fits into profiles (item 1).
- **Human force-frequency curve as `ForceModel` default**: proposed for the next release (item 1).
- **`HillModel` type counts from `Muscle`** (fibre-typing follow-up).
- **Type I / IIa interleaving at low thresholds** (Garnett 1979), if a source gives the overlap width.
- **6 m/s fibre CV bound**: Del Vecchio's max is 6.21. Kept for now (Devon 2026-11-02). Since the 2026-11-03 change the `Muscle` fallback (4.32 ± 0.71 over unit means) puts ~15 % of FDI fibres there; per-unit CVs from recruitment force put 1 %. Revisit if the fallback is used for results.
- **`--max-recruitment` alignment** with the human force-frequency curve at the next recalibration.
- **Library findings out of scope so far**: NERLab rheobase spans 3.6× for a 48× threshold range; NERLab rests at ~0 mV but the yaml initialises −67 mV; inhibitory synapses at −75 mV absolute are far too strong for NERLab.
- **Defects found by the settings inventory** (2026-11-03, `docs/superpowers/specs/2026-11-03-settings-inventory.md` §5; not yet fixed, each needs a test first): passive mode string `"pas"` vs `"passive"`; GTO `"generic"` raises NameError; inhibitory-synapse fallback; optimiser default threshold mode raises (`deluca__slope` never passed); docstring/default mismatches (bone radius, `peak_hz`, axon units); FDI calibration file is gitignored.
- **Duplicated authorities** (same inventory): muscle conductivity sEMG 0.09/0.4 vs iEMG 0.063/0.33 S/m; fibre length 30 / 31.7 / 38.9 mm; fibre CV (sEMG 4.2, `Muscle` 4.32 ± 0.71, dispersion slope) — the last is now resolvable with `fiber_properties="muscle"`. iEMG output normalised to peak 1 (no physical amplitude); iEMG hard-codes λ; iEMG `execution=` still runs the pre-0.13.0 kernel. Inputs for item 1 and item 2.
- **Surface RMS below Botelho with `Muscle` fibres**: at the λ fit the corner–centre RMS is ~0.17 mV, under the IQR 0.19–0.26 mV (adjacent pins inside). Reported, not tuned. Cause found 2026-11-03 (ablation `delsys5r_wexp_muscle_lam0.7_wcov*`, same fibres and spikes): the within-unit fibre CV spread alone. At λ 0.7, `Muscle` fibres with within-unit CoV 0 / 0.05 / 0.11 (as generated) give cc RMS 0.199 / 0.193 / 0.173 mV, adj 0.236 / 0.230 / 0.215 mV, mean MDF 171.9 / 170.0 / 163.1 Hz; the surface-CV control (CoV 0.05) gives 0.191 / 0.227 mV, 171.9 Hz, so diameters and unit-mean CVs barely matter. The within-unit CoV is not set anywhere: it emerges from the per-type diameter log-SD 0.2 and 0.2 m/s scatter (both assumptions), and the per-fibre rescale to Del Vecchio's 0.71 m/s, which is a between-unit SD (realised between-unit SD of unit means 0.32 m/s, within-unit 0.41–0.45 m/s). No human within-unit measurement found; cat TA fibre CSA CoV 0.25 within units (Martin et al. 1988, abstract only) gives CV CoV ≈ 0.09 (0.07–0.12). λ and within-unit CoV both lower MDF, so fix the CoV from the literature before refitting λ. Proposed: make within-unit CV CoV and between-unit CV SD separate `Muscle` parameters (default change: needs Devon). **Done 2026-11-03** (Devon approved): within-unit spread from Martin et al. 1988 (full text read), unit-mean CVs from recruitment force; refit λ ≈ 0.63, RMS 0.19 / 0.23 mV, inside Botelho's IQR.
- **Per-unit CVs have little spread within a recruitment band** (0.12 m/s vs Del Vecchio's 0.50 at RT 0–30 %), because `cv_from_recruitment_force` is deterministic. **Done 2026-11-03** (Devon approved): residual scatter 0.39 m/s (`54458d14d`). At λ 0.63 it lowers mean MDF 165 → 160 Hz (one draw, seed 20261103); RMS unchanged. Not refitted (Devon 2026-11-03): mean MDF 160 Hz is inside Botelho's IQR, and their model (one subject, no code) cannot be reproduced closely enough to chase; the PSD shape also differs and is left as is. λ stays ≈ 0.63.
- **6 m/s bound with the `Muscle` fallback**: ~16 % of default FDI fibres clipped. Devon 2026-11-03: keep the model, warn above 5 % (`54458d14d`).
- **Review minors, `Muscle` CV work** (fresh review 2026-11-03, deferred): type-mean test tolerance 0.5 µm cannot catch a missing per-type factor (check the ladder analytically to 1e-6); confirm the `None`-path hash against `a30ff0a84` and note it in the test; validate `within_unit_fiber_diameter_log_sd` in `__init__` (now fails after `assign_mfs2mns`) and accept any 2-sequence; `cv_from_recruitment_force` should raise on NaN / negative `residual_sd__m_s` and say it advances the global RNG; no count of fibres clipped at 2 m/s; tests for NaN per-unit CVs and per-unit CVs with `None`.
- **Intramuscular EMG**: deferred (Devon 2026-11-02), kept open.
- **Future data**: possible HD-iEMG + HD-sEMG FDI dataset for a CV comparison paper (not available yet).

## Done

- 2026-11-03: surface EMG reads `Muscle` fibres (`121546f55`, `d9dd47e7b`, `130ee929b`, `d3db69520`, `1ba5b18bb`); λ refit on Botelho: ≈ 0.72 (see PR doc).
- 2026-11-03: within-unit fibre spread from Martin 1988 and per-unit `Muscle` CVs (`6a2719c4a`, `d82a03e5d`, `84870f42c`); λ refit ≈ 0.63 with RMS inside Botelho's range.
- 2026-11-02: fibre typing by motor unit (`a3dc1d7`…`c9f39c04d`); % MVC / CV helpers (`c1331ab76`); decisions on calibration, Gfluctdv, force-frequency, λ, CV clip (`5e904a67f`).
