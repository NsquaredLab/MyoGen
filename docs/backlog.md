# MyoGen backlog (running task list)

Started 2026-11-03 (Devon). Future work we must not lose sight of. Newest decisions in the linked docs; this list says what is open and why. Move items to "Done" with the commit or issue that closed them.

## Priority: future work

### 1. Named simulation profiles

**Why.** MyoGen has many interconnected settings; signal-quality ones are easy to miss because they are off by default and buried (Gfluctdv noise was silently off; the human force-frequency curve, surface fibre dispersion, Muscle-fibre surface CVs and λ are all opt-in). A user who does not know they exist gets a worse signal without warning.

**Idea (to be designed).** A small set of named profiles, each filling every interconnected setting with consistent defaults across the neuron pool, `Muscle`, `ForceModel`, surface EMG and intramuscular EMG. Explicit arguments always override the profile.
- `"physiological"`: every validated mechanism on, with literature values (Gfluctdv on and calibrated, human force-frequency curve, fibre typing by unit, surface EMG reading `Muscle` fibres with the FDI λ ≈ 0.72 fit, two-layer end-plates with nerve delays, tendon-end stagger, …). λ is fitted per muscle and per fibre source (0.7 with surface CVs, ≈ 0.72 with `Muscle` fibres), so the profile must carry it with the fibre source.
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
- **6 m/s fibre CV bound**: ~2 % of fibres sit there at 4.32 ± 0.71; Del Vecchio's max is 6.21. Kept for now (Devon 2026-11-02).
- **`--max-recruitment` alignment** with the human force-frequency curve at the next recalibration.
- **Library findings out of scope so far**: NERLab rheobase spans 3.6× for a 48× threshold range; NERLab rests at ~0 mV but the yaml initialises −67 mV; inhibitory synapses at −75 mV absolute are far too strong for NERLab.
- **Defects found by the settings inventory** (2026-11-03, `docs/superpowers/specs/2026-11-03-settings-inventory.md` §5; not yet fixed, each needs a test first): passive mode string `"pas"` vs `"passive"`; GTO `"generic"` raises NameError; inhibitory-synapse fallback; optimiser default threshold mode raises (`deluca__slope` never passed); docstring/default mismatches (bone radius, `peak_hz`, axon units); FDI calibration file is gitignored.
- **Duplicated authorities** (same inventory): muscle conductivity sEMG 0.09/0.4 vs iEMG 0.063/0.33 S/m; fibre length 30 / 31.7 / 38.9 mm; fibre CV (sEMG 4.2, `Muscle` 4.32 ± 0.71, dispersion slope) — the last is now resolvable with `fiber_properties="muscle"`. iEMG output normalised to peak 1 (no physical amplitude); iEMG hard-codes λ; iEMG `execution=` still runs the pre-0.13.0 kernel. Inputs for item 1 and item 2.
- **Surface RMS below Botelho with `Muscle` fibres**: at the λ fit the corner–centre RMS is ~0.17 mV, under the IQR 0.19–0.26 mV (adjacent pins inside). Reported, not tuned. Candidate causes to check: within-unit CV spread 0.11 (is the `Muscle` CV spread physiological within a unit?), active-unit diameters.
- **Intramuscular EMG**: deferred (Devon 2026-11-02), kept open.
- **Future data**: possible HD-iEMG + HD-sEMG FDI dataset for a CV comparison paper (not available yet).

## Done

- 2026-11-03: surface EMG reads `Muscle` fibres (`121546f55`, `d9dd47e7b`, `130ee929b`, `d3db69520`, `1ba5b18bb`); λ refit on Botelho: ≈ 0.72 (see PR doc).
- 2026-11-02: fibre typing by motor unit (`a3dc1d7`…`c9f39c04d`); % MVC / CV helpers (`c1331ab76`); decisions on calibration, Gfluctdv, force-frequency, λ, CV clip (`5e904a67f`).
