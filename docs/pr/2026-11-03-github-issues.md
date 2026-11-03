# GitHub issues for the 2026-10-30 to 2026-11-03 changes (draft)

Status: Draft, not opened. Backlog item 2 (`docs/backlog.md`). Nothing is posted to GitHub until Devon approves the list and the texts.

Repository: `NsquaredLab/MyoGen` (no open issues; labels: `bug`, `enhancement`, `documentation`). All hashes are on `calib/fdi-dd-main`, which holds every change below. The GitHub copy of `fix/semg-amplitude-scale` (`6137959e7`) predates the 2026-10-30 identity rewrite, so its hashes differ; map them with `.git/identity-rewrite-20261030/commit-map.txt` or delete that branch once the PRs are open.

## Routing

Recommend two PRs, each closing its issues with `Fixes #n`:
- **PR A, surface and intramuscular physics fixes**: issues 1–4. The commits up to `028a14362` are a contiguous prefix of the branch (they were `fix/semg-amplitude-scale`); `42a56ca03` (issue 3) and `334eb667b` (issue 2) sit later and would be cherry-picked or the PR would include the later prefix.
- **PR B, models, calibration and tooling**: issues 5–10.

One PR closing all ten is simpler to produce and still traceable through the issues; two PRs are easier to review. Devon to choose.

Each issue body follows: **Problem** (what was wrong or missing, with evidence), **Change**, **Commits**, **Evidence** (tests, benchmarks), **Decisions** (where recorded), **Impact** (what changes for users; upgrade notes).

---

### 1. Surface EMG: amplitude, timing and geometry errors against Farina et al. 2004 — `bug`

**Problem.** Reported in an external tutorial notebook and confirmed on 0.13.0 (2026-10-30): the angular integral B(k_z) kept only its real part (`bipolar_longitudinal` channels ~10⁻⁷ of their value, rotated arrays distorted); MUAP templates labelled at 2× their grid rate (durations halved, spectra an octave high, CV estimates 2×); Rosenfalck IAP evaluated on `z/2` (2× too long); fibre semi-lengths L1/L2 swapped; each fibre circularly shifted onto its own peak (up to ~60 % bipolar MUAP error); arrays broadcast into the first array's shape; Fourier transforms missing `dz` and `Fs/N` (amplitude ∝ 1/CV); ad-hoc source `A = 96/10` instead of the core-conductor line current; a failing unit silently returned zeros.
**Change.** All corrected; a single fibre matches an exact homogeneous insulated-cylinder solution within 2 %.
**Commits.** `21a412f27`, `abeec9751`, `60df7c2e9` (breaking), `3f0905a31`; tests `4b71b4afe`, `2380a068f`, `8ddecc19e`.
**Evidence.** Layered conductor and finite fibre against independent solutions (bipolar chain < 1 %); old-vs-new comparison on the FDI benchmark (nexus `MyoGen_sEMG/Assets/semg_old_vs_new_2026-11-03/`, stage A → B: 28 mV / 455 Hz → 0.25 mV / 235 Hz).
**Decisions.** `docs/pr/2026-11-01-upstream-changes.md` §2.
**Impact.** sEMG outputs from 0.5.0–0.13.0 are invalid in amplitude, duration and spectrum; regenerate (CHANGELOG upgrade notes).

### 2. Surface EMG defaults: skin conductivity, FDI skin and fat, fibre diameter — `enhancement` (breaking)

**Problem.** Default skin σ 4.88·10⁻⁴ S/m (a measured composite value) raised amplitudes 3.5–6× in this cylinder; skin/fat thickness and the 50 µm fibre diameter had no FDI source.
**Change.** Skin σ 1 S/m, an effective value fitted by Roeleveld et al. 1997 (as used by Farina 2004); skin 1.2 mm (Li 2018), fat 1.3 mm (from Nishikawa 2024 total depth); `SurfaceEMG(fiber_diameter__um)` 57.9 µm, the RMS FDI diameter from Polgar et al. 1973.
**Commits.** `a8ff28728`, `975b91a50`, `8f8120557`, `8d3e81190`, `334eb667b`; Roeleveld benchmark `ba50b9af1`, `d9a2f9f63`, `cfb6da298`, `a6dbe79e3`, `3b061a80a`.
**Evidence.** `benchmarks/semg_roeleveld1997/` (no skin σ reproduces the measured fall-off; 1 S/m is closest); Botelho 2019 Delsys comparison (PR doc §6).
**Decisions.** PR doc §3 and §6 (skin σ kept 2026-11-01, Polgar 2026-11-02).
**Impact.** Default amplitudes change (thickness −40 %, diameter +34 %); pass the old values explicitly to keep them.

### 3. Intramuscular EMG: SFAP units, IAP length, wave polarity, dz — `bug`

**Problem.** Muscle conductivities 63 / 330 "S/m" (1000× tissue); IAP at half length with 8× current; wave towards the shorter fibre half subtracted; sum over z without dz (amplitude ∝ 1/resolution). The errors nearly cancelled at the default step near the wire.
**Change.** Corrected on CPU and GPU; single-fibre SFAP within 3 % of an independent SI line source at 0.1–1 mm.
**Commits.** `42a56ca03` (breaking).
**Evidence.** `tests/test_iemg_sfap_units.py`.
**Decisions.** PR doc §3 "iEMG SFAP units".
**Impact.** iEMG MUAPs typically 20–50× larger at a fine wire; anything tuned on old amplitudes, including absolute noise floors, must be redone. Point electrodes in an infinite medium still overestimate the near field.

### 4. Two-layer end-plate spreads and citation — `bug`, `documentation`

**Problem.** The end-plate model used the 1.5–2× wider spreads of the reference code, not the published equations; the model was cited as "Konstantin et al. 2020" (first author's given name).
**Change.** Published Akhmadeev et al. 2020 eq. 5–6 coefficients by default, `nmj_coefficients="reference_code"` to reproduce; nerve model in a shared module for sEMG and iEMG; citation corrected (`konstantin` threshold mode name kept).
**Commits.** `5ce917555`, `e805d299f`, `061ab2bde` (breaking), `2ea926dd7`, `028a14362`.
**Evidence.** Limb sweep: within-unit end-plate SD 1.68 → 1.07 mm.
**Impact.** iEMG outputs change through narrower end-plates.

### 5. Gfluctdv conductance noise was silent — `bug`

**Problem.** `AlphaMN.insert_Gfluctdv()` left the mechanism's GLOBAL multipliers at 0: no noise, and noise-scale searches (including main's own finetune example) were ineffective. Fixed on the fresh-trunk line in April 2026 but never reached `main`.
**Change.** Multipliers set to 1 on insertion.
**Commits.** `249f7091e`.
**Evidence.** `tests/test_gfluctdv_noise.py` (failed first with std 0).
**Impact.** Pools calling `insert_Gfluctdv()` now have noisy discharge; refit DD parameters fitted while it was silent. Follow-up (separate issue later): a non-zero default.

### 6. `Muscle` fibre conduction velocities and large fibre layouts — `bug`

**Problem.** CV–diameter slope used (m/s)/mm with diameters in mm: every fibre ~1.1 m/s, clipped to exactly 2.0 m/s (iEMG). Circular layouts beyond ~313 000 fibres filled only the diagonals and were capped at 399 499.
**Change.** CV from the Nandedkar & Stålberg 1983 line in µm (later replaced by issue 7's model); larger layouts on a scrambled Halton sequence with a warning; existing layouts unchanged.
**Commits.** `3c2f4c2db`.
**Evidence.** `tests/test_muscle_fibre_properties.py`.
**Impact.** iEMG fibre CVs change (2.0 → physiological).

### 7. `Muscle` fibre model: typed by motor unit, Polgar diameters, Del Vecchio CVs, within-unit spread — `enhancement` (breaking)

**Problem.** Fibre type drawn per fibre (every unit mixed types; size unrelated to threshold); diameters a generic log-normal; CV spread applied over fibres although Del Vecchio's 0.71 m/s is a spread of unit CVs; within-unit spread an unsourced residual.
**Change.** One type per unit in recruitment order up to 57.4 % type I fibres (Polgar 1973); one diameter ladder across the pool (type means 50.0 / 67.0 µm); within-unit diameter log-SD 0.08 / 0.13 (Martin et al. 1988); CV on one Nandedkar–Stålberg line with 4.32 ± 0.71 m/s over unit means (Del Vecchio 2018), or per-unit CVs (`motor_unit_conduction_velocities__m_per_s`, e.g. from recruitment force with 0.39 m/s residual scatter); warning when > 5 % of fibres hit the 6 m/s bound; `within_unit_fiber_diameter_log_sd=None` restores the pre-2026-11-03 fibres exactly.
**Commits.** `92ccd8718`, `77ac3ac9a`, `c6f095ca9`, `a3dc1d70f`, `69f123713`, `962449123`, `c9f39c04d`, `6a2719c4a`, `d82a03e5d`, `54458d14d`.
**Evidence.** `tests/test_fiber_types.py`, `tests/test_muscle_fibre_properties.py`, `tests/test_ramp_recruitment_force.py`; FDI checks in the design notes.
**Decisions.** `docs/superpowers/specs/2026-11-02-fdi-fibre-typing-design.md`, `2026-11-03-within-unit-fibre-spread-design.md`.
**Impact.** iEMG (and opt-in sEMG `Muscle` fibres) change; `_generate_fiber_properties()` needs `assign_mfs2mns()` first.

### 8. Surface fibre dispersion and `Muscle` fibres in surface EMG — `enhancement`

**Problem.** Every fibre of a surface MUAP shared one end-plate, one CV and one length; surface CVs were built separately from the fibres iEMG uses.
**Change.** Opt-in `SurfaceFiberDispersion`: per-fibre end-plates (two-layer model), axon-branch delays, CV linear in threshold or per unit, within-unit CV spread, d²-scaled source normalised over the pool, tendon-end band (independent or spatially correlated), `literature_candidate()` preset; `fiber_properties="muscle"` reads every fibre's CV and diameter from `Muscle`. Off by default and bit-identical when off.
**Commits.** `6a77faa93`, `b72afc7a6`, `4c80ca30e`, `4d57e08df`, `1d95f27df`, `0dd6fb363`, `33ad5c497`, `121546f55`, `d9dd47e7b`, `130ee929b`, `1ba5b18bb`; harnesses `b269c2401`, `7b125d6a7`, `7728a93d1`, `3def08f8d`, `9754b7b39`, `5609c78ab`, `2e3c1a18a`, `d982bc93f`, `16eee41dc`, `30d3b3843`, `60a6b1ef7`, `d3db69520`, `3265550d8`.
**Evidence.** `benchmarks/semg_dispersion/` (README findings); old-vs-new stages C–F.
**Decisions.** `docs/superpowers/specs/2026-11-03-surface-emg-muscle-fibres-design.md`; PR doc §3, §6.

### 9. Rosenfalck IAP length scale option and the FDI fit — `enhancement`

**Problem.** Simulated FDI MDF ~15–20 % above Botelho et al. 2019 at every placement with the published λ = 1 / mm.
**Change.** `SurfaceEMG(iap_lambda__per_mm=1.0)`; default unchanged. Documented FDI fit: λ ≈ 0.63 with `Muscle` fibres and per-unit CVs from recruitment force (cc / adj RMS 0.19 / 0.23 mV, mean MDF 165 Hz; 160 Hz with the CV scatter). No further fitting to Botelho (Devon 2026-11-03: one subject, no code, not reproducible closely; PSD shape differs).
**Commits.** `e4d1e8d7e`; docs `a30ff0a84`, `6c3cc9103`.
**Evidence.** λ sweeps (PR doc §6).
**Impact.** None unless set.

### 10. FDI descending-drive calibration tooling — `enhancement`

**Problem.** No calibration path for an FDI pool on `main`; the DD wiring gave large motor neurons ~2× less current per input, so the largest units recruited late; % MVC needed a force model.
**Change.** Multilevel DD optimiser (sequential or parallel Optuna) with equal fan-in and size-compensated weights, recruitment ceiling in % MVC, ranked ceiling, high-force rate checks, fitted weight exponent; `ForceModel(force_frequency_shape=...)` with a human-unit curve (opt-in); `recruitment_force_pct_mvc` and `cv_from_recruitment_force`; `fdi_spikes_force.py`.
**Commits.** `3f81754f4`, `1224eb2ea`, `f006a012b`, `15c8f62ef`, `39b1a1768`, `15d074533`, `c1331ab76`.
**Evidence.** Calibration results in PR doc §5 (`_human59_wexp`).
**Decisions.** PR doc §5–6. Open: whether the optimiser belongs in `myogen/` or stays a bench tool.

### 11. Literature register and inline citations — `documentation`

**Change.** `docs/literature.md` built by `scripts/build_literature.py` (82 sources, verification status, code links); sourced constants cite `[key p. N]` inline; `tests/test_literature_register.py` checks that every cited key is registered.
**Commits.** `893d15020`, `0f1eb542d`, `c4686fe82`, `5851b02e3`, and the `docs(literature)` commits.

---

## Follow-up issues (open later, not part of these PRs)

Named simulation profiles (backlog item 1); non-zero Gfluctdv default; human force-frequency curve as default; defects from the settings inventory (`docs/superpowers/specs/2026-11-03-settings-inventory.md` §5); duplicated authorities (conductivities, fibre length, CV); NERLab library findings (rheobase span, initial voltage, inhibitory reversal).
