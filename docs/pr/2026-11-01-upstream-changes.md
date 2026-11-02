# Upstream changes for MyoGen: sEMG amplitude, FDI calibration, ported fixes

Working list for the PR(s) into `github/main`. Started 2026-11-01. Keep it current;
tick an item only when it is committed, tested and its evidence is linked.

Base: `github/main` = `b231049c9` (v0.13.0, 2026-10-28). No newer upstream commits as of 2026-11-01.


Literature behind every default and calibration choice in this PR, with links to the code that uses it: [docs/literature.md](../literature.md).

## 1. Branches and push state

| Branch | Content | State |
|---|---|---|
| `fix/semg-amplitude-scale` | sEMG fixes, tests, harnesses (§2, §3) | local tip `e4d1e8d7e`. `github` holds an older history up to `6137959e7`; the reworded skin-σ commit (§3) rewrote it, so the next push needs `--force-with-lease` |
| `calib/fdi-dd-main` | `fix/semg-amplitude-scale` + optimiser port + ported fixes + calibration arms (§4, §5) | local only, rebased onto `e4d1e8d7e` |
| `bench/semg-roeleveld1997` | Roeleveld 1997 skin-σ benchmark | merged into `fix/semg-amplitude-scale` (identical files); branch kept for reference |
| `fix/semg-amplitude-scale` (local) | Delsys 5-pin sensor + `_fs__Hz` harness | committed `d982bc93f` |

No rebase was needed: every sEMG branch already sits on v0.13.0, so pushing is a fast-forward.

## 2. sEMG physics fixes (on `fix/semg-amplitude-scale`)

Bugs found against Farina et al. (2004), each with a test:

- [x] `21a412f27` complex B(k_z) (imag part was dropped: near-zero bipolar), physical IAP length scale (z/2 stretch), L1/L2 swap (tendons moved with the IZ), per-fibre recentering (up to 40 % bipolar distortion), array broadcasting.
- [x] `abeec9751` continuous-transform scaling: amplitude no longer scales with 1/v.
- [x] `60df7c2e9` **breaking**: physical source current σᵢ·πa²·∂²Vm/∂z² replaces the ad-hoc A = 96/10.
- [x] `3f0905a31` raise when a motor unit fails instead of returning zeros (silent zero MUAPs before).
- [x] Tests: `4b71b4afe`, `2380a068f` (equal-conductivity layers reduce to homogeneous), `8ddecc19e` (layered conductor and finite fibre vs independent solutions; bipolar chain < 1 %).

## 3. sEMG defaults and features (on `fix/semg-amplitude-scale`)

- [x] `a8ff28728` + `975b91a50` **breaking**: skin σ default 1 S/m. Provenance corrected in `975b91a50`: an effective value fitted by Roeleveld et al. 1997 (biceps fall-off), not a gel/hydrated-skin measurement; the `a8ff28728` message was reworded 2026-11-01 to the same provenance (Farina 2004 from Roeleveld 1997, effective value); trees unchanged.
- [x] `8f8120557` **breaking**: FDI literature skin 1.2 mm (Li 2018) and fat 1.3 mm defaults.
- [x] `8d3e81190` fibre-length defaults documented as FDI values.
- [x] Dispersion features (all optional, off unless stated): per-fibre velocity on a shared grid (`6a77faa93`), per-fibre end-plates / nerve delays / velocities (`b72afc7a6`), CV linear in threshold + CV–diameter line (`4c80ca30e`), two-layer end-plate coefficients (`e805d299f`, default in `061ab2bde` **breaking**), tendon-end band and correlated ends (`4d57e08df`, `1d95f27df`), `literature_candidate` preset (`0dd6fb363`), iEMG `nmj_coefficients` (`2ea926dd7`), shared nerve-model module (`5ce917555`).
- [x] Harnesses and changelog: `b269c2401`, `7b125d6a7`, `7728a93d1`, `3def08f8d`, `9754b7b39`, `5609c78ab`, `2e3c1a18a`.
- [x] Delsys 5-pin sensor and `_fs__Hz` harness: `d982bc93f`.
- [x] `e4d1e8d7e` optional Rosenfalck IAP length scale `SurfaceEMG(iap_lambda__per_mm=1.0)`; default unchanged (see §6). Tests: exact cylinder at λ 0.7 / 0.5, passed to every fibre on both routes, invalid values and `use_unified=True` refused. CHANGELOG entry included.
- [x] Roeleveld 1997 skin-σ benchmark (`benchmarks/semg_roeleveld1997/`, bench only) included in this PR, Devon 2026-11-01: it is the evidence for the 1 S/m default. Its 5 commits sit on `fix/semg-amplitude-scale` after the IAP λ commit.
- [x] `334eb667b` **breaking** (on `calib/fdi-dd-main`): FDI fibre diameters from Polgar et al. 1973 (Devon 2026-11-02). `SurfaceEMG(fiber_diameter__um)` 50 → 57.9 µm (RMS; source ∝ d²; default sEMG amplitude ×1.34); `Muscle` log-normal mean 57.2 µm (was median 50; realised mean 55.9 µm after the unchanged 20–80 µm clip, which now holds ~10 % of fibres at 80 µm), mean fibre CV ~3.45 → ~3.7 m/s. The Botelho/Delsys arms already used 57.9 µm, so their numbers do not move.
- [x] (on `calib/fdi-dd-main`, Devon 2026-11-02: "go ahead with the normalisation and Polgar mixture") `Muscle` diameters: Polgar two-type mixture (57.4 % type I at 50.0 µm, type II at 67.0 µm, log-SD 0.2 per type truncated at 3 log-SD (Devon 2026-11-02), assumptions), 20–80 µm clip removed; mean 57.2 µm; iEMG only. Fibre CV (Devon 2026-11-02, option 1 after "anchor the CV to the mean"): ordered by diameter through the Nandedkar line plus 0.2 m/s scatter, rescaled to new `mean_fiber_conduction_velocity__m_per_s` 3.89 / `sd_fiber_conduction_velocity__m_per_s` 0.50 (Del Vecchio 2018 HD-sEMG, TA units recruited 0–30 % MVC); effective slope ~0.035 m/s/µm. `mean_conduction_velocity__m_per_s` 4.2 (Nishizono, FDI) still sets sEMG. Casolo et al. 2023 line (2.9 + 0.021·d) tried and not adopted. Dispersion `diameter_scales_source`: d² normalised over every fibre of the pool (was relative to the diameter at the mean CV, 65 µm, which put the effective diameter off `fiber_diameter__um`); report `diameter_mean__um` is now the effective diameter, `line_diameter_mean__um` the line value. Divides the earlier Delsys sweep RMS by C = Σd²/(N·d_mean²): 1.55 for the threshold-slope arms, 1.17 for the recruitment-force-CV arms; MDF unchanged. Corrected λ 0.7 + recruitment-force CV: 0.192 / 0.227 mV (Botelho 0.21, IQR 0.19–0.26). Nandedkar 1983 and Blijham 2006 read: the line is a construct (biceps ranges pinned to 25/55/85 µm), Blijham's measured fit gives the same relative spread within 2 %.
- [x] `028a14362` citation fix: "Konstantin et al. 2020" is Akhmadeev, Yu, Le Carpentier, Aoustin & Farina (2020, IEEE TBME 67:2005); docstrings, comments, example and CHANGELOG corrected, `konstantin` threshold mode name kept. On `fix/semg-amplitude-scale`; `calib/fdi-dd-main` rebased onto it 2026-11-01 (clean).

### iEMG SFAP units (on `calib/fdi-dd-main`, 2026-11-02, Devon: "do both")

- [x] `42a56ca03` **breaking**: four errors in `bioelectric.py` / `MotorUnitSim.calc_sfaps`. (1) Conductivities 63 / 330 "S/m" (1000× tissue) → 0.063 / 0.33 S/m. (2) IAP at `-2·arg` with `-4·V'`: half-length IAP, 8× current → λ = 1 mm as in sEMG. (3) Wave towards the shorter half subtracted (inverted) → added. (4) Sum over z without dz (amplitude ∝ 1/dz) → ×dz on CPU and GPU. Tests `tests/test_iemg_sfap_units.py`: production SFAP vs independent SI line source within 3 % at 0.1 / 0.5 / 1 mm (before: 2.2× / 4× / 8× low), mirror symmetry, reflection with unequal halves, dz independence, GPU = CPU. Internal commit `3b4ee8a0d` (2026-03-29) fixed (1)–(2) on fresh-trunk lines but never reached main; (3) is in every line, including fresh-trunk.
- Effect, FDI cylinder (r 6.91 mm, 267 fibres/mm², bipolar fine wire 2 mm IED at x 2 mm, 8 mm from mid-length, 15 MUs): MUAP p-p 0.03–1.8 mV → 0.9–37 mV; MNF 300–1150 Hz (erratic) → 290–490 Hz (falls with unit size); duration 6–9 → 5–8 ms. Point electrodes in an infinite homogeneous medium overestimate the near field (no electrode area, no near-field correction such as Albers), so the new amplitudes are physically consistent but high against needle recordings; the old ones looked realistic only because the bugs cancelled. Script: scratchpad `state_1102/iemg_muaps.py`.

## 4. Ported from the fresh-trunk line (fixes that never reached main)

- [x] **Gfluctdv silent on main** — `ee4aab7e4`. `Gfluctdv.mod` GLOBAL `multex`/`multin` default to 0 and `AlphaMN.insert_Gfluctdv()` never set them, so the mechanism added zero noise. Fresh-trunk fixed this in `bef5cbe79` (2026-04-17). Test: `tests/test_gfluctdv_noise.py` (failed first with std 0). Also affects main's own `examples/02_finetune/01_optimize_dd_for_target_firing_rate.py`, whose noise search was ineffective.
- [x] **Fibre CV slope bug on main** (`muscle.py`, `_generate_fiber_properties`): `k = 5.5 (m/s)/mm` with diameters in mm gave 0.8 + 5.5·0.05 ≈ 1.1 m/s, then the clip raised every fibre to 2.0 m/s (17 593 of 17 593 fibres at exactly 2.000 m/s, r 4 mm muscle). Used by main's iEMG (`intramuscular_emg.py:256`); surface EMG uses the mean CV and is unaffected. **Fix uses CV = 0.95 + 0.05·d[µm] m/s** (Nandedkar & Stålberg 1983; 3.7 m/s at 55 µm), the line already used by `SurfaceFiberDispersion`; both now read the shared constants `CV_INTERCEPT__M_S`, `CV_PER_DIAMETER__M_S_PER_MM` in `muscle.py`. Not fresh-trunk's `be9cdfe35` numbers (d/13 + 1.5, 5.3 m/s at 50 µm, cited to the same paper but matching neither it nor Blijham et al.'s invasive fit 0.043·d + 0.83). Noise (SD 0.2 m/s) and clip (2–6 m/s) unchanged. Mean fibre CV on main moves from 2.0 to ~3.45 m/s. Test: `tests/test_muscle_fibre_properties.py`.
- [x] **Fibre table cap / shape on main** (`muscle.py`, `generate_muscle_fiber_centers`): `voronoi_pi1e5.csv` covers the square [-1.25, 1.25]²; only ~313 k points lie in its inscribed disc. Beyond that the outer ring held only the corners (outer-ring sector count 0 on the diagonals' complement), and beyond 399 499 the fibre count was silently capped. Fix: unchanged for counts the disc covers (FDI ≈ 54 k, all existing callers below ~313 k get identical layouts); larger counts use a scrambled Halton sequence in the disc (seeded from MyoGen's RNG) with a warning. Chosen over fresh-trunk `a82923b2b` (adds a 23 MB 1 M-point CSV and still caps beyond it). Tests: `tests/test_muscle_fibre_properties.py` (count not capped; outer ring covers the full circle).
- Already resolved on main independently (no action): `correlate`→`convolve` in `SurfaceEMG` (fresh-trunk `3039771ba`); RNG propagation after `set_random_seed` (main uses `get_random_generator()`).
- Out of scope here: fresh-trunk's iEMG rewrite (323 commits on `intramuscular/*` and `electrodes.py`, 68 fix-like). That is its own upstreaming project; nothing in this PR depends on it.
- Audit method (2026-11-01): `git diff $(git merge-base github/main fresh-trunk) fresh-trunk` restricted to source files that exist on main (12 files), plus fix-like fresh-trunk commits on those files, each checked against main's current code. The neuron path (`simulator/neuron`, `nmodl_files`, `utils/helper.py`, `utils/nmodl.py`) has no other code differences besides `nrnmech.dll` rebuilds and the unused beta/effort drive features (ARCH-25/26/30).

## 5. FDI descending-drive calibration tooling (on `calib/fdi-dd-main`)

- [x] `5460831fe` multilevel DD optimiser + `ramp_analysis`, `fr_optimization_targets`, `parallel_budget`, `fdi_spikes_force.py` (force mapping with the fixed Fuglevand ForceModel, upstream `e794774db`). Thresholds from `RecruitmentThresholds` (muscle-profile library is not on main).
- [x] Results: 120-MU FDI calibrations (Fuglevand 50× thresholds, De Luca 1982 targets) with Gfluctdv enabled. Recruitment ending at 70 % excitation: error 1.317, ISI CV 13.9 %, at 25 %: 90/88 active, 20.3/20.8 Hz. Ending at 52 %: error 0.771, ISI CV 14.8 %, 96/97 active, 22.0/21.0 Hz; its best `gfluctdv_scale` 0.243 is at the top of the search range (0.25), so widen the range before treating ISI CV as converged. Noise-off run (52 %) for reference: ISI CV 8 %.
- [x] Force mapping with the fixed `ForceModel`, cross-checked with an independent Fuglevand implementation (within 0.1 % MVC): 25 % MVC at excitation 0.227 (70 % calibration) and 0.208 (52 %); 25 % excitation = 28.6 % and 30.8 % MVC.
- [x] Recruitment thresholds on the MVC ramp (52 % calibration; force at each unit's first spike, 100 ms smoothing): 68 % of units recruited below 10 % MVC, 77 % below 25 %, 88 % below 50 %; median 4.3 % MVC. Requiring sustained firing (3 ISIs < 200 ms) gives 63 / 75 / 88 %. The 70 % calibration: 62 / 77 / 88 %. Low end matches Milner-Brown et al. 1973 (FDI: over half of 145 units recruited below 200 g with MVC well above 2 kg, i.e. below ~10 % MVC; their sampling favours low-threshold units). High end: the 10 largest units (designed thresholds 0.38–0.52 excitation, 50–68 % MVC) start late, at 0.62–1.0 excitation, up to ~95 % MVC; the designed last threshold (0.52 excitation) is 68 % MVC. These units are silent at 25 % MVC either way, so the sEMG comparisons are unaffected. Literature upper limit for FDI recruitment not yet checked against a source.

- [x] Root-level launch scripts `run_fdi_dd.sh` / `run_fdi_dd_52.sh` removed (they hard-coded this machine's venv and NEURON paths). The calibration commands, run from `examples/finetune/`:
  ```
  python 01_optimize_dd_multilevel.py --target-muscle fdi --n-mu 120 --n-trials 100 --parallel --n-workers 6 --fresh \
    --threshold-mode fuglevand --recruitment-range 50 --max-recruitment 0.52 --target-model de_luca \
    --de-luca-params '{"fr_onset_hz": 8.9, "fr_peak_early_hz": 69.0, "fr_peak_late_hz": 44.0, "steepness": 1.55}' \
    --results-tag fuglevand50_rec52_deluca82_main
  ```
  (`--max-recruitment 0.7`, tag `fuglevand50_deluca82_main` for the 70 % run.)
- [x] `GFLUCTDV_RANGE` widened to (0.05, 0.5). The `_gf05` rerun was stopped before finishing (Devon 2026-11-01): it used the old wiring, which the next item replaces.
- [x] Late recruitment of the largest units diagnosed (2026-11-01). Rheobase is linear in the designed threshold (5.2–18.6 nA, 3.6× for a 48× threshold range) and the DD pool is linear, but every MN got ~51 inputs (binomial, 38–66) of one weight, and large dendrites shunt more: one input delivered ~1.9× less current to a large soma than to a small one (soma clamped at rest), and the current per 100 Hz of drive falls ~45 % from low to full drive. ±13 % input-count scatter scrambles the top units, whose thresholds are ~3 % apart.
- [x] Fix (Devon 2026-11-01, optimiser only): equal fan-in, input weights scaled by each MN's dendrite-to-soma transfer (`h.Impedance`), and a ceiling term on the excitation at which the last unit is recruited (separate ramp to full excitation). Verified with the main52 DD parameters: per-input soma current top/low 1.03 at 100 Hz, 1.17 at 400 Hz, 1.24 at 874 Hz; recruitment order matches the design exactly (rank correlation 1.0). All earlier calibrations (main52, main70, gf05) are not comparable and are refused by `run_dd_ramp`.
- [x] Recalibrated with the new wiring and the ceiling in excitation (`--max-recruitment 0.41`, `_wired`, 102 trials): best error 1.555 (rates 0.97, counts 0.46, ceiling 0.13), ISI CV 16 %, Gfluctdv 0.21, last unit at excitation 0.39. But the new wiring changed the force curve: 0.41 is now ~70 % MVC, and `fdi_spikes_force.py` puts the last unit at 76 % MVC (excitation 0.435; 95 % of units by 51 % MVC). An excitation target has to be re-derived whenever the force curve moves.
- [x] Ceiling scored in % MVC instead (Devon 2026-11-02, option A): `--last-recruit-mvc PCT` runs an MVC ramp with `ForceModel` and penalises the force at the last recruitment; `--max-recruitment` now only sets the rate-coding targets. `fdi_spikes_force.py` reuses the optimiser's MVC ramp.
- [ ] Recalibrate with `--last-recruit-mvc 55` (De Luca 1982: no FDI recruitment above 52 % MVC; Škarabot 2023: up to 59 %) and confirm with `fdi_spikes_force.py`.
  - Ran (`_mvc55`, 102 trials): best unchanged from `_wired` (same startup trial), last unit at 66.3 % MVC, error 2.04 (ceiling 0.62). Seed spread of those parameters 69–76 % MVC (median 70). Trials at 55–59 % MVC need Gfluctdv 0.32–0.48 (ISI CV 14–20 %) and recruit too many units early (120/110 at the 30 % checkpoint, 74/55 at 5 %). Open: which constraint wins, and whether the ceiling ramp should reuse the checkpoint ramp's wiring (it is redrawn now, so `fdi_spikes_force.py` cannot reproduce the optimiser's value). The checkpoints are excitation levels, not force: 20/25/30 % are ~27/38/50 % MVC with these parameters.
  - Why ~70 % (2026-11-02): not the network (last unit at the designed excitation in every seed). Fuglevand's force-frequency gain saturates too early: refit to human single-unit stimulation data (Macefield 1996, Fuglevand 1999, McNulty 2000; S(r) = 1 − exp(−a r^c), a 0.93, c 1.29; rms 0.05 vs 0.13; predicts twitch:tetanus 0.23, measured 0.22–0.36, Fuglevand 0.11). Same spikes: last unit 63.3 % MVC (6 seeds, 63.0–67.7) vs 70.0 with Fuglevand; 2–3 units above 59 %. 52 % (De Luca 1982) and 59 % (Škarabot 2023) are sample maxima; 59 % is the stronger bound. 25 % MVC drive unchanged (excitation 0.186). Model rates ~20 % below De Luca 1982 at 40/80 % MVC (20/32 vs 25/41 Hz).
  - Proposed: opt-in force-frequency parameters in ForceModel (default Fuglevand), score the ceiling on the fitted curve with target 59 %, add De Luca 40/80 % rate checks. No recalibration run.
  - Approved and done (`39b1a1768`): `ForceModel(force_frequency_shape=...)` (default Fuglevand), optimiser `--force-frequency human`, `--ceiling-rank 1` (unit with one above it), `--high-force-rates` (De Luca 1982 Table 1).
- [x] Recalibrated with `--last-recruit-mvc 59 --ceiling-rank 1 --force-frequency human --high-force-rates` (`_human59`, 102 trials, 76 min): best error 1.79, dd_neurons 168, conn_prob 0.21, drive 967 Hz, Gfluctdv 0.25, ISI CV 16.9 %. Last unit at 59.0 % MVC, second-last 58.0 %. Checkpoints (sim/target units, Hz): 5 % 66/55 9.6/11.2; 10 % 82/76 12.2/13.9; 20 % 102/97 18.3/18.9; 25 % 111/104 20.5/21.1; 30 % 117/110 23.2/23.2. `fdi_spikes_force.py` (redrawn wiring): last unit 62.5 %, second-last 56.0 %; 25 % MVC at excitation 0.169 (104 units, 15.4 Hz). Full suite: 429 passed, 2 skipped.
  - High-force rates stay ~19 % low: 20.6/33.5 Hz at 40/80 % MVC vs 25.3/41.4. No trial with a good fit got closer (best 40 % rate of any trial 23.5 Hz; 80 % rates near 40 Hz only in trials that over-recruit at the checkpoints, error > 11). The four network parameters cannot raise high-force rates without breaking the low-force targets; the cause is upstream (motor-neuron rate gain or the averaging: model mean over all active units vs De Luca's ≤8 tracked units per contraction).
  - Investigated 2026-11-02 (nexus note, "Are spinal mechanisms behind the low high-force rates?"):
    - Not the averaging: settled low-threshold units are also at 21/33 Hz.
    - Rates are near-uniform across units (SD ~1 Hz at 80 % MVC vs 9.6), with no onion skin.
    - The NERLab PIC is present but inert: paired-unit ΔF 0.2 Hz vs ~4.8 pps reported for FDI.
    - Stronger or effort-dependent PICs (gamma 0.5, 1.0, 0.2→1.0) and push-pull or balanced inhibition leave the 80 % MVC rate at 29–34 Hz. More PIC inverts the onion skin.
    - With uniform rates, the twitch law and force-frequency curve put 80 % MVC at ~32 Hz; 41 Hz is unreachable without an onion skin.
    - Probes (same day). DD weights `w·(k_max/k_i)^p` (calibration uses p = 1):
      - p = 0.5: 22.7/35.0 Hz, SD 5.0, small−large +13.7 Hz at 80 % MVC, last unit 75 %.
      - p = 0: 22.5/36.2 Hz, SD 7.0, +18.1 Hz, last unit ~101 %.
      - The input spread across sizes is the lever for the onion skin and the high-force rates, but it raises the ceiling. Recruitment ceiling and onion skin trade off through the same weighting; only a recalibration with p free can test both together.
    - Powers2017 cell model as configured: rates 3–14 Hz (f-I 0.02–0.4 Hz/nA), largest cell blocks at 4× rheobase. Not usable for FDI without re-parameterisation.
    - FDI ΔF benchmark: 4.80 [4.31, 5.30] pps (Beauchamp et al. 2023, full-text table); model 0.2.
- [x] Recalibrated with the DD weight exponent p free (`15d074533`, `--fit-weight-exponent`, p in [0, 1]; otherwise the `_human59` setup; `_human59_wexp`, 150 trials, 123 min). Best trial 85, error 1.25 (`_human59`: 1.79):
  - dd_neurons 147, conn_prob 0.36, drive 760 Hz, Gfluctdv 0.36, **p 0.67**, ISI CV 20.3 %.
  - Ceiling: last unit 62.9 % MVC, ranked unit 62.8 % (target 59). `fdi_spikes_force.py` (wiring redrawn): 66.9 / 63.5 %.
  - High-force rates 22.8 / 36.7 Hz at 40 / 80 % MVC (`_human59` 20.6 / 33.5; De Luca 25.3 / 41.4): about 45 % of the gap closed. Force script: 21.9 / 36.0.
  - Onion skin at 80 % MVC (force script, active units by threshold quartile, low to high): 37.3 / 37.2 / 37.2 / 32.5 Hz (`_human59`: 32.3 / 32.5 / 33.7 / 33.7). At MVC 55 → 51 Hz (`_human59` 50 → 54).
  - Checkpoints (sim/target units; Hz): 5 % 61/55, 9.3/11.2; 10 % 81/76, 12.9/13.9; 20 % 99/97, 18.9/18.9; 25 % 107/104, 21.1/21.1; 30 % 111/110, 23.8/23.2. 25 % MVC at excitation 0.188 (103 units, 16.7 Hz).
  - Trade-off across the 54 trials with error < 3 (median ranked ceiling; 40 / 80 % rates): p < 0.3 75.9 %, 25.0 / 41.1 Hz; 0.3–0.5 68.5 %, 24.2 / 39.5; 0.5–0.7 63.3 %, 23.1 / 36.7; 0.7–0.9 60.1 %, 22.3 / 36.0; ≥ 0.9 61.6 %, 20.5 / 31.5. No trial meets both 59 % and De Luca's rates; those with ceiling ≤ 61 % and 80 % rates above 41 Hz miss the checkpoint targets (error 3.5–7).
  - p is a modelling assumption (how much descending input favours small motor neurons); no direct literature value. Decision for Devon: `_human59_wexp` (better rates and overall fit, ceiling 4–8 pp above 59 %) or `_human59` (ceiling met, rates ~19 % low, no onion skin).
- [ ] Out of scope, library: the NERLab pool's rheobase spans only 3.6× for a 48× threshold range (yaml interpolation curves).
- [ ] Out of scope, library: NERLab motor neurons rest at ~0 mV but `alpha_mn_default.yaml` initialises them at −67 mV (start-up transient).
- [ ] Out of scope, library: inhibitory synapses use −75 mV (absolute frame), i.e. 75 mV below the NERLab rest; any inhibition onto NERLab motor neurons is far too strong.

### ForceModel twitch parameters against FDI data

Milner-Brown, Stein & Yemm 1973, *J Physiol* 230:359–370 (doi 10.1113/jphysiol.1973.sp010192; methods in 228:285–306, doi 10.1113/jphysiol.1973.sp010087), 145 FDI units from 3 subjects, against `ForceModel` defaults on the 120-MU pool above (Fuglevand 1993):

| | Milner-Brown 1973 (FDI) | MyoGen `ForceModel` |
|---|---|---|
| Twitch shape | near-critically damped second order (damping ratio 1.0–2.0, mean 1.2) | P·t/T·e^(1−t/T), the critically damped impulse response |
| Twitch force distribution | exponential, ~0.1–10 g (~100-fold), halves every ~1.2 g | exponential, 47-fold (`twitch_force_range=None` → threshold ratio) |
| Twitch force vs threshold | log-log slope 0.92–1.04, r > 0.8 | slope 0.99 |
| Recruitment-threshold range | ~100-fold (one subject) | 48-fold (Fuglevand 50×, chosen by the De Luca 1982 rate fit) |
| Contraction time | 30–100 ms; means 51.5–59.1 ms (SD 11–17); > 80 % under 70 ms | 30–89 ms; mean 54.4 ms (SD 17.1); 78 % under 70 ms |
| Contraction time vs threshold | larger units faster, r ≈ −0.6 (weaker or n.s. in 2 of 3 subjects) | deterministic, r = −0.99 |

The generic Fuglevand parameters match FDI closely (Fuglevand 1993 drew on these data), so % MVC from `ForceModel` is supported for FDI. Open: FDI may span ~100-fold in threshold and twitch force rather than 50-fold (would need recalibration); the model has no contraction-time scatter around the threshold trend (small effect on pool force).
- [ ] Decide whether the optimiser belongs in `myogen/` + `examples/finetune/` upstream or stays a bench tool. Main already has `examples/02_finetune/01_optimize_dd_for_target_firing_rate.py`; reconcile or replace.
- [x] `--extra-trap NAME:LEVEL` option in `fdi_spikes_force.py`: included in `5460831fe`.

## 6. Recommendations (need Devon's decision before the PR)

- [ ] **Non-zero Gfluctdv default.** Main has no default: noise is on only if a caller runs `insert_Gfluctdv()` and scales it. Recommend an `AlphaMN__Pool` parameter (e.g. `gfluctdv_scale`) with a non-zero default, so pools have literature ISI variability (15–22 %, LIT-17) out of the box. Reference values: VL 0.12 → ~20 % ISI CV at 30 % MVC (fresh-trunk VAL-42); FDI value from the running calibration. Document the value as calibrated, per muscle, with the target it was fitted to.
- [x] **Skin σ default (1 S/m).** Decided 2026-11-01 (Devon): keep, documented as an effective cylinder value (docstring, CHANGELOG, reworded commit). Evidence so far: Roeleveld fall-off tests (rerun on main, unchanged) show no skin value reproduces the measured fall-off; 1 S/m is closest. Botelho 2019 Delsys comparison at a force-verified 26 % MVC: 1 S/m gives 0.149 (corner−centre) / 0.181 mV (adjacent), 0.15 S/m gives 0.541 / 0.617 mV, vs measured 0.21 mV. Retest on main-calibrated trains (52 % ramp, 25 % MVC, fibre hash `0799727c71`, 20 placements): 1 S/m gives 0.136 / 0.175 mV (MDF 202 / 198 Hz), 0.15 S/m gives 0.486 / 0.653 mV (MDF 240 / 232 Hz), vs measured 0.21 mV (IQR 0.19–0.26) and MDF ~165 Hz (IQR 152–185). 1 S/m is ~20 % low in amplitude and ~20 % high in MDF; see the IAP item below. The comparison target is Botelho's *measurement*. Their parameters (skin 4.88e-4 S/m, fibres 24.8–40.6 µm, muscle 0.40/0.09 S/m) belong to an anatomical FE model (finite, curved, bone, tendons) and do not transfer to the infinite layered cylinder: the same tissue σ produces different potentials in the two conductors. In the cylinder, skin σ is an effective parameter (as Roeleveld 1997 found for their own cylinder), so its value should be justified by matching measurements, not by citing measured tissue σ or another model's inputs.
- [x] **IAP length (Rosenfalck λ) or mean CV: MDF gap.** Decided 2026-11-01 (Devon): keep λ = 1 / mm as the default; expose λ as an optional parameter (`e4d1e8d7e`) documented with the FDI fit. Revisit the default only with a second independent dataset (RMS and MDF, known electrode and force) and a source for human IAP duration. Simulated FDI MDF is ~15–20 % above Botelho's measurement at every placement. The muscle layer does low-pass with depth (per-MU MDF 213 Hz at 0–2 mm, ~145 Hz beyond 4 mm), but five superficial MUs carry 39 % of the energy. Bipolar probes at skin 1 S/m (temporary switch, not on this branch): λ 0.7 gives MDF −9 %, RMS +22 %; λ 0.5 gives −21 %, +39 %; mean CV 4.2 → 3.7 m/s gives −10 %, +9 %. Both directions also close the amplitude gap. Botelho 2019 used α = 0.5 "to match the frequency content" of their recordings. Any λ ≠ 1 is a tuned modelling assumption, not a measured value. Delsys confirmation (8 placements, skin 1 S/m, cc / adj): λ 1 0.138 / 0.184 mV, MDF 200 / 198 Hz; λ 0.7 0.166 / 0.220 mV, 175 / 175 Hz; λ 0.5 0.186 / 0.246 mV, 150 / 150 Hz; CV 3.7 m/s 0.148 / 0.198 mV, 176 / 175 Hz. Measured: 0.21 mV, ~165 Hz. λ 0.7 puts adjacent-pin RMS and MDF inside the measured IQR (corner−centre RMS stays ~20 % low); λ 0.5 overshoots MDF; CV 3.7 fixes MDF but leaves RMS low.
- [ ] **% MVC needs a force model.** Nothing in MyoGen maps excitation to % MVC except `ForceModel`. Comparisons against % MVC literature should go through it; consider a helper and a doc note.
- [ ] **Branch-base check** in the contributor notes: before branching, verify the base contains upstream fixes for the modules touched (this PR exists partly because of a branch cut from the stale fresh-trunk line).

## 7. Before opening the PR

- [x] Full test suite on `calib/fdi-dd-main` with the Gfluctdv, CV and fibre-layout fixes: 387 passed, 2 skipped (2026-11-01, 29 min). On `fix/semg-amplitude-scale` with IAP λ (`e4d1e8d7e`): 389 passed, 2 skipped (10 min). `calib/fdi-dd-main` rebased onto it: 393 passed, 2 skipped (10 min). Rerun after any further change.
- [x] CHANGELOG entries: §2–§3 sEMG fixes and IAP λ on `fix/semg-amplitude-scale`; §4 ports, optimiser and upgrade notes on `calib/fdi-dd-main`. §6 open items get entries once decided.
- [x] Commit message of `a8ff28728` reworded (history rewritten locally on all three branches; code identical).
- [ ] Push as DRohlf (`--force-with-lease` for `fix/semg-amplitude-scale`); author Devon Rohlf <devon.rohlf@fau.de>.
