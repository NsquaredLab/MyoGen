# Upstream changes for MyoGen: sEMG amplitude, FDI calibration, ported fixes

Working list for the PR(s) into `github/main`. Started 2026-11-01. Keep it current;
tick an item only when it is committed, tested and its evidence is linked.

Base: `github/main` = `b231049c9` (v0.13.0, 2026-10-28). No newer upstream commits as of 2026-11-01.

## 1. Branches and push state

| Branch | Content | State |
|---|---|---|
| `fix/semg-amplitude-scale` | sEMG fixes, tests, harnesses (§2, §3) | pushed to `github` up to `6137959e7`; local tip `1a9174f69` (3 bench/test commits ahead) |
| `calib/fdi-dd-main` | `fix/semg-amplitude-scale` + optimiser port + ported fixes + calibration arms (§4, §5) | local only, rebased onto `9a5672b58` |
| `bench/semg-roeleveld1997` | Roeleveld 1997 skin-σ benchmark (bench only, no `myogen/` changes) | local; 5 commits not in the PR line |
| `fix/semg-amplitude-scale` (local) | Delsys 5-pin sensor + `_fs__Hz` harness | committed `9a5672b58` |

No rebase was needed: every sEMG branch already sits on v0.13.0, so pushing is a fast-forward.

## 2. sEMG physics fixes (on `fix/semg-amplitude-scale`)

Bugs found against Farina et al. (2004), each with a test:

- [x] `21a412f27` complex B(k_z) (imag part was dropped: near-zero bipolar), physical IAP length scale (z/2 stretch), L1/L2 swap (tendons moved with the IZ), per-fibre recentering (up to 40 % bipolar distortion), array broadcasting.
- [x] `abeec9751` continuous-transform scaling: amplitude no longer scales with 1/v.
- [x] `60df7c2e9` **breaking**: physical source current σᵢ·πa²·∂²Vm/∂z² replaces the ad-hoc A = 96/10.
- [x] `6137959e7` raise when a motor unit fails instead of returning zeros (silent zero MUAPs before).
- [x] Tests: `4b71b4afe`, `0d93447b7` (equal-conductivity layers reduce to homogeneous), `f5a389265` (layered conductor and finite fibre vs independent solutions; bipolar chain < 1 %).

## 3. sEMG defaults and features (on `fix/semg-amplitude-scale`)

- [x] `f43287720` + `48ad8c97e` **breaking**: skin σ default 1 S/m. Provenance corrected in `48ad8c97e`: an effective value fitted by Roeleveld et al. 1997 (biceps fall-off), not a gel/hydrated-skin measurement; the `f43287720` message still says "hydrated skin" and should be reworded in the PR description.
- [x] `15bfd2006` **breaking**: FDI literature skin 1.2 mm (Li 2018) and fat 1.3 mm defaults.
- [x] `905ba8edc` fibre-length defaults documented as FDI values.
- [x] Dispersion features (all optional, off unless stated): per-fibre velocity on a shared grid (`69fc2b30b`), per-fibre end-plates / nerve delays / velocities (`c46ec998c`), CV linear in threshold + CV–diameter line (`61f1d7348`), two-layer end-plate coefficients (`852a7fc30`, default in `faa64f80a` **breaking**), tendon-end band and correlated ends (`787b15694`, `c92f8ccbd`), `literature_candidate` preset (`a755b7d97`), iEMG `nmj_coefficients` (`dea9188c0`), shared nerve-model module (`4b0c3b9ce`).
- [x] Harnesses and changelog: `a4d63765a`, `2eb674364`, `db67a0d5b`, `a8e4e8a32`, `c56c865e0`, `7b9ea764e`, `1a9174f69`.
- [x] Delsys 5-pin sensor and `_fs__Hz` harness: `9a5672b58`.
- [ ] Decide whether `bench/semg-roeleveld1997` goes in the same PR (bench only).

## 4. Ported from the fresh-trunk line (fixes that never reached main)

- [x] **Gfluctdv silent on main** — `286d3c3f2`. `Gfluctdv.mod` GLOBAL `multex`/`multin` default to 0 and `AlphaMN.insert_Gfluctdv()` never set them, so the mechanism added zero noise. Fresh-trunk fixed this in `bef5cbe79` (2026-04-17). Test: `tests/test_gfluctdv_noise.py` (failed first with std 0). Also affects main's own `examples/02_finetune/01_optimize_dd_for_target_firing_rate.py`, whose noise search was ineffective.
- [x] **Fibre CV slope bug on main** (`muscle.py`, `_generate_fiber_properties`): `k = 5.5 (m/s)/mm` with diameters in mm gave 0.8 + 5.5·0.05 ≈ 1.1 m/s, then the clip raised every fibre to 2.0 m/s (17 593 of 17 593 fibres at exactly 2.000 m/s, r 4 mm muscle). Used by main's iEMG (`intramuscular_emg.py:256`); surface EMG uses the mean CV and is unaffected. **Fix uses CV = 0.95 + 0.05·d[µm] m/s** (Nandedkar & Stålberg 1983; 3.7 m/s at 55 µm), the line already used by `SurfaceFiberDispersion`; both now read the shared constants `CV_INTERCEPT__M_S`, `CV_PER_DIAMETER__M_S_PER_MM` in `muscle.py`. Not fresh-trunk's `be9cdfe35` numbers (d/13 + 1.5, 5.3 m/s at 50 µm, cited to the same paper but matching neither it nor Blijham et al.'s invasive fit 0.043·d + 0.83). Noise (SD 0.2 m/s) and clip (2–6 m/s) unchanged. Mean fibre CV on main moves from 2.0 to ~3.45 m/s. Test: `tests/test_muscle_fibre_properties.py`.
- [x] **Fibre table cap / shape on main** (`muscle.py`, `generate_muscle_fiber_centers`): `voronoi_pi1e5.csv` covers the square [-1.25, 1.25]²; only ~313 k points lie in its inscribed disc. Beyond that the outer ring held only the corners (outer-ring sector count 0 on the diagonals' complement), and beyond 399 499 the fibre count was silently capped. Fix: unchanged for counts the disc covers (FDI ≈ 54 k, all existing callers below ~313 k get identical layouts); larger counts use a scrambled Halton sequence in the disc (seeded from MyoGen's RNG) with a warning. Chosen over fresh-trunk `a82923b2b` (adds a 23 MB 1 M-point CSV and still caps beyond it). Tests: `tests/test_muscle_fibre_properties.py` (count not capped; outer ring covers the full circle).
- Already resolved on main independently (no action): `correlate`→`convolve` in `SurfaceEMG` (fresh-trunk `3039771ba`); RNG propagation after `set_random_seed` (main uses `get_random_generator()`).
- Out of scope here: fresh-trunk's iEMG rewrite (323 commits on `intramuscular/*` and `electrodes.py`, 68 fix-like). That is its own upstreaming project; nothing in this PR depends on it.
- Audit method (2026-11-01): `git diff $(git merge-base github/main fresh-trunk) fresh-trunk` restricted to source files that exist on main (12 files), plus fix-like fresh-trunk commits on those files, each checked against main's current code. The neuron path (`simulator/neuron`, `nmodl_files`, `utils/helper.py`, `utils/nmodl.py`) has no other code differences besides `nrnmech.dll` rebuilds and the unused beta/effort drive features (ARCH-25/26/30).

## 5. FDI descending-drive calibration tooling (on `calib/fdi-dd-main`)

- [x] `586cdccfa` multilevel DD optimiser + `ramp_analysis`, `fr_optimization_targets`, `parallel_budget`, `fdi_spikes_force.py` (force mapping with the fixed Fuglevand ForceModel, upstream `e794774db`). Thresholds from `RecruitmentThresholds` (muscle-profile library is not on main).
- [x] Results: 120-MU FDI calibrations (Fuglevand 50× thresholds, De Luca 1982 targets) with Gfluctdv enabled. Recruitment ending at 70 % excitation: error 1.317, ISI CV 13.9 %, at 25 %: 90/88 active, 20.3/20.8 Hz. Ending at 52 %: error 0.771, ISI CV 14.8 %, 96/97 active, 22.0/21.0 Hz; its best `gfluctdv_scale` 0.243 is at the top of the search range (0.25), so widen the range before treating ISI CV as converged. Noise-off run (52 %) for reference: ISI CV 8 %.
- [x] Force mapping with the fixed `ForceModel`, cross-checked with an independent Fuglevand implementation (within 0.1 % MVC): 25 % MVC at excitation 0.227 (70 % calibration) and 0.208 (52 %); 25 % excitation = 28.6 % and 30.8 % MVC.

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
- [x] `--extra-trap NAME:LEVEL` option in `fdi_spikes_force.py`: included in `586cdccfa`.

## 6. Recommendations (need Devon's decision before the PR)

- [ ] **Non-zero Gfluctdv default.** Main has no default: noise is on only if a caller runs `insert_Gfluctdv()` and scales it. Recommend an `AlphaMN__Pool` parameter (e.g. `gfluctdv_scale`) with a non-zero default, so pools have literature ISI variability (15–22 %, LIT-17) out of the box. Reference values: VL 0.12 → ~20 % ISI CV at 30 % MVC (fresh-trunk VAL-42); FDI value from the running calibration. Document the value as calibrated, per muscle, with the target it was fitted to.
- [ ] **Skin σ default (1 S/m).** Keep as a documented effective value or change. Evidence so far: Roeleveld fall-off tests (rerun on main, unchanged) show no skin value reproduces the measured fall-off; 1 S/m is closest. Botelho 2019 Delsys comparison at a force-verified 26 % MVC: 1 S/m gives 0.149 (corner−centre) / 0.181 mV (adjacent), 0.15 S/m gives 0.541 / 0.617 mV, vs measured 0.21 mV. Retest on main-calibrated trains (52 % ramp, 25 % MVC, fibre hash `0799727c71`, 20 placements): 1 S/m gives 0.136 / 0.175 mV (MDF 202 / 198 Hz), 0.15 S/m gives 0.486 / 0.653 mV (MDF 240 / 232 Hz), vs measured 0.21 mV (IQR 0.19–0.26) and MDF ~165 Hz (IQR 152–185). 1 S/m is ~20 % low in amplitude and ~20 % high in MDF; see the IAP item below. The comparison target is Botelho's *measurement*. Their parameters (skin 4.88e-4 S/m, fibres 24.8–40.6 µm, muscle 0.40/0.09 S/m) belong to an anatomical FE model (finite, curved, bone, tendons) and do not transfer to the infinite layered cylinder: the same tissue σ produces different potentials in the two conductors. In the cylinder, skin σ is an effective parameter (as Roeleveld 1997 found for their own cylinder), so its value should be justified by matching measurements, not by citing measured tissue σ or another model's inputs.
- [ ] **IAP length (Rosenfalck λ) or mean CV: MDF gap.** Simulated FDI MDF is ~15–20 % above Botelho's measurement at every placement. The muscle layer does low-pass with depth (per-MU MDF 213 Hz at 0–2 mm, ~145 Hz beyond 4 mm), but five superficial MUs carry 39 % of the energy. Bipolar probes at skin 1 S/m (temporary switch, not on this branch): λ 0.7 gives MDF −9 %, RMS +22 %; λ 0.5 gives −21 %, +39 %; mean CV 4.2 → 3.7 m/s gives −10 %, +9 %. Both directions also close the amplitude gap. Botelho 2019 used α = 0.5 "to match the frequency content" of their recordings. Any λ ≠ 1 is a tuned modelling assumption, not a measured value. Delsys confirmation (8 placements, skin 1 S/m, cc / adj): λ 1 0.138 / 0.184 mV, MDF 200 / 198 Hz; λ 0.7 0.166 / 0.220 mV, 175 / 175 Hz; λ 0.5 0.186 / 0.246 mV, 150 / 150 Hz; CV 3.7 m/s 0.148 / 0.198 mV, 176 / 175 Hz. Measured: 0.21 mV, ~165 Hz. λ 0.7 puts adjacent-pin RMS and MDF inside the measured IQR (corner−centre RMS stays ~20 % low); λ 0.5 overshoots MDF; CV 3.7 fixes MDF but leaves RMS low.
- [ ] **% MVC needs a force model.** Nothing in MyoGen maps excitation to % MVC except `ForceModel`. Comparisons against % MVC literature should go through it; consider a helper and a doc note.
- [ ] **Branch-base check** in the contributor notes: before branching, verify the base contains upstream fixes for the modules touched (this PR exists partly because of a branch cut from the stale fresh-trunk line).

## 7. Before opening the PR

- [x] Full test suite on `calib/fdi-dd-main` with the Gfluctdv, CV and fibre-layout fixes: 387 passed, 2 skipped (2026-11-01, 29 min). Rerun after any further change.
- [ ] CHANGELOG entries for §3 breaking changes, §4 ports, §6 decisions.
- [ ] Commit messages: reword `f43287720` rationale in the PR description (see §3).
- [ ] Push as DRohlf; author Devon Rohlf <devon.rohlf@fau.de>.
