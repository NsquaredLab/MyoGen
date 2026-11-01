# Upstream changes for MyoGen: sEMG amplitude, FDI calibration, ported fixes

Working list for the PR(s) into `github/main`. Started 2026-11-01. Keep it current;
tick an item only when it is committed, tested and its evidence is linked.

Base: `github/main` = `b231049c9` (v0.13.0, 2026-10-28). No newer upstream commits as of 2026-11-01.

## 1. Branches and push state

| Branch | Content | State |
|---|---|---|
| `fix/semg-amplitude-scale` | sEMG fixes, tests, harnesses (§2, §3) | pushed to `github` up to `6137959e7`; local tip `1a9174f69` (3 bench/test commits ahead) |
| `calib/fdi-dd-main` | `fix/semg-amplitude-scale` + optimiser port + Gfluctdv fix (§4, §5) | local only, tip `001cbb105` |
| `bench/semg-roeleveld1997` | Roeleveld 1997 skin-σ benchmark (bench only, no `myogen/` changes) | local; 5 commits not in the PR line |
| `semg-notebook-fixes` worktree | uncommitted: Delsys 5-pin sensor + `_fs__Hz` in `gen_arm.py`, arms in `arms.json` | **commit before PR** |

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
- [ ] Commit the Delsys 5-pin sensor and `_fs__Hz` harness changes (uncommitted in `semg-notebook-fixes`).
- [ ] Decide whether `bench/semg-roeleveld1997` goes in the same PR (bench only).

## 4. Ported from the fresh-trunk line (fixes that never reached main)

- [x] **Gfluctdv silent on main** — `001cbb105`. `Gfluctdv.mod` GLOBAL `multex`/`multin` default to 0 and `AlphaMN.insert_Gfluctdv()` never set them, so the mechanism added zero noise. Fresh-trunk fixed this in `bef5cbe79` (2026-04-17). Test: `tests/test_gfluctdv_noise.py` (failed first with std 0). Also affects main's own `examples/02_finetune/01_optimize_dd_for_target_firing_rate.py`, whose noise search was ineffective.
- [ ] **Fibre CV slope bug on main** (`muscle.py:326`): `k = 5.5 (m/s)/mm` with diameters in mm gives 0.8 + 5.5·0.05 ≈ 1.1 m/s, then the clip raises every fibre to 2.0 m/s (checked 2026-11-01: 17 593 of 17 593 fibres at exactly 2.000 m/s, r 4 mm muscle). Fresh-trunk `be9cdfe35` (2026-03-31) uses CV ≈ d[µm]/13 + 1.5 (Nandedkar & Stålberg 1983), i.e. `k = 77 (m/s)/mm`, `c = 1.5 m/s`, clip 2.5–6. Used by main's iEMG (`intramuscular_emg.py:256`); surface EMG uses the mean CV and is unaffected. Needs a failing-first test and port.
- [ ] **Fibre table cap / shape on main** (`muscle.py:441-495`): `voronoi_pi1e5.csv` has 201 049 points in the unit disc. Above ~314 k fibres the selection reaches the square's corners (non-circular layout); above 399 499 it silently caps the fibre count. Not hit by FDI (≈ 54 k fibres at 8 mm, 267/mm²), hit by large muscles (VL r 20 mm at 312/mm² ≈ 392 k). Fresh-trunk `a82923b2b` adds a 1 M-point table and a warning. Port at least the warning.
- Already resolved on main independently (no action): `correlate`→`convolve` in `SurfaceEMG` (fresh-trunk `3039771ba`); RNG propagation after `set_random_seed` (main uses `get_random_generator()`).
- Out of scope here: fresh-trunk's iEMG rewrite (323 commits on `intramuscular/*` and `electrodes.py`, 68 fix-like). That is its own upstreaming project; nothing in this PR depends on it.
- Audit method (2026-11-01): `git diff $(git merge-base github/main fresh-trunk) fresh-trunk` restricted to source files that exist on main (12 files), plus fix-like fresh-trunk commits on those files, each checked against main's current code. The neuron path (`simulator/neuron`, `nmodl_files`, `utils/helper.py`, `utils/nmodl.py`) has no other code differences besides `nrnmech.dll` rebuilds and the unused beta/effort drive features (ARCH-25/26/30).

## 5. FDI descending-drive calibration tooling (on `calib/fdi-dd-main`)

- [x] `da648a2e4` multilevel DD optimiser + `ramp_analysis`, `fr_optimization_targets`, `parallel_budget`, `fdi_spikes_force.py` (force mapping with the fixed Fuglevand ForceModel, upstream `e794774db`). Thresholds from `RecruitmentThresholds` (muscle-profile library is not on main).
- [ ] Results: 120-MU FDI calibrations (recruitment ending at 70 % and 52 % excitation, De Luca 1982 targets) rerunning with Gfluctdv enabled. Noise-off run (52 %): counts and rates on target (25 %: 93/97 active, 21.3/21.0 Hz) but ISI CV 8 % (target 15–22 %).
- [ ] Decide whether the optimiser belongs in `myogen/` + `examples/finetune/` upstream or stays a bench tool. Main already has `examples/02_finetune/01_optimize_dd_for_target_firing_rate.py`; reconcile or replace.
- [ ] `--extra-trap NAME:LEVEL` option in `fdi_spikes_force.py` exists only on the old fresh-trunk calibration branch; port if still needed.

## 6. Recommendations (need Devon's decision before the PR)

- [ ] **Non-zero Gfluctdv default.** Main has no default: noise is on only if a caller runs `insert_Gfluctdv()` and scales it. Recommend an `AlphaMN__Pool` parameter (e.g. `gfluctdv_scale`) with a non-zero default, so pools have literature ISI variability (15–22 %, LIT-17) out of the box. Reference values: VL 0.12 → ~20 % ISI CV at 30 % MVC (fresh-trunk VAL-42); FDI value from the running calibration. Document the value as calibrated, per muscle, with the target it was fitted to.
- [ ] **Skin σ default (1 S/m).** Keep as a documented effective value or change. Evidence so far: Roeleveld fall-off tests (rerun on main, unchanged) show no skin value reproduces the measured fall-off; 1 S/m is closest. Botelho 2019 Delsys comparison at a force-verified 26 % MVC: 1 S/m gives 0.149 (corner−centre) / 0.181 mV (adjacent) vs measured 0.21 mV; the 0.15 S/m arm is pending. Final retest on main-calibrated trains pending.
- [ ] **% MVC needs a force model.** Nothing in MyoGen maps excitation to % MVC except `ForceModel`. Comparisons against % MVC literature should go through it; consider a helper and a doc note.
- [ ] **Branch-base check** in the contributor notes: before branching, verify the base contains upstream fixes for the modules touched (this PR exists partly because of a branch cut from the stale fresh-trunk line).

## 7. Before opening the PR

- [ ] Full test suite on `calib/fdi-dd-main` (neuron subset: 64 passed after `001cbb105`).
- [ ] CHANGELOG entries for §3 breaking changes, §4 ports, §6 decisions.
- [ ] Commit messages: reword `f43287720` rationale in the PR description (see §3).
- [ ] Push as DRohlf; author Devon Rohlf <devon.rohlf@fau.de>.
