# Digital Twin of the Motor Neuron Pool — Roadmap

Goal: turn the differentiable MyoGen/Jaxley pipeline into a *digital twin* of the
motoneuron pool — a model whose hidden physiology (PICs, excitation, inhibition,
afferent gain) can be **inferred from motor output** via gradients, validated
against NEURON and ultimately against human HD-sEMG.

**⚠ SUPERSEDED FOR RESEARCH DIRECTION: the canonical plan is now
`DIGITAL_TWIN_ROADMAP_REVISED.md` (supervisor review, 2026-10-13).** This file
remains the *implementation-status* record (what the harness does, measured
numbers, caveats); research sequencing, claim boundaries, gates, and the
immediate implementation queue (Deliverables A/B/C) are defined in the
revised document.

**Status (updated 2026-10-13, after external review): what exists today is a
differentiable proof-of-concept harness, NOT a validated digital twin or a
working inverse solver.** Simulator validation (Phase 1, strict form below)
must pass before inverse-problem or mechanistic-attribution claims are made.
The revised research sequence is:

> define the inference problem → verify the forward model → qualify
> observations and sensitivities → identifiability / protocol design →
> staged synthetic recovery → mechanism discrimination under mismatch →
> human predictive evaluation.

Builds on: `docs/DIFFERENTIABILITY_STATUS.md` (end-to-end JAX differentiability,
`run_jax` / `value_and_grad_run` / `compile_run`) and the NERLab MN architecture
(`docs/session_notes/2026-06-02_jaxley_nerlab_architecture_migration.md`).

All experiment code lives in `experiments/twin/` (shared harness + one runnable
script per phase). Results land in `experiments/twin/results/`.

---

## Phase 0 — Twin harness (infrastructure)

Shared library the phases build on — **implemented vs planned**:

| Piece | Implemented | Planned (not yet in harness) |
|---|---|---|
| `poolbuild.py` `PoolTwin` | gCaL / gNaP scaling, drive gain, exc/inh weights; synapse site soma **or dendrite**; fixed **or live** driving force; single-exp **or Exp2Syn** kernel; connectivity seed | afferent input / afferent gain, spindle params, muscle/force, EMG, full closed loop (use `closed_loop.py` / ex11 meanwhile) |
| `protocols.py` | triangular ramp, trapezoid, step, hold | noise/chirp protocols, protocol optimization |
| `observables.py` | smoothed rates (differentiable); recruitment/derecruitment, hysteresis, ΔF with Gorassini/Afsharipour safeguards (analysis tier) | force & EMG observables, after-discharge stats, differentiable ΔF surrogate |

Caveats that must be stated with any result produced by the harness:

- Inhibition is a **negative-current effective input**, not a shunting
  conductance on the membrane — an inferred `w_inh` is an effective parameter,
  not a physiological inhibitory conductance.
- `drive_gain` (presynaptic rate) and `w_exc` (postsynaptic amplitude) are
  strongly degenerate over simple protocols; fits that fix one of them have
  not solved the full excitation-inference problem.
- The drive generator with `dd_batch=5` is a **gamma/Erlang renewal process**
  (more regular than Poisson); `dd_batch=1` gives true Poisson.

## Phase 1 — Validate the twin (MyoGen/NEURON ↔ Jaxley)

Already validated (see memory + session notes): membrane dynamics (ex02 0.3 %),
population firing (ex03 ≤10 %), force (ex08 0.19 %), cell params Rin/Ir/AP/AHP/F–I
(ex10 ≤5 %), full network (ex11 plausible).

**Gap — PIC-specific behaviour, not yet validated. Strict validation package
(must pass before Phases 3–5 claims):**

1. **Intrinsic single-cell validation** — identical NEURON and Jaxley cells
   under deterministic somatic AND dendritic current injection; compare
   voltage traces, spike times, dendritic Ca and NaP currents, recruitment /
   derecruitment currents, plateau termination.
2. **PIC parameter grid** — gCaL ×{0, 0.5, 1, 2, 4, 6.5, 7.5} crossed with
   NaP ×{1, 5}. The repository's established bistable regime is roughly
   pool `gamma` 1.3–1.5 (≈ gCaL ×6.5–7.5 at the default `gamma=0.2`) with
   NaP ×5 — measured there: Jaxley ΔF ≈ 10 Hz (CaL×6.5) and ≈ 16 Hz with
   7/8 MNs self-sustained (CaL×6.5 + NaP×5); NEURON hysteresis 0.088 nA
   (control) → 0.921 nA (strong PIC). **Estimator provenance:** those ΔF
   values came from the earlier *permissive* pairing criteria; the strict
   Gorassini/Afsharipour estimator now in `observables.py` yields **zero
   valid pairs** in the fully bistable condition (every unit fires
   self-sustained to the end of the protocol, so no control outlasts its
   test unit) — there, drive-hysteresis (−10 to −14 Hz at ×6.5, measured
   2026-10-13) and self-sustained counts (7/7) carry the evidence, exactly
   the ΔF saturation limitation the methods literature warns about. A
   low-range-only grid (×0–2) finding no bistability supports ONLY "none
   detected in that range".
3. **Matched synaptic validation** — same presynaptic event matrix,
   connectivity, synapse location (dendrite), rise/decay constants, reversal
   potentials, and weights in both simulators; no independent re-tuning of
   the Jaxley weight during validation. (`PoolTwin` now supports
   `syn_site="dendrite"`, `live_driving_force=True`, Exp2Syn kernel.) The
   fixed-driving-force somatic IIR remains available as a fast surrogate but
   is NOT a basis for mechanistic simulator validation.
4. **Pool validation over seeds** — recruitment, rate trajectories, ΔF,
   hysteresis, active counts, after-discharge distributions over multiple
   connectivity AND spike seeds, with confidence intervals.
5. **Pass criteria defined before running** — tolerances for waveform error,
   spike counts, recruitment/derecruitment thresholds, ΔF, population stats.

Deliverables:
- `phase1_pic_validation.py` — grid/ΔF/hysteresis runner (`--scales … 6.5`,
  `--gnap-scale`, `--n-seeds` / `--n-conn-seeds` reported separately,
  `--syn-site dendrite --live-df --tau-rise 0.2 --tau-decay 0.3`,
  `--w-exc/--w-inh/--e-exc/--e-inh`).
- `validation_protocol.py` — **the paired-validation path**: builds one
  immutable, content-hashed artifact (frozen event matrices, connectivity,
  synaptic constants incl. the recorded E_inh convention, cell scales,
  current-step waveforms); `run_jaxley(artifact)` implemented
  (matched events via `PoolTwin.simulate_events`, intrinsic steps via
  `simulate_current`); `run_neuron(artifact)` is a documented stub — its
  checklist is the next implementation step (NEURON is installed in the
  venv). Matched runs must use NEURON's own synaptic weight (0.15 µS), not
  the 0.009 µS surrogate value — at τ=0.2/0.3 ms the surrogate weight
  delivers ~no charge.

### Phase 1B — FIRST PAIRED NEURON ↔ Jaxley RESULT (2026-10-18)

`validation_protocol.py --backend both` (process-isolated subprocesses — JAX
and NEURON's compiled mechanisms segfault when mixed in one process; also
fixed: building a second NERLab pool while the first lives segfaults
`finitialize`, the runner reuses one pool). Artifact v2 (hash ef1bd7e8…),
matched construction (normalized thresholds, v_init = 0, Exp2Syn 0.2/0.3 ms
on the dendrite, zero NetCon delay, backward Euler both sides, sample
alignment NEURON[k+1]↔Jaxley[k]). Evidence:
`evidence/val_comparison_ef1bd7e84b9abd03_*.json`, figure
`results/val_comparison_overlay.png`.

| Phase | Result |
|---|---|
| Matched synaptic events (w=0.009 µS, subthreshold by design) | soma RMSE **0.00035 mV**, max |Δ| **1.06 µV**; dendrite similar |
| 5 nA dendritic step (subthreshold) | RMSE 0.003 mV |
| 15 nA step | per-MN spike counts **identical** (67/65/62/54/0×4), recruitment order identical; 180/248 spikes within ±5 ms; soma RMSE 3.1 mV (spike-edge misalignment dominates) |
| 30 nA step | counts 745 vs 744 (one spike in 745); 6/8 recruited identically; 462/745 within ±5 ms |

Reading: membrane + synaptic implementations agree to µV at subthreshold;
spiking agrees in recruitment and counts with slow spike-time drift over
seconds (implementation-level, both backward Euler — characterize, don't
demand equality, per the revised roadmap's distribution-vs-trace guidance).
Firing-regime synaptic comparison (w = 0.15 µS, artifact bac41198…):
**pool over the full 19 s ramp — Jaxley 407 vs NEURON 405 spikes, 5/8
active both, recruitment identical, 356/407 (87 %) of spikes within
±5 ms**; dendritic RMSE 0.38 mV (figure
`results/val_comparison_firing_overlay.png`). Ongoing synaptic input
re-synchronizes the trajectories, so spike-time agreement is better than
in the deterministic steps. Pass criteria for confirmatory runs should now
be frozen FROM these exploratory error distributions (Phase 1C rule).

**Deliverable C result (v2 nesting-correct, 2026-10-18): at the sampled
resolution, releasing protocol-specific input amplitudes did NOT widen the
gCaL sampled compatibility interval** — known vs released intervals
identical in all replicates. (v1 of this analysis refined the two profiles
over different candidate sets, violating the nesting property
L_unknown ≤ L_known at 1–2 points per replicate; v2 minimizes both over
one shared candidate union, asserts nesting per point, and has an
analytic regression test. "Sampled compatibility interval" = deterministic
level set at EPS, not a statistical confidence interval — terminology
applies to Deliverable B's intervals as well.) Scope: input *realization*
frozen; only amplitudes released; w_exc structurally non-identifiable by
declared symmetry. Comparison evidence now carries the same provenance
manifest as B/C; direct tests exist for run_neuron (subprocess smoke),
backend sample alignment, and the spike-matching pipeline (30 tests).

### Phase 1C — FROZEN acceptance spec + CONFIRMATORY PIC grid (2026-10-18)

- **Acceptance spec v1** (`experiments/twin/configs/phase1_acceptance_v1.yaml`,
  FROZEN): thresholds derived as T = max(k·E_numerical, α·D_scientific) from
  exploratory data only; verdict classes PASS / FAIL /
  OUTSIDE_VALIDATED_ENVELOPE / NOT_EVALUABLE; evaluator in
  `experiments/twin/acceptance.py`. Revisions require v2 — no silent edits.
- **Confirmatory grid** (`pic_grid.py`, previously-unused seeds 101/202,
  gCaL {0, .5, 1, 2, 4, 6.5, 7.5} × gNaP {1, 5}, slow-ramp ΔF protocol,
  identical frozen artifacts both backends, evidence `pic_grid_full_*`):
  **PASS under the frozen policy — 5/221 metric cells failed (2.3 % ≤ 10 %).
  Bistability boundary agreement EXACT at both NaP levels** (NaP×1: both
  backends first bistable at CaL 7.5, 6/8 units; NaP×5: both at 6.5, 7–8/8
  units; both seeds). Spike-count agreement ≤ 0.5 % everywhere (up to
  3321 vs 3310 at the extreme). **Validated envelope: gNaP×1 → CaL 0–7.5
  passes fully; gNaP×5 → CaL 0.5–4.0** (failures at the strongly-bistable
  corner are ±5 ms-timing-floor and one rate-MSE exceedance — the
  transition region remains unresolved at spec-v1 tolerances, exactly the
  envelope formulation the revised roadmap calls for). dt-refinement
  E_numerical: 0.72 Hz² (ordinary regime) vs 3.49 Hz² (bistable corner) —
  note for a future spec v2: the corner's numerical floor exceeds the v1
  rate-MSE threshold, so that cell's FAIL is tolerance-limited, not
  necessarily physical.
- Downstream rule now in force: inverse-problem claims are limited to the
  validated envelope above.

### Phase 3 ladder step 2 — CaL + NaP profiles (r2a, 2026-10-18)

`r2a_profiles.py` (full, evidence `r2a_profiles_full_*`): 3-D dense grids
(CaL × NaP × w_exc, truth on every axis, w_exc reoptimized as nuisance,
binned observable, per-replicate EPS 0.63/0.64, predeclared ±20 %
resolution; hard forward only). **Gate verdict: CaL and NaP do NOT advance
as separately-inferable parameters** — rep0 passes both (CaL
[1.587, 1.6], NaP [0.70, 0.707] — essentially the truth point) but rep1
fails both via ONE isolated secondary mode at (CaL 2.0, NaP 0.25) whose
loss sits within EPS for that realization. Structure finding: the joint
(CaL, NaP) compatibility sets are nearly point-like at the truth (3–4 of
140 grid cells) in both replicates; the problem is not a continuous
tradeoff ridge but an isolated distant compensation mode, and it is NOT a
CaL·NaP product invariance (truth-cluster products ≈ 1.12 vs 0.50 at the
mode). Consequences per the gated plan: (a) no R2 optimizer for separated
CaL+NaP yet; (b) the "combined PIC strength" fallback is NOT supported as
a simple product; (c) the concrete next experiment is protocol design to
discriminate the secondary mode — the validated PIC grid already shows
NaP strongly scales spike counts at higher drive, so adding a
higher-drive or bistable-adjacent fit protocol and re-profiling is the
targeted fix.

### Gradient-repair diagnostics G1–G3 (v2 gates, FINAL — 2026-10-18;
evidence `gradient_repair_x64*_*`, 35 regression tests)

All verdicts below are from the corrected gates (FD-convergence-plateau +
directional cosine; zero reference slopes classed "unresolved"; float64):

- **G1 current-injection**: PASS at dt 0.025 AND 0.0125 at every horizon
  up to 1 s — AD matches the detected FD plateau to 1e-5…9e-4 with cosine
  1.000, gradient magnitudes O(1). dt 0.1/0.05 at T ≥ 0.5 s are
  FD_UNCONVERGED (no local plateau exists — the discretization-chaos
  regime; honestly classified, not failed). Residual dt-drift at T = 1 s
  (0.69 → 0.79 between dt 0.025 and 0.0125) ⇒ full timestep convergence
  still open at long horizons.
- **G1 synaptic regime (the actual fitting regime): PASS at ALL timesteps
  including dt 0.1**, every horizon — gradients O(1), FD-verified,
  cosine 1.000. Irregular event-driven firing does not accumulate the
  deterministic phase-locked sensitivity; the explosion pathology belongs
  to the deterministic current-injection case at coarse dt.
- **G2 under float64: the straight-through direction AGREES with the hard
  CRN-FD reference in 19/20 coordinate-variants (1 unresolved, 0 wrong)**
  across β ∈ {5,10,20}, threshold ±1 mV, dt ∈ {0.1, 0.05}, event shift.
  The float32 run of the SAME estimator was wrong 19/20 — the
  wrong-direction pathology was predominantly a float32 accumulation
  artifact, not (at this operating point) inherent surrogate bias.
  Consequence: float32 surrogate gradients remain disqualified; float64
  surrogate gradients re-enter as a G4 candidate.
- **G3: PASS at the dt 0.025 operating step** — interpolated crossing-time
  AD ≡ FD (<5 %) for 7–8 leading spikes at all tested currents
  (12/15/30 nA); at dt 0.1 only 2–5, confirming the earlier failure was
  timestep-limited. Scope unchanged: frozen-index estimator, no spike
  creation/deletion or matching yet.

Final clean-round verdicts (same source revision, structured gates,
evidence `gradient_repair_{f32,x64,x64_syn}_*`):

- **G2 f32 vs x64 (identical code)**: f32 wrong 19/20 → x64 agree 17/20,
  **wrong 2/20 (the vth = 51 mV variant, both coordinates)**, unresolved 1.
  Precision repairs most of the direction pathology but a residual
  threshold-placement-sensitive surrogate bias persists in f64; magnitudes
  remain 1e8–1e11 (sign ≠ usefulness). The f64 surrogate is a G4 candidate
  with known defects, nothing more.
- **G3 (AD-independent plateau selection)**: PASS at dt 0.025 — 7/8/8
  leading spikes across 12/15/30 nA.
- **G1 b_config (EXACT Deliverable-B pool/protocol/operating point,
  single zero-inserted event realization across dts)**: the first run's
  0.5–1 s horizons preceded recruitment (0 spikes — certified subthreshold
  only; caught by per-case n_spikes reporting). Firing-regime rerun:
  **dt 0.025 PASSES through 2 s of the protocol (60 pool spikes,
  AD ≡ FD-plateau rel 2e-6, cosine 1.000, |g| ≈ 6)**; dt 0.1 is
  FD_UNCONVERGED at 2 s (|g| ~ 1e12 — the 8-MN recruitment cascade behaves
  like the deterministic case, NOT like the benign 1-cell synaptic case);
  dt 0.025 at the full 4 s is FD_UNCONVERGED (|g| ~ 1e2, growing).

**G4 design is thereby fixed**: exact B pool/protocols at dt 0.025 for the
gradient path, losses accumulated over ≤2 s windows with forward state
preserved across boundaries (no rest resets); estimator candidates =
f64 surrogate BPTT (defective but cheap), windowed event-time loss,
voltage-hazard loss; CRN finite differences as control; multi-seed T5
design; gates = ≥80 % accepted hard-forward steps, zero non-finite,
equal-budget Deliverable-B recovery with held-out prediction.

### G4 v2 — verdict (2026-10-18, reviewed wording; evidence
`g4_descent_full_*`, 39 tests)

Stage A (8 cases × 3 start types, full-trace exact-B objective, dt 0.025,
2 s gradient windows): aggregate accepted fractions — surrogate joint 0.83
/ gCaL-only 0.96 / w-only 0.67; CRN-FD reference 0.83 / 0.92 / 0.67; zero
non-finite. **The CRN-FD control is a finite-scale (5 %) secant direction
on a piecewise-constant hard objective, NOT an exact derivative**; the
surrogate's AD gradient is exact for the surrogate graph only — agreement
is empirical directional validation. Case-level: 2/48 paired probes
disagree; of 8 joint failures each, 7 are shared; directional cosine mean
0.90 with ONE near-opposite outlier (−0.9997). Defensible statement:
*aggregate acceptance was similar with mostly aligned directions but
isolated case-level disagreements; the surrogate passes the prespecified
aggregate gate within this envelope — it is NOT pointwise equivalent to
finite differences, and no inferential comparison was performed (≈8
independent cases, not 48 independent probes).*

Stage B (32 runs): w_exc ≤20 % in 32/32 (typically 2–4 %); **gCaL 0/32 and
within-EPS 0/32 for BOTH methods**, including gCaL-wrong-only starts
(optimizers compensate via w and sit at gCaL ≈ 1.0). The conditional
inverse problem is NOT solved by local gradients.

Profile (case 0 ONLY, ~12 × 10 grid with truth inserted, w minimized over
a discrete grid): an isolated exact-fit point at the inserted noiseless
same-model truth with substantially higher losses at adjacent samples
(neighbors 1.32 / 1.74). **Evidence for a narrow basin — NOT proof of a
singleton interval or of global/structural identifiability**; the basin
could be brittle event-sequence matching (frozen events, no noise, same
model), which the repeatability experiments below must test.

**Status: qualified local optimizer for the tested envelope; not yet a
standalone recovery method.** The local-direction question is settled for
this envelope; recovery of gCaL is open and is a global-optimization +
robustness question, addressed next by: adaptive profiles with CONTINUOUS
w-optimization and boundary refinement across all cases; repeatability-
based tolerances (does the basin survive event-realization changes /
noise?); a hybrid solver (coarse gCaL scan × gradient-in-w, multiple
retained basins, local refinement) against an equal-budget derivative-free
baseline.

### G5 — basin characterization + hybrid recovery (2026-10-18,
`g5_basin.py`, evidence `g5_basin_full_*`)

All 8 cases, G4's full-trace objective, continuous w-optimization
(golden-section, 10 evals) at every gCaL, boundary bisection to 2 %:

- **Hybrid solver (coarse 9-point gCaL scan × continuous w-opt → top-3
  basins → qualified-gradient refinement, ~180-eval budget): gCaL
  recovered ≤20 % in 8/8 cases — errors 1 % (low-truth) to 6 %
  (high-truth, limited by scan resolution). Equal-budget Nelder–Mead:
  0/8 (stalls at gCaL ≈ 1.0 every time).** Conditional two-parameter
  recovery (gCaL + w_exc) is now demonstrated in this envelope.
- **Basin width**: at the numerical tolerance the compatibility intervals
  are a few percent wide (e.g. [1.45, 1.60] around truth 1.5) — narrow
  but NOT singletons; interval accuracy is limited by the 10-eval
  w-optimization (two cases' intervals sit 3–5 % off truth for exactly
  that reason — a declared approximation, not a physiology claim).
- **Brittleness answered**: repeatability tolerance T_rep (rate-MSE of
  truth vs an INDEPENDENT event realization) is 0.6–2.2 Hz², far above
  the numerical EPS — and the basin SURVIVES it in 8/8 cases at
  consistently ±25 % ([1.125, 1.875] / [0.525, 0.875]). The gCaL
  information is not event-sequence matching: ±25 % is robust to
  realization variability, with the sharper few-percent needle inside.
- Scope unchanged: conditional (known frozen inputs, same model, no
  observation noise beyond the realization-tolerance analysis); the
  R3 rungs (noise, partial observation, unknown realizations) remain.

## Phase 2 — The computational leap (batched simulation + gradients)

- `jax.vmap` the compiled forward over K parameter sets (and/or seeds):
  one pool, thousands of parameter vectors in parallel.
- Benchmark: wall-time vs K for (a) forward `compile_run`, (b)
  `value_and_grad_run`, on CPU (this Mac) and GPU (cluster) — cells·steps/s and
  parameter-sets/s; compare against a NEURON single-run baseline number.
- Deliverable: `phase2_benchmark.py` → scaling table/plot; the headline
  "N simulations of an M-MN pool per second, with gradients".
- Honest current status: the saved CPU benchmark covers 5 MNs at K ∈ {1, 4} —
  it demonstrates that batching works, not yet "thousands of motoneurons /
  parameter sets"; that claim needs the GPU run at realistic pool sizes.

## Phase 3 — The inverse problem (synthetic, known ground truth)

- Simulate observations with known θ* in **hard** spike mode (the scientific
  forward), θ = {gCaL, gNaP, excitation, inhibition, afferent gain}.
- Hide θ*, start from perturbed θ₀, minimize observable-space loss
  (firing rates ⊕ force; later EMG) with optax Adam via `value_and_grad_run`
  (surrogate spikes for the backward pass).
- Multi-start; report recovery error per parameter, loss landscape slices.
- Deliverable: `phase3_inverse_mvp.py` → recovered-vs-true scatter.

## Phase 4 — Mechanistic attribution ⭐

- Construct **output-matched cohorts**: PIC↑ vs afferent-gain↑ vs inhibition↓
  tuned (by optimization) to produce indistinguishable force / mean firing.
- Fit the full twin to each cohort's observations; test whether the gradient fit
  concentrates on the *true* mechanism.
- Deliverable: attribution confusion matrix over cohorts × recovered mechanism,
  as a function of which observables the fit sees (force only → +rates → +ΔF →
  +EMG).

## Phase 5 — Identifiability map

- J = ∂y/∂θ via `jax.jacrev` over the observable vector y(θ) at physiological
  operating points; Fisher information Jᵀ J, SVD spectra, sloppiness analysis.
- Vary the observable set: EMG, force, recruitment order, firing rates, ΔF, and
  combinations → which physiology is identifiable from which measurements.
- Deliverable: identifiability heatmap (observable set × parameter), condition
  numbers, null-space directions (which parameter combinations trade off).

## Phase 6 — Human validation

- Input: decomposed HD-sEMG spike trains + force from real contractions.
- Fit the twin to one contraction (spike-train/force likelihood), predict unseen
  contractions (different ramp slopes/levels); compare predicted vs measured
  ΔF, recruitment, rate profiles.
- Needs: data-loading path (NWB per ex13), MU-matching across contractions,
  measurement model (decomposition yield/errors).

---

## Risks / notes

- **Exploding gradients (measured 2026-09-23)**: BPTT through ~40k steps of the
  stiff MN dynamics yields |∇| ~1e16–1e22 in both surrogate *and* rate modes —
  the explosion is in the recurrent voltage dynamics, not the spike
  discretization. Raw Adam silently freezes (g² overflows float32 → inf →
  update 0). Current mitigation: element-wise `optax.clip` before Adam
  (momentum sign-descent). Open: truncated-BPTT windows, smoothed-dynamics
  losses, gradient-magnitude-aware observables.
- **Surrogate-gradient bias**: fit with surrogate/rate backward, always *verify*
  in hard mode; rate-based observables (smoothed rates, force) are less biased
  than spike-time losses. Rate mode also has a real *forward* bias (≈35 % lower
  mean rate on the test pool) — don't fit rate-mode sims to hard-mode data.
- **Voltage frame**: NERLab MNs live in the 1952 HH frame (V_rest ≈ 0 mV,
  spike detection at +50 mV). All observables must use the pool's frame.
- **Stochasticity**: fixed PRNG key ⇒ pathwise gradients; for population
  observables average over keys (vmap over seeds — free on GPU).
- **CPU here, GPU later**: everything is written device-agnostic; the Mac dev
  box runs CPU JAX, the benchmark script takes `--device` and just reports what
  it finds.
- Two routes to jaxley-parameter gradients exist, and the harness deliberately
  uses the second: (a) `make_trainable("caL_gcaLbar")` before
  `build_init_and_step_fn` → params appear in `get_parameters()` (the
  DIFFERENTIABILITY_STATUS.md route, used with `run_jax`); (b) **bypass
  `make_trainable` entirely** by scaling leaves of the full `all_params` dict
  returned by `init_fn` inside the traced computation — gradients flow because
  `step_fn` consumes that dict directly. `PoolTwin` uses (b); the NaN-sentinel
  masking it requires is handled in `_scaled_jaxley_params`.
- Inverse-problem status (2026-10-13): surrogate sign-descent moves some
  parameters toward their targets but is **unstable and has not demonstrated
  a valid joint recovery** (best-loss iterate ≈ 40 % below init at best, final
  iterate worse than init). Phase 3 results must be reported at the best
  iterate with that caveat; clipped-gradient trajectories must not feed
  Fisher-information / identifiability claims (Phase 5 needs hard-mode loss
  landscapes, perturbation tests, and gradient checks on continuous
  observables instead).
- **Gradient-qualification gate (Deliverable A, 2026-10-13 —
  `experiments/twin/sensitivity_checks.py`, raw numbers in
  `results/sensitivity_checks.json`): REVERSE-MODE SURROGATE GRADIENTS ARE
  NOT PROMOTED.** T1 plumbing clean; T2 continuous-dynamics derivative
  correct (autodiff ≡ centered FD to ≤0.45 % in the convergence region, so
  the explosion is a property of the dynamics, not an implementation bug);
  T3 drive_gain surrogate sign agrees with the expected-value hard slope
  (+8.6e13 vs +7.2 ± sd over 8 keys — sign informative, magnitude
  meaningless); T4 explosion envelope measured: gradient COMPONENTS ≈ 7e14 at
  0.5 s and ≈ 1e27–1e28 at 1 s (finite — the earlier "Inf at 1 s" was a
  float32 norm-computation overflow, now fixed with a float64 norm); true
  component NaN from 2 s. **T5 FAILED (16-case crossed design: 2 connectivity seeds × 4
  independent event realizations × 2 true points; 0 failures;
  baseline-aware coordinate probes): joint surrogate steps improved the hard
  loss in 16/32 trials. Improvement clusters by the (connectivity × event)
  realization pair — 4 of 8 pairs improve in every trial, the other 4 fail
  in every trial (per-event-seed improvement fractions 0.0 / 1.0 / 0.5 /
  0.5). Coordinate-wise agreement: gCaL 8/12 resolved (4 unresolved — no
  direction improved on baseline), w_exc 8/16; both consistent with chance
  at these sample sizes.** This demonstrates strong realization dependence
  of the fixed-realization (pathwise) surrogate direction and blocks
  gradient promotion; more independent realizations would be needed to
  estimate its prevalence precisely or characterize it as chance-like. The
  result is consistent with — though does not by itself mechanistically
  explain — the phase-3 drift. Version-controllable evidence copies with a
  reproducibility manifest (git revision + dirty flag + SHA256 of the
  untracked diagnostic sources): `experiments/twin/evidence/`. Scoped to
  THIS harness configuration, objective, and frozen realizations — not a
  NEURON-validated forward model. Consequence
  per the revised roadmap's gate: Deliverable B runs on the hard-forward
  grid/refinement baseline; gradient methods stay experimental (candidate
  repairs to test: float64 dynamics, sub-0.5 s windows, forward-mode
  sensitivities, CRN finite differences).
- **Deliverable B — CANONICAL RESULT (v3, binned observable,
  2026-10-18; evidence `deliverable_b_full_binned_*`; the v2 pointwise
  results were invalidated by the profile-axis bug and EPS-sharing — see
  git history of this file for the full retraction).** Dense 21×21+truth
  landscapes, 3 (connectivity × event) replicates, profile-at-truth == 0
  asserted everywhere, EPS per replicate × observation model (dt-halving at
  truth): full-obs 0.30–0.68 Hz², partial-obs 0.42–0.83 — the binned
  observable cuts the pointwise chaotic floor (2.09) by ~3–7×.
  **Identifiability (known-input, noiseless, same-model, grid resolution
  ≈7 %): BOTH parameters are locally identifiable in all 6
  replicate × observation-model combinations — profile ≤ EPS only within
  one grid step of the truth** (gCaL interval = {1.6} in 5/6, [1.6, 1.74]
  in one; w_exc = [0.01034, 0.0105] or {0.0105}). Resolution statements are
  bounded by grid spacing; no sub-grid claims. Estimators: grid+refine best
  candidates (1.46/1.63 Hz²) still above EPS — point estimation at EPS
  precision needs finer search; profiles (which contain the truth) are the
  identifiability instrument. **The gradient method improved for the first
  time under the binned observable** — the default start descended 15
  iterations to (1.183, 0.01065), fit 1.84, beating its start and nearly
  matching the grid; the perturbed start still fails non-finite (1/2
  failures per model). Binning smooths the loss enough for sign-descent to
  make progress — worth a follow-up gradient re-qualification (T5) under
  the binned objective before any promotion. Held-out error remains
  peak-dominated; held-out at truth = 0 exactly.
  Next: Deliverable C (running), NEURON↔Jaxley paired validation (running).
