# Digital Twin of the Motoneuron Pool — Revised Research and Implementation Roadmap

**Date:** 13 October 2026  
**Basis:** the supplied “Digital Twin of the Motor Neuron Pool — Roadmap”, updated 13 October 2026.  
**Review scope:** scientific design and implementation priorities, not a reproduced audit of the repository. Statements about existing functionality and numerical results below are reported by the supplied roadmap, not independently verified here. Proposed work is explicitly prospective.

## 1. Goal and first result

Retain the long-term goal: use MyoGen/Jaxley to infer aspects of motoneuron-pool physiology from motor output and predict responses under new conditions.

Replace the immediate goal of recovering all hidden physiology with:

> Establish which PIC and effective-drive parameters can be distinguished under specified protocols and observation errors, and determine whether qualified gradients improve recovery over a hard-forward baseline.

The first result should be a small, reproducible inverse-problem experiment, not a large cohort attribution figure. A successful fit, unique physiological identification, and successful prediction are different outcomes and must be reported separately.

Use this research sequence:

**Define the inference problem → verify the forward model → qualify observations and sensitivities → determine identifiability and informative protocols → demonstrate recovery → test mechanism discrimination under model mismatch → evaluate human prediction.**

Forward-only identifiability work and gradient debugging may run in parallel. Do not make all scientific progress depend on repairing long-horizon reverse-mode differentiation. Conversely, do not make a gradient-based inference claim without qualifying the gradients used for that claim.

## 2. What the supplied roadmap actually establishes

The source describes a differentiable proof-of-concept harness, not a validated twin or a working joint inverse solver [S1, lines 8–16, 166–173]. It reports:

- `PoolTwin` support for CaL/NaP scaling, drive gain, excitation/inhibition weights, alternative synapse locations and kernels, and fixed/live driving-force options. Afferent input, force, EMG, and the full closed loop are not yet part of this harness [S1, lines 27–35].
- Effective negative-current inhibition, rather than a physiological shunting conductance; a strong drive-gain/excitatory-weight degeneracy; and an Erlang renewal input when `dd_batch=5` [S1, lines 37–46].
- Legacy forward comparisons, but an outstanding strict PIC-specific validation package [S1, lines 48–83].
- Long-horizon gradient explosions, a biased rate-forward mode, and unstable inverse fitting without valid joint recovery [S1, lines 138–173].

Treat existing forward comparisons as scoped regression evidence. Do not silently promote them to validation of a new synaptic configuration, PIC regime, observation model, or human inference task.

The supplied document references code and session notes that were not included. Their numerical claims must be reproduced from versioned configurations before being used as acceptance evidence.

## 3. Define the inference problem before choosing the optimizer

For contraction c of subject s, write the conceptual model as:

    y_sc ~ MeasurementModel(
        Simulator(theta_s, u_sc(t), x0_sc, omega_sc),
        eta_sc
    )

Here:

- `theta_s`: the physiological or effective parameters being investigated.
- `u_sc(t)`: trial-specific neural inputs, which are known in a controlled simulation but generally not measured in human recordings.
- `x0_sc`: initial and history-dependent states, including channel gates, calcium and synaptic states.
- `omega_sc`: stochastic realization and, where appropriate, connectivity realization.
- `eta_sc`: measurement and muscle nuisance parameters, such as missing-unit detection, timing error, twitch properties and electrode geometry.
- `y_sc`: the actual measured data or explicitly defined summaries.

A target force trajectory is not the same variable as descending neural drive. Every experiment must declare which of these quantities is known, estimated, shared, fixed, or marginalized. A conditional fit with known inputs is a useful experiment, but it is not a demonstration of joint excitation inference.

### Parameter ladder

**Level A — numerical diagnostic:** one cell, deterministic known current, one CaL parameter. Intracellular voltage/current observations are allowed here only to test implementation and sensitivities.

**Level B — first output-only inverse problem:** a small heterogeneous pool, two unknowns: CaL scaling and one continuously parameterized effective excitatory-input amplitude. Fix NaP and inhibition explicitly. Use known drive shape and an identical observation definition for every method.

**Level C — conditional physiological separation:** add NaP, then inhibitory strength, separately. Each addition requires an identifiability test. Fixing a parameter at this stage is a declared restriction, not evidence that the parameter has been identified.

**Level D — latent inputs and observation uncertainty:** estimate a low-dimensional trial-specific input representation and measurement nuisance parameters jointly with the retained pool parameters.

**Level E — afferent feedback:** add afferent gain only after defining the pathway, its input, delays, mechanical dependencies and accessible perturbations, and verifying the implemented forward pathway.

Do not assume that CaL and NaP are individually recoverable. If only a combination or a protocol-specific functional measure is supported, report that result and its uncertainty. Such a measure is not automatically a general PIC-conductance estimate.

## 4. Phase 0 — Reproducible experiment specification

**Purpose:** make each claim traceable to an exact forward model, objective and parameterization.

Keep `experiments/twin/` as the experiment entry point. Reuse existing harness functions and runnable scripts where possible; do not reorganize the project solely to renumber phases.

Add a machine-readable experiment specification containing:

- Code revision, dependency versions, device, precision, integration method, time step and simulated duration in physical units.
- Pool size, compartment counts, morphology/channel parameters, units and voltage convention.
- Synapse site, kernel normalization, reversal potential, delays, event construction and inhibition representation.
- Parameter transformations, bounds, baseline values and mapping to the actual channel/input leaves being modified.
- Parameter-dependent versus fixed initialization, preconditioning and rest history.
- Connectivity, input-event and measurement-noise seeds, separately identified.
- Observations, smoothing and extraction rules, missing-data rules, objective, optimizer budget, checkpoint selection and train/validation/test splits.

The source's HH voltage-frame convention must be preserved explicitly [S1, lines 151–152]. Check reversal potentials and detection thresholds under the same convention; do not apply a voltage offset to only one part of the model.

Separate three computational contracts:

1. **Scientific hard forward:** the original dynamics and observation/event definitions used to generate and evaluate scientific outputs.
2. **Surrogate backward:** a separately identified derivative estimator. Its primal output must match the scientific hard forward when advertised as hard-forward/surrogate-backward.
3. **Approximate rate forward:** a different forward model. It may be studied as a reduced model or warm start, but must not silently replace the scientific forward.

**Deliverable:** experiment manifest, tiny deterministic fixtures, and an evidence registry linking every gate to raw results and its allowed operating envelope.

**Gate:** a clean rerun reproduces the declared fixtures and identifies every parameter that actually changes the simulation. No gate passes solely because a script exits successfully.

## 5. Phase 1 — Forward verification and physiological operating envelope

**Purpose:** establish that the implementation represents the intended equations within a defined range. Agreement with NEURON is cross-implementation verification, not independent proof of human physiological validity.

### 1A. Intrinsic dynamics

Retain the source plan's matched somatic and dendritic current injections and PIC grid. Match units, morphology, initial states, channel definitions and stimulation exactly. Compare subthreshold voltage, short-horizon spike timing/counts, channel-current trajectories, recruitment/derecruitment, plateau activation and termination.

Add time-step refinement and precision diagnostics: compare the chosen time step with smaller steps, and use float64 reference runs where feasible. Separate numerical instability from sensitivity of a correct trajectory. Use multiple cell types across the intended pool distribution.

Retain the source's high-CaL/high-NaP cases as **stress and bifurcation tests**, not as human priors. Record absolute conductance densities, baseline scaling and current magnitudes; a multiplier alone does not define a physiological regime. Add appropriately scoped channel-off controls and refine the grid around changes in firing/plateau state. Do not require bistability in every cell or every parameter region.

Long stochastic spike trains need not remain phase-aligned indefinitely to demonstrate agreement in their scientific summary distributions. Predeclare which tests require trace/spike-time equivalence and which require agreement of distributions.

### 1B. Synaptic and pool dynamics

Retain identical event matrices, connectivity, dendritic placement, kernels, weights and reversal potentials across simulators. Do not retune Jaxley to force agreement with NEURON.

An effective negative-current inhibitory model can be verified as an effective-current model. To interpret an inferred parameter as an inhibitory conductance or evaluate shunting effects, implement and verify the corresponding conductance-based pathway first. A live driving-force option for another synapse does not establish that inhibitory shunting has been implemented.

Across independently sampled pools and event realizations, compare recruitment, rate trajectories, active-unit counts, hysteresis, eligible ΔF statistics and after-discharge. Use identical ΔF eligibility rules in both simulators and report the fraction of eligible pairs.

**After-discharge is a primary endpoint.** When units do not derecruit before recording ends, treat duration as right-censored. Do not substitute a maximal duration as an observed termination time, set unavailable ΔF to zero, or discard these trials without reporting them.

### 1C. Acceptance criteria

Define separate tolerances for numerical agreement and biological adequacy. Choose numerical tolerances using refinement studies and the smallest output differences the inverse experiment intends to distinguish. Freeze final thresholds before confirmatory runs; exploratory runs may establish the thresholds, but cannot also serve as their independent confirmation.

The source does not supply enough error distributions or intended biological effect sizes to justify universal numerical cutoffs. Do not invent them as established standards.

**Deliverable:** forward verification report with trace panels, regime boundaries, error distributions, failed cases and a machine-readable validated envelope.

**Gate:** the exact configuration and range used in the next scientific experiment pass the relevant tests. Failure outside that envelope limits scope rather than automatically blocking all work.

## 6. Phase 2 — Qualify observations and sensitivities

Run the two workstreams below in parallel on the verified model.

### 2A. Observation model

Start with spike-derived observations; force and raw EMG are not mandatory for the first recovery experiment.

Build an observation ladder:

1. Oracle: all simulated spike trains, exact identities and known inputs.
2. Partial observation: only a subset of units, with the observation fraction reported.
3. Imperfect observation: missed/extra spikes, timing errors, uncertain recruitment/derecruitment, and incomplete or uncertain unit matching.
4. Selection sensitivity: compare random missingness with plausible recruitment-, amplitude- or quality-dependent detection scenarios. Until calibrated, label these as stress assumptions rather than measured decomposition performance.
5. Force: add a verified muscle/twitch model, noise and nuisance uncertainty.
6. Raw HD-sEMG: add only when waveform information or activity from undecomposed units is needed for a defined inference question, with a validated signal-generation/observation pathway.

Process simulated and experimental observations through the same extraction rules. Estimate noise and repeatability from appropriate calibration data when available. Test the sensitivity of conclusions to uncertain observation-error models.

ΔF is derived from motor-unit discharge data, and its behavior depends on pair selection and discharge characteristics [R4]. Adding ΔF to sufficiently resolved spike/rate data is not a new independent measurement. It can be a useful summary or objective emphasis; model the dependence or call the loss a composite objective. Do not multiply dependent likelihood terms and interpret the resulting sharpness as new evidence.

Similarly, force is not automatically independent of complete spike trains and a known muscle model. It may provide useful additional constraints with partial neural observation or uncertain muscle dynamics. Raw EMG may contain information from undecomposed units, but its added value must be tested rather than assumed.

**Deliverable:** observation operator and corruption tests, a declared initial objective, and eligibility/censoring reports.

**Gate:** the recovery experiment uses only available observations and propagates the stated uncertainty. No hidden oracle identities, inputs or unrecruited-unit labels enter a human-facing fit.

### 2B. Gradient qualification

The source reports exploding gradients and unstable optimization [S1, lines 140–150, 166–173]. Move this from a caveat to a qualification experiment.

**Test 1 — parameter plumbing.** Verify that each parameter changes the intended leaves, cells and compartments only. Audit masks, NaN sentinels, detached computations and parameter-dependent initialization. A nonzero gradient alone is not sufficient.

**Test 2 — continuous sensitivities.** Start with one cell, deterministic inputs, short trajectories and smooth voltage/current objectives. Compare autodiff directional derivatives against centered finite differences across several perturbation sizes and integration settings. Distinguish a finite-difference convergence region from steps dominated by roundoff or numerical/event changes.

**Test 3 — events and stochastic inputs.** Audit the full path from each parameter through random sampling, event times, binning and observation extraction. Fixing a PRNG key provides reproducibility, not automatic differentiability with respect to a distribution parameter. For example, a hard threshold event can have zero derivative almost everywhere for fixed random input while its expected event count changes with the rate. JAX's Poisson sampler produces integer-valued samples [R5]; the actual harness's Erlang/event implementation must be examined separately.

Smoothing a hard-binned spike train after threshold detection does not, by itself, remove the discontinuity in how the bins depend on parameters. Identify whether the derivative comes from continuous event-time treatment, an explicit surrogate, a stochastic gradient estimator or a different forward approximation.

For the first joint inverse experiment, prefer known event input with a continuously scaled effective amplitude, or a deterministic current/conductance input. This deliberately avoids presynaptic-rate inference and must be labeled accordingly.

**Test 4 — horizon and regime.** Extend tests from short trajectories to the actual ramp/hold durations, including plateau transitions. Report gradient behavior by horizon in seconds, parameter region, precision and objective. Distinguish derivative correctness from optimization usefulness near a regime transition.

**Test 5 — hard-forward descent.** For surrogate derivatives, evaluate proposed parameter changes with the scientific hard forward, using independent validation seeds as well as controlled common-random-number comparisons. Report agreement with meaningful finite perturbations, accepted-step fraction, achieved hard-loss improvement and failure rate. Do not demand infinitesimal finite-difference agreement from a discontinuous hard-binned spike objective; test continuous sensitivities and practical descent separately.

### Optimizer decision ladder

Normalize parameter scales and residuals. Use bounded steps or trust-region/line-search safeguards where appropriate. Clipping may prevent overflow, but it does not validate derivative direction or physiological inference. Log unclipped and clipped gradients and optimizer failures.

Checkpointing trades memory for recomputation; it does not repair an unstable sensitivity [R6]. Shorter backpropagation windows alter the derivative and may omit long-history effects. If truncation is studied, preserve all forward states across boundaries, vary window length and evaluate complete uninterrupted hard-forward trajectories.

If needed, evaluate multiple shooting as a separate constrained formulation: introduce boundary states and enforce continuity of voltage, gates, calcium, synaptic and other relevant states. Do not restart each window from rest or allow unconstrained boundary states to explain away physiology. Benchmark this approach; do not promise it will solve the gradient problem [R7].

For the initial two- to five-parameter tasks, compare forward-mode sensitivities, finite perturbations and derivative-free optimization before committing to large reverse-mode Jacobians. These are alternatives to assess, not automatic cures for sensitive dynamics.

**Deliverable:** parameter-by-objective-by-horizon gradient report plus a simple hard-forward recovery baseline.

**Gate:** promote a gradient method only within the range where it is numerically qualified and reliably improves the declared hard-forward objective. Otherwise continue forward-only identifiability/recovery work while the gradient method remains experimental.

## 7. Phase 3 — Identifiability and protocol design before attribution

**Purpose:** determine what information the experiment contains before interpreting an optimizer's preferred solution.

Begin with the two-parameter PIC/effective-drive problem. Compute hard-forward loss surfaces and approximate equivalence regions across several true parameter points, including near-transition cases. Repeat across observation levels and input assumptions.

Distinguish an exact model symmetry from a practical trade-off. If two inputs enter only as `(a-b)*u(t)`, only their difference is observable in that illustrative submodel. But a rate/weight product that controls mean synaptic input need not be an exact symmetry of all higher-order output statistics. Test, rather than assume, whether variability or temporal structure breaks the source's reported drive-gain/excitatory-weight degeneracy.

### Local sensitivity analysis

For a smooth observation mean `mu(phi)`, use dimensionless or explicitly scaled parameters `phi`. With a fixed Gaussian observation covariance `Sigma`, the local information matrix is:

    I(phi) = J(phi).T @ inverse(Sigma) @ J(phi)
    J(phi) = d mu / d phi

This is not universally `J.T @ J`. The unweighted expression assumes a suitable whitened unit-noise model. Correlated rate samples, shared units and redundant summaries need corresponding treatment. If covariance depends on parameters, this mean-sensitivity term is not the full Gaussian Fisher information.

Use qualified sensitivities only. An autodiff matrix computed through surrogate events is a surrogate-sensitivity diagnostic unless justified for the scientific observation distribution. Do not infer Fisher information from clipped optimizer gradients.

Vary nuisance parameters and latent-input coefficients rather than holding them at truth in every analysis. Profile over them, or use an appropriate local nuisance adjustment when valid. Choose forward- versus reverse-mode Jacobians according to parameter and observation dimensions, not by habit.

### Nonlocal checks and uncertainty

Use profile likelihoods or explicitly calibrated loss profiles: fix one parameter, re-optimize the others and nuisance quantities, and examine whether substantially different values remain compatible with the observations. Profile-likelihood methods address parameter ambiguity beyond a single local curvature calculation [R2]. If using a composite discrepancy rather than a likelihood, do not label its profiles as likelihood confidence intervals without calibration.

Evaluate multiple operating points, initial histories, seeds and pool realizations. Distinguish data-supported constraints from parameter bounds or informative priors. Use simulation-based coverage tests before reporting confidence/credible intervals as calibrated. Bootstrap whole simulated subjects/pools/trials at the appropriate level, not all motor-unit pairs as independent observations.

### Protocol selection

Start with a finite, interpretable candidate set: different ramp slopes and amplitudes, ramp–hold–ramp trials, varied conditioning history/rest, and brief controlled perturbations in simulation. Compare protocols under an equal recording-time and measurement-quality budget.

For afferent questions, define feasible measured mechanical/sensory perturbations before optimizing the protocol. A simulated known inhibitory pulse is not automatically a selective intervention available in a human experiment.

Select protocols using held-out parameter draws and nuisance conditions. Seek parameter discrimination and predictive utility, not merely a large local determinant at one favorable point. A protocol search must not exploit an unvalidated surrogate regime.

**Deliverable:** identifiable parameter combinations, ambiguity regions, uncertainty calibration and a small selected protocol set.

**Gate:** proceed with the subset of parameters and claims supported at a prespecified scientific resolution. A reproducible non-identifiability result is a valid outcome and a reason to change the measurement/protocol or parameterization.

## 8. Phase 4 — Staged synthetic recovery

**Purpose:** separate optimizer success, observation limitations and model mismatch.

Use nested experiments rather than starting with all five mechanisms:

- **R0:** one parameter, known deterministic input, same-model noiseless data. This is an implementation test.
- **R1:** CaL scaling plus effective excitatory-input amplitude, multiple true points and initializations, output-only observations.
- **R2:** add NaP and then inhibition individually, conditional on Phase 3 evidence.
- **R3:** incomplete/noisy observations and unknown event realizations; estimate retained parameters and nuisance quantities.
- **R4:** generate observations in NEURON and fit with Jaxley. This tests cross-implementation robustness but still shares the biological model assumptions.
- **R5:** introduce controlled structural mismatch: altered cell heterogeneity, AHP/leak properties, synapse placement/kinetics, or other plausible omitted mechanisms. Add twitch/measurement mismatch only when those components are included.

Same-simulator recovery with identical structure and fixed inputs is a necessary unit test, not a sufficient argument for human physiological recovery.

### Required comparisons

For the small initial problem, use a coarse grid followed by bounded refinement as the transparent reference. For larger low-dimensional problems, choose one suitable derivative-free baseline rather than adding a large optimizer zoo. Compare qualified gradient optimization using equal wall-time or explicitly accounted evaluation budgets, including failed restarts and gradient costs.

Distinguish methods that share the target input realization from methods that must infer/marginalize unknown realizations. Common random numbers can reduce comparison noise, but a hidden ground-truth seed must not become an undeclared inference input.

### Required reporting

Report prediction error and parameter error separately, bias, uncertainty coverage, competing optima, boundary hits, failure rate and wall-time to an accepted solution. Display all starts in the analysis, not only successful recoveries.

Predeclare checkpoint selection. Selecting the minimum training-loss iterate is legitimate when stated, but report the final iterate and stability too. Use separate validation data for tuning/early stopping and an untouched test set for final recovery/generalization claims. Never select checkpoints by closeness to the true hidden parameter.

**Deliverable:** recovered-versus-true plots with uncertainty, hard-forward fits, held-out predictions, baseline comparisons and full failure accounting.

**Gate:** the declared recovery resolution and uncertainty coverage hold on untouched synthetic cases within the stated envelope. A low loss without parameter recovery supports a predictive fit, not identified physiology.

## 9. Phase 5 — Mechanism discrimination under ambiguity and mismatch

**Purpose:** keep the output-matched-cohort idea, but make the experiment test information rather than optimizer preference.

Define hypotheses in terms of the implemented model. Distinguish increased CaL/NaP conductance, reduced effective inhibition, increased afferent gain and changes in neural drive. Do not label an unimplemented afferent pathway or a negative-current term as a verified physiological conductance mechanism.

Construct cohorts matched on specified baseline observations, with a prespecified equivalence margin tied to measurement repeatability. State whether matching concerns means, full trajectories or their probability distributions. Failure to reject a difference is not evidence of equivalence.

Include two distinct experiments:

**Information-null control:** when hypotheses are genuinely indistinguishable under the observations available to the fitter, the method should remain uncertain. With balanced labels and exactly equal observation distributions, above-chance classification would imply leakage or a violation of the matching construction, not discovery of hidden information.

**Perturbation-discrimination test:** preserve baseline matching, then evaluate a held-out protocol whose predicted responses differ across hypotheses. Determine whether the new measurements actually resolve the ambiguity. Match or randomize nuisance distributions across mechanisms so that a classifier cannot exploit seed, recording-length, missingness or file-structure artifacts.

Include pure mechanisms, combinations, null/no-change cases and out-of-model generators. Allow mixed or indeterminate conclusions rather than forcing every case into one label. A single optimizer's parameter allocation is not a mechanism probability.

Compare candidate models with explicit complexity/prior assumptions and held-out predictive performance. Report calibrated evidence or uncertainty where supported. Evaluate attribution under misspecification, not only when the fitted model exactly contains the data-generating mechanism.

**Deliverable:** baseline-equivalence plots, held-out perturbation responses, confusion matrices with indeterminate outcomes, and false-attribution rates under mixtures/mismatch.

**Gate:** informative protocols improve mechanism discrimination while information-null and out-of-model controls do not produce unjustified confidence. Scope conclusions to model-conditional mechanism evidence; do not equate them with causal identification in humans.

## 10. Phase 6 — Human predictive evaluation

**Purpose:** establish subject-specific predictive utility before claiming access to true hidden conductances.

Use decomposed HD-sEMG and force as available. Start with a clearly defined muscle/task, repeated contractions, documented history and quality assessment. Implement the source's NWB/loading and motor-unit matching needs, with explicit uncertainty when identities are not reliable [S1, lines 127–134]. Inspect data availability and repeatability early; do not postpone all contact with human data until this phase.

Fit shared subject-level parameters across multiple calibration contractions. Allow declared trial-specific inputs, initial states and measurement terms. A shared-physiology assumption must be tested across the chosen tasks; changes in neuromodulatory state or fatigue must not be silently absorbed into a supposedly constant subject trait.

### Choose the prediction task explicitly

**Conditional held-out-unit prediction:** estimate the new trial's latent drive from a designated subset of observed units, freeze the subject physiology, and predict disjoint held-out units. No scored unit contributes to input estimation. Report this as conditional prediction, not input-free forecasting.

**New-protocol forecasting:** use a separately specified mapping from the known task/target to neural input, or a calibrated closed-loop controller. Freeze the relevant rules before testing and propagate input uncertainty. Do not insert an unmeasured neural-drive waveform as though it were known.

**Held-out force prediction:** do not reconstruct drive from the same measured force trace and then use agreement with that trace as independent validation. The force target may be a known task instruction, but it is not the measured output and is not a measured descending command.

Use held-out contractions with different slopes, amplitudes, holds and histories. Add sessions or re-donning tests when claiming session transfer. Compare against a simpler fixed-physiology/common-drive baseline with the same available inputs and a comparable fitting budget.

Evaluate force, firing-rate trajectories, recruitment/derecruitment, ΔF where eligible, after-discharge, uncertainty calibration and test–retest stability. Keep participants/contractions as the appropriate independent evaluation units. Do not use large numbers of unit pairs to inflate the effective sample size.

Human output prediction does not directly reveal true CaL, NaP or inhibitory conductances. Initially report effective/model-conditional parameters, identifiable combinations and predictive uncertainty. Stronger mechanistic claims require additional independent constraints or perturbation evidence.

**Deliverable:** leakage-free held-out predictions, repeatability and uncertainty results, baseline comparisons, and a precise statement of what the model can and cannot infer.

## 11. Computational benchmarking — a supporting workstream

Retain the existing benchmark script if present, but change its purpose from a headline to measuring feasible experiment cost.

Measure compilation separately from steady-state execution. Synchronize timed JAX work with `block_until_ready()` or an equivalent appropriate synchronization; device dispatch alone is not completed computation [R3].

Benchmark the scientific forward and qualified gradient methods across pool/compartment size, simulated duration, batch size, precision, recording configuration and seed count. Report peak memory, failures, hardware and full configurations. Use chunked batches when a full `vmap` exceeds memory.

Averaging over seeds is not free on a GPU. Batching can improve throughput but still consumes computation and memory. Distinguish many parameter sets of a small pool from one large physiological pool.

Report both simulation throughput and wall-time to a successful, verified recovery. A fast derivative that requires many failed starts may not improve inference. Compare matched workloads against NEURON where relevant; do not present a single-thread NEURON baseline as a universal state-of-the-art speed comparison.

The source's five-cell, K={1,4} result remains a batching smoke test until larger, relevant configurations are actually measured [S1, lines 85–96].

## 12. The immediate implementation queue

Do not build the whole future pipeline at once. Implement these three deliverables first, reusing the existing harness.

### Deliverable A — scoped forward and gradient diagnostics

Reproduce a one-cell and small-pool verified case. Add smooth-objective finite-difference checks, event/rate-parameter plumbing checks, horizon diagnostics and hard-forward step evaluation. Preserve raw failures.

### Deliverable B — a two-parameter hard-forward experiment

Choose CaL scaling and effective excitatory-input amplitude, with NaP/inhibition fixed and known drive shape. Use a ramp and a ramp–hold–ramp protocol, plus an untouched protocol for prediction. Compare a transparent grid/refinement baseline with the current gradient method. Display the loss surface, equivalent solutions, recovered parameters and held-out outputs.

Start with exact unit observations, then repeat with a declared partial-observation model. Do not add muscle/EMG generation merely to make the figure look more complete.

### Deliverable C — an unknown-input stress test

Release a small number of neural-drive nuisance coefficients, keeping the protocol and measurement assumptions explicit. Recompute profiles and recovery. This reveals whether the apparent PIC estimate survived only because drive was treated as known.

Choose subsequent work from these outcomes: fix derivatives, improve the protocol, reduce/reparameterize the inferred physiology, or expand the observation model. Do not default to adding more free parameters.

## 13. Suggested implementation organization

These are proposed additions or responsibilities, not claims that files currently exist:

    experiments/twin/
      poolbuild.py                  # Reuse the existing forward construction.
      protocols.py                  # Reuse and extend protocol definitions.
      observables.py                # Shared extraction and eligibility rules.
      specifications.py             # Versioned experiment/model contracts.
      observation_model.py          # Partial observation, errors, censoring.
      objectives.py                 # Declared likelihoods/composite losses.
      sensitivity_checks.py         # Continuous checks and hard-step tests.
      inference.py                  # Baselines and qualified gradient methods.
      identifiability.py            # Profiles, local sensitivities, ambiguity.
      protocol_design.py            # Candidate protocol comparison.
      reporting.py                  # Raw-result-linked reports and failures.
      configs/
      results/
      tests/

Keep existing `phase1_pic_validation.py`, `phase2_benchmark.py` and `phase3_inverse_mvp.py` entry points if they are present. Extend or wrap them rather than breaking existing commands. The experiment manifest should contain the scientific phase and purpose, so old filenames do not determine the research order.

Every result should identify its model revision, parameter scope, data split, observation assumptions, gradient mode and gate status. Never replace a failed run with an unmarked successful rerun using different tolerances or seeds.

## 14. Claim boundaries

**After forward verification:** “The Jaxley implementation reproduces the reference model in the tested configurations.”

**After conditional synthetic recovery:** “These parameters/combinations can be recovered under the stated known-input and observation assumptions.”

**After observation/mismatch tests:** “Recovery or ambiguity estimates remain reliable under the tested missingness, noise and model deviations.”

**After human prediction:** “The calibrated subject model predicts specified held-out outputs with quantified uncertainty.”

**Only with further independent evidence:** stronger claims about actual hidden human physiology or causal mechanism attribution.

The long-term digital-twin goal is unchanged. The revised plan makes each step earn the next claim without requiring that every physiological parameter become uniquely identifiable.

## Sources and methodological references

**[S1] Supplied roadmap.** “Digital Twin of the Motor Neuron Pool — Roadmap”, updated 13 October 2026; uploaded as `Pasted text(20261013-061211).txt`. Line references use the line labels shown with the supplied file in the conversation. Repository results described there were not rerun for this review.

**[R1] Deistler et al.** Jaxley: differentiable simulation enables large-scale training of detailed biophysical models of neural dynamics. *Nature Methods* 22, 2649–2657 (2025). DOI: `10.1038/s41592-025-02895-w`. Supports the differentiable-simulation approach; does not validate this particular harness or its long-horizon gradients.

**[R2] Raue et al.** Structural and practical identifiability analysis of partially observed dynamical models by exploiting the profile likelihood. *Bioinformatics* 25, 1923–1929 (2009). DOI: `10.1093/bioinformatics/btp358`.

**[R3] JAX official documentation.** “Benchmarking JAX code.” Documentation path: `docs.jax.dev/en/latest/benchmarking.html`. Consulted 13 October 2026.

**[R4] Stephenson and Maluf.** Dependence of the paired motor unit analysis on motor unit discharge characteristics in the human tibialis anterior muscle. *Journal of Neuroscience Methods* 198, 84–92 (2011). DOI: `10.1016/j.jneumeth.2011.03.018`.

**[R5] JAX official documentation.** `jax.random.poisson`. Documentation path: `docs.jax.dev/en/latest/_autosummary/jax.random.poisson.html`. Consulted 13 October 2026. The discrete-sampling example in this plan is explanatory reasoning, not evidence that the harness uses this function.

**[R6] JAX official documentation.** “Gradient checkpointing with jax.checkpoint (jax.remat).” Documentation path: `docs.jax.dev/en/latest/gradient-checkpointing.html`. Consulted 13 October 2026.

**[R7] Massaroli et al.** Differentiable Multiple Shooting Layers (2021). arXiv: `2106.03885`. Multiple shooting is a proposed method to evaluate here, not an established solution for this model.

All experimental designs, proposed gates, implementation responsibilities and prioritization decisions in this document are recommendations from this review, not results established by these references.
