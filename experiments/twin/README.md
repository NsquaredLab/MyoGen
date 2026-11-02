# Digital twin of the motor neuron pool — experiments

Research harness for the digital-twin program. **Canonical research plan:
`docs/DIGITAL_TWIN_ROADMAP_REVISED.md`** (supervisor review, 2026-10-13 —
sequence, gates, claim boundaries, Deliverables A/B/C);
`docs/DIGITAL_TWIN_ROADMAP.md` records implementation status and the
implemented-vs-planned table. This harness covers the drive→pool pathway
only; afferents, muscle/force, and EMG are not wired in yet.

**Status: differentiable proof-of-concept harness.** Simulator validation
(roadmap Phase 1 strict package) has not passed yet; inverse-recovery results
are unconverged and reported at the best iterate with caveats. Do not cite
this as a validated digital twin.

## Layout

| File | Role |
|---|---|
| `poolbuild.py` | Build a NERLab MN pool + compiled step fn + differentiable θ from a `TwinSpec`; knobs: gCaL / gNaP scales, drive gain, exc/inh weights, reversal potentials; synapse site (soma/dendrite), fixed/live driving force, single-exp/Exp2Syn kernel; `simulate_events()` accepts exact event + connectivity matrices for matched-input validation. No afferent/muscle/EMG pathway (see roadmap) |
| `protocols.py` | Command waveforms: triangular ramp (ΔF), trapezoid, step, hold |
| `observables.py` | Smoothed rates (differentiable) + ΔF, recruitment order, hysteresis (analysis) |
| `validation_protocol.py` | Paired NEURON↔Jaxley validation: immutable content-hashed protocol artifact (events, connectivity, synaptic constants, current steps); Jaxley runner implemented, NEURON runner stubbed with checklist |
| `phase1_pic_validation.py` | PIC behaviour: ramp hysteresis, ΔF dose–response vs `gCaL`/`gNaP` scales; matched-synapse CLI; drive vs connectivity variability separated |
| `phase2_benchmark.py` | vmap-batched forward + gradient throughput benchmark |
| `phase3_inverse_mvp.py` | Synthetic inverse problem: hide θ*, recover with optax + surrogate gradients |
| `results/` | Figures / pickles (git-ignored artifacts) |

## Running

Scripts are self-contained; run from this directory with the project venv:

```bash
cd experiments/twin
../../.venv/bin/python phase1_pic_validation.py
```
