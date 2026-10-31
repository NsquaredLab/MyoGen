# Surface-EMG fibre-dispersion sweep

Paired simulations behind the `SurfaceFiberDispersion` options and the
`SurfaceFiberDispersion.literature_candidate()` preset. Each arm reuses the same
muscle, fibres, spike trains and electrode as its scenario's baseline and
changes only the dispersion configuration, so MUAPs are compared unit by unit.
Fibre and spike hashes are stored in every output and checked when scoring.

## Scenarios

| Scenario | Muscle | Electrode |
|---|---|---|
| `limb` | cylinder r 15 mm, bone 3 mm, fat 4 mm, 70 ± 5 mm fibres, 350 fibres/mm², 100 MUs | bipolar, 5 mm contacts, IED 10 mm, 20 mm from the end-plate band |
| `fdi` | default `Muscle` (tuned for the first dorsal interosseous), 120 MUs | bipolar, 2 mm contacts, IED 5 mm, 8 mm from the end-plate band |

Both: 30 % excitation trapezoid, Fuglevand thresholds (50-fold range) recruited up
to 70 % excitation, 8–35 Hz rate coding, ISI CV 0.2, 2048 Hz. Scoring applies a
20–450 Hz band-pass and no noise.

## Arms

`arms.json` holds every arm per scenario (`baseline` is the undispersed model).
Arms run before the published end-plate coefficients became the default pin
`"nmj_coefficients": "reference_code"`, so they reproduce. `_skin_thickness__mm`
is a harness key for the skin layer.

## Running

```bash
# from the repository root; PY selects the interpreter. Check that the
# recorded myogen path (key "myogen" in each output) is this checkout; set
# PYTHONPATH to the checkout if another one is installed.
PY=python bash benchmarks/semg_dispersion/run_arms.sh fdi out/fdi           # all FDI arms
PY=python bash benchmarks/semg_dispersion/run_arms.sh limb out/limb baseline candidate_published
python benchmarks/semg_dispersion/score.py fdi out/fdi          # scorecard.tsv, semg_dispersion_sweep.pdf
python benchmarks/semg_dispersion/plot_signals.py fdi out/fdi   # semg_dispersion_signals.pdf
python benchmarks/semg_dispersion/plot_amplitude.py out/limb out/fdi out/amplitude.pdf
```

A limb arm takes about 20–25 min on CPU and an FDI arm about 5 min; the runner
does three at a time and skips arms already present.

## Findings (2026-10-31)

- Within-unit fibre CV spread is the strongest lever on MUAP peak-to-peak; a
  CV difference between units only rescales time.
- In the limb scenario, dispersion mostly blunts the end-of-fibre phase of small
  and mid units; plateau RMS falls by about 6 %. In the FDI scenario the short,
  sharp MUAPs of large superficial units lose more, and plateau RMS falls by
  about 17 %.
- Tendon ends: a 5 mm band of independent ends lowers median MUAP
  peak-to-peak by about 17 % (limb); ends correlated over 5 mm across the
  cross-section, so that neighbours end close together, by about 11 %.
- FDI skin thickness 1.29 → 3 mm lowers RMS by about 60 %; it is the strongest
  amplitude lever found so far.

`benchmarks/semg_versions` compares outputs before and after the unreleased surface-EMG fixes.
