# Roeleveld et al. (1997) fall-off test of the skin conductivity

MyoGen's default skin conductivity, 1 S/m, comes from Roeleveld et al. (1997),
J Electromyogr Kinesiol 7:221–232. They chose it because, in their cylinder,
only a well-conducting outer layer made the negative peak (Nu) of biceps
motor-unit potentials fall off with depth and observation distance anywhere
near the measured rates. Their simulated units had all 300 fibres at one point.
They named missing territory size (real units 0.5–2.5 cm across) as an
alternative explanation, and MyoGen spreads fibres over a territory.

This benchmark repeats their test in MyoGen's surface volume conductor with
point and distributed territories, at 1 S/m and at measured skin values.
It answers how amplitude falls off relative to the nearest unit, not absolute
amplitude, so it does not set the step-4 amplitude bands.

## Setup

`gen.py` follows their model III: cylinder r 35 mm, muscle r 32 mm, fat 1 mm,
skin 2 mm, no bone; muscle 0.1 S/m radial, 0.5 S/m axial, fat 0.05 S/m; 120 mm
fibres, 3.8 m/s; monopolar contacts 1.2 mm across. Electrode rows sit 0, 12, 24
and 36 mm from each unit's end-plate (their electrodes 1–4); columns sit at 0,
±12, ±24 and ±36 mm of arc (their positions A–D). Units are territory discs of
radius ρ centred 5–15 mm below the skin surface, clipped to the muscle; depth
is the realised centroid. ρ = 0 is a point unit (20 fibres at one point).

The `Muscle` is built without its own fibre placement; fibre centres and unit
assignment are set directly (the only fibre data `SurfaceEMG` reads). Each
unit runs alone so the electrode rows follow its end-plate, which `gen.py`
recovers by replaying `SurfaceEMG`'s seeded end-plate draw (`centre: "zone"`
checks it: latencies on rows ±6 and ±18 mm from the end-plate must match).

Differences from their model, kept as modelling assumptions: Rosenfalck
intracellular action potential instead of their 4.8 mm tripole; no end-plate
scatter within a unit (theirs: SD 3 mm); fibre length uniform ±4 mm with both
ends moving together (theirs: each end SD 4 mm); one conduction velocity for
all units (they fitted 3.0–3.8 m/s per unit).

## Measured targets

From Roeleveld et al. (1997), Discussion, citing their 52 biceps units: the
negative peak falls with unit depth as depth^-1.8 and with radial
observation distance from the unit as r^-1.0. Their Fig. 4 gives, for three
representative units at 5, 7.5 and 15 mm, Nu above the unit (highest of
electrodes 1–4) of 100 : 45 : 11 % measured, 100 : 43 : 9 % for model III
(skin 1 S/m) and 100 : 35 : 4.6 % for model IIb (3 mm fat, no skin), with IIb
at 5 mm 4.17 times model III at 5 mm.

## Decision rule (fixed 2026-10-31, before any full run)

1. **Numerical convergence.** Nu at A and D for the 5 mm point unit, at skin
   1 S/m and 4.88·10⁻⁴ S/m, changes by under 2 % between the chosen angular
   harmonics `M` and twice that, and between the chosen `n_t` and twice that.
   *Amended after the check ran:* `M` = 256 and above returns all-zero MUAPs
   (a numerical breakdown, see below), so `M` = 128 was checked against 160,
   192 and 224 instead.

   *Result (2026-10-31):* `M` = 128 is within 0.2 % of 224 at A–D, at 5 and
   15 mm, for both skin values. `M` = 64 is not: it moves D by 3 % (1 S/m) and
   53 % (4.88·10⁻⁴ S/m). Doubling `n_t` and both sampling rates
   (256 at 2048/10 000 Hz → 512 at 4096/20 000 Hz) changes A–D by under 0.5 %.
   Defaults: `M` 128, `n_t` 256, 2048 Hz, 10 000 Hz internal. The end-plate
   replay holds: on the 5 mm point unit the peak latencies at ±6 and ±18 mm
   from the replayed end-plate are equal, 3.2 ms apart (12 mm at 3.8 m/s).
2. **Replication of their model** (point units). Model III (skin 1 S/m):
   Nu-above ratios 7.5/5 and 15/5 mm within ±15 % of 0.43 and 0.09. Model IIb
   (skin σ = fat σ = 0.05 S/m, same thicknesses, i.e. 3 mm of fat): within
   ±15 % of 0.35 and 0.046, and IIb/III at 5 mm within ±15 % of 4.17. If
   replication fails, stop: the comparison below would not mean anything.
3. **Fall-off test.** Arms: skin σ ∈ {1, 0.15, 4.88·10⁻⁴} S/m × ρ ∈ {0, 2.5,
   5, 7.5, 12.5} mm. For each, the depth exponent (log Nu at A, 24 mm row,
   against log centroid depth, 5–15 mm) and the median lateral exponent (log
   Nu against log distance over A–D).
   - **Territory suffices:** at a measured σ (0.15 or 4.88·10⁻⁴), some
     ρ ≤ 12.5 mm gives a depth exponent of 1.8 ± 0.3 and a lateral exponent
     ≤ 1.3. Then 1 S/m loses its justification.
   - **1 S/m needed:** only skin 1 S/m (any ρ) meets both. It stays, as a
     documented effective value.
   - **Neither:** inconclusive; the cylinder misses something else.
   The outcome is reported; changing the default is Devon's decision.

## Result of step 2 (2026-10-31): replication fails

Point units, Nu above the unit (highest of the four rows), per fibre:

| Ratio | MyoGen | Roeleveld | Off by | Limit ±15 % |
|---|---|---|---|---|
| III 7.5 / 5 mm | 0.367 | 0.43 | −15 % | borderline |
| III 15 / 5 mm | 0.071 | 0.09 | −21 % | fail |
| IIb 7.5 / 5 mm | 0.275 | 0.35 | −21 % | fail |
| IIb 15 / 5 mm | 0.032 | 0.046 | −30 % | fail |
| IIb / III at 5 mm | 4.88 | 4.17 | +17 % | fail |

MyoGen's point units fall off faster with depth than Roeleveld's, and more so
without the conducting skin. Per the rule, the fall-off arms (`s1`, `s0.15`,
`s4.88e-4`, already run) are not interpreted. The listed difference most
likely to cause this is end-plate scatter: theirs has SD 3 mm within a unit
(about 0.8 ms at 3.8 m/s), which lowers the sharp potentials of shallow units
more than the broad ones of deep units. Their compact tripole source would
push the other way.

### Rerun with their end-plate scatter (2026-10-31): replication passes

MyoGen's two-layer end-plate model cannot give one fixed SD (its spread
depends on unit size, 0.97–3.46 mm at the scale that gives about 3 mm), so
production code is left alone. `replicate.py` uses the fact that the cylinder
is unbounded along the fibres: moving an end-plate by d equals moving the
electrode by −d, so the unit's expected MUAP is a single-fibre profile (one
column, rows every 1 mm, `gen.py` with `z_rows_mm`) smoothed along z by a
Gaussian of SD 3 mm. The fibre ends move with the end-plate here; theirs
scatter on their own (SD 4 mm), which matters little 24–36 mm from the
end-plate. Same limits as before:

| Ratio | SD 0 | SD 3 mm | Roeleveld | Off by (3 mm) |
|---|---|---|---|---|
| III 7.5 / 5 mm | 0.367 | 0.408 | 0.43 | −5 % |
| III 15 / 5 mm | 0.071 | 0.085 | 0.09 | −6 % |
| IIb 7.5 / 5 mm | 0.275 | 0.317 | 0.35 | −10 % |
| IIb 15 / 5 mm | 0.032 | 0.042 | 0.046 | −9 % |
| IIb / III at 5 mm | 4.88 | 4.42 | 4.17 | +6 % |

SD 0 reproduces the first run exactly, which checks the method. With the
scatter, all five pass. The fall-off arms were run without scatter, so they
do not match the replicated model; they need a rerun with the same 3 mm
scatter (rows 12–36 mm around the 24 mm row, one column per position A–D)
before step 3 is read.

## Result of step 3 (2026-10-31): neither, inconclusive

Arms rerun with the replicated model (`run_scatter_arms.sh`: one column per
position A–D, rows 12–36 mm, smoothed with the 3 mm end-plate Gaussian;
`falloff.py`). Target: depth exponent 1.8 ± 0.3 and lateral exponent ≤ 1.3
(measured 1.0).

| Skin S/m | ρ 0 | ρ 2.5 | ρ 5 | ρ 7.5 | ρ 12.5* |
|---|---|---|---|---|---|
| 1 | 2.54 / 1.73 | 2.68 / 1.80 | 2.70 / 1.94 | 2.48 / 2.02 | 0.97 / 0.83 |
| 0.15 | 3.02 / 2.40 | 3.19 / 2.49 | 3.20 / 2.66 | 2.87 / 2.78 | 1.73 / 3.05 |
| 4.88·10⁻⁴ | 3.40 / 2.75 | 3.60 / 2.86 | 3.56 / 3.06 | 3.11 / 3.21 | 1.70 / 3.61 |

Depth exponent / median lateral exponent; no arm meets both.
\*ρ 12.5 mm discs are mostly clipped by the muscle edge: centroids span only
10.0–14.5 mm for nominal depths 5–15 mm, so its exponents describe the
clipping, not a unit at that depth.

- Territory does not explain the slow measured fall-off: from ρ 0 to 7.5 mm
  the lateral exponent rises at every conductivity (nearby fibres boost the
  potential right above the unit), the opposite of what was proposed.
- 1 S/m is the closest of the three at every ρ ≤ 7.5 mm but still falls off
  faster than measured (lateral 1.7–2.0 against 1.0, depth 2.5–2.7 against
  1.8), as Roeleveld et al. also reported for their model III.
- The measured values are much further off (lateral 2.4–3.2, depth 2.9–3.6).
- Without the end-plate scatter all exponents are larger still (SD 0 run
  of `falloff.py`), with the same ranking.

Caveat: the measured depths came from scanning-EMG estimates of each unit's
electrophysiological centre, here replaced by the fibre centroid. The 1.8 / 1.0
exponents are the pooled fit over 52 units quoted in the Discussion (the
regression itself is in Roeleveld et al. 1997b, not read); the lateral limit
treats 1.0 as a within-unit slope, which that fit may not be. A direct check
that needs no exponent is their Fig. 6 (measured/simulated Nu against
distance for model III, which this setup replicates); done below.

## Fig. 6 check (2026-10-31): no physical skin value fits

Fig. 6 plots measured/simulated Nu against radial distance r from the unit,
normalised to the 5 mm unit at A, using their average measured data. Model III
(skin 1 S/m), read from the figure by eye (+-0.2):

| unit depth | A | B | C | D |
|---|---|---|---|---|
| 5 mm (r 5/12/22/32) | 1.0 | 1.4 | 2.4 | 4.2 |
| 7.5 mm | 1.4 | 1.6 | 2.5 | 3.9 |
| 15 mm (r 15/30) | 2.2 | | | 3.4 |

Our 1 S/m arm replicates III (step 2), so measured/ours at skin s is
Fig6_III x ours(1 S/m) / ours(s). `fig6check.py` computes it from the point
arms (`sc_s*_r0`); rows 12-36, end-plate SD 3 mm at the 24 mm row
(their A row is 36 mm for the 5 and 15 mm units; the unsmoothed 36 mm row
gives the same picture). 1 = match:

| skin S/m | 5 mm: A B C D | 7.5 mm: A B C D | 15 mm: A D |
|---|---|---|---|
| 100 | 1.00 0.99 1.12 1.48 | 1.22 1.06 1.13 1.38 | 1.24 1.22 |
| 10 | 1.00 1.05 1.27 1.76 | 1.25 1.13 1.29 1.65 | 1.38 1.46 |
| 3 | 1.00 1.16 1.59 2.44 | 1.31 1.29 1.65 2.29 | 1.67 2.03 |
| 1 | 1.00 1.40 2.40 4.20 | 1.40 1.60 2.50 3.90 | 2.20 3.40 |
| 0.15 | 1.00 2.66 7.35 13.34 | 1.64 2.92 6.90 11.52 | 3.74 8.99 |
| 4.88e-4 | 1.00 6.31 17.97 26.35 | 1.92 5.69 15.16 22.38 | 5.68 16.91 |

Only about 100 S/m reproduces the measured spread, which is not a skin value.
Measured skin conductivities are 3-6x further off than 1 S/m. The measured
lateral and depth spread is therefore not a skin effect; 1 S/m is the
least-bad value of a model that misses something, and its effect on absolute
amplitude is not physically grounded. Step 3 found that territories steepen
rather than flatten the fall-off, so territory size is not the missing piece
either. The 3, 10 and 100 S/m arms were run for this check only (same grid as
`run_scatter_arms.sh`, territory radius 0).

## Side finding: angular resolution limit

With `sampling_points_in_theta_domain` ≥ 256, `SurfaceEMG` returns MUAPs that
are exactly zero for every electrode (seen at 256 and 512 in this geometry;
224 works). The volume-conductor matrices replace NaN entries with zero
(`volume_conductor.py`), which hides the failure. Not investigated or fixed
here.

## Running

MUAPs go through the optimised CPU route (`simulate_muaps(execution=ExecutionPolicy())`),
which reuses the conductor preparation across fibres. It matched the
original route to 3·10⁻¹⁸ mV on the check units and runs at about 0.12 s per
fibre here.

```bash
# from the repository root, with PYTHONPATH set to this checkout if needed
# step 2 (replication), with and without end-plate scatter
python benchmarks/semg_roeleveld1997/gen.py artifacts/semg_roeleveld1997/rep_III.npz '{"skin_sigma": 1.0, "rhos_mm": [0], "depths_mm": [5, 7.5, 15], "point_fibres": 1, "z_rows_mm": [-15, 51]}'
python benchmarks/semg_roeleveld1997/gen.py artifacts/semg_roeleveld1997/rep_IIb.npz '{"skin_sigma": 0.05, "rhos_mm": [0], "depths_mm": [5, 7.5, 15], "point_fibres": 1, "z_rows_mm": [-15, 51]}'
python benchmarks/semg_roeleveld1997/replicate.py artifacts/semg_roeleveld1997/rep_III.npz artifacts/semg_roeleveld1997/rep_IIb.npz 0 3
# step 3 (fall-off), about 20 min with 15 processes
PY=python bash benchmarks/semg_roeleveld1997/run_scatter_arms.sh artifacts/semg_roeleveld1997/scatter
python benchmarks/semg_roeleveld1997/falloff.py artifacts/semg_roeleveld1997/scatter 3
```

`run_arms.sh` and `analyze.py` are the first, scatter-free version (sparse
grid, positions on both sides); they are superseded by the commands above
and kept only to reproduce the first run.

Arms: `III` (skin 1 S/m) and `IIb` (skin σ = fat σ = 0.05 S/m) with point
units; `s1` (1 S/m, territories), `s0.15` and `s4.88e-4` (point units and
territories).
