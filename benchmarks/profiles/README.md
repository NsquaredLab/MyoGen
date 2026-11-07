# Profile card validation

`validate_card.py` runs a profile card end to end and checks it against limits frozen before the run.

```bash
python benchmarks/profiles/validate_card.py fdi_semg benchmarks/profiles/results/fdi_semg.json
python benchmarks/profiles/validate_card.py --mvc-mapping fdi_semg.published benchmarks/profiles/results/fdi_semg.published_mvc.json
python benchmarks/profiles/validate_card.py fdi_semg.published benchmarks/profiles/results/fdi_semg.published.json
```

Run at Idle priority with two logical cores left free; `--n-jobs` (default 16) sets the MUAP workers.

## What it checks (`fdi_semg`)

- **Protocol**: `card.simulate` at 25 % MVC with the sEMG benchmark timing (0.5 s rest, 2 s ramp, 4 s hold, 2 s ramp,
  0.5 s rest), the calibrated drive.
- **Recording**: `card.muscle()`, `card.surface_emg` with eight Delsys 5-pin sensors (z 3 / 9 mm, rotation 0 / 45°,
  θ 0 / 180°; 3 × 3 monopolar grids, 2.5 mm pitch, 0.25 mm pins) at 10240 Hz, as arm F of
  `benchmarks/semg_dispersion`.
- **Score** (`delsys.py`): corner–centre (cc) and adjacent-corner (adj) channels, 20–2000 Hz, plateau RMS and median
  frequency, medians over sensors × channels.
- **Limits** (frozen by Devon 2026-11-03, from arm F re-scored with `delsys.py`): cc and adj RMS within ±10 % of
  0.196 / 0.233 mV, median MDF within ±5 % of 162.5 / 157.5 Hz. The membrane noise of the drive is not seeded
  (NEURON `normrand`), so a rerun differs slightly; the limits allow for that.

The published variant is run without limits (its components are the published ones, not the fitted ones); its result
records what it gives.

## MVC mapping

`--mvc-mapping` runs the card's drive on the MVC ramp of `examples/02_finetune/07_fdi_spikes_force.py` (0.5 s rest,
excitation 0 → 1 at 0.1 / s, 3 s hold; MVC = mean force over the last 2 s; force smoothed over 0.25 s) with the card's
force model and reports `drive.mvc_model_units` and `drive.excitation_at_pct_mvc`. The `fdi_semg.published` values
were measured this way, since its force-frequency curve differs from the calibrated card's.

Results are in `results/`; each card's `validation:` block points to its file.
