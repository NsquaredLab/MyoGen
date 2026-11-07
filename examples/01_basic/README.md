# Basics

This example gallery contains tutorials demonstrating core functionality and
workflow patterns. These examples cover the essential building blocks, from
motor-unit recruitment to EMG signal generation.

!!! note
    The gallery is source-only by default: examples are rendered from source but
    are not executed during a normal documentation build. Thumbnails and
    screenshots may reflect a prior verified run, so run an example locally to
    reproduce its outputs. Maintainers can opt into execution with
    `MKDOCS_GALLERY_PLOT=true`.

The examples are self-contained and can be run independently, though they may
depend on each other for context. Following the numbered order provides the
best learning experience.

## Profiles

Examples 01, 03, 04, 05, 06 and 08 simulate the first dorsal interosseous (FDI)
with the `fdi_semg` **profile card** (`myogen/profiles/fdi_semg.yaml`): every
validated setting in one place, each value with its source.

```python
from myogen import load_profile

card = load_profile()                       # fdi_semg
run = card.simulate([(1, 0, 0), (4, 0, 25), (4, 25, 25), (4, 25, 0), (1, 0, 0)])  # % MVC
muscle = card.muscle()
semg = card.surface_emg(muscle, electrode_arrays)   # your electrodes
```

- **Override** a value with a keyword, e.g. `card.muscle(fat_thickness__mm=3 * pq.mm)`. The override
  stays on that card object for every later build (`card.thresholds(n_units=60)` then `card.muscle()`
  builds 60 units); passing the card value again clears it. Use a fresh `load_profile()` for the card as is.
- **Other keywords** (electrodes, sampling, units to simulate, ...) pass through to the class and are
  listed under `keywords` in the built object's `profile_record`.
- **Notes**: when an overridden value is one the drive calibration or the λ fit
  depends on, the card prints a note (once per card and field per run) saying what else to check.
  Silence them with `warnings.simplefilter("ignore", myogen.ProfileNote)`.
- **Record**: `card.record()` lists the card, its version and every override; built
  objects carry it as `profile_record`, spike and surface-EMG blocks as the
  `myogen_profile` annotation.
- **Seeds**: each builder draws its build from its own seed (the card's `seeds`, or `seed=`; recorded as `seed`),
  so a build does not depend on what ran before it, and then gives your generator and seed back. This differs from
  the rest of MyoGen, where library code never seeds and the script owns the seed (one `set_random_seed` call, or
  the default 180319, for the whole script).
- **What reproduces**: on any platform, `card.thresholds()` and `card.muscle()` give the validated thresholds and
  fibres (each fibre's motor unit, diameter and conduction velocity). `card.simulate()` gives the same spike counts
  per unit; spike times agree to NEURON's floating point, which differs between platforms (about 1e-4 relative in
  the sum of spike times). sEMG from a card therefore matches the validation within its limits, not bit for bit.
- **Sampling**: generate sEMG at 2048 Hz (the `SurfaceEMG` default) and iEMG at 10240 Hz. The card's validation
  (`benchmarks/profiles/validate_card.py`) records sEMG at 10240 Hz only to score the 20-2000 Hz band of its Delsys
  comparator.
- `load_profile("fdi_semg.published")` gives each component as its source paper
  describes it (for comparison with those papers and older MyoGen results).
