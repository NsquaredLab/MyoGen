# Surface EMG before and after the unreleased surface-EMG fixes

Simulates the same recording with two myogen checkouts and compares them
through one recording chain (3 µV amplifier noise, 20–450 Hz band-pass). The
limb-like scenario matches `benchmarks/semg_dispersion` (`limb`): muscle,
fibres and spike trains are seeded identically and their hashes must agree.

```bash
# run once per checkout; <expected_worktree> is a path fragment that myogen.__file__ must contain
PYTHONPATH=<old checkout> python benchmarks/semg_versions/gen_real.py old <old checkout> out/old.npz 350
PYTHONPATH=<new checkout> python benchmarks/semg_versions/gen_real.py new <new checkout> out/new.npz 350
python benchmarks/semg_versions/plot_real.py out    # semg_realistic_old_vs_new.pdf and PNGs
```

Each checkout keeps its own defaults, including skin conductivity.
