# Clean and noisy surface EMG

This example generates eight seconds of sEMG using the public `SurfaceEMG` API
and `ExecutionPolicy`. It then adds independent white Gaussian noise at 20 dB
SNR to each channel. It is an illustration, not a performance benchmark or
an experimental validation dataset.

From the repository root, after installing MyoGen and compiling its mechanisms:

```sh
uv sync --locked --group dev
uv run poe setup_myogen
uv run python benchmarks/semg_demo/generate.py
uv run python benchmarks/semg_demo/add_noise.py
```

Both scripts accept `--output-dir PATH`; the default is the ignored directory
`artifacts/semg-demo`. To redraw the clean plot without repeating the simulation:

```sh
uv run python benchmarks/semg_demo/generate.py --plot-only
```

Outputs include clean/noisy PNG and PDF plots, compressed NumPy arrays, and JSON
metadata. Signal arrays are in microvolts; figures show millivolts. The noisy
archive also contains the clean signal and the added noise separately. The clean
archive remains unchanged. Re-running a script replaces its own output files.

The fixed seed is 20261019 for the muscle and 20261020 for the prescribed firing
and independent noise generators. The muscle uses 100 motor units, with the first
24 simulated, an 8-by-2 monopolar electrode grid at 5 mm spacing, 1 mm electrode
radius, 2,048 Hz output and 4,096 Hz internal sampling. The prescribed firing uses
rate-modulated gamma-renewal intervals and a ramp/hold/ramp schedule; it does not
run a neuronal-drive model. No filtering or amplitude normalization is applied.

SNR uses each channel's mean-square clean signal over the full eight seconds,
including rest. Noise RMS is therefore approximately 10% of that channel's
full-recording RMS. Noise is stationary and independent between channels; it
does not model motion artifacts, electrode drift or mains interference.

**Absolute model amplitude remains unvalidated.** Passing software tests and
adding noise do not establish biological calibration. See the
[validation summary](../../docs/gl/validation.md).
