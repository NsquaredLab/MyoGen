# Validation summary and surface-EMG example

## Runtime validation

The automated suite passed, including optional GPU, spike-binning and
experimental-data checks. Cache-lifetime regressions and repeated
pressure/recovery checks also passed. The documentation build and public-import
checks passed.

Session cleanup releases outstanding leases and finalizer references while
preserving caches on live simulators. Numerical comparison checks retain the
original waveform units, ordering and random-state behavior.

These checks cover software behavior. Broader population-level and performance
validation remains pending; synthetic stress cases do not establish biological
accuracy or deployment-wide resource requirements.

A [compact validation summary](validation.json) is provided without local-machine
or individual-run metadata.

## Clean and noisy sEMG

The [reproduction scripts](https://github.com/NsquaredLab/MyoGen/blob/main/benchmarks/semg_demo/README.md)
generate an illustrative recording with prescribed motor-unit firing and a
monopolar electrode array. This is separate from the benchmark cohort.

![Clean sEMG with channel detail, MUAPs and prescribed firing](figures/semg-clean.png)

The noisy example adds independent white Gaussian noise at 20 dB SNR per
channel. Signal and noise are saved separately for comparison.

![sEMG with Gaussian noise and a clean/noisy comparison](figures/semg-noisy-20dB.png)

Absolute model amplitude remains unvalidated. The plots use the model's output
units without a fitted amplitude correction. Adding noise does not establish
physiological calibration.
