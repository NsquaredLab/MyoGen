# Baseline verification

GitLab base: fc077cb4fa89b8cb23c7f0ec4fb50dc8fef05831.
CPU-only pytest: 127 passed, 3 skipped, 1 deselected, 83 warnings in 67.56 s.
The deselection is the unchanged slow clinical iEMG synthesis integration test.
Skips: unavailable CuPy and two unavailable experimental-data checks.
Warnings remain in artifacts/gl/baseline.txt and baseline.xml.
No scientific source changed before the baseline or profiling run.

The standard corrected-conductor tests passed, including absolute-frequency
reconstruction, even/odd grids, angular periodicity, thin-bone behavior and
the homogeneous-cylinder closed-form check. These mathematical regression
checks are separate from public-population waveform equivalence.

The corrected sEMG profiling run uses all 88 fibres of public MUs 0-3 within
the complete frozen 100-MU/58,755-fibre population, with 5x5 contacts and the
original full frequency grids. Profiling overhead is excluded from speed claims.
