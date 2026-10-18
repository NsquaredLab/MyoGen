# Basic MyoGen CPU optimization on corrected GitLab main

## Summary

Add opt-in CPU acceleration and adaptive memory management to public/basic
iEMG and sEMG, based on GitLab main fc077cb4fa89b8cb23c7f0ec4fb50dc8fef05831.
Calls without ExecutionPolicy keep existing execution. All configured fibres,
sampling, physiological parameters, electrode behavior and float64 precision
remain intact.

This replaces the GitHub-based optimization in !3. The earlier branch preserved
a baseline missing GitLab's conductor corrections; this implementation retains
those corrections and validates against fresh corrected-GitLab references.
No anatomical/VL implementation or unpublished development changes are included.

## Changelog

- Add explicit ExecutionPolicy, execution reports, CPU-only dispatch and lazy
  public imports while preserving public API and Neo output compatibility.
- Add central adaptive-memory reservations using actual available physical RAM:
  reserve max(4 GiB, 20% of RAM), an additional margin, and an optional managed
  allocation ceiling. Support zero admission, minimum-fit failure, monitoring,
  safe-boundary batch/concurrency reduction, cache eviction and bounded retries.
  Start at most two workers and prevent nested numerical oversubscription.
- Replace full-recording per-MU convolution lists with bounded direct/blocked-FFT
  accumulation. Reuse resampled templates, compatible transforms and content
  keys; preserve overlap/cropping, boolean spike occupancy, normalization,
  timing, units and ordering.
- Stream iEMG fibre batches into MU accumulators, with strict float64 compiled
  current density. Preserve full-MU grids and propagation conventions; retain
  direct calc_sfaps diagnostics.
- Reuse sEMG frequency/source/electrode preparations and fixed-boundary Bessel
  terms. Prepare the output response through A.T w = c and h = w.T b,
  retaining the corrected shared layered-cylinder conductor, integer angular
  orders, absolute-frequency reconstruction, inner-radius Kn scaling, and both
  five-/seven-variable systems. Preserve GitLab's Cython removal.
- Add opt-in versioned, checksummed, atomic completed-MU disk caching, with a
  distinct corrected-sEMG namespace. Keep realized-input dependency keys,
  random-draw order, final RNG state and fresh-call realization semantics.
- Add regression/failure tests, reproducible benchmarks, approved plan documents,
  a public CPU example, Windows setup instructions and measured acceptance reports.
  Fix empty-array fingerprints and DC-only preparation diagnostics.

- Follow-up review fixes preserve surface half-sample rounding, retain worker
  admission through allocation, serialize and recover cache publications, and
  synchronize the dependency lockfile. Add ten focused regression cases and a
  reproducible check of the approved real cohorts; see [verification](fixes.md).

## Validation

Fresh post-fix CPU suite: **257 passed, 6 skipped, 1 deselected**, no failures,
in 77.89 seconds. The unchanged slow clinical iEMG synthesis test remains excluded
under the agreed limited scope. GPU/unavailable-data tests are skipped.

The 470-fibre iEMG and 88-fibre sEMG cohorts also passed fresh post-fix checks
against saved corrected references, including exact RNG/bins/metadata, bitwise
cache reloads and repeated complete recordings. Maximum recording error was
2.66e-15, with zero changed peak samples. No new performance claim is made.

Recorded numerical evidence:

- Full optimized iEMG: all 100 MUs / 58,755 fibres, complete waveform comparison,
  maximum error 8.88e-15; exact identity, RNG and metadata checks.
- Corrected sEMG: fresh complete selected-MU reference comparisons and 450 actual
  sampled fibres across the full population; full raw/resampled waveforms compared.
  Maximum sampled-fibre error 8.33e-17 mV.
- Contract checks cover all 58,755 realized sEMG fibre inputs, ordering and final
  RNG; these are distinct from waveform acceptance.
- Tests cover pressure, concurrent reservations, allocation failures, retry
  accumulation, cache invalidation, FFT boundaries and conductor algebra.
- Per-channel waveform gate: max absolute error <= 1e-12 + 1e-9 * reference peak.
  No alignment, rescaling or amplitude correction is used to force agreement.

The recorded long repeated-schedule checks retain exact spike bins and full
waveform acceptance; peak-index differences at exactly tied repeated maxima are
explicitly documented. These checks establish numerical implementation behavior,
not new biological validation.

## Measurements

Same i7-13800H laptop, four physical-core affinity, native threads=1, CPU only;
actual host RAM is 63.83 GiB.

- Fresh selected-source comparison: iEMG 470 fibres, 25.208 s reference versus
  3.399 s optimized (7.42x); sEMG 88 fibres, 120.729 s versus 8.264 s (14.61x).
- Full optimized iEMG: 325.233 s. No new full-population reference timing claimed.
- Matched optimized branch comparison: the same 470 sEMG fibres, same 5x5
  electrodes, one worker, one numerical thread, three fresh processes each.
  Old median 21.677 s; corrected GitLab median 22.255 s: 2.67% longer.
  Realized inputs, RNG and metadata match; repetitions are bitwise identical
  within each branch. Cross-branch waveform identity is not an accuracy gate.

## Remaining limits

Physical four-core/16-GiB acceptance, exhaustive corrected sEMG full-population
reference comparison, complete larger-MU corrected reference comparisons,
full corrected sEMG pipeline timing and broad sustained-pressure benchmarks
remain pending. No new full-population sEMG run was performed for delivery.
Historical GitHub-based sEMG accuracy and total-runtime figures are not inherited.

The old checkout and ignored benchmark artifacts remain available locally at
the retired commit. Generated arrays, logs and compiled mechanisms stay out of Git.

## Review and reproduction

- [Results and acceptance limits](report.md)
- [Matched 470-fibre comparison](m470.md)
- [Matched measurement manifest](m470.json)
- [Usage and Windows setup](usage.md)
- [Corrected-baseline port plan](plan.md)
