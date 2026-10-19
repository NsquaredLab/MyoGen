# Corrected GitLab CPU port: results

The later [validation summary and sEMG demonstration](validation.md) records
the cache-lifetime fix, automated test coverage, and noise examples.
The historical measurements and remaining population-validation limits below are
historical evidence and are not replaced by the illustrative demo.

The later [matched 470-fibre sEMG comparison](m470.md) measures this port against
the final old speedup branch under identical worker/thread settings.

Branch: perf/basic-gitlab-cpu. Base: fc077cb4fa89b8cb23c7f0ec4fb50dc8fef05831.
The later [review fixes and verification](fixes.md) cover rounding boundaries,
worker reservations, cache publication/recovery and dependency locking.

## What changed

- Added explicit ExecutionPolicy CPU execution while retaining existing default execution.
- Ported central adaptive-memory admission, monitored headroom, bounded retries,
  safe-boundary batch/concurrency reduction, and execution reports.
- Ported bounded direct/blocked-FFT recording accumulation, resampled-template
  and compatible-transform reuse, and exact boolean spike-bin semantics.
- Ported streamed iEMG MUAP construction and strict float64 compiled current density.
  Full MU time grids, spatial samples and propagation delays remain intact.
- Adapted sEMG frequency/source, electrode, fixed-boundary Bessel and output-response
  preparations to GitLab's shared corrected volume conductor.
- Preserved absolute-frequency index reconstruction, integer angular orders,
  inner-radius muscle Kn scaling, seven-/five-variable systems and Cython removal.
- Ported opt-in checksummed atomic completed-MU caching, lazy public imports,
  dependencies and the runnable public example. Corrected sEMG has its own cache namespace.
- Fixed empty-array fingerprints and empty response diagnostics for DC-only grids.
  The CPU shift-padding test now runs without CuPy and reflects existing behavior.

No anatomical code or dependencies, reduced fibres, changed sampling, altered
physiology, or mixed-precision arithmetic were introduced.

## Measurement conditions

CPU-only comparisons used matched worker and numerical-thread settings.
The measurements do not establish performance on other machines.
The policy itself starts at most two workers and can admit zero work.

Frozen public input: 100 MUs, 58,755 assigned fibres, seed 180319. Selected-MU
measurements retain every fibre in each selected MU and all public sample grids.
Loading, comparison and serialization are outside timed calls. Each reference
workload has one unprofiled observation; these are observations, not confidence
intervals. Optimized source measurements include first-use setup in that process.
Other applications remained running. No unrelated process was stopped.

The baseline CPU suite passed 127 tests with three skips and one deselection.
The unchanged slow clinical synthesis test was excluded from both baseline and
final limited passes. Profiling preceded production edits. iEMG current density
and sEMG repeated source/boundary calculations remained useful targets.
Profiler thread-wait entries overlap numerical work and are not attributed as
separate bottlenecks; profiler runtimes are excluded from speed claims.

## Source generation: fresh paired cohorts

| Complete selected MUs | Reference s | Optimized s | Observed ratio |
|---|---:|---:|---:|
| iEMG, MUs 0/25/50; 470 fibres | 25.208 | 3.399 | 7.42x |
| sEMG, MUs 0-3; 88 fibres | 120.729 | 8.264 | 14.61x |

Fresh corrected-GitLab high-level sEMG output is the reference. Final RNG,
shape/order, units, sampling metadata and peak-sample locations match.
Maximum complete-MU errors were 3.33e-16
for raw iEMG templates and 8.88e-16 mV
for sEMG templates. No alignment, rescaling or amplitude correction was applied.

Full optimized iEMG: all 58,755 fibres completed in
325.233 s (5.42 min).
Its saved full-population reference was reused only after git proved the
intramuscular, electrode, utility, muscle and physiological-distribution paths
identical between cceb8c6 and fc077cb4. Fresh selected-MU checks independently
connect the port to the GitLab baseline. Full-template maximum error:
8.88e-15; metadata and final RNG exact.
The older reference timing is not reused to claim a new full-population speedup.

## Recording synthesis

Both paths consume the same fresh corrected selected-MU templates. There are
two real 100-neuron spike pools. The 60-second stress schedule repeats the real
four-second schedule 15 times; it is a numerical duration test, not new physiology.

| Workload | Reference s | Optimized s | Observed ratio |
|---|---:|---:|---:|
| iEMG, 4 s recording | 0.291 | 0.096 | 3.03x |
| iEMG, 60 s recording | 3.569 | 1.163 | 3.07x |
| sEMG, 4 s recording | 1.243 | 0.447 | 2.78x |
| sEMG, 60 s recording | 18.606 | 6.576 | 2.83x |

Repeated four-second calls on the same optimized simulator reuse preparations:

- iEMG: 0.091 s median across 10 additional optimized calls; all compared.
- sEMG: 0.439 s median across 10 additional optimized calls; all compared.

Every final waveform, shape, ordering, unit and sampling annotation was compared.
All occupied bins match exactly for both durations and both full neural pools.
iEMG comparisons cover the raw MUAPs before normalization and assembled output
after the existing population normalization. sEMG comparisons retain absolute mV.
The repeated 60-second schedules contain exactly tied reference maxima. There
were six iEMG and ten sEMG argmax-index differences; every candidate index lands
on a reference maximum with exactly zero amplitude gap. FFT roundoff changes
which repeated cycle wins the tie. No non-tied peak change was accepted. These
ties are reported rather than aligned, rescaled, or hidden.
Selected-cohort assembly timings are not full-100-MU assembly estimates.

## Wider numerical and failure evidence

- 450 fixed real fibres spanning all 100 MUs: fresh pinned GitLab reference
  calculations versus optimized source, complete raw and resampled arrays.
  Maximum errors: 8.33e-17 mV raw and 8.33e-17 mV resampled.
  Worst per-channel error used 1.31e-06 of its allowed bound.
  No peak-sample mismatch; RNG unchanged. All paired raw arrays are preserved locally.
- Exact input capture: all 58,755 sEMG fibre identities, ordering, realized
  numerical arguments and final RNG match. Zero-return capture proves contracts
  only; it is separate from waveform evidence. All 100 iEMG realized MU arrays
  and final RNG also match.
- Complete actual MU 0 (14 fibres) passed raw/summed/resampled checks for bone-free,
  lower fat conductivity, rotated bipolar and Laplacian configurations.
- Real four-MU sEMG allocation failure on the third fibre recovered without
  duplicate accumulation; all 88 completed fibres match reference.
- Persistent reuse restored four completed MUs exactly after seed/RNG restoration.
  A subsequent call without restoration produced a new realization with zero cache hits.
- Corrected-conductor regression gates cover N=512/513/2048, odd/even angular
  grids, bone radii 0/1/10, changed dependencies, DC-only and empty grids.
  Existing closed-form, symmetry, angular-periodicity and thin-bone tests remain passing.
- Contract tests cover concurrent reservations, pressure/zero admission, minimum
  allocation failure, recovery hysteresis, monitor failure, cancellation, bounded
  retries, FFT overlap/cropping, single-thread FFTs, cache corruption/publication,
  input mutation and public import compatibility. Synthetic fixtures are not
  biological or population waveform evidence.

The unchanged per-channel numerical gate is maximum absolute error
<= 1e-12 + 1e-9 * reference peak, with exact identity/RNG/metadata checks.

Final CPU suite: **247 passed, 6 skipped, 1 deselected**;
no failures. GPU and unavailable experimental-data tests account for skips.
Warnings and the complete JUnit output remain in local artifacts.

## Remaining validation

Exhaustive corrected-sEMG full-population comparison, larger-MU reference checks,
full-pipeline timing and broader pressure benchmarks remain pending. Selected
cohorts and synthetic stress tests do not establish deployment-wide performance.

## Reproduce and inspect

Use [usage.md](usage.md) for interpreter, PYTHONPATH, NMODL setup and policy details.
The six preserved approved documents are p0.md through p5.md; plan.md records
the corrected GitLab adaptation. All benchmark scripts are benchmarks/gl_*.py.
Pass --inputs with the trusted existing public fixture directory.
gl_reference.py reads the pinned GitLab source directly and verifies unchanged
helpers. gl_fulli.py checks reference-source ancestry before reuse.
Generated arrays, logs and profiles remain in ignored artifacts/gl; the committed
measurements.json contains numerical summaries without local environment metadata.
Benchmarks refuse to overwrite completed run directories; preserve them.
