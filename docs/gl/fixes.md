# CPU review fixes and verification

Branch: perf/basic-gitlab-cpu. Review base: 647aa16d1.
Completed 2026-10-18 on the existing Windows laptop.

## Implemented fixes

- Preserve the surface reference native-period conversion and multiplication
  order before rounding the resampled template length. The interpolation axis
  uses the same period. Supported half-sample cases now preserve the reference
  output lengths (90 and 38 samples), duration and Neo metadata.
- Keep a worker's admitted minimum reserved throughout its computation.
  Allocations consume that promise rather than charging it twice. Child leases
  return borrowed minimum capacity; extra batch capacity is released. Retained
  results remain accounted for after the worker finishes. Fresh physical-memory
  checks still apply when a worker has unused admission credit.
- Serialize disk-cache publication and eviction across instances, threads and
  processes using an OS file lock. A durable ownership journal permits cleanup
  after interruption before or after manifest publication. Recovery preserves
  the current completed-MU payload, unrelated files and files still held by
  Windows readers. Later mutations retry cleanup for those readers.
- Regenerate uv.lock for psutil and the explicit threadpoolctl dependency.
  psutil 7.2.2 was resolved from the local cache; existing package versions were
  retained. The shared Python environment was not modified.

Old unjournalled cache orphans cannot be safely distinguished from unrelated
files and are deliberately preserved. New writes are journalled. Lock files
remain in place so all writers coordinate on the same file. Cache reads retain
checksum validation and atomic completed-MU manifests.

## Regression verification

The original implementation failed both rounding regressions, the concurrent
worker-capacity regression, and the concurrent-publication regression. It also
left extra payloads after injected process exits immediately before and after
manifest publication. The corresponding tests pass with the fixes.

Ten new synthetic contract cases cover those failures plus physical pressure
following a successful preflight, separate-process writers, retained-result
ownership and admission-credit recycling. These fixtures establish runtime and
metadata contracts, not biological behavior.

Fresh broad CPU suite: **257 passed, 6 skipped, 1 deselected, 80 warnings**, in
77.89 seconds. Four physical-core affinity (0/2/4/6), numerical threads=1,
CuPy disabled. The existing agreed exclusion remains:
test_iemg_synthesis_returns_signal_and_reports_tail_ratio. Four GPU tests were
skipped and two tests lacked experimental data. Warnings match the previous
suite categories (deprecations, binning and existing conductor edge arithmetic).

Command through the prescribed interpreter, with this checkout and its
.runtime-deps prepended to the existing PYTHONPATH:

    pytest tests -q -p no:cacheprovider -k "not test_iemg_synthesis_returns_signal_and_reports_tail_ratio" --junitxml=artifacts/gl/fix2-tests.xml

The first suite launch failed in the temporary logging wrapper before running
any tests; its log is retained separately. The corrected launch above passed.
Evidence: artifacts/gl/fix2-tests.txt and artifacts/gl/fix2-tests.xml.

uv lock --check --offline passed after regeneration; it failed as expected before
regeneration. No dependency installation or full build was performed. Independent
follow-up review found no remaining actionable issues in these changes.

## Previously approved real-cohort checks

The saved complete public input population remains 100 MUs / 58,755 fibres.
Only the previously approved complete MUs were evaluated. Reference artifacts
are the saved corrected-GitLab results; no new reference source run was needed.

| Check | iEMG | sEMG |
|---|---:|---:|
| Complete selected MUs | 0, 25, 50 | 0, 1, 2, 3 |
| Evaluated fibres | 470 | 88 |
| Outer source workers | 2 | 1 |
| Raw MUAP maximum absolute error | 3.3306690738754696e-16 | 8.881784197001252e-16 mV |
| Recording maximum absolute error | 1.4432899320127035e-15 | 2.6645352591003757e-15 mV |
| Cache hits after restoring realization | 3 | 4 |
| Changed recording peak samples | 0 | 0 |

RNG state, complete spike-bin arrays, shape, ordering, units and timing metadata
are exact. Cache-loaded MUAP values are bitwise identical to the fresh values.
Complete four-second recordings passed on both first and repeated assembly,
including the existing iEMG population normalization. The per-channel gate is
maximum absolute error <= 1e-12 + 1e-9 * reference peak, without alignment or
rescaling. These are selected-cohort numerical results, not population or
biological validation.

Reproduce with benchmarks/gl_fix.py --inputs <trusted-public-fixture-directory>
--label <new-short-label>. Evidence is in artifacts/gl/fix/result.json and its
per-channel comparison CSV/JSON files. Source and fixture hashes are recorded.
No new performance claim or full-population run accompanies these fixes.
Physical four-core/16-GiB and exhaustive corrected-sEMG gates remain pending.
