Add opt-in adaptive CPU execution for iEMG and sEMG, with bounded memory use, reusable preparations and completed-MU caching. Calls without an execution policy retain existing behavior and the corrected surface-conductor model.

Original implementation and numerical work by **Devon Rohlf**. His authorship is preserved.

- Fix session cleanup so discarded simulators release cached data while live simulators retain reusable caches.
- Preserve numerical precision, random-state behavior and output metadata.
- Include reproducible clean/noisy sEMG examples and compatible API documentation.

Validation covered the automated test suite, optional GPU and data-dependent checks, cache lifetime and pressure recovery. The documentation build passes.

Broader population-level and performance validation remains pending. The sEMG examples are illustrative; their absolute amplitude is not calibrated.
