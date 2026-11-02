# Corrected GitLab CPU optimization

Base: fc077cb4fa89b8cb23c7f0ec4fb50dc8fef05831 (corrected baseline).

The earlier GitHub-based optimization and draft MR !3 are preserved. Their
sEMG outputs and speedups are not reference evidence for this branch.

## Sequence

1. Record baseline regression status and profile corrected GitLab code before
   porting any implementation. Use CPU only, matched CPU settings and
   one native numerical thread and the unchanged frozen public 100-MU /
   58,755-fibre input population. Generated references are fresh GitLab output.
2. Port independent adaptive-memory, bounded assembly, cache and iEMG changes.
   Keep existing execution as the default. Preserve GitLab's Cython removal.
3. Adapt only measured surface hotspots to the shared corrected
   layered_cylinder_transfer; retain exact absolute-frequency reconstruction,
   integer angular orders, inner-radius Kn scaling and the common authority.
4. Validate exact inputs/RNG/metadata/spike bins and whole-waveform error
   <= 1e-12 + 1e-9 per-channel reference peak, before and after normalization.
   Keep all model samples, fibres and precision. No alignment or amplitude
   correction to force agreement. Small tests are contract evidence only.
5. Start with selected complete public MUs and sampled real fibres, then the
   approved 450-fibre / complete MUs 50 and 75 surface scope if useful.
   Limit reference timings to one unprofiled observation per workload.
   Profiling diagnostics are separate. No multi-day full reference benchmark.
6. Record measured new-baseline timings, memory and validation limits; commit
   independent changes locally. Keep the existing MR and all other worktrees
   unchanged. Deployment-scale acceptance remains separate.

## Environment

Use an isolated project environment with the declared dependencies and compiled
NMODL mechanisms. Benchmark reference and optimized paths under matched CPU and
thread settings. Retain detailed local diagnostics outside the repository.

Baseline CPU tests historically excluded the slow clinical synthesis test and
unavailable optional checks. See validation.md for the broader validation summary.
