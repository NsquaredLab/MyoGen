# Corrected GitLab CPU optimization

Base: fc077cb4fa89b8cb23c7f0ec4fb50dc8fef05831 (gitlab/main, checked 2026-10-18).
Branch: perf/basic-gitlab-cpu; short managed worktree: bgl/MyoGen.

The earlier GitHub-based optimization and draft MR !3 are preserved. Their
sEMG outputs and speedups are not reference evidence for this branch.

## Sequence

1. Record baseline regression status and profile corrected GitLab code before
   porting any implementation. Use CPU only, four physical-core affinity,
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
   unchanged. Physical four-core/16-GiB acceptance remains separate.

## Environment

Use the specified main-project venv interpreter; prepend this checkout and its
local .runtime-deps to existing PYTHONPATH. Shared installation is unchanged.
Local NMODL mechanisms compiled successfully. psutil is copied from the prior
isolated basic worktree into this worktree only. Resource detection reports
14 physical cores, 63.83 GiB RAM and approximately 34 GiB available; GPU use is
disabled for benchmarks. Other agents' Python processes are left untouched.

Baseline CPU pytest excludes only the unchanged slow clinical iEMG synthesis
test for this limited pass. Any skips, failures and warnings are retained.
