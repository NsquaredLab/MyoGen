# Corrected GitLab CPU execution

This branch starts at GitLab main fc077cb4fa89b8cb23c7f0ec4fb50dc8fef05831.
It preserves the shared corrected volume conductor and the removal of the
obsolete surface Cython implementation. See [measurements and limits](report.md).
The six original approved plan documents are preserved as [p0](p0.md),
[p1](p1.md), [p2](p2.md), [p3](p3.md), [p4](p4.md), and [p5](p5.md).
Their historical GitHub base is superseded by [this port plan](plan.md);
their previous benchmark results are not evidence for corrected sEMG.

## Opt in

Calls without an execution policy keep existing execution.

    from myogen.simulator import ExecutionPolicy
    policy = ExecutionPolicy(max_workers=2)
    muaps = emg.simulate_muaps(execution=policy)
    print(dict(emg.execution_report))

Both recording methods accept the same keyword:

    iemg.simulate_intramuscular_emg(spikes, execution=policy)
    semg.simulate_surface_emg(spikes, execution=policy)

All configured fibres, full sample grids, float64 arithmetic, random-draw order,
propagation delays, electrode behavior, and Neo outputs are retained.
MotorUnitSim.calc_sfaps() remains available as a diagnostic.
Source generation limits native numerical pools to one thread and starts at
most two outer workers. Surface generation uses one ordered worker to reuse
preparations across MUs. Recording FFTs explicitly use one worker.

The iEMG path streams fibre batches into the MU accumulator and uses a serial,
strict-arithmetic Numba current-density kernel. First use may compile or load
machine code. The sEMG path reuses exact frequency/source terms, electrode
preparations, fixed-boundary Bessel terms and the corrected conductor output
functional. Fibre-dependent source terms are still evaluated for each fibre.
The ordinary transpose solve A.T w = c gives h = w.T b; it does not change
the corrected matrix, frequency reconstruction, or five-variable reduction.

Recording assembly preserves boolean spike occupancy and NumPy's same
convolution crop, including a template longer than its recording. Direct and
blocked FFT accumulation avoid retaining one entire recording per MU.
Resampled templates and compatible transforms are content-keyed and reused.
iEMG retains its population normalization; sEMG retains absolute mV output.

## Memory and failures

The default reserve is max(4 GiB, 20% of physical RAM), plus an allocation margin
of max(512 MiB, 10% of currently available headroom). Available physical memory
accounts for other applications. A central controller coordinates pending and
resident buffers across all workers in a call. Reservations are promises,
not allocations; spare RAM is not filled.

Set ExecutionPolicy(memory_budget_mib=4096) for an additional managed-buffer
ceiling. This is not a process RSS cap. Required inputs, retained results and
estimated native workspaces count toward admission. Reports expose both managed
accounting and process RSS.

Pressure is polled each second and before admission. It can stop all new jobs,
evict unused caches, shrink subsequent batches and reduce concurrency at safe
boundaries. Growth needs five healthy samples and is limited to 25%. An active
native operation finishes before responding. An impossible minimum allocation,
monitor failure, or 60 seconds without progress raises MemoryBudgetError.
Failed batches retry identical realized inputs before any contribution commits;
partial contributions are never double-added. Failed calls do not install a
new successful result; inspect the exception and execution_report.

## Persistent cache

Disk caching is opt-in: ExecutionPolicy(cache_dir="cache/mu"). Use a short path.
Completed MU arrays use checksummed, versioned NPY/JSON entries and atomic
manifest publication. Runtime cache reads never deserialize pickle. Owned valid
entries are limited to 4 GiB; unrelated files are preserved.

The corrected sEMG namespace is surface-muap-gitlab-fc077-v1, so entries from the
old GitHub-based conductor cannot mask the scientific corrections. Numerical
changes require a namespace/version change. All consumed actual inputs, grids,
conductivities, electrode settings and realized random parameters determine keys.

A new simulate_muaps() call always makes the normal reference draws before
checking the cache. To resume the same realization, restore the saved public
seed AND the generator state from before the original MUAP call. Restoring only
the generator state is insufficient for iEMG clustering sub-seeds.

Do not mutate a simulator or its inputs during a call, or call the same mutable
simulator concurrently. Built-in surface arrays must have matching grid
dimensions. Custom electrode subclasses and use_unified=True use existing
execution. use_gpu=True conflicts with an explicit CPU policy and raises.

## Local setup and examples

Use the project's specified venv interpreter, never bare python. Preserve other
PYTHONPATH entries while prepending this checkout:

    $env:PYTHONPATH = "$PWD;$PWD\.runtime-deps;$env:PYTHONPATH"
    & 'C:\Users\DRRohlf\PycharmProjects\MyoGen\.venv\Scripts\python.exe' -c 'import myogen; myogen._setup_myogen(strict=True); myogen.load_nmodl_mechanisms(strict=True)'

The isolated checkout has its own compiled NMODL mechanisms. psutil and
threadpoolctl are declared dependencies; this laptop's existing shared venv
uses a checkout-local psutil copy in ignored .runtime-deps.
Selected imports are lazy; top-level NEURON initialization remains.

Run examples/01_basic/16_adaptive_cpu_execution.py --help for saved public inputs,
both modalities, repeated recordings and explicit cache restart. The surface
example selects MUs 0-3; --all-mus evaluates every configured MU.

See report.md before citing performance or parity. Four-core affinity on this
63.83-GiB laptop does not establish acceptance on a physical four-core/16-GiB
machine, and selected real fibres do not establish exhaustive population parity.
