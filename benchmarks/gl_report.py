"""Collect benchmark evidence into local, ignored reports for private review."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from gl_common import ROOT, OUT, BASE, dump_json, sha


def read(name):
    return json.loads((OUT/name/"result.json").read_text(encoding="utf-8"))


def main():
    sr,so,ir,io=[read(k) for k in ("sr","so","ir","io")]
    spot,full,record,contracts,variants=[read(k) for k in ("spot","ifull","record","contracts","variants")]
    assert all(x["status"] in ("complete","passed") for x in (sr,so,ir,io,spot,full,record,contracts,variants))
    tree=ET.parse(OUT/"final.xml")
    suites=tree.getroot().findall("testsuite")
    tests=sum(int(x.get("tests","0")) for x in suites)
    skipped=sum(int(x.get("skipped","0")) for x in suites)
    failures=sum(int(x.get("failures","0"))+int(x.get("errors","0")) for x in suites)
    assert failures==0
    rows=[]
    for kind,ref,opt in (("iEMG, MUs 0/25/50; 470 fibres",ir,io),("sEMG, MUs 0-3; 88 fibres",sr,so)):
        a,b=ref["calls"][0]["seconds"],opt["calls"][0]["seconds"]
        rows.append(f"| {kind} | {a:.3f} | {b:.3f} | {a/b:.2f}x |")
    assembly=[]
    for item in record["results"]:
        comparison=item["comparison"]
        assert comparison["timing_mismatches"] == comparison["exact_reference_peak_ties"]
    for r in record["results"]:
        a,b=r["reference"]["seconds"],r["optimized"]["seconds"]
        assembly.append(f"| {r['kind']}EMG, {r['duration']} s recording | {a:.3f} | {b:.3f} | {a/b:.2f}x |")
    repeats=[]
    for r in record["results"]:
        if "repeated" in r:
            values=sorted(x["seconds"] for x in r["repeated"])
            median=(values[4]+values[5])/2
            repeats.append(f"- {r['kind']}EMG: {median:.3f} s median across 10 additional optimized calls; all compared.")
    raw=max(r["raw"]["max_abs"] for r in spot["results"])
    resampled=max(r["resampled"]["max_abs"] for r in spot["results"])
    ratio=max(r["resampled"]["max_gate_fraction"] for r in spot["results"])
    fullcmp=full["full_population_comparison"]["waveform"]
    facts=dict(base=BASE,source_cohorts={k:read(k) for k in ("sr","so","ir","io")},
               spot=dict(count=450,mus=100,max_raw_error=raw,max_resampled_error=resampled,
                         max_gate_fraction=ratio,
                         peak_sample_mismatches=sum(x["raw"]["peak_sample_mismatches"]+
                            x["resampled"]["peak_sample_mismatches"] for x in spot["results"])),
               full_iemg=full,recordings=record,contracts=contracts,variants=variants,
               tests=dict(passed=tests-skipped,skipped=skipped,failures=failures),
               artifacts={str(p.relative_to(ROOT)):sha(p) for p in sorted(OUT.rglob("result.json"))},
               test_log_sha256=sha(OUT/"final-tests.txt"))
    dump_json(OUT/"measurements.json",facts)
    text=f"""# Corrected GitLab CPU port: results

Branch: perf/basic-gitlab-cpu. Base: {BASE}.
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
{chr(10).join(rows)}

Fresh corrected-GitLab high-level sEMG output is the reference. Final RNG,
shape/order, units, sampling metadata and peak-sample locations match.
Maximum complete-MU errors were {io['calls'][0]['comparison']['maximum_error']:.3g}
for raw iEMG templates and {so['calls'][0]['comparison']['maximum_error']:.3g} mV
for sEMG templates. No alignment, rescaling or amplitude correction was applied.

Full optimized iEMG: all 58,755 fibres completed in
{full['calls'][0]['seconds']:.3f} s ({full['calls'][0]['seconds']/60:.2f} min).
Its saved full-population reference was reused only after git proved the
intramuscular, electrode, utility, muscle and physiological-distribution paths
identical between cceb8c6 and fc077cb4. Fresh selected-MU checks independently
connect the port to the GitLab baseline. Full-template maximum error:
{fullcmp['maximum_error']:.3g}; metadata and final RNG exact.
The older reference timing is not reused to claim a new full-population speedup.

## Recording synthesis

Both paths consume the same fresh corrected selected-MU templates. There are
two real 100-neuron spike pools. The 60-second stress schedule repeats the real
four-second schedule 15 times; it is a numerical duration test, not new physiology.

| Workload | Reference s | Optimized s | Observed ratio |
|---|---:|---:|---:|
{chr(10).join(assembly)}

Repeated four-second calls on the same optimized simulator reuse preparations:

{chr(10).join(repeats)}

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
  Maximum errors: {raw:.3g} mV raw and {resampled:.3g} mV resampled.
  Worst per-channel error used {ratio:.3g} of its allowed bound.
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

Final CPU suite: **{tests-skipped} passed, {skipped} skipped, 1 deselected**;
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
"""
    (OUT/"report.md").write_text(text,encoding="utf-8")
    print("Saved local reports under artifacts/gl; review before publishing summaries")

if __name__=="__main__":
    main()
