"""Reuse the full iEMG reference only after proving its scientific source unchanged."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from gl_common import ROOT, OUT, BASE, setup, dump_json, sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs",type=Path,required=True)
    args=parser.parse_args()
    paths=["myogen/simulator/core/emg/intramuscular",
           "myogen/simulator/core/emg/electrodes.py","myogen/utils",
           "myogen/simulator/core/muscle.py","myogen/simulator/core/physiological_distribution.py"]
    old="cceb8c6346c39af7aed96c6b6d1312ecb6d7a4c1"
    subprocess.run(["git","-c","safe.directory="+ROOT.as_posix(),"diff","--exit-code",
                    old,BASE,"--",*paths],cwd=ROOT,check=True)
    sys.argv=["gl_run.py","ifull","--inputs",str(args.inputs),"--kind","i","--mus","","--optimized"]
    from gl_run import main as run
    run()
    import joblib
    from gl_compare import compare
    comparison=compare(joblib.load(args.inputs/"iemg-reference-0.joblib"),
                       joblib.load(OUT/"ifull/out0.joblib"),destination=OUT/"ifull/comparison")
    result=json.loads((OUT/"ifull/result.json").read_text())
    expected_rng=json.loads((args.inputs/"iemg-reference-0-rng.json").read_text())
    assert expected_rng == result["calls"][0]["rng"]
    result["full_population_comparison"]=dict(waveform=comparison,
        source_equivalence="All listed iEMG source/dependency paths identical between cceb8c6 and fc077cb4",
        identical_paths=paths,reference_sha256=sha(args.inputs/"iemg-reference-0.joblib"),
        reused_reference_only=True,reference_timing_reused=False)
    dump_json(OUT/"ifull/result.json",result)

if __name__=="__main__":
    main()
