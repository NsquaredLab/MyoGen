"""
FDI Spike Trains and Force from an Optimised Descending Drive
=============================================================

Usage: python 07_fdi_spikes_force.py <results_dir> <out_dir> [--n-mu N] [--max-recruitment R]

1. MVC: slow ramp of the drive from rest to 100 % (10 s) and a 3 s hold; force from the
   Fuglevand (1993) [fuglevand1993] twitch model; MVC = mean force over the last 2 s of the hold.
2. The excitation (drive fraction above threshold, as in the optimiser) at which the ramp
   force first reaches 25 % MVC.
3. Benchmark trapezoids with the sEMG harness timing (0.5 s rest, 2 s ramp, 4 s hold, 2 s ramp,
   0.5 s rest): one held at the 25 % MVC excitation, one at 25 % excitation. Each is saved as
   spike times per unit (s) with its force.
Thresholds, drive mapping and the network come from 06_optimize_dd_multilevel.py itself.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import quantities as pq

here = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("opt", here / "06_optimize_dd_multilevel.py")
opt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(opt)

from myogen import set_random_seed  # noqa: E402
from myogen.utils.nmodl import load_nmodl_mechanisms  # noqa: E402

load_nmodl_mechanisms()

ap = argparse.ArgumentParser()
ap.add_argument("results_dir", type=Path)
ap.add_argument("out_dir", type=Path)
ap.add_argument("--n-mu", type=int, default=120)
ap.add_argument("--threshold-mode", default=None, help="override the results file (older files lack it)")
ap.add_argument("--recruitment-range", type=float, default=None)
ap.add_argument("--max-recruitment", type=float, default=None)
ap.add_argument("--extra-trap", action="append", default=[], metavar="NAME:LEVEL",
                help="additional trapezoid at a fixed excitation level (e.g. from an external force model)")
args = ap.parse_args()
args.out_dir.mkdir(parents=True, exist_ok=True)

res = json.load(open(args.results_dir / "dd_optimized_params.json"))
tgt, best = res["target"], res["best_trial"]
opt.THRESHOLD_MODE = args.threshold_mode or tgt.get("threshold_mode", "combined")
opt.RECRUITMENT_RANGE = args.recruitment_range or float(tgt.get("recruitment_range", 30.0))
opt.MAX_RECRUITMENT = args.max_recruitment or float(tgt.get("max_recruitment", 1.0))
opt.FORCE_FREQUENCY_SHAPE = tuple(tgt.get("force_frequency_shape", opt.FORCE_FREQUENCY_SHAPE))  # as calibrated
opt.CEILING_RANK = int(tgt.get("ceiling_rank", 0))
set_random_seed(42)
thresholds, _ = opt._thresholds("fdi", args.n_mu)
dd = {k: best[k] for k in ("dd_neurons", "conn_probability", "dd_drive__Hz", "synaptic_weight", "gamma_shape",
                           "gfluctdv_scale")}
dd["network"] = best.get("network")  # absent in pre-2026-11-01 files: run_dd_ramp refuses them
thr_hz = float(best["threshold_hz"])
dt = opt.TIMESTEP_MS
FS_FORCE = opt.FORCE_FS_HZ


def simulate(segments):
    """Drive, spikes and force as in the optimiser (``opt.simulate_force``)."""
    return opt.simulate_force(thresholds, dd, thr_hz, segments)


def save(name, e, block, force, mvc):
    trains = [np.asarray(st.rescale(pq.s).magnitude) for st in block.segments[0].spiketrains]
    np.savez(args.out_dir / f"{name}.npz", spikes=np.array(trains, dtype=object), force=force / mvc,
             excitation=e[:: int(round(1000 / dt / FS_FORCE)) or 1], fs_force=FS_FORCE, thresholds=thresholds,
             dd=json.dumps(dd), threshold_hz=thr_hz, results_dir=str(args.results_dir))


# 1. MVC and the force-excitation curve (the optimiser's MVC ramp, ``opt.run_mvc_ramp``)
r = opt.run_mvc_ramp(thresholds, dd, thr_hz)
e, block, f, mvc = r.e, r.block, r.force, r.mvc
t = np.arange(f.size) / FS_FORCE
e_t = np.interp(t, np.arange(e.size) * dt / 1000, e)
ramp = (t > opt.MVC_REST_S) & (t < opt.MVC_REST_S + 1.0 / opt.CEILING_RAMP_PER_S)
smooth = opt.smoothed_force(f, FS_FORCE)
curve = {int(p): float(e_t[ramp][np.argmax(smooth[ramp] >= p / 100 * mvc)]) for p in (5, 10, 20, 25, 30, 50)}
save("mvc_ramp", e, block, f, mvc)
print("MVC force (model units):", mvc)
print("excitation at % MVC:", curve)
print(f"force-frequency shape {opt.FORCE_FREQUENCY_SHAPE}; last unit recruited at {r.pct_last:.1f} % MVC, "
      f"unit with {opt.CEILING_RANK} above at {r.pct_ranked:.1f} %, unrecruited {r.n_unrecruited}; rates {r.rates}",
      flush=True)

# 2. Benchmark trapezoids
summary = {"mvc": mvc, "excitation_at_pct_mvc": curve, "pct_mvc_last_recruited": r.pct_last,
           "n_unrecruited": r.n_unrecruited, "pct_mvc_ceiling_ranked": r.pct_ranked, "ceiling_rank": opt.CEILING_RANK,
           "force_frequency_shape": list(opt.FORCE_FREQUENCY_SHAPE), "rates_at_pct_mvc": r.rates}


def trapezoid(name, level):
    e, block, f = simulate([(0.5, 0.0, 0.0), (2.0, 0.0, level), (4.0, level, level), (2.0, level, 0.0), (0.5, 0.0, 0.0)])
    t = np.arange(f.size) / FS_FORCE
    hold = (t > 3.0) & (t < 6.0)
    rates = [np.sum((st.magnitude > 3.0) & (st.magnitude < 6.0)) / 3.0 for st in block.segments[0].spiketrains]
    active = [r for r in rates if r >= 1.0]
    summary[name] = {"excitation": level, "hold_force_pct_mvc": float(100 * np.mean(f[hold]) / mvc),
                     "n_active": len(active), "mean_rate_hz": float(np.mean(active)) if active else 0.0}
    save(name, e, block, f, mvc)
    print(name, summary[name], flush=True)
    return summary[name]["hold_force_pct_mvc"]


f_exc = trapezoid("trap_25exc", 0.25)
f_mvc = trapezoid("trap_25mvc", curve[25])
if abs(f_mvc - 25.0) > 1.0 and f_mvc != f_exc:
    # One secant step on the hold force (the ramp estimate lags the isometric hold).
    trapezoid("trap_25mvc", curve[25] + (25.0 - f_mvc) * (curve[25] - 0.25) / (f_mvc - f_exc))
for spec in args.extra_trap:
    name, level = spec.split(":")
    trapezoid(name, float(level))
json.dump(summary, open(args.out_dir / "summary.json", "w"), indent=1)
