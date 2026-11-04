"""Validate a profile card against its frozen limits, or measure a card's MVC mapping.

    python validate_card.py fdi_semg results/fdi_semg.json [--n-jobs 16]
    python validate_card.py --mvc-mapping fdi_semg.published results/fdi_semg.published_mvc.json

Validation runs the card end to end, as a user would: ``card.simulate`` (the calibrated drive, a 25 % MVC trapezoid
with the sEMG benchmark timing: 0.5 s rest, 2 s ramp, 4 s hold, 2 s ramp, 0.5 s rest), ``card.muscle()``,
``card.surface_emg`` with eight Delsys 5-pin sensors (z 3 / 9 mm, rotation 0 / 45 deg, theta 0 / 180 deg; 3 x 3
monopolar grids, 2.5 mm pitch, 0.25 mm pins, as Pereira Botelho et al. 2019 and arm F of
benchmarks/semg_dispersion) at 10240 Hz, then scores cc / adj plateau RMS and median frequency (delsys.py). The
result JSON holds the values, the limits, pass / fail, the card record and the git commit.

10240 Hz is an analysis setting of this comparison, not a recording rate to copy: it lets the scorer use the
20-2000 Hz band of the Delsys comparator, on which the limits were frozen. Generated sEMG is normally sampled at
2048 Hz (the ``SurfaceEMG`` default; iEMG at 10240 Hz).

The MVC mapping runs the card's drive on the MVC ramp of examples/finetune/fdi_spikes_force.py (0.5 s rest,
excitation 0 -> 1 at 0.1 / s, 3 s hold; MVC = mean force over the last 2 s; force smoothed over 0.25 s) with the
card's force model, and reports the MVC and the excitation at which the ramp force first reaches 5, 10, 20, 25, 30
and 50 % MVC: the values of ``drive.mvc_model_units`` and ``drive.excitation_at_pct_mvc``.
"""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import quantities as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import delsys  # noqa: E402

from myogen import load_profile, set_random_seed  # noqa: E402
from myogen.profiles._notes import building_from_card  # noqa: E402
from myogen.simulator import SurfaceElectrodeArray  # noqa: E402
from myogen.utils.neo import signal_to_grid  # noqa: E402
from myogen.utils.nmodl import load_nmodl_mechanisms  # noqa: E402

# Frozen by Devon 2026-11-03, before this script existed: arm F of benchmarks/semg_dispersion re-scored with
# delsys.py; cc / adj plateau RMS within +-10 %, median MDF within +-5 %.
LIMITS_FROZEN = "2026-11-03"
REFERENCE = {"cc": (0.196, 162.5), "adj": (0.233, 157.5)}
RMS_TOL, MDF_TOL = 0.10, 0.05
LIMITED_CARDS = {"fdi_semg"}  # other cards (e.g. the published variant) are run and recorded, not judged

FS__HZ = 10240.0  # for the comparator's 20-2000 Hz band only; generated sEMG is normally 2048 Hz (see above)
RAMP_S, HOLD_S, LEVEL_PCT_MVC = 2.0, 4.0, 25.0
SENSORS = [(z, rot, th) for z in (3.0, 9.0) for rot in (0.0, 45.0) for th in (0.0, 180.0)]

MVC_REST_S, MVC_RAMP_PER_S, MVC_HOLD_S, MVC_LAST_S = 0.5, 0.1, 3.0, 2.0
FORCE_SMOOTH_S = 0.25
MVC_LEVELS = (5, 10, 20, 25, 30, 50)


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def validate(name: str, n_jobs: int) -> dict:
    card = load_profile(name)
    protocol = [(0.5, 0, 0), (RAMP_S, 0, LEVEL_PCT_MVC), (HOLD_S, LEVEL_PCT_MVC, LEVEL_PCT_MVC),
                (RAMP_S, LEVEL_PCT_MVC, 0), (0.5, 0, 0)]
    t0 = time.time()
    run = card.simulate(protocol)
    trains = run.spikes.segments[0].spiketrains
    muscle = card.muscle(n_jobs=n_jobs)
    counts = np.asarray(muscle.resulting_number_of_innervated_fibers)
    mus = [m for m, st in enumerate(trains) if len(st) > 0 and counts[m] > 0]
    bend = (card.value("muscle.radius__mm") + card.value("muscle.fat_thickness__mm")
            + card.value("muscle.skin_thickness__mm")) * pq.mm
    arrays = [SurfaceElectrodeArray(3, 3, 2.5 * pq.mm, 0.25 * pq.mm, center_point__mm_deg=(z * pq.mm, th * pq.deg),
                                    bending_radius__mm=bend, rotation_angle__deg=rot * pq.deg,
                                    differentiation_mode="monopolar") for z, rot, th in SENSORS]
    semg = card.surface_emg(muscle, arrays, sampling_frequency__Hz=FS__HZ * pq.Hz,
                            internal_sampling_frequency__Hz=FS__HZ * pq.Hz, MUs_to_simulate=mus)  # as arm F
    semg.simulate_muaps(n_jobs=n_jobs, verbose=False, use_gpu=False)
    emg = semg.simulate_surface_emg(run.spikes, verbose=False)
    mono = np.stack([np.asarray(signal_to_grid(g.segments[0].analogsignals[0].rescale(pq.mV)), dtype=float)
                     for g in emg.groups], axis=1)
    scores = delsys.score(mono, FS__HZ, RAMP_S, HOLD_S)
    hold = (np.arange(run.force_pct_mvc.size) / run.fs_force__Hz > 0.5 + RAMP_S + 0.5) & \
           (np.arange(run.force_pct_mvc.size) / run.fs_force__Hz < 0.5 + RAMP_S + HOLD_S - 0.5)
    checks = {}
    for kind, (rms, mdf) in scores.items():
        ref_rms, ref_mdf = REFERENCE[kind]
        checks[kind] = {"rms_mV": rms, "mdf_Hz": mdf}
        if name in LIMITED_CARDS:
            checks[kind]["rms_limits_mV"] = [ref_rms * (1 - RMS_TOL), ref_rms * (1 + RMS_TOL)]
            checks[kind]["mdf_limits_Hz"] = [ref_mdf * (1 - MDF_TOL), ref_mdf * (1 + MDF_TOL)]
            checks[kind]["passed"] = bool(
                checks[kind]["rms_limits_mV"][0] <= rms <= checks[kind]["rms_limits_mV"][1]
                and checks[kind]["mdf_limits_Hz"][0] <= mdf <= checks[kind]["mdf_limits_Hz"][1])
    return {"card": card.record(), "date": datetime.date.today().isoformat(), "git_commit": git_commit(),
            "script": "benchmarks/profiles/validate_card.py", "limits_frozen": LIMITS_FROZEN,
            "protocol": protocol, "fs_Hz": FS__HZ, "sensors": SENSORS, "n_units_simulated": len(mus),
            "n_active": int(sum(len(st) > 0 for st in trains)),
            "hold_force_pct_mvc": float(np.mean(run.force_pct_mvc[hold])),
            "scores": checks, "checked": True,
            "passed": all(c["passed"] for c in checks.values()) if name in LIMITED_CARDS else None,
            "runtime_s": round(time.time() - t0, 1)}


def mvc_mapping(name: str) -> dict:
    from myogen.simulator.neuron.descending_drive_network import excitation_to_drive_hz, run_descending_drive

    card = load_profile(name)
    thr = card.thresholds()
    params = card.dd_params()
    dt = float(card.value("drive.timestep_ms"))
    segments = [(MVC_REST_S, 0.0, 0.0), (1.0 / MVC_RAMP_PER_S, 0.0, 1.0), (MVC_HOLD_S, 1.0, 1.0)]
    e, hz = excitation_to_drive_hz(segments, float(card.value("drive.threshold_hz")), params["dd_drive__Hz"],
                                   timestep_ms=dt, noise_seed=int(card.seeds["drive"]))
    set_random_seed(int(card.seeds["drive"]))
    with building_from_card():
        block = run_descending_drive(thr, hz, params, len(e) * dt, timestep_ms=dt,
                                     config_file=card.value("motor_neurons.config_file"))
    fm = card.force_model()
    fs = float(card.value("force.recording_frequency__Hz"))
    force = np.asarray(fm.generate_force(block, verbose=False).magnitude, dtype=float)[:, 0]
    t = np.arange(force.size) / fs
    mvc = float(np.mean(force[t >= t[-1] - MVC_LAST_S]))
    w = max(int(FORCE_SMOOTH_S * fs), 1)
    smooth = np.convolve(force, np.ones(w) / w, mode="same")
    e_t = np.interp(t, np.arange(e.size) * dt / 1000.0, e)
    ramp = (t > MVC_REST_S) & (t < MVC_REST_S + 1.0 / MVC_RAMP_PER_S)
    curve = {p: float(e_t[ramp][np.argmax(smooth[ramp] >= p / 100 * mvc)]) for p in MVC_LEVELS}
    return {"card": card.record(), "date": datetime.date.today().isoformat(), "git_commit": git_commit(),
            "script": "benchmarks/profiles/validate_card.py --mvc-mapping", "mvc_model_units": mvc,
            "excitation_at_pct_mvc": curve}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("card")
    ap.add_argument("out", type=Path)
    ap.add_argument("--mvc-mapping", action="store_true", help="measure the MVC mapping instead of validating")
    ap.add_argument("--n-jobs", type=int, default=16)
    args = ap.parse_args()
    load_nmodl_mechanisms()
    result = mvc_mapping(args.card) if args.mvc_mapping else validate(args.card, args.n_jobs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k in ("scores", "passed", "mvc_model_units",
                                                               "excitation_at_pct_mvc", "hold_force_pct_mvc")},
                     indent=2))


if __name__ == "__main__":
    main()
