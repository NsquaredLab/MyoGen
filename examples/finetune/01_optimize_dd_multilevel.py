#!/usr/bin/env python3
"""
Multi-level DD optimization using trapezoidal ramps.

Optimizes DD network parameters to match Fuglevand recruitment targets
across multiple %MVC levels simultaneously. Uses a single trapezoidal
ramp per trial instead of separate constant-drive simulations.

Usage:
    cd examples/finetune
    python 01_optimize_dd_multilevel.py --n-mu 200 --n-trials 100
    python 01_optimize_dd_multilevel.py --n-mu 150 --n-trials 100
    python 01_optimize_dd_multilevel.py --sweep  # runs all 4 MU counts
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
import warnings

os.environ["MPLBACKEND"] = "Agg"
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import optuna
import quantities as pq
from neo import Block, Segment, SpikeTrain
from neuron import h

# Add examples/basic to path for results directory convention
SCRIPT_DIR = Path(__file__).resolve().parent
BASIC_DIR = SCRIPT_DIR.parent / "basic"
sys.path.insert(0, str(BASIC_DIR))

from myogen import set_random_seed
from myogen.simulator import RecruitmentThresholds
from myogen.simulator.neuron import Network
from myogen.simulator.neuron.populations import AlphaMN__Pool, DescendingDrive__Pool
from myogen.utils.fr_optimization_targets import compute_fr_targets
from myogen.utils.helper import get_gamma_shape_for_mvc
from myogen.utils.nmodl import load_nmodl_mechanisms
from myogen.utils.ramp_analysis import extract_ramp_recruitment

# ─── Constants ────────────────────────────────────────────────
GAMMA_SHAPE = 3.0
SYNAPTIC_WEIGHT = 0.05  # µS
GFLUCTDV_RANGE = (0.05, 0.5)  # Optuna search range for Gfluctdv scale (FDI best was 0.243 at the old 0.25 cap)
ISI_CV_TARGET = (0.15, 0.22)   # Literature target: 15-22% (LIT-17)
ISI_CV_WEIGHT = 2.0            # Penalty weight for ISI CV outside target
TIMESTEP_MS = 0.1
RAMP_MVC_PCT = 30  # Ramp target: 0→30% MVC
REST_S = 3.0
RAMP_S = 4.0
PLATEAU_S = 6.0
CHECKPOINTS = [5, 10, 20, 25, 30]
CHECKPOINT_WEIGHTS = {5: 1.0, 10: 3.0, 20: 2.0, 25: 2.0, 30: 1.0}
# Threshold and target options; set from the command line in main().
THRESHOLD_MODE = "combined"   # RecruitmentThresholds mode for the pool
RECRUITMENT_RANGE = 30.0      # largest / smallest threshold
MAX_RECRUITMENT = 1.0         # excitation at which the last unit is recruited (targets only)
DE_LUCA_PARAMS: dict = {}     # overrides for de_luca_targets
RESULTS_TAG = "polygon"


def _thresholds(profile, n_mu):
    """Pool thresholds (normalised to 1) and the excitation thresholds used for the targets.

    ``profile`` is the muscle key; on this branch thresholds come straight from ``RecruitmentThresholds``
    (the muscle-profile library is not part of main).
    """
    thr, _ = RecruitmentThresholds(N=n_mu, recruitment_range__ratio=RECRUITMENT_RANGE, mode=THRESHOLD_MODE)
    thr = np.asarray(thr, dtype=float)
    thr = thr / thr.max()
    return thr, thr * MAX_RECRUITMENT
# Fuglevand model parameters (back-compat)
MIN_FR_HZ = 8.0
MAX_FR_HZ = 17.0

RESULTS_BASE = BASIC_DIR / "results"


def build_ramp_drive(threshold_hz, max_drive_hz, timestep_ms=TIMESTEP_MS):
    """Build a 0→30% MVC trapezoidal ramp drive signal."""
    peak_hz = threshold_hz + (RAMP_MVC_PCT / 100.0) * (max_drive_hz - threshold_hz)
    rest_hz = max(threshold_hz * 0.8, 0)

    ramp_ms = RAMP_S * 1000
    plat_ms = PLATEAU_S * 1000
    total_ms = (2 * REST_S + 2 * RAMP_S + PLATEAU_S) * 1000
    n_steps = int(total_ms / timestep_ms)

    signal = np.full(n_steps, rest_hz)
    rest1_end = int(REST_S * 1000 / timestep_ms)
    ramp_up_end = rest1_end + int(ramp_ms / timestep_ms)
    plat_end = ramp_up_end + int(plat_ms / timestep_ms)
    ramp_down_end = plat_end + int(ramp_ms / timestep_ms)

    signal[rest1_end:ramp_up_end] = np.linspace(rest_hz, peak_hz, ramp_up_end - rest1_end)
    signal[ramp_up_end:plat_end] = peak_hz
    signal[plat_end:ramp_down_end] = np.linspace(peak_hz, rest_hz, ramp_down_end - plat_end)

    rng = np.random.default_rng(42)
    signal += np.abs(rng.normal(0, peak_hz * 0.01, size=n_steps))

    return signal, total_ms


def detect_threshold(thresholds, dd_params, min_fr=5.0):
    """Binary search for the drive Hz that produces sustained firing."""
    max_drive = dd_params["dd_drive__Hz"]
    lo, hi = 0.05, 0.60
    threshold_hz = hi * max_drive
    for _ in range(8):
        mid = (lo + hi) / 2.0
        drive_hz = mid * max_drive
        block = run_dd_constant(thresholds, drive_hz, dd_params, duration_ms=3000.0)
        has_sustained = any(
            len(st) >= int(min_fr * 3.0)
            for st in block.segments[0].spiketrains
        )
        if has_sustained:
            threshold_hz = drive_hz
            hi = mid
        else:
            lo = mid
    return float(threshold_hz)


def _insert_gfluctdv(mn_pool, scale):
    """Insert Gfluctdv independent noise on all MN dendrites at given scale."""
    if scale <= 0:
        return
    for cell in mn_pool:
        cell.insert_Gfluctdv()
        for d in cell.dend:
            d.std_e_Gfluctdv *= scale
            d.std_i_Gfluctdv *= scale
            d.g_e0_Gfluctdv *= scale
            d.g_i0_Gfluctdv *= scale


def run_dd_constant(thresholds, drive_hz, dd_params, duration_ms=3000.0):
    """Run a constant-drive DD simulation (for threshold detection)."""
    gamma_shape = get_gamma_shape_for_mvc(100, mvc_shape_value=dd_params["gamma_shape"])
    gfluctdv_scale = dd_params.get("gfluctdv_scale", 0.0)
    mn_pool = AlphaMN__Pool(
        recruitment_thresholds__array=thresholds,
        config_file="alpha_mn_default.yaml",
    )
    _insert_gfluctdv(mn_pool, gfluctdv_scale)
    dd_pool = DescendingDrive__Pool(
        n=dd_params["dd_neurons"],
        timestep__ms=TIMESTEP_MS * pq.ms,
        process_type="gamma",
        shape=gamma_shape,
    )
    network = Network({"DD": dd_pool, "aMN": mn_pool})
    network.connect(
        source="DD", target="aMN",
        probability=dd_params["conn_probability"],
        weight__uS=dd_params["synaptic_weight"] * pq.uS,
    )
    network.connect_from_external(
        source="cortical_input", target="DD", weight__uS=1.0 * pq.uS,
    )
    dd_netcons = network.get_netcons("cortical_input", "DD")

    mn_recorders = []
    for cell in mn_pool:
        rec = h.Vector()
        nc = h.NetCon(cell.soma(0.5)._ref_v, None, sec=cell.soma)
        nc.threshold = 50
        nc.record(rec)
        mn_recorders.append(rec)

    h.load_file("stdrun.hoc")
    h.dt = TIMESTEP_MS
    h.tstop = duration_ms

    for s, v in zip(*mn_pool.get_initialization_data()):
        s.v = v
    for s, v in zip(*dd_pool.get_initialization_data()):
        s.v = v
    h.finitialize()

    n_steps = int(duration_ms / TIMESTEP_MS)
    drive = np.full(n_steps, drive_hz) + np.clip(
        np.random.default_rng(42).normal(0, drive_hz * 0.02, size=n_steps), 0, None,
    )

    step = 0
    while h.t < h.tstop:
        d = drive[min(step, n_steps - 1)]
        for dd_cell in dd_pool:
            if dd_cell.integrate(d):
                if h.t < h.tstop:
                    dd_netcons[dd_cell.pool__ID].event(h.t + 1)
        h.fadvance()
        step += 1

    fs = 10240.0
    t_stop_s = duration_ms / 1000.0
    block = Block()
    seg = Segment()
    block.segments.append(seg)
    seg.spiketrains = [
        SpikeTrain(
            rec.as_numpy() / 1000 * pq.s,
            t_stop=t_stop_s * pq.s,
            sampling_rate=fs * pq.Hz,
            sampling_period=(1 / fs) * pq.s,
            name=str(i),
        )
        for i, rec in enumerate(mn_recorders)
    ]
    return block


def run_dd_ramp(thresholds, drive_signal, dd_params, total_ms):
    """Run a DD simulation with time-varying drive (trapezoidal ramp)."""
    gamma_shape = get_gamma_shape_for_mvc(100, mvc_shape_value=dd_params["gamma_shape"])
    gfluctdv_scale = dd_params.get("gfluctdv_scale", 0.0)
    mn_pool = AlphaMN__Pool(
        recruitment_thresholds__array=thresholds,
        config_file="alpha_mn_default.yaml",
    )
    _insert_gfluctdv(mn_pool, gfluctdv_scale)
    dd_pool = DescendingDrive__Pool(
        n=dd_params["dd_neurons"],
        timestep__ms=TIMESTEP_MS * pq.ms,
        process_type="gamma",
        shape=gamma_shape,
    )
    network = Network({"DD": dd_pool, "aMN": mn_pool})
    network.connect(
        source="DD", target="aMN",
        probability=dd_params["conn_probability"],
        weight__uS=dd_params["synaptic_weight"] * pq.uS,
    )
    network.connect_from_external(
        source="cortical_input", target="DD", weight__uS=1.0 * pq.uS,
    )
    dd_netcons = network.get_netcons("cortical_input", "DD")

    mn_recorders = []
    for cell in mn_pool:
        rec = h.Vector()
        nc = h.NetCon(cell.soma(0.5)._ref_v, None, sec=cell.soma)
        nc.threshold = 50
        nc.record(rec)
        mn_recorders.append(rec)

    h.load_file("stdrun.hoc")
    h.dt = TIMESTEP_MS
    h.tstop = total_ms

    for s, v in zip(*mn_pool.get_initialization_data()):
        s.v = v
    for s, v in zip(*dd_pool.get_initialization_data()):
        s.v = v
    h.finitialize()

    n_steps = len(drive_signal)
    step = 0
    while h.t < h.tstop:
        d = drive_signal[min(step, n_steps - 1)]
        for dd_cell in dd_pool:
            if dd_cell.integrate(d):
                if h.t < h.tstop:
                    dd_netcons[dd_cell.pool__ID].event(h.t + 1)
        h.fadvance()
        step += 1

    fs = 10240.0
    t_stop_s = total_ms / 1000.0
    block = Block()
    seg = Segment()
    block.segments.append(seg)
    seg.spiketrains = [
        SpikeTrain(
            rec.as_numpy() / 1000 * pq.s,
            t_stop=t_stop_s * pq.s,
            sampling_rate=fs * pq.Hz,
            sampling_period=(1 / fs) * pq.s,
            name=str(i),
        )
        for i, rec in enumerate(mn_recorders)
    ]
    return block


def make_objective(thresholds, fug_targets):
    """Create the Optuna objective function for a given MU count."""

    def objective(trial):
        try:
            dd_neurons = trial.suggest_int("dd_neurons", 50, 500)
            conn_probability = trial.suggest_float("conn_prob", 0.05, 0.5)
            dd_drive__Hz = trial.suggest_float("dd_drive", 100.0, 1000.0)
            gfluctdv_scale = trial.suggest_float(
                "gfluctdv_scale", *GFLUCTDV_RANGE,
            )

            dd_params = {
                "dd_neurons": dd_neurons,
                "conn_probability": conn_probability,
                "dd_drive__Hz": dd_drive__Hz,
                "synaptic_weight": SYNAPTIC_WEIGHT,
                "gamma_shape": GAMMA_SHAPE,
                "gfluctdv_scale": gfluctdv_scale,
            }

            # Detect threshold for this param set
            threshold_hz = detect_threshold(thresholds, dd_params)

            # Build and run trapezoidal ramp
            drive_signal, total_ms = build_ramp_drive(threshold_hz, dd_drive__Hz)
            block = run_dd_ramp(thresholds, drive_signal, dd_params, total_ms)

            # Extract recruitment curve at checkpoints
            ramp_data = extract_ramp_recruitment(
                block, drive_signal, threshold_hz, dd_drive__Hz,
                checkpoints_pct=CHECKPOINTS,
            )

            # Score against Fuglevand targets
            total_error = 0.0
            for mvc_pct in CHECKPOINTS:
                sim = ramp_data[mvc_pct]
                tgt = fug_targets[mvc_pct]
                w = CHECKPOINT_WEIGHTS.get(mvc_pct, 1.0)

                if tgt["n_active"] == 0:
                    total_error += w * sim["n_active"] * 0.1
                    continue

                n_err = abs(sim["n_active"] - tgt["n_active"]) / max(tgt["n_active"], 1)
                fr_err = (
                    abs(sim["mean_fr_hz"] - tgt["mean_fr_hz"]) / max(tgt["mean_fr_hz"], 1.0)
                    if sim["n_active"] > 0 else 1.0
                )
                total_error += w * (n_err + fr_err)

            # ISI CV measurement at plateau (ARCH-23)
            plat_start_s = (REST_S + RAMP_S)
            plat_end_s = plat_start_s + PLATEAU_S
            isi_cvs = []
            for st in block.segments[0].spiketrains:
                times_s = np.array(st.rescale("s").magnitude)
                plat = times_s[(times_s >= plat_start_s) & (times_s < plat_end_s)]
                if len(plat) < 5:
                    continue
                isis = np.diff(plat) * 1000
                m = float(np.mean(isis))
                if m > 0:
                    isi_cvs.append(float(np.std(isis)) / m)
            mean_cv = float(np.mean(isi_cvs)) if isi_cvs else 0.0

            # Penalize ISI CV outside the 15-22% target range
            cv_lo, cv_hi = ISI_CV_TARGET
            if mean_cv < cv_lo:
                total_error += ISI_CV_WEIGHT * (cv_lo - mean_cv) / cv_lo
            elif mean_cv > cv_hi:
                total_error += ISI_CV_WEIGHT * (mean_cv - cv_hi) / cv_hi

            # Store metadata
            trial.set_user_attr("threshold_hz", threshold_hz)
            trial.set_user_attr("isi_cv_mean", mean_cv)
            trial.set_user_attr("gfluctdv_scale", gfluctdv_scale)
            for mvc_pct in CHECKPOINTS:
                trial.set_user_attr(f"n_active_{mvc_pct}", ramp_data[mvc_pct]["n_active"])
                trial.set_user_attr(f"mean_fr_{mvc_pct}", ramp_data[mvc_pct]["mean_fr_hz"])

            if trial.number % 5 == 0:
                active_str = " | ".join(
                    f"{p}%:{ramp_data[p]['n_active']}/{fug_targets[p]['n_active']}"
                    for p in CHECKPOINTS
                )
                print(f"  Trial {trial.number}: err={total_error:.3f} | {active_str} | "
                      f"drive={dd_drive__Hz:.0f} thr={threshold_hz:.0f} "
                      f"gf={gfluctdv_scale:.3f} cv={mean_cv*100:.1f}%")

            return total_error

        except Exception as e:
            print(f"  Trial {trial.number} failed: {e}")
            return 1000.0

    return objective


def optimize_for_mu_count(n_mu, n_trials, profile, target_model="de_luca",
                          fresh=False,
                          target_muscle="vl",
                          sampler_seed=42,
                          save_json=True):
    """Run the full optimization for one MU count.

    Parameters
    ----------
    n_mu : int
        Motor unit count.
    n_trials : int
        Optuna trial count.
    profile : MuscleProfile
        Profile to generate thresholds from.
    target_model : {"fuglevand", "de_luca", "experimental"}
        Rate coding target model (see fr_optimization_targets).
    target_muscle : str
        Muscle name for experimental lookup (ignored by other models).
    sampler_seed : int
        TPE sampler seed. Use different values per parallel worker.
    save_json : bool
        Whether to save the best-trial JSON. False for parallel workers
        (the coordinator process saves after all workers finish).
    """
    print(f"\n{'='*60}")
    print(f"Optimizing DD for {n_mu} MUs (density=312/mm2, Lexell anchor)")
    print(f"Target model: {target_model}")
    print(f"{'='*60}")

    set_random_seed(42)
    thresholds, target_thresholds = _thresholds(profile, n_mu)

    # Rate coding targets from selected model
    if target_model == "fuglevand":
        fug = compute_fr_targets(
            target_thresholds, mvc_levels=CHECKPOINTS, model="fuglevand",
            min_fr_hz=MIN_FR_HZ, max_fr_hz=MAX_FR_HZ,
        )
    elif target_model == "experimental":
        fug = compute_fr_targets(
            target_thresholds, mvc_levels=CHECKPOINTS, model="experimental",
            muscle_name=target_muscle,
        )
    else:  # de_luca (default)
        fug = compute_fr_targets(
            target_thresholds, mvc_levels=CHECKPOINTS, model="de_luca", **DE_LUCA_PARAMS,
        )
    print(f"Rate coding targets ({target_model}):")
    for mvc_pct in CHECKPOINTS:
        t = fug[mvc_pct]
        print(f"  {mvc_pct}% MVC: {t['n_active']} active, {t['mean_fr_hz']:.1f} Hz")

    # Optuna study
    results_dir = RESULTS_BASE / f"dd_{target_muscle}_{n_mu}mu_{RESULTS_TAG}"
    results_dir.mkdir(exist_ok=True, parents=True)
    db_path = results_dir / "optuna_multilevel.db"
    if fresh and db_path.exists():
        db_path.unlink()
        print(f"  Removed existing {db_path} (--fresh)")
    storage = f"sqlite:///{db_path}"

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=sampler_seed, multivariate=True,
            warn_independent_sampling=False,
        ),
        study_name=f"dd_multilevel_{n_mu}mu",
        storage=storage,
        load_if_exists=True,
    )

    objective_fn = make_objective(thresholds, fug)
    t0 = time.time()
    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=False)
    elapsed = time.time() - t0

    # Best trial
    best = study.best_trial
    print(f"\nBest trial {best.number} (err={best.value:.3f}, {elapsed:.0f}s):")
    print(f"  dd_neurons={best.params['dd_neurons']}, "
          f"conn_prob={best.params['conn_prob']:.3f}, "
          f"dd_drive={best.params['dd_drive']:.0f} Hz, "
          f"gfluctdv={best.params.get('gfluctdv_scale', 0):.3f}")
    print(f"  ISI CV: {best.user_attrs.get('isi_cv_mean', 0)*100:.1f}% "
          f"(target: {ISI_CV_TARGET[0]*100:.0f}-{ISI_CV_TARGET[1]*100:.0f}%)")
    for p in CHECKPOINTS:
        sim_n = best.user_attrs.get(f"n_active_{p}", "?")
        sim_fr = best.user_attrs.get(f"mean_fr_{p}", 0)
        tgt = fug[p]
        print(f"  {p}% MVC: sim={sim_n}/{tgt['n_active']} active, "
              f"FR={float(sim_fr):.1f}/{tgt['mean_fr_hz']:.1f} Hz")

    # Save results (skip for parallel workers — coordinator saves)
    result = {
        "target": {
            "model": target_model,
            "threshold_mode": THRESHOLD_MODE,
            "recruitment_range": RECRUITMENT_RANGE,
            "max_recruitment": MAX_RECRUITMENT,
            "de_luca_params": DE_LUCA_PARAMS,
            "targets": {str(k): v for k, v in fug.items()},
            "min_fr_hz": MIN_FR_HZ,
            "max_fr_hz": MAX_FR_HZ,
        },
        "best_trial": {
            "trial_number": best.number,
            "objective_value": best.value,
            "dd_neurons": best.params["dd_neurons"],
            "conn_probability": best.params["conn_prob"],
            "dd_drive__Hz": best.params["dd_drive"],
            "synaptic_weight": SYNAPTIC_WEIGHT,
            "gamma_shape": GAMMA_SHAPE,
            "gfluctdv_scale": best.params.get("gfluctdv_scale", 0.0),
            "isi_cv_mean": best.user_attrs.get("isi_cv_mean", 0.0),
            "threshold_hz": best.user_attrs.get("threshold_hz"),
            **{f"n_active_{p}": best.user_attrs.get(f"n_active_{p}") for p in CHECKPOINTS},
            **{f"mean_fr_{p}": best.user_attrs.get(f"mean_fr_{p}") for p in CHECKPOINTS},
        },
        "optimization": {
            "n_motor_units": n_mu,
            "n_trials": len(study.trials),
            "ramp_mvc_pct": RAMP_MVC_PCT,
            "checkpoints": CHECKPOINTS,
            "checkpoint_weights": CHECKPOINT_WEIGHTS,
            "elapsed_s": elapsed,
        },
    }
    if save_json:
        out_path = results_dir / "dd_optimized_params.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=float)
        print(f"Saved: {out_path}")

    return result


def _save_best_from_study(study, n_mu, target_muscle, target_model,
                          fug_targets, elapsed):
    """Extract best trial from an Optuna study and save to JSON."""
    best = study.best_trial
    results_dir = RESULTS_BASE / f"dd_{target_muscle}_{n_mu}mu_{RESULTS_TAG}"

    print(f"\nBest trial {best.number} (err={best.value:.3f}, {elapsed:.0f}s):")
    print(f"  dd_neurons={best.params['dd_neurons']}, "
          f"conn_prob={best.params['conn_prob']:.3f}, "
          f"dd_drive={best.params['dd_drive']:.0f} Hz, "
          f"gfluctdv={best.params.get('gfluctdv_scale', 0):.3f}")
    print(f"  ISI CV: {best.user_attrs.get('isi_cv_mean', 0)*100:.1f}% "
          f"(target: {ISI_CV_TARGET[0]*100:.0f}-{ISI_CV_TARGET[1]*100:.0f}%)")
    for p in CHECKPOINTS:
        sim_n = best.user_attrs.get(f"n_active_{p}", "?")
        sim_fr = best.user_attrs.get(f"mean_fr_{p}", 0)
        tgt = fug_targets[p]
        print(f"  {p}% MVC: sim={sim_n}/{tgt['n_active']} active, "
              f"FR={float(sim_fr):.1f}/{tgt['mean_fr_hz']:.1f} Hz")

    result = {
        "target": {
            "model": target_model,
            "threshold_mode": THRESHOLD_MODE,
            "recruitment_range": RECRUITMENT_RANGE,
            "max_recruitment": MAX_RECRUITMENT,
            "de_luca_params": DE_LUCA_PARAMS,
            "targets": {str(k): v for k, v in fug_targets.items()},
            "min_fr_hz": MIN_FR_HZ,
            "max_fr_hz": MAX_FR_HZ,
        },
        "best_trial": {
            "trial_number": best.number,
            "objective_value": best.value,
            "dd_neurons": best.params["dd_neurons"],
            "conn_probability": best.params["conn_prob"],
            "dd_drive__Hz": best.params["dd_drive"],
            "synaptic_weight": SYNAPTIC_WEIGHT,
            "gamma_shape": GAMMA_SHAPE,
            "gfluctdv_scale": best.params.get("gfluctdv_scale", 0.0),
            "isi_cv_mean": best.user_attrs.get("isi_cv_mean", 0.0),
            "threshold_hz": best.user_attrs.get("threshold_hz"),
            **{f"n_active_{p}": best.user_attrs.get(f"n_active_{p}") for p in CHECKPOINTS},
            **{f"mean_fr_{p}": best.user_attrs.get(f"mean_fr_{p}") for p in CHECKPOINTS},
        },
        "optimization": {
            "n_motor_units": n_mu,
            "n_trials": len([t for t in study.trials
                             if t.state == optuna.trial.TrialState.COMPLETE]),
            "ramp_mvc_pct": RAMP_MVC_PCT,
            "checkpoints": CHECKPOINTS,
            "checkpoint_weights": CHECKPOINT_WEIGHTS,
            "elapsed_s": elapsed,
        },
    }
    out_path = results_dir / "dd_optimized_params.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"Saved: {out_path}")
    return result


def run_parallel(args):
    """Launch parallel worker processes against a shared Optuna DB.

    Each worker is a separate Python/NEURON process. Optuna coordinates
    trial assignment through the shared SQLite database. Workers use
    different TPE sampler seeds for diverse exploration.
    """
    from myogen.utils.parallel_budget import estimate_safe_njobs

    n_mu = args.n_mu
    n_trials = args.n_trials
    target_muscle = args.target_muscle
    target_model = args.target_model
    profile = target_muscle

    # Determine worker count
    if args.n_workers:
        n_workers = args.n_workers
    else:
        n_workers = estimate_safe_njobs(
            n_mu=n_mu, dd_neurons=500,  # worst-case from search range
            max_jobs=8,
        )

    trials_per_worker = math.ceil(n_trials / n_workers)

    # Set up results directory and DB
    results_dir = RESULTS_BASE / f"dd_{target_muscle}_{n_mu}mu_{RESULTS_TAG}"
    results_dir.mkdir(exist_ok=True, parents=True)
    db_path = results_dir / "optuna_multilevel.db"
    if args.fresh and db_path.exists():
        db_path.unlink()
        print(f"  Removed existing {db_path} (--fresh)")

    storage = f"sqlite:///{db_path}"
    study_name = f"dd_multilevel_{n_mu}mu"

    # Pre-create study so workers don't race on creation
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )

    # Compute targets for final reporting
    set_random_seed(42)
    thresholds, target_thresholds = _thresholds(profile, n_mu)
    if target_model == "fuglevand":
        fug = compute_fr_targets(target_thresholds, mvc_levels=CHECKPOINTS,
                                 model="fuglevand",
                                 min_fr_hz=MIN_FR_HZ, max_fr_hz=MAX_FR_HZ)
    elif target_model == "experimental":
        fug = compute_fr_targets(target_thresholds, mvc_levels=CHECKPOINTS,
                                 model="experimental",
                                 muscle_name=target_muscle)
    else:
        fug = compute_fr_targets(target_thresholds, mvc_levels=CHECKPOINTS,
                                 model="de_luca", **DE_LUCA_PARAMS)

    print(f"\n{'='*60}")
    print(f"Parallel DD optimization: {target_muscle} {n_mu} MUs")
    print(f"Workers: {n_workers}, trials/worker: {trials_per_worker}")
    print(f"Total trials: ~{n_workers * trials_per_worker} "
          f"(requested {n_trials})")
    print(f"Target model: {target_model}")
    print(f"{'='*60}\n")

    # Spawn workers
    script = str(Path(__file__).resolve())
    python = sys.executable
    log_dir = results_dir / "worker_logs"
    log_dir.mkdir(exist_ok=True)

    procs = []
    log_files = []
    t0 = time.time()
    for w in range(n_workers):
        log_path = log_dir / f"worker_{w}.log"
        cmd = [
            python, script,
            "--n-mu", str(n_mu),
            "--n-trials", str(trials_per_worker),
            "--target-model", target_model,
            "--target-muscle", target_muscle,
            "--threshold-mode", THRESHOLD_MODE,
            "--recruitment-range", str(RECRUITMENT_RANGE),
            "--max-recruitment", str(MAX_RECRUITMENT),
            "--de-luca-params", json.dumps(DE_LUCA_PARAMS),
            "--results-tag", RESULTS_TAG,
            "--_worker-seed", str(42 + w),
        ]
        lf = open(log_path, "w")
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
        procs.append((w, p))
        log_files.append(lf)
        print(f"  Worker {w} started (PID {p.pid}, seed {42 + w})")

    # Wait for all workers with periodic progress reporting
    target_total = n_workers * trials_per_worker
    while True:
        # Check every 30s, but break immediately when all are done
        for _ in range(30):
            time.sleep(1)
            if all(p.poll() is not None for _, p in procs):
                break

        try:
            study = optuna.load_study(study_name=study_name, storage=storage)
            completed = len([t for t in study.trials
                             if t.state == optuna.trial.TrialState.COMPLETE])
            best_val = study.best_value if completed > 0 else float("inf")
            elapsed = time.time() - t0
            rate = completed / elapsed * 3600 if elapsed > 0 else 0
            print(f"  [{elapsed / 60:.0f}m] {completed}/{target_total} trials "
                  f"complete, best err={best_val:.3f}, "
                  f"rate={rate:.0f} trials/hr", flush=True)
        except Exception:
            pass

        if all(p.poll() is not None for _, p in procs):
            break

    # Close log files
    for lf in log_files:
        lf.close()

    elapsed = time.time() - t0
    failures = [(w, p.returncode) for w, p in procs if p.returncode != 0]
    if failures:
        print(f"\n  WARNING: {len(failures)} worker(s) failed:")
        for w, rc in failures:
            print(f"    Worker {w}: exit code {rc} "
                  f"(see {log_dir / f'worker_{w}.log'})")

    # Load final study and save best trial
    study = optuna.load_study(study_name=study_name, storage=storage)
    completed = len([t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"\nOptimization complete: {completed} trials in {elapsed:.0f}s")
    _save_best_from_study(study, n_mu, target_muscle, target_model,
                          fug, elapsed)


def main():
    load_nmodl_mechanisms()
    h.secondorder = 2

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-mu", type=int, default=200,
                    help="MU count to optimize for")
    ap.add_argument("--n-trials", type=int, default=100,
                    help="Optuna trials")
    ap.add_argument("--sweep", action="store_true",
                    help="Sweep all MU counts [150, 200, 250, 300]")
    ap.add_argument("--fresh", action="store_true",
                    help="Delete existing Optuna database and start fresh")
    ap.add_argument("--parallel", action="store_true",
                    help="Run optimization with multiple parallel worker "
                         "processes. Each worker gets its own NEURON instance.")
    ap.add_argument("--n-workers", type=int, default=None,
                    help="Number of parallel workers (default: auto from "
                         "parallel_budget, capped at 8)")
    ap.add_argument("--target-model", type=str, default="de_luca",
                    choices=["fuglevand", "de_luca", "experimental"],
                    help="Rate coding target model. de_luca (default) is "
                         "more realistic than fuglevand linear. experimental "
                         "uses a muscle-specific lookup table.")
    ap.add_argument("--target-muscle", type=str, default="vl",
                    help="Muscle profile key. Used for results directory, "
                         "experimental FR lookup, and profile selection.")
    ap.add_argument("--_worker-seed", type=int, default=None,
                    help=argparse.SUPPRESS)  # internal: per-worker sampler seed
    ap.add_argument("--threshold-mode", default="combined",
                    choices=["fuglevand", "deluca", "konstantin", "combined"])
    ap.add_argument("--recruitment-range", type=float, default=30.0)
    ap.add_argument("--max-recruitment", type=float, default=1.0,
                    help="Excitation fraction at which the last unit is recruited (targets).")
    ap.add_argument("--de-luca-params", type=str, default="{}",
                    help="JSON overrides for de_luca_targets, e.g. fr_onset_hz.")
    ap.add_argument("--results-tag", type=str, default="polygon")
    args = ap.parse_args()
    global THRESHOLD_MODE, RECRUITMENT_RANGE, MAX_RECRUITMENT, DE_LUCA_PARAMS, RESULTS_TAG
    THRESHOLD_MODE, RECRUITMENT_RANGE = args.threshold_mode, args.recruitment_range
    MAX_RECRUITMENT, RESULTS_TAG = args.max_recruitment, args.results_tag
    DE_LUCA_PARAMS = json.loads(args.de_luca_params)

    if args.parallel:
        run_parallel(args)
        return

    profile = args.target_muscle
    is_worker = args._worker_seed is not None

    if args.sweep:
        for n_mu in [150, 200, 250, 300]:
            optimize_for_mu_count(n_mu, args.n_trials, profile,
                                  target_model=args.target_model,
                                  target_muscle=args.target_muscle,
                                  fresh=args.fresh)
    else:
        optimize_for_mu_count(
            args.n_mu, args.n_trials, profile,
            target_model=args.target_model,
            target_muscle=args.target_muscle,
            fresh=args.fresh if not is_worker else False,
            sampler_seed=args._worker_seed if is_worker else 42,
            save_json=not is_worker,
        )


if __name__ == "__main__":
    main()
