#!/usr/bin/env python3
"""
Multilevel Descending-Drive Optimisation on a Trapezoidal Ramp
==============================================================

Optimizes DD network parameters to match Fuglevand [fuglevand1993] recruitment targets
across multiple %MVC levels simultaneously. Uses a single trapezoidal
ramp per trial instead of separate constant-drive simulations.

Usage:
    cd examples/02_finetune
    python 06_optimize_dd_multilevel.py --n-mu 200 --n-trials 100
    python 06_optimize_dd_multilevel.py --n-mu 150 --n-trials 100
    python 06_optimize_dd_multilevel.py --sweep  # runs all 4 MU counts
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

os.environ["MPLBACKEND"] = "Agg"

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import optuna
import quantities as pq
from neo import Block, Segment, SpikeTrain
from neuron import h

SCRIPT_DIR = Path(__file__).resolve().parent

from myogen import set_random_seed
from myogen.simulator import RecruitmentThresholds
from myogen.simulator.core.force.force_model import ForceModel
from myogen.simulator.core.force.force_utils import FUGLEVAND_FORCE_FREQUENCY, HUMAN_UNIT_FORCE_FREQUENCY
from myogen.simulator.neuron import Network
from myogen.simulator.neuron.populations import AlphaMN__Pool, DescendingDrive__Pool
from myogen.utils.fr_optimization_targets import compute_fr_targets
from myogen.utils.helper import get_gamma_shape_for_mvc
from myogen.utils.nmodl import load_nmodl_mechanisms
from myogen.utils.ramp_analysis import extract_ramp_recruitment

# ─── Constants ────────────────────────────────────────────────
GAMMA_SHAPE = 3.0
SYNAPTIC_WEIGHT = 0.05  # µS, DD input weight on the smallest (best-coupled) motor neuron
NETWORK = {"fan_in": "equal", "weight_scaling": "dendritic_transfer"}  # DD -> MN wiring, see build_dd_mn_network
WEIGHT_EXPONENT_RANGE = (0.0, 1.0)  # fitted range of the DD weight exponent (--fit-weight-exponent)
FIT_WEIGHT_EXPONENT = False
CEILING_WEIGHT = 3.0       # Penalty weight for the recruitment ceiling (force at the last recruitment vs LAST_RECRUIT_PCT_MVC)
CEILING_RAMP_PER_S = 0.1   # Excitation per second in the MVC ramp (also used by 07_fdi_spikes_force.py)
MVC_REST_S, MVC_HOLD_S = 0.5, 3.0  # MVC ramp: rest, ramp 0 -> 1 at CEILING_RAMP_PER_S, hold at full excitation
FORCE_FS_HZ = 2048.0       # ForceModel sampling rate
FORCE_SMOOTH_S = 0.25      # Moving-average window on the ramp force
GFLUCTDV_RANGE = (0.05, 0.5)  # Optuna search range for Gfluctdv scale (FDI best was 0.243 at the old 0.25 cap)
ISI_CV_TARGET = (0.15, 0.22)   # Literature target: 15-22% at 30 % MVC [enokaduchateau2017] [dideriksen2012] (as synthesised in nexus LIT-17; range not checked against the papers)
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
MAX_RECRUITMENT = 1.0         # excitation at which the last unit is recruited (rate-coding targets only)
LAST_RECRUIT_PCT_MVC: float | None = None  # force (% MVC) at which the last unit is recruited; None: no ceiling term
CEILING_RANK = 0              # score the unit with CEILING_RANK units recruited above it (0: the last unit)
FORCE_FREQUENCY_SHAPE = FUGLEVAND_FORCE_FREQUENCY  # (a, c) of the ForceModel force-frequency saturation
FORCE_FREQUENCY_SHAPES = {"fuglevand": FUGLEVAND_FORCE_FREQUENCY, "human": HUMAN_UNIT_FORCE_FREQUENCY}
# Mean FDI peak firing rates (Hz) in 40 and 80 % MVC contractions [deluca1982 p. 118 Table 1].
HIGH_FORCE_RATE_TARGETS = {40: 25.3, 80: 41.4}
HIGH_FORCE_RATE_WEIGHT = 1.0  # per level, relative rate error; scored on the MVC ramp
SCORE_HIGH_FORCE_RATES = False
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
# Fuglevand model parameters [fuglevand1993] (back-compat)
MIN_FR_HZ = 8.0
MAX_FR_HZ = 17.0

RESULTS_BASE = SCRIPT_DIR.parent.parent / "results"  # same root as the other 02_finetune examples


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


def dendritic_transfer(mn_pool):
    """Fraction of a steady current injected at each MN's DD synapse site that reaches its clamped soma.

    DC transfer impedance (synapse site -> soma) over soma input impedance, from ``h.Impedance`` linearised at
    the initial state (active channels included). Call before ``_insert_gfluctdv``.
    """
    for s, v in zip(*mn_pool.get_initialization_data()):
        s.v = v
    h.finitialize()
    k = []
    for cell in mn_pool:
        syn = next(s for s in cell.synapse__list if s.e >= -40)
        seg = syn.get_segment()
        imp = h.Impedance()
        imp.loc(0.5, sec=cell.soma)
        imp.compute(0, 1)
        k.append(imp.transfer(seg.x, sec=seg.sec) / imp.input(0.5, sec=cell.soma))
    return np.array(k)


def build_dd_mn_network(mn_pool, dd_pool, dd_params):
    """Wire the DD pool to the motor neurons and the cortical input to the DD pool.

    ``dd_params["network"]`` must be ``NETWORK``; parameter files without it (calibrated with random fan-in and
    one weight for all neurons) are refused rather than reinterpreted.

    - Equal fan-in: every MN receives exactly round(conn_probability * dd_neurons) DD inputs, drawn without
      replacement. Modelling assumption (descending drive shared across the pool); it removes the binomial
      +/-13 % input-count scatter that reordered the largest units, whose thresholds are ~3 % apart.
    - Dendritic-transfer weights: weight_i = synaptic_weight * (k_max / k_i) ** p (``dendritic_transfer``), p from
      ``network["weight_exponent"]`` (default 1). With p = 1 each input delivers the same current to a clamped soma
      at low drive whatever the neuron's size, and recruitment order follows rheobase: equal effective synaptic
      current per input, the equal-share case of Heckman & Binder 1991 (J Neurophysiol 65:952) [heckmanbinder1991]. p < 1 gives larger
      neurons less effective current per input (p = 0: one weight for all). Modelling assumption either way;
      measured effective currents are input-system specific (cat data) and none are known for corticospinal
      input to a hand muscle. Not compensated: the Gfluctdv mean-conductance shunt and the driving-force
      saturation at high drive.
    """
    net_cfg = dict(dd_params.get("network") or {})
    p = net_cfg.pop("weight_exponent", 1.0)
    if not isinstance(p, (int, float)) or isinstance(p, bool) or not np.isfinite(p) or p < 0:
        raise ValueError(f"network weight_exponent must be a finite number >= 0, got {p!r}")
    if net_cfg != NETWORK:
        raise ValueError(
            f"dd_params['network'] must be {NETWORK}, got {net_cfg!r}; parameter files from before 2026-11-01 "
            "were calibrated with random fan-in and uniform weights and must be recalibrated."
        )
    n_dd = len(dd_pool)
    n_in = int(round(dd_params["conn_probability"] * n_dd))
    if n_in < 1:
        raise ValueError(f"conn_probability * dd_neurons rounds to {n_in} inputs per motor neuron")
    transfer = dendritic_transfer(mn_pool)
    weights = dd_params["synaptic_weight"] * (transfer.max() / transfer) ** p
    network = Network({"DD": dd_pool, "aMN": mn_pool})
    dd_to_mn = network.connect_fan_in("DD", "aMN", n_inputs=n_in, weights__uS=weights * pq.uS,
                                      delay__ms=1.0 * pq.ms)
    network.connect_from_external(source="cortical_input", target="DD", weight__uS=1.0 * pq.uS)
    return SimpleNamespace(network=network, dd_to_mn=dd_to_mn,
                           cortical_netcons=network.get_netcons("cortical_input", "DD"),
                           transfer=transfer, weights=weights)


def network_block(weight_exponent=None):
    """The ``network`` block of a parameter set: NETWORK, plus the DD weight exponent when it was fitted."""
    return dict(NETWORK) if weight_exponent is None else {**NETWORK, "weight_exponent": float(weight_exponent)}


def recruitment_times(block, window_ms=500.0, min_spikes=3):
    """Recruitment time (s) of each unit: the first spike of the first ``min_spikes`` spikes within ``window_ms``
    (sustained firing, as in ``extract_ramp_recruitment``). NaN for units never recruited."""
    out = []
    for st in block.segments[0].spiketrains:
        t = np.asarray(st.rescale("s").magnitude)
        hits = np.nonzero(t[min_spikes - 1:] - t[:len(t) - min_spikes + 1] < window_ms / 1000.0)[0]
        out.append(float(t[hits[0]]) if len(hits) else np.nan)
    return np.array(out)


def recruitment_excitation(block, excitation, timestep_ms=TIMESTEP_MS, window_ms=500.0, min_spikes=3):
    """Excitation at each unit's recruitment (see ``recruitment_times``). NaN for units never recruited."""
    t_rec = recruitment_times(block, window_ms, min_spikes)
    idx = np.clip(np.round(np.nan_to_num(t_rec) * 1000.0 / timestep_ms).astype(int), 0, len(excitation) - 1)
    return np.where(np.isnan(t_rec), np.nan, np.asarray(excitation, dtype=float)[idx])


def smoothed_force(force, fs):
    w = max(int(FORCE_SMOOTH_S * fs), 1)
    return np.convolve(force, np.ones(w) / w, mode="same")


def last_recruitment_pct_mvc(t_rec, force, fs, mvc, rank=0):
    """Force (% MVC, smoothed) at the recruitment of the unit with ``rank`` units recruited after it (0: the last
    unit), and the number of units never recruited. Unrecruited units count as recruited after it; more of them than
    ``rank`` push the value past 100 % (100 * (1 + (n_unrecruited - rank) / n))."""
    t_rec = np.asarray(t_rec, dtype=float)
    n_unrec = int(np.isnan(t_rec).sum())
    if n_unrec > rank:
        return 100.0 * (1.0 + (n_unrec - rank) / len(t_rec)), n_unrec
    t = np.sort(t_rec[~np.isnan(t_rec)])[::-1][rank - n_unrec]
    i = min(int(round(t * fs)), len(force) - 1)
    return float(100.0 * smoothed_force(force, fs)[i] / mvc), n_unrec


def rates_at_pct_mvc(trains, force, fs, mvc, levels=tuple(HIGH_FORCE_RATE_TARGETS), window_s=1.0, min_spikes=3):
    """Mean firing rate (Hz) of the active units (>= min_spikes in window_s) centred where the smoothed force first
    reaches each level (% MVC). ``trains`` are spike times in s. NaN for a level never reached."""
    sm = smoothed_force(force, fs)
    out = {}
    for lev in levels:
        hit = np.nonzero(sm >= lev / 100.0 * mvc)[0]
        if not len(hit):
            out[lev] = float("nan")
            continue
        t0 = hit[0] / fs
        counts = [np.sum((np.asarray(st) > t0 - window_s / 2) & (np.asarray(st) <= t0 + window_s / 2)) for st in trains]
        rates = [c / window_s for c in counts if c >= min_spikes]
        out[lev] = float(np.mean(rates)) if rates else 0.0
    return out


def high_force_rate_penalty(rates):
    """HIGH_FORCE_RATE_WEIGHT * relative rate error at each HIGH_FORCE_RATE_TARGETS level (1 per missing level)."""
    err = 0.0
    for lev, tgt in HIGH_FORCE_RATE_TARGETS.items():
        r = rates.get(lev, float("nan"))
        err += 1.0 if not np.isfinite(r) else abs(r - tgt) / tgt
    return HIGH_FORCE_RATE_WEIGHT * err


def ceiling_penalty(pct_last, target_pct=None):
    """CEILING_WEIGHT * relative miss of the last recruitment force; 0 when no target is set."""
    if target_pct is None:
        return 0.0
    return CEILING_WEIGHT * abs(pct_last - target_pct) / target_pct


def mvc_drive(segments, threshold_hz, max_hz):
    """Piecewise-linear excitation [(duration_s, start, end)] and its DD rate (Hz): threshold_hz + e (max - threshold),
    0.8 threshold_hz at rest, plus 1 % half-normal noise."""
    parts = [np.linspace(a, b, int(round(d * 1000 / TIMESTEP_MS)), endpoint=False) for d, a, b in segments]
    e = np.concatenate(parts)
    hz = np.where(e > 0, threshold_hz + e * (max_hz - threshold_hz), 0.8 * threshold_hz)
    hz = hz + np.abs(np.random.default_rng(42).normal(0, hz.max() * 0.01, size=hz.size))
    return e, hz


def make_force_model(thresholds):
    """ForceModel (Fuglevand 1993 [fuglevand1993] twitches) at FORCE_FS_HZ with the FORCE_FREQUENCY_SHAPE saturation."""
    return ForceModel(recruitment_thresholds=thresholds, recording_frequency__Hz=FORCE_FS_HZ * pq.Hz,
                      force_frequency_shape=FORCE_FREQUENCY_SHAPE)


def simulate_force(thresholds, dd_params, threshold_hz, segments):
    """Run the drive of ``mvc_drive``; return (excitation, block, force at FORCE_FS_HZ) (see ``make_force_model``)."""
    e, hz = mvc_drive(segments, threshold_hz, dd_params["dd_drive__Hz"])
    block = run_dd_ramp(thresholds, hz, dd_params, len(e) * TIMESTEP_MS)
    return e, block, np.asarray(make_force_model(thresholds).generate_force(block).magnitude)[:, 0]


def run_mvc_ramp(thresholds, dd_params, threshold_hz):
    """MVC ramp: MVC_REST_S rest, excitation 0 -> 1 at CEILING_RAMP_PER_S, MVC_HOLD_S hold at 1.
    MVC is the mean force over the last 2 s of the hold. Returns a namespace with e, block, force, mvc,
    t_rec (recruitment times, s), pct_last / n_unrecruited (see ``last_recruitment_pct_mvc``), pct_ranked
    (the CEILING_RANK unit) and rates (``rates_at_pct_mvc`` at the HIGH_FORCE_RATE_TARGETS levels)."""
    ramp_s = 1.0 / CEILING_RAMP_PER_S
    e, block, f = simulate_force(thresholds, dd_params, threshold_hz,
                                 [(MVC_REST_S, 0.0, 0.0), (ramp_s, 0.0, 1.0), (MVC_HOLD_S, 1.0, 1.0)])
    t = np.arange(f.size) / FORCE_FS_HZ
    t_end = MVC_REST_S + ramp_s + MVC_HOLD_S
    mvc = float(np.mean(f[(t > t_end - 2.0) & (t < t_end)]))
    t_rec = recruitment_times(block)
    pct_last, n_unrec = last_recruitment_pct_mvc(t_rec, f, FORCE_FS_HZ, mvc)
    pct_ranked, _ = last_recruitment_pct_mvc(t_rec, f, FORCE_FS_HZ, mvc, rank=CEILING_RANK)
    trains = [np.asarray(st.rescale("s").magnitude) for st in block.segments[0].spiketrains]
    rates = rates_at_pct_mvc(trains, f, FORCE_FS_HZ, mvc)
    return SimpleNamespace(e=e, block=block, force=f, mvc=mvc, t_rec=t_rec, pct_last=pct_last,
                           n_unrecruited=n_unrec, pct_ranked=pct_ranked, rates=rates)


def run_dd_constant(thresholds, drive_hz, dd_params, duration_ms=3000.0):
    """Run a constant-drive DD simulation (for threshold detection)."""
    gamma_shape = get_gamma_shape_for_mvc(100, mvc_shape_value=dd_params["gamma_shape"])
    gfluctdv_scale = dd_params.get("gfluctdv_scale", 0.0)
    mn_pool = AlphaMN__Pool(
        recruitment_thresholds__array=thresholds,
        config_file="alpha_mn_default.yaml",
    )
    dd_pool = DescendingDrive__Pool(
        n=dd_params["dd_neurons"],
        timestep__ms=TIMESTEP_MS * pq.ms,
        process_type="gamma",
        shape=gamma_shape,
    )
    wiring = build_dd_mn_network(mn_pool, dd_pool, dd_params)
    _insert_gfluctdv(mn_pool, gfluctdv_scale)
    dd_netcons = wiring.cortical_netcons

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
    dd_pool = DescendingDrive__Pool(
        n=dd_params["dd_neurons"],
        timestep__ms=TIMESTEP_MS * pq.ms,
        process_type="gamma",
        shape=gamma_shape,
    )
    wiring = build_dd_mn_network(mn_pool, dd_pool, dd_params)
    _insert_gfluctdv(mn_pool, gfluctdv_scale)
    dd_netcons = wiring.cortical_netcons

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
            weight_exponent = (trial.suggest_float("weight_exponent", *WEIGHT_EXPONENT_RANGE)
                               if FIT_WEIGHT_EXPONENT else None)

            dd_params = {
                "dd_neurons": dd_neurons,
                "conn_probability": conn_probability,
                "dd_drive__Hz": dd_drive__Hz,
                "synaptic_weight": SYNAPTIC_WEIGHT,
                "gamma_shape": GAMMA_SHAPE,
                "gfluctdv_scale": gfluctdv_scale,
                "network": network_block(weight_exponent),
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

            # Recruitment ceiling: the ramp above stops at RAMP_MVC_PCT, so the last unit is found on a separate
            # MVC ramp, scored by the force (% MVC) at its recruitment. Units never recruited push it past 100 %.
            # The same ramp gives the firing rates at high force (De Luca 1982) [deluca1982], which the ramp above never reaches.
            pct_last, pct_ranked, n_unrecruited, hf_rates = float("nan"), float("nan"), None, {}
            if LAST_RECRUIT_PCT_MVC is not None or SCORE_HIGH_FORCE_RATES:
                mvc_ramp = run_mvc_ramp(thresholds, dd_params, threshold_hz)
                pct_last, n_unrecruited = mvc_ramp.pct_last, mvc_ramp.n_unrecruited
                pct_ranked, hf_rates = mvc_ramp.pct_ranked, mvc_ramp.rates
                total_error += ceiling_penalty(pct_ranked, LAST_RECRUIT_PCT_MVC)
                if SCORE_HIGH_FORCE_RATES:
                    total_error += high_force_rate_penalty(hf_rates)

            # Store metadata
            trial.set_user_attr("pct_mvc_last_recruited", pct_last)
            trial.set_user_attr("pct_mvc_ceiling_ranked", pct_ranked)
            for lev, r in hf_rates.items():
                trial.set_user_attr(f"mean_fr_{lev}_mvc_ramp", r)
            trial.set_user_attr("n_unrecruited", n_unrecruited)
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
                      f"gf={gfluctdv_scale:.3f} cv={mean_cv*100:.1f}% last={pct_last:.0f}%MVC"
                      + (f" p={weight_exponent:.2f}" if weight_exponent is not None else ""))

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
          f"gfluctdv={best.params.get('gfluctdv_scale', 0):.3f}, "
          f"weight exponent={best.params.get('weight_exponent', 1.0):.2f}")
    print(f"  ISI CV: {best.user_attrs.get('isi_cv_mean', 0)*100:.1f}% "
          f"(target: {ISI_CV_TARGET[0]*100:.0f}-{ISI_CV_TARGET[1]*100:.0f}%)")
    print(f"  Last unit recruited at {best.user_attrs.get('pct_mvc_last_recruited', float('nan')):.1f} % MVC; "
          f"unit with {CEILING_RANK} above it at {best.user_attrs.get('pct_mvc_ceiling_ranked', float('nan')):.1f} % "
          f"(target {LAST_RECRUIT_PCT_MVC}), unrecruited {best.user_attrs.get('n_unrecruited', '?')}")
    print("  MVC-ramp rates: " + ", ".join(
        f"{lev} % MVC {best.user_attrs.get(f'mean_fr_{lev}_mvc_ramp', float('nan')):.1f}/{tgt:.1f} Hz"
        for lev, tgt in HIGH_FORCE_RATE_TARGETS.items()))
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
            "last_recruit_pct_mvc": LAST_RECRUIT_PCT_MVC,
            "ceiling_rank": CEILING_RANK,
            "force_frequency_shape": list(FORCE_FREQUENCY_SHAPE),
            "high_force_rate_targets": HIGH_FORCE_RATE_TARGETS if SCORE_HIGH_FORCE_RATES else None,
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
            "network": network_block(best.params.get("weight_exponent")),
            "pct_mvc_last_recruited": best.user_attrs.get("pct_mvc_last_recruited"),
            "pct_mvc_ceiling_ranked": best.user_attrs.get("pct_mvc_ceiling_ranked"),
            **{f"mean_fr_{lev}_mvc_ramp": best.user_attrs.get(f"mean_fr_{lev}_mvc_ramp")
               for lev in HIGH_FORCE_RATE_TARGETS},
            "n_unrecruited": best.user_attrs.get("n_unrecruited"),
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
            "ceiling_weight": CEILING_WEIGHT,
            "ceiling_ramp_per_s": CEILING_RAMP_PER_S,
            "mvc_rest_s": MVC_REST_S,
            "high_force_rate_weight": HIGH_FORCE_RATE_WEIGHT,
            "mvc_hold_s": MVC_HOLD_S,
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
          f"gfluctdv={best.params.get('gfluctdv_scale', 0):.3f}, "
          f"weight exponent={best.params.get('weight_exponent', 1.0):.2f}")
    print(f"  ISI CV: {best.user_attrs.get('isi_cv_mean', 0)*100:.1f}% "
          f"(target: {ISI_CV_TARGET[0]*100:.0f}-{ISI_CV_TARGET[1]*100:.0f}%)")
    print(f"  Last unit recruited at {best.user_attrs.get('pct_mvc_last_recruited', float('nan')):.1f} % MVC; "
          f"unit with {CEILING_RANK} above it at {best.user_attrs.get('pct_mvc_ceiling_ranked', float('nan')):.1f} % "
          f"(target {LAST_RECRUIT_PCT_MVC}), unrecruited {best.user_attrs.get('n_unrecruited', '?')}")
    print("  MVC-ramp rates: " + ", ".join(
        f"{lev} % MVC {best.user_attrs.get(f'mean_fr_{lev}_mvc_ramp', float('nan')):.1f}/{tgt:.1f} Hz"
        for lev, tgt in HIGH_FORCE_RATE_TARGETS.items()))
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
            "last_recruit_pct_mvc": LAST_RECRUIT_PCT_MVC,
            "ceiling_rank": CEILING_RANK,
            "force_frequency_shape": list(FORCE_FREQUENCY_SHAPE),
            "high_force_rate_targets": HIGH_FORCE_RATE_TARGETS if SCORE_HIGH_FORCE_RATES else None,
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
            "network": network_block(best.params.get("weight_exponent")),
            "pct_mvc_last_recruited": best.user_attrs.get("pct_mvc_last_recruited"),
            "pct_mvc_ceiling_ranked": best.user_attrs.get("pct_mvc_ceiling_ranked"),
            **{f"mean_fr_{lev}_mvc_ramp": best.user_attrs.get(f"mean_fr_{lev}_mvc_ramp")
               for lev in HIGH_FORCE_RATE_TARGETS},
            "n_unrecruited": best.user_attrs.get("n_unrecruited"),
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
            "ceiling_weight": CEILING_WEIGHT,
            "ceiling_ramp_per_s": CEILING_RAMP_PER_S,
            "mvc_rest_s": MVC_REST_S,
            "high_force_rate_weight": HIGH_FORCE_RATE_WEIGHT,
            "mvc_hold_s": MVC_HOLD_S,
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
            *(["--last-recruit-mvc", str(LAST_RECRUIT_PCT_MVC)] if LAST_RECRUIT_PCT_MVC is not None else []),
            "--ceiling-rank", str(CEILING_RANK),
            "--force-frequency", next(k for k, v in FORCE_FREQUENCY_SHAPES.items() if v == FORCE_FREQUENCY_SHAPE),
            *(["--high-force-rates"] if SCORE_HIGH_FORCE_RATES else []),
            *(["--fit-weight-exponent"] if FIT_WEIGHT_EXPONENT else []),
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
                    help="Excitation fraction at which the last unit is recruited (rate-coding targets).")
    ap.add_argument("--last-recruit-mvc", type=float, default=None,
                    help="Force (%% MVC) at which the last unit should be recruited, scored on a separate MVC ramp. "
                         "Default: no ceiling term.")
    ap.add_argument("--ceiling-rank", type=int, default=0,
                    help="Score the unit with K units recruited above it instead of the last unit (default 0).")
    ap.add_argument("--force-frequency", choices=sorted(FORCE_FREQUENCY_SHAPES), default="fuglevand",
                    help="ForceModel force-frequency saturation: fuglevand (1993, default) [fuglevand1993] or human (fit to human "
                         "single motor units, force_utils.HUMAN_UNIT_FORCE_FREQUENCY).")
    ap.add_argument("--fit-weight-exponent", action="store_true",
                    help="Also fit the DD weight exponent p in [0, 1]: weights scale as (k_max / k_i) ** p "
                         "(default p = 1, equal current per input at rest; see build_dd_mn_network).")
    ap.add_argument("--high-force-rates", action="store_true",
                    help="Also score MVC-ramp firing rates at 40 and 80 %% MVC against De Luca et al. 1982 [deluca1982].")
    ap.add_argument("--de-luca-params", type=str, default="{}",
                    help="JSON overrides for de_luca_targets, e.g. fr_onset_hz.")
    ap.add_argument("--results-tag", type=str, default="polygon")
    args = ap.parse_args()
    global THRESHOLD_MODE, RECRUITMENT_RANGE, MAX_RECRUITMENT, LAST_RECRUIT_PCT_MVC, DE_LUCA_PARAMS, RESULTS_TAG
    global CEILING_RANK, FORCE_FREQUENCY_SHAPE, SCORE_HIGH_FORCE_RATES, FIT_WEIGHT_EXPONENT
    THRESHOLD_MODE, RECRUITMENT_RANGE = args.threshold_mode, args.recruitment_range
    MAX_RECRUITMENT, RESULTS_TAG = args.max_recruitment, args.results_tag
    DE_LUCA_PARAMS = json.loads(args.de_luca_params)
    LAST_RECRUIT_PCT_MVC = args.last_recruit_mvc
    CEILING_RANK, SCORE_HIGH_FORCE_RATES = args.ceiling_rank, args.high_force_rates
    FIT_WEIGHT_EXPONENT = args.fit_weight_exponent
    FORCE_FREQUENCY_SHAPE = FORCE_FREQUENCY_SHAPES[args.force_frequency]
    if LAST_RECRUIT_PCT_MVC is None:
        print("Recruitment ceiling term off (pass --last-recruit-mvc PCT to score the last unit's recruitment force).")

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
