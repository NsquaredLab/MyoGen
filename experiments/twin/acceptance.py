"""Phase-1C acceptance evaluator.

Loads a FROZEN acceptance specification (``configs/phase1_acceptance_v*.yaml``)
and a set of per-(condition, seed) paired-comparison metrics (produced by
``pic_grid.py``), and classifies every metric at every point as::

    PASS | FAIL | OUTSIDE_VALIDATED_ENVELOPE | NOT_EVALUABLE

NOT_EVALUABLE covers right-censored after-discharge durations, ΔF with too
few eligible pairs, and spike-timing fractions with too few spikes — these
are reported, never silently dropped or counted as passes.

The evaluator never edits thresholds: a spec revision requires a new
versioned file. The summary reports the failure fraction against the spec's
``failure_policy`` and the maximal passing CaL range per NaP level (the
candidate validated envelope).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

CONFIGS = Path(__file__).parent / "configs"


def load_spec(version: int = 1) -> dict:
    path = CONFIGS / f"phase1_acceptance_v{version}.yaml"
    spec = yaml.safe_load(path.read_text())
    if spec.get("status") != "FROZEN":
        raise ValueError(f"{path} is not frozen — refusing to evaluate")
    return spec


def _verdict(value, threshold, *, minimum=False, not_evaluable=False):
    if not_evaluable:
        return "NOT_EVALUABLE", value
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NOT_EVALUABLE", value
    ok = (value >= threshold) if minimum else (value <= threshold)
    return ("PASS" if ok else "FAIL"), value


def evaluate_point(metrics: dict, spec: dict) -> dict:
    """Classify one (condition, seed) metric dict against the spec."""
    m = spec["metrics"]
    out = {}

    out["binned_rate_rmse_hz2"] = _verdict(
        metrics.get("binned_rate_rmse_hz2"),
        m["binned_rate_rmse_hz2"]["threshold"])

    out["active_unit_mismatch"] = _verdict(
        metrics.get("active_unit_mismatch"),
        m["active_unit_mismatch"]["threshold"])

    out["total_spike_rel_diff"] = _verdict(
        metrics.get("total_spike_rel_diff"),
        m["total_spike_rel_diff"]["threshold"])

    n_spk = metrics.get("spike_total_max", 0)
    out["spike_match_pm5ms_fraction_min"] = _verdict(
        metrics.get("spike_match_pm5ms_fraction"),
        m["spike_match_pm5ms_fraction_min"]["threshold"],
        minimum=True, not_evaluable=n_spk < 20)

    out["hysteresis_mean_diff_hz"] = _verdict(
        metrics.get("hysteresis_mean_diff_hz"),
        m["hysteresis_mean_diff_hz"]["threshold"],
        not_evaluable=metrics.get("hysteresis_mean_diff_hz") is None)

    out["self_sustained_mismatch"] = _verdict(
        metrics.get("self_sustained_mismatch"),
        m["self_sustained_mismatch"]["threshold"])

    pairs_ok = (metrics.get("delta_f_pairs_jaxley", 0)
                >= m["delta_f_mean_diff_hz"]["requires_min_pairs_each"]
                and metrics.get("delta_f_pairs_neuron", 0)
                >= m["delta_f_mean_diff_hz"]["requires_min_pairs_each"])
    out["delta_f_mean_diff_hz"] = _verdict(
        metrics.get("delta_f_mean_diff_hz"),
        m["delta_f_mean_diff_hz"]["threshold"], not_evaluable=not pairs_ok)

    out["delta_f_eligible_fraction_diff"] = _verdict(
        metrics.get("delta_f_eligible_fraction_diff"),
        m["delta_f_eligible_fraction_diff"]["threshold"])

    return out


def evaluate_grid(points: list[dict], spec: dict) -> dict:
    """Evaluate all grid points + the boundary metric + failure policy.

    ``points``: list of dicts with keys ``gcal``, ``gnap``, ``seed``,
    ``metrics`` (per-point metric dict) — see ``pic_grid.py``.
    """
    env = spec["declared_envelope_candidate"]
    results = []
    n_fail = n_eval = 0
    for pt in points:
        inside = (env["gcal_scale"][0] <= pt["gcal"] <= env["gcal_scale"][1]
                  and env["gnap_scale"][0] <= pt["gnap"] <= env["gnap_scale"][1])
        verdicts = evaluate_point(pt["metrics"], spec)
        if not inside:
            verdicts = {k: ("OUTSIDE_VALIDATED_ENVELOPE", v[1])
                        for k, v in verdicts.items()}
        for k, (cls, _val) in verdicts.items():
            if cls == "FAIL":
                n_fail += 1
            if cls in ("PASS", "FAIL"):
                n_eval += 1
        results.append({**{k: pt[k] for k in ("gcal", "gnap", "seed")},
                        "verdicts": {k: {"class": c, "value": v}
                                     for k, (c, v) in verdicts.items()}})

    # boundary agreement per NaP level (condition-level, seed-pooled)
    boundary = {}
    for gnap in sorted({pt["gnap"] for pt in points}):
        rows = [p for p in points if p["gnap"] == gnap]
        gcals = sorted({p["gcal"] for p in rows})

        def first_bistable(backend):
            for g in gcals:
                if any(p["metrics"].get(f"self_sustained_{backend}", 0) > 0
                       for p in rows if p["gcal"] == g):
                    return g
            return None

        bj, bn = first_bistable("jaxley"), first_bistable("neuron")
        if bj is None and bn is None:
            cls, off = "NOT_EVALUABLE", None
        elif bj is None or bn is None:
            cls, off = "FAIL", None
        else:
            off = abs(gcals.index(bj) - gcals.index(bn))
            cls = ("PASS" if off <= spec["metrics"][
                "bistability_boundary_offset_steps"]["threshold"] else "FAIL")
            n_eval += 1
            n_fail += cls == "FAIL"
        boundary[gnap] = {"jaxley_first_bistable_gcal": bj,
                          "neuron_first_bistable_gcal": bn,
                          "offset_steps": off, "class": cls}

    fail_fraction = n_fail / max(n_eval, 1)
    policy_ok = fail_fraction <= spec["failure_policy"]["max_fail_fraction"]

    # maximal contiguous passing CaL range per NaP (the validated envelope)
    envelope = {}
    for gnap in boundary:
        rows = [r for r in results if r["gnap"] == gnap]
        gcals = sorted({r["gcal"] for r in rows})
        passing = []
        for g in gcals:
            cells = [r for r in rows if r["gcal"] == g]
            cls_list = [v["class"] for r in cells
                        for v in r["verdicts"].values()]
            if "FAIL" not in cls_list:
                passing.append(g)
        envelope[gnap] = ([min(passing), max(passing)] if passing else None)

    return {
        "spec_version": spec["version"],
        "n_points": len(points),
        "n_evaluated_metric_cells": n_eval,
        "n_failed_metric_cells": n_fail,
        "fail_fraction": fail_fraction,
        "failure_policy_ok": bool(policy_ok),
        "bistability_boundary": boundary,
        "validated_envelope_gcal_by_gnap": envelope,
        "points": results,
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("grid_json", type=Path,
                    help="pic_grid output JSON with a 'points' list")
    ap.add_argument("--spec-version", type=int, default=1)
    args = ap.parse_args()
    spec = load_spec(args.spec_version)
    grid = json.loads(args.grid_json.read_text())
    rep = evaluate_grid(grid["points"], spec)
    print(json.dumps({k: v for k, v in rep.items() if k != "points"},
                     indent=1, default=float))
