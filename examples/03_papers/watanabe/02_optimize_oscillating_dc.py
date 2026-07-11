"""
Optimize Oscillating DC Offset to Match Reference Force
===========================================================================

This example optimizes the DC offset component of an oscillating descending drive to
match the reference force produced by constant drive (from script 01).
The drive pattern is: `DC_offset + 20*sin(2π*20*t)`.

This finds the equivalent of the original 58 pps value (Phase 3) that produces the same
force as constant drive (Phase 1) when combined with 20 Hz oscillation.

!!! note
    **Purpose**: Find DC offset for Phase 3 that matches Phase 1 force

    - Reference: Constant drive force from script 01 (configurable, default 40 Hz)
    - Optimize: DC offset + 20 Hz oscillation to match that force
    - Result: DC offset < constant drive (oscillation adds activation)

!!! important
    **Prerequisites**: Run `01_compute_baseline_force.py` first

**Output**: ``results/watanabe_optimization/dc_offset_optimized.json``

**MyoGen Components Used**:

- [`AlphaMN__Pool`][myogen.simulator.neuron.populations.AlphaMN__Pool]:
  Same motor neuron pool as script 01. Created fresh for each optimization trial
  to ensure independent network realizations.

- [`DescendingDrive__Pool`][myogen.simulator.neuron.populations.DescendingDrive__Pool]:
  Poisson spike generators driven by time-varying rate: ``DC_offset + amplitude*sin(2πft)``.
  The ``integrate()`` method advances the internal Poisson process by one timestep.

- [`Network`][myogen.simulator.Network]:
  Rebuilds the network for each trial with same connectivity parameters (30%).

- [`ForceModel`][myogen.simulator.ForceModel]:
  Computes steady-state force from spike trains to evaluate objective function.

- **External dependency**: ``optuna`` for Bayesian optimization (TPE sampler).

**Key Concept**: The oscillation itself contributes to motor neuron activation, so
the DC offset in Phase 3 must be *lower* than the constant drive in Phase 1 to
produce the same mean force. The original paper found 58 Hz offset vs 65 Hz constant.

**Workflow Position**: Step 2 of 6

**Next Step**: Run ``03_10pct_mvc_simulation.py`` to run the full 3-phase simulation.
"""

# %%

##############################################################################
# Import Libraries
# ----------------

import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import optuna

sys.path.insert(0, str(Path(__file__).parent))
from _oscillating_dc_helpers import (  # noqa: E402
    DD_CONNECTIVITY,
    MAX_FORCE_N,
    N_DD_NEURONS,
    N_MOTOR_UNITS,
    N_TRIALS,
    OSC_AMPLITUDE__HZ,
    OSC_FREQUENCY__HZ,
    REFERENCE_DRIVE__HZ,
    REFERENCE_FORCE__N,
    RESULTS_DIR,
    STUDY_NAME,
    SYNAPTIC_WEIGHT,
    TARGET_FORCE__N,
    make_storage,
    objective,
    recruitment_thresholds,
)

plt.style.use("fivethirtyeight")

##############################################################################
# Configuration Banner
# --------------------

print("\nLoading Reference Force (Constant Drive)")
print("=" * 50)
print(f"Reference force: {REFERENCE_FORCE__N:.2f} N ({REFERENCE_DRIVE__HZ:.1f} Hz constant)")
print(f"Target force: {TARGET_FORCE__N:.2f} N (match with oscillation)")
print(f"Force scaling: {MAX_FORCE_N:.0f} N maximum")
print(f"Oscillation: {OSC_FREQUENCY__HZ} Hz, amplitude {OSC_AMPLITUDE__HZ} Hz")
print(f"DD neurons: {N_DD_NEURONS}")
print(f"Connection probability: {DD_CONNECTIVITY:.1%}")
print("=" * 50 + "\n")

##############################################################################
# Run Optimization (Parallel Workers)
# ------------------------------------
#
# Each trial is a fully independent NEURON simulation (~80 s), so we fan out
# across worker *processes*. NEURON's ``h`` object is a process-global singleton
# and cannot be shared across threads. Workers are separate interpreter
# processes (``_optimize_dc_worker.py``) — they do not re-import this gallery
# file, avoiding the macOS ``spawn`` fork-bomb problem. All workers write to a
# single shared SQLite Optuna study.
#
# Override the worker count via ``MYOGEN_OPTUNA_WORKERS`` (e.g. ``=1`` for
# strictly sequential, or a small number on memory-constrained CI).

print(f"\nOptimizing DC Offset to Match Reference Force (Oscillating Drive)")
print("=" * 50)
print(f"Target force: {TARGET_FORCE__N:.2f} N ({REFERENCE_DRIVE__HZ:.1f} Hz reference)")
print(f"Oscillation: {OSC_FREQUENCY__HZ} Hz (amplitude {OSC_AMPLITUDE__HZ} Hz)")
print(f"Optimizing: DC offset component")
print(f"Trials: {N_TRIALS}\n")

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42),
    study_name=STUDY_NAME,
    storage=make_storage(),
    load_if_exists=True,
)

n_workers = int(os.environ.get("MYOGEN_OPTUNA_WORKERS", min(8, N_TRIALS)))
n_workers = max(1, min(n_workers, N_TRIALS))

per_worker = [N_TRIALS // n_workers + (1 if i < N_TRIALS % n_workers else 0) for i in range(n_workers)]
worker_script = str(Path(__file__).parent / "_optimize_dc_worker.py")
print(f"Launching {n_workers} worker process(es): trial split {per_worker}")
procs = [
    subprocess.Popen([sys.executable, worker_script, "--n-trials", str(n), "--seed", str(42 + i)])
    for i, n in enumerate(per_worker) if n > 0
]
exit_codes = [p.wait() for p in procs]
if any(code != 0 for code in exit_codes):
    raise RuntimeError(f"Optuna worker(s) failed with exit codes {exit_codes}")

# Reload to see all trials written by the workers
study = optuna.load_study(study_name=STUDY_NAME, storage=make_storage())

##############################################################################
# Analyze Results
# ---------------

best_trial = study.best_trial

print("\nOptimization Complete")
print("=" * 50)
print(f"Best trial: {best_trial.number}")
print(f"Force error: {best_trial.value:.1%}")
print("\nForce Results:")
print(f"  Target:   {TARGET_FORCE__N:.2f} N ({REFERENCE_DRIVE__HZ:.1f} Hz reference)")
print(f"  Achieved: {best_trial.user_attrs['force_achieved']:.2f} N")
print(f"  Error:    {best_trial.user_attrs['force_error']:.1%}")
print("\nOptimized Parameters:")
print(f"  DC offset: {best_trial.user_attrs['dc_offset__Hz']:.2f} Hz")
print(f"  Oscillation: {OSC_FREQUENCY__HZ} Hz (amplitude {OSC_AMPLITUDE__HZ} Hz)")
print(f"  Active MUs: {best_trial.user_attrs['n_active']}/{N_MOTOR_UNITS}")
print(
    f"  Firing rate: {best_trial.user_attrs['FR_mean']:.1f}±{best_trial.user_attrs['FR_std']:.1f} Hz"
)

##############################################################################
# Save Results
# ------------

dd_parameters = {
    "dd_neurons": N_DD_NEURONS,
    "dd_connectivity": DD_CONNECTIVITY,
    "synaptic_weight__uS": SYNAPTIC_WEIGHT,
    "dc_offset__Hz": best_trial.user_attrs["dc_offset__Hz"],
    "process_type": "poisson",
}

results = {
    "reference_drive__Hz": REFERENCE_DRIVE__HZ,
    "reference_force__N": REFERENCE_FORCE__N,
    "target_force__N": TARGET_FORCE__N,
    "achieved_force__N": best_trial.user_attrs["force_achieved"],
    "force_error": best_trial.user_attrs["force_error"],
    "force_scaling": {
        "max_force__N": MAX_FORCE_N,
    },
    "oscillation": {
        "frequency__Hz": OSC_FREQUENCY__HZ,
        "amplitude__Hz": OSC_AMPLITUDE__HZ,
    },
    "dd_parameters": dd_parameters,
}

json_path = RESULTS_DIR / "dc_offset_optimized.json"
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)

import joblib
joblib.dump(study, RESULTS_DIR / "study_oscillating_dc.pkl")

print(f"\nSaved results: {json_path}")
print(f"\nNext step: Run 03_10pct_mvc_simulation.py to run the full 3-phase simulation")

##############################################################################
# Visualize Optimization History
# -------------------------------

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Extract trial data
trial_numbers = [t.number for t in study.trials]
force_achieved = [t.user_attrs.get("force_achieved", None) for t in study.trials]
force_errors = [t.value for t in study.trials]
dc_offset_vals = [t.params.get("dc_offset", None) for t in study.trials]

# 1. Optimization progress
ax_error = fig.add_subplot(gs[0, 0])
ax_error.plot(trial_numbers, force_errors, "o-", alpha=0.6, markersize=4, color=colors[0])
ax_error.axhline(
    best_trial.value, linestyle="--", color=colors[1], label=f"Best: {best_trial.value:.2%}"
)
ax_error.set_ylabel("Relative Error")
ax_error.set_title(f"Optimization Progress (DC Offset for {REFERENCE_DRIVE__HZ:.0f} Hz Match)")
ax_error.set_yscale("log")
ax_error.grid(True, alpha=0.3)
ax_error.legend(framealpha=1.0, edgecolor="none")
ax_error.set_xlabel("Trial")

# 2. Force convergence
ax_force = fig.add_subplot(gs[1, 0])
ax_force.plot(trial_numbers, force_achieved, "o-", alpha=0.6, markersize=4, color=colors[0])
ax_force.axhline(
    TARGET_FORCE__N, linestyle="--", color=colors[1], label=f"Target: {TARGET_FORCE__N:.2f} N"
)
ax_force.set_xlabel("Trial")
ax_force.set_ylabel("Force (N)")
ax_force.set_title("Force Convergence")
ax_force.legend(framealpha=1.0, edgecolor="none")

# 3. DC offset distribution
ax_hist = fig.add_subplot(gs[0, 1])
ax_hist.hist(dc_offset_vals, bins=20, alpha=0.7, color=colors[2])
ax_hist.axvline(
    best_trial.params["dc_offset"],
    linestyle="--",
    color=colors[1],
    label=f"Best: {best_trial.params['dc_offset']:.1f}Hz",
)
ax_hist.set_xlabel("DC Offset (Hz)")
ax_hist.set_ylabel("Number of Trials")
ax_hist.set_title("DC Offset Distribution")
ax_hist.legend(framealpha=1.0, edgecolor="none")

# 4. Force vs DC offset relationship
ax_scatter = fig.add_subplot(gs[1, 1])
ax_scatter.scatter(dc_offset_vals, force_achieved, c=force_errors, s=50, alpha=0.6)
ax_scatter.scatter(
    best_trial.params["dc_offset"],
    best_trial.user_attrs["force_achieved"],
    s=200,
    marker="*",
    color=colors[3],
    label="Best Trial",
)
ax_scatter.axhline(
    TARGET_FORCE__N, linestyle="--", color=colors[1], alpha=0.5, label="Target Force"
)
ax_scatter.set_xlabel("DC Offset (Hz)")
ax_scatter.set_ylabel("Force (N)")
ax_scatter.set_title("Force-DC Offset Relationship")
ax_scatter.legend(framealpha=1.0, edgecolor="none")

plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "optimization_history_dc_offset.png",
    dpi=150,
    bbox_inches="tight",
)
plt.show()

print("\n[DONE] DC offset optimization complete!")
print(f"Best DC offset: {best_trial.user_attrs['dc_offset__Hz']:.2f} Hz")
print(f"Original Watanabe: 65 Hz constant → 58 Hz + oscillation (ratio: {58 / 65:.3f})")
print(
    f"This implementation: {REFERENCE_DRIVE__HZ:.1f} Hz constant → {best_trial.user_attrs['dc_offset__Hz']:.1f} Hz + oscillation (ratio: {best_trial.user_attrs['dc_offset__Hz'] / REFERENCE_DRIVE__HZ:.3f})"
)

# mkdocs_gallery_thumbnail_path = "gallery_thumbs/02_optimize_oscillating_dc.png"
