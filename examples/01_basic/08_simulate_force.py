"""
Force Generation — Jaxley Spike Trains
========================================

This example converts the calibrated Jaxley motor-unit spike trains produced by
Example 03 into muscle force. The force model is simulator-independent, so the
Jaxley and NEURON versions differ only in which saved spike-train artifact they
load.

The model includes motor-unit twitch scaling, nonlinear force-frequency gain,
and temporal summation. Recruitment thresholds and the MVC normalization come
from the same ``fdi_semg`` profile card used by Example 03.

References
----------

[1] Fuglevand AJ, Winter DA, Patla AE (1993). Models of recruitment and rate
coding organization in motor-unit pools. J Neurophysiol 70:2470–2488.
"""

# %%
##############################################################################
# Import Libraries
# ----------------

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np

from myogen import load_profile
from myogen.utils.plotting.force import plot_twitch_parameter_assignment, plot_twitches

plt.style.use("fivethirtyeight")

save_path = Path("./results")
save_path.mkdir(exist_ok=True)

##############################################################################
# Create the Shared Force Model
# -----------------------------

card = load_profile()  # fdi_semg
force_model = card.force_model()

print("Force model statistics:")
print(f"\tNumber of motor units: {force_model._number_of_neurons}")
print(f"\tRecruitment ratio: {force_model._recruitment_ratio:.1f}")
print(
    "\tPeak force range: "
    f"{force_model.peak_twitch_forces__unitless[0]:.3f} - "
    f"{force_model.peak_twitch_forces__unitless[-1]:.3f}"
)
print(
    "\tContraction time range: "
    f"{force_model.contraction_times__samples[0]:.1f} - "
    f"{force_model.contraction_times__samples[-1]:.1f} samples"
)

##############################################################################
# Visualize Twitch Assignment
# ----------------------------

fig, axes = plt.subplots(2, 1, figsize=(8, 12))
plot_twitch_parameter_assignment(
    force_model,
    axes[0],
    [10, 20, 40],
    flip_x=True,
    apply_default_formatting=True,
)
axes[0].set_title("Twitch Parameter Assignment")
plot_twitches(force_model, axes[1], [10, 20, 40], apply_default_formatting=True)
axes[1].set_title("Motor Unit Twitches")
plt.tight_layout()
plt.savefig(save_path / "ex08_jaxley_twitches.png", dpi=150, bbox_inches="tight")
plt.show()

##############################################################################
# Load Jaxley Example 03 Spike Trains
# ------------------------------------

spike_path = save_path / "trapezoid_dd_spike_trains_jaxley.pkl"
if not spike_path.exists():
    raise FileNotFoundError(
        f"{spike_path} does not exist. Run 03_simulate_spike_trains_descending_drive.py first."
    )

spike_train_block = joblib.load(spike_path)
n_spike_trains = len(spike_train_block.segments[0].spiketrains)
if n_spike_trains != force_model._number_of_neurons:
    raise ValueError(
        "Example 03 spike count and force-model pool size differ: "
        f"{n_spike_trains} != {force_model._number_of_neurons}"
    )

##############################################################################
# Generate Force
# --------------

force_output = force_model.generate_force(spike_train__Block=spike_train_block)
force = force_output.magnitude[:, 0]
force_pct_mvc = 100.0 * force / float(card.value("drive.mvc_model_units"))

rng = np.random.default_rng(42)
noise_level = 0.015
noisy_force = force + rng.normal(size=force.size) * noise_level * np.mean(force)

print("Jaxley Example 03 force:")
print(f"\tPeak: {force_pct_mvc.max():.1f} % MVC")
print(f"\tMean: {force_pct_mvc.mean():.1f} % MVC")

##############################################################################
# Visualize Force Output
# ----------------------

fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
time_s = force_output.times.rescale("s")
axes[0].plot(time_s, force_pct_mvc, linewidth=2, label="Jaxley-derived force")
axes[0].set_ylabel("Force (% MVC)")
axes[0].set_title("Calibrated Descending-Drive Force")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].plot(time_s, noisy_force, linewidth=1, alpha=0.8, label="Noisy force")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Force (model units)")
axes[1].set_title("Force with Measurement Noise")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.savefig(save_path / "ex08_jaxley_force.png", dpi=150, bbox_inches="tight")
plt.show()

# mkdocs_gallery_thumbnail_path = "gallery_thumbs/08_simulate_force.png"
