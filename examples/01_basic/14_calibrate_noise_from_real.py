"""
Calibrate the Noise Profile from a Real Recording
=================================================

This example takes a real iEMG recording, derives a device-matched
noise profile with `myogen.utils.calibrate_realistic_noise_profile`,
then generates simulated noise with the derived parameters and overlays
the real and simulated PSDs in a single figure so the match is
visually verifiable.

This is the same calibration pipeline used to derive MyoGen's default
``quattrocento`` profile:

1. Notch out powerline harmonics (50/100/150/200/250 Hz for EU).
2. Extract pure-noise residuals from rest segments
   (or high-frequency content above 2 kHz as a fallback).
3. Fit ``noise_floor_uv``, ``spectral_slope``, ``peak_hz``,
   ``excess_kurtosis``, ``powerline_amplitude``, and per-harmonic ratios.

The derived dict can be unpacked directly into the simulator via::

    iemg.add_noise(snr__dB=20.0, noise_type="realistic", **profile)

!!! note
    This example expects a MAT v7.3 file with an `EMGSIGNAL` struct
    (`data` shape `(n_channels, n_samples)`, `rate` in Hz) and a
    `target_signal` (drive envelope, `0` = rest). Adjust the
    loader for other formats — only the µV array + sample rate
    + optional rest mask actually feed the calibration.
"""
# sphinx_gallery_thumbnail_number = -1

# %%

##############################################################################
# Import Libraries
# ----------------

import os
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from scipy import signal as sig
from scipy.stats import kurtosis

from myogen import set_random_seed
from myogen.utils import (
    calibrate_realistic_noise_profile,
    generate_realistic_noise,
    tune_noise_profile_with_optuna,
)

# Clean, Nature-style theme (matches the parameter-sweep figure).
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 4.5,
    "axes.titlesize": 5.5,
    "axes.labelsize": 4.5,
    "xtick.labelsize": 4.0,
    "ytick.labelsize": 4.0,
    "legend.fontsize": 4.0,
    "axes.linewidth": 0.5,
    "axes.edgecolor": "#222222",
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.0,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.0,
    "ytick.major.size": 2.0,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#cccccc",
    "svg.fonttype": "none",
    # Keep every sample in the saved vector — no silent path thinning, so the
    # MUAPs stay zoomable in Affinity/Illustrator.
    "path.simplify": False,
})

# Resolve relative to this file so the example saves into
# examples/01_basic/results/ regardless of the current working
# directory at run time.
save_path = (Path(__file__).resolve().parent / "results")
save_path.mkdir(exist_ok=True, parents=True)

##############################################################################
# Load the Real Recording
# -----------------------
#
# Point this at one of your own ramp recordings. Override the path without
# editing the file via the ``MYOGEN_IEMG_MAT`` environment variable; the
# calibration only uses (signal in µV, sampling rate, rest mask).
DEFAULT_MAT = Path.home() / "Downloads" / "Ramps" / "30MVC_pos_iEMG.mat"
REAL_MAT = Path(os.environ.get("MYOGEN_IEMG_MAT", DEFAULT_MAT))
TARGET_MUSCLE = "VL"  # one of "VM", "RF", "VL"

if REAL_MAT.exists():
    with h5py.File(REAL_MAT, "r") as f:
        fs_hz = float(f["EMGSIGNAL/rate"][()].squeeze())
        emg_mv = f["EMGSIGNAL/data"][:]  # (n_channels, n_samples), mV
        target = f["target_signal"][:].squeeze()  # commanded drive

    # 3 channels per muscle: VM (1-3), RF (4-6), VL (7-9), 1-indexed in MATLAB.
    muscle_channel_map = {"VM": (0, 1, 2), "RF": (3, 4, 5), "VL": (6, 7, 8)}
    ch_indices = muscle_channel_map[TARGET_MUSCLE]

    # Convert mV → µV, transpose to (n_samples, n_channels) for the calibrator.
    real_uv = emg_mv[list(ch_indices), :].T * 1000.0
    rest_mask = target < 0.01
    source = f"{REAL_MAT.name} ({TARGET_MUSCLE}, channels {ch_indices})"
else:
    # No recording found — synthesise a short stand-in so the example runs
    # anywhere: three channels of the realistic device noise the model
    # produces, plus a brief higher-amplitude "contraction" in the middle so
    # the rest mask and the panel-a trace have structure. Point
    # MYOGEN_IEMG_MAT at a real .mat to calibrate against your own hardware.
    fs_hz = 10240.0
    n_demo = int(60 * fs_hz)
    demo_rng = np.random.default_rng(0)
    real_uv = np.stack(
        [
            generate_realistic_noise(n_demo, fs_hz, 4.0, rng=np.random.default_rng(ch))
            for ch in range(3)
        ],
        axis=1,
    )
    rest_mask = np.ones(n_demo, dtype=bool)
    burst = slice(int(0.40 * n_demo), int(0.60 * n_demo))
    real_uv[burst] += demo_rng.normal(0.0, 40.0, size=(burst.stop - burst.start, 3))
    rest_mask[burst] = False
    source = "synthetic demo (set MYOGEN_IEMG_MAT for real calibration)"

print(f"Loaded {source}")
print(f"  fs           : {fs_hz:.0f} Hz")
print(f"  duration     : {real_uv.shape[0] / fs_hz:.0f} s")
print(f"  rest fraction: {rest_mask.mean() * 100:.1f}%")

##############################################################################
# Calibrate the Noise Profile
# ---------------------------
#
# The calibration pipeline in a single call — derives the same parameter set
# the simulator's ``noise_type="realistic"`` mode consumes.

profile = calibrate_realistic_noise_profile(
    real_uv, fs_hz, rest_mask=rest_mask, powerline_hz=50.0
)

print("\nDerived profile:")
for key, value in profile.items():
    if isinstance(value, list):
        print(f"  {key:30s} {[round(x, 3) for x in value]}")
    elif isinstance(value, float):
        print(f"  {key:30s} {value:.4f}")
    else:
        print(f"  {key:30s} {value}")

##############################################################################
# Refine the Profile with Optuna
# ------------------------------
#
# The closed-form profile fits the moments but leaves residual shape
# mismatch (e.g., low-frequency rolloff curvature). Optuna runs a TPE
# search around the closed-form solution, minimising the L1 log-PSD
# distance — a few seconds of compute typically halves the residual.

print("\nRunning Optuna refinement (this takes a few seconds)...")
tuned_profile = tune_noise_profile_with_optuna(
    real_uv,
    fs_hz,
    profile,
    rest_mask=rest_mask,
    n_trials=120,
)

print("\nTuned profile:")
for key, value in tuned_profile.items():
    if isinstance(value, list):
        print(f"  {key:30s} {[round(x, 3) for x in value]}")
    elif isinstance(value, float):
        print(f"  {key:30s} {value:.4f}")
    else:
        print(f"  {key:30s} {value}")

##############################################################################
# Generate Simulated Noise from Both Profiles
# -------------------------------------------
#
# Match the duration of the rest segments used for calibration so the
# Welch averages are over comparable lengths.


def _make_sim_noise(prof: dict, n: int) -> np.ndarray:
    return generate_realistic_noise(
        n,
        fs_hz,
        noise_rms=prof["noise_floor_uv"],
        spectral_slope=prof["spectral_slope"],
        excess_kurtosis=prof["excess_kurtosis"],
        powerline_hz=prof["powerline_hz"],
        powerline_amplitude=prof["powerline_amplitude"],
        powerline_harmonic_ratios=prof["powerline_harmonic_ratios"],
        peak_hz=prof["peak_hz"],
        analog_hpf_hz=prof["analog_hpf_hz"],
    )


set_random_seed(0)
n_rest_samples = int(rest_mask.sum())
sim_noise = _make_sim_noise(profile, n_rest_samples)
set_random_seed(0)
sim_noise_tuned = _make_sim_noise(tuned_profile, n_rest_samples)

print(f"\nSimulated noise RMS (closed-form): {np.sqrt(np.mean(sim_noise ** 2)):.3f} µV")
print(f"Simulated noise RMS (tuned)      : {np.sqrt(np.mean(sim_noise_tuned ** 2)):.3f} µV")
print(f"Real (rest-mask) RMS             : {np.sqrt(np.mean(real_uv[rest_mask] ** 2)):.3f} µV (averaged across channels)")

##############################################################################
# PSD Overlay — Real vs Simulated
# -------------------------------
#
# Welch PSDs on log-frequency, dB axes. Real (averaged across the
# muscle's channels) in solid grey; simulated in colour. A close match
# means the profile reproduces the device + environment noise.

nperseg = min(8192, n_rest_samples)
noverlap = int(nperseg * 0.75)

# Per-channel real PSD on rest, then median across channels.
real_psd_stack = []
for ch in range(real_uv.shape[1]):
    f_real, psd_ch = sig.welch(
        real_uv[rest_mask, ch],
        fs=fs_hz,
        nperseg=nperseg,
        noverlap=noverlap,
        average="median",
    )
    real_psd_stack.append(psd_ch)
real_psd = np.median(np.stack(real_psd_stack, axis=0), axis=0)

f_sim, sim_psd = sig.welch(
    sim_noise,
    fs=fs_hz,
    nperseg=nperseg,
    noverlap=noverlap,
    average="median",
)
f_sim_t, sim_psd_tuned = sig.welch(
    sim_noise_tuned,
    fs=fs_hz,
    nperseg=nperseg,
    noverlap=noverlap,
    average="median",
)

real_db = 10.0 * np.log10(np.maximum(real_psd, 1e-30))
sim_db = 10.0 * np.log10(np.maximum(sim_psd, 1e-30))
sim_db_tuned = 10.0 * np.log10(np.maximum(sim_psd_tuned, 1e-30))

REAL_C = "#111111"        # real recording
CLOSED_C = "#9aa0a6"      # closed-form profile
TUNED_C = "#d62728"       # Optuna-tuned profile

# Per-knob accent colours — identical to the parameter-sweep figure so the
# two figures read as a matched pair.
KNOB_COLOR = {
    "noise_floor_uv": "#1f77b4",
    "spectral_slope": "#ff7f0e",
    "peak_hz": "#2ca02c",
    "powerline_amplitude": "#7e3ca2",
    "analog_hpf_hz": "#d62728",
    "excess_kurtosis": "#7f7f7f",
}


def _panel_letter(ax, letter, *, dx=-0.02):
    ax.text(dx, 1.04, letter, transform=ax.transAxes, fontsize=7,
            fontweight="bold", color="#111111", ha="right", va="bottom")


# Cleanest channel (lowest kurtosis) for the example traces.
_dm_all = real_uv[rest_mask] - real_uv[rest_mask].mean(axis=0)
clean_ch = int(np.argmin([kurtosis(_dm_all[:, c]) for c in range(_dm_all.shape[1])]))

fig = plt.figure(figsize=(7.18, 3.2))
outer = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[0.7, 2.0],
                          width_ratios=[1.55, 1.0], hspace=0.6, wspace=0.42)
ax_act = fig.add_subplot(outer[0, :])                 # activation segment (motor units), full width
left = outer[1, 0].subgridspec(2, 1, height_ratios=[3.0, 1.0], hspace=0.10)
ax_psd = fig.add_subplot(left[0])
ax_res = fig.add_subplot(left[1], sharex=ax_psd)
ax_par = fig.add_subplot(outer[1, 1])

##############################################################################
# Panel a — PSD match (real vs closed-form vs Optuna-tuned) + residual
# --------------------------------------------------------------------

for f_line in (50.0, 100.0, 150.0, 200.0, 250.0):
    ax_psd.axvline(f_line, color="#dddddd", linewidth=0.5, zorder=0)
ax_psd.semilogx(f_real, real_db, color=REAL_C, linewidth=1.3, label="Real (rest)")
ax_psd.semilogx(f_sim, sim_db, color=CLOSED_C, linewidth=1.0, label="Closed-form")
ax_psd.semilogx(f_sim_t, sim_db_tuned, color=TUNED_C, linewidth=1.1, label="Optuna-tuned")
ax_psd.set_xlim(5.0, fs_hz / 2.0)
ax_psd.set_ylabel("PSD (dB)")
ax_psd.tick_params(labelbottom=False)
ax_psd.grid(True, axis="y", alpha=0.22, linewidth=0.5)
ax_psd.legend(loc="upper right", facecolor="white", edgecolor="#cccccc",
              framealpha=0.95, handlelength=1.6, borderpad=0.4)
ax_psd.set_title(f"Calibrated noise model vs real device ({TARGET_MUSCLE})",
                 loc="left", fontweight="bold", color="#111111", pad=4)
_panel_letter(ax_psd, "b", dx=-0.08)

# Residual strip: simulated − real (closer to 0 is better).
ax_res.axhspan(-3, 3, color="#eeeeee", zorder=0)
ax_res.axhline(0, color="#888888", linewidth=0.6)
ax_res.semilogx(f_real, sim_db - real_db, color=CLOSED_C, linewidth=0.9)
ax_res.semilogx(f_real, sim_db_tuned - real_db, color=TUNED_C, linewidth=1.0)
ax_res.set_xlim(5.0, fs_hz / 2.0)
ax_res.set_xticks([10, 50, 100, 500, 1000, 5000])
ax_res.get_xaxis().set_major_formatter(ScalarFormatter())
ax_res.minorticks_off()
ax_res.set_ylim(-8, 8)
ax_res.set_xlabel("Frequency (Hz)")
ax_res.set_ylabel("Δ (dB)")

##############################################################################
# Panel b — how much Optuna moved each knob (% change from the quick fit)
# -----------------------------------------------------------------------
#
# Each bar shows where Optuna settled *within that knob's search range*,
# normalised to 0 % (lower bound) – 100 % (upper bound). This is directly
# comparable across knobs and immune to the near-zero-baseline blow-up that
# a (tuned − quick)/quick ratio suffers. The closed-form starting point is
# marked with a tick so the move is visible. Colour matches the sweep tiles.
#
# Bounds replicate tune_noise_profile_with_optuna's _objective exactly
# (search ranges are defined relative to the closed-form `profile`).

# excess_kurtosis is intentionally omitted: it's a time-domain 4th-moment
# property the log-PSD objective can't constrain, so its tuned value is
# meaningless here. Only knobs the spectral fit actually moves.
_NYQ = fs_hz / 2.0
PARAMS = [
    ("noise_floor_uv", "Noise floor",
     lambda i: (max(i * 0.5, 0.1), i * 2.0)),
    ("spectral_slope", "Spectral slope",
     lambda i: (i - 0.8, i + 0.8)),
    ("peak_hz", "Mid-band peak",
     lambda i: (max(i * 0.5, 100.0), min(i * 2.0, _NYQ * 0.9))),
    ("powerline_amplitude", "Powerline amp",
     lambda i: (max(i * 0.3, 0.0), i * 3.0 + 0.05)),
    ("analog_hpf_hz", "Analog HPF",
     lambda i: (max(i * 0.3, 1.0), i * 3.0)),
]

# Closed-form sits at 0 (centre). The bar runs toward whichever bound
# Optuna moved to, normalised so the lower bound = −100 % and the upper
# bound = +100 %. Each side is scaled independently (the search ranges are
# asymmetric about the closed-form value), so 0 always means "no change"
# and ±100 % always means "pinned to a search bound" — comparable across rows.
for row, (key, label, bounds) in enumerate(PARAMS):
    y = len(PARAMS) - 1 - row
    c = KNOB_COLOR[key]
    q, t = profile[key], tuned_profile[key]
    lo, hi = bounds(q)
    if t >= q:
        frac = (t - q) / (hi - q) * 100.0 if (hi - q) > 1e-12 else 0.0
    else:
        frac = (t - q) / (q - lo) * 100.0 if (q - lo) > 1e-12 else 0.0
    frac = float(np.clip(frac, -100.0, 100.0))
    ax_par.barh(y, frac, height=0.55, color=c, zorder=2)
    off = 4 if frac >= 0 else -4
    ax_par.text(frac + off, y, f"{frac:+.0f}%", va="center",
                ha="left" if frac >= 0 else "right",
                fontsize=4.0, color=c, fontweight="bold")

# closed-form reference line down the middle
ax_par.axvline(0, color="#111111", linewidth=0.9, zorder=3)

ax_par.set_ylim(-0.6, len(PARAMS) - 0.4)
ax_par.set_yticks(range(len(PARAMS)))
ax_par.set_yticklabels([label for _, label, _ in PARAMS][::-1], fontsize=4.0)
ax_par.set_xlim(-130, 130)
ax_par.set_xticks([-100, 0, 100])
ax_par.set_xticklabels(["−100%", "0", "+100%"])
ax_par.set_xlabel("Move within search range  (0 = closed-form, ±100 % = bounds)")
ax_par.tick_params(length=2)
ax_par.set_title("How far Optuna moved each knob",
                 loc="left", fontweight="bold", color="#111111", pad=4, fontsize=5.5)
ax_par.grid(True, axis="x", alpha=0.18, linewidth=0.5)
_panel_letter(ax_par, "c", dx=-0.32)

##############################################################################
# Panel a — the real recording in full, with a zoom on the rest noise
# ------------------------------------------------------------------
#
# The whole signal (cleanest channel) shows the contraction towering over
# the rest periods (shaded) — signal ≫ noise. The boxed rest window is
# zoomed on the right to reveal the device-noise texture we calibrate to.

_sig = real_uv[:, clean_ch].astype(float)
_sig = _sig - _sig.mean()

# One full activation block (motor units firing) next to one full rest
# block (device noise). Plotted at full sample rate; the dense lines are
# rasterised so every sample is rendered faithfully without a giant vector
# path. Both panels share the same µV scale so the signal≫noise gap is honest.
def _runs_of(mask):
    e = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.where(e == 1)[0], np.where(e == -1)[0]))


def _longest(runs):
    return max(runs, key=lambda se: se[1] - se[0])


SEG_S = 45.0                                           # window length per panel
_sn = int(SEG_S * fs_hz)


def _center_window(run):
    s, e = run
    c = (s + e) // 2
    a = int(np.clip(c - _sn // 2, 0, _sig.size - _sn))
    return a, a + _sn


_a0, _a1 = _center_window(_longest(_runs_of(~rest_mask)))   # 60 s around a contraction
_at = np.arange(_a1 - _a0) / fs_hz                     # s
_ymax = 1.05 * np.max(np.abs(_sig[_a0:_a1]))

# Activation — motor units tower over the noise floor, full width.
ax_act.plot(_at, _sig[_a0:_a1], color=REAL_C, linewidth=0.15, rasterized=True)
ax_act.set_xlim(0, _at[-1])
ax_act.set_ylim(-_ymax, _ymax)
ax_act.set_xlabel("Time (s)")
ax_act.set_ylabel("µV")
ax_act.set_title(f"Real iEMG — motor units during contraction  (t≈{_a0 / fs_hz:.0f} s, {_at[-1]:.0f} s window)",
                 loc="left", fontweight="bold", color="#111111", pad=4, fontsize=5.5)
_panel_letter(ax_act, "a", dx=-0.08)

sns.despine(ax=ax_act, offset=2, trim=False)
sns.despine(ax=ax_psd, offset=2, trim=False)
sns.despine(ax=ax_res, offset=2, trim=False)
sns.despine(ax=ax_par, offset=2, trim=False, left=False)

plt.tight_layout()
# dpi=600 sets the resolution of the rasterised iEMG traces inside the
# otherwise-vector SVG/PDF (text, axes, panels b/c stay vector).
fig.savefig(save_path / "noise_calibration_overlay.png", dpi=600, transparent=True)
fig.savefig(save_path / "noise_calibration_overlay.svg", dpi=600, transparent=True)
fig.savefig(save_path / "noise_calibration_overlay.pdf", dpi=600, transparent=True)
plt.show()

##############################################################################
# How Close is the Match?
# -----------------------
#
# We summarise the agreement in three bands so it's a single-number
# read on how well the simulator reproduces the real spectrum.

bands = [("20-100 Hz", 20.0, 100.0), ("100-500 Hz", 100.0, 500.0), ("500-2000 Hz", 500.0, 2000.0)]


def _band_power(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        return 0.0
    return float(np.sum(psd[mask]) * (freqs[1] - freqs[0]))


print("\nBand-power match (dB difference; |Δ| < 3 dB is a tight match):")
print(f"  {'band':<14}{'real (dB)':>12}{'sim (dB)':>12}{'Δ closed':>11}{'Δ tuned':>11}")
for label, lo, hi in bands:
    p_real = _band_power(f_real, real_psd, lo, hi)
    p_sim = _band_power(f_sim, sim_psd, lo, hi)
    p_sim_t = _band_power(f_sim_t, sim_psd_tuned, lo, hi)
    db_real = 10.0 * np.log10(max(p_real, 1e-30))
    db_sim = 10.0 * np.log10(max(p_sim, 1e-30))
    db_sim_t = 10.0 * np.log10(max(p_sim_t, 1e-30))
    print(
        f"  {label:<14}{db_real:>12.2f}{db_sim_t:>12.2f}"
        f"{db_sim - db_real:>11.2f}{db_sim_t - db_real:>11.2f}"
    )
print(f"\nOptuna final log-PSD L1 loss: {tuned_profile['_optuna_loss']:.4f}")
