"""
SCI pathological iEMG, mechanistically: same model, vary only the PIC
====================================================================

This is the mechanistic remake of the original SCI iEMG example. That version
reproduced three discharge phenotypes by **hand-sculpting the descending drive**
(a tonic floor for loss-of-derecruitment, 6 Hz bursts for clonus). Here all
panels share the **same muscle, electrode, and motoneuron pool**, and the
phenotypes emerge from the motoneuron **persistent inward current (PIC)** state:

1. **Voluntary modulation** -- healthy PIC; a 0.5 Hz voluntary command recruits
   and **derecruits** the pool each cycle.
2. **Loss of derecruitment** -- up-regulated PIC (``gamma``); the *same* command
   no longer lets units go silent at the troughs.
3. **Modulation -> spasm** -- up-regulated PIC; the voluntary command runs for
   the first half then stops, but the PIC sustains an **involuntary discharge**.

Modelling choices and honest caveats:

* **Model = NERLab** (Cisi-Kohn-Elias), the MyoGen manuscript's motoneuron.
  SCI spasticity is induced purely by up-regulating the PIC: the dendritic Ca
  PIC (``gamma``) PLUS somatic NaP (``nap_factor``) -- NERLab needs both to
  self-sustain (gamma alone only amplifies). Healthy column: gamma=0.5, NaPx1;
  SCI columns: gamma=1.3, NaPx5.
* **PIC magnitude is physiological.** The single-cell sustained dendritic Ca PIC
  is ~-9 nA, within the experimental ~5-15 nA range. NERLab's caL does not
  inactivate, so a recruited unit latches (the plateau persists) rather than
  decaying.
* **Firing variability is drive-driven; the spasm CV collapses.** Voluntary
  firing is irregular (ISI CV ~24%) -- the variability comes mostly from the
  noisy descending drive, plus a small injected OU membrane-noise current
  (``mn_noise``) that scales with the drive and leaves only a floor when it
  withdraws. When the drive stops, firing is paced by the intrinsic PIC and the
  CV collapses to ~5-6 % -- squarely in the Gorassini (2004) self-sustained band
  (5.4 +/- 1.6 %), reproducing their finding that spasms fire more regularly
  than voluntary effort.
* **Rate caveat.** NERLab gives realistic voluntary rates (~23 Hz, cf. FDI), and
  the self-sustained spasm fires at ~11 Hz -- about half the voluntary rate, but
  ~2x the ~5 Hz of real SCI spasms. NERLab's brief slow-K AHP keeps its f-I onset
  somewhat high; it cannot be tuned down without killing the self-sustain or
  contradicting the SCI literature (which REDUCES the AHP), so we accept it.
* **No afferent/reflex loops.** The original's third phenotype was 6 Hz *clonus*
  (a stretch-reflex-loop oscillation); here it is replaced by an intrinsic
  PIC-driven spasm. The spasm does **not** self-terminate within the window --
  NERLab's Ca PIC does not deactivate, so a real off-switch needs inhibition /
  afferent input (illustrated in the single-cell mechanism panel), which is
  outside this open-loop pool.
"""
# sphinx_gallery_thumbnail_number = -1
import sys
from pathlib import Path

import joblib
import numpy as np
import quantities as pq
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from neo import AnalogSignal

from myogen import set_random_seed, get_random_generator
from myogen.utils.types import pps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pic_protocols as pic

def apply_pub_style():
    """Publication style (scienceplots science+nature), matching the MyoGen
    ISI-CV figures -- NOT fivethirtyeight. Clean white background, sans-serif,
    thick axes, top/right spines off (re-enabled per twin axis). Importing
    myogen applies the seaborn lavender theme as a side effect, so this is
    re-asserted right before the figure is built, not only at import time."""
    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", "nature"])
        sns.set_context("paper", font_scale=1.0)
    except Exception:
        plt.style.use("fivethirtyeight")
    plt.rcParams["text.usetex"] = False
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["svg.fonttype"] = "none"        # keep text editable in SVG/PDF
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Liberation Sans", "Roboto", "DejaVu Sans"]
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.spines.left"] = True
    plt.rcParams["axes.spines.bottom"] = True
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["xtick.top"] = False
    plt.rcParams["ytick.right"] = False
    plt.rcParams["axes.linewidth"] = 1.0     # comparison fig used 2.0 -- too
    plt.rcParams["xtick.major.width"] = 1.0  # thick for this dense composite
    plt.rcParams["ytick.major.width"] = 1.0
    plt.rcParams["xtick.minor.visible"] = False
    plt.rcParams["ytick.minor.visible"] = False
    plt.rcParams["path.simplify"] = False        # draw every sample on dense EMG


apply_pub_style()

set_random_seed(42)
N_MU = 40
TOTAL_S = 8.0
NAP_CEILING = 0.00215
# Peak OU membrane-noise current (nA) per motoneuron, reached at full descending
# drive; its amplitude SCALES WITH THE DRIVE (synaptic bombardment tracks input)
# and falls to MN_NOISE*NOISE_FLOOR when the drive withdraws. So voluntary firing
# is irregular (CV ~10%) while a drive-off self-sustained spasm, paced by the
# intrinsic PIC, is regular (CV ~4%) -- the Gorassini (2004) spasm signature.
# Without this the PIC discharge is artificially clock-like (CV <1%, an artifact).
MN_NOISE = 0.4
NOISE_FLOOR = 0.3
# Model left at its native mAHP. We earlier tried halving mAHP to lift the
# (low ~6 pps) discharge rate, but mAHP is exactly what REGULARISES firing:
# reducing it pushed the ISI CV from ~10% to ~26% and destroyed the spasm CV
# collapse (14.7% instead of ~4%). Since the diagnostic spasticity signature is
# the low ISI CV, not the rate (Gorassini 2004: rate even RISES with drive), we
# keep mAHP intact and accept the model's low voluntary discharge rate.
MAHP_FACTOR = 1.0
DD_WEIGHT = 0.15
SPASM_ONSET_S = 4.0   # time the voluntary command stops -> PIC-sustained spasm
XTICKS = np.arange(0, TOTAL_S + 1, 2)
SPASM_LABEL = "Modulation -> spasm (SCI)"

try:
    save_path = Path(__file__).parent / "results"
except NameError:
    save_path = Path.cwd() / "results"
save_path.mkdir(exist_ok=True, parents=True)


def modulation_then_silence(peak_pps=45.0, freq_hz=0.5, stop_s=4.0,
                            total_s=TOTAL_S):
    """Cyclic voluntary command for the first `stop_s`, then zero (the input
    stops but the PIC sustains firing -> spasm)."""
    n = int(total_s * 1000.0 / float(pic._POOL_TIMESTEP.magnitude))
    t = np.linspace(0.0, total_s, n, endpoint=False)
    cmd = (peak_pps / 2.0) * (1.0 - np.cos(2.0 * np.pi * freq_hz * t))
    cmd[t >= stop_s] = 0.0
    noise = np.clip(get_random_generator().normal(0, 1.0, size=n), 0, None)
    return AnalogSignal((cmd + noise) * pps,
                        sampling_period=(total_s / n) * pq.s)


# %%
# Shared peripheral model: one muscle + electrode + MUAP set for all conditions.
# Cache the (expensive) MUAP computation so re-runs are cheap.
cache = save_path / f"sci_mechanistic_iemg_sim_n{N_MU}.pkl"
if cache.exists():
    iemg_sim = joblib.load(cache)
else:
    muscle, _ = pic.build_muscle(N_MU)
    iemg_sim = pic.make_iemg_simulator(muscle, N_MU)
    joblib.dump(iemg_sim, cache)

voluntary, _ = pic.cyclic_voluntary_drive(peak_pps=90.0, freq_hz=0.5,
                                          total_s=TOTAL_S)
spasm_drive = modulation_then_silence(peak_pps=90.0, stop_s=SPASM_ONSET_S)

# NERLab (Cisi-Kohn-Elias) motoneuron -- the model used in the MyoGen
# manuscript. SCI spasticity is induced purely by up-regulating the motoneuron
# PIC: the dendritic Ca PIC (``gamma``) plus somatic NaP (``nap_factor``), which
# together produce self-sustained firing. label -> (drive, gamma, nap_factor).
# Rate caveat: NERLab gives realistic voluntary rates (~23 Hz, cf. FDI), and the
# self-sustained spasm fires at ~11 Hz (the drive-off discharge drops to roughly
# half the voluntary rate). That is ~2x the ~5 Hz of real SCI spasms (Gorassini
# 2004) -- NERLab's brief slow-K AHP keeps its floor somewhat high -- but the
# loss of derecruitment and the ISI-CV collapse are reproduced cleanly.
MODEL = "NERLab"
conditions = {
    "Voluntary modulation (healthy)": (voluntary, 0.5, 1.0),
    "Loss of derecruitment (SCI)": (voluntary, 1.3, 5.0),
    "Modulation -> spasm (SCI)": (spasm_drive, 1.3, 5.0),
}

# Run the three pool simulations once and cache the (slim) results, so figure
# tweaks re-render in seconds. Delete this .pkl after changing any simulation
# parameter (model, N_MU, gamma, lambda, drive, SNR) to force a re-simulation.
results_cache = save_path / f"sci_mechanistic_sims_{MODEL}_n{N_MU}.pkl"
if results_cache.exists():
    results = joblib.load(results_cache)
else:
    results = {}
    for label, (drive, gamma, napf) in conditions.items():
        block = pic.run_pool(drive, n_mu=N_MU, gamma=gamma, model=MODEL,
                             nap_factor=napf, total_s=TOTAL_S,
                             mn_noise=MN_NOISE, noise_floor=NOISE_FLOOR,
                             dd_weight__uS=DD_WEIGHT)
        emg = pic.synthesize_iemg(block, N_MU, iemg_sim=iemg_sim, snr_dB=20)
        results[label] = {
            "block": block,
            "iemg": {"iemg": emg["iemg"], "times": emg["times"]},
        }
        active = sum(1 for st in block.segments[0].spiketrains if len(st) > 0)
        print(f"{label}: active MUs={active}/{N_MU}")
    joblib.dump(results, results_cache)

# Discharge rate and ISI CV are recomputed fresh each run (cheap) so the
# binning/windowing can be tuned without re-running the simulations.
for _label, _data in results.items():
    _data["rate"] = pic.population_rate(_data["block"], TOTAL_S, win_s=0.8,
                                        step_s=0.02)  # 0.8 s window, 20 ms steps
    _data["cv"] = pic.population_cv(_data["block"], TOTAL_S, win_s=1.0, step_s=0.02)

labels = list(conditions.keys())

_CV_CAND = np.array([0.2, 0.3, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15,
                     20, 25, 30, 40, 50])


def cv_axis_range(cv):
    """Per-row data-driven (min->max) log y-range + nice ticks for one panel's
    CV trace, so each condition's CV fills its own axis."""
    v = cv[np.isfinite(cv) & (cv > 0)]
    if v.size == 0:
        return (1.0, 10.0), _CV_CAND[(_CV_CAND >= 1) & (_CV_CAND <= 10)]
    ylim = (float(v.min()) / 1.15, float(v.max()) * 1.15)
    ticks = _CV_CAND[(_CV_CAND >= ylim[0]) & (_CV_CAND <= ylim[1])]
    return ylim, ticks


def mark_spasm_onset(ax, label, y_frac=0.97):
    """On the modulation->spasm panel, draw a dashed line at the moment the
    voluntary command stops and the PIC-sustained spasm begins."""
    if label != SPASM_LABEL:
        return
    ax.axvline(SPASM_ONSET_S, color="0.25", linestyle="--", linewidth=1.2,
               zorder=6)
    ax.annotate("command off -> spasm", xy=(SPASM_ONSET_S, y_frac),
                xycoords=("data", "axes fraction"), xytext=(4, 0),
                textcoords="offset points", va="top", ha="left",
                color="0.25", zorder=7)


def overlay_rate_envelope(ax, centers, rate, span_max):
    """Draw the discharge rate as a red envelope scaled per row: its OWN min sits
    at the EMG zero line and its max at the EMG peak (span_max), so the envelope
    fills the panel. The right axis reports the TRUE rate in pps over the
    envelope's [min, max]; the curve is scaled only for viewing."""
    rmin = float(rate.min())
    rmax = float(rate.max())
    if rmax <= rmin:
        rmax = rmin + 1.0
    ax.plot(centers, (rate - rmin) / (rmax - rmin) * span_max, color="red",
            linewidth=1.5, alpha=0.9, zorder=5)
    ax_r = ax.twinx()
    lo, hi = ax.get_ylim()
    # invert the view scaling back to pps: y = 0 -> rmin, y = span_max -> rmax
    to_pps = lambda y: rmin + (y / span_max) * (rmax - rmin)
    ax_r.set_ylim(to_pps(lo), to_pps(hi))   # real pps; DR min aligns with EMG 0
    ax_r.set_ylabel("Discharge rate (pps)", color="red")
    ax_r.tick_params(axis="y", colors="red")
    span = rmax - rmin
    tick_step = 1 if span <= 6 else (2 if span <= 12 else 5)
    lo_tick = np.ceil(rmin / tick_step) * tick_step
    ax_r.set_yticks(np.arange(lo_tick, rmax + tick_step / 2, tick_step))
    ax_r.grid(False)
    return ax_r


# %%
# Single-cell PIC mechanism (cached): a Powers2017 cell near the bistable
# threshold -- a brief pulse latches the dendritic Ca plateau (self-sustained
# firing) and a gentle inhibition switches it off. The cell-level basis of the
# pool spasm.
mech_cache = save_path / "sci_mechanistic_singlecell.pkl"
if mech_cache.exists():
    mech = joblib.load(mech_cache)
else:
    mech = pic.single_cell_pic_mechanism()
    joblib.dump(mech, mech_cache)

# Gorassini et al. 2004 (Brain) self-sustained-firing ISI CV: 5.4 +/- 1.6 %.
GOR_CV, GOR_CV_SD = 5.4, 1.6


def despine_fig(fig):
    """sns.despine top+right on every axis (twin axes keep their right tick
    labels so the secondary scale still reads without a spine line)."""
    for ax in fig.axes:
        is_twin = ax.spines["right"].get_visible()
        sns.despine(ax=ax, top=True, right=True, offset=5, trim=not is_twin)
        if is_twin:
            ax.tick_params(axis="y", which="both", right=True, labelright=True)


def save_fig(fig, stem):
    fig.tight_layout()
    # dpi pins the resolution of any rasterized=True artists (the raster markers)
    fig.savefig(save_path / f"{stem}.svg", bbox_inches="tight", dpi=300)
    fig.savefig(save_path / f"{stem}.pdf", bbox_inches="tight", dpi=300)


# The three figures share the SAME descending drive; only the motoneuron PIC
# state varies across conditions (columns). Each is its own figure with its own
# size, so panels can be assembled freely.
apply_pub_style()                       # re-assert (myogen imports re-theme sns)

# %%
# Figure 1 -- single-cell PIC mechanism: Vm and the up-regulated inward currents.
# figure width 7.18 in = 180 mm (Nature double-column); heights per figure.
fig_m, (ax_mv, ax_mi) = plt.subplots(2, 1, figsize=(7.18, 3.0), sharex=True)
ax_mv.plot(mech["t"], mech["vm"], color="k", linewidth=0.5, rasterized=True)
ax_mv.set_ylabel("Vm (mV)")
ax_mv.set_title("Single-cell PIC mechanism (NERLab): a pulse latches the "
                "dendritic Ca PIC -> self-sustained firing; inhibition switches "
                "it off", fontsize=10)
# Dendritic Ca (caL) is the bistable PLATEAU -- the real PIC carrying the firing.
# The somatic Na (napp NaP) is SPIKE-COUPLED (>+60 mV, ~0 between spikes): boosted
# x5 to help engage the Ca plateau, but NOT a subthreshold PIC. Floored so the ~0
# off-state is finite on the log axis.
ax_mi.plot(mech["t"], np.clip(-np.asarray(mech["pic_nA"]), 1e-2, None),
           color="red", linewidth=0.8, label="dendritic Ca (PIC plateau)",
           rasterized=True)
ax_mi.plot(mech["t"], np.clip(-np.asarray(mech["nap_nA"]), 1e-2, None),
           color="darkorange", linewidth=0.7, alpha=0.8,
           label="somatic Na (spike-coupled boost)", rasterized=True)
ax_mi.set_yscale("log")
ax_mi.set_ylim(0.008, 20)               # floor == clip -> off-state at bottom edge
ax_mi.set_yticks([0.01, 0.1, 1, 10])
ax_mi.yaxis.set_major_formatter(ScalarFormatter())   # plain numbers, not 10^x
ax_mi.minorticks_off()
ax_mi.set_ylabel("inward current (-nA)")
ax_mi.set_xlabel("Time (s)")
ax_mi.set_xlim(mech["t"][0], mech["t"][-1])
ax_mi.legend(loc="center right", fontsize=7, frameon=False)
for axm in (ax_mv, ax_mi):
    axm.axvspan(*mech["t_pulse"], color="0.85", zorder=0)
    axm.axvspan(*mech["t_inhib"], color="#cfe0ff", zorder=0)
    axm.grid(True, alpha=0.3)
ax_mv.annotate("pulse", xy=(np.mean(mech["t_pulse"]), 0.96),
               xycoords=("data", "axes fraction"), ha="center", va="top",
               fontsize=8, color="0.3")
ax_mv.annotate("inhibition", xy=(np.mean(mech["t_inhib"]), 0.96),
               xycoords=("data", "axes fraction"), ha="center", va="top",
               fontsize=8, color="#2b5fb0")
despine_fig(fig_m)
save_fig(fig_m, "sci_mechanistic_mechanism")

# %%
# Figure 2 -- motor-unit rasters (first-spike order) with the descending drive
# (grey, right axis) overlaid. Same input, only the PIC state varies.
fig_r, axes_r = plt.subplots(1, len(labels), figsize=(7.18, 2.8))
for j, (label, ax_r) in enumerate(zip(labels, axes_r)):
    block = results[label]["block"]
    drive, gamma, napf = conditions[label]
    sts = block.segments[0].spiketrains
    active = [u for u in range(len(sts)) if len(sts[u]) > 0]
    order = sorted(active, key=lambda u: float(sts[u].rescale("s").magnitude.min()))
    colors = plt.cm.rainbow(np.linspace(0, 1, max(len(order), 1)))
    n_units = max(len(order), 1)
    # eventplot tick-mark raster (the MyoGen raster style), rainbow by first-spike
    # order; rasterized so the dense ticks embed as a light bitmap.
    ax_r.eventplot([sts[u].rescale("s").magnitude for u in order],
                   lineoffsets=np.arange(len(order)), colors=list(colors),
                   linelengths=0.8, linewidths=0.7, rasterized=True)
    ax_r.set_ylim(-0.8, n_units - 0.2)
    ax_r.set_xlim(0, TOTAL_S)
    ax_r.set_xticks(XTICKS)
    ax_r.set_xlabel("Time (s)")
    ax_r.set_title(label, fontsize=10)
    if j == 0:
        ax_r.set_ylabel("MU (1st-spike order)")

    ax_dr = ax_r.twinx()                    # descending drive overlay (pps)
    ax_dr.spines["right"].set_visible(True)
    ax_dr.tick_params(axis="y", which="both", right=True, labelright=True)
    d_dt = float(drive.sampling_period.rescale(pq.s).magnitude)
    d_t = np.arange(len(drive)) * d_dt
    ax_dr.plot(d_t, np.asarray(drive.magnitude).ravel(), color="0.2",
               linewidth=1.3, alpha=0.7, zorder=6)
    ax_dr.set_ylim(0, 98)
    badge = f"PIC $\\gamma$={gamma}" + (f", NaP$\\times${napf:.0f}" if napf > 1 else "")
    ax_dr.text(0.03, 0.87, badge, transform=ax_dr.transAxes,
               fontsize=9, color=("teal" if napf <= 1 else "crimson"))
    if j == len(labels) - 1:
        ax_dr.set_ylabel("drive (pps)", color="0.2")
    ax_dr.tick_params(axis="y", colors="0.2")
    ax_dr.grid(False)
    mark_spasm_onset(ax_dr, label)
despine_fig(fig_r)
save_fig(fig_r, "sci_mechanistic_raster")

# %%
# Figure 3 -- intramuscular EMG (black) with the ISI CV (purple, right axis)
# overlaid. The green band is the Gorassini 2004 self-sustained CV (5.4 +/-
# 1.6 %); the spasm column collapses into it once the drive withdraws.
fig_e, axes_e = plt.subplots(1, len(labels), figsize=(7.18, 2.5))
for j, (label, ax_e) in enumerate(zip(labels, axes_e)):
    emg = results[label]["iemg"]
    ax_e.plot(emg["times"], emg["iemg"], linewidth=0.12, color="k", zorder=1)
    em = float(np.abs(emg["iemg"]).max())
    ax_e.set_ylim(-em * 1.1, em * 1.1)
    ax_e.set_xlim(0, TOTAL_S)
    ax_e.set_xticks(XTICKS)
    ax_e.set_xlabel("Time (s)")
    ax_e.set_title(label, fontsize=10)
    if j == 0:
        ax_e.set_ylabel("iEMG (a.u.)")
    mark_spasm_onset(ax_e, label)

    ax_c = ax_e.twinx()
    ax_c.spines["right"].set_visible(True)
    ax_c.tick_params(axis="y", which="both", right=True, labelright=True)
    ax_c.set_zorder(ax_e.get_zorder() + 1)
    ax_c.patch.set_visible(False)
    c_cv, cv = results[label]["cv"]
    ax_c.axhspan(GOR_CV - GOR_CV_SD, GOR_CV + GOR_CV_SD, color="green",
                 alpha=0.10, zorder=0)
    ax_c.plot(c_cv, np.ma.masked_invalid(cv), color="purple", linewidth=1.4,
              zorder=3)
    ax_c.set_yscale("log")
    _ylim, _ticks = cv_axis_range(cv)
    ax_c.set_ylim(*_ylim)
    ax_c.set_yticks(_ticks)
    ax_c.yaxis.set_major_formatter(ScalarFormatter())
    ax_c.minorticks_off()
    ax_c.tick_params(axis="y", colors="purple")
    if j == len(labels) - 1:
        ax_c.set_ylabel("ISI CV (%)", color="purple")
    ax_c.grid(False)
despine_fig(fig_e)
save_fig(fig_e, "sci_mechanistic_iemg")
plt.show()
