"""
SCI pathological iEMG, mechanistically: same model, vary only the PIC
====================================================================

Instead of hand-sculpting the descending drive, the three discharge phenotypes
emerge from the motoneuron persistent inward current (PIC). All panels share the
same muscle, electrode, and motoneuron pool:

1. Voluntary modulation -- healthy PIC (gamma=0.5, NaPx1); a 0.5 Hz command
   recruits and derecruits the pool each cycle.
2. Loss of derecruitment -- up-regulated PIC (gamma=1.3, NaPx5); the same command
   no longer lets units fall silent at the troughs.
3. Modulation -> spasm -- up-regulated PIC; the command stops half-way but the
   PIC sustains an involuntary discharge.

Model: NERLab (Cisi-Kohn-Elias). Self-sustained firing needs both the dendritic
Ca PIC (gamma) and somatic NaP (nap_factor). Firing variability is drive-driven:
voluntary firing is irregular (ISI CV ~19%) from the noisy drive plus a small
drive-scaled OU membrane noise (``mn_noise``); when the drive withdraws the
intrinsic PIC paces firing and the CV collapses to ~7% (cf. Gorassini 2004). The
Ca PIC does not deactivate, so the spasm does not self-terminate within the window
(a real off-switch needs inhibition; see the single-cell panel).
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
import _pic_protocols as pic

def apply_pub_style():
    """Publication style (scienceplots science+nature), matching the MyoGen
    ISI-CV figures -- NOT fivethirtyeight. Clean white background, sans-serif,
    thick axes, top/right spines off (re-enabled per twin axis). Importing
    myogen applies the seaborn lavender theme as a side effect, so this is
    re-asserted right before the figure is built, not only at import time."""
    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", "nature"])
        sns.set_context("paper", font_scale=0.7)   # 7.18 in panels are small
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
# Peak OU membrane-noise current (nA) per motoneuron, scaled by the drive (falls
# to MN_NOISE*NOISE_FLOOR when the drive withdraws). Calibrated so voluntary firing
# is irregular (ISI CV ~19%) while the drive-off self-sustained spasm stays regular
# (~7%, cf. Gorassini 2004).
MN_NOISE = 2.0
# Fraction of MN_NOISE that remains once the command withdraws. Kept low so the
# drive-off self-sustained spasm noise stays ~0.12 nA (regular firing, ~7% CV)
# even though the drive-on noise is now large (2.0 nA, irregular voluntary firing).
NOISE_FLOOR = 0.06
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

# NERLab (Cisi-Kohn-Elias) motoneuron, the MyoGen manuscript's model. SCI
# spasticity is induced by up-regulating the PIC: dendritic Ca (``gamma``) plus
# somatic NaP (``nap_factor``). label -> (drive, gamma, nap_factor).
MODEL = "NERLab"
conditions = {
    "Voluntary modulation (healthy)": (voluntary, 0.5, 1.0),
    "Loss of derecruitment (SCI)": (voluntary, 1.3, 5.0),
    "Modulation -> spasm (SCI)": (spasm_drive, 1.3, 5.0),
}

# Run the three pool simulations once and cache the (slim) results, so figure
# tweaks re-render in seconds. Delete this .pkl after changing any simulation
# parameter (N_MU, gamma, nap_factor, drive, SNR) to force a re-simulation.
results_cache = save_path / f"sci_mechanistic_sims_{MODEL}_n{N_MU}.pkl"
if results_cache.exists():
    results = joblib.load(results_cache)
else:
    results = {}
    for label, (drive, gamma, napf) in conditions.items():
        block = pic.run_pool(drive, n_mu=N_MU, gamma=gamma,
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

# ISI CV is only meaningful over a quasi-stationary (constant-rate) segment, so we
# compute the drift-insensitive CV2 per unit only inside the discharge plateaus
# (drive within `PLATEAU_FRAC` of peak) and, for the spasm, over the steady
# self-sustained discharge, then report the median across units.
EPOCH_MARGIN = 0.3     # s, keeps the rest epoch off the command-stop transient
PLATEAU_FRAC = 0.75    # plateau = drive > 75 % of its peak (near-stationary)


def plateau_intervals(drive, frac=PLATEAU_FRAC, t_lo=0.0, t_hi=TOTAL_S,
                      smooth_s=0.1, min_dur_s=0.1):
    """Contiguous (t0, t1) windows where the drive ENVELOPE exceeds `frac` of its
    peak and lies within [t_lo, t_hi) -- the near-stationary plateau of each cycle.
    The command carries synaptic noise, so the threshold is applied to a
    `smooth_s` moving-average envelope and runs shorter than `min_dur_s` (noise
    flicker) are discarded."""
    dt = float(drive.sampling_period.rescale(pq.s).magnitude)
    t = np.arange(len(drive)) * dt
    d = np.asarray(drive.magnitude).ravel().astype(float)
    w = max(1, int(round(smooth_s / dt)))
    env = np.convolve(d, np.ones(w) / w, mode="same")     # command envelope
    mask = (env > frac * env.max()) & (t >= t_lo) & (t < t_hi)
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            if t[j - 1] - t[i] >= min_dur_s:              # drop noise flicker
                out.append((float(t[i]), float(t[j - 1])))
            i = j
        else:
            i += 1
    return out


def per_mu_cv2(block, intervals, max_isi=0.3, min_isi=3):
    """Per-unit CV2 (drift-insensitive) pooling ISIs across the plateau
    `intervals`, as an array over units. Consecutive-ISI pairs are taken WITHIN an
    interval only; ISIs longer than `max_isi` (gaps) are dropped, so each value is
    a within-plateau (stationary) estimate."""
    vals = []
    for st in block.segments[0].spiketrains:
        s = np.sort(st.rescale("s").magnitude)
        num, cnt, n_isi = 0.0, 0, 0
        for a0, a1 in intervals:
            ss = s[(s >= a0) & (s < a1)]
            if len(ss) < 2:
                continue
            isi = np.diff(ss)
            isi = isi[isi <= max_isi]
            n_isi += len(isi)
            if len(isi) >= 2:
                a, b = isi[:-1], isi[1:]
                num += float(np.sum(2.0 * np.abs(b - a) / (a + b)))
                cnt += len(a)
        if n_isi >= min_isi and cnt > 0:
            vals.append(100.0 * num / cnt)
    return np.asarray(vals)


for _label, _data in results.items():
    _data["rate"] = pic.population_rate(_data["block"], TOTAL_S, win_s=0.8,
                                        step_s=0.02)  # 0.8 s window, 20 ms steps
    _drive = conditions[_label][0]
    if _label == SPASM_LABEL:
        # one segment per modulation-cycle plateau while the command is on, then
        # the steady self-sustained discharge (one block) after it stops.
        _segments = plateau_intervals(_drive, t_lo=0.0, t_hi=SPASM_ONSET_S) + \
            [(SPASM_ONSET_S + EPOCH_MARGIN, TOTAL_S - EPOCH_MARGIN)]
    else:
        _segments = plateau_intervals(_drive)
    # CV2 is computed WITHIN each cycle separately, so each marker shows that
    # cycle's own median across units (cycle-to-cycle variation is preserved, not
    # pooled into a single value).
    _ec = []
    for _t0, _t1 in _segments:
        _v = per_mu_cv2(_data["block"], [(_t0, _t1)])
        if _v.size:
            _ec.append((_t0, _t1, float(np.median(_v))))
    _data["epoch_cv"] = _ec

labels = list(conditions.keys())

def mark_spasm_onset(ax, label, y_frac=0.97):
    """On the modulation->spasm panel, draw a dashed vertical line at the moment
    the voluntary command stops and the PIC-sustained spasm begins (line only --
    no text label; annotate in post if needed)."""
    if label != SPASM_LABEL:
        return
    ax.axvline(SPASM_ONSET_S, color="0.25", linestyle="--", linewidth=1.2,
               zorder=6)


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
    def to_pps(y):
        return rmin + (y / span_max) * (rmax - rmin)
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
# Single-cell PIC mechanism (cached): a NERLab cell near the bistable
# threshold -- a brief pulse latches the dendritic Ca plateau (self-sustained
# firing) and a gentle inhibition switches it off. The cell-level basis of the
# pool spasm.
mech_cache = save_path / "sci_mechanistic_singlecell.pkl"
if mech_cache.exists():
    mech = joblib.load(mech_cache)
else:
    mech = pic.single_cell_pic_mechanism()
    joblib.dump(mech, mech_cache)

def despine_fig(fig):
    """sns.despine top+right on every axis (twin axes keep their right tick
    labels so the secondary scale still reads without a spine line)."""
    for ax in fig.axes:
        is_twin = ax.spines["right"].get_visible()
        sns.despine(ax=ax, top=True, right=True, offset=5, trim=not is_twin)
        if is_twin:
            ax.tick_params(axis="y", which="both", right=True, labelright=True)


RASTER_DPI = 1200   # resolution of the rasterized=True artists (dense ticks/traces)


def save_fig(fig, stem):
    fig.tight_layout()
    fig.savefig(save_path / f"{stem}.svg", bbox_inches="tight", dpi=RASTER_DPI)
    fig.savefig(save_path / f"{stem}.pdf", bbox_inches="tight", dpi=RASTER_DPI)


# The three figures share the SAME descending drive; only the motoneuron PIC
# state varies across conditions (columns). Each is its own figure with its own
# size, so panels can be assembled freely.
apply_pub_style()                       # re-assert (myogen imports re-theme sns)

# %%
# Figure 1 -- single-cell PIC mechanism: Vm and the up-regulated inward currents.
# figure width 7.18 in = 180 mm (Nature double-column); heights per figure.
fig_m, (ax_mv, ax_mi) = plt.subplots(2, 1, figsize=(7.18, 0.75), sharex=True)
ax_mv.plot(mech["t"], mech["vm"], color="k", linewidth=0.5, rasterized=True)
ax_mv.set_ylabel("Vm (mV)")
ax_mv.set_title("Single-cell PIC mechanism (NERLab)")
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
ax_mi.legend(loc="center right", frameon=False)
for axm in (ax_mv, ax_mi):
    axm.axvspan(*mech["t_pulse"], color="0.85", zorder=0)
    axm.axvspan(*mech["t_inhib"], color="#cfe0ff", zorder=0)
    axm.grid(True, alpha=0.3)
ax_mv.annotate("pulse", xy=(np.mean(mech["t_pulse"]), 0.96),
               xycoords=("data", "axes fraction"), ha="center", va="top",
               color="0.3")
ax_mv.annotate("inhibition", xy=(np.mean(mech["t_inhib"]), 0.96),
               xycoords=("data", "axes fraction"), ha="center", va="top",
               color="#2b5fb0")
despine_fig(fig_m)
save_fig(fig_m, "sci_mechanistic_mechanism")

# %%
# Figure 2 -- motor-unit rasters (first-spike order) with the descending drive
# (grey, right axis) overlaid. Same input, only the PIC state varies.
fig_r, axes_r = plt.subplots(1, len(labels), figsize=(7.18, 1.4))
for j, (label, ax_r) in enumerate(zip(labels, axes_r)):
    block = results[label]["block"]
    drive, gamma, napf = conditions[label]
    sts = block.segments[0].spiketrains
    active = [u for u in range(len(sts)) if len(sts[u]) > 0]
    order = sorted(active, key=lambda u: float(sts[u].rescale("s").magnitude.min()))
    colors = plt.cm.rainbow(np.linspace(0, 1, max(len(order), 1)))
    n_units = max(len(order), 1)
    # eventplot tick-mark raster (the MyoGen raster style), rainbow by first-spike
    # order. Rasterized (too many spikes for vector), at high dpi so the dense
    # thin ticks don't alias/moire.
    ax_r.eventplot([sts[u].rescale("s").magnitude for u in order],
                   lineoffsets=np.arange(len(order)), colors=list(colors),
                   linelengths=0.8, linewidths=0.7, rasterized=True)
    ax_r.set_ylim(-0.8, n_units - 0.2)
    ax_r.set_xlim(0, TOTAL_S)
    ax_r.set_xticks(XTICKS)
    ax_r.set_xlabel("Time (s)")
    ax_r.set_title(label)
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
    if j == len(labels) - 1:
        ax_dr.set_ylabel("drive (pps)", color="0.2")
    ax_dr.tick_params(axis="y", colors="0.2")
    ax_dr.grid(False)
    mark_spasm_onset(ax_dr, label)
despine_fig(fig_r)
save_fig(fig_r, "sci_mechanistic_raster")

# %%
# Figure 3 -- intramuscular EMG (black) with the per-epoch ISI CV (purple, right
# axis): one value for the active modulation epoch and, for the spasm column,
# one for the quiet self-sustained epoch (which collapses to a regular ~7 %).
fig_e, axes_e = plt.subplots(1, len(labels), figsize=(7.18, 1.25))
# shared iEMG y-axis so amplitudes are comparable across conditions
em_global = max(float(np.abs(results[lab]["iemg"]["iemg"]).max()) for lab in labels)
for j, (label, ax_e) in enumerate(zip(labels, axes_e)):
    emg = results[label]["iemg"]
    ax_e.plot(emg["times"], emg["iemg"], linewidth=0.12, color="k", zorder=1)
    ax_e.set_ylim(-em_global * 1.1, em_global * 1.1)
    ax_e.set_xlim(0, TOTAL_S)
    ax_e.set_xticks(XTICKS)
    ax_e.set_xlabel("Time (s)")
    ax_e.set_title(label)
    if j == 0:
        ax_e.set_ylabel("iEMG (a.u.)")
    mark_spasm_onset(ax_e, label)

    ax_c = ax_e.twinx()
    ax_c.spines["right"].set_visible(True)
    ax_c.tick_params(axis="y", which="both", right=True, labelright=True)
    ax_c.set_zorder(ax_e.get_zorder() + 1)
    ax_c.patch.set_visible(False)
    for t0, t1, med in results[label]["epoch_cv"]:   # one marker per cycle, true median
        ax_c.plot([t0, t1], [med, med], color="purple", linewidth=2.4,
                  solid_capstyle="round", zorder=3)
        ax_c.annotate(f"{med:.0f}%", xy=(0.5 * (t0 + t1), med), xytext=(0, 3),
                      textcoords="offset points", ha="center", va="bottom",
                      fontsize=5.5, color="purple")
    ax_c.set_yscale("log")
    ax_c.set_ylim(3.0, 30.0)
    ax_c.set_yticks([5, 10, 20])
    ax_c.yaxis.set_major_formatter(ScalarFormatter())
    ax_c.minorticks_off()
    ax_c.tick_params(axis="y", colors="purple")
    if j == len(labels) - 1:
        ax_c.set_ylabel("ISI CV (%)", color="purple")
    ax_c.grid(False)
despine_fig(fig_e)
save_fig(fig_e, "sci_mechanistic_iemg")

# %%
# Figure 4 -- mean per-unit discharge rate over time (shared y so conditions are
# comparable). Voluntary modulation ramps up/down; loss of derecruitment stays
# elevated at the drive troughs; the spasm rate drops after the command stops
# (t=4 s) but does NOT go to zero -- the PIC-sustained discharge (cf. Gorassini
# 2004, who reports self-sustained firing rate).
fig_d, axes_d = plt.subplots(1, len(labels), figsize=(7.18, 2.0))
r_max = max(float(results[lab]["rate"][1].max()) for lab in labels)
for j, (label, ax_d) in enumerate(zip(labels, axes_d)):
    centers, rate = results[label]["rate"]
    ax_d.plot(centers, rate, color="0.15", linewidth=1.0)
    ax_d.set_ylim(0, r_max * 1.05)
    ax_d.set_xlim(0, TOTAL_S)
    ax_d.set_xticks(XTICKS)
    ax_d.set_xlabel("Time (s)")
    ax_d.set_title(label)
    if j == 0:
        ax_d.set_ylabel("discharge rate (pps)")
    mark_spasm_onset(ax_d, label)
despine_fig(fig_d)
save_fig(fig_d, "sci_mechanistic_rate")
plt.show()
