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

* **Model = Powers2017**, not the manuscript's NERLab. We use it because its
  mAHP + inactivating dendritic Ca PIC put the self-sustained discharge at a
  physiological ~6-8 Hz (Gorassini 2004 ~5.2 Hz); NERLab's plateau floor is
  ~12-16 Hz, too high. This is a deliberate second model for this figure, not a
  silent swap -- a single-model limitation to acknowledge.
* **Low discharge rate, by design.** Powers2017's mAHP caps the firing rate near
  ~6 pps -- below voluntary FDI rates (~12-20 pps). We deliberately do NOT lift
  it: mAHP is exactly what regularises firing, so reducing it (to raise the rate)
  pushes the ISI CV to ~26% and destroys the spasm CV collapse. Because the
  diagnostic spasticity signature is the low ISI CV -- not the rate (Gorassini
  2004 found the self-sustained rate even RISES with added drive) -- we keep
  mAHP intact and accept the low rate. ``mahp_factor`` exposes this trade-off.
* **The PIC is bistable (all-or-nothing).** A gamma sweep shows it does not
  engage below gamma~1.15 and latches at its full ~22 nA dendritic plateau
  at/above it -- there is no "small PIC" self-sustained regime. This is the
  expected regenerative L-type Ca plateau (Lee & Heckman bistability), not a
  tunable knob. The ~22 nA is the *dendritic* current; somatic voltage-clamp
  estimates (~5-15 nA) underread it because of poor space clamp of the distal
  dendrite, so a large dendritic value is consistent with the modelling lit
  (ElBasiouny & Heckman).
* **Firing variability is injected and drive-scaled.** An independent OU
  membrane-noise current per motoneuron (``mn_noise``) models synaptic
  bombardment; its amplitude tracks the descending drive and falls to a small
  intrinsic floor when the drive withdraws. So voluntary firing is irregular
  (CV ~10%) and the drive-off self-sustained spasm -- paced by the intrinsic PIC
  -- is regular (CV ~4%), reproducing the Gorassini (2004) finding that spasms
  fire more regularly than voluntary effort. Without injected noise the PIC
  discharge is near-deterministic (CV <1%, an artifact).
* **No afferent/reflex loops.** The original's third phenotype was 6 Hz *clonus*
  (a stretch-reflex-loop oscillation); here it is replaced by an intrinsic
  PIC-driven spasm. This spasm does **not** self-terminate within the window --
  Powers2017's Ca PIC does not deactivate at these voltages, so a real off-
  switch needs inhibition/afferent input (illustrated separately in the
  single-cell demo), which is outside this open-loop pool.
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

plt.style.use("fivethirtyeight")
plt.rcParams["path.simplify"] = False  # draw every sample on dense EMG traces

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
MN_NOISE = 3.0
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

# Powers2017 motoneuron model: it has the mAHP (Ca-activated K) and an
# inactivating dendritic Ca PIC, so its self-sustained discharge sits at the
# physiological ~6-8 Hz (lit ~5.2 Hz; NERLab's plateau floor is ~12-16 Hz, too
# high). label -> (descending drive, gamma, lambda_factor).
MODEL = "Powers2017"
conditions = {
    "Voluntary modulation (healthy)": (voluntary, 0.5, 1.0),
    "Loss of derecruitment (SCI)": (voluntary, 1.3, 1.0),
    "Modulation -> spasm (SCI)": (spasm_drive, 1.3, 1.0),
}

# Run the three pool simulations once and cache the (slim) results, so figure
# tweaks re-render in seconds. Delete this .pkl after changing any simulation
# parameter (model, N_MU, gamma, lambda, drive, SNR) to force a re-simulation.
results_cache = save_path / f"sci_mechanistic_sims_{MODEL}_n{N_MU}.pkl"
if results_cache.exists():
    results = joblib.load(results_cache)
else:
    results = {}
    for label, (drive, gamma, lam) in conditions.items():
        block = pic.run_pool(drive, n_mu=N_MU, gamma=gamma, model=MODEL,
                             lambda_factor=lam, total_s=TOTAL_S,
                             mn_noise=MN_NOISE, noise_floor=NOISE_FLOOR,
                             dd_weight__uS=DD_WEIGHT, mahp_factor=MAHP_FACTOR)
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


# %%
# Composite figure -- the whole story on one canvas.
#   Top band: single-cell PIC mechanism (Vm + dendritic Ca PIC current).
#   Grid: the SAME descending drive with only the motoneuron PIC varied (columns
#   = healthy / loss of derecruitment / spasm), shown as drive -> raster -> iEMG
#   -> ISI CV (rows). The green band on the CV row is the Gorassini 2004
#   self-sustained-firing CV (5.4 +/- 1.6 %); the spasm column collapses into it
#   once the drive withdraws.
fig = plt.figure(figsize=(13, 13))
gs = fig.add_gridspec(5, 3, height_ratios=[0.9, 0.8, 0.5, 1.8, 1.2],
                      hspace=0.5, wspace=0.46)

# --- top band: single-cell mechanism (spans all 3 columns) ---
ax_mv = fig.add_subplot(gs[0, :])
ax_mi = fig.add_subplot(gs[1, :], sharex=ax_mv)
ax_mv.plot(mech["t"], mech["vm"], color="k", linewidth=0.5)
ax_mv.set_ylabel("Vm (mV)")
ax_mv.set_title("Single-cell PIC mechanism (Powers2017, near bistable threshold): "
                "pulse latches the Ca plateau -> self-sustained firing; "
                "inhibition switches it off", fontsize=10)
ax_mi.plot(mech["t"], mech["pic_nA"], color="red", linewidth=0.8)
ax_mi.set_ylabel("dend Ca PIC (nA)")
ax_mi.set_xlabel("Time (s)")
ax_mi.set_xlim(mech["t"][0], mech["t"][-1])
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

# --- population grid: drive / raster / iEMG(+ISI CV)  x  3 conditions ---
for j, label in enumerate(labels):
    block = results[label]["block"]
    drive, gamma, _lam = conditions[label]
    first = j == 0

    # drive (input command, pps) + PIC-state badge
    ax_d = fig.add_subplot(gs[2, j])
    d_dt = float(drive.sampling_period.rescale(pq.s).magnitude)
    d_t = np.arange(len(drive)) * d_dt
    ax_d.plot(d_t, np.asarray(drive.magnitude).ravel(), color="0.35", linewidth=0.9)
    ax_d.set_ylim(0, 98)
    ax_d.set_title(label, fontsize=10)
    ax_d.text(0.03, 0.80, f"PIC $\\gamma$={gamma}", transform=ax_d.transAxes,
              fontsize=9, color=("teal" if gamma < 1.0 else "crimson"))
    if first:
        ax_d.set_ylabel("drive (pps)")
    mark_spasm_onset(ax_d, label, y_frac=0.9)

    # raster (first-spike-ordered rainbow markers)
    ax_r = fig.add_subplot(gs[3, j], sharex=ax_d)
    sts = block.segments[0].spiketrains
    active = [u for u in range(len(sts)) if len(sts[u]) > 0]
    order = sorted(active, key=lambda u: float(sts[u].rescale("s").magnitude.min()))
    colors = plt.cm.rainbow(np.linspace(0, 1, max(len(order), 1)))
    n_units = max(len(order), 1)
    for rank, u in enumerate(order):
        st = sts[u].rescale("s").magnitude
        ax_r.scatter(st, [rank] * len(st), s=6, facecolor=colors[rank],
                     edgecolors="black", linewidth=0.15, marker="o", alpha=0.85)
    buf = max(1.0, 0.05 * n_units)          # y-buffer so edge markers aren't clipped
    ax_r.set_ylim(-0.5 - buf, n_units - 0.5 + buf)
    if first:
        ax_r.set_ylabel("MU (1st-spike order)")
    mark_spasm_onset(ax_r, label)

    # iEMG (first channel, black) with the ISI CV (purple, log) overlaid on a
    # twin axis; the green band is the Gorassini 2004 self-sustained CV (5.4+-1.6%)
    ax_e = fig.add_subplot(gs[4, j], sharex=ax_d)
    emg = results[label]["iemg"]
    ax_e.plot(emg["times"], emg["iemg"], linewidth=0.12, color="k", zorder=1)
    em = float(np.abs(emg["iemg"]).max())
    ax_e.set_ylim(-em * 1.1, em * 1.1)
    if first:
        ax_e.set_ylabel("iEMG (a.u.)")
    ax_e.set_xlabel("Time (s)")
    ax_e.set_xlim(0, TOTAL_S)
    ax_e.set_xticks(XTICKS)
    mark_spasm_onset(ax_e, label)

    ax_c = ax_e.twinx()
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

fig.savefig(save_path / "sci_mechanistic_composite.svg")
fig.savefig(save_path / "sci_mechanistic_composite.pdf")
plt.show()
