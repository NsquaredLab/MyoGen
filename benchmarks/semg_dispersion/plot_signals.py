"""Signal and MUAP views of the dispersion sweep, for judging the changes by eye.

Usage: python plot_signals.py <limb|fdi> <outdir>   (outdir holds arm_<name>.npz, incl. arm_baseline.npz)
Same recording chain as score_disp.py (20-450 Hz band-pass, no noise).
Per factor: one page of EMG (whole recording, 1 s plateau, 100 ms plateau
zoom, 250 ms at ~10 % effort) and one page of MUAPs (8 MUs, shared scale per MU).
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from scipy.signal import butter, resample, sosfiltfilt  # noqa: E402

SCENARIO, here = sys.argv[1], Path(sys.argv[2])
base_path = here / "arm_baseline.npz"
ARMS = [a for a in json.loads((Path(__file__).with_name("arms.json")).read_text())[SCENARIO] if a != "baseline"]
B = dict(np.load(base_path, allow_pickle=True))
FS = float(B["fs"])
SOS = butter(4, [20, 450], btype="bandpass", fs=FS, output="sos")
GROUPS = {
    "limb": {"Nerve-branch delays": ["nerve_cv_10_4", "nerve_cv_5_2", "nerve_cv_2.5_1"],
             "End-plate scatter": ["nmj_intra_x0.5", "nmj_intra_x1", "nmj_intra_x2", "nmj_two_layer_x1"],
             "Muscle-fibre CV": ["fcv_within_5", "fcv_within_10", "fcv_inter_30", "fcv_both", "fcv_both_diameter"],
             "Combined": ["combined", "combined_diameter"],
             "Literature-anchored candidate": ["candidate", "candidate_within5"],
             "End-plate coefficients: code vs published": ["candidate_within5", "candidate_published"],
             "Tendon-end band (on the published candidate)": ["candidate_published", "tendon_w5", "tendon_w10"],
             "Correlated tendon ends (5 mm band, 0.3 mm jitter)": ["tendon_w5", "tendon_w5_corr2", "tendon_w5_corr5",
                                                                   "tendon_w5_corr10"],
             "All effects (+ diameter-scaled source)": ["tendon_w5_corr5", "full_diameter"]},
    "fdi": {"FDI: published candidate and tendon ends": ["candidate_published", "tendon_w5", "tendon_w5_corr5",
                                                         "tendon_w10"],
            "FDI: all effects (+ diameter-scaled source)": ["tendon_w5_corr5", "full_diameter"],
            "FDI: skin thickness 2 and 3 mm": ["baseline_skin2", "full_skin2", "baseline_skin3", "full_skin3"],
            "FDI: literature layers, undispersed (skin 1.2 mm)": [
                "baseline_s1.2_f1.3_sig1", "baseline_s1.2_f0.3_sig0.15", "baseline_s1.2_f1.3_sig0.15",
                "baseline_s1.2_f2.0_sig0.15"],
            "FDI: literature layers, all effects (skin 1.2 mm)": [
                "full_s1.2_f1.3_sig1", "full_s1.2_f0.3_sig0.15", "full_s1.2_f1.3_sig0.15", "full_s1.2_f2.0_sig0.15"]},
}[SCENARIO]
LABELS = {"nerve_cv_10_4": "nerve CV 10/4 m/s", "nerve_cv_5_2": "nerve CV 5/2 m/s (iEMG default)",
          "nerve_cv_2.5_1": "nerve CV 2.5/1 m/s (stress)", "nmj_intra_x0.5": "end-plate SD 0.6 mm",
          "nmj_intra_x1": "end-plate SD 1.3 mm", "nmj_intra_x2": "end-plate SD 2.6 mm",
          "nmj_two_layer_x1": "two-layer (MU centre moves too)", "fcv_within_5": "fibre CV CoV 5 % in MU",
          "fcv_within_10": "fibre CV CoV 10 % in MU", "fcv_inter_30": "MU CV ±15 % by recruitment",
          "fcv_both": "inter ±15 % + within 5 %", "fcv_both_diameter": "inter + within + diameter source",
          "combined": "two-layer + nerve 5/2 + fibre CV", "combined_diameter": "combined + diameter source",
          "candidate": "candidate",
          "candidate_within5": "cand. +5 % (code)",
          "candidate_published": "cand. +5 % (published)",
          "tendon_w5": "independent ends, 5 mm band", "tendon_w10": "independent ends, 10 mm band",
          "tendon_w5_corr2": "5 mm band, correlated over 2 mm", "tendon_w5_corr5": "5 mm band, correlated over 5 mm",
          "tendon_w5_corr10": "5 mm band, correlated over 10 mm",
          "full_diameter": "all effects + diameter-scaled source", "baseline_skin2": "undispersed, skin 2 mm",
          "full_skin2": "all effects, skin 2 mm", "baseline_skin3": "undispersed, skin 3 mm",
          "full_skin3": "all effects, skin 3 mm",
          "baseline_s1.2_f1.3_sig1": "undispersed, fat 1.3 mm, skin σ 1 S/m",
          "full_s1.2_f1.3_sig1": "all effects, fat 1.3 mm, skin σ 1 S/m",
          "baseline_s1.2_f1.3_sig0.15": "undispersed, fat 1.3 mm, skin σ 0.15 S/m",
          "full_s1.2_f1.3_sig0.15": "all effects, fat 1.3 mm, skin σ 0.15 S/m",
          "baseline_s1.2_f0.3_sig0.15": "undispersed, fat 0.3 mm, skin σ 0.15 S/m",
          "full_s1.2_f0.3_sig0.15": "all effects, fat 0.3 mm, skin σ 0.15 S/m",
          "baseline_s1.2_f2.0_sig0.15": "undispersed, fat 2.0 mm, skin σ 0.15 S/m",
          "full_s1.2_f2.0_sig0.15": "all effects, fat 2.0 mm, skin σ 0.15 S/m"}


def chain(x_mV):
    return sosfiltfilt(SOS, np.asarray(x_mV, float) * 1e3)


def load(name):
    D = B if name == "baseline" else dict(np.load(here / f"arm_{name}.npz", allow_pickle=True))
    for key in ("fibre_hash", "spike_hash"):
        assert str(D[key]) == str(B[key]), (name, key)
    m = D["bip_muaps"][:, :, 0, 0] * 1e3
    return chain(D["bip_emg"][:, 0, 0]), resample(m, m.shape[1] * 8, axis=1)


t = np.arange(len(B["bip_emg"])) / FS
ramp, hold, level = float(B["ramp"]), float(B["hold"]), float(B["level"])
t10 = 0.5 + ramp * 0.10 / level
mus = [int(m) for m in B["mus"]]
nfib = np.array([B["counts"][m] for m in mus])
x0, u0 = load("baseline")
tm = (np.arange(u0.shape[1]) - u0.shape[1] // 2) / (float(B["pair_muap_fs"]) * 8) * 1e3
by_size = np.argsort(nfib)
pick = [int(by_size[int(q * (len(mus) - 1))]) for q in (0.0, 0.2, 0.4, 0.6, 0.8)]
for k in np.argsort(np.ptp(u0, axis=1))[::-1]:
    if len(pick) == 8:
        break
    if k not in pick:
        pick.append(int(k))
pick.sort(key=lambda k: nfib[k])

WINDOWS = [("whole recording", 0, t[-1], False), ("1 s of plateau", 4.0, 5.0, False),
           ("100 ms of plateau (grey = baseline)", 4.30, 4.40, True),
           ("250 ms at ~10 % effort (grey = baseline)", t10 - 0.125, t10 + 0.125, True)]

pdf = PdfPages(here / "semg_dispersion_signals.pdf")
for gname, members in GROUPS.items():
    names = ["baseline"] + members
    data = {n: load(n) for n in names}
    lim = 1.05 * max(np.max(np.abs(data[n][0])) for n in names)

    fig, axs = plt.subplots(len(names), 4, figsize=(16.5, 2.0 + 1.55 * len(names)),
                            gridspec_kw=dict(width_ratios=[1.4, 1.4, 1, 1]), squeeze=False)
    fig.suptitle(f"{gname}: bipolar sEMG, 30 % MVC trapezoid (20–450 Hz, no noise; same y-scale in every panel"
                 " of a column)", fontsize=12)
    for row, n in enumerate(names):
        x = data[n][0]
        color = "k" if n == "baseline" else f"C{row - 1}"
        for col, (title, a, b, overlay) in enumerate(WINDOWS):
            ax = axs[row, col]
            w = (t >= a) & (t <= b)
            if overlay and n != "baseline":
                ax.plot(t[w], x0[w], color="0.7", lw=1.4)
            ax.plot(t[w], x[w], color=color, lw=0.6 if col < 2 else 1.0)
            ylim = lim if col != 3 else 1.05 * max(np.max(np.abs(data[m][0][w])) for m in names)
            ax.set_ylim(-ylim, ylim)
            ax.set_xlim(a, b)
            ax.tick_params(labelsize=7)
            if row == 0:
                ax.set_title(title, fontsize=9)
            if col == 0:
                rms = np.sqrt(np.mean(x[(t > 3.0) & (t < 6.0)] ** 2))
                ax.set_ylabel(f"{LABELS.get(n, n)}\nRMS {rms:.0f} µV", fontsize=8)
            if row == len(names) - 1:
                ax.set_xlabel("s", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig)
    plt.close(fig)

    fig, axs = plt.subplots(2, 4, figsize=(16.5, 8.3))
    fig.suptitle(f"{gname}: MUAPs (bipolar, before band-pass). Black = baseline. Ordered by MU size.", fontsize=12)
    for ax, k in zip(axs.flat, pick):
        ax.plot(tm, u0[k], "k", lw=2.0, label="baseline")
        for i, n in enumerate(members):
            ax.plot(tm, data[n][1][k], color=f"C{i}", lw=1.1, label=LABELS.get(n, n))
        ax.axhline(0, color="0.8", lw=0.6)
        ax.set_xlim(-2, 18)
        ax.set_title(f"MU {mus[k]}: {nfib[k]} fibres, baseline PtP {np.ptp(u0[k]):.0f} µV", fontsize=9)
        ax.set_xlabel("ms from firing", fontsize=8)
        ax.set_ylabel("µV", fontsize=8)
        ax.tick_params(labelsize=7)
    axs.flat[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    pdf.savefig(fig)
    plt.close(fig)
pdf.close()
print("MUs shown:", [mus[k] for k in pick])
