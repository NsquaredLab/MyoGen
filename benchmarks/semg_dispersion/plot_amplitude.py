"""Amplitude with every effect explored, limb scenario and default FDI cylinder.

Usage: python plot_amplitude.py <limb_outdir> <fdi_outdir> <out.pdf>
Arms: undispersed baseline; published candidate (two-layer end-plates, nerve 5/2 m/s,
threshold-linear MU CV, 5 % within-MU CV); + correlated tendon ends (5 mm band,
5 mm correlation, 0.3 mm jitter); + diameter-scaled source. 20-450 Hz band-pass, no noise.
FDI literature values are drawn for orientation only: their electrodes differ from ours
and our effort axis is excitation, not force.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from scipy.signal import butter, resample, sosfiltfilt, welch  # noqa: E402

LIMB, FDI, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
SCENARIOS = [
    ("Limb-like cylinder (r 15 mm, 70 mm fibres, fat 4 mm; bipolar 5 mm contacts, IED 10 mm)",
     LIMB / "arm_baseline.npz", LIMB, False),
    ("Default FDI cylinder (r 6.9 mm, 31.7 mm fibres, fat 0.3 mm; bipolar 2 mm contacts, IED 5 mm)",
     FDI / "arm_baseline.npz", FDI, True),
]
ARMS = [("baseline", "undispersed (fixes only)"), ("candidate_published", "published candidate"),
        ("tendon_w5_corr5", "+ correlated tendon ends"), ("full_diameter", "+ diameter-scaled source")]
COLORS = ["k", "C0", "C2", "C3"]


def load(base, here, name):
    D = dict(np.load(base if name == "baseline" else here / f"arm_{name}.npz", allow_pickle=True))
    B = dict(np.load(base, allow_pickle=True))
    for key in ("fibre_hash", "spike_hash"):
        assert str(D[key]) == str(B[key]), (name, key)
    return D


pdf = PdfPages(OUT)
rows = []
for title, base, here, fdi in SCENARIOS:
    data = {n: load(base, here, n) for n, _ in ARMS}
    B = data["baseline"]
    fs = float(B["fs"])
    sos = butter(4, [20, 450], btype="bandpass", fs=fs, output="sos")
    ramp, hold, level = float(B["ramp"]), float(B["hold"]), float(B["level"])
    t = np.arange(len(B["bip_emg"])) / fs
    plateau = (t > 0.5 + ramp + 0.5) & (t < 0.5 + ramp + hold - 0.5)
    nfib = np.array([B["counts"][m] for m in B["mus"]])

    fig = plt.figure(figsize=(16.5, 9.5))
    fig.suptitle(f"Amplitude with all explored effects: {title}\n30 % excitation trapezoid, 20–450 Hz, no noise",
                 fontsize=11)
    gs = fig.add_gridspec(4, 3, width_ratios=[1.3, 1, 1])
    lim = 0
    traces = {}
    for name, _ in ARMS:
        traces[name] = sosfiltfilt(sos, data[name]["bip_emg"][:, 0, 0] * 1.0)  # mV
        lim = max(lim, np.max(np.abs(traces[name][(t >= 4.3) & (t <= 4.5)])))
    for i, (name, label) in enumerate(ARMS):
        ax = fig.add_subplot(gs[i, 0])
        w = (t >= 4.3) & (t <= 4.5)
        ax.plot(t[w], traces[name][w], color=COLORS[i], lw=0.8)
        rms = np.sqrt(np.mean(traces[name][plateau] ** 2))
        ax.set_ylim(-1.05 * lim, 1.05 * lim)
        ax.set_ylabel(f"{label}\nRMS {rms:.3f} mV", fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.set_title("200 ms of plateau (mV, same scale)", fontsize=9)
        if i == len(ARMS) - 1:
            ax.set_xlabel("s", fontsize=8)

    ax = fig.add_subplot(gs[:2, 1])
    centres = np.arange(0.7, 0.5 + ramp + 1.0, 0.05)
    for i, (name, label) in enumerate(ARMS):
        r = [np.sqrt(np.mean(traces[name][np.abs(t - c) <= 0.125] ** 2)) for c in centres]
        exc = np.clip((centres - 0.5) / ramp, 0, 1) * level * 100
        ax.plot(exc, r, color=COLORS[i], lw=1.4, label=label)
    if fdi:
        ax.errorbar([25], [0.21], yerr=[[0.08], [0.08]], fmt="s", color="m", capsize=4,
                    label="Botelho 2019: 0.21 mV at 25 %MVC\n(1 subject, Delsys 5-pin)")
    ax.set_xlabel("excitation (%)", fontsize=8)
    ax.set_ylabel("RMS, 250 ms windows (mV)", fontsize=8)
    ax.set_title("RMS along the ramp", fontsize=9)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

    ax = fig.add_subplot(gs[2:, 1])
    for i, (name, label) in enumerate(ARMS):
        f, p = welch(traces[name][plateau], fs=fs, nperseg=1024)
        ax.semilogy(f, p, color=COLORS[i], lw=1.0, label=label)
    ax.set_xlim(0, 500)
    ax.set_xlabel("Hz", fontsize=8)
    ax.set_ylabel("PSD (mV²/Hz)", fontsize=8)
    ax.set_title("Plateau spectrum", fontsize=9)
    ax.tick_params(labelsize=7)

    ax = fig.add_subplot(gs[:3, 2])
    for i, (name, label) in enumerate(ARMS):
        m = data[name]["bip_muaps"][:, :, 0, 0]
        ptp = np.ptp(resample(m, m.shape[1] * 8, axis=1), axis=1)
        ax.scatter(nfib, ptp, s=9, color=COLORS[i], alpha=0.7, label=label)
        f, p = welch(traces[name][plateau], fs=fs, nperseg=1024)
        band = (f >= 20) & (f <= 450)
        mnf = np.sum(f[band] * p[band]) / np.sum(p[band])
        rows.append([title.split(" (")[0], label, f"{np.sqrt(np.mean(traces[name][plateau] ** 2)):.3f}",
                     f"{mnf:.0f}", f"{np.median(ptp):.3f}", f"{np.percentile(ptp, 90):.2f}", f"{ptp.max():.2f}"])
    if fdi:
        ax.axhspan(0.05, 2.0, color="m", alpha=0.08,
                   label="published single-MU PtP ≈0.05–2 mV\n(belly–tendon / Delsys; for orientation)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("fibres in MU", fontsize=8)
    ax.set_ylabel("MUAP peak-to-peak (mV)", fontsize=8)
    ax.set_title("MUAP amplitude by MU size", fontsize=9)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

    ax = fig.add_subplot(gs[3, 2])
    ax.axis("off")
    mine = [r[1:] for r in rows if r[0] == title.split(" (")[0]]
    tab = ax.table(cellText=mine, colLabels=["arm", "plateau RMS mV", "MNF Hz", "MUAP PtP median mV",
                                             "p90 mV", "max mV"], loc="center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(7)
    tab.scale(1, 1.3)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    pdf.savefig(fig)
    fig.savefig(OUT.with_name(f"amplitude_{'fdi' if fdi else 'limb'}.png"), dpi=100)
    plt.close(fig)
pdf.close()
for r in rows:
    print("\t".join(r))
