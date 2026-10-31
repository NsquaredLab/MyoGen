"""Scorecard for the dispersion sweep. Definitions fixed before any arm was scored.

Usage: python score.py <limb|fdi> <outdir>   (outdir holds arm_<name>.npz, incl. arm_baseline.npz)

Recording chain: 4th-order zero-phase 20-450 Hz band-pass, no added noise
(noise is handled later). Plateau = 0.5 s after the ramp to 0.5 s before
ramp-down. Signal features: plateau RMS; MNF and MDF of the Welch PSD
(1024-sample segments) within 20-450 Hz; ramp RMS in 250 ms windows centred
where excitation reaches 10 % and 20 %. MUAP features (bipolar templates
before the chain, upsampled x8): peak-to-peak; duration = first to last
sample with |MUAP| >= 10 % of its peak; MNF of the template within 20-450 Hz.
Per-MU changes are paired against the baseline (same MU) before summarising;
size groups are fibre-count terciles of the simulated MUs.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from scipy.signal import butter, resample, sosfiltfilt, welch  # noqa: E402

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
            "FDI: skin thickness 2 and 3 mm": ["baseline_skin2", "full_skin2", "baseline_skin3", "full_skin3"]},
}[SCENARIO]


def chain(x_mV):
    return sosfiltfilt(SOS, np.asarray(x_mV, float) * 1e3)


def band_mnf_mdf(f, p):
    band = (f >= 20) & (f <= 450)
    f, p = f[band], p[band]
    c = np.cumsum(p)
    return float(np.sum(f * p) / np.sum(p)), float(f[np.searchsorted(c, c[-1] / 2)])


def load(path):
    D = dict(np.load(path, allow_pickle=True))
    for key in ("fibre_hash", "spike_hash"):
        assert str(D[key]) == str(B[key]), (path, key)
    assert list(D["mus"]) == list(B["mus"]), path
    return D


t = np.arange(len(B["bip_emg"])) / FS
ramp, hold, level = float(B["ramp"]), float(B["hold"]), float(B["level"])
plateau = (t > 0.5 + ramp + 0.5) & (t < 0.5 + ramp + hold - 0.5)
mus = [int(m) for m in B["mus"]]
nfib = np.array([B["counts"][m] for m in mus])
tercile = np.digitize(nfib, np.quantile(nfib, [1 / 3, 2 / 3]))


def ramp_rms(x, frac):
    tc = 0.5 + ramp * frac / level
    w = np.abs(t - tc) <= 0.125
    return float(np.sqrt(np.mean(x[w] ** 2)))


def muap_features(D):
    m = D["bip_muaps"][:, :, 0, 0] * 1e3  # µV
    fs = float(D["pair_muap_fs"]) * 8
    up = resample(m, m.shape[1] * 8, axis=1)
    ptp = np.ptp(up, axis=1)
    dur, mnf = [], []
    for u in up:
        idx = np.nonzero(np.abs(u) >= 0.1 * np.max(np.abs(u)))[0]
        dur.append((idx[-1] - idx[0]) / fs * 1e3)
        f = np.fft.rfftfreq(len(u) * 4, 1 / fs)
        p = np.abs(np.fft.rfft(u, len(u) * 4)) ** 2
        mnf.append(band_mnf_mdf(f, p)[0])
    return ptp, np.array(dur), np.array(mnf), up


def score(D):
    x = chain(D["bip_emg"][:, 0, 0])
    f, p = welch(x[plateau], fs=FS, nperseg=1024)
    mnf, mdf = band_mnf_mdf(f, p)
    ptp, dur, mmnf, up = muap_features(D)
    r = D["report"].item() if "report" in D and D["report"].size else {}
    pick = (lambda k: float(np.nanmedian(np.asarray(r[k])[mus]))) if r else (lambda k: 0.0)
    return dict(x=x, f=f, p=p, up=up, ptp=ptp, dur=dur, mmnf=mmnf,
                row=dict(rms=float(np.sqrt(np.mean(x[plateau] ** 2))), mnf=mnf, mdf=mdf,
                         rms10=ramp_rms(x, 0.10), rms20=ramp_rms(x, 0.20),
                         zi_sd=pick("zi_sd__mm"), delay_sd=pick("delay_sd__ms"), cv_cov=pick("cv_cov"),
                         amp=pick("amplitude_scale_mean") if r else 1.0))


base = score(B)
results = {"baseline": base}
for name in ARMS:
    path = here / f"arm_{name}.npz"
    if path.exists():
        results[name] = score(load(path))

rows = []
header = ["arm", "realised: end-plate SD mm", "nerve delay SD ms", "fibre CV CoV", "source scale",
          "plateau RMS µV", "Δ%", "MNF Hz", "MDF Hz", "RMS@10% µV", "RMS@20% µV",
          "MUAP PtP Δ% (med, paired)", "small / mid / large", "MUAP dur Δms (med)", "MUAP MNF Δ Hz (med)"]
for name, R in results.items():
    q = R["row"]
    dptp = (R["ptp"] / base["ptp"] - 1) * 100
    groups = " / ".join(f"{np.median(dptp[tercile == g]):+.1f}" for g in range(3))
    rows.append([name, f"{q['zi_sd']:.2f}", f"{q['delay_sd']:.3f}", f"{q['cv_cov']:.3f}", f"{q['amp']:.3f}",
                 f"{q['rms']:.1f}", f"{(q['rms'] / base['row']['rms'] - 1) * 100:+.1f}",
                 f"{q['mnf']:.1f}", f"{q['mdf']:.1f}", f"{q['rms10']:.1f}", f"{q['rms20']:.1f}",
                 f"{np.median(dptp):+.1f}", groups, f"{np.median(R['dur'] - base['dur']):+.2f}",
                 f"{np.median(R['mmnf'] - base['mmnf']):+.1f}"])
(here / "scorecard.tsv").write_text("\n".join("\t".join(r) for r in [header] + rows), encoding="utf-8")
for r in rows:
    print("\t".join(r))

pdf = PdfPages(here / "semg_dispersion_sweep.pdf")
fig = plt.figure(figsize=(16.5, 8.3))
fig.suptitle("Dispersion sweep: bipolar sEMG, 30 % MVC plateau (20–450 Hz band-pass, no noise)", fontsize=13)
ax = fig.add_axes([0.01, 0.12, 0.98, 0.8])
ax.axis("off")
tab = ax.table(cellText=rows, colLabels=header, loc="upper center", cellLoc="center")
tab.auto_set_font_size(False)
tab.set_fontsize(7)
tab.scale(1, 1.5)
fig.text(0.01, 0.02, __doc__.split("\n\n", 2)[2].replace("\n", " "), fontsize=7, wrap=True)
pdf.savefig(fig)
plt.close(fig)

fig, axs = plt.subplots(2, 2, figsize=(11.7, 8.3))
fig.suptitle("Change from the undispersed baseline", fontsize=13)
names = [n for n in results if n != "baseline"]
for ax, (key, label) in zip(axs.flat, (("rms", "plateau RMS (%)"), ("mnf", "MNF (Hz)"),
                                       ("ptp", "MUAP peak-to-peak, median paired (%)"),
                                       ("dur", "MUAP duration, median paired (ms)"))):
    if key == "rms":
        vals = [(results[n]["row"]["rms"] / base["row"]["rms"] - 1) * 100 for n in names]
    elif key == "mnf":
        vals = [results[n]["row"]["mnf"] - base["row"]["mnf"] for n in names]
    elif key == "ptp":
        vals = [np.median((results[n]["ptp"] / base["ptp"] - 1) * 100) for n in names]
    else:
        vals = [np.median(results[n]["dur"] - base["dur"]) for n in names]
    colors = [f"C{next(i for i, g in enumerate(GROUPS.values()) if n in g)}" for n in names]
    ax.barh(names, vals, color=colors)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title(label, fontsize=10)
    ax.tick_params(labelsize=7)
    ax.invert_yaxis()
fig.tight_layout(rect=(0, 0, 1, 0.95))
pdf.savefig(fig)
plt.close(fig)

for gname, members in GROUPS.items():
    members = [m for m in members if m in results]
    if not members:
        continue
    fig = plt.figure(figsize=(11.7, 8.3))
    fig.suptitle(gname, fontsize=13)
    gs = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(gs[0, :2])
    ax.semilogy(base["f"], base["p"], "k", lw=1.5, label="baseline")
    for i, m in enumerate(members):
        ax.semilogy(results[m]["f"], results[m]["p"], color=f"C{i}", lw=1, label=m)
    ax.set_xlim(0, 500)
    ax.set_xlabel("Hz")
    ax.set_ylabel("PSD µV²/Hz")
    ax.legend(fontsize=7)
    ax = fig.add_subplot(gs[0, 2])
    z = (t >= 4.0) & (t <= 4.15)
    ax.plot(t[z], base["x"][z], "k", lw=1, label="baseline")
    ax.plot(t[z], results[members[-1]]["x"][z], color=f"C{len(members) - 1}", lw=0.8, label=members[-1])
    ax.set_xlabel("s")
    ax.set_ylabel("µV")
    ax.legend(fontsize=7)
    for col, g in enumerate(range(3)):
        ax = fig.add_subplot(gs[1, col])
        k = int(np.nonzero(tercile == g)[0][np.argsort(nfib[tercile == g])[np.sum(tercile == g) // 2]])
        tm = (np.arange(base["up"].shape[1]) - base["up"].shape[1] // 2) / (float(B["pair_muap_fs"]) * 8) * 1e3
        ax.plot(tm, base["up"][k], "k", lw=1.5, label="baseline")
        for i, m in enumerate(members):
            ax.plot(tm, results[m]["up"][k], color=f"C{i}", lw=1, label=m)
        ax.set_xlim(-3, 25)
        ax.set_title(f"MU {mus[k]} ({nfib[k]} fibres)", fontsize=9)
        ax.set_xlabel("ms from firing")
        ax.set_ylabel("µV")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    pdf.savefig(fig)
    plt.close(fig)
pdf.close()
