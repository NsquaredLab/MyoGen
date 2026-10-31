"""Finished sEMG recordings (old = main, new = fixed) through one recording chain."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from scipy.signal import butter, sosfiltfilt, welch  # noqa: E402

here = Path(sys.argv[1])
O = dict(np.load(here / "old.npz", allow_pickle=True))
N = dict(np.load(here / "new.npz", allow_pickle=True))
for key in ("fibre_hash", "spike_hash"):
    assert str(O[key]) == str(N[key]), key
assert list(O["mus"]) == list(N["mus"])
FS = float(N["fs"])
CO, CN = "#c0392b", "#1f5fa8"
NOISE_UV = 3.0
SOS = butter(4, [20, 450], btype="bandpass", fs=FS, output="sos")


def finish(x_mV, seed):
    """Amplifier noise (same realisation for both versions) + 20-450 Hz band-pass; returns uV."""
    noise = np.random.default_rng(seed).normal(0, NOISE_UV, len(x_mV))
    return sosfiltfilt(SOS, x_mV * 1e3 + noise)


def channels(D):
    pair = D["pair_emg"][:, :, 0]
    return dict(bip=finish(D["bip_emg"][:, 0, 0], 1), diff=finish(pair[:, 1] - pair[:, 0], 1),
                mono=finish(pair[:, 0], 2))


CO_, CN_ = channels(O), channels(N)
t = np.arange(len(CN_["bip"])) / FS
ramp, hold = float(N["ramp"]), float(N["hold"])
plateau = (t > 0.5 + ramp + 0.5) & (t < 0.5 + ramp + hold - 0.5)


def rms(x):
    return float(np.sqrt(np.mean(x[plateau] ** 2)))


def mnf_mdf(x):
    f, p = welch(x[plateau], fs=FS, nperseg=1024)
    band = (f >= 20) & (f <= 450)
    f, p = f[band], p[band]
    c = np.cumsum(p)
    return float(np.sum(f * p) / np.sum(p)), float(f[np.searchsorted(c, c[-1] / 2)])


def excitation(tt):
    tt = tt - 0.5
    return float(N["level"]) * np.clip(np.minimum(tt / ramp, (2 * ramp + hold - tt) / ramp), 0, 1)


pdf = PdfPages(here / "semg_realistic_old_vs_new.pdf")


def save(fig, name):
    pdf.savefig(fig)
    fig.savefig(here / f"{name}.png", dpi=150)
    plt.close(fig)


n_active = sum(len(s) > 0 for s in N["spike_times"])
setup = (f"VL-like cylinder: r = 15 mm, bone 3 mm, fat 4 mm, skin 1.29 mm; 100 MUs, {int(N['counts'].sum())} fibres "
         f"({float(N['density']):.0f}/mm²), 70 mm fibres, CV 4.2 m/s. Ramp to 30 % MVC (Fuglevand rate coding), "
         f"{n_active} MUs active.\nBipolar 5 mm contacts, IED 10 mm, centred 20 mm from the end-plate band. Chain: "
         f"+{NOISE_UV:.0f} µV RMS noise (same realisation), 4th-order 20–450 Hz zero-phase band-pass, 2048 Hz. "
         f"Skin σ: old {float(O['skin']):.2g} S/m, new {float(N['skin']):.2g} S/m (each version's default).")

# ---- page 1: whole contraction ---------------------------------------------------
fig, axs = plt.subplots(4, 1, figsize=(11.7, 8.3), sharex=True, gridspec_kw=dict(height_ratios=[0.8, 2, 2, 2]))
fig.suptitle("Finished bipolar sEMG, same muscle, spikes, electrodes and recording chain", fontsize=13)
ax = axs[0]
ax.plot(t, excitation(t) * 100, color="k")
ax.set_ylabel("excitation\n% max", fontsize=8)
spk = np.concatenate([np.asarray(s, float) for s in N["spike_times"]])
ax2 = ax.twinx()
ax2.hist(spk, bins=np.arange(0, t[-1], 0.1), color="0.7", alpha=0.6)
ax2.set_ylabel("spikes / 100 ms", fontsize=8)
for ax, x, c, title in ((axs[1], CO_["bip"], CO, "OLD (main): bipolar channel as MyoGen outputs it"),
                        (axs[2], CO_["diff"], CO, "OLD (main): difference of the two monopolar contacts"),
                        (axs[3], CN_["bip"], CN, "NEW (fixed): bipolar channel")):
    ax.plot(t, x, color=c, lw=0.4)
    ax.set_ylabel("µV")
    ax.set_title(f"{title}   — plateau RMS {rms(x):,.1f} µV", fontsize=10, loc="left")
axs[-1].set_xlabel("time (s)")
fig.text(0.01, 0.005, setup, fontsize=7)
fig.tight_layout(rect=(0, 0.06, 1, 0.96))
save(fig, "r1_contraction")

# ---- page 2: 300 ms zoom ------------------------------------------------------------
z = (t >= 4.0) & (t <= 4.3)
fig, axs = plt.subplots(3, 1, figsize=(11.7, 8.3), sharex=True)
fig.suptitle("300 ms of the 30 % MVC plateau (each panel on its own scale)", fontsize=13)
for ax, x, c, title in ((axs[0], CO_["bip"], CO, "OLD bipolar channel: only the 3 µV amplifier noise survives (P1)"),
                        (axs[1], CO_["diff"], CO, "OLD, difference of monopolar contacts: MUAPs ~2× too short, amplitude in mV"),
                        (axs[2], CN_["bip"], CN, "NEW bipolar channel")):
    ax.plot(t[z], x[z], color=c, lw=0.8)
    ax.set_ylabel("µV")
    ax.set_title(title, fontsize=10, loc="left")
axs[-1].set_xlabel("time (s)")
fig.tight_layout(rect=(0, 0, 1, 0.95))
save(fig, "r2_zoom")

# ---- page 3: spectra ------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(11.7, 5.5))
fig.suptitle("Plateau power spectra after the recording chain", fontsize=13)
for x, c, lab in ((CO_["diff"], CO, "old, monopolar difference"), (CN_["bip"], CN, "new, bipolar")):
    f, p = welch(x[plateau], fs=FS, nperseg=1024)
    mnf, mdf = mnf_mdf(x)
    axs[0].semilogy(f, p, color=c, label=f"{lab}: MNF {mnf:.0f} Hz, MDF {mdf:.0f} Hz")
    axs[1].plot(f, p / p.max(), color=c, label=lab)
    axs[1].axvline(mnf, color=c, ls="--", lw=0.8)
f, p = welch(CO_["bip"][plateau], fs=FS, nperseg=1024)
axs[0].semilogy(f, p, color=CO, ls=":", label="old, bipolar channel (noise only)")
axs[0].set_ylabel("PSD (µV²/Hz)")
axs[1].set_ylabel("normalised PSD")
for ax in axs:
    ax.set_xlim(0, 500)
    ax.set_xlabel("frequency (Hz)")
    ax.legend(fontsize=8)
fig.tight_layout(rect=(0, 0, 1, 0.93))
save(fig, "r3_spectra")

# ---- page 4: MUAP templates --------------------------------------------------------------
mus, counts = [int(m) for m in N["mus"]], N["counts"]
order = np.argsort([counts[m] for m in mus])
picks = [order[0], order[len(order) // 2], order[-1]]
fig, axs = plt.subplots(2, 3, figsize=(11.7, 7))
fig.suptitle("Bipolar MUAP templates (MyoGen output before the recording chain)", fontsize=13)
for col, k in enumerate(picks):
    for row, (D, c, name, key) in enumerate(((O, CO, "old: monopolar difference", "diff"), (N, CN, "new: bipolar", "bip"))):
        ax = axs[row, col]
        if key == "diff":
            m = (D["pair_muaps"][k][:, 1, 0] - D["pair_muaps"][k][:, 0, 0]) * 1e3
        else:
            m = D["bip_muaps"][k][:, 0, 0] * 1e3
        fs = float(D["pair_muap_fs"])
        tm = (np.arange(len(m)) - len(m) // 2) / fs * 1e3
        ax.plot(tm, m, color=c)
        ax.set_xlim(-5, 20)
        ax.set_title(f"{name} — MU {mus[k]} ({int(counts[mus[k]])} fibres)\nPtP {np.ptp(m):,.1f} µV", fontsize=9)
        ax.set_xlabel("ms from firing")
        ax.set_ylabel("µV")
fig.tight_layout(rect=(0, 0, 1, 0.94))
save(fig, "r4_muaps")

# ---- page 5: numbers -------------------------------------------------------------------------
fig = plt.figure(figsize=(11.7, 5))
fig.suptitle("Plateau summary (after noise + 20–450 Hz band-pass)", fontsize=13)
ax = fig.add_axes([0.05, 0.25, 0.9, 0.6])
ax.axis("off")
rows = []
for name, x in (("old: bipolar channel", CO_["bip"]), ("old: monopolar difference", CO_["diff"]),
                ("old: monopolar", CO_["mono"]), ("new: bipolar channel", CN_["bip"]),
                ("new: monopolar difference", CN_["diff"]), ("new: monopolar", CN_["mono"])):
    mnf, mdf = mnf_mdf(x)
    rows.append([name, f"{rms(x):,.1f}", f"{np.ptp(x[plateau]):,.0f}", f"{mnf:.0f}", f"{mdf:.0f}"])
tab = ax.table(cellText=rows, colLabels=["signal", "RMS µV", "PtP µV", "MNF Hz", "MDF Hz"], loc="center",
               cellLoc="center")
tab.auto_set_font_size(False)
tab.set_fontsize(9)
tab.scale(1, 1.6)
fig.text(0.05, 0.08, setup, fontsize=7.5)
save(fig, "r5_summary")
pdf.close()
for r in rows:
    print(r)
