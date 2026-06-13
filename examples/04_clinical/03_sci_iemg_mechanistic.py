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
2. **Loss of derecruitment** -- up-regulated PIC (``gamma`` + somatic NaP); the
   *same* command no longer lets units go silent at the troughs.
3. **Modulation -> spasm** -- up-regulated PIC; the voluntary command runs for
   the first half then stops, but the PIC sustains an **involuntary discharge**.

(The original's third phenotype was 6 Hz *clonus*, a stretch-reflex-loop
oscillation -- not a PIC phenomenon -- so it is replaced here by a PIC-driven
spasm. NERLab model throughout, consistent with the manuscript. NERLab's caL
does not inactivate, so recruited units latch rather than slowly decaying.)
"""
# sphinx_gallery_thumbnail_number = -1
import sys
from pathlib import Path

import numpy as np
import quantities as pq
import seaborn as sns
from matplotlib import pyplot as plt
plt.rcParams["path.simplify"] = False  # draw every sample on dense EMG traces
from neo import AnalogSignal

from myogen import set_random_seed, get_random_generator
from myogen.utils.types import pps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pic_protocols as pic

set_random_seed(42)
N_MU = 40
TOTAL_S = 8.0
NAP_CEILING = 0.00215

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
import joblib

cache = save_path / f"sci_mechanistic_iemg_sim_n{N_MU}.pkl"
if cache.exists():
    iemg_sim = joblib.load(cache)
else:
    muscle, _ = pic.build_muscle(N_MU)
    iemg_sim = pic.make_iemg_simulator(muscle, N_MU)
    joblib.dump(iemg_sim, cache)

voluntary, _ = pic.cyclic_voluntary_drive(peak_pps=45.0, freq_hz=0.5,
                                          total_s=TOTAL_S)
spasm_drive = modulation_then_silence()

# label -> (descending drive, gamma, nap_factor)
conditions = {
    "Voluntary modulation (healthy PIC)": (voluntary, 0.2, 1.0),
    "Loss of derecruitment (SCI PIC)": (voluntary, 1.5, 5.0),
    "Modulation -> spasm (SCI PIC)": (spasm_drive, 1.5, 5.0),
}

results = {}
for label, (drive, gamma, nap) in conditions.items():
    block = pic.run_pool(drive, n_mu=N_MU, gamma=gamma, nap_factor=nap,
                         nap_ceiling=NAP_CEILING, total_s=TOTAL_S)
    emg = pic.synthesize_iemg(block, N_MU, iemg_sim=iemg_sim, snr_dB=20)
    centers, rate = pic.population_rate(block, TOTAL_S)
    results[label] = dict(drive=drive, block=block, iemg=emg, rate=(centers, rate))
    active = sum(1 for st in block.segments[0].spiketrains if len(st) > 0)
    print(f"{label}: active MUs={active}/{N_MU} | trough rate="
          f"{pic.rate_in_windows(centers, rate, [2.0, 4.0, 6.0]):.1f} pps")


# %%
def raster(ax, block):
    sts = block.segments[0].spiketrains
    active = [u for u in range(len(sts)) if len(sts[u]) > 0]
    order = sorted(active, key=lambda u: float(sts[u].rescale("s").magnitude.min()))
    colors = plt.cm.rainbow(np.linspace(0, 1, max(len(order), 1)))
    for rank, u in enumerate(order):
        st = sts[u].rescale("s").magnitude
        ax.scatter(st, [rank] * len(st), s=5, color=colors[rank], marker="|",
                   linewidths=0.5)


labels = list(conditions.keys())

# Figure 1 -- motor unit rasters
fig, axes = plt.subplots(len(labels), 1, figsize=(9, 7), sharex=True)
for ax, label in zip(axes, labels):
    raster(ax, results[label]["block"])
    ax.set_ylabel("MU")
    ax.set_title(label)
axes[-1].set_xlabel("time (s)")
for ax in axes:
    sns.despine(ax=ax, trim=True, offset=2)
fig.tight_layout()
fig.savefig(save_path / "sci_mechanistic_raster.svg")
fig.savefig(save_path / "sci_mechanistic_raster.pdf")

# Figure 2 -- intramuscular EMG (first channel)
fig, axes = plt.subplots(len(labels), 1, figsize=(9, 7), sharex=True)
for ax, label in zip(axes, labels):
    emg = results[label]["iemg"]
    ax.plot(emg["times"], emg["iemg"], lw=0.2, color="k")
    ax.set_ylabel("iEMG (a.u.)")
    ax.set_title(label)
axes[-1].set_xlabel("time (s)")
for ax in axes:
    sns.despine(ax=ax, trim=True, offset=2)
fig.tight_layout()
fig.savefig(save_path / "sci_mechanistic_iemg.svg")
fig.savefig(save_path / "sci_mechanistic_iemg.pdf")
plt.show()
