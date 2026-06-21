"""
Mechanistic Loss of Derecruitment: Same Drive, Flip the PIC
===========================================================

The earlier SCI example reproduced *loss of derecruitment* by **hand-adding a
tonic floor** to the descending drive (a 42 pps offset so units never went
silent). That imposes the phenotype on the *input*.

Here the **same** voluntary command -- a 0.5 Hz sinusoid that returns to zero at
every trough -- is used for both conditions. The only thing that differs is the
motoneuron's persistent-inward-current (PIC) state:

- **Healthy PIC** (``gamma=0.2``): the pool recruits at each peak and
  **derecruits** at each trough (rate returns to zero).
- **SCI / up-regulated PIC** (``gamma=1.5``, somatic NaP x5): once recruited the
  units **keep firing through the troughs** -- *loss of derecruitment emerges
  from the PIC*, not from the input.

So the "tonic floor" the old example drew by hand is now an **output** of the
mechanism. NERLab model throughout (consistent with the manuscript).

.. note::
    NERLab's dendritic Ca PIC (``caL``) does not inactivate, so recruited units
    stay recruited; the firing rate still modulates with the drive but never
    returns to silence. Reproducing the slow *decay* of the PIC would require a
    calcium-dependent-inactivation channel model (future work).
"""
# sphinx_gallery_thumbnail_number = -1

# %%

##############################################################################
# Import Libraries
# ----------------

import sys
from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from myogen import set_random_seed

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pic_protocols as pic  # noqa: E402

plt.style.use("fivethirtyeight")
plt.rcParams["path.simplify"] = False  # draw every sample on dense EMG traces

# %%

##############################################################################
# Setup
# -----

set_random_seed(42)
N_MU = 40
TOTAL_S = 8.0
NAP_CEILING = 0.00215
TROUGHS = [2.0, 4.0, 6.0]
PEAKS = [1.0, 3.0, 5.0, 7.0]

try:
    save_path = Path(__file__).parent / "results"
except NameError:
    save_path = Path.cwd() / "results"
save_path.mkdir(exist_ok=True, parents=True)

# %%

##############################################################################
# Same Voluntary Command, Healthy vs Up-Regulated PIC
# ---------------------------------------------------
#
# The same 0.5 Hz command drives both pools; only the PIC state differs. The
# population rate sampled at the troughs reveals whether each pool derecruits.

drive, total_s = pic.cyclic_voluntary_drive(peak_pps=45.0, freq_hz=0.5,
                                            total_s=TOTAL_S)
ctrl = pic.run_pool(drive, n_mu=N_MU, gamma=0.2, nap_factor=1.0, total_s=total_s)
sci = pic.run_pool(drive, n_mu=N_MU, gamma=1.5, nap_factor=5.0,
                   nap_ceiling=NAP_CEILING, total_s=total_s)

tc, rc = pic.population_rate(ctrl, total_s)
ts, rs = pic.population_rate(sci, total_s)
print(f"control: trough rate={pic.rate_in_windows(tc, rc, TROUGHS):.1f} pps  "
      f"peak rate={pic.rate_in_windows(tc, rc, PEAKS):.1f} pps")
print(f"SCI    : trough rate={pic.rate_in_windows(ts, rs, TROUGHS):.1f} pps  "
      f"peak rate={pic.rate_in_windows(ts, rs, PEAKS):.1f} pps")

# %%

##############################################################################
# Figure: Rasters and Population Rate
# -----------------------------------
#
# Drive, healthy-PIC raster, SCI-PIC raster, and the population rate. The healthy
# rate returns to zero at every trough; the SCI rate stays elevated.


def raster(ax, block, title):
    """First-spike-sorted rainbow raster."""
    sts = block.segments[0].spiketrains
    active = [u for u in range(len(sts)) if len(sts[u]) > 0]
    order = sorted(active, key=lambda u: float(sts[u].rescale("s").magnitude.min()))
    colors = plt.cm.rainbow(np.linspace(0, 1, max(len(order), 1)))
    ax.eventplot([sts[u].rescale("s").magnitude for u in order],
                 lineoffsets=np.arange(len(order)), colors=list(colors),
                 linelengths=0.8, linewidths=0.7, rasterized=True)
    ax.set_ylim(-0.8, max(len(order), 1) - 0.2)
    ax.set_ylabel("MU (1st-spike order)")
    ax.set_title(title)


fig, axes = plt.subplots(4, 1, figsize=(9, 8), sharex=True,
                         gridspec_kw={"height_ratios": [1, 2, 2, 1.2]})
axes[0].plot(np.linspace(0, total_s, len(drive)),
             np.asarray(drive.magnitude).ravel(), lw=0.3, color="0.5")
axes[0].set_ylabel("drive (pps)")
axes[0].set_title("Same voluntary command (returns to zero each cycle)")
raster(axes[1], ctrl, "Healthy PIC -- derecruits at every trough")
raster(axes[2], sci, "SCI / up-regulated PIC -- loss of derecruitment")
axes[3].plot(tc, rc, color="tab:blue", label="healthy")
axes[3].plot(ts, rs, color="tab:red", label="SCI")
for x in TROUGHS:
    axes[3].axvline(x, color="0.8", lw=0.6, ls=":")
axes[3].set_ylabel("pop. rate (pps)")
axes[3].set_xlabel("time (s)")
axes[3].legend(loc="upper right", fontsize=7)
axes[3].set_title("Population rate: healthy returns to zero, SCI stays elevated")
for ax in axes:
    sns.despine(ax=ax, trim=True, offset=2)
fig.tight_layout()
fig.savefig(save_path / "pic_loss_of_derecruitment.svg")
fig.savefig(save_path / "pic_loss_of_derecruitment.pdf")
plt.show()
