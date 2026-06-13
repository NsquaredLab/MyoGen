"""
Physiological PIC / spasticity: same input, flip the PIC state
==============================================================

Reproduces the persistent-inward-current (PIC) signature mechanistically by
up-regulating motoneuron dendritic Ca PIC (``gamma``) and somatic Na PIC
(``gnapbar_napp``) instead of hand-sculpting descending drive.

Top: a single type-S motoneuron — same dendritic ramp, control vs SCI, showing
amplification and self-sustained firing (recruitment/derecruitment hysteresis).
Bottom: a motor pool — the same brief low-MVC voluntary command yields a
self-sustained involuntary discharge (a spasm) only when the PIC is up-regulated.
"""
# sphinx_gallery_thumbnail_number = -1
import sys
from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from myogen import set_random_seed

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pic_protocols as pic

set_random_seed(42)
NAP_CEILING = 0.00215
N_MU = 60

try:
    save_path = Path(__file__).parent / "results"
except NameError:
    save_path = Path.cwd() / "results"
save_path.mkdir(exist_ok=True, parents=True)

# --- single-cell ramp (mechanism): control vs SCI ---
ctrl_cell = pic.ramp_hysteresis(gamma=0.2, nap_factor=1.0, imax_nA=12.0)
sci_cell = pic.ramp_hysteresis(gamma=1.5, nap_factor=5.0, imax_nA=12.0,
                               nap_ceiling=NAP_CEILING)
print(f"single-cell: ctrl I_on={ctrl_cell['i_on']:.2f}/I_off={ctrl_cell['i_off']:.2f} "
      f"| SCI I_on={sci_cell['i_on']:.2f}/I_off={sci_cell['i_off']:.2f}")

# --- pool spasm (function): SCI condition, full scale ---
command, t_off_s, total_s = pic.brief_command_drive(total_s=10.0)
muscle, _ = pic.build_muscle(N_MU)
iemg_sim = pic.make_iemg_simulator(muscle, N_MU)
sci_pool = pic.run_pool(command, n_mu=N_MU, gamma=1.5, nap_factor=5.0,
                        nap_ceiling=NAP_CEILING, total_s=total_s)
sci_emg = pic.synthesize_iemg(sci_pool, N_MU, iemg_sim=iemg_sim,
                              snr_dB=None, t_off_s=t_off_s)
print(f"pool: spikes after command offset = {pic.spikes_after(sci_pool, t_off_s)} "
      f"| tail/on RMS = {sci_emg['tail_ratio']:.3f}")

# --- two-row figure ---
fig, axes = plt.subplots(2, 1, figsize=(10, 6))
axes[0].plot(ctrl_cell["t"] / 1000.0, ctrl_cell["v"], lw=0.3, color="tab:blue",
             label="control (gamma=0.2)")
axes[0].plot(sci_cell["t"] / 1000.0, sci_cell["v"], lw=0.3, color="k",
             label="SCI (gamma=1.5, NaP x5)")
axes[0].set_title("Single MN — same dendritic ramp: SCI shows self-sustained firing")
axes[0].set_ylabel("Vm (mV)")
axes[0].legend(loc="upper right", fontsize=7)
axes[1].plot(sci_emg["times"], sci_emg["iemg"], lw=0.2, color="k")
axes[1].axvline(t_off_s, color="tab:red", lw=0.8, ls="--")
axes[1].set_title("Motor pool iEMG (SCI) — spasm persists past command offset (red)")
axes[1].set_xlabel("time (s)"); axes[1].set_ylabel("iEMG (a.u.)")
for ax in axes:
    sns.despine(ax=ax, trim=True, offset=2)
fig.tight_layout()
fig.savefig(save_path / "pic_spasticity.svg")
fig.savefig(save_path / "pic_spasticity.pdf")
plt.show()
