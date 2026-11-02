"""Add reproducible independent white Gaussian noise to saved sEMG channels."""
from pathlib import Path
import argparse
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir', type=Path, default=Path('artifacts/semg-demo'))
args = parser.parse_args()
out = args.output_dir
source = out / 'semg.npz'
saved = np.load(source)
original_metadata = json.loads((out / 'metadata.json').read_text())
t, clean = saved['time_s'], saved['semg_uV']
snr_db, seed = 20.0, 20261020
rng = np.random.default_rng(seed)
signal_power = np.mean(clean**2, axis=0)
noise_sd = np.sqrt(signal_power / 10**(snr_db / 10))
noise = rng.normal(size=clean.shape) * noise_sd[None, :]
noisy = clean + noise
realized_snr = 10*np.log10(signal_power / np.mean(noise**2, axis=0))
assert noisy.shape == clean.shape and np.isfinite(noisy).all()
assert np.max(np.abs(realized_snr - snr_db)) < .3
np.savez_compressed(out / 'semg_noisy_20dB.npz',
                    **{key: saved[key] for key in saved.files if key != 'semg_uV'},
                    semg_uV=noisy, clean_semg_uV=clean, noise_uV=noise)
metadata = dict(original_metadata)
metadata.update(added_noise=True, noise_type='independent additive white Gaussian',
                noise_seed=seed, target_snr_db=snr_db,
                snr_reference='Per-channel mean square over the full 8 s recording, including rest.',
                realized_snr_db=realized_snr.tolist(), noise_sd_uV=noise_sd.tolist(),
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
(out / 'metadata_noisy_20dB.json').write_text(json.dumps(metadata, indent=2)+'\n')

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#c8ced8', 'axes.labelcolor': '#354052',
                     'xtick.color': '#556176', 'ytick.color': '#556176'})
fig = plt.figure(figsize=(13, 10), facecolor='white')
gs = fig.add_gridspec(3, 1, height_ratios=[1, .9, 1.65],
                      left=.085, right=.97, top=.86, bottom=.09, hspace=.58)
fig.text(.085, .95, 'Surface EMG with added noise', fontsize=23,
         weight='bold', color='#162334')
fig.text(.085, .912, '20 dB SNR per channel · independent white Gaussian noise · 16 channels · 2,048 Hz',
         fontsize=12, color='#556176')
channel = original_metadata['strongest_channel_zero_based']
clean_mV, noisy_mV = clean/1000, noisy/1000
teal, orange = '#0a7181', '#c26428'
ax = fig.add_subplot(gs[0])
ax.plot(t, noisy_mV[:, channel], color=orange, lw=.5)
ax.axvspan(3, 3.25, color='#e8ad50', alpha=.22, lw=0)
ax.set(xlim=(0, 8), xlabel='Time (s)', ylabel='Model output (mV)')
ax.set_title(f'Noisy recording / channel {channel+1}', loc='left', fontsize=11, pad=9)
ax.grid(axis='y', alpha=.13)

window = (t >= 3) & (t <= 3.25)
x = (t[window] - 3)*1000
zoom = fig.add_subplot(gs[1])
zoom.plot(x, noisy_mV[window, channel], color=orange, lw=1, alpha=.85, label='With noise (20 dB)')
zoom.plot(x, clean_mV[window, channel], color=teal, lw=1.05, label='Clean model output')
zoom.set(xlim=(0, 250), xlabel='Time from 3.0 s (ms)', ylabel='Model output (mV)')
zoom.set_title('Same 250 ms segment / clean versus noisy', loc='left', fontsize=11, pad=9)
zoom.grid(axis='y', alpha=.13)
zoom.legend(frameon=False, loc='upper right', ncols=2, fontsize=9)

multi = fig.add_subplot(gs[2])
channels = np.arange(8)*2 + channel % 2
spacing = float(np.ptp(noisy_mV[window][:, channels], axis=0).max())*1.18
offsets = np.arange(8)[::-1]*spacing
for ch, offset in zip(channels, offsets):
    multi.plot(x, noisy_mV[window, ch]+offset, color=orange, lw=.7)
multi.set(xlim=(0,250), xlabel='Time from 3.0 s (ms)', yticks=offsets,
          yticklabels=[f'Ch {ch+1:02d}' for ch in channels])
multi.set_title('Noisy signals across one electrode column', loc='left', fontsize=11, pad=9)
multi.spines['left'].set_visible(False)
multi.tick_params(axis='y', length=0)
multi.text(1, .005, f'Equal gain · vertical spacing {spacing:.1f} mV',
           transform=multi.transAxes, ha='right', fontsize=8, color='#556176')
fig.text(.085, .025, 'Noise RMS ≈ 10% of each channel’s full-recording signal RMS · absolute model amplitude unvalidated',
         fontsize=9, color='#667184')
fig.savefig(out / 'semg_noisy_20dB.png', dpi=180, facecolor='white')
fig.savefig(out / 'semg_noisy_20dB.pdf', facecolor='white')
plt.close(fig)
print(json.dumps({'target_snr_db': snr_db, 'realized_snr_db_range':
                  [float(realized_snr.min()), float(realized_snr.max())],
                  'shape': list(noisy.shape), 'output': str(out / 'semg_noisy_20dB.png')}))
