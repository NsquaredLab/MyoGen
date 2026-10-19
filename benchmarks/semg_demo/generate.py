"""Generate a reproducible sEMG illustration with prescribed firing.

Run in an installed MyoGen environment; output defaults to artifacts/semg-demo.
The spike trains are a demonstration input, not a fitted neural-drive model.
"""
from pathlib import Path
import argparse
import json
import subprocess
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import quantities as pq
from neo import Block, Segment, SpikeTrain
import myogen
from myogen import simulator


def main(out):
    start = time.monotonic()
    seed, fs, duration = 20261019, 2048, 8.0
    myogen.set_random_seed(seed)
    rng = np.random.default_rng(seed + 1)
    thresholds, _ = simulator.RecruitmentThresholds(
        N=100, recruitment_range__ratio=100, mode='fuglevand')
    muscle = simulator.Muscle(recruitment_thresholds=thresholds,
                              radius_bone__mm=1.0*pq.mm, grid_resolution=128)
    print('Building muscle and assigning fibers', flush=True)
    muscle.distribute_innervation_centers()
    muscle.generate_muscle_fiber_centers(verbose=False)
    muscle.assign_mfs2mns(n_jobs=2, verbose=False)
    muscle._generate_fiber_properties()
    selected = list(range(24))
    counts = np.asarray(muscle.resulting_number_of_innervated_fibers, dtype=int)
    assert np.all(counts[selected] > 0)
    print(f'Muscle: {int(counts.sum())} fibers; selected: {int(counts[selected].sum())}', flush=True)
    array = simulator.SurfaceElectrodeArray(
        num_rows=8, num_cols=2, inter_electrode_distance__mm=5*pq.mm,
        electrode_radius__mm=1*pq.mm, differentiation_mode='monopolar',
        bending_radius__mm=muscle.radius__mm + muscle.skin_thickness__mm + muscle.fat_thickness__mm)
    model = simulator.SurfaceEMG(
        muscle, [array], sampling_frequency__Hz=float(fs)*pq.Hz,
        internal_sampling_frequency__Hz=4096.0*pq.Hz,
        sampling_points_in_t_and_z_domains=256,
        sampling_points_in_theta_domain=32, MUs_to_simulate=selected)
    policy = simulator.ExecutionPolicy(memory_budget_mib=4096, max_workers=2)
    print('Generating surface MUAPs', flush=True)
    muaps = model.simulate_muaps(n_jobs=2, use_gpu=False, verbose=False, execution=policy)
    muap_seconds = time.monotonic() - start
    print(f'MUAPs ready after {muap_seconds:.1f} s; generating spike trains', flush=True)

    t = np.arange(round(duration*fs))/fs
    envelope = np.interp(t, [0, .5, 2, 6, 7.5, duration], [0, 0, 1, 1, 0, 0])
    block, segment = Block(name='Prescribed demonstration firing'), Segment(name='Pool_0')
    block.segments.append(segment)
    all_spikes = []
    for mu in range(100):
        if mu in selected:
            threshold = .05 + .65*mu/(len(selected)-1)
            drive = np.maximum(0, (envelope-threshold)/(1-threshold))
            rate = np.where(envelope > threshold, 8 + (10-3*mu/23)*drive, 0)
            integrated = np.cumsum(rate)/fs
            events, level = [], rng.uniform(.1, 1.0)
            while level < integrated[-1]:
                idx = np.searchsorted(integrated, level)
                events.append(t[idx])
                level += rng.gamma(shape=25, scale=1/25)
            events = np.asarray(events)
        else:
            events = np.empty(0)
        train = SpikeTrain(events*pq.s, t_stop=duration*pq.s, sampling_rate=float(fs)*pq.Hz)
        segment.spiketrains.append(train)
        all_spikes.append(events)
    signals = model.simulate_surface_emg(block, verbose=False, execution=policy)
    signal = signals.groups[0].segments[0].analogsignals[0]
    data = signal.rescale(pq.uV).magnitude
    assert data.shape == (len(t), 16) and np.isfinite(data).all()
    assert np.max(np.abs(data)) > 0
    muap_data = np.stack([s.analogsignals[0].rescale(pq.uV).magnitude
                         for s in muaps.groups[0].segments])
    muap_times = muaps.groups[0].segments[0].analogsignals[0].times.rescale(pq.ms).magnitude
    event_mu = np.concatenate([np.full(len(s), mu, dtype=int) for mu, s in enumerate(all_spikes)])
    event_t = np.concatenate(all_spikes)
    np.savez_compressed(out/'semg.npz', time_s=t, semg_uV=data,
                        muap_time_ms=muap_times, muaps_uV=muap_data,
                        spike_time_s=event_t, spike_mu=event_mu,
                        electrode_z_mm=array.pos_z.rescale(pq.mm).magnitude,
                        electrode_theta_rad=array.pos_theta.rescale(pq.rad).magnitude,
                        selected_mus=np.array(selected), fiber_counts=counts)
    rms = np.sqrt(np.mean(data[(t >= 2) & (t < 6)]**2, axis=0))
    strongest = int(np.argmax(rms))
    try:
        revision = subprocess.run(
            ['git', '-C', str(Path(myogen.__file__).resolve().parents[1]), 'rev-parse', 'HEAD'],
            capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    metadata = dict(commit=revision, seed=seed,
                    duration_s=duration, sampling_rate_hz=fs, internal_sampling_rate_hz=4096,
                    selected_mus=selected, muscle_fibers=int(counts.sum()),
                    simulated_fibers=int(counts[selected].sum()), electrode_grid=[8, 2],
                    electrode_spacing_mm=5, electrode_radius_mm=1, mode='monopolar',
                    muscle_grid_resolution=128, added_noise=False, filtering=False,
                    amplitude_unit='uV', strongest_channel_zero_based=strongest,
                    plateau_rms_uV=rms.tolist(), peak_to_peak_uV=np.ptp(data,axis=0).tolist(),
                    elapsed_seconds=time.monotonic()-start,
                    firing='Prescribed rate-modulated gamma-renewal trains; interval CV=0.2 in rate-time; 24 of 100 MUs active. No neuronal simulation or experimental fitting.',
                    purpose='Illustration; separate from Devon frozen-cohort benchmark.')
    (out/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')

    print(json.dumps(metadata, indent=2), flush=True)
    plot(out, t, data, muap_data, muap_times, all_spikes, strongest, duration, selected)


def plot(out, t, data, muap_data, muap_times, all_spikes, strongest, duration, selected):
    # Display mV for readable axes; saved arrays remain in microvolts.
    data, muap_data = data / 1000, muap_data / 1000
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                         'axes.spines.top':False, 'axes.spines.right':False,
                         'axes.labelcolor':'#354052', 'xtick.color':'#556176',
                         'ytick.color':'#556176', 'axes.edgecolor':'#c8ced8',
                         'savefig.facecolor':'#ffffff'})
    fig = plt.figure(figsize=(13, 9), facecolor='white')
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.55, .72], width_ratios=[1.8, 1],
                          left=.085, right=.97, top=.85, bottom=.09, hspace=.55, wspace=.3)
    fig.text(.085, .95, 'Generated surface EMG', fontsize=23, weight='bold', color='#162334')
    fig.text(.085, .908, 'MyoGen  ·  24 active motor units  ·  16 monopolar channels  ·  2,048 Hz',
             fontsize=12, color='#556176')
    ax = fig.add_subplot(gs[0,:])
    ax.plot(t, data[:,strongest], color='#0a7181', lw=.55)
    ax.axvspan(3, 3.25, color='#e8ad50', alpha=.25, lw=0)
    ax.set(xlim=(0,duration), xlabel='Time (s)', ylabel='Model output (mV)')
    ax.set_title(f'8-second recording  /  channel {strongest+1} (highest plateau RMS)', loc='left', fontsize=11, pad=9)
    ax.grid(axis='y', alpha=.13)

    zoom = fig.add_subplot(gs[1,0])
    window = (t>=3)&(t<=3.25)
    col = strongest % 2
    channels = np.arange(8)*2+col
    spacing = max(float(np.ptp(data[window][:,channels],axis=0).max())*1.18, 1e-12)
    offsets = np.arange(8)[::-1]*spacing
    for ch, offset in zip(channels, offsets):
        zoom.plot((t[window]-3)*1000, data[window,ch]+offset, lw=.85, color='#0a7181')
    zoom.set(xlim=(0,250), xlabel='Time from 3.0 s (ms)', yticks=offsets,
             yticklabels=[f'Ch {ch+1:02d}' for ch in channels])
    zoom.set_title('250 ms close-up  /  one electrode column', loc='left', fontsize=11, pad=9)
    zoom.spines['left'].set_visible(False)
    zoom.tick_params(axis='y', length=0)
    # Same gain on all channels; label the physical size of the offset.
    zoom.text(1,.01, f'Equal gain · vertical spacing {spacing:.1f} mV',
              transform=zoom.transAxes, ha='right', fontsize=8, color='#556176')

    ax_muap = fig.add_subplot(gs[1,1])
    mu = selected[int(np.argmax(np.ptp(muap_data[selected,:,strongest],axis=1)))]
    for ch in channels:
        ax_muap.plot(muap_times, muap_data[mu,:,ch], lw=1.1, alpha=.8)
    ax_muap.set(xlabel='Time (ms)', ylabel='Model output (mV)')
    ax_muap.set_title(f'One MUAP  /  motor unit {mu+1}\nSame eight electrodes', loc='left', fontsize=11, pad=9)
    ax_muap.grid(alpha=.13)

    raster = fig.add_subplot(gs[2,:])
    raster.eventplot(all_spikes[:24], lineoffsets=np.arange(1,25), linelengths=.75,
                     linewidths=.6, colors='#61758c')
    raster.set(xlim=(0,duration), ylim=(.2,24.8), xlabel='Time (s)', ylabel='Motor unit', yticks=[1,12,24])
    raster.set_title('Prescribed motor-unit discharges', loc='left', fontsize=11, pad=9)
    fig.text(.085,.025, 'Simulated illustration · no added noise or filtering · absolute amplitude unvalidated',
             fontsize=9, color='#667184')
    fig.savefig(out/'semg_preview.png', dpi=180)
    fig.savefig(out/'semg_preview.pdf')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/semg-demo'))
    parser.add_argument('--plot-only', action='store_true', help='Replot existing arrays without simulation')
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        saved = np.load(out/'semg.npz')
        metadata = json.loads((out/'metadata.json').read_text())
        spikes = [saved['spike_time_s'][saved['spike_mu']==mu] for mu in range(100)]
        plot(out, saved['time_s'], saved['semg_uV'], saved['muaps_uV'], saved['muap_time_ms'],
             spikes, metadata['strongest_channel_zero_based'], metadata['duration_s'],
             metadata['selected_mus'])
    else:
        main(out)
