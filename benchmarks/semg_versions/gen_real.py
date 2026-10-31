"""Realistic frozen sEMG recording with whichever myogen is on sys.path.

Usage: python gen_real.py <label> <expected_worktree> <out.npz> <density_per_mm2> [n_jobs] [max_mus]
Muscle, end-plates, fibre lengths and spike trains are seeded identically; each code
version keeps its own defaults (including skin conductivity).
"""
import hashlib
import sys
import time

import numpy as np
import quantities as pq
from neo import Block, Segment, SpikeTrain

label, expected, out, density = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
n_jobs = int(sys.argv[5]) if len(sys.argv) > 5 else 8
max_mus = int(sys.argv[6]) if len(sys.argv) > 6 else None
import myogen  # noqa: E402

assert expected.lower() in myogen.__file__.lower(), myogen.__file__
from myogen import set_random_seed  # noqa: E402
from myogen.simulator import Muscle, RecruitmentThresholds, SurfaceElectrodeArray, SurfaceEMG  # noqa: E402
from myogen.utils.neo import signal_to_grid  # noqa: E402

N_MU, FS = 100, 2048.0
RAMP, HOLD, LEVEL = 2.0, 4.0, 0.30  # s, s, fraction of max excitation
DURATION = 2 * RAMP + HOLD + 1.0    # 0.5 s rest before and after

# ---- muscle ------------------------------------------------------------------
set_random_seed(42)
thresholds, _ = RecruitmentThresholds(N=N_MU, recruitment_range__ratio=50, mode="fuglevand")
muscle = Muscle(recruitment_thresholds=thresholds, radius__mm=15 * pq.mm, length__mm=80 * pq.mm,
                fiber_density__fibers_per_mm2=density * pq.mm**-2, mean_fiber_length__mm=70 * pq.mm,
                var_fiber_length__mm=5 * pq.mm, radius_bone__mm=3 * pq.mm, fat_thickness__mm=4 * pq.mm)
muscle.distribute_innervation_centers()
muscle.generate_muscle_fiber_centers(verbose=False)
muscle.assign_mfs2mns(verbose=False, n_jobs=n_jobs)
counts = np.asarray(muscle.resulting_number_of_innervated_fibers)
fibre_hash = hashlib.sha1(b"".join(
    np.ascontiguousarray(muscle.resulting_fiber_assignment(i)).tobytes() for i in range(N_MU))).hexdigest()
skin = float(muscle.skin_conductivity__S_m.rescale(pq.S / pq.m).magnitude)

# ---- spike trains: Fuglevand et al. (1993) rate coding, frozen ---------------
rng = np.random.default_rng(123)
rte = np.asarray(thresholds, dtype=float)
rte = rte / rte.max() * 0.7  # last unit recruited at 70 % excitation


def excitation(t):
    t = t - 0.5
    return LEVEL * np.clip(np.minimum(t / RAMP, (2 * RAMP + HOLD - t) / RAMP), 0, 1)


trains, active = [], []
for mu in range(N_MU):
    times, t = [], 0.0
    while t < DURATION:
        e = excitation(t)
        if e < rte[mu]:
            t += 1 / FS
            continue
        rate = min(8.0 + 100 * (e - rte[mu]), 35.0)  # 8 Hz at recruitment, 1 Hz/%, 35 Hz peak
        times.append(t)
        t += rng.gamma(1 / 0.2**2, 0.2**2 / rate)  # ISI CV 0.2
    times = np.asarray([x for x in times if x < DURATION])
    active.append(len(times) > 0)
    trains.append(SpikeTrain(np.round(times * FS) / FS * pq.s, t_stop=DURATION * pq.s, sampling_rate=FS * pq.Hz))
block = Block()
seg = Segment()
seg.spiketrains = trains
block.segments.append(seg)
spike_hash = hashlib.sha1(b"".join(np.asarray(st.magnitude).tobytes() for st in trains)).hexdigest()
mus = [m for m in range(N_MU) if active[m] and counts[m] > 0]
if max_mus:
    mus = mus[:max_mus]

bend = 15 + 4 + 1.29
pair = SurfaceElectrodeArray(2, 1, 10 * pq.mm, 2.5 * pq.mm, center_point__mm_deg=(20 * pq.mm, 0 * pq.deg),
                             bending_radius__mm=bend * pq.mm, differentiation_mode="monopolar")
bip = SurfaceElectrodeArray(1, 1, 10 * pq.mm, 2.5 * pq.mm, center_point__mm_deg=(20 * pq.mm, 0 * pq.deg),
                            bending_radius__mm=bend * pq.mm, differentiation_mode="bipolar_longitudinal")


def run(array):
    set_random_seed(7)
    semg = SurfaceEMG(muscle, [array], sampling_frequency__Hz=FS * pq.Hz, MUs_to_simulate=mus,
                      internal_sampling_frequency__Hz=FS * pq.Hz)
    t0 = time.time()
    muaps = semg.simulate_muaps(n_jobs=n_jobs, verbose=False, use_gpu=False)
    sig0 = muaps.groups[0].segments[mus[0]].analogsignals[0]
    templates = np.array([np.asarray(signal_to_grid(muaps.groups[0].segments[m].analogsignals[0].rescale(pq.mV)))
                          for m in mus])
    emg = semg.simulate_surface_emg(block, verbose=False).groups[0].segments[0].analogsignals[0]
    return dict(muaps=templates, muap_fs=float(sig0.sampling_rate.rescale(pq.Hz).magnitude),
                emg=np.asarray(signal_to_grid(emg.rescale(pq.mV))), fs=float(emg.sampling_rate.rescale(pq.Hz).magnitude),
                seconds=time.time() - t0)


p, b = run(pair), run(bip)
np.savez(out, label=label, myogen=myogen.__file__, mus=mus, counts=counts, rte=rte, skin=skin, density=density,
         fibre_hash=fibre_hash, spike_hash=spike_hash, duration=DURATION, ramp=RAMP, hold=HOLD, level=LEVEL,
         spike_times=np.array([np.asarray(trains[m].magnitude) for m in range(N_MU)], dtype=object),
         pair_muaps=p["muaps"], pair_muap_fs=p["muap_fs"], pair_emg=p["emg"], fs=p["fs"],
         bip_muaps=b["muaps"], bip_emg=b["emg"], allow_pickle=True)
print(label, "density", density, "fibres total", int(counts.sum()), "simulated MUs", len(mus),
      "fibres simulated", int(counts[mus].sum()), "skin", skin, "secs", round(p["seconds"]), round(b["seconds"]),
      "fibre", fibre_hash[:10], "spikes", spike_hash[:10], flush=True)
