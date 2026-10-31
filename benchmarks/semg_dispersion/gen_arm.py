"""Simulate one sweep arm: a frozen sEMG scenario with a fibre-dispersion configuration.

Usage: python gen_arm.py <scenario> <label> <out.npz> <config-json> [n_jobs]

Scenarios (muscle, spike trains, electrode and seeds are fixed; only the
dispersion configuration changes, so arms are paired against the baseline):

- ``limb``: limb-like cylinder, r 15 mm, bone 3 mm, fat 4 mm, 80 mm long,
  70 +/- 5 mm fibres, 350 fibres/mm2, 100 MUs; bipolar 5 mm contacts, IED 10 mm,
  centred 20 mm from the end-plate band.
- ``fdi``: the default ``Muscle`` (tuned for the first dorsal interosseous),
  120 MUs; bipolar 2 mm contacts, IED 5 mm, centred 8 mm from the end-plate band;
  IAP window 150 mm so slow dispersed fibres fit.

Both use a 30 % excitation trapezoid (2 s ramps, 4 s hold, 0.5 s rest at each
end), Fuglevand thresholds over a 50-fold range recruited up to 70 % excitation,
8-35 Hz rate coding with ISI CV 0.2, 2048 Hz. An empty configuration ({}) is the
undispersed baseline. Harness-only keys set the outer layers:
``_skin_thickness__mm`` (default 1.29), ``_fat_thickness__mm`` (limb 4, FDI 0.3)
and ``_skin_conductivity__S_per_m`` (default 1). Only the bipolar channel is simulated. The ``myogen`` that is imported
(recorded in the output) must be the checkout under test; set PYTHONPATH if needed.
"""
import hashlib
import json
import sys
import time

import numpy as np
import quantities as pq
from neo import Block, Segment, SpikeTrain

import myogen
from myogen import set_random_seed
from myogen.simulator import Muscle, RecruitmentThresholds, SurfaceElectrodeArray, SurfaceEMG
from myogen.simulator.core.emg.surface.fiber_dispersion import SurfaceFiberDispersion
from myogen.utils.neo import signal_to_grid

scenario, label, out, config = sys.argv[1], sys.argv[2], sys.argv[3], json.loads(sys.argv[4])
n_jobs = int(sys.argv[5]) if len(sys.argv) > 5 else 6
if scenario not in ("limb", "fdi"):
    raise SystemExit(f"unknown scenario {scenario!r}")
skin = config.pop("_skin_thickness__mm", 1.29)
fat = config.pop("_fat_thickness__mm", 4.0 if scenario == "limb" else 0.3)
skin_sigma = config.pop("_skin_conductivity__S_per_m", 1.0)
if "nerve_cvs__m_per_s" in config:
    config["nerve_cvs__m_per_s"] = tuple(config["nerve_cvs__m_per_s"])
dispersion = SurfaceFiberDispersion(**config)

N_MU, FS = (100 if scenario == "limb" else 120), 2048.0
RAMP, HOLD, LEVEL = 2.0, 4.0, 0.30
DURATION = 2 * RAMP + HOLD + 1.0

set_random_seed(42)
thresholds, _ = RecruitmentThresholds(N=N_MU, recruitment_range__ratio=50, mode="fuglevand")
if scenario == "limb":
    muscle = Muscle(recruitment_thresholds=thresholds, radius__mm=15 * pq.mm, length__mm=80 * pq.mm,
                    fiber_density__fibers_per_mm2=350 * pq.mm**-2, mean_fiber_length__mm=70 * pq.mm,
                    var_fiber_length__mm=5 * pq.mm, radius_bone__mm=3 * pq.mm, fat_thickness__mm=fat * pq.mm,
                    skin_thickness__mm=skin * pq.mm, skin_conductivity__S_per_m=skin_sigma * pq.S / pq.m)
    radius = 15.0
else:
    muscle = Muscle(recruitment_thresholds=thresholds, skin_thickness__mm=skin * pq.mm, fat_thickness__mm=fat * pq.mm,
                    skin_conductivity__S_per_m=skin_sigma * pq.S / pq.m)
    radius = 6.91
muscle.distribute_innervation_centers()
muscle.generate_muscle_fiber_centers(verbose=False)
muscle.assign_mfs2mns(verbose=False, n_jobs=n_jobs)
counts = np.asarray(muscle.resulting_number_of_innervated_fibers)
fibre_hash = hashlib.sha1(b"".join(
    np.ascontiguousarray(muscle.resulting_fiber_assignment(i)).tobytes() for i in range(N_MU))).hexdigest()

rng = np.random.default_rng(123)
rte = np.asarray(thresholds, dtype=float)
rte = rte / rte.max() * 0.7


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
        rate = min(8.0 + 100 * (e - rte[mu]), 35.0)
        times.append(t)
        t += rng.gamma(1 / 0.2**2, 0.2**2 / rate)
    times = np.asarray([x for x in times if x < DURATION])
    active.append(len(times) > 0)
    trains.append(SpikeTrain(np.round(times * FS) / FS * pq.s, t_stop=DURATION * pq.s, sampling_rate=FS * pq.Hz))
block = Block()
seg = Segment()
seg.spiketrains = trains
block.segments.append(seg)
spike_hash = hashlib.sha1(b"".join(np.asarray(st.magnitude).tobytes() for st in trains)).hexdigest()
mus = [m for m in range(N_MU) if active[m] and counts[m] > 0]

if scenario == "limb":
    bip = SurfaceElectrodeArray(1, 1, 10 * pq.mm, 2.5 * pq.mm, center_point__mm_deg=(20 * pq.mm, 0 * pq.deg),
                                bending_radius__mm=(radius + fat + skin) * pq.mm,
                                differentiation_mode="bipolar_longitudinal")
    kernel = None
else:
    bip = SurfaceElectrodeArray(1, 1, 5 * pq.mm, 1.0 * pq.mm, center_point__mm_deg=(8 * pq.mm, 0 * pq.deg),
                                bending_radius__mm=(radius + fat + skin) * pq.mm,
                                differentiation_mode="bipolar_longitudinal")
    kernel = 150.0
set_random_seed(7)
semg = SurfaceEMG(muscle, [bip], sampling_frequency__Hz=FS * pq.Hz, MUs_to_simulate=mus,
                  internal_sampling_frequency__Hz=FS * pq.Hz, fiber_dispersion=dispersion,
                  iap_kernel_length__mm=kernel)
t0 = time.time()
muaps = semg.simulate_muaps(n_jobs=n_jobs, verbose=False, use_gpu=False)
sig0 = muaps.groups[0].segments[mus[0]].analogsignals[0]
templates = np.array([np.asarray(signal_to_grid(muaps.groups[0].segments[m].analogsignals[0].rescale(pq.mV)))
                      for m in mus])
emg = semg.simulate_surface_emg(block, verbose=False).groups[0].segments[0].analogsignals[0]
report = semg.fiber_dispersion_report or {}
np.savez(out, label=label, scenario=scenario, myogen=myogen.__file__, config=json.dumps(config), skin=skin, fat=fat, skin_sigma=skin_sigma, mus=mus, counts=counts,
         fibre_hash=fibre_hash, spike_hash=spike_hash, ramp=RAMP, hold=HOLD, level=LEVEL,
         bip_muaps=templates, pair_muap_fs=float(sig0.sampling_rate.rescale(pq.Hz).magnitude),
         bip_emg=np.asarray(signal_to_grid(emg.rescale(pq.mV))), fs=float(emg.sampling_rate.rescale(pq.Hz).magnitude),
         report=np.array({k: v for k, v in report.items() if k != "config"}, dtype=object),
         seconds=time.time() - t0, allow_pickle=True)
print(label, "fibres", int(counts[mus].sum()), "secs", round(time.time() - t0), "fibre", fibre_hash[:10],
      "spikes", spike_hash[:10], flush=True)
