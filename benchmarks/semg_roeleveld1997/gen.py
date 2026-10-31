"""Simulate monopolar surface MUAPs in the biceps-like cylinder of Roeleveld et al. (1997).

Usage: python gen.py <out.npz> <config-json>

Geometry and conductivities follow Roeleveld et al. (1997), J Electromyogr
Kinesiol 7:221-232 (their model III): cylinder radius 35 mm, muscle radius 32 mm,
fat 1 mm, skin 2 mm, no bone; muscle 0.1 S/m radial and 0.5 S/m axial, fat
0.05 S/m, skin 1 S/m; fibres 120 mm long, conduction velocity 3.8 m/s;
monopolar contacts 1.2 mm in diameter. Config keys override these
(``skin_sigma``, ``fat_sigma``, ``skin_mm``, ``fat_mm``) and set the motor
units and the numerical resolution:

- ``depths_mm``: territory centre depth below the skin surface.
- ``rhos_mm``: territory radius; 0 puts ``point_fibres`` fibres at one point,
  as Roeleveld et al. did, otherwise ``n_fibres`` fibres are drawn uniformly in
  the disc and clipped to the muscle (so the realised centroid depth is stored).
- ``M`` (angular harmonics), ``n_t`` (output samples), ``fs`` and ``fs_internal`` (Hz).
- ``execution``: ``"policy"`` (default) uses the optimised CPU route
  (``ExecutionPolicy``), ``"default"`` the original per-fibre route.
- ``centre``: ``"roeleveld"`` puts the electrode rows 0, 12, 24 and 36 mm from
  each unit's end-plate (their electrodes 1-4); ``"zone"`` centres the rows on
  the end-plate (-18, -6, 6, 18 mm), to check the end-plate replication.

Each unit is simulated on its own with seven columns around the circumference
at 0, +-12, +-24 and +-36 mm of arc (their positions A-D on both sides), so the
electrode rows can follow that unit's end-plate. ``SurfaceEMG`` draws each
unit's end-plate offset from the global generator as its first draw; this
script replays that draw from the same seed (``gen.py`` checks it with
``centre: "zone"``). The ``Muscle`` is built without its own fibre placement:
the fibre centres and their unit assignment are set directly, which are the
only fibre data ``SurfaceEMG`` reads.
"""
import json
import sys
import time

import numpy as np
import quantities as pq

import myogen
from myogen import get_random_generator, set_random_seed
from myogen.simulator import ExecutionPolicy, Muscle, RecruitmentThresholds, SurfaceElectrodeArray, SurfaceEMG
from myogen.utils.neo import signal_to_grid

out, cfg = sys.argv[1], json.loads(sys.argv[2])
SKIN_SIGMA = cfg.get("skin_sigma", 1.0)
FAT_SIGMA = cfg.get("fat_sigma", 0.05)
SKIN_MM, FAT_MM, R_MUSCLE = cfg.get("skin_mm", 2.0), cfg.get("fat_mm", 1.0), 32.0
R_TOTAL = R_MUSCLE + FAT_MM + SKIN_MM
DEPTHS = cfg.get("depths_mm", [5.0, 6.0, 7.5, 10.0, 12.5, 15.0])
RHOS = cfg.get("rhos_mm", [0.0, 2.5, 5.0, 7.5, 12.5])
N_FIBRES, POINT_FIBRES = cfg.get("n_fibres", 300), cfg.get("point_fibres", 20)
M, N_T = cfg.get("M", 128), cfg.get("n_t", 256)
FS, FS_INT = cfg.get("fs", 2048.0), cfg.get("fs_internal", 10000.0)
CENTRE = cfg.get("centre", "roeleveld")
FAST = cfg.get("execution", "policy") == "policy"
CV, FIBRE_MM, FIBRE_VAR_MM = 3.8, 120.0, 4.0
SEED_FIBRES, SEED_EMG = cfg.get("seed", 1), 7

units = [(rho, depth) for rho in RHOS for depth in DEPTHS]
rng = np.random.default_rng(SEED_FIBRES)
positions, assignment, centroid_depth = [], [], []
for mu, (rho, depth) in enumerate(units):
    centre = np.array([R_TOTAL - depth, 0.0])
    if rho == 0:
        xy = np.repeat(centre[None], POINT_FIBRES, axis=0)
    else:
        xy = np.empty((0, 2))
        while len(xy) < N_FIBRES:
            cand = centre + rho * np.sqrt(rng.uniform(size=(4 * N_FIBRES, 1))) * np.stack(
                [np.cos(a := rng.uniform(0, 2 * np.pi, 4 * N_FIBRES)), np.sin(a)], axis=1)
            xy = np.vstack([xy, cand[np.hypot(*cand.T) < R_MUSCLE - 0.25]])
        xy = xy[:N_FIBRES]
    positions.append(xy)
    assignment.append(np.full(len(xy), mu))
    centroid_depth.append(R_TOTAL - np.hypot(*xy.mean(axis=0)))

thresholds, _ = RecruitmentThresholds(N=len(units), recruitment_range__ratio=10, mode="fuglevand")
muscle = Muscle(thresholds, radius__mm=R_MUSCLE * pq.mm, length__mm=150 * pq.mm,
                mean_conduction_velocity__m_per_s=CV * pq.m / pq.s, mean_fiber_length__mm=FIBRE_MM * pq.mm,
                var_fiber_length__mm=FIBRE_VAR_MM * pq.mm, radius_bone__mm=0 * pq.mm,
                fat_thickness__mm=FAT_MM * pq.mm, skin_thickness__mm=SKIN_MM * pq.mm,
                muscle_conductivity_radial__S_per_m=0.1 * pq.S / pq.m,
                muscle_conductivity_longitudinal__S_per_m=0.5 * pq.S / pq.m,
                fat_conductivity__S_per_m=FAT_SIGMA * pq.S / pq.m,
                skin_conductivity__S_per_m=SKIN_SIGMA * pq.S / pq.m)
muscle._muscle_fiber_centers__mm = np.vstack(positions) * pq.mm
muscle._assignment = np.concatenate(assignment).astype(float)

# Replay SurfaceEMG's end-plate draw: uniform over 10 % of the mean fibre length.
set_random_seed(SEED_EMG)
zones = get_random_generator().uniform(-FIBRE_MM * 0.1 / 2, FIBRE_MM * 0.1 / 2, size=len(units))

muaps, t0 = [], time.time()
for mu in range(len(units)):
    z_centre = zones[mu] + (18.0 if CENTRE == "roeleveld" else 0.0)
    ring = SurfaceElectrodeArray(4, 7, 12 * pq.mm, 0.6 * pq.mm, center_point__mm_deg=(z_centre * pq.mm, 0 * pq.deg),
                                 bending_radius__mm=R_TOTAL * pq.mm, differentiation_mode="monopolar")
    set_random_seed(SEED_EMG)
    semg = SurfaceEMG(muscle, [ring], sampling_frequency__Hz=FS * pq.Hz, sampling_points_in_t_and_z_domains=N_T,
                      sampling_points_in_theta_domain=M, MUs_to_simulate=[mu],
                      internal_sampling_frequency__Hz=FS_INT * pq.Hz)
    block = semg.simulate_muaps(n_jobs=1, verbose=False,
                                execution=ExecutionPolicy(max_workers=1) if FAST else None)
    sig = block.groups[0].segments[mu].analogsignals[0]
    muaps.append(np.moveaxis(np.asarray(signal_to_grid(sig.rescale(pq.mV))), 0, -1))  # row, col, time
    fs_muap = float(sig.sampling_rate.rescale(pq.Hz).magnitude)
    print(f"unit {mu} rho {units[mu][0]} depth {units[mu][1]} fibres {len(positions[mu])} "
          f"{time.time() - t0:.0f} s", flush=True)

np.savez(out, myogen=myogen.__file__, config=json.dumps(cfg), rho=[u[0] for u in units],
         depth=[u[1] for u in units], centroid_depth=centroid_depth, n_fibres=[len(p) for p in positions],
         zones=zones, muaps=np.array(muaps), fs_muap=fs_muap, r_total=R_TOTAL, cv=CV,
         row_z_from_zone=np.array([0, 12, 24, 36]) - (0 if CENTRE == "roeleveld" else 18),
         col_arc_mm=np.arange(-3, 4) * 12.0, seconds=time.time() - t0)
