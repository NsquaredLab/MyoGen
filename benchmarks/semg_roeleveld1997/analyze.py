"""Fall-off of the monopolar negative peak (Nu) with depth and observation distance.

Usage: python analyze.py <arm.npz> [<arm.npz> ...]

Nu is the peak of the main propagating phase, per fibre (Roeleveld et al.
normalised measured MUAPs by unit size). Its sign is taken from the
shallowest unit at position A, 24 mm from the end-plate. For each arm:

- ``above``: Nu above the unit, highest over the four rows (their Fig. 4 scale),
  relative to the shallowest point unit.
- depth exponent: slope of log Nu at A (24 mm row) against log centroid depth,
  per territory radius.
- lateral exponent: slope of log Nu against log distance from the territory
  centroid to the electrode over positions A-D (both sides averaged), per unit,
  then the median over depths.
"""
import json
import sys

import numpy as np

ROW_24, COL_A = 2, 3


def load(path):
    d = np.load(path, allow_pickle=True)
    muaps = d["muaps"]  # unit, row, col, time
    point = np.flatnonzero(d["rho"] == 0)
    ref = point[np.argmin(d["depth"][point])] if len(point) else int(np.argmin(d["depth"]))
    trace = muaps[ref, ROW_24, COL_A]
    sign = np.sign(trace[np.argmax(np.abs(trace))])
    nu = (sign * muaps).max(axis=-1) / d["n_fibres"][:, None, None]
    side = 0.5 * (nu[:, ROW_24, COL_A:] + nu[:, ROW_24, COL_A::-1])  # A, B, C, D
    arc = d["col_arc_mm"][COL_A:]
    rc = float(d["r_total"]) - d["centroid_depth"]
    r = np.sqrt(rc[:, None] ** 2 + d["r_total"] ** 2 - 2 * rc[:, None] * d["r_total"] * np.cos(arc / d["r_total"]))
    return d, nu, side, r, ref


def slope(x, y):
    return -np.polyfit(np.log(x), np.log(y), 1)[0]


for path in sys.argv[1:]:
    d, nu, side, r, ref = load(path)
    cfg = json.loads(str(d["config"]))
    above = nu[:, :, COL_A].max(axis=1)
    print(f"\n{path}  skin {cfg.get('skin_sigma', 1.0)} S/m, fat {cfg.get('fat_sigma', 0.05)} S/m, "
          f"M {cfg.get("M", 128)}, n_t {cfg.get("n_t", 256)}")
    print("rho  depth  centroid  above/ref  Nu_A(24)  A:B:C:D        lateral_exp")
    for mu in range(len(d["rho"])):
        print(f"{d['rho'][mu]:4.1f} {d['depth'][mu]:6.1f} {d['centroid_depth'][mu]:8.2f} "
              f"{above[mu] / above[ref]:9.3f} {side[mu, 0]:.3e} "
              f"{':'.join(f'{v:.2f}' for v in side[mu] / side[mu, 0])}  {slope(r[mu], side[mu]):.2f}")
    for rho in np.unique(d["rho"]):
        sel = np.flatnonzero(d["rho"] == rho)
        if len(sel) > 1:
            lat = [slope(r[mu], side[mu]) for mu in sel]
            print(f"rho {rho:4.1f}: depth exponent {slope(d['centroid_depth'][sel], side[sel, 0]):.2f} "
                  f"(above, max over rows {slope(d['centroid_depth'][sel], above[sel]):.2f}), "
                  f"lateral exponent median {np.median(lat):.2f} [{min(lat):.2f}, {max(lat):.2f}]")
