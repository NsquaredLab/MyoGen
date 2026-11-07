"""Step 3 of the README rule: fall-off exponents with Roeleveld's end-plate scatter.

Usage: python falloff.py <dir> [sd_mm]

Reads ``sc_s<skin>_r<rho>.npz`` from ``run_scatter_arms.sh`` (one column per
position A-D at 0/12/24/36 mm of arc, rows every 1 mm from 12 to 36 mm from
the end-plate). Each profile is smoothed along z by a Gaussian of SD ``sd``
(default 3 mm, as in ``replicate.py``) and read at the 24 mm row. Nu is per
fibre, with its sign taken from the shallowest unit at A.

- depth exponent: -slope of log Nu at A against log centroid depth (5-15 mm).
- lateral exponent: -slope of log Nu against log distance from the territory
  centroid to the electrode over A-D, per unit; the median over depths.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

DEPTH_TARGET, DEPTH_TOL, LATERAL_MAX = 1.8, 0.3, 1.3
MEASURED = (0.15, 0.000488)


def slope(x, y):
    return -np.polyfit(np.log(x), np.log(y), 1)[0]


def arm(path, sd):
    d = np.load(path, allow_pickle=True)
    z = list(d["row_z_from_zone"])
    prof = d["muaps"] / d["n_fibres"][:, None, None, None]  # unit, row, col, time
    i = z.index(24)
    if sd > 0:
        half = int(round(4 * sd))
        if i - half < 0 or i + half >= len(z):
            raise SystemExit(f"rows {z[0]}..{z[-1]} too short for a {sd} mm kernel")
        k = np.exp(-0.5 * (np.arange(-half, half + 1) / sd) ** 2)
        trace = np.tensordot(k / k.sum(), prof[:, i - half:i + half + 1], axes=([0], [1]))
    else:
        trace = prof[:, i]
    ref = trace[int(np.argmin(d["centroid_depth"])), 0]
    nu = (np.sign(ref[np.argmax(np.abs(ref))]) * trace).max(axis=-1)  # unit, col
    rt, rc = float(d["r_total"]), float(d["r_total"]) - d["centroid_depth"]
    arc = d["col_arc_mm"]
    r = np.sqrt(rc[:, None] ** 2 + rt ** 2 - 2 * rc[:, None] * rt * np.cos(arc / rt))
    lateral = [slope(r[u], nu[u]) for u in range(len(nu))]
    return dict(depth_exp=slope(d["centroid_depth"], nu[:, 0]), lateral=lateral,
                centroid=d["centroid_depth"], nu=nu)


out = Path(sys.argv[1])
sd = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
rows = []
for path in sorted(out.glob("sc_s*_r*.npz")):
    skin, rho = map(float, re.match(r"sc_s(.+)_r(.+)\.npz", path.name).groups())
    a = arm(path, sd)
    ok = abs(a["depth_exp"] - DEPTH_TARGET) <= DEPTH_TOL and np.median(a["lateral"]) <= LATERAL_MAX
    rows.append((skin, rho, a["depth_exp"], float(np.median(a["lateral"])), min(a["lateral"]),
                 max(a["lateral"]), a["centroid"].min(), a["centroid"].max(), ok))

print(f"end-plate SD {sd} mm; target depth {DEPTH_TARGET} +- {DEPTH_TOL}, lateral <= {LATERAL_MAX} (measured 1.0)")
print("skin S/m   rho mm  depth_exp  lateral median [min, max]   centroid mm   meets both")
for skin, rho, de, lm, lmin, lmax, cmin, cmax, ok in sorted(rows):
    print(f"{skin:<9g} {rho:6.1f} {de:10.2f} {lm:9.2f} [{lmin:.2f}, {lmax:.2f}]   {cmin:5.2f}-{cmax:5.2f}   {'yes' if ok else 'no'}")
measured_ok = any(ok for skin, *_, ok in rows if skin in MEASURED)
one_ok = any(ok for skin, *_, ok in rows if skin == 1.0)
outcome = ("territory suffices" if measured_ok else "1 S/m needed" if one_ok else "neither: inconclusive")
print(f"outcome: {outcome}")
(out / f"falloff_sd{sd:g}.json").write_text(json.dumps(
    {"sd_mm": sd, "outcome": outcome,
     "rows": [dict(zip(("skin", "rho", "depth_exp", "lateral_median", "lateral_min", "lateral_max",
                        "centroid_min", "centroid_max", "meets_both"), map(lambda v: v if isinstance(v, bool) else float(v), r)))
              for r in rows]}, indent=1))
