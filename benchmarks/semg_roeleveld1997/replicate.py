"""Step 2 of the README rule with Roeleveld's end-plate scatter added.

Usage: python replicate.py <III.npz> <IIb.npz> [sd_mm ...]

Inputs come from ``gen.py`` with ``z_rows_mm`` (one column above each point
unit, rows every 1 mm from its end-plate). MyoGen's cylinder is unbounded
along the fibres, so moving a fibre's end-plate by d gives the same potential
as moving the electrode by -d. A unit whose fibres have end-plates scattered
normally with SD ``sd`` (Roeleveld et al. 1997: 3 mm) is therefore the
single-fibre profile smoothed along z by that Gaussian. The fibre ends move
with the end-plate here; in their model the ends scatter on their own.
"""
import sys

import numpy as np

TARGET = {"III": (0.43, 0.09), "IIb": (0.35, 0.046)}
ROWS = (0, 12, 24, 36)


def nu_above(path, sd):
    d = np.load(path, allow_pickle=True)
    z = d["row_z_from_zone"]
    prof = d["muaps"][:, :, 0, :] / d["n_fibres"][:, None, None]  # unit, row, time
    if sd > 0:
        k = np.exp(-0.5 * (np.arange(-4 * sd, 4 * sd + 1) / sd) ** 2)
        k /= k.sum()
        half = len(k) // 2
        need = [i for i, v in enumerate(z) if v in ROWS]
        if z[need[0]] - z[0] < half or z[-1] - z[need[-1]] < half:
            raise SystemExit(f"rows {z[0]}..{z[-1]} too short for a {sd} mm kernel")
        prof = np.stack([np.tensordot(k, prof[:, i - half:i + half + 1], axes=([0], [1])) for i in need], axis=1)
    else:
        prof = prof[:, [list(z).index(r) for r in ROWS]]
    ref = prof[0, ROWS.index(24)]
    sign = np.sign(ref[np.argmax(np.abs(ref))])
    above = (sign * prof).max(axis=-1).max(axis=1)
    return dict(zip(d["depth"], above))


for sd in [float(x) for x in sys.argv[3:]] or [0.0, 3.0]:
    res = {name: nu_above(path, sd) for name, path in zip(("III", "IIb"), sys.argv[1:3])}
    print(f"\nend-plate SD {sd} mm")
    for name, a in res.items():
        for depth, target in zip((7.5, 15.0), TARGET[name]):
            r = a[depth] / a[5.0]
            print(f"  {name} {depth:>4}/5 mm  {r:.3f}  target {target}  off {100 * (r / target - 1):+.0f} %  "
                  f"{'pass' if abs(r / target - 1) <= 0.15 else 'FAIL'}")
    r = res["IIb"][5.0] / res["III"][5.0]
    print(f"  IIb/III at 5 mm  {r:.2f}  target 4.17  off {100 * (r / 4.17 - 1):+.0f} %  "
          f"{'pass' if abs(r / 4.17 - 1) <= 0.15 else 'FAIL'}")
