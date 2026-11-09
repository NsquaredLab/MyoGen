"""Exponent-free check against Roeleveld 1997 Fig. 6 (model III curves, read from the figure, +-0.2)."""
import sys, glob, re
import numpy as np

FIG6_III = {5.0: [1.0, 1.4, 2.4, 4.2], 7.5: [1.4, 1.6, 2.5, 3.9], 15.0: [2.2, np.nan, np.nan, 3.4]}

def nu(path, row, sd):
    d = np.load(path, allow_pickle=True)
    z = list(d["row_z_from_zone"]); prof = d["muaps"] / d["n_fibres"][:, None, None, None]  # unit,row,col,t
    i = z.index(row)
    if sd > 0:
        k = np.exp(-0.5 * (np.arange(-4 * sd, 4 * sd + 1) / sd) ** 2); k /= k.sum(); h = len(k) // 2
        assert i - h >= 0 and i + h < len(z), "rows too short"
        p = np.tensordot(k, prof[:, i - h:i + h + 1], axes=([0], [1]))  # unit,col,t
    else:
        p = prof[:, i]
    ref = p[0, 0]; sign = np.sign(ref[np.argmax(np.abs(ref))])
    n = (sign * p).max(axis=-1)  # unit, col
    depth = list(d["depth"])
    return {dp: n[depth.index(dp)] / n[depth.index(5.0), 0] for dp in FIG6_III}

base = sys.argv[1]
arms = {float(re.search(r"sc_s([0-9.e-]+)_r0", f).group(1)): f for f in glob.glob(base + "/sc_s*_r0.npz")}
for row, sd in ((24, 3.0), (36, 0.0)):
    ref = nu(arms[1.0], row, sd)
    print(f"\nrow {row} mm, end-plate SD {sd}: measured/simulated Nu (1 = match), cols A B C D")
    for s in sorted(arms, reverse=True):
        a = nu(arms[s], row, sd)
        print(f"  skin {s:<9g}" + "  ".join(f"{dp:>4} mm: " + " ".join(f"{v:5.2f}" for v in np.array(FIG6_III[dp]) * ref[dp] / a[dp]) for dp in FIG6_III))
