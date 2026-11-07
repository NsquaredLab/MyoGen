"""CPU implementation of the public iEMG finite-difference current density.

Only the explicitly selected execution path uses this kernel. The public
bioelectric implementation remains the reference and diagnostic API.
"""
import math

import numpy as np
from numba import njit

from ..intramuscular.bioelectric import shift_padding


@njit(cache=True, nogil=True, fastmath=False)
def _density(t, z, active, mirror, short_mask, end_mask, zi, v, dz, positive, scale):
    rows, columns = len(active), len(t)
    wave = np.empty((rows + 1, columns), dtype=np.float64)
    for row in range(rows + 1):
        # A masked derivative needs only its two endpoints. This skips work
        # whose contribution is exactly zero, without truncating any tail.
        needed = (row < rows and active[row]) or (row > 0 and active[row - 1])
        for col in range(columns):
            value = 0.0
            if needed:
                # zeta = v t - |z - zi| in mm, Rosenfalck length scale 1 mm, as
                # in bioelectric.get_current_density (until 0.13.0 both read
                # -2 * (...) and scaled by 4: half the IAP length, 8x the current).
                if positive:
                    x = -(z[row] - zi - v * t[col])
                else:
                    x = -(-z[row] + zi - v * t[col])
                if x > 0:
                    value = 96.0 * (3 * x**2 - x**3) * math.exp(-x)
                if positive:
                    value = -value
            wave[row, col] = value
    # Ascending rows retain the unmodified next endpoint until it is consumed.
    for row in range(rows):
        for col in range(columns):
            if active[row]:
                wave[row, col] = (wave[row + 1, col] - wave[row, col]) / dz
            else:
                wave[row, col] = 0.0
    result = np.empty((rows, columns), dtype=np.float64)
    for row in range(rows):
        source = mirror[row]
        for col in range(columns):
            short = wave[source, col] if source >= 0 and short_mask[row] else 0.0
            # The mirrored wave is already d2V/dz2 of the opposite wave, so the
            # two add (they were subtracted until 0.13.0).
            value = wave[row, col] + short
            result[row, col] = (value if end_mask[row] else 0.0) * scale
    return result


def current_density(t, z, zi, L1, L2, v, d=55e-3,
                    suppress_endplate_density=True, endplate_width=0.5):
    """Evaluate the same float64 model with two dense numerical buffers."""
    t, z = np.asarray(t).ravel(), np.asarray(z).ravel()
    zi, L1, L2, v, d = map(float, (zi, L1, L2, v, d))
    dz = np.mean(np.diff(z))
    extended = np.concatenate((z, z[[-1]] + dz))
    positive = L1 >= L2
    if positive:
        coordinate = z - zi - L1 / 2
        active = (coordinate <= L1 / 2) & (coordinate >= -L1 / 2) & ((z-zi)/v > 0)
        coordinate = z - zi + L2 / 2
        short_mask = (coordinate <= L2 / 2) & (coordinate >= -L2 / 2)
    else:
        coordinate = z - zi + L2 / 2
        active = (coordinate <= L2 / 2) & (coordinate >= -L2 / 2) & ((-z+zi)/v > 0)
        coordinate = z - zi - L1 / 2
        short_mask = (coordinate <= L1 / 2) & (coordinate >= -L1 / 2)
    shift = round(float((L1 + L2 - float(extended.max()) + L2 - L1) / dz))
    # Reuse the public padding convention, including its large-negative-shift
    # slicing semantics. Zero is a sentinel; valid row indices are offset by 1.
    mirror = shift_padding(np.arange(len(z), 0, -1), shift, axis=0) - 1
    end_mask = ((z <= zi-endplate_width) | (z >= zi+endplate_width)
                if suppress_endplate_density else np.ones(len(z), dtype=bool))
    scale = (1.01 / 1000.0) * (np.pi * (d / 2)**2)
    return _density(t, extended, active, mirror, short_mask, end_mask,
                    zi, v, dz, positive, scale)
