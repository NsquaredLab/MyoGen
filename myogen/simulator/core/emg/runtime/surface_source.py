"""Reuse exact frequency sums/differences in the public source expression."""
import numpy as np

from .cache import fingerprint


def _compress(grid):
    values, inverse = np.unique(grid, return_inverse=True)
    # A generic grid may have little repetition. Keep its useful grid directly
    # rather than retaining an additional full-sized lookup table and index.
    if values.size >= grid.size / 2:
        return grid, None
    return values, inverse.reshape(grid.shape)


class SourceGrid:
    """Exact lookup, without rounding, interpolation or frequency truncation."""
    def __init__(self, kz, kt, psi, velocity):
        self.kz = np.array(kz, copy=True)
        self.psi = np.array(psi, copy=True)
        self.velocity = float(velocity)
        self.eps, self.eps_index = _compress(self.kz[:, None] + kt[None, :]/self.velocity)
        self.beta, self.beta_index = _compress(self.kz[:, None] - kt[None, :]/self.velocity)
        self.base_key = None
        self.base = None
        for value in self.arrays():
            value.flags.writeable = False

    def arrays(self):
        return [value for value in (self.kz, self.psi, self.eps, self.eps_index,
                self.beta, self.beta_index, self.base) if value is not None]

    def __call__(self, zone, left, right):
        key = fingerprint(zone)
        if key != self.base_key:
            # Preserve the reference multiplication order. Only the identical
            # row phase is evaluated once instead of once per matrix element.
            phase = np.exp(-1j*self.kz*zone)
            self.base = 1j * np.multiply(self.kz[:, None]/self.velocity,
                np.multiply(self.psi[None, :], phase[:, None]))
            self.base.flags.writeable = False
            self.base_key = key
        a = np.exp(-1j*self.eps*left/2) * (np.sinc(self.eps*left/2/np.pi)*left)
        b = np.exp(1j*self.beta*right/2) * (np.sinc(self.beta*right/2/np.pi)*right)
        if self.eps_index is not None:
            a = a[self.eps_index]
        if self.beta_index is not None:
            b = b[self.beta_index]
        np.subtract(a, b, out=a)
        np.multiply(self.base, a, out=a)
        return a
