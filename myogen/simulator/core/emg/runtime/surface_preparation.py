"""Reusable output functionals for the public cylindrical conductor."""
import numpy as np
from .cache import fingerprint


class OutputResponse:
    """Prepare A.T w=c (ordinary transpose, not conjugate transpose)."""
    def __init__(self, matrix):
        matrix = np.asarray(matrix)
        if matrix.shape[-1] not in (5, 7) or matrix.shape[-2] != matrix.shape[-1]:
            raise ValueError("Expected five- or seven-variable conductor matrices")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("Conductor preparation contains nonfinite entries")
        selector = np.zeros((*matrix.shape[:-1], 1), dtype=matrix.dtype)
        selector[..., -2:, 0] = 1
        weights = np.linalg.solve(matrix.swapaxes(-1, -2), selector)
        residual = matrix.swapaxes(-1, -2) @ weights - selector
        scale = (np.linalg.norm(matrix, ord=np.inf, axis=(-2, -1)) *
                 np.max(np.abs(weights), axis=(-2, -1)) + 1)
        relative = np.max(np.abs(residual), axis=(-2, -1)) / scale
        if not np.all(np.isfinite(weights)) or np.any(relative > 1e-12):
            raise ValueError("Prepared conductor response failed its solve residual gate")
        self._weights = weights.copy()
        self._weights.flags.writeable = False
        self.maximum_backward_residual = float(np.max(relative, initial=0.0))

    def __call__(self, sources):
        return np.sum(self._weights[..., 0] * sources[..., 0], axis=-1)


class SurfacePreparation:
    """Per-call immutable preparations, reused only within one bound geometry."""
    def __init__(self):
        self.key = None
        self.grid = None
        self.matrix = None
        self.incomplete = None
        self.response = None
        self.electrode = None
        self.phase_key = None
        self.phase = None
        self.source_key = None
        self.source = None
        self.boundary_key = None
        self.boundary = None
        self.transfer_key = None
        self.transfer_grid = None

    def boundary_terms(self, theta, kz, inner, outer):
        """Reuse exact Bessel terms at fixed conductor boundaries.

        Fibre radius is deliberately absent: its two Bessel evaluations remain
        per fibre. Hash the realized grids and radii, and publish only after
        all four arrays have been successfully prepared.
        """
        from ..surface.volume_conductor import log_In, log_Kn
        key = fingerprint(theta, kz, inner, outer)
        if key != self.boundary_key:
            values = (
                log_In(theta, inner * kz),
                log_Kn(theta, outer * kz),
                np.logaddexp(log_In(theta + 1, inner * kz),
                             log_In(theta - 1, inner * kz)) - np.log(2),
                np.logaddexp(log_Kn(theta + 1, outer * kz),
                             log_Kn(theta - 1, outer * kz)) - np.log(2),
            )
            for value in values:
                value.flags.writeable = False
            self.boundary = values
            self.boundary_key = key
        return self.boundary

    def source_current(self, kz, kt, psi, velocity, zone, left, right):
        from .surface_source import SourceGrid
        key = fingerprint(kz, kt, psi, velocity)
        if key != self.source_key:
            self.source = SourceGrid(kz, kt, psi, velocity)
            self.source_key = key
        return self.source(zone, left, right)

    def phi(self, source, response, positions, frequencies):
        """Contract the unchanged source with separable electrode z phases."""
        key = fingerprint(positions, frequencies)
        if key != self.phase_key:
            self.phase = np.exp(1j * positions[..., None] * frequencies)
            self.phase.flags.writeable = False
            self.phase_key = key
        weights = response * self.phase
        return ((weights.reshape(-1, len(frequencies)) @ source)
                .reshape(*positions.shape, source.shape[-1]) *
                ((frequencies[1] - frequencies[0]) / (2 * np.pi)))

    def arrays(self):
        values = [self.matrix, self.incomplete, self.phase]
        values.extend(self.boundary or ())
        values.extend(self.grid or ())
        values.extend(self.transfer_grid or ())
        values.extend(self.electrode or ())
        if self.response is not None:
            values.append(self.response._weights)
        if self.source is not None:
            values.extend(self.source.arrays())
        return [value for value in values if isinstance(value, np.ndarray)]

    def bind_transfer(self, *dependencies):
        key = fingerprint(*dependencies)
        if self.transfer_key is not None and key != self.transfer_key:
            raise ValueError("Prepared conductor dependencies changed")
        self.transfer_key = key

    def freeze_transfer_grid(self, values):
        self.transfer_grid = tuple(values)
        for value in self.transfer_grid:
            if isinstance(value, np.ndarray):
                value.flags.writeable = False

    def bind(self, key):
        if self.key is not None and key != self.key:
            raise ValueError("Prepared surface dependencies changed")
        self.key = key

    def freeze_grid(self, values):
        self.grid = tuple(values)
        for value in self.grid:
            if isinstance(value, np.ndarray):
                value.flags.writeable = False

    def conductor(self, matrix, incomplete, bone):
        if self.response is None:
            self.matrix = matrix
            self.incomplete = incomplete
            self.matrix.flags.writeable = False
            self.incomplete.flags.writeable = False
            flat = matrix.reshape(-1, 7, 7)
            if not bone:
                flat = flat[..., 2:, [1, 3, 4, 5, 6]]
            self.response = OutputResponse(flat)
        return self.response
