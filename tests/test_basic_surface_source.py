"""Algebra/cache contracts only; public-cohort runs supply waveform evidence."""
import numpy as np

from myogen.simulator.core.emg.runtime.surface_source import SourceGrid


def expression(kz, kt, psi, v, zi, left, right):
    z, t = np.meshgrid(kz, kt, indexing="ij")
    value = 1j * np.multiply(z/v, np.multiply(psi[None, :], np.exp(-1j*z*zi)))
    eps, beta = z+t/v, z-t/v
    a = np.exp(-1j*eps*left/2) * (np.sinc(eps*left/2/np.pi)*left)
    b = np.exp(1j*beta*right/2) * (np.sinc(beta*right/2/np.pi)*right)
    return value * (a-b)


def test_exact_frequency_factorization_and_changed_scalar_inputs():
    kz = np.linspace(-3, 3, 17)
    kt = kz*4
    psi = np.arange(17) + 1j*np.arange(17)[::-1]
    grid = SourceGrid(kz, kt, psi, 4.)
    for zi, left, right in ((.1, 2., 3.), (.1, 3., 4.), (-.2, 1., 2.)):
        np.testing.assert_allclose(grid(zi, left, right),
            expression(kz, kt, psi, 4., zi, left, right), rtol=1e-14, atol=1e-14)
    assert all(not array.flags.writeable for array in grid.arrays())


def test_distinct_adjacent_floating_values_are_never_quantized():
    kz = np.array([1., np.nextafter(1., 2.), 1.5])
    kt = np.array([0., .25, .5])
    psi = np.array([1+1j, 2+2j, 3+3j])
    grid = SourceGrid(kz, kt, psi, 1.)
    reconstructed = grid.eps if grid.eps_index is None else grid.eps[grid.eps_index]
    np.testing.assert_array_equal(reconstructed, kz[:, None] + kt[None, :])
    assert reconstructed[0, 0] != reconstructed[1, 0]
    assert grid.eps_index is None and grid.beta_index is None
    np.testing.assert_array_equal(grid(.2, 2., 3.),
                                  expression(kz, kt, psi, 1., .2, 2., 3.))


def test_source_preparation_tracks_all_actual_grid_dependencies():
    from myogen.simulator.core.emg.runtime.surface_preparation import SurfacePreparation
    preparation = SurfacePreparation()
    kz, kt, psi = np.arange(5.), np.arange(5.), np.arange(5.)+1j
    preparation.source_current(kz, kt, psi, 4., .2, 2., 3.)
    old = preparation.source
    preparation.source_current(kz, kt, psi, 4., .3, 2., 3.)
    assert preparation.source is old
    velocity = 4.
    for dependency in (psi, kz, kt, None):
        old = preparation.source
        if dependency is None:
            velocity += .5
        else:
            dependency[0] += .25
        actual = preparation.source_current(kz, kt, psi, velocity, .3, 2., 3.)
        assert preparation.source is not old
        np.testing.assert_array_equal(actual,
            expression(kz, kt, psi, velocity, .3, 2., 3.))
