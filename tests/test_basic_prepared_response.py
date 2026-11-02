"""Algebra-only contract checks; these do not establish sEMG waveform parity."""
import numpy as np
import pytest
from myogen.simulator.core.emg.runtime.surface_preparation import OutputResponse, SurfacePreparation


@pytest.mark.parametrize("variables", [5, 7])
def test_transposed_output_response_matches_linear_system_identity(variables):
    rng = np.random.default_rng(184)
    matrix = rng.normal(size=(12, variables, variables))
    matrix += np.eye(variables) * 8
    sources = rng.normal(size=(12, variables, 1))
    result = np.linalg.solve(matrix, sources)
    expected = result[..., -2, 0] + result[..., -1, 0]
    response = OutputResponse(matrix)
    np.testing.assert_allclose(response(sources), expected, rtol=1e-12, atol=1e-14)
    matrix[:] = 0
    np.testing.assert_allclose(response(sources), expected, rtol=1e-12, atol=1e-14)


def test_nonfinite_and_singular_matrices_are_rejected():
    with pytest.raises(ValueError):
        OutputResponse(np.full((2, 5, 5), np.nan))
    with pytest.raises(np.linalg.LinAlgError):
        OutputResponse(np.zeros((2, 7, 7)))


def test_prepared_phase_contraction_is_the_same_complex_sum():
    """Pure matrix contraction identity; no physiological signal evidence."""
    rng = np.random.default_rng(194)
    source = rng.normal(size=(13, 17)) + 1j * rng.normal(size=(13, 17))
    response = rng.normal(size=(2, 3, 13))
    position = rng.normal(size=(2, 3))
    frequency = np.linspace(-2, 2, 13)
    expected = np.sum(source[None, None] * response[..., None] *
                      np.exp(1j * position[..., None, None] * frequency[None, None, :, None]),
                      axis=2) * (frequency[1] - frequency[0]) / (2 * np.pi)
    prepared = SurfacePreparation()
    actual = prepared.phi(source, response, position, frequency)
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-13)
    # Position mutation must invalidate the reusable phase.
    position += .5
    other = SurfacePreparation().phi(source, response, position, frequency)
    np.testing.assert_array_equal(prepared.phi(source, response, position, frequency), other)


@pytest.mark.parametrize("variables", [5, 7])
def test_scaled_system_retains_output_identity(variables):
    matrix = np.broadcast_to(np.eye(variables), (2, variables, variables)).copy()
    matrix *= np.geomspace(1e-10, 1e10, variables)[None, :, None]
    source = np.ones((2, variables, 1))
    response = OutputResponse(matrix)
    result = np.linalg.solve(matrix, source)
    np.testing.assert_allclose(response(source), result[..., -2, 0] + result[..., -1, 0],
                               rtol=1e-12, atol=1e-14)

@pytest.mark.parametrize("bone_radius", [0., .7])
def test_boundary_terms_reuse_only_identical_dependencies(monkeypatch, bone_radius):
    """Special-function/cache contract only, not physiological evidence."""
    from myogen.simulator.core.emg.surface import volume_conductor as module
    theta, kz = np.meshgrid(np.arange(4.), np.linspace(.01, 2., 7), indexing="ij")
    am, bm = bone_radius, 3.
    prepared = SurfacePreparation()
    calls = [0, 0]
    original_i, original_k = module.log_In, module.log_Kn
    def log_i(*args):
        calls[0] += 1
        return original_i(*args)
    def log_k(*args):
        calls[1] += 1
        return original_k(*args)
    monkeypatch.setattr(module, "log_In", log_i)
    monkeypatch.setattr(module, "log_Kn", log_k)
    with np.errstate(divide="ignore", invalid="ignore"):
        first = prepared.boundary_terms(theta, kz, am, bm)
        assert calls == [3, 3]
        assert prepared.boundary_terms(theta, kz, am, bm) is first
        assert calls == [3, 3]
        assert all(not value.flags.writeable for value in first)
        assert all(any(value is tracked for tracked in prepared.arrays()) for value in first)
        for dependency in ("theta", "kz", "am", "bm"):
            if dependency == "theta":
                theta[0, 0] += 1
            elif dependency == "kz":
                kz[0, 0] += .1
            elif dependency == "am":
                am += .1
            else:
                bm += .1
            previous = prepared.boundary_terms(theta, kz, am, bm)
            assert previous is not first
            expected = (
                original_i(theta, am*kz), original_k(theta, bm*kz),
                np.logaddexp(original_i(theta+1, am*kz), original_i(theta-1, am*kz))-np.log(2),
                np.logaddexp(original_k(theta+1, bm*kz), original_k(theta-1, bm*kz))-np.log(2),
            )
            for actual, reference in zip(previous, expected):
                np.testing.assert_array_equal(actual, reference)
            first = previous


def test_boundary_cache_is_not_published_after_failed_preparation(monkeypatch):
    """Allocation-failure contract only."""
    from myogen.simulator.core.emg.surface import volume_conductor as module
    theta, kz = np.meshgrid(np.arange(3.), np.linspace(.1, 1., 4), indexing="ij")
    prepared = SurfacePreparation()
    original = module.log_Kn
    def fail(*args):
        raise MemoryError("injected preparation failure")
    monkeypatch.setattr(module, "log_Kn", fail)
    with pytest.raises(MemoryError):
        prepared.boundary_terms(theta, kz, .7, 3.)
    assert prepared.boundary is None and prepared.boundary_key is None
    monkeypatch.setattr(module, "log_Kn", original)
    actual = prepared.boundary_terms(theta, kz, .7, 3.)
    assert len(actual) == 4
