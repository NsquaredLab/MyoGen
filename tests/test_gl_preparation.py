"""Corrected-conductor algebra/regression gates, not population evidence."""
import numpy as np
import pytest
from myogen.simulator.core.emg.runtime.surface_preparation import SurfacePreparation
from myogen.simulator.core.emg.surface.volume_conductor import layered_cylinder_transfer

GEOMETRY = dict(r=40., r_bone=10., th_fat=2., th_skin=1., R=30.,
                sig_muscle_rho=.1, sig_muscle_z=.5, sig_fat=.04,
                sig_skin=1., sig_bone=.02)

@pytest.mark.parametrize("bone", [0., 1., 10.])
@pytest.mark.parametrize("N,M", [(512,32), (513,33), (2048,32)])
def test_prepared_corrected_system_entire_grid_and_varying_source(bone, N, M):
    kz = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(N, d=150./(N-1)))
    theta = np.fft.fftshift(np.fft.fftfreq(M, d=1./M))
    prep = SurfacePreparation()
    geom = dict(GEOMETRY, r_bone=bone)
    for radius in (30., 25.):
        geom["R"] = radius
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            ref, _, _ = layered_cylinder_transfer(kz, theta, **geom)
            actual, _, _ = layered_cylinder_transfer(kz, theta, _prepared=prep, **geom)
        assert actual.shape == (N,M)
        assert np.all(np.isfinite(actual))
        gate = 1e-12 + 1e-9*np.max(np.abs(ref), axis=0)
        assert np.all(np.max(np.abs(actual-ref),axis=0) <= gate)
        assert np.all(actual[kz == 0] == 0)
        assert prep.response.maximum_backward_residual <= 1e-12
    assert all(not value.flags.writeable for value in prep.transfer_grid if isinstance(value,np.ndarray))

@pytest.mark.parametrize("kz", [np.array([0.]), np.array([])])
def test_empty_solve_grid_preserves_reference(kz):
    theta=np.arange(-2.,3.)
    ref, _, _ = layered_cylinder_transfer(kz, theta, **GEOMETRY)
    actual, _, _ = layered_cylinder_transfer(kz, theta, _prepared=SurfacePreparation(), **GEOMETRY)
    np.testing.assert_array_equal(actual,ref)

@pytest.mark.parametrize("changed", ["kz","theta",*GEOMETRY.keys()])
def test_transfer_rejects_changed_dependencies_except_fibre_radius(changed):
    kz, theta = np.array([-.2,0.,.1]), np.array([-1.,0.,1.])
    prep=SurfacePreparation()
    layered_cylinder_transfer(kz,theta,_prepared=prep,**GEOMETRY)
    geom=GEOMETRY.copy()
    if changed == "kz":
        kz[0] -= .1
    elif changed == "theta":
        theta[0] -= 1.
    else:
        geom[changed] += .01
    if changed == "R":
        layered_cylinder_transfer(kz,theta,_prepared=prep,**geom)
    else:
        with pytest.raises(ValueError,match="dependencies changed"):
            layered_cylinder_transfer(kz,theta,_prepared=prep,**geom)
