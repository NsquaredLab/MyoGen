"""Physical units of the intramuscular SFAP (production path through ``MotorUnitSim.calc_sfaps``).

Reference: an independent line source in an anisotropic infinite medium, in SI units.
Rosenfalck V_m(zeta) = 96 zeta^3 exp(-zeta) - 90 mV with zeta in mm [rosenfalck1969];
I_m = sigma_i pi (d/2)^2 d2V_m/dz2 with sigma_i = 1.01 S/m [rosenfalck1969];
phi = 1/(4 pi sigma_r) sum I_m dz / sqrt(K r^2 + (z - z_e)^2), K = sigma_z / sigma_r,
sigma_r = 0.063 S/m and sigma_z = 0.33 S/m [andreassenrosenfalck1980] (as cited in ``bioelectric.py``).
Two waves leave the end-plate; the comparison window ends before they reach the fibre ends.
"""
import numpy as np
import pytest

from myogen.simulator.core.emg.intramuscular.motor_unit_sim import MotorUnitSim

D_MM = 0.055
V_MM_S = 4000.0
L_MM = 100.0
ZI_MM = 50.0
ZE_MM = 60.0
DT_S = 1 / 50_000
T_END_S = 40.0 / V_MM_S  # waves still 10 mm from the fibre ends


def _production_sfap(r_mm, dz_mm, use_gpu=False, ze_mm=ZE_MM, zi_mm=ZI_MM, t_end_s=T_END_S):
    sim = MotorUnitSim(
        muscle_fiber_centers__mm=np.array([[0.0, 0.0]]),
        muscle_length__mm=L_MM,
        muscle_fiber_diameters__mm=np.array([D_MM]),
        muscle_fiber_conduction_velocity__mm_per_s=np.array([V_MM_S]),
        nominal_center__mm=np.array([0.0, 0.0]),
    )
    sim._neuromuscular_z_coordinates__mm = np.array([[zi_mm]])
    sim.nerve_paths = np.zeros((1, 2))
    sim.calc_sfaps(0, DT_S, dz_mm, np.array([[r_mm, 0.0, ze_mm]]), verbose=False, use_gpu=use_gpu)
    sfap = np.asarray(sim.sfaps)[:, 0, 0]
    t = np.arange(sfap.size) * DT_S
    return sfap[t < t_end_s]


def _si_reference(r_mm, dz_mm=0.01):
    z = np.arange(0.0, L_MM + dz_mm, dz_mm)
    t = np.arange(0.0, T_END_S, DT_S)
    zeta = V_MM_S * t[None, :] - np.abs(z[:, None] - ZI_MM)
    zp = np.clip(zeta, 0.0, None)
    vm = 96.0 * zp**3 * np.exp(-zp) - 90.0
    d2v = np.gradient(np.gradient(vm, dz_mm, axis=0), dz_mm, axis=0) * 1e3  # mV/mm^2 -> V/m^2
    d2v[np.abs(z - ZI_MM) < 0.5] = 0.0  # end-plate zone suppressed, as in get_current_density
    i_m = 1.01 * np.pi * (D_MM * 1e-3 / 2) ** 2 * d2v  # A/m
    g = 1.0 / (4 * np.pi * 0.063 * np.sqrt(0.33 / 0.063 * (r_mm * 1e-3) ** 2 + ((z - ZE_MM) * 1e-3) ** 2))
    return (i_m * g[:, None]).sum(0) * dz_mm * 1e-3 * 1e3  # V -> mV


@pytest.mark.parametrize("r_mm", [0.1, 0.5, 1.0])
def test_sfap_matches_si_line_source(r_mm):
    sim = np.ptp(_production_sfap(r_mm, 0.01))
    ref = np.ptp(_si_reference(r_mm))
    assert sim == pytest.approx(ref, rel=0.03), f"r {r_mm} mm: production {sim:.4g} mV, SI {ref:.4g} mV"


def test_sfap_mirror_symmetric_about_endplate():
    """Both waves leave the end-plate with the same polarity: V_m(v t - |z - zi|) is symmetric in z."""
    plus = _production_sfap(0.3, 0.01, ze_mm=ZI_MM + 10.0)
    minus = _production_sfap(0.3, 0.01, ze_mm=ZI_MM - 10.0)
    np.testing.assert_allclose(minus, plus, rtol=0, atol=0.02 * np.ptp(plus))


def test_sfap_reflection_with_unequal_halves():
    """End-plate at 40 mm or at 60 mm (L1 < L2 and L1 > L2): reflected geometries record the same SFAP."""
    a = _production_sfap(0.3, 0.01, ze_mm=50.0, zi_mm=40.0, t_end_s=np.inf)
    b = _production_sfap(0.3, 0.01, ze_mm=50.0, zi_mm=60.0, t_end_s=np.inf)
    np.testing.assert_allclose(a, b, rtol=0, atol=0.02 * np.ptp(a))


def test_sfap_independent_of_spatial_step():
    a = np.ptp(_production_sfap(0.3, 0.01))
    b = np.ptp(_production_sfap(0.3, 0.02))
    assert a == pytest.approx(b, rel=0.02)


def test_gpu_sfap_matches_cpu():
    pytest.importorskip("cupy")
    cpu = _production_sfap(0.3, 0.01, use_gpu=False)
    gpu = _production_sfap(0.3, 0.01, use_gpu=True)
    np.testing.assert_allclose(gpu, cpu, rtol=1e-6, atol=1e-9 * np.ptp(cpu))
