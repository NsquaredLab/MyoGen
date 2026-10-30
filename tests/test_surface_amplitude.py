"""Absolute sEMG amplitude against an exact volume-conductor solution.

For a homogeneous, isotropic cylinder with an insulating surface the potential
of a point current has a closed Fourier-Bessel form (Jackson eq. 3.148 plus a
Neumann correction; Wronskian I_n K_n' - I_n' K_n = -1/x):

    phi(z, rho=r, theta) = sum_n int dk  I_n(|k| rho0) / (4 pi^2 sigma |k| r I_n'(|k| r))
                                         * S(k) e^{ikz} e^{in theta}

with S(k) the z-Fourier transform of the line current density. It is written
here independently of MyoGen's layered-conductor code.
"""
import math

import numpy as np
import pytest
import quantities as pq
from scipy.special import ive

from myogen.simulator.core.emg.electrodes import SurfaceElectrodeArray
from myogen.simulator.core.emg.runtime.surface_preparation import SurfacePreparation
from myogen.simulator.core.emg.surface import simulate_fiber

SIGMA = 0.3       # S/m, every layer
SOURCE_SCALE = 96 / 10  # mV/mm^3, MyoGen's Rosenfalck dVm/dz amplitude
DU = 0.02         # mm


def exact_surface_profile(i_line, du, rho0, r, sigma, n_max=200):
    """Surface potential along z directly above a line source sampled at spacing du."""
    n = len(i_line)
    k = 2 * math.pi * np.fft.fftfreq(n, d=du)
    S = np.fft.fft(i_line) * du
    x0, x1 = np.abs(k[1:]) * rho0, np.abs(k[1:]) * r
    total = np.zeros(n - 1)
    for order in range(-n_max, n_max + 1):
        m = abs(order)
        dI = 0.5 * (ive(m - 1, x1) + ive(m + 1, x1))  # I_m'(x) e^{-x}
        with np.errstate(divide="ignore", invalid="ignore"):
            term = ive(m, x0) / dI * np.exp(x0 - x1)
        total += np.where(np.isfinite(term), term, 0.0)  # high orders underflow; they vanish
    g = np.concatenate([[0.0], total / (4 * math.pi**2 * sigma * x1)])
    return np.real(np.fft.ifft(S * g)) * n * (2 * math.pi / (n * du))


def rosenfalck_line_source(u):
    """i(z) = d2Vm/dz2 * SOURCE_SCALE / A for a front at u = 0 travelling towards +z."""
    s = -u
    dvm = np.where(s >= 0, SOURCE_SCALE * np.exp(-s) * (3 * s**2 - s**3), 0.0)
    return -np.gradient(dvm, u[1] - u[0])


def _myogen_passing_wave(v, r, depth, prepared, z0=30.0, n=1024, kernel=300.0):
    array = SurfaceElectrodeArray(1, 1, 5 * pq.mm, 0.01 * pq.mm, center_point__mm_deg=(z0 * pq.mm, 0 * pq.deg),
                                  bending_radius__mm=r * pq.mm, differentiation_mode="monopolar")
    phi, _, _ = simulate_fiber._simulate_fiber_v2_python(
        Fs=10.0, v=v, N=n, M=64, r=r, r_bone=0.0, th_fat=0.05, th_skin=0.05, R=r - depth, L1=140.0, L2=140.0,
        zi=0.0, electrode_array=array, sig_muscle_rho=SIGMA, sig_muscle_z=SIGMA, sig_fat=SIGMA, sig_skin=SIGMA,
        fiber_length__mm=kernel, use_gpu=False, theta_offset=0.0,
        _prepared=SurfacePreparation() if prepared else None)
    t = (np.arange(n) - n // 2) * kernel / (v * (n - 1))  # ms from the firing instant
    window = (t > (z0 - 12) / v) & (t < (z0 + 12) / v)  # the wave passing the electrode only
    return phi[0, 0][window]


@pytest.mark.parametrize("prepared", [False, True], ids=["default", "prepared"])
@pytest.mark.parametrize("v", [2.0, 4.0])
@pytest.mark.parametrize("r, depth", [(10.0, 2.0), (20.0, 3.0)])
def test_single_fibre_amplitude_matches_the_exact_cylinder_solution(r, depth, v, prepared):
    u = np.arange(-60, 60, DU)
    exact = exact_surface_profile(rosenfalck_line_source(u), DU, r - depth, r, SIGMA)
    simulated = _myogen_passing_wave(v, r, depth, prepared)
    # Amplitude must not depend on conduction velocity: the spatial wave is the same.
    assert np.ptp(simulated) == pytest.approx(np.ptp(exact), rel=0.02)
