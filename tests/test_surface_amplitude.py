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
SIGMA_I = 1.01    # S/m, intracellular conductivity (default)
DIAMETER = 0.05   # mm, fibre diameter (default)
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


def rosenfalck_line_source(u, sigma_i=SIGMA_I, diameter=DIAMETER):
    """Line current density sigma_i * pi a^2 * d2Vm/dz2 of a front at u = 0 travelling towards +z.

    Vm = 96 z^3 e^{-z} - 90 mV (Rosenfalck 1969), z in mm behind the front.
    """
    s = -u
    dvm = np.where(s >= 0, 96 * np.exp(-s) * (3 * s**2 - s**3), 0.0)
    return sigma_i * math.pi * (diameter / 2) ** 2 * -np.gradient(dvm, u[1] - u[0])


def _myogen_passing_wave(v, r, depth, prepared, z0=30.0, n=1024, kernel=300.0, th_fat=0.05, th_skin=0.05,
                         r_bone=0.0, **fibre):
    array = SurfaceElectrodeArray(1, 1, 5 * pq.mm, 0.01 * pq.mm, center_point__mm_deg=(z0 * pq.mm, 0 * pq.deg),
                                  bending_radius__mm=r * pq.mm, differentiation_mode="monopolar")
    phi, _, _ = simulate_fiber._simulate_fiber_v2_python(
        Fs=10.0, v=v, N=n, M=64, r=r, r_bone=r_bone, th_fat=th_fat, th_skin=th_skin, R=r - depth, L1=140.0,
        L2=140.0, zi=0.0, electrode_array=array, sig_muscle_rho=SIGMA, sig_muscle_z=SIGMA, sig_fat=SIGMA,
        sig_skin=SIGMA, sig_bone=SIGMA,
        fiber_length__mm=kernel, use_gpu=False, theta_offset=0.0,
        _prepared=SurfacePreparation() if prepared else None, **fibre)
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


@pytest.mark.parametrize("prepared", [False, True], ids=["default", "prepared"])
@pytest.mark.parametrize("layers", [dict(th_fat=1.5, th_skin=1.0), dict(r_bone=1.0),
                                    dict(th_fat=1.5, th_skin=1.0, r_bone=1.0)],
                         ids=["thick-fat-skin", "bone", "both"])
def test_interfaces_between_equal_conductivities_are_transparent(layers, prepared):
    # Production geometries use real layer thicknesses and a bone; with every
    # conductivity equal they must reduce to the same exact homogeneous solution.
    r, depth = 12.0, 5.0
    u = np.arange(-60, 60, DU)
    exact = exact_surface_profile(rosenfalck_line_source(u), DU, r - depth, r, SIGMA)
    simulated = _myogen_passing_wave(4.0, r, depth, prepared, **layers)
    assert np.ptp(simulated) == pytest.approx(np.ptp(exact), rel=0.02)


@pytest.mark.parametrize("fibre, scale", [(dict(fiber_diameter__mm=0.08), (0.08 / DIAMETER) ** 2),
                                          (dict(sig_intracellular=0.5), 0.5 / SIGMA_I)])
def test_source_scales_with_fibre_cross_section_and_intracellular_conductivity(fibre, scale):
    reference = np.ptp(_myogen_passing_wave(4.0, 10.0, 2.0, False))
    assert np.ptp(_myogen_passing_wave(4.0, 10.0, 2.0, False, **fibre)) == pytest.approx(scale * reference, rel=1e-6)


@pytest.mark.parametrize("route", ["default", "execution"])
@pytest.mark.usefixtures("ample_session_memory")
def test_surface_emg_passes_fibre_properties_to_every_fibre(route, monkeypatch):
    from myogen import set_random_seed
    from myogen.simulator import ExecutionPolicy, Muscle, RecruitmentThresholds, SurfaceEMG
    from myogen.simulator.core.emg.runtime import semg as runtime_semg
    from myogen.simulator.core.emg.surface import surface_emg

    set_random_seed(42)
    thresholds, _ = RecruitmentThresholds(N=10, recruitment_range__ratio=20, mode="konstantin")
    muscle = Muscle(thresholds, fiber_density__fibers_per_mm2=5 * pq.mm**-2)
    muscle.distribute_innervation_centers()
    muscle.generate_muscle_fiber_centers(verbose=False)
    muscle.assign_mfs2mns(verbose=False, n_jobs=1)
    calls = []

    def spy(**kwargs):
        calls.append((kwargs["fiber_diameter__mm"], kwargs["sig_intracellular"]))
        array = kwargs["electrode_array"]
        return np.zeros((array.num_rows, array.num_cols, kwargs["N"])), None, None

    monkeypatch.setattr(surface_emg, "_simulate_fiber_v2_python", spy)
    monkeypatch.setattr(runtime_semg, "_simulate_fiber_v2_python", spy)
    array = SurfaceElectrodeArray(1, 1, 5 * pq.mm, 0.5 * pq.mm, center_point__mm_deg=(8 * pq.mm, 0 * pq.deg))
    semg = SurfaceEMG(muscle, [array], MUs_to_simulate=[1, 2], fiber_diameter__um=60.0,
                      intracellular_conductivity__S_m=0.9)
    if route == "default":
        semg.simulate_muaps(n_jobs=1, verbose=False, use_gpu=False)
    else:
        semg.simulate_muaps(n_jobs=1, verbose=False, execution=ExecutionPolicy())
    assert calls and all(d == pytest.approx(0.06) and s == pytest.approx(0.9) for d, s in calls)


def test_default_skin_conductivity_is_farina_value():
    # 1 S/m as in Farina et al. (2004), from Roeleveld et al. (1997): an effective, fitted value.
    # The previous default (4.88e-4 S/m, measured skin) raised surface amplitudes 3.5-6x.
    from myogen.simulator import Muscle, RecruitmentThresholds

    thresholds, _ = RecruitmentThresholds(N=5, recruitment_range__ratio=10, mode="konstantin")
    muscle = Muscle(thresholds)
    assert float(muscle.skin_conductivity__S_m.rescale(pq.S / pq.m).magnitude) == pytest.approx(1.0)


def test_default_layers_are_fdi_literature_values():
    # Skin 1.2 mm over the hand dorsum (Li et al. 2018, controls); fat 1.3 mm makes the total depth
    # to muscle 2.5 mm, the low end of the FDI subcutaneous depth of Nishikawa et al. (2024).
    from myogen.simulator import Muscle, RecruitmentThresholds

    thresholds, _ = RecruitmentThresholds(N=5, recruitment_range__ratio=10, mode="konstantin")
    muscle = Muscle(thresholds)
    assert float(muscle.skin_thickness__mm.rescale(pq.mm).magnitude) == pytest.approx(1.2)
    assert float(muscle.fat_thickness__mm.rescale(pq.mm).magnitude) == pytest.approx(1.3)


@pytest.mark.parametrize("route", ["default", "execution"])
@pytest.mark.usefixtures("ample_session_memory")
def test_a_failing_motor_unit_raises_instead_of_returning_zeros(route, monkeypatch):
    # A per-unit failure (e.g. a singular conductor system at high angular order) used to be logged and
    # replaced by an all-zero MUAP, which looks like a valid, silent unit.
    from myogen import set_random_seed
    from myogen.simulator import ExecutionPolicy, Muscle, RecruitmentThresholds, SurfaceEMG
    from myogen.simulator.core.emg.runtime import semg as runtime_semg
    from myogen.simulator.core.emg.surface import surface_emg

    set_random_seed(42)
    thresholds, _ = RecruitmentThresholds(N=10, recruitment_range__ratio=20, mode="konstantin")
    muscle = Muscle(thresholds, fiber_density__fibers_per_mm2=5 * pq.mm**-2)
    muscle.distribute_innervation_centers()
    muscle.generate_muscle_fiber_centers(verbose=False)
    muscle.assign_mfs2mns(verbose=False, n_jobs=1)

    def broken(**kwargs):
        raise np.linalg.LinAlgError("Singular matrix")

    monkeypatch.setattr(surface_emg, "_simulate_fiber_v2_python", broken)
    monkeypatch.setattr(runtime_semg, "_simulate_fiber_v2_python", broken)
    array = SurfaceElectrodeArray(1, 1, 5 * pq.mm, 0.5 * pq.mm, center_point__mm_deg=(8 * pq.mm, 0 * pq.deg))
    semg = SurfaceEMG(muscle, [array], MUs_to_simulate=[1, 2])
    with pytest.raises(np.linalg.LinAlgError):
        if route == "default":
            semg.simulate_muaps(n_jobs=1, verbose=False, use_gpu=False)
        else:
            semg.simulate_muaps(n_jobs=1, verbose=False, execution=ExecutionPolicy())


def _bessel_ratios():
    from scipy.special import kve

    def ir(n, a, b):  # I_n(a) / I_n(b)
        return ive(n, a) / ive(n, b) * np.exp(a - b)

    def ipr(n, a, b):  # I_n'(a) / I_n(b)
        return 0.5 * (ive(n - 1, a) + ive(n + 1, a)) / ive(n, b) * np.exp(a - b)

    def kr(n, a, b):  # K_n(a) / K_n(b)
        return kve(n, a) / kve(n, b) * np.exp(b - a)

    def kpr(n, a, b):  # K_n'(a) / K_n(b)
        return -0.5 * (kve(n - 1, a) + kve(n + 1, a)) / kve(n, b) * np.exp(b - a)

    return ir, ipr, kr, kpr, kve


def layered_surface_profile(i_line, du, rho0, rm, rf, rs, sr, sz, sf, ss, n_max=120, k_max=40.0):
    """Surface potential above a line source in an insulated cylinder: anisotropic muscle (sr, sz) to rm,
    isotropic fat (sf) to rf and skin (ss) to rs. Per angular order and k_z, the muscle field is the free-space
    term I_n(q rho<) K_n(q rho>) / sr (q = |k| sqrt(sz / sr)) plus A I_n(q rho); each outer layer is
    B I_n(|k| rho) + C K_n(|k| rho); potential and normal current are continuous and the surface is insulated."""
    ir, ipr, kr, kpr, kve = _bessel_ratios()
    n = len(i_line)
    k = 2 * math.pi * np.fft.fftfreq(n, d=du)
    S = np.fft.fft(i_line) * du
    sel = (np.abs(k) > 0) & (np.abs(k) < k_max)
    kk = np.abs(k[sel])
    q = math.sqrt(sz / sr) * kk
    total = np.zeros(kk.size)
    for order in range(-n_max, n_max + 1):
        m = abs(order)
        with np.errstate(all="ignore"):
            src = ive(m, q * rho0) * np.exp(q * rho0 - q * rm) / sr
            A = np.zeros((kk.size, 5, 5))
            b = np.zeros((kk.size, 5))
            A[:, 0, 0], A[:, 0, 1], A[:, 0, 2] = 1, -ir(m, kk * rm, kk * rf), -1
            b[:, 0] = -src * kve(m, q * rm)
            A[:, 1, 0] = sr * q * ipr(m, q * rm, q * rm)
            A[:, 1, 1], A[:, 1, 2] = -sf * kk * ipr(m, kk * rm, kk * rf), -sf * kk * kpr(m, kk * rm, kk * rm)
            b[:, 1] = sr * q * src * 0.5 * (kve(m - 1, q * rm) + kve(m + 1, q * rm))
            A[:, 2, 1], A[:, 2, 2] = 1, kr(m, kk * rf, kk * rm)
            A[:, 2, 3], A[:, 2, 4] = -ir(m, kk * rf, kk * rs), -1
            A[:, 3, 1], A[:, 3, 2] = sf * kk * ipr(m, kk * rf, kk * rf), sf * kk * kpr(m, kk * rf, kk * rm)
            A[:, 3, 3], A[:, 3, 4] = -ss * kk * ipr(m, kk * rf, kk * rs), -ss * kk * kpr(m, kk * rf, kk * rf)
            A[:, 4, 3], A[:, 4, 4] = ipr(m, kk * rs, kk * rs), kpr(m, kk * rs, kk * rf)
            ok = np.all(np.isfinite(A), axis=(1, 2)) & np.all(np.isfinite(b), axis=1)
            x = np.zeros((kk.size, 5))
            if ok.any():
                x[ok] = np.linalg.solve(A[ok], b[ok][..., None])[..., 0]
            phi = x[:, 3] + x[:, 4] * kr(m, kk * rs, kk * rf)
        total += np.where(np.isfinite(phi), phi, 0.0)
    g = np.zeros(n)
    g[sel] = total / (4 * math.pi**2)
    return np.real(np.fft.ifft(S * g)) * n * (2 * math.pi / (n * du))


def _myogen_layered(v, rm, depth, sr, sz, fat, skin, sf, ss, z0=30.0, n=1024, kernel=300.0, L=140.0,
                    mode="monopolar", ied=5.0, window=True):
    # The volume conductor's r is the outer (skin) radius; R is the fibre's radial position.
    array = SurfaceElectrodeArray(1, 1, ied * pq.mm, 0.01 * pq.mm, center_point__mm_deg=(z0 * pq.mm, 0 * pq.deg),
                                  bending_radius__mm=(rm + fat + skin) * pq.mm, differentiation_mode=mode)
    phi, _, _ = simulate_fiber._simulate_fiber_v2_python(
        Fs=10.0, v=v, N=n, M=64, r=rm + fat + skin, r_bone=0.0, th_fat=fat, th_skin=skin, R=rm - depth, L1=L, L2=L,
        zi=0.0, electrode_array=array, sig_muscle_rho=sr, sig_muscle_z=sz, sig_fat=sf, sig_skin=ss, sig_bone=sr,
        fiber_length__mm=kernel, use_gpu=False, theta_offset=0.0)
    if not window:
        return phi[0, 0]
    t = (np.arange(n) - n // 2) * kernel / (v * (n - 1))
    return phi[0, 0][(t > (z0 - 12) / v) & (t < (z0 + 12) / v)]


def test_layered_solver_reduces_to_the_homogeneous_solution():
    u = np.arange(-60, 60, DU)
    src = rosenfalck_line_source(u)
    layered = layered_surface_profile(src, DU, 8.0, 9.0, 9.5, 10.0, SIGMA, SIGMA, SIGMA, SIGMA)
    assert np.ptp(layered) == pytest.approx(np.ptp(exact_surface_profile(src, DU, 8.0, 10.0, SIGMA)), rel=1e-3)


@pytest.mark.parametrize("skin_sigma", [1.0, 0.15])
@pytest.mark.parametrize("depth", [0.5, 2.0])
def test_anisotropic_muscle_under_fat_and_skin_matches_independent_solution(depth, skin_sigma):
    # FDI-like layers: muscle 0.1 / 0.5 S/m (radial / axial), fat 0.05 S/m 1.3 mm, skin 1.2 mm.
    rm = 6.91
    u = np.arange(-60, 60, DU)
    ref = layered_surface_profile(rosenfalck_line_source(u), DU, rm - depth, rm, rm + 1.3, rm + 2.5,
                                  0.1, 0.5, 0.05, skin_sigma)
    simulated = _myogen_layered(4.0, rm, depth, 0.1, 0.5, 1.3, 1.2, 0.05, skin_sigma)
    assert np.ptp(simulated) == pytest.approx(np.ptp(ref), rel=0.01)


@pytest.mark.parametrize("ied", [5.0, 10.0])
def test_finite_fibre_bipolar_matches_sealed_end_cable_source(ied):
    # End-plate generation and tendon extinction: the transmembrane current of a sealed-end fibre
    # (axial current -sigma_i pi a^2 dVm/dz, zero beyond the ends), convolved with the layered point-source
    # potential, against MyoGen's finite-fibre source. Bipolar, so the far-field common mode cancels.
    rm, fat, skin, depth, half, v, ze = 6.91, 1.3, 1.2, 2.0, 15.0, 4.0, 8.0
    u = np.arange(-80, 80, DU)
    delta = np.zeros_like(u)
    delta[np.argmin(np.abs(u))] = 1.0 / DU
    green = layered_surface_profile(delta, DU, rm - depth, rm, rm + fat, rm + fat + skin, 0.1, 0.5, 0.05, 1.0,
                                    n_max=150)
    zf = np.arange(-half, half + DU / 2, DU)
    channels = []
    for z in (ze + ied / 2, ze - ied / 2):
        g = np.interp(z - zf, u, green)
        phi = []
        for t in np.arange(0, 14, 0.02):
            s = v * t - np.abs(zf)
            vm = np.where(s >= 0, 96 * s**3 * np.exp(-np.clip(s, 0, None)), 0.0)
            axial = np.concatenate([[0.0], -SIGMA_I * math.pi * (DIAMETER / 2) ** 2 * np.diff(vm) / DU, [0.0]])
            phi.append(np.sum(-np.diff(axial) * g))
        channels.append(np.array(phi))
    reference = channels[0] - channels[1]
    simulated = _myogen_layered(v, rm, depth, 0.1, 0.5, fat, skin, 0.05, 1.0, z0=ze, n=4096, L=half,
                                mode="bipolar_longitudinal", ied=ied, window=False)
    assert np.ptp(simulated) == pytest.approx(np.ptp(reference), rel=0.01)
