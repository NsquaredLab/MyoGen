"""Per-fibre dispersion in the surface-EMG model.

Axon-branch delays and end-plate positions come from the two-layer nerve model
that basic MyoGen already uses for intramuscular EMG (Akhmadeev et al. 2020);
muscle-fibre conduction velocity may vary per fibre. Every option defaults to
off, and off must reproduce the undispersed model exactly.
"""
import dataclasses
import numpy as np
import pytest
import quantities as pq
from scipy.signal import resample

from myogen import derive_subseed, get_random_generator, set_random_seed
from myogen.simulator import (
    ExecutionPolicy,
    Muscle,
    RecruitmentThresholds,
    SurfaceElectrodeArray,
    SurfaceEMG,
)
from myogen.simulator.core.emg.intramuscular.motor_unit_sim import MotorUnitSim
from myogen.simulator.core.emg.nerve_branches import (
    nerve_delays__ms,
    nmj_branch_schedule,
    two_layer_nmj,
)
from myogen.simulator.core.emg.surface import simulate_fiber
from myogen.simulator.core.emg.surface.fiber_dispersion import (
    SurfaceFiberDispersion,
    realize_fiber_dispersion,
)
from myogen.utils.neo import signal_to_grid

GEOMETRY = dict(r=8.5, r_bone=0.0, th_fat=0.3, th_skin=1.29, sig_muscle_rho=0.09,
                sig_muscle_z=0.4, sig_fat=4.07e-2, sig_skin=1.0)
N = 512
KERNEL = 100.0  # mm


def _array(z0=10.0, mode="monopolar"):
    return SurfaceElectrodeArray(1, 1, 10 * pq.mm, 0.5 * pq.mm,
                                 center_point__mm_deg=(z0 * pq.mm, 0 * pq.deg),
                                 bending_radius__mm=GEOMETRY["r"] * pq.mm,
                                 differentiation_mode=mode)


def _fibre(v, v_grid=None, n=N, zi=0.0):
    kwargs = {} if v_grid is None else {"v_grid": v_grid}
    phi, _, _ = simulate_fiber._simulate_fiber_v2_python(
        Fs=10.0, v=v, N=n, M=32, R=6.5, L1=20.0 - zi, L2=20.0 + zi, zi=zi,
        electrode_array=_array(), fiber_length__mm=KERNEL, use_gpu=False, **kwargs, **GEOMETRY)
    return phi[0, 0]


# ---- two-layer nerve model: one authority shared with intramuscular EMG ----------

def test_two_layer_function_reproduces_motor_unit_sim():
    set_random_seed(5)
    xy = get_random_generator().uniform(-3, 3, size=(40, 2))
    set_random_seed(11)
    sim = MotorUnitSim(muscle_fiber_centers__mm=xy, muscle_length__mm=80.0,
                       muscle_fiber_diameters__mm=np.full(40, 0.05),
                       muscle_fiber_conduction_velocity__mm_per_s=np.full(40, 4000.0),
                       nominal_center__mm=np.zeros(2))
    sim.sim_nmj_branches_two_layers(3, 40.0, 2.0, 0.8)
    set_random_seed(11)
    nmj_z, paths = two_layer_nmj(xy, 3, 40.0, 2.0, 0.8, rng=get_random_generator(),
                                 kmeans_seed=derive_subseed(3))
    np.testing.assert_allclose(np.ravel(sim._neuromuscular_z_coordinates__mm), nmj_z, rtol=0, atol=1e-12)
    np.testing.assert_allclose(sim.nerve_paths, paths, rtol=0, atol=1e-12)


def test_branch_schedule_grows_with_motor_unit_size():
    thresholds = np.array([1.0, np.e, np.e**2, np.e**3])
    # Default: published coefficients (Akhmadeev et al. 2020, eq. 5-6): a_mu=1, b_mu=2.5, a_sigma=0.25, b_sigma=1 mm
    n_branches, branch_sd, arbor_sd = nmj_branch_schedule(thresholds)
    np.testing.assert_array_equal(n_branches, [1, 2, 3, 4])
    spread = np.concatenate([[0], np.cumsum(thresholds)[:-1]]) / thresholds.sum()
    np.testing.assert_allclose(branch_sd, 1.0 + 2.5 * spread)
    np.testing.assert_allclose(arbor_sd, 0.25 + 1.0 * spread)
    # Akhmadeev's reference simulator code
    _, branch_sd, arbor_sd = nmj_branch_schedule(thresholds, coefficients="reference_code")
    np.testing.assert_allclose(branch_sd, 1.5 + 4.0 * spread)
    np.testing.assert_allclose(arbor_sd, 0.5 + 1.5 * spread)
    with pytest.raises(ValueError):
        nmj_branch_schedule(thresholds, coefficients="other")


def test_surface_dispersion_can_use_the_published_coefficients(muscle):
    zones = np.zeros(12)
    script = _realize(muscle, SurfaceFiberDispersion(nmj_model="two_layer_intra", nmj_coefficients="reference_code"),
                      zones)
    paper = _realize(muscle, SurfaceFiberDispersion(nmj_model="two_layer_intra"), zones)
    big = max(range(12), key=lambda i: len(paper.zi__mm[i]))
    assert np.std(paper.zi__mm[big]) < np.std(script.zi__mm[big])
    with pytest.raises(ValueError):
        SurfaceFiberDispersion(nmj_coefficients="other")


def test_nerve_delay_is_path_over_branch_velocity():
    paths = np.array([[5.0, 2.0], [0.0, 4.0]])  # mm
    np.testing.assert_allclose(nerve_delays__ms(paths, (5.0, 2.0)), [2.0, 2.0])


# ---- fibre velocity is decoupled from the time grid -------------------------------

def test_reference_grid_velocity_defaults_to_the_fibre_velocity():
    np.testing.assert_array_equal(_fibre(4.2, v_grid=4.2), _fibre(4.2))


@pytest.mark.parametrize("v", [3.4, 5.0])
def test_fibre_on_a_shared_grid_matches_its_own_grid(v):
    # Native run: dt = KERNEL / (v (N-1)). Shared run: dt = KERNEL / (4.2 (N-1)).
    n = 1024
    native = _fibre(v, n=n)
    shared = _fibre(v, v_grid=4.2, n=n)
    t_native = (np.arange(n) - n // 2) * KERNEL / (v * (n - 1))
    t_shared = (np.arange(n) - n // 2) * KERNEL / (4.2 * (n - 1))
    up = 16
    fine = resample(native, n * up)
    t_fine = (np.arange(n * up) - n // 2 * up) * KERNEL / (v * (n - 1)) / up
    expected = np.interp(t_shared, t_fine, fine, left=0.0, right=0.0)
    inside = np.abs(t_shared) < 0.45 * min(t_native.max(), t_shared.max())
    err = np.max(np.abs(shared[inside] - expected[inside])) / np.ptp(expected)
    assert err < 0.01


# ---- fractional delay ---------------------------------------------------------------

def test_delay_moves_the_signal_by_the_requested_time():
    fs = 20.0  # kHz
    t = np.arange(2048) / fs
    x = np.exp(-((t - 40.0) / 0.8) ** 2)[None, None, :]
    y = simulate_fiber.delay_fibre_signal(x, 1.37, fs)
    up = resample(y[0, 0], 2048 * 50)
    assert np.argmax(up) / 50 / fs == pytest.approx(41.37, abs=0.01)
    np.testing.assert_array_equal(simulate_fiber.delay_fibre_signal(x, 0.0, fs), x)


# ---- realisation --------------------------------------------------------------------

def _muscle(**kw):
    set_random_seed(42)
    thresholds, _ = RecruitmentThresholds(N=12, recruitment_range__ratio=30, mode="fuglevand")
    muscle = Muscle(thresholds, radius__mm=6 * pq.mm, length__mm=60 * pq.mm,
                    fiber_density__fibers_per_mm2=6 * pq.mm**-2, mean_fiber_length__mm=50 * pq.mm,
                    var_fiber_length__mm=2 * pq.mm, radius_bone__mm=0 * pq.mm,
                    fat_thickness__mm=1 * pq.mm, **kw)
    muscle.distribute_innervation_centers()
    muscle.generate_muscle_fiber_centers(verbose=False)
    muscle.assign_mfs2mns(verbose=False, n_jobs=1)
    return muscle


@pytest.fixture(scope="module")
def muscle():
    return _muscle()


def _realize(muscle, config, zones=None, **kwargs):
    counts = np.asarray(muscle.resulting_number_of_innervated_fibers)
    zones = np.zeros(len(counts)) if zones is None else zones
    positions = [np.asarray(muscle.resulting_fiber_assignment(i)) for i in range(len(counts))]
    return realize_fiber_dispersion(config, positions, np.asarray(muscle.recruitment_thresholds),
                                    zones, mean_cv__m_s=4.2, **kwargs)


def test_default_config_is_the_undispersed_model(muscle):
    zones = np.linspace(-2, 2, 12)
    real = _realize(muscle, SurfaceFiberDispersion(), zones)
    for mu, z in enumerate(zones):
        n = len(real.cv__m_s[mu])
        np.testing.assert_array_equal(real.zi__mm[mu], np.full(n, z))
        np.testing.assert_array_equal(real.cv__m_s[mu], np.full(n, 4.2))
        np.testing.assert_array_equal(real.delay__ms[mu], np.zeros(n))
        np.testing.assert_array_equal(real.amplitude_scale[mu], np.ones(n))
    assert not SurfaceFiberDispersion().active


def test_fibre_cv_spreads_follow_the_configuration(muscle):
    real = _realize(muscle, SurfaceFiberDispersion(fiber_cv_inter_mu_spread=0.3,
                                                   fiber_cv_within_mu_cov=0.08))
    means = np.array([np.mean(c) for c in real.cv__m_s if len(c)])
    assert means[0] == pytest.approx(4.2 * 0.85, rel=0.05)
    assert means[-1] == pytest.approx(4.2 * 1.15, rel=0.05)
    big = max(range(12), key=lambda i: len(real.cv__m_s[i]))
    c = real.cv__m_s[big]
    assert np.std(c) / np.mean(c) == pytest.approx(0.08, rel=0.25)
    assert np.all(real.amplitude_scale[big] == 1.0)  # amplitude coupling is a separate switch


def test_fibre_cv_rises_linearly_with_recruitment_threshold(muscle):
    real = _realize(muscle, SurfaceFiberDispersion(fiber_cv_threshold_slope=0.63))
    rt = np.asarray(muscle.recruitment_thresholds, dtype=float)
    rt = rt / rt.max()
    for mu, cv in enumerate(real.cv__m_s):
        np.testing.assert_allclose(cv, 4.2 * (1 + 0.63 * (rt[mu] - rt.mean())))
    with pytest.raises(ValueError):
        SurfaceFiberDispersion(fiber_cv_threshold_slope=0.6, fiber_cv_inter_mu_spread=0.3)


def test_fibre_cv_per_unit_sets_each_unit_mean_and_keeps_end_plates(muscle):
    per_unit = tuple(3.5 + 0.1 * i for i in range(12))
    real = _realize(muscle, SurfaceFiberDispersion(nmj_model="two_layer", fiber_cv_per_unit__m_s=per_unit))
    ref = _realize(muscle, SurfaceFiberDispersion(nmj_model="two_layer"))
    for mu, cv in enumerate(real.cv__m_s):
        np.testing.assert_allclose(cv, per_unit[mu])
        np.testing.assert_array_equal(real.zi__mm[mu], ref.zi__mm[mu])
    with pytest.raises(ValueError):
        SurfaceFiberDispersion(fiber_cv_per_unit__m_s=per_unit, fiber_cv_threshold_slope=0.63)
    with pytest.raises(ValueError):
        SurfaceFiberDispersion(fiber_cv_per_unit__m_s=(4.0, -1.0))
    with pytest.raises(ValueError):
        _realize(muscle, SurfaceFiberDispersion(fiber_cv_per_unit__m_s=per_unit[:5]))


def test_diameter_coupling_scales_source_with_cross_section(muscle):
    cfg = SurfaceFiberDispersion(fiber_cv_within_mu_cov=0.08, diameter_scales_source=True)
    real = _realize(muscle, cfg)
    big = max(range(12), key=lambda i: len(real.cv__m_s[i]))
    # Diameter from the CV line v = 0.95 m/s + 0.05 m/s/um * d (Nandedkar & Stalberg 1983)
    diameter__um = (real.cv__m_s[big] - 0.95) / 0.05
    ratio = real.amplitude_scale[big] / real.amplitude_scale[big][0]
    np.testing.assert_allclose(ratio, (diameter__um / diameter__um[0]) ** 2)
    assert real.report["line_diameter_mean__um"][big] == pytest.approx(np.mean(diameter__um))


def test_report_gives_the_effective_fibre_diameter(muscle):
    """``diameter_mean__um`` is the diameter each fibre's source uses: the line diameter rescaled so
    the muscle's root-mean-square equals ``fiber_diameter__um`` (the line diameter is reported apart)."""
    cfg = SurfaceFiberDispersion(fiber_cv_within_mu_cov=0.08, diameter_scales_source=True)
    real = _realize(muscle, cfg, fiber_diameter__um=57.9)
    effective = [57.9 * np.sqrt(scale) for scale in real.amplitude_scale]
    for mu, d in enumerate(effective):
        assert real.report["diameter_mean__um"][mu] == pytest.approx(np.mean(d))
    assert np.sqrt(np.mean(np.concatenate(effective) ** 2)) == pytest.approx(57.9)
    off = _realize(muscle, SurfaceFiberDispersion(fiber_cv_within_mu_cov=0.08), fiber_diameter__um=57.9)
    np.testing.assert_allclose(off.report["diameter_mean__um"], 57.9)
    assert np.all(np.isnan(_realize(muscle, cfg).report["diameter_mean__um"]))


@pytest.mark.parametrize("cfg", [
    dict(fiber_cv_within_mu_cov=0.08),
    dict(fiber_cv_threshold_slope=0.63),
    dict(fiber_cv_per_unit__m_s=tuple(3.5 + 0.1 * i for i in range(12))),
])
def test_diameter_scaled_source_averages_one_over_the_muscle(muscle, cfg):
    """The source factor is normalised over every fibre of the pool, so ``fiber_diameter__um``
    stays the population root-mean-square diameter and the CV line only sets the spread."""
    real = _realize(muscle, SurfaceFiberDispersion(diameter_scales_source=True, **cfg))
    assert np.mean(np.concatenate(real.amplitude_scale)) == pytest.approx(1.0)
    diameter__um = (np.concatenate(real.cv__m_s) - 0.95) / 0.05
    np.testing.assert_allclose(np.concatenate(real.amplitude_scale),
                               diameter__um**2 / np.mean(diameter__um**2))


def test_intra_mu_scatter_keeps_each_motor_unit_anchor(muscle):
    zones = np.linspace(-2, 2, 12)
    real = _realize(muscle, SurfaceFiberDispersion(nmj_model="two_layer_intra"), zones)
    for mu in range(12):
        if len(real.zi__mm[mu]) > 1:
            assert np.mean(real.zi__mm[mu]) == pytest.approx(zones[mu], abs=1e-9)
            assert np.std(real.zi__mm[mu]) > 0.1


def test_nerve_delays_switch_independently_of_end_plates(muscle):
    real = _realize(muscle, SurfaceFiberDispersion(nerve_delays=True))
    big = max(range(12), key=lambda i: len(real.cv__m_s[i]))
    np.testing.assert_array_equal(real.zi__mm[big], np.zeros(len(real.zi__mm[big])))
    assert np.std(real.delay__ms[big]) > 0
    assert real.report["delay_sd__ms"][big] == pytest.approx(np.std(real.delay__ms[big]))


def _realize_with_lengths(muscle, config, mean_length=70.0, var=2.8):
    counts = np.asarray(muscle.resulting_number_of_innervated_fibers)
    positions = [np.asarray(muscle.resulting_fiber_assignment(i)) for i in range(len(counts))]
    lengths = [mean_length + np.random.default_rng(i).uniform(-var, var, size=n) for i, n in enumerate(counts)]
    real = realize_fiber_dispersion(config, positions, np.asarray(muscle.recruitment_thresholds),
                                    np.zeros(len(counts)), mean_cv__m_s=4.2, fiber_lengths__mm=lengths,
                                    mean_fiber_length__mm=mean_length)
    return real, lengths


def test_fibre_ends_follow_the_fibre_length_by_default(muscle):
    real, lengths = _realize_with_lengths(muscle, SurfaceFiberDispersion(nerve_delays=True))
    for mu, length in enumerate(lengths):
        np.testing.assert_array_equal(real.end_plus__mm[mu], length / 2)
        np.testing.assert_array_equal(real.end_minus__mm[mu], -length / 2)


def test_tendon_band_staggers_each_end_independently(muscle):
    real, _ = _realize_with_lengths(muscle, SurfaceFiberDispersion(tendon_end_band__mm=8.0))
    assert SurfaceFiberDispersion(tendon_end_band__mm=8.0).active
    plus = np.concatenate(real.end_plus__mm)
    minus = np.concatenate(real.end_minus__mm)
    assert np.all(np.abs(plus - 35.0) <= 4.0) and np.all(np.abs(minus + 35.0) <= 4.0)
    assert np.std(plus) == pytest.approx(8.0 / np.sqrt(12), rel=0.1)
    assert abs(np.corrcoef(plus, minus)[0, 1]) < 4 / np.sqrt(len(plus))  # independent within sampling error
    big = max(range(12), key=lambda i: len(real.end_plus__mm[i]))
    assert real.report["end_sd__mm"][big] == pytest.approx(np.std(real.end_plus__mm[big]))


def _end_variogram(muscle, real, near__mm, far__mm):
    xy = np.concatenate([np.asarray(muscle.resulting_fiber_assignment(i))[:, :2] for i in range(12)])
    plus = np.concatenate(real.end_plus__mm)
    i, j = np.triu_indices(len(xy), k=1)
    d = np.hypot(*(xy[i] - xy[j]).T)
    sq = (plus[i] - plus[j]) ** 2
    return np.mean(sq[d < near__mm]), np.mean(sq[d > far__mm])


def test_correlated_tendon_ends_vary_smoothly_between_neighbours(muscle):
    config = SurfaceFiberDispersion(tendon_end_band__mm=8.0, tendon_end_correlation__mm=2.0,
                                    tendon_end_jitter__mm=0.3)
    real, _ = _realize_with_lengths(muscle, config)
    near, far = _end_variogram(muscle, real, 0.5, 8.0)
    assert near < 0.2 * far
    independent, _ = _realize_with_lengths(muscle, SurfaceFiberDispersion(tendon_end_band__mm=8.0))
    near_independent, _ = _end_variogram(muscle, independent, 0.5, 8.0)
    assert near < 0.2 * near_independent
    sd = real.report["end_sd__mm"]
    assert np.nanmedian(sd) < np.nanmedian(independent.report["end_sd__mm"])


def test_tendon_end_correlation_needs_a_band_and_room_for_the_jitter():
    with pytest.raises(ValueError, match="tendon_end_band"):
        SurfaceFiberDispersion(tendon_end_correlation__mm=2.0)
    with pytest.raises(ValueError, match="jitter"):
        SurfaceFiberDispersion(tendon_end_band__mm=1.0, tendon_end_correlation__mm=2.0,
                               tendon_end_jitter__mm=0.5)
    with pytest.raises(ValueError, match="non-negative"):
        SurfaceFiberDispersion(tendon_end_band__mm=5.0, tendon_end_correlation__mm=-1.0)


def test_realisation_leaves_the_global_generator_untouched(muscle):
    set_random_seed(3)
    expected = get_random_generator().uniform(size=4)
    set_random_seed(3)
    _realize(muscle, SurfaceFiberDispersion(nmj_model="two_layer", nerve_delays=True,
                                            fiber_cv_within_mu_cov=0.05))
    np.testing.assert_array_equal(get_random_generator().uniform(size=4), expected)


# ---- SurfaceEMG integration ---------------------------------------------------------

def _semg(muscle, dispersion=None, mus=(4, 11), **kw):
    array = SurfaceElectrodeArray(1, 1, 10 * pq.mm, 2.5 * pq.mm,
                                  center_point__mm_deg=(12 * pq.mm, 0 * pq.deg),
                                  bending_radius__mm=8.3 * pq.mm,
                                  differentiation_mode="bipolar_longitudinal")
    set_random_seed(7)
    return SurfaceEMG(muscle, [array], sampling_frequency__Hz=2048 * pq.Hz,
                      internal_sampling_frequency__Hz=2048 * pq.Hz,
                      MUs_to_simulate=list(mus), fiber_dispersion=dispersion, **kw)


def _muaps(semg):
    block = semg.simulate_muaps(n_jobs=1, verbose=False, use_gpu=False)
    return np.array([np.asarray(signal_to_grid(s.analogsignals[0].rescale(pq.mV)))
                     for s in block.groups[0].segments])


def test_inactive_dispersion_is_bit_identical(muscle):
    np.testing.assert_array_equal(_muaps(_semg(muscle)),
                                  _muaps(_semg(muscle, SurfaceFiberDispersion())))


def test_dispersion_changes_the_muap_and_is_reported(muscle):
    base = _muaps(_semg(muscle))
    semg = _semg(muscle, SurfaceFiberDispersion(nerve_delays=True, fiber_cv_within_mu_cov=0.08))
    dispersed = _muaps(semg)
    assert not np.allclose(base[11], dispersed[11])
    report = semg.fiber_dispersion_report
    assert report["delay_sd__ms"][11] > 0 and report["cv_cov"][11] > 0


def test_tendon_band_changes_the_muap(muscle):
    base = _muaps(_semg(muscle))
    staggered = _muaps(_semg(muscle, SurfaceFiberDispersion(tendon_end_band__mm=10.0)))
    assert not np.allclose(base[11], staggered[11])


@pytest.mark.usefixtures("ample_session_memory")
def test_other_routes_refuse_active_dispersion(muscle):
    semg = _semg(muscle, SurfaceFiberDispersion(nerve_delays=True))
    with pytest.raises(ValueError, match="dispersion"):
        semg.simulate_muaps(n_jobs=1, verbose=False, execution=ExecutionPolicy())


def test_literature_candidate_preset():
    preset = SurfaceFiberDispersion.literature_candidate()
    assert preset.active
    assert (preset.nmj_model, preset.nmj_scatter_scale, preset.nmj_coefficients) == ("two_layer_intra", 1.0, "published")
    assert preset.nerve_delays and preset.nerve_cvs__m_per_s == (5.0, 2.0)
    assert (preset.fiber_cv_threshold_slope, preset.fiber_cv_within_mu_cov, preset.fiber_cv_inter_mu_spread) == (
        0.63, 0.05, 0.0)
    assert not preset.diameter_scales_source
    assert (preset.tendon_end_band__mm, preset.tendon_end_correlation__mm, preset.tendon_end_jitter__mm) == (
        5.0, 5.0, 0.3)
    assert SurfaceFiberDispersion.literature_candidate(seed=4).seed == 4


def test_intramuscular_emg_selects_the_end_plate_coefficients(monkeypatch):
    from myogen.simulator import IntramuscularElectrodeArray, IntramuscularEMG
    from myogen.simulator.core.emg.intramuscular.motor_unit_sim import MotorUnitSim

    set_random_seed(5)
    thresholds, _ = RecruitmentThresholds(N=4, recruitment_range__ratio=30)
    muscle = Muscle(recruitment_thresholds=thresholds, radius__mm=2.0 * pq.mm, radius_bone__mm=0.5 * pq.mm,
                    fiber_density__fibers_per_mm2=50 * pq.mm**-2, grid_resolution=32, autorun=True)
    electrode = IntramuscularElectrodeArray(num_electrodes=2, inter_electrode_distance__mm=0.5 * pq.mm,
                                            position__mm=(0.0 * pq.mm, 0.0 * pq.mm, 15.0 * pq.mm))
    seen = []
    original = MotorUnitSim.sim_nmj_branches_two_layers

    def capture(self, **kwargs):
        seen.append((kwargs["branches_z_std"], kwargs["arborization_z_std"]))
        return original(self, **kwargs)

    monkeypatch.setattr(MotorUnitSim, "sim_nmj_branches_two_layers", capture)
    for coefficients in ("published", "reference_code"):
        seen.clear()
        kwargs = {} if coefficients == "published" else {"nmj_coefficients": coefficients}
        iemg = IntramuscularEMG(muscle_model=muscle, electrode_array=electrode, **kwargs)
        iemg._initialize_motor_units(verbose=False)
        iemg._simulate_neuromuscular_junctions(verbose=False)
        _, branch_sd, arbor_sd = nmj_branch_schedule(np.asarray(muscle.recruitment_thresholds), coefficients)
        np.testing.assert_allclose([s[0] for s in seen], branch_sd)
        np.testing.assert_allclose([s[1] for s in seen], arbor_sd)
    with pytest.raises(ValueError, match="nmj_coefficients"):
        IntramuscularEMG(muscle_model=muscle, electrode_array=electrode, nmj_coefficients="paper")


# ---- fibre properties from Muscle ---------------------------------------------------

def test_muscle_fibre_source_is_active():
    assert SurfaceFiberDispersion(fiber_properties="muscle").active


@pytest.mark.parametrize("field, value", [("fiber_cv_threshold_slope", 0.63), ("fiber_cv_inter_mu_spread", 0.3),
                                          ("fiber_cv_within_mu_cov", 0.05), ("fiber_cv_per_unit__m_s", (4.0,))])
def test_muscle_fibre_source_rejects_surface_cv_options(field, value):
    with pytest.raises(ValueError, match=field):
        SurfaceFiberDispersion(fiber_properties="muscle", **{field: value})


def test_fibre_property_source_must_be_known():
    with pytest.raises(ValueError, match="fiber_properties"):
        SurfaceFiberDispersion(fiber_properties="x")


def test_muscle_candidate_preset():
    expected = dataclasses.replace(SurfaceFiberDispersion.literature_candidate(seed=3), fiber_properties="muscle",
                                   fiber_cv_threshold_slope=0.0, fiber_cv_within_mu_cov=0.0,
                                   diameter_scales_source=True)
    assert SurfaceFiberDispersion.muscle_candidate(seed=3) == expected


def _muscle_arrays(muscle):
    counts = np.asarray(muscle.resulting_number_of_innervated_fibers)
    return [np.linspace(3, 5, n) for n in counts], [np.linspace(40, 80, n) for n in counts]


def test_muscle_source_uses_the_given_fibre_cvs(muscle):
    cvs, ds = _muscle_arrays(muscle)
    real = _realize(muscle, SurfaceFiberDispersion(fiber_properties="muscle"), fiber_cv__m_s=cvs,
                    fiber_diameters__um=ds, fiber_diameter__um=57.9)
    for mu, cv in enumerate(cvs):
        np.testing.assert_array_equal(real.cv__m_s[mu], cv)
        if len(cv):
            assert real.report["cv_mean__m_s"][mu] == pytest.approx(cv.mean(), rel=1e-12)


def test_muscle_source_scales_by_absolute_cross_section(muscle):
    cvs, ds = _muscle_arrays(muscle)
    real = _realize(muscle, SurfaceFiberDispersion(fiber_properties="muscle", diameter_scales_source=True),
                    fiber_cv__m_s=cvs, fiber_diameters__um=ds, fiber_diameter__um=50.0)
    for mu, d in enumerate(ds):
        np.testing.assert_allclose(real.amplitude_scale[mu], (d / 50.0) ** 2, rtol=1e-12)
        if len(d):
            assert real.report["diameter_mean__um"][mu] == pytest.approx(d.mean(), rel=1e-12)
            assert np.isnan(real.report["line_diameter_mean__um"][mu])


def test_muscle_source_without_scaling_keeps_unit_scale(muscle):
    cvs, ds = _muscle_arrays(muscle)
    real = _realize(muscle, SurfaceFiberDispersion(fiber_properties="muscle"), fiber_cv__m_s=cvs,
                    fiber_diameters__um=ds, fiber_diameter__um=57.9)
    for scale in real.amplitude_scale:
        np.testing.assert_array_equal(scale, np.ones(len(scale)))


def test_muscle_source_keeps_the_other_draws(muscle):
    cvs, ds = _muscle_arrays(muscle)
    common = dict(nmj_model="two_layer_intra", nerve_delays=True, tendon_end_band__mm=5.0, seed=5)
    kw = dict(fiber_lengths__mm=[np.full(len(c), 50.0) for c in cvs], mean_fiber_length__mm=50.0)
    config = _realize(muscle, SurfaceFiberDispersion(**common), **kw)
    from_muscle = _realize(muscle, SurfaceFiberDispersion(fiber_properties="muscle", **common),
                           fiber_cv__m_s=cvs, fiber_diameters__um=ds, **kw)
    for a, b in ((config.zi__mm, from_muscle.zi__mm), (config.delay__ms, from_muscle.delay__ms),
                 (config.end_plus__mm, from_muscle.end_plus__mm)):
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)


def test_muscle_source_needs_the_arrays(muscle):
    cvs, ds = _muscle_arrays(muscle)
    with pytest.raises(ValueError, match="fiber_cv__m_s"):
        _realize(muscle, SurfaceFiberDispersion(fiber_properties="muscle"))
    with pytest.raises(ValueError, match="fiber_diameters__um"):
        _realize(muscle, SurfaceFiberDispersion(fiber_properties="muscle", diameter_scales_source=True),
                 fiber_cv__m_s=cvs, fiber_diameter__um=57.9)
    short = [cv[:-1] if len(cv) else cv for cv in cvs]
    with pytest.raises(ValueError, match="motor unit"):
        _realize(muscle, SurfaceFiberDispersion(fiber_properties="muscle"), fiber_cv__m_s=short)


@pytest.fixture(scope="module")
def typed():
    muscle = _muscle(fiber_model="typed")
    muscle._generate_fiber_properties()
    return muscle


def test_surface_emg_takes_independent_fibres_from_muscle_too():
    """``fiber_properties="muscle"`` needs only fibre diameters and CVs, which the default model also gives."""
    muscle = _muscle()
    muscle._generate_fiber_properties()
    semg = _semg(muscle, SurfaceFiberDispersion(fiber_properties="muscle", diameter_scales_source=True),
                 iap_kernel_length__mm=250.0)
    _muaps(semg)
    cv = np.asarray(muscle.muscle_fiber_conduction_velocities__mm_per_s.rescale(pq.m / pq.s).magnitude)
    for mu in (4, 11):
        mask = muscle.assignment == mu
        assert semg.fiber_dispersion_report["cv_mean__m_s"][mu] == pytest.approx(cv[mask].mean(), rel=1e-12)


def test_surface_emg_takes_each_fibre_from_muscle(typed):
    semg = _semg(typed, SurfaceFiberDispersion(fiber_properties="muscle", diameter_scales_source=True))
    _muaps(semg)
    report = semg.fiber_dispersion_report
    cv = np.asarray(typed.muscle_fiber_conduction_velocities__mm_per_s.rescale(pq.m / pq.s).magnitude)
    d = np.asarray(typed.muscle_fiber_diameters__mm.rescale(pq.um).magnitude)
    for mu in range(len(typed.recruitment_thresholds)):
        mask = typed.assignment == mu
        if mask.any():
            assert report["cv_mean__m_s"][mu] == pytest.approx(cv[mask].mean(), rel=1e-12)
            assert report["diameter_mean__um"][mu] == pytest.approx(d[mask].mean(), rel=1e-12)


def test_muscle_fibres_make_fibre_diameter_irrelevant(typed):
    config = SurfaceFiberDispersion(fiber_properties="muscle", diameter_scales_source=True)
    np.testing.assert_allclose(_muaps(_semg(typed, config, fiber_diameter__um=40.0)),
                               _muaps(_semg(typed, config, fiber_diameter__um=70.0)), rtol=1e-10, atol=0)


def test_muscle_fibres_change_the_muap(typed):
    assert not np.allclose(_muaps(_semg(typed)), _muaps(_semg(typed, SurfaceFiberDispersion(fiber_properties="muscle"))))


def test_muscle_fibres_need_generated_properties(muscle):
    with pytest.raises(ValueError, match="_generate_fiber_properties"):
        _semg(muscle, SurfaceFiberDispersion(fiber_properties="muscle")).simulate_muaps(n_jobs=1, verbose=False,
                                                                                        use_gpu=False)


def test_window_check_scales_the_iap_tail_with_lambda(muscle):
    config = SurfaceFiberDispersion(nerve_delays=True)
    _muaps(_semg(muscle, config, mus=(11,), iap_kernel_length__mm=105.0))
    with pytest.raises(ValueError, match="iap_kernel_length__mm"):
        _muaps(_semg(muscle, config, mus=(11,), iap_kernel_length__mm=105.0, iap_lambda__per_mm=0.4))


def test_each_fibre_gets_its_own_muscle_values(typed, monkeypatch):
    from myogen.simulator.core.emg.surface import surface_emg

    seen = {}
    real = surface_emg.realize_fiber_dispersion

    def capture(config, positions, *args, **kwargs):
        seen.update(positions=positions, **kwargs)
        return real(config, positions, *args, **kwargs)

    monkeypatch.setattr(surface_emg, "realize_fiber_dispersion", capture)
    _muaps(_semg(typed, SurfaceFiberDispersion(fiber_properties="muscle", diameter_scales_source=True)))
    centres = np.asarray(typed.muscle_fiber_centers__mm.rescale(pq.mm).magnitude)[:, :2]
    cv = np.asarray(typed.muscle_fiber_conduction_velocities__mm_per_s.rescale(pq.m / pq.s).magnitude)
    d = np.asarray(typed.muscle_fiber_diameters__mm.rescale(pq.um).magnitude)
    for mu, positions in enumerate(seen["positions"]):
        index = [int(np.flatnonzero(np.all(centres == xy, axis=1))[0]) for xy in positions]
        np.testing.assert_array_equal(seen["fiber_cv__m_s"][mu], cv[index])
        np.testing.assert_array_equal(seen["fiber_diameters__um"][mu], d[index])
