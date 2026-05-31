import numpy as np
import quantities as pq
from neo import AnalogSignal, Block, Group, Segment, SpikeTrain


def _single_spike_block() -> Block:
    block = Block()
    segment = Segment(name="Pool_0")
    spiketrain = SpikeTrain([5] * pq.ms, t_start=0 * pq.ms, t_stop=11 * pq.ms)
    spiketrain.sampling_period = 1 * pq.ms
    segment.spiketrains.append(spiketrain)
    block.segments.append(segment)
    return block


def test_nmodl_loader_returns_true_only_for_insertable_mechanisms():
    from neuron import h

    from myogen.utils.nmodl import load_nmodl_mechanisms

    assert load_nmodl_mechanisms(quiet=True, strict=True)
    section = h.Section()
    section.insert("caL")


def test_population_connection_delay_uses_requested_delay_plus_axon_delay():
    from neuron import h

    from myogen.simulator.neuron.network import _connect_one_to_one, _connect_populations

    class Source:
        def __init__(self, global_id: int):
            self.ns = h.NetStim()
            self.global__ID = global_id
            self.axon_delay__ms = 2.0

    class Target:
        def __init__(self):
            self.section = h.Section()
            self.synapse__list = [h.ExpSyn(0.5, sec=self.section)]

    for connect in (_connect_populations, _connect_one_to_one):
        populations = {"source": [Source(0)], "target": [Target()]}
        if connect is _connect_populations:
            netcons = connect(
                populations,
                "source",
                "target",
                1.0,
                synaptic_delay=7.0,
                synaptic_weight=0.25,
            )
        else:
            netcons = connect(
                "source",
                "target",
                populations,
                1.0,
                synaptic_delay=7.0,
                synaptic_weight=0.25,
            )

        assert len(netcons) == 1
        assert netcons[0].delay == 9.0
        assert netcons[0].weight[0] == 0.25


def test_surface_emg_uses_convolution_not_correlation(monkeypatch):
    from myogen.simulator.core.emg.surface import surface_emg as surface_module
    from myogen.simulator.core.emg.surface.surface_emg import SurfaceEMG
    from myogen.utils.neo import create_grid_signal

    monkeypatch.setattr(surface_module, "HAS_CUPY", False)

    simulator = object.__new__(SurfaceEMG)
    simulator._MUs_to_simulate = [0]
    simulator._sampling_frequency__Hz = 1000.0
    simulator._surface_emg__Block = None
    simulator._spike_train__Block = None

    muap_block = Block()
    group = Group(name="ElectrodeArray_0")
    segment = Segment(name="MUAP_0")
    muap = np.array([[[1.0]], [[2.0]], [[3.0]], [[4.0]]])
    segment.analogsignals.append(
        create_grid_signal(muap * pq.mV, grid_shape=(1, 1), sampling_rate=1000 * pq.Hz)
    )
    group.segments.append(segment)
    muap_block.groups.append(group)
    simulator._muaps__Block = muap_block

    result = SurfaceEMG.simulate_surface_emg(simulator, _single_spike_block(), verbose=False)
    observed = result.groups[0].segments[0].analogsignals[0].magnitude[:, 0]

    expected = np.array([0, 0, 0, 0, 1, 2, 3, 4, 0, 0, 0], dtype=float)
    np.testing.assert_allclose(observed, expected)


def test_surface_emg_resampling_handles_fp_rounding_lengths(monkeypatch):
    """Regression for issue #12.

    The temporal resampling in ``simulate_surface_emg`` built its interpolation
    grids with ``np.arange`` and a float step. When ``N * timestep`` is not
    exactly representable in IEEE 754 the grid came out one element too long,
    so ``np.interp`` raised ``ValueError: fp and xp are not of the same
    length``. ``N = 1001`` bins at ``dt = 1 ms`` hits that case
    (``1001 * 0.001 == 1.0010000000000001``).
    """
    from myogen.simulator.core.emg.surface import surface_emg as surface_module
    from myogen.simulator.core.emg.surface.surface_emg import SurfaceEMG
    from myogen.utils.neo import create_grid_signal

    monkeypatch.setattr(surface_module, "HAS_CUPY", False)

    simulator = object.__new__(SurfaceEMG)
    simulator._MUs_to_simulate = [0]
    simulator._sampling_frequency__Hz = 1000.0
    simulator._surface_emg__Block = None
    simulator._spike_train__Block = None

    muap_block = Block()
    group = Group(name="ElectrodeArray_0")
    segment = Segment(name="MUAP_0")
    muap = np.array([[[1.0]], [[2.0]], [[3.0]], [[4.0]]])
    segment.analogsignals.append(
        create_grid_signal(muap * pq.mV, grid_shape=(1, 1), sampling_rate=1000 * pq.Hz)
    )
    group.segments.append(segment)
    muap_block.groups.append(group)
    simulator._muaps__Block = muap_block

    block = Block()
    pool = Segment(name="Pool_0")
    spiketrain = SpikeTrain([5] * pq.ms, t_start=0 * pq.ms, t_stop=1001 * pq.ms)
    spiketrain.sampling_period = 1 * pq.ms
    pool.spiketrains.append(spiketrain)
    block.segments.append(pool)

    # Must not raise, and the resampled output keeps the native length
    # (output rate 1 kHz == spike-train rate, so N stays 1001).
    result = SurfaceEMG.simulate_surface_emg(simulator, block, verbose=False)
    observed = result.groups[0].segments[0].analogsignals[0].magnitude[:, 0]
    assert observed.shape[0] == 1001


def test_intramuscular_emg_uses_convolution_not_correlation(monkeypatch):
    from myogen.simulator.core.emg.intramuscular import (
        intramuscular_emg as intramuscular_module,
    )
    from myogen.simulator.core.emg.intramuscular.intramuscular_emg import (
        IntramuscularEMG,
    )

    monkeypatch.setattr(intramuscular_module, "HAS_CUPY", False)

    simulator = object.__new__(IntramuscularEMG)
    simulator._MUs_to_simulate = [0]
    simulator._sampling_frequency__Hz = 1000.0
    simulator._intramuscular_emg__Block = None
    simulator._spike_train__Block = None

    muap_block = Block()
    segment = Segment(name="MUAP_0")
    segment.analogsignals.append(
        AnalogSignal(
            np.array([[1.0], [2.0], [3.0], [4.0]]) * pq.dimensionless,
            sampling_rate=1000 * pq.Hz,
        )
    )
    muap_block.segments.append(segment)
    simulator._muaps__Block = muap_block

    result = IntramuscularEMG.simulate_intramuscular_emg(
        simulator,
        _single_spike_block(),
        verbose=False,
    )
    observed = result.segments[0].analogsignals[0].magnitude[:, 0]

    expected = np.array([0, 0, 0, 0, 0.25, 0.5, 0.75, 1.0, 0, 0, 0])
    np.testing.assert_allclose(observed, expected)


def test_force_model_vectorized_matches_force_model():
    """ForceModelVectorized must produce the same force output as ForceModel.

    The two classes share the IPI/gain/twitch pipeline via ``force_utils`` so the
    optional vectorized path cannot silently drift from the reference model.
    """
    from myogen.simulator.core.force.force_model import ForceModel
    from myogen.simulator.core.force.force_model_vectorized import (
        ForceModelVectorized,
    )

    rng = np.random.default_rng(0)
    n_mus = 4
    recruitment_thresholds = np.linspace(0.1, 1.0, n_mus)
    recording_frequency = 2000.0  # Hz
    t_stop__s = 0.5
    sampling_period = (1.0 / recording_frequency) * pq.s

    segment = Segment(name="Pool_0")
    for _ in range(n_mus):
        spike_times__s = np.sort(rng.uniform(0.0, t_stop__s, size=8))
        spiketrain = SpikeTrain(
            spike_times__s * pq.s, t_start=0 * pq.s, t_stop=t_stop__s * pq.s
        )
        spiketrain.sampling_period = sampling_period
        segment.spiketrains.append(spiketrain)
    block = Block()
    block.segments.append(segment)

    # Both models take a quantity for the recording frequency and must yield
    # the same force for the same spike trains.
    reference = ForceModel(
        recruitment_thresholds=recruitment_thresholds,
        recording_frequency__Hz=recording_frequency * pq.Hz,
    )
    vectorized = ForceModelVectorized(
        recruitment_thresholds=recruitment_thresholds,
        recording_frequency__Hz=recording_frequency * pq.Hz,
    )

    reference_force = reference.generate_force(block, verbose=False)
    vectorized_force = vectorized.generate_force(block, verbose=False)

    assert reference_force.shape == vectorized_force.shape
    np.testing.assert_allclose(
        vectorized_force.magnitude,
        reference_force.magnitude,
        rtol=1e-6,
        atol=1e-9,
    )


def test_to_float_rescales_quantities_to_native_neuron_units():
    """NEURON NetCon values must be rescaled to native units, not raw magnitudes.

    The buggy path took ``float(quantity)`` (NEURON's __float__ coercion), which
    drops the unit and returns the raw magnitude -- so a value in a non-native unit
    would be stored wrong (e.g. 1 s stored as 1.0 ms). These cases all differ
    between the buggy (raw magnitude) and fixed (rescaled) behavior.
    """
    from myogen.simulator.neuron.network import _to_float

    assert _to_float(1.0 * pq.s, pq.ms) == 1000.0  # buggy gave 1.0
    assert _to_float(0.0006 * pq.mS, pq.uS) == 0.6  # buggy gave 0.0006
    assert _to_float(-0.01 * pq.V, pq.mV) == -10.0  # buggy gave -0.01
    assert _to_float(0.6 * pq.uS, pq.uS) == 0.6  # already native
    assert _to_float(7.0, pq.ms) == 7.0  # plain float passes through


def test_default_synaptic_params_rescale_nonnative_delay():
    """A delay given in seconds must be stored in NEURON as milliseconds."""
    from neuron import h

    from myogen.simulator.neuron.network import _apply_default_synaptic_params

    class _Src:  # no axon_delay__ms attribute
        pass

    section = h.Section()
    syn = h.ExpSyn(0.5, sec=section)
    stim = h.NetStim()  # keep a reference: NEURON segfaults if the source is GC'd
    netcon = h.NetCon(stim, syn)

    _apply_default_synaptic_params(netcon, _Src(), synaptic_delay=1.0 * pq.s)

    assert netcon.delay == 1000.0  # 1 s -> 1000 ms; buggy code stored 1.0
    assert netcon.weight[0] == 0.6  # DEFAULT_SYNAPTIC_WEIGHT magnitude (uS)
    assert netcon.threshold == -10.0  # DEFAULT_SPIKE_THRESHOLD magnitude (mV)


def test_intramuscular_add_noise_realistic_preserves_per_channel_snr():
    """``noise_type='realistic'`` honours MyoGen's per-channel SNR contract.

    The colored-noise pipeline is calibrated against real fine-wire iEMG
    (1/f base + mid-band emphasis + heavy tails + powerline). The
    integration must still hit ``snr__dB`` independently per electrode
    so channels with different amplitudes get appropriately scaled
    noise, matching the legacy gaussian path.
    """
    from myogen import set_random_seed
    from myogen.simulator.core.emg.intramuscular.intramuscular_emg import (
        IntramuscularEMG,
    )

    set_random_seed(0)

    fs_hz = 4096.0
    n_samples = int(8.0 * fs_hz)  # 8 s — long enough for the RMS to settle
    rng = np.random.default_rng(0)

    # Two channels with deliberately different amplitudes (5x apart).
    base = rng.standard_normal(n_samples)
    emg_array = np.stack([base, 5.0 * base], axis=1)

    block = Block()
    segment = Segment(name="Pool_0")
    segment.analogsignals.append(
        AnalogSignal(
            emg_array * pq.dimensionless,
            t_start=0 * pq.s,
            sampling_rate=fs_hz * pq.Hz,
        )
    )
    block.segments.append(segment)

    simulator = object.__new__(IntramuscularEMG)
    simulator._intramuscular_emg__Block = block
    simulator._noisy_intramuscular_emg__Block = None
    simulator._sampling_frequency__Hz = fs_hz

    snr_db = 20.0
    noisy_block = IntramuscularEMG.add_noise(
        simulator,
        snr__dB=snr_db,
        noise_type="realistic",
        # Disable powerline so the residual is pure colored noise — the
        # per-channel SNR check would otherwise count the 50 Hz line as
        # additional "noise" beyond the budgeted RMS.
        powerline_amplitude=0.0,
    )

    noisy = noisy_block.segments[0].analogsignals[0].magnitude
    assert noisy.shape == emg_array.shape

    for ch in range(emg_array.shape[1]):
        signal_rms = float(np.sqrt(np.mean(emg_array[:, ch] ** 2)))
        noise_rms = float(np.sqrt(np.mean((noisy[:, ch] - emg_array[:, ch]) ** 2)))
        observed_snr_db = 20.0 * np.log10(signal_rms / noise_rms)
        # The colored-noise pipeline normalises to the budgeted RMS via
        # signal stats, so the realised per-channel SNR should sit within
        # ±1.5 dB of the target across an 8 s window.
        assert abs(observed_snr_db - snr_db) < 1.5, (
            f"channel {ch}: SNR {observed_snr_db:.2f} dB != {snr_db} ± 1.5 dB"
        )


def test_intramuscular_add_noise_realistic_injects_powerline_peak():
    """An exaggerated powerline amplitude must show as a 50 Hz peak."""
    from myogen import set_random_seed
    from myogen.simulator.core.emg.intramuscular.intramuscular_emg import (
        IntramuscularEMG,
    )

    set_random_seed(0)

    fs_hz = 2048.0
    n_samples = int(4.0 * fs_hz)
    rng = np.random.default_rng(1)

    emg_array = (rng.standard_normal((n_samples, 1)) * 0.01).astype(float)

    block = Block()
    segment = Segment(name="Pool_0")
    segment.analogsignals.append(
        AnalogSignal(
            emg_array * pq.dimensionless,
            t_start=0 * pq.s,
            sampling_rate=fs_hz * pq.Hz,
        )
    )
    block.segments.append(segment)

    simulator = object.__new__(IntramuscularEMG)
    simulator._intramuscular_emg__Block = block
    simulator._noisy_intramuscular_emg__Block = None
    simulator._sampling_frequency__Hz = fs_hz

    noisy_block = IntramuscularEMG.add_noise(
        simulator,
        snr__dB=0.0,  # noise RMS == signal RMS
        noise_type="realistic",
        powerline_hz=50.0,
        powerline_amplitude=0.8,  # exaggerated
        powerline_harmonic_ratios=[1.0, 0.0, 0.0, 0.0, 0.0],  # fundamental only
    )

    residual = (
        noisy_block.segments[0].analogsignals[0].magnitude[:, 0]
        - emg_array[:, 0]
    )
    freqs = np.fft.rfftfreq(len(residual), d=1.0 / fs_hz)
    psd = np.abs(np.fft.rfft(residual)) ** 2

    bin_50 = int(np.argmin(np.abs(freqs - 50.0)))
    # Pick a quiet reference band away from 50 Hz that the colored noise
    # actually covers, so we are measuring the powerline spike, not the
    # spectral slope.
    ref_lo = int(np.argmin(np.abs(freqs - 200.0)))
    ref_hi = int(np.argmin(np.abs(freqs - 400.0)))
    ref_psd = float(np.mean(psd[ref_lo : ref_hi + 1]))
    peak_psd = float(np.mean(psd[max(0, bin_50 - 1) : bin_50 + 2]))

    assert peak_psd > 10.0 * ref_psd, (
        f"50 Hz peak ({peak_psd:.2e}) not >10x ref band ({ref_psd:.2e})"
    )


def test_baseline_drift_off_by_default_preserves_noise_rms():
    """Default ``baseline_drift_rms_uv=0`` must leave generated noise unchanged.

    Regression guard for the SNR contract — historical callers don't pass
    the new kwargs and expect ``noise_rms`` to be the only thing that
    controls the output RMS.
    """
    from myogen.utils import generate_realistic_noise

    fs_hz = 10240.0
    n = int(8.0 * fs_hz)

    # Explicit seeded rng so the two calls are bit-identical when drift
    # is at its default (no extra rng draws).
    no_drift = generate_realistic_noise(
        n, fs_hz, noise_rms=5.0, rng=np.random.default_rng(0)
    )
    default_drift = generate_realistic_noise(
        n, fs_hz, noise_rms=5.0, baseline_drift_rms_uv=0.0,
        rng=np.random.default_rng(0),
    )
    np.testing.assert_array_equal(no_drift, default_drift)
    rms = float(np.sqrt(np.mean(no_drift ** 2)))
    assert abs(rms - 5.0) < 0.05, f"broadband RMS {rms:.3f} != 5.0 ± 0.05"


def test_baseline_drift_amplitude_and_band():
    """Baseline drift hits the requested RMS and stays in the LF band.

    The drift is additive on top of broadband, so total noise RMS
    should match ``sqrt(noise_rms**2 + drift_rms**2)``. The band is
    now a hard band-limit at ``baseline_drift_high_hz``, so at least
    90% of the drift power must land below that cutoff.
    """
    from scipy import signal as sig

    from myogen.utils import generate_realistic_noise

    fs_hz = 10240.0
    n = int(16.0 * fs_hz)
    noise_rms = 5.0
    drift_rms = 2.0
    high_hz = 1.0

    # Same rng seed for both calls so the broadband draws match, then
    # ``with_drift - no_drift`` isolates the drift component exactly.
    no_drift = generate_realistic_noise(
        n, fs_hz, noise_rms=noise_rms, rng=np.random.default_rng(0)
    )
    with_drift = generate_realistic_noise(
        n, fs_hz, noise_rms=noise_rms,
        baseline_drift_rms_uv=drift_rms,
        baseline_drift_alpha=1.75,
        baseline_drift_high_hz=high_hz,
        rng=np.random.default_rng(0),
    )

    expected_total = float(np.sqrt(noise_rms ** 2 + drift_rms ** 2))
    observed_total = float(np.sqrt(np.mean(with_drift ** 2)))
    assert abs(observed_total - expected_total) < 0.15, (
        f"total RMS {observed_total:.3f} != expected {expected_total:.3f} ± 0.15"
    )

    drift_only = with_drift - no_drift
    drift_only_rms = float(np.sqrt(np.mean(drift_only ** 2)))
    assert abs(drift_only_rms - drift_rms) < 0.15, (
        f"drift-only RMS {drift_only_rms:.3f} != requested {drift_rms:.3f}"
    )

    # >= 90% of drift power must lie below the hard band edge.
    sos_lp = sig.butter(4, high_hz, fs=fs_hz, btype="low", output="sos")
    drift_in_band = sig.sosfiltfilt(sos_lp, drift_only)
    drift_in_band_rms = float(np.sqrt(np.mean(drift_in_band ** 2)))
    assert drift_in_band_rms >= 0.9 * drift_only_rms, (
        f"in-band (<{high_hz} Hz) drift RMS {drift_in_band_rms:.3f} is only "
        f"{100 * drift_in_band_rms / drift_only_rms:.0f}% of total drift RMS "
        f"{drift_only_rms:.3f} — drift is leaking above the band limit"
    )


def test_baseline_drift_slope_matches_alpha():
    """Fitted log-log PSD slope over the drift band matches -alpha.

    The sub-1-Hz band has only ~8 Welch bins at 0.125 Hz resolution,
    so the slope estimate has significant variance. We use a longer
    signal (64 s) to get more Welch segments and loosen the tolerance
    accordingly — this is a sanity check on the spectral SHAPE, not a
    precision unit test of the fitter.
    """
    from scipy import signal as sig

    from myogen.utils import generate_realistic_noise

    fs_hz = 10240.0
    n = int(64.0 * fs_hz)
    alpha_in = 2.0
    high_hz = 1.0

    drift = generate_realistic_noise(
        n, fs_hz, noise_rms=0.0,
        powerline_hz=0.0,
        baseline_drift_rms_uv=5.0,
        baseline_drift_alpha=alpha_in,
        baseline_drift_high_hz=high_hz,
        rng=np.random.default_rng(0),
    )

    nperseg = min(len(drift), int(8 * fs_hz))
    freqs, psd = sig.welch(drift, fs=fs_hz, nperseg=nperseg, average="median")
    fit_mask = (freqs >= freqs[1]) & (freqs <= high_hz) & (psd > 0)
    assert np.count_nonzero(fit_mask) >= 3, "too few PSD bins for slope fit"
    slope, _ = np.polyfit(np.log10(freqs[fit_mask]), np.log10(psd[fit_mask]), 1)
    fitted_alpha = -slope
    assert abs(fitted_alpha - alpha_in) < 1.0, (
        f"fitted alpha {fitted_alpha:.2f} != target {alpha_in:.2f} ± 1.0"
    )


def test_calibrate_baseline_drift_profile_round_trip():
    """Calibration on a known-drift trace recovers RMS and α to within tolerance.

    The generation and calibration bands must agree. With
    ``low_hz=None`` the generator can place power at frequencies
    below what a typical Welch nperseg can resolve, which would bias
    the recovered RMS downward — that is a real limitation of any
    PSD-based fitter on short traces. To exercise the round-trip
    fairly we pin both the generation and the calibration to the same
    band [0.1, 1.0] Hz, well above the Welch resolution.
    """
    from myogen.utils import (
        calibrate_baseline_drift_profile,
        generate_realistic_noise,
    )

    fs_hz = 10240.0
    n = int(64.0 * fs_hz)
    target_rms = 3.0
    target_alpha = 1.75
    low_hz = 0.1
    high_hz = 1.0

    drift = generate_realistic_noise(
        n, fs_hz, noise_rms=0.0,
        powerline_hz=0.0,
        baseline_drift_rms_uv=target_rms,
        baseline_drift_alpha=target_alpha,
        baseline_drift_low_hz=low_hz,
        baseline_drift_high_hz=high_hz,
        rng=np.random.default_rng(0),
    )

    profile = calibrate_baseline_drift_profile(
        drift, fs_hz, band=(low_hz, high_hz)
    )

    assert {
        "baseline_drift_rms_uv",
        "baseline_drift_alpha",
        "baseline_drift_low_hz",
        "baseline_drift_high_hz",
    } <= set(profile.keys())
    assert abs(profile["baseline_drift_rms_uv"] - target_rms) < 0.30 * target_rms, (
        f"recovered RMS {profile['baseline_drift_rms_uv']:.3f} != "
        f"target {target_rms:.3f} ± 30%"
    )
    assert abs(profile["baseline_drift_alpha"] - target_alpha) < 1.0, (
        f"recovered α {profile['baseline_drift_alpha']:.2f} != "
        f"target {target_alpha:.2f} ± 1.0"
    )
    assert profile["baseline_drift_high_hz"] == high_hz


def test_baseline_drift_rejects_invalid_params():
    """Negative/zero/out-of-range parameter values must raise."""
    import pytest

    from myogen.utils import generate_realistic_noise

    fs_hz = 10240.0
    n = 4096

    with pytest.raises(ValueError, match="baseline_drift_rms_uv"):
        generate_realistic_noise(
            n, fs_hz, noise_rms=5.0, baseline_drift_rms_uv=-1.0
        )
    with pytest.raises(ValueError, match="baseline_drift_alpha"):
        generate_realistic_noise(
            n, fs_hz, noise_rms=5.0,
            baseline_drift_rms_uv=1.0, baseline_drift_alpha=0.0,
        )
    with pytest.raises(ValueError, match="baseline_drift_high_hz"):
        generate_realistic_noise(
            n, fs_hz, noise_rms=5.0,
            baseline_drift_rms_uv=1.0, baseline_drift_high_hz=0.0,
        )
    with pytest.raises(ValueError, match="baseline_drift_high_hz"):
        # high_hz >= Nyquist
        generate_realistic_noise(
            n, fs_hz, noise_rms=5.0,
            baseline_drift_rms_uv=1.0, baseline_drift_high_hz=fs_hz / 2.0,
        )
    with pytest.raises(ValueError, match="baseline_drift_low_hz"):
        generate_realistic_noise(
            n, fs_hz, noise_rms=5.0,
            baseline_drift_rms_uv=1.0, baseline_drift_low_hz=0.0,
        )
    with pytest.raises(ValueError, match="baseline_drift_low_hz"):
        # low_hz >= high_hz
        generate_realistic_noise(
            n, fs_hz, noise_rms=5.0,
            baseline_drift_rms_uv=1.0,
            baseline_drift_low_hz=2.0,
            baseline_drift_high_hz=1.0,
        )
