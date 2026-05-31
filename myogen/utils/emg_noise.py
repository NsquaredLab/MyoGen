"""
Realistic intramuscular EMG noise model.

Based on analysis of real bipolar fine-wire iEMG recordings (vastus
medialis / lateralis, stainless-steel bipolar electrodes).

Key findings that inform the model:
  1. Noise is signal-independent (constant floor, not SNR-scaled).
     The HF noise floor during rest vs active contraction is identical
     (ratio ~0.9-1.0 across studies).
  2. Noise is spectrally colored with a mid-band emphasis.  The PSD
     peaks around 500-1000 Hz (from electrode/amplifier characteristics)
     with a slope of ~-0.5 above the peak.
  3. Amplitude distribution has heavier tails than Gaussian
     (excess kurtosis 1-6 for controlled isometric, higher for
     dynamic tasks).

Usage:
    from iemg_noise import add_realistic_noise

    noisy = add_realistic_noise(
        clean_emg,
        fs_hz=10240,
        target_snr_db=20.0,   # SNR determines noise floor amplitude
        spectral_slope=-0.5,
    )

The noise_rms is computed from the signal's active-period RMS and the
target_snr_db, then applied as a CONSTANT floor (same amplitude during
rest and contraction).  This matches real recordings where the noise
floor is instrumental, not physiological.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sig


def _colored_noise(
    n_samples: int,
    fs_hz: float,
    spectral_slope: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate colored noise with a given spectral slope.

    Uses the spectral-shaping method: generate white noise in the
    frequency domain, multiply by |f|^(slope/2), inverse FFT.

    Parameters
    ----------
    n_samples : int
        Number of output samples.
    fs_hz : float
        Sampling rate (Hz).
    spectral_slope : float
        PSD slope in log-log space.  0 = white, -1 = pink, -2 = brown.
        Typical iEMG noise: -0.4 to -0.8.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    ndarray
        Noise signal, zero-mean, unit variance.
    """
    n_fft = n_samples + (n_samples % 2)  # ensure even

    # White noise in frequency domain (complex Gaussian)
    white = rng.standard_normal(n_fft) + 1j * rng.standard_normal(n_fft)

    # Frequency axis
    freqs = np.fft.fftfreq(n_fft, d=1.0 / fs_hz)

    # Shaping filter: |f|^(slope/2) for PSD slope
    # Avoid DC (f=0) — set it to 0 to remove DC offset
    shape = np.zeros(n_fft)
    nonzero = freqs != 0
    shape[nonzero] = np.abs(freqs[nonzero]) ** (spectral_slope / 2.0)

    # Apply shaping
    colored = np.fft.ifft(white * shape).real[:n_samples]

    # Normalize to unit variance
    std = colored.std()
    if std > 0:
        colored /= std

    return colored


def _add_powerline(
    noise: np.ndarray,
    fs_hz: float,
    freq_hz: float,
    amplitude_fraction: float,
    rng: np.random.Generator,
    harmonic_ratios: list[float] | None = None,
    frequency_drift_hz: float = 0.3,
    amplitude_modulation_depth: float = 0.15,
) -> np.ndarray:
    """Add powerline interference with harmonics, frequency drift, and AM.

    Powerline is applied as an additional layer on top of the device
    noise colour — the fundamental at ``freq_hz`` plus harmonics with
    amplitudes scaled by ``harmonic_ratios``. This matches real
    recordings where 100 / 150 / 200 Hz harmonics often have comparable
    energy to the fundamental.

    Real mains frequency is not exactly 50 or 60 Hz — it wanders
    ±0.05–0.2 Hz on multi-second timescales (frequency containment
    reserve, load shifts), and the amplitude wobbles a few percent
    from load coupling. Modelling these as a slow random walk in the
    instantaneous frequency plus a low-pass-modulated amplitude
    envelope turns the PSD line from a delta-like spike into a curved
    peak with shoulders/sidebands — exactly what is observed in real
    recordings.

    Parameters
    ----------
    noise : ndarray
        Input noise signal.
    fs_hz : float
        Sampling rate.
    freq_hz : float
        Powerline fundamental (50 or 60 Hz).
    amplitude_fraction : float
        Amplitude of the fundamental as a fraction of noise RMS.
        Typical: 0.05-0.3 depending on grounding quality.
    rng : np.random.Generator
        Random generator (for random per-harmonic phases and the
        frequency drift trajectory).
    harmonic_ratios : list of float, optional
        Amplitude ratios of harmonics relative to the fundamental.
        ``harmonic_ratios[0]`` multiplies the fundamental (default 1.0),
        ``harmonic_ratios[1]`` the 2nd harmonic (2× freq_hz), etc.
        Default: ``[1.0, 0.5, 0.3, 0.15, 0.08]`` (50, 100, 150, 200, 250 Hz).
    frequency_drift_hz : float, default 0.3
        Standard deviation of the slow instantaneous-frequency drift of
        the fundamental, in Hz. Harmonics drift proportionally (k× the
        fundamental). Set to 0 for a pure-tone powerline (sharp delta
        in the PSD). 0.3 Hz is a phenomenological default that reproduces
        the typical 2–5 Hz FWHM of real-recording mains peaks (covers
        grid frequency drift + ground-loop variations + window leakage).
    amplitude_modulation_depth : float, default 0.15
        Fractional AM depth applied to the powerline carriers. 0.15
        means the envelope wobbles by roughly ±15%. The modulating
        signal is low-pass filtered noise (cutoff ~2 Hz) so sidebands
        appear within ±2 Hz of each line. Set to 0 to disable.

    Returns
    -------
    ndarray
        Noise with powerline + harmonics added.
    """
    if harmonic_ratios is None:
        harmonic_ratios = [1.0, 0.5, 0.3, 0.15, 0.08]

    n = len(noise)
    nyq = fs_hz / 2.0
    base = amplitude_fraction * noise.std()

    # Slow frequency drift of the fundamental — low-pass-filtered random
    # walk normalised to the requested std. Cutoff ~0.5 Hz so the drift
    # varies on multi-second timescales like real grid frequency.
    if frequency_drift_hz > 0:
        drift = rng.standard_normal(n)
        sos_drift = sig.butter(2, 0.5 / nyq, btype="low", output="sos")
        drift = sig.sosfiltfilt(sos_drift, drift)
        drift_std = drift.std()
        if drift_std > 0:
            drift = drift * (frequency_drift_hz / drift_std)
    else:
        drift = np.zeros(n)

    # Slow amplitude modulation — low-pass-filtered white noise centred
    # at 1.0, generating ±depth wobble that produces narrow sidebands
    # around each harmonic.
    if amplitude_modulation_depth > 0:
        am = rng.standard_normal(n)
        sos_am = sig.butter(2, 2.0 / nyq, btype="low", output="sos")
        am = sig.sosfiltfilt(sos_am, am)
        am_std = am.std()
        if am_std > 0:
            am = 1.0 + amplitude_modulation_depth * (am / am_std)
        else:
            am = np.ones(n)
        am = np.clip(am, 0.0, None)
    else:
        am = np.ones(n)

    powerline = np.zeros(n)
    for k, ratio in enumerate(harmonic_ratios):
        f_base = freq_hz * (k + 1)
        if f_base >= nyq or ratio <= 0:
            continue
        # Instantaneous frequency = f_base + (k+1) * drift  (harmonics
        # share the mains drift, scaled by the harmonic order).
        f_inst = f_base + (k + 1) * drift
        # Instantaneous phase = 2π ∫ f(t) dt, plus a random offset.
        phase_inst = 2.0 * np.pi * np.cumsum(f_inst) / fs_hz
        phase_inst += rng.uniform(0, 2 * np.pi)
        powerline += base * ratio * am * np.sin(phase_inst)
    return noise + powerline


def _make_heavy_tailed(
    noise: np.ndarray,
    excess_kurtosis: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Reshape noise to have heavier tails (higher kurtosis).

    Uses a sparse-spike approach: replace a small fraction of samples
    with larger values drawn from a heavier distribution.  This mimics
    the occasional artifacts and cross-talk spikes seen in real iEMG.

    Parameters
    ----------
    noise : ndarray
        Input noise (should be zero-mean, unit-ish variance).
    excess_kurtosis : float
        Target excess kurtosis (0 = Gaussian, 3-6 = typical real iEMG).
    rng : np.random.Generator
        Random generator.

    Returns
    -------
    ndarray
        Noise with heavier tails, re-normalized to original std.
    """
    if excess_kurtosis <= 0.5:
        return noise

    original_std = noise.std()

    # Conservative spike injection — fewer, smaller spikes than before.
    # Real iEMG kurtosis (1-6) comes from a small fraction of outliers.
    spike_frac = min(0.005, excess_kurtosis / 2000.0)
    n_spikes = max(1, int(len(noise) * spike_frac))

    # t-distribution with moderate df for controlled tail weight
    df = max(4.0, 12.0 - excess_kurtosis)
    spike_amplitudes = rng.standard_t(df, size=n_spikes)

    # Scale: 3-5x noise std (not 3-8x as before)
    spike_scale = 2.5 + excess_kurtosis * 0.3
    spike_amplitudes *= spike_scale

    # Insert at random positions
    positions = rng.choice(len(noise), size=n_spikes, replace=False)
    noise = noise.copy()
    noise[positions] += spike_amplitudes

    # Re-normalize to original std
    noise *= original_std / noise.std()

    return noise


def _emg_band_shape(
    noise: np.ndarray,
    fs_hz: float,
    peak_hz: float = 1000.0,
    bandwidth_hz: float = 1400.0,
    emphasis_blend: float = 0.40,
) -> np.ndarray:
    """Shape noise to emphasize the EMG band (200-1800 Hz).

    Real iEMG noise has a mild spectral peak in the 500-1500 Hz range
    (from electrode-tissue interface and amplifier bandwidth) but its
    overall shape is a smooth monotonic decay in log-log, not a sharp
    resonance. This function applies a broad, low-order bandpass blended
    with the broadband colour to preserve a continuous PSD without the
    bumps a narrow high-order filter introduces.

    Parameters
    ----------
    noise : ndarray
        Input noise signal.
    fs_hz : float
        Sampling rate.
    peak_hz : float, default 1000.0
        Center frequency of the spectral emphasis.
    bandwidth_hz : float, default 1400.0
        Bandwidth around the peak — wide enough to avoid bulges.
    emphasis_blend : float, default 0.40
        Fraction of bandpass-emphasized signal in the blend. Lower
        values produce smoother PSDs closer to monotonic decay.

    Returns
    -------
    ndarray
        Spectrally shaped noise, variance restored.

    Notes
    -----
    The analog HPF that strips infra-low drift is handled in a separate
    explicit stage inside :func:`generate_realistic_noise`
    (``analog_hpf_hz``), so this function performs only the band
    emphasis. Stacking two HPFs here would double the rolloff order
    and cut the PSD below the cutoff far steeper than any real device.
    """
    nyq = fs_hz / 2.0
    original_std = noise.std()

    # Broad 2nd-order bandpass emphasis — low order avoids ringing/ripple
    # that shows up as bulges in the PSD of the shaped noise.
    low_bp = max((peak_hz - bandwidth_hz / 2) / nyq, 0.01)
    high_bp = min((peak_hz + bandwidth_hz / 2) / nyq, 0.99)
    sos_bp = sig.butter(2, [low_bp, high_bp], btype="band", output="sos")
    emphasis = sig.sosfiltfilt(sos_bp, noise)

    # Blend: modest emphasis + dominant broadband
    blended = emphasis_blend * emphasis + (1 - emphasis_blend) * noise

    # Restore variance
    blended *= original_std / max(blended.std(), 1e-12)

    return blended


def _baseline_drift(
    n_samples: int,
    fs_hz: float,
    target_rms: float,
    alpha: float,
    low_hz: float,
    high_hz: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate band-limited 1/f^α baseline drift at a target RMS.

    Models the slow low-frequency wander seen in real recordings —
    electrode/interface noise, postural shifts, breathing, sweating.

    Spectral form is paper-constrained: PSD ∝ ``1/f^alpha`` over an
    explicit band ``[low_hz, high_hz]`` and zero outside, consistent
    with Huigen, Peper & Grimbergen 2002 (DOI 10.1007/BF02344216) and
    Gondran et al. 1996 (DOI 10.1007/BF02523851), which report α in
    the 1.5–2.0 range for electrode/interface noise. The hard band
    limit (rather than a Lorentzian rolloff) keeps the model scoped
    to baseline wander; broadband movement artifacts (0–20 Hz per
    De Luca et al. 2010, DOI 10.1016/j.jbiomech.2010.01.027) are a
    separate phenomenon and require a different contaminant model.

    The drift represents the residual that survives the amplifier's
    analog HPF — added after the broadband HPF stage so the caller
    sees the same amplitude they requested. A downstream user-applied
    HPF will attenuate it further (realistic).

    Public input validation lives in :func:`generate_realistic_noise`;
    this helper assumes its arguments have already been checked.
    """
    if target_rms == 0 or n_samples < 4:
        return np.zeros(n_samples)

    n_fft = n_samples + (n_samples % 2)
    white = rng.standard_normal(n_fft) + 1j * rng.standard_normal(n_fft)
    freqs = np.fft.fftfreq(n_fft, d=1.0 / fs_hz)
    abs_freqs = np.abs(freqs)

    shape = np.zeros(n_fft)
    band = (abs_freqs > 0) & (abs_freqs >= low_hz) & (abs_freqs <= high_hz)
    if not np.any(band):
        # No FFT bin landed in [low_hz, high_hz] — recording too short
        # for the requested baseline-drift band. Caller is responsible
        # for raising; this helper just returns zeros.
        return np.zeros(n_samples)
    shape[band] = abs_freqs[band] ** (-alpha / 2.0)

    drift = np.fft.ifft(white * shape).real[:n_samples]
    drift -= float(np.mean(drift))

    rms = float(np.sqrt(np.mean(drift ** 2)))
    if rms > 0:
        drift *= target_rms / rms

    return drift


def generate_realistic_noise(
    n_samples: int,
    fs_hz: float,
    noise_rms: float,
    *,
    spectral_slope: float = -0.5,
    excess_kurtosis: float = 3.0,
    powerline_hz: float = 50.0,
    powerline_amplitude: float = 0.1,
    powerline_harmonic_ratios: list[float] | None = None,
    powerline_frequency_drift_hz: float = 0.3,
    powerline_amplitude_modulation_depth: float = 0.15,
    peak_hz: float = 1000.0,
    analog_hpf_hz: float = 10.0,
    baseline_drift_rms_uv: float = 0.0,
    baseline_drift_alpha: float = 1.75,
    baseline_drift_low_hz: float | None = None,
    baseline_drift_high_hz: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a single channel of realistic iEMG noise.

    Parameters
    ----------
    n_samples : int
        Length of the noise signal.
    fs_hz : float
        Sampling rate (Hz).
    noise_rms : float
        RMS amplitude of the noise floor (same units as EMG signal).
    spectral_slope : float, default -0.5
        PSD slope.  0 = white, -1 = pink.  Real iEMG: -0.4 to -0.8.
    excess_kurtosis : float, default 3.0
        Target excess kurtosis (0 = Gaussian).
        Controlled isometric: 1-6.  Dynamic tasks: 10-50.
    powerline_hz : float, default 50.0
        Powerline frequency.  Set to 0 to disable.
    powerline_amplitude : float, default 0.1
        Powerline amplitude as fraction of noise RMS.
    powerline_harmonic_ratios : list of float, optional
        Per-harmonic amplitude ratios relative to the fundamental.
        ``None`` uses the default ``[1.0, 0.5, 0.3, 0.15, 0.08]``.
    powerline_frequency_drift_hz : float, default 0.3
        Standard deviation of the slow random walk of the mains
        instantaneous frequency. Broadens the 50/60 Hz line peak from
        a delta into a ~2-5 Hz FWHM bump (typical of real recordings).
        Set to 0 for a pure-tone powerline.
    powerline_amplitude_modulation_depth : float, default 0.15
        Fractional AM depth on the powerline carriers (~15% wobble).
        Produces narrow sidebands within ±2 Hz of each line. Set to
        0 to disable.
    peak_hz : float, default 750.0
        Center of the EMG-band spectral emphasis.
    analog_hpf_hz : float, default 10.0
        Analog HPF cutoff applied before the powerline is mixed in.
        Mimics the amp's input AC-coupling. Set to 0 to disable.
    baseline_drift_rms_uv : float, default 0.0
        Target RMS of the band-limited 1/f^α baseline drift (post-HPF
        residual). Same units as ``noise_rms``. Set to 0 to disable
        — that is the default, keeping legacy behaviour unchanged.
        Must be >= 0.

        Spectral form is paper-constrained: PSD ∝ 1/f^α over an
        explicit band ``[baseline_drift_low_hz, baseline_drift_high_hz]``,
        consistent with electrode/interface noise studies (Huigen,
        Peper & Grimbergen 2002, DOI 10.1007/BF02344216; Gondran
        et al. 1996, DOI 10.1007/BF02523851). The default α=1.75 and
        ``high_hz=1.0`` are the midpoint and upper bound of the
        reported electrode-noise regime, not validated physiological
        constants for intramuscular EMG. Calibrate against real
        recordings via :func:`calibrate_baseline_drift_profile` if
        you need amplitude-accurate simulation. Movement artifacts
        (broadband, 0–20 Hz per De Luca et al. 2010, DOI
        10.1016/j.jbiomech.2010.01.027) are a separate phenomenon
        and would need their own contaminant model.

        .. note::
           **SNR contract**: drift is **additive on top** of the
           broadband noise. ``noise_rms`` still controls the
           broadband floor exactly, but enabling drift makes the
           total noise RMS slightly higher
           (``sqrt(noise_rms**2 + drift_rms**2)``). If you are
           using ``snr__dB`` upstream to hit a target SNR, factor
           this in or leave drift at 0.

        .. note::
           When called from a multi-channel context (e.g.
           :func:`add_realistic_noise`), each channel receives an
           **independent** drift realisation. Real electrode-array
           drift is often partially common-mode across nearby
           electrodes; this model does not capture that correlation.
    baseline_drift_alpha : float, default 1.75
        PSD slope α (positive number) for ``PSD ∝ 1/f^α``. Must be
        > 0. The literature regime is α ∈ [1.5, 2.0]; the default
        sits at the midpoint.
    baseline_drift_low_hz : float or None, default None
        Lower edge of the drift band, in Hz. ``None`` resolves to the
        lowest nonzero FFT bin for the current segment length. Must
        be > 0 when provided.
    baseline_drift_high_hz : float, default 1.0
        Upper edge of the drift band, in Hz. Must be > 0 and
        < fs_hz / 2. The default keeps the model scoped to sub-1 Hz
        baseline wander (see references above).
    rng : np.random.Generator or None
        Random generator.  Uses default if None.

    Returns
    -------
    ndarray, shape (n_samples,)
        Realistic noise signal.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Validate baseline-drift kwargs at the public-API entry point — the
    # guarded ``if baseline_drift_rms_uv > 0`` branch below would
    # otherwise let bad values pass through silently. Each error names
    # the offending kwarg so callers (and tests) can pattern-match.
    if baseline_drift_rms_uv < 0:
        raise ValueError(
            f"baseline_drift_rms_uv must be >= 0, got {baseline_drift_rms_uv!r}"
        )
    if baseline_drift_alpha <= 0:
        raise ValueError(
            f"baseline_drift_alpha must be > 0, got {baseline_drift_alpha!r}"
        )
    if baseline_drift_high_hz <= 0:
        raise ValueError(
            f"baseline_drift_high_hz must be > 0, got {baseline_drift_high_hz!r}"
        )
    if baseline_drift_high_hz >= fs_hz / 2.0:
        raise ValueError(
            f"baseline_drift_high_hz must be < Nyquist (fs_hz / 2 = "
            f"{fs_hz / 2.0:.3f}), got {baseline_drift_high_hz!r}"
        )
    if baseline_drift_low_hz is not None:
        if baseline_drift_low_hz <= 0:
            raise ValueError(
                f"baseline_drift_low_hz must be > 0, got "
                f"{baseline_drift_low_hz!r}"
            )
        if baseline_drift_low_hz >= baseline_drift_high_hz:
            raise ValueError(
                f"baseline_drift_low_hz ({baseline_drift_low_hz!r}) must be "
                f"< baseline_drift_high_hz ({baseline_drift_high_hz!r})"
            )

    # 1. Colored noise base
    noise = _colored_noise(n_samples, fs_hz, spectral_slope, rng)

    # 2. EMG-band spectral emphasis (replaces simple bandpass)
    noise = _emg_band_shape(noise, fs_hz, peak_hz=peak_hz)

    # 3. Heavy tails (conservative)
    noise = _make_heavy_tailed(noise, excess_kurtosis, rng)

    # 4. Analog high-pass filter — mimics the amplifier's front-end HPF
    # (Quattrocento: 10 Hz, Trigno: 20 Hz, clinical: 2 Hz). Without this
    # stage the broadband base extends flat down to DC, while real
    # recordings show a clear corner below 10-20 Hz. Applied before
    # rescaling so the post-HPF RMS hits the target exactly.
    if analog_hpf_hz > 0:
        sos_hpf = sig.butter(2, analog_hpf_hz, fs=fs_hz, btype="high", output="sos")
        noise = sig.sosfiltfilt(sos_hpf, noise)

    # 5. Scale the BROADBAND base to ``noise_rms`` BEFORE adding powerline
    # interference. ``noise_rms`` is the device's broadband floor RMS;
    # powerline is an additional environmental layer that adds power on
    # top — it should not eat the broadband budget. After this step,
    # ``noise.std() == noise_rms`` so ``_add_powerline`` will use the
    # broadband floor as the reference for ``base = amplitude * noise.std()``.
    current_rms = np.sqrt(np.mean(noise**2))
    if current_rms > 0:
        noise *= noise_rms / current_rms

    # 5b. Baseline drift — slow low-frequency wander attributed to
    # the electrode/interface (Huigen 2002; Gondran 1996). Additive
    # in absolute units, post-HPF (i.e., what actually survives into
    # the recorded signal). Off by default.
    if baseline_drift_rms_uv > 0:
        # Resolve ``low_hz=None`` to the lowest nonzero FFT bin so the
        # drift band always has at least one sample. If the recording
        # is so short that even this bin is above ``high_hz``, raise
        # rather than silently emit zeros.
        n_fft = n_samples + (n_samples % 2)
        lowest_nonzero = float(fs_hz / n_fft)
        drift_low_hz = (
            lowest_nonzero
            if baseline_drift_low_hz is None
            else float(baseline_drift_low_hz)
        )
        if drift_low_hz > baseline_drift_high_hz:
            raise ValueError(
                "baseline drift band is empty: lowest available FFT bin "
                f"({lowest_nonzero:.4f} Hz) exceeds baseline_drift_high_hz "
                f"({baseline_drift_high_hz}). Increase n_samples or "
                "baseline_drift_high_hz."
            )
        noise = noise + _baseline_drift(
            n_samples, fs_hz,
            target_rms=baseline_drift_rms_uv,
            alpha=baseline_drift_alpha,
            low_hz=drift_low_hz,
            high_hz=baseline_drift_high_hz,
            rng=rng,
        )

    # 6. Powerline interference (fundamental + harmonics, additive on top
    #    of the device colour; harmonics 50/100/150/200/250 Hz by default).
    if powerline_hz > 0 and powerline_amplitude > 0:
        noise = _add_powerline(
            noise, fs_hz, powerline_hz, powerline_amplitude, rng,
            harmonic_ratios=powerline_harmonic_ratios,
            frequency_drift_hz=powerline_frequency_drift_hz,
            amplitude_modulation_depth=powerline_amplitude_modulation_depth,
        )

    return noise


def add_realistic_noise(
    clean_emg: np.ndarray,
    fs_hz: float,
    target_snr_db: float = 20.0,
    *,
    noise_rms: float | None = None,
    spectral_slope: float = -0.5,
    excess_kurtosis: float = 3.0,
    powerline_hz: float = 50.0,
    powerline_amplitude: float = 0.1,
    powerline_harmonic_ratios: list[float] | None = None,
    powerline_frequency_drift_hz: float = 0.3,
    powerline_amplitude_modulation_depth: float = 0.15,
    peak_hz: float = 1000.0,
    analog_hpf_hz: float = 10.0,
    baseline_drift_rms_uv: float = 0.0,
    baseline_drift_alpha: float = 1.75,
    baseline_drift_low_hz: float | None = None,
    baseline_drift_high_hz: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add realistic noise to a clean EMG signal.

    The noise floor is constant (signal-independent), matching real
    intramuscular EMG recordings.  The amplitude is determined either
    by ``target_snr_db`` (auto-calibrated from the signal) or by an
    explicit ``noise_rms``.

    When using ``target_snr_db``, the noise RMS is computed from the
    signal's overall RMS:

        noise_rms = signal_rms / 10^(target_snr_db / 20)

    This value is then applied as a CONSTANT floor — same amplitude
    during rest and contraction.  During active periods the effective
    SNR matches ``target_snr_db``; during rest the noise dominates
    (matching real recordings).

    Parameters
    ----------
    clean_emg : ndarray, shape (n_samples,) or (n_samples, n_channels)
        Clean EMG signal.
    fs_hz : float
        Sampling rate (Hz).
    target_snr_db : float, default 20.0
        Target SNR during active contraction.  Used to auto-compute
        noise_rms from the signal.  Ignored if noise_rms is provided.
    noise_rms : float or None
        Explicit noise floor RMS.  If provided, overrides target_snr_db.
        Must be in the same units as the EMG signal.
    spectral_slope : float, default -0.5
        PSD slope.  See generate_realistic_noise().
    excess_kurtosis : float, default 3.0
        Target excess kurtosis.
    powerline_hz : float, default 50.0
        Powerline frequency.  Set to 0 to disable.
    powerline_amplitude : float, default 0.1
        Powerline amplitude as fraction of noise RMS.
    powerline_harmonic_ratios : list of float, optional
        Per-harmonic amplitude ratios relative to the fundamental.
    powerline_frequency_drift_hz : float, default 0.3
        STD of the slow drift of the mains instantaneous frequency.
        See :func:`generate_realistic_noise`.
    powerline_amplitude_modulation_depth : float, default 0.15
        Fractional AM depth on the powerline carriers. See
        :func:`generate_realistic_noise`.
    peak_hz : float, default 750.0
        Center of EMG-band spectral emphasis.
    analog_hpf_hz : float, default 10.0
        Analog HPF cutoff applied before powerline is mixed in.
    baseline_drift_rms_uv : float, default 0.0
        Target RMS of the band-limited 1/f^α drift (off by default).
        See :func:`generate_realistic_noise` for the paper-constrained
        spectral form and calibration guidance. Each channel receives
        an independent drift realisation; common-mode array drift
        is not modelled.
    baseline_drift_alpha : float, default 1.75
        Drift PSD slope α. Midpoint of the [1.5, 2.0] electrode-noise
        regime (Huigen 2002, Gondran 1996).
    baseline_drift_low_hz : float or None, default None
        Lower edge of the drift band (lowest nonzero FFT bin when None).
    baseline_drift_high_hz : float, default 1.0
        Upper edge of the drift band. Default scopes the knob to
        sub-1 Hz baseline wander; broadband movement artifacts are
        out of scope.
    rng : np.random.Generator or None
        Random generator.

    Returns
    -------
    ndarray
        Noisy EMG signal (same shape as input).
    """
    if rng is None:
        rng = np.random.default_rng()

    emg = np.asarray(clean_emg)
    is_1d = emg.ndim == 1

    if is_1d:
        emg = emg[:, np.newaxis]

    n_samples, n_channels = emg.shape

    # Compute noise RMS from signal if not explicitly provided
    if noise_rms is None:
        signal_rms = np.sqrt(np.mean(emg**2))
        if signal_rms > 0:
            noise_rms = signal_rms / (10 ** (target_snr_db / 20))
        else:
            noise_rms = 1e-6  # fallback for silent signal

    noisy = emg.copy()

    for ch in range(n_channels):
        noise = generate_realistic_noise(
            n_samples,
            fs_hz,
            noise_rms,
            spectral_slope=spectral_slope,
            excess_kurtosis=excess_kurtosis,
            powerline_hz=powerline_hz,
            powerline_amplitude=powerline_amplitude,
            powerline_harmonic_ratios=powerline_harmonic_ratios,
            powerline_frequency_drift_hz=powerline_frequency_drift_hz,
            powerline_amplitude_modulation_depth=powerline_amplitude_modulation_depth,
            peak_hz=peak_hz,
            analog_hpf_hz=analog_hpf_hz,
            baseline_drift_rms_uv=baseline_drift_rms_uv,
            baseline_drift_alpha=baseline_drift_alpha,
            baseline_drift_low_hz=baseline_drift_low_hz,
            baseline_drift_high_hz=baseline_drift_high_hz,
            rng=rng,
        )
        noisy[:, ch] += noise

    if is_1d:
        noisy = noisy[:, 0]

    return noisy


def estimate_noise_rms_from_recording(
    emg: np.ndarray,
    fs_hz: float,
    *,
    highpass_hz: float = 2000.0,
) -> float:
    """Estimate the noise floor RMS from a real recording.

    Uses the high-frequency content (above ``highpass_hz``) as a proxy
    for the noise floor, since real MU activity is mostly below 2 kHz.

    Parameters
    ----------
    emg : ndarray, shape (n_samples,) or (n_samples, n_channels)
        Raw EMG recording (can include both rest and active periods).
    fs_hz : float
        Sampling rate.
    highpass_hz : float, default 2000.0
        Frequency above which content is assumed to be noise.

    Returns
    -------
    float
        Estimated noise RMS (averaged across channels if multi-channel).
    """
    emg = np.asarray(emg)
    if emg.ndim == 1:
        emg = emg[:, np.newaxis]

    sos = sig.butter(4, highpass_hz, fs=fs_hz, btype="high", output="sos")
    rms_values = []
    for ch in range(emg.shape[1]):
        hf = sig.sosfilt(sos, emg[:, ch])
        rms_values.append(np.sqrt(np.mean(hf**2)))

    return float(np.mean(rms_values))


def _notch_powerline(
    x: np.ndarray, fs_hz: float, powerline_hz: float, q: float = 30.0
) -> np.ndarray:
    """Cascade of IIR notches at the powerline fundamental + harmonics.

    Strips 50/60 Hz and its 2nd–5th harmonics from a 1-D trace using
    zero-phase filtering, leaving the broadband colour intact. Same
    convention as the real-signal preprocessing used to derive the
    default Quattrocento profile.
    """
    out = x.astype(np.float64, copy=True)
    nyq = fs_hz / 2.0
    for k in range(1, 6):
        f = powerline_hz * k
        if f >= nyq or f <= 0:
            continue
        b, a = sig.iirnotch(f, q, fs_hz)
        out = sig.filtfilt(b, a, out)
    return out


def calibrate_realistic_noise_profile(
    real_signal_uv: np.ndarray,
    fs_hz: float,
    *,
    rest_mask: np.ndarray | None = None,
    powerline_hz: float = 50.0,
    notch_q: float = 30.0,
    hf_floor_hz: float = 2000.0,
    welch_nperseg: int | None = None,
) -> dict:
    """Estimate realistic-noise parameters from a real iEMG recording.

    Reproduces the calibration pipeline used to derive the default profile:

    1. Apply cascaded IIR notches at the powerline fundamental +
       harmonics (50/100/150/200/250 Hz for EU; 60/120/180/240 for US).
    2. Isolate a pure-noise residual:

       * if ``rest_mask`` is provided, take only the samples where the
         subject was at rest;
       * otherwise high-pass the trace above ``hf_floor_hz`` (default
         2 kHz) — a fallback that exploits the empirical fact
         that real iEMG noise is signal-independent.

    3. Compute the Welch PSD of the residual; fit the spectral slope
       above the in-band peak and locate ``peak_hz``.
    4. Standardise the residual and compute its excess kurtosis.
    5. Recover the powerline amplitude by comparing pre-notch vs
       post-notch RMS in a ±1 Hz band around each harmonic.

    The output dict is shaped to be unpacked straight into
    :func:`generate_realistic_noise` /
    :meth:`myogen.simulator.IntramuscularEMG.add_noise` so the simulator
    will reproduce the same noise statistics as the real recording.

    Parameters
    ----------
    real_signal_uv : ndarray, shape (n_samples,) or (n_samples, n_channels)
        Raw iEMG, **in µV**. Multi-channel arrays are flattened across
        channels for the statistics (each channel contributes equally).
    fs_hz : float
        Sampling rate of the recording.
    rest_mask : ndarray of bool, optional
        Per-sample mask, same length as the time axis. ``True`` marks
        rest samples used for the noise statistics. If ``None``, the
        high-frequency fallback is used (samples above ``hf_floor_hz``).
    powerline_hz : float, default 50.0
        Powerline frequency. Use ``60.0`` for North-American grids.
    notch_q : float, default 30.0
        Quality factor of the IIR notches.
    hf_floor_hz : float, default 2000.0
        High-pass cutoff for the HF-fallback noise estimate (used when
        ``rest_mask`` is ``None``). Should sit above the MUAP band.
    welch_nperseg : int, optional
        Welch segment length. ``None`` picks ``min(len(residual), 4 *
        int(fs_hz))`` so we get ~0.25 Hz frequency resolution.

    Returns
    -------
    dict
        Keys ready to splat into :func:`generate_realistic_noise`:

        * ``noise_floor_uv`` — RMS of the noise residual (µV).
        * ``spectral_slope`` — PSD slope in log-log space above the peak.
        * ``peak_hz`` — frequency of the EMG-band emphasis (Hz).
        * ``excess_kurtosis`` — Fisher kurtosis of the standardised residual.
        * ``powerline_hz`` — echo of the input (Hz).
        * ``powerline_amplitude`` — fundamental amplitude as a fraction
          of ``noise_floor_uv``.
        * ``powerline_harmonic_ratios`` — measured per-harmonic amplitude
          ratios relative to the fundamental (length 5).
    """
    sig_arr = np.asarray(real_signal_uv, dtype=np.float64)
    if sig_arr.ndim == 1:
        sig_arr = sig_arr[:, np.newaxis]
    n_samples, n_channels = sig_arr.shape

    if rest_mask is not None:
        rest_mask = np.asarray(rest_mask, dtype=bool)
        if rest_mask.shape != (n_samples,):
            raise ValueError(
                f"rest_mask shape {rest_mask.shape} does not match the "
                f"time axis length {n_samples}"
            )
        if not rest_mask.any():
            raise ValueError("rest_mask must contain at least one True sample")

    # 1. Notched copy (broadband-only signal, no powerline content).
    notched = np.empty_like(sig_arr)
    for ch in range(n_channels):
        notched[:, ch] = _notch_powerline(
            sig_arr[:, ch], fs_hz, powerline_hz, q=notch_q
        )

    # 2. Build the noise residual per channel.
    if rest_mask is not None:
        # Take only rest samples, but apply on the *notched* trace so
        # the residual is free of powerline content.
        residual_per_ch = [notched[rest_mask, ch] for ch in range(n_channels)]
    else:
        # HF fallback: high-pass above hf_floor_hz, full duration.
        sos_hp = sig.butter(4, hf_floor_hz, fs=fs_hz, btype="high", output="sos")
        residual_per_ch = [
            sig.sosfiltfilt(sos_hp, notched[:, ch]) for ch in range(n_channels)
        ]

    # 3. ``noise_floor_uv`` is deliberately derived from the integrated
    # median PSD rather than from per-channel rest RMS: real-rest RMS
    # includes sporadic artefacts (breathing, micro-MUs) that the median
    # Welch PSD downstream suppresses, which keeps the calibration and the
    # overlay on the same statistic.

    # 4. Per-channel Welch PSDs averaged before slope/peak extraction —
    # pooling residuals across channels with different scales / artifact
    # loads distorts the pooled distribution and miscomputes statistics
    # like kurtosis. Per-channel + median aggregation is robust against
    # a single bad channel.
    nperseg = welch_nperseg
    if nperseg is None:
        nperseg = min(min(len(r) for r in residual_per_ch), int(4 * fs_hz))
    nperseg = max(64, nperseg)
    psd_stack = []
    for r in residual_per_ch:
        freqs, psd_ch = sig.welch(
            r,
            fs=fs_hz,
            nperseg=nperseg,
            noverlap=int(nperseg * 0.75),
            average="median",
        )
        psd_stack.append(psd_ch)
    psd = np.median(np.stack(psd_stack, axis=0), axis=0)

    # Noise floor = sqrt of total median PSD power. By construction,
    # generating a noise trace with this RMS will produce a Welch
    # median PSD whose integral matches the real one.
    df_psd = float(freqs[1] - freqs[0])
    noise_floor_uv = float(np.sqrt(max(np.sum(psd) * df_psd, 0.0)))

    band_lo, band_hi = 100.0, 2000.0
    in_band = (freqs >= band_lo) & (freqs <= band_hi)

    def _measured_slope(spectrum: np.ndarray, freqs_axis: np.ndarray, peak_hz_local: float) -> float:
        """Linear fit of log10(PSD) above the peak."""
        fit_lo_l = peak_hz_local * 1.2
        fit_hi_l = fs_hz / 2.0 * 0.9
        fmask = (freqs_axis >= fit_lo_l) & (freqs_axis <= fit_hi_l) & (spectrum > 0)
        if np.count_nonzero(fmask) < 4:
            return float("nan")
        s, _ = np.polyfit(
            np.log10(freqs_axis[fmask]), np.log10(spectrum[fmask]), 1
        )
        return float(s)

    if not np.any(in_band):
        peak_hz = 1000.0
        spectral_slope = -0.5
    else:
        peak_idx_local = int(np.argmax(psd[in_band]))
        peak_hz = float(freqs[in_band][peak_idx_local])
        target_slope = _measured_slope(psd, freqs, peak_hz)
        if not np.isfinite(target_slope):
            spectral_slope = -0.5
        else:
            # The `spectral_slope` knob feeds the colored-noise base
            # BEFORE _emg_band_shape blends in an additional bandpass
            # emphasis, so the input slope is not the output slope. We
            # back-solve by running a couple of probe simulations: fit
            # a linear model output_slope ≈ a * input_slope + b on two
            # probe runs, then invert.
            probe_n = min(int(fs_hz * 8), n_samples)  # 8 s probes
            probe_rng = np.random.default_rng(0)
            probe_in = [-0.2, -1.2]
            probe_out: list[float] = []
            for slope_in in probe_in:
                probe_noise = generate_realistic_noise(
                    probe_n,
                    fs_hz,
                    noise_rms=1.0,
                    spectral_slope=slope_in,
                    excess_kurtosis=0.0,
                    powerline_hz=0.0,
                    powerline_amplitude=0.0,
                    peak_hz=peak_hz,
                    rng=probe_rng,
                )
                f_p, psd_p = sig.welch(
                    probe_noise,
                    fs=fs_hz,
                    nperseg=min(probe_n, int(2 * fs_hz)),
                    noverlap=int(min(probe_n, int(2 * fs_hz)) * 0.75),
                    average="median",
                )
                probe_out.append(_measured_slope(psd_p, f_p, peak_hz))
            if all(np.isfinite(probe_out)):
                a = (probe_out[1] - probe_out[0]) / (probe_in[1] - probe_in[0])
                b = probe_out[0] - a * probe_in[0]
                if abs(a) > 1e-6:
                    spectral_slope = float((target_slope - b) / a)
                else:
                    spectral_slope = float(target_slope)
            else:
                spectral_slope = float(target_slope)

    # 5b. Analog HPF corner — detect from the low-frequency PSD shape.
    # Estimate an in-band reference level (median PSD in 30 Hz-min(peak/2,
    # 200 Hz)), then walk from DC upward to the first bin where the PSD
    # crosses (reference − 3 dB). That bin is the −3 dB corner frequency
    # of whatever HPF (analog + digital) shaped the input.
    ref_lo, ref_hi = 30.0, min(max(peak_hz / 2.0, 60.0), 200.0)
    ref_mask = (freqs >= ref_lo) & (freqs <= ref_hi) & (psd > 0)
    if np.any(ref_mask):
        ref_psd = float(np.median(psd[ref_mask]))
        threshold = ref_psd * 0.5  # -3 dB in power
        rising_mask = (freqs > 0) & (freqs < ref_lo) & (psd >= threshold)
        if rising_mask.any():
            analog_hpf_hz = float(freqs[rising_mask][0])
        else:
            analog_hpf_hz = 10.0
    else:
        analog_hpf_hz = 10.0

    # 5. Excess kurtosis per channel → median (robust to outlier channels).
    kurt_per_ch = []
    for r in residual_per_ch:
        centered = r - np.mean(r)
        std = np.std(centered)
        if std > 0:
            m4 = np.mean(centered ** 4) / (std ** 4)
            kurt_per_ch.append(float(m4 - 3.0))
    excess_kurtosis = float(np.median(kurt_per_ch)) if kurt_per_ch else 0.0

    # 6. Powerline amplitude + harmonic ratios — directly from the
    # *un-notched* signal's PSD. For each harmonic, measure the excess
    # power inside a ±2 Hz band centred on the line over the broadband
    # floor estimated from nearby frequencies. Excess power equates to
    # ``amplitude² / 2`` for a sinusoid, so amplitude = sqrt(2 * excess).
    # This avoids the bias from the iirnotch's finite passband width
    # that the previous variance-difference approach inherited.
    raw_psd_stack = []
    for ch in range(n_channels):
        f_raw, p_raw_ch = sig.welch(
            sig_arr[:, ch],
            fs=fs_hz,
            nperseg=nperseg,
            noverlap=int(nperseg * 0.75),
            average="median",
        )
        raw_psd_stack.append(p_raw_ch)
    raw_psd = np.median(np.stack(raw_psd_stack, axis=0), axis=0)
    df_raw = float(f_raw[1] - f_raw[0])

    harmonic_amps = []
    nyq = fs_hz / 2.0
    for k in range(1, 6):
        f = powerline_hz * k
        if f >= nyq:
            harmonic_amps.append(0.0)
            continue
        peak_lo, peak_hi = f - 2.0, f + 2.0
        peak_mask = (f_raw >= peak_lo) & (f_raw <= peak_hi)
        peak_power = float(np.sum(raw_psd[peak_mask]) * df_raw)
        # Neighbouring band for the broadband floor (exclude the peak
        # window itself).
        nbr_lo1, nbr_hi1 = max(f - 8.0, 0.5), f - 2.0
        nbr_lo2, nbr_hi2 = f + 2.0, min(f + 8.0, nyq * 0.999)
        nbr_mask = (
            ((f_raw >= nbr_lo1) & (f_raw <= nbr_hi1))
            | ((f_raw >= nbr_lo2) & (f_raw <= nbr_hi2))
        )
        if not np.any(nbr_mask):
            harmonic_amps.append(0.0)
            continue
        floor_density = float(np.median(raw_psd[nbr_mask]))
        floor_in_peak_band = floor_density * (peak_hi - peak_lo)
        excess = max(peak_power - floor_in_peak_band, 0.0)
        harmonic_amps.append(float(np.sqrt(2.0 * excess)))
    fundamental_amp = harmonic_amps[0]
    if fundamental_amp > 0 and noise_floor_uv > 0:
        powerline_amplitude = float(fundamental_amp / noise_floor_uv)
        powerline_harmonic_ratios = [
            float(a / fundamental_amp) for a in harmonic_amps
        ]
    else:
        powerline_amplitude = 0.0
        powerline_harmonic_ratios = [1.0, 0.0, 0.0, 0.0, 0.0]

    return {
        "noise_floor_uv": noise_floor_uv,
        "spectral_slope": spectral_slope,
        "peak_hz": peak_hz,
        "excess_kurtosis": excess_kurtosis,
        "powerline_hz": float(powerline_hz),
        "powerline_amplitude": powerline_amplitude,
        "powerline_harmonic_ratios": powerline_harmonic_ratios,
        "analog_hpf_hz": float(analog_hpf_hz),
    }


def calibrate_baseline_drift_profile(
    real_signal_uv: np.ndarray,
    fs_hz: float,
    *,
    rest_mask: np.ndarray | None = None,
    band: tuple[float | None, float] = (None, 1.0),
) -> dict:
    """Estimate baseline-drift parameters from a real recording.

    Fits the band-limited 1/f^α drift model
    (see :func:`generate_realistic_noise`) to the low-frequency PSD
    of ``real_signal_uv`` over ``band = (low_hz, high_hz)``. Returns
    a dict with the four ``baseline_drift_*`` kwargs that
    :func:`generate_realistic_noise` and downstream APIs accept, so
    the user can splat it directly:

        >>> profile = calibrate_baseline_drift_profile(real_uv, fs_hz)
        >>> generate_realistic_noise(n, fs_hz, noise_rms=5.0, **profile)

    The amplitude is returned in the **same units as the input
    signal** (µV when the input is in µV); the function does *not*
    standardise the channels because that would erase the scale.

    .. note::
       The model's spectral form (PSD ∝ 1/f^α over a bounded band)
       is paper-constrained for electrode/interface noise (Huigen
       2002, Gondran 1996). The fitted parameters become validated
       only when this calibration is run on real recordings — they
       are not pre-baked physiological constants.

    Parameters
    ----------
    real_signal_uv : ndarray, shape (n_samples,) or (n_samples, n_channels)
        Raw real recording in µV. Multi-channel inputs are aggregated
        via a median Welch PSD (robust to one bad channel).
    fs_hz : float
        Sampling rate.
    rest_mask : ndarray of bool, optional
        Per-sample boolean mask; ``True`` marks rest samples to use
        for the fit. ``None`` uses the full trace.
    band : tuple of (float or None, float), default (None, 1.0)
        Fit band. The lower bound defaults to the lowest nonzero
        Welch bin available given the chosen segment length; the
        upper bound defaults to 1 Hz (sub-1 Hz baseline wander per
        Boyer 2023, DOI 10.3390/s23062927).

    Returns
    -------
    dict
        ``{"baseline_drift_rms_uv", "baseline_drift_alpha",
        "baseline_drift_low_hz", "baseline_drift_high_hz"}``.
    """
    band_low_arg, band_high = band
    if band_high <= 0:
        raise ValueError(f"band[1] (high_hz) must be > 0, got {band_high!r}")
    if band_low_arg is not None and band_low_arg <= 0:
        raise ValueError(f"band[0] (low_hz) must be > 0 or None, got {band_low_arg!r}")
    if band_low_arg is not None and band_low_arg >= band_high:
        raise ValueError(
            f"band[0] ({band_low_arg!r}) must be < band[1] ({band_high!r})"
        )

    sig_arr = np.asarray(real_signal_uv, dtype=np.float64)
    if sig_arr.ndim == 1:
        sig_arr = sig_arr[:, np.newaxis]
    n_total, n_channels = sig_arr.shape

    if rest_mask is not None:
        rest_mask = np.asarray(rest_mask, dtype=bool)
        if rest_mask.shape != (n_total,):
            raise ValueError(
                f"rest_mask shape {rest_mask.shape} does not match the "
                f"time axis length {n_total}"
            )
        if not rest_mask.any():
            raise ValueError("rest_mask must contain at least one True sample")
        per_ch = [sig_arr[rest_mask, ch] for ch in range(n_channels)]
    else:
        per_ch = [sig_arr[:, ch] for ch in range(n_channels)]

    n_rest = min(len(x) for x in per_ch)
    if n_rest < 16:
        raise ValueError(
            f"need at least 16 samples per channel after the rest mask, got {n_rest}"
        )

    # Aim for ~0.125 Hz frequency resolution at the default sub-1 Hz
    # band; clamp to what the data supports. If the user supplied a
    # tighter ``band_low``, expand the segment so several bins land
    # above it when the recording is long enough.
    target_nperseg = max(int(8 * fs_hz), 1024)
    if band_low_arg is not None:
        target_nperseg = max(target_nperseg, int(4.0 * fs_hz / band_low_arg))
    nperseg = min(n_rest, target_nperseg)
    nperseg = max(64, nperseg)
    noverlap = int(nperseg * 0.5)

    psd_stack = []
    for r in per_ch:
        freqs_ref, psd_ch = sig.welch(
            r, fs=fs_hz, nperseg=nperseg, noverlap=noverlap, average="median",
        )
        psd_stack.append(psd_ch)
    psd = np.median(np.stack(psd_stack, axis=0), axis=0)
    df = float(freqs_ref[1] - freqs_ref[0])

    if band_low_arg is None:
        band_low = float(freqs_ref[1])  # lowest nonzero Welch bin
    else:
        band_low = float(band_low_arg)
    if band_high >= fs_hz / 2.0:
        raise ValueError(
            f"band[1] ({band_high!r}) must be < Nyquist (fs_hz/2 = "
            f"{fs_hz / 2.0:.3f})"
        )
    if band_low >= band_high:
        raise ValueError(
            f"resolved band_low ({band_low:.4f} Hz) >= band_high "
            f"({band_high:.4f} Hz). Recording may be too short for "
            "the requested band."
        )

    fit_mask = (freqs_ref >= band_low) & (freqs_ref <= band_high) & (psd > 0)
    if np.count_nonzero(fit_mask) < 3:
        raise ValueError(
            "fewer than 3 positive PSD bins in the calibration band; "
            "increase the recording length or widen the band."
        )

    # alpha = -slope of log10(PSD) vs log10(f) over the band.
    log_f = np.log10(freqs_ref[fit_mask])
    log_p = np.log10(psd[fit_mask])
    slope, _ = np.polyfit(log_f, log_p, 1)
    alpha = float(-slope)

    # RMS = sqrt(sum(PSD * df)) within the band, preserving µV units.
    band_power = float(np.sum(psd[fit_mask]) * df)
    rms_uv = float(np.sqrt(max(band_power, 0.0)))

    return {
        "baseline_drift_rms_uv": rms_uv,
        "baseline_drift_alpha": alpha,
        "baseline_drift_low_hz": band_low,
        "baseline_drift_high_hz": float(band_high),
    }


def tune_noise_profile_with_optuna(
    real_signal_uv: np.ndarray,
    fs_hz: float,
    initial_profile: dict,
    *,
    rest_mask: np.ndarray | None = None,
    n_trials: int = 100,
    psd_freq_band_hz: tuple[float, float] = (2.0, 2000.0),
    seed: int = 0,
    show_progress_bar: bool = False,
) -> dict:
    """Refine a noise profile with Optuna to minimise PSD distance vs real.

    Closed-form calibration (:func:`calibrate_realistic_noise_profile`)
    matches the first few moments and spectral summary statistics but
    can leave residual shape mismatch — most visibly the low-frequency
    rolloff curvature and the line-peak shape. This function takes the
    closed-form profile as a starting point and runs an Optuna TPE
    search around it, minimising the **L1 distance between log-PSDs**
    of the simulated and real signals in ``psd_freq_band_hz``.

    Cost is fast: each trial generates ~30 s of simulated noise and
    computes one Welch PSD. 100 trials ≈ a few seconds on a laptop.

    Parameters
    ----------
    real_signal_uv : ndarray, shape (n_samples,) or (n_samples, n_channels)
        Real iEMG signal in µV (multi-channel will be aggregated as
        median PSD across channels).
    fs_hz : float
        Sampling rate.
    initial_profile : dict
        Output of :func:`calibrate_realistic_noise_profile`. Used as
        starting point + to define the search bounds.
    rest_mask : ndarray of bool, optional
        Per-sample rest mask. If provided, the reference PSD is built
        from the rest samples only — same convention as the closed-form
        calibrator.
    n_trials : int, default 100
        Number of Optuna trials.
    psd_freq_band_hz : tuple of float, default (2.0, 2000.0)
        Frequency band over which the log-PSD distance is computed.
        Restrict to the band you actually care about — typically the
        EMG band, leaving anti-alias rolloff out.
    seed : int, default 0
        Optuna sampler seed.
    show_progress_bar : bool, default False
        Pass through to ``optuna.study.optimize`` for a tqdm bar.

    Returns
    -------
    dict
        Tuned profile in the same shape as ``initial_profile``, with
        an extra key ``"_optuna_loss"`` reporting the final L1 log-PSD
        distance.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # 1. Build the real reference PSD using the same conventions as the
    # calibrator (rest-mask + notch + median across channels).
    sig_arr = np.asarray(real_signal_uv, dtype=np.float64)
    if sig_arr.ndim == 1:
        sig_arr = sig_arr[:, np.newaxis]
    n_samples, n_channels = sig_arr.shape

    powerline_hz = float(initial_profile["powerline_hz"])
    notched = np.empty_like(sig_arr)
    for ch in range(n_channels):
        notched[:, ch] = _notch_powerline(sig_arr[:, ch], fs_hz, powerline_hz)

    if rest_mask is not None:
        rest_mask = np.asarray(rest_mask, dtype=bool)
        residual_per_ch = [notched[rest_mask, ch] for ch in range(n_channels)]
    else:
        sos_hp = sig.butter(4, 2000.0, fs=fs_hz, btype="high", output="sos")
        residual_per_ch = [
            sig.sosfiltfilt(sos_hp, notched[:, ch]) for ch in range(n_channels)
        ]

    nperseg = min(min(len(r) for r in residual_per_ch), int(4 * fs_hz))
    nperseg = max(64, nperseg)
    noverlap = int(nperseg * 0.75)

    # Frequency axis for the band mask — identical grid for every Welch
    # PSD computed below, so one call on the first residual suffices.
    freqs_ref = sig.welch(
        residual_per_ch[0], fs=fs_hz, nperseg=nperseg, noverlap=noverlap,
        average="median",
    )[0]

    # Reference PSD on the UN-notched signal, so the powerline lines
    # contribute to the fit during the powerline-amplitude trials.
    raw_stack = []
    for ch in range(n_channels):
        if rest_mask is not None:
            r = sig_arr[rest_mask, ch]
        else:
            r = sig_arr[:, ch]
        _, p = sig.welch(
            r, fs=fs_hz, nperseg=nperseg, noverlap=noverlap, average="median"
        )
        raw_stack.append(p)
    raw_real_psd = np.median(np.stack(raw_stack, axis=0), axis=0)
    log_real_raw = np.log10(np.maximum(raw_real_psd, 1e-30))

    band_mask = (freqs_ref >= psd_freq_band_hz[0]) & (freqs_ref <= psd_freq_band_hz[1])

    # 2. Search bounds — narrow ranges around the initial profile.
    init = initial_profile
    sim_len = int(min(n_samples, fs_hz * 30))  # 30 s probes — fast & enough resolution
    base_rng = np.random.default_rng(seed)

    def _objective(trial: "optuna.trial.Trial") -> float:
        noise_floor_uv = trial.suggest_float(
            "noise_floor_uv",
            max(init["noise_floor_uv"] * 0.5, 0.1),
            init["noise_floor_uv"] * 2.0,
        )
        spectral_slope = trial.suggest_float(
            "spectral_slope",
            init["spectral_slope"] - 0.8,
            init["spectral_slope"] + 0.8,
        )
        peak_hz = trial.suggest_float(
            "peak_hz",
            max(init["peak_hz"] * 0.5, 100.0),
            min(init["peak_hz"] * 2.0, fs_hz / 2.0 * 0.9),
        )
        excess_kurtosis = trial.suggest_float(
            "excess_kurtosis",
            max(init["excess_kurtosis"] - 1.5, 0.0),
            init["excess_kurtosis"] + 3.0,
        )
        powerline_amplitude = trial.suggest_float(
            "powerline_amplitude",
            max(init["powerline_amplitude"] * 0.3, 0.0),
            init["powerline_amplitude"] * 3.0 + 0.05,
        )
        analog_hpf_hz = trial.suggest_float(
            "analog_hpf_hz",
            max(init["analog_hpf_hz"] * 0.3, 1.0),
            init["analog_hpf_hz"] * 3.0,
        )

        # Fixed: harmonic ratios + powerline_hz (these are well-determined).
        trial_noise = generate_realistic_noise(
            sim_len,
            fs_hz,
            noise_rms=noise_floor_uv,
            spectral_slope=spectral_slope,
            excess_kurtosis=excess_kurtosis,
            powerline_hz=powerline_hz,
            powerline_amplitude=powerline_amplitude,
            powerline_harmonic_ratios=init["powerline_harmonic_ratios"],
            peak_hz=peak_hz,
            analog_hpf_hz=analog_hpf_hz,
            rng=np.random.default_rng(base_rng.integers(0, 2**31 - 1)),
        )
        _, trial_psd = sig.welch(
            trial_noise, fs=fs_hz, nperseg=nperseg, noverlap=noverlap, average="median"
        )
        log_trial = np.log10(np.maximum(trial_psd, 1e-30))
        # Compare against the un-notched real PSD so the powerline lines
        # contribute to the fit.
        loss = float(np.mean(np.abs(log_trial[band_mask] - log_real_raw[band_mask])))
        return loss

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    # Seed the search with the closed-form result so the first trial
    # is already a good baseline.
    study.enqueue_trial(
        {
            "noise_floor_uv": float(init["noise_floor_uv"]),
            "spectral_slope": float(init["spectral_slope"]),
            "peak_hz": float(init["peak_hz"]),
            "excess_kurtosis": float(init["excess_kurtosis"]),
            "powerline_amplitude": float(init["powerline_amplitude"]),
            "analog_hpf_hz": float(init["analog_hpf_hz"]),
        }
    )
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=show_progress_bar)

    best = study.best_params
    tuned = {
        "noise_floor_uv": float(best["noise_floor_uv"]),
        "spectral_slope": float(best["spectral_slope"]),
        "peak_hz": float(best["peak_hz"]),
        "excess_kurtosis": float(best["excess_kurtosis"]),
        "powerline_hz": float(powerline_hz),
        "powerline_amplitude": float(best["powerline_amplitude"]),
        "powerline_harmonic_ratios": list(init["powerline_harmonic_ratios"]),
        "analog_hpf_hz": float(best["analog_hpf_hz"]),
        "_optuna_loss": float(study.best_value),
    }
    return tuned
