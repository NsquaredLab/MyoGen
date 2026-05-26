import numpy as np

from myogen.simulator.core.emg.intramuscular.bioelectric import (
    hr_shift_template,
    shift_padding,
)
from myogen.simulator.core.emg.intramuscular.motor_unit_sim import MotorUnitSim


def test_shift_padding_right_shift_preserves_tail():
    """A positive (right) shift must zero only the vacated head, not the tail.

    Regression for a mis-port of the MATLAB reference that also zeroed ``vec[-sh:]``,
    destroying valid signal samples at the tail of shifted SFAPs/IAPs.
    """
    vec = np.arange(1, 9, dtype=float)  # [1, 2, ..., 8]
    out = shift_padding(vec.copy(), 2, axis=0)
    np.testing.assert_array_equal(
        out, np.array([0, 0, 1, 2, 3, 4, 5, 6], dtype=float)
    )


def test_shift_padding_left_shift_zeros_tail():
    """A negative (left) shift zeros only the vacated tail."""
    vec = np.arange(1, 9, dtype=float)
    out = shift_padding(vec.copy(), -2, axis=0)
    np.testing.assert_array_equal(
        out, np.array([3, 4, 5, 6, 7, 8, 0, 0], dtype=float)
    )


def test_shift_padding_zero_shift_is_identity():
    vec = np.arange(1, 9, dtype=float)
    out = shift_padding(vec.copy(), 0, axis=0)
    np.testing.assert_array_equal(out, vec)


def _build_impulse_sim_for_delay(delay_samples: float) -> MotorUnitSim:
    """Build a single-fiber MotorUnitSim primed to test shift_sfaps.

    The returned sim has one fiber, one electrode, an impulse SFAP at a
    known index, and a pre-computed mnap_delays / Npt / sfaps such that
    shift_sfaps(dt=1.0) will shift the impulse by `delay_samples`.
    """
    sim = MotorUnitSim(
        muscle_fiber_centers__mm=np.array([[0.0, 0.0]]),
        muscle_length__mm=100.0,
        muscle_fiber_diameters__mm=np.array([0.05]),
        muscle_fiber_conduction_velocity__mm_per_s=np.array([4000.0]),
        nominal_center__mm=np.array([0.0, 0.0]),
    )
    n_samples = 256
    impulse_index = 50
    sfap = np.zeros(n_samples, dtype=float)
    sfap[impulse_index] = 1.0
    sim.sfaps = sfap.reshape(n_samples, 1, 1)  # (time, electrode, fiber)
    sim.Npt = 1
    # Bypass calc_mnap_delays by providing mnap_delays and a matching
    # nerve_paths/conduction_velocities pair so that calc_mnap_delays would
    # reproduce the same value (delay_samples * dt with dt=1.0).
    sim.nerve_paths = np.array([[delay_samples, 0.0]])
    sim.neuromuscular_junction_conduction_velocities__mm_per_s = [1.0, 1.0]
    return sim, impulse_index


def test_shift_sfaps_applies_delay_with_correct_sign_convention():
    """shift_sfaps must delay the impulse FORWARD in time by delay_samples.

    Regression for a sign mismatch between shift_padding (integer shift,
    forward in time) and hr_shift_template (DFT phase factor, backward in
    time). The fractional component must be negated so the composed shift
    equals the requested NMJ propagation delay.
    """
    delay_samples = 10.3
    sim, impulse_index = _build_impulse_sim_for_delay(delay_samples)

    sim.shift_sfaps(dt=1.0)

    shifted = sim.sfaps[:, 0, 0]
    # Predicted impulse location (forward shift) at index 60.3. After the
    # composition, energy should be concentrated near index 60-61, with the
    # peak/centroid matching the requested delay.
    target = impulse_index + delay_samples
    indices = np.arange(shifted.size, dtype=float)
    # Sinc interpolation spreads the impulse over many bins; restrict the
    # centroid window to a ±5 sample neighbourhood around the prediction.
    window_lo = int(np.floor(target)) - 5
    window_hi = int(np.ceil(target)) + 5
    window = slice(window_lo, window_hi + 1)
    weights = shifted[window]
    centroid = float(np.sum(indices[window] * weights) / np.sum(weights))
    # Tolerance accounts for FFT-based sinc interpolation precision (truncation
    # at window edges). Pre-fix, this centroid was off by ~1 sample (delta on
    # the order of 2*fractional_delay); post-fix it matches the target to
    # within ~1e-4 samples.
    assert abs(centroid - target) < 1e-3, (
        f"shift_sfaps centroid {centroid:.6f} did not match expected delay "
        f"target {target:.6f} (delta={centroid - target:.3e})"
    )


def test_shift_sfaps_integer_only_delay_matches_shift_padding():
    """A whole-sample delay must lie exactly at the predicted index."""
    delay_samples = 12.0
    sim, impulse_index = _build_impulse_sim_for_delay(delay_samples)

    sim.shift_sfaps(dt=1.0)

    shifted = sim.sfaps[:, 0, 0]
    target_index = impulse_index + int(delay_samples)
    # fractional_delay is exactly 0.0 so hr_shift_template is a no-op modulo
    # floating-point round-off through the FFT/IFFT path.
    assert abs(shifted[target_index] - 1.0) < 1e-12
    # All other samples should be ~zero (no fractional fan-out).
    mask = np.ones_like(shifted, dtype=bool)
    mask[target_index] = False
    assert np.max(np.abs(shifted[mask])) < 1e-12


def test_hr_shift_template_dft_sign_convention():
    """Document the hr_shift_template sign convention so shift_sfaps is safe.

    hr_shift_template uses ``exp(+1j * 2π * delay * k / N)`` which is the
    DFT pair for ``x[n + delay]`` — i.e. a positive ``delay`` advances the
    waveform BACKWARD in time. ``shift_padding`` instead uses the natural
    forward convention. To compose them safely the fractional argument
    must be negated, which is what ``MotorUnitSim.shift_sfaps`` does.
    """
    n = 64
    x = np.zeros(n)
    x[20] = 1.0
    shifted = hr_shift_template(x, 1.0)
    # Peak should land at index 19 (advanced backwards), not 21.
    assert int(np.argmax(shifted)) == 19
