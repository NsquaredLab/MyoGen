"""Regression tests for the RNG architecture (R1 Code Review 1 and 2).

These tests verify that ``set_random_seed`` propagates correctly through the
accessor pattern and that no module captures a stale reference to the global
RNG or to the deprecated module-level ``SEED`` constant.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import myogen
from myogen import get_random_generator, get_random_seed, set_random_seed


def _draw(n: int = 8) -> np.ndarray:
    return get_random_generator().integers(0, 10**9, n)


def test_same_seed_yields_same_draws():
    set_random_seed(12345)
    first = _draw()
    set_random_seed(12345)
    second = _draw()
    assert np.array_equal(first, second)


def test_different_seed_yields_different_draws():
    set_random_seed(1)
    first = _draw()
    set_random_seed(2)
    second = _draw()
    assert not np.array_equal(first, second)


def test_get_random_seed_reflects_latest_set_call():
    set_random_seed(777)
    assert get_random_seed() == 777
    set_random_seed(999)
    assert get_random_seed() == 999


def test_accessor_returns_current_generator_after_rebind():
    """A caller that holds ``get_random_generator`` (the function, not the generator) must see the new RNG after each ``set_random_seed`` call — this is the regression for CR1."""
    set_random_seed(100)
    before = _draw()
    set_random_seed(200)
    after = _draw()
    assert not np.array_equal(before, after)

    # Restoring the original seed must reproduce the first sequence.
    set_random_seed(100)
    restored = _draw()
    assert np.array_equal(before, restored)


def test_derived_seed_pattern_respects_set_random_seed():
    """Regression for CR2: code that derives sub-seeds via ``get_random_seed() + offset`` must observe new values after ``set_random_seed``."""
    set_random_seed(10)
    derived_a = get_random_seed() + 3 * 7
    set_random_seed(20)
    derived_b = get_random_seed() + 3 * 7
    assert derived_a != derived_b


def test_deprecated_RANDOM_GENERATOR_emits_warning_and_returns_current():
    set_random_seed(42)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gen_via_deprecated = myogen.RANDOM_GENERATOR
    assert any(
        issubclass(w.category, DeprecationWarning) for w in caught
    ), "accessing myogen.RANDOM_GENERATOR must emit a DeprecationWarning"
    assert gen_via_deprecated is get_random_generator(), (
        "deprecated attribute must return the current global generator"
    )

    # Confirm it reflects later ``set_random_seed`` calls.
    set_random_seed(43)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        gen_again = myogen.RANDOM_GENERATOR
    assert gen_again is get_random_generator()
    assert gen_again is not gen_via_deprecated


def test_deprecated_SEED_emits_warning_and_returns_current_seed():
    set_random_seed(5150)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        seed_via_deprecated = myogen.SEED
    assert any(
        issubclass(w.category, DeprecationWarning) for w in caught
    ), "accessing myogen.SEED must emit a DeprecationWarning"
    assert seed_via_deprecated == 5150


def test_derive_subseed_avoids_old_collision_pattern():
    """Regression for the earlier ``SEED + (a+1)*(b+1)`` collision: (0, 5) and (1, 2) both produced +6. The new SeedSequence-based mixing is uint32 and therefore not literally collision-free, but it must not reproduce the swapped-factor collision, and a 16x16 grid must be collision-free in practice."""
    set_random_seed(42)
    assert myogen.derive_subseed(0, 5) != myogen.derive_subseed(1, 2)
    assert myogen.derive_subseed(1, 2) != myogen.derive_subseed(2, 1)

    subseeds = {
        (a, b): myogen.derive_subseed(a, b)
        for a in range(16)
        for b in range(16)
    }
    assert len(set(subseeds.values())) == len(subseeds), (
        "derive_subseed must yield a unique value across a 16x16 label grid"
    )


def test_derive_subseed_tracks_set_random_seed():
    set_random_seed(1)
    sub_a = myogen.derive_subseed(3, 7)
    set_random_seed(2)
    sub_b = myogen.derive_subseed(3, 7)
    assert sub_a != sub_b, (
        "derive_subseed must change when the global seed changes"
    )


def test_modules_that_use_accessor_are_not_stale_after_seed_change():
    """End-to-end: reimport a module that calls ``get_random_generator()`` and confirm its RNG draws track the current seed."""
    from myogen.simulator.core.muscle import muscle as muscle_module

    # The muscle module imports ``get_random_generator`` at top level; after
    # ``set_random_seed``, any call it makes to ``get_random_generator()`` must
    # return the new generator. Verify by calling the function directly.
    set_random_seed(11)
    gen_before = muscle_module.get_random_generator()
    set_random_seed(22)
    gen_after = muscle_module.get_random_generator()
    assert gen_before is not gen_after
    assert gen_after is get_random_generator()


# ---------------------------------------------------------------------------
# Regression tests for higher-level pipeline reproducibility
# ---------------------------------------------------------------------------

import quantities as pq  # noqa: E402  (placed after stdlib/third-party block above)
from myogen import simulator  # noqa: E402


def _small_recruitment_thresholds():
    """4-MU pool with a physiological range, deterministic (no RNG used)."""
    return simulator.RecruitmentThresholds(N=4, recruitment_range__ratio=30)[0]


def _build_small_muscle(seed: int):
    """Build a minimal Muscle (radius 2 mm, 50 fibers/mm², grid 32) under *seed*."""
    set_random_seed(seed)
    return simulator.Muscle(
        recruitment_thresholds=_small_recruitment_thresholds(),
        radius__mm=2.0 * pq.mm,
        radius_bone__mm=0.5 * pq.mm,
        fiber_density__fibers_per_mm2=50 * pq.mm**-2,
        grid_resolution=32,
        autorun=True,
    )


def _small_electrode():
    return simulator.IntramuscularElectrodeArray(
        num_electrodes=2,
        inter_electrode_distance__mm=0.5 * pq.mm,
        position__mm=(0.0 * pq.mm, 0.0 * pq.mm, 15.0 * pq.mm),
    )


def _build_iemg_with_muaps(seed: int):
    """Build a muscle + iEMG simulator and compute MUAPs for MU 0 only."""
    muscle = _build_small_muscle(seed)
    iemg = simulator.IntramuscularEMG(
        muscle_model=muscle,
        electrode_array=_small_electrode(),
        MUs_to_simulate=[0],
        sampling_frequency__Hz=10240.0 * pq.Hz,
    )
    set_random_seed(seed)
    iemg.simulate_muaps(n_jobs=1, verbose=False)
    return iemg


def _make_spike_block():
    """Minimal neo Block with a single spike at 20 ms, compatible with 10240 Hz iEMG."""
    from neo import Block, Segment, SpikeTrain

    block = Block()
    segment = Segment(name="Pool_0")
    spiketrain = SpikeTrain(
        [0.020] * pq.s,
        t_start=0 * pq.s,
        t_stop=0.05 * pq.s,
    )
    spiketrain.sampling_period = (1.0 / 10240.0) * pq.s
    segment.spiketrains.append(spiketrain)
    block.segments.append(segment)
    return block


def test_muscle_fiber_placement_is_reproducible_under_seed():
    """Muscle fiber placement and MU assignment must be byte-identical under the same seed.

    Regression for any un-seeded RNG path inside generate_muscle_fiber_centers
    or assign_mfs2mns. Two builds under seed 7 must yield identical
    muscle_fiber_centers__mm arrays and assignment arrays. A build
    under seed 8 must differ (sanity check that the seed is actually observed).
    """
    muscle_a = _build_small_muscle(7)
    muscle_b = _build_small_muscle(7)

    assert np.array_equal(
        muscle_a.muscle_fiber_centers__mm.magnitude,
        muscle_b.muscle_fiber_centers__mm.magnitude,
    ), "muscle_fiber_centers__mm must be identical under the same seed"

    assert np.array_equal(
        muscle_a.assignment,
        muscle_b.assignment,
    ), "MF-to-MU assignment must be identical under the same seed"

    assert np.array_equal(
        muscle_a.resulting_number_of_innervated_fibers,
        muscle_b.resulting_number_of_innervated_fibers,
    ), "resulting_number_of_innervated_fibers must be identical under the same seed"

    # Sanity: a different seed must produce a different result.
    muscle_c = _build_small_muscle(8)
    assert not np.array_equal(
        muscle_a.assignment,
        muscle_c.assignment,
    ), "assignment must differ under a different seed (sanity check)"


def test_intramuscular_muap_block_is_reproducible_under_seed():
    """MU 0 MUAP waveform from IntramuscularEMG.simulate_muaps must be
    byte-identical between two independent runs under the same seed.

    Guards against non-determinism in the bioelectric computation chain
    (fiber conductance look-up, SFAP summation) when run with n_jobs=1.
    """
    iemg_a = _build_iemg_with_muaps(7)
    iemg_b = _build_iemg_with_muaps(7)

    mag_a = iemg_a.muaps__Block.segments[0].analogsignals[0].magnitude
    mag_b = iemg_b.muaps__Block.segments[0].analogsignals[0].magnitude

    assert np.array_equal(mag_a, mag_b), (
        "MUAP waveform (MU 0) must be byte-identical across two seeded runs; "
        "max abs diff = " + str(np.max(np.abs(mag_a - mag_b)))
    )


def test_intramuscular_emg_noise_is_reproducible_under_seed():
    """Noise added by IntramuscularEMG.add_noise must be byte-identical
    when the global seed is reset to the same value before each call.

    Regression for any unseeded np.random or random call inside
    add_noise / the noise-generation helpers.
    """
    spike_block = _make_spike_block()

    iemg_a = _build_iemg_with_muaps(7)
    iemg_a.simulate_intramuscular_emg(spike_train__Block=spike_block, verbose=False)
    set_random_seed(7)
    iemg_a.add_noise(snr__dB=20)
    noisy_a = iemg_a.noisy_intramuscular_emg__Block.segments[0].analogsignals[0].magnitude

    iemg_b = _build_iemg_with_muaps(7)
    iemg_b.simulate_intramuscular_emg(spike_train__Block=spike_block, verbose=False)
    set_random_seed(7)
    iemg_b.add_noise(snr__dB=20)
    noisy_b = iemg_b.noisy_intramuscular_emg__Block.segments[0].analogsignals[0].magnitude

    assert np.array_equal(noisy_a, noisy_b), (
        "noisy_intramuscular_emg__Block must be byte-identical when the seed is "
        "reset to the same value before add_noise(); "
        "max abs diff = " + str(np.max(np.abs(noisy_a - noisy_b)))
    )
