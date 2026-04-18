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
