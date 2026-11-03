"""
Differentiable modified Bessel functions I_n(x), K_n(x) for integer order in JAX.

JAX's ``jax.scipy.special`` ships only ``i0/i1/i0e/i1e`` — no ``k0/k1`` and no
arbitrary-order ``iv/kv``. The surface-EMG volume-conductor model
(``core/emg/surface/simulate_fiber.py``) evaluates ``iv``/``kv`` at integer orders
up to ~16, so a native-JAX, differentiable implementation is required to make the
EMG differentiable w.r.t. tissue conductivities, conduction velocity, and geometry
(milestone M7).

The forward pass is built entirely from smooth ops (JAX ``i0``/``i1``, polynomial
approximations, logs, and stable Bessel recurrences), so ``jax.grad`` produces the
exact derivative automatically — validated against scipy and the analytic
identities ``I_n'=(I_{n-1}+I_{n+1})/2``, ``K_n'=-(K_{n-1}+K_{n+1})/2`` to ~1e-7
(the accuracy floor of JAX's i0/i1) across orders 0–16 and x∈[0.05, 40].

See ``scripts/prototype_bessel_jax.py`` for the validation harness.
"""

from __future__ import annotations

from math import factorial

import jax.numpy as jnp
from jax.scipy.special import i0, i1

__all__ = ["k0", "k1", "iv_int", "kv_int"]


def k0(x):
    """Modified Bessel function K_0(x), x > 0 (Abramowitz & Stegun 9.8.5/9.8.6)."""
    x = jnp.asarray(x)
    y_s = x * x / 4.0
    small = (-jnp.log(x / 2.0) * i0(x)) + (
        -0.57721566 + y_s * (0.42278420 + y_s * (0.23069756 + y_s * (0.03488590
        + y_s * (0.00262698 + y_s * (0.00010750 + y_s * 0.00000740)))))
    )
    y_l = 2.0 / x
    large = (jnp.exp(-x) / jnp.sqrt(x)) * (
        1.25331414 + y_l * (-0.07832358 + y_l * (0.02189568 + y_l * (-0.01062446
        + y_l * (0.00587872 + y_l * (-0.00251540 + y_l * 0.00053208)))))
    )
    return jnp.where(x <= 2.0, small, large)


def k1(x):
    """Modified Bessel function K_1(x), x > 0 (Abramowitz & Stegun 9.8.7/9.8.8)."""
    x = jnp.asarray(x)
    y_s = x * x / 4.0
    small = (jnp.log(x / 2.0) * i1(x)) + (1.0 / x) * (
        1.0 + y_s * (0.15443144 + y_s * (-0.67278579 + y_s * (-0.18156897
        + y_s * (-0.01919402 + y_s * (-0.00110404 + y_s * (-0.00004686))))))
    )
    y_l = 2.0 / x
    large = (jnp.exp(-x) / jnp.sqrt(x)) * (
        1.25331414 + y_l * (0.23498619 + y_l * (-0.03655620 + y_l * (0.01504268
        + y_l * (-0.00780353 + y_l * (0.00325614 + y_l * (-0.00068245)))))))
    return jnp.where(x <= 2.0, small, large)


def kv_int(n: int, x):
    """K_n(x) for non-negative integer n via stable upward recurrence.

    ``K_{m+1}(x) = K_{m-1}(x) + (2m/x) K_m(x)``, seeded by K_0, K_1.
    """
    x = jnp.asarray(x)
    if n == 0:
        return k0(x)
    if n == 1:
        return k1(x)
    km1, kk = k0(x), k1(x)
    for m in range(1, n):
        km1, kk = kk, km1 + (2.0 * m / x) * kk
    return kk


def _iv_series(n: int, x, terms: int = 34):
    """Ascending series I_n(x) = Σ_k (x/2)^(2k+n) / (k! (n+k)!) — small-x branch."""
    half = x / 2.0
    term = half ** n / float(factorial(n))
    total = term
    for k in range(1, terms):
        term = term * (half * half) / (k * (n + k))
        total = total + term
    return total


def iv_int(n: int, x, miller_start: int = 60, x_switch: float = 1.0):
    """I_n(x) for non-negative integer n, differentiable for all x > 0.

    ``x >= x_switch``: Miller downward recurrence (the stable direction for I_n),
    seeded at a high order and normalised by matching the known I_0 (JAX ``i0``).
    ``x < x_switch``: ascending series, which keeps value and gradient accurate
    where Miller's ``(2k/x)`` factors would blow up. Blended with ``jnp.where``.
    """
    x = jnp.asarray(x)
    if n == 0:
        return i0(x)
    if n == 1:
        return i1(x)
    x_safe = jnp.where(x >= x_switch, x, jnp.asarray(x_switch, x.dtype))
    M = max(miller_start, n + 40)
    seq = [None] * (M + 2)
    seq[M + 1] = jnp.zeros_like(x_safe)
    seq[M] = jnp.ones_like(x_safe)
    for k in range(M, 0, -1):
        seq[k - 1] = seq[k + 1] + (2.0 * k / x_safe) * seq[k]
    miller = seq[n] * (i0(x_safe) / seq[0])
    series = _iv_series(n, x)
    return jnp.where(x >= x_switch, miller, series)
