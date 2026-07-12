"""Differentiable / RL proof-of-concept for MyoGen (JAX + jaxley).

This package is a **proof of concept**, deliberately separate from the imperative
``myogen.kernel`` and from the NEURON substrate. It exists to prove one thing: that a
single pure, JAX-native forward model can serve *both*

* gradient-based **autodiff** (fit MU-pool parameters to a target EMG via ``jax.grad``), and
* **reinforcement learning** (the same forward model wrapped as a vectorizable env),

built on real jaxley for the neural dynamics. NEURON stays a reference oracle; nothing here
is wired into production. Install with the ``diff`` extra (``pip install -e '.[diff]'``).

The shared atom is ``model.forward(params, drive, key) -> {voltages, spikes, emg}`` — a pure
function. ``autodiff`` wraps it in ``jax.grad``; ``env`` wraps the identical function as an
RL environment.
"""

from __future__ import annotations
