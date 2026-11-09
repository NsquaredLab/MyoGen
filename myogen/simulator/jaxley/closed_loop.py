"""
Differentiable closed-loop entry point for the MyoGen Jaxley backend.

This module packages the full ``jax.lax.scan`` neuromuscular loop — Jaxley
biophysics + synaptic conductances + stochastic afferent/descending generators +
Hill muscle + spindle + GTO + joint dynamics — into a single pure function
``run_jax`` for gradient / accelerator workflows.

JIT / grad usage
----------------
``run_jax`` cannot be handed to ``jax.jit`` *directly*: ``ClosedLoopConfig`` is
intentionally not hashable (it holds arrays), and a few integer Hill fields
(``hill_p["N"]``/``"Ntype1"``) are used in shape contexts (e.g. ``jnp.arange(N)``)
so they must stay static rather than being traced. Use the provided wrappers, which
perform the static/dynamic split for you:

* :func:`value_and_grad_run` — gradients (differentiates only float leaves; integer
  fields held static). This is the primary entry point for optimization.
* :func:`compile_run` — a ``jax.jit``-compiled forward pass bound to a config
  (Python-scalar fields baked static, all arrays traced).

Until now the loop only existed inline in
``examples/01_basic/11_simulate_spinal_network_jaxley.py``. Here it is a library
function with a clean split between:

* a **differentiable parameter PyTree** (``params``): everything you might take a
  gradient with respect to — Jaxley channel/synapse parameters, Hill/spindle/GTO/
  joint numeric leaves, and synaptic weights; and
* a **static configuration** (``ClosedLoopConfig``): network topology, connectivity
  matrices, counts, delays, timestep, initial states, and the compiled Jaxley
  ``step_fn``.

The stochastic generators are seeded from an explicit ``key`` argument so runs are
reproducible across JIT and devices. ``spike_mode`` selects hard (scientific
default), surrogate-gradient, or rate spikes and is recorded in the returned
metadata.

Typical use
-----------
::

    cfg = ClosedLoopConfig(...)               # built once from the network
    params = default_params(cfg, ...)         # differentiable PyTree
    outputs, meta = run_jax(params, cfg, inputs, jax.random.PRNGKey(0))

    # gradient of a scalar loss on the force trace w.r.t. all float params:
    loss = lambda out: jnp.mean(out["force"] ** 2)
    val, grads = value_and_grad_run(loss, params, cfg, inputs,
                                    jax.random.PRNGKey(0), spike_mode="surrogate")
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from myogen.simulator.jaxley.jax_models import (
    make_scan_step,
    poisson_init,
    gamma_init,
)

__all__ = [
    "ClosedLoopConfig",
    "run_jax",
    "compile_run",
    "value_and_grad_run",
    "partition_differentiable",
]


@dataclass(frozen=True)
class ClosedLoopConfig:
    """Static (non-differentiable) configuration for :func:`run_jax`.

    Everything here is closed over as a constant during tracing. Anything you may
    want gradients for lives in the ``params`` PyTree instead (see
    :func:`run_jax`).
    """

    # Compiled Jaxley single-step function (from ``build_init_and_step_fn``).
    step_fn: Callable
    external_inds: Any
    rec_inds: Any

    # Population counts.
    n_gii: int
    n_gib: int
    n_mn: int
    nDD: int
    nIa: int
    nII: int
    nIb: int

    # Afferent response thresholds and gamma-ISI shape parameters.
    ia_rts: Any
    ii_rts: Any
    ib_rts: Any
    ia_shape: float
    ii_shape: float
    ib_shape: float
    dd_N_batch: int

    # Connectivity matrices (pre_idx → post_idx).
    dd_to_mn_mat: Any
    ia_to_mn_mat: Any
    ii_to_gii_mat: Any
    ib_to_gib_mat: Any

    # Per-cell axonal delay steps.
    ia_delay_steps_arr: Any
    ii_delay_steps_arr: Any
    ib_delay_steps_arr: Any
    delay_steps: Any
    max_ia_delay_steps: int
    max_ii_delay_steps: int
    max_ib_delay_steps: int

    # Voltage-frame / synapse scalars.
    e_exc: float
    v_rest: float
    e_exc_mn: float
    mn_spike_threshold_mV: float
    tau_syn_decay: float
    dt_ms: float
    dt_s: float

    # Initial states.
    init_neural_states: Any
    init_hill_state: Any
    init_spindle_state: Any
    init_gto_state: Any
    init_joint_state: Any
    init_prev_v: float = -70.0

    # Default spike mode; can be overridden per call.
    spike_mode: str = "hard"


def _build_scan_step(params: dict, config: ClosedLoopConfig, spike_mode: str):
    """Instantiate the ``lax.scan`` step closure from params + static config.

    ``params`` leaves flow in as traced values, so gradients propagate through
    Jaxley parameters, Hill/spindle/GTO/joint leaves, and synaptic weights.
    """
    w = params["weights"]
    return make_scan_step(
        jaxley_step_fn=config.step_fn,
        jaxley_params=params["jaxley"],
        external_inds=config.external_inds,
        rec_inds=config.rec_inds,
        n_gii=config.n_gii,
        n_gib=config.n_gib,
        n_mn=config.n_mn,
        ia_rts=config.ia_rts,
        ii_rts=config.ii_rts,
        ib_rts=config.ib_rts,
        ia_shape=config.ia_shape,
        ii_shape=config.ii_shape,
        ib_shape=config.ib_shape,
        dd_N_batch=config.dd_N_batch,
        dd_to_mn_mat=config.dd_to_mn_mat,
        ia_to_mn_mat=config.ia_to_mn_mat,
        ii_to_gii_mat=config.ii_to_gii_mat,
        ib_to_gib_mat=config.ib_to_gib_mat,
        ia_delay_steps_arr=config.ia_delay_steps_arr,
        ii_delay_steps_arr=config.ii_delay_steps_arr,
        ib_delay_steps_arr=config.ib_delay_steps_arr,
        delay_steps=config.delay_steps,
        hill_p=params["hill"],
        spindle_p=params["spindle"],
        gto_p=params["gto"],
        joint_p=params["joint"],
        base_dd_weight=w["base_dd"],
        base_ia_weight=w["base_ia"],
        in_weight=w["in_weight"],
        e_exc=config.e_exc,
        v_rest=config.v_rest,
        e_exc_mn=config.e_exc_mn,
        mn_spike_threshold_mV=config.mn_spike_threshold_mV,
        mn_current_scale=w["mn_current_scale"],
        tau_syn_decay=config.tau_syn_decay,
        dt_ms=config.dt_ms,
        dt_s=config.dt_s,
        spike_mode=spike_mode,
    )


def _build_init_carry(params: dict, config: ClosedLoopConfig, key) -> dict:
    """Assemble the scan carry, seeding the stochastic generators from ``key``."""
    kd, ka, ki, kb = jax.random.split(key, 4)
    n_total = config.n_gii + config.n_gib + config.n_mn
    return {
        "neural": config.init_neural_states,
        "phys": {
            "hill": config.init_hill_state,
            "spindle": config.init_spindle_state,
            "gto": config.init_gto_state,
            "joint": config.init_joint_state,
        },
        "g_dd": jnp.zeros(config.n_mn, dtype=jnp.float32),
        "g_ia": jnp.zeros(config.n_mn, dtype=jnp.float32),
        "g_ii": jnp.zeros(config.n_gii, dtype=jnp.float32),
        "g_ib": jnp.zeros(config.n_gib, dtype=jnp.float32),
        "prev_v": jnp.full(n_total, config.init_prev_v, dtype=jnp.float32),
        "dd_st": poisson_init(config.nDD, config.dd_N_batch, key=kd),
        "ia_st": gamma_init(config.nIa, config.ia_shape, key=ka),
        "ii_st": gamma_init(config.nII, config.ii_shape, key=ki),
        "ib_st": gamma_init(config.nIb, config.ib_shape, key=kb),
        "prev_Iay": jnp.float32(0.0),
        "prev_IIy": jnp.float32(0.0),
        "prev_Iby": jnp.float32(0.0),
        "ia_delay_buf": jnp.zeros((config.nIa, config.max_ia_delay_steps), dtype=jnp.float32),
        "ii_delay_buf": jnp.zeros((config.nII, config.max_ii_delay_steps), dtype=jnp.float32),
        "ib_delay_buf": jnp.zeros((config.nIb, config.max_ib_delay_steps), dtype=jnp.float32),
    }


def run_jax(
    params: dict,
    config: ClosedLoopConfig,
    inputs: dict,
    key,
    spike_mode: Optional[str] = None,
):
    """Run the full closed-loop neuromuscular simulation as a pure function.

    Parameters
    ----------
    params : dict
        Differentiable PyTree with keys ``{"jaxley", "hill", "spindle", "gto",
        "joint", "weights"}``. ``weights`` is a dict of ``{"base_dd", "base_ia",
        "in_weight", "mn_current_scale"}``. Any float leaf may be differentiated.
    config : ClosedLoopConfig
        Static configuration (topology, connectivity, initial states, ``step_fn``).
    inputs : dict
        Per-step drive arrays stacked over time, each shape ``(n_steps, ...)``:
        ``{"DDdrive", "gDyn", "gStat", "tap_dL", "tap_dV"}``.
    key : jax PRNGKey
        Seeds the stochastic afferent/descending generators (split internally).
    spike_mode : {"hard", "surrogate", "rate"}, optional
        Overrides ``config.spike_mode`` for this call.

    Returns
    -------
    outputs : dict
        Per-step outputs stacked to ``(n_steps, ...)`` — see
        ``make_scan_step`` (``force``, ``torque``, ``v_mn``, ``mn_spikes``, ...).
    metadata : dict
        ``{"spike_mode", "n_steps", "n_gii", "n_gib", "n_mn"}``. The active spike
        mode is recorded here so downstream results carry provenance.
    """
    mode = spike_mode if spike_mode is not None else config.spike_mode
    if mode not in ("hard", "surrogate", "rate"):
        raise ValueError(
            f"spike_mode must be 'hard', 'surrogate', or 'rate'; got {mode!r}"
        )
    scan_step = _build_scan_step(params, config, mode)
    init_carry = _build_init_carry(params, config, key)
    _, outputs = jax.lax.scan(scan_step, init_carry, inputs)

    # n_steps from any stacked input leaf.
    any_leaf = jtu.tree_leaves(inputs)[0]
    metadata = {
        "spike_mode": mode,
        "n_steps": int(any_leaf.shape[0]),
        "n_gii": config.n_gii,
        "n_gib": config.n_gib,
        "n_mn": config.n_mn,
    }
    return outputs, metadata


# --------------------------------------------------------------------------- #
# JIT helper
# --------------------------------------------------------------------------- #

def _is_static_scalar(x) -> bool:
    """Python scalar (int/bool/str) with no array shape — must stay static for JIT."""
    return isinstance(x, (int, bool, str)) and not hasattr(x, "shape")


def compile_run(config: ClosedLoopConfig, params_template: dict, spike_mode: str = "hard"):
    """Return a ``jax.jit``-compiled ``run_jax`` bound to ``config``.

    ``run_jax`` is not directly jittable (see module docstring). This factory does
    the static/dynamic split: Python-scalar fields of ``params`` (e.g. the integer
    Hill counts used in ``jnp.arange``) are baked static from ``params_template``,
    all array leaves are traced, and ``config`` is closed over. Later calls must
    keep the same integer/name fields as the template (you optimize the float
    arrays, not the counts).

    Returns
    -------
    call(params, inputs, key) -> outputs
        Jitted; returns the outputs dict (metadata is static and omitted).
    """
    leaves0, treedef = jtu.tree_flatten(params_template)
    stat_mask = [_is_static_scalar(l) for l in leaves0]
    static_leaves = [l if m else None for l, m in zip(leaves0, stat_mask)]

    @jax.jit
    def _run(dyn_leaves, inputs, key):
        leaves = [s if m else d for s, d, m in zip(static_leaves, dyn_leaves, stat_mask)]
        full = jtu.tree_unflatten(treedef, leaves)
        outputs, _meta = run_jax(full, config, inputs, key, spike_mode=spike_mode)
        return outputs

    def call(params, inputs, key):
        leaves = jtu.tree_flatten(params)[0]
        dyn_leaves = [None if m else l for l, m in zip(leaves, stat_mask)]
        return _run(dyn_leaves, inputs, key)

    return call


# --------------------------------------------------------------------------- #
# Gradient helpers
# --------------------------------------------------------------------------- #

def _is_differentiable_leaf(x) -> bool:
    """True for float scalars / floating arrays; False for ints, bools, callables."""
    if isinstance(x, bool):
        return False
    if isinstance(x, (int,)):
        return False
    if isinstance(x, float):
        return True
    dtype = getattr(x, "dtype", None)
    if dtype is None:
        return False
    return jnp.issubdtype(dtype, jnp.floating) or jnp.issubdtype(dtype, jnp.complexfloating)


def partition_differentiable(params):
    """Split a params PyTree into (differentiable, static) leaf lists + rebuild info.

    Integer/bool leaves (e.g. Hill's ``Ntype1``/``N``) are placed in ``static`` so
    ``jax.grad`` — which rejects integer inputs — only differentiates float leaves.

    Returns
    -------
    diff_leaves : list
        Float leaves; ``None`` where the corresponding leaf is static.
    static_leaves : list
        Non-float leaves; ``None`` where the corresponding leaf is differentiable.
    treedef, mask : rebuild info for :func:`_combine`.
    """
    leaves, treedef = jtu.tree_flatten(params)
    mask = [_is_differentiable_leaf(l) for l in leaves]
    diff_leaves = [l if m else None for l, m in zip(leaves, mask)]
    static_leaves = [None if m else l for l, m in zip(leaves, mask)]
    return diff_leaves, static_leaves, treedef, mask


def _combine(diff_leaves, static_leaves, treedef, mask):
    leaves = [d if m else s for d, s, m in zip(diff_leaves, static_leaves, mask)]
    return jtu.tree_unflatten(treedef, leaves)


def value_and_grad_run(
    loss_fn: Callable,
    params: dict,
    config: ClosedLoopConfig,
    inputs: dict,
    key,
    spike_mode: str = "surrogate",
    has_aux: bool = False,
):
    """``jax.value_and_grad`` of ``loss_fn(run_jax(...))`` w.r.t. float params.

    Only floating leaves of ``params`` are differentiated (see
    :func:`partition_differentiable`); integer leaves are held constant. The
    default ``spike_mode="surrogate"`` keeps the forward trajectory identical to
    the hard-spike model while providing a usable (biased) gradient through spikes.

    Parameters
    ----------
    loss_fn : Callable
        ``loss_fn(outputs) -> scalar``, or ``-> (scalar, aux)`` when ``has_aux``.
    has_aux : bool
        If True, ``loss_fn`` returns ``(value, aux)`` and this returns
        ``(value, grads, aux)`` (``aux`` is not differentiated).

    Returns
    -------
    (value, grads) — or ``(value, grads, aux)`` when ``has_aux`` — where ``grads``
    is a params-shaped PyTree whose non-differentiable leaves are ``None``.
    """
    diff_leaves, static_leaves, treedef, mask = partition_differentiable(params)

    def f(diff):
        full = _combine(diff, static_leaves, treedef, mask)
        outputs, meta = run_jax(full, config, inputs, key, spike_mode=spike_mode)
        return loss_fn(outputs)

    result = jax.value_and_grad(f, has_aux=has_aux)(diff_leaves)
    if has_aux:
        (value, aux), grad_leaves = result
        return value, jtu.tree_unflatten(treedef, grad_leaves), aux
    value, grad_leaves = result
    return value, jtu.tree_unflatten(treedef, grad_leaves)
