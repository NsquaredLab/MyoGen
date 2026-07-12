"""A Pythonic, flexible, differentiable spiking-network façade.

This is a prototype of the proposed MyoGen API: you *build* a network imperatively (plain
objects, ``add``/``connect`` by object reference — mutation is fine at build time), then *run*
it purely (``simulate`` compiles to one ``jax.lax.scan`` and is ``jax.grad``-able end to end).
The build phase is ordinary Python; the run phase is the pure boundary. One verb — ``connect``
— wires everything (population→population synapses and signal→population sensory edges); cycles
(recurrent inhibition, afferent feedback) are handled by a uniform one-tick delay.

Populations expose a single contract — ``step(state, current, key) -> (state, spikes)`` — so a
jaxley biophysical stage (PR #23) drops in behind the same interface. Here we use a
differentiable LIF (surrogate-gradient spikes) so the whole thing is small and runnable, and so
the load-bearing claims — *arbitrary topology stays one differentiable scan*, and *every
connection weight is a gradient target* (fit reflex gains) — can be shown concretely.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from myogen.diff.emg import EMGConfig, default_emg_config, spikes_to_emg
from myogen.diff.spikes import _surrogate_heaviside


# --------------------------------------------------------------------------------------------
# Populations — the per-tick dynamics contract
# --------------------------------------------------------------------------------------------


class Population:
    """Base handle: an identity you hold and reference. Subclasses define the per-tick step."""

    n: int

    def __init__(self, n: int, name: str | None = None):
        self.n = n
        self.name = name or type(self).__name__
        self._id: int | None = None  # assigned by Network.add

    def __rshift__(self, other: "Population") -> "Edge":
        return Edge(self, other)  # enables `net.params[Ia >> aMN]`

    def __repr__(self):
        return f"{self.name}(n={self.n})"

    # contract implemented by subclasses -------------------------------------------------
    def init_state(self):
        raise NotImplementedError

    def step(self, state, current, key):
        """(state, current[n], key) -> (new_state, spikes[n]). Pure; differentiable."""
        raise NotImplementedError


class LIFPopulation(Population):
    """Leaky integrate-and-fire with a surrogate-gradient spike (differentiable)."""

    def __init__(self, n, *, tau_ms=10.0, dt_ms=0.05, v_thresh=1.0, r_input=1.0, name=None):
        super().__init__(n, name)
        self.alpha = dt_ms / tau_ms  # leak per tick
        self.v_thresh = v_thresh
        self.r_input = r_input

    def init_state(self):
        return jnp.zeros(self.n)  # membrane potential at rest

    def step(self, v, current, key):
        v = v + self.alpha * (-v + self.r_input * current)
        spikes = _surrogate_heaviside(v - self.v_thresh)          # 1 where above threshold
        v = v - spikes * self.v_thresh                            # soft reset (differentiable)
        return v, spikes


class DriveInput(Population):
    """A source population whose 'spikes' are the external drive fed in at ``simulate``."""

    def init_state(self):
        return jnp.zeros(self.n)

    def step(self, state, current, key):
        # its output is the injected current itself (treated as a rate/spike proxy)
        return state, current


# --------------------------------------------------------------------------------------------
# Edges / connections
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Edge:
    source: Population
    target: Population


@dataclass
class Connection:
    source: Population
    target: Population
    mask: jnp.ndarray  # (n_target, n_source) frozen 0/1 topology
    sign: float        # +1 excitatory, -1 inhibitory
    delay_steps: int   # >=1; one-tick minimum breaks cycles
    fan_in: jnp.ndarray = field(default=None)  # per-target presynaptic partner count (>=1)


# --------------------------------------------------------------------------------------------
# Params — object-indexed, and a JAX pytree so jax.grad just works
# --------------------------------------------------------------------------------------------


class Params:
    """Trainable weights, one scalar gain per connection. A pytree; indexable by ``src >> tgt``."""

    def __init__(self, weights: list, edges: list[tuple[int, int]]):
        self.weights = weights          # list[jax.Array] (leaves)
        self._edges = edges             # list[(source_id, target_id)]  (static)

    def __getitem__(self, key: Edge):
        i = self._edges.index((key.source._id, key.target._id))
        return self.weights[i]

    def tree_flatten(self):
        return tuple(self.weights), tuple(self._edges)

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(list(children), list(aux))


jax.tree_util.register_pytree_node(Params, Params.tree_flatten, Params.tree_unflatten)


# --------------------------------------------------------------------------------------------
# Result — indexed by the population object, not a string
# --------------------------------------------------------------------------------------------


@dataclass
class _PopView:
    spikes: jnp.ndarray  # (time, n)


@dataclass
class Result:
    _spikes: dict  # {pop_id: (time, n)}
    _id_of: dict   # {Population: pop_id}
    emg: jnp.ndarray | None = None

    def __getitem__(self, pop: Population) -> _PopView:
        return _PopView(spikes=self._spikes[pop._id])


# --------------------------------------------------------------------------------------------
# Network — imperative build phase, pure run phase
# --------------------------------------------------------------------------------------------


class Network:
    def __init__(self, dt_ms: float = 0.05, *, seed: int = 0):
        self.dt_ms = dt_ms
        self._pops: list[Population] = []
        self._conns: list[Connection] = []
        self._emg_source: Population | None = None
        self._emg_cfg: EMGConfig | None = None
        self._key = jax.random.PRNGKey(seed)

    # ---- build phase (mutable, Pythonic) ----
    def add(self, pop: Population) -> Population:
        pop._id = len(self._pops)
        self._pops.append(pop)
        return pop

    def connect(self, source: Population, target: Population, *, weight=1.0,
                p: float = 1.0, inhibitory: bool = False, delay_steps: int = 1) -> Connection:
        """Wire source→target. `weight` is the trainable gain; `p` sets a frozen sparsity mask."""
        if delay_steps < 1:
            raise ValueError("delay_steps must be >= 1 (a same-tick edge could form an "
                             "algebraic cycle; the one-tick minimum breaks all loops).")
        self._key, k = jax.random.split(self._key)
        mask = (jax.random.uniform(k, (target.n, source.n)) < p).astype(jnp.float64)
        fan_in = jnp.maximum(mask.sum(axis=1), 1.0)  # presynaptic partners per target (>=1)
        conn = Connection(source, target, mask=mask, sign=-1.0 if inhibitory else 1.0,
                          delay_steps=delay_steps, fan_in=fan_in)
        conn._init_weight = float(weight)  # type: ignore[attr-defined]
        self._conns.append(conn)
        return conn

    def emg_from(self, source: Population, *, n_channels: int = 4):
        self._emg_source = source
        self._emg_cfg = default_emg_config(source.n, n_channels)

    # ---- params (object-indexed pytree) ----
    @property
    def params(self) -> Params:
        weights = [jnp.asarray(c._init_weight, dtype=jnp.float64) for c in self._conns]
        edges = [(c.source._id, c.target._id) for c in self._conns]
        return Params(weights, edges)

    # ---- pure run phase ----
    def simulate(self, duration_ms: float, *, drive: dict | None = None,
                 params: Params | None = None) -> Result:
        params = params if params is not None else self.params
        n_steps = int(duration_ms / self.dt_ms)
        drive = drive or {}
        # external drive per population, shape (n_steps, n) — zeros if undriven
        ext = [jnp.zeros((n_steps, p.n)) for p in self._pops]
        for pop, sig in drive.items():
            ext[pop._id] = jnp.broadcast_to(jnp.asarray(sig), (n_steps, pop.n))
        ext = tuple(ext)

        weights = params.weights
        pops = self._pops
        conns = self._conns

        def tick(carry, inp):
            states, prev_spikes = carry
            drive_t = inp
            # input current to each population = external drive + one-tick-delayed synapses
            currents = [drive_t[i] for i in range(len(pops))]
            for c, w in zip(conns, weights):
                # mean spike activity over each target's presynaptic partners × gain (scale-invariant)
                contrib = c.sign * w * (c.mask @ prev_spikes[c.source._id]) / c.fan_in
                currents[c.target._id] = currents[c.target._id] + contrib
            new_states, spikes = [], []
            for i, pop in enumerate(pops):
                st, sp = pop.step(states[i], currents[i], None)
                new_states.append(st); spikes.append(sp)
            return (tuple(new_states), tuple(spikes)), tuple(spikes)

        init = (tuple(p.init_state() for p in pops),
                tuple(jnp.zeros(p.n) for p in pops))
        _, spike_hist = jax.lax.scan(tick, init, ext)  # spike_hist[i]: (n_steps, n)

        spikes = {p._id: spike_hist[i] for i, p in enumerate(pops)}
        emg = None
        if self._emg_source is not None:
            src = spike_hist[self._emg_source._id].T  # (n, time)
            emg = spikes_to_emg(src, self._emg_cfg)
        return Result(_spikes=spikes, _id_of={p: p._id for p in pops}, emg=emg)

    # ---- RL env (Brax/gymnax styling) over the same tick ----
    def as_env(self, *, action: Population, reward, observe: Population,
               duration_ms: float, base_drive=0.0):
        return Env(self, action=action, reward=reward, observe=observe,
                   duration_ms=duration_ms, base_drive=base_drive)


@dataclass
class Env:
    net: Network
    action: Population
    reward: object
    observe: Population
    duration_ms: float
    base_drive: float = 0.0

    def reset(self, key):
        return key, jnp.zeros(self.observe.n)

    def step(self, key, state, action, params: Params | None = None):
        drive = {self.action: self.base_drive + jnp.asarray(action)}
        r = self.net.simulate(self.duration_ms, drive=drive, params=params)
        obs = r[self.observe].spikes.mean(axis=0)
        reward = self.reward(r)
        return key, obs, reward, jnp.array(True)
