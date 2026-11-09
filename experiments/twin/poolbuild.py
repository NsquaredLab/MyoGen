"""Build a differentiable NERLab motor-neuron-pool twin.

Wraps the canonical ex03 build path (AlphaMN__Pool → jx.Network →
``build_init_and_step_fn``) into a reusable object with:

* **excitatory descending drive**: per-source renewal-process spikes (the
  pure-JAX hazard generator from ``jax_models``; with ``dd_batch > 1`` this is
  a gamma/Erlang renewal process, NOT pure Poisson — ``dd_batch=1`` gives
  Poisson) → synaptic conductance → current;
* **inhibitory input**: an independent renewal population through the same
  synapse model with E_inh = −15 mV (the frame-shifted analog of −80 mV);
* a **configurable synapse model** (see below) so the twin can either match
  the fast ex03 surrogate or approximate NEURON's dendritic ``Exp2Syn``;
* a **differentiable parameter PyTree** ``theta``:
  ``{"gCaL_scale", "gNaP_scale", "drive_gain", "w_exc", "w_inh"}`` — the PIC
  conductances enter by scaling the ``caL_gcaLbar`` / ``napp_gnapbar`` leaves
  of the jaxley parameter dict inside the traced computation.

Synapse model options (``TwinSpec``)
------------------------------------
``syn_site``
    ``"soma"`` (ex03 surrogate convention) or ``"dendrite"`` — the dendrite is
    the compartment carrying ``caL``, which is where NEURON's MyoGen places its
    excitatory ``Exp2Syn``; dendritic placement lets synaptic depolarization
    engage the PIC the way the reference simulator does.
``live_driving_force``
    ``False``: current = g · (E − V_rest), the validated ex03 approximation.
    ``True``: current = g · (E − V_site(t−dt)) with the previous-step voltage
    of the synapse-bearing compartment (one-step lag breaks the algebraic
    loop). Required for apples-to-apples NEURON comparison.
``syn_tau_rise_ms``
    ``None``: single-exponential decay conductance (legacy IIR).
    A value (e.g. 0.5): peak-normalized double-exponential (Exp2Syn-like) with
    rise ``syn_tau_rise_ms`` and decay ``tau_syn_ms``.

Defaults keep the original fast-surrogate behavior (soma, fixed driving
force, single-exponential), bit-compatible with the earlier harness.

NOTE: this harness implements the drive→pool pathway only. Afferent input,
spindle/GTO, muscle force, and EMG observables are NOT yet wired into
``PoolTwin`` (see the roadmap's implemented-vs-planned table); for the full
closed loop use ``myogen.simulator.jaxley.closed_loop`` / ex11.

Voltage frame: NERLab/1952-HH — V_rest ≈ 0 mV, spikes peak ≈ +90 mV,
detection threshold +50 mV.

Typical use::

    twin = PoolTwin(TwinSpec(n_mn=20, seed=1))
    t, cmd = protocols.triangular_ramp(twin.spec.dt_ms, peak=40.0)
    drives = {"dd": jnp.asarray(cmd), "inh": jnp.zeros_like(jnp.asarray(cmd))}
    out = twin.simulate(twin.default_theta(), drives, jax.random.PRNGKey(0),
                        spike_mode="hard")
    # out["v"], out["v_dend"], out["spikes"]: (n_steps, n_mn)

    val, grads = twin.loss_and_grad(loss_fn, theta, drives, key)
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

import jaxley as jx
from jaxley.integrate import build_init_and_step_fn

from myogen.simulator.jaxley.populations import AlphaMN__Pool
from myogen.simulator.jaxley.jax_models import (
    poisson_init,
    poisson_step,
    spike_detect,
)

__all__ = ["TwinSpec", "PoolTwin", "fuglevand_thresholds"]

# NERLab (1952-HH) frame constants — see docs/session_notes/2026-06-02.
V_REST_MV = 0.0
E_EXC_MV = 70.0
E_INH_MV = -15.0
SPIKE_THRESHOLD_MV = 50.0

# spike_mode (MN detection) → stochastic-generator mode (jax_models convention).
_GEN_MODE = {"hard": "hard", "surrogate": "pathwise", "rate": "rate"}


def fuglevand_thresholds(n: int, recruitment_range: float = 30.0) -> np.ndarray:
    """Exponential recruitment-threshold distribution (Fuglevand et al. 1993)."""
    i = np.arange(1, n + 1)
    return np.exp(i * np.log(recruitment_range) / n) / recruitment_range


@dataclass(frozen=True)
class TwinSpec:
    """Static specification of the pool twin (nothing here is differentiated).

    ``seed`` controls the fixed random connectivity (and is the default base
    for generator seeding in the scripts). Two twins with the same spec are
    identical; change ``seed`` to resample connectivity.
    """

    n_mn: int = 20
    n_dd: int = 40  # descending-drive sources
    n_inh: int = 40  # inhibitory sources
    dd_batch: int = 5  # renewal order: 1 = Poisson, >1 = gamma/Erlang (more regular)
    dt_ms: float = 0.1
    connect_p: float = 0.5  # source→MN connection probability (ex03 value)
    tau_syn_ms: float = 5.0  # decay time constant
    w_exc_uS: float = 0.009  # ex03's validated IIR weight (≡ NEURON Exp2Syn 0.15)
    w_inh_uS: float = 0.009
    recruitment_range: float = 30.0
    seed: int = 0
    # Synapse model (see module docstring). Defaults = legacy fast surrogate.
    syn_site: str = "soma"  # "soma" | "dendrite"
    live_driving_force: bool = False
    syn_tau_rise_ms: float | None = None  # None → single-exp; value → Exp2Syn
    # Reversal potentials (NERLab 1952-HH frame). e_inh default −15 mV is the
    # FRAME-CORRECTED analog of a modern −80 mV (driving force −15 mV at
    # V_rest = 0). NEURON's NERLab synapses literally use −75 mV; matching
    # that number is a deliberate choice (`e_inh_mV=-75.0`) and any paired
    # comparison must state which convention it used.
    e_exc_mV: float = E_EXC_MV
    e_inh_mV: float = E_INH_MV


class PoolTwin:
    """A NERLab MN pool with differentiable drive/inhibition/PIC parameters."""

    def __init__(self, spec: TwinSpec):
        if spec.syn_site not in ("soma", "dendrite"):
            raise ValueError(f"syn_site must be 'soma' or 'dendrite', got {spec.syn_site!r}")
        if spec.syn_tau_rise_ms is not None and not (
            0.0 < spec.syn_tau_rise_ms < spec.tau_syn_ms
        ):
            raise ValueError("syn_tau_rise_ms must satisfy 0 < rise < tau_syn_ms")
        self.spec = spec
        rng = np.random.default_rng(spec.seed)

        thresholds = fuglevand_thresholds(spec.n_mn, spec.recruitment_range)
        pool = AlphaMN__Pool(
            recruitment_thresholds__array=thresholds,
            model="NERLab",
            mode="active",
        )

        # Pool wrappers → clean jx.Network. Recordings: soma then dendrite
        # voltage for every MN (dendrite needed for live driving force and for
        # PIC diagnostics). Stimuli: one slot per MN at the synapse site.
        cells = []
        for cw in pool:
            cell = cw.cell
            cell.delete_recordings()
            cell.delete_stimuli()
            cells.append(cell)
        net = jx.Network(cells)

        syn_branch = 0 if spec.syn_site == "soma" else 1
        placeholder = jnp.zeros(2)  # registers the slot; content unused
        for i in range(spec.n_mn):
            net.cell(i).branch(0).loc(0.5).record("v")  # soma
        for i in range(spec.n_mn):
            net.cell(i).branch(1).loc(0.5).record("v")  # dendrite
        for i in range(spec.n_mn):
            net.cell(i).branch(syn_branch).loc(0.5).stimulate(placeholder)

        # jx.Network() can silently revert v to the modern −70 mV frame;
        # re-assert the NERLab rest before initializing states.
        net.set("v", V_REST_MV)
        net.init_states()

        net.to_jax()
        init_fn, step_fn = build_init_and_step_fn(net)
        states0, all_params = init_fn(net.get_parameters())

        # The per-compartment parameter arrays carry NaN / sentinel entries for
        # compartments that lack the channel (soma vs dendrite vs axon); scale
        # only the valid, positive conductance entries or gradients go NaN.
        # (This deliberately bypasses jaxley's make_trainable: gradients flow
        # through the full all_params dict handed to step_fn.)
        self._pic_masks = {}
        for name in ("caL_gcaLbar", "napp_gnapbar"):
            if name not in all_params:
                raise KeyError(
                    f"{name!r} not in jaxley all_params "
                    f"(keys: {sorted(all_params)}) — jaxley layout changed?"
                )
            arr = np.asarray(all_params[name])
            self._pic_masks[name] = jnp.asarray(np.isfinite(arr) & (arr > 0))

        self.net = net
        self.pool = pool
        self.thresholds = thresholds
        self._step_fn = step_fn
        self._states0 = states0
        self._all_params = all_params
        self._external_inds = net.external_inds.copy()
        rec = jnp.array(net.recordings.rec_index.to_numpy(), dtype=jnp.int32)
        self._rec_soma = rec[: spec.n_mn]
        self._rec_dend = rec[spec.n_mn :]

        # Fixed random connectivity, (n_mn, n_sources) so `mat @ spikes` → (n_mn,).
        self._exc_mat = jnp.asarray(
            (rng.random((spec.n_mn, spec.n_dd)) < spec.connect_p).astype(np.float32)
        )
        self._inh_mat = jnp.asarray(
            (rng.random((spec.n_mn, spec.n_inh)) < spec.connect_p).astype(np.float32)
        )

        # Conductance-kernel constants.
        self._a_decay = float(np.exp(-spec.dt_ms / spec.tau_syn_ms))
        if spec.syn_tau_rise_ms is not None:
            t1, t2 = spec.syn_tau_rise_ms, spec.tau_syn_ms
            self._a_rise = float(np.exp(-spec.dt_ms / t1))
            tp = (t1 * t2 / (t2 - t1)) * np.log(t2 / t1)
            self._exp2_norm = float(1.0 / (np.exp(-tp / t2) - np.exp(-tp / t1)))
        else:
            self._a_rise = None
            self._exp2_norm = None

        self._sim = jax.jit(self._run, static_argnames=("spike_mode",))
        self._sim_events = jax.jit(self._run_events, static_argnames=("spike_mode",))
        self._sim_current = jax.jit(self._run_current, static_argnames=("spike_mode",))

    # ------------------------------------------------------------------ #

    def default_theta(self) -> dict:
        return {
            "gCaL_scale": jnp.float32(1.0),
            "gNaP_scale": jnp.float32(1.0),
            "drive_gain": jnp.float32(1.0),
            "w_exc": jnp.float32(self.spec.w_exc_uS),
            "w_inh": jnp.float32(self.spec.w_inh_uS),
        }

    def _scaled_jaxley_params(self, theta: dict) -> dict:
        p = dict(self._all_params)
        for name, scale in (
            ("caL_gcaLbar", theta["gCaL_scale"]),
            ("napp_gnapbar", theta["gNaP_scale"]),
        ):
            m = self._pic_masks[name]
            valid = jnp.where(m, p[name], 0.0)
            p[name] = jnp.where(m, valid * scale, p[name])
        return p

    def _syn_init(self) -> dict:
        n = self.spec.n_mn
        z = lambda: jnp.zeros(n)  # dtype follows active precision (x64-safe)
        if self.spec.syn_tau_rise_ms is None:
            return {"g_exc": z(), "g_inh": z()}
        return {"x1_exc": z(), "x2_exc": z(), "x1_inh": z(), "x2_inh": z()}

    def _syn_step(self, syn: dict, inc_exc, inc_inh) -> tuple[dict, jnp.ndarray, jnp.ndarray]:
        """Advance conductance states; return (new_state, g_exc, g_inh) in µS."""
        if self.spec.syn_tau_rise_ms is None:
            g_exc = self._a_decay * syn["g_exc"] + inc_exc
            g_inh = self._a_decay * syn["g_inh"] + inc_inh
            return {"g_exc": g_exc, "g_inh": g_inh}, g_exc, g_inh
        N = self._exp2_norm
        x1e = self._a_rise * syn["x1_exc"] + inc_exc * N
        x2e = self._a_decay * syn["x2_exc"] + inc_exc * N
        x1i = self._a_rise * syn["x1_inh"] + inc_inh * N
        x2i = self._a_decay * syn["x2_inh"] + inc_inh * N
        return (
            {"x1_exc": x1e, "x2_exc": x2e, "x1_inh": x1i, "x2_inh": x2i},
            x2e - x1e,
            x2i - x1i,
        )

    def _driving_forces(self, carry):
        spec = self.spec
        if spec.live_driving_force:
            v_site = (carry["prev_v_soma"] if spec.syn_site == "soma"
                      else carry["prev_v_dend"])
            return spec.e_exc_mV - v_site, spec.e_inh_mV - v_site
        return (spec.e_exc_mV - V_REST_MV), (spec.e_inh_mV - V_REST_MV)

    def _membrane_step(self, carry, i_stim, jax_params, spike_mode):
        """One jaxley step + spike detection; returns (carry updates, outputs)."""
        st = self._step_fn(
            carry["st"], jax_params, {"i": i_stim},
            self._external_inds, delta_t=self.spec.dt_ms,
        )
        v_soma = st["v"][self._rec_soma]
        v_dend = st["v"][self._rec_dend]
        sp = spike_detect(v_soma, carry["prev_v_soma"], SPIKE_THRESHOLD_MV,
                          spike_mode)
        return st, v_soma, v_dend, sp

    def _run(self, theta: dict, drives: dict, key, spike_mode: str = "hard"):
        """Pure simulation function (jitted as ``self.simulate``).

        drives : {"dd": (n_steps,), "inh": (n_steps,)} command rates in Hz
            (per source neuron). ``theta["drive_gain"]`` multiplies "dd".
        """
        spec = self.spec
        gen_mode = _GEN_MODE[spike_mode]
        jax_params = self._scaled_jaxley_params(theta)

        k_dd, k_inh = jax.random.split(key)
        carry0 = {
            "st": self._states0,
            "syn": self._syn_init(),
            "prev_v_soma": jnp.full(spec.n_mn, V_REST_MV),
            "prev_v_dend": jnp.full(spec.n_mn, V_REST_MV),
            "dd_st": poisson_init(spec.n_dd, spec.dd_batch, key=k_dd),
            "inh_st": poisson_init(spec.n_inh, spec.dd_batch, key=k_inh),
        }

        def body(carry, xs):
            dd_rate_t, inh_rate_t = xs
            dd_st, dd_sp = poisson_step(
                carry["dd_st"], theta["drive_gain"] * dd_rate_t,
                spec.dt_ms, spec.dd_batch, gen_mode,
            )
            inh_st, inh_sp = poisson_step(
                carry["inh_st"], inh_rate_t, spec.dt_ms, spec.dd_batch, gen_mode,
            )
            inc_exc = (self._exc_mat @ dd_sp.astype(jnp.float32)) * theta["w_exc"]
            inc_inh = (self._inh_mat @ inh_sp.astype(jnp.float32)) * theta["w_inh"]
            syn, g_exc, g_inh = self._syn_step(carry["syn"], inc_exc, inc_inh)
            df_exc, df_inh = self._driving_forces(carry)
            i_stim = g_exc * df_exc + g_inh * df_inh  # nA = µS × mV
            st, v_soma, v_dend, sp = self._membrane_step(
                carry, i_stim, jax_params, spike_mode)
            new_carry = {
                "st": st, "syn": syn,
                "prev_v_soma": v_soma, "prev_v_dend": v_dend,
                "dd_st": dd_st, "inh_st": inh_st,
            }
            return new_carry, {
                "v": v_soma, "v_dend": v_dend, "spikes": sp.astype(jnp.float32),
            }

        xs = (drives["dd"], drives["inh"])
        _, out = jax.lax.scan(body, carry0, xs)
        return out

    def _run_events(self, theta: dict, events: dict, spike_mode: str = "hard"):
        """Deterministic forward from exact presynaptic event matrices.

        ``events`` = {"exc": (n_steps, n_exc_src), "inh": (n_steps, n_inh_src),
        "exc_conn": (n_mn, n_exc_src), "inh_conn": (n_mn, n_inh_src)}.
        No stochastic generators are involved — the same matrices can be fed
        verbatim to NEURON for matched-input validation.
        """
        jax_params = self._scaled_jaxley_params(theta)
        exc_conn, inh_conn = events["exc_conn"], events["inh_conn"]
        carry0 = {
            "st": self._states0,
            "syn": self._syn_init(),
            "prev_v_soma": jnp.full(self.spec.n_mn, V_REST_MV),
            "prev_v_dend": jnp.full(self.spec.n_mn, V_REST_MV),
        }

        def body(carry, xs):
            exc_t, inh_t = xs
            inc_exc = (exc_conn @ exc_t) * theta["w_exc"]
            inc_inh = (inh_conn @ inh_t) * theta["w_inh"]
            syn, g_exc, g_inh = self._syn_step(carry["syn"], inc_exc, inc_inh)
            df_exc, df_inh = self._driving_forces(carry)
            i_stim = g_exc * df_exc + g_inh * df_inh
            st, v_soma, v_dend, sp = self._membrane_step(
                carry, i_stim, jax_params, spike_mode)
            new_carry = {
                "st": st, "syn": syn,
                "prev_v_soma": v_soma, "prev_v_dend": v_dend,
            }
            return new_carry, {
                "v": v_soma, "v_dend": v_dend, "spikes": sp.astype(jnp.float32),
            }

        _, out = jax.lax.scan(body, carry0, (events["exc"], events["inh"]))
        return out

    def _run_events_carry(self, theta: dict, events: dict, carry0,
                          spike_mode: str = "hard"):
        """Like ``_run_events`` but starts from an explicit carry and returns
        ``(outputs, final_carry)`` — enables windowed losses that preserve
        forward state across window boundaries (never restart from rest;
        gradients may be stopped at boundaries by the caller)."""
        jax_params = self._scaled_jaxley_params(theta)
        exc_conn, inh_conn = events["exc_conn"], events["inh_conn"]
        if carry0 is None:
            carry0 = {
                "st": self._states0,
                "syn": self._syn_init(),
                "prev_v_soma": jnp.full(self.spec.n_mn, V_REST_MV),
                "prev_v_dend": jnp.full(self.spec.n_mn, V_REST_MV),
            }

        def body(carry, xs):
            exc_t, inh_t = xs
            inc_exc = (exc_conn @ exc_t) * theta["w_exc"]
            inc_inh = (inh_conn @ inh_t) * theta["w_inh"]
            syn, g_exc, g_inh = self._syn_step(carry["syn"], inc_exc, inc_inh)
            df_exc, df_inh = self._driving_forces(carry)
            i_stim = g_exc * df_exc + g_inh * df_inh
            st, v_soma, v_dend, sp = self._membrane_step(
                carry, i_stim, jax_params, spike_mode)
            new_carry = {
                "st": st, "syn": syn,
                "prev_v_soma": v_soma, "prev_v_dend": v_dend,
            }
            return new_carry, {
                "v": v_soma, "v_dend": v_dend, "spikes": sp.astype(v_soma.dtype),
            }

        carry_f, out = jax.lax.scan(body, carry0,
                                    (events["exc"], events["inh"]))
        return out, carry_f

    def _run_current(self, theta: dict, i_ext, spike_mode: str = "hard"):
        """Deterministic forward from an exact injected-current waveform.

        ``i_ext``: (n_steps, n_mn) current in nA, injected at the registered
        stimulus site (``spec.syn_site`` compartment). No synapse model is
        involved — this is the intrinsic (current-injection) validation path.
        """
        jax_params = self._scaled_jaxley_params(theta)
        carry0 = {
            "st": self._states0,
            "prev_v_soma": jnp.full(self.spec.n_mn, V_REST_MV),
            "prev_v_dend": jnp.full(self.spec.n_mn, V_REST_MV),
        }

        def body(carry, i_t):
            st, v_soma, v_dend, sp = self._membrane_step(
                carry, i_t, jax_params, spike_mode)
            return (
                {"st": st, "prev_v_soma": v_soma, "prev_v_dend": v_dend},
                {"v": v_soma, "v_dend": v_dend, "spikes": sp.astype(jnp.float32)},
            )

        _, out = jax.lax.scan(body, carry0, i_ext)
        return out

    # ------------------------------------------------------------------ #

    def simulate_current(self, theta, i_ext_nA, spike_mode: str = "hard"):
        """Forward run injecting ``i_ext_nA`` (n_steps, n_mn) at the stimulus
        site — for deterministic single-cell / intrinsic validation."""
        i_ext = jnp.asarray(i_ext_nA, dtype=jnp.float32)
        if i_ext.ndim != 2 or i_ext.shape[1] != self.spec.n_mn:
            raise ValueError(f"i_ext must be (n_steps, {self.spec.n_mn})")
        return self._sim_current(theta, i_ext, spike_mode=spike_mode)

    def simulate_events(
        self,
        theta,
        exc_events,
        inh_events=None,
        exc_connectivity=None,
        inh_connectivity=None,
        spike_mode: str = "hard",
    ):
        """Forward run from EXACT presynaptic events — the matched-input path.

        Parameters
        ----------
        exc_events : (n_steps, n_exc_src)
            0/1 (or float) excitatory event matrix.
        inh_events : (n_steps, n_inh_src), optional
            Defaults to no inhibitory input.
        exc_connectivity, inh_connectivity : (n_mn, n_src), optional
            Default to the twin's own fixed random connectivity; pass explicit
            matrices to reproduce a protocol artifact exactly.

        Returns ``{"v", "v_dend", "spikes"}``, each ``(n_steps, n_mn)``.
        """
        spec = self.spec
        exc_events = jnp.asarray(exc_events, dtype=jnp.float32)
        n_steps = exc_events.shape[0]
        if inh_events is None:
            inh_events = jnp.zeros((n_steps, 0), dtype=jnp.float32)
        else:
            inh_events = jnp.asarray(inh_events, dtype=jnp.float32)
        if exc_connectivity is None:
            if exc_events.shape[1] != spec.n_dd:
                raise ValueError(
                    f"exc_events has {exc_events.shape[1]} sources but the twin "
                    f"has n_dd={spec.n_dd}; pass exc_connectivity explicitly")
            exc_connectivity = self._exc_mat
        if inh_connectivity is None:
            if inh_events.shape[1] == spec.n_inh:
                inh_connectivity = self._inh_mat
            else:
                inh_connectivity = jnp.zeros(
                    (spec.n_mn, inh_events.shape[1]), dtype=jnp.float32)
        events = {
            "exc": exc_events,
            "inh": inh_events,
            "exc_conn": jnp.asarray(exc_connectivity, dtype=jnp.float32),
            "inh_conn": jnp.asarray(inh_connectivity, dtype=jnp.float32),
        }
        if events["exc_conn"].shape != (spec.n_mn, exc_events.shape[1]):
            raise ValueError("exc_connectivity shape mismatch")
        if events["inh_conn"].shape != (spec.n_mn, inh_events.shape[1]):
            raise ValueError("inh_connectivity shape mismatch")
        return self._sim_events(theta, events, spike_mode=spike_mode)

    def simulate(self, theta, drives, key, spike_mode: str = "hard"):
        """Jitted forward run → ``{"v", "v_dend", "spikes"}``, each ``(n_steps, n_mn)``."""
        return self._sim(theta, drives, key, spike_mode=spike_mode)

    def loss_and_grad(
        self, loss_fn, theta, drives, key, spike_mode: str = "surrogate"
    ):
        """``value_and_grad`` of ``loss_fn(outputs)`` w.r.t. ``theta``."""

        def f(th):
            return loss_fn(self._sim(th, drives, key, spike_mode=spike_mode))

        return jax.value_and_grad(f)(theta)
