"""
NERLab Channel Models — Jaxley Backend.
=======================================

JAX/Jaxley implementations of the channels used by MyoGen's production NEURON
motor-neuron model (``model="NERLab"`` in ``AlphaMN__Pool``).  These are direct
ports of the NMODL files:

    myogen/simulator/nmodl_files/napp.mod      -> ``napp`` (5-gate Na + K + leak)
    myogen/simulator/nmodl_files/caL.mod       -> ``caL`` (L-type Ca, no inactivation)

The static-current ``Constant`` mechanism is already available as
``myogen.simulator.jaxley.channels.Constant`` (no duplication here).

**Voltage convention — IMPORTANT.**  NERLab uses the *original 1952 Hodgkin-Huxley*
convention, NOT the modern absolute-V convention used by the Powers2017 channels
elsewhere in this package.  Concretely:

    V_rest    ≈   0 mV   (set by ``el_napp = 0``)
    ENa       = +120 mV  (spike peaks reach ~+90 to +110 mV)
    EK        = -10 mV   (AHPs dip to ~-5 to -10 mV)
    vtraub    =   0 mV   (no extra voltage offset in the gating equations)

Do NOT mix these channels with the Powers2017 channels on the same cell.  Cells
using NERLab channels must initialise V at 0 mV, not -65 mV.

Author: ported for the Jaxley/NEURON architecture-parity work, 2026.
"""

from typing import Dict, Optional

import jax.numpy as jnp
from jaxley.channels import Channel
from jaxley.solver_gate import exponential_euler, save_exp


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _vtrap(x: jnp.ndarray, y: float) -> jnp.ndarray:
    """JAX-safe port of NEURON's ``vtrap(x, y) = x / (exp(x/y) - 1)``.

    Uses the L'Hopital expansion ``y * (1 - x / y / 2)`` when ``|x/y|`` is small,
    avoiding the 0/0 singularity at ``x == 0``.  Implemented with ``jnp.where``
    so it is differentiable and works under ``jit``.
    """
    safe = jnp.abs(x / y) < 1e-4
    # When |x/y| < 1e-4 use the linear approximation; otherwise the analytic form.
    # The denominator term is computed in a way that never divides by exactly 0.
    denom = save_exp(x / y) - 1.0
    # Avoid exact zero in the denominator under jit: where ``safe`` is True the
    # value isn't used, but JAX still evaluates both branches, so guard it.
    denom_safe = jnp.where(safe, 1.0, denom)
    return jnp.where(safe, y * (1.0 - x / y / 2.0), x / denom_safe)


# -----------------------------------------------------------------------------
# napp — Hodgkin-Huxley-style channel set used in the NERLab motor neuron
# -----------------------------------------------------------------------------

class napp(Channel):
    """5-gate Na/K/leak channel set used by MyoGen's NERLab soma.

    Gates
    -----
    m (×3) : fast sodium activation
    h (×1) : fast sodium inactivation
    p (×3) : persistent sodium activation
    n (×4) : fast potassium activation
    r (×2) : slow potassium activation

    Currents
    --------
    ina = (gnabar * m^3 * h + gnapbar * p^3) * (v - ena)
    ik  = (gkfbar * n^4 + gksbar * r^2)      * (v - ek)
    il  = gl * (v - el)

    All gating equations are evaluated at ``v2 = v - vtraub``.  The default
    ``vtraub = 0`` matches the production NERLab YAML.
    """

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = self._name

        # Conductance and reversal defaults match napp.mod / alpha_mn_default.yaml.
        self.channel_params = {
            # Maximal conductances
            f"{prefix}_gnabar":  0.030,        # S/cm^2  (fast Na)
            f"{prefix}_gnapbar": 0.000033,     # S/cm^2  (persistent Na)
            f"{prefix}_gkfbar":  0.016,        # S/cm^2  (fast K)
            f"{prefix}_gksbar":  0.004,        # S/cm^2  (slow K)
            f"{prefix}_gl":      0.0003,       # S/cm^2  (leak)
            # Reversals  (NERLab / original-HH convention)
            f"{prefix}_ena":     120.0,        # mV
            f"{prefix}_ek":      -10.0,        # mV
            f"{prefix}_el":      -54.3,        # mV  (NEURON default; overridden to 0 by pool)
            # Voltage offset used by every gate
            f"{prefix}_vtraub":  0.0,          # mV  (production NERLab uses 0)
            # Threshold-like parameter (not used in dynamics; legacy NEURON parameter)
            f"{prefix}_mact":    15.0,
            # Slow-K constant deactivation rate (β_r)
            f"{prefix}_rinact":  0.05,         # /ms
            # ---- Per-gate kinetic constants (from YAML) ----
            # m
            f"{prefix}_m_alpha_A": 0.64,  f"{prefix}_m_alpha_v_offset": 15.0, f"{prefix}_m_alpha_k": 4.0,
            f"{prefix}_m_beta_A":  0.56,  f"{prefix}_m_beta_v_offset":  40.0, f"{prefix}_m_beta_k":  5.0,
            # h
            f"{prefix}_h_alpha_A": 0.928, f"{prefix}_h_alpha_v_offset": 17.0, f"{prefix}_h_alpha_tau": 18.0,
            f"{prefix}_h_beta_A":  9.0,   f"{prefix}_h_beta_v_offset":  40.0, f"{prefix}_h_beta_k":   5.0,
            # p
            f"{prefix}_p_alpha_A": 0.64,  f"{prefix}_p_alpha_v_offset":  5.0, f"{prefix}_p_alpha_k":  4.0,
            f"{prefix}_p_beta_A":  0.56,  f"{prefix}_p_beta_v_offset":  30.0, f"{prefix}_p_beta_k":   5.0,
            # n
            f"{prefix}_n_alpha_A": 0.08,  f"{prefix}_n_alpha_v_offset": 15.0, f"{prefix}_n_alpha_k":  7.0,
            f"{prefix}_n_beta_A":  2.0,   f"{prefix}_n_beta_v_offset":  10.0, f"{prefix}_n_beta_tau":40.0,
            # r
            f"{prefix}_r_alpha_A": 3.5,   f"{prefix}_r_alpha_v_offset": 55.0, f"{prefix}_r_alpha_k":  4.0,
        }
        # Initial states near a NERLab resting condition (v ≈ 0). Will be
        # overwritten by ``init_states`` when the cell is built; these only
        # matter if the user forgets to initialise.
        self.channel_states = {
            f"{prefix}_m": 0.01,
            f"{prefix}_h": 0.99,
            f"{prefix}_p": 0.07,
            f"{prefix}_n": 0.06,
            f"{prefix}_r": 0.0,
        }
        self.current_name = "i_napp"

    # ---- gating helpers (alpha, beta) ----

    @staticmethod
    def _m_rates(v2, p):
        alpha = p["m_alpha_A"] * _vtrap(p["m_alpha_v_offset"] - v2, p["m_alpha_k"])
        beta  = p["m_beta_A"]  * _vtrap(v2 - p["m_beta_v_offset"],  p["m_beta_k"])
        return alpha, beta

    @staticmethod
    def _h_rates(v2, p):
        alpha = p["h_alpha_A"] * save_exp((p["h_alpha_v_offset"] - v2) / p["h_alpha_tau"])
        beta  = p["h_beta_A"]  / (save_exp((p["h_beta_v_offset"] - v2) / p["h_beta_k"]) + 1.0)
        return alpha, beta

    @staticmethod
    def _p_rates(v2, p):
        alpha = p["p_alpha_A"] * _vtrap(p["p_alpha_v_offset"] - v2, p["p_alpha_k"])
        beta  = p["p_beta_A"]  * _vtrap(v2 - p["p_beta_v_offset"],  p["p_beta_k"])
        return alpha, beta

    @staticmethod
    def _n_rates(v2, p):
        alpha = p["n_alpha_A"] * _vtrap(p["n_alpha_v_offset"] - v2, p["n_alpha_k"])
        beta  = p["n_beta_A"]  * save_exp((p["n_beta_v_offset"] - v2) / p["n_beta_tau"])
        return alpha, beta

    @staticmethod
    def _r_rates(v2, p):
        alpha = p["r_alpha_A"] / (save_exp((p["r_alpha_v_offset"] - v2) / p["r_alpha_k"]) + 1.0)
        beta  = p["rinact"]
        return alpha, beta

    @staticmethod
    def _inf_tau(alpha, beta):
        sumab = alpha + beta + 1e-12       # avoid div-by-zero in pathological regimes
        return alpha / sumab, 1.0 / sumab

    def update_states(self, states, dt, v, params):
        prefix = self._name
        # Pack the kinetic constants into a name->array dict for the helpers.
        p = {
            k.replace(f"{prefix}_", ""): params[k]
            for k in params if k.startswith(prefix) and not k.endswith(("_gnabar", "_gnapbar", "_gkfbar", "_gksbar", "_gl", "_ena", "_ek", "_el", "_mact"))
        }
        v2 = v - params[f"{prefix}_vtraub"]

        m_inf, m_tau = self._inf_tau(*self._m_rates(v2, p))
        h_inf, h_tau = self._inf_tau(*self._h_rates(v2, p))
        p_inf, p_tau = self._inf_tau(*self._p_rates(v2, p))
        n_inf, n_tau = self._inf_tau(*self._n_rates(v2, p))
        r_inf, r_tau = self._inf_tau(*self._r_rates(v2, p))

        m = states[f"{prefix}_m"]; h = states[f"{prefix}_h"]
        ppstate = states[f"{prefix}_p"]
        n = states[f"{prefix}_n"]; r = states[f"{prefix}_r"]

        return {
            f"{prefix}_m": exponential_euler(m,        dt, m_inf, m_tau),
            f"{prefix}_h": exponential_euler(h,        dt, h_inf, h_tau),
            f"{prefix}_p": exponential_euler(ppstate,  dt, p_inf, p_tau),
            f"{prefix}_n": exponential_euler(n,        dt, n_inf, n_tau),
            f"{prefix}_r": exponential_euler(r,        dt, r_inf, r_tau),
        }

    def compute_current(self, states, v, params):
        prefix = self._name
        m = states[f"{prefix}_m"]; h = states[f"{prefix}_h"]
        pp = states[f"{prefix}_p"]
        n = states[f"{prefix}_n"]; r = states[f"{prefix}_r"]

        gna  = params[f"{prefix}_gnabar"]  * (m ** 3) * h
        gnap = params[f"{prefix}_gnapbar"] * (pp ** 3)
        gkf  = params[f"{prefix}_gkfbar"]  * (n ** 4)
        gks  = params[f"{prefix}_gksbar"]  * (r ** 2)
        gl   = params[f"{prefix}_gl"]

        ena = params[f"{prefix}_ena"]
        ek  = params[f"{prefix}_ek"]
        el  = params[f"{prefix}_el"]

        ina = (gna + gnap) * (v - ena)
        ik  = (gkf + gks)  * (v - ek)
        il  = gl * (v - el)
        return ina + ik + il

    def init_state(self, states, v, params, dt):
        """Compute steady-state gate values at the holding voltage `v`."""
        prefix = self._name
        p = {
            k.replace(f"{prefix}_", ""): params[k]
            for k in params if k.startswith(prefix) and not k.endswith(("_gnabar", "_gnapbar", "_gkfbar", "_gksbar", "_gl", "_ena", "_ek", "_el", "_mact"))
        }
        v2 = v - params[f"{prefix}_vtraub"]
        m_inf, _ = self._inf_tau(*self._m_rates(v2, p))
        h_inf, _ = self._inf_tau(*self._h_rates(v2, p))
        p_inf, _ = self._inf_tau(*self._p_rates(v2, p))
        n_inf, _ = self._inf_tau(*self._n_rates(v2, p))
        r_inf, _ = self._inf_tau(*self._r_rates(v2, p))
        return {
            f"{prefix}_m": m_inf,
            f"{prefix}_h": h_inf,
            f"{prefix}_p": p_inf,
            f"{prefix}_n": n_inf,
            f"{prefix}_r": r_inf,
        }


# -----------------------------------------------------------------------------
# caL — L-type calcium channel used in the NERLab dendrite (no inactivation)
# -----------------------------------------------------------------------------

class caL(Channel):
    """L-type Ca channel from MyoGen's NERLab dendrite.

    Single activation gate ``L`` with a fixed time constant ``Ltau``:

        Linf = 1 / (1 + exp(-(v2 + 30)))     (NMODL uses k = -1 mV → extremely steep)
        dL/dt = (Linf - L) / Ltau
        icaL  = gcaLbar * L * gama * (v - ecaL)

    A passive leak ``il = gl * (v - el)`` is bundled in (matches caL.mod).
    """

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gcaLbar": 0.030,        # S/cm^2
            f"{prefix}_gl":      0.0003,       # S/cm^2  (dendritic leak)
            f"{prefix}_el":      -54.3,        # mV  (overridden to 0 by NERLab pool)
            f"{prefix}_ecaL":    140.0,        # mV
            f"{prefix}_vtraub":  70.0,         # mV  (NEURON default; NERLab uses ~35)
            f"{prefix}_gama":    1.0,
            f"{prefix}_Ltau":    20.0,         # ms
        }
        self.channel_states = {
            f"{prefix}_L": 0.0,
        }
        self.current_name = "i_caL"

    @staticmethod
    def _Linf(v2):
        # NMODL: Linf = 1 / (exp((v2+30)/-1) + 1) — equivalent to logistic with k=-1 mV.
        return 1.0 / (1.0 + save_exp((v2 + 30.0) / -1.0))

    def update_states(self, states, dt, v, params):
        prefix = self._name
        v2 = v - params[f"{prefix}_vtraub"]
        L_inf = self._Linf(v2)
        L_new = exponential_euler(states[f"{prefix}_L"], dt, L_inf, params[f"{prefix}_Ltau"])
        return {f"{prefix}_L": L_new}

    def compute_current(self, states, v, params):
        prefix = self._name
        gcaL = params[f"{prefix}_gcaLbar"] * states[f"{prefix}_L"]
        icaL = gcaL * params[f"{prefix}_gama"] * (v - params[f"{prefix}_ecaL"])
        il   = params[f"{prefix}_gl"] * (v - params[f"{prefix}_el"])
        return icaL + il

    def init_state(self, states, v, params, dt):
        prefix = self._name
        v2 = v - params[f"{prefix}_vtraub"]
        return {f"{prefix}_L": self._Linf(v2)}


