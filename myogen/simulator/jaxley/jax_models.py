"""
JAX-functional physiological models for closed-loop neuromuscular simulation.

Provides pure JAX step functions for muscle spindle, GTO, Hill muscle, and
joint dynamics. All functions are ``lax.scan``-compatible: no Python side
effects, no mutable class state — state is a plain dict of JAX arrays.

Pattern
-------
::

    state = model_init(...)
    new_state, outputs = model_step(state, inputs, params)

These functions are intended for two use cases:

1. **Python for loop** (Phase "now"): call each ``model_step`` inside a
   regular Python loop alongside Jaxley's ``step_fn``. The NumPy-based
   ``HillModel`` / ``SpindleModel`` wrappers can be replaced one at a time.

2. **``lax.scan``** (Phase "later"): combine all ``model_step`` functions and
   Jaxley's ``step_fn`` inside a single ``jax.lax.scan`` body for full GPU
   acceleration and automatic differentiation through the closed loop.

References
----------
- Mileusnic et al. (2006) — spindle model
- Lin & Crago (2002) — GTO model
- Fuglevand et al. (1993) — motor unit force model
- Hill (1938) — muscle mechanics
"""

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial


# ============================================================================
# SPIKE DETECTION — hard / surrogate-gradient / rate modes
# ============================================================================
#
# Spike detection is an upward threshold crossing:
#     spike = (v > v_th) & (prev_v <= v_th)
# The boolean cast that drives downstream conductances/force has zero gradient,
# which severs autodiff through the spike train. ``spike_detect`` provides three
# modes selectable at build time:
#
#   "hard"      — exact boolean crossing (bit-identical to the original model).
#                 Scientific default; no usable gradient through the spike.
#   "surrogate" — forward pass returns the exact hard crossing; the backward pass
#                 substitutes a smooth fast-sigmoid surrogate derivative on the
#                 crossing margin (SuperSpike, Zenke & Ganguli 2018). Straight-
#                 through, so the forward trajectory is unchanged.
#   "rate"      — replaces the discrete crossing with a continuous sigmoid of the
#                 margin. Fully smooth but the forward output differs from the
#                 spiking model (use for optimization where that is acceptable).
#
# ``SURROGATE_BETA`` controls the surrogate/rate slope (larger = sharper).

SURROGATE_BETA = 10.0


@partial(jax.custom_jvp, nondiff_argnums=())
def _straight_through_step(margin):
    """Heaviside step of ``margin`` with a fast-sigmoid surrogate gradient.

    Forward: ``(margin > 0)`` as float32 — exact. Backward (custom JVP): the
    fast-sigmoid derivative ``beta / (1 + beta*|margin|)**2`` so gradients flow
    through the crossing without changing the forward value.
    """
    return (margin > 0.0).astype(jnp.float32)


@_straight_through_step.defjvp
def _straight_through_step_jvp(primals, tangents):
    (margin,) = primals
    (dm,) = tangents
    primal_out = _straight_through_step(margin)
    beta = jnp.float32(SURROGATE_BETA)
    surrogate = beta / (1.0 + beta * jnp.abs(margin)) ** 2
    return primal_out, surrogate * dm


def spike_detect(v, prev_v, v_th, mode: str = "hard"):
    """Detect an upward crossing of ``v_th`` between ``prev_v`` and ``v``.

    Parameters
    ----------
    v, prev_v : arrays   Current and previous-step membrane voltage.
    v_th : float         Threshold in the cells' voltage frame.
    mode : {"hard","surrogate","rate"}

    Returns
    -------
    spikes : array
        ``mode="hard"`` → bool crossing (backward-compatible).
        ``mode="surrogate"`` → float32, forward-identical to hard, smooth grad.
        ``mode="rate"`` → float32 continuous sigmoid of the crossing margin.
    """
    if mode == "hard":
        return (v > v_th) & (prev_v <= v_th)
    margin = v - v_th
    if mode == "surrogate":
        # Gate the smooth crossing to the "upward" event so forward matches hard.
        rising = (prev_v <= v_th).astype(jnp.float32)
        return _straight_through_step(margin) * rising
    if mode == "rate":
        rising = jax.nn.sigmoid(jnp.float32(SURROGATE_BETA) * (v_th - prev_v))
        return jax.nn.sigmoid(jnp.float32(SURROGATE_BETA) * margin) * rising
    raise ValueError(f"unknown spike_mode {mode!r}")


# ============================================================================
# MUSCLE SPINDLE — Mileusnic et al. (2006)
# ============================================================================

def spindle_init() -> dict:
    """Return zeroed spindle state at rest."""
    return {
        "a_bag1": jnp.float32(0.0),
        "a_bag2": jnp.float32(0.0),
        "T":  jnp.zeros(3, dtype=jnp.float32),   # [Bag1, Bag2, Chain] tensions
        "dT": jnp.zeros(3, dtype=jnp.float32),   # tension rates
    }


def spindle_params_from_dict(spindle_params: dict) -> dict:
    """
    Convert a SpindleModel parameter dict to a JAX-ready params dict.

    Parameters
    ----------
    spindle_params : dict
        As returned by ``SpindleModel.create_default_spindle_parameters()``.

    Returns
    -------
    dict
        Same keys, all values cast to Python float (scalar JAX-friendly).
    """
    return {k: float(v) for k, v in spindle_params.items()}


def spindle_step(state: dict, L: float, V: float, A: float,
                 gd: float, gs: float, dt_s: float, p: dict):
    """
    One step of the Mileusnic et al. (2006) muscle spindle model.

    Parameters
    ----------
    state : dict
        ``{a_bag1, a_bag2, T: (3,), dT: (3,)}``
    L, V, A : float
        Muscle length [L0], velocity [L0/s], acceleration [L0/s²].
    gd, gs : float
        Dynamic and static gamma fusimotor drive [Hz].
    dt_s : float
        Timestep in seconds.
    p : dict
        Spindle parameters (from ``spindle_params_from_dict``).

    Returns
    -------
    new_state : dict
    (Ia, II) : tuple of float — primary and secondary afferent rates [Hz]
    """
    a_bag1 = state["a_bag1"]
    a_bag2 = state["a_bag2"]
    T  = state["T"]
    dT = state["dT"]

    P = p["P"]

    # --- Fusimotor activations (Hill-type saturation, Eq. 4-5) ---
    def _act(g, f0):
        return jnp.where(g > 0.0, g**P / (g**P + f0**P), 0.0)

    target_bag1 = _act(gd, p["fBag1"])
    target_bag2 = _act(gs, p["fBag2"])
    a_chain     = _act(gs, p["fChain"])

    # RK4 for bag activation ODE: da/dt = (target - a) / tau
    def _bag_rk4(a_prev, target, tau):
        k1 = (target - a_prev) / tau
        k2 = (target - (a_prev + dt_s / 2 * k1)) / tau
        k3 = (target - (a_prev + dt_s / 2 * k2)) / tau
        k4 = (target - (a_prev + dt_s       * k3)) / tau
        return a_prev + dt_s / 6 * (k1 + 2*k2 + 2*k3 + k4)

    new_a_bag1 = _bag_rk4(a_bag1, target_bag1, p["tau1"])
    new_a_bag2 = _bag_rk4(a_bag2, target_bag2, p["tau2"])

    # Per-fiber coefficients (shape 3)
    acts    = jnp.array([new_a_bag1, new_a_bag2, a_chain])
    b0      = jnp.array([p["b0Bag1"], p["b0Bag2"], p["b0Chain"]])
    b_fusi  = jnp.array([p["b1Bag1"], p["b2Bag2"], p["b2Chain"]])
    G       = jnp.array([p["G1"],     p["G2"],     p["G2Chain"]])
    b_coef  = b0 + b_fusi * acts       # (3,)
    gf      = G  * acts                # (3,) gamma force

    # --- Intrafusal fiber tension 2nd-order ODE (RK4, Eq. 1-3) ---
    K_SR = p["K_SR"]; M = p["M"]; K_PR = p["K_PR"]
    L0_SR = p["L0_SR"]; L0_PR = p["L0_PR"]; R = p["R"]
    a_exp = p["a"]; C_L = p["C_L"]; C_S = p["C_S"]

    def _tension_accel(T_val, z_val, b_c, gamma_f):
        vel_diff = V - z_val / K_SR
        # Nonlinear force-velocity (asymmetric lengthening/shortening)
        fv = jnp.where(
            vel_diff >= 0,
            C_L * b_c * jnp.abs(vel_diff)**a_exp * (L - L0_SR - T_val/K_SR - R),
            C_S * b_c * jnp.abs(vel_diff)**a_exp * (L - L0_SR - T_val/K_SR - R),
        )
        spring = K_PR * (L - L0_SR - T_val/K_SR - L0_PR)
        return K_SR / M * (fv + M*A + gamma_f - T_val + spring)

    def _fiber_rk4(T_prev, z_prev, b_c, gamma_f):
        k1y = z_prev
        k1z = _tension_accel(T_prev, z_prev, b_c, gamma_f)
        k2y = z_prev + dt_s/2 * k1z
        k2z = _tension_accel(T_prev + dt_s/2 * k1y, k2y, b_c, gamma_f)
        k3y = z_prev + dt_s/2 * k2z
        k3z = _tension_accel(T_prev + dt_s/2 * k2y, k3y, b_c, gamma_f)
        k4y = z_prev + dt_s * k3z
        k4z = _tension_accel(T_prev + dt_s * k3y, k4y, b_c, gamma_f)
        new_T = T_prev + dt_s/6 * (k1y + 2*k2y + 2*k3y + k4y)
        new_z = z_prev + dt_s/6 * (k1z + 2*k2z + 2*k3z + k4z)
        return new_T, new_z

    # Unrolled over 3 fibers (compile-time constant — fine for lax.scan)
    new_T  = jnp.zeros(3)
    new_dT = jnp.zeros(3)
    for fi in range(3):
        t_new, z_new = _fiber_rk4(T[fi], dT[fi], b_coef[fi], gf[fi])
        new_T  = new_T.at[fi].set(t_new)
        new_dT = new_dT.at[fi].set(z_new)

    # --- Afferent firing rates (Eq. 5-7) ---
    threshold = p["LN_SR"] - p["L0_SR"]

    ia_bag1  = p["gBag1"]    * jnp.maximum(0.0, new_T[0]/K_SR - threshold)
    ia_bag2  = p["gBag2A1"]  * jnp.maximum(0.0, new_T[1]/K_SR - threshold)
    ia_chain = p["gChainA1"] * jnp.maximum(0.0, new_T[2]/K_SR - threshold)

    # Occlusion (Eq. 6)
    B2C = ia_bag2 + ia_chain
    Ia  = jnp.where(B2C >= ia_bag1, B2C + p["S"]*ia_bag1, ia_bag1 + p["S"]*B2C)

    # Secondary afferent (Eq. 7): Bag2 + Chain only
    def _ii_contrib(fi, gain):
        sr = new_T[fi+1]/K_SR - threshold
        pr = L - new_T[fi+1]/K_SR - L0_SR - p["LN_PR"]
        return gain * jnp.maximum(
            0.0,
            p["X"] * p["Lsec"] / L0_SR * sr
            + (1 - p["X"]) * p["Lsec"] / p["L0_PR"] * pr,
        )

    II = _ii_contrib(0, p["gBag2A2"]) + _ii_contrib(1, p["gChainA2"])

    new_state = {"a_bag1": new_a_bag1, "a_bag2": new_a_bag2, "T": new_T, "dT": new_dT}
    return new_state, (Ia, II)


# ============================================================================
# GOLGI TENDON ORGAN — Lin & Crago (2002)
# ============================================================================

def gto_init() -> dict:
    """Return zeroed GTO state."""
    return {"prev_firing": jnp.float32(0.0)}


def gto_step(state: dict, force_N: float, p: dict):
    """
    One step of the GTO model.

    Parameters
    ----------
    state : dict  ``{prev_firing}``
    force_N : float  Muscle force in Newtons (absolute, not normalised).
    p : dict  ``{G1, G2, filter_alpha}``

    Returns
    -------
    new_state, Ib : Ib afferent firing rate [Hz]
    """
    force    = jnp.maximum(0.0, force_N)
    instant  = p["G1"] * jnp.log(force / p["G2"] + 1.0)
    alpha    = p["filter_alpha"]
    Ib       = (1.0 - alpha) * state["prev_firing"] + alpha * instant
    return {"prev_firing": Ib}, Ib


def gto_params_from_dict(gto_params: dict, filter_alpha: float = 0.3) -> dict:
    """Build GTO params dict from GolgiTendonOrganModel parameter dict."""
    return {
        "G1": float(gto_params["G1"]),
        "G2": float(gto_params["G2"]),
        "filter_alpha": float(filter_alpha),
    }


# ============================================================================
# JOINT DYNAMICS — second-order Euler
# ============================================================================

def joint_init(angle_deg: float = 0.0, velocity_deg_s: float = 0.0) -> dict:
    """Return initial joint state."""
    return {
        "angle_rad":     jnp.float32(np.radians(angle_deg)),
        "velocity_rad_s": jnp.float32(np.radians(velocity_deg_s)),
    }


def joint_step(state: dict, torque_Nm: float, dt_s: float, p: dict):
    """
    One step of second-order joint dynamics: I·α = τ - B·ω - K·θ.

    Parameters
    ----------
    state : dict  ``{angle_rad, velocity_rad_s}``
    torque_Nm : float  Applied muscle torque [N·m].
    dt_s : float  Timestep [s].
    p : dict  ``{inertia, damping, stiffness}``

    Returns
    -------
    new_state, angle_deg
    """
    spring  = -p["stiffness"] * state["angle_rad"]
    damping = -p["damping"]   * state["velocity_rad_s"]
    accel   = (torque_Nm + spring + damping) / p["inertia"]
    new_vel   = state["velocity_rad_s"] + accel   * dt_s
    new_angle = state["angle_rad"]      + new_vel * dt_s
    return {"angle_rad": new_angle, "velocity_rad_s": new_vel}, jnp.degrees(new_angle)


# ============================================================================
# HILL MUSCLE MODEL — Fuglevand et al. (1993) + Hill mechanics
# ============================================================================

def hill_init_params(hillD: dict, Ntype1: int, Ntype2: int, dt_ms: float) -> dict:
    """
    Pre-compute per-MU IIR constants and all static Hill model parameters.

    This function runs **once** at simulation setup and may call scipy
    (via ``ForceSatParams``). The returned dict never changes during simulation
    and is safe to use inside ``lax.scan`` as a static argument.

    Parameters
    ----------
    hillD : dict
        As returned by ``HillModel.create_default_muscle_parameters()``.
    Ntype1, Ntype2 : int
        Number of Type I and Type II motor units.
    dt_ms : float
        Timestep in milliseconds.

    Returns
    -------
    dict
        All parameters as JAX float32 arrays/scalars.
    """
    from myogen.simulator.jaxley.muscle import ForceSatParams

    N = Ntype1 + Ntype2
    fs = ForceSatParams(hillD, Ntype1, Ntype2)   # handles scipy internally

    T_ms   = fs.T        # contraction times (ms) — same units as dt_ms
    c      = fs.c
    tet_f  = fs.tetF
    P_amp  = fs.P        # peak force amplitudes

    # Normalised twitch amplitudes (Fuglevand Eq.)
    fP      = hillD["fP"]
    raw     = np.array([fP * P_amp[i] / tet_f[i] for i in range(N)])
    twiAmp  = raw / np.sum([fP * P_amp[i] for i in range(N)])

    # IIR coefficients for motor unit force filter
    # f[t+1] = A*f[t] - B*f_prev + C*spike   (dt and T both in ms)
    A_iir = 2.0 * np.exp(-dt_ms / T_ms)
    B_iir = np.exp(-2.0 * dt_ms / T_ms)
    C_iir = (dt_ms / T_ms) * np.exp(1.0 - dt_ms / T_ms)

    return {
        # Motor unit dynamics
        "A_iir":   jnp.array(A_iir,  dtype=jnp.float32),
        "B_iir":   jnp.array(B_iir,  dtype=jnp.float32),
        "C_iir":   jnp.array(C_iir,  dtype=jnp.float32),
        "c":       jnp.array(c,       dtype=jnp.float32),
        "twiAmp":  jnp.array(twiAmp, dtype=jnp.float32),
        "Ntype1":  int(Ntype1),
        "N":       int(N),
        # Global geometry
        "alfa0":   float(hillD["alfa0"]),
        "F0":      float(hillD["F0"]),
        "L0_m":    float(hillD["L0"]),
        "m":       float(hillD["m"]),
        "Kpe":     float(hillD["Kpe"]),
        "b_damp":  float(hillD["b"]),
        "Em_0":    float(hillD["Em_0"]),
        # Tendon
        "LT_0":    float(hillD["LT_0"]),
        "Kse":     float(hillD["Kse"]),
        "cT":      float(hillD["cT"]),
        "LT_r":    float(hillD["LT_r"]),
        # Force-length (Type I)
        "b1": float(hillD["b1"]), "o1": float(hillD["o1"]), "r1": float(hillD["r1"]),
        # Force-length (Type II)
        "b2": float(hillD["b2"]), "o2": float(hillD["o2"]), "r2": float(hillD["r2"]),
        # Force-velocity (Type I)
        "Vmax1": float(hillD["Vmax1"]),
        "av01":  float(hillD["av01"]),  "av11": float(hillD["av11"]),
        "av21":  float(hillD["av21"]),  "bv1":  float(hillD["bv1"]),
        "cv01":  float(hillD["cv01"]),  "cv11": float(hillD["cv11"]),
        # Force-velocity (Type II)
        "Vmax2": float(hillD["Vmax2"]),
        "av02":  float(hillD["av02"]),  "av12": float(hillD["av12"]),
        "av22":  float(hillD["av22"]),  "bv2":  float(hillD["bv2"]),
        "cv02":  float(hillD["cv02"]),  "cv12": float(hillD["cv12"]),
        # MTU polynomial coefficients (degree-4, 5 terms each)
        "Ak": jnp.array(hillD["Ak"], dtype=jnp.float32),
        "Bk": jnp.array(hillD["Bk"], dtype=jnp.float32),
        "dt_ms": float(dt_ms),
    }


def differentiable_twitch_params(RP, Tl, RT, fP, Ntype1, Ntype2, dt_ms, tetF):
    """Differentiable recompute of the Fuglevand twitch/IIR parameters (M4).

    ``hill_init_params`` derives ``twiAmp`` and the motor-unit IIR coefficients
    from the recruitment constants (``RP``, ``Tl``, ``RT``, ``fP``) via
    ``ForceSatParams`` — which uses scipy (``newton``/``lfilter``) and Python loops,
    breaking the gradient. This function reproduces the **closed-form** path in pure
    JAX so gradients flow ``{RP, Tl, RT, fP} → twiAmp, A_iir, B_iir, C_iir``.

    The tetanic-saturation term ``tetF`` (the scipy-Newton output) is passed in as a
    **frozen constant** — gradients w.r.t. the saturation-shape constants are the
    low-value tail and are intentionally not propagated (see docs/M1... / plan M4).
    With ``tetF`` frozen, ``twiAmp`` is still differentiable w.r.t. ``fP`` and the
    peak amplitudes ``P`` (hence ``RP``), and the IIR coefficients are fully
    differentiable w.r.t. ``Tl``/``RP``/``RT`` through the contraction times ``T``.

    Parameters
    ----------
    RP, Tl, RT, fP : float or 0-d array
        Fuglevand recruitment-range, longest twitch time, twitch-time range, and
        peak-force scale. Pass as JAX values to differentiate w.r.t. them.
    Ntype1, Ntype2 : int
        Motor-unit counts (static).
    dt_ms : float
        Timestep.
    tetF : array, shape (N,)
        Frozen tetanic-force normalisers from a one-time ``ForceSatParams`` call.

    Returns
    -------
    dict  ``{"twiAmp", "A_iir", "B_iir", "C_iir", "T"}`` — JAX arrays, differentiable.
    """
    N = Ntype1 + Ntype2
    idx = jnp.arange(1, N + 1, dtype=jnp.float32)

    # Peak twitch amplitudes: P[i] = exp(b*i), b = log(RP)/N     (fPeakAmp)
    b = jnp.log(RP) / N
    P = jnp.exp(b * idx)

    # Contraction times: T[i] = Tl * (1/P[i])**(1/c),  c = log(RP)/log(RT)  (fTwitchTime, durType==1)
    c_time = jnp.log(RP) / jnp.log(RT)
    T = Tl * (1.0 / P) ** (1.0 / c_time)

    # Normalised twitch amplitudes (tetF frozen)
    raw = fP * P / tetF
    twiAmp = raw / jnp.sum(fP * P)

    A_iir = 2.0 * jnp.exp(-dt_ms / T)
    B_iir = jnp.exp(-2.0 * dt_ms / T)
    C_iir = (dt_ms / T) * jnp.exp(1.0 - dt_ms / T)
    return {"twiAmp": twiAmp, "A_iir": A_iir, "B_iir": B_iir, "C_iir": C_iir, "T": T}


def hill_init_state(L0: float, N: int, max_delay_steps: int = 200) -> dict:
    """
    Return zeroed Hill model state.

    Parameters
    ----------
    L0 : float
        Initial normalised muscle length.
    N : int
        Total number of motor units (Ntype1 + Ntype2).
    max_delay_steps : int
        Size of axonal delay buffer (steps).  200 steps @ 0.1 ms/step = 20 ms
        max delay, which covers all physiological axonal delays.
    """
    return {
        "L":            jnp.float32(L0),
        "V":            jnp.float32(0.0),
        "f":            jnp.zeros(N, dtype=jnp.float32),
        "f_prev":       jnp.zeros(N, dtype=jnp.float32),
        "spike_buffer": jnp.zeros((N, max_delay_steps), dtype=jnp.float32),
    }


def hill_step(state: dict, mn_spikes, delay_steps, angle_rad: float, p: dict):
    """
    One step of the Hill muscle model.

    Parameters
    ----------
    state : dict
        ``{L, V, f, f_prev, spike_buffer}``
    mn_spikes : array-like, shape (N,), bool/float
        Which motor units have an MN spike THIS step (before delay).
    delay_steps : array-like, shape (N,), int
        Per-MU axonal delay in timesteps (≥ 1).
    angle_rad : float
        Current joint angle in radians.
    p : dict
        Pre-computed params from ``hill_init_params()``.

    Returns
    -------
    new_state : dict
    (L, V, force_norm, torque_norm) : all normalised to F0 / L0
    """
    L, V     = state["L"], state["V"]
    f        = state["f"]
    f_prev   = state["f_prev"]
    buf      = state["spike_buffer"]
    N        = p["N"]
    Ntype1   = p["Ntype1"]

    # 1. Read spikes that have arrived (front of delay buffer)
    current_spikes = buf[:, 0]

    # 2. Shift buffer left, clear last slot, write new spikes at delay positions
    #    delay D means write to slot D-1: after D rolls it reaches slot 0.
    mn_sp_f = jnp.array(mn_spikes, dtype=jnp.float32)
    shifted  = jnp.roll(buf, -1, axis=1).at[:, -1].set(0.0)
    new_buf  = shifted.at[jnp.arange(N), delay_steps - 1].add(mn_sp_f)

    # 3. Motor unit force IIR filter
    f_new = p["A_iir"] * f - p["B_iir"] * f_prev + p["C_iir"] * current_spikes

    # 4. Force saturation sigmoid
    def _sig(c_val, x):
        ec = jnp.exp(-c_val * x)
        return (1.0 - ec) / (1.0 + ec)

    fsat = p["twiAmp"] * _sig(p["c"], f_new)
    F1   = jnp.sum(fsat[:Ntype1])
    F2   = jnp.sum(fsat[Ntype1:])

    # 5. Hill force functions
    def _fL(LM, b, o, r):
        return jnp.exp(-(jnp.abs((LM**b - 1.0) / o))**r)

    def _fV(LM, vel, bv, av0, av1, av2, cv0, cv1, Vmax):
        conc = (bv - vel * (av0 + av1*LM + av2*LM**2)) / (bv + vel)
        ecc  = (Vmax - vel) / (Vmax + vel * (cv0 + cv1*LM))
        return jnp.where(vel > 0.0, conc, ecc)

    def _fCE(LM, vel):
        fce1 = _fL(LM, p["b1"], p["o1"], p["r1"]) * _fV(
            LM, vel, p["bv1"], p["av01"], p["av11"], p["av21"], p["cv01"], p["cv11"], p["Vmax1"])
        fce2 = _fL(LM, p["b2"], p["o2"], p["r2"]) * _fV(
            LM, vel, p["bv2"], p["av02"], p["av12"], p["av22"], p["cv02"], p["cv12"], p["Vmax2"])
        return F1 * fce1 + F2 * fce2

    def _fPE(LM, vel):
        return jnp.exp(p["Kpe"] * (LM - 1.0) / p["Em_0"]) / jnp.exp(p["Kpe"]) + p["b_damp"] * vel

    def _penn(LM):
        return jnp.arcsin(jnp.sin(p["alfa0"]) / LM)

    def _MTU(angle):
        return jnp.dot(p["Ak"], angle ** jnp.arange(5, dtype=jnp.float32))

    def _moment_arm(angle):
        return jnp.dot(p["Bk"], angle ** jnp.arange(5, dtype=jnp.float32))

    def _LT(angle, LM):
        term2 = LM * p["L0_m"] * jnp.cos(_penn(LM))
        return (_MTU(angle) - term2) / p["LT_0"]

    def _fSE(angle, LM):
        LT = _LT(angle, LM)
        return p["Kse"] * p["cT"] * jnp.log(jnp.exp((LT - p["LT_r"]) / p["cT"]) + 1.0)

    def _dVdt(LM, vel, angle):
        ratio   = p["F0"] / p["m"]
        tendon  = _fSE(angle, LM) * jnp.cos(_penn(LM))
        active  = (_fCE(LM, vel) + _fPE(LM, vel)) * jnp.cos(_penn(LM))**2
        return ratio * (tendon - active)

    # 6. RK4 for L and V
    dt_s = p["dt_ms"] * 1e-3
    k1L = V;            k1V = _dVdt(L, V, angle_rad)
    k2L = V + dt_s/2*k1V; k2V = _dVdt(L + dt_s/2*k1L, k2L, angle_rad)
    k3L = V + dt_s/2*k2V; k3V = _dVdt(L + dt_s/2*k2L, k3L, angle_rad)
    k4L = V + dt_s   *k3V; k4V = _dVdt(L + dt_s   *k3L, k4L, angle_rad)

    new_L = L + dt_s/6 * (k1L + 2*k2L + 2*k3L + k4L)
    new_V = V + dt_s/6 * (k1V + 2*k2V + 2*k3V + k4V)

    # 7. Force and torque (normalised)
    force  = _fSE(angle_rad, new_L)
    torque = force * _moment_arm(angle_rad)

    # 8. Per-fiber-type summed activations (for plotting)
    type1_act = F1
    type2_act = F2

    new_state = {
        "L":            new_L,
        "V":            new_V,
        "f":            f_new,
        "f_prev":       f,
        "spike_buffer": new_buf,
    }
    return new_state, (new_L, new_V, force, torque, type1_act, type2_act)


# ============================================================================
# POISSON SPIKE GENERATOR  (descending drive / DD cells)
# ============================================================================

def poisson_init(N_cells: int, N_batch: int, seed: int = 42, key=None) -> dict:
    """
    Initialise Poisson spike generator state.

    Mirrors ``_PoissonProcessGenerator__Jaxley``:
    ``yi += rate * dt_ms * 1e-3``; spike when ``yi >= thres``;
    ``thres ~ Gamma(N_batch, 1) / N_batch`` resampled after each spike.

    Parameters
    ----------
    N_cells : int   Number of independent generators.
    N_batch : int   Poisson batch size (higher → more regular; 1 = pure Poisson).
    seed : int   Fallback seed used only when ``key`` is None.
    key : jax PRNGKey, optional
        Explicit PRNG key. When provided it takes precedence over ``seed`` so the
        stochastic stream is controlled from the top-level ``run_jax`` call and is
        reproducible across JIT and runs.

    Returns
    -------
    dict  ``{yi: (N,), thres: (N,), keys: (N, 2)}``
    """
    key  = jax.random.PRNGKey(seed) if key is None else key
    keys = jax.random.split(key, N_cells)
    thres = (
        jax.vmap(lambda k: jax.random.gamma(k, jnp.float32(N_batch)))(keys)
        / jnp.float32(N_batch)
    )
    return {
        "yi":    jnp.zeros(N_cells, dtype=jnp.float32),
        "thres": thres.astype(jnp.float32),
        "keys":  keys,
    }


def poisson_step(state: dict, rate_hz, dt_ms: float, N_batch: int, mode: str = "hard") -> tuple:
    """
    One step of the Poisson spike generator.

    Parameters
    ----------
    state : dict  ``{yi, thres, keys}``
    rate_hz : float or (N,)  Instantaneous rate(s) [Hz].
    dt_ms : float  Timestep [ms].
    N_batch : int  Same as in ``poisson_init``.
    mode : {"hard", "pathwise", "rate"}
        ``"hard"`` — original discrete hazard crossing (no gradient w.r.t. rate).
        ``"pathwise"`` — reparameterised: the sampled threshold is frozen
        (``stop_gradient``) and the crossing uses a straight-through surrogate, so
        the forward spike train is identical to ``"hard"`` yet gradients flow
        ``rate → yi → spike``. Preserves stochastic timing.
        ``"rate"`` — emit the expected spikes-per-step ``rate*dt`` (continuous),
        giving an exact, low-variance gradient w.r.t. rate. Forward output is
        continuous rather than 0/1.

    Returns
    -------
    new_state, spikes
        ``spikes`` is bool in ``"hard"`` mode and float32 otherwise.
    """
    return _hazard_step(state, rate_hz, dt_ms, N_batch, mode)


def _hazard_step(state, rate_hz, dt_ms, gamma_conc, mode):
    """Shared hazard-accumulation update for the Poisson/Gamma generators.

    ``gamma_conc`` is the Gamma concentration used to resample the threshold
    (``N_batch`` for Poisson, ``shape`` for Gamma) — both use ``Gamma(conc)/conc``.
    """
    yi, thres, keys = state["yi"], state["thres"], state["keys"]
    split_keys = jax.vmap(jax.random.split)(keys)   # (N, 2, 2)
    new_keys   = split_keys[:, 0, :]
    sub_keys   = split_keys[:, 1, :]

    inc = jnp.asarray(rate_hz, dtype=jnp.float32) * jnp.float32(dt_ms * 1e-3)

    if mode == "rate":
        # Expected spikes per step; state advances but never emits discretely.
        yi_new = yi + inc
        spikes = jnp.broadcast_to(inc, jnp.shape(yi_new))
        return {"yi": yi_new, "thres": thres, "keys": new_keys}, spikes

    yi_new = yi + inc
    spikes_hard = yi_new >= thres                      # bool, drives state reset
    new_thres = (
        jax.vmap(lambda k: jax.random.gamma(k, jnp.float32(gamma_conc)))(sub_keys)
        / jnp.float32(gamma_conc)
    ).astype(jnp.float32)
    # Reset accumulator / resample threshold on a spike (forward-identical to hard).
    yi_reset   = jnp.where(spikes_hard, jnp.zeros_like(yi_new), yi_new)
    thres_out  = jnp.where(spikes_hard, new_thres, thres)

    if mode == "hard":
        spikes = spikes_hard
    elif mode == "pathwise":
        # Freeze the resampled threshold so the crossing margin's only parameter
        # dependence is through yi (hence through rate); surrogate grad on it.
        margin = yi_new - jax.lax.stop_gradient(thres)
        spikes = _straight_through_step(margin)
    else:
        raise ValueError(f"unknown generator mode {mode!r}")

    return {"yi": yi_reset, "thres": thres_out, "keys": new_keys}, spikes


# ============================================================================
# GAMMA SPIKE GENERATOR  (Ia / II / Ib afferent cells)
# ============================================================================

def gamma_init(N_cells: int, shape: float, seed: int = 42, key=None) -> dict:
    """
    Initialise Gamma-ISI spike generator state.

    Mirrors ``_GammaProcessGenerator__Cython``: hazard-accumulation threshold
    crossing.  Identical structure to ``poisson_init`` but threshold drawn from
    Gamma(shape, 1) / shape rather than Gamma(1, 1).

    Parameters
    ----------
    N_cells : int  Number of independent generators.
    shape : float  Gamma shape parameter (k).  1 = Poisson, >1 = more regular.
    seed : int   Fallback seed used only when ``key`` is None.
    key : jax PRNGKey, optional
        Explicit PRNG key; takes precedence over ``seed`` (see ``poisson_init``).

    Returns
    -------
    dict  ``{yi: (N,), thres: (N,), keys: (N, 2)}``
    """
    key  = jax.random.PRNGKey(seed) if key is None else key
    keys = jax.random.split(key, N_cells)
    thres = (
        jax.vmap(lambda k: jax.random.gamma(k, jnp.float32(shape)))(keys)
        / jnp.float32(shape)
    )
    return {
        "yi":    jnp.zeros(N_cells, dtype=jnp.float32),
        "thres": thres.astype(jnp.float32),
        "keys":  keys,
    }


def gamma_step(state: dict, rate_hz, dt_ms: float, shape: float, mode: str = "hard") -> tuple:
    """
    One step of the Gamma-ISI spike generator (hazard accumulation).

    Mirrors ``_GammaProcessGenerator__Cython.compute()``:

    ::

        yi += rate * dt_ms * 1e-3
        if yi >= thres:
            spike; yi = 0; thres ~ Gamma(shape, 1) / shape

    Parameters
    ----------
    state : dict  ``{yi, thres, keys}``
    rate_hz : float or (N,)  Rate(s) [Hz].  0 → silent.
    dt_ms : float  Timestep [ms].
    shape : float  Gamma shape parameter (same as in ``gamma_init``).
    mode : {"hard", "pathwise", "rate"}  See :func:`poisson_step`.

    Returns
    -------
    new_state, spikes
        ``spikes`` is bool in ``"hard"`` mode and float32 otherwise.
    """
    return _hazard_step(state, rate_hz, dt_ms, shape, mode)


# ============================================================================
# CONNECTIVITY UTILITIES
# ============================================================================

def make_connectivity_matrix(
    forward_map: dict, N_pre: int, N_post: int
) -> jnp.ndarray:
    """
    Convert a sparse ``{pre_idx: [post_idx, ...]}`` dict to a dense
    ``(N_pre, N_post)`` JAX float32 matrix.

    Usage inside scan::

        g_post += jnp.dot(spikes.astype(jnp.float32), mat) * weight
    """
    mat = np.zeros((N_pre, N_post), dtype=np.float32)
    for pre_idx, post_list in forward_map.items():
        for post_idx in post_list:
            mat[pre_idx, post_idx] = 1.0
    return jnp.array(mat)


# ============================================================================
# COMBINED PHYSIOLOGY STEP
# ============================================================================

def update_physiology(
    phys_carry: dict,
    mn_spikes,
    tap_dL_coeff_t: float,
    tap_dV_coeff_t: float,
    gDyn_t: float,
    gStat_t: float,
    delay_steps,
    hill_p: dict,
    spindle_p: dict,
    gto_p: dict,
    joint_p: dict,
    dt_ms: float,
    dt_s: float,
) -> tuple:
    """
    One step of the full physiology pipeline:
    Hill muscle → tap perturbation (spindle input only) →
    spindle → GTO → joint dynamics.

    Parameters
    ----------
    phys_carry : dict  ``{hill, spindle, gto, joint}``
    mn_spikes : (N_mn,) bool/float  MN spikes this step.
    tap_dL_coeff_t, tap_dV_coeff_t : float  Per-step tap coefficients.
    gDyn_t, gStat_t : float  Gamma fusimotor drives [Hz].
    delay_steps : (N_mn,) int32  Per-MU axonal delays [timesteps].
    hill_p, spindle_p, gto_p, joint_p : dict  Pre-computed parameter dicts.
    dt_ms, dt_s : float  Timestep.

    Returns
    -------
    new_phys_carry, (new_L, new_V, force_norm, torque_norm, type1_act, type2_act, Ia, II, Ib, angle_deg)
    """
    angle_rad    = phys_carry["joint"]["angle_rad"]
    prev_V_hill  = phys_carry["hill"]["V"]

    hill_state, (new_L, new_V, force_norm, torque_norm, type1_act, type2_act) = hill_step(
        phys_carry["hill"], mn_spikes, delay_steps, angle_rad, hill_p
    )

    L_sp = new_L * (jnp.float32(1.0) + jnp.float32(tap_dL_coeff_t))
    V_sp = new_V + new_L * jnp.float32(tap_dV_coeff_t)
    A_sp = (new_V - prev_V_hill) / jnp.float32(dt_s)

    spindle_state, (Ia, II) = spindle_step(
        phys_carry["spindle"], L_sp, V_sp, A_sp,
        gDyn_t, gStat_t, dt_s, spindle_p
    )

    force_N = jnp.float32(hill_p["F0"]) * force_norm
    gto_state, Ib = gto_step(phys_carry["gto"], force_N, gto_p)

    torque_Nm = jnp.float32(hill_p["F0"]) * torque_norm
    joint_state, angle_deg = joint_step(phys_carry["joint"], torque_Nm, dt_s, joint_p)

    new_phys = {
        "hill":    hill_state,
        "spindle": spindle_state,
        "gto":     gto_state,
        "joint":   joint_state,
    }
    return new_phys, (new_L, new_V, force_norm, torque_norm, type1_act, type2_act, Ia, II, Ib, angle_deg)


# ============================================================================
# lax.scan STEP FUNCTION FACTORY
# ============================================================================

def make_scan_step(
    jaxley_step_fn,
    jaxley_params,
    external_inds,
    rec_inds,
    n_gii: int,
    n_gib: int,
    n_mn: int,
    ia_rts,
    ii_rts,
    ib_rts,
    ia_shape: float,
    ii_shape: float,
    ib_shape: float,
    dd_N_batch: int,
    dd_to_mn_mat,
    ia_to_mn_mat,
    ii_to_gii_mat,
    ib_to_gib_mat,
    ia_delay_steps_arr,
    ii_delay_steps_arr,
    ib_delay_steps_arr,
    delay_steps,
    hill_p: dict,
    spindle_p: dict,
    gto_p: dict,
    joint_p: dict,
    base_dd_weight: float,
    base_ia_weight: float,
    in_weight: float,
    e_exc: float,
    v_rest: float,
    mn_current_scale,
    tau_syn_decay: float,
    dt_ms: float,
    dt_s: float,
    e_exc_mn: float | None = None,
    mn_spike_threshold_mV: float = 0.0,
    spike_mode: str = "hard",
):
    """
    Factory returning a ``lax.scan``-compatible step function.

    All static parameters are closed over; the returned ``scan_step``
    has the fixed signature ``(new_carry, output_t) = scan_step(carry, input_t)``.

    Carry structure
    ---------------
    ::

        {
          "neural":    Jaxley states,
          "phys":      {"hill", "spindle", "gto", "joint"},
          "g_dd":      (n_mn,),  "g_ia": (n_mn,),
          "g_ii":      (n_gii,), "g_ib": (n_gib,),
          "prev_v":    (n_gii+n_gib+n_mn,),
          "dd_st":     poisson_init output,
          "ia_st / ii_st / ib_st": gamma_init outputs,
          "prev_Iay / prev_IIy / prev_Iby": float32 scalars,
          "ia_delay_buf":  (nIa, max_ia_delay_steps),   # per-cell FIFO delay queues
          "ii_delay_buf":  (nII, max_ii_delay_steps),
          "ib_delay_buf":  (nIb, max_ib_delay_steps),
        }

    Input structure (per step, stacked by lax.scan)
    ------------------------------------------------
    ::

        {"DDdrive", "gDyn", "gStat", "tap_dL", "tap_dV"}  — all float32 scalars

    Output structure (stacked → shape (T, ...) after scan)
    -------------------------------------------------------
    ::

        {"v_mn", "mn_spikes", "gii_spikes", "gib_spikes",
         "ia_spikes", "ii_spikes", "ib_spikes",
         "L", "force", "torque", "Iay", "IIy", "Iby", "angle_deg"}

    Notes
    -----
    Afferent axonal delays are implemented as per-cell FIFO shift-register
    queues in the carry.  Each afferent cell has its own delay slot so spike
    timing faithfully reflects individual conduction velocities.
    """
    _rec_inds        = jnp.array(rec_inds,         dtype=jnp.int32)
    _ia_rts          = jnp.array(ia_rts,           dtype=jnp.float32)
    _ii_rts          = jnp.array(ii_rts,           dtype=jnp.float32)
    _ib_rts          = jnp.array(ib_rts,           dtype=jnp.float32)
    _delay_steps     = jnp.array(delay_steps,      dtype=jnp.int32)
    _ia_delay_steps  = jnp.array(ia_delay_steps_arr, dtype=jnp.int32)
    _ii_delay_steps  = jnp.array(ii_delay_steps_arr, dtype=jnp.int32)
    _ib_delay_steps  = jnp.array(ib_delay_steps_arr, dtype=jnp.int32)
    _nIa             = len(ia_delay_steps_arr)
    _nII             = len(ii_delay_steps_arr)
    _nIb             = len(ib_delay_steps_arr)
    _dd_to_mn        = jnp.asarray(dd_to_mn_mat,   dtype=jnp.float32)
    _ia_to_mn        = jnp.asarray(ia_to_mn_mat,   dtype=jnp.float32)
    _ii_to_gii       = jnp.asarray(ii_to_gii_mat,  dtype=jnp.float32)
    _ib_to_gib       = jnp.asarray(ib_to_gib_mat,  dtype=jnp.float32)
    _mn_scl          = jnp.asarray(mn_current_scale, dtype=jnp.float32)
    _base_dd         = jnp.float32(base_dd_weight)
    _base_ia         = jnp.float32(base_ia_weight)
    _in_w            = jnp.float32(in_weight)
    _e_exc           = jnp.float32(e_exc)
    _v_rest          = jnp.float32(v_rest)
    # ``e_exc_mn``: excitatory reversal for the MOTOR-NEURON section of the
    # combined network. Optional; defaults to ``e_exc`` (single-frame behavior
    # for backward compatibility with older callers). When the MN cells live in
    # a different voltage frame from the interneurons (e.g. NERLab MNs at
    # V_rest ≈ 0 mV mixed with modern-frame gII/gIb at V_rest ≈ -70 mV), pass
    # the NERLab AMPA reversal (~+70 mV) here so the per-step driving force
    # ``df_mn = e_exc_mn − V_mn`` evaluates correctly. ``mn_spike_threshold_mV``
    # similarly lets the MN spike detector use a NERLab-appropriate +50 mV
    # crossing instead of the 0 mV crossing used by the modern-frame cells.
    _e_exc_mn        = jnp.float32(e_exc_mn if e_exc_mn is not None else e_exc)
    _mn_v_th         = jnp.float32(mn_spike_threshold_mV)
    _decay           = jnp.float32(tau_syn_decay)
    _dt_ms           = float(dt_ms)
    _dt_s            = float(dt_s)
    _n_gii           = n_gii
    _n_gib           = n_gib
    _spike_mode      = spike_mode
    # Map the loop-level spike mode onto the stochastic-generator mode so the
    # descending/afferent inputs are differentiated consistently with the
    # neural spike detectors: hard→hard, surrogate→pathwise, rate→rate.
    _gen_mode        = {"hard": "hard", "surrogate": "pathwise", "rate": "rate"}[spike_mode]

    def scan_step(carry, input_t):
        # ------------------------------------------------------------------ #
        # 1. Decay conductances, then read arriving delayed contributions.
        #    Per-cell delay buffers: row i = FIFO for afferent cell i.
        #    Slot 0 holds contributions ready to apply this step.
        # ------------------------------------------------------------------ #
        ia_arrivals = jnp.dot(carry["ia_delay_buf"][:, 0], _ia_to_mn) * _base_ia  # (n_mn,)
        ii_arrivals = jnp.dot(carry["ii_delay_buf"][:, 0], _ii_to_gii) * _in_w    # (n_gii,)
        ib_arrivals = jnp.dot(carry["ib_delay_buf"][:, 0], _ib_to_gib) * _in_w    # (n_gib,)
        g_dd = carry["g_dd"] * _decay
        g_ia = carry["g_ia"] * _decay + ia_arrivals
        g_ii = carry["g_ii"] * _decay + ii_arrivals
        g_ib = carry["g_ib"] * _decay + ib_arrivals

        # ------------------------------------------------------------------ #
        # 2. DD Poisson step
        # ------------------------------------------------------------------ #
        dd_st_new, dd_spikes = poisson_step(
            carry["dd_st"], input_t["DDdrive"], _dt_ms, dd_N_batch, _gen_mode
        )
        g_dd = g_dd + jnp.dot(dd_spikes.astype(jnp.float32), _dd_to_mn) * _base_dd

        # ------------------------------------------------------------------ #
        # 3. Afferent Gamma steps (driven by prev-step rates)
        #    Then enqueue new contributions into delay FIFOs.
        # ------------------------------------------------------------------ #
        ia_rates = jnp.where(carry["prev_Iay"] >= _ia_rts, carry["prev_Iay"], jnp.float32(0.0))
        ii_rates = jnp.where(carry["prev_IIy"] >= _ii_rts, carry["prev_IIy"], jnp.float32(0.0))
        ib_rates = jnp.where(carry["prev_Iby"] >= _ib_rts, carry["prev_Iby"], jnp.float32(0.0))

        ia_st_new, ia_spikes = gamma_step(carry["ia_st"], ia_rates, _dt_ms, ia_shape, _gen_mode)
        ii_st_new, ii_spikes = gamma_step(carry["ii_st"], ii_rates, _dt_ms, ii_shape, _gen_mode)
        ib_st_new, ib_spikes = gamma_step(carry["ib_st"], ib_rates, _dt_ms, ib_shape, _gen_mode)

        # Per-cell delay FIFOs: shift left, clear wrap slot, write new spikes
        # at each cell's own delay position (delay d → slot d-1 after shift).
        # After d more rolls the spike reaches slot 0 where it is read.
        ia_sp_f = ia_spikes.astype(jnp.float32)
        ii_sp_f = ii_spikes.astype(jnp.float32)
        ib_sp_f = ib_spikes.astype(jnp.float32)
        new_ia_delay_buf = (
            jnp.roll(carry["ia_delay_buf"], -1, axis=1).at[:, -1].set(0.0)
            .at[jnp.arange(_nIa), _ia_delay_steps - 1].add(ia_sp_f)
        )
        new_ii_delay_buf = (
            jnp.roll(carry["ii_delay_buf"], -1, axis=1).at[:, -1].set(0.0)
            .at[jnp.arange(_nII), _ii_delay_steps - 1].add(ii_sp_f)
        )
        new_ib_delay_buf = (
            jnp.roll(carry["ib_delay_buf"], -1, axis=1).at[:, -1].set(0.0)
            .at[jnp.arange(_nIb), _ib_delay_steps - 1].add(ib_sp_f)
        )

        # ------------------------------------------------------------------ #
        # 4. Build stimulus currents [nA] = g [µS] × driving_force [mV]
        #    Use live (prev-step) voltages for driving force.
        # ------------------------------------------------------------------ #
        df_gii = _e_exc    - carry["prev_v"][:_n_gii]
        df_gib = _e_exc    - carry["prev_v"][_n_gii: _n_gii + _n_gib]
        df_mn  = _e_exc_mn - carry["prev_v"][_n_gii + _n_gib:]   # NERLab-aware
        stims = jnp.concatenate([
            g_ii * df_gii,
            g_ib * df_gib,
            (g_dd + g_ia) * _mn_scl * df_mn,
        ])

        # ------------------------------------------------------------------ #
        # 5. Advance Jaxley neural network one step
        # ------------------------------------------------------------------ #
        new_neural = jaxley_step_fn(
            carry["neural"], jaxley_params, {"i": stims}, external_inds,
            delta_t=_dt_ms,
        )

        # ------------------------------------------------------------------ #
        # 6. Extract voltages (all recordings assumed to be "v")
        # ------------------------------------------------------------------ #
        v_all  = new_neural["v"][_rec_inds]
        v_gii  = v_all[:_n_gii]
        v_gib  = v_all[_n_gii: _n_gii + _n_gib]
        v_mn   = v_all[_n_gii + _n_gib:]

        prev_v_gii = carry["prev_v"][:_n_gii]
        prev_v_gib = carry["prev_v"][_n_gii: _n_gii + _n_gib]
        prev_v_mn  = carry["prev_v"][_n_gii + _n_gib:]

        # ------------------------------------------------------------------ #
        # 7. Spike detection (upward crossing; hard / surrogate / rate mode)
        # ------------------------------------------------------------------ #
        mn_spikes_now  = spike_detect(v_mn,  prev_v_mn,  _mn_v_th,        _spike_mode)
        gii_spikes_now = spike_detect(v_gii, prev_v_gii, jnp.float32(0.0), _spike_mode)
        gib_spikes_now = spike_detect(v_gib, prev_v_gib, jnp.float32(0.0), _spike_mode)

        # ------------------------------------------------------------------ #
        # 8. Physiology step
        # ------------------------------------------------------------------ #
        new_phys, (new_L, new_V, force_norm, torque_norm, type1_act, type2_act, Ia, II, Ib, angle_deg) = (
            update_physiology(
                carry["phys"],
                mn_spikes_now,
                input_t["tap_dL"],
                input_t["tap_dV"],
                input_t["gDyn"],
                input_t["gStat"],
                _delay_steps,
                hill_p, spindle_p, gto_p, joint_p,
                _dt_ms, _dt_s,
            )
        )

        # ------------------------------------------------------------------ #
        # 9. New carry
        # ------------------------------------------------------------------ #
        new_carry = {
            "neural":    new_neural,
            "phys":      new_phys,
            "g_dd":      g_dd,
            "g_ia":      g_ia,
            "g_ii":      g_ii,
            "g_ib":      g_ib,
            "prev_v":    v_all,
            "dd_st":     dd_st_new,
            "ia_st":     ia_st_new,
            "ii_st":     ii_st_new,
            "ib_st":     ib_st_new,
            "prev_Iay":      jnp.float32(Ia),
            "prev_IIy":      jnp.float32(II),
            "prev_Iby":      jnp.float32(Ib),
            "ia_delay_buf":  new_ia_delay_buf,
            "ii_delay_buf":  new_ii_delay_buf,
            "ib_delay_buf":  new_ib_delay_buf,
        }

        # ------------------------------------------------------------------ #
        # 10. Per-step outputs (lax.scan stacks these into (T, ...) arrays)
        # ------------------------------------------------------------------ #
        output_t = {
            "v_mn":       v_mn,
            "mn_spikes":  mn_spikes_now,
            "gii_spikes": gii_spikes_now,
            "gib_spikes": gib_spikes_now,
            "dd_spikes":  dd_spikes,
            "ia_spikes":  ia_spikes,
            "ii_spikes":  ii_spikes,
            "ib_spikes":  ib_spikes,
            "L":          new_L,
            "force":      force_norm,
            "torque":     torque_norm,
            "type1_act":  type1_act,
            "type2_act":  type2_act,
            "Iay":        Ia,
            "IIy":        II,
            "Iby":        Ib,
            "angle_deg":  angle_deg,
            # Spindle internal states (for plotting bag/chain activation and tensions)
            "bag1_act":   new_phys["spindle"]["a_bag1"],
            "bag2_act":   new_phys["spindle"]["a_bag2"],
            "spin_T":     new_phys["spindle"]["T"],       # (3,) [Bag1, Bag2, Chain]
        }

        return new_carry, output_t

    return scan_step
