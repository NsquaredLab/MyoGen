"""
Custom Ion Channel Models for Motor Neuron Simulations - Jaxley Backend.

This module provides JAX-compatible implementations of the ion channels used
in the Powers2017 motor neuron model, originally implemented in NMODL for NEURON.

Channels implemented:
- Na3rp: Fast sodium with slow inactivation (action potentials)
- Naps: Persistent sodium (bistability, PICs)
- KdrRL: Delayed rectifier potassium (repolarization)
- MAHP: Calcium-dependent potassium with calcium dynamics (mAHP)
- Gh: Hyperpolarization-activated cation current (sag)
- LCaInact: L-type calcium with inactivation (dendritic PICs)

All channels follow the Jaxley Channel API and can be inserted into Jaxley cells.

References:
- Powers et al. (2017): Motor neuron ion channel models
- Booth et al. (1997): L-type calcium channel kinetics
- Fleidervish et al.: Slow sodium inactivation
"""

from typing import Dict, Optional, Tuple
import jax.numpy as jnp
from jax import lax
from jaxley.channels import Channel
from jaxley.solver_gate import exponential_euler, save_exp


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def safe_exp(x: jnp.ndarray, clip_val: float = 100.0) -> jnp.ndarray:
    """Exponential with clipping to prevent overflow."""
    return jnp.exp(jnp.clip(x, -clip_val, clip_val))


def vtrap(x: jnp.ndarray, y: float) -> jnp.ndarray:
    """
    Trapping function for rate equations to avoid division by zero.
    
    When |x/y| < 1e-6: returns y * (1 - x/y/2)
    Otherwise: returns x / (exp(x/y) - 1)
    """
    ratio = x / y
    return jnp.where(
        jnp.abs(ratio) < 1e-6,
        y * (1.0 - ratio / 2.0),
        x / (save_exp(ratio) - 1.0)
    )


def trap0(v: jnp.ndarray, th: float, a: float, q: float) -> jnp.ndarray:
    """
    Rate equation trap function from na3rp.mod.

    Parameters
    ----------
    v : membrane potential (mV)
    th : threshold voltage (mV)
    a : rate constant
    q : slope factor (mV)
    """
    diff = v - th
    # Use safe_exp with clipping to prevent overflow, matches NEURON's exp behavior
    # save_exp from jaxley may have different clipping behavior
    exp_term = safe_exp(-diff / q)
    return jnp.where(
        jnp.abs(diff) > 1e-6,
        a * diff / (1.0 - exp_term),
        a * q
    )


# =============================================================================
# NA3RP - FAST SODIUM CHANNEL WITH SLOW INACTIVATION
# =============================================================================

class Na3rp(Channel):
    """
    Fast sodium channel with slow inactivation (na3rp).
    
    Three-state gating: m³hs where:
    - m: fast activation
    - h: fast inactivation  
    - s: slow inactivation (Fleidervish et al.)
    
    Parameters from Magee/Migliore model with Powers modifications.
    """
    
    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True  # Required for Jaxley 0.5.0+
        super().__init__(name)
        
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gbar": 0.010,  # S/cm²
            f"{prefix}_sh": 8.0,       # mV - threshold shift
            f"{prefix}_ar": 1.0,       # slow inact recovery (1=none, 0=max)
            f"{prefix}_eNa": 60.0,     # Sodium reversal potential
            f"{prefix}_thinf": -50.0,  # mV - inactivation inf half-voltage (NEURON: thinf_na3rp)
            f"{prefix}_qinf": 4.0,     # mV - inactivation inf slope (NEURON: qinf_na3rp)
        }
        self.channel_states = {
            f"{prefix}_m": 0.0,
            f"{prefix}_h": 1.0,
            f"{prefix}_s": 1.0,
        }
        self.current_name = f"i_Na3rp"

        # Rate parameters from NMODL
        self.tha = -30.0    # mV - activation half-voltage
        self.qa = 7.2       # mV - activation slope
        self.Ra = 0.4       # /ms - activation rate
        self.Rb = 0.124     # /ms - deactivation rate

        self.thi1 = -45.0   # mV - inactivation half-voltage
        self.thi2 = -45.0   # mV
        self.qd = 1.5       # mV - inactivation slope
        self.qg = 1.5       # mV
        self.Rd = 0.03      # /ms - inactivation rate
        self.Rg = 0.01      # /ms - recovery rate
        
        # Slow inactivation parameters
        self.a0s = 0.001    # /ms
        self.b0s = 0.0034   # /ms
        self.asvh = -85.0   # mV
        self.bsvh = -17.0   # mV
        self.avs = 30.0     # mV
        self.bvs = 10.0     # mV
        
        self.mmin = 0.02    # ms - minimum tau_m
        self.hmin = 0.5     # ms - minimum tau_h
        self.q10 = 2.0
    
    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt,
        v,
        params: Dict[str, jnp.ndarray],
    ) -> Dict[str, jnp.ndarray]:
        """Update gating variables using exponential Euler."""
        prefix = self._name
        m = states[f"{prefix}_m"]
        h = states[f"{prefix}_h"]
        s = states[f"{prefix}_s"]

        sh = params[f"{prefix}_sh"]
        ar = params[f"{prefix}_ar"]
        thinf = params[f"{prefix}_thinf"]
        qinf = params[f"{prefix}_qinf"]

        celsius = 37.0
        qt = self.q10 ** ((celsius - 24.0) / 10.0)

        # m (activation)
        a_m = trap0(v, self.tha + sh, self.Ra, self.qa)
        b_m = trap0(-v, -self.tha - sh, self.Rb, self.qa)
        tau_m = jnp.maximum(1.0 / (a_m + b_m) / qt, self.mmin)
        m_inf = a_m / (a_m + b_m)

        # h (fast inactivation) - uses thinf and qinf from params
        a_h = trap0(v, self.thi1 + sh, self.Rd, self.qd)
        b_h = trap0(-v, -self.thi2 - sh, self.Rg, self.qg)
        tau_h = jnp.maximum(1.0 / (a_h + b_h) / qt, self.hmin)
        h_inf = 1.0 / (1.0 + save_exp((v - thinf - sh) / qinf))
        
        # s (slow inactivation)
        alps = self.a0s * save_exp((self.asvh - v) / self.avs)
        bets = self.b0s / (save_exp((self.bsvh - v) / self.bvs) + 1.0)
        tau_s = 1.0 / (alps + bets)
        c = alps * tau_s
        s_inf = c + ar * (1.0 - c)
        
        # Exponential Euler integration
        m_new = exponential_euler(m, dt, m_inf, tau_m)
        h_new = exponential_euler(h, dt, h_inf, tau_h)
        s_new = exponential_euler(s, dt, s_inf, tau_s)
        
        return {
            f"{prefix}_m": m_new,
            f"{prefix}_h": h_new,
            f"{prefix}_s": s_new,
        }
    
    def compute_current(
        self,
        states: Dict[str, jnp.ndarray],
        v,
        params: Dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Compute sodium current: I = gbar * m³ * h * s * (v - E_Na)."""
        prefix = self._name
        m = states[f"{prefix}_m"]
        h = states[f"{prefix}_h"]
        s = states[f"{prefix}_s"]
        gbar = params[f"{prefix}_gbar"]
        e_na = params[f"{prefix}_eNa"]
        
        g = gbar * (m ** 3) * h * s
        return g * (v - e_na)
    
    def init_state(
        self,
        states: Dict[str, jnp.ndarray],
        v,
        params: Dict[str, jnp.ndarray],
        delta_t,
    ) -> Dict[str, jnp.ndarray]:
        """Initialize states to steady-state values."""
        prefix = self._name
        sh = params[f"{prefix}_sh"]
        ar = params[f"{prefix}_ar"]
        thinf = params[f"{prefix}_thinf"]
        qinf = params[f"{prefix}_qinf"]

        # m (activation)
        a_m = trap0(v, self.tha + sh, self.Ra, self.qa)
        b_m = trap0(-v, -self.tha - sh, self.Rb, self.qa)
        m_inf = a_m / (a_m + b_m)

        # h (fast inactivation) - uses thinf and qinf from params
        h_inf = 1.0 / (1.0 + save_exp((v - thinf - sh) / qinf))
        
        # s (slow inactivation)
        alps = self.a0s * save_exp((self.asvh - v) / self.avs)
        bets = self.b0s / (save_exp((self.bsvh - v) / self.bvs) + 1.0)
        tau_s = 1.0 / (alps + bets)
        c = alps * tau_s
        s_inf = c + ar * (1.0 - c)
        
        return {
            f"{prefix}_m": m_inf,
            f"{prefix}_h": h_inf,
            f"{prefix}_s": s_inf,
        }


# =============================================================================
# NAPS - PERSISTENT SODIUM CHANNEL
# =============================================================================

class Naps(Channel):
    """
    Persistent sodium channel (naps).
    
    Two-state gating: ms where:
    - m: activation (no inactivation at normal voltages)
    - s: slow inactivation
    
    Important for subthreshold amplification and bistability.
    """
    
    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True  # Required for Jaxley 0.5.0+
        super().__init__(name)
        
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gbar": 0.0052085,  # S/cm²
            f"{prefix}_sh": 0.0,           # mV - threshold shift
            f"{prefix}_ar": 1.0,           # slow inact (1=none)
            f"{prefix}_vslope": 6.8,       # mV - activation slope
            f"{prefix}_eNa": 60.0,         # Sodium reversal potential
            f"{prefix}_asvh": -85.0,       # mV - slow inact half-act voltage
            f"{prefix}_bsvh": -17.0,       # mV - slow inact half-deact voltage
        }
        self.channel_states = {
            f"{prefix}_m": 0.0,
            f"{prefix}_s": 1.0,
        }
        self.current_name = f"i_Naps"
        
        self.mtau = 1.0  # ms - fixed activation time constant
        
        # Slow inactivation parameters (same as na3rp)
        self.a0s = 0.001
        self.b0s = 0.0034
        self.asvh = -85.0
        self.bsvh = -17.0
        self.avs = 30.0
        self.bvs = 10.0
    
    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt,
        v,
        params: Dict[str, jnp.ndarray],
    ) -> Dict[str, jnp.ndarray]:
        """Update gating variables."""
        prefix = self._name
        m = states[f"{prefix}_m"]
        s = states[f"{prefix}_s"]
        
        sh = params[f"{prefix}_sh"]
        ar = params[f"{prefix}_ar"]
        vslope = params[f"{prefix}_vslope"]
        asvh = params[f"{prefix}_asvh"]
        bsvh = params[f"{prefix}_bsvh"]

        # m (activation) - simple Boltzmann
        m_inf = 1.0 / (1.0 + save_exp(-(v + 52.3 - sh) / vslope))
        tau_m = self.mtau

        # s (slow inactivation)
        alps = self.a0s * save_exp((asvh - v) / self.avs)
        bets = self.b0s / (save_exp((bsvh - v) / self.bvs) + 1.0)
        tau_s = 1.0 / (alps + bets)
        c = alps * tau_s
        s_inf = c + ar * (1.0 - c)
        
        m_new = exponential_euler(m, dt, m_inf, tau_m)
        s_new = exponential_euler(s, dt, s_inf, tau_s)
        
        return {
            f"{prefix}_m": m_new,
            f"{prefix}_s": s_new,
        }
    
    def compute_current(
        self,
        states: Dict[str, jnp.ndarray],
        v,
        params: Dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Compute persistent sodium current: I = gbar * m * s * (v - E_Na)."""
        prefix = self._name
        m = states[f"{prefix}_m"]
        s = states[f"{prefix}_s"]
        gbar = params[f"{prefix}_gbar"]
        e_na = params[f"{prefix}_eNa"]
        
        g = gbar * m * s
        return g * (v - e_na)
    
    def init_state(
        self,
        states: Dict[str, jnp.ndarray],
        v,
        params: Dict[str, jnp.ndarray],
        delta_t,
    ) -> Dict[str, jnp.ndarray]:
        """Initialize to steady-state."""
        prefix = self._name
        sh = params[f"{prefix}_sh"]
        ar = params[f"{prefix}_ar"]
        vslope = params[f"{prefix}_vslope"]
        asvh = params[f"{prefix}_asvh"]
        bsvh = params[f"{prefix}_bsvh"]

        m_inf = 1.0 / (1.0 + save_exp(-(v + 52.3 - sh) / vslope))

        alps = self.a0s * save_exp((asvh - v) / self.avs)
        bets = self.b0s / (save_exp((bsvh - v) / self.bvs) + 1.0)
        tau_s = 1.0 / (alps + bets)
        c = alps * tau_s
        s_inf = c + ar * (1.0 - c)
        
        return {
            f"{prefix}_m": m_inf,
            f"{prefix}_s": s_inf,
        }


# =============================================================================
# KDRRL - DELAYED RECTIFIER POTASSIUM CHANNEL
# =============================================================================

class KdrRL(Channel):
    """
    Delayed rectifier potassium channel (kdrRL).
    
    Single gating variable with 4th power: m⁴
    Responsible for action potential repolarization.
    """
    
    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True  # Required for Jaxley 0.5.0+
        super().__init__(name)
        
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gbar": 0.1,       # S/cm²
            f"{prefix}_mVh": -25.0,      # mV - half-activation voltage
            f"{prefix}_mslp": 20.0,      # mV - activation slope
            f"{prefix}_eK": -80.0,       # Potassium reversal potential
            f"{prefix}_tmin": 1.4,       # ms - minimum tau (NEURON global: 0.8)
            f"{prefix}_taumax": 11.9,    # ms - maximum tau (NEURON global: 20.0)
        }
        self.channel_states = {
            f"{prefix}_m": 0.0,
        }
        self.current_name = f"i_KdrRL"

        # Tau parameters (voltage dependence, not modified by NEURON)
        self.tVh = -39.0     # mV - tau half-voltage
        self.tslp = 5.5      # mV - tau slope
    
    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt,
        v,
        params: Dict[str, jnp.ndarray],
    ) -> Dict[str, jnp.ndarray]:
        """Update activation gating variable."""
        prefix = self._name
        m = states[f"{prefix}_m"]

        mVh = params[f"{prefix}_mVh"]
        mslp = params[f"{prefix}_mslp"]
        tmin = params[f"{prefix}_tmin"]
        taumax = params[f"{prefix}_taumax"]

        m_inf = 1.0 / (1.0 + save_exp(-(v - mVh) / mslp))

        b = save_exp((v - self.tVh) / self.tslp)
        f = (1.0 + b) ** 2
        tau_m = tmin + taumax * b / f

        m_new = exponential_euler(m, dt, m_inf, tau_m)

        return {f"{prefix}_m": m_new}
    
    def compute_current(
        self,
        states: Dict[str, jnp.ndarray],
        v,
        params: Dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Compute potassium current: I = gbar * m⁴ * (v - E_K)."""
        prefix = self._name
        m = states[f"{prefix}_m"]
        gbar = params[f"{prefix}_gbar"]
        e_k = params[f"{prefix}_eK"]
        
        g = gbar * (m ** 4)
        return g * (v - e_k)
    
    def init_state(
        self,
        states: Dict[str, jnp.ndarray],
        v,
        params: Dict[str, jnp.ndarray],
        delta_t,
    ) -> Dict[str, jnp.ndarray]:
        """Initialize to steady-state."""
        prefix = self._name
        mVh = params[f"{prefix}_mVh"]
        mslp = params[f"{prefix}_mslp"]
        
        m_inf = 1.0 / (1.0 + save_exp(-(v - mVh) / mslp))
        
        return {f"{prefix}_m": m_inf}


# =============================================================================
# MAHP - CALCIUM-DEPENDENT POTASSIUM CHANNEL (MEDIUM AHP)
# =============================================================================

class MAHP(Channel):
    """
    Calcium-dependent potassium channel responsible for medium AHP.
    
    Includes a simplified calcium channel and calcium dynamics.
    Two gating systems:
    - mca: calcium channel activation (voltage-dependent)
    - n: KCa channel activation (calcium-dependent)
    
    Also tracks intracellular calcium concentration [Ca]_i.
    """
    
    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True  # Required for Jaxley 0.5.0+
        super().__init__(name)
        
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gkcamax": 0.03,     # S/cm² - KCa conductance
            f"{prefix}_gcamax": 3e-5,      # S/cm² - Ca conductance
            f"{prefix}_mvhalfca": -30.0,   # mV - Ca channel half-activation
            f"{prefix}_mslpca": 4.0,       # mV - Ca channel slope
            f"{prefix}_mtauca": 1.0,       # ms - Ca channel activation time constant
            f"{prefix}_tau_ca": 20.0,      # ms - calcium removal time constant
            f"{prefix}_eK": -80.0,         # Potassium reversal potential
            f"{prefix}_eCa": 120.0,        # Calcium reversal potential
        }
        self.channel_states = {
            f"{prefix}_mca": 0.0,    # Ca channel activation
            f"{prefix}_n": 0.0,      # KCa activation
            f"{prefix}_cai": 0.0001, # [Ca]_i in mM
        }
        self.current_name = f"i_MAHP"
        
        self.mtauca = 1.0     # ms - Ca channel tau
        self.depth = 0.1      # um - shell depth for Ca dynamics
        self.cainf = 0.0001   # mM - resting [Ca]_i
        self.caix = 2.0       # Ca cooperativity for KCa
        self.fKCa = 0.1       # KCa activation rate
        self.bKCa = 0.1       # KCa deactivation rate
        
        # Faraday constant (C/mol)
        self.FARADAY = 96485.0
    
    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt,
        v,
        params: Dict[str, jnp.ndarray],
    ) -> Dict[str, jnp.ndarray]:
        """Update Ca channel, KCa channel, and calcium concentration."""
        prefix = self._name
        mca = states[f"{prefix}_mca"]
        n = states[f"{prefix}_n"]
        cai = states[f"{prefix}_cai"]
        
        mvhalfca = params[f"{prefix}_mvhalfca"]
        mslpca = params[f"{prefix}_mslpca"]
        mtauca = params[f"{prefix}_mtauca"]
        gcamax = params[f"{prefix}_gcamax"]
        tau_ca = params[f"{prefix}_tau_ca"]
        e_ca = params[f"{prefix}_eCa"]

        # Update Ca channel activation (Boltzmann)
        mca_inf = 1.0 / (1.0 + save_exp(-(v - mvhalfca) / mslpca))
        mca_new = exponential_euler(mca, dt, mca_inf, mtauca)

        # Compute calcium current using OLD mca (matches NEURON's cnexp behavior)
        # In NEURON, the DERIVATIVE block sees ica from the PREVIOUS BREAKPOINT,
        # which was computed with the old mca value before state updates.
        ica = gcamax * mca * (v - e_ca)  # Use OLD mca, not mca_new!

        # Update [Ca]_i: drive from Ca current, decay to cainf
        # ODE: cai' = drive + (cainf - cai) / tau_ca
        # Rewrite as: cai' = (cainf + drive*tau_ca - cai) / tau_ca
        # So: cai_inf = cainf + drive*tau_ca, solved with exponential Euler (matches cnexp)
        drive = -10000.0 * ica / (2.0 * self.FARADAY * self.depth)
        drive = jnp.maximum(drive, 0.0)  # Cannot pump inward

        cai_inf = self.cainf + drive * tau_ca
        cai_new = exponential_euler(cai, dt, cai_inf, tau_ca)
        cai_new = jnp.maximum(cai_new, self.cainf)  # Floor at resting level

        # Update KCa activation using OLD cai (matches NEURON's cnexp behavior)
        # In NEURON, rates(cai) is called in DERIVATIVE with current cai, not updated cai
        ca_uM = jnp.maximum(cai - self.cainf, 0.0) * 1e3  # Use OLD cai, not cai_new!
        a = self.fKCa * (ca_uM ** self.caix)
        b = self.bKCa
        tau_n = 1.0 / (a + b + 1e-10)
        n_inf = a * tau_n
        n_new = exponential_euler(n, dt, n_inf, tau_n)
        
        return {
            f"{prefix}_mca": mca_new,
            f"{prefix}_n": n_new,
            f"{prefix}_cai": cai_new,
        }
    
    def compute_current(
        self,
        states: Dict[str, jnp.ndarray],
        v,
        params: Dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """
        Compute total current (KCa + Ca).
        
        Returns the sum of:
        - I_KCa = gkcamax * n * (v - E_K)
        - I_Ca = gcamax * mca * (v - E_Ca)
        """
        prefix = self._name
        mca = states[f"{prefix}_mca"]
        n = states[f"{prefix}_n"]
        
        gkcamax = params[f"{prefix}_gkcamax"]
        gcamax = params[f"{prefix}_gcamax"]
        e_k = params[f"{prefix}_eK"]
        e_ca = params[f"{prefix}_eCa"]
        
        i_kca = gkcamax * n * (v - e_k)
        i_ca = gcamax * mca * (v - e_ca)
        
        return i_kca + i_ca
    
    def init_state(
        self,
        states: Dict[str, jnp.ndarray],
        v,
        params: Dict[str, jnp.ndarray],
        delta_t,
    ) -> Dict[str, jnp.ndarray]:
        """Initialize to steady-state."""
        prefix = self._name
        mvhalfca = params[f"{prefix}_mvhalfca"]
        mslpca = params[f"{prefix}_mslpca"]
        
        mca_inf = 1.0 / (1.0 + save_exp(-(v - mvhalfca) / mslpca))
        
        ca_uM = 0.0  # At rest, no excess calcium
        a = self.fKCa * (ca_uM ** self.caix)
        b = self.bKCa
        tau_n = 1.0 / (a + b + 1e-10)
        n_inf = a * tau_n
        
        return {
            f"{prefix}_mca": mca_inf,
            f"{prefix}_n": n_inf,
            f"{prefix}_cai": jnp.full_like(v, self.cainf),
        }


# =============================================================================
# GH - HYPERPOLARIZATION-ACTIVATED CATION CURRENT (H-CURRENT)
# =============================================================================

class Gh(Channel):
    """
    Hyperpolarization-activated cation current (gh).
    
    Single gating variable n with voltage-dependent kinetics.
    Activated by hyperpolarization, contributes to sag and rebound.
    
    Non-specific cation current (mixed Na+/K+).
    """
    
    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True  # Required for Jaxley 0.5.0+
        super().__init__(name)
        
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gbar": 0.001,     # S/cm²
            f"{prefix}_half": -80.0,     # mV - half-activation voltage
            f"{prefix}_slp": 8.0,        # mV - slope
            f"{prefix}_eH": -41.0,       # mV - reversal potential
            f"{prefix}_htau": 50.0,      # ms - time constant
        }
        self.channel_states = {
            f"{prefix}_n": 0.0,
        }
        self.current_name = f"i_Gh"
    
    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt,
        v,
        params: Dict[str, jnp.ndarray],
    ) -> Dict[str, jnp.ndarray]:
        """Update activation gating variable."""
        prefix = self._name
        n = states[f"{prefix}_n"]
        
        half = params[f"{prefix}_half"]
        slp = params[f"{prefix}_slp"]
        htau = params[f"{prefix}_htau"]
        
        # Activated by hyperpolarization
        n_inf = 1.0 / (1.0 + save_exp((v - half) / slp))
        n_new = exponential_euler(n, dt, n_inf, htau)
        
        return {f"{prefix}_n": n_new}
    
    def compute_current(
        self,
        states: Dict[str, jnp.ndarray],
        v,
        params: Dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Compute H-current: I = gbar * n * (v - E_h)."""
        prefix = self._name
        n = states[f"{prefix}_n"]
        gbar = params[f"{prefix}_gbar"]
        e_h = params[f"{prefix}_eH"]
        
        return gbar * n * (v - e_h)
    
    def init_state(
        self,
        states: Dict[str, jnp.ndarray],
        v,
        params: Dict[str, jnp.ndarray],
        delta_t,
    ) -> Dict[str, jnp.ndarray]:
        """Initialize to steady-state."""
        prefix = self._name
        half = params[f"{prefix}_half"]
        slp = params[f"{prefix}_slp"]
        
        n_inf = 1.0 / (1.0 + save_exp((v - half) / slp))
        
        return {f"{prefix}_n": n_inf}


# =============================================================================
# L_CA_INACT - L-TYPE CALCIUM CHANNEL WITH INACTIVATION (DENDRITIC PIC)
# =============================================================================

class LCaInact(Channel):
    """
    L-type calcium channel with voltage-dependent inactivation.
    
    Two gating variables:
    - m: activation (fast, ~20 ms)
    - h: inactivation (slow, ~1500 ms)
    
    Responsible for dendritic persistent inward currents (PICs) and
    bistable firing behavior in motor neurons.
    
    Parameters from Booth et al. (1997) J Neurophysiol 78:3371-3385.
    """
    
    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True  # Required for Jaxley 0.5.0+
        super().__init__(name)
        
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gbar": 0.0003,      # S/cm²
            f"{prefix}_theta_m": -30.0,    # mV - activation half-voltage
            f"{prefix}_kappa_m": -6.0,     # mV - activation slope (negative)
            f"{prefix}_tau_m": 20.0,       # ms - activation time constant
            f"{prefix}_theta_h": 14.0,     # mV - inactivation half-voltage
            f"{prefix}_kappa_h": 4.0,      # mV - inactivation slope
            f"{prefix}_tau_h": 1500.0,     # ms - inactivation time constant
            f"{prefix}_eCa": 80.0,         # mV - calcium reversal potential
        }
        self.channel_states = {
            f"{prefix}_m": 0.0,
            f"{prefix}_h": 1.0,
        }
        self.current_name = f"i_LCaInact"
    
    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt,
        v,
        params: Dict[str, jnp.ndarray],
    ) -> Dict[str, jnp.ndarray]:
        """Update activation and inactivation gating variables."""
        prefix = self._name
        m = states[f"{prefix}_m"]
        h = states[f"{prefix}_h"]
        
        theta_m = params[f"{prefix}_theta_m"]
        kappa_m = params[f"{prefix}_kappa_m"]
        tau_m = params[f"{prefix}_tau_m"]
        theta_h = params[f"{prefix}_theta_h"]
        kappa_h = params[f"{prefix}_kappa_h"]
        tau_h = params[f"{prefix}_tau_h"]
        
        # Boltzmann activation (note: kappa_m is negative)
        m_inf = 1.0 / (1.0 + save_exp((v - theta_m) / kappa_m))
        # Boltzmann inactivation
        h_inf = 1.0 / (1.0 + save_exp((v - theta_h) / kappa_h))
        
        m_new = exponential_euler(m, dt, m_inf, tau_m)
        h_new = exponential_euler(h, dt, h_inf, tau_h)
        
        return {
            f"{prefix}_m": m_new,
            f"{prefix}_h": h_new,
        }
    
    def compute_current(
        self,
        states: Dict[str, jnp.ndarray],
        v,
        params: Dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Compute L-type Ca current: I = gbar * m * h * (v - E_Ca)."""
        prefix = self._name
        m = states[f"{prefix}_m"]
        h = states[f"{prefix}_h"]
        gbar = params[f"{prefix}_gbar"]
        vca = params[f"{prefix}_eCa"]
        
        return gbar * m * h * (v - vca)
    
    def init_state(
        self,
        states: Dict[str, jnp.ndarray],
        v,
        params: Dict[str, jnp.ndarray],
        delta_t,
    ) -> Dict[str, jnp.ndarray]:
        """Initialize to steady-state."""
        prefix = self._name
        theta_m = params[f"{prefix}_theta_m"]
        kappa_m = params[f"{prefix}_kappa_m"]
        theta_h = params[f"{prefix}_theta_h"]
        kappa_h = params[f"{prefix}_kappa_h"]
        
        m_inf = 1.0 / (1.0 + save_exp((v - theta_m) / kappa_m))
        h_inf = 1.0 / (1.0 + save_exp((v - theta_h) / kappa_h))
        
        return {
            f"{prefix}_m": m_inf,
            f"{prefix}_h": h_inf,
        }


# =============================================================================
# LEAK CHANNEL (INCLUDED FOR COMPLETENESS)
# =============================================================================

class LeakChannel(Channel):
    """
    Simple leak channel for passive membrane properties.
    
    I = g_leak * (v - E_leak)
    """
    
    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True  # Required for Jaxley 0.5.0+
        super().__init__(name)
        
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gLeak": 1e-4,     # S/cm²
            f"{prefix}_eLeak": -70.0,    # mV
        }
        self.channel_states = {}
        self.current_name = f"i_Leak"
    
    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt,
        v,
        params: Dict[str, jnp.ndarray],
    ) -> Dict[str, jnp.ndarray]:
        """No states to update for leak channel."""
        return {}
    
    def compute_current(
        self,
        states: Dict[str, jnp.ndarray],
        v,
        params: Dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Compute leak current."""
        prefix = self._name
        g = params[f"{prefix}_gLeak"]
        e = params[f"{prefix}_eLeak"]
        return g * (v - e)
    
    def init_state(
        self,
        states: Dict[str, jnp.ndarray],
        v,
        params: Dict[str, jnp.ndarray],
        delta_t,
    ) -> Dict[str, jnp.ndarray]:
        """No states to initialize."""
        return {}


# =============================================================================
# CONVENIENCE FUNCTION FOR MOTOR NEURON CHANNEL SET
# =============================================================================

def get_motor_neuron_channels_soma(model: str = "Powers2017") -> list:
    """
    Get the complete set of soma channels for a motor neuron.
    
    Parameters
    ----------
    model : str
        "Powers2017" or "NERLab" parameter sets.
        
    Returns
    -------
    list
        List of channel instances configured for soma.
    """
    if model == "Powers2017":
        return [
            Na3rp(),
            Naps(),
            KdrRL(),
            MAHP(),
            Gh(),
            LeakChannel(),
        ]
    else:  # NERLab
        return [
            Na3rp(),
            KdrRL(),
            MAHP(),
            Gh(),
            LeakChannel(),
        ]


def get_motor_neuron_channels_dendrite(model: str = "Powers2017") -> list:
    """
    Get the complete set of dendritic channels for a motor neuron.
    
    Dendrites have L-type Ca channels for PICs but fewer Na channels.
    
    Parameters
    ----------
    model : str
        "Powers2017" or "NERLab" parameter sets.
        
    Returns
    -------
    list
        List of channel instances configured for dendrites.
    """
    if model == "Powers2017":
        return [
            LCaInact(),
            Gh(),
            LeakChannel(),
        ]
    else:  # NERLab
        return [
            LCaInact(),
            Gh(),
            LeakChannel(),
        ]
