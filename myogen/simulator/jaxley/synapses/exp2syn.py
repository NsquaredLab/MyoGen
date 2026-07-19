"""
Double-Exponential Synapse (Exp2Syn) for Jaxley.

This module provides Jaxley-compatible implementations of synaptic conductances
following the double-exponential kinetics used in NEURON's Exp2Syn mechanism.

The synaptic conductance follows:
    g(t) = weight * factor * (exp(-t/tau2) - exp(-t/tau1))

where factor normalizes the peak conductance.
"""

from typing import Dict, Optional, Tuple
import jax.numpy as jnp
from jaxley.synapses import Synapse


def safe_exp(x: jnp.ndarray, clip_val: float = 100.0) -> jnp.ndarray:
    """Exponential with clipping to prevent overflow."""
    return jnp.exp(jnp.clip(x, -clip_val, clip_val))


class Exp2Syn(Synapse):
    """
    Double-exponential synapse (Exp2Syn).
    
    Implements the standard NEURON Exp2Syn mechanism with:
    - Rise time constant (tau1)
    - Decay time constant (tau2)  
    - Reversal potential (e)
    
    The conductance waveform is:
        g(t) = weight * factor * (exp(-t/tau2) - exp(-t/tau1))
    
    where factor = 1 / (peak_norm) normalizes peak conductance to weight.
    
    Parameters
    ----------
    name : str, optional
        Name of the synapse instance.
    tau1 : float
        Rise time constant in ms. Default: 0.2 ms.
    tau2 : float
        Decay time constant in ms. Must be > tau1. Default: 2.0 ms.
    e : float
        Reversal potential in mV. Default: 0.0 mV (excitatory).
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        tau1: float = 0.2,    # ms - rise time
        tau2: float = 2.0,    # ms - decay time
        e: float = 0.0,       # mV - reversal potential
    ):
        super().__init__(name=name or "Exp2Syn")
        
        if tau2 <= tau1:
            raise ValueError(f"tau2 ({tau2}) must be greater than tau1 ({tau1})")
        
        self.tau1 = tau1
        self.tau2 = tau2
        self.e = e
        
        # Compute normalization factor so peak conductance equals weight
        self._compute_factor()
    
    def _compute_factor(self):
        """Compute factor to normalize peak conductance."""
        # Time to peak: tp = (tau1 * tau2 / (tau2 - tau1)) * ln(tau2/tau1)
        tp = (self.tau1 * self.tau2 / (self.tau2 - self.tau1)) * jnp.log(self.tau2 / self.tau1)
        
        # Peak value of unnormalized waveform
        peak = jnp.exp(-tp / self.tau2) - jnp.exp(-tp / self.tau1)
        
        # Factor to normalize peak to 1
        self.factor = 1.0 / peak if peak > 0 else 1.0
    
    @property
    def synapse_params(self) -> Dict[str, float]:
        """Parameters that can be set per-synapse."""
        return {
            f"{self.name}_tau1": self.tau1,
            f"{self.name}_tau2": self.tau2,
            f"{self.name}_e": self.e,
            f"{self.name}_g": 0.0,  # Conductance (set by weight)
        }
    
    @property
    def synapse_states(self) -> Dict[str, float]:
        """State variables for synaptic dynamics."""
        return {
            f"{self.name}_A": 0.0,  # Rising phase variable
            f"{self.name}_B": 0.0,  # Decaying phase variable
        }
    
    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        delta_t: float,
        pre_voltage: jnp.ndarray,
        post_voltage: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
    ) -> Dict[str, jnp.ndarray]:
        """
        Update synaptic state variables.
        
        The conductance g = B - A follows double-exponential kinetics:
        - dA/dt = -A/tau1
        - dB/dt = -B/tau2
        
        On presynaptic spike: A += weight*factor, B += weight*factor
        """
        prefix = self.name
        A = states[f"{prefix}_A"]
        B = states[f"{prefix}_B"]
        
        tau1 = params[f"{prefix}_tau1"]
        tau2 = params[f"{prefix}_tau2"]
        
        # Exponential decay
        A_new = A * safe_exp(-delta_t / tau1)
        B_new = B * safe_exp(-delta_t / tau2)
        
        return {
            f"{prefix}_A": A_new,
            f"{prefix}_B": B_new,
        }
    
    def compute_current(
        self,
        states: Dict[str, jnp.ndarray],
        pre_voltage: jnp.ndarray,
        post_voltage: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """
        Compute synaptic current.
        
        I = g * (B - A) * (v - e)
        
        where (B - A) gives the double-exponential conductance waveform.
        """
        prefix = self.name
        A = states[f"{prefix}_A"]
        B = states[f"{prefix}_B"]
        
        e = params[f"{prefix}_e"]
        g = params[f"{prefix}_g"]
        
        # Conductance = B - A (normalized double exponential)
        g_syn = g * self.factor * (B - A)
        
        return g_syn * (post_voltage - e)
    
    def init_state(
        self,
        states: Dict[str, jnp.ndarray],
        pre_voltage: jnp.ndarray,
        post_voltage: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
        delta_t: float,
    ) -> Dict[str, jnp.ndarray]:
        """Initialize to resting state (no active synaptic input)."""
        prefix = self.name
        return {
            f"{prefix}_A": jnp.zeros_like(post_voltage),
            f"{prefix}_B": jnp.zeros_like(post_voltage),
        }


class ExcitatorySynapse(Exp2Syn):
    """
    Excitatory synapse (AMPA-like).
    
    Fast kinetics with reversal at 0 mV.
    
    Parameters
    ----------
    name : str, optional
        Name of the synapse instance.
    tau1 : float
        Rise time constant in ms. Default: 0.2 ms.
    tau2 : float
        Decay time constant in ms. Default: 2.0 ms.
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        tau1: float = 0.2,
        tau2: float = 2.0,
    ):
        super().__init__(
            name=name or "AMPA",
            tau1=tau1,
            tau2=tau2,
            e=0.0,  # Excitatory reversal
        )


class InhibitorySynapse(Exp2Syn):
    """
    Inhibitory synapse (GABA-A like).
    
    Slower kinetics with reversal at -75 mV.
    
    Parameters
    ----------
    name : str, optional
        Name of the synapse instance.
    tau1 : float
        Rise time constant in ms. Default: 0.5 ms.
    tau2 : float
        Decay time constant in ms. Default: 10.0 ms.
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        tau1: float = 0.5,
        tau2: float = 10.0,
    ):
        super().__init__(
            name=name or "GABA",
            tau1=tau1,
            tau2=tau2,
            e=-75.0,  # Inhibitory reversal
        )


class NMDASynapse(Synapse):
    """
    NMDA synapse with voltage-dependent Mg2+ block.
    
    Implements:
    - Slow kinetics (tau1 ~ 2 ms, tau2 ~ 100 ms)
    - Voltage-dependent Mg2+ block: B(V) = 1 / (1 + [Mg]/3.57 * exp(-0.062*V))
    - Mixed Na+/Ca2+ reversal at 0 mV
    
    Parameters
    ----------
    name : str, optional
        Name of the synapse instance.
    tau1 : float
        Rise time constant in ms. Default: 2.0 ms.
    tau2 : float
        Decay time constant in ms. Default: 100.0 ms.
    mg_conc : float
        Extracellular Mg2+ concentration in mM. Default: 1.0 mM.
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        tau1: float = 2.0,
        tau2: float = 100.0,
        mg_conc: float = 1.0,
    ):
        super().__init__(name=name or "NMDA")
        
        if tau2 <= tau1:
            raise ValueError(f"tau2 ({tau2}) must be greater than tau1 ({tau1})")
        
        self.tau1 = tau1
        self.tau2 = tau2
        self.mg_conc = mg_conc
        self.e = 0.0  # NMDA reversal
        
        # Compute normalization factor
        tp = (self.tau1 * self.tau2 / (self.tau2 - self.tau1)) * jnp.log(self.tau2 / self.tau1)
        peak = jnp.exp(-tp / self.tau2) - jnp.exp(-tp / self.tau1)
        self.factor = 1.0 / peak if peak > 0 else 1.0
    
    @property
    def synapse_params(self) -> Dict[str, float]:
        return {
            f"{self.name}_tau1": self.tau1,
            f"{self.name}_tau2": self.tau2,
            f"{self.name}_mg": self.mg_conc,
            f"{self.name}_e": self.e,
            f"{self.name}_g": 0.0,
        }
    
    @property
    def synapse_states(self) -> Dict[str, float]:
        return {
            f"{self.name}_A": 0.0,
            f"{self.name}_B": 0.0,
        }
    
    def _mg_block(self, v: jnp.ndarray, mg: float) -> jnp.ndarray:
        """Voltage-dependent Mg2+ block factor."""
        return 1.0 / (1.0 + mg / 3.57 * safe_exp(-0.062 * v))
    
    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        delta_t: float,
        pre_voltage: jnp.ndarray,
        post_voltage: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
    ) -> Dict[str, jnp.ndarray]:
        prefix = self.name
        A = states[f"{prefix}_A"]
        B = states[f"{prefix}_B"]
        
        tau1 = params[f"{prefix}_tau1"]
        tau2 = params[f"{prefix}_tau2"]
        
        A_new = A * safe_exp(-delta_t / tau1)
        B_new = B * safe_exp(-delta_t / tau2)
        
        return {
            f"{prefix}_A": A_new,
            f"{prefix}_B": B_new,
        }
    
    def compute_current(
        self,
        states: Dict[str, jnp.ndarray],
        pre_voltage: jnp.ndarray,
        post_voltage: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        prefix = self.name
        A = states[f"{prefix}_A"]
        B = states[f"{prefix}_B"]
        
        e = params[f"{prefix}_e"]
        g = params[f"{prefix}_g"]
        mg = params[f"{prefix}_mg"]
        
        # Conductance with Mg block
        g_syn = g * self.factor * (B - A) * self._mg_block(post_voltage, mg)
        
        return g_syn * (post_voltage - e)
    
    def init_state(
        self,
        states: Dict[str, jnp.ndarray],
        pre_voltage: jnp.ndarray,
        post_voltage: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
        delta_t: float,
    ) -> Dict[str, jnp.ndarray]:
        prefix = self.name
        return {
            f"{prefix}_A": jnp.zeros_like(post_voltage),
            f"{prefix}_B": jnp.zeros_like(post_voltage),
        }
