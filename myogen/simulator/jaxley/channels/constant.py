"""
Constant current channel for Jaxley.

Converted from constant.mod:
    NEURON {
      SUFFIX Constant
      NONSPECIFIC_CURRENT i
      RANGE i, ic
    }
    PARAMETER {
      ic = 0.000 (mA/cm2)
    }
    BREAKPOINT {
      i = ic
    }

This is the simplest possible channel - just returns a constant current.
Useful for testing and validation.
"""

import jax.numpy as jnp
from jaxley.channels import Channel
from typing import Dict, Any


class Constant(Channel):
    """
    Constant current channel.
    
    Provides a simple non-specific current that doesn't change over time.
    Useful for baseline current injection or testing.
    
    Parameters
    ----------
    ic : float, optional
        Constant current density in mA/cm². Default is 0.0.
    name : str, optional
        Name prefix for parameters. If None, uses class name.
    """
    
    def __init__(self, ic: float = 0.0, name: str | None = None):
        # Required for Jaxley 0.5.0+: declare current units
        self.current_is_in_mA_per_cm2 = True
        
        super().__init__(name)
        
        # Ensure prefix is never None
        prefix = self._name if self._name is not None else "Constant"
        
        # Channel parameters - matching NMODL PARAMETER block
        self.channel_params = {
            f"{prefix}_ic": ic,  # Constant current (mA/cm²)
        }
        
        # No state variables - current is constant
        self.channel_states = {}
    
    def update_states(
        self,
        states: Dict[str, Any],
        dt: float,
        v: float,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Update channel states.
        
        Since this is a constant current, there are no states to update.
        
        Parameters
        ----------
        states : dict
            Current channel states (empty for this channel).
        dt : float
            Time step in milliseconds.
        v : float
            Membrane voltage in mV.
        params : dict
            Channel parameters.
            
        Returns
        -------
        dict
            Updated states (empty for this channel).
        """
        return {}
    
    def compute_current(
        self,
        states: Dict[str, Any],
        v: float,
        params: Dict[str, Any]
    ) -> float:
        """
        Calculate ionic current.
        
        Simply returns the constant current parameter.
        
        Parameters
        ----------
        states : dict
            Current channel states (empty for this channel).
        v : float
            Membrane voltage in mV.
        params : dict
            Channel parameters containing 'Constant_ic'.
            
        Returns
        -------
        float
            Current in mA/cm².
        """
        prefix = self._name if self._name is not None else "Constant"
        return params[f"{prefix}_ic"]
    
    def init_state(
        self,
        states: Dict[str, Any],
        v: float,
        params: Dict[str, Any],
        delta_t: float
    ) -> Dict[str, Any]:
        """
        Initialize channel states.
        
        Since this channel has no states, returns empty dict.
        
        Parameters
        ----------
        states : dict
            Initial states (empty).
        v : float
            Initial membrane voltage in mV.
        params : dict
            Channel parameters.
        delta_t : float
            Time step in milliseconds.
            
        Returns
        -------
        dict
            Initialized states (empty).
        """
        return {}
