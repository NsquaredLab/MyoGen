"""
Golgi Tendon Organ (GTO) Model - Jaxley Backend (Pure Python/NumPy)

This module provides the GTO model using pure NumPy for the Jaxley backend.
Maintains full API compatibility with the NEURON version.

Based on Lin & Crago (2002) with implementation by Elias (2014).
"""

from typing import Any, Dict

import numpy as np

from myogen.utils.decorators import beartowertype
from myogen.utils.types import Quantity__ms


class _GolgiTendonOrgan__NumPy:
    """
    Pure NumPy implementation of Golgi Tendon Organ model.
    
    Implements logarithmic force-to-firing relationship from Lin & Crago (2002).
    """
    
    def __init__(self, gtoD: Dict[str, Any], tstop__ms: float, dt__ms: float):
        self.params = gtoD
        self.tstop = tstop__ms
        self.dt = dt__ms
        
        # Initialize storage arrays
        self.n_steps = int(tstop__ms / dt__ms) + 1
        self.time = np.linspace(0, tstop__ms, self.n_steps)
        
        # Output array
        self.Ib = np.zeros(self.n_steps)  # Ib afferent firing rate
        
        # State variables
        self.step_idx = 0
        
        # Digital filter state (simple low-pass)
        self.prev_firing = 0.0
        self.filter_alpha = 0.3  # Smoothing factor
        
    def integrate(self, muscle_force__N: float) -> float:
        """
        Integrate GTO model for one time step.
        
        Parameters
        ----------
        muscle_force__N : float
            Current muscle force in Newtons
            
        Returns
        -------
        float
            Ib afferent firing rate in Hz
        """
        # Logarithmic force-to-firing relationship
        # firing_rate = G1 * log(force/G2 + 1)
        G1 = self.params["G1"]
        G2 = self.params["G2"]
        
        # Ensure force is non-negative
        force = max(0, muscle_force__N)
        
        # Compute instantaneous firing rate
        firing_rate = G1 * np.log(force / G2 + 1.0)
        
        # Apply simple low-pass filter for temporal dynamics
        if self.step_idx > 0:
            firing_rate = (1 - self.filter_alpha) * self.prev_firing + self.filter_alpha * firing_rate
        
        self.prev_firing = firing_rate
        
        # Store firing rate
        self.Ib[self.step_idx] = firing_rate
        
        # Increment step
        self.step_idx += 1
        
        return firing_rate


@beartowertype
class GolgiTendonOrganModel:
    """
    API wrapper for the Golgi Tendon Organ (GTO) model - Jaxley Backend.

    This class provides an intuitive interface for creating GTO models
    with user-friendly parameter names that are internally mapped to the
    correct format expected by the underlying GTO implementation.

    The Golgi Tendon Organ is a proprioceptive sensory organ located at the
    muscle-tendon junction that detects muscle force/tension and provides
    feedback for motor control and protection against excessive forces.

    The model is based on Lin & Crago (2002) and implements a logarithmic
    force-to-firing relationship with digital filtering for realistic
    afferent discharge patterns.

    Parameters
    ----------
    simulation_time__ms : Quantity__ms
        Total simulation time in milliseconds
    time_step__ms : Quantity__ms
        Integration time step in milliseconds
    gto_parameters : Dict[str, Any]
        Dictionary containing GTO model parameters
    """

    def __init__(
        self,
        simulation_time__ms: Quantity__ms,
        time_step__ms: Quantity__ms,
        gto_parameters: Dict[str, Any],
    ):
        # Store original parameters (immutable)
        self.simulation_time__ms = simulation_time__ms
        self.time_step__ms = time_step__ms
        self.gto_parameters = gto_parameters.copy()

        # Private working copies for internal use
        self._simulation_time__ms = simulation_time__ms
        self._time_step__ms = time_step__ms
        self._gto_parameters = gto_parameters.copy()

        # Validate inputs
        self._validate_parameters()

        # Create the underlying GTO model
        self._gto_model = self._create_gto_model()

    def _validate_parameters(self) -> None:
        """Validate input parameters."""
        if self._simulation_time__ms <= 0:
            raise ValueError("simulation_time__ms must be positive")

        if self._time_step__ms <= 0:
            raise ValueError("time_step__ms must be positive")

        if self._simulation_time__ms <= self._time_step__ms:
            raise ValueError("simulation_time__ms must be greater than time_step__ms")

    def _create_gto_model(self) -> _GolgiTendonOrgan__NumPy:
        """
        Create the underlying GTO model instance.

        This method maps the user-friendly parameter names to the format
        expected by the GTO constructor.
        """
        return _GolgiTendonOrgan__NumPy(
            gtoD=self._gto_parameters,
            tstop__ms=self._simulation_time__ms.magnitude,
            dt__ms=self._time_step__ms.magnitude,
        )

    def integrate(self, muscle_force__N: float) -> float:
        """
        Integrate the GTO model for one time step.

        Parameters
        ----------
        muscle_force__N : float
            Current muscle force in Newtons

        Returns
        -------
        float
            Ib afferent firing rate in Hz
        """
        return self._gto_model.integrate(muscle_force__N)

    @property
    def ib_afferent_firing__Hz(self) -> np.ndarray:
        """Get Ib afferent firing rate time series in Hz."""
        return np.asarray(self._gto_model.Ib)

    def __repr__(self) -> str:
        """String representation of the GTO model."""
        return (
            f"GolgiTendonOrganModel(t_sim={self.simulation_time__ms}ms, dt={self.time_step__ms}ms)"
        )

    @staticmethod
    def create_default_gto_parameters() -> Dict[str, Any]:
        """
        Create default Golgi Tendon Organ parameter dictionary.

        The GTO model uses a logarithmic force-to-firing relationship:
        firing_rate = G1 * log(force/G2 + 1)

        This is followed by digital filtering to create realistic temporal
        dynamics in the afferent discharge pattern.

        Returns
        -------
        Dict[str, Any]
            Dictionary of GTO parameters with detailed explanations

        Notes
        -----
        Model based on:
        - Lin & Crago (2002): Mathematical model framework
        - Aniss et al. (1990b): Human GTO physiological data
        - Elias PhD thesis (pg 83): Implementation details

        The logarithmic relationship captures the GTO's ability to encode
        force over a wide dynamic range, from threshold detection of small
        forces to saturation at high forces, providing force feedback for
        motor control and protective reflexes.
        """
        return {
            # Force-to-firing relationship parameters
            # Firing rate [Hz] = G1 * log(force[N]/G2 + 1)
            "G1": 40,  # Logarithmic gain coefficient [Hz]
            # Controls the sensitivity of force-to-firing conversion
            # Higher G1 = more sensitive to force changes
            # Typical range: 30-60 Hz for different muscles
            # Source: Lin & Crago (2002), Elias thesis uses 40 Hz
            "G2": 4,  # Force scaling coefficient [N]
            # Sets the force level for logarithmic scaling
            # Lower G2 = more sensitive to low forces
            # Higher G2 = requires higher forces for activation
            # Typical range: 2-8 N depending on muscle strength
            # Source: Calibrated for human muscle force ranges
        }

    @staticmethod
    def create_gto_parameters_for_muscle(muscle_type: str = "FDI") -> Dict[str, Any]:
        """
        Create GTO parameters optimized for specific muscle types.

        Parameters
        ----------
        muscle_type : str, optional
            Type of muscle ("FDI", "Sol", "generic"), by default "FDI"

        Returns
        -------
        Dict[str, Any]
            Dictionary of muscle-specific GTO parameters

        Notes
        -----
        Different muscles have different force production capabilities
        and thus require different GTO sensitivity parameters:

        - FDI (First Dorsal Interosseous): Small hand muscle, low forces
        - Sol (Soleus): Large calf muscle, high forces
        - Generic: General-purpose parameters
        """
        if muscle_type == "FDI":
            # Small hand muscle - more sensitive to small forces
            return {
                "G1": 45,  # Higher sensitivity for small force detection
                "G2": 2,  # Lower threshold for activation at small forces
            }

        elif muscle_type == "Sol":
            # Large calf muscle - less sensitive, handles high forces
            return {
                "G1": 35,  # Lower sensitivity appropriate for large forces
                "G2": 8,  # Higher threshold matching muscle's force capacity
            }

        elif muscle_type == "generic":
            # General-purpose parameters
            return GolgiTendonOrganModel.create_default_gto_parameters()

        else:
            raise ValueError(f"Unknown muscle type: {muscle_type}. Use 'FDI', 'Sol', or 'generic'.")
