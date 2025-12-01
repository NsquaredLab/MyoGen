"""
Muscle Spindle Model API Wrapper

This module provides a clean API wrapper for the muscle spindle model,
allowing for intuitive parameter names while maintaining compatibility
with the underlying Spindle implementation.
"""

from typing import Any, Dict

import numpy as np

from myogen.simulator.neuron._cython._spindle import _Spindle__Cython
from myogen.utils.decorators import beartowertype
from myogen.utils.types import Quantity__ms


@beartowertype
class SpindleModel:
    """
    API wrapper for the muscle spindle model.

    This class provides an intuitive interface for creating muscle spindle models
    with user-friendly parameter names that are internally mapped to the
    correct format expected by the underlying Spindle implementation.

    The muscle spindle is a proprioceptive sensory organ that detects changes
    in muscle length and velocity, providing feedback for motor control.

    Parameters
    ----------
    simulation_time__ms : Quantity__ms
        Total simulation time in milliseconds
    time_step__ms : Quantity__ms
        Integration time step in milliseconds
    spindle_parameters : Dict[str, Any]
        Dictionary containing spindle model parameters
    """

    def __init__(
        self,
        simulation_time__ms: Quantity__ms,
        time_step__ms: Quantity__ms,
        spindle_parameters: Dict[str, Any],
    ):
        self.simulation_time__ms = simulation_time__ms
        self.time_step__ms = time_step__ms
        self.spindle_parameters = spindle_parameters.copy()

        # Private working copies for internal use
        self._simulation_time__ms = simulation_time__ms
        self._time_step__ms = time_step__ms
        self._spindle_parameters = spindle_parameters.copy()

        # Validate inputs
        self._validate_parameters()

        # Create the underlying Spindle model
        self._spindle_model = self._create_spindle_model()

    def _validate_parameters(self) -> None:
        """Validate input parameters."""
        if self._simulation_time__ms <= 0:
            raise ValueError("simulation_time__ms must be positive")

        if self._time_step__ms <= 0:
            raise ValueError("time_step__ms must be positive")

        if self._simulation_time__ms <= self._time_step__ms:
            raise ValueError("simulation_time__ms must be greater than time_step__ms")

    def _create_spindle_model(self) -> _Spindle__Cython:
        """
        Create the underlying Spindle model instance.

        This method maps the user-friendly parameter names to the format
        expected by the Spindle cython constructor.
        """
        return _Spindle__Cython(
            tstop=self._simulation_time__ms.magnitude,
            dt=self._time_step__ms.magnitude,
            spinD=self._spindle_parameters,
        )

    def integrate(
        self,
        muscle_length__L0: float,
        muscle_velocity__L0_per_s: float,
        muscle_acceleration__L0_per_s2: float,
        gamma_dynamic_drive__Hz: float,
        gamma_static_drive__Hz: float,
    ) -> tuple[float, float]:
        """
        Integrate the spindle model for one time step.

        Parameters
        ----------
        muscle_length__L0 : float
            Current muscle length normalized to L0
        muscle_velocity__L0_per_s : float
            Current muscle velocity in L0/s
        muscle_acceleration__L0_per_s2 : float
            Current muscle acceleration in L0/s²
        gamma_dynamic_drive__Hz : float
            Gamma dynamic motor neuron drive frequency in Hz
        gamma_static_drive__Hz : float
            Gamma static motor neuron drive frequency in Hz

        Returns
        -------
        tuple[float, float]
            Primary afferent (Ia) and secondary afferent (II) firing rates in Hz
        """
        return self._spindle_model.integrate(
            muscle_length__L0,
            muscle_velocity__L0_per_s,
            muscle_acceleration__L0_per_s2,
            gamma_dynamic_drive__Hz,
            gamma_static_drive__Hz,
        )

    @property
    def primary_afferent_firing__Hz(self) -> np.ndarray:
        """Get primary afferent (Ia) firing rate time series in Hz."""
        return np.asarray(self._spindle_model.Ia)

    @property
    def secondary_afferent_firing__Hz(self) -> np.ndarray:
        """Get secondary afferent (II) firing rate time series in Hz."""
        return np.asarray(self._spindle_model.II)

    @property
    def bag1_activation(self) -> np.ndarray:
        """Get Bag1 fiber activation time series."""
        return np.asarray(self._spindle_model.aBag1)

    @property
    def bag2_activation(self) -> np.ndarray:
        """Get Bag2 fiber activation time series."""
        return np.asarray(self._spindle_model.aBag2)

    @property
    def chain_activation(self) -> np.ndarray:
        """Get Chain fiber activation time series."""
        return np.asarray(self._spindle_model.aChain)

    @property
    def intrafusal_tensions(self) -> np.ndarray:
        """Get intrafusal fiber tensions matrix (3 × time_points) [Bag1, Bag2, Chain]."""
        return np.asarray(self._spindle_model.T)

    @property
    def time_vector(self) -> np.ndarray:
        """Get simulation time vector in milliseconds."""
        return np.asarray(self._spindle_model.time)

    def __repr__(self) -> str:
        """String representation of the spindle model."""
        return f"SpindleModel(t_sim={self.simulation_time__ms}ms, dt={self.time_step__ms}ms)"

    @staticmethod
    def create_default_spindle_parameters(
        species: str = "human", deafferent_ia: bool = False, deafferent_ii: bool = False
    ) -> Dict[str, Any]:
        """
        Create default spindle parameter dictionary.

        Parameters
        ----------
        species : str, optional
            Species type ("human" or "cat"), by default "human"
        deafferent_ia : bool, optional
            Whether to simulate Ia afferent deafferentation, by default False
        deafferent_ii : bool, optional
            Whether to simulate II afferent deafferentation, by default False

        Returns
        -------
        Dict[str, Any]
            Dictionary of spindle parameters with detailed explanations

        Raises
        ------
        ValueError
            If species is not recognized
        """
        # Base spindle parameters (Mileusnic et al., 2006)
        spindle_params = {
            # Fusimotor activation parameters
            "fBag1": 60,  # Fusimotor frequency to activation constant for Bag1 [Hz]
            "fBag2": 60,  # Fusimotor frequency to activation constant for Bag2 [Hz]
            "fChain": 90,  # Fusimotor frequency to activation constant for Chain [Hz]
            "P": 2,  # Fusimotor frequency to activation power constant
            # Force generation coefficients
            "G1": 0.0289,  # Dynamic fusimotor input force generation coef [FU]
            "G2": 0.0636,  # Static fusimotor input force generation coef [FU]
            "G2Chain": 0.0954,  # Static fusimotor input force gen coef for Chain [FU]
            # Sensory Region (SR) mechanical parameters
            "K_SR": 10.4649,  # SR spring constant [FU/L0] - detects length changes
            "L0_SR": 0.04,  # SR rest length [L0] - baseline length
            "LN_SR": 0.0423,  # SR threshold length [L0] - minimum for activation
            # Polar Region (PR) mechanical parameters
            "K_PR": 0.15,  # PR spring constant [FU/L0] - contractile region
            "L0_PR": 0.76,  # PR rest length [L0] - baseline contractile length
            "LN_PR": 0.89,  # PR threshold length [L0] - minimum for activation
            # Intrafusal fiber mechanical properties
            "M": 0.0002,  # Intrafusal fiber mass [FU/(L0/s²)] - inertial component
            # Passive damping coefficients [FU/(L0/s)] - baseline viscosity
            "b0Bag1": 0.0605,  # Bag1 passive damping
            "b0Bag2": 0.0822,  # Bag2 passive damping
            "b0Chain": 0.0822,  # Chain passive damping
            # Fusimotor-dependent damping coefficients [FU/(L0/s)]
            "b1Bag1": 0.2592,  # Dynamic fusimotor damping for Bag1
            "b2Bag2": -0.0460,  # Static fusimotor damping for Bag2
            "b2Chain": -0.0690,  # Static fusimotor damping for Chain
            # Force-velocity relationship parameters
            "a": 0.3,  # Nonlinear velocity dependence power constant
            "C_L": 1,  # Lengthening coefficient of asymmetry in F-V curve
            "C_S": 0.42,  # Shortening coefficient of asymmetry in F-V curve
            "R": 0.46,  # Fascicle length where force production is zero [L0]
            # Afferent firing properties
            "X": 0.7,  # Secondary afferent percentage on sensory region [0-1]
            "Lsec": 0.04,  # Secondary afferent rest length [L0]
            "S": 0.156,  # Occlusion factor for primary afferent interactions
            # Temporal dynamics (low-pass filtering)
            "tau1": 0.149,  # Bag1 activation time constant [s] - fast dynamics
            "tau2": 0.205,  # Bag2 activation time constant [s] - slow dynamics
            # Afferent sensitivity gains [Hz/L0] - firing rate per unit stretch
            "gBag1": 6500,  # Bag1 contribution to primary afferent (Ia)
            "gBag2A1": 3250,  # Bag2 contribution to primary afferent (Ia)
            "gChainA1": 3250,  # Chain contribution to primary afferent (Ia)
            "gBag2A2": 3500,  # Bag2 contribution to secondary afferent (II)
            "gChainA2": 3500,  # Chain contribution to secondary afferent (II)
        }

        # Species-specific and deafferentation modifications
        if species == "human":
            if not deafferent_ii and not deafferent_ia:
                # Normal human spindle (Case 1, Elias thesis pg 66)
                pass  # Use default values above

            elif deafferent_ii and not deafferent_ia:
                # Human with Type II deafferentation (Case 2, Elias thesis pg 66)
                spindle_params.update(
                    {
                        "gBag1": 7000,  # Enhanced Bag1 sensitivity
                        "gBag2A1": 3750,  # Enhanced Bag2 primary sensitivity
                        "gChainA1": 3750,  # Enhanced Chain primary sensitivity
                        "gBag2A2": 0,  # No Bag2 secondary afferents
                        "gChainA2": 0,  # No Chain secondary afferents
                    }
                )

            elif not deafferent_ii and deafferent_ia:
                # Human with Ia deafferentation (Case 3, Elias thesis pg 66)
                spindle_params.update(
                    {
                        "gBag1": 0,  # No Bag1 primary afferents
                        "gBag2A1": 0,  # No Bag2 primary afferents
                        "gChainA1": 0,  # No Chain primary afferents
                        "gBag2A2": 4500,  # Enhanced Bag2 secondary sensitivity
                        "gChainA2": 4500,  # Enhanced Chain secondary sensitivity
                    }
                )

        elif species == "cat":
            # Cat spindle parameters (original Mileusnic values)
            spindle_params.update(
                {
                    "gBag1": 20000,  # Higher sensitivity in cat
                    "gBag2A1": 10000,  # Higher Bag2 primary sensitivity
                    "gChainA1": 10000,  # Higher Chain primary sensitivity
                    "gBag2A2": 7250,  # Higher Bag2 secondary sensitivity
                    "gChainA2": 7250,  # Higher Chain secondary sensitivity
                }
            )

        else:
            raise ValueError(f"Unknown species: {species}. Use 'human' or 'cat'.")

        return spindle_params
