"""
Muscle Spindle Model - Jaxley Backend (Pure Python/NumPy)

This module provides the muscle spindle model using pure NumPy for the Jaxley backend.
Maintains full API compatibility with the NEURON version.

Based on the model by Mileusnic et al. (2006) and implementation by Elias (2014).
"""

from typing import Any, Dict

import numpy as np

from myogen.utils.decorators import beartowertype
from myogen.utils.types import Quantity__ms


class _Spindle__NumPy:
    """
    Pure NumPy implementation of muscle spindle model.

    Faithfully implements the full Mileusnic et al. (2006) model:
    - Hill-type fusimotor activation (RK4 for Bag1/Bag2, algebraic for Chain)
    - Second-order mechanical ODE for intrafusal fiber tensions (RK4)
    - Nonlinear force-velocity damping with asymmetric lengthening/shortening
    - Afferent occlusion mechanism for primary afferents (Ia)
    - Two-component secondary afferent (II) from sensory + polar regions

    Matches the NEURON Cython implementation (_spindle.pyx) equation-by-equation.
    """

    def __init__(self, tstop: float, dt: float, spinD: Dict[str, Any]):
        self.tstop = tstop
        self.dt = dt
        self.params = spinD

        # Initialize storage arrays
        self.n_steps = int(tstop / dt) + 1
        self.time = np.linspace(0, tstop, self.n_steps)

        # Output arrays
        self.Ia = np.zeros(self.n_steps)  # Primary afferent firing [pps]
        self.II = np.zeros(self.n_steps)  # Secondary afferent firing [pps]
        self.aBag1 = np.zeros(self.n_steps)  # Bag1 activation [0-1]
        self.aBag2 = np.zeros(self.n_steps)  # Bag2 activation [0-1]
        self.aChain = np.zeros(self.n_steps)  # Chain activation [0-1]
        self.T = np.zeros((3, self.n_steps))  # Tensions [Bag1, Bag2, Chain] [FU]
        self.dT = np.zeros((3, self.n_steps))  # Tension rates [FU/s]

        # State variables
        self.step_idx = 0

    def _tension_accel(self, T_val, z_val, L, V, A, b_coef, gamma_force):
        """Compute d²T/dt² for the intrafusal fiber 2nd-order ODE.

        Mileusnic et al. 2006, Eq. 1-3: spring-mass-damper system with
        nonlinear force-velocity relationship.
        """
        p = self.params
        vel_diff = V - z_val / p["K_SR"]
        fv = p["C_L"] if vel_diff >= 0 else p["C_S"]
        sgn = 1.0 if vel_diff >= 0 else -1.0

        fv_term = (fv * b_coef * sgn
                   * abs(vel_diff) ** p["a"]
                   * (L - p["L0_SR"] - T_val / p["K_SR"] - p["R"]))
        spring = p["K_PR"] * (L - p["L0_SR"] - T_val / p["K_SR"] - p["L0_PR"])

        return p["K_SR"] / p["M"] * (
            fv_term + p["M"] * A + gamma_force - T_val + spring
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
        Integrate spindle model for one time step.

        Parameters
        ----------
        muscle_length__L0 : float
            Normalized muscle length
        muscle_velocity__L0_per_s : float
            Muscle velocity [L0/s]
        muscle_acceleration__L0_per_s2 : float
            Muscle acceleration [L0/s²]
        gamma_dynamic_drive__Hz : float
            Dynamic gamma drive [pps]
        gamma_static_drive__Hz : float
            Static gamma drive [pps]

        Returns
        -------
        tuple[float, float]
            (Ia firing rate, II firing rate) in Hz
        """
        p = self.params
        # NEURON stores dt in seconds (dt * 1e-3), tau in seconds — all consistent
        dt_s = self.dt / 1000.0
        i = self.step_idx

        gDyn = gamma_dynamic_drive__Hz
        gStat = gamma_static_drive__Hz
        L = muscle_length__L0
        V = muscle_velocity__L0_per_s
        A = muscle_acceleration__L0_per_s2

        # === FUSIMOTOR ACTIVATIONS (Mileusnic Eq. 4-5) ===
        # Hill-type saturation: gamma^P / (gamma^P + f^P)
        # Gives exactly 0 at zero gamma, saturates to 1 at high gamma.
        P = p["P"]
        target_bag1 = (gDyn ** P / (gDyn ** P + p["fBag1"] ** P)
                       if gDyn > 0 else 0.0)
        target_bag2 = (gStat ** P / (gStat ** P + p["fBag2"] ** P)
                       if gStat > 0 else 0.0)
        # Chain: algebraic (instantaneous, no ODE)
        a_chain = (gStat ** P / (gStat ** P + p["fChain"] ** P)
                   if gStat > 0 else 0.0)

        # Bag1: RK4 integration of da/dt = (target - a) / tau
        if i > 0:
            a_prev = self.aBag1[i - 1]
            tau = p["tau1"]
            k1 = (target_bag1 - a_prev) / tau
            k2 = (target_bag1 - (a_prev + dt_s / 2 * k1)) / tau
            k3 = (target_bag1 - (a_prev + dt_s / 2 * k2)) / tau
            k4 = (target_bag1 - (a_prev + dt_s * k3)) / tau
            a_bag1 = a_prev + dt_s / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            a_bag1 = target_bag1

        # Bag2: RK4 integration of da/dt = (target - a) / tau
        if i > 0:
            a_prev = self.aBag2[i - 1]
            tau = p["tau2"]
            k1 = (target_bag2 - a_prev) / tau
            k2 = (target_bag2 - (a_prev + dt_s / 2 * k1)) / tau
            k3 = (target_bag2 - (a_prev + dt_s / 2 * k2)) / tau
            k4 = (target_bag2 - (a_prev + dt_s * k3)) / tau
            a_bag2 = a_prev + dt_s / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            a_bag2 = target_bag2

        self.aBag1[i] = a_bag1
        self.aBag2[i] = a_bag2
        self.aChain[i] = a_chain

        # === INTRAFUSAL FIBER TENSIONS (2nd-order Mileusnic ODE via RK4) ===
        # Three fiber types: Bag1 (dynamic), Bag2 (static), Chain (static)
        # Each has: dT/dt = z, dz/dt = K_SR/M * [FV_damping + inertial + spring]
        activations = [a_bag1, a_bag2, a_chain]

        for fi in range(3):
            act = activations[fi]

            # Per-fiber damping and gamma force coefficients
            if fi == 0:  # Bag1
                b_coef = p["b0Bag1"] + p["b1Bag1"] * act
                gamma_force = p["G1"] * act
            elif fi == 1:  # Bag2
                b_coef = p["b0Bag2"] + p["b2Bag2"] * act
                gamma_force = p["G2"] * act
            else:  # Chain
                b_coef = p["b0Chain"] + p["b2Chain"] * act
                gamma_force = p["G2Chain"] * act

            T_prev = self.T[fi, i - 1] if i > 0 else 0.0
            z_prev = self.dT[fi, i - 1] if i > 0 else 0.0

            # Coupled RK4 for 2nd-order system
            acc = self._tension_accel
            k1y = z_prev
            k1z = acc(T_prev, z_prev, L, V, A, b_coef, gamma_force)

            k2y = z_prev + dt_s / 2 * k1z
            k2z = acc(T_prev + dt_s / 2 * k1y,
                      z_prev + dt_s / 2 * k1z,
                      L, V, A, b_coef, gamma_force)

            k3y = z_prev + dt_s / 2 * k2z
            k3z = acc(T_prev + dt_s / 2 * k2y,
                      z_prev + dt_s / 2 * k2z,
                      L, V, A, b_coef, gamma_force)

            k4y = z_prev + dt_s * k3z
            k4z = acc(T_prev + dt_s * k3y,
                      z_prev + dt_s * k3z,
                      L, V, A, b_coef, gamma_force)

            self.T[fi, i] = T_prev + dt_s / 6 * (k1y + 2 * k2y + 2 * k3y + k4y)
            self.dT[fi, i] = z_prev + dt_s / 6 * (k1z + 2 * k2z + 2 * k3z + k4z)

        # === AFFERENT FIRING RATES (Mileusnic Eq. 5-6) ===
        threshold = p["LN_SR"] - p["L0_SR"]  # 0.0423 - 0.04 = 0.0023

        # Primary afferent (Ia) — per-fiber contribution from SR stretch
        ia_bag1 = p["gBag1"] * max(0.0, self.T[0, i] / p["K_SR"] - threshold)
        ia_bag2 = p["gBag2A1"] * max(0.0, self.T[1, i] / p["K_SR"] - threshold)
        ia_chain = p["gChainA1"] * max(0.0, self.T[2, i] / p["K_SR"] - threshold)

        # Occlusion (Mileusnic Eq. 6): nonlinear interaction between fiber types
        B2C = ia_bag2 + ia_chain
        if B2C >= ia_bag1:
            Ia_rate = B2C + p["S"] * ia_bag1
        else:
            Ia_rate = ia_bag1 + p["S"] * B2C

        # Secondary afferent (II) — Bag2 and Chain only (Mileusnic Eq. 7)
        # Two components: sensory region stretch + polar region stretch
        II_rate = 0.0
        for fi in range(1, 3):
            gain = p["gBag2A2"] if fi == 1 else p["gChainA2"]
            sr_stretch = self.T[fi, i] / p["K_SR"] - threshold
            pr_stretch = (L - self.T[fi, i] / p["K_SR"]
                          - p["L0_SR"] - p["LN_PR"])
            contribution = gain * (
                p["X"] * p["Lsec"] / p["L0_SR"] * sr_stretch
                + (1 - p["X"]) * p["Lsec"] / p["L0_PR"] * pr_stretch
            )
            II_rate += max(0.0, contribution)

        self.Ia[i] = Ia_rate
        self.II[i] = II_rate

        self.step_idx += 1
        return Ia_rate, II_rate


@beartowertype
class SpindleModel:
    """
    API wrapper for the muscle spindle model - Jaxley Backend.

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

    def _create_spindle_model(self) -> _Spindle__NumPy:
        """
        Create the underlying Spindle model instance.

        This method maps the user-friendly parameter names to the format
        expected by the Spindle constructor.
        """
        return _Spindle__NumPy(
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
            # Activation time constants (Mileusnic 2006, Table 1)
            # Bag1/Bag2: ODE time constants; Chain: algebraic (instantaneous)
            "tau1": 0.149,  # Bag1 activation time constant [s] (dynamic bag)
            "tau2": 0.205,  # Bag2 activation time constant [s] (static bag)
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
