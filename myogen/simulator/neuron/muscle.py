"""
Hill Muscle Model API Wrapper

This module provides a clean API wrapper for the Hill muscle model,
allowing for intuitive parameter names while maintaining compatibility
with the underlying Hill implementation.
"""

from typing import Any, Dict, Literal

import numpy as np

from myogen.simulator.neuron._cython._hill import _HillMuscleModel__Cython
from myogen.utils.decorators import beartowertype
from myogen.utils.types import Quantity__ms


@beartowertype
class HillModel:
    """
    API wrapper for the Hill muscle model.

    This class provides an intuitive interface for creating Hill muscle models
    with user-friendly parameter names that are internally mapped to the
    correct format expected by the underlying Hill implementation.

    Parameters
    ----------
    simulation_time__ms : float
        Total simulation time in milliseconds
    time_step__ms : float
        Integration time step in milliseconds
    muscle_parameters : Dict[str, Any]
        Dictionary containing Hill muscle model parameters
    n_motor_units_type1 : int
        Number of type I motor units
    n_motor_units_type2 : int
        Number of type II motor units
    initial_joint_angle__deg : float
        Initial joint angle in degrees
    initial_muscle_length__L0 : float, optional
        Initial muscle length normalized to L0. If -1, automatically calculated
        from joint angle. Must be between 0.7 and 1.3 if specified.
    muscle_role : str, optional
        Muscle role for antagonist pairs ("flexor" or "extensor"), by default "flexor".
        Used for joint dynamics calculations and result organization.
    """

    @beartowertype
    def __init__(
        self,
        simulation_time__ms: Quantity__ms,
        time_step__ms: Quantity__ms,
        muscle_parameters: Dict[str, Any],
        n_motor_units_type1: int,
        n_motor_units_type2: int,
        initial_joint_angle__deg: float,
        initial_muscle_length__L0: float = -1.0,
        muscle_role: Literal["flexor", "extensor"] = "flexor",
    ):
        # Store original parameters (immutable)
        self.simulation_time__ms = simulation_time__ms
        self.time_step__ms = time_step__ms
        self.muscle_parameters = muscle_parameters.copy()
        self.n_motor_units_type1 = n_motor_units_type1
        self.n_motor_units_type2 = n_motor_units_type2
        self.initial_joint_angle__deg = initial_joint_angle__deg
        self.initial_muscle_length__L0 = initial_muscle_length__L0
        self.muscle_role = muscle_role

        # Private working copies for internal use
        self._simulation_time__ms = simulation_time__ms
        self._time_step__ms = time_step__ms
        self._muscle_parameters = muscle_parameters.copy()
        self._n_motor_units_type1 = n_motor_units_type1
        self._n_motor_units_type2 = n_motor_units_type2
        self._initial_joint_angle__deg = initial_joint_angle__deg
        self._initial_muscle_length__L0 = initial_muscle_length__L0
        self._muscle_role = muscle_role

        # Validate inputs
        self._validate_parameters()

        # Create the underlying Hill model
        self._hill_model = self._create_hill_model()

    def _validate_parameters(self) -> None:
        """Validate input parameters."""
        if self._simulation_time__ms <= 0:
            raise ValueError("simulation_time__ms must be positive")

        if self._time_step__ms <= 0:
            raise ValueError("time_step__ms must be positive")

        if self._simulation_time__ms <= self._time_step__ms:
            raise ValueError("simulation_time__ms must be greater than time_step__ms")

        if self._n_motor_units_type1 < 0:
            raise ValueError("n_motor_units_type1 must be non-negative")

        if self._n_motor_units_type2 < 0:
            raise ValueError("n_motor_units_type2 must be non-negative")

        if self._initial_muscle_length__L0 != -1.0 and (
            self._initial_muscle_length__L0 < 0.7 or self._initial_muscle_length__L0 > 1.3
        ):
            raise ValueError("initial_muscle_length__L0 must be -1 or between 0.7 and 1.3")

        if self._muscle_role not in ["flexor", "extensor"]:
            raise ValueError("muscle_role must be 'flexor' or 'extensor'")

    def _create_hill_model(self) -> _HillMuscleModel__Cython:
        """
        Create the underlying Hill model instance.

        This method maps the user-friendly parameter names to the format
        expected by the Hill constructor.
        """

        # Create Hill model with mapped parameters
        return _HillMuscleModel__Cython(
            tstop__ms=self._simulation_time__ms,
            dt__ms=self._time_step__ms,
            hillD=self._muscle_parameters,
            Ntype1=self._n_motor_units_type1,
            Ntype2=self._n_motor_units_type2,
            artAng0=np.radians(self._initial_joint_angle__deg),  # Convert to radians
            L0=self._initial_muscle_length__L0,
        )

    def add_spike(self, motor_unit_id: int, delay__ms: float = 0.0) -> None:
        """
        Add a spike event for a specific motor unit.

        Parameters
        ----------
        motor_unit_id : int
            ID of the motor unit (0-based index)
        delay__ms : float, optional
            Spike delay in milliseconds, by default 0.0
        """
        self._hill_model.addSpike(motor_unit_id, delay__ms)

    def integrate(self, joint_angle__deg: float) -> tuple[float, float, float]:
        """
        Integrate the muscle model for one time step.

        Parameters
        ----------
        joint_angle__deg : float
            Current joint angle in degrees

        Returns
        -------
        tuple[float, float, float]
            Muscle length (normalized to L0), velocity (L0/s), acceleration (L0/s^2)
        """
        return self._hill_model.integrate(np.radians(joint_angle__deg))

    @property
    def muscle_length(self) -> np.ndarray:
        """Get muscle length time series (normalized to L0)."""
        return np.asarray(self._hill_model.L)

    @property
    def muscle_velocity(self) -> np.ndarray:
        """Get muscle velocity time series (L0/s)."""
        return np.asarray(self._hill_model.V)

    @property
    def muscle_acceleration(self) -> np.ndarray:
        """Get muscle acceleration time series (L0/s^2)."""
        return np.asarray(self._hill_model.A)

    @property
    def muscle_force(self) -> np.ndarray:
        """Get muscle force time series (normalized to F0)."""
        return np.asarray(self._hill_model.force)

    @property
    def muscle_torque(self) -> np.ndarray:
        """Get muscle torque time series (F0*m)."""
        return np.asarray(self._hill_model.torque)

    @property
    def signed_muscle_torque(self) -> np.ndarray:
        """Get muscle torque with correct sign for joint dynamics (F0*m)."""
        torque = np.asarray(self._hill_model.torque)
        # Extensor muscles produce negative torque (opposing flexion)
        return -torque if self._muscle_role == "extensor" else torque

    @property
    def type1_activation(self) -> np.ndarray:
        """Get Type I motor unit activation time series."""
        return np.asarray(self._hill_model.F1)

    @property
    def type2_activation(self) -> np.ndarray:
        """Get Type II motor unit activation time series."""
        return np.asarray(self._hill_model.F2)

    @property
    def motor_unit_forces(self) -> np.ndarray:
        """Get individual motor unit forces matrix (N_units x time_points)."""
        return np.asarray(self._hill_model.f)

    @property
    def time_vector(self) -> np.ndarray:
        """Get simulation time vector in milliseconds."""
        return np.asarray(self._hill_model.time)

    @property
    def F0(self) -> float:
        """Get maximum isometric force (F0) in Newtons."""
        return self._hill_model.F0

    @property
    def L0(self) -> float:
        """Get optimal muscle length (L0) in meters."""
        return self._hill_model.L0

    def __repr__(self) -> str:
        """String representation of the Hill muscle model."""
        return (
            f"HillMuscleModel("
            f"role={self.muscle_role}, "
            f"t_sim={self.simulation_time__ms}ms, "
            f"dt={self.time_step__ms}ms, "
            f"n_MU_I={self.n_motor_units_type1}, "
            f"n_MU_II={self.n_motor_units_type2}, "
            f"F0={self.F0:.1f}N, "
            f"L0={self.L0 * 1000:.1f}mm)"
        )

    # Convenience function for creating default muscle parameter dictionaries
    @staticmethod
    def create_default_muscle_parameters(muscle_type: str = "FDI") -> Dict[str, Any]:
        """
        Create default muscle parameter dictionary.

        Parameters
        ----------
        muscle_type : str, optional
            Type of muscle model ("FDI", "Sol"), by default "FDI"

        Returns
        -------
        Dict[str, Any]
            Dictionary of muscle parameters

        Raises
        ------
        ValueError
            If muscle_type is not recognized
        """
        if muscle_type == "FDI":
            return {
                # Muscle geometry
                "alfa0": 0.1606,  # Initial pennation angle [rad]
                "F0": 33.75,  # Maximum isometric force [N]
                "L0": 38.9e-3,  # Optimal fascicle length [m]
                "m": 4.67e-3,  # Muscle mass [kg]
                # Passive elements
                "Kpe": 5,  # Passive elastic element stiffness [F0/L0]
                "b": 0.01,  # Muscle fiber viscous element [F0*s/L0]
                "Em_0": 0.5,  # Normalized muscle deformation
                # Tendon parameters
                "LT_0": 49e-3,  # Tendon length for max isometric force [m]
                "Kse": 27.8,  # Tendon elastic element [F0/LT_0]
                "cT": 0.0047,  # Toe region coefficient
                "LT_r": 0.964,  # Linear region start [LT_0]
                # Force-Length curve parameters (Type I fibers)
                # F_L = exp(-|((L^b - 1)/o)|^r) where L is normalized length
                "b1": 2.3,  # Shape parameter for Type I length-force curve
                "o1": 1.12,  # Width parameter for Type I length-force curve
                "r1": 1.62,  # Asymmetry parameter for Type I length-force curve
                # Force-Length curve parameters (Type II fibers)
                "b2": 1.55,  # Shape parameter for Type II length-force curve
                "o2": 0.75,  # Width parameter for Type II length-force curve
                "r2": 2.12,  # Asymmetry parameter for Type II length-force curve
                # Force-Velocity curve parameters (Type I fibers)
                # For concentric: F_V = (bv - V*(av0 + av1*L + av2*L²))/(bv + V)
                # For eccentric: F_V = (Vmax - V)/(Vmax + V*(cv0 + cv1*L))
                "Vmax1": -7.88,  # Maximum shortening velocity for Type I [L0/s]
                "av01": -4.7,  # Concentric velocity coefficient a0 for Type I
                "av11": 8.41,  # Concentric velocity coefficient a1 for Type I (length-dependent)
                "av21": -5.34,  # Concentric velocity coefficient a2 for Type I (length²-dependent)
                "bv1": 0.35,  # Concentric force-velocity scaling for Type I
                "cv01": 5.88,  # Eccentric velocity coefficient c0 for Type I
                "cv11": 0,  # Eccentric velocity coefficient c1 for Type I (length-dependent)
                # Force-Velocity curve parameters (Type II fibers)
                "Vmax2": -9.15,  # Maximum shortening velocity for Type II [L0/s]
                "av02": -1.53,  # Concentric velocity coefficient a0 for Type II
                "av12": 0,  # Concentric velocity coefficient a1 for Type II
                "av22": 0,  # Concentric velocity coefficient a2 for Type II
                "bv2": 0.69,  # Concentric force-velocity scaling for Type II
                "cv02": 5.7,  # Eccentric velocity coefficient c0 for Type II
                "cv12": 9.18,  # Eccentric velocity coefficient c1 for Type II
                # Muscle-tendon length and moment arm coefficients
                "Ak": [
                    85.199931e-3,
                    -1.184782e-4,
                    -4.6264098e-7,
                    9.416143e-10,
                    4.854117e-12,
                ],
                "Bk": [6.82847e-3, 4.8396e-5, 3.6942e-8, 6.3113e-10, -6.35837e-11],
                # Motor unit parameters
                "RP": 130,  # Range of twitch force amplitude
                "fP": 3,  # First peak twitch force [mN]
                "RT": 3,  # Range of contraction time
                "durType": 1,  # Distribution type (1=exponential)
                "Tl": 90,  # Longest twitch duration [ms]
                "fsatf": 50,  # First MU saturation frequency [Hz]
                "lsatf": 100,  # Last MU saturation frequency [Hz]
                "satType": 1,  # Saturation type (1=linear)
            }

        elif muscle_type == "Sol":
            return {
                # Soleus muscle parameters (larger, stronger muscle)
                "alfa0": 0.494,
                "F0": 3586,
                "L0": 49e-3,
                "m": 0.526,
                "Kpe": 5,
                "b": 0.005,
                "Em_0": 0.5,
                "LT_0": 0.289,
                "Kse": 27.8,
                "cT": 0.0047,
                "LT_r": 0.964,
                "b1": 2.3,
                "o1": 1.12,
                "r1": 1.62,
                "b2": 1.55,
                "o2": 0.75,
                "r2": 2.12,
                "Vmax1": -7.88,
                "av01": -4.7,
                "av11": 8.41,
                "av21": -5.34,
                "bv1": 0.35,
                "cv01": 5.88,
                "cv11": 0,
                "Vmax2": -9.15,
                "av02": -1.53,
                "av12": 0,
                "av22": 0,
                "bv2": 0.69,
                "cv02": 5.7,
                "cv12": 9.18,
                "Ak": [0.323, 7.219e-4, -2.243e-6, -3.148e-8, 9.274e-11],
                "Bk": [-0.041, 2.574e-4, 5.451e-6, -2.219e-8, -5.494e-11],
                "RP": 130,
                "fP": 3,
                "RT": 3,
                "durType": 1,
                "Tl": 90,
                "fsatf": 50,
                "lsatf": 100,
                "satType": 1,
            }

        else:
            raise ValueError(f"Unknown muscle type: {muscle_type}. Use 'FDI' or 'Sol'.")
