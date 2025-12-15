"""
Joint dynamics module for closed-loop neuromechanical control.

This module provides the JointDynamics class that integrates muscle torques
into joint motion using second-order biomechanical equations. This enables
closed-loop control where motor commands drive joint movement through
realistic joint biomechanics.
"""

import numpy as np
from typing import Tuple
from myogen.utils.decorators import beartowertype


@beartowertype
class JointDynamics:
    """
    Joint dynamics integrator for closed-loop neuromechanical control.

    This class implements second-order joint dynamics using the equation:
    I⋅α = τ - B⋅ω - K⋅θ
    where α = angular acceleration, τ = torque, ω = angular velocity,
    θ = joint angle, I = inertia, B = damping, K = stiffness.

    Parameters
    ----------
    inertia__kg_m2 : float
        Joint rotational inertia in kg⋅m².
    damping__Nm_s_per_rad : float
        Joint viscous damping coefficient in N⋅m⋅s/rad.
        Controls velocity-dependent resistance.
    stiffness__Nm_per_rad : float, default=0.0
        Joint elastic stiffness in N⋅m/rad.
        Set to 0 for passive joints, >0 for spring-loaded joints.
    initial_angle__deg : float, default=0.0
        Initial joint angle in degrees.
    initial_velocity__deg_per_s : float, default=0.0
        Initial angular velocity in degrees per second.

    Attributes
    ----------
    angle__rad : float
        Current joint angle in radians.
    velocity__rad_per_s : float
        Current angular velocity in rad/s.
    angle__deg : float
        Current joint angle in degrees (computed property).
    """

    def __init__(
        self,
        inertia__kg_m2: float,
        damping__Nm_s_per_rad: float,
        stiffness__Nm_per_rad: float = 0.0,
        initial_angle__deg: float = 0.0,
        initial_velocity__deg_per_s: float = 0.0,
    ) -> None:
        # Input validation
        if inertia__kg_m2 <= 0:
            raise ValueError(
                f"inertia__kg_m2 must be positive, got {inertia__kg_m2}. "
                "Typical values: finger=0.001-0.01, elbow=0.1-0.5, knee=1.0-5.0 kg⋅m²"
            )

        if damping__Nm_s_per_rad < 0:
            raise ValueError(
                f"damping__Nm_s_per_rad must be non-negative, got {damping__Nm_s_per_rad}. "
                "Typical values: 0.001-0.1 N⋅m⋅s/rad"
            )

        if stiffness__Nm_per_rad < 0:
            raise ValueError(
                f"stiffness__Nm_per_rad must be non-negative, got {stiffness__Nm_per_rad}. "
                "Use 0 for passive joints, >0 for spring-loaded joints"
            )

        # Immutable public parameters
        self.inertia__kg_m2 = inertia__kg_m2
        self.damping__Nm_s_per_rad = damping__Nm_s_per_rad
        self.stiffness__Nm_per_rad = stiffness__Nm_per_rad
        self.initial_angle__deg = initial_angle__deg
        self.initial_velocity__deg_per_s = initial_velocity__deg_per_s

        # Private working copies
        self._inertia = inertia__kg_m2
        self._damping = damping__Nm_s_per_rad
        self._stiffness = stiffness__Nm_per_rad

        # Joint state variables
        self.angle__rad = np.radians(initial_angle__deg)
        self.velocity__rad_per_s = np.radians(initial_velocity__deg_per_s)

    @property
    def angle__deg(self) -> float:
        """Current joint angle in degrees."""
        return np.degrees(self.angle__rad)

    def integrate(self, torque__Nm: float, dt__s: float) -> Tuple[float, float]:
        """
        Integrate joint dynamics for one time step.

        Parameters
        ----------
        torque__Nm : float
            Applied muscle torque in N⋅m.
        dt__s : float
            Integration time step in seconds.

        Returns
        -------
        tuple[float, float]
            Updated (angle__deg, velocity__deg_per_s).
        """
        # Second-order dynamics: I⋅α = τ - B⋅ω - K⋅θ
        spring_torque = -self._stiffness * self.angle__rad
        damping_torque = -self._damping * self.velocity__rad_per_s

        angular_acceleration = (
            torque__Nm + spring_torque + damping_torque
        ) / self._inertia

        # Euler integration (could use RK4 for higher accuracy)
        self.velocity__rad_per_s += angular_acceleration * dt__s
        self.angle__rad += self.velocity__rad_per_s * dt__s

        return self.angle__deg, np.degrees(self.velocity__rad_per_s)

    def reset(self) -> None:
        """Reset joint to initial conditions."""
        self.angle__rad = np.radians(self.initial_angle__deg)
        self.velocity__rad_per_s = np.radians(self.initial_velocity__deg_per_s)

    def get_state(self) -> dict:
        """
        Get current joint state.

        Returns
        -------
        dict
            Dictionary containing current angle, velocity, and parameters.
        """
        return {
            "angle__deg": self.angle__deg,
            "angle__rad": self.angle__rad,
            "velocity__deg_per_s": np.degrees(self.velocity__rad_per_s),
            "velocity__rad_per_s": self.velocity__rad_per_s,
            "inertia__kg_m2": self._inertia,
            "damping__Nm_s_per_rad": self._damping,
            "stiffness__Nm_per_rad": self._stiffness,
        }
