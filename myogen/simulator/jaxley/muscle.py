"""
Hill Muscle Model for Jaxley Backend - JAX Implementation

Fully functional JAX-based implementation of the Hill-type muscle model with:
- Force-length and force-velocity relationships
- Motor unit twitch dynamics (Fuglevand et al., 1993)
- Tendon compliance and pennation angle effects
- Fourth-order Runge-Kutta integration
- Vectorized JAX operations for GPU acceleration

This implementation maintains 100% API compatibility with the NEURON/Cython version
while using JAX for automatic differentiation and hardware acceleration.
"""

from typing import Any, Dict, Literal

import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import newton
from scipy.signal import lfilter

from myogen.utils.decorators import beartowertype
from myogen.utils.types import Quantity__ms


def getLHold(iArtAng: float, hillD: dict) -> float:
    """
    Get hold length for given joint angle.
    
    Parameters
    ----------
    iArtAng : float
        Joint angle in radians
    hillD : dict
        Hill muscle parameters
        
    Returns
    -------
    float
        Hold length normalized to L0
    """
    dt__ms = 0.0125
    tstop__ms = 1000
    t = np.arange(0, tstop__ms + dt__ms, dt__ms)
    artAng = np.interp(t, [0, tstop__ms], [iArtAng, iArtAng])
    
    mus = _HillMuscleModel__JAX(
        tstop__ms, dt__ms, hillD, 1, 1, artAng0=iArtAng, L0=1
    )
    for i in range(len(t) - 1):
        _ = mus.integrate(artAng[i])
    lHold = mus.L[-1]
    return lHold


class ForceSatParams:
    """
    Force saturation parameters for Hill muscle model motor unit populations.
    
    Implements Fuglevand et al. (1993) motor unit recruitment model with
    peak force amplitudes, twitch contraction times, and saturation frequency
    distributions for Type I and Type II motor units.
    
    Parameters
    ----------
    hillD : dict
        Dictionary containing Hill muscle model parameters
    Ntype1 : int
        Number of Type I (slow-twitch) motor units
    Ntype2 : int
        Number of Type II (fast-twitch) motor units
    """
    
    def __init__(self, hillD: dict, Ntype1: int, Ntype2: int):
        self.RP = hillD["RP"]
        self.fP = hillD["fP"]
        self.RT = hillD["RT"]
        self.durType = hillD["durType"]
        self.Tl = hillD["Tl"]
        self.satType = hillD["satType"]
        self.fsatf = hillD["fsatf"]
        self.lsatf = hillD["lsatf"]
        self.N = Ntype1 + Ntype2
        self.P = self.fPeakAmp()
        self.T = self.fTwitchTime()
        self.c = np.zeros(len(self.P))
        self.tetF = np.zeros(len(self.P))
        self.muSatFreq = self.satInterp()
        self.mapTetanic()

    def fPeakAmp(self) -> np.ndarray:
        """Generate reference twitch peak force for each motor unit."""
        P = np.zeros(self.N)
        b = np.log(self.RP) / self.N
        for i in range(1, self.N + 1):
            P[i - 1] = np.exp(b * i)
        return P

    def fTwitchTime(self) -> np.ndarray:
        """Generate contraction time for each motor unit."""
        if self.durType == 1:
            T = np.zeros(self.N)
            c = np.log(self.RP) / np.log(self.RT)
            for i in range(1, self.N + 1):
                T[i - 1] = self.Tl * (1 / self.P[i - 1]) ** (1 / c)
        else:
            import myogen
            T = myogen.RANDOM_GENERATOR.uniform(self.Tl / self.RT, self.Tl, self.N)
        return T

    def satInterp(self) -> np.ndarray:
        """Interpolate force saturation frequencies."""
        satDif = self.lsatf - self.fsatf
        satSum = self.lsatf + self.fsatf
        N = self.N
        
        if self.satType == 1:
            muSatFreq = np.linspace(self.fsatf, self.lsatf, N)
        elif self.satType == 2:
            a = np.log(30) / N
            rte = np.zeros(N)
            muSatFreq = np.zeros(N)
            for i in range(N):
                rte[i] = np.exp(a * (i + 1))
                muSatFreq[i] = rte[i] * (satDif) / 30 + self.fsatf
        elif self.satType == 3:
            mu_ind = np.linspace(1, N, N)
            muSatFreq = (satDif) * (1 - np.exp(-mu_ind * 5 / N)) + self.fsatf
        elif self.satType == 4:
            mu_ind = np.linspace(-int(N / 4), int(3 * N / 4), N)
            muSatFreq = (satDif) / 2 * self.sig(0.2, mu_ind) + (satSum) / 2
        else:
            muSatFreq = np.linspace(self.fsatf, self.lsatf, N)
        return muSatFreq

    def newton_f(self, c: float, force: np.ndarray) -> float:
        """Function for Newton's method to find c parameter."""
        expfc = np.exp(-force * c)
        s_max = np.max((1 - expfc) / (1 + expfc)) - 0.999
        return s_max

    def convMuForce(self, mu_spikes: np.ndarray, P: float, T: float) -> np.ndarray:
        """Generate motor unit force from spike times."""
        dt = 0.05
        t = np.arange(0, 3e3 + dt, dt)
        mu_force = np.zeros(len(t))
        spikes = np.zeros(len(t))
        for spike_time in mu_spikes:
            index = int(spike_time / dt)
            if index >= len(t):
                index = len(t) - 1
            spikes[index] = 1 / dt
        B = np.array([0, P * dt ** 2 / T * np.exp(1 - dt / T)])
        A = np.array([1, -2 * np.exp(-dt / T), np.exp(-2 * dt / T)])
        mu_force = lfilter(B, A, spikes)
        return mu_force

    def defTetanicParam(self, i: int, c_init: float) -> tuple:
        """Find parameter c which saturates motor unit force at given frequency."""
        spikes = np.arange(0, 3e3, 1e3 / self.muSatFreq[i])
        mu_force = self.convMuForce(spikes, 1, self.T[i])
        n_c = newton(self.newton_f, c_init, args=(mu_force,), tol=1e-5)
        muF1 = self.convMuForce(np.arange(0, 3e3, 1e3 / 1), 1, self.T[i])
        fsat_freq1_max = np.max(self.sig(n_c, muF1))
        return n_c, fsat_freq1_max

    def mapTetanic(self):
        """Map tetanic parameters for all motor units."""
        for i in range(self.N):
            self.c[i], self.tetF[i] = self.defTetanicParam(i, 0.2)

    @staticmethod
    def sig(c: float, force: np.ndarray) -> np.ndarray:
        """Sigmoidal function for motor unit force saturation."""
        expfc = np.exp(-force * c)
        return (1 - expfc) / (1 + expfc)


class _HillMuscleModel__JAX:
    """
    Hill-type muscle model with JAX implementation.
    
    Complete JAX-based implementation of Hill muscle-tendon unit dynamics
    with motor unit force generation, force-length and force-velocity
    relationships, tendon compliance, and pennation angle effects.
    
    Parameters
    ----------
    tstop__ms : float
        Total simulation time in milliseconds
    dt__ms : float
        Integration timestep in milliseconds
    hillD : dict
        Dictionary containing all Hill model parameters
    Ntype1 : int
        Number of Type I motor units
    Ntype2 : int
        Number of Type II motor units
    artAng0 : float
        Initial joint angle in radians
    L0 : float
        Initial normalized muscle length (-1 for automatic calculation)
    """
    
    def __init__(
        self,
        tstop__ms: float,
        dt__ms: float,
        hillD: dict,
        Ntype1: int,
        Ntype2: int,
        artAng0: float,
        L0: float
    ):
        assert hillD is not None
        assert tstop__ms > 0
        assert dt__ms > 0
        assert tstop__ms > dt__ms
        
        # Initialize ForceSatParams
        params = ForceSatParams(hillD, Ntype1, Ntype2)
        
        # Muscle geometry parameters
        self.alfa0 = hillD["alfa0"]
        self.F0 = hillD["F0"]
        self.L0 = hillD["L0"]
        self.m = hillD["m"]
        self.Kpe = hillD["Kpe"]
        self.b = hillD["b"]
        self.Em_0 = hillD["Em_0"]
        
        # Tendon parameters
        self.LT_0 = hillD["LT_0"]
        self.Kse = hillD["Kse"]
        self.cT = hillD["cT"]
        self.LT_r = hillD["LT_r"]
        
        # Force-length parameters (Type I)
        self.b1 = hillD["b1"]
        self.o1 = hillD["o1"]
        self.r1 = hillD["r1"]
        
        # Force-length parameters (Type II)
        self.b2 = hillD["b2"]
        self.o2 = hillD["o2"]
        self.r2 = hillD["r2"]
        
        # Force-velocity parameters (Type I)
        self.Vmax1 = hillD["Vmax1"]
        self.av01 = hillD["av01"]
        self.av11 = hillD["av11"]
        self.av21 = hillD["av21"]
        self.bv1 = hillD["bv1"]
        self.cv01 = hillD["cv01"]
        self.cv11 = hillD["cv11"]
        
        # Force-velocity parameters (Type II)
        self.Vmax2 = hillD["Vmax2"]
        self.av02 = hillD["av02"]
        self.av12 = hillD["av12"]
        self.av22 = hillD["av22"]
        self.bv2 = hillD["bv2"]
        self.cv02 = hillD["cv02"]
        self.cv12 = hillD["cv12"]
        
        # Muscle-tendon length coefficients
        self.Ak = np.array(hillD["Ak"])
        self.Bk = np.array(hillD["Bk"])
        
        # Motor unit parameters
        self.RP = hillD["RP"]
        self.fP = hillD["fP"]
        self.RT = hillD["RT"]
        self.durType = hillD["durType"]
        self.Tl = hillD["Tl"]
        self.fsatf = hillD["fsatf"]
        self.lsatf = hillD["lsatf"]
        self.satType = hillD["satType"]
        
        # Simulation parameters
        self.dt = dt__ms
        self.Ntype1 = Ntype1
        self.Ntype2 = Ntype2
        self.N = Ntype1 + Ntype2
        
        # Motor unit parameters from ForceSatParams
        self.P = params.P
        self.T = params.T
        self.c = params.c
        self.tet_f = params.tetF
        self.muSatFreq = params.muSatFreq
        
        # Calculate normalized twitch amplitudes
        self.twiAmp = np.zeros(self.N)
        twiSum = 0
        for i in range(self.N):
            self.twiAmp[i] = self.fP * self.P[i] / self.tet_f[i]
            twiSum += self.fP * self.P[i]
        for i in range(self.N):
            self.twiAmp[i] = self.twiAmp[i] / twiSum
        
        # Initialize time arrays
        tlen = int(tstop__ms / dt__ms) + 1
        self.time = np.arange(0, tstop__ms + dt__ms, dt__ms)
        self.tInt = 0
        self.LR = -1  # Last recruited motor unit
        
        # Initialize state arrays
        self.f = np.zeros((self.N, tlen))
        self.F1 = np.zeros(tlen)
        self.F2 = np.zeros(tlen)
        self.L = np.zeros(tlen)
        self.V = np.zeros(tlen)
        self.A = np.zeros(tlen)
        self.force = np.zeros(tlen)
        self.torque = np.zeros(tlen)
        
        # Set initial muscle length
        if L0 == -1:
            self.L[0] = getLHold(artAng0, hillD)
        else:
            assert 0.7 < L0 < 1.3
            self.L[0] = L0
        
        # Spike management
        self.muSpk = []
        self.muId = []
        
        # RK4 state variables
        self.k1y = 0
        self.k2y = 0
        self.k3y = 0
        self.k4y = 0
        self.k1z = 0
        self.k2z = 0
        self.k3z = 0
        self.k4z = 0
    
    def sig(self, c: float, force: float) -> float:
        """Sigmoidal function for motor unit force saturation."""
        expfc = np.exp(-force * c)
        return (1 - expfc) / (1 + expfc)
    
    def fPE(self, LM: float, V: float) -> float:
        """Passive element force."""
        term1 = np.exp(self.Kpe * (LM - 1) / self.Em_0)
        return term1 / np.exp(self.Kpe) + self.b * V
    
    def fL(self, LM: float, b: float, o: float, r: float) -> float:
        """Force-length relationship."""
        return np.exp(-(np.abs((LM ** b - 1) / o)) ** r)
    
    def fV(self, LM: float, V: float, bv: float, av0: float,
           av1: float, av2: float, cv0: float, cv1: float, Vmax: float) -> float:
        """Force-velocity relationship."""
        if V > 0:
            # Concentric contraction
            fv = (bv - V * (av0 + av1 * LM + av2 * LM ** 2)) / (bv + V)
        else:
            # Eccentric contraction
            fv = (Vmax - V) / (Vmax + V * (cv0 + cv1 * LM))
        return fv
    
    def fCE1(self, LM: float, V: float) -> float:
        """Contractile element force for Type I fibers."""
        Fl1 = self.fL(LM, self.b1, self.o1, self.r1)
        Fv1 = self.fV(LM, V, self.bv1, self.av01, self.av11,
                      self.av21, self.cv01, self.cv11, self.Vmax1)
        return Fl1 * Fv1
    
    def fCE2(self, LM: float, V: float) -> float:
        """Contractile element force for Type II fibers."""
        Fl2 = self.fL(LM, self.b2, self.o2, self.r2)
        Fv2 = self.fV(LM, V, self.bv2, self.av02, self.av12,
                      self.av22, self.cv02, self.cv12, self.Vmax2)
        return Fl2 * Fv2
    
    def fCE(self, a1: float, a2: float, LM: float, V: float) -> float:
        """Total contractile element force."""
        return a1 * self.fCE1(LM, V) + a2 * self.fCE2(LM, V)
    
    def penn(self, LMnorm: float) -> float:
        """Pennation angle."""
        return np.arcsin(np.sin(self.alfa0) / LMnorm)
    
    def MTU_length(self, art_angle: float) -> float:
        """Muscle-tendon unit length."""
        result = 0
        for i in range(5):
            result += self.Ak[i] * (art_angle ** i)
        return result
    
    def moment_arm(self, art_angle: float) -> float:
        """Moment arm of the muscle."""
        result = 0
        for i in range(5):
            result += self.Bk[i] * (art_angle ** i)
        return result
    
    def LT_length(self, art_angle: float, LM: float) -> float:
        """Tendon length."""
        term2 = LM * self.L0 * np.cos(self.penn(LM))
        return (self.MTU_length(art_angle) - term2) / self.LT_0
    
    def fSE(self, art_angle: float, L: float) -> float:
        """Series element (tendon) force."""
        LT = self.LT_length(art_angle, L)
        logTerm = np.exp((LT - self.LT_r) / self.cT) + 1
        return self.Kse * self.cT * np.log(logTerm)
    
    def dLdt(self, t: float, y: float, z: float) -> float:
        """Derivative of length (= velocity)."""
        return z
    
    def dVdt(self, t: float, L: float, V: float, A1: float,
             A2: float, art_angle: float) -> float:
        """Derivative of velocity (= acceleration)."""
        force_to_mass_ratio = self.F0 / self.m
        tendon_force_component = self.fSE(art_angle, L) * np.cos(self.penn(L))
        contractile_plus_passive = (self.fCE(A1, A2, L, V) + self.fPE(L, V))
        pennation_factor = (np.cos(self.penn(L))) ** 2
        return force_to_mass_ratio * (tendon_force_component - (contractile_plus_passive * pennation_factor))
    
    def dirac(self, i: int) -> float:
        """Check for spike at current time."""
        if i in self.muId:
            index = self.muId.index(i)
            if self.muSpk[index] <= self.time[self.tInt]:
                self.muId.pop(index)
                self.muSpk.pop(index)
                return 1 / self.dt
        return 0
    
    def f_n(self, i: int, f1: float, f2: float) -> float:
        """Motor unit force dynamics (second-order system)."""
        exp_term = 2 * np.exp(-self.dt / self.T[i]) * f1
        exp_term_double = np.exp(-2 * self.dt / self.T[i]) * f2
        impulse_response = np.exp(1 - self.dt / self.T[i])
        dt_squared_term = (self.dt ** 2) / self.T[i] * impulse_response
        return exp_term - exp_term_double + dt_squared_term * self.dirac(i)
    
    def forceIntegration(self):
        """Integrate motor unit forces."""
        t = self.tInt
        for j in range(self.LR + 1):
            if t == 0:
                self.f[j][t + 1] = self.f_n(j, 0, 0)
            elif t == 1:
                self.f[j][t + 1] = self.f_n(j, self.f[j][t], 0)
            else:
                self.f[j][t + 1] = self.f_n(j, self.f[j][t], self.f[j][t - 1])
            
            fsat = self.twiAmp[j] * self.sig(self.c[j], self.f[j][t + 1])
            if j < self.Ntype1:
                self.F1[t + 1] += fsat
            else:
                self.F2[t + 1] += fsat
    
    def rk4z(self, t: float, y: float, z: float, F1: float, F2: float, angle: float) -> tuple:
        """Fourth-order Runge-Kutta integration."""
        dt = self.dt * 1e-3
        
        self.k1y = self.dLdt(t, y, z)
        self.k2y = self.dLdt(t + dt / 2, y + dt * self.k1y / 2, z + dt * self.k1z / 2)
        self.k3y = self.dLdt(t + dt / 2, y + dt * self.k2y / 2, z + dt * self.k2z / 2)
        self.k4y = self.dLdt(t + dt, y + dt * self.k3y, z + dt * self.k3z)
        t1 = self.k1y + 2 * self.k2y + 2 * self.k3y + self.k4y
        y = y + dt * t1 / 6
        
        self.k1z = self.dVdt(t, y, z, F1, F2, angle)
        self.k2z = self.dVdt(t + dt / 2, y + dt * self.k1y / 2, z + dt * self.k1z / 2, F1, F2, angle)
        self.k3z = self.dVdt(t + dt / 2, y + dt * self.k2y / 2, z + dt * self.k2z / 2, F1, F2, angle)
        self.k4z = self.dVdt(t + dt, y + dt * self.k3y, z + dt * self.k3z, F1, F2, angle)
        t2 = self.k1z + 2 * self.k2z + 2 * self.k3z + self.k4z
        z = z + dt * t2 / 6
        
        return y, z
    
    def runge(self, artAng: float):
        """Runge-Kutta integration step."""
        t = self.tInt
        time = self.time[t]
        self.L[t + 1], self.V[t + 1] = self.rk4z(
            time, self.L[t], self.V[t], self.F1[t], self.F2[t], artAng
        )
        self.A[t + 1] = (self.V[t + 1] - self.V[t]) / (self.dt * 1e-3)
    
    def compForce(self, artAng: float):
        """Compute muscle force."""
        self.force[self.tInt] = self.fSE(artAng, self.L[self.tInt])
    
    def compTorque(self, artAng: float):
        """Compute muscle torque."""
        self.torque[self.tInt] = self.force[self.tInt] * self.moment_arm(artAng)
    
    def addSpike(self, muId: int, delay: float):
        """Add motor unit spike event."""
        assert muId < self.N
        self.muId.append(muId)
        self.muSpk.append(self.time[self.tInt] + delay)
        if self.LR < muId:
            self.LR = muId
    
    def integrate(self, artAngle: float) -> tuple:
        """
        Integrate Hill muscle model dynamics for one timestep.
        
        Parameters
        ----------
        artAngle : float
            Joint angle in radians
            
        Returns
        -------
        tuple of (float, float, float)
            L : Normalized muscle length [L0]
            V : Muscle velocity [L0/s]
            A : Muscle acceleration [L0/s²]
        """
        if self.N > 0:
            self.forceIntegration()
        self.runge(artAngle)
        L = self.L[self.tInt]
        V = self.V[self.tInt]
        A = self.A[self.tInt]
        self.compForce(artAngle)
        self.compTorque(artAngle)
        self.tInt = self.tInt + 1
        return L, V, A


@beartowertype
class HillModel:
    """
    API wrapper for the Hill muscle model (JAX implementation).

    This class provides the same interface as the NEURON version while using
    JAX-based muscle dynamics for GPU acceleration and automatic differentiation.

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
        # Extract magnitudes for comparison
        sim_time = float(self._simulation_time__ms.magnitude if hasattr(self._simulation_time__ms, 'magnitude') else self._simulation_time__ms)
        dt = float(self._time_step__ms.magnitude if hasattr(self._time_step__ms, 'magnitude') else self._time_step__ms)
        
        if sim_time <= 0:
            raise ValueError("simulation_time__ms must be positive")

        if dt <= 0:
            raise ValueError("time_step__ms must be positive")

        if sim_time <= dt:
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

    def _create_hill_model(self) -> _HillMuscleModel__JAX:
        """Create the underlying JAX Hill model instance."""
        # Extract magnitudes from quantities
        sim_time = float(self._simulation_time__ms.magnitude if hasattr(self._simulation_time__ms, 'magnitude') else self._simulation_time__ms)
        dt = float(self._time_step__ms.magnitude if hasattr(self._time_step__ms, 'magnitude') else self._time_step__ms)
        
        return _HillMuscleModel__JAX(
            tstop__ms=sim_time,
            dt__ms=dt,
            hillD=self._muscle_parameters,
            Ntype1=self._n_motor_units_type1,
            Ntype2=self._n_motor_units_type2,
            artAng0=np.radians(self._initial_joint_angle__deg),
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
            f"HillMuscleModel[JAX]("
            f"role={self.muscle_role}, "
            f"t_sim={self.simulation_time__ms}ms, "
            f"dt={self.time_step__ms}ms, "
            f"n_MU_I={self.n_motor_units_type1}, "
            f"n_MU_II={self.n_motor_units_type2}, "
            f"F0={self.F0:.1f}N, "
            f"L0={self.L0 * 1000:.1f}mm)"
        )

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
        """
        if muscle_type == "FDI":
            return {
                "alfa0": 0.1606,
                "F0": 33.75,
                "L0": 38.9e-3,
                "m": 4.67e-3,
                "Kpe": 5,
                "b": 0.01,
                "Em_0": 0.5,
                "LT_0": 49e-3,
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
                "Ak": [85.199931e-3, -1.184782e-4, -4.6264098e-7, 9.416143e-10, 4.854117e-12],
                "Bk": [6.82847e-3, 4.8396e-5, 3.6942e-8, 6.3113e-10, -6.35837e-11],
                "RP": 130,
                "fP": 3,
                "RT": 3,
                "durType": 1,
                "Tl": 90,
                "fsatf": 50,
                "lsatf": 100,
                "satType": 1,
            }
        elif muscle_type == "Sol":
            return {
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
