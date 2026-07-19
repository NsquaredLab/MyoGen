"""
Neural Cell Models for Spinal Cord Simulations - Jaxley Backend (API Layer).

This module provides Jaxley-based implementations maintaining full API
compatibility with the NEURON version. This is the API-compatible layer.

For biophysical implementations with actual channel insertion, see:
- myogen.simulator.jaxley.cells.biophysical
"""

import itertools
from typing import Literal, Optional

import numpy as np
import quantities as pq
import jaxley as jx
from jaxley.channels import Leak

import myogen
from myogen.utils.decorators import beartowertype
from myogen.utils.types import Quantity__m, Quantity__m_per_s, Quantity__ms, Quantity__mV


# ============================================================================
# STOCHASTIC SPIKE GENERATORS
# ============================================================================

class _PoissonProcessGenerator__Jaxley:
    """
    Poisson process spike generator for Jaxley cells.
    
    Uses the same algorithm as the Cython implementation: accumulates input
    intensity and compares against an exponentially-distributed threshold.
    
    Parameters
    ----------
    seed : int
        Random seed for reproducibility
    N : int  
        Batch size for threshold generation (affects statistical properties)
    dt : float
        Time step in milliseconds
    """
    
    def __init__(self, seed, N, dt):
        self.N = N  # Batch size for threshold (not max rate!)
        self.dt = dt  # Time step in ms
        self.rng = np.random.RandomState(seed)
        
        # State variables matching Cython implementation
        self.yi = 0.0  # Accumulated input intensity
        self.thres = self._generate_threshold()  # Exponential threshold
    
    def _generate_threshold(self):
        """Generate exponential threshold using batch method."""
        aux = 1.0
        for _ in range(self.N):
            aux *= self.rng.random()
        return -(1.0 / self.N) * np.log(aux)
    
    def compute(self, y):
        """
        Generate Poisson spike for given input rate.
        
        Parameters
        ----------
        y : float
            Input intensity (rate) in Hz at current time step
            
        Returns
        -------
        int
            1 if spike occurs, 0 otherwise
        """
        spk = 0
        # Integrate input intensity: y is rate in Hz, dt is in ms
        self.yi += y * self.dt * 1e-3
        
        # Check if threshold crossed
        if self.yi >= self.thres:
            spk = 1
            self.yi = 0.0  # Reset accumulator
            self.thres = self._generate_threshold()  # New threshold
        
        return spk


class _GammaProcessGenerator__Jaxley:
    """Gamma process spike generator for Jaxley cells."""
    
    def __init__(self, seed, timestep__ms, shape):
        self.timestep__ms = timestep__ms
        self.shape = shape
        self.rng = np.random.RandomState(seed)
        self._t_next = 0.0
        self._t = 0.0
    
    def compute(self, rate__Hz):
        """Generate Gamma spike (0 or 1)."""
        self._t += float(self.timestep__ms.magnitude if hasattr(self.timestep__ms, 'magnitude') else self.timestep__ms)
        
        if self._t >= self._t_next and rate__Hz > 0:
            mean_isi = 1000.0 / rate__Hz  # Convert Hz to ms
            scale = mean_isi / self.shape
            self._t_next = self._t + self.rng.gamma(self.shape, scale)
            return 1
        return 0


# ============================================================================
# BASE CELL CLASS
# ============================================================================

class _Cell:
    """
    Base class for all neural cell models (Jaxley backend).
    
    Maintains full API compatibility with NEURON version while using Jaxley internally.
    """

    _gid__iterator = itertools.count(0)

    @beartowertype
    def __init__(self, class__ID: int, pool__ID: int | None = None):
        self.global__ID = next(self._gid__iterator)
        self.class__ID = class__ID
        self.pool__ID = pool__ID

        self._create_sections()
        self._build_topology()
        self._define_geometry()
        self._define_biophysics()

        self.create_axon()
        self.synapse__list = []

        self.axon_length__m: Quantity__m | None = None
        self.conduction_velocity__m_per_s: Quantity__m_per_s | None = None
        self.axon_delay__ms: Quantity__ms | None = None
        
        # For spike detection
        self._prev_voltage = -70.0
        self.spike_threshold = 0.0

    @beartowertype
    def __repr__(self) -> str:
        return f"{self.__class__.__name__} [global ID: {self.global__ID}, class ID: {self.class__ID}, pool ID: {self.pool__ID if self.pool__ID is not None else 'N/A'}]"

    @beartowertype
    def _create_sections(self):
        """Create Jaxley cell morphology."""
        pass

    @beartowertype
    def _build_topology(self):
        """Connect sections (for multi-compartment cells)."""
        pass

    @beartowertype
    def _define_geometry(self):
        """Set geometric parameters using Jaxley API."""
        pass

    @beartowertype
    def _define_biophysics(self):
        """Insert channels and set biophysical parameters."""
        pass

    @beartowertype
    def create_axon(
        self,
        length__m: Quantity__m = 0 * pq.m,
        conduction_velocity__m_per_s: Quantity__m_per_s = 50 * pq.m / pq.s,
    ):
        """Define axonal conduction properties (API-compatible with NEURON)."""
        self.axon_length__m = length__m
        self.conduction_velocity__m_per_s = conduction_velocity__m_per_s
        self.axon_delay__ms = (self.axon_length__m / self.conduction_velocity__m_per_s).rescale(pq.ms)

    def create_synapses(
        self,
        synapse_location,
        reversal_potential__mV: Quantity__mV = 0 * pq.mV,
        rise_time_constant__ms: Quantity__ms = 0.2 * pq.ms,
        decay_time_constant__ms: Quantity__ms = 0.3 * pq.ms,
    ) -> dict:
        """Create a synapse specification (API-compatible with NEURON).

        Returns a dict ``{tau1, tau2, e, location}`` stored in ``self.synapse__list``.
        This is a connectivity specification, **not** a live Jaxley synapse object.
        Actual synaptic currents in simulation examples are computed analytically
        (exponential conductance kernel × driving force) and injected via
        ``jx.integrate()``. Making these real Jaxley ``Synapse`` objects would require
        a monolithic ``jx.Network`` — incompatible with per-cell integration.
        """
        assert decay_time_constant__ms > rise_time_constant__ms, (
            "decay_time_constant__ms must be greater than tau1"
        )
        
        # Create synapse specification
        syn = {
            "location": synapse_location,
            "tau1": float(rise_time_constant__ms.magnitude if hasattr(rise_time_constant__ms, 'magnitude') else rise_time_constant__ms),
            "tau2": float(decay_time_constant__ms.magnitude if hasattr(decay_time_constant__ms, 'magnitude') else decay_time_constant__ms),
            "e": float(reversal_potential__mV.magnitude if hasattr(reversal_potential__mV, 'magnitude') else reversal_potential__mV),
        }
        
        # NERLab model default
        if hasattr(self, "model") and self.model == "NERLab" and reversal_potential__mV == 0 * pq.mV:
            syn["e"] = 70
        
        self.synapse__list.append(syn)
        return syn
    
    def get_voltage(self) -> float:
        """Get current membrane voltage."""
        if hasattr(self, 'cell') and hasattr(self.cell, 'states'):
            return float(self.cell.states.get('v', -70.0))
        return -70.0
    
    def apply_synaptic_input(self, weight: float, synapse_type: str = "excitatory"):
        """Apply synaptic input (placeholder for actual Jaxley synapse activation)."""
        # This would activate the synapse in actual Jaxley simulation
        pass


# ============================================================================
# INTERNEURONS
# ============================================================================

@beartowertype
class INgII(_Cell):
    """
    Group II interneuron - Jaxley implementation.
    
    Single-compartment spinal interneuron with active ion channels.
    Based on Bui et al. (2003).
    """

    _ids2 = itertools.count(0)

    def __init__(self, class__ID: Optional[int] = None, pool__ID: int | None = None):
        self.soma_branch_idx = 0
        super().__init__(class__ID if class__ID is not None else next(self._ids2), pool__ID)
        self.create_synapses(self.cell)

    def _create_sections(self):
        """Create Jaxley cell with proper Branch structure."""
        comp = jx.Compartment()
        soma_branch = jx.Branch(comp, ncomp=1)
        self.cell = jx.Cell(soma_branch, parents=[-1])
        self.soma = self.cell.branch(self.soma_branch_idx)

    def _define_geometry(self):
        """Set geometry from Bui et al. (2003)."""
        Amu = 81390 + 3113
        Aci = 1.96 * (891.5 + 46.141) / np.sqrt(8)
        i = 0
        A = [Amu - Aci, Amu + Aci]
        D = np.sqrt(A[i] / np.pi)

        self.cell.branch(self.soma_branch_idx).set("radius", D / 2.0)
        self.cell.branch(self.soma_branch_idx).set("length", D)
        self.cell.set("axial_resistivity", 70.0)
        self.cell.set("capacitance", 1.0)

    def _define_biophysics(self):
        """Insert biophysical channels. Default conductances from Bui et al. (2003);
        overridden per-cell by the population pool via .set()."""
        from myogen.simulator.jaxley.channels import Na3rp, KdrRL, MAHP, Gh

        soma = self.cell.branch(self.soma_branch_idx)
        soma.insert(Na3rp())
        soma.insert(KdrRL())
        soma.insert(MAHP())
        soma.insert(Gh())
        soma.insert(Leak())

        # Fresh view required after inserts (Jaxley stale view bug)
        soma = self.cell.branch(self.soma_branch_idx)

        # Na3rp (fast sodium)
        soma.set("Na3rp_gbar", 0.003)
        soma.set("Na3rp_sh", 1.0)
        soma.set("Na3rp_qinf", 8.0)
        soma.set("Na3rp_thinf", -50.0)
        soma.set("Na3rp_ar", 1.0)
        soma.set("Na3rp_eNa", 55.0)

        # KdrRL (delayed rectifier potassium)
        soma.set("KdrRL_gbar", 0.015)
        soma.set("KdrRL_mVh", -21.0)
        soma.set("KdrRL_tmin", 0.8)
        soma.set("KdrRL_taumax", 20.0)
        soma.set("KdrRL_eK", -80.0)

        # MAHP (medium afterhyperpolarization)
        soma.set("MAHP_gcamax", 3e-6)
        soma.set("MAHP_gkcamax", 5e-4)
        soma.set("MAHP_tau_ca", 70.0)
        soma.set("MAHP_mvhalfca", -22.0)
        soma.set("MAHP_mtauca", 2.0)
        soma.set("MAHP_eK", -80.0)
        soma.set("MAHP_eCa", 120.0)

        # Gh (H-current / HCN)
        soma.set("Gh_gbar", 2.5e-5)
        soma.set("Gh_half", -77.0)
        soma.set("Gh_htau", 30.0)
        soma.set("Gh_eH", -41.0)

        # Leak (passive)
        soma.set("Leak_gLeak", 5e-5)
        soma.set("Leak_eLeak", -71.0)
    
    def setup_recording(self) -> None:
        """Set up voltage recording at soma."""
        self.cell.delete_recordings()
        self.cell.branch(self.soma_branch_idx).loc(0.5).record("v")
    
    def setup_stimulus(self, current) -> None:
        """Set up current injection at soma."""
        self.cell.delete_stimuli()
        self.cell.branch(self.soma_branch_idx).loc(0.5).stimulate(current)
    
    def simulate(self, current=None, dt: float = 0.025, t_max: float = 100.0):
        """Run simulation with optional current injection."""
        self.setup_recording()
        if current is not None:
            self.setup_stimulus(current)
        return jx.integrate(self.cell, delta_t=dt, t_max=t_max)


@beartowertype
class INgIb(INgII):
    """Golgi tendon organ (Ib) interneuron."""

    _ids2 = itertools.count(0)

    def __init__(self, pool__ID: int | None = None):
        super().__init__(next(self._ids2), pool__ID)


# ============================================================================
# DESCENDING DRIVE
# ============================================================================

@beartowertype
class DD(_Cell, _PoissonProcessGenerator__Jaxley):
    """Descending drive with Poisson spike generation."""

    _ids2 = itertools.count(0)

    def __init__(self, N, dt, pool__ID: int | None = None):
        class_id = next(self._ids2)
        self.soma_branch_idx = 0
        _Cell.__init__(self, class_id, pool__ID)
        _PoissonProcessGenerator__Jaxley.__init__(
            self, myogen.SEED + (self.class__ID + 1) * (self.global__ID + 1), N, dt
        )

    def __repr__(self) -> str:
        return _Cell.__repr__(self)
    
    def _create_sections(self):
        """Create simple Jaxley cell with proper Branch structure."""
        comp = jx.Compartment()
        soma_branch = jx.Branch(comp, ncomp=1)
        self.cell = jx.Cell(soma_branch, parents=[-1])
        self.ns = self.cell  # Compatibility alias
        
    def _define_geometry(self):
        """Simple point neuron."""
        self.cell.branch(self.soma_branch_idx).set("radius", 5.0)
        self.cell.branch(self.soma_branch_idx).set("length", 10.0)
        self.cell.set("axial_resistivity", 70.0)
        self.cell.set("capacitance", 1.0)
    
    def _define_biophysics(self):
        """Passive only."""
        soma = self.cell.branch(self.soma_branch_idx)
        soma.insert(Leak())
        self.cell.set("Leak_gLeak", 1e-4)
        self.cell.set("Leak_eLeak", -65.0)

    def integrate(self, y):
        """Integrate drive signal to generate spikes."""
        return self.compute(y) if y > 0 else 0


@beartowertype
class DD_Gamma(_Cell, _GammaProcessGenerator__Jaxley):
    """Descending drive with Gamma process spike generation."""

    _ids2 = itertools.count(0)

    def __init__(self, timestep__ms: Quantity__ms, shape: float = 3.0, pool__ID: int | None = None):
        class_id = next(self._ids2)
        self.soma_branch_idx = 0
        _Cell.__init__(self, class_id, pool__ID)
        _GammaProcessGenerator__Jaxley.__init__(
            self, myogen.SEED + (self.class__ID + 1) * (self.global__ID + 1), timestep__ms, shape
        )

    def __repr__(self) -> str:
        return _Cell.__repr__(self)
    
    def _create_sections(self):
        """Create simple Jaxley cell with proper Branch structure."""
        comp = jx.Compartment()
        soma_branch = jx.Branch(comp, ncomp=1)
        self.cell = jx.Cell(soma_branch, parents=[-1])
        self.ns = self.cell  # Compatibility alias
        
    def _define_geometry(self):
        """Simple point neuron."""
        self.cell.branch(self.soma_branch_idx).set("radius", 5.0)
        self.cell.branch(self.soma_branch_idx).set("length", 10.0)
        self.cell.set("axial_resistivity", 70.0)
        self.cell.set("capacitance", 1.0)
    
    def _define_biophysics(self):
        """Passive only."""
        soma = self.cell.branch(self.soma_branch_idx)
        soma.insert(Leak())
        self.cell.set("Leak_gLeak", 1e-4)
        self.cell.set("Leak_eLeak", -65.0)

    def integrate(self, rate__Hz):
        """Integrate firing rate to generate spikes."""
        return self.compute(rate__Hz) if rate__Hz > 0 else 0


# ============================================================================
# AFFERENTS
# ============================================================================

@beartowertype
class AffIa(_Cell, _GammaProcessGenerator__Jaxley):
    """Ia afferent from muscle spindles."""

    _ids2 = itertools.count(0)

    def __init__(
        self,
        RT: float,
        N: int,
        timestep__ms: Quantity__ms,
        initN: int = 0,
        class__ID: int | None = None,
        pool__ID: int | None = None,
    ):
        class_id = next(self._ids2) if class__ID is None else class__ID
        self.soma_branch_idx = 0
        _Cell.__init__(self, class_id, pool__ID)
        
        # Store recruitment threshold and individual firing rate variability
        self.RT = RT
        self.IFR = myogen.RANDOM_GENERATOR.normal(5, 2.5)  # Individual firing rate variability
        
        # Shape parameter for Gamma process (controls ISI regularity)
        shape = N  # Use N as shape parameter
        _GammaProcessGenerator__Jaxley.__init__(
            self, myogen.SEED + (self.class__ID + 1) * (self.global__ID + 1), timestep__ms, shape
        )

    def __repr__(self) -> str:
        return _Cell.__repr__(self)
    
    def _create_sections(self):
        """Create simple Jaxley cell with proper Branch structure."""
        comp = jx.Compartment()
        soma_branch = jx.Branch(comp, ncomp=1)
        self.cell = jx.Cell(soma_branch, parents=[-1])
        self.ns = self.cell
        
    def _define_geometry(self):
        """Afferent fiber geometry."""
        self.cell.branch(self.soma_branch_idx).set("radius", 5.0)
        self.cell.branch(self.soma_branch_idx).set("length", 10.0)
        self.cell.set("axial_resistivity", 70.0)
        self.cell.set("capacitance", 1.0)
    
    def _define_biophysics(self):
        """Passive properties."""
        soma = self.cell.branch(self.soma_branch_idx)
        soma.insert(Leak())
        self.cell.set("Leak_gLeak", 1e-4)
        self.cell.set("Leak_eLeak", -70.0)

    def integrate(self, rate__Hz):
        """Integrate spindle activity to generate spikes."""
        return self.compute(rate__Hz) if rate__Hz > 0 else 0


@beartowertype
class AffII(AffIa):
    """Group II afferent from muscle spindles."""

    _ids2 = itertools.count(0)

    def __init__(
        self,
        RT: float,
        N: int,
        timestep__ms: Quantity__ms,
        initN: int = 0,
        class__ID: int | None = None,
        pool__ID: int | None = None,
    ):
        super().__init__(RT, N, timestep__ms, initN, class__ID, pool__ID)


@beartowertype
class AffIb(AffIa):
    """Ib afferent from Golgi tendon organs."""

    _ids2 = itertools.count(0)

    def __init__(
        self,
        RT: float,
        N: int,
        timestep__ms: Quantity__ms,
        initN: int = 0,
        class__ID: int | None = None,
        pool__ID: int | None = None,
    ):
        super().__init__(RT, N, timestep__ms, initN, class__ID, pool__ID)


# ============================================================================
# MOTOR NEURON
# ============================================================================

# Try importing jaxley-mech channels (Hodgkin-Huxley based)
# Note: In jaxley-mech 0.1.0, the module is 'hh', in 0.3.1+ it's 'hodgkin52'
JAXLEY_MECH_AVAILABLE = False
JaxleyMechNa = None
JaxleyMechK = None
JaxleyMechLeak = None

try:
    # Try newer module path first (jaxley-mech >= 0.3.0)
    from jaxley_mech.channels.hodgkin52 import (
        Na as JaxleyMechNa,
        K as JaxleyMechK,
        Leak as JaxleyMechLeak,
    )
    JAXLEY_MECH_AVAILABLE = True
except ImportError:
    try:
        # Try older module path (jaxley-mech 0.1.0 - 0.2.0)
        from jaxley_mech.channels.hh import (
            Na as JaxleyMechNa,
            K as JaxleyMechK,
            Leak as JaxleyMechLeak,
        )
        JAXLEY_MECH_AVAILABLE = True
    except ImportError:
        pass


@beartowertype
class AlphaMN(_Cell):
    """
    Alpha motor neuron - Jaxley implementation.

    Multi-compartment motor neuron with active channels and PICs.
    For the full biophysical version with actual channel insertion, use:
        from myogen.simulator.jaxley.cells import BiophysicalMotorNeuron

    Implements the Henneman size principle: soma size scales with recruitment
    threshold, creating physiological heterogeneity:
    - Low threshold → small soma → high Rin → low rheobase
    - High threshold → large soma → low Rin → high rheobase

    Parameters
    ----------
    segments__count : int
        Number of segments (compartments).
    mode : str
        "active" for ion channels, "passive" for leak only.
    dendrites__count : int
        Number of dendrites (currently unused in single-compartment mode).
    model : str
        "NERLab" or "Powers2017" for model-specific parameters.
    use_jaxley_mech : bool
        If True, use jaxley-mech Hodgkin-Huxley channels (Na, K, Leak)
        instead of custom motor neuron channels. Default is False.
    class__ID : int, optional
        Class ID for the cell.
    pool__ID : int, optional
        Pool ID for the cell.
    recruitment_threshold : float, optional
        Recruitment threshold for this motor neuron (0-1 range).
        Used to scale soma size according to the size principle.
        If None, uses a default medium-sized soma.
    """

    _ids2 = itertools.count(0)

    def __init__(
        self,
        segments__count: int = 1,
        mode: Literal["active", "passive"] = "active",
        dendrites__count: int = 4,
        model: Literal["NERLab", "Powers2017"] = "NERLab",
        use_jaxley_mech: bool = False,
        gamma: float = 0.2,
        class__ID: Optional[int] = None,
        pool__ID: int | None = None,
        recruitment_threshold: Optional[float] = None,
        neuron_params: Optional[dict] = None,
    ):
        self.dendrites__count = dendrites__count
        self.segments__count = segments__count
        self.mode = mode
        self.model = model
        self.use_jaxley_mech = use_jaxley_mech
        self.gamma = gamma                    # dendritic Ca scale factor (NERLab caL_gama)
        self.soma_branch_idx = 0
        self.recruitment_threshold = recruitment_threshold
        self.neuron_params = neuron_params  # Pre-computed parameters from pool

        # Note: use_jaxley_mech=True no longer requires jaxley-mech package.
        # It uses our custom channels (Na3rp, KdrRL, etc.) with a 2-compartment architecture.

        super().__init__(class__ID if class__ID is not None else next(self._ids2), pool__ID)
        self.create_synapses(self.cell)

        # Motor neuron spike detection threshold (0 mV crossing)
        self.spike_threshold = 0.0

    def _create_sections(self):
        """Create cell morphology.

        Dispatch by ``self.model``:
          - ``"NERLab"``  : soma (ncomp=1) + 1 isopotential dendrite (ncomp=1)
                           — matches the production NEURON architecture.
          - ``"Powers2017"`` + ``use_jaxley_mech=True``: soma + 1 cable
                           dendrite (ncomp=4) — cable filtering of PIC.
          - ``"Powers2017"`` + ``use_jaxley_mech=False``: soma only
                           (single-compartment legacy path).
        """
        comp = jx.Compartment()
        if self.model == "NERLab":
            soma_branch = jx.Branch(comp, ncomp=1)
            dend_branch = jx.Branch(comp, ncomp=1)  # isopotential, matches NEURON
            self.cell = jx.Cell(
                [soma_branch, dend_branch],
                parents=np.array([-1, 0]),
            )
            self.soma = self.cell.branch(0)
            self.dendrite_branch_indices = [1]
            self.dend_ncomp = 1
            self.dend = None
        elif self.use_jaxley_mech:
            soma_branch = jx.Branch(comp, ncomp=1)
            dend_branch = jx.Branch(comp, ncomp=4)  # Single cable, 4 compartments
            self.cell = jx.Cell(
                [soma_branch, dend_branch],
                parents=np.array([-1, 0]),
            )
            self.soma = self.cell.branch(0)
            self.dendrite_branch_indices = [1]  # Single dendrite cable
            self.dend_ncomp = 4
            self.dend = None
        else:
            soma_branch = jx.Branch(comp, ncomp=1)
            self.cell = jx.Cell(soma_branch, parents=[-1])
            self.soma = self.cell.branch(self.soma_branch_idx)
            self.dendrite_branch_indices = []
            self.dend = None

    def _define_geometry(self):
        """Set motor neuron geometry with recruitment-threshold-dependent scaling.

        Implements the Henneman size principle:
        - Low recruitment threshold → small soma → high Rin → low rheobase
        - High recruitment threshold → large soma → low Rin → high rheobase

        When use_jaxley_mech=True: sets geometry for soma (branch 0) AND 4 dendrites (branches 1-4).
        When use_jaxley_mech=False: sets geometry for soma only.

        When neuron_params is provided (from pool), uses pre-computed parameters.
        Otherwise, calculates parameters based on recruitment_threshold.
        """
        # NERLab path: soma is sphere-like (L = diam), uses its own param keys.
        if self.model == "NERLab":
            self._define_geometry_nerlab()
            return

        # Use pre-computed parameters from pool if available
        if self.neuron_params is not None:
            effective_diameter = self.neuron_params["soma_diameter"]
            effective_length = self.neuron_params["soma_length"]
            capacitance = self.neuron_params["capacitance"]
            axial_resistivity = self.neuron_params["axial_resistivity"]
        else:
            # Fallback: calculate from recruitment_threshold (for standalone cells)
            if self.recruitment_threshold is not None:
                scale_factor = np.sqrt(min(self.recruitment_threshold, 1.0))
            else:
                scale_factor = 0.5  # Default to medium-sized neuron

            if self.use_jaxley_mech:
                # Powers2017 soma geometry
                effective_diameter = 22.0 + scale_factor * (30.0 - 22.0)
                effective_length = 2952.0 + scale_factor * (3665.0 - 2952.0)
                capacitance = 1.356 + scale_factor * (1.879 - 1.356)
                axial_resistivity = 0.001  # isopotential soma
            else:
                # Custom channels - scale around NEURON geometry
                effective_diameter = 15.0 + scale_factor * (30.0 - 15.0)
                effective_length = 2000.0 + scale_factor * (4000.0 - 2000.0)
                capacitance = 1.35546  # uF/cm² - matches NEURON soma.cm
                axial_resistivity = 0.001  # ohm-cm - matches NEURON soma.Ra

        # Apply soma geometry
        soma = self.cell.branch(self.soma_branch_idx)
        soma.set("radius", effective_diameter / 2.0)
        soma.set("length", effective_length)
        soma.set("capacitance", capacitance)
        soma.set("axial_resistivity", axial_resistivity)

        # Set dendrite geometry (single cable with ncomp=4)
        # Total area = 4× NEURON single dend (d×2, L×2). Cable Ra provides filtering.
        # Jaxley .set("length", L) sets EACH compartment to L, so divide total by ncomp.
        if self.use_jaxley_mech and self.dendrite_branch_indices:
            dend_idx = self.dendrite_branch_indices[0]
            dend = self.cell.branch(dend_idx)
            ncomp = self.dend_ncomp
            if self.neuron_params is not None:
                dend.set("radius", self.neuron_params["dend_diameter"] / 2.0)
                dend.set("length", self.neuron_params["dend_length"] / ncomp)
                dend.set("capacitance", self.neuron_params["dend_cm"])
                dend.set("axial_resistivity", self.neuron_params["dend_Ra"])
            else:
                # Fallback dendrite geometry (2× NEURON, 4× area cable)
                if self.recruitment_threshold is not None:
                    sf = np.sqrt(min(self.recruitment_threshold, 1.0))
                else:
                    sf = 0.5
                total_d = 17.46 + sf * (23.82 - 17.46)
                total_L = 3588.0 + sf * (4454.0 - 3588.0)
                dend.set("radius", total_d / 2.0)
                dend.set("length", total_L / ncomp)
                dend.set("capacitance", 0.868 + sf * (0.880 - 0.868))
                dend.set("axial_resistivity", 51.04 - sf * (51.04 - 40.76))

    def _define_biophysics(self):
        """Insert channels (active or passive based on mode).

        NERLab path is wholly separate (napp/caL bundle their own passive leak),
        so it doesn't share the Leak()-channel scaffolding used by Powers2017.
        """
        if self.model == "NERLab":
            self._insert_nerlab_channels()
            return

        soma = self.cell.branch(self.soma_branch_idx)

        # Insert Leak on soma
        soma.insert(Leak())

        if self.use_jaxley_mech and self.dendrite_branch_indices:
            # Insert Leak on each dendrite (params set in _insert_jaxley_mech_channels)
            for dend_idx in self.dendrite_branch_indices:
                self.cell.branch(dend_idx).insert(Leak())
        else:
            # Single compartment defaults
            self.cell.set("Leak_gLeak", 5e-5)
            self.cell.set("Leak_eLeak", -70.0)

        if self.mode == "passive":
            return

        # For active mode, insert biophysical channels
        if self.mode == "active":
            if self.use_jaxley_mech:
                self._insert_jaxley_mech_channels()
            else:
                self._insert_active_channels()
    
    # ---------------------------------------------------------------- NERLab

    def _define_geometry_nerlab(self):
        """Set geometry for the NERLab architecture (soma sphere + 1 dendrite cylinder).

        Soma : L = diam (sphere-like, matches NEURON ``cell.soma.L = diam``).
        Dend : single isopotential compartment (ncomp=1).

        Uses per-cell ``neuron_params`` from the pool when available; otherwise
        falls back to mid-range NERLab defaults so a standalone cell still works.
        """
        if self.neuron_params is not None:
            p = self.neuron_params
            soma_diam = float(p["soma_diameter"])
            dend_diam = float(p["dend_diameter"])
            dend_len  = float(p["dend_length"])
            soma_cm   = float(p["soma_capacitance"])
            soma_Ra   = float(p["soma_axial_resistivity"])
            dend_cm   = float(p["dend_capacitance"])
            dend_Ra   = float(p["dend_axial_resistivity"])
        else:
            # Mid-range NERLab fallback (matches scripts/test_nerlab_single_cell.py).
            soma_diam, dend_diam, dend_len = 95.0, 70.0, 8000.0
            soma_cm = dend_cm = 1.0
            soma_Ra = dend_Ra = 70.0

        soma = self.cell.branch(self.soma_branch_idx)
        soma.set("length", soma_diam)
        soma.set("radius", soma_diam / 2.0)
        soma.set("capacitance", soma_cm)
        soma.set("axial_resistivity", soma_Ra)

        dend = self.cell.branch(self.dendrite_branch_indices[0])
        dend.set("length", dend_len)
        dend.set("radius", dend_diam / 2.0)
        dend.set("capacitance", dend_cm)
        dend.set("axial_resistivity", dend_Ra)

    def _insert_nerlab_channels(self):
        """Insert NERLab channels (napp on soma + caL on dendrite).

        NERLab cells live in the 1952 HH voltage frame; cells using this path
        must initialise V at 0 mV (the pool does this automatically when
        ``model="NERLab"``).
        """
        from myogen.simulator.jaxley.channels import napp, caL

        soma = self.cell.branch(self.soma_branch_idx)
        soma.insert(napp())

        dend = self.cell.branch(self.dendrite_branch_indices[0])
        dend.insert(caL())

        # Jaxley stale-view bug (see MEMORY.md): branch views go stale after
        # .insert(); re-fetch fresh views before any .set() calls.
        soma = self.cell.branch(self.soma_branch_idx)
        dend = self.cell.branch(self.dendrite_branch_indices[0])

        # Resolve per-cell params (pool path) or fall back to NERLAB_PARAMS midpoints.
        p = self.neuron_params or {}
        def pick(key, fallback):
            return float(p[key]) if key in p else fallback

        # --- soma napp ---
        soma.set("napp_gnabar",  pick("gnabar",  0.055))
        soma.set("napp_gnapbar", pick("gnapbar", 0.00055))
        soma.set("napp_gkfbar",  pick("gkfbar",  0.0022))
        soma.set("napp_gksbar",  pick("gksbar",  0.018))
        soma.set("napp_gl",      pick("gls",     0.00125))
        soma.set("napp_rinact",  pick("rinact",  0.04))
        soma.set("napp_el",      pick("el_napp",     0.0))
        soma.set("napp_vtraub",  pick("vtraub_napp", 0.0))
        soma.set("napp_ena",     pick("ena",   120.0))
        soma.set("napp_ek",      pick("ek",    -10.0))

        # --- dendrite caL ---
        dend.set("caL_gcaLbar", pick("gcaLbar",    9e-6))
        dend.set("caL_gl",      pick("gl_caL",     1.2e-4))
        dend.set("caL_el",      pick("el_caL",     0.0))
        dend.set("caL_ecaL",    pick("ecaL",       140.0))
        dend.set("caL_vtraub",  pick("vtraub_caL", 34.5))
        dend.set("caL_Ltau",    pick("ltau_caL",   65.0))
        # gama scales dendritic Ca conductance. NEURON's NERLab uses gamma=0.2
        # (config/alpha_mn_default.yaml); a hard-coded 1.0 was making the PIC
        # ~5× too strong, contributing to excess firing.
        dend.set("caL_gama",    self.gamma)

    def _insert_jaxley_mech_channels(self):
        """Insert Powers2017 channels on soma + 4 dendrites.

        Soma (branch 0): Na3rp, Naps, KdrRL, MAHP, Gh, Leak
        Dendrites (branches 1-4): LCaInact (dendritic PIC), Gh, Leak

        Single multi-compartment dendrite cable (ncomp=4) with cable-filtered PIC.
        Cable Ra provides temporal filtering of PIC dynamics in place of NEURON's discrete
        4-isopotential-dendrite topology.

        When neuron_params is provided (from pool), uses pre-computed conductances.
        Otherwise, calculates conductances based on recruitment_threshold.
        """
        from myogen.simulator.jaxley.channels import (
            Na3rp, Naps, KdrRL, MAHP, Gh, LCaInact,
        )

        soma = self.cell.branch(self.soma_branch_idx)

        # Get parameters (from pool or fallback)
        if self.neuron_params is not None:
            p = self.neuron_params
        else:
            # Fallback: midpoint parameters for standalone cell
            if self.recruitment_threshold is not None:
                sf = np.sqrt(min(self.recruitment_threshold, 1.0))
            else:
                sf = 0.5
            p = {
                "gNa3rp": 0.01 + sf * (0.022 - 0.01),
                "gNaPS": 2.6e-5 - sf * (2.6e-5 - 2.0e-5),
                "gKdrRL": 0.015 + sf * (0.02 - 0.015),
                "gMAHP_ca": 6.4e-6 + sf * (1.015e-5 - 6.4e-6),
                "gMAHP_k": 5.0625e-4 + sf * (6.75e-4 - 5.0625e-4),
                "tauMAHP": 90.0 - sf * (90.0 - 30.0),
                "gGh_soma": 3.0e-5 + sf * (2.3e-4 - 3.0e-5),
                "gLeak_soma": 1.5e-4 + sf * (3.77e-4 - 1.5e-4),
                "eLeak_soma": -71.0 - sf * 1.0,
                # Dendrite: NEURON native density (4× area from cable geometry)
                "gLeak_dend": 7.93e-5 + sf * (1.75e-4 - 7.93e-5),
                "eLeak_dend": -71.0 - sf * 1.0,
                "gGh_dend": 3.0e-5 + sf * (2.3e-4 - 3.0e-5),
                "gLCa_dend": 7.21875e-5 + sf * (9.69375e-5 - 7.21875e-5),
                "theta_m_LCa": -42.0 + sf * (-39.0 - (-42.0)),
                "theta_h_LCa": 10.0 + sf * (-10.0 - 10.0),
                "eNa": 55.0, "eK": -80.0, "eCa": 120.0,
            }

        # === INSERT ALL CHANNELS FIRST ===
        # (branch views become stale after insert, so insert all then get fresh views)

        # Soma channels
        soma.insert(Na3rp())
        soma.insert(Naps())
        soma.insert(KdrRL())
        soma.insert(MAHP())
        soma.insert(Gh())

        # Dendrite channels (single cable — inserts on all ncomp compartments)
        dend_idx = self.dendrite_branch_indices[0]
        self.cell.branch(dend_idx).insert(LCaInact())
        self.cell.branch(dend_idx).insert(Gh())

        # === SET PARAMETERS (fresh views after all inserts) ===
        soma = self.cell.branch(self.soma_branch_idx)

        # Na3rp (fast sodium with persistent component)
        soma.set("Na3rp_gbar", p["gNa3rp"])
        soma.set("Na3rp_sh", 1.0)
        soma.set("Na3rp_qinf", 8.0)
        soma.set("Na3rp_thinf", -50.0)
        soma.set("Na3rp_ar", 1.0)
        soma.set("Na3rp_eNa", p["eNa"])

        # Naps (persistent sodium)
        soma.set("Naps_gbar", p["gNaPS"])
        soma.set("Naps_sh", 5.0)
        soma.set("Naps_vslope", 5.0)
        soma.set("Naps_ar", 1.0)
        soma.set("Naps_asvh", -90.0)   # NEURON value
        soma.set("Naps_bsvh", -22.0)   # NEURON value
        soma.set("Naps_eNa", p["eNa"])

        # KdrRL (delayed rectifier potassium)
        soma.set("KdrRL_gbar", p["gKdrRL"])
        soma.set("KdrRL_mVh", -21.0)
        soma.set("KdrRL_tmin", 0.8)
        soma.set("KdrRL_taumax", 20.0)
        soma.set("KdrRL_eK", p["eK"])

        # MAHP (medium afterhyperpolarization)
        soma.set("MAHP_gcamax", p["gMAHP_ca"])
        soma.set("MAHP_gkcamax", p["gMAHP_k"])
        soma.set("MAHP_tau_ca", p["tauMAHP"])
        soma.set("MAHP_mvhalfca", -22.0)   # NEURON global override (h.mvhalfca_mAHP = -22)
        soma.set("MAHP_mtauca", 2.0)
        soma.set("MAHP_eK", p["eK"])
        soma.set("MAHP_eCa", p["eCa"])

        # Gh — soma
        soma.set("Gh_gbar", p["gGh_soma"])
        soma.set("Gh_half", -77.0)
        soma.set("Gh_htau", 30.0)
        soma.set("Gh_eH", -41.0)

        # Leak — soma (already inserted in _define_biophysics)
        soma.set("Leak_gLeak", p["gLeak_soma"])
        soma.set("Leak_eLeak", p["eLeak_soma"])

        # === DENDRITE PARAMETERS (single cable, all 4 compartments uniform) ===
        # Cable Ra provides temporal filtering — no theta_m offsets needed.
        dend = self.cell.branch(self.dendrite_branch_indices[0])

        # LCaInact (L-type calcium with inactivation — THE dendritic PIC)
        dend.set("LCaInact_gbar", p["gLCa_dend"])
        dend.set("LCaInact_theta_m", p["theta_m_LCa"])
        dend.set("LCaInact_theta_h", p["theta_h_LCa"])
        dend.set("LCaInact_tau_m", 40.0)
        dend.set("LCaInact_tau_h", 800.0)  # Faster than NEURON (2500); steep kappa_h provides selectivity
        dend.set("LCaInact_kappa_h", 3.0)  # Steeper than NEURON (5) for voltage-selective inactivation at plateau
        dend.set("LCaInact_kappa_m", -6.0)
        dend.set("LCaInact_eCa", p["eCa"])

        # Gh — dendrite
        dend.set("Gh_gbar", p["gGh_dend"])
        dend.set("Gh_half", -77.0)
        dend.set("Gh_htau", 30.0)
        dend.set("Gh_eH", -41.0)

        # Leak — dendrite (already inserted in _define_biophysics)
        dend.set("Leak_gLeak", p["gLeak_dend"])
        dend.set("Leak_eLeak", p["eLeak_dend"])

    def _insert_active_channels(self):
        """Insert active ion channels for biophysical simulation."""
        from myogen.simulator.jaxley.channels import (
            Na3rp, Naps, KdrRL, MAHP, Gh
        )
        
        soma = self.cell.branch(self.soma_branch_idx)
        
        # Insert channels with default parameters
        soma.insert(Na3rp())
        soma.insert(Naps())
        soma.insert(KdrRL())
        soma.insert(MAHP())
        soma.insert(Gh())
        
        # Set model-specific parameters using cell.set()
        # Parameters tuned for large effective cell (~628,000 µm² membrane)
        # These are scaled appropriately for 10-20 nA input currents
        if self.model == "NERLab":
            # Na3rp parameters - Fast sodium for action potentials
            # MATCHED TO NEURON Powers2017: gbar=0.01, sh=1.0, qinf=8.0
            self.cell.set("Na3rp_gbar", 0.01)      # S/cm² - matches NEURON
            self.cell.set("Na3rp_sh", 1.0)         # mV - threshold shift (matches NEURON)
            self.cell.set("Na3rp_ar", 1.0)         # slow inact recovery (1 = no slow inact)
            self.cell.set("Na3rp_eNa", 55.0)       # mV
            self.cell.set("Na3rp_thinf", -50.0)    # mV - matches NEURON thinf_na3rp
            self.cell.set("Na3rp_qinf", 8.0)       # mV - CRITICAL: matches NEURON qinf_na3rp

            # Naps parameters - Persistent sodium (small)
            self.cell.set("Naps_gbar", 2.6e-05)    # S/cm² - matches NEURON
            self.cell.set("Naps_sh", 5.0)          # mV - matches NEURON
            self.cell.set("Naps_ar", 1.0)
            self.cell.set("Naps_eNa", 55.0)

            # KdrRL parameters - Delayed rectifier for repolarization
            # MATCHED TO NEURON Powers2017: gMax=0.015
            self.cell.set("KdrRL_gbar", 0.015)     # S/cm² - matches NEURON
            self.cell.set("KdrRL_mVh", -21.0)      # mV - matches NEURON global h.mVh_kdrRL
            self.cell.set("KdrRL_mslp", 20.0)      # mV
            self.cell.set("KdrRL_eK", -80.0)       # mV
            self.cell.set("KdrRL_tmin", 0.8)       # ms - matches NEURON global h.tmin_kdrRL
            self.cell.set("KdrRL_taumax", 20.0)    # ms - matches NEURON global h.taumax_kdrRL

            # MAHP parameters - Medium AHP (provides spike adaptation)
            # MATCHED TO NEURON Powers2017
            self.cell.set("MAHP_gkcamax", 0.00045) # S/cm² - matches NEURON
            self.cell.set("MAHP_gcamax", 6.4e-6)   # S/cm² - matches NEURON
            self.cell.set("MAHP_mvhalfca", -22.0)  # mV - matches NEURON global
            self.cell.set("MAHP_mslpca", 4.0)
            self.cell.set("MAHP_tau_ca", 90.0)     # ms - matches NEURON
            self.cell.set("MAHP_eK", -80.0)
            self.cell.set("MAHP_eCa", 120.0)

            # Gh parameters - H-current for sag
            self.cell.set("Gh_gbar", 3e-5)         # S/cm² - matches NEURON
            self.cell.set("Gh_half", -77.0)        # mV - matches NEURON
            self.cell.set("Gh_slp", 8.0)           # mV
            self.cell.set("Gh_eH", -41.0)          # mV
            self.cell.set("Gh_htau", 30.0)         # ms - matches NEURON global
            
        else:  # Powers2017
            # Na3rp parameters - MATCHED TO NEURON Powers2017
            self.cell.set("Na3rp_gbar", 0.01)      # S/cm² - matches NEURON
            self.cell.set("Na3rp_sh", 1.0)         # mV - matches NEURON
            self.cell.set("Na3rp_ar", 1.0)
            self.cell.set("Na3rp_eNa", 55.0)
            self.cell.set("Na3rp_thinf", -50.0)    # mV - matches NEURON
            self.cell.set("Na3rp_qinf", 8.0)       # mV - CRITICAL: matches NEURON

            # Naps parameters - MATCHED TO NEURON Powers2017
            self.cell.set("Naps_gbar", 2.6e-05)    # S/cm² - matches NEURON
            self.cell.set("Naps_sh", 5.0)          # mV - matches NEURON
            self.cell.set("Naps_ar", 1.0)
            self.cell.set("Naps_eNa", 55.0)

            # KdrRL parameters - MATCHED TO NEURON Powers2017
            self.cell.set("KdrRL_gbar", 0.015)     # S/cm² - matches NEURON
            self.cell.set("KdrRL_mVh", -21.0)      # mV - matches NEURON global
            self.cell.set("KdrRL_mslp", 20.0)
            self.cell.set("KdrRL_eK", -80.0)
            self.cell.set("KdrRL_tmin", 0.8)       # ms - matches NEURON global h.tmin_kdrRL
            self.cell.set("KdrRL_taumax", 20.0)    # ms - matches NEURON global h.taumax_kdrRL

            # MAHP parameters - MATCHED TO NEURON Powers2017
            self.cell.set("MAHP_gkcamax", 0.00045) # S/cm² - matches NEURON
            self.cell.set("MAHP_gcamax", 6.4e-6)   # S/cm² - matches NEURON
            self.cell.set("MAHP_mvhalfca", -22.0)  # mV - matches NEURON global
            self.cell.set("MAHP_mslpca", 4.0)
            self.cell.set("MAHP_tau_ca", 90.0)     # ms - matches NEURON
            self.cell.set("MAHP_eK", -80.0)
            self.cell.set("MAHP_eCa", 120.0)

            # Gh parameters - MATCHED TO NEURON Powers2017
            self.cell.set("Gh_gbar", 3e-5)         # S/cm² - matches NEURON
            self.cell.set("Gh_half", -77.0)        # mV - matches NEURON
            self.cell.set("Gh_slp", 8.0)
            self.cell.set("Gh_eH", -41.0)
            self.cell.set("Gh_htau", 30.0)         # ms - matches NEURON global
    
    def setup_recording(self) -> None:
        """Set up voltage recording at soma."""
        self.cell.delete_recordings()
        self.cell.branch(self.soma_branch_idx).loc(0.5).record("v")
    
    def setup_stimulus(self, current) -> None:
        """Set up current injection at soma."""
        self.cell.delete_stimuli()
        self.cell.branch(self.soma_branch_idx).loc(0.5).stimulate(current)
    
    def simulate(self, current=None, dt: float = 0.025, t_max: float = 100.0):
        """
        Run simulation with optional current injection.
        
        Parameters
        ----------
        current : array, optional
            Current waveform (nA).
        dt : float
            Time step (ms).
        t_max : float
            Total time (ms).
        
        Returns
        -------
        voltages : array
            Recorded voltages.
        """
        self.setup_recording()
        
        if current is not None:
            self.setup_stimulus(current)
        
        return jx.integrate(self.cell, delta_t=dt, t_max=t_max)
