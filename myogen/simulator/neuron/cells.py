"""
Neural Cell Models for Spinal Cord Simulations.
"""

import itertools
from typing import Literal, Optional

import numpy as np
import quantities as pq
from neuron import h

from myogen import RANDOM_GENERATOR, SEED
from myogen.simulator.neuron._cython._gamma_process_generator import (
    _GammaProcessGenerator__Cython,
)
from myogen.simulator.neuron._cython._poisson_process_generator import (
    _PoissonProcessGenerator__Cython,
)
from myogen.utils.decorators import beartowertype
from myogen.utils.types import Quantity__m, Quantity__m_per_s, Quantity__ms, Quantity__mV


class _Cell:
    """
    Base class for all neural cell models in the spinal cord simulation.

    Provides common functionality for neuron identification, morphology construction,
    and basic cellular properties. All specific neuron types inherit from this class
    and implement their unique biophysical characteristics.

    The class manages global cell identification through a class-level iterator and
    provides template methods for section creation, topology building, geometry
    definition, and biophysics specification that are overridden by subclasses.

    Parameters
    ----------
    class__ID : int
        Unique identifier within the cell's type/class. Used for cell-type specific
        indexing and identification within populations.
    pool__ID : int, optional
        Identifier for the pool/population this cell belongs to, by default None.

    Attributes
    ----------
    global__ID : int
        Globally unique identifier across all cell instances. Automatically assigned
        using a class-level counter to ensure uniqueness across simulation.
    class__ID : int
        Class-specific identifier for this cell instance.
    pool__ID : int or None
        Pool identifier if cell belongs to a specific population.
    synapse__list : list
        List of synaptic mechanisms attached to this cell. Populated by
        create_synapses() method calls.
    axon_length__m : float
        Axonal length in meters. Set by create_axon() method.
    conduction_velocity__m_per_s : float
        Axonal conduction velocity in m/s. Set by create_axon() method.
    axon_delay__ms : float
        Axonal conduction delay in milliseconds. Calculated from length and velocity.
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

    @beartowertype
    def __repr__(self) -> str:
        return f"{self.__class__.__name__} [global ID: {self.global__ID}, class ID: {self.class__ID}, pool ID: {self.pool__ID if self.pool__ID is not None else 'N/A'}]"

    @beartowertype
    def _create_sections(self):
        """
        Create the morphological sections of the cell.

        Template method to be overridden by subclasses. Should create all
        NEURON Section objects that comprise the cell's morphology (soma,
        dendrites, axon, etc.).
        """
        pass

    @beartowertype
    def _build_topology(self):
        """
        Connect sections to define the cell's morphological topology.

        Template method to be overridden by subclasses. Should connect
        all sections created in _create_sections() to form the complete
        cell morphology with proper electrical connectivity.
        """
        pass

    @beartowertype
    def _define_geometry(self):
        """
        Set the 3D geometry and physical properties of each section.

        Template method to be overridden by subclasses. Should define
        length, diameter, axial resistance (Ra), and membrane capacitance (cm)
        for all sections based on experimental morphometric data.
        """
        pass

    @beartowertype
    def _define_biophysics(self):
        """
        Insert ion channels and set biophysical properties.

        Template method to be overridden by subclasses. Should insert
        appropriate ion channel mechanisms and set their conductance values,
        reversal potentials, and kinetic parameters based on physiological
        measurements.
        """
        pass

    @beartowertype
    def create_axon(
        self,
        length__m: Quantity__m = 0 * pq.m,
        conduction_velocity__m_per_s: Quantity__m_per_s = 50 * pq.m / pq.s,
    ):
        """
        Define axonal conduction properties for spike propagation delays.

        Sets the axonal length and conduction velocity, then calculates the
        propagation delay for spike transmission. This is used in network
        simulations to account for realistic axonal delays between cells.

        Parameters
        ----------
        length__m : Quantity__m, default 0 m
            Axonal length in meters, by default 0. When 0, no propagation
            delay is introduced.
        conduction_velocity__m_per_s : Quantity__m_per_s, default 50 m/s
            Axonal conduction velocity in meters per second, by default 50.
            Typical values range from 0.5-2 m/s for unmyelinated axons to
            50-120 m/s for myelinated motor axons.
        """
        self.axon_length__m = length__m
        self.conduction_velocity__m_per_s = conduction_velocity__m_per_s
        self.axon_delay__ms = (self.axon_length__m / self.conduction_velocity__m_per_s).rescale(
            pq.ms
        )

    def create_synapses(
        self,
        synapse_location: h.Section,  # type: ignore
        reversal_potential__mV: Quantity__mV = 0 * pq.mV,
        rise_time_constant__ms: Quantity__ms = 0.2 * pq.ms,
        decay_time_constant__ms: Quantity__ms = 0.3 * pq.ms,
    ) -> h.Exp2Syn:  # type: ignore
        """
        Add an exponentially decaying synapse to the specified location.

        Creates a double-exponential (Exp2Syn) synapse with specified time
        constants and reversal potential. The synapse uses rise time tau1
        and decay time tau2 to model realistic synaptic current kinetics.

        Parameters
        ----------
        synapse_location : h.Section
            The section location where the synapse should be placed.
            Synapse is positioned at the midpoint (0.5) of the section.
        reversal_potential__mV : float, optional
            Synaptic reversal potential in mV, by default 0. For excitatory
            synapses, typically 0 mV; for inhibitory synapses, typically
            -70 to -80 mV.
        rise_time_constant__ms : float, optional
            Synaptic rise time constant in ms, by default 0.2. Must be less
            than tau2 for proper double-exponential kinetics.
        decay_time_constant__ms : float, optional
            Synaptic decay time constant in ms, by default 0.3. Must be
            greater than tau1 for realistic synaptic waveforms.

        Returns
        -------
        h.Exp2Syn
            The created NEURON synapse object, which can be used for
            connecting presynaptic spike sources.

        Notes
        -----
        Default parameters (tau1=0.2, tau2=0.3, weight=0.6) are based on
        experimental data from Sanches (2018). For NERLab model compatibility,
        the reversal potential is automatically set to 70 mV when the cell
        has the "model" attribute set to "NERLab".

        References
        ----------
        - Sanches et al. (2018): Synaptic parameter measurements
        - Segev et al. (1990): Alpha function synaptic modeling
        """
        assert decay_time_constant__ms > rise_time_constant__ms, (
            "decay_time_constant__ms must be greater than tau1 for proper kinetics"
        )
        syn = h.Exp2Syn(synapse_location(0.5))
        syn.tau1 = rise_time_constant__ms
        syn.tau2 = decay_time_constant__ms
        if hasattr(self, "model") and self.model == "NERLab":
            syn.e = 70
        else:
            syn.e = reversal_potential__mV
        self.synapse__list.append(syn)  # synapse__list is defined in Cell
        return syn


# INTERNEURONS
@beartowertype
class INgII(_Cell):
    """
    Group II interneuron (Ib inhibitory interneuron).

    Models spinal interneurons that receive input from Group II muscle spindle
    afferents and provide inhibitory connections to motor neurons. These cells
    are part of the spinal reflex circuitry and contribute to muscle tone regulation
    and reciprocal inhibition between antagonist muscles.

    The cell model is based on experimental data from Bui et al. (2003) for
    Ia inhibitory interneurons, with single-compartment soma containing
    physiologically realistic ion channels including sodium, potassium, h-current,
    and calcium-dependent afterhyperpolarization channels.

    Parameters
    ----------
    class__ID : int, optional
        Unique identifier within INgII population. If None, automatically
        assigned from internal counter, by default None.
    pool__ID : int, optional
        Pool identifier for grouping related interneurons, by default None.

    Attributes
    ----------
    soma : h.Section
        NEURON soma section with active ion channels and synaptic mechanisms.

    Notes
    -----
    The geometry and biophysics are based on cat spinal cord measurements:
    - Membrane area: ~81,390 um² (Bui et al., 2003)
    - Input resistance: ~20 MΩ (from g_pas = 5e-5 S/cm²)
    - Resting potential: -71 mV
    - Active channels: Na+, K+, h-current, Ca²⁺-dependent K+

    References
    ----------
    Bui et al. (2003). Morphological and electrophysiological characterization
    of Ia inhibitory interneurons in cat spinal cord.
    """

    _ids2 = itertools.count(0)

    def __init__(self, class__ID: Optional[int] = None, pool__ID: int | None = None):
        super().__init__(class__ID if class__ID is not None else next(self._ids2), pool__ID)
        self.create_synapses(self.soma)

    def _create_sections(self):
        """Create the morphological sections - single soma compartment."""
        self.soma = h.Section(name="soma", cell=self)

    def _define_geometry(self):
        """
        Set soma geometry based on experimental morphometric data.

        Implements measurements from Bui et al. (2003) for Ia inhibitory
        interneurons, calculating diameter from measured membrane surface area
        assuming spherical geometry.
        """
        # Data from Bui et al., 2003 Ia Inhibitory interneurons
        Amu = 81390 + 3113  # Mean membrane area (um²)
        Aci = 1.96 * (891.5 + 46.141) / np.sqrt(8)  # Area confidence interval
        i = 0  # Use lower bound of area distribution
        A = [Amu - Aci, Amu + Aci]
        D = np.sqrt(A[i] / np.pi)  # Diameter from area, assuming sphere

        self.soma.L = D  # Length = diameter for sphere (um)
        self.soma.diam = D  # Diameter (um)
        self.soma.Ra = 70  # Axial resistance (Ω⋅cm)
        self.soma.cm = 1  # Membrane capacitance (uF/cm²)
        self.soma.nseg = 1  # Single compartment

    def _define_biophysics(self):
        """
        Insert ion channels and set biophysical parameters.

        Implements the complete set of voltage-gated channels found in
        spinal interneurons, with conductance values tuned to reproduce
        experimental firing patterns and membrane properties.
        """
        # Passive membrane properties
        self.soma.insert("pas")
        self.soma.g_pas = 5e-5  # Leak conductance (S/cm²) → Rin ~20 MΩ
        self.soma.e_pas = -71.0  # Leak reversal potential (mV)

        # Active ion channels
        for mech in ["na3rp", "kdrRL", "gh", "mAHP"]:
            self.soma.insert(mech)

        # H-current (hyperpolarization-activated cation current)
        self.soma.ghbar_gh = 2.5e-5  # Max conductance affects Rin, τm, AHP duration
        self.soma.half_gh = -70.0  # Half-activation voltage affects AHP duration
        self.soma.ek = -80.0  # Potassium reversal potential (mV)

        # Sodium channels (action potential generation)
        self.soma.gbar_na3rp = 0.003  # Max conductance: AP amplitude, current threshold
        self.soma.qinf_na3rp = 8.0  # Inactivation: AP amplitude (inverse relationship)
        self.soma.sh_na3rp = 1.0  # Voltage shift: rheobase (direct relationship)
        self.soma.thinf_na3rp = -50  # Inactivation half-point (mV)

        # Potassium delayed rectifier (action potential repolarization)
        self.soma.gMax_kdrRL = 0.015  # Max conductance: rheobase, AHP duration/magnitude

        # Calcium-dependent potassium (afterhyperpolarization)
        self.soma.gcamax_mAHP = 3e-6  # Ca²⁺ conductance: AHP magnitude (direct)
        self.soma.gkcamax_mAHP = 0.0005  # K(Ca) conductance: AHP magnitude (direct)
        self.soma.tau_mAHP = 70.0  # Ca²⁺ removal time constant: AHP duration (direct)


@beartowertype
class INgIb(INgII):
    """
    Golgi tendon organ (Ib) interneuron.

    Spinal interneuron that receives input from Golgi tendon organs (Ib afferents)
    and provides inhibitory feedback to motor neurons. Part of the force feedback
    control system that prevents excessive muscle tension and contributes to
    smooth force gradation during voluntary contractions.

    Inherits all biophysical properties from INgII but with distinct class
    identification for network connectivity and functional role differentiation.

    Parameters
    ----------
    pool__ID : int, optional
        Pool identifier for grouping related Ib interneurons, by default None.

    Notes
    -----
    Uses identical membrane properties as INgII since detailed experimental
    characterization of Ib interneurons is limited. The functional distinction
    is primarily in their afferent input sources and efferent connectivity
    patterns within spinal reflex circuits.
    """

    _ids2 = itertools.count(0)

    def __init__(self, pool__ID: int | None = None):
        super().__init__(next(self._ids2), pool__ID)


# AFFERENTS AND DESCENDING TRACTS
@beartowertype
class DD(_Cell, _PoissonProcessGenerator__Cython):
    """
    Descending drive neuron for cortical/supraspinal input simulation.

    Models descending inputs from higher brain centers (motor cortex, brainstem)
    as Poisson spike trains with activity-dependent firing rates. Uses a dummy
    NEURON cell for interface compatibility while generating realistic spike
    patterns through an Cython-optimized Poisson process generator.

    The firing rate is proportional to the input drive signal, allowing
    simulation of voluntary motor commands, reflex modulation, and other
    descending influences on spinal motor circuits.

    Parameters
    ----------
    N : int
        Maximum firing rate in Hz when input drive is at maximum. Determines
        the scaling factor for converting drive signals to spike rates.
    dt : float
        Simulation time step in milliseconds. Must match the integration
        time step used in the main simulation loop.
    pool__ID : int, optional
        Pool identifier for grouping related descending neurons, by default None.

    Attributes
    ----------
    ns : h.DUMMY
        NEURON dummy cell for interface compatibility with network connections.
        Does not contain actual membrane mechanisms.

    Notes
    -----
    The Poisson spike generation uses a unique random seed based on the cell's
    global and class IDs to ensure reproducible but independent spike trains
    across different DD instances while maintaining global reproducibility.
    """

    _ids2 = itertools.count(0)

    def __init__(self, N, dt, pool__ID: int | None = None):
        self.ns = h.DUMMY()  # Dummy cell
        _Cell.__init__(self, next(self._ids2), pool__ID)
        _PoissonProcessGenerator__Cython.__init__(
            self, SEED + (self.class__ID + 1) * (self.global__ID + 1), N, dt
        )

    def __repr__(self) -> str:
        return _Cell.__repr__(self)

    def integrate(self, y):
        """
        Generate Poisson spikes based on input drive level.

        Parameters
        ----------
        y : float
            Drive signal level (0-1). Values > 0 generate spikes proportional
            to the drive strength; values ≤ 0 produce no spikes.

        Returns
        -------
        int
            Number of spikes generated in this time step (0 or 1 for Poisson).
        """
        return self.compute(y) if y > 0 else 0


@beartowertype
class DD_Gamma(_Cell, _GammaProcessGenerator__Cython):
    """
    Descending drive neuron using Gamma process for more regular spike patterns.

    Models descending inputs from higher brain centers (motor cortex, brainstem)
    as Gamma-distributed spike trains with activity-dependent firing rates. Unlike
    the Poisson-based DD class, this produces more regular spike patterns typical
    of cortical neurons, with regularity controlled by the shape parameter.

    Uses a dummy NEURON cell for interface compatibility while generating realistic
    spike patterns through a Cython-optimized Gamma process generator.

    The firing rate is controlled directly by the input drive signal (in Hz),
    allowing simulation of voluntary motor commands, reflex modulation, and other
    descending influences on spinal motor circuits.

    Parameters
    ----------
    timestep__ms : Quantity__ms
        Simulation time step in milliseconds. Must match the integration
        time step used in the main simulation loop.
    shape : float, optional
        Shape parameter (k) controlling spike regularity, by default 3.0.
        - shape=1: Poisson-like (irregular) firing
        - shape=2-5: Typical cortical neuron regularity
        - Higher values: More regular, clock-like firing
        The coefficient of variation (CV) of ISIs is 1/sqrt(shape).
    pool__ID : int, optional
        Pool identifier for grouping related descending neurons, by default None.

    Attributes
    ----------
    ns : h.DUMMY
        NEURON dummy cell for interface compatibility with network connections.
        Does not contain actual membrane mechanisms.

    Notes
    -----
    The Gamma spike generation uses a unique random seed based on the cell's
    global and class IDs to ensure reproducible but independent spike trains
    across different DD_Gamma instances while maintaining global reproducibility.

    For shape=1, this is equivalent to Poisson process (same as DD class).
    For shape>1, produces more regular firing with CV = 1/sqrt(shape):
    - shape=4 gives CV=0.5 (fairly regular)
    - shape=9 gives CV=0.33 (very regular)
    """

    _ids2 = itertools.count(0)

    def __init__(
        self,
        timestep__ms: Quantity__ms,
        shape: float = 3.0,
        pool__ID: int | None = None,
    ):
        self.ns = h.DUMMY()  # Dummy cell
        _Cell.__init__(self, next(self._ids2), pool__ID)
        _GammaProcessGenerator__Cython.__init__(
            self,
            SEED + (self.class__ID + 1) * (self.global__ID + 1),
            shape,
            timestep__ms.magnitude,
        )

    def __repr__(self) -> str:
        return _Cell.__repr__(self)

    def integrate(self, y):
        """
        Generate Gamma-process spikes based on input drive level.

        Parameters
        ----------
        y : float
            Drive signal in pps (firing rate). Values > 0 generate spikes with
            the specified rate and regularity; values ≤ 0 produce no spikes.

        Returns
        -------
        int
            Number of spikes generated in this time step (0 or 1 for Gamma process).
        """
        return self.compute(y) if y > 0 else 0


@beartowertype
class AffIa(_Cell, _PoissonProcessGenerator__Cython):
    """
    Primary muscle spindle afferent (Ia afferent) with recruitment threshold.

    Models muscle spindle primary endings that respond to muscle length changes
    and stretch velocity. These afferents provide monosynaptic excitatory input
    to homonymous motor neurons (stretch reflex) and disynaptic inhibition to
    antagonist motor neurons via Ia inhibitory interneurons.

    The firing rate follows a recruitment threshold model where the afferent
    only fires when the input signal exceeds its recruitment threshold (RT).
    Individual variability in firing patterns is introduced through a random
    component (IFR) drawn from a normal distribution.

    Parameters
    ----------
    RT : float
        Recruitment threshold - minimum input level required for activation.
        Represents the stretch sensitivity of the particular spindle ending.
    N : int
        Maximum firing rate in Hz when fully activated. Determines the
        gain of the length-to-frequency transduction.
    timestep__ms : Quantity__ms
        Simulation time step as a Quantity with units of milliseconds.
    initN : int, optional
        Initial spike count for rate computation, by default 0.
    class__ID : int, optional
        Unique identifier within AffIa population. Auto-assigned if None.
    pool__ID : int, optional
        Pool identifier for muscle-specific grouping, by default None.

    Attributes
    ----------
    RT : float
        Recruitment threshold for activation.
    IFR : float
        Individual firing rate variability factor drawn from normal distribution
        (mean=5, std=2.5). Adds realistic inter-afferent differences.
    ns : h.DUMMY
        NEURON dummy cell for network interface compatibility.

    Notes
    -----
    The activation level is computed as: act = input - RT + IFR
    This allows for individual variability while maintaining the threshold
    behavior characteristic of muscle spindle afferents.
    """

    _ids2 = itertools.count(0)

    def __init__(
        self,
        RT,
        N,
        timestep__ms: Quantity__ms,
        initN=0,
        class__ID: Optional[int] = None,
        pool__ID: int | None = None,
    ):
        self.ns = h.DUMMY()  # Dummy cell

        self.RT = RT  # Recruitment Threshold
        self.IFR = RANDOM_GENERATOR.normal(5, 2.5)  # Individual variability

        _Cell.__init__(self, class__ID if class__ID is not None else next(self._ids2), pool__ID)
        _PoissonProcessGenerator__Cython.__init__(
            self,
            seed=SEED + (self.class__ID + 1) * (self.global__ID + 1),
            N=N,
            dt=timestep__ms.magnitude,
            Ninit=initN,
        )

    def integrate(self, y):
        """
        Generate spikes based on input level relative to recruitment threshold.

        Parameters
        ----------
        y : float
            Input signal level (typically muscle length/stretch percentage).

        Returns
        -------
        int
            Number of spikes generated (0 or 1) based on Poisson process
            with rate determined by activation level above threshold.
        """
        act = y - self.RT + self.IFR
        return self.compute(act) if act > 0 else 0


@beartowertype
class AffII(AffIa):
    """
    Secondary muscle spindle afferent (Group II afferent).

    Models muscle spindle secondary endings that primarily respond to muscle
    length (tonic stretch) with less sensitivity to velocity compared to
    Ia afferents. These afferents contribute to postural reflexes and provide
    input to Group II interneurons in spinal reflex circuits.

    Inherits all functional properties from AffIa but with distinct class
    identification for network connectivity. Typically have higher recruitment
    thresholds and lower maximum firing rates compared to Ia afferents.

    Parameters
    ----------
    RT : float
        Recruitment threshold for activation.
    N : int
        Maximum firing rate in Hz.
    pool__ID : int, optional
        Pool identifier for muscle-specific grouping.
    *args, **kwargs
        Additional arguments passed to AffIa constructor.

    Notes
    -----
    Physiologically, Group II afferents have:
    - Higher recruitment thresholds than Ia afferents
    - Lower maximum firing rates (~50-80 Hz vs 100+ Hz)
    - Greater sensitivity to length vs velocity
    - Different spinal connectivity patterns
    """

    _ids2 = itertools.count(0)

    def __init__(self, RT, N, pool__ID: int | None = None, *args, **kwargs):
        super().__init__(RT, N, class__ID=next(self._ids2), pool__ID=pool__ID, *args, **kwargs)


@beartowertype
class AffIb(AffIa):
    """
    Golgi tendon organ afferent (Ib afferent).

    Models Golgi tendon organ receptors that respond to muscle force/tension
    rather than length. These afferents provide force feedback through
    inhibitory connections via Ib interneurons, contributing to force
    regulation and preventing excessive muscle tension.

    Uses the same computational model as muscle spindle afferents but with
    distinct class identification. In physiological applications, the input
    signal represents muscle force rather than length.

    Parameters
    ----------
    RT : float
        Force recruitment threshold for activation.
    N : int
        Maximum firing rate in Hz at full force.
    pool__ID : int, optional
        Pool identifier for muscle-specific grouping.
    *args, **kwargs
        Additional arguments passed to AffIa constructor.

    Notes
    -----
    Golgi tendon organs have:
    - Force sensitivity proportional to muscle tension
    - Higher thresholds than spindle afferents
    - Inhibitory effects on homonymous motor neurons
    - Force-dependent recruitment characteristics
    """

    _ids2 = itertools.count(0)

    def __init__(self, RT, N, pool__ID: int | None = None, *args, **kwargs):
        super().__init__(RT, N, class__ID=next(self._ids2), pool__ID=pool__ID, *args, **kwargs)


# MOTORNEURON
@beartowertype
class AlphaMN(_Cell):
    """
    Alpha motor neuron with detailed compartmental structure and biophysics.

    Parameters
    ----------
    segments__count : int, optional
        Number of segments per section for spatial discretization, by default 1.
        Higher values improve accuracy but increase computation time.
    mode : str, optional
        Biophysical model mode: "active" includes voltage-gated channels,
        "passive" uses only leak conductance, by default "active".
    dendrites__count : int, optional
        Number of dendrites (1 or 4), by default 4. Affects total dendritic
        surface area and synaptic integration properties.
    model : str, optional
        Ion channel parameter set: "Powers2017" (default) or "NERLab".
        Different experimental calibrations and channel implementations.
    class__ID : int, optional
        Override for class ID assignment. Auto-assigned if None.
    pool__ID : int, optional
        Motor neuron pool identifier for muscle-specific grouping.
    *args, **kwargs
        Additional arguments passed to _Cell constructor.

    Attributes
    ----------
    soma : h.Section
        NEURON soma section with voltage-gated ion channels.
    dend : list[h.Section]
        List of dendritic sections with calcium channels and synapses.
    class__ID : int
        Class specific identifier.
    dendrites__count : int
        Number of dendritic compartments.
    segments__count : int
        Spatial discretization level.
    mode : str
        Active or passive membrane model.
    model : str
        Ion channel parameter set identifier.

    References
    ----------
    - Powers & Binder (2001): Motor neuron biophysical characterization
    - Powers et al. (2017): Updated ion channel models and parameters
    """

    _ids2 = itertools.count(0)

    def __init__(
        self,
        segments__count: int = 1,
        mode: Literal["active", "passive"] = "active",
        dendrites__count: int = 4,
        model: Literal["NERLab", "Powers2017"] = "NERLab",
        class__ID: int | None = None,
        pool__ID: int | None = None,
        *args,
        **kwargs,
    ):
        if self.__class__.__name__ == "AlphaMN" and class__ID is None:
            self._class__ID = next(self._ids2)
        if class__ID is not None:
            self._class__ID = class__ID
        self.dendrites__count = dendrites__count
        self.segments__count = segments__count
        self.mode = mode
        self.model = model
        super().__init__(class__ID=self._class__ID, pool__ID=pool__ID, *args, **kwargs)

        for d in self.dend:
            self.create_synapses(d)

    def _create_sections(self):
        """
        Create soma and dendritic sections for the motor neuron.

        Creates a single soma section and the specified number of dendritic
        sections based on the dendrites__count parameter. All sections are automatically
        registered with the NEURON simulation environment.
        """
        self.soma = h.Section(name="soma", cell=self)
        self.dend = [h.Section(name="dend", cell=self) for _ in range(self.dendrites__count)]

    def _define_geometry(self):
        """
        Set morphological dimensions based on experimental motor neuron data.

        Implements realistic dimensions from cat lumbar motor neurons with
        anatomically accurate surface areas, diameters, and membrane properties.
        Dendritic geometry adapts based on the number of dendrites to maintain
        physiological total dendritic surface area.
        """
        # Soma geometry (cat lumbar motor neurons)
        self.soma.L = 2952  # Length (um) - effective sphere diameter
        self.soma.diam = 22  # Diameter (um)
        self.soma.Ra = 0.001  # Axial resistance (Ω⋅cm) - very low for soma
        self.soma.cm = 1.35546  # Membrane capacitance (uF/cm²)
        self.soma.nseg = self.segments__count

        # Dendritic geometry - dimensions scale with dendrite count
        for d in self.dend:
            if self.dendrites__count == 4:
                d.L = 1794.13  # Length (um) per dendrite
                d.diam = 8.73071  # Diameter (um)
            if self.dendrites__count == 1:
                d.L = 2848  # Single large dendrite
                d.diam = 22  # Larger diameter
            d.Ra = 51.038  # Dendritic axial resistance (Ω⋅cm)
            d.cm = 0.867781  # Dendritic membrane capacitance (uF/cm²)
            d.nseg = self.segments__count

    def _build_topology(self):
        """
        Connect dendrites to soma with physiological topology.

        Connects dendrites alternately to opposite poles of the soma
        (positions 0 and 1) to simulate the bipolar dendritic tree
        characteristic of spinal motor neurons.
        """
        for i, d in enumerate(self.dend):
            d.connect(self.soma(i % 2))

    def insert_Gfluctdv(self):
        """
        Insert fluctuating conductance mechanism for synaptic background noise.

        Adds the Gfluctdv mechanism to all dendrites to simulate ongoing
        synaptic background activity with stochastic excitatory and inhibitory
        conductance fluctuations. This provides realistic membrane potential
        variability and affects firing threshold and patterns.

        Notes
        -----
        Parameters are set for:
        - g_e0/g_i0: Mean excitatory/inhibitory conductance (1e-5 S/cm²)
        - std_e/std_i: Conductance fluctuation amplitude (1.2e-5 S/cm²)
        - tau_e/tau_i: Correlation time constants (20 ms)

        These values produce realistic subthreshold membrane fluctuations
        (~2-3 mV RMS) typical of in vivo motor neuron recordings.
        """
        for d in self.dend:
            d.insert("Gfluctdv")
            # Excitatory fluctuations
            d.g_e0_Gfluctdv = 1e-5  # Mean excitatory conductance (S/cm²)
            d.std_e_Gfluctdv = 1.2e-5  # Conductance noise amplitude (S/cm²)
            d.tau_e_Gfluctdv = 20  # Correlation time constant (ms)
            # Inhibitory fluctuations
            d.g_i0_Gfluctdv = 1e-5  # Mean inhibitory conductance (S/cm²)
            d.std_i_Gfluctdv = 1.2e-5  # Conductance noise amplitude (S/cm²)
            d.tau_i_Gfluctdv = 20  # Correlation time constant (ms)

    def _define_biophysics(self, gamma=1):
        """
        Insert ion channels and set biophysical parameters for motor neuron.

        Implements detailed voltage-gated ion channel models based on experimental
        characterization. Supports both Powers2017 and NERLab parameter sets with
        different channel implementations and calibrations.

        The Powers2017 model includes the complete complement of motor neuron
        ion channels with parameters tuned to reproduce experimentally measured
        firing patterns, input resistance, membrane time constants, and
        plateau potential behaviors.

        Parameters
        ----------
        gamma : float, optional
            Scaling factor for dendritic calcium channel density, by default 1.
            Values > 1 increase PIC amplitude; values < 1 reduce bistability.
            Useful for simulating motor neuron size-dependent PIC properties.

        Notes
        -----
        **Powers2017 Ion Channel Complement:**

        *Soma Channels:*
        - na3rp: Fast sodium channels for action potential generation
        - naps: Persistent sodium channels for amplification and bistability
        - kdrRL: Delayed rectifier potassium for action potential repolarization
        - mAHP: Calcium-dependent potassium for spike afterhyperpolarization
        - gh: Hyperpolarization-activated cation current for membrane properties

        *Dendritic Channels:*
        - L_Ca_inact: L-type calcium channels for persistent inward currents (PICs)
        - gh: Hyperpolarization-activated current for dendritic integration

        The channel densities and kinetics are calibrated to produce:
        - Input resistance: 1-10 MΩ (size-dependent)
        - Membrane time constant: 5-15 ms
        - Action potential amplitude: 80-100 mV
        - AHP duration: 50-200 ms
        - PIC threshold: 5-15 mV below spike threshold
        """
        if self.model == "NERLab":
            if self.mode == "active":
                self.soma.insert("napp")
                self.soma.insert("Constant")
                self.dend[0].insert("caL")
                self.dend[0].insert("Constant")
            else:
                print("NERLab passive model is not defined yet.")
                print("try active mode.")

        if self.model == "Powers2017":
            # Channel types and their roles:
            # PIC Channels:
            #   - L_CA_inact: L-type Calcium channels
            #   - nas: Na slow inactivation Channel
            # AP Channels:
            #   - na3rp: Na current
            #   - kdrRL: Potassium Delayed Rectifier Channel
            # AHP Channel:
            #   - mAHP: Calcium-dependent potassium Channel
            # Passive Channels:
            #   - gh: Hodgkin-Huxley Potassium h channel
            #   - pas: passive mechanisms

            # Soma passive properties
            self.soma.insert("pas")
            self.soma.g_pas = 8.109e-05
            self.soma.e_pas = -71.0

            # Active soma mechanisms
            if self.mode == "active":
                for mech in ["na3rp", "naps", "kdrRL", "mAHP", "gh"]:
                    self.soma.insert(mech)

                # Sodium channels (na3rp)
                self.soma.gbar_na3rp = 0.01
                self.soma.sh_na3rp = 1.0
                self.soma.ar_na3rp = 1.0
                self.soma.qinf_na3rp = 8.0
                self.soma.thinf_na3rp = -50

                # Persistent sodium channels (naps)
                self.soma.gbar_naps = 2.6e-05
                self.soma.sh_naps = 5.0
                self.soma.ar_naps = 1.0

                # Potassium delayed rectifier (kdrRL)
                self.soma.gMax_kdrRL = 0.015

                # Calcium-dependent potassium (mAHP)
                self.soma.gcamax_mAHP = 6.4e-06
                self.soma.gkcamax_mAHP = 0.00045
                self.soma.taur_mAHP = 90.0
                self.soma.ek = -80.0

                # H-current (gh)
                self.soma.ghbar_gh = 3e-05
                self.soma.half_gh = -77.0

            # Global channel parameters
            h.vslope_naps = 5  # activation slope for persistent sodium channels (mV)
            h.asvh_naps = -90  # slow inactivation voltage half-point for persistent sodium (mV)
            h.bsvh_naps = -22  # slow inactivation voltage parameter for persistent sodium (mV)
            h.mvhalfca_mAHP = (
                -22
            )  # calcium activation voltage half-point for Ca-dependent K channels (mV)
            h.mtauca_mAHP = 2  # calcium time constant for Ca-dependent K channels (ms)
            h.tau_m_L_Ca_inact = 40  # activation time constant for L-type calcium channels (ms)
            h.tau_h_L_Ca_inact = (
                2500.0  # inactivation time constant for L-type calcium channels (ms)
            )
            h.kappa_h_L_Ca_inact = 5.0  # inactivation slope factor for L-type calcium channels
            h.mVh_kdrRL = -21.0  # half-activation voltage for K delayed rectifier channels (mV)
            h.tmin_kdrRL = 0.8  # minimum time constant for K delayed rectifier channels (ms)
            h.taumax_kdrRL = 20.0  # maximum time constant for K delayed rectifier channels (ms)
            h.htau_gh = 30.0  # time constant for h-current channels (ms)

            # Dendritic mechanisms
            if self.mode == "pas":
                dendMechs = ["pas"]
            elif self.mode == "active":
                dendMechs = ["pas", "L_Ca_inact", "gh"]
                if self.dendrites__count == 1:
                    gca = [1.05e-4]
                else:
                    gca = [8.5e-05, 9.5e-5, 1e-4, 1.15e-4]

            # Apply dendritic properties
            for i, d in enumerate(self.dend):
                for mech in dendMechs:
                    d.insert(mech)
                d.g_pas = 7.93445e-05
                d.e_pas = -71.0

                if self.mode == "active":
                    d.gcabar_L_Ca_inact = gca[i] * gamma
                    d.ghbar_gh = 3e-05
                    d.half_gh = -77.0
                    d.theta_m_L_Ca_inact = -42.0
                    d.theta_h_L_Ca_inact = 10.0
