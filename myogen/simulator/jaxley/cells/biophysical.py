"""
Biophysical Motor Neuron Cell Builder for Jaxley.

This module provides functions to construct multi-compartment motor neurons
with realistic morphology and biophysics using Jaxley's cell infrastructure.

The cells include:
- Soma with action potential generating channels
- Dendrites with persistent inward currents (PICs)
- Proper channel distributions based on Powers2017 or NERLab models
"""

from typing import Dict, List, Literal, Optional, Tuple
import numpy as np
import jax.numpy as jnp
import jaxley as jx
from jaxley.channels import Leak

# Custom MyoGen channels (motor neuron specific)
from myogen.simulator.jaxley.channels import (
    Na3rp,
    Naps,
    KdrRL,
    MAHP,
    Gh,
    LCaInact,
    LeakChannel,
)

# Try importing jaxley-mech channels (Hodgkin-Huxley based)
try:
    from jaxley_mech.channels.hodgkin52 import (
        Na as JaxleyMechNa,
        K as JaxleyMechK,
        Leak as JaxleyMechLeak,
    )
    JAXLEY_MECH_AVAILABLE = True
except ImportError:
    JAXLEY_MECH_AVAILABLE = False
    JaxleyMechNa = None
    JaxleyMechK = None
    JaxleyMechLeak = None


# =============================================================================
# MORPHOLOGY PARAMETERS
# =============================================================================

# Powers & Binder (2001) motor neuron morphology
POWERS2017_MORPHOLOGY = {
    "soma": {
        "length": 25.0,      # um
        "diameter": 25.0,    # um (approximate sphere)
        "nseg": 1,
    },
    "dendrite": {
        "length": 5500.0,    # um - equivalent dendrite length
        "diameter": 6.0,     # um - average diameter
        "nseg": 10,          # Number of segments
        "n_dendrites": 4,    # Number of equivalent dendrites
    },
}

# NERLab simplified morphology
NERLAB_MORPHOLOGY = {
    "soma": {
        "length": 22.0,
        "diameter": 22.0,
        "nseg": 1,
    },
    "dendrite": {
        "length": 4000.0,
        "diameter": 5.0,
        "nseg": 5,
        "n_dendrites": 1,
    },
}

# =============================================================================
# CHANNEL CONDUCTANCE PARAMETERS
# =============================================================================

POWERS2017_CHANNELS = {
    "soma": {
        "na3rp": {"gbar": 0.05, "sh": 8.0, "ar": 1.0},
        "naps": {"gbar": 0.001, "sh": 0.0, "ar": 1.0},
        "kdrRL": {"gbar": 0.3, "mVh": -25.0, "mslp": 20.0},
        "mAHP": {"gkcamax": 0.03, "gcamax": 3e-5, "tau_ca": 20.0},
        "gh": {"gbar": 2.5e-5, "half": -70.0, "slp": 8.0},
        "leak": {"g_leak": 5e-5, "e_leak": -70.0},
    },
    "dendrite": {
        "L_Ca_inact": {"gbar": 6.7e-5, "theta_m": -30.0, "tau_h": 1500.0},
        "gh": {"gbar": 0.0001, "half": -70.0, "slp": 8.0},
        "leak": {"g_leak": 5e-5, "e_leak": -70.0},
    },
    "axial_resistivity": 70.0,     # Ohm-cm
    "membrane_capacitance": 1.0,   # uF/cm²
    "e_na": 60.0,                  # mV
    "e_k": -80.0,                  # mV
    "e_ca": 80.0,                  # mV
}

NERLAB_CHANNELS = {
    "soma": {
        "na3rp": {"gbar": 0.03, "sh": 1.0, "ar": 1.0},
        "naps": {"gbar": 0.0002, "sh": 0.0, "ar": 1.0},
        "kdrRL": {"gbar": 0.015, "mVh": -25.0, "mslp": 20.0},
        "mAHP": {"gkcamax": 0.0005, "gcamax": 5e-5, "tau_ca": 70.0},
        "gh": {"gbar": 2.5e-5, "half": -70.0, "slp": 8.0},
        "leak": {"g_leak": 5e-5, "e_leak": -71.0},
    },
    "dendrite": {
        "L_Ca_inact": {"gbar": 1e-5, "theta_m": -30.0, "tau_h": 1500.0},
        "gh": {"gbar": 2.5e-5, "half": -70.0, "slp": 8.0},
        "leak": {"g_leak": 5e-5, "e_leak": -71.0},
    },
    "axial_resistivity": 70.0,
    "membrane_capacitance": 1.35546,
    "e_na": 60.0,
    "e_k": -80.0,
    "e_ca": 120.0,
}


# =============================================================================
# BIOPHYSICAL MOTOR NEURON CLASS
# =============================================================================

# Channel conductance parameters for jaxley-mech Hodgkin-Huxley channels
# Standard Hodgkin-Huxley 1952 squid giant axon parameters from jaxley-mech documentation
JAXLEY_MECH_CHANNELS = {
    "soma": {
        "na": {"gNa": 120e-3, "eNa": 50.0},      # S/cm² - fast sodium
        "k": {"gK": 36e-3, "eK": -77.0},          # S/cm² - delayed rectifier
        "leak": {"gLeak": 0.3e-3, "eLeak": -54.3},  # S/cm² - passive leak
    },
    "dendrite": {
        "leak": {"gLeak": 0.1e-3, "eLeak": -65.0},  # Passive dendrite with leak only
    },
    "axial_resistivity": 70.0,     # Ohm-cm
    "membrane_capacitance": 1.0,   # uF/cm² - standard membrane capacitance
}


class BiophysicalMotorNeuron:
    """
    Multi-compartment motor neuron with realistic biophysics.
    
    Constructs a Jaxley cell with:
    - Soma containing Na+, K+, Ca2+-dependent K+ (AHP), and H-current
    - Dendrites with L-type Ca2+ for persistent inward currents
    
    Parameters
    ----------
    model : str
        "Powers2017" or "NERLab" - determines parameters.
    n_dendrites : int, optional
        Number of dendrites (default from model).
    segments_per_dendrite : int, optional
        Compartments per dendrite (default from model).
    use_multicompartment : bool
        If True, create actual multi-compartment cell with dendrites.
        If False (default), create single-compartment soma only.
    use_jaxley_mech : bool
        If True, use jaxley-mech Hodgkin-Huxley channels (Na, K, Leak)
        instead of custom motor neuron channels. Default is False.
    """
    
    def __init__(
        self,
        model: Literal["Powers2017", "NERLab"] = "Powers2017",
        n_dendrites: Optional[int] = None,
        segments_per_dendrite: Optional[int] = None,
        use_multicompartment: bool = False,
        use_jaxley_mech: bool = False,
    ):
        self.model = model
        self.use_multicompartment = use_multicompartment
        self.use_jaxley_mech = use_jaxley_mech
        
        # Validate jaxley-mech availability
        if use_jaxley_mech and not JAXLEY_MECH_AVAILABLE:
            raise ImportError(
                "jaxley-mech is not installed. Install it with: pip install jaxley-mech"
            )
        
        # Get parameters based on model
        if model == "Powers2017":
            self.morphology = POWERS2017_MORPHOLOGY.copy()
            self.channel_params = POWERS2017_CHANNELS.copy()
        else:
            self.morphology = NERLAB_MORPHOLOGY.copy()
            self.channel_params = NERLAB_CHANNELS.copy()
        
        # Override if specified
        if n_dendrites is not None:
            self.morphology["dendrite"]["n_dendrites"] = n_dendrites
        if segments_per_dendrite is not None:
            self.morphology["dendrite"]["nseg"] = segments_per_dendrite
        
        # Build the cell
        self.cell = None
        self.soma = None
        self.dendrites = []
        self.soma_branch_idx = 0
        self.dendrite_branch_indices = []
        
        self._build_cell()
        self._define_morphology()
        self._insert_channels()
    
    def _build_cell(self):
        """Create the Jaxley cell structure."""
        if self.use_multicompartment:
            self._build_multicompartment_cell()
        else:
            self._build_single_compartment_cell()
    
    def _build_single_compartment_cell(self):
        """Create a single-compartment (soma only) cell."""
        # Create a single compartment
        comp = jx.Compartment()
        
        # Create a branch with 1 compartment for soma
        soma_branch = jx.Branch(comp, ncomp=1)
        
        # Create cell from branch
        self.cell = jx.Cell(soma_branch, parents=[-1])
        self.soma = self.cell.branch(0)
        self.soma_branch_idx = 0
    
    def _build_multicompartment_cell(self):
        """Create a multi-compartment cell with soma and dendrites.
        
        Uses proper Jaxley topology: soma is root (-1), all dendrites connect to soma (0).
        Each dendrite branch has multiple compartments for spatial discretization.
        """
        soma_params = self.morphology["soma"]
        dend_params = self.morphology["dendrite"]
        
        n_dendrites = dend_params["n_dendrites"]
        nseg_dend = dend_params["nseg"]
        
        # Create compartment template
        comp = jx.Compartment()
        
        # Create branches: soma (1 comp) + n_dendrites (nseg_dend comps each)
        # Build list of branches for Cell constructor
        branches = []
        
        # Branch 0: Soma (1 compartment)
        soma_branch = jx.Branch(comp, ncomp=1)
        branches.append(soma_branch)
        
        # Branches 1 to n_dendrites: Dendrites
        for _ in range(n_dendrites):
            dend_branch = jx.Branch(comp, ncomp=nseg_dend)
            branches.append(dend_branch)
        
        # Build parent structure:
        # - Branch 0 (soma) is root: parent = -1
        # - Branches 1..n (dendrites) connect to soma: parent = 0
        parents = np.array([-1] + [0] * n_dendrites)
        
        # Create cell with proper topology
        self.cell = jx.Cell(branches, parents=parents)
        
        # Store references
        self.soma = self.cell.branch(0)
        self.soma_branch_idx = 0
        self.dendrite_branch_indices = list(range(1, n_dendrites + 1))
        self.dendrites = [self.cell.branch(idx) for idx in self.dendrite_branch_indices]
    
    def _define_morphology(self):
        """Set up the cell morphology (geometry)."""
        soma_params = self.morphology["soma"]

        # Set global membrane properties
        # Use JAXLEY_MECH_CHANNELS params when use_jaxley_mech=True for proper HH tuning
        if self.use_jaxley_mech:
            self.cell.set("axial_resistivity", JAXLEY_MECH_CHANNELS["axial_resistivity"])
            self.cell.set("capacitance", JAXLEY_MECH_CHANNELS["membrane_capacitance"])
        else:
            self.cell.set("axial_resistivity", self.channel_params["axial_resistivity"])
            self.cell.set("capacitance", self.channel_params["membrane_capacitance"])
        
        # Soma geometry
        self.cell.branch(self.soma_branch_idx).set("radius", soma_params["diameter"] / 2.0)
        self.cell.branch(self.soma_branch_idx).set("length", soma_params["length"])
        
        # Dendrite geometry (if multi-compartment)
        if self.use_multicompartment and self.dendrite_branch_indices:
            dend_params = self.morphology["dendrite"]
            for idx in self.dendrite_branch_indices:
                self.cell.branch(idx).set("radius", dend_params["diameter"] / 2.0)
                self.cell.branch(idx).set("length", dend_params["length"])
    
    def _insert_channels(self):
        """Insert ion channels with appropriate conductances."""
        if self.use_jaxley_mech:
            self._insert_soma_channels_jaxley_mech()
            if self.use_multicompartment and self.dendrite_branch_indices:
                self._insert_dendrite_channels_jaxley_mech()
        else:
            self._insert_soma_channels()
            if self.use_multicompartment and self.dendrite_branch_indices:
                self._insert_dendrite_channels()
    
    def _insert_soma_channels(self):
        """Insert channels into soma."""
        soma_channels = self.channel_params["soma"]
        soma = self.cell.branch(self.soma_branch_idx)

        # 1. Fast sodium (Na3rp)
        na3rp_params = soma_channels.get("na3rp", {})
        soma.insert(Na3rp())
        soma.set("Na3rp_gbar", na3rp_params.get("gbar", 0.05))
        soma.set("Na3rp_sh", na3rp_params.get("sh", 8.0))
        soma.set("Na3rp_ar", na3rp_params.get("ar", 1.0))
        soma.set("Na3rp_eNa", self.channel_params["e_na"])

        # 2. Persistent sodium (Naps)
        naps_params = soma_channels.get("naps", {})
        soma.insert(Naps())
        soma.set("Naps_gbar", naps_params.get("gbar", 0.001))
        soma.set("Naps_sh", naps_params.get("sh", 0.0))
        soma.set("Naps_ar", naps_params.get("ar", 1.0))
        soma.set("Naps_eNa", self.channel_params["e_na"])

        # 3. Delayed rectifier potassium (KdrRL)
        kdr_params = soma_channels.get("kdrRL", {})
        soma.insert(KdrRL())
        soma.set("KdrRL_gbar", kdr_params.get("gbar", 0.3))
        soma.set("KdrRL_mVh", kdr_params.get("mVh", -25.0))
        soma.set("KdrRL_mslp", kdr_params.get("mslp", 20.0))
        soma.set("KdrRL_eK", self.channel_params["e_k"])

        # 4. Calcium-dependent potassium / mAHP (MAHP)
        mahp_params = soma_channels.get("mAHP", {})
        soma.insert(MAHP())
        soma.set("MAHP_gkcamax", mahp_params.get("gkcamax", 0.03))
        soma.set("MAHP_gcamax", mahp_params.get("gcamax", 3e-5))
        soma.set("MAHP_tau_ca", mahp_params.get("tau_ca", 20.0))
        soma.set("MAHP_eK", self.channel_params["e_k"])
        soma.set("MAHP_eCa", self.channel_params["e_ca"])

        # 5. H-current (Gh)
        gh_params = soma_channels.get("gh", {})
        soma.insert(Gh())
        soma.set("Gh_gbar", gh_params.get("gbar", 2.5e-5))
        soma.set("Gh_half", gh_params.get("half", -70.0))
        soma.set("Gh_slp", gh_params.get("slp", 8.0))

        # 6. Leak
        leak_params = soma_channels.get("leak", {})
        soma.insert(LeakChannel())
        soma.set("LeakChannel_gLeak", leak_params.get("g_leak", 5e-5))
        soma.set("LeakChannel_eLeak", leak_params.get("e_leak", -70.0))
    
    def _insert_dendrite_channels(self):
        """Insert channels into dendrites."""
        dend_channels = self.channel_params["dendrite"]

        for idx in self.dendrite_branch_indices:
            dend = self.cell.branch(idx)

            # 1. L-type calcium (for PICs)
            lca_params = dend_channels.get("L_Ca_inact", {})
            dend.insert(LCaInact())
            dend.set("LCaInact_gbar", lca_params.get("gbar", 6.7e-5))
            dend.set("LCaInact_theta_m", lca_params.get("theta_m", -30.0))
            dend.set("LCaInact_tau_h", lca_params.get("tau_h", 1500.0))
            dend.set("LCaInact_eCa", self.channel_params["e_ca"])

            # 2. H-current
            gh_params = dend_channels.get("gh", {})
            dend.insert(Gh())
            dend.set("Gh_gbar", gh_params.get("gbar", 0.0001))
            dend.set("Gh_half", gh_params.get("half", -70.0))
            dend.set("Gh_slp", gh_params.get("slp", 8.0))

            # 3. Leak
            leak_params = dend_channels.get("leak", {})
            dend.insert(LeakChannel())
            dend.set("LeakChannel_gLeak", leak_params.get("g_leak", 5e-5))
            dend.set("LeakChannel_eLeak", leak_params.get("e_leak", -70.0))
    
    def _insert_soma_channels_jaxley_mech(self):
        """Insert hybrid channels into soma: Na3rp + jaxley-mech K + Leak + MAHP.

        This hybrid approach combines:
        - Na3rp (custom): Fast sodium with slow inactivation (more realistic than HH Na)
        - K (jaxley-mech): Standard delayed rectifier potassium
        - Leak (jaxley-mech): Passive leak
        - MAHP (custom): Calcium-dependent K+ for AHP (limits firing rate)

        This provides the best of both worlds:
        - Na3rp's slow inactivation prevents excessive high-frequency firing
        - MAHP provides medium AHP for physiological firing rates (~10-40 Hz)
        - jaxley-mech K provides reliable repolarization
        """
        jm_params = JAXLEY_MECH_CHANNELS["soma"]
        soma = self.cell.branch(self.soma_branch_idx)

        # 1. Fast sodium with slow inactivation (Na3rp - custom channel)
        # Na3rp has three gating variables: m³hs
        # - m: fast activation
        # - h: fast inactivation
        # - s: slow inactivation (Fleidervish et al.) - KEY DIFFERENCE from HH Na
        na = Na3rp(name="Na3rp")
        soma.insert(na)
        # Na3rp needs ~1.5x HH gNa to compensate for slow inactivation (s gate)
        # and MAHP's hyperpolarizing current
        soma.set("Na3rp_gbar", jm_params["na"]["gNa"] * 1.5)  # ~0.18 S/cm²
        soma.set("Na3rp_sh", 0.0)        # mV - threshold shift (0 = easier to fire)
        soma.set("Na3rp_ar", 1.0)        # slow inact recovery (1=minimal slow inact)
        soma.set("Na3rp_eNa", jm_params["na"]["eNa"])
        soma.set("Na3rp_thinf", -50.0)   # mV - inactivation half-voltage
        soma.set("Na3rp_qinf", 4.0)      # mV - inactivation slope

        # 2. Delayed rectifier potassium (Hodgkin-Huxley K from jaxley-mech)
        k = JaxleyMechK(name="K")
        soma.insert(k)
        soma.set("K_gK", jm_params["k"]["gK"])
        soma.set("K_eK", jm_params["k"]["eK"])

        # 3. Leak (Hodgkin-Huxley Leak from jaxley-mech)
        leak = JaxleyMechLeak(name="Leak")
        soma.insert(leak)
        soma.set("Leak_gLeak", jm_params["leak"]["gLeak"])
        soma.set("Leak_eLeak", jm_params["leak"]["eLeak"])

        # 4. MAHP (calcium-dependent K+ for AHP) to limit firing rate
        # Provides medium after-hyperpolarization that limits firing to ~10-40 Hz
        mahp = MAHP(name="MAHP")
        soma.insert(mahp)
        soma.set("MAHP_gkcamax", 0.03)   # S/cm² - KCa conductance (AHP)
        soma.set("MAHP_gcamax", 3e-5)    # S/cm² - Ca conductance (small, for dynamics)
        soma.set("MAHP_tau_ca", 20.0)    # ms - calcium removal time constant
        soma.set("MAHP_eK", jm_params["k"]["eK"])  # Use same K reversal as K channel
        soma.set("MAHP_eCa", 120.0)      # mV - calcium reversal potential

    def _insert_dendrite_channels_jaxley_mech(self):
        """Insert jaxley-mech channels into dendrites.
        
        For the jaxley-mech mode, dendrites only have passive leak channels
        (no PICs or active conductances).
        """
        jm_params = JAXLEY_MECH_CHANNELS["dendrite"]
        
        for idx in self.dendrite_branch_indices:
            dend = self.cell.branch(idx)
            
            # Passive leak only for dendrites
            leak = JaxleyMechLeak(name="Leak")
            dend.insert(leak)
            dend.set("Leak_gLeak", jm_params["leak"]["gLeak"])
            dend.set("Leak_eLeak", jm_params["leak"]["eLeak"])
    
    def get_cell(self) -> jx.Cell:
        """Return the constructed Jaxley cell."""
        return self.cell
    
    def set_initial_voltage(self, v: float = -70.0):
        """Set initial membrane voltage."""
        self.cell.set("v", v)
    
    def get_channel_list(self) -> List[str]:
        """Return list of inserted channel names."""
        if self.use_jaxley_mech:
            # Hybrid channels: Na3rp (custom) + jaxley-mech K/Leak + MAHP (custom)
            soma_channels = ["Na3rp", "K", "Leak", "MAHP"]
            if self.use_multicompartment:
                dend_channels = ["Leak"]
                return {"soma": soma_channels, "dendrite": dend_channels}
            return soma_channels
        else:
            # Custom motor neuron channels
            soma_channels = ["Na3rp", "Naps", "KdrRL", "MAHP", "Gh", "Leak"]
            if self.use_multicompartment:
                dend_channels = ["LCaInact", "Gh", "Leak"]
                return {"soma": soma_channels, "dendrite": dend_channels}
            return soma_channels
    
    def setup_recording(self, location: str = "soma") -> None:
        """
        Set up voltage recording at specified location.
        
        Parameters
        ----------
        location : str
            "soma" or "dendrite"
        """
        self.cell.delete_recordings()
        
        if location == "soma":
            self.cell.branch(self.soma_branch_idx).loc(0.5).record("v")
        elif location == "dendrite" and self.dendrite_branch_indices:
            # Record from first dendrite
            self.cell.branch(self.dendrite_branch_indices[0]).loc(0.5).record("v")
    
    def setup_stimulus(self, current, location: str = "soma") -> None:
        """
        Set up current injection at specified location.
        
        Parameters
        ----------
        current : array
            Current waveform (nA).
        location : str
            "soma" or "dendrite"
        """
        self.cell.delete_stimuli()
        
        if location == "soma":
            self.cell.branch(self.soma_branch_idx).loc(0.5).stimulate(current)
        elif location == "dendrite" and self.dendrite_branch_indices:
            self.cell.branch(self.dendrite_branch_indices[0]).loc(0.5).stimulate(current)
    
    def simulate(
        self,
        current=None,
        dt: float = 0.025,
        t_max: float = 100.0,
        record_location: str = "soma",
        stim_location: str = "soma",
    ):
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
        record_location : str
            Where to record voltage.
        stim_location : str
            Where to inject current.
        
        Returns
        -------
        voltages : array
            Recorded voltages.
        """
        self.setup_recording(record_location)
        
        if current is not None:
            self.setup_stimulus(current, stim_location)
        
        return jx.integrate(self.cell, delta_t=dt, t_max=t_max)


# =============================================================================
# BIOPHYSICAL INTERNEURON CLASS
# =============================================================================

class BiophysicalInterneuron:
    """
    Single-compartment interneuron with active channels.
    
    Based on Bui et al. (2003) spinal interneuron model.
    
    Parameters
    ----------
    cell_type : str
        "INgII" or "INgIb" - interneuron type.
    """
    
    # Interneuron channel parameters (from NEURON version)
    INTERNEURON_CHANNELS = {
        "na3rp": {"gbar": 0.003, "sh": 1.0, "ar": 1.0},
        "kdrRL": {"gbar": 0.015, "mVh": -25.0, "mslp": 20.0},
        "mAHP": {"gkcamax": 0.0005, "gcamax": 3e-6, "tau_ca": 70.0},
        "gh": {"gbar": 2.5e-5, "half": -70.0, "slp": 8.0},
        "leak": {"g_leak": 5e-5, "e_leak": -71.0},
    }
    
    def __init__(self, cell_type: Literal["INgII", "INgIb"] = "INgII"):
        self.cell_type = cell_type
        self.cell = None
        self.soma_branch_idx = 0
        
        self._build_cell()
        self._define_morphology()
        self._insert_channels()
    
    def _build_cell(self):
        """Create single-compartment Jaxley cell."""
        comp = jx.Compartment()
        soma_branch = jx.Branch(comp, ncomp=1)
        self.cell = jx.Cell(soma_branch, parents=[-1])
    
    def _define_morphology(self):
        """Set interneuron morphology from Bui et al. (2003)."""
        import numpy as np
        
        # Membrane areas from Bui et al.
        Amu = 81390 + 3113  # um²
        Aci = 1.96 * (891.5 + 46.141) / np.sqrt(8)
        A = Amu - Aci  # Use lower bound
        
        # Equivalent sphere diameter
        D = np.sqrt(A / np.pi)
        
        self.cell.branch(self.soma_branch_idx).set("radius", D / 2.0)
        self.cell.branch(self.soma_branch_idx).set("length", D)
        self.cell.set("axial_resistivity", 70.0)
        self.cell.set("capacitance", 1.0)
    
    def _insert_channels(self):
        """Insert interneuron channels."""
        params = self.INTERNEURON_CHANNELS
        soma = self.cell.branch(self.soma_branch_idx)

        # Fast sodium
        soma.insert(Na3rp())
        soma.set("Na3rp_gbar", params["na3rp"]["gbar"])
        soma.set("Na3rp_sh", params["na3rp"]["sh"])
        soma.set("Na3rp_ar", params["na3rp"]["ar"])
        soma.set("Na3rp_eNa", 60.0)

        # Delayed rectifier
        soma.insert(KdrRL())
        soma.set("KdrRL_gbar", params["kdrRL"]["gbar"])
        soma.set("KdrRL_mVh", params["kdrRL"]["mVh"])
        soma.set("KdrRL_mslp", params["kdrRL"]["mslp"])
        soma.set("KdrRL_eK", -80.0)

        # mAHP
        soma.insert(MAHP())
        soma.set("MAHP_gkcamax", params["mAHP"]["gkcamax"])
        soma.set("MAHP_gcamax", params["mAHP"]["gcamax"])
        soma.set("MAHP_tau_ca", params["mAHP"]["tau_ca"])
        soma.set("MAHP_eK", -80.0)
        soma.set("MAHP_eCa", 120.0)

        # H-current
        soma.insert(Gh())
        soma.set("Gh_gbar", params["gh"]["gbar"])
        soma.set("Gh_half", params["gh"]["half"])
        soma.set("Gh_slp", params["gh"]["slp"])

        # Leak
        soma.insert(LeakChannel())
        soma.set("LeakChannel_gLeak", params["leak"]["g_leak"])
        soma.set("LeakChannel_eLeak", params["leak"]["e_leak"])
    
    def get_cell(self) -> jx.Cell:
        """Return the constructed Jaxley cell."""
        return self.cell
    
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


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_motor_neuron(
    model: Literal["Powers2017", "NERLab"] = "Powers2017",
    n_dendrites: Optional[int] = None,
    initial_voltage: float = -70.0,
    use_multicompartment: bool = False,
    use_jaxley_mech: bool = False,
) -> jx.Cell:
    """
    Factory function to create a biophysical motor neuron.
    
    Parameters
    ----------
    model : str
        "Powers2017" or "NERLab".
    n_dendrites : int, optional
        Number of dendrites.
    initial_voltage : float
        Initial membrane voltage (mV).
    use_multicompartment : bool
        If True, create multi-compartment cell with dendrites.
    use_jaxley_mech : bool
        If True, use jaxley-mech Hodgkin-Huxley channels (Na, K, Leak)
        instead of custom motor neuron channels. Default is False.
    
    Returns
    -------
    jx.Cell
        Configured Jaxley cell.
    """
    builder = BiophysicalMotorNeuron(
        model=model,
        n_dendrites=n_dendrites,
        use_multicompartment=use_multicompartment,
        use_jaxley_mech=use_jaxley_mech,
    )
    builder.set_initial_voltage(initial_voltage)
    return builder.get_cell()


def create_motor_neuron_builder(
    model: Literal["Powers2017", "NERLab"] = "Powers2017",
    n_dendrites: Optional[int] = None,
    initial_voltage: float = -70.0,
    use_multicompartment: bool = False,
    use_jaxley_mech: bool = False,
) -> BiophysicalMotorNeuron:
    """
    Factory function to create a biophysical motor neuron builder.
    
    Returns the builder object so you have access to simulate(), setup_recording(), etc.
    
    Parameters
    ----------
    model : str
        "Powers2017" or "NERLab".
    n_dendrites : int, optional
        Number of dendrites.
    initial_voltage : float
        Initial membrane voltage (mV).
    use_multicompartment : bool
        If True, create multi-compartment cell with dendrites.
    use_jaxley_mech : bool
        If True, use jaxley-mech Hodgkin-Huxley channels (Na, K, Leak)
        instead of custom motor neuron channels. Default is False.
    
    Returns
    -------
    BiophysicalMotorNeuron
        The builder object with simulation methods.
    """
    builder = BiophysicalMotorNeuron(
        model=model,
        n_dendrites=n_dendrites,
        use_multicompartment=use_multicompartment,
        use_jaxley_mech=use_jaxley_mech,
    )
    builder.set_initial_voltage(initial_voltage)
    return builder


def create_interneuron(
    cell_type: Literal["INgII", "INgIb"] = "INgII",
    initial_voltage: float = -70.0,
) -> jx.Cell:
    """
    Factory function to create a biophysical interneuron.
    
    Parameters
    ----------
    cell_type : str
        "INgII" or "INgIb".
    initial_voltage : float
        Initial membrane voltage (mV).
    
    Returns
    -------
    jx.Cell
        Configured Jaxley cell.
    """
    builder = BiophysicalInterneuron(cell_type=cell_type)
    builder.cell.set("v", initial_voltage)
    return builder.get_cell()


def create_interneuron_builder(
    cell_type: Literal["INgII", "INgIb"] = "INgII",
    initial_voltage: float = -70.0,
) -> BiophysicalInterneuron:
    """
    Factory function to create a biophysical interneuron builder.
    
    Returns the builder object so you have access to simulate(), setup_recording(), etc.
    
    Parameters
    ----------
    cell_type : str
        "INgII" or "INgIb".
    initial_voltage : float
        Initial membrane voltage (mV).
    
    Returns
    -------
    BiophysicalInterneuron
        The builder object with simulation methods.
    """
    builder = BiophysicalInterneuron(cell_type=cell_type)
    builder.cell.set("v", initial_voltage)
    return builder
