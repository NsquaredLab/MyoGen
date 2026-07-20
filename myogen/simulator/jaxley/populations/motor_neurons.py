"""
Alpha motor neuron populations for motor output - Jaxley Backend.

This module contains the population class for alpha motor neurons, which form
the final common pathway for motor control and drive muscle contraction.

When use_jaxley_mech=True, neurons use a 2-compartment architecture (soma + 1 dendrite)
with the full Powers2017 channel set: Na3rp, Naps, KdrRL, MAHP, Gh on soma;
LCaInact (dendritic PIC), Gh, Leak on dendrite. The single dendrite uses scaled
geometry (4× surface area via d×2.0, L×2.0) to represent NEURON's 4 dendrites.

When use_jaxley_mech=False, neurons use single-compartment custom channels.

The pool implements the Henneman size principle with exponential parameter
distributions (similar to NEURON), creating physiological heterogeneity:
- Most motor neurons are small/low-threshold (recruited first)
- Fewer motor neurons are large/high-threshold (recruited last)
- Parameters scale non-linearly to match experimental observations
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import quantities as pq

from myogen.simulator.jaxley import cells
from myogen.utils.decorators import beartowertype
from myogen.utils.types import RECRUITMENT_THRESHOLDS__ARRAY

from .base import _exp_interp, _Pool


# Powers2017 motor neuron parameter ranges for soma + 1 dendrite architecture.
# Format: (min, max, curvature) where curvature controls exponential shape
# Curvature 0.3 = moderately exponential, 0.5 = linear
#
# MATCHED TO NEURON Powers2017: Full channel set with proper recruitment heterogeneity.
# Soma: Na3rp, Naps, KdrRL, MAHP, Gh, Leak
# Dendrite: LCaInact (dendritic PIC), Gh, Leak
#
# Tuning note — intentional deviations from NEURON native values:
# These adjustments compensate for the *cable dendrite architecture*, not for the
# Jaxley solver. Jaxley's default solver is implicit backward Euler (bwd_euler);
# a side-by-side test (BE vs Crank-Nicolson, scripts/figure_solver_comparison.py)
# at dt = 25 µs shows the two solvers give identical firing rates and spike shapes
# on this model. The shifts below are required because NEURON-native parameter
# values fire ~3× too fast at low currents and depol-block at ~16 nA when run on
# the cable-dendrite topology — the cable filtering cannot tame NEURON-native PIC
# strength on its own. (NEURON achieves stable PIC behaviour implicitly via its
# discrete 4-isopotential-dendrite topology.) Adjustments:
#
#   LCaInact h-gate (inactivation):
#     theta_h: NEURON=+14 mV → Jaxley=(-38, -42) mV
#       Reason: cable-filtered spikes don't drive dendrite above +14 mV, so NEURON's
#       h_inf never inactivates PIC. Shifting theta_h to plateau voltage (~-40 mV)
#       enables PIC self-termination without eliminating it during normal firing.
#     kappa_h: NEURON=4 mV → Jaxley=3 mV (steeper, more selective inactivation)
#     tau_h: NEURON=1500 ms → Jaxley=800 ms (faster inactivation prevents plateau trap)
#
#   gLCa: NEURON native × 0.75 (cable-architecture PIC strength tuning — at 1.0×
#     the cable arch enters depolarisation block from ~16 nA)
#
#   gMAHP_k: NEURON native × 1.125 (slightly elevated to compensate for lower AHP
#     in cable arch vs. NEURON's isopotential compartment)
#
# All other parameters (geometry, soma channels, reversal potentials) match NEURON native.
JAXLEY_HH_PARAMS = {
    # === SOMA GEOMETRY ===
    "soma_diameter_range": (22.0, 30.0, 0.3),       # µm (NEURON: 22-30)
    "soma_length_range": (2952.0, 3665.0, 0.3),     # µm (NEURON: 2952-3665)
    "soma_capacitance_range": (1.356, 1.879, 0.3),  # µF/cm²
    "soma_axial_resistivity": 0.001,                 # Ω·cm (isopotential soma)

    # === SOMA CHANNELS ===
    "gNa3rp_range": (0.01, 0.022, 0.3),             # S/cm²
    "gNaPS_range": (2.6e-5, 2.0e-5, 0.3),             # S/cm² (NEURON native)
    "gKdrRL_range": (0.015, 0.02, 0.3),              # S/cm²
    "gMAHP_ca_range": (6.4e-6, 1.015e-5, 0.075),      # S/cm² (NEURON native)
    "gMAHP_k_range": (5.0625e-4, 6.75e-4, 0.3),               # S/cm² (1.125× NEURON — 0.75× PIC allows lower MAHP)
    "tauMAHP_range": (90.0, 30.0, 0.3),              # ms (slower in small MNs)
    "gGh_soma_range": (3.0e-5, 2.3e-4, 0.3),        # S/cm²
    "gLeak_soma_range": (1.5e-4, 3.77e-4, 0.3),      # S/cm² (min raised to increase rheobase ≥4 nA)
    "eLeak_soma_range": (-71.0, -72.0, 0.5),         # mV

    # === DENDRITE GEOMETRY (single cable, ncomp=4, 4× NEURON single-dend area) ===
    # d×2.0, L×2.0 gives 4× total area matching NEURON's 4 dendrites.
    # Cable Ra provides temporal filtering of PIC dynamics.
    "dend_diameter_range": (17.46, 23.82, 0.3),      # µm (NEURON × 2.0, cable)
    "dend_length_range": (3588.0, 4454.0, 0.3),      # µm (NEURON × 2.0, total cable length)
    "dend_capacitance_range": (0.868, 0.880, 0.3),   # µF/cm² (NEURON native)
    "dend_Ra_range": (51.04, 40.76, 0.3),            # Ω·cm (NEURON native, critical for filtering)

    # === DENDRITE CHANNELS (NEURON native densities on all cable compartments) ===
    # Same density as NEURON; 4× area gives 4× total PIC conductance.
    "gLeak_dend_range": (7.93e-5, 1.75e-4, 0.3),    # S/cm² (NEURON native)
    "eLeak_dend_range": (-71.0, -72.0, 0.5),         # mV
    "gGh_dend_range": (3.0e-5, 2.3e-4, 0.3),        # S/cm² (NEURON native)
    "gLCa_dend_range": (7.21875e-5, 9.69375e-5, 0.3),  # S/cm² (0.75× NEURON base — balanced PIC)
    "theta_m_LCa_range": (-42.0, -39.0, 0.3),        # mV (uniform, cable handles staggering)
    "theta_h_LCa_range": (-38.0, -42.0, 0.3),         # mV (inactivate PIC at plateau ~-40 mV, steep kappa_h=3 for selectivity)

    # === FIXED REVERSAL POTENTIALS ===
    "eNa": 55.0,   # mV
    "eK": -80.0,    # mV
    "eCa": 120.0,   # mV
}


# =============================================================================
# NERLAB MOTOR-NEURON PARAMETERS  (matches the production NEURON model)
# =============================================================================
# Source: config/alpha_mn_default.yaml, section ``nerlab:``. This is the model
# the production NEURON pool uses (``model="NERLab"`` is the YAML default and
# every production example takes that default).
#
# IMPORTANT — encoding convention:
#   Each ``_range`` entry is ``(small_cell_value, large_cell_value)``. We
#   interpolate LINEARLY across normalised recruitment thresholds:
#       value[i] = rt_norm[i] * (large - small) + small
#   This mirrors NEURON's ``special_interp`` (see motor_neurons.py L390-396),
#   which collapses to the same linear form once you account for its
#   ``negative`` flag. The YAML ``[min, max, curve, negative]`` 4-tuples
#   FLIP the (small, large) assignment when ``negative=True``, so six entries
#   below (gnapbar, gkfbar, gksbar, gcaLbar, vtraub_caL, ltau_caL) are stored
#   with the YAML values swapped relative to a naive read. The ``# YAML neg``
#   marker flags each such entry. See:
#     myogen/simulator/neuron/populations/motor_neurons.py:390 (special_interp)
#     config/alpha_mn_default.yaml:14-90 (raw NERLab parameter ranges)
#
# Architecture:
#   - soma : 1 compartment, sphere-like (L = diam)
#   - dend : 1 compartment, ISOPOTENTIAL (ncomp=1)   ← matches NEURON 1-dend
#   - soma channels   : napp  (Na fast + Na persistent + Kfast + Kslow + leak)
#   - dendrite channels: caL  (L-type Ca with no inactivation + leak)
#
# Voltage convention: original 1952 Hodgkin-Huxley
#   V_rest ≈ 0 mV, ENa = +120 mV, EK = -10 mV, ECa = +140 mV
# Cells using NERLAB_PARAMS must initialise V at 0 mV (not -65 mV) and
# detect spikes at a positive threshold (~+50 mV).
NERLAB_PARAMS = {
    # ---- soma geometry  (sphere-like: L = diam in NERLab cells.py) ----
    "soma_diameter_range":    (78.0, 113.0),  # µm
    "soma_capacitance":        1.0,           # µF/cm²
    "soma_axial_resistivity":  70.0,          # Ω·cm

    # ---- soma napp channel conductances ----
    "gnabar_range":  (0.0325,    0.0775),     # S/cm²
    "gnapbar_range": (0.00067,   0.00043),    # S/cm²  YAML neg: small=0.00067, large=0.00043
    "gkfbar_range":  (0.0015,    0.0028),     # S/cm²  YAML neg
    "gksbar_range":  (0.016,     0.02),       # S/cm²  YAML neg
    "gls_range":     (0.000952,  0.001538),   # S/cm²  (napp leak; 1/1050 .. 1/650)
    "rinact_range":  (0.018,     0.063),      # /ms    (r-gate beta)

    # ---- dendrite geometry ----
    "dend_diameter_range":     (48.0,   90.0),    # µm
    "dend_length_range":       (5500.0, 10600.0), # µm
    "dend_capacitance":         1.0,              # µF/cm²
    "dend_axial_resistivity":   70.0,             # Ω·cm

    # ---- dendrite caL channel ----
    "gcaLbar_range":    (6.2e-6,   1.25e-5),  # S/cm²  YAML neg: small=6.2e-6, large=1.25e-5
    "vtraub_caL_range": (34.0,     35.0),     # mV     YAML neg
    "ltau_caL_range":   (47.0,     90.0),     # ms     YAML neg
    "gl_caL_range":     (7.69e-5,  1.65e-4),  # S/cm²  (dendritic leak)

    # ---- fixed napp / caL reversals & offsets (NERLab production values) ----
    "ena":          120.0,  # mV
    "ek":           -10.0,  # mV
    "el_napp":        0.0,  # mV
    "vtraub_napp":    0.0,  # mV
    "ecaL":          140.0, # mV
    "el_caL":          0.0, # mV
}

# Sentinel: NERLab cells live in the 1952-HH frame. When the pool is built with
# model="NERLab" and the user did NOT explicitly override these voltage-frame
# defaults, the pool overrides them to NERLab-appropriate values.
NERLAB_DEFAULT_VHOLD_MV = 0.0
NERLAB_DEFAULT_SPIKE_THRESHOLD_MV = 50.0


@beartowertype
class AlphaMN__Pool(_Pool):
    """
    Container for a population of alpha motor neurons - Jaxley Backend.

    Manages a collection of AlphaMN (alpha motor neuron) cells. These cells form
    the final common pathway for motor control.

    Implements the Henneman size principle with exponential parameter distributions:
    - Low threshold neurons: small soma, high Rin, low rheobase, recruited first
    - High threshold neurons: large soma, low Rin, high rheobase, recruited last

    Parameters are distributed using exponential interpolation (like NEURON version),
    creating realistic motor neuron pool heterogeneity.

    Parameters
    ----------
    n : int
        Number of alpha motor neurons to create.
    recruitment_thresholds__array : RECRUITMENT_THRESHOLDS__ARRAY, optional
        Array of recruitment thresholds for each motor neuron, by default None.
        When provided, parameters are interpolated based on threshold values.
    model : str, optional
        Motor neuron model type ("NERLab" or "Powers2017"), by default "NERLab".
    mode : str, optional
        Simulation mode ("active" or "passive"), by default "active".
    axon_velocities : tuple[float, float], optional
        Min and max axon conduction velocities (m/s), by default (50, 65).
    axon_length : float, optional
        Length of the axon (mm), by default 0.6.
    gamma : float, optional
        Neuromodulation level (a.u.), by default 0.2.
    cell_index : Optional[int], optional
        Specific cell index to create (creates only one cell), by default None.
    initial_voltage__mV : float or list[float], optional
        Initial membrane voltage (mV), by default -67.
    spike_threshold__mV : float, optional
        Spike detection threshold for recording motor neuron spikes, by default 0.0.
        Uses zero-crossing detection for the 2-compartment Powers2017 architecture.
    use_jaxley_mech : bool, optional
        If True, use 2-compartment (soma + dendrite) Powers2017 channels.
        If False, use single-compartment custom channels. Default is False.

    Attributes
    ----------
    neuron_params : dict
        Dictionary of per-neuron parameter arrays computed during pool creation.
        Keys include: 'soma_diameter', 'soma_length', 'capacitance', 'gNa', 'gK',
        'gLeak', 'eLeak', etc. Useful for inspection and analysis.
    """

    def __init__(
        self,
        n: int | None = None,
        recruitment_thresholds__array: RECRUITMENT_THRESHOLDS__ARRAY | None = None,
        model: str = "NERLab",
        mode: str = "active",
        axon_velocities: tuple[float, float] = (50, 65),
        axon_length: float = 0.6,
        gamma: float = 0.2,
        cell_index: Optional[int] = None,
        initial_voltage__mV: Union[float, list[float], None] = None,
        spike_threshold__mV: Optional[float] = None,
        use_jaxley_mech: bool = False,
    ):
        self.recruitment_thresholds__array = recruitment_thresholds__array

        if self.recruitment_thresholds__array is not None:
            self.n = len(self.recruitment_thresholds__array)
        else:
            if n is None:
                raise ValueError("Either n or recruitment_thresholds__array must be provided.")
            self.n = n

        self.model = model
        self.mode = mode
        self.axon_velocities = axon_velocities
        self.axon_length = axon_length
        self.gamma = gamma
        self.cell_index = cell_index
        self.use_jaxley_mech = use_jaxley_mech

        # Voltage-frame defaults depend on the model. NERLab uses the original
        # 1952 HH convention (V_rest ≈ 0 mV, ENa = +120, EK = -10); Powers2017
        # uses modern absolute voltages (V_rest ≈ -67 mV). If the caller didn't
        # explicitly pass these, pick the right frame automatically.
        if initial_voltage__mV is None:
            initial_voltage__mV = (
                NERLAB_DEFAULT_VHOLD_MV if self.model == "NERLab" else -67.0
            )
        if spike_threshold__mV is None:
            spike_threshold__mV = (
                NERLAB_DEFAULT_SPIKE_THRESHOLD_MV if self.model == "NERLab" else 0.0
            )

        # Generate per-neuron parameter arrays (like NEURON version)
        self.neuron_params = self._generate_parameter_arrays()

        # Create cells with pre-computed parameters
        _cells = self._create_cells()

        super().__init__(
            cells=_cells,
            initial_voltage__mV=initial_voltage__mV,
            spike_threshold__mV=spike_threshold__mV,
        )

    def _generate_parameter_arrays(self) -> dict:
        """Generate per-neuron parameter arrays.

        Dispatches on ``self.model``:
          - ``"NERLab"``   → ``_generate_nerlab_params`` (production NEURON model)
          - ``"Powers2017"`` → original Powers2017 cable-architecture params
        """
        if self.model == "NERLab":
            return self._generate_nerlab_params()
        return self._generate_powers2017_params()

    def _generate_nerlab_params(self) -> dict:
        """Per-neuron NERLab parameter arrays, linearly interpolated over recruitment
        thresholds (same convention as Powers2017 path: low threshold → small index).

        Mirrors the layout of ``NEURON: _create_nerlab_cells`` in
        ``myogen/simulator/neuron/populations/motor_neurons.py``.
        """
        p = NERLAB_PARAMS
        n = self.n
        if self.recruitment_thresholds__array is not None:
            rt = np.array(self.recruitment_thresholds__array)
            rt_norm = (rt - rt.min()) / (rt.max() - rt.min() + 1e-10)
            def lin(lo, hi):
                return rt_norm * (hi - lo) + lo
        else:
            def lin(lo, hi):
                return np.linspace(lo, hi, n)

        params: dict = {}
        # Geometry
        params["soma_diameter"]  = lin(*p["soma_diameter_range"])
        params["dend_diameter"]  = lin(*p["dend_diameter_range"])
        params["dend_length"]    = lin(*p["dend_length_range"])
        params["soma_capacitance"]      = np.full(n, p["soma_capacitance"])
        params["soma_axial_resistivity"] = np.full(n, p["soma_axial_resistivity"])
        params["dend_capacitance"]      = np.full(n, p["dend_capacitance"])
        params["dend_axial_resistivity"] = np.full(n, p["dend_axial_resistivity"])

        # napp (soma) channel conductances
        params["gnabar"]  = lin(*p["gnabar_range"])
        params["gnapbar"] = lin(*p["gnapbar_range"])
        params["gkfbar"]  = lin(*p["gkfbar_range"])
        params["gksbar"]  = lin(*p["gksbar_range"])
        params["gls"]     = lin(*p["gls_range"])
        params["rinact"]  = lin(*p["rinact_range"])

        # caL (dendrite) channel
        params["gcaLbar"]    = lin(*p["gcaLbar_range"])
        params["vtraub_caL"] = lin(*p["vtraub_caL_range"])
        params["ltau_caL"]   = lin(*p["ltau_caL_range"])
        params["gl_caL"]     = lin(*p["gl_caL_range"])

        # Fixed reversals & offsets
        for k in ("ena", "ek", "el_napp", "vtraub_napp", "ecaL", "el_caL"):
            params[k] = np.full(n, p[k])
        return params

    def _generate_powers2017_params(self) -> dict:
        """(Original) per-neuron Powers2017 cable-architecture parameters."""
        params = {}

        if self.recruitment_thresholds__array is not None:
            # Interpolate based on recruitment threshold (like NEURON)
            # Normalize thresholds to 0-1 range for interpolation
            rt = np.array(self.recruitment_thresholds__array)
            rt_norm = (rt - rt.min()) / (rt.max() - rt.min() + 1e-10)

            def interp_by_threshold(min_val, max_val, curv):
                """Linear interpolation based on normalized recruitment threshold.

                Matches NEURON's special_interp which uses LINEAR mapping:
                    param = threshold * (max - min) + min

                The recruitment thresholds are already exponentially distributed,
                so linear mapping produces exponential parameter distributions
                (matching NEURON behavior exactly).

                Parameters
                ----------
                min_val : float
                    Value for lowest threshold neuron
                max_val : float
                    Value for highest threshold neuron
                curv : float
                    Curvature parameter (unused — kept for API compatibility)
                """
                return rt_norm * (max_val - min_val) + min_val
        else:
            # Use exponential interpolation (like NEURON _exp_interp)
            def interp_by_threshold(min_val, max_val, curv):
                """Exponential interpolation for n neurons."""
                if curv >= 0.45:
                    return np.linspace(min_val, max_val, self.n)
                return _exp_interp(min_val, max_val, self.n, curv=curv)

        # Generate all parameter arrays
        p = JAXLEY_HH_PARAMS

        # Soma geometry
        params["soma_diameter"] = interp_by_threshold(*p["soma_diameter_range"])
        params["soma_length"] = interp_by_threshold(*p["soma_length_range"])
        params["capacitance"] = interp_by_threshold(*p["soma_capacitance_range"])
        params["axial_resistivity"] = np.full(self.n, p["soma_axial_resistivity"])

        # Soma channels
        params["gNa3rp"] = interp_by_threshold(*p["gNa3rp_range"])
        params["gNaPS"] = interp_by_threshold(*p["gNaPS_range"])
        params["gKdrRL"] = interp_by_threshold(*p["gKdrRL_range"])
        params["gMAHP_ca"] = interp_by_threshold(*p["gMAHP_ca_range"])
        params["gMAHP_k"] = interp_by_threshold(*p["gMAHP_k_range"])
        params["tauMAHP"] = interp_by_threshold(*p["tauMAHP_range"])
        params["gGh_soma"] = interp_by_threshold(*p["gGh_soma_range"])
        params["gLeak_soma"] = interp_by_threshold(*p["gLeak_soma_range"])
        params["eLeak_soma"] = interp_by_threshold(*p["eLeak_soma_range"])

        # Backward compat aliases (used in analysis code / get_parameter_summary)
        params["gNa"] = params["gNa3rp"]
        params["gK"] = params["gKdrRL"]
        params["gLeak"] = params["gLeak_soma"]
        params["eLeak"] = params["eLeak_soma"]

        # Dendrite geometry
        params["dend_diameter"] = interp_by_threshold(*p["dend_diameter_range"])
        params["dend_length"] = interp_by_threshold(*p["dend_length_range"])
        params["dend_cm"] = interp_by_threshold(*p["dend_capacitance_range"])
        params["dend_Ra"] = interp_by_threshold(*p["dend_Ra_range"])

        # Dendrite channels
        params["gLeak_dend"] = interp_by_threshold(*p["gLeak_dend_range"])
        params["eLeak_dend"] = interp_by_threshold(*p["eLeak_dend_range"])
        params["gGh_dend"] = interp_by_threshold(*p["gGh_dend_range"])
        params["gLCa_dend"] = interp_by_threshold(*p["gLCa_dend_range"])
        params["theta_m_LCa"] = interp_by_threshold(*p["theta_m_LCa_range"])
        params["theta_h_LCa"] = interp_by_threshold(*p["theta_h_LCa_range"])

        # Fixed reversal potentials
        params["eNa"] = np.full(self.n, p["eNa"])
        params["eK"] = np.full(self.n, p["eK"])
        params["eCa"] = np.full(self.n, p["eCa"])

        return params

    def _create_cells(self) -> list:
        """Create motor neurons using Jaxley backend.

        Each motor neuron is created with pre-computed parameters from
        self.neuron_params, implementing the Henneman size principle:
        - Low threshold → small soma → high Rin → low rheobase
        - High threshold → large soma → low Rin → high rheobase
        """
        vcon = np.linspace(self.axon_velocities[0], self.axon_velocities[1], self.n)

        # Determine cell creation range
        if self.cell_index is not None:
            init, end = self.cell_index, self.cell_index + 1
        else:
            init, end = 0, self.n

        _cells = []
        for i in range(init, end):
            # Get recruitment threshold for this neuron (if available)
            recruitment_threshold = None
            if self.recruitment_thresholds__array is not None:
                recruitment_threshold = float(self.recruitment_thresholds__array[i])

            # NERLab path: build a tight per-cell parameter dict and skip the
            # Powers2017-shaped one below.
            if self.model == "NERLab":
                neuron_params = {k: self.neuron_params[k][i] for k in self.neuron_params}
                cell = cells.AlphaMN(
                    mode=self.mode,
                    model=self.model,
                    pool__ID=i,
                    use_jaxley_mech=self.use_jaxley_mech,  # ignored for NERLab
                    gamma=self.gamma,                       # ← was missing; controlled caL_gama
                    recruitment_threshold=recruitment_threshold,
                    neuron_params=neuron_params,
                )
                cell.create_axon(
                    length__m=self.axon_length * pq.m,
                    conduction_velocity__m_per_s=vcon[i] * pq.m / pq.s,
                )
                _cells.append(cell)
                continue

            # Powers2017 path — original code below.
            # Extract pre-computed parameters for this neuron
            neuron_params = {
                # Soma geometry
                "soma_diameter": self.neuron_params["soma_diameter"][i],
                "soma_length": self.neuron_params["soma_length"][i],
                "capacitance": self.neuron_params["capacitance"][i],
                "axial_resistivity": self.neuron_params["axial_resistivity"][i],
                # Soma channels
                "gNa3rp": self.neuron_params["gNa3rp"][i],
                "gNaPS": self.neuron_params["gNaPS"][i],
                "gKdrRL": self.neuron_params["gKdrRL"][i],
                "gMAHP_ca": self.neuron_params["gMAHP_ca"][i],
                "gMAHP_k": self.neuron_params["gMAHP_k"][i],
                "tauMAHP": self.neuron_params["tauMAHP"][i],
                "gGh_soma": self.neuron_params["gGh_soma"][i],
                "gLeak_soma": self.neuron_params["gLeak_soma"][i],
                "eLeak_soma": self.neuron_params["eLeak_soma"][i],
                # Dendrite geometry
                "dend_diameter": self.neuron_params["dend_diameter"][i],
                "dend_length": self.neuron_params["dend_length"][i],
                "dend_cm": self.neuron_params["dend_cm"][i],
                "dend_Ra": self.neuron_params["dend_Ra"][i],
                # Dendrite channels
                "gLeak_dend": self.neuron_params["gLeak_dend"][i],
                "eLeak_dend": self.neuron_params["eLeak_dend"][i],
                "gGh_dend": self.neuron_params["gGh_dend"][i],
                "gLCa_dend": self.neuron_params["gLCa_dend"][i],
                "theta_m_LCa": self.neuron_params["theta_m_LCa"][i],
                "theta_h_LCa": self.neuron_params["theta_h_LCa"][i],
                # Reversal potentials
                "eNa": self.neuron_params["eNa"][i],
                "eK": self.neuron_params["eK"][i],
                "eCa": self.neuron_params["eCa"][i],
                # Backward compat aliases
                "gNa": self.neuron_params["gNa"][i],
                "gK": self.neuron_params["gK"][i],
                "gLeak": self.neuron_params["gLeak"][i],
                "eLeak": self.neuron_params["eLeak"][i],
            }

            cell = cells.AlphaMN(
                mode=self.mode,
                model=self.model,
                pool__ID=i,
                use_jaxley_mech=self.use_jaxley_mech,
                recruitment_threshold=recruitment_threshold,
                neuron_params=neuron_params,  # Pass pre-computed parameters
            )

            # Create axon with appropriate delay
            cell.create_axon(
                length__m=self.axon_length * pq.m,
                conduction_velocity__m_per_s=vcon[i] * pq.m / pq.s,
            )

            _cells.append(cell)

        return _cells

    def get_parameter_summary(self) -> str:
        """Return a summary of the motor neuron pool parameter distributions.

        Returns
        -------
        str
            Formatted string showing parameter ranges and statistics.
        """
        lines = [
            "=" * 60,
            "Motor Neuron Pool Parameter Summary",
            "=" * 60,
            f"Number of neurons: {self.n}",
            f"Model: {self.model}",
            f"Mode: {self.mode}",
            f"Powers2017 2-compartment (use_jaxley_mech): {self.use_jaxley_mech}",
            "",
            "Parameter Ranges (min - max):",
            "-" * 40,
        ]

        for param_name, values in self.neuron_params.items():
            if isinstance(values, np.ndarray) and len(values) > 1:
                lines.append(f"  {param_name}: {values.min():.4g} - {values.max():.4g}")
            else:
                lines.append(f"  {param_name}: {values[0]:.4g} (fixed)")

        lines.append("=" * 60)
        return "\n".join(lines)
