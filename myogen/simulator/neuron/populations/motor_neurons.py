"""
Alpha motor neuron populations for motor output.

This module contains the population class for alpha motor neurons, which form
the final common pathway for motor control and drive muscle contraction.
"""

from typing import Optional, Union

import numpy as np

from myogen.simulator.neuron import cells
from myogen.utils.decorators import beartowertype
from myogen.utils.types import RECRUITMENT_THRESHOLDS__ARRAY

from .base import _exp_interp, _Pool


@beartowertype
class AlphaMN__Pool(_Pool):
    """
    Container for a population of alpha motor neurons.

    Manages a collection of AlphaMN (alpha motor neuron) cells with different
    biophysical models: ModALS or Powers2017. These cells form the final
    common pathway for motor control.

    Parameters
    ----------
    n : int
        Number of alpha motor neurons to create.
    model : str
        Motor neuron model type ("ModALS" or "Powers2017").
    mode : str
        Simulation mode ("active" or "passive").
    axon_velocities : tuple[float, float]
        Min and max axon conduction velocities (m/s).
    axon_length : float
        Length of the axon (mm).
    gamma : float
        Neuromodulation level (a.u.).
    cell_index : Optional[int], optional
        Specific cell index to create (creates only one cell), by default None.
    lambda_factor : float, optional
        Lambda factor for Powers2017 model persistent sodium scaling, by default 1.0.

    Powers2017 Model Parameters (required when model="Powers2017")
    --------------------------------------------------------
    soma_length_range : tuple[float, float, float], optional
        Soma length [min, max, curve] (μm).
    soma_diameter_range : tuple[float, float, float], optional
        Soma diameter [min, max, curve] (μm).
    soma_capacitance_range : tuple[float, float, float], optional
        Soma capacitance [min, max, curve] (μF/cm²).
    soma_passive_conductance_range : tuple[float, float, float], optional
        Soma passive conductance [min, max, curve] (S/cm²).
    soma_passive_reversal_range : tuple[float, float, float], optional
        Soma passive reversal potential [min, max, curve] (mV).
    soma_na3rp_conductance_range : tuple[float, float, float], optional
        Soma Na3RP conductance [min, max, curve] (S/cm²).
    soma_naps_conductance_range : tuple[float, float, float], optional
        Soma NaPS conductance [min, max, curve] (S/cm²).
    soma_kdrrl_conductance_range : tuple[float, float, float], optional
        Soma KDRRL conductance [min, max, curve] (S/cm²).
    soma_mahp_ca_conductance_range : tuple[float, float, float], optional
        Soma mAHP calcium conductance [min, max, curve] (S/cm²).
    soma_mahp_k_conductance_range : tuple[float, float, float], optional
        Soma mAHP potassium conductance [min, max, curve] (S/cm²).
    soma_mahp_tau_range : tuple[float, float, float], optional
        Soma mAHP time constant [min, max, curve] (ms).
    soma_gh_conductance_range : tuple[float, float, float], optional
        Soma h-current conductance [min, max, curve] (S/cm²).
    dendrite_length_range : tuple[float, float, float], optional
        Dendrite length [min, max, curve] (μm).
    dendrite_diameter_range : tuple[float, float, float], optional
        Dendrite diameter [min, max, curve] (μm).
    dendrite_passive_conductance_range : tuple[float, float, float], optional
        Dendrite passive conductance [min, max, curve] (S/cm²).
    dendrite_passive_reversal_range : tuple[float, float, float], optional
        Dendrite passive reversal potential [min, max, curve] (mV).
    dendrite_resistance_range : tuple[float, float, float], optional
        Dendrite axial resistance [min, max, curve] (Ω·cm).
    dendrite_capacitance_range : tuple[float, float, float], optional
        Dendrite capacitance [min, max, curve] (μF/cm²).
    dendrite_gh_conductance_range : tuple[float, float, float], optional
        Dendrite h-current conductance [min, max, curve] (S/cm²).
    dendrite_ca_conductance_ranges : tuple[tuple[float, float, float], ...], optional
        L-type Ca conductance ranges for each dendrite (4 tuples).
    dendrite_ca_theta_m_range : tuple[float, float, float], optional
        Ca channel activation threshold [min, max, curve] (mV).
    dendrite_ca_theta_h_range : tuple[float, float, float], optional
        Ca channel inactivation threshold [min, max, curve] (mV).
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
        lambda_factor: float = 1.0,
        initial_voltage__mV: Union[float, list[float]] = -67,
        # Powers2017 parameters
        # Soma parameters
        soma_length_range: tuple[float, float, float] = (2952, 3665, 0.3),
        soma_diameter_range: tuple[float, float, float] = (22, 30, 0.3),
        soma_capacitance_range: tuple[float, float, float] = (1.35546, 1.87853, 0.3),
        soma_passive_conductance_range: tuple[float, float, float] = (
            8.11e-5,
            3.77e-4,
            0.3,
        ),
        soma_passive_reversal_range: tuple[float, float, float] = (-71, -72, 0.3),
        soma_na3rp_conductance_range: tuple[float, float, float] = (0.01, 0.022, 0.3),
        soma_naps_conductance_range: tuple[float, float, float] = (2.6e-5, 2e-5, 0.3),
        soma_kdrrl_conductance_range: tuple[float, float, float] = (0.015, 0.02, 0.3),
        soma_mahp_ca_conductance_range: tuple[float, float, float] = (
            6.4e-6,
            1.015e-5,
            0.075,
        ),
        soma_mahp_k_conductance_range: tuple[float, float, float] = (4.5e-4, 6e-4, 0.3),
        soma_mahp_tau_range: tuple[float, float, float] = (90, 30, 0.3),
        soma_gh_conductance_range: tuple[float, float, float] = (3e-5, 2.3e-4, 0.3),
        # Dendrite parameters
        dendrite_length_range: tuple[float, float, float] = (1794.13, 2226.91, 0.3),
        dendrite_diameter_range: tuple[float, float, float] = (8.73071, 11.9055, 0.3),
        dendrite_passive_conductance_range: tuple[float, float, float] = (
            7.93e-5,
            1.75e-4,
            0.3,
        ),
        dendrite_passive_reversal_range: tuple[float, float, float] = (-71, -72, 0.3),
        dendrite_resistance_range: tuple[float, float, float] = (51.038, 40.755, 0.3),
        dendrite_capacitance_range: tuple[float, float, float] = (
            0.867781,
            0.880407,
            0.3,
        ),
        dendrite_gh_conductance_range: tuple[float, float, float] = (3e-5, 2.3e-4, 0.3),
        dendrite_ca_conductance_ranges: tuple[tuple[float, float, float], ...] = (
            (8.5e-5, 1.18e-4, 0.3),
            (9.5e-5, 1.28e-4, 0.3),
            (1e-4, 1.38e-4, 0.3),
            (1.15e-4, 1.53e-4, 0.3),
        ),
        dendrite_ca_theta_m_range: tuple[float, float, float] = (-42, -39, 0.3),
        dendrite_ca_theta_h_range: tuple[float, float, float] = (10, -10, 0.3),
    ):
        self.n = n
        self.recruitment_thresholds__array = recruitment_thresholds__array

        if self.recruitment_thresholds__array is not None:
            self.n = len(self.recruitment_thresholds__array)

        if self.n is None and self.recruitment_thresholds__array is None:
            raise ValueError(
                "Either n or recruitment_thresholds__array must be provided."
            )

        self.model = model
        self.mode = mode
        self.axon_velocities = axon_velocities
        self.axon_length = axon_length
        self.gamma = gamma
        self.cell_index = cell_index
        self.lambda_factor = lambda_factor

        # Store Powers2017 parameters
        self.soma_length_range = soma_length_range
        self.soma_diameter_range = soma_diameter_range
        self.soma_capacitance_range = soma_capacitance_range
        self.soma_passive_conductance_range = soma_passive_conductance_range
        self.soma_passive_reversal_range = soma_passive_reversal_range
        self.soma_na3rp_conductance_range = soma_na3rp_conductance_range
        self.soma_naps_conductance_range = soma_naps_conductance_range
        self.soma_kdrrl_conductance_range = soma_kdrrl_conductance_range
        self.soma_mahp_ca_conductance_range = soma_mahp_ca_conductance_range
        self.soma_mahp_k_conductance_range = soma_mahp_k_conductance_range
        self.soma_mahp_tau_range = soma_mahp_tau_range
        self.soma_gh_conductance_range = soma_gh_conductance_range
        self.dendrite_length_range = dendrite_length_range
        self.dendrite_diameter_range = dendrite_diameter_range
        self.dendrite_passive_conductance_range = dendrite_passive_conductance_range
        self.dendrite_passive_reversal_range = dendrite_passive_reversal_range
        self.dendrite_resistance_range = dendrite_resistance_range
        self.dendrite_capacitance_range = dendrite_capacitance_range
        self.dendrite_gh_conductance_range = dendrite_gh_conductance_range
        self.dendrite_ca_conductance_ranges = dendrite_ca_conductance_ranges
        self.dendrite_ca_theta_m_range = dendrite_ca_theta_m_range
        self.dendrite_ca_theta_h_range = dendrite_ca_theta_h_range

        if model == "NERLab":
            _cells = self._create_nerlab_cells()
        elif model == "Powers2017":
            _cells = self._create_powers2017_cells()
        else:
            raise ValueError("Could not find the specific model for alpha MNs.")

        super().__init__(cells=_cells, initial_voltage__mV=initial_voltage__mV)

    def _create_nerlab_cells(self) -> list:
        """Create motor neurons using the NERLab model."""

        def special_interp(x, y, n, curv=None, negative=None):
            return self.recruitment_thresholds__array * (y - x) + x

        if self.recruitment_thresholds__array is None:
            interpF = _exp_interp
        else:
            interpF = special_interp

        # Soma parameters (using expInterp with hardcoded values from original)
        Diam_soma = interpF(78, 113, self.n, curv=1.0 / 14)
        Gnabar = interpF(0.0325, 0.0775, self.n, curv=1 / 2.5)
        Gnapbar = interpF(0.00043, 0.00067, self.n, curv=1 / 2.1, negative=True)
        Gkfbar = interpF(0.0028, 0.0015, self.n, curv=1 / 25, negative=True)
        Gksbar = interpF(0.02, 0.016, self.n, curv=1.0 / 6, negative=True)
        Mact = interpF(13, 20, self.n, curv=1 / 3)
        Rinact = interpF(0.018, 0.063, self.n, curv=1 / 4)
        Gls = interpF(1.0 / 1050, 1.0 / 650, self.n, curv=1 / 2.5)

        # Dendrite parameters
        Diam_dend = interpF(48.0, 90.0, self.n, curv=1.0 / 5)
        L_dend = interpF(5500, 10600, self.n, curv=1.0 / 12)
        GcaLbar = interpF(1.25e-5, 6.2e-6, self.n, curv=1 / 3, negative=True)
        Vtraub_caL = interpF(35, 34, self.n, curv=1 / 30, negative=True)
        LTAU_caL = interpF(90, 47, self.n, curv=1 / 3, negative=True)
        Gl_caL = interpF(1.0 / 13000, 1.0 / 6050, self.n, curv=1 / 2.5)

        vcon = np.linspace(self.axon_velocities[0], self.axon_velocities[1], self.n)

        # Determine cell creation range
        if self.cell_index is not None:
            init, end = self.cell_index, self.cell_index + 1
        else:
            init, end = 0, self.n

        _cells = []
        for i in range(init, end):
            cell = cells.AlphaMN(
                segments__count=1,
                mode=self.mode,
                dendrites__count=1,
                model=self.model,
                class__ID=self.cell_index,
                pool__ID=i,
            )
            cell.create_axon(length__m=self.axon_length, conduction_velocity__m_per_s=vcon[i])

            # Soma biophysical parameters
            cell.soma.L = Diam_soma[i]
            cell.soma.diam = Diam_soma[i]
            cell.soma.ena = 120.0
            cell.soma.ek = -10.0
            cell.soma.el_napp = 0.0
            cell.soma.vtraub_napp = 0.0
            cell.soma.Ra = 70.0
            cell.soma.cm = 1.0
            cell.soma.gl_napp = Gls[i]
            cell.soma.gnabar_napp = Gnabar[i]
            cell.soma.gnapbar_napp = Gnapbar[i]
            cell.soma.gkfbar_napp = Gkfbar[i]
            cell.soma.gksbar_napp = Gksbar[i]
            cell.soma.mact_napp = Mact[i]
            cell.soma.rinact_napp = Rinact[i]

            # Dendrite parameters
            cell.dend[0].Ra = 70.0
            cell.dend[0].cm = 1.0
            cell.dend[0].L = L_dend[i]
            cell.dend[0].diam = Diam_dend[i]
            cell.dend[0].ecaL = 140
            cell.dend[0].gama_caL = self.gamma
            cell.dend[0].gcaLbar_caL = GcaLbar[i]
            cell.dend[0].vtraub_caL = Vtraub_caL[i]
            cell.dend[0].Ltau_caL = LTAU_caL[i]
            cell.dend[0].gl_caL = Gl_caL[i]
            cell.dend[0].el_caL = 0.0
            _cells.append(cell)

        return _cells

    def _create_powers2017_cells(self) -> list:
        """Create motor neurons using the Powers2017 model."""
        if self.recruitment_thresholds__array is None:
            interpF = lambda x, y, z: _exp_interp(first=x, last=y, n=self.n, curv=z)
        else:
            interpF = lambda x, y, _: self.recruitment_thresholds__array * (y - x) + x

        # Geometry parameters
        sL = interpF(*self.soma_length_range)
        sdiam = interpF(*self.soma_diameter_range)
        scm = interpF(*self.soma_capacitance_range)

        # Biophysics parameters
        sg_pas = interpF(*self.soma_passive_conductance_range)
        se_pas = interpF(*self.soma_passive_reversal_range)
        sgbar_na3rp = interpF(*self.soma_na3rp_conductance_range)
        sgbar_naps = interpF(*self.soma_naps_conductance_range)
        sgMax_kdrRL = interpF(*self.soma_kdrrl_conductance_range)
        sgcamax_mAHP = interpF(*self.soma_mahp_ca_conductance_range)
        sgkcamax_mAHP = interpF(*self.soma_mahp_k_conductance_range)
        staur_mAHP = interpF(*self.soma_mahp_tau_range)
        sghbar_gh = interpF(*self.soma_gh_conductance_range)

        # Dendrite parameters
        dL = interpF(*self.dendrite_length_range)
        ddiam = interpF(*self.dendrite_diameter_range)
        dg_pas = interpF(*self.dendrite_passive_conductance_range)
        de_pas = interpF(*self.dendrite_passive_reversal_range)
        dRa = interpF(*self.dendrite_resistance_range)
        dcm = interpF(*self.dendrite_capacitance_range)
        dghbar_gh = interpF(*self.dendrite_gh_conductance_range)

        # L-type calcium channels for each dendrite
        d_ca_conductances = [
            interpF(*ca_range) for ca_range in self.dendrite_ca_conductance_ranges
        ]
        dtheta_m_L_Ca_inact = interpF(*self.dendrite_ca_theta_m_range)
        dtheta_h_L_Ca_inact = interpF(*self.dendrite_ca_theta_h_range)

        vcon = np.linspace(*self.axon_velocities, self.n)

        if self.cell_index is not None:
            init, end = self.cell_index, self.cell_index + 1
        else:
            init, end = 0, self.n

        _cells = []
        for i in range(init, end):
            cell = cells.AlphaMN(
                segments__count=1,
                mode=self.mode,
                dendrites__count=4,
                model=self.model,
                class__ID=self.cell_index,
                pool__ID=i,
            )

            # Set soma parameters
            cell.soma.L = sL[i]
            cell.soma.diam = sdiam[i]
            cell.create_axon(length__m=self.axon_length, conduction_velocity__m_per_s=vcon[i])
            cell.soma.g_pas = sg_pas[i]
            cell.soma.e_pas = se_pas[i]
            cell.soma.cm = scm[i]
            cell.soma.gbar_na3rp = sgbar_na3rp[i]
            cell.soma.gbar_naps = sgbar_naps[i] * self.lambda_factor
            cell.soma.gMax_kdrRL = sgMax_kdrRL[i]
            cell.soma.gcamax_mAHP = sgcamax_mAHP[i]
            cell.soma.gkcamax_mAHP = sgkcamax_mAHP[i]
            cell.soma.taur_mAHP = staur_mAHP[i]
            cell.soma.ghbar_gh = sghbar_gh[i]

            # Set dendrite parameters
            for j, d in enumerate(cell.dend):
                d.L = dL[i]
                d.diam = ddiam[i]
                d.g_pas = dg_pas[i]
                d.e_pas = de_pas[i]
                d.Ra = dRa[i]
                d.cm = dcm[i]
                d.ghbar_gh = dghbar_gh[i]

                if self.mode == "active":
                    d.gcabar_L_Ca_inact = d_ca_conductances[j][i] * self.gamma
                    d.theta_m_L_Ca_inact = dtheta_m_L_Ca_inact[i]
                    d.theta_h_L_Ca_inact = dtheta_h_L_Ca_inact[i]

            _cells.append(cell)

        return _cells
