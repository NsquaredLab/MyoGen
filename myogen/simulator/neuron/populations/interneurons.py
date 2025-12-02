"""
Interneuron populations for spinal circuit processing.

This module contains population classes for different types of interneurons
that provide inhibitory and excitatory connections within spinal circuits.
"""

from typing import Optional, Union

import numpy as np

from myogen.simulator.neuron import cells
from myogen.utils.decorators import beartowertype

from .base import _get_interneuron_diameter_range__um, _Pool


@beartowertype
class GII__Pool(_Pool):
    """
    Container for a population of group II interneurons.

    Manages a collection of INgII (group II interneuron) cells that provide
    inhibitory feedback in spinal circuits, processing type II afferent input.

    Parameters
    ----------
    n : int
        Number of group II interneurons to create.
    soma_length_range__um : tuple[float, float]
        Min and max soma length (um). By default, it is set to the estimated range for interneurons from Bui et al. 2003 [1]_.
    soma_diameter_range : tuple[float, float]
        Min and max soma diameter (um). By default, it is set to the estimated range for interneurons from Bui et al. 2003 [1]_.
    passive_conductance_range : tuple[float, float]
        Min and max passive membrane conductance (S/cm²).
    na3rp_conductance_range : tuple[float, float]
        Min and max Na3RP sodium channel conductance (S/cm²).
    kdrrl_conductance_range : tuple[float, float]
        Min and max KDRRL potassium channel conductance (S/cm²).
    mahp_ca_conductance_range : tuple[float, float]
        Min and max mAHP calcium conductance (S/cm²).
    mahp_k_conductance_range : tuple[float, float]
        Min and max mAHP potassium conductance (S/cm²).
    mahp_tau_range : tuple[float, float]
        Min and max mAHP time constant (ms).
    gh_conductance_range : tuple[float, float]
        Min and max h-current conductance (S/cm²).
    axon_velocities : tuple[float, float]
        Min and max axon conduction velocities (m/s).
    axon_length : float
        Length of the axon (mm).
    cell_index : int, optional
        Specific cell index to create (creates only one cell), by default None.

    References
    ----------
    .. [1] Bui, T.V., Cushing, S., Dewey, D., Fyffe, R.E., Rose, P.K., 2003. Comparison of the Morphological and Electrotonic Properties of Renshaw Cells, Ia Inhibitory Interneurons, and Motoneurons in the Cat. Journal of Neurophysiology 90, 2900–2918. https://doi.org/10.1152/jn.00533.2003

    """

    def __init__(
        self,
        n: int,
        soma_length_range__um: tuple[float, float] = _get_interneuron_diameter_range__um(),
        soma_diameter_range: tuple[float, float] = _get_interneuron_diameter_range__um(),
        passive_conductance_range: tuple[float, float] = (3e-5, 7e-5),
        na3rp_conductance_range: tuple[float, float] = (0.003, 0.01),
        kdrrl_conductance_range: tuple[float, float] = (0.015, 0.015),
        mahp_ca_conductance_range: tuple[float, float] = (3e-6, 3e-6),
        mahp_k_conductance_range: tuple[float, float] = (5e-4, 5e-4),
        mahp_tau_range: tuple[float, float] = (60, 60),
        gh_conductance_range: tuple[float, float] = (2.5e-5, 2.5e-5),
        axon_velocities: tuple[float, float] = (10, 10),
        axon_length: float = 0.05,
        cell_index: Optional[int] = None,
        initial_voltage__mV: Union[float, list[float]] = -70.0,
    ):
        self.n = n
        self.soma_length_range__um = soma_length_range__um
        self.soma_diameter_range = soma_diameter_range
        self.passive_conductance_range = passive_conductance_range
        self.na3rp_conductance_range = na3rp_conductance_range
        self.kdrrl_conductance_range = kdrrl_conductance_range
        self.mahp_ca_conductance_range = mahp_ca_conductance_range
        self.mahp_k_conductance_range = mahp_k_conductance_range
        self.mahp_tau_range = mahp_tau_range
        self.gh_conductance_range = gh_conductance_range
        self.axon_velocities = axon_velocities
        self.axon_length = axon_length
        self.cell_index = cell_index

        sL = np.linspace(*soma_length_range__um, n)
        sdiam = np.linspace(*soma_diameter_range, n)
        sg_pas = np.linspace(*passive_conductance_range, n)
        sgbar_na3rp = np.linspace(*na3rp_conductance_range, n)
        sgMax_kdrRL = np.linspace(*kdrrl_conductance_range, n)
        sgcamax_mAHP = np.linspace(*mahp_ca_conductance_range, n)
        sgkcamax_mAHP = np.linspace(*mahp_k_conductance_range, n)
        stau_mAHP = np.linspace(*mahp_tau_range, n)
        sghbar_gh = np.linspace(*gh_conductance_range, n)
        vcon = np.linspace(*axon_velocities, n)

        if cell_index is not None:
            init, end = cell_index, cell_index + 1
        else:
            init, end = 0, n

        _cells = []
        for i, (
            sL_i,
            sdiam_i,
            sg_pas_i,
            sgbar_na3rp_i,
            sgMax_kdrRL_i,
            sgcamax_mAHP_i,
            sgkcamax_mAHP_i,
            stau_mAHP_i,
            sghbar_gh_i,
            vcon_i,
        ) in enumerate(
            zip(
                sL[init:end],
                sdiam[init:end],
                sg_pas[init:end],
                sgbar_na3rp[init:end],
                sgMax_kdrRL[init:end],
                sgcamax_mAHP[init:end],
                sgkcamax_mAHP[init:end],
                stau_mAHP[init:end],
                sghbar_gh[init:end],
                vcon[init:end],
            )
        ):
            gII = cells.INgII(pool__ID=i)

            gII.soma.L = sL_i
            gII.soma.diam = sdiam_i
            gII.soma.g_pas = sg_pas_i
            gII.soma.gbar_na3rp = sgbar_na3rp_i
            gII.soma.gMax_kdrRL = sgMax_kdrRL_i
            gII.soma.gcamax_mAHP = sgcamax_mAHP_i
            gII.soma.gkcamax_mAHP = sgkcamax_mAHP_i
            gII.soma.tau_mAHP = stau_mAHP_i
            gII.soma.ghbar_gh = sghbar_gh_i

            import quantities as pq

            gII.create_axon(
                length__m=axon_length * pq.m, conduction_velocity__m_per_s=vcon_i * pq.m / pq.s
            )
            _cells.append(gII)

        super().__init__(cells=_cells, initial_voltage__mV=initial_voltage__mV)


@beartowertype
class GIb__Pool(_Pool):
    """
    Container for a population of group Ib interneurons.

    Manages a collection of INgIb (group Ib interneuron) cells that provide
    inhibitory feedback in spinal circuits, processing type Ib afferent input
    from Golgi tendon organs.

    Parameters
    ----------
    n : int
        Number of group Ib interneurons to create.
    soma_length_range : tuple[float, float]
        Min and max soma length (um).
    soma_diameter_range : tuple[float, float]
        Min and max soma diameter (um).
    passive_conductance_range : tuple[float, float]
        Min and max passive membrane conductance (S/cm²).
    na3rp_conductance_range : tuple[float, float]
        Min and max Na3RP sodium channel conductance (S/cm²).
    kdrrl_conductance_range : tuple[float, float]
        Min and max KDRRL potassium channel conductance (S/cm²).
    mahp_ca_conductance_range : tuple[float, float]
        Min and max mAHP calcium conductance (S/cm²).
    mahp_k_conductance_range : tuple[float, float]
        Min and max mAHP potassium conductance (S/cm²).
    mahp_tau_range : tuple[float, float]
        Min and max mAHP time constant (ms).
    gh_conductance_range : tuple[float, float]
        Min and max h-current conductance (S/cm²).
    axon_velocities : tuple[float, float]
        Min and max axon conduction velocities (m/s).
    axon_length : float
        Length of the axon (mm).
    cell_index : Optional[int], optional
        Specific cell index to create (creates only one cell), by default None.
    """

    def __init__(
        self,
        n: int,
        soma_length_range: tuple[float, float] = _get_interneuron_diameter_range__um(),
        soma_diameter_range: tuple[float, float] = _get_interneuron_diameter_range__um(),
        passive_conductance_range: tuple[float, float] = (3e-5, 8e-5),
        na3rp_conductance_range: tuple[float, float] = (0.01, 0.03),
        kdrrl_conductance_range: tuple[float, float] = (0.035, 0.028),
        mahp_ca_conductance_range: tuple[float, float] = (1e-6, 6e-6),
        mahp_k_conductance_range: tuple[float, float] = (3e-4, 4.5e-4),
        mahp_tau_range: tuple[float, float] = (120, 90),
        gh_conductance_range: tuple[float, float] = (2.5e-5, 2.5e-5),
        axon_velocities: tuple[float, float] = (10, 10),
        axon_length: float = 0.05,
        cell_index: int | None = None,
        initial_voltage__mV: float | list[float] = -70.0,
    ):
        self.n = n
        self.soma_length_range = soma_length_range
        self.soma_diameter_range = soma_diameter_range
        self.passive_conductance_range = passive_conductance_range
        self.na3rp_conductance_range = na3rp_conductance_range
        self.kdrrl_conductance_range = kdrrl_conductance_range
        self.mahp_ca_conductance_range = mahp_ca_conductance_range
        self.mahp_k_conductance_range = mahp_k_conductance_range
        self.mahp_tau_range = mahp_tau_range
        self.gh_conductance_range = gh_conductance_range
        self.axon_velocities = axon_velocities
        self.axon_length = axon_length
        self.cell_index = cell_index

        sL = np.linspace(*soma_length_range, n)
        sdiam = np.linspace(*soma_diameter_range, n)
        sg_pas = np.linspace(*passive_conductance_range, n)
        sgbar_na3rp = np.linspace(*na3rp_conductance_range, n)
        sgMax_kdrRL = np.linspace(*kdrrl_conductance_range, n)
        sgcamax_mAHP = np.linspace(*mahp_ca_conductance_range, n)
        sgkcamax_mAHP = np.linspace(*mahp_k_conductance_range, n)
        stau_mAHP = np.linspace(*mahp_tau_range, n)
        sghbar_gh = np.linspace(*gh_conductance_range, n)
        vcon = np.linspace(*axon_velocities, n)

        if cell_index is not None:
            init, end = cell_index, cell_index + 1
        else:
            init, end = 0, n

        _cells = []
        for i, (
            sL_i,
            sdiam_i,
            sg_pas_i,
            sgbar_na3rp_i,
            sgMax_kdrRL_i,
            sgcamax_mAHP_i,
            sgkcamax_mAHP_i,
            stau_mAHP_i,
            sghbar_gh_i,
            vcon_i,
        ) in enumerate(
            zip(
                sL[init:end],
                sdiam[init:end],
                sg_pas[init:end],
                sgbar_na3rp[init:end],
                sgMax_kdrRL[init:end],
                sgcamax_mAHP[init:end],
                sgkcamax_mAHP[init:end],
                stau_mAHP[init:end],
                sghbar_gh[init:end],
                vcon[init:end],
            )
        ):
            gIb = cells.INgIb(pool__ID=i)

            gIb.soma.L = sL_i
            gIb.soma.diam = sdiam_i
            gIb.soma.g_pas = sg_pas_i
            gIb.soma.gbar_na3rp = sgbar_na3rp_i
            gIb.soma.gMax_kdrRL = sgMax_kdrRL_i
            gIb.soma.gcamax_mAHP = sgcamax_mAHP_i
            gIb.soma.gkcamax_mAHP = sgkcamax_mAHP_i
            gIb.soma.tau_mAHP = stau_mAHP_i
            gIb.soma.ghbar_gh = sghbar_gh_i

            import quantities as pq

            gIb.create_axon(
                length__m=axon_length * pq.m, conduction_velocity__m_per_s=vcon_i * pq.m / pq.s
            )

            _cells.append(gIb)

        super().__init__(cells=_cells, initial_voltage__mV=initial_voltage__mV)
