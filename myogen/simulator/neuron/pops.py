from functools import partial
from typing import Optional, Union

import numpy as np
from scipy.optimize import curve_fit

from myogen import setup_myogen
from myogen.simulator.neuron import cells
from myogen.utils.decorators import beartowertype


def _exp_crescent(x, a, b, c):
    return a * np.exp(b * x) + c


def _exp_decrescent(x, a, b, c):
    return a * np.exp(-b * x) + c


@beartowertype
def _exp_interp(
    first: float, last: float, n: int, curv: float = 0.33, negative: bool = False
):
    assert curv <= 0.5
    c1 = first <= last
    if negative:
        c1 = last <= first
    x = [0, 2, 4, 4]  # This is a hack to hide curve_fit warnings [covariance]
    xp = np.linspace(0, 4, n)
    if c1:
        yn = np.array([first, first + (last - first) * curv, last, last]) / first
        popt, _ = curve_fit(_exp_crescent, x, yn)
        param = _exp_crescent(xp, *popt) * first
    else:
        yn = np.array([first, last + (first - last) * curv, last, last]) / last
        popt, _ = curve_fit(_exp_decrescent, x, yn)
        param = _exp_decrescent(xp, *popt) * last
    return param


@beartowertype
class _Pool:
    """
    Base class for neuron cell populations.

    Provides common functionality for managing groups of neurons including
    voltage initialization for populations with real NEURON sections.

    Parameters
    ----------
    cells : list
        List of neuron cells in the population.
    initial_voltage__mV : Union[float, list[float]], optional
        Initial membrane voltage(s) in millivolts for populations with real
        NEURON sections. Can be a single value applied to all cells or a list
        of per-cell values. If None (default), no voltage initialization is
        performed, which is appropriate for dummy cell populations.

    Notes
    -----
    Populations with dummy cells (e.g., DescendingDrive, AffIa, AffIb) should
    not provide initial_voltage__mV as they have no real soma or dendrite
    sections to initialize. Populations with real NEURON sections (e.g.,
    AlphaMN, AffII, GII, GIb) should provide appropriate voltage values.
    """

    def __init__(
        self,
        cells: list,
        initial_voltage__mV: Optional[Union[float, list[float]]] = None,
    ):
        self._cells = cells

        # Handle initial voltage - only if provided
        if initial_voltage__mV is not None:
            n_cells = len(cells)
            if isinstance(initial_voltage__mV, (int, float)):
                self.initial_voltage_values__mV = [initial_voltage__mV] * n_cells
            else:
                assert len(initial_voltage__mV) == n_cells, (
                    f"initial_voltage__mV list length ({len(initial_voltage__mV)}) must match number of cells ({n_cells})"
                )
                self.initial_voltage_values__mV = list(initial_voltage__mV)
        else:
            self.initial_voltage_values__mV = None

    def __iter__(self):
        """Enable iteration over the cells."""
        return iter(self._cells)

    def __getitem__(self, index):
        """Return the cell at the specified index."""
        return self._cells[index]

    def __len__(self):
        """Return the number of cells in the population."""
        return len(self._cells)

    def get_initialization_data(self) -> tuple[list, list]:
        """
        Return sections and their initial voltages for NEURON simulation setup.

        Collects all soma and dendrite sections from the neuron population
        along with their corresponding initial voltage values for proper
        NEURON simulation initialization. Returns empty lists if this
        population doesn't have voltage initialization (e.g., dummy cells).

        Returns
        -------
        tuple[list, list]
            First list contains NEURON section objects (soma and dendrites).
            Second list contains corresponding initial voltages in mV.
            Both lists will be empty if population has no voltage initialization.
        """
        # Return empty lists if no voltage initialization needed
        if self.initial_voltage_values__mV is None:
            return [], []

        sections = []
        voltages = []

        for cell_idx, cell in enumerate(self._cells):
            cell_voltage = self.initial_voltage_values__mV[cell_idx]

            # Skip dummy cells (they don't have real NEURON sections)
            if (
                hasattr(cell, "ns")
                and hasattr(cell.ns, "__class__")
                and "DUMMY" in str(type(cell.ns))
            ):
                continue

            # Add soma section
            if hasattr(cell, "soma"):
                sections.append(cell.soma)
                voltages.append(cell_voltage)

            # Add all dendrite sections
            if hasattr(cell, "dend"):
                for dendrite in cell.dend:
                    sections.append(dendrite)
                    voltages.append(cell_voltage)

        return sections, voltages


@beartowertype
class DescendingDrive__Pool(_Pool):
    """
    Container for a population of descending drive neurons.

    Manages a collection of DD (descending drive) cells that generate
    Poisson random processes for cortical input to spinal circuits.

    Parameters
    ----------
    n : int
        Number of descending drive neurons to create.
    poisson_random_process_order : int
        Order parameter for the Poisson process generation.
    timestep__ms : float
        Time step for simulation (ms).
    """

    def __init__(self, n: int, poisson_random_process_order: int, timestep__ms: float):
        self.n = n
        self.poisson_random_process_order = poisson_random_process_order
        self.timestep__ms = timestep__ms

        _cells = []

        super().__init__(
            cells=[
                cells.DD(N=poisson_random_process_order, dt=timestep__ms)
                for _ in range(n)
            ]
        )


@beartowertype
class AffIa__Pool(_Pool):
    """
    Container for a population of afferent Ia neurons.

    Manages a collection of AffIa (type Ia afferent) cells that provide
    proprioceptive feedback from muscle spindles to spinal circuits.

    Parameters
    ----------
    n : int
        Number of type Ia afferent neurons to create.
    recruitment_thresholds : tuple[float, float]
        Min and max recruitment thresholds (Hz).
    axon_velocities : tuple[float, float]
        Min and max axon conduction velocities (m/s).
    axon_length__m : float
        Length of the axon (m).
    poisson_random_process_order : int
        Order parameter for the Poisson process generation.
    timestep__ms : float
        Time step for simulation (ms).
    init_order : int
        Initial order parameter for afferent initialization.
    """

    def __init__(
        self,
        n: int,
        timestep__ms: float,
        recruitment_thresholds: tuple[float, float] = (0, 40),
        axon_velocities: tuple[float, float] = (58, 72),
        axon_length__m: float = 0.6,
        poisson_random_process_order: int = 145,
        init_order: int = 0,
    ):
        self.n = n
        self.recruitment_thresholds = recruitment_thresholds
        self.axon_velocities = axon_velocities
        self.axon_length = axon_length__m
        self.poisson_random_process_order = poisson_random_process_order
        self.timestep__ms = timestep__ms
        self.init_order = init_order

        rt = np.linspace(*recruitment_thresholds, n)
        vcon = np.linspace(*axon_velocities, n)

        _cells = []
        for rt_i, vcon_i in zip(rt, vcon):
            ia = cells.AffIa(
                RT=rt_i,
                N=poisson_random_process_order,
                dt=timestep__ms,
                initN=init_order,
            )
            ia.create_axon(length__m=axon_length__m, velcon__m_per_s=vcon_i)
            _cells.append(ia)

        super().__init__(cells=_cells)


@beartowertype
class AffII__Pool(_Pool):
    """
    Container for a population of afferent II neurons.

    Manages a collection of AffII (type II afferent) cells that provide
    secondary proprioceptive feedback from muscle spindles to spinal circuits.

    Parameters
    ----------
    n : int
        Number of type II afferent neurons to create.
    recruitment_thresholds : tuple[float, float]
        Min and max recruitment thresholds (Hz).
    axon_velocities : tuple[float, float]
        Min and max axon conduction velocities (m/s).
    axon_length__m : float
        Length of the axon (m).
    poisson_random_process_order : int
        Order parameter for the Poisson process generation.
    timestep__ms : float
        Time step for simulation (ms).
    init_order : int
        Initial order parameter for afferent initialization.
    """

    def __init__(
        self,
        n: int,
        timestep__ms: float,
        recruitment_thresholds: tuple[float, float] = (0, 50),
        axon_velocities: tuple[float, float] = (32, 52),
        axon_length__m: float = 0.6,
        poisson_random_process_order: int = 500,
        init_order: int = 0,
    ):
        self.n = n
        self.recruitment_thresholds = recruitment_thresholds
        self.axon_velocities = axon_velocities
        self.axon_length = axon_length__m
        self.poisson_random_process_order = poisson_random_process_order
        self.timestep__ms = timestep__ms
        self.init_order = init_order

        rt = np.linspace(*recruitment_thresholds, n)
        vcon = np.linspace(*axon_velocities, n)

        _cells = []
        for rt_i, vcon_i in zip(rt, vcon):
            ii = cells.AffII(
                RT=rt_i,
                N=poisson_random_process_order,
                dt=timestep__ms,
                initN=init_order,
            )
            ii.create_axon(length__m=axon_length__m, velcon__m_per_s=vcon_i)
            _cells.append(ii)

        super().__init__(cells=_cells)


@beartowertype
class AffIb__Pool(_Pool):
    """
    Container for a population of afferent Ib neurons.

    Manages a collection of AffIb (type Ib afferent) cells that provide
    primary proprioceptive feedback from Golgi tendon organs to spinal circuits.

    Parameters
    ----------
    n : int
        Number of type Ib afferent neurons to create.
    recruitment_thresholds : tuple[float, float]
        Min and max recruitment thresholds (Hz).
    axon_velocities : tuple[float, float]
        Min and max axon conduction velocities (m/s).
    axon_length : float
        Length of the axon (mm).
    poisson_random_process_order : int
        Order parameter for the Poisson process generation.
    timestep__ms : float
        Time step for simulation (ms).
    init_order : int
        Initial order parameter for afferent initialization.
    """

    def __init__(
        self,
        n: int,
        timestep__ms: float,
        recruitment_thresholds: tuple[float, float] = (0, 50),
        axon_velocities: tuple[float, float] = (62, 66),
        axon_length: float = 0.6,
        poisson_random_process_order: int = 30,
        init_order: int = 0,
    ):
        self.n = n
        self.recruitment_thresholds = recruitment_thresholds
        self.axon_velocities = axon_velocities
        self.axon_length = axon_length
        self.poisson_random_process_order = poisson_random_process_order
        self.timestep__ms = timestep__ms
        self.init_order = init_order

        rt = np.linspace(*recruitment_thresholds, n)
        vcon = np.linspace(*axon_velocities, n)

        _cells = []
        for rt_i, vcon_i in zip(rt, vcon):
            ib = cells.AffIb(
                RT=rt_i,
                N=poisson_random_process_order,
                dt=timestep__ms,
                initN=init_order,
            )
            ib.create_axon(length__m=axon_length, velcon__m_per_s=vcon_i)
            _cells.append(ib)

        super().__init__(cells=_cells)


def _get_interneuron_diameter_range__μm() -> tuple[float, float]:
    """Estimate interneuron soma diameter range based on Biu et al. 2003 [1]_.

    Returns
    -------
    tuple[float, float]
        Estimated diameter range (min, max) in micrometers.

    References
    ----------
    .. [1] Bui, T.V., Cushing, S., Dewey, D., Fyffe, R.E., Rose, P.K., 2003. Comparison of the Morphological and Electrotonic Properties of Renshaw Cells, Ia Inhibitory Interneurons, and Motoneurons in the Cat. Journal of Neurophysiology 90, 2900–2918. https://doi.org/10.1152/jn.00533.2003

    """
    A_cell = 81390 + 3113
    A_ci = 1.96 * (891.5 + 46.141) / np.sqrt(8)
    A = [A_cell - A_ci, A_cell + A_ci]
    return np.sqrt(A[0] / np.pi), np.sqrt(A[1] / np.pi)


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
    soma_length_range__μm : tuple[float, float]
        Min and max soma length (μm). By default, it is set to the estimated range for interneurons from Bui et al. 2003 [1]_.
    soma_diameter_range : tuple[float, float]
        Min and max soma diameter (μm). By default, it is set to the estimated range for interneurons from Bui et al. 2003 [1]_.
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
        soma_length_range__μm: tuple[
            float, float
        ] = _get_interneuron_diameter_range__μm(),
        soma_diameter_range: tuple[
            float, float
        ] = _get_interneuron_diameter_range__μm(),
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
        self.soma_length_range__μm = soma_length_range__μm
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

        sL = np.linspace(*soma_length_range__μm, n)
        sdiam = np.linspace(*soma_diameter_range, n)
        sg_pas = np.linspace(*passive_conductance_range, n)
        sgbar_na3rp = np.linspace(*na3rp_conductance_range, n)
        sgMax_kdrRL = np.linspace(*kdrrl_conductance_range, n)
        sgcamax_mAHP = np.linspace(*mahp_ca_conductance_range, n)
        sgkcamax_mAHP = np.linspace(*mahp_k_conductance_range, n)
        staur_mAHP = np.linspace(*mahp_tau_range, n)
        sghbar_gh = np.linspace(*gh_conductance_range, n)
        vcon = np.linspace(*axon_velocities, n)

        if cell_index is not None:
            init, end = cell_index, cell_index + 1
        else:
            init, end = 0, n

        _cells = []
        for (
            sL_i,
            sdiam_i,
            sg_pas_i,
            sgbar_na3rp_i,
            sgMax_kdrRL_i,
            sgcamax_mAHP_i,
            sgkcamax_mAHP_i,
            staur_mAHP_i,
            sghbar_gh_i,
            vcon_i,
        ) in zip(
            sL[init:end],
            sdiam[init:end],
            sg_pas[init:end],
            sgbar_na3rp[init:end],
            sgMax_kdrRL[init:end],
            sgcamax_mAHP[init:end],
            sgkcamax_mAHP[init:end],
            staur_mAHP[init:end],
            sghbar_gh[init:end],
            vcon[init:end],
        ):
            gII = cells.INgII()

            gII.soma.L = sL_i
            gII.soma.diam = sdiam_i
            gII.soma.g_pas = sg_pas_i
            gII.soma.gbar_na3rp = sgbar_na3rp_i
            gII.soma.gMax_kdrRL = sgMax_kdrRL_i
            gII.soma.gcamax_mAHP = sgcamax_mAHP_i
            gII.soma.gkcamax_mAHP = sgkcamax_mAHP_i
            gII.soma.taur_mAHP = staur_mAHP_i
            gII.soma.ghbar_gh = sghbar_gh_i

            gII.create_axon(length__m=axon_length, velcon__m_per_s=vcon_i)
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
        Min and max soma length (μm).
    soma_diameter_range : tuple[float, float]
        Min and max soma diameter (μm).
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
        soma_length_range: tuple[float, float] = _get_interneuron_diameter_range__μm(),
        soma_diameter_range: tuple[
            float, float
        ] = _get_interneuron_diameter_range__μm(),
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
        staur_mAHP = np.linspace(*mahp_tau_range, n)
        sghbar_gh = np.linspace(*gh_conductance_range, n)
        vcon = np.linspace(*axon_velocities, n)

        if cell_index is not None:
            init, end = cell_index, cell_index + 1
        else:
            init, end = 0, n

        _cells = []
        for (
            sL_i,
            sdiam_i,
            sg_pas_i,
            sgbar_na3rp_i,
            sgMax_kdrRL_i,
            sgcamax_mAHP_i,
            sgkcamax_mAHP_i,
            staur_mAHP_i,
            sghbar_gh_i,
            vcon_i,
        ) in zip(
            sL[init:end],
            sdiam[init:end],
            sg_pas[init:end],
            sgbar_na3rp[init:end],
            sgMax_kdrRL[init:end],
            sgcamax_mAHP[init:end],
            sgkcamax_mAHP[init:end],
            staur_mAHP[init:end],
            sghbar_gh[init:end],
            vcon[init:end],
        ):
            gIb = cells.INgIb()

            gIb.soma.L = sL_i
            gIb.soma.diam = sdiam_i
            gIb.soma.g_pas = sg_pas_i
            gIb.soma.gbar_na3rp = sgbar_na3rp_i
            gIb.soma.gMax_kdrRL = sgMax_kdrRL_i
            gIb.soma.gcamax_mAHP = sgcamax_mAHP_i
            gIb.soma.gkcamax_mAHP = sgkcamax_mAHP_i
            gIb.soma.taur_mAHP = staur_mAHP_i
            gIb.soma.ghbar_gh = sghbar_gh_i

            gIb.create_axon(length__m=axon_length, velcon__m_per_s=vcon_i)

            _cells.append(gIb)

        super().__init__(cells=_cells, initial_voltage__mV=initial_voltage__mV)


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
        n: int,
        model: str = "Powers2017",
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

        if model == "ModALS":
            _cells = self._create_modals_cells()
        elif model == "Powers2017":
            _cells = self._create_powers2017_cells()
        else:
            raise ValueError("Could not find the specific model for alpha MNs.")

        super().__init__(cells=_cells, initial_voltage__mV=initial_voltage__mV)

    def _create_modals_cells(self) -> list:
        """Create motor neurons using the ModALS model."""
        interpF = _exp_interp
        numCells = self.n

        # Soma parameters (using expInterp with hardcoded values from original)
        Diam_soma = interpF(78, 113, numCells, curv=1.0 / 14)
        Gnabar = interpF(0.0325, 0.0775, numCells, curv=1 / 2.5)
        Gnapbar = interpF(0.00043, 0.00067, numCells, curv=1 / 2.1, negative=True)
        Gkfbar = interpF(0.0028, 0.0015, numCells, curv=1 / 25, negative=True)
        Gksbar = interpF(0.02, 0.016, numCells, curv=1.0 / 6, negative=True)
        Mact = interpF(13, 20, numCells, curv=1 / 3)
        Rinact = interpF(0.018, 0.063, numCells, curv=1 / 4)
        Gls = interpF(1.0 / 1050, 1.0 / 650, numCells, curv=1 / 2.5)

        # Dendrite parameters
        Diam_dend = interpF(48.0, 90.0, numCells, curv=1.0 / 5)
        L_dend = interpF(5500, 10600, numCells, curv=1.0 / 12)
        GcaLbar = interpF(1.25e-5, 6.2e-6, numCells, curv=1 / 3, negative=True)
        Vtraub_caL = interpF(35, 34, numCells, curv=1 / 30, negative=True)
        LTAU_caL = interpF(90, 47, numCells, curv=1 / 3, negative=True)
        Gl_caL = interpF(1.0 / 13000, 1.0 / 6050, numCells, curv=1 / 2.5)

        vcon = np.linspace(self.axon_velocities[0], self.axon_velocities[1], self.n)

        # Determine cell creation range
        if self.cell_index is not None:
            init, end = self.cell_index, self.cell_index + 1
        else:
            init, end = 0, self.n

        _cells = []
        for i in range(init, end):
            cell = cells.AlphaMN(
                nseg=1, mode=self.mode, n_dend=1, model=self.model, rid=self.cell_index
            )
            cell.create_axon(length__m=self.axon_length, velcon__m_per_s=vcon[i])

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
        interpF = lambda x, y, z: _exp_interp(first=x, last=y, n=self.n, curv=z)

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
                nseg=1, mode=self.mode, n_dend=4, model=self.model, rid=self.cell_index
            )

            # Set soma parameters
            cell.soma.L = sL[i]
            cell.soma.diam = sdiam[i]
            cell.create_axon(length__m=self.axon_length, velcon__m_per_s=vcon[i])
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


if __name__ == "__main__":
    from myogen import setup_myogen

    setup_myogen()

    dt = 0.05

    dd__pool = DescendingDrive__Pool(
        n=10, poisson_random_process_order=16, timestep__ms=dt
    )

    alphaMN__pool = AlphaMN__Pool(
        n=10,
        model="Powers2017",
        mode="active",
        axon_velocities=(44, 53),
        axon_length=0.9,  # cm
        gamma=1.0,
        # Soma parameters
        soma_length_range=(2952, 3665, 0.3),
        soma_diameter_range=(22, 30, 0.3),
        soma_capacitance_range=(1.35546, 1.87853, 0.3),
        soma_passive_conductance_range=(8.11e-5, 3.77e-4, 0.3),
        soma_passive_reversal_range=(-71, -72, 0.3),
        soma_na3rp_conductance_range=(0.01, 0.022, 0.3),
        soma_naps_conductance_range=(2.6e-5, 2e-5, 0.3),
        soma_kdrrl_conductance_range=(0.015, 0.02, 0.3),
        soma_mahp_ca_conductance_range=(6.4e-6, 1.015e-5, 0.075),
        soma_mahp_k_conductance_range=(4.5e-4, 6e-4, 0.3),
        soma_mahp_tau_range=(90, 30, 0.3),
        soma_gh_conductance_range=(3e-5, 2.3e-4, 0.3),
        # Dendrite parameters
        dendrite_length_range=(1794.13, 2226.91, 0.3),
        dendrite_diameter_range=(8.73071, 11.9055, 0.3),
        dendrite_passive_conductance_range=(7.93e-5, 1.75e-4, 0.3),
        dendrite_passive_reversal_range=(-71, -72, 0.3),
        dendrite_resistance_range=(51.038, 40.755, 0.3),
        dendrite_capacitance_range=(0.867781, 0.880407, 0.3),
        dendrite_gh_conductance_range=(3e-5, 2.3e-4, 0.3),
        # Ca channel parameters - 4 dendrites
        dendrite_ca_conductance_ranges=(
            (8.5e-5, 1.18e-4, 0.3),
            (9.5e-5, 1.28e-4, 0.3),
            (1e-4, 1.38e-4, 0.3),
            (1.15e-4, 1.53e-4, 0.3),
        ),
        dendrite_ca_theta_m_range=(-42, -39, 0.3),
        dendrite_ca_theta_h_range=(10, -10, 0.3),
    )

    ia_pool = AffIa__Pool(
        n=5,
        poisson_random_process_order=25,
        recruitment_thresholds=(0, 150),
        axon_velocities=(62, 67),
        axon_length__m=1.0,  # cm
        timestep__ms=dt,
    )

    ii_pool = AffII__Pool(
        n=5,
        poisson_random_process_order=25,
        recruitment_thresholds=(0, 50),
        axon_velocities=(30, 35),
        axon_length__m=1.0,  # cm
        timestep__ms=dt,
    )

    popD = {
        "dd": dd__pool,
        "aMN": alphaMN__pool,
        "Ia": ia_pool,
        "II": ii_pool,
    }

    for x, y in zip(popD.keys(), popD.values()):
        print("Type: {}, List: {}".format(x, y))
        for cell in y:
            print(
                "Cell: {}, Global ID: {}, Class ID: {}".format(
                    cell.__class__.__name__, cell.global_ID, cell.class_ID
                )
            )
