"""
Base classes and utility functions for neuron population management - Jaxley Backend

This module provides the foundational `_Pool` class and helper functions used
by all neuron population classes in the MyoGen Jaxley simulator.

Naming Convention
-----------------
MyoGen uses a double-underscore (`__`) separator in neuron pool class names to improve
readability and maintain consistency. The pattern is:

    {NeuronType}__{Container}

Examples:
    - ``AlphaMN__Pool`` - Pool of alpha motor neurons
    - ``AffIa__Pool`` - Pool of type Ia afferent neurons
    - ``GII__Pool`` - Pool of group II interneurons
    - ``DescendingDrive__Pool`` - Pool of descending drive neurons

The double underscore clearly separates the neuron type from the container class,
making the code more readable than alternatives like ``AlphaMNPool`` or ``AlphaMN_Pool``.
This convention is used throughout the neuron populations module.

Note: The base class uses a single underscore prefix (``_Pool``) following Python's
convention for internal/private classes not intended for direct instantiation by users.
"""

from typing import Optional, Union

import numpy as np
from scipy.optimize import curve_fit

from myogen.utils.decorators import beartowertype


def _exp_crescent(x, a, b, c):
    return a * np.exp(b * x) + c


def _exp_decrescent(x, a, b, c):
    return a * np.exp(-b * x) + c


@beartowertype
def _exp_interp(first: float, last: float, n: int, curv: float = 0.33, negative: bool = False):
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


def _get_interneuron_diameter_range__um() -> tuple[float, float]:
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
class _Pool:
    """
    Base class for neuron cell populations - Jaxley Backend.

    Provides common functionality for managing groups of neurons. Unlike the NEURON
    version, Jaxley doesn't use physical sections, so voltage initialization is
    handled through Jaxley's state management.

    Parameters
    ----------
    cells : list
        List of neuron cells in the population.
    initial_voltage__mV : Union[float, list[float]], optional
        Initial membrane voltage(s) in millivolts. Can be a single value applied
        to all cells or a list of per-cell values. Used for Jaxley state initialization.
        If None (default), no voltage initialization is performed.
    spike_threshold__mV : float, optional
        Spike detection threshold in millivolts for recording spikes from
        this population. Motor neurons typically need higher thresholds
        (e.g., 50.0 mV) while interneurons use lower thresholds (-10.0 mV).
        By default -10.0.

    Notes
    -----
    Populations with dummy cells (e.g., DescendingDrive, AffIa, AffIb) should
    not provide initial_voltage__mV as they have no real cellular dynamics to
    initialize. Populations with real Jaxley cells (e.g., AlphaMN, AffII, GII, GIb)
    should provide appropriate voltage values.

    Different neuron types have different action potential amplitudes:
    - Motor neurons: typically reach 80-100 mV, need threshold ~50 mV
    - Interneurons: typically reach 30-50 mV, can use default -10 mV
    """

    def __init__(
        self,
        cells: list,
        initial_voltage__mV: Optional[Union[float, list[float]]] = None,
        spike_threshold__mV: float = -10.0,
    ):
        self._cells = cells
        self.spike_threshold__mV = spike_threshold__mV

        # Handle initial voltage - only if provided
        if initial_voltage__mV is not None:
            n_cells = len(cells)
            if isinstance(initial_voltage__mV, (int, float)):
                self.initial_voltage_values__mV = [initial_voltage__mV] * n_cells
            else:
                assert len(initial_voltage__mV) == n_cells, (
                    f"initial_voltage__mV list length ({len(initial_voltage__mV)}) "
                    f"must match number of cells ({n_cells})"
                )
                self.initial_voltage_values__mV = list(initial_voltage__mV)
            # Actually apply the voltage to each underlying jx.Cell, so code that
            # bypasses get_initialization_data() (e.g. user reaches in via
            # ``cell_wrapper.cell`` and calls ``jx.integrate`` directly) still sees
            # cells initialised at the correct resting potential. Without this,
            # cells stay at Jaxley's default v = -70 mV regardless of what the
            # pool was told to use, which silently breaks NERLab cells (V_rest ≈ 0)
            # and any other model whose intended rest is not -70.
            self._apply_initial_voltages()
        else:
            self.initial_voltage_values__mV = None

    def _apply_initial_voltages(self) -> None:
        """Set v on each underlying ``jx.Cell`` and re-initialise channel states.

        Iterates the wrappers in ``self._cells``; each is expected to have a
        ``.cell`` attribute pointing at the underlying ``jx.Cell`` (true for
        AlphaMN, interneuron wrappers, etc.). Wrappers without a ``.cell``
        attribute (e.g. dummy / spike-generator populations) are skipped.
        """
        for cw, v_init in zip(self._cells, self.initial_voltage_values__mV):
            jx_cell = getattr(cw, "cell", None)
            if jx_cell is None:
                continue
            try:
                jx_cell.set("v", float(v_init))
                jx_cell.init_states()
            except Exception:
                # Some wrappers might not yet have their cell fully constructed
                # at this point; failing silently is preferable to crashing the
                # whole pool build. Code reaching in later can still init by hand.
                pass

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
        Return cells and their initial voltages for Jaxley simulation setup.

        For Jaxley backend, this returns cells that need voltage initialization.
        Unlike NEURON, we don't work with section objects, but with Jaxley cells.

        Returns
        -------
        tuple[list, list]
            First list contains Jaxley cell objects.
            Second list contains corresponding initial voltages in mV.
            Both lists will be empty if population has no voltage initialization.
        """
        # Return empty lists if no voltage initialization needed
        if self.initial_voltage_values__mV is None:
            return [], []

        cells_to_init = []
        voltages = []

        for cell_idx, cell in enumerate(self._cells):
            cell_voltage = self.initial_voltage_values__mV[cell_idx]

            # Skip dummy cells (they don't have Jaxley cell objects)
            if not hasattr(cell, "cell") or cell.cell is None:
                continue

            cells_to_init.append(cell)
            voltages.append(cell_voltage)

        return cells_to_init, voltages
