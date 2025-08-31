import pyNN.neuron as sim
from beartype.typing import Sequence

from myogen.utils.decorators import beartowertype
from myogen.utils.types import CURRENT__AnalogSignal


@beartowertype
def inject_currents_into_populations(
    populations: Sequence[sim.Population],
    input_current__AnalogSignal: CURRENT__AnalogSignal,
):
    """
    Injects input currents into the specified populations.

    Parameters
    ----------
    populations : list[Population]
        The populations of neurons to inject current into.
    input_current__AnalogSignal : CURRENT__AnalogSignal
        The analog signal of input currents to inject into the population.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the number of populations does not match the number of input current matrices.
    """
    # For NEO AnalogSignal, shape is (time_points, n_channels)
    if len(populations) != (n_channels := input_current__AnalogSignal.shape[1]):
        raise ValueError(
            f"Number of populations ({len(populations)}) does not match number of input current matrices ({n_channels})."
        )

    times = input_current__AnalogSignal.times.magnitude

    for i, population in enumerate(populations):
        sim.StepCurrentSource(
            times=times,
            amplitudes=input_current__AnalogSignal[:, i].magnitude.squeeze(),
        ).inject_into(population)
