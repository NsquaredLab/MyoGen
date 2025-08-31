import numpy as np
import pyNN.neuron as sim
from beartype.typing import Sequence

from myogen.utils.decorators import beartowertype


@beartowertype
def compute_maximum_activation_current_thresholds(
    populations: Sequence[sim.Population],
    test_duration__ms: float = 500.0,
    test_timestep__ms: float = 0.5,
    tolerance__nA: float = 1.0,
    initial_high_current__nA: float = 1000.0,
    max_current__nA: float = 10000.0,
) -> list[float]:
    """
    Computes the minimum current thresholds for maximum activation
    using binary search optimization for multiple pyNN populations.

    This function performs batch processing to find the minimum current amplitude
    required to activate all neurons in each motor neuron population. It uses short
    test simulations to determine activation status for each population.

    Parameters
    ----------
    populations : Sequence[sim.Population]
        List of pyNN populations of motor neurons to test.
    test_duration__ms : float, default=500.0
        Duration of test simulation in milliseconds. Default is 500.0 ms.
    test_timestep__ms : float, default=0.5
        Timestep for test simulation in milliseconds. Default is 0.5 ms.
    tolerance__nA : float, default=1.0
        Tolerance for binary search convergence in nanoamperes. Default is 1.0 nA.
    initial_high_current__nA : float, default=1000.0
        Initial upper bound for binary search in nanoamperes. Default is 1000.0 nA.
    max_current__nA : float, default=10000.0
        Maximum current limit for safety in nanoamperes. Default is 10000.0 nA.

    Returns
    -------
    list[float]
        List of minimum current thresholds in nA needed to activate all neurons
        in each population, in the same order as the input populations

    Raises
    ------
    ValueError
        If no current within the reasonable range can activate all neurons

    Notes
    -----
    This function temporarily modifies the pyNN simulation state. It should be
    called when no other simulation is running. The function automatically
    handles simulation setup and cleanup.

    Examples
    --------
    >>> import pyNN.neuron as sim
    >>> from myogen.simulator.pyNN.neurons import MotorNeuron__Population
    >>> from myogen.utils.pyNN import compute_maximum_activation_current_thresholds
    >>>
    >>> # Create multiple motor neuron populations
    >>> thresholds1 = np.linspace(0.1, 1.0, 50)
    >>> thresholds2 = np.linspace(0.2, 0.8, 30)
    >>> populations = [
    ...     MotorNeuron__Population(thresholds1),
    ...     MotorNeuron__Population(thresholds2)
    ... ]
    >>>
    >>> # Compute maximum activation thresholds for all populations
    >>> thresholds = compute_maximum_activation_current_thresholds(populations)
    >>> for i, threshold in enumerate(thresholds):
    ...     print(f"Population {i+1} threshold: {threshold:.1f} nA")
    """

    def compute_single_population_threshold(population: sim.Population) -> float:
        """Compute threshold for a single population."""

        def test_current_activates_all_neurons(current__nA: float) -> bool:
            """Test if a given current activates all neurons in the population."""

            # Calculate number of time points
            n_timepoints = int(test_duration__ms / test_timestep__ms)

            # Create time array and current amplitude array
            times = np.arange(0, test_duration__ms, test_timestep__ms)
            amplitudes = np.full(n_timepoints, current__nA)

            # Setup simulation with test parameters
            sim.reset()
            sim.setup(timestep=test_timestep__ms)

            try:
                # Create current source and inject into population
                sim.StepCurrentSource(times=times, amplitudes=amplitudes).inject_into(
                    population
                )

                # Record spikes
                population.record("spikes")

                # Run simulation
                sim.run(test_duration__ms)

                # Check if the number of active neurons matches the population size
                return (
                    sum(
                        1
                        for spike_train in population.get_data().segments[0].spiketrains
                        if len(spike_train) > 0
                    )
                    == population.size
                )

            finally:
                # Clean up simulation
                sim.end()

        # Binary search for minimum current
        low_current = 0.0
        high_current = initial_high_current__nA

        # First, find an upper bound that works
        while not test_current_activates_all_neurons(high_current):
            high_current *= 2
            if high_current > max_current__nA:
                raise ValueError(
                    f"Could not find current that activates all neurons within reasonable range "
                    f"(max tested: {max_current__nA} nA)"
                )

        # Binary search between low and high
        while high_current - low_current > tolerance__nA:
            mid_current = (low_current + high_current) / 2

            if test_current_activates_all_neurons(mid_current):
                high_current = mid_current
            else:
                low_current = mid_current

        return high_current

    # Process each population and return list of thresholds
    thresholds = []
    for i, population in enumerate(populations):
        try:
            thresholds.append(compute_single_population_threshold(population))
        except ValueError as e:
            raise ValueError(f"Failed to compute threshold for population {i}: {e}")

    return thresholds
