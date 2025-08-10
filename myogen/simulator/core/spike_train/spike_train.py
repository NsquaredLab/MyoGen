from typing import Any, Literal

import neo
import numpy as np
from tqdm import tqdm
import pyNN.neuron as sim
from pyNN.neuron.morphology import uniform, centre
from pyNN.parameters import IonicSpecies
from pyNN.random import NumpyRNG

from myogen import RANDOM_GENERATOR
from myogen.simulator.core.spike_train import functions, classes
from myogen.utils.types import (
    SPIKE_TRAIN__MATRIX,
    INPUT_CURRENT__MATRIX,
    CORTICAL_INPUT__MATRIX,
    beartowertype,
)


@beartowertype
class MotorNeuronPool:
    """
    Motor neuron pool with specified parameters for EMG simulation.

    This class creates and manages a pool of motor neurons with physiologically
    realistic parameters. It supports spike train generation through NEURON
    backend integration and provides both current injection and cortical input
    stimulation modes.

    Parameters
    ----------
    recruitment_thresholds : np.ndarray
        Array of recruitment thresholds for the motor neurons (normalized 0-1)
    diameter_soma_min__mm : float, default=77.5
        Minimum diameter of the soma in millimeters
    diameter_soma_max__mm : float, default=82.5
        Maximum diameter of the soma in millimeters
    y_min__mm : float, default=18.0
        Minimum y coordinate of the soma in millimeters
    y_max__mm : float, default=36.0
        Maximum y coordinate of the soma in millimeters
    diameter_dend_min__mm : float, default=41.5
        Minimum diameter of the dendrite in millimeters
    diameter_dend_max__mm : float, default=62.5
        Maximum diameter of the dendrite in millimeters
    x_min__mm : float, default=-5500
        Minimum x coordinate of the dendrite in millimeters
    x_max__mm : float, default=-6789
        Maximum x coordinate of the dendrite in millimeters
    vt_min__mV : float, default=12.35
        Minimum voltage threshold of the neuron in millivolts
    vt_max__mV : float, default=20.9
        Maximum voltage threshold of the neuron in millivolts
    kf_cond_min__S_per_cm2 : float, default=4
        Minimum conductance density of the potassium fast channel in S/cm²
    kf_cond_max__S_per_cm2 : float, default=0.5
        Maximum conductance density of the potassium fast channel in S/cm²
    coefficient_of_variation : float, default=0.01
        Coefficient of variation for parameter noise (0-1)

    Attributes
    ----------
    spike_trains : SPIKE_TRAIN__MATRIX
        Matrix of spike trains - available after generate_spike_trains()
    active_neuron_indices : list[np.ndarray]
        Indices of active neurons per pool - available after generate_spike_trains()
    data : list[neo.Segment]
        Neo segments with recorded data - available after generate_spike_trains()
    mvc_current_threshold : float
        Minimum current for maximum voluntary contraction - computed on access

    Examples
    --------
    >>> pool = MotorNeuronPool(recruitment_thresholds=np.linspace(0, 1, 100))
    >>> current_matrix = np.random.randn(5, 1000) * 50
    >>> spike_trains, indices, data = pool.generate_spike_trains(current_matrix)
    >>> print(f"Generated spike trains shape: {pool.spike_trains.shape}")
    """

    def __init__(
        self,
        recruitment_thresholds: np.ndarray,
        diameter_soma_min__mm: float = 77.5,
        diameter_soma_max__mm: float = 82.5,
        y_min__mm: float = 18.0,
        y_max__mm: float = 36.0,
        diameter_dend_min__mm: float = 41.5,
        diameter_dend_max__mm: float = 62.5,
        x_min__mm: float = -5500,
        x_max__mm: float = -6789,
        vt_min__mV: float = 12.35,
        vt_max__mV: float = 20.9,
        kf_cond_min__S_per_cm2: float = 4,
        kf_cond_max__S_per_cm2: float = 0.5,
        coefficient_of_variation: float = 0.01,
    ):
        # Store immutable public parameters
        self.recruitment_thresholds = recruitment_thresholds
        self.diameter_soma_min__mm = diameter_soma_min__mm
        self.diameter_soma_max__mm = diameter_soma_max__mm
        self.y_min__mm = y_min__mm
        self.y_max__mm = y_max__mm
        self.diameter_dend_min__mm = diameter_dend_min__mm
        self.diameter_dend_max__mm = diameter_dend_max__mm
        self.x_min__mm = x_min__mm
        self.x_max__mm = x_max__mm
        self.vt_min__mV = vt_min__mV
        self.vt_max__mV = vt_max__mV
        self.kf_cond_min__S_per_cm2 = kf_cond_min__S_per_cm2
        self.kf_cond_max__S_per_cm2 = kf_cond_max__S_per_cm2
        self.coefficient_of_variation = coefficient_of_variation

        # Store private copies for internal modifications
        self._recruitment_thresholds = recruitment_thresholds.copy()
        self._diameter_soma_min__mm = diameter_soma_min__mm
        self._diameter_soma_max__mm = diameter_soma_max__mm
        self._y_min__mm = y_min__mm
        self._y_max__mm = y_max__mm
        self._diameter_dend_min__mm = diameter_dend_min__mm
        self._diameter_dend_max__mm = diameter_dend_max__mm
        self._x_min__mm = x_min__mm
        self._x_max__mm = x_max__mm
        self._vt_min__mV = vt_min__mV
        self._vt_max__mV = vt_max__mV
        self._kf_cond_min__S_per_cm2 = kf_cond_min__S_per_cm2
        self._kf_cond_max__S_per_cm2 = kf_cond_max__S_per_cm2
        self._coefficient_of_variation = coefficient_of_variation

    def _create_motor_neuron_pool(self):
        """Create a pool of motor neurons with specified parameters

        Returns
        -------
        cell_type: classes.cell_class
            Cell type of the motor neuron pool
        """
        rng = RANDOM_GENERATOR

        n_neurons = len(self._recruitment_thresholds)

        somas = functions.create_somas(
            recruitment_thresholds=self._recruitment_thresholds,
            diameter_min=self._diameter_soma_min__mm,
            diameter_max=self._diameter_soma_max__mm,
            y_min=self._y_min__mm,
            y_max=self._y_max__mm,
            CV=self._coefficient_of_variation,
        )
        dends = functions.create_dends(
            recruitment_thresholds=self._recruitment_thresholds,
            somas=somas,
            diameter_min=self._diameter_dend_min__mm,
            diameter_max=self._diameter_dend_max__mm,
            y_min=self._y_min__mm,
            y_max=self._y_max__mm,
            x_min=self._x_min__mm,
            x_max=self._x_max__mm,
            CV=self._coefficient_of_variation,
        )

        # scale recruitment thresholds to v_min and v_max
        vt = (
            -70
            + (
                self._vt_min__mV
                + (self._vt_max__mV - self._vt_min__mV) * self._recruitment_thresholds
            )
            + rng.normal(size=n_neurons) * self._coefficient_of_variation
        )

        return classes.cell_class(
            morphology=functions.soma_dend(somas, dends),
            cm=1,  # mF / cm**2
            Ra=0.070,  # ohm.mm
            ionic_species={
                "na": IonicSpecies("na", reversal_potential=50),
                "ks": IonicSpecies("ks", reversal_potential=-80),
                "kf": IonicSpecies("kf", reversal_potential=-80),
            },
            pas_soma={
                "conductance_density": functions.create_cond(
                    n_neurons,
                    7e-4,
                    7e-4,
                    "soma",
                    CV=10 * self._coefficient_of_variation,
                ),
                "e_rev": -70,
            },  #
            pas_dend={
                "conductance_density": functions.create_cond(
                    n_neurons,
                    7e-4,
                    7e-4,
                    "dendrite",
                    CV=10 * self._coefficient_of_variation,
                ),
                "e_rev": -70,
            },  #
            na={"conductance_density": uniform("soma", 30), "vt": list(vt)},
            kf={
                "conductance_density": functions.create_cond(
                    n_neurons,
                    self._kf_cond_min__S_per_cm2,
                    self._kf_cond_max__S_per_cm2,
                    "soma",
                    CV=self._coefficient_of_variation,
                ),
                "vt": list(vt),
            },
            ks={"conductance_density": uniform("soma", 0.1), "vt": list(vt)},
            syn={"locations": centre("dendrite"), "e_syn": 0, "tau_syn": 0.6},
        )

    def generate_spike_trains(
        self,
        input_current__matrix: INPUT_CURRENT__MATRIX | None = None,
        cortical_input__matrix: CORTICAL_INPUT__MATRIX | None = None,
        timestep__ms: float = 0.05,
        noise_mean__nA: float = 30,  # noqa N803
        noise_stdev__nA: float = 30,  # noqa N803
        CST_number: int = 400,
        connection_prob: float = 0.3,
        what_to_record: list[
            dict[Literal["variables", "to_file", "sampling_interval", "locations"], Any]
        ] = [
            {"variables": ["v"], "locations": ["dendrite", "soma"]},
        ],
    ) -> tuple[SPIKE_TRAIN__MATRIX, list[np.ndarray], list[neo.Segment]]:
        """
        Generate the spike trains for as many neuron pools as input currents there are

        Each motor neuron pools have each "neurons_per_pool" neurons.
        The input currents are injected into each pool, and the spike trains are recorded.

        Parameters
        ----------
        input_current__matrix : INPUT_CURRENT__MATRIX, optional
            Matrix of shape (n_pools, t_points) containing current values
            Each row represents the current for one pool
        cortical_input__matrix : CORTICAL_INPUT__MATRIX, optional
            Matrix of shape (n_pools, t_points) containing cortical input values
            Each row represents the cortical input for one pool
        timestep__ms : float
            Simulation timestep__ms in ms
        noise_mean__nA : float
            Mean of the noise current in nA
        noise_stdev__nA : float
            Standard deviation of the noise current in nA
        CST_number : int
            Number of neurons in the cortical input population. Only used if cortical_input__matrix is provided. Default is 400.
        connection_prob : float
            Probability of a connection between a cortical input neuron and a motor neuron. Only used if cortical_input__matrix is provided. Default is 0.3.
        what_to_record: WhatToRecord
            List of dictionaries specifying what to record.

            Each dictionary contains the following keys:
                - variables: list of strings specifying the variables to record
                - to_file: bool specifying whether to save the recorded data to a file
                - sampling_interval: int specifying the sampling interval in ms
                - locations: list of strings specifying the locations to record from

            See pyNN documentation for more details: https://pynn.readthedocs.io/en/stable/recording.html.

            Spike trains are recorded by default.

        Returns
        -------
        spike_trains : SPIKE_TRAIN__MATRIX
            Matrix of shape (n_pools, neurons_per_pool, t_points) containing spike trains
            Each row represents the spike train for one pool
            Each column represents the spike train for one neuron
            Each element represents whether the neuron spiked at that time point
        active_neuron_indices : list[np.ndarray]
            List of arrays of indices of the active neurons in each pool
        data : list[neo.core.segment.Segment]
            List of neo segments containing the recorded data
        """
        self.timestep__ms = timestep__ms
        sim.setup(timestep=self.timestep__ms)

        if input_current__matrix is not None:
            n_pools = input_current__matrix.shape[0]
            simulation_time_points = input_current__matrix.shape[-1]
        elif cortical_input__matrix is not None:
            n_pools = cortical_input__matrix.shape[0]
            simulation_time_points = cortical_input__matrix.shape[-1]
        else:
            raise ValueError(
                "Either 'input_current__matrix' or 'cortical_input__matrix' must be provided."
            )

        # Create motor neuron pools
        pools: list[sim.Population] = []
        for pool_idx in tqdm(
            range(n_pools), desc="Creating motor neuron pools", unit="pool"
        ):
            pools.append(
                sim.Population(
                    len(self._recruitment_thresholds),
                    self._create_motor_neuron_pool(),
                    initial_values={"v": -70},
                )
            )

        # Inject currents into each pool - one current per pool
        if input_current__matrix is not None:
            self.times = np.arange(input_current__matrix.shape[-1]) * self.timestep__ms
            for input_current, pool in tqdm(
                zip(input_current__matrix, pools),
                total=len(pools),
                desc="Injecting currents into pools",
                unit="pool",
            ):
                current_source = sim.StepCurrentSource(
                    times=self.times, amplitudes=input_current
                )
                current_source.inject_into(pool, location="soma")

                # Add Gaussian noise current to each neuron in the pool
                for neuron in tqdm(
                    pool,
                    desc=f"Adding noise to neurons in pool",
                    unit="neuron",
                    leave=False,
                ):
                    noise_source = sim.NoisyCurrentSource(
                        mean=noise_mean__nA,
                        stdev=noise_stdev__nA,
                        start=0,
                        stop=self.times[-1] + self.timestep__ms,
                        dt=self.timestep__ms,
                    )
                    noise_source.inject_into([neuron], location="soma")

        callbacks = []
        projections = []
        # Create cortical input
        if cortical_input__matrix is not None:
            for FR, pool in zip(cortical_input__matrix, pools):
                spike_source = sim.Population(
                    CST_number, classes.SpikeSourceGammaStart(alpha=1)
                )
                synapse = sim.StaticSynapse(weight=0.6, delay=0.2)
                projection = sim.Projection(
                    spike_source,
                    pool,
                    sim.FixedProbabilityConnector(
                        connection_prob, location_selector="dendrite", rng=NumpyRNG()
                    ),
                    synapse,
                    receptor_type="syn",
                )
                projections.append(projection)
                callbacks.append(
                    classes.SetRateFromArray(
                        population_source=spike_source,
                        population_neuron=pool,
                        firing_rate=FR,
                        timestep__ms=self.timestep__ms,
                    )
                )

        # Set up recording
        for pool in pools:
            pool.record("spikes")
            for record in what_to_record:
                pool.record(**record)

        # Run simulation
        sim.run(simulation_time_points * self.timestep__ms, callbacks=callbacks)
        sim.end()

        # Store results privately
        self._data = [pool.get_data().segments[0] for pool in pools]

        # Convert spike times to binary arrays and save
        self._time_indices = np.arange(simulation_time_points)
        self._spike_trains = np.array(
            [
                [
                    np.isin(
                        self._time_indices, np.array(st / self.timestep__ms).astype(int)
                    )
                    for st in d.spiketrains
                ]
                for d in self._data
            ]
        )

        self._active_neuron_indices = [
            np.argwhere(x)[:, 0] for x in np.sum(self._spike_trains, axis=-1) != 0
        ]

        return self._spike_trains, self._active_neuron_indices, self._data

    @property
    def spike_trains(self) -> SPIKE_TRAIN__MATRIX:
        """
        Matrix of spike trains for each pool and neuron.

        Returns
        -------
        SPIKE_TRAIN__MATRIX
            Matrix of shape (n_pools, neurons_per_pool, t_points) containing spike trains

        Raises
        ------
        AttributeError
            If spike trains have not been computed. Run generate_spike_trains() first.
        """
        if not hasattr(self, "_spike_trains"):
            raise AttributeError(
                "Spike trains not computed. Run generate_spike_trains() first."
            )
        return self._spike_trains

    @property
    def active_neuron_indices(self) -> list[np.ndarray]:
        """
        List of arrays containing indices of active neurons in each pool.

        Returns
        -------
        list[np.ndarray]
            List of arrays of indices of the active neurons in each pool

        Raises
        ------
        AttributeError
            If active neuron indices have not been computed. Run generate_spike_trains() first.
        """
        if not hasattr(self, "_active_neuron_indices"):
            raise AttributeError(
                "Active neuron indices not computed. Run generate_spike_trains() first."
            )
        return self._active_neuron_indices

    @property
    def data(self) -> list[neo.Segment]:
        """
        List of neo segments containing recorded data from simulation.

        Returns
        -------
        list[neo.core.segment.Segment]
            List of neo segments containing the recorded data

        Raises
        ------
        AttributeError
            If data has not been computed. Run generate_spike_trains() first.
        """
        if not hasattr(self, "_data"):
            raise AttributeError(
                "Simulation data not computed. Run generate_spike_trains() first."
            )
        return self._data

    def compute_mvc_current_threshold(self) -> float:
        """
        Computes the minimum current threshold for maximum voluntary contraction
        using binary search optimization.

        Returns
        -------
        float
            Minimum current threshold in nA needed to activate all neurons
        """

        def test_current_activates_all_neurons(current_nA: float) -> bool:
            """Test if a given current activates all neurons in the pool.

            Parameters
            ----------
            current_nA : float
                Current amplitude in nA

            Returns
            -------
            bool
                True if all neurons are activated, False otherwise
            """
            # Create short test simulation parameters
            test_duration_ms = 500  # Short 500ms test
            test_timestep_ms = 0.5
            n_timepoints = int(test_duration_ms / test_timestep_ms)

            # Create constant current input for testing
            test_current = np.full((1, n_timepoints), current_nA)

            # Run short simulation
            spike_trains, active_neuron_indices, _ = self.generate_spike_trains(
                input_current__matrix=test_current,
                timestep__ms=test_timestep_ms,
                noise_mean__nA=0,  # Reduce noise for more consistent results
                noise_stdev__nA=0,
                what_to_record=[],  # Minimal recording for speed
            )

            # Check if all neurons are active
            n_total_neurons = len(self.recruitment_thresholds)
            n_active_neurons = len(active_neuron_indices[0])

            return n_active_neurons == n_total_neurons

        # Binary search for minimum current
        low_current = 0.0
        high_current = 1000.0  # Start with reasonable upper bound
        tolerance = 1.0  # 1 nA tolerance

        # First, find an upper bound that works
        while not test_current_activates_all_neurons(high_current):
            high_current *= 2
            if high_current > 10000:  # Safety limit
                raise ValueError(
                    "Could not find current that activates all neurons within reasonable range"
                )

        # Binary search between low and high
        while high_current - low_current > tolerance:
            mid_current = (low_current + high_current) / 2

            if test_current_activates_all_neurons(mid_current):
                high_current = mid_current
            else:
                low_current = mid_current

        return high_current

    @property
    def mvc_current_threshold(self) -> float:
        """
        Property that returns the minimum current threshold for maximum voluntary contraction.

        Returns
        -------
        float
            Minimum current threshold in nA needed to activate all neurons
        """
        return self.compute_mvc_current_threshold()
