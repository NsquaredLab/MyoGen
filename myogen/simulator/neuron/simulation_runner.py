from itertools import count
from typing import Any, Callable, Optional, Union

import numpy as np
import quantities as pq
from neo import AnalogSignal, Block, Segment, SpikeTrain
from tqdm import tqdm

from myogen.simulator.neuron.network import Network

from neuron import h

from myogen.utils.decorators import beartowertype
from myogen.utils.types import Quantity__m_per_s, Quantity__ms


@beartowertype
class SimulationRunner:
    """
    Manages NEURON simulation execution with automated setup, initialization,
    and result collection for neuromuscular simulations.

    Provides a clean interface for running complex neuromuscular simulations
    while maintaining full user control over populations, connections, and
    step-by-step simulation logic. Automatically handles NEURON environment
    setup, voltage initialization, and structured result collection.

    Separates simulation control from plotting and analysis concerns.
    """

    # Smart defaults for common MyoGen model output attributes
    _DEFAULT_MODEL_OUTPUTS = {
        "HillModel": [
            "muscle_length",
            "muscle_force",
            "muscle_torque",
            "type1_activation",
            "type2_activation",
        ],
        "SpindleModel": [
            "primary_afferent_firing__Hz",
            "secondary_afferent_firing__Hz",
            "bag1_activation",
            "bag2_activation",
            "chain_activation",
            "intrafusal_tensions",
        ],
        "GolgiTendonOrganModel": ["ib_afferent_firing__Hz"],
    }

    def __init__(
        self,
        network: Network,
        models: dict[str, Any],
        step_callback: Callable[[Any], Any],
        model_outputs: Optional[dict[str, Union[list[str], None]]] = None,
        temperature__celsius: float = 36.0,
    ):
        """
        Initialize SimulationRunner with network, models, and step callback.

        Parameters
        ----------
        network : Network
            Configured Network instance with populations and connections.
        models : Dict[str, Any]
            Physiological models (e.g., {"hill": hill_model, "spin": spindle_model}).
        step_callback : Callable
            User-defined function called at each simulation timestep.
        model_outputs : Optional[Dict[str, Union[List[str], None]]], optional
            Explicit model output attributes to collect. None uses smart defaults.
            Format: {"model_name": ["attr1", "attr2"]} or {"model_name": None}
            for defaults, by default None.
        temperature__celsius : float, optional
            NEURON simulation temperature, by default 36.0.
        """
        # Store immutable parameters following project pattern
        self.network = network
        self.populations = network.populations  # Expose populations from network
        self.models = models
        self.step_callback = step_callback
        self.model_outputs = model_outputs
        self.temperature__celsius = temperature__celsius

        # Private working copies
        self._network = network
        self._populations = network.populations  # Get populations from network
        self._models = models
        self._step_callback = step_callback
        self._model_outputs = self._resolve_model_outputs()
        self._temperature__celsius = temperature__celsius

        # Runtime state
        self._trace_vectors: dict[str, dict[int, Any]] = {}
        self._step_counter = None
        self._progress_bar = None
        self._total_steps = None

        # Setup internal spike recording vectors
        self._spike_recording = self._setup_spike_recording()

    def _resolve_model_outputs(self) -> dict[str, list[str]]:
        """
        Resolve model output attributes using smart defaults and user overrides.

        Returns
        -------
        Dict[str, List[str]]
            Final mapping of model names to output attribute lists.
        """
        resolved = {}

        for model_name, model_instance in self._models.items():
            model_class_name = model_instance.__class__.__name__

            # Check for user override
            if self.model_outputs and model_name in self.model_outputs:
                user_specified = self.model_outputs[model_name]
                if user_specified is None:
                    # Use defaults for this model
                    resolved[model_name] = self._DEFAULT_MODEL_OUTPUTS.get(model_class_name, [])
                else:
                    # Use explicit user specification
                    resolved[model_name] = user_specified
            else:
                # Use smart defaults based on model class
                resolved[model_name] = self._DEFAULT_MODEL_OUTPUTS.get(model_class_name, [])

        return resolved

    def _setup_spike_recording(self) -> dict[str, Any]:
        """
        Create NEURON spike recording vectors for all populations.

        Returns
        -------
        dict[str, Any]
            Dictionary containing 'idvec' and 'spkvec' with NEURON Vectors for each population.
        """
        from neuron import h

        idvec = {}
        spkvec = {}

        for pop_name in self._populations.keys():
            idvec[pop_name] = h.Vector()
            spkvec[pop_name] = h.Vector()

        return {"idvec": idvec, "spkvec": spkvec}

    def _setup_network_spike_recording(self) -> None:
        """
        Configure the network with spike recording vectors and activate recording.
        """
        # Set spike recording on network
        self._network.spike_recording = self._spike_recording

        # Setup spike recording (calls NEURON setup)
        self._network.setup_spike_recording()

    def run(
        self,
        duration__ms: Quantity__ms,
        timestep__ms: Quantity__ms,
        membrane_recording: Optional[dict[str, list[int]]] = None,
    ) -> Block:
        """
        Execute NEURON simulation with automated setup and result collection.

        Parameters
        ----------
        duration__ms : Quantity__ms
            Total simulation duration in milliseconds.
        timestep__ms : Quantity__ms
            Integration timestep in milliseconds.
        membrane_recording : Optional[Dict[str, List[int]]], optional
            Populations and cell indices for membrane potential recording.
            Format: {"population_name": [cell_id1, cell_id2, ...]}, by default None.

        Returns
        -------
        Block
            Structured simulation results containing:
            - spikes: Spike timing and ID data for all populations
            - membrane: Membrane potential traces (if requested)
            - models: Output data from all physiological models
            - simulation: Time vector and simulation metadata

        Raises
        ------
        ValueError
            If model output attributes don't exist on model instances.
        RuntimeError
            If NEURON simulation fails to complete.
        """
        try:
            # Setup NEURON environment
            self._setup_neuron_environment(duration__ms, timestep__ms)

            # Setup optional membrane recording
            if membrane_recording:
                self._setup_membrane_recording(membrane_recording)

            # Initialize population voltages
            self._initialize_voltages()

            # Register step callback for closed-loop dynamics
            self._register_step_callback()

            # Validate model outputs before simulation
            self._validate_model_outputs()

            # Setup spike recording on network
            self._setup_network_spike_recording()

            h.run()

            # Close progress bar (with error handling)
            if self._progress_bar is not None:
                try:
                    self._progress_bar.close()
                except (TypeError, AttributeError):
                    # Ignore progress bar closing errors
                    pass

            print("Simulation completed")

            # Collect and structure results
            results = self._collect_results(duration__ms, timestep__ms)

            return results

        except Exception as e:
            # Close progress bar in case of error (with error handling)
            if self._progress_bar is not None:
                try:
                    self._progress_bar.close()
                except (TypeError, AttributeError):
                    # Ignore progress bar closing errors
                    pass
            raise RuntimeError(f"Simulation failed: {str(e)}") from e

    def _setup_neuron_environment(
        self, duration__ms: Quantity__ms, timestep__ms: Quantity__ms
    ) -> None:
        """Configure NEURON global simulation parameters."""
        h.load_file("stdrun.hoc")
        h.celsius = self._temperature__celsius
        h.tstop = duration__ms
        h.dt = timestep__ms

        # Calculate total steps for progress bar
        self._total_steps = int(duration__ms / timestep__ms)

        # Store timestep for progress bar updates
        self._timestep__ms = timestep__ms

        # Initialize progress bar
        self._progress_bar = tqdm(
            total=duration__ms.magnitude,
            desc="Simulation Progress",
            unit="ms",
        )

        # Reset step counter for step callback
        self._step_counter = count(0)

    def _setup_membrane_recording(self, membrane_recording: dict[str, list[int]]) -> None:
        """Setup membrane potential recording vectors for specified populations."""
        self._trace_vectors = {}

        for pop_name, cell_indices in membrane_recording.items():
            if pop_name not in self._populations:
                raise ValueError(f"Population '{pop_name}' not found in populations")

            pop_traces = {}
            population = self._populations[pop_name]

            for cell_idx in cell_indices:
                if cell_idx >= len(population):
                    raise ValueError(
                        f"Cell index {cell_idx} out of range for population "
                        f"'{pop_name}' (size: {len(population)})"
                    )

                vector = h.Vector()
                vector.record(population[cell_idx].soma(0.5)._ref_v)
                pop_traces[cell_idx] = vector

            self._trace_vectors[pop_name] = pop_traces

    def _initialize_voltages(self) -> None:
        """Automatically collect and set initial voltages for all populations."""
        sections = []
        voltages = []

        for population in self._populations.values():
            try:
                sec_list, v_hold = population.get_initialization_data()
                sections.extend(sec_list)
                voltages.extend(v_hold)
            except (AttributeError, TypeError):
                # Skip populations without initialization data
                continue

        if sections:

            def set_initial_voltages():
                for sec, voltage in zip(sections, voltages):
                    sec.v = voltage

            h.FInitializeHandler(0, set_initial_voltages)

    def _register_step_callback(self) -> None:
        """Register user's step callback with NEURON's integration system."""

        duration__ms = h.tstop  # Use NEURON's tstop as simulation duration

        # Track last progress bar update time to avoid multiple updates per timestep
        last_progress_time = -1

        # Create wrapper that provides step counter access to callback and updates progress
        def step_wrapper():
            nonlocal last_progress_time

            # Check if we've exceeded simulation time - don't process if so
            if h.t >= h.tstop:
                return

            # Update progress bar based on actual simulation time progression (not callback count)
            if self._progress_bar is not None:
                # Only update progress bar when simulation time has actually advanced
                if h.t > last_progress_time:
                    time_advance = h.t - max(0, last_progress_time)
                    try:
                        self._progress_bar.update(time_advance)
                        last_progress_time = h.t
                    except (TypeError, AttributeError) as e:
                        # Disable progress bar when it fails
                        print(f"Progress bar error (disabling): {e}")
                        self._progress_bar = None
                        return

            # Call user's step callback only if we haven't exceeded time limit
            if h.t < h.tstop:
                return self._step_callback(self._step_counter)

        h.CVode().extra_scatter_gather(0, step_wrapper)

    def _validate_model_outputs(self) -> None:
        """Validate that all specified model output attributes exist."""
        for model_name, output_attrs in self._model_outputs.items():
            if model_name not in self._models:
                raise ValueError(f"Model '{model_name}' not found in models")

            model_instance = self._models[model_name]

            for attr_name in output_attrs:
                if not hasattr(model_instance, attr_name):
                    raise ValueError(
                        f"Model '{model_name}' ({model_instance.__class__.__name__}) "
                        f"does not have attribute '{attr_name}'"
                    )

    def _collect_results(self, duration__ms: Quantity__ms, timestep__ms: Quantity__ms) -> Block:
        """
        Collect simulation results from network, models, and recordings.

        Returns structured results compatible with existing analysis code.
        """
        block = Block()

        for pop_name in self._populations.keys():
            segment = Segment(name=pop_name)

            if self._spike_recording and pop_name in self._spike_recording.get("spkvec", {}):
                spike_times = self._spike_recording["spkvec"][pop_name].as_numpy()
                spike_ids = self._spike_recording["idvec"][pop_name].as_numpy()

                for spike_id in sorted(np.unique(spike_ids)):
                    times_for_id = spike_times[spike_ids == spike_id]
                    if len(times_for_id) > 0:
                        segment.spiketrains.append(
                            SpikeTrain(
                                name=str(int(spike_id)),
                                times=(times_for_id * pq.ms).rescale(pq.s),
                                t_start=0.0 * pq.s,
                                t_stop=duration__ms.rescale(pq.s),
                                sampling_rate=(1.0 / timestep__ms.rescale(pq.s)).rescale(pq.Hz),
                            )
                        )

            for cell_idx, vector in self._trace_vectors.get(pop_name, {}).items():
                segment.analogsignals.append(
                    AnalogSignal(
                        name=str(cell_idx),
                        sampling_period=timestep__ms.rescale(pq.s),
                        signal=vector * pq.mV,
                    )
                )

            block.segments.append(segment)

        for model_name, output_attrs in self._model_outputs.items():
            segment = Segment(name=model_name)

            model_instance = self._models[model_name]
            for attr_name in output_attrs:
                attr_value = getattr(model_instance, attr_name)

                if hasattr(attr_value, "__iter__"):
                    segment.analogsignals.append(
                        AnalogSignal(
                            name=attr_name,
                            sampling_period=timestep__ms.rescale(pq.s),
                            signal=attr_value * pq.dimensionless,
                        )
                    )
                elif isinstance(attr_value, (int, float, str)):
                    segment.annotations[attr_name] = attr_value

            block.segments.append(segment)

        block.annotations["time__ms"] = duration__ms
        block.annotations["timestep__ms"] = timestep__ms
        block.annotations["temperature__celsius"] = self._temperature__celsius
        block.annotations["active_MNs"] = np.unique(spike_ids).astype(int)

        return block

    def get_model_outputs(self, model_name: str) -> list[str]:
        """
        Get the list of output attributes that will be collected for a model.

        Parameters
        ----------
        model_name : str
            Name of the model as specified in the models dictionary.

        Returns
        -------
        List[str]
            List of attribute names that will be collected from this model.
        """
        return self._model_outputs.get(model_name, [])

    def set_model_outputs(self, model_name: str, output_attrs: list[str]) -> None:
        """
        Override the output attributes for a specific model.

        Parameters
        ----------
        model_name : str
            Name of the model as specified in the models dictionary.
        output_attrs : List[str]
            List of attribute names to collect from this model.

        Raises
        ------
        ValueError
            If model_name is not found in the models dictionary.
        """
        if model_name not in self._models:
            raise ValueError(f"Model '{model_name}' not found in models")

        self._model_outputs[model_name] = output_attrs
