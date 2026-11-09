"""
Simulation Runner - Jaxley Backend

Coordinates real-time stepping of non-Jaxley components (muscle, proprioception,
interneurons) in neuromuscular simulations.

Architectural note — split between SimulationRunner and jx.integrate():
    This class does NOT call ``jx.integrate()`` on neural cells. Neural cell
    integration happens *before* the SimulationRunner loop in each example:
    spike trains are computed by running ``jx.integrate(cell)`` per motor neuron,
    and the resulting spike times are used to drive the muscle/proprioception models
    that SimulationRunner steps in real time. This split is intentional — it keeps
    neural biophysics (GPU-accelerated, full ODE integration) separate from the
    real-time physiological models (CPU, time-stepped).
"""

from itertools import count
from typing import Any, Callable, Optional, Union

import numpy as np
import quantities as pq
from neo import AnalogSignal, Block, Segment, SpikeTrain
from tqdm import tqdm

from myogen.simulator.jaxley.network import Network
from myogen.utils.decorators import beartowertype
from myogen.utils.types import Quantity__ms


@beartowertype
class SimulationRunner:
    """
    Manages Jaxley simulation execution with automated setup, initialization,
    and result collection for neuromuscular simulations.

    Provides a clean interface for running complex neuromuscular simulations
    while maintaining full user control over populations, connections, and
    step-by-step simulation logic.

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
            Simulation temperature, by default 36.0.
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
        self._populations = network.populations
        self._models = models
        self._step_callback = step_callback
        self._model_outputs = self._resolve_model_outputs()
        self._temperature__celsius = temperature__celsius

        # Runtime state
        self._trace_vectors: dict[str, dict[int, list]] = {}
        self._step_counter = None
        self._progress_bar = None
        self._total_steps = None
        self._current_time = 0.0

        # Setup internal spike recording
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
        Create spike recording storage for all populations.

        Returns
        -------
        dict[str, Any]
            Dictionary containing 'idvec' and 'spkvec' with lists for each population.
        """
        idvec = {}
        spkvec = {}

        for pop_name in self._populations.keys():
            idvec[pop_name] = []
            spkvec[pop_name] = []

        return {"idvec": idvec, "spkvec": spkvec}

    def _setup_network_spike_recording(self) -> None:
        """
        Configure the network with spike recording vectors and activate recording.
        """
        # Set spike recording on network
        self._network.spike_recording = self._spike_recording

        # Setup spike recording
        self._network.setup_spike_recording()

    def run(
        self,
        duration__ms: Quantity__ms,
        timestep__ms: Quantity__ms,
        membrane_recording: Optional[dict[str, list[int]]] = None,
    ) -> Block:
        """
        Execute Jaxley simulation with automated setup and result collection.

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
            If simulation fails to complete.
        """
        try:
            # Setup simulation environment
            self._setup_simulation_environment(duration__ms, timestep__ms)

            # Setup optional membrane recording
            if membrane_recording:
                self._setup_membrane_recording(membrane_recording)

            # Initialize population voltages
            self._initialize_voltages()

            # Validate model outputs before simulation
            self._validate_model_outputs()

            # Setup spike recording on network
            self._setup_network_spike_recording()

            # Run simulation loop
            self._run_simulation_loop(duration__ms, timestep__ms)

            # Close progress bar
            if self._progress_bar is not None:
                try:
                    self._progress_bar.close()
                except (TypeError, AttributeError):
                    pass

            print("Simulation completed")

            # Collect and structure results
            results = self._collect_results(duration__ms, timestep__ms)

            return results

        except Exception as e:
            # Close progress bar in case of error
            if self._progress_bar is not None:
                try:
                    self._progress_bar.close()
                except (TypeError, AttributeError):
                    pass
            raise RuntimeError(f"Simulation failed: {str(e)}") from e

    def _setup_simulation_environment(
        self, duration__ms: Quantity__ms, timestep__ms: Quantity__ms
    ) -> None:
        """Configure simulation parameters."""
        self._duration__ms = duration__ms
        self._timestep__ms = timestep__ms
        self._current_time = 0.0

        # Calculate total steps for progress bar
        self._total_steps = int(duration__ms.magnitude / timestep__ms.magnitude)

        # Initialize progress bar
        self._progress_bar = tqdm(
            total=duration__ms.magnitude,
            desc="Simulation Progress",
            unit="ms",
        )

        # Reset step counter for step callback
        self._step_counter = count(0)

    def _setup_membrane_recording(self, membrane_recording: dict[str, list[int]]) -> None:
        """Setup membrane potential recording storage for specified populations."""
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

                # Initialize storage list for this cell's membrane potential
                pop_traces[cell_idx] = []

            self._trace_vectors[pop_name] = pop_traces

    def _initialize_voltages(self) -> None:
        """Automatically collect and set initial voltages for all populations."""
        # For Jaxley backend, voltage initialization is handled differently
        # Store initialization data for later use
        self._initialization_data = {}

        for pop_name, population in self._populations.items():
            try:
                cells_to_init, voltages = population.get_initialization_data()
                if cells_to_init:
                    self._initialization_data[pop_name] = {
                        "cells": cells_to_init,
                        "voltages": voltages,
                    }
            except (AttributeError, TypeError):
                # Skip populations without initialization data
                continue

    def _run_simulation_loop(self, duration__ms: Quantity__ms, timestep__ms: Quantity__ms) -> None:
        """
        Execute the main simulation time-stepping loop.
        
        For Jaxley backend, this manually steps through time and calls the user's
        step callback at each timestep.
        """
        dt = timestep__ms.magnitude
        t_stop = duration__ms.magnitude

        while self._current_time < t_stop:
            # Call user's step callback
            try:
                self._step_callback(self._step_counter)
            except StopIteration:
                break

            # Record membrane potentials if requested
            if self._trace_vectors:
                self._record_membrane_potentials()

            # Update simulation time
            self._current_time += dt

            # Update progress bar
            if self._progress_bar is not None:
                try:
                    self._progress_bar.update(dt)
                except (TypeError, AttributeError):
                    self._progress_bar = None

    def _record_membrane_potentials(self) -> None:
        """Record membrane potentials for cells with active recording."""
        for pop_name, cell_traces in self._trace_vectors.items():
            population = self._populations[pop_name]
            for cell_idx, trace_list in cell_traces.items():
                cell = population[cell_idx]
                # Membrane potential comes from cell.v if available.
                # Note: this path is only reached if membrane_recording is passed to run(),
                # which no current example does. Neural voltage traces are obtained from
                # jx.integrate() output, not from stepping through SimulationRunner.
                if hasattr(cell, 'v'):
                    trace_list.append(cell.v)

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

        # Collect spike data for each population
        for pop_name in self._populations.keys():
            segment = Segment(name=pop_name)

            if self._spike_recording and pop_name in self._spike_recording.get("spkvec", {}):
                spike_times = np.array(self._spike_recording["spkvec"][pop_name])
                spike_ids = np.array(self._spike_recording["idvec"][pop_name])

                if len(spike_times) > 0:
                    for spike_id in sorted(np.unique(spike_ids)):
                        times_for_id = spike_times[spike_ids == spike_id]
                        if len(times_for_id) > 0:
                            segment.spiketrains.append(
                                SpikeTrain(
                                    name=f"{pop_name}_cell{int(spike_id)}_spikes",
                                    times=(times_for_id * pq.ms).rescale(pq.s),
                                    t_start=0.0 * pq.s,
                                    t_stop=duration__ms.rescale(pq.s),
                                    sampling_rate=(1.0 / timestep__ms.rescale(pq.s)).rescale(pq.Hz),
                                    cell_idx=int(spike_id),
                                )
                            )

            # Collect membrane potential traces
            for cell_idx, trace_data in self._trace_vectors.get(pop_name, {}).items():
                if trace_data:
                    segment.analogsignals.append(
                        AnalogSignal(
                            name=f"{pop_name}_cell{cell_idx}_Vm",
                            sampling_period=timestep__ms.rescale(pq.s),
                            signal=np.array(trace_data) * pq.mV,
                            cell_idx=cell_idx,
                        )
                    )

            block.segments.append(segment)

        # Collect model outputs
        for model_name, output_attrs in self._model_outputs.items():
            segment = Segment(name=model_name)

            model_instance = self._models[model_name]
            for attr_name in output_attrs:
                attr_value = getattr(model_instance, attr_name)

                if hasattr(attr_value, "__iter__") and not isinstance(attr_value, str):
                    segment.analogsignals.append(
                        AnalogSignal(
                            name=f"{model_name}_{attr_name}",
                            sampling_period=timestep__ms.rescale(pq.s),
                            signal=np.asarray(attr_value) * pq.dimensionless,
                            attr_name=attr_name,
                        )
                    )
                elif isinstance(attr_value, (int, float, str)):
                    segment.annotations[attr_name] = attr_value

            block.segments.append(segment)

        # Add metadata
        block.annotations["time__ms"] = duration__ms
        block.annotations["timestep__ms"] = timestep__ms
        block.annotations["temperature__celsius"] = self._temperature__celsius
        
        # Extract active motor neurons if spike data exists
        all_spike_ids = []
        for pop_name in self._populations.keys():
            if pop_name in self._spike_recording.get("idvec", {}):
                spike_ids = self._spike_recording["idvec"][pop_name]
                if spike_ids:
                    all_spike_ids.extend(spike_ids)
        
        if all_spike_ids:
            block.annotations["active_MNs"] = np.unique(all_spike_ids).astype(int)

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
