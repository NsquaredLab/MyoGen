"""
Jaxley Backend for MyoGen Neural Simulations.

This package provides JAX-accelerated implementations of neural models
for motor control simulations, as an alternative to the NEURON backend.

Submodules:
- channels: Ion channel mechanisms (Na, K, Ca, etc.)
- synapses: Synaptic mechanisms (Exp2Syn, NMDA)
- cells: Biophysical cell builders
- populations: Population-level containers
- network: Network connectivity
- simulation_engine: Core simulation infrastructure
- integrate: Core integration functions for Jaxley simulation
"""

# Import cell classes (API-compatible with NEURON)
from myogen.simulator.jaxley.cells_api import (
    AlphaMN,
    INgII,
    INgIb,
    DD,
    DD_Gamma,
    AffIa,
    AffII,
    AffIb,
    _Cell,
)

# Import biophysical builders
from myogen.simulator.jaxley.cells.biophysical import (
    BiophysicalMotorNeuron,
    BiophysicalInterneuron,
    create_motor_neuron,
    create_motor_neuron_builder,
    create_interneuron,
    create_interneuron_builder,
)

# Import channels
from myogen.simulator.jaxley.channels import (
    Na3rp,
    Naps,
    KdrRL,
    MAHP,
    Gh,
    LCaInact,
    LeakChannel,
    get_motor_neuron_channels_soma,
    get_motor_neuron_channels_dendrite,
)

# Import synapses
from myogen.simulator.jaxley.synapses import (
    Exp2Syn,
    ExcitatorySynapse,
    InhibitorySynapse,
    NMDASynapse,
)

# Import simulation engine
from myogen.simulator.jaxley.simulation_engine import (
    JaxleyNetworkSimulator,
    JaxleyCellSimulator,
    SimulationConfig,
    CurrentInjection,
    SpikeRecord,
    compute_firing_rate,
    compute_isi_statistics,
)

# Import integration module
from myogen.simulator.jaxley.integrate import (
    step_current,
    ramp_current,
    sinusoidal_current,
    detect_spikes,
    compute_firing_rate_from_spikes,
    compute_isi_cv,
    simulate_cell,
    simulate_population,
    compute_fi_curve,
    CellSimulationResult,
    PopulationSimulationResult,
    NetworkSimulator,
    SynapticConnection,
)

# Import network
from myogen.simulator.jaxley.network import (
    Network,
    JaxleyConnection,
)

# Differentiable closed-loop entry point + gradient helpers
from myogen.simulator.jaxley.closed_loop import (
    ClosedLoopConfig,
    run_jax,
    compile_run,
    value_and_grad_run,
    partition_differentiable,
)

# Differentiable EMG synthesis
from myogen.simulator.jaxley.emg import (
    surface_emg_jax,
    intramuscular_emg_jax,
    resample_muaps,
)

# Spike-mode primitive + differentiable Bessel functions
from myogen.simulator.jaxley.jax_models import (
    spike_detect,
    differentiable_twitch_params,
)
from myogen.simulator.jaxley.bessel import (
    iv_int,
    kv_int,
)

# Import JIT-compiled and batched simulation utilities
from myogen.simulator.jaxley.jit_simulation import (
    create_jitted_simulator,
    create_parameter_sweep_simulator,
    create_batched_simulator,
    simulate_population_batched,
    BatchedPopulationResult,
    SynapticEvent,
    SynapseStateManager,
    JITNetworkSimulator,
    run_fi_curve_batched,
    warmup_jit,
)

__all__ = [
    # Cells (API-compatible)
    "AlphaMN",
    "INgII",
    "INgIb",
    "DD",
    "DD_Gamma",
    "AffIa",
    "AffII",
    "AffIb",
    "_Cell",
    # Biophysical builders
    "BiophysicalMotorNeuron",
    "BiophysicalInterneuron",
    "create_motor_neuron",
    "create_motor_neuron_builder",
    "create_interneuron",
    "create_interneuron_builder",
    # Channels
    "Na3rp",
    "Naps",
    "KdrRL",
    "MAHP",
    "Gh",
    "LCaInact",
    "LeakChannel",
    "get_motor_neuron_channels_soma",
    "get_motor_neuron_channels_dendrite",
    # Synapses
    "Exp2Syn",
    "ExcitatorySynapse",
    "InhibitorySynapse",
    "NMDASynapse",
    # Simulation
    "JaxleyNetworkSimulator",
    "JaxleyCellSimulator",
    "SimulationConfig",
    "CurrentInjection",
    "SpikeRecord",
    "compute_firing_rate",
    "compute_isi_statistics",
    # Integration
    "step_current",
    "ramp_current",
    "sinusoidal_current",
    "detect_spikes",
    "compute_firing_rate_from_spikes",
    "compute_isi_cv",
    "simulate_cell",
    "simulate_population",
    "compute_fi_curve",
    "CellSimulationResult",
    "PopulationSimulationResult",
    "NetworkSimulator",
    "SynapticConnection",
    # JIT and Batched Simulation
    "create_jitted_simulator",
    "create_parameter_sweep_simulator",
    "create_batched_simulator",
    "simulate_population_batched",
    "BatchedPopulationResult",
    "SynapticEvent",
    "SynapseStateManager",
    "JITNetworkSimulator",
    "run_fi_curve_batched",
    "warmup_jit",
    # Network
    "Network",
    "JaxleyConnection",
    # Differentiable pipeline (closed loop, EMG, Bessel)
    "ClosedLoopConfig",
    "run_jax",
    "compile_run",
    "value_and_grad_run",
    "partition_differentiable",
    "surface_emg_jax",
    "intramuscular_emg_jax",
    "resample_muaps",
    "spike_detect",
    "differentiable_twitch_params",
    "iv_int",
    "kv_int",
]
