# Jaxley Backend Status and NEURON Replacement Roadmap

Date: 2026-05-01

This document explains the current state of the Jaxley backend in MyoGen. It is meant for someone reading the codebase who needs to understand what has already been implemented, how the current Jaxley code is intended to be used, and what remains before Jaxley can fully replace the NEURON backend.

The short version: MyoGen now has a substantial Jaxley backend that can run many of the same workflows as the NEURON backend, but it is not yet a complete drop-in replacement. The current Jaxley implementation is best understood as a working parallel backend with several approximations around synaptic transmission, network execution, and closed-loop simulation.

## Current Status At A Glance

Jaxley is currently useful for:

- running single-cell motor-neuron simulations
- extracting electrophysiological parameters
- testing current-injection workflows
- generating open-loop spike-train examples
- driving downstream muscle/proprioception/force code from generated spike trains
- producing Neo-compatible outputs for existing analysis code

Jaxley is not yet ready to fully replace NEURON for:

- live event-based synaptic networks
- exact `NetCon`/synapse semantics
- quantitatively matched motor-neuron firing behavior across all examples
- validated closed-loop spinal simulations
- transparent backend switching through one common public API

In practice, the Jaxley backend should currently be treated as an active replacement effort rather than a completed replacement.

## Where the Jaxley Backend Lives

The main backend code is under:

```text
myogen/simulator/jaxley/
```

Important modules:

```text
myogen/simulator/jaxley/cells_api.py
myogen/simulator/jaxley/populations/
myogen/simulator/jaxley/channels/
myogen/simulator/jaxley/synapses/
myogen/simulator/jaxley/network.py
myogen/simulator/jaxley/integrate.py
myogen/simulator/jaxley/simulation_runner.py
myogen/simulator/jaxley/muscle.py
myogen/simulator/jaxley/proprioception/
myogen/simulator/jaxley/joint_dynamics.py
```

Jaxley-specific examples live alongside the NEURON examples:

```text
examples/01_basic/*_jaxley.py
```

These examples are useful for understanding the current usage pattern and the places where Jaxley behavior still differs from NEURON.

Files with `_jaxley.py` in their name should be read as Jaxley-specific implementations or experiments. They are not guaranteed to be numerically equivalent to the corresponding NEURON example.

## What Has Already Been Implemented

### Backend Package Structure

A parallel `myogen.simulator.jaxley` backend exists. It mirrors much of the NEURON package structure:

- cells
- populations
- networks
- current injection utilities
- muscle dynamics
- proprioception models
- joint dynamics
- simulation/result collection helpers

The Jaxley backend generally tries to keep the same conceptual API as the NEURON backend, but code still usually switches backend by changing import paths rather than by selecting a backend through a single shared interface.

### Motor Neuron Pool

`AlphaMN__Pool` exists for Jaxley and is the main motor-neuron population used in the Jaxley examples.

When `use_jaxley_mech=True`, the current implementation creates a Powers2017-like motor neuron with:

- soma branch
- one dendritic cable with multiple compartments
- scaled dendritic area intended to approximate the four dendrites used in the NEURON model
- recruitment-threshold-based parameter interpolation
- size-dependent conductances and morphology

Implemented channel set:

- soma: `Na3rp`, `Naps`, `KdrRL`, `MAHP`, `Gh`, leak
- dendrite: `LCaInact`, `Gh`, leak

The intent is to preserve the Henneman size principle: lower-threshold, smaller motor neurons should recruit before larger, higher-threshold motor neurons.

### Custom Jaxley Channels

Motor-neuron channel implementations exist in:

```text
myogen/simulator/jaxley/channels/motor_neuron_channels.py
```

The backend includes custom versions of the motor-neuron channels used by the NEURON mechanisms. Earlier channel-solver mistakes were fixed by using the correct exponential-Euler style gate update for steady-state/time-constant dynamics.

This was an important milestone because the custom channels are the path toward closer NEURON matching. However, matching the exact NEURON dynamics remains an ongoing task because the solvers, topology, and numerical behavior are different.

### Descending Drive and Afferent Pools

Jaxley versions of descending-drive and afferent populations exist:

- `DescendingDrive__Pool`
- `DescendingDrive_Gamma__Pool`
- `AffIa__Pool`
- `AffII__Pool`
- `AffIb__Pool`

The Jaxley backend includes Poisson and gamma spike generators for open-loop drive generation. These are used to generate spike trains from time-varying drive signals in the examples.

### Interneurons

Jaxley versions of spinal interneuron populations exist:

- `GII__Pool`
- `GIb__Pool`

They provide API-compatible population objects for spinal-network experiments, but full validation against the NEURON interneuron behavior is still needed.

### Muscle, Proprioception, and Joint Dynamics

Jaxley-side implementations exist for the non-neural components:

- Hill muscle model
- spindle model
- Golgi tendon organ model
- joint dynamics

Some of these are backend-neutral or NumPy/JAX-oriented rather than direct Jaxley cell simulations. They are already useful for examples and pipeline work, but they still need systematic validation in closed-loop network contexts.

### Example Workflows

Jaxley versions exist for several basic workflows:

- current-injection spike trains
- descending-drive spike trains
- force simulation
- neuron parameter extraction
- spinal network simulation

These examples demonstrate that the Jaxley backend can produce Neo-style outputs and drive downstream analysis/plotting. They are also the main place where backend differences become visible.

### Neo Result Compatibility

The Jaxley examples generally convert outputs to Neo structures:

- `Block`
- `Segment`
- `SpikeTrain`
- `AnalogSignal`

This is important because downstream analysis and plotting can often remain the same even when the simulator backend changes.

## How the Current Jaxley Backend Works

The current Jaxley backend has two broad simulation styles.

### 1. Direct Jaxley Cell Integration

For single cells or independent cells, examples use Jaxley integration directly or through helper functions. This is the cleanest current path for:

- current injection
- voltage traces
- spike detection
- parameter extraction
- F-I curve work

This is the most mature part of the Jaxley backend.

### 2. Staged or Approximate Network Simulation

For larger network workflows, the current implementation often separates the problem into stages:

1. Generate presynaptic spike trains.
2. Convert those spikes into conductance or current waveforms.
3. Inject the resulting current into Jaxley motor neurons.
4. Post-process voltages into spike trains.
5. Use spike trains to drive muscle/proprioception/force models.

This is practical and often fast, but it is not the same as a fully live NEURON network with event-delivered synapses.

For example, the current Jaxley descending-drive workflow uses an analytical conductance update in a JAX scan and injects current into motor neurons using a fixed driving force. That makes the workflow controllable and fast, but it is an approximation: the postsynaptic current is not produced by a live synapse object using the instantaneous postsynaptic voltage.

## Important Current Limitations

### Jaxley Network Objects Are Not Yet NEURON NetCons

`myogen/simulator/jaxley/network.py` provides NetCon-like connection metadata and an API that resembles the NEURON backend. However, these connection objects generally describe connectivity; they do not automatically create a full live Jaxley computation graph equivalent to NEURON `NetCon` plus `Exp2Syn`.

This is one of the central limitations for replacing NEURON.

### Synaptic Event Semantics Are Not Fully Matched

NEURON uses event-driven `NetCon` objects and synapse mechanisms such as `Exp2Syn`. These have specific semantics:

- event delivery
- delay
- thresholding
- weights
- rise and decay time constants
- reversal potentials
- target section/location

The Jaxley backend has synapse-related code and experiments, but the stable examples still often use analytical conductance/current approximations rather than live event-based synapses.

To replace NEURON, Jaxley needs a robust and validated way to reproduce these synaptic semantics.

### Motor Neuron Dynamics Are Close but Not Equivalent

The Jaxley motor neuron model has made major progress:

- it can recruit in plausible order
- it has active soma and dendrite channels
- it includes persistent inward current mechanisms
- it supports electrophysiology extraction workflows

But it is not yet quantitatively equivalent to NEURON. The largest open issue is matching firing-rate behavior:

- F-I gain
- plateau firing rates
- AHP behavior
- dendritic PIC effects
- discharge-rate variability
- force consequences of spike-train differences

Tuning one example can also affect others, so motor-neuron tuning needs to be handled as a backend-wide calibration problem, not as a one-example fix.

### Background Variability Is Still Crude

NEURON contains support for dendritic fluctuating conductance noise through `Gfluctdv`, although not every example uses it directly. The Jaxley experiments so far have mostly used direct current noise or private synthetic inputs when variability was needed.

For physiological replacement, Jaxley should move toward conductance-based background variability:

- private excitatory conductance fluctuations
- private inhibitory conductance fluctuations
- private Poisson background synapses
- optional synaptic delay jitter

Direct OU current noise can be useful for experiments, but it is not a complete replacement for synaptic background conductance.

### Closed-Loop Simulation Is Not Fully Unified

The Jaxley `SimulationRunner` currently coordinates non-neural components and result collection, but neural biophysics is often integrated before the runner loop in the examples.

A full NEURON replacement needs a clearer story for closed-loop simulation:

- neural integration
- synaptic event delivery
- proprioceptive feedback
- interneuron dynamics
- muscle dynamics
- joint dynamics
- recordings

These pieces either need to run inside one coherent JAX/Jaxley loop or be staged in a way that is documented and validated against NEURON.

### Backend Switching Is Not Yet Transparent

The codebase has parallel imports:

```python
from myogen.simulator.neuron...
from myogen.simulator.jaxley...
```

This works for examples, but it is not a true backend abstraction. A user still needs to know which backend-specific classes to import. A full replacement should eventually allow backend selection through configuration or factory functions.

## What Still Needs To Be Done

### 1. Define Backend Equivalence Targets

Before replacing NEURON, the project needs explicit acceptance criteria.

Useful targets include:

- recruitment order
- active motor-neuron count
- total spike count
- plateau firing-rate distribution
- ISI CV distribution
- F-I gain
- rheobase
- input resistance
- AP/AHP shape
- force output from the same spike trains
- runtime and memory behavior

Without these targets, visual plot matching can be misleading.

### 2. Validate Single-Cell Electrophysiology

The motor neuron model should be validated before large network behavior is tuned.

Needed work:

- compare Jaxley and NEURON parameter extraction under matched protocols
- tune F-I curves and rheobase distributions
- verify channel state initialization
- verify AHP and PIC behavior
- confirm that tuning does not break current-injection examples

### 3. Implement Validated Live Synaptic Transmission

A real replacement needs a Jaxley equivalent of the NEURON event/synapse path.

Needed work:

- reproduce `Exp2Syn`-like rise/decay behavior
- represent event delays
- support excitatory and inhibitory reversal potentials
- support probabilistic and deterministic connectivity
- support one-to-one connectivity
- support spike recording and event recording
- validate against small NEURON test networks before using large examples

### 4. Add Physiological Background Variability

The Jaxley backend needs a cleaner background-input model than direct injected current noise.

Possible approaches:

- implement a Jaxley equivalent of `Gfluctdv`
- add private excitatory/inhibitory Poisson conductance inputs
- add delay jitter to shared synaptic inputs
- expose noise parameters in a controlled, documented way

### 5. Build Regression/Comparison Tests

The backend needs tests that compare output summaries against saved NEURON references.

Recommended comparison examples:

- Example 2: current-injection spike trains
- Example 3: descending-drive spike trains
- Example 8: force simulation
- Example 10: neuron parameter extraction
- Example 11: spinal network simulation

Tests should compare numeric summaries first, then plots second.

### 6. Create a Backend Selection Layer

Long term, users should not need separate imports everywhere.

Possible future API:

```python
from myogen.simulator import get_backend

sim = get_backend("jaxley")
pool = sim.populations.AlphaMN__Pool(...)
```

or a project-level configuration option. The exact design is open, but the goal is to make Jaxley/NEURON switching intentional and discoverable.

### 7. Clarify Which Jaxley Paths Are Stable

Some Jaxley code is stable enough for examples; some code is experimental. This should be documented directly in module docstrings or developer notes.

Current stable/useful paths:

- current injection into Jaxley motor neurons
- parameter extraction
- analytical conductance/current workflows
- Neo output conversion
- non-neural model stepping

Experimental or incomplete paths:

- full live Jaxley synaptic networks as NEURON replacements
- closed-loop spinal networks with all neural and physiological models unified
- exact NEURON synapse semantics
- backend-transparent API

## Practical Guidance For Readers

If you are reading the current code:

- Treat the Jaxley backend as a working parallel backend, not a finished NEURON replacement.
- Use the Jaxley examples to understand current workflows.
- Do not assume Jaxley and NEURON examples are quantitatively equivalent just because they share names.
- Expect some examples to use staged approximations rather than live network simulation.
- Validate changes with numeric metrics, not only raster plots.
- Tune motor-neuron intrinsic properties separately from network synaptic parameters.
- Keep NEURON as the reference backend until Jaxley has systematic validation.

## Summary

The Jaxley backend has already implemented much of the required structure for replacing NEURON: motor neurons, active channels, populations, spike generators, examples, muscle and proprioception models, and Neo-compatible outputs.

The remaining work is mostly about equivalence and integration:

- live event-based synaptic transmission
- quantitative motor-neuron calibration
- physiological background variability
- closed-loop network orchestration
- backend-level API parity
- systematic validation against NEURON references

Jaxley is therefore a strong foundation, but not yet a complete NEURON replacement. The current code should be read as a serious in-progress backend with stable example paths and several known approximation layers.
