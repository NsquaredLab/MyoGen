# MyoGen — Jaxley branch

> **This is the Jaxley branch of MyoGen.** It uses **Jaxley** (JAX) as the simulation
> backend and makes the full neuromuscular closed loop **differentiable end-to-end**.
> The original NEURON-based implementation lives on the upstream **`main` branch**; use
> that branch if you need NEURON as the primary simulator. This branch keeps NEURON only
> as an optional validation reference.

MyoGen is a modular neuromuscular simulation framework for generating physiologically
grounded motor-unit activity, muscle force, and EMG (surface and intramuscular).

On this branch, Jaxley is JAX-based, runs on CPU/GPU, and — the headline feature — lets
you take gradients of force, joint kinematics, and EMG with respect to neural, muscle,
and joint parameters. That is what the NEURON branch cannot do.

---

## Why Jaxley + differentiability

The whole simulation — spiking motor neurons → muscle force → spindle/GTO
proprioception → joint dynamics → EMG — runs as a single `jax.lax.scan` and is a pure
function of its parameters. That makes it possible to:

- **Fit a subject-specific digital twin**: gradient-descend neural/muscle parameters so
  simulated EMG or force matches a real recording.
- **Optimize** descending drive, recruitment, muscle mechanics, or reflex gains against
  any differentiable objective.
- **Accelerate** on GPU via JIT compilation.

Gradients flow through spikes via a selectable **surrogate-gradient** mode while the
forward pass stays bit-identical to the exact spiking model.

## Installation

```bash
pip install myogen                 # default: Jaxley backend only (no NEURON/MPI)
pip install "myogen[neuron]"       # add the optional NEURON validation backend
```

`import myogen` requires no NEURON runtime, compiler, or MPI.

## Quick start — a differentiable simulation

```python
import jax, jax.numpy as jnp
from myogen.simulator.jaxley import (
    run_jax, value_and_grad_run, compile_run, surface_emg_jax,
)

# Build a ClosedLoopConfig + params PyTree for your network (see
# examples/01_basic/11_simulate_spinal_network.py for a full setup), then:

# 1) Forward simulation (hard spikes = scientific default)
outputs, meta = run_jax(params, config, inputs, jax.random.PRNGKey(0), spike_mode="hard")
force, angle, spikes = outputs["force"], outputs["angle_deg"], outputs["mn_spikes"]

# 2) Differentiate a loss w.r.t. all float parameters (surrogate spikes)
def loss(out):
    return jnp.mean(out["force"] ** 2)

value, grads = value_and_grad_run(loss, params, config, inputs,
                                  jax.random.PRNGKey(0), spike_mode="surrogate")

# 3) Differentiable EMG: convolve spikes with (frozen) MUAP templates
emg = surface_emg_jax(spikes.T[None].astype("float32"), muap_templates)

# 4) JIT-compiled forward pass
run = compile_run(config, params, spike_mode="hard")
outputs = run(params, inputs, jax.random.PRNGKey(0))
```

### Spike modes

| `spike_mode` | Forward pass | Gradient through spikes | Use for |
|---|---|---|---|
| `"hard"` (default) | exact discrete spikes | none (spikes non-differentiable) | scientific simulation |
| `"surrogate"` | **identical to `hard`** | smooth surrogate (biased) | optimization / fitting |
| `"rate"` | continuous rate proxy | exact | clean gradients when a continuous forward is acceptable |

The active mode is recorded in the returned `metadata`.

## Differentiable API (in `myogen.simulator.jaxley`)

- `run_jax(params, config, inputs, key, spike_mode=...)` — run the closed loop; returns
  `(outputs, metadata)`.
- `value_and_grad_run(loss_fn, params, config, inputs, key, ...)` — gradients of a loss
  w.r.t. float parameters (integer fields held constant automatically).
- `compile_run(config, params_template, spike_mode=...)` — a `jax.jit`-compiled forward
  pass bound to a config.
- `surface_emg_jax`, `intramuscular_emg_jax` — differentiable EMG from spikes + static
  MUAP templates.
- `ClosedLoopConfig` — static (non-differentiable) network configuration.

What is differentiable today: force, joint kinematics, and EMG **with respect to neural,
muscle, and joint parameters**. Differentiating EMG w.r.t. tissue conductivities /
geometry (the volume-conductor model) is future work; the enabling piece
(differentiable Bessel functions, `myogen.simulator.jaxley.bessel`) is already
validated.

## Branches & backends

MyoGen is developed on two branches with different primary simulators:

| | **Jaxley branch** (this one) | NEURON branch (upstream `main`) |
|---|---|---|
| Primary simulator | Jaxley (JAX) | NEURON |
| Differentiable | **yes** | no |
| GPU | yes | no |
| Install | `pip install myogen` | (that branch's instructions) |
| Best for | gradient-based fitting, digital twins, GPU runs | exact NEURON semantics / reference results |

On **this** branch Jaxley is the default and only required backend — `import myogen`
needs no NEURON runtime, compiler, or MPI. NEURON is still available here as an
**optional** validation reference (`pip install "myogen[neuron]"`, then
`from myogen.simulator.neuron import ...`), but it is not needed for normal use.

The top-level `myogen.simulator` classes (`HillModel`, `Network`, `SimulationRunner`,
`SpindleModel`, `GolgiTendonOrganModel`, `JointDynamics`) resolve to the **Jaxley**
implementations on this branch.

### Motor-neuron model

The default motor neuron is the **NERLab** single-isopotential-dendrite cell matching
the production NEURON model: a soma (channel `napp`: fast Na + persistent Na + fast K +
slow K + leak) plus one isopotential dendrite (channel `caL`: L-type Ca + leak), in the
original 1952 Hodgkin–Huxley voltage frame (V_rest ≈ 0 mV, spikes peak ≈ +90 mV). An
older Powers2017 cable model is available via `model="Powers2017"`.

## Where the code lives

```text
myogen/simulator/jaxley/
  closed_loop.py         # run_jax / value_and_grad_run / compile_run (differentiable entry point)
  jax_models.py          # pure-JAX spindle / GTO / Hill / joint steps + spike_detect + scan factory
  emg.py                 # differentiable surface / intramuscular EMG
  bessel.py              # differentiable modified Bessel functions (for future volume-conductor grads)
  populations/           # AlphaMN__Pool, DescendingDrive__Pool, AffIa/II/Ib__Pool, GII/GIb__Pool
  channels/              # NERLab (napp, caL) and Powers2017 channels
  muscle.py, proprioception/, joint_dynamics.py, network.py, simulation_runner.py

myogen/simulator/core/   # backend-agnostic muscle geometry, force, MUAP, surface/intramuscular EMG
myogen/simulator/neuron/ # optional NEURON backend (validation reference)
```

## Examples

Canonical examples default to Jaxley — see `examples/README.md` for the full map.
`XX_<name>.py` is the Jaxley version; `XX_<name>_neuron.py` (where present) is the
NEURON counterpart and needs `myogen[neuron]`. The full closed-loop spinal network is
`examples/01_basic/11_simulate_spinal_network.py`.

## Status & caveats

- Surrogate spike gradients are **biased** — inherent to differentiating spiking models;
  use `rate` mode for unbiased gradients when a continuous forward output is acceptable.
- This branch running on Jaxley is distinct from "Jaxley reproduces NEURON exactly
  everywhere" — some quantitative gaps remain (e.g. F-I gain on some examples). If you
  need NEURON as the reference, use this branch's optional `[neuron]` extra or the
  upstream NEURON branch.
- Not yet ported to Jaxley: the `02_finetune` and `03_papers/watanabe` NEURON
  simulations (runnable via the extra).

## Further reading

- `docs/DIFFERENTIABILITY_STATUS.md` — what is differentiable, how it was validated.
- `docs/NEURON_DECOUPLING.md` — how Jaxley became the default and NEURON optional.
- `docs/M1_bessel_feasibility.md` — differentiable volume-conductor feasibility.
- `examples/README.md` — the Jaxley (default) vs NEURON example map.
