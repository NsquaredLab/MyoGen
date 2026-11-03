# Differentiable API (Jaxley)

The Jaxley backend exposes a **differentiable, JIT-compilable** neuromuscular closed
loop. These functions let you take gradients of force, joint kinematics, and EMG with
respect to neural, muscle, and joint parameters (see the
[Differentiability status](../DIFFERENTIABILITY_STATUS.md) page for the design and
validation). Import them from `myogen.simulator.jaxley`.

## Closed-loop simulation & gradients

::: myogen.simulator.jaxley.run_jax

::: myogen.simulator.jaxley.value_and_grad_run

::: myogen.simulator.jaxley.compile_run

::: myogen.simulator.jaxley.ClosedLoopConfig

::: myogen.simulator.jaxley.partition_differentiable

## Spike modes

::: myogen.simulator.jaxley.spike_detect

## Differentiable EMG

Convolve (surrogate/rate) spike trains with static MUAP templates. Differentiable with
respect to everything upstream of the spikes.

::: myogen.simulator.jaxley.surface_emg_jax

::: myogen.simulator.jaxley.intramuscular_emg_jax

::: myogen.simulator.jaxley.resample_muaps

## Differentiable muscle parameters

::: myogen.simulator.jaxley.differentiable_twitch_params

## Differentiable Bessel functions

Native-JAX modified Bessel functions of integer order (used by the volume-conductor
work); differentiable via autodiff.

::: myogen.simulator.jaxley.iv_int

::: myogen.simulator.jaxley.kv_int
