# Simulator

## Recruitment
::: myogen.simulator.RecruitmentThresholds

## Motor-unit populations

Jaxley is the default backend. The NEURON backend (optional `neuron` extra) exposes
API-compatible equivalents; both are documented below.

### Jaxley (default)

::: myogen.simulator.jaxley.populations.AlphaMN__Pool
::: myogen.simulator.jaxley.populations.DescendingDrive__Pool

`DescendingDrive_Gamma__Pool` is a backward-compatibility alias for a gamma-process
descending drive.

::: myogen.simulator.jaxley.populations.DescendingDrive_Gamma__Pool
::: myogen.simulator.jaxley.populations.AffIa__Pool
::: myogen.simulator.jaxley.populations.AffII__Pool
::: myogen.simulator.jaxley.populations.AffIb__Pool
::: myogen.simulator.jaxley.populations.GII__Pool
::: myogen.simulator.jaxley.populations.GIb__Pool

### NEURON (optional backend, `myogen[neuron]`)

::: myogen.simulator.neuron.populations.AlphaMN__Pool
::: myogen.simulator.neuron.populations.DescendingDrive__Pool
::: myogen.simulator.neuron.populations.DescendingDrive_Gamma__Pool
::: myogen.simulator.neuron.populations.AffIa__Pool
::: myogen.simulator.neuron.populations.AffII__Pool
::: myogen.simulator.neuron.populations.AffIb__Pool
::: myogen.simulator.neuron.populations.GII__Pool
::: myogen.simulator.neuron.populations.GIb__Pool

## Network & runner
::: myogen.simulator.Network
::: myogen.simulator.SimulationRunner

## Muscle & force
::: myogen.simulator.Muscle
::: myogen.simulator.HillModel
::: myogen.simulator.ForceModel
::: myogen.simulator.ForceModelVectorized

## EMG
::: myogen.simulator.SurfaceEMG
::: myogen.simulator.IntramuscularEMG
::: myogen.simulator.SurfaceElectrodeArray
::: myogen.simulator.IntramuscularElectrodeArray

## Proprioception
::: myogen.simulator.SpindleModel
::: myogen.simulator.GolgiTendonOrganModel
::: myogen.simulator.JointDynamics

## Geometry & biomechanics
::: myogen.simulator.MuscleGeometry
::: myogen.simulator.JointGeometry
::: myogen.simulator.JointBiomechanics

## Grid / Neo utilities
::: myogen.simulator.create_grid_signal
::: myogen.simulator.signal_to_grid
::: myogen.simulator.get_electrode
::: myogen.simulator.get_row
::: myogen.simulator.get_column

The deprecated `GridAnalogSignal` compatibility class is intentionally excluded.
