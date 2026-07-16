# Simulator

## Recruitment
::: myogen.simulator.RecruitmentThresholds

## Neuron populations
::: myogen.simulator.neuron.populations.AlphaMN__Pool
::: myogen.simulator.neuron.populations.DescendingDrive__Pool

`DescendingDrive_Gamma__Pool` is a backward-compatibility alias. New code should
prefer `DescendingDrive__Pool(process_type="gamma", shape=...)`.

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
