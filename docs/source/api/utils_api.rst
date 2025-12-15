Utils Module
============

This module contains utility functions for setup, NMODL file handling, current generation, plotting, and type definitions.


Current Generation
^^^^^^^^^^^^^^^^^^

.. currentmodule:: myogen.utils.currents

.. autosummary::
   :toctree: ../generated/
   :template: autosummary/function.rst

   create_ramp_current
   create_step_current
   create_sinusoidal_current
   create_sawtooth_current
   create_trapezoid_current


NEURON Utilities
^^^^^^^^^^^^^^^^

.. currentmodule:: myogen.utils.neuron.inject_currents_into_populations

.. autosummary::
   :toctree: ../generated/
   :template: autosummary/function.rst

   inject_currents_into_populations
   inject_currents_and_simulate_spike_trains


Plotting & Visualization
^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: myogen.utils.plotting

.. autosummary::
   :toctree: ../generated/
   :template: autosummary/function.rst

   plot_raster_spikes
   plot_membrane_potentials
   plot_muscle_dynamics
   plot_antagonist_muscle_comparison
   plot_spindle_dynamics
   plot_gto_dynamics


Type Definitions
^^^^^^^^^^^^^^^^

.. currentmodule:: myogen.utils.types

Quantity Types
""""""""""""""

.. autosummary::
   :toctree: ../generated/
   :template: autosummary/data.rst

   Quantity__s
   Quantity__ms
   Quantity__rad
   Quantity__deg
   Quantity__mV
   Quantity__uV
   Quantity__nA
   Quantity__uS
   Quantity__S_per_m
   Quantity__Hz
   Quantity__pps
   Quantity__mm
   Quantity__m
   Quantity__mm2
   Quantity__per_mm2
   Quantity__m_per_s
   Quantity__mm_per_s

Neo Data Types
""""""""""""""

.. autosummary::
   :toctree: ../generated/
   :template: autosummary/data.rst

   CURRENT__AnalogSignal
   FORCE__AnalogSignal
   SPIKE_TRAIN__Block
   SURFACE_MUAP__Block
   SURFACE_EMG__Block
   INTRAMUSCULAR_MUAP__Block
   INTRAMUSCULAR_EMG__Block

Matrix & Array Types
""""""""""""""""""""

.. autosummary::
   :toctree: ../generated/
   :template: autosummary/data.rst

   CORTICAL_INPUT__MATRIX
   RECRUITMENT_THRESHOLDS__ARRAY
   JOINT_ANGLE__ARRAY
   MOMENT_ARM__MATRIX 