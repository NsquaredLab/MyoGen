API Documentation
=================

This section provides complete API reference documentation for MyoGen's modules and functions.

MyoGen is organized into two main modules:

.. list-table::
   :widths: 20 80
   :header-rows: 0

   * - :ref:`Simulator <simulator-module>`  
     - Core functionality for neuromuscular simulation including motor unit recruitment, muscle modeling, and EMG generation
   * - :ref:`Utils <utils-module>`
     - Utility functions for setup, current generation, plotting, and type definitions

.. _simulator-module:

Simulator Module
----------------

.. currentmodule:: myogen.simulator

The simulator module provides comprehensive functionality for neuromuscular modeling:

- **Motor Unit Recruitment**: Generate physiologically realistic recruitment thresholds
- **Neural Simulation**: Simulate motor neuron pools with spike train generation  
- **Muscle Modeling**: Model muscle fiber distribution and motor unit territories using Voronoi tessellation
- **EMG Generation**: Create surface EMG signals from motor unit action potentials and spike trains

Motor Unit Recruitment Thresholds
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Generate recruitment thresholds based on physiological distributions.

.. currentmodule:: myogen.simulator
.. autosummary::
   :toctree: ../generated/
   :template: autosummary/function.rst
   :recursive:
   
   generate_mu_recruitment_thresholds


Motor Neuron Pool
^^^^^^^^^^^^^^^^^

Simulate motor neuron pools with realistic firing patterns and spike train generation.

.. currentmodule:: myogen.simulator
.. autosummary::
   :toctree: ../generated/
   :template: autosummary/class.rst
   :recursive:
   
   MotorNeuronPool


Muscle Model
^^^^^^^^^^^^

Model muscle fiber distribution and motor unit territories using advanced geometric techniques.

.. currentmodule:: myogen.simulator
.. autosummary::
   :toctree: ../generated/
   :template: autosummary/class.rst
   :recursive:
   
   Muscle


Surface EMG Generation
^^^^^^^^^^^^^^^^^^^^^^

Generate surface EMG signals from motor unit action potentials and spike trains.

.. note::
   iEMG simulation is not yet implemented.

.. currentmodule:: myogen.simulator
.. autosummary::
   :toctree: ../generated/
   :template: autosummary/class.rst
   :recursive:
   
   SurfaceEMG


.. _utils-module:

Utils Module
------------

The utils module provides essential utility functions for:

- **Setup & Configuration**: Initialize MyoGen and handle NMODL compilation
- **Current Generation**: Create various current waveforms (ramp, step, sinusoidal, etc.)
- **Visualization**: Plot simulation results and analysis
- **Type Safety**: Type definitions for numpy arrays with specific dimensions

Setup Functions
^^^^^^^^^^^^^^^

Essential functions for setting up MyoGen and handling NMODL files.

.. currentmodule:: myogen.utils

.. autosummary::
   :toctree: ../generated/
   :template: autosummary/function.rst
   :recursive:

   setup_myogen
   load_nmodl_files

Current Generation
^^^^^^^^^^^^^^^^^^

Create various types of input current waveforms for neural stimulation.

.. currentmodule:: myogen.utils.currents

.. autosummary::
   :toctree: ../generated/
   :template: autosummary/function.rst
   :recursive:

   create_ramp_current
   create_step_current
   create_sinusoidal_current
   create_sawtooth_current
   create_trapezoid_current


Plotting & Visualization
^^^^^^^^^^^^^^^^^^^^^^^^

Comprehensive plotting functions for visualizing simulation results.

.. currentmodule:: myogen.utils.plotting

.. autosummary::
   :toctree: ../generated/
   :template: autosummary/function.rst
   :recursive:

   plot_recruitment_thresholds
   plot_spike_trains
   plot_surface_emg
   plot_muap_grid
   plot_input_current__matrix


Type Definitions
^^^^^^^^^^^^^^^^

Type aliases for numpy arrays with specific dimensions, providing type safety and documentation.

.. currentmodule:: myogen.utils.types

.. autosummary::
   :toctree: ../generated/
   :template: autosummary/data.rst
   :recursive:

   INPUT_CURRENT__MATRIX
   SPIKE_TRAIN__MATRIX
   MUAP_SHAPE__TENSOR
   SURFACE_EMG__TENSOR
