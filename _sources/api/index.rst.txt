API Documentation
=================

Welcome to the MyoGen API reference!
This section provides a complete overview of all modules, classes, and functions available in MyoGen for neuromuscular simulation and analysis.

MyoGen is organized into the following modules:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - **Module**
     - **Description**
   * - Simulator
     - Core functionality for neuromuscular simulation, including motor unit recruitment, muscle modeling, and EMG generation.
   * - Utils
     - Utility functions for setup, NMODL file handling, current generation, plotting, and type definitions.
   * - Currents
     - Functions for generating various input current waveforms (ramp, step, sinusoidal, etc.). *(Submodule of Utils)*
   * - Plotting
     - Visualization tools for simulation results and analysis. *(Submodule of Utils)*
   * - Types
     - Type definitions for structured data and type safety. *(Submodule of Utils)*

**Browse the API by module:**

.. toctree::
   :maxdepth: 2
   :caption: API Modules

   simulator_api
   utils_api

---

**How to use this documentation:**

- Click on a module above to see all its classes and functions.
- Each API page provides autosummary tables with links to detailed docstrings and usage examples.
- For a practical introduction, see the `examples` section in the documentation sidebar.

If you are new to MyoGen, start with the Simulator module to understand the core simulation workflow, then explore the utility and plotting modules as needed.

---

If you have questions or need further help, please refer to the `README <../../README.md>`_ or open an issue on `GitHub <https://github.com/NSquaredLab/MyoGen>`_.
