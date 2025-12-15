API Documentation
=================

Welcome to the MyoGen API reference!
This section provides a complete overview of all modules, classes, and functions available in MyoGen for neuromuscular simulation and analysis.

MyoGen is organized into the following modules:

.. grid:: 2
   :gutter: 0
   :margin: 0

   .. card:: MyoGen Core
      :link: myogen_api.html
      :class-card: sd-shadow-xs sd-bg-light myogen-card

      Package-level functions and global objects for random number generation and setup.

   .. card:: Simulator
      :link: simulator_api.html
      :class-card: sd-shadow-xs sd-bg-light myogen-card

      Core functionality for neuromuscular simulation, including motor unit recruitment, muscle modeling, and EMG generation.

   .. card:: Utils
      :link: utils_api.html
      :class-card: sd-shadow-xs sd-bg-light myogen-card

      Utility functions for setup, NMODL file handling, current generation, plotting, and type definitions.

   .. card:: Currents
      :link: currents_api.html
      :class-card: sd-shadow-xs sd-bg-light myogen-card

      Functions for generating various input current waveforms (ramp, step, sinusoidal, etc.). *(Submodule of Utils)*

   .. card:: Plotting
      :link: plotting_api.html
      :class-card: sd-shadow-xs sd-bg-light myogen-card
   
      Visualization tools for simulation results and analysis. *(Submodule of Utils)*

   .. card:: Types
      :link: types_api.html
      :class-card: sd-shadow-xs sd-bg-light myogen-card

      Type definitions for structured data and type safety. *(Submodule of Utils)*

**How to use this documentation:**

- Click on a module above to see all its classes and functions.
- Each API page provides autosummary tables with links to detailed docstrings and usage examples.
- For a practical introduction, see the `examples` section in the documentation sidebar.

If you are new to MyoGen, start with the Simulator module to understand the core simulation workflow, then explore the utility and plotting modules as needed.

If you have questions or need further help, please refer to the `README <../../README.md>`_ or open an issue on `GitHub <https://github.com/NSquaredLab/MyoGen>`_.


.. toctree::
   :maxdepth: 2
   :caption: API Modules
   :hidden:

   myogen_api
   simulator_api
   utils_api