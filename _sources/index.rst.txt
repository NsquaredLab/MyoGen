.. |logo| image:: _static/myogen_logo.png
   :height: 80px
   :align: middle

Welcome to |logo|
==================

**Computational Framework for Neuromuscular Simulation and EMG Modeling**

MyoGen is a Python framework designed for simulating neuromuscular systems and generating realistic EMG signals. 
It provides researchers with easily accessible tools for modeling motor unit behavior, muscle dynamics, and EMG signals.

.. raw:: html

    <div>
        <form class="bd-search align-items-center" action="search.html" method="get">
          <input type="search" class="form-control search-front-page" name="q" id="search-input" placeholder="&#128269; Search the docs ..." aria-label="Search the docs ..." autocomplete="off">
        </form>
    </div>
    </br>

The framework is designed with the following goals:

1. **Accurate Simulation**: Physiologically realistic models of motor units, muscles, and EMG signals
2. **Ease of Use**: Simple APIs that abstract complex neuromuscular modeling and allow for rapid prototyping
3. **Extensibility**: The framework is designed to be extensible, allowing for the addition of new models and simulation approaches
4. **Utility Functions**: Current generation, plotting tools, and setup utilities

.. warning::
   MyoGen is still in an early stage of development.    
   The framework is not yet fully functional and the documentation is still under construction.  

   We are actively working on the framework and will be adding new features and improving the documentation regularly.

   We appreciate your understanding and interest in the project!

Installation
------------
1. **Clone the Repository:**

   .. code-block:: bash

      git clone https://github.com/NsquaredLab/MyoGen.git
      cd myogen

2. **Install uv:** If you don't have it yet, install ``uv``. Follow the instructions on the `uv GitHub page <https://github.com/astral-sh/uv>`_.

3. **Set up Virtual Environment & Install Dependencies:**

   .. code-block:: bash

      # Install base dependencies
      uv sync
      
      # To run documentation/examples, install optional groups:
      uv sync --group docs

Package Structure
-----------------

.. code-block:: text

   MyoGen/
   ├── myogen/              # Main package source code
   │   ├── simulator/       # Core simulation functionality
   │   │   ├── core/        # Core simulation components
   │   │   │   ├── emg/     # EMG signal generation
   │   │   │   ├── muscle/  # Muscle modeling
   │   │   │   └── spike_train/ # Motor neuron simulation
   │   │   └── ...
   │   ├── utils/           # Utility functions and tools
   │   │   ├── plotting/    # Visualization utilities
   │   │   ├── currents.py  # Current generation
   │   │   └── nmodl.py     # NMODL file handling
   │   └── ...
   ├── examples/            # Example scripts and tutorials
   ├── docs/                # Documentation source
   ├── pyproject.toml       # Project metadata and dependencies
   └── uv.lock              # Pinned versions of dependencies



.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API Documentation
   
   api/index

.. toctree::
   :maxdepth: 1
   :caption: Examples & Tutorials
   :hidden:

   auto_examples/index