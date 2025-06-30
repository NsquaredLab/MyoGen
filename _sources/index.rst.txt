.. |logo| image:: _static/myogen_logo.png
   :height: 80px
   :align: middle

Welcome to |logo|
==================

**Biophysical Simulation Toolkit for EMG Signal Generation**

MyoGen is a biophysical simulation toolkit for generating surface and intramuscular electromyography (EMG) signals. Built on established physiological principles and validated anatomical data, MyoGen provides researchers, engineers, and educators with an accessible platform for EMG signal simulation that spans from motor unit recruitment to surface electrode recordings.

The toolkit implements a complete simulation pipeline: from motor unit recruitment thresholds and spike train generation using the NEURON simulator, through anatomically accurate muscle modeling with realistic fiber distributions, to surface EMG signal synthesis via Motor Unit Action Potentials (MUAPs).

.. raw:: html

    <div>
        <form class="bd-search align-items-center" action="search.html" method="get">
          <input type="search" class="form-control search-front-page" name="q" id="search-input" placeholder="&#128269; Search the docs ..." aria-label="Search the docs ..." autocomplete="off">
        </form>
    </div>
    </br>

Key features include:

- **Motor Unit Recruitment Modeling**: Four validated models (Fuglevand, De Luca, Konstantin, Combined) for physiologically realistic recruitment patterns
- **Biophysical Spike Train Simulation**: NEURON-based motor neuron modeling with detailed calcium dynamics and membrane properties  
- **Anatomically Accurate Muscle Models**: Spatial distribution of motor units and muscle fibers based on anatomical measurements
- **Surface EMG Synthesis**: Multi-layered volume conductor modeling for realistic surface EMG signal generation
- **Multi-Electrode Array Support**: Simulation of high-density electrode grids with configurable spatial arrangements
- **Flexible Input Current Patterns**: Built-in generators for sinusoidal, ramp, step, trapezoid, and sawtooth stimulation waveforms
- **GPU Acceleration**: Automatic CuPy integration for fast parallel processing of large-scale simulations
- **Comprehensive Visualization**: Built-in plotting tools for recruitment thresholds, spike trains, muscle anatomy, MUAPs, and surface EMG
- **Reproducible Research**: Deterministic random number generation and parameter saving for reproducible simulations

.. warning::
   MyoGen is still under development and the API is subject to change.

Installation
------------

1. **Clone the Repository:**

   .. code-block:: bash

      git clone https://github.com/NsquaredLab/MyoGen.git
      cd MyoGen

2. **(Windows only) Install NEURON 8.2.6:**

   .. warning::
      Make sure to install version 8.2.6.

   Download from: https://github.com/neuronsimulator/nrn/releases/download/8.2.6/nrn-8.2.6.w64-mingw-py-38-39-310-311-312-setup.exe

3. **Install UV:**
   
   Follow the instructions at https://docs.astral.sh/uv/#highlights

4. **Create Environment:**

   .. code-block:: bash

      uv sync

5. **Add CuPy if CUDA is Available:**

   .. code-block:: bash

      uv pip install cupy-cuda12x

6. **Activate the Environment:**

   .. code-block:: bash

      source .venv/bin/activate

7. **Run Setup to Compile NMODL Files:**

   .. warning::
      This step is required. Please do not skip it.

   .. code-block:: bash

      python -c "from myogen.utils import setup_myogen; setup_myogen()"

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