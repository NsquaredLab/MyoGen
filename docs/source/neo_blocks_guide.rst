Working with Neo Block Data Structures
=======================================

MyoGen uses **Neo Block** objects as the primary data containers for storing simulation results. This guide explains how to extract and work with data from the different Block types.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

MyoGen uses five different Neo Block types to store simulation data:

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Block Type
     - Purpose
     - Dimensions
   * - ``SPIKE_TRAIN__Block``
     - Neural firing times
     - 1D spike times
   * - ``SURFACE_MUAP__Block``
     - Surface MUAP templates
     - 3D electrode grid
   * - ``SURFACE_EMG__Block``
     - Surface EMG signals
     - 3D electrode grid
   * - ``INTRAMUSCULAR_MUAP__Block``
     - Intramuscular MUAP templates
     - 2D electrode array
   * - ``INTRAMUSCULAR_EMG__Block``
     - Intramuscular EMG signals
     - 2D electrode array

Neo Blocks provide standardized containers with:

- Automatic unit tracking (mV, µV, seconds, etc.)
- Metadata storage (sampling rate, duration, etc.)
- Compatibility with analysis tools like Elephant
- Integration with neuroscience workflows

Quick Start
-----------

Basic Access Patterns
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import joblib

   # Load spike trains
   spike_trains = joblib.load("spike_trains.pkl")
   spiketrain = spike_trains.segments[0].spiketrains[0]
   spike_times = spiketrain.magnitude  # NumPy array

   # Load surface EMG
   surface_emg = joblib.load("surface_emg.pkl")
   emg_signal = surface_emg.groups[0].segments[0].analogsignals[0]
   electrode_data = emg_signal[:, row, col]  # Time series at electrode

   # Load intramuscular EMG
   im_emg = joblib.load("intramuscular_emg.pkl")
   emg_signal = im_emg.segments[0].analogsignals[0]
   electrode_data = emg_signal[:, electrode_idx]  # Time series

Block Structure Details
------------------------

SPIKE_TRAIN__Block
~~~~~~~~~~~~~~~~~~~

**Structure**: Block → Segments (motor pools) → SpikeTrain objects

The ``SPIKE_TRAIN__Block`` stores neural firing patterns from motor neuron pools. Each segment represents a motor pool, and each spiketrain contains the firing times of an individual neuron.

.. code-block:: python

   # Access spike trains
   motor_pool = spike_train__Block.segments[pool_idx]
   spiketrain = motor_pool.spiketrains[neuron_idx]

   # Extract data
   spike_times__s = spiketrain.magnitude  # NumPy array
   n_spikes = len(spiketrain)
   duration = spiketrain.t_stop
   sampling_rate = spiketrain.sampling_rate

**Example: Calculate firing rate**

.. code-block:: python

   import elephant.statistics

   motor_pool = spike_train__Block.segments[0]
   firing_rates = []

   for spiketrain in motor_pool.spiketrains:
       if len(spiketrain) > 0:
           rate = elephant.statistics.mean_firing_rate(spiketrain)
           firing_rates.append(rate.magnitude)

   print(f"Mean firing rate: {np.mean(firing_rates):.2f} Hz")

SURFACE_MUAP__Block
~~~~~~~~~~~~~~~~~~~~

**Structure**: Block → Groups (electrode arrays) → Segments (MUAP indices) → AnalogSignals (3D)

Stores Motor Unit Action Potential (MUAP) templates for surface electrode arrays. Each group represents an electrode array, each segment represents a MUAP, and signals are 3D arrays (time × rows × columns).

.. code-block:: python

   # Access MUAPs
   electrode_array = muaps__Block.groups[array_idx]
   muap_segment = electrode_array.segments[muap_idx]
   muap_signal = muap_segment.analogsignals[0]

   # Extract data
   muap_array = muap_signal.magnitude  # Shape: (time, rows, cols)
   time__s = muap_signal.times.magnitude
   electrode_muap = muap_signal[:, row, col]  # Specific electrode

**Example: Analyze MUAP amplitude**

.. code-block:: python

   # Get MUAP from motor unit 5, electrode at row 2, column 3
   muap_signal = muaps__Block.groups[0].segments[5].analogsignals[0]

   row, col = 2, 3
   electrode_muap = muap_signal[:, row, col]

   print(f"Peak-to-peak: {np.ptp(electrode_muap.magnitude):.2f} {electrode_muap.units}")
   print(f"Max amplitude: {np.max(np.abs(electrode_muap.magnitude)):.2f} {electrode_muap.units}")

SURFACE_EMG__Block
~~~~~~~~~~~~~~~~~~~

**Structure**: Block → Groups (electrode arrays) → Segments (motor pools) → AnalogSignals (3D)

Stores synthesized surface EMG signals. Similar structure to SURFACE_MUAP__Block, but contains the full EMG signal (longer duration) rather than templates.

.. code-block:: python

   # Access surface EMG
   electrode_array = surface_emg__Block.groups[array_idx]
   motor_pool = electrode_array.segments[pool_idx]
   emg_signal = motor_pool.analogsignals[0]

   # Extract data
   emg_array = emg_signal.magnitude  # Shape: (time, rows, cols)
   electrode_emg = emg_signal[:, row, col]  # Specific electrode

**Example: Calculate RMS amplitude**

.. code-block:: python

   emg_signal = surface_emg__Block.groups[0].segments[0].analogsignals[0]
   electrode_emg = emg_signal[:, 2, 2].magnitude

   rms = np.sqrt(np.mean(electrode_emg ** 2))
   print(f"RMS amplitude: {rms:.2f} {emg_signal.units}")

INTRAMUSCULAR_MUAP__Block
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Structure**: Block → Segments (MUAP indices) → AnalogSignals (2D)

Stores MUAP templates for intramuscular electrodes. Each segment represents a MUAP, and signals are 2D arrays (time × electrodes).

.. code-block:: python

   # Access intramuscular MUAPs
   muap_segment = im_muaps__Block.segments[muap_idx]
   muap_signal = muap_segment.analogsignals[0]

   # Extract data
   muap_array = muap_signal.magnitude  # Shape: (time, electrodes)
   electrode_muap = muap_signal[:, electrode_idx]

INTRAMUSCULAR_EMG__Block
~~~~~~~~~~~~~~~~~~~~~~~~~

**Structure**: Block → Segments (motor pools) → AnalogSignals (2D)

Stores synthesized intramuscular EMG signals. Each segment represents a motor pool, and signals are 2D arrays (time × electrodes).

.. code-block:: python

   # Access intramuscular EMG
   motor_pool = im_emg__Block.segments[pool_idx]
   emg_signal = motor_pool.analogsignals[0]

   # Extract data
   emg_array = emg_signal.magnitude  # Shape: (time, electrodes)
   electrode_emg = emg_signal[:, electrode_idx]

Common Operations
-----------------

Accessing Metadata
~~~~~~~~~~~~~~~~~~~

All Neo signals have metadata properties:

.. code-block:: python

   signal.magnitude          # NumPy array of values
   signal.times.magnitude    # Time axis (in seconds)
   signal.sampling_rate      # Sampling frequency
   signal.sampling_period    # Time between samples
   signal.t_start           # Start time
   signal.t_stop            # Stop time
   signal.duration          # Total duration
   signal.units             # Physical units (mV, uV, etc.)
   signal.shape             # Array dimensions

Extracting Time Windows
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # For spike trains
   windowed_train = spiketrain.time_slice(t_start, t_stop)

   # For analog signals
   start_idx = int(t_start * emg_signal.sampling_rate.magnitude)
   end_idx = int(t_stop * emg_signal.sampling_rate.magnitude)
   windowed_signal = emg_signal[start_idx:end_idx]

Converting Units
~~~~~~~~~~~~~~~~

.. code-block:: python

   import quantities as pq

   # Check current units
   print(signal.units)  # e.g., mV

   # Convert to different units
   signal_uV = signal.rescale(pq.uV)

Saving and Loading
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import joblib

   # Save blocks
   joblib.dump(spike_train__Block, "spike_trains.pkl")
   joblib.dump(surface_emg__Block, "surface_emg.pkl")

   # Load blocks
   spike_trains = joblib.load("spike_trains.pkl")
   surface_emg = joblib.load("surface_emg.pkl")

Shape Reference
---------------

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - Block Type
     - Dimensionality
     - Shape Format
   * - SPIKE_TRAIN
     - 1D
     - ``(n_spikes,)``
   * - SURFACE_MUAP
     - 3D
     - ``(time, rows, cols)``
   * - SURFACE_EMG
     - 3D
     - ``(time, rows, cols)``
   * - INTRAMUSCULAR_MUAP
     - 2D
     - ``(time, electrodes)``
   * - INTRAMUSCULAR_EMG
     - 2D
     - ``(time, electrodes)``

Data Flow Pipeline
------------------

The simulation pipeline follows this flow:

1. **Generate Spike Trains** → ``SPIKE_TRAIN__Block``

   - When neurons fire

2. **Generate MUAPs** → ``SURFACE_MUAP__Block`` or ``INTRAMUSCULAR_MUAP__Block``

   - What individual motor units look like

3. **Convolve MUAPs with Spike Trains** → ``SURFACE_EMG__Block`` or ``INTRAMUSCULAR_EMG__Block``

   - Final synthesized EMG signals

Complete Example
----------------

.. code-block:: python

   import joblib
   import numpy as np
   import matplotlib.pyplot as plt
   from myogen.utils.types import SPIKE_TRAIN__Block, SURFACE_EMG__Block

   # 1. Load data
   spike_trains = joblib.load("spike_trains.pkl")
   surface_emg = joblib.load("surface_emg.pkl")

   # 2. Extract spike times
   motor_pool = spike_trains.segments[0]
   neuron_5 = motor_pool.spiketrains[5]
   spike_times = neuron_5.magnitude

   # 3. Extract EMG from center electrode
   emg_signal = surface_emg.groups[0].segments[0].analogsignals[0]
   center_row = emg_signal.shape[1] // 2
   center_col = emg_signal.shape[2] // 2
   electrode_emg = emg_signal[:, center_row, center_col]

   # 4. Plot
   fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

   # Spike raster
   ax1.scatter(spike_times, [0]*len(spike_times), s=10, c='black')
   ax1.set_ylabel('Neuron 5')
   ax1.set_ylim(-0.5, 0.5)

   # EMG signal
   time = emg_signal.times.magnitude
   ax2.plot(time, electrode_emg.magnitude)
   ax2.set_xlabel('Time (s)')
   ax2.set_ylabel(f'EMG ({electrode_emg.units})')

   plt.tight_layout()
   plt.show()

Troubleshooting
---------------

Missing 'groups' attribute
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Issue**: ``AttributeError: 'Block' object has no attribute 'groups'``

**Solution**: This is likely a spike train or intramuscular block. Those use ``segments`` not ``groups``.

IndexError when accessing electrodes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Issue**: ``IndexError`` when accessing electrodes

**Solution**: Check the shape first:

.. code-block:: python

   print(signal.shape)  # e.g., (150000, 13, 5)
   # Then access within bounds: signal[:, 0:13, 0:5]

Wrong units
~~~~~~~~~~~

**Issue**: Values seem wrong (very large or very small)

**Solution**: Check the units:

.. code-block:: python

   print(signal.units)  # e.g., mV or µV
   signal.rescale(pq.mV)  # Convert to desired unit

Additional Resources
--------------------

- Example: ``examples/01_basic/12_extract_data_from_neo_blocks.py``
- `Neo Documentation <https://neo.readthedocs.io/>`_
- `Elephant (Electrophysiology Analysis) <https://elephant.readthedocs.io/>`_
- `Quantities (Unit Handling) <https://python-quantities.readthedocs.io/>`_
