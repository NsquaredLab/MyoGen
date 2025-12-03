Type Definitions
================

This module contains type definitions for structured data and type safety.

.. currentmodule:: myogen.utils.types

Type Aliases
^^^^^^^^^^^^

The following type aliases are available for use with beartype validation:

.. py:data:: INPUT_CURRENT__MATRIX
   :type: TypeAlias

   2D array representing input current patterns.

   **Shape:** ``(n_units, n_timesteps)``

.. py:data:: CORTICAL_INPUT__MATRIX
   :type: TypeAlias

   2D array representing cortical input patterns.

   **Shape:** ``(n_units, n_timesteps)``

.. py:data:: SPIKE_TRAIN__MATRIX
   :type: TypeAlias

   2D boolean array representing spike train patterns.

   **Shape:** ``(n_units, n_timesteps)``

.. py:data:: SURFACE_MUAP_SHAPE__TENSOR
   :type: TypeAlias

   4D array representing surface motor unit action potential shapes.

   **Shape:** ``(n_units, n_rows, n_cols, n_timesteps)``

.. py:data:: INTRAMUSCULAR_MUAP_SHAPE__TENSOR
   :type: TypeAlias

   3D array representing intramuscular motor unit action potential shapes.

   **Shape:** ``(n_units, n_channels, n_timesteps)``

.. py:data:: SURFACE_EMG__TENSOR
   :type: TypeAlias

   4D array representing surface EMG signals.

   **Shape:** ``(n_rows, n_cols, n_timesteps)``

.. py:data:: INTRAMUSCULAR_EMG__TENSOR
   :type: TypeAlias

   3D array representing intramuscular EMG signals.

   **Shape:** ``(n_channels, n_timesteps)`` 