Type Definitions
================

This module contains type definitions for structured data and type safety with Beartype validation.

.. currentmodule:: myogen.utils.types

Physical Quantity Types
^^^^^^^^^^^^^^^^^^^^^^^

Time Units
----------

.. autosummary::
   :toctree: ../generated/
   :template: autosummary/data.rst

   Quantity__s
   Quantity__ms

Angles
------

.. autosummary::
   :toctree: ../generated/
   :template: autosummary/data.rst

   Quantity__rad
   Quantity__deg

Electrical Potential
--------------------

.. autosummary::
   :toctree: ../generated/
   :template: autosummary/data.rst

   Quantity__mV
   Quantity__uV

Electrical Current
------------------

.. autosummary::
   :toctree: ../generated/
   :template: autosummary/data.rst

   Quantity__nA

Electrical Conductance
----------------------

.. autosummary::
   :toctree: ../generated/
   :template: autosummary/data.rst

   Quantity__uS
   Quantity__S_per_m

Frequency
---------

.. autosummary::
   :toctree: ../generated/
   :template: autosummary/data.rst

   Quantity__Hz
   Quantity__pps

Length & Areas
--------------

.. autosummary::
   :toctree: ../generated/
   :template: autosummary/data.rst

   Quantity__mm
   Quantity__m
   Quantity__mm2
   Quantity__per_mm2

Velocity
--------

.. autosummary::
   :toctree: ../generated/
   :template: autosummary/data.rst

   Quantity__m_per_s
   Quantity__mm_per_s

Signal Types (Neo)
^^^^^^^^^^^^^^^^^^

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

Array Types
^^^^^^^^^^^

.. autosummary::
   :toctree: ../generated/
   :template: autosummary/data.rst

   CORTICAL_INPUT__MATRIX
   RECRUITMENT_THRESHOLDS__ARRAY
   JOINT_ANGLE__ARRAY
   MOMENT_ARM__MATRIX
