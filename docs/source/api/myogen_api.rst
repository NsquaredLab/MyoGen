MyoGen Core
===========

This module contains the core MyoGen package-level functions and objects for random number generation and setup.


.. currentmodule:: myogen

.. autosummary::
   :toctree: ../generated/
   :template: autosummary/function.rst
   :recursive:

   set_random_seed
   load_nmodl_mechanisms

Global Objects
--------------

.. data:: RANDOM_GENERATOR
   :type: numpy.random.Generator

   Global random number generator for reproducibility across MyoGen simulations.

   This is a :class:`numpy.random.Generator` instance initialized with the default seed.
   Use :func:`set_random_seed` to change the seed for reproducible simulations.

.. data:: SEED
   :type: int

   Default random seed value (180319) used to initialize :data:`RANDOM_GENERATOR`.
