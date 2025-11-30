import numpy as np
from numpy.random import Generator

SEED: int = 180319  # Seed for reproducibility
RANDOM_GENERATOR: Generator = np.random.default_rng(SEED)


def set_random_seed(seed: int = SEED) -> None:
    """
    Set the random seed for reproducibility.

    Parameters
    ----------
    seed : int, optional
        Seed value to set, by default SEED
    """
    global RANDOM_GENERATOR
    RANDOM_GENERATOR = np.random.default_rng(seed)
    print(f"Random seed set to {seed}.")


def _setup_myogen(quiet: bool = False) -> bool:
    """
    Set up MyoGen with NEURON mechanism compilation and loading.

    This function handles the compilation and loading of NMODL files required for
    neural simulations. It supports both basic MyoGen mechanisms and extended
    spinal circuit mechanisms from spindle_network integration.

    The function now prioritizes PyNN's built-in load_mechanisms() functionality
    for better integration with PyNN simulations, falling back to manual compilation
    when PyNN is not available or fails.

    Parameters
    ----------
    enable_spinal_circuits : bool, optional
        If True, include spinal mechanism files (ion channels, synaptic mechanisms,
        stimulation tools) for proprioceptive modeling and closed-loop control.
        Requires spindle_network-3 integration, by default False
    force_reload : bool, optional
        If True, force recompilation even if mechanisms appear loaded, by default False
    quiet : bool, optional
        If True, suppress most output messages, by default False

    Returns
    -------
    bool
        True if setup completed successfully, False otherwise
    """
    from Cython.Build import cythonize
    from setuptools import Extension, setup

    setup(
        ext_modules=cythonize(
            [
                Extension(
                    "myogen.simulator.neuron._cython._spindle",
                    ["myogen/simulator/neuron/_cython/_spindle.pyx"],
                    extra_compile_args=["-O3", "-march=native", "-ffast-math"],
                ),
                Extension(
                    "myogen.simulator.neuron._cython._hill",
                    ["myogen/simulator/neuron/_cython/_hill.pyx"],
                    extra_compile_args=["-O3", "-march=native", "-ffast-math"],
                ),
                Extension(
                    "myogen.simulator.neuron._cython._gto",
                    ["myogen/simulator/neuron/_cython/_gto.pyx"],
                    extra_compile_args=["-O3", "-march=native", "-ffast-math"],
                ),
                Extension(
                    "myogen.simulator.neuron._cython._poisson_process_generator",
                    ["myogen/simulator/neuron/_cython/_poisson_process_generator.pyx"],
                    extra_compile_args=["-O3", "-march=native", "-ffast-math"],
                ),
                Extension(
                    "myogen.simulator.neuron._cython._gamma_process_generator",
                    ["myogen/simulator/neuron/_cython/_gamma_process_generator.pyx"],
                    extra_compile_args=["-O3", "-march=native", "-ffast-math"],
                ),
                Extension(
                    "myogen.simulator.neuron._cython._simulate_fiber",
                    ["myogen/simulator/neuron/_cython/_simulate_fiber.pyx"],
                    extra_compile_args=[
                        "-O3",
                        "-march=native",
                        "-ffast-math",
                        "-fopenmp",
                        "-Wno-maybe-uninitialized",
                    ],
                    extra_link_args=[
                        "-fopenmp",
                        "-Wl,--no-as-needed",  # Force libmvec to be linked even if not directly needed
                        "-lmvec",  # Link vector math library
                    ],
                    libraries=["mvec"],  # Explicitly link libmvec for vectorized math
                ),
            ],
            compiler_directives={"embedsignature": True},
            nthreads=-2,
        ),
        script_args=["build_ext", "--inplace"],
        include_dirs=[np.get_include()],
    )

    try:
        from myogen.utils.nmodl import compile_nmodl_files

        # Note: load_nmodl_files now includes PyNN integration by default
        return compile_nmodl_files(quiet=quiet)
    except ImportError as e:
        if not quiet:
            print(f"Warning: NEURON not available, skipping mechanism setup: {e}")
        return False
    except Exception as e:
        if not quiet:
            print(f"Error during MyoGen setup: {e}")
        return False


from myogen.utils.nmodl import load_nmodl_mechanisms

__all__ = [
    "RANDOM_GENERATOR",
    "SEED",
    "set_random_seed",
    "load_nmodl_mechanisms",
    "_setup_myogen",
]
