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


def _setup_myogen(quiet: bool = False, force_rebuild: bool = False) -> bool:
    """
    Set up MyoGen with NEURON mechanism compilation and loading.

    This function handles the compilation and loading of NMODL files required for
    neural simulations. It also compiles Cython extensions if they are not already
    available (typically only needed in development mode).

    Parameters
    ----------
    quiet : bool, optional
        If True, suppress most output messages, by default False
    force_rebuild : bool, optional
        If True, force recompilation of Cython extensions even if they appear
        to be already compiled, by default False

    Returns
    -------
    bool
        True if setup completed successfully, False otherwise
    """
    # Check if Cython extensions are already compiled (from installed package)
    cython_modules = [
        "myogen.simulator.neuron._cython._spindle",
        "myogen.simulator.neuron._cython._hill",
        "myogen.simulator.neuron._cython._gto",
        "myogen.simulator.neuron._cython._poisson_process_generator",
        "myogen.simulator.neuron._cython._gamma_process_generator",
        "myogen.simulator.neuron._cython._simulate_fiber",
    ]

    all_compiled = True
    if not force_rebuild:
        for module_name in cython_modules:
            try:
                __import__(module_name)
            except ImportError:
                all_compiled = False
                break
    else:
        all_compiled = False

    # Only compile Cython extensions if not already available
    if not all_compiled:
        if not quiet:
            print("Compiling Cython extensions (development mode)...")

        import os
        from pathlib import Path

        # Check if .pyx files exist (development install)
        myogen_root = Path(__file__).parent
        pyx_files_exist = (myogen_root / "simulator" / "neuron" / "_cython" / "_spindle.pyx").exists()

        if not pyx_files_exist:
            if not quiet:
                print("Cython source files not found. This is expected for installed packages.")
                print("Cython extensions should have been compiled during installation.")
            # Try importing again to give a clearer error if truly missing
            try:
                from myogen.simulator.neuron._cython import _spindle
                if not quiet:
                    print("Cython extensions are available.")
            except ImportError as e:
                if not quiet:
                    print(f"Error: Cython extensions are not available: {e}")
                    print("Please reinstall MyoGen or run setup from a development clone.")
                return False
        else:
            # Development mode: compile in-place
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
                            extra_compile_args=["-O3", "-march=native"],
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
                            extra_compile_args=["-O3", "-march=native", "-ffast-math"],
                        ),
                    ],
                    compiler_directives={"embedsignature": True},
                    nthreads=4,
                ),
                script_args=["build_ext", "--inplace"],
                include_dirs=[np.get_include()],
            )

            if not quiet:
                print("Cython extensions compiled successfully.")
    elif not quiet:
        print("Cython extensions already available.")

    # Compile NMODL files for NEURON
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
