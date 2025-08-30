import numpy as np
from Cython.Build import cythonize
from numpy.random import Generator
from setuptools import Extension, setup

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


def setup_myogen(
    force_reload: bool = True,
    quiet: bool = False,
) -> bool:
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

    Examples
    --------
    Basic setup with standard MyoGen mechanisms:

    >>> import myogen
    >>> myogen.setup_myogen()

    Setup with spinal circuits for proprioceptive modeling:

    >>> import myogen
    >>> myogen.setup_myogen(enable_spinal_circuits=True)

    Notes
    -----
    This function should be called before using any NEURON-based simulation
    components. It will automatically detect the platform and use appropriate
    compilation tools, prioritizing PyNN's load_mechanisms() for better integration.

    When enable_spinal_circuits=True, additional mechanisms become available:
    - Ion channels: na3rp, naps, kdrRL, gh, mAHP, L_Ca_inact, caL, napp
    - Synaptic: Gfluctdv, vecevent
    - Stimulation: GammaStim, nsloc, izap, constant, dummy

    For PyNN integration, the function will attempt to use pyNN.neuron.simulator.load_mechanisms()
    which provides better compatibility with PyNN simulations and handles compilation automatically.
    """
    setup(
        ext_modules=cythonize(
            [
                Extension(
                    "myogen.simulator.neuron._cython._spindle",
                    ["myogen/simulator/neuron/_cython/_spindle.pyx"],
                ),
                Extension(
                    "myogen.simulator.neuron._cython._hill",
                    ["myogen/simulator/neuron/_cython/_hill.pyx"],
                ),
                Extension(
                    "myogen.simulator.neuron._cython._gto",
                    ["myogen/simulator/neuron/_cython/_gto.pyx"],
                ),
                Extension(
                    "myogen.simulator.neuron._cython._emg",
                    ["myogen/simulator/neuron/_cython/_emg.pyx"],
                ),
                Extension(
                    "myogen.simulator.neuron._cython._poisson_process_generator",
                    ["myogen/simulator/neuron/_cython/_poisson_process_generator.pyx"],
                ),
            ]
        ),
        script_args=["build_ext", "--inplace"],
        include_dirs=[np.get_include()],
    )

    try:
        from myogen.utils.nmodl import load_nmodl_files

        # Note: load_nmodl_files now includes PyNN integration by default
        return load_nmodl_files(force_reload=force_reload, quiet=quiet)
    except ImportError as e:
        if not quiet:
            print(f"Warning: NEURON not available, skipping mechanism setup: {e}")
        return False
    except Exception as e:
        if not quiet:
            print(f"Error during MyoGen setup: {e}")
        return False
