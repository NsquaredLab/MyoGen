"""Build script for MyoGen with Cython extensions and NMODL compilation."""
from setuptools import setup
from setuptools.command.build_py import build_py
from Cython.Build import cythonize
from setuptools.extension import Extension
import numpy as np
import os
from pathlib import Path


class BuildWithNMODL(build_py):
    """Custom build command that compiles NMODL files after building Python modules."""

    def run(self):
        # Run the standard build
        super().run()

        # Compile NMODL files
        self.compile_nmodl()

    def compile_nmodl(self):
        """Compile NMODL files if NEURON is available."""
        try:
            # Import the compilation function from myogen
            import platform
            import subprocess

            nmodl_path = Path("myogen") / "simulator" / "nmodl_files"

            if not nmodl_path.exists():
                print("Warning: NMODL files directory not found, skipping NMODL compilation")
                return

            mod_files = list(nmodl_path.glob("*.mod"))
            if not mod_files:
                print("Warning: No .mod files found, skipping NMODL compilation")
                return

            print(f"Compiling {len(mod_files)} NMODL files...")

            # Try to compile based on platform
            if platform.system() == "Windows":
                self._compile_nmodl_windows(nmodl_path)
            else:
                self._compile_nmodl_unix(nmodl_path)

            print("NMODL compilation complete!")

        except Exception as e:
            print(f"Warning: NMODL compilation failed (this is optional): {e}")
            print("You can compile NMODL files later by running: from myogen import _setup_myogen; _setup_myogen()")

    def _compile_nmodl_windows(self, nmodl_path):
        """Compile NMODL on Windows."""
        # Try to find NEURON installation
        neuron_homes = [
            Path(os.environ.get("NEURONHOME", "")),
            Path("C:/nrn"),
            Path("C:/Program Files/NEURON"),
        ]

        neuron_home = None
        for home in neuron_homes:
            if home.exists() and (home / "bin" / "mknrndll.bat").exists():
                neuron_home = home
                break

        if not neuron_home:
            raise FileNotFoundError("mknrndll.bat not found - NEURON may not be installed")

        mknrndll_path = neuron_home / "bin" / "mknrndll.bat"

        # Set up environment with NEURON paths
        env = os.environ.copy()
        neuron_lib_path = str(neuron_home / "lib" / "python")

        # Add NEURON lib/python to PATH for DLL loading
        if "PATH" in env:
            env["PATH"] = f"{neuron_lib_path};{env['PATH']}"
        else:
            env["PATH"] = neuron_lib_path

        # Change to nmodl directory and compile
        original_dir = os.getcwd()
        try:
            os.chdir(nmodl_path)
            # Remove existing DLLs
            for dll_file in nmodl_path.glob("*nrnmech.dll"):
                dll_file.unlink()

            subprocess.run(
                ["cmd", "/c", str(mknrndll_path)],
                capture_output=True,
                text=True,
                check=True,
                env=env  # Use modified environment
            )
        finally:
            os.chdir(original_dir)

    def _compile_nmodl_unix(self, nmodl_path):
        """Compile NMODL on Unix-like systems."""
        import subprocess
        subprocess.run(
            ["nrnivmodl", "."],
            cwd=nmodl_path,
            capture_output=True,
            text=True,
            check=True
        )


# Define the Cython extensions
extensions = [
    Extension(
        "myogen.simulator.neuron._cython._spindle",
        ["myogen/simulator/neuron/_cython/_spindle.pyx"],
        extra_compile_args=["-O3"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        "myogen.simulator.neuron._cython._hill",
        ["myogen/simulator/neuron/_cython/_hill.pyx"],
        extra_compile_args=["-O3"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        "myogen.simulator.neuron._cython._gto",
        ["myogen/simulator/neuron/_cython/_gto.pyx"],
        extra_compile_args=["-O3"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        "myogen.simulator.neuron._cython._poisson_process_generator",
        ["myogen/simulator/neuron/_cython/_poisson_process_generator.pyx"],
        extra_compile_args=["-O3"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        "myogen.simulator.neuron._cython._gamma_process_generator",
        ["myogen/simulator/neuron/_cython/_gamma_process_generator.pyx"],
        extra_compile_args=["-O3"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        "myogen.simulator.neuron._cython._simulate_fiber",
        ["myogen/simulator/neuron/_cython/_simulate_fiber.pyx"],
        extra_compile_args=["-O3"],
        include_dirs=[np.get_include()],
    ),
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={"embedsignature": True, "language_level": "3"},
        nthreads=4,
    ),
    cmdclass={
        'build_py': BuildWithNMODL,
    },
)
