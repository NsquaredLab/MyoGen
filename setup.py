"""Build script for MyoGen with Cython extensions and NMODL compilation."""
from setuptools import setup
from setuptools.command.build_py import build_py
from Cython.Build import cythonize
from setuptools.extension import Extension
import numpy as np
import os
import platform
import subprocess
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
            # Compile in the build directory so files are included in the wheel
            build_nmodl_path = Path(self.build_lib) / "myogen" / "simulator" / "nmodl_files"

            if not build_nmodl_path.exists():
                print("Warning: NMODL files directory not found in build, skipping NMODL compilation")
                return

            mod_files = list(build_nmodl_path.glob("*.mod"))
            if not mod_files:
                print("Warning: No .mod files found, skipping NMODL compilation")
                return

            print(f"Compiling {len(mod_files)} NMODL files...")

            # Try to compile based on platform
            if platform.system() == "Windows":
                self._compile_nmodl_windows(build_nmodl_path)
            else:
                self._compile_nmodl_unix(build_nmodl_path)

            print("NMODL compilation complete!")

        except Exception as e:
            if platform.system() == "Windows":
                # On Windows, NEURON is required - fail the build with clear instructions
                error_msg = (
                    "\n" + "="*70 + "\n"
                    "ERROR: NEURON installation required for MyoGen on Windows\n"
                    "="*70 + "\n\n"
                    "MyoGen requires NEURON to be installed before building.\n\n"
                    "Please install NEURON by following these steps:\n"
                    "1. Download NEURON from: https://neuron.yale.edu/neuron/download\n"
                    "2. Run the Windows installer (nrn-X.X.X.w64-mingw-py-XX-setup.exe)\n"
                    "3. After installation, retry: pip install myogen\n\n"
                    f"Original error: {e}\n"
                    "="*70 + "\n"
                )
                raise RuntimeError(error_msg) from e
            else:
                # On Linux/macOS, allow optional NMODL compilation
                print(f"Warning: NMODL compilation failed (this is optional): {e}")
                print("You can compile NMODL files later by running: from myogen import _setup_myogen; _setup_myogen()")

    def _compile_nmodl_windows(self, nmodl_path):
        """Compile NMODL on Windows."""
        # Try to find NEURON installation first
        neuron_homes = [
            Path(os.environ.get("NEURONHOME", "")),
            Path("C:/nrn"),
            Path("C:/Program Files/NEURON"),
        ]

        neuron_home = None
        for home in neuron_homes:
            if home.exists():
                neuron_home = home
                break

        if not neuron_home:
            print("\nWARNING: NEURON installation directory not found")
            print("Installation will continue without compiling NEURON mechanisms")
            return

        # Add NEURON's bin directory to DLL search path (Python 3.8+)
        neuron_bin = neuron_home / "bin"
        if neuron_bin.exists():
            try:
                # This is the proper way to add DLL directories on Windows
                os.add_dll_directory(str(neuron_bin))
                print(f"Added NEURON DLL directory: {neuron_bin}")
            except (AttributeError, OSError) as e:
                print(f"Could not add DLL directory: {e}")

        # Now try to import NEURON
        try:
            import neuron
            from neuron import h
            print("NEURON imported successfully")
        except ImportError as e:
            print("\n" + "="*70)
            print("WARNING: NEURON import failed")
            print("="*70)
            print(f"\nError: {e}")
            print(f"NEURON home: {neuron_home}")
            print(f"NEURON bin: {neuron_bin}")
            print("\nInstallation will continue without compiling NEURON mechanisms")
            print("You can compile them later by running:")
            print("  python -c \"from myogen import _setup_myogen; _setup_myogen()\"")
            print("="*70 + "\n")
            return

        # Verify mknrndll.bat exists
        mknrndll_path = neuron_home / "bin" / "mknrndll.bat"
        if not mknrndll_path.exists():
            print(f"\nWARNING: mknrndll.bat not found at {mknrndll_path}")
            print("Installation will continue without compiling NEURON mechanisms")
            return

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
