"""
Initialize and set up NMODL (NEURON MODeling Language) files for the model.

This module handles the compilation and loading of NMODL files, which are used to define
custom mechanisms and models in NEURON simulations. It performs the following steps:
1. Locates and copies NMODL files to the appropriate directory
2. Compiles the NMODL files (platform-specific approach)
3. Loads the compiled files into NEURON

The module is automatically executed when the package is imported.
"""

import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

from neuron import h


def find_nmodl_directory() -> Path:
    """Create isolated NMODL directory for MyoGen mechanisms."""
    # Use MyoGen's own nmodl_files directory for isolated compilation
    src_path = Path(__file__).parent.parent / "simulator" / "nmodl_files"
    return src_path


def _get_mod_files(nmodl_path: Path) -> List[Path]:
    """Get .mod files from NMODL directory."""
    mod_files = list(nmodl_path.glob("*.mod"))
    if not mod_files:
        print("Warning: No .mod files found in NMODL directory")
        return []

    print(f"Found {len(mod_files)} .mod files to compile")
    for mod_file in mod_files:
        print(f"  {mod_file.name}")

    return mod_files


def _find_mknrndll() -> Optional[Path]:
    """Find the mknrndll executable on Windows systems."""
    # Common locations for mknrndll
    possible_locations = [
        Path(os.environ.get("NEURONHOME", "")) / "bin",
        Path(os.environ.get("NEURONHOME", "")) / "mingw",
        Path("C:/nrn/bin"),
        Path("C:/Program Files/NEURON/bin"),
        Path("C:/Program Files (x86)/NEURON/bin"),
    ]

    print("Searching for mknrndll.bat in common locations...")
    for location in possible_locations:
        if location and location.parent.exists():  # Check if parent directory exists
            mknrndll_path = location / "mknrndll.bat"
            print(f"  Checking: {mknrndll_path}")
            if mknrndll_path.exists():
                print(f"  ✓ Found: {mknrndll_path}")
                return mknrndll_path
            else:
                print(f"  ✗ Not found")

    # Try to find it in PATH
    print("Searching for mknrndll.bat in PATH...")
    try:
        result = subprocess.run(
            ["where", "mknrndll.bat"], capture_output=True, text=True, check=False
        )
        if result.returncode == 0:
            found_path = Path(result.stdout.strip())
            print(f"  ✓ Found in PATH: {found_path}")
            return found_path
        else:
            print("  ✗ Not found in PATH")
    except Exception as e:
        print(f"  ✗ Error searching PATH: {e}")

    print("mknrndll.bat not found. Please ensure NEURON is properly installed.")
    return None


def _compile_mod_files_windows(nmodl_path: Path) -> None:
    """Compile NMODL files on Windows using mknrndll."""
    mknrndll_path = _find_mknrndll()

    if mknrndll_path is None:
        raise FileNotFoundError(
            "Could not find mknrndll.bat. Please make sure NEURON is properly installed "
            "and NEURONHOME environment variable is set correctly."
        )

    print(f"Using mknrndll: {mknrndll_path}")

    # Change to the directory containing the mod files and run mknrndll.bat
    original_dir = os.getcwd()
    try:
        os.chdir(nmodl_path)

        # Remove any existing DLL files to avoid conflicts
        for dll_file in nmodl_path.glob("*nrnmech.dll"):
            try:
                dll_file.unlink()
                print(f"Removed existing DLL: {dll_file.name}")
            except Exception as e:
                print(f"Warning: Could not remove {dll_file.name}: {e}")

        # On Windows, we need to use cmd.exe to run batch files
        cmd = ["cmd", "/c", str(mknrndll_path)]
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)

        # Check if stderr has any warnings (not necessarily errors)
        if result.stderr:
            print(f"Compilation warnings/info: {result.stderr}")

    except subprocess.CalledProcessError as e:
        print(f"Error during compilation: {e.stderr}")
        print(f"Stdout: {e.stdout}")
        raise
    finally:
        os.chdir(original_dir)


def _compile_mod_files_unix(nmodl_path: Path) -> None:
    """Compile NMODL files on Unix-like systems using nrnivmodl."""
    try:
        print(f"Compiling NMODL files from {nmodl_path}")
        # Use nrnivmodl directly for better control
        result = subprocess.run(
            ["nrnivmodl", str(nmodl_path)],
            cwd=nmodl_path.parent,
            capture_output=True,
            text=True,
            check=True,
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"Compilation warnings: {result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to compile NMODL files: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        raise
    except FileNotFoundError:
        # Fallback to pyNN utility
        try:
            from pyNN.utility.build import compile_nmodl

            compile_nmodl(nmodl_path)
        except ImportError:
            print(
                "Error: Neither nrnivmodl nor pyNN.utility.build.compile_nmodl available"
            )
            raise


def _compile_and_load_mod_files(
    nmodl_path: Path, mod_files: List[Path], quiet: bool = False
) -> None:
    """Compile and load NMODL files into NEURON based on platform."""
    if not mod_files:
        if not quiet:
            print("No mod files to compile")
        return

    # Platform-specific compilation
    if platform.system() == "Windows":
        _compile_mod_files_windows(nmodl_path)
        # Load Windows DLL
        dll_files = list(nmodl_path.glob("*nrnmech.dll"))
        if dll_files:
            dll_path = dll_files[0]
            try:
                h.nrn_load_dll(str(dll_path))
                if not quiet:
                    print(f"Successfully loaded {dll_path.name}")
            except Exception as e:
                if not quiet and "already exists" not in str(e):
                    print(f"Warning: Error loading {dll_path.name}: {str(e)}")
        else:
            if not quiet:
                print(f"Warning: No nrnmech.dll file found after compilation")
    else:
        _compile_mod_files_unix(nmodl_path)
        # On Unix, look for the compiled shared library in the most likely locations
        possible_lib_paths = [
            nmodl_path / "x86_64" / "libnrnmech.so",
            nmodl_path.parent / "x86_64" / "libnrnmech.so",
            nmodl_path / "x86_64" / ".libs" / "libnrnmech.so",
            nmodl_path.parent / "x86_64" / ".libs" / "libnrnmech.so",
        ]

        lib_loaded = False
        for lib_path in possible_lib_paths:
            if lib_path.exists() and not lib_loaded:
                try:
                    if not quiet:
                        print(f"Loading compiled library: {lib_path}")
                    h.nrn_load_dll(str(lib_path))
                    if not quiet:
                        print(f"Successfully loaded {lib_path}")
                    lib_loaded = True
                    break
                except Exception as e:
                    # Only show warnings if not about duplicates and not in quiet mode
                    if (
                        not quiet
                        and "already exists" not in str(e)
                        and "hocobj_call error" not in str(e)
                    ):
                        print(f"Warning: Failed to load {lib_path}: {str(e)}")

        if not lib_loaded and not quiet:
            print("Warning: Could not find or load compiled shared library")
            print("Available directories and files:")
            for item in nmodl_path.parent.iterdir():
                print(f"  {item.name} ({'dir' if item.is_dir() else 'file'})")


def load_nmodl_files(force_reload: bool = False, quiet: bool = False):
    """
    Main function to handle NMODL file setup using PyNN's load_mechanisms.

    This function first attempts to use PyNN's built-in mechanism loading
    functionality before falling back to the manual compilation approach.
    PyNN's load_mechanisms is more robust and handles compilation automatically.

    Args:
        force_reload: If True, force recompilation even if mechanisms seem loaded
        quiet: If True, suppress most output messages
    """

    def log(message: str):
        if not quiet:
            print(message)

    # Check if mechanisms are already loaded
    if not force_reload:
        try:
            from neuron import h

            # Test if our custom mechanisms are already loaded by trying to use them
            test_section = h.Section()
            mechanisms_working = True

            try:
                # Try to insert and use our key mechanisms
                test_section.insert("motoneuron")
                test_section.insert("caL")
                test_section.insert("kdrRL")

                # Clean up
                test_section.uninsert("motoneuron")
                test_section.uninsert("caL")
                test_section.uninsert("kdrRL")

                if not quiet:
                    log("NMODL mechanisms already loaded and working, skipping reload")
                return True

            except Exception:
                mechanisms_working = False
            finally:
                # Clean up test section
                test_section = None

        except ImportError:
            log("Warning: NEURON not available, skipping NMODL loading")
            return False
        except Exception as e:
            if not quiet:
                log(f"Warning: Error checking mechanism status: {e}")

    # If we get here, mechanisms need to be loaded
    try:
        nmodl_path = find_nmodl_directory()
        if not quiet:
            log(f"Loading NMODL files from {nmodl_path}")

        mod_files = _get_mod_files(nmodl_path)

        if mod_files:
            if not quiet:
                log(f"Found {len(mod_files)} .mod files to compile")
                for mod_file in mod_files:
                    log(f"  {mod_file.name}")

            # First try using PyNN's load_mechanisms function
            try:
                from pyNN.neuron.simulator import load_mechanisms

                if not quiet:
                    log("Using PyNN's load_mechanisms functionality")
                load_mechanisms(str(nmodl_path))
                if not quiet:
                    log("Successfully loaded mechanisms using PyNN load_mechanisms")
                return True

            except ImportError:
                if not quiet:
                    log("PyNN not available, falling back to manual compilation")
            except Exception as e:
                if not quiet:
                    log(
                        f"PyNN load_mechanisms failed: {e}, falling back to manual compilation"
                    )

            # Fallback to manual compilation
            if not quiet:
                log("Using manual NMODL compilation")
            _compile_and_load_mod_files(nmodl_path, mod_files, quiet=quiet)
            if not quiet:
                log("NMODL files processing complete!")
            return True
        else:
            if not quiet:
                log("Warning: No NMODL files were processed")
            return False

    except Exception as e:
        if not quiet:
            log(f"Error during NMODL setup: {str(e)}")
            import traceback

            traceback.print_exc()
        return False


# load_nmodl_files()
