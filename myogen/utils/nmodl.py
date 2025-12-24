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
import subprocess
from pathlib import Path
from typing import List, Optional


def find_nmodl_directory() -> Path:
    """Create isolated NMODL directory for MyoGen mechanisms."""
    # Use MyoGen's own nmodl_files directory for isolated compilation
    return Path(__file__).parent.parent / "simulator" / "nmodl_files"


def _get_mod_files(nmodl_path: Path) -> List[Path]:
    """Get .mod files from NMODL directory."""
    return list(nmodl_path.glob("*.mod"))


def _find_neuron_home_from_registry() -> Optional[Path]:
    """Find NEURON installation path from Windows Registry."""
    if platform.system() != "Windows":
        return None

    try:
        import winreg
    except ImportError:
        print("Warning: winreg module not available")
        return None

    # Registry keys to check (in order of preference)
    # Try common NEURON registry key patterns
    direct_neuron_keys = [
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\NEURON", winreg.KEY_READ | winreg.KEY_WOW64_64KEY),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\NEURON_Simulator", winreg.KEY_READ | winreg.KEY_WOW64_64KEY),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\NEURON", winreg.KEY_READ | winreg.KEY_WOW64_32KEY),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\NEURON_Simulator", winreg.KEY_READ | winreg.KEY_WOW64_32KEY),
        (winreg.HKEY_CURRENT_USER, r"SOFTWARE\NEURON", winreg.KEY_READ),
        (winreg.HKEY_CURRENT_USER, r"SOFTWARE\NEURON_Simulator", winreg.KEY_READ),
    ]

    # Check uninstall registry keys
    uninstall_keys = [
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall", winreg.KEY_READ | winreg.KEY_WOW64_64KEY),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall", winreg.KEY_READ | winreg.KEY_WOW64_32KEY),
        (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall", winreg.KEY_READ),
    ]

    registry_paths = direct_neuron_keys + uninstall_keys

    print("Searching for NEURON in Windows Registry...")

    for hkey, subkey_path, access_flag in registry_paths:
        try:
            with winreg.OpenKey(hkey, subkey_path, 0, access_flag) as key:
                # For direct NEURON keys, look for InstallPath or similar
                if "NEURON" in subkey_path and "Uninstall" not in subkey_path:
                    # Special handling for NEURON_Simulator - check nrn subkey
                    if "NEURON_Simulator" in subkey_path:
                        try:
                            with winreg.OpenKey(key, "nrn", 0, winreg.KEY_READ) as nrn_key:
                                install_path, _ = winreg.QueryValueEx(nrn_key, "Install_Dir")
                                if install_path:
                                    neuron_path = Path(install_path)
                                    if neuron_path.exists():
                                        print(f"  (OK) Found NEURON in registry: {neuron_path}")
                                        return neuron_path
                        except FileNotFoundError:
                            pass

                    # Try standard value names
                    try:
                        install_path, _ = winreg.QueryValueEx(key, "InstallPath")
                        if install_path:
                            neuron_path = Path(install_path)
                            if neuron_path.exists():
                                print(f"  (OK) Found NEURON in registry: {neuron_path}")
                                return neuron_path
                    except FileNotFoundError:
                        # Try alternative value names
                        for value_name in ["Path", "InstallLocation", "Install_Dir", ""]:
                            try:
                                install_path, _ = winreg.QueryValueEx(key, value_name)
                                if install_path:
                                    neuron_path = Path(install_path)
                                    if neuron_path.exists():
                                        print(f"  (OK) Found NEURON in registry: {neuron_path}")
                                        return neuron_path
                            except FileNotFoundError:
                                continue

                # For Uninstall keys, enumerate subkeys to find NEURON
                elif "Uninstall" in subkey_path:
                    num_subkeys = winreg.QueryInfoKey(key)[0]
                    for i in range(num_subkeys):
                        try:
                            subkey_name = winreg.EnumKey(key, i)
                            if "NEURON" in subkey_name.upper() or "NRN" in subkey_name.upper():
                                with winreg.OpenKey(key, subkey_name) as app_key:
                                    # Try InstallLocation first
                                    try:
                                        install_location, _ = winreg.QueryValueEx(app_key, "InstallLocation")
                                        if install_location:
                                            neuron_path = Path(install_location)
                                            if neuron_path.exists():
                                                print(f"  (OK) Found NEURON in registry (Uninstall): {neuron_path}")
                                                return neuron_path
                                    except FileNotFoundError:
                                        pass

                                    # Try to parse UninstallString as fallback
                                    try:
                                        uninstall_string, _ = winreg.QueryValueEx(app_key, "UninstallString")
                                        if uninstall_string:
                                            # Extract path from uninstall string (e.g., "c:\nrn\uninstall.exe")
                                            uninstall_path = Path(uninstall_string.strip('"'))
                                            neuron_path = uninstall_path.parent
                                            if neuron_path.exists():
                                                print(f"  (OK) Found NEURON in registry (UninstallString): {neuron_path}")
                                                return neuron_path
                                    except FileNotFoundError:
                                        pass
                        except OSError:
                            continue
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"  (X) Error checking registry path {subkey_path}: {e}")
            continue

    print("  (X) NEURON not found in registry")
    return None


def _find_mknrndll() -> Optional[Path]:
    """Find the mknrndll executable on Windows systems."""
    # First, try to find NEURON from Windows Registry
    neuron_home_from_registry = _find_neuron_home_from_registry()

    # Build list of possible locations, prioritizing registry-found path
    possible_locations = []

    if neuron_home_from_registry:
        possible_locations.extend([
            neuron_home_from_registry / "bin",
            neuron_home_from_registry / "mingw",
        ])

    # Add environment variable locations
    if os.environ.get("NEURONHOME"):
        possible_locations.extend([
            Path(os.environ.get("NEURONHOME", "")) / "bin",
            Path(os.environ.get("NEURONHOME", "")) / "mingw",
        ])

    # Add common hardcoded locations as fallback
    possible_locations.extend([
        Path("C:/nrn/bin"),
        Path("C:/Program Files/NEURON/bin"),
        Path("C:/Program Files (x86)/NEURON/bin"),
    ])

    print("Searching for mknrndll.bat in common locations...")
    for location in possible_locations:
        if location and location.parent.exists():  # Check if parent directory exists
            mknrndll_path = location / "mknrndll.bat"
            print(f"  Checking: {mknrndll_path}")
            if mknrndll_path.exists():
                print(f"  (OK) Found: {mknrndll_path}")
                return mknrndll_path
            else:
                print("  (X) Not found")

    # Try to find it in PATH
    print("Searching for mknrndll.bat in PATH...")
    try:
        result = subprocess.run(
            ["where", "mknrndll.bat"], capture_output=True, text=True, check=False
        )
        if result.returncode == 0:
            found_path = Path(result.stdout.strip())
            print(f"  (OK) Found in PATH: {found_path}")
            return found_path
        else:
            print("  (X) Not found in PATH")
    except Exception as e:
        print(f"  (X) Error searching PATH: {e}")

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
        # Run nrnivmodl from within the nmodl_files directory to keep output there
        result = subprocess.run(
            ["nrnivmodl", "."],
            cwd=nmodl_path,  # Changed from nmodl_path.parent to nmodl_path
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
        print("Error: nrnivmodl not found. Please ensure NEURON is properly installed.")
        raise


def compile_nmodl_files(quiet: bool = False) -> bool:
    """
    Compile NMODL files to shared libraries (run once during project setup).

    This function handles the compilation of .mod files into shared libraries
    that can be loaded by NEURON. It uses manual nrnivmodl compilation to avoid
    conflicts with PyNN's auto-loading mechanisms.

    Args:
        quiet: If True, suppress output messages

    Returns:
        bool: True if compilation succeeded, False otherwise
    """

    def log(msg):
        return print(msg) if not quiet else None

    try:
        nmodl_path = find_nmodl_directory()
        log(f"Compiling NMODL files from {nmodl_path}")

        mod_files = list(nmodl_path.glob("*.mod"))
        if not mod_files:
            log("Warning: No .mod files found to compile")
            return False

        log(f"Found {len(mod_files)} .mod files to compile")

        log("Using manual NMODL compilation")
        if platform.system() == "Windows":
            _compile_mod_files_windows(nmodl_path)
        else:
            _compile_mod_files_unix(nmodl_path)

        log("NMODL compilation complete!")
        return True

    except Exception as e:
        log(f"Error during NMODL compilation: {str(e)}")
        return False


def load_nmodl_mechanisms(quiet: bool = True) -> bool:
    """
    Load pre-compiled NMODL mechanisms into current NEURON session.

    This function loads previously compiled mechanisms into NEURON.
    It should be called at the start of every script that uses NEURON.

    Args:
        quiet: If True, suppress output messages

    Returns:
        bool: True if mechanisms loaded successfully, False otherwise
    """

    def log(msg):
        return print(msg) if not quiet else None

    # On Windows, add NEURON paths to PATH before importing
    if platform.system() == "Windows":
        # Priority: Registry > NEURONHOME env var > hardcoded C: paths
        neuron_home = _find_neuron_home_from_registry()

        if not neuron_home and os.environ.get("NEURONHOME"):
            neuron_home = Path(os.environ.get("NEURONHOME"))
            if not neuron_home.exists():
                neuron_home = None

        # Fallback to hardcoded C: drive paths
        if not neuron_home:
            neuron_homes_fallback = [
                Path("C:/nrn"),
                Path("C:/Program Files/NEURON"),
            ]
            for home in neuron_homes_fallback:
                if home.exists():
                    neuron_home = home
                    break

        if neuron_home:
            # Add both bin and lib/python directories
            neuron_bin = neuron_home / "bin"
            neuron_lib_path = neuron_home / "lib" / "python"

            paths_to_add = []
            if neuron_bin.exists():
                paths_to_add.append(str(neuron_bin))
            if neuron_lib_path.exists():
                paths_to_add.append(str(neuron_lib_path))

            if paths_to_add:
                current_path = os.environ.get("PATH", "")
                for path in paths_to_add:
                    if path not in current_path:
                        os.environ["PATH"] = f"{path};{os.environ['PATH']}"
                log(f"Added NEURON paths to PATH: {', '.join(paths_to_add)}")

    try:
        import neuron
        from neuron import h

        # Test if mechanisms are already loaded
        try:
            test_section = h.Section()
            test_section.insert("caL")
            test_section = None  # Clean up
            log("NMODL mechanisms already loaded, skipping reload")
            return True
        except Exception:
            pass  # Mechanisms not loaded, continue

        # Load mechanisms from MyoGen's nmodl directory
        nmodl_path = find_nmodl_directory()
        log(f"Loading NMODL mechanisms from {nmodl_path}")

        neuron.load_mechanisms(str(nmodl_path), warn_if_already_loaded=quiet)
        log("Successfully loaded NMODL mechanisms")
        return True

    except ImportError as e:
        print(f"Warning: NEURON not available, skipping mechanism loading: {str(e)}")
        return False
    except Exception as e:
        print(f"Error loading NMODL mechanisms: {str(e)}")
        return False
