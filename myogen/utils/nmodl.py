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


# Cache for registry lookup (module-level to persist across calls)
_neuron_home_cache: Optional[Path] = None
_neuron_home_cache_checked: bool = False


def _find_neuron_home_from_registry(quiet: bool = True) -> Optional[Path]:
    """Find NEURON installation path from Windows Registry."""
    global _neuron_home_cache, _neuron_home_cache_checked

    # Return cached result if already searched
    if _neuron_home_cache_checked:
        return _neuron_home_cache

    if platform.system() != "Windows":
        _neuron_home_cache_checked = True
        return None

    try:
        import winreg
    except ImportError:
        if not quiet:
            print("Warning: winreg module not available")
        _neuron_home_cache_checked = True
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

    if not quiet:
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
                                        if not quiet:
                                            print(f"  (OK) Found NEURON in registry: {neuron_path}")
                                        _neuron_home_cache = neuron_path
                                        _neuron_home_cache_checked = True
                                        return neuron_path
                        except FileNotFoundError:
                            pass

                    # Try standard value names
                    try:
                        install_path, _ = winreg.QueryValueEx(key, "InstallPath")
                        if install_path:
                            neuron_path = Path(install_path)
                            if neuron_path.exists():
                                if not quiet:
                                    print(f"  (OK) Found NEURON in registry: {neuron_path}")
                                _neuron_home_cache = neuron_path
                                _neuron_home_cache_checked = True
                                return neuron_path
                    except FileNotFoundError:
                        # Try alternative value names
                        for value_name in ["Path", "InstallLocation", "Install_Dir", ""]:
                            try:
                                install_path, _ = winreg.QueryValueEx(key, value_name)
                                if install_path:
                                    neuron_path = Path(install_path)
                                    if neuron_path.exists():
                                        if not quiet:
                                            print(f"  (OK) Found NEURON in registry: {neuron_path}")
                                        _neuron_home_cache = neuron_path
                                        _neuron_home_cache_checked = True
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
                                                if not quiet:
                                                    print(f"  (OK) Found NEURON in registry (Uninstall): {neuron_path}")
                                                _neuron_home_cache = neuron_path
                                                _neuron_home_cache_checked = True
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
                                                if not quiet:
                                                    print(f"  (OK) Found NEURON in registry (UninstallString): {neuron_path}")
                                                _neuron_home_cache = neuron_path
                                                _neuron_home_cache_checked = True
                                                return neuron_path
                                    except FileNotFoundError:
                                        pass
                        except OSError:
                            continue
        except FileNotFoundError:
            continue
        except Exception as e:
            if not quiet:
                print(f"  (X) Error checking registry path {subkey_path}: {e}")
            continue

    if not quiet:
        print("  (X) NEURON not found in registry")
    _neuron_home_cache_checked = True
    return None


def _find_mknrndll() -> Optional[Path]:
    """Find the mknrndll executable on Windows systems."""
    # First, try to find NEURON from Windows Registry
    # Show output during build process
    neuron_home_from_registry = _find_neuron_home_from_registry(quiet=False)

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


class NMODLLoadError(Exception):
    """Exception raised when NMODL mechanisms fail to load."""

    pass


def load_nmodl_mechanisms(quiet: bool = True, strict: bool = False) -> bool:
    """
    Load pre-compiled NMODL mechanisms into current NEURON session.

    This function loads previously compiled mechanisms into NEURON.
    It should be called at the start of every script that uses NEURON.

    Args:
        quiet: If True, suppress output messages
        strict: If True, raise exceptions on errors instead of returning False.
                Recommended for production code to catch configuration issues early.

    Returns:
        bool: True if mechanisms loaded successfully, False otherwise

    Raises:
        NMODLLoadError: If strict=True and mechanisms fail to load
    """

    def log(msg):
        return print(msg) if not quiet else None

    def error(msg):
        """Handle errors based on strict mode."""
        if strict:
            raise NMODLLoadError(msg)
        else:
            print(f"WARNING: {msg}")
            return False

    # On Windows, add NEURON paths to PATH before importing
    if platform.system() == "Windows":
        # Priority: Registry > NEURONHOME env var > hardcoded C: paths
        neuron_home = _find_neuron_home_from_registry(quiet=quiet)

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
        return error(f"NEURON not available, cannot load mechanisms: {str(e)}")
    except Exception as e:
        return error(f"Failed to load NMODL mechanisms: {str(e)}")


def get_mechanism_parameters(mechanism_name: str) -> list:
    """
    Get list of valid parameters for a NEURON mechanism.

    Args:
        mechanism_name: Name of the mechanism (e.g., "mAHP", "na3rp")

    Returns:
        List of parameter names available for this mechanism

    Raises:
        ValueError: If mechanism is not loaded or doesn't exist
    """
    from neuron import h

    # Create a temporary section to inspect the mechanism
    temp_section = h.Section()
    try:
        temp_section.insert(mechanism_name)
    except Exception as e:
        raise ValueError(f"Mechanism '{mechanism_name}' not found. Is it loaded? Error: {e}")

    seg = temp_section(0.5)
    mech = getattr(seg, mechanism_name)

    # Get all non-private, non-callable attributes
    params = []
    for attr in dir(mech):
        if not attr.startswith("_"):
            try:
                val = getattr(mech, attr)
                if not callable(val):
                    params.append(attr)
            except Exception:
                pass

    # Clean up
    temp_section = None

    return params


def validate_mechanism_parameter(section, param_name: str, mechanism_name: str = None) -> None:
    """
    Validate that a parameter exists on a section before setting it.

    Args:
        section: NEURON Section object
        param_name: Parameter name (e.g., "gcamax_mAHP" or just "gcamax" with mechanism_name)
        mechanism_name: Optional mechanism name if param_name doesn't include suffix

    Raises:
        AttributeError: If the parameter doesn't exist on the section

    Example:
        >>> validate_mechanism_parameter(soma, "gcamax_mAHP")  # Validates before setting
        >>> soma.gcamax_mAHP = 1e-5  # Now safe to set
    """
    # If mechanism name provided, construct full parameter name
    if mechanism_name:
        full_param = f"{param_name}_{mechanism_name}"
    else:
        full_param = param_name

    # Try to access the parameter - will raise AttributeError if invalid
    try:
        _ = getattr(section, full_param)
    except AttributeError:
        # Extract mechanism name from param if not provided (e.g., "gcamax_mAHP" -> "mAHP")
        inferred_mech = None
        if "_" in full_param and not mechanism_name:
            parts = full_param.rsplit("_", 1)
            if len(parts) == 2:
                inferred_mech = parts[1]

        # Get available parameters for the mechanism
        available = []
        mech_to_check = mechanism_name or inferred_mech
        mech_found = False

        if mech_to_check:
            try:
                available = get_mechanism_parameters(mech_to_check)
                available = [f"{p}_{mech_to_check}" for p in available]
                mech_found = True
            except ValueError:
                # Mechanism doesn't exist - will suggest alternatives below
                pass

        # Fallback: show all section params
        if not available:
            for attr in dir(section):
                if not attr.startswith("_") and not callable(getattr(section, attr, None)):
                    available.append(attr)

        # Build helpful error message
        msg = f"Parameter '{full_param}' does not exist on section.\n"

        # If mechanism wasn't found, suggest similar ones
        if mech_to_check and not mech_found:
            # Get list of inserted mechanisms
            inserted = []
            seg = section(0.5)
            for mech in seg:
                inserted.append(mech.name())
            if inserted:
                msg += f"Inserted mechanisms: {inserted}\n"
                # Check for similar mechanism names (simple prefix match)
                if len(mech_to_check) >= 3:
                    similar = [m for m in inserted if m.lower().startswith(mech_to_check[:3].lower())]
                    if similar:
                        msg += f"Did you mean: {similar}?\n"

        msg += f"Available parameters{' for ' + mech_to_check if mech_found else ''}: {available}"
        raise AttributeError(msg)


def set_mechanism_param(section, param_name: str, value, validate: bool = True) -> None:
    """
    Set a mechanism parameter with optional validation.

    Args:
        section: NEURON Section object
        param_name: Parameter name (e.g., "gcamax_mAHP")
        value: Value to set
        validate: If True, validate parameter exists first (default: True)

    Raises:
        AttributeError: If validate=True and parameter doesn't exist

    Example:
        >>> set_mechanism_param(soma, "gcamax_mAHP", 1e-5)  # Safe setting
        >>> set_mechanism_param(soma, "gcamax_mAHPr", 1e-5)  # Raises AttributeError (typo)
    """
    if validate:
        validate_mechanism_parameter(section, param_name)
    setattr(section, param_name, value)
