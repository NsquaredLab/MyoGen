"""
Shared helper for locating a NEURON installation on Windows via the Registry.

This module is the **single source of truth** for the Windows Registry lookup
logic. It is imported by:

* ``myogen/utils/nmodl.py`` at runtime, via a normal package import.
* ``setup.py`` during ``pip``/``uv`` wheel builds, via ``importlib.util`` to
  bypass ``myogen/__init__.py`` (which performs heavy imports — numpy, NEURON
  mechanism loading, etc. — that are not available in the PEP 517 build
  isolation environment).

Because of the second consumer, this module **must remain stdlib-only**. The
build isolation env only ships ``setuptools``, ``Cython``, ``numpy``, and
``scipy``.
"""

import platform
from pathlib import Path
from typing import Optional


# Module-level cache so repeated lookups (e.g. during _setup_myogen +
# load_nmodl_mechanisms) don't re-scan the Registry.
_neuron_home_cache: Optional[Path] = None
_neuron_home_cache_checked: bool = False


# Imported by both ``setup.py`` (via importlib.util to skip myogen/__init__.py)
# and ``myogen/utils/nmodl.py`` (via normal package import).
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
