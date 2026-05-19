"""
Windows CUDA DLL discovery for pip-installed nvidia-* packages.

CuPy 13 cannot locate DLLs shipped by ``pip install nvidia-cuda-nvrtc-cu12``
and similar wheels because Python 3.8+ no longer searches PATH for DLLs.
This module registers those directories via ``os.add_dll_directory`` and
pre-loads the NVRTC builtins library required for JIT compilation.

No-op on Linux / macOS and when the nvidia packages are absent.
"""

import sys


def setup() -> None:
    """Register CUDA DLL paths from pip-installed nvidia-* packages."""
    if sys.platform != "win32":
        return

    import ctypes
    import os
    import pathlib

    for site_dir in [pathlib.Path(p) for p in sys.path if "site-packages" in p]:
        # Register every nvidia/*/bin so cublas, cusolver, cufft etc. are found
        for bin_dir in site_dir.glob("nvidia/*/bin"):
            if bin_dir.is_dir():
                os.add_dll_directory(str(bin_dir))

        # Pre-load nvrtc-builtins (required by CuPy for JIT kernel compilation)
        for nvrtc_dll in site_dir.glob(
            "nvidia/cuda_nvrtc/bin/nvrtc-builtins*.dll"
        ):
            try:
                ctypes.WinDLL(str(nvrtc_dll))
            except OSError:
                pass
