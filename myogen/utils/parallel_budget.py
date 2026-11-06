"""Estimate safe parallelism for NEURON-based DD optimization trials.

Each Optuna trial spawns a full NEURON simulation with AlphaMN__Pool +
DescendingDrive__Pool + Network.  NEURON uses global state, so parallel
trials must run in separate **processes** (not threads).  Each process
needs its own memory for:

- AlphaMN pool: ~1.5 MB per MN (compartmental model + ion channels)
- DescendingDrive pool: ~0.1 MB per DD neuron
- Network connectivity: ~0.5 MB per MN (synaptic connections)
- Spike recorders + drive signal: ~10 MB overhead
- Python/NEURON runtime: ~200 MB base per process

Usage::

    from myogen.utils.parallel_budget import estimate_safe_njobs

    n_jobs = estimate_safe_njobs(n_mu=300, dd_neurons=280)
    study.optimize(objective, n_trials=100, n_jobs=n_jobs)
"""

import os
import platform


def _get_system_memory_gb() -> float:
    """Return total system RAM in GB."""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024**3)
    except ImportError:
        pass

    # Fallback for Windows without psutil
    if platform.system() == "Windows":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            c_ulonglong = ctypes.c_ulonglong
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", c_ulonglong),
                    ("ullAvailPhys", c_ulonglong),
                    ("ullTotalPageFile", c_ulonglong),
                    ("ullAvailPageFile", c_ulonglong),
                    ("ullTotalVirtual", c_ulonglong),
                    ("ullAvailVirtual", c_ulonglong),
                    ("ullAvailExtendedVirtual", c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullTotalPhys / (1024**3)
        except Exception:
            pass

    # Last resort: assume 16 GB
    return 16.0


def _get_available_memory_gb() -> float:
    """Return available (free) RAM in GB."""
    try:
        import psutil
        return psutil.virtual_memory().available / (1024**3)
    except ImportError:
        pass

    if platform.system() == "Windows":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            c_ulonglong = ctypes.c_ulonglong
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", c_ulonglong),
                    ("ullAvailPhys", c_ulonglong),
                    ("ullTotalPageFile", c_ulonglong),
                    ("ullAvailPageFile", c_ulonglong),
                    ("ullTotalVirtual", c_ulonglong),
                    ("ullAvailVirtual", c_ulonglong),
                    ("ullAvailExtendedVirtual", c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullAvailPhys / (1024**3)
        except Exception:
            pass

    return 8.0


def estimate_trial_memory_gb(
    n_mu: int = 300,
    dd_neurons: int = 280,
) -> float:
    """Estimate memory footprint of one DD optimization trial in GB.

    Based on empirical measurements of NEURON AlphaMN pool simulations.

    Parameters
    ----------
    n_mu : int
        Number of motor units (AlphaMN cells).
    dd_neurons : int
        Number of descending drive neurons.

    Returns
    -------
    float
        Estimated memory per trial process in GB.
    """
    base_process_gb = 0.25  # Python + NEURON runtime overhead
    per_mn_mb = 1.5         # compartmental model + channels + synapses
    per_dd_mb = 0.1         # DD gamma cell (lightweight)
    overhead_mb = 50        # spike recorders, drive signal, network state

    trial_mb = (
        base_process_gb * 1024
        + n_mu * per_mn_mb
        + dd_neurons * per_dd_mb
        + overhead_mb
    )
    return trial_mb / 1024


def estimate_safe_njobs(
    n_mu: int = 300,
    dd_neurons: int = 280,
    reserve_gb: float = 4.0,
    max_jobs: int | None = None,
    verbose: bool = True,
) -> int:
    """Estimate the number of parallel Optuna trials that can run safely.

    Checks available RAM and CPU cores, estimates per-trial memory,
    and returns a conservative ``n_jobs`` value that leaves headroom
    for the OS and other applications.

    Parameters
    ----------
    n_mu : int
        Motor units per trial.
    dd_neurons : int
        DD neurons per trial (used in memory estimate).
    reserve_gb : float
        RAM to reserve for OS + other applications (default 4 GB).
    max_jobs : int or None
        Hard cap on parallelism. None = no cap beyond hardware.
    verbose : bool
        Print the estimation breakdown.

    Returns
    -------
    int
        Recommended ``n_jobs`` for ``study.optimize()``. Always >= 1.
    """
    total_ram = _get_system_memory_gb()
    available_ram = _get_available_memory_gb()
    cpu_cores = os.cpu_count() or 4
    trial_mem = estimate_trial_memory_gb(n_mu, dd_neurons)

    # Use available RAM minus reserve, not total
    usable_ram = max(available_ram - reserve_gb, trial_mem)
    n_by_ram = int(usable_ram / trial_mem)

    # Leave at least 1 core for OS + responsiveness
    n_by_cpu = max(cpu_cores - 1, 1)

    n_jobs = max(1, min(n_by_ram, n_by_cpu))

    if max_jobs is not None:
        n_jobs = min(n_jobs, max_jobs)

    if verbose:
        print(f"  Parallel budget estimation:")
        print(f"    System RAM: {total_ram:.1f} GB total, "
              f"{available_ram:.1f} GB available")
        print(f"    CPU cores: {cpu_cores}")
        print(f"    Per-trial memory: ~{trial_mem:.2f} GB "
              f"({n_mu} MNs + {dd_neurons} DD neurons)")
        print(f"    Reserve: {reserve_gb:.1f} GB")
        print(f"    Limit by RAM: {n_by_ram} trials")
        print(f"    Limit by CPU: {n_by_cpu} trials")
        print(f"    => Recommended n_jobs = {n_jobs}")

    return n_jobs
