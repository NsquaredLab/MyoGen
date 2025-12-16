def __getattr__(name):
    """Lazy import to avoid loading NEURON when importing _cython submodules."""
    if name == "Network":
        from myogen.simulator.neuron.network import Network
        return Network
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["Network"]
