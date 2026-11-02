"""Opt-in fixtures for synthetic execution contracts."""

from functools import partial

import pytest


@pytest.fixture
def ample_session_memory(monkeypatch):
    """Supply headroom to public-API tests without changing their policies.

    Only tests requesting this fixture replace the session's default probe.
    Explicit probes, memory ceilings, and the pressure/probe tests stay intact.
    """
    from myogen.simulator.core.emg.runtime import session
    from tests._memory import ample_memory

    monkeypatch.setattr(session, "MemoryController",
                        partial(session.MemoryController, probe=ample_memory))
