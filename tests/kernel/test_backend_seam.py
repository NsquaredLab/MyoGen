from __future__ import annotations

import numpy as np

from myogen.kernel.protocols import Backend, Stage
from myogen.kernel.simulation import Simulation
from tests.kernel._doubles import BackendStage, CountingBackend


def test_stage_delegates_to_backend_through_driver():
    backend = CountingBackend()
    stage = BackendStage(backend)
    assert isinstance(stage, Stage)
    assert isinstance(backend, Backend)

    sim = Simulation(stage, n_units=1, dt_s=0.01, n_steps=4)
    sim.run()

    # backend.advance ran once per tick, writing its running counter
    assert np.array_equal(sim.state.view("counter"), np.array([0.0, 1.0, 2.0, 3.0]))
    assert backend.calls[0] == "init"
