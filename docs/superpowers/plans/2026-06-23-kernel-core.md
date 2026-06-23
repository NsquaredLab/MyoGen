# MyoGen Kernel Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the backend-agnostic simulation kernel (`myogen/kernel/`) — `SimState` shared buffer, `Stage`/`Backend` protocols, the `Simulation` lockstep driver with closed-loop feedback, and the buffer-first `SimResult` with opt-in neo/NWB adapters — fully testable in CI with no NEURON/jaxley/FEM.

**Architecture:** A `SimState` holds named, contiguous SI float arrays (`xp` = numpy by default). `Stage` objects claim buffer slices in `setup()` and write them in place in `step()`. A `Simulation` runs all stages in fixed list order each tick; feedback emerges naturally (a stage reading a buffer a later stage writes gets the previous tick's value — a one-tick delay). `SimResult` snapshots the buffers as raw arrays and exposes `to_neo()`/`to_nwb()` as explicit, lazy methods so neo/quantities never touch the hot path.

**Tech Stack:** Python 3.12+, numpy (core), neo + quantities + pynwb (adapters only, already core deps), pytest. No jax/torch dependency in this slice (`xp` is parameterizable for later).

**Scope:** This is plan 1 of several. It delivers the kernel only. `NEURONBackend`, the real stages (Network/Muscle/Joint/Afferents/EMG), jaxley, and FEM are follow-on plans built on these protocols. Source spec: `docs/superpowers/specs/2026-06-23-pufferlib-inspired-api-design.md` (§4 core protocols, §4 `SimResult`, §9 testing).

**Conventions:**
- Run tests with: `uv run pytest tests/kernel -v` (falls back to `pytest tests/kernel -v` if not using uv).
- All new kernel modules start with `from __future__ import annotations`.
- Hot-path modules (`state.py`, `protocols.py`, `simulation.py`) MUST NOT import `neo` or `quantities` at module level. Only `result.py` may, and only lazily (inside methods).

---

### Task 1: Package scaffold + `SimState`

**Files:**
- Create: `myogen/kernel/__init__.py`
- Create: `myogen/kernel/state.py`
- Create: `tests/kernel/__init__.py`
- Test: `tests/kernel/test_state.py`

- [ ] **Step 1: Write the failing test**

Create `tests/kernel/__init__.py` (empty file) and `tests/kernel/test_state.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from myogen.kernel.state import SimState


def test_alloc_returns_zeroed_array_and_view_returns_same_object():
    state = SimState(n_units=4, dt_s=0.001, n_steps=10)
    buf = state.alloc("force", (10, 2))
    assert buf.shape == (10, 2)
    assert buf.dtype == np.float64
    assert np.all(buf == 0.0)
    # view returns the SAME underlying object (zero-copy contract)
    assert state.view("force") is buf
    assert state.has("force") is True


def test_view_missing_buffer_raises_keyerror():
    state = SimState(n_units=1, dt_s=0.001, n_steps=1)
    assert state.has("nope") is False
    with pytest.raises(KeyError, match="no buffer named 'nope'"):
        state.view("nope")


def test_double_alloc_raises():
    state = SimState(n_units=1, dt_s=0.001, n_steps=1)
    state.alloc("x", (3,))
    with pytest.raises(KeyError, match="already allocated"):
        state.alloc("x", (3,))


def test_alloc_respects_dtype_and_xp_default_is_numpy():
    state = SimState(n_units=1, dt_s=0.001, n_steps=1)
    assert state.xp is np
    buf = state.alloc("spikes", (5, 1), dtype=np.int8)
    assert buf.dtype == np.int8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/kernel/test_state.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'myogen.kernel'`

- [ ] **Step 3: Write minimal implementation**

Create `myogen/kernel/__init__.py`:

```python
from __future__ import annotations

from myogen.kernel.state import SimState

__all__ = ["SimState"]
```

Create `myogen/kernel/state.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from types import ModuleType

import numpy as np


@dataclass
class SimState:
    """The shared simulation buffer.

    Single source of truth for a run. Holds named, contiguous SI float arrays.
    Stages claim buffers in ``setup()`` and write them in place in ``step()``.
    No physical units and no neo objects ever live here — SI is implied.
    """

    n_units: int
    dt_s: float
    n_steps: int
    xp: ModuleType = np
    t: int = 0  # current step index, set by the driver each tick
    _buffers: dict = field(default_factory=dict, repr=False)

    def alloc(self, name: str, shape: tuple[int, ...], dtype=None):
        """Allocate a zeroed buffer and return it. Raises if name is taken."""
        if name in self._buffers:
            raise KeyError(f"buffer {name!r} already allocated")
        if dtype is None:
            dtype = self.xp.float64
        buf = self.xp.zeros(shape, dtype=dtype)
        self._buffers[name] = buf
        return buf

    def view(self, name: str):
        """Return the buffer named ``name`` (the same object, zero-copy)."""
        try:
            return self._buffers[name]
        except KeyError:
            raise KeyError(f"no buffer named {name!r}") from None

    def has(self, name: str) -> bool:
        return name in self._buffers
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/kernel/test_state.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add myogen/kernel/__init__.py myogen/kernel/state.py tests/kernel/__init__.py tests/kernel/test_state.py
git commit -m "feat(kernel): add SimState shared buffer"
```

---

### Task 2: `Stage` and `Backend` protocols

**Files:**
- Create: `myogen/kernel/protocols.py`
- Modify: `myogen/kernel/__init__.py`
- Test: `tests/kernel/test_protocols.py`

- [ ] **Step 1: Write the failing test**

Create `tests/kernel/test_protocols.py`:

```python
from __future__ import annotations

from myogen.kernel.protocols import Backend, Stage
from myogen.kernel.state import SimState


class _GoodStage:
    def setup(self, state: SimState) -> None:
        state.alloc("x", (state.n_steps,))

    def step(self, state: SimState, t: int) -> None:
        state.view("x")[t] = 1.0


class _GoodBackend:
    def init(self, state: SimState) -> None: ...
    def advance(self, state: SimState, dt_s: float) -> None: ...
    def teardown(self) -> None: ...


class _NotAStage:
    def setup(self, state) -> None: ...
    # missing step()


def test_stage_protocol_is_runtime_checkable():
    assert isinstance(_GoodStage(), Stage)
    assert not isinstance(_NotAStage(), Stage)


def test_backend_protocol_is_runtime_checkable():
    assert isinstance(_GoodBackend(), Backend)
    assert not isinstance(_GoodStage(), Backend)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/kernel/test_protocols.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'myogen.kernel.protocols'`

- [ ] **Step 3: Write minimal implementation**

Create `myogen/kernel/protocols.py`:

```python
from __future__ import annotations

from typing import Protocol, runtime_checkable

from myogen.kernel.state import SimState


@runtime_checkable
class Stage(Protocol):
    """A thin pipeline stage. Owns API/units/validation; writes buffers in place.

    Stages do not return values — they mutate the shared ``SimState``. A stage
    that needs physics delegates to a ``Backend`` behind a seam.
    """

    def setup(self, state: SimState) -> None:
        """Claim (allocate) the buffers this stage writes."""
        ...

    def step(self, state: SimState, t: int) -> None:
        """Advance one tick, writing results into the state buffers in place."""
        ...


@runtime_checkable
class Backend(Protocol):
    """The thing that actually integrates physics, behind a seam.

    Implementations: NEURONBackend / JaxleyBackend (dynamics seam),
    Analytic / Hill / FEM (muscle seam). Not exercised by the kernel itself
    beyond protocol conformance — concrete backends are follow-on plans.
    """

    def init(self, state: SimState) -> None: ...

    def advance(self, state: SimState, dt_s: float) -> None: ...

    def teardown(self) -> None: ...
```

Update `myogen/kernel/__init__.py`:

```python
from __future__ import annotations

from myogen.kernel.protocols import Backend, Stage
from myogen.kernel.state import SimState

__all__ = ["SimState", "Stage", "Backend"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/kernel/test_protocols.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add myogen/kernel/protocols.py myogen/kernel/__init__.py tests/kernel/test_protocols.py
git commit -m "feat(kernel): add Stage and Backend protocols"
```

---

### Task 3: `Simulation` driver — lockstep run

**Files:**
- Create: `myogen/kernel/simulation.py`
- Create: `tests/kernel/_doubles.py`
- Modify: `myogen/kernel/__init__.py`
- Test: `tests/kernel/test_simulation.py`

Note: `SimResult` does not exist yet. In this task `Simulation.run()` returns the `SimState`; Task 5 changes it to return a `SimResult`. The test here asserts on the final `SimState`.

- [ ] **Step 1: Write the failing test**

Create `tests/kernel/_doubles.py` (shared test doubles — NOT a test module):

```python
"""Test doubles for the kernel. Tiny, dependency-free Stage/Backend fakes."""
from __future__ import annotations

from myogen.kernel.state import SimState


class RampStage:
    """Writes t into buffer ``name`` at row t each step (a deterministic ramp)."""

    def __init__(self, name: str = "ramp"):
        self.name = name

    def setup(self, state: SimState) -> None:
        state.alloc(self.name, (state.n_steps,))

    def step(self, state: SimState, t: int) -> None:
        state.view(self.name)[t] = float(t)


class CountingBackend:
    """A Backend double that records lifecycle calls and integrates a counter."""

    def __init__(self):
        self.calls: list[str] = []
        self.advances = 0

    def init(self, state: SimState) -> None:
        self.calls.append("init")
        state.alloc("counter", (state.n_steps,))

    def advance(self, state: SimState, dt_s: float) -> None:
        state.view("counter")[state.t] = self.advances
        self.advances += 1

    def teardown(self) -> None:
        self.calls.append("teardown")
```

Create `tests/kernel/test_simulation.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from myogen.kernel.simulation import Simulation
from tests.kernel._doubles import RampStage


def test_run_steps_each_stage_once_per_tick():
    sim = Simulation(RampStage("a"), RampStage("b"), n_units=2, dt_s=0.001, n_steps=5)
    state = sim.run()
    assert np.array_equal(state.view("a"), np.arange(5, dtype=float))
    assert np.array_equal(state.view("b"), np.arange(5, dtype=float))


def test_run_calls_setup_before_stepping_and_sets_t():
    sim = Simulation(RampStage(), n_units=1, dt_s=0.001, n_steps=3)
    state = sim.run()
    # final t is the last step index
    assert state.t == 2


def test_simulation_requires_at_least_one_stage():
    with pytest.raises(ValueError, match="at least one stage"):
        Simulation(n_units=1, dt_s=0.001, n_steps=1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/kernel/test_simulation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'myogen.kernel.simulation'`

- [ ] **Step 3: Write minimal implementation**

Create `myogen/kernel/simulation.py`:

```python
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from myogen.kernel.state import SimState


class Simulation:
    """Backend-agnostic driver. Owns dt and the lockstep tick over the stages.

    Stages run in fixed list order each tick. Closed-loop feedback emerges
    naturally: a stage that reads a buffer a *later* stage writes sees the
    previous tick's value (a one-tick delay where conduction delays live).
    """

    def __init__(
        self,
        *stages,
        n_units: int,
        dt_s: float,
        n_steps: int,
        xp=np,
    ):
        if not stages:
            raise ValueError("Simulation requires at least one stage")
        self.stages = list(stages)
        self.state = SimState(n_units=n_units, dt_s=dt_s, n_steps=n_steps, xp=xp)
        self._callbacks: list[Callable[[SimState, int], None]] = []
        self._setup_done = False

    def on_step(self, fn: Callable[[SimState, int], None]) -> None:
        """Register a per-tick callback (closed-loop hook). Gets raw buffers."""
        self._callbacks.append(fn)

    def setup(self) -> None:
        for stage in self.stages:
            stage.setup(self.state)
        self._setup_done = True

    def run(self) -> SimState:
        if not self._setup_done:
            self.setup()
        s = self.state
        for t in range(s.n_steps):
            s.t = t
            for stage in self.stages:
                stage.step(s, t)
            for cb in self._callbacks:
                cb(s, t)
        return s
```

Update `myogen/kernel/__init__.py`:

```python
from __future__ import annotations

from myogen.kernel.protocols import Backend, Stage
from myogen.kernel.simulation import Simulation
from myogen.kernel.state import SimState

__all__ = ["SimState", "Stage", "Backend", "Simulation"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/kernel/test_simulation.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add myogen/kernel/simulation.py myogen/kernel/__init__.py tests/kernel/_doubles.py tests/kernel/test_simulation.py
git commit -m "feat(kernel): add Simulation lockstep driver"
```

---

### Task 4: Closed-loop feedback (one-tick delay) + `on_step` callback

**Files:**
- Test: `tests/kernel/test_feedback.py`

This task adds no production code — it proves the feedback and callback semantics the driver already supports, locking them with tests (they are load-bearing for the whole closed-loop design).

- [ ] **Step 1: Write the failing test**

Create `tests/kernel/test_feedback.py`:

```python
from __future__ import annotations

import numpy as np

from myogen.kernel.simulation import Simulation
from myogen.kernel.state import SimState


class _Writer:
    """Stage B: writes 'signal'[t] = t each tick (runs AFTER the reader)."""

    def setup(self, state: SimState) -> None:
        state.alloc("signal", (state.n_steps,))

    def step(self, state: SimState, t: int) -> None:
        state.view("signal")[t] = float(t)


class _Reader:
    """Stage A: reads 'signal' from the PREVIOUS tick into 'fed_back'."""

    def setup(self, state: SimState) -> None:
        state.alloc("fed_back", (state.n_steps,))

    def step(self, state: SimState, t: int) -> None:
        prev = state.view("signal")[t - 1] if t > 0 else 0.0
        state.view("fed_back")[t] = prev


def test_feedback_has_one_tick_delay():
    # Reader runs before Writer, so it sees last tick's signal: a one-tick delay.
    sim = Simulation(_Reader(), _Writer(), n_units=1, dt_s=0.001, n_steps=4)
    state = sim.run()
    # signal = [0,1,2,3]; fed_back = [0, signal[0], signal[1], signal[2]] = [0,0,1,2]
    assert np.array_equal(state.view("fed_back"), np.array([0.0, 0.0, 1.0, 2.0]))


def test_on_step_callback_runs_each_tick_with_raw_state():
    sim = Simulation(_Writer(), n_units=1, dt_s=0.001, n_steps=3)
    seen: list[float] = []
    sim.on_step(lambda s, t: seen.append(float(s.view("signal")[t])))
    sim.run()
    assert seen == [0.0, 1.0, 2.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/kernel/test_feedback.py -v`
Expected: PASS immediately is NOT expected — but if the driver is correct it will PASS. If it FAILS, the driver's ordering/callback logic is wrong and must be fixed in `simulation.py`. Run and confirm the behavior.

(Note: because Task 3 already implemented the driver correctly, these tests should pass. They exist to *lock* the semantics. If they fail, fix `simulation.py` until they pass.)

- [ ] **Step 3: (only if Step 2 failed) Fix the driver**

If `test_feedback_has_one_tick_delay` fails, ensure stages run in list order within a single tick (not re-ordered) and `s.t` is set before stages run. If `test_on_step_callback` fails, ensure callbacks run after all stages each tick. The Task 3 implementation already does this; no change expected.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/kernel/test_feedback.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add tests/kernel/test_feedback.py
git commit -m "test(kernel): lock one-tick feedback and on_step semantics"
```

---

### Task 5: `SimResult.from_state` (buffer-first snapshot)

**Files:**
- Create: `myogen/kernel/result.py`
- Modify: `myogen/kernel/simulation.py` (return `SimResult` from `run()`)
- Modify: `myogen/kernel/__init__.py`
- Modify: `tests/kernel/test_simulation.py` (run() now returns SimResult; assert via `.state` accessor)
- Test: `tests/kernel/test_result.py`

Convention for `from_state`: spikes are stored in a `(n_steps, n_units)` buffer named `"spikes"` (nonzero = spike); force in `"force"` `(n_steps, n_pools)`; surface EMG in `"surface_emg"` `(n_steps, n_rows, n_cols)`. All optional.

- [ ] **Step 1: Write the failing test**

Create `tests/kernel/test_result.py`:

```python
from __future__ import annotations

import numpy as np

from myogen.kernel.result import SimResult
from myogen.kernel.state import SimState


def _state_with_spikes_and_force() -> SimState:
    state = SimState(n_units=2, dt_s=0.5, n_steps=4)
    spikes = state.alloc("spikes", (4, 2), dtype=np.int8)
    spikes[1, 0] = 1  # unit 0 spikes at step 1 -> t = 0.5 s
    spikes[3, 1] = 1  # unit 1 spikes at step 3 -> t = 1.5 s
    force = state.alloc("force", (4, 1))
    force[:, 0] = [0.0, 1.0, 2.0, 3.0]
    state.t = 3
    return state


def test_from_state_extracts_spike_times_in_seconds():
    res = SimResult.from_state(_state_with_spikes_and_force())
    assert res.n_units == 2
    assert res.dt_s == 0.5
    assert np.array_equal(res.spike_times_s[0], np.array([0.5]))
    assert np.array_equal(res.spike_times_s[1], np.array([1.5]))


def test_from_state_copies_force_array():
    res = SimResult.from_state(_state_with_spikes_and_force())
    assert np.array_equal(res.force_N[:, 0], np.array([0.0, 1.0, 2.0, 3.0]))


def test_from_state_handles_missing_optional_buffers():
    state = SimState(n_units=3, dt_s=0.001, n_steps=2)
    res = SimResult.from_state(state)
    assert res.force_N is None
    assert res.surface_emg_V is None
    assert len(res.spike_times_s) == 3
    for arr in res.spike_times_s:
        assert arr.size == 0


def test_from_state_infers_grid_shape_from_emg():
    state = SimState(n_units=1, dt_s=0.001, n_steps=2)
    state.alloc("surface_emg", (2, 8, 4))
    res = SimResult.from_state(state)
    assert res.grid_shape == (8, 4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/kernel/test_result.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'myogen.kernel.result'`

- [ ] **Step 3: Write minimal implementation**

Create `myogen/kernel/result.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from myogen.kernel.state import SimState


@dataclass(slots=True)
class SimResult:
    """Buffer-first results facade. Raw SI float arrays only.

    neo/NWB are opt-in *methods* (not properties) so they never sneak into the
    hot path. SI is implied by field-name suffixes (_s, _N, _V).
    """

    spike_times_s: list  # list[np.ndarray], one per unit, seconds
    force_N: np.ndarray | None
    surface_emg_V: np.ndarray | None
    dt_s: float
    t_start_s: float
    n_units: int
    grid_shape: tuple[int, ...] | None = None

    @classmethod
    def from_state(cls, state: SimState) -> "SimResult":
        if state.has("spikes"):
            spikes = np.asarray(state.view("spikes"))
            spike_times = [
                np.flatnonzero(spikes[:, u]).astype(float) * state.dt_s
                for u in range(state.n_units)
            ]
        else:
            spike_times = [np.empty(0) for _ in range(state.n_units)]

        force = np.asarray(state.view("force")) if state.has("force") else None
        emg = (
            np.asarray(state.view("surface_emg"))
            if state.has("surface_emg")
            else None
        )
        grid = tuple(emg.shape[1:]) if emg is not None and emg.ndim == 3 else None

        return cls(
            spike_times_s=spike_times,
            force_N=force,
            surface_emg_V=emg,
            dt_s=state.dt_s,
            t_start_s=0.0,
            n_units=state.n_units,
            grid_shape=grid,
        )
```

Update `myogen/kernel/simulation.py` — change the import and the `run()` return:

Replace the import line `from myogen.kernel.state import SimState` with:

```python
from myogen.kernel.result import SimResult
from myogen.kernel.state import SimState
```

Replace the `run` method's signature and final `return s` so it reads:

```python
    def run(self) -> SimResult:
        if not self._setup_done:
            self.setup()
        s = self.state
        for t in range(s.n_steps):
            s.t = t
            for stage in self.stages:
                stage.step(s, t)
            for cb in self._callbacks:
                cb(s, t)
        return SimResult.from_state(s)
```

Update `myogen/kernel/__init__.py`:

```python
from __future__ import annotations

from myogen.kernel.protocols import Backend, Stage
from myogen.kernel.result import SimResult
from myogen.kernel.simulation import Simulation
from myogen.kernel.state import SimState

__all__ = ["SimState", "Stage", "Backend", "Simulation", "SimResult"]
```

Update `tests/kernel/test_simulation.py` — `run()` now returns a `SimResult`, so the three tests must read the buffers through the simulation's `state`, not the return value. Replace the three test bodies with:

```python
def test_run_steps_each_stage_once_per_tick():
    sim = Simulation(RampStage("a"), RampStage("b"), n_units=2, dt_s=0.001, n_steps=5)
    sim.run()
    assert np.array_equal(sim.state.view("a"), np.arange(5, dtype=float))
    assert np.array_equal(sim.state.view("b"), np.arange(5, dtype=float))


def test_run_calls_setup_before_stepping_and_sets_t():
    sim = Simulation(RampStage(), n_units=1, dt_s=0.001, n_steps=3)
    sim.run()
    assert sim.state.t == 2


def test_simulation_requires_at_least_one_stage():
    with pytest.raises(ValueError, match="at least one stage"):
        Simulation(n_units=1, dt_s=0.001, n_steps=1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/kernel -v`
Expected: PASS (all kernel tests, including the updated `test_simulation.py` and new `test_result.py`)

- [ ] **Step 5: Commit**

```bash
git add myogen/kernel/result.py myogen/kernel/simulation.py myogen/kernel/__init__.py tests/kernel/test_result.py tests/kernel/test_simulation.py
git commit -m "feat(kernel): add SimResult buffer-first snapshot; run() returns it"
```

---

### Task 6: `SimResult.to_neo` / `to_nwb` adapters (lazy, off the hot path)

**Files:**
- Modify: `myogen/kernel/result.py`
- Test: `tests/kernel/test_result_adapters.py`

`to_neo()` builds a `neo.Block` with one `Segment`: a `SpikeTrain` per unit, an `AnalogSignal` for force (N), and an `AnalogSignal` for EMG (V, flattened to 2-D). `to_nwb()` delegates to the existing `myogen.utils.nwb.export_to_nwb` and is tested via monkeypatch (no dependence on its exact signature here).

- [ ] **Step 1: Write the failing test**

Create `tests/kernel/test_result_adapters.py`:

```python
from __future__ import annotations

import numpy as np
import quantities as pq

from myogen.kernel.result import SimResult


def _make_result() -> SimResult:
    return SimResult(
        spike_times_s=[np.array([0.5]), np.array([1.5])],
        force_N=np.array([[0.0], [1.0], [2.0], [3.0]]),
        surface_emg_V=np.zeros((4, 2, 2)),
        dt_s=0.5,
        t_start_s=0.0,
        n_units=2,
        grid_shape=(2, 2),
    )


def test_to_neo_builds_block_with_spiketrains_and_signals():
    block = _make_result().to_neo()
    seg = block.segments[0]
    assert len(seg.spiketrains) == 2
    assert np.allclose(seg.spiketrains[0].rescale(pq.s).magnitude, [0.5])
    # one force signal (N) + one emg signal (V)
    units = {str(sig.units.dimensionality) for sig in seg.analogsignals}
    assert "N" in units
    assert "V" in units


def test_to_neo_force_signal_sampling_rate_matches_dt():
    block = _make_result().to_neo()
    force_sig = next(
        s for s in block.segments[0].analogsignals
        if str(s.units.dimensionality) == "N"
    )
    assert np.isclose(force_sig.sampling_rate.rescale(pq.Hz).magnitude, 2.0)  # 1/0.5


def test_to_nwb_delegates_to_export_to_nwb(monkeypatch):
    captured = {}

    def fake_export(block, path, **kwargs):
        captured["block"] = block
        captured["path"] = path
        return path

    monkeypatch.setattr("myogen.utils.nwb.export_to_nwb", fake_export)
    result = _make_result()
    out = result.to_nwb("/tmp/out.nwb")
    assert out == "/tmp/out.nwb"
    assert captured["path"] == "/tmp/out.nwb"
    # delegated a real neo.Block built from this result
    assert len(captured["block"].segments[0].spiketrains) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/kernel/test_result_adapters.py -v`
Expected: FAIL with `AttributeError: 'SimResult' object has no attribute 'to_neo'`

- [ ] **Step 3: Write minimal implementation**

Append two methods to the `SimResult` class in `myogen/kernel/result.py` (lazy imports keep neo/quantities off the hot path):

```python
    def to_neo(self):
        """Build a neo.Block on demand. neo/quantities imported lazily here only."""
        import neo
        import quantities as pq

        n_t = 0
        if self.force_N is not None:
            n_t = len(self.force_N)
        elif self.surface_emg_V is not None:
            n_t = self.surface_emg_V.shape[0]
        t_stop = (self.t_start_s + n_t * self.dt_s) * pq.s
        rate = (1.0 / self.dt_s) * pq.Hz
        t_start = self.t_start_s * pq.s

        block = neo.Block()
        seg = neo.Segment()
        block.segments.append(seg)

        for times in self.spike_times_s:
            seg.spiketrains.append(
                neo.SpikeTrain(np.asarray(times) * pq.s, t_stop=t_stop)
            )

        if self.force_N is not None:
            seg.analogsignals.append(
                neo.AnalogSignal(
                    np.asarray(self.force_N) * pq.N,
                    sampling_rate=rate,
                    t_start=t_start,
                )
            )

        if self.surface_emg_V is not None:
            emg = np.asarray(self.surface_emg_V)
            seg.analogsignals.append(
                neo.AnalogSignal(
                    emg.reshape(emg.shape[0], -1) * pq.V,
                    sampling_rate=rate,
                    t_start=t_start,
                )
            )

        return block

    def to_nwb(self, path, **kwargs):
        """Export to NWB by delegating to the existing exporter via to_neo()."""
        from myogen.utils.nwb import export_to_nwb

        return export_to_nwb(self.to_neo(), path, **kwargs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/kernel/test_result_adapters.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add myogen/kernel/result.py tests/kernel/test_result_adapters.py
git commit -m "feat(kernel): add lazy to_neo/to_nwb adapters to SimResult"
```

---

### Task 7: Backend-seam demo + hot-path import guard

**Files:**
- Modify: `tests/kernel/_doubles.py` (add a stage that delegates to a Backend)
- Test: `tests/kernel/test_backend_seam.py`
- Test: `tests/kernel/test_hot_path_imports.py`

This proves (a) a `Stage` can delegate physics to a `Backend` through the driver (the seam holds for the future NEURON/jaxley backends), and (b) the hot-path modules never import neo/quantities at module level.

- [ ] **Step 1: Write the failing test**

Append to `tests/kernel/_doubles.py`:

```python
class BackendStage:
    """A Stage that delegates each step to a Backend (the seam pattern)."""

    def __init__(self, backend):
        self.backend = backend

    def setup(self, state: SimState) -> None:
        self.backend.init(state)

    def step(self, state: SimState, t: int) -> None:
        self.backend.advance(state, state.dt_s)
```

Create `tests/kernel/test_backend_seam.py`:

```python
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
```

Create `tests/kernel/test_hot_path_imports.py`:

```python
from __future__ import annotations

from pathlib import Path

import myogen.kernel.protocols as protocols_mod
import myogen.kernel.simulation as simulation_mod
import myogen.kernel.state as state_mod

HOT_PATH_MODULES = [state_mod, protocols_mod, simulation_mod]


def test_hot_path_modules_have_no_module_level_neo_or_quantities():
    for mod in HOT_PATH_MODULES:
        src = Path(mod.__file__).read_text()
        for line in src.splitlines():
            stripped = line.strip()
            assert not stripped.startswith("import neo"), f"{mod.__name__}: {stripped}"
            assert not stripped.startswith("import quantities"), f"{mod.__name__}: {stripped}"
            assert not stripped.startswith("from neo"), f"{mod.__name__}: {stripped}"
            assert not stripped.startswith("from quantities"), f"{mod.__name__}: {stripped}"


def test_result_module_imports_neo_only_lazily():
    # result.py may use neo, but ONLY inside methods (indented), never at module level.
    import myogen.kernel.result as result_mod

    src = Path(result_mod.__file__).read_text()
    for line in src.splitlines():
        if line.startswith("import neo") or line.startswith("import quantities"):
            raise AssertionError(f"module-level import in result.py: {line!r}")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/kernel/test_backend_seam.py -v`
Expected: FAIL with `ImportError: cannot import name 'BackendStage'` (until `_doubles.py` is updated in Step 1 — if Step 1's edit is already saved, this test passes and only the new file matters; run both)

Run: `uv run pytest tests/kernel/test_hot_path_imports.py -v`
Expected: PASS if the modules were written cleanly (they should be). If it FAILS, a hot-path module is importing neo/quantities at module level — move that import into `result.py` or make it lazy.

- [ ] **Step 3: Write minimal implementation**

No production code changes are expected — the seam already works via the existing protocols/driver, and the hot-path modules were written without neo/quantities. If `test_hot_path_imports` fails, fix the offending module by removing/lazying the import. The `BackendStage` double added in Step 1 is the only new code.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/kernel -v`
Expected: PASS (entire kernel suite green)

- [ ] **Step 5: Commit**

```bash
git add tests/kernel/_doubles.py tests/kernel/test_backend_seam.py tests/kernel/test_hot_path_imports.py
git commit -m "test(kernel): prove backend seam and hot-path import discipline"
```

---

## Final verification

After all tasks:

```bash
uv run pytest tests/kernel -v
```

Expected: all kernel tests pass. The kernel is now a working, CI-testable, backend-agnostic simulation core with no NEURON/jaxley/FEM dependency — ready for the follow-on `NEURONBackend` and real-stage plans.

## Out of scope (follow-on plans)

- `NEURONBackend` adapting today's NEURON/NMODL machinery to the `Backend` protocol, with parity tests against the legacy `SimulationRunner`.
- Real stages: `Network` (populations + connectivity spec), `Muscle` (geometry + force backends), `Joint`, `Afferents`, `SurfaceEMG`/`IntramuscularEMG`.
- Spec layer (`RecruitmentThresholds`, drive/stimuli, electrode arrays) and conveniences (`make()`, TOML).
- `JaxleyBackend` and `FEMBackend`.
- Migration shim (`run().to_neo()` ≡ legacy `run()`) and example/script porting.
```
