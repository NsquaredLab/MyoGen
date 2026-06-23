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
