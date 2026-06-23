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
