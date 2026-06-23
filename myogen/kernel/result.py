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

        force = np.array(state.view("force")) if state.has("force") else None
        emg = (
            np.array(state.view("surface_emg"))
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
