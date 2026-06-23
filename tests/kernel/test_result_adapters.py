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
