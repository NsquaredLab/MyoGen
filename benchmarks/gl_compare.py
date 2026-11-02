"""Compare complete recorded Neo artifacts; never align or rescale signals."""
import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np

from gl_common import OUT, dump_json


def segments(block):
    parents = block.groups if block.groups else [block]
    return [(index, segment) for index, parent in enumerate(parents) for segment in parent.segments]


def compare(reference, candidate, *, destination=None, plots=False):
    assert reference.name == candidate.name and reference.annotations == candidate.annotations
    assert len(reference.groups) == len(candidate.groups)
    for expected_group, actual_group in zip(reference.groups, candidate.groups):
        assert expected_group.name == actual_group.name
        assert expected_group.annotations == actual_group.annotations
    expected, actual = segments(reference), segments(candidate)
    assert len(expected) == len(actual), "segment count"
    rows = []
    pages = None
    preview_written = False
    if plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        pages = PdfPages(str(destination) + ".pdf")
    try:
        for (a_group, a_seg), (b_group, b_seg) in zip(expected, actual):
            assert a_group == b_group and a_seg.name == b_seg.name
            assert a_seg.annotations == b_seg.annotations
            assert len(a_seg.analogsignals) == len(b_seg.analogsignals)
            for a, b in zip(a_seg.analogsignals, b_seg.analogsignals):
                for field in ("shape", "units", "sampling_rate", "t_start", "t_stop", "name", "annotations"):
                    assert getattr(a, field) == getattr(b, field), (a_seg.name, field)
                assert a.array_annotations.keys() == b.array_annotations.keys()
                for key in a.array_annotations:
                    np.testing.assert_array_equal(a.array_annotations[key], b.array_annotations[key])
                x, y = a.magnitude, b.magnitude
                assert np.all(np.isfinite(x)) and np.all(np.isfinite(y))
                error = np.max(np.abs(x - y), axis=0)
                peak = np.max(np.abs(x), axis=0)
                gate = 1e-12 + 1e-9 * peak
                assert np.all(error <= gate), (a_seg.name, np.max(error / gate))
                # Normalized Gram matrices measure channel relationships without
                # changing either waveform used by the acceptance comparison.
                gram_x, gram_y = x.T @ x, y.T @ y
                gram_error = np.max(np.abs(gram_x - gram_y)) / max(1e-300, np.max(np.abs(gram_x)))
                for channel in range(x.shape[1]):
                    energy_x, energy_y = np.sum(x[:, channel]**2), np.sum(y[:, channel]**2)
                    rows.append(dict(array=a_group, segment=a_seg.name, channel=channel,
                                     peak_reference=peak[channel], maximum_error=error[channel],
                                     gate_ratio=error[channel] / gate[channel],
                                     peak_to_peak_error=np.ptp(y[:, channel]) - np.ptp(x[:, channel]),
                                     energy_reference=energy_x, energy_error=energy_y - energy_x,
                                     peak_sample_reference=int(np.argmax(np.abs(x[:, channel]))),
                                     peak_sample_candidate=int(np.argmax(np.abs(y[:, channel]))),
                                     reference_peak_gap_at_candidate=float(peak[channel] - abs(
                                         x[int(np.argmax(np.abs(y[:, channel]))), channel])),
                                     relative_channel_gram_error=gram_error))
                if pages is not None and np.any(x):
                    n = x.shape[1]
                    columns = 5 if n == 25 else min(4, n)
                    fig, axes = plt.subplots(int(np.ceil(n / columns)), columns, squeeze=False,
                                             figsize=(12, 2.1 * int(np.ceil(n / columns))), layout="constrained")
                    t = np.arange(len(x)) / float(a.sampling_rate) * 1000
                    for channel, ax in enumerate(axes.flat):
                        if channel >= n:
                            ax.set_visible(False)
                            continue
                        ax.plot(t, x[:, channel], color="#254b7e", lw=1.2, label="Reference")
                        ax.plot(t, y[:, channel], color="#cf6040", lw=.8, ls="--", label="CPU")
                        ax.set_title(f"Contact {channel}", fontsize=9)
                        ax.set_xlabel("Time (ms)")
                        ax.set_ylabel(str(a.units))
                    axes.flat[0].legend(fontsize=8)
                    fig.suptitle(f"Array {a_group}, {a_seg.name}: complete waveform, no alignment")
                    pages.savefig(fig)
                    if not preview_written:
                        fig.savefig(str(destination) + "-preview.png", dpi=130)
                        preview_written = True
                    plt.close(fig)
    finally:
        if pages is not None:
            pages.close()
    summary = dict(status="passed", segments=len(expected), channels=len(rows),
                   maximum_error=max(row["maximum_error"] for row in rows),
                   maximum_gate_ratio=max(row["gate_ratio"] for row in rows),
                   timing_mismatches=sum(row["peak_sample_reference"] != row["peak_sample_candidate"] for row in rows),
                   exact_reference_peak_ties=sum(row["peak_sample_reference"] != row["peak_sample_candidate"] and
                                                row["reference_peak_gap_at_candidate"] == 0 for row in rows),
                   maximum_relative_channel_gram_error=max(row["relative_channel_gram_error"] for row in rows),
                   metadata="exact", threshold="1e-12 + 1e-9 * per-channel reference peak")
    if destination is not None:
        dump_json(Path(str(destination) + ".json"), summary)
        with Path(str(destination) + ".csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    for name in (args.reference, args.candidate):
        assert (OUT / f"{name}.joblib").is_file(), name
    rng_a = json.loads((OUT / f"{args.reference}-rng.json").read_text())
    rng_b = json.loads((OUT / f"{args.candidate}-rng.json").read_text())
    assert rng_a == rng_b, "final RNG state"
    result = compare(joblib.load(OUT / f"{args.reference}.joblib"),
                     joblib.load(OUT / f"{args.candidate}.joblib"),
                     destination=OUT / f"comparison-{args.candidate}", plots=args.plots)
    dump_json(OUT / f"comparison-{args.candidate}.json", dict(result, final_rng="exact"))
    print(json.dumps(dict(result, final_rng="exact")), flush=True)
