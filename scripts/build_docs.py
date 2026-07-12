#!/usr/bin/env python3
"""Build the MyoGen documentation, resuming the example gallery past segfaults.

mkdocs-gallery executes every example sequentially in one long-lived process.
MyoGen's examples drive NEURON, whose native runtime state accumulates and
eventually segfaults the build (there is no per-example process isolation in
mkdocs-gallery). mkdocs-gallery md5-caches each *successfully executed* example
into ``docs/auto_examples``, which persists between invocations, so re-running
resumes past the crash in a fresh process with less accumulated state. This is
the same reason the old Sphinx workflow ran ``make html || make html``.

This wrapper therefore retries the build a bounded number of times, but with
guards so it can never (a) loop forever, or (b) deploy an incomplete gallery:

* After every attempt the cache is validated. A NEURON SIGSEGV happens *during*
  execution, before mkdocs-gallery writes the ``.py.md5`` stamp, so a crashing
  example is never stamped and simply re-runs. But mkdocs-gallery writes the
  stamp just before the markdown/figures, so — defensively — any stamp whose
  rendered outputs are missing is deleted so that example re-runs.
* If a *failed* attempt produced no new complete examples, we abort loudly
  rather than spin.
* A build that exits 0 but reports an example that "failed to execute" (a plain
  Python error, which retrying will not fix) is a hard failure — we do not ship
  a broken page.
* On success every executable example must be present with all of its figures.

Run it from inside the docs environment so ``properdocs`` is on PATH::

    MKDOCS_GALLERY_PLOT=true uv run --group docs python scripts/build_docs.py

Without ``MKDOCS_GALLERY_PLOT`` the gallery is source-only and one attempt
suffices.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
GALLERY_OUT = ROOT / "docs" / "auto_examples"
SUBDIRS = ["01_basic", "02_finetune", "03_papers/watanabe", "04_clinical"]
# keep in sync with ignore_pattern in docs/gallery_conf.py
IGNORE = re.compile(
    r"(14_calibrate_noise_from_real|_oscillating_dc_helpers|_optimize_dc_worker|_pic_protocols)\.py"
)
MAX_ATTEMPTS = 12  # heavy watanabe/clinical tail clears ~1-2 examples per fresh process
# markdown image sources look like: ![alt](./images/mkd_glr_<name>_001.png){...}
_IMG_SRC = re.compile(r"!\[[^\]]*\]\((\.?/?images/[^)]+\.(?:png|svg))\)")


def executable_examples() -> set[str]:
    """Names of example .py files that the gallery is expected to execute."""
    names: set[str] = set()
    for sub in SUBDIRS:
        for py in sorted((EXAMPLES / sub).glob("*.py")):
            if not IGNORE.search(py.name):
                names.add(py.stem)
    return names


def validate_cache() -> tuple[set[str], list[str]]:
    """Return (complete example names, deleted-as-incomplete names).

    An example is complete when its ``.py.md5`` stamp, its rendered ``.md`` and
    every image the ``.md`` references all exist and are non-empty. Incomplete
    stamps are deleted so the example re-runs on the next attempt.
    """
    complete: set[str] = set()
    dropped: list[str] = []
    for sub in SUBDIRS:
        gdir = GALLERY_OUT / sub
        if not gdir.exists():
            continue
        for stamp in gdir.rglob("*.py.md5"):
            name = stamp.name[: -len(".py.md5")]
            md = stamp.parent / f"{name}.md"
            ok = md.exists()
            if ok:
                for ref in _IMG_SRC.findall(md.read_text(errors="replace")):
                    img = stamp.parent / ref.lstrip("./")
                    if not img.exists() or img.stat().st_size == 0:
                        ok = False
                        break
            if ok:
                complete.add(name)
            else:
                stamp.unlink(missing_ok=True)
                dropped.append(name)
    return complete, dropped


def run_build() -> tuple[int, bool]:
    """Run one ``properdocs build``; return (exit code, any example failed)."""
    proc = subprocess.run(
        ["properdocs", "build"],
        cwd=ROOT,
        env={**os.environ},
        text=True,
        capture_output=True,
    )
    # stream a trimmed tail so CI logs stay useful without the tqdm flood
    sys.stdout.write(proc.stdout[-6000:])
    sys.stderr.write(proc.stderr[-3000:])
    sys.stdout.flush()
    failed = "failed to execute correctly" in (proc.stdout + proc.stderr)
    return proc.returncode, failed


def main() -> int:
    target = executable_examples()
    executing = os.environ.get("MKDOCS_GALLERY_PLOT", "").lower() in {"1", "true", "yes"}
    print(f"docs build: {len(target)} executable examples; "
          f"gallery execution {'ON' if executing else 'OFF (source-only)'}")

    prev_complete = -1
    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"::group::docs build attempt {attempt}", flush=True)
        code, failed = run_build()
        print("::endgroup::", flush=True)

        complete, dropped = validate_cache()
        if dropped:
            print(f"  re-running {len(dropped)} example(s) with incomplete outputs: "
                  f"{', '.join(sorted(dropped))}")
        print(f"attempt {attempt}: exit={code} failed_examples={failed} "
              f"complete={len(complete)}/{len(target)}", flush=True)

        if code == 0:
            if failed:
                print("::error::an example failed to execute (Python error); "
                      "retrying will not fix it. Not deploying a broken gallery.")
                return 1
            missing = target - complete if executing else set()
            if missing:
                print(f"::error::build succeeded but these examples are missing "
                      f"figures/pages: {', '.join(sorted(missing))}")
                return 1
            print(f"docs build complete on attempt {attempt}.")
            return 0

        # crashed (segfault etc.): only keep going if we actually made progress
        if not executing:
            print("::error::source-only build failed (not a gallery segfault).")
            return code
        if len(complete) <= prev_complete:
            print(f"::error::no progress on attempt {attempt} "
                  f"({len(complete)} <= {prev_complete} complete). Aborting retries.")
            return 1
        prev_complete = len(complete)

    print(f"::error::gallery did not complete after {MAX_ATTEMPTS} attempts "
          f"({prev_complete}/{len(target)} examples).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
