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
# Only these subdirs are executed (must match filename_pattern in
# docs/gallery_conf.py); watanabe/clinical render source-only, so they are not
# expected to produce figures and are not part of the completeness target.
EXECUTE_SUBDIRS = ["01_basic", "02_finetune"]
# keep in sync with ignore_pattern in docs/gallery_conf.py
IGNORE = re.compile(
    r"(14_calibrate_noise_from_real|_oscillating_dc_helpers|_optimize_dc_worker|_pic_protocols)\.py"
)
MAX_ATTEMPTS = 12  # heavy watanabe/clinical tail clears ~1-2 examples per fresh process
# markdown image sources look like: ![alt](./images/mkd_glr_<name>_001.png){...}
_IMG_SRC = re.compile(r"!\[[^\]]*\]\((\.?/?images/[^)]+\.(?:png|svg))\)")
# mkdocs-gallery logs: "<path>/<name>.py failed to execute correctly: Traceback"
_FAILED = re.compile(r"([^\s/\\]+)\.py failed to execute correctly")


def executable_examples() -> set[str]:
    """Names of example .py files that the gallery is expected to execute."""
    names: set[str] = set()
    for sub in EXECUTE_SUBDIRS:
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
            ok = md.exists() and md.stat().st_size > 0
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


def drop_stamps(names: set[str]) -> None:
    """Delete the .py.md5 stamps for the given examples so they re-run."""
    for sub in SUBDIRS:
        for name in names:
            for stamp in (GALLERY_OUT / sub).rglob(f"{name}.py.md5"):
                stamp.unlink(missing_ok=True)


def run_build() -> tuple[int, set[str]]:
    """Run one ``properdocs build``; return (exit code, names of failed examples)."""
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
    failed = set(_FAILED.findall(proc.stdout + proc.stderr))
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

        # A failed example is untrustworthy even if it got a stamp (mkdocs-gallery
        # renders a broken page): drop its stamp so it re-executes, and don't
        # count it complete. A deterministically-failing example then shows up as
        # "no progress" below rather than a silent broken page in the deploy.
        drop_stamps(failed)
        complete, dropped = validate_cache()
        rerun = sorted(failed | set(dropped))
        if rerun:
            print(f"  will re-run {len(rerun)} example(s): {', '.join(rerun)}")
        print(f"attempt {attempt}: exit={code} failed={sorted(failed) or 'none'} "
              f"complete={len(complete)}/{len(target)}", flush=True)

        if code == 0 and not failed:
            missing = target - complete if executing else set()
            if missing:
                print(f"::error::build reported success but these examples are "
                      f"missing figures/pages: {', '.join(sorted(missing))}")
                return 1
            print(f"docs build complete on attempt {attempt}.")
            return 0

        # crashed (segfault) or an example failed: retry only if making progress
        if not executing:
            print("::error::source-only build failed (not a gallery segfault).")
            return code
        if len(complete) <= prev_complete:
            print(f"::error::no progress on attempt {attempt} "
                  f"({len(complete)} <= {prev_complete} complete). A deterministically "
                  f"failing example won't be fixed by retrying. Aborting.")
            return 1
        prev_complete = len(complete)

    print(f"::error::gallery did not complete after {MAX_ATTEMPTS} attempts "
          f"({prev_complete}/{len(target)} examples).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
