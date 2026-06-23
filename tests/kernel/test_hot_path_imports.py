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
        if (
            line.startswith("import neo")
            or line.startswith("import quantities")
            or line.startswith("from neo")
            or line.startswith("from quantities")
        ):
            raise AssertionError(f"module-level neo/quantities import in result.py: {line!r}")
