"""``myogen._seed_scope`` error handling, and the dependencies ``import myogen`` needs."""
import tomllib
from pathlib import Path

import pytest

import myogen
import myogen.simulator.neuron.cells as cells

REPO = Path(__file__).resolve().parents[1]


def test_seed_scope_raises_reset_errors_and_gives_the_generator_back(monkeypatch):
    myogen.set_random_seed(5)
    before = myogen.get_random_generator()

    def broken():
        raise RuntimeError("reset failed")

    monkeypatch.setattr(cells, "reset_cell_id_counters", broken)
    with pytest.raises(RuntimeError, match="reset failed"):
        with myogen._seed_scope(1):
            pass
    assert myogen.get_random_generator() is before
    assert myogen.get_random_seed() == 5


def test_pyyaml_is_declared():
    """``import myogen`` loads the profile cards, which need PyYAML."""
    deps = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"]
    assert any(d.lower().startswith("pyyaml") for d in deps)
