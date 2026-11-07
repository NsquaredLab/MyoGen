"""On pushes to main the docs build executes the examples that docs/gallery_conf.py selects.

mkdocs-gallery runs them with exec(), where ``__file__`` is undefined, and without command-line arguments; a failed
example fails the build (scripts/build_docs.py). Scripts that cannot run that way render source-only.
"""
import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _conf_value(key):
    src = (REPO / "docs" / "gallery_conf.py").read_text(encoding="utf-8")
    return ast.literal_eval(re.search(rf'"{key}":\s*(r"[^"]*")', src).group(1))


def _executed_examples():
    run, skip = re.compile(_conf_value("filename_pattern")), re.compile(_conf_value("ignore_pattern"))
    for py in sorted((REPO / "examples").rglob("*.py")):
        path = "/" + py.relative_to(REPO).as_posix()
        if run.search(path) and not skip.search(path):
            yield py


def _unguarded_file_uses(py):
    """Lines that use ``__file__`` outside a ``try`` whose handler catches NameError."""
    tree = ast.parse(py.read_text(encoding="utf-8"))
    guarded = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try) and any(
            h.type is not None and "NameError" in ast.unparse(h.type) for h in node.handlers
        ):
            for stmt in node.body:
                guarded.update(id(n) for n in ast.walk(stmt))
    return [n.lineno for n in ast.walk(tree)
            if isinstance(n, ast.Name) and n.id == "__file__" and id(n) not in guarded]


def test_executed_examples_do_not_need_file():
    bad = {py.relative_to(REPO).as_posix(): lines for py in _executed_examples() if (lines := _unguarded_file_uses(py))}
    assert bad == {}


def test_source_only_lists_agree():
    pattern = _conf_value("filename_pattern")
    in_conf = set(re.findall(r"(\w+)\\.py\$", pattern))
    src = (REPO / "scripts" / "build_docs.py").read_text(encoding="utf-8")
    in_build = ast.literal_eval(re.search(r"^SOURCE_ONLY = (\{.*\})$", src, re.M).group(1))
    assert in_conf == in_build
