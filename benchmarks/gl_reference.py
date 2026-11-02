"""Load the pinned, unmodified GitLab numerical reference without another checkout."""
import ast
import subprocess
import types
from gl_common import ROOT, BASE


def surface_reference():
    from myogen.simulator.core.emg.surface import simulate_fiber as live
    def original(relative):
        return subprocess.check_output([
            "git", "-c", "safe.directory="+ROOT.as_posix(),
            "show", BASE+":"+relative], cwd=ROOT).decode("utf-8")
    prefix = "myogen/simulator/core/emg/surface/"
    conductor = types.ModuleType("gl_baseline_conductor")
    exec(compile(original(prefix+"volume_conductor.py"),
                 "<GitLab fc077 conductor>", "exec"), conductor.__dict__)
    old = ast.parse(original(prefix+"simulate_fiber.py"))
    current = ast.parse((ROOT/prefix/"simulate_fiber.py").read_text(encoding="utf-8"))
    # Helpers, including Numba arithmetic, must be the pinned reference helpers.
    functions = {n.name:n for n in current.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    for node in old.body:
        if isinstance(node, ast.FunctionDef) and node.name not in ("_simulate_fiber_v2_python", "simulate_fiber_v2"):
            assert ast.dump(node) == ast.dump(functions[node.name]), node.name
    node=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=="_simulate_fiber_v2_python")
    namespace=vars(live).copy()
    namespace["layered_cylinder_transfer"]=conductor.layered_cylinder_transfer
    exec(compile(ast.Module(body=[node],type_ignores=[]),"<GitLab fc077 fibre>","exec"),namespace)
    return namespace["_simulate_fiber_v2_python"]
