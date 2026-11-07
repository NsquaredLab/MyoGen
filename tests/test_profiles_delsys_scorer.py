"""The Delsys cc/adj scorer reproduces the archived arm-F numbers the fdi_semg limits were frozen from."""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
# Arm delsys5r_wexp_musclertsc_lam0.63 of benchmarks/semg_dispersion (33 MB, not in the repo); set MYOGEN_ARM_F_NPZ to its path.
ARM_F = Path(os.environ.get("MYOGEN_ARM_F_NPZ", "arm_F_not_set.npz"))


def _delsys():
    spec = importlib.util.spec_from_file_location("delsys", ROOT / "benchmarks" / "profiles" / "delsys.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not ARM_F.exists(), reason="set MYOGEN_ARM_F_NPZ to the arm-F result")
def test_scorer_reproduces_arm_f():
    d = np.load(ARM_F, allow_pickle=True)
    res = _delsys().score(d["mono_emg"], float(d["fs"]), float(d["ramp"]), float(d["hold"]))
    assert set(res) == {"cc", "adj"}
    assert round(res["cc"][0], 3) == 0.196 and res["cc"][1] == pytest.approx(162.5, abs=0.5)
    assert round(res["adj"][0], 3) == 0.233 and res["adj"][1] == pytest.approx(157.5, abs=0.5)


def test_scorer_rejects_wrong_shape():
    with pytest.raises(ValueError, match="time, sensors, 3, 3"):
        _delsys().score(np.zeros((1000, 3, 3)), 10240.0, 2.0, 4.0)
