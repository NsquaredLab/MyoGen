"""Gfluctdv (Destexhe 2001 fluctuating conductance) must produce noise once inserted.

Gfluctdv.mod scales both conductances by GLOBAL multipliers ``multex``/``multin``
that default to 0, so inserting the mechanism without enabling them gives zero
conductance and no membrane noise.
"""

import numpy as np
from neuron import h

from myogen import load_nmodl_mechanisms
from myogen.simulator.neuron.populations import AlphaMN__Pool


def test_insert_gfluctdv_produces_conductance_noise():
    load_nmodl_mechanisms()
    h.multex_Gfluctdv = 0  # the .mod default; a previous test must not mask it
    h.multin_Gfluctdv = 0
    pool = AlphaMN__Pool(recruitment_thresholds__array=np.array([0.1, 1.0]))
    cell = pool[0]
    cell.insert_Gfluctdv()

    d = cell.dend[0]
    g_e = h.Vector().record(d(0.5)._ref_g_e_Gfluctdv)
    h.dt = 0.025
    h.finitialize(-70)
    while h.t < 50:
        h.fadvance()

    g = np.asarray(g_e)
    assert np.std(g) > 0.1 * d.std_e_Gfluctdv, f"g_e std {np.std(g):.3g} S/cm2: Gfluctdv inserted but silent"
