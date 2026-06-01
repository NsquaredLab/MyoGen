"""
SCI-like Pathological iEMG: Modulation, Loss of Derecruitment, and Spasticity
=============================================================================

This example builds **three intramuscular EMG (iEMG) signals** that illustrate a
progression of motor-control pathology seen after **spinal cord injury (SCI)**.
All three come from the *same* muscle, electrode, and NEURON pipeline
(``DescendingDrive__Pool`` -> ``AlphaMN__Pool`` -> spike trains ->
``IntramuscularEMG``). They differ **only in how the descending drive is shaped**:

1. **Voluntary modulation** -- a sinusoidal cortical drive that returns to zero
   each cycle, so motor units recruit and **derecruit** every cycle.
2. **Loss of derecruitment** -- the same sinusoid riding on a tonic *floor*, so
   once recruited the units **never go silent** (a functional stand-in for
   persistent-inward-current self-sustained firing).
3. **Modulation then spasticity** -- voluntary modulation that switches to
   involuntary **~6 Hz clonic bursts**.

.. note::
    This example is *self-contained*: it generates its own recruitment
    thresholds and muscle model, so it does not depend on the ``01_basic``
    gallery outputs.
"""
# sphinx_gallery_thumbnail_number = -1

# %%

##############################################################################
# Import Libraries
# ----------------

import itertools
from pathlib import Path

import joblib
import numpy as np
import quantities as pq
import seaborn as sns
from matplotlib import pyplot as plt
from neo import AnalogSignal, Block, Segment, SpikeTrain
from neuron import h

from myogen import get_random_generator, set_random_seed, simulator
from myogen.simulator.neuron import Network
from myogen.simulator.neuron.populations import AlphaMN__Pool, DescendingDrive__Pool
from myogen.utils.nmodl import load_nmodl_mechanisms
from myogen.utils.types import pps

plt.style.use("fivethirtyeight")

# %%

##############################################################################
# Setup
# -----
#
# Fix the seed, load NEURON mechanisms, and create a results directory next to
# this script (the gallery runs each subsection in its own directory, so we use
# a ``__file__``-relative path rather than a bare ``./results``).

set_random_seed(42)
load_nmodl_mechanisms()

try:
    _script_dir = Path(__file__).parent
except NameError:  # __file__ is undefined when run as a raw cell
    _script_dir = Path.cwd()
save_path = _script_dir / "results"
save_path.mkdir(exist_ok=True, parents=True)

# Simulation constants shared by all three conditions
N_MU = 60  # motor units (kept lean so the gallery build stays fast)
timestep = 0.1 * pq.ms
simulation_time = 9000 * pq.ms
time_points = int(simulation_time / timestep)
base_freq__Hz = 0.5  # voluntary modulation frequency
clonus_freq__Hz = 6.0  # spastic clonus burst frequency

# %%

##############################################################################
# Recruitment Thresholds and Muscle Model
# ---------------------------------------
#
# Generate recruitment thresholds (Combined model) and build the muscle model.
# Both are cached so re-runs during authoring are cheap.

recruitment_thresholds, _ = simulator.RecruitmentThresholds(
    N=N_MU,
    recruitment_range__ratio=100,
    deluca__slope=5,
    konstantin__max_threshold__ratio=1.0,
    mode="combined",
)

muscle_cache = save_path / "sci_muscle_model.pkl"
if muscle_cache.exists():
    muscle = joblib.load(muscle_cache)
else:
    muscle = simulator.Muscle(
        recruitment_thresholds=recruitment_thresholds,
        radius_bone__mm=1.0 * pq.mm,
        fiber_density__fibers_per_mm2=400 * pq.mm**-2,
        fat_thickness__mm=10 * pq.mm,
        autorun=True,
    )
    joblib.dump(muscle, muscle_cache)

print(f"Muscle built: {N_MU} MUs, "
      f"{sum(muscle.resulting_number_of_innervated_fibers)} fibers total")

# %%

##############################################################################
# Intramuscular EMG Setup
# -----------------------
#
# Create a differential needle electrode and the iEMG simulator, then compute
# the motor unit action potentials (MUAPs) **once**. The three conditions below
# reuse these MUAPs and only re-run the (cheaper) spike-train -> EMG convolution.

electrode = simulator.IntramuscularElectrodeArray(
    num_electrodes=4,
    inter_electrode_distance__mm=2.0 * pq.mm,
    differentiation_mode="consecutive",
    position__mm=(0.0 * pq.mm, 0.0 * pq.mm, 15.0 * pq.mm),
)

iemg_sim = simulator.IntramuscularEMG(
    muscle_model=muscle,
    electrode_array=electrode,
    MUs_to_simulate=list(range(N_MU)),
)

print("Computing MUAPs (once)...")
muaps__Block = iemg_sim.simulate_muaps(n_jobs=2)
joblib.dump(iemg_sim, save_path / "sci_iemg_simulator.pkl")
