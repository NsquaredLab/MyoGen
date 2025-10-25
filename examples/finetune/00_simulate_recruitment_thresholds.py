"""
Recruitment Thresholds
=================================

The first step in using **MyoGen** is to generate the **recruitment thresholds** of the **motor units** (MUs).

.. note::
    A **recruitment threshold** is the minimum force required to activate a MU.

    In **MyoGen**, the threshold is defined between ``0`` and ``1``, where ``0`` is the minimum force required to activate a MU and ``1`` is the maximum force required to activate a MU.

**MyoGen** offers **4 different models** to generate the recruitment thresholds:

* **Fuglevand** model: Classic exponential distribution (*Fuglevand et al., 1993*)
* **De Luca** model: Slope-corrected exponential distribution (*De Luca & Contessa, 2012*)
* **Konstantin** model: Exponential with explicit maximum threshold control (*Konstantin et al., 2019*)
* **Combined** model: Hybrid approach combining De Luca shape with Konstantin scaling (*Ours*)
"""

# %%
##############################################################################
# Import Libraries
# ----------------
from pathlib import Path

import joblib
from matplotlib import pyplot as plt
import scienceplots  # noqa
import seaborn as sns

from myogen import simulator
from myogen.utils.plotting import plot_recruitment_thresholds

plt.style.use(["science", "nature"])
sns.set_context("paper", font_scale=2)

# Disable LaTeX rendering (not required)
plt.rcParams["text.usetex"] = False
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.figsize"] = (10, 4.5)

# Keep text editable in SVG/PDF exports (for Illustrator)
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42

# Set font to Liberation Sans or Roboto
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Liberation Sans", "Roboto", "DejaVu Sans"]

# Remove top and right spines and ticks
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["xtick.top"] = False
plt.rcParams["ytick.right"] = False

# Make ticks and axis lines thicker
plt.rcParams["axes.linewidth"] = 2.0
plt.rcParams["xtick.major.width"] = 2.0
plt.rcParams["ytick.major.width"] = 2.0

# Remove minor ticks
plt.rcParams["xtick.minor.visible"] = False
plt.rcParams["ytick.minor.visible"] = False

# Adjust subplot spacing to prevent label overlap
plt.rcParams["figure.subplot.left"] = 0.2
plt.rcParams["figure.subplot.bottom"] = 0.15

##############################################################################
# Define Common Parameters
# -------------------------
# Each recruitment threshold simulation is defined by the following parameters:
#
# - ``N``: Number of motor units in the pool
# - ``recruitment_range``: Recruitment range (max_threshold / min_threshold)
#
# .. note::
#    The **recruitment_range** is defined as the ratio between the ``maximum`` and ``minimum`` recruitment thresholds.
#    For example, if the **recruitment_range** is ``50``, the biggest MU will have a **recruitment threshold** ``50`` times bigger than the smallest MU.

n_motor_units = 120  # Number of motor units in the pool
recruitment_range = 100  # Recruitment range (max_threshold / min_threshold)

# Create results directory
save_path = Path("./results")
save_path.mkdir(exist_ok=True)

##############################################################################
# Fuglevand Model
# ---------------
#
# The Fuglevand model uses a simple exponential distribution for recruitment
# thresholds. This is the classic approach from Fuglevand et al. (1993).
#
# **No additional parameters needed** - only requires the common parameters.
#
# .. important::
#    **MyoGen** is intended to be used with the following API:
#
#    .. code-block:: python
#
#       from myogen import simulator

rt_fuglevand, rtz_fuglevand = simulator.RecruitmentThresholds(
    N=n_motor_units, recruitment_range__ratio=recruitment_range, mode="fuglevand"
)

_, ax = plt.subplots()
plot_recruitment_thresholds(
    rt_fuglevand, [ax], model_name="Fuglevand", colors="#90b8e0"
)
plt.tight_layout()
plt.show()

##############################################################################
# De Luca Model
# -------------
#
# The De Luca model includes a slope correction parameter that allows control
# over the shape of the recruitment threshold distribution.
#
# **Additional parameter:**
#
# - ``deluca__slope``: Controls the shape of the distribution

deluca_slopes = [0.001, 5, 25, 50]  # Different slope values to demonstrate variety

deluca_results = {}
for slope in deluca_slopes:
    rt, _ = simulator.RecruitmentThresholds(
        N=n_motor_units,
        recruitment_range__ratio=recruitment_range,
        deluca__slope=slope,
        mode="deluca",
    )
    deluca_results[slope] = rt

_, ax = plt.subplots()
plot_recruitment_thresholds(
    deluca_results,
    [ax],
    model_name="De Luca",
)
plt.tight_layout()
plt.show()

##############################################################################
# Konstantin Model
# ----------------
#
# The Konstantin model provides explicit control over the maximum recruitment
# threshold while maintaining physiological recruitment patterns.
#
# **Additional parameter:**
#
# - ``konstantin__max_threshold``: Maximum recruitment threshold

konstantin_max_threshold = 1.0  # Maximum recruitment threshold

rt_konstantin, rtz_konstantin = simulator.RecruitmentThresholds(
    N=n_motor_units,
    recruitment_range__ratio=recruitment_range,
    konstantin__max_threshold__ratio=konstantin_max_threshold,
    mode="konstantin",
)

_, ax = plt.subplots()
plot_recruitment_thresholds(
    rt_konstantin,
    [ax],
    model_name="Konstantin",
    y_max=konstantin_max_threshold,
    colors="#90b8e0",
    markers="s",
)
plt.tight_layout()
plt.show()

##############################################################################
# Combined Model
# --------------
#
# The Combined model merges De Luca's shape control with Konstantin's scaling,
# offering the most flexibility for custom recruitment patterns.
#
# **Additional parameters:**
#
# - ``deluca__slope``: Controls the shape of the distribution
# - ``konstantin__max_threshold``: Maximum recruitment threshold

combined_slopes = [0.001, 5, 25, 50]  # Slopes for combined model
combined_max_threshold = 1.0  # Maximum threshold for combined model

combined_results = {}
for slope in combined_slopes:
    rt, _ = simulator.RecruitmentThresholds(
        N=n_motor_units,
        recruitment_range__ratio=recruitment_range,
        deluca__slope=slope,
        konstantin__max_threshold__ratio=combined_max_threshold,
        mode="combined",
    )
    combined_results[slope] = rt

##############################################################################
# Save Recruitment Thresholds
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# .. note::
#    All **MyoGen** objects can be saved to a file using ``joblib``. This is useful to **avoid re-running expensive simulations** if you need to use the same parameters.

joblib.dump(combined_results[5], save_path / "thresholds.pkl")

##############################################################################
# Plot Combined Model Results
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

_, ax = plt.subplots()
plot_recruitment_thresholds(
    combined_results,
    [ax],
    y_max=combined_max_threshold,
)
plt.tight_layout()
plt.savefig(save_path / "combined_recruitment_thresholds.svg", transparent=True)
plt.show()
