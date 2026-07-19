"""
Extract Neuron Parameters - Jaxley Backend
==========================================

This example demonstrates how to extract electrophysiological parameters from
MyoGen motor neurons using the **Jaxley** (JAX-based) simulator instead of NEURON.

**Channels Used** (NERLab — matches production NEURON model):
    - Soma: napp (Na fast + Na persistent + Kfast + Kslow + leak)
    - Dendrite: caL (L-type Ca, no inactivation + leak)

**Voltage convention** (NERLab / original 1952-HH frame): V_rest ≈ 0 mV,
ENa = +120 mV, EK = -10 mV, spike peaks ≈ +90 mV. Spike threshold below
is +50 mV in this frame.

**Parameters Extracted**:

- **Vhold**: Resting membrane potential (soma)
- **Rin**: Input resistance (small hyperpolarizing steps, passive range)
- **tau**: Membrane time constant (small sustained step, charging transient)
- **Ir**: Rheobase (0.1 nA resolution, 50 ms pulse)
- **AP**: Action potential amplitude
- **AHP**: Afterhyperpolarization depth
- **AHPdur**: Full AHP duration
- **FI_gain**: Frequency-current gain (rheobase-relative, ascending linear range)

.. note::
    **Protocol differences from NEURON**: The stimulus protocols here are designed
    to stay within the linear/passive membrane range of the Jaxley cable model.
    Rin uses small (-0.5 to -2.5 nA) steps; tau uses a small (-1 nA) sustained
    step and fits the charging onset; rheobase uses 0.1 nA resolution.
    These are not guaranteed to match NEURON's exact protocol step-by-step, but
    they measure the same biophysical quantities in a cable-appropriate way.

**Usage**:

    # Extract from default 5 neurons
    python 10_extract_neuron_parameters.py

    # Extract from custom number of neurons
    python 10_extract_neuron_parameters.py --n-neurons 10
"""

# %%

##############################################################################
# Import Libraries
# ----------------

import logging
import os
import sys
from contextlib import contextmanager
from io import StringIO

os.environ["MPLBACKEND"] = "Agg"
if "DISPLAY" in os.environ:
    del os.environ["DISPLAY"]

from pathlib import Path

import jax.numpy as jnp
import jaxley as jx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import myogen
from myogen.simulator import RecruitmentThresholds
from myogen.simulator.jaxley.populations import AlphaMN__Pool

# Suppress Jaxley verbose output
logging.getLogger("jaxley").setLevel(logging.WARNING)
logging.getLogger("jax").setLevel(logging.WARNING)


@contextmanager
def suppress_stdout():
    """Context manager to suppress stdout (for Jaxley verbose messages)."""
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout


# Simple plotting style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_context("paper", font_scale=1.2)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Simulation parameters for NERLab cells (1952-HH voltage frame).
DT_MS = 0.025  # Time step (ms)
SPIKE_THRESHOLD_MV = 50.0  # NERLab APs cross +50 mV reliably; +90 mV at peak

##############################################################################
# Core Parameter Extraction Functions (Batched)
# ----------------------------------------------
#
# All measurement functions operate on a pre-built jx.Network containing
# ALL neurons.  Each protocol step is one jx.integrate() call that runs all
# neurons in parallel, instead of n_neurons serial calls per step.
#
# Call counts per protocol (n = number of neurons):
#   Vhold  : 1  call  (was n)
#   Rin    : 5  calls (was 5n)
#   tau    : 1  call  (was n)
#   Rh     : ~11 calls batched binary search (was ~100n linear scan)
#   AP/AHP : 1  call  (was variable × n)
#   F-I    : 12 calls (was 12n)
#   Total  : ~31 calls regardless of n  (vs ~130 × n before)


def _run_batch(net, n_neurons, current_arrays, t_max_ms: float) -> np.ndarray:
    """
    Run all neurons simultaneously with per-neuron current waveforms.

    Resets voltage/states, re-registers stimuli, calls jx.integrate once.

    Returns
    -------
    np.ndarray, shape (n_neurons, n_timesteps)
    """
    net.set("v", 0.0)         # NERLab resting potential (1952-HH frame)
    net.init_states()
    net.delete_stimuli()
    for i, arr in enumerate(current_arrays):
        net.cell(i).branch(0).loc(0.5).stimulate(jnp.array(arr))
    voltages = jx.integrate(net, delta_t=DT_MS, t_max=t_max_ms)
    return np.array(voltages)   # (n_neurons, n_timesteps)


def _run_batch_same(net, n_neurons, current_array, t_max_ms: float) -> np.ndarray:
    """Same current waveform to all neurons. Returns (n_neurons, n_timesteps)."""
    return _run_batch(net, n_neurons, [current_array] * n_neurons, t_max_ms)


def get_vholds(net, n_neurons: int) -> np.ndarray:
    """Resting potential for all neurons — 1 jx.integrate call."""
    tstop = 500.0
    current = np.zeros(int(tstop / DT_MS))
    V = _run_batch_same(net, n_neurons, current, tstop)
    tail = max(1, int(V.shape[1] * 0.1))
    return np.array([float(np.mean(V[i, -tail:])) for i in range(n_neurons)])


def get_rins(net, n_neurons: int, vholds: np.ndarray) -> np.ndarray:
    """
    Input resistance for all neurons — 5 jx.integrate calls.

    Uses small hyperpolarizing steps (-0.5 to -2.5 nA, 300 ms) to stay in
    the passive linear range and avoid slow conductance activation.
    """
    amplitudes = [-0.5, -1.0, -1.5, -2.0, -2.5]
    tstop = 300.0
    n_t = int(tstop / DT_MS)
    tail = max(1, int(n_t * 0.2))
    delta_vs = [[] for _ in range(n_neurons)]

    for amp in amplitudes:
        current = np.ones(n_t) * amp
        V = _run_batch_same(net, n_neurons, current, tstop)
        for i in range(n_neurons):
            v_ss = float(np.mean(V[i, -tail:]))
            delta_vs[i].append(v_ss - vholds[i])

    rins = []
    for i in range(n_neurons):
        try:
            coeffs = np.polyfit(amplitudes, delta_vs[i], 1)
            rins.append(coeffs[0])
        except Exception:
            rins.append(np.nan)
    return np.array(rins)


def get_taus(net, n_neurons: int, vholds: np.ndarray) -> np.ndarray:
    """
    Membrane time constant — brief impulse protocol (matches NEURON ex10).

    Injects -20 nA for 1 ms, then fits exponential recovery from the most
    hyperpolarized point back toward vhold. This isolates the fast membrane RC
    tau and avoids contamination from slow currents (Gh sag, MAHP) that bias
    sustained-step fits toward much longer time constants.
    """
    step_amp = -20.0   # nA — brief impulse
    pulse_dur = 1.0    # ms
    delay_ms = 5.0     # ms — settling before pulse
    tstop    = 200.0   # ms
    n_t      = int(tstop / DT_MS)
    delay_pts = int(delay_ms / DT_MS)
    pulse_pts = int(pulse_dur / DT_MS)

    current = np.zeros(n_t)
    current[delay_pts:delay_pts + pulse_pts] = step_amp
    V = _run_batch_same(net, n_neurons, current, tstop)

    t_array = np.arange(n_t) * DT_MS
    taus = []
    for i in range(n_neurons):
        v = V[i]
        v_min_idx = int(np.argmin(v))
        rec = np.where(v[v_min_idx:] > vholds[i] - 0.1)[0]
        v_end_idx = rec[0] + v_min_idx if len(rec) > 0 else n_t - 1
        t_fit = t_array[v_min_idx:v_end_idx]
        v_fit = v[v_min_idx:v_end_idx]
        if len(t_fit) < 5:
            taus.append(np.nan)
            continue
        log_v = np.log(np.maximum(np.abs(v_fit - vholds[i]), 1e-10))
        try:
            coeffs = np.polyfit(t_fit, log_v, 1)
            tau = -1.0 / coeffs[0]
            taus.append(max(tau, 0.1) if tau > 0 else np.nan)
        except Exception:
            taus.append(np.nan)
    return np.array(taus)


def get_rheobases(net, n_neurons: int, vholds: np.ndarray,
                  resolution: float = 0.1) -> np.ndarray:
    """
    Rheobase for all neurons via batched binary search — ~11 jx.integrate calls total.

    At each search step, ALL neurons are run simultaneously with their individual
    current amplitudes (different lo/hi brackets per neuron).  This is
    O(log₂(150/0.1)) ≈ 11 batched calls regardless of n_neurons.
    """
    tstop     = 100.0
    pulse_dur = 50.0
    n_t       = int(tstop / DT_MS)
    pulse_pts = int(pulse_dur / DT_MS)
    spike_thresholds = vholds + 40.0

    def _spiked_batch(amps: np.ndarray) -> np.ndarray:
        """Run all neurons with per-neuron amplitudes; return bool array."""
        currents = []
        for amp in amps:
            c = np.zeros(n_t)
            c[:pulse_pts] = amp
            currents.append(c)
        V = _run_batch(net, n_neurons, currents, tstop)
        return np.array([np.max(V[i]) >= spike_thresholds[i] for i in range(n_neurons)])

    # Phase 1: exponential scan to bracket each neuron's rheobase
    lo    = np.zeros(n_neurons)
    hi    = np.ones(n_neurons) * 0.5
    found = np.zeros(n_neurons, dtype=bool)

    while not np.all(found) and np.all(hi[~found] <= 150.0):
        spiked = _spiked_batch(hi)
        found  = found | spiked
        if np.all(found):
            break
        lo[~spiked] = hi[~spiked]
        hi[~spiked] *= 2.0

    no_spike = hi > 150.0
    if np.any(no_spike & ~found):
        logger.warning(f"No spike found for neuron(s): {np.where(no_spike & ~found)[0]}")

    # Phase 2: binary search to resolution within each bracket
    n_iters = int(np.ceil(np.log2(np.max(hi - lo + 1e-9) / resolution))) + 2
    for _ in range(n_iters):
        if np.max(hi - lo) <= resolution:
            break
        mid    = (lo + hi) / 2.0
        spiked = _spiked_batch(mid)
        hi[spiked]  = mid[spiked]
        lo[~spiked] = mid[~spiked]

    result = np.round(hi / resolution) * resolution
    result[no_spike & ~found] = np.nan
    return result


def get_ap_ahp(net, n_neurons: int, vholds: np.ndarray,
               rheobases: np.ndarray) -> list[dict]:
    """
    AP amplitude and AHP — brief intense impulse for guaranteed single spike.

    1 ms pulse at 100 nA delivers ~100 pC of charge — enough to depolarize any
    physiological MN past threshold regardless of membrane time constant.
    This eliminates the failure mode (~46% NaN with the previous rheobase+5 nA
    over 10 ms protocol) where slow-membrane cells failed to reach threshold
    within the brief pulse. The brief impulse also reliably elicits a single
    AP rather than a train, simplifying AHP analysis.
    """
    tstop     = 900.0
    pulse_dur = 1.0    # ms — brief impulse
    pulse_amp = 100.0  # nA — well above all rheobases
    delay     = 5.0    # ms
    n_t       = int(tstop / DT_MS)
    delay_pts = int(delay / DT_MS)
    pulse_pts = int(pulse_dur / DT_MS)
    spike_thresholds = vholds + 40.0

    current = np.zeros(n_t)
    current[delay_pts:delay_pts + pulse_pts] = pulse_amp
    V = _run_batch_same(net, n_neurons, current, tstop)

    t_array = np.arange(n_t) * DT_MS

    results = []
    for i in range(n_neurons):
        v = V[i]
        if np.max(v) < spike_thresholds[i]:
            results.append({"AP__mV": np.nan, "AHP__mV": np.nan, "AHPdur__ms": np.nan})
            continue
        peak_idx   = int(np.argmax(v))
        post_peak  = v[peak_idx:]
        valley_idx = peak_idx + int(np.argmin(post_peak))
        ap         = float(v[peak_idx] - vholds[i])
        ahp        = float(vholds[i] - v[valley_idx])
        rec = np.where(v[valley_idx:] > vholds[i] - 0.15)[0]
        ahp_dur = float(t_array[rec[0] + valley_idx] - t_array[peak_idx]) if len(rec) > 0 else np.nan
        results.append({"AP__mV": ap, "AHP__mV": ahp, "AHPdur__ms": ahp_dur})
    return results


def get_fi_gains(net, n_neurons: int, vholds: np.ndarray,
                 rheobases: np.ndarray) -> np.ndarray:
    """
    F-I gain for all neurons — 12 jx.integrate calls.

    Uses rheobase-relative current steps (rheobase_i + delta, delta in 0..11 nA).
    Each of the 12 levels is one batched call across all neurons.
    """
    tstop        = 3000.0
    n_t          = int(tstop / DT_MS)
    settle_pts   = int(500.0 / DT_MS)
    analysis_ms  = 2500.0
    deltas       = np.arange(0.0, 12.0, 1.0)   # 12 levels
    all_rates    = np.zeros((n_neurons, len(deltas)))

    for j, delta in enumerate(deltas):
        amps     = np.where(np.isnan(rheobases), 0.0, rheobases + delta)
        currents = [np.ones(n_t) * a for a in amps]
        V        = _run_batch(net, n_neurons, currents, tstop)
        for i in range(n_neurons):
            if np.isnan(rheobases[i]):
                continue
            v = V[i, settle_pts:]
            n_spikes = int(np.sum((v[:-1] < SPIKE_THRESHOLD_MV) & (v[1:] >= SPIKE_THRESHOLD_MV)))
            all_rates[i, j] = n_spikes / analysis_ms * 1000.0

    gains = []
    for i in range(n_neurons):
        rates = all_rates[i]
        if not np.any(rates > 0):
            gains.append(np.nan)
            continue
        peak_idx  = int(np.argmax(rates))
        end_idx   = min(max(peak_idx + 1, 3), len(rates))
        fit_currents = (rheobases[i] + deltas)[:end_idx]
        fit_rates    = rates[:end_idx]
        mask = fit_rates > 0
        if np.sum(mask) >= 2:
            coeffs = np.polyfit(fit_currents[mask], fit_rates[mask], 1)
            gains.append(float(coeffs[0]))
        else:
            gains.append(np.nan)
    return np.array(gains)


##############################################################################
# Visualization Functions
# -----------------------


def visualize_results(df: pd.DataFrame, save_path: Path) -> None:
    """Create basic visualization plots of extracted parameters."""
    logger.info("Creating visualizations")

    df_valid = df.dropna(subset=["Ir__nA"])
    if len(df_valid) == 0:
        logger.warning("No valid data for visualization")
        return

    x = df_valid["cell_index"] + 1

    # Figure 1: Key parameters
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Rheobase
    axes[0, 0].scatter(x, df_valid["Ir__nA"], s=100, alpha=0.7)
    axes[0, 0].set_ylabel("Rheobase (nA)")
    axes[0, 0].set_title("Rheobase Current")
    axes[0, 0].grid(True, alpha=0.3)

    # Input Resistance
    axes[0, 1].scatter(x, df_valid["Rin__MOhm"], s=100, alpha=0.7, color="C1")
    axes[0, 1].set_ylabel("Input Resistance (MΩ)")
    axes[0, 1].set_title("Input Resistance")
    axes[0, 1].grid(True, alpha=0.3)

    # AHP Duration
    axes[1, 0].scatter(x, df_valid["AHPdur__ms"], s=100, alpha=0.7, color="C2")
    axes[1, 0].set_xlabel("Motor Unit #")
    axes[1, 0].set_ylabel("AHP Duration (ms)")
    axes[1, 0].set_title("AHP Duration")
    axes[1, 0].grid(True, alpha=0.3)

    # F-I Gain
    axes[1, 1].scatter(x, df_valid["FI_gain__Hz_per_nA"], s=100, alpha=0.7, color="C3")
    axes[1, 1].set_xlabel("Motor Unit #")
    axes[1, 1].set_ylabel("F-I Gain (Hz/nA)")
    axes[1, 1].set_title("F-I Gain")
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("Jaxley NERLab Channel Parameters", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig_path = save_path / "neuron_parameters_jaxley.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved figure to {fig_path}")
    plt.show()

    # Figure 2: Rin vs Tau
    fig2, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        df_valid["Rin__MOhm"],
        df_valid["tau__ms"],
        c=df_valid["cell_index"],
        cmap="viridis",
        s=100,
        alpha=0.7,
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Motor Unit Index")
    ax.set_xlabel("Input Resistance (MΩ)")
    ax.set_ylabel("Membrane Time Constant (ms)")
    ax.set_title("Input Resistance vs Time Constant (Jaxley NERLab)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig2_path = save_path / "rin_tau_relationship_jaxley.png"
    plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved figure to {fig2_path}")
    plt.show()


##############################################################################
# Main Function
# -------------


def main():
    """Main demonstration function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract electrophysiological parameters from MyoGen motor neurons (Jaxley)"
    )
    parser.add_argument(
        "--n-neurons",
        type=int,
        default=5,
        help="Number of neurons to extract (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Setup
    save_path = Path("./results")
    save_path.mkdir(exist_ok=True)

    myogen.set_random_seed(args.seed)
    logger.info(f"Random seed set to {args.seed}")

    # Generate recruitment thresholds (combined model, matching ex01)
    logger.info(f"Generating {args.n_neurons} neurons with combined threshold model")
    recruitment_thresholds, _ = RecruitmentThresholds(
        N=args.n_neurons,
        recruitment_range__ratio=100,
        mode="combined",
        deluca__slope=5,
    )

    # Create motor neuron pool — model="NERLab" default matches production NEURON.
    logger.info("Creating motor neuron pool (NERLab — napp + caL)")
    with suppress_stdout():
        pool = AlphaMN__Pool(
            recruitment_thresholds__array=recruitment_thresholds,
            mode="active",
        )
    n_neurons = len(pool)
    logger.info(f"Created pool with {n_neurons} neurons")

    # Build a single jx.Network from all neurons.
    # All measurement functions call jx.integrate on this network once per
    # protocol step, parallelising across neurons instead of looping over them.
    logger.info("Building jx.Network for batched parameter extraction")
    with suppress_stdout():
        mn_cells = []
        for cw in pool:
            cell = cw.cell
            cell.delete_recordings()
            cell.delete_stimuli()
            mn_cells.append(cell)
        net = jx.Network(mn_cells)
        for i in range(n_neurons):
            net.cell(i).branch(0).loc(0.5).record("v")

    print("\n" + "=" * 60)
    print("Running Jaxley NERLab Parameter Extraction (batched)")
    print(f"  {n_neurons} neurons × ~31 total jx.integrate calls")
    print("=" * 60)

    logger.info("Extracting Vhold (1 call)...")
    vholds = get_vholds(net, n_neurons)

    logger.info("Extracting Rin (5 calls)...")
    rins = get_rins(net, n_neurons, vholds)

    logger.info("Extracting tau (1 call)...")
    taus = get_taus(net, n_neurons, vholds)

    logger.info("Extracting rheobase (~11 calls, batched binary search)...")
    rheobases = get_rheobases(net, n_neurons, vholds)

    logger.info("Extracting AP/AHP (1 call)...")
    ap_ahp_list = get_ap_ahp(net, n_neurons, vholds, rheobases)

    logger.info("Extracting F-I gain (12 calls)...")
    fi_gains = get_fi_gains(net, n_neurons, vholds, rheobases)

    # Assemble per-neuron result dicts
    results = []
    for i in range(n_neurons):
        results.append({
            "cell_index":             i,
            "recruitment_threshold":  float(recruitment_thresholds[i]),
            "vhold_soma__mV":         float(vholds[i]),
            "Rin__MOhm":              float(rins[i]),
            "tau__ms":                float(taus[i]),
            "Ir__nA":                 float(rheobases[i]) if not np.isnan(rheobases[i]) else np.nan,
            **ap_ahp_list[i],
            "FI_gain__Hz_per_nA":    float(fi_gains[i]) if not np.isnan(fi_gains[i]) else np.nan,
        })
        logger.info(
            f"  MN {i}: Vhold={vholds[i]:.1f} mV  Rin={rins[i]:.2f} MΩ  "
            f"Ir={rheobases[i]:.1f} nA  FI={fi_gains[i]:.2f} Hz/nA"
        )

    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    csv_path = save_path / "neuron_parameters_jaxley.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved results to {csv_path}")

    # Visualize
    visualize_results(df, save_path)

    print("\n" + "=" * 60)
    print("Summary (Jaxley NERLab Channels)")
    print("=" * 60)
    print(f"Simulator: Jaxley with NERLab channels (soma napp; dendrite caL)")
    print(f"Neurons extracted: {args.n_neurons}")
    print(f"\nParameter ranges:")
    print(f"  Rheobase: {df['Ir__nA'].min():.2f} - {df['Ir__nA'].max():.2f} nA")
    print(f"  Input resistance: {df['Rin__MOhm'].min():.2f} - {df['Rin__MOhm'].max():.2f} MΩ")
    print(f"  Time constant: {df['tau__ms'].min():.2f} - {df['tau__ms'].max():.2f} ms")
    print(f"  AP amplitude: {df['AP__mV'].min():.2f} - {df['AP__mV'].max():.2f} mV")
    print(f"  AHP depth: {df['AHP__mV'].min():.2f} - {df['AHP__mV'].max():.2f} mV")
    print(f"  F-I gain: {df['FI_gain__Hz_per_nA'].min():.2f} - {df['FI_gain__Hz_per_nA'].max():.2f} Hz/nA")

    print("\n" + "=" * 60)
    print("[DONE] Jaxley NERLab parameter extraction complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
