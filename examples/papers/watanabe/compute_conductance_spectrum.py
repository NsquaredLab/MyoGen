"""
Compute Conductance Spectrum from NEURON Simulations
=====================================================

This script demonstrates how to compute the power spectrum of synaptic conductances
rather than membrane potentials. There are two approaches:

**Approach 1: Record conductances during simulation** (RECOMMENDED)
  - Modify 10a_paper_watanabe.py to record synaptic conductances
  - Requires re-running the simulation

**Approach 2: Estimate from membrane potential** (APPROXIMATION)
  - Use membrane equation: I = C*dV/dt + g_leak*(V - E_leak)
  - Compute total conductance from V time series
  - Less accurate but works with existing data

**Why conductance spectrum?**
Membrane potential reflects the *output* of the neuron (voltage response).
Synaptic conductance reflects the *input* to the neuron (synaptic drive).
Conductance spectrum reveals the frequency content of synaptic inputs before
they're filtered by membrane time constant and active conductances.
"""

import os
os.environ["MPLBACKEND"] = "Agg"
if "DISPLAY" in os.environ:
    del os.environ["DISPLAY"]

from pathlib import Path
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots  # noqa

# Configure plotting
plt.style.use(["science", "nature"])
sns.set_context("paper", font_scale=2.5)
plt.rcParams["text.usetex"] = False
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300

print("="*70)
print("CONDUCTANCE SPECTRUM ANALYSIS")
print("="*70)

# Check if conductance data exists
script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent.parent.parent
save_path = repo_root / "results"
watanabe_path = save_path / "watanabe"
chunks_path = save_path / "watanabe_chunks"

# Load first chunk to check structure
chunk_files = sorted(chunks_path.glob("chunk_*.pkl"))
if not chunk_files:
    raise FileNotFoundError(f"No chunk files found in {chunks_path}")

first_chunk = joblib.load(chunk_files[0])
aMN_data = first_chunk['membrane_data']['aMN']

print(f"\nData availability check:")
print(f"  Chunk structure: {first_chunk.keys()}")
print(f"  membrane_data contains: {first_chunk['membrane_data'].keys()}")

# Check if conductance data exists
first_mn_idx = list(aMN_data.keys())[0]
if isinstance(aMN_data[first_mn_idx], dict):
    print(f"  First neuron data type: dict with keys {aMN_data[first_mn_idx].keys()}")
    has_conductance = any('g' in key.lower() or 'cond' in key.lower()
                          for key in aMN_data[first_mn_idx].keys())
else:
    print(f"  First neuron data type: array with shape {aMN_data[first_mn_idx].shape}")
    print(f"  This appears to be voltage-only data")
    has_conductance = False

if has_conductance:
    print("\n✓ Conductance data found! Computing conductance spectrum...")
    # TODO: Add code to compute spectrum from recorded conductances
else:
    print("\n✗ No conductance data found in chunks.")
    print("\nCurrent simulation only saved membrane potentials (voltage).")
    print("\nOPTION 1: Modify simulation to record conductances (RECOMMENDED)")
    print("-" * 70)
    print("In continuous_saver.py, modify record_step() to save conductances:")
    print("""
    def record_step(self, timestep__ms: float):
        # ... existing code ...

        for cell_idx in cell_indices:
            if cell_idx < len(population):
                # Record voltage (existing)
                voltage = population[cell_idx].soma(0.5).v

                # NEW: Record synaptic conductances
                # Total excitatory conductance (AMPA/NMDA)
                g_exc = population[cell_idx].soma(0.5).g_exc

                # Total inhibitory conductance (GABA)
                g_inh = population[cell_idx].soma(0.5).g_inh

                # Store in dict instead of single value
                self.current_chunk_data[pop_name][cell_idx].append({
                    'voltage': voltage,
                    'g_exc': g_exc,
                    'g_inh': g_inh,
                    'g_total': g_exc + g_inh
                })
    """)
    print("\nThen re-run 10a_paper_watanabe.py to generate new chunks with conductances.")

    print("\n\nOPTION 2: Estimate total conductance from voltage (APPROXIMATION)")
    print("-" * 70)
    print("Use membrane equation to estimate conductance changes:")
    print("  dV/dt = (I_syn - g_leak*(V - E_leak)) / C_m")
    print("  g_total ≈ C_m * dV/dt / (V - E_reversal)")
    print("\nThis is an approximation because:")
    print("  - Assumes passive membrane (ignores Na/K/Ca currents)")
    print("  - Single compartment (ignores dendritic filtering)")
    print("  - Unknown E_reversal (mixed excitation/inhibition)")
    print("\nWould you like me to implement this approximation? (y/n)")

    # For now, demonstrate with voltage spectrum as comparison
    print("\n\n" + "="*70)
    print("DEMONSTRATION: Voltage Spectrum vs. Approximate Conductance Spectrum")
    print("="*70)

    # Passive membrane parameters (typical values for motor neurons)
    C_m__uF_per_cm2 = 1.0  # Membrane capacitance
    g_leak__mS_per_cm2 = 0.1  # Leak conductance
    E_leak__mV = -70.0  # Leak reversal potential
    E_syn__mV = 0.0  # Mixed synaptic reversal (between excitation and inhibition)
    soma_area__cm2 = 0.01  # Soma surface area (~100 μm diameter)

    # Scale to soma
    C_m = C_m__uF_per_cm2 * soma_area__cm2  # μF
    g_leak = g_leak__mS_per_cm2 * soma_area__cm2  # mS

    print(f"\nUsing passive membrane parameters:")
    print(f"  C_m = {C_m} μF")
    print(f"  g_leak = {g_leak} mS")
    print(f"  E_leak = {E_leak__mV} mV")
    print(f"  E_syn = {E_syn__mV} mV (assumed)")

    # Load voltage data for one neuron from Phase 3
    print(f"\nLoading voltage data from Phase 3 (1 second window)...")
    phase3_start_ms = 123000  # 123 seconds
    phase3_end_ms = 124000  # 124 seconds

    # Find relevant chunks
    voltage_data = []
    time_data = []

    selected_neuron = list(aMN_data.keys())[len(aMN_data)//2]  # Middle neuron

    for chunk_file in chunk_files:
        chunk = joblib.load(chunk_file)

        if chunk['time_end'] < phase3_start_ms or chunk['time_start'] > phase3_end_ms:
            continue

        if selected_neuron in chunk['membrane_data']['aMN']:
            voltage_data.append(chunk['membrane_data']['aMN'][selected_neuron])
            time_data.append(chunk['times'])

    if voltage_data:
        V = np.concatenate(voltage_data)
        t = np.concatenate(time_data)

        # Filter to exact window
        mask = (t >= phase3_start_ms) & (t < phase3_end_ms)
        V = V[mask]
        t = t[mask]

        dt = first_chunk['timestep__ms']

        print(f"  Loaded {len(V)} samples at dt={dt} ms")
        print(f"  Voltage range: {V.min():.1f} to {V.max():.1f} mV")

        # Compute dV/dt using central difference
        dV_dt = np.gradient(V, dt)  # mV/ms

        # Estimate total synaptic conductance
        # From: dV/dt = (I_syn - g_leak*(V - E_leak)) / C_m
        # Rearrange: I_syn = C_m * dV/dt + g_leak * (V - E_leak)
        # Then: g_syn = I_syn / (V - E_syn)

        I_leak = g_leak * (V - E_leak)  # μA
        I_total = C_m * dV_dt + I_leak  # μA

        # Avoid division by zero
        denominator = V - E_syn__mV
        denominator[np.abs(denominator) < 1.0] = 1.0  # Clip small values

        g_syn_estimate = I_total / denominator  # mS

        # Compute power spectra
        from scipy import signal as scipy_signal

        fs = 1000.0 / dt  # Sampling frequency in Hz
        nperseg = min(8192, len(V))

        # Voltage spectrum
        f_V, psd_V = scipy_signal.welch(V, fs=fs, nperseg=nperseg,
                                        scaling='density')

        # Conductance spectrum (estimate)
        f_g, psd_g = scipy_signal.welch(g_syn_estimate, fs=fs, nperseg=nperseg,
                                        scaling='density')

        # Plot comparison
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Voltage spectrum
        ax = axes[0]
        ax.semilogy(f_V, psd_V, 'b-', linewidth=2, alpha=0.8)
        ax.axvline(20, color='red', linestyle='--', linewidth=2, label='20 Hz drive')
        ax.set_xlim(0, 100)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD (mV²/Hz)')
        ax.set_title(f'Membrane Potential Spectrum (MU #{selected_neuron}, Phase 3)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        sns.despine(ax=ax)

        # Conductance spectrum (estimate)
        ax = axes[1]
        ax.semilogy(f_g, psd_g, 'g-', linewidth=2, alpha=0.8)
        ax.axvline(20, color='red', linestyle='--', linewidth=2, label='20 Hz drive')
        ax.set_xlim(0, 100)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD (mS²/Hz)')
        ax.set_title('Estimated Synaptic Conductance Spectrum (Approximation)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        sns.despine(ax=ax)

        plt.tight_layout()
        plt.savefig(watanabe_path / "voltage_vs_conductance_spectrum_comparison.pdf",
                    dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: {watanabe_path / 'voltage_vs_conductance_spectrum_comparison.pdf'}")

        print("\n" + "="*70)
        print("IMPORTANT NOTES ON APPROXIMATION")
        print("="*70)
        print("This estimated conductance spectrum is an APPROXIMATION:")
        print("  ✓ Shows general frequency content of synaptic inputs")
        print("  ✓ Useful for qualitative comparison")
        print("  ✗ Not quantitatively accurate (active currents ignored)")
        print("  ✗ Reversal potential assumed constant (unrealistic)")
        print("\nFor accurate conductance analysis, use OPTION 1 and re-run simulation")
        print("with conductance recording enabled in continuous_saver.py")
        print("="*70)

    else:
        print("  Could not load voltage data for Phase 3")
