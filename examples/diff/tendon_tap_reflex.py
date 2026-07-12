"""Tendon-tap stretch reflex, in the differentiable MyoGen API.

A tap on the tendon synchronously stretches the muscle, firing a volley of Ia afferents. That
volley drives the homonymous α-motor-neuron pool *monosynaptically* — the tendon jerk — while,
through an interneuron, inhibiting the antagonist (reciprocal inhibition), and, via Renshaw
cells, damping itself (recurrent inhibition).

The circuit is built imperatively with plain objects and one ``connect`` verb, then run as a
single differentiable ``lax.scan``. Because every synapse is a trainable gain, the reflex is
differentiable w.r.t. the monosynaptic Ia→MN gain — so reflex gains can be fit to data.
"""

import jax
import jax.numpy as jnp

from myogen.diff.network import LIFPopulation, Network

jax.config.update("jax_enable_x64", True)

DT_MS, DURATION_MS = 0.05, 200.0
N_STEPS = int(DURATION_MS / DT_MS)
N = 30                       # units per population
TAP_MS = 100.0              # time of the tendon tap

# --- build the reflex circuit: populations are objects you hold and reference -----------------
net = Network(dt_ms=DT_MS, seed=0)

ia      = net.add(LIFPopulation(N, name="Ia",      tau_ms=2.0, r_input=1.0))  # fast spindle primary afferents
mn      = net.add(LIFPopulation(N, name="MN",      tau_ms=3.0, r_input=6.0))  # agonist α-motor neurons
mn_ant  = net.add(LIFPopulation(N, name="MN_ant",  tau_ms=3.0, r_input=6.0))  # antagonist α-motor neurons
ia_in   = net.add(LIFPopulation(N, name="Ia-IN",   tau_ms=3.0, r_input=6.0))  # Ia inhibitory interneurons
renshaw = net.add(LIFPopulation(N, name="Renshaw", tau_ms=3.0, r_input=6.0))  # Renshaw cells

# --- one verb wires every pathway; the comments are the physiology ----------------------------
net.connect(ia,      mn,      weight=5.0, p=0.9, delay_steps=1)                   # monosynaptic stretch reflex
net.connect(ia,      ia_in,   weight=5.0, p=0.9, delay_steps=1)                   # Ia → Ia-interneuron
net.connect(ia_in,   mn_ant,  weight=3.0, p=0.8, inhibitory=True, delay_steps=1)  # reciprocal inhibition of antagonist
net.connect(mn,      renshaw, weight=5.0, p=0.9, delay_steps=1)                   # MN → Renshaw
net.connect(renshaw, mn,      weight=2.0, p=0.6, inhibitory=True, delay_steps=1)  # recurrent inhibition (a loop)

net.emg_from(mn, n_channels=8)  # surface EMG over the agonist

# --- stimuli: tonic background drive + the tendon tap -----------------------------------------
tonic     = jnp.ones(N) * 0.15                          # agonist held just below threshold
tonic_ant = jnp.ones(N) * 0.22                          # antagonist tonically active (so inhibition shows as a pause)
tap = jnp.zeros((N_STEPS, N))                           # a 3 ms synchronous Ia volley at the tap time
tap = tap.at[int(TAP_MS / DT_MS):int((TAP_MS + 3) / DT_MS)].set(4.0)

drive = {mn: tonic, mn_ant: tonic_ant, ia: tap}

# --- run: one pure, differentiable call -------------------------------------------------------
r = net.simulate(DURATION_MS, drive=drive)

post = slice(int(TAP_MS / DT_MS), int((TAP_MS + 15) / DT_MS))   # reflex window, just after the tap
base4 = slice(int((TAP_MS - 4) / DT_MS), int(TAP_MS / DT_MS))   # 4 ms just before the tap
tap4 = slice(int(TAP_MS / DT_MS), int((TAP_MS + 4) / DT_MS))    # 4 ms during the tap (inhibition active)

print(f"Ia afferent volley (tap):        {float(r[ia].spikes[tap4].sum()):.0f} spikes")
print(f"MN reflex volley (post-tap):     {float(r[mn].spikes[post].sum()):.0f} spikes (background ~0)")
print(f"MN_ant, 4 ms before → during:    {float(r[mn_ant].spikes[base4].sum()):.0f} → {float(r[mn_ant].spikes[tap4].sum()):.0f} spikes  (reciprocal-inhibition pause)")
print(f"surface EMG:                     {r.emg.shape}")

# --- the differentiable payoff: the reflex is differentiable w.r.t. its own gains -------------
def reflex_size(params):
    out = net.simulate(DURATION_MS, drive=drive, params=params)
    return out[mn].spikes[post].sum()

grads = jax.grad(reflex_size)(net.params)
print(f"\nd(reflex) / d(Ia→MN gain):      {float(grads[ia >> mn]):+.3f}")
print(f"d(reflex) / d(Renshaw⊣MN gain): {float(grads[renshaw >> mn]):+.3f}")
print("→ the tendon jerk is differentiable w.r.t. its reflex gains — fit them to data.")
