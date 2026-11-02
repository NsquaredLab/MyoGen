# Settings inventory for named simulation profiles

**Status:** Draft (input for backlog item 1, "Named simulation profiles", `docs/backlog.md`).
**Snapshot:** every `file:line` refers to commit `3bbde23eb` on `calib/fdi-dd-main`. The code was read from a `git archive` of that commit, so later edits to `surface/*` (in progress 2026-11-03) shift line numbers.
**Method:** code read only, nothing run. "(inferred)" marks a statement reasoned from the code and not stated in code, docs or the register. "none found" in the source column means there is no register key, locator or author-year in the code or `docs/literature.md`. Citations are quoted exactly as they are written in the code or register.
**Flags:** **OFF** = off by default · **OPT-IN** = the better or physiological variant has to be requested · **NON-PHYS** = the default is non-physiological, or only effective or fitted · **DEAD** = the setting has no effect · **ASSUMPTION** = a modelling assumption with no measurement · **NUM** = numerical. The rest are physiological defaults or neutral.

## 1. Summary

**249 table rows** (some group related fields): neuron 95, muscle 35, force 12, sEMG 54, iEMG 32, optimiser 21. Flagged rows: **15 OFF, 7 OPT-IN, 20 NON-PHYS, 9 DEAD**. Section 3 lists 15 interaction groups.

Top 10 easy-to-miss settings, ranked by how much signal quality a user loses without knowing they exist:
1. **Gfluctdv noise OFF**: no pool calls `insert_Gfluctdv()`; its unscaled values are not the calibrated FDI scale (×0.358).
2. **iEMG output has no physical amplitude**: MUAPs are divided by the largest one (dimensionless; no switch; `intramuscular_emg.py:626-628`).
3. **`ForceModel(force_frequency_shape)` = Fuglevand (2, 3)**: the human-unit curve (0.93, 1.29) is OPT-IN; Fuglevand saturates by ~20 Hz.
4. **`ForceModel(twitch_force_range=None)`**: RP = threshold ratio (30 in the optimiser, 50 in the FDI calibration), not ~100.
5. **`SurfaceEMG(fiber_dispersion=None)` OFF**: one end-plate and one CV per unit, no nerve delays, so MUAPs are over-coherent.
6. **sEMG uses one CV (4.2) and one diameter (57.9 µm)** for every fibre; Muscle per-fibre CVs, diameters and types reach iEMG only.
7. **iEMG `execution=` route uses the pre-0.13.0 kernel** (half-length IAP, mirrored wave subtracted, no `* dz`).
8. **`SurfaceElectrodeArray(bending_radius__mm=0)` becomes eps**: grids with more than one column get meaningless angles; the default centre z = 0 sits on the IZ (inferred).
9. **`iap_lambda__per_mm = 1.0`** (FDI fit 0.7 not adopted); iEMG hard-codes λ = 1.
10. **Library DD = Poisson, `Network.connect` all-to-all at 0.6 µS**: every FDI calibration used gamma 3, equal fan-in and 0.05 µS transfer-scaled weights.

## 2. Settings by subsystem

The columns are the same in every table: Location · Name · Default (exact) · What it changes · Physiological/literature value + source · Flag · Must stay consistent with · Per-muscle.

### 2.1 Neuron pool and spinal network

The branch has no common-drive, beta-band or effort-fluctuation module. A grep of `myogen/` for `beta|effort|common_drive|fluct` finds only napp gate names, the sEMG `k_beta` and `insert_Gfluctdv`. Outside the library, the only caller of `insert_Gfluctdv` is the DD optimiser (`examples/finetune/01_optimize_dd_multilevel.py:155-165`).

**A. Cell model and pool level (`alpha_mn_default.yaml`, `AlphaMN__Pool`)**

| Location | Name | Default | What it changes | Source | Flag | Consistent with | Per-muscle |
|---|---|---|---|---|---|---|---|
| myogen/config/alpha_mn_default.yaml:5; populations/motor_neurons.py:117,220,372-377; neuron/cells.py:769 | `model` | `"NERLab"` | Mechanism set, rheobase, f–I curve, PIC, AHP, voltage frame | NERLab: none found. Powers2017: `[powersheckman2017]` (yaml:89, "values not checked") | Docstrings disagree with the code: cells.py:730 says "Powers2017", motor_neurons.py:27 says "ModALS" | YAML block, `initial_voltage__mV`, `spike_threshold__mV`, synapse reversals, Gfluctdv `E_e`/`E_i`, NetCon thresholds, `celsius` | generic |
| yaml:6; cells.py:993 | `mode` | `"active"` | Active vs passive membrane | none found | Passive mode is broken: the code checks `"pas"` but the Literal is `"passive"` | — | generic |
| yaml:7; motor_neurons.py:509,639 | `axon_velocities` | `[50, 65]` | MN axon delay on NetCons and on the muscle callback. Recorded soma spikes do not include it | none found | Unsourced | `axon_length` | generic |
| yaml:8; motor_neurons.py:531,664 | `axon_length` | `0.6` (comment "(mm)") | Used as `* pq.m`, giving 9–12 ms delay | none found | **NON-PHYS label**: the unit comment contradicts the code | `axon_velocities` | generic |
| yaml:9; motor_neurons.py:224,592,689 | `gamma` | `0.2` | Scales the dendritic L-Ca PIC (NERLab `gama_caL`; Powers `gcabar_L_Ca_inact`) | none found. The register's beauchamp2023 entry gives "MyoGen NERLab pool ΔF 0.2 pps" vs human FDI ΔF 4.80 | **NON-PHYS** (unsourced; PIC inert) | `gcaLbar_range`, Powers `ca_conductance_ranges`, DD calibration | generic |
| yaml:10; motor_neurons.py:671 | `lambda_factor` | `1.0` | Multiplies Powers `gbar_naps` | `[powersheckman2017]` (yaml comment) | No effect under NERLab | `naps` range | generic |
| yaml:11; populations/base.py:128-136 | `initial_voltage__mV` | `-67` | v at t = 0 | none found | **NON-PHYS for NERLab**: that model rests at ≈ 0 mV (backlog), so the run starts with a transient | Model frame | generic |
| yaml:12; network.py:1000-1003 | `spike_threshold__mV` | `50.0` | Threshold of the recording NetCons only, so it decides which spikes reach Neo | none found | Model-dependent. Under Powers2017 `ena` is unset, so 50 mV may be unreachable (inferred) | `DEFAULT_SPIKE_THRESHOLD` (-10), injection-helper threshold (-10) | generic |
| motor_neurons.py:115,214-215,392-403 | `recruitment_thresholds__array` | `None` | If given, every YAML range is mapped **linearly** in RT and `curve` is ignored. NERLab ranges with `negative: true` are reversed. If None, `_exp_interp` with curvature is used | none found | Buried. The optimiser always passes it, so every `curve` value is unused in production | All YAML ranges, DD weights, ForceModel thresholds | generic |
| populations/base.py:45-60 | `_exp_interp(curv=0.33, negative=False)` | `0.33`, `False` | Size gradient when no RT array is given | none found | Unclear | YAML `curve` | generic |
| motor_neurons.py:520-522,649-651 | segment / dendrite counts (hard-coded) | NERLab `1`/`1`; Powers `1`/`4` | Cable discretisation; total Gfluctdv current (1 vs 4 dendrites) | none found | Buried, not exposed | Gfluctdv scale | generic |
| motor_neurons.py:153-165; utils/config_loader.py:61-80,119-151 | `config_file` + merge | `"alpha_mn_default.yaml"`, recursive merge | Precedence: kwarg > specific YAML > default YAML. NERLab keys can only be overridden through YAML | — | Buried | — | generic |

**B. NERLab YAML ranges** (`nerlab:`, `[first, last, curve, negative]`; the comment says `[min, max, curve]`). None of these has a source.

| Location | Name | Default | What it changes | Source | Flag | Consistent with | Per-muscle |
|---|---|---|---|---|---|---|---|
| yaml:18 | soma `diameter_range` | `[78, 113, 14, false]` | Soma area, rheobase | none found | Unsourced | `gls_range` | generic |
| yaml:19 | `gnabar_range` | `[0.0325, 0.0775, 2.5, false]` | Fast Na | none found | Unsourced | — | generic |
| yaml:20 | `gnapbar_range` | `[0.00043, 0.00067, 2.1, true]` | Persistent Na | none found | Reversed on the RT path | `gamma` | generic |
| yaml:21 | `gkfbar_range` | `[0.0028, 0.0015, 25, true]` | Fast K | none found | Unsourced | — | generic |
| yaml:22 | `gksbar_range` | `[0.02, 0.016, 6.0, true]` | Slow K: AHP, adaptation, maximum rate | none found | Unsourced | `rinact` | generic |
| yaml:23; napp.mod:29,150-189 | `mact_range` | `[13, 20, 3, false]` | Nothing (declared but never used in `rates()`) | none found | **DEAD** | — | generic |
| yaml:24 | `rinact_range` | `[0.018, 0.063, 4, false]` | AHP duration | none found | Unsourced | `gksbar` | generic |
| yaml:25 | `gls_range` | `[0.000952381, 0.001538462, 2.5, false]` | Soma leak | none found | Unsourced | Diameter | generic |
| yaml:28-31 | `ena`, `ek`, `el_napp`, `vtraub_napp` | `120.0`, `-10.0`, `0.0`, `0.0` | Voltage frame with rest at ≈ 0 mV | none found | **NON-PHYS** absolute frame (by design); napp.mod:9-11 says "-65 mV" | v_init, synapse `e`, Gfluctdv `E_e`/`E_i`, thresholds | generic |
| yaml:32-33 | soma `Ra`, `cm` | `70.0`, `1.0` | Soma cable | none found | Unsourced | — | generic |
| yaml:37 | dendrite `diameter_range` | `[48.0, 90.0, 5.0, false]` | Dendritic area and coupling | none found | Unsourced | — | generic |
| yaml:38 | dendrite `length_range` | `[5500, 10600, 12.0, false]` | Distance from synapse to soma | none found | Unsourced | DD dendritic-transfer weights | generic |
| yaml:39 | `gcaLbar_range` | `[1.25e-5, 6.2e-6, 3, true]` | L-Ca PIC conductance | none found | Unsourced | `gamma` | generic |
| yaml:40; caL.mod:76,81 | `vtraub_caL_range` | `[35, 34, 30, true]` | PIC half-activation; the slope is hard-coded at 1 mV | none found | Unsourced | Frame | generic |
| yaml:41 | `ltau_caL_range` | `[90, 47, 3, true]` | PIC activation τ | none found | Unsourced | — | generic |
| yaml:42 | `gl_caL_range` | `[7.69231e-5, 0.000165289, 2.5, false]` | Dendritic leak | none found | Unsourced | — | generic |
| yaml:45-48 | dendrite `Ra`, `cm`, `ecaL`, `el_caL` | `70.0`, `1.0`, `140`, `0.0` | Cable; PIC driving force | none found | Unsourced | Frame | generic |
| yaml:53-87 | napp m/h/p/n/r gate kinetics | Summarised (e.g. `m_alpha_A 0.64, v_off 15, k 4`) | Spike threshold and AHP kinetics | none found | Duplicates napp.mod:52-57 exactly | napp.mod | generic |

**C. Powers2017 YAML** (`powers2017:`, only when `model="Powers2017"`). The block comment says `[powersheckman2017]` "values not checked".

| Location | Name | Default | What it changes | Source | Flag | Consistent with | Per-muscle |
|---|---|---|---|---|---|---|---|
| yaml:94-98 | soma length / diameter / capacitance / passive g / passive e ranges | `[2952, 3665, 0.3]`, `[22, 30, 0.3]`, `[1.35546, 1.87853, 0.3]`, `[8.11e-5, 3.77e-4, 0.3]`, `[-71, -72, 0.3]` | Soma passive properties | `[powersheckman2017]` (block) | Unchecked | cells.py:811-827 hard-coded copies | generic |
| yaml:99-105 | na3rp / naps / kdrRL / mAHP Ca / mAHP K / mAHP τ / gh ranges | `[0.01, 0.022, 0.3]`, `[2.6e-5, 2.0e-5, 0.3]`, `[0.015, 0.02, 0.3]`, `[6.4e-6, 1.015e-5, 0.075]`, `[4.5e-4, 6.0e-4, 0.3]`, `[90, 30, 0.3]`, `[3.0e-5, 2.3e-4, 0.3]` | Soma active properties: spike, PIC, AHP, Ih | same | Unchecked; `ena` is never set | `lambda_factor` | generic |
| yaml:109-115 | dendrite length / diameter / passive / Ra / cm / gh ranges | `[1794.13, 2226.91, 0.3]`, `[8.73071, 11.9055, 0.3]`, `[7.93e-5, 1.75e-4, 0.3]`, `[-71, -72, 0.3]`, `[51.038, 40.755, 0.3]`, `[0.867781, 0.880407, 0.3]`, `[3.0e-5, 2.3e-4, 0.3]` | Dendritic cable | same | Unchecked | — | generic |
| yaml:118-125 | `ca_conductance_ranges` (4 dendrites), `ca_theta_m_range`, `ca_theta_h_range` | d1 `[8.5e-5, 1.18e-4]` … d4 `[1.15e-4, 1.53e-4]`; `[-42, -39, 0.3]`; `[10, -10, 0.3]` | Dendritic PIC (× `gamma`) | same | Unchecked; scaled to 20 % by `gamma` | `gamma` | generic |

**D. Cell-level settings, synapses and Gfluctdv insertion (`neuron/cells.py`)**

| Location | Name | Default | What it changes | Source | Flag | Consistent with | Per-muscle |
|---|---|---|---|---|---|---|---|
| cells.py:164-166,212-214 | Exp2Syn rise, decay, location | `0.2 * pq.ms`, `0.3 * pq.ms`, `synapse_location(0.5)` | PSC time course | "Sanches (2018) [sanches2018]" (register: "Cited in code; not checked") | Unchecked | dt, weight | generic |
| cells.py:164,215-219 | Exp2Syn `reversal_potential__mV` | `0 * pq.mV`; replaced by `syn.e = 70` when `model == "NERLab"` | Excitatory driving force | none found | Buried auto-switch | Frame | generic |
| cells.py:786-788 | AlphaMN inhibitory synapse | `reversal_potential__mV=-75 * pq.mV` | Inhibitory driving force | none found | **NON-PHYS for NERLab**: -75 mV in a rest-≈0 frame (backlog) | `INHIBITORY_REVERSAL_THRESHOLD` | generic |
| cells.py:766-769 | AlphaMN constructor | `segments__count=1`, `mode="active"`, `dendrites__count=4`, `model="NERLab"` | Standalone-cell defaults; the pool overrides them | `[powersbinder2001]`, `[powersheckman2017]` | Docstring says the default is Powers2017 | Pool | generic |
| cells.py:811-827 | AlphaMN geometry literals | soma `L 2952, diam 22, Ra 0.001, cm 1.35546`; dendrites `1794.13/8.73071` or `2848/22` | Morphology. Soma `Ra 0.001` survives for Powers | none found | Duplicates the YAML minima | YAML `powers2017` | generic |
| cells.py:950-972,1012 | Powers values the pool does not overwrite | `sh_na3rp 1.0, ar_na3rp 1.0, qinf_na3rp 8.0, thinf_na3rp -50; sh_naps 5.0, ar_naps 1.0; ek -80.0; half_gh -77.0` | Spike threshold, PIC voltage, Ih | none found | Buried (not in YAML) | YAML | generic |
| cells.py:975-990 | Powers mechanism GLOBALs `h.*` | `vslope_naps 5, asvh_naps -90, bsvh_naps -22, mvhalfca_mAHP -22, mtauca_mAHP 2, tau_m_L_Ca_inact 40, tau_h_L_Ca_inact 2500.0, kappa_h_L_Ca_inact 5.0, mVh_kdrRL -21.0, tmin_kdrRL 0.8, taumax_kdrRL 20.0, htau_gh 30.0` | Kinetics for the whole process, including interneurons | none found | Buried; depends on build order | Interneuron kinetics | generic |
| cells.py:873,997-1014 | `_define_biophysics(gamma=1)` + hard-coded `gca` | `gamma=1` (never passed); `gca [8.5e-05, 9.5e-5, 1e-4, 1.15e-4]` | Overwritten by the pool | none found | **DEAD** / duplicate | YAML Ca ranges | generic |
| cells.py:915-923; constant.mod:11 | `Constant` mechanism | `ic = 0` | Inert current hook, never set | none found | **DEAD** | — | generic |
| cells.py:840-871 | `insert_Gfluctdv()` (setup call) | Not called by any pool or constructor | Stochastic dendritic conductance noise: ISI CV, discharge variability, recruitment jitter | Mechanism: `destexhe2001` (register, SEC); the parameter values have no source | **OFF** | DD calibration (`gfluctdv_scale`), dendrite count, frame | generic |
| cells.py:860-861; Gfluctdv.mod:93,118-119 | `h.multex_Gfluctdv`, `h.multin_Gfluctdv` | `.mod`: `0`, `0`; set to `1` by `insert_Gfluctdv` | Master gains: mean × mult, std × √mult | none found | **OFF** in the `.mod`; a GLOBAL, so it switches every cell in the process | `insert_Gfluctdv` | generic |
| cells.py:865,869 | `g_e0_Gfluctdv`, `g_i0_Gfluctdv` | `1e-5`, `1e-5` (S/cm²) | Mean background conductance (shunt) | none found; the `.mod` values `0.0001`/`0.0005` are overridden | Unsourced; the optimiser multiplies by its scale | Optimiser `scale` | generic |
| cells.py:866,870 | `std_e_Gfluctdv`, `std_i_Gfluctdv` | `1.2e-5`, `1.2e-5` | Noise amplitude, ISI CV | none found ("~2-3 mV RMS" is unsourced) | std > mean, so clipping at g < 0 is frequent (inferred) | Optimiser `scale` | generic |
| cells.py:867,871 | `tau_e_Gfluctdv`, `tau_i_Gfluctdv` | `20`, `20` (ms) | Noise bandwidth | none found; the `.mod` values `2.728`/`10.49` are overridden | Unsourced | dt | generic |
| Gfluctdv.mod:106-107 | `E_e`, `E_i` | `0 (mV)`, `-75 (mV)` (not set from Python) | Noise reversal potentials | Destexhe et al. 2001 (`.mod` header) | **NON-PHYS for NERLab**: `E_e` sits at rest, so the net effect is shunting or inhibitory (inferred) | Frame | generic |
| Gfluctdv.mod:161,163,180-185 | rectification; RNG | `if(g_e < 0) { g_e = 0 }`; `normrand`, `new_seed` never called | Skews the noise; noise is outside `myogen.set_random_seed` | — | Buried, unseeded | Seed | generic |
| cells.py:410-438 | `DD` Poisson cell | Seed `derive_subseed(class__ID, global__ID)`; spikes if `y > 0` | DD spike train; the rate is in Hz (the docstring says "(0-1)") | none found | Docstring contradicts code | `timestep` = `h.dt` | generic |
| cells.py:495 | `DD_Gamma.shape` | `3.0` | DD ISI CV ≈ 0.58 | none found | Unsourced | Pool `shape` | generic |
| cells.py:596,624 | `AffIa.IFR` | `get_random_generator().normal(5, 2.5)` | Per-afferent rate offset | none found | Unsourced | Afferent RT | generic |
| cells.py:76-81,135-136 | `create_axon` defaults | `0 * pq.m`, `50 * pq.m / pq.s`, then reset to `None` | Nothing | none found | **DEAD** | — | generic |
| cells.py:286-332 | INgII geometry and biophysics | `Amu = 81390 + 3113`; `g_pas 5e-5, e_pas -71.0, ghbar_gh 2.5e-5, half_gh -70.0, ek -80.0, gbar_na3rp 0.003, …, tau_mAHP 70.0` | Interneuron excitability | "[bui2003]" (register: "not checked") | Unchecked | Powers GLOBALs | generic |
| cells.py:271; network.py:80-82 | Interneuron synapse | One excitatory `create_synapses(self.soma)` (e = 0) | Inhibitory connections onto interneurons fall back to it and become excitatory | none found | Bug (buried) | `Network` `inhibitory` | generic |

**E. Other NMODL parameters that Python relies on**

| Location | Name | Default | What it changes | Source | Flag | Consistent with | Per-muscle |
|---|---|---|---|---|---|---|---|
| napp.mod:95; caL.mod:38 | Temperature | No Q10 (`celsius` unused) | NERLab is temperature-independent | — | — | — | generic |
| na3rp.mod:29,99 | `q10`, reference temperature | `2`, `24` °C | Na kinetics of Powers MNs and all interneurons | none found | Depends on `h.celsius` | `celsius` | generic |
| na3rp.mod:45; naps.mod:15; mAHP.mod:81; gh.mod:23 | `ena`, `eca`, `eh` | Not set from Python ("must be explicitly def. in hoc"); `eh -41` | AP peak, Ca and Ih reversals | none found | Buried / unset | `spike_threshold__mV` | generic |
| naps.mod:13,18-19; kdrRL.mod:22,26-27; mAHP.mod:34-36; gh.mod:20; L_Ca_inact.mod:26-33 | `.mod` defaults overridden by the GLOBALs above | e.g. `vslope 6.8` → 5; `mVh -25` → -21; `htau 50` → 30; `tau_h 1500` → 2500 | — | none found | Two authorities for each value | cells.py:975-990 | generic |
| caL.mod:22-29,52,81 | `gcaLbar`, `gl`, `el`, `vtraub`, `gama`, `Ltau`; Linf slope | `.030`, `.0003`, `-54.3`, `70.0`, `1`, `20`; slope `-1` mV | Overridden by YAML except the slope | none found | Slope buried | `gamma` | generic |
| dummy.mod:29-31 | `DUMMY` NET_RECEIVE | `net_event(t)` | External → DD NetCon weight is ignored (inferred) | — | — | `connect_from_external` weight | generic |

**F. Interneuron, descending-drive and afferent pools** (format: GII / GIb where both are given)

| Location | Name | Default | What it changes | Source | Flag | Consistent with | Per-muscle |
|---|---|---|---|---|---|---|---|
| interneurons.py:64-65/198-199; base.py:63-79 | Soma size ranges | `_get_interneuron_diameter_range__um()` | Interneuron size | "Bui et al. 2003 [1]" (docstring) | Unchecked | — | generic |
| interneurons.py:66-72/200-206 | Passive, na3rp, kdrRL, mAHP Ca/K/τ, gh ranges | `(3e-5, 7e-5)`/`(3e-5, 8e-5)`; `(0.003, 0.01)`/`(0.01, 0.03)`; `(0.015, 0.015)`/`(0.035, 0.028)`; `(3e-6, 3e-6)`, `(5e-4, 5e-4)`, `(60, 70)`/`(1e-6, 6e-6)`, `(3e-4, 4.5e-4)`, `(120, 90)`; `(2.5e-5, 2.5e-5)` | Interneuron excitability and maximum rate | none found | Unsourced | `celsius`, Powers GLOBALs | generic |
| interneurons.py:73-74/207-208 | `axon_velocities`, `axon_length` | `(10, 10)`, `0.05` (docstring "(mm)", used `* pq.m`) | 5 ms delay | none found | Unit label contradicts code | — | generic |
| interneurons.py:76/210; base.py:122 | `initial_voltage__mV`; pool `spike_threshold__mV` | `-70.0`; `-10.0` | Initial state; recording threshold | none found | Physiological for this frame | — | generic |
| descending_drive.py:44,48-49 | `timestep__ms` | `None` (raises) | Generator step | — | Required | **Must equal `h.dt`** | generic |
| descending_drive.py:45,56-66 | `process_type` | `"poisson"` | DD ISI CV 1 (Poisson) vs 1/√shape (gamma), which sets input irregularity | none found | **Differs from every calibration**, which used `"gamma"` | DD calibration | generic |
| descending_drive.py:46,104 | `shape` | `3.0` | Gamma CV ≈ 0.58 | none found | Unsourced | Optimiser `GAMMA_SHAPE` 3.0 | generic |
| _cython/_poisson_process_generator.pyx:153-158; _gamma_process_generator.pyx:65,201 | Generator integration | `yi += y*dt*1e-3`; at most one spike per step; first threshold drawn fresh at t = 0 | Exact only as dt → 0; possible synchrony at onset (inferred) | — | NUM | dt | generic |
| afferents.py:50/116/182 | `recruitment_thresholds` (Ia/II/Ib) | `(0, 40)` | Afferent rate threshold | none found | Unsourced | Spindle/GTO rates | generic |
| afferents.py:51-54/117-120/183-186 | `axon_velocities__m_per_s` | Ia `(61, 75)`, II `(30, 50)`, Ib `(64, 72)` | Conduction delay | none found | Unsourced | Axon length | generic |
| afferents.py:55/121/187 | Axon length | Ia `0.6 * pq.m`, II `0.6 * pq.m`, **Ib `0.6 * pq.mm`** | Delay: Ia/II ≈ 8–20 ms, Ib ≈ 0.01 ms | none found | Ib is 1000× shorter (likely a unit slip, inferred) | Ia/II | generic |
| afferents.py:56/122/188 | `shape` | `145`, `772`, `145` | Afferent regularity (CV 8.3 %, 3.6 %) | none found | Unsourced | — | generic |
| afferents.py:57; cells.py:589,599-604 | `init_order` / `initN` | `0` | Nothing | — | **DEAD** | — | generic |

**G. Network, NEURON globals and current injection**

| Location | Name | Default | What it changes | Source | Flag | Consistent with | Per-muscle |
|---|---|---|---|---|---|---|---|
| network.py:28 | `INHIBITORY_REVERSAL_THRESHOLD` | `-40.0` | Classifies synapses as excitatory or inhibitory by `e` | none found | — | Synapse reversals | generic |
| network.py:33 | `DEFAULT_SYNAPTIC_WEIGHT` | `0.6 * pq.uS` | Default PSP size | `[sanches2018]` (cells.py:199) | Production (optimiser) uses 0.05 µS scaled by dendritic transfer | DD calibration | generic |
| network.py:34 | `DEFAULT_SPIKE_THRESHOLD` | `-10.0 * pq.mV` | Presynaptic detection for every connection, including `connect_to_muscle` | none found | Under NERLab (rest ≈ 0) it may be crossed only at start-up (inferred) | `spike_threshold__mV` 50 | generic |
| network.py:35,187-195 | `DEFAULT_SYNAPTIC_DELAY` | `1.0 * pq.ms` (+ axon delay) | Synaptic delay | none found | — | Axon settings | generic |
| network.py:1015-1020,309-361 | `Network.connect(probability, deterministic, inhibitory)` | `1.0`, `False`, `False` | All-to-all by default (fully common input); binomial fan-in when p < 1 | none found | Differs from the optimiser's equal fan-in | DD `conn_probability` | generic |
| network.py:56-102 | `_select_synapse` | Random choice among matching synapses | Which dendrite receives each input | — | Buried randomness | Seed | generic |
| network.py:1110-1118,1132 | `connect_to_muscle` | Weight, delay and threshold defaults as above | MN → Hill events | none found | Docstring says weight 1.0, code 0.6 | — | generic |
| network.py:1187-1236 | `connect_from_external` | `0.6 uS` (docstring 0.8), `1.0 ms`, `-10 mV` | External → DD (weight ignored by DUMMY, inferred) | none found | Docstring contradicts code | — | generic |
| network.py:903,1453-1502 | Demo reflex network | DD → aMN p 0.9 / w 0.6; Ia → aMN 0.9 / 0.6; … | Example only | "table 7, pg 104 [Elias PhD tesis]" (not in the register) | Demo, not a default | — | generic |
| simulation_runner.py:58,253 | `temperature__celsius` → `h.celsius` | `36.0` | Powers and interneuron Na kinetics (Q10 2) | none found | Set only on this path | Injection path (NEURON default 6.3 °C, inferred) | generic |
| simulation_runner.py:254-256 | `h.dt`, `h.tstop`, `h.secondorder` | Caller's values; `2` | Integration (Crank–Nicolson) | — | NUM | Every generator timestep | generic |
| simulation_runner.py:299-319 | Voltage initialisation | `FInitializeHandler(0, …)`; `h.v_init` never set | Applies the pool `initial_voltage__mV` | — | — | yaml:11 | generic |
| utils/neuron/inject_currents_into_populations.py:63-75,86,155,168-174 | Current-injection path | IClamp at soma(0.5), the same waveform for every cell; detection `-10.0 * pq.mV`; only `h.finitialize()` (no dt or celsius) | Noise-free drive; NEURON default dt 0.025 ms and 6.3 °C (inferred) | none found | Differs from SimulationRunner | SimulationRunner globals | generic |
| utils/currents.py:194,457-461 | Waveform defaults | Sawtooth `widths__ratio=0.5`; trapezoid rise / plateau / fall `100.0` / `200.0` / `100.0` ms | Input shape | — | — | — | generic |

**H. Proprioception and closed-loop mechanics** (the Hill force path is in §2.3)

| Location | Name | Default | What it changes | Source | Flag | Consistent with | Per-muscle |
|---|---|---|---|---|---|---|---|
| proprioception/spindle.py:163,188-234 | `create_default_spindle_parameters` | `species="human"`; dict summarised (fBag1/2 60, fChain 90, G1 0.0289, G2 0.0636, … tau1 0.149 s, tau2 0.205 s) | Ia/II rates | "(Mileusnic et al., 2006)", "Elias thesis pg 66" (neither in the register) | Unregistered sources | Afferent RT | human / cat presets |
| spindle.py:229-233,268-275 | Afferent gains | Human `gBag1 6500, gBag2A1 3250, gChainA1 3250, gBag2A2 3500, gChainA2 3500`; cat 20000/10000/10000/7250/7250 | Ia/II sensitivity | "original Mileusnic values" (cat) | Unregistered | — | species |
| proprioception/golgi.py:146-160 | `create_default_gto_parameters` | `"G1": 40`, `"G2": 4` | Ib rate = G1·log(F/G2 + 1) | "Lin & Crago (2002)", "Aniss et al. (1990b)", "Elias PhD thesis (pg 83)" (not in the register) | G2 "Calibrated" | Hill F0 | generic |
| golgi.py:163,186-205 | `create_gto_parameters_for_muscle` | `muscle_type="FDI"`: `G1 45, G2 2`; Sol `35, 8`; `"generic"` raises NameError | Ib sensitivity | none found | Bug for "generic" | Hill F0 | **FDI / Sol** |
| _cython/_gto.pyx:65-70 | GTO filter | Hard-coded `0.4/4.4/4` polynomial coefficients | Ib dynamics | none found | Buried | dt | generic |
| joint_dynamics.py:65-67,133-135 | JointDynamics | `stiffness__Nm_per_rad=0.0`, initial angle / velocity `0.0`; explicit Euler | Joint angle → muscle length → spindle | none found | — | dt | generic |

### 2.2 Muscle, fibre properties and recruitment thresholds

`SurfaceEMG` reads from `Muscle`:
- the mean CV, `mean_fiber_length__mm`, `var_fiber_length__mm`, bone, fat and skin, the four conductivities and the fibre assignment;
- the thresholds, when dispersion is on.

`IntramuscularEMG` reads `length__mm`, the fibre centres, the **per-fibre** diameters and CVs, and the thresholds. Neither EMG path reads what the other one reads (surface_emg.py:207-248; intramuscular_emg.py:244-273).

| Location | Name | Default | What it changes | Source | Flag | Consistent with | Per-muscle |
|---|---|---|---|---|---|---|---|
| core/muscle/muscle.py:36 | `CV_INTERCEPT__M_S` | `0.95` | Unused in `Muscle` (per-fibre CVs are z-scored and rescaled). Used by dispersion `diameter_scales_source` | "[nandedkar1983]"; "[delvecchio2018 p. 7 eq. 1]". Register: "a modelling construct, not a measured regression" | ASSUMPTION | `fiber_dispersion.py:114` | generic |
| muscle.py:37 | `CV_PER_DIAMETER__M_S_PER_MM` | `50.0` | `Muscle`: how tightly CV follows diameter, relative to the 0.2 m/s scatter (inferred). Dispersion: line slope | "[nandedkar1983 p. 163]" | ASSUMPTION | CV scatter, `fiber_dispersion.py:113` | generic |
| muscle.py:252 | `radius__mm` | `6.91 * pq.mm` | Area, fibre count, territories, sEMG muscle radius | "as determined by Jacobson et al. 1992 [3] [jacobson1992]". Register infantolino2010: "derived CSA ~89 mm² vs model 150 mm²" | Larger than the cited anatomy | Density, N, electrode bending radius | FDI |
| muscle.py:253 | `length__mm` | `30.0 * pq.mm` | **iEMG only**: every fibre spans 0…30 mm; end-plate centre at 50 % | "a nominal value …" (no source) | ASSUMPTION; a second fibre-length authority | `mean_fiber_length__mm` 31.7 (sEMG), Hill L0 38.9 mm | FDI (nominal) |
| muscle.py:254 | `fiber_density__fibers_per_mm2` | `350 * pq.mm**-2` | Fibre count (≈ 52 502), fibres per MU, MUAP amplitude | "as determined by Botelho et al. 2019 [7] [botelho2019]". Register feinstein1955: "~40,000 fibres, ~340 fibres/MU" | ≈ 437 fibres/MU at 120 MUs (inferred) | Radius, N | FDI |
| muscle.py:255 | `max_innervation_area_to_total_muscle_area__ratio` | `1 / 4` | Largest territory; territory widths | "a pragmatic upper bound for the FDI, with no single published source" | ASSUMPTION | `conf` | FDI |
| muscle.py:347-361 | Area / fibre count ∝ `rt / max(rt)` (rule, no argument) | — | MU size range equals the threshold ratio RR (inferred) | "[akhmadeev2020]" for the MU distribution; no source for linearity | ASSUMPTION, buried | RR; ForceModel RP | generic |
| muscle.py:256 | `mean_conduction_velocity__m_per_s` | `4.2 * pq.m / pq.s` | **sEMG only**: one CV for every fibre; the time and k_z grids | "[nishizono1990 p. 77]" | Physiological (FDI), but a single value | iEMG 4.32; Nandedkar line at 57.9 µm = 3.85 | FDI |
| muscle.py:257 | `mean_fiber_conduction_velocity__m_per_s` | `4.32 * pq.m / pq.s` | **iEMG only**: mean per-fibre CV | "[delvecchio2018 p. 2]" (tibialis anterior) | TA data in an FDI-default muscle; unit means ≈ 3.9 | sEMG 4.2 | TA → FDI |
| muscle.py:258 | `sd_fiber_conduction_velocity__m_per_s` | `0.71 * pq.m / pq.s` | iEMG CV spread | "[delvecchio2018 p. 2]" | ~2 % of fibres at the 6 m/s clip | Clip | TA → FDI |
| muscle.py:422 | CV scatter (hard-coded) | `normal(0, 0.2, n_fibers)` | CV–diameter correlation | "0.2 m/s of scatter (a modelling assumption)" | ASSUMPTION, not exposed | Line slope | generic |
| muscle.py:431-433 | CV clip | `2.0`–`6.0` m/s | Caps tails | "Sanity bounds 2-6 m/s (a modelling assumption, not sourced …)" | ASSUMPTION (below Del Vecchio's maximum of 6.21) | Mean, SD | generic |
| muscle.py:259 | `type_i_fiber_fraction` | `0.574` | Which units are type I (strict threshold order) | "FDI 57.4 % [polgar1973 p. 312 Table 4]"; "[garnett1979 p. 42]" (order is a simplification) | iEMG only | HillModel type counts; sEMG 57.9 µm | FDI |
| muscle.py:260 | `type_mean_fiber_diameters__um` | `(50.0, 67.0)` | iEMG source ∝ d², CV ordering | "[polgar1973 p. 310 Table 2]" | iEMG only | `SurfaceEMG.fiber_diameter__um` 57.9 (a separate literal) | FDI |
| muscle.py:261 | `type_fiber_diameter_log_sd` | `0.2` | Within-type spread | "(a modelling assumption)" | ASSUMPTION | `fiber_types.py:82` | generic |
| muscle.py:262 | `mean_fiber_length__mm` | `31.7 * pq.mm` | **sEMG only**: semi-lengths, IZ spread (0.1·L), kernel window (2.5·L) | "[jacobson1992 Table 1]". Register: "Table 1 unverified", ABS; Corrections #8 "Never read in full" | Locator unverified | `length__mm` 30 (iEMG) | FDI |
| muscle.py:263 | `var_fiber_length__mm` | `2.8 * pq.mm` | sEMG uniform ±2.8 mm length jitter | "[jacobson1992 Table 1] … most likely describes variation between specimens rather than between fibres" | **NON-PHYS** as a within-muscle spread (per its own docstring) | Dispersion `tendon_end_band__mm` replaces it | FDI |
| muscle.py:264 | `radius_bone__mm` | `0 * pq.mm` | sEMG bone layer: none | Docstring says "Default is set to 1 mm" (contradicts code); no source | Bone off | `sig_bone` (fixed at 0) | FDI (inferred) |
| muscle.py:265 | `fat_thickness__mm` | `1.3 * pq.mm` | sEMG fat layer | "… Nishikawa et al. 2024 [12] [nishikawa2024] … minus the skin" | Derived | Skin | FDI |
| muscle.py:266 | `skin_thickness__mm` | `1.2 * pq.mm` | sEMG skin layer | "… Li et al. 2018 [11] [li2018]" | Physiological | Fat | FDI |
| muscle.py:267-268 | `muscle_conductivity_radial__S_per_m`, `_longitudinal__S_per_m` | `0.09`, `0.4` S/m | sEMG muscle σ and anisotropy | "[botelho2019 Table 1]" | **Not used by iEMG** (0.063 / 0.33) | `bioelectric.py:101-102` | generic |
| muscle.py:269 | `fat_conductivity__S_per_m` | `4.07e-2 * pq.S / pq.m` | sEMG fat σ | "[botelho2019 Table 1]" | Physiological | — | generic |
| muscle.py:270 | `skin_conductivity__S_per_m` | `1.0 * pq.S / pq.m` | sEMG skin σ. A resistive skin raises amplitude 3.5–6× (docstring) | "[farina2004]" via "[roeleveld1997]": "not a measured skin conductivity"; measured 4.55e-4 to 4.88e-4 "[gabriel1996]" | **NON-PHYS** (effective, fitted) | Fat σ | generic |
| muscle.py:271 | `grid_resolution` | `256` | Innervation-centre placement grid | none found | NUM | — | no |
| muscle.py:492-499 | Innervation-centre rescale | `(center_offset / max_dist) * self._radius__mm` | Farthest centre placed on the border (inferred) | none found | ASSUMPTION, buried | Radius | no |
| muscle.py:550,559-571 | Fibre-position table / Halton fallback | Table up to ~313 k fibres, then scrambled Halton (with a warning) | Fibre packing | none found | NUM | Density, radius | no |
| muscle.py:605,684,839-843 | `assign_mfs2mns(n_neighbours=3, conf=0.999)` | `3`, `0.999` | Intermingling of MUs; territory compactness | Docstring ranges only, no source | ASSUMPTION | Territory ratio | no |
| muscle.py:272,363-367,391-395 | `autorun` | `False` | When False, nothing is built; fibre properties need the private `_generate_fiber_properties()` after `assign_mfs2mns` | — | Pipeline is opt-in | Call order | no |
| core/muscle/fiber_types.py:80-82 | `fiber_type_ladder` defaults | `0.574`, `(50.0, 67.0)`, `0.2` | Same as Muscle | "[polgar1973 p. 312 Table 4]", "[polgar1973 p. 310 Table 2]" | Duplicate literals | muscle.py:259-261 | FDI |
| fiber_types.py:70,133 | Truncations | `3.0 * np.hypot(sigma_between, sigma_within)`; `np.clip(norm.ppf(q), -3.0, 3.0)` | Diameter tails (≈ 26–117 µm) | "truncated at 3 SD of the pool (a modelling assumption)" | ASSUMPTION | CV clip | generic |
| fiber_types.py:124-126,150 | Boundary-unit rule; pool target mean | `argmin(|cum/cum[-1] - f|)`; `realised*m1 + (1-realised)*m2` | Type split; pool mean 57.2 µm | Design spec §5.1 (decision); `[polgar1973]` | ASSUMPTION | 57.9 µm RMS (sEMG) | FDI |
| core/physiological_distribution.py:21-22 | `N`, `recruitment_range__ratio` | Required | Number of units; threshold spread. Through the linear rule also MU size range, and through ForceModel the RP | Docstring examples use 50; register fuglevand1993 "50-fold" | No library default (optimiser: `--n-mu 200`, RR `30.0`) | Muscle, ForceModel, DD calibration | FDI 120 (spec only) |
| physiological_distribution.py:23,212-215 | `deluca__slope` | `None` | Curvature for `deluca` / `combined`; **required** by both | Docstring "Typical values range from 0.001-100" | No default, yet the optimiser's default `combined` needs it (raises, verified) | `mode` | no |
| physiological_distribution.py:24-25 | `konstantin__max_threshold__ratio`, `mode` | `1.0`, `"konstantin"` | Threshold distribution, hence recruitment, MU sizes and twitches | konstantin "Akhmadeev et al. (2020) [3] [akhmadeev2020]"; fuglevand "[fuglevand1993]"; deluca "[delucacontessa2012]"; combined: none | Library default `konstantin`; optimiser default `combined`; FDI calibration `fuglevand` | The same array must reach the pool, Muscle and ForceModel (not enforced) | no |
| physiological_distribution.py:165,173,187 | `/ 100` scaling; `rtz` renormalisation | `np.exp(...) / 100`; `rtz * max(rt)/max(rtz)` | In deluca mode the realised range ≠ RR (inferred); `rtz` rescaled in every mode | none found | Unclear | ForceModel uses the realised ratio as RP | no |

### 2.3 Force

| Location | Name | Default | What it changes | Source | Flag | Consistent with | Per-muscle |
|---|---|---|---|---|---|---|---|
| core/force/force_model.py:111 | `longest_duration_rise_time__ms` | `90.0 * pq.ms` | T_L, twitch time course | "(T_L in _[1])" (no locator for 90); register milnerbrown1973b "supports ForceModel defaults" | Plausible | Hill `Tl` 90 | generic |
| force_model.py:112 | `contraction_time_range_factor` | `3.0` | RT, contraction-time spread | "Generally between 2 and 5" (no locator) | Plausible | Hill `RT` 3 | generic |
| force_model.py:113,179-182 | `twitch_force_range` | `None` | None makes RP = `rt[-1]/rt[0]`, setting the peak-twitch spread (eq. 13) and the CT exponent (eq. 14) | "Fuglevand et al. (1993) [fuglevand1993] use RP = 100 independently of the recruitment range RR = 30"; register milnerbrown1973b "~100-fold vs model 50-fold (open)" | **NON-PHYS** fallback (backward compatibility) | RR; Hill `RP` 130 | generic |
| force_model.py:114 | `force_frequency_shape` | `FUGLEVAND_FORCE_FREQUENCY` | Force–frequency saturation S(r) = 1 − exp(−a r^c) | "[fuglevand1993 eq. 17]" | Human curve is **OPT-IN** | Optimiser `--force-frequency`, the calibration JSON, Hill `fsatf`/`lsatf` | generic |
| core/force/force_utils.py:4 | `FUGLEVAND_FORCE_FREQUENCY` | `(2.0, 3.0)` | Default (a, c) | "[fuglevand1993] (not read in full; no locator)" | rms error 0.13 against human units (force_utils.py:7) | — | generic |
| force_utils.py:8 | `HUMAN_UNIT_FORCE_FREQUENCY` | `(0.93, 1.29)` | Units keep gaining force above ~20 Hz; twitch:tetanus 0.23 | "[macefield1996 Fig. 6C; CT Table 1], [fuglevand1999 Table 1, Fig. 6A], [mcnulty2000 Table 2; CT Table 1]" | **OPT-IN** | `force_frequency_shape` | generic (hand / toe units) |
| force_utils.py:11,30-34 | `fuglevand_gain(a=2.0, c=3.0)`; knee and normalisation | `mask = r > 0.4`; `/((1 - exp(-a*0.4**c))/0.4)` | Gain is 1 below r = 0.4; sets the absolute tetanic force for any shape | "[fuglevand1993 eq. 17]" | Duplicate literal of (2, 3); the "9.05 P" saturation note (force_model.py:87) holds only for (2, 3) | Shape | generic |
| force_model.py:378 | First-discharge gain | `[1.0]` | First spike of each unit has gain 1 | none found | Convention | — | no |
| core/force/force_model_vectorized.py:4-32 | `ForceModelVectorized` | Inherits | Same | "[fuglevand1993]" | Docstring omits `twitch_force_range` and `force_frequency_shape` | ForceModel | generic |
| neuron/muscle.py:55-56; _cython/_hill.pyx:557-572 | `HillModel(n_motor_units_type1, n_motor_units_type2)` | Required | Units with index < Ntype1 drive the type-I force–length and force–velocity curves | none found | **Not linked to Muscle fibre typing**; typed by index | `Muscle.motor_unit_fiber_types` (~86 % type I) | user |
| neuron/muscle.py:237,256-317 | `create_default_muscle_parameters(muscle_type="FDI")` | FDI dict: `alfa0 0.1606, F0 33.75, L0 38.9e-3, m 4.67e-3`; tendon / passive `Kpe 5 … LT_r 0.964`; FL / FV coefficients (identical to Sol); `Ak`, `Bk`; `RP 130, fP 3, RT 3, durType 1, Tl 90, fsatf 50, lsatf 100, satType 1` | Closed-loop muscle force (a second force authority) | Only author-years in `ForceSatParams`: "Fuglevand, A. J., Winter, D. A., & Patla, A. E. (1993)", "Cisi, R. R., & Kohn, A. F. (2008)"; the values have no source | Unsourced; its own force–frequency law and RP | ForceModel; Muscle fibre length | **FDI / Sol** |
| core/force/biomechanics.py:78,176-182,241,393-395 | `JointBiomechanics(use_simplified_model=True)` | `radius * 0.8`, `abs(sin(angle + π/4))`, floor `0.1`, torques summed in one direction | Torque | none found | Unsourced; no in-package consumer | — | no |

### 2.4 Surface EMG

Default-case grid (derived from the code at FDI defaults, inferred):
- N_int = round(256 × 10000/2048) = 1250;
- K = 2.5 × 31.7 = 79.25 mm;
- grid rate ≈ 66 kHz, dz ≈ 0.063 mm;
- MUAP template ≈ 13.6 kHz over K/v ≈ 18.9 ms.

**A. SurfaceEMG (`core/emg/surface/surface_emg.py`)**

| Location | Name | Default | What it changes | Source | Flag | Consistent with | Per-muscle |
|---|---|---|---|---|---|---|---|
| surface_emg.py:144 | `sampling_frequency__Hz` | `2048.0 * pq.Hz` | Output rate, reached by linear interpolation with no anti-alias filter (:928-936). Also sets N_int = round(256·fs_int/fs), so raising fs coarsens the internal grid (inferred) | "as used by the Quattrocento …" (device) | NUM; the coupling is easy to miss | Spike dt, `internal_sampling_frequency__Hz` | generic |
| surface_emg.py:145 | `sampling_points_in_t_and_z_domains` | `256` | Template length (Fourier resample, low-pass); N_int | none found | NUM | fs ratio, K | generic |
| surface_emg.py:146 | `sampling_points_in_theta_domain` | `32` | Number of angular orders (−16…15) | none found. The docstring's ">64 … overflow" warning may predate the log-space solver (inferred) | NUM | Electrode angular spacing | generic |
| surface_emg.py:147 | `MUs_to_simulate` | `None` | Units left out produce zero templates | — | Subset is opt-in | Pool size | no |
| surface_emg.py:148,174-176 | `internal_sampling_frequency__Hz` | `None` → `10000.0 * pq.Hz` | Only sets N_int; the real grid rate is v(N_int−1)/K | none found | NUM; the name is misleading | fs, K, CV | generic |
| surface_emg.py:149,488-494,676-683 | `iap_kernel_length__mm` / `IAP_SCALE_FACTOR` | `None` → `mean_fiber_length * 2.5` (defined at :492 and :682) | Periodic window K, so MUAP window K/v. No wrap guard without dispersion | none found | NUM; **not resized for λ < 1** (the docstring tells the user to enlarge it) | λ, fibre length, CV, dispersion | generic |
| surface_emg.py:150 | `use_unified` | `False` | Routes through `simulate_fiber_hybrid` (§2.4 F) | none found | Experimental, opt-in | Refuses λ ≠ 1 and dispersion; ignores diameter and σi | no |
| surface_emg.py:151 | `fiber_diameter__um` | `57.9` | Source ∝ d²; one value for every fibre | "[polgar1973 p. 310 Table 2]", "[polgar1973 p. 312 Table 4]" (RMS) | Physiological (FDI RMS) | Muscle type means and fraction (separate literals); hybrid 55.0; iEMG fallback 55e-3 | FDI |
| surface_emg.py:152 | `intracellular_conductivity__S_m` | `1.01` | Source ∝ σi | none in the docstring; register lowery2002 "σi 1.01 S/m (Table I)", hodgkinnakajima1972 | Physiological | iEMG hard-codes 1.01 (bioelectric.py:318) | generic |
| surface_emg.py:153 | `fiber_dispersion` | `None` | Per-fibre end-plates, nerve delays, CVs, tendon ends, d² source (§2.4 C) | see §2.4 C | **OFF** (bit-identical when off) | §2.4 C | no |
| surface_emg.py:154 | `iap_lambda__per_mm` | `1.0` | Rosenfalck length scale; λ < 1 stretches the IAP, lowers MDF and raises RMS | "[rosenfalck1969]"; "Botelho et al. (2019) [botelho2019] used 0.5"; FDI fit 0.7 against "[botelho2019]". Register: "FDI fit λ 0.7 documented, not adopted" | Published default; the FDI fit is OPT-IN | K; dispersion 10 mm tail; iEMG λ (hard 1); refused by `use_unified` | generic (FDI fit 0.7) |
| surface_emg.py:321-324,340-344 | IZ spread (hard-coded) | `self._mean_fiber_length__mm * 0.1`; `uniform(-v/2, v/2)` per MU | Per-MU end-plate z, ±1.585 mm at FDI (a width, despite the name "variance") | "10% of the mean fiber length (see Botelho et al. 2019) [botelho2019]" | ASSUMPTION, buried, not an argument | `runtime/semg.py:47-49` duplicate; dispersion two-layer | FDI (scales with L) |
| surface_emg.py:353-360 | Fibre-length variation | `uniform(-var, var)` per fibre | Tendon positions, extinction timing | via Muscle (see §2.2) | **NON-PHYS** as a within-muscle spread | `tendon_end_band__mm` | FDI |
| surface_emg.py:533; simulate_fiber.py:104-118 | `fiber_semi_lengths` | `L1 = l/2 - zi`, `L2 = l/2 + zi` | Tendons fixed at ±l/2 about z = 0 | "[farina2004 eq. 1]" | ASSUMPTION | Electrode z | generic |
| surface_emg.py:476,526,578 | CV used | `float(self._mean_conduction_velocity__m_s)` | Every fibre at the Muscle mean (4.2) | "[nishizono1990 p. 77]" (via Muscle) | Single CV unless dispersion is on | Muscle 4.32 ± 0.71 (iEMG) | FDI |
| surface_emg.py:496-508 | Dispersion wrap guard | `reach = … + 10.0  # mm, plus the IAP tail` | Raises on wrap-around | none found | NUM; the 10 mm assumes λ = 1 (inferred) | λ, K | generic |
| surface_emg.py:769-774,798-828,835,861-866,892-936 | Spike binning, template → dt resample, convolution, output decimation | `bin_spike_trains(...)`; `np.interp`; `np.convolve(..., mode="same")`; `np.interp` to fs | Boolean bins (two spikes in a bin merge, inferred); linear resampling with no anti-alias; no recording band-pass anywhere (inferred) | n/a | NUM | Spike dt, fs | no |
| surface_emg.py:961,1018-1035 | `add_noise(snr__dB, noise_type="gaussian")` | SNR required; `"gaussian"` | White noise scaled to each channel's own power | "Typical physiological EMG has SNR ranging from 10-40 dB" (no source) | OPT-IN; no 1/f, mains or fixed floor | fs | generic |

**B. Muscle quantities read by sEMG:** radius, mean CV, fibre length and variation, bone, fat, skin and the four conductivities. These rows are in §2.2. Two more:

| Location | Name | Default | What it changes | Source | Flag | Consistent with | Per-muscle |
|---|---|---|---|---|---|---|---|
| simulate_fiber.py:290; volume_conductor.py:72 | `sig_bone` | `0` / `0.0` | Bone σ; never passed, so bone is insulating even when `radius_bone > 0` | none found | ASSUMPTION, not exposed | `radius_bone__mm` | no |
| surface_emg.py:245-248 | `radius_total` | `radius + fat + skin` | Skin radius in the conductor and in the electrode-size filter | — | — | Electrode `bending_radius__mm` (not linked) | FDI |

**C. SurfaceFiberDispersion (`surface/fiber_dispersion.py`).** Every field is **OFF** by default (`active`, :176-181).

| Location | Name | Default | What it changes | Source | Flag | Consistent with | Per-muscle |
|---|---|---|---|---|---|---|---|
| :103 | `nmj_model` | `"uniform_mu"` | One end-plate per MU; `two_layer` scatters about 0.0; `two_layer_intra` re-centres on the MU zone | "[akhmadeev2020 eq. 5-6]" | **OFF** | iEMG two-layer (always on) | generic |
| :104 | `nmj_scatter_scale` | `1.0` | Multiplies both spreads | n/a | Knob | — | no |
| :105 | `nmj_coefficients` | `"published"` | Branch / arbor SD coefficients | "[akhmadeev2020 eq. 5-6]"; `reference_code` is 1.5–2× wider | Physiological | iEMG `nmj_coefficients` | generic |
| :106 | `nerve_delays` | `False` | Axon-branch delays | "[akhmadeev2020]" | **OFF** (always on in iEMG) | iEMG | generic |
| :107 | `nerve_cvs__m_per_s` | `(5.0, 2.0)` | Delay magnitude | Code: "as in intramuscular EMG (Akhmadeev et al. 2020) [akhmadeev2020]". Register: "from the reference code (paper: 10/1)" | **NON-PHYS**: reference-code values, not the paper's | iEMG `branch_cvs__m_per_s` | generic |
| :108 | `fiber_cv_inter_mu_spread` | `0.0` | Linear CV ramp in rank | none found | **OFF** | Excludes slope and per-unit values | no |
| :109 | `fiber_cv_threshold_slope` | `0.0` | Unit CV linear in threshold; the mean is over units, not fibres (inferred) | "[delvecchio2018 p. 2]"; 0.63 "derived from those values" | **OFF** (`literature_candidate` 0.63) | Muscle 4.2 (TA slope on the FDI mean) | TA |
| :110 | `fiber_cv_within_mu_cov` | `0.0` | Within-unit CV spread (3-SD clip) | "Del Vecchio et al. (2018) call it unknown [delvecchio2018 p. 5]"; 0.05 "A modelling assumption" | **OFF** | — | no |
| :111 | `fiber_cv_per_unit__m_s` | `None` | Explicit per-unit CVs | n/a | **OFF** | Muscle per-fibre CVs (not wired) | no |
| :112 | `diameter_scales_source` | `False` | Per-fibre d² source from the CV line, normalised over the pool | "[nandedkar1983]", "[delvecchio2018 p. 7 eq. 1]", "[blijham2006 p. 1840]" | **OFF** (also off in `literature_candidate`) | `fiber_diameter__um` stays the RMS | generic |
| :113-114 | `cv_per_diameter__m_s_per_mm`, `cv_intercept__m_s` | `CV_PER_DIAMETER__M_S_PER_MM`, `CV_INTERCEPT__M_S` | Line | as above | — | muscle.py:36-37 | generic |
| :115 | `tendon_end_band__mm` | `0.0` | Independent end positions within a band; replaces the length jitter | `literature_candidate` 5.0 "[keenan2005]"; register: "no measurement of fibre-end stagger found" | **OFF** | `var_fiber_length__mm` | FDI |
| :116-117 | `tendon_end_correlation__mm`, `tendon_end_jitter__mm` | `0.0`, `0.0` | Smooth end-position field; per-fibre offset | `literature_candidate` 5.0 / 0.3: "assumptions; no measurement" | **OFF** | Band | no |
| :118 | `seed` | `0` | Seeds every dispersion draw, independent of `set_random_seed` | n/a | Buried: every run gets the same realisation | Global seed | no |
| :213; :197 | `end_margin__mm`; `n_features` | `1.0`; `1000` | End-plate clip; random-feature field | none found | NUM, not exposed | — | no |
| :153-174 | `literature_candidate()` | `two_layer_intra`, published, delays on, (5, 2), slope 0.63, cov 0.05, band 5, corr 5, jitter 0.3 | Preset; nothing uses it automatically | per field | OPT-IN preset | — | mixed |

**D. Two-layer nerve model (`core/emg/nerve_branches.py`, shared with iEMG)**

| Location | Name | Default | What it changes | Source | Flag | Consistent with | Per-muscle |
|---|---|---|---|---|---|---|---|
| :23-26 | `NMJ_SCHEDULE_COEFFICIENTS` | `"published": (1.0, 2.5, 0.25, 1.0)`; `"reference_code": (1.5, 4.0, 0.5, 1.5)` | Branch and arbor z SDs | "[akhmadeev2020 eq. 5-6]"; reference "s08_cl_init_muaps.m" | Physiological | Both EMG paths | generic |
| :42-44 | Branch count; spread factor | `1 + round(log(thr/thr[0]))`; cumulative threshold share | Branches per MU; SD grows with rank | "[akhmadeev2020]" | ASSUMPTION | Thresholds | generic |
| :67,75-88 | k-means; path origin; delay | `max_iter=100`; centre = mean fibre xy and mean z; Σ path/CV | Branch grouping and delays. `nominal_center__mm` is passed but unused | none found / "[akhmadeev2020]" | ASSUMPTION | Branch CVs | generic |

**E. Volume conductor and kernel**

| Location | Name | Default | What it changes | Source | Flag | Consistent with | Per-muscle |
|---|---|---|---|---|---|---|---|
| simulate_fiber.py:464 | Rosenfalck `A` | `96  # mV/mm^3` | IAP amplitude | "[rosenfalck1969]"; register farinamerletti2001 "96 mV/mm³ with λ = 1 / mm" | Not exposed | iEMG `D1=96.0` | generic |
| simulate_fiber.py:481-486 | λ in the source | `lam = iap_lambda__per_mm` | IAP shape. This is the **only** place where λ is a parameter | "[rosenfalck1969]" | — | K | generic |
| simulate_fiber.py:301-304,704 | Function defaults | `sig_intracellular 1.01`, `fiber_diameter__mm 0.0579`, `iap_lambda__per_mm 1.0` | Duplicates the SurfaceEMG defaults | as above | Duplicates | surface_emg.py:151-154 | FDI |
| simulate_fiber.py:439-449,476 | k grid; DC offset; IAP z grid | `fftfreq`; `0.5 * dk_t`; linspace over ±L/2 | IAP truncated at K/2 (negligible at λ = 1, inferred) | none | NUM | K, λ | generic |
| simulate_fiber.py:560-563 | Electrode size filter | `H_size = 2 * Jn(1, arg)/arg` | Circular discs only | none | ASSUMPTION | `electrode_radius__mm`, r_total | generic |
| volume_conductor.py:130,199,241-243 | Anisotropy; insulating outer boundary | `sqrt(σz/σρ)`; row 6 | Conductor | "[farina2004]" | ASSUMPTION | Muscle σ | generic |
| volume_conductor.py:33,245-246,264-273 | Overflow handling | `_MAX_LOG_SAFE = 700.0`; inf/nan → 0 | Silently zeroes overflowing terms | none | NUM (silent) | N, radii | generic |

**F. Electrodes (`core/emg/electrodes.py`, `SurfaceElectrodeArray`) and routes**

| Location | Name | Default | What it changes | Source | Flag | Consistent with | Per-muscle |
|---|---|---|---|---|---|---|---|
| electrodes.py:62-65 | `num_rows`, `num_cols`, `inter_electrode_distance__mm`, `electrode_radius__mm` | Required | Grid, spacing, disc filter | none found | — | Bending radius | no |
| electrodes.py:66 | `center_point__mm_deg` | `(0.0 * pq.mm, 0.0 * pq.deg)` | Grid centre at z = 0, the mean IZ (inferred) | none found | **NON-PHYS** placement (over the IZ, inferred) | IZ convention | no |
| electrodes.py:67,100-102 | `bending_radius__mm` | `0.0 * pq.mm` → `np.finfo(np.float32).eps` (verified) | Angular step IED / bending radius. Meaningless for multi-column grids and transversal or Laplacian modes (inferred). The docstring says "radius of the muscle", but the arc needs r_total (+36 % IED at FDI, inferred) | none found | **NON-PHYS** | `radius_total` (never set or validated) | no |
| electrodes.py:68 | `rotation_angle__deg` | `0.0 * pq.deg` | Grid rotation | "[farina2004 eq. 36]", "[farina2004 eq. 38]" | Aligned | — | no |
| electrodes.py:69-71,318-362 | `differentiation_mode` | `"monopolar"` | Spatial filter; `laplacian` is `-k_total_sq`, a small-k approximation (inferred) | bipolar "[farina2004 eq. 30]"; Laplacian none found | ASSUMPTION (Laplacian) | IED | no |
| electrodes.py:105-117 | `num_channels` | — | Does not match the returned rows × cols grid | — | Inconsistent | — | no |
| fiber_simulation.py:253-510 (via `use_unified=True`) | Hybrid route | `fiber_diameter_um: float = 55.0`, `sigma_i: float = 1.01`; "empirical factor of 4.2" (:400-406); linspace k_t; `psi[i] = -source[-i]`; real `B_kz`; per-fibre `np.roll` to the window centre | **Silently ignores** `fiber_diameter__um` and σi. Different scaling and reflection; loses bipolar_longitudinal (inferred); destroys fibre timing within a MU | "Validated: 150 MUs → RMS ratio 0.99× vs real HD-sEMG" (comment, no source) | **NON-PHYS** route | Refuses λ ≠ 1 (:202-203) and dispersion (:367-368) | no |
| runtime/semg.py:27-38,47-57,107-127 | `execution=ExecutionPolicy(...)` route | — | Honours λ, diameter, σi, K, N, M, geometry. **Raises** on dispersion, `use_unified`, `use_gpu`, electrode subclasses and mixed shapes. Re-hard-codes `.1` and `2.5` | — | Opt-in route | surface_emg.py:322, 492 | no |

### 2.5 Intramuscular EMG and noise

The branch has no Albers near-field correction, fascicle attenuation, `nmj_scatter_mode`, mid-field LOD, `electrode_profiles` / `ELECTRODE_LIBRARY` or `cross_muscle_background`. A grep of `myogen/` finds none of them. There is no electrode or device band-limiting of the clean iEMG.

| Location | Name | Default | What it changes | Source | Flag | Consistent with | Per-muscle |
|---|---|---|---|---|---|---|---|
| core/emg/intramuscular/intramuscular_emg.py:107 | `sampling_frequency__Hz` | `10240.0 * pq.Hz` | SFAP time grid and output rate | none found (Quattrocento named only) | NUM / device | Spike dt; sEMG 2048 | generic |
| intramuscular_emg.py:568-595,691-719 | Spike-train dt (implicit) | `first_spiketrain.sampling_period` | Templates are linearly resampled to the spike dt (no anti-alias), convolved, then interpolated to fs. A coarse dt low-passes and aliases | none found | NUM, hidden (inferred) | NEURON dt ≤ 1/10240 s | generic |
| intramuscular_emg.py:108 | `spatial_resolution__mm` | `0.01 * pq.mm` | Line-source grid; also an amplitude factor (`* dz`, motor_unit_sim.py:325,383) | none found | NUM. The `execution=` route lacks `* dz` (runtime/iemg.py:59) | Both routes | generic |
| intramuscular_emg.py:109 | `endplate_center__percent` | `50` | End-plate band centre, the same for every MU | none found | ASSUMPTION | sEMG IZ set separately | generic |
| intramuscular_emg.py:110,126,137,373 | `nmj_jitter__s` | `35e-6 * pq.s` | **Nothing**: templates use `calc_muap(jitter_std=0.0)` (verified) | "Akhmadeev et al. 2020 [1] [akhmadeev2020]" | **DEAD**; every discharge reuses the same template | — | generic |
| intramuscular_emg.py:111-114; motor_unit_sim.py:71-74,385-416 | `branch_cvs__m_per_s` | `(5.0 * pq.m / pq.s, 2.0 * pq.m / pq.s)` | Axon-branch delays, always applied | "(5.0, 2.0) m/s as determined by Akhmadeev et al. 2020 [1] [akhmadeev2020]". Register: "from the reference code (paper: 10/1)" | **NON-PHYS** (reference-code values) | Duplicated at motor_unit_sim.py:71-74 and fiber_dispersion.py:107 | generic |
| intramuscular_emg.py:115 | `MUs_to_simulate` | `None` | Subset; other units are zero | — | Opt-in | — | generic |
| intramuscular_emg.py:116; nerve_branches.py:23-26 | `nmj_coefficients` | `"published"` | End-plate scatter, so MUAP width and high-frequency content | "[akhmadeev2020 eq. 5-6]" | Physiological | sEMG default `uniform_mu` does not use it | generic |
| intramuscular/bioelectric.py:101-102 | `sigma_r`, `sigma_z` | `0.063`, `0.33` (S/m) | Amplitude ∝ 1/σr; radial decay | "(from Andreassen & Rosenfalck 1980) [andreassenrosenfalck1980]" (register: LINK, "not checked") | Hard-coded, not exposed | **Muscle 0.09 / 0.4 [botelho2019 Table 1] (sEMG)**; fiber_simulation.py:58-59 copy | generic |
| bioelectric.py:283,290 | IAP λ | Implicit 1 / mm | MUAP duration and spectrum | "Rosenfalck length scale 1 mm, as in the sEMG model" | Hard-coded, not exposed | SurfaceEMG `iap_lambda__per_mm` | generic |
| bioelectric.py:46,70 | Rosenfalck `D1` (and `D2 -90.0`, unused) | `96.0` | IAP amplitude | "Rosenfalck 1969 [rosenfalck1969]" | Hard-coded | sEMG `A = 96`; runtime/density.py:30 | generic |
| bioelectric.py:318,327 | `sigma_i_S_per_m` | `1.01` | Source strength ∝ σi | none in the iEMG code (register cites it for sEMG) | Hard-coded | SurfaceEMG argument; runtime/density.py:74 | generic |
| bioelectric.py:227 | `suppress_endplate_density`, `endplate_width` | `True`, `0.5` | Zeroes the source within ±0.5 mm of each end-plate | none found | ASSUMPTION, hard-coded | runtime/density.py:51 | generic |
| bioelectric.py:227 | `d` fallback | `55e-3` | Used only without diameters; the production path passes Muscle diameters | none found | Inactive | — | generic |
| motor_unit_sim.py:205-208,312-313 | `min_radial_dist` | `None` → mean MU fibre diameter | Clamp on fibre–electrode radius, so the peak amplitude of the nearest fibres | none found | NUM, unsourced; stands in for the missing electrode size | runtime/iemg.py:31 | generic |
| motor_unit_sim.py:90-95 | Fibre ends | `0` … `muscle_length__mm` | Every fibre spans [0, 30 mm], so the terminal waves and MUAP duration are the same for all | none found | **NON-PHYS**: Muscle fibre lengths and tendon stagger are ignored | `mean_fiber_length__mm` 31.7 (sEMG) | FDI (via Muscle) |
| motor_unit_sim.py:302-311 | Radial geometry | xy distance to the fibre centre | Fibres parallel to z (no pennation) | none found | ASSUMPTION | — | generic |
| intramuscular_emg.py:552-563,614-618 | MUAP centring | Template centre on the spike; `mode="same"` | Onset leads the discharge by about half a template (inferred) | none found | ASSUMPTION (inferred) | STA conventions | generic |
| intramuscular_emg.py:621-628; runtime/assembly.py:104-109 | Global template normalisation | `muap_shapes /= max_muap_amplitude` (verified) | **Discards all absolute amplitude**: peak = 1, output dimensionless. The comment calls the MUAPs "arbitrary units" | none found | **NON-PHYS, no switch**; contradicts the docstrings that say mV | Every absolute noise setting; sEMG in mV | generic |
| core/emg/electrodes.py:417-418 | `num_electrodes`, `inter_electrode_distance__mm` | Required; `0.5 * pq.mm` | Point positions | none found | Device choice | — | generic |
| electrodes.py:419-428 | `position__mm`, `orientation__rad` | `(0, 0, 0)` mm; `(0, 0, 0)` rad | Default at z = 0, the **fibre end**; needle parallel to the fibres | none found | **NON-PHYS** placement (inferred) | `length__mm`, end-plate centre | generic |
| electrodes.py:429,476-488 | `differentiation_mode` | `"consecutive"` | **Nothing in IntramuscularEMG**: the output is monopolar | none found | **DEAD** for iEMG | — | generic |
| electrodes.py:430-431,613-708 | `trajectory_distance__mm`, `trajectory_steps` | `0.0 * pq.mm`, `1` | More point channels; the mixing matrix is unused | none found | Opt-in, partly **DEAD** | — | generic |
| (absent) | Electrode area / impedance | — | Points only | none found | **NON-PHYS** | — | generic |
| intramuscular_emg.py:742-745,889-894 | `add_noise(snr__dB, noise_type="gaussian")` | SNR required; `"gaussian"` | White noise per channel, referenced to the whole-record channel power | none found | Coloured noise is OPT-IN; the reference differs from `emg_noise.py:28-31` ("active-period RMS") | `add_realistic_noise` uses the global RMS | generic |
| intramuscular_emg.py:747-755; utils/emg_noise.py:388-396 | `spectral_slope`, `excess_kurtosis`, `powerline_hz`, `powerline_amplitude`, `powerline_harmonic_ratios`, `powerline_frequency_drift_hz`, `powerline_amplitude_modulation_depth`, `peak_hz` | `-0.5`, `3.0`, `50.0`, `0.1`, `None` → `[1.0, 0.5, 0.3, 0.15, 0.08]`, `0.3`, `0.15`, `1000.0` | Realistic noise shape. Mains is **on** once `noise_type="realistic"` | none found ("phenomenological default") | Opt-in; unsourced; the `peak_hz` docstring says 750.0 | Lab mains | generic |
| emg_noise.py:240-248,265-267 | Heavy-tail and band-emphasis constants | `min(0.005, k/2000)`, `max(4.0, 12.0 - k)`, `2.5 + k*0.3`; `bandwidth_hz 1400.0`, `emphasis_blend 0.40` | Noise shape | none found | Hard-coded, unsourced | — | generic |
| intramuscular_emg.py:755; emg_noise.py:396 | `analog_hpf_hz` | `10.0` | HPF on the **noise only** | none found (comment lists devices) | Signal and noise bandwidths differ | — | generic |
| intramuscular_emg.py:756-759; emg_noise.py:397-400 | `baseline_drift_rms_uv`, `baseline_drift_alpha`, `baseline_drift_low_hz`, `baseline_drift_high_hz` | `0.0`, `1.75`, `None`, `1.0` | 1/f^α drift with an absolute RMS | Form "Huigen 2002 [huigen2002], Gondran 1996 [gondran1996]"; α "midpoint of the [1.5, 2.0] …" | **OFF**. The µV value is applied to a block normalised to peak 1, so the unit is meaningless | Block units | generic |
| emg_noise.py:604 | `add_realistic_noise(target_snr_db=20.0)` | `20.0` | Standalone helper using the global RMS | none found | Not used by IntramuscularEMG | — | generic |
| runtime/iemg.py; runtime/density.py:26-45; runtime/policy.py:18 | `execution=` route | `None` (opt-in) | Uses `x = -2 * (z - zi - v * t)`, `value *= -4`, `wave - short`, no `* dz` (verified). Refuses GPU. Cache namespace `"intramuscular-muap-density-v2"` is unchanged since the fixes | — | **NON-PHYS** route: pre-0.13.0 physics | Must match bioelectric.py:281-309 (it does not) | generic |
| intramuscular_emg.py:445-494 | `_analyze_detectable_motor_units` constants | `2.0`, `0.1`, `0.05` | Nothing; never called | none found | **DEAD** | — | generic |

### 2.6 Descending-drive optimiser and FDI calibration (`examples/finetune/01_optimize_dd_multilevel.py`)

These are the settings a calibration bakes into the spike trains. Every row is in `examples/finetune/01_optimize_dd_multilevel.py` unless another file is named. The CLI is summarised, not every flag.

| Location | Name | Default | What it changes | Source | Flag | Consistent with | Per-muscle |
|---|---|---|---|---|---|---|---|
| :55; :357,426; utils/helper.py:6-27 | `GAMMA_SHAPE` → `get_gamma_shape_for_mvc(100, mvc_shape_value=3.0)` | `3.0` | DD gamma shape (CV ≈ 0.58), `process_type="gamma"` (:366,435) | none found | Library default is Poisson | `DescendingDrive__Pool` | generic |
| :56 | `SYNAPTIC_WEIGHT` | `0.05` (µS, smallest MN) | DD input strength | none found | Library default 0.6 µS | `Network` default | generic |
| :57,188-235 | `NETWORK` | `{"fan_in": "equal", "weight_scaling": "dendritic_transfer"}` | Equal fan-in; weight `w·(k_max/k_i)^p`. Files without the block are refused | "the equal-share case of Heckman & Binder 1991 … [heckmanbinder1991]" (modelling assumption) | ASSUMPTION; **not in the library** (only in this script) | `Network.connect` | generic |
| :58-59,207 | `weight_exponent` p; `WEIGHT_EXPONENT_RANGE`; `FIT_WEIGHT_EXPONENT` | `1.0` if absent; `(0.0, 1.0)`; `False` | How much descending input favours small MNs (onion skin vs ceiling) | none ("modelling assumption"; PR doc §5) | ASSUMPTION; FDI best 0.666 | — | FDI |
| :65,155-165 | `GFLUCTDV_RANGE`, `_insert_gfluctdv(scale)` | `(0.05, 0.5)`; `std_e/std_i/g_e0/g_i0 *= scale` after `insert_Gfluctdv()`; scale ≤ 0 → no noise | Noise level searched with the drive | register destexhe2001 (mechanism) | The noise switch lives **in the script**; `run_dd_constant` falls back to `0.0` (noise off) when the key is missing (:358) | `cells.py:840-871` | FDI 0.358 |
| :66-67 | `ISI_CV_TARGET`, `ISI_CV_WEIGHT` | `(0.15, 0.22)`, `2.0` | Penalty on ISI CV | "[enokaduchateau2017] [dideriksen2012] (as synthesised in nexus LIT-17; range not checked against the papers)" | Unchecked | Gfluctdv | generic |
| :68,382,451; :1007 | `TIMESTEP_MS`, `h.dt`, `h.secondorder` | `0.1`; `2` | Integration step (also the DD generator step) | none found | NUM; iEMG needs a finer spike dt (10240 Hz ≈ 0.098 ms) | iEMG/sEMG resampling | generic |
| :69-74 | Ramp protocol | `RAMP_MVC_PCT 30`, rest / ramp / plateau `3.0` / `4.0` / `6.0` s, `CHECKPOINTS [5, 10, 20, 25, 30]`, weights `{5: 1.0, 10: 3.0, 20: 2.0, 25: 2.0, 30: 1.0}` | Calibration targets (checkpoints are excitation levels, not % MVC; PR doc §5) | none found | ASSUMPTION | — | generic |
| :76,97,1034 | `THRESHOLD_MODE` / `--threshold-mode` | `"combined"` | Pool thresholds | see §2.2 | **The default raises** (no `deluca__slope`, verified). FDI calibrations used `fuglevand` | RecruitmentThresholds | — |
| :77,1036 | `RECRUITMENT_RANGE` | `30.0` | Threshold range RR | none at this line | FDI calibration 50 | ForceModel RP, Muscle sizes | FDI 50 |
| :78,1037 | `MAX_RECRUITMENT` | `1.0` | Excitation at the last recruitment (rate targets only) | `[deluca1982]` (register, ceiling) | FDI `_human59_wexp` 0.41 | Force-frequency curve (re-derive when it moves) | FDI |
| :60-62,79-80,1039-1042 | `LAST_RECRUIT_PCT_MVC`, `CEILING_RANK`, `CEILING_WEIGHT`, `CEILING_RAMP_PER_S`, `MVC_REST_S`, `MVC_HOLD_S` | `None`, `0`, `3.0`, `0.1`, `0.5`, `3.0` | Recruitment-ceiling term on an MVC ramp | register deluca1982, skarabot2023 | **OFF** by default; FDI 59 / rank 1 | ForceModel shape | FDI 59 |
| :63-64,81-82,322-325,1044 | `FORCE_FS_HZ`, `FORCE_SMOOTH_S`, `FORCE_FREQUENCY_SHAPE` / `--force-frequency` | `2048.0`, `0.25`, `FUGLEVAND_FORCE_FREQUENCY` / `"fuglevand"` | % MVC mapping inside the optimiser | §2.3 | Human curve OPT-IN; FDI used human | ForceModel; `fdi_spikes_force.py:50` reads it back | FDI human |
| :83-86,1050 | `HIGH_FORCE_RATE_TARGETS`, `HIGH_FORCE_RATE_WEIGHT`, `SCORE_HIGH_FORCE_RATES` | `{40: 25.3, 80: 41.4}`, `1.0`, `False` | High-force rate penalty | "[deluca1982 p. 118 Table 1]" | **OFF** | — | FDI |
| :87,1052; utils/fr_optimization_targets.py:102-108 | `DE_LUCA_PARAMS` / `de_luca_targets` defaults | `{}` → `fr_onset_hz 8.0, fr_peak_early_hz 16.0, fr_peak_late_hz 12.0, steepness 3.0` | Rate-coding targets | "[delucacontessa2012]"; 16 Hz "tuned to approach Skarabot 2023 [skarabot2023] untrained VL" | Defaults are **VL**; FDI passes 8.9 / 69.0 / 44.0 / 1.55 | — | VL default |
| :101-103; fr_optimization_targets.py:48-52 | `MIN_FR_HZ`, `MAX_FR_HZ` | `8.0`, `17.0` | Fuglevand targets | "[fuglevand1993]" | — | — | generic |
| :1010,1024,1029 | `--n-mu`, `--target-model`, `--target-muscle` | `200`, `"de_luca"`, `"vl"` | Pool size, targets, results directory. `_thresholds` ignores `profile` (:91-96) | — | Defaults are **VL / 200**, not FDI / 120 | — | VL |
| :128-129,318 | Drive noise | `np.abs(rng.normal(0, peak_hz * 0.01))`, seed `42` | 1 % half-normal drive noise with a fixed seed | none found | ASSUMPTION | — | generic |
| :640,902 | `set_random_seed(42)` | `42` | Network and wiring realisation | — | Calibrations hold for this seed (seed spread 69–76 % MVC; PR doc §5) | Downstream seed | generic |
| :105; `.gitignore:172` | `RESULTS_BASE` / calibration files | `examples/basic/results/dd_{muscle}_{n}mu_{tag}/dd_optimized_params.json` | The parameter set that defines a calibrated pool | — | **Gitignored**: calibrations exist only locally | Everything in §3 group 9 | FDI |
| examples/finetune/fdi_spikes_force.py:36-50 | Force-script defaults | `--n-mu 120`; threshold mode / range / max recruitment taken from the JSON (fallback `"combined"`); `FORCE_FREQUENCY_SHAPE` read from the JSON | Reproduces the calibration's force mapping. The MVC-ramp wiring is redrawn, so the ceiling differs (66.9 vs 62.9 % MVC) | — | — | Optimiser | FDI |

## 3. Interaction groups (settings that must move together)

1. **Fibre conduction velocity.**
   - Members:
     - `Muscle.mean_conduction_velocity__m_per_s` 4.2 (sEMG, `[nishizono1990 p. 77]`, FDI);
     - `Muscle.mean/sd_fiber_conduction_velocity__m_per_s` 4.32 ± 0.71 (iEMG, `[delvecchio2018 p. 2]`, TA);
     - the Nandedkar line `CV_INTERCEPT__M_S` / `CV_PER_DIAMETER__M_S_PER_MM`;
     - the dispersion `fiber_cv_threshold_slope` (0.63 in `literature_candidate`), `fiber_cv_within_mu_cov` and `fiber_cv_per_unit__m_s`;
     - the 2–6 m/s clip.
   - Why: the same fibres get different CVs in the two modalities.
   - State: **inconsistent.** The register (nandedkar1983) says: "At FDI values it is inconsistent: 57.9 µm (polgar1973) → 3.85 m/s, 4.2 m/s (nishizono1990) → 65 µm". sEMG does not read the per-fibre CVs (backlog: "Surface EMG reads `Muscle` fibres").
2. **Fibre diameter and source strength.**
   - Members: Muscle `type_mean_fiber_diameters__um` (50, 67) and `type_i_fiber_fraction` 0.574 (iEMG per-fibre d²); `SurfaceEMG.fiber_diameter__um` 57.9 (RMS of the same mixture); `fiber_types.py` defaults; dispersion `diameter_scales_source` normalisation; hybrid 55.0; iEMG fallback `d=55e-3`; σi 1.01 (sEMG argument, iEMG literal, hybrid default).
   - State: consistent by construction at the defaults but held by **separate literals**. Changing the Muscle type means does not move 57.9. The hybrid route differs.
3. **IAP length scale λ and the kernel window.**
   - Members: `SurfaceEMG.iap_lambda__per_mm`; K = 2.5·L (`IAP_SCALE_FACTOR`, defined three times); the dispersion 10 mm tail; the IAP grid truncated at K/2; iEMG λ (hard-coded 1); the `execution=` iEMG kernel (pre-fix, effective factor 2).
   - State: consistent only at λ = 1. K is not resized for λ < 1, and iEMG cannot follow a λ change.
4. **Volume-conductor conductivities.**
   - Members: Muscle σ (0.09 / 0.4, fat 4.07e-2, skin 1.0; sEMG); `bioelectric.py` σr / σz (0.063 / 0.33; iEMG); `fiber_simulation.py:58-59` (unused copy); `sig_bone` 0.
   - State: **inconsistent.** Two values for the same tissue, from two sources (botelho2019 vs andreassenrosenfalck1980, unverified).
5. **End-plates and nerve branches.**
   - Members: iEMG `nmj_coefficients`, `endplate_center__percent`, `branch_cvs__m_per_s` (always applied); sEMG IZ uniform ±0.05·L (hard-coded twice); dispersion `nmj_model`, `nmj_coefficients`, `nerve_delays`, `nerve_cvs__m_per_s`, `nmj_scatter_scale`.
   - State: same authority (`nerve_branches.py`) and the same coefficients. The **defaults differ**: iEMG uses two layers with delays, sEMG uses `uniform_mu` with no delays. Both use the reference-code branch CVs 5/2, not the paper's 10/1.
6. **Fibre length and tendon ends.**
   - Members: `Muscle.length__mm` 30 (iEMG span); `mean_fiber_length__mm` 31.7 and `var_fiber_length__mm` 2.8 (sEMG); dispersion `tendon_end_*`; HillModel `L0` 38.9 mm; register infantolino2010 fascicle 29.9 mm.
   - State: **inconsistent** (30 / 31.7 / 38.9); iEMG has no length variation.
7. **Recruitment range, twitch range and MU size.**
   - Members: `RecruitmentThresholds.recruitment_range__ratio` (optimiser 30, FDI calibration 50); Muscle fibre-count ratio (= RR, linear rule); ForceModel RP (= RR when `twitch_force_range=None`); HillModel `RP` 130; Fuglevand RP 100 (force_model docstring); register milnerbrown1973b ~100-fold.
   - State: **five different values in play.** The same threshold array has to reach the pool, Muscle and ForceModel; nothing enforces it.
8. **Force–frequency law.**
   - Members: `ForceModel.force_frequency_shape`; optimiser `--force-frequency` (saved to the calibration JSON, re-read by `fdi_spikes_force.py`); `MAX_RECRUITMENT` (an excitation target that has to be re-derived when the curve moves; PR doc §5); HillModel `fsatf/lsatf/satType` (its own sigmoid law, unsourced).
   - State: ForceModel and the optimiser are consistent through the JSON. Hill is a separate authority.
9. **DD calibration bundle** (FDI `_human59_wexp`).
   - Members, as calibrated:
     - wiring and drive: `dd_neurons` 147, `conn_probability` 0.360, `dd_drive__Hz` 759.6, `synaptic_weight` 0.05, `gamma_shape` 3.0, the `network` block with p 0.666;
     - noise: `gfluctdv_scale` 0.358;
     - thresholds and targets: threshold mode `fuglevand`, RR 50, n_mu 120, max_recruitment 0.41;
     - force: force-frequency (0.93, 1.29), ceiling 59 % rank 1;
     - rates: De Luca params 8.9 / 69.0 / 44.0 / 1.55;
     - fixed in the script: seed 42, dt 0.1 ms, NERLab YAML, `gamma` 0.2.
   - Why: the file is valid only with all of these. A change to the cell model, YAML, Gfluctdv values, threshold mode or range, n_mu, force curve or wiring invalidates it (the optimiser refuses files without the `network` block).
   - State: the bundle exists only in a gitignored local file; the library has no way to build this pool.
10. **Voltage frame and spike detection.**
    - Members: `model`; YAML `ena`/`ek`/`el`; `initial_voltage__mV` -67; `spike_threshold__mV` 50; `DEFAULT_SPIKE_THRESHOLD` -10 (connections, `connect_to_muscle`, injection helper); Exp2Syn excitatory `e` (auto 70) and inhibitory -75; Gfluctdv `E_e` 0 / `E_i` -75; `INHIBITORY_REVERSAL_THRESHOLD` -40.
    - State: **inconsistent for NERLab** (rest ≈ 0). Inhibition and the Gfluctdv reversals are in an absolute frame (backlog: "inhibitory synapses at −75 mV absolute are far too strong for NERLab").
11. **Time step and sampling chain.**
    - Members: `h.dt` (optimiser 0.1 ms; SimulationRunner caller; injection path NEURON default); DD / afferent / spindle / GTO / Hill timesteps; Gfluctdv `exp_e` (taken at INITIAL); spike-train `sampling_period`; iEMG 10240 Hz; sEMG 2048 Hz output, ≈ 13.6 kHz templates, N_int coupled to fs.
    - State: nothing validates that the spike dt is fine enough for the templates. A 0.1 ms spike dt (10 kHz) is just below the iEMG rate of 10240 Hz.
12. **PIC and neuromodulation.** `gamma` 0.2 ↔ NERLab `gcaLbar`, `vtraub_caL`, `ltau_caL`, `gnapbar` ↔ Powers `ca_conductance_ranges`, `lambda_factor` ↔ inhibition ↔ DD calibration. State: the PIC is inert (ΔF 0.2 vs 4.8 pps, register beauchamp2023).
13. **Fibre types to force.** Muscle typing (by threshold, ~86 % of units type I) vs HillModel `n_motor_units_type1/2` (free, by index) vs ForceModel (no types). State: **not linked** (backlog: "`HillModel` type counts from `Muscle`").
14. **Electrode geometry and muscle radius.** `SurfaceElectrodeArray.bending_radius__mm` ↔ `radius_total` (radius + fat + skin) ↔ IED ↔ `electrode_radius__mm`. For iEMG: `position__mm` ↔ `length__mm` ↔ `endplate_center__percent`. State: not linked; the defaults are degenerate (eps radius; electrode at the fibre end).
15. **Noise and signal units.** iEMG `snr__dB` (relative, per channel) ↔ `baseline_drift_rms_uv` (absolute) ↔ the normalised dimensionless iEMG block ↔ `add_realistic_noise` (global RMS). sEMG `add_noise` is relative and white only. State: **inconsistent** for absolute iEMG noise.

Process-wide side effects also cross subsystems: the Powers `h.*` GLOBALs retune interneurons, and `multex`/`multin` switch Gfluctdv for every cell once any cell inserts it.

## 4. Per-muscle values currently present

- **FDI (the default everywhere it is muscle-specific):**
  - `Muscle`: radius 6.91 mm, density 350/mm², length 30 mm, territory ratio 1/4, mean CV 4.2 (sEMG), fibre length 31.7 ± 2.8, skin 1.2 / fat 1.3 mm, type fraction 0.574, type means 50 / 67 µm, conductivities from botelho2019. Location: `core/muscle/muscle.py:252-270`.
  - `SurfaceEMG.fiber_diameter__um` 57.9.
  - HillModel and GTO presets `"FDI"` (`neuron/muscle.py:237-317`, `proprioception/golgi.py:186-205`).
  - `fr_optimization_targets.EXPERIMENTAL_FR_TARGETS["fdi"]` (`utils/fr_optimization_targets.py:222`).
  - Optimiser `HIGH_FORCE_RATE_TARGETS` (FDI De Luca 1982).
  - The FDI DD calibrations in `examples/basic/results/dd_fdi_120mu_*` (gitignored; nine directories; adopted: `_fuglevand50_rec41_deluca82_human59_wexp`).
- **Tibialis anterior values applied to FDI:** `Muscle.mean/sd_fiber_conduction_velocity` 4.32 ± 0.71 and the dispersion CV-threshold slope 0.63 (`[delvecchio2018]`). `keenan2005` (FDI) supplies the tendon band.
- **VL:**
  - No VL physics on this branch.
  - The VL defaults are the optimiser's `--target-muscle vl` and `--n-mu 200`, the `de_luca_targets` defaults (16 / 12 Hz "untrained VL"), and `EXPERIMENTAL_FR_TARGETS["vl"]` and `["vl_trained"]` (`fr_optimization_targets.py:202-220`, `[skarabot2023]`).
  - The VL Gfluctdv reference value (0.12, fresh-trunk VAL-42) appears only in the PR doc §6.
- **Soleus:** Hill and GTO `"Sol"` presets. The Hill FL / FV / tendon coefficients are identical to FDI.
- **Not on this branch:** `MuscleProfile` / `MUSCLE_LIBRARY` (the optimiser's `_thresholds` docstring: "the muscle-profile library is not part of main"), so there is no per-muscle registry.

## 5. Gaps

### 5.1 Named in the brief or project docs, not present at `3bbde23eb`

`muscle/profiles.py` (`MuscleProfile`, `MUSCLE_LIBRARY`), `emg/electrode_profiles.py` (`ELECTRODE_LIBRARY`), `emg/cross_muscle_background.py` (`CrossMuscleBackgroundSpec`), `neuron/drive/common_drive.py` (`compose_common_drive`, `BetaBandSpec`, `EffortFluctuationSpec`), Albers near-field correction, fascicle and perimysial attenuation, `nmj_scatter_mode` / `mu_anchored`, mid-field LOD. These are fresh-trunk features; CLAUDE.md describes that line, not this branch. A profile design must not assume them.

### 5.2 Settings with no source

- **Neuron:** every NERLab YAML value, `gamma` 0.2, axon settings, the Python Gfluctdv values (20 ms, 1e-5, 1.2e-5), the DD and afferent shapes, the interneuron ranges, the network defaults (0.6 µS, -10 mV, 1 ms, p 1.0), the GTO filter. The Powers2017 block is "not checked". Spindle (Mileusnic 2006), GTO (Lin & Crago 2002, Aniss 1990b) and "Elias PhD thesis" are cited in docstrings but **not in the register**.
- **Muscle:** `length__mm` 30, territory ratio 1/4, the linear area-vs-threshold rule, `n_neighbours` 3, `conf` 0.999, `grid_resolution` 256, the border rescale, CV scatter 0.2 and the 2–6 clip (labelled assumptions), the 3-SD truncations.
- **Force:** no locator for T_L 90 or RT 3. About 40 values of the HillModel FDI dict are unsourced; the FL / FV coefficients are identical for FDI and Sol.
- **sEMG:** the 256 / 32 / 10 kHz grid, `IAP_SCALE_FACTOR` 2.5, the 10 mm tail, the IZ 10 % rule (cites botelho2019 without a locator), electrode defaults, the Laplacian form, `sig_bone`, the hybrid's 55 µm and ×4.2.
- **iEMG:** 10240 Hz, dz 0.01, end-plate centre 50 %, `suppress_endplate_density` / `endplate_width` 0.5, `min_radial_dist`, fibre ends [0, L], IED 0.5, every realistic-noise shape constant. σr / σz (andreassenrosenfalck1980) is LINK and "not checked".
- **Optimiser:** `SYNAPTIC_WEIGHT` 0.05, `GAMMA_SHAPE` 3, the ramp protocol and weights, 1 % drive noise. The ISI CV target is "not checked against the papers".

### 5.3 Contradictory defaults and docs

- **Register status vs code locators (verified):**
  - nandedkar1983 is "FT … Read 2026-11-02" (literature.md:73), but Corrections #8 says "**Never read in full:** Nandedkar & Stålberg 1983 …, Jacobson 1992 Table 1, Heckman & Binder 1991" (literature.md:165).
  - The code cites `[nandedkar1983 p. 163]` and `[jacobson1992 Table 1]`.
  - `force_utils.py:4` says fuglevand1993 is "(not read in full; no locator)", while `force_model.py` cites `[fuglevand1993 eq. 13]`, `eq. 14`, `eq. 17`.
  - The register summarises fuglevand1993 as "50-fold" thresholds, while force_model.py:51-52 says RR = 30.
- **Muscle and electrodes:**
  - `radius_bone__mm`: docstring 1 mm, code 0 (verified).
  - `bending_radius__mm`: the docstring says "radius of the muscle"; the geometry needs r_total (inferred).
- **Neuron:**
  - Default cell model: docstrings say "Powers2017" (cells.py:730) and "ModALS" (motor_neurons.py:27); the code says `"NERLab"`.
  - `axon_length` unit is "(mm)" in comments but used as m. Afferent Ib is 0.6 mm vs Ia/II 0.6 m.
  - The DD docstring says "(0-1)" but the code uses Hz.
  - `connect_to_muscle` / `connect_from_external` docstrings say weight 1.0 / 0.8; the code uses 0.6.
  - napp.mod says rest is -65 mV; the YAML frame rests at ≈ 0.
- **iEMG:** docstrings say mV, but the output is a normalised dimensionless block. The `peak_hz` docstring says 750.0, the code 1000.0.
- **Force:** the "9.05 P" saturation note is true only for the Fuglevand shape (≈ 4.4 P for the human shape, inferred). The `ForceModelVectorized` docstring omits two arguments.
- **Library vs calibration defaults:** library threshold mode `konstantin` vs optimiser `combined` vs calibration `fuglevand`; library DD `poisson` vs calibration `gamma`; library weight 0.6 µS all-to-all vs calibration 0.05 µS, equal fan-in, dendritic transfer.

### 5.4 Duplicated authorities (one physical quantity set in two or more places)

- **Fibre CV:** Muscle 4.2 (sEMG) / Muscle 4.32 ± 0.71 (iEMG) / dispersion slope and per-unit values.
- **Fibre diameter:** Muscle type means / SurfaceEMG 57.9 / `fiber_types.py` defaults / `simulate_fiber.py` defaults (0.0579) / hybrid 55 / iEMG fallback 55e-3.
- **σi 1.01:** SurfaceEMG argument / `simulate_fiber.py:301` / `bioelectric.py:318` / `runtime/density.py:74` / `fiber_simulation.py:282`.
- **Muscle σ:** Muscle (sEMG) / `bioelectric.py:101-102` / `fiber_simulation.py:58-59`.
- **λ:** SurfaceEMG (exposed) / `bioelectric.py` (hard 1) / `runtime/density.py` (pre-fix factor 2) / `fiber_simulation.py` (hard 1).
- **Rosenfalck A = 96:** `simulate_fiber.py:464`, `bioelectric.py`, `fiber_simulation.py`, `runtime/density.py:30`.
- **Fibre length:** `Muscle.length__mm` (iEMG) / `mean_fiber_length__mm` (sEMG) / Hill `L0`.
- **IZ spread `0.1·L`:** `surface_emg.py:322` and `runtime/semg.py:48`. **`IAP_SCALE_FACTOR` 2.5:** `surface_emg.py:492,682` and `runtime/semg.py:57`.
- **Branch CVs (5, 2):** `intramuscular_emg.py:111-114`, `motor_unit_sim.py:71-74`, `fiber_dispersion.py:107`.
- **Force:** ForceModel and HillModel (twitch T_L / RT literals duplicated, RP 30/50 vs 130, separate force–frequency laws). Fuglevand (2, 3) appears as a constant and again as `fuglevand_gain` defaults.
- **Spike thresholds:** YAML 50 (recording) / -10 (connections, muscle, injection helper).
- **Powers2017 parameters:** YAML ranges / `cells.py` literals / `.mod` defaults overridden by `h.*` GLOBALs. The PIC scale `gamma` is in the pool and in `_define_biophysics(gamma=1)` (dead). The napp gate kinetics are in the YAML and in napp.mod.
- **NEURON globals:** SimulationRunner sets `celsius` 36, dt and `secondorder` 2; the current-injection path sets none.
- **Electrode-size filter:** written three times (`simulate_fiber.py:560`, `fiber_simulation.py:210,464`).

### 5.5 Route-dependent behaviour

- **iEMG `execution=`:** runs the pre-0.13.0 kernel (verified), its cache namespace is unchanged since the fixes, and it refuses GPU.
- **sEMG `use_unified=True`:** silently drops `fiber_diameter__um` and σi, uses ×4.2 and a different reflection, and re-centres each fibre. It refuses λ ≠ 1 and dispersion.
- **sEMG `execution=`:** raises on dispersion, `use_unified`, `use_gpu`, electrode subclasses and mixed shapes.
- **No effect on any iEMG route:** `nmj_jitter__s`, `differentiation_mode`, the trajectory mixing matrix.

### 5.6 Defects found while reading (out of scope to fix; each needs its own issue)

- The optimiser's default `--threshold-mode combined` raises: `deluca__slope` is never passed (verified).
- `AlphaMN` passive mode is broken: `"pas"` vs `"passive"` (cells.py:993).
- `create_gto_parameters_for_muscle("generic")` raises NameError (golgi.py:202).
- Inhibitory connections onto interneurons become excitatory (single-synapse fallback).
- Gfluctdv noise and `SurfaceFiberDispersion.seed=0` are outside `myogen.set_random_seed`.
- The iEMG `execution=` route differs from the direct route by about 100× in amplitude and in shape.
- Defaults that make a degenerate geometry: sEMG bending radius = eps; iEMG electrode at the fibre end.

### 5.7 What a profile cannot reach today

- **Not constructor arguments:**
  - Gfluctdv on/off and scale: a setup call, and the scaling lives in the example script.
  - The DD wiring rules (equal fan-in, dendritic-transfer weights, exponent p): script-only.
  - iEMG λ, σr / σz, σi, `endplate_width`, fibre ends.
  - sEMG IZ spread, `IAP_SCALE_FACTOR`, `sig_bone`.
  - Muscle CV scatter and clip.
  - NEURON `celsius` / dt on the injection path.
- **Not tracked:** the calibration file. A profile would need to own or reference it, and record every filled setting in a manifest (backlog item 1, open questions).
