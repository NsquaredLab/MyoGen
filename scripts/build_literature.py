"""Build docs/literature.md, the literature register, from the table below.

Usage:
    python scripts/build_literature.py [--nexus NOTE.md]   # write docs/literature.md (and an Obsidian mirror)
    python scripts/build_literature.py --check             # exit 1 if the file is stale or code cites unknown keys

Code cites a source by its key and a locator in a comment, e.g. ``# [deluca1982 p. 118 Table 1]``. The code is
scanned for such citations (CITE_DIRS); each source lists where it is cited, and unknown keys are errors.
"""
import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
_ap.add_argument("--nexus", type=Path, default=None, help="also write the Obsidian mirror note here")
_ap.add_argument("--check", action="store_true", help="write nothing; fail if docs/literature.md is stale or a key is unknown")
ARGS = _ap.parse_args()
NEXUS = ARGS.nexus
DATE = "2026-11-02"

# Repo targets (relative to docs/)
R = {
    "semg": ("SurfaceEMG", "../myogen/simulator/core/emg/surface/surface_emg.py"),
    "simfib": ("simulate_fiber", "../myogen/simulator/core/emg/surface/simulate_fiber.py"),
    "muscle": ("Muscle", "../myogen/simulator/core/muscle/muscle.py"),
    "disp": ("SurfaceFiberDispersion", "../myogen/simulator/core/emg/surface/fiber_dispersion.py"),
    "nerve": ("nerve_branches", "../myogen/simulator/core/emg/nerve_branches.py"),
    "iemg": ("IntramuscularEMG", "../myogen/simulator/core/emg/intramuscular/intramuscular_emg.py"),
    "force": ("ForceModel", "../myogen/simulator/core/force/force_model.py"),
    "frt": ("fr_optimization_targets", "../myogen/utils/fr_optimization_targets.py"),
    "opt": ("DD optimiser", "../examples/02_finetune/06_optimize_dd_multilevel.py"),
    "fsf": ("07_fdi_spikes_force", "../examples/02_finetune/07_fdi_spikes_force.py"),
    "futil": ("force_utils", "../myogen/simulator/core/force/force_utils.py"),
    "mnpool": ("AlphaMN__Pool", "../myogen/simulator/neuron/populations/motor_neurons.py"),
    "mncfg": ("alpha_mn_default.yaml", "../myogen/config/alpha_mn_default.yaml"),
    "roe": ("Roeleveld bench", "../benchmarks/semg_roeleveld1997/README.md"),
    "dispb": ("dispersion bench", "../benchmarks/semg_dispersion/README.md"),
    "t_amp": ("test_surface_amplitude", "../tests/test_surface_amplitude.py"),
    "t_fib": ("test_muscle_fibre_properties", "../tests/test_muscle_fibre_properties.py"),
    "t_disp": ("test_surface_dispersion", "../tests/test_surface_dispersion.py"),
    "t_gfl": ("test_gfluctdv_noise", "../tests/test_gfluctdv_noise.py"),
    "t_dd": ("test_dd_optimizer_network", "../tests/test_dd_optimizer_network.py"),
    "cl": ("CHANGELOG", "../CHANGELOG.md"),
}
MAIN = "2026-10-31 sEMG amplitude fixes and fibre dispersion"
BANDS = "2026-10-31 FDI step-4 literature bands (draft)"
# Nexus anchors: (note, heading or None). Headings with ':' or backticks are linked at note level.
N = {
    "partA": (MAIN, None, "Part A"), "partB": (MAIN, None, "Part B"),
    "values": (MAIN, "Values chosen and why", None),
    "recheck": (MAIN, "Re-check on the default (FDI) cylinder", None),
    "skinlit": (MAIN, "Literature on FDI skin, fat and skin conductivity (2026-10-31)", None),
    "layers": (MAIN, "FDI with literature layer values (2026-10-31)", None),
    "sigma": (MAIN, "Where does skin σ = 1 S/m come from? (search 2026-10-31)", None),
    "fig6": (MAIN, "Roeleveld 1997 Fig. 6 check (exponent-free) and source strength (2026-10-31)", None),
    "hwang": (MAIN, "Hwang et al. 2013 read in full (FDI fibre diameter)", None),
    "polgar": (MAIN, "Polgar et al. 1973 read in full (FDI fibre diameter)", None),
    "diam": (MAIN, "FDI fibre-diameter bracket, simulated (2026-10-31)", None),
    "rates": (MAIN, None, "Firing rate: De Luca 1982 onion-skin rates"),
    "pool": (MAIN, "Pool size (FDI, 57.9 µm, De Luca 1982 rates)", None),
    "botcorr": (MAIN, None, "Correction: what the Botelho 2019 reference is"),
    "rebase": (MAIN, "Rebase onto main and re-test (2026-11-01 afternoon)", None),
    "mdf": (MAIN, "MDF gap (2026-11-01 evening)", None),
    "iapdec": (MAIN, None, "Decision: IAP length"),
    "late": (MAIN, "Late recruitment of the largest FDI units (2026-11-01 night)", None),
    "ceilwhy": (MAIN, "Why the ceiling sits at ~70 % MVC (2026-11-02)", None),
    "spinal": (MAIN, "Are spinal mechanisms behind the low high-force rates? (2026-11-02)", None),
    "calib": (MAIN, "Calibrations (120 MUs, Fuglevand 50-fold thresholds, De Luca 1982 targets)", None),
    "b1": (BANDS, "1. Single-MU surface MUAP peak-to-peak amplitude", None),
    "b2": (BANDS, "2. MUAP duration", None), "b3": (BANDS, "3. Global sEMG amplitude", None),
    "b4": (BANDS, "4. Spectrum", None), "b5": (BANDS, "5. Anatomy and physiology", None),
}

# key, section, citation, identifiers, status, use, supports, repo targets, nexus anchors
S = []
def src(*a):
    S.append(dict(zip(("key", "sec", "cite", "ids", "status", "use", "what", "repo", "nexus"), a)))

# ── A. Volume conductor, tissue, source strength, IAP ──
src("farina2004", "A", "Farina D, Mesin L, Martina S, Merletti R (2004). A surface EMG generation model with multilayer cylindrical description of the volume conductor. IEEE Trans Biomed Eng 51(3):415–426.",
    "doi:10.1109/TBME.2003.820998", "FT", "ADOPTED",
    "The sEMG volume conductor; reference for the notebook fixes (bipolar B(k_z), MUAP time axis, IAP z axis, fibre semi-lengths, amplitude). Layer σ (skin 1, fat 0.05, muscle 0.5/0.1 S/m) given without source.",
    ["semg", "simfib", "muscle", "t_amp", "cl"], ["partA", "sigma"])
src("roeleveld1997", "A", "Roeleveld K, Blok JH, Stegeman DF, van Oosterom A (1997). Volume conduction models for surface EMG; confrontation with measurements. J Electromyogr Kinesiol 7(4):221–232.",
    "doi:10.1016/S1050-6411(97)00009-6", "FT", "ADOPTED",
    "Provenance of skin σ = 1 S/m: a fitted effective value reproducing biceps MUAP fall-off (52 MUs), not a measurement. Replicated in the Roeleveld bench.",
    ["muscle", "roe", "t_amp", "cl"], ["sigma", "fig6", "rebase"])
src("roeleveld1997b", "A", "Roeleveld K et al. (1997), companion JEK paper on MUAP spread / exponents (details unverified).",
    "", "LINK", "CONTEXT", "Source of the depth/lateral fall-off regression used in the bench; not read.", ["roe"], ["sigma"])
src("blok2002", "A", "Blok JH, Stegeman DF, van Oosterom A (2002). Three-layer volume conductor model and software package for applications in surface electromyography. Ann Biomed Eng 30:566–577.",
    "doi:10.1114/1.1475345", "FT", "CONTEXT",
    "Skin 1.0 S/m 'taken from Roeleveld'; calls it 'somewhat ambiguous'. σi 0.5 S/m, d 50 µm (source-strength comparison).",
    ["muscle"], ["sigma", "fig6"])
src("lowery2002", "A", "Lowery MM, Stoykov NS, Taflove A, Kuiken TA (2002). A multiple-layer finite-element model of the surface EMG signal. IEEE Trans Biomed Eng 49(5):446–454.",
    "doi:10.1109/10.995683", "FT", "CONTEXT",
    "Counterpoint: measured skin 4.55e-4 S/m (Gabriel 1996); Roeleveld's skin 'considerably higher than measured'. σi 1.01 S/m (Table I).",
    ["muscle", "semg"], ["sigma"])
src("farinarainoldi1999", "A", "Farina D, Rainoldi A (1999). Compensation of the effect of sub-cutaneous tissue layers on surface EMG: a simulation study. Med Eng Phys 21:487–496.",
    "doi:10.1016/S1350-4533(99)00075-2", "FT", "CONTEXT",
    "Skin/fat ratio 20 (7 also tested), fat/muscle 0.5, cited to Roeleveld (EEG abstract), Geddes & Baker, Gootzen. 20 × 0.05 = 1 S/m.",
    [], ["sigma"])
src("farinamerletti2001", "A", "Farina D, Merletti R (2001). A novel approach for precise simulation of the EMG signal detected by surface electrodes. IEEE Trans Biomed Eng 48(6):637–646.",
    "doi:10.1109/10.923782; PMID 11396594", "FT", "CONTEXT",
    "Same conductivity ratios; IAP 96 mV/mm³ with λ = 1 / mm.", ["simfib"], ["sigma"])
src("merlettifarina2016", "A", "Merletti R, Farina D (eds) (2016). Surface Electromyography: Physiology, Engineering and Applications. Wiley-IEEE Press (ch. 8, Lowery).",
    "", "FT", "CONTEXT", "§8.6.3 gives muscle 0.093–0.8 and fat 0.02–0.1 S/m with sources, but no skin value.", [], ["sigma"])
src("botelho2019", "A", "Pereira Botelho D, Curran K, Lowery MM (2019). Anatomically accurate model of EMG during index finger flexion and abduction derived from diffusion tensor imaging. PLoS Comput Biol 15(8):e1007267.",
    "doi:10.1371/journal.pcbi.1007267; PMC6738720", "FT + FIG", "CONTEXT",
    "Main FDI comparator, measured: RMS 0.21 mV at 25 % MVC (IQR 0.19–0.26), MDF ~165 Hz (IQR 152–185), one subject, Delsys 5-pin. Existing muscle/fat σ defaults. Rejected as defaults: dry skin 4.88e-4 S/m; Rosenfalck α 0.5 (precedent for the optional λ).",
    ["muscle", "semg", "cl"], ["botcorr", "mdf", "b3", "b4", "layers"])
src("gabriel1996", "A", "Gabriel S, Lau RW, Gabriel C (1996). The dielectric properties of biological tissues (series). Phys Med Biol 41.",
    "", "SEC", "CONTEXT", "Dry skin ~2–5e-4 S/m; source of Lowery's and Botelho's skin values.", ["muscle"], ["sigma"])
src("itis", "A", "IT'IS Foundation tissue properties database (low-frequency conductivity).", "https://itis.swiss/virtual-population/tissue-properties/",
    "FT", "CONTEXT", "Skin 0.148 ± 0.042 S/m; 0.15 S/m in the cylinder gives ~2.5–2.9× too much amplitude.", [], ["skinlit", "sigma"])
src("dideriksen2019", "A", "Dideriksen JL, Farina D (2019). PLoS Comput Biol (amplitude cancellation and neural drive; title unverified).",
    "", "FT", "CONTEXT", "Uses skin 2 mm / 1 S/m, attributed to 'a previously proposed model'.", [], ["sigma"])
src("hodgkinnakajima1972", "A", "Hodgkin AL, Nakajima S (1972). The effect of diameter on the electrical constants of frog skeletal muscle fibres. J Physiol 221:105.",
    "doi:10.1113/jphysiol.1972.sp009742", "ABS", "CONTEXT",
    "Gi 0.59 S/m at 20 °C (≈1.0 S/m at 37 °C, Q10 1.37, our arithmetic): supports σi = 1.01 S/m in the line-source strength.", ["semg"], ["fig6"])
src("rosenfalck1969", "A", "Rosenfalck P (1969). Intra- and extracellular potential fields of active nerve and muscle fibres. Acta Physiol Scand Suppl 321.",
    "PMID 5383732", "SEC", "ADOPTED",
    "IAP Vm = 96 (λz)³ e^(−λz) − 90 mV, λ = 1 / mm (default; Devon 2026-11-01). λ exposed as SurfaceEMG(iap_lambda__per_mm). FDI fit λ 0.7 documented, not adopted.",
    ["simfib", "semg", "t_amp", "cl"], ["iapdec", "mdf"])
src("dimitrova2006", "A", "Dimitrova NA, Dimitrov GV (2006). Electromyography (EMG) modeling. Wiley Encyclopedia of Biomedical Engineering.",
    "", "FT", "CONTEXT", "IAP profile length = duration × CV; spectral content scales with IAP duration (MDF gap).", [], ["mdf"])
src("dimitrov1998", "A", "Dimitrov GV, Dimitrova NA et al. (1998). Three papers, Med Eng Phys 20:374, 580, 702 (titles unverified).",
    "", "FT", "CONTEXT", "Spectra vs IAP duration and CV. No measured human IAP duration found.", [], ["mdf"])
src("rodriguezfalces2013", "A", "Rodriguez-Falces J et al. (2013). Med Biol Eng Comput 50:447 (IAP / dipole metrics; title unverified).",
    "doi:10.1007/s11517-012-0879-7", "FT", "CONTEXT", "IAP duration metrics for the λ question.", [], ["mdf"])
src("petersen2019", "A", "Petersen E, Rostalski P (2019). A comprehensive mathematical model of motor unit pool organization, surface electromyography, and force generation. Front Physiol 10:176.",
    "doi:10.3389/fphys.2019.00176; PMC6418040", "FT", "CONTEXT", "Farina-model implementation with λ = 1.", [], ["mdf"])
src("andreassenrosenfalck1980", "A", "Andreassen & Rosenfalck (1980), muscle conductivities for the intramuscular kernel (details from code comment, unverified).",
    "", "LINK", "ADOPTED", "Stated source of the radial and longitudinal muscle conductivities of the intramuscular point-electrode kernel in bioelectric.py (same defaults in fiber_simulation.py). Until 0.13.0 the defaults were 63 / 330 labelled S/m (1000x tissue values); since 2026-11-02 they are 0.063 / 0.33 S/m, the values the code comment attributes to this paper. Not in Zotero. Cited in code; not checked.", [], [])
src("huigen2002", "A", "Huigen E, Peper A, Grimbergen CA (2002). Investigation into the origin of the noise of surface electrodes. Med Biol Eng Comput 40:332–338.",
    "doi:10.1007/BF02344216", "LINK", "ADOPTED", "With gondran1996, the stated basis of the 1/f^α baseline-drift form with α in 1.5–2.0 (default α 1.75, the midpoint) in IntramuscularEMG.add_noise and emg_noise.py. Not in Zotero. Cited in code; not checked.", [], [])
src("gondran1996", "A", "Gondran C, Siebert E, Yacoub S, Novakov E (1996). Noise of surface bio-potential electrodes based on NASICON ceramic and Ag–AgCl. Med Biol Eng Comput 34:460–466.",
    "doi:10.1007/BF02523851", "LINK", "ADOPTED", "With huigen2002, the stated basis of the 1/f^α baseline-drift form with α in 1.5–2.0 (default α 1.75) in IntramuscularEMG.add_noise and emg_noise.py. Not in Zotero. Cited in code; not checked.", [], [])
src("deluca2010", "A", "De Luca CJ, Gilmore LD, Kuznetsov M, Roy SH (2010). Filtering the surface EMG signal: Movement artifact and baseline noise contamination. J Biomech 43(8):1573–1579.",
    "doi:10.1016/j.jbiomech.2010.01.027; PMID 20206934", "LINK", "CONTEXT", "Cited in IntramuscularEMG.add_noise for broadband movement artifacts (0–20 Hz), which the baseline-drift model leaves out. Citation from Zotero (no full text attached), PMID from nexus note LIT-05. Cited in code; not checked.", [], [])

# ── B. Fibre CV, end-plates, tendon ends, branch delays ──
src("nandedkar1983", "B", "Nandedkar SD, Stålberg E (1983). Simulation of single muscle fibre action potentials. Med Biol Eng Comput 21:158–165.",
    "doi:10.1007/BF02441531", "FT", "ADOPTED",
    "Fibre CV = 0.95 + 0.05·d[µm] m/s (fixes main's 2.0 m/s for every fibre); CV–diameter line for diameter-scaled sources. Read 2026-11-02: a modelling construct, not a measured regression — Stålberg 1966 biceps diameters 25–85 µm and CV 3.7 ± 0.71 m/s, with 25 / 55 / 85 µm pinned to the 2-SD band 2.2 / 3.7 / 5.2 m/s; linear, square and square-root forms gave similar amplitudes (p. 163, Fig. 6). Also eq. 1 of delvecchio2018 (p. 7). At FDI values it is inconsistent: 57.9 µm (polgar1973) → 3.85 m/s, 4.2 m/s (nishizono1990) → 65 µm.",
    ["muscle", "disp", "t_fib", "t_disp", "cl"], ["values"])
src("nishizono1990", "B", "Nishizono H, Fujimoto T, Ohtake H, Miyashita M (1990). Muscle fiber conduction velocity and contractile properties estimated from surface electrode arrays. Electroencephalogr Clin Neurophysiol 75:75–81.",
    "doi:10.1016/0013-4694(90)90154-C; PMID 1688776", "FT", "ADOPTED", "FDI motor-unit CV by spike-triggered surface arrays (5 mm IED): 50 units from 4 men, 3.2–5.0 m/s, mean 4.2 ± 0.5 (p. 77) — MyoGen's default mean. Units recruited at 120–930 g, i.e. 5–42 % of the 2.2 kg mean MVC (p. 78); skin temperature held at 31–32 °C (p. 77); CV correlated with threshold force, r = 0.52 (p. 79). Botelho 2019 takes its CV range from this paper.", ["muscle"], ["values", "b5"])
src("kouyoumdjian2023", "B", "Kouyoumdjian JA et al. (2023). Muscle fiber conduction velocity in situ revisited. Front Neurol 14:1118510.",
    "doi:10.3389/fneur.2023.1118510; PMC9996126", "FT", "CONTEXT", "TA single-fibre CV 4.10 ± 0.66 m/s.", [], ["values"])
src("zalewska2012", "B", "Zalewska E et al. (2012). A method for determination of muscle fiber diameter using single fiber potential (SFP) analysis. Med Biol Eng Comput.",
    "doi:10.1007/s11517-012-0965-x; PMC3514689", "FT", "CONTEXT", "Alternative CV–diameter line v = 2.2 + 0.05(d − 25).", [], ["values"])
src("blijham2006", "B", "Blijham PJ, ter Laak HJ, Schelhaas HJ, van Engelen BGM, Stegeman DF, Zwarts MJ (2006). Relation between muscle fiber conduction velocity and fiber size in neuromuscular disorders. J Appl Physiol 100:1837–1841.",
    "doi:10.1152/japplphysiol.01009.2005", "FT", "CONTEXT",
    "Read 2026-11-02. Invasive MFCV (direct stimulation, biceps) against needle-biopsy diameters (mostly quadriceps; 53 of 96 in the same muscle) in 96 patients, 24 of whom had no neuromuscular disorder (pp. 1837–1838). Only extremes were paired: smallest fibre with slowest MFCV, largest with fastest; R 0.58 / 0.46 (p. 1838 Table 1). Overall fit MFCV = 0.043·d[µm] + 0.83 m/s (p. 1840, Fig. 5), 'remarkably similar' to nandedkar1983 (p. 1840). Control ranges biceps 30–70, quadriceps 50–95 µm, citing polgar1973 (p. 1840). Cited with nandedkar1983 by delvecchio2018 (p. 7, ref. 6). Supports the slope and intercept of the Nandedkar line; patient data, so not adopted as its replacement. Cited in code for the equal relative spread (within 2 %) of the two lines and for the quadriceps range against the removed 80 µm clip.", [], ["values"])
src("delvecchio2018", "B", "Del Vecchio A, Negro F, Felici F, Farina D (2018). Distribution of muscle fibre conduction velocity for representative samples of motor units in the full recruitment range of the tibialis anterior muscle. Acta Physiol 222:e12930.",
    "doi:10.1111/apha.12930", "FT", "ADOPTED",
    "TA, 10 subjects, 406 MUs: MU CV 2.78–6.21 m/s (4.32 ± 0.71), linear in recruitment threshold (R² 0.70 ± 0.09); 3.89 ± 0.50 m/s at RT 0–30 % MVC, 5.12 ± 0.45 at 50–70 % (p. 2, Fig. 1 p. 3). HD-sEMG grid of 64 electrodes. Since 2026-11-02 (Devon) 4.32 ± 0.71 sets the Muscle CV defaults (intramuscular EMG; was 3.89 ± 0.50 before fibre typing), matched over unit means since 2026-11-03 (was over fibres); R² 0.70 gives the 0.39 m/s residual scatter of cv_from_recruitment_force (derived); the linear CV-threshold relation is why fibre typing uses one diameter ladder across the pool. Within-unit fibre CV distribution unknown (p. 5); CoV 5 % is a modelling assumption (Devon). CV→diameter by MFD = (MUCV − 0.95)/0.05 (p. 7 eq. 1). Basis of fiber_cv_threshold_slope (0.63 × mean, derived). Earlier cited in error as the 2017 J Appl Physiol paper.",
    ["disp", "cl", "muscle"], ["values"])
src("delvecchio2017", "B", "Del Vecchio A, Negro F, Felici F, Farina D (2017). Associations between motor unit action potential parameters and surface EMG features. J Appl Physiol 123:835–843.",
    "doi:10.1152/japplphysiol.00482.2017", "FT", "CONTEXT",
    "TA, 13 subjects, ramps to 70 % MVC: MU CV linear in recruitment threshold (R² 0.64 ± 0.14); intercept 3.88 ± 0.35 m/s (Table 2), mean slope 0.0224 m/s per % MVC of threshold (mean of the 13 subject slopes in Table 3). Global MDF poorly related to recruitment. Not cited in code.",
    [], ["values"])
src("akhmadeev2020", "B", "Akhmadeev K, Yu T, Le Carpentier E, Aoustin Y, Farina D (2020). Simulation of motor unit action potential recordings from intramuscular multichannel scanning electrodes. IEEE Trans Biomed Eng 67(7):2005–2014. Cited in MyoGen as 'Konstantin et al. 2020' (Konstantin is the first author's given name).",
    "doi:10.1109/TBME.2019.2953680; code: github.com/smetanadvorak/iemg_simulator", "FT + CODE", "ADOPTED",
    "Two-layer end-plate / nerve-branch model (shared iEMG/sEMG). Published eq. 5–6 coefficients are the default; reference-code coefficients kept as nmj_coefficients='reference_code'. Branch CVs 5/2 m/s from the reference code (paper: 10/1).",
    ["nerve", "iemg", "disp", "t_disp", "cl"], ["partB", "values"])
src("stalbergkarlsson2001", "B", "Stålberg E, Karlsson L (2001). Simulation of the normal concentric needle electromyogram by using a muscle model. Clin Neurophysiol 112:464–471.",
    "doi:10.1016/S1388-2457(01)00459-X", "FT", "CONTEXT", "Bound: end-plate position 0 ± 1 mm; per-fibre delay 0–1 ms. Piasecki 2020 reuses this value (one chain, not two anchors). Companion paper: Simulation of EMG in pathological situations, Clin Neurophysiol 112:869–878, doi:10.1016/S1388-2457(01)00498-9 (title and pages were mixed up here before 2026-11-03; checked against Zotero). Their fibre diameter 50 ± 5 µm within each motor unit is a model setting, no measurement cited (literature check 2026-11-03).", [], ["values"])
src("piasecki2020", "B", "Piasecki M et al. (2020/2021). Near-fibre EMG (medRxiv; Clin Neurophysiol 132:1089).",
    "doi:10.1101/2020.10.07.20208348", "FT", "CONTEXT", "Healthy end-plate SD 1 mm (from Stålberg & Karlsson).", [], ["values"])
src("lateva2010", "B", "Lateva ZC, McGill KC, Johanson ME (2010). J Appl Physiol 108:1530.",
    "PMC2886675", "FT", "CONTEXT", "Myelinated branch ~50 m/s bounds fast branch delays; fibre-end spread.", [], ["values"])
src("keenan2005", "B", "Keenan KG, Farina D et al. (2005). J Appl Physiol 98:120 (FDI sEMG simulation).",
    "doi:10.1152/japplphysiol.00894.2004", "FT", "ADOPTED",
    "Convention for the opt-in 5 mm tendon-end band (no measurement of fibre-end stagger found).", ["disp", "cl"], ["values"])
src("farinafosci2002", "B", "Farina D, Fosci M, Merletti R (2002). J Appl Physiol 92:235.",
    "doi:10.1152/jappl.2002.92.1.235", "FT", "CONTEXT", "IZ / tendon spread 10 mm used as the sensitivity bound.", [], ["values"])
src("jahanmiri2015", "B", "Jahanmiri-Nezhad F et al. (2015). Front Hum Neurosci 9:239.",
    "doi:10.3389/fnhum.2015.00239", "FT", "CONTEXT", "Single-MU IZ spread 9–18 mm.", [], ["values"])

# ── C. FDI anatomy and layers ──
src("jacobson1992", "C", "Jacobson MD, Raab R, Fazeli BM, Abrams RA, Botte MJ, Lieber RL (1992). Architectural design of the human intrinsic hand muscles. J Hand Surg Am 17(5):804–809.",
    "doi:10.1016/0363-5023(92)90446-V", "ABS", "ADOPTED",
    "FDI fibre length 31.7 mm and var_fiber_length__mm 2.8 mm (used as a uniform half-width; probably a between-specimen spread). **Table 1 unverified.**",
    ["muscle", "cl"], ["recheck"])
src("li2018", "C", "Li H, Furst DE, Jin H et al. (2018). High-frequency ultrasound of the skin in systemic sclerosis. Arthritis Res Ther 20:181.",
    "doi:10.1186/s13075-018-1686-9", "FT", "ADOPTED", "Skin thickness 1.2 mm: hand dorsum over MCP II–III, median of 31 controls (0.79–1.50).",
    ["muscle", "t_amp", "cl"], ["skinlit", "b5"])
src("nishikawa2024", "C", "Nishikawa Y, Watanabe K, Holobar A et al. (2024). Sex differences in laterality of motor unit firing behavior of the first dorsal interosseous muscle. Eur J Appl Physiol 124:1979–1990.",
    "doi:10.1007/s00421-024-05420-7", "FT", "ADOPTED", "FDI depth to muscle 2.54–3.30 mm; fat 1.3 mm = 2.5 − 1.2 (derived from the low end).",
    ["muscle", "t_amp", "cl"], ["skinlit", "b5"])
src("storchle2018", "C", "Störchle P, Müller W, Sengeis M et al. (2018). Measurement of mean subcutaneous fat thickness. Sci Rep 8:16268.",
    "doi:10.1038/s41598-018-34213-0", "FT", "REJECTED", "Old fat default 0.3 mm (whole-hand average, excludes skin).", ["muscle"], ["skinlit"])
src("brodar1960", "C", "Brodar V (1960). Observations on skin thickness and subcutaneous tissue in man. Z Morphol Anthropol 50:386–395.",
    "", "SEC", "REJECTED", "Old skin default 1.29 mm (whole-body male average).", ["muscle"], ["skinlit"])
src("krammer2025", "C", "Krammer et al. (2025). Diagnostics 15:1600.", "", "FT", "CONTEXT", "Hand-dorsum dermis 0.52 ± 0.16 mm.", [], ["skinlit"])
src("infantolino2010", "C", "Infantolino BW, Challis JH (2010). Architectural properties of the first dorsal interosseous muscle. J Anat 216:463.",
    "doi:10.1111/j.1469-7580.2009.01196.x; PMC2849523", "FT", "CONTEXT", "Volume 5.2 mL, fascicle 29.9 mm; derived CSA ~89 mm² vs model 150 mm².", [], ["pool"])
src("martin1988", "B", "Martin TP, Bodine-Fowler S, Roy RR, Eldred E, Edgerton VR (1988). Metabolic and fiber size properties of cat tibialis anterior motor units. Am J Physiol 255:C43–C50.",
    "doi:10.1152/ajpcell.1988.255.1.C43", "FT", "CONTEXT",
    "Cat TA, 14 units by glycogen depletion, 8–10 unit fibres each, single 20-µm sections (p. C44). Within-unit fibre cross-section CoV per unit (Table 1 p. C46): type S 14/21/15/16 %, FR 10/18/19/34 %, FI 33/20 %, FF 28/30/31/32 %; mean 25 % against 46 % for non-unit fibres (p. C45–C46); Bodine et al. 1987 found 28 % vs 38 % (p. C46). Within-unit variability approximates that of the unit type (p. C49). Only within-unit fibre-size data found (read 2026-11-03). Diameter CoV ≈ area CoV / 2: S ≈ 0.08, fast ≈ 0.13. Muscle default within-unit diameter log-SD is 0.166 (from type_log_sd 0.2, an assumption), above every unit here.",
    [], ["values"])
src("andreassen1987", "B", "Andreassen S, Arendt-Nielsen L (1987). Muscle fibre conduction velocity in motor units of the human anterior tibial muscle: a new size principle parameter. J Physiol 391:561–571.",
    "doi:10.1113/jphysiol.1987.sp016756", "FT", "CONTEXT",
    "Human TA, 33 units by intramuscular axon microstimulation, surface tripolar array 15 mm: unit CV 2.6–5.3 m/s, 3.7 ± 0.7 (p. 567, Fig. 3D p. 566), a between-unit spread matching delvecchio2018 (0.71). Unit CV correlates with twitch torque r = 0.87 (p. 568 Fig. 4B). Argues (p. 569) that a wide unit-CV range needs fibres of one unit to share about the same CV and diameter; not measured. Twitch-torque ratio for FDI ~100 / 40 / 60–120 from other studies (Table 1 p. 565).",
    [], ["values"])
src("burketsairis1973", "C", "Burke RE, Tsairis P (1973). Anatomy and innervation ratios in motor units of cat gastrocnemius. J Physiol 234:749–765.",
    "doi:10.1113/jphysiol.1973.sp010370", "FT", "CONTEXT",
    "Cat MG: innervation ratios FF ~550–750, FR ~400–550 (p. 756; Table 2 p. 758); unit fibres spread over a wide territory, 1–2 per primary fascicle, 3–5 per 100 fibres (p. 753); mean unit fibre areas FF 5290, FR 2890, S 1734 µm² (p. 760). No within-unit fibre-area spread given.",
    [], ["values"])
src("polgar1973", "C", "Polgar J, Johnson MA, Weightman D, Appleton D (1973). Data on fibre size in thirty-six human muscles: an autopsy study. J Neurol Sci 19:307–318.",
    "doi:10.1016/0022-510X(73)90094-4", "FT", "ADOPTED",
    "FDI fibre diameter defaults (since 2026-11-02, Devon). FDI (autopsy): type I 50.0 µm, type II 67.0 µm [p. 310 Table 2]; type I fibres 57.4 % [p. 312 Table 4]. Weighted by that proportion: mean 57.2 µm (Muscle since 2026-11-02: fibre types by motor unit on one diameter ladder that keeps these type means, log-SD 0.2 per type an assumption; was a per-fibre two-type mixture earlier the same day), root-mean-square 57.9 µm (SurfaceEMG source default, since the source scales with d²). Replaces the 50 µm attributed to brooke1970. Type II − type I 17.02 µm, P < 0.001 (p. 312 Table 4). FDI 95 % limits for a single subject's mean (between post-mortems, not the fibre-to-fibre spread): type I 35–66, type II 47–86 µm (p. 313 Table 5; methods p. 309). Minor-axis diameters, ≥ 200 fibres per sample (p. 308). Gives no within-muscle fibre SD.", [], ["polgar", "diam"])
src("garnett1979", "C", "Garnett RAF, O'Donovan MJ, Stephens JA, Taylor A (1979). Motor unit organization of human medial gastrocnemius. J Physiol 287:33–43.",
    "doi:10.1113/jphysiol.1979.sp012643; PMC1281479", "FT", "CONTEXT",
    "57 human MG units (microstimulation, glycogen depletion, biopsy). S units small/slow/resistant, FR intermediate, FF large/fatigable [p. 33]; S units type 1 fibres, FF type 2b [p. 39–41]. Ranked by size, S and FR interleave at low force; only FF added above ~30 % cumulative twitch tension [p. 42]. Sample biased against largest/smallest units [p. 41]. Basis (with limits) for typing units by threshold (Muscle fibre typing, 2026-11-02, cited in fiber_types.py and Muscle docstring): strict type I-first order is a simplification.", ["muscle"], [])
src("hwang2013", "C", "Hwang K, Huan F, Kim DJ (2013). Muscle fibre types of the lumbrical, interossei, flexor, and extensor muscles moving the index finger. J Plast Surg Hand Surg 47:268–272.",
    "doi:10.3109/2000656X.2012.755988", "FT", "REJECTED",
    "FDI ~31 µm (elderly, paraffin, diameter undefined): judged an underestimate. Source of Botelho's 24.8–40.6 µm.", [], ["hwang", "diam"])
src("feinstein1955", "C", "Feinstein B, Lindegård B, Nyman E, Wohlfart G (1955). Morphologic studies of motor units in normal human muscles. Acta Anat 23:127–142.",
    "", "SEC", "CONTEXT", "FDI ~119 MUs, ~40,000 fibres, ~340 fibres/MU (via Piasecki 2016, De Luca 1982).", [], ["pool", "b5"])
src("zong2021", "C", "Zong Y et al. (2021). IEEE Trans Neural Syst Rehabil Eng 29:1199.",
    "doi:10.1109/TNSRE.2021.3088061; PMC8780215", "FT", "CONTEXT", "FDI MUNE 108 ± 21; CMAP 16.75 mV (ceiling).", [], ["b3", "b5"])
src("brooke1970", "C", "Brooke MH, Kaiser KK (1970). Muscle fiber types: how many and what kind? Arch Neurol 23:369–379.",
    "doi:10.1001/archneur.1970.00480280083010", "LINK", "CONTEXT",
    "The code cited it for the log-normal fibre diameters (median 50 µm until 0.13.0) and for the 20–80 µm clip, removed 2026-11-02 (it held ~10 % of fibres at 80 µm). By its title a histochemical fibre-typing paper; the values are not checked against it. The diameter mean now comes from polgar1973.", [], [])

# ── D–G. Recruitment, motor-neuron model, force, EMG benchmarks (section letter per entry; titles in TOPICS) ──
src("deluca1982", "D", "De Luca CJ, LeFever RS, McCue MP, Xenakis AP (1982). Behaviour of human motor units in different muscles during linearly varying contractions. J Physiol 329:113–128.",
    "doi:10.1113/jphysiol.1982.sp014293", "FT + FIG", "ADOPTED",
    "FDI rate-coding targets (onset 8.9 Hz, Table 1 peaks; fitted shape a modelling assumption) and the recruitment ceiling (~52 % MVC) behind --max-recruitment; Table 1 mean rates at 40/80 % MVC (25.3/41.4 Hz) are HIGH_FORCE_RATE_TARGETS (--high-force-rates).",
    ["opt", "frt"], ["rates", "calib"])
src("delucacontessa2012", "D", "De Luca CJ, Contessa P (2012). Hierarchical control of motor units in voluntary contractions. J Neurophysiol 107:178–195.",
    "", "SEC", "ADOPTED", "Exponential onion-skin rate law used for the de_luca targets.", ["frt", "opt"], ["rates"])
src("skarabot2023", "D", "Škarabot J et al. (2023). J Physiol (FDI/VL motor-unit decomposition).",
    "PMC10952962", "FT", "CONTEXT", "FDI recruitment 0.8–59 % MVF (ceiling); VL peak rates in fr_optimization_targets.", ["frt"], ["b5"])
src("milnerbrown1973a", "F", "Milner-Brown HS, Stein RB, Yemm R (1973). The contractile properties of human motor units during voluntary isometric contractions. J Physiol 228:285–306.",
    "doi:10.1113/jphysiol.1973.sp010087", "FT", "CONTEXT", "Methods; twitch shape.", [], ["rebase"])
src("milnerbrown1973b", "F", "Milner-Brown HS, Stein RB, Yemm R (1973). The orderly recruitment of human motor units during voluntary isometric contractions. J Physiol 230:359–370.",
    "doi:10.1113/jphysiol.1973.sp010192; PMC1350367", "FT", "CONTEXT",
    "145 FDI units: twitch vs threshold slope ~1, CT 30–100 ms; supports ForceModel defaults; >half of units below ~10 % MVC (threshold check). Twitch range ~100-fold vs model 50-fold (open).",
    ["force"], ["rebase"])
src("milnerbrown1975", "G", "Milner-Brown HS, Stein RB (1975). The relation between the surface electromyogram and muscular force. J Physiol 246:549–569.",
    "doi:10.1113/jphysiol.1975.sp010904", "FT", "CONTEXT", "FDI surface MUAP 0.14–0.40 mV (belly–tendon); duration 1.6–2.2 ms.", [], ["b1", "b2"])
src("fuglevand1999", "F", "Fuglevand AJ, Macefield VG, Bigland-Ritchie B (1999). Force-frequency and fatigue properties of motor units in muscles that control digits of the human hand. J Neurophysiol 81(4):1718–1729.",
    "doi:10.1152/jn.1999.81.4.1718", "FT", "ADOPTED", "13 hand units (intraneural stimulation): half-max force 9.1 Hz (CT 65.6 ms) / 15.5 Hz (CT 45.9 ms); 50 Hz gives 91 % of maximum, maximum at 50–100 Hz; twitch:tetanus 0.36 / 0.25; no relation between twitch force and CT. Used in the force-frequency refit (a 0.93, c 1.29).", ["force", "futil", "opt"], ["ceilwhy"])
src("mcnulty2000", "F", "McNulty PA, Falland KJ, Macefield VG (2000). Comparison of contractile properties of single motor units in human intrinsic and extrinsic finger muscles. J Physiol 526(2):445–456.",
    "doi:10.1111/j.1469-7793.2000.t01-2-00445.x", "FT", "ADOPTED", "28 hand units, 14 in FDI: intrinsic CT 70.3 ms (32–111), half-max 11.0 Hz, 34 % at 8 Hz, maximum force at 34.7 Hz (14–66), twitch:tetanus 0.27; CT unrelated to twitch force. Used in the force-frequency refit.", ["force", "futil", "opt"], ["ceilwhy"])
src("macefield1996", "F", "Macefield VG, Fuglevand AJ, Bigland-Ritchie B (1996). Contractile properties of single motor units in human toe extensors assessed by intraneural motor axon stimulation. J Neurophysiol 75(6):2509–2519.",
    "doi:10.1152/jn.1996.75.6.2509", "FT", "ADOPTED", "13 toe-extensor units: mean-force curve (Fig. 6C) 48/65/77/87/92 % at 10/15/20/30/50 Hz, CT 74.8 ms; maximum at 15–100 Hz (median 50); twitch:tetanus 0.22 (0.28 potentiated); CT unrelated to twitch force. Fuglevand gain saturates by ~20 Hz at this CT; used in the force-frequency refit (HUMAN_UNIT_FORCE_FREQUENCY, opt-in via ForceModel(force_frequency_shape=...) and --force-frequency human).", ["force", "futil", "opt"], ["ceilwhy"])
src("fuglevand1993", "F", "Fuglevand AJ, Winter DA, Patla AE (1993). Models of recruitment and rate coding organization in motor-unit pools. J Neurophysiol 70(6):2470–2488.",
    "doi:10.1152/jn.1993.70.6.2470", "SEC", "ADOPTED",
    "Twitch/force model (upstream fix e794774db, checked against an independent implementation) and 50-fold exponential thresholds.",
    ["force", "fsf", "opt"], ["rebase"])
src("destexhe2001", "E", "Destexhe A, Rudolph M, Fellous J-M, Sejnowski TJ (2001). Fluctuating synaptic conductances recreate in vivo-like activity in neocortical neurons. Neuroscience 107:13–24.",
    "", "SEC", "ADOPTED", "Gfluctdv conductance noise (was silent on main); scale calibrated per drive.", ["t_gfl", "opt", "cl"], [])
src("heckmanbinder1991", "E", "Heckman CJ, Binder MD (1991). Computer simulation of the steady-state input-output function of the cat medial gastrocnemius motoneuron pool. J Neurophysiol 65:952–967.",
    "PMID 2051212", "ABS", "ADOPTED",
    "Stated basis of the modelling assumption 'equal effective synaptic current per input' (DD input weights scaled by dendrite→soma transfer). Cat MG data; no data for corticospinal input to a hand muscle.",
    ["opt", "t_dd", "cl"], ["late"])
src("heckmanbinder1993", "E", "Heckman CJ, Binder MD (1993). J Neurophysiol 70 (effects of different synaptic input systems on recruitment; details unverified).",
    "", "LINK", "CONTEXT", "Likelier source for input-system differences; not read.", [], ["late"])
src("beauchamp2023", "E", "Beauchamp JA, Pearcey GEP, Khurram OU, Chardon M, Wang YC, Powers RK, Dewald JPA, Heckman CJ (2023). A geometric approach to quantifying the neuromodulatory effects of persistent inward currents on individual motor unit discharge patterns. J Neural Eng 20(1):016034.",
    "doi:10.1088/1741-2552/acb1d7; PMID 36626825; PMC9885522", "FT", "CONTEXT",
    "Human FDI (2296 MUs, triangular ramps to 30 % max): ΔF 4.80 [4.31, 5.30] pps, brace height 34.5 % rTri, acceleration slope 3.12, attenuation slope 0.624 pps/MVT (estimated-metric table). Benchmark for the PIC checks: MyoGen NERLab pool ΔF 0.2 pps. In their model pool ΔF depends on both neuromodulation and the inhibition pattern.", ["mnpool", "mncfg"], ["spinal"])
src("gorassini2002", "E", "Gorassini M, Yang JF, Siu M, Bennett DJ (2002). Intrinsic activation of human motoneurons: possible contribution to motor unit excitation. J Neurophysiol 87:1850–1858.",
    "doi:10.1152/jn.00024.2001; PMID 11929906", "LINK", "CONTEXT", "Origin of the paired motor-unit ΔF method used in the spinal check 2. Not read.", [], ["spinal"])
src("afsharipour2020", "E", "Afsharipour B, Manzur N, Duchcherer J, Fenrich KF, Thompson CK, Negro F, et al. (2020). Estimation of self-sustained activity produced by persistent inward currents using firing rate profiles of multiple motor units in humans. J Neurophysiol 124:63–85.",
    "doi:10.1152/jn.00194.2020; PMID 32459555", "LINK", "CONTEXT", "ΔF pairing criteria (control/test timing, rate-rate correlation); check 2 uses the common criteria from memory, not checked against this paper.", [], ["spinal"])
src("powersheckman2017", "E", "Powers RK, Heckman CJ (2017). Synaptic control of the shape of the motoneuron pool input-output function. J Neurophysiol 117:1171–1184.",
    "doi:10.1152/jn.00850.2016; PMID 28053245; PMC5340877; https://modeldb.science/239582", "ABS", "CONTEXT",
    "Cat MG model pool: PICs raise gain and make pool rate a nonlinear function of excitation; neuromodulation and balanced inhibition increase the nonlinearity, push-pull inhibition reduces it; later-recruited units give higher force. Origin of MyoGen's Powers2017 cell model: its mechanisms na3rp, naps, kdrRL, mAHP, gh, L_Ca_inact and Gfluctdv are those of ModelDB 239582, this paper's model (2026-11-02 diff: whitespace, a seed RANGE and comment lines). Cell parameter values not compared.", ["mnpool", "mncfg"], ["spinal"])
src("revillfuglevand2011", "E", "Revill AL, Fuglevand AJ (2011). Effects of persistent inward currents, accommodation, and adaptation on motor unit behavior: a simulation study. J Neurophysiol 106:1467–1479.",
    "doi:10.1152/jn.00419.2011; PMID 21697447", "LINK", "CONTEXT", "PIC simulation of rate saturation and onion skin. Not read.", [], ["spinal"])
src("revillfuglevand2017", "E", "Revill AL, Fuglevand AJ (2017). Inhibition linearizes firing rate responses in human motor units: implications for the role of persistent inward currents. J Physiol 595:179–191.",
    "doi:10.1113/JP272823; PMID 27470946", "ABS", "CONTEXT", "Human MU rates rise steeply at recruitment then level off at modest rates while force keeps rising; reflex inhibition removes the steep rise (attributed to PICs).", [], ["spinal"])
src("senneff2019", "E", "Senneff S, McManus L, Lowery MM (2019). Investigating the effect of persistent inward currents on motor unit firing rates and beta-band coherence in a model of the first dorsal interosseous muscle. Conf Proc IEEE EMBC 2019:2293–2296.",
    "doi:10.1109/EMBC.2019.8857534; PMID 31946358", "ABS", "CONTEXT", "Prior FDI motoneuron-pool model with PICs at 20–40 % MVC: PICs shift mean rates and reduce beta-band coherence. Prior art for the spinal checks.", [], ["spinal"])
src("powersbinder2001", "E", "Powers RK, Binder MD (2001). Input-output functions of mammalian motoneurons. Rev Physiol Biochem Pharmacol 143:137–263.",
    "doi:10.1007/BFb0115594", "LINK", "CONTEXT",
    "Review of motoneuron input-output properties; cited in the AlphaMN cell docstring. Not read here.", [], [])
src("bui2003", "E", "Bui et al. (2003). Morphological and electrophysiological characterization of Ia inhibitory interneurons in cat spinal cord (details from code comment, unverified).",
    "", "LINK", "ADOPTED", "Membrane area (81,390 µm²) and soma geometry of the INgII interneuron model in cells.py. Cited in code; not checked.", [], [])
src("sanches2018", "E", "Sanches et al. (2018), synaptic parameter measurements (details from code comment, unverified).",
    "", "LINK", "ADOPTED", "Default Exp2Syn synapse parameters (tau1 0.2 ms, tau2 0.3 ms, weight 0.6) in cells.py. Cited in code; not checked.", [], [])
src("segev1990", "E", "Segev et al. (1990), alpha-function synaptic modelling (details from code comment, unverified).",
    "", "LINK", "CONTEXT", "Listed as the reference for the synapse model form in cells.py (the code uses a double-exponential Exp2Syn). Cited in code; not checked.", [], [])
src("enokaduchateau2017", "D", "Enoka RM, Duchateau J (2017). Rate coding and the control of muscle force. Cold Spring Harb Perspect Med 7:a029702.",
    "doi:10.1101/cshperspect.a029702", "LINK", "CONTEXT",
    "Review: ISI CV 10–30 % in voluntary contractions, falling with force. With dideriksen2012, the basis of the optimiser's ISI CV target 15–22 % at 30 % MVC as synthesised in nexus LIT-17. Not read here.", [], [])
src("dideriksen2012", "D", "Dideriksen JL, Negro F, Enoka RM, Farina D (2012). Motor unit recruitment strategies and muscle properties determine the influence of synaptic noise on force steadiness. J Neurophysiol 107:3357–3369.",
    "doi:10.1152/jn.00938.2011", "LINK", "CONTEXT",
    "Simulation of synaptic noise and force steadiness; cited with enokaduchateau2017 for the ISI CV target (nexus LIT-17). Not read here.", [], [])
src("moritz2005", "D", "Moritz CT, Barry BK, Pascoe MA, Enoka RM (2005). J Neurophysiol 93:2449 (FDI discharge variability; details from memory, unverified).",
    "", "LINK", "CONTEXT", "ISI CV falls with rate — **quoted from memory, unverified**.", [], [])
src("hu2013", "G", "Hu X, Rymer WZ, Suresh NL (2013). J Neurophysiol 110:1205; companion J NeuroEng Rehabil (2013).",
    "PMC4073930; PMC3849018", "FT + FIG", "CONTEXT", "FDI decomposition; Delsys 5-pin geometry; MUAP amplitude linear in threshold.", [], ["b1", "b2"])
src("mcmanus2015", "G", "McManus L et al. (2015). J Neurophysiol 113:3186.", "doi:10.1152/jn.00146.2015", "FT", "CONTEXT", "FDI surface MDF 166.9 Hz.", [], ["b4"])
src("mcmanus2019", "G", "McManus L, Flood MW, Lowery MM (2019). J Neurophysiol.", "doi:10.1152/jn.00228.2019; PMC6766730", "FT + FIG", "CONTEXT",
    "MDF ~120–150 Hz (400 Hz low-pass). 'MUAP amplitude' rejected as a target (metric undefined).", [], ["b1", "b4"])

TOPICS = {"A": "Volume conductor and recording (tissue layers, conductivities, source strength, IAP)",
          "B": "Muscle fibres and innervation (conduction velocity, end-plates, tendon ends, branch delays)",
          "C": "Muscle anatomy (FDI dimensions, layers, motor-unit counts)",
          "D": "Recruitment and rate coding (thresholds, firing rates, recruitment ceiling)",
          "E": "Motor-neuron model and spinal mechanics (synaptic input, PICs, inhibition, noise)",
          "F": "Muscle force (twitch, contraction time, force-frequency)",
          "G": "Experimental EMG benchmarks (FDI amplitude and spectra)"}

INDEX = [
    ("sEMG volume conductor", ["farina2004"], ["semg", "simfib"]),
    ("Skin σ = 1 S/m (effective, fitted)", ["roeleveld1997", "blok2002", "farinarainoldi1999", "farinamerletti2001", "farina2004", "lowery2002", "gabriel1996", "itis", "merlettifarina2016"], ["muscle", "roe"]),
    ("Skin thickness 1.2 mm", ["li2018", "brodar1960", "krammer2025"], ["muscle"]),
    ("Fat thickness 1.3 mm", ["nishikawa2024", "storchle2018"], ["muscle"]),
    ("Muscle and fat σ", ["botelho2019", "merlettifarina2016"], ["muscle"]),
    ("Source strength σi·π(d/2)²", ["lowery2002", "blok2002", "hodgkinnakajima1972"], ["semg"]),
    ("IAP shape and λ (iap_lambda__per_mm)", ["rosenfalck1969", "farinamerletti2001", "botelho2019", "dimitrova2006", "dimitrov1998", "rodriguezfalces2013", "petersen2019"], ["simfib", "semg"]),
    ("Fibre CV vs diameter", ["nandedkar1983", "blijham2006", "zalewska2012"], ["muscle", "disp"]),
    ("Mean fibre CV 4.2 m/s", ["nishizono1990", "kouyoumdjian2023"], ["muscle"]),
    ("MU CV vs threshold; within-MU CV spread", ["delvecchio2018", "delvecchio2017"], ["disp"]),
    ("End-plate spread (two-layer model)", ["akhmadeev2020", "stalbergkarlsson2001", "piasecki2020", "jahanmiri2015"], ["nerve", "iemg", "disp"]),
    ("Nerve-branch delays", ["akhmadeev2020", "lateva2010"], ["nerve"]),
    ("Tendon-end band", ["keenan2005", "farinafosci2002", "lateva2010"], ["disp"]),
    ("FDI fibre length and spread", ["jacobson1992"], ["muscle"]),
    ("FDI fibre diameter", ["polgar1973", "hwang2013"], []),
    ("FDI MU number, fibres, CSA", ["feinstein1955", "zong2021", "infantolino2010"], []),
    ("FDI recruitment ceiling (--last-recruit-mvc)", ["deluca1982", "skarabot2023", "milnerbrown1973b", "macefield1996"], ["opt"]),
    ("Rate-coding targets", ["deluca1982", "delucacontessa2012"], ["frt", "opt"]),
    ("Twitch and force (ForceModel)", ["fuglevand1993", "milnerbrown1973a", "milnerbrown1973b", "fuglevand1999", "mcnulty2000", "macefield1996"], ["force", "fsf"]),
    ("Human-unit force-frequency curve (opt-in)", ["macefield1996", "fuglevand1999", "mcnulty2000"], ["futil", "force", "opt"]),
    ("High-force firing rates (--high-force-rates)", ["deluca1982"], ["opt"]),
    ("Gfluctdv noise, ISI CV", ["destexhe2001", "moritz2005"], ["opt"]),
    ("DD input weighting, equal fan-in", ["heckmanbinder1991", "heckmanbinder1993"], ["opt"]),
    ("Motoneuron PICs, ΔF, rate patterns (diagnostics 2026-11-02)", ["beauchamp2023", "gorassini2002", "afsharipour2020", "powersheckman2017", "revillfuglevand2011", "revillfuglevand2017", "senneff2019", "deluca1982"], ["mnpool", "mncfg"]),
    ("FDI sEMG comparators (RMS, MDF, MUAP)", ["botelho2019", "mcmanus2015", "mcmanus2019", "milnerbrown1975", "hu2013", "zong2021"], []),
]

CORRECTIONS = [
    "**Skin σ = 1 S/m rationale.** First explained as gel electrodes bypassing the stratum corneum (Farina 2004). Unsourced and withdrawn: the value is Roeleveld et al. 1997's fitted effective skin, inherited by Blok 2002, Farina & Rainoldi 1999, Farina & Merletti 2001 and Farina 2004. Commit message reworded; nexus Part A step 3 still has the old wording.",
    "**Roeleveld 1997 DOI** is 10.1016/S1050-6411(97)00009-6 (an earlier (96)00034-8 was wrong).",
    "**'Konstantin et al. 2020'** in MyoGen docstrings (muscle.py, intramuscular_emg.py, nerve_branches.py) is Akhmadeev et al. 2020. Fixed on `fix/semg-amplitude-scale` (028a14362); the `konstantin` threshold mode keeps its name.",
    "**Botelho 2019 simulated amplitude**: median ~0.16 mV; 0.10 mV was one sample trace. Their 24.8–40.6 µm diameters are Hwang 2013's FDI mean ± SD.",
    "**Feinstein 1955** authors are Feinstein, Lindegård, Nyman, Wohlfart; ~40,000 fibres (not 40,500).",
    "**Piasecki 2020 and Stålberg & Karlsson 2001** are one citation chain, not two anchors.",
    "**Jacobson 1992's 2.8 mm** is probably between-specimen spread; MyoGen uses it as a uniform half-width (docstring corrected).",
    "**Unverified, quoted from memory:** Moritz 2005 (ISI CV), Kukulka & Clamann 1981. **Never read in full:** Nandedkar & Stålberg 1983 (adopted CV line), Jacobson 1992 Table 1, Heckman & Binder 1991.",
    "**McManus 2019 'MUAP amplitude'**: metric undefined, not usable as a target.",
    "**ISI CV target 15–22 %** in the optimiser cites 'LIT-17' (iEMG nexus literature note); its primary source was not re-checked in these sessions.",
]

STATUS = "FT full text · FIG read off a figure · ABS abstract only · SEC known only through a citing paper · CODE source code · LINK named only"
USE = "ADOPTED sets a default or code value (or is its stated basis) · CONTEXT comparator, bound or background · REJECTED considered and not used"


def idlinks(ids):
    out = []
    for part in [p.strip() for p in ids.split(";") if p.strip()]:
        if part.startswith("doi:"):
            d = part[4:]
            out.append(f"[doi:{d}](https://doi.org/{d})")
        elif part.startswith("PMC"):
            out.append(f"[{part}](https://pmc.ncbi.nlm.nih.gov/articles/{part}/)")
        elif part.startswith("PMID"):
            out.append(f"[{part}](https://pubmed.ncbi.nlm.nih.gov/{part.split()[1]}/)")
        elif part.startswith("http"):
            out.append(f"[link]({part})")
        elif part.startswith("code:"):
            out.append(f"[code](https://{part[5:].strip()})")
        else:
            out.append(part)
    return " · ".join(out)


def repo_links(keys):
    return ", ".join(f"[{R[k][0]}]({R[k][1]})" for k in keys) or "—"


def nexus_links(keys, table=True):
    out = []
    bar = "\\|" if table else "|"  # a bare | would split a table cell
    for k in keys:
        note, head, label = N[k]
        out.append(f"[[{note}#{head}{bar}{head}]]" if head else f"[[{note}]] › {label}")
    return "<br>".join(out) or "—"


by_key = {s["key"]: s for s in S}

# In-code citations: [key] or [key locator], key = letters + year (+ optional letter), e.g. [mcnulty2000 Table 2].
CITE = re.compile(r"\[([a-z]+\d{4}[a-z]?)((?:[ ,;:][^\]\n]*)?)\]")
CITE_DIRS = ["myogen", "examples/finetune", "benchmarks", "tests", "scripts"]
CITE_SKIP = ("_archive", "__pycache__", "build_literature.py")


def scan_code():
    """{key: [(path relative to the repo, line, citation text)]} for every citation in CITE_DIRS."""
    hits = {}
    for d in CITE_DIRS:
        for f in sorted((REPO / d).rglob("*")):
            if f.suffix not in {".py", ".yaml", ".yml"} or any(x in f.parts or x == f.name for x in CITE_SKIP):
                continue
            for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                for m in CITE.finditer(line):
                    hits.setdefault(m.group(1), []).append((f.relative_to(REPO).as_posix(), i, m.group(0)))
    return hits


CITED = scan_code()
UNKNOWN = {k: v for k, v in CITED.items() if k not in by_key}


def cited_links(key):
    return ", ".join(f"[{Path(p).name}:{i}](../{p}#L{i})" for p, i, _ in CITED.get(key, []))


def render(nexus):
    L = []
    if nexus:
        L += ["---", f"created: {DATE}", "tags:", "  - semg", "  - literature", "  - register", "---", "",
              f"> Mirror of the repo register `docs/literature.md` (branch `calib/fdi-dd-main`), generated {DATE} from one table. "
              "The repo file is canonical for code links; this note adds links into the session notes. "
              f"Parent: [[README - Start Here]]. Full transcript sweep (≈120 sources incl. passing mentions and local copies): [[Literature sweep 2026-10-28 to 2026-11-01]]. "
              "Generator: `scripts/build_literature.py` in the repo (edit its table, rerun with `--nexus`, commit `docs/literature.md`).", ""]
    L += ["# Literature register", ""]
    if not nexus:
        L += ["Sources behind MyoGen's surface-EMG defaults, the fibre-dispersion options and the FDI descending-drive calibration "
              f"(work of 2026-10-30 to {DATE}). Each parameter links to its sources; each source links to the code, tests and notes that use it. "
              "Every parameter needs a literature basis or must be labelled a modelling assumption.", "",
              "**Citing in code:** a sourced value carries an inline comment with the key and a locator, e.g. "
              "`HIGH_FORCE_RATE_TARGETS = {40: 25.3, 80: 41.4}  # [deluca1982 p. 118 Table 1]`. Give only locators you have read; "
              "write \"no locator\" otherwise. Each key below resolves to the full citation and DOI.", "",
              "To add a source: add it to the table in `scripts/build_literature.py`, link it from the parameter index, cite the key in the code, "
              "and rerun the script (`--check` fails on unknown keys or a stale file). \"Cited at\" lists are generated from the code.", ""]
    L += [f"**Status:** {STATUS}  ", f"**Use:** {USE}", "", "## Parameter index", "",
          "| Parameter or claim | Sources | " + ("Session notes" if nexus else "Code") + " |", "|---|---|---|"]
    for name, keys, code in INDEX:
        srcs = ", ".join((f"[[#{by_key[k]['key']}]]" if nexus else f"[{k}](#{k})") for k in keys)
        if nexus:
            notes = sorted({n for k in keys for n in by_key[k]["nexus"]}, key=list(N).index)
            col = nexus_links(notes)
        else:
            col = repo_links(code)
        L.append(f"| {name} | {srcs} | {col} |")
    for sec, title in TOPICS.items():
        L += ["", f"## {sec}. {title}", ""]
        if nexus:
            for s in [x for x in S if x["sec"] == sec]:
                L += [f"### {s['key']}", f"{s['cite']} {idlinks(s['ids'])}", "",
                      f"- **{s['use']}**, {s['status']}: {s['what']}",
                      f"- Notes: {nexus_links(s['nexus'], table=False).replace('<br>', '; ')}",
                      f"- Repo: {', '.join(R[k][0] for k in s['repo']) or '—'}", ""]
        else:
            L += ["| Source | Status | Use | What it supports | Used in |", "|---|---|---|---|---|"]
            for s in [x for x in S if x["sec"] == sec]:
                ids = idlinks(s["ids"])
                used = repo_links(s["repo"]) + (f"<br>Cited at: {cited_links(s['key'])}" if s["key"] in CITED else "")
                L.append(f"| <a id=\"{s['key']}\"></a>**{s['key']}**: {s['cite']}{' ' + ids if ids else ''} | {s['status']} | {s['use']} | {s['what']} | {used} |")
    L += ["", "## Corrections and open verification", ""] + [f"{i}. {c}" for i, c in enumerate(CORRECTIONS, 1)]
    if nexus:
        L += ["", "## Local copies", "",
              "PDF/text copies (git-ignored), relative to the repository root: `sources/literature/2026-10_semg_fdi/lit/` "
              "(subfolders fdi/, fdi_layers/, fdi_pool/, iap/, mb73/, skin_sigma/, source_strength/, iemg_simulator/), with hwang2013/, polgar1973/ and roeleveld/ next to lit/. "
              "Inventory: [[Literature sweep 2026-10-28 to 2026-11-01]] §G."]
    return "\n".join(L) + "\n"


OUT = REPO / "docs" / "literature.md"
missing = [k for _, ks, _ in INDEX for k in ks if k not in by_key]
for k, v in UNKNOWN.items():
    for p, i, c in v:
        print(f"unknown citation key {k!r} at {p}:{i}: {c}", file=sys.stderr)
n_cites = sum(len(v) for v in CITED.values())
if ARGS.check:
    stale = not OUT.exists() or OUT.read_text(encoding="utf-8") != render(False)
    if stale:
        print("docs/literature.md is stale: run python scripts/build_literature.py", file=sys.stderr)
    sys.exit(1 if (stale or UNKNOWN or missing) else 0)
OUT.write_text(render(False), encoding="utf-8")
if NEXUS is not None:
    NEXUS.write_text(render(True), encoding="utf-8")
print(len(S), "sources;", len(INDEX), "index rows;", n_cites, "code citations of", len(CITED), "keys; missing index keys:",
      missing, "; unknown code keys:", sorted(UNKNOWN))
sys.exit(1 if (UNKNOWN or missing) else 0)
