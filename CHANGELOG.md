# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **Python 3.13 support.** `requires-python` is widened to `>=3.12,<3.14`, a `Python :: 3.13` classifier is added, and the CI test and wheel matrices now cover 3.13 (`cp312-* cp313-*`). NEURON 8.2.7 ships cp313 wheels, and the Cython extensions build and import under Python 3.13 against NumPy 2.x (verified locally).
- **NumPy 2.x support.** With `elephant` gone (the reason for the `numpy<2.0` pin), the runtime requirement is relaxed to `numpy>=1.26` and the build-system now compiles the Cython extensions against `numpy>=2.0`. Wheels built against NumPy 2.0 remain backward-compatible with NumPy 1.x at runtime (verified: the compiled extensions import under both NumPy 2.4 and 1.26). The `scipy<1.17` build cap is unrelated (manylinux2014 wheel availability) and is unchanged.

### Removed
- **`elephant` (and `viziphant`) dependency dropped entirely.** Spike binning via `elephant.conversion.BinnedSpikeTrain` is replaced by a dependency-free `myogen.utils.bin_spike_trains` helper (an exact reimplementation, verified bit-identical across grid/edge/fractional/sparse cases), and `elephant.statistics.isi` in `utils/helper.py` is replaced by `numpy.diff` (identical for sorted spike trains). The example scripts now compute firing rates, PSTHs and rasters natively (the `viziphant` raster is replaced by a small matplotlib helper). The `[elephant]` optional extra and the `elephant`/`viziphant` dev/docs dependencies are removed. `import myogen`, the test suite, and the docs build no longer require `elephant`.

## [0.9.0] - 2026-04-19

### Added
- **RNG accessors**: new public API for reproducible random draws
  - `myogen.get_random_generator()` — always returns the current global RNG (tracks the latest `set_random_seed` call)
  - `myogen.get_random_seed()` — returns the seed currently in effect
  - `myogen.derive_subseed(*labels)` — deterministic sub-seed helper for seeding non-NumPy generators (Cython spike generators, sklearn `random_state`, etc.) so they also track `set_random_seed`
- **Optional `elephant` extra**: install via `pip install myogen[elephant]`. The extra bundles both `elephant>=1.1.1` and `viziphant>=0.4.0` (viziphant was moved out of core dependencies because its only transitive import chain in core came from `elephant`). Core install is now genuinely elephant-free — verified with `importlib.util.find_spec('elephant') is None` on a fresh `uv sync`. The five modules that import elephant (`utils/helper.py`, `surface_emg.py`, `intramuscular_emg.py`, `force_model.py`, `force_model_vectorized.py`) were already guarded with `try/except ImportError` and now emit an accurate install hint when the extra is missing. `neo>=0.14.0` is now an explicit core dependency — it was previously only reached transitively through `elephant`/`viziphant`, so removing those would otherwise break `import myogen`
- **Test suite** under `tests/`
  - `test_determinism.py` — 10 regressions covering seed propagation, sub-seed collision avoidance, and deprecation warnings for the legacy RNG names
  - `test_firing_rate_statistics.py` — 4 regressions covering `FR_std` and `CV_ISI` NaN edge cases
  - Added `pytest>=8.0` to the `dev` dependency-group and a `[tool.pytest.ini_options]` section
- **Per-muscle ISI/CV figure**: new `plot_cv_vs_fr_per_muscle()` in `examples/02_finetune/05_plot_isi_cv_multi_muscle_comparison.py` emits a 3-panel VL / VM / FDI breakdown alongside the existing pooled plot

### Changed
- **RNG architecture** (internal refactor): the six internal modules and eight example scripts that previously did `from myogen import RANDOM_GENERATOR, SEED` at module scope now call the accessor at each site, eliminating the stale-reference bug where `set_random_seed` didn't reach downstream modules or seed-derived generators
- **Per-cell seed derivation**: the three sites in `myogen/simulator/neuron/cells.py` that built a per-cell seed as `SEED + (class_id+1)*(global_id+1)` — a formula that collided on swapped factors, e.g. `(0, 5)` and `(1, 2)` both resolved to `+6` — now use `derive_subseed(class_id, global_id)` (collision-free with respect to label order, built on `numpy.random.SeedSequence`). The `motor_unit_sim.py` `KMeans` `random_state` uses the same helper. **Consequence**: for a given global seed, RNG output differs from earlier releases; byte-identical reproduction of pre-0.9.0 outputs is not possible without matching code
- **Example figures**: bar graphs in `02_finetune/01_optimize_dd_for_target_firing_rate.py`, `02_finetune/02_compute_force_from_optimized_dd.py` and `03_papers/watanabe/01_compute_baseline_force.py` replaced with dumbbell / violin + box + jittered-scatter plots that show the underlying distribution (all points when n<10, violin + box when n≥10)
- **NEURON version alignment**: `README.md`, `docs/source/README.md`, `docs/source/index.md`, and `setup.py` Windows-install messaging now reference NEURON **8.2.7** (matching the Linux/macOS pip pin in `pyproject.toml` and the CI workflow) with the correct installer filename (`py-39-310-311-312-313`)
- **Docs build**: pinned `sphinx<9` in the `docs` dependency-group to work around a `sphinx-hoverxref` regression on Sphinx 9, and moved `[tool.pytest.ini_options]` in `pyproject.toml` so it no longer re-parents the `docs` dependency-group

### Deprecated
- **`myogen.RANDOM_GENERATOR`** and **`myogen.SEED`** module attributes. They remain accessible via a module-level `__getattr__` that emits a `DeprecationWarning` and returns the current RNG / current seed. External code should migrate to `get_random_generator()` / `get_random_seed()`; module-level `from myogen import RANDOM_GENERATOR` captures a stale reference and does not reflect later `set_random_seed` calls

### Fixed
- **`FR_std = NaN` with a single active unit**: `myogen/utils/helper.py::calculate_firing_rate_statistics` now returns `FR_std = 0.0` when fewer than two units pass the firing-rate filter, instead of propagating the `np.std(..., ddof=1)` NaN into downstream ensemble statistics
- **`CV_ISI = NaN` for n=1 ISI**: same function's per-neuron branch now returns `0.0` when `min_spikes_for_cv` is set to 2 and a neuron has exactly two spikes
- **`set_random_seed` did not propagate**: module-level `from myogen import RANDOM_GENERATOR, SEED` imports in the internal modules captured a stale reference at import time; reseeding the package rebinding did not affect them. The accessor refactor above is the fix
- **Typos / placeholder docstrings**: `extandable` → `extensible` in README and docs; `"as determined by no one"` docstring comments in `myogen/simulator/core/muscle/muscle.py` replaced with concrete physiological rationale

## [0.8.5] - 2026-01-15

### Fixed
- **Beartype Type Validators**: Fixed `SURFACE_MUAP__Block` and `SURFACE_EMG__Block` type validators to expect 2D signal shapes `(samples, n_electrodes)` instead of 3D, matching the actual output from `create_grid_signal()` which was refactored for NWB compatibility in v0.8.3
- **Firing Rate Calculation**: Fixed `inf` Hz firing rates in example `02_simulate_spike_trains_current_injection.py` by requiring at least 2 spikes to compute rate over spike range (single-spike neurons caused division by zero)

### Changed
- **Build System**: Pinned build dependencies in `pyproject.toml` to exact versions from `uv.lock` (`numpy==1.26.4`, `scipy==1.16.3`) to ensure reproducible wheel builds

## [0.8.4] - 2026-01-06

### Added
- **Verbose Parameter**: Added `verbose=True` parameter throughout the codebase to control tqdm progress bars and informational print statements
  - `ForceModel.generate_force(verbose=True)` - controls progress bars during force generation
  - `ForceModelVectorized.generate_force(verbose=True)` - controls print statements
  - `SurfaceEMG.simulate_muaps(verbose=True)` and `simulate_surface_emg(verbose=True)` - controls progress bars
  - `IntramuscularEMG.simulate_muaps(verbose=True)` and `simulate_intramuscular_emg(verbose=True)` - controls progress bars and print statements
  - `Muscle.generate_muscle_fiber_centers(verbose=True)` and `assign_mfs2mns(verbose=True)` - controls progress bars and status messages
  - `MotorUnitSim.calc_sfaps(verbose=True)` - controls progress bar during SFAP calculation
  - `SimulationRunner.run(verbose=True)` - controls simulation progress bar and completion message
  - `ContinuousSaver(verbose=True)`, `load_and_combine_chunks(verbose=True)`, `convert_chunks_to_neo(verbose=True)` - controls all logging output
  - `plot_innervation_areas_2d(verbose=True)` - controls plotting progress bar
  - All parameters default to `True` (original behavior). Set `verbose=False` to silence output.

## [0.8.3] - 2026-01-01

### Added
- **NWB Export Support**: New utilities for exporting simulation data to Neurodata Without Borders (NWB) format
  - Added `myogen.utils.nwb` module with `export_to_nwb()`, `export_simulation_to_nwb()`, and `validate_nwb()` functions
  - New optional dependency group `[nwb]` with `pynwb>=2.8.0` and `nwbinspector>=0.5.0`
  - Example script `13_load_and_inspect_nwb_data.py` demonstrating NWB export and loading
- **Grid Signal Utilities**: New `myogen.utils.neo` module for grid-annotated signals
  - `create_grid_signal()`: Create NWB-compatible AnalogSignals with grid metadata in annotations
  - `signal_to_grid()`: Convert 2D signals back to 3D grid format for analysis

### Changed
- **Grid Signal Storage**: Electrode array signals now stored as 2D (time, n_electrodes) for NWB compatibility
  - Grid structure preserved in signal annotations (`grid_shape`, `electrode_positions`, `ied`)
  - Use `signal_to_grid()` to convert back to 3D (time, rows, cols) for analysis
- **Neo Signal Naming**: Improved signal naming conventions for NWB compatibility
  - AnalogSignal names now follow NWB-compliant patterns

### Fixed
- **Sphinx Gallery NEURON State**: Added `reset_neuron()` function to reset NEURON state between examples
  - Fixes "works on second run" issue caused by HOC state persistence
- **Example Scripts**: Updated examples to use new grid signal API
  - `05_simulate_surface_muaps.py`: Use `signal_to_grid()` for 3D grid access
  - `06_simulate_surface_emg.py`: Access grid dimensions from annotations
  - `12_extract_data_from_neo_blocks.py`: Use `signal_to_grid()` for grid analysis

## [0.8.2] - 2025-12-28

### Added
- **Windows Prerequisites Documentation**: Added Visual C++ Build Tools requirement to README
  - Users must install [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
  - Required components: MSVC Build Tools for x64/x86 (Latest), MSVC v143 – VS 2022 C++ x64/x86 build tools, Windows 11 SDK (latest), C++ core desktop features

## [0.8.1] - 2025-12-28

### Changed
- **Dependencies**: Removed unused dependencies
  - Removed `pyqt6` (not used in codebase)
  - Moved `scipy-stubs` to dev dependencies only (was duplicated)

## [0.8.0] - 2025-12-28

### Added
- **Spinal Network Simulator**: Enhanced connectivity and afferent population support
  - Improved interneuron and motor neuron connectivity patterns
  - Added Group II (GII) afferent pathway support
  - Enhanced afferent population configuration options
- **Strict Mode for NEURON**: New error handling mode for mechanism loading
  - Added `strict` parameter to `load_nmodl_files()` function
  - Warnings by default, exceptions opt-in with `strict=True`
  - Clearer error messages when NEURON mechanisms fail to load
- **API Documentation**: Added missing classes to API docs and module exports
  - `HillModel` - Hill-type muscle model
  - `Network` - Spinal network container
  - `SimulationRunner` - Simulation orchestration
  - `JointDynamics` - Joint dynamics model
  - `SpindleModel` - Muscle spindle proprioceptor
  - `GolgiTendonOrganModel` - Golgi tendon organ proprioceptor
  - `ContinuousSaver` and `convert_chunks_to_neo` utility functions

### Changed
- **NEO Blocks Documentation**: Consolidated 4 documentation files into single `docs/neo_blocks.md`
  - Reduced from ~47KB across multiple files to ~4.5KB single digestible document
  - Clearer structure with block types, access patterns, and troubleshooting
- **API Documentation Order**: Reorganized `simulator_api.rst` to follow simulation workflow
  - RecruitmentThresholds → Populations → Network → SimulationRunner → Muscle → Proprioception → JointDynamics → Force → Electrodes → EMG
- **Watanabe Examples**: Updated paper citations from 2013 to 2015 (J. Neurosci.)
  - Corrected DOI references across all example scripts
  - Simplified README to essential information only
- **README**: Rewritten for beginners with uv as primary installation method

### Fixed
- **Documentation Links**: Fixed examples toctree paths (`01_basic`, `02_finetune`)
- **Literature Reproductions Link**: Added `watanabe-reproduction` ref label for cross-references

## [0.7.0] - 2025-12-24

### Added
- **Windows Registry Detection**: Automatic detection of NEURON installations on any drive
  - Implements Windows Registry-based NEURON discovery for multi-drive support
  - Checks `HKCU\SOFTWARE\NEURON_Simulator\nrn\Install_Dir` and `HKLM` equivalents
  - Parses uninstall registry entries to locate NEURON installation path
  - Supports NEURON installations on D:, E:, or any drive (not just C:)
  - Registry detection takes priority over hardcoded paths for more reliable discovery

### Changed
- **NEURON Detection Priority**: Updated detection order for Windows installations
  1. Windows Registry (new - works for any drive)
  2. NEURONHOME environment variable
  3. Hardcoded C: drive paths (backwards compatibility)
  4. PATH search for mknrndll.bat
- Applied registry detection to both build-time (`setup.py`) and runtime (`nmodl.py`) functions

## [0.6.11] - 2025-12-16

### Fixed
- **Windows Installation**: Improved NEURON DLL loading during installation
  - Uses `os.add_dll_directory()` to properly add NEURON bin to DLL search path
  - Installation proceeds even if NEURON DLLs can't be loaded
  - CSV files and package data get installed successfully regardless
  - NEURON mechanisms can be compiled later after fixing NEURON setup
  - Better error messages showing NEURON paths for debugging

## [0.6.10] - 2025-12-16

### Fixed
- **Windows Installation**: Added NEURON DLL detection in setup.py
  - Verifies NEURON can be imported before compilation
  - Provides clear PATH configuration instructions if DLLs not found
  - Resolves "DLL load failed while importing hoc" error
- **Package Data**: Added CSV files to package distribution
  - Includes voronoi_pi1e5.csv and other data files
  - Added to both MANIFEST.in and pyproject.toml package-data

## [0.6.9] - 2025-12-16

### Fixed
- **CI Build**: Fixed multi-line command execution in GitHub Actions
  - Changed CIBW_BEFORE_BUILD commands to single-line with && chaining
  - Resolves command concatenation error on macOS builds

### Changed
- **CI Build**: Temporarily disabled macOS wheel building
  - Focusing on Linux wheels to unblock PyPI publication
  - macOS users can install from source distribution

## [0.6.8] - 2025-12-16

### Fixed
- **Build System**: Compile NEURON mechanisms before wheel build in CI
  - Added nrnivmodl compilation step in CIBW_BEFORE_BUILD
  - Mechanisms now properly included in wheels published to PyPI
  - Verified locally that wheels contain x86_64/libnrnmech.so

## [0.6.7] - 2025-12-16

### Fixed
- **CRITICAL: Compiled NEURON Mechanisms Missing from PyPI Wheels**: Fixed CI workflow to preserve compiled mechanisms in published wheels
  - Replaced `auditwheel repair` with custom wheel renaming that preserves x86_64/ and arm64/ directories
  - Wheels now correctly tagged as manylinux_2_17_x86_64 while including all compiled NEURON mechanisms
  - Resolves "NEURON mechanisms not found" errors when installing from PyPI
  - Resolves "unsupported platform tag 'linux_x86_64'" upload error to PyPI
  - Users can now successfully `pip install myogen` or `uv add MyoGen` without compilation errors

## [0.6.6] - 2025-12-16

### Changed
- **Package Data Inclusion**: Enhanced MANIFEST.in and pyproject.toml to include additional file types
  - Improved package data inclusion for better distribution completeness
  - Ensures all necessary files are included in source distributions and wheels

## [0.6.5] - 2025-12-16

### Fixed
- **NEURON Mechanisms Not Loading**: Critical fix for "argument not a density mechanism name" error
  - Added automatic loading of NMODL mechanisms on MyoGen import
  - Added architecture-specific directories (x86_64, aarch64, arm64) to package-data
  - Mechanisms now load automatically from installed wheels
  - Resolves `ValueError: argument not a density mechanism name` for napp, kdrRL, etc.

## [0.6.4] - 2025-12-16

### Fixed
- **Missing Config Files**: Added YAML configuration files to package distribution
  - Fixed `FileNotFoundError: Configuration file not found: alpha_mn_default.yaml`
  - Added `**/*.yaml` to `tool.setuptools.package-data` in pyproject.toml
  - Config files now properly included in wheels and source distributions

## [0.6.2] - 2025-12-16

### Changed
- **Package Description**: Updated pyproject.toml description to match README overview
  - Now matches the language used in README for consistency
  - Emphasizes: modular framework, physiologically grounded, motor-unit activity, muscle force, surface EMG
- **README Installation Section**: Improved layout for better user experience
  - Quick install commands (`uv add MyoGen` / `pip install MyoGen`) now at top
  - Windows NEURON warning prominently displayed before install commands
  - Detailed system requirements and prerequisites moved below quick start
  - Better visual organization with horizontal rules

## [0.6.1] - 2025-12-16

### Fixed
- **README Display on PyPI**: Fixed logo not showing on PyPI package page
  - Replaced relative logo path with absolute GitHub URL
  - Logo now displays correctly on https://pypi.org/project/MyoGen/
- **Version Badge**: Updated from 0.5.0 to 0.6.1

## [0.6.0] - 2025-12-16

### Added
- **Package Metadata**: Complete PyPI metadata following Python packaging best practices
  - Added authors and maintainers information
  - Added license field (AGPL-3.0-or-later)
  - Added comprehensive classifiers for PyPI categorization
  - Added project URLs (homepage, documentation, repository, issues, changelog)
  - Added keywords for better package discoverability
- **Automatic Build System**: Comprehensive automatic NMODL compilation during package build
  - New `setup.py` with custom `BuildWithNMODL` class that compiles NMODL files during wheel building
  - Automatic Cython extension compilation integrated into build process
  - Platform-specific NMODL compilation for Linux (nrnivmodl) and Windows (mknrndll.bat)
  - No more manual `uv run poe setup_myogen` required for end users installing via pip
- **CI/CD Workflow**: GitHub Actions workflow for automated wheel building and publishing
  - Uses `cibuildwheel` for PyPI-compatible manylinux wheels (Linux) and universal wheels (macOS)
  - Automatic wheel building for Linux (x86_64) and macOS ARM (arm64) with pre-compiled NMODL mechanisms
  - Updated to use macos-latest (ARM) runner (macos-13 deprecated)
  - Wheels built on manylinux2014 with proper `manylinux_2_17_x86_64` platform tags
  - MPI libraries marked as external dependencies (users install system MPI separately)
  - Multi-platform testing of built wheels before publishing
  - OIDC-based PyPI publishing (no API tokens needed)
  - Automated MPI/OpenMPI installation in CI for both Linux and macOS
  - Source distribution (sdist) building alongside wheels
  - Auto-trigger PyPI publishing on GitHub releases and version tags
  - Automatic upload of wheels and sdist to GitHub release assets
- **Windows Installation Support**: Clear error handling for Windows users
  - Installation fails with helpful error message if NEURON not pre-installed on Windows
  - Directs users to download NEURON installer from official source
  - Automatic build from source when NEURON is available

### Changed
- **Platform Dependencies**: Updated NEURON dependency to be platform-specific
  - `neuron==8.2.7` now only installed automatically on Linux and macOS
  - Windows users must manually install NEURON before pip installing MyoGen
  - Combined dependency declaration for both Linux and macOS using `sys_platform` conditions
- **Elephant Dependency**: Added elephant from PyPI as required dependency
  - Elephant 1.1.1 included for spike train conversion and statistics
  - Provides `BinnedSpikeTrain` conversion and `isi` for EMG/force simulations
  - Graceful error handling if elephant import fails (development safety)
- **NumPy Version**: Pinned to numpy <2.0 for elephant compatibility
  - Elephant 1.1.1 requires numpy <2.0
  - Ensures stable, tested dependency versions
- **Sphinx Workflow**: Kept `setup_myogen` call in documentation build workflow
  - Required for editable installs used in CI/CD environments
  - Ensures NMODL mechanisms are compiled for documentation examples
- **Wheel Distribution**: Changed wheel building strategy
  - Pre-built wheels only for Linux and macOS (platforms with pip-installable NEURON)
  - Windows users install from source distribution with automatic build
  - Improved test isolation by removing source checkout from wheel testing job

### Fixed
- **CI Testing**: Fixed wheel testing to use installed packages instead of source
  - Removed source checkout from test_wheels job to prevent import conflicts
  - Tests now properly validate compiled Cython extensions in wheels
  - Ensures wheels contain all necessary compiled components

### Removed
- **Windows Wheels**: No longer building Windows wheels in CI
  - Windows users install from source distribution instead
  - Simplifies build process and avoids NEURON installation complications on Windows

## [0.5.0] - 2025-12-15

### Changed
- Version bump to 0.5.0
- Updated documentation links to latest version
- Updated Python version badge to indicate support for Python 3.12
- Improved basically everything

## [0.4.0] - 2025-08-10

### Added
- **Type System Enhancement**: New `RECRUITMENT_THRESHOLDS__ARRAY` custom type alias for 1D recruitment threshold arrays with runtime validation via Beartype
- **Development Guidelines**: Comprehensive `CLAUDE.md` with development protocols:
  - Git workflow with logical commit chunking and co-authorship requirements
  - Example development guidelines with professional plotting styles and Sphinx Gallery format
  - API testing requirements and font warning suppression protocols
  - CHANGELOG.md update requirements for all changes
- **Naming Standards**: Enhanced CLAUDE.md with comprehensive naming conventions:
  - Prohibition of unclear abbreviations (e.g., `mf`, `cv`)
  - Mandatory unit suffixes for all physical quantities
  - Consistent spatial coordinate notation (`positions__mm`, `centers__mm`)
  - Velocity notation standards (`conduction_velocities__mm_per_s`)
- **Class Documentation**: Added comprehensive Attributes sections to class docstrings documenting all computed properties
- **EMG API Consistency**: Updated SurfaceEMG and IntramuscularEMG classes with proper MyoGen framework patterns:
  - Immutable public argument pattern: constructor arguments accessible but never modified
  - Private result storage: simulation results stored in `_private` attributes  
  - Property-based access: computed results accessed via validated `@property` methods
  - Comprehensive error handling: informative errors with guidance for beginners
  - Enhanced method docstrings: document where results are stored after execution
- **Electrode Array Framework**: Standardized SurfaceElectrodeArray and IntramuscularElectrodeArray classes:
  - Applied immutable public arguments pattern with private copies for internal use
  - Added comprehensive property-based access to computed attributes (pos_z, pos_theta, electrode_positions)
  - Enhanced error handling with informative messages and guidance for beginners
  - Added detailed Attributes sections to class docstrings documenting all computed properties
- **Force Model Integration**: Enhanced ForceModel class with improved parameter naming and type validation:
  - Applied @beartowertype decorator for runtime parameter validation
  - Standardized parameter names with clear unit suffixes (contraction_time_range__unitless)
  - Updated property names for consistency (peak_twitch_forces__unitless, contraction_times__samples)
- **Current Generation API**: Standardized current generation functions with consistent parameter naming:
  - Updated sawtooth_current: widths → widths__ratio, timestep_ms → timestep__ms
  - Updated step_current: timestep_ms → timestep__ms  
  - Updated trapezoid_current: timestep_ms → timestep__ms
- **Import Organization**: Restructured import statements across all modules for better dependency management and consistency

### Changed
- **API Breaking Changes**: Enhanced `generate_mu_recruitment_thresholds` function with improved type safety:
  - Added `@beartowertype` decorator for runtime parameter validation
  - Updated parameter names with scientific unit suffixes:
    - `recruitment_range` → `recruitment_range__ratio` (dimensionless ratio)
    - `konstantin__max_threshold` → `konstantin__max_threshold__ratio` (dimensionless ratio)
    - `deluca__slope__per_hundred_units` → `deluca__slope` (dimensionless shape parameter)
  - Return type now uses `RECRUITMENT_THRESHOLDS__ARRAY` custom type
  - Enhanced docstring with explicit dimensionality information and corrected examples
- **Muscle Class API Breaking Changes**: Standardized attribute naming with clear unit suffixes and descriptive names:
  - `mf_centers` → `muscle_fiber_centers__mm` (muscle fiber center positions in mm)
  - `mf_diameters` → `muscle_fiber_diameters__mm` (muscle fiber diameters in mm)  
  - `mf_cv` → `muscle_fiber_conduction_velocities__mm_per_s` (conduction velocities in mm/s)
  - `muscle_border` → `muscle_border__mm` (muscle boundary points in mm)
  - `innervation_center_positions` → `innervation_center_positions__mm` (motor unit centers in mm)
  - Applied class-level `@beartowertype` decorator for automatic method type validation
  - Enhanced class docstring with comprehensive Attributes section documenting all computed properties
  - Updated plotting utilities and EMG simulation modules to use new attribute names
- **Example Updates**: Updated all examples to use new API parameter names and consistent patterns:
  - Updated `00_simulate_recruitment_thresholds.py` with new parameter names
  - Updated `02_simulate_muscle.py` with reduced fiber density for faster demonstration
  - Updated `03_simulate_surface_muaps.py` to remove deprecated MUs_to_simulate parameter
  - Updated `04_simulate_surface_emg.py` with corrected file path references
  - Updated `05_simulate_currents.py` with standardized current generation parameter names
  - Updated `06_simulate_force.py` with new force model parameter names and property access
  - Updated `08_simulate_intramuscular_emg.py` with new recruitment threshold parameter names
- **Repository Cleanup**: Enhanced .gitignore to exclude generated files and development artifacts:
  - Added .idea/ (IDE configuration files)
  - Added docs/source/auto_examples/, docs/source/generated/, docs/source/sg_execution_times.rst (Sphinx generated files)
  - Added examples/results/, results/ (temporary simulation outputs)
  - Added test_*.png, *.code-workspace (development artifacts)

### Fixed
- **Parameter Naming**: Removed misleading `__per_hundred_units` suffix from `deluca__slope` parameter that incorrectly suggested it had units
- **Scientific Accuracy**: Corrected `deluca__slope` description as dimensionless curvature control parameter rather than rate parameter

### Removed
- **Deprecated Plotting Module**: Removed `myogen/utils/plotting/plotting.py` module with generic plotting functions
  - Functionality has been moved to specialized plotting modules (currents.py, force.py, muscle.py, etc.)
  - This provides better organization and clearer separation of concerns

## [0.3.0] - 2025-08-08

### Added
- **Intramuscular EMG Framework**: Enhanced intramuscular EMG simulation capabilities with improved framework
- Optional CUDA/CuPy support for accelerated computations in intramuscular EMG
- New type definitions for EMG simulation components:
  - `CORTICAL_INPUT__MATRIX` type for cortical input data
  - `INTRAMUSCULAR_MUAP_SHAPE__TENSOR` for intramuscular electrode arrays

### Changed
- **Major Refactor**: Improved intramuscular EMG simulation framework with better type annotations and imports
- Enhanced `IntramuscularEMG` class with better parameter handling
- Optimized bioelectric field calculations for needle electrodes
- Updated surface EMG components for consistency with intramuscular framework
- Enhanced muscle model integration with electrode positioning
- Improved motor unit simulation with better parameter handling
- Enhanced recruitment thresholds plotting utilities
- Updated project dependencies and compatibility with latest package versions
- Enhanced Sphinx documentation configuration with better type aliases and extensions
- Renamed `MUAP_SHAPE__TENSOR` to `SURFACE_MUAP_SHAPE__TENSOR` for better clarity

### Fixed
- Ensured Mermaid diagrams have consistent white background in documentation
- Improved readability and visual consistency in documentation
- Enhanced development experience with updated VSCode settings

## [0.2.0] - 2025-07-26

### Added
- **Cortical Input Module**: Comprehensive cortical input generation functionality with multiple waveform types:
  - `create_sinusoidal_cortical_input()` - Sinusoidal cortical inputs with configurable amplitude, frequency, offset, and phase
  - `create_sawtooth_cortical_input()` - Sawtooth waveform inputs with adjustable width and phase
  - `create_step_cortical_input()` - Step function inputs with configurable duration and height
  - `create_ramp_cortical_input()` - Linear ramp inputs between start and end firing rates
  - `create_trapezoid_cortical_input()` - Trapezoidal inputs with configurable rise, plateau, and fall times
- New example script `07_simulate_cortical_input.py` demonstrating cortical input simulation
- Enhanced spike train plotting functionality with improved visualization tools

### Changed
- **API Simplification**: Removed explicit `load_nmodl_files()` calls from example scripts for cleaner user experience
- Improved code readability and organization in spike train classes
- Enhanced error handling and callback mechanisms in spike train simulation

### Fixed
- Fixed callback errors in spike train simulation for improved stability
- Improved code structure and readability across multiple modules

## [0.1.1] - 2025-07-26

### Added
- Surface electrode array classes (`SurfaceElectrodeArray`) for EMG simulation
- Numba dependency for performance optimization

### Changed
- Refactored surface EMG simulation for improved performance and API consistency
- Updated current creation functions for better usability
- MUAPs will be min-max scaled when generating surface EMG signals to avoid numerical instability
- Generated EMG signals will be sampled to the sampling rate of the MUAPs to avoid numerical instability
- Enhanced Muscle model documentation and updated parameters for improved accuracy
- Improved simulation scripts readability and updated muscle parameters
- Updated numpy version requirement to >=1.26 for better compatibility
- Updated time axis calculation for surface EMG plotting
- Adapted saved surface EMG data format to work with new API

### Fixed
- Improved handling of NaN values in MUAP scaling
- Updated tensor dimensions in type annotations for better type safety
- Commented out IntramuscularEMG and IntramuscularElectrodeArray imports to resolve import issues

### Removed
- Unnecessary files cleaned up from repository

## [0.1.0] - 2025-07-19

### Added
- Initial release of MyoGen EMG simulation toolkit
- Surface EMG simulation capabilities
- Motor neuron pool modeling
- Muscle fiber simulation
- Force generation modeling
- Comprehensive plotting utilities
- Documentation with Sphinx
- Example gallery with multiple simulation scenarios
- Support for Python >=3.12

### Features
- **Surface EMG Simulation**: Complete framework for simulating surface electromyography signals
- **Motor Unit Modeling**: Physiological motor neuron pool simulation with recruitment thresholds
- **Muscle Mechanics**: Detailed muscle fiber and force generation modeling
- **Signal Processing**: Tools for EMG signal analysis and visualization
- **Extensible Architecture**: Modular design for easy extension and customization

### Documentation
- Comprehensive API documentation
- Tutorial examples covering key use cases
- Gallery of simulation examples with visualizations
- Getting started guide and installation instructions

---

## Types of Changes
- **Added** for new features
- **Changed** for changes in existing functionality  
- **Deprecated** for soon-to-be removed features
- **Removed** for now removed features
- **Fixed** for any bug fixes
- **Security** for vulnerability fixes 