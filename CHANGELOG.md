# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.7] - 2025-12-16

### Fixed
- **CRITICAL: Compiled NEURON Mechanisms Missing from PyPI Wheels**: Fixed CI workflow to preserve compiled mechanisms in published wheels
  - Disabled `auditwheel repair` (Linux) and `delocate` (macOS) which were stripping out x86_64/ and arm64/ directories
  - Wheels now correctly include libnrnmech.so and all compiled NEURON mechanisms
  - Resolves "NEURON mechanisms not found" errors when installing from PyPI
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