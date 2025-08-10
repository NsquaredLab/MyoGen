# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Type System Enhancement**: New `RECRUITMENT_THRESHOLDS__ARRAY` custom type alias for 1D recruitment threshold arrays with runtime validation via Beartype
- **Development Guidelines**: Comprehensive `CLAUDE.md` with development protocols:
  - Git workflow with logical commit chunking and co-authorship requirements
  - Example development guidelines with plt.xkcd() usage and Sphinx Gallery format
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
- **Example Updates**: Updated `00_simulate_recruitment_thresholds.py` to use new API parameter names

### Fixed
- **Parameter Naming**: Removed misleading `__per_hundred_units` suffix from `deluca__slope` parameter that incorrectly suggested it had units
- **Scientific Accuracy**: Corrected `deluca__slope` description as dimensionless curvature control parameter rather than rate parameter

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