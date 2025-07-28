# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

### Fixed

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