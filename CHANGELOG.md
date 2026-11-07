# Changelog

All notable changes to **`ice-cfd-validation-assistant`** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-11-07

### Added
- **`ice_cfd_validator.constants`**:
  - Exact CODATA 2018 / CIPM-2007 / NIST SP 811 fundamental constants (`R_UNIVERSAL`, `MOLAR_MASS_DRY_AIR`, `R_SPECIFIC_DRY_AIR`, `STD_GRAVITY_M_S2`, `STD_ATM_PA`, `ZERO_CELSIUS_K`, `FEET_TO_M`).
  - Eight strict-SI conversion dictionaries (`PRESSURE_TO_PA`, `VOLUME_TO_M3`, `LENGTH_TO_M`, `MASS_TO_KG`, `TIME_TO_S`, `ANGLE_TO_DEG`, `TEMP_TO_K`, `HEAT_RATE_TO_J_PER_DEG`) supporting metric, CGS (`dyne/cm^2`, `erg/CA`), and affine Celsius (`degC -> K`) tokens.
  - Vectorized ISO 2533:1975 International Standard Atmosphere model (`compute_isa_conditions`, `ISAConditions`) covering both the linear-lapse troposphere (`0–11,000 m`) and isothermal lower stratosphere (`11,000–20,000 m`).
- **`ice_cfd_validator.io`**:
  - `forte_parser.py`: Automated 1-row and 2-row header parsers (`parse_thermo_csv`, `parse_spray_csv`, `parse_dynamic_csv`) returning immutable `ForteThermoData`, `ForteSprayData`, and `ForteDynamicData` containers with strict positivity bounds and `source_units` audit provenance.
  - `ecn_loader.py`: Sandia Engine Combustion Network (ECN Spray A / Spray D) experimental loader (`load_ecn_spray_data`, `ECNSprayDataset`) supporting comma- and whitespace-delimited tables, `#`/`%` comment headers, pre-SOI timestamp trimming, and experimental standard-deviation (`u_D`) extraction.
- **`ice_cfd_validator.signal`**:
  - `savgol_zero_phase` and `butterworth_zero_phase` (`sosfiltfilt`) zero-phase pressure smoothing and acoustic bandpass filters preventing artificial `CA50` crank-angle phase lag.
  - Loop-free `compute_derivative` combining 5-point $\mathcal{O}(\Delta\theta^4)$ central/Fornberg boundary stencils on uniform grids with analytical `CubicSpline` differentiation on non-uniform adaptive CFD grids.
- **`ice_cfd_validator.thermo`**:
  - `kinematics.py`: Centered and wrist-pin-offset slider-crank solver (`SliderCrankGeometry`, `KinematicsState`, `compute_kinematics`, `compute_slider_crank_kinematics`) with exact analytical $dV/d\theta$ (`rad -> deg` Jacobian) and sub-degree motored TDC loss-angle auditor (`verify_tdc_phasing`, `TDCPhasingResult`).
  - `nasa_polynomials.py`: 14-coefficient (two-interval `200–1000 K` & `1000–6000 K`) NASA RP-1311 / GRI-Mech 3.0 ideal-gas property engine (`NASAGasMixture`, `DRY_AIR`, `DIESEL_BURNED_PRODUCTS`, `STOICHIOMETRIC_PRODUCTS`, `mix_nasa_species`, `make_combustion_mixture`, `compute_cp`, `compute_cv`, `compute_gamma`).
  - `heat_transfer.py`: Vectorized Woschni (1967, SAE 670931) and Hohenberg (1979, SAE 790825) in-cylinder wall heat-transfer models (`WoschniCorrelation`, `HohenbergCorrelation`, `compute_woschni_heat_transfer`, `compute_hohenberg_heat_transfer`).
  - `heat_release.py`: Heywood (2018) single-zone First-Law net and gross AHRR (`compute_heat_release`), sub-degree cubic-spline `CA10`/`CA50`/`CA90` MFB solver (`extract_burn_fractions`), Gross/Net `IMEP` & `PMEP` (`compute_imep`), and `MAPO` + Eng (2002) acoustic Ringing Intensity (`compute_knock_metrics`).
- **`ice_cfd_validator.spray`**:
  - Vectorized continuous ($C^0$) Hiroyasu & Arai (1990, SAE 900475) two-regime spray penetration (`hiroyasu_arai_penetration`).
  - Vectorized Newton-exact Naber & Siebers (1996, SAE 960034) / Siebers (1999) momentum scaling (`naber_siebers_penetration`).
  - High-altitude gas-density over-penetration and sub-step piston-bowl wall-impingement auditor (`evaluate_altitude_spray_risk`) plus asynchronous CFD error norms (`compute_spray_validation_metrics`: `RMSE`, `MAPE`, $L_2$, $L_\infty$).
- **`ice_cfd_validator.vv`**:
  - Roache (1998), Celik et al. (2008), and ASME V&V 20-2009 three-mesh verification module (`compute_grid_size`, `compute_gci`, `GridSizeResult`, `GCIResult`) featuring 3D cube-root grid sizing, non-uniform $q(p)$ fixed-point / Brent root-finding, and `NaN`-proof Adaptive Mesh Refinement (AMR) oscillatory convergence handling.
- **`ice_cfd_validator.plotting`**:
  - Headless Agg 300-DPI SAE/IEEE STIX-mathtext diagnostic figure generators (`plot_thermo_diagnostics`, `plot_pv_log_diagram`, `plot_altitude_spray_comparison`, `plot_gci_convergence`).
- **Verification, Showcase & CI/CD**:
  - End-to-end non-proprietary Aero-Diesel altitude sweep (`examples/altitude_sweep_demo.py`: `0 ft`, `15,000 ft`, `30,000 ft`) and exported 300-DPI figures in `assets/`.
  - 39 automated unit, analytical roundtrip, AST vectorization, memory-leak, and `pyrefly` static-typing audit tests under `tests/`.
  - Multi-version Ubuntu GitHub Actions workflow (`.github/workflows/ci_validation.yml`) for Python 3.11, 3.12, and 3.13.

[0.1.0]: https://github.com/alpztel/ice-cfd-validation-assistant/releases/tag/v0.1.0

