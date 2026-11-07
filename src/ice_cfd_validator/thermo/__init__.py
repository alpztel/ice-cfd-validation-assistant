"""0D/1D thermodynamic cycle diagnostics, NASA polynomials, heat transfer, and knock metrics."""

from __future__ import annotations

from ice_cfd_validator.thermo.heat_release import (
    BurnFractionMetrics,
    HeatReleaseResult,
    IMEPResult,
    KnockMetrics,
    compute_heat_release,
    compute_imep,
    compute_knock_metrics,
    extract_burn_fractions,
)
from ice_cfd_validator.thermo.heat_transfer import (
    HeatTransferResult,
    HohenbergCorrelation,
    WoschniCorrelation,
    compute_hohenberg_heat_transfer,
    compute_woschni_heat_transfer,
)
from ice_cfd_validator.thermo.kinematics import (
    KinematicsState,
    SliderCrankGeometry,
    TDCPhasingResult,
    compute_kinematics,
    compute_slider_crank_kinematics,
    verify_tdc_phasing,
)
from ice_cfd_validator.thermo.nasa_polynomials import (
    DIESEL_BURNED_PRODUCTS,
    DRY_AIR,
    STOICHIOMETRIC_PRODUCTS,
    NASAGasMixture,
    compute_cp,
    compute_cv,
    compute_gamma,
    make_combustion_mixture,
    mix_nasa_species,
)

__all__ = [
    "DIESEL_BURNED_PRODUCTS",
    "DRY_AIR",
    "STOICHIOMETRIC_PRODUCTS",
    "BurnFractionMetrics",
    "HeatReleaseResult",
    "HeatTransferResult",
    "HohenbergCorrelation",
    "IMEPResult",
    "KinematicsState",
    "KnockMetrics",
    "NASAGasMixture",
    "SliderCrankGeometry",
    "TDCPhasingResult",
    "WoschniCorrelation",
    "compute_cp",
    "compute_cv",
    "compute_gamma",
    "compute_heat_release",
    "compute_hohenberg_heat_transfer",
    "compute_imep",
    "compute_kinematics",
    "compute_knock_metrics",
    "compute_slider_crank_kinematics",
    "compute_woschni_heat_transfer",
    "extract_burn_fractions",
    "make_combustion_mixture",
    "mix_nasa_species",
    "verify_tdc_phasing",
]
