r"""Aero-Diesel auto-ignition kinetics, Livengood-Wu integral, and CFD ignition diagnostics.

Provides, in strict SI units and fully vectorized form:

1. Cetane-number-dependent apparent activation energy and global Hardenberg &
   Hase (1979) ignition delay with high-altitude misfire threshold guard.
2. Instantaneous-state Arrhenius delay correlations (Wolfer 1938, Assanis et
   al. 2003) with cetane-number activation energy scaling and standard
   fuel profiles (EN590 Diesel, Jet-A1/JP-8, Synthetic FT-SPK).
3. Dynamic Livengood-Wu (1955) auto-ignition integral solver with sub-degree
   start-of-combustion root finding (Hermite / PCHIP) and misfire detection.
4. Multi-criteria ANSYS Forte 3D CFD ignition diagnostics: evaporative-cooling
   trough and zero-crossing AHRR timing with cooling energy $Q_{evap}$,
   cumulative positive heat thresholds CA01/CA02/CA05/CA50, maximum pressure
   acceleration $\max(d^2P/d\theta^2)$, and two-stage cool flame (LTHR / NTC /
   HTHR) separation.
"""

from __future__ import annotations

from ice_cfd_validator.ignition.cfd_detector import (
    CFDIgnitionMetrics,
    TwoStageIgnitionResult,
    detect_two_stage_ignition,
    extract_cfd_ignition_metrics,
)
from ice_cfd_validator.ignition.correlations import (
    ASSANIS_2003,
    HH_ACTIVATION_NUMERATOR_J_PER_MOL,
    HH_CETANE_OFFSET,
    HH_MISFIRE_MESSAGE,
    HH_PRESSURE_THRESHOLD_BAR,
    HH_PRESSURE_THRESHOLD_PA,
    HH_REFERENCE_RT_J_PER_MOL,
    PREDEFINED_FUELS,
    WOLFER_1938,
    ArrheniusDelayCoefficients,
    FuelIgnitionProps,
    HardenbergHaseResult,
    arrhenius_state_delay,
    coefficients_from_fuel,
    compute_apparent_activation_energy,
    hardenberg_hase_delay,
    scale_activation_energy,
)
from ice_cfd_validator.ignition.livengood_wu import (
    LivengoodWuResult,
    livengood_wu_integral,
    solve_livengood_wu,
)

__all__ = [
    "ASSANIS_2003",
    "CFDIgnitionMetrics",
    "FuelIgnitionProps",
    "HardenbergHaseResult",
    "HH_ACTIVATION_NUMERATOR_J_PER_MOL",
    "HH_CETANE_OFFSET",
    "HH_MISFIRE_MESSAGE",
    "HH_PRESSURE_THRESHOLD_BAR",
    "HH_PRESSURE_THRESHOLD_PA",
    "HH_REFERENCE_RT_J_PER_MOL",
    "LivengoodWuResult",
    "PREDEFINED_FUELS",
    "TwoStageIgnitionResult",
    "WOLFER_1938",
    "ArrheniusDelayCoefficients",
    "arrhenius_state_delay",
    "coefficients_from_fuel",
    "compute_apparent_activation_energy",
    "detect_two_stage_ignition",
    "extract_cfd_ignition_metrics",
    "hardenberg_hase_delay",
    "livengood_wu_integral",
    "scale_activation_energy",
    "solve_livengood_wu",
]
