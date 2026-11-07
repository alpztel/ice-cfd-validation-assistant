"""High-altitude Aero-Diesel spray breakup and penetration scaling correlations."""

from __future__ import annotations

from ice_cfd_validator.spray.correlations import (
    AltitudeSprayRiskResult,
    HiroyasuAraiResult,
    NaberSiebersResult,
    SprayValidationMetrics,
    compute_spray_validation_metrics,
    evaluate_altitude_spray_risk,
    hiroyasu_arai_penetration,
    naber_siebers_penetration,
)

__all__ = [
    "AltitudeSprayRiskResult",
    "HiroyasuAraiResult",
    "NaberSiebersResult",
    "SprayValidationMetrics",
    "compute_spray_validation_metrics",
    "evaluate_altitude_spray_risk",
    "hiroyasu_arai_penetration",
    "naber_siebers_penetration",
]
