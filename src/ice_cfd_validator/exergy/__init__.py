"""Thermodynamic Second Law exergy analysis and irreversibility quantification package.

Implements the Rakopoulos & Giakoumis (2006) and Caton (2000) exergy formulations for
internal combustion engines under altitude-dependent atmospheric dead states ($P_0, T_0, \\rho_0$).
"""

from __future__ import annotations

from ice_cfd_validator.exergy.dead_state import (
    DIESEL_EN590_EXERGY_FACTOR,
    HYDROGEN_EXERGY_FACTOR,
    JET_A1_EXERGY_FACTOR,
    DeadStateConditions,
    FuelChemicalExergy,
    compute_fuel_chemical_exergy,
)
from ice_cfd_validator.exergy.second_law import (
    ExergyCycleBudget,
    ExergyRatesResult,
    compute_exergy_cycle_budget,
    compute_exergy_rates,
    compute_standard_entropy,
    compute_thermomechanical_exergy,
)

__all__ = [
    "DIESEL_EN590_EXERGY_FACTOR",
    "DeadStateConditions",
    "ExergyCycleBudget",
    "ExergyRatesResult",
    "FuelChemicalExergy",
    "HYDROGEN_EXERGY_FACTOR",
    "JET_A1_EXERGY_FACTOR",
    "compute_exergy_cycle_budget",
    "compute_exergy_rates",
    "compute_fuel_chemical_exergy",
    "compute_standard_entropy",
    "compute_thermomechanical_exergy",
]

