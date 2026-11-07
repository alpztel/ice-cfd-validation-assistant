"""Atmospheric dead state determination and fuel chemical exergy modeling.

Provides the reference thermodynamic dead state $(P_0, T_0, \\rho_0)$ as a function of
geopotential altitude via the ISO 2533 International Standard Atmosphere (ISA),
and evaluates fuel specific chemical exergy $e_{f,\\text{chem}} = \\varphi \\cdot LHV$
via Kotas (1995) and Rodriguez (1980) correlations for liquid and gaseous fuels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np

from ice_cfd_validator.constants import compute_isa_conditions
from ice_cfd_validator.exceptions import PhysicalBoundsError

# Standard specific chemical exergy factors (phi) from authoritative literature
DIESEL_EN590_EXERGY_FACTOR: Final[float] = 1.066
JET_A1_EXERGY_FACTOR: Final[float] = 1.066
HYDROGEN_EXERGY_FACTOR: Final[float] = 0.985

# Standard IUPAC atomic molar masses in g/mol (dimensionless when computing ratios)
_M_H: Final[float] = 1.00794
_M_C: Final[float] = 12.0107
_M_O: Final[float] = 15.9994
_M_S: Final[float] = 32.065

PredefinedFuel = Literal["diesel", "jet-a1", "hydrogen", "h2"]


@dataclass(frozen=True, slots=True)
class DeadStateConditions:
    """Atmospheric reference dead state conditions $(P_0, T_0, \\rho_0)$ at altitude.

    Parameters
    ----------
    altitude_m : float
        Geopotential altitude above mean sea level $z$ [$\\text{m}$].
    p0_pa : float
        Atmospheric dead state reference static pressure $P_0$ [$\\text{Pa}$].
    t0_k : float
        Atmospheric dead state reference static temperature $T_0$ [$\\text{K}$].
    rho0_kg_per_m3 : float
        Atmospheric dead state reference ambient air density $\\rho_0$ [$\\text{kg/m}^3$].
    """

    altitude_m: float
    p0_pa: float
    t0_k: float
    rho0_kg_per_m3: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.altitude_m <= 15000.0):
            raise PhysicalBoundsError(
                f"Altitude {self.altitude_m} m exceeds valid ISA tropospheric domain [0, 15000] m."
            )
        if self.p0_pa <= 0.0 or self.t0_k <= 0.0 or self.rho0_kg_per_m3 <= 0.0:
            raise PhysicalBoundsError("Dead state physical properties (P0, T0, rho0) must be strictly positive.")

    @classmethod
    def from_altitude(cls, altitude_m: float) -> DeadStateConditions:
        """Construct reference dead state from ISA atmosphere model at specified altitude.

        Parameters
        ----------
        altitude_m : float
            Geopotential altitude above mean sea level in meters ($0 \\le z \\le 15,000\\,\\text{m}$).

        Returns
        -------
        DeadStateConditions
            Populated immutable dead state object with $(P_0, T_0, \\rho_0)$ in SI units.
        """
        alt_arr = np.asarray([float(altitude_m)], dtype=np.float64)
        isa = compute_isa_conditions(alt_arr)
        return cls(
            altitude_m=float(altitude_m),
            p0_pa=float(isa.pressure_pa[0]),
            t0_k=float(isa.temperature_k[0]),
            rho0_kg_per_m3=float(isa.density_kg_m3[0]),
        )


@dataclass(frozen=True, slots=True)
class FuelChemicalExergy:
    """Specific chemical exergy and availability factor for a hydrocarbon or alternative fuel.

    Parameters
    ----------
    lhv_j_per_kg : float
        Lower heating value $LHV$ [$\\text{J/kg}$].
    phi_factor : float
        Chemical exergy-to-LHV ratio $\\varphi = e_{f,\\text{chem}} / LHV$ [-].
    e_chem_j_per_kg : float
        Specific chemical exergy $e_{f,\\text{chem}} = \\varphi \\cdot LHV$ [$\\text{J/kg}$].
    """

    lhv_j_per_kg: float
    phi_factor: float
    e_chem_j_per_kg: float


def compute_fuel_chemical_exergy(
    lhv_j_per_kg: float,
    *,
    h_c_mass: float | None = None,
    o_c_mass: float = 0.0,
    s_c_mass: float = 0.0,
    h_c_atomic: float | None = None,
    o_c_atomic: float = 0.0,
    s_c_atomic: float = 0.0,
    fuel_type: PredefinedFuel | str | None = None,
) -> FuelChemicalExergy:
    """Compute specific chemical exergy $e_{f,\\text{chem}} = \\varphi \\cdot LHV$ [$\\text{J/kg}$].

    Applies the Kotas (1995) and Rodriguez (1980) correlation for liquid hydrocarbon
    fuels of general composition $\\text{C}_x\\text{H}_y\\text{O}_z\\text{S}_w$:

    $$\\varphi_{\\text{liquid}} = 1.0401 + 0.1728 \\frac{h}{c} + 0.0432 \\frac{o}{c}
      + 0.2169 \\frac{s}{c} \\left( 1 - 2.0628 \\frac{h}{c} \\right)$$

    where $h/c, o/c, s/c$ are mass ratios of hydrogen, oxygen, and sulfur to carbon.
    Both mass ratios and atomic ratios ($H/C, O/C, S/C$) are supported.

    Parameters
    ----------
    lhv_j_per_kg : float
        Lower heating value in Joules per kilogram ($LHV > 0$).
    h_c_mass : float | None, default=None
        Mass ratio $h/c = (y M_H) / (x M_C)$ [-].
    o_c_mass : float, default=0.0
        Mass ratio $o/c = (z M_O) / (x M_C)$ [-].
    s_c_mass : float, default=0.0
        Mass ratio $s/c = (w M_S) / (x M_C)$ [-].
    h_c_atomic : float | None, default=None
        Atomic ratio $H/C = y/x$ [-]. Converted to mass ratio via $M_H/M_C$.
    o_c_atomic : float, default=0.0
        Atomic ratio $O/C = z/x$ [-].
    s_c_atomic : float, default=0.0
        Atomic ratio $S/C = w/x$ [-].
    fuel_type : str | None, default=None
        Predefined fuel shorthand: 'diesel', 'jet-a1', 'hydrogen', or 'h2'.

    Returns
    -------
    FuelChemicalExergy
        Populated immutable chemical exergy dataclass with $(LHV, \\varphi, e_{f,\\text{chem}})$.
    """
    if lhv_j_per_kg <= 0.0:
        raise PhysicalBoundsError(f"Fuel lower heating value LHV must be strictly positive (> 0), got {lhv_j_per_kg}.")

    # Predefined fuel shortcuts
    if fuel_type is not None:
        normalized_name = fuel_type.strip().lower()
        if normalized_name in ("diesel", "diesel_en590", "en590"):
            phi = DIESEL_EN590_EXERGY_FACTOR
            return FuelChemicalExergy(lhv_j_per_kg=float(lhv_j_per_kg), phi_factor=phi, e_chem_j_per_kg=phi * lhv_j_per_kg)
        if normalized_name in ("jet-a1", "jeta1", "jp8", "f34"):
            phi = JET_A1_EXERGY_FACTOR
            return FuelChemicalExergy(lhv_j_per_kg=float(lhv_j_per_kg), phi_factor=phi, e_chem_j_per_kg=phi * lhv_j_per_kg)
        if normalized_name in ("hydrogen", "h2"):
            phi = HYDROGEN_EXERGY_FACTOR
            return FuelChemicalExergy(lhv_j_per_kg=float(lhv_j_per_kg), phi_factor=phi, e_chem_j_per_kg=phi * lhv_j_per_kg)

    # Resolve mass ratios
    if h_c_mass is not None:
        hc_m = float(h_c_mass)
        oc_m = float(o_c_mass)
        sc_m = float(s_c_mass)
    elif h_c_atomic is not None:
        hc_m = float(h_c_atomic) * (_M_H / _M_C)
        oc_m = float(o_c_atomic) * (_M_O / _M_C)
        sc_m = float(s_c_atomic) * (_M_S / _M_C)
    else:
        # Default fallback to standard Aero-Diesel / Jet-A1 (H/C = 1.85)
        hc_m = 1.85 * (_M_H / _M_C)
        oc_m = 0.0
        sc_m = 0.0

    if hc_m < 0.0 or oc_m < 0.0 or sc_m < 0.0:
        raise PhysicalBoundsError("Fuel composition ratios must be non-negative.")

    # Kotas / Rodriguez correlation
    phi = 1.0401 + 0.1728 * hc_m + 0.0432 * oc_m + 0.2169 * sc_m * (1.0 - 2.0628 * hc_m)
    e_chem = phi * float(lhv_j_per_kg)

    return FuelChemicalExergy(
        lhv_j_per_kg=float(lhv_j_per_kg),
        phi_factor=float(phi),
        e_chem_j_per_kg=float(e_chem),
    )
