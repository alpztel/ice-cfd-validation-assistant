"""Fundamental physical constants, ISO 2533 ISA atmosphere, and strict SI unit factors.

All conversion tables map normalized ANSYS Forte header unit tokens to exact
multiplicative (or affine for temperature) SI scaling factors following NIST
SP 811 (Thompson & Taylor, 2008). International Standard Atmosphere (ISA)
relations follow ISO 2533:1975 from sea level up to 20,000 m.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ice_cfd_validator.exceptions import PhysicalBoundsError

FloatArray: TypeAlias = NDArray[np.float64]

# Fundamental thermodynamic and standard reference constants (CODATA 2018 / ISO 2533)
R_UNIVERSAL: Final[float] = 8.314462618               # Universal gas constant [J/(mol*K)]
MOLAR_MASS_DRY_AIR: Final[float] = 0.0289647          # CIPM-2007 dry air molar mass [kg/mol]
R_SPECIFIC_DRY_AIR: Final[float] = R_UNIVERSAL / MOLAR_MASS_DRY_AIR  # ~287.055 [J/(kg*K)]
STD_GRAVITY_M_S2: Final[float] = 9.80665              # Standard geopotential gravity [m/s^2]
STD_ATM_PA: Final[float] = 101_325.0                  # Standard sea-level pressure [Pa]
ZERO_CELSIUS_K: Final[float] = 273.15                 # Ice-point temperature offset [K]
FEET_TO_M: Final[float] = 0.3048                      # Exact international foot [m/ft]

# ISO 2533:1975 International Standard Atmosphere (ISA) layer constants
ISA_T0_K: Final[float] = 288.15                       # Sea-level standard temperature [K]
ISA_P0_PA: Final[float] = STD_ATM_PA                  # Sea-level standard pressure [Pa]
ISA_RHO0_KG_M3: Final[float] = ISA_P0_PA / (R_SPECIFIC_DRY_AIR * ISA_T0_K)  # ~1.225 [kg/m^3]
ISA_LAPSE_RATE_K_PER_M: Final[float] = 0.0065         # Tropospheric temperature lapse rate [K/m]
ISA_TROPOPAUSE_ALT_M: Final[float] = 11_000.0         # Tropopause geopotential altitude [m]
ISA_STRATOSPHERE_TOP_M: Final[float] = 20_000.0       # Upper bound of lower stratosphere [m]
ISA_T_TROPOPAUSE_K: Final[float] = (
    ISA_T0_K - ISA_LAPSE_RATE_K_PER_M * ISA_TROPOPAUSE_ALT_M
)                                                     # 216.65 [K]
ISA_GAMMA_DRY_AIR: Final[float] = 1.4                 # Reference diatomic gamma for sound speed [-]

PRESSURE_TO_PA: Final[dict[str, float]] = {
    "Pa": 1.0, "pa": 1.0, "PA": 1.0,
    "kPa": 1.0e3, "kpa": 1.0e3, "KPA": 1.0e3,
    "MPa": 1.0e6, "MPA": 1.0e6,
    "mPa": 1.0e-3,
    "bar": 1.0e5, "Bar": 1.0e5, "BAR": 1.0e5,
    "mbar": 1.0e2, "MBAR": 1.0e2,
    "atm": STD_ATM_PA, "ATM": STD_ATM_PA,
    "dyne/cm^2": 0.1, "dyne/cm2": 0.1, "dyn/cm^2": 0.1, "dyn/cm2": 0.1,
}

VOLUME_TO_M3: Final[dict[str, float]] = {
    "m^3": 1.0, "m3": 1.0, "M^3": 1.0, "M3": 1.0,
    "cm^3": 1.0e-6, "cm3": 1.0e-6, "CM^3": 1.0e-6, "CM3": 1.0e-6,
    "cc": 1.0e-6, "CC": 1.0e-6, "mL": 1.0e-6, "ml": 1.0e-6,
    "mm^3": 1.0e-9, "mm3": 1.0e-9, "MM^3": 1.0e-9, "MM3": 1.0e-9,
    "L": 1.0e-3, "l": 1.0e-3, "liter": 1.0e-3, "LITER": 1.0e-3,
}

LENGTH_TO_M: Final[dict[str, float]] = {
    "m": 1.0, "M": 1.0,
    "cm": 1.0e-2, "CM": 1.0e-2,
    "mm": 1.0e-3, "MM": 1.0e-3,
    "um": 1.0e-6, "uM": 1.0e-6, "UM": 1.0e-6,
    "micron": 1.0e-6, "microns": 1.0e-6, "MICRON": 1.0e-6,
    "ft": FEET_TO_M, "FT": FEET_TO_M,
}

MASS_TO_KG: Final[dict[str, float]] = {
    "kg": 1.0, "KG": 1.0, "Kg": 1.0,
    "g": 1.0e-3, "G": 1.0e-3,
    "mg": 1.0e-6, "MG": 1.0e-6,
    "ug": 1.0e-9, "UG": 1.0e-9,
}

TIME_TO_S: Final[dict[str, float]] = {
    "s": 1.0, "S": 1.0, "sec": 1.0, "SEC": 1.0,
    "ms": 1.0e-3, "MS": 1.0e-3, "msec": 1.0e-3, "MSEC": 1.0e-3,
    "us": 1.0e-6, "US": 1.0e-6, "usec": 1.0e-6, "USEC": 1.0e-6,
}

ANGLE_TO_DEG: Final[dict[str, float]] = {
    "deg": 1.0, "DEG": 1.0, "CA": 1.0, "ca": 1.0,
    "CAD": 1.0, "cad": 1.0, "degCA": 1.0, "DEGCA": 1.0,
    "ATDC": 1.0, "degATDC": 1.0,
    "rad": 180.0 / math.pi, "RAD": 180.0 / math.pi,
}

TEMP_TO_K: Final[dict[str, tuple[float, float]]] = {
    "K": (1.0, 0.0), "k": (1.0, 0.0), "kelvin": (1.0, 0.0), "KELVIN": (1.0, 0.0),
    "C": (1.0, ZERO_CELSIUS_K), "c": (1.0, ZERO_CELSIUS_K),
    "degC": (1.0, ZERO_CELSIUS_K), "DEGC": (1.0, ZERO_CELSIUS_K),
    "celsius": (1.0, ZERO_CELSIUS_K), "CELSIUS": (1.0, ZERO_CELSIUS_K),
}

HEAT_RATE_TO_J_PER_DEG: Final[dict[str, float]] = {
    "J/deg": 1.0, "J/DEG": 1.0, "J/CA": 1.0, "J/ca": 1.0,
    "J/CAD": 1.0, "J/cad": 1.0, "J/degCA": 1.0,
    "kJ/deg": 1.0e3, "kJ/CA": 1.0e3, "kJ/CAD": 1.0e3,
    "mJ/deg": 1.0e-3, "mJ/CA": 1.0e-3, "mJ/CAD": 1.0e-3,
    "erg/deg": 1.0e-7, "erg/CA": 1.0e-7,
    "J/rad": math.pi / 180.0, "J/RAD": math.pi / 180.0,
}


@dataclass(frozen=True, slots=True)
class ISAConditions:
    """Vectorized ISO 2533 International Standard Atmosphere state in strict SI units.

    Attributes
    ----------
    altitude_m : NDArray[np.float64]
        Geopotential altitude $z$ [m].
    temperature_k : NDArray[np.float64]
        Ambient static temperature $T_{\\text{amb}}(z)$ [K].
    pressure_pa : NDArray[np.float64]
        Ambient static pressure $P_{\\text{amb}}(z)$ [Pa].
    density_kg_m3 : NDArray[np.float64]
        Ambient air density $\\rho_{\\text{amb}}(z)$ [$\\text{kg/m}^3$].
    speed_of_sound_m_s : NDArray[np.float64]
        Local acoustic speed $a(z) = \\sqrt{\\gamma R_{\\text{air}} T_{\\text{amb}}(z)}$ [m/s].
    """

    altitude_m: FloatArray
    temperature_k: FloatArray
    pressure_pa: FloatArray
    density_kg_m3: FloatArray
    speed_of_sound_m_s: FloatArray


def compute_isa_conditions(altitude_m: ArrayLike) -> ISAConditions:
    """Compute vectorized ISO 2533 ISA ambient state from sea level up to 20,000 m.

    Covers both the linear-lapse troposphere ($0 \\le z \\le 11,000\\ \\text{m}$,
    $L = 6.5\\ \\text{K/km}$) and the isothermal lower stratosphere
    ($11,000 < z \\le 20,000\\ \\text{m}$, $T = 216.65\\ \\text{K}$), exceeding the
    $15,000\\ \\text{m}$ (~$49,213\\ \\text{ft}$) Aero-Diesel flight envelope.

    Parameters
    ----------
    altitude_m : ArrayLike
        Scalar or array of geopotential altitudes $z$ [m] within ``[-500, 20000]`` m.

    Returns
    -------
    ISAConditions
        Vectorized SI ambient state (``temperature_k``, ``pressure_pa``,
        ``density_kg_m3``, ``speed_of_sound_m_s``).
    """
    z = np.atleast_1d(np.asarray(altitude_m, dtype=np.float64))
    if not np.all(np.isfinite(z)) or np.any((z < -500.0) | (z > ISA_STRATOSPHERE_TOP_M)):
        raise PhysicalBoundsError(
            f"ISA altitude_m must be finite and within [-500.0, {ISA_STRATOSPHERE_TOP_M:.1f}] m."
        )

    barometric_exp = STD_GRAVITY_M_S2 / (R_SPECIFIC_DRY_AIR * ISA_LAPSE_RATE_K_PER_M)
    p_tropopause_pa = ISA_P0_PA * ((ISA_T_TROPOPAUSE_K / ISA_T0_K) ** barometric_exp)

    # Clamp branch inputs before np.where so unused branches never see negative temperatures
    z_tropo = np.minimum(z, ISA_TROPOPAUSE_ALT_M)
    z_strato = np.maximum(z, ISA_TROPOPAUSE_ALT_M)

    t_tropo = ISA_T0_K - ISA_LAPSE_RATE_K_PER_M * z_tropo
    p_tropo = ISA_P0_PA * ((t_tropo / ISA_T0_K) ** barometric_exp)

    p_strato = p_tropopause_pa * np.exp(
        -STD_GRAVITY_M_S2 * (z_strato - ISA_TROPOPAUSE_ALT_M)
        / (R_SPECIFIC_DRY_AIR * ISA_T_TROPOPAUSE_K)
    )

    in_stratosphere = z > ISA_TROPOPAUSE_ALT_M
    temp_k = np.asarray(np.where(in_stratosphere, ISA_T_TROPOPAUSE_K, t_tropo), dtype=np.float64)
    press_pa = np.asarray(np.where(in_stratosphere, p_strato, p_tropo), dtype=np.float64)
    rho_kg_m3 = np.asarray(press_pa / (R_SPECIFIC_DRY_AIR * temp_k), dtype=np.float64)
    sound_m_s = np.asarray(
        np.sqrt(ISA_GAMMA_DRY_AIR * R_SPECIFIC_DRY_AIR * temp_k),
        dtype=np.float64,
    )

    return ISAConditions(
        altitude_m=z,
        temperature_k=temp_k,
        pressure_pa=press_pa,
        density_kg_m3=rho_kg_m3,
        speed_of_sound_m_s=sound_m_s,
    )
