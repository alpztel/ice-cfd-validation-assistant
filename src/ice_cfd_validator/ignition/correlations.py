"""Fuel auto-ignition properties and empirical ignition-delay correlations.

Implements, in strict SI units and fully vectorized form:

- Cetane-number-dependent apparent activation energy and the global
  Hardenberg & Hase (1979, SAE Paper 790493) ignition-delay correlation as
  reported by Heywood (2018, Eq. 10.37), including an explicit high-altitude
  low-pressure singularity guard.
- Instantaneous-state Arrhenius ignition-delay correlations of Wolfer (1938)
  and Assanis et al. (2003, SAE Paper 2003-01-1039) with optional cetane-number
  scaling of the activation energy.

References
----------
.. [1] Hardenberg, H. O., & Hase, F. W. (1979). An Empirical Formula for
   Computing the Pressure Rise Delay of a Fuel from Its Cetane Number and from
   the Relevant Parameters of Direct-Injection Diesel Engines. SAE Technical
   Paper 790493. https://doi.org/10.4271/790493
.. [2] Heywood, J. B. (2018). Internal Combustion Engine Fundamentals (2nd ed.).
   McGraw-Hill Education.
.. [3] Wolfer, H. H. (1938). Der Zündverzug im Dieselmotor. VDI-Forschungsheft 392.
.. [4] Assanis, D. N., Filipi, Z. S., Fiveland, S. B., & Syrimis, M. (2003). A
   Predictive Ignition Delay Correlation Under Steady-State and Transient
   Operation of a Direct Injection Diesel Engine. Journal of Engineering for
   Gas Turbines and Power, 125(2), 450-457 (SAE Paper 2003-01-1039).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ice_cfd_validator.constants import R_UNIVERSAL
from ice_cfd_validator.exceptions import PhysicalBoundsError

FloatArray: TypeAlias = NDArray[np.float64]
PressureCoupling: TypeAlias = Literal["multiplicative", "additive"]

HH_ACTIVATION_NUMERATOR_J_PER_MOL: Final[float] = 618_840.0
"""Hardenberg-Hase activation-energy numerator $E_a (CN + 25)$ [J/mol]."""

HH_CETANE_OFFSET: Final[float] = 25.0
"""Hardenberg-Hase cetane-number offset in $E_a = 618840 / (CN + 25)$ [-]."""

HH_REFERENCE_RT_J_PER_MOL: Final[float] = 17_190.0
"""Hardenberg-Hase reference molar energy $R_u T_{\\text{ref}}$ [J/mol]."""

HH_PRESSURE_THRESHOLD_BAR: Final[float] = 12.4
"""Hardenberg-Hase pressure singularity $P_{TDC} = 12.4$ bar (1.24 MPa)."""

HH_PRESSURE_THRESHOLD_PA: Final[float] = HH_PRESSURE_THRESHOLD_BAR * 1.0e5
"""Hardenberg-Hase pressure singularity expressed in SI units [Pa]."""

CETANE_REFERENCE: Final[float] = 50.0
"""Reference cetane number for Arrhenius activation-energy scaling [-]."""

HH_MISFIRE_MESSAGE: Final[str] = (
    "P_TDC lies below the Hardenberg-Hase 1.24 MPa physical auto-ignition "
    "threshold (high-altitude misfire regime)."
)


@dataclass(frozen=True, slots=True)
class FuelIgnitionProps:
    """Auto-ignition and evaporation properties of a compression-ignition fuel.

    Attributes
    ----------
    name : str
        Fuel identifier.
    cetane_number : float
        Cetane number $CN$ [-].
    stoichiometric_afr : float
        Stoichiometric air-fuel mass ratio $AFR_{\\text{stoich}}$ [-].
    latent_heat_evap_j_per_kg : float
        Latent heat of vaporization $h_{fg}$ [J/kg].
    arrhenius_a_ms : float
        Arrhenius pre-exponential factor $A$ [ms] (pressure in bar).
    pressure_exponent_n : float
        Pressure exponent $n$ in $\\tau \\propto P^{-n}$ [-].
    equiv_ratio_exponent_k : float
        Equivalence-ratio exponent $k$ in $\\tau \\propto \\phi^{-k}$ [-].
    """

    name: str
    cetane_number: float
    stoichiometric_afr: float
    latent_heat_evap_j_per_kg: float
    arrhenius_a_ms: float
    pressure_exponent_n: float
    equiv_ratio_exponent_k: float


@dataclass(frozen=True, slots=True)
class ArrheniusDelayCoefficients:
    """Coefficient set of an instantaneous-state Arrhenius ignition-delay law.

    The law reads
    $\\tau = 10^{-3} A \\phi^{-k} (P / 10^5)^{-n} \\exp[E_a / (R_u T)]$ [s].

    Attributes
    ----------
    name : str
        Literature identifier.
    a_ms : float
        Pre-exponential factor $A$ [ms].
    pressure_exponent_n : float
        Pressure exponent $n$ [-].
    equiv_ratio_exponent_k : float
        Equivalence-ratio exponent $k$ [-].
    activation_temperature_k : float
        Reference activation temperature $E_{a,\\text{ref}} / R_u$ [K] at $CN = 50$.
    """

    name: str
    a_ms: float
    pressure_exponent_n: float
    equiv_ratio_exponent_k: float
    activation_temperature_k: float

    @property
    def activation_energy_j_per_mol(self) -> float:
        """Reference apparent activation energy $E_{a,\\text{ref}}$ [J/mol]."""
        return self.activation_temperature_k * R_UNIVERSAL


@dataclass(frozen=True, slots=True)
class HardenbergHaseResult:
    """Hardenberg & Hase (1979) global ignition-delay prediction.

    Attributes
    ----------
    ignition_delay_deg : NDArray[np.float64]
        Ignition delay in crank angle $\\Delta\\theta_{id}$ [deg].
    ignition_delay_s : NDArray[np.float64]
        Ignition delay in time $\\tau_{id} = \\Delta\\theta_{id} / (6\\,\\text{RPM})$ [s].
    activation_energy_j_per_mol : float
        Apparent activation energy $E_a = 618840 / (CN + 25)$ [J/mol].
    """

    ignition_delay_deg: FloatArray
    ignition_delay_s: FloatArray
    activation_energy_j_per_mol: float


WOLFER_1938: Final[ArrheniusDelayCoefficients] = ArrheniusDelayCoefficients(
    name="Wolfer (1938)",
    a_ms=0.44,
    pressure_exponent_n=1.19,
    equiv_ratio_exponent_k=0.0,
    activation_temperature_k=4650.0,
)
"""Wolfer (1938): $\\tau = 0.44\\,P^{-1.19} \\exp(4650/T)$ [ms], $P$ in bar."""

ASSANIS_2003: Final[ArrheniusDelayCoefficients] = ArrheniusDelayCoefficients(
    name="Assanis et al. (2003)",
    a_ms=2.4,
    pressure_exponent_n=1.02,
    equiv_ratio_exponent_k=0.2,
    activation_temperature_k=2100.0,
)
"""Assanis et al. (2003): $\\tau = 2.4\\,\\phi^{-0.2} P^{-1.02} \\exp(2100/T)$ [ms]."""

PREDEFINED_FUELS: Final[dict[str, FuelIgnitionProps]] = {
    "EN590_DIESEL": FuelIgnitionProps(
        name="EN590_DIESEL",
        cetane_number=52.0,
        stoichiometric_afr=14.5,
        latent_heat_evap_j_per_kg=250_000.0,
        arrhenius_a_ms=ASSANIS_2003.a_ms,
        pressure_exponent_n=ASSANIS_2003.pressure_exponent_n,
        equiv_ratio_exponent_k=ASSANIS_2003.equiv_ratio_exponent_k,
    ),
    "JET_A1_JP8": FuelIgnitionProps(
        name="JET_A1_JP8",
        cetane_number=42.0,
        stoichiometric_afr=14.7,
        latent_heat_evap_j_per_kg=300_000.0,
        arrhenius_a_ms=ASSANIS_2003.a_ms,
        pressure_exponent_n=ASSANIS_2003.pressure_exponent_n,
        equiv_ratio_exponent_k=ASSANIS_2003.equiv_ratio_exponent_k,
    ),
    "SYNTHETIC_FT_SPK": FuelIgnitionProps(
        name="SYNTHETIC_FT_SPK",
        cetane_number=58.0,
        stoichiometric_afr=14.9,
        latent_heat_evap_j_per_kg=270_000.0,
        arrhenius_a_ms=ASSANIS_2003.a_ms,
        pressure_exponent_n=ASSANIS_2003.pressure_exponent_n,
        equiv_ratio_exponent_k=ASSANIS_2003.equiv_ratio_exponent_k,
    ),
}
"""Standard Aero-Diesel fuel profiles (EN590 Diesel, Jet-A1/JP-8/F-34, FT-SPK)."""


def compute_apparent_activation_energy(cetane_number: float) -> float:
    """Compute the Hardenberg & Hase (1979) apparent activation energy.

    $$E_a = \\frac{618840}{CN + 25} \\quad [\\text{J/mol}]$$

    Parameters
    ----------
    cetane_number : float
        Fuel cetane number $CN$ [-]; must be strictly positive.

    Returns
    -------
    float
        Apparent activation energy $E_a$ [J/mol].

    Raises
    ------
    PhysicalBoundsError
        If ``cetane_number`` is non-finite or not strictly positive.
    """
    cn = float(cetane_number)
    if not np.isfinite(cn) or cn <= 0.0:
        raise PhysicalBoundsError(f"Cetane number must be finite and > 0, got {cn}.")
    return HH_ACTIVATION_NUMERATOR_J_PER_MOL / (cn + HH_CETANE_OFFSET)


def hardenberg_hase_delay(
    temperature_tdc_k: ArrayLike,
    pressure_tdc_pa: ArrayLike,
    mean_piston_speed_m_s: ArrayLike,
    rpm: float,
    cetane_number: float,
    *,
    pressure_coupling: PressureCoupling = "additive",
) -> HardenbergHaseResult:
    """Evaluate the Hardenberg & Hase (1979) global ignition delay.

    Following Heywood (2018, Eq. 10.37) and the project specification, with
    $P_{TDC}$ supplied in SI units and converted internally to bar
    ($P_{\\text{bar}} = 10^{-5} P_{TDC}$):

    $$\\Delta\\theta_{id} = (0.36 + 0.22\\,\\bar{S}_p)
      \\exp\\left[E_a\\left(\\frac{1}{R_u T_{TDC}} - \\frac{1}{17190}\\right)
      + \\left(\\frac{21.2}{P_{\\text{bar}} - 12.4}\\right)^{0.63}\\right] \\quad [\\text{deg}]$$

    $$\\tau_{id} = \\frac{\\Delta\\theta_{id}}{6\\,\\text{RPM}} \\quad [\\text{s}]$$

    where $17190$ J/mol is the original reference molar energy $R_u T_{\\text{ref}}$ and
    $R_u = 8.314462618$ J/(mol K). A multiplicative coupling
    (``pressure_coupling="multiplicative"``) is also supported:
    $\\exp\\left[E_a(1/(R_u T) - 1/17190) \\cdot (21.2/(P-12.4))^{0.63}\\right]$.

    Parameters
    ----------
    temperature_tdc_k : ArrayLike
        Charge temperature during the delay period $T_{TDC}$ [K].
    pressure_tdc_pa : ArrayLike
        Charge pressure during the delay period $P_{TDC}$ [Pa].
    mean_piston_speed_m_s : ArrayLike
        Mean piston speed $\\bar{S}_p$ [m/s].
    rpm : float
        Engine speed [rev/min].
    cetane_number : float
        Fuel cetane number $CN$ [-].
    pressure_coupling : {"additive", "multiplicative"}, optional
        Coupling between the thermal activation and pressure term in the
        exponent (default ``"additive"`` per project specification).

    Returns
    -------
    HardenbergHaseResult
        Delay in crank-angle degrees and in seconds (broadcast input shape).

    Raises
    ------
    ValueError
        If any $P_{TDC} \\le 1.24$ MPa, where the pressure term is singular or
        complex (high-altitude misfire regime).
    PhysicalBoundsError
        For non-positive temperature, speed, or negative piston speed.
    """
    t_k = np.asarray(temperature_tdc_k, dtype=np.float64)
    p_pa = np.asarray(pressure_tdc_pa, dtype=np.float64)
    sp = np.asarray(mean_piston_speed_m_s, dtype=np.float64)
    rpm_f = float(rpm)

    if not np.all(np.isfinite(t_k)) or np.any(t_k <= 0.0):
        raise PhysicalBoundsError("T_TDC must be finite and strictly positive [K].")
    if not np.all(np.isfinite(p_pa)):
        raise PhysicalBoundsError("P_TDC must be finite [Pa].")
    if not np.all(np.isfinite(sp)) or np.any(sp < 0.0):
        raise PhysicalBoundsError("Mean piston speed must be finite and >= 0 [m/s].")
    if not np.isfinite(rpm_f) or rpm_f <= 0.0:
        raise PhysicalBoundsError(f"Engine speed must be finite and > 0 rpm, got {rpm_f}.")

    p_bar = p_pa * 1.0e-5
    if np.any(p_bar <= HH_PRESSURE_THRESHOLD_BAR):
        raise ValueError(HH_MISFIRE_MESSAGE)

    e_a = compute_apparent_activation_energy(cetane_number)
    thermal_term = e_a * (1.0 / (R_UNIVERSAL * t_k) - 1.0 / HH_REFERENCE_RT_J_PER_MOL)
    pressure_term = (21.2 / (p_bar - HH_PRESSURE_THRESHOLD_BAR)) ** 0.63

    if pressure_coupling == "additive":
        exponent = thermal_term + pressure_term
    else:
        exponent = thermal_term * pressure_term

    delay_deg = (0.36 + 0.22 * sp) * np.exp(exponent)
    delay_s = delay_deg / (6.0 * rpm_f)

    return HardenbergHaseResult(
        ignition_delay_deg=np.asarray(delay_deg, dtype=np.float64),
        ignition_delay_s=np.asarray(delay_s, dtype=np.float64),
        activation_energy_j_per_mol=e_a,
    )


def coefficients_from_fuel(
    fuel: FuelIgnitionProps,
    reference: ArrheniusDelayCoefficients = ASSANIS_2003,
) -> ArrheniusDelayCoefficients:
    """Build an Arrhenius coefficient set from a fuel's $A$, $n$, $k$ values.

    Parameters
    ----------
    fuel : FuelIgnitionProps
        Fuel providing ``arrhenius_a_ms``, ``pressure_exponent_n`` and
        ``equiv_ratio_exponent_k``.
    reference : ArrheniusDelayCoefficients, optional
        Correlation supplying the reference activation temperature at $CN = 50$.

    Returns
    -------
    ArrheniusDelayCoefficients
        Combined coefficient set (use with ``cetane_number=fuel.cetane_number``).
    """
    return ArrheniusDelayCoefficients(
        name=f"{fuel.name} / {reference.name}",
        a_ms=fuel.arrhenius_a_ms,
        pressure_exponent_n=fuel.pressure_exponent_n,
        equiv_ratio_exponent_k=fuel.equiv_ratio_exponent_k,
        activation_temperature_k=reference.activation_temperature_k,
    )


def scale_activation_energy(
    reference_activation_energy_j_per_mol: float,
    cetane_number: float,
) -> float:
    """Scale an activation energy from $CN_{\\text{ref}} = 50$ to an arbitrary fuel.

    $$E_a(CN) = E_{a,\\text{ref}} \\, \\frac{50 + 25}{CN + 25}$$

    Parameters
    ----------
    reference_activation_energy_j_per_mol : float
        Activation energy at the reference cetane number [J/mol].
    cetane_number : float
        Target fuel cetane number [-].

    Returns
    -------
    float
        Scaled activation energy [J/mol].
    """
    ratio = compute_apparent_activation_energy(cetane_number) / compute_apparent_activation_energy(
        CETANE_REFERENCE
    )
    return float(reference_activation_energy_j_per_mol) * ratio


def arrhenius_state_delay(
    pressure_pa: ArrayLike,
    temperature_k: ArrayLike,
    equivalence_ratio: ArrayLike = 1.0,
    *,
    coefficients: ArrheniusDelayCoefficients = ASSANIS_2003,
    cetane_number: float | None = None,
) -> FloatArray:
    """Evaluate the local instantaneous-state Arrhenius ignition delay.

    $$\\tau(\\theta) = 10^{-3} A\\,\\phi^{-k} \\left(\\frac{P(\\theta)}{10^5}\\right)^{-n}
      \\exp\\left(\\frac{E_a(CN)}{R_u T(\\theta)}\\right) \\quad [\\text{s}]$$

    Parameters
    ----------
    pressure_pa : ArrayLike
        Cylinder pressure $P(\\theta)$ [Pa].
    temperature_k : ArrayLike
        Cylinder (charge) temperature $T(\\theta)$ [K].
    equivalence_ratio : ArrayLike, optional
        Fuel-air equivalence ratio $\\phi$ [-] (default 1.0); must be > 0.
    coefficients : ArrheniusDelayCoefficients, optional
        ``WOLFER_1938`` or ``ASSANIS_2003`` (default) or a custom set.
    cetane_number : float | None, optional
        If given, scales $E_a$ via $E_{a,\\text{ref}} (50 + 25) / (CN + 25)$.

    Returns
    -------
    NDArray[np.float64]
        Local ignition delay $\\tau$ [s] with the broadcast input shape.

    Raises
    ------
    PhysicalBoundsError
        For non-positive or non-finite pressure, temperature, or equivalence ratio.
    """
    p_pa = np.asarray(pressure_pa, dtype=np.float64)
    t_k = np.asarray(temperature_k, dtype=np.float64)
    phi = np.asarray(equivalence_ratio, dtype=np.float64)

    if not np.all(np.isfinite(p_pa)) or np.any(p_pa <= 0.0):
        raise PhysicalBoundsError("Pressure must be finite and strictly positive [Pa].")
    if not np.all(np.isfinite(t_k)) or np.any(t_k <= 0.0):
        raise PhysicalBoundsError("Temperature must be finite and strictly positive [K].")
    if not np.all(np.isfinite(phi)) or np.any(phi <= 0.0):
        raise PhysicalBoundsError("Equivalence ratio must be finite and strictly positive [-].")

    e_a = coefficients.activation_energy_j_per_mol
    if cetane_number is not None:
        e_a = scale_activation_energy(e_a, cetane_number)

    tau_s = (
        1.0e-3
        * coefficients.a_ms
        * phi ** (-coefficients.equiv_ratio_exponent_k)
        * (p_pa * 1.0e-5) ** (-coefficients.pressure_exponent_n)
        * np.exp(e_a / (R_UNIVERSAL * t_k))
    )
    return np.asarray(tau_s, dtype=np.float64)

