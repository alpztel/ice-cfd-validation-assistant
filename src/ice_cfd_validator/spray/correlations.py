"""High-altitude Aero-Diesel spray penetration, scaling, and wall-impingement models.

Implements vectorized Hiroyasu & Arai (1990, SAE Paper 900475) two-regime tip
penetration, Naber & Siebers (1996, SAE Paper 960034) / Siebers (1999) vapor
dispersion scaling, altitude gas-density wall-wetting risk diagnostics, and
3D CFD vs. experimental error norms in strict SI units.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ice_cfd_validator.exceptions import PhysicalBoundsError

FloatArray = NDArray[np.float64]
SprayModelType = Literal["hiroyasu_arai", "naber_siebers"]
NaberSiebersMethod = Literal["exact", "explicit"]


@dataclass(frozen=True, slots=True)
class HiroyasuAraiResult:
    """Hiroyasu & Arai (1990) two-regime spray tip penetration results in SI units.

    Attributes
    ----------
    time_s : NDArray[np.float64]
        Elapsed time after start of injection $t$ [s].
    penetration_m : NDArray[np.float64]
        Spray tip penetration $S(t)$ [m].
    breakup_time_s : float
        Jet intact-core breakup time $t_b$ [s].
    breakup_length_m : float
        Intact-core breakup length $L_b = S(t_b)$ [m].
    """

    time_s: FloatArray
    penetration_m: FloatArray
    breakup_time_s: float
    breakup_length_m: float


@dataclass(frozen=True, slots=True)
class NaberSiebersResult:
    """Naber & Siebers (1996) / Siebers (1999) non-vaporizing/vapor spray penetration.

    Attributes
    ----------
    time_s : NDArray[np.float64]
        Elapsed time after start of injection $t$ [s].
    penetration_m : NDArray[np.float64]
        Spray tip penetration $S(t) = x^+ \\tilde{S}(\\tilde{t})$ [m].
    t_tilde : NDArray[np.float64]
        Dimensionless time $\\tilde{t} = t / t^+$ [-].
    s_tilde : NDArray[np.float64]
        Dimensionless penetration $\\tilde{S} = S / x^+$ [-].
    x_plus_m : float
        Characteristic momentum length scale $x^+$ [m].
    t_plus_s : float
        Characteristic momentum time scale $t^+ = x^+ / U_f$ [s].
    u_f_m_s : float
        Effective nozzle exit fuel velocity $U_f$ [$\\text{m/s}$].
    """

    time_s: FloatArray
    penetration_m: FloatArray
    t_tilde: FloatArray
    s_tilde: FloatArray
    x_plus_m: float
    t_plus_s: float
    u_f_m_s: float


@dataclass(frozen=True, slots=True)
class AltitudeSprayRiskResult:
    """High-altitude gas-density spray over-penetration and wall-impingement audit.

    Attributes
    ----------
    time_s : NDArray[np.float64]
        Evaluated time array $t$ [s].
    penetration_sea_level_m : NDArray[np.float64]
        Reference sea-level penetration trace $S_1(t)$ at $\\rho_{g,1}$ [m].
    penetration_altitude_m : NDArray[np.float64]
        High-altitude penetration trace $S_2(t)$ at $\\rho_{g,2}$ [m].
    asymptotic_penetration_ratio : float
        Theoretical post-breakup scaling factor $(\\rho_{g,1} / \\rho_{g,2})^{0.25}$ [-].
    terminal_penetration_ratio : float
        Evaluated penetration ratio $S_2(t_{\\text{end}}) / S_1(t_{\\text{end}})$ [-].
    wall_impingement_detected : bool
        ``True`` if $S_2(t)$ reaches or exceeds ``wall_distance_m`` within ``time_s``.
    impingement_time_s : float | None
        Exact sub-step time $t_{\\text{imp}}$ [s] when $S_2(t_{\\text{imp}}) = L_{\\text{wall}}$,
        or ``None`` if no impingement occurs during ``time_s``.
    sea_level_impingement_time_s : float | None
        Exact sub-step time [s] when $S_1(t) = L_{\\text{wall}}$, or ``None``.
    max_penetration_altitude_m : float
        Peak penetration reached at altitude over ``time_s`` [m].
    """

    time_s: FloatArray
    penetration_sea_level_m: FloatArray
    penetration_altitude_m: FloatArray
    asymptotic_penetration_ratio: float
    terminal_penetration_ratio: float
    wall_impingement_detected: bool
    impingement_time_s: float | None
    sea_level_impingement_time_s: float | None
    max_penetration_altitude_m: float


@dataclass(frozen=True, slots=True)
class SprayValidationMetrics:
    """Error norms comparing 3D CFD spray penetration against reference data.

    Attributes
    ----------
    rmse_m : float
        Root-Mean-Square Error [m].
    mape_pct : float
        Mean Absolute Percentage Error [%] over samples exceeding ``mape_floor_m``.
    l2_relative_error : float
        Relative $L_2$ norm $\\|S_{\\text{cfd}} - S_{\\text{ref}}\\|_2 / \\|S_{\\text{ref}}\\|_2$ [-].
    max_abs_error_m : float
        Maximum pointwise absolute deviation $L_\\infty$ [m].
    n_samples : int
        Number of compared time samples [-].
    """

    rmse_m: float
    mape_pct: float
    l2_relative_error: float
    max_abs_error_m: float
    n_samples: int


def _validate_spray_inputs(
    time_s: ArrayLike,
    delta_p_pa: float,
    rho_liquid_kg_m3: float,
    rho_gas_kg_m3: float,
    nozzle_diameter_m: float,
) -> FloatArray:
    t = np.asarray(time_s, dtype=np.float64)
    if not np.all(np.isfinite(t)) or np.any(t < 0.0):
        raise PhysicalBoundsError("Spray time_s must be finite and >= 0 s.")
    if delta_p_pa <= 0.0 or not math.isfinite(delta_p_pa):
        raise PhysicalBoundsError("Injection pressure drop delta_p_pa must be > 0 Pa.")
    if rho_liquid_kg_m3 <= 0.0 or not math.isfinite(rho_liquid_kg_m3):
        raise PhysicalBoundsError("Liquid fuel density rho_liquid_kg_m3 must be > 0 kg/m^3.")
    if rho_gas_kg_m3 <= 0.0 or not math.isfinite(rho_gas_kg_m3):
        raise PhysicalBoundsError("In-cylinder gas density rho_gas_kg_m3 must be > 0 kg/m^3.")
    if nozzle_diameter_m <= 0.0 or not math.isfinite(nozzle_diameter_m):
        raise PhysicalBoundsError("Nozzle orifice diameter nozzle_diameter_m must be > 0 m.")
    return t


def hiroyasu_arai_penetration(
    time_s: ArrayLike,
    *,
    delta_p_pa: float,
    rho_liquid_kg_m3: float,
    rho_gas_kg_m3: float,
    nozzle_diameter_m: float,
    c_breakup: float = 28.65,
    c_post: float = 2.95,
    c_pre: float | None = None,
) -> HiroyasuAraiResult:
    """Compute vectorized Hiroyasu & Arai (1990) spray tip penetration in SI units.

    Evaluates breakup time $t_b$ and piecewise jet/spray tip penetration (SAE 900475):

    $$t_b = 28.65 \\frac{\\rho_l d_o}{\\sqrt{\\rho_g \\Delta P}}$$

    $$S(t) = \\begin{cases}
      c_{\\text{pre}} \\left(\\dfrac{2 \\Delta P}{\\rho_l}\\right)^{0.5} t, & t < t_b \\\\[8pt]
      2.95 \\left(\\dfrac{\\Delta P}{\\rho_g}\\right)^{0.25} (d_o t)^{0.5}, & t \\ge t_b
    \\end{cases}$$

    Note
    ----
    In SAE 900475, Hiroyasu & Arai defined $t_b$ by intersecting the linear and
    square-root regimes and reported rounded 2-significant-figure $c_{\\text{pre}} \\approx 0.39$.
    When ``c_pre=None`` (default), exact analytical intersection continuity
    $c_{\\text{pre}} = c_{\\text{post}} / \\sqrt{2 c_{\\text{breakup}}} \\approx 0.389713$
    is enforced so $|S(t_b^-) - S(t_b^+)| = 0$ to IEEE-754 machine precision while
    preserving exact $t_b = 28.65 \\rho_l d_o / \\sqrt{\\rho_g \\Delta P}$ and post-breakup
    constant $2.95$.

    Parameters
    ----------
    time_s : ArrayLike
        Time after start of injection $t$ [s] ($\\ge 0$).
    delta_p_pa : float
        Effective injection pressure differential $\\Delta P = P_{\\text{inj}} - P_{\\text{cyl}}$ [Pa].
    rho_liquid_kg_m3 : float
        Liquid fuel density $\\rho_l$ [$\\text{kg/m}^3$].
    rho_gas_kg_m3 : float
        Ambient in-cylinder gas density $\\rho_g$ [$\\text{kg/m}^3$].
    nozzle_diameter_m : float
        Injector nozzle hole diameter $d_o$ [m].
    c_breakup : float, default=28.65
        Dimensionless breakup time constant [-].
    c_post : float, default=2.95
        Post-breakup penetration constant [-].
    c_pre : float | None, default=None
        Pre-breakup velocity constant [-]; defaults to $c_{\\text{post}} / \\sqrt{2 c_{\\text{breakup}}}$
        ($\\approx 0.39$) for exact $C^0$ continuity at $t = t_b$.

    Returns
    -------
    HiroyasuAraiResult
        SI penetration curve $S(t)$ [m], breakup time $t_b$ [s], and breakup length $L_b$ [m].
    """
    t = _validate_spray_inputs(
        time_s, delta_p_pa, rho_liquid_kg_m3, rho_gas_kg_m3, nozzle_diameter_m
    )
    if c_breakup <= 0.0 or c_post <= 0.0 or (c_pre is not None and c_pre <= 0.0):
        raise PhysicalBoundsError("Hiroyasu-Arai correlation constants must be > 0.")

    c_pre_eff = c_post / math.sqrt(2.0 * c_breakup) if c_pre is None else c_pre
    t_b = c_breakup * (rho_liquid_kg_m3 * nozzle_diameter_m) / math.sqrt(
        rho_gas_kg_m3 * delta_p_pa
    )

    v_pre = c_pre_eff * math.sqrt(2.0 * delta_p_pa / rho_liquid_kg_m3)
    k_post = c_post * ((delta_p_pa / rho_gas_kg_m3) ** 0.25) * math.sqrt(nozzle_diameter_m)

    s_pre = v_pre * t
    s_post = k_post * np.sqrt(t)
    penetration_m = np.asarray(np.where(t < t_b, s_pre, s_post), dtype=np.float64)
    l_b = k_post * math.sqrt(t_b)

    return HiroyasuAraiResult(
        time_s=t,
        penetration_m=penetration_m,
        breakup_time_s=t_b,
        breakup_length_m=l_b,
    )


def _newton_step_naber_siebers(s_est: FloatArray, t_tilde: FloatArray) -> FloatArray:
    """Single vectorized Newton-Raphson update on Naber & Siebers (1996, Eq. 16)."""
    root_term = np.sqrt(1.0 + 16.0 * (s_est**2))
    t_model = (
        0.5 * s_est
        + 0.25 * s_est * root_term
        + (1.0 / 16.0) * np.log(4.0 * s_est + root_term)
    )
    dt_ds = 0.5 * (1.0 + root_term)
    return np.maximum(s_est - (t_model - t_tilde) / dt_ds, 0.0)


def naber_siebers_penetration(
    time_s: ArrayLike,
    *,
    delta_p_pa: float,
    rho_liquid_kg_m3: float,
    rho_gas_kg_m3: float,
    nozzle_diameter_m: float,
    spray_angle_deg: float,
    dispersion_const_a: float = 0.66,
    ca: float = 1.0,
    cv: float = 1.0,
    method: NaberSiebersMethod = "exact",
) -> NaberSiebersResult:
    """Compute vectorized Naber & Siebers (1996) / Siebers (1999) spray penetration.

    Evaluates characteristic momentum scales (SAE Paper 960034, Eq. 12–21):

    $$x^+ = \\frac{\\sqrt{C_a}\\,d_o \\sqrt{\\rho_l / \\rho_g}}{a \\tan(\\theta_{\\text{spray}} / 2)},
      \\qquad U_f = C_v \\sqrt{\\frac{2 \\Delta P}{\\rho_l}},
      \\qquad t^+ = \\frac{x^+}{U_f}$$

    and resolves dimensionless penetration $\\tilde{S}(\\tilde{t})$ for $\\tilde{t} = t / t^+$.

    Parameters
    ----------
    time_s : ArrayLike
        Time after start of injection $t$ [s] ($\\ge 0$).
    delta_p_pa : float
        Injection pressure differential $\\Delta P$ [Pa].
    rho_liquid_kg_m3 : float
        Fuel density $\\rho_l$ [$\\text{kg/m}^3$].
    rho_gas_kg_m3 : float
        Ambient gas density $\\rho_g$ [$\\text{kg/m}^3$].
    nozzle_diameter_m : float
        Geometric nozzle orifice diameter $d_o$ [m].
    spray_angle_deg : float
        Full spreading cone angle $\\theta_{\\text{spray}}$ [deg] ($0 < \\theta_{\\text{spray}} < 180$).
    dispersion_const_a : float, default=0.66
        Naber & Siebers dispersion constant $a$ [-].
    ca : float, default=1.0
        Orifice area-contraction coefficient $C_a \\in (0, 1]$ [-].
    cv : float, default=1.0
        Orifice velocity coefficient $C_v \\in (0, 1]$ [-].
    method : {"exact", "explicit"}, default="exact"
        ``"exact"`` inverts SAE 960034 Eq. 16 to machine precision via 5 unrolled
        vectorized Newton steps; ``"explicit"`` evaluates Eq. 21 ($n = 2.2$).

    Returns
    -------
    NaberSiebersResult
        SI penetration $S(t)$ [m], dimensionless $(\\tilde{t}, \\tilde{S})$, and scales $(x^+, t^+, U_f)$.
    """
    t = _validate_spray_inputs(
        time_s, delta_p_pa, rho_liquid_kg_m3, rho_gas_kg_m3, nozzle_diameter_m
    )
    if not (0.0 < spray_angle_deg < 180.0):
        raise PhysicalBoundsError("spray_angle_deg must satisfy 0 < spray_angle_deg < 180 deg.")
    if dispersion_const_a <= 0.0:
        raise PhysicalBoundsError("dispersion_const_a must be > 0.")
    if not (0.0 < ca <= 1.0) or not (0.0 < cv <= 1.0):
        raise PhysicalBoundsError("Contraction and velocity coefficients ca, cv must be in (0, 1].")

    half_angle_rad = math.radians(0.5 * spray_angle_deg)
    d_eff = math.sqrt(ca) * nozzle_diameter_m
    x_plus = (d_eff * math.sqrt(rho_liquid_kg_m3 / rho_gas_kg_m3)) / (
        dispersion_const_a * math.tan(half_angle_rad)
    )
    u_f = cv * math.sqrt(2.0 * delta_p_pa / rho_liquid_kg_m3)
    t_plus = x_plus / u_f

    t_tilde = t / t_plus
    # Explicit SAE 960034 Eq. (21) approximation (n = 2.2), valid at t_tilde = 0
    n_exp = 2.2
    s_tilde = t_tilde / ((1.0 + np.power(t_tilde, 0.5 * n_exp)) ** (1.0 / n_exp))

    if method == "exact":
        # 5 fixed vectorized Newton-Raphson updates converge globally to < 1e-15
        s_tilde = _newton_step_naber_siebers(s_tilde, t_tilde)
        s_tilde = _newton_step_naber_siebers(s_tilde, t_tilde)
        s_tilde = _newton_step_naber_siebers(s_tilde, t_tilde)
        s_tilde = _newton_step_naber_siebers(s_tilde, t_tilde)
        s_tilde = _newton_step_naber_siebers(s_tilde, t_tilde)
    elif method != "explicit":
        raise ValueError(f"Unknown Naber-Siebers method '{method}'. Expected 'exact' or 'explicit'.")

    penetration_m = np.asarray(x_plus * s_tilde, dtype=np.float64)
    return NaberSiebersResult(
        time_s=t,
        penetration_m=penetration_m,
        t_tilde=np.asarray(t_tilde, dtype=np.float64),
        s_tilde=np.asarray(s_tilde, dtype=np.float64),
        x_plus_m=x_plus,
        t_plus_s=t_plus,
        u_f_m_s=u_f,
    )


def _exact_impingement_time_hiroyasu(
    wall_distance_m: float,
    *,
    delta_p_pa: float,
    rho_liquid_kg_m3: float,
    rho_gas_kg_m3: float,
    nozzle_diameter_m: float,
    c_breakup: float = 28.65,
    c_post: float = 2.95,
) -> float:
    res_single = hiroyasu_arai_penetration(
        np.array([0.0], dtype=np.float64),
        delta_p_pa=delta_p_pa,
        rho_liquid_kg_m3=rho_liquid_kg_m3,
        rho_gas_kg_m3=rho_gas_kg_m3,
        nozzle_diameter_m=nozzle_diameter_m,
        c_breakup=c_breakup,
        c_post=c_post,
    )
    if wall_distance_m < res_single.breakup_length_m:
        c_pre_eff = c_post / math.sqrt(2.0 * c_breakup)
        v_pre = c_pre_eff * math.sqrt(2.0 * delta_p_pa / rho_liquid_kg_m3)
        return wall_distance_m / v_pre
    k_post = c_post * ((delta_p_pa / rho_gas_kg_m3) ** 0.25) * math.sqrt(nozzle_diameter_m)
    return (wall_distance_m / k_post) ** 2


def _exact_impingement_time_naber_siebers(
    wall_distance_m: float,
    *,
    x_plus_m: float,
    t_plus_s: float,
) -> float:
    s_w = wall_distance_m / x_plus_m
    root_term = math.sqrt(1.0 + 16.0 * (s_w**2))
    t_tilde_w = 0.5 * s_w + 0.25 * s_w * root_term + (1.0 / 16.0) * math.log(4.0 * s_w + root_term)
    return t_plus_s * t_tilde_w


def evaluate_altitude_spray_risk(
    time_s: ArrayLike,
    *,
    delta_p_pa: float,
    rho_liquid_kg_m3: float,
    rho_gas_sea_level_kg_m3: float,
    rho_gas_altitude_kg_m3: float,
    nozzle_diameter_m: float,
    wall_distance_m: float,
    model: SprayModelType = "hiroyasu_arai",
    spray_angle_deg: float = 20.0,
) -> AltitudeSprayRiskResult:
    """Assess spray over-penetration and piston-bowl wall impingement at high altitude.

    Compares in-cylinder gas density reduction ($\\rho_{g,1} \\to \\rho_{g,2}$) for
    Aero-Diesel operation and resolves exact wall-impingement onset time $t_{\\text{imp}}$
    against bowl/liner clearance $L_{\\text{wall}}$ [m].

    Parameters
    ----------
    time_s : ArrayLike
        Injection/analysis time vector $t$ [s] (1D strictly non-negative).
    delta_p_pa : float
        Injection pressure differential $\\Delta P$ [Pa].
    rho_liquid_kg_m3 : float
        Fuel density $\\rho_l$ [$\\text{kg/m}^3$].
    rho_gas_sea_level_kg_m3 : float
        Sea-level TDC ambient gas density $\\rho_{g,1}$ [$\\text{kg/m}^3$].
    rho_gas_altitude_kg_m3 : float
        High-altitude TDC ambient gas density $\\rho_{g,2}$ [$\\text{kg/m}^3$].
    nozzle_diameter_m : float
        Nozzle hole diameter $d_o$ [m].
    wall_distance_m : float
        Available combustion-chamber / piston-bowl path length $L_{\\text{wall}}$ [m].
    model : {"hiroyasu_arai", "naber_siebers"}, default="hiroyasu_arai"
        Underlying penetration correlation.
    spray_angle_deg : float, default=20.0
        Full spray cone angle [deg] when ``model="naber_siebers"``.

    Returns
    -------
    AltitudeSprayRiskResult
        Penetration traces, scaling ratios, and wall-impingement onset times [s].
    """
    if wall_distance_m <= 0.0 or not math.isfinite(wall_distance_m):
        raise PhysicalBoundsError("Chamber wall distance wall_distance_m must be > 0 m.")
    t_arr = np.asarray(time_s, dtype=np.float64)
    if t_arr.ndim != 1 or t_arr.size < 1:
        raise ValueError("time_s must be a non-empty 1D array.")

    t_max_eval = float(np.max(t_arr))
    t_min_eval = float(np.min(t_arr))

    if model == "hiroyasu_arai":
        sl = hiroyasu_arai_penetration(
            t_arr,
            delta_p_pa=delta_p_pa,
            rho_liquid_kg_m3=rho_liquid_kg_m3,
            rho_gas_kg_m3=rho_gas_sea_level_kg_m3,
            nozzle_diameter_m=nozzle_diameter_m,
        )
        alt = hiroyasu_arai_penetration(
            t_arr,
            delta_p_pa=delta_p_pa,
            rho_liquid_kg_m3=rho_liquid_kg_m3,
            rho_gas_kg_m3=rho_gas_altitude_kg_m3,
            nozzle_diameter_m=nozzle_diameter_m,
        )
        s_sl = sl.penetration_m
        s_alt = alt.penetration_m
        t_imp_alt_raw = _exact_impingement_time_hiroyasu(
            wall_distance_m,
            delta_p_pa=delta_p_pa,
            rho_liquid_kg_m3=rho_liquid_kg_m3,
            rho_gas_kg_m3=rho_gas_altitude_kg_m3,
            nozzle_diameter_m=nozzle_diameter_m,
        )
        t_imp_sl_raw = _exact_impingement_time_hiroyasu(
            wall_distance_m,
            delta_p_pa=delta_p_pa,
            rho_liquid_kg_m3=rho_liquid_kg_m3,
            rho_gas_kg_m3=rho_gas_sea_level_kg_m3,
            nozzle_diameter_m=nozzle_diameter_m,
        )
    elif model == "naber_siebers":
        ns_sl = naber_siebers_penetration(
            t_arr,
            delta_p_pa=delta_p_pa,
            rho_liquid_kg_m3=rho_liquid_kg_m3,
            rho_gas_kg_m3=rho_gas_sea_level_kg_m3,
            nozzle_diameter_m=nozzle_diameter_m,
            spray_angle_deg=spray_angle_deg,
        )
        ns_alt = naber_siebers_penetration(
            t_arr,
            delta_p_pa=delta_p_pa,
            rho_liquid_kg_m3=rho_liquid_kg_m3,
            rho_gas_kg_m3=rho_gas_altitude_kg_m3,
            nozzle_diameter_m=nozzle_diameter_m,
            spray_angle_deg=spray_angle_deg,
        )
        s_sl = ns_sl.penetration_m
        s_alt = ns_alt.penetration_m
        t_imp_alt_raw = _exact_impingement_time_naber_siebers(
            wall_distance_m, x_plus_m=ns_alt.x_plus_m, t_plus_s=ns_alt.t_plus_s
        )
        t_imp_sl_raw = _exact_impingement_time_naber_siebers(
            wall_distance_m, x_plus_m=ns_sl.x_plus_m, t_plus_s=ns_sl.t_plus_s
        )
    else:
        raise ValueError(f"Unknown spray model '{model}'. Expected 'hiroyasu_arai' or 'naber_siebers'.")

    asymptotic_ratio = (rho_gas_sea_level_kg_m3 / rho_gas_altitude_kg_m3) ** 0.25
    terminal_ratio = (
        float(s_alt[-1] / s_sl[-1]) if s_sl[-1] > 0.0 else asymptotic_ratio
    )
    max_s_alt = float(np.max(s_alt))
    impinged_alt = max_s_alt >= wall_distance_m and t_imp_alt_raw <= t_max_eval
    impinged_sl = float(np.max(s_sl)) >= wall_distance_m and t_imp_sl_raw <= t_max_eval

    return AltitudeSprayRiskResult(
        time_s=t_arr,
        penetration_sea_level_m=s_sl,
        penetration_altitude_m=s_alt,
        asymptotic_penetration_ratio=asymptotic_ratio,
        terminal_penetration_ratio=terminal_ratio,
        wall_impingement_detected=impinged_alt,
        impingement_time_s=max(t_imp_alt_raw, t_min_eval) if impinged_alt else None,
        sea_level_impingement_time_s=max(t_imp_sl_raw, t_min_eval) if impinged_sl else None,
        max_penetration_altitude_m=max_s_alt,
    )


def compute_spray_validation_metrics(
    cfd_penetration_m: ArrayLike,
    ref_penetration_m: ArrayLike,
    *,
    cfd_time_s: ArrayLike | None = None,
    ref_time_s: ArrayLike | None = None,
    mape_floor_m: float = 1.0e-6,
) -> SprayValidationMetrics:
    """Compute RMSE [m], MAPE [%], and relative $L_2$ error between CFD and reference spray.

    When ``cfd_time_s`` and ``ref_time_s`` are provided, ``cfd_penetration_m`` is
    linearly interpolated onto the overlapping ``ref_time_s`` support before
    computing vectorized error norms.

    Parameters
    ----------
    cfd_penetration_m : ArrayLike
        3D CFD liquid or vapor penetration array $S_{\\text{cfd}}$ [m].
    ref_penetration_m : ArrayLike
        Empirical correlation or ECN experimental penetration $S_{\\text{ref}}$ [m].
    cfd_time_s : ArrayLike | None, optional
        CFD time stamps [s] when asynchronous interpolation is needed.
    ref_time_s : ArrayLike | None, optional
        Reference/experimental time stamps [s].
    mape_floor_m : float, default=1e-6
        Minimum reference penetration [m] included in MAPE denominator to avoid
        division by zero at $t = 0$.

    Returns
    -------
    SprayValidationMetrics
        RMSE [m], MAPE [%], relative $L_2$ norm [-], and $L_\\infty$ [m].
    """
    s_cfd = np.asarray(cfd_penetration_m, dtype=np.float64)
    s_ref = np.asarray(ref_penetration_m, dtype=np.float64)
    if s_cfd.ndim != 1 or s_ref.ndim != 1:
        raise ValueError("cfd_penetration_m and ref_penetration_m must be 1D arrays.")
    if not (np.all(np.isfinite(s_cfd)) and np.all(np.isfinite(s_ref))):
        raise ValueError("Penetration arrays must contain finite values.")

    if cfd_time_s is not None or ref_time_s is not None:
        if cfd_time_s is None or ref_time_s is None:
            raise ValueError("Provide both cfd_time_s and ref_time_s for asynchronous alignment.")
        t_cfd = np.asarray(cfd_time_s, dtype=np.float64)
        t_ref = np.asarray(ref_time_s, dtype=np.float64)
        if t_cfd.shape != s_cfd.shape or t_ref.shape != s_ref.shape:
            raise ValueError("Time arrays must match their respective penetration array shapes.")
        if np.any(np.diff(t_cfd) <= 0.0):
            raise ValueError("cfd_time_s must be strictly increasing for interpolation.")

        overlap = (t_ref >= t_cfd[0]) & (t_ref <= t_cfd[-1])
        if int(np.count_nonzero(overlap)) < 1:
            raise ValueError("cfd_time_s and ref_time_s share zero overlapping time range.")
        t_eval = t_ref[overlap]
        s_ref = s_ref[overlap]
        s_cfd = np.asarray(np.interp(t_eval, t_cfd, s_cfd), dtype=np.float64)
    elif s_cfd.shape != s_ref.shape or s_cfd.size == 0:
        raise ValueError(
            "cfd_penetration_m and ref_penetration_m must be non-empty arrays of identical length "
            "when time arrays are omitted."
        )

    if np.any(s_cfd < 0.0) or np.any(s_ref < 0.0):
        raise PhysicalBoundsError("Spray penetration values cannot be negative (< 0 m).")

    diff = s_cfd - s_ref
    rmse_m = float(np.sqrt(np.mean(diff**2)))
    max_abs_m = float(np.max(np.abs(diff)))

    ref_l2 = float(np.linalg.norm(s_ref))
    if ref_l2 <= 0.0:
        raise PhysicalBoundsError("Reference spray penetration has zero L2 norm everywhere.")
    l2_rel = float(np.linalg.norm(diff) / ref_l2)

    valid_mape = s_ref > mape_floor_m
    if np.any(valid_mape):
        mape_pct = float(100.0 * np.mean(np.abs(diff[valid_mape]) / s_ref[valid_mape]))
    else:
        mape_pct = 0.0

    return SprayValidationMetrics(
        rmse_m=rmse_m,
        mape_pct=mape_pct,
        l2_relative_error=l2_rel,
        max_abs_error_m=max_abs_m,
        n_samples=s_ref.size,
    )

