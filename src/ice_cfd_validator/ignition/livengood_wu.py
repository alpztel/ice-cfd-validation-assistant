"""Dynamic Livengood-Wu (1955) auto-ignition integral solver.

Solves the cumulative ignition integral over a time-varying in-cylinder state

$$I(\\theta) = \\int_{\\theta_{SOI}}^{\\theta}
  \\frac{d\\theta'}{6\\,\\text{RPM}\\,\\tau(P(\\theta'), T(\\theta'), \\phi)}$$

and locates the start of combustion $\\theta_{SOC}$ where $I(\\theta_{SOC}) = 1$
with sub-degree accuracy independent of the crank-angle grid spacing.

Numerical scheme (loop-free):

1. $\\ln\\tau(\\theta)$ is interpolated with a not-a-knot cubic spline to place an
   exact integration node at an off-grid $\\theta_{SOI}$ (eliminates grid
   alignment error; exponential Arrhenius profiles are reproduced exactly when
   $\\ln\\tau$ is locally polynomial of degree $\\le 3$).
2. The integrand $f = 1/(6\\,\\text{RPM}\\,\\tau)$ is integrated with the
   analytical antiderivative of a not-a-knot cubic spline ($\\mathcal{O}(h^4)$).
3. $\\theta_{SOC}$ is bracketed by ``searchsorted`` and refined with Brent's
   method on a cubic Hermite interpolant of $I(\\theta)$ that uses the *exact*
   physical slope $dI/d\\theta = f(\\theta)$ (``"hermite"``, $\\mathcal{O}(h^4)$), or
   on a shape-preserving PCHIP interpolant (``"pchip"``).

References
----------
.. [1] Livengood, J. C., & Wu, P. C. (1955). Correlation of Autoignition
   Phenomena in Internal Combustion Engines and Rapid Compression Machines.
   Fifth Symposium (International) on Combustion, 347-356.
.. [2] Fritsch, F. N., & Carlson, R. E. (1980). Monotone Piecewise Cubic
   Interpolation. SIAM Journal on Numerical Analysis, 17(2), 238-246.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicHermiteSpline, CubicSpline, PchipInterpolator
from scipy.optimize import brentq

from ice_cfd_validator.exceptions import PhysicalBoundsError
from ice_cfd_validator.ignition.correlations import (
    ASSANIS_2003,
    ArrheniusDelayCoefficients,
    arrhenius_state_delay,
)

FloatArray: TypeAlias = NDArray[np.float64]
RootInterpolator: TypeAlias = Literal["hermite", "pchip"]

_SNAP_RELATIVE_TOL: float = 1.0e-9
_BRENT_XTOL: float = 1.0e-14
_BRENT_RTOL: float = 4.0 * float(np.finfo(np.float64).eps)


@dataclass(frozen=True, slots=True)
class LivengoodWuResult:
    """Livengood-Wu auto-ignition integral solution.

    Attributes
    ----------
    crank_angle_deg : NDArray[np.float64]
        Crank-angle grid $\\theta$ [deg].
    integral : NDArray[np.float64]
        Cumulative ignition integral $I(\\theta)$ [-] ($0$ for $\\theta \\le \\theta_{SOI}$).
    tau_s : NDArray[np.float64]
        Local ignition delay $\\tau(\\theta)$ [s].
    theta_soi_deg : float
        Start of injection $\\theta_{SOI}$ [deg].
    theta_soc_deg : float
        Start of combustion where $I = 1$ [deg], ``nan`` on misfire.
    ignition_delay_deg : float
        $\\theta_{SOC} - \\theta_{SOI}$ [deg], ``nan`` on misfire.
    ignition_delay_s : float
        $(\\theta_{SOC} - \\theta_{SOI}) / (6\\,\\text{RPM})$ [s], ``nan`` on misfire.
    misfire_detected : bool
        ``True`` if $\\max I(\\theta) < 1$ within the supplied window.
    max_integral_value : float
        $\\max I(\\theta)$ [-].
    rpm : float
        Engine speed [rev/min].
    """

    crank_angle_deg: FloatArray
    integral: FloatArray
    tau_s: FloatArray
    theta_soi_deg: float
    theta_soc_deg: float
    ignition_delay_deg: float
    ignition_delay_s: float
    misfire_detected: bool
    max_integral_value: float
    rpm: float


def _validate_grid(theta: FloatArray, name: str = "crank_angle_deg") -> None:
    if theta.ndim != 1 or theta.size < 2:
        raise PhysicalBoundsError(f"{name} must be a 1D array with at least 2 samples.")
    if not np.all(np.isfinite(theta)):
        raise PhysicalBoundsError(f"{name} contains non-finite values.")
    if np.any(np.diff(theta) <= 0.0):
        raise PhysicalBoundsError(f"{name} must be strictly increasing.")


def _interp_log_tau(theta: FloatArray, tau: FloatArray, theta_q: float) -> float:
    """Interpolate $\\tau$ at an off-grid angle through cubic interpolation of $\\ln\\tau$."""
    log_tau = np.log(tau)
    if theta.size >= 4:
        return float(np.exp(CubicSpline(theta, log_tau, bc_type="not-a-knot")(theta_q)))
    return float(np.exp(np.interp(theta_q, theta, log_tau)))


def _cumulative_integral(theta_w: FloatArray, f_w: FloatArray) -> FloatArray:
    """Spline-antiderivative cumulative integral (trapezoid fallback for < 4 nodes)."""
    if theta_w.size >= 4:
        anti = CubicSpline(theta_w, f_w, bc_type="not-a-knot").antiderivative()
        cum = np.asarray(anti(theta_w), dtype=np.float64) - float(anti(theta_w[0]))
    else:
        cum = np.asarray(cumulative_trapezoid(f_w, theta_w, initial=0.0), dtype=np.float64)
    # Guarantee monotonicity (no-op for any well-resolved positive integrand)
    return np.maximum.accumulate(cum)


def solve_livengood_wu(
    crank_angle_deg: ArrayLike,
    tau_s: ArrayLike,
    rpm: float,
    theta_soi_deg: float,
    *,
    root_interpolator: RootInterpolator = "hermite",
) -> LivengoodWuResult:
    """Solve the Livengood-Wu integral for a prescribed local delay profile.

    $$I(\\theta) = \\int_{\\theta_{SOI}}^{\\theta} \\frac{d\\theta'}{6\\,\\text{RPM}\\,\\tau(\\theta')},
      \\qquad I(\\theta_{SOC}) = 1$$

    Parameters
    ----------
    crank_angle_deg : ArrayLike
        Strictly increasing crank-angle grid $\\theta$ [deg] (uniform or not).
    tau_s : ArrayLike
        Local ignition delay $\\tau(\\theta) > 0$ on the same grid [s].
    rpm : float
        Engine speed [rev/min].
    theta_soi_deg : float
        Start of injection $\\theta_{SOI}$ [deg], may lie between grid nodes.
    root_interpolator : {"hermite", "pchip"}, optional
        Interpolant of $I(\\theta)$ used for sub-degree root refinement.
        ``"hermite"`` (default) uses the exact slope $dI/d\\theta = 1/(6\\,\\text{RPM}\\,\\tau)$;
        ``"pchip"`` uses Fritsch-Carlson monotone slopes.

    Returns
    -------
    LivengoodWuResult
        Integral trace, $\\theta_{SOC}$, ignition delay and misfire flag. On
        misfire ($\\max I < 1$) all SOC/delay fields are ``nan``.

    Raises
    ------
    PhysicalBoundsError
        For invalid grids, non-positive delays/speed, or SOI outside the grid.
    """
    theta = np.asarray(crank_angle_deg, dtype=np.float64)
    tau = np.asarray(tau_s, dtype=np.float64)
    rpm_f = float(rpm)
    soi = float(theta_soi_deg)

    _validate_grid(theta)
    if tau.shape != theta.shape:
        raise PhysicalBoundsError("tau_s must have the same shape as crank_angle_deg.")
    if not np.all(np.isfinite(tau)) or np.any(tau <= 0.0):
        raise PhysicalBoundsError("Local ignition delay tau_s must be finite and > 0 [s].")
    if not np.isfinite(rpm_f) or rpm_f <= 0.0:
        raise PhysicalBoundsError(f"Engine speed must be finite and > 0 rpm, got {rpm_f}.")
    if not np.isfinite(soi) or soi < theta[0] or soi >= theta[-1]:
        raise PhysicalBoundsError(
            f"theta_soi_deg={soi} must lie within [{theta[0]}, {theta[-1]}) deg."
        )

    deg_per_s = 6.0 * rpm_f
    snap_tol = _SNAP_RELATIVE_TOL * max(1.0, float(np.median(np.diff(theta))))

    after_mask = theta > soi + snap_tol
    tau_soi = _interp_log_tau(theta, tau, soi)
    theta_w = np.concatenate(([soi], theta[after_mask]))
    tau_w = np.concatenate(([tau_soi], tau[after_mask]))
    f_w = 1.0 / (deg_per_s * tau_w)  # dI/dtheta [1/deg]

    integral_w = _cumulative_integral(theta_w, f_w)
    integral = np.zeros_like(theta)
    integral[after_mask] = integral_w[1:]
    max_integral = float(integral_w[-1])

    if max_integral < 1.0:
        return LivengoodWuResult(
            crank_angle_deg=theta,
            integral=integral,
            tau_s=tau,
            theta_soi_deg=soi,
            theta_soc_deg=float("nan"),
            ignition_delay_deg=float("nan"),
            ignition_delay_s=float("nan"),
            misfire_detected=True,
            max_integral_value=max_integral,
            rpm=rpm_f,
        )

    k = int(np.searchsorted(integral_w, 1.0, side="left"))
    if integral_w[k] == 1.0:
        theta_soc = float(theta_w[k])
    else:
        if root_interpolator == "hermite":
            interp_i = CubicHermiteSpline(theta_w, integral_w, f_w)
        else:
            interp_i = PchipInterpolator(theta_w, integral_w)

        def _residual(x: float) -> float:
            return float(interp_i(x)) - 1.0

        brent_res = brentq(
            _residual,
            float(theta_w[k - 1]),
            float(theta_w[k]),
            xtol=_BRENT_XTOL,
            rtol=_BRENT_RTOL,
        )
        theta_soc = float(brent_res[0] if isinstance(brent_res, tuple) else brent_res)

    delay_deg = theta_soc - soi
    return LivengoodWuResult(
        crank_angle_deg=theta,
        integral=integral,
        tau_s=tau,
        theta_soi_deg=soi,
        theta_soc_deg=theta_soc,
        ignition_delay_deg=delay_deg,
        ignition_delay_s=delay_deg / deg_per_s,
        misfire_detected=False,
        max_integral_value=max_integral,
        rpm=rpm_f,
    )


def livengood_wu_integral(
    crank_angle_deg: ArrayLike,
    pressure_pa: ArrayLike,
    temperature_k: ArrayLike,
    rpm: float,
    theta_soi_deg: float,
    *,
    equivalence_ratio: ArrayLike = 1.0,
    coefficients: ArrheniusDelayCoefficients = ASSANIS_2003,
    cetane_number: float | None = None,
    root_interpolator: RootInterpolator = "hermite",
) -> LivengoodWuResult:
    """Solve the Livengood-Wu integral along a dynamic $P(\\theta)$, $T(\\theta)$ history.

    The local delay is evaluated with :func:`arrhenius_state_delay` and passed to
    :func:`solve_livengood_wu`.

    Parameters
    ----------
    crank_angle_deg : ArrayLike
        Strictly increasing crank-angle grid $\\theta$ [deg].
    pressure_pa : ArrayLike
        Cylinder pressure history $P(\\theta)$ [Pa].
    temperature_k : ArrayLike
        Cylinder temperature history $T(\\theta)$ [K].
    rpm : float
        Engine speed [rev/min].
    theta_soi_deg : float
        Start of injection [deg].
    equivalence_ratio : ArrayLike, optional
        Scalar or per-angle equivalence ratio $\\phi$ [-].
    coefficients : ArrheniusDelayCoefficients, optional
        Arrhenius coefficient set (``ASSANIS_2003`` default, or ``WOLFER_1938``).
    cetane_number : float | None, optional
        Optional cetane-number activation-energy scaling.
    root_interpolator : {"hermite", "pchip"}, optional
        Sub-degree root interpolant (see :func:`solve_livengood_wu`).

    Returns
    -------
    LivengoodWuResult
        Integral trace, $\\theta_{SOC}$, delays and misfire flag.
    """
    theta = np.asarray(crank_angle_deg, dtype=np.float64)
    tau = arrhenius_state_delay(
        pressure_pa,
        temperature_k,
        equivalence_ratio,
        coefficients=coefficients,
        cetane_number=cetane_number,
    )
    tau_b = np.broadcast_to(tau, theta.shape).astype(np.float64)
    return solve_livengood_wu(
        theta,
        tau_b,
        rpm,
        theta_soi_deg,
        root_interpolator=root_interpolator,
    )

