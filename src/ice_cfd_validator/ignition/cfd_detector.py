"""ANSYS Forte 3D CFD ignition diagnostics from apparent heat-release traces.

Extracts multi-criteria start-of-combustion metrics from the apparent net heat
release rate $dQ_n/d\\theta$ [J/deg] and cylinder pressure $P(\\theta)$ [Pa]:

1. Evaporative-cooling trough and negative-to-positive AHRR zero crossing
   $\\theta_{SOC,\\text{AHRR0}}$ with the integrated cooling energy $Q_{evap}$ [J].
2. Sub-degree cumulative-heat thresholds $\\theta_{CA01}$, $\\theta_{CA02}$,
   $\\theta_{CA05}$ (and $\\theta_{CA50}$).
3. Maximum pressure acceleration $\\max(d^2P/d\\theta^2)$ in $[\\theta_{SOI}, \\theta_{CA50}]$.
4. Two-stage (low-temperature "cool flame" LTHR vs. high-temperature HTHR)
   ignition and negative-temperature-coefficient (NTC) dwell detection.

All routines are loop-free (``numpy`` / ``scipy``) and operate in strict SI units.

References
----------
.. [1] Heywood, J. B. (2018). Internal Combustion Engine Fundamentals (2nd ed.).
   McGraw-Hill Education, Ch. 10.
.. [2] Westbrook, C. K. (2000). Chemical Kinetics of Hydrocarbon Ignition in
   Practical Combustion Systems. Proceedings of the Combustion Institute, 28,
   1563-1577.
.. [3] Dec, J. E. (1997). A Conceptual Model of DI Diesel Combustion Based on
   Laser-Sheet Imaging. SAE Technical Paper 970873.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import brentq
from scipy.signal import find_peaks

from ice_cfd_validator.exceptions import PhysicalBoundsError
from ice_cfd_validator.signal.filtering import compute_derivative

FloatArray: TypeAlias = NDArray[np.float64]

_NAN: float = float("nan")
_SNAP_RELATIVE_TOL: float = 1.0e-9
_BRENT_XTOL: float = 1.0e-13
_BRENT_RTOL: float = 4.0 * float(np.finfo(np.float64).eps)


@dataclass(frozen=True, slots=True)
class TwoStageIgnitionResult:
    """Low-temperature (LTHR) / high-temperature (HTHR) heat-release separation.

    Attributes
    ----------
    two_stage_detected : bool
        ``True`` if a resolved LTHR peak precedes the HTHR peak with an NTC dwell.
    theta_lthr_peak_deg : float
        Sub-degree crank angle of the LTHR (cool-flame) peak [deg].
    lthr_peak_rate_j_per_deg : float
        Peak LTHR apparent heat release rate [J/deg].
    temperature_lthr_k : float
        Charge temperature at the LTHR peak [K] (``nan`` if not supplied).
    q_lthr_j : float
        Positive heat released from the window start up to the NTC point [J].
    theta_ntc_deg : float
        Sub-degree crank angle of the NTC dwell (AHRR local minimum) [deg].
    ntc_min_rate_j_per_deg : float
        AHRR at the NTC dwell [J/deg].
    theta_hthr_peak_deg : float
        Sub-degree crank angle of the main HTHR peak [deg].
    hthr_peak_rate_j_per_deg : float
        Peak HTHR apparent heat release rate [J/deg].
    """

    two_stage_detected: bool
    theta_lthr_peak_deg: float
    lthr_peak_rate_j_per_deg: float
    temperature_lthr_k: float
    q_lthr_j: float
    theta_ntc_deg: float
    ntc_min_rate_j_per_deg: float
    theta_hthr_peak_deg: float
    hthr_peak_rate_j_per_deg: float


@dataclass(frozen=True, slots=True)
class CFDIgnitionMetrics:
    """Multi-criteria CFD start-of-combustion diagnostics.

    Attributes
    ----------
    theta_soi_deg : float
        Start of injection $\\theta_{SOI}$ [deg].
    evaporative_dip_detected : bool
        ``True`` if $dQ_n/d\\theta < 0$ occurs after SOI.
    theta_min_ahrr_deg : float
        Crank angle of the evaporative-cooling AHRR minimum [deg].
    min_ahrr_j_per_deg : float
        Minimum (most negative) AHRR after SOI [J/deg].
    theta_soc_ahrr0_deg : float
        Negative-to-positive AHRR zero crossing after the trough [deg].
    ignition_delay_ahrr0_deg : float
        $\\theta_{SOC,\\text{AHRR0}} - \\theta_{SOI}$ [deg].
    evaporative_cooling_energy_j : float
        $Q_{evap} = \\int_{\\theta_{SOI}}^{\\theta_{SOC}} \\max(0, -dQ_n/d\\theta)\\,d\\theta$ [J].
    total_positive_heat_j : float
        $\\int_{\\theta_{SOI}}^{\\theta_{end}} \\max(0, dQ_n/d\\theta)\\,d\\theta$ [J].
    theta_ca01_deg : float
        1 % cumulative positive heat-release angle [deg].
    theta_ca02_deg : float
        2 % cumulative positive heat-release angle [deg].
    theta_ca05_deg : float
        5 % cumulative positive heat-release angle [deg].
    theta_ca50_deg : float
        50 % cumulative positive heat-release angle [deg].
    theta_max_d2p_deg : float
        Angle of maximum $d^2P/d\\theta^2$ within $[\\theta_{SOI}, \\theta_{CA50}]$ [deg].
    max_d2p_pa_per_deg2 : float
        Maximum pressure acceleration $d^2P/d\\theta^2$ [Pa/deg^2].
    two_stage : TwoStageIgnitionResult
        LTHR / NTC / HTHR separation.
    """

    theta_soi_deg: float
    evaporative_dip_detected: bool
    theta_min_ahrr_deg: float
    min_ahrr_j_per_deg: float
    theta_soc_ahrr0_deg: float
    ignition_delay_ahrr0_deg: float
    evaporative_cooling_energy_j: float
    total_positive_heat_j: float
    theta_ca01_deg: float
    theta_ca02_deg: float
    theta_ca05_deg: float
    theta_ca50_deg: float
    theta_max_d2p_deg: float
    max_d2p_pa_per_deg2: float
    two_stage: TwoStageIgnitionResult


def _as_1d(values: ArrayLike, name: str, size: int | None = None) -> FloatArray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise PhysicalBoundsError(f"{name} must be a 1D array.")
    if size is not None and arr.size != size:
        raise PhysicalBoundsError(f"{name} must have {size} samples, got {arr.size}.")
    if not np.all(np.isfinite(arr)):
        raise PhysicalBoundsError(f"{name} contains non-finite values.")
    return arr


def _parabolic_vertex(theta: FloatArray, y: FloatArray, idx: int) -> tuple[float, float]:
    """Exact quadratic vertex through (idx-1, idx, idx+1) on a non-uniform grid."""
    if idx <= 0 or idx >= theta.size - 1:
        return float(theta[idx]), float(y[idx])
    x0, x1, x2 = float(theta[idx - 1]), float(theta[idx]), float(theta[idx + 1])
    y0, y1, y2 = float(y[idx - 1]), float(y[idx]), float(y[idx + 1])
    d01 = (y1 - y0) / (x1 - x0)
    d12 = (y2 - y1) / (x2 - x1)
    a = (d12 - d01) / (x2 - x0)
    if a == 0.0:
        return x1, y1
    b = d01 - a * (x0 + x1)
    x_v = float(np.clip(-b / (2.0 * a), x0, x2))
    y_v = y0 + d01 * (x_v - x0) + a * (x_v - x0) * (x_v - x1)
    return x_v, float(y_v)


def _monotone_crossing(theta: FloatArray, y: FloatArray, target: float) -> float:
    """Sub-step crossing of a non-decreasing trace via PCHIP + Brent refinement."""
    if y.size < 2 or not (y[0] <= target <= y[-1]):
        return _NAN
    k = int(np.searchsorted(y, target, side="left"))
    if y[k] == target:
        return float(theta[k])
    pchip = PchipInterpolator(theta, y)

    def _residual(x: float) -> float:
        return float(pchip(x)) - target

    brent_res = brentq(
        _residual,
        float(theta[k - 1]),
        float(theta[k]),
        xtol=_BRENT_XTOL,
        rtol=_BRENT_RTOL,
    )
    return float(brent_res[0] if isinstance(brent_res, tuple) else brent_res)


def _window_with_endpoints(
    theta: FloatArray,
    values: FloatArray,
    start: float,
    stop: float,
    value_start: float,
    value_stop: float,
) -> tuple[FloatArray, FloatArray]:
    """Return nodes in [start, stop] with exact (possibly off-grid) endpoints inserted."""
    snap = _SNAP_RELATIVE_TOL * max(1.0, float(np.median(np.diff(theta))))
    inner = (theta > start + snap) & (theta < stop - snap)
    theta_w = np.concatenate(([start], theta[inner], [stop]))
    values_w = np.concatenate(([value_start], values[inner], [value_stop]))
    return theta_w, values_w


def _spline_integral(theta_w: FloatArray, values_w: FloatArray) -> float:
    """$\\mathcal{O}(h^4)$ spline quadrature with a trapezoid fallback for < 4 nodes."""
    if theta_w.size >= 4:
        spline = CubicSpline(theta_w, values_w, bc_type="not-a-knot")
        return float(spline.integrate(float(theta_w[0]), float(theta_w[-1])))
    return float(trapezoid(values_w, theta_w))


def detect_two_stage_ignition(
    crank_angle_deg: ArrayLike,
    dq_net_j_per_deg: ArrayLike,
    temperature_k: ArrayLike | None = None,
    *,
    theta_start_deg: float | None = None,
    lthr_temperature_window_k: tuple[float, float] = (550.0, 750.0),
    min_relative_prominence: float = 0.005,
) -> TwoStageIgnitionResult:
    """Detect cool-flame (LTHR), NTC dwell, and main (HTHR) heat-release stages.

    The HTHR peak is the global AHRR maximum after ``theta_start_deg``. LTHR
    candidates are prominent local AHRR maxima (``scipy.signal.find_peaks``)
    preceding the HTHR peak; if ``temperature_k`` is supplied the candidate must
    lie inside ``lthr_temperature_window_k`` (typically 550-750 K for heavy
    hydrocarbons such as Diesel and Jet-A1). The NTC dwell is the AHRR minimum
    between the LTHR and HTHR peaks. Peak/dwell angles are refined with exact
    three-point parabolic vertices.

    Parameters
    ----------
    crank_angle_deg : ArrayLike
        Strictly increasing crank angle $\\theta$ [deg].
    dq_net_j_per_deg : ArrayLike
        Apparent net heat release rate $dQ_n/d\\theta$ [J/deg].
    temperature_k : ArrayLike | None, optional
        Bulk charge temperature $T(\\theta)$ [K] for LTHR window gating.
    theta_start_deg : float | None, optional
        Lower bound of the search window [deg] (default: first sample).
    lthr_temperature_window_k : tuple[float, float], optional
        Admissible LTHR peak temperature range [K].
    min_relative_prominence : float, optional
        Minimum LTHR peak prominence relative to the HTHR peak rate [-].

    Returns
    -------
    TwoStageIgnitionResult
        Stage angles, rates and LTHR energy (``nan`` fields if single-stage).
    """
    theta = _as_1d(crank_angle_deg, "crank_angle_deg")
    if theta.size < 5 or np.any(np.diff(theta) <= 0.0):
        raise PhysicalBoundsError("crank_angle_deg must be strictly increasing with >= 5 samples.")
    dq = _as_1d(dq_net_j_per_deg, "dq_net_j_per_deg", theta.size)
    temp = None if temperature_k is None else _as_1d(temperature_k, "temperature_k", theta.size)

    start = float(theta[0]) if theta_start_deg is None else float(theta_start_deg)
    win = np.flatnonzero(theta >= start)
    theta_w = theta[win]
    dq_w = dq[win]

    single_stage = TwoStageIgnitionResult(
        two_stage_detected=False,
        theta_lthr_peak_deg=_NAN,
        lthr_peak_rate_j_per_deg=_NAN,
        temperature_lthr_k=_NAN,
        q_lthr_j=_NAN,
        theta_ntc_deg=_NAN,
        ntc_min_rate_j_per_deg=_NAN,
        theta_hthr_peak_deg=_NAN,
        hthr_peak_rate_j_per_deg=_NAN,
    )
    if theta_w.size < 5:
        return single_stage

    i_hthr = int(np.argmax(dq_w))
    hthr_rate = float(dq_w[i_hthr])
    if hthr_rate <= 0.0:
        return single_stage
    theta_hthr, hthr_rate_v = _parabolic_vertex(theta_w, dq_w, i_hthr)

    peaks, _ = find_peaks(dq_w, prominence=min_relative_prominence * hthr_rate)
    candidates = peaks[(peaks < i_hthr) & (dq_w[peaks] > 0.0)]
    if temp is not None and candidates.size > 0:
        temp_w = temp[win]
        t_lo, t_hi = lthr_temperature_window_k
        candidates = candidates[(temp_w[candidates] >= t_lo) & (temp_w[candidates] <= t_hi)]

    if candidates.size == 0:
        return TwoStageIgnitionResult(
            two_stage_detected=False,
            theta_lthr_peak_deg=_NAN,
            lthr_peak_rate_j_per_deg=_NAN,
            temperature_lthr_k=_NAN,
            q_lthr_j=_NAN,
            theta_ntc_deg=_NAN,
            ntc_min_rate_j_per_deg=_NAN,
            theta_hthr_peak_deg=theta_hthr,
            hthr_peak_rate_j_per_deg=hthr_rate_v,
        )

    i_lthr = int(candidates[np.argmax(dq_w[candidates])])
    i_ntc = i_lthr + int(np.argmin(dq_w[i_lthr : i_hthr + 1]))
    theta_lthr, lthr_rate = _parabolic_vertex(theta_w, dq_w, i_lthr)
    theta_ntc, ntc_rate = _parabolic_vertex(theta_w, dq_w, i_ntc)

    q_pos = np.maximum(dq_w, 0.0)
    q_cum = cumulative_trapezoid(q_pos, theta_w, initial=0.0)
    q_lthr = float(np.interp(theta_ntc, theta_w, q_cum))
    t_lthr = _NAN if temp is None else float(np.interp(theta_lthr, theta, temp))

    return TwoStageIgnitionResult(
        two_stage_detected=True,
        theta_lthr_peak_deg=theta_lthr,
        lthr_peak_rate_j_per_deg=lthr_rate,
        temperature_lthr_k=t_lthr,
        q_lthr_j=q_lthr,
        theta_ntc_deg=theta_ntc,
        ntc_min_rate_j_per_deg=ntc_rate,
        theta_hthr_peak_deg=theta_hthr,
        hthr_peak_rate_j_per_deg=hthr_rate_v,
    )


def extract_cfd_ignition_metrics(
    crank_angle_deg: ArrayLike,
    dq_net_j_per_deg: ArrayLike,
    pressure_pa: ArrayLike,
    theta_soi_deg: float,
    *,
    temperature_k: ArrayLike | None = None,
    end_deg: float | None = None,
    lthr_temperature_window_k: tuple[float, float] = (550.0, 750.0),
) -> CFDIgnitionMetrics:
    """Extract multi-criteria CFD ignition metrics from AHRR and pressure traces.

    Criterion 1 - evaporative trough: after $\\theta_{SOI}$ the minimum of
    $dQ_n/d\\theta$ is located and the first subsequent negative-to-positive zero
    crossing ($d^2Q_n/d\\theta^2 > 0$) is refined with PCHIP + Brent's method to
    give $\\theta_{SOC,\\text{AHRR0}}$. The cooling energy

    $$Q_{evap} = \\int_{\\theta_{SOI}}^{\\theta_{SOC}} \\max(0, -dQ_n/d\\theta)\\,d\\theta$$

    is integrated with $\\mathcal{O}(h^4)$ spline quadrature on a window whose
    end-points are placed exactly on $\\theta_{SOI}$ and $\\theta_{SOC}$.

    Criterion 2 - cumulative positive heat thresholds CA01/CA02/CA05/CA50.

    Criterion 3 - maximum $d^2P/d\\theta^2$ ($\\mathcal{O}(h^4)$ differentiation)
    within $[\\theta_{SOI}, \\theta_{CA50}]$, refined by a parabolic vertex.

    Parameters
    ----------
    crank_angle_deg : ArrayLike
        Strictly increasing crank angle $\\theta$ [deg] (>= 5 samples).
    dq_net_j_per_deg : ArrayLike
        Apparent net heat release rate $dQ_n/d\\theta$ [J/deg].
    pressure_pa : ArrayLike
        Cylinder pressure $P(\\theta)$ [Pa].
    theta_soi_deg : float
        Start of injection [deg].
    temperature_k : ArrayLike | None, optional
        Charge temperature $T(\\theta)$ [K] (enables LTHR temperature gating).
    end_deg : float | None, optional
        Upper bound of the analysis window [deg] (default: last sample).
    lthr_temperature_window_k : tuple[float, float], optional
        Admissible LTHR peak temperature range [K].

    Returns
    -------
    CFDIgnitionMetrics
        All ignition criteria (``nan`` where a criterion is not resolvable).

    Raises
    ------
    PhysicalBoundsError
        For invalid arrays, non-positive pressure, or SOI outside the window.
    """
    theta = _as_1d(crank_angle_deg, "crank_angle_deg")
    if theta.size < 5 or np.any(np.diff(theta) <= 0.0):
        raise PhysicalBoundsError("crank_angle_deg must be strictly increasing with >= 5 samples.")
    dq = _as_1d(dq_net_j_per_deg, "dq_net_j_per_deg", theta.size)
    p_pa = _as_1d(pressure_pa, "pressure_pa", theta.size)
    if np.any(p_pa <= 0.0):
        raise PhysicalBoundsError("pressure_pa must be strictly positive [Pa].")

    soi = float(theta_soi_deg)
    end = float(theta[-1]) if end_deg is None else min(float(end_deg), float(theta[-1]))
    if not (theta[0] <= soi < end):
        raise PhysicalBoundsError(f"theta_soi_deg={soi} must lie within [{theta[0]}, {end}) deg.")

    dq_pchip = PchipInterpolator(theta, dq)
    dq_soi = float(dq_pchip(soi))
    dq_end = float(dq_pchip(end))

    # ---------------------------------------------------------------- Window [SOI, end]
    theta_w, dq_w = _window_with_endpoints(theta, dq, soi, end, dq_soi, dq_end)

    # ------------------------------------------------- Criterion 1: evaporative trough
    i_min = int(np.argmin(dq_w))
    min_rate = float(dq_w[i_min])
    dip_detected = min_rate < 0.0
    theta_soc = _NAN
    q_evap = 0.0
    if dip_detected:
        neg_to_pos = np.flatnonzero((dq_w[:-1] < 0.0) & (dq_w[1:] >= 0.0))
        neg_to_pos = neg_to_pos[neg_to_pos >= i_min]
        if neg_to_pos.size > 0:
            k = int(neg_to_pos[0])
            if dq_w[k + 1] == 0.0:
                theta_soc = float(theta_w[k + 1])
            else:

                def _ahrr(x: float) -> float:
                    return float(dq_pchip(x))

                brent_res = brentq(
                    _ahrr,
                    float(theta_w[k]),
                    float(theta_w[k + 1]),
                    xtol=_BRENT_XTOL,
                    rtol=_BRENT_RTOL,
                )
                theta_soc = float(brent_res[0] if isinstance(brent_res, tuple) else brent_res)
            theta_e, dq_e = _window_with_endpoints(theta, dq, soi, theta_soc, dq_soi, 0.0)
            q_evap = _spline_integral(theta_e, np.maximum(-dq_e, 0.0))

    # ------------------------------------------- Criterion 2: cumulative positive heat
    q_cum = cumulative_trapezoid(np.maximum(dq_w, 0.0), theta_w, initial=0.0)
    q_total = float(q_cum[-1])
    if q_total > 0.0:
        mfb = q_cum / q_total
        ca01 = _monotone_crossing(theta_w, mfb, 0.01)
        ca02 = _monotone_crossing(theta_w, mfb, 0.02)
        ca05 = _monotone_crossing(theta_w, mfb, 0.05)
        ca50 = _monotone_crossing(theta_w, mfb, 0.50)
    else:
        ca01 = ca02 = ca05 = ca50 = _NAN

    # ---------------------------------------- Criterion 3: max pressure acceleration
    d2p = compute_derivative(theta, compute_derivative(theta, p_pa))
    upper = ca50 if np.isfinite(ca50) else end
    acc_idx = np.flatnonzero((theta >= soi) & (theta <= upper))
    if acc_idx.size > 0:
        i_acc = int(acc_idx[np.argmax(d2p[acc_idx])])
        theta_acc, d2p_max = _parabolic_vertex(theta, d2p, i_acc)
        theta_acc = float(np.clip(theta_acc, soi, upper))
    else:
        theta_acc, d2p_max = _NAN, _NAN

    two_stage = detect_two_stage_ignition(
        theta,
        dq,
        temperature_k,
        theta_start_deg=soi,
        lthr_temperature_window_k=lthr_temperature_window_k,
    )

    return CFDIgnitionMetrics(
        theta_soi_deg=soi,
        evaporative_dip_detected=dip_detected,
        theta_min_ahrr_deg=float(theta_w[i_min]),
        min_ahrr_j_per_deg=min_rate,
        theta_soc_ahrr0_deg=theta_soc,
        ignition_delay_ahrr0_deg=theta_soc - soi,
        evaporative_cooling_energy_j=q_evap,
        total_positive_heat_j=q_total,
        theta_ca01_deg=ca01,
        theta_ca02_deg=ca02,
        theta_ca05_deg=ca05,
        theta_ca50_deg=ca50,
        theta_max_d2p_deg=theta_acc,
        max_d2p_pa_per_deg2=d2p_max,
        two_stage=two_stage,
    )
