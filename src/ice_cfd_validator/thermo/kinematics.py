"""Slider-crank kinematics and thermodynamic TDC phase alignment diagnostics.

Computes instantaneous cylinder volume $V(\\theta)$, analytical derivative
$dV/d\\theta$, convective heat-transfer surface area $A_w(\\theta)$, and motored
TDC loss-angle verification (Heywood, 2018; Pipitone, 2008).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

from ice_cfd_validator.exceptions import PhysicalBoundsError, TDCPhaseError

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class KinematicsState:
    """Instantaneous slider-crank kinematic state arrays in strict SI units.

    Attributes
    ----------
    crank_angle_deg : NDArray[np.float64]
        Crank angle $\\theta$ [deg].
    volume_m3 : NDArray[np.float64]
        Instantaneous cylinder volume $V(\\theta)$ [$\\text{m}^3$].
    dv_dtheta_m3_per_deg : NDArray[np.float64]
        Analytical volume derivative $dV/d\\theta$ [$\\text{m}^3/\\text{deg}$].
    wall_area_m2 : NDArray[np.float64]
        Total exposed combustion chamber surface area $A_w(\\theta)$ [$\\text{m}^2$].
    piston_displacement_m : NDArray[np.float64]
        Axial piston displacement from TDC $s(\\theta)$ [m].
    """

    crank_angle_deg: FloatArray
    volume_m3: FloatArray
    dv_dtheta_m3_per_deg: FloatArray
    wall_area_m2: FloatArray
    piston_displacement_m: FloatArray


@dataclass(frozen=True, slots=True)
class SliderCrankGeometry:
    """Offset or centered slider-crank geometric specification (Heywood, 2018, Ch. 2).

    Parameters
    ----------
    bore_m : float
        Cylinder bore $B$ [m].
    stroke_m : float
        Nominal crank-throw stroke $S = 2a$ [m].
    conrod_m : float
        Connecting-rod center-to-center length $l$ [m].
    compression_ratio : float
        Geometric compression ratio $r_c = V_{\\max} / V_{\\min}$ [-].
    pin_offset_m : float, default=0.0
        Wrist-pin lateral offset $e$ [m].
    crown_area_multiplier : float, default=1.0
        Ratio of actual head + piston-bowl crown area to flat bore area $2 A_p$ [-].
    """

    bore_m: float
    stroke_m: float
    conrod_m: float
    compression_ratio: float
    pin_offset_m: float = 0.0
    crown_area_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if self.bore_m <= 0.0 or self.stroke_m <= 0.0 or self.conrod_m <= 0.0:
            raise PhysicalBoundsError("Bore, stroke, and connecting-rod length must be > 0 m.")
        if self.compression_ratio <= 1.0:
            raise PhysicalBoundsError("Compression ratio r_c must be > 1.0.")
        crank_radius = 0.5 * self.stroke_m
        if self.conrod_m <= crank_radius + abs(self.pin_offset_m):
            raise PhysicalBoundsError(
                "Connecting rod length l must exceed crank radius + |pin_offset|."
            )
        if self.crown_area_multiplier <= 0.0:
            raise PhysicalBoundsError("crown_area_multiplier must be > 0.")

    @property
    def piston_area_m2(self) -> float:
        """Projected piston area $A_p = \\pi B^2 / 4$ [$\\text{m}^2$]."""
        return 0.25 * math.pi * (self.bore_m**2)

    @property
    def displaced_volume_m3(self) -> float:
        """Exact swept cylinder volume $V_d$ [$\\text{m}^3$] accounting for pin offset."""
        a = 0.5 * self.stroke_m
        y_max = math.sqrt((self.conrod_m + a) ** 2 - self.pin_offset_m**2)
        y_min = math.sqrt((self.conrod_m - a) ** 2 - self.pin_offset_m**2)
        return self.piston_area_m2 * (y_max - y_min)

    @property
    def clearance_volume_m3(self) -> float:
        """Minimum clearance volume $V_c = V_d / (r_c - 1)$ [$\\text{m}^3$]."""
        return self.displaced_volume_m3 / (self.compression_ratio - 1.0)

    def evaluate(self, crank_angle_deg: FloatArray) -> KinematicsState:
        """Evaluate $V(\\theta)$, $dV/d\\theta$, and $A_w(\\theta)$ across crank angles.

        Implements exact offset slider-crank kinematics (Heywood, 2018, Eq. 2.4–2.7):

        $$s(\\theta) = \\sqrt{(l + a)^2 - e^2} - a \\cos\\theta - \\sqrt{l^2 - (a \\sin\\theta - e)^2}$$

        Parameters
        ----------
        crank_angle_deg : NDArray[np.float64]
            Crank angle array $\\theta$ [deg].

        Returns
        -------
        KinematicsState
            Vectorized kinematic state in SI units ($\\text{m}^3$, $\\text{m}^3/\\text{deg}$, $\\text{m}^2$).
        """
        theta_deg = np.asarray(crank_angle_deg, dtype=np.float64)
        if not np.all(np.isfinite(theta_deg)):
            raise ValueError("crank_angle_deg contains NaN or infinite values.")
        theta_rad = np.deg2rad(theta_deg)

        a = 0.5 * self.stroke_m
        l_rod = self.conrod_m
        e_off = self.pin_offset_m
        a_p = self.piston_area_m2
        v_c = self.clearance_volume_m3

        sin_t = np.sin(theta_rad)
        cos_t = np.cos(theta_rad)
        lateral = a * sin_t - e_off
        rod_proj = np.sqrt(l_rod**2 - lateral**2)
        y_max = math.sqrt((l_rod + a) ** 2 - e_off**2)

        s_m = y_max - a * cos_t - rod_proj
        ds_dtheta_rad = a * sin_t + (a * cos_t * lateral) / rod_proj

        volume_m3 = v_c + a_p * s_m
        dv_dtheta_m3_per_deg = a_p * ds_dtheta_rad * (math.pi / 180.0)

        # Exposed head + bowl crown + instantaneous liner surface area (Heywood, Eq. 2.7)
        clearance_height_m = v_c / a_p
        wall_area_m2 = (
            2.0 * self.crown_area_multiplier * a_p
            + math.pi * self.bore_m * (clearance_height_m + s_m)
        )

        return KinematicsState(
            crank_angle_deg=theta_deg,
            volume_m3=volume_m3,
            dv_dtheta_m3_per_deg=dv_dtheta_m3_per_deg,
            wall_area_m2=wall_area_m2,
            piston_displacement_m=s_m,
        )


def compute_slider_crank_kinematics(
    crank_angle_deg: FloatArray,
    *,
    bore_m: float,
    stroke_m: float,
    conrod_m: float,
    compression_ratio: float,
    pin_offset_m: float = 0.0,
    crown_area_multiplier: float = 1.0,
) -> KinematicsState:
    """Functional wrapper computing $V(\\theta)$, $dV/d\\theta$, and $A_w(\\theta)$ in SI units."""
    geom = SliderCrankGeometry(
        bore_m=bore_m,
        stroke_m=stroke_m,
        conrod_m=conrod_m,
        compression_ratio=compression_ratio,
        pin_offset_m=pin_offset_m,
        crown_area_multiplier=crown_area_multiplier,
    )
    return geom.evaluate(crank_angle_deg)


def compute_kinematics(
    crank_angle_deg: FloatArray,
    geometry: SliderCrankGeometry,
) -> KinematicsState:
    """Functional interface evaluating ``geometry`` across ``crank_angle_deg``."""
    return geometry.evaluate(crank_angle_deg)


@dataclass(frozen=True, slots=True)
class TDCPhasingResult:
    """Diagnostic result of motored TDC crank-angle phase audit.

    Attributes
    ----------
    theta_vmin_deg : float
        Sub-degree crank angle of geometric minimum volume $\\theta_{V_{\\min}}$ [deg].
    theta_pmax_deg : float
        Sub-degree crank angle of motored peak pressure $\\theta_{P_{\\max}}$ [deg].
    loss_angle_deg : float
        Measured thermodynamic loss angle $\\theta_{V_{\\min}} - \\theta_{P_{\\max}}$ [deg]
        (positive when motored $P_{\\max}$ precedes geometric TDC due to wall heat loss).
    phase_error_deg : float
        Signed crank-angle phase deviation relative to expected loss angle [deg].
    is_valid : bool
        ``True`` if ``abs(phase_error_deg) <= tolerance_deg``.
    """

    theta_vmin_deg: float
    theta_pmax_deg: float
    loss_angle_deg: float
    phase_error_deg: float
    is_valid: bool


def _refine_extremum_deg(theta_deg: FloatArray, values: FloatArray, *, find_max: bool) -> float:
    idx = int(np.argmax(values) if find_max else np.argmin(values))
    if idx == 0 or idx == theta_deg.size - 1 or theta_deg.size < 4:
        return float(theta_deg[idx])

    lo = max(0, idx - 3)
    hi = min(theta_deg.size, idx + 4)
    local_theta = theta_deg[lo:hi]
    spline = CubicSpline(local_theta, values[lo:hi], bc_type="not-a-knot")
    roots = spline.derivative(1).roots(extrapolate=False)
    if roots.size == 0:
        return float(theta_deg[idx])
    nearest_idx = int(np.argmin(np.abs(roots - theta_deg[idx])))
    return float(roots[nearest_idx])


def verify_tdc_phasing(
    crank_angle_deg: FloatArray,
    motored_pressure_pa: FloatArray,
    volume_m3: FloatArray | None = None,
    *,
    expected_loss_angle_deg: float = 0.0,
    tolerance_deg: float = 0.5,
    search_window_deg: tuple[float, float] = (-90.0, 90.0),
    raise_on_error: bool = True,
) -> TDCPhasingResult:
    """Audit crank-angle phase alignment between motored $P_{\\max}$ and $V_{\\min}$.

    Prevents severe IMEP bias (~5–10% per degree CA shift; Pipitone, 2008) by
    resolving sub-degree $\\theta_{P_{\\max}}$ and $\\theta_{V_{\\min}}$ near
    compression TDC via local cubic splines and checking deviation from
    ``expected_loss_angle_deg``.

    Parameters
    ----------
    crank_angle_deg : NDArray[np.float64]
        Strictly increasing crank angle array $\\theta$ [deg].
    motored_pressure_pa : NDArray[np.float64]
        Non-firing (motored) cylinder pressure trace $P_m(\\theta)$ [Pa].
    volume_m3 : NDArray[np.float64] | None, optional
        Instantaneous cylinder volume $V(\\theta)$ [$\\text{m}^3$]. When ``None``,
        geometric minimum volume is taken as centered TDC ($\\theta_{V_{\\min}} = 0^\\circ$).
    expected_loss_angle_deg : float, default=0.0
        Reference thermodynamic heat-loss angle $\\theta_{V_{\\min}} - \\theta_{P_{\\max}}$ [deg]
        (0.0 deg for adiabatic/polytropic reference; typically +0.3 to +1.0 deg with heat loss).
    tolerance_deg : float, default=0.5
        Maximum admissible absolute phase deviation [deg].
    search_window_deg : tuple[float, float], default=(-90.0, 90.0)
        Crank-angle window isolating compression TDC from gas-exchange TDC on 720-deg traces.
    raise_on_error : bool, default=True
        If ``True``, raises :class:`TDCPhaseError` on exceedance; otherwise emits
        a :class:`UserWarning` and returns ``is_valid=False``.

    Returns
    -------
    TDCPhasingResult
        Sub-degree phase alignment diagnostic metrics.
    """
    theta = np.asarray(crank_angle_deg, dtype=np.float64)
    p_mot = np.asarray(motored_pressure_pa, dtype=np.float64)

    if theta.ndim != 1 or p_mot.ndim != 1 or theta.size != p_mot.size or theta.size < 5:
        raise ValueError("crank_angle_deg and motored_pressure_pa must be matching 1D arrays (N >= 5).")
    if np.any(np.diff(theta) <= 0.0):
        raise ValueError("crank_angle_deg must be strictly increasing.")
    if np.any(p_mot <= 0.0):
        raise PhysicalBoundsError("motored_pressure_pa must be strictly positive (> 0 Pa).")
    if tolerance_deg <= 0.0:
        raise ValueError("tolerance_deg must be > 0.")

    mask = (theta >= search_window_deg[0]) & (theta <= search_window_deg[1])
    use_sub = int(np.count_nonzero(mask)) >= 5
    theta_sub = theta[mask] if use_sub else theta
    p_sub = p_mot[mask] if use_sub else p_mot

    if volume_m3 is not None:
        vol = np.asarray(volume_m3, dtype=np.float64)
        if vol.ndim != 1 or vol.size != theta.size:
            raise ValueError("volume_m3 must be a 1D array matching crank_angle_deg.")
        if np.any(vol <= 0.0):
            raise PhysicalBoundsError("volume_m3 must be strictly positive (> 0 m^3).")
        vol_sub = vol[mask] if use_sub else vol
        theta_vmin = _refine_extremum_deg(theta_sub, vol_sub, find_max=False)
    else:
        theta_vmin = 0.0

    theta_pmax = _refine_extremum_deg(theta_sub, p_sub, find_max=True)
    loss_angle = theta_vmin - theta_pmax
    phase_err = theta_pmax - (theta_vmin - expected_loss_angle_deg)
    is_valid = abs(phase_err) <= tolerance_deg

    if not is_valid:
        msg = (
            f"TDC phase alignment exceeded +/-{tolerance_deg:.2f} deg limit: "
            f"theta_Vmin={theta_vmin:.4f} deg, theta_Pmax={theta_pmax:.4f} deg, "
            f"loss_angle={loss_angle:.4f} deg, phase_error={phase_err:+.4f} deg."
        )
        if raise_on_error:
            raise TDCPhaseError(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)

    return TDCPhasingResult(
        theta_vmin_deg=theta_vmin,
        theta_pmax_deg=theta_pmax,
        loss_angle_deg=loss_angle,
        phase_error_deg=phase_err,
        is_valid=is_valid,
    )
