"""In-cylinder convective wall heat-transfer correlations (Woschni & Hohenberg).

Computes instantaneous spatially averaged convective heat-transfer coefficient
$h_c(\\theta)$ [$\\text{W}/(\\text{m}^2\\cdot\\text{K})$] and crank-angle resolved
wall heat-loss rate $dQ_{ht}/d\\theta$ [$\\text{J/deg}$] (Woschni, 1967;
Hohenberg, 1979; Heywood, 2018, Ch. 12).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from ice_cfd_validator.exceptions import PhysicalBoundsError

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class HeatTransferResult:
    """Instantaneous wall heat-transfer diagnostics in strict SI units.

    Attributes
    ----------
    hc_w_m2k : NDArray[np.float64]
        Convective heat-transfer coefficient $h_c(\\theta)$ [$\\text{W}/(\\text{m}^2\\cdot\\text{K})$].
    dq_ht_j_per_deg : NDArray[np.float64]
        Wall heat-loss rate $dQ_{ht}/d\\theta$ [$\\text{J/deg}$] (positive for heat leaving gas).
    """

    hc_w_m2k: FloatArray
    dq_ht_j_per_deg: FloatArray


@dataclass(frozen=True, slots=True)
class WoschniCorrelation:
    """Woschni (1967) universal in-cylinder convective heat-transfer model.

    Implements SAE Paper 670931 (Eq. 11) / ANSYS Forte Theory Manual in strict SI inputs
    with $P \\cdot 10^{-5}$ ($\\text{Pa} \\to \\text{bar}$) scaling and $T^{-0.53}$ exponent:

    $$h_c = 130 B^{-0.2} (P \\cdot 10^{-5})^{0.8} T^{-0.53} w^{0.8}$$

    $$w = \\left(C_1 + 0.308 \\frac{v_u}{\\bar{S}_p}\\right) \\bar{S}_p
      + C_2 \\frac{V_d T_r}{P_r V_r} \\max(P - P_m, 0)$$

    Parameters
    ----------
    bore_m : float
        Cylinder bore $B$ [m].
    stroke_m : float
        Piston stroke $S$ [m].
    rpm : float
        Engine rotational speed $N$ [rev/min].
    wall_temp_k : float
        Effective area-weighted chamber wall temperature $T_w$ [K].
    displaced_volume_m3 : float | None, optional
        Swept volume $V_d$ [$\\text{m}^3$]; defaults to $(\\pi/4) B^2 S$.
    c1 : float, default=2.28
        Closed-cycle gas velocity coefficient $C_1$ [-].
    c2 : float, default=3.24e-3
        Combustion-induced turbulence velocity coefficient $C_2$ [$\\text{m}/(\\text{s}\\cdot\\text{K})$].
    swirl_velocity_m_s : float, default=0.0
        Circumferential in-cylinder swirl velocity $v_u$ [$\\text{m/s}$].
    """

    bore_m: float
    stroke_m: float
    rpm: float
    wall_temp_k: float
    displaced_volume_m3: float | None = None
    c1: float = 2.28
    c2: float = 3.24e-3
    swirl_velocity_m_s: float = 0.0

    def __post_init__(self) -> None:
        if self.bore_m <= 0.0 or self.stroke_m <= 0.0 or self.rpm <= 0.0 or self.wall_temp_k <= 0.0:
            raise PhysicalBoundsError("bore_m, stroke_m, rpm, and wall_temp_k must be > 0.")
        if self.swirl_velocity_m_s < 0.0:
            raise PhysicalBoundsError("swirl_velocity_m_s must be >= 0 m/s.")

    def evaluate(
        self,
        pressure_pa: FloatArray,
        temperature_k: FloatArray,
        wall_area_m2: FloatArray,
        *,
        motored_pressure_pa: FloatArray | None = None,
        p_ref_pa: float | None = None,
        v_ref_m3: float | None = None,
        t_ref_k: float | None = None,
    ) -> HeatTransferResult:
        """Compute Woschni $h_c(\\theta)$ and $dQ_{ht}/d\\theta$ across crank angles."""
        p_pa = np.asarray(pressure_pa, dtype=np.float64)
        t_k = np.asarray(temperature_k, dtype=np.float64)
        a_w = np.asarray(wall_area_m2, dtype=np.float64)
        if not (p_pa.shape == t_k.shape == a_w.shape):
            raise ValueError("pressure_pa, temperature_k, and wall_area_m2 must share identical shape.")
        if not (np.all(p_pa > 0.0) and np.all(t_k > 0.0) and np.all(a_w > 0.0)):
            raise PhysicalBoundsError("pressure_pa, temperature_k, and wall_area_m2 must be finite and > 0.")

        mean_piston_speed = 2.0 * self.stroke_m * (self.rpm / 60.0)
        w_base = self.c1 * mean_piston_speed + 0.308 * self.swirl_velocity_m_s
        w_gas = np.full_like(p_pa, w_base)

        if motored_pressure_pa is not None:
            if p_ref_pa is None or v_ref_m3 is None or t_ref_k is None:
                raise ValueError(
                    "p_ref_pa, v_ref_m3, and t_ref_k are required when motored_pressure_pa is given."
                )
            if p_ref_pa <= 0.0 or v_ref_m3 <= 0.0 or t_ref_k <= 0.0:
                raise PhysicalBoundsError("Reference state (p_ref_pa, v_ref_m3, t_ref_k) must be > 0.")
            p_mot = np.asarray(motored_pressure_pa, dtype=np.float64)
            if p_mot.shape != p_pa.shape:
                raise ValueError("motored_pressure_pa must match pressure_pa shape.")
            v_d = (
                self.displaced_volume_m3
                if self.displaced_volume_m3 is not None
                else 0.25 * math.pi * (self.bore_m**2) * self.stroke_m
            )
            w_gas = w_gas + (self.c2 * v_d * t_ref_k / (p_ref_pa * v_ref_m3)) * np.maximum(
                p_pa - p_mot, 0.0
            )

        hc = (
            130.0
            * (self.bore_m**-0.2)
            * ((p_pa * 1.0e-5) ** 0.8)
            * (t_k**-0.53)
            * (w_gas**0.8)
        )
        dq_ht = hc * a_w * (t_k - self.wall_temp_k) / (6.0 * self.rpm)
        return HeatTransferResult(
            hc_w_m2k=np.asarray(hc, dtype=np.float64),
            dq_ht_j_per_deg=np.asarray(dq_ht, dtype=np.float64),
        )


@dataclass(frozen=True, slots=True)
class HohenbergCorrelation:
    """Hohenberg (1979) direct-injection diesel convective heat-transfer model.

    Implements SAE Paper 790825 using instantaneous volume $V(\\theta)$ [$\\text{m}^3$]
    as characteristic length scale without requiring a motored reference trace:

    $$h_c = C_1 V(\\theta)^{-0.06} (P \\cdot 10^{-5})^{0.8} T^{-0.4} (\\bar{S}_p + C_2)^{0.8}$$

    Parameters
    ----------
    stroke_m : float
        Piston stroke $S$ [m].
    rpm : float
        Engine rotational speed $N$ [rev/min].
    wall_temp_k : float
        Effective area-weighted chamber wall temperature $T_w$ [K].
    c1 : float, default=130.0
        Hohenberg calibration constant $C_1$ [-].
    c2 : float, default=1.4
        Turbulence velocity offset $C_2$ [$\\text{m/s}$].
    """

    stroke_m: float
    rpm: float
    wall_temp_k: float
    c1: float = 130.0
    c2: float = 1.4

    def __post_init__(self) -> None:
        if self.stroke_m <= 0.0 or self.rpm <= 0.0 or self.wall_temp_k <= 0.0:
            raise PhysicalBoundsError("stroke_m, rpm, and wall_temp_k must be > 0.")

    def evaluate(
        self,
        pressure_pa: FloatArray,
        volume_m3: FloatArray,
        temperature_k: FloatArray,
        wall_area_m2: FloatArray,
    ) -> HeatTransferResult:
        """Compute Hohenberg $h_c(\\theta)$ and $dQ_{ht}/d\\theta$ across crank angles."""
        p_pa = np.asarray(pressure_pa, dtype=np.float64)
        v_m3 = np.asarray(volume_m3, dtype=np.float64)
        t_k = np.asarray(temperature_k, dtype=np.float64)
        a_w = np.asarray(wall_area_m2, dtype=np.float64)
        if not (p_pa.shape == v_m3.shape == t_k.shape == a_w.shape):
            raise ValueError(
                "pressure_pa, volume_m3, temperature_k, and wall_area_m2 must share identical shape."
            )
        if not (
            np.all(p_pa > 0.0)
            and np.all(v_m3 > 0.0)
            and np.all(t_k > 0.0)
            and np.all(a_w > 0.0)
        ):
            raise PhysicalBoundsError(
                "pressure_pa, volume_m3, temperature_k, and wall_area_m2 must be finite and > 0."
            )

        mean_piston_speed = 2.0 * self.stroke_m * (self.rpm / 60.0)
        hc = (
            self.c1
            * (v_m3**-0.06)
            * ((p_pa * 1.0e-5) ** 0.8)
            * (t_k**-0.4)
            * ((mean_piston_speed + self.c2) ** 0.8)
        )
        dq_ht = hc * a_w * (t_k - self.wall_temp_k) / (6.0 * self.rpm)
        return HeatTransferResult(
            hc_w_m2k=np.asarray(hc, dtype=np.float64),
            dq_ht_j_per_deg=np.asarray(dq_ht, dtype=np.float64),
        )


def compute_woschni_heat_transfer(
    pressure_pa: FloatArray,
    temperature_k: FloatArray,
    wall_area_m2: FloatArray,
    *,
    bore_m: float,
    stroke_m: float,
    rpm: float,
    wall_temp_k: float,
    displaced_volume_m3: float | None = None,
    motored_pressure_pa: FloatArray | None = None,
    p_ref_pa: float | None = None,
    v_ref_m3: float | None = None,
    t_ref_k: float | None = None,
    c1: float = 2.28,
    c2: float = 3.24e-3,
    swirl_velocity_m_s: float = 0.0,
) -> HeatTransferResult:
    """Functional interface for Woschni (1967) $h_c(\\theta)$ and $dQ_{ht}/d\\theta$."""
    model = WoschniCorrelation(
        bore_m=bore_m,
        stroke_m=stroke_m,
        rpm=rpm,
        wall_temp_k=wall_temp_k,
        displaced_volume_m3=displaced_volume_m3,
        c1=c1,
        c2=c2,
        swirl_velocity_m_s=swirl_velocity_m_s,
    )
    return model.evaluate(
        pressure_pa,
        temperature_k,
        wall_area_m2,
        motored_pressure_pa=motored_pressure_pa,
        p_ref_pa=p_ref_pa,
        v_ref_m3=v_ref_m3,
        t_ref_k=t_ref_k,
    )


def compute_hohenberg_heat_transfer(
    pressure_pa: FloatArray,
    volume_m3: FloatArray,
    temperature_k: FloatArray,
    wall_area_m2: FloatArray,
    *,
    stroke_m: float,
    rpm: float,
    wall_temp_k: float,
    c1: float = 130.0,
    c2: float = 1.4,
) -> HeatTransferResult:
    """Functional interface for Hohenberg (1979) $h_c(\\theta)$ and $dQ_{ht}/d\\theta$."""
    model = HohenbergCorrelation(
        stroke_m=stroke_m,
        rpm=rpm,
        wall_temp_k=wall_temp_k,
        c1=c1,
        c2=c2,
    )
    return model.evaluate(pressure_pa, volume_m3, temperature_k, wall_area_m2)
