"""Shared analytical fixtures for open-source ICE CFD verification tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
import pytest

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PolytropicStrokeReference:
    """Analytical textbook slider-crank polytropic compression stroke [-180, 0] deg."""

    theta_deg: FloatArray
    volume_m3: FloatArray
    pressure_pa: FloatArray
    temperature_k: FloatArray
    polytropic_exp: float
    exact_work_j: float


@pytest.fixture
def analytical_polytropic_stroke() -> PolytropicStrokeReference:
    """Generate an NDA-free textbook polytropic compression stroke (Heywood, 2018).

    Uses generic textbook geometry: bore $B = 0.085\\,\\text{m}$, stroke $S = 0.090\\,\\text{m}$,
    connecting-rod ratio $R_{\\text{cr}} = l / a = 3.5$, compression ratio $r_c = 16.0$,
    and polytropic exponent $\\kappa = 1.35$.
    """
    theta_deg = np.linspace(-180.0, 0.0, 1801, dtype=np.float64)
    theta_rad = np.deg2rad(theta_deg)

    v_disp = (np.pi / 4.0) * (0.085**2) * 0.090
    r_c = 16.0
    v_clearance = v_disp / (r_c - 1.0)
    rod_ratio = 3.5

    # Exact slider-crank kinematics (Heywood, 2018, Eq. 2.4)
    s_term = np.cos(theta_rad) + np.sqrt(rod_ratio**2 - np.sin(theta_rad) ** 2)
    volume_m3 = v_clearance * (1.0 + 0.5 * (r_c - 1.0) * (rod_ratio + 1.0 - s_term))

    kappa = 1.35
    p1_pa = 101_325.0
    t1_k = 310.0
    v1_m3 = float(volume_m3[0])

    pressure_pa = p1_pa * (v1_m3 / volume_m3) ** kappa
    temperature_k = t1_k * (v1_m3 / volume_m3) ** (kappa - 1.0)

    exact_work_j = float(
        (pressure_pa[-1] * volume_m3[-1] - pressure_pa[0] * volume_m3[0]) / (1.0 - kappa)
    )

    return PolytropicStrokeReference(
        theta_deg=theta_deg,
        volume_m3=volume_m3,
        pressure_pa=pressure_pa,
        temperature_k=temperature_k,
        polytropic_exp=kappa,
        exact_work_j=exact_work_j,
    )

