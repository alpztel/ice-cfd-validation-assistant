"""Shared analytical fixtures and audit helpers for ICE CFD validation tests."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess

import numpy as np
from numpy.typing import NDArray
import pytest
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

from ice_cfd_validator.signal.filtering import compute_derivative
from ice_cfd_validator.thermo.kinematics import SliderCrankGeometry
from ice_cfd_validator.thermo.nasa_polynomials import DRY_AIR

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
    """Generate an textbook polytropic compression stroke (Heywood, 2018)."""
    theta_deg = np.linspace(-180.0, 0.0, 1801, dtype=np.float64)
    theta_rad = np.deg2rad(theta_deg)

    v_disp = (np.pi / 4.0) * (0.085**2) * 0.090
    r_c = 16.0
    v_clearance = v_disp / (r_c - 1.0)
    rod_ratio = 3.5

    s_term = np.cos(theta_rad) + np.sqrt(rod_ratio**2 - np.sin(theta_rad) ** 2)
    volume_m3 = v_clearance * (1.0 + 0.5 * (r_c - 1.0) * (rod_ratio + 1.0 - s_term))

    kappa = 1.35
    p1_pa = 101_325.0
    t1_k = 310.0
    v1_m3 = float(volume_m3[0])

    pressure_pa = p1_pa * ((v1_m3 / volume_m3) ** kappa)
    temperature_k = t1_k * ((v1_m3 / volume_m3) ** (kappa - 1.0))
    p2_pa = float(pressure_pa[-1])
    v2_m3 = float(volume_m3[-1])
    exact_work_j = (p2_pa * v2_m3 - p1_pa * v1_m3) / (1.0 - kappa)

    return PolytropicStrokeReference(
        theta_deg=theta_deg,
        volume_m3=np.asarray(volume_m3, dtype=np.float64),
        pressure_pa=np.asarray(pressure_pa, dtype=np.float64),
        temperature_k=np.asarray(temperature_k, dtype=np.float64),
        polytropic_exp=kappa,
        exact_work_j=exact_work_j,
    )


@pytest.fixture
def standard_engine_geom() -> SliderCrankGeometry:
    """Standard baseline Aero-Diesel / research engine geometry fixture.

    Parameters: Bore 80 mm, Stroke 90 mm, Conrod 150 mm, CR 16.0.
    """
    return SliderCrankGeometry(
        bore_m=0.080,
        stroke_m=0.090,
        conrod_m=0.150,
        compression_ratio=16.0,
    )


@dataclass(frozen=True, slots=True)
class SyntheticRK45Cycle:
    """High-precision ODE-integrated closed-cycle engine trajectory."""

    theta_deg: FloatArray
    volume_m3: FloatArray
    dv_dtheta_m3_per_deg: FloatArray
    pressure_pa: FloatArray
    temperature_k: FloatArray
    gross_heat_release_rate_j_per_deg: FloatArray
    wall_heat_loss_rate_j_per_deg: FloatArray
    trapped_mass_kg: float


def generate_synthetic_rk45_cycle(
    geom: SliderCrankGeometry,
    *,
    n_points: int = 721,
    trapped_mass_kg: float = 5.5e-4,
    t_ivc_k: float = 320.0,
    q_comb_total_j: float = 1000.0,
    q_ht_total_j: float = 250.0,
    isentropic: bool = False,
) -> SyntheticRK45Cycle:
    """Generate high-precision RK45 integrated synthetic engine cycle."""
    theta = np.linspace(-180.0, 180.0, n_points)
    kin = geom.evaluate(theta)
    v = kin.volume_m3
    dv = kin.dv_dtheta_m3_per_deg
    r_spec = DRY_AIR.specific_gas_constant

    if isentropic:
        dq_comb_rate = np.zeros_like(theta)
        dq_ht_rate = np.zeros_like(theta)
    else:
        fraction = np.clip((theta - (-10.0)) / 50.0, 0.0, 1.0)
        wiebe = 1.0 - np.exp(-5.0 * (fraction ** 3.0))
        dq_comb_cum = q_comb_total_j * wiebe
        dq_comb_rate = compute_derivative(theta, dq_comb_cum)
        dq_ht_rate = np.maximum(
            (q_ht_total_j / 30.0) * np.exp(-0.5 * ((theta - 15.0) / 25.0) ** 2), 0.0
        )

    v_spline = CubicSpline(theta, v)
    dv_spline = v_spline.derivative()
    dq_c_spline = CubicSpline(theta, dq_comb_rate)
    dq_h_spline = CubicSpline(theta, dq_ht_rate)

    def cycle_rhs(th: float, state: np.ndarray) -> list[float]:
        t_curr = float(state[0])
        cv_curr = float(np.squeeze(DRY_AIR.cv(t_curr)))
        vol = float(v_spline(th))
        dvol = float(dv_spline(th))
        p_curr = trapped_mass_kg * r_spec * t_curr / vol
        dq_c = float(dq_c_spline(th))
        dq_h = float(dq_h_spline(th))
        return [(dq_c - dq_h - p_curr * dvol) / (trapped_mass_kg * cv_curr)]

    ode_sol = solve_ivp(
        cycle_rhs,
        [-180.0, 180.0],
        [t_ivc_k],
        t_eval=theta,
        method="RK45",
        rtol=1e-11,
        atol=1e-13,
    )
    assert ode_sol.success, "Synthetic RK45 cycle ODE failed!"
    t_sol = np.asarray(ode_sol.y[0], dtype=np.float64)
    p_sol = np.asarray(trapped_mass_kg * r_spec * t_sol / v, dtype=np.float64)

    return SyntheticRK45Cycle(
        theta_deg=theta,
        volume_m3=v,
        dv_dtheta_m3_per_deg=dv,
        pressure_pa=p_sol,
        temperature_k=t_sol,
        gross_heat_release_rate_j_per_deg=dq_comb_rate,
        wall_heat_loss_rate_j_per_deg=dq_ht_rate,
        trapped_mass_kg=trapped_mass_kg,
    )


def assert_package_vectorized_and_typed(subpackage: str | None = None) -> None:
    """Central AST zero-loop vectorization and pyrefly typechecker helper."""
    project_root = Path(__file__).resolve().parents[1]
    pkg_root = project_root / "src" / "ice_cfd_validator"
    target_dir = pkg_root / subpackage if subpackage else pkg_root

    assert target_dir.exists(), f"Directory {target_dir} does not exist!"

    numerical_subpackages = {"signal", "thermo", "spray", "vv", "ignition", "exergy"}
    loop_violations: list[str] = []

    if subpackage in numerical_subpackages or subpackage is None:
        for py_file in sorted(target_dir.rglob("*.py")):
            if py_file.name == "__init__.py":
                continue
            if subpackage is None and py_file.parent.name not in numerical_subpackages:
                continue
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, (ast.For, ast.While)):
                    iter_src = ast.unparse(node.iter) if isinstance(node, ast.For) else ""
                    if "mole_fractions.items()" in iter_src:
                        continue
                    loop_violations.append(
                        f"{py_file.relative_to(project_root).as_posix()}:{node.lineno}"
                    )

        assert len(loop_violations) == 0, f"Found explicit loops in numerical package: {loop_violations}"

    # Verify static typing via pyrefly
    pyrefly_path = shutil.which("pyrefly")
    if pyrefly_path is None:
        venv_pyrefly = project_root / ".venv" / "Scripts" / "pyrefly.exe"
        pyrefly_path = str(venv_pyrefly) if venv_pyrefly.exists() else "pyrefly"

    proc = subprocess.run([pyrefly_path, "check"], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"pyrefly type check failed:\n{proc.stdout}\n{proc.stderr}"
