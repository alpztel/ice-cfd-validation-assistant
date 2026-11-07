"""Analytical and edge-case unit tests for ANSYS Forte CSV parsers."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import simpson

from ice_cfd_validator.exceptions import (
    ForteParseError,
    PhysicalBoundsError,
    UnitDetectionError,
)
from ice_cfd_validator.io.forte_parser import (
    parse_dynamic_csv,
    parse_spray_csv,
    parse_thermo_csv,
)
from tests.conftest import PolytropicStrokeReference


def test_thermo_unit_invariance_and_polytropic_pdv_work(
    analytical_polytropic_stroke: PolytropicStrokeReference,
    tmp_path: Path,
) -> None:
    """Verify MPa/cm^3 and bar/mm^3 exports yield identical SI states and exact P*dV work."""
    ref = analytical_polytropic_stroke
    t_ms = (ref.theta_deg - ref.theta_deg[0]) / (6.0 * 2000.0) * 1.0e3

    # Format A: 1-row bracketed headers (MPa, cm^3, K, ms, g)
    df_a = pd.DataFrame({
        "Crank_Angle[deg]": ref.theta_deg,
        "Pressure[MPa]": ref.pressure_pa * 1.0e-6,
        "Volume[cm^3]": ref.volume_m3 * 1.0e6,
        "Temp[K]": ref.temperature_k,
        "Time[ms]": t_ms,
        "Mass[g]": np.full_like(ref.theta_deg, 0.625),
    })
    path_a = tmp_path / "thermo_mpa_cm3.csv"
    path_a.write_text(
        "# ANSYS Forte Spatially Averaged Thermo Output\n" + df_a.to_csv(index=False),
        encoding="utf-8",
    )

    # Format B: 2-row header + separate unit row (bar, mm^3, degC)
    df_b = pd.DataFrame({
        "Crank Angle": ref.theta_deg,
        "Cyl_Press": ref.pressure_pa * 1.0e-5,
        "Cyl_Volume": ref.volume_m3 * 1.0e9,
        "Mean_Temp": ref.temperature_k - 273.15,
    })
    body_b = "\n".join(df_b.to_csv(index=False).splitlines()[1:])
    csv_b = "Crank Angle, Cyl_Press, Cyl_Volume, Mean_Temp\n(CA), (bar), (mm³), (°C)\n" + body_b

    thermo_a = parse_thermo_csv(path_a)
    thermo_b = parse_thermo_csv(StringIO(csv_b))

    np.testing.assert_allclose(thermo_a.pressure_pa, ref.pressure_pa, rtol=1e-12)
    np.testing.assert_allclose(thermo_a.volume_m3, ref.volume_m3, rtol=1e-12)
    np.testing.assert_allclose(thermo_b.pressure_pa, thermo_a.pressure_pa, rtol=1e-12)
    np.testing.assert_allclose(thermo_b.volume_m3, thermo_a.volume_m3, rtol=1e-12)
    assert thermo_a.temperature_k is not None and thermo_b.temperature_k is not None
    np.testing.assert_allclose(thermo_b.temperature_k, thermo_a.temperature_k, atol=1e-11)
    assert thermo_a.trapped_mass_kg is not None
    np.testing.assert_allclose(thermo_a.trapped_mass_kg, 6.25e-4, rtol=1e-12)

    # Prove absence of 10^6 P*dV scale error via Simpson integration vs exact polytropic work
    computed_work_j = float(simpson(thermo_b.pressure_pa, x=thermo_b.volume_m3))
    np.testing.assert_allclose(computed_work_j, ref.exact_work_j, rtol=1e-6)


def test_case_sensitive_si_prefix_mpa_vs_millipascal() -> None:
    """Ensure MPa (10^6 Pa) and mPa (10^-3 Pa) are never conflated by case folding."""
    csv_mega = "CA[deg], Pressure[MPa], Volume[cm^3], Temp[K]\n0.0, 2.5, 50.0, 800.0\n"
    csv_milli = "CA[deg], Pressure[mPa], Volume[cm^3], Temp[K]\n0.0, 2.5, 50.0, 800.0\n"

    d_mega = parse_thermo_csv(StringIO(csv_mega))
    d_milli = parse_thermo_csv(StringIO(csv_milli))

    assert d_mega.pressure_pa[0] == pytest.approx(2.5e6)
    assert d_milli.pressure_pa[0] == pytest.approx(2.5e-3)


def test_spray_csv_explicit_time_and_rpm_fallback() -> None:
    """Verify spray.csv SI conversion (mm, cm, um, ms, mg) and CA->time RPM fallback."""
    csv_explicit = (
        "Time [ms], Crank_Angle [deg], Liquid_Penetration [mm], "
        "Vapor_Penetration [cm], SMD [µm], Injected_Mass [mg]\n"
        "0.0, -10.0, 0.0, 0.0, 18.5, 0.0\n"
        "0.5, -2.8, 24.5, 3.1, 14.2, 12.4\n"
        "1.0, 4.4, 31.0, 4.8, 11.0, 25.0\n"
    )
    spray = parse_spray_csv(StringIO(csv_explicit))

    np.testing.assert_allclose(spray.time_s, [0.0, 5.0e-4, 1.0e-3], rtol=1e-12)
    np.testing.assert_allclose(spray.liquid_penetration_m, [0.0, 0.0245, 0.031], rtol=1e-12)
    assert spray.vapor_penetration_m is not None
    np.testing.assert_allclose(spray.vapor_penetration_m, [0.0, 0.031, 0.048], rtol=1e-12)
    assert spray.smd_m is not None
    np.testing.assert_allclose(spray.smd_m, [18.5e-6, 14.2e-6, 11.0e-6], rtol=1e-12)
    assert spray.injected_mass_kg is not None
    np.testing.assert_allclose(spray.injected_mass_kg, [0.0, 12.4e-6, 25.0e-6], rtol=1e-12)

    # Fallback: derive elapsed time from Crank_Angle at 2400 RPM (6 * 2400 = 14400 deg/s)
    csv_no_time = "Crank_Angle [deg], Liquid_Penetration [mm]\n-10.0, 0.0\n4.4, 31.0\n"
    spray_ca = parse_spray_csv(StringIO(csv_no_time), rpm=2400.0)
    np.testing.assert_allclose(spray_ca.time_s, [0.0, 1.0e-3], rtol=1e-12)


def test_dynamic_csv_amr_cell_metrics() -> None:
    """Verify dynamic.csv parses AMR cell counts and cell volume/size in SI."""
    csv_dyn = (
        "Crank_Angle [deg], Total_Cells [-], Mean_Cell_Volume [mm^3], Min_Cell_Size [mm]\n"
        "-20.0, 125000, 0.512, 0.50\n"
        "0.0, 480000, 0.064, 0.25\n"
    )
    dyn = parse_dynamic_csv(StringIO(csv_dyn))
    np.testing.assert_allclose(dyn.cell_count, [125_000.0, 480_000.0])
    assert dyn.mean_cell_volume_m3 is not None
    np.testing.assert_allclose(dyn.mean_cell_volume_m3, [0.512e-9, 0.064e-9])
    assert dyn.min_cell_size_m is not None
    np.testing.assert_allclose(dyn.min_cell_size_m, [5.0e-4, 2.5e-4])


@pytest.mark.parametrize(
    ("csv_payload", "expected_exc"),
    [
        ("CA[deg], Pressure, Volume[cm^3], Temp[K]\n0.0, 1.2, 50.0, 700.0\n", UnitDetectionError),
        ("CA[deg], Pressure[cm^3], Volume[cm^3], Temp[K]\n0.0, 1.2, 50.0, 700.0\n", UnitDetectionError),
        ("CA[deg], Pressure[MPa], Volume[cm^3], Temp[degC]\n0.0, 1.2, 50.0, -300.0\n", PhysicalBoundsError),
        ("CA[deg], Pressure[bar], Volume[cm^3], Temp[K]\n0.0, -1.0, 50.0, 300.0\n", PhysicalBoundsError),
        ("CA[deg], Pressure[MPa], Temp[K]\n0.0, 1.2, 700.0\n", ForteParseError),
    ],
)
def test_strict_unit_and_physical_bounds_errors(
    csv_payload: str,
    expected_exc: type[Exception],
) -> None:
    """Ensure missing/invalid units and non-physical thermodynamic states raise immediately."""
    with pytest.raises(expected_exc):
        parse_thermo_csv(StringIO(csv_payload))
