"""Verification tests for ISO 2533 ISA atmosphere and 300-DPI SAE/IEEE plotting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ice_cfd_validator.constants import (
    ISA_P0_PA,
    ISA_T0_K,
    ISA_T_TROPOPAUSE_K,
    R_SPECIFIC_DRY_AIR,
    R_UNIVERSAL,
    compute_isa_conditions,
)
from ice_cfd_validator.exceptions import PhysicalBoundsError
from ice_cfd_validator.plotting.publication_plots import (
    AltitudeSprayPlotCase,
    plot_altitude_spray_comparison,
    plot_gci_convergence,
    plot_pv_log_diagram,
    plot_thermo_diagnostics,
)
from ice_cfd_validator.spray.correlations import (
    hiroyasu_arai_penetration,
    naber_siebers_penetration,
)
from ice_cfd_validator.thermo.heat_release import (
    compute_heat_release,
    extract_burn_fractions,
)
from ice_cfd_validator.thermo.kinematics import SliderCrankGeometry
from ice_cfd_validator.vv.gci import compute_gci


def test_isa_iso2533_troposphere_tropopause_and_stratosphere() -> None:
    """Verify ISA state at 0 m, 11,000 m continuity, and 15,000 m (~49,213 ft)."""
    assert R_UNIVERSAL == pytest.approx(8.314462618, rel=1.0e-12)

    altitudes_m = np.array([0.0, 4572.0, 9144.0, 10_999.999, 11_000.001, 15_000.0])
    isa = compute_isa_conditions(altitudes_m)

    # Sea-level standard state (0 m)
    assert isa.temperature_k[0] == pytest.approx(ISA_T0_K, abs=1.0e-12)
    assert isa.pressure_pa[0] == pytest.approx(ISA_P0_PA, abs=1.0e-9)
    assert isa.density_kg_m3[0] == pytest.approx(1.225, rel=1.0e-3)

    # Tropopause continuity across z = 11,000 m
    assert abs(float(isa.temperature_k[3] - isa.temperature_k[4])) < 1.0e-4
    assert abs(float(isa.pressure_pa[3] - isa.pressure_pa[4])) < 1.0e-2
    assert abs(float(isa.density_kg_m3[3] - isa.density_kg_m3[4])) < 1.0e-6

    # Lower stratosphere at 15,000 m (~49,213 ft): isothermal T = 216.65 K, P ~ 12,044.6 Pa
    assert isa.temperature_k[5] == pytest.approx(ISA_T_TROPOPAUSE_K, abs=1.0e-12)
    assert isa.pressure_pa[5] == pytest.approx(12_044.5, rel=5.0e-4)

    # Ideal-gas consistency everywhere: P == rho * R_air * T
    np.testing.assert_allclose(
        isa.pressure_pa,
        isa.density_kg_m3 * R_SPECIFIC_DRY_AIR * isa.temperature_k,
        rtol=1.0e-14,
    )

    with pytest.raises(PhysicalBoundsError):
        compute_isa_conditions(25_000.0)


def test_publication_plots_render_300dpi_png(tmp_path: Path) -> None:
    """Verify all four SAE/IEEE plotting routines export non-empty 300-DPI PNGs."""
    geom = SliderCrankGeometry(
        bore_m=0.085,
        stroke_m=0.090,
        conrod_m=0.155,
        compression_ratio=16.5,
    )
    theta = np.linspace(-180.0, 180.0, 721, dtype=np.float64)
    kin = geom.evaluate(theta)

    # Synthetic firing-like pressure trace (polytropic + Gaussian combustion bell)
    p_mot = 1.8e5 * ((float(kin.volume_m3[0]) / kin.volume_m3) ** 1.35)
    comb_bump = 3.2e6 * np.exp(-0.5 * ((theta - 12.0) / 14.0) ** 2) * (theta > -10.0)
    pressure_pa = p_mot + comb_bump

    hrr = compute_heat_release(
        theta,
        pressure_pa,
        kin.volume_m3,
        dv_dtheta_m3_per_deg=kin.dv_dtheta_m3_per_deg,
        gamma=1.35,
        dq_ht_j_per_deg=np.full_like(theta, 1.5),
    )
    burn = extract_burn_fractions(theta, hrr.dq_net_j_per_deg, start_deg=-10.0, end_deg=55.0)

    p1 = tmp_path / "thermo.png"
    p2 = tmp_path / "pv_log.png"
    p3 = tmp_path / "spray.png"
    p4 = tmp_path / "gci.png"

    plot_thermo_diagnostics(theta, pressure_pa, hrr, burn, save_path=p1, dpi=300)
    plot_pv_log_diagram(kin.volume_m3, pressure_pa, theta, save_path=p2, dpi=300)

    t_s = np.linspace(1.0e-6, 0.8e-3, 120, dtype=np.float64)
    ha = hiroyasu_arai_penetration(
        t_s,
        delta_p_pa=100.0e6,
        rho_liquid_kg_m3=810.0,
        rho_gas_kg_m3=14.0,
        nozzle_diameter_m=90.0e-6,
    )
    ns = naber_siebers_penetration(
        t_s,
        delta_p_pa=100.0e6,
        rho_liquid_kg_m3=810.0,
        rho_gas_kg_m3=14.0,
        nozzle_diameter_m=90.0e-6,
        spray_angle_deg=19.0,
    )
    case = AltitudeSprayPlotCase(
        label="15,000 ft",
        rho_gas_kg_m3=14.0,
        time_s=t_s,
        hiroyasu_penetration_m=ha.penetration_m,
        siebers_penetration_m=ns.penetration_m,
        cfd_time_s=t_s[::12],
        cfd_penetration_m=ha.penetration_m[::12],
        impingement_time_s=0.68e-3,
    )
    plot_altitude_spray_comparison([case], wall_distance_m=0.038, save_path=p3, dpi=300)

    gci = compute_gci(10.0 + 2.5 * 1.0, 10.0 + 2.5 * 2.25, 10.0 + 2.5 * 5.0625, r21=1.5, r32=1.5)
    plot_gci_convergence(gci, save_path=p4, dpi=300)

    for png_file in (p1, p2, p3, p4):
        assert png_file.exists()
        assert png_file.stat().st_size > 15_000
        # Verify valid PNG 8-byte magic signature
        assert png_file.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
