"""Comprehensive unit and strict audit tests for plotting/ and end-to-end pipeline."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ice_cfd_validator.plotting import (
    AltitudeSprayPlotCase,
    plot_altitude_spray_comparison,
    plot_gci_convergence,
    plot_pv_log_diagram,
    plot_thermo_diagnostics,
)
from ice_cfd_validator.spray import (
    hiroyasu_arai_penetration,
    naber_siebers_penetration,
)
from ice_cfd_validator.thermo import (
    SliderCrankGeometry,
    compute_heat_release,
    extract_burn_fractions,
)
from ice_cfd_validator.vv import compute_gci
from tests.conftest import assert_package_vectorized_and_typed

matplotlib.use("Agg")


@pytest.mark.unit
class TestUnitPlottingAndPipeline:
    """Unit tests for headless 300-DPI publication visualizers."""

    def test_publication_plots_render_300dpi_png(self, tmp_path: Path) -> None:
        """Verify all four SAE/IEEE plotting routines export non-empty 300-DPI PNGs."""
        geom = SliderCrankGeometry(
            bore_m=0.085,
            stroke_m=0.090,
            conrod_m=0.155,
            compression_ratio=16.5,
        )
        theta = np.linspace(-180.0, 180.0, 721, dtype=np.float64)
        kin = geom.evaluate(theta)

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
            assert png_file.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.strict_audit
class TestStrictAuditPlottingAndPipeline:
    """Strict audits for headless canvas, memory leaks, and publication quality."""

    def test_audit_headless_plotting_memory_leak_and_demo_execution(self, tmp_path: Path) -> None:
        """DENETIM 5: Agg headless backend, zero leaked pyplot figures & end-to-end demo PNG audit."""
        plt.close("all")
        figs_before = len(plt.get_fignums())

        geom = SliderCrankGeometry(bore_m=0.085, stroke_m=0.090, conrod_m=0.155, compression_ratio=16.5)
        theta = np.linspace(-180.0, 180.0, 721, dtype=np.float64)
        kin = geom.evaluate(theta)
        p_pa = 1.8e5 * ((float(kin.volume_m3[0]) / kin.volume_m3) ** 1.35) + 3.0e6 * np.exp(
            -0.5 * ((theta - 10.0) / 12.0) ** 2
        ) * (theta > -10.0)
        hrr = compute_heat_release(
            theta,
            p_pa,
            kin.volume_m3,
            gamma=1.35,
            dv_dtheta_m3_per_deg=kin.dv_dtheta_m3_per_deg,
            dq_ht_j_per_deg=np.full_like(theta, 1.2),
        )
        burn = extract_burn_fractions(theta, hrr.dq_gross_j_per_deg, start_deg=-10.0, end_deg=55.0)

        t_s = np.linspace(1.0e-6, 0.8e-3, 100, dtype=np.float64)
        ha = hiroyasu_arai_penetration(
            t_s, delta_p_pa=1.0e8, rho_liquid_kg_m3=810.0, rho_gas_kg_m3=15.0, nozzle_diameter_m=9.0e-5
        )
        ns = naber_siebers_penetration(
            t_s,
            delta_p_pa=1.0e8,
            rho_liquid_kg_m3=810.0,
            rho_gas_kg_m3=15.0,
            nozzle_diameter_m=9.0e-5,
            spray_angle_deg=19.0,
        )
        spray_case = AltitudeSprayPlotCase(
            label="Audit Case",
            rho_gas_kg_m3=15.0,
            time_s=t_s,
            hiroyasu_penetration_m=ha.penetration_m,
            siebers_penetration_m=ns.penetration_m,
        )
        gci = compute_gci(12.5, 15.625, 22.65625, h1=1.0, h2=1.5, h3=2.25)

        plot_thermo_diagnostics(theta, p_pa, hrr, burn, save_path=tmp_path / "f1.png")
        plot_pv_log_diagram(kin.volume_m3, p_pa, theta, save_path=tmp_path / "f2.png")
        plot_altitude_spray_comparison([spray_case], wall_distance_m=0.038, save_path=tmp_path / "f3.png")
        plot_gci_convergence(gci, save_path=tmp_path / "f4.png")

        figs_after = len(plt.get_fignums())

        project_root = Path(__file__).resolve().parents[1]
        demo_script = project_root / "examples" / "altitude_sweep_demo.py"
        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"
        proc = subprocess.run(
            [sys.executable, str(demo_script)],
            cwd=str(project_root),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, f"altitude_sweep_demo.py failed:\n{proc.stderr}"

        expected_assets = (
            "thermo_diagnostics.png",
            "pv_log_diagram.png",
            "altitude_spray_comparison.png",
            "gci_convergence.png",
        )
        asset_info: list[tuple[str, float, bool]] = []
        for fname in expected_assets:
            fpath = project_root / "assets" / fname
            assert fpath.exists(), f"Missing generated asset: {fpath}"
            raw = fpath.read_bytes()
            has_png_magic = raw[:8] == b"\x89PNG\r\n\x1a\n"
            size_kb = len(raw) / 1024.0
            asset_info.append((fname, size_kb, has_png_magic))

        assert figs_after == 0
        for fname, size_kb, magic_ok in asset_info:
            assert magic_ok, f"Invalid PNG header in {fname}"
            assert size_kb > 30.0, f"Expected > 30 KB for 300-DPI figure {fname}, got {size_kb:.1f} KB"

    def test_audit_plotting_vectorization_and_typing(self) -> None:
        """AUDIT: AST scan confirms zero explicit loops and strict typing in plotting/ package."""
        assert_package_vectorized_and_typed("plotting")
