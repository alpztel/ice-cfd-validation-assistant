"""Analytical verification tests for high-altitude Aero-Diesel spray scaling and ECN I/O."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
import pytest

from ice_cfd_validator.exceptions import PhysicalBoundsError, UnitDetectionError
from ice_cfd_validator.io import load_ecn_spray_data, parse_spray_csv
from ice_cfd_validator.spray import (
    compute_spray_validation_metrics,
    evaluate_altitude_spray_risk,
    hiroyasu_arai_penetration,
    naber_siebers_penetration,
)


def test_hiroyasu_arai_breakup_time_continuity() -> None:
    """Test 1: Analytical left/right limit continuity at t = t_b (|S(t_b^-) - S(t_b^+)| < 1e-12 m)."""
    # Open-literature ECN Spray A nominal condition (n-dodecane, 150 MPa rail - 6 MPa ambient)
    delta_p = 144.0e6
    rho_l = 703.0
    rho_g = 22.8
    d_o = 90.0e-6

    probe = hiroyasu_arai_penetration(
        np.array([0.0], dtype=np.float64),
        delta_p_pa=delta_p,
        rho_liquid_kg_m3=rho_l,
        rho_gas_kg_m3=rho_g,
        nozzle_diameter_m=d_o,
    )
    t_b = probe.breakup_time_s
    expected_tb = 28.65 * (rho_l * d_o) / np.sqrt(rho_g * delta_p)
    assert abs(t_b - expected_tb) < 1.0e-18

    # Evaluate immediate IEEE-754 left neighbor t_b^- and exact right limit t_b^+
    t_left = float(np.nextafter(t_b, 0.0))
    t_right = t_b
    res_limits = hiroyasu_arai_penetration(
        np.array([t_left, t_right], dtype=np.float64),
        delta_p_pa=delta_p,
        rho_liquid_kg_m3=rho_l,
        rho_gas_kg_m3=rho_g,
        nozzle_diameter_m=d_o,
    )
    s_left = float(res_limits.penetration_m[0])
    s_right = float(res_limits.penetration_m[1])
    jump_m = abs(s_left - s_right)

    print(
        f"\n[Test 1] t_b = {t_b * 1e6:.4f} us | S(t_b^-) = {s_left:.15f} m | "
        f"S(t_b^+) = {s_right:.15f} m | |fark| = {jump_m:.3e} m"
    )
    assert jump_m < 1.0e-12, f"Discontinuity at t_b: |S(t_b^-) - S(t_b^+)| = {jump_m:.3e} m"
    assert s_left <= s_right  # Strict physical monotonicity dS/dt >= 0


def test_altitude_gas_density_halving_scaling_law() -> None:
    """Test 2: Halving gas density (0.5 * rho_g) scales post-breakup S(t) by exact 2^0.25."""
    delta_p = 120.0e6
    rho_l = 830.0
    rho_g_sl = 24.0
    rho_g_alt = 0.5 * rho_g_sl  # 50% density drop at high altitude
    d_o = 120.0e-6

    # Time vector strictly in post-breakup regime for both densities (0.2 ms to 2.5 ms)
    t_post = np.linspace(2.0e-4, 2.5e-3, 500, dtype=np.float64)

    res_sl = hiroyasu_arai_penetration(
        t_post,
        delta_p_pa=delta_p,
        rho_liquid_kg_m3=rho_l,
        rho_gas_kg_m3=rho_g_sl,
        nozzle_diameter_m=d_o,
    )
    res_alt = hiroyasu_arai_penetration(
        t_post,
        delta_p_pa=delta_p,
        rho_liquid_kg_m3=rho_l,
        rho_gas_kg_m3=rho_g_alt,
        nozzle_diameter_m=d_o,
    )

    assert float(t_post[0]) > res_sl.breakup_time_s
    assert float(t_post[0]) > res_alt.breakup_time_s

    exact_factor = 2.0**0.25  # 1.189207115002721
    ratios = res_alt.penetration_m / res_sl.penetration_m
    print(
        f"\n[Test 2] rho_g: {rho_g_sl} -> {rho_g_alt} kg/m^3 | "
        f"S_alt / S_sl = {float(np.mean(ratios)):.12f} (2^0.25 = {exact_factor:.12f}, "
        f"maks |hata| = {float(np.max(np.abs(ratios - exact_factor))):.3e})"
    )
    np.testing.assert_allclose(ratios, exact_factor, rtol=1.0e-14, atol=1.0e-15)
    assert exact_factor == pytest.approx(1.189207115, abs=1.0e-9)

    risk = evaluate_altitude_spray_risk(
        t_post,
        delta_p_pa=delta_p,
        rho_liquid_kg_m3=rho_l,
        rho_gas_sea_level_kg_m3=rho_g_sl,
        rho_gas_altitude_kg_m3=rho_g_alt,
        nozzle_diameter_m=d_o,
        wall_distance_m=0.050,
    )
    assert risk.asymptotic_penetration_ratio == pytest.approx(exact_factor, rel=1.0e-14)
    assert risk.terminal_penetration_ratio == pytest.approx(exact_factor, rel=1.0e-14)


def test_naber_siebers_implicit_inversion_and_asymptotics() -> None:
    """Verify Naber & Siebers (1996, SAE 960034 Eq. 16) exact inversion and limits."""
    t_s = np.geomspace(1.0e-7, 5.0e-3, 300, dtype=np.float64)
    ns = naber_siebers_penetration(
        t_s,
        delta_p_pa=140.0e6,
        rho_liquid_kg_m3=703.0,
        rho_gas_kg_m3=22.8,
        nozzle_diameter_m=90.0e-6,
        spray_angle_deg=21.0,
        ca=0.90,
        cv=0.88,
        method="exact",
    )

    # Substitute solved s_tilde back into SAE 960034 Eq. (16) and verify residual < 1e-14
    s_t = ns.s_tilde
    root_term = np.sqrt(1.0 + 16.0 * (s_t**2))
    t_tilde_reconstructed = (
        0.5 * s_t + 0.25 * s_t * root_term + (1.0 / 16.0) * np.log(4.0 * s_t + root_term)
    )
    np.testing.assert_allclose(t_tilde_reconstructed, ns.t_tilde, rtol=1.0e-14, atol=1.0e-15)

    # Short-time limit: S(t) -> U_f * t as t -> 0
    assert ns.penetration_m[0] / (ns.u_f_m_s * t_s[0]) == pytest.approx(1.0, rel=1.0e-3)
    # Far-field limit: s_tilde / sqrt(t_tilde) -> 1 - 1/(4*sqrt(t_tilde)) -> 1 as t_tilde -> inf
    assert float(ns.s_tilde[-1] / np.sqrt(ns.t_tilde[-1])) == pytest.approx(1.0, rel=1.0e-2)


def test_altitude_wall_impingement_onset_detection() -> None:
    """Verify high-altitude density drop triggers bowl wall wetting and resolves exact t_imp."""
    t_s = np.linspace(0.0, 6.0e-4, 601, dtype=np.float64)
    wall_dist_m = 0.045  # 45 mm piston bowl radius

    for model_name in ("hiroyasu_arai", "naber_siebers"):
        risk = evaluate_altitude_spray_risk(
            t_s,
            delta_p_pa=110.0e6,
            rho_liquid_kg_m3=825.0,
            rho_gas_sea_level_kg_m3=28.0,
            rho_gas_altitude_kg_m3=11.0,
            nozzle_diameter_m=130.0e-6,
            wall_distance_m=wall_dist_m,
            model=model_name,
            spray_angle_deg=19.5,
        )
        assert risk.wall_impingement_detected
        assert risk.impingement_time_s is not None
        assert risk.sea_level_impingement_time_s is None

        # Verify S_alt(impingement_time_s) == wall_distance_m to machine precision
        if model_name == "hiroyasu_arai":
            check = hiroyasu_arai_penetration(
                [risk.impingement_time_s],
                delta_p_pa=110.0e6,
                rho_liquid_kg_m3=825.0,
                rho_gas_kg_m3=11.0,
                nozzle_diameter_m=130.0e-6,
            ).penetration_m[0]
        else:
            check = naber_siebers_penetration(
                [risk.impingement_time_s],
                delta_p_pa=110.0e6,
                rho_liquid_kg_m3=825.0,
                rho_gas_kg_m3=11.0,
                nozzle_diameter_m=130.0e-6,
                spray_angle_deg=19.5,
            ).penetration_m[0]
        assert float(check) == pytest.approx(wall_dist_m, rel=1.0e-13)


def test_ecn_loader_and_cfd_spray_validation_metrics(tmp_path: Path) -> None:
    """Verify ECN Spray A/D loader SI normalization and CFD vs ECN RMSE/MAPE/L2 metrics."""
    ecn_txt = (
        "% Sandia Engine Combustion Network - Spray A Benchmark\n"
        "% Ambient: 900 K, 22.8 kg/m^3, Nozzle: 90 um\n"
        "time_asoi   vapor_penetration   liquid_length   vapor_std\n"
        "[ms]        [mm]                [mm]            [mm]\n"
        "-0.05       0.00                0.00            0.00\n"
        "0.00        0.00                0.00            0.05\n"
        "0.50        32.00               10.20           0.85\n"
        "1.00        48.00               10.40           1.10\n"
        "1.50        60.00               10.35           1.30\n"
    )
    ecn_path = tmp_path / "ECN_SprayA_nominal.txt"
    ecn_path.write_text(ecn_txt, encoding="utf-8")

    ecn = load_ecn_spray_data(ecn_path)
    assert ecn.condition_label == "ECN_SprayA_nominal"
    np.testing.assert_allclose(ecn.time_s, [0.0, 5.0e-4, 1.0e-3, 1.5e-3], rtol=1e-14)
    assert ecn.vapor_penetration_m is not None
    assert ecn.liquid_penetration_m is not None
    assert ecn.vapor_std_m is not None
    np.testing.assert_allclose(ecn.vapor_penetration_m, [0.0, 0.032, 0.048, 0.060], rtol=1e-14)
    np.testing.assert_allclose(ecn.liquid_penetration_m, [0.0, 0.0102, 0.0104, 0.01035], rtol=1e-14)
    np.testing.assert_allclose(ecn.vapor_std_m, [0.00005, 0.00085, 0.0011, 0.0013], rtol=1e-14)

    # ANSYS Forte spray.csv with uniform +5% vapor bias on finer asynchronous grid
    t_cfd_ms = np.linspace(0.0, 1.5, 31, dtype=np.float64)
    s_vap_ref_mm = np.interp(t_cfd_ms, [0.0, 0.5, 1.0, 1.5], [0.0, 32.0, 48.0, 60.0])
    s_vap_cfd_mm = 1.05 * s_vap_ref_mm
    s_liq_cfd_mm = np.full_like(t_cfd_ms, 10.3)

    forte_lines = ["Time[ms],Liquid_Penetration[mm],Vapor_Penetration[mm]"] + [
        f"{tm:.4f},{sl:.4f},{sv:.4f}"
        for tm, sl, sv in zip(t_cfd_ms, s_liq_cfd_mm, s_vap_cfd_mm, strict=True)
    ]
    forte_spray = parse_spray_csv(StringIO("\n".join(forte_lines)))
    assert forte_spray.vapor_penetration_m is not None

    metrics = compute_spray_validation_metrics(
        forte_spray.vapor_penetration_m,
        ecn.vapor_penetration_m,
        cfd_time_s=forte_spray.time_s,
        ref_time_s=ecn.time_s,
    )
    assert metrics.n_samples == 4
    assert metrics.mape_pct == pytest.approx(5.0, rel=1.0e-4)
    assert metrics.l2_relative_error == pytest.approx(0.05, rel=1.0e-4)
    expected_rmse = float(np.sqrt(np.mean((0.05 * np.array([0.0, 0.032, 0.048, 0.060])) ** 2)))
    assert metrics.rmse_m == pytest.approx(expected_rmse, rel=1.0e-4)

    # Missing unit without fallback must raise UnitDetectionError; negative penetration raises PhysicalBoundsError
    with pytest.raises(UnitDetectionError):
        load_ecn_spray_data(StringIO("time, vapor_penetration\n0.0, 0.0\n1.0, 25.0\n"))
    with pytest.raises(PhysicalBoundsError):
        hiroyasu_arai_penetration(
            [0.001], delta_p_pa=-10.0, rho_liquid_kg_m3=800.0, rho_gas_kg_m3=20.0, nozzle_diameter_m=1e-4
        )
