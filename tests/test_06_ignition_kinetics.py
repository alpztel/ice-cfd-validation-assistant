"""Comprehensive unit and strict audit tests for ignition/ package."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from ice_cfd_validator.constants import R_UNIVERSAL
from ice_cfd_validator.exceptions import PhysicalBoundsError
from ice_cfd_validator.ignition import (
    ASSANIS_2003,
    HH_ACTIVATION_NUMERATOR_J_PER_MOL,
    PREDEFINED_FUELS,
    WOLFER_1938,
    arrhenius_state_delay,
    compute_apparent_activation_energy,
    detect_two_stage_ignition,
    extract_cfd_ignition_metrics,
    hardenberg_hase_delay,
    livengood_wu_integral,
    solve_livengood_wu,
)
from tests.conftest import assert_package_vectorized_and_typed


@pytest.mark.unit
class TestUnitIgnitionKinetics:
    """Unit tests for Livengood-Wu solver, auto-ignition kinetics, and CFD diagnostics."""

    def test_1_isobaric_isothermal_livengood_wu_exact_root(self) -> None:
        """TEST 1: Verify Livengood-Wu solver recovers the exact analytical root on a coarse grid."""
        p0 = 4.0e6
        t0 = 850.0
        rpm = 2400.0
        theta_soi = -10.0

        tau_0 = float(arrhenius_state_delay(p0, t0, 1.0, coefficients=ASSANIS_2003))
        assert tau_0 > 0.0

        deg_per_s = 6.0 * rpm
        theta_soc_exact = theta_soi + deg_per_s * tau_0

        theta_grid = np.arange(-25.0, 30.5, 0.5, dtype=np.float64)
        nearest_node = float(theta_grid[np.argmin(np.abs(theta_grid - theta_soc_exact))])
        node_distance = abs(theta_soc_exact - nearest_node)
        assert node_distance > 1.0e-4

        tau_arr = np.full_like(theta_grid, tau_0)
        res = solve_livengood_wu(theta_grid, tau_arr, rpm, theta_soi, root_interpolator="hermite")

        assert not res.misfire_detected
        error = abs(res.theta_soc_deg - theta_soc_exact)
        assert error < 1.0e-11

        p_arr = np.full_like(theta_grid, p0)
        t_arr = np.full_like(theta_grid, t0)
        res_dyn = livengood_wu_integral(
            theta_grid, p_arr, t_arr, rpm, theta_soi, coefficients=ASSANIS_2003
        )
        error_dyn = abs(res_dyn.theta_soc_deg - theta_soc_exact)
        assert error_dyn < 1.0e-11

    def test_2_exponential_delay_closed_form_integral(self) -> None:
        """TEST 2: Verify non-linear closed-form analytical Livengood-Wu root recovery."""
        tau_0 = 3.5e-3
        alpha = 0.12
        rpm = 2400.0
        theta_soi = -12.34

        deg_per_s = 6.0 * rpm
        theta_soc_exact = theta_soi + (1.0 / alpha) * np.log(1.0 + deg_per_s * tau_0 * alpha)

        theta_grid = np.arange(-30.0, 30.2, 0.2, dtype=np.float64)
        tau_grid = tau_0 * np.exp(-alpha * (theta_grid - theta_soi))

        res = solve_livengood_wu(theta_grid, tau_grid, rpm, theta_soi, root_interpolator="hermite")

        assert not res.misfire_detected
        error = abs(res.theta_soc_deg - theta_soc_exact)
        assert error < 1.0e-5

    def test_3_hardenberg_hase_units_fuel_and_singularity_guard(self) -> None:
        """TEST 3: Verify Hardenberg-Hase activation energy, fuel trend, and misfire guard."""
        ea_50 = compute_apparent_activation_energy(50.0)
        expected_ea_50 = HH_ACTIVATION_NUMERATOR_J_PER_MOL / 75.0
        assert ea_50 == pytest.approx(expected_ea_50, rel=1.0e-12)

        t_tdc = 900.0
        p_tdc = 6.0e6
        sp = 7.2
        rpm = 2400.0

        res_jet = hardenberg_hase_delay(t_tdc, p_tdc, sp, rpm, cetane_number=PREDEFINED_FUELS["JET_A1_JP8"].cetane_number)
        res_dsl = hardenberg_hase_delay(t_tdc, p_tdc, sp, rpm, cetane_number=PREDEFINED_FUELS["EN590_DIESEL"].cetane_number)

        tau_jet_s = float(res_jet.ignition_delay_s)
        tau_dsl_s = float(res_dsl.ignition_delay_s)
        assert tau_jet_s > tau_dsl_s
        assert res_jet.activation_energy_j_per_mol > res_dsl.activation_energy_j_per_mol

        res_jet_add = hardenberg_hase_delay(
            t_tdc, p_tdc, sp, rpm, cetane_number=42.0, pressure_coupling="additive"
        )
        res_dsl_add = hardenberg_hase_delay(
            t_tdc, p_tdc, sp, rpm, cetane_number=52.0, pressure_coupling="additive"
        )
        assert float(res_jet_add.ignition_delay_s) > float(res_dsl_add.ignition_delay_s)

        p_subcritical = 1.10e6
        with pytest.raises(ValueError, match="1.24 MPa physical auto-ignition threshold"):
            hardenberg_hase_delay(t_tdc, p_subcritical, sp, rpm, cetane_number=50.0)

        with pytest.raises(PhysicalBoundsError):
            compute_apparent_activation_energy(-5.0)
        with pytest.raises(PhysicalBoundsError):
            hardenberg_hase_delay(-300.0, p_tdc, sp, rpm, cetane_number=50.0)

    def test_4_evaporative_cooling_and_two_stage_ahrr_detection(self) -> None:
        """TEST 4: Verify evaporative cooling zero-crossing, Q_evap energy, and LTHR/HTHR separation."""
        theta_soi = -15.0
        theta_soc_target = -5.0
        q_evap_exact = 15.0
        a_evap = q_evap_exact * np.pi / 20.0

        theta = np.linspace(-25.0, 45.0, 1401, dtype=np.float64)
        dip_mask = (theta >= theta_soi) & (theta < theta_soc_target)
        dq_evap = np.where(dip_mask, -a_evap * np.sin(np.pi * (theta - theta_soi) / 10.0), 0.0)

        t_diff_l = np.maximum(0.0, theta - theta_soc_target)
        t_diff_h = np.maximum(0.0, theta - 1.0)
        lthr_wiebe = np.where(
            theta >= theta_soc_target,
            15.0 * (t_diff_l / 3.0) ** 2 * np.exp(-2.0 * (t_diff_l / 3.0)),
            0.0,
        )
        hthr_wiebe = np.where(
            theta >= 1.0,
            50.0 * (t_diff_h / 10.0) ** 2 * np.exp(-2.0 * (t_diff_h / 10.0)),
            0.0,
        )
        dq_pos = lthr_wiebe + hthr_wiebe
        dq_net = dq_evap + dq_pos

        v_ratio = 1.0 + 15.0 * np.sin(np.radians(np.clip(theta + 180.0, 0.0, 360.0) * 0.5)) ** 2
        p_motored = 2.0e5 * (16.5 / v_ratio) ** 1.35
        q_cum = np.cumsum(np.maximum(dq_pos, 0.0)) * (theta[1] - theta[0])
        p_pa = p_motored + q_cum * 4.0e4
        temp_k = 620.0 + 8.0 * (theta - theta_soi) + q_cum * 0.8

        metrics = extract_cfd_ignition_metrics(
            theta, dq_net, p_pa, theta_soi_deg=theta_soi, temperature_k=temp_k
        )

        assert metrics.evaporative_dip_detected
        assert abs(metrics.theta_soc_ahrr0_deg - theta_soc_target) < 1.0e-4
        assert abs(metrics.evaporative_cooling_energy_j - q_evap_exact) < 1.0e-3
        assert metrics.two_stage.two_stage_detected
        assert -3.0 < metrics.two_stage.theta_lthr_peak_deg < 0.0
        assert 0.0 < metrics.two_stage.theta_ntc_deg < 3.0
        assert 9.0 < metrics.two_stage.theta_hthr_peak_deg < 13.0

    def test_5_high_altitude_misfire_and_ast_vectorization_audit(self) -> None:
        """TEST 5: Verify misfire detection under sub-critical altitude conditions."""
        theta_grid = np.linspace(-20.0, 30.0, 251, dtype=np.float64)
        p_cold = np.full_like(theta_grid, 40_000.0)
        t_cold = np.full_like(theta_grid, 450.0)
        rpm = 2400.0
        theta_soi = -10.0

        res_misfire = livengood_wu_integral(
            theta_grid, p_cold, t_cold, rpm, theta_soi, coefficients=ASSANIS_2003
        )

        assert res_misfire.misfire_detected
        assert res_misfire.max_integral_value < 1.0
        assert np.isnan(res_misfire.theta_soc_deg)
        assert np.isnan(res_misfire.ignition_delay_deg)
        assert np.isnan(res_misfire.ignition_delay_s)


@pytest.mark.strict_audit
class TestStrictAuditIgnitionKinetics:
    """Strict analytical audits for ignition kinetics, Arrhenius scaling, and subgrid interpolation."""

    def test_audit_1_hardenberg_hase_exact_benchmark_and_singularity(self) -> None:
        """DENETİM 1: Exact analytical hand-calculation benchmark and singularity guard."""
        cn = 50.0
        ea_expected = HH_ACTIVATION_NUMERATOR_J_PER_MOL / (cn + 25.0)
        ea_calc = compute_apparent_activation_energy(cn)
        assert ea_calc == pytest.approx(ea_expected, rel=1.0e-12)

        t_tdc = 17190.0 / R_UNIVERSAL
        p_tdc = 3.36e6
        sp = 10.0
        rpm = 2400.0

        expected_delay_deg = 2.56 * np.e
        expected_delay_s = expected_delay_deg / (6.0 * rpm)

        res = hardenberg_hase_delay(t_tdc, p_tdc, sp, rpm, cn)
        delay_deg_num = float(res.ignition_delay_deg)
        delay_s_num = float(res.ignition_delay_s)

        rel_err_deg = abs(delay_deg_num - expected_delay_deg) / expected_delay_deg
        rel_err_s = abs(delay_s_num - expected_delay_s) / expected_delay_s

        assert rel_err_deg < 1.0e-11
        assert rel_err_s < 1.0e-11

        p_real = 4.5e6
        t_real = 880.0
        sp_real = 9.5

        res_jet = hardenberg_hase_delay(t_real, p_real, sp_real, rpm, PREDEFINED_FUELS["JET_A1_JP8"].cetane_number)
        res_dsl = hardenberg_hase_delay(t_real, p_real, sp_real, rpm, PREDEFINED_FUELS["EN590_DIESEL"].cetane_number)

        tau_jet = float(res_jet.ignition_delay_s)
        tau_dsl = float(res_dsl.ignition_delay_s)
        rel_diff = (tau_jet - tau_dsl) / tau_dsl

        assert tau_jet > tau_dsl
        assert rel_diff >= 0.095

        with pytest.raises(ValueError, match="1.24 MPa physical auto-ignition threshold"):
            hardenberg_hase_delay(t_real, 1.24e6, sp_real, rpm, 50.0)

        with pytest.raises(ValueError, match="1.24 MPa physical auto-ignition threshold"):
            hardenberg_hase_delay(t_real, 1.0e6, sp_real, rpm, 50.0)

    def test_audit_2_arrhenius_pressure_scaling_and_si_units(self) -> None:
        """DENETİM 2: Verify tau(P1)/tau(P2) == 2^n and output in seconds."""
        p1 = 3.0e6
        p2 = 6.0e6
        temp = 900.0
        phi = 1.0

        tau_w1 = float(arrhenius_state_delay(p1, temp, phi, coefficients=WOLFER_1938))
        tau_w2 = float(arrhenius_state_delay(p2, temp, phi, coefficients=WOLFER_1938))
        ratio_wolfer = tau_w1 / tau_w2
        expected_ratio_wolfer = 2.0 ** WOLFER_1938.pressure_exponent_n
        err_w = abs(ratio_wolfer - expected_ratio_wolfer) / expected_ratio_wolfer

        tau_a1 = float(arrhenius_state_delay(p1, temp, phi, coefficients=ASSANIS_2003))
        tau_a2 = float(arrhenius_state_delay(p2, temp, phi, coefficients=ASSANIS_2003))
        ratio_assanis = tau_a1 / tau_a2
        expected_ratio_assanis = 2.0 ** ASSANIS_2003.pressure_exponent_n
        err_a = abs(ratio_assanis - expected_ratio_assanis) / expected_ratio_assanis

        assert err_w < 1.0e-12
        assert err_a < 1.0e-12
        assert 1.0e-5 < tau_w1 < 0.1
        assert 1.0e-5 < tau_a1 < 0.1

    def test_audit_3_livengood_wu_subgrid_alignment_and_nonlinear_integral(self) -> None:
        """DENETİM 3: Verify sub-grid SOI alignment on coarse grid and non-linear integral."""
        theta_coarse = np.arange(-30.0, 31.0, 1.0, dtype=np.float64)
        theta_soi = -14.37
        tau_0 = 1.25e-3
        rpm = 2000.0
        theta_soc_exact = theta_soi + 15.000000

        tau_arr = np.full_like(theta_coarse, tau_0)
        res_a = solve_livengood_wu(theta_coarse, tau_arr, rpm, theta_soi)
        err_soc_a = abs(res_a.theta_soc_deg - theta_soc_exact)
        assert err_soc_a < 1.0e-9

        tau_0_b = 3.5e-3
        alpha = 0.12
        rpm_b = 2400.0
        theta_soi_b = -12.34
        theta_soc_exact_b = theta_soi_b + (1.0 / alpha) * np.log(1.0 + 6.0 * rpm_b * tau_0_b * alpha)

        theta_grid_b = np.arange(-30.0, 30.2, 0.2, dtype=np.float64)
        tau_grid_b = tau_0_b * np.exp(-alpha * (theta_grid_b - theta_soi_b))

        res_b = solve_livengood_wu(theta_grid_b, tau_grid_b, rpm_b, theta_soi_b)
        err_soc_b = abs(res_b.theta_soc_deg - theta_soc_exact_b)
        assert err_soc_b < 1.0e-5

    def test_audit_4_evaporative_cooling_integral_and_criteria_chronology(self) -> None:
        """DENETİM 4: Analytical evaporative cooling sine integral and chronological SOC criteria."""
        theta_soi = -12.25
        delta_theta_evap = 8.0
        theta_soc_exact = theta_soi + delta_theta_evap
        q_evap_exact = 18.500
        a_evap = q_evap_exact * np.pi / (2.0 * delta_theta_evap)

        theta = np.linspace(-25.0, 40.0, 1301, dtype=np.float64)
        dip_mask = (theta >= theta_soi) & (theta <= theta_soc_exact)
        dq_evap = np.where(
            dip_mask,
            -a_evap * np.sin(np.pi * (theta - theta_soi) / delta_theta_evap),
            0.0,
        )

        q_comb = 950.0
        delta_theta_comb = 35.0
        m_wiebe = 2.0
        a_wiebe = 5.0

        t_comb = np.maximum(0.0, theta - theta_soc_exact)
        phi_norm = t_comb / delta_theta_comb
        dq_comb = np.where(
            theta >= theta_soc_exact,
            q_comb
            * a_wiebe
            * (m_wiebe + 1.0)
            / delta_theta_comb
            * (phi_norm ** m_wiebe)
            * np.exp(-a_wiebe * (phi_norm ** (m_wiebe + 1.0))),
            0.0,
        )

        dq_net = dq_evap + dq_comb
        p_pa = 3.0e6 * np.exp(-0.5 * ((theta - 5.0) / 12.0) ** 2) + 1.0e5
        temp_k = 600.0 + 800.0 * (p_pa / np.max(p_pa))

        metrics = extract_cfd_ignition_metrics(
            theta, dq_net, p_pa, theta_soi_deg=theta_soi, temperature_k=temp_k
        )

        err_soc = abs(metrics.theta_soc_ahrr0_deg - theta_soc_exact)
        err_qevap = abs(metrics.evaporative_cooling_energy_j - q_evap_exact)

        assert err_soc < 1.0e-3
        assert err_qevap < 1.0e-2
        assert theta_soi < metrics.theta_soc_ahrr0_deg
        assert metrics.theta_soc_ahrr0_deg < metrics.theta_ca01_deg
        assert metrics.theta_ca01_deg < metrics.theta_ca02_deg
        assert metrics.theta_ca02_deg < metrics.theta_ca05_deg

    def test_audit_5_two_stage_ignition_and_false_positive_immunity(self) -> None:
        """DENETİM 5: Two-stage cool flame detection and single-stage false-positive prevention."""
        theta = np.linspace(-20.0, 30.0, 1001, dtype=np.float64)

        lthr_peak_target = -9.5
        hthr_peak_target = 6.0
        dq_lthr = 8.0 * np.exp(-0.5 * ((theta - lthr_peak_target) / 2.0) ** 2)
        dq_hthr = 65.0 * np.exp(-0.5 * ((theta - hthr_peak_target) / 3.5) ** 2)
        dq_two_stage = dq_lthr + dq_hthr
        temp_two_stage = 580.0 + 8.0 * (theta + 20.0)

        res_two_stage = detect_two_stage_ignition(
            theta, dq_two_stage, temp_two_stage, theta_start_deg=-15.0
        )

        assert res_two_stage.two_stage_detected
        assert abs(res_two_stage.theta_lthr_peak_deg - lthr_peak_target) < 0.5
        assert abs(res_two_stage.theta_hthr_peak_deg - hthr_peak_target) < 0.5
        assert lthr_peak_target < res_two_stage.theta_ntc_deg < hthr_peak_target

        t_single = np.maximum(0.0, theta + 5.0)
        dq_single_stage = np.where(
            theta >= -5.0,
            50.0 * (t_single / 8.0) ** 2 * np.exp(-2.0 * (t_single / 8.0)),
            0.0,
        )
        temp_single = 600.0 + 30.0 * t_single

        res_single = detect_two_stage_ignition(
            theta, dq_single_stage, temp_single, theta_start_deg=-15.0
        )

        assert not res_single.two_stage_detected
        assert np.isnan(res_single.theta_lthr_peak_deg)
        assert np.isnan(res_single.theta_ntc_deg)

    def test_audit_ignition_vectorization_and_typing(self) -> None:
        """AUDIT: AST scan confirms zero explicit loops and strict typing in ignition/ package."""
        assert_package_vectorized_and_typed("ignition")
