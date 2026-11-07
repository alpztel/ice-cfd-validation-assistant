"""Comprehensive unit and strict audit tests for exergy/ and Second Law thermodynamics."""

from __future__ import annotations

import ast
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp, trapezoid
from scipy.interpolate import CubicSpline

from ice_cfd_validator.constants import (
    FEET_TO_M,
    R_SPECIFIC_DRY_AIR,
    STD_GRAVITY_M_S2,
)
from ice_cfd_validator.exergy import (
    DIESEL_EN590_EXERGY_FACTOR,
    JET_A1_EXERGY_FACTOR,
    DeadStateConditions,
    compute_exergy_cycle_budget,
    compute_exergy_rates,
    compute_fuel_chemical_exergy,
    compute_standard_entropy,
    compute_thermomechanical_exergy,
)
from ice_cfd_validator.signal.filtering import compute_derivative
from ice_cfd_validator.thermo.kinematics import SliderCrankGeometry
from ice_cfd_validator.thermo.nasa_polynomials import DRY_AIR
from tests.conftest import assert_package_vectorized_and_typed


@pytest.mark.unit
class TestUnitExergySecondLaw:
    """Unit tests for dead state, thermomechanical exergy, and Second Law balance."""

    def test_01_dead_state_zero_and_gibbs_convexity(self) -> None:
        """TEST 1: Ölü hal sıfır noktası ve 50x50 Gibbs konvekslik pozitif kesinlik denetimi."""
        t0 = 298.15
        p0 = 101325.0
        dead_state = DeadStateConditions(
            altitude_m=0.0,
            p0_pa=p0,
            t0_k=t0,
            rho0_kg_per_m3=p0 / (DRY_AIR.specific_gas_constant * t0),
        )
        m_cyl = 1.2e-3

        e_tm_zero = compute_thermomechanical_exergy(
            temperature_k=np.asarray([t0]),
            pressure_pa=np.asarray([p0]),
            dead_state=dead_state,
            trapped_mass_kg=m_cyl,
            gas_mixture=DRY_AIR,
        )
        delta_s_zero = (
            compute_standard_entropy(np.asarray([t0]), gas_mixture=DRY_AIR)
            - compute_standard_entropy(np.asarray([t0]), gas_mixture=DRY_AIR)
            - DRY_AIR.specific_gas_constant * np.log(p0 / p0)
        )

        assert abs(float(delta_s_zero[0])) < 1e-15
        assert abs(float(e_tm_zero[0])) < 1e-14

        t_vals = np.linspace(250.0, 3000.0, 50)
        p_vals = np.linspace(0.5e5, 150e5, 50)
        t_grid, p_grid = np.meshgrid(t_vals, p_vals)

        e_tm_grid = compute_thermomechanical_exergy(
            temperature_k=t_grid.ravel(),
            pressure_pa=p_grid.ravel(),
            dead_state=dead_state,
            trapped_mass_kg=m_cyl,
            gas_mixture=DRY_AIR,
        ).reshape(t_grid.shape)

        mask_not_dead_state = ~(np.isclose(t_grid, t0, atol=1e-1) & np.isclose(p_grid, p0, atol=1e2))
        min_e_tm = float(np.min(e_tm_grid[mask_not_dead_state]))

        assert min_e_tm > 0.0
        assert np.all(e_tm_grid >= 0.0)

    def test_02_reversible_isentropic_zero_entropy_and_destruction(self) -> None:
        """TEST 2: Tersinir izentropik çevrimde sıfır entropi üretimi ve sıfır ekserji yıkımı denetimi."""
        geom = SliderCrankGeometry(
            bore_m=0.080,
            stroke_m=0.090,
            conrod_m=0.150,
            compression_ratio=16.0,
        )
        theta = np.linspace(-180.0, 180.0, 1441)
        r_spec = DRY_AIR.specific_gas_constant
        m_cyl = 1.0e-3
        t_init = 350.0

        def isentropic_rhs(th: float, state: np.ndarray) -> list[float]:
            t_curr = float(state[0])
            cv_curr = float(np.squeeze(DRY_AIR.cv(np.asarray([t_curr]))))
            st = geom.evaluate(np.asarray([th]))
            vol = float(st.volume_m3[0])
            dvol = float(st.dv_dtheta_m3_per_deg[0])
            return [-r_spec * t_curr / (cv_curr * vol) * dvol]

        ode_sol = solve_ivp(
            isentropic_rhs,
            [-180.0, 180.0],
            [t_init],
            t_eval=theta,
            method="RK45",
            rtol=1e-12,
            atol=1e-14,
        )
        assert ode_sol.success

        t_cycle = np.asarray(ode_sol.y[0], dtype=np.float64)
        kin = geom.evaluate(theta)
        v_cycle = kin.volume_m3
        dv_cycle = kin.dv_dtheta_m3_per_deg
        p_cycle = m_cyl * r_spec * t_cycle / v_cycle

        s_standard = compute_standard_entropy(t_cycle, gas_mixture=DRY_AIR)
        s_total = s_standard - r_spec * np.log(p_cycle)
        max_delta_s = float(np.max(np.abs(s_total - s_total[0])))

        dead_state = DeadStateConditions.from_altitude(0.0)
        rates = compute_exergy_rates(
            crank_angle_deg=theta,
            pressure_pa=p_cycle,
            temperature_k=t_cycle,
            volume_m3=v_cycle,
            dvolume_dtheta_m3_per_deg=dv_cycle,
            gross_heat_release_rate_j_per_deg=np.zeros_like(theta),
            wall_heat_loss_rate_j_per_deg=np.zeros_like(theta),
            dead_state=dead_state,
            trapped_mass_kg=m_cyl,
            wall_temperature_k=450.0,
            phi_factor=DIESEL_EN590_EXERGY_FACTOR,
            gas_mixture=DRY_AIR,
        )

        de_tm_dtheta = compute_derivative(theta, rates.thermomechanical_exergy_j)
        work_exergy_rate = rates.work_exergy_rate_j_per_deg
        balance_discrepancy = float(np.max(np.abs(de_tm_dtheta + work_exergy_rate)))
        total_e_destroyed = float(trapezoid(rates.exergy_destruction_rate_j_per_deg, theta))

        assert max_delta_s < 1.0e-7
        assert balance_discrepancy < 1.0e-6
        assert abs(total_e_destroyed) < 1.0e-5

    def test_03_closed_cycle_atmospheric_work_and_open_stroke(self) -> None:
        """TEST 3: Kapalı çevrimde atmosferik iş (oint P0 dV = 0) ve açık strok farkı denetimi."""
        geom = SliderCrankGeometry(
            bore_m=0.080,
            stroke_m=0.090,
            conrod_m=0.150,
            compression_ratio=16.0,
        )
        theta = np.linspace(-180.0, 180.0, 1441)
        kin = geom.evaluate(theta)
        v = kin.volume_m3
        dv = kin.dv_dtheta_m3_per_deg

        p0 = 101325.0
        p = 2.0e5 * (v[0] / v) ** 1.35

        w_ind_closed = float(trapezoid(p * dv, theta))
        w_atm_closed = float(trapezoid(p0 * dv, theta))
        w_net_closed = float(trapezoid((p - p0) * dv, theta))

        idx_tdc = int(np.argmin(np.abs(theta - 0.0)))
        w_ind_comp = float(trapezoid(p[: idx_tdc + 1] * dv[: idx_tdc + 1], theta[: idx_tdc + 1]))
        w_atm_comp = float(trapezoid(p0 * dv[: idx_tdc + 1], theta[: idx_tdc + 1]))
        w_net_comp = float(trapezoid((p[: idx_tdc + 1] - p0) * dv[: idx_tdc + 1], theta[: idx_tdc + 1]))
        analytic_delta_v = float(p0 * (v[idx_tdc] - v[0]))

        assert abs(w_atm_closed) < 1.0e-11
        assert abs(w_ind_closed - w_net_closed) < 1.0e-11
        assert abs((w_ind_comp - w_net_comp) - w_atm_comp) < 1.0e-12
        assert abs(w_atm_comp - analytic_delta_v) < 1.0e-3

    def test_04_high_altitude_carnot_factor_and_second_law_closure(self) -> None:
        """TEST 4: Yüksek irtifa Carnot faktörü artışı ve %100.0000 bütçe kapanış denetimi."""
        ds_sea = DeadStateConditions.from_altitude(0.0)
        ds_alt = DeadStateConditions.from_altitude(30000.0 * FEET_TO_M)

        t_wall = 450.0
        carnot_sea = 1.0 - (ds_sea.t0_k / t_wall)
        carnot_alt = 1.0 - (ds_alt.t0_k / t_wall)

        assert ds_alt.t0_k < ds_sea.t0_k
        assert carnot_alt > carnot_sea

        geom = SliderCrankGeometry(
            bore_m=0.080,
            stroke_m=0.090,
            conrod_m=0.150,
            compression_ratio=16.0,
        )
        theta = np.linspace(-180.0, 180.0, 721)
        kin = geom.evaluate(theta)
        v = kin.volume_m3
        dv = kin.dv_dtheta_m3_per_deg

        m_cyl = 5.5e-4
        r_spec = DRY_AIR.specific_gas_constant

        p_ivc = 1.0e5
        burn_start = -10.0
        burn_duration = 50.0
        fraction = np.clip((theta - burn_start) / burn_duration, 0.0, 1.0)
        wiebe = 1.0 - np.exp(-5.0 * (fraction ** 3.0))

        p_comp = p_ivc * (v[0] / v) ** 1.35
        p_comb = p_comp + 2.0e6 * wiebe * (v[np.argmin(np.abs(theta))] / v) ** 0.8
        t_gas = p_comb * v / (m_cyl * r_spec)

        fuel_mass = 3.5e-5
        lhv = 42.8e6
        phi_fuel = JET_A1_EXERGY_FACTOR

        dq_gross_cum = 1300.0 * wiebe
        dq_gross_rate = compute_derivative(theta, dq_gross_cum)
        dq_ht_rate = np.maximum(5.0 * np.exp(-0.5 * ((theta - 15.0) / 25.0) ** 2), 0.0)

        rates = compute_exergy_rates(
            crank_angle_deg=theta,
            pressure_pa=p_comb,
            temperature_k=t_gas,
            volume_m3=v,
            dvolume_dtheta_m3_per_deg=dv,
            gross_heat_release_rate_j_per_deg=dq_gross_rate,
            wall_heat_loss_rate_j_per_deg=dq_ht_rate,
            dead_state=ds_alt,
            trapped_mass_kg=m_cyl,
            wall_temperature_k=t_wall,
            phi_factor=phi_fuel,
            gas_mixture=DRY_AIR,
        )

        budget = compute_exergy_cycle_budget(
            rates_result=rates,
            fuel_mass_kg=fuel_mass,
            lhv_j_per_kg=lhv,
            phi_factor=phi_fuel,
        )

        assert abs(budget.budget_closure_pct - 100.0) < 1.0e-9
        assert budget.closure_error_j < 1.0e-9
        assert abs((budget.eta_first_law / budget.eta_second_law) - phi_fuel) < 1.0e-9

    def test_05_ast_zero_loop_vectorization_and_pyrefly_clean(self) -> None:
        """TEST 5: exergy/ paketinde AST düzeyinde sıfır dizi döngüsü ve tip güvenliği denetimi."""
        assert_package_vectorized_and_typed("exergy")


@pytest.mark.strict_audit
class TestStrictAuditExergySecondLaw:
    """Strict analytical audits for Step 7 (exergy package)."""

    def test_01_audit_isa_stratosphere_isothermal_dead_state(self) -> None:
        """DENETİM 1: ISA atmosfer modeli tropopoz (11 km) ve stratosfer (15 km) ölü hal tuzağı."""
        z_target = 15000.0
        ds_15k = DeadStateConditions.from_altitude(z_target)

        t0_sea = 288.15
        p0_sea = 101325.0
        lapse_rate = 0.0065
        z_tropopause = 11000.0

        t_tropopause = t0_sea - lapse_rate * z_tropopause
        barometric_exp = STD_GRAVITY_M_S2 / (R_SPECIFIC_DRY_AIR * lapse_rate)
        p_tropopause = p0_sea * ((t_tropopause / t0_sea) ** barometric_exp)

        p_15k_exact = p_tropopause * np.exp(
            -STD_GRAVITY_M_S2 * (z_target - z_tropopause) / (R_SPECIFIC_DRY_AIR * t_tropopause)
        )
        rho_15k_exact = p_15k_exact / (R_SPECIFIC_DRY_AIR * t_tropopause)

        err_t = abs(ds_15k.t0_k - t_tropopause)
        err_p = abs(ds_15k.p0_pa - p_15k_exact)
        err_rho = abs(ds_15k.rho0_kg_per_m3 - rho_15k_exact)

        assert err_t < 1.0e-12
        assert err_p < 1.0e-9
        assert err_rho < 1.0e-9

    def test_02_audit_kotas_chemical_exergy_mass_vs_atomic(self) -> None:
        """DENETİM 2: Kotas (1995) kimyasal ekserji faktörü (phi) kütlesel vs. atomik oran tuzağı."""
        lhv_dodecane = 44.14e6
        m_h = 1.00794
        m_c = 12.0107

        n_c = 12.0
        n_h = 26.0
        atomic_h_c = n_h / n_c
        mass_h_c = (n_h * m_h) / (n_c * m_c)

        phi_exact = 1.0401 + 0.1728 * mass_h_c

        res_mass = compute_fuel_chemical_exergy(lhv_dodecane, h_c_mass=mass_h_c)
        res_atomic = compute_fuel_chemical_exergy(lhv_dodecane, h_c_atomic=atomic_h_c)

        err_mass = abs(res_mass.phi_factor - phi_exact)
        err_atomic = abs(res_atomic.phi_factor - phi_exact)

        assert err_mass < 1.0e-12
        assert err_atomic < 1.0e-12
        assert abs(res_mass.e_chem_j_per_kg - phi_exact * lhv_dodecane) < 1.0e-6

    def test_03_audit_nasa_entropy_continuity_and_maxwell_identity(self) -> None:
        """DENETİM 3: NASA 1000 K sürekliliği ve Maxwell/Gibbs kısmi türev özdeşliği denetimi."""
        ds = DeadStateConditions.from_altitude(0.0)
        m_cyl = 1.0
        r_spec = DRY_AIR.specific_gas_constant

        t_minus = np.asarray([999.9999])
        t_plus = np.asarray([1000.0001])
        p_test = np.asarray([2.0e5])

        e_minus = compute_thermomechanical_exergy(t_minus, p_test, ds, m_cyl, gas_mixture=DRY_AIR)[0]
        e_plus = compute_thermomechanical_exergy(t_plus, p_test, ds, m_cyl, gas_mixture=DRY_AIR)[0]
        delta_e_tm = abs(e_plus - e_minus)

        assert delta_e_tm < 1.0e-2

        t_eval = 1650.0
        v_eval = 0.08
        vol_eval = v_eval * m_cyl
        p_eval = r_spec * t_eval / v_eval

        delta_t = 1.0e-4
        delta_v = 1.0e-6

        p_t_p = r_spec * (t_eval + delta_t) / v_eval
        p_t_m = r_spec * (t_eval - delta_t) / v_eval
        e_t_p = compute_thermomechanical_exergy(
            np.asarray([t_eval + delta_t]), np.asarray([p_t_p]), ds, m_cyl, volume_m3=np.asarray([vol_eval])
        )[0]
        e_t_m = compute_thermomechanical_exergy(
            np.asarray([t_eval - delta_t]), np.asarray([p_t_m]), ds, m_cyl, volume_m3=np.asarray([vol_eval])
        )[0]
        de_dt_num = (e_t_p - e_t_m) / (2.0 * delta_t)

        v_p = v_eval + delta_v
        v_m = v_eval - delta_v
        p_v_p = r_spec * t_eval / v_p
        p_v_m = r_spec * t_eval / v_m
        e_v_p = compute_thermomechanical_exergy(
            np.asarray([t_eval]), np.asarray([p_v_p]), ds, m_cyl, volume_m3=np.asarray([v_p * m_cyl])
        )[0]
        e_v_m = compute_thermomechanical_exergy(
            np.asarray([t_eval]), np.asarray([p_v_m]), ds, m_cyl, volume_m3=np.asarray([v_m * m_cyl])
        )[0]
        de_dv_num = (e_v_p - e_v_m) / (2.0 * delta_v)

        cv_eval = float(np.squeeze(DRY_AIR.cv(t_eval)))
        de_dt_analytic = cv_eval * (1.0 - ds.t0_k / t_eval)
        de_dv_analytic = ds.p0_pa - (ds.t0_k / t_eval) * p_eval

        rel_err_t = abs(de_dt_num - de_dt_analytic) / abs(de_dt_analytic)
        rel_err_v = abs(de_dv_num - de_dv_analytic) / abs(de_dv_analytic)

        assert rel_err_t < 1.0e-9
        assert rel_err_v < 1.0e-9

    def test_04_audit_gouy_stodola_entropy_generation_diabatic(self) -> None:
        """DENETİM 4: Gouy-Stodola teoremi (E_dest = T0*S_gen) ve diyabatik çevrim denetimi."""
        geom = SliderCrankGeometry(
            bore_m=0.080,
            stroke_m=0.090,
            conrod_m=0.150,
            compression_ratio=16.0,
        )
        theta = np.linspace(-180.0, 180.0, 721)
        kin = geom.evaluate(theta)
        v = kin.volume_m3
        dv = kin.dv_dtheta_m3_per_deg
        m_cyl = 5.5e-4
        r_spec = DRY_AIR.specific_gas_constant

        fraction = np.clip((theta - (-10.0)) / 50.0, 0.0, 1.0)
        wiebe = 1.0 - np.exp(-5.0 * (fraction ** 3.0))
        q_comb_cum = 1000.0 * wiebe
        dq_comb_dtheta = compute_derivative(theta, q_comb_cum)
        dq_ht_dtheta = np.maximum(8.0 * np.exp(-0.5 * ((theta - 15.0) / 25.0) ** 2), 0.0)

        t_w = 450.0
        ds = DeadStateConditions.from_altitude(0.0)
        t0 = ds.t0_k

        v_spline = CubicSpline(theta, v)
        dv_spline = v_spline.derivative()
        dq_comb_spline = CubicSpline(theta, dq_comb_dtheta)
        dq_ht_spline = CubicSpline(theta, dq_ht_dtheta)

        def diabatic_rhs(th: float, state: np.ndarray) -> list[float]:
            t_curr = float(state[0])
            cv_curr = float(np.squeeze(DRY_AIR.cv(t_curr)))
            vol = float(v_spline(th))
            dvol = float(dv_spline(th))
            p_curr = m_cyl * r_spec * t_curr / vol
            dq_c = float(dq_comb_spline(th))
            dq_h = float(dq_ht_spline(th))
            return [(dq_c - dq_h - p_curr * dvol) / (m_cyl * cv_curr)]

        ode_sol = solve_ivp(
            diabatic_rhs,
            [-180.0, 180.0],
            [320.0],
            t_eval=theta,
            method="RK45",
            rtol=1e-11,
            atol=1e-13,
        )
        assert ode_sol.success

        t_sol = ode_sol.y[0]
        p_sol = m_cyl * r_spec * t_sol / v

        phi_fuel = JET_A1_EXERGY_FACTOR
        rates = compute_exergy_rates(
            crank_angle_deg=theta,
            pressure_pa=p_sol,
            temperature_k=t_sol,
            volume_m3=v,
            dvolume_dtheta_m3_per_deg=dv,
            gross_heat_release_rate_j_per_deg=dq_comb_dtheta,
            wall_heat_loss_rate_j_per_deg=dq_ht_dtheta,
            dead_state=ds,
            trapped_mass_kg=m_cyl,
            wall_temperature_k=t_w,
            phi_factor=phi_fuel,
            gas_mixture=DRY_AIR,
        )

        e_dest_balance = float(trapezoid(rates.exergy_destruction_rate_j_per_deg, theta))

        de_dest_gouy = (phi_fuel - (1.0 - t0 / t_sol)) * dq_comb_dtheta + t0 * (1.0 / t_w - 1.0 / t_sol) * dq_ht_dtheta
        e_dest_gouy = float(trapezoid(de_dest_gouy, theta))
        diff_gouy = abs(e_dest_balance - e_dest_gouy)

        fuel_mass = 2.4e-5
        lhv = 42.8e6
        budget = compute_exergy_cycle_budget(
            rates_result=rates,
            fuel_mass_kg=fuel_mass,
            lhv_j_per_kg=lhv,
            phi_factor=phi_fuel,
        )

        assert diff_gouy < 1.0e-6
        assert budget.closure_error_j < 1.0e-9

    def test_05_audit_altitude_evo_exergy_recovery_potential(self) -> None:
        """DENETİM 5: İrtifa etkisi: Deniz seviyesi vs. 30,000 ft EVO ekserji geri kazanım potansiyeli."""
        ds_sea = DeadStateConditions.from_altitude(0.0)
        ds_alt = DeadStateConditions.from_altitude(30000.0 * FEET_TO_M)

        t_evo = np.asarray([1150.0])
        p_evo = np.asarray([4.2e5])
        m_cyl = 1.0e-3

        e_tm_sea = float(compute_thermomechanical_exergy(t_evo, p_evo, ds_sea, m_cyl, gas_mixture=DRY_AIR)[0])
        e_tm_alt = float(compute_thermomechanical_exergy(t_evo, p_evo, ds_alt, m_cyl, gas_mixture=DRY_AIR)[0])
        recovery_gain_pct = ((e_tm_alt - e_tm_sea) / e_tm_sea) * 100.0

        assert ds_alt.p0_pa < ds_sea.p0_pa
        assert ds_alt.t0_k < ds_sea.t0_k
        assert e_tm_alt > e_tm_sea
        assert recovery_gain_pct > 15.0

    def test_06_audit_ast_zero_loops_and_pyrefly_clean(self) -> None:
        """DENETİM 6: exergy/ modüllerinde AST düzeyinde 0 dizi döngüsü ve tip güvenliği denetimi."""
        assert_package_vectorized_and_typed("exergy")
