"""Independent strict physical, dimensional, and numerical audit for Steps 1 & 2."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import pytest
from scipy.integrate import simpson, solve_ivp

from ice_cfd_validator.io import parse_thermo_csv
from ice_cfd_validator.signal import compute_derivative
from ice_cfd_validator.thermo import (
    DIESEL_BURNED_PRODUCTS,
    DRY_AIR,
    HohenbergCorrelation,
    SliderCrankGeometry,
    WoschniCorrelation,
    compute_cp,
    compute_cv,
    compute_gamma,
    compute_heat_release,
    compute_imep,
    extract_burn_fractions,
)


def test_audit_1_kinematics_analytical_vs_numerical_deg_rad_trap() -> None:
    """DENETIM 1: Slider-crank compression ratio & analytical vs numerical dV/dtheta (deg/rad trap)."""
    geom = SliderCrankGeometry(
        bore_m=0.085,
        stroke_m=0.088,
        conrod_m=0.145,
        compression_ratio=16.5,
    )

    # Kontrol A: V(180 deg) / V(0 deg) == r_c = 16.5 (error < 1e-12)
    kin_endpoints = geom.evaluate(np.array([0.0, 180.0], dtype=np.float64))
    v_tdc = float(kin_endpoints.volume_m3[0])
    v_bdc = float(kin_endpoints.volume_m3[1])
    cr_computed = v_bdc / v_tdc
    cr_abs_err = abs(cr_computed - 16.5)

    # Kontrol B: Analytical dV/dtheta vs numerical compute_derivative(theta, V)
    theta_deg = np.linspace(-180.0, 180.0, 7201, dtype=np.float64)
    kin = geom.evaluate(theta_deg)
    dv_ana = kin.dv_dtheta_m3_per_deg
    dv_num = compute_derivative(theta_deg, kin.volume_m3, method="central4")

    peak_dv = float(np.max(np.abs(dv_ana)))
    peak_norm_rel_err = float(np.max(np.abs(dv_ana - dv_num)) / peak_dv)

    # Pointwise relative error away from exact dead-center zero crossings (|sin(theta)| > 0.05)
    non_dc_mask = np.abs(np.sin(np.deg2rad(theta_deg))) > 0.05
    pointwise_rel_err = float(
        np.max(np.abs(dv_ana[non_dc_mask] - dv_num[non_dc_mask]) / np.abs(dv_ana[non_dc_mask]))
    )
    scale_ratio = float(np.max(np.abs(dv_ana)) / np.max(np.abs(dv_num)))

    print("\n" + "=" * 74)
    print("[DENETIM 1] Biyel-Krank Kinematigi ve deg/rad Tuzagi Kontrolu")
    print(f"  V(0 deg) [TDC]               = {v_tdc:.12e} m^3")
    print(f"  V(180 deg) [BDC]             = {v_bdc:.12e} m^3")
    print(f"  V(180)/V(0)                  = {cr_computed:.15f} (Hedef: 16.5, |hata| = {cr_abs_err:.3e})")
    print(f"  max|dV/dtheta| (Analitik)    = {np.max(np.abs(dv_ana)):.12e} m^3/deg")
    print(f"  max|dV/dtheta| (Sayisal)     = {np.max(np.abs(dv_num)):.12e} m^3/deg")
    print(f"  Olcek Orani (Ana / Num)      = {scale_ratio:.12f} (57.295x rad/deg hatasi YOK)")
    print(f"  Tepe-Normalize Bagil Hata    = {peak_norm_rel_err:.3e} (< 1e-6)")
    print(f"  Noktasal Maks. Bagil Hata    = {pointwise_rel_err:.3e} (< 1e-6)")
    print("=" * 74)

    assert cr_abs_err < 1.0e-12
    assert peak_norm_rel_err < 1.0e-6
    assert pointwise_rel_err < 1.0e-6


def test_audit_2_nasa_polynomials_continuity_and_mayer_consistency() -> None:
    """DENETIM 2: NASA 7-term polynomials bounds, 1000 K seam continuity, and Mayer relation."""
    # Kontrol A: Dry air gamma at 300 K and 2000 K
    gam_300 = float(compute_gamma(300.0, DRY_AIR))
    gam_2000 = float(compute_gamma(2000.0, DRY_AIR))

    # Kontrol B: 1000 K seam continuity (999.999 K vs 1000.001 K)
    t_seam = np.array([999.999, 1000.001], dtype=np.float64)
    cp_air_seam = compute_cp(t_seam, DRY_AIR)
    gam_air_seam = compute_gamma(t_seam, DRY_AIR)
    delta_cp_air = float(abs(cp_air_seam[1] - cp_air_seam[0]))
    delta_gam_air = float(abs(gam_air_seam[1] - gam_air_seam[0]))

    cp_prod_seam = compute_cp(t_seam, DIESEL_BURNED_PRODUCTS)
    gam_prod_seam = compute_gamma(t_seam, DIESEL_BURNED_PRODUCTS)
    delta_cp_prod = float(abs(cp_prod_seam[1] - cp_prod_seam[0]))
    delta_gam_prod = float(abs(gam_prod_seam[1] - gam_prod_seam[0]))

    # Kontrol C: Mayer relation cp(T) - cv(T) == R_specific across [200, 6000] K
    t_grid = np.linspace(200.0, 6000.0, 5801, dtype=np.float64)
    cp_grid = compute_cp(t_grid, DRY_AIR)
    cv_grid = compute_cv(t_grid, DRY_AIR)
    r_spec = DRY_AIR.specific_gas_constant
    max_mayer_err = float(np.max(np.abs((cp_grid - cv_grid) - r_spec)))

    print("\n" + "=" * 74)
    print("[DENETIM 2] NASA 7-Katsayili Polinomlar ve Termodinamik Tutarlilik")
    print(f"  gamma_air(300 K)             = {gam_300:.8f} (Beklenen: [1.398, 1.402])")
    print(f"  gamma_air(2000 K)            = {gam_2000:.8f} (Beklenen: [1.280, 1.310])")
    print(f"  1000 K Gecisi Kuru Hava      : |d_cp| = {delta_cp_air:.4e} J/(kg*K), |d_gamma| = {delta_gam_air:.4e}")
    print(f"  1000 K Gecisi Yanma Urunleri : |d_cp| = {delta_cp_prod:.4e} J/(kg*K), |d_gamma| = {delta_gam_prod:.4e}")
    print(f"  R_specific (Kuru Hava)       = {r_spec:.8f} J/(kg*K)")
    print(f"  Mayer max|cp - cv - R_spec|  = {max_mayer_err:.3e} J/(kg*K)")
    print("=" * 74)

    assert 1.398 <= gam_300 <= 1.402
    assert 1.280 <= gam_2000 <= 1.310
    assert delta_gam_air < 1.0e-4
    assert delta_gam_prod < 1.0e-4
    assert delta_cp_air < 0.05
    assert delta_cp_prod < 0.05
    assert max_mayer_err < 1.0e-11


def test_audit_3_forward_ode_to_inverse_ahrr_energy_conservation() -> None:
    """DENETIM 3: Forward variable-gamma(T) Wiebe ODE -> Inverse AHRR energy & phasing audit."""
    geom = SliderCrankGeometry(
        bore_m=0.085,
        stroke_m=0.088,
        conrod_m=0.145,
        compression_ratio=16.5,
    )
    theta_deg = np.linspace(-180.0, 180.0, 7201, dtype=np.float64)

    # Analytical Wiebe parameters
    q_total = 1200.0
    theta_0 = -10.0
    delta_theta = 40.0
    m_shape = 2.0
    a_eff = 6.908

    exact_ca10 = theta_0 + delta_theta * (-np.log(1.0 - 0.10) / a_eff) ** (1.0 / (m_shape + 1.0))
    exact_ca50 = theta_0 + delta_theta * (-np.log(1.0 - 0.50) / a_eff) ** (1.0 / (m_shape + 1.0))
    exact_ca90 = theta_0 + delta_theta * (-np.log(1.0 - 0.90) / a_eff) ** (1.0 / (m_shape + 1.0))

    p_ivc = 150_000.0
    t_ivc = 330.0
    v_ivc = float(geom.evaluate(np.array([-180.0], dtype=np.float64)).volume_m3[0])
    r_spec = DRY_AIR.specific_gas_constant
    m_trapped = (p_ivc * v_ivc) / (r_spec * t_ivc)

    def dq_wiebe_dtheta(th: float) -> float:
        if th <= theta_0:
            return 0.0
        u = (th - theta_0) / delta_theta
        exponent = -a_eff * (u ** (m_shape + 1.0))
        if exponent < -700.0:
            return 0.0
        return (
            q_total
            * (a_eff * (m_shape + 1.0) / delta_theta)
            * (u**m_shape)
            * float(np.exp(exponent))
        )

    def forward_ode_rhs(th: float, y: NDArray[np.float64]) -> list[float]:
        p_curr = float(y[0])
        kin_pt = geom.evaluate(np.array([th], dtype=np.float64))
        v_curr = float(kin_pt.volume_m3[0])
        dv_curr = float(kin_pt.dv_dtheta_m3_per_deg[0])
        t_curr = (p_curr * v_curr) / (m_trapped * r_spec)
        gam_curr = float(DRY_AIR.gamma(t_curr))
        dp_dth = ((gam_curr - 1.0) / v_curr) * dq_wiebe_dtheta(th) - gam_curr * (
            p_curr / v_curr
        ) * dv_curr
        return [dp_dth]

    sol = solve_ivp(
        forward_ode_rhs,
        t_span=(-180.0, 180.0),
        y0=[p_ivc],
        t_eval=theta_deg,
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
    )
    assert sol.success, f"RK45 integration failed: {sol.message}"

    p_syn = np.asarray(sol.y[0], dtype=np.float64)
    kin = geom.evaluate(theta_deg)
    t_syn = (p_syn * kin.volume_m3) / (m_trapped * r_spec)

    # Inverse AHRR reconstruction
    hrr = compute_heat_release(
        theta_deg,
        p_syn,
        kin.volume_m3,
        temperature_k=t_syn,
        dv_dtheta_m3_per_deg=kin.dv_dtheta_m3_per_deg,
        mixture=DRY_AIR,
        derivative_method="central4",
    )
    mfb = extract_burn_fractions(
        theta_deg,
        hrr.dq_net_j_per_deg,
        start_deg=-15.0,
        end_deg=80.0,
    )
    imep_res = compute_imep(
        theta_deg,
        p_syn,
        kin.volume_m3,
        displaced_volume_m3=geom.displaced_volume_m3,
        dv_dtheta_m3_per_deg=kin.dv_dtheta_m3_per_deg,
    )

    q_cum_cycle = float(hrr.cumulative_net_j[-1])
    q_err_j = abs(q_cum_cycle - q_total)
    q_err_pct = 100.0 * q_err_j / q_total

    err_ca10 = abs(mfb.ca10_deg - exact_ca10)
    err_ca50 = abs(mfb.ca50_deg - exact_ca50)
    err_ca90 = abs(mfb.ca90_deg - exact_ca90)

    work_from_imep_vd = imep_res.imep_net_pa * geom.displaced_volume_m3
    work_simpson = float(simpson(p_syn * kin.dv_dtheta_m3_per_deg, x=theta_deg))
    work_consistency_err = abs(imep_res.indicated_work_net_j - work_from_imep_vd)

    print("\n" + "=" * 74)
    print("[DENETIM 3] Forward ODE (RK45, gamma(T)) -> Inverse AHRR Enerji Korunumu")
    print(f"  Girdi Toplam Isi (Q_total)   = {q_total:.6f} J")
    print(f"  Cozucu Kumulatif Net Isi     = {q_cum_cycle:.6f} J (|hata| = {q_err_j:.4e} J, %{q_err_pct:.5f})")
    print(f"  Yanma Penceresi Toplam Isi   = {mfb.total_heat_j:.6f} J")
    print(f"  CA10 : Analitik = {exact_ca10:+.5f} deg | AHRR = {mfb.ca10_deg:+.5f} deg | |fark| = {err_ca10:.3e} deg")
    print(f"  CA50 : Analitik = {exact_ca50:+.5f} deg | AHRR = {mfb.ca50_deg:+.5f} deg | |fark| = {err_ca50:.3e} deg")
    print(f"  CA90 : Analitik = {exact_ca90:+.5f} deg | AHRR = {mfb.ca90_deg:+.5f} deg | |fark| = {err_ca90:.3e} deg")
    print(f"  Kapali Cevrim Isi (oint PdV) = {imep_res.indicated_work_net_j:.8f} J (Simpson: {work_simpson:.8f} J)")
    print(f"  IMEP * V_d                   = {work_from_imep_vd:.8f} J (|fark| = {work_consistency_err:.3e} J)")
    print("=" * 74)

    assert q_err_j < 0.6 and q_err_pct < 0.05
    assert abs(mfb.total_heat_j - q_total) < 0.6
    assert err_ca10 < 0.05
    assert err_ca50 < 0.05
    assert err_ca90 < 0.05
    assert work_consistency_err < 1.0e-11
    assert abs(imep_res.indicated_work_net_j - work_simpson) < 1.0e-4


def test_audit_4_woschni_and_hohenberg_pressure_unit_scale_trap() -> None:
    """DENETIM 4: Woschni & Hohenberg unit scaling check at motored TDC (5 MPa, 900 K, Sp=10 m/s)."""
    geom = SliderCrankGeometry(
        bore_m=0.085,
        stroke_m=0.088,
        conrod_m=0.145,
        compression_ratio=16.5,
    )
    mean_piston_speed = 10.0  # m/s
    rpm = (mean_piston_speed * 60.0) / (2.0 * geom.stroke_m)

    kin_tdc = geom.evaluate(np.array([0.0], dtype=np.float64))
    p_tdc_pa = np.array([5.0e6], dtype=np.float64)
    t_tdc_k = np.array([900.0], dtype=np.float64)

    woschni = WoschniCorrelation(
        bore_m=geom.bore_m,
        stroke_m=geom.stroke_m,
        rpm=rpm,
        wall_temp_k=450.0,
        displaced_volume_m3=geom.displaced_volume_m3,
    )
    res_w = woschni.evaluate(
        p_tdc_pa,
        t_tdc_k,
        kin_tdc.wall_area_m2,
        motored_pressure_pa=p_tdc_pa,
        p_ref_pa=150_000.0,
        v_ref_m3=geom.displaced_volume_m3 + geom.clearance_volume_m3,
        t_ref_k=330.0,
    )

    hohenberg = HohenbergCorrelation(
        stroke_m=geom.stroke_m,
        rpm=rpm,
        wall_temp_k=450.0,
    )
    res_h = hohenberg.evaluate(
        p_tdc_pa,
        kin_tdc.volume_m3,
        t_tdc_k,
        kin_tdc.wall_area_m2,
    )

    hc_w = float(res_w.hc_w_m2k[0])
    hc_h = float(res_h.hc_w_m2k[0])

    print("\n" + "=" * 74)
    print("[DENETIM 4] Woschni & Hohenberg Motored UON (TDC) Birim Olcek Kontrolu")
    print(f"  Kosullar: P = 5.0 MPa ({p_tdc_pa[0]:.1e} Pa), T = 900 K, S_p = {mean_piston_speed:.1f} m/s ({rpm:.1f} RPM)")
    print(f"  Woschni   h_c(TDC) = {hc_w:.2f} W/(m^2*K), dQ_ht/dtheta = {res_w.dq_ht_j_per_deg[0]:.4f} J/deg")
    print(f"  Hohenberg h_c(TDC) = {hc_h:.2f} W/(m^2*K), dQ_ht/dtheta = {res_h.dq_ht_j_per_deg[0]:.4f} J/deg")
    print("=" * 74)

    assert 1500.0 <= hc_w <= 6000.0, f"Woschni h_c={hc_w:.2f} W/(m^2*K) outside [1500, 6000]"
    assert 1500.0 <= hc_h <= 6000.0, f"Hohenberg h_c={hc_h:.2f} W/(m^2*K) outside [1500, 6000]"


def test_audit_5_forte_parser_complex_headers_and_pv_scaling(tmp_path: Path) -> None:
    """DENETIM 5: ANSYS Forte mixed-unit headers Crank_Angle_(deg), Pressure_(MPa), Cylinder_Volume_(cm^3), Wall_Heat_Loss_(J/CA)."""
    csv_file = tmp_path / "forte_mixed_headers_thermo.csv"
    csv_content = (
        "Crank_Angle_(deg), Pressure_(MPa), Cylinder_Volume_(cm^3), Wall_Heat_Loss_(J/CA)\n"
        "0.0, 2.5, 450.0, 14.2\n"
        "1.0, 2.4, 452.0, 13.8\n"
    )
    csv_file.write_text(csv_content, encoding="utf-8")

    parsed = parse_thermo_csv(csv_file)
    p_si = float(parsed.pressure_pa[0])
    v_si = float(parsed.volume_m3[0])
    pv_joules = p_si * v_si
    assert parsed.wall_heat_loss_j_per_deg is not None
    q_wall_si = float(parsed.wall_heat_loss_j_per_deg[0])

    print("\n" + "=" * 74)
    print("[DENETIM 5] ANSYS Forte Karmasik Baslik ve P*V Birim Olcekleme")
    print(f"  Basliklar : Crank_Angle_(deg), Pressure_(MPa), Cylinder_Volume_(cm^3), Wall_Heat_Loss_(J/CA)")
    print(f"  Ham girdi : P = 2.5 MPa, V = 450.0 cm^3, dQ_wall = 14.2 J/CA")
    print(f"  SI Cikti  : P = {p_si:.6e} Pa, V = {v_si:.6e} m^3, dQ_wall = {q_wall_si:.4f} J/deg")
    print(f"  P * V     = {pv_joules:.10f} J (Hedef: 1125.0 J)")
    print("=" * 74)

    assert p_si == pytest.approx(2.5e6, rel=1e-14, abs=1e-9)
    assert v_si == pytest.approx(4.5e-4, rel=1e-14, abs=1e-18)
    assert q_wall_si == pytest.approx(14.2, rel=1e-14, abs=1e-12)
    assert pv_joules == pytest.approx(1125.0, rel=1e-14, abs=1e-11)


def test_audit_6_vectorization_ast_and_static_type_quality() -> None:
    """DENETIM 6: AST-level audit proving zero element-wise array loops in numerical modules."""
    project_root = Path(__file__).resolve().parents[1]
    pkg_root = project_root / "src" / "ice_cfd_validator"
    numerical_dirs = [pkg_root / "signal", pkg_root / "thermo", pkg_root / "spray"]

    loop_report: list[str] = []
    elementwise_violations: list[str] = []
    untyped_public_funcs: list[str] = []

    for py_file in sorted(pkg_root.rglob("*.py")):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        rel_path = py_file.relative_to(project_root).as_posix()

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                if node.returns is None:
                    untyped_public_funcs.append(f"{rel_path}:{node.lineno}:{node.name}")

            if isinstance(node, (ast.For, ast.While)):
                iter_src = ast.unparse(node.iter) if isinstance(node, ast.For) else "while"
                entry = f"{rel_path}:{node.lineno} -> for/while ({iter_src})"
                loop_report.append(entry)
                if py_file.parent in numerical_dirs:
                    # Only dictionary metadata iteration over species (<= 5 keys) is allowed
                    if "mole_fractions.items()" not in iter_src:
                        elementwise_violations.append(entry)

    print("\n" + "=" * 74)
    print("[DENETIM 6] AST Vektorizasyon ve Statik Tip Kalite Denetimi")
    print(f"  Taranan Paket Dizini         : {pkg_root}")
    print(f"  signal/ ve thermo/ Dizi Dongusu Ihlali : {len(elementwise_violations)}")
    print(f"  Donus Tipi Eksik Public Fonksiyon      : {len(untyped_public_funcs)}")
    print("  Tum Paketteki Dongu Noktalari (Yalnizca baslik/sozluk metaverisi):")
    for item in loop_report:
        print(f"    - {item}")
    print("=" * 74)

    assert len(elementwise_violations) == 0, f"Element-wise loop violations: {elementwise_violations}"
    assert len(untyped_public_funcs) == 0, f"Untyped public functions: {untyped_public_funcs}"
