"""Analytical verification tests for signal filtering, kinematics, NASA thermo, and AHRR."""

from __future__ import annotations

from io import StringIO

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.signal import butter, lfilter, lfilter_zi

from ice_cfd_validator.exceptions import PhysicalBoundsError, TDCPhaseError
from ice_cfd_validator.io import parse_thermo_csv
from ice_cfd_validator.signal import (
    butterworth_zero_phase,
    compute_derivative,
    savgol_zero_phase,
)
from ice_cfd_validator.thermo import (
    DIESEL_BURNED_PRODUCTS,
    DRY_AIR,
    STOICHIOMETRIC_PRODUCTS,
    HohenbergCorrelation,
    SliderCrankGeometry,
    WoschniCorrelation,
    compute_cp,
    compute_cv,
    compute_gamma,
    compute_heat_release,
    compute_hohenberg_heat_transfer,
    compute_imep,
    compute_kinematics,
    compute_knock_metrics,
    compute_woschni_heat_transfer,
    extract_burn_fractions,
    verify_tdc_phasing,
)


def test_polytropic_zero_apparent_heat_release() -> None:
    """Test 1: Analytical polytropic stroke must satisfy max(|dQ_n/dtheta|) < 1e-8 J/deg."""
    geom = SliderCrankGeometry(
        bore_m=0.085,
        stroke_m=0.090,
        conrod_m=0.1575,
        compression_ratio=16.0,
    )
    theta_deg = np.linspace(-180.0, 180.0, 7201, dtype=np.float64)
    kin = geom.evaluate(theta_deg)

    kappa = 1.35
    p0_pa = 101_325.0
    v0_m3 = float(kin.volume_m3[0])
    pressure_pa = p0_pa * (v0_m3 / kin.volume_m3) ** kappa

    # 1a) Uniform grid with 4th-order central finite differences (analytical & numerical dV/dtheta)
    hrr = compute_heat_release(
        theta_deg,
        pressure_pa,
        kin.volume_m3,
        gamma=kappa,
        dv_dtheta_m3_per_deg=kin.dv_dtheta_m3_per_deg,
        derivative_method="central4",
    )
    max_abs_dq = float(np.max(np.abs(hrr.dq_net_j_per_deg)))
    assert max_abs_dq < 1.0e-8, f"Polytropic residual {max_abs_dq:.3e} >= 1e-8 J/deg"

    hrr_num_dv = compute_heat_release(
        theta_deg,
        pressure_pa,
        kin.volume_m3,
        gamma=kappa,
        derivative_method="central4",
    )
    assert float(np.max(np.abs(hrr_num_dv.dq_net_j_per_deg))) < 1.0e-8

    # 1b) Reversible polytropic closed compression+expansion must yield zero net work
    imep = compute_imep(
        theta_deg,
        pressure_pa,
        kin.volume_m3,
        displaced_volume_m3=geom.displaced_volume_m3,
        dv_dtheta_m3_per_deg=kin.dv_dtheta_m3_per_deg,
    )
    assert abs(imep.imep_gross_pa) < 1.0e-4
    assert abs(imep.imep_net_pa) < 1.0e-4


def test_zero_phase_filtering_preserves_wiebe_ca50_under_noise() -> None:
    """Test 2: Noisy synthetic Wiebe cycle filtered zero-phase must match analytical CA50 < 0.1 deg."""
    geom = SliderCrankGeometry(
        bore_m=0.085,
        stroke_m=0.090,
        conrod_m=0.1575,
        compression_ratio=16.0,
    )
    dtheta = 0.05
    theta_deg = np.linspace(-180.0, 180.0, 7201, dtype=np.float64)

    # Exact Wiebe parameters (Heywood, 2018, Eq. 9.5)
    theta_soi = -8.0
    delta_theta_b = 40.0
    wiebe_a = 5.0
    wiebe_m = 2.0
    q_total_j = 950.0
    exact_ca50_deg = theta_soi + delta_theta_b * (np.log(2.0) / wiebe_a) ** (1.0 / (wiebe_m + 1.0))

    p_ivc = 150_000.0
    t_ivc = 330.0
    v_ivc = float(geom.evaluate(np.array([-180.0])).volume_m3[0])
    r_spec = DRY_AIR.specific_gas_constant
    trapped_mass_kg = (p_ivc * v_ivc) / (r_spec * t_ivc)

    def wiebe_dq_dtheta(th: float) -> float:
        if th <= theta_soi or th >= theta_soi + delta_theta_b:
            return 0.0
        u = (th - theta_soi) / delta_theta_b
        return (
            q_total_j
            * (wiebe_a * (wiebe_m + 1.0) / delta_theta_b)
            * (u**wiebe_m)
            * float(np.exp(-wiebe_a * (u ** (wiebe_m + 1.0))))
        )

    def rhs(th: float, y: list[float]) -> list[float]:
        p_val = y[0]
        kin_pt = geom.evaluate(np.array([th], dtype=np.float64))
        v_val = float(kin_pt.volume_m3[0])
        dv_val = float(kin_pt.dv_dtheta_m3_per_deg[0])
        t_val = (p_val * v_val) / (trapped_mass_kg * r_spec)
        gam = float(DRY_AIR.gamma(np.array([t_val], dtype=np.float64))[0])
        dp = ((gam - 1.0) / v_val) * wiebe_dq_dtheta(th) - gam * (p_val / v_val) * dv_val
        return [dp]

    sol = solve_ivp(
        rhs,
        t_span=(-180.0, 180.0),
        y0=[p_ivc],
        t_eval=theta_deg,
        method="DOP853",
        rtol=1e-11,
        atol=1e-9,
    )
    assert sol.success
    p_clean = np.asarray(sol.y[0], dtype=np.float64)
    kin = geom.evaluate(theta_deg)

    # Superimpose Gaussian transducer noise (sigma = 12 kPa)
    rng = np.random.default_rng(2026)
    p_noisy = p_clean + rng.normal(0.0, 12_000.0, size=p_clean.size)

    # 1) Savitzky-Golay zero-phase filter
    p_sg = savgol_zero_phase(p_noisy, window_length=51, polyorder=3)
    t_sg = (p_sg * kin.volume_m3) / (trapped_mass_kg * r_spec)
    hrr_sg = compute_heat_release(
        theta_deg,
        p_sg,
        kin.volume_m3,
        temperature_k=t_sg,
        dv_dtheta_m3_per_deg=kin.dv_dtheta_m3_per_deg,
        mixture=DRY_AIR,
    )
    mfb_sg = extract_burn_fractions(
        theta_deg, hrr_sg.dq_net_j_per_deg, start_deg=-12.0, end_deg=40.0
    )
    assert abs(mfb_sg.ca50_deg - exact_ca50_deg) < 0.1

    # 2) Butterworth filtfilt zero-phase filter
    fs_per_deg = 1.0 / dtheta
    p_bw = butterworth_zero_phase(p_noisy, fs=fs_per_deg, cutoff=0.35, order=4, btype="low")
    t_bw = (p_bw * kin.volume_m3) / (trapped_mass_kg * r_spec)
    hrr_bw = compute_heat_release(
        theta_deg,
        p_bw,
        kin.volume_m3,
        temperature_k=t_bw,
        dv_dtheta_m3_per_deg=kin.dv_dtheta_m3_per_deg,
        mixture=DRY_AIR,
    )
    mfb_bw = extract_burn_fractions(
        theta_deg, hrr_bw.dq_net_j_per_deg, start_deg=-12.0, end_deg=40.0
    )
    assert abs(mfb_bw.ca50_deg - exact_ca50_deg) < 0.1

    # Contrast: causal single-pass Butterworth (lfilter) introduces severe CA50 phase lag (> 0.4 deg)
    ba_causal = butter(4, 0.35, btype="low", fs=fs_per_deg, output="ba")
    assert isinstance(ba_causal, tuple) and len(ba_causal) == 2
    zi = lfilter_zi(ba_causal[0], ba_causal[1]) * p_noisy[0]
    p_causal_raw, _ = lfilter(ba_causal[0], ba_causal[1], p_noisy, zi=zi)
    p_causal = np.asarray(p_causal_raw, dtype=np.float64)
    t_causal = (p_causal * kin.volume_m3) / (trapped_mass_kg * r_spec)
    hrr_causal = compute_heat_release(
        theta_deg,
        p_causal,
        kin.volume_m3,
        temperature_k=t_causal,
        dv_dtheta_m3_per_deg=kin.dv_dtheta_m3_per_deg,
        mixture=DRY_AIR,
    )
    mfb_causal = extract_burn_fractions(
        theta_deg, hrr_causal.dq_net_j_per_deg, start_deg=-12.0, end_deg=45.0
    )
    assert mfb_causal.ca50_deg - exact_ca50_deg > 0.4


def test_nonuniform_grid_fourth_order_central_and_spline_derivatives() -> None:
    """Verify O(h^4) differentiation on adaptive non-uniform crank-angle steps."""
    # Clustered grid near TDC mimicking ANSYS Forte adaptive time stepping
    s = np.linspace(-1.0, 1.0, 801, dtype=np.float64)
    theta_deg = 180.0 * (0.65 * s + 0.35 * (s**3))
    y = np.sin(np.deg2rad(theta_deg)) + 0.25 * np.cos(2.0 * np.deg2rad(theta_deg))
    exact_dydtheta = (np.pi / 180.0) * (
        np.cos(np.deg2rad(theta_deg)) - 0.5 * np.sin(2.0 * np.deg2rad(theta_deg))
    )

    d_central4 = compute_derivative(theta_deg, y, method="central4")
    d_spline = compute_derivative(theta_deg, y, method="spline")
    d_auto = compute_derivative(theta_deg, y, method="auto")

    np.testing.assert_allclose(d_central4, exact_dydtheta, atol=1e-9, rtol=1e-7)
    np.testing.assert_allclose(d_spline[1:-1], exact_dydtheta[1:-1], atol=1e-8, rtol=1e-7)
    np.testing.assert_allclose(d_spline, exact_dydtheta, atol=5e-8, rtol=5e-6)
    np.testing.assert_allclose(d_auto, d_spline, rtol=1e-14)


def test_tdc_phasing_audit_offset_kinematics_and_720deg_cycle() -> None:
    """Verify TDC phase detector flags > 0.5 deg shifts, offset dV/dtheta, and 720-deg cycles."""
    geom_offset = SliderCrankGeometry(
        bore_m=0.085,
        stroke_m=0.090,
        conrod_m=0.1575,
        compression_ratio=16.0,
        pin_offset_m=0.0015,
    )
    theta_deg = np.linspace(-360.0, 360.0, 7201, dtype=np.float64)
    kin = compute_kinematics(theta_deg, geom_offset)

    num_dv = compute_derivative(theta_deg, kin.volume_m3, method="central4")
    np.testing.assert_allclose(kin.dv_dtheta_m3_per_deg, num_dv, atol=1e-11, rtol=1e-8)
    assert float(np.max(kin.volume_m3) / np.min(kin.volume_m3)) == pytest.approx(16.0, rel=1e-6)

    # Aligned motored pressure over full 720-deg 4-stroke cycle passes near compression TDC
    p_aligned = 101_325.0 * (np.max(kin.volume_m3) / kin.volume_m3) ** 1.35
    res_ok = verify_tdc_phasing(theta_deg, p_aligned, kin.volume_m3, tolerance_deg=0.5)
    assert res_ok.is_valid
    assert abs(res_ok.phase_error_deg) < 1.0e-3

    # Inject +0.65 deg encoder phase error -> must raise TDCPhaseError or warn
    kin_shifted = geom_offset.evaluate(theta_deg - 0.65)
    p_shifted = 101_325.0 * (np.max(kin_shifted.volume_m3) / kin_shifted.volume_m3) ** 1.35
    with pytest.raises(TDCPhaseError):
        verify_tdc_phasing(theta_deg, p_shifted, kin.volume_m3, tolerance_deg=0.5)

    with pytest.warns(UserWarning, match="TDC phase alignment exceeded"):
        res_bad = verify_tdc_phasing(
            theta_deg, p_shifted, kin.volume_m3, tolerance_deg=0.5, raise_on_error=False
        )
    assert not res_bad.is_valid
    assert res_bad.phase_error_deg == pytest.approx(0.65, abs=5.0e-3)


def test_nasa_continuity_and_wall_heat_transfer_models() -> None:
    """Verify NASA 1000 K continuity, cp-cv=R identity, and Woschni/Hohenberg HRR coupling."""
    t_seam = np.array([300.0, 999.999, 1000.001, 2200.0], dtype=np.float64)
    cp_air = compute_cp(t_seam, DRY_AIR)
    cv_air = compute_cv(t_seam, DRY_AIR)
    gam_air = compute_gamma(t_seam, DRY_AIR)
    gam_prod = compute_gamma(t_seam, DIESEL_BURNED_PRODUCTS)
    gam_stoich = compute_gamma(t_seam, STOICHIOMETRIC_PRODUCTS)

    np.testing.assert_allclose(cp_air - cv_air, DRY_AIR.specific_gas_constant, rtol=1e-13)
    assert gam_air[0] == pytest.approx(1.400, abs=2e-3)
    assert gam_air[1] == pytest.approx(gam_air[2], rel=1e-5)
    assert gam_prod[1] == pytest.approx(gam_prod[2], rel=1e-5)
    assert gam_stoich[3] < gam_prod[3] < gam_air[3]

    with pytest.raises(PhysicalBoundsError):
        compute_cp(np.array([150.0]), DRY_AIR)
    with pytest.raises(PhysicalBoundsError):
        compute_cp(np.array([np.nan]), DRY_AIR)

    geom = SliderCrankGeometry(0.085, 0.090, 0.1575, 16.0)
    theta = np.linspace(-30.0, 30.0, 601, dtype=np.float64)
    kin = geom.evaluate(theta)
    p_mot = 150_000.0 * (geom.evaluate(np.array([-180.0])).volume_m3[0] / kin.volume_m3) ** 1.35
    p_fire = p_mot * (1.0 + 0.6 * np.exp(-((theta - 8.0) / 12.0) ** 2))
    t_gas = 900.0 + 900.0 * np.exp(-((theta - 10.0) / 18.0) ** 2)

    ht_w = compute_woschni_heat_transfer(
        p_fire,
        t_gas,
        kin.wall_area_m2,
        bore_m=0.085,
        stroke_m=0.090,
        rpm=2000.0,
        wall_temp_k=450.0,
        motored_pressure_pa=p_mot,
        p_ref_pa=150_000.0,
        v_ref_m3=geom.displaced_volume_m3 + geom.clearance_volume_m3,
        t_ref_k=330.0,
    )
    ht_w_cls = WoschniCorrelation(0.085, 0.090, rpm=2000.0, wall_temp_k=450.0).evaluate(
        p_fire,
        t_gas,
        kin.wall_area_m2,
        motored_pressure_pa=p_mot,
        p_ref_pa=150_000.0,
        v_ref_m3=geom.displaced_volume_m3 + geom.clearance_volume_m3,
        t_ref_k=330.0,
    )
    np.testing.assert_allclose(ht_w.hc_w_m2k, ht_w_cls.hc_w_m2k, rtol=1e-14)

    ht_h = compute_hohenberg_heat_transfer(
        p_fire, kin.volume_m3, t_gas, kin.wall_area_m2, stroke_m=0.090, rpm=2000.0, wall_temp_k=450.0
    )
    ht_h_cls = HohenbergCorrelation(0.090, rpm=2000.0, wall_temp_k=450.0).evaluate(
        p_fire, kin.volume_m3, t_gas, kin.wall_area_m2
    )
    np.testing.assert_allclose(ht_h.dq_ht_j_per_deg, ht_h_cls.dq_ht_j_per_deg, rtol=1e-14)

    assert 1000.0 < float(np.max(ht_w.hc_w_m2k)) < 6000.0
    assert 1000.0 < float(np.max(ht_h.hc_w_m2k)) < 6000.0

    hrr = compute_heat_release(
        theta,
        p_fire,
        kin.volume_m3,
        temperature_k=t_gas,
        dv_dtheta_m3_per_deg=kin.dv_dtheta_m3_per_deg,
        dq_ht_j_per_deg=ht_h.dq_ht_j_per_deg,
    )
    np.testing.assert_allclose(
        hrr.dq_gross_j_per_deg, hrr.dq_net_j_per_deg + ht_h.dq_ht_j_per_deg, rtol=1e-14
    )


def test_knock_mapo_ringing_intensity_and_forte_parser_integration() -> None:
    """Verify MAPO, Eng (2002) Ringing Intensity, and end-to-end ForteThermoData pipeline."""
    rpm = 2000.0
    dtheta = 0.05  # fs = 6 * 2000 / 0.05 = 240 kHz
    theta = np.arange(-20.0, 40.0, dtheta, dtype=np.float64)
    time_s = (theta - theta[0]) / (6.0 * rpm)

    p_base = 8.0e6 * np.exp(-((theta - 5.0) / 25.0) ** 2) + 1.0e6
    amp_pa = 1.5e5
    envelope = np.exp(-((theta - 6.0) / 4.0) ** 2)
    p_knock = p_base + amp_pa * envelope * np.cos(2.0 * np.pi * 8_000.0 * time_s)
    t_gas = np.full_like(theta, 2100.0)

    metrics = compute_knock_metrics(theta, p_knock, t_gas, rpm=rpm)
    assert metrics.knock_pressure_pa is not None
    assert metrics.mapo_pa == pytest.approx(amp_pa, rel=0.03)
    assert metrics.ringing_intensity_w_m2 > 0.0

    # End-to-end check: non-SI Forte CSV (MPa, cm^3, degC) -> parse_thermo_csv -> compute_heat_release
    geom = SliderCrankGeometry(0.085, 0.090, 0.1575, 16.0)
    ca_csv = np.linspace(-60.0, 60.0, 1201, dtype=np.float64)
    kin_csv = geom.evaluate(ca_csv)
    p_mpa = (101_325.0 * (kin_csv.volume_m3[0] / kin_csv.volume_m3) ** 1.35) * 1.0e-6
    v_cm3 = kin_csv.volume_m3 * 1.0e6
    t_degc = np.full_like(ca_csv, 600.0 - 273.15)
    lines = ["CA[deg],Pressure[MPa],Volume[cm^3],Temp[degC]"] + [
        f"{c:.6f},{p:.12e},{v:.12e},{t:.6f}"
        for c, p, v, t in zip(ca_csv, p_mpa, v_cm3, t_degc, strict=True)
    ]
    forte_data = parse_thermo_csv(StringIO("\n".join(lines)))
    hrr_forte = compute_heat_release(
        forte_data.crank_angle_deg,
        forte_data.pressure_pa,
        forte_data.volume_m3,
        gamma=1.35,
    )
    assert float(np.max(np.abs(hrr_forte.dq_net_j_per_deg))) < 1.0e-8
