"""End-to-end Aero-Diesel high-altitude CFD validation & diagnostic showcase.

Simulates an open-literature 4-stroke turbocharged Aero-Diesel single cylinder across
three flight altitudes (Sea Level 0 ft, 15,000 ft, and 30,000 ft) using strictly
non-proprietary textbook parameters:
  1. ISO 2533 International Standard Atmosphere (ISA) boundary scaling
  2. Closed-cycle First-Law ODE synthesis + zero-phase filtering + Heywood AHRR,
     CA10/CA50/CA90 combustion phasing, and indicated mean effective pressure (IMEP)
  3. High-altitude Jet-A1/Diesel spray penetration scaling (Hiroyasu-Arai 1990 &
     Siebers 1999) and piston-bowl wall impingement risk detection
  4. Three-mesh ASME V&V 20-2009 / Celik et al. (2008) Grid Convergence Index (GCI)
  5. Automated export of four 300-DPI SAE/IEEE publication figures into ``assets/``
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

# Allow direct execution via ``python examples/altitude_sweep_demo.py``
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from ice_cfd_validator import (  # noqa: E402
    DRY_AIR,
    FEET_TO_M,
    AltitudeSprayPlotCase,
    BurnFractionMetrics,
    GCIResult,
    HeatReleaseResult,
    IMEPResult,
    SliderCrankGeometry,
    SprayValidationMetrics,
    ThermoOverlayTrace,
    WoschniCorrelation,
    compute_gci,
    compute_grid_size,
    compute_heat_release,
    compute_imep,
    compute_isa_conditions,
    compute_spray_validation_metrics,
    evaluate_altitude_spray_risk,
    extract_burn_fractions,
    hiroyasu_arai_penetration,
    naber_siebers_penetration,
    plot_altitude_spray_comparison,
    plot_gci_convergence,
    plot_pv_log_diagram,
    plot_thermo_diagnostics,
    savgol_zero_phase,
)

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class AltitudeScenarioSummary:
    """Aggregated engineering diagnostics for a single flight altitude."""

    label: str
    altitude_ft: float
    altitude_m: float
    t_amb_k: float
    p_amb_kpa: float
    rho_amb_kg_m3: float
    p_ivc_kpa: float
    t_ivc_k: float
    rho_soi_kg_m3: float
    q_fuel_j: float
    theta_deg: FloatArray
    volume_m3: FloatArray
    pressure_pa: FloatArray
    hrr: HeatReleaseResult
    burn: BurnFractionMetrics
    imep: IMEPResult
    spray_plot_case: AltitudeSprayPlotCase
    spray_metrics: SprayValidationMetrics
    wall_impingement: bool
    impingement_time_ms: float | None
    gci: GCIResult


def _wiebe_heat_rate_j_per_deg(
    theta_deg: float,
    *,
    q_total_j: float,
    theta_soi_deg: float,
    duration_deg: float,
    a_factor: float = 6.908,
    m_shape: float = 2.0,
) -> float:
    """Single-Wiebe analytical chemical heat-release rate $dQ_{ch}/d\\theta$ [J/deg]."""
    if theta_deg <= theta_soi_deg or theta_deg >= theta_soi_deg + duration_deg:
        return 0.0
    u = (theta_deg - theta_soi_deg) / duration_deg
    return (
        q_total_j
        * (a_factor * (m_shape + 1.0) / duration_deg)
        * (u**m_shape)
        * math.exp(-a_factor * (u ** (m_shape + 1.0)))
    )


def simulate_aero_diesel_altitude_case(
    altitude_ft: float,
    geom: SliderCrankGeometry,
    *,
    engine_speed_rpm: float = 2400.0,
    wall_temp_k: float = 460.0,
    bowl_wall_distance_m: float = 0.038,
    rng_seed: int = 42,
) -> AltitudeScenarioSummary:
    """Synthesize and validate one Aero-Diesel operating point at ``altitude_ft``."""
    alt_m = altitude_ft * FEET_TO_M
    isa = compute_isa_conditions(alt_m)
    t_amb_k = float(isa.temperature_k[0])
    p_amb_pa = float(isa.pressure_pa[0])
    rho_amb = float(isa.density_kg_m3[0])

    # Single-stage fixed-pressure-ratio turbocharger + air-to-air intercooler
    pi_comp = 2.40
    eta_comp = 0.76
    intercooler_eff = 0.70
    coolant_sink_k = max(t_amb_k + 12.0, 295.0)

    p_ivc_pa = p_amb_pa * pi_comp
    t_comp_out_k = t_amb_k * (1.0 + ((pi_comp ** (0.4 / 1.4)) - 1.0) / eta_comp)
    t_ivc_k = t_comp_out_k - intercooler_eff * (t_comp_out_k - coolant_sink_k) + 18.0

    # Crank-angle grid [-180, +180] deg ATDC (closed high-pressure cycle)
    theta_deg = np.linspace(-180.0, 180.0, 3601, dtype=np.float64)
    kin = geom.evaluate(theta_deg)
    r_spec = DRY_AIR.specific_gas_constant
    v_ivc_m3 = float(kin.volume_m3[0])
    trapped_mass_kg = (p_ivc_pa * v_ivc_m3) / (r_spec * t_ivc_k)

    # Fueling limited by minimum excess-air ratio (lambda >= 1.45 at high altitude)
    afr_stoich = 14.6
    lhv_jet_a1_j_per_kg = 43.2e6
    max_clean_fuel_kg = trapped_mass_kg / (1.45 * afr_stoich)
    nominal_fuel_kg = 2.45e-5  # 24.5 mg/cycle nominal cruise demand
    injected_fuel_kg = min(nominal_fuel_kg, max_clean_fuel_kg)
    q_fuel_j = injected_fuel_kg * lhv_jet_a1_j_per_kg * 0.985

    # Ignition delay lengthens as intake manifold pressure drops with altitude
    id_shift_deg = 2.2 * ((101_325.0 * pi_comp / p_ivc_pa) ** 0.38 - 1.0)
    theta_soi_comb_deg = -7.5 + id_shift_deg
    burn_duration_deg = 38.0 + 1.8 * id_shift_deg

    mean_piston_speed_m_s = 2.0 * geom.stroke_m * (engine_speed_rpm / 60.0)
    omega_deg_per_s = 6.0 * engine_speed_rpm

    # Forward closed-cycle First-Law ODE integration:
    # dP/dtheta = ((gamma(T)-1)/V) * (dQ_ch/dtheta - dQ_ht/dtheta) - gamma(T)*(P/V)*(dV/dtheta)
    def cycle_ode(th: float, state: FloatArray) -> list[float]:
        p_curr = max(float(state[0]), 1.0e3)
        th_arr = np.array([th], dtype=np.float64)
        k_pt = geom.evaluate(th_arr)
        v_curr = float(k_pt.volume_m3[0])
        dv_dth = float(k_pt.dv_dtheta_m3_per_deg[0])
        a_wall = float(k_pt.wall_area_m2[0])

        t_curr = min(max((p_curr * v_curr) / (trapped_mass_kg * r_spec), 210.0), 5800.0)
        gamma_curr = float(np.asarray(DRY_AIR.gamma(t_curr)).item())

        dq_ch = _wiebe_heat_rate_j_per_deg(
            th,
            q_total_j=q_fuel_j,
            theta_soi_deg=theta_soi_comb_deg,
            duration_deg=burn_duration_deg,
        )

        # Motored reference pressure for Woschni combustion-induced swirl/velocity term
        p_mot = p_ivc_pa * ((v_ivc_m3 / v_curr) ** 1.35)
        w_gas = 2.28 * mean_piston_speed_m_s + 3.24e-3 * (
            geom.displaced_volume_m3 * t_ivc_k / (p_ivc_pa * v_ivc_m3)
        ) * max(p_curr - p_mot, 0.0)
        hc = (
            130.0
            * (geom.bore_m**-0.2)
            * ((p_curr * 1.0e-5) ** 0.8)
            * (t_curr**-0.53)
            * (w_gas**0.8)
        )
        dq_ht = hc * a_wall * (t_curr - wall_temp_k) / omega_deg_per_s

        dp_dth = ((gamma_curr - 1.0) / v_curr) * (dq_ch - dq_ht) - (
            gamma_curr * p_curr / v_curr
        ) * dv_dth
        return [dp_dth]

    ode_sol = solve_ivp(
        cycle_ode,
        t_span=(-180.0, 180.0),
        y0=[p_ivc_pa],
        t_eval=theta_deg,
        method="RK45",
        rtol=1.0e-8,
        atol=1.0e-4,
    )
    p_exact_pa = np.asarray(ode_sol.y[0], dtype=np.float64)

    # Add realistic piezoelectric transducer noise (3.5 kPa RMS) & apply zero-phase filter
    rng = np.random.default_rng(rng_seed + int(round(altitude_ft)))
    p_measured_pa = p_exact_pa + rng.normal(0.0, 3_500.0, size=p_exact_pa.shape)
    p_filtered_pa = savgol_zero_phase(p_measured_pa, window_length=31, polyorder=3)
    t_filtered_k = (p_filtered_pa * kin.volume_m3) / (trapped_mass_kg * r_spec)

    # Woschni (1967) wall heat-transfer & Heywood (2018) inverse First-Law AHRR
    p_motored_arr = p_ivc_pa * ((v_ivc_m3 / kin.volume_m3) ** 1.35)
    woschni = WoschniCorrelation(
        bore_m=geom.bore_m,
        stroke_m=geom.stroke_m,
        rpm=engine_speed_rpm,
        wall_temp_k=wall_temp_k,
        displaced_volume_m3=geom.displaced_volume_m3,
    )
    ht_res = woschni.evaluate(
        p_filtered_pa,
        t_filtered_k,
        kin.wall_area_m2,
        motored_pressure_pa=p_motored_arr,
        p_ref_pa=p_ivc_pa,
        v_ref_m3=v_ivc_m3,
        t_ref_k=t_ivc_k,
    )
    hrr_res = compute_heat_release(
        theta_deg,
        p_filtered_pa,
        kin.volume_m3,
        temperature_k=t_filtered_k,
        dv_dtheta_m3_per_deg=kin.dv_dtheta_m3_per_deg,
        dq_ht_j_per_deg=ht_res.dq_ht_j_per_deg,
        mixture=DRY_AIR,
    )
    burn_res = extract_burn_fractions(
        theta_deg,
        hrr_res.dq_gross_j_per_deg,
        start_deg=-12.0,
        end_deg=65.0,
    )
    imep_res = compute_imep(
        theta_deg,
        p_filtered_pa,
        kin.volume_m3,
        displaced_volume_m3=geom.displaced_volume_m3,
        dv_dtheta_m3_per_deg=kin.dv_dtheta_m3_per_deg,
    )

    # In-cylinder gas density at Start of Injection (-8 deg ATDC) for spray scaling
    soi_idx = int(np.argmin(np.abs(theta_deg - (-8.0))))
    rho_soi = float(trapped_mass_kg / kin.volume_m3[soi_idx])

    # High-pressure Jet-A1/Diesel spray evaluation over [0, 0.85 ms]
    t_spray_s = np.linspace(1.0e-6, 0.85e-3, 300, dtype=np.float64)
    delta_p_inj_pa = 100.0e6
    rho_fuel_kg_m3 = 810.0
    nozzle_d_m = 90.0e-6
    spray_cone_deg = 19.0

    ha_res = hiroyasu_arai_penetration(
        t_spray_s,
        delta_p_pa=delta_p_inj_pa,
        rho_liquid_kg_m3=rho_fuel_kg_m3,
        rho_gas_kg_m3=rho_soi,
        nozzle_diameter_m=nozzle_d_m,
    )
    ns_res = naber_siebers_penetration(
        t_spray_s,
        delta_p_pa=delta_p_inj_pa,
        rho_liquid_kg_m3=rho_fuel_kg_m3,
        rho_gas_kg_m3=rho_soi,
        nozzle_diameter_m=nozzle_d_m,
        spray_angle_deg=spray_cone_deg,
    )

    # Synthetic ANSYS Forte 3D CFD discrete spray samples (14 RANS output checkpoints)
    t_cfd_s = np.linspace(0.05e-3, 0.85e-3, 14, dtype=np.float64)
    ha_cfd_ref = hiroyasu_arai_penetration(
        t_cfd_s,
        delta_p_pa=delta_p_inj_pa,
        rho_liquid_kg_m3=rho_fuel_kg_m3,
        rho_gas_kg_m3=rho_soi,
        nozzle_diameter_m=nozzle_d_m,
    )
    ns_cfd_ref = naber_siebers_penetration(
        t_cfd_s,
        delta_p_pa=delta_p_inj_pa,
        rho_liquid_kg_m3=rho_fuel_kg_m3,
        rho_gas_kg_m3=rho_soi,
        nozzle_diameter_m=nozzle_d_m,
        spray_angle_deg=spray_cone_deg,
    )
    cfd_pen_m = (
        0.62 * ha_cfd_ref.penetration_m
        + 0.38 * ns_cfd_ref.penetration_m
        + rng.normal(0.0, 0.35e-3, size=t_cfd_s.shape)
    )
    spray_val_metrics = compute_spray_validation_metrics(
        cfd_pen_m,
        ha_res.penetration_m,
        cfd_time_s=t_cfd_s,
        ref_time_s=t_spray_s,
    )

    # Wall impingement onset relative to piston bowl radius (L_wall = 38.0 mm)
    risk_res = evaluate_altitude_spray_risk(
        t_spray_s,
        delta_p_pa=delta_p_inj_pa,
        rho_liquid_kg_m3=rho_fuel_kg_m3,
        rho_gas_sea_level_kg_m3=max(rho_soi, 26.0),
        rho_gas_altitude_kg_m3=rho_soi,
        nozzle_diameter_m=nozzle_d_m,
        wall_distance_m=bowl_wall_distance_m,
        model="hiroyasu_arai",
    )

    label_str = (
        f"Sea Level (0 ft)"
        if altitude_ft == 0.0
        else f"{int(altitude_ft):,} ft ({alt_m:,.0f} m)"
    )
    spray_case = AltitudeSprayPlotCase(
        label=label_str,
        rho_gas_kg_m3=rho_soi,
        time_s=t_spray_s,
        hiroyasu_penetration_m=ha_res.penetration_m,
        siebers_penetration_m=ns_res.penetration_m,
        cfd_time_s=t_cfd_s,
        cfd_penetration_m=cfd_pen_m,
        impingement_time_s=risk_res.impingement_time_s,
    )

    # Three-grid ASME V&V 20-2009 mesh independence study for Peak Cylinder Pressure
    grid_sizes = compute_grid_size(
        cell_counts=(1_920_000.0, 680_000.0, 235_000.0),
        domain_volume_m3=float(np.max(kin.volume_m3)),
    )
    p_peak_true = float(np.max(p_filtered_pa))
    # Realistic 2nd-order RANS spatial truncation error (~0.85% on fine mesh, p_eff ~ 1.92)
    c_mesh = -0.0085 * p_peak_true / (grid_sizes.h1**1.92)
    phi1_pa = p_peak_true + c_mesh * (grid_sizes.h1**1.92)
    phi2_pa = p_peak_true + c_mesh * (grid_sizes.h2**1.92)
    phi3_pa = p_peak_true + c_mesh * (grid_sizes.h3**1.92)
    gci_res = compute_gci(phi1_pa, phi2_pa, phi3_pa, grid=grid_sizes)

    return AltitudeScenarioSummary(
        label=label_str,
        altitude_ft=altitude_ft,
        altitude_m=alt_m,
        t_amb_k=t_amb_k,
        p_amb_kpa=p_amb_pa * 1.0e-3,
        rho_amb_kg_m3=rho_amb,
        p_ivc_kpa=p_ivc_pa * 1.0e-3,
        t_ivc_k=t_ivc_k,
        rho_soi_kg_m3=rho_soi,
        q_fuel_j=q_fuel_j,
        theta_deg=theta_deg,
        volume_m3=kin.volume_m3,
        pressure_pa=p_filtered_pa,
        hrr=hrr_res,
        burn=burn_res,
        imep=imep_res,
        spray_plot_case=spray_case,
        spray_metrics=spray_val_metrics,
        wall_impingement=risk_res.wall_impingement_detected,
        impingement_time_ms=(
            risk_res.impingement_time_s * 1.0e3
            if risk_res.impingement_time_s is not None
            else None
        ),
        gci=gci_res,
    )


def _print_summary_table(results: list[AltitudeScenarioSummary]) -> None:
    """Print an engineering summary table of thermodynamic, spray, and GCI metrics."""
    sep = "=" * 116
    print("\n" + sep)
    print(
        "  ICE-CFD-VALIDATION-ASSISTANT | AERO-DIESEL ALTITUDE SWEEP DIAGNOSTIC & V&V SUMMARY"
    )
    print(sep)
    header = (
        f"{'Altitude Scenario':<22} | {'P_amb':>7} | {'rho_SOI':>7} | {'P_max':>7} | "
        f"{'IMEP_n':>7} | {'CA50':>6} | {'S_max':>6} | {'Wall Imp.':>11} | "
        f"{'Order p':>7} | {'GCI_fine':>8}"
    )
    units = (
        f"{'(ISA ISO 2533)':<22} | {'[kPa]':>7} | {'[kg/m3]':>7} | {'[MPa]':>7} | "
        f"{'[bar]':>7} | {'[deg]':>6} | {'[mm]':>6} | {'(L=38.0mm)':>11} | "
        f"{'[-]':>7} | {'[%]':>8}"
    )
    print(header)
    print(units)
    print("-" * 116)

    for r in results:
        p_max_mpa = float(np.max(r.pressure_pa)) * 1.0e-6
        imep_bar = r.imep.imep_net_pa * 1.0e-5
        s_max_mm = float(np.max(r.spray_plot_case.hiroyasu_penetration_m)) * 1.0e3
        imp_str = (
            f"t={r.impingement_time_ms:.2f} ms"
            if r.impingement_time_ms is not None
            else "SAFE"
        )
        print(
            f"{r.label:<22} | {r.p_amb_kpa:7.2f} | {r.rho_soi_kg_m3:7.2f} | "
            f"{p_max_mpa:7.2f} | {imep_bar:7.2f} | {r.burn.ca50_deg:6.2f} | "
            f"{s_max_mm:6.1f} | {imp_str:>11} | {r.gci.apparent_order_p:7.3f} | "
            f"{r.gci.gci_fine_21 * 100.0:7.3f}%"
        )
    print(sep + "\n")


def main() -> None:
    """Execute the 3-altitude Aero-Diesel validation sweep and export 300-DPI figures."""
    geom = SliderCrankGeometry(
        bore_m=0.085,
        stroke_m=0.090,
        conrod_m=0.155,
        compression_ratio=16.5,
    )
    bowl_wall_distance_m = 0.038

    altitudes_ft = [0.0, 15_000.0, 30_000.0]
    summaries = [
        simulate_aero_diesel_altitude_case(
            alt_ft,
            geom,
            bowl_wall_distance_m=bowl_wall_distance_m,
        )
        for alt_ft in altitudes_ft
    ]

    _print_summary_table(summaries)

    assets_dir = _REPO_ROOT / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    sl, mid, high = summaries[0], summaries[1], summaries[2]

    # 1. Three-panel thermodynamic cycle diagnostics (Sea Level + 15k/30k overlays)
    thermo_path = assets_dir / "thermo_diagnostics.png"
    plot_thermo_diagnostics(
        sl.theta_deg,
        sl.pressure_pa,
        sl.hrr,
        sl.burn,
        primary_label=sl.label,
        overlay_traces=[
            ThermoOverlayTrace(
                label=mid.label,
                theta_deg=mid.theta_deg,
                pressure_pa=mid.pressure_pa,
                hrr_result=mid.hrr,
                burn_metrics=mid.burn,
            ),
            ThermoOverlayTrace(
                label=high.label,
                theta_deg=high.theta_deg,
                pressure_pa=high.pressure_pa,
                hrr_result=high.hrr,
                burn_metrics=high.burn,
            ),
        ],
        save_path=thermo_path,
        dpi=300,
    )

    # 2. Logarithmic P-V diagram & polytropic exponent verification
    pv_path = assets_dir / "pv_log_diagram.png"
    plot_pv_log_diagram(
        sl.volume_m3,
        sl.pressure_pa,
        sl.theta_deg,
        primary_label=sl.label,
        overlay_loops=[
            (mid.label, mid.volume_m3, mid.pressure_pa),
            (high.label, high.volume_m3, high.pressure_pa),
        ],
        save_path=pv_path,
        dpi=300,
    )

    # 3. High-altitude spray penetration & wall-impingement risk chart
    spray_path = assets_dir / "altitude_spray_comparison.png"
    plot_altitude_spray_comparison(
        [s.spray_plot_case for s in summaries],
        wall_distance_m=bowl_wall_distance_m,
        save_path=spray_path,
        dpi=300,
    )

    # 4. ASME V&V 20-2009 3-mesh GCI convergence & Richardson extrapolation chart
    gci_path = assets_dir / "gci_convergence.png"
    plot_gci_convergence(
        sl.gci,
        metric_name=r"Sea-Level Peak Cylinder Pressure, $P_{\max}$",
        metric_unit="MPa",
        scale_factor=1.0e-6,
        save_path=gci_path,
        dpi=300,
    )

    print("Saved 300-DPI SAE/IEEE publication figures:")
    for p in (thermo_path, pv_path, spray_path, gci_path):
        size_kb = p.stat().st_size / 1024.0
        print(f"  [OK] {p.relative_to(_REPO_ROOT).as_posix()} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
