"""Independent strict physical, mathematical, and software audit for Steps 3, 4 & 5."""

from __future__ import annotations

import ast
import math
import os
import shutil
import subprocess
import sys
import sysconfig
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ice_cfd_validator.constants import compute_isa_conditions
from ice_cfd_validator.exceptions import PhysicalBoundsError
from ice_cfd_validator.io import load_ecn_spray_data, parse_spray_csv
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
from ice_cfd_validator.vv import compute_gci, compute_grid_size


def test_audit_1_hiroyasu_arai_breakup_regime_and_density_scaling() -> None:
    """DENETIM 1: Hiroyasu-Arai (1990) pre/post breakup regimes, rho_g^-0.25 scaling & continuity."""
    delta_p_pa = 1.5e8        # 150 MPa
    rho_l = 800.0             # kg/m^3 (Jet-A1 / Aero-Diesel)
    d_o = 1.4e-4              # 140 um
    rho_g_sl = 22.0           # kg/m^3 (Sea Level)
    rho_g_alt = 11.0          # kg/m^3 (High Altitude, 0.5 * rho_g_sl)

    # Probe breakup times first
    probe_sl = hiroyasu_arai_penetration(
        [1.0e-6],
        delta_p_pa=delta_p_pa,
        rho_liquid_kg_m3=rho_l,
        rho_gas_kg_m3=rho_g_sl,
        nozzle_diameter_m=d_o,
    )
    probe_alt = hiroyasu_arai_penetration(
        [1.0e-6],
        delta_p_pa=delta_p_pa,
        rho_liquid_kg_m3=rho_l,
        rho_gas_kg_m3=rho_g_alt,
        nozzle_diameter_m=d_o,
    )
    tb_sl = probe_sl.breakup_time_s
    tb_alt = probe_alt.breakup_time_s

    # Kontrol A: Early regime (t = 0.5 * tb_sl < tb_sl < tb_alt) is strictly independent of rho_g
    t_early = 0.5 * tb_sl
    s1_early = float(
        hiroyasu_arai_penetration(
            [t_early],
            delta_p_pa=delta_p_pa,
            rho_liquid_kg_m3=rho_l,
            rho_gas_kg_m3=rho_g_sl,
            nozzle_diameter_m=d_o,
        ).penetration_m[0]
    )
    s2_early = float(
        hiroyasu_arai_penetration(
            [t_early],
            delta_p_pa=delta_p_pa,
            rho_liquid_kg_m3=rho_l,
            rho_gas_kg_m3=rho_g_alt,
            nozzle_diameter_m=d_o,
        ).penetration_m[0]
    )
    early_abs_diff = abs(s1_early - s2_early)

    # Kontrol B: Post-breakup regime (t = 1.5 ms > max(tb_sl, tb_alt)) scales as 2^0.25
    t_late = 1.5e-3
    assert t_late > max(tb_sl, tb_alt)
    s1_late = float(
        hiroyasu_arai_penetration(
            [t_late],
            delta_p_pa=delta_p_pa,
            rho_liquid_kg_m3=rho_l,
            rho_gas_kg_m3=rho_g_sl,
            nozzle_diameter_m=d_o,
        ).penetration_m[0]
    )
    s2_late = float(
        hiroyasu_arai_penetration(
            [t_late],
            delta_p_pa=delta_p_pa,
            rho_liquid_kg_m3=rho_l,
            rho_gas_kg_m3=rho_g_alt,
            nozzle_diameter_m=d_o,
        ).penetration_m[0]
    )
    late_ratio = s2_late / s1_late
    exact_two_pow_025 = 2.0**0.25
    late_ratio_err = abs(late_ratio - exact_two_pow_025)

    # Kontrol C: Left/right limit continuity at t = tb_sl (both default C0 & textbook c_pre=0.39)
    t_left = float(np.nextafter(tb_sl, 0.0))
    t_right = tb_sl
    res_cont = hiroyasu_arai_penetration(
        [t_left, t_right],
        delta_p_pa=delta_p_pa,
        rho_liquid_kg_m3=rho_l,
        rho_gas_kg_m3=rho_g_sl,
        nozzle_diameter_m=d_o,
    )
    cont_jump_m = abs(float(res_cont.penetration_m[1] - res_cont.penetration_m[0]))

    res_rounded = hiroyasu_arai_penetration(
        [t_left, t_right],
        delta_p_pa=delta_p_pa,
        rho_liquid_kg_m3=rho_l,
        rho_gas_kg_m3=rho_g_sl,
        nozzle_diameter_m=d_o,
        c_breakup=28.65,
        c_post=2.95,
        c_pre=0.39,
    )
    rounded_rel_jump_pct = (
        abs(float(res_rounded.penetration_m[1] - res_rounded.penetration_m[0]))
        / float(res_rounded.penetration_m[1])
        * 100.0
    )

    print("\n" + "=" * 78)
    print("[DENETIM 1] Hiroyasu-Arai (1990) Kirilma Ani (t_b) ve Yogunluk Olcekleme")
    print(f"  t_b (SL, 22 kg/m^3)          : {tb_sl * 1e6:.6f} us | t_b (Alt, 11 kg/m^3): {tb_alt * 1e6:.6f} us")
    print(f"  Kontrol A |S1 - S2|(0.5*tb)  : {early_abs_diff:.3e} m (Sinir < 1e-15 m)")
    print(f"  Kontrol B S2/S1 (1.5 ms)     : {late_ratio:.12f} vs 2^0.25={exact_two_pow_025:.12f} (|hata|={late_ratio_err:.3e})")
    print(f"  Kontrol C |S(tb+)-S(tb-)| C0 : {cont_jump_m:.3e} m (< 1e-12 m) | Yuvarlatilmis (0.39) bagil fark: %{rounded_rel_jump_pct:.5f} (< %0.08)")
    print("=" * 78)

    assert early_abs_diff < 1.0e-15
    assert late_ratio_err < 1.0e-12
    assert cont_jump_m < 1.0e-12
    assert rounded_rel_jump_pct < 0.08


def test_audit_2_naber_siebers_asymptotics_half_angle_and_unit_scaling(tmp_path: Path) -> None:
    """DENETIM 2: Naber-Siebers asymptotics, half-angle tan(theta/2) trap & ms/mm unit scaling."""
    delta_p_pa = 1.2e8
    rho_l = 810.0
    rho_g = 18.0
    d_o = 1.0e-4

    ref_20 = naber_siebers_penetration(
        [1.0e-4],
        delta_p_pa=delta_p_pa,
        rho_liquid_kg_m3=rho_l,
        rho_gas_kg_m3=rho_g,
        nozzle_diameter_m=d_o,
        spray_angle_deg=20.0,
    )
    t_plus_20 = ref_20.t_plus_s

    # Kontrol A: Dimensionless short-time (t_tilde = 1e-4) and long-time (t_tilde = 1e4) limits
    ns_asym = naber_siebers_penetration(
        [1.0e-4 * t_plus_20, 1.0e4 * t_plus_20],
        delta_p_pa=delta_p_pa,
        rho_liquid_kg_m3=rho_l,
        rho_gas_kg_m3=rho_g,
        nozzle_diameter_m=d_o,
        spray_angle_deg=20.0,
    )
    short_ratio = float(ns_asym.s_tilde[0] / ns_asym.t_tilde[0])
    long_ratio = float(ns_asym.s_tilde[1] / math.sqrt(float(ns_asym.t_tilde[1])))
    short_err = abs(short_ratio - 1.0)
    long_err = abs(long_ratio - 1.0)

    # Kontrol B: Half-angle & deg->rad check (theta_cone = 20 deg -> tan(10 deg) vs 30 deg -> tan(15 deg))
    expected_x_plus_20 = (d_o * math.sqrt(rho_l / rho_g)) / (0.66 * math.tan(math.radians(10.0)))
    x_plus_rel_err = abs(ref_20.x_plus_m - expected_x_plus_20) / expected_x_plus_20

    t_far_s = 1.0e6 * t_plus_20
    ns_20_far = naber_siebers_penetration(
        [t_far_s],
        delta_p_pa=delta_p_pa,
        rho_liquid_kg_m3=rho_l,
        rho_gas_kg_m3=rho_g,
        nozzle_diameter_m=d_o,
        spray_angle_deg=20.0,
    )
    ns_30_far = naber_siebers_penetration(
        [t_far_s],
        delta_p_pa=delta_p_pa,
        rho_liquid_kg_m3=rho_l,
        rho_gas_kg_m3=rho_g,
        nozzle_diameter_m=d_o,
        spray_angle_deg=30.0,
    )
    cone_pen_ratio = float(ns_30_far.penetration_m[0] / ns_20_far.penetration_m[0])
    expected_cone_ratio = math.sqrt(math.tan(math.radians(10.0)) / math.tan(math.radians(15.0)))
    cone_ratio_err = abs(cone_pen_ratio - expected_cone_ratio)

    # Kontrol C: Forte spray.csv and ECN loader ms -> s and mm -> m exact scaling
    forte_csv = tmp_path / "spray.csv"
    forte_csv.write_text(
        "Time_(ms),Liquid_Penetration_(mm),Vapor_Penetration_(mm)\n"
        "1.0,45.0,52.5\n",
        encoding="utf-8",
    )
    ecn_csv = tmp_path / "ECN_SprayA.csv"
    ecn_csv.write_text(
        "# Sandia ECN Spray A synthetic unit test\n"
        "Time [ms], Liquid_Penetration [mm], Vapor_Penetration [mm]\n"
        "1.0, 45.0, 52.5\n",
        encoding="utf-8",
    )
    forte_parsed = parse_spray_csv(forte_csv)
    ecn_parsed = load_ecn_spray_data(ecn_csv)
    assert ecn_parsed.liquid_penetration_m is not None

    print("\n" + "=" * 78)
    print("[DENETIM 2] Naber & Siebers (1996) Asimptotlar, Koni Yari Acisi ve Birim Testi")
    print(f"  Kontrol A Kisa Limit (t~=1e-4): S~/t~ = {short_ratio:.8f} (|hata|={short_err:.3e} < 1e-3)")
    print(f"  Kontrol A Uzun Limit (t~=1e+4): S~/sqrt(t~) = {long_ratio:.8f} (|hata|={long_err:.3e} < 1e-2)")
    print(f"  Kontrol B x+(20 deg) tan(10°) : {ref_20.x_plus_m:.9f} m (bagil hata={x_plus_rel_err:.3e})")
    print(f"  Kontrol B S(30°)/S(20°) Uzun  : {cone_pen_ratio:.8f} vs analitik={expected_cone_ratio:.8f} (|hata|={cone_ratio_err:.3e})")
    print(f"  Kontrol C Forte (t, S) SI     : ({float(forte_parsed.time_s[0]):.6e} s, {float(forte_parsed.liquid_penetration_m[0]):.6f} m)")
    print(f"  Kontrol C ECN   (t, S) SI     : ({float(ecn_parsed.time_s[0]):.6e} s, {float(ecn_parsed.liquid_penetration_m[0]):.6f} m)")
    print("=" * 78)

    assert short_err < 1.0e-3
    assert long_err < 1.0e-2
    assert x_plus_rel_err < 1.0e-14
    assert cone_ratio_err < 1.0e-4
    assert float(forte_parsed.time_s[0]) == pytest.approx(1.0e-3, rel=1.0e-15, abs=1.0e-18)
    assert float(forte_parsed.liquid_penetration_m[0]) == pytest.approx(0.045, rel=1.0e-15, abs=1.0e-18)
    assert float(ecn_parsed.time_s[0]) == pytest.approx(1.0e-3, rel=1.0e-15, abs=1.0e-18)
    assert float(ecn_parsed.liquid_penetration_m[0]) == pytest.approx(0.045, rel=1.0e-15, abs=1.0e-18)


def test_audit_3_asme_gci_nonuniform_transcendental_3d_cube_root_and_amr() -> None:
    """DENETIM 3: Non-uniform Celik q(p) transcendental test, 3D (N1/Ni)^(1/3) & AMR guard."""
    # Kontrol A: phi(h) = 12.5 + 0.85 * h^1.85 on h = (1.0, 1.4, 2.38) -> r21=1.4, r32=1.7
    phi_exact = 12.500000
    c_coeff = 0.85
    p_true = 1.850000
    h1, h2, h3 = 1.0, 1.4, 2.38

    phi1 = phi_exact + c_coeff * (h1**p_true)
    phi2 = phi_exact + c_coeff * (h2**p_true)
    phi3 = phi_exact + c_coeff * (h3**p_true)

    gci = compute_gci(phi1, phi2, phi3, h1=h1, h2=h2, h3=h3, tol=1.0e-12)
    p_err = abs(gci.apparent_order_p - p_true)
    phi_ext_err = abs(gci.phi_ext_21 - phi_exact)

    # Kontrol B: 3D cell counts N1=800,000, N2=100,000, N3=12,500 -> r21 = r32 = 2.0 (NOT 8.0)
    grid_3d = compute_grid_size(cell_counts=(800_000, 100_000, 12_500))

    # Kontrol C: AMR oscillatory regime & reversed grid ordering guard
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        amr_gci = compute_gci(8.05, 8.12, 8.08, r21=1.4, r32=1.7)

    with pytest.raises(PhysicalBoundsError):
        compute_grid_size(cell_counts=(12_500, 100_000, 800_000))
    with pytest.raises(PhysicalBoundsError):
        compute_gci(phi3, phi2, phi1, h1=h3, h2=h2, h3=h1)

    print("\n" + "=" * 78)
    print("[DENETIM 3] ASME V&V 20-2009 GCI Transandantal q(p), 3B Kupkok ve AMR Denetimi")
    print(f"  Kontrol A p_bulunan vs p_gercek : {gci.apparent_order_p:.10f} vs {p_true:.10f} (|hata|={p_err:.3e} < 1e-9)")
    print(f"  Kontrol A phi_ext^21 vs exact   : {gci.phi_ext_21:.10f} vs {phi_exact:.10f} (|hata|={phi_ext_err:.3e} < 1e-9)")
    print(f"  Kontrol A q(p) terimi           : {gci.q_offset:.10f}")
    print(f"  Kontrol B 3B r21, r32 (800k/100k/12.5k): r21={grid_3d.r21:.8f}, r32={grid_3d.r32:.8f} (1B tuzagi=8.0)")
    print(f"  Kontrol C AMR Rejim & p         : regime={amr_gci.convergence_regime}, s={amr_gci.sign_s:+.1f}, p={amr_gci.apparent_order_p:.6f}")
    print("=" * 78)

    assert p_err < 1.0e-9
    assert phi_ext_err < 1.0e-9
    assert abs(grid_3d.r21 - 2.0) < 1.0e-12
    assert abs(grid_3d.r32 - 2.0) < 1.0e-12
    assert amr_gci.convergence_regime == "OSCILLATORY"
    assert math.isfinite(amr_gci.apparent_order_p) and math.isfinite(amr_gci.gci_fine_21)


def test_audit_4_isa_iso2533_sea_level_tropopause_and_stratosphere() -> None:
    """DENETIM 4: ISO 2533 ISA state at 0 m, 11,000 m tropopause & 15,000 m isothermal stratosphere."""
    isa = compute_isa_conditions([0.0, 11_000.0, 15_000.0])

    t0, p0, rho0 = float(isa.temperature_k[0]), float(isa.pressure_pa[0]), float(isa.density_kg_m3[0])
    t11, p11, rho11 = float(isa.temperature_k[1]), float(isa.pressure_pa[1]), float(isa.density_kg_m3[1])
    t15, p15, rho15 = float(isa.temperature_k[2]), float(isa.pressure_pa[2]), float(isa.density_kg_m3[2])

    print("\n" + "=" * 78)
    print("[DENETIM 4] Uluslararasi Standart Atmosfer (ISA - ISO 2533) Denetimi")
    print(f"  Kontrol A (z =     0 m) : T = {t0:.4f} K | P = {p0:.3f} Pa | rho = {rho0:.6f} kg/m^3")
    print(f"  Kontrol B (z = 11000 m) : T = {t11:.4f} K | P = {p11:.3f} Pa | rho = {rho11:.6f} kg/m^3")
    print(f"  Kontrol C (z = 15000 m) : T = {t15:.4f} K | P = {p15:.3f} Pa | rho = {rho15:.6f} kg/m^3")
    print("=" * 78)

    # Kontrol A: Sea Level (0 m)
    assert t0 == pytest.approx(288.15, abs=1.0e-10)
    assert p0 == pytest.approx(101_325.0, abs=1.0e-8)
    assert rho0 == pytest.approx(1.2250, abs=1.0e-4)

    # Kontrol B: Tropopause (11,000 m)
    assert t11 == pytest.approx(216.65, abs=1.0e-10)
    assert abs(p11 - 22_632.06) <= 5.0
    assert abs(rho11 - 0.36392) <= 1.0e-4

    # Kontrol C: Lower Stratosphere (15,000 m) — strictly isothermal at 216.65 K (NOT 190.65 K)
    assert abs(t15 - 190.65) > 25.0
    assert t15 == pytest.approx(216.65, abs=1.0e-10)
    assert abs(p15 - 12_044.57) <= 10.0


def test_audit_5_headless_plotting_memory_leak_and_demo_execution(tmp_path: Path) -> None:
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

    # Run examples/altitude_sweep_demo.py under explicit headless MPLBACKEND=Agg
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

    print("\n" + "=" * 78)
    print("[DENETIM 5] Headless Matplotlib (Agg), Bellek Sizintisi ve Demo Denetimi")
    print(f"  Matplotlib Test Ortami         : rcParams svg.fonttype={matplotlib.rcParams['svg.fonttype']}")
    print(f"  Acik Kalan Pyplot Figur Sayisi : once={figs_before}, sonra={figs_after} (0 sizinti)")
    for fname, size_kb, magic_ok in asset_info:
        print(f"  assets/{fname:<32}: {size_kb:7.1f} KB | PNG Imzasi (\\x89PNG) = {magic_ok}")
    print("=" * 78)

    assert figs_after == 0
    for fname, size_kb, magic_ok in asset_info:
        assert magic_ok, f"Invalid PNG header in {fname}"
        assert size_kb > 30.0, f"Expected > 30 KB for 300-DPI figure {fname}, got {size_kb:.1f} KB"


def test_audit_6_project_wide_ast_vectorization_and_pyrefly() -> None:
    """DENETIM 6: Project-wide AST loop audit across all numerical modules + pyrefly 0-error check."""
    project_root = Path(__file__).resolve().parents[1]
    pkg_root = project_root / "src" / "ice_cfd_validator"
    numerical_targets = {
        pkg_root / "signal",
        pkg_root / "thermo",
        pkg_root / "spray",
        pkg_root / "vv",
    }

    elementwise_violations: list[str] = []
    untyped_public_funcs: list[str] = []
    all_loops: list[str] = []

    for py_file in sorted(pkg_root.rglob("*.py")):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        rel_path = py_file.relative_to(project_root).as_posix()

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                if node.returns is None:
                    untyped_public_funcs.append(f"{rel_path}:{node.lineno}:{node.name}")

            if isinstance(node, (ast.For, ast.While)):
                iter_src = ast.unparse(node.iter) if isinstance(node, ast.For) else "while"
                entry = f"{rel_path}:{node.lineno} -> ({iter_src})"
                all_loops.append(entry)
                if py_file.parent in numerical_targets or py_file.name == "constants.py":
                    if "mole_fractions.items()" not in iter_src:
                        elementwise_violations.append(entry)

    # Locate pyrefly binary in active venv or PATH
    venv_pyrefly = Path(sys.executable).with_name(
        "pyrefly.exe" if sys.platform == "win32" else "pyrefly"
    )
    pyrefly_bin = str(venv_pyrefly) if venv_pyrefly.exists() else shutil.which("pyrefly")
    assert pyrefly_bin is not None, "pyrefly binary not found in environment."

    pyrefly_cmd: list[str] = [pyrefly_bin, "check"]
    if not (project_root / ".venv" / "Lib" / "site-packages").exists():
        purelib = sysconfig.get_path("purelib")
        platlib = sysconfig.get_path("platlib")
        if purelib:
            pyrefly_cmd.extend(["--site-package-path", purelib])
        if platlib and platlib != purelib:
            pyrefly_cmd.extend(["--site-package-path", platlib])

    pyrefly_proc = subprocess.run(
        pyrefly_cmd,
        cwd=str(project_root),
        capture_output=True,
        text=True,
        check=False,
    )
    combined_pyrefly_out = (pyrefly_proc.stdout + "\n" + pyrefly_proc.stderr).strip()

    print("\n" + "=" * 78)
    print("[DENETIM 6] Tum Proje AST Vektorizasyon ve Pyrefly Statik Tip Denetimi")
    print(f"  Sayisal Paketlerde Dizi Dongusu Ihlali (signal/thermo/spray/vv/constants): {len(elementwise_violations)}")
    print(f"  Donus Tipi Eksik Public Fonksiyon Sayisi                                 : {len(untyped_public_funcs)}")
    print(f"  Pyrefly Cikis Kodu                                                       : {pyrefly_proc.returncode}")
    for line in combined_pyrefly_out.splitlines():
        if "INFO" in line or "ERROR" in line:
            print(f"  Pyrefly -> {line}")
    print("=" * 78)

    assert len(elementwise_violations) == 0, f"Loop violations: {elementwise_violations}"
    assert len(untyped_public_funcs) == 0, f"Untyped public functions: {untyped_public_funcs}"
    assert pyrefly_proc.returncode == 0, f"pyrefly failed:\n{combined_pyrefly_out}"

