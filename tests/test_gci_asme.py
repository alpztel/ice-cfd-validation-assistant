"""Analytical verification tests for ASME V&V 20-2009 / Celik et al. (2008) GCI module."""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pytest

from ice_cfd_validator.exceptions import PhysicalBoundsError
from ice_cfd_validator.io.forte_parser import parse_dynamic_csv
from ice_cfd_validator.vv.gci import (
    compute_gci,
    compute_grid_size,
)


def test_exact_second_order_analytical_convergence() -> None:
    """Test 1: Exact 2nd-order polynomial phi(h) = 10.0 + 2.5*h^2 with r21 = r32 = 1.5."""
    h1, h2, h3 = 1.0, 1.5, 2.25
    phi1 = 10.0 + 2.5 * (h1**2)  # 12.5
    phi2 = 10.0 + 2.5 * (h2**2)  # 15.625
    phi3 = 10.0 + 2.5 * (h3**2)  # 22.65625

    res = compute_gci(phi1, phi2, phi3, h1=h1, h2=h2, h3=h3)

    print(
        f"\n[GCI Test 1] p = {res.apparent_order_p:.8f} | phi_ext^21 = {res.phi_ext_21:.8f} | "
        f"GCI_fine^21 = {res.gci_fine_21:.6f} | asym_coeff = {res.asymptotic_ratio_coeff:.8f} | "
        f"asym_rel = {res.asymptotic_ratio:.8f}"
    )

    assert res.r21 == pytest.approx(1.5, rel=1.0e-14)
    assert res.r32 == pytest.approx(1.5, rel=1.0e-14)
    assert res.convergence_regime == "MONOTONIC"
    assert res.apparent_order_p == pytest.approx(2.0, abs=1.0e-12)
    assert res.phi_ext_21 == pytest.approx(10.0, abs=1.0e-12)
    assert res.phi_ext_32 == pytest.approx(10.0, abs=1.0e-12)
    assert res.ea_21 == pytest.approx(0.25, abs=1.0e-14)
    assert res.e_ext_21 == pytest.approx(0.25, abs=1.0e-14)
    assert res.gci_fine_21 == pytest.approx(0.25, abs=1.0e-14)

    # Coefficient-scale asymptotic ratio is identically 1.000000; relative ratio scales as phi1/phi2
    assert res.asymptotic_ratio_coeff == pytest.approx(1.0, abs=1.0e-12)
    assert res.asymptotic_ratio * (phi2 / phi1) == pytest.approx(1.0, abs=1.0e-12)

    # In the small-h asymptotic limit (phi1 / phi2 -> 1), relative asymptotic_ratio -> 1.000 directly
    res_fine = compute_gci(
        10.0 + 2.5 * (0.01**2),
        10.0 + 2.5 * (0.015**2),
        10.0 + 2.5 * (0.0225**2),
        h1=0.01,
        h2=0.015,
        h3=0.0225,
    )
    assert res_fine.asymptotic_ratio == pytest.approx(1.0, abs=1.0e-4)


def test_nonuniform_refinement_fixed_point_convergence() -> None:
    """Test 2: Non-uniform ratios (r21=1.4, r32=1.7) resolve q(p) != 0 and converge to p = 2.0."""
    r21 = 1.4
    r32 = 1.7
    h1 = 1.0
    h2 = h1 * r21
    h3 = h2 * r32

    phi1 = 10.0 + 2.5 * (h1**2)
    phi2 = 10.0 + 2.5 * (h2**2)
    phi3 = 10.0 + 2.5 * (h3**2)

    res = compute_gci(phi1, phi2, phi3, r21=r21, r32=r32, tol=1.0e-12)

    expected_q = math.log((r21**2 - 1.0) / (r32**2 - 1.0))
    fixed_point_rhs = (
        abs(math.log(abs(res.epsilon32 / res.epsilon21)) + res.q_offset) / math.log(r21)
    )
    residual = abs(res.apparent_order_p - fixed_point_rhs)

    print(
        f"\n[GCI Test 2] r21={r21}, r32={r32} | q(p) = {res.q_offset:.8f} "
        f"(analitik={expected_q:.8f}) | p = {res.apparent_order_p:.12f} | "
        f"phi_ext^21 = {res.phi_ext_21:.12f} | |p - g(p)| = {residual:.3e}"
    )

    assert abs(res.q_offset) > 0.5
    assert res.q_offset == pytest.approx(expected_q, abs=1.0e-11)
    assert residual < 1.0e-11
    assert res.apparent_order_p == pytest.approx(2.0, abs=1.0e-11)
    assert res.phi_ext_21 == pytest.approx(10.0, abs=1.0e-11)
    assert res.asymptotic_ratio_coeff == pytest.approx(1.0, abs=1.0e-12)

    # Stiff non-uniform regime (r21=1.35, r32=2.5) where unrelaxed Picard |g'(p)| > 1 diverges
    h1_s, h2_s, h3_s = 1.0, 1.35, 1.35 * 2.5
    res_stiff = compute_gci(
        10.0 + 2.5 * (h1_s**2),
        10.0 + 2.5 * (h2_s**2),
        10.0 + 2.5 * (h3_s**2),
        h1=h1_s,
        h2=h2_s,
        h3=h3_s,
    )
    assert res_stiff.apparent_order_p == pytest.approx(2.0, abs=1.0e-11)
    assert res_stiff.phi_ext_21 == pytest.approx(10.0, abs=1.0e-11)


def test_amr_oscillatory_convergence_no_nan_or_warning() -> None:
    """Test 3: AMR oscillatory input (phi1=8.05, phi2=8.12, phi3=8.08) flags OSCILLATORY cleanly."""
    phi1, phi2, phi3 = 8.05, 8.12, 8.08

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        res_uniform = compute_gci(phi1, phi2, phi3, r21=1.5, r32=1.5)
        res_nonuniform = compute_gci(phi1, phi2, phi3, r21=1.4, r32=1.6)

    print(
        f"\n[GCI Test 3] eps21={res_uniform.epsilon21:+.4f}, eps32={res_uniform.epsilon32:+.4f}, "
        f"R={res_uniform.convergence_ratio_r:+.4f} | regime={res_uniform.convergence_regime} | "
        f"p_uni={res_uniform.apparent_order_p:.6f}, p_nonuni={res_nonuniform.apparent_order_p:.6f}, "
        f"GCI_fine^21={res_uniform.gci_fine_21:.6f}"
    )

    for res in (res_uniform, res_nonuniform):
        assert res.epsilon32 / res.epsilon21 < 0.0
        assert res.sign_s == -1.0
        assert res.convergence_regime == "OSCILLATORY"
        assert res.convergence_description == "Oscillatory Convergence"
        assert math.isfinite(res.apparent_order_p) and res.apparent_order_p > 0.0
        assert math.isfinite(res.phi_ext_21)
        assert math.isfinite(res.gci_fine_21) and res.gci_fine_21 > 0.0
        assert math.isfinite(res.asymptotic_ratio)

    expected_p_uniform = abs(math.log(abs(-0.04 / 0.07))) / math.log(1.5)
    assert res_uniform.apparent_order_p == pytest.approx(expected_p_uniform, abs=1.0e-12)


def test_compute_grid_size_3d_cells_volumes_and_asme_warning() -> None:
    """Verify 3D h_i from mean cell volumes and cell counts, plus ASME r < 1.3 UserWarning."""
    # Mean cell volumes corresponding to h = (1.0 mm, 1.5 mm, 2.25 mm)
    grid_vol = compute_grid_size(
        mean_cell_volumes_m3=(1.0e-9, 3.375e-9, 11.390625e-9),
    )
    assert grid_vol.is_dimensional_m is True
    assert grid_vol.meets_asme_threshold is True
    assert grid_vol.h1 == pytest.approx(1.0e-3, rel=1.0e-14)
    assert grid_vol.h2 == pytest.approx(1.5e-3, rel=1.0e-14)
    assert grid_vol.h3 == pytest.approx(2.25e-3, rel=1.0e-14)
    assert grid_vol.r21 == pytest.approx(1.5, rel=1.0e-14)
    assert grid_vol.r32 == pytest.approx(1.5, rel=1.0e-14)

    # Normalized cell counts: N_1 = 8e5, N_2 = 2.37037e5, N_3 = 1e5 -> h_1 = 1.0 < h_2 < h_3
    grid_cnt = compute_grid_size(cell_counts=(800_000.0, 250_000.0, 80_000.0))
    assert grid_cnt.is_dimensional_m is False
    assert grid_cnt.h1 == pytest.approx(1.0, rel=1.0e-14)
    assert grid_cnt.h2 == pytest.approx((800_000.0 / 250_000.0) ** (1.0 / 3.0), rel=1.0e-14)
    assert grid_cnt.h3 == pytest.approx((800_000.0 / 80_000.0) ** (1.0 / 3.0), rel=1.0e-14)
    assert grid_cnt.meets_asme_threshold is True

    # Sub-threshold refinement (r21 = 1.2 < 1.3) must emit ASME V&V 20-2009 UserWarning
    with pytest.warns(UserWarning, match="ASME V&V 20-2009"):
        sub_grid = compute_grid_size(mean_cell_volumes_m3=(1.0e-9, 1.728e-9, 4.096e-9))
    assert sub_grid.r21 == pytest.approx(1.2, rel=1.0e-14)
    assert sub_grid.meets_asme_threshold is False

    # Reversed ordering (Coarse -> Fine) must raise PhysicalBoundsError
    with pytest.raises(PhysicalBoundsError):
        compute_grid_size(cell_counts=(100_000.0, 300_000.0, 900_000.0))


def test_monotonic_divergence_and_forte_dynamic_csv_integration(tmp_path: Path) -> None:
    """Verify R > 1 flags DIVERGENT and integrate ForteDynamicData cell counts into GCI."""
    # Monotonic divergence: |eps21| > |eps32| with same sign -> R = (-3.0)/(-1.5) = 2.0 > 1
    div_res = compute_gci(15.0, 12.0, 10.5, r21=1.5, r32=1.5)
    assert div_res.convergence_ratio_r == pytest.approx(2.0, rel=1.0e-14)
    assert div_res.convergence_regime == "DIVERGENT"
    assert div_res.convergence_description == "Monotonic Divergence"

    # End-to-end integration with ANSYS Forte dynamic.csv AMR cell counts at peak combustion
    counts_by_mesh = [2_160_000, 640_000, 189_630]  # r21 = 1.5, r32 ~ 1.5
    peak_cells: list[float] = []
    for idx, n_peak in enumerate(counts_by_mesh):
        csv_file = tmp_path / f"dynamic_mesh_{idx + 1}.csv"
        csv_file.write_text(
            "Crank_Angle_(deg),Active_Cells_(-),Min_Cell_Size_(mm)\n"
            f"-10.0,{int(0.6 * n_peak)},0.50\n"
            f"5.0,{n_peak},0.25\n",
            encoding="utf-8",
        )
        dyn = parse_dynamic_csv(csv_file)
        peak_cells.append(float(np.max(dyn.cell_count)))

    grid = compute_grid_size(cell_counts=peak_cells, domain_volume_m3=5.0e-4)
    gci = compute_gci(11.82e6, 11.75e6, 11.58e6, grid=grid)
    assert grid.is_dimensional_m is True
    assert gci.convergence_regime == "MONOTONIC"
    assert gci.gci_fine_21 > 0.0 and gci.gci_fine_21 < 0.05

