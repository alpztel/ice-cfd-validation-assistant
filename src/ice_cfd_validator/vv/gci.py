"""ASME V&V 20-2009 / Roache (1998) / Celik et al. (2008) Grid Convergence Index (GCI).

Implements three-grid mesh independence verification, Adaptive Mesh Refinement (AMR)
oscillatory convergence detection, fixed-point apparent order-of-accuracy ($p$)
estimation for non-uniform grid refinement ratios ($r_{21} \\neq r_{32}$), Richardson
extrapolation, and numerical discretization uncertainty quantification for 3D ICE CFD.

References
----------
- Roache, P. J. (1998). *Verification and Validation in Computational Science and
  Engineering*. Hermosa Publishers.
- Celik, I. B., Ghia, U., Roache, P. J., Freitas, C. J., Coleman, H., & Raad, P. E.
  (2008). Procedure for Estimation and Reporting of Uncertainty Due to Discretization
  in CFD Applications. *ASME Journal of Fluids Engineering*, 130(7), 078001.
- ASME V&V 20-2009 (2009). *Standard for Verification and Validation in Computational
  Fluid Dynamics and Heat Transfer*. American Society of Mechanical Engineers.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq, fixed_point

from ice_cfd_validator.exceptions import PhysicalBoundsError

FloatArray: TypeAlias = NDArray[np.float64]
ConvergenceRegime: TypeAlias = Literal["MONOTONIC", "OSCILLATORY", "DIVERGENT"]

ASME_MIN_REFINEMENT_RATIO: Final[float] = 1.3
DEFAULT_THREE_GRID_FS: Final[float] = 1.25


@dataclass(frozen=True, slots=True)
class GridSizeResult:
    """Representative 3D grid spacings and refinement ratios for a three-mesh study.

    Grid indexing follows the ASME V&V 20-2009 convention:
    ``1`` = Fine, ``2`` = Medium, ``3`` = Coarse ($0 < h_1 < h_2 < h_3$).

    Attributes
    ----------
    h1 : float
        Representative spacing of the fine mesh (`m` if dimensional, `-` if normalized).
    h2 : float
        Representative spacing of the medium mesh (`m` if dimensional, `-` if normalized).
    h3 : float
        Representative spacing of the coarse mesh (`m` if dimensional, `-` if normalized).
    r21 : float
        Fine-to-medium grid refinement ratio $r_{21} = h_2 / h_1$ [-].
    r32 : float
        Medium-to-coarse grid refinement ratio $r_{32} = h_3 / h_2$ [-].
    is_dimensional_m : bool
        ``True`` if $h_i$ carries physical SI length units [m]; ``False`` if normalized
        dimensionless spacing $h_i = (N_1 / N_i)^{1/3}$ [-].
    meets_asme_threshold : bool
        ``True`` if both $r_{21} \\ge 1.3$ and $r_{32} \\ge 1.3$ per ASME V&V 20-2009.
    """

    h1: float
    h2: float
    h3: float
    r21: float
    r32: float
    is_dimensional_m: bool
    meets_asme_threshold: bool


@dataclass(frozen=True, slots=True)
class GCIResult:
    """Complete ASME V&V 20-2009 / Celik et al. (2008) three-grid GCI report.

    Attributes
    ----------
    h1 : float
        Fine-grid representative size $h_1$ (`m` or `-`).
    h2 : float
        Medium-grid representative size $h_2$ (`m` or `-`).
    h3 : float
        Coarse-grid representative size $h_3$ (`m` or `-`).
    r21 : float
        Fine-to-medium refinement ratio $r_{21} = h_2 / h_1$ [-].
    r32 : float
        Medium-to-coarse refinement ratio $r_{32} = h_3 / h_2$ [-].
    phi1 : float
        Fine-grid scalar quantity of interest $\\phi_1$.
    phi2 : float
        Medium-grid scalar quantity of interest $\\phi_2$.
    phi3 : float
        Coarse-grid scalar quantity of interest $\\phi_3$.
    epsilon21 : float
        Medium-minus-fine solution difference $\\epsilon_{21} = \\phi_2 - \\phi_1$.
    epsilon32 : float
        Coarse-minus-medium solution difference $\\epsilon_{32} = \\phi_3 - \\phi_2$.
    convergence_ratio_r : float
        Grid convergence ratio $R = \\epsilon_{21} / \\epsilon_{32}$ [-].
    sign_s : float
        Oscillation sign factor $s = \\text{sgn}(\\epsilon_{32} / \\epsilon_{21}) \\in \\{-1.0, +1.0\\}$.
    convergence_regime : ConvergenceRegime
        Classification flag: ``"MONOTONIC"`` ($0 < R < 1$), ``"OSCILLATORY"`` ($R < 0$),
        or ``"DIVERGENT"`` ($R \\ge 1$).
    convergence_description : str
        Human-readable regime label (``"Monotonic Convergence"``,
        ``"Oscillatory Convergence"``, or ``"Monotonic Divergence"``).
    apparent_order_p : float
        Observed / apparent order of accuracy $p$ [-].
    q_offset : float
        Non-uniform refinement correction $q(p) = \\ln((r_{21}^p - s) / (r_{32}^p - s))$ [-].
    phi_ext_21 : float
        Richardson extrapolated fine-grid value $\\phi_{\\text{ext}}^{21}$.
    phi_ext_32 : float
        Richardson extrapolated medium-grid value $\\phi_{\\text{ext}}^{32}$.
    ea_21 : float
        Approximate relative fine-medium error $e_a^{21} = |(\\phi_1 - \\phi_2) / \\phi_1|$ [-].
    ea_32 : float
        Approximate relative medium-coarse error $e_a^{32} = |(\\phi_2 - \\phi_3) / \\phi_2|$ [-].
    e_ext_21 : float
        Extrapolated relative error $e_{\\text{ext}}^{21} = |(\\phi_{\\text{ext}}^{21} - \\phi_1) / \\phi_{\\text{ext}}^{21}|$ [-].
    gci_fine_21 : float
        Fine-grid convergence index $\\text{GCI}_{\\text{fine}}^{21} = F_s e_a^{21} / (r_{21}^p - 1)$ [-].
    gci_32 : float
        Medium-grid convergence index $\\text{GCI}_{32} = F_s e_a^{32} / (r_{32}^p - 1)$ [-].
    asymptotic_ratio : float
        Roache asymptotic range indicator $\\text{GCI}_{32} / (r_{21}^p \\cdot \\text{GCI}_{\\text{fine}}^{21})$ [-].
    asymptotic_ratio_coeff : float
        Scale-invariant error-coefficient asymptotic ratio
        $[|\\epsilon_{32}| / (r_{32}^p - 1)] / [r_{21}^p |\\epsilon_{21}| / (r_{21}^p - 1)]$ [-],
        equal to ``1.0`` identically for exact power-law convergence $\\phi(h) = \\phi_0 + c h^p$.
    safety_factor_fs : float
        Factor of safety $F_s$ [-] (``1.25`` standard for three-grid studies).
    """

    h1: float
    h2: float
    h3: float
    r21: float
    r32: float
    phi1: float
    phi2: float
    phi3: float
    epsilon21: float
    epsilon32: float
    convergence_ratio_r: float
    sign_s: float
    convergence_regime: ConvergenceRegime
    convergence_description: str
    apparent_order_p: float
    q_offset: float
    phi_ext_21: float
    phi_ext_32: float
    ea_21: float
    ea_32: float
    e_ext_21: float
    gci_fine_21: float
    gci_32: float
    asymptotic_ratio: float
    asymptotic_ratio_coeff: float
    safety_factor_fs: float


def _warn_if_sub_asme_refinement(
    r21: float,
    r32: float,
    min_refinement_ratio: float,
) -> bool:
    """Check ASME V&V 20-2009 $r \\ge 1.3$ guideline and emit ``UserWarning`` if violated."""
    meets = (r21 >= min_refinement_ratio) and (r32 >= min_refinement_ratio)
    if not meets:
        warnings.warn(
            f"Grid refinement ratio below ASME V&V 20-2009 / Celik et al. (2008) recommended "
            f"minimum ({min_refinement_ratio:.2f}): r21={r21:.4f}, r32={r32:.4f}. "
            f"Discretization error separation from iterative/round-off noise may degrade.",
            UserWarning,
            stacklevel=3,
        )
    return meets


def compute_grid_size(
    *,
    mean_cell_volumes_m3: Sequence[float] | None = None,
    cell_counts: Sequence[float] | None = None,
    domain_volume_m3: float | None = None,
    min_refinement_ratio: float = ASME_MIN_REFINEMENT_RATIO,
    warn_below_threshold: bool = True,
) -> GridSizeResult:
    """Compute 3D representative grid sizes ($h_1 < h_2 < h_3$) and refinement ratios.

    Supports either direct mean cell volumes $h_i = (\\Delta V_{\\text{mean}, i})^{1/3}$ [m]
    or active cell counts $h_i = (N_1 / N_i)^{1/3}$ [-] (or $(V_{\\text{domain}} / N_i)^{1/3}$ [m]
    when ``domain_volume_m3`` is supplied).

    Parameters
    ----------
    mean_cell_volumes_m3 : Sequence[float] | None, optional
        Three-element sequence ``(dV_fine, dV_medium, dV_coarse)`` in [$\\text{m}^3$].
    cell_counts : Sequence[float] | None, optional
        Three-element sequence ``(N_fine, N_medium, N_coarse)`` of total 3D cell counts
        ($N_1 > N_2 > N_3 > 0$).
    domain_volume_m3 : float | None, optional
        Total 3D computational domain volume [$\\text{m}^3$] used when converting
        ``cell_counts`` into physical spacing $h_i = (V_{\\text{domain}} / N_i)^{1/3}$ [m].
    min_refinement_ratio : float, default=1.3
        Minimum recommended refinement ratio per ASME V&V 20-2009.
    warn_below_threshold : bool, default=True
        If ``True``, emit a ``UserWarning`` when $r_{21} < 1.3$ or $r_{32} < 1.3$.

    Returns
    -------
    GridSizeResult
        Validated representative mesh sizes and refinement ratios.
    """
    if (mean_cell_volumes_m3 is None) == (cell_counts is None):
        raise ValueError(
            "Provide exactly one of 'mean_cell_volumes_m3' or 'cell_counts' (length 3)."
        )
    if min_refinement_ratio <= 1.0 or not math.isfinite(min_refinement_ratio):
        raise PhysicalBoundsError(
            f"min_refinement_ratio must be finite and > 1.0; got {min_refinement_ratio}."
        )

    if mean_cell_volumes_m3 is not None:
        vols = np.asarray(mean_cell_volumes_m3, dtype=np.float64)
        if vols.shape != (3,):
            raise ValueError(
                f"mean_cell_volumes_m3 must contain exactly 3 values (fine, medium, coarse); "
                f"got shape {vols.shape}."
            )
        if not np.all(np.isfinite(vols)) or np.any(vols <= 0.0):
            raise PhysicalBoundsError(
                "All mean_cell_volumes_m3 values must be finite and strictly positive (> 0 m^3)."
            )
        h_arr = np.cbrt(vols)
        is_dimensional = True
    else:
        assert cell_counts is not None
        counts = np.asarray(cell_counts, dtype=np.float64)
        if counts.shape != (3,):
            raise ValueError(
                f"cell_counts must contain exactly 3 values (N1_fine, N2_medium, N3_coarse); "
                f"got shape {counts.shape}."
            )
        if not np.all(np.isfinite(counts)) or np.any(counts <= 0.0):
            raise PhysicalBoundsError(
                "All cell_counts values must be finite and strictly positive (> 0)."
            )
        if domain_volume_m3 is not None:
            if not math.isfinite(domain_volume_m3) or domain_volume_m3 <= 0.0:
                raise PhysicalBoundsError(
                    f"domain_volume_m3 must be finite and > 0 m^3; got {domain_volume_m3}."
                )
            h_arr = np.cbrt(domain_volume_m3 / counts)
            is_dimensional = True
        else:
            h_arr = np.cbrt(counts[0] / counts)
            is_dimensional = False

    h1, h2, h3 = float(h_arr[0]), float(h_arr[1]), float(h_arr[2])
    if not (h1 < h2 < h3):
        raise PhysicalBoundsError(
            f"Grids must be strictly ordered from Fine (1) to Medium (2) to Coarse (3) "
            f"such that h1 < h2 < h3; got h=({h1:.6e}, {h2:.6e}, {h3:.6e})."
        )

    r21 = h2 / h1
    r32 = h3 / h2

    if warn_below_threshold:
        meets = _warn_if_sub_asme_refinement(r21, r32, min_refinement_ratio)
    else:
        meets = (r21 >= min_refinement_ratio) and (r32 >= min_refinement_ratio)

    return GridSizeResult(
        h1=h1,
        h2=h2,
        h3=h3,
        r21=r21,
        r32=r32,
        is_dimensional_m=is_dimensional,
        meets_asme_threshold=meets,
    )


def _celik_q_vectorized(
    p_arr: FloatArray,
    ln_r21: float,
    ln_r32: float,
    sign_s: float,
) -> FloatArray:
    """Overflow- and cancellation-safe vectorized evaluation of Celik's $q(p)$ term.

    Computes $q(p) = \\ln((r_{21}^p - s) / (r_{32}^p - s))$ for $p \\ge 0$ without
    overflowing at large $p$ or dividing by zero as $p \\to 0^+$ when $s = +1$.
    """
    p_pos = np.maximum(p_arr, 0.0)
    u21 = p_pos * ln_r21
    u32 = p_pos * ln_r32

    if sign_s < 0.0:
        # s = -1: ln((r21^p + 1) / (r32^p + 1)) = logaddexp(p*ln(r21), 0) - logaddexp(p*ln(r32), 0)
        return np.asarray(np.logaddexp(u21, 0.0) - np.logaddexp(u32, 0.0), dtype=np.float64)

    # s = +1: ln((r21^p - 1) / (r32^p - 1)) with L'Hopital limit ln(ln(r21)/ln(r32)) as p -> 0+
    u21_safe = np.maximum(u21, 1.0e-15)
    u32_safe = np.maximum(u32, 1.0e-15)

    u21_small = np.minimum(u21_safe, 40.0)
    u32_small = np.minimum(u32_safe, 40.0)
    u21_large = np.maximum(u21_safe, 40.0)
    u32_large = np.maximum(u32_safe, 40.0)

    log_num = np.where(
        u21_safe > 40.0,
        u21_large + np.log1p(-np.exp(-u21_large)),
        np.log(np.expm1(u21_small)),
    )
    log_den = np.where(
        u32_safe > 40.0,
        u32_large + np.log1p(-np.exp(-u32_large)),
        np.log(np.expm1(u32_small)),
    )
    lhopital_limit = math.log(ln_r21 / ln_r32)
    return np.asarray(np.where(p_pos <= 1.0e-14, lhopital_limit, log_num - log_den), dtype=np.float64)


def _celik_rhs_vectorized(
    p_arr: FloatArray,
    ln_abs_eps_ratio: float,
    ln_r21: float,
    ln_r32: float,
    sign_s: float,
) -> FloatArray:
    """Vectorized right-hand side $g(p) = |\\ln|\\epsilon_{32}/\\epsilon_{21}| + q(p)| / \\ln(r_{21})$."""
    q_val = _celik_q_vectorized(p_arr, ln_r21, ln_r32, sign_s)
    return np.asarray(np.abs(ln_abs_eps_ratio + q_val) / ln_r21, dtype=np.float64)


def _solve_apparent_order_celik(
    abs_eps_ratio: float,
    r21: float,
    r32: float,
    sign_s: float,
    *,
    tol: float,
    max_iter: int,
) -> tuple[float, float]:
    """Solve Celik et al. (2008) implicit fixed-point equation for apparent order $p$.

    Returns ``(apparent_order_p, q_offset)`` with residual $|p - g(p)| \\le \\text{tol}$.
    """
    ln_r21 = math.log(r21)
    ln_r32 = math.log(r32)
    ln_abs_eps_ratio = math.log(abs_eps_ratio)

    # Exact closed-form solution when r21 == r32 (q(p) == 0 identically)
    if abs(r21 - r32) <= 1.0e-15 * max(r21, r32):
        p_closed = abs(ln_abs_eps_ratio) / ln_r21
        return p_closed, 0.0

    # Check whether the primary convergent branch ln|eps32/eps21| + q(p) = +p*ln(r21)
    # possesses its unique strictly-monotonic root on (0, +inf): H_+(0+) < 0.
    q_zero = float(_celik_q_vectorized(np.array([0.0], dtype=np.float64), ln_r21, ln_r32, sign_s)[0])
    has_positive_branch_root = (ln_abs_eps_ratio + q_zero) > 0.0

    # Better initial estimate using mean refinement ratio ln((r21 + r32)/2)
    p0 = max(abs(ln_abs_eps_ratio) / (0.5 * (ln_r21 + ln_r32)), 1.0e-4)

    def target_rhs_vec(p_vec: FloatArray) -> FloatArray:
        if has_positive_branch_root:
            # Rewrite p*ln(r21) = ln|eps32/eps21| + q(p) as the unconditionally
            # contractive equivalent fixed-point map in r32 when r32 > r21:
            # p*(ln_r32 if r32 > r21) -> solved directly via strictly monotonic H_+(p) = 0
            q_vec = _celik_q_vectorized(p_vec, ln_r21, ln_r32, sign_s)
            return np.maximum((ln_abs_eps_ratio + q_vec) / ln_r21, 0.0)
        return _celik_rhs_vectorized(p_vec, ln_abs_eps_ratio, ln_r21, ln_r32, sign_s)

    def relaxed_map(p_vec: FloatArray) -> FloatArray:
        return 0.5 * p_vec + 0.5 * target_rhs_vec(p_vec)

    def branch_residual(p_val: float) -> float:
        p_arr = np.array([p_val], dtype=np.float64)
        if has_positive_branch_root:
            q_val = float(_celik_q_vectorized(p_arr, ln_r21, ln_r32, sign_s)[0])
            return p_val - (ln_abs_eps_ratio + q_val) / ln_r21
        g_val = float(_celik_rhs_vectorized(p_arr, ln_abs_eps_ratio, ln_r21, ln_r32, sign_s)[0])
        return p_val - g_val

    def celik_full_residual(p_val: float) -> float:
        p_arr = np.array([p_val], dtype=np.float64)
        g_val = float(_celik_rhs_vectorized(p_arr, ln_abs_eps_ratio, ln_r21, ln_r32, sign_s)[0])
        return p_val - g_val

    p_sol: float | None = None
    try:
        fp_raw = fixed_point(
            relaxed_map,
            np.array([p0], dtype=np.float64),
            xtol=min(tol * 0.01, 1.0e-13),
            maxiter=max_iter,
            method="del2",
        )
        p_cand = float(np.asarray(fp_raw, dtype=np.float64).ravel()[0])
        if (
            math.isfinite(p_cand)
            and p_cand > 0.0
            and abs(branch_residual(p_cand)) <= tol
            and abs(celik_full_residual(p_cand)) <= tol
        ):
            p_sol = p_cand
    except RuntimeError:
        p_sol = None

    if p_sol is None:
        # Vectorized bracket scan on [0, p_upper] followed by guaranteed Brent root-finding
        p_upper = max(50.0, 10.0 * p0 + 10.0)
        p_scan = np.concatenate(
            (
                np.array([0.0], dtype=np.float64),
                np.geomspace(1.0e-8, p_upper, 1024, dtype=np.float64),
            )
        )
        if has_positive_branch_root:
            f_scan = p_scan - (
                ln_abs_eps_ratio + _celik_q_vectorized(p_scan, ln_r21, ln_r32, sign_s)
            ) / ln_r21
        else:
            f_scan = p_scan - _celik_rhs_vectorized(
                p_scan, ln_abs_eps_ratio, ln_r21, ln_r32, sign_s
            )
        sign_changes = np.flatnonzero(f_scan[:-1] * f_scan[1:] <= 0.0)
        if sign_changes.size == 0:
            raise PhysicalBoundsError(
                "Celik et al. (2008) apparent order p equation has no positive root for the "
                f"given refinement ratios (r21={r21:.4f}, r32={r32:.4f}) and solution "
                f"differences (|eps32/eps21|={abs_eps_ratio:.6e})."
            )
        idx = sign_changes[0]
        brent_out = brentq(
            branch_residual,
            float(p_scan[idx]),
            float(p_scan[idx + 1]),
            xtol=min(tol * 0.01, 1.0e-14),
            rtol=1.0e-14,
            maxiter=max_iter,
            full_output=False,
        )
        p_sol = float(brent_out[0] if isinstance(brent_out, tuple) else brent_out)

    q_final = float(
        _celik_q_vectorized(np.array([p_sol], dtype=np.float64), ln_r21, ln_r32, sign_s)[0]
    )
    return p_sol, q_final


def _safe_relative_error(numerator_diff: float, reference: float) -> float:
    """Compute $|\\Delta \\phi / \\phi_{\\text{ref}}|$ without raising on a zero reference."""
    if abs(numerator_diff) == 0.0:
        return 0.0
    if abs(reference) == 0.0:
        return float("nan")
    return abs(numerator_diff / reference)


def compute_gci(
    phi1: float,
    phi2: float,
    phi3: float,
    *,
    grid: GridSizeResult | None = None,
    h1: float | None = None,
    h2: float | None = None,
    h3: float | None = None,
    r21: float | None = None,
    r32: float | None = None,
    safety_factor_fs: float = DEFAULT_THREE_GRID_FS,
    tol: float = 1.0e-11,
    max_iter: int = 500,
    min_refinement_ratio: float = ASME_MIN_REFINEMENT_RATIO,
    warn_below_threshold: bool = True,
) -> GCIResult:
    """Evaluate ASME V&V 20-2009 / Celik et al. (2008) Grid Convergence Index (GCI).

    Grid spacing may be supplied via ``grid`` (:class:`GridSizeResult`), explicit
    spacings ``(h1, h2, h3)`` ($0 < h_1 < h_2 < h_3$), or refinement ratios
    ``(r21, r32)`` ($r_{21} > 1, r_{32} > 1$).

    Parameters
    ----------
    phi1, phi2, phi3 : float
        Scalar CFD quantities of interest on the Fine (``1``), Medium (``2``), and
        Coarse (``3``) meshes, respectively.
    grid : GridSizeResult | None, optional
        Pre-computed grid size object from :func:`compute_grid_size`.
    h1, h2, h3 : float | None, optional
        Representative mesh spacings satisfying $0 < h_1 < h_2 < h_3$.
    r21, r32 : float | None, optional
        Direct grid refinement ratios $r_{21} = h_2 / h_1 > 1$ and $r_{32} = h_3 / h_2 > 1$.
    safety_factor_fs : float, default=1.25
        Roache factor of safety $F_s$ (``1.25`` for three-grid studies).
    tol : float, default=1.0e-11
        Fixed-point residual convergence tolerance ($< 10^{-10}$ per specification).
    max_iter : int, default=500
        Maximum fixed-point / root-finder iterations.
    min_refinement_ratio : float, default=1.3
        ASME V&V 20-2009 recommended minimum grid refinement ratio.
    warn_below_threshold : bool, default=True
        Emit ``UserWarning`` if $r_{21} < 1.3$ or $r_{32} < 1.3$.

    Returns
    -------
    GCIResult
        Immutable verification report including convergence regime, apparent order $p$,
        Richardson extrapolation $\\phi_{\\text{ext}}^{21}$, fine-grid $\\text{GCI}_{\\text{fine}}^{21}$,
        and asymptotic range ratios.
    """
    if not (math.isfinite(phi1) and math.isfinite(phi2) and math.isfinite(phi3)):
        raise PhysicalBoundsError(
            f"Grid solutions (phi1, phi2, phi3) must be finite numbers; "
            f"got ({phi1}, {phi2}, {phi3})."
        )
    if not math.isfinite(safety_factor_fs) or safety_factor_fs <= 0.0:
        raise PhysicalBoundsError(
            f"safety_factor_fs must be finite and > 0; got {safety_factor_fs}."
        )
    if not math.isfinite(tol) or tol <= 0.0 or tol >= 1.0e-6:
        raise ValueError(f"tol must satisfy 0 < tol < 1e-6; got {tol}.")

    # Resolve (h1, h2, h3) and (r21, r32) from exactly one grid specification mode
    has_grid_obj = grid is not None
    has_h_tuple = (h1 is not None) or (h2 is not None) or (h3 is not None)
    has_r_tuple = (r21 is not None) or (r32 is not None)

    if (int(has_grid_obj) + int(has_h_tuple) + int(has_r_tuple)) != 1:
        raise ValueError(
            "Provide exactly one grid specification: 'grid', '(h1, h2, h3)', or '(r21, r32)'."
        )

    if grid is not None:
        h1_val, h2_val, h3_val = grid.h1, grid.h2, grid.h3
        r21_val, r32_val = grid.r21, grid.r32
        if warn_below_threshold and not grid.meets_asme_threshold:
            _warn_if_sub_asme_refinement(r21_val, r32_val, min_refinement_ratio)
    elif has_h_tuple:
        if h1 is None or h2 is None or h3 is None:
            raise ValueError("All three of (h1, h2, h3) must be provided together.")
        if not (math.isfinite(h1) and math.isfinite(h2) and math.isfinite(h3)) or h1 <= 0.0:
            raise PhysicalBoundsError(
                f"Grid spacings (h1, h2, h3) must be finite and > 0; got ({h1}, {h2}, {h3})."
            )
        if not (h1 < h2 < h3):
            raise PhysicalBoundsError(
                f"Grid spacings must satisfy strict Fine < Medium < Coarse ordering "
                f"(h1 < h2 < h3); got ({h1}, {h2}, {h3})."
            )
        h1_val, h2_val, h3_val = float(h1), float(h2), float(h3)
        r21_val = h2_val / h1_val
        r32_val = h3_val / h2_val
        if warn_below_threshold:
            _warn_if_sub_asme_refinement(r21_val, r32_val, min_refinement_ratio)
    else:
        if r21 is None or r32 is None:
            raise ValueError("Both r21 and r32 must be provided together.")
        if not (math.isfinite(r21) and math.isfinite(r32)) or r21 <= 1.0 or r32 <= 1.0:
            raise PhysicalBoundsError(
                f"Refinement ratios r21 and r32 must be finite and > 1.0; got r21={r21}, r32={r32}."
            )
        r21_val = float(r21)
        r32_val = float(r32)
        h1_val = 1.0
        h2_val = r21_val
        h3_val = r21_val * r32_val
        if warn_below_threshold:
            _warn_if_sub_asme_refinement(r21_val, r32_val, min_refinement_ratio)

    phi1_f, phi2_f, phi3_f = float(phi1), float(phi2), float(phi3)
    eps21 = phi2_f - phi1_f
    eps32 = phi3_f - phi2_f

    # Degenerate zero-difference checks
    if abs(eps21) == 0.0 and abs(eps32) == 0.0:
        return GCIResult(
            h1=h1_val,
            h2=h2_val,
            h3=h3_val,
            r21=r21_val,
            r32=r32_val,
            phi1=phi1_f,
            phi2=phi2_f,
            phi3=phi3_f,
            epsilon21=0.0,
            epsilon32=0.0,
            convergence_ratio_r=0.0,
            sign_s=1.0,
            convergence_regime="MONOTONIC",
            convergence_description="Monotonic Convergence",
            apparent_order_p=float("inf"),
            q_offset=0.0,
            phi_ext_21=phi1_f,
            phi_ext_32=phi2_f,
            ea_21=0.0,
            ea_32=0.0,
            e_ext_21=0.0,
            gci_fine_21=0.0,
            gci_32=0.0,
            asymptotic_ratio=1.0,
            asymptotic_ratio_coeff=1.0,
            safety_factor_fs=float(safety_factor_fs),
        )

    if abs(eps21) == 0.0 or abs(eps32) == 0.0:
        raise PhysicalBoundsError(
            f"Degenerate grid triplet with one zero difference (epsilon21={eps21}, "
            f"epsilon32={eps32}); apparent order p via Celik et al. (2008) is undefined."
        )

    # Convergence ratio R = eps21 / eps32 and AMR oscillation sign s = sgn(eps32 / eps21)
    conv_ratio_r = eps21 / eps32
    eps_ratio_32_21 = eps32 / eps21
    sign_s = 1.0 if eps_ratio_32_21 > 0.0 else -1.0

    if conv_ratio_r < 0.0:
        regime: ConvergenceRegime = "OSCILLATORY"
        regime_desc = "Oscillatory Convergence"
    elif conv_ratio_r < 1.0:
        regime = "MONOTONIC"
        regime_desc = "Monotonic Convergence"
    else:
        regime = "DIVERGENT"
        regime_desc = "Monotonic Divergence"

    p_order, q_val = _solve_apparent_order_celik(
        abs(eps_ratio_32_21),
        r21_val,
        r32_val,
        sign_s,
        tol=tol,
        max_iter=max_iter,
    )

    if p_order <= 1.0e-15:
        raise PhysicalBoundsError(
            "Apparent order of accuracy p collapsed to 0 (|epsilon32| == |epsilon21| with "
            "uniform refinement); Richardson extrapolation denominator (r21^p - 1) is singular."
        )

    r21_pow_p = r21_val**p_order
    r32_pow_p = r32_val**p_order
    denom_21 = r21_pow_p - 1.0
    denom_32 = r32_pow_p - 1.0

    phi_ext_21 = (r21_pow_p * phi1_f - phi2_f) / denom_21
    phi_ext_32 = (r32_pow_p * phi2_f - phi3_f) / denom_32

    ea_21 = _safe_relative_error(phi1_f - phi2_f, phi1_f)
    ea_32 = _safe_relative_error(phi2_f - phi3_f, phi2_f)
    e_ext_21 = _safe_relative_error(phi_ext_21 - phi1_f, phi_ext_21)

    fs = float(safety_factor_fs)
    gci_fine_21 = (fs * ea_21) / denom_21
    gci_32 = (fs * ea_32) / denom_32

    if gci_fine_21 > 0.0 and math.isfinite(gci_fine_21) and math.isfinite(gci_32):
        asymptotic_ratio = gci_32 / (r21_pow_p * gci_fine_21)
    else:
        asymptotic_ratio = float("nan")

    dim_err_21 = abs(eps21) / denom_21
    dim_err_32 = abs(eps32) / denom_32
    asymptotic_ratio_coeff = dim_err_32 / (r21_pow_p * dim_err_21)

    return GCIResult(
        h1=h1_val,
        h2=h2_val,
        h3=h3_val,
        r21=r21_val,
        r32=r32_val,
        phi1=phi1_f,
        phi2=phi2_f,
        phi3=phi3_f,
        epsilon21=eps21,
        epsilon32=eps32,
        convergence_ratio_r=conv_ratio_r,
        sign_s=sign_s,
        convergence_regime=regime,
        convergence_description=regime_desc,
        apparent_order_p=p_order,
        q_offset=q_val,
        phi_ext_21=phi_ext_21,
        phi_ext_32=phi_ext_32,
        ea_21=ea_21,
        ea_32=ea_32,
        e_ext_21=e_ext_21,
        gci_fine_21=gci_fine_21,
        gci_32=gci_32,
        asymptotic_ratio=asymptotic_ratio,
        asymptotic_ratio_coeff=asymptotic_ratio_coeff,
        safety_factor_fs=fs,
    )
