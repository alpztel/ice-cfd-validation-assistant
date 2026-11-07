"""SAE / IEEE publication-grade diagnostic figures (300 DPI, STIX LaTeX mathtext).

Provides headless-safe Matplotlib visualizers for 0D/1D ICE thermodynamic cycle
diagnostics, log-P vs. log-(V/V_max) polytropic index verification, high-altitude
Aero-Diesel spray tip penetration & wall-impingement risk, and ASME V&V 20-2009
three-grid Richardson extrapolation / GCI uncertainty reporting.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, TypeAlias

import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from numpy.typing import NDArray

from ice_cfd_validator.thermo.heat_release import BurnFractionMetrics, HeatReleaseResult
from ice_cfd_validator.vv.gci import GCIResult

# Ensure headless compatibility when pyplot is not used interactively
matplotlib.rcParams["svg.fonttype"] = "none"

FloatArray: TypeAlias = NDArray[np.float64]

_SAE_IEEE_RCPARAMS: Final[dict[str, Any]] = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 10.0,
    "axes.labelsize": 10.5,
    "axes.titlesize": 11.0,
    "axes.linewidth": 0.9,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 4.2,
    "ytick.major.size": 4.2,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.fontsize": 8.6,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "#CCCCCC",
    "figure.dpi": 300,
    "savefig.dpi": 300,
}

_PALETTE: Final[tuple[str, ...]] = (
    "#004488",  # Deep Navy (Sea Level / Primary)
    "#DDAA33",  # Ochre Gold (Mid Altitude)
    "#BB5566",  # Crimson Rose (High Altitude)
    "#117733",  # Forest Green
    "#882255",  # Wine Purple
)


@dataclass(frozen=True, slots=True)
class ThermoOverlayTrace:
    """Optional secondary thermodynamic cycle trace for multi-altitude overlays.

    Attributes
    ----------
    label : str
        Legend label (e.g., ``"15,000 ft (4,572 m)"``).
    theta_deg : NDArray[np.float64]
        Crank angle array $\\theta$ [deg].
    pressure_pa : NDArray[np.float64]
        In-cylinder pressure $P(\\theta)$ [Pa].
    hrr_result : HeatReleaseResult
        Apparent net/gross heat release rate object.
    burn_metrics : BurnFractionMetrics
        Extracted MFB profile and CA10/CA50/CA90 angles.
    """

    label: str
    theta_deg: FloatArray
    pressure_pa: FloatArray
    hrr_result: HeatReleaseResult
    burn_metrics: BurnFractionMetrics


@dataclass(frozen=True, slots=True)
class AltitudeSprayPlotCase:
    """Container for a single altitude/density spray comparison curve set.

    Attributes
    ----------
    label : str
        Scenario label (e.g., ``"Sea Level (0 ft)"``).
    rho_gas_kg_m3 : float
        In-cylinder ambient gas density $\\rho_g$ [$\\text{kg/m}^3$].
    time_s : NDArray[np.float64]
        Continuous time array $t$ [s] for analytical correlations.
    hiroyasu_penetration_m : NDArray[np.float64]
        Hiroyasu & Arai (1990) spray tip penetration $S_{\\text{HA}}(t)$ [m].
    siebers_penetration_m : NDArray[np.float64]
        Naber & Siebers (1996) / Siebers (1999) penetration $S_{\\text{NS}}(t)$ [m].
    cfd_time_s : NDArray[np.float64] | None, optional
        Discrete ANSYS Forte CFD sample timestamps $t_{\\text{CFD}}$ [s].
    cfd_penetration_m : NDArray[np.float64] | None, optional
        Discrete ANSYS Forte CFD spray tip penetration $S_{\\text{CFD}}$ [m].
    impingement_time_s : float | None, optional
        Detected wall-impingement onset time $t_{\\text{imp}}$ [s], if reached.
    """

    label: str
    rho_gas_kg_m3: float
    time_s: FloatArray
    hiroyasu_penetration_m: FloatArray
    siebers_penetration_m: FloatArray
    cfd_time_s: FloatArray | None = None
    cfd_penetration_m: FloatArray | None = None
    impingement_time_s: float | None = None


def _finalize_and_save(
    fig: Figure,
    save_path: str | Path | None,
    dpi: int,
) -> Figure:
    """Attach an Agg canvas and optionally export high-resolution PNG/PDF output."""
    FigureCanvasAgg(fig)
    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    return fig


def plot_thermo_diagnostics(
    theta_deg: FloatArray,
    pressure_pa: FloatArray,
    hrr_result: HeatReleaseResult,
    burn_metrics: BurnFractionMetrics,
    *,
    primary_label: str = "Sea Level (0 ft)",
    overlay_traces: Sequence[ThermoOverlayTrace] | None = None,
    theta_limits_deg: tuple[float, float] = (-40.0, 80.0),
    title: str | None = None,
    save_path: str | Path | None = None,
    dpi: int = 300,
) -> Figure:
    """Generate a 3-panel SAE/IEEE thermodynamic diagnostic chart ($P$, AHRR, MFB).

    Parameters
    ----------
    theta_deg : NDArray[np.float64]
        Crank angle array $\\theta$ [deg ATDC].
    pressure_pa : NDArray[np.float64]
        In-cylinder pressure $P(\\theta)$ [Pa].
    hrr_result : HeatReleaseResult
        First-Law apparent net and gross heat release rate result.
    burn_metrics : BurnFractionMetrics
        Cumulative Mass Fraction Burned (MFB) and CA10/CA50/CA90 timings.
    primary_label : str, default="Sea Level (0 ft)"
        Legend label for the primary cycle trace.
    overlay_traces : Sequence[ThermoOverlayTrace] | None, optional
        Additional altitude/operating-point traces to overlay for comparison.
    theta_limits_deg : tuple[float, float], default=(-40.0, 80.0)
        Horizontal crank-angle zoom window [deg ATDC].
    title : str | None, optional
        Optional figure title above the top panel.
    save_path : str | Path | None, optional
        Output PNG/PDF file path (saved at ``dpi`` when provided).
    dpi : int, default=300
        Export resolution in dots per inch.

    Returns
    -------
    matplotlib.figure.Figure
        Rendered 3-subplot diagnostic figure.
    """
    with matplotlib.rc_context(_SAE_IEEE_RCPARAMS):
        fig = Figure(figsize=(7.6, 8.4), dpi=dpi, constrained_layout=True)
        ax_p = fig.add_subplot(3, 1, 1)
        ax_q = fig.add_subplot(3, 1, 2, sharex=ax_p)
        ax_mfb = fig.add_subplot(3, 1, 3, sharex=ax_p)

        # Panel (a): Cylinder Pressure [MPa]
        p_mpa = np.asarray(pressure_pa, dtype=np.float64) * 1.0e-6
        ax_p.plot(
            theta_deg,
            p_mpa,
            color=_PALETTE[0],
            lw=2.0,
            label=f"{primary_label} ($P_{{\\max}}={ float(np.max(p_mpa)):.2f}$ MPa)",
        )

        # Panel (b): Net & Gross Heat Release Rate [J/deg]
        ax_q.plot(
            hrr_result.crank_angle_deg,
            hrr_result.dq_net_j_per_deg,
            color=_PALETTE[0],
            lw=1.9,
            label=f"{primary_label} — Net $dQ_n/d\\theta$",
        )
        if np.any(np.abs(hrr_result.dq_ht_j_per_deg) > 0.0):
            ax_q.plot(
                hrr_result.crank_angle_deg,
                hrr_result.dq_gross_j_per_deg,
                color=_PALETTE[0],
                lw=1.35,
                ls="--",
                alpha=0.85,
                label=f"{primary_label} — Gross $dQ_{{ch}}/d\\theta$",
            )
            ax_q.fill_between(
                hrr_result.crank_angle_deg,
                hrr_result.dq_net_j_per_deg,
                hrr_result.dq_gross_j_per_deg,
                color=_PALETTE[0],
                alpha=0.10,
            )

        # Panel (c): Mass Fraction Burned [%] + CA10 / CA50 / CA90 vertical markers
        ax_mfb.plot(
            burn_metrics.crank_angle_deg,
            burn_metrics.mfb * 100.0,
            color=_PALETTE[0],
            lw=2.0,
            label=(
                f"{primary_label} "
                f"(CA50$={burn_metrics.ca50_deg:.1f}^\\circ$)"
            ),
        )

        # Optional multi-altitude overlays
        if overlay_traces is not None:
            for idx, tr in enumerate(overlay_traces):
                col = _PALETTE[(idx + 1) % len(_PALETTE)]
                ls_trace = "-." if idx % 2 == 0 else ":"
                tr_p_mpa = np.asarray(tr.pressure_pa, dtype=np.float64) * 1.0e-6
                ax_p.plot(
                    tr.theta_deg,
                    tr_p_mpa,
                    color=col,
                    lw=1.8,
                    ls=ls_trace,
                    label=f"{tr.label} ($P_{{\\max}}={float(np.max(tr_p_mpa)):.2f}$ MPa)",
                )
                ax_q.plot(
                    tr.hrr_result.crank_angle_deg,
                    tr.hrr_result.dq_net_j_per_deg,
                    color=col,
                    lw=1.7,
                    ls=ls_trace,
                    label=f"{tr.label} — Net $dQ_n/d\\theta$",
                )
                ax_mfb.plot(
                    tr.burn_metrics.crank_angle_deg,
                    tr.burn_metrics.mfb * 100.0,
                    color=col,
                    lw=1.8,
                    ls=ls_trace,
                    label=f"{tr.label} (CA50$={tr.burn_metrics.ca50_deg:.1f}^\\circ$)",
                )

        # CA10, CA50, CA90 markers for primary reference trace
        ca_markers = (
            ("CA10", burn_metrics.ca10_deg, 10.0),
            ("CA50", burn_metrics.ca50_deg, 50.0),
            ("CA90", burn_metrics.ca90_deg, 90.0),
        )
        for tag, ca_val, pct_val in ca_markers:
            ax_mfb.axvline(ca_val, color="#444444", lw=0.95, ls="--", alpha=0.75)
            ax_mfb.axhline(pct_val, color="#888888", lw=0.65, ls=":", alpha=0.55)
            ax_mfb.scatter([ca_val], [pct_val], color=_PALETTE[0], s=26, zorder=5)
            ax_mfb.annotate(
                f"{tag}\n${ca_val:.1f}^\\circ$",
                xy=(ca_val, pct_val),
                xytext=(5.0, -14.0 if tag == "CA90" else 6.0),
                textcoords="offset points",
                fontsize=8.2,
                color="#222222",
                bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "#CCCCCC", "alpha": 0.88},
            )

        for ax in (ax_p, ax_q, ax_mfb):
            ax.axvline(0.0, color="#666666", lw=0.8, ls=":", alpha=0.7)
            ax.grid(True, which="major", ls="--", lw=0.55, alpha=0.30)
            ax.set_xlim(theta_limits_deg)

        ax_q.axhline(0.0, color="#555555", lw=0.75, alpha=0.7)

        ax_p.set_ylabel(r"Cylinder Pressure, $P$ [MPa]")
        ax_q.set_ylabel(r"Heat Release Rate, $dQ/d\theta$ [J/deg]")
        ax_mfb.set_ylabel(r"Mass Fraction Burned, $\mathrm{MFB}$ [%]")
        ax_mfb.set_xlabel(r"Crank Angle, $\theta$ [deg ATDC]")
        ax_mfb.set_ylim(-4.0, 108.0)

        if title is not None:
            ax_p.set_title(title, pad=8.0, fontweight="bold")
        else:
            ax_p.set_title(
                r"0D/1D First-Law Thermodynamic Cycle Diagnostics ($P$, AHRR, MFB)",
                pad=8.0,
            )

        ax_p.legend(loc="upper right")
        ax_q.legend(loc="upper right")
        ax_mfb.legend(loc="lower right")

        ax_p.tick_params(labelbottom=False)
        ax_q.tick_params(labelbottom=False)

        return _finalize_and_save(fig, save_path, dpi)


def plot_pv_log_diagram(
    volume_m3: FloatArray,
    pressure_pa: FloatArray,
    theta_deg: FloatArray,
    *,
    comp_window_deg: tuple[float, float] = (-135.0, -25.0),
    exp_window_deg: tuple[float, float] = (35.0, 135.0),
    primary_label: str = "Sea Level (0 ft)",
    overlay_loops: Sequence[tuple[str, FloatArray, FloatArray]] | None = None,
    title: str | None = None,
    save_path: str | Path | None = None,
    dpi: int = 300,
) -> Figure:
    """Plot $\\log_{10}(P) - \\log_{10}(V / V_{\\max})$ diagram with polytropic slope fits.

    Extracts compression ($\\kappa_{\\text{comp}}$) and expansion ($\\kappa_{\\text{exp}}$)
    polytropic exponents via linear least-squares regression of $\\ln P$ against
    $\\ln(V / V_{\\max})$ over the user-specified crank-angle windows.

    Parameters
    ----------
    volume_m3 : NDArray[np.float64]
        Instantaneous cylinder volume $V(\\theta)$ [$\\text{m}^3$].
    pressure_pa : NDArray[np.float64]
        In-cylinder pressure $P(\\theta)$ [Pa].
    theta_deg : NDArray[np.float64]
        Crank angle array $\\theta$ [deg ATDC].
    comp_window_deg : tuple[float, float], default=(-135.0, -25.0)
        Compression polytropic regression interval [deg ATDC].
    exp_window_deg : tuple[float, float], default=(35.0, 135.0)
        Expansion polytropic regression interval [deg ATDC].
    primary_label : str, default="Sea Level (0 ft)"
        Label for the primary indicator loop.
    overlay_loops : Sequence[tuple[str, NDArray[np.float64], NDArray[np.float64]]] | None, optional
        Optional ``(label, volume_m3, pressure_pa)`` tuples for multi-altitude comparison.
    title : str | None, optional
        Custom plot title.
    save_path : str | Path | None, optional
        Output PNG/PDF path.
    dpi : int, default=300
        Export resolution in dots per inch.

    Returns
    -------
    matplotlib.figure.Figure
        Rendered $\\log P - \\log(V/V_{\\max})$ figure.
    """
    v_arr = np.asarray(volume_m3, dtype=np.float64)
    p_mpa = np.asarray(pressure_pa, dtype=np.float64) * 1.0e-6
    th_arr = np.asarray(theta_deg, dtype=np.float64)

    v_max = float(np.max(v_arr))
    v_norm = v_arr / v_max
    log_v = np.log10(v_norm)
    log_p = np.log10(p_mpa)

    comp_mask = (th_arr >= comp_window_deg[0]) & (th_arr <= comp_window_deg[1])
    exp_mask = (th_arr >= exp_window_deg[0]) & (th_arr <= exp_window_deg[1])

    comp_slope, comp_intercept = np.polyfit(log_v[comp_mask], log_p[comp_mask], 1)
    exp_slope, exp_intercept = np.polyfit(log_v[exp_mask], log_p[exp_mask], 1)
    kappa_comp = float(-comp_slope)
    kappa_exp = float(-exp_slope)

    with matplotlib.rc_context(_SAE_IEEE_RCPARAMS):
        fig = Figure(figsize=(7.0, 5.4), dpi=dpi, constrained_layout=True)
        ax = fig.add_subplot(1, 1, 1)

        ax.plot(
            log_v,
            log_p,
            color=_PALETTE[0],
            lw=2.0,
            label=primary_label,
        )

        if overlay_loops is not None:
            for idx, (lbl, v_ov, p_ov) in enumerate(overlay_loops):
                col = _PALETTE[(idx + 1) % len(_PALETTE)]
                ls_ov = "-." if idx % 2 == 0 else ":"
                v_ov_norm = np.asarray(v_ov, dtype=np.float64) / float(np.max(v_ov))
                p_ov_mpa = np.asarray(p_ov, dtype=np.float64) * 1.0e-6
                ax.plot(
                    np.log10(v_ov_norm),
                    np.log10(p_ov_mpa),
                    color=col,
                    lw=1.7,
                    ls=ls_ov,
                    label=lbl,
                )

        # Plot fitted polytropic tangent lines (offset slightly only in linewidth/style)
        lv_comp = np.linspace(float(np.min(log_v[comp_mask])), float(np.max(log_v[comp_mask])), 64)
        lv_exp = np.linspace(float(np.min(log_v[exp_mask])), float(np.max(log_v[exp_mask])), 64)

        ax.plot(
            lv_comp,
            comp_slope * lv_comp + comp_intercept,
            color="#117733",
            lw=2.1,
            ls="--",
            label=rf"Compression Fit: $\kappa_{{\mathrm{{comp}}}} = {kappa_comp:.3f}$",
        )
        ax.plot(
            lv_exp,
            exp_slope * lv_exp + exp_intercept,
            color="#882255",
            lw=2.1,
            ls="--",
            label=rf"Expansion Fit: $\kappa_{{\mathrm{{exp}}}} = {kappa_exp:.3f}$",
        )

        # Highlight TDC point (theta closest to 0 deg)
        tdc_idx = int(np.argmin(np.abs(th_arr)))
        ax.scatter(
            [float(log_v[tdc_idx])],
            [float(log_p[tdc_idx])],
            color="#BB5566",
            s=36,
            zorder=6,
            label=r"TDC ($\theta = 0^\circ$)",
        )

        ax.grid(True, which="major", ls="--", lw=0.55, alpha=0.32)
        ax.set_xlabel(r"Normalized Log Volume, $\log_{10}(V / V_{\max})$ [-]")
        ax.set_ylabel(r"Log Cylinder Pressure, $\log_{10}(P / 1\ \mathrm{MPa})$ [-]")
        ax.set_title(
            title
            if title is not None
            else r"Logarithmic $P-V$ Diagram & Polytropic Exponent Verification",
            pad=8.0,
        )
        ax.legend(loc="lower left")

        return _finalize_and_save(fig, save_path, dpi)


def plot_altitude_spray_comparison(
    cases: Sequence[AltitudeSprayPlotCase],
    *,
    wall_distance_m: float,
    title: str | None = None,
    save_path: str | Path | None = None,
    dpi: int = 300,
) -> Figure:
    """Plot Aero-Diesel spray penetration vs. time across altitudes with wall limit $L_{\\text{wall}}$.

    Parameters
    ----------
    cases : Sequence[AltitudeSprayPlotCase]
        Ordered altitude scenarios containing Hiroyasu-Arai (1990), Siebers (1999),
        and optional ANSYS Forte CFD spray penetration curves.
    wall_distance_m : float
        Piston bowl / cylinder liner radial clearance $L_{\\text{wall}}$ [m].
    title : str | None, optional
        Custom figure title.
    save_path : str | Path | None, optional
        Output PNG/PDF file path.
    dpi : int, default=300
        Export resolution in dots per inch.

    Returns
    -------
    matplotlib.figure.Figure
        Rendered spray penetration comparison figure.
    """
    wall_mm = wall_distance_m * 1.0e3
    markers: tuple[str, ...] = ("o", "s", "^", "D", "v")

    with matplotlib.rc_context(_SAE_IEEE_RCPARAMS):
        fig = Figure(figsize=(7.6, 5.4), dpi=dpi, constrained_layout=True)
        ax = fig.add_subplot(1, 1, 1)

        max_s_mm = wall_mm * 1.12
        max_t_ms = 0.0

        for idx, case in enumerate(cases):
            col = _PALETTE[idx % len(_PALETTE)]
            mk = markers[idx % len(markers)]
            t_ms = np.asarray(case.time_s, dtype=np.float64) * 1.0e3
            s_ha_mm = np.asarray(case.hiroyasu_penetration_m, dtype=np.float64) * 1.0e3
            s_ns_mm = np.asarray(case.siebers_penetration_m, dtype=np.float64) * 1.0e3

            max_t_ms = max(max_t_ms, float(np.max(t_ms)))
            max_s_mm = max(max_s_mm, float(np.max(s_ha_mm)), float(np.max(s_ns_mm)))

            ax.plot(
                t_ms,
                s_ha_mm,
                color=col,
                lw=2.0,
                ls="-",
                label=(
                    rf"{case.label} ($\rho_g={case.rho_gas_kg_m3:.1f}\ \mathrm{{kg/m^3}}$)"
                    r" — Hiroyasu-Arai"
                ),
            )
            ax.plot(
                t_ms,
                s_ns_mm,
                color=col,
                lw=1.45,
                ls="--",
                alpha=0.82,
                label=f"{case.label} — Siebers (1999)",
            )

            if case.cfd_time_s is not None and case.cfd_penetration_m is not None:
                cfd_t_ms = np.asarray(case.cfd_time_s, dtype=np.float64) * 1.0e3
                cfd_s_mm = np.asarray(case.cfd_penetration_m, dtype=np.float64) * 1.0e3
                max_s_mm = max(max_s_mm, float(np.max(cfd_s_mm)))
                ax.scatter(
                    cfd_t_ms,
                    cfd_s_mm,
                    color=col,
                    marker=mk,
                    s=28,
                    facecolors="white",
                    linewidths=1.35,
                    zorder=5,
                    label=f"{case.label} — ANSYS Forte CFD",
                )

            if case.impingement_time_s is not None:
                t_imp_ms = case.impingement_time_s * 1.0e3
                ax.scatter([t_imp_ms], [wall_mm], color=col, marker="X", s=64, zorder=7)
                ax.annotate(
                    rf"$t_{{\mathrm{{imp}}}}={t_imp_ms:.2f}$ ms",
                    xy=(t_imp_ms, wall_mm),
                    xytext=(-18.0, 9.0 + 11.0 * (idx % 2)),
                    textcoords="offset points",
                    fontsize=8.2,
                    color=col,
                    fontweight="bold",
                    bbox={
                        "boxstyle": "round,pad=0.18",
                        "fc": "white",
                        "ec": col,
                        "alpha": 0.90,
                    },
                )

        # Piston bowl / cylinder wall boundary line and hazard shading
        y_top = max_s_mm * 1.08
        ax.axhline(
            wall_mm,
            color="#AA1122",
            lw=1.75,
            ls="-.",
            zorder=4,
            label=rf"Piston Bowl Wall Limit, $L_{{\mathrm{{wall}}}} = {wall_mm:.1f}$ mm",
        )
        ax.axhspan(wall_mm, y_top, color="#BB5566", alpha=0.09, zorder=1)

        ax.set_xlim(0.0, max_t_ms * 1.02)
        ax.set_ylim(0.0, y_top)
        ax.grid(True, which="major", ls="--", lw=0.55, alpha=0.30)
        ax.set_xlabel(r"Time After Start of Injection, $t$ [ms]")
        ax.set_ylabel(r"Spray Tip Penetration, $S(t)$ [mm]")
        ax.set_title(
            title
            if title is not None
            else r"High-Altitude Aero-Diesel Spray Scaling & Wall Impingement Assessment",
            pad=8.0,
        )
        ax.legend(loc="lower right", ncol=1, fontsize=7.9)

        return _finalize_and_save(fig, save_path, dpi)


def plot_gci_convergence(
    gci_result: GCIResult,
    *,
    metric_name: str = r"Peak Cylinder Pressure, $P_{\max}$",
    metric_unit: str = "MPa",
    scale_factor: float = 1.0,
    title: str | None = None,
    save_path: str | Path | None = None,
    dpi: int = 300,
) -> Figure:
    """Plot ASME V&V 20-2009 three-grid convergence, $\\phi_{\\text{ext}}^{21}$, and GCI bar.

    Parameters
    ----------
    gci_result : GCIResult
        Output of :func:`ice_cfd_validator.vv.gci.compute_gci`.
    metric_name : str, default="Peak Cylinder Pressure, $P_{\\max}$"
        Quantity-of-interest axis label.
    metric_unit : str, default="MPa"
        Display engineering unit string.
    scale_factor : float, default=1.0
        Optional display multiplier (e.g. ``1e-6`` to display ``Pa`` in ``MPa``).
    title : str | None, optional
        Custom figure title.
    save_path : str | Path | None, optional
        Output PNG/PDF file path.
    dpi : int, default=300
        Export resolution in dots per inch.

    Returns
    -------
    matplotlib.figure.Figure
        Rendered ASME V&V 20-2009 grid convergence figure.
    """
    h_norm = np.array(
        [1.0, gci_result.r21, gci_result.r21 * gci_result.r32],
        dtype=np.float64,
    )
    phi_vals = (
        np.array([gci_result.phi1, gci_result.phi2, gci_result.phi3], dtype=np.float64)
        * scale_factor
    )
    phi_ext = gci_result.phi_ext_21 * scale_factor
    u_fine = gci_result.gci_fine_21 * abs(phi_vals[0])
    u_med = gci_result.gci_32 * abs(phi_vals[1])

    h_curve = np.linspace(0.0, float(h_norm[2]) * 1.04, 256, dtype=np.float64)
    p_vis = min(max(gci_result.apparent_order_p, 0.25), 8.0)
    phi_curve = phi_ext + (phi_vals[0] - phi_ext) * (h_curve**p_vis)

    unit_suffix = f" [{metric_unit}]" if metric_unit else ""

    with matplotlib.rc_context(_SAE_IEEE_RCPARAMS):
        fig = Figure(figsize=(7.0, 5.1), dpi=dpi, constrained_layout=True)
        ax = fig.add_subplot(1, 1, 1)

        # Continuous order-p Richardson fit curve
        ax.plot(
            h_curve,
            phi_curve,
            color="#555555",
            lw=1.45,
            ls="--",
            label=rf"Order-$p$ Richardson Fit ($p = {gci_result.apparent_order_p:.2f}$)",
        )

        # Extrapolated continuum asymptote & marker at h/h_1 -> 0
        ax.axhline(phi_ext, color="#117733", lw=1.1, ls=":", alpha=0.85)
        ax.scatter(
            [0.0],
            [phi_ext],
            color="#117733",
            marker="D",
            s=58,
            zorder=6,
            label=(
                rf"Richardson Continuum $\phi_{{\mathrm{{ext}}}}^{{21}} = {phi_ext:.4f}$"
                f" {metric_unit}".rstrip()
            ),
        )

        # Fine, Medium, Coarse discrete mesh solutions
        ax.plot(
            h_norm,
            phi_vals,
            color=_PALETTE[0],
            lw=1.2,
            alpha=0.55,
        )
        mesh_tags = (
            r"Fine Mesh ($h_1 / h_1 = 1.00$)",
            rf"Medium Mesh ($h_2 / h_1 = {h_norm[1]:.2f}$)",
            rf"Coarse Mesh ($h_3 / h_1 = {h_norm[2]:.2f}$)",
        )
        mesh_markers = ("o", "s", "^")
        for i in (0, 1, 2):
            ax.scatter(
                [float(h_norm[i])],
                [float(phi_vals[i])],
                color=_PALETTE[0],
                marker=mesh_markers[i],
                s=52,
                zorder=6,
                label=f"{mesh_tags[i]}: {phi_vals[i]:.4f} {metric_unit}".rstrip(),
            )

        # GCI_fine^21 discretization uncertainty error bar at h_1 / h_1 = 1.0
        ax.errorbar(
            [1.0],
            [float(phi_vals[0])],
            yerr=[u_fine],
            fmt="none",
            ecolor="#BB5566",
            elinewidth=2.1,
            capsize=6.0,
            capthick=1.8,
            zorder=7,
            label=(
                rf"$\pm\mathrm{{GCI}}_{{\mathrm{{fine}}}}^{{21}} = "
                rf"\pm {gci_result.gci_fine_21 * 100.0:.2f}\%$ "
                rf"($\pm {u_fine:.4f}$ {metric_unit})".rstrip()
            ),
        )
        ax.errorbar(
            [float(h_norm[1])],
            [float(phi_vals[1])],
            yerr=[u_med],
            fmt="none",
            ecolor="#DDAA33",
            elinewidth=1.5,
            capsize=4.5,
            capthick=1.4,
            zorder=5,
            alpha=0.85,
        )

        is_descending = phi_ext >= float(phi_vals[-1])
        box_x = 0.97 if is_descending else 0.03
        box_ha = "right" if is_descending else "left"
        legend_loc = "lower left" if is_descending else "lower right"

        summary_txt = (
            f"ASME V&V 20-2009 Summary\n"
            f"Regime: {gci_result.convergence_regime}\n"
            f"$r_{{21}} = {gci_result.r21:.3f},\\ r_{{32}} = {gci_result.r32:.3f}$\n"
            f"Apparent Order $p = {gci_result.apparent_order_p:.3f}$\n"
            f"$\\mathrm{{GCI}}_{{\\mathrm{{fine}}}}^{{21}} = {gci_result.gci_fine_21 * 100.0:.2f}\\%$\n"
            f"Asymptotic Ratio $= {gci_result.asymptotic_ratio:.3f}$"
        )
        ax.text(
            box_x,
            0.96,
            summary_txt,
            transform=ax.transAxes,
            va="top",
            ha=box_ha,
            fontsize=8.4,
            bbox={
                "boxstyle": "round,pad=0.35",
                "fc": "#FAFAFA",
                "ec": "#BBBBBB",
                "alpha": 0.93,
            },
        )

        ax.set_xlim(-0.12, float(h_norm[2]) * 1.08)
        ax.margins(y=0.10)
        ax.grid(True, which="major", ls="--", lw=0.55, alpha=0.30)
        ax.set_xlabel(r"Normalized Representative Grid Spacing, $h_i / h_1$ [-]")
        ax.set_ylabel(f"{metric_name}{unit_suffix}")
        ax.set_title(
            title
            if title is not None
            else r"ASME V&V 20-2009 Grid Convergence & Discretization Uncertainty",
            pad=8.0,
        )
        ax.legend(loc=legend_loc, fontsize=7.9)

        return _finalize_and_save(fig, save_path, dpi)
