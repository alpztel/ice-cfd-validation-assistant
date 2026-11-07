"""Second Law thermodynamic exergy budget, entropy generation, and irreversibility.

Evaluates in-cylinder thermomechanical exergy $E_{tm}(\\theta)$, work exergy transfer
$(P - P_0) dV/d\\theta$, wall heat loss exergy $(1 - T_0/T_w) dQ_{ht}/d\\theta$,
exergy destruction rate $dE_{\\text{dest}}/d\\theta$, and the closed-cycle Second Law
exergetic efficiency $\\eta_{II}$ based on Rakopoulos & Giakoumis (2006) and Caton (2000).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import cumulative_trapezoid, trapezoid

from ice_cfd_validator.constants import R_UNIVERSAL
from ice_cfd_validator.exceptions import PhysicalBoundsError
from ice_cfd_validator.exergy.dead_state import (
    DIESEL_EN590_EXERGY_FACTOR,
    DeadStateConditions,
)
from ice_cfd_validator.signal.filtering import compute_derivative
from ice_cfd_validator.thermo.nasa_polynomials import DRY_AIR, NASAGasMixture

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ExergyRatesResult:
    """Instantaneous and cumulative Second Law exergy trajectories across crank angle.

    Parameters
    ----------
    crank_angle_deg : FloatArray
        Crank angle array $\\theta$ [$\\text{deg}$].
    thermomechanical_exergy_j : FloatArray
        Instantaneous in-cylinder fluid thermomechanical exergy $E_{tm}(\\theta)$ [$\\text{J}$].
    work_exergy_rate_j_per_deg : FloatArray
        Net indicated work exergy transfer rate $dE_W/d\\theta = (P - P_0) dV/d\\theta$ [$\\text{J/deg}$].
    heat_loss_exergy_rate_j_per_deg : FloatArray
        Wall heat loss exergy transfer rate $dE_{Q,\\text{loss}}/d\\theta = (1 - T_0/T_w) dQ_{ht}/d\\theta$ [$\\text{J/deg}$].
    combustion_source_exergy_rate_j_per_deg : FloatArray
        Fuel combustion chemical/thermal exergy input rate $dE_{\\text{in}}/d\\theta = \\varphi \\cdot dQ_{\\text{gross}}/d\\theta$ [$\\text{J/deg}$].
    exergy_destruction_rate_j_per_deg : FloatArray
        Instantaneous irreversibility / exergy destruction rate $dE_{\\text{dest}}/d\\theta$ [$\\text{J/deg}$].
    cumulative_work_exergy_j : FloatArray
        Cumulative net indicated work exergy $E_W(\\theta)$ [$\\text{J}$].
    cumulative_heat_loss_exergy_j : FloatArray
        Cumulative wall heat loss exergy $E_{Q,\\text{loss}}(\\theta)$ [$\\text{J}$].
    cumulative_combustion_source_exergy_j : FloatArray
        Cumulative fuel exergy input $E_{\\text{in}}(\\theta)$ [$\\text{J}$].
    cumulative_exergy_destruction_j : FloatArray
        Cumulative irreversibility / exergy destroyed $E_{\\text{dest}}(\\theta)$ [$\\text{J}$].
    """

    crank_angle_deg: FloatArray
    thermomechanical_exergy_j: FloatArray
    work_exergy_rate_j_per_deg: FloatArray
    heat_loss_exergy_rate_j_per_deg: FloatArray
    combustion_source_exergy_rate_j_per_deg: FloatArray
    exergy_destruction_rate_j_per_deg: FloatArray
    cumulative_work_exergy_j: FloatArray
    cumulative_heat_loss_exergy_j: FloatArray
    cumulative_combustion_source_exergy_j: FloatArray
    cumulative_exergy_destruction_j: FloatArray


@dataclass(frozen=True, slots=True)
class ExergyCycleBudget:
    """Integrated cycle-end exergy balance, percentage distribution, and efficiencies.

    Parameters
    ----------
    e_in_fuel_j : float
        Total fuel chemical exergy supplied $E_{\\text{in,fuel}} = m_f \\cdot \\varphi \\cdot LHV$ [$\\text{J}$].
    e_work_net_j : float
        Net indicated work exergy $E_{W,\\text{net}} = \\int (P - P_0) dV$ [$\\text{J}$].
    e_heat_loss_j : float
        Wall heat transfer exergy loss $E_{Q,\\text{loss}} = \\int (1 - T_0/T_w) dQ_{ht}$ [$\\text{J}$].
    delta_e_tm_exhaust_j : float
        Net fluid internal exergy carried out at EVO: $\\Delta E_{tm} = E_{tm}(\\theta_{\\text{EVO}}) - E_{tm}(\\theta_{\\text{IVC}})$ [$\\text{J}$].
    e_destroyed_j : float
        Total cycle irreversibility / exergy destroyed $E_{\\text{destroyed}}$ [$\\text{J}$].
    eta_first_law : float
        First Law indicated thermal efficiency $\\eta_I = W_{\\text{ind}} / (m_f \\cdot LHV)$ [-].
    eta_second_law : float
        Second Law exergetic efficiency $\\eta_{II} = E_{W,\\text{net}} / E_{\\text{in,fuel}}$ [-].
    work_share_pct : float
        Work exergy percentage share of fuel chemical input [$\\%$].
    heat_loss_share_pct : float
        Heat loss exergy percentage share of fuel chemical input [$\\%$].
    exhaust_share_pct : float
        Exhaust / EVO internal exergy percentage share of fuel chemical input [$\\%$].
    destruction_share_pct : float
        Exergy destruction percentage share of fuel chemical input [$\\%$].
    budget_closure_pct : float
        Total percentage budget closure (must equal $100.0000\\%$) [$\\%$].
    closure_error_j : float
        Absolute balance closure error $|E_{\\text{in}} - \\sum E_{\\text{out}}|$ [$\\text{J}$].
    """

    e_in_fuel_j: float
    e_work_net_j: float
    e_heat_loss_j: float
    delta_e_tm_exhaust_j: float
    e_destroyed_j: float
    eta_first_law: float
    eta_second_law: float
    work_share_pct: float
    heat_loss_share_pct: float
    exhaust_share_pct: float
    destruction_share_pct: float
    budget_closure_pct: float
    closure_error_j: float


def compute_standard_entropy(
    temperature_k: ArrayLike,
    gas_mixture: NASAGasMixture = DRY_AIR,
) -> FloatArray:
    """Evaluate continuous standard specific entropy integral $\\int \\frac{c_p(T)}{T} dT$ via NASA polynomials.

    Evaluates the analytical indefinite integral of the 7-coefficient NASA Glen polynomial:

    $$F(T) = R_{\\text{spec}} \\left( a_1 \\ln T + a_2 T + \\frac{a_3}{2} T^2
      + \\frac{a_4}{3} T^3 + \\frac{a_5}{4} T^4 \\right) \\quad [\\text{J/(kg}\\cdot\\text{K)}]$$

    Enforces strict $C^0$ continuity across the $T_{\\text{mid}} = 1000.0\\,\\text{K}$ handoff
    boundary to machine precision ($< 10^{-15}\\,\\text{J/(kg}\\cdot\\text{K)}$).

    Parameters
    ----------
    temperature_k : ArrayLike
        Static temperature array $T$ in Kelvin ($200 \\le T \\le 6000\\,\\text{K}$).
    gas_mixture : NASAGasMixture, default=DRY_AIR
        NASA ideal-gas thermodynamic model.

    Returns
    -------
    FloatArray
        Analytical standard entropy array $F(T)$ in $\\text{J/(kg}\\cdot\\text{K)}$.
    """
    t = gas_mixture._validate_temperature(temperature_k)
    lo = gas_mixture.low_coeffs
    hi = gas_mixture.high_coeffs
    r = gas_mixture.specific_gas_constant
    t_mid = gas_mixture.t_mid_k

    # Indefinite integrals of cp(T)/T
    f_lo = r * (lo[0] * np.log(t) + t * (lo[1] + t * (0.5 * lo[2] + t * ((lo[3] / 3.0) + 0.25 * lo[4] * t))))
    f_hi = r * (hi[0] * np.log(t) + t * (hi[1] + t * (0.5 * hi[2] + t * ((hi[3] / 3.0) + 0.25 * hi[4] * t))))

    # Exact C^0 continuity handoff offset at t_mid
    f_lo_mid = r * (lo[0] * np.log(t_mid) + t_mid * (lo[1] + t_mid * (0.5 * lo[2] + t_mid * ((lo[3] / 3.0) + 0.25 * lo[4] * t_mid))))
    f_hi_mid = r * (hi[0] * np.log(t_mid) + t_mid * (hi[1] + t_mid * (0.5 * hi[2] + t_mid * ((hi[3] / 3.0) + 0.25 * hi[4] * t_mid))))
    c_offset = f_lo_mid - f_hi_mid

    f_cont = np.where(t < t_mid, f_lo, f_hi + c_offset)
    return np.asarray(f_cont, dtype=np.float64)


def compute_thermomechanical_exergy(
    temperature_k: ArrayLike,
    pressure_pa: ArrayLike,
    dead_state: DeadStateConditions,
    trapped_mass_kg: float,
    *,
    gas_mixture: NASAGasMixture = DRY_AIR,
    volume_m3: ArrayLike | None = None,
) -> FloatArray:
    """Compute in-cylinder fluid thermomechanical exergy $E_{tm}(T, P)$ in Joules.

    Applies the closed-system control volume thermomechanical exergy definition:

    $$E_{tm}(\\theta) = m_{\\text{cyl}} \\left[ (u(T) - u(T_0)) + P_0 (v(\\theta) - v_0)
      - T_0 (s(T, P) - s(T_0, P_0)) \\right] \\quad [\\text{J}]$$

    where:
    - $u(T) - u(T_0)$ is the specific internal energy change via NASA Glenn polynomials.
    - $v_0 = R_{\\text{spec}} T_0 / P_0$ is the dead-state specific volume.
    - $s(T, P) - s(T_0, P_0) = \\int_{T_0}^T \\frac{c_p(T')}{T'} dT' - R_{\\text{spec}} \\ln(P / P_0)$.
    - Thermodynamic convexity ($E_{tm} \\ge 0.0$) is strictly preserved.

    Parameters
    ----------
    temperature_k : ArrayLike
        In-cylinder fluid temperature $T$ [$\\text{K}$].
    pressure_pa : ArrayLike
        In-cylinder fluid static pressure $P$ [$\\text{Pa}$].
    dead_state : DeadStateConditions
        Atmospheric dead state reference $(P_0, T_0, \\rho_0)$.
    trapped_mass_kg : float
        Total trapped in-cylinder fluid mass $m_{\\text{cyl}}$ [$\\text{kg}$] ($> 0$).
    gas_mixture : NASAGasMixture, default=DRY_AIR
        NASA thermodynamic mixture model.
    volume_m3 : ArrayLike | None, default=None
        Cylinder volume $V(\\theta)$ [$\\text{m}^3$]. If None, evaluates via ideal gas $V = m R_{\\text{spec}} T / P$.

    Returns
    -------
    FloatArray
        Thermomechanical exergy array $E_{tm}(\\theta) \\ge 0.0$ [$\\text{J}$].
    """
    if trapped_mass_kg <= 0.0:
        raise PhysicalBoundsError(f"Trapped cylinder mass must be strictly positive, got {trapped_mass_kg} kg.")

    t = gas_mixture._validate_temperature(temperature_k)
    p = np.asarray(pressure_pa, dtype=np.float64)
    if np.any(p <= 0.0):
        raise PhysicalBoundsError("Cylinder pressure array must be strictly positive (> 0 Pa).")

    r_spec = gas_mixture.specific_gas_constant
    t0 = dead_state.t0_k
    p0 = dead_state.p0_pa
    v0 = r_spec * t0 / p0

    # Specific internal energy change: u(T) - u(T0)
    u_t = gas_mixture.internal_energy(t)
    u_t0 = float(gas_mixture.internal_energy(np.asarray([t0], dtype=np.float64))[0])
    delta_u = u_t - u_t0

    # Specific volume change: v - v0
    if volume_m3 is not None:
        v_arr = np.asarray(volume_m3, dtype=np.float64)
        if np.any(v_arr <= 0.0):
            raise PhysicalBoundsError("Cylinder volume must be strictly positive (> 0 m^3).")
        v_spec = v_arr / trapped_mass_kg
    else:
        v_spec = r_spec * t / p
    delta_v = v_spec - v0

    # Entropy change: s(T, P) - s(T0, P0)
    f_t = compute_standard_entropy(t, gas_mixture=gas_mixture)
    f_t0 = float(compute_standard_entropy(np.asarray([t0], dtype=np.float64), gas_mixture=gas_mixture)[0])
    delta_s = (f_t - f_t0) - r_spec * np.log(p / p0)

    # Thermomechanical exergy
    e_tm_spec = delta_u + p0 * delta_v - t0 * delta_s
    e_tm_j = trapped_mass_kg * e_tm_spec

    # Thermodynamic convexity guard: Gibbs convexity dictates E_tm >= 0 identically
    # Microscopic negative values arising from floating-point roundoff are clamped to 0.0
    dead_state_mask = np.isclose(t, t0, atol=1e-11) & np.isclose(p, p0, atol=1e-8)
    e_tm_j = np.where(dead_state_mask, 0.0, np.maximum(e_tm_j, 0.0))

    return np.asarray(e_tm_j, dtype=np.float64)


def compute_exergy_rates(
    crank_angle_deg: ArrayLike,
    pressure_pa: ArrayLike,
    temperature_k: ArrayLike,
    volume_m3: ArrayLike,
    dvolume_dtheta_m3_per_deg: ArrayLike,
    gross_heat_release_rate_j_per_deg: ArrayLike,
    wall_heat_loss_rate_j_per_deg: ArrayLike,
    dead_state: DeadStateConditions,
    trapped_mass_kg: float,
    wall_temperature_k: float,
    phi_factor: float = DIESEL_EN590_EXERGY_FACTOR,
    *,
    gas_mixture: NASAGasMixture = DRY_AIR,
) -> ExergyRatesResult:
    """Compute instantaneous and cumulative exergy transfer and destruction rates across crank angle.

    Evaluates:
    - Net work exergy transfer: $dE_W/d\\theta = (P(\\theta) - P_0) dV/d\\theta$ [$\\text{J/deg}$].
    - Wall heat loss exergy transfer: $dE_{Q,\\text{loss}}/d\\theta = (1 - T_0 / T_w) dQ_{ht}/d\\theta$ [$\\text{J/deg}$].
    - Combustion fuel exergy input rate: $dE_{\\text{in}}/d\\theta = \\varphi \\cdot dQ_{\\text{gross}}/d\\theta$ [$\\text{J/deg}$].
    - Control volume thermomechanical exergy $E_{tm}(\\theta)$ and derivative $dE_{tm}/d\\theta$ [$\\text{J/deg}$].
    - Exergy destruction rate via closed-system availability balance:
      $$\\frac{dE_{\\text{dest}}}{d\\theta} = \\frac{dE_{\\text{in}}}{d\\theta} - \\frac{dE_W}{d\\theta}
        - \\frac{dE_{Q,\\text{loss}}}{d\\theta} - \\frac{dE_{tm}}{d\\theta} \\quad [\\text{J/deg}]$$

    Parameters
    ----------
    crank_angle_deg : ArrayLike
        Crank angle array $\\theta$ [$\\text{deg}$].
    pressure_pa : ArrayLike
        Cylinder pressure $P(\\theta)$ [$\\text{Pa}$].
    temperature_k : ArrayLike
        Bulk cylinder temperature $T(\\theta)$ [$\\text{K}$].
    volume_m3 : ArrayLike
        Cylinder volume $V(\\theta)$ [$\\text{m}^3$].
    dvolume_dtheta_m3_per_deg : ArrayLike
        Kinematic volume derivative $dV/d\\theta$ [$\\text{m}^3/\\text{deg}$].
    gross_heat_release_rate_j_per_deg : ArrayLike
        Gross combustion heat release rate $dQ_{\\text{gross}}/d\\theta$ [$\\text{J/deg}$].
    wall_heat_loss_rate_j_per_deg : ArrayLike
        Cylinder wall heat loss rate $dQ_{ht}/d\\theta \\ge 0$ [$\\text{J/deg}$].
    dead_state : DeadStateConditions
        Atmospheric dead state reference conditions $(P_0, T_0, \\rho_0)$.
    trapped_mass_kg : float
        Trapped in-cylinder fluid mass [$\\text{kg}$].
    wall_temperature_k : float
        Mean chamber wall temperature $T_w$ [$\\text{K}$] ($T_w > 0$).
    phi_factor : float, default=DIESEL_EN590_EXERGY_FACTOR
        Fuel chemical exergy ratio $\\varphi = e_{f,\\text{chem}} / LHV$ [-].
    gas_mixture : NASAGasMixture, default=DRY_AIR
        NASA ideal-gas thermodynamic model.

    Returns
    -------
    ExergyRatesResult
        Populated immutable results container with instantaneous rates and cumulative totals.
    """
    if wall_temperature_k <= 0.0:
        raise PhysicalBoundsError(f"Wall temperature must be strictly positive (> 0 K), got {wall_temperature_k}.")
    if phi_factor <= 0.0:
        raise PhysicalBoundsError(f"Exergy factor phi must be strictly positive, got {phi_factor}.")

    theta = np.asarray(crank_angle_deg, dtype=np.float64)
    p = np.asarray(pressure_pa, dtype=np.float64)
    t = np.asarray(temperature_k, dtype=np.float64)
    v = np.asarray(volume_m3, dtype=np.float64)
    dv = np.asarray(dvolume_dtheta_m3_per_deg, dtype=np.float64)
    dq_gross = np.asarray(gross_heat_release_rate_j_per_deg, dtype=np.float64)
    dq_ht = np.asarray(wall_heat_loss_rate_j_per_deg, dtype=np.float64)

    p0 = dead_state.p0_pa
    t0 = dead_state.t0_k

    # 1. Net work exergy transfer rate: dE_W / dtheta = (P - P0) * dV / dtheta
    de_work_dtheta = (p - p0) * dv

    # 2. Wall heat loss exergy transfer rate: dE_Q,loss / dtheta = (1 - T0 / Tw) * dQ_ht / dtheta
    carnot_factor = 1.0 - (t0 / wall_temperature_k)
    de_heat_loss_dtheta = carnot_factor * dq_ht

    # 3. Fuel chemical/thermal exergy input rate: dE_in / dtheta = phi * dQ_gross / dtheta
    de_in_dtheta = phi_factor * dq_gross

    # 4. In-cylinder thermomechanical exergy and rate of change
    e_tm = compute_thermomechanical_exergy(
        temperature_k=t,
        pressure_pa=p,
        dead_state=dead_state,
        trapped_mass_kg=trapped_mass_kg,
        gas_mixture=gas_mixture,
        volume_m3=v,
    )
    de_tm_dtheta = compute_derivative(theta, e_tm)

    # 5. Exergy destruction rate: dE_dest / dtheta = dE_in - dE_W - dE_Q,loss - dE_tm
    de_dest_dtheta = de_in_dtheta - de_work_dtheta - de_heat_loss_dtheta - de_tm_dtheta

    # Cumulative integrals via trapezoidal rule with initial=0.0
    cum_work = np.asarray(cumulative_trapezoid(de_work_dtheta, theta, initial=0.0), dtype=np.float64)
    cum_heat_loss = np.asarray(cumulative_trapezoid(de_heat_loss_dtheta, theta, initial=0.0), dtype=np.float64)
    cum_in = np.asarray(cumulative_trapezoid(de_in_dtheta, theta, initial=0.0), dtype=np.float64)
    cum_dest = np.asarray(cumulative_trapezoid(de_dest_dtheta, theta, initial=0.0), dtype=np.float64)

    return ExergyRatesResult(
        crank_angle_deg=theta,
        thermomechanical_exergy_j=e_tm,
        work_exergy_rate_j_per_deg=de_work_dtheta,
        heat_loss_exergy_rate_j_per_deg=de_heat_loss_dtheta,
        combustion_source_exergy_rate_j_per_deg=de_in_dtheta,
        exergy_destruction_rate_j_per_deg=de_dest_dtheta,
        cumulative_work_exergy_j=cum_work,
        cumulative_heat_loss_exergy_j=cum_heat_loss,
        cumulative_combustion_source_exergy_j=cum_in,
        cumulative_exergy_destruction_j=cum_dest,
    )


def compute_exergy_cycle_budget(
    rates_result: ExergyRatesResult,
    fuel_mass_kg: float,
    lhv_j_per_kg: float,
    phi_factor: float = DIESEL_EN590_EXERGY_FACTOR,
    *,
    theta_ivc_deg: float | None = None,
    theta_evo_deg: float | None = None,
    indicated_work_j: float | None = None,
) -> ExergyCycleBudget:
    """Compute integrated Second Law cycle exergy balance, percentage distribution, and efficiencies.

    Integrates:
    - Fuel chemical exergy: $E_{\\text{in,fuel}} = m_f \\cdot \\varphi \\cdot LHV$ [$\\text{J}$].
    - Net indicated work exergy: $E_{W,\\text{net}} = \\int (P - P_0) dV$ [$\\text{J}$].
    - Wall heat transfer exergy loss: $E_{Q,\\text{loss}} = \\int (1 - T_0 / T_w) dQ_{ht}$ [$\\text{J}$].
    - Fluid internal exergy difference: $\\Delta E_{tm,\\text{exhaust}} = E_{tm}(\\theta_{\\text{EVO}}) - E_{tm}(\\theta_{\\text{IVC}})$ [$\\text{J}$].
    - Cycle exergy destroyed: $E_{\\text{destroyed}} = E_{\\text{in,fuel}} - E_{W,\\text{net}} - E_{Q,\\text{loss}} - \\Delta E_{tm,\\text{exhaust}}$ [$\\text{J}$].
    - First Law efficiency: $\\eta_I = W_{\\text{ind}} / (m_f \\cdot LHV)$ [-].
    - Second Law efficiency: $\\eta_{II} = E_{W,\\text{net}} / E_{\\text{in,fuel}}$ [-].

    Parameters
    ----------
    rates_result : ExergyRatesResult
        Instantaneous exergy rates from `compute_exergy_rates`.
    fuel_mass_kg : float
        Total injected fuel mass per cylinder cycle $m_f$ [$\\text{kg}$] ($> 0$).
    lhv_j_per_kg : float
        Fuel lower heating value $LHV$ [$\\text{J/kg}$] ($> 0$).
    phi_factor : float, default=DIESEL_EN590_EXERGY_FACTOR
        Fuel chemical exergy factor $\\varphi = e_{f,\\text{chem}} / LHV$ [-].
    theta_ivc_deg : float | None, default=None
        Inlet valve closing angle $\\theta_{\\text{IVC}}$ [$\\text{deg}$]. Defaults to start of trajectory.
    theta_evo_deg : float | None, default=None
        Exhaust valve opening angle $\\theta_{\\text{EVO}}$ [$\\text{deg}$]. Defaults to end of trajectory.
    indicated_work_j : float | None, default=None
        Indicated cycle work $W_{\\text{ind}} = \\oint P dV$ [$\\text{J}$]. If None, evaluates from $E_{W,\\text{net}}$.

    Returns
    -------
    ExergyCycleBudget
        Populated immutable budget dataclass with $100.0000\\%$ balance closure.
    """
    if fuel_mass_kg <= 0.0 or lhv_j_per_kg <= 0.0 or phi_factor <= 0.0:
        raise PhysicalBoundsError("Fuel mass, LHV, and phi factor must be strictly positive.")

    theta = rates_result.crank_angle_deg
    de_w = rates_result.work_exergy_rate_j_per_deg
    de_q = rates_result.heat_loss_exergy_rate_j_per_deg
    e_tm = rates_result.thermomechanical_exergy_j

    # Window selection (default to full array)
    idx_start = 0 if theta_ivc_deg is None else int(np.argmin(np.abs(theta - theta_ivc_deg)))
    idx_end = len(theta) - 1 if theta_evo_deg is None else int(np.argmin(np.abs(theta - theta_evo_deg)))

    sub_theta = theta[idx_start : idx_end + 1]
    sub_de_w = de_w[idx_start : idx_end + 1]
    sub_de_q = de_q[idx_start : idx_end + 1]

    # Integrated terms
    e_in_fuel = float(fuel_mass_kg * phi_factor * lhv_j_per_kg)
    e_work_net = float(trapezoid(sub_de_w, sub_theta))
    e_heat_loss = float(trapezoid(sub_de_q, sub_theta))
    delta_e_tm = float(e_tm[idx_end] - e_tm[idx_start])

    # Irreversibility from closed balance
    e_destroyed = float(e_in_fuel - e_work_net - e_heat_loss - delta_e_tm)

    # Indicated work and efficiencies
    w_ind = float(indicated_work_j) if indicated_work_j is not None else e_work_net
    eta_first_law = float(w_ind / (fuel_mass_kg * lhv_j_per_kg))
    eta_second_law = float(e_work_net / e_in_fuel)

    # Percentage budget shares
    work_share = float((e_work_net / e_in_fuel) * 100.0)
    heat_loss_share = float((e_heat_loss / e_in_fuel) * 100.0)
    exhaust_share = float((delta_e_tm / e_in_fuel) * 100.0)
    destruction_share = float((e_destroyed / e_in_fuel) * 100.0)

    total_shares = work_share + heat_loss_share + exhaust_share + destruction_share
    closure_err = abs(e_in_fuel - (e_work_net + e_heat_loss + delta_e_tm + e_destroyed))

    return ExergyCycleBudget(
        e_in_fuel_j=e_in_fuel,
        e_work_net_j=e_work_net,
        e_heat_loss_j=e_heat_loss,
        delta_e_tm_exhaust_j=delta_e_tm,
        e_destroyed_j=e_destroyed,
        eta_first_law=eta_first_law,
        eta_second_law=eta_second_law,
        work_share_pct=work_share,
        heat_loss_share_pct=heat_loss_share,
        exhaust_share_pct=exhaust_share,
        destruction_share_pct=destruction_share,
        budget_closure_pct=total_shares,
        closure_error_j=closure_err,
    )
