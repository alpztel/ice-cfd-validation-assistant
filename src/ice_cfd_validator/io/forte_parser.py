"""ANSYS Forte CSV parsers with automatic header unit detection and SI conversion.

Converts spatially averaged 3D CFD monitor files (``thermo.csv``, ``spray.csv``,
``dynamic.csv``) into strictly typed SI data containers prior to any numerical
differentiation or thermodynamic integration, eliminating $10^6$-order $P\\,dV$
scaling errors (Heywood, 2018; NIST SP 811, Thompson & Taylor, 2008).
"""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
import re
from typing import IO, Final

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from ice_cfd_validator.constants import (
    ANGLE_TO_DEG,
    HEAT_RATE_TO_J_PER_DEG,
    LENGTH_TO_M,
    MASS_TO_KG,
    PRESSURE_TO_PA,
    TEMP_TO_K,
    TIME_TO_S,
    VOLUME_TO_M3,
)
from ice_cfd_validator.exceptions import (
    ForteParseError,
    PhysicalBoundsError,
    UnitDetectionError,
)

FloatArray = NDArray[np.float64]

_HEADER_UNIT_RE: Final[re.Pattern[str]] = re.compile(
    r"^(.*?)\s*(?:\[([^\]]*)\]|\(([^)]*)\))\s*$"
)

_CA_ALIASES: Final[tuple[str, ...]] = ("crank_angle", "crankangle", "ca", "theta")
_TIME_ALIASES: Final[tuple[str, ...]] = ("time", "sim_time", "time_after_soi", "asoi")
_PRESSURE_ALIASES: Final[tuple[str, ...]] = (
    "pressure", "cyl_pressure", "cylinder_pressure", "cyl_press", "mean_pressure", "p",
)
_VOLUME_ALIASES: Final[tuple[str, ...]] = (
    "volume", "cyl_volume", "cylinder_volume", "total_volume", "v",
)
_TEMP_ALIASES: Final[tuple[str, ...]] = (
    "temperature", "temp", "mean_temp", "mean_temperature", "cyl_temp", "t",
)
_MASS_ALIASES: Final[tuple[str, ...]] = (
    "mass", "total_mass", "trapped_mass", "gas_mass", "cyl_mass",
)
_WALL_HEAT_LOSS_ALIASES: Final[tuple[str, ...]] = (
    "wall_heat_loss", "dq_ht", "dq_wall", "heat_loss",
    "wall_heat_transfer", "wall_heat_transfer_rate",
)
_LIQ_PEN_ALIASES: Final[tuple[str, ...]] = (
    "liquid_penetration", "liq_pen", "liq_penetration",
    "liquid_tip_penetration", "liquid_length",
)
_VAP_PEN_ALIASES: Final[tuple[str, ...]] = (
    "vapor_penetration", "vap_pen", "vap_penetration",
    "vapor_tip_penetration", "fuel_vapor_penetration",
)
_SMD_ALIASES: Final[tuple[str, ...]] = ("smd", "sauter_mean_diameter", "d32")
_INJ_MASS_ALIASES: Final[tuple[str, ...]] = (
    "injected_mass", "inj_mass", "fuel_injected", "cum_injected_mass",
)
_CELL_COUNT_ALIASES: Final[tuple[str, ...]] = (
    "total_cells", "active_cells", "cell_count", "num_cells", "ncells",
)
_CELL_VOL_ALIASES: Final[tuple[str, ...]] = (
    "mean_cell_volume", "avg_cell_volume", "cell_volume",
)
_MIN_SIZE_ALIASES: Final[tuple[str, ...]] = ("min_cell_size", "min_grid_size")


@dataclass(frozen=True, slots=True)
class ForteThermoData:
    """SI-normalized in-cylinder thermodynamic state history from ``thermo.csv``.

    Attributes
    ----------
    crank_angle_deg : NDArray[np.float64]
        Crank angle $\\theta$ [deg].
    crank_angle_rad : NDArray[np.float64]
        Crank angle $\\theta$ [rad].
    pressure_pa : NDArray[np.float64]
        Spatially averaged absolute cylinder pressure $P$ [Pa].
    volume_m3 : NDArray[np.float64]
        Instantaneous cylinder volume $V$ [$\\text{m}^3$].
    temperature_k : NDArray[np.float64] | None
        Mass-averaged in-cylinder gas temperature $T$ [K], if present in input.
    time_s : NDArray[np.float64] | None
        Simulation time $t$ [s], if present in input.
    trapped_mass_kg : NDArray[np.float64] | None
        In-cylinder trapped gas mass $m$ [kg], if present in input.
    wall_heat_loss_j_per_deg : NDArray[np.float64] | None
        Crank-angle wall heat-loss rate $dQ_{ht}/d\\theta$ [$\\text{J/deg}$], if present in input.
    source_units : dict[str, str]
        Original unit tokens detected from file headers for audit traceability.
    """

    crank_angle_deg: FloatArray
    crank_angle_rad: FloatArray
    pressure_pa: FloatArray
    volume_m3: FloatArray
    temperature_k: FloatArray | None = None
    time_s: FloatArray | None = None
    trapped_mass_kg: FloatArray | None = None
    wall_heat_loss_j_per_deg: FloatArray | None = None
    source_units: dict[str, str] | None = None


@dataclass(frozen=True, slots=True)
class ForteSprayData:
    """SI-normalized fuel spray evolution history from ``spray.csv``.

    Attributes
    ----------
    time_s : NDArray[np.float64]
        Elapsed spray/simulation time $t$ [s].
    liquid_penetration_m : NDArray[np.float64]
        Liquid-phase spray tip penetration $S_L$ [m].
    vapor_penetration_m : NDArray[np.float64] | None
        Vapor-phase spray penetration $S_V$ [m], if available.
    smd_m : NDArray[np.float64] | None
        Droplet Sauter Mean Diameter $D_{32}$ [m], if available.
    crank_angle_deg : NDArray[np.float64] | None
        Crank angle $\\theta$ [deg], if available.
    crank_angle_rad : NDArray[np.float64] | None
        Crank angle $\\theta$ [rad], if available.
    injected_mass_kg : NDArray[np.float64] | None
        Cumulative injected fuel mass $m_{\\text{inj}}$ [kg], if available.
    source_units : dict[str, str] | None
        Original unit tokens detected from file headers for audit traceability.
    """

    time_s: FloatArray
    liquid_penetration_m: FloatArray
    vapor_penetration_m: FloatArray | None = None
    smd_m: FloatArray | None = None
    crank_angle_deg: FloatArray | None = None
    crank_angle_rad: FloatArray | None = None
    injected_mass_kg: FloatArray | None = None
    source_units: dict[str, str] | None = None


@dataclass(frozen=True, slots=True)
class ForteDynamicData:
    """SI-normalized dynamic mesh / AMR history from ``dynamic.csv``.

    Attributes
    ----------
    cell_count : NDArray[np.float64]
        Instantaneous active computational cell count $N_i$ [-].
    crank_angle_deg : NDArray[np.float64] | None
        Crank angle $\\theta$ [deg], if available.
    time_s : NDArray[np.float64] | None
        Simulation time $t$ [s], if available.
    mean_cell_volume_m3 : NDArray[np.float64] | None
        Mean cell volume $\\Delta V_{\\text{mean}}$ [$\\text{m}^3$], if available.
    min_cell_size_m : NDArray[np.float64] | None
        Minimum AMR grid spacing $\\Delta x_{\\min}$ [m], if available.
    source_units : dict[str, str] | None
        Original unit tokens detected from file headers for audit traceability.
    """

    cell_count: FloatArray
    crank_angle_deg: FloatArray | None = None
    time_s: FloatArray | None = None
    mean_cell_volume_m3: FloatArray | None = None
    min_cell_size_m: FloatArray | None = None
    source_units: dict[str, str] | None = None


def _normalize_name(raw_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", raw_name.strip().lower()).strip("_")


def _normalize_unit(raw_unit: str) -> str:
    return (
        raw_unit.strip()
        .strip("[]()")
        .strip()
        .replace("³", "^3")
        .replace("²", "^2")
        .replace("µ", "u")
        .replace("μ", "u")
        .replace("°", "")
        .replace(" ", "")
    )


def _is_numeric_row(line: str) -> bool:
    tokens = [tok.strip() for tok in line.split(",") if tok.strip()]
    if not tokens:
        return False
    try:
        for tok in tokens:
            float(tok)
        return True
    except ValueError:
        return False


def _read_raw_columns(
    source: Path | str | IO[str],
) -> dict[str, tuple[FloatArray, str]]:
    """Parse 1-row or 2-row ANSYS Forte CSV headers and extract float64 columns."""
    if isinstance(source, (str, Path)):
        text = Path(source).read_text(encoding="utf-8-sig")
    else:
        text = source.read()

    active_lines = [
        ln for ln in text.splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    if len(active_lines) < 2:
        raise ForteParseError("CSV contains insufficient header/data rows.")

    header_tokens = [tok.strip() for tok in active_lines[0].split(",")]
    has_unit_row = not _is_numeric_row(active_lines[1])
    if has_unit_row and len(active_lines) < 3:
        raise ForteParseError("CSV contains header and unit rows but zero numeric rows.")

    unit_row_tokens = (
        [tok.strip() for tok in active_lines[1].split(",")]
        if has_unit_row
        else [""] * len(header_tokens)
    )
    data_start = 2 if has_unit_row else 1

    try:
        df = pd.read_csv(
            StringIO("\n".join(active_lines[data_start:])),
            header=None,
            dtype=np.float64,
            skipinitialspace=True,
        )
    except ValueError as exc:
        raise ForteParseError(f"Non-numeric entries detected in data block: {exc}") from exc

    if df.shape[1] != len(header_tokens):
        raise ForteParseError(
            f"Column count mismatch: {len(header_tokens)} headers vs {df.shape[1]} data columns."
        )

    columns: dict[str, tuple[FloatArray, str]] = {}
    for idx, raw_hdr in enumerate(header_tokens):
        match = _HEADER_UNIT_RE.match(raw_hdr)
        if match:
            col_name = _normalize_name(match.group(1))
            inline_unit = match.group(2) if match.group(2) is not None else (match.group(3) or "")
        else:
            col_name = _normalize_name(raw_hdr)
            inline_unit = ""

        row_unit = unit_row_tokens[idx] if idx < len(unit_row_tokens) else ""
        unit_token = _normalize_unit(inline_unit or row_unit)
        arr: FloatArray = df.iloc[:, idx].to_numpy(dtype=np.float64, copy=True)
        if not np.all(np.isfinite(arr)):
            raise ForteParseError(f"Column '{col_name}' contains NaN or infinite values.")
        columns[col_name] = (arr, unit_token)

    return columns


def _resolve_scaled(
    columns: dict[str, tuple[FloatArray, str]],
    aliases: tuple[str, ...],
    scale_map: dict[str, float],
    canonical_name: str,
    *,
    required: bool = True,
) -> tuple[FloatArray | None, str | None]:
    for alias in aliases:
        if alias in columns:
            raw_arr, unit = columns[alias]
            if not unit or unit not in scale_map:
                raise UnitDetectionError(
                    f"Column '{alias}' ({canonical_name}) has missing or invalid unit "
                    f"'{unit}'. Expected one of {sorted(scale_map.keys())}."
                )
            return raw_arr * scale_map[unit], unit
    if required:
        raise ForteParseError(
            f"Required '{canonical_name}' column not found. Checked aliases: {aliases}."
        )
    return None, None


def _resolve_temperature(
    columns: dict[str, tuple[FloatArray, str]],
    *,
    required: bool = False,
) -> tuple[FloatArray | None, str | None]:
    for alias in _TEMP_ALIASES:
        if alias in columns:
            raw_arr, unit = columns[alias]
            if not unit or unit not in TEMP_TO_K:
                raise UnitDetectionError(
                    f"Temperature column '{alias}' has missing or invalid unit '{unit}'. "
                    f"Expected one of {sorted(TEMP_TO_K.keys())}."
                )
            scale, offset = TEMP_TO_K[unit]
            return raw_arr * scale + offset, unit
    if required:
        raise ForteParseError(
            f"Required 'temperature' column not found. Checked aliases: {_TEMP_ALIASES}."
        )
    return None, None


def parse_thermo_csv(source: Path | str | IO[str]) -> ForteThermoData:
    """Parse ANSYS Forte ``thermo.csv`` and convert state variables to strict SI.

    Enforces pre-differentiation conversion $P \\to \\text{Pa}$, $V \\to \\text{m}^3$,
    and $T \\to \\text{K}$ so First-Law $P\\,dV$ and $V\\,dP$ terms avoid $10^6$
    unit scaling errors (Heywood, 2018).

    Parameters
    ----------
    source : Path | str | IO[str]
        Filesystem path or text stream of an ANSYS Forte ``thermo.csv`` export.

    Returns
    -------
    ForteThermoData
        Frozen SI-normalized thermodynamic dataset (Pa, $\\text{m}^3$, K, kg, s, deg/rad).

    Raises
    ------
    ForteParseError
        If required columns are missing or numeric parsing fails.
    UnitDetectionError
        If any physical column lacks a valid pressure, volume, temperature, or angle unit.
    PhysicalBoundsError
        If converted $P \\le 0\\,\\text{Pa}$, $V \\le 0\\,\\text{m}^3$, or $T \\le 0\\,\\text{K}$.
    """
    columns = _read_raw_columns(source)

    ca_deg, ca_unit = _resolve_scaled(columns, _CA_ALIASES, ANGLE_TO_DEG, "crank_angle")
    pres_pa, p_unit = _resolve_scaled(columns, _PRESSURE_ALIASES, PRESSURE_TO_PA, "pressure")
    vol_m3, v_unit = _resolve_scaled(columns, _VOLUME_ALIASES, VOLUME_TO_M3, "volume")
    temp_k, t_unit = _resolve_temperature(columns, required=False)
    time_s, time_unit = _resolve_scaled(
        columns, _TIME_ALIASES, TIME_TO_S, "time", required=False
    )
    mass_kg, m_unit = _resolve_scaled(
        columns, _MASS_ALIASES, MASS_TO_KG, "trapped_mass", required=False
    )
    wall_ht, ht_unit = _resolve_scaled(
        columns, _WALL_HEAT_LOSS_ALIASES, HEAT_RATE_TO_J_PER_DEG, "wall_heat_loss", required=False
    )

    assert ca_deg is not None and pres_pa is not None and vol_m3 is not None
    assert ca_unit is not None and p_unit is not None and v_unit is not None

    if np.any(pres_pa <= 0.0):
        raise PhysicalBoundsError("Converted cylinder pressure contains values <= 0 Pa.")
    if np.any(vol_m3 <= 0.0):
        raise PhysicalBoundsError("Converted cylinder volume contains values <= 0 m^3.")
    if temp_k is not None and np.any(temp_k <= 0.0):
        raise PhysicalBoundsError("Converted cylinder temperature contains values <= 0 K.")
    if mass_kg is not None and np.any(mass_kg <= 0.0):
        raise PhysicalBoundsError("Converted trapped gas mass contains values <= 0 kg.")

    units: dict[str, str] = {
        "crank_angle": ca_unit,
        "pressure": p_unit,
        "volume": v_unit,
    }
    if t_unit is not None:
        units["temperature"] = t_unit
    if time_unit is not None:
        units["time"] = time_unit
    if m_unit is not None:
        units["trapped_mass"] = m_unit
    if ht_unit is not None:
        units["wall_heat_loss"] = ht_unit

    return ForteThermoData(
        crank_angle_deg=ca_deg,
        crank_angle_rad=np.deg2rad(ca_deg),
        pressure_pa=pres_pa,
        volume_m3=vol_m3,
        temperature_k=temp_k,
        time_s=time_s,
        trapped_mass_kg=mass_kg,
        wall_heat_loss_j_per_deg=wall_ht,
        source_units=units,
    )


def parse_spray_csv(
    source: Path | str | IO[str],
    *,
    rpm: float | None = None,
) -> ForteSprayData:
    """Parse ANSYS Forte ``spray.csv`` and convert penetration/time arrays to SI.

    Converts liquid/vapor penetration and droplet SMD to meters [m] and time to
    seconds [s] for Hiroyasu & Arai (1990) and Siebers (1999) scaling validation.
    When ``Time`` is absent from ``spray.csv``, elapsed time is computed from
    crank angle via $\\Delta t = (\\theta - \\theta_0) / (6\\,\\text{RPM})$.

    Parameters
    ----------
    source : Path | str | IO[str]
        Filesystem path or text stream of an ANSYS Forte ``spray.csv`` export.
    rpm : float | None, optional
        Engine speed $N$ [rev/min] used to convert crank angle to seconds [s]
        only when an explicit ``Time`` column is absent.

    Returns
    -------
    ForteSprayData
        Frozen SI-normalized spray evolution dataset (s, m, kg, deg/rad).
    """
    columns = _read_raw_columns(source)

    liq_m, liq_unit = _resolve_scaled(
        columns, _LIQ_PEN_ALIASES, LENGTH_TO_M, "liquid_penetration"
    )
    vap_m, vap_unit = _resolve_scaled(
        columns, _VAP_PEN_ALIASES, LENGTH_TO_M, "vapor_penetration", required=False
    )
    smd_m, smd_unit = _resolve_scaled(
        columns, _SMD_ALIASES, LENGTH_TO_M, "smd", required=False
    )
    time_s, time_unit = _resolve_scaled(
        columns, _TIME_ALIASES, TIME_TO_S, "time", required=False
    )
    ca_deg, ca_unit = _resolve_scaled(
        columns, _CA_ALIASES, ANGLE_TO_DEG, "crank_angle", required=False
    )
    inj_kg, inj_unit = _resolve_scaled(
        columns, _INJ_MASS_ALIASES, MASS_TO_KG, "injected_mass", required=False
    )

    assert liq_m is not None and liq_unit is not None

    if time_s is None:
        if ca_deg is None or rpm is None:
            raise ForteParseError(
                "spray.csv lacks a 'Time' column; provide crank angle column and rpm > 0."
            )
        if rpm <= 0.0:
            raise PhysicalBoundsError("Engine speed rpm must be strictly positive (> 0).")
        time_s = (ca_deg - ca_deg[0]) / (6.0 * rpm)
        time_unit = "derived_from_ca_rpm"

    if np.any(liq_m < 0.0):
        raise PhysicalBoundsError("Liquid spray penetration cannot be negative (< 0 m).")
    if vap_m is not None and np.any(vap_m < 0.0):
        raise PhysicalBoundsError("Vapor spray penetration cannot be negative (< 0 m).")
    if smd_m is not None and np.any(smd_m < 0.0):
        raise PhysicalBoundsError("Droplet SMD cannot be negative (< 0 m).")
    if inj_kg is not None and np.any(inj_kg < 0.0):
        raise PhysicalBoundsError("Cumulative injected fuel mass cannot be negative (< 0 kg).")

    assert time_unit is not None
    units: dict[str, str] = {"time": time_unit, "liquid_penetration": liq_unit}
    if vap_unit is not None:
        units["vapor_penetration"] = vap_unit
    if smd_unit is not None:
        units["smd"] = smd_unit
    if ca_unit is not None:
        units["crank_angle"] = ca_unit
    if inj_unit is not None:
        units["injected_mass"] = inj_unit

    return ForteSprayData(
        time_s=time_s,
        liquid_penetration_m=liq_m,
        vapor_penetration_m=vap_m,
        smd_m=smd_m,
        crank_angle_deg=ca_deg,
        crank_angle_rad=np.deg2rad(ca_deg) if ca_deg is not None else None,
        injected_mass_kg=inj_kg,
        source_units=units,
    )


def parse_dynamic_csv(source: Path | str | IO[str]) -> ForteDynamicData:
    """Parse ANSYS Forte ``dynamic.csv`` AMR / mesh statistics into SI units.

    Extracts active computational cell count $N_i$ and optional mean cell volume
    $\\Delta V_{\\text{mean}}$ [$\\text{m}^3$] for ASME V&V 20-2009 representative
    grid size calculation $h_i = (N_1 / N_i)^{1/3}$ or $(\\Delta V_{\\text{mean}})^{1/3}$
    (Celik et al., 2008).

    Parameters
    ----------
    source : Path | str | IO[str]
        Filesystem path or text stream of an ANSYS Forte ``dynamic.csv`` export.

    Returns
    -------
    ForteDynamicData
        Frozen SI-normalized dynamic mesh history.
    """
    columns = _read_raw_columns(source)

    cell_count: FloatArray | None = None
    cell_unit = "count"
    for alias in _CELL_COUNT_ALIASES:
        if alias in columns:
            raw_arr, raw_u = columns[alias]
            if raw_u not in ("", "-", "count", "cells", "none"):
                raise UnitDetectionError(
                    f"Cell count column '{alias}' must be dimensionless, got '{raw_u}'."
                )
            cell_count = raw_arr
            cell_unit = raw_u or "dimensionless"
            break

    if cell_count is None:
        raise ForteParseError(
            f"Required 'cell_count' column not found. Checked aliases: {_CELL_COUNT_ALIASES}."
        )
    if np.any(cell_count <= 0.0):
        raise PhysicalBoundsError("Computational cell count must be strictly positive (> 0).")

    ca_deg, ca_unit = _resolve_scaled(
        columns, _CA_ALIASES, ANGLE_TO_DEG, "crank_angle", required=False
    )
    time_s, time_unit = _resolve_scaled(
        columns, _TIME_ALIASES, TIME_TO_S, "time", required=False
    )
    mean_vol_m3, vol_unit = _resolve_scaled(
        columns, _CELL_VOL_ALIASES, VOLUME_TO_M3, "mean_cell_volume", required=False
    )
    min_size_m, size_unit = _resolve_scaled(
        columns, _MIN_SIZE_ALIASES, LENGTH_TO_M, "min_cell_size", required=False
    )

    if mean_vol_m3 is not None and np.any(mean_vol_m3 <= 0.0):
        raise PhysicalBoundsError("Mean cell volume must be strictly positive (> 0 m^3).")
    if min_size_m is not None and np.any(min_size_m <= 0.0):
        raise PhysicalBoundsError("Minimum cell size must be strictly positive (> 0 m).")

    units: dict[str, str] = {"cell_count": cell_unit}
    if ca_unit is not None:
        units["crank_angle"] = ca_unit
    if time_unit is not None:
        units["time"] = time_unit
    if vol_unit is not None:
        units["mean_cell_volume"] = vol_unit
    if size_unit is not None:
        units["min_cell_size"] = size_unit

    return ForteDynamicData(
        cell_count=cell_count,
        crank_angle_deg=ca_deg,
        time_s=time_s,
        mean_cell_volume_m3=mean_vol_m3,
        min_cell_size_m=min_size_m,
        source_units=units,
    )

