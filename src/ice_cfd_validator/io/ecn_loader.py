"""Engine Combustion Network (ECN Spray A / Spray D) experimental data loader.

Parses open-literature Sandia/ECN comma- or whitespace-delimited penetration
datasets (with `#` or `%` metadata headers and inline or 2nd-row units) into
strict SI containers (`s`, `m`) for CFD validation and ASME V&V 20-2009 $u_D$
uncertainty quantification.
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

from ice_cfd_validator.constants import LENGTH_TO_M, TIME_TO_S
from ice_cfd_validator.exceptions import (
    ForteParseError,
    PhysicalBoundsError,
    UnitDetectionError,
)

FloatArray = NDArray[np.float64]

_ECN_HEADER_RE: Final[re.Pattern[str]] = re.compile(
    r"^(.*?)\s*(?:\[([^\]]*)\]|\(([^)]*)\))\s*$"
)

_TIME_ALIASES: Final[tuple[str, ...]] = (
    "time", "t", "time_asoi", "asoi", "t_asoi", "elapsed_time",
)
_LIQ_ALIASES: Final[tuple[str, ...]] = (
    "liquid_penetration", "liq_pen", "liq_penetration",
    "liquid_length", "ll", "s_liq", "l_liq", "liquid",
)
_VAP_ALIASES: Final[tuple[str, ...]] = (
    "vapor_penetration", "vap_pen", "vap_penetration",
    "vapor_length", "s_vap", "s", "penetration", "jet_penetration", "vapor",
)
_LIQ_STD_ALIASES: Final[tuple[str, ...]] = (
    "liquid_std", "liq_std", "ll_std", "sigma_liq", "liquid_ci", "liq_uncertainty",
)
_VAP_STD_ALIASES: Final[tuple[str, ...]] = (
    "vapor_std", "vap_std", "s_std", "sigma_vap", "vapor_ci", "vap_uncertainty",
)


@dataclass(frozen=True, slots=True)
class ECNSprayDataset:
    """SI-normalized Engine Combustion Network (ECN) experimental spray dataset.

    Attributes
    ----------
    time_s : NDArray[np.float64]
        Time after start of injection $t$ [s] ($\\ge 0$).
    liquid_penetration_m : NDArray[np.float64] | None
        Ensemble-averaged liquid-phase penetration / liquid length $S_L(t)$ [m].
    vapor_penetration_m : NDArray[np.float64] | None
        Ensemble-averaged vapor-phase spray tip penetration $S_V(t)$ [m].
    liquid_std_m : NDArray[np.float64] | None
        Experimental standard deviation / uncertainty of liquid length [m], if available.
    vapor_std_m : NDArray[np.float64] | None
        Experimental standard deviation / uncertainty of vapor penetration [m], if available.
    condition_label : str
        Dataset identifier (e.g., ``"ECN_Spray_A"`` or filename stem).
    source_units : dict[str, str]
        Detected unit tokens from raw ECN header for traceability.
    """

    time_s: FloatArray
    liquid_penetration_m: FloatArray | None = None
    vapor_penetration_m: FloatArray | None = None
    liquid_std_m: FloatArray | None = None
    vapor_std_m: FloatArray | None = None
    condition_label: str = "ECN_Spray"
    source_units: dict[str, str] | None = None


def _split_ecn_row(line: str, is_csv: bool) -> list[str]:
    if is_csv:
        return [tok.strip() for tok in line.split(",")]
    return [tok.strip() for tok in line.split()]


def _normalize_ecn_name(raw_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", raw_name.strip().lower()).strip("_")


def _normalize_ecn_unit(raw_unit: str) -> str:
    return (
        raw_unit.strip()
        .strip("[]()")
        .strip()
        .replace("µ", "u")
        .replace("μ", "u")
        .replace(" ", "")
    )


def _is_numeric_tokens(tokens: list[str]) -> bool:
    if not tokens:
        return False
    try:
        for tok in tokens:
            float(tok)
        return True
    except ValueError:
        return False


def _resolve_ecn_column(
    columns: dict[str, tuple[FloatArray, str]],
    aliases: tuple[str, ...],
    scale_map: dict[str, float],
    canonical: str,
    default_unit: str | None,
) -> tuple[FloatArray | None, str | None]:
    for alias in aliases:
        if alias in columns:
            raw_arr, raw_u = columns[alias]
            unit = raw_u or (default_unit or "")
            if not unit or unit not in scale_map:
                raise UnitDetectionError(
                    f"ECN column '{alias}' ({canonical}) has missing/invalid unit '{unit}'. "
                    f"Expected one of {sorted(scale_map.keys())}."
                )
            return raw_arr * scale_map[unit], unit
    return None, None


def load_ecn_spray_data(
    source: Path | str | IO[str],
    *,
    condition_label: str | None = None,
    default_time_unit: str | None = None,
    default_length_unit: str | None = None,
    trim_negative_time: bool = True,
) -> ECNSprayDataset:
    """Load ECN Spray A / Spray D experimental CSV or whitespace text data into SI units.

    Supports both comma-delimited (`.csv`) and whitespace-delimited (`.txt`/`.dat`)
    Sandia ECN tables with `#` or `%` comment blocks and inline (`Time [ms]`) or
    second-row (`[ms]`) unit declarations.

    Parameters
    ----------
    source : Path | str | IO[str]
        Path or open text stream of an ECN experimental penetration dataset.
    condition_label : str | None, optional
        Dataset label; defaults to file stem or ``"ECN_Spray"``.
    default_time_unit : str | None, optional
        Fallback time unit (e.g., ``"ms"``) ONLY when unitless plain-text columns
        are loaded; strict header unit detection is enforced when ``None``.
    default_length_unit : str | None, optional
        Fallback length unit (e.g., ``"mm"``) ONLY when unitless plain-text columns
        are loaded; strict header unit detection is enforced when ``None``.
    trim_negative_time : bool, default=True
        If ``True``, drops pre-SOI rows ($t < 0\\,\\text{s}$) and clamps tiny negative
        noise on post-SOI penetration to $0\\,\\text{m}$ only if $\\ge -10^{-9}\\,\\text{m}$.

    Returns
    -------
    ECNSprayDataset
        SI-normalized (`s`, `m`) ECN experimental benchmark dataset.
    """
    if isinstance(source, (str, Path)):
        path_obj = Path(source)
        text = path_obj.read_text(encoding="utf-8-sig")
        label = condition_label or path_obj.stem
    else:
        text = source.read()
        label = condition_label or "ECN_Spray"

    active_lines = [
        ln.strip()
        for ln in text.splitlines()
        if ln.strip() and not ln.lstrip().startswith(("#", "%"))
    ]
    if len(active_lines) < 2:
        raise ForteParseError("ECN dataset contains insufficient header/data rows.")

    is_csv = "," in active_lines[0]
    header_tokens = _split_ecn_row(active_lines[0], is_csv=is_csv)
    second_tokens = _split_ecn_row(active_lines[1], is_csv=is_csv)
    has_unit_row = not _is_numeric_tokens(second_tokens)

    if has_unit_row and len(active_lines) < 3:
        raise ForteParseError("ECN dataset contains header and unit rows but zero numeric rows.")

    unit_tokens = second_tokens if has_unit_row else [""] * len(header_tokens)
    data_start = 2 if has_unit_row else 1

    try:
        if is_csv:
            df = pd.read_csv(
                StringIO("\n".join(active_lines[data_start:])),
                header=None,
                dtype=np.float64,
                skipinitialspace=True,
            )
        else:
            df = pd.read_csv(
                StringIO("\n".join(active_lines[data_start:])),
                header=None,
                sep=r"\s+",
                dtype=np.float64,
            )
    except ValueError as exc:
        raise ForteParseError(f"Non-numeric entries in ECN data table: {exc}") from exc

    if df.shape[1] != len(header_tokens):
        raise ForteParseError(
            f"ECN column count mismatch: {len(header_tokens)} headers vs {df.shape[1]} columns."
        )

    columns: dict[str, tuple[FloatArray, str]] = {}
    for idx, raw_hdr in enumerate(header_tokens):
        match = _ECN_HEADER_RE.match(raw_hdr)
        if match:
            col_name = _normalize_ecn_name(match.group(1))
            inline_u = match.group(2) if match.group(2) is not None else (match.group(3) or "")
        else:
            col_name = _normalize_ecn_name(raw_hdr)
            inline_u = ""
        row_u = unit_tokens[idx] if idx < len(unit_tokens) else ""
        unit_tok = _normalize_ecn_unit(inline_u or row_u)
        arr: FloatArray = df.iloc[:, idx].to_numpy(dtype=np.float64, copy=True)
        if not np.all(np.isfinite(arr)):
            raise ForteParseError(f"ECN column '{col_name}' contains NaN or infinite values.")
        columns[col_name] = (arr, unit_tok)

    time_s, t_unit = _resolve_ecn_column(
        columns, _TIME_ALIASES, TIME_TO_S, "time", default_time_unit
    )
    if time_s is None or t_unit is None:
        raise ForteParseError(
            f"Required ECN 'time' column not found. Checked aliases: {_TIME_ALIASES}."
        )

    liq_m, liq_unit = _resolve_ecn_column(
        columns, _LIQ_ALIASES, LENGTH_TO_M, "liquid_penetration", default_length_unit
    )
    vap_m, vap_unit = _resolve_ecn_column(
        columns, _VAP_ALIASES, LENGTH_TO_M, "vapor_penetration", default_length_unit
    )
    if liq_m is None and vap_m is None:
        raise ForteParseError(
            "ECN dataset must contain at least one liquid or vapor penetration column."
        )

    liq_std_m, liq_std_u = _resolve_ecn_column(
        columns, _LIQ_STD_ALIASES, LENGTH_TO_M, "liquid_std", default_length_unit
    )
    vap_std_m, vap_std_u = _resolve_ecn_column(
        columns, _VAP_STD_ALIASES, LENGTH_TO_M, "vapor_std", default_length_unit
    )

    if trim_negative_time:
        keep = time_s >= 0.0
        if not np.any(keep):
            raise PhysicalBoundsError("All ECN time stamps are negative (< 0 s).")
        time_s = time_s[keep]
        if liq_m is not None:
            liq_m = liq_m[keep]
        if vap_m is not None:
            vap_m = vap_m[keep]
        if liq_std_m is not None:
            liq_std_m = liq_std_m[keep]
        if vap_std_m is not None:
            vap_std_m = vap_std_m[keep]

    if liq_m is not None and np.any(liq_m < 0.0):
        raise PhysicalBoundsError("ECN liquid penetration contains negative values (< 0 m).")
    if vap_m is not None and np.any(vap_m < 0.0):
        raise PhysicalBoundsError("ECN vapor penetration contains negative values (< 0 m).")
    if liq_std_m is not None and np.any(liq_std_m < 0.0):
        raise PhysicalBoundsError("ECN liquid_std contains negative values (< 0 m).")
    if vap_std_m is not None and np.any(vap_std_m < 0.0):
        raise PhysicalBoundsError("ECN vapor_std contains negative values (< 0 m).")

    units: dict[str, str] = {"time": t_unit}
    if liq_unit is not None:
        units["liquid_penetration"] = liq_unit
    if vap_unit is not None:
        units["vapor_penetration"] = vap_unit
    if liq_std_u is not None:
        units["liquid_std"] = liq_std_u
    if vap_std_u is not None:
        units["vapor_std"] = vap_std_u

    return ECNSprayDataset(
        time_s=time_s,
        liquid_penetration_m=liq_m,
        vapor_penetration_m=vap_m,
        liquid_std_m=liq_std_m,
        vapor_std_m=vap_std_m,
        condition_label=label,
        source_units=units,
    )

