"""ASME V&V 20-2009 Grid Convergence Index (GCI) and Richardson extrapolation."""

from __future__ import annotations

from ice_cfd_validator.vv.gci import (
    ASME_MIN_REFINEMENT_RATIO,
    DEFAULT_THREE_GRID_FS,
    ConvergenceRegime,
    GCIResult,
    GridSizeResult,
    compute_gci,
    compute_grid_size,
)

__all__ = [
    "ASME_MIN_REFINEMENT_RATIO",
    "DEFAULT_THREE_GRID_FS",
    "ConvergenceRegime",
    "GCIResult",
    "GridSizeResult",
    "compute_gci",
    "compute_grid_size",
]
