"""Domain-specific exceptions for ICE CFD verification and validation."""

from __future__ import annotations


class ValidationAssistantError(Exception):
    """Base exception for all ``ice_cfd_validator`` errors."""


class ForteParseError(ValidationAssistantError):
    """Raised when an ANSYS Forte CSV file is malformed or lacks required columns."""


class UnitDetectionError(ForteParseError):
    """Raised when a physical header unit is missing, unknown, or dimensionally invalid."""


class PhysicalBoundsError(ValidationAssistantError):
    """Raised when SI-converted arrays violate fundamental thermodynamic/kinematic bounds."""


class TDCPhaseError(ValidationAssistantError):
    """Raised when motored peak pressure violates thermodynamic TDC phasing tolerance."""

