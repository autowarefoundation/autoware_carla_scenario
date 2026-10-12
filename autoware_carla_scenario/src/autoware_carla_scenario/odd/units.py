"""Units an ODD condition may be written in, and conversion between them (openodd's)."""

from __future__ import annotations

from openodd import UnitError, Units, convert, normalize_unit

__all__ = ["UNITS", "UnitError", "Units", "convert", "normalize_unit"]

#: The built-in units.
UNITS = Units()
