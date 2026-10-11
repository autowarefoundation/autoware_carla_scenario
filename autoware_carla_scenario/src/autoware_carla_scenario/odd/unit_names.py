"""How units are spelled: the canonical spelling of an alias (``kph`` -> ``km/h``).

The same table as openodd's ``normalize_unit`` (a test keeps the two
together), in plain Python so that modules checked with Codon can use it.
"""

from __future__ import annotations

__all__ = ["normalize_unit"]

_ALIASES = {
    "kph": "km/h",
    "kmh": "km/h",
    "kmph": "km/h",
    "mps": "m/s",
    "msec": "ms",
    "sec": "s",
    "hr": "h",
    "mm/hr": "mm/h",
    "degree": "deg",
    "degrees": "deg",
    "percent": "%",
}


def normalize_unit(unit: str) -> str:
    """The canonical spelling of *unit* (``kph`` -> ``km/h``); ``""`` for none."""
    u = unit.strip()
    lower = u.lower()
    if lower in _ALIASES:
        return _ALIASES[lower]
    return u
