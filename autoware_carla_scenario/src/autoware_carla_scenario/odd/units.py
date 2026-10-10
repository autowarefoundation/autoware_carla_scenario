"""Units an ODD condition may be written in, and conversion between them.

An attribute's probe returns its value in one unit (``km/h`` for a speed), and
an OpenODD condition may name another (``"<= 30 m/s"``).  The condition's
number is converted into the probe's unit once, when the ODD is loaded, so
nothing is converted while a run is sampled.
"""

from __future__ import annotations

__all__ = ["UnitError", "convert", "normalize_unit"]


class UnitError(ValueError):
    """A unit that is unknown, or that cannot be converted into another."""


#: unit -> (dimension, factor to the dimension's base unit)
_UNITS: dict[str, tuple[str, float]] = {
    # length (m)
    "m": ("length", 1.0),
    "km": ("length", 1000.0),
    "cm": ("length", 0.01),
    "mm": ("length", 0.001),
    # speed (m/s)
    "m/s": ("speed", 1.0),
    "km/h": ("speed", 1.0 / 3.6),
    "mph": ("speed", 0.44704),
    # acceleration (m/s^2)
    "m/s^2": ("acceleration", 1.0),
    "m/s2": ("acceleration", 1.0),
    # time (s)
    "s": ("time", 1.0),
    "ms": ("time", 0.001),
    "min": ("time", 60.0),
    "h": ("time", 3600.0),
    # angle (deg)
    "deg": ("angle", 1.0),
    "rad": ("angle", 57.29577951308232),
    # precipitation rate (mm/h)
    "mm/h": ("precipitation_rate", 1.0),
    # ratio (%)
    "%": ("ratio", 1.0),
    "percent": ("ratio", 1.0),
}

_ALIASES = {
    "kph": "km/h",
    "kmh": "km/h",
    "kmph": "km/h",
    "mps": "m/s",
    "degree": "deg",
    "degrees": "deg",
    "sec": "s",
}


def normalize_unit(unit: str) -> str:
    """The canonical spelling of *unit* (``kph`` -> ``km/h``); ``""`` for none."""
    u = unit.strip()
    return _ALIASES.get(u.lower(), u)


def convert(value: float, from_unit: str, to_unit: str) -> float:
    """*value* in *from_unit*, expressed in *to_unit*.

    An empty unit on either side means the value is already in the other's
    unit, and so does the same unit on both sides, known or not.

    Raises:
        UnitError: if a unit is unknown or the two measure different things.
    """
    src, dst = normalize_unit(from_unit), normalize_unit(to_unit)
    if not src or not dst or src == dst:
        return value
    if src not in _UNITS:
        raise UnitError(f"unknown unit {from_unit!r}")
    if dst not in _UNITS:
        raise UnitError(f"unknown unit {to_unit!r}")
    (src_dim, src_factor), (dst_dim, dst_factor) = _UNITS[src], _UNITS[dst]
    if src_dim != dst_dim:
        raise UnitError(
            f"cannot convert {from_unit!r} ({src_dim}) into {to_unit!r} ({dst_dim})"
        )
    return value * src_factor / dst_factor
