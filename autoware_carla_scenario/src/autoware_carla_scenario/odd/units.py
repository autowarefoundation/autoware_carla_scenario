"""Units an ODD condition may be written in, and conversion between them.

An attribute's probe returns its value in one unit (``km/h`` for a speed),
and an OpenODD condition may name another (``"<= 30 m/s"``).  The
condition's number is converted into the probe's unit once, when the ODD is
loaded, so nothing is converted while a run is sampled.

Units belong to OpenODD's *unit types* (the basic unit types of its YAML
mapping: ``velocity``, ``length``, ``precipitation_rate``, ...).  A unit
converts to its type's base unit by ``value * scale + offset``, which covers
temperatures as well.  A taxonomy can add units with OpenODD's
``conversion:`` block (:meth:`Units.add_conversions`).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

__all__ = ["UNITS", "UnitError", "Units", "convert", "normalize_unit"]


class UnitError(ValueError):
    """A unit that is unknown, or that cannot be converted into another."""


#: unit -> (unit type, scale, offset) to the unit type's base unit.
_BUILTIN: dict[str, tuple[str, float, float]] = {
    # length (m)
    "m": ("length", 1.0, 0.0),
    "km": ("length", 1000.0, 0.0),
    "cm": ("length", 0.01, 0.0),
    "mm": ("length", 0.001, 0.0),
    "mi": ("length", 1609.344, 0.0),
    "in": ("length", 0.0254, 0.0),
    "ft": ("length", 0.3048, 0.0),
    # velocity (m/s)
    "m/s": ("velocity", 1.0, 0.0),
    "km/h": ("velocity", 1.0 / 3.6, 0.0),
    "mph": ("velocity", 0.44704, 0.0),
    # acceleration (m/s^2)
    "m/s^2": ("acceleration", 1.0, 0.0),
    "g": ("acceleration", 9.80665, 0.0),
    # time (s)
    "s": ("time", 1.0, 0.0),
    "ms": ("time", 0.001, 0.0),
    "min": ("time", 60.0, 0.0),
    "h": ("time", 3600.0, 0.0),
    # angle (deg)
    "deg": ("angle", 1.0, 0.0),
    "rad": ("angle", 57.29577951308232, 0.0),
    # precipitation rate (mm/h)
    "mm/h": ("precipitation_rate", 1.0, 0.0),
    # temperature (C)
    "C": ("temperature", 1.0, 0.0),
    "K": ("temperature", 1.0, -273.15),
    "F": ("temperature", 5.0 / 9.0, -32.0 * 5.0 / 9.0),
    # fraction (1); % is fraction * 100
    "%": ("fraction", 0.01, 0.0),
    # bandwidth (bit/s)
    "bps": ("bandwidth", 1.0, 0.0),
    "kbps": ("bandwidth", 1e3, 0.0),
    "Mbps": ("bandwidth", 1e6, 0.0),
    "Gbps": ("bandwidth", 1e9, 0.0),
    # frequency (Hz)
    "Hz": ("frequency", 1.0, 0.0),
    "kHz": ("frequency", 1e3, 0.0),
    # illuminance (lx)
    "lx": ("illuminance", 1.0, 0.0),
    # count and occurrence
    "count": ("count", 1.0, 0.0),
    "1/h": ("occurrence", 1.0, 0.0),
    "1/hr": ("occurrence", 1.0, 0.0),
    "occ/hr": ("occurrence", 1.0, 0.0),
}

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

#: Unit types that are the same dimension under another name.
_SAME_TYPE = {"duration": "time", "distance": "length", "speed": "velocity"}


def normalize_unit(unit: str) -> str:
    """The canonical spelling of *unit* (``kph`` -> ``km/h``); ``""`` for none."""
    u = unit.strip()
    return _ALIASES.get(u.lower(), _ALIASES.get(u, u))


def _unit_type(name: str) -> str:
    return _SAME_TYPE.get(name, name)


class Units:
    """A table of units: the built-in ones, and those a taxonomy adds."""

    def __init__(self) -> None:
        self._table: dict[str, tuple[str, float, float]] = dict(_BUILTIN)

    def add_conversions(self, conversion: Mapping[str, Any]) -> None:
        """Add the units of an OpenODD ``conversion:`` block.

        ``conversion: {<unit type>: {<from>: {<to>: {scale, offset}}}}``
        reads "a value in *from* is ``value * scale + offset`` in *to*".  A
        unit known already anchors its type, so the others join it.
        """
        for unit_type, sources in conversion.items():
            for src, targets in (sources or {}).items():
                for dst, spec in (targets or {}).items():
                    scale = float((spec or {}).get("scale", 1.0))
                    offset = float((spec or {}).get("offset", 0.0))
                    src_n, dst_n = normalize_unit(str(src)), normalize_unit(str(dst))
                    if src_n not in self._table and dst_n not in self._table:
                        # Neither is known: make *src* the base of the type.
                        self._table[src_n] = (_unit_type(str(unit_type)), 1.0, 0.0)
                    if src_n in self._table and dst_n not in self._table:
                        t, s_scale, s_offset = self._table[src_n]
                        # dst = src * scale + offset  =>  src = (dst - offset) / scale
                        self._table[dst_n] = (
                            t,
                            s_scale / scale,
                            s_offset - s_scale * offset / scale,
                        )
                    elif dst_n in self._table and src_n not in self._table:
                        t, d_scale, d_offset = self._table[dst_n]
                        self._table[src_n] = (
                            t,
                            d_scale * scale,
                            d_scale * offset + d_offset,
                        )

    def unit_type(self, unit: str) -> Optional[str]:
        """The unit type *unit* measures, or ``None`` if it is unknown."""
        entry = self._table.get(normalize_unit(unit))
        return entry[0] if entry else None

    def convert(self, value: float, from_unit: str, to_unit: str) -> float:
        """*value* in *from_unit*, expressed in *to_unit*.

        An empty unit on either side means the value is already in the
        other's unit, and so does the same unit on both sides, known or not.

        Raises:
            UnitError: if a unit is unknown or the two measure different things.
        """
        src, dst = normalize_unit(from_unit), normalize_unit(to_unit)
        if not src or not dst or src == dst:
            return value
        if src not in self._table:
            raise UnitError(f"unknown unit {from_unit!r}")
        if dst not in self._table:
            raise UnitError(f"unknown unit {to_unit!r}")
        (src_type, src_scale, src_offset) = self._table[src]
        (dst_type, dst_scale, dst_offset) = self._table[dst]
        if src_type != dst_type:
            raise UnitError(
                f"cannot convert {from_unit!r} ({src_type}) into {to_unit!r} ({dst_type})"
            )
        base = value * src_scale + src_offset
        return (base - dst_offset) / dst_scale

    def check_type(self, unit: str, unit_type: str) -> None:
        """Refuse *unit* when it does not measure *unit_type* (both known)."""
        actual = self.unit_type(unit)
        expected = _unit_type(unit_type)
        known_types = {t for t, _, _ in self._table.values()}
        if actual is not None and expected in known_types and actual != expected:
            raise UnitError(f"{unit!r} measures {actual}, not {unit_type}")


#: The built-in units.
UNITS = Units()


def convert(value: float, from_unit: str, to_unit: str) -> float:
    """*value* in *from_unit*, expressed in *to_unit*, with the built-in units."""
    return UNITS.convert(value, from_unit, to_unit)
