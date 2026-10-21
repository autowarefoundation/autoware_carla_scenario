"""Where a construction zone's cones, barriers and signs go (#402).

TeraSim's ``ConstructionAdversity`` (``adversity/static/construction.py``),
its layout only: given where along a lane the work starts and ends, the zones
of a MUTCD work zone and the objects of each, as ``(s, t)`` along the lane --
``s`` from where the lane is measured, ``t`` off its centreline, positive to
the left.  :class:`~..actions.construction_zone.ConstructionZoneAction` puts
them in the world.

The zones, from ``start`` to ``end``:

* **warning** -- ``[start - warning_zone_length, start]``: signs on the
  shoulder, every ``warning_zone_spacing``;
* **taper_in** -- ``[start, start + taper_length_in]``: cones from the lane's
  edge (0.3 m in from it) to the work zone's offset, linearly or on an s-curve
  (``3p² - 2p³``);
* **buffer** -- ``taper_in``'s end, ``buffer_zone_length`` on: cone, cone,
  barrier, at the work zone's offset;
* **work** -- the buffer's end to ``end - taper_length_out``: the
  ``construction_type`` (cone, barrier, or for ``mixed`` cone, cone, barrier);
* **taper_out** -- ``[end - taper_length_out, end]``: cones back to the edge;
* **termination** -- ``[end, end + termination_zone_length]``: signs.

Objects step from a zone's start while short of its end, by the zone's
spacing: ``spacing`` (0.7 of it in a taper, 0.8 in the buffer), or with
``use_dynamic_spacing`` and a ``speed_limit_mph`` MUTCD's -- the speed in feet
for a taper, twice it for the buffer and the work zone, at least 30 m or three
times it for warning and termination.  The offsets stay 0.3 m inside the lane.

Where TeraSim picks the closed side from the SUMO lane's index (``> 1`` is
the left), the side here is given: ``"right"`` (default) or ``"left"``.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "DEFAULT_CONSTRUCTION",
    "BARRIER",
    "CONE",
    "SIGN",
    "ConstructionObject",
    "ConstructionSettings",
    "construction_layout",
    "construction_zones",
]

#: The kinds of object a zone is made of.
CONE = "cone"
BARRIER = "barrier"
SIGN = "sign"

#: How far inside the lane's edge (m) the cones stand.
_EDGE_MARGIN_M = 0.3
#: One foot (m): MUTCD's spacings are the speed limit in feet.
_FOOT_M = 0.3048


@dataclass(frozen=True)
class ConstructionSettings:
    """TeraSim's construction-zone numbers, as the defaults.

    Attributes:
        construction_type: The work zone's objects: ``"cone"``,
            ``"barrier"`` or ``"mixed"`` (cone, cone, barrier).
        spacing: Between objects (m), before a zone's factor.
        work_zone_offset: The work zone's objects' offset off the centreline
            (m, positive to the left), kept 0.3 m inside the lane.
        speed_limit_mph: The road's speed limit, for ``use_dynamic_spacing``.
        use_dynamic_spacing: Space by MUTCD's speed-based rules instead.
        taper_length_in: The entry taper's length (m).
        taper_length_out: The exit taper's length (m).
        taper_type: ``"linear"`` or ``"curved"`` (an s-curve).
        warning_zone_length: Before the start (m).
        warning_zone_spacing: Between the warning signs (m).
        buffer_zone_length: After the entry taper (m).
        termination_zone_length: After the end (m).
        warning_sign_offset: How far (m) out on the shoulder, past the lane's
            edge, the signs stand.
        side: Which side of the lane the zone closes in from: ``"right"`` or
            ``"left"``.
    """

    construction_type: str = CONE
    spacing: float = 20.0
    work_zone_offset: float = 0.0
    speed_limit_mph: float = 0.0
    use_dynamic_spacing: bool = False
    taper_length_in: float = 60.0
    taper_length_out: float = 30.0
    taper_type: str = "linear"
    warning_zone_length: float = 100.0
    warning_zone_spacing: float = 30.0
    buffer_zone_length: float = 10.0
    termination_zone_length: float = 30.0
    warning_sign_offset: float = 2.5
    side: str = "right"


DEFAULT_CONSTRUCTION = ConstructionSettings()


@dataclass(frozen=True)
class ConstructionObject:
    """One cone, barrier or sign of a zone.

    Attributes:
        kind: :data:`CONE`, :data:`BARRIER` or :data:`SIGN`.
        zone: The zone it belongs to (``"warning"``, ``"taper_in"``, ...).
        s: Along the lane (m), from where it is measured.
        t: Off the centreline (m), positive to the left.
    """

    kind: str
    zone: str
    s: float
    t: float


def construction_zones(
    start: float, end: float, settings: ConstructionSettings = DEFAULT_CONSTRUCTION
) -> list[tuple[str, float, float]]:
    """The zones from *start* to *end* (m along the lane), as ``(zone,
    from, to)``, in order; a zone of no length is left out."""
    zones: list[tuple[str, float, float]] = []
    if settings.warning_zone_length > 0.0:
        zones.append(("warning", max(0.0, start - settings.warning_zone_length), start))
    if settings.taper_length_in > 0.0:
        zones.append(("taper_in", start, start + settings.taper_length_in))
    buffer_start = start + settings.taper_length_in
    if settings.buffer_zone_length > 0.0:
        zones.append(
            ("buffer", buffer_start, buffer_start + settings.buffer_zone_length)
        )
    work_start = buffer_start + settings.buffer_zone_length
    work_end = end - settings.taper_length_out
    if work_end > work_start:
        zones.append(("work", work_start, work_end))
    if settings.taper_length_out > 0.0:
        zones.append(("taper_out", end - settings.taper_length_out, end))
    if settings.termination_zone_length > 0.0:
        zones.append(("termination", end, end + settings.termination_zone_length))
    return zones


def _spacing(zone: str, settings: ConstructionSettings) -> float:
    """The spacing (m) between a zone's objects."""
    if not settings.use_dynamic_spacing or settings.speed_limit_mph <= 0.0:
        if zone == "warning":
            return settings.warning_zone_spacing
        if zone in ("taper_in", "taper_out"):
            return settings.spacing * 0.7
        if zone == "buffer":
            return settings.spacing * 0.8
        return settings.spacing
    feet = settings.speed_limit_mph * _FOOT_M
    if zone in ("taper_in", "taper_out"):
        return feet
    if zone in ("work", "buffer"):
        return 2.0 * feet
    return max(30.0, 3.0 * feet)  # the warning and termination zones


def _kinds(zone: str, settings: ConstructionSettings) -> list[str]:
    """The objects a zone is made of, in turn."""
    if zone in ("warning", "termination"):
        return [SIGN]
    if zone in ("taper_in", "taper_out"):
        return [CONE]
    if zone == "buffer" or settings.construction_type == "mixed":
        return [CONE, CONE, BARRIER]
    return [settings.construction_type]


def _offset(
    zone: str,
    s: float,
    zone_start: float,
    zone_end: float,
    half: float,
    sign: float,
    settings: ConstructionSettings,
) -> float:
    """A cone's or barrier's offset (m) off the centreline: inside the lane,
    at most *half* off it, on the closed side (*sign*)."""
    edge = half * sign
    work = min(max(settings.work_zone_offset, -half), half)
    if zone in ("buffer", "work"):
        return work
    length = zone_end - zone_start
    progress = 0.0 if length <= 0.0 else (s - zone_start) / length
    if settings.taper_type == "curved":
        progress = 3.0 * progress * progress - 2.0 * progress * progress * progress
    if zone == "taper_in":
        return edge + progress * (work - edge)
    return work + progress * (edge - work)


def construction_layout(
    start: float,
    end: float,
    lane_width: float,
    settings: ConstructionSettings = DEFAULT_CONSTRUCTION,
) -> list[ConstructionObject]:
    """Every object of a zone from *start* to *end* (m along a lane
    *lane_width* wide), zone by zone.

    A sign stands on the shoulder of the closed side, ``warning_sign_offset``
    past the lane's edge; everything else inside the lane.

    Raises:
        ValueError: *end* not after *start*, or a spacing that is not
            positive.
    """
    if not end > start:
        raise ValueError(f"a construction zone ends after it starts, not at {end}")
    sign = 1.0 if settings.side == "left" else -1.0
    half = lane_width / 2.0 - _EDGE_MARGIN_M
    shoulder = (lane_width / 2.0 + abs(settings.warning_sign_offset)) * sign
    objects: list[ConstructionObject] = []
    for zone, zone_start, zone_end in construction_zones(start, end, settings):
        step = _spacing(zone, settings)
        if not step > 0.0:
            raise ValueError(f"the {zone} zone's spacing is {step}, not positive")
        kinds = _kinds(zone, settings)
        s = zone_start
        index = 0
        while s < zone_end:
            kind = kinds[index % len(kinds)]
            t = shoulder
            if kind != SIGN:
                t = _offset(zone, s, zone_start, zone_end, half, sign, settings)
            objects.append(ConstructionObject(kind, zone, s, t))
            s += step
            index += 1
    return objects
