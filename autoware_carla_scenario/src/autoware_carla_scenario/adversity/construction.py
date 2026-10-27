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

``layout="urban"`` lays out TeraSim's ``UrbanConstructionAdversity``
(``adversity/static/urban_construction.py``, #445) instead: one approach of
its corridor, from ``start`` (where the edge begins) to ``end`` (the
intersection it ends at), with no taper in, buffer or termination zone:

* **warning** -- from ``start``, ``warning_zone_length`` but no more than 0.3
  of the approach: at most three signs, every ``warning_zone_spacing``;
* **work** -- the warning zone's end to the merge taper's start (with no merge
  taper, to 0.9 of the approach): along the closed side's edge, 0.5 m in,
  every ``spacing``, short of its end; of ``construction_type`` ``"cone"``,
  every fifth object is a barrier (TeraSim adds the barrier beside the cone);
* **merge_signs** -- ``merge_signs`` signs, every 50 m before the merge taper;
* **merge_taper** -- the last ``merge_distance`` before ``end``: cones every
  half ``spacing``, from the closed side's edge across to the other one, so
  the lane is shut before the intersection (TeraSim's ``merge_before``).

The warning zone and the merge taper step up to their end inclusive, as
TeraSim's do.  :data:`URBAN_CONSTRUCTION` holds TeraSim's urban numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "DEFAULT_CONSTRUCTION",
    "HIGHWAY",
    "URBAN",
    "URBAN_CONSTRUCTION",
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

#: The layouts: TeraSim's ``ConstructionAdversity`` and its
#: ``UrbanConstructionAdversity``.
HIGHWAY = "highway"
URBAN = "urban"

#: How far inside the lane's edge (m) the cones stand.
_EDGE_MARGIN_M = 0.3
#: How far inside the lane's edge (m) the urban work zone's objects stand.
_URBAN_WORK_MARGIN_M = 0.5
#: The urban warning zone: at most this share of the approach, and this many
#: signs.
_URBAN_WARNING_SHARE = 0.3
_URBAN_WARNING_SIGNS = 3
#: With no merge taper, the urban work zone ends this far along the approach.
_URBAN_WORK_SHARE = 0.9
#: Every this many objects of the urban work zone, a barrier.
_URBAN_BARRIER_EVERY = 5
#: The merge taper's cones are this share of ``spacing`` apart.
_URBAN_TAPER_SHARE = 0.5
#: Between the merge signs (m).
_MERGE_SIGN_SPACING_M = 50.0
#: Slack on the inclusive end of an urban zone, for the steps' rounding.
_EPSILON_M = 1e-9
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
        layout: :data:`HIGHWAY` (``ConstructionAdversity``) or :data:`URBAN`
            (``UrbanConstructionAdversity``).  An urban layout reads
            ``construction_type``, ``spacing``, ``taper_type``, the warning
            zone's numbers, ``side`` and the two below; it has no taper in,
            buffer or termination zone and no work zone offset.
        merge_distance: Urban: the merge taper's length (m) before the end,
            where the lane is shut ahead of the intersection; 0 for none.
        merge_signs: Urban: how many signs stand before the merge taper,
            50 m apart.
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
    layout: str = HIGHWAY
    merge_distance: float = 150.0
    merge_signs: int = 2


DEFAULT_CONSTRUCTION = ConstructionSettings()

#: TeraSim's ``UrbanConstructionAdversity`` numbers: work-zone cones every
#: 10 m, merge-taper cones every 5 m, signs 2 m out on the shoulder, a warning
#: zone of up to 200 m (its Ann Arbor example) and a 150 m merge taper.
URBAN_CONSTRUCTION = ConstructionSettings(
    spacing=10.0,
    warning_zone_length=200.0,
    warning_sign_offset=2.0,
    layout=URBAN,
)


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
    if settings.layout == URBAN:
        return _urban_zones(start, end, settings)
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


def _merge_start(start: float, end: float, settings: ConstructionSettings) -> float:
    """Where the urban merge taper starts: ``merge_distance`` before *end*,
    not before *start*."""
    return max(start, end - settings.merge_distance)


def _urban_zones(
    start: float, end: float, settings: ConstructionSettings
) -> list[tuple[str, float, float]]:
    """:func:`construction_zones` of an urban layout."""
    zones: list[tuple[str, float, float]] = []
    warning_end = start + min(
        max(settings.warning_zone_length, 0.0), _URBAN_WARNING_SHARE * (end - start)
    )
    if warning_end > start:
        zones.append(("warning", start, warning_end))
    merging = settings.merge_distance > 0.0
    work_end = start + _URBAN_WORK_SHARE * (end - start)
    if merging:
        work_end = _merge_start(start, end, settings)
    if work_end > warning_end:
        zones.append(("work", warning_end, work_end))
    if merging:
        merge_start = _merge_start(start, end, settings)
        if settings.merge_signs > 0:
            first = merge_start - settings.merge_signs * _MERGE_SIGN_SPACING_M
            zones.append(("merge_signs", max(start, first), merge_start))
        zones.append(("merge_taper", merge_start, end))
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
    if zone in ("warning", "termination", "merge_signs"):
        return [SIGN]
    if zone == "merge_taper":
        return [CONE]
    if settings.layout == URBAN and settings.construction_type == CONE:
        # TeraSim's: a barrier every fifth object, from the first.
        kinds = [CONE] * _URBAN_BARRIER_EVERY
        kinds[0] = BARRIER
        return kinds
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
    progress = _shaped(progress, settings)
    if zone == "taper_in":
        return edge + progress * (work - edge)
    return work + progress * (edge - work)


def _shaped(progress: float, settings: ConstructionSettings) -> float:
    """*progress* through a taper, on an s-curve if it is ``curved``."""
    if settings.taper_type == "curved":
        return 3.0 * progress * progress - 2.0 * progress * progress * progress
    return progress


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
        ValueError: *end* not after *start*, a spacing that is not
            positive, or a layout that is neither :data:`HIGHWAY` nor
            :data:`URBAN`.
    """
    if not end > start:
        raise ValueError(f"a construction zone ends after it starts, not at {end}")
    sign = 1.0 if settings.side == "left" else -1.0
    half = lane_width / 2.0 - _EDGE_MARGIN_M
    shoulder = (lane_width / 2.0 + abs(settings.warning_sign_offset)) * sign
    if settings.layout == URBAN:
        return _urban_layout(start, end, lane_width, sign, shoulder, settings)
    if settings.layout != HIGHWAY:
        raise ValueError(f"no construction layout '{settings.layout}'")
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


def _urban_layout(
    start: float,
    end: float,
    lane_width: float,
    sign: float,
    shoulder: float,
    settings: ConstructionSettings,
) -> list[ConstructionObject]:
    """:func:`construction_layout` of an urban layout: TeraSim's
    ``UrbanConstructionAdversity`` on one approach."""
    edge = max(lane_width / 2.0 - _EDGE_MARGIN_M, 0.0) * sign
    curb = max(lane_width / 2.0 - _URBAN_WORK_MARGIN_M, 0.0) * sign
    objects: list[ConstructionObject] = []
    for zone, zone_start, zone_end in construction_zones(start, end, settings):
        if zone == "merge_signs":
            # Nearest the taper first, 50 m apart, none before the start.
            placed: list[float] = []
            for index in range(settings.merge_signs):
                s = max(start, zone_end - (index + 1) * _MERGE_SIGN_SPACING_M)
                if s not in placed:
                    placed.append(s)
                    objects.append(ConstructionObject(SIGN, zone, s, shoulder))
            continue
        if zone == "warning":
            step = settings.warning_zone_spacing
        elif zone == "work":
            step = settings.spacing
        else:
            step = settings.spacing * _URBAN_TAPER_SHARE
        if not step > 0.0:
            raise ValueError(f"the {zone} zone's spacing is {step}, not positive")
        if zone == "merge_taper":
            # TeraSim's: cones at both ends, from the closed side's edge
            # across the lane to the other's.
            count = int((zone_end - zone_start) / step + _EPSILON_M)
            for index in range(count + 1):
                s = min(zone_start + index * step, zone_end)
                progress = _shaped(index / max(count, 1), settings)
                objects.append(
                    ConstructionObject(CONE, zone, s, edge - 2.0 * edge * progress)
                )
            continue
        kinds = _kinds(zone, settings)
        # The warning zone up to its end, as TeraSim's; the work zone short
        # of it, which is where the merge taper's first cone stands.
        last = zone_end + _EPSILON_M if zone == "warning" else zone_end - _EPSILON_M
        index = 0
        s = zone_start
        while s <= last:
            if zone == "warning":
                if index >= _URBAN_WARNING_SIGNS:
                    break
                objects.append(ConstructionObject(SIGN, zone, s, shoulder))
            else:
                kind = kinds[index % len(kinds)]
                objects.append(ConstructionObject(kind, zone, s, curb))
            index += 1
            s = zone_start + index * step
    return objects
