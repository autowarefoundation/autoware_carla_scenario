"""Which road users a condition looks at: an :class:`EntityFilter`.

A condition about one entity names it; a condition about *some* entities --
"any vehicle within 30 m ahead of the ego", "a pedestrian on these
lanelets", "whichever background vehicle is braking at it" -- describes them
instead, with an :class:`EntityFilter`:

* by name: role-name patterns (``npc*``, ``background*``, ``sumo:*``), so
  the TrafficManager's and SUMO's background vehicles, which have no entity
  of their own, can be picked out as well as a scenario's named ones;
* by kind: vehicles, pedestrians, or both;
* relative to a reference entity (the ego by default): the distance between
  their centres, and the offset ahead (negative: behind) and to the left
  (negative: to the right) in the reference's frame, on the ground plane;
* by place: on any of a set of Lanelet2 lanelets.

Every bound is optional; a filter with none takes every named vehicle and
pedestrian in the world but the reference.  :meth:`EntityFilter.select`
answers which match, now, with each one's distance and offsets.  Actors
parked under the map (spawned out of sight, see
:data:`~.entity_existence.HIDDEN_BELOW_M`) and actors without a role name are
never taken.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional, Sequence

from ..coordinate.poses import CarlaWorldPose
from .entity_existence import HIDDEN_BELOW_M

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

__all__ = [
    "EntityFilter",
    "RoadUserKind",
    "SelectedEntity",
    "entity_filter_from_fields",
    "name_matches",
]


class RoadUserKind(Enum):
    """Which road users a filter takes."""

    ANY = "any"
    VEHICLE = "vehicle"
    PEDESTRIAN = "pedestrian"


@dataclass(frozen=True)
class SelectedEntity:
    """One road user a filter took, and where it is from the reference.

    Attributes:
        name: Its role name.
        actor_id: Its CARLA actor id.
        distance: From the reference's centre to its (m); ``nan`` without a
            reference in the world.
        ahead: Its offset along the reference's heading (m, negative behind).
        left: Its offset to the reference's left (m, negative to the right).
    """

    name: str
    actor_id: int
    distance: float
    ahead: float
    left: float


@dataclass(frozen=True)
class EntityFilter:
    """Which road users to take: by name, kind, place, and relative to a
    reference entity.

    Attributes:
        names: Role-name patterns, any of which a name must match (``*``: any
            run of characters, ``?``: any one); empty takes every name.
        kind: Vehicles, pedestrians, or both.
        reference: The entity the relative bounds are measured from.
        min_distance_m / max_distance_m: The distance between the centres.
        min_ahead_m / max_ahead_m: The offset along the reference's heading
            (negative: behind it).
        min_left_m / max_left_m: The offset to the reference's left
            (negative: to its right).
        lanelet_ids: Lanelets one of which it must be on (Lanelet2's own
            "inside" test, so overlapping lanelets in a junction each count);
            empty takes it anywhere.
        include_reference: Take the reference itself, when it otherwise
            matches (its distance and offsets are 0).

    A relative bound needs the reference in the world: without it, a filter
    with one takes nothing.
    """

    names: tuple[str, ...] = ()
    kind: RoadUserKind = RoadUserKind.ANY
    reference: str = "Ego"
    min_distance_m: Optional[float] = None
    max_distance_m: Optional[float] = None
    min_ahead_m: Optional[float] = None
    max_ahead_m: Optional[float] = None
    min_left_m: Optional[float] = None
    max_left_m: Optional[float] = None
    lanelet_ids: tuple[int, ...] = ()
    include_reference: bool = False

    def validate(self) -> None:
        """Refuse bounds that can never hold.

        Raises:
            ValueError: A distance below zero, or a lower bound above its
                upper one.
        """
        if self.min_distance_m is not None and self.min_distance_m < 0.0:
            raise ValueError("min_distance_m must not be negative")
        if self.max_distance_m is not None and self.max_distance_m < 0.0:
            raise ValueError("max_distance_m must not be negative")
        _check_range("distance", self.min_distance_m, self.max_distance_m)
        _check_range("ahead", self.min_ahead_m, self.max_ahead_m)
        _check_range("left", self.min_left_m, self.max_left_m)

    @property
    def relative(self) -> bool:
        """Whether a bound is measured from the reference."""
        return (
            self.min_distance_m is not None
            or self.max_distance_m is not None
            or self.min_ahead_m is not None
            or self.max_ahead_m is not None
            or self.min_left_m is not None
            or self.max_left_m is not None
        )

    def describe(self) -> str:
        """The filter in a few words, for a result message."""
        parts: list[str] = []
        if self.kind is RoadUserKind.VEHICLE:
            parts.append("vehicles")
        elif self.kind is RoadUserKind.PEDESTRIAN:
            parts.append("pedestrians")
        else:
            parts.append("entities")
        if self.names:
            parts.append("named " + " or ".join(self.names))
        bounds: list[str] = []
        _say_range(bounds, "distance", self.min_distance_m, self.max_distance_m)
        _say_range(bounds, "ahead", self.min_ahead_m, self.max_ahead_m)
        _say_range(bounds, "left", self.min_left_m, self.max_left_m)
        if bounds:
            parts.append(f"with {', '.join(bounds)} from '{self.reference}'")
        if self.lanelet_ids:
            ids = ", ".join(str(i) for i in self.lanelet_ids)
            parts.append(f"on lanelet {ids}")
        return " ".join(parts)

    def takes_name(self, name: str) -> bool:
        """Whether a road user called *name* passes the name patterns."""
        if not self.names:
            return True
        for pattern in self.names:
            if name_matches(pattern, name):
                return True
        return False

    def select(self, world: "carla.World") -> list[SelectedEntity]:
        """The road users in *world* that pass the filter now, in the
        world's actor order."""
        actors = list(world.get_actors())
        reference: Optional["carla.Actor"] = None
        for actor in actors:
            if _role_name(actor) == self.reference:
                reference = actor
                break
        if reference is None and self.relative:
            return []
        origin = (0.0, 0.0, 0.0)
        forward = (1.0, 0.0)
        if reference is not None:
            # Annotated, so Codon unwraps the Optional (docs/typecheck.md).
            found: "carla.Actor" = reference
            transform = found.get_transform()
            origin = (
                float(transform.location.x),
                float(transform.location.y),
                float(transform.location.z),
            )
            heading = transform.get_forward_vector()
            fx, fy = float(heading.x), float(heading.y)
            norm = math.hypot(fx, fy)
            if norm < 1e-9:
                if self.min_ahead_m is not None or self.max_ahead_m is not None:
                    return []
                if self.min_left_m is not None or self.max_left_m is not None:
                    return []
            else:
                forward = (fx / norm, fy / norm)
        selected: list[SelectedEntity] = []
        for actor in actors:
            entity = self._take(actor, reference, origin, forward)
            if entity is not None:
                selected.append(entity)
        return selected

    def _take(
        self,
        actor: "carla.Actor",
        reference: Optional["carla.Actor"],
        origin: tuple[float, float, float],
        forward: tuple[float, float],
    ) -> Optional[SelectedEntity]:
        type_id = str(actor.type_id)
        vehicle = type_id.startswith("vehicle.")
        walker = type_id.startswith("walker.")
        if not vehicle and not walker:
            return None
        if self.kind is RoadUserKind.VEHICLE and not vehicle:
            return None
        if self.kind is RoadUserKind.PEDESTRIAN and not walker:
            return None
        name = _role_name(actor)
        if not name:
            return None
        if name == self.reference and not self.include_reference:
            return None
        if not self.takes_name(name):
            return None
        transform = actor.get_transform()
        location = transform.location
        if float(location.z) < HIDDEN_BELOW_M:
            return None
        distance = math.nan
        ahead = math.nan
        left = math.nan
        if reference is not None:
            dx = float(location.x) - origin[0]
            dy = float(location.y) - origin[1]
            dz = float(location.z) - origin[2]
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            ahead = dx * forward[0] + dy * forward[1]
            # CARLA's frame is left-handed: left of (fx, fy) is (fy, -fx).
            left = dx * forward[1] - dy * forward[0]
        if not _within(distance, self.min_distance_m, self.max_distance_m):
            return None
        if not _within(ahead, self.min_ahead_m, self.max_ahead_m):
            return None
        if not _within(left, self.min_left_m, self.max_left_m):
            return None
        if self.lanelet_ids and not _on_any(transform, self.lanelet_ids):
            return None
        return SelectedEntity(name, int(actor.id), distance, ahead, left)


def entity_filter_from_fields(
    names: str = "",
    kind: str = "ANY",
    reference: Optional[str] = None,
    min_distance_m: Optional[float] = None,
    max_distance_m: Optional[float] = None,
    min_ahead_m: Optional[float] = None,
    max_ahead_m: Optional[float] = None,
    min_left_m: Optional[float] = None,
    max_left_m: Optional[float] = None,
    lanelet_ids: Optional[Sequence[int]] = None,
    include_reference: bool = False,
) -> EntityFilter:
    """An :class:`EntityFilter` from a scenario document's fields: *names*
    is one text of patterns, apart by commas or spaces; *kind* a
    :class:`RoadUserKind` by name; no *reference* is the ego.

    Raises:
        ValueError: An unknown *kind*, or bounds that can never hold.
    """
    patterns: list[str] = []
    for part in names.replace(",", " ").split():
        patterns.append(part)
    lanelets: list[int] = []
    if lanelet_ids is not None:
        for lanelet in lanelet_ids:
            lanelets.append(int(lanelet))
    built = EntityFilter(
        names=tuple(patterns),
        kind=_kind_named(kind),
        reference=_reference_or_ego(reference),
        min_distance_m=min_distance_m,
        max_distance_m=max_distance_m,
        min_ahead_m=min_ahead_m,
        max_ahead_m=max_ahead_m,
        min_left_m=min_left_m,
        max_left_m=max_left_m,
        lanelet_ids=tuple(lanelets),
        include_reference=include_reference,
    )
    built.validate()
    return built


def _kind_named(name: str) -> RoadUserKind:
    wanted = str(name).upper()
    for kind in (RoadUserKind.ANY, RoadUserKind.VEHICLE, RoadUserKind.PEDESTRIAN):
        if kind.name == wanted:
            return kind
    raise ValueError(f"no road user kind {name!r}: ANY, VEHICLE or PEDESTRIAN")


def _reference_or_ego(reference: Optional[str]) -> str:
    if reference is None or not str(reference):
        return "Ego"
    return str(reference)


def name_matches(pattern: str, name: str) -> bool:
    """Whether *name* matches *pattern*: ``*`` any run of characters
    (none too), ``?`` any one; everything else itself, case and all."""
    p = 0
    n = 0
    star = -1
    resume = 0
    while n < len(name):
        if p < len(pattern) and (pattern[p] == "?" or pattern[p] == name[n]):
            p += 1
            n += 1
        elif p < len(pattern) and pattern[p] == "*":
            star = p
            resume = n
            p += 1
        elif star >= 0:
            p = star + 1
            resume += 1
            n = resume
        else:
            return False
    while p < len(pattern) and pattern[p] == "*":
        p += 1
    return p == len(pattern)


def _role_name(actor: "carla.Actor") -> str:
    attributes = actor.attributes
    if "role_name" in attributes:
        return str(attributes["role_name"])
    return ""


def _within(value: float, low: Optional[float], high: Optional[float]) -> bool:
    if low is not None and not value >= low:
        return False
    if high is not None and not value <= high:
        return False
    return True


def _check_range(what: str, low: Optional[float], high: Optional[float]) -> None:
    if low is not None and high is not None and low > high:
        raise ValueError(f"min_{what}_m ({low}) is above max_{what}_m ({high})")


def _say_range(
    into: list[str], what: str, low: Optional[float], high: Optional[float]
) -> None:
    if low is not None and high is not None:
        into.append(f"{what} {low:g}..{high:g} m")
    elif low is not None:
        into.append(f"{what} >= {low:g} m")
    elif high is not None:
        into.append(f"{what} <= {high:g} m")


def _on_any(transform: "carla.Transform", lanelet_ids: tuple[int, ...]) -> bool:
    from ..coordinate.transform import on_lanelet  # noqa: PLC0415 - the map layer

    pose = CarlaWorldPose.from_carla_transform(transform)
    for lanelet in lanelet_ids:
        if on_lanelet(pose, lanelet):
            return True
    return False
