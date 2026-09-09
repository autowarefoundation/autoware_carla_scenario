"""Every place a scenario names, gathered into one drawing of the map.

The arrangement canvas says *what* happens and in what order.  It cannot say
where: a spawn is a lanelet id in a lane head, the goal is another one beside
it, and the lanelet a condition watches is a number on a card -- three numbers
that mean nothing without the map open in another window.  This module answers
"which places does this scenario name", so the editor can draw them above the
timeline on the map it already renders.

Binding one pattern
-------------------
A scenario whose lanelet is left to a constraint search does not name a place at
all: it names a *set*, and the sweeper runs the scenario once per member.
Outlining all 37 matches draws the search rather than the scenario, and no
single one of those runs looks like that picture.  So one member is **bound** --
the same way a sweep binds it, through
:func:`~autoware_carla_scenario.authoring.hydra_config.swept_slot` and the
framework's own constraint engine -- and the panel says which of how many is on
screen.  Stepping the pattern index walks the runs a ``--multirun`` sweep would
perform, one concrete scenario at a time.

Everything else keeps the id stored beside it, because that is exactly what a
run that does not sweep uses: the sweeper enumerates a single target key, so at
most one lanelet in a document is ever chosen for it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..authoring.hydra_config import swept_slot
from ..authoring.models import (
    GoalSpec,
    LaneletSlot,
    ScenarioDocument,
    SpawnSpec,
    condition_refs,
)
from ..authoring.registry import get_action_spec, get_condition_spec
from . import map_preview

__all__ = ["MapPoint", "MapView", "build_map_view"]


@dataclass(frozen=True)
class MapPoint:
    """One lanelet a scenario names, ready to be drawn and listed.

    Attributes:
        key: The :class:`~autoware_carla_scenario.authoring.models.LaneletSlot`
            key this point came from, unique in a document.
        kind: ``spawn``, ``goal`` or ``watched`` -- what the place is *for*,
            which is what decides the glyph it is drawn with.
        owner_id: The object the inspector opens when the point is clicked.
        entity_id: The actor the point belongs to, for the track colour, or
            ``None`` when nothing about the place names an actor.
        title: Who or what names the place ("Ego", "Position (Lanelet2)").
        role: What the place is to them ("spawn", "goal", "lanelet").
        lanelet_id: The lanelet drawn.  ``0`` means nothing is chosen yet.
        detail: Where along the lanelet, when the document says.
        note: How the id was arrived at -- pinned, bound from a search, or the
            default a search falls back to.
        bound: Whether :attr:`lanelet_id` came from binding a search pattern
            rather than from the document.
    """

    key: str
    kind: str
    owner_id: str
    entity_id: Optional[str]
    title: str
    role: str
    lanelet_id: int
    detail: str = ""
    note: str = ""
    bound: bool = False

    @property
    def placed(self) -> bool:
        """Whether this point has a lanelet to draw at all."""
        return self.lanelet_id > 0


@dataclass
class MapView:
    """The scenario as places on its map, with one search pattern bound.

    Attributes:
        points: Every lanelet the document names, in document order.
        swept_key: The slot a sweep enumerates, or ``""`` when nothing is
            searched.
        swept_label: That slot, named to a person.
        pattern: Which match is bound, zero-based.
        pattern_count: How many lanelets the search matched.
        map_loaded: Whether the server has the map parsed, which is what
            binding a pattern needs.
        error: Why no pattern could be bound.
    """

    points: list[MapPoint] = field(default_factory=list)
    swept_key: str = ""
    swept_label: str = ""
    pattern: int = 0
    pattern_count: int = 0
    map_loaded: bool = False
    error: str = ""

    @property
    def highlight_ids(self) -> list[int]:
        """The lanelets the map outlines: every place, each named once.

        The viewer has a single highlight channel, so the outline says "this
        scenario touches here" and the pins say which place is which.
        """
        seen: list[int] = []
        for point in self.points:
            if point.placed and point.lanelet_id not in seen:
                seen.append(point.lanelet_id)
        return seen

    @property
    def searching(self) -> bool:
        """Whether a lanelet in this scenario is left to a search."""
        return bool(self.swept_key)

    @property
    def bound(self) -> bool:
        """Whether a concrete pattern is on screen."""
        return any(point.bound for point in self.points)

    @property
    def next_pattern(self) -> int:
        """The pattern after this one, wrapping at the last."""
        if self.pattern_count < 1:
            return 0
        return (self.pattern + 1) % self.pattern_count

    @property
    def previous_pattern(self) -> int:
        """The pattern before this one, wrapping at the first."""
        if self.pattern_count < 1:
            return 0
        return (self.pattern - 1) % self.pattern_count

    def owners_by_lanelet(self) -> dict[str, str]:
        """Return ``{lanelet id: object id}`` for click-to-select on the map.

        A lanelet named twice -- an ego that watches the lane it starts on --
        opens the first thing that named it, which is document order and so is
        the same object the point list shows first.
        """
        owners: dict[str, str] = {}
        for point in self.points:
            if point.placed:
                owners.setdefault(str(point.lanelet_id), point.owner_id)
        return owners


def _entity_of(document: ScenarioDocument, slot: LaneletSlot) -> Optional[str]:
    """Return the actor a slot belongs to, for the track colour.

    A spawn and a goal are the entity's own.  A lanelet parameter belongs to
    whoever performs the action or is named by the condition, so a place NPC1
    is watched for wears NPC1's colour rather than a colour of its own.
    """
    entity = document.entity(slot.owner_id)
    if entity is not None:
        return entity.id
    action = document.action(slot.owner_id)
    if action is not None:
        return action.actor or None
    condition = document.condition(slot.owner_id)
    if condition is not None:
        refs = condition_refs(condition, "entity")
        return refs[0] if refs else None
    return None


def _naming(document: ScenarioDocument, slot: LaneletSlot) -> tuple[str, str, str]:
    """Return ``(kind, title, role)`` for one slot.

    The title is who names the place and the role is what it is to them, so a
    pin reads "Ego · spawn" -- the same two halves the point list is sorted by.
    """
    entity = document.entity(slot.owner_id)
    if entity is not None and slot.field in ("spawn", "goal"):
        return slot.field, entity.display_name, slot.field
    action = document.action(slot.owner_id)
    if action is not None:
        action_spec = get_action_spec(action.type)
        named = action.title or (action_spec.title if action_spec else action.type)
        return "watched", named, _field_label(action_spec, slot.field)
    condition = document.condition(slot.owner_id)
    if condition is not None:
        condition_spec = get_condition_spec(condition.type)
        watched = condition_spec.title if condition_spec else condition.type
        return "watched", watched, _field_label(condition_spec, slot.field)
    return "watched", slot.label, slot.field


def _field_label(spec: object, name: str) -> str:
    """Return the human label of a spec's field, or the field name."""
    for candidate in getattr(spec, "fields", ()):
        if candidate.name == name:
            return str(candidate.label).lower()
    return name


def _detail(slot: LaneletSlot) -> str:
    """Return where along the lanelet the document puts this point."""
    holder = slot.holder
    if isinstance(holder, SpawnSpec):
        if holder.s.mode == "derived":
            return "offset derived from the map"
        return f"{holder.s.value:g} m along it"
    if isinstance(holder, GoalSpec):
        return f"{holder.s:g} m along it"
    return ""


def build_map_view(
    document: ScenarioDocument, *, pattern: int = 0, load_map: bool = False
) -> MapView:
    """Return every place *document* names, with one search pattern bound.

    Args:
        document: The scenario being edited.
        pattern: Which match of the searched lanelet to bind, wrapping at the
            last so the arrows can be held down.
        load_map: Parse the map when it is not cached yet.  Left off, a map
            nobody has loaded leaves the searched place unbound rather than
            making every edit wait on parsing a city.

    Returns:
        A :class:`MapView`.  A failure to evaluate the search is reported in
        :attr:`MapView.error`, never raised: the places a document *pins* are
        drawable whatever the search does.
    """
    view = MapView()

    swept = swept_slot(document)
    bound_id = 0
    if swept is not None:
        view.swept_key = swept.key
        view.swept_label = swept.label
        result = map_preview.evaluate_slot(document, swept, load_map=load_map)
        view.map_loaded = result.map_loaded
        view.error = result.error
        view.pattern_count = len(result.matched_ids)
        if result.matched_ids:
            view.pattern = pattern % len(result.matched_ids)
            bound_id = result.matched_ids[view.pattern]

    for slot in document.lanelet_slots():
        kind, title, role = _naming(document, slot)
        is_swept = slot.key == view.swept_key
        bound = is_swept and bound_id > 0
        lanelet_id = bound_id if bound else slot.lanelet_id
        if bound:
            note = f"bound: match {view.pattern + 1} of {view.pattern_count}"
        elif slot.searching and is_swept:
            note = "searched, not bound yet"
        elif slot.searching:
            # A second search runs pinned to its default: the sweeper enumerates
            # one target key, which `validator` warns about separately.
            note = "searched, but not the slot the sweep drives"
        elif lanelet_id:
            note = "pinned"
        else:
            note = "not chosen yet"
        view.points.append(
            MapPoint(
                key=slot.key,
                kind=kind,
                owner_id=slot.owner_id,
                entity_id=_entity_of(document, slot),
                title=title,
                role=role,
                lanelet_id=lanelet_id,
                detail=_detail(slot),
                note=note,
                bound=bound,
            )
        )
    return view
