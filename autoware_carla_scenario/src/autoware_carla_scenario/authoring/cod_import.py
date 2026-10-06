"""Build Scenario IR documents from CodSceneClassifier output.

`CodSceneClassifier <https://github.com/tier4/CodSceneClassifier>`_ labels an
Autoware drive log: it splits the recording into scenes and gives each one the
ego's manoeuvre, the signal it faced and up to five road users, each described
by categories (``leading``, ``to_the_left_of_the_subject_vehicle(s)``,
``cutting_in``) and one relative position and speed sample in the ego's
``base_link`` frame.

What it does **not** carry is anything that pins a scene to a map: no lanelet
or lane ID, no map name, no route, no trajectories.  A scene is therefore
imported as a *logical* scenario rather than a replay:

* the ego's spawn is a **constraint search** for the kind of place the scene
  happened in -- a left-turn lanelet, a lane with a neighbour to change into,
  a traffic-light stop line -- so the scenario expands on whatever map it is
  run on, which is the only option anyway (the logs come from maps CARLA
  does not have);
* every other road user is **derived** from that pick with the
  ``route_offset`` / ``route_offset_s`` bindings, at the distance and in the
  lane the classifier recorded relative to the ego;
* the categories become actions (``lane_change``, ``turn``, ``set_speed``,
  ``walk_straight``, ``traffic_signal``) and the ego's decision becomes the
  PASS assertion.

Everything that cannot be expressed is reported on the :class:`ImportedScene`
as a note rather than dropped silently, so a reviewer sees what the document
is an approximation of.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping, Optional, Sequence

from .models import (
    ActionNode,
    Assertions,
    BindingRef,
    ConditionNode,
    ConstraintNode,
    EgoDriver,
    Entity,
    GoalSpec,
    LaneletChoice,
    MapRef,
    ScenarioDocument,
    SpawnSpec,
    SValue,
)
from .persistence import save_document
from .registry import default_params, get_action_spec, get_condition_spec

__all__ = [
    "CodImportOptions",
    "ImportedScene",
    "import_cod_file",
    "import_cod_output",
    "main",
    "scene_to_document",
]

#: The ego's own place in the pick, by what the scene is about.
Anchor = Literal["turn", "lane_change", "red_light", "lane"]

#: Ego decisions with no IR equivalent: imported as lane following.
_UNSUPPORTED_EGO_LATERAL = frozenset(
    {
        "in_lane_avoidance",
        "out_of_lane_avoidance",
        "pulling_over",
        "pulling_out",
        "emergency_braking",
    }
)

#: Pedestrian contexts in which the walker is moving.
_WALKING_CONTEXTS = frozenset(
    {
        "crossing_crosswalk",
        "crossing_road",
        "approaching_crosswalk",
        "walking_in_road",
        "walking_on_sidewalk",
    }
)
#: Pedestrian contexts that face across the road rather than along it.
_CROSSING_CONTEXTS = frozenset(
    {
        "crossing_crosswalk",
        "crossing_road",
        "approaching_crosswalk",
        "standing_near_crosswalk",
    }
)

_DEFAULT_MAP = MapRef(
    group="nishishinjuku",
    name="NishishinjukuMap",
    xodr_path="data/nishishinjuku_carla.xodr",
    lanelet2_path="data/nishishinjuku.osm",
)


@dataclass(frozen=True)
class CodImportOptions:
    """Choices the classifier output cannot make for the importer.

    Attributes:
        driven_by: Who drives the ego.
        ego_speed_kmh: The ego's initial speed when the scene starts on the
            move (the classifier records the ego's decision, not its speed).
        approach_m: How far before a junction or a traffic-light stop line the
            ego starts.
        lane_start_m: Where along a plain lane the ego starts.
        speed_change_kmh: How much an NPC labelled accelerating or decelerating
            changes its speed by.
        max_pedestrians_per_group: Cap on walkers spawned for a collapsed group.
        include_blacklist: Also import blacklisted scenes.
        map: The map the documents run on.
    """

    driven_by: EgoDriver = "autoware"
    ego_speed_kmh: float = 30.0
    approach_m: float = 25.0
    lane_start_m: float = 5.0
    speed_change_kmh: float = 15.0
    max_pedestrians_per_group: int = 3
    include_blacklist: bool = False
    map: MapRef = field(default_factory=lambda: _DEFAULT_MAP.model_copy(deep=True))


@dataclass
class ImportedScene:
    """One classifier scene and the document built from it.

    Attributes:
        document: The scenario.
        notes: What the document approximates or leaves out.
        session: The classifier's session key (``HH-MM-SS``).
        scene_id: The classifier's scene id (unique only within its list).
        blacklisted: Whether the scene came from ``blacklist_scenes``.
    """

    document: ScenarioDocument
    notes: list[str]
    session: str
    scene_id: str
    blacklisted: bool = False


# ---------------------------------------------------------------------------
# Node helpers
# ---------------------------------------------------------------------------


def _condition(type_id: str, **params: Any) -> ConditionNode:
    """Return a condition node seeded with its spec defaults plus *params*."""
    spec = get_condition_spec(type_id)
    assert spec is not None  # noqa: S101 -- built-in types are always registered
    merged = default_params(spec.fields)
    merged.update(params)
    return ConditionNode(type=type_id, params=merged)


def _action(
    type_id: str, title: str, actor: Optional[str] = None, **params: Any
) -> ActionNode:
    """Return an action node seeded with its spec defaults plus *params*."""
    spec = get_action_spec(type_id)
    assert spec is not None  # noqa: S101 -- built-in types are always registered
    merged = default_params(spec.fields)
    merged.update(params)
    return ActionNode(
        type=type_id, title=title, actor=actor, params=merged, phase=spec.default_phase
    )


def _derived(binding: str, **params: Any) -> LaneletChoice:
    """A lanelet derived from the sweep's pick."""
    return LaneletChoice(
        mode="derived", binding=BindingRef(type=binding, params=params)
    )


def _not_excluded() -> ConstraintNode:
    """Keep the search off lanelets the map has no 3-D model for."""
    return ConstraintNode(
        type="not",
        constraints=[
            ConstraintNode(
                type="in_set", params={"values": "${map.no_3d_model_lanelet_ids}"}
            )
        ],
    )


def _all(*constraints: ConstraintNode) -> list[ConstraintNode]:
    return [ConstraintNode(type="and", constraints=list(constraints))]


# ---------------------------------------------------------------------------
# Reading the classifier's output
# ---------------------------------------------------------------------------


def _signal(scene: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    """The scene's traffic signal element, if one was recorded."""
    try:
        element = scene["scenery"]["scenery_elements"][0]
        signal = element["road_infrastructure"]["road_auxiliary_objects"]["signal"]
    except (KeyError, IndexError, TypeError):
        return None
    return signal if isinstance(signal, Mapping) else None


def _geometry(scene: Mapping[str, Any]) -> Mapping[str, Any]:
    """The scene's ``drivable_area_geometry`` element, or an empty mapping."""
    try:
        geometry = scene["scenery"]["scenery_elements"][0]["drivable_area_geometry"]
    except (KeyError, IndexError, TypeError):
        return {}
    return geometry if isinstance(geometry, Mapping) else {}


def _duration_seconds(scene: Mapping[str, Any]) -> float:
    """The scene's length, from its nanosecond timestamps."""
    try:
        return max(0.0, (int(scene["end_time"]) - int(scene["start_time"])) / 1e9)
    except (KeyError, ValueError, TypeError):
        return 0.0


def _snake(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


# ---------------------------------------------------------------------------
# The ego
# ---------------------------------------------------------------------------


@dataclass
class _EgoPlan:
    """What the ego's decision turns into."""

    anchor: Anchor
    spawn: SpawnSpec
    goal: GoalSpec
    #: Where the ego stands relative to the pick's start, along the road.
    base_distance: float
    pass_conditions: list[ConditionNode]
    initial_speed_kmh: float


def _plan_ego(
    scene: Mapping[str, Any],
    duration: float,
    options: CodImportOptions,
    notes: list[str],
) -> _EgoPlan:
    decisions = scene.get("driving_decisions") or {}
    lateral = str(decisions.get("lateral") or "follow_lane")
    longitudinal = str(decisions.get("longitudinal") or "")
    events = set(scene.get("event") or ())
    signal = _signal(scene)
    colour = str(signal.get("signal_color") or "") if signal else ""

    stopping = longitudinal in ("standing_still", "driving_forward_decelerating")
    starting = longitudinal == "standing_still" or "start_from_stop" in events
    speed = 0.0 if starting else options.ego_speed_kmh

    if lateral in ("turn_left", "turn_right"):
        direction = lateral.removeprefix("turn_")
        constraints = [
            ConstraintNode(type="turn_direction", params={"value": direction}),
            ConstraintNode(type="has_stop_line"),
            _not_excluded(),
        ]
        if colour:
            constraints[1] = ConstraintNode(type="has_traffic_light_stop_line")
        return _EgoPlan(
            anchor="turn",
            spawn=SpawnSpec(
                mode="constraint_search",
                lanelet_id=183,
                s=SValue(
                    mode="derived",
                    value=0.0,
                    binding=BindingRef(
                        type="stop_line_offset", params={"offset": options.approach_m}
                    ),
                ),
                constraints=_all(*constraints),
            ),
            goal=GoalSpec(
                lanelet_id=141,
                s=10.0,
                mode="derived",
                binding=BindingRef(type="route_through", params={"depth": 1}),
            ),
            base_distance=-options.approach_m,
            pass_conditions=[_reached_pick()],
            initial_speed_kmh=speed,
        )

    if lateral in ("change_lane_left", "change_lane_right"):
        side = lateral.removeprefix("change_lane_")
        target = _derived("adjacent", side=side)
        reached = _condition("entity_lane_position", entity="ego", lanelet_id=183)
        reached.searches["lanelet_id"] = target.model_copy(deep=True)
        return _EgoPlan(
            anchor="lane_change",
            spawn=SpawnSpec(
                mode="constraint_search",
                lanelet_id=183,
                s=SValue(value=options.lane_start_m),
                constraints=_all(
                    ConstraintNode(type="has_adjacent", params={"value": side}),
                    ConstraintNode(
                        type="lanelet_length",
                        params={"rule": "greater_than_or_equal", "value": 60.0},
                    ),
                    ConstraintNode(
                        type="not", constraints=[ConstraintNode(type="is_junction")]
                    ),
                    _not_excluded(),
                ),
            ),
            goal=GoalSpec(
                lanelet_id=182,
                s=50.0,
                mode="derived",
                binding=BindingRef(type="adjacent", params={"side": side}),
            ),
            base_distance=options.lane_start_m,
            pass_conditions=[ConditionNode(type="sticky", children=[reached])],
            initial_speed_kmh=speed,
        )

    if lateral in _UNSUPPORTED_EGO_LATERAL:
        notes.append(
            f"ego decision {lateral!r} has no IR equivalent; imported as lane "
            "following."
        )
    elif lateral == "branching":
        notes.append(
            "ego decision 'branching' needs a fork constraint the sweeper does "
            "not have; imported as lane following."
        )

    if colour == "red" and stopping:
        stop = _condition("standstill", entity="ego", duration=2.0)
        return _EgoPlan(
            anchor="red_light",
            spawn=SpawnSpec(
                mode="constraint_search",
                lanelet_id=183,
                s=SValue(
                    mode="derived",
                    value=0.0,
                    binding=BindingRef(
                        type="stop_line_offset", params={"offset": options.approach_m}
                    ),
                ),
                constraints=_all(
                    ConstraintNode(type="has_traffic_light_stop_line"),
                    _not_excluded(),
                ),
            ),
            goal=GoalSpec(
                lanelet_id=141,
                s=10.0,
                mode="derived",
                binding=BindingRef(type="route_through", params={"depth": 2}),
            ),
            base_distance=-options.approach_m,
            pass_conditions=[ConditionNode(type="sticky", children=[stop])],
            initial_speed_kmh=speed,
        )

    return _EgoPlan(
        anchor="lane",
        spawn=SpawnSpec(
            mode="constraint_search",
            lanelet_id=183,
            s=SValue(value=options.lane_start_m),
            constraints=_all(
                ConstraintNode(
                    type="lanelet_length",
                    params={"rule": "greater_than_or_equal", "value": 50.0},
                ),
                ConstraintNode(
                    type="not", constraints=[ConstraintNode(type="is_junction")]
                ),
                _not_excluded(),
            ),
        ),
        goal=GoalSpec(
            lanelet_id=141,
            s=10.0,
            mode="derived",
            binding=BindingRef(type="route_through", params={"depth": 1}),
        ),
        base_distance=options.lane_start_m,
        pass_conditions=[
            _condition(
                "elapsed_time",
                rule="greater_than_or_equal",
                duration_seconds=round(max(duration, 5.0), 1),
            )
        ],
        initial_speed_kmh=speed,
    )


def _reached_pick() -> ConditionNode:
    """Sticky "the ego has been on the searched lanelet"."""
    on_pick = _condition("entity_lane_position", entity="ego", lanelet_id=183)
    on_pick.searches["lanelet_id"] = _derived("matched")
    return ConditionNode(type="sticky", children=[on_pick])


# ---------------------------------------------------------------------------
# Other road users
# ---------------------------------------------------------------------------

_LONGITUDINAL_FALLBACK_M = {
    "in_front_of_the_subject_vehicle(s)": 20.0,
    "behind_the_subject_vehicle(s)": -15.0,
    "beside_the_subject_vehicle(s)": 0.0,
}


def _vehicle_side(state: Mapping[str, Any]) -> Optional[str]:
    """The ``route_offset`` side of a vehicle, or ``None`` when crossing."""
    direction = str(state.get("initial_direction") or "")
    if direction == "oncoming":
        return "opposite"
    if direction.startswith("crossing"):
        return None
    lateral = str(state.get("initial_lateral_position") or "")
    if lateral.startswith("to_the_left"):
        return "left"
    if lateral.startswith("to_the_right"):
        return "right"
    return "same"


def _relative_distance(raw: Mapping[str, Any]) -> float:
    """The road user's longitudinal offset from the ego, in metres."""
    position = raw.get("position")
    if isinstance(position, Mapping) and position.get("x") is not None:
        return float(position["x"])
    state = raw.get("state_or_initial_state") or {}
    return _LONGITUDINAL_FALLBACK_M.get(
        str(state.get("initial_longitudinal_position") or ""), 10.0
    )


def _relative_spawn(
    distance: float, side: str, *, t: float = 0.0, heading: float = 0.0
) -> SpawnSpec:
    params = {"distance": round(distance, 2), "side": side}
    return SpawnSpec(
        mode="derived",
        lanelet_id=183,
        binding=BindingRef(type="route_offset", params=dict(params)),
        s=SValue(
            mode="derived",
            value=0.0,
            binding=BindingRef(type="route_offset_s", params=dict(params)),
        ),
        t=t,
        heading=heading,
    )


def _add_vehicle(
    raw: Mapping[str, Any],
    entity_id: str,
    base_distance: float,
    options: CodImportOptions,
    entities: list[Entity],
    actions: list[ActionNode],
    notes: list[str],
) -> None:
    state = raw.get("state_or_initial_state") or {}
    behavior = raw.get("behavior") or {}
    side = _vehicle_side(state)
    if side is None:
        notes.append(
            f"{entity_id}: a vehicle crossing the ego's path needs a crossing-lane "
            "binding the sweeper does not have; left out."
        )
        return
    if raw.get("type") != "vehicle":
        notes.append(f"{entity_id}: {raw.get('type')} spawned as a car.")

    speed_kmh = round(float(raw.get("velocity") or 0.0) * 3.6, 1)
    entity = Entity(
        id=entity_id,
        kind="vehicle",
        title=_title(raw),
        initial_speed_kmh=speed_kmh,
        spawn=_relative_spawn(base_distance + _relative_distance(raw), side),
    )
    entities.append(entity)

    longitudinal = str(behavior.get("longitudinal_action") or "")
    target: Optional[float] = None
    if longitudinal == "standing_still":
        entity.initial_speed_kmh = 0.0
        target = 0.0
    elif longitudinal == "driving_forward_accelerating":
        target = speed_kmh + options.speed_change_kmh
    elif longitudinal == "driving_forward_decelerating":
        target = max(0.0, speed_kmh - options.speed_change_kmh)
    elif longitudinal == "driving_forward_keeping_speed":
        target = speed_kmh
    if target is not None:
        actions.append(
            _action(
                "set_speed",
                f"{entity_id} {longitudinal.replace('_', ' ')}",
                actor=entity_id,
                target_speed_kmh=round(target, 1),
                rate_kmh_s=None if target == speed_kmh else 5.0,
            )
        )

    lateral = str(behavior.get("lateral_action") or "")
    change: Optional[str] = None
    if lateral in ("change_lane_left", "change_lane_right"):
        change = lateral.removeprefix("change_lane_")
    elif raw.get("context") == "cutting_in" and side in ("left", "right"):
        change = "right" if side == "left" else "left"
    if change is not None:
        lane_change = _action(
            "lane_change",
            f"{entity_id} changes lane {change}",
            actor=entity_id,
            direction=change,
        )
        lane_change.trigger = _condition(
            "elapsed_time", rule="greater_than_or_equal", duration_seconds=1.0
        )
        actions.append(lane_change)
        notes.append(
            f"{entity_id}: when the lane change starts is not recorded; it fires "
            "1 s in."
        )
    elif lateral in ("turn_left", "turn_right"):
        turn = _action(
            "turn",
            f"{entity_id} turns {lateral.removeprefix('turn_')}",
            actor=entity_id,
            direction=lateral.removeprefix("turn_"),
        )
        # From the start: the turn action looks ahead for the junction itself.
        turn.trigger = _condition(
            "elapsed_time", rule="greater_than_or_equal", duration_seconds=0.0
        )
        actions.append(turn)
    elif lateral == "swerving":
        notes.append(f"{entity_id}: swerving has no IR action; left out.")


def _add_pedestrians(
    raw: Mapping[str, Any],
    entity_id: str,
    base_distance: float,
    options: CodImportOptions,
    entities: list[Entity],
    actions: list[ActionNode],
    notes: list[str],
) -> None:
    context = str(raw.get("context") or "")
    position = raw.get("position") if isinstance(raw.get("position"), Mapping) else {}
    state = raw.get("state_or_initial_state") or {}
    lateral = str(state.get("initial_lateral_position") or "")
    y = position.get("y") if position else None
    if y is None:
        y = (
            4.0
            if lateral.startswith("left")
            else -4.0
            if lateral.startswith("right")
            else 0.0
        )
    y = max(-8.0, min(8.0, float(y)))

    if context in _CROSSING_CONTEXTS or context.endswith("_in_road"):
        # Face across the road, towards the side it is not on.
        heading = -math.pi / 2 if y > 0 else math.pi / 2
    elif str(state.get("initial_direction") or "").startswith("oncoming"):
        heading = math.pi
    else:
        heading = 0.0
    walking = context in _WALKING_CONTEXTS

    count = int(raw.get("pedestrian_count") or 1)
    spawned = min(count, max(1, options.max_pedestrians_per_group))
    if spawned < count:
        notes.append(f"{entity_id}: a group of {count}, {spawned} spawned.")
    if context in (
        "crossing_crosswalk",
        "approaching_crosswalk",
        "standing_near_crosswalk",
    ):
        notes.append(
            f"{entity_id}: placed by its offset from the ego, not on a crosswalk "
            "(no crosswalk binding yet)."
        )
    if raw.get("type") == "bicycle":
        notes.append(f"{entity_id}: a bicycle, spawned as a pedestrian.")

    distance = base_distance + _relative_distance(raw)
    for index in range(spawned):
        member = entity_id if spawned == 1 else f"{entity_id}_{index + 1}"
        entities.append(
            Entity(
                id=member,
                kind="pedestrian",
                title=_title(raw),
                spawn=_relative_spawn(
                    distance + index * 1.0, "same", t=round(y, 2), heading=heading
                ),
            )
        )
        if walking:
            actions.append(
                _action(
                    "walk_straight",
                    f"{member} {context.replace('_', ' ')}",
                    actor=member,
                    speed_ms=1.4,
                )
            )


def _require_neighbours(ego_plan: _EgoPlan, entities: list[Entity]) -> None:
    """Search only where the lanes the road users are placed in exist.

    A pick without the lane a ``route_offset`` names is dropped by the sweep,
    so a search that ignores them mostly expands into nothing.  Only where the
    ego's own lane is searched for: a turn's pick is the junction lanelet, and
    the road users stand on the approach to it.
    """
    if ego_plan.anchor == "turn":
        return
    sides = sorted(
        {
            str(e.spawn.binding.params.get("side"))
            for e in entities
            if e.kind != "ego" and e.spawn.binding is not None
        }
        & {"left", "right"}
    )
    root = ego_plan.spawn.constraints[0]
    present = {
        c.params.get("value") for c in root.constraints if c.type == "has_adjacent"
    }
    for side in sides:
        if side not in present:
            root.constraints.insert(
                0, ConstraintNode(type="has_adjacent", params={"value": side})
            )


def _title(raw: Mapping[str, Any]) -> str:
    parts = [
        str(raw.get(key) or "")
        for key in ("type", "role", "_role", "context")
        if raw.get(key)
    ]
    return " / ".join(parts)


# ---------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------


def scene_to_document(
    scene: Mapping[str, Any],
    *,
    date: str = "unknown",
    session: str = "",
    options: CodImportOptions | None = None,
    blacklisted: bool = False,
) -> ImportedScene:
    """Return the document one classifier scene describes.

    Args:
        scene: One entry of ``whitelist_scenes`` (or ``blacklist_scenes``).
        date: The output's ``date``.
        session: The ``time_series`` key the scene sits under.
        options: Importer choices; defaults when omitted.
        blacklisted: Whether the scene is a blacklisted one.
    """
    options = options or CodImportOptions()
    notes: list[str] = []
    scene_id = str(scene.get("scene_id") or "scene")
    duration = _duration_seconds(scene)
    ego_plan = _plan_ego(scene, duration, options, notes)

    ego = Entity(
        id="ego",
        kind="ego",
        title="Ego",
        driven_by=options.driven_by,
        initial_speed_kmh=ego_plan.initial_speed_kmh,
        spawn=ego_plan.spawn,
        goal=ego_plan.goal,
    )
    entities: list[Entity] = [ego]
    actions: list[ActionNode] = []

    signal = _signal(scene)
    if signal and signal.get("signal_color"):
        colour = str(signal["signal_color"]).capitalize()
        actions.append(
            _action("traffic_signal", f"Signal {colour.lower()}", state=colour)
        )
        notes.append(
            f"signal {signal['signal_color']!r} is set on every traffic light, "
            "not only the ego's."
        )

    vehicles = pedestrians = 0
    for raw in scene.get("dynamic_entities") or ():
        if not isinstance(raw, Mapping):
            continue
        if raw.get("type") in ("pedestrian", "bicycle"):
            pedestrians += 1
            _add_pedestrians(
                raw,
                f"ped{pedestrians}",
                ego_plan.base_distance,
                options,
                entities,
                actions,
                notes,
            )
        else:
            vehicles += 1
            _add_vehicle(
                raw,
                f"npc{vehicles}",
                ego_plan.base_distance,
                options,
                entities,
                actions,
                notes,
            )

    _require_neighbours(ego_plan, entities)

    geometry = _geometry(scene)
    junction = (geometry.get("junctions") or {}).get("intersection") or {}
    plane = geometry.get("horizontal_plane") or {}
    if junction.get("junction_type"):
        notes.append(
            f"junction type {junction['junction_type']!r} is not searched for "
            "(no constraint yet)."
        )
    if plane.get("horizontal_plane_type") not in (None, "straight"):
        notes.append(
            f"road shape {plane['horizontal_plane_type']!r} is not searched for "
            "(no curvature constraint yet)."
        )

    timeout = float(max(20, math.ceil(duration * 2) + 10))
    decisions = scene.get("driving_decisions") or {}
    summary = ", ".join(
        str(v) for v in (decisions.get("lateral"), decisions.get("longitudinal")) if v
    )
    if blacklisted:
        summary = f"blacklisted ({scene.get('justification', '')})"
    description_lines = [
        f"Imported from CodSceneClassifier: {date} {session} {scene_id} "
        f"({duration:.1f} s; events: {', '.join(scene.get('event') or ())}).",
        f"Ego: {summary}.",
    ]
    if notes:
        description_lines.append("Approximations:")
        description_lines.extend(f"- {note}" for note in notes)

    document = ScenarioDocument(
        id=_snake(f"cod_{date}_{session}_{scene_id}") or "cod_scene",
        title=f"{scene_id}: {summary}" if summary else scene_id,
        description="\n".join(description_lines),
        timeout_seconds=timeout,
        map=options.map.model_copy(deep=True),
        entities=entities,
        actions=actions,
        assertions=Assertions(
            **{
                "pass": ego_plan.pass_conditions,
                "fail": [
                    _condition("collision"),
                    _condition("timeout", timeout_seconds=timeout),
                ],
            }
        ),
    )
    document.sync_layout()
    return ImportedScene(
        document=document,
        notes=notes,
        session=session,
        scene_id=scene_id,
        blacklisted=blacklisted,
    )


def _scenes(
    data: Mapping[str, Any], include_blacklist: bool
) -> Iterator[tuple[str, Mapping[str, Any], bool]]:
    for session, series in (data.get("time_series") or {}).items():
        if not isinstance(series, Mapping):
            continue
        for scene in series.get("whitelist_scenes") or ():
            yield str(session), scene, False
        if include_blacklist:
            for scene in series.get("blacklist_scenes") or ():
                yield str(session), scene, True


def import_cod_output(
    data: Mapping[str, Any], options: CodImportOptions | None = None
) -> list[ImportedScene]:
    """Return a document for every scene of one classifier output.

    Args:
        data: A parsed ``<date>.json`` (the light deliverable or a debug
            ``events_*.json``).
        options: Importer choices; defaults when omitted.
    """
    options = options or CodImportOptions()
    date = str(data.get("date") or "unknown")
    return [
        scene_to_document(
            scene, date=date, session=session, options=options, blacklisted=black
        )
        for session, scene, black in _scenes(data, options.include_blacklist)
        if isinstance(scene, Mapping)
    ]


def import_cod_file(
    path: str | Path, options: CodImportOptions | None = None
) -> list[ImportedScene]:
    """Read a classifier output file and return its documents."""
    with Path(path).open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, Mapping):
        raise ValueError(f"{path}: expected a JSON object at the top level.")
    return import_cod_output(data, options)


def main(argv: Sequence[str] | None = None) -> int:
    """``scenario-import-cod``: write one Scenario IR document per scene."""
    parser = argparse.ArgumentParser(
        description="Build Scenario IR documents from CodSceneClassifier output."
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Classifier JSON files.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("cod_scenarios"),
        help="Directory to write <scenario id>.yaml into.",
    )
    parser.add_argument(
        "--driven-by", choices=("autoware", "autopilot"), default="autoware"
    )
    parser.add_argument("--ego-speed-kmh", type=float, default=30.0)
    parser.add_argument("--include-blacklist", action="store_true")
    args = parser.parse_args(argv)

    options = CodImportOptions(
        driven_by=args.driven_by,
        ego_speed_kmh=args.ego_speed_kmh,
        include_blacklist=args.include_blacklist,
    )
    report: list[dict[str, Any]] = []
    for path in args.inputs:
        for imported in import_cod_file(path, options):
            target = args.output / f"{imported.document.id}.yaml"
            save_document(imported.document, target)
            report.append(
                {
                    "source": str(path),
                    "session": imported.session,
                    "scene_id": imported.scene_id,
                    "document": str(target),
                    "notes": imported.notes,
                }
            )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "import_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(report)} scenario(s) to {args.output}")
    return 0
