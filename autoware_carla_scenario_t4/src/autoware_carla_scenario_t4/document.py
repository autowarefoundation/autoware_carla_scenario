"""Write a transcribed T4 scene down as a scenario document.

The document is the one the Scenario Editor edits and the declarative runtime
runs: the ego starts where the recording's ego started and is sent where it
went, and every road user is an entity with a *Follow Trajectory* card
replaying its track.  Open it in the editor to add what the test is about --
the conditions, a different ego driver, a road user removed or changed.  It
passes when the recording's duration has elapsed and fails on an ego collision.

Entities spawn on a lanelet, so every first pose is placed on the Lanelet2 map
the scene was recorded on (:class:`LaneletLocator`); the trajectories
themselves stay in the map frame, as vertices.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from autoware_carla_scenario.authoring.models import (
    DEFAULT_MODELS,
    ActionNode,
    Assertions,
    ConditionNode,
    Entity,
    GoalSpec,
    MapRef,
    ScenarioDocument,
    SpawnSpec,
    SValue,
)
from autoware_carla_scenario.trajectory import MapPose

from .reader import T4_FRAME_RATE_HZ, T4Category, T4ObjectTrack, T4SceneTranscription

logger = logging.getLogger(__name__)

__all__ = [
    "LaneletLocator",
    "LaneletPlacement",
    "load_lanelet_map",
    "transcription_to_document",
]

#: Seconds of slack the document's timeout gives beyond the recording.
_TIMEOUT_SLACK_S = 10.0
#: How far a pose may be from a lanelet (m) and still be placed on it.
_MAX_PLACEMENT_DISTANCE_M = 5.0
#: How many lanelets nearest a pose are weighed against its heading.
_CANDIDATES = 8


# ---------------------------------------------------------------------------
# Placing a pose on the map
# ---------------------------------------------------------------------------


def load_lanelet_map(
    lanelet2_path: Union[str, Path], xodr_path: Optional[Union[str, Path]] = None
) -> Any:
    """Load a Lanelet2 map the way the scenario runtime does.

    Same origin and projector as :class:`~autoware_carla_scenario.MapManager`,
    so a lanelet pose worked out here is the pose the run places.
    """
    # autoware_lanelet2_extension_python registers the MGRS projector, and has
    # to be imported before lanelet2.
    import autoware_lanelet2_extension_python.projection  # noqa: F401, PLC0415
    import lanelet2  # noqa: PLC0415

    from autoware_carla_scenario.coordinate.projection import (  # noqa: PLC0415
        map_origin,
        resolve_projector,
    )

    lanelet2_path = Path(lanelet2_path)
    (lat, lon, _alt), _ = map_origin(
        lanelet2_path, None if xodr_path is None else Path(xodr_path)
    )
    projector, _ = resolve_projector(lanelet2_path, lanelet2.io.Origin(lat, lon))
    return lanelet2.io.load(str(lanelet2_path), projector)


@dataclass(frozen=True)
class LaneletPlacement:
    """A map pose as a lanelet pose: ``s`` along the centreline, ``t`` left of it."""

    lanelet_id: int
    s: float
    t: float
    #: Yaw relative to the lanelet's direction, radians, anticlockwise.
    heading: float


class LaneletLocator:
    """Finds the lanelet a map-frame pose stands on.

    Lanelets overlap at their edges and stack at junctions, so the nearest
    one is not always the one meant: the candidates the pose is inside (or
    nearest to) are weighed by how well their direction matches its heading,
    which is what tells a lane from the opposite one beside it.
    """

    def __init__(self, lanelet_map: Any) -> None:
        self._map = lanelet_map

    def locate(self, pose: MapPose) -> Optional[LaneletPlacement]:
        """Where *pose* is on the map, or ``None`` when no lanelet is near."""
        import lanelet2  # noqa: PLC0415

        point = lanelet2.core.BasicPoint2d(pose.x, pose.y)
        nearest = lanelet2.geometry.findNearest(
            self._map.laneletLayer, point, _CANDIDATES
        )
        best: Optional[tuple[float, float, LaneletPlacement]] = None
        for distance, lanelet in nearest:
            if distance > _MAX_PLACEMENT_DISTANCE_M:
                continue
            centerline = lanelet2.geometry.to2D(lanelet.centerline)
            arc = lanelet2.geometry.toArcCoordinates(centerline, point)
            direction = _direction_at(centerline, arc.length)
            turn = 0.0 if pose.yaw is None else _wrap(pose.yaw - direction)
            placement = LaneletPlacement(
                lanelet_id=int(lanelet.id),
                s=max(0.0, float(arc.length)),
                t=float(arc.distance),
                heading=turn,
            )
            # Inside first, then aligned, then near.
            score = (round(distance, 2), abs(turn))
            if best is None or score < best[:2]:
                best = (score[0], score[1], placement)
        return None if best is None else best[2]


def _direction_at(centerline: Any, s: float) -> float:
    """The centreline's direction (rad) at arc length *s*."""
    points = [(p.x, p.y) for p in centerline]
    travelled = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        step = math.hypot(x1 - x0, y1 - y0)
        if travelled + step >= s and step > 1e-9:
            return math.atan2(y1 - y0, x1 - x0)
        travelled += step
    (x0, y0), (x1, y1) = points[-2], points[-1]
    return math.atan2(y1 - y0, x1 - x0)


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------


def transcription_to_document(
    transcription: T4SceneTranscription,
    locator: LaneletLocator,
    map_ref: MapRef,
    *,
    categories: Optional[Sequence[T4Category]] = None,
    following_mode: str = "position",
    replay_ego: bool = False,
    ego_driver: str = "autopilot",
    vehicle_type: Optional[str] = None,
    pedestrian_type: Optional[str] = None,
    vertex_interval_s: float = 0.2,
    document_id: Optional[str] = None,
    title: Optional[str] = None,
) -> ScenarioDocument:
    """The scenario document that replays *transcription*.

    Args:
        transcription: The scene.
        locator: Places poses on the scene's Lanelet2 map.
        map_ref: The map the document runs on -- the scene's area map.
        categories: Which road users to replay; ``None`` replays all.
        following_mode: ``position`` (an exact replay, the default) or
            ``follow`` (a controller drives each vehicle along its track).
        replay_ego: Put the ego on the recorded drive too.  Only for an ego
            nothing else drives, so not with ``ego_driver="autoware"``.
        ego_driver: ``autopilot`` or ``autoware``.
        vehicle_type: Blueprint of every replayed vehicle; the editor's default
            when ``None``.
        pedestrian_type: Blueprint of every pedestrian and cyclist (CARLA 0.10
            has no bicycle); the editor's default when ``None``.
        vertex_interval_s: Keep a vertex every this many seconds of the
            recording (its last always); 0 keeps every frame.  The replay
            interpolates between them.
        document_id: The document's id; derived from the scene name if omitted.
        title: The document's title; the scene name if omitted.

    Raises:
        ValueError: If the ego's start is on no lanelet, or ``replay_ego`` is
            asked of an Autoware ego.
    """
    if replay_ego and ego_driver == "autoware":
        raise ValueError("an Autoware ego drives itself; it cannot also be replayed")
    # The scenario clock starts at the ego's first valid frame.
    origin_s = transcription.ego_frames[0] / T4_FRAME_RATE_HZ
    every = max(1, round(vertex_interval_s * T4_FRAME_RATE_HZ))

    ego_start = locator.locate(transcription.ego_start)
    if ego_start is None:
        raise ValueError(
            f"the ego of {transcription.scene_name} starts on no lanelet of this "
            "map: is it the scene's area map "
            f"({transcription.area_map_id or 'not stated'})?"
        )
    destination = transcription.goal or transcription.ego_end
    goal = locator.locate(destination)
    ego = Entity(
        id="ego",
        kind="ego",
        title="Ego",
        driven_by=ego_driver,  # type: ignore[arg-type]
        spawn=SpawnSpec(
            lanelet_id=ego_start.lanelet_id, s=SValue(value=_r(ego_start.s))
        ),
        goal=None
        if goal is None
        else GoalSpec(lanelet_id=goal.lanelet_id, s=_r(goal.s)),
    )
    if goal is None:
        logger.warning(
            "%s: the destination is on no lanelet; the ego has no goal",
            transcription.scene_name,
        )

    entities = [ego]
    actions: list[ActionNode] = []
    if replay_ego:
        actions.append(
            _follow(
                "ego",
                _rows(transcription.ego_frames, transcription.ego_poses, every),
                origin_s,
                following_mode,
                hidden=False,
                title="Replay the recorded drive",
            )
        )

    wanted = set(categories) if categories else None
    skipped = 0
    for track in transcription.objects:
        if wanted is not None and track.category not in wanted:
            continue
        placement = locator.locate(track.poses[0])
        if placement is None:
            skipped += 1
            continue
        entity_id = f"{track.category.value}_{track.track_id}"
        walker = track.category is not T4Category.VEHICLE
        kind = "pedestrian" if walker else "vehicle"
        late = track.start_time > origin_s + 1e-9
        hidden = late and following_mode == "position"
        entities.append(
            Entity(
                id=entity_id,
                kind=kind,  # type: ignore[arg-type]
                title=_title(track),
                vehicle_type=(pedestrian_type if walker else vehicle_type)
                or DEFAULT_MODELS[kind],
                spawn=SpawnSpec(
                    lanelet_id=placement.lanelet_id,
                    s=SValue(value=_r(placement.s)),
                    t=_r(placement.t),
                    heading=_r(placement.heading, 4),
                    hidden=hidden,
                ),
            )
        )
        actions.append(
            _follow(
                entity_id,
                _rows(track.frames, track.poses, every),
                origin_s,
                following_mode,
                hidden=following_mode == "position",
                title=f"Replay {_title(track)}",
            )
        )
    if skipped:
        logger.warning(
            "%s: %d road user(s) start on no lanelet and were left out",
            transcription.scene_name,
            skipped,
        )

    duration = max(transcription.duration, 0.1)
    return ScenarioDocument(
        id=document_id or _identifier(transcription.scene_name),
        title=title or transcription.scene_name,
        description=(
            f"Replay of T4 scene {transcription.scene_name}"
            + (
                f" (area map {transcription.area_map_id})"
                if transcription.area_map_id
                else ""
            )
            + (
                f", labels by {transcription.annotation_source}"
                if transcription.annotation_source
                else ""
            )
            + "."
        ),
        timeout_seconds=_r(duration + _TIMEOUT_SLACK_S, 1),
        map=map_ref,
        entities=entities,
        actions=actions,
        assertions=Assertions.model_validate(
            {
                "pass": [
                    ConditionNode(
                        type="elapsed_time",
                        params={"duration_seconds": _r(duration, 2)},
                    )
                ],
                # The recording had none, so a collision is the replay's own.
                "fail": [
                    ConditionNode(
                        type="collision",
                        params={"min_impulse": 0.0, "target_type": "ANY"},
                    )
                ],
            }
        ),
    )


def _follow(
    actor: str,
    rows: list[list[Optional[float]]],
    origin_s: float,
    following_mode: str,
    *,
    hidden: bool,
    title: str,
) -> ActionNode:
    """A *Follow Trajectory* card replaying *rows* on the scenario clock."""
    return ActionNode(
        type="follow_trajectory",
        title=title,
        actor=actor,
        phase="pre_tick",
        params={
            "path_source": "vertices",
            "vertices": rows,
            "lanelet_ids": None,
            "speed_kmh": None,
            "lateral_offset_m": 0.0,
            "time_domain": "absolute",
            "time_scale": 1.0,
            # Scene time origin_s is the scenario's t = 0.
            "time_offset": _r(-origin_s, 2),
            "following_mode": following_mode,
            "initial_distance_offset": 0.0,
            "hidden_outside_trajectory": hidden,
        },
    )


def _rows(
    frames: Sequence[int], poses: Sequence[MapPose], every: int
) -> list[list[Optional[float]]]:
    """Every *every*-th vertex (and the last), as ``[x, y, yaw, time]``."""
    keep = list(range(0, len(frames), every))
    if keep[-1] != len(frames) - 1:
        keep.append(len(frames) - 1)
    return [
        [
            _r(poses[i].x, 3),
            _r(poses[i].y, 3),
            None if poses[i].yaw is None else _r(poses[i].yaw, 4),  # type: ignore[arg-type]
            _r(frames[i] / T4_FRAME_RATE_HZ, 2),
        ]
        for i in keep
    ]


def _title(track: T4ObjectTrack) -> str:
    return f"{track.category.value.capitalize()} {track.track_id}"


def _identifier(name: str) -> str:
    """*name* as a document id: lower_snake_case, starting with a letter."""
    text = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    if not text or not text[0].isalpha():
        text = f"t4_{text}"
    return text


def _r(value: float, digits: int = 3) -> float:
    return round(float(value), digits)
