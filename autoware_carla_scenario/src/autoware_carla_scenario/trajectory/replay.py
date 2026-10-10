"""Turn a T4 scene transcription into a running scenario.

Two levels, for two kinds of scenario:

* :func:`replay_t4_objects` -- called from any scenario's ``setup()``: spawns
  the scene's road users and gives each a
  :class:`~autoware_carla_scenario.actions.FollowTrajectoryAction` along its
  recorded track.  The scenario keeps everything else -- its ego, its goal,
  its pass and fail conditions.
* :class:`T4ReplayScenario` -- a whole scenario: the ego starts where the
  recording did and is sent where it went, the road users are replayed around
  it, and the run passes when the recording ends.  Subclass it to add the
  conditions the test is about.

The scenario has to load the map the scene was recorded on
(:attr:`~.t4.T4SceneTranscription.area_map_id`): the transcription's poses are
in that map's frame and are placed through it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional, Sequence, Union

from ..actions.follow_trajectory import HIDDEN_DEPTH_M, FollowTrajectoryAction
from ..actions.base import TickTiming
from ..conditions.elapsed_time import ElapsedTimeCondition
from ..constants import EGO_ROLE_NAME
from ..coordinate.poses import CarlaWorldPose
from ..entity._spawn import SpawnTransform
from ..scenario_base import BaseScenario, EgoConfig
from .model import (
    MapPose,
    ReferenceContext,
    TrajectoryFollowingMode,
    TrajectoryTiming,
)
from .t4 import T4Category, T4ObjectTrack, T4SceneTranscription

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_PEDESTRIAN_BLUEPRINT",
    "DEFAULT_VEHICLE_BLUEPRINT",
    "T4ReplayScenario",
    "default_t4_blueprint",
    "replay_t4_objects",
    "scene_timing",
]

#: The car every T4 vehicle is replayed as, unless told otherwise.
DEFAULT_VEHICLE_BLUEPRINT = "vehicle.tesla.model3"
#: The walker pedestrians -- and cyclists, which CARLA 0.10 has no bicycle
#: for -- are replayed as.
DEFAULT_PEDESTRIAN_BLUEPRINT = "walker.pedestrian.0001"

#: How far above the road (m) a vehicle is spawned, so its wheels clear it.
_VEHICLE_SPAWN_LIFT_M = 0.3

#: Picks the CARLA blueprint a track is replayed as.
BlueprintChooser = Callable[[T4ObjectTrack], str]


def default_t4_blueprint(track: T4ObjectTrack) -> str:
    """:data:`DEFAULT_VEHICLE_BLUEPRINT` for a vehicle, a walker for the rest."""
    if track.category is T4Category.VEHICLE:
        return DEFAULT_VEHICLE_BLUEPRINT
    return DEFAULT_PEDESTRIAN_BLUEPRINT


def scene_timing() -> TrajectoryTiming:
    """The scene clock on the scenario's: frame 0 at the run's first tick."""
    return TrajectoryTiming(domain=ReferenceContext.ABSOLUTE)


def replay_t4_objects(
    scenario: BaseScenario,
    transcription: T4SceneTranscription,
    *,
    following_mode: TrajectoryFollowingMode = TrajectoryFollowingMode.POSITION,
    time_reference: Optional[TrajectoryTiming] = None,
    categories: Optional[Sequence[T4Category]] = None,
    blueprint_for: BlueprintChooser = default_t4_blueprint,
    role_prefix: str = "t4",
    hide_when_absent: bool = True,
) -> List[FollowTrajectoryAction]:
    """Spawn the scene's road users and replay their tracks.

    Call from ``setup()``: entities can only be spawned before the warm-up.
    Each track becomes an entity named ``{role_prefix}_{category}_{track_id}``
    and a pre-tick :class:`FollowTrajectoryAction` registered on *scenario*.

    A track the recording picks up late, or loses early, would otherwise stand
    at its first or last pose for the rest of the run, blocking whatever lane
    it is in.  With *hide_when_absent* (``POSITION`` mode only) it is out of the
    world until its first sighting and after its last.  In ``FOLLOW`` mode it
    waits at its first pose instead.

    Args:
        scenario: The scenario being set up.
        transcription: The scene.
        following_mode: How the tracks are followed.
        time_reference: How the scene clock maps onto the scenario's; ``None``
            puts frame 0 at the run's first tick.
        categories: Which road users to replay; ``None`` replays all.
        blueprint_for: The CARLA blueprint for a track; a ``walker.*`` one
            spawns a pedestrian, anything else a vehicle.
        role_prefix: Prefix of every entity's role name.
        hide_when_absent: Keep a track out of the world outside the time it
            was recorded.

    Returns:
        The registered actions, one per track spawned.  A track that cannot be
        spawned is skipped with a warning rather than failing the run.
    """
    from ..entity.pedestrian_entity import (  # noqa: PLC0415
        PedestrianEntity,
        PedestrianEntityConfig,
    )
    from ..entity.vehicle_entity import VehicleEntity, VehicleEntityConfig  # noqa: PLC0415

    timing = time_reference if time_reference is not None else scene_timing()
    hide = hide_when_absent and following_mode is TrajectoryFollowingMode.POSITION
    world = scenario.world
    height = _ground(world)
    actions: List[FollowTrajectoryAction] = []
    for track in transcription.objects_of(*(categories or ())):
        role = f"{role_prefix}_{track.category.value}_{track.track_id}"
        blueprint = blueprint_for(track)
        walker = blueprint.startswith("walker.")
        start = _placed(
            track.poses[0], height, 0.0 if walker else _VEHICLE_SPAWN_LIFT_M
        )
        late = hide and timing.trajectory_time(0.0, 0.0) < track.start_time
        if late:
            # Out of the way of everything until its first sighting.
            start = CarlaWorldPose(
                x=start.x, y=start.y, z=start.z - HIDDEN_DEPTH_M, yaw=start.yaw
            )
        spawn = SpawnTransform(start.to_carla_transform())
        try:
            if walker:
                pedestrian = PedestrianEntity(
                    PedestrianEntityConfig(
                        role_name=role, spawn_location=spawn, walker_type=blueprint
                    )
                )
                actor = pedestrian.spawn(world)
                scenario.register_pedestrian(pedestrian)
            else:
                vehicle = VehicleEntity(
                    VehicleEntityConfig(
                        role_name=role, spawn_location=spawn, vehicle_type=blueprint
                    )
                )
                actor = vehicle.spawn(world)
                scenario.register_entity(vehicle)
        except (RuntimeError, ValueError) as exc:
            logger.warning("T4 replay: %s not spawned: %s", role, exc)
            continue
        if late:
            # Spawned under the map: without this it would fall until it is due.
            actor.set_simulate_physics(False)
        action = FollowTrajectoryAction(
            role,
            track.trajectory(name=role),
            time_reference=timing,
            following_mode=following_mode,
            timing=TickTiming.PRE_TICK,
            label=f"follow_{role}",
            hidden_outside_trajectory=hide,
        )
        scenario.register_pre_tick(action)
        actions.append(action)
    logger.info(
        "T4 replay of %s: %d of %d road users",
        transcription.scene_name,
        len(actions),
        len(transcription.objects),
    )
    return actions


class T4ReplayScenario(BaseScenario):
    """A T4 scene, replayed around the ego.

    In ``setup()`` the ego is placed where the recording's ego started, its
    goal is the scene's destination (or where the recording ended), and every
    road user is replayed with :func:`replay_t4_objects`.  The run passes when
    the recording's duration has elapsed; a subclass adds the conditions that
    make it a test (``CollisionCondition``, a distance kept, a stop made).

    Args:
        transcription: The scene, or the path of a saved transcription.
        ego_config: The ego's vehicle and goal.  Its spawn location is replaced
            by the recording's start, and its goal used when it names one.
        replay_ego: Also put the ego on the recorded drive.  Only for an ego
            nothing else drives (a TrafficManager ego); an external stack is
            refused by the action.
        following_mode: How the road users (and a replayed ego) follow.
        categories: Which road users to replay; ``None`` replays all.
        blueprint_for: The CARLA blueprint for a track.
        **kwargs: Passed to :class:`BaseScenario`.
    """

    def __init__(
        self,
        transcription: Union[T4SceneTranscription, str, Path],
        ego_config: Optional[EgoConfig] = None,
        *,
        replay_ego: bool = False,
        following_mode: TrajectoryFollowingMode = TrajectoryFollowingMode.POSITION,
        categories: Optional[Sequence[T4Category]] = None,
        blueprint_for: BlueprintChooser = default_t4_blueprint,
        **kwargs: object,
    ) -> None:
        if not isinstance(transcription, T4SceneTranscription):
            transcription = T4SceneTranscription.load(transcription)
        if ego_config is None:
            # A placeholder location: setup() puts the ego where the recording
            # starts, which needs the loaded map.
            ego_config = EgoConfig(spawn_location=_unplaced())
        super().__init__(ego_config, **kwargs)  # type: ignore[arg-type]
        self.transcription = transcription
        self.replay_ego = replay_ego
        self.following_mode = following_mode
        self.categories = categories
        self.blueprint_for = blueprint_for
        self.object_actions: List[FollowTrajectoryAction] = []

    def setup(self) -> None:
        """Place the ego, name its goal and replay the road users."""
        from ..conditions.base import find_actor_by_role_name  # noqa: PLC0415
        from ..coordinate.transform import to_lanelet2  # noqa: PLC0415

        world = self.world
        height = _ground(world)
        scene = self.transcription
        start = _placed(scene.ego_start, height, _VEHICLE_SPAWN_LIFT_M)
        self.ego_config.spawn_location = SpawnTransform(start.to_carla_transform())
        if self.goal_pose is None:
            destination = scene.goal if scene.goal is not None else scene.ego_end
            try:
                self.goal_pose = to_lanelet2(_placed(destination, height, 0.0))
            except Exception as exc:  # noqa: BLE001 -- off the Lanelet2 map
                logger.warning(
                    "T4 replay: the recording's destination is on no lanelet (%s); "
                    "the ego has no goal",
                    exc,
                )
        ego_actor = lambda: find_actor_by_role_name(world, EGO_ROLE_NAME)  # noqa: E731
        self.follow_with_spectator(ego_actor)
        self.register_route_to_goal(start)

        if self.replay_ego:
            self.register_pre_tick(
                FollowTrajectoryAction(
                    EGO_ROLE_NAME,
                    scene.ego_trajectory(),
                    time_reference=scene_timing(),
                    following_mode=self.following_mode,
                    label="follow_t4_ego",
                )
            )
        self.object_actions = replay_t4_objects(
            self,
            scene,
            following_mode=self.following_mode,
            categories=self.categories,
            blueprint_for=self.blueprint_for,
        )
        self.register_pass_condition(
            ElapsedTimeCondition(max(scene.duration, 0.1), label="t4_scene_end")
        )

    def is_done(self) -> bool:
        """Never early: the run ends with the recording, or on a condition."""
        return False


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _ground(world: "carla.World") -> Callable[[float, float], Optional[float]]:
    from .resolve import road_height  # noqa: PLC0415

    return road_height(world.get_map())


def _placed(
    pose: MapPose,
    height: Callable[[float, float], Optional[float]],
    lift: float,
) -> CarlaWorldPose:
    """A map-frame pose in CARLA, on the road surface under it plus *lift*."""
    from .resolve import resolve_position  # noqa: PLC0415

    x, y, z, yaw = resolve_position(pose)
    if z is None:
        z = height(x, y)
    return CarlaWorldPose(
        x=x, y=y, z=(z if z is not None else 0.0) + lift, yaw=yaw or 0.0
    )


def _unplaced() -> SpawnTransform:
    import typesafe_carla.carla as carla  # noqa: PLC0415

    return SpawnTransform(carla.Transform())
