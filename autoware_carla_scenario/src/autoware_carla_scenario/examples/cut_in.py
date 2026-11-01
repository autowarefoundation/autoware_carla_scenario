"""Example scenario: a vehicle in the next lane cuts in just ahead of the ego.

An NPC starts in the lane beside the ego, a little ahead, and after a moment
changes into the ego's lane. The ego must not hit it: the run passes once the
NPC has been in the ego's lane -- on its spawn lanelet or one that follows it
before the next junction (:func:`npc_in_lanes_ahead`) -- for ``hold_seconds`` without a collision. It
fails on any collision of the ego (``fail_on_any_collision``; off, only on one
with the NPC), and when the ego leaves the lanes ahead of it -- its spawn
lanelet and the lanelets that follow it -- for ``leaving_lane_seconds``: a lane
change, a turn onto a lanelet that does not follow it, or leaving the road
(``fail_on_leaving_lane``).

``side`` is the side of the ego the NPC comes from, so it changes lanes the
other way. Its lanelet is the one beside the ego's, which a sweep binds with
the ``adjacent`` binding.

Where it starts is given one of two ways. ``npc_ahead_m`` is a raw setting:
the NPC's ``s`` along its lanelet less the ego's along its own. ``npc_gap_m``
is the gap ``vehicle_ahead_gap_m`` should read at the start -- bumper to
bumper, along the ego's lane: the NPC's ``s`` is solved for, from the two
spawn poses and the two vehicles' sizes, so it does
(:func:`~autoware_carla_scenario.place_at_vehicle_ahead_gap`) -- the same
wherever the scenario is placed, however the two lanelets are cut and curve.
The scenario's controls set ``npc_gap_m``, so a gap drawn from an ODD is the
gap the run starts at.

Typical usage
-------------
On every lane of a map with a lane beside it, one case each::

    uv run scenario --multirun hydra/sweeper=lanelet_constraint \\
        scenario=cut_in/left map=<map>
"""

from __future__ import annotations

import logging

import lanelet2.geometry
import typesafe_carla.carla as carla

from autoware_carla_scenario import (
    EGO_ROLE_NAME,
    BaseCondition,
    BaseScenario,
    CollisionCondition,
    EgoConfig,
    ElapsedTimeCondition,
    EntityLanePositionCondition,
    EntityRole,
    GroundProjectionConfig,
    LaneChangeAction,
    LaneChangeDirection,
    Lanelet2Pose,
    NotCondition,
    OrCondition,
    PersistentCondition,
    SetSpeedAction,
    SpawnTransform,
    StickyCondition,
    TimeoutCondition,
    TrafficLightTarget,
    TrafficSignalAction,
    VehicleEntity,
    VehicleEntityConfig,
    lanelets_ahead,
    place_at_vehicle_ahead_gap,
    snap_to_carla_road,
    to_carla_world,
    to_opendrive,
    vehicle_bumpers,
)
from autoware_carla_scenario.coordinate import MapManager
from autoware_carla_scenario.failure import FailureCategory
from autoware_carla_scenario.odd import lanelet_in_junction

from .configs import CutInConfig

logger = logging.getLogger(__name__)

#: The role of the vehicle that cuts in.
CUT_IN_ROLE = EntityRole.npc(1)

#: The way the NPC changes lanes, by the side it comes from.
_CHANGE_TOWARDS_EGO: dict[str, LaneChangeDirection] = {
    "left": LaneChangeDirection.RIGHT,
    "right": LaneChangeDirection.LEFT,
}


def left_lanes_ahead(
    lanelet_id: int, distance_m: float, seconds: float
) -> BaseCondition:
    """The ego has been off the lanes ahead of *lanelet_id* for *seconds*.

    A composition of the framework's conditions, no condition of its own: an
    ``EntityLanePositionCondition`` for each lanelet :func:`lanelets_ahead`
    returns, under an ``OrCondition`` (the ego is on one of them), under a
    logical ``NotCondition`` (``holds_when_pending=True``: on none of them),
    held for *seconds* (``PersistentCondition``) so that a tick off at a seam
    between lanelets does not count.  On none of them is in a lane beside
    them, on a lanelet that turns off them, or on no lanelet at all -- off the
    road.  The position tested is the vehicle's centre, its CARLA location.

    Not every lanelet :func:`lanelets_ahead` returns is kept: following every
    branch, it can loop round a block into the lane beside the ego's, and
    those lanelets would then count as its own (#1080).  So
    :func:`ego_lanes_ahead` leaves out each lanelet beside one of
    :func:`lanes_ahead_before_junction` that it reaches only that way.

    *seconds* of zero or less fails on the first tick off them.
    """
    # Appended one by one, and chosen by if/else: Codon gives neither a list
    # comprehension nor a conditional expression the base class's type.
    on_lanes: list[BaseCondition] = []
    for lanelet in ego_lanes_ahead(lanelet_id, distance_m):
        on_lanes.append(
            EntityLanePositionCondition(
                EGO_ROLE_NAME,
                Lanelet2Pose(lanelet_id=lanelet, s=0.0),
                label=f"ego_on_lanelet_{lanelet}",
            )
        )
    on_any: BaseCondition
    if len(on_lanes) == 1:
        on_any = on_lanes[0]
    else:
        on_any = OrCondition(on_lanes, label="ego_on_its_lanes")
    if seconds <= 0.0:
        return NotCondition(on_any, label="ego_left_its_lane", holds_when_pending=True)
    return PersistentCondition(
        NotCondition(on_any, label="ego_off_its_lanes", holds_when_pending=True),
        seconds,
        label="ego_left_its_lane",
    )


def lanes_ahead_before_junction(lanelet_id: int, distance_m: float) -> list[int]:
    """*lanelet_id* and the lanelets ahead of it up to the first junction.

    :func:`lanelets_ahead` reaches through junctions and down every branch, and
    in a junction lanelets overlap: a vehicle crossing straight on in the lane
    beside is inside the ego lane's left or right turn.  So only the lanelets
    reached from *lanelet_id* through no junction lanelet are kept -- the lane
    the ego is on until the next intersection, where a lane change ends.  In
    the order :func:`lanelets_ahead` gives, *lanelet_id* first.
    """
    mm = MapManager.get_instance()
    layer = mm.lanelet_map.laneletLayer
    graph = mm.routing_graph
    ahead = lanelets_ahead(lanelet_id, distance_m)
    kept: list[int] = [lanelet_id]
    # Until nothing more is added: a lanelet may be listed before the
    # predecessor that reaches it without a junction.
    grew = True
    while grew:
        grew = False
        for candidate in ahead:
            if candidate in kept:
                continue
            lanelet = layer[candidate]
            if lanelet_in_junction(lanelet):
                continue
            for previous in graph.previous(lanelet, False):
                if int(previous.id) in kept:
                    kept.append(candidate)
                    grew = True
                    break
    ordered: list[int] = []
    for candidate in ahead:
        if candidate in kept:
            ordered.append(candidate)
    return ordered


def _beside(lanelet_id: int) -> list[int]:
    """The lanelets beside *lanelet_id*, either side, a lane change allowed
    across the line between or not (the routing graph's neighbours)."""
    mm = MapManager.get_instance()
    graph = mm.routing_graph
    lanelet = mm.lanelet_map.laneletLayer[lanelet_id]
    found: list[int] = []
    left = graph.left(lanelet)
    if left is not None:
        found.append(int(left.id))
    right = graph.right(lanelet)
    if right is not None:
        found.append(int(right.id))
    adjacent_left = graph.adjacentLeft(lanelet)
    if adjacent_left is not None:
        found.append(int(adjacent_left.id))
    adjacent_right = graph.adjacentRight(lanelet)
    if adjacent_right is not None:
        found.append(int(adjacent_right.id))
    return found


def _length(lanelet_id: int) -> float:
    """The length of *lanelet_id*'s centerline (m), as :func:`lanelets_ahead`
    measures it."""
    lanelet = MapManager.get_instance().lanelet_map.laneletLayer[lanelet_id]
    return float(lanelet2.geometry.length(lanelet2.geometry.to2D(lanelet.centerline)))


def _begins(ahead: list[int], distance_m: float) -> dict[int, float]:
    """How far from the start of ``ahead[0]`` each lanelet of *ahead* begins
    (m), by the shortest chain -- *ahead* as :func:`lanelets_ahead` gives it,
    nearest first, so the predecessor on that chain always comes earlier."""
    mm = MapManager.get_instance()
    layer = mm.lanelet_map.laneletLayer
    graph = mm.routing_graph
    begins: dict[int, float] = {}
    for candidate in ahead:
        if len(begins) == 0:
            begins[candidate] = 0.0
            continue
        nearest = distance_m
        for previous in graph.previous(layer[candidate], False):
            previous_id = int(previous.id)
            if previous_id in begins:
                ends_at = begins[previous_id] + _length(previous_id)
                if ends_at < nearest:
                    nearest = ends_at
        begins[candidate] = nearest
    return begins


def ego_lanes_ahead(lanelet_id: int, distance_m: float) -> list[int]:
    """The lanelets :func:`left_lanes_ahead` keeps the ego on.

    :func:`lanelets_ahead` of *lanelet_id*, less every lanelet beside one of
    :func:`lanes_ahead_before_junction` -- the routing graph's ``left`` and
    ``right``, and ``adjacentLeft`` and ``adjacentRight`` across a solid
    line -- that it reaches only past the end of the lanelet it is beside.
    Following every branch through junctions, :func:`lanelets_ahead` can loop
    round a block and come back into the lane beside the ego's, which a lane
    change reaches; it is never the ego's lane.  One it reaches before then
    is a branch of the ego's own lanes -- a turn through the junction ahead
    that ends beside one of them -- and is kept.  A lanelet of
    :func:`lanes_ahead_before_junction` itself is always kept.  In the order
    :func:`lanelets_ahead` gives, *lanelet_id* first.
    """
    own = lanes_ahead_before_junction(lanelet_id, distance_m)
    ahead = lanelets_ahead(lanelet_id, distance_m)
    begins = _begins(ahead, distance_m)
    # Each lanelet beside the ego's, and the furthest any lanelet of the
    # ego's it is beside ends: reached only past that, it is a loop.
    beside_ends: dict[int, float] = {}
    for lanelet in own:
        ends_at = begins[lanelet] + _length(lanelet)
        for neighbour in _beside(lanelet):
            if neighbour in own:
                continue
            if neighbour not in beside_ends or beside_ends[neighbour] < ends_at:
                beside_ends[neighbour] = ends_at
    kept: list[int] = []
    for lanelet in ahead:
        if lanelet in beside_ends and begins[lanelet] >= beside_ends[lanelet]:
            continue
        kept.append(lanelet)
    return kept


def npc_in_lanes_ahead(lanelet_id: int, distance_m: float) -> BaseCondition:
    """The cut-in NPC is in the ego's lane: on one of the lanes ahead of *lanelet_id*.

    *lanelet_id* and the lanelets that follow it up to the first junction
    (:func:`lanes_ahead_before_junction`), the lane beside never among them,
    so a lane change the NPC finishes on the lanelet after the ego's spawn
    lanelet counts as a cut-in too.  Not the junction lanelets
    :func:`left_lanes_ahead` also keeps the ego on, nor those past them: they
    overlap the lane beside's, and the NPC driving straight on in its own lane
    would count.  An ``EntityLanePositionCondition`` per lanelet, under an
    ``OrCondition`` when there is more than one, latched by a
    ``StickyCondition``: the NPC has cut in once it has been on any of them.
    """
    # Appended one by one, and chosen by if/else, as in left_lanes_ahead.
    on_lanes: list[BaseCondition] = []
    for lanelet in lanes_ahead_before_junction(lanelet_id, distance_m):
        on_lanes.append(
            EntityLanePositionCondition(
                CUT_IN_ROLE,
                Lanelet2Pose(lanelet_id=lanelet, s=0.0),
                label=f"npc_on_lanelet_{lanelet}",
            )
        )
    on_any: BaseCondition
    if len(on_lanes) == 1:
        on_any = on_lanes[0]
    else:
        on_any = OrCondition(on_lanes, label="npc_in_ego_lane")
    return StickyCondition(on_any, label="npc_in_ego_lane_sticky")


class CutInScenario(BaseScenario):
    """Spawn the ego and an NPC beside it; the NPC changes into the ego's lane."""

    _config: CutInConfig

    def __init__(
        self,
        ego_config: EgoConfig,
        spawn_pose: Lanelet2Pose,
        config: CutInConfig | None = None,
        ground_projection: GroundProjectionConfig | None = None,
    ) -> None:
        super().__init__(
            ego_config, spawn_pose=spawn_pose, ground_projection=ground_projection
        )
        self._config = config or CutInConfig()

    def setup(self) -> None:
        """Spawn the NPC beside the ego, and register its lane change and the conditions."""
        world = self.world
        cfg = self._config
        spawn = self._spawn_pose
        if spawn is None:
            raise ValueError("cut_in needs an ego spawn pose")
        if cfg.side not in _CHANGE_TOWARDS_EGO:
            raise ValueError(f"cut_in side must be 'left' or 'right', got {cfg.side!r}")

        self._setup_ego_spawn()
        self.register_init(
            TrafficSignalAction(
                state=carla.TrafficLightState.Green,
                lanelet2_traffic_light_ids=TrafficLightTarget.ALL,
                label="set_all_green",
            )
        )

        npc_gap_m = cfg.npc_gap_m
        if npc_gap_m is None:
            npc_pose = Lanelet2Pose(
                lanelet_id=cfg.npc_lanelet_id, s=spawn.s + cfg.npc_ahead_m
            )
        else:
            # The gap as the measure reads it, from where the ego spawns: x, y
            # and heading come off its lanelet, as _setup_ego_spawn places it.
            # Bumper to bumper: both vehicles' sizes, from their blueprints.
            ego_pose: Lanelet2Pose = spawn
            ego_at = to_carla_world(ego_pose).to_carla_transform()
            npc_pose = place_at_vehicle_ahead_gap(
                ego_at,
                cfg.npc_lanelet_id,
                npc_gap_m,
                ego_bumpers=vehicle_bumpers(world, self.ego_config.vehicle_type),
                npc_bumpers=vehicle_bumpers(world, cfg.npc_vehicle_type),
            )
        npc_snapped = snap_to_carla_road(
            npc_pose, world, ground_projection=self._ground_projection
        )
        npc = VehicleEntity(
            VehicleEntityConfig(
                role_name=str(CUT_IN_ROLE),
                spawn_location=SpawnTransform(npc_snapped.to_carla_transform()),
                vehicle_type=cfg.npc_vehicle_type,
                initial_speed_kmh=cfg.npc_initial_speed_kmh,
                spawn_retry_max_count=self.ego_config.spawn_retry_max_count,
                spawn_retry_t_step=self.ego_config.spawn_retry_t_step,
                spawn_retry_z_step=self.ego_config.spawn_retry_z_step,
                od_pose=to_opendrive(npc_pose),
                ground_projection=self._ground_projection,
            )
        )
        npc.spawn(world)
        self.register_entity(npc)
        logger.info(
            "Cut-in NPC on lanelet %d at s=%.2f, %s of the ego",
            cfg.npc_lanelet_id,
            npc_pose.s,
            cfg.side,
        )

        # Held at that speed, not only started at it: on a road whose limit is
        # higher, the TrafficManager would otherwise speed it up and away from
        # the ego -- and it refuses a forced lane change at speed (above some
        # 10 m/s on CARLA 0.10's Town04 highway).
        self.register_pre_tick(
            SetSpeedAction(
                CUT_IN_ROLE, cfg.npc_initial_speed_kmh, label="npc_initial_speed"
            )
        )
        self.register_pre_tick(
            LaneChangeAction(
                CUT_IN_ROLE,
                _CHANGE_TOWARDS_EGO[cfg.side],
                condition=ElapsedTimeCondition(
                    cfg.cut_in_after_seconds, label="cut_in_time"
                ),
                label="cut_in",
            )
        )

        # Latched once the NPC is on the ego's lane -- its spawn lanelet or one
        # that follows it before the next junction, wherever the NPC finishes
        # its lane change; then held for hold_seconds with no collision.
        cut_in = npc_in_lanes_ahead(spawn.lanelet_id, cfg.lanes_ahead_m)
        self.register_pass_condition(
            PersistentCondition(cut_in, cfg.hold_seconds, label="npc_cut_in_held")
        )
        if cfg.fail_on_any_collision:
            self.register_fail_condition(CollisionCondition(label="ego_collided"))
        else:
            self.register_fail_condition(
                CollisionCondition(target=CUT_IN_ROLE, label="hit_cut_in_vehicle")
            )
        if cfg.fail_on_leaving_lane:
            self.register_fail_condition(
                left_lanes_ahead(
                    spawn.lanelet_id, cfg.lanes_ahead_m, cfg.leaving_lane_seconds
                ),
                category=FailureCategory.LANE_KEEPING,
            )
        self.register_fail_condition(
            TimeoutCondition(cfg.timeout_seconds, label="scenario_timeout")
        )

    def is_done(self) -> bool:
        """Always ``False`` — termination is driven by pass/fail conditions."""
        return False
