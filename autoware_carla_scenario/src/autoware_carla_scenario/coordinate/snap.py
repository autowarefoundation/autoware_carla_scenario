"""Snap a pose onto the actual CARLA road surface.

The coordinate transform pipeline (Lanelet2 → OpenDRIVE → CARLA) can produce
positions that are slightly off the drivable surface due to geometry mismatches
between the Lanelet2 map and the XODR road network.  :func:`snap_to_carla_road`
corrects the position via OpenDRIVE projection (for :class:`Lanelet2Pose` input)
or the CARLA waypoint API (for :class:`CarlaWorldPose` input), and z via the
nearest spawn point.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Union, overload

import carla

from .poses import CarlaWorldPose, Lanelet2Pose, OpenDrivePose

logger = logging.getLogger(__name__)

#: Maximum distance (m) to the nearest spawn point before a warning is emitted.
_SPAWN_POINT_WARN_DISTANCE: float = 10.0


@dataclass(frozen=True)
class GroundProjectionConfig:
    """Tunable parameters for the ground-projection z-refinement.

    The ray is cast downward from ``z_estimate + ray_distance_upper`` and
    searches a total of ``ray_distance_upper + ray_distance_lower`` metres,
    i.e. from ``z_estimate + upper`` to ``z_estimate - lower``.
    """

    #: Search range (m) above the estimated z.
    ray_distance_upper: float = 5.0

    #: Search range (m) below the estimated z.
    ray_distance_lower: float = 5.0

    #: Minimum absolute z delta (m) to log an info message.
    z_log_threshold: float = 0.01


@overload
def snap_to_carla_road(
    pose: Lanelet2Pose,
    world: "carla.World",
    *,
    ground_projection: GroundProjectionConfig = ...,
) -> CarlaWorldPose: ...


@overload
def snap_to_carla_road(
    pose: OpenDrivePose,
    world: "carla.World",
    *,
    ground_projection: GroundProjectionConfig = ...,
) -> CarlaWorldPose: ...


@overload
def snap_to_carla_road(
    pose: CarlaWorldPose,
    world: "carla.World",
    *,
    ground_projection: GroundProjectionConfig = ...,
) -> CarlaWorldPose: ...


def snap_to_carla_road(
    pose: Union[CarlaWorldPose, Lanelet2Pose, OpenDrivePose],
    world: "carla.World",
    *,
    ground_projection: GroundProjectionConfig = GroundProjectionConfig(),
) -> CarlaWorldPose:
    """Snap a pose onto the CARLA drivable surface.

    When given a :class:`Lanelet2Pose`, the function converts to an approximate
    CARLA position via the Lanelet2 centerline, projects it onto the nearest
    OpenDRIVE road, then re-converts through the XODR geometry.  This
    round-trip ensures the final position lies on the XODR road surface that
    CARLA trusts.

    When given an :class:`OpenDrivePose`, the function uses
    ``carla.Map.get_waypoint_xodr(road_id, lane_id, s)`` to obtain the exact
    position on the CARLA road surface.  This is the most accurate path
    because it uses CARLA's own OpenDRIVE projection—no spawn-point z
    approximation is needed.

    When given a :class:`CarlaWorldPose`, the function uses the CARLA waypoint
    API to project x/y onto the nearest road.

    For Lanelet2 and CARLA-world inputs z is corrected using the nearest CARLA
    spawn point (spawn-point elevations match the physics-engine ground plane).

    If the pose cannot be matched to any road, a warning is logged and the
    original pose (or its CARLA equivalent) is returned unchanged.

    Parameters
    ----------
    pose:
        A Lanelet2, OpenDRIVE, or CARLA world pose to snap.
    world:
        An active ``carla.World`` instance.
    ground_projection:
        Parameters for the ground-projection z-refinement.  Defaults to
        :class:`GroundProjectionConfig` with its default values.

    Returns
    -------
    CarlaWorldPose
        A new pose snapped to the road surface.
    """
    if isinstance(pose, Lanelet2Pose):
        return _snap_lanelet2_to_its_centreline(
            pose, world, ground_projection=ground_projection
        )
    if isinstance(pose, OpenDrivePose):
        return _snap_opendrive_via_waypoint_xodr(
            pose, world, ground_projection=ground_projection
        )
    return _snap_carla_via_waypoint(pose, world, ground_projection=ground_projection)


# ---------------------------------------------------------------------------
# Lanelet2Pose path – read the lanelet the pose names
# ---------------------------------------------------------------------------


def _snap_lanelet2_to_its_centreline(
    pose: Lanelet2Pose,
    world: "carla.World",
    *,
    ground_projection: GroundProjectionConfig,
) -> CarlaWorldPose:
    """Place a Lanelet2 pose where its own lanelet says, and put it on the ground.

    A :class:`Lanelet2Pose` is a Frenet pose on one named lanelet: ``s`` along
    that lanelet's centreline, ``t`` across it. Everything needed to turn it into
    a position is therefore in the lanelet, and the map Autoware plans on is the
    Lanelet2 one -- so the lanelet's own answer is not an approximation of the
    right answer, it *is* the right answer.

    This used to go round by OpenDRIVE: project the centreline point onto the
    road's reference line for ``(s, t)``, move ``t`` to the lane centre, convert
    back through the XODR geometry. The stated reason was to land on the surface
    CARLA trusts, but the only part of the pose CARLA is actually the authority
    on is its height, which is read off the ground below. The round trip bought
    nothing for x and y and cost a great deal:

    * ``t`` was laid off the pose's own heading rather than the line it was
      measured against, so a lanelet running against its road's reference line
      -- the common case on a left-hand-traffic map converted from Lanelet2 --
      came back reflected by 2t, up to 3.5 m;
    * the lane centre it corrected to followed a lane id that, for those same
      lanelets, named the lane on the other side of the reference line, moving a
      goal into the opposing lane;
    * the projection took the nearest vertex of the reference line over the
      whole road, so a road that passes near itself resolved ``s`` onto the
      wrong stretch;
    * and the heading had to be taken back off the lanelet anyway, for exactly
      the reason the position now is.

    None of those can arise from reading the lanelet directly.

    Height still comes from CARLA: the nearest spawn point gives the elevation
    the physics engine uses, and a ray cast refines it, so a pose written on a
    map with its own idea of elevation still lands on the road.
    """
    from .transform import to_carla_world  # noqa: PLC0415

    on_lanelet = to_carla_world(pose)

    snapped_z = _z_from_nearest_spawn_point(on_lanelet.x, on_lanelet.y, world)
    base_z = snapped_z if snapped_z is not None else on_lanelet.z
    refined_z = refine_z_with_ground_projection(
        on_lanelet.x,
        on_lanelet.y,
        base_z,
        world,
        ground_projection=ground_projection,
    )

    result = CarlaWorldPose(
        x=on_lanelet.x,
        y=on_lanelet.y,
        z=refined_z,
        roll=on_lanelet.roll,
        pitch=on_lanelet.pitch,
        yaw=on_lanelet.yaw,
    )

    logger.info(
        "snap (Lanelet2): lanelet %d s=%.2f t=%.2f -> CARLA (%.2f, %.2f, %.3f) yaw=%.1f",
        pose.lanelet_id,
        pose.s,
        pose.t,
        result.x,
        result.y,
        result.z,
        result.yaw,
    )
    return result


# ---------------------------------------------------------------------------
# OpenDrivePose path – snap via get_waypoint_xodr
# ---------------------------------------------------------------------------


def _snap_opendrive_via_waypoint_xodr(
    pose: OpenDrivePose,
    world: "carla.World",
    *,
    ground_projection: GroundProjectionConfig,
) -> CarlaWorldPose:
    """Snap an OpenDRIVE pose using ``carla.Map.get_waypoint_xodr``.

    This is the most accurate snap path because CARLA resolves
    ``(road_id, lane_id, s)`` directly against its internal OpenDRIVE
    geometry, producing exact x/y/z and yaw on the road surface.
    """
    carla_map = world.get_map()

    waypoint = carla_map.get_waypoint_xodr(
        int(pose.road_id),
        pose.lane_id,
        pose.s,
    )
    if waypoint is None:
        logger.warning(
            "get_waypoint_xodr(road=%s, lane=%d, s=%.2f) returned None; "
            "falling back to coordinate transform",
            pose.road_id,
            pose.lane_id,
            pose.s,
        )
        from .transform import to_carla_world  # noqa: PLC0415

        return to_carla_world(pose)

    tf = waypoint.transform
    refined_z = refine_z_with_ground_projection(
        tf.location.x,
        tf.location.y,
        tf.location.z,
        world,
        ground_projection=ground_projection,
    )

    result = CarlaWorldPose(
        x=tf.location.x,
        y=tf.location.y,
        z=refined_z,
        yaw=tf.rotation.yaw,
    )

    logger.info(
        "snap (OpenDRIVE): road='%s' lane=%d s=%.2f -> "
        "CARLA (%.2f, %.2f, %.3f) yaw=%.1f",
        pose.road_id,
        pose.lane_id,
        pose.s,
        result.x,
        result.y,
        result.z,
        result.yaw,
    )

    return result


# ---------------------------------------------------------------------------
# CarlaWorldPose path – snap via waypoint API
# ---------------------------------------------------------------------------


def _snap_carla_via_waypoint(
    pose: CarlaWorldPose,
    world: "carla.World",
    *,
    ground_projection: GroundProjectionConfig,
) -> CarlaWorldPose:
    """Snap a CARLA world pose using the waypoint API.

    Uses ``get_waypoint()`` to project x/y onto the nearest road and the
    nearest spawn point for z.  The original yaw is preserved.
    """
    carla_map = world.get_map()

    waypoint = carla_map.get_waypoint(
        carla.Location(x=pose.x, y=pose.y, z=0.0),
    )
    if waypoint is None:
        logger.warning(
            "get_waypoint() returned None for (%.2f, %.2f); "
            "returning original pose unchanged",
            pose.x,
            pose.y,
        )
        return pose

    snapped_x = waypoint.transform.location.x
    snapped_y = waypoint.transform.location.y

    snapped_z = _z_from_nearest_spawn_point(snapped_x, snapped_y, world)
    base_z = snapped_z if snapped_z is not None else pose.z
    refined_z = refine_z_with_ground_projection(
        snapped_x, snapped_y, base_z, world, ground_projection=ground_projection
    )

    result = CarlaWorldPose(
        x=snapped_x,
        y=snapped_y,
        z=refined_z,
        roll=pose.roll,
        pitch=pose.pitch,
        yaw=pose.yaw,
    )

    logger.debug(
        "snap (CARLA): (%.2f, %.2f, %.2f) -> (%.2f, %.2f, %.2f)",
        pose.x,
        pose.y,
        pose.z,
        result.x,
        result.y,
        result.z,
    )

    return result


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _z_from_nearest_spawn_point(
    x: float, y: float, world: "carla.World"
) -> float | None:
    """Return z from the nearest CARLA spawn point, or ``None`` if unavailable.

    Emits a warning when the nearest spawn point is more than
    :data:`_SPAWN_POINT_WARN_DISTANCE` metres away.
    """
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        return None

    best_dist2 = float("inf")
    best_z: float | None = None
    for sp in spawn_points:
        d2 = (sp.location.x - x) ** 2 + (sp.location.y - y) ** 2
        if d2 < best_dist2:
            best_dist2 = d2
            best_z = sp.location.z

    if best_z is not None:
        dist = math.sqrt(best_dist2)
        if dist > _SPAWN_POINT_WARN_DISTANCE:
            logger.warning(
                "Nearest spawn point is %.1fm away from (%.1f, %.1f); "
                "z=%.2f may be inaccurate",
                dist,
                x,
                y,
                best_z,
            )

    return best_z


def refine_z_with_ground_projection(
    x: float,
    y: float,
    z_estimate: float,
    world: "carla.World",
    *,
    ground_projection: GroundProjectionConfig = GroundProjectionConfig(),
) -> float:
    """Refine *z_estimate* by casting a downward ray via ``world.ground_projection``.

    A ray is cast from ``(x, y, z_estimate + offset)`` downward.  If the ray
    hits the physics-mesh ground surface, the hit-point's z coordinate is
    returned.

    Parameters
    ----------
    x, y:
        CARLA world horizontal coordinates.
    z_estimate:
        The best z estimate from existing logic (spawn-point / waypoint).
    world:
        An active ``carla.World`` instance.
    ground_projection:
        Parameters controlling the ray search range and logging threshold.
        Defaults to :class:`GroundProjectionConfig` with its default values.

    Returns
    -------
    float
        The refined z coordinate.

    Raises
    ------
    RuntimeError
        If ``ground_projection`` is unavailable, fails, or returns no hit.
        A missing ground hit indicates that the spawn location is outside the
        physics mesh (e.g. off-road or the map is not loaded correctly).
    """
    cfg = ground_projection
    origin = carla.Location(
        x=x,
        y=y,
        z=z_estimate + cfg.ray_distance_upper,
    )
    search_distance = cfg.ray_distance_upper + cfg.ray_distance_lower

    try:
        result = world.ground_projection(origin, search_distance)
    except AttributeError:
        raise RuntimeError(
            "world.ground_projection() is not available in this CARLA version. "
            "A CARLA build that supports ground_projection is required."
        ) from None
    except RuntimeError as exc:
        raise RuntimeError(
            f"world.ground_projection() failed at ({x:.2f}, {y:.2f}, "
            f"{origin.z:.2f}): {exc}. "
            "The physics mesh may not be loaded."
        ) from exc

    if result is None:
        raise RuntimeError(
            f"ground_projection returned no hit at ({x:.2f}, {y:.2f}) "
            f"with z_estimate={z_estimate:.3f}. "
            "The spawn location is likely outside the drivable surface "
            "or the CARLA map is not loaded correctly."
        )

    if result.label == carla.CityObjectLabel.NONE:
        logger.warning(
            "ground_projection hit at (%.2f, %.2f) has CityObjectLabel.NONE. "
            "Enable semantic tags on the map's ground meshes for more "
            "reliable ground detection.",
            x,
            y,
        )

    ground_z = result.location.z
    delta = ground_z - z_estimate
    if abs(delta) > cfg.z_log_threshold:
        logger.info(
            "ground_projection refined z: %.3f -> %.3f (delta=%.3f) at (%.1f, %.1f)",
            z_estimate,
            ground_z,
            delta,
            x,
            y,
        )
    return ground_z
