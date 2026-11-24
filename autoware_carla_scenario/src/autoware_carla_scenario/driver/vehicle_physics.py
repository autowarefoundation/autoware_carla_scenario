"""The ego's own CARLA vehicle parameters that shape its steering (``driver.vehicle``).

CARLA's vehicle physics has two parameters on the steering's side:

* each steered wheel's ``max_steer_angle`` (degrees), the wheel angle at full lock;
* the vehicle's ``steering_curve``, ``(speed km/h, factor)`` points: how much of
  the steer the wheels get at each speed.

:class:`VehicleSteeringConfig` holds the values a run sets (``None`` keeps the
blueprint's) and :func:`apply_vehicle_steering` writes them to an actor.  CARLA
has no steering dead time or servo response to set; these are all there is.

The trajectory follower's angle-to-steer map (``driver.control``'s
``max_steer_angle_deg`` and ``steer_exponent``) is a measurement of how CARLA
answers, not a copy of these parameters: CARLA 0.10's Chaos vehicles turn
``56 deg * steer**2`` while their wheels report 70 degrees.  Changing either
parameter here can change that answer, and the map is not changed with it:
measure the response again and set ``driver.control`` to match.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Mapping, Optional, Tuple

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

__all__ = [
    "VehicleSteeringConfig",
    "apply_vehicle_steering",
]

logger = logging.getLogger(__name__)

#: The largest wheel angle at full lock this accepts, degrees.
_MAX_STEER_ANGLE_LIMIT_DEG = 90.0


@dataclass(frozen=True)
class VehicleSteeringConfig:
    """The ego's steering parameters in CARLA's vehicle physics; ``None`` keeps
    what the blueprint has."""

    max_steer_angle_deg: Optional[float] = None
    """The wheel angle at full lock, degrees, set on every steered wheel (a wheel
    ``affected_by_steering``, or one with a ``max_steer_angle`` above zero).  In
    ``(0, 90]``."""

    steering_curve: Optional[Tuple[Tuple[float, float], ...]] = None
    """The ``(speed km/h, factor)`` points of CARLA's ``steering_curve``: the
    share of the steer the wheels get at each speed, interpolated between the
    points.  Speeds zero or more and increasing, factors zero or more;
    ``((0, 1), (120, 1))`` lets every speed steer fully."""

    def __post_init__(self) -> None:
        angle = self.max_steer_angle_deg
        if angle is not None and not 0.0 < angle <= _MAX_STEER_ANGLE_LIMIT_DEG:
            raise ValueError(
                f"max_steer_angle_deg must be in (0, {_MAX_STEER_ANGLE_LIMIT_DEG:g}], "
                f"got {angle}"
            )
        curve = self.steering_curve
        if curve is None:
            return
        if not curve:
            raise ValueError("steering_curve must have at least one point, got []")
        previous = -math.inf
        for speed, factor in curve:
            if speed < 0.0 or factor < 0.0:
                raise ValueError(
                    f"steering_curve's speeds and factors must be zero or more, "
                    f"got ({speed}, {factor})"
                )
            if speed <= previous:
                raise ValueError(
                    f"steering_curve's speeds must increase, got {speed} after "
                    f"{previous}"
                )
            previous = speed

    @property
    def changes_anything(self) -> bool:
        """Whether this sets any parameter."""
        return self.max_steer_angle_deg is not None or self.steering_curve is not None

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "VehicleSteeringConfig":
        """The config of a ``driver.vehicle`` mapping: a null value keeps the
        blueprint's; ``steering_curve`` is a list of ``[speed, factor]`` pairs.

        Raises:
            ValueError: naming the key, for a key this config does not define or a
                value outside its domain.
        """
        from ..value_text import shown  # noqa: PLC0415
        from .control import _config_number  # noqa: PLC0415

        known = {f.name for f in fields(cls)}
        unknown = sorted(set(mapping) - known)
        if unknown:
            raise ValueError(
                f"{unknown[0]}: Unknown VehicleSteeringConfig key(s) {unknown}. "
                f"Known keys: {sorted(known)}"
            )
        angle = mapping.get("max_steer_angle_deg")
        curve = mapping.get("steering_curve")
        points: Optional[Tuple[Tuple[float, float], ...]] = None
        if curve is not None:
            if isinstance(curve, (str, Mapping)) or not hasattr(curve, "__iter__"):
                raise ValueError(
                    f"steering_curve must be a list of [speed, factor] pairs, "
                    f"got {shown(curve)}"
                )
            pairs = []
            for index, point in enumerate(curve):
                key = f"steering_curve[{index}]"
                if isinstance(point, (str, Mapping)) or not hasattr(point, "__len__"):
                    raise ValueError(
                        f"{key} must be a [speed, factor] pair, got {shown(point)}"
                    )
                if len(point) != 2:
                    raise ValueError(
                        f"{key} must be a [speed, factor] pair, got {shown(point)}"
                    )
                pairs.append(
                    (_config_number(key, point[0]), _config_number(key, point[1]))
                )
            points = tuple(pairs)
        return cls(
            max_steer_angle_deg=(
                None if angle is None else _config_number("max_steer_angle_deg", angle)
            ),
            steering_curve=points,
        )


def apply_vehicle_steering(actor: "carla.Actor", config: VehicleSteeringConfig) -> None:
    """Write *config*'s parameters into *actor*'s vehicle physics; one it leaves
    ``None`` stays as it is.  Nothing is applied when *config* sets nothing.

    CARLA applies a physics control on its next tick.
    """
    if not config.changes_anything:
        return
    import typesafe_carla.carla as carla  # noqa: PLC0415

    from ..utils.vehicles import as_vehicle  # noqa: PLC0415

    vehicle = as_vehicle(actor)
    physics = vehicle.get_physics_control()
    angle = config.max_steer_angle_deg
    if angle is not None:
        wheels = list(physics.wheels)
        steered = 0
        for wheel in wheels:
            if wheel.affected_by_steering or float(wheel.max_steer_angle) > 0.0:
                wheel.max_steer_angle = angle
                steered += 1
        if steered == 0:
            logger.warning(
                "driver.vehicle.max_steer_angle_deg: actor %s has no steered wheel",
                actor.id,
            )
        physics.wheels = wheels
    if config.steering_curve is not None:
        physics.steering_curve = [
            carla.Vector2D(speed, factor) for speed, factor in config.steering_curve
        ]
    vehicle.apply_physics_control(physics)
    logger.info(
        "driver.vehicle: max_steer_angle %s deg, steering_curve %s",
        "kept" if angle is None else f"{angle:g}",
        "kept" if config.steering_curve is None else list(config.steering_curve),
    )
