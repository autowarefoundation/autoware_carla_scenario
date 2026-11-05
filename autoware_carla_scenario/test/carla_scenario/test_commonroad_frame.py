"""``commonroad.frame.read_frame()`` on a stand-in world, pinned to the value.

The road users' states are compared whole, field for field, so that moving
the geometry and the actor filtering into typed helpers (#1551) changes
nothing: a box off the actor's origin, an acceleration, an actor whose
acceleration or angular velocity cannot be read, a degenerate box, the kinds,
the actors that are not road users, one hidden under the map, one beyond the
radius, and the order the others come in.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from autoware_carla_scenario.commonroad import frame as cr_frame
from autoware_carla_scenario.constants import EGO_ROLE_NAME

from ._fake_world import ActorsById


def _v(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=z)


class _Actor:
    def __init__(
        self,
        actor_id: int,
        location: tuple[float, float, float],
        yaw: float,
        *,
        type_id: str = "vehicle.npc",
        attributes: Optional[dict[str, str]] = None,
        offset: tuple[float, float] = (0.0, 0.0),
        extent: tuple[float, float] = (2.25, 0.9),
        velocity: tuple[float, float] = (0.0, 0.0),
        acceleration: Optional[tuple[float, float]] = (0.0, 0.0),
        yaw_rate: Optional[float] = 0.0,
    ) -> None:
        self.id = actor_id
        self.is_alive = True
        self.type_id = type_id
        self.attributes = {"role_name": ""} if attributes is None else attributes
        self.bounding_box = SimpleNamespace(
            extent=_v(extent[0], extent[1], 0.8), location=_v(*offset, 0.8)
        )
        self._location = _v(*location)
        self._yaw = yaw
        self._velocity = _v(*velocity)
        self._acceleration = acceleration
        self._yaw_rate = yaw_rate

    def get_transform(self) -> Any:
        return SimpleNamespace(
            location=self._location, rotation=SimpleNamespace(yaw=self._yaw)
        )

    def get_velocity(self) -> Any:
        return self._velocity

    def get_acceleration(self) -> Any:
        if self._acceleration is None:
            raise RuntimeError("no acceleration")
        return _v(*self._acceleration)

    def get_angular_velocity(self) -> Any:
        if self._yaw_rate is None:
            raise RuntimeError("no angular velocity")
        return _v(0.0, 0.0, self._yaw_rate)


class _Attributes(dict):  # type: ignore[type-arg]
    """Attributes whose ``base_type`` cannot be read."""

    def get(self, key: str, default: Any = None) -> Any:
        if key == "base_type":
            raise RuntimeError("no base_type")
        return super().get(key, default)


class _World(ActorsById):
    def __init__(self, actors: list[_Actor]) -> None:
        self._actors = actors

    def get_actors(self) -> list[_Actor]:
        return self._actors


def _world() -> _World:
    return _World(
        [
            _Actor(7, (100.0, 3.0, 0.0), 0.0, type_id="vehicle.far"),
            _Actor(
                3,
                (12.0, -4.0, 0.2),
                -30.0,
                type_id="vehicle.carlamotors.firetruck",
                attributes={"role_name": "", "base_type": "Truck"},
                offset=(0.5, -0.2),
                velocity=(4.0, -1.0),
                acceleration=(0.5, 0.25),
                yaw_rate=-5.0,
            ),
            _Actor(
                1,
                (10.0, 2.0, 0.5),
                30.0,
                type_id="vehicle.lincoln.mkz",
                attributes={"role_name": str(EGO_ROLE_NAME), "base_type": "car"},
                offset=(1.0, 0.5),
                velocity=(8.0, 3.0),
                acceleration=(1.0, -2.0),
                yaw_rate=12.0,
            ),
            _Actor(20, (11.0, 2.0, 0.0), 0.0, type_id="sensor.camera.rgb"),
            _Actor(21, (11.0, 2.0, 0.0), 0.0, type_id="static.prop.cone"),
            _Actor(
                9,
                (-20.0, 10.0, 0.0),
                90.0,
                type_id="walker.pedestrian.0001",
                attributes={"role_name": ""},
                extent=(0.0, 0.01),
                velocity=(0.0, 1.4),
                acceleration=None,
                yaw_rate=None,
            ),
            _Actor(11, (15.0, 5.0, -150.0), 0.0, type_id="vehicle.hidden"),
            _Actor(
                5,
                (40.0, 2.0, 0.0),
                180.0,
                type_id="vehicle.vespa",
                attributes=_Attributes(role_name=""),
            ),
            _Actor(
                4,
                (30.0, 2.0, 0.0),
                0.0,
                type_id="vehicle.bike",
                attributes={"role_name": "", "base_type": "bicycle"},
            ),
        ]
    )


def _state(
    actor_id: int,
    kind: str,
    x: float,
    y: float,
    yaw: float,
    vx: float,
    vy: float,
    ax: float,
    ay: float,
    yaw_rate: float,
    length: float,
    width: float,
) -> dict[str, Any]:
    return {
        "id": actor_id,
        "kind": kind,
        "x": x,
        "y": y,
        "yaw": yaw,
        "vx": vx,
        "vy": vy,
        "ax": ax,
        "ay": ay,
        "yaw_rate": yaw_rate,
        "length": length,
        "width": width,
    }


def _rotated(
    x: float, y: float, yaw_deg: float, off_x: float, off_y: float
) -> tuple[float, float]:
    yaw = math.radians(yaw_deg)
    cx = x + off_x * math.cos(yaw) - off_y * math.sin(yaw)
    cy = y + off_x * math.sin(yaw) + off_y * math.cos(yaw)
    return cx, -cy


def test_the_frame_is_pinned_field_for_field() -> None:
    frame = cr_frame.read_frame(_world(), radius_m=40.0)  # type: ignore[arg-type]
    assert frame is not None
    ego_x, ego_y = _rotated(10.0, 2.0, 30.0, 1.0, 0.5)
    assert frame.ego == _state(
        1,
        "car",
        ego_x,
        ego_y,
        -math.radians(30.0),
        8.0,
        -3.0,
        1.0,
        2.0,
        -math.radians(12.0),
        4.5,
        1.8,
    )
    truck_x, truck_y = _rotated(12.0, -4.0, -30.0, 0.5, -0.2)
    # The world's order, the ego, the sensor, the prop, the hidden car and
    # the one 90 m off left out.
    assert frame.others == [
        _state(
            3,
            "truck",
            truck_x,
            truck_y,
            math.radians(30.0),
            4.0,
            1.0,
            0.5,
            -0.25,
            math.radians(5.0),
            4.5,
            1.8,
        ),
        # Neither its acceleration nor its yaw rate could be read: zero; a
        # degenerate box: 0.1 m.
        _state(
            9,
            "pedestrian",
            -20.0,
            -10.0,
            -math.radians(90.0),
            0.0,
            -1.4,
            0.0,
            0.0,
            0.0,
            0.1,
            0.1,
        ),
        # No base_type to read: a car.
        _state(
            5,
            "car",
            40.0,
            -2.0,
            -math.radians(180.0),
            0.0,
            -0.0,
            0.0,
            -0.0,
            -0.0,
            4.5,
            1.8,
        ),
        _state(4, "bicycle", 30.0, -2.0, -0.0, 0.0, -0.0, 0.0, -0.0, -0.0, 4.5, 1.8),
    ]


def test_the_radius_is_inclusive_and_planar() -> None:
    # 30 m off in x and y, and a little up a slope: within a 30 m radius.
    ego = _Actor(1, (0.0, 0.0, 0.0), 0.0, attributes={"role_name": str(EGO_ROLE_NAME)})
    edge = _Actor(2, (30.0, 0.0, 2.0), 0.0)
    beyond = _Actor(3, (30.0, 0.001, 0.0), 0.0)
    frame = cr_frame.read_frame(_World([ego, edge, beyond]), 30.0)  # type: ignore[arg-type]
    assert frame is not None
    assert [o["id"] for o in frame.others] == [2]


def test_just_above_the_hidden_depth_is_in_the_scene() -> None:
    # The ego as deep, so that the two are on the same level of road.
    ego = _Actor(
        1, (0.0, 0.0, -100.0), 0.0, attributes={"role_name": str(EGO_ROLE_NAME)}
    )
    at = _Actor(2, (1.0, 0.0, -100.0), 0.0)
    below = _Actor(3, (1.0, 0.0, -100.001), 0.0)
    frame = cr_frame.read_frame(_World([ego, at, below]), 30.0)  # type: ignore[arg-type]
    assert frame is not None
    assert [o["id"] for o in frame.others] == [2]


def test_a_road_user_on_another_level_of_road_is_out_of_the_scene() -> None:
    """A road user on a bridge over the ego, or in an underpass beneath it, is
    on no lane of the ego's: the planar scenario would put it there (#1580).
    Within a metre of the ego's box in height, it is on the ego's level."""
    ego = _Actor(1, (0.0, 0.0, 0.0), 0.0, attributes={"role_name": str(EGO_ROLE_NAME)})
    over = _Actor(2, (5.0, 0.0, 8.0), 0.0)
    under = _Actor(3, (5.0, 0.0, -8.0), 0.0)
    up_a_slope = _Actor(4, (10.0, 0.0, 1.6 + 0.9), 0.0)
    beyond_the_margin = _Actor(5, (10.0, 0.0, 1.6 + 1.1), 0.0)
    frame = cr_frame.read_frame(  # type: ignore[arg-type]
        _World([ego, over, under, up_a_slope, beyond_the_margin]), 30.0
    )
    assert frame is not None
    assert [o["id"] for o in frame.others] == [4]


def test_a_road_user_up_a_slope_is_on_the_ego_s_level() -> None:
    """A lead 30 m ahead on a 10 % grade stands 3 m above the ego, its box
    clear of the ego's by 1.4 m: the same road climbing, not another level,
    while a road 8 m overhead 20 m away is (#1597)."""
    ego = _Actor(1, (0.0, 0.0, 0.0), 0.0, attributes={"role_name": str(EGO_ROLE_NAME)})
    lead = _Actor(2, (30.0, 0.0, 3.0), 0.0)
    overhead = _Actor(3, (20.0, 0.0, 8.0), 0.0)
    frame = cr_frame.read_frame(_World([ego, lead, overhead]), 100.0)  # type: ignore[arg-type]
    assert frame is not None
    assert [o["id"] for o in frame.others] == [2]


@pytest.mark.parametrize("radius", [0.0, 5.0])
def test_alone(radius: float) -> None:
    ego = _Actor(1, (0.0, 0.0, 0.0), 0.0, attributes={"role_name": str(EGO_ROLE_NAME)})
    frame = cr_frame.read_frame(_World([ego]), radius)  # type: ignore[arg-type]
    assert frame is not None and frame.others == []
