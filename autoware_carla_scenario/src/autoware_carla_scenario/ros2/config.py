"""Settings for the ego entity that talks to ROS 2 nodes over DDS.

:class:`Ros2EgoConfig` says what the entity publishes about the ego and which
topics it takes commands from; :class:`CommandSourceConfig` is one of those
command sources.  Neither imports a DDS implementation, so a config can be built
and validated where none is installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional, Tuple

from ..constants import US_PER_S
from ..sensor.carla_camera import CarlaCameraSensorConfig

__all__ = [
    "COMMAND_KINDS",
    "ON_STALE_ACTIONS",
    "CommandSourceConfig",
    "Ros2CameraConfig",
    "Ros2EgoConfig",
]

#: What a command source carries:
#:
#: * ``"trajectory"`` -- ``autoware_planning_msgs/msg/Trajectory`` on
#:   :attr:`CommandSourceConfig.topic`, tracked by the framework's own
#:   trajectory follower.
#: * ``"steer_accel"`` -- two ``std_msgs/msg/Float64`` topics, a steering angle
#:   and an acceleration, as VisionPilot publishes them; handed to CARLA's
#:   Ackermann controller.
#: * ``"control"`` -- ``autoware_control_msgs/msg/Control`` on
#:   :attr:`CommandSourceConfig.topic`; handed to CARLA's Ackermann controller.
COMMAND_KINDS: Tuple[str, ...] = ("trajectory", "steer_accel", "control")

#: What the entity does once every source has gone stale:
#:
#: * ``"hold"`` -- keep applying the last command, as a ROS 2 node that relays
#:   commands to CARLA does when its upstream stops.
#: * ``"brake"`` -- brake to a stop.
ON_STALE_ACTIONS: Tuple[str, ...] = ("hold", "brake")


@dataclass(frozen=True)
class CommandSourceConfig:
    """One ROS 2 publisher the ego takes commands from.

    Sources are arbitrated every tick: of those whose last command arrived at
    most :attr:`stale_s` ago, the one with the highest :attr:`priority` drives.

    Attributes:
        name: The source's name in logs and :attr:`Ros2EgoEntity.active_source`.
        kind: One of :data:`COMMAND_KINDS`.
        topic: The command topic, for ``trajectory`` and ``control``.
        steering_topic: The steering topic, for ``steer_accel``.
        acceleration_topic: The acceleration topic, for ``steer_accel``.
        priority: Higher wins among the fresh sources.
        stale_s: Simulated seconds after its last command that a source stops
            counting as fresh.
        steering_sign: Multiplies a ``steer_accel`` steering angle before it
            reaches CARLA, whose Ackermann steer is positive to the right.
            VisionPilot already steers positive to the right, so ``1.0``.
        speed_horizon_s: A ``steer_accel`` acceleration becomes the Ackermann
            target speed ``speed + acceleration * speed_horizon_s``, as
            VisionPilot's ``carla_control_publisher`` builds it.
        jerk_mps3: The Ackermann jerk limit, for ``steer_accel`` and for a
            ``control`` command that leaves its jerk undefined.
    """

    name: str
    kind: str
    topic: str = ""
    steering_topic: str = ""
    acceleration_topic: str = ""
    priority: int = 0
    stale_s: float = 0.3
    steering_sign: float = 1.0
    speed_horizon_s: float = 2.0
    jerk_mps3: float = 10.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a command source needs a name")
        if self.kind not in COMMAND_KINDS:
            raise ValueError(
                f"command source {self.name!r}: unknown kind {self.kind!r}, "
                f"expected one of {COMMAND_KINDS}"
            )
        if self.kind == "steer_accel":
            if not self.steering_topic or not self.acceleration_topic:
                raise ValueError(
                    f"command source {self.name!r}: steer_accel needs "
                    "steering_topic and acceleration_topic"
                )
        elif not self.topic:
            raise ValueError(f"command source {self.name!r}: {self.kind} needs a topic")
        if self.stale_s <= 0.0:
            raise ValueError(
                f"command source {self.name!r}: stale_s must be positive, "
                f"got {self.stale_s}"
            )

    @property
    def topics(self) -> Tuple[str, ...]:
        """Every topic this source reads."""
        # tuple() of a list: the check reads it as a list, so both returns
        # have one type (docs/typecheck.md).
        if self.kind == "steer_accel":
            return tuple([self.steering_topic, self.acceleration_topic])
        return tuple([self.topic])

    @property
    def stale_us(self) -> int:
        """:attr:`stale_s` in microseconds, the clock commands are timed on."""
        return int(round(self.stale_s * US_PER_S))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "CommandSourceConfig":
        """Build from a plain mapping (e.g. a Hydra node).

        Raises:
            ValueError: If *mapping* holds a key this config does not define.
        """
        from ..utils.config import checked_options  # noqa: PLC0415

        return cls(**checked_options(cls, mapping))


@dataclass(frozen=True)
class Ros2CameraConfig:
    """A camera on the ego whose frames the entity publishes itself.

    For a CARLA server built without ROS 2, which publishes no camera of its
    own.  Each frame goes out as a ``sensor_msgs/msg/Image`` in ``bgra8``, stamped
    with its simulation time, as CARLA's own ROS 2 interface publishes it.  The
    rendering and mount fields are a driver camera's
    (:class:`~..driver.base.DriverCameraConfig`): CARLA's convention, x forward,
    y right, z up, degrees.

    Attributes:
        topic: The image topic.  CARLA's own interface names it
            ``/carla/<role name>/<sensor ros_name>/image``.
        frame_id: ``header.frame_id`` of the published images; the
            :attr:`logical_id` when empty.
        logical_id: The camera's name in logs.
    """

    topic: str = ""
    frame_id: str = ""
    logical_id: str = "camera"
    image_width: int = 960
    image_height: int = 604
    fov: float = 120.0
    fps: float = 20.0
    position_x: float = 1.5
    position_y: float = 0.0
    position_z: float = 1.6
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    def __post_init__(self) -> None:
        if not self.topic:
            raise ValueError(f"camera {self.logical_id!r} needs a topic")
        if self.image_width <= 0 or self.image_height <= 0 or self.fps <= 0.0:
            raise ValueError(
                f"camera {self.logical_id!r}: image size and fps must be positive"
            )

    @property
    def header_frame_id(self) -> str:
        """The ``header.frame_id`` its images carry."""
        return self.frame_id if self.frame_id else self.logical_id

    def to_sensor_config(self) -> CarlaCameraSensorConfig:
        """The CARLA sensor configuration of this camera."""
        return CarlaCameraSensorConfig(
            image_width=self.image_width,
            image_height=self.image_height,
            fov=self.fov,
            fps=self.fps,
            position_x=self.position_x,
            position_y=self.position_y,
            position_z=self.position_z,
            roll=self.roll,
            pitch=self.pitch,
            yaw=self.yaw,
        )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "Ros2CameraConfig":
        """Build from a plain mapping (e.g. a Hydra node).

        Raises:
            ValueError: If *mapping* holds a key this config does not define.
        """
        from ..utils.config import checked_options  # noqa: PLC0415

        return cls(**checked_options(cls, mapping))


def _no_cameras() -> Tuple[Ros2CameraConfig, ...]:
    # tuple() of a typed list, as _default_sources: the check reads a
    # variable-length tuple field as a list (docs/typecheck.md).
    cameras: List[Ros2CameraConfig] = []
    return tuple(cameras)


def _default_sources() -> Tuple[CommandSourceConfig, ...]:
    # tuple() of a list, not a display: the check reads a variable-length
    # tuple field as a list (docs/typecheck.md).
    return tuple(
        [
            CommandSourceConfig(
                name="planner",
                kind="trajectory",
                topic="/planning/scenario_planning/trajectory",
            )
        ]
    )


@dataclass(frozen=True)
class Ros2EgoConfig:
    """How the ego meets ROS 2 nodes over DDS.

    Topics are ROS 2 names (``/vehicle/speed``); ``None`` leaves that one
    unpublished.  Poses are published in the right-handed local frame -- CARLA's
    world with ``y`` flipped, as CARLA's own ROS 2 interface publishes it -- with
    the origin on the rear axle, Autoware's ``base_link``.

    Attributes:
        domain_id: The DDS domain (``ROS_DOMAIN_ID``).
        frame_id: ``header.frame_id`` of the published odometry.
        child_frame_id: Its ``child_frame_id``.
        kinematic_state_topic: ``nav_msgs/msg/Odometry`` of the ego.
        steering_status_topic: ``autoware_vehicle_msgs/msg/SteeringReport``.
        speed_topic: ``std_msgs/msg/Float64``, the ground speed in m/s.
        clock_topic: ``rosgraph_msgs/msg/Clock``, for a CARLA server built
            without ROS 2, whose own ``/clock`` is then missing.
        turn_indicators_command_topic: ``autoware_vehicle_msgs/msg/TurnIndicatorsCommand``
            the ego's turn indicators are lit from, whichever source drives;
            ``None`` leaves them to the scenario.
        turn_indicators_report_topic: ``autoware_vehicle_msgs/msg/TurnIndicatorsReport``
            of the indicators the ego shows, as a vehicle interface reports them.
        rear_axle_offset_m: Offset from the actor origin back to ``base_link``;
            ``None`` derives it from the wheel physics.
        on_stale: One of :data:`ON_STALE_ACTIONS`.
        sources: The command sources, see :class:`CommandSourceConfig`.
        cameras: Cameras whose frames the entity publishes, for a CARLA server
            built without ROS 2; see :class:`Ros2CameraConfig`.
    """

    domain_id: int = 0
    frame_id: str = "map"
    child_frame_id: str = "base_link"
    kinematic_state_topic: Optional[str] = "/localization/kinematic_state"
    steering_status_topic: Optional[str] = "/vehicle/status/steering_status"
    speed_topic: Optional[str] = "/vehicle/speed"
    clock_topic: Optional[str] = None
    turn_indicators_command_topic: Optional[str] = (
        "/control/command/turn_indicators_cmd"
    )
    turn_indicators_report_topic: Optional[str] = (
        "/vehicle/status/turn_indicators_status"
    )
    rear_axle_offset_m: Optional[float] = None
    on_stale: str = "hold"
    sources: Tuple[CommandSourceConfig, ...] = field(default_factory=_default_sources)
    cameras: Tuple[Ros2CameraConfig, ...] = field(default_factory=_no_cameras)

    def __post_init__(self) -> None:
        if self.on_stale not in ON_STALE_ACTIONS:
            raise ValueError(
                f"unknown on_stale {self.on_stale!r}, expected one of "
                f"{ON_STALE_ACTIONS}"
            )
        if not self.sources:
            raise ValueError("the ego needs at least one command source")
        names = [source.name for source in self.sources]
        if len(set(names)) != len(names):
            raise ValueError(f"command source names must be unique, got {names}")
        topics = [camera.topic for camera in self.cameras]
        if len(set(topics)) != len(topics):
            raise ValueError(f"camera topics must be unique, got {topics}")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "Ros2EgoConfig":
        """Build from a plain mapping (e.g. a Hydra node).

        A ``control`` entry, if present, is ignored: the trajectory follower is
        configured separately, through
        :meth:`~autoware_carla_scenario.driver.control.ControlConfig.from_mapping`.

        Raises:
            ValueError: If *mapping* holds a key this config does not define.
        """
        from ..utils.config import checked_options  # noqa: PLC0415

        values = checked_options(cls, mapping, ignore={"control"})
        sources = values.pop("sources", None)
        if sources is not None:
            values["sources"] = tuple(
                CommandSourceConfig.from_mapping(source) for source in sources
            )
        cameras = values.pop("cameras", None)
        if cameras is not None:
            values["cameras"] = tuple(
                Ros2CameraConfig.from_mapping(camera) for camera in cameras
            )
        return cls(**values)
