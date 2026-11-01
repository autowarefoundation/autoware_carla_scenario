"""The HILS bridge (``scenario-hils``): signals, configuration, measurements, and
the bridge itself against a fake CARLA world, its devices on hilspy's
in-process ``mem://`` bus."""

from __future__ import annotations

import itertools
import math
import threading
import time
import types
from pathlib import Path
from typing import Any, Callable, List, Optional

import numpy as np
import pytest

from autoware_carla_scenario.hils import (
    SIGNALS,
    EgoState,
    HilsConfigError,
    ego_signals,
    load_config,
    parse_config,
)
from autoware_carla_scenario.hils.cli import example_config, main
from autoware_carla_scenario.hils.measurements import (
    LidarRevolution,
    camera_frame,
    lidar_scan_fields,
)

from ._fake_world import ActorsById, FakeActorList

hilspy = pytest.importorskip("hilspy")

_bus = itertools.count()


def _mem() -> str:
    """A fresh in-process bus, so tests running together never share one."""
    return f"mem://hils-test-{next(_bus)}"


# -- signals --------------------------------------------------------------------


def _state(**kwargs: Any) -> EgoState:
    defaults: dict[str, Any] = {
        "time": 1.0,
        "frame": 20,
        "location": (0.0, 0.0, 0.0),
        "rotation": (0.0, 0.0, 0.0),
    }
    return EgoState(**{**defaults, **kwargs})


def test_signals_cover_every_documented_name() -> None:
    assert set(ego_signals(_state())) == set(SIGNALS)


def test_driving_east_is_forward_and_a_course_of_90() -> None:
    s = ego_signals(_state(velocity=(10.0, 0.0, 0.0), acceleration=(1.0, 0.0, 0.0)))
    assert s["speed"] == pytest.approx(10.0)
    assert s["speed_kmh"] == pytest.approx(36.0)
    assert s["longitudinal_velocity"] == pytest.approx(10.0)
    assert s["lateral_velocity"] == pytest.approx(0.0)
    assert s["longitudinal_acceleration"] == pytest.approx(1.0)
    assert s["velocity_east"] == pytest.approx(10.0)
    assert s["velocity_north"] == pytest.approx(0.0)
    assert s["course"] == pytest.approx(90.0)
    assert s["heading"] == pytest.approx(90.0)


def test_the_vehicle_frame_follows_the_yaw_and_has_y_left() -> None:
    # Facing south (CARLA yaw 90: world +y), moving south and drifting east.
    s = ego_signals(_state(rotation=(0.0, 90.0, 0.0), velocity=(1.0, 5.0, 0.0)))
    assert s["heading"] == pytest.approx(180.0)
    assert s["longitudinal_velocity"] == pytest.approx(5.0)
    # East is the vehicle's left when it faces south.
    assert s["lateral_velocity"] == pytest.approx(1.0)
    assert s["velocity_north"] == pytest.approx(-5.0)


def test_turning_right_is_a_negative_yaw_rate() -> None:
    s = ego_signals(_state(angular_velocity=(0.0, 0.0, 10.0)))
    assert s["yaw_rate"] == pytest.approx(-10.0)
    assert s["yaw_rate_rad"] == pytest.approx(math.radians(-10.0))


def test_geolocation_and_controls() -> None:
    s = ego_signals(
        _state(geolocation=(35.5, 139.5, 40.0), throttle=0.3, reverse=True, gear=2)
    )
    assert s["latitude_rad"] == pytest.approx(math.radians(35.5))
    assert s["altitude"] == 40.0
    assert (s["throttle"], s["reverse"], s["gear"]) == (0.3, 1.0, 2.0)


# -- configuration --------------------------------------------------------------


def _device(**kwargs: Any) -> dict[str, Any]:
    return {"devices": {"d": {"url": "mem://x", **kwargs}}}


def _messages(fields: dict[str, Any]) -> dict[str, Any]:
    return _device(codec="nmea", messages=[{"name": "GGA", "fields": fields}])


def test_the_example_configuration_parses() -> None:
    config = parse_config(__import__("yaml").safe_load(example_config()))
    assert [d.name for d in config.devices] == [
        "vehicle_can",
        "gnss",
        "lidar_top",
        "front_camera",
    ]
    assert config.clock == "simulation"
    assert config.role_name == "Ego"
    lidar = config.devices[2].sensor
    assert lidar is not None and lidar.is_lidar
    assert lidar.attributes["rotation_frequency"] == "10"


def test_fields_are_constants_signals_or_scaled_signals() -> None:
    config = parse_config(
        _messages(
            {
                "a": 4,
                "b": "D",
                "c": "$speed",
                "d": {"signal": "speed", "scale": 2.0, "offset": 1.0},
                "e": "$$speed",
            }
        )
    )
    values = config.devices[0].messages[0].values({"speed": 3.0})
    assert values == {"a": 4, "b": "D", "c": 3.0, "d": 7.0, "e": "$speed"}
    assert config.devices[0].messages[0].period is None


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ({"devices": {}}, "no device"),
        ({"device": {}}, "unknown key"),
        (_messages({"a": "$nope"}), "unknown signal 'nope'"),
        (_messages({"a": {"scale": 2}}), "names a signal"),
        (
            _device(
                codec="raw",
                sensor={"blueprint": "sensor.camera.rgb"},
                messages=[{"name": "M"}],
            ),
            "has no messages or inputs",
        ),
        (
            _device(
                codec="raw",
                sensor={"blueprint": "sensor.lidar.ray_cast"},
            ),
            "lidar:MODEL codec",
        ),
        (
            _device(codec="sbf", sensor={"blueprint": "sensor.camera.rgb"}),
            "sent raw",
        ),
        (
            _device(sensor={"blueprint": "sensor.camera.rgb", "encoding": "gif"}),
            "'gif'",
        ),
        (_device(sensor={"blueprint": "sensor.other.gnss"}), "neither"),
        (
            _device(sensor={"blueprint": "sensor.camera.rgb", "transform": {"w": 1}}),
            "unknown key",
        ),
        ({**_messages({"a": 1}), "clock": "fast"}, "'fast'"),
        # The CARLA port is an integer port number, never truncated or wrapped.
        ({**_messages({}), "carla": {"port": 2000.7}}, "carla.port: expected an"),
        ({**_messages({}), "carla": {"port": True}}, "carla.port: expected an"),
        ({**_messages({}), "carla": {"port": "2000"}}, "carla.port: expected an"),
        ({**_messages({}), "carla": {"port": math.inf}}, "carla.port: expected an"),
        ({**_messages({}), "carla": {"port": math.nan}}, "carla.port: expected an"),
        ({**_messages({}), "carla": {"port": -5}}, "carla.port: must be within"),
        ({**_messages({}), "carla": {"port": 0}}, "carla.port: must be within"),
        ({**_messages({}), "carla": {"port": 70000}}, "carla.port: must be within"),
        # Every duration is a finite number of the right sign.
        ({**_messages({}), "carla": {"timeout": -1.0}}, "carla.timeout: must be"),
        ({**_messages({}), "carla": {"timeout": 0}}, "carla.timeout: must be"),
        ({**_messages({}), "carla": {"timeout": math.nan}}, "carla.timeout: .*finite"),
        ({**_messages({}), "carla": {"timeout": math.inf}}, "carla.timeout: .*finite"),
        ({**_messages({}), "ego": {"wait": -3.0}}, "ego.wait: must not be negative"),
        ({**_messages({}), "ego": {"wait": math.inf}}, "ego.wait: .*finite"),
        ({**_messages({}), "ego": {"wait": math.nan}}, "ego.wait: .*finite"),
        (
            _device(codec="nmea", messages=[{"name": "GGA", "period": math.nan}]),
            r"messages\[0\].period: .*finite",
        ),
        (
            _device(codec="nmea", messages=[{"name": "GGA", "period": math.inf}]),
            r"messages\[0\].period: .*finite",
        ),
        (
            _device(codec="nmea", messages=[{"name": "GGA", "period": 0.0}]),
            r"messages\[0\].period: must be positive",
        ),
        (
            {**_messages({}), "lifecycle": {"startup": [{"sleep": math.nan}]}},
            r"startup\[0\].sleep: .*finite",
        ),
        (
            {**_messages({}), "lifecycle": {"startup": [{"sleep": -0.5}]}},
            r"startup\[0\].sleep: must not be negative",
        ),
        (
            {
                **_messages({}),
                "lifecycle": {
                    "startup": [{"wait": "d", "name": "M", "timeout": math.nan}]
                },
            },
            r"startup\[0\].timeout: .*finite",
        ),
        (
            {
                **_messages({}),
                "lifecycle": {"startup": [{"wait": "d", "name": "M", "timeout": 0}]},
            },
            r"startup\[0\].timeout: must be positive",
        ),
        # The other numbers are finite too.
        (_messages({"a": {"signal": "speed", "scale": math.nan}}), "scale: .*finite"),
        (_messages({"a": {"signal": "speed", "offset": math.inf}}), "offset: .*finite"),
        (
            _device(
                codec="raw",
                sensor={"blueprint": "sensor.camera.rgb", "transform": {"z": math.nan}},
            ),
            "transform.z: .*finite",
        ),
        (
            _device(
                codec="lidar:vlp16",
                sensor={
                    "blueprint": "sensor.lidar.ray_cast",
                    "intensity_scale": math.inf,
                },
            ),
            "intensity_scale: .*finite",
        ),
    ],
)
def test_a_bad_configuration_is_refused(document: Any, message: str) -> None:
    with pytest.raises(HilsConfigError, match=message):
        parse_config(document)


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("carla: {port: .inf}", "carla.port: expected an"),
        ("carla: {port: .nan}", "carla.port: expected an"),
        ("carla: {port: 2000.7}", "carla.port: expected an"),
        ("carla: {port: 70000}", "carla.port: must be within"),
        ("carla: {timeout: .nan}", "carla.timeout: .*finite"),
        ("carla: {timeout: -.inf}", "carla.timeout: .*finite"),
        ("ego: {wait: .inf}", "ego.wait: .*finite"),
        ("lifecycle: {startup: [{sleep: .nan}]}", "sleep: .*finite"),
        (
            "lifecycle: {startup: [{wait: d, name: M, timeout: .inf}]}",
            "timeout: .*finite",
        ),
    ],
)
def test_a_bad_number_in_the_file_is_refused(
    tmp_path: Path, line: str, message: str
) -> None:
    path = tmp_path / "hils.yaml"
    path.write_text(f"{line}\ndevices: {{d: {{url: mem://x, codec: nmea}}}}\n")
    with pytest.raises(HilsConfigError, match=message):
        load_config(path)


def test_the_port_and_durations_in_range_are_read() -> None:
    config = parse_config(
        {
            **_messages({}),
            "carla": {"port": 65535, "timeout": 2},
            "ego": {"wait": 0},
            "lifecycle": {"startup": [{"sleep": 0}]},
        }
    )
    assert (config.port, config.timeout, config.ego_wait) == (65535, 2.0, 0.0)
    assert config.steps("startup")[0].seconds == 0.0


def test_the_command_refuses_a_bad_port_override(tmp_path: Path) -> None:
    path = tmp_path / "hils.yaml"
    path.write_text(example_config())
    for port in ("0", "70000", "2000.7"):
        with pytest.raises(SystemExit):
            main([str(path), "--check", "--port", port])


def test_relative_files_are_relative_to_the_configuration(tmp_path: Path) -> None:
    (tmp_path / "vehicle.dbc").write_text('VERSION ""\n')
    (tmp_path / "canvas.yaml").write_text("renderer: {}\n")
    path = tmp_path / "hils.yaml"
    path.write_text(
        "canvas: canvas.yaml\n"
        "devices:\n"
        "  can:\n"
        "    url: can://vcan0\n"
        "    codec: dbc:vehicle.dbc\n"
        "    messages: [{name: M, fields: {S: $speed}}]\n"
        "  lidar:\n"
        "    url: udp://127.0.0.1:2368\n"
        "    codec: lidar:vlp16\n"
        "    sensor: {blueprint: sensor.lidar.ray_cast}\n"
    )
    config = load_config(path)
    assert config.canvas == tmp_path / "canvas.yaml"
    assert config.devices[0].codec == f"dbc:{tmp_path / 'vehicle.dbc'}"
    # Not a file: a model name stays as it is.
    assert config.devices[1].codec == "lidar:vlp16"


def test_a_missing_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(HilsConfigError, match="not a file"):
        load_config(tmp_path / "absent.yaml")


# -- measurements ---------------------------------------------------------------


def test_a_lidar_scan_has_y_left_and_byte_intensities() -> None:
    points = np.array([[1.0, 2.0, 3.0, 0.5], [4.0, -5.0, 6.0, 2.0]], np.float32)
    fields = lidar_scan_fields(points)
    np.testing.assert_array_equal(fields["x"], [1.0, 4.0])
    np.testing.assert_array_equal(fields["y"], [-2.0, 5.0])
    np.testing.assert_array_equal(fields["z"], [3.0, 6.0])
    np.testing.assert_array_equal(fields["intensity"], [128, 255])
    assert fields["intensity"].dtype == np.uint8


def _points(n: int) -> np.ndarray:
    return np.full((n, 4), float(n), np.float32)


def test_arcs_gather_into_revolutions() -> None:
    # 10 Hz at 20 Hz ticks: half a revolution a tick.
    revolution = LidarRevolution(10.0, 0.05)
    assert revolution.add(0.05, _points(1)) is None
    whole = revolution.add(0.10, _points(2))
    assert whole is not None and len(whole) == 3
    assert revolution.add(0.15, _points(4)) is None
    whole = revolution.add(0.20, _points(8))
    assert whole is not None and len(whole) == 12


def test_a_lidar_turning_once_a_measurement_yields_each() -> None:
    # One revolution a tick, measured every other tick (sensor_tick).
    revolution = LidarRevolution(20.0, 0.05)
    for i in range(1, 4):
        whole = revolution.add(0.1 * i, _points(i))
        assert whole is not None and len(whole) == i


def test_a_restarted_world_clock_starts_a_new_revolution() -> None:
    revolution = LidarRevolution(10.0, 0.05)
    assert revolution.add(5.0, _points(1)) is None
    assert revolution.add(0.05, _points(2)) is None  # a new episode
    whole = revolution.add(0.10, _points(4))
    assert whole is not None and len(whole) == 6


def test_camera_frames_in_every_encoding() -> None:
    bgra = np.zeros((2, 3, 4), np.uint8)
    bgra[..., 0], bgra[..., 1], bgra[..., 2], bgra[..., 3] = 10, 20, 30, 255
    raw = bgra.tobytes()
    assert np.asarray(camera_frame(raw, 3, 2, "bgra")).shape == (2, 3, 4)
    assert list(np.asarray(camera_frame(raw, 3, 2, "bgr"))[0, 0]) == [10, 20, 30]
    assert list(np.asarray(camera_frame(raw, 3, 2, "rgb"))[0, 0]) == [30, 20, 10]
    import cv2

    for encoding in ("jpeg", "png"):
        data = camera_frame(raw, 3, 2, encoding)
        assert isinstance(data, bytes)
        decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        assert decoded.shape == (2, 3, 3)
    with pytest.raises(ValueError, match="unknown camera encoding"):
        camera_frame(raw, 3, 2, "gif")


# -- the bridge, against a fake CARLA ---------------------------------------------


class _Vector:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        self.x, self.y, self.z = x, y, z


class _Rotation:
    def __init__(self, pitch: float = 0.0, yaw: float = 0.0, roll: float = 0.0) -> None:
        self.pitch, self.yaw, self.roll = pitch, yaw, roll


class _Transform:
    def __init__(self, location: Any, rotation: Any) -> None:
        self.location, self.rotation = location, rotation


class _Control:
    throttle, brake, steer, hand_brake, reverse, gear = 0.5, 0.0, 0.1, False, False, 1


class _Ego:
    type_id = "vehicle.test"

    def __init__(self, actor_id: int, role: str = "Ego") -> None:
        self.id = actor_id
        self.attributes = {"role_name": role}
        self.is_alive = True
        self.controls: List[Any] = []

    def get_control(self) -> _Control:
        return _Control()

    def apply_control(self, control: Any) -> None:
        self.controls.append(control)

    def apply_ackermann_control(self, control: Any) -> None:
        self.controls.append(control)


class _Blueprint:
    def __init__(self, blueprint_id: str) -> None:
        self.id = blueprint_id
        self.attributes: dict[str, str] = {"role_name": ""}

    def has_attribute(self, name: str) -> bool:
        return name in self.attributes

    def set_attribute(self, name: str, value: str) -> None:
        self.attributes[name] = value


class _Sensor:
    def __init__(self, blueprint: _Blueprint, parent: _Ego, transform: Any) -> None:
        self.attributes = dict(blueprint.attributes)
        self.blueprint = blueprint.id
        self.parent = parent
        self.transform = transform
        self.callback: Optional[Callable[[Any], None]] = None
        self.is_alive = True

    def listen(self, callback: Callable[[Any], None]) -> None:
        self.callback = callback

    def stop(self) -> None:
        self.callback = None

    def destroy(self) -> None:
        self.is_alive = False


class _ActorSnapshot:
    def __init__(self, x: float, speed: float) -> None:
        self._x, self._speed = x, speed

    def get_transform(self) -> _Transform:
        return _Transform(_Vector(self._x, 0.0, 0.0), _Rotation())

    def get_velocity(self) -> _Vector:
        return _Vector(self._speed)

    def get_acceleration(self) -> _Vector:
        return _Vector()

    def get_angular_velocity(self) -> _Vector:
        return _Vector()


class _Snapshot:
    def __init__(self, frame: int, elapsed: float, actors: dict[int, Any]) -> None:
        self.frame = frame
        self.delta_seconds = 0.05
        self.timestamp = types.SimpleNamespace(elapsed_seconds=elapsed)
        self._actors = actors

    def find(self, actor_id: int) -> Optional[Any]:
        return self._actors.get(actor_id)


class _Map:
    def transform_to_geolocation(self, location: Any) -> Any:
        return types.SimpleNamespace(
            latitude=35.0 + location.x * 1e-5, longitude=139.0, altitude=10.0
        )


class _World(ActorsById):
    """Ticks on demand: :meth:`tick` queues a snapshot the bridge then waits for."""

    def __init__(self) -> None:
        self.actors = FakeActorList()
        self.sensors: List[_Sensor] = []
        self.pending: List[_Snapshot] = []
        self.frame = 0
        self.speed = 10.0
        self._lock = threading.Lock()

    def get_actors(self) -> FakeActorList:
        return FakeActorList(a for a in self.actors if a.is_alive)

    def get_map(self) -> _Map:
        return _Map()

    def get_blueprint_library(self) -> Any:
        return types.SimpleNamespace(find=_Blueprint)

    def spawn_actor(
        self, blueprint: _Blueprint, transform: Any, attach_to: Any
    ) -> _Sensor:
        sensor = _Sensor(blueprint, attach_to, transform)
        self.sensors.append(sensor)
        return sensor

    def get_snapshot(self) -> _Snapshot:
        elapsed = self.frame * 0.05
        actors = {
            a.id: _ActorSnapshot(self.speed * elapsed, self.speed)
            for a in self.actors
            if a.is_alive
        }
        return _Snapshot(self.frame, elapsed, actors)

    def tick(self) -> None:
        with self._lock:
            self.frame += 1
            self.pending.append(self.get_snapshot())

    def wait_for_tick(self, timeout: float) -> _Snapshot:
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                if self.pending:
                    return self.pending.pop(0)
            if time.monotonic() >= deadline:
                raise RuntimeError("time-out while waiting for the simulator")
            time.sleep(0.001)


def _carla(world: _World) -> types.SimpleNamespace:
    class Client:
        def __init__(self, host: str, port: int) -> None:
            self.address = (host, port)

        def set_timeout(self, timeout: float) -> None:
            pass

        def get_world(self) -> _World:
            return world

    return types.SimpleNamespace(
        Client=Client, Transform=_Transform, Location=_Vector, Rotation=_Rotation
    )


def _drain(channel: Any) -> List[Any]:
    return list(iter(lambda: channel.recv(0.05), None))


@pytest.fixture
def world() -> _World:
    w = _World()
    w.actors.append(_Ego(1))
    return w


def _bridge(document: dict[str, Any], world: _World) -> Any:
    from autoware_carla_scenario.hils.bridge import HilsBridge

    bridge = HilsBridge(parse_config(document), carla_module=_carla(world))
    bridge.connect()
    bridge.open_devices()
    return bridge


def test_periodic_messages_follow_the_ego_on_simulated_time(world: _World) -> None:
    url = _mem()
    receiver = hilspy.open(url, "nmea")
    bridge = _bridge(
        {
            "devices": {
                "gps": {
                    "url": url,
                    "codec": "nmea",
                    "messages": [
                        {
                            "name": "GGA",
                            "period": 0.1,
                            "fields": {
                                "latitude": "$latitude",
                                "longitude": "$longitude",
                                "fix_type": "Rtk",
                            },
                        }
                    ],
                }
            }
        },
        world,
    )
    try:
        world.tick()
        assert bridge.step()
        assert bridge.ego is not None and bridge.ego.id == 1
        # The first update starts the message: sent at once.
        first = _drain(receiver)
        assert [m.name for m in first] == ["GGA"]
        assert first[0]["latitude"] == pytest.approx(35.0 + 0.5e-5)
        for _ in range(4):  # 0.2 s more of simulated time: two more sends
            world.tick()
            assert bridge.step()
        later = _drain(receiver)
        assert len(later) == 2
        assert later[-1]["latitude"] > first[0]["latitude"]
        # No tick in time: nothing is sent, the bridge carries on.
        assert not bridge.step(0.01)
    finally:
        bridge.close()
        receiver.close()


def _lidar_measurement(time: float, n: int, half: int) -> Any:
    """*n* points on the *half* (0 or 1) of a circle one arc sweeps."""
    angles = np.linspace(half * np.pi, (half + 1) * np.pi, n, endpoint=False)
    points = np.zeros((n, 4), np.float32)
    points[:, 0] = 10.0 * np.cos(angles)
    points[:, 1] = 10.0 * np.sin(angles)
    points[:, 3] = 0.5
    return types.SimpleNamespace(timestamp=time, raw_data=points.tobytes())


def test_lidar_revolutions_go_out_as_the_sensors_packets(world: _World) -> None:
    url = _mem()
    receiver = hilspy.open(url, "lidar:vlp16", frame="packet")
    bridge = _bridge(
        {
            "devices": {
                "lidar": {
                    "url": url,
                    "codec": "lidar:vlp16",
                    "sensor": {
                        "blueprint": "sensor.lidar.ray_cast",
                        "transform": {"z": 2.0},
                        "attributes": {"rotation_frequency": 10, "channels": 16},
                    },
                }
            }
        },
        world,
    )
    try:
        world.tick()
        bridge.step()
        [sensor] = world.sensors
        assert sensor.parent.id == 1
        assert sensor.attributes["role_name"] == "lidar"
        assert sensor.attributes["channels"] == "16"
        assert sensor.transform.location.z == 2.0
        assert sensor.callback is not None
        # Half a revolution a tick: the first arc alone sends nothing.
        sensor.callback(_lidar_measurement(0.05, 500, 0))
        world.tick()
        bridge.step()
        assert _drain(receiver) == []
        sensor.callback(_lidar_measurement(0.10, 500, 1))
        for _ in range(4):  # the revolution's packets, paced over 100 ms
            world.tick()
            bridge.step()
        packets = _drain(receiver)
        assert packets and all(p.name == "scan" for p in packets)
        assert sum(len(p["x"]) for p in packets) == 1000
    finally:
        bridge.close()
        receiver.close()


def test_camera_frames_and_a_new_ego(world: _World) -> None:
    url = _mem()
    receiver = hilspy.open(url)
    bridge = _bridge(
        {
            "devices": {
                "camera": {
                    "url": url,
                    "sensor": {"blueprint": "sensor.camera.rgb", "encoding": "rgb"},
                }
            }
        },
        world,
    )
    try:
        world.tick()
        bridge.step()
        [sensor] = world.sensors
        image = np.zeros((2, 2, 4), np.uint8)
        image[..., 2] = 200  # red, in BGRA
        assert sensor.callback is not None
        sensor.callback(
            types.SimpleNamespace(
                timestamp=0.05, raw_data=image.tobytes(), width=2, height=2
            )
        )
        world.tick()
        bridge.step()
        [frame] = _drain(receiver)
        assert frame.data == bytes([200, 0, 0] * 4)

        # The runner's episode ends: the ego goes, and the sensor with it.
        world.actors[0].is_alive = False
        world.tick()
        bridge.step()
        assert bridge.ego is None
        assert not sensor.is_alive
        # The next episode's ego gets a sensor of its own.
        world.actors.append(_Ego(2))
        bridge._next_scan = 0.0
        world.tick()
        bridge.step()
        assert bridge.ego is not None and bridge.ego.id == 2
        assert [s.parent.id for s in world.sensors] == [1, 2]
    finally:
        bridge.close()
        receiver.close()
    assert not world.sensors[1].is_alive


def test_on_the_wall_clock_sensor_frames_go_out_from_a_thread(world: _World) -> None:
    url = _mem()
    receiver = hilspy.open(url)
    bridge = _bridge(
        {
            "clock": "wall",
            "devices": {
                "camera": {"url": url, "sensor": {"blueprint": "sensor.camera.rgb"}}
            },
        },
        world,
    )
    try:
        assert bridge.clock is None
        world.tick()
        bridge.step()
        [sensor] = world.sensors
        assert sensor.callback is not None
        image = np.arange(16, dtype=np.uint8).tobytes()
        sensor.callback(
            types.SimpleNamespace(timestamp=0.05, raw_data=image, width=2, height=2)
        )
        world.tick()
        bridge.step()
        frame = receiver.recv(2.0)
        assert frame is not None and frame.data == image
    finally:
        bridge.close()
        receiver.close()


def _camera(world: _World, clock: str = "simulation") -> tuple[Any, Any]:
    url = _mem()
    receiver = hilspy.open(url)
    bridge = _bridge(
        {
            "clock": clock,
            "devices": {
                "camera": {"url": url, "sensor": {"blueprint": "sensor.camera.rgb"}}
            },
        },
        world,
    )
    return bridge, receiver


def _frame(raw: bytes) -> Any:
    return types.SimpleNamespace(timestamp=0.05, raw_data=raw, width=2, height=2)


def test_a_frame_measured_as_the_scenario_ends_is_not_sent(world: _World) -> None:
    bridge, receiver = _camera(world)
    try:
        world.tick()
        bridge.step()
        [sensor] = world.sensors
        callback = sensor.callback
        assert callback is not None
        # The ego's last tick: CARLA hands its frame over after the bridge
        # flushed that tick.
        callback(_frame(bytes(16)))
        bridge._finish()
        world.tick()
        bridge.step()
        assert _drain(receiver) == []
    finally:
        bridge.close()
        receiver.close()


def test_after_a_new_ego_only_its_sensor_frames_are_sent(world: _World) -> None:
    bridge, receiver = _camera(world)
    try:
        world.tick()
        bridge.step()
        [old] = world.sensors
        old_callback = old.callback
        assert old_callback is not None
        old_callback(_frame(b"OLD-EGO-FRAME-01"))
        # The run's ``end`` and the next run's ``start``, before the next tick.
        bridge._finish()
        world.actors.append(_Ego(2))
        bridge._begin(world.actors[1], "next")
        new = world.sensors[1]
        assert new.callback is not None
        new.callback(_frame(b"NEW-EGO-FRAME-01"))
        # A callback of the old sensor still in CARLA's hands comes late.
        old_callback(_frame(b"OLD-EGO-FRAME-02"))
        world.tick()
        bridge.step()
        assert [m.data for m in _drain(receiver)] == [b"NEW-EGO-FRAME-01"]
    finally:
        bridge.close()
        receiver.close()


def test_a_callback_in_flight_while_the_sensor_goes_is_not_sent(
    world: _World,
) -> None:
    bridge, receiver = _camera(world)
    entered, release = threading.Event(), threading.Event()

    class _InFlight:
        timestamp, width, height = 0.05, 2, 2

        @property
        def raw_data(self) -> bytes:
            # CARLA's thread is inside the callback when the sensor goes, and
            # queues the frame only after detach drained the queue.
            entered.set()
            assert release.wait(5.0)
            return bytes(16)

    try:
        world.tick()
        bridge.step()
        [sensor] = world.sensors
        callback = sensor.callback
        assert callback is not None
        in_flight = _start(lambda: callback(_InFlight()))
        assert entered.wait(5.0)
        bridge._finish()
        release.set()
        in_flight.join(5.0)
        assert not in_flight.is_alive()
        world.tick()
        bridge.step()
        assert _drain(receiver) == []
    finally:
        release.set()
        bridge.close()
        receiver.close()


@pytest.mark.parametrize("clock", ["simulation", "wall"])
def test_a_callback_while_the_sensor_stops_is_not_sent(
    world: _World, clock: str
) -> None:
    bridge, receiver = _camera(world, clock)
    try:
        world.tick()
        bridge.step()
        [sensor] = world.sensors
        callback = sensor.callback
        assert callback is not None
        stop = sensor.stop

        def stop_with_a_last_frame() -> None:
            callback(_frame(b"OLD-EGO-FRAME-01"))
            stop()

        sensor.stop = stop_with_a_last_frame  # type: ignore[method-assign]
        bridge._finish()
        world.tick()
        bridge.step()
        # The next scenario's frame is the first one sent (one thread sends
        # every frame of the device, in order, with the wall clock).
        world.actors.append(_Ego(2))
        bridge._begin(world.actors[1], "next")
        new = world.sensors[1]
        assert new.callback is not None
        new.callback(_frame(b"NEW-EGO-FRAME-01"))
        world.tick()
        bridge.step()
        frame = receiver.recv(2.0)
        assert frame is not None and frame.data == b"NEW-EGO-FRAME-01"
        assert _drain(receiver) == []
    finally:
        bridge.close()
        receiver.close()


def test_on_the_wall_clock_no_frame_is_sent_after_the_sensor_goes(
    world: _World,
) -> None:
    bridge, receiver = _camera(world, "wall")
    [device] = bridge._sensors
    entered, release, moved_on = (threading.Event() for _ in range(3))

    class _GatedChannel:
        """Holds the device's first send until released."""

        def __init__(self, inner: Any) -> None:
            self._inner = inner
            self._first = True

        def send(self, *args: Any) -> None:
            if self._first:
                self._first = False
                entered.set()
                assert release.wait(5.0)
            self._inner.send(*args)

    device.channel = _GatedChannel(device.channel)
    next_generation = device._next_generation

    def signalled_next_generation() -> int:
        generation = next_generation()
        moved_on.set()
        return generation

    try:
        world.tick()
        bridge.step()
        [sensor] = world.sensors
        assert sensor.callback is not None
        sensor.callback(_frame(b"OLD-EGO-FRAME-01"))
        world.tick()
        bridge.step()
        # The device's thread is sending the first frame ...
        assert entered.wait(5.0)
        # ... while the second waits in the outbox as the scenario ends.
        sensor.callback(_frame(b"OLD-EGO-FRAME-02"))
        world.tick()
        bridge.step()
        device._next_generation = signalled_next_generation
        finishing = _start(bridge._finish)
        assert moved_on.wait(5.0)
        device._next_generation = next_generation
        # The send under way finishes before the scenario has ended.
        release.set()
        finishing.join(5.0)
        assert not finishing.is_alive()
        world.actors.append(_Ego(2))
        bridge._begin(world.actors[1], "next")
        new = world.sensors[1]
        assert new.callback is not None
        new.callback(_frame(b"NEW-EGO-FRAME-01"))
        world.tick()
        bridge.step()
        sent = [receiver.recv(2.0), receiver.recv(2.0)]
        assert [m.data for m in sent if m is not None] == [
            b"OLD-EGO-FRAME-01",
            b"NEW-EGO-FRAME-01",
        ]
        assert _drain(receiver) == []
    finally:
        release.set()
        bridge.close()
        receiver.close()


def test_a_send_that_hangs_does_not_keep_the_scenario_from_ending(
    world: _World, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from autoware_carla_scenario.hils import bridge as bridge_module

    monkeypatch.setattr(bridge_module, "_SEND_WAIT_S", 0.05)
    bridge, receiver = _camera(world, "wall")
    [device] = bridge._sensors
    entered, release = threading.Event(), threading.Event()

    class _HungChannel:
        def send(self, *args: Any) -> None:
            entered.set()
            assert release.wait(5.0)

    device.channel = _HungChannel()
    try:
        world.tick()
        bridge.step()
        [sensor] = world.sensors
        assert sensor.callback is not None
        sensor.callback(_frame(bytes(16)))
        world.tick()
        bridge.step()
        assert entered.wait(5.0)
        with caplog.at_level("WARNING"):
            bridge._finish()
        assert not bridge.active
        assert any("still under way" in r.getMessage() for r in caplog.records)
    finally:
        release.set()
        bridge.close()
        receiver.close()


def test_run_waits_for_carla_and_returns_once_stopped(
    world: _World, monkeypatch: pytest.MonkeyPatch
) -> None:
    from autoware_carla_scenario.hils import bridge as bridge_module

    monkeypatch.setattr(bridge_module, "_CONNECT_RETRY_S", 0.0)
    carla = _carla(world)
    refusals = [RuntimeError("Connection refused")]
    connect = carla.Client.get_world

    def get_world(client: Any) -> _World:
        if refusals:
            raise refusals.pop()
        return connect(client)

    monkeypatch.setattr(carla.Client, "get_world", get_world)
    url = _mem()
    receiver = hilspy.open(url, "nmea")
    fields = {"latitude": "$latitude", "longitude": "$longitude", "fix_type": "Rtk"}
    gga = {"name": "GGA", "period": 0.1, "fields": fields}
    config = parse_config(
        {"devices": {"d": {"url": url, "codec": "nmea", "messages": [gga]}}}
    )
    bridge = bridge_module.HilsBridge(config, carla_module=carla)
    ticks = 3

    def wait_for_tick(timeout: float) -> _Snapshot:
        nonlocal ticks
        if ticks == 0:
            bridge.stop()
            raise RuntimeError("time-out")
        ticks -= 1
        world.tick()
        return world.pending.pop(0)

    monkeypatch.setattr(world, "wait_for_tick", wait_for_tick)
    try:
        bridge.run()  # refused once, then three ticks: 0.1 s of simulated time
        assert not refusals
        assert bridge.ego is not None
        assert [m.name for m in _drain(receiver)] == ["GGA", "GGA"]
    finally:
        bridge.close()
        receiver.close()


def test_a_canvas_configuration_needs_carla_canvas(
    world: _World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    from autoware_carla_scenario.canvas import CanvasUnavailableError

    canvas = tmp_path / "canvas.yaml"
    canvas.write_text("renderer: {}\n")
    monkeypatch.setitem(sys.modules, "carla_canvas", None)
    monkeypatch.setitem(sys.modules, "carla_canvas.plugin", None)
    with pytest.raises(CanvasUnavailableError, match="carla_canvas"):
        _bridge({"canvas": str(canvas), **_messages({"a": 1})}, world)


# -- the command ----------------------------------------------------------------


def test_the_command_checks_prints_and_refuses(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "hils.yaml"
    path.write_text(example_config())
    assert main([str(path), "--check", "--port", "3000"]) == 0
    out = capsys.readouterr().out
    assert "CARLA localhost:3000, ego 'Ego', simulation clock" in out
    assert "lidar_top: udp://192.168.1.201:2368 (lidar:pandar_xt32)" in out

    assert main(["--example"]) == 0
    assert capsys.readouterr().out == example_config()
    assert main(["--signals"]) == 0
    assert "speed_kmh" in capsys.readouterr().out

    assert main([]) == 2
    path.write_text("devices: {}\n")
    assert main([str(path)]) == 2
    assert "no device" in capsys.readouterr().err


# -- ECU to simulation, and the lifecycle ------------------------------------------

_DBC = """VERSION ""


NS_ :

BS_:

BU_: SIM ECU


BO_ 256 Wake: 1 SIM
 SG_ On : 0|1@1+ (1,0) [0|1] "" ECU

BO_ 257 VehicleSpeed: 2 SIM
 SG_ Speed : 0|16@1+ (0.01,0) [0|655.35] "km/h" ECU

BO_ 512 ControlCmd: 3 ECU
 SG_ Throttle : 0|8@1+ (0.5,0) [0|127.5] "%" SIM
 SG_ Steer : 8|16@1- (0.001,0) [-1|1] "" SIM

BO_ 768 EcuStatus: 1 ECU
 SG_ Ready : 0|1@1+ (1,0) [0|1] "" SIM
"""


class _Ecu:
    """A device under test on a CAN bus: answers Wake(On=1) with EcuStatus(Ready=1)
    and, once :attr:`control` is set, sends it as ControlCmd."""

    def __init__(self, url: str, dbc: Path, answers: bool = True) -> None:
        self.channel = hilspy.open(url, f"dbc:{dbc}")
        self.received: List[Any] = []
        self.control: Optional[dict[str, float]] = None
        self._answers = answers
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while not self._stop.is_set():
            message = self.channel.recv(0.02)
            if message is not None:
                self.received.append(message)
                if self._answers and message.name == "Wake" and message["On"] == 1:
                    self.channel.send("EcuStatus", {"Ready": 1})
            if self.control is not None:
                self.channel.send("ControlCmd", self.control)

    def names(self) -> List[Any]:
        return [(m.name, dict(m.fields)) if m.name else m.id for m in self.received]

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self.channel.close()


def _until(condition: Callable[[], bool], world: Optional[_World] = None) -> bool:
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if condition():
            return True
        if world is not None:
            world.tick()
        time.sleep(0.01)
    return condition()


def _settled(count: Callable[[], int]) -> int:
    """*count* once it stops changing for 0.1 s (bounded): what was sent before
    a cut or a stop has reached the ECU's thread, however loaded the host."""
    deadline = time.monotonic() + 5.0
    last = count()
    while time.monotonic() < deadline:
        time.sleep(0.1)
        now = count()
        if now == last:
            return now
        last = now
    return last


def _put_to_sleep(ecu: "_Ecu") -> bool:
    """Whether the ECU's last Wake is Wake(On=0), after a Wake(On=1).

    A VehicleSpeed sent just before the periodic messages stop can still land
    after it, so this does not ask for Wake(On=0) to be the very last message.
    """
    wakes = [n for n in ecu.names() if isinstance(n, tuple) and n[0] == "Wake"]
    return ("Wake", {"On": 1}) in wakes and wakes[-1] == ("Wake", {"On": 0})


def _hils_config(url: str, dbc: Path, **extra: Any) -> dict[str, Any]:
    return {
        "runner": {"listen": "127.0.0.1:0"},
        "devices": {
            "can": {
                "url": url,
                "codec": f"dbc:{dbc}",
                "messages": [
                    {
                        "name": "VehicleSpeed",
                        "period": 0.05,
                        "fields": {"Speed": "$speed_kmh"},
                    }
                ],
                "inputs": [
                    {
                        "name": "ControlCmd",
                        "command": {
                            "throttle": {"field": "Throttle", "scale": 0.01},
                            "steer": "Steer",
                        },
                    }
                ],
            }
        },
        "lifecycle": {
            "startup": [{"send": "can", "name": "Wake", "fields": {"On": 0}}],
            "scenario_start": [
                {"send": "can", "name": "Wake", "fields": {"On": 1}},
                {
                    "wait": "can",
                    "name": "EcuStatus",
                    "fields": {"Ready": 1},
                    "timeout": 2,
                },
            ],
            "scenario_end": [{"send": "can", "name": "Wake", "fields": {"On": 0}}],
            "shutdown": [{"send": "can", "id": 0x7FF, "data": "DE AD"}],
        },
        **extra,
    }


@pytest.fixture
def dbc(tmp_path: Path) -> Path:
    path = tmp_path / "vehicle.dbc"
    path.write_text(_DBC)
    return path


def test_inputs_and_lifecycle_parse(dbc: Path) -> None:
    config = parse_config(_hils_config("mem://x", dbc, commands={"mode": "control"}))
    assert config.runner_listen == "127.0.0.1:0"
    [can] = config.devices
    [control] = can.inputs
    assert control.values({"Throttle": 40.0, "Steer": -0.25}) == {
        "throttle": 0.4,
        "steer": -0.25,
    }
    assert control.values({"Throttle": "high"}) == {}
    kinds = [s.kind for s in config.steps("scenario_start")]
    assert kinds == ["send", "wait"]
    [raw] = config.steps("shutdown")
    assert (raw.data, raw.frame_id) == (b"\xde\xad", 0x7FF)
    assert (
        parse_config({**_messages({"a": 1}), "runner": {"listen": None}}).runner_listen
        == ""
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"commands": {"mode": "fly"}}, "'fly'"),
        (
            {
                "lifecycle": {
                    "startup": [
                        {"send": "can", "name": "Wake", "fields": {"On": "$speed"}}
                    ]
                }
            },
            "no ego at startup",
        ),
        (
            {"lifecycle": {"startup": [{"send": "nope", "name": "Wake"}]}},
            "no device 'nope'",
        ),
        ({"lifecycle": {"startup": [{"sleep": 1, "send": "can"}]}}, "a step is one of"),
        ({"lifecycle": {"startup": [{"send": "can"}]}}, "either a name"),
        ({"lifecycle": {"startup": [{"send": "can", "data": "zz"}]}}, "hexadecimal"),
        ({"lifecycle": {"startup": [{"wait": "can"}]}}, "names the message"),
        ({"lifecycle": {"boot": []}}, "unknown key"),
    ],
)
def test_bad_inputs_and_steps_are_refused(
    dbc: Path, change: dict[str, Any], message: str
) -> None:
    with pytest.raises(HilsConfigError, match=message):
        parse_config({**_hils_config("mem://x", dbc), **change})


def test_an_input_names_fields_of_the_command_mode(dbc: Path) -> None:
    document = _hils_config("mem://x", dbc, commands={"mode": "ackermann"})
    with pytest.raises(
        HilsConfigError, match="'throttle' is not a field of a ackermann"
    ):
        parse_config(document)


def test_the_resident_bridge_serves_a_run_end_to_end(world: _World, dbc: Path) -> None:
    """A run's HilsEgoEntity and the bridge, over TCP: the ECU is woken and
    waited for, drives the ego, and is put to sleep; startup and shutdown run
    once around it all."""
    from autoware_carla_scenario.entity.hils_entity import HilsEgoConfig, HilsEgoEntity
    from autoware_carla_scenario.hils.bridge import HilsBridge

    url = _mem()
    ecu = _Ecu(url, dbc)
    bridge = HilsBridge(
        parse_config(_hils_config(url, dbc)), carla_module=_carla(world)
    )
    thread = threading.Thread(target=bridge.run, daemon=True)
    thread.start()
    try:
        assert _until(lambda: bridge.link_address is not None)
        assert _until(lambda: ("Wake", {"On": 0}) in ecu.names())  # startup
        host, port = bridge.link_address  # type: ignore[misc]
        entity = HilsEgoEntity(
            HilsEgoConfig(host=host, port=port, ready_timeout_s=5.0), episode_name="t"
        )
        ego: _Ego = world.actors[0]
        entity._vehicle = ego  # type: ignore[assignment]

        entity.on_scenario_start(world)  # type: ignore[arg-type]
        assert not entity.termination_requested
        tick = lambda: entity.on_tick(world, 0.0)  # type: ignore[arg-type]  # noqa: E731
        assert _until(lambda: (tick(), entity.is_initialized)[1], world)
        assert bridge.active and bridge.ego is ego
        assert ("Wake", {"On": 1}) in ecu.names()
        # The ECU's inputs flow while the scenario runs.
        assert _until(
            lambda: any(m.name == "VehicleSpeed" for m in ecu.received), world
        )
        # No command yet: the ego is held on its brakes.
        tick()
        assert ego.controls[-1].brake == 1.0

        ecu.control = {"Throttle": 40.0, "Steer": -0.25}
        assert _until(lambda: (tick(), entity.command is not None)[1], world)
        tick()
        control = ego.controls[-1]
        assert control.throttle == pytest.approx(0.4)
        assert control.steer == pytest.approx(-0.25)
        assert control.brake == 0.0

        ecu.control = None
        entity.on_scenario_end(world)  # type: ignore[arg-type]
        assert not bridge.active
        assert _until(lambda: _put_to_sleep(ecu))
        speeds = _settled(
            lambda: sum(1 for m in ecu.received if m.name == "VehicleSpeed")
        )
        time.sleep(0.2)
        assert sum(1 for m in ecu.received if m.name == "VehicleSpeed") == speeds
    finally:
        bridge.stop()
        thread.join(timeout=5.0)
        bridge.close()
        assert _until(lambda: 0x7FF in ecu.names())  # shutdown
        ecu.close()


def test_an_ecu_that_does_not_answer_fails_the_run(world: _World, dbc: Path) -> None:
    from autoware_carla_scenario.entity.hils_entity import HilsEgoConfig, HilsEgoEntity
    from autoware_carla_scenario.hils.bridge import HilsBridge

    url = _mem()
    ecu = _Ecu(url, dbc, answers=False)
    document = _hils_config(url, dbc)
    document["lifecycle"]["scenario_start"][1]["timeout"] = 0.2
    bridge = HilsBridge(parse_config(document), carla_module=_carla(world))
    thread = threading.Thread(target=bridge.run, daemon=True)
    thread.start()
    try:
        assert _until(lambda: bridge.link_address is not None)
        host, port = bridge.link_address  # type: ignore[misc]
        entity = HilsEgoEntity(HilsEgoConfig(host=host, port=port))
        entity._vehicle = world.actors[0]  # type: ignore[assignment]
        entity.on_scenario_start(world)  # type: ignore[arg-type]
        # The world ticks meanwhile, as the runner's does: the entity closes
        # the link as the error comes, and a tick the bridge serves once the
        # run is gone finds the ego still in the world (issue #644).
        assert _until(
            lambda: (
                entity.on_tick(world, 0.0),  # type: ignore[func-returns-value, arg-type]
                entity.termination_requested,
            )[1],
            world,
        )
        assert "sent no EcuStatus with {'Ready': 1}" in str(entity.termination_reason)
        assert not entity.is_initialized
        # The run going away ends the scenario on the bridge too.
        entity.on_scenario_end(world)  # type: ignore[arg-type]
        assert _until(lambda: not bridge.active)
        assert _until(lambda: _put_to_sleep(ecu))
    finally:
        bridge.stop()
        thread.join(timeout=5.0)
        bridge.close()
        ecu.close()


def test_a_failed_runs_ego_does_not_start_a_scenario_of_the_bridges(
    world: _World, dbc: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #644: ticks queued when a failed run's link goes away find its ego
    still in the world; the bridge must not start a scenario of its own for
    it.  An ego that appears afterwards still does, as without a run."""
    from autoware_carla_scenario.entity.hils_entity import HilsEgoConfig, HilsEgoEntity
    from autoware_carla_scenario.hils import bridge as bridge_module

    # Every tick scans for the ego: the queued ticks below all would.
    monkeypatch.setattr(bridge_module, "_EGO_SCAN_INTERVAL_S", 0.0)
    url = _mem()
    ecu = _Ecu(url, dbc, answers=False)
    document = _hils_config(url, dbc)
    document["lifecycle"]["scenario_start"][1]["timeout"] = 0.2
    bridge = bridge_module.HilsBridge(
        parse_config(document), carla_module=_carla(world)
    )
    # The bridge's loop is driven here, one call at a time.
    bridge.startup()
    bridge.connect()
    try:
        host, port = bridge.link_address  # type: ignore[misc]
        entity = HilsEgoEntity(HilsEgoConfig(host=host, port=port))
        entity._vehicle = world.actors[0]  # type: ignore[assignment]
        entity.on_scenario_start(world)  # type: ignore[arg-type]

        # The bridge serves the run's start, fails it and reports the error;
        # the entity closes the link as the error comes.
        def served(condition: Callable[[], bool]) -> Callable[[], bool]:
            def check() -> bool:
                bridge.handle_events()
                entity.on_tick(world, 0.0)  # type: ignore[arg-type]
                return condition()

            return check

        assert _until(served(lambda: entity.termination_requested))
        assert "sent no EcuStatus" in str(entity.termination_reason)
        # The run going away ends the scenario the failed start left running.
        assert _until(served(lambda: not bridge.active))

        # Ticks the runner queued meanwhile: the ego is still in the world.
        for _ in range(3):
            world.tick()
        for _ in range(3):
            assert bridge.step(0.5)
            assert not bridge.active
        assert bridge.ego is None
        wakes = _settled(lambda: ecu.names().count(("Wake", {"On": 1})))
        assert wakes == 1

        # Without a run, an ego that appears later starts a scenario.
        world.actors[0].is_alive = False
        world.actors.append(_Ego(2))
        world.tick()
        assert bridge.step(0.5)
        assert bridge.active and bridge.ego is world.actors[1]
    finally:
        bridge.close()
        ecu.close()


def test_an_ended_runs_ego_is_ignored_while_it_lives(
    world: _World, dbc: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #644, for a run that ends normally: its ego, still in the world
    once the link closes, does not start a scenario of the bridge's own.
    Once that actor is gone the bridge forgets it, so an actor CARLA gives
    the same id afterwards does."""
    from autoware_carla_scenario.entity.hils_entity import HilsEgoConfig, HilsEgoEntity
    from autoware_carla_scenario.hils import bridge as bridge_module

    monkeypatch.setattr(bridge_module, "_EGO_SCAN_INTERVAL_S", 0.0)
    url = _mem()
    ecu = _Ecu(url, dbc)
    bridge = bridge_module.HilsBridge(
        parse_config(_hils_config(url, dbc)), carla_module=_carla(world)
    )
    bridge.startup()
    bridge.connect()
    try:
        host, port = bridge.link_address  # type: ignore[misc]
        entity = HilsEgoEntity(HilsEgoConfig(host=host, port=port))
        entity._vehicle = world.actors[0]  # type: ignore[assignment]
        entity.on_scenario_start(world)  # type: ignore[arg-type]

        def ready() -> bool:
            bridge.handle_events()
            entity.on_tick(world, 0.0)  # type: ignore[arg-type]
            return entity.is_initialized

        assert _until(ready)
        # The run ends: it waits for the bridge's ``ended``, then closes.
        ending = threading.Thread(target=entity.on_scenario_end, args=(world,))
        ending.start()

        def served(condition: Callable[[], bool]) -> Callable[[], bool]:
            def check() -> bool:
                bridge.handle_events()
                return condition()

            return check

        assert _until(served(lambda: not ending.is_alive()))
        ending.join()
        assert _until(served(lambda: not bridge._session))
        assert not bridge.active

        # The ego stays in the world: ticks do not start a scenario for it.
        for _ in range(3):
            world.tick()
        for _ in range(3):
            assert bridge.step(0.5)
            assert not bridge.active
        assert _settled(lambda: ecu.names().count(("Wake", {"On": 1}))) == 1

        # The ego is destroyed; an actor with its id appears later.
        world.actors[0].is_alive = False
        world.tick()
        assert bridge.step(0.5)
        assert not bridge.active
        world.actors[0].is_alive = True
        world.tick()
        assert bridge.step(0.5)
        assert bridge.active and bridge.ego is world.actors[0]
    finally:
        bridge.close()
        ecu.close()


def test_without_a_run_the_ego_appearing_starts_the_scenario(
    world: _World, dbc: Path
) -> None:
    from autoware_carla_scenario.hils.bridge import HilsBridge

    url = _mem()
    ecu = _Ecu(url, dbc)
    document = _hils_config(url, dbc, runner={"listen": None})
    bridge = HilsBridge(parse_config(document), carla_module=_carla(world))
    bridge.startup()
    bridge.connect()
    try:
        assert bridge.link_address is None
        world.tick()
        assert bridge.step(0.5)
        assert bridge.active
        assert ("Wake", {"On": 1}) in ecu.names()
        world.actors[0].is_alive = False
        world.tick()
        bridge.step(0.5)
        assert not bridge.active
        assert _until(lambda: ecu.names()[-1] == ("Wake", {"On": 0}))
    finally:
        bridge.close()
        ecu.close()


def test_a_failing_startup_step_stops_the_bridge(world: _World, dbc: Path) -> None:
    from autoware_carla_scenario.hils.bridge import HilsBridge, HilsStepError

    document = _hils_config(_mem(), dbc)
    document["lifecycle"]["startup"] = [
        {"wait": "can", "name": "EcuStatus", "timeout": 0.05}
    ]
    bridge = HilsBridge(parse_config(document), carla_module=_carla(world))
    try:
        with pytest.raises(HilsStepError, match="startup: can sent no EcuStatus"):
            bridge.run()
    finally:
        bridge.close()


def test_a_communication_loss_stops_its_data_while_it_runs(
    world: _World, dbc: Path
) -> None:
    """CommunicationLossAction through the run's entity: the bridge stops the
    data named, each way, for as long as the action runs."""
    from autoware_carla_scenario.actions import CommunicationLossAction
    from autoware_carla_scenario.entity.hils_entity import HilsEgoConfig, HilsEgoEntity
    from autoware_carla_scenario.entity.registry import (
        register_entity,
        unregister_entity,
    )
    from autoware_carla_scenario.hils.bridge import HilsBridge

    url = _mem()
    ecu = _Ecu(url, dbc)
    bridge = HilsBridge(
        parse_config(_hils_config(url, dbc)), carla_module=_carla(world)
    )
    thread = threading.Thread(target=bridge.run, daemon=True)
    thread.start()
    try:
        assert _until(lambda: bridge.link_address is not None)
        host, port = bridge.link_address  # type: ignore[misc]
        entity = HilsEgoEntity(
            HilsEgoConfig(host=host, port=port, ready_timeout_s=5.0, stale_s=0.2)
        )
        ego: _Ego = world.actors[0]
        entity._vehicle = ego  # type: ignore[assignment]
        register_entity("Ego", entity)
        entity.on_scenario_start(world)  # type: ignore[arg-type]
        tick = lambda: entity.on_tick(world, 0.0)  # type: ignore[arg-type]  # noqa: E731
        assert _until(lambda: (tick(), entity.is_initialized)[1], world)
        channel = bridge._channels["can"]

        def speeds() -> int:
            return sum(1 for m in ecu.received if m.name == "VehicleSpeed")

        # VehicleSpeed to the ECU stops; the ECU's commands still arrive.
        ecu.control = {"Throttle": 40.0, "Steer": 0.0}
        assert _until(lambda: (tick(), entity.command is not None)[1], world)
        speed_loss = CommunicationLossAction(
            "can", messages=["VehicleSpeed"], direction="to_ecu", label="speed_loss"
        )
        speed_loss.tick(world, 1.0)  # type: ignore[arg-type]
        assert speed_loss.cutting
        assert _until(lambda: channel.disconnected("send"), world)
        assert not channel.disconnected("recv")
        before = _settled(speeds)
        for _ in range(5):
            world.tick()
            time.sleep(0.02)
            tick()
        assert speeds() == before
        # Polled: stale after 0.2 s, the ego needs a command read that recently.
        assert _until(
            lambda: (tick(), ego.controls[-1].throttle == pytest.approx(0.4))[1],
            world,
        )

        # Everything from the ECU, for 0.5 s: its command goes stale and the
        # ego brakes, then the commands come back.
        command_loss = CommunicationLossAction(direction="from_ecu", duration=0.5)
        command_loss.tick(world, 2.0)  # type: ignore[arg-type]
        assert _until(lambda: channel.disconnected("recv"), world)
        # Polled, not slept: the entity times a command from the tick that
        # reads it off the link, so one the bridge forwarded before the cut
        # took effect is fresh on the first tick after it, however long ago
        # the cut started.
        assert _until(lambda: (tick(), ego.controls[-1].brake == 1.0)[1], world)
        command_loss.tick(world, 2.3)  # type: ignore[arg-type]
        assert command_loss.cutting
        command_loss.tick(world, 2.5)  # type: ignore[arg-type]
        assert not command_loss.cutting
        assert _until(lambda: not channel.disconnected("recv"), world)
        assert _until(
            lambda: (tick(), ego.controls[-1].throttle == pytest.approx(0.4))[1],
            world,
        )
        # The first one still runs, with no end: VehicleSpeed is still stopped.
        assert channel.disconnected("send") and speeds() == before

        # The scenario's end carries everything again.
        entity.on_scenario_end(world)  # type: ignore[arg-type]
        assert not bridge.active and not channel.disconnected()
        speed_loss.close()
        ecu.control = None
    finally:
        unregister_entity("Ego")
        bridge.stop()
        thread.join(timeout=5.0)
        bridge.close()
        ecu.close()


def test_the_bridge_stops_what_its_cuts_name_together(world: _World, dbc: Path) -> None:
    from autoware_carla_scenario.hils.bridge import HilsBridge

    url = _mem()
    ecu = _Ecu(url, dbc, answers=False)
    document = _hils_config(url, dbc, runner={"listen": None})
    document["lifecycle"] = {
        "scenario_end": [{"send": "can", "name": "Wake", "fields": {"On": 0}}]
    }
    bridge = HilsBridge(parse_config(document), carla_module=_carla(world))
    bridge.startup()
    bridge.connect()
    try:
        with pytest.raises(ValueError, match="no scenario runs"):
            bridge.disconnect("x")
        world.tick()
        assert bridge.step(0.5) and bridge.active
        with pytest.raises(ValueError, match="no device 'nope'; the devices are can"):
            bridge.disconnect("x", "nope")
        with pytest.raises(ValueError, match="direction must be one of"):
            bridge.disconnect("x", direction="send")
        channel = bridge._channels["can"]

        def arrives(expected: set[str]) -> bool:
            """Whether what of VehicleSpeed (periodic) and Wake (sent now)
            reaches the ECU is *expected*: polled, as on a loaded host the
            ECU's thread can take longer than the 0.1 s to pick it up."""
            ecu.received.clear()
            channel.send("Wake", {"On": 1})
            for _ in range(3):
                world.tick()
                bridge.step(0.5)
            time.sleep(0.1)
            return _until(lambda: {m.name for m in ecu.received} == expected)

        assert arrives({"VehicleSpeed", "Wake"})
        bridge.disconnect("a", "can", ["VehicleSpeed"], "to_ecu")
        bridge.disconnect("b", None, ["Wake"], "to_ecu")
        assert arrives(set())
        bridge.reconnect("a")
        assert arrives({"VehicleSpeed"})
        bridge.disconnect("c", "can")  # everything, both ways
        assert arrives(set()) and channel.disconnected("recv")
        bridge.reconnect("c")
        bridge.reconnect("unknown")
        assert arrives({"VehicleSpeed"}) and not channel.disconnected("recv")
        # A cut naming no message is refused, and leaves the others as they are.
        with pytest.raises(ValueError, match="messages names none"):
            bridge.disconnect("e", "can", [])
        assert set(bridge._cuts) == {"b"}
        # The scenario's end carries everything again, before its steps: the
        # ECU hears Wake(On=0) although everything was stopped.
        bridge.disconnect("d")
        ecu.received.clear()
        world.actors[0].is_alive = False
        world.tick()
        bridge.step(0.5)
        assert not bridge.active and not channel.disconnected()
        assert _until(lambda: ("Wake", {"On": 0}) in ecu.names())
    finally:
        bridge.close()
        ecu.close()


# -- the run's entity, against a scripted bridge -----------------------------------


class _ScriptedBridge:
    """Accepts *runs* runs in turn, greets each, and answers its ``start`` with
    *answers*."""

    def __init__(
        self, answers: List[dict[str, Any]], protocol: int = 1, runs: int = 1
    ) -> None:
        import socket

        from autoware_carla_scenario.hils.protocol import LineReader, encode

        self.server = socket.create_server(("127.0.0.1", 0))
        self.port = self.server.getsockname()[1]
        self.received: List[dict[str, Any]] = []
        self._answers = answers
        self._protocol = protocol
        self._encode = encode
        self._reader_type = LineReader
        self.conn: Optional[Any] = None
        self._runs = runs
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self) -> None:
        for _ in range(self._runs):
            conn, _ = self.server.accept()
            self.conn = conn
            conn.sendall(self._encode({"type": "hello", "protocol": self._protocol}))
            reader = self._reader_type(conn)
            while not reader.closed:
                for message in reader.read(timeout=0.05):
                    self.received.append(message)
                    if message["type"] == "start":
                        for answer in self._answers:
                            conn.sendall(self._encode(answer))
                    elif message["type"] == "end":
                        conn.sendall(self._encode({"type": "ended"}))

    def send(self, message: dict[str, Any]) -> None:
        assert self.conn is not None
        self.conn.sendall(self._encode(message))


def _entity(port: int, **config: Any) -> Any:
    from autoware_carla_scenario.entity.hils_entity import HilsEgoConfig, HilsEgoEntity

    entity = HilsEgoEntity(HilsEgoConfig(host="127.0.0.1", port=port, **config))
    entity._vehicle = _Ego(1)  # type: ignore[assignment]
    entity.on_scenario_start(None)  # type: ignore[arg-type]
    return entity


def _ticked(entity: Any, condition: Callable[[], bool]) -> bool:
    return _until(lambda: (entity.on_tick(None, 0.0), condition())[1])


def test_an_unreachable_bridge_ends_the_run() -> None:
    import socket

    with socket.create_server(("127.0.0.1", 0)) as probe:
        port = probe.getsockname()[1]
    entity = _entity(port, connect_timeout_s=0.5)
    assert entity.termination_requested
    assert "cannot be reached" in entity.termination_reason


def test_another_protocol_ends_the_run() -> None:
    entity = _entity(_ScriptedBridge([], protocol=99).port)
    assert "speaks protocol 99" in entity.termination_reason


def test_a_bridge_error_or_silence_ends_the_run() -> None:
    entity = _entity(_ScriptedBridge([{"type": "error", "message": "boom"}]).port)
    assert _ticked(entity, lambda: entity.termination_requested)
    assert entity.termination_reason == "The HILS bridge: boom"
    silent = _entity(_ScriptedBridge([]).port, ready_timeout_s=0.05)
    assert _ticked(silent, lambda: silent.termination_requested)
    assert "did not initialize the ECUs within" in silent.termination_reason


@pytest.mark.parametrize("on_stale", ["brake", "hold"])
def test_a_stale_command_brakes_or_holds(on_stale: str) -> None:
    command = {"type": "command", "mode": "control", "throttle": 0.6, "gear": 3}
    bridge = _ScriptedBridge([{"type": "ready"}, command])
    entity = _entity(bridge.port, stale_s=0.1, on_stale=on_stale)
    ego = entity.actor
    assert _ticked(entity, lambda: entity.command is not None)
    assert entity.is_initialized
    entity.on_tick(None, 0.0)
    assert ego.controls[-1].throttle == pytest.approx(0.6)
    assert ego.controls[-1].gear == 3
    time.sleep(0.15)
    entity.on_tick(None, 0.0)
    if on_stale == "brake":
        assert ego.controls[-1].brake == 1.0
    else:
        assert ego.controls[-1].throttle == pytest.approx(0.6)
    bridge.send(command)
    assert _ticked(entity, lambda: ego.controls[-1].throttle == pytest.approx(0.6))
    entity.on_scenario_end(None)
    assert [m["type"] for m in bridge.received] == ["start", "end"]
    assert bridge.received[0]["ego_id"] == 1


def test_a_retry_numbers_its_cuts_afresh_and_waits_for_a_command() -> None:
    # A retried scenario reuses the entity (#672): the bridge restored the
    # previous attempt's cuts, and its command does not drive this one.
    command = {"type": "command", "mode": "control", "throttle": 0.6, "gear": 3}
    bridge = _ScriptedBridge([{"type": "ready"}, command], runs=2)
    entity = _entity(bridge.port)
    assert _ticked(entity, lambda: entity.command is not None)
    assert entity.disconnect("can") == "1"
    entity.on_scenario_end(None)

    entity.on_scenario_start(None)  # type: ignore[arg-type]

    assert entity.command is None
    assert not entity.is_initialized
    assert entity.disconnect("can") == "1"
    entity.on_scenario_end(None)


def test_an_ackermann_command_drives_carlas_controller() -> None:
    command = {"type": "command", "mode": "ackermann", "steer": 0.1, "speed": 5.0}
    entity = _entity(_ScriptedBridge([{"type": "ready"}, command]).port)
    assert _ticked(entity, lambda: entity.command is not None)
    entity.on_tick(None, 0.0)
    control = entity.actor.controls[-1]
    assert (control.steer, control.speed) == (pytest.approx(0.1), pytest.approx(5.0))


def test_the_runner_builds_the_hils_entity() -> None:
    from omegaconf import OmegaConf

    from autoware_carla_scenario.entity.hils_entity import HilsEgoEntity
    from autoware_carla_scenario.examples.run import build_ego_entity

    cfg = OmegaConf.create(
        {"ego": {"entity": "hils"}, "hils": {"port": 7800}, "scenario": {"name": "s"}}
    )
    entity = build_ego_entity(cfg)
    assert isinstance(entity, HilsEgoEntity)
    assert entity.config.port == 7800
    with pytest.raises(ValueError, match="unknown hils setting"):
        build_ego_entity(
            OmegaConf.create({"ego": {"entity": "hils"}, "hils": {"prot": 1}})
        )


def test_the_communication_loss_action_asks_the_bridge() -> None:
    from autoware_carla_scenario.actions import CommunicationLossAction
    from autoware_carla_scenario.conditions import ElapsedTimeCondition
    from autoware_carla_scenario.entity.registry import (
        register_entity,
        unregister_entity,
    )

    with pytest.raises(ValueError, match="direction must be one of"):
        CommunicationLossAction(direction="send")
    with pytest.raises(ValueError, match="duration must be positive"):
        CommunicationLossAction(duration=0.0)
    action = CommunicationLossAction(
        "can",
        messages="VehicleSpeed, scan,VehicleSpeed",
        direction="from_ecu",
        until=ElapsedTimeCondition(5.0, label="later"),
    )
    assert (action.device, action.messages, action.direction, action.duration) == (
        "can",
        ("VehicleSpeed", "scan"),
        "from_ecu",
        None,
    )
    assert CommunicationLossAction("", messages=" ").messages is None
    assert CommunicationLossAction(messages=["scan"]).messages == ("scan",)
    # Registered for initialization, it runs on the tick loop: the bridge is
    # reached only once the ego has started.
    assert action.needs_tick_loop

    # Another ego: nothing to stop, and nothing breaks.
    CommunicationLossAction().tick(None, 0.0)  # type: ignore[arg-type]

    bridge = _ScriptedBridge([{"type": "ready"}])
    entity = _entity(bridge.port)
    register_entity("Ego", entity)
    try:
        action.tick(None, 1.0)  # type: ignore[arg-type]
        assert action.cutting
        everything = CommunicationLossAction()
        everything.tick(None, 1.0)  # type: ignore[arg-type]
        assert _until(lambda: len(bridge.received) == 3)
        cut, every = bridge.received[1:]
        assert {k: v for k, v in cut.items() if k != "id"} == {
            "type": "disconnect",
            "device": "can",
            "messages": ["VehicleSpeed", "scan"],
            "direction": "from_ecu",
        }
        assert every["device"] is None and every["messages"] is None
        assert cut["id"] != every["id"]
        # Running until its end condition fires: then the data comes back.
        action.tick(None, 3.0)  # type: ignore[arg-type]
        assert action.cutting
        action.tick(None, 5.0)  # type: ignore[arg-type]
        assert not action.cutting
        assert _until(lambda: len(bridge.received) == 4)
        assert bridge.received[3] == {"type": "reconnect", "id": cut["id"]}
        # The bridge refusing a cut ends the run.
        bridge.send({"type": "error", "message": "disconnect: no device 'can'"})
        assert _ticked(entity, lambda: entity.termination_requested)
        assert "no device 'can'" in entity.termination_reason
        assert entity.disconnect("can") is None
        everything.close()
        assert not everything.cutting
    finally:
        unregister_entity("Ego")


# -- non-finite command fields (#730) ----------------------------------------------

_NON_FINITE = [math.nan, math.inf, -math.inf]


def _command_fields() -> List[tuple[str, str]]:
    from autoware_carla_scenario.hils.protocol import COMMAND_FIELDS

    return [(mode, key) for mode, fields in COMMAND_FIELDS.items() for key in fields]


def _finite_control(control: Any) -> None:
    for name in (
        "throttle",
        "brake",
        "steer",
        "steer_speed",
        "speed",
        "acceleration",
        "jerk",
        "gear",
    ):
        value = getattr(control, name, None)
        if value is not None:
            assert math.isfinite(value), (name, value)


@pytest.mark.parametrize("value", _NON_FINITE, ids=["nan", "inf", "-inf"])
@pytest.mark.parametrize("mode,key", _command_fields())
def test_a_non_finite_command_field_is_dropped_by_the_run(
    mode: str, key: str, value: float, caplog: pytest.LogCaptureFixture
) -> None:
    # A bridge (or any sender) writing NaN/Infinity must not reach CARLA as a
    # NaN control nor crash the tick (#730): the field counts as absent.
    other = (
        "steer" if key != "steer" else ("throttle" if mode == "control" else "speed")
    )
    command = {"type": "command", "mode": mode, key: value, other: 0.25}
    bridge = _ScriptedBridge([{"type": "ready"}, command, command])
    entity = _entity(bridge.port)
    with caplog.at_level("WARNING"):
        assert _ticked(entity, lambda: entity.command is not None)
        entity.on_tick(None, 0.0)
        bridge.send(command)
        time.sleep(0.05)
        entity.on_tick(None, 0.0)
    assert not entity.termination_requested, entity.termination_reason
    assert entity.command == (mode, {other: 0.25})
    control = entity.actor.controls[-1]
    _finite_control(control)
    assert getattr(control, other) == pytest.approx(0.25)
    dropped = [r for r in caplog.records if "non-finite" in r.getMessage()]
    assert len(dropped) == 1
    entity.on_scenario_end(None)


def test_a_huge_integer_command_field_is_dropped_by_the_run() -> None:
    # JSON integers are unbounded: float() of one overflows.
    command = {"type": "command", "mode": "control", "gear": 10**400, "throttle": 0.5}
    entity = _entity(_ScriptedBridge([{"type": "ready"}, command]).port)
    assert _ticked(entity, lambda: entity.command is not None)
    entity.on_tick(None, 0.0)
    assert not entity.termination_requested
    assert entity.command == ("control", {"throttle": 0.5})


@pytest.mark.parametrize("value", _NON_FINITE, ids=["nan", "inf", "-inf"])
@pytest.mark.parametrize("mode,key", _command_fields())
def test_a_non_finite_ecu_field_is_dropped_by_the_bridge(
    mode: str, key: str, value: float, caplog: pytest.LogCaptureFixture
) -> None:
    from autoware_carla_scenario.hils.config import InputSource, InputSpec

    spec = InputSpec(
        name="Command",
        command={key: InputSource(field="Raw"), "_ok": InputSource(field="Ok")},
    )
    with caplog.at_level("WARNING"):
        for _ in range(3):
            assert spec.values({"Raw": value, "Ok": 1.0}) == {"_ok": 1.0}
    assert InputSource(field="Raw").resolve({"Raw": value}) is None
    dropped = [r for r in caplog.records if "non-finite" in r.getMessage()]
    assert len(dropped) == 1
    assert key in dropped[0].getMessage()


def test_a_scale_that_overflows_is_dropped_by_the_bridge() -> None:
    from autoware_carla_scenario.hils.config import InputSource

    assert InputSource(field="Raw", scale=1e308).resolve({"Raw": 10.0}) is None
    assert InputSource(field="Raw", scale=2.0).resolve({"Raw": 10**400}) is None
    assert InputSource(field="Raw", scale=2.0).resolve({"Raw": 1.5}) == 3.0


def test_the_bridge_forwards_no_non_finite_command(world: _World, dbc: Path) -> None:
    from autoware_carla_scenario.hils.bridge import HilsBridge
    from autoware_carla_scenario.hils.protocol import COMMAND_FIELDS

    document = _hils_config(_mem(), dbc, commands={"mode": "control"})
    bridge = HilsBridge(parse_config(document), carla_module=_carla(world))
    sent: List[dict[str, Any]] = []
    bridge._forwarding = True
    bridge._link = types.SimpleNamespace(send=sent.append)  # type: ignore[assignment]
    [spec] = bridge.config.devices
    [input_spec] = spec.inputs
    message = types.SimpleNamespace(
        name=input_spec.name, fields={"Throttle": 40.0, "Steer": math.nan}
    )
    bridge._on_input(spec, message)
    message.fields = {"Throttle": math.inf, "Steer": -0.25}
    bridge._on_input(spec, message)
    assert bridge.command == {"throttle": 0.4, "steer": -0.25}
    assert [set(m) - {"type", "mode"} for m in sent] == [
        {"throttle"},
        {"throttle", "steer"},
    ]
    assert all(
        set(m) - {"type", "mode"} <= set(COMMAND_FIELDS["control"]) for m in sent
    )
    assert all(
        math.isfinite(v)
        for m in sent
        for k, v in m.items()
        if k not in ("type", "mode")
    )


@pytest.mark.parametrize("value", [".nan", ".inf", "-.inf"])
def test_a_non_finite_input_constant_or_scale_is_refused(dbc: Path, value: str) -> None:
    import yaml

    number = yaml.safe_load(value)
    for command in (
        {"throttle": number},
        {"throttle": {"field": "Throttle", "scale": number}},
        {"throttle": {"field": "Throttle", "offset": number}},
    ):
        document = _hils_config(_mem(), dbc, commands={"mode": "control"})
        document["devices"]["can"]["inputs"][0]["command"] = command
        with pytest.raises(HilsConfigError, match="finite"):
            parse_config(document)


@pytest.mark.parametrize("value", [1e39, -1e39, 1e300])
def test_a_command_field_beyond_a_32_bit_float_is_dropped(value: float) -> None:
    # Finite, but CARLA's 32-bit float controls would make it an infinity.
    from autoware_carla_scenario.hils.config import InputSource, InputSpec

    command = {"type": "command", "mode": "ackermann", "speed": value, "steer": 0.1}
    bridge = _ScriptedBridge([{"type": "ready"}, command])
    entity = _entity(bridge.port)
    assert _ticked(entity, lambda: entity.command is not None)
    entity.on_tick(None, 0.0)
    assert entity.command == ("ackermann", {"steer": 0.1})
    _finite_control(entity.actor.controls[-1])
    entity.on_scenario_end(None)

    spec = InputSpec(name="Command", command={"speed": InputSource(field="Raw")})
    assert spec.values({"Raw": value}) == {}
    assert InputSource(field="Raw").resolve({"Raw": value}) is None
    assert InputSource(field="Raw").resolve({"Raw": 3.0e38}) == 3.0e38


@pytest.mark.parametrize("gear", [1e30, -1e30, 2.0**63, 2**40])
def test_a_huge_finite_gear_does_not_raise_in_on_tick(gear: float) -> None:
    # int() of a huge float is a huge int, which CARLA's VehicleControl
    # refuses (SystemError/OverflowError) in on_tick: it is clamped instead.
    command = {"type": "command", "mode": "control", "gear": gear, "throttle": 0.5}
    entity = _entity(_ScriptedBridge([{"type": "ready"}, command]).port)
    assert _ticked(entity, lambda: entity.command is not None)
    entity.on_tick(None, 0.0)
    assert not entity.termination_requested
    control = entity.actor.controls[-1]
    assert control.gear == (2**31 - 1 if gear > 0 else -(2**31))
    assert control.throttle == pytest.approx(0.5)
    entity.on_scenario_end(None)


def test_a_scale_too_large_for_a_float_is_refused(dbc: Path) -> None:
    document = _hils_config(_mem(), dbc, commands={"mode": "control"})
    document["devices"]["can"]["inputs"][0]["command"] = {
        "throttle": {"field": "Throttle", "scale": 10**400}
    }
    with pytest.raises(HilsConfigError, match="float"):
        parse_config(document)


# -- forwarding the ECUs' commands from several receiving threads ------------------


def _two_device_bridge(world: _World, dbc: Path) -> Any:
    """A bridge forwarding commands, and two devices' inputs: the steering on
    one, the pedals on another, each received on a thread of its own."""
    from autoware_carla_scenario.hils.bridge import HilsBridge
    from autoware_carla_scenario.hils.config import InputSource, InputSpec

    document = _hils_config(_mem(), dbc, commands={"mode": "control"})
    bridge = HilsBridge(parse_config(document), carla_module=_carla(world))
    bridge._forwarding = True
    steering = types.SimpleNamespace(
        inputs=[InputSpec("Steer", {"steer": InputSource(field="S")})]
    )
    pedals = types.SimpleNamespace(
        inputs=[InputSpec("Pedal", {"throttle": InputSource(field="T")})]
    )
    return bridge, steering, pedals


def _fields(message: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in message.items() if k not in ("type", "mode")}


class _BlockingLink:
    """A link whose first send blocks until :attr:`release` is set (and then
    raises, with *fail*): that sender is preempted on a slow link."""

    def __init__(self, fail: bool = False) -> None:
        self.sent: List[dict[str, Any]] = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self._fail = fail
        self._calls = 0
        self._lock = threading.Lock()

    def send(self, message: Any) -> bool:
        with self._lock:
            self._calls += 1
            first = self._calls == 1
        if first:
            self.entered.set()
            assert self.release.wait(5.0)
            if self._fail:
                raise OSError("the link broke")
        self.sent.append(dict(message))
        return True


def _start(target: Callable[[], None]) -> threading.Thread:
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread


def test_the_last_command_forwarded_is_the_newest_merged(
    world: _World, dbc: Path
) -> None:
    bridge, steering, pedals = _two_device_bridge(world, dbc)
    link = _BlockingLink()
    bridge._link = link
    a = _start(
        lambda: bridge._on_input(
            steering, types.SimpleNamespace(name="Steer", fields={"S": 0.2})
        )
    )
    assert link.entered.wait(5.0)
    # A is sending; B merges its fields meanwhile and forwards them.
    b = _start(
        lambda: bridge._on_input(
            pedals, types.SimpleNamespace(name="Pedal", fields={"T": 0.8})
        )
    )
    assert _until(lambda: bridge.command == {"steer": 0.2, "throttle": 0.8})
    link.release.set()
    a.join(5.0)
    b.join(5.0)
    assert not a.is_alive() and not b.is_alive()
    assert link.sent, "nothing forwarded"
    assert _fields(link.sent[-1]) == {"steer": 0.2, "throttle": 0.8}
    assert link.sent[-1]["type"] == "command"
    assert link.sent[-1]["mode"] == "control"


def test_a_failed_forward_does_not_block_the_next(world: _World, dbc: Path) -> None:
    bridge, steering, pedals = _two_device_bridge(world, dbc)
    link = _BlockingLink(fail=True)
    bridge._link = link
    errors: List[BaseException] = []

    def first() -> None:
        try:
            bridge._on_input(
                steering, types.SimpleNamespace(name="Steer", fields={"S": 0.2})
            )
        except OSError as exc:
            errors.append(exc)

    a = _start(first)
    assert link.entered.wait(5.0)
    b = _start(
        lambda: bridge._on_input(
            pedals, types.SimpleNamespace(name="Pedal", fields={"T": 0.8})
        )
    )
    assert _until(lambda: bridge.command == {"steer": 0.2, "throttle": 0.8})
    link.release.set()
    a.join(5.0)
    b.join(5.0)
    assert not a.is_alive() and not b.is_alive()
    assert len(errors) == 1
    # The failed send's fields still reach the run, with the newer ones.
    assert [_fields(m) for m in link.sent] == [{"steer": 0.2, "throttle": 0.8}]
    # Nothing is left holding a lock: merging, reading and ending go on.
    bridge._on_input(steering, types.SimpleNamespace(name="Steer", fields={"S": 0.1}))
    assert _fields(link.sent[-1]) == {"steer": 0.1, "throttle": 0.8}
    assert bridge.command == {"steer": 0.1, "throttle": 0.8}


def test_forwarding_to_a_closed_link_does_not_block(world: _World, dbc: Path) -> None:
    import queue as queue_module

    from autoware_carla_scenario.hils.link import RunnerLink

    bridge, steering, pedals = _two_device_bridge(world, dbc)
    link = RunnerLink("127.0.0.1:0", queue_module.Queue())
    link.start()
    link.close()
    bridge._link = link
    done = threading.Event()

    def both() -> None:
        bridge._on_input(
            steering, types.SimpleNamespace(name="Steer", fields={"S": 0.2})
        )
        bridge._on_input(pedals, types.SimpleNamespace(name="Pedal", fields={"T": 0.8}))
        done.set()

    _start(both).join(5.0)
    assert done.is_set()
    assert bridge.command == {"steer": 0.2, "throttle": 0.8}


def test_a_single_device_forwards_every_command_in_order(
    world: _World, dbc: Path
) -> None:
    bridge, steering, _ = _two_device_bridge(world, dbc)
    sent: List[dict[str, Any]] = []
    bridge._link = types.SimpleNamespace(send=sent.append)
    for steer in (0.1, 0.2, 0.3):
        bridge._on_input(
            steering, types.SimpleNamespace(name="Steer", fields={"S": steer})
        )
    assert [_fields(m) for m in sent] == [{"steer": s} for s in (0.1, 0.2, 0.3)]
    # A message with none of the inputs forwards nothing.
    bridge._on_input(steering, types.SimpleNamespace(name="Other", fields={"S": 1}))
    assert len(sent) == 3


def test_forwarding_starts_with_the_newest_command_once(
    world: _World, dbc: Path
) -> None:
    bridge, steering, _ = _two_device_bridge(world, dbc)
    sent: List[dict[str, Any]] = []
    bridge._link = types.SimpleNamespace(send=sent.append)
    bridge._forwarding = False
    bridge._on_input(steering, types.SimpleNamespace(name="Steer", fields={"S": 0.2}))
    assert sent == []
    bridge._forward(start=True)
    assert [_fields(m) for m in sent] == [{"steer": 0.2}]
    bridge._forward()
    assert len(sent) == 1


def test_no_command_follows_ended(world: _World, dbc: Path) -> None:
    bridge, steering, _ = _two_device_bridge(world, dbc)
    link = _BlockingLink()
    bridge._link = link
    bridge._active = True
    # No device is open for the configuration's scenario_end steps.
    bridge._run_steps = lambda phase, strict: None
    a = _start(
        lambda: bridge._on_input(
            steering, types.SimpleNamespace(name="Steer", fields={"S": 0.2})
        )
    )
    assert link.entered.wait(5.0)
    # The run ends while the command is being sent: ``ended`` waits for it.
    ending = _start(lambda: bridge._handle(("end", {})))
    ending.join(0.2)
    link.release.set()
    a.join(5.0)
    ending.join(5.0)
    assert not a.is_alive() and not ending.is_alive()
    assert [m["type"] for m in link.sent] == ["command", "ended"]
    # A command merged after the end is not forwarded.
    bridge._on_input(steering, types.SimpleNamespace(name="Steer", fields={"S": 0.3}))
    assert [m["type"] for m in link.sent] == ["command", "ended"]
