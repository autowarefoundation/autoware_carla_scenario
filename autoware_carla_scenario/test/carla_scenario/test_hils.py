"""The HILS bridge (``scenario-hils``): signals, configuration, measurements, and
the bridge itself against a fake CARLA world, its devices on hilspy's
in-process ``mem://`` bus."""

from __future__ import annotations

import itertools
import math
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
        (_device(), "either messages or a sensor"),
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
    ],
)
def test_a_bad_configuration_is_refused(document: Any, message: str) -> None:
    with pytest.raises(HilsConfigError, match=message):
        parse_config(document)


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

    def get_control(self) -> _Control:
        return _Control()


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

    def tick(self) -> None:
        self.frame += 1
        elapsed = self.frame * 0.05
        actors = {
            a.id: _ActorSnapshot(self.speed * elapsed, self.speed)
            for a in self.actors
            if a.is_alive
        }
        self.pending.append(_Snapshot(self.frame, elapsed, actors))

    def wait_for_tick(self, timeout: float) -> _Snapshot:
        if not self.pending:
            raise RuntimeError("time-out of 0.0 ms while waiting for the simulator")
        return self.pending.pop(0)


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
        assert not bridge.step()
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
