"""LiDAR for a driver policy: the sensor, the rig frame, and the payload it rides in."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from carla_driver_interface.protocol import carla_driver_pb2, pack_lidar_sweep
from autoware_carla_scenario.driver.base import DriverClientConfig, DriverLidarConfig
from autoware_carla_scenario.driver.geometry import Pose
from autoware_carla_scenario.driver.observation import (
    lidar_points_to_rig,
    sensor_pose_in_rig,
)
from autoware_carla_scenario.driver.renderer import RendererDataBuilder
from autoware_carla_scenario.sensor import carla_lidar
from autoware_carla_scenario.sensor.carla_lidar import CarlaLidarSensor
from autoware_carla_scenario.sensor.lidar_weather import (
    FOG_REFERENCE_SCALE,
    LidarWeather,
    apply_weather,
)

from .test_driver_renderer import _ego, _world

# ---------------------------------------------------------------------------
# Rig frame
# ---------------------------------------------------------------------------


def test_a_level_mount_mirrors_y_and_lifts_by_the_mount_height() -> None:
    pose = sensor_pose_in_rig(0.5, 0.0, 2.0, 0.0, 0.0, 0.0, rear_axle_offset_m=-1.5)
    raw = np.array([[10.0, 3.0, -1.0, 0.25]], dtype=np.float32)  # CARLA: y right

    rig = lidar_points_to_rig(raw, pose)

    # x shifts by the mount and the actor->rig offset; y flips; z lifts.
    np.testing.assert_allclose(rig[0], [10.0 + 0.5 + 1.5, -3.0, 1.0, 0.25], atol=1e-6)
    assert rig.dtype == np.float32


def test_a_yawed_mount_rotates_the_sweep_the_way_carla_would() -> None:
    # CARLA yaw +90 degrees turns the sensor to the vehicle's right, so a point
    # straight ahead of the sensor lies on the rig's right: -y.
    pose = sensor_pose_in_rig(0.0, 0.0, 0.0, 0.0, 0.0, 90.0, rear_axle_offset_m=0.0)
    rig = lidar_points_to_rig(np.array([[5.0, 0.0, 0.0, 1.0]], dtype=np.float32), pose)
    np.testing.assert_allclose(rig[0, :3], [0.0, -5.0, 0.0], atol=1e-5)


def test_a_rolled_mount_puts_a_point_below_the_sensor_on_the_ground() -> None:
    # Rolled 180 degrees (mounted upside down), the sensor's "up" is the ground.
    pose = sensor_pose_in_rig(0.0, 0.0, 2.0, 180.0, 0.0, 0.0, rear_axle_offset_m=0.0)
    rig = lidar_points_to_rig(np.array([[0.0, 0.0, 2.0, 1.0]], dtype=np.float32), pose)
    np.testing.assert_allclose(rig[0, :3], [0.0, 0.0, 0.0], atol=1e-5)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_lidar_config_rejects_an_inverted_field_of_view() -> None:
    with pytest.raises(ValueError, match="lower_fov_deg"):
        DriverLidarConfig(upper_fov_deg=-30.0, lower_fov_deg=10.0)


def test_lidar_logical_ids_must_be_unique() -> None:
    with pytest.raises(ValueError, match="unique"):
        DriverClientConfig(lidars=(DriverLidarConfig(), DriverLidarConfig()))


def test_lidars_and_map_build_from_a_mapping() -> None:
    config = DriverClientConfig.from_mapping(
        {
            "lidars": [{"logical_id": "lidar_top", "channels": 32, "position_z": 1.9}],
            "map_dir": "/maps",
            "map_formats": ["lanelet2", "opendrive"],
        }
    )
    assert config.lidars == (
        DriverLidarConfig(logical_id="lidar_top", channels=32, position_z=1.9),
    )
    assert config.map_dir == "/maps"
    assert config.map_formats == ("lanelet2", "opendrive")
    with pytest.raises(ValueError, match="Unknown DriverLidarConfig"):
        DriverClientConfig.from_mapping({"lidars": [{"rotation_frequency": 10.0}]})


# ---------------------------------------------------------------------------
# Sensor
# ---------------------------------------------------------------------------


def _measurement(frame: int, n: int = 3) -> SimpleNamespace:
    points = np.full((n, 4), float(frame), dtype=np.float32)
    return SimpleNamespace(frame=frame, raw_data=points.tobytes())


def test_the_sweep_of_the_frame_asked_for_is_taken_and_the_backlog_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(carla_lidar, "_SWEEP_TIMEOUT", 0.05)
    lidar = CarlaLidarSensor(DriverLidarConfig())
    for frame in (10, 11):
        lidar._receive(_measurement(frame))  # noqa: SLF001

    sweep = lidar.get_sweep(11)
    assert sweep is not None and sweep.shape == (3, 4) and sweep[0, 0] == 11.0
    # Nothing for frame 12 arrives: told nothing rather than the past.
    assert lidar.get_sweep(12) is None
    # A newer sweep than asked for is still this tick's best answer.
    lidar._receive(_measurement(14))  # noqa: SLF001
    sweep = lidar.get_sweep(13)
    assert sweep is not None and sweep[0, 0] == 14.0


def test_between_sweeps_the_newest_is_taken_with_its_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(carla_lidar, "_SWEEP_TIMEOUT", 0.05)
    lidar = CarlaLidarSensor(DriverLidarConfig(), sweep_interval_s=0.1)
    assert lidar.sweep_interval_ticks == 2
    lidar._receive(_measurement(10))  # noqa: SLF001
    sweep = lidar.latest_sweep(10)
    assert sweep is not None and sweep[0] == 10
    # The tick after: no sweep is due, the one of frame 10 is the newest.
    sweep = lidar.latest_sweep(11)
    assert sweep is not None and sweep[0] == 10 and sweep[1][0, 0] == 10.0
    # Two ticks on, one is due and waited for.
    lidar._receive(_measurement(12))  # noqa: SLF001
    sweep = lidar.latest_sweep(12)
    assert sweep is not None and sweep[0] == 12


def test_a_sweep_a_tick_off_the_policy_is_still_this_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(carla_lidar, "_SWEEP_TIMEOUT", 0.05)
    lidar = CarlaLidarSensor(DriverLidarConfig(), sweep_interval_s=0.1)
    for frame in (10, 12):
        lidar._receive(_measurement(frame))  # noqa: SLF001
    sweep = lidar.latest_sweep(13)
    assert sweep is not None and sweep[0] == 12
    # Nothing comes for frame 14: by 15 the sweep of 12 is an interval old.
    assert lidar.latest_sweep(15) is None


def test_the_sweep_interval_is_whole_ticks() -> None:
    with pytest.raises(ValueError, match="sweep_interval_s"):
        CarlaLidarSensor(DriverLidarConfig(), sweep_interval_s=0.07)
    with pytest.raises(ValueError, match="sweep_interval_s"):
        CarlaLidarSensor(DriverLidarConfig(), sweep_interval_s=0.0)


def test_the_listener_hands_the_measurement_to_receive() -> None:
    lidar = CarlaLidarSensor(DriverLidarConfig())
    lidar._on_sensor_data(_measurement(7))  # type: ignore[arg-type]  # noqa: SLF001
    # typesafe_carla's Codon library hands over the SensorData instead.
    data = SimpleNamespace(as_lidar=lambda: _measurement(8))
    lidar._on_sensor_data(data)  # type: ignore[arg-type]  # noqa: SLF001
    assert [lidar._sweeps.get_nowait()[0] for _ in range(2)] == [7, 8]  # noqa: SLF001


def test_attach_pins_one_revolution_per_tick() -> None:
    blueprint = SimpleNamespace(attributes={}, has_attribute=lambda name: True)
    blueprint.set_attribute = blueprint.attributes.__setitem__
    world = SimpleNamespace(
        get_blueprint_library=lambda: SimpleNamespace(find=lambda name: blueprint),
        spawn_actor=lambda bp, transform, attach_to: SimpleNamespace(
            listen=lambda cb: None
        ),
    )
    CarlaLidarSensor(DriverLidarConfig(channels=32)).attach(
        world, SimpleNamespace(type_id="vehicle.lincoln.mkz")
    )
    assert blueprint.attributes["rotation_frequency"] == "20.0"
    assert blueprint.attributes["channels"] == "32"
    assert blueprint.attributes["dropoff_general_rate"] == "0.0"
    # Every tick: CARLA's default sensor_tick is left alone.
    assert "sensor_tick" not in blueprint.attributes


def test_attach_sweeps_once_an_interval_still_a_revolution_a_sweep() -> None:
    blueprint = SimpleNamespace(attributes={}, has_attribute=lambda name: True)
    blueprint.set_attribute = blueprint.attributes.__setitem__
    world = SimpleNamespace(
        get_blueprint_library=lambda: SimpleNamespace(find=lambda name: blueprint),
        spawn_actor=lambda bp, transform, attach_to: SimpleNamespace(
            listen=lambda cb: None
        ),
    )
    CarlaLidarSensor(DriverLidarConfig(), sweep_interval_s=0.1).attach(
        world, SimpleNamespace(type_id="vehicle.lincoln.mkz")
    )
    assert blueprint.attributes["rotation_frequency"] == "20.0"
    # Just under the interval, so two ticks' float deltas reach it.
    assert 0.0999 < float(blueprint.attributes["sensor_tick"]) < 0.1


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------


def _ring(range_m: float, n: int = 2000) -> np.ndarray:
    """*n* returns at *range_m* all around the sensor, at intensity 1."""
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    points = np.zeros((n, 4), dtype=np.float32)
    points[:, 0] = range_m * np.cos(angles)
    points[:, 1] = range_m * np.sin(angles)
    points[:, 3] = 1.0
    return points


def test_clear_weather_and_no_gain_leave_a_sweep_untouched() -> None:
    raw = _ring(50.0)
    assert apply_weather(raw, LidarWeather(), 1.0, 0, 0) is raw
    assert apply_weather(raw, LidarWeather(precipitation=100.0), 0.0, 0, 0) is raw


def test_fog_thins_far_returns_first_and_dims_them() -> None:
    fog = LidarWeather(fog_density=100.0, fog_density_scale=FOG_REFERENCE_SCALE)
    near = apply_weather(_ring(5.0), fog, 1.0, 0, 0)
    far = apply_weather(_ring(40.0), fog, 1.0, 0, 0)
    assert len(far) < len(near) < 2000
    far_ranges = np.linalg.norm(far[:, :3], axis=1)
    survivors = far[far_ranges > 30.0]
    assert survivors.size and float(survivors[:, 3].max()) < 0.01
    # The ones that came back early lie between the sensor and the ring.
    assert np.any(far_ranges < 30.0)


def test_fog_only_scatters_beyond_where_it_starts() -> None:
    raw = _ring(20.0)
    sweep = apply_weather(
        raw,
        LidarWeather(
            fog_density=100.0, fog_distance=30.0, fog_density_scale=FOG_REFERENCE_SCALE
        ),
        1.0,
        0,
        0,
    )
    np.testing.assert_allclose(sweep, raw)


def test_rain_moves_returns_along_their_rays() -> None:
    sweep = apply_weather(_ring(80.0), LidarWeather(precipitation=100.0), 1.0, 0, 0)
    ranges = np.linalg.norm(sweep[:, :3], axis=1)
    kept = ranges[ranges > 60.0]
    assert 0.0 < float(np.std(kept)) < 2.0
    assert len(sweep) < 2000


def test_the_fog_follows_the_servers_fog_density_scale() -> None:
    thin = apply_weather(_ring(40.0), LidarWeather(fog_density=100.0), 1.0, 0, 0)
    thick_fog = LidarWeather(fog_density=100.0, fog_density_scale=0.01)
    thick = apply_weather(_ring(40.0), thick_fog, 1.0, 0, 0)
    # CARLA's default draws a tenth of the fog the tuned scale does.
    assert len(thick) < len(thin)


def test_lidar_config_rejects_a_non_positive_fog_density_scale() -> None:
    with pytest.raises(ValueError, match="fog_density_scale"):
        DriverLidarConfig(fog_density_scale=0.0)


def test_weather_noise_scales_the_effect() -> None:
    fog = LidarWeather(fog_density=50.0)
    mild = apply_weather(_ring(40.0), fog, 0.5, 0, 0)
    strong = apply_weather(_ring(40.0), fog, 2.0, 0, 0)
    assert len(strong) < len(mild)


def test_lidar_config_rejects_negative_weather_noise_and_seed() -> None:
    with pytest.raises(ValueError, match="weather_noise"):
        DriverLidarConfig(weather_noise=-1.0)
    with pytest.raises(ValueError, match="noise_seed"):
        DriverLidarConfig(noise_seed=-1)


def _attached(lidar: CarlaLidarSensor, weather: SimpleNamespace) -> None:
    blueprint = SimpleNamespace(
        set_attribute=lambda name, value: None, has_attribute=lambda name: True
    )
    world = SimpleNamespace(
        get_blueprint_library=lambda: SimpleNamespace(find=lambda name: blueprint),
        spawn_actor=lambda bp, transform, attach_to: SimpleNamespace(
            listen=lambda cb: None
        ),
        get_weather=lambda: weather,
    )
    lidar.attach(world, SimpleNamespace(type_id="vehicle.lincoln.mkz"))


def _ring_measurement(frame: int) -> SimpleNamespace:
    return SimpleNamespace(frame=frame, raw_data=_ring(40.0).tobytes())


def test_the_sensor_puts_the_worlds_weather_on_its_sweeps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(carla_lidar, "_SWEEP_TIMEOUT", 0.05)
    weather = SimpleNamespace(precipitation=0.0, fog_density=80.0, fog_distance=0.0)
    lidar = CarlaLidarSensor(DriverLidarConfig())
    _attached(lidar, weather)
    lidar._receive(_ring_measurement(10))  # noqa: SLF001
    first = lidar.get_sweep(10)
    assert first is not None and len(first) < 2000
    # Asked again for the same sweep, the same noise.
    again = lidar.get_sweep(10)
    assert again is not None and np.array_equal(first, again)
    # The weather clears mid-run: the next sweep is CARLA's own.
    weather.fog_density = 0.0
    lidar._receive(_ring_measurement(11))  # noqa: SLF001
    clear = lidar.get_sweep(11)
    assert clear is not None and len(clear) == 2000


def test_the_weather_noise_repeats_with_its_seed_and_can_be_turned_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(carla_lidar, "_SWEEP_TIMEOUT", 0.05)
    weather = SimpleNamespace(precipitation=100.0, fog_density=0.0, fog_distance=0.0)
    sweeps = []
    for config in (
        DriverLidarConfig(noise_seed=3),
        DriverLidarConfig(noise_seed=3),
        DriverLidarConfig(noise_seed=4),
        DriverLidarConfig(weather_noise=0.0),
    ):
        lidar = CarlaLidarSensor(config)
        _attached(lidar, weather)
        lidar._receive(_ring_measurement(10))  # noqa: SLF001
        sweep = lidar.get_sweep(10)
        assert sweep is not None
        sweeps.append(sweep)
    assert np.array_equal(sweeps[0], sweeps[1])
    assert not np.array_equal(sweeps[0], sweeps[2])
    np.testing.assert_array_equal(sweeps[3], _ring(40.0))


# ---------------------------------------------------------------------------
# Payload
# ---------------------------------------------------------------------------


def _sweep() -> carla_driver_pb2.LidarSweep:
    return pack_lidar_sweep(
        "lidar_top", 1_000, np.ones((5, 4)), Pose.identity().to_proto()
    )


def test_sweeps_ride_in_the_renderer_payload() -> None:
    builder = RendererDataBuilder(_world(), _ego(), DriverClientConfig())
    message = carla_driver_pb2.CarlaRendererData()
    message.ParseFromString(builder.build(1_000, Pose.identity(), lidar=[_sweep()]))
    assert [sweep.logical_id for sweep in message.lidar] == ["lidar_top"]
    assert message.map_name == "Town10HD_Opt"


def test_a_ground_truth_failure_does_not_cost_the_sweep() -> None:
    world = _world()
    world.get_weather.side_effect = RuntimeError("CARLA hiccup")
    builder = RendererDataBuilder(world, _ego(), DriverClientConfig())
    assert builder.build(1_000, Pose.identity()) == b""
    message = carla_driver_pb2.CarlaRendererData()
    message.ParseFromString(builder.build(1_000, Pose.identity(), lidar=[_sweep()]))
    assert len(message.lidar) == 1 and message.snapshot_timestamp_us == 1_000
