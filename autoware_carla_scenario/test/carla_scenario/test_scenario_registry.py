"""Unit tests for the pluggable scenario registry (Issue #420)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import typesafe_carla.carla as carla
import pytest
from omegaconf import DictConfig, OmegaConf

from autoware_carla_scenario import BaseScenario, EgoConfig, SpawnTransform


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummyConfig:
    """Minimal config class that stores kwargs as attributes."""

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


class _DummyScenario(BaseScenario):
    """Scenario stub for registry tests."""

    def __init__(self, ego_config: EgoConfig, **kwargs: Any) -> None:
        super().__init__(ego_config)
        self.init_kwargs = kwargs

    def setup(self) -> None:
        pass

    def is_done(self) -> bool:
        return True


def _make_ego() -> EgoConfig:
    return EgoConfig(
        spawn_location=SpawnTransform(
            carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0))
        ),
        vehicle_type="vehicle.mini.cooper",
    )


def _make_unknown_scenario_cfg() -> DictConfig:
    """Config with an unregistered scenario name, suitable for error-path tests."""
    return OmegaConf.create(
        {
            "scenario": {"name": "__nonexistent__"},
            "entity": {
                "ground_projection_ray_distance_upper": 10.0,
                "ground_projection_ray_distance_lower": 1.0,
                "spawn_retry_max_count": 3,
                "spawn_retry_t_step": 0.1,
                "spawn_retry_z_step": 0.1,
            },
            "ego": {
                "vehicle_type": "vehicle.mini.cooper",
                "initial_speed_kmh": 0.0,
                "spawn_lanelet_id": 1,
                "spawn_s": 0.0,
            },
        }
    )


# ---------------------------------------------------------------------------
# Tests for registry API
# ---------------------------------------------------------------------------


class TestScenarioRegistry:
    """Tests for register_scenario, register_scenario_builder, get_scenario_registry."""

    def test_builtin_scenarios_are_registered(self) -> None:
        """All four built-in scenarios should be present in the registry."""
        from autoware_carla_scenario.examples.run import get_scenario_registry

        registry = get_scenario_registry()
        expected = {
            "intersection_passing",
            "traffic_light_compliance",
            "lane_change",
            "temporary_stop",
        }
        assert expected.issubset(set(registry.keys()))

    def test_get_scenario_registry_returns_copy(self) -> None:
        """Mutating the returned dict must not affect the internal registry."""
        from autoware_carla_scenario.examples.run import get_scenario_registry

        copy1 = get_scenario_registry()
        copy1["__test_sentinel__"] = lambda *_: None  # type: ignore[assignment]
        copy2 = get_scenario_registry()
        assert "__test_sentinel__" not in copy2

    def test_register_scenario(self) -> None:
        """register_scenario() should create a builder in the registry."""
        from autoware_carla_scenario import (
            get_scenario_builder,
            get_scenario_registry,
            register_scenario,
            unregister_scenario,
        )

        name = "__test_register_scenario__"
        try:
            register_scenario(name, _DummyScenario, _DummyConfig)
            assert name in get_scenario_registry()

            # Verify the builder works.
            ego = _make_ego()
            spawn_pose = MagicMock()
            ground_projection = MagicMock()
            scenario_dict: dict[str, Any] = {"name": name, "foo": "bar"}
            builder = get_scenario_builder(name)
            assert builder is not None
            result = builder(ego, scenario_dict, spawn_pose, ground_projection)
            assert isinstance(result, _DummyScenario)
        finally:
            unregister_scenario(name)

    def test_register_scenario_builder(self) -> None:
        """register_scenario_builder() should accept a custom callable."""
        from autoware_carla_scenario import (
            get_scenario_builder,
            register_scenario_builder,
            unregister_scenario,
        )

        name = "__test_register_builder__"
        sentinel = _DummyScenario(_make_ego())

        def custom_builder(
            ego: EgoConfig,
            scenario_dict: dict[str, Any],
            spawn_pose: Any,
            ground_projection: Any,
        ) -> BaseScenario:
            return sentinel

        try:
            register_scenario_builder(name, custom_builder)
            builder = get_scenario_builder(name)
            assert builder is custom_builder
            result = builder(_make_ego(), {}, MagicMock(), MagicMock())
            assert result is sentinel
        finally:
            unregister_scenario(name)

    def test_register_overwrites_existing(self) -> None:
        """Re-registering under the same name should overwrite silently."""
        from autoware_carla_scenario import (
            get_scenario_builder,
            register_scenario,
            unregister_scenario,
        )

        name = "__test_overwrite__"
        try:
            register_scenario(name, _DummyScenario, _DummyConfig)
            old_builder = get_scenario_builder(name)
            register_scenario(name, _DummyScenario, _DummyConfig)
            # Builder should be a new closure.
            assert get_scenario_builder(name) is not old_builder
        finally:
            unregister_scenario(name)


# ---------------------------------------------------------------------------
# Tests for build_scenario with injectable builder
# ---------------------------------------------------------------------------


class TestBuildScenarioInjectable:
    """Tests for the build_scenario_fn parameter."""

    def test_build_scenario_fn_overrides_registry(self) -> None:
        """When build_scenario_fn is given, it should be used instead of the registry."""
        from autoware_carla_scenario.examples.run import build_scenario

        ego = _make_ego()
        sentinel = _DummyScenario(ego)

        def custom_fn(cfg: DictConfig) -> tuple[EgoConfig, BaseScenario]:
            return ego, sentinel

        cfg = OmegaConf.create({"scenario": {"name": "nonexistent"}})
        result_ego, result_scenario = build_scenario(cfg, build_scenario_fn=custom_fn)
        assert result_ego is ego
        assert result_scenario is sentinel

    def test_an_injected_scenario_is_given_the_configured_waypoints(self) -> None:
        """The way there must reach an injected scenario, like the goal does.

        `build_scenario_fn` bypasses the constructor `build_ego_and_spawn`
        fills, so whatever `_apply_ego_config` leaves behind is dropped
        silently.  Dropping the waypoints would let Autoware plan the shortcut
        they exist to prevent, on exactly the path downstream projects use.
        """
        from autoware_carla_scenario.examples.run import build_scenario

        ego = _make_ego()
        sentinel = _DummyScenario(ego)

        cfg = OmegaConf.create(
            {
                "scenario": {"name": "nonexistent"},
                "ego": {"waypoint_lanelet_ids": [11, 22]},
            }
        )
        build_scenario(cfg, build_scenario_fn=lambda _cfg: (ego, sentinel))

        assert [p.lanelet_id for p in sentinel.waypoint_poses] == [11, 22]

    def test_an_injected_scenario_keeps_its_own_waypoints_when_none_configured(
        self,
    ) -> None:
        """No waypoints configured means none were asked for, not none intended.

        A scenario may derive its own in `setup()`; an empty config list must
        not overwrite them, for the same reason a missing goal does not.
        """
        from autoware_carla_scenario.coordinate import Lanelet2Pose
        from autoware_carla_scenario.examples.run import build_scenario

        ego = _make_ego()
        sentinel = _DummyScenario(ego)
        sentinel.waypoint_poses = [Lanelet2Pose(lanelet_id=99, s=0.0)]

        cfg = OmegaConf.create({"scenario": {"name": "nonexistent"}})
        build_scenario(cfg, build_scenario_fn=lambda _cfg: (ego, sentinel))

        assert [p.lanelet_id for p in sentinel.waypoint_poses] == [99]

    def test_build_scenario_unknown_name_raises(self) -> None:
        """build_scenario should raise ValueError for unregistered names."""
        from autoware_carla_scenario.examples.run import build_scenario

        with pytest.raises(ValueError, match="Unknown scenario name"):
            build_scenario(_make_unknown_scenario_cfg())

    def test_build_scenario_unknown_name_lists_registered(self) -> None:
        """The error message should list registered scenario names."""
        from autoware_carla_scenario.examples.run import build_scenario

        with pytest.raises(ValueError, match="Registered scenarios"):
            build_scenario(_make_unknown_scenario_cfg())


# ---------------------------------------------------------------------------
# Tests for build_ego_entity
# ---------------------------------------------------------------------------


class TestBuildEgoEntity:
    """Tests for the ego entity selected by build_ego_entity()."""

    def test_missing_ego_group_selects_autopilot(self) -> None:
        """A config with no 'ego' group at all should select autopilot, not raise.

        build_scenario() calls build_ego_entity() even when an injected
        build_scenario_fn supplied the EgoConfig, and such a config has no reason
        to carry the group Hydra would compose.
        """
        from autoware_carla_scenario.examples.run import build_ego_entity

        cfg = OmegaConf.create({"scenario": {"name": "whatever"}})
        assert build_ego_entity(cfg) is None

    def test_empty_ego_group_selects_autopilot(self) -> None:
        """An 'ego' group without an 'entity' key should select autopilot."""
        from autoware_carla_scenario.examples.run import build_ego_entity

        cfg = OmegaConf.create({"ego": {"vehicle_type": "vehicle.tesla.model3"}})
        assert build_ego_entity(cfg) is None

    def test_unknown_entity_raises(self) -> None:
        """An unrecognised ego.entity should raise rather than silently autopilot."""
        from autoware_carla_scenario.examples.run import build_ego_entity

        cfg = OmegaConf.create({"ego": {"entity": "__nope__"}})
        with pytest.raises(ValueError, match="Unknown ego.entity"):
            build_ego_entity(cfg)


class TestBuildAutowareLauncher:
    """``autoware.launcher`` selects who starts Autoware for each scenario."""

    @staticmethod
    def _cfg(entity: str = "autoware", **launcher: object) -> DictConfig:
        return OmegaConf.create(
            {
                "ego": {"entity": entity},
                "scenario": {"name": "left_turn"},
                "autoware": {"address": "localhost:0", "launcher": launcher},
            }
        )

    def test_none_by_default(self) -> None:
        from autoware_carla_scenario.examples.run import build_autoware_launcher

        assert build_autoware_launcher(self._cfg(type="none")) is None
        assert (
            build_autoware_launcher(OmegaConf.create({"ego": {"entity": "autoware"}}))
            is None
        )

    def test_only_an_autoware_ego_has_one(self) -> None:
        from autoware_carla_scenario.examples.run import build_autoware_launcher

        cfg = self._cfg("autopilot", type="command", command=["true"])
        assert build_autoware_launcher(cfg) is None

    def test_docker_is_shared_by_every_ego_of_the_batch(self, tmp_path) -> None:
        from autoware_carla_scenario.autoware_stack import DockerAutowareLauncher
        from autoware_carla_scenario.examples.run import (
            build_autoware_launcher,
            build_ego_entity,
        )

        cfg = self._cfg(
            type="docker",
            workspace=str(tmp_path),
            launch=[
                "autoware_launch",
                "x.launch.xml",
                "bridge_address:={bridge_address}",
            ],
            ros_domain_ids=[3, 4],
            # The command launcher's key: the group holds both types' keys.
            command=["true"],
        )
        launcher = build_autoware_launcher(cfg)

        assert isinstance(launcher, DockerAutowareLauncher)
        assert launcher.config.workspace == tmp_path
        assert launcher.config.ros_domain_ids == (3, 4)
        assert build_autoware_launcher(cfg) is launcher
        entity = build_ego_entity(cfg)
        assert entity is not None
        assert entity.launcher is launcher  # type: ignore[attr-defined]

    def test_command(self) -> None:
        from autoware_carla_scenario.autoware_stack import CommandAutowareLauncher
        from autoware_carla_scenario.examples.run import build_autoware_launcher

        launcher = build_autoware_launcher(
            self._cfg(type="command", command=["ros2", "launch", "{bridge_address}"])
        )

        assert isinstance(launcher, CommandAutowareLauncher)
        assert tuple(launcher.argv) == ("ros2", "launch", "{bridge_address}")

    def test_unknown_type_raises(self) -> None:
        from autoware_carla_scenario.examples.run import build_autoware_launcher

        with pytest.raises(ValueError, match="Unknown autoware.launcher.type"):
            build_autoware_launcher(self._cfg(type="kubernetes"))

    @pytest.mark.parametrize("kind", ["none", "docker", "command"])
    @pytest.mark.parametrize("typo", ["stop_timout_s", "workspce"])
    def test_an_unknown_key_is_refused(self, kind: str, typo: str) -> None:
        """#1413: a key neither launcher reads is a typo, not a default."""
        from autoware_carla_scenario.examples.run import build_autoware_launcher

        cfg = self._cfg(
            type=kind,
            workspace="/nowhere",
            launch=["autoware_launch", "x.launch.xml"],
            command=["true"],
        )
        cfg.autoware.launcher[typo] = 60
        with pytest.raises(
            ValueError,
            match=rf"^Unknown autoware\.launcher key\(s\): "
            rf"\['autoware\.launcher\.{typo}'\]\. Known keys: .*'stop_timeout_s'",
        ):
            build_autoware_launcher(cfg)

    def test_the_shipped_launcher_keys_are_accepted(self) -> None:
        """The shipped config holds the keys of both types; each type takes it."""
        from hydra import compose, initialize_config_dir

        from autoware_carla_scenario.autoware_stack import (
            CommandAutowareLauncher,
            DockerAutowareLauncher,
        )
        from autoware_carla_scenario.examples.run import build_autoware_launcher

        conf = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "autoware_carla_scenario"
            / "examples"
            / "conf"
        )
        common = [
            "ego.entity=autoware",
            "autoware.launcher.launch=[autoware_launch,x.launch.xml]",
            "autoware.launcher.command=[ros2,launch]",
            "autoware.launcher.stop_timeout_s=7",
            "+autoware.launcher.docker=podman",
            "+autoware.launcher.container_prefix=acs-test",
        ]
        with initialize_config_dir(config_dir=str(conf), version_base=None):
            docker = compose(
                config_name="config",
                overrides=[*common, "autoware.launcher.type=docker"],
            )
            command = compose(
                config_name="config",
                overrides=[*common, "autoware.launcher.type=command"],
            )
            none = compose(config_name="config", overrides=common)

        docker_launcher = build_autoware_launcher(docker)
        assert isinstance(docker_launcher, DockerAutowareLauncher)
        assert docker_launcher.config.stop_timeout_s == 7.0
        assert docker_launcher.config.docker == "podman"
        assert docker_launcher.config.container_prefix == "acs-test"
        command_launcher = build_autoware_launcher(command)
        assert isinstance(command_launcher, CommandAutowareLauncher)
        assert tuple(command_launcher.argv) == ("ros2", "launch")
        assert build_autoware_launcher(none) is None

    @pytest.mark.parametrize("kind", ["docker", "command"])
    @pytest.mark.parametrize(
        "value", [float("nan"), float("inf"), -5, 0, "abc", None, True]
    )
    def test_a_stop_timeout_that_is_not_positive_and_finite_is_refused(
        self, kind: str, value: object
    ) -> None:
        """#1450: a NaN never escalates a stack that ignores SIGINT, a negative
        one skips its grace, and a string used to fail only at stop time."""
        from autoware_carla_scenario.examples.run import build_autoware_launcher

        cfg = self._cfg(
            type=kind,
            workspace="/nowhere",
            launch=["autoware_launch", "x.launch.xml"],
            command=["true"],
            stop_timeout_s=value,
        )
        with pytest.raises(
            ValueError,
            match=r"^autoware\.launcher\.stop_timeout_s must be a finite number > 0",
        ):
            build_autoware_launcher(cfg)

    @pytest.mark.parametrize(
        ("value", "message"),
        [
            # The range is DockerAutowareConfig's, prefixed with the group.
            ([-3, 999], r"must be ROS domain ids, whole numbers from 0 to 232, got -3"),
            ([3, 233], r"must be ROS domain ids, whole numbers from 0 to 232, got 233"),
            # The types are build_autoware_launcher's: no raw TypeError.
            (7, r"must be a list of ROS domain ids"),
            ("7", r"must be a list of ROS domain ids"),
            ([3, True], r"must be a list of ROS domain ids"),
            ([3, 4.5], r"must be a list of ROS domain ids"),
            ([3, "4"], r"must be a list of ROS domain ids"),
        ],
    )
    def test_a_ros_domain_id_that_is_not_one_is_refused(
        self, value: object, message: str
    ) -> None:
        """#1450: refused up front, not by the stack's DDS in the container."""
        from autoware_carla_scenario.examples.run import build_autoware_launcher

        cfg = self._cfg(
            type="docker",
            workspace="/nowhere",
            launch=["autoware_launch", "x.launch.xml"],
            ros_domain_ids=value,
        )
        with pytest.raises(
            ValueError, match=rf"^autoware\.launcher\.ros_domain_ids {message}"
        ):
            build_autoware_launcher(cfg)

    def test_the_ros_domain_id_range_and_null_are_taken(self) -> None:
        from autoware_carla_scenario.autoware_stack import DockerAutowareLauncher
        from autoware_carla_scenario.examples.run import build_autoware_launcher

        launch = ["autoware_launch", "x.launch.xml"]
        launcher = build_autoware_launcher(
            self._cfg(
                type="docker",
                workspace="/nowhere",
                launch=launch,
                ros_domain_ids=[0, 232],
                stop_timeout_s=1,
            )
        )
        assert isinstance(launcher, DockerAutowareLauncher)
        assert launcher.config.ros_domain_ids == (0, 232)
        assert launcher.config.stop_timeout_s == 1.0
        launcher = build_autoware_launcher(
            self._cfg(
                type="docker",
                workspace="/nowhere",
                launch=launch,
                ros_domain_ids=None,
            )
        )
        assert isinstance(launcher, DockerAutowareLauncher)
        assert launcher.config.ros_domain_ids == ()

    @pytest.mark.parametrize(
        ("kind", "name"),
        [
            ("docker", "launch"),
            ("docker", "colcon_args"),
            ("docker", "docker_args"),
            ("command", "command"),
        ],
    )
    @pytest.mark.parametrize("value", ["--shm-size=8g", "ros2 launch x y", 7, {}])
    def test_an_argument_list_that_is_not_a_list_is_refused(
        self, kind: str, name: str, value: object
    ) -> None:
        """#1484: a string used to be split into its characters."""
        from autoware_carla_scenario.examples.run import build_autoware_launcher

        cfg = self._cfg(
            type=kind,
            workspace="/nowhere",
            launch=["autoware_launch", "x.launch.xml"],
            command=["true"],
        )
        cfg.autoware.launcher[name] = value
        with pytest.raises(
            ValueError,
            match=rf"^autoware\.launcher\.{name} must be a list of arguments, not ",
        ):
            build_autoware_launcher(cfg)

    @pytest.mark.parametrize(
        ("kind", "name"), [("docker", "docker_args"), ("command", "command")]
    )
    @pytest.mark.parametrize("item", [["a"], {"a": 1}, None, True])
    def test_an_argument_that_is_not_a_string_or_number_is_refused(
        self, kind: str, name: str, item: object
    ) -> None:
        """#1484: its str() (``"['a']"``, ``"None"``) is not the argument meant."""
        from autoware_carla_scenario.examples.run import build_autoware_launcher

        cfg = self._cfg(
            type=kind,
            workspace="/nowhere",
            launch=["autoware_launch", "x.launch.xml"],
            command=["true"],
        )
        cfg.autoware.launcher[name] = ["ok", item]
        with pytest.raises(
            ValueError,
            match=rf"^autoware\.launcher\.{name}\[1\] must be a string or a number",
        ):
            build_autoware_launcher(cfg)

    def test_argument_lists_are_made_strings(self) -> None:
        from autoware_carla_scenario.autoware_stack import (
            CommandAutowareLauncher,
            DockerAutowareLauncher,
        )
        from autoware_carla_scenario.examples.run import build_autoware_launcher

        docker = build_autoware_launcher(
            self._cfg(
                type="docker",
                workspace="/nowhere",
                launch=["autoware_launch", "x.launch.xml", "rate:=10"],
                colcon_args=None,
                docker_args=["--shm-size=8g", "--cpus", 4],
                env={"A": 1, "B": "two words"},
                mounts={"/data": "/home/aw/data"},
            )
        )
        assert isinstance(docker, DockerAutowareLauncher)
        assert docker.config.launch == ("autoware_launch", "x.launch.xml", "rate:=10")
        assert docker.config.colcon_args == ()
        assert docker.config.docker_args == ("--shm-size=8g", "--cpus", "4")
        assert docker.config.env == {"A": "1", "B": "two words"}
        assert docker.config.mounts == {"/data": "/home/aw/data"}
        command = build_autoware_launcher(
            self._cfg(type="command", command=["sleep", 5], env=None)
        )
        assert isinstance(command, CommandAutowareLauncher)
        assert tuple(command.argv) == ("sleep", "5")
        assert command.env == {}

    @pytest.mark.parametrize(
        ("kind", "name", "value", "message"),
        [
            ("docker", "env", "A=1", r"env must be a mapping"),
            ("command", "env", ["A=1"], r"env must be a mapping"),
            ("docker", "mounts", "/data:/data", r"mounts must be a mapping"),
            ("docker", "env", {"A": None}, r"env\.A must be a string or a number"),
            ("command", "env", {"A": [1]}, r"env\.A must be a string or a number"),
            (
                "docker",
                "bridge_parameters",
                "x",
                r"bridge_parameters must be a mapping",
            ),
            (
                "command",
                "bridge_parameters",
                [1],
                r"bridge_parameters must be a mapping",
            ),
        ],
    )
    def test_a_mapping_setting_that_is_not_one_is_refused(
        self, kind: str, name: str, value: object, message: str
    ) -> None:
        """#1484: refused naming the key, not by an AttributeError at run time."""
        from autoware_carla_scenario.examples.run import build_autoware_launcher

        cfg = self._cfg(
            type=kind,
            workspace="/nowhere",
            launch=["autoware_launch", "x.launch.xml"],
            command=["true"],
        )
        cfg.autoware.launcher[name] = value
        with pytest.raises(ValueError, match=rf"^autoware\.launcher\.{message}"):
            build_autoware_launcher(cfg)

    def test_a_config_refusal_names_the_group(self) -> None:
        """DockerAutowareConfig's own refusals are prefixed with the group."""
        from autoware_carla_scenario.examples.run import build_autoware_launcher

        with pytest.raises(
            ValueError, match=r"^autoware\.launcher\.launch is required"
        ):
            build_autoware_launcher(self._cfg(type="docker", workspace="/nowhere"))


# ---------------------------------------------------------------------------
# Tests for build_ego_and_spawn helper
# ---------------------------------------------------------------------------


def _ego_and_spawn_cfg(**ego_overrides: object) -> DictConfig:
    """Return a resolved-config shape build_ego_and_spawn() accepts."""
    ego: dict[str, object] = {
        "vehicle_type": "vehicle.tesla.model3",
        "initial_speed_kmh": 30.0,
        "spawn_lanelet_id": 42,
        "spawn_s": 5.0,
    }
    ego.update(ego_overrides)
    return OmegaConf.create(
        {
            "entity": {
                "ground_projection_ray_distance_upper": 10.0,
                "ground_projection_ray_distance_lower": 1.0,
                "spawn_retry_max_count": 5,
                "spawn_retry_t_step": 0.2,
                "spawn_retry_z_step": 0.3,
            },
            "ego": ego,
        }
    )


class TestBuildEgoAndSpawn:
    """Tests for the build_ego_and_spawn() helper."""

    def test_returns_ego_spawn_and_ground_projection(self) -> None:
        """build_ego_and_spawn should return a 3-tuple from the config."""
        from autoware_carla_scenario import (
            GroundProjectionConfig,
            Lanelet2Pose,
        )
        from autoware_carla_scenario.examples.run import build_ego_and_spawn

        ego, spawn_pose, ground_projection = build_ego_and_spawn(_ego_and_spawn_cfg())

        assert isinstance(ego, EgoConfig)
        assert ego.vehicle_type == "vehicle.tesla.model3"
        assert ego.initial_speed_kmh == 30.0

        assert isinstance(spawn_pose, Lanelet2Pose)
        assert spawn_pose.lanelet_id == 42
        assert spawn_pose.s == 5.0

        assert isinstance(ground_projection, GroundProjectionConfig)
        assert ground_projection.ray_distance_upper == 10.0
        assert ground_projection.ray_distance_lower == 1.0

    def test_the_configured_goal_travels_with_the_ego_config(self) -> None:
        """The goal is part of the ego's configuration, not a separate parcel."""
        from autoware_carla_scenario.examples.run import build_ego_and_spawn

        cfg = _ego_and_spawn_cfg(goal_lanelet_id=265, goal_s=12.5)
        ego, _, _ = build_ego_and_spawn(cfg)

        assert ego.goal_pose is not None
        assert ego.goal_pose.lanelet_id == 265
        assert ego.goal_pose.s == 12.5

    def test_a_config_without_a_goal_waits_for_the_scenario(self) -> None:
        """A scenario may know the destination the config does not.

        ``IntersectionPassingScenario`` derives the ego's goal in ``setup()``
        from the route it asserts, so a config that names none is not yet wrong;
        ``register_route_to_goal`` refuses an ego that still has nowhere to go.
        """
        from autoware_carla_scenario.examples.run import build_ego_and_spawn

        ego, _, _ = build_ego_and_spawn(_ego_and_spawn_cfg(entity="autoware"))

        assert ego.goal_pose is None


# ---------------------------------------------------------------------------
# Tests for build_goal_pose helper
# ---------------------------------------------------------------------------


class TestBuildGoalPose:
    """``ego.goal_lanelet_id`` is what routes an ego that plans for itself."""

    def test_a_configured_goal_becomes_a_lanelet2_pose(self) -> None:
        from autoware_carla_scenario import Lanelet2Pose
        from autoware_carla_scenario.examples.run import build_goal_pose

        cfg = OmegaConf.create({"ego": {"goal_lanelet_id": 265, "goal_s": 12.5}})
        goal = build_goal_pose(cfg)

        assert isinstance(goal, Lanelet2Pose)
        assert goal.lanelet_id == 265
        assert goal.s == 12.5

    def test_no_goal_configured_returns_none(self) -> None:
        # The ordinary case: only an ego that routes itself needs a goal, so an
        # unset one must not become lanelet 0.
        from autoware_carla_scenario.examples.run import build_goal_pose

        cfg = OmegaConf.create({"ego": {"spawn_lanelet_id": 242}})
        assert build_goal_pose(cfg) is None

    def test_missing_ego_group_returns_none(self) -> None:
        from autoware_carla_scenario.examples.run import build_goal_pose

        cfg = OmegaConf.create({"scenario": {"name": "whatever"}})
        assert build_goal_pose(cfg) is None

    def test_goal_s_defaults_to_the_start_of_the_lanelet(self) -> None:
        from autoware_carla_scenario.examples.run import build_goal_pose

        cfg = OmegaConf.create({"ego": {"goal_lanelet_id": 265}})
        goal = build_goal_pose(cfg)

        assert goal is not None
        assert goal.s == 0.0


def test_run_batch_gives_each_scenario_its_own_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from autoware_carla_scenario.examples import run

    monkeypatch.setenv("AUTOWARE_CARLA_SCENARIO_TYPECHECK", "off")
    built: list[object] = []

    def fake_build(cfg: object, **_: object) -> tuple[None, object]:
        built.append(object())
        return None, built[-1]

    added: list[tuple[object, float | None]] = []

    class FakeQueue:
        def __init__(self, **_: object) -> None:
            pass

        def add(
            self,
            scenario: object,
            *,
            timeout_seconds: float | None = None,
            case: object = None,
        ) -> None:
            added.append((scenario, timeout_seconds))

        def __enter__(self) -> "FakeQueue":
            return self

        def __exit__(self, *_: object) -> None:
            pass

        def run_all(self, **_: object) -> list[object]:
            return []

        def errored_indices(self) -> list[int]:
            return []

    monkeypatch.setattr(run, "build_scenario", fake_build)
    monkeypatch.setattr(run, "ScenarioQueue", FakeQueue)
    monkeypatch.setattr(
        run,
        "resolve_map_paths",
        lambda _m: SimpleNamespace(
            install_xodr=None,
            overwrite_xodr=False,
            opendrive_path=None,
            lanelet2_path=None,
            name="M",
            projector_type=None,
        ),
    )
    monkeypatch.setattr(run, "build_traffic_backend", lambda _cfg: None)
    monkeypatch.setattr(run, "_make_batch_output_dir", lambda: None)
    monkeypatch.setattr(run, "_print_summary", lambda *a, **k: True)

    with pytest.raises(SystemExit):
        run.run_batch(["lane_change/left", "lane_change_fail/right"], [])

    assert added == [(built[0], 10.0), (built[1], 1.5)]


def test_a_batch_lets_go_of_each_scenario_once_it_has_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A batch keeps each case's result, not the finished scenario: one kept to
    the end keeps everything its run built (entities, conditions, the ego's
    driver), and a long batch grows with every case."""
    import gc
    import weakref
    from types import SimpleNamespace

    from autoware_carla_scenario import ScenarioResult, scenario_queue
    from autoware_carla_scenario.examples import run

    monkeypatch.setenv("AUTOWARE_CARLA_SCENARIO_TYPECHECK", "off")

    class _Built:
        """What a built scenario stands for here; weakly referenceable."""

    built: list[weakref.ref[_Built]] = []

    def fake_build(cfg: object, **_: object) -> tuple[None, object]:
        scenario = _Built()
        built.append(weakref.ref(scenario))
        return None, scenario

    alive_while_running: list[list[int]] = []

    class FakeRunner:
        """Keeps no scenario (a MagicMock keeps every argument it is given)."""

        def __init__(self, *_: object, **__: object) -> None:
            self.xodr_path = None

        def load_map_by_name(self, _name: str) -> None:
            pass

        def run_scenario(self, scenario: object, **_: object) -> ScenarioResult:
            gc.collect()
            index = next(k for k, ref in enumerate(built) if ref() is scenario)
            alive_while_running.append(
                [k for k, ref in enumerate(built[:index]) if ref() is not None]
            )
            return ScenarioResult(passed=True, message="ok", elapsed_seconds=0.0)

    summarised: list[object] = []

    def fake_summary(_labels: object, results: list[object], **_: object) -> bool:
        summarised.extend(results)
        return True

    monkeypatch.setattr(run, "build_scenario", fake_build)
    monkeypatch.setattr(scenario_queue, "CarlaServerManager", MagicMock())
    monkeypatch.setattr(scenario_queue, "ScenarioRunner", FakeRunner)
    monkeypatch.setattr(
        run,
        "resolve_map_paths",
        lambda _m: SimpleNamespace(
            install_xodr=None,
            overwrite_xodr=False,
            opendrive_path=None,
            lanelet2_path=None,
            name="M",
            projector_type=None,
        ),
    )
    monkeypatch.setattr(run, "build_traffic_backend", lambda _cfg: None)
    monkeypatch.setattr(run, "_make_batch_output_dir", lambda: None)
    monkeypatch.setattr(run, "_print_summary", fake_summary)

    with pytest.raises(SystemExit):
        run.run_batch(
            ["lane_change/left", "lane_change_fail/right", "lane_change/left"], []
        )

    # While each case ran, every case before it was already gone ...
    assert alive_while_running == [[], [], []]
    # ... and the summary still has every result.
    assert len(summarised) == 3


def test_a_batch_goes_on_past_a_case_that_cannot_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A case that raises on every attempt is reported as errored, the cases
    after it still run, and the batch exits non-zero."""
    import json
    from types import SimpleNamespace

    from autoware_carla_scenario import ScenarioResult, scenario_queue
    from autoware_carla_scenario.examples import run

    monkeypatch.setenv("AUTOWARE_CARLA_SCENARIO_TYPECHECK", "off")
    built: list[object] = []

    def fake_build(cfg: object, **_: object) -> tuple[None, object]:
        built.append(object())
        return None, built[-1]

    ran: list[int] = []

    class FakeRunner:
        def __init__(self, *_: object, **__: object) -> None:
            self.xodr_path = None

        def load_map_by_name(self, _name: str) -> None:
            pass

        def reset_world(self, scenario_name: str = "") -> None:
            pass

        def run_scenario(self, scenario: object, **_: object) -> ScenarioResult:
            index = next(k for k, b in enumerate(built) if b is scenario)
            ran.append(index)
            if index == 1:
                raise RuntimeError("ground_projection returned no hit")
            return ScenarioResult(passed=True, message="ok", elapsed_seconds=1.0)

    monkeypatch.setattr(run, "build_scenario", fake_build)
    monkeypatch.setattr(scenario_queue, "CarlaServerManager", MagicMock())
    monkeypatch.setattr(scenario_queue, "ScenarioRunner", FakeRunner)
    monkeypatch.setattr(
        run,
        "resolve_map_paths",
        lambda _m: SimpleNamespace(
            install_xodr=None,
            overwrite_xodr=False,
            opendrive_path=None,
            lanelet2_path=None,
            name="M",
            projector_type=None,
        ),
    )
    monkeypatch.setattr(run, "build_traffic_backend", lambda _cfg: None)
    monkeypatch.setattr(run, "_make_batch_output_dir", lambda: tmp_path)
    monkeypatch.setattr(scenario_queue, "_wait_with_progress", lambda *_, **__: None)

    with pytest.raises(SystemExit) as exited:
        run.run_batch(
            ["lane_change/left", "lane_change_fail/right", "lane_change/left"], []
        )

    assert exited.value.code == 1
    assert ran == [0, 1, 2]
    out = capsys.readouterr().out
    assert "[ERROR] lane_change_fail/right" in out
    assert "ground_projection returned no hit" in out
    assert out.count("[PASS] lane_change/left") == 2
    assert "2/3 scenarios passed, 1 errored" in out
    entries = json.loads((tmp_path / "batch_results.json").read_text())
    assert [e.get("error", False) for e in entries] == [False, True, False]
    assert [e["passed"] for e in entries] == [True, False, True]
