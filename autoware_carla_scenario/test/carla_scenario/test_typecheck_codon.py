"""The static check with Codon: correct scenarios compile, wrong ones are refused.

Skipped where no Codon compiler is installed (typesafe-carla's toolchain, a
dependency on Linux x86_64 and aarch64).
"""

from __future__ import annotations

import importlib
import re
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
import yaml

from autoware_carla_scenario.typecheck import (
    ScenarioTypeError,
    TypeCheckResult,
    available_toolchain,
    typecheck_scenario,
)

pytestmark = pytest.mark.skipif(
    available_toolchain() is None, reason="no Codon compiler"
)

_REPO = Path(__file__).resolve().parents[3]
_TEMPLATE_SRC = _REPO / "examples" / "scenario_package_template" / "src"

_CONFIG = """
from dataclasses import dataclass, field


@dataclass
class CaseConfig:
    name: str = "case"
    goal_lanelet_ids: list[int] = field(default_factory=lambda: [460, 265])
    timeout_seconds: float = 10.0
    min_speed_kmh: float | None = None
"""

_HEADER = """
from __future__ import annotations

import logging

import typesafe_carla.carla as carla

from autoware_carla_scenario import (
    EGO_ROLE_NAME,
    AndCondition,
    BaseCondition,
    BaseScenario,
    ComparisonRule,
    EgoConfig,
    ElapsedTimeCondition,
    FollowTrajectoryAction,
    GroundProjectionConfig,
    LaneChangeDirection,
    Lanelet2Pose,
    MapPose,
    OpenDrivePose,
    ReferenceContext,
    RelativeLanePose,
    SamplingEvent,
    ScenarioResult,
    SpeedCondition,
    StickyCondition,
    TemporaryStopCondition,
    TickTiming,
    TimeoutCondition,
    Trajectory,
    TrajectoryFollowingMode,
    TrajectoryTimeCondition,
    TrajectoryTiming,
    TrajectoryVertex,
    TurnAction,
    TurnDirection,
    TurnIndicatorAction,
    TurnIndicatorCondition,
    TurnIndicatorState,
    snap_to_carla_road,
)

from .configs import CaseConfig

logger = logging.getLogger(__name__)


class CaseScenario(BaseScenario):
    _config: CaseConfig

    def __init__(
        self,
        ego_config: EgoConfig,
        spawn_pose: Lanelet2Pose,
        config: CaseConfig | None = None,
        ground_projection: GroundProjectionConfig | None = None,
    ) -> None:
        super().__init__(ego_config, spawn_pose=spawn_pose, ground_projection=ground_projection)
        self._config = config or CaseConfig()

    def is_done(self) -> bool:
        return False

    def setup(self) -> None:
        cfg = self._config
"""

_VALID_SETUP = """
        self.derive_goal_from_route(cfg.goal_lanelet_ids)
        self._setup_ego_spawn()
        self.register_pre_tick(
            TurnAction(EGO_ROLE_NAME, TurnDirection.LEFT, timing=TickTiming.PRE_TICK)
        )
        if cfg.min_speed_kmh is not None:
            self.register_fail_condition(
                AndCondition(
                    [
                        ElapsedTimeCondition(0.3, label="speed_check_delay"),
                        SpeedCondition(
                            entity_name=EGO_ROLE_NAME,
                            value=cfg.min_speed_kmh / 3.6,
                            rule=ComparisonRule.LESS_THAN,
                            label="ego_min_speed",
                        ),
                    ]
                )
            )
        self.register_fail_condition(TimeoutCondition(cfg.timeout_seconds, label="timeout"))
        logger.info("goal lanelets: %s", cfg.goal_lanelet_ids)
"""

_counter = 0


def _write_case(
    tmp_path: Path,
    setup: str,
    extra: str = "",
    config: str = _CONFIG,
    header: str = _HEADER,
) -> tuple[type, type, Path]:
    """A scenario package in *tmp_path* whose setup() body is *setup*."""
    global _counter
    _counter += 1
    package = f"acs_typecheck_case_{_counter}"
    root = tmp_path / package
    root.mkdir()
    (root / "__init__.py").write_text("")
    (root / "configs.py").write_text(config)
    scenario = root / "scenario.py"
    scenario.write_text(
        header + textwrap.indent(textwrap.dedent(setup), " " * 8) + extra
    )
    sys.path.insert(0, str(tmp_path))
    try:
        module = importlib.import_module(f"{package}.scenario")
        configs = importlib.import_module(f"{package}.configs")
    finally:
        sys.path.remove(str(tmp_path))
    return module.CaseScenario, configs.CaseConfig, scenario


def _line_of(path: Path, needle: str) -> int:
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if needle in line:
            return number
    raise AssertionError(f"{needle!r} not in {path}")


def _only_error(result: TypeCheckResult) -> Any:
    assert not result.ok, result.format()
    assert result.diagnostics, result.output
    return result.diagnostics[0]


# ---------------------------------------------------------------------------
# What passes
# ---------------------------------------------------------------------------


def _scenario_configs() -> list[str]:
    from autoware_carla_scenario.examples import run

    return run._resolve_scenario_glob("**/*")


@pytest.mark.parametrize("config_name", _scenario_configs())
def test_every_built_in_scenario_config_compiles(config_name: str) -> None:
    from autoware_carla_scenario.examples import run
    from autoware_carla_scenario.registry import get_scenario_classes

    cfg = run._compose_config(config_name, [])
    classes = get_scenario_classes(str(cfg.scenario.name))
    assert classes is not None
    result = typecheck_scenario(*classes, run._to_dict(cfg.scenario))
    assert result.ok, result.format()
    # Every function is called by the check: none is left out (#192).
    assert not result.warnings, result.format()


def test_the_scenario_package_template_compiles() -> None:
    sys.path.insert(0, str(_TEMPLATE_SRC))
    try:
        from my_scenario_package.configs import ReachGoalConfig
        from my_scenario_package.reach_goal import ReachGoalScenario
    finally:
        sys.path.remove(str(_TEMPLATE_SRC))
    conf = _TEMPLATE_SRC / "my_scenario_package" / "conf" / "scenario" / "reach_goal"
    values = yaml.safe_load((conf / "default.yaml").read_text())["scenario"]
    result = typecheck_scenario(ReachGoalScenario, ReachGoalConfig, values)
    assert result.ok, result.format()
    # Every function is called by the check: none is left out (#192).
    assert not result.warnings, result.format()


def test_a_scaffolded_scenario_package_compiles(tmp_path: Path) -> None:
    from autoware_carla_scenario.scaffold import create_scenario_package

    root = create_scenario_package("typecheck_scaffold_pkg", output_dir=tmp_path).root
    src = root / "src"
    sys.path.insert(0, str(src))
    try:
        configs = importlib.import_module("typecheck_scaffold_pkg.configs")
        scenario = importlib.import_module("typecheck_scaffold_pkg.typecheck_scaffold")
    finally:
        sys.path.remove(str(src))
    conf = src / "typecheck_scaffold_pkg" / "conf" / "scenario" / "typecheck_scaffold"
    values = yaml.safe_load((conf / "default.yaml").read_text())["scenario"]
    config_cls = next(v for k, v in vars(configs).items() if k.endswith("Config"))
    scenario_cls = next(
        v
        for k, v in vars(scenario).items()
        if k.endswith("Scenario") and k != "BaseScenario"
    )
    result = typecheck_scenario(scenario_cls, config_cls, values)
    assert result.ok, result.format()
    # Every function is called by the check: none is left out (#192).
    assert not result.warnings, result.format()


def test_a_correct_scenario_compiles(tmp_path: Path) -> None:
    scenario, config, _ = _write_case(tmp_path, _VALID_SETUP)
    result = typecheck_scenario(scenario, config, {"min_speed_kmh": 5})
    assert result.ok, result.format()


_COVER_SETUP = """
self.register_cover(
    "ego_speed",
    lambda world: 3.0,
    unit="km/h",
    range=(0.0, 60.0),
    every=10.0,
    event=SamplingEvent.TICK,
)
self.register_cover(
    "gap",
    ego_gap,
    buckets=[0.0, 5.0, 10.0, 30.0],
    ignore=lambda v: v < 0.0,
    event=SamplingEvent.TICK,
    cover_by="meters",
    target=50,
    min_stay=1.0,
)
self.register_cover(
    "turn",
    lambda world: TurnDirection.LEFT,
    values=[TurnDirection.LEFT, TurnDirection.RIGHT],
    event=ElapsedTimeCondition(1.0, label="sample_at_1s"),
    text="the turn taken",
    target=2,
)
self.register_cover("braking", lambda world: False, values=[False, True])
self.register_cross(
    "speed_x_gap", ["ego_speed", "gap"], text="speed and gap", cover_by="seconds", target=2.5
)
"""

_COVER_EXTRA = """


def ego_gap(world: carla.World) -> float | None:
    return 4.0
"""


def test_turn_indicators_compile(tmp_path: Path) -> None:
    """An NPC signals right once the ego signals left."""
    setup = """
        self.register_pre_tick(
            TurnIndicatorAction(
                "npc1",
                TurnIndicatorState.RIGHT,
                TurnIndicatorCondition(
                    EGO_ROLE_NAME, TurnIndicatorState.LEFT, label="ego_signals_left"
                ),
                label="npc_signals_right",
            )
        )
"""
    scenario, config, _ = _write_case(tmp_path, setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_cover_items_compile(tmp_path: Path) -> None:
    scenario, config, _ = _write_case(tmp_path, _COVER_SETUP, _COVER_EXTRA)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_a_cover_event_that_is_neither_an_event_nor_a_condition_is_refused(
    tmp_path: Path,
) -> None:
    setup = 'self.register_cover("x", ego_gap, values=[1], event="end")\n'
    scenario, config, path = _write_case(tmp_path, setup, _COVER_EXTRA)
    error = _only_error(typecheck_scenario(scenario, config))
    assert "event must be a SamplingEvent or a condition" in error.message
    assert error.line == _line_of(path, 'event="end"')


_ODD_MODULE = """
from __future__ import annotations

import typesafe_carla.carla as carla

from autoware_carla_scenario import (
    OddAttribute,
    OddDefinition,
    OddModule,
    any_of,
    module_holds,
    register_odd,
)
from autoware_carla_scenario.odd import INTENSITY_LEVELS, lanelet_location, rain, speed_limit_kph


def yaw_rate(world: carla.World) -> float | None:
    return 0.0


def build() -> OddDefinition:
    location = OddAttribute(
        "scenery.location",
        lanelet_location,
        values=["urban", "nonurban"],
        cover_by="meters",
        target=200,
        min_stay=2.0,
    )
    speed_limit = OddAttribute(
        "scenery.speed_limit", speed_limit_kph, unit="km/h", buckets=[0, 30, 60, 100]
    )
    weather = OddAttribute("environment.rain", rain, values=INTENSITY_LEVELS)
    yaw = OddAttribute("dynamic.yaw_rate", yaw_rate, range=(-30.0, 30.0), every=10.0)
    return OddDefinition(
        "urban",
        [location, speed_limit, weather, yaw],
        [
            OddModule("roads", include_and=[location.is_in(["urban"]), speed_limit.between(0, 60)]),
            OddModule("weather", exclude_or=[weather.is_in(["heavy"])], labels=["fair"]),
            OddModule("steady", include_or=[yaw.at_most(20.0), any_of((yaw.less_than(25.0),))]),
            OddModule(
                "root",
                include_and=[module_holds("roads"), module_holds("fair"), module_holds("steady")],
                exclude_or=[yaw.is_unknown()],
                labels=[],
            ),
            OddModule("empty", include_and=[], active=False),
            OddModule(
                "fast_urban",
                include_and=[location.is_in(["urban"]), speed_limit.at_least(50)],
                situation=True,
                cover_by="meters",
                target=100,
                min_stay=2.0,
            ),
        ],
        roots=("root",),
    )


register_odd("urban_case", build)
"""


def _write_odd(tmp_path: Path, source: str) -> Any:
    global _counter
    _counter += 1
    package = f"acs_typecheck_odd_{_counter}"
    root = tmp_path / package
    root.mkdir()
    (root / "__init__.py").write_text("")
    (root / "odds.py").write_text(source)
    sys.path.insert(0, str(tmp_path))
    try:
        return importlib.import_module(f"{package}.odds"), root / "odds.py"
    finally:
        sys.path.remove(str(tmp_path))


def test_an_odd_written_in_python_compiles(tmp_path: Path) -> None:
    from autoware_carla_scenario.typecheck import typecheck_odd

    module, _ = _write_odd(tmp_path, _ODD_MODULE)
    result = typecheck_odd(module.build)
    assert result.ok, result.format()


def test_a_module_with_two_include_sections_is_refused(tmp_path: Path) -> None:
    from autoware_carla_scenario.typecheck import typecheck_odd

    source = _ODD_MODULE.replace(
        'OddModule("steady", include_or=',
        'OddModule("steady", include_and=[yaw.at_most(1.0)], include_or=',
    )
    module, _ = _write_odd(tmp_path, source)
    error = _only_error(typecheck_odd(module.build))
    assert "one include section" in error.message, error.format()


def test_an_odd_condition_of_the_wrong_type_is_refused_at_its_line(
    tmp_path: Path,
) -> None:
    from autoware_carla_scenario.typecheck import typecheck_odd

    source = _ODD_MODULE.replace(
        "speed_limit.between(0, 60)", 'speed_limit.between("0", 60)'
    )
    module, path = _write_odd(tmp_path, source)
    error = _only_error(typecheck_odd(module.build))
    assert "expected a float" in error.message, error.format()
    assert error.path == str(path)
    assert error.line == _line_of(path, 'between("0"')


def test_a_custom_condition_is_checked_through_its_check_method(tmp_path: Path) -> None:
    custom = """

class AfterCondition(BaseCondition):
    _after: float

    def __init__(self, after: float) -> None:
        super().__init__("after")
        self._after = after

    def check(self, world: carla.World, elapsed: float) -> ScenarioResult | None:
        if elapsed < self._after:
            return None
        return ScenarioResult(passed=True, message="done", elapsed_seconds=elapsed)
"""
    setup = "self.register_pass_condition(StickyCondition(AfterCondition(3.0)))\n"
    scenario, config, _ = _write_case(tmp_path, setup, custom)
    assert typecheck_scenario(scenario, config).ok

    broken = custom.replace('message="done"', "message=elapsed")
    scenario, config, path = _write_case(tmp_path, setup, broken)
    error = _only_error(typecheck_scenario(scenario, config))
    assert error.path == str(path)
    assert error.line == _line_of(path, "message=elapsed")


def test_the_carla_api_is_typesafe_carla(tmp_path: Path) -> None:
    # Calls the framework's own model never declared: `carla` is the whole of
    # typesafe_carla's typed API.
    setup = """
        library = self.world.get_blueprint_library()
        blueprint = library.find("vehicle.tesla.model3")
        blueprint.set_attribute("role_name", "npc")
        waypoint = self.world.get_map().get_waypoint(carla.Location(x=1.0, y=2.0))
        if waypoint is not None:
            npc = self.world.try_spawn_actor(blueprint, waypoint.transform)
            if npc is not None:
                npc.as_vehicle().apply_control(carla.VehicleControl(throttle=0.5))
        self.world.set_weather(carla.WeatherParameters.ClearNoon)
        """
    scenario, config, _ = _write_case(tmp_path, setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_a_scenario_following_a_trajectory_compiles(tmp_path: Path) -> None:
    """Vertices in every frame, a timing, both modes and the T4 replay."""
    setup = """
        self._setup_ego_spawn()
        path = Trajectory(
            "npc_path",
            [
                TrajectoryVertex(MapPose(100.0, 200.0, yaw=0.5), TrajectoryTimeCondition(0.0)),
                TrajectoryVertex(Lanelet2Pose(10, 5.0), TrajectoryTimeCondition(2.0)),
                TrajectoryVertex(
                    snap_to_carla_road(Lanelet2Pose(10, 9.0), self.world), TrajectoryTimeCondition(4.0)
                ),
                TrajectoryVertex(RelativeLanePose(30.0, d_lane=1), TrajectoryTimeCondition(6.0)),
                TrajectoryVertex(
                    RelativeLanePose(ds=-5, offset=0.5, d_lane=-1, yaw=0.1, entity_ref=EGO_ROLE_NAME),
                    TrajectoryTimeCondition(8.0),
                ),
            ],
        )
        self.register_pre_tick(
            FollowTrajectoryAction(
                "npc1",
                path,
                TrajectoryTiming(ReferenceContext.ABSOLUTE, scale=2.0),
                TrajectoryFollowingMode.FOLLOW,
                condition=ElapsedTimeCondition(1.0, label="go"),
                label="npc1_follow",
            )
        )
        self.register_pre_tick(
            FollowTrajectoryAction(
                "npc2", path, hidden_outside_trajectory=True, label="npc2_follow"
            )
        )
        """
    scenario, config, _ = _write_case(tmp_path, setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_a_trajectory_with_waypoint_conditions_compiles(tmp_path: Path) -> None:
    """Each vertex departs on a time, any other condition, or nothing."""
    setup = """
        self._setup_ego_spawn()
        path = Trajectory(
            "gated",
            [
                TrajectoryVertex(MapPose(100.0, 200.0), TrajectoryTimeCondition(0.0)),
                TrajectoryVertex(
                    MapPose(110.0, 200.0), advance=ElapsedTimeCondition(5.0, label="go")
                ),
                TrajectoryVertex(
                    MapPose(120.0, 200.0),
                    advance=AndCondition(
                        [
                            ElapsedTimeCondition(6.0, label="late"),
                            SpeedCondition(EGO_ROLE_NAME, 1.0, ComparisonRule.LESS_THAN, label="slow"),
                        ]
                    ),
                ),
                TrajectoryVertex(MapPose(125.0, 200.0), advance=TrajectoryTimeCondition(8.0)),
                TrajectoryVertex(MapPose(130.0, 200.0), TrajectoryTimeCondition(9.0)),
            ],
        )
        untimed = Trajectory("untimed", [TrajectoryVertex(MapPose(0.0, 0.0)), TrajectoryVertex(MapPose(9.0, 0.0))])
        if path.is_gated and path.has_times:
            untimed = untimed.gated({0: StickyCondition(ElapsedTimeCondition(1.0, label="start"))})
        first = path.vertices[0].time
        if first is not None:
            logger.info("first departs at %f", first)
        gate = path.vertices[1].gate
        if gate is not None:
            logger.info("second departs on %s", gate.label)
        action = FollowTrajectoryAction("npc1", path, TrajectoryTiming(), label="npc1_gated", speed=8.0)
        self.register_pre_tick(action)
        self.register_pre_tick(FollowTrajectoryAction("npc2", untimed, label="npc2_gated"))
        held = action.held_vertex
        if held is not None:
            logger.info("held at %d", held)
        """
    scenario, config, _ = _write_case(tmp_path, setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


# ---------------------------------------------------------------------------
# What is refused, and where it is reported
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("setup", "needle", "message"),
    [
        (
            "self.register_fail_condition(TimeoutCondition(10.0))\n",
            "TimeoutCondition(10.0)",
            "TimeoutCondition(): label is a required keyword argument",
        ),
        (
            'self.register_fail_condition(TimeoutCondition("10", label="t"))\n',
            'TimeoutCondition("10"',
            "no function 'TimeoutCondition.__init__'",
        ),
        (
            "self.register_pre_tick(TurnAction(EGO_ROLE_NAME, LaneChangeDirection.LEFT))\n",
            "LaneChangeDirection.LEFT",
            "no function 'TurnAction.__init__'",
        ),
        (
            "self.register_pass_condition(TurnAction(EGO_ROLE_NAME, TurnDirection.LEFT))\n",
            "register_pass_condition",
            "'TurnAction' does not match expected type 'BaseCondition'",
        ),
        (
            'self.register_fail_condition(SpeedCondition(42, 1.0, ComparisonRule.LESS_THAN, label="s"))\n',
            "SpeedCondition(42",
            "expected an EntityRole or a str",
        ),
        (
            "snap_to_carla_road(Lanelet2Pose(1, 0.0), self.world).to_carla_transfrom()\n",
            "to_carla_transfrom",
            "has no attribute 'to_carla_transfrom'",
        ),
        (
            "self.derive_goal_from_route(cfg.goal_lanelet_id)\n",
            "cfg.goal_lanelet_id)",
            "has no attribute 'goal_lanelet_id'",
        ),
        (
            "self.world.get_spectator().set_transform(carla.Location(x=1.0))\n",
            "set_transform(carla.Location",
            "'Location' does not match expected type 'Transform'",
        ),
        (
            "self.world.get_blueprint_library().find(3)\n",
            "find(3)",
            "'int' does not match expected type 'str'",
        ),
        (
            'Trajectory("t", [TrajectoryVertex("here"), TrajectoryVertex(MapPose(1.0, 2.0))])\n',
            'TrajectoryVertex("here"',
            "expected a trajectory position",
        ),
        (
            "TrajectoryVertex(MapPose(1.0, 2.0), 3.0)\n",
            "TrajectoryVertex(MapPose(1.0, 2.0), 3.0)",
            "'float' does not match expected type 'BaseCondition'",
        ),
        (
            "RelativeLanePose(10.0, entity_ref=3)\n",
            "RelativeLanePose(10.0",
            "expected an EntityRole or a str",
        ),
        (
            'FollowTrajectoryAction("npc1", Trajectory("t", [TrajectoryVertex(MapPose(1.0, 2.0)), '
            "TrajectoryVertex(MapPose(3.0, 4.0))]), TrajectoryFollowingMode.FOLLOW)\n",
            "FollowTrajectoryAction(",
            "TrajectoryTiming",
        ),
        (
            'self.register_pass_condition(TemporaryStopCondition(EGO_ROLE_NAME, ["here"], label="stop"))\n',
            "TemporaryStopCondition(EGO_ROLE_NAME",
            "expected a pose",
        ),
    ],
    ids=[
        "missing-label",
        "str-for-float",
        "wrong-enum",
        "action-as-condition",
        "int-role",
        "method-typo",
        "config-field-typo",
        "carla-wrong-argument",
        "carla-int-for-str",
        "trajectory-str-position",
        "trajectory-bare-time",
        "relative-lane-int-entity-ref",
        "trajectory-mode-for-timing",
        "temporary-stop-str-position",
    ],
)
def test_a_wrong_scenario_is_refused_at_its_line(
    tmp_path: Path, setup: str, needle: str, message: str
) -> None:
    scenario, config, path = _write_case(tmp_path, setup)
    error = _only_error(typecheck_scenario(scenario, config))
    assert message in error.message, error.format()
    assert error.path == str(path)
    assert error.line == _line_of(path, needle)


@pytest.mark.parametrize(
    ("member", "position", "message"),
    [
        ("OpenDrivePose", 'OpenDrivePose("1", -1, 2.0)', None),
        ("MapPose", "MapPose(1.0, 2.0)", "expected a Union of poses"),
    ],
    ids=["poses", "non-pose-member"],
)
def test_temporary_stop_takes_a_union_of_poses_only(
    tmp_path: Path, member: str, position: str, message: str | None
) -> None:
    """An element of a list of a union is a Union: _expect_pose checks each member."""
    header = _HEADER.replace(
        "    _config: CaseConfig\n",
        f"    _config: CaseConfig\n    _stops: list[Lanelet2Pose | {member}]\n",
    )
    setup = f"""
        self._stops = []
        self._stops.append(Lanelet2Pose(1, 0.0))
        self._stops.append({position})
        self.register_pass_condition(
            TemporaryStopCondition(EGO_ROLE_NAME, self._stops, label="stop")
        )
    """
    scenario, config, path = _write_case(tmp_path, setup, header=header)
    result = typecheck_scenario(scenario, config)
    if message is None:
        assert result.ok, result.format()
    else:
        error = _only_error(result)
        assert message in error.message, error.format()
        assert error.line == _line_of(path, "TemporaryStopCondition(EGO_ROLE_NAME")


_DEFAULT_HEADER = _HEADER.replace(
    "    _config: CaseConfig\n",
    '    _config: CaseConfig\n    RETRIES: int = "three"\n',
)


@pytest.mark.parametrize(
    ("extra", "header", "needle", "message", "label"),
    [
        (
            "\n    def _helper(self) -> float:\n"
            "        return self._config.timeout_secs\n",
            _HEADER,
            "self._config.timeout_secs",
            "has no attribute 'timeout_secs'",
            "<scenario check: CaseScenario._helper()>",
        ),
        (
            "\n    def describe(self) -> str:\n        return 3 + self._config.name\n",
            _HEADER,
            "3 + self._config.name",
            "unsupported operand type(s) for +",
            "<scenario check: CaseScenario.describe()>",
        ),
        (
            "\n    def initial_speeds_mps(self, ego_actor: carla.Actor | None) -> dict[int, float]:\n"
            '        return {1: "fast"}\n',
            _HEADER,
            'return {1: "fast"}',
            "does not match expected type",
            "<scenario check: CaseScenario.initial_speeds_mps(",
        ),
        (
            "\n\ndef unused_helper(x: float) -> float:\n    return x.missing()\n",
            _HEADER,
            "x.missing()",
            "has no attribute 'missing'",
            "<scenario check: unused_helper(x: float)>",
        ),
        (
            "\n    @property\n    def limit(self) -> float:\n"
            "        return self._config.timeout_seconds\n\n"
            "    @limit.setter\n    def limit(self, seconds: float) -> None:\n"
            "        self._config.timeout_seconds = str(seconds)\n",
            _HEADER,
            "self._config.timeout_seconds = str(seconds)",
            "does not match expected type",
            "<scenario check: CaseScenario.limit (setter)>",
        ),
        (
            "",
            _DEFAULT_HEADER,
            'RETRIES: int = "three"',
            "does not match expected type",
            "reached by <scenario check: class-level default of CaseScenario.RETRIES>",
        ),
    ],
    ids=[
        "private-helper",
        "public-method",
        "hook-override",
        "module-function",
        "setter",
        "class-level-default",
    ],
)
def test_code_setup_does_not_reach_is_checked_at_its_line(
    tmp_path: Path, extra: str, header: str, needle: str, message: str, label: str
) -> None:
    # Every function of the scenario's own modules is compiled (#192), and
    # every property setter (#200), with the call that reached it in the trace.
    scenario, config, path = _write_case(tmp_path, _VALID_SETUP, extra, header=header)
    error = _only_error(typecheck_scenario(scenario, config))
    assert message in error.message, error.format()
    assert error.path == str(path)
    assert error.line == _line_of(path, needle)
    assert any(label in frame for frame in error.trace), error.format()


_PARENT_HEADER = _HEADER.replace(
    "class CaseScenario(BaseScenario):", "class CaseParent(BaseScenario):"
)

_INHERITING_SCENARIO = """


class CaseScenario(CaseParent):
    def is_done(self) -> bool:
        return True
"""


def test_a_subclass_inheriting_its_bases_init_compiles(tmp_path: Path) -> None:
    # Codon 0.19 inherits no __init__: the check gives the subclass one passing
    # everything on to CaseParent's, as Python does (#201).
    scenario, config, _ = _write_case(
        tmp_path, _VALID_SETUP, _INHERITING_SCENARIO, header=_PARENT_HEADER
    )
    result = typecheck_scenario(scenario, config, {"min_speed_kmh": 5})
    assert result.ok, result.format()
    assert not result.warnings, result.format()


def test_a_wrong_argument_to_an_inherited_init_is_refused(tmp_path: Path) -> None:
    extra = _INHERITING_SCENARIO + (
        "\n\ndef build(ego: EgoConfig) -> CaseParent:\n"
        '    return CaseScenario(ego, Lanelet2Pose(1, 0.0), config="fast")\n'
    )
    scenario, config, path = _write_case(
        tmp_path, _VALID_SETUP, extra, header=_PARENT_HEADER
    )
    error = _only_error(typecheck_scenario(scenario, config))
    assert "no function 'CaseParent.__init__'" in error.message, error.format()
    # At the class that inherits the __init__, with the call in the trace.
    assert error.path == str(path)
    assert error.line == _line_of(path, "class CaseScenario(CaseParent):")
    call = _line_of(path, 'config="fast"')
    assert any(f"scenario.py:{call}]" in frame for frame in error.trace), error.format()


def test_an_init_inherited_from_a_later_base_compiles(tmp_path: Path) -> None:
    # Python takes CaseParent's __init__, past the mixin, which has none; the
    # check calls it by name, as Codon's super() is the mixin's.
    extra = (
        "\n\n\nclass CaseMixin:\n"
        "    def greeting(self) -> str:\n"
        '        return "hello"\n'
        "\n\nclass CaseScenario(CaseMixin, CaseParent):\n"
        "    def is_done(self) -> bool:\n"
        "        return True\n"
        "\n\ndef build(ego: EgoConfig) -> CaseParent:\n"
        "    return CaseScenario(ego, Lanelet2Pose(1, 0.0), config=CaseConfig())\n"
    )
    scenario, config, _ = _write_case(
        tmp_path, _VALID_SETUP, extra, header=_PARENT_HEADER
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()
    assert not result.warnings, result.format()


_OPTIONAL_HEADER = _HEADER.replace(
    "import logging\n", "import logging\nfrom typing import Optional\n"
)


@pytest.mark.parametrize(
    "extra",
    [
        "\n    def _find(self) -> Optional[carla.Actor]:\n        return None\n",
        "\n    def _where(self) -> carla.Transform | None:\n        return\n",
        "\n    def _found(self) -> Optional[carla.Actor]:\n"
        "        found: Optional[carla.Actor] = None\n"
        "        return found\n",
    ],
    ids=["optional-actor", "optional-transform", "local-variable"],
)
def test_an_optional_function_returning_only_none_compiles(
    tmp_path: Path, extra: str
) -> None:
    # A bare None is Codon's NoneType, and nothing else realizes carla.Actor:
    # the check returns Optional[X]() and realizes X before the call (#202).
    scenario, config, _ = _write_case(
        tmp_path, _VALID_SETUP, extra, header=_OPTIONAL_HEADER
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()
    assert not result.warnings, result.format()


def test_an_error_in_an_optional_function_returning_none_is_reported(
    tmp_path: Path,
) -> None:
    extra = (
        "\n    def _find(self) -> Optional[carla.Actor]:\n"
        "        self._config.timeout_seconds = self._config.name\n"
        "        return None\n"
    )
    scenario, config, path = _write_case(
        tmp_path, _VALID_SETUP, extra, header=_OPTIONAL_HEADER
    )
    error = _only_error(typecheck_scenario(scenario, config))
    assert "does not match expected type" in error.message, error.format()
    assert error.path == str(path)
    assert error.line == _line_of(path, "timeout_seconds = self._config.name")
    assert any("<scenario check: CaseScenario._find()>" in f for f in error.trace)


def test_an_unreached_config_method_is_checked_at_its_line(tmp_path: Path) -> None:
    # The scenario's other modules get the appended calls too (#192).
    config = (
        _CONFIG + "\n    def describe(self) -> str:\n        return 3 + self.name\n"
    )
    scenario, config_cls, _ = _write_case(tmp_path, _VALID_SETUP, config=config)
    path = Path(sys.modules[config_cls.__module__].__file__ or "")
    error = _only_error(typecheck_scenario(scenario, config_cls))
    assert "unsupported operand type(s) for +" in error.message, error.format()
    assert error.path == str(path)
    assert error.line == _line_of(path, "return 3 + self.name")
    assert any("<scenario check: CaseConfig.describe()>" in f for f in error.trace)


def test_a_helper_the_check_cannot_call_is_left_to_what_reaches_it(
    tmp_path: Path,
) -> None:
    # A parameter without an annotation gives the appended calls nothing to
    # call with: the helper is skipped with a warning, not refused, and is
    # compiled where setup() reaches it, as before.
    extra = "\n    def _scaled(self, x):\n        return x * 2.0\n"
    scenario, config, path = _write_case(tmp_path, _VALID_SETUP, extra)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()
    assert not result.diagnostics
    [warning] = result.warnings
    assert warning.severity == "warning"
    assert "CaseScenario._scaled: parameter `x` has no annotation" in warning.message
    assert "compiled only where" in warning.message
    assert warning.path == str(path)
    assert warning.line == _line_of(path, "def _scaled(self, x)")
    assert f"{path}:{warning.line}: warning: " in result.format()

    # Reached from setup(), it is compiled for what setup() passes.
    called = _VALID_SETUP + '        self._scaled("fast")\n'
    scenario, config, path = _write_case(tmp_path, called, extra)
    result = typecheck_scenario(scenario, config)
    error = _only_error(result)
    assert error.line == _line_of(path, "return x * 2.0"), error.format()
    assert len(result.warnings) == 1


_SHAPES_HEADER = _HEADER.replace(
    "import logging\n",
    "import logging\nfrom dataclasses import dataclass\nfrom enum import Enum\n"
    "from typing import Any, Optional, Protocol, TypeVar\n",
)

_SHAPES = """

    _target: carla.Actor | None = None

    def _pose_s(self, pose: Lanelet2Pose | OpenDrivePose) -> float:
        if isinstance(pose, Lanelet2Pose):
            return pose.s
        return pose.s

    @staticmethod
    def _kmh(v: float) -> float:
        return v * 3.6

    @classmethod
    def _twice(cls, n: int) -> list[int]:
        return [n] * 2

    def _goal_sum(self) -> int:
        def add(a: int, b: int) -> int:
            return a + b

        total = 0
        for goal in self._config.goal_lanelet_ids:
            total = add(total, goal)
        return total

    @property
    def timeout(self) -> float:
        return self._config.timeout_seconds

    @timeout.setter
    def timeout(self, seconds: float | int) -> None:
        self._config.timeout_seconds = float(seconds)

    def _target_x(self) -> float:
        if self._target is None:
            return 0.0
        return self._target.get_location().x

    def _describe(self, actor: carla.Actor) -> str:
        if hasattr(actor, "type_id"):
            return actor.type_id
        return "unknown"


class Phase(Enum):
    START = 0
    END = 1


@dataclass
class Waypoint:
    lanelet_id: int
    s: float = 0.0
    tag: Optional[str] = None


class HasSpeed(Protocol):
    def speed(self) -> float: ...


class _Strategy:
    def pick(self, ids: list[int]) -> int:
        raise NotImplementedError


class FirstStrategy(_Strategy):
    def pick(self, ids: list[int]) -> int:
        return ids[0]


T = TypeVar("T")


def first(xs: list[T]) -> T:
    return xs[0]


class NearGoal(BaseCondition):
    _threshold: float

    def __init__(self, threshold: float, label: str = "near") -> None:
        super().__init__(label=label)
        self._threshold = threshold

    def check(self, world: carla.World, elapsed: float) -> ScenarioResult | None:
        if elapsed < self._threshold:
            return None
        return ScenarioResult(passed=True, message="done", elapsed_seconds=elapsed)

    def get_details(self) -> dict[str, Any]:
        return {"threshold": self._threshold, "kind": "near"}
"""


def test_valid_code_setup_does_not_reach_still_compiles(tmp_path: Path) -> None:
    # Compiling every function (#192) must not refuse a valid scenario: a
    # union parameter, static and class methods, a nested function, a
    # property and its setter (#200), an Optional narrowed by a None check, hasattr, an enum, a
    # dataclass, a Protocol and an abstract stub (`raise NotImplementedError`,
    # which Codon 0.19 crashes on as a class).  A generic function and a
    # get_details() returning `dict[str, Any]` of mixed values (what the
    # library check lists as UNCALLED) are left out, with a warning each.
    scenario, config, path = _write_case(
        tmp_path, _VALID_SETUP, _SHAPES, header=_SHAPES_HEADER
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()
    assert [(w.line, w.severity) for w in result.warnings] == [
        (_line_of(path, "def first("), "warning"),
        (_line_of(path, "def get_details("), "warning"),
    ], result.format()
    assert "annotated with a TypeVar" in result.warnings[0].message
    assert "returns `dict[str, Any]`" in result.warnings[1].message


def test_a_config_value_of_the_wrong_type_is_reported_at_its_key(
    tmp_path: Path,
) -> None:
    scenario, config, _ = _write_case(tmp_path, _VALID_SETUP)
    error = _only_error(
        typecheck_scenario(scenario, config, {"timeout_seconds": "fast"})
    )
    assert error.path == "scenario.timeout_seconds"
    assert "'str' does not match expected type 'float'" in error.message

    error = _only_error(typecheck_scenario(scenario, config, {"timeout_secs": 3}))
    assert error.path == "scenario.timeout_secs"


_NESTED_CONFIG = """
from dataclasses import dataclass, field


@dataclass
class Goal:
    lanelet_id: int = 0


@dataclass
class CaseConfig:
    goal: Goal = field(default_factory=Goal)
"""

_CONVERTING_CONFIG = (
    _NESTED_CONFIG
    + """
    def __post_init__(self) -> None:
        if isinstance(self.goal, dict):
            self.goal = Goal(**self.goal)
"""
)


def test_a_nested_mapping_compiles_only_where_the_config_converts_it(
    tmp_path: Path,
) -> None:
    # The runner passes the YAML mapping as a dict: cfg.goal.lanelet_id works
    # at run time only if the config turns it into a Goal.
    setup = "self.derive_goal_from_route([cfg.goal.lanelet_id])\n"
    values = {"goal": {"lanelet_id": 3}}
    scenario, config, _ = _write_case(tmp_path, setup, config=_CONVERTING_CONFIG)
    assert typecheck_scenario(scenario, config, values).ok

    scenario, config, _ = _write_case(tmp_path, setup, config=_NESTED_CONFIG)
    error = _only_error(typecheck_scenario(scenario, config, values))
    assert error.path == "scenario.goal"


def test_an_undeclared_attribute_is_refused_before_compiling(tmp_path: Path) -> None:
    scenario, config, path = _write_case(tmp_path, "self._count = 3\n")
    error = _only_error(typecheck_scenario(scenario, config))
    assert "CaseScenario._count is assigned without a type" in error.message
    assert "`_count: int`" in error.message
    assert error.line == _line_of(path, "self._count = 3")


def test_a_module_without_a_codon_model_is_named(tmp_path: Path) -> None:
    scenario, config, path = _write_case(tmp_path, "import json\njson.dumps({})\n")
    error = _only_error(typecheck_scenario(scenario, config))
    assert "json has no Codon model" in error.message
    assert error.path == str(path)


def test_a_scenario_may_use_the_pathlib_shim_and_nothing_beyond_it(
    tmp_path: Path,
) -> None:
    # The library check's pathlib shim serves the scenario check too: what it
    # declares compiles, the file-system queries included (with CPython's
    # signatures), and any other name of pathlib is refused.
    scenario, config, _ = _write_case(
        tmp_path,
        "from pathlib import Path\n"
        'logger.info("%s %s", str(Path("a") / "b"), Path("a").exists())\n',
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()
    scenario, config, path = _write_case(
        tmp_path,
        'from pathlib import Path\nlogger.info("%s", Path("a").chmod(0o644))\n',
    )
    error = _only_error(typecheck_scenario(scenario, config))
    assert "chmod" in error.message
    assert error.path == str(path)


def test_a_scenario_may_use_the_os_and_shutil_shims_and_nothing_beyond_them(
    tmp_path: Path,
) -> None:
    # The os and shutil shims serve the scenario check too, as do Python's
    # exceptions Codon lacks (the prelude imports them): a scenario may read
    # the environment and copy a file, and a name the shims do not declare is
    # refused.
    scenario, config, _ = _write_case(
        tmp_path,
        "import os\n"
        "import shutil\n"
        'logger.info("%s %s", os.environ.get("ACS_X", ""), os.getenv("ACS_X"))\n'
        "try:\n"
        '    shutil.copy2("a.xodr", "b.xodr")\n'
        "except FileNotFoundError as exc:\n"
        '    logger.info("%s %s", str(exc), shutil.which("git"))\n',
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()
    scenario, config, path = _write_case(
        tmp_path, 'import shutil\nshutil.copytree("a", "b")\n'
    )
    error = _only_error(typecheck_scenario(scenario, config))
    assert "copytree" in error.message
    assert error.path == str(path)


def test_a_scenario_may_use_the_subprocess_and_metadata_shims_and_nothing_beyond_them(
    tmp_path: Path,
) -> None:
    # The subprocess and importlib.metadata shims serve the scenario check
    # too, and a handler of a tuple of exceptions is split for it: a name the
    # shims do not declare is refused, at its line, past the lines the split
    # handler's copy adds.
    setup = (
        "import subprocess\n"
        "from importlib import metadata\n"
        "try:\n"
        '    done = subprocess.run(["true"], capture_output=True, text=True, timeout=5)\n'
        '    logger.info("%s %s", done.stdout, metadata.version("autoware-carla-scenario"))\n'
        "except (OSError, subprocess.SubprocessError) as exc:\n"
        '    logger.info("%s", str(exc))\n'
    )
    scenario, config, _ = _write_case(tmp_path, setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()
    wrong = 'logger.info("%s", subprocess.check_output(["true"]))'
    scenario, config, path = _write_case(tmp_path, setup + wrong + "\n")
    error = _only_error(typecheck_scenario(scenario, config))
    assert "check_output" in error.message
    assert error.path == str(path)
    lines = path.read_text().splitlines()
    assert error.line == next(i for i, t in enumerate(lines, 1) if wrong in t)


def test_a_scenario_may_use_the_signal_and_atexit_shims_and_nothing_beyond_them(
    tmp_path: Path,
) -> None:
    # The signal and atexit shims serve the scenario check too: a scenario may
    # register a function to run at exit, install a handler and signal a
    # process, and a name the shims do not declare is refused.
    setup = (
        "import atexit\n"
        "import os\n"
        "import signal\n"
        "atexit.register(os.getpid)\n"
        "signal.signal(signal.SIGUSR1, lambda signum, frame: None)\n"
        "signal.signal(signal.SIGUSR2, signal.SIG_IGN)\n"
        'logger.info("%d", signal.SIGTERM)\n'
    )
    scenario, config, _ = _write_case(tmp_path, setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()
    wrong = 'logger.info("%s", str(signal.getsignal(signal.SIGUSR1)))'
    scenario, config, path = _write_case(tmp_path, setup + wrong + "\n")
    error = _only_error(typecheck_scenario(scenario, config))
    assert "getsignal" in error.message
    assert error.path == str(path)
    assert error.line == _line_of(path, wrong)


def test_a_scenario_may_use_the_carla_driver_interface_model_and_nothing_beyond_it(
    tmp_path: Path,
) -> None:
    # codon/carla_driver_interface/ is a top-level model directory, like a
    # shim: the scenario check lets the import through, and Codon compiles it
    # against what the model declares (Pose and Trajectory of geometry).
    scenario, config, _ = _write_case(
        tmp_path,
        "from carla_driver_interface.geometry import Pose\n"
        'logger.info("%f", Pose.from_xyz_yaw(1.0, 2.0, 0.0, 0.5).yaw)\n',
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()
    # The protobuf conversions are not modelled.
    scenario, config, path = _write_case(
        tmp_path,
        "from carla_driver_interface.geometry import Pose\n"
        'logger.info("%s", str(Pose.identity().to_proto()))\n',
    )
    error = _only_error(typecheck_scenario(scenario, config))
    assert "to_proto" in error.message
    assert error.path == str(path)


@pytest.mark.parametrize(
    ("setup", "needle", "message"),
    [
        (
            "from autoware_carla_scenario.autoware_bridge.grpc_server import "
            "GrpcAutowareBridgeServer\nGrpcAutowareBridgeServer()\n",
            "GrpcAutowareBridgeServer()",
            "GrpcAutowareBridgeServer is not modelled",
        ),
        (
            "from autoware_carla_scenario.entity import CarlaDriverEntity\n"
            "CarlaDriverEntity()\n",
            "CarlaDriverEntity()",
            "CarlaDriverEntity is not modelled",
        ),
        (
            "from autoware_carla_scenario.driver.egodriver_client import "
            "EgoDriverGrpcClient\nEgoDriverGrpcClient(None)\n",
            "EgoDriverGrpcClient(None)",
            "EgoDriverGrpcClient is not modelled",
        ),
        (
            "from autoware_carla_scenario.autoware_stack.launcher import "
            "AutowareLauncher\nAutowareLauncher()\n",
            "AutowareLauncher()",
            "AutowareLauncher is a Protocol",
        ),
    ],
    ids=["grpc-bridge-server", "carla-driver-entity", "egodriver-client", "launcher"],
)
def test_a_class_the_model_declares_only_for_a_re_export_cannot_be_built(
    tmp_path: Path, setup: str, needle: str, message: str
) -> None:
    # The checked packages re-export these, so the model declares them; their
    # configs are not modelled (or, for a Protocol, Python refuses to build
    # one), so a scenario that builds one is refused rather than passed.
    scenario, config, path = _write_case(tmp_path, setup)
    error = _only_error(typecheck_scenario(scenario, config))
    assert message in error.message, error.format()
    assert error.path == str(path)
    assert error.line == _line_of(path, needle)


_METRICS_SETUP = """
from autoware_carla_scenario.metrics import (
    BUILT_IN_METRICS,
    OFFROAD,
    Metric,
    is_metric,
    read_metric,
)
offroad: Metric = BUILT_IN_METRICS[OFFROAD]
for key, metric in BUILT_IN_METRICS.items():
    logger.info("%s %s [%s] %s", key, metric.title, metric.unit, metric.aggregation)
if is_metric(offroad.key):
    logger.info("%s", str(read_metric(OFFROAD, self.world)))
    logger.info("%s", str(offroad.read(self.world)))
"""


def test_a_scenario_may_use_the_metrics_model(tmp_path: Path) -> None:
    # Every name of metrics.__all__ is in the model (#203).
    scenario, config, _ = _write_case(tmp_path, _METRICS_SETUP)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


@pytest.mark.parametrize(
    ("wrong", "message"),
    [
        # Metric is a frozen dataclass, a @tuple in the model.
        ('offroad.unit = "km"', "cannot modify tuple attributes"),
        # A Measure has a version; a Metric does not.
        ('logger.info("%d", offroad.version)', "version"),
    ],
    ids=["frozen-field", "measure-attribute"],
)
def test_a_misused_metric_is_refused_at_its_line(
    tmp_path: Path, wrong: str, message: str
) -> None:
    scenario, config, path = _write_case(tmp_path, _METRICS_SETUP + wrong + "\n")
    error = _only_error(typecheck_scenario(scenario, config))
    assert message in error.message, error.format()
    assert error.path == str(path)
    assert error.line == _line_of(path, wrong)


_OWN_METRIC_READERS = """


def read_optional(world: carla.World) -> float | None:
    return 1.0


def read_float(world: carla.World) -> float:
    return 1.0


def read_text(world: carla.World) -> str:
    return "far"
"""


@pytest.mark.parametrize(
    "build",
    [
        'Metric("mine", read_optional)',
        'Metric(key="mine", read=read_float, unit="m", aggregation="min")',
        'Metric("mine", lambda world: 1.0, "Mine")',
    ],
    ids=["optional", "float-keywords", "lambda"],
)
def test_a_scenario_may_build_its_own_metric(tmp_path: Path, build: str) -> None:
    # Python's Metric takes any callable of a world that returns a float; a
    # Codon Callable field takes one of its exact type only, and crashed the
    # compiler on a function returning `float` or on a lambda.
    setup = (
        "from autoware_carla_scenario.metrics import Metric\n"
        f"mine: Metric = {build}\n"
        'logger.info("%s %s", mine.key, str(mine.read(self.world)))\n'
    )
    scenario, config, _ = _write_case(tmp_path, setup, extra=_OWN_METRIC_READERS)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_a_metric_read_that_returns_no_number_is_refused(tmp_path: Path) -> None:
    wrong = 'mine = Metric("mine", read_text)'
    setup = "from autoware_carla_scenario.metrics import Metric\n" + wrong + "\n"
    scenario, config, path = _write_case(tmp_path, setup, extra=_OWN_METRIC_READERS)
    result = typecheck_scenario(scenario, config)
    assert not result.ok, result.format()
    assert any(
        d.path == str(path) and d.line == _line_of(path, wrong)
        for d in result.diagnostics
    ), result.format()


def test_a_scenario_may_read_the_pose_to_map_frame_returns(tmp_path: Path) -> None:
    scenario, config, _ = _write_case(
        tmp_path,
        "from autoware_carla_scenario.coordinate import to_map_frame\n"
        "pose = to_map_frame(Lanelet2Pose(1, 0.0)).moved_forward(1.0)\n"
        'logger.info("%f %f", pose.position.x, pose.rotation.w)\n',
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


# ---------------------------------------------------------------------------
# The runner's step
# ---------------------------------------------------------------------------


def test_build_scenario_refuses_a_wrong_scenario_before_building_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from omegaconf import OmegaConf

    from autoware_carla_scenario.examples import run
    from autoware_carla_scenario.registry import register_scenario, unregister_scenario

    scenario, config, _ = _write_case(
        tmp_path, "self.register_fail_condition(TimeoutCondition(1.0))\n"
    )
    register_scenario("typecheck_case", scenario, config)

    def must_not_run(_cfg: Any) -> Any:
        raise AssertionError("the scenario was built before it was checked")

    monkeypatch.setattr(run, "build_ego_and_spawn", must_not_run)
    try:
        with pytest.raises(ScenarioTypeError, match=re.escape("label is a required")):
            run.build_scenario(
                OmegaConf.create({"scenario": {"name": "typecheck_case"}})
            )
        # Switched off, the runner goes on to build it.
        with pytest.raises(AssertionError, match="built before it was checked"):
            run.build_scenario(
                OmegaConf.create(
                    {"typecheck": "off", "scenario": {"name": "typecheck_case"}}
                )
            )
    finally:
        unregister_scenario("typecheck_case")


def test_a_batch_refuses_a_wrong_value_before_reading_the_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from autoware_carla_scenario.examples import run

    def must_not_run(_cfg: Any) -> Any:
        raise AssertionError("the map was resolved before the scenarios were checked")

    monkeypatch.setattr(run, "resolve_map_paths", must_not_run)
    with pytest.raises(ScenarioTypeError, match=re.escape("scenario.timeout_seconds")):
        run.run_batch(
            ["lane_change/left", "lane_change/right"],
            ["scenario.timeout_seconds=fast"],
        )


def test_scenario_check_fails_an_unknown_scenario_name(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from autoware_carla_scenario.typecheck.cli import main

    assert main(["scenario=lane_change/left", "scenario.name=no_such_scenario"]) == 1
    assert "unknown scenario name 'no_such_scenario'" in capsys.readouterr().out
