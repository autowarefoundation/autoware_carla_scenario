"""The static check with Codon: correct scenarios compile, wrong ones are refused.

Skipped where no Codon compiler is installed (typesafe-carla's toolchain, a
dependency on Linux x86_64 and aarch64).
"""

from __future__ import annotations

import ast
import enum
import importlib
import os
import re
import subprocess
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
    modules: dict[str, str] | None = None,
) -> tuple[type, type, Path]:
    """A scenario package in *tmp_path* whose setup() body is *setup*.

    *modules* are more of its files, by their path in the package
    (``helpers.py``, ``sub/helpers.py``); ``{package}`` in them and in
    *header* is the package's name.
    """
    global _counter
    _counter += 1
    package = f"acs_typecheck_case_{_counter}"
    root = tmp_path / package
    root.mkdir()
    (root / "__init__.py").write_text("")
    (root / "configs.py").write_text(config)
    for relative, text in (modules or {}).items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text.replace("{package}", package))
    header = header.replace("{package}", package)
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


def test_a_traffic_parameters_action_compiles(tmp_path: Path) -> None:
    """The run's traffic, and one vehicle, draw new parameters."""
    setup = """
        self.register_post_tick(
            TrafficParametersAction(
                VehicleParameters(speed_factor=Spread(0.8, 0.05), lane_changes=False),
                condition=ElapsedTimeCondition(20.0, label="after_20s"),
            )
        )
        self.register_post_tick(
            TrafficParametersAction(
                vehicles=VehicleParameters(ignore_lights=Spread(1.0)),
                entity_name="npc1",
                label="reckless",
            )
        )
        self.register_post_tick(
            TrafficParametersAction(
                pedestrians=PedestrianParameters(walking_speed_mps=Spread(1.8, 0.2))
            )
        )
"""
    header = _HEADER.replace(
        "import logging\n",
        "import logging\n"
        "from autoware_carla_scenario import (PedestrianParameters, Spread,\n"
        "    TrafficParametersAction, VehicleParameters)\n",
        1,
    )
    scenario, config, _ = _write_case(tmp_path, setup, header=header)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_a_traffic_parameter_takes_a_value_or_a_spread(tmp_path: Path) -> None:
    """A plain number is a spread of no width."""
    setup = """
        self.register_post_tick(
            TrafficParametersAction(
                VehicleParameters(speed_factor=0.8, min_gap_m=3, decel=Spread(4.0, 0.5))
            )
        )
"""
    header = _HEADER.replace(
        "import logging\n",
        "import logging\n"
        "from autoware_carla_scenario import (Spread, TrafficParametersAction,\n"
        "    VehicleParameters)\n",
        1,
    )
    scenario, config, _ = _write_case(tmp_path, setup, header=header)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_a_traffic_parameter_refuses_what_is_neither(tmp_path: Path) -> None:
    setup = """
        self.register_post_tick(
            TrafficParametersAction(VehicleParameters(speed_factor="fast"))
        )
"""
    header = _HEADER.replace(
        "import logging\n",
        "import logging\n"
        "from autoware_carla_scenario import TrafficParametersAction, VehicleParameters\n",
        1,
    )
    scenario, config, _ = _write_case(tmp_path, setup, header=header)
    assert not typecheck_scenario(scenario, config).ok


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


_COVER_ARGS_HEADER = _HEADER.replace(
    "import logging\n",
    "import logging\nfrom enum import Enum\n"
    "from autoware_carla_scenario.odd import OddAttribute, ego_speed_kph\n",
    1,
)

_COVER_ARGS_EXTRA = """

    def _gap(self, world: carla.World) -> float | None:
        return None

    def _is_negative(self, value: float) -> bool:
        return value < 0.0


class Phase(Enum):
    APPROACH = "approach"
    STOP = "stop"
"""


def test_cover_values_and_ignore_of_the_right_kind_compile(tmp_path: Path) -> None:
    # values= takes any iterable or an enum class, and ignore= a callable of
    # one value of what the expression returns, unwrapped from its Optional
    # (#279).
    setup = """
        self.register_cover(
            "dir", lambda w: "left", values=["left", "right"], ignore=lambda v: v == "none"
        )
        self.register_cover("flag", lambda w: True, values=[False, True])
        self.register_cover("turn", lambda w: TurnDirection.LEFT, values=TurnDirection)
        self.register_cover("phase", lambda w: Phase.STOP, values=Phase)
        self.register_cover("lane", lambda w: 1, values=(1, 2), ignore=self._is_negative)
        self.register_cover("none", lambda w: 1, values=[])
        self.register_cover("gap", self._gap, buckets=[0.0, 5.0], ignore=lambda v: v < 0.0)
        speed = OddAttribute("speed", ego_speed_kph, values=[10.0, 20.0])
        turn = OddAttribute("turn", ego_speed_kph, values=TurnDirection)
        logger.info("%s %s", speed.name, turn.name)
"""
    scenario, config, _ = _write_case(
        tmp_path,
        _TIMEOUT_SETUP + textwrap.dedent(setup),
        _COVER_ARGS_EXTRA,
        header=_COVER_ARGS_HEADER,
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()
    assert not result.warnings, result.format()


@pytest.mark.parametrize(
    ("setup", "message"),
    [
        (
            'self.register_cover("speed", lambda w: 1.0, range=(0.0, 10.0), every=1.0, '
            "ignore=lambda: True)\n",
            "BaseScenario.register_cover(): ignore must take one argument",
        ),
        (
            'self.register_cover("speed", lambda w: 1.0, buckets=[0.0, 5.0, 10.0], '
            'ignore="negative")\n',
            "BaseScenario.register_cover(): ignore must be a callable",
        ),
        (
            'self.register_cover("speed", lambda w: 1.0, buckets=[0.0, 5.0, 10.0], '
            "ignore=lambda v: v.upper())\n",
            "'float' object has no attribute 'upper'",
        ),
        (
            'self.register_cover("lane", lambda w: 1, values=3)\n',
            "BaseScenario.register_cover(): values must be an iterable",
        ),
        (
            'logger.info("%s", OddAttribute("speed", ego_speed_kph, values=3).name)\n',
            "OddAttribute(): values must be an iterable",
        ),
    ],
    ids=[
        "ignore-no-argument",
        "ignore-str",
        "ignore-wrong-value-type",
        "values-int",
        "odd-values-int",
    ],
)
def test_a_wrong_cover_ignore_or_values_is_refused_at_its_line(
    tmp_path: Path, setup: str, message: str
) -> None:
    # openodd's collector counts an ignore that raises as no sample, so a
    # wrong one emptied the item for the whole run; a non-iterable values=
    # raised in setup() (#279).
    scenario, config, path = _write_case(
        tmp_path, _TIMEOUT_SETUP + setup, header=_COVER_ARGS_HEADER
    )
    error = _only_error(typecheck_scenario(scenario, config))
    assert message in error.message, error.format()
    assert error.line == _line_of(path, setup.strip().split("(")[0] + "(")


_ACTOR_HEADER = _HEADER.replace(
    "    _config: CaseConfig\n",
    "    _config: CaseConfig\n    _npc: carla.Vehicle\n    _pedestrian: carla.Walker\n",
    1,
)

_ACTOR_GETTERS = """

    def _npc_or_none(self) -> carla.Vehicle | None:
        return None

    def _pedestrian_or_none(self) -> carla.Walker | None:
        return None

    def _spawned(self, npc: carla.Vehicle, pedestrian: carla.Walker) -> None:
        # Assigned, or Python raises AttributeError at the first read (#588).
        self._npc = npc
        self._pedestrian = pedestrian
"""


def test_an_actor_getter_of_any_actor_subclass_compiles(tmp_path: Path) -> None:
    # Python's Callable[[], Optional[carla.Actor]] takes a getter of a
    # Vehicle, a Walker, an Optional of one or None; Codon 0.19 converts no
    # Optional[Vehicle] to an Optional[Actor] (#278).
    setup = """
        self.follow_with_spectator(lambda: self._npc)
        self.log_actor_position(lambda: self._npc, label="npc")
        self.follow_with_spectator(self._npc_or_none)
        self.log_actor_position(lambda: self._pedestrian, label="pedestrian")
        self.follow_with_spectator(self._pedestrian_or_none)
        self.follow_with_spectator(lambda: self.world.get_actor(1))
        self.log_actor_position(lambda: None)
        logger.info("%s", self.initial_speeds_mps(self._npc))
        logger.info("%s", self.initial_speeds_mps(self._npc_or_none()))
        logger.info("%s", self.initial_speeds_mps(None))
"""
    scenario, config, _ = _write_case(
        tmp_path,
        _TIMEOUT_SETUP + textwrap.dedent(setup),
        _ACTOR_GETTERS,
        header=_ACTOR_HEADER,
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()
    assert not result.warnings, result.format()


@pytest.mark.parametrize(
    ("setup", "message"),
    [
        (
            "self.follow_with_spectator(lambda: self.ego_entity)\n",
            "follow_with_spectator(): the getter must return a carla.Actor",
        ),
        (
            'self.log_actor_position(lambda: 3, label="three")\n',
            "log_actor_position(): the getter must return a carla.Actor",
        ),
        (
            "self.follow_with_spectator(lambda: self.world)\n",
            "follow_with_spectator(): the getter must return a carla.Actor",
        ),
        (
            'logger.info("%s", self.initial_speeds_mps(self.world))\n',
            "initial_speeds_mps(): ego_actor must be a carla.Actor",
        ),
    ],
    ids=["ego-entity", "int", "world", "initial-speeds-world"],
)
def test_an_actor_getter_returning_no_actor_is_refused_at_its_line(
    tmp_path: Path, setup: str, message: str
) -> None:
    # An EgoVehicle is the framework's entity, not a carla.Actor (#278).
    scenario, config, path = _write_case(
        tmp_path, _TIMEOUT_SETUP + setup, header=_ACTOR_HEADER
    )
    error = _only_error(typecheck_scenario(scenario, config))
    assert message in error.message, error.format()
    assert error.line == _line_of(path, setup.strip()[:30])


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


_ODD_READS = """
from __future__ import annotations

from autoware_carla_scenario import OddAttribute, OddDefinition, OddModule, SamplingEvent
from autoware_carla_scenario.coverage import CoverGroup
from autoware_carla_scenario.odd import lanelet_location, speed_limit_kph


def build() -> OddDefinition:
    location = OddAttribute("scenery.location", lanelet_location, values=["urban", "nonurban"])
    speed = OddAttribute(
        "scenery.speed_limit", speed_limit_kph, unit="km/h", buckets=[0, 30, 60, 100]
    )
    urban = location.is_in(["urban"])
    roads = OddModule("roads", include_and=[urban], labels=["fair"])
    roads.as_situation(cover_by="meters", target=100.0)
    names = urban.attributes() + urban.references() + roads.labels + location.derived_from
    text = urban.describe() + location.cover_by + location.source[1]
    if roads.situation and not location.derived and location.min_stay is None:
        for condition in roads.conditions():
            text += condition.describe()
    item = roads.item
    if item is not None:
        text += item.amount + item.cover_by + item.bucket_of("holds") + item.labels[0]
    odd = OddDefinition("urban", [location, speed], [roads], roots=["roads"])
    names += odd.roots + odd.unmeasured()
    names += odd.outside_buckets(location) + odd.outside_buckets("scenery.location")
    for each in odd.cover_items():
        if each.numeric and each.buckets is not None and each.range is None:
            text += str(each.edges[0]) + each.unit + each.text + str(each.target)
        text += str(each.attribute) + str(each.min_stay) + str(each.every)
        if isinstance(each.event, SamplingEvent) and each.group == CoverGroup.ODD:
            text += "sampled on a tick"
    for module in odd.situations():
        text += module.name
    # Instance attributes (#307): what the definition and an attribute hold.
    for attribute in odd.attributes:
        text += attribute.name + str(attribute.target)
        cover = attribute.item
        if cover is not None:
            text += cover.name
    version = odd.version
    revision = odd.revision
    text += odd.structure_digest + str(len(odd.sources)) + str(version) + str(revision)
    return odd
"""


def test_an_odd_reads_what_openodd_s_classes_hold(tmp_path: Path) -> None:
    # openodd's classes are in the member guarantee (#298): what an ODD
    # builder reads of a module, an attribute, a condition, the definition
    # and its cover items compiles, as it runs in Python.
    from autoware_carla_scenario.typecheck import typecheck_odd

    module, _ = _write_odd(tmp_path, _ODD_READS)
    module.build()
    result = typecheck_odd(module.build)
    assert result.ok, result.format()


@pytest.mark.parametrize(
    ("read", "message"),
    [
        (
            "roads.describe()",
            "OddModule.describe is not part of the Codon model, on purpose",
        ),
        (
            'urban.evaluate({"scenery.location": "urban"})',
            "OddCondition.evaluate is not part of the Codon model, on purpose",
        ),
        (
            # An instance attribute (#307), listed with its reason.
            "location.probe",
            "OddAttribute.probe is not part of the Codon model, on purpose",
        ),
    ],
    ids=["module-describe", "condition-evaluate", "attribute-probe"],
)
def test_an_odd_member_the_model_leaves_out_says_why(
    tmp_path: Path, read: str, message: str
) -> None:
    # openodd's Condition is the package's OddCondition: listed under the
    # name the package gives it.
    from autoware_carla_scenario.typecheck import typecheck_odd

    source = _ODD_READS.replace(
        "    roads.as_situation(", f"    print({read})\n    roads.as_situation(", 1
    )
    module, path = _write_odd(tmp_path, source)
    error = _only_error(typecheck_odd(module.build))
    assert "object has no attribute" in error.message, error.format()
    assert message in error.message, error.format()
    assert error.line == _line_of(path, read)


@pytest.mark.parametrize(
    ("old", "new", "needle", "where"),
    [
        (
            'roots=("root",)',
            'roots="root"',
            "return OddDefinition(",
            "OddDefinition(): roots",
        ),
        (
            'labels=["fair"]',
            'labels="fair"',
            'labels="fair"',
            "OddModule(): labels",
        ),
    ],
    ids=["roots", "labels"],
)
def test_a_str_for_a_sequence_of_names_is_refused_in_an_odd(
    tmp_path: Path, old: str, new: str, needle: str, where: str
) -> None:
    # Python's typing takes a str as a Sequence[str], and openodd reads each
    # of its characters as a name (#299): the check refuses it at its line.
    # _ODD_MODULE's own (a tuple, a list, an empty list) compile, as does
    # is_in("urban"), which CPython takes.
    from autoware_carla_scenario.typecheck import typecheck_odd

    assert old in _ODD_MODULE
    module, path = _write_odd(tmp_path, _ODD_MODULE.replace(old, new, 1))
    error = _only_error(typecheck_odd(module.build))
    assert (
        f"{where}: expected a sequence of names (a list or a tuple of str), not a str"
        in error.message
    ), error.format()
    assert error.path == str(path)
    assert error.line == _line_of(path, needle)


def test_a_str_is_values_an_attribute_is_in(tmp_path: Path) -> None:
    # The valid look-alike of #299: OddAttribute.is_in() takes an
    # Iterable[Any], and CPython takes a str's characters as its values.
    from autoware_carla_scenario.typecheck import typecheck_odd

    source = _ODD_MODULE.replace('location.is_in(["urban"])', 'location.is_in("urban")')
    assert source != _ODD_MODULE
    module, _ = _write_odd(tmp_path, source)
    module.build()
    result = typecheck_odd(module.build)
    assert result.ok, result.format()


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
            "TimeoutCondition(): timeout_seconds: expected float, got str",
        ),
        (
            "self.register_pre_tick(TurnAction(EGO_ROLE_NAME, LaneChangeDirection.LEFT))\n",
            "LaneChangeDirection.LEFT",
            "TurnAction(): direction: expected TurnDirection, got LaneChangeDirection",
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
            "initial_speeds_mps() returns dict[int,str], but the function is declared",
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


_HOOK_HEADER = _HEADER.replace(
    "    AndCondition,", "    AndCondition,\n    BaseAction,\n    EgoVehicle,"
)

_TIMEOUT_SETUP = (
    'self.register_fail_condition(TimeoutCondition(cfg.timeout_seconds, label="t"))\n'
)

_ALWAYS_BOOL = """

class Always(BaseCondition):
    def check(self, world: carla.World, elapsed: float) -> bool:
        return True
"""


def _scenario_method(signature: str, body: str) -> str:
    return f"\n    def {signature}:\n        {body}\n"


@pytest.mark.parametrize(
    ("setup", "extra", "header", "needle", "hook"),
    [
        (
            _TIMEOUT_SETUP + 'self.register_pass_condition(Always(label="always"))\n',
            _ALWAYS_BOOL,
            _HOOK_HEADER,
            "def check(self, world: carla.World, elapsed: float) -> bool:",
            "BaseCondition.check() returns Optional[ScenarioResult]",
        ),
        (
            _TIMEOUT_SETUP,
            _ALWAYS_BOOL,  # never registered: the module check reaches it
            _HOOK_HEADER,
            "def check(self, world: carla.World, elapsed: float) -> bool:",
            "BaseCondition.check() returns Optional[ScenarioResult]",
        ),
        (
            _TIMEOUT_SETUP,
            """

class Status(BaseCondition):
    def check(self, world: carla.World, elapsed: float) -> ConditionStatus:
        return ConditionStatus("s", True, "")
""",
            _HOOK_HEADER.replace(
                "from .configs import CaseConfig",
                "from autoware_carla_scenario.conditions import ConditionStatus\n\n"
                "from .configs import CaseConfig",
            ),
            "def check(self, world: carla.World, elapsed: float) -> ConditionStatus:",
            "BaseCondition.check() returns Optional[ScenarioResult]",
        ),
        (
            _TIMEOUT_SETUP,
            """

class LongTimeout(TimeoutCondition):
    def check(self, world: carla.World, elapsed: float) -> str:
        return "late"
""",
            _HOOK_HEADER,
            "def check(self, world: carla.World, elapsed: float) -> str:",
            "BaseCondition.check() returns Optional[ScenarioResult]",
        ),
        (
            _TIMEOUT_SETUP,
            _scenario_method(
                "initial_speeds_mps(self, ego_actor: carla.Actor | None) -> list[float]",
                "return [3.0]",
            ),
            _HOOK_HEADER,
            "def initial_speeds_mps(",
            "BaseScenario.initial_speeds_mps() returns Dict[int, float]",
        ),
        (
            _TIMEOUT_SETUP,
            _scenario_method(
                "initial_speeds_mps(self, ego_actor: carla.Actor | None) -> dict[int, int]",
                "return {1: 3}",
            ),
            _HOOK_HEADER,
            "def initial_speeds_mps(",
            "BaseScenario.initial_speeds_mps() returns Dict[int, float]",
        ),
        (
            _TIMEOUT_SETUP,
            "",
            _HOOK_HEADER.replace(
                "def is_done(self) -> bool:\n        return False",
                'def is_done(self) -> str:\n        return "no"',
            ),
            "def is_done(self)",
            "BaseScenario.is_done() returns bool",
        ),
        (
            _TIMEOUT_SETUP,
            "",
            _HOOK_HEADER.replace(
                "def is_done(self) -> bool:\n        return False",
                "def is_done(self) -> int:\n        return 0",
            ),
            "def is_done(self)",
            "BaseScenario.is_done() returns bool",
        ),
        (
            _TIMEOUT_SETUP,
            "",
            _HOOK_HEADER.replace(
                "def is_done(self) -> bool:\n        return False",
                "def is_done(self) -> bool | None:\n        return False",
            ),
            "def is_done(self)",
            "this override returns an Optional",
        ),
        (
            _TIMEOUT_SETUP + "return 1\n",
            "",
            _HOOK_HEADER.replace("def setup(self) -> None:", "def setup(self) -> int:"),
            "def setup(self)",
            "BaseScenario.setup() returns None",
        ),
        (
            _TIMEOUT_SETUP,
            "\n    @property\n    def ego_requires_goal(self) -> str:\n"
            '        return "yes"\n',
            _HOOK_HEADER,
            "def ego_requires_goal(self)",
            "BaseScenario.ego_requires_goal returns bool",
        ),
        (
            _TIMEOUT_SETUP,
            """

class Honk(BaseAction):
    def __init__(self) -> None:
        super().__init__("honk")

    def execute(self, world: carla.World) -> bool:
        return True
""",
            _HOOK_HEADER,
            "def execute(self, world: carla.World) -> bool:",
            "BaseAction.execute() returns None",
        ),
        (
            _TIMEOUT_SETUP,
            """

class Ego(EgoVehicle):
    def debug_markers(self) -> list[str]:
        return []
""",
            _HOOK_HEADER,
            "def debug_markers(self)",
            "EgoVehicle.debug_markers() returns DebugMarkers",
        ),
    ],
    ids=[
        "check-bool-registered",
        "check-bool-unregistered",
        "check-condition-status",
        "check-str-of-a-built-in-condition",
        "initial-speeds-list",
        "initial-speeds-dict-of-int",
        "is-done-str",
        "is-done-int",
        "is-done-optional",
        "setup-int",
        "property-str",
        "action-execute-bool",
        "ego-debug-markers-list",
    ],
)
def test_a_hook_override_returning_another_type_is_refused_at_its_def(
    tmp_path: Path, setup: str, extra: str, header: str, needle: str, hook: str
) -> None:
    # Codon compiled each override against its own annotation only, and the
    # framework's callers are model stubs, so these passed (#238): the
    # result of each override is now checked strictly against the type the
    # framework reads, with no conversion (`done: bool = "no"` compiles).
    scenario, config, path = _write_case(tmp_path, setup, extra, header=header)
    result = typecheck_scenario(scenario, config)
    error = _only_error(result)
    assert hook in error.message, error.format()
    assert error.path == str(path)
    assert error.line == _line_of(path, needle)


_VALID_HOOKS = """

    def initial_speeds_mps(self, ego_actor: carla.Actor | None) -> dict[int, float]:
        return {} if ego_actor is None else {ego_actor.id: 3.0}

    def require_goal(self) -> Lanelet2Pose:
        return Lanelet2Pose(1, 0.0)

    @property
    def ego_requires_goal(self) -> bool:
        return False

    def measure(self, key: str, world: carla.World) -> float:
        return 1.0

    def check(self) -> str:
        return "not a hook of BaseScenario"

    def create_ego(self) -> EgoVehicle:
        raise NotImplementedError("only raises: Codon types its result None")


class Result(ScenarioResult):
    pass


class NeverCondition(BaseCondition):
    def check(self, world: carla.World, elapsed: float) -> None:
        return None


class AfterCondition(BaseCondition):
    def check(self, world: carla.World, elapsed: float) -> ScenarioResult:
        return ScenarioResult(passed=elapsed > 1.0, message="", elapsed_seconds=elapsed)


class MaybeCondition(BaseCondition):
    def check(self, world: carla.World, elapsed: float) -> ScenarioResult | None:
        if elapsed < 1.0:
            return None
        return ScenarioResult(passed=True, message="", elapsed_seconds=elapsed)


class OwnResultCondition(BaseCondition):
    def check(self, world: carla.World, elapsed: float) -> Result:
        return Result(passed=True, message="", elapsed_seconds=elapsed)


class Honk(BaseAction):
    def __init__(self) -> None:
        super().__init__("honk")

    def execute(self, world: carla.World) -> None:
        pass


class Tracker:
    def is_done(self) -> str:
        return "not a hook: Tracker derives from no framework class"

    def check(self, world: carla.World, elapsed: float) -> bool:
        return True
"""


def test_hook_overrides_returning_the_framework_type_compile(tmp_path: Path) -> None:
    # The look-alikes of the refused cases: what Python's typing takes for
    # the hook's type (a ScenarioResult or None where Optional[ScenarioResult]
    # is read, a subclass of it, a bool), and methods that override no hook.
    header = _HOOK_HEADER.replace(
        "def is_done(self) -> bool:\n        return False",
        "def is_done(self) -> bool:\n        return self._config.timeout_seconds < 0.0",
    )
    setup = (
        _TIMEOUT_SETUP
        + 'self.register_pass_condition(AfterCondition(label="after"))\n'
        + 'self.register_pass_condition(MaybeCondition(label="maybe"))\n'
        + 'self.register_pass_condition(OwnResultCondition(label="own"))\n'
        + 'self.register_fail_condition(NeverCondition(label="never"))\n'
        + "self.register_pre_tick(Honk())\n"
    )
    scenario, config, _ = _write_case(tmp_path, setup, _VALID_HOOKS, header=header)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


_NONE_HEADER = _HEADER.replace(
    "    snap_to_carla_road,", "    find_actor_by_role_name,\n    snap_to_carla_road,"
)


@pytest.mark.parametrize(
    ("setup", "needle", "message"),
    [
        (
            "threshold: float = cfg.min_speed_kmh / 3.6\n"
            'logger.info("threshold %f", threshold)\n',
            "cfg.min_speed_kmh / 3.6",
            'Unsupported operand types for / ("None" and "float")',
        ),
        (
            'logger.info("goal %d", self.goal_pose.lanelet_id)\n',
            "self.goal_pose.lanelet_id",
            'Item "None" of "Lanelet2Pose | None" has no attribute "lanelet_id"',
        ),
        (
            'npc = find_actor_by_role_name(self.world, "npc")\n'
            'logger.info("npc at %f", npc.get_location().x)\n',
            "npc.get_location()",
            'Item "None"',
        ),
        (
            "self.register_fail_condition(\n"
            '    TimeoutCondition(cfg.min_speed_kmh, label="t")\n'
            ")\n",
            "TimeoutCondition(cfg.min_speed_kmh",
            '"float | None"; expected "float"',
        ),
    ],
    ids=["operator", "attribute", "actor-attribute", "argument"],
)
def test_an_optional_used_without_a_none_test_is_refused(
    tmp_path: Path, setup: str, needle: str, message: str
) -> None:
    # Codon unwraps an Optional implicitly, so these compiled and failed in
    # Python on None (#239); mypy's strict Optional reports each.
    scenario, config, path = _write_case(
        tmp_path, _TIMEOUT_SETUP + setup, header=_NONE_HEADER
    )
    result = typecheck_scenario(scenario, config)
    error = _only_error(result)
    assert len(result.diagnostics) == 1, result.format()
    assert message in error.message, error.format()
    assert "test the value for None" in error.message
    assert error.path == str(path)
    assert error.line == _line_of(path, needle)


def test_an_optional_tested_for_none_compiles(tmp_path: Path) -> None:
    # The look-alikes: narrowed by `is not None`, by an assert, by an early
    # return, or given a default with `or`.
    setup = (
        _TIMEOUT_SETUP
        + "if cfg.min_speed_kmh is not None:\n"
        + '    logger.info("threshold %f", cfg.min_speed_kmh / 3.6)\n'
        + "goal = self.goal_pose\n"
        + "assert goal is not None\n"
        + 'logger.info("goal %d", goal.lanelet_id)\n'
        + 'logger.info("speed %f", (cfg.min_speed_kmh or 0.0) / 3.6)\n'
    )
    extra = """

    def _npc_x(self) -> float:
        npc = find_actor_by_role_name(self.world, "npc")
        if npc is None:
            return 0.0
        return npc.get_location().x
"""
    scenario, config, _ = _write_case(tmp_path, setup, extra, header=_NONE_HEADER)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


_CARLA_OPTIONALS = """

    def _lane_yaw(self, location: carla.Location) -> float:
        wp = self.world.get_map().get_waypoint(location, project_to_road=False)
        return wp.transform.rotation.yaw

    def _actor_x(self, actor_id: int) -> float:
        return self.world.get_actors().find(actor_id).get_location().x
"""


def test_an_optional_carla_result_used_without_a_none_test_is_refused(
    tmp_path: Path,
) -> None:
    # typesafe_carla's CPython package has no type information, so mypy read
    # all of CARLA as Any and these passed both halves of the check (#267).
    scenario, config, path = _write_case(tmp_path, _TIMEOUT_SETUP, _CARLA_OPTIONALS)
    result = typecheck_scenario(scenario, config)
    assert not result.ok, result.format()
    assert [(d.line, d.message.split(":")[0]) for d in result.diagnostics] == [
        (
            _line_of(path, "return wp.transform"),
            'Item "None" of "Waypoint | None" has no attribute "transform"',
        ),
        (
            _line_of(path, ".find(actor_id)"),
            'Item "None" of "Actor | None" has no attribute "get_location"',
        ),
    ], result.format()
    assert all("test the value for None" in d.message for d in result.diagnostics)


def test_a_type_error_mentioning_none_is_not_called_optional_misuse(
    tmp_path: Path,
) -> None:
    # A reader returning a str for one that returns a float: Codon's error.
    # mypy's message names None ("str | None"), but the value is never None,
    # so it is not reported again as an Optional to test (#267).
    extra = """

    def _lane_name(self, world: carla.World) -> Optional[str]:
        return "left"
"""
    setup = _TIMEOUT_SETUP + 'self.register_measure("lane", self._lane_name, unit="")\n'
    header = _HEADER.replace(
        "import logging", "import logging\nfrom typing import Optional"
    )
    scenario, config, _ = _write_case(tmp_path, setup, extra, header=header)
    result = typecheck_scenario(scenario, config)
    error = _only_error(result)
    assert "expected a float" in error.message, result.format()
    # The error's own hint may say a measure can be None (#599); no other
    # diagnostic may.
    assert not [
        d for d in result.diagnostics if d is not error and "None" in d.message
    ], result.format()


_VARIADIC_HELPERS = """


def opts(**kw: float) -> float:
    return kw["speed"]


def pair(*xs: float) -> float:
    return xs[0] + xs[1]


def unused(**kw: float) -> float:
    return kw["speed"]


def unused_args(*xs: float) -> str:
    return xs[0].upper()
"""


def test_a_variadic_helper_the_scenario_calls_is_compiled_through_its_calls(
    tmp_path: Path,
) -> None:
    # The appended calls pass a **kwargs no keyword and a *args one value, so
    # kw["speed"] and xs[1] were refused for what setup() passes (#266).
    setup = _TIMEOUT_SETUP + 'logger.info("%f %f", opts(speed=2.0), pair(1.0, 2.0))\n'
    helpers = _VARIADIC_HELPERS.split("\n\ndef unused(")[0] + "\n"
    scenario, config, _ = _write_case(tmp_path, setup, helpers)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()
    assert not result.warnings, result.format()

    # One nobody calls: still called, so its error is found, unless it reads
    # its **kwargs by key, which is then left out with a warning.
    scenario, config, path = _write_case(tmp_path, setup, _VARIADIC_HELPERS)
    result = typecheck_scenario(scenario, config)
    error = _only_error(result)
    assert "'float' object has no attribute 'upper'" in error.message
    assert error.line == _line_of(path, "xs[0].upper()")
    [warning] = result.warnings
    assert warning.line == _line_of(path, "def unused(")
    assert "reads `**kw` by key" in warning.message, warning.format()


@pytest.mark.parametrize(
    ("helper", "call", "hint"),
    [
        (
            "def scaled(base: float, **factors: float) -> float:\n"
            "    out = base\n"
            "    for _name, f in factors.items():\n"
            "        out *= f\n"
            "    return out\n",
            "scaled(2.0, a=1.5)",
            'kw.get("name", default)',
        ),
        (
            "def mean(*values: float) -> float:\n"
            "    return sum(values) / len(values) if values else 0.0\n",
            "mean(1.0, 2.0)",
            "len(t) > 0",
        ),
    ],
    ids=["kwargs-items", "args-truth"],
)
def test_codons_limits_on_kwargs_and_args_say_what_to_write(
    tmp_path: Path, helper: str, call: str, hint: str
) -> None:
    # Valid Python Codon refuses: **kwargs is a NamedTuple, *args a Tuple (#266).
    setup = _TIMEOUT_SETUP + f'logger.info("%f", {call})\n'
    scenario, config, _ = _write_case(tmp_path, setup, "\n\n" + helper)
    error = _only_error(typecheck_scenario(scenario, config))
    assert hint in error.message, error.format()


def test_a_plain_tuples_truth_is_not_told_of_args(tmp_path: Path) -> None:
    # The refusal is Codon's for any tuple: its hint names no *args the
    # scenario never wrote (#278).
    setup = 'window = (1.0, 2.0)\nif window:\n    logger.info("window %s", window)\n'
    scenario, config, path = _write_case(tmp_path, _TIMEOUT_SETUP + setup)
    error = _only_error(typecheck_scenario(scenario, config))
    assert "no tuple a truth value" in error.message, error.format()
    assert "len(t) > 0" in error.message, error.format()
    assert "len(args)" not in error.message, error.format()
    assert error.line == _line_of(path, "if window:")


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
    # In the author's terms (#527): no `self`, no "no function".
    assert (
        error.message
        == "CaseParent() cannot be called with (EgoConfig, Lanelet2Pose, str)"
    ), error.format()
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


def test_cases_that_differ_only_in_their_values_compile_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sweep's cases are one program to Codon when only their values
    differ (Driver.shape): the first case's verdict, a failure too, is every
    such case's."""
    from autoware_carla_scenario.typecheck import check

    monkeypatch.setenv(check.CACHE_DIR_ENV, "")  # this process's verdicts only
    monkeypatch.setattr(check, "_CACHE", {})
    compiled: list[str] = []
    compile_ = check._compile

    def counted(name: str, sources: Any, driver: Any, *args: Any) -> Any:
        compiled.append(driver.source)
        return compile_(name, sources, driver, *args)

    monkeypatch.setattr(check, "_compile", counted)
    scenario, config, _ = _write_case(tmp_path, _VALID_SETUP)
    first = typecheck_scenario(scenario, config, {"min_speed_kmh": 5, "name": "a"})
    assert first.ok, first.format()
    second = typecheck_scenario(scenario, config, {"min_speed_kmh": 9, "name": "b"})
    assert second is first
    assert len(compiled) == 1

    for value in ("fast", "slow"):
        error = _only_error(
            typecheck_scenario(scenario, config, {"timeout_seconds": value})
        )
        assert error.path == "scenario.timeout_seconds"
    assert len(compiled) == 2


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


_CHOICE_CONFIG = """
import enum
from dataclasses import dataclass


class Mode(enum.Enum):
    FAST = "fast"
    SLOW = "slow"


@dataclass
class CaseConfig:
    name: str = "p"
    timeout_seconds: float = 5.0
    mode: Mode = Mode.FAST
    pos: tuple[float, float] = (0.0, 1.0)
"""

_CONVERTING_CHOICE_CONFIG = (
    _CHOICE_CONFIG
    + """
    def __post_init__(self) -> None:
        if isinstance(self.mode, str):
            self.mode = Mode(self.mode)
        self.pos = (float(self.pos[0]), float(self.pos[1]))
"""
)


def test_an_enum_or_tuple_the_config_converts_compiles(tmp_path: Path) -> None:
    # The runner passes the YAML str / list to the constructor: it compiles
    # only where the config's __post_init__ converts them (#531).
    setup = "logger.info('%s', cfg.mode.value)\nlogger.info('%f', cfg.pos[1])\n"
    values = {"mode": "slow", "pos": [1.0, 2.0]}
    scenario, config, _ = _write_case(tmp_path, setup, config=_CONVERTING_CHOICE_CONFIG)
    result = typecheck_scenario(scenario, config, values)
    assert result.ok, result.format()

    scenario, config, _ = _write_case(tmp_path, setup, config=_CHOICE_CONFIG)
    paths = {d.path for d in typecheck_scenario(scenario, config, values).diagnostics}
    assert "scenario.mode" in paths


_LEGS_CONFIG = """
from dataclasses import dataclass, field


@dataclass
class Leg:
    lanelet_id: int = 0
    speed: float = 1.0


@dataclass
class CaseConfig:
    legs: list[Leg] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.legs = [Leg(**n) if isinstance(n, dict) else n for n in self.legs]
"""


def test_a_bad_value_in_a_nested_list_names_its_field(tmp_path: Path) -> None:
    setup = "logger.info('%d', cfg.legs[0].lanelet_id)\n"
    scenario, config, _ = _write_case(tmp_path, setup, config=_LEGS_CONFIG)
    values = {"legs": [{"lanelet_id": 1}, {"lanelet_id": 2, "speed": "x"}]}
    error = _only_error(typecheck_scenario(scenario, config, values))
    assert error.path == "scenario.legs[1].speed"


_BOOL_CONFIG = """
from dataclasses import dataclass


@dataclass
class CaseConfig:
    timeout_seconds: float = 10.0
    enabled: bool = True
"""

_REQUIRED_BOOL_CONFIG = """
from dataclasses import dataclass


@dataclass
class CaseConfig:
    enabled: bool
    timeout_seconds: float = 10.0
"""


@pytest.mark.parametrize(
    "config_source", [_BOOL_CONFIG, _REQUIRED_BOOL_CONFIG], ids=["default", "required"]
)
def test_a_string_for_a_top_level_bool_is_reported_at_its_key(
    tmp_path: Path, config_source: str
) -> None:
    # Hydra reads `scenario.enabled=no` as the str "no", which Codon converts
    # to a bool field (#598): the value is checked strictly, as a nested
    # field's is, by assignment and by the constructor (a required field)
    # alike.  An int for a float field still passes.
    setup = "if cfg.enabled:\n    logger.info('%f', cfg.timeout_seconds)\n"
    scenario, config, _ = _write_case(tmp_path, setup, config=config_source)
    error = _only_error(typecheck_scenario(scenario, config, {"enabled": "no"}))
    assert error.path == "scenario.enabled", error.format()
    assert "'str' does not match expected type 'bool'" in error.message
    if config_source is _REQUIRED_BOOL_CONFIG:
        return  # _HEADER's `config or CaseConfig()` needs every default

    result = typecheck_scenario(
        scenario, config, {"enabled": False, "timeout_seconds": 3}
    )
    assert result.ok, result.format()


def test_an_undeclared_attribute_is_refused_before_compiling(tmp_path: Path) -> None:
    scenario, config, path = _write_case(tmp_path, "self._count = 3\n")
    error = _only_error(typecheck_scenario(scenario, config))
    assert "CaseScenario._count is assigned without a type" in error.message
    assert "`_count: int`" in error.message
    assert error.line == _line_of(path, "self._count = 3")


def test_a_json_document_is_held_to_the_members_of_jsonvalue(tmp_path: Path) -> None:
    # What loads() returns has no type the check knows: it is converted
    # explicitly, and handing it on as a float or a str is refused.
    scenario, config, _ = _write_case(
        tmp_path,
        'import json\ncfg = json.loads("{}")\nspeed = float(cfg["v"])\n'
        'name = str(cfg["n"])\n',
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()
    scenario, config, _ = _write_case(
        tmp_path, 'import json\ncfg = json.loads("{}")\nspeed: float = cfg["v"]\n'
    )
    error = _only_error(typecheck_scenario(scenario, config))
    # The declared type is named first, as for a return (#564), with a hint.
    assert "speed is assigned a JSONValue, but is declared float" in error.message
    assert 'convert it where you use it, `float(doc["speed"])`' in error.message
    assert "does not match expected type" not in error.message


def test_a_json_document_is_read_from_a_file_and_given_defaults(
    tmp_path: Path,
) -> None:
    # The usual ways to read a side file (#562): a builtin open() (with
    # CPython's text keywords), a default for get(), a slice of a list.
    scenario, config, _ = _write_case(
        tmp_path,
        "import json\n"
        'with open("wp.json") as f:\n'
        "    doc = json.load(f)\n"
        'other = json.load(open("wp.json", "r", encoding="utf-8"))\n'
        'lane = int(doc.get("lane", 1))\n'
        'speed = float(doc.get("speed", 5.0))\n'
        'name = str(other.get("name", "x"))\n'
        'sub = doc["wps"][1:3]\n'
        "for wp in sub:\n"
        '    speed += float(wp["v"])\n'
        'with open("out.json", "w", encoding="utf-8", newline="") as g:\n'
        '    json.dump({"a": speed}, g)\n',
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()
    # What the get() default allows is not a way round the conversion.
    scenario, config, _ = _write_case(
        tmp_path,
        'import json\ndoc = json.loads("{}")\nspeed: float = doc.get("v", 5.0)\n',
    )
    error = _only_error(typecheck_scenario(scenario, config))
    assert "speed is assigned a JSONValue, but is declared float" in error.message


@pytest.mark.parametrize(
    ("setup", "needle"),
    [
        ('xs: list[float] = doc["xs"]\n', "xs is assigned a JSONValue"),
        ('a = doc["v"] + 1.0\n', "unsupported operand"),
        ('a = doc["v"] > 3.0\n', "unsupported operand"),
        ('a = sum(doc["xs"])\n', "unsupported operand"),
        ('for wp in doc["wps"]:\n    b = wp.x\n', "no attribute 'x'"),
        ('doc["v"] = 2.0\n', "read-only here"),
    ],
)
def test_a_json_misuse_says_to_convert_the_value(
    tmp_path: Path, setup: str, needle: str
) -> None:
    scenario, config, _ = _write_case(
        tmp_path, 'import json\ndoc = json.loads("{}")\n' + setup
    )
    error = _only_error(typecheck_scenario(scenario, config))
    assert needle in error.message
    assert " a JSON " in error.message, error.message  # the hint


@pytest.mark.parametrize(
    "import_line",
    ["import json\n", "import json as js\n", "from json import loads\n"],
)
def test_wrong_code_in_an_isinstance_branch_of_a_json_value_is_refused(
    tmp_path: Path, import_line: str
) -> None:
    # Codon decides isinstance by the static type and a JSONValue is none of
    # dict/list/str, so the branch it guards was not compiled (#563).
    loads = "loads" if "from" in import_line else import_line.split()[-1] + ".loads"
    for branch, needle in (
        ('    q: int = "wrong"\n', "q is assigned a str"),
        ("    z = v.nonexistent()\n", "nonexistent"),
        ("    total = 0\n    for x in v:\n        total += x\n", "operand"),
    ):
        scenario, config, _ = _write_case(
            tmp_path,
            import_line
            + f'v = {loads}("{{}}")["k"]\nif isinstance(v, (dict, list)):\n'
            + branch,
        )
        error = _only_error(typecheck_scenario(scenario, config))
        assert needle in error.message, error.message


def test_a_json_value_is_told_apart_with_isinstance(tmp_path: Path) -> None:
    scenario, config, _ = _write_case(
        tmp_path,
        "import json\n"
        'doc = json.loads("{}")\n'
        "def total(v) -> float:\n"
        "    out = 0.0\n"
        "    if isinstance(v, list):\n"
        "        for x in v:\n"
        "            out += float(x)\n"
        "    elif isinstance(v, dict):\n"
        "        for k, x in v.items():\n"
        "            out += float(x)\n"
        "    elif isinstance(v, str):\n"
        "        out = float(len(v))\n"
        "    else:\n"
        "        out = float(v)\n"
        "    return out\n"
        'top = total(doc["k"])\n'
        'wps = doc["wps"]\n'
        "if isinstance(wps, list):\n"
        "    first = wps[0]\n"
        "    top += float(first)\n"
        "elif isinstance(wps, str):\n"
        "    top += float(wps)\n",
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()
    # The else branch of the same test is compiled too.
    scenario, config, _ = _write_case(
        tmp_path,
        'import json\nv = json.loads("{}")["k"]\n'
        'if isinstance(v, list):\n    n = len(v)\nelse:\n    q: int = "wrong"\n',
    )
    assert "q is assigned" in _only_error(typecheck_scenario(scenario, config)).message


def test_isinstance_of_a_known_type_is_left_alone_in_a_json_module(
    tmp_path: Path,
) -> None:
    # The rewrite adds a static false for any type but a JSONValue: dead code
    # of a known type stays uncompiled, as in a module without json.
    scenario, config, _ = _write_case(
        tmp_path,
        'import json\ny = 5\nif isinstance(y, str):\n    q: int = "wrong"\n',
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_a_module_without_a_codon_model_is_named(tmp_path: Path) -> None:
    scenario, config, path = _write_case(
        tmp_path, 'import textwrap\ntextwrap.dedent("")\n'
    )
    error = _only_error(typecheck_scenario(scenario, config))
    assert "textwrap has no Codon model" in error.message
    assert error.path == str(path)


@pytest.mark.parametrize(
    ("setup", "needle", "message"),
    [
        (
            "from autoware_carla_scenario.actions import ActionState\n",
            "import ActionState",
            "autoware_carla_scenario.actions has no ActionState in Python",
        ),
        (
            "from autoware_carla_scenario.coordinate import Lanelet2Pose as P, RoadNetwork\n",
            "RoadNetwork",
            "autoware_carla_scenario.coordinate has no RoadNetwork in Python",
        ),
        (
            "from autoware_carla_scenario.coverage import prepare_measures\n",
            "prepare_measures",
            "autoware_carla_scenario.coverage has no prepare_measures in Python",
        ),
        (
            "from autoware_carla_scenario.entity import EgoConfig as E\n",
            "import EgoConfig",
            "autoware_carla_scenario.entity has no EgoConfig in Python",
        ),
        (
            "from autoware_carla_scenario import _lists\n",
            "import _lists",
            "autoware_carla_scenario has no _lists in Python",
        ),
        (
            "import autoware_carla_scenario._tc\n",
            "_tc",
            "autoware_carla_scenario._tc is a module of the Codon model only",
        ),
        (
            "from autoware_carla_scenario._lists import _acs_list\n",
            "_lists import",
            "autoware_carla_scenario._lists is a module of the Codon model only",
        ),
    ],
    ids=[
        "actions-ActionState",
        "coordinate-RoadNetwork",
        "coverage-prepare_measures",
        "entity-EgoConfig",
        "package-_lists",
        "module-_tc",
        "module-_lists",
    ],
)
def test_an_import_python_cannot_run_is_refused_at_its_line(
    tmp_path: Path, setup: str, needle: str, message: str
) -> None:
    # The model declares these, Python's package lacks them (#214): an import
    # inside a function is first run when the scenario runs, so the check
    # must refuse it; at the top of a module, Python refuses it on import.
    scenario, config, path = _write_case(
        tmp_path, setup + textwrap.dedent(_VALID_SETUP)
    )
    result = typecheck_scenario(scenario, config)
    error = _only_error(result)
    assert len(result.diagnostics) == 1, result.format()
    assert message in error.message, error.format()
    assert "ImportError" in error.message or "fails when" in error.message
    assert error.path == str(path)
    assert error.line == _line_of(path, needle)


def test_an_import_python_can_run_or_never_runs_compiles(tmp_path: Path) -> None:
    # Names both have, and what only `if TYPE_CHECKING:` imports (Python
    # never runs it), in a function.
    setup = """
from typing import TYPE_CHECKING
from autoware_carla_scenario.actions import TurnIndicatorAction as Indicator
from autoware_carla_scenario.coordinate import Lanelet2Pose as Pose
if TYPE_CHECKING:
    from autoware_carla_scenario.coordinate import RoadNetwork
pose: Pose = Pose(460, 1.0)
indicators: list[Indicator] = []
logger.info("%s %d", str(pose.lanelet_id), len(indicators))
"""
    scenario, config, _ = _write_case(tmp_path, setup + textwrap.dedent(_VALID_SETUP))
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_python_finds_a_submodule_imported_by_name() -> None:
    # `from package import submodule` works in Python whether or not the
    # package has imported the submodule yet.
    from autoware_carla_scenario.typecheck.check import _missing_at_run_time

    package = "autoware_carla_scenario"
    assert _missing_at_run_time(package, ["coordinate", "BaseScenario", "*"]) == []
    assert _missing_at_run_time(f"{package}.utils", ["traffic_light"]) == []
    assert _missing_at_run_time(f"{package}.coordinate", ["RoadNetwork"]) != []


def test_a_lazy_name_that_fails_to_import_is_left_to_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A module's __getattr__ (the package's lazy imports) may raise more than
    # AttributeError, an optional dependency missing: not the check's to report.
    import types

    from autoware_carla_scenario.typecheck.check import _missing_at_run_time

    def lazy(name: str) -> object:
        raise ImportError(f"no optional dependency for {name}")

    module = types.ModuleType("acs_typecheck_lazy_case")
    module.__getattr__ = lazy  # type: ignore[method-assign]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert _missing_at_run_time(module.__name__, ["Thing"]) == []


def test_a_failing_hint_never_hides_codons_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The hints on what to declare (#218) are added after Codon failed; if
    # they fail themselves, Codon's own errors are still reported.
    from autoware_carla_scenario.typecheck import check

    def broken(*_args: object) -> list[Any]:
        raise RuntimeError("unexpected Codon output")

    monkeypatch.setattr(check, "_uninferred_attributes", broken)
    scenario, config, path = _write_case(
        tmp_path, "self.register_fail_condition(TimeoutCondition(10.0))\n"
    )
    result = typecheck_scenario(scenario, config)
    error = _only_error(result)
    assert "label is a required keyword argument" in error.message, result.format()
    assert error.path == str(path)


_ABSTRACT_CONDITION = """

class GapCondition(BaseCondition):
    def __init__(self) -> None:
        super().__init__("gap")

    def _check(self, world: carla.World, elapsed: float) -> ScenarioResult | None:
        return None
"""

_ABSTRACT_ACTION = """

class HonkAction(BaseAction):
    def __init__(self) -> None:
        super().__init__("honk")

    def _execute(self, world: carla.World) -> None:
        pass
"""


@pytest.mark.parametrize(
    ("setup", "extra", "header", "cls", "missing", "where"),
    [
        (
            "",
            "",
            _HEADER.replace("def is_done(self)", "def is_finished(self)"),
            "CaseScenario",
            "is_done",
            "by the runner",
        ),
        (
            "self.register_fail_condition(GapCondition())\n",
            _ABSTRACT_CONDITION,
            _HEADER,
            "GapCondition",
            "check",
            "GapCondition()",
        ),
        (
            "self.register_pre_tick(HonkAction())\n",
            _ABSTRACT_ACTION,
            _HEADER.replace("    AndCondition,", "    AndCondition,\n    BaseAction,"),
            "HonkAction",
            "execute",
            "HonkAction()",
        ),
    ],
    ids=["scenario-is_done", "condition-check", "action-execute"],
)
def test_a_class_built_with_an_abstract_method_undefined_is_refused(
    tmp_path: Path,
    setup: str,
    extra: str,
    header: str,
    cls: str,
    missing: str,
    where: str,
) -> None:
    # Codon drops @abstractmethod and the model gives these methods bodies,
    # so only Python can tell (#215): building one raises TypeError.
    scenario, config, path = _write_case(tmp_path, setup, extra, header=header)
    result = typecheck_scenario(scenario, config)
    error = _only_error(result)
    assert len(result.diagnostics) == 1, result.format()
    assert f"{cls} is built" in error.message, error.format()
    assert f"leaves abstract {missing} undefined" in error.message
    assert error.path == str(path)
    assert error.line == _line_of(path, f"class {cls}(")
    if where != "by the runner":
        where = f"at scenario.py:{_line_of(path, where)}"
    assert f"({where})" in error.message


def test_an_abstract_base_never_built_compiles(tmp_path: Path) -> None:
    # A base of the scenario's own conditions may stay abstract: only what
    # is built must define every abstract method.
    extra = """

class _GapBase(BaseCondition):
    @abstractmethod
    def limit(self) -> float:
        raise NotImplementedError

    def check(self, world: carla.World, elapsed: float) -> ScenarioResult | None:
        if elapsed < self.limit():
            return None
        return ScenarioResult(passed=True, message="gap", elapsed_seconds=elapsed)


class GapCondition(_GapBase):
    def __init__(self) -> None:
        super().__init__("gap")

    def limit(self) -> float:
        return 3.0
"""
    header = _HEADER.replace(
        "import logging\n", "import logging\nfrom abc import abstractmethod\n"
    )
    setup = "self.register_pass_condition(GapCondition())\n"
    scenario, config, _ = _write_case(tmp_path, setup, extra, header=header)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


_NO_SUPER_CONDITION = """

class LateCondition(BaseCondition):
    _after: float

    def __init__(self, after: float) -> None:
        self._after = after

    def check(self, world: carla.World, elapsed: float) -> ScenarioResult | None:
        if elapsed > self._after:
            return ScenarioResult(passed=True, message="late", elapsed_seconds=elapsed)
        return None
"""

_UNASSIGNED_CONDITION = """

class CountCondition(BaseCondition):
    _count: int
    _limit: int

    def __init__(self, limit: int, *, label: str) -> None:
        super().__init__(label=label)
        self._limit = limit

    def check(self, world: carla.World, elapsed: float) -> ScenarioResult | None:
        self._count += 1
        if self._count > self._limit:
            return ScenarioResult(passed=True, message="count", elapsed_seconds=elapsed)
        return None
"""


@pytest.mark.parametrize(
    ("setup", "extra", "header", "needle", "message"),
    [
        (
            "self.register_pass_condition(LateCondition(3.0))\n",
            _NO_SUPER_CONDITION,
            _HEADER,
            "def __init__(self, after: float)",
            "LateCondition.__init__ never calls BaseCondition.__init__, "
            "which sets label",
        ),
        (
            "",
            "",
            _HEADER.replace(
                "super().__init__(ego_config, spawn_pose=spawn_pose, "
                "ground_projection=ground_projection)",
                "self.ego_spawn = spawn_pose",
            ),
            "def __init__(",
            "CaseScenario.__init__ never calls BaseScenario.__init__",
        ),
        (
            'self.register_pass_condition(CountCondition(3, label="c"))\n',
            _UNASSIGNED_CONDITION,
            _HEADER,
            "    _count: int",
            "CountCondition._count is declared but never assigned",
        ),
    ],
    ids=["condition-without-super", "scenario-without-super", "never-assigned"],
)
def test_an_attribute_python_never_sets_is_refused(
    tmp_path: Path, setup: str, extra: str, header: str, needle: str, message: str
) -> None:
    # Codon zero-initialises every field, so these compiled; Python raises
    # AttributeError at the first read (#587, #588).
    scenario, config, path = _write_case(
        tmp_path, _TIMEOUT_SETUP + setup, extra, header=header
    )
    result = typecheck_scenario(scenario, config)
    error = _only_error(result)
    assert len(result.diagnostics) == 1, result.format()
    assert message in error.message, error.format()
    assert error.path == str(path)
    assert error.line == _line_of(path, needle)


def test_inits_calling_the_base_and_assigned_attributes_compile(
    tmp_path: Path,
) -> None:
    # The look-alikes: a subclass of the scenario's own condition calling
    # super(), a base called by name inside an `if`, and attributes assigned
    # by a tuple target or only in another method (#587, #588).
    extra = """

class LateCondition(BaseCondition):
    _after: float
    _seen: int
    _last: float

    def __init__(self, after: float) -> None:
        if after >= 0.0:
            super().__init__("late")
        else:
            BaseCondition.__init__(self, "never")
        self._after, self._seen = after, 0

    def check(self, world: carla.World, elapsed: float) -> ScenarioResult | None:
        self._seen += 1
        self._last = elapsed
        if elapsed > self._after:
            return ScenarioResult(passed=True, message="late", elapsed_seconds=elapsed)
        return None


class LaterCondition(LateCondition):
    def __init__(self) -> None:
        super().__init__(9.0)
"""
    setup = (
        _TIMEOUT_SETUP
        + "self.register_pass_condition(LateCondition(3.0))\n"
        + "self.register_pass_condition(LaterCondition())\n"
    )
    scenario, config, _ = _write_case(tmp_path, setup, extra)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


_TRACKER = """

class GapTracker:
    {declarations}
    def __init__(self, limit: float = 5.0) -> None:
        self._samples = []
        self.limit = limit

    def add(self, value: float) -> None:
        self._samples.append(value)

    def too_close(self) -> bool:
        return any(s < self.limit for s in self._samples)


class Greeter:
    def __init__(self) -> None:
        self.greeting = "hello"

    def greet(self, name: str) -> str:
        return "hello " + name
"""


def test_an_undeclared_attribute_of_an_annotated_helper_is_refused(
    tmp_path: Path,
) -> None:
    # A class with a class-level annotation has only the attributes it
    # declares (#218): name the one it does not, with the hint.
    extra = _TRACKER.format(declarations="limit: float\n")
    setup = "GapTracker().add(1.0)\n"
    scenario, config, path = _write_case(tmp_path, setup, extra)
    result = typecheck_scenario(scenario, config)
    error = _only_error(result)
    assert len(result.diagnostics) == 1, result.format()
    assert "GapTracker._samples is assigned without a type" in error.message
    assert "`_samples: list[...]`" in error.message
    assert error.line == _line_of(path, "self._samples = []")


def test_a_helper_whose_attributes_codon_cannot_infer_is_named(tmp_path: Path) -> None:
    # A class that declares nothing has its attributes inferred from its
    # __init__; where Codon cannot (an empty list), its error names the
    # attribute holding the helper, and the check names what to declare.
    extra = _TRACKER.format(declarations="") + (
        "\n\nclass Holder:\n"
        "    _tracker: GapTracker\n\n"
        "    def __init__(self) -> None:\n"
        "        self._tracker = GapTracker()\n"
    )
    setup = "Holder()._tracker.add(1.0)\n"
    scenario, config, path = _write_case(tmp_path, setup, extra)
    result = typecheck_scenario(scenario, config)
    assert not result.ok, result.format()
    hints = [d for d in result.diagnostics if "is assigned without a type" in d.message]
    assert {(d.message.split(" ")[0], d.line) for d in hints} >= {
        ("GapTracker._samples", _line_of(path, "self._samples = []")),
    }, result.format()


def test_a_helper_method_reading_an_inferred_attribute_is_named(
    tmp_path: Path,
) -> None:
    # The scenario check calls each method on a value it does not build with
    # __init__, so an attribute Codon would infer from it has no type there
    # (`self: Counter[NoneType]`): the check names what to declare.
    extra = """

class Counter:
    def __init__(self) -> None:
        self.count = 0

    def get(self) -> int:
        return self.count
"""
    setup = 'logger.info("%d", Counter().get())\n'
    scenario, config, path = _write_case(tmp_path, setup, extra)
    result = typecheck_scenario(scenario, config)
    assert not result.ok, result.format()
    hints = [d for d in result.diagnostics if "is assigned without a type" in d.message]
    assert [(d.message.split(" ")[0], d.line) for d in hints] == [
        ("Counter.count", _line_of(path, "self.count = 0"))
    ], result.format()
    assert "`count: int`" in hints[0].message


def test_declared_and_inferable_helpers_compile(tmp_path: Path) -> None:
    # Every attribute declared, or none declared and each inferable
    # (Greeter, whose methods read none).
    extra = _TRACKER.format(declarations="_samples: list[float]\n    limit: float\n")
    setup = (
        "tracker = GapTracker()\ntracker.add(1.0)\n"
        'logger.info("%s %s", tracker.too_close(), Greeter().greet("ego"))\n'
    )
    scenario, config, _ = _write_case(tmp_path, setup, extra)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


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
        'from pathlib import Path\nlogger.info("%s", Path("a").is_socket())\n',
    )
    error = _only_error(typecheck_scenario(scenario, config))
    assert "is_socket" in error.message
    assert "not modelled" in error.message, error.format()
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
        tmp_path, 'import shutil\nshutil.disk_usage("a")\n'
    )
    error = _only_error(typecheck_scenario(scenario, config))
    assert "disk_usage" in error.message
    assert "not modelled" in error.message, error.format()
    assert error.path == str(path)


#: The file helpers #568 and #569 added to the pathlib, os, os.path and shutil
#: shims, each used in a scenario (the library check holds their signatures,
#: ``test_typecheck_library.py``).
_FILE_HELPER_SETUP = """
import os
import os.path
import shutil
from pathlib import Path
base = Path("/tmp/acs") / "a.txt"
logger.info("%s %s", base.with_suffix(".csv").name, base.with_stem("b").name)
logger.info("%s %d", base.joinpath("x", "y").as_posix(), len(base.parts))
logger.info("%s %s", base.suffixes, sorted(Path("/tmp").iterdir()))
base.touch()
base.rename("/tmp/acs/b.txt")
Path("/tmp/acs").rmdir()
logger.info("%d", base.write_bytes(b"x"))
logger.info("%d", len(os.listdir("/tmp")))
os.makedirs("/tmp/acs", exist_ok=True)
os.remove("/tmp/acs/a.txt")
for root, dirs, files in os.walk("/tmp"):
    logger.info("%s %d %d", root, len(dirs), len(files))
logger.info("%d", os.path.getsize("/tmp/a"))
logger.info("%s %s", os.path.relpath("/tmp/a", "/tmp"), os.path.realpath("/tmp"))
shutil.copy("/tmp/a", "/tmp/b")
shutil.copytree("/tmp/a", "/tmp/b", dirs_exist_ok=True)
"""


def test_a_scenario_may_use_the_file_helpers_of_pathlib_os_and_shutil(
    tmp_path: Path,
) -> None:
    scenario, config, _ = _write_case(tmp_path, _FILE_HELPER_SETUP)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


@pytest.mark.parametrize(
    ("setup", "message", "hint"),
    [
        (
            'from pathlib import Path\nPath("a").is_socket()\n',
            "'Path' object has no attribute 'is_socket'",
            "not modelled",
        ),
        (
            "import os\nos.scandir('a')\n",
            "cannot import name 'scandir' from 'os",
            "not modelled",
        ),
        (
            "import os\nos.path.commonpath(['a'])\n",
            "cannot import name 'commonpath' from 'os.path'",
            "not modelled",
        ),
        (
            "import shutil\nshutil.disk_usage('a')\n",
            "cannot import name 'disk_usage' from 'shutil'",
            "not modelled",
        ),
        (
            'from pathlib import Path\nPath("a").with_sufix(".x")\n',
            "'Path' object has no attribute 'with_sufix'",
            "did you mean 'with_suffix'?",
        ),
        (
            "import shutil\nshutil.copytre('a', 'b')\n",
            "cannot import name 'copytre' from 'shutil'",
            "did you mean 'copytree'?",
        ),
        (
            "import time\ntime.strftime('%Y')\n",
            "cannot import name 'strftime' from 'time'",
            "datetime.datetime.now()",
        ),
        (
            "import datetime\ndatetime.datetime.now().strftime('%Y')\n",
            "'datetime' object has no attribute 'strftime'",
            "now.year:04d",
        ),
    ],
    ids=[
        "path",
        "os",
        "os-path",
        "shutil",
        "path-typo",
        "shutil-typo",
        "time-strftime",
        "datetime-strftime",
    ],
)
def test_a_name_the_file_and_time_shims_lack_is_refused_as_not_modelled_or_misspelt(
    tmp_path: Path, setup: str, message: str, hint: str
) -> None:
    scenario, config, _ = _write_case(tmp_path, setup)
    error = _only_error(typecheck_scenario(scenario, config))
    assert message in error.message, error.format()
    assert hint in error.message, error.format()


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
            'self.register_fail_condition(TimeoutCondition("10"))\n',
            'TimeoutCondition("10")',
            "TimeoutCondition(): no constructor takes these arguments "
            "(required: keyword label)",
        ),
        (
            'self.register_fail_condition(SpeedCondition("npc1"))\n',
            'SpeedCondition("npc1")',
            "SpeedCondition(): no constructor takes these arguments",
        ),
        (
            'self.register_pre_tick(TurnAction("npc1"))\n',
            'TurnAction("npc1")',
            "TurnAction(): no constructor takes these arguments "
            "(required: entity_name, direction)",
        ),
        (
            "from autoware_carla_scenario import VehicleEntity\nVehicleEntity()\n",
            "VehicleEntity()",
            "VehicleEntity(): no constructor takes these arguments (required: config)",
        ),
        (
            "from autoware_carla_scenario import PedestrianEntity\nPedestrianEntity()\n",
            "PedestrianEntity()",
            "PedestrianEntity(): no constructor takes these arguments "
            "(required: config)",
        ),
        (
            "from autoware_carla_scenario.debug_markers import DebugMarkers\n"
            "markers = DebugMarkers([])\n",
            "DebugMarkers([])",
            "DebugMarkers(): no constructor takes these arguments (it takes none)",
        ),
        (
            "from autoware_carla_scenario.coverage import CoverItem\n"
            'item = CoverItem("speed")\n',
            'CoverItem("speed")',
            "CoverItem(): no constructor takes these arguments "
            "(required: name, expression)",
        ),
        (
            "from autoware_carla_scenario.coverage import CoverItem\n"
            'item = CoverItem(name="speed")\n',
            'CoverItem(name="speed")',
            "CoverItem(): no constructor takes these arguments "
            "(required: name, expression)",
        ),
        (
            "from autoware_carla_scenario.odd import OddCondition\n"
            "condition = OddCondition(0)\n",
            "OddCondition(0)",
            "OddCondition(): no constructor takes these arguments (it takes none)",
        ),
        (
            # Its fields are Python's since #298: Python builds it by them,
            # the model does not (plan_route_coverage() does).
            "from autoware_carla_scenario.odd import RouteCoverage, RouteLanelet\n"
            "lanelets: list[RouteLanelet] = []\n"
            "expected: dict[str, dict[str, float]] = {}\n"
            'coverage = RouteCoverage(odd="odd", name="route", lanelets=lanelets, '
            'expected_m=expected, map_attributes=["speed"])\n',
            'RouteCoverage(odd="odd"',
            "RouteCoverage: plan_route_coverage() builds it",
        ),
        (
            "from autoware_carla_scenario.ui.models import RunProgress\n"
            'progress = RunProgress(1, 2, "case", "running")\n',
            'RunProgress(1, 2, "case", "running")',
            "RunProgress(): its fields are keyword-only",
        ),
        (
            "from autoware_carla_scenario import FrameMismatchError\n"
            'error = FrameMismatchError("lanelet2")\n',
            'FrameMismatchError("lanelet2")',
            "FrameMismatchError(): no constructor takes these arguments "
            "(required: frame_a, frame_b, operation)",
        ),
        (
            "from autoware_carla_scenario.autoware_stack.devcontainer import "
            "DevContainerError\n"
            'error = DevContainerError("DevContainerError", "no image")\n',
            'DevContainerError("DevContainerError", "no image")',
            "DevContainerError(): no constructor takes these arguments "
            "(required: message)",
        ),
        (
            "from autoware_carla_scenario.autoware_stack.devcontainer import "
            "DevContainerError\n"
            'error = DevContainerError(typename="no image")\n',
            'DevContainerError(typename="no image")',
            "DevContainerError(): no constructor takes these arguments "
            "(required: message)",
        ),
        (
            "from autoware_carla_scenario import FrameMismatchError\n"
            'error = FrameMismatchError("lanelet2", message="add")\n',
            'FrameMismatchError("lanelet2", message="add")',
            "FrameMismatchError(): no constructor takes these arguments "
            "(required: frame_a, frame_b, operation)",
        ),
        (
            "from autoware_carla_scenario.odd import OddAttribute\n"
            'attribute = OddAttribute(name="speed", unit="m", text="t")\n',
            'OddAttribute(name="speed", unit="m", text="t")',
            "OddAttribute(): no constructor takes these arguments "
            "(required: name, probe)",
        ),
        (
            "from autoware_carla_scenario import EntityRole\n"
            'role = EntityRole(_value="npc1")\n',
            'EntityRole(_value="npc1")',
            "EntityRole(): no constructor takes these arguments (required: value)",
        ),
    ],
    ids=[
        "timeout-str-alone",
        "condition-label-alone",
        "action-label-alone",
        "vehicle-entity-no-arguments",
        "pedestrian-entity-no-arguments",
        "debug-markers-by-field",
        "cover-item-by-field",
        "cover-item-by-field-name",
        "odd-condition-by-field",
        "route-coverage-by-fields",
        "run-progress-by-position",
        "frame-mismatch-by-exception-constructor",
        "devcontainer-error-by-exception-constructor",
        "devcontainer-error-by-exception-keyword",
        "frame-mismatch-by-base-exception-keyword",
        "odd-attribute-by-field-names",
        "entity-role-by-field-name",
    ],
)
def test_what_codon_builds_by_the_fields_alone_is_refused_at_its_line(
    tmp_path: Path, setup: str, needle: str, message: str
) -> None:
    # Codon builds a class by its fields too, by position: a class without
    # fields with no arguments (#225), a condition or an action by the label
    # it inherits alone (#236), any class whose constructor Python's fields
    # differ from by them (#244), and an exception by Exception's
    # constructor (#244), by position or by keyword. Python refuses each, and
    # the model's __init__ of those parameters, a compile_error, refuses them
    # too.
    scenario, config, path = _write_case(tmp_path, setup)
    error = _only_error(typecheck_scenario(scenario, config))
    assert message in error.message, error.format()
    assert error.path == str(path)
    assert error.line == _line_of(path, needle)


def test_what_the_constructors_take_still_compiles(tmp_path: Path) -> None:
    # The look-alikes of the calls refused above: the constructor is declared
    # after the refusing __init__, so it wins every call it takes.
    setup = """
        from autoware_carla_scenario import (
            AlwaysTrueCondition,
            PedestrianEntity,
            PedestrianEntityConfig,
            SpawnPointIndex,
            SpawnTransform,
            VehicleEntity,
            VehicleEntityConfig,
        )
        self.register_fail_condition(TimeoutCondition(label="timeout"))
        self.register_fail_condition(TimeoutCondition(10, label="int_timeout"))
        self.register_fail_condition(TimeoutCondition(10.0, label="float_timeout"))
        self.register_pass_condition(AlwaysTrueCondition())
        self.register_pass_condition(TrajectoryTimeCondition(3.0))
        self.register_fail_condition(
            SpeedCondition("npc1", 1.0, ComparisonRule.LESS_THAN, label="slow")
        )
        self.register_pass_condition(
            AndCondition([ElapsedTimeCondition(1.0, label="elapsed")])
        )
        self.register_pre_tick(TurnAction("npc1", TurnDirection.LEFT))
        npc = VehicleEntity(VehicleEntityConfig("npc1", SpawnPointIndex(0)))
        walker = PedestrianEntity(
            PedestrianEntityConfig("walker1", SpawnTransform(carla.Transform()))
        )
        logger.info("%s %s", npc.role_name, walker.role_name)
        from autoware_carla_scenario import CoordinateFrame, FrameMismatchError
        from autoware_carla_scenario.autoware_stack.devcontainer import (
            DevContainerError,
        )
        from autoware_carla_scenario.coverage import CoverItem
        from autoware_carla_scenario.debug_markers import DebugMarkers
        from autoware_carla_scenario.maps.cache import MapCacheError
        from autoware_carla_scenario.odd import OddCondition
        from autoware_carla_scenario.ui.models import RunProgress
        markers = DebugMarkers()
        item = CoverItem("speed", lambda world: 1.0)
        odd_condition = OddCondition()
        progress = RunProgress(current=1, total=2, scenario_name="case", status="running")
        error = FrameMismatchError(
            CoordinateFrame.LANELET2, CoordinateFrame.OPENDRIVE, "distance"
        )
        logger.info(
            "%s %s %d %s %s %s",
            str(bool(markers)),
            item.name,
            progress.current,
            str(error),
            str(DevContainerError("no image")),
            str(MapCacheError("no map")),
        )
        """
    scenario, config, _ = _write_case(tmp_path, setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


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


def read_int(world: carla.World) -> int:
    return 1


def read_text(world: carla.World) -> str:
    return "far"
"""


@pytest.mark.parametrize(
    "build",
    [
        'Metric("mine", read_optional)',
        'Metric(key="mine", read=read_float, unit="m", aggregation="min")',
        'Metric("mine", lambda world: 1.0, "Mine")',
        'Metric("mine", lambda world: None)',
        'Metric("mine", read_int)',
    ],
    ids=["optional", "float-keywords", "lambda", "none-lambda", "int"],
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


_MEASURES_SETUP = """
from autoware_carla_scenario.measures import (
    BUILT_IN_MEASURES,
    VEHICLE_AHEAD_GAP_M,
    Bumpers,
    Measure,
    check_measure_key,
    measure_version,
    measured_scenario,
    read_measure,
    set_measured_scenario,
)
gap: Measure = BUILT_IN_MEASURES[VEHICLE_AHEAD_GAP_M]
for key, measure in BUILT_IN_MEASURES.items():
    logger.info("%s %s [%s] %d", key, measure.text, measure.unit, measure.version)
check_measure_key(gap.key, "setup")
if measured_scenario() is None:
    set_measured_scenario(self)
logger.info("%d", measure_version(gap.key))
logger.info("%s", str(read_measure(VEHICLE_AHEAD_GAP_M, self.world)))
logger.info("%s", str(gap.read(self.world)))
set_measured_scenario(None)
bumpers = Bumpers(front=2.0, rear=1)
logger.info("%f", bumpers.front + bumpers.rear)
"""


def test_a_scenario_may_use_the_measures_model(tmp_path: Path) -> None:
    # Every name of measures.__all__ is in the model (#206).
    scenario, config, _ = _write_case(tmp_path, _MEASURES_SETUP)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


@pytest.mark.parametrize(
    ("wrong", "message"),
    [
        # Measure and Bumpers are frozen dataclasses, @tuples in the model.
        ('gap.unit = "km"', "cannot modify tuple attributes"),
        ("bumpers.front = 3.0", "cannot modify tuple attributes"),
        # A Metric has an aggregation; a Measure does not.
        ('logger.info("%s", gap.aggregation)', "aggregation"),
        # The measured scenario is a BaseScenario or None.
        ('set_measured_scenario("me")', "BaseScenario"),
    ],
    ids=["frozen-measure", "frozen-bumpers", "metric-attribute", "not-a-scenario"],
)
def test_a_misused_measure_is_refused_at_its_line(
    tmp_path: Path, wrong: str, message: str
) -> None:
    scenario, config, path = _write_case(tmp_path, _MEASURES_SETUP + wrong + "\n")
    error = _only_error(typecheck_scenario(scenario, config))
    assert message in error.message, error.format()
    assert error.path == str(path)
    assert error.line == _line_of(path, wrong)


@pytest.mark.parametrize(
    "build",
    [
        'Measure("mine", read_optional)',
        'Measure(key="mine", read=read_float, unit="m", version=2)',
        'Measure("mine", lambda world: 1.0, "m", "Mine")',
        'Measure("mine", lambda world: None)',
        'Measure("mine", read_int)',
    ],
    ids=["optional", "float-keywords", "lambda", "none-lambda", "int"],
)
def test_a_scenario_may_build_its_own_measure(tmp_path: Path, build: str) -> None:
    # Python's Measure takes any callable of a world that returns a float, as
    # Metric does (#203); the model's __new__ takes the same.
    setup = (
        "from autoware_carla_scenario.measures import Measure\n"
        f"mine: Measure = {build}\n"
        'logger.info("%s %d %s", mine.key, mine.version, str(mine.read(self.world)))\n'
    )
    scenario, config, _ = _write_case(tmp_path, setup, extra=_OWN_METRIC_READERS)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_a_measure_read_that_returns_no_number_is_refused(tmp_path: Path) -> None:
    wrong = 'mine = Measure("mine", read_text)'
    setup = "from autoware_carla_scenario.measures import Measure\n" + wrong + "\n"
    scenario, config, path = _write_case(tmp_path, setup, extra=_OWN_METRIC_READERS)
    result = typecheck_scenario(scenario, config)
    assert not result.ok, result.format()
    assert any(
        d.path == str(path) and d.line == _line_of(path, wrong)
        for d in result.diagnostics
    ), result.format()


#: A method of the case scenario (it follows setup() in the class body).
_GAP_METHOD = """
    def _read_gap(self, world: carla.World) -> float | None:
        return 1.0
"""


@pytest.mark.parametrize(
    "read",
    [
        "read_optional",
        "read_float",
        "read_int",
        "lambda world: 1.0",
        "lambda world: None",
        "self._read_gap",
    ],
    ids=["optional", "float", "int", "lambda", "none-lambda", "bound-method"],
)
def test_a_scenario_may_register_a_measure_read_as_python_does(
    tmp_path: Path, read: str
) -> None:
    # register_measure() checks *read* as Measure and Metric do (#216): a
    # float, an int or None, or an Optional of one. A bound method is the
    # reader odd.md's "Scenario measures" shows.
    setup = f'self.register_measure("mine", {read}, unit="m")\n'
    scenario, config, _ = _write_case(
        tmp_path, setup, extra=_GAP_METHOD + _OWN_METRIC_READERS
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


@pytest.mark.parametrize(
    "wrong",
    [
        'self.register_measure("mine", lambda world: "x")',
        'self.register_measure("mine", read_text)',
    ],
    ids=["str-lambda", "str-function"],
)
def test_a_registered_measure_read_that_returns_no_number_is_refused(
    tmp_path: Path, wrong: str
) -> None:
    # The model checked *read* as a cover expression, which takes any return
    # type, so a reader of a str passed the check (#216).
    scenario, config, path = _write_case(
        tmp_path, wrong + "\n", extra=_OWN_METRIC_READERS
    )
    result = typecheck_scenario(scenario, config)
    assert not result.ok, result.format()
    assert any(
        d.path == str(path)
        and d.line == _line_of(path, wrong)
        and "expected a float (or an int)" in d.message
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
# Public names the model declares, and those it leaves out on purpose (#237)
# ---------------------------------------------------------------------------

_PUBLIC_NAMES_SETUP = """
from autoware_carla_scenario import (
    BaseAction,
    BridgePose,
    CarlaDriverEntity,
    EgoDriverGrpcClient,
    GrpcAutowareBridgeServer,
    LaneChangeAction,
    MapManager,
    TrafficContext,
    get_stop_line_linestrings,
    register_backend,
    resolve_odd,
    to_map_frame,
)
from autoware_carla_scenario.action_state import ActionState
from autoware_carla_scenario.conditions import (
    ActionStateCondition,
    LaneChangeSettledCondition,
    compare,
    find_actor_in_list,
)
from autoware_carla_scenario.coordinate import AnyPose
from autoware_carla_scenario.debug_markers import DRIVEN_PATH_COLOR, TRAJECTORY_COLOR
from autoware_carla_scenario.entity_role import find_actor_in_list as find_actor
from autoware_carla_scenario.entity_role import find_actor_in_world
from autoware_carla_scenario.environment import ENVIRONMENT_SUFFIX, LOCK_SUFFIX
from autoware_carla_scenario.kinematics import (
    CoordinateFrame,
    FrameMismatchError,
    frame_of,
    normalize_angle_deg,
)
from autoware_carla_scenario.maps.cache import (
    AUTOWARE_DATA_DIR_ENV,
    CACHE_ROOT_ENV,
    TIP_SUFFIX,
)
from autoware_carla_scenario.trajectory_recorder import EGO_ROLES, RECORDED_KINDS
turn = TurnAction(EGO_ROLE_NAME, TurnDirection.LEFT, timing=TickTiming.PRE_TICK)
self.register_pre_tick(turn)
# Actions of different classes, item by item: still valid now that a display
# of them compiles too (test_a_dict_display_of_different_classes_compiles).
lane_change = LaneChangeAction(EGO_ROLE_NAME, LaneChangeDirection.LEFT)
actions: dict[str, BaseAction] = {}
actions["turn"] = turn
actions["lane_change"] = lane_change
self.register_pass_condition(
    ActionStateCondition("turn", ActionState.COMPLETE, actions, label="turned")
)
self.register_fail_condition(
    ActionStateCondition("turn", "runningState", {"turn": turn})
)
# Looked up on every check, so the mapping may be empty when it is built.
self.register_fail_condition(ActionStateCondition("a", "completeState", {}, label="done"))
self.register_fail_condition(LaneChangeSettledCondition(EGO_ROLE_NAME, label="settled"))
self.register_fail_condition(LaneChangeSettledCondition("npc1"))
heading = normalize_angle_deg(370.0)
if compare(heading, ComparisonRule.LESS_THAN, 15.0, tolerance=0.5):
    logger.info("%f", heading)
pose: AnyPose = Lanelet2Pose(460, 1.0)
if frame_of(pose) == CoordinateFrame.LANELET2:
    logger.info("lanelet2")
try:
    logger.info("%f", to_map_frame(Lanelet2Pose(460, 1.0)).position.x)
except FrameMismatchError as exc:
    logger.info("%s", str(exc))
actor = find_actor_in_list(self.world.get_actors(), EGO_ROLE_NAME)
if actor is not None:
    logger.info("%s", actor.type_id)
listed: list[carla.Actor] = []
ego = find_actor_in_world(self.world, EGO_ROLE_NAME)
if ego is not None:
    logger.info("%s", ego.type_id)
other = find_actor(listed, "npc1")
if other is not None:
    logger.info("%s", other.type_id)
logger.info("%d", len(get_stop_line_linestrings(460)))
logger.info("%s %s %s", ENVIRONMENT_SUFFIX, LOCK_SUFFIX, TIP_SUFFIX)
logger.info("%s %s", CACHE_ROOT_ENV, AUTOWARE_DATA_DIR_ENV)
logger.info("%s %s", EGO_ROLES[0], RECORDED_KINDS[1])
logger.info("%d %d", TRAJECTORY_COLOR[1], DRIVEN_PATH_COLOR[0])
"""


def test_the_public_names_the_model_now_declares_compile(tmp_path: Path) -> None:
    # Each was refused (`cannot import name`, or no import path at all for
    # ActionState) though Python exports it (#237); each is used as Python's
    # signature allows.
    scenario, config, _ = _write_case(
        tmp_path, _PUBLIC_NAMES_SETUP + textwrap.dedent(_VALID_SETUP)
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


_ADVERSARIAL_SETUP = """
from autoware_carla_scenario import (
    AdversarialBrakeAction,
    AdversarialCutInAction,
    AdversarialMergeAction,
    AdversityFeasibleCondition,
    ConstructionSettings,
    ConstructionZoneAction,
    AdversityKind,
    AdversityLocation,
    HeadOnAction,
    IgnoreLeaderAction,
    JaywalkAction,
    LaneChangeAbortAction,
    PedestrianRunRedLightAction,
    PropAction,
    PropOperation,
    RunTrafficRuleAction,
    Lanelet2Pose,
    MapPose,
    StalledObjectAction,
    StopCrossingAction,
    TrajectoryFollowingMode,
    collision_objects,
)
self.register_pre_tick(AdversarialBrakeAction("npc1", AdversityLocation.ANY))
self.register_pre_tick(
    ConstructionZoneAction(
        12, 20.0, 80.0, ConstructionSettings(construction_type="mixed"), end_time=30.0
    )
)
self.register_pre_tick(LaneChangeAbortAction("npc2", AdversityLocation.HIGHWAY))
self.register_pre_tick(AdversarialMergeAction("npc2", AdversityLocation.HIGHWAY))
self.register_pre_tick(HeadOnAction("npc3", AdversityLocation.ANY, label="head_on"))
self.register_pre_tick(
    IgnoreLeaderAction(
        "npc1",
        AdversityLocation.INTERSECTION,
        following_mode=TrajectoryFollowingMode.POSITION,
    )
)
self.register_pre_tick(
    AdversarialCutInAction(
        "npc2",
        AdversityLocation.HIGHWAY,
        ElapsedTimeCondition(3.0, label="later"),
        label="cut",
        once=False,
    )
)
self.register_pre_tick(RunTrafficRuleAction("npc3", AdversityLocation.ROUNDABOUT))
self.register_pre_tick(JaywalkAction("p1", AdversityLocation.ANY, seed=7))
self.register_pre_tick(PedestrianRunRedLightAction("p1", AdversityLocation.INTERSECTION))
self.register_pre_tick(StopCrossingAction("p1", AdversityLocation.ANY, seed=3))
self.register_pre_tick(StalledObjectAction("npc3", Lanelet2Pose(1, 2.0), 1.0, 5.0))
self.register_pre_tick(StalledObjectAction("npc4", MapPose(1.0, 2.0, 0.5), label="stalled"))
self.register_pre_tick(
    PropAction("cone1", PropOperation.ADD, Lanelet2Pose(1, 4.0), blueprint="static.prop.streetbarrier")
)
self.register_pre_tick(PropAction("box", position=MapPose(1.0, 2.0), on_ground=False))
self.register_pre_tick(
    PropAction(
        "cone1",
        PropOperation.REMOVE,
        condition=ElapsedTimeCondition(8.0, label="gone"),
        label="clear",
    )
)
for crashed in collision_objects(
    "npc1", "npc2", Lanelet2Pose(1, 2.0), Lanelet2Pose(1, 6.0), end_time=9.0
):
    self.register_pre_tick(crashed)
feasible = AdversityFeasibleCondition(
    "npc1", AdversityKind.CUT_IN, AdversityLocation.ANY, label="may_cut_in", seed=1
)
logger.info(
    "%s %s %s %s %s",
    feasible.kind.name,
    feasible.location.value,
    AdversityKind.HEAD_ON.name,
    AdversityKind.STOP_CROSSING.value,
    AdversityKind.MERGE.name,
)
self.register_fail_condition(feasible)
"""


_NADE_SETUP = """
from autoware_carla_scenario import (
    AdversarialTrafficAction,
    AdversityKind,
    AdversityLocation,
    NadeAdversity,
    NadeSettings,
    SamplingEvent,
)
adversities = [
    NadeAdversity(AdversityKind.CUT_IN, AdversityLocation.HIGHWAY, 0.002),
    NadeAdversity(AdversityKind.BRAKE_CHECK, AdversityLocation.ANY, 0.01),
]
nade = AdversarialTrafficAction(
    adversities, seed=3, settings=NadeSettings(max_controlled=3, fixed_is_probability=0.1)
)
self.register_pre_tick(nade)
logger.info("%f %d", nade.weight, adversities[0].kind == AdversityKind.CUT_IN)
# #397: the kind sampled, as a cover item; the run's weight.
self.register_cover(
    "adversity.kind", nade.sampled_kind, values=AdversityKind, event=SamplingEvent.TICK
)
logger.info("%f", self.importance_sampling_weight * nade.importance_sampling_weight)
"""


def test_the_nade_action_compiles(tmp_path: Path) -> None:
    # #395: the action and its list's entries, from the top level.
    scenario, config, _ = _write_case(
        tmp_path, textwrap.dedent(_NADE_SETUP) + textwrap.dedent(_VALID_SETUP)
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_the_adversarial_actions_compile(tmp_path: Path) -> None:
    # #393: the actions, their condition and its enums, from the top level.
    scenario, config, _ = _write_case(
        tmp_path, textwrap.dedent(_ADVERSARIAL_SETUP) + textwrap.dedent(_VALID_SETUP)
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


@pytest.mark.parametrize(
    "setup",
    [
        'AdversarialBrakeAction("npc1")\n',  # the location is required
        'JaywalkAction("p1", AdversityKind.JAYWALKING)\n',
        'AdversityFeasibleCondition("npc1", AdversityKind.CUT_IN, AdversityLocation.ANY)\n',
    ],
)
def test_an_adversarial_action_built_wrong_is_refused(
    tmp_path: Path, setup: str
) -> None:
    imports = (
        "from autoware_carla_scenario import (\n"
        "    AdversarialBrakeAction, AdversityFeasibleCondition, AdversityKind,\n"
        "    AdversityLocation, JaywalkAction,\n"
        ")\n"
    )
    scenario, config, _ = _write_case(
        tmp_path, imports + setup + textwrap.dedent(_VALID_SETUP)
    )
    assert not typecheck_scenario(scenario, config).ok


@pytest.mark.parametrize(
    ("setup", "needle", "message"),
    [
        (
            'ActionStateCondition("turn", 1, {"turn": turn})\n',
            "ActionStateCondition(",
            "expected an ActionState or a str",
        ),
        (
            'ActionStateCondition("turn", ActionState.COMPLETE, {"turn": 1})\n',
            "ActionStateCondition(",
            "_acs_action",
        ),
        (
            'ActionStateCondition("turn", "completeState", {"turn": turn}, "done")\n',
            "ActionStateCondition(",
            "ActionStateCondition(): label is keyword-only",
        ),
        (
            "LaneChangeSettledCondition(3)\n",
            "LaneChangeSettledCondition(3)",
            "expected an EntityRole or a str",
        ),
        (
            "compare(1.0, 2.0, 3.0)\n",
            "compare(1.0",
            "",
        ),
    ],
    ids=[
        "state-int",
        "actions-not-actions",
        "label-by-position",
        "settled-int",
        "compare-rule",
    ],
)
def test_a_misused_new_public_name_is_refused_at_its_line(
    tmp_path: Path, setup: str, needle: str, message: str
) -> None:
    # What Python's signatures refuse, the model refuses too.
    imports = (
        "from autoware_carla_scenario.action_state import ActionState\n"
        "from autoware_carla_scenario.conditions import (\n"
        "    ActionStateCondition, LaneChangeSettledCondition, compare,\n"
        ")\n"
        "turn = TurnAction(EGO_ROLE_NAME, TurnDirection.LEFT)\n"
    )
    scenario, config, path = _write_case(tmp_path, imports + setup)
    error = _only_error(typecheck_scenario(scenario, config))
    assert message in error.message, error.format()
    assert error.path == str(path)
    assert error.line == _line_of(path, needle)


# ---------------------------------------------------------------------------
# Dict displays whose values mix classes (#250)
# ---------------------------------------------------------------------------

_MIXED_DICT_IMPORTS = """
from autoware_carla_scenario import BaseAction, LaneChangeAction
from autoware_carla_scenario.action_state import ActionState
from autoware_carla_scenario.conditions import ActionStateCondition
turn = TurnAction(EGO_ROLE_NAME, TurnDirection.LEFT)
lc = LaneChangeAction(EGO_ROLE_NAME, LaneChangeDirection.LEFT)
"""


@pytest.mark.parametrize(
    "setup",
    [
        """
        self.register_pass_condition(
            ActionStateCondition(
                "lc", ActionState.COMPLETE, {"turn": turn, "lc": lc}, label="changed"
            )
        )
        """,
        """
        actions: dict[str, BaseAction] = {"turn": turn, "lc": lc}
        self.register_fail_condition(
            ActionStateCondition("turn", ActionState.COMPLETE, actions, label="turned")
        )
        """,
        """
        conditions = {
            "timeout": TimeoutCondition(10.0, label="t"),
            "slow": SpeedCondition(
                EGO_ROLE_NAME, 1.0, ComparisonRule.LESS_THAN, label="slow"
            ),
        }
        for condition in conditions.values():
            self.register_fail_condition(condition)
        """,
    ],
    ids=["argument", "annotated", "conditions"],
)
def test_a_dict_display_of_different_classes_compiles(
    tmp_path: Path, setup: str
) -> None:
    # Codon types a dict display by its first item: each of these was refused
    # (`'LaneChangeAction' does not match expected type 'TurnAction'`) until
    # the check rewrote the display (_acs_dict, _acs_dict_into).
    setup = _MIXED_DICT_IMPORTS + textwrap.dedent(setup)
    scenario, config, _ = _write_case(tmp_path, setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


_OWN_RULES = """
from __future__ import annotations

from dataclasses import dataclass, field


class Rule:
    def apply(self, v: float) -> float:
        return v


class Dbl(Rule):
    def apply(self, v: float) -> float:
        return v * 2.0


class Neg(Rule):
    def apply(self, v: float) -> float:
        return -v

    def flip(self) -> bool:
        return True


class Other:
    def apply(self, v: float) -> float:
        return v


RULES: list[Rule] = [Dbl()]
PAIR: list[Rule] = [Dbl(), Neg()]


def make() -> list[Rule]:
    return [Dbl(), Neg()]


def make_one() -> list[Rule]:
    return [Dbl()]


def total(rules: list[Rule], v: float = 1.0) -> float:
    out = v
    for rule in rules:
        out = rule.apply(out)
    return out


def use_argument() -> float:
    return total([Dbl(), Neg()]) + total([Neg()], v=2.0)


def use_keyword() -> float:
    return total(rules=[Dbl(), Neg()])


def use_common_base() -> float:
    mixed = [Dbl(), Neg()]
    return total(mixed)


@dataclass
class Holder:
    rules: list[Rule] = field(default_factory=lambda: [Dbl(), Neg()])
    one: list[Rule] = field(default_factory=lambda: [Dbl()])


class Registry:
    rules: list[Rule] = [Dbl(), Neg()]
    untyped = [Dbl(), Neg()]
"""

_OWN_RULES_IMPORT = (
    "from .configs import CaseConfig\n"
    "from .rules import (\n"
    "    PAIR, RULES, Dbl, Holder, Neg, Other, Registry, Rule, make, make_one, total,\n"
    "    use_argument, use_common_base, use_keyword,\n"
    ")\n"
)


def _own_rules_case(tmp_path: Path, setup: str) -> tuple[type, type, Path]:
    header = _HEADER.replace("from .configs import CaseConfig\n", _OWN_RULES_IMPORT, 1)
    return _write_case(tmp_path, setup, header=header, modules={"rules.py": _OWN_RULES})


@pytest.mark.parametrize(
    "setup",
    [
        'logger.info("%f", total(RULES) + total(PAIR))\n',
        """
        rules: list[Rule] = [Dbl(), Neg()]
        one: list[Rule] = [Dbl()]
        logger.info("%f", total(rules) + total(one))
        """,
        'logger.info("%f", total(make()) + total(make_one()))\n',
        'logger.info("%f", use_argument())\n',
        'logger.info("%f", use_keyword())\n',
        """
        holder = Holder()
        logger.info("%f", total(holder.rules) + total(holder.one))
        """,
        """
        logger.info("%f", total(Registry().rules) + total(Registry.untyped))
        """,
        'logger.info("%f", use_common_base())\n',
        """
        rules: list[Rule] = [
            Dbl(),  # doubled
            Neg(),
        ]
        logger.info("%f", total(rules))
        """,
    ],
    ids=[
        "module-level",
        "local",
        "return",
        "argument",
        "keyword-argument",
        "dataclass-factory",
        "class-attribute",
        "common-base",
        "multi-line",
    ],
)
def test_a_list_display_of_the_scenarios_own_subclasses_compiles(
    tmp_path: Path, setup: str
) -> None:
    # Codon types a list display by its first element and a List[Dbl] is no
    # List[Rule]: each of these was refused (`_acs_list(*items:
    # Tuple[Dbl,Neg])`, #480) until the check made the display a list of the
    # annotated element type, or of the nearest class the elements share.
    scenario, config, _ = _own_rules_case(tmp_path, setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


@pytest.mark.parametrize(
    ("setup", "needle", "message"),
    [
        (
            "rules: list[Rule] = [Dbl(), Other()]\n",
            "rules: list[Rule]",
            "'Other' does not match expected type 'Rule'",
        ),
        (
            "rules: list[Rule] = [Dbl(), Neg()]\nrules[0].flip()\n",
            "rules[0].flip()",
            "flip",
        ),
    ],
    ids=["local", "subclass-only-method"],
)
def test_a_list_display_of_own_classes_still_checks_its_elements(
    tmp_path: Path, setup: str, needle: str, message: str
) -> None:
    # The list is a list of the base: a class that is not a subclass is
    # refused, and so is a method only a subclass has.
    scenario, config, path = _own_rules_case(tmp_path, setup)
    error = _only_error(typecheck_scenario(scenario, config))
    assert message in error.message, error.format()
    assert "_acs_list" not in error.message, error.format()
    assert error.path == str(path), error.format()
    assert error.line == _line_of(path, needle), error.format()


def test_a_list_display_of_unrelated_classes_is_refused_with_a_hint(
    tmp_path: Path,
) -> None:
    # No class is shared (and nothing annotated): the refusal says so and
    # names the fix, without the helper that reported it.
    setup = 'mixed = [Dbl(), Other()]\nlogger.info("%d", len(mixed))\n'
    scenario, config, path = _own_rules_case(tmp_path, setup)
    error = _only_error(typecheck_scenario(scenario, config))
    assert "_acs_list" not in error.message, error.format()
    assert "list[Base]" in error.message, error.format()
    assert not any("_acs_list" in frame for frame in error.trace), error.format()
    assert error.path == str(path), error.format()
    assert error.line == _line_of(path, "mixed = [Dbl(), Other()]"), error.format()


_OWN_SHAPES = """
class Shape:
    def area(self) -> float:
        return 0.0


class Circle(Shape):
    def area(self) -> float:
        return 3.0


class Square(Shape):
    def area(self) -> float:
        return 1.0


class Other:
    def area(self) -> float:
        return 5.0


def total_list(shapes: list[Shape]) -> float:
    out = 0.0
    for shape in shapes:
        out += shape.area()
    return out


def total_dict(shapes: dict[str, Shape]) -> float:
    out = 0.0
    for shape in shapes.values():
        out += shape.area()
    return out


def more(shapes: list[Shape]) -> float:
    return total_list(shapes + [Square()])


def unknown() -> list[Shape]:
    return [Circle()]
"""


def _own_shapes_case(tmp_path: Path, setup: str) -> tuple[type, type, Path]:
    # The classes are defined in the scenario's module, above its class: the
    # check reads the bases of one module at a time.
    header = _HEADER.replace(
        "\n\nclass CaseScenario", "\n" + _OWN_SHAPES + "\n\nclass CaseScenario", 1
    )
    return _write_case(tmp_path, setup, header=header)


@pytest.mark.parametrize(
    "setup",
    [
        """
        d = {"a": Circle(), "b": Square()}
        logger.info("%f", d["a"].area() + d["b"].area())
        """,
        """
        d = {1: Square(), 2: Circle()}
        logger.info("%f", d[1].area() + total_dict({"x": d[2]}))
        """,
        """
        d: dict[str, Shape] = {"a": Circle()}
        logger.info("%f", total_dict(d))
        """,
        """
        d: dict[str, Shape] = {"a": Circle(), "b": Square()}
        logger.info("%f", total_dict(d))
        """,
        'logger.info("%f", total_dict({"a": Circle(), "b": Square()}))\n',
        'logger.info("%f", total_dict(shapes={"a": Circle(), "b": Square()}))\n',
        'logger.info("%f", total_dict({"a": Square()}))\n',
        """
        a: list[Shape] = [Circle()]
        b = a + [Square()]
        logger.info("%f", total_list(b))
        """,
        """
        a: list[Shape] = [Circle()]
        b = [Square(), Circle()] + a
        logger.info("%f", total_list(b))
        """,
        """
        a: list[Shape] = [Circle()]
        a += [Square(), Circle()]
        logger.info("%f", total_list(a))
        """,
        """
        a: list[Shape] = [Circle()]
        a.extend([Square(), Circle()])
        logger.info("%f", total_list(a))
        """,
        """
        a = [Circle(), Square()]
        a += [Square()]
        logger.info("%f", total_list(a) + more(a))
        """,
        'logger.info("%f", more([Circle()]))\n',
    ],
    ids=[
        "dict-common-base",
        "dict-int-keys",
        "dict-annotated-single",
        "dict-annotated",
        "dict-argument",
        "dict-keyword-argument",
        "dict-argument-single",
        "list-plus-display",
        "display-plus-list",
        "list-augmented",
        "list-extend",
        "inferred-list-augmented",
        "parameter-plus-display",
    ],
)
def test_a_dict_or_a_sum_of_the_scenarios_own_subclasses_compiles(
    tmp_path: Path, setup: str
) -> None:
    # #489: a dict display of subclasses, and a list display added to a
    # list[Base] (`a + [Sub()]`, `a += [Sub()]`, `a.extend([Sub()])`), was
    # typed by its first element: `'Square' does not match expected type
    # 'Circle'`, `unsupported operand type(s) for +: 'List[Shape]' and
    # 'List[Square]'`.
    scenario, config, _ = _own_shapes_case(tmp_path, setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


@pytest.mark.parametrize(
    "setup",
    [
        """
        rules: list[Rule] = [Low(), High()]
        logger.info("%f", total(rules))
        """,
        """
        rules = [Low(), High()]
        logger.info("%f", total(rules))
        """,
        'logger.info("%f", total([Low(), High()]))\n',
        """
        d = {"a": Low(), "b": High()}
        e: dict[str, Rule] = {"a": Low()}
        logger.info("%f", d["a"].weigh() + e["a"].weigh() + total_map({"a": Low()}))
        """,
        """
        rules: list[Rule] = [Low()]
        rules += [High()]
        logger.info("%f", total(rules))
        """,
    ],
    ids=["annotated", "common-base", "argument", "dicts", "augmented"],
)
def test_a_base_defined_below_the_scenario_class_is_named_in_a_body(
    tmp_path: Path, setup: str
) -> None:
    # #489: the annotation naming a class defined further down is dropped, and
    # the display was refused although it was annotated. A function body is
    # typed when it is first called, after the whole module, so the display
    # may name the class.
    extra = """


class Rule:
    def weigh(self) -> float:
        return 1.0


class Low(Rule):
    def weigh(self) -> float:
        return 0.5


class High(Rule):
    def weigh(self) -> float:
        return 2.0


def total(rules: list[Rule]) -> float:
    out = 0.0
    for rule in rules:
        out += rule.weigh()
    return out


def total_map(rules: dict[str, Rule]) -> float:
    return total(list(rules.values()))
"""
    scenario, config, _ = _write_case(tmp_path, setup, extra=extra)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


@pytest.mark.parametrize(
    ("setup", "needle", "message"),
    [
        (
            'd = {"a": Circle(), "b": Other()}\n',
            'd = {"a": Circle()',
            "'Other' does not match expected type 'Circle'",
        ),
        (
            'd: dict[str, Shape] = {"a": Circle(), "b": Other()}\n',
            "d: dict[str, Shape]",
            "'Other' does not match expected type 'Shape'",
        ),
        (
            'total_dict({"a": Circle(), "b": Other()})\n',
            "total_dict({",
            "'Other' does not match expected type 'Shape'",
        ),
        (
            "a: list[Shape] = [Circle()]\nb = a + [Other()]\n",
            "b = a + [Other()]",
            "'Other' does not match expected type 'Shape'",
        ),
    ],
    ids=["unrelated", "annotated", "argument", "list-plus-display"],
)
def test_a_dict_or_a_sum_with_a_non_subclass_is_still_refused(
    tmp_path: Path, setup: str, needle: str, message: str
) -> None:
    scenario, config, path = _own_shapes_case(tmp_path, setup)
    error = _only_error(typecheck_scenario(scenario, config))
    assert message in error.message, error.format()
    assert "_acs_dict_into" not in error.message, error.format()
    assert error.path == str(path), error.format()
    assert error.line == _line_of(path, needle), error.format()


@pytest.mark.parametrize(
    ("setup", "needle", "hint"),
    [
        (
            'k = "a"\nd = {k: Circle(), "b": Square()}\n',
            "d = {k:",
            "dict[str, Base]",
        ),
        (
            "b = unknown() + [Square()]\n",
            "b = unknown() + [Square()]",
            "list[Base]",
        ),
    ],
    ids=["dict-keys-not-literal", "sum-of-unknown-list"],
)
def test_a_display_the_check_cannot_type_is_refused_with_a_hint(
    tmp_path: Path, setup: str, needle: str, hint: str
) -> None:
    scenario, config, path = _own_shapes_case(tmp_path, setup)
    error = _only_error(typecheck_scenario(scenario, config))
    assert hint in error.message, error.format()
    assert error.path == str(path), error.format()
    assert error.line == _line_of(path, needle), error.format()


@pytest.mark.parametrize(
    ("setup", "needle"),
    [
        ('d = {1: Circle(), "b": Square()}\n', "d = {1: Circle()"),
        ("a: list[Shape] = [Circle()]\nb = [Other()] + a\n", "b = [Other()] + a"),
        ("a: list[Shape] = [Circle()]\na += [Other()]\n", "a += [Other()]"),
        ("a: list[Shape] = [Circle()]\na.extend([Other()])\n", "a.extend("),
        (
            "a: list[Shape] = [Circle()]\nfor a in [[Circle()]]:\n    b = a + [Square()]\n",
            "b = a + [Square()]",
        ),
    ],
    ids=[
        "mixed-keys",
        "display-plus-list",
        "augmented",
        "extend",
        "loop-variable",
    ],
)
def test_a_dict_or_a_sum_that_is_no_list_of_the_base_is_refused(
    tmp_path: Path, setup: str, needle: str
) -> None:
    # #489: the rewrite must not make a wrong value acceptable, or one of a
    # variable that merely shares the tracked name.
    scenario, config, path = _own_shapes_case(tmp_path, setup)
    result = typecheck_scenario(scenario, config)
    assert not result.ok
    assert any(
        d.path == str(path) and d.line == _line_of(path, needle)
        for d in result.diagnostics
    ), result.format()


def test_a_base_defined_below_a_class_level_display_is_named_in_the_hint(
    tmp_path: Path,
) -> None:
    # A class body runs at once: Codon cannot name a class defined further
    # down, whatever the annotation (Python reads the factory's lambda later),
    # so the hint says where to define it rather than telling the author to
    # annotate what is annotated.
    extra = """


@dataclass
class Holder:
    rules: list[Rule] = field(default_factory=lambda: [Low(), High()])


class Rule: ...


class Low(Rule): ...


class High(Rule): ...
"""
    header = _HEADER.replace(
        "import logging\n",
        "import logging\nfrom dataclasses import dataclass, field\n",
        1,
    )
    scenario, config, path = _write_case(tmp_path, "pass\n", extra=extra, header=header)
    error = _only_error(typecheck_scenario(scenario, config))
    assert "define `Rule` above" in error.message, error.format()
    assert "annotate the target" not in error.message, error.format()
    assert error.path == str(path), error.format()


@pytest.mark.parametrize(
    ("setup", "needle"),
    [
        (
            'ActionStateCondition("turn", ActionState.COMPLETE, {"turn": turn, "n": 1})\n',
            "ActionStateCondition(",
        ),
        (
            'actions: dict[str, BaseAction] = {"turn": turn, "n": 1}\n',
            "actions: dict",
        ),
    ],
    ids=["argument", "annotated"],
)
def test_a_dict_display_mixing_an_action_and_a_non_action_is_refused(
    tmp_path: Path, setup: str, needle: str
) -> None:
    # The rewritten display is still a dict of actions: a value that is no
    # action is refused at the display's line, as Python's signature would.
    scenario, config, path = _write_case(tmp_path, _MIXED_DICT_IMPORTS + setup)
    error = _only_error(typecheck_scenario(scenario, config))
    assert (
        "'int' does not match expected type 'BaseAction'" in error.message
    ), error.format()
    assert error.path == str(path), error.format()
    assert error.line == _line_of(path, needle), error.format()


@pytest.mark.parametrize(
    "setup",
    [
        'speeds = {"a": 1, "b": cfg.timeout_seconds}\nv: float = speeds["b"]\n',
        'limits = {"a": None, "b": cfg.timeout_seconds}\n',
        'limits = {"a": cfg.timeout_seconds, "b": None}\n',
        'names = {"a": cfg.name, "b": None}\n',
        'limits: dict[str, float | None] = {"a": 1.0, "b": cfg.min_speed_kmh}\n',
    ],
    ids=["int-then-float", "none-first", "none-last", "str-none", "optional"],
)
def test_a_dict_display_codon_typed_before_still_compiles(
    tmp_path: Path, setup: str
) -> None:
    # The rewrite only makes a condition (or an action) its base class: an int
    # and a float, or a value and None, stay Codon's display to type, which
    # promotes the int and makes the value optional (#250).
    scenario, config, _ = _write_case(tmp_path, setup + 'logger.info("ok")\n')
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


@pytest.mark.parametrize(
    ("setup", "needle", "message"),
    [
        (
            "from autoware_carla_scenario.route import parse_route_search\n",
            "parse_route_search",
            "autoware_carla_scenario.route.parse_route_search is not part of the "
            "Codon model, on purpose: the route search",
        ),
        (
            "from autoware_carla_scenario import ScenarioRunner\n",
            "ScenarioRunner",
            "autoware_carla_scenario.ScenarioRunner is not part of the Codon "
            "model, on purpose: the runner",
        ),
        (
            "from autoware_carla_scenario.actions.base import ActionState\n",
            "actions.base",
            "autoware_carla_scenario.actions.base is not part of the Codon model; "
            "import ActionState from autoware_carla_scenario.action_state instead",
        ),
        (
            "from autoware_carla_scenario.conditions.comparison import compare\n",
            "conditions.comparison",
            "autoware_carla_scenario.conditions.comparison is not part of the "
            "Codon model; import compare from autoware_carla_scenario.conditions "
            "instead",
        ),
        (
            "from autoware_carla_scenario.registry import get_scenario_registry\n",
            "registry",
            "autoware_carla_scenario.registry is not part of the Codon model, and "
            "no module of the model declares get_scenario_registry as Python "
            "exports it; autoware_carla_scenario.get_scenario_registry is left "
            "out on purpose: the scenario registry",
        ),
    ],
    ids=["route-search", "runner", "action-state", "compare", "registry"],
)
def test_an_import_of_what_the_model_leaves_out_says_why(
    tmp_path: Path, setup: str, needle: str, message: str
) -> None:
    # Refused before compiling, with the reason (NOT_MODELLED) or the module
    # of the model that declares the name as Python exports it, never a
    # module Python cannot import it from (#237).
    scenario, config, path = _write_case(
        tmp_path, setup + textwrap.dedent(_VALID_SETUP)
    )
    result = typecheck_scenario(scenario, config)
    error = _only_error(result)
    assert len(result.diagnostics) == 1, result.format()
    assert message in error.message, error.format()
    assert error.path == str(path)
    assert error.line == _line_of(path, needle)


# ---------------------------------------------------------------------------
# An Optional of a class with subclasses (#263)
# ---------------------------------------------------------------------------

_NO_CONDITION_IMPORTS = """
from autoware_carla_scenario import BaseAction
ind = TurnIndicatorAction(EGO_ROLE_NAME, TurnIndicatorState.LEFT)
"""


@pytest.mark.parametrize(
    "setup",
    [
        """
        actions: dict[str, BaseAction] = {}
        actions["i"] = ind
        for key in actions:
            logger.info("%s %s", key, str(actions[key].done))
        """,
        """
        actions: list[BaseAction] = []
        actions.append(ind)
        for action in actions:
            logger.info("%s", action.label)
        """,
        """
        logger.info("%s", str(ind.done))
        """,
        """
        self.register_pre_tick(ind)
        action: BaseAction = ind
        logger.info("%s", str(action.done))
        """,
        """
        pending: BaseAction | None = None
        if pending is None:
            logger.info("none")
        """,
    ],
    ids=["dict", "list", "action", "registered", "optional-action"],
)
def test_actions_compile_in_a_scenario_that_builds_no_condition(
    tmp_path: Path, setup: str
) -> None:
    # Each crashed Codon 0.19's code generation ("LLVM: error: value doesn't
    # match function result type 'ptr'", in opt_ref_new_rtti of BaseCondition)
    # while no condition was built: an action's `condition = None` is an
    # Optional[BaseCondition] None, and registering the actions made no
    # difference. The model builds a BaseCondition and a BaseAction once.
    scenario, config, _ = _write_case(
        tmp_path, _NO_CONDITION_IMPORTS + textwrap.dedent(setup)
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_an_optional_of_every_model_class_with_subclasses_compiles(
    tmp_path: Path,
) -> None:
    # The model's own classes with subclasses, each held as an Optional None
    # in a program that builds nothing (#263), with every subclass imported:
    # Codon crashes only once a subclass is defined, and some live in another
    # module (AutowareTrafficLight, MGRSProjector). VehicleEntityConfig is
    # left out: the scenario check builds an EgoConfig, its subclass, for
    # every scenario, and building one in the model ahead of EgoConfig would
    # make the subclass the crash. So is Ros2EgoTransport: the scenario check
    # has no ROS 2 transport (_ros2.codon), and its constructors are
    # compile_errors, so the model cannot build one.
    from autoware_carla_scenario.typecheck.check import (
        _base_workspace,
        find_supported_codon,
        model_dir,
    )
    from autoware_carla_scenario.typecheck.toolchain import (
        codon_environment,
        codon_path_dir,
    )

    defined: dict[str, str] = {}
    subclasses: dict[str, list[tuple[str, str]]] = {}
    for path in sorted(model_dir().rglob("*.codon")):
        parts = list(path.relative_to(model_dir()).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.ClassDef) and not node.decorator_list:
                defined.setdefault(node.name, ".".join(parts))
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        subclasses.setdefault(base.id, []).append(
                            (".".join(parts), node.name)
                        )
    classes = sorted(
        (defined[name], name)
        for name in subclasses.keys() - {"VehicleEntityConfig", "Ros2EgoTransport"}
        if name in defined
    )
    assert ("autoware_carla_scenario.conditions", "BaseCondition") in classes
    assert ("autoware_carla_scenario.actions", "BaseAction") in classes
    assert ("lanelet2.core", "TrafficLight") in classes
    program = [f"from {m} import {n} as _C{i}" for i, (m, n) in enumerate(classes)]
    program += [
        f"from {m} import {n} as _S{i}"
        for i, (m, n) in enumerate(s for _, c in classes for s in subclasses[c])
    ]
    for i in range(len(classes)):
        program += [f"_x{i}: Optional[_C{i}] = None", f"print(_x{i} is None)"]
    toolchain = find_supported_codon()
    _base_workspace(tmp_path, None, codon_path_dir())
    (tmp_path / "acs_optionals.py").write_text("\n".join(program) + "\n")
    proc = subprocess.run(  # noqa: S603 - a fixed compiler invocation
        [
            str(toolchain.executable),
            "build",
            "-llvm",
            "-o",
            os.devnull,
            "acs_optionals.py",
        ],
        cwd=tmp_path,
        env=codon_environment(toolchain, tmp_path),
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


_UNBUILT_RESULT = """


class MyResult(ScenarioResult):
    pass
"""


def test_an_optional_codon_cannot_generate_is_named_at_the_class(
    tmp_path: Path,
) -> None:
    # A subclass of ScenarioResult nothing builds, defined after the model
    # compiled ScenarioResult: Codon 0.19 checks `ScenarioResult | None`,
    # then generates IR that LLVM refuses. The check names the class and the
    # workaround at the subclass's line, never Codon's bare LLVM error, and
    # keeps no verdict of it (a crash).
    setup = 'result: ScenarioResult | None = None\nif result is None:\n    logger.info("none")\n'
    scenario, config, path = _write_case(tmp_path, setup, _UNBUILT_RESULT)
    result = typecheck_scenario(scenario, config)
    error = _only_error(result)
    assert result.crashed, result.format()
    assert (
        "Codon 0.19 generated code it cannot compile for "
        "autoware_carla_scenario.conditions.ScenarioResult" in error.message
    ), error.format()
    assert "Build a MyResult where setup() reaches it" in error.message
    assert error.path == str(path), error.format()
    assert error.line == _line_of(path, "class MyResult"), error.format()
    assert error.trace and error.trace[0].startswith("LLVM: error:"), error.format()

    # The workaround compiles.
    built = setup + 'logger.info("%s", MyResult(True, "", 0.0).message)\n'
    scenario, config, _ = _write_case(tmp_path, built, _UNBUILT_RESULT)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


# ---------------------------------------------------------------------------
# An empty sequence (#264), and the members the model declares (#265)
# ---------------------------------------------------------------------------

_SEQUENCE_HEADER = _HEADER.replace(
    "from .configs import CaseConfig",
    """from autoware_carla_scenario import (
    BridgePose,
    EgoVehicle,
    MapManager,
    OrCondition,
    RoutingAction,
    TrafficSignalAction,
    TrafficSourceAction,
    VehicleEntity,
)
from autoware_carla_scenario.odd import OddDefinition, OddModule, all_of, any_of
from autoware_carla_scenario.route import scenario_route, scenario_route_frame

from .configs import CaseConfig""",
)

_GOAL = "Lanelet2Pose(lanelet_id=1, s=0.0)"


@pytest.mark.parametrize(
    "setup",
    [
        # RoutingAction's own default, `waypoints=()`.
        f"self.register_pre_tick(RoutingAction(EGO_ROLE_NAME, {_GOAL}))\n",
        f"self.register_pre_tick(RoutingAction(EGO_ROLE_NAME, {_GOAL}, waypoints=[]))\n",
        "self.waypoint_poses = []\n",
        f"""
        ego = self.ego_entity
        if ego is not None:
            ego.route_to(self.world, {_GOAL}, waypoints=[])
            ego.route_to(self.world, {_GOAL}, waypoints=[Lanelet2Pose(lanelet_id=2, s=0.0)],
                         ground_projection=GroundProjectionConfig())
        """,
        # Every other Sequence[X] the model checks the elements of.
        """
        self.derive_goal_from_route(())
        self.derive_waypoints_from_route([])
        self.register_pre_tick(TrafficSignalAction(0, ()))
        self.register_fail_condition(OrCondition((), label="never"))
        self.register_fail_condition(AndCondition([], label="none"))
        self.register_fail_condition(TemporaryStopCondition(EGO_ROLE_NAME, [], label="stops"))
        logger.info("%d", len(Trajectory("none", []).vertices))
        self.register_cover("speed", lambda world: 1.0, buckets=[])
        odd = OddDefinition("empty", [], [], roots=[])
        module = OddModule("m", include_and=[], labels=())
        logger.info("%s %s %s", odd.name, module.name, str(all_of([]) is any_of(())))
        """,
    ],
    ids=[
        "routing-default",
        "routing-empty-list",
        "waypoint-poses",
        "route-to",
        "others",
    ],
)
def test_an_empty_sequence_compiles(tmp_path: Path, setup: str) -> None:
    # An empty display has no element type in Codon (`[]` is a
    # List[NoneType], `()` a Tuple of no items), and each was refused where
    # Python takes a Sequence[X], RoutingAction's own default first (#264).
    scenario, config, _ = _write_case(
        tmp_path,
        "self._setup_ego_spawn()\n" + textwrap.dedent(setup) + _TIMEOUT_SETUP,
        header=_SEQUENCE_HEADER,
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


@pytest.mark.parametrize(
    ("setup", "message"),
    [
        (
            f"ego.route_to(self.world, {_GOAL}, waypoints=[460, 265])\n",
            "'int' does not match expected type 'Lanelet2Pose'",
        ),
        (
            f"ego.route_to(self.world, {_GOAL}, ground_projection=5)\n",
            "expected a GroundProjectionConfig or None",
        ),
        (
            f"self.register_pre_tick(RoutingAction(EGO_ROLE_NAME, {_GOAL}, waypoints=[None]))\n",
            "'NoneType' does not match expected type 'Lanelet2Pose'",
        ),
        (
            'self.derive_goal_from_route(["460"])\n',
            "expected a sequence of ints",
        ),
    ],
    ids=["route-to-ints", "route-to-projection", "a-list-of-none", "route-of-str"],
)
def test_a_sequence_of_another_type_is_refused_at_its_line(
    tmp_path: Path, setup: str, message: str
) -> None:
    # route_to() checked neither argument (#264); an empty display is
    # accepted, a display of something else still is not.
    scenario, config, path = _write_case(
        tmp_path,
        "ego = self.ego_entity\nif ego is not None:\n    " + setup + _TIMEOUT_SETUP,
        header=_SEQUENCE_HEADER,
    )
    error = _only_error(typecheck_scenario(scenario, config))
    assert message in error.message, error.format()
    assert error.path == str(path)
    assert error.line == _line_of(path, setup.split("(")[0].strip())


_MEMBERS_SETUP = """
self._setup_ego_spawn()
routing = RoutingAction(EGO_ROLE_NAME, Lanelet2Pose(lanelet_id=1, s=0.0))
for waypoint in routing.waypoints:
    logger.info("%d", waypoint.lanelet_id)
self.register_pre_tick(routing)
measures = self.measures
for key in sorted(measures):
    logger.info("%s %s", key, measures[key].unit)
origin = MapManager.get_instance().geo_origin
pose = BridgePose.from_yaw(1.0, 2.0, 0.0, 0.5)
turned = BridgePose.from_rpy(1.0, 2.0, 0.0, 0.0, 0.0, 0.5)
logger.info("%f", origin[0] + pose.position.x + turned.position.y)
if TrajectoryTimeCondition.IS_TRAJECTORY_TIME:
    logger.info("a trajectory's clock")
route = scenario_route()
if route is not None:
    frame = scenario_route_frame()
    lanelet_id, along = frame.locate(route.length / 2.0)
    x, y, heading = frame.point(frame.length)
    logger.info("%d %f %f %f %f %d", lanelet_id, along, x, y, heading, frame.lanelet_at(0.0).id)
"""

_MEMBERS_EXTRA = """

    def _drive(self, npc: VehicleEntity, world: carla.World, source: TrafficSourceAction) -> None:
        npc.set_desired_speed(world, 20.0)
        npc.change_lane(world, LaneChangeDirection.LEFT)
        if npc.lane_change_finished(world):
            npc.forget_lane_change()
        npc.turn_at_junction(world, TurnDirection.LEFT)
        npc.turn_at_junction(world, TurnDirection.RIGHT, search_distance=50.0)
        npc.release_from_traffic(world)
        npc.set_client(self.client, tm_port=8000)
        logger.info("%d", source.spawned)
        ego = self.ego_entity
        if ego is not None:
            ego.set_desired_speed(world, 10.0)
            ego.change_lane(world, LaneChangeDirection.RIGHT)
"""


def test_the_members_the_model_now_declares_compile(tmp_path: Path) -> None:
    # Each was refused with Codon's bare "has no attribute" though Python
    # defines it (#265): BaseScenario.measures, RoutingAction.waypoints,
    # the manoeuvres VehicleEntity and EgoVehicle inherit from BackendDriven.
    scenario, config, _ = _write_case(
        tmp_path,
        _MEMBERS_SETUP + _TIMEOUT_SETUP,
        _MEMBERS_EXTRA,
        header=_SEQUENCE_HEADER,
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


_OWN_EGO = """


class Ego(EgoVehicle):
    def set_desired_speed(self, world: carla.World, speed_kmh: float) -> None:
        pass

    def lane_change_finished(self, world: carla.World) -> {returns}:
        return {value}
"""


@pytest.mark.parametrize(
    ("returns", "value", "message"),
    [
        ("bool", "True", None),
        ("str", '"yes"', "EgoVehicle.lane_change_finished() returns bool"),
    ],
    ids=["bool", "str"],
)
def test_an_ego_s_own_manoeuvre_is_checked_as_a_hook(
    tmp_path: Path, returns: str, value: str, message: str | None
) -> None:
    # A manoeuvre the model now declares on EgoVehicle is a hook of it: an
    # override compiles, and one returning another type is refused (#238).
    scenario, config, path = _write_case(
        tmp_path,
        "self._setup_ego_spawn()\n" + _TIMEOUT_SETUP,
        _OWN_EGO.format(returns=returns, value=value),
        header=_SEQUENCE_HEADER,
    )
    result = typecheck_scenario(scenario, config)
    if message is None:
        assert result.ok, result.format()
        return
    error = _only_error(result)
    assert message in error.message, error.format()
    assert error.path == str(path)
    assert error.line == _line_of(path, "def lane_change_finished")


_OWN_TURN = """
{caller}

class Npc(VehicleEntity):
    def turn_at_junction(self, world: carla.World, direction: TurnDirection, **kwargs) -> None:
        pass
"""

_TURN_CALLER = """
    def _drive(self, npc: VehicleEntity, world: carla.World) -> None:
        npc.turn_at_junction(world, TurnDirection.LEFT)
"""


@pytest.mark.parametrize("calls", [False, True], ids=["defined", "called"])
def test_an_own_turn_at_junction_compiles_until_it_is_called(
    tmp_path: Path, calls: bool
) -> None:
    # The limitation docs/typecheck.md states: an override of
    # turn_at_junction(**kwargs) compiles, but Codon 0.19's dispatch thunk
    # cannot call it. Declaring the method on the base (#265) must not make
    # a scenario that only defines it fail; a Codon that lifts the limit
    # fails the second case, and the docs need the change.
    scenario, config, path = _write_case(
        tmp_path,
        "self._setup_ego_spawn()\n" + _TIMEOUT_SETUP,
        _OWN_TURN.format(caller=_TURN_CALLER if calls else ""),
        header=_SEQUENCE_HEADER,
    )
    result = typecheck_scenario(scenario, config)
    if not calls:
        assert result.ok, result.format()
        return
    error = _only_error(result)
    assert "cannot typecheck" in error.message, error.format()
    assert error.line == _line_of(path, "npc.turn_at_junction(")


@pytest.mark.parametrize(
    ("setup", "needle", "message"),
    [
        (
            "backend = self.traffic_backend\n",
            "self.traffic_backend",
            "BaseScenario.traffic_backend is not part of the Codon model, on "
            "purpose: takes or returns the run's TrafficBackend",
        ),
        (
            'logger.info("%d", len(TimeoutCondition(1.0, label="t").get_details()))\n',
            "get_details()",
            "BaseCondition.get_details is not part of the Codon model, on purpose",
        ),
        (
            # An instance attribute (#307): a class held as a value.
            'logger.info("%s", self.ego_type.requires_goal)\n',
            "self.ego_type",
            "BaseScenario.ego_type is not part of the Codon model, on purpose: "
            "the EgoVehicle subclass",
        ),
    ],
    ids=["traffic-backend", "condition-details", "ego-type"],
)
def test_a_member_the_model_leaves_out_says_why(
    tmp_path: Path, setup: str, needle: str, message: str
) -> None:
    # Codon's "has no attribute", with the reason NOT_MODELLED_MEMBERS gives
    # (#265), as an import of a name left out is refused with its reason.
    scenario, config, path = _write_case(
        tmp_path, "self._setup_ego_spawn()\n" + setup + _TIMEOUT_SETUP
    )
    error = _only_error(typecheck_scenario(scenario, config))
    assert "object has no attribute" in error.message, error.format()
    assert message in error.message, error.format()
    assert error.path == str(path)
    assert error.line == _line_of(path, needle)


def _framework_enums() -> dict[str, tuple[str, type[enum.Enum]]]:
    """Every enum the package exports, and the two a scenario imports from a
    submodule: name -> (module, the enum)."""
    import autoware_carla_scenario as acs
    from autoware_carla_scenario.action_state import ActionState
    from autoware_carla_scenario.coverage import CoverGroup

    out = {
        name: ("autoware_carla_scenario", getattr(acs, name))
        for name in sorted(set(acs.__all__) | set(acs._LAZY_IMPORTS))
        if isinstance(getattr(acs, name, None), type)
        and issubclass(getattr(acs, name), enum.Enum)
    }
    out["ActionState"] = ("autoware_carla_scenario.action_state", ActionState)
    out["CoverGroup"] = ("autoware_carla_scenario.coverage", CoverGroup)
    return out


def _enum_reads() -> tuple[str, str]:
    """(imports, setup): each framework enum member's name and value read,
    and the enums' own methods and properties (#307)."""
    enums = _framework_enums()
    # Those _HEADER does not import already: Codon 0.19 cannot resolve a
    # name imported twice once it is used (#311).
    imports = "".join(
        f"from {module} import {name}\n"
        for name, (module, _) in enums.items()
        if f"    {name},\n" not in _HEADER
    )
    lines = []
    for name, (_module, python) in enums.items():
        for member in python:
            lines.append(f"names.append({name}.{member.name}.name)")
        first = next(iter(python))
        kind = type(first.value).__name__
        lines.append(f"{name.lower()}_value: {kind} = {name}.{first.name}.value")
    return imports, "names: list[str] = []\n" + "\n".join(lines) + "\n"


_INSTANCE_READS = """
self._setup_ego_spawn()
timeout = TimeoutCondition(cfg.timeout_seconds, label="timeout")
logger.info("limit %f", timeout.timeout_seconds)
longer = LongerTimeout(5.0, label="longer")
logger.info("longer %f", longer.timeout_seconds)
logger.info("ego requires goal: %s", self.ego_requires_goal)
logger.info("%s", ComparisonRule.LESS_THAN.text)
logger.info("%s", LaneChangeDirection.LEFT.to_carla_bool())
logger.info("%s", CollisionTargetType.VEHICLE.matches("vehicle.tesla.model3"))
logger.info("%s", CollisionTargetType.ANY.matches(None))
frame = scenario_route_frame()
first = frame.lanelets[0]
layer = frame.map.laneletLayer
graph = frame.graph
logger.info("%f %d %d", frame.match.length, len(frame.lanelets), first.id)
"""

# A subclass's __init__ may assign an attribute the model reads as a
# property (#307): the assignment declares the subclass's own attribute.
_LONGER_TIMEOUT = """

class LongerTimeout(TimeoutCondition):
    def __init__(self, timeout_seconds: float, label: str) -> None:
        super().__init__(timeout_seconds, label=label)
        self.timeout_seconds = timeout_seconds * 2.0
"""


def test_a_scenario_reads_instance_attributes_and_enum_members(
    tmp_path: Path,
) -> None:
    # Instance attributes (TimeoutCondition.timeout_seconds, RouteFrame's)
    # and every framework enum member's name, value and own methods compile
    # (#307), as they run in Python.
    imports, enum_reads = _enum_reads()
    header = _HEADER.replace(
        "import logging\n",
        "import logging\n"
        "from autoware_carla_scenario.route import scenario_route_frame\n" + imports,
        1,
    )
    scenario, config, _ = _write_case(
        tmp_path,
        _INSTANCE_READS + enum_reads + _TIMEOUT_SETUP,
        _LONGER_TIMEOUT,
        header=header,
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


_ROUTE_HEADER = _HEADER.replace(
    "import logging\n",
    "import logging\nfrom autoware_carla_scenario.route import scenario_route\n",
    1,
)

_FIELD_READS = """
self._setup_ego_spawn()
path = Trajectory(
    "p",
    [
        TrajectoryVertex(Lanelet2Pose(460, 1.0)),
        TrajectoryVertex(MapPose(10.0, 2.0)),
        TrajectoryVertex(Lanelet2Pose(460, 9.0)),
    ],
)
gated = path.gated({-1: StickyCondition(ElapsedTimeCondition(2.0, label="go"))})
position = gated.vertices[0].position
if isinstance(position, Lanelet2Pose):
    logger.info("starts on a lanelet")
# A vertex's position is a TrajectoryPosition, which a vertex takes.
again = Trajectory("q", [TrajectoryVertex(v.position) for v in gated.vertices])
match = scenario_route()
if match is not None:
    logger.info(
        "%d %f %f %d", match.lanelet_ids[0], match.start_s, match.end_s, match.index
    )
"""


def test_a_scenario_reads_a_frozen_dataclass_s_fields(tmp_path: Path) -> None:
    # Public dataclass fields are in the member guarantee (#298): a vertex's
    # position (a union of ten poses) and the running route's lanelets, read
    # as properties of the model.
    scenario, config, _ = _write_case(
        tmp_path, _FIELD_READS + _TIMEOUT_SETUP, header=_ROUTE_HEADER
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


_SEGMENT_READS = """
self._setup_ego_spawn()
match = scenario_route()
if match is not None:
    for segment in match.segments:
        logger.info("%s %f %f %f", segment.kind, segment.start, segment.end, segment.length)
        logger.info("%s %d", segment.turn, len(segment.lanelet_ids))
    for junction in match.junctions:
        logger.info("%s %d", junction.turn, junction.lanelet_ids[0])
    if len(match.junctions) > 0:
        first = match.junction(0)
        logger.info("%s %f %f", first.turn, first.start, first.end)
"""


def test_a_scenario_reads_the_route_s_segments_and_junctions(tmp_path: Path) -> None:
    # RouteMatch.segments, junctions and junction() and RouteSegmentMatch were
    # left out for a stale "variadic tuples have no Codon type" (#380).
    scenario, config, _ = _write_case(
        tmp_path, _SEGMENT_READS + _TIMEOUT_SETUP, header=_ROUTE_HEADER
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_a_scenario_misreading_the_route_s_segments_is_refused(
    tmp_path: Path,
) -> None:
    # The segment fields are typed as Python types them: a length is a float
    # and junction() takes an int index.
    for number, (read, fragment) in enumerate(
        (
            ('logger.info("%s", match.segments[0].length + "x")', "match.segments[0]"),
            ('logger.info("%s", match.junction("0").turn)', 'match.junction("0")'),
            ("count = match.junctions[0].turn + 1", "match.junctions[0].turn"),
        )
    ):
        setup = "self._setup_ego_spawn()\nmatch = scenario_route()\n"
        setup += "if match is not None:\n    " + read + "\n"
        case = tmp_path / str(number)
        case.mkdir()
        scenario, config, path = _write_case(
            case,
            setup + _TIMEOUT_SETUP,
            header=_ROUTE_HEADER,
        )
        error = _only_error(typecheck_scenario(scenario, config))
        assert error.line == _line_of(path, fragment), error.format()


def test_a_dataclass_field_the_model_leaves_out_says_why(tmp_path: Path) -> None:
    setup = _FIELD_READS.replace("match.lanelet_ids[0],", "len(match.to_config()),", 1)
    scenario, config, path = _write_case(
        tmp_path, setup + _TIMEOUT_SETUP, header=_ROUTE_HEADER
    )
    error = _only_error(typecheck_scenario(scenario, config))
    assert "object has no attribute" in error.message, error.format()
    assert (
        "RouteMatch.to_config is not part of the Codon model, on purpose"
        in error.message
    ), error.format()
    assert error.line == _line_of(path, "match.to_config")


_CROSS_SETUP = """
self.register_cover("speed", lambda world: 3.0, range=(0.0, 30.0), every=5.0)
self.register_cover("gap", lambda world: 4.0, range=(0.0, 30.0), every=5.0)
"""


@pytest.mark.parametrize(
    ("items", "ok"),
    [
        ('"speed"', False),
        ('["speed", "gap"]', True),
        ('("speed", "gap")', True),
        ('("speed",)', True),
    ],
    ids=["str", "list", "tuple", "one-tuple"],
)
def test_a_str_for_the_items_of_a_cross_is_refused(
    tmp_path: Path, items: str, ok: bool
) -> None:
    # register_cross() takes a Sequence[str] of cover items' names, and
    # Python reads a str's characters as names (#299); a list or a tuple of
    # them compiles.
    setup = _CROSS_SETUP + f'self.register_cross("speed_x", {items})\n'
    scenario, config, path = _write_case(tmp_path, setup + _TIMEOUT_SETUP)
    result = typecheck_scenario(scenario, config)
    if ok:
        assert result.ok, result.format()
        return
    error = _only_error(result)
    assert (
        "BaseScenario.register_cross(): items: expected a sequence of names "
        "(a list or a tuple of str), not a str: wrap it in a list"
    ) in error.message, error.format()
    assert error.line == _line_of(path, "register_cross(")


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


# ---------------------------------------------------------------------------
# An Optional item of a list display (#308)
# ---------------------------------------------------------------------------

_ITEM_READERS = """

    def _speed(self, world: carla.World) -> float:
        return 3.0

    def _maybe_name(self) -> str | None:
        return None
"""


@pytest.mark.parametrize(
    ("setup", "needle", "message"),
    [
        (
            'self.register_cover("speed", self._speed, '
            "buckets=[0.0, 10.0, cfg.min_speed_kmh])\n",
            "buckets=[0.0, 10.0, cfg.min_speed_kmh]",
            'List item 2 has incompatible type "float | None"; expected "float"',
        ),
        (
            'self.register_cover("speed", self._speed, range=(0.0, 100.0), every=10.0)\n'
            'self.register_cross("c", ["speed", self._maybe_name()])\n',
            "self.register_cross(",
            'List item 1 has incompatible type "str | None"; expected "str"',
        ),
    ],
    ids=["cover-buckets", "cross-names"],
)
def test_an_optional_item_of_a_list_display_is_refused(
    tmp_path: Path, setup: str, needle: str, message: str
) -> None:
    # Codon unwraps the element where the model iterates the sequence, so
    # these compiled, and Python failed on the None (#308).
    scenario, config, path = _write_case(
        tmp_path, _TIMEOUT_SETUP + setup, _ITEM_READERS
    )
    result = typecheck_scenario(scenario, config)
    error = _only_error(result)
    assert len(result.diagnostics) == 1, result.format()
    assert message in error.message, error.format()
    assert "test the value for None" in error.message
    assert error.line == _line_of(path, needle)


def test_an_item_with_a_default_for_none_compiles(tmp_path: Path) -> None:
    setup = (
        _TIMEOUT_SETUP
        + 'self.register_cover("speed", self._speed, '
        + "buckets=[0.0, 10.0, cfg.min_speed_kmh or 20.0])\n"
        + 'self.register_cross("c", ["speed", self._maybe_name() or "speed"])\n'
    )
    scenario, config, _ = _write_case(tmp_path, setup, _ITEM_READERS)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


# ---------------------------------------------------------------------------
# A framework union used as one of its members (#309)
# ---------------------------------------------------------------------------

_LANELET_OF = """

    def _lanelet_of(self, pose: Lanelet2Pose) -> int:
        return pose.lanelet_id
"""


@pytest.mark.parametrize(
    ("use", "message"),
    [
        (
            'logger.info("%d", v.position.lanelet_id)\n',
            'of "CarlaWorldPose | Lanelet2Pose | OpenDrivePose | MapPose | '
            "RelativeLanePose | RouteLanePose | RouteOppositePose | "
            'RouteCrossingPose | RouteCrosswalkPose | RouteRoadsidePose" have no '
            'attribute "lanelet_id"',
        ),
        (
            'logger.info("%d", self._lanelet_of(v.position))\n',
            'Argument 1 to "_lanelet_of" of "CaseScenario" has incompatible type',
        ),
    ],
    ids=["attribute", "argument"],
)
def test_a_trajectory_position_used_as_one_pose_is_refused(
    tmp_path: Path, use: str, message: str
) -> None:
    # Codon holds TrajectoryVertex.position as a Union and converts it to
    # the member a use needs at run time, so these compiled, and Python
    # raised AttributeError on the MapPose (#309).
    setup = (
        _TIMEOUT_SETUP
        + 'traj = Trajectory("t", [TrajectoryVertex(MapPose(1.0, 2.0))])\n'
        + "for v in traj.vertices:\n"
        + "    "
        + use
    )
    scenario, config, path = _write_case(tmp_path, setup, _LANELET_OF)
    result = typecheck_scenario(scenario, config)
    error = _only_error(result)
    # mypy reports a missing attribute once per member lacking it: one here.
    assert len(result.diagnostics) == 1, result.format()
    assert message in error.message, error.format()
    assert "(TrajectoryPosition)" in error.message, error.format()
    assert "narrow it with isinstance first" in error.message, error.format()
    assert error.line == _line_of(path, use.strip())


def test_a_trajectory_position_narrowed_or_passed_on_compiles(tmp_path: Path) -> None:
    setup = (
        _TIMEOUT_SETUP
        + 'traj = Trajectory("t", [TrajectoryVertex(MapPose(1.0, 2.0))])\n'
        + "moved: list[TrajectoryVertex] = []\n"
        + "for v in traj.vertices:\n"
        + "    p = v.position\n"
        + "    if isinstance(p, Lanelet2Pose):\n"
        + '        logger.info("%d", self._lanelet_of(p) + p.lanelet_id)\n'
        + "    moved.append(TrajectoryVertex(v.position, advance=v.advance))\n"
    )
    scenario, config, _ = _write_case(tmp_path, setup, _LANELET_OF)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


# ---------------------------------------------------------------------------
# A module of the scenario's package imported as a module (#310)
# ---------------------------------------------------------------------------

_HELPERS = """
from __future__ import annotations

import typesafe_carla.carla as carla


def speed_kph(world: carla.World) -> float | None:
    return 3.0
"""

_HELPERS_SETUP = (
    _TIMEOUT_SETUP
    + 'self.register_cover("speed", helpers.speed_kph, range=(0.0, 100.0), every=10.0)\n'
)


@pytest.mark.parametrize(
    ("statement", "where"),
    [
        ("from . import helpers", "helpers.py"),
        ("from {package} import helpers", "helpers.py"),
        ("import {package}.helpers as helpers", "helpers.py"),
        ("from .sub import helpers", "sub/helpers.py"),
        ("from . import configs as _configs, helpers", "helpers.py"),
    ],
    ids=["relative", "absolute", "import-as", "subpackage", "with-another"],
)
def test_a_module_of_the_package_imported_as_a_module_compiles(
    tmp_path: Path, statement: str, where: str
) -> None:
    # Codon 0.19 resolves `from X import name` only as a name of X/__init__
    # and `import a.b` only from the importing file's directory, so these
    # were refused; the check rewrites them into `import helpers as helpers`
    # (#310).
    modules = {where: _HELPERS}
    if where.startswith("sub/"):
        modules["sub/__init__.py"] = ""
    header = _HEADER.replace(
        "from .configs import CaseConfig\n",
        f"from .configs import CaseConfig\n{statement}\n",
        1,
    )
    scenario, config, _ = _write_case(
        tmp_path, _HELPERS_SETUP, header=header, modules=modules
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_a_module_of_the_package_imported_without_a_name_is_refused(
    tmp_path: Path,
) -> None:
    # `import pkg.helpers` binds pkg, which Codon cannot resolve from inside
    # it: refused at its line, naming the form that works.
    header = _HEADER.replace(
        "from .configs import CaseConfig\n",
        "from .configs import CaseConfig\nimport {package}.helpers\n",
        1,
    )
    setup = _HELPERS_SETUP.replace("helpers.speed_kph", "{package}.helpers.speed_kph")
    scenario, config, path = _write_case(
        tmp_path,
        setup.replace("{package}", f"acs_typecheck_case_{_counter + 1}"),
        header=header,
        modules={"helpers.py": _HELPERS},
    )
    error = _only_error(typecheck_scenario(scenario, config))
    assert "import the module with a name" in error.message, error.format()
    assert ".helpers as helpers`" in error.message, error.format()
    assert error.path == str(path)
    assert error.line == _line_of(path, "import acs_typecheck_case_")


_ALL_HELPERS = """
__all__ = ["make_label"]
SPEED = 3.0


def make_label(i: int) -> str:
    return f"ego_{i}"
"""

_STAR_HEADER = _HEADER.replace(
    "from .configs import CaseConfig\n",
    "from .configs import CaseConfig\nfrom .helpers import *\n",
    1,
)


def test_a_star_import_binds_the_all_of_the_module_only(tmp_path: Path) -> None:
    # Python binds `__all__` only, so SPEED raises NameError there (#506).
    setup = _TIMEOUT_SETUP + 'logger.info("%s", make_label(1))\n'
    scenario, config, _ = _write_case(
        tmp_path, setup, header=_STAR_HEADER, modules={"helpers.py": _ALL_HELPERS}
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()

    scenario, config, path = _write_case(
        tmp_path,
        setup + "x = SPEED\n",
        header=_STAR_HEADER,
        modules={"helpers.py": _ALL_HELPERS},
    )
    error = _only_error(typecheck_scenario(scenario, config))
    assert "name 'SPEED' is not defined" in error.message, error.format()
    assert error.line == _line_of(path, "x = SPEED")


_CYCLE_A = """
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

class A:
    v: int
{attribute}
    def __init__(self, v: int) -> None:
        self.v = v
{init}
    def total(self, b: "B") -> int:
        return b.w + self.v

if TYPE_CHECKING:
    from .b import B  # noqa: E402
"""

_CYCLE_B = """
from __future__ import annotations

from .a import A

class B:
    w: int
    a: A

    def __init__(self, w: int) -> None:
        self.w = w
        self.a = A(w)
"""

_CYCLE_HEADER = _HEADER.replace(
    "from .configs import CaseConfig\n",
    "from .configs import CaseConfig\nfrom .a import A\nfrom .b import B\n",
    1,
)
_CYCLE_SETUP = _TIMEOUT_SETUP + "n = B(1).a.total(B(2))\n"


def test_two_modules_naming_each_other_in_signatures_compile(
    tmp_path: Path,
) -> None:
    # The `if TYPE_CHECKING:` import at the end of the module, and the
    # signature naming it dropped (#507).
    modules = {
        "a.py": _CYCLE_A.replace("{attribute}\n", "").replace("{init}\n", ""),
        "b.py": _CYCLE_B,
    }
    scenario, config, _ = _write_case(
        tmp_path, _CYCLE_SETUP, header=_CYCLE_HEADER, modules=modules
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_an_attribute_typed_by_a_class_of_a_module_cycle_is_explained(
    tmp_path: Path,
) -> None:
    # Codon cannot name a class of the other module in an attribute
    # declaration, whatever the order: the hint names what compiles.
    modules = {
        "a.py": _CYCLE_A.replace("{attribute}", '    peer: Optional["B"]').replace(
            "{init}", "        self.peer = None"
        ),
        "b.py": _CYCLE_B,
    }
    scenario, config, path = _write_case(
        tmp_path, _CYCLE_SETUP, header=_CYCLE_HEADER, modules=modules
    )
    error = _only_error(typecheck_scenario(scenario, config))
    assert (
        "a module cycle cannot hold a class attribute" in error.message
    ), error.format()
    assert error.path == str(path.parent / "a.py")
    assert error.line == _line_of(path.parent / "a.py", "peer: Optional")


# ---------------------------------------------------------------------------
# An import repeated in a module (#354)
# ---------------------------------------------------------------------------

_REPEAT_BODY = """
        self._setup_ego_spawn()
        self.register_fail_condition(TimeoutCondition(5.0, label="t"))
        logger.info("%s", ComparisonRule.LESS_THAN.value)
        """


@pytest.mark.parametrize(
    "repeat",
    [
        "from autoware_carla_scenario import ComparisonRule\n",
        "from autoware_carla_scenario import TimeoutCondition, ComparisonRule\n",
        "from autoware_carla_scenario import ComparisonRule, ComparisonRule\n",
        "from autoware_carla_scenario import ComparisonRule as ComparisonRule\n",
        "import typesafe_carla.carla as carla\nimport logging\n",
    ],
)
def test_an_import_repeated_in_the_module_compiles(tmp_path: Path, repeat: str) -> None:
    header = _HEADER.replace("import logging\n", "import logging\n" + repeat, 1)
    scenario, config, _ = _write_case(tmp_path, _REPEAT_BODY, header=header)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_an_import_repeated_inside_one_statement_compiles(tmp_path: Path) -> None:
    header = _HEADER.replace(
        "    TimeoutCondition,\n", "    TimeoutCondition,\n    TimeoutCondition,\n", 1
    )
    scenario, config, _ = _write_case(tmp_path, _REPEAT_BODY, header=header)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_a_name_rebound_by_a_later_import_before_any_use_compiles(
    tmp_path: Path,
) -> None:
    other = (
        "from autoware_carla_scenario import ScalarComparisonRule as ComparisonRule\n"
    )
    header = _HEADER.replace("import logging\n", "import logging\n" + other, 1)
    scenario, config, _ = _write_case(tmp_path, _REPEAT_BODY, header=header)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_a_name_rebound_after_it_was_used_is_refused_at_the_import(
    tmp_path: Path,
) -> None:
    other = (
        "from autoware_carla_scenario import ScalarComparisonRule as ComparisonRule\n"
        "_FIRST = ComparisonRule\n"
    )
    header = _HEADER.replace("import logging\n", "import logging\n" + other, 1)
    scenario, config, path = _write_case(tmp_path, _REPEAT_BODY, header=header)
    error = _only_error(typecheck_scenario(scenario, config))
    assert error.path == str(path)
    # The header's own import statement (it opens one line above its first
    # name) is the later binding.
    assert error.line == _line_of(path, "    EGO_ROLE_NAME,") - 1
    assert f"first on line {_line_of(path, 'ScalarComparisonRule')}" in error.message
    assert "`ComparisonRule` is imported again" in error.message


# ---------------------------------------------------------------------------
# A scenario's own dataclass, IntEnum / (str, Enum) and NamedTuple (#365)
# ---------------------------------------------------------------------------

_DATA_IMPORTS = "\n\nimport dataclasses\nimport enum\nimport typing\nfrom dataclasses import dataclass\n\n\n"


def _data_case(tmp_path: Path, definitions: str, setup: str) -> tuple[Any, Any, Path]:
    return _write_case(
        tmp_path, _TIMEOUT_SETUP + textwrap.dedent(setup), _DATA_IMPORTS + definitions
    )


_DATACLASSES = """
@dataclass
class Fz:
    a: int


@dataclass(order=True)
class Ordered:
    a: int
    b: str = "x"


@dataclass(frozen=True)
class Key:
    a: int


@dataclass(frozen=True, order=True)
class SubKey(Key):
    b: float = 1.0


@dataclass(eq=False)
class ByIdentity:
    a: int
"""


def test_a_scenarios_dataclass_compares_orders_and_hashes(tmp_path: Path) -> None:
    # Python's @dataclass generates these; the check drops the decorator, so it
    # appends them after the module (#365).
    setup = """
        assert Fz(1) == Fz(1)
        assert Fz(1) != Fz(2)
        assert Fz(1) in [Fz(1)]
        assert sorted([Ordered(2), Ordered(1)])[0] == Ordered(1)
        assert Ordered(1) <= Ordered(2) and Ordered(2) >= Ordered(1)
        assert Ordered(1) < Ordered(2) and Ordered(2) > Ordered(1)
        assert {Key(1): 2}[Key(1)] == 2
        assert len({SubKey(1, 2.0), SubKey(1, 2.0)}) == 1
        assert SubKey(1) < SubKey(1, 2.0)
        assert hash(Key(3)) == hash(Key(3))
    """
    scenario, config, _ = _data_case(tmp_path, _DATACLASSES, setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


_ENUMS = """
class Level(enum.IntEnum):
    LOW = 1
    HIGH = 2


class Mode(str, enum.Enum):
    FAST = "fast"
    SLOW = "slow"
"""


def test_an_int_enum_and_a_str_enum_compare_like_their_values(tmp_path: Path) -> None:
    setup = """
        assert Level.LOW == 1 and Level.LOW != 2
        assert Level.LOW < Level.HIGH and Level.HIGH >= Level.LOW
        assert Level.LOW <= 1 and Level.HIGH > 1
        assert int(Level.HIGH) == 2
        assert Level.LOW.value + 1 == 2
        assert Mode.FAST == "fast" and Mode.FAST != "slow"
        assert Mode.FAST == Mode.FAST and Mode.FAST != Mode.SLOW
        assert Mode.FAST.value.upper() == "FAST"
        assert Level.LOW in [Level.LOW, Level.HIGH]
    """
    scenario, config, _ = _data_case(tmp_path, _ENUMS, setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


_ITERATED_ENUMS = (
    _ENUMS
    + """

class Plain(enum.Enum):
    A = "a"
    B = "b"
    ALIAS = "a"
"""
)


def test_an_enum_class_is_iterated_and_looked_up_by_name(tmp_path: Path) -> None:
    # `for m in Mode`, `list(Mode)`, `len(Mode)` and `Mode["A"]` (#377).
    setup = """
        assert [m.name for m in Plain] == ["A", "B"]
        assert len(list(Plain)) == 2 and len(Plain) == 2
        assert Plain["B"].value == "b" and Plain["ALIAS"] is not None
        assert {m: m.value for m in Plain}[Plain.A] == "a"
        assert [m.value for m in Mode] == ["fast", "slow"]
        assert sorted(Level)[0] == Level.LOW and max(Level) == Level.HIGH
        assert Level["HIGH"] > Level["LOW"] and Mode["SLOW"] == "slow"
        assert Mode.FAST in Mode and Plain.A in list(Plain)
        names = {m.name.lower(): m for m in Level}
        assert names["low"] == Level.LOW
        total = 0
        for i, m in enumerate(Level):
            total += i + int(m)
        assert total == 4
        try:
            Plain["X"]
            assert False
        except KeyError:
            pass
    """
    scenario, config, _ = _data_case(tmp_path, _ITERATED_ENUMS, setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


_CLS_ENUMS = """
class Gear(enum.Enum):
    P = "p"
    D = "d"

    @classmethod
    def names(cls) -> typing.List[str]:
        return [g.name for g in cls]

    @classmethod
    def everything(cls) -> typing.List["Gear"]:
        return list(cls)

    @classmethod
    def count(cls) -> int:
        return len(cls)

    @classmethod
    def known(cls, g: "Gear") -> bool:
        return g in cls

    @classmethod
    def parse(cls, name: str) -> "Gear":
        return cls[name]

    def peers(self) -> typing.List["Gear"]:
        return [m for m in type(self) if m != self]

    def siblings(self) -> int:
        return len(self.__class__) + len(list(type(self))) + type(self)["P"].value.__len__()


class Speed(enum.IntEnum):
    LOW = 1
    HIGH = 2

    @classmethod
    def fastest(cls) -> "Speed":
        return max(cls)

    @classmethod
    def total(cls) -> int:
        return sum([int(s) for s in cls]) + len(list(cls)) + (1 if cls["LOW"] in cls else 0)


class Bits(enum.Flag):
    A = 1
    B = 2
    AB = A | B

    @classmethod
    def singles(cls) -> typing.List["Bits"]:
        return list(cls)

    @classmethod
    def has(cls, b: "Bits") -> bool:
        return b in cls and len(cls) == 2 and cls["AB"] == cls.A | cls.B


class Tag(str, enum.Enum):  # enum.StrEnum is Python 3.11+
    X = "x"
    Y = "y"

    @classmethod
    def values(cls) -> typing.List[str]:
        return [t.value for t in cls]

    @classmethod
    def parse(cls, s: str) -> "Tag":
        for t in cls:
            if t.value == s:
                return t
        return cls["X"]
"""


def test_an_enum_class_is_iterated_through_cls_in_its_own_methods(
    tmp_path: Path,
) -> None:
    # `cls` in a classmethod, `type(self)` and `self.__class__` in an instance
    # method (#481).
    setup = """
        assert Gear.names() == ["P", "D"] and Gear.count() == 2
        assert Gear.everything()[1] == Gear.D and Gear.known(Gear.P)
        assert Gear.parse("D") == Gear.D and Gear.D.peers() == [Gear.P]
        assert Gear.P.siblings() == 5
        assert Speed.fastest() == Speed.HIGH and Speed.total() == 6
        assert Bits.singles() == [Bits.A, Bits.B] and Bits.has(Bits.A)
        assert Tag.values() == ["x", "y"] and Tag.parse("y") == Tag.Y
    """
    scenario, config, _ = _data_case(tmp_path, _CLS_ENUMS, setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


_CLS_MISUSE = """
class Gear(enum.Enum):
    P = 0
    D = 1

    @classmethod
    def lambda_shadow(cls) -> int:
        f = lambda cls: len(cls)
        return f([1, 2])

    @classmethod
    def rebound(cls) -> typing.List[str]:
        cls = [1, 2]
        return [str(g) for g in cls]
"""


def test_a_lambdas_own_cls_and_a_rebound_cls_are_not_the_enum(
    tmp_path: Path,
) -> None:
    setup = """
        assert Gear.lambda_shadow() == 2 and Gear.rebound() == ["1", "2"]
    """
    scenario, config, _ = _data_case(tmp_path, _CLS_MISUSE, setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("return [g for g in cls.__members__]", "no attribute '__members__'"),
        ("return [g for g in _rows(cls)]", "does not match expected type 'int'"),
        # Another name for the class: refused, with the hint (#481).
        (
            "k = cls\n        return [str(g) for g in k]",
            "write the class's name (`for m in Gear`)",
        ),
    ],
)
def test_an_enum_class_held_by_another_name_is_refused(
    tmp_path: Path, body: str, message: str
) -> None:
    definitions = f"""
def _rows(c: int) -> typing.List[int]:
    return [1]


class Gear(enum.Enum):
    P = 0
    D = 1

    @classmethod
    def names(cls) -> typing.List[str]:
        {body}
"""
    scenario, config, path = _data_case(tmp_path, definitions, "Gear.names()\n")
    result = typecheck_scenario(scenario, config)
    assert not result.ok, result.format()
    assert message in result.format()


def test_the_framework_enums_are_iterated_and_looked_up_by_name(
    tmp_path: Path,
) -> None:
    # The model's enums read the same rewrite as a scenario's own (#437).
    setup = """
        assert [m.name for m in TurnDirection] == ["LEFT", "RIGHT", "STRAIGHT"]
        assert len(list(TurnDirection)) == 3 and len(TurnDirection) == 3
        assert TurnDirection["LEFT"] == TurnDirection.LEFT
        assert ComparisonRule["LESS_THAN"] == ComparisonRule.LESS_THAN
        assert TurnDirection.RIGHT in TurnDirection
        assert {m.value: m for m in TickTiming}["pre_tick"] == TickTiming.PRE_TICK
        for rule in ComparisonRule:
            assert rule.name != ""
        try:
            TurnDirection["NOPE"]
            assert False
        except KeyError:
            pass
    """
    scenario, config, _ = _write_case(tmp_path, _TIMEOUT_SETUP + textwrap.dedent(setup))
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


@pytest.mark.parametrize(
    "line",
    [
        "TurnDirection.LEFT | TurnDirection.RIGHT",
        "TurnDirection.LEFT in TurnDirection.RIGHT",
        "~TurnDirection.LEFT",
    ],
)
def test_a_framework_enum_that_is_no_flag_has_no_flag_operators(
    tmp_path: Path, line: str
) -> None:
    scenario, config, path = _write_case(tmp_path, _TIMEOUT_SETUP + f"r = {line}\n")
    error = _only_error(typecheck_scenario(scenario, config))
    assert error.line == _line_of(path, f"r = {line}"), error.format()


_FLAGS = """
class Perm(enum.IntFlag):
    NONE = 0
    R = 1
    W = 2
    RW = R | W
    X = enum.auto()


class Bits(enum.Flag):
    A = 1
    B = 2
    AB = A | B
    C = enum.auto()
    ALL = A | B | C
"""


_FLAG_USERS = """

MODULE_DEFAULT = Perm.R | Perm.W


class Holder:
    mode: Perm
    DEFAULT = Perm.R | Perm.W

    def __init__(self) -> None:
        self.mode: Perm = Holder.DEFAULT


def pick(TurnDirection: typing.List[int]) -> int:
    # A parameter called like a framework enum is not that enum.
    return len(TurnDirection) + sum([t for t in TurnDirection])
"""


def test_flag_and_intflag_members_combine(tmp_path: Path) -> None:
    setup = """
        rw: Perm = Perm.R | Perm.W
        assert rw == Perm.RW and rw == 3 and int(rw) == 3 and rw.value == 3
        assert Perm.R in rw and Perm.X not in rw and Perm.RW in rw
        assert (rw & Perm.W) == Perm.W and (rw ^ Perm.R) == Perm.W
        assert (rw | 4).value == 7 and (4 | rw).value == 7
        assert (~Perm.R).value == 6 and bool(rw) and not Perm.NONE
        both: Bits = Bits.A | Bits.B
        assert both == Bits.AB and Bits.A in both and Bits.C not in both
        assert (both & Bits.A) == Bits.A and (both ^ Bits.A) == Bits.B
        assert (~Bits.A).value == 6 and bool(both) and not (Bits.A & Bits.B)
        assert Bits.ALL.value == 7 and both != Bits.A
        mask = Perm.NONE
        for p in Perm:
            mask = mask | p
        assert mask.value == 7
        assert [p.name for p in Bits] == ["A", "B", "C"]
        assert Perm["RW"] == Perm.RW and len(Perm) == 3
        assert {rw: 1}[Perm.RW] == 1
        assert Perm.R < Perm.W and int(Perm.W) == 2
        assert Holder().mode == Perm.R | Perm.W == Holder.DEFAULT == MODULE_DEFAULT
        assert pick([1, 2]) == 5
    """
    scenario, config, _ = _data_case(tmp_path, _FLAGS + _FLAG_USERS, setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


@pytest.mark.parametrize(
    ("line", "hint"),
    [
        ("Perm.R + Perm.W", "combines with `|`, `&`, `^`, `~` and `in`"),
        ("Perm.R + Perm.W", "use its .value"),
        ("Bits.A + Bits.B", "combines with `|`, `&`, `^`, `~` and `in`"),
    ],
)
def test_the_hint_for_flag_arithmetic_names_the_operators_it_has(
    tmp_path: Path, line: str, hint: str
) -> None:
    scenario, config, _ = _data_case(tmp_path, _FLAGS, f"r = {line}\n")
    error = _only_error(typecheck_scenario(scenario, config))
    assert hint in error.message, error.format()


@pytest.mark.parametrize(
    "line",
    [
        "Perm.R | Bits.A + 1",
        "Bits.A < Bits.B",
        "int(Bits.A)",
        "Bits.A + Bits.B",
        "Bits.A + 1",
        "Bits.A | 1",
        "Bits.A | Perm.R",
        "Plain.A | Plain.B",
        "Plain.A in Plain.B",
        "~Plain.A",
    ],
)
def test_a_flag_does_not_order_convert_or_add(tmp_path: Path, line: str) -> None:
    scenario, config, path = _data_case(
        tmp_path,
        _FLAGS + "\n\nclass Plain(enum.Enum):\n    A = 1\n    B = 2\n",
        f"r = {line}\n",
    )
    error = _only_error(typecheck_scenario(scenario, config))
    assert error.line == _line_of(path, f"r = {line}"), error.format()


def test_an_error_in_an_iterated_enum_keeps_its_line(tmp_path: Path) -> None:
    line = "r = [m.nope for m in Plain]"
    scenario, config, path = _data_case(tmp_path, _ITERATED_ENUMS, f"{line}\n")
    error = _only_error(typecheck_scenario(scenario, config))
    assert error.path == str(path)
    assert error.line == _line_of(path, line)


_NAMED_TUPLES = """
class Pt(typing.NamedTuple):
    x: int
    y: float = 1.0

    def scaled(self, k: float) -> float:
        return (self.x + self.y) * k


class Pair(typing.NamedTuple):
    a: int
    b: int = 0
"""


def test_a_namedtuple_class_is_a_tuple_class(tmp_path: Path) -> None:
    setup = """
        p = Pt(1)
        assert p.x == 1 and p.y == 1.0
        assert p == Pt(1, 1.0) and p != Pt(2)
        assert Pt(1) < Pt(2)
        assert {p: 1}[Pt(1)] == 1
        assert p.scaled(2.0) == 4.0
        q = Pair(1, 2)
        a, b = q
        assert q[0] == a and len(q) == 2
        for v in q:
            assert v >= 0
    """
    scenario, config, _ = _data_case(tmp_path, _NAMED_TUPLES, setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


@pytest.mark.parametrize(
    ("definitions", "line", "hint"),
    [
        (_DATACLASSES, "Fz(1) < Fz(2)", "order=True"),
        (_DATACLASSES, "{Fz(1): 2}", "frozen=True"),
    ],
    ids=["unordered", "unhashable"],
)
def test_what_a_dataclass_does_not_generate_is_refused_with_a_hint(
    tmp_path: Path, definitions: str, line: str, hint: str
) -> None:
    scenario, config, path = _data_case(tmp_path, definitions, f"r = {line}\n")
    error = _only_error(typecheck_scenario(scenario, config))
    assert hint in error.message, error.format()
    assert error.path == str(path)
    assert error.line == _line_of(path, f"r = {line}")


@pytest.mark.parametrize(
    ("call", "definitions", "hint"),
    [
        (
            "takes_str(Mode.FAST)",
            "def takes_str(s: str) -> int:\n    return len(s)\n",
            ".value",
        ),
        (
            "takes_int(Level.LOW)",
            "def takes_int(n: int) -> int:\n    return n\n",
            ".value",
        ),
        ("Level.LOW + 1", "", "use its .value"),
    ],
    ids=["str-parameter", "int-parameter", "arithmetic"],
)
def test_an_enum_member_is_not_passed_as_its_value_type(
    tmp_path: Path, call: str, definitions: str, hint: str
) -> None:
    scenario, config, path = _data_case(
        tmp_path, _ENUMS + "\n\n" + definitions, f"r = {call}\n"
    )
    error = _only_error(typecheck_scenario(scenario, config))
    assert hint in error.message, error.format()
    assert error.line == _line_of(path, f"r = {call}")


_PLAIN_ENUMS = """
class Plain(enum.Enum):
    A = 1
    B = 2


class Bits(enum.Flag):
    A = 1
    B = 2


class Word(str, enum.Enum):
    A = "a"
"""


@pytest.mark.parametrize(
    "line",
    ["Plain.A < Plain.B", "int(Plain.A)", "Bits.A < Bits.B", "Word.A < Word.A"]
    + ["int(Word.A)", "Plain.A + 1"],
)
def test_only_an_int_enum_orders_and_converts_like_its_value(
    tmp_path: Path, line: str
) -> None:
    # Python refuses these on a plain Enum / Flag / (str, Enum): the shim's
    # ordering and int() belong to the IntEnum base alone.
    scenario, config, path = _data_case(tmp_path, _PLAIN_ENUMS, f"r = {line}\n")
    error = _only_error(typecheck_scenario(scenario, config))
    assert error.line == _line_of(path, f"r = {line}"), error.format()


def test_enum_members_are_dictionary_keys_and_a_plain_enum_equals_only_members(
    tmp_path: Path,
) -> None:
    setup = """
        d = {Plain.A: 1, Plain.B: 2}
        e = {Level.LOW: 1}
        assert d[Plain.A] == 1 and e[Level.LOW] == 1
        assert Plain.A == Plain.A and Plain.A != Plain.B
        assert Plain.A.value == 1 and Plain.A.value != 2
        assert Word.A == "a"
    """
    scenario, config, _ = _data_case(
        tmp_path, _ENUMS + _PLAIN_ENUMS, textwrap.dedent(setup)
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_an_enum_compared_with_the_wrong_type_is_refused(tmp_path: Path) -> None:
    scenario, config, path = _data_case(tmp_path, _ENUMS, "r = Level.LOW == 'low'\n")
    error = _only_error(typecheck_scenario(scenario, config))
    assert "unsupported operand type(s) for ==" in error.message, error.format()


@pytest.mark.parametrize(
    ("line", "hint"),
    [
        ("Pt(1)._replace(x=2)", "build a new Pt(...)"),
        ("Pt(1)._asdict()", "build a new Pt(...)"),
        ("x, y = Pt(1)", "read the fields by name"),
        ("Pt(1)[1]", "read the fields by name"),
    ],
    ids=["replace", "asdict", "unpack-mixed", "index-mixed"],
)
def test_what_a_namedtuple_cannot_do_in_the_check_says_what_to_write(
    tmp_path: Path, line: str, hint: str
) -> None:
    scenario, config, path = _data_case(tmp_path, _NAMED_TUPLES, f"{line}\n")
    error = _only_error(typecheck_scenario(scenario, config))
    assert hint in error.message, error.format()
    assert error.line == _line_of(path, line)


# ---------------------------------------------------------------------------
# The names of typing a scenario may import (docs/typecheck.md, #364)
# ---------------------------------------------------------------------------

#: Each supported name of typing that the typing shim adds (or that is not a
#: Codon builtin): the import and the module-level code that uses it. The
#: cases cover every documented name (``test_typing_names``).
_TYPING_CASES: dict[str, tuple[str, str]] = {
    "Iterator": (
        "Iterator",
        "def count(n: int) -> Iterator[int]:\n    yield n\n",
    ),
    "Deque": (
        "Deque",
        "def window() -> Deque[int]:\n    d: Deque[int] = Deque()\n"
        "    d.append(1)\n    return d\n",
    ),
    "DefaultDict": (
        "DefaultDict",
        "def groups() -> DefaultDict[str, int]:\n"
        "    d: DefaultDict[str, int] = DefaultDict(int)\n"
        '    d["a"] += 1\n    return d\n',
    ),
    "Counter": (
        "Counter",
        "def tally() -> Counter[str]:\n    c: Counter[str] = Counter()\n"
        '    c["a"] += 1\n    return c\n',
    ),
    "Final": (
        "Final",
        "LIMIT: Final = 3\nOTHER: Final[int] = 4\n",
    ),
    "Annotated": (
        "Annotated",
        "def scaled(x: Annotated[float, 'm/s']) -> Annotated[float, 'km/h']:\n"
        "    return x * 3.6\n",
    ),
    "NoReturn": (
        "NoReturn",
        "def fail(why: str) -> NoReturn:\n    raise ValueError(why)\n",
    ),
    "Type": (
        "Type",
        "def build(kind: Type[int]) -> int:\n    return 1\n",
    ),
    "Sized": (
        "Sized",
        "def size(x: Sized) -> int:\n    return 1\n",
    ),
    "Hashable": (
        "Hashable",
        "def key(x: Hashable) -> int:\n    return 1\n",
    ),
    "cast": (
        "cast",
        "def widen(x: int) -> float:\n    return cast(float, x) * 1.5\n"
        "def same(x: Optional[int]) -> int:\n    return cast(int, x) + 1\n"
        "def hush(x: int) -> int:\n    return cast('Callable', x)\n",
    ),
    # #514: the names that only a type checker reads.
    "Self": (
        "Self",
        "class Counter:\n    n: int\n    def __init__(self, n: int) -> None:\n"
        "        self.n = n\n    def inc(self) -> Self:\n        self.n += 1\n"
        "        return self\n"
        "class Twice(Counter):\n    def inc(self) -> Self:\n        self.n += 2\n"
        "        return self\n",
    ),
    "override": (
        "override",
        "class Base:\n    def f(self) -> int:\n        return 1\n"
        "class Sub(Base):\n    @override\n    def f(self) -> int:\n"
        "        return 2\n",
    ),
    "TypeAlias": (
        "TypeAlias",
        "Pose: TypeAlias = tuple[float, float]\n"
        "def norm(p: Pose) -> float:\n    return p[0] + p[1]\n",
    ),
    "NewType": (
        "NewType",
        "LaneId = NewType('LaneId', int)\nPair = NewType('Pair', tuple[int, int])\n"
        "def nxt(i: LaneId) -> LaneId:\n    return LaneId(i + 1)\n"
        "def swap(p: Pair) -> Pair:\n    return Pair((p[1], p[0]))\n",
    ),
    "Never": (
        "Never",
        "def stop(why: str) -> Never:\n    raise RuntimeError(why)\n",
    ),
    "assert_never": (
        "assert_never",
        "def label(k: int) -> str:\n    if k == 0:\n        return 'a'\n"
        "    elif k == 1:\n        return 'b'\n    else:\n"
        "        assert_never(k)\n"
        "def other(k: int) -> str:\n    if k == 0:\n        return 'a'\n"
        "    return assert_never(k)\n",
    ),
    "TypeGuard": (
        "TypeGuard",
        "def is_pair(v: tuple[float, ...]) -> TypeGuard[tuple[float, float]]:\n"
        "    return len(v) == 2\n",
    ),
    "TypeIs": (
        "TypeIs",
        "def is_pair(v: tuple[float, ...]) -> TypeIs[tuple[float, float]]:\n"
        "    return len(v) == 2\n",
    ),
    "Container": (
        "Container",
        "def has(c: Container[int], x: int) -> bool:\n    return x in c\n",
    ),
}

_TYPING_CALLS = {
    "Iterator": "for i in count(2):\n    logger.info('%d', i)\n",
    "Deque": "logger.info('%d', len(window()))\n",
    "DefaultDict": "logger.info('%d', len(groups()))\n",
    "Counter": "logger.info('%d', len(tally()))\n",
    "Final": "logger.info('%d', LIMIT + OTHER)\n",
    "Annotated": "logger.info('%f', scaled(2.0))\n",
    "NoReturn": "if cfg.timeout_seconds < 0:\n    fail('x')\n",
    "Type": "logger.info('%d', build(int))\n",
    "Sized": "logger.info('%d', size([1]))\n",
    "Hashable": "logger.info('%d', key(1))\n",
    "cast": "logger.info('%f %d %d', widen(2), same(1), hush(1))\n",
    "Self": "logger.info('%d %d', Counter(1).inc().n, Twice(1).inc().n)\n",
    "override": "logger.info('%d', Sub().f())\n",
    "TypeAlias": "logger.info('%f', norm((1.0, 2.0)))\n",
    "NewType": "logger.info('%d %d', nxt(LaneId(1)), swap(Pair((1, 2)))[0])\n",
    "Never": "if cfg.timeout_seconds < 0:\n    stop('x')\n",
    "assert_never": "logger.info('%s %s', label(0), other(0))\n",
    "TypeGuard": "logger.info('%s', is_pair((1.0, 2.0)))\n",
    "TypeIs": "logger.info('%s', is_pair((1.0, 2.0)))\n",
    "Container": "logger.info('%s', has([1, 2], 2))\n",
}


@pytest.mark.parametrize("name", sorted(_TYPING_CASES))
def test_a_supported_typing_name_compiles_in_a_scenario(
    tmp_path: Path, name: str
) -> None:
    import typing

    if not hasattr(typing, name):
        pytest.skip(f"typing has no {name} before this Python")
    imports, helper = _TYPING_CASES[name]
    imports = f"{imports}, Optional, Callable" if name == "cast" else imports
    header = _HEADER.replace(
        "from .configs import CaseConfig\n",
        f"from typing import {imports}\n\nfrom .configs import CaseConfig\n",
        1,
    )
    scenario, config, _ = _write_case(
        tmp_path,
        _TIMEOUT_SETUP + _TYPING_CALLS[name],
        "\n\n" + helper,
        header=header,
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


@pytest.mark.parametrize(
    ("name", "usage", "hint"),
    [
        (
            "Generic",
            'T = TypeVar("T")\n\n\nclass Box(Generic[T]):\n    pass\n',
            "leave the type parameters",
        ),
        (
            "TypedDict",
            "class Point(TypedDict):\n    x: float\n",
            "@dataclass",
        ),
        ("ChainMap", "", "docs/typecheck.md"),
    ],
)
def test_an_unsupported_typing_name_is_refused_with_a_hint(
    tmp_path: Path, name: str, usage: str, hint: str
) -> None:
    header = _HEADER.replace(
        "from .configs import CaseConfig\n",
        f"from typing import TypeVar, {name}\n\nfrom .configs import CaseConfig\n",
        1,
    )
    scenario, config, path = _write_case(
        tmp_path, _TIMEOUT_SETUP, "\n\n" + usage, header=header
    )
    error = _only_error(typecheck_scenario(scenario, config))
    assert f"cannot import name '{name}' from 'typing'" in error.message
    assert hint in error.message, error.format()
    assert error.path == str(path)
    assert error.line == _line_of(path, f"from typing import TypeVar, {name}")


#: Names the documented list has that typing gained after Python 3.10 (the
#: suite runs on it too; typing_extensions backports them, below).
_TYPING_NEWER_THAN_3_10 = {"Self", "override", "Never", "assert_never", "TypeIs"}

#: What Codon's own typing has, which the shim does not declare.
_CODON_TYPING_BUILTINS = {
    "Optional",
    "List",
    "Dict",
    "Set",
    "Tuple",
    "Callable",
    "Union",
    "ClassVar",
    "Generator",
    "overload",
}


def test_typing_names() -> None:
    # The documented list, the typing shim and the cases above agree.
    root = Path(__file__).resolve().parents[2]
    docs = (root / "docs" / "typecheck.md").read_text()
    block = docs.split("<!-- typing-names:start -->")[1].split(
        "<!-- typing-names:end -->"
    )[0]
    documented = set(re.findall(r"`(\w+)`", block))

    shim = ast.parse(
        (root / "src/autoware_carla_scenario/typecheck/codon/typing.codon").read_text()
    )
    declared: set[str] = set()
    for node in shim.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            declared.add(node.name)
        elif isinstance(node, ast.Assign):
            declared |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            declared.add(node.target.id)
        elif isinstance(node, ast.ImportFrom):
            declared |= {a.asname or a.name for a in node.names}
    assert documented == declared | _CODON_TYPING_BUILTINS
    # Every name the shim adds beyond Python's typing exists there too.
    import typing

    assert documented <= set(dir(typing)) | _TYPING_NEWER_THAN_3_10
    # Every name the shim adds beyond Codon is compiled by a test above.
    assert set(_TYPING_CASES) <= documented


# ---------------------------------------------------------------------------
# The names of Codon's functools, itertools and collections (#378)
# ---------------------------------------------------------------------------

#: Each name the check adds to a standard library module that Codon's lacks:
#: the import line and the module-level code that uses it, and a call from
#: setup().  The cases cover every documented name (``test_stdlib_names``).
_STDLIB_CASES: dict[str, tuple[str, str, str]] = {
    "functools.reduce": (
        "import operator\nfrom functools import reduce\n",
        "def total(xs: list[int]) -> int:\n    return reduce(operator.add, xs)\n"
        "def total_from(xs: list[int]) -> int:\n"
        "    return reduce(lambda a, b: a + b, xs, 10)\n",
        "logger.info('%d %d', total([1, 2]), total_from([1, 2]))\n",
    ),
    "functools.reduce (module)": (
        "import functools\nimport operator\n",
        "def total(xs: list[int]) -> int:\n"
        "    return functools.reduce(operator.add, xs)\n",
        "logger.info('%d', total([1, 2]))\n",
    ),
    "functools.lru_cache": (
        "import functools\nfrom functools import lru_cache\n",
        "@lru_cache(maxsize=8)\ndef square(x: int) -> int:\n    return x * x\n"
        "@functools.lru_cache()\ndef cube(x: int) -> int:\n    return x * x * x\n"
        "@lru_cache\ndef fourth(x: int) -> int:\n    return x * x * x * x\n"
        "class Memo:\n"
        "    @lru_cache(maxsize=None)\n"
        "    def twice(self, x: int) -> int:\n        return 2 * x\n",
        "logger.info('%d %d %d %d', square(2), cube(2), fourth(2), Memo().twice(2))\n",
    ),
    "functools.cache": (
        "import functools\nfrom functools import cache\n",
        "@cache\ndef square(x: int) -> int:\n    return x * x\n"
        "@functools.cache\ndef cube(x: int) -> int:\n    return x * x * x\n",
        "logger.info('%d %d', square(2), cube(2))\n",
    ),
    "functools.cached_property": (
        "from functools import cached_property\n",
        "class Box:\n    side: int\n\n    def __init__(self, side: int):\n        self.side = side\n\n"
        "    @cached_property\n    def area(self) -> int:\n"
        "        return self.side * self.side\n",
        "logger.info('%d', Box(2).area)\n",
    ),
    "itertools.pairwise": (
        "from itertools import chain, pairwise\n",
        "def gaps(xs: list[int]) -> list[int]:\n"
        "    return [b - a for a, b in pairwise(xs)]\n"
        "def flat(xs: list[int]) -> list[int]:\n"
        "    return list(chain(xs, xs))\n",
        "logger.info('%d %d', len(gaps([1, 3, 6])), len(flat([1])))\n",
    ),
    "collections.OrderedDict": (
        "from collections import Counter, OrderedDict\n",
        "def ordered() -> OrderedDict[str, int]:\n"
        "    d: OrderedDict[str, int] = OrderedDict()\n"
        '    d["a"] = 1\n    d["b"] = 2\n    return d\n'
        "def tally() -> Counter[str]:\n    c: Counter[str] = Counter()\n"
        '    c["a"] += 1\n    return c\n',
        "logger.info('%d %d', len(ordered()), len(tally()))\n",
    ),
    "collections.deque (extend, maxlen)": (
        "from collections import deque\n",
        "def window(xs: list[float]) -> list[float]:\n"
        "    d: deque[float] = deque(maxlen=3)\n"
        "    d.extend(xs)\n    d.extendleft([0.5])\n"
        "    d.extend(d)\n    return list(d)\n"
        "def kept(xs: list[int]) -> deque[int]:\n"
        "    return deque(xs, maxlen=2)\n"
        "def kept_kw(xs: list[int]) -> deque[int]:\n"
        "    return deque(xs, maxlen=2)\n",
        "assert window([1.0, 2.0, 3.0, 4.0]) == [3.0, 4.0]\n"
        "assert list(kept([1, 2, 3])) == [2, 3] and list(kept_kw([1])) == [1]\n"
        "m: deque[int] = deque(maxlen=3)\n"
        "m.extend([1, 2, 3, 4])\n"
        "assert list(m) == [2, 3, 4]\n"
        "z: deque[int] = deque(maxlen=0)\n"
        "z.append(1)\nz.appendleft(2)\nz.extend([3])\n"
        "assert list(z) == [] and len(z) == 0\n"
        "logger.info('%d', len(window([1.0])))\n",
    ),
    "collections.deque (rotate, search)": (
        "from collections import deque\n",
        "def turned() -> list[int]:\n"
        "    d = deque([1, 2, 3, 4])\n"
        "    d.rotate(1)\n    d.rotate(-2)\n    d.rotate()\n"
        "    return list(d)\n"
        "def edited() -> list[int]:\n"
        "    d = deque([1, 2, 3, 2])\n"
        "    d.remove(2)\n    d.reverse()\n    d.insert(1, 9)\n"
        "    d[0] = 7\n    del d[2]\n"
        "    assert d.count(7) == 1 and d.index(9) == 1\n"
        "    assert d == d.copy() and d != deque([1])\n"
        "    return list(d)\n",
        "assert turned() == [4, 1, 2, 3]\n"
        "assert edited() == [7, 9, 2]\n"
        "logger.info('%d', len(turned()))\n",
    ),
    "math.dist": (
        "import math\nfrom math import dist\n",
        "def gap(a: tuple[float, float], b: tuple[float, float]) -> float:\n"
        "    return dist(a, b)\n"
        "def gap3(a: list[float], b: list[float]) -> float:\n"
        "    return math.dist(a, b)\n",
        "assert gap((0.0, 0.0), (3.0, 4.0)) == 5.0\n"
        "assert gap3([1.0, 2.0, 3.0], [4.0, 6.0, 3.0]) == 5.0\n"
        "logger.info('%f', gap((0.0, 0.0), (3.0, 4.0)))\n",
    ),
    "math.isqrt": (
        "import math\n",
        "def root(n: int) -> int:\n    return math.isqrt(n)\n",
        "assert root(10) == 3 and root(16) == 4 and root(0) == 0\n"
        "logger.info('%d', root(99))\n",
    ),
    "math.comb": (
        "import math\n",
        "def choose(n: int, k: int) -> int:\n    return math.comb(n, k)\n",
        "assert choose(5, 2) == 10 and choose(5, 7) == 0\n"
        "logger.info('%d', choose(60, 30))\n",
    ),
    "math.perm": (
        "import math\n",
        "def arrange(n: int) -> int:\n    return math.perm(n)\n"
        "def arrange_k(n: int, k: int) -> int:\n    return math.perm(n, k)\n",
        "assert arrange(5) == 120 and arrange_k(5, 2) == 20\n"
        "logger.info('%d', arrange_k(3, 5))\n",
    ),
    "math.cbrt": (
        "import math\n",
        "def cube_root(x: float) -> float:\n    return math.cbrt(x)\n",
        "assert cube_root(8.0) == 2.0 and cube_root(-27.0) == -3.0\n"
        "logger.info('%f', cube_root(1.0))\n",
    ),
    "math.exp2": (
        "import math\n",
        "def power(x: float) -> float:\n    return math.exp2(x)\n",
        "assert power(10.0) == 1024.0\nlogger.info('%f', power(1.0))\n",
    ),
    "math.hypot": (
        "import math\nfrom math import hypot\n",
        "def norm3(x: float, y: float, z: float) -> float:\n"
        "    return math.hypot(x, y, z)\n"
        "def norm2(x: float, y: float) -> float:\n    return hypot(x, y)\n"
        "def norm_mixed() -> float:\n    return hypot(1, 2.0, 2) + hypot() + hypot(-3)\n",
        "assert norm3(1.0, 2.0, 2.0) == 3.0 and norm2(3.0, 4.0) == 5.0\n"
        "assert norm_mixed() == 6.0\n"
        "logger.info('%f', norm3(1.0, 2.0, 2.0))\n",
    ),
    "statistics.fmean": (
        "import statistics\nfrom statistics import fmean\n",
        "def average(xs: list[float]) -> float:\n    return fmean(xs)\n"
        "def average_ints(xs: list[int]) -> float:\n"
        "    return statistics.fmean(xs)\n"
        "def weighted(xs: list[float], ws: list[float]) -> float:\n"
        "    return fmean(xs, ws) + fmean(xs, None)\n",
        "assert average([1.0, 2.0, 4.5]) == 2.5 and average_ints([1, 2, 3]) == 2.0\n"
        "assert weighted([1.0, 2.0, 4.5], [1.0, 0.0, 1.0]) == 5.25\n"
        "logger.info('%f', average([1.0, 2.0]))\n",
    ),
    "random.Random (choices, triangular)": (
        "import random\n",
        "def draw(rng: random.Random) -> int:\n"
        "    a = rng.choices([1, 2], weights=[1, 2], k=3)\n"
        "    b = rng.choices([1, 2], weights=[0.5, 1.5], k=3)\n"
        "    c = rng.choices([1, 2], k=3)\n"
        "    d = rng.choices([1, 2], cum_weights=[1, 3])\n"
        "    e = rng.choices(['a', 'b'], [1, 1])\n"
        "    return len(a) + len(b) + len(c) + len(d) + len(e)\n"
        "def skew(rng: random.Random) -> float:\n"
        "    return rng.triangular(0.0, 1.0) + rng.triangular() "
        "+ rng.triangular(0.0, 1.0, 0.25)\n",
        "rng = random.Random(1)\n"
        "assert draw(rng) == 11\n"
        "assert 0.0 <= skew(rng) <= 3.0\n"
        "logger.info('%d', draw(rng))\n",
    ),
    "re.Match (group, m[n], groups, groupdict)": (
        "import re\n",
        "def number(text: str) -> str:\n"
        "    m = re.match(r'(?P<word>[a-z]+)(\\d+)', text)\n"
        "    if m is None:\n"
        "        return ''\n"
        "    word, digits = m.groups()\n"
        "    first, second = m.group(1, 2)\n"
        "    kept: str = m.groupdict()['word']\n"
        "    return m.group(2) + m[1] + word + digits + first + second + kept\n"
        "def number_or_empty(text: str) -> str:\n"
        "    m = re.search(r'(\\d+)', text)\n"
        "    return m.group(1) if m else ''\n",
        "assert number('npc12') == '12npcnpc12npc12npc'\n"
        "logger.info('%s %s', number(cfg.name), number_or_empty(cfg.name))\n",
    ),
    "re.Pattern (split)": (
        "import re\n",
        "def pieces(text: str) -> int:\n"
        "    return len(re.compile(r'(-)').split(text)) + len(re.split(r'-', text))\n",
        "logger.info('%d', pieces(cfg.name))\n",
    ),
    "random.triangular": (
        "import random\nfrom random import triangular\n",
        "def skew() -> float:\n"
        "    return random.triangular(0.0, 1.0) + triangular(0.0, 2.0, 1.0) "
        "+ triangular()\n",
        "assert 0.0 <= skew() <= 4.0\nlogger.info('%f', skew())\n",
    ),
    "random.choices": (
        "import random\nfrom random import choices\n",
        "def pick() -> int:\n"
        "    a = random.choices([1, 2], weights=[1.0, 2.0], k=3)\n"
        "    b = choices([1, 2], weights=[1, 2])\n"
        "    c = random.choices([1, 2], k=2)\n"
        "    d = random.choices([1, 2], cum_weights=[1, 3])\n"
        "    return len(a) + len(b) + len(c) + len(d)\n",
        "assert pick() == 7\nlogger.info('%d', pick())\n",
    ),
}

#: Each name or module refused with a hint: the import line, the code using
#: it, the error Codon gives, and the hint the check adds.
_STDLIB_REFUSALS: dict[str, tuple[str, str, str, str]] = {
    "dataclasses.asdict": (
        "from dataclasses import asdict\n",
        "",
        "cannot import name 'asdict' from 'dataclasses'",
        "build the dict from the fields you need",
    ),
    "dataclasses.InitVar": (
        "from dataclasses import InitVar\n",
        "",
        "cannot import name 'InitVar' from 'dataclasses'",
        "make the value a parameter of a @classmethod",
    ),
    "dataclasses.KW_ONLY": (
        "from dataclasses import KW_ONLY\n",
        "",
        "cannot import name 'KW_ONLY' from 'dataclasses'",
        "drop the marker and pass the fields by keyword",
    ),
    "functools.wraps": (
        "from functools import wraps\n",
        "",
        "cannot import name 'wraps' from 'functools'",
        "docs/typecheck.md",
    ),
    "math.ulp": (
        "from math import ulp\n",
        "",
        "cannot import name 'ulp' from 'math'",
        "docs/typecheck.md",
    ),
    "contextlib.contextmanager": (
        "from contextlib import contextmanager\n",
        "",
        "cannot import name 'contextmanager' from 'contextlib'",
        "write a class with __enter__ and __exit__",
    ),
    "warnings": (
        "import warnings\n",
        "",
        "warnings has no Codon model",
        "logger.warning",
    ),
    "json.JSONEncoder": (
        "from json import JSONEncoder\n",
        "",
        "cannot import name 'JSONEncoder' from 'json'",
        "dumps, dump, loads, load and JSONDecodeError",
    ),
    "types": ("import types\n", "", "types has no Codon model", "@dataclass"),
    "textwrap": ("import textwrap\n", "", "textwrap has no Codon model", "join()"),
    "traceback": (
        "import traceback\n",
        "",
        "traceback has no Codon model",
        "logger.exception",
    ),
    "io": ("import io\n", "", "io has no Codon model", "pathlib.Path"),
    "csv": ("import csv\n", "", "csv has no Codon model", "split(',')"),
    "uuid": ("import uuid\n", "", "uuid has no Codon model", "os.getpid()"),
}


def _stdlib_header(imports: str) -> str:
    return _HEADER.replace(
        "from .configs import CaseConfig\n",
        f"{imports}\nfrom .configs import CaseConfig\n",
        1,
    )


@pytest.mark.parametrize("name", sorted(_STDLIB_CASES))
def test_a_supported_stdlib_name_compiles_in_a_scenario(
    tmp_path: Path, name: str
) -> None:
    imports, helper, call = _STDLIB_CASES[name]
    scenario, config, _ = _write_case(
        tmp_path,
        _TIMEOUT_SETUP + call,
        "\n\n" + helper,
        header=_stdlib_header(imports),
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


# ---------------------------------------------------------------------------
# Valid Python the transform rewrites for Codon (#376)
# ---------------------------------------------------------------------------

_REWRITE_CLASSES = """
from functools import total_ordering


class Marker: ...


class Failure(Exception): ...


class Detailed(Failure): ...


class ByLt:
    x: int

    def __init__(self, x: int) -> None:
        self.x = x

    def __lt__(self, other: "ByLt") -> bool:
        return self.x < other.x


@total_ordering
class ByLe:
    x: int

    def __init__(self, x: int) -> None:
        self.x = x

    def __le__(self, other: "ByLe") -> bool:
        return self.x <= other.x

    def __eq__(self, other: "ByLe") -> bool:
        return self.x == other.x


@total_ordering
class ByLtAndEq:
    x: int

    def __init__(self, x: int) -> None:
        self.x = x

    def __lt__(self, other: "ByLtAndEq") -> bool:
        return self.x < other.x

    def __eq__(self, other: "ByLtAndEq") -> bool:
        return self.x == other.x


class Unordered:
    x: int

    def __init__(self, x: int) -> None:
        self.x = x
"""


# ---------------------------------------------------------------------------
# The logging model (#496)
# ---------------------------------------------------------------------------

#: Calls of logging the model accepts, as CPython's signatures allow them.
_LOGGING_ACCEPTED: dict[str, str] = {
    "log with a variable level": (
        "level = logging.INFO if cfg.timeout_seconds > 1.0 else logging.DEBUG\n"
        'logger.log(level, "x %s", 1)\n'
        'logger.log(logging.WARNING, "x", exc_info=False, stack_info=True)\n'
    ),
    "deprecated warn": 'logger.warn("x %s", 1)\n',
    "extra with int values": (
        'logger.info("x", extra={"k": 1})\n'
        'logger.warning("x %s", 1, extra={"a": 1})\n'
    ),
    "extra with str values": 'logger.error("x", extra={"k": "v"})\n',
    "extra and stacklevel on every level method": (
        'extra = {"k": "v"}\n'
        'logger.debug("x", extra=extra, stacklevel=2)\n'
        'logger.info("x %s", 1, stacklevel=2)\n'
        'logger.warning("x", stack_info=True, stacklevel=3, extra=extra)\n'
        'logger.error("x", exc_info=True, extra=extra)\n'
        'logger.exception("x", stacklevel=2, extra=extra)\n'
        'logger.critical("x", extra=extra, stacklevel=2)\n'
        'logger.log(logging.INFO, "x", extra=extra, stacklevel=2)\n'
    ),
    "a message that is not a str": "logger.info(123)\nlogger.log(logging.INFO, 4.5)\n",
    "module-level functions": (
        'logging.debug("root")\nlogging.info("root %s", 1, extra={"k": 1})\n'
        'logging.warning("root", stacklevel=2)\nlogging.error("root")\n'
        'logging.exception("root")\nlogging.critical("root")\n'
        'logging.log(logging.INFO, "root", stack_info=True)\n'
        'logging.warn("root")\n'
    ),
    "basicConfig": (
        "logging.basicConfig(level=logging.INFO)\n"
        'logging.basicConfig(level="DEBUG", format="%(message)s", '
        'datefmt="%H:%M", style="%", force=True)\n'
        'logging.basicConfig(filename="run.log", filemode="w", encoding="utf-8")\n'
        "logging.basicConfig(stream=sys.stderr, handlers=[logging.StreamHandler()])\n"
    ),
    "handlers and formatters": (
        "handler = logging.StreamHandler()\n"
        "handler.setLevel(logging.INFO)\n"
        'handler.setFormatter(logging.Formatter("%(message)s", datefmt="%H:%M"))\n'
        "logger.addHandler(handler)\n"
        "logger.removeHandler(handler)\n"
        'file_handler = logging.FileHandler("run.log", mode="w", encoding="utf-8")\n'
        "logger.addHandler(file_handler)\n"
        "logger.addHandler(logging.NullHandler())\n"
    ),
    "levels and queries": (
        'logger.setLevel("INFO")\nlogger.setLevel(logging.NOTSET)\n'
        "logger.propagate = False\n"
        "enabled = logger.isEnabledFor(logging.DEBUG)\n"
        "effective = logger.getEffectiveLevel() + logging.CRITICAL + logging.FATAL\n"
        "logger.getChild('sub').info('x %s %s %s', enabled, effective, logging.WARN)\n"
        "logger.hasHandlers()\n"
        "n_handlers = len(logger.handlers)\n"
        "root = logging.getLogger()\nroot2 = logging.getLogger(None)\n"
        'name = logging.getLevelName(logging.INFO) + " " + str(n_handlers)\n'
        'number = logging.getLevelName("INFO") + 1\n'
        'logging.Formatter("%(message)s", defaults={"k": "v"})\n'
    ),
    "varargs of mixed types and exception in except": (
        'logger.info("%s %d %.1f", "a", 1, 2.5)\n'
        "try:\n    raise ValueError('x')\nexcept ValueError:\n"
        '    logger.exception("x")\n'
    ),
}


def test_the_logging_calls_compile_in_a_scenario(tmp_path: Path) -> None:
    # One scenario holds every accepted call: each alone costs a Codon run.
    setup = _TIMEOUT_SETUP + "".join(
        f"if cfg.timeout_seconds > {i}.5e9:\n" + textwrap.indent(call, "    ")
        for i, call in enumerate(_LOGGING_ACCEPTED.values())
    )
    scenario, config, _ = _write_case(
        tmp_path, setup, header=_stdlib_header("import sys\n")
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


#: Calls CPython refuses, and the one valid call Codon cannot type, with the
#: error the check gives.
_LOGGING_REFUSED: dict[str, tuple[str, str]] = {
    "log with a str level": (
        'logger.log("INFO", "x")\n',
        "does not match expected type 'int'",
    ),
    "log without a message": (
        "logger.log(logging.INFO)\n",
        "missing 1 required positional argument: 'msg'",
    ),
    "unknown keyword": ('logger.info("x", color="red")\n', "'color'"),
    "stacklevel as a str": (
        'logger.info("x", stacklevel="2")\n',
        "does not match expected type 'int'",
    ),
    "stack_info as an int": (
        'logger.info("x", stack_info=1)\n',
        "does not match expected type 'bool'",
    ),
    # Valid Python, refused by Codon's dict display (typed by its first value);
    # the check's hint says how to write it.
    "extra with mixed values": (
        'logger.info("x", extra={"k": "v", "n": 2})\n',
        "a dict display is typed by its first value",
    ),
    "module-level unknown keyword": ('logging.info("x", color="red")\n', "'color'"),
    "basicConfig positional": ("logging.basicConfig(logging.INFO)\n", "keyword-only"),
    "basicConfig unknown keyword": ("logging.basicConfig(colour=1)\n", "'colour'"),
    "basicConfig level of the wrong type": (
        "logging.basicConfig(level=1.5)\n",
        "takes a level as an int or a str",
    ),
    "setLevel of the wrong type": (
        "logger.setLevel(1.5)\n",
        "takes a level as an int or a str",
    ),
    "addHandler of a non-handler": (
        'logger.addHandler("x")\n',
        "does not match expected type 'Handler'",
    ),
}


@pytest.mark.parametrize("name", sorted(_LOGGING_REFUSED))
def test_a_logging_call_python_refuses_is_refused(tmp_path: Path, name: str) -> None:
    call, message = _LOGGING_REFUSED[name]
    scenario, config, path = _write_case(
        tmp_path, _TIMEOUT_SETUP + call, header=_stdlib_header("")
    )
    error = _only_error(typecheck_scenario(scenario, config))
    assert message in error.message, error.format()
    assert error.path == str(path)


def test_an_ellipsis_class_body_compiles(tmp_path: Path) -> None:
    setup = """
        marker = Marker()
        try:
            raise Detailed("why")
        except Failure:
            pass
    """
    scenario, config, _ = _data_case(tmp_path, _REWRITE_CLASSES, setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


@pytest.mark.parametrize("name", sorted(_STDLIB_REFUSALS))
def test_an_unsupported_stdlib_name_is_refused_with_a_hint(
    tmp_path: Path, name: str
) -> None:
    imports, usage, message, hint = _STDLIB_REFUSALS[name]
    scenario, config, path = _write_case(
        tmp_path, _TIMEOUT_SETUP, "\n\n" + usage, header=_stdlib_header(imports)
    )
    error = _only_error(typecheck_scenario(scenario, config))
    assert message in error.message, error.format()
    assert hint in error.message, error.format()
    assert error.path == str(path)
    first = imports.strip().splitlines()[0]
    assert error.line == path.read_text().splitlines().index(first) + 1


#: Names the check adds that Python's own module has from 3.11 on only: the
#: check accepts them on 3.10 too (docs/typecheck.md).
_PYTHON_3_11_NAMES = {"math.cbrt", "math.exp2"}


def test_stdlib_names() -> None:
    # The documented lists, the transform, the extensions and the cases agree.
    import itertools
    import collections
    import functools
    import math
    import random
    import statistics

    from autoware_carla_scenario.typecheck.transform import FUNCTOOLS_NAMES

    root = Path(__file__).resolve().parents[2]
    docs = (root / "docs" / "typecheck.md").read_text()

    def listed(marker: str) -> set[str]:
        block = docs.split(f"<!-- {marker}:start -->")[1].split(
            f"<!-- {marker}:end -->"
        )[0]
        return set(re.findall(r"`([\w.]+)`", block))

    declared = {f"functools.{name}" for name in FUNCTOOLS_NAMES}
    extensions = root / "src/autoware_carla_scenario/typecheck/codon/_stdlib"
    for path in extensions.glob("*.codon"):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                declared.add(f"{path.stem}.{node.name}")
            elif isinstance(node, ast.Assign):
                declared |= {
                    f"{path.stem}.{t.id}"
                    for t in node.targets
                    if isinstance(t, ast.Name)
                }
    assert listed("stdlib-names") == declared
    for name in declared - _PYTHON_3_11_NAMES:
        module, _, attribute = name.partition(".")
        assert hasattr(
            {
                "functools": functools,
                "itertools": itertools,
                "collections": collections,
                "math": math,
                "random": random,
                "statistics": statistics,
                "re": re,
            }[module],
            attribute,
        ), name
    covered = {name.split(" ")[0] for name in _STDLIB_CASES}
    assert covered == declared
    # Every refusal documented is tested, and the other way round.
    assert listed("stdlib-refused") == {name for name in _STDLIB_REFUSALS}
    # Every module refused with a hint is documented and tested (#438).
    from autoware_carla_scenario.typecheck.check import _UNMODELLED_HINTS

    assert {name for name in _STDLIB_REFUSALS if "." not in name} == set(
        _UNMODELLED_HINTS
    )


_FILE_DEFINITIONS = """
import os
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
HERE = os.path.dirname(os.path.abspath(__file__))
NAME: str = __file__


def here() -> Path:
    return Path(__file__).parent


def named() -> str:
    return os.path.basename(__file__)
"""


def test_the_file_of_a_module_is_defined(tmp_path: Path) -> None:
    # `__file__` is the module's path, as a str, at module level, in a
    # function and in os.path and pathlib calls (#515).
    setup = """
        assert NAME.endswith("scenario.py")
        assert named() == "scenario.py"
        assert str(here()) == os.path.dirname(NAME)
        logger.info("%s %s %s", DATA, HERE, NAME)
    """
    scenario, config, _ = _data_case(tmp_path, _FILE_DEFINITIONS, setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_an_ellipsis_class_is_still_an_empty_class(tmp_path: Path) -> None:
    scenario, config, path = _data_case(
        tmp_path, _REWRITE_CLASSES, "r = Marker().missing\n"
    )
    error = _only_error(typecheck_scenario(scenario, config))
    assert "no attribute 'missing'" in error.message, error.format()
    assert error.line == _line_of(path, "r = Marker().missing")


def test_a_dict_display_with_double_star_compiles(tmp_path: Path) -> None:
    setup = """
        a = {"x": 1}
        b = {"y": 2}
        merged = {**a, **b}
        copy = {**a}
        copy["z"] = 3
        mixed = {"w": 0, **a, "v": 5, **b}
        assert len(merged) == 2 and len(a) == 1 and len(mixed) == 4
        assert merged["y"] == 2 and mixed["v"] == 5
    """
    scenario, config, _ = _data_case(tmp_path, "", setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_a_dict_display_merging_other_value_types_is_refused(tmp_path: Path) -> None:
    line = 'r = {**{"a": 1}, **{"b": "x"}}'
    scenario, config, path = _data_case(tmp_path, "", line + "\n")
    error = _only_error(typecheck_scenario(scenario, config))
    assert "'str' does not match expected type 'int'" in error.message, error.format()
    assert error.line == _line_of(path, line)


def test_a_dict_display_of_mixed_primitives_gets_the_one_type_hint(
    tmp_path: Path,
) -> None:
    line = 'info: dict[str, Any] = {"a": 1, "b": "x", "c": 2.5}'
    setup = line + '\nlogger.info("%s", info["b"])\n'
    scenario, config, path = _data_case(tmp_path, "from typing import Any\n", setup)
    error = _only_error(typecheck_scenario(scenario, config))
    assert "dataclass" in error.message, error.format()
    assert "common base" not in error.message, error.format()
    assert error.line == _line_of(path, line), error.format()


def test_a_self_iterator_class_gets_the_generator_hint(tmp_path: Path) -> None:
    definitions = """
class Countdown:
    n: int

    def __init__(self, n: int) -> None:
        self.n = n

    def __iter__(self) -> "Countdown":
        return self

    def __next__(self) -> int:
        if self.n <= 0:
            raise StopIteration
        self.n -= 1
        return self.n
"""
    line = "for k in Countdown(3):"
    setup = f"total = 0\n{line}\n    total += k\n"
    scenario, config, path = _data_case(tmp_path, definitions, setup)
    error = _only_error(typecheck_scenario(scenario, config))
    assert "expected iterable expression" in error.message, error.format()
    assert "generator" in error.message, error.format()
    assert "yield" in error.message, error.format()
    assert error.line == _line_of(path, line), error.format()


def test_a_scalar_mismatch_outside_a_dict_display_gets_no_dict_hint(
    tmp_path: Path,
) -> None:
    definitions = """
def show(a: str, b: str) -> str:
    return a + b
"""
    line = 'shown = show(f"{1}", 2)'
    scenario, config, _ = _data_case(tmp_path, definitions, line + "\n")
    error = _only_error(typecheck_scenario(scenario, config))
    assert "dataclass" not in error.message, error.format()


def test_a_self_iterator_in_a_comprehension_gets_the_generator_hint(
    tmp_path: Path,
) -> None:
    definitions = """
class Countdown:
    n: int

    def __init__(self, n: int) -> None:
        self.n = n

    def __iter__(self) -> "Countdown":
        return self

    def __next__(self) -> int:
        if self.n <= 0:
            raise StopIteration
        self.n -= 1
        return self.n
"""
    setup = "ks = [k for k in Countdown(3)]\nlogger.info('%s', len(ks))\n"
    scenario, config, _ = _data_case(tmp_path, definitions, setup)
    error = _only_error(typecheck_scenario(scenario, config))
    assert "yield" in error.message, error.format()


def test_str_format_on_a_literal_compiles(tmp_path: Path) -> None:
    setup = """
        n = 7
        s = "{:>5}|{}".format(1, 2)
        t = "{a}-{b:.2f} {{x}} {0}".format(n, a="q", b=2.0)
        u = "{!r} {:,}".format("s", 1000000)
        v = "{0}{0}{1}".format(len([1, 2]), "k")
        w = "no fields".format()
        x = ("multi {} "
             "line {}").format(
            1, 2
        )
        assert s == "    1|2" and t == "q-2.00 {x} 7" and v == "22k"
        assert w == "no fields" and x == "multi 1 line 2"
    """
    scenario, config, _ = _data_case(tmp_path, "", setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


@pytest.mark.parametrize(
    ("setup", "hint"),
    [
        ('r = "{:>{}}".format(5, 3)', "a format spec with a field inside it"),
        ('r = "{} {}".format(1)', "positional argument 1 but only 1 given"),
        ('r = "{k}".format(j=1)', "keyword 'k'"),
        ('r = "{} {1}".format(1, 2)', "mixes automatic and manual"),
        ('r = "{}".format(*[1])', "*args or **kwargs"),
        ('fmt = "{}"\nr = fmt.format(1)', "write an f-string"),
    ],
    ids=["nested-spec", "missing", "keyword", "mixed", "starred", "not-a-literal"],
)
def test_str_format_that_cannot_be_rewritten_is_refused(
    tmp_path: Path, setup: str, hint: str
) -> None:
    scenario, config, path = _data_case(tmp_path, "", setup + "\n")
    error = _only_error(typecheck_scenario(scenario, config))
    assert hint in error.message, error.format()
    assert error.line == _line_of(path, setup.splitlines()[-1])


def test_fstring_format_specs_compile(tmp_path: Path) -> None:
    """Every spec form compiles: Codon's own, and those the check rewrites (#434)."""
    setup = """
        x = 3.14159
        n = 255
        w = 8
        name = "ab"
        # Codon's parser takes these as written.
        a = f"{x:.2f}|{x:e}|{x:g}|{x:.1%}|{x:010.3f}|{n:#x}|{n:08b}|{n:,}|{n:_x}"
        b = f"{n:>8}|{n:<8}|{n:^8}|{n:*^8}|{name!r:>8}|{name!s:^6}|{x=}|{x=:.2f}"
        # A sign, a field in the spec, a spec Codon does not know: rewritten.
        c = f"{x:+.2f}|{x: .2f}|{x:-.2f}|{n:+d}|{n:+}|{n:=+8d}|{x:^+10.2f}|{x:*<+10.2f}"
        d = f"{x:>{w}.2f}|{x:{w}.2f}|{n:{w}d}|{x:.{w}f}|{name:>{w}}|{name!r:{w}}"
        e = f"{x=:+.2f}|{n!r:+}|{x:q}|{x:{w}{w}}|{{lit}} {x:+.1f} {{lit}}"
        f = "{:+.2f}|{:>5}".format(x, x)
        g = "{:+.2f} {:+d} {!r:>5}".format(x, n, name)
        h = f'''multi
        {x:+.2f}
        line {n:{w}d}'''
        for text in (a, b, c, d, e, f, g, h):
            assert len(text) > 0
        k: str = f"{x:+.2f}"
        m: str = f"{x:>{w}}"
        assert len(k) + len(m) > 0
    """
    scenario, config, _ = _data_case(tmp_path, "", setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_fstring_specs_in_a_class_level_default_compile(tmp_path: Path) -> None:
    """The driver's copy of a class-level default is rewritten too (#434, #433)."""
    definitions = """
        W = 6


        class Labels:
            SIGNED: str = f"{3.5:+.1f}"
            PADDED: str = f"{7:>{W}}"
            TEXT: str = "{:+.1f}".format(2.5)
    """
    scenario, config, _ = _data_case(
        tmp_path, textwrap.dedent(definitions), "assert len(Labels().SIGNED) > 0\n"
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ('r: int = f"{3.5:+.2f}"', "int"),
        ('r: int = f"{3.5:>{4}.2f}"', "int"),
        ('r: float = "{:+.2f}".format(3.5)', "float"),
        ('r = f"{3.5:+.2f}" + 1', "str"),
    ],
    ids=["sign", "nested", "format", "operand"],
)
def test_a_rewritten_fstring_is_still_a_str(
    tmp_path: Path, line: str, message: str
) -> None:
    scenario, config, path = _data_case(tmp_path, "", line + "\n")
    error = _only_error(typecheck_scenario(scenario, config))
    assert message in error.message, error.format()
    assert error.line == _line_of(path, line)


@pytest.mark.parametrize(
    ("line", "hint"),
    [
        ("r = f\"{'{}'.format(3)}\"", "a brace in an expression"),
        ('r = f"{len({1, 2})}"', "a brace in an expression"),
        ('r = f"{ {1: 2}[1] :>3}"', "a brace in an expression"),
        ("r = f\"{f'{3}'}\"", "f-string inside an f-string"),
        ('r = f"{3.5!a:+.2f}"', "!a together with a format spec"),
    ],
    ids=["format", "set", "dict", "nested", "ascii"],
)
def test_an_fstring_codon_cannot_parse_is_refused_at_its_line(
    tmp_path: Path, line: str, hint: str
) -> None:
    scenario, config, path = _data_case(tmp_path, "", "pad = 1\n" + line + "\n")
    error = _only_error(typecheck_scenario(scenario, config))
    assert hint in error.message, error.format()
    assert error.line == _line_of(path, line)


@pytest.mark.parametrize(
    ("message", "column"),
    [
        ("syntax error, unexpected '.'", 4),
        ("syntax error, expecting <format_spec>.", 4),
    ],
)
def test_a_syntax_error_in_an_fstring_field_is_reported_at_its_line(
    tmp_path: Path, message: str, column: int
) -> None:
    """Codon puts it at line 1 of the field: -3 once the prelude is subtracted."""
    from autoware_carla_scenario.typecheck import check

    source = "a = 1\nb = 2\n\nr = f'{a} {b:+.2f}'\n"
    prelude = check.PRELUDE
    rel = "pkg/scenario.py"
    (tmp_path / "pkg").mkdir()
    (tmp_path / rel).write_text(prelude + source)
    ws = check._Workspace(tmp_path, None)
    ws.files[rel] = "/user/scenario.py"
    ws.origins[rel] = [1, 2, 3, 4]
    output = f"{rel}:1 ({column}): error: {message}\n"
    (diagnostic,) = check._parse_output(output, ws)
    assert diagnostic.path == "/user/scenario.py"
    assert diagnostic.line == 4
    assert diagnostic.message == message


def _parsed_at(tmp_path: Path, source: str, line: int, span: str) -> Any:
    """Codon's error at *line* of *source* (after the prelude), columns
    *span*, parsed for a checked copy that is *source* unchanged."""
    from autoware_carla_scenario.typecheck import check

    rel = "pkg/scenario.py"
    (tmp_path / "pkg").mkdir()
    (tmp_path / rel).write_text(check.PRELUDE + source)
    ws = check._Workspace(tmp_path, None)
    ws.files[rel] = "/user/scenario.py"
    ws.origins[rel] = list(range(1, source.count("\n") + 1))
    output = f"{rel}:{check._PRELUDE_LINES + line} ({span}): error: bad operand\n"
    (diagnostic,) = check._parse_output(output, ws)
    assert diagnostic.path == "/user/scenario.py"
    return diagnostic


@pytest.mark.parametrize(
    ("source", "line", "span", "expected"),
    [
        # Codon puts an error in an f-string field on the line after the
        # f-string, at the expression's column plus the field's length (#610).
        ("a = 1\ns = f\"{a} {a + 'x'}\"\nb = 2\n", 3, "18-25", (2, None)),
        # A field in a call's argument: the f-string's own line.
        ("a = 1\nprint(1,\n  f\"{len(a + 'x')}\")\nb = 2\n", 4, "21-28", (3, None)),
        # On the last line: past the end of the file.
        ("a = 1\ns = f\"{a + 'x'}\"\n", 3, "14-21", (2, None)),
        # A field on the second line of a triple-quoted f-string.
        ('a = 1\ns = f"""{a}\n  {a + \'x\'}"""\nb = 2\n', 3, "20-27", (3, None)),
        # An ordinary error on the line after an f-string with a field of the
        # same width stays where Codon puts it, column and all.
        ("a = 1\ns = f\"{a} {a + 'x'}\"\nb = a + 'x'\n", 3, "5-12", (3, 5)),
        # No f-string on the line before: left alone.
        ("a = 1\ns = 2\nb = a + 'x'\n", 3, "5-12", (3, 5)),
        # A comprehension's `for c in nnnn`, which Codon locates from its
        # `for` and Python's tree gives no position, is the line's own.
        (
            'abcd = 1\nnnnn = 5\ns = f"{abcd + 123456}"\n'
            "x = [cc * 22222222 for c in nnnn]\n",
            4,
            "20-33",
            (4, 20),
        ),
        # A column further from the field than the f-string's end is not
        # one of its fields.
        ('a = 1\ns = f"{a + 1}"\nb = 2\n', 3, "15-20", (3, 15)),
        # Implicitly concatenated f-strings (#616), at the columns Codon
        # reports: the first literal's column, plus the field's offset in
        # its own literal, its length and the expression's offset in it.
        ('n = 1\ns = "aaaa" f"b{n + \'x\'}"\nt = 2\n', 3, "15-22", (2, None)),
        ('n = 1\ns = ("a" f"b{n + \'x\'}")\nt = 2\n', 3, "16-23", (2, None)),
        ('n = 1\ns = "a" f"b{n} {n + \'x\'}"\nt = 2\n', 3, "19-26", (2, None)),
        (
            'n = 1\ns = "a" f"b{1 + (n + \'x\'):>4}" "z"\nt = 2\n',
            3,
            "29-36",
            (2, None),
        ),
        # On several lines: the field's own line.
        (
            'n = 1\ns = ("aaa"\n  "bb"\n     f"b{n + \'x\'}")\nt = 2\n',
            3,
            "16-23",
            (4, None),
        ),
        ('n = 1\ns = "a" \\\n    f"b{n + \'x\'}"\nt = 2\n', 3, "15-22", (3, None)),
        # The line's own error stays, column and all.
        (
            "n = 1\ns = \"aaaa\" f\"b{n + 'x'}\"\nt = [0, 1, 2, n + 'x']\n",
            3,
            "15-22",
            (3, 15),
        ),
        # So does one of a call on several lines that starts at that column,
        # whose width Codon counts across the line break.
        (
            'n = 1\ns = "" f"{n + 100000000000000}"\n'
            'result_value_long_name = g(\n            "x")\n',
            3,
            "26-45",
            (3, 26),
        ),
        # A form feed is no line break for Python.
        ('n = 1  # a\fb\ns = "a" f"b{n + \'x\'}"\nt = 2\n', 3, "15-22", (2, None)),
    ],
)
def test_an_error_in_an_fstring_field_is_moved_to_its_line(
    tmp_path: Path, source: str, line: int, span: str, expected: tuple[int, Any]
) -> None:
    diagnostic = _parsed_at(tmp_path, source, line, span)
    assert (diagnostic.line, diagnostic.column) == expected


def test_a_format_spec_on_a_bool_is_hinted(tmp_path: Path) -> None:
    line = 'r = f"{True:+d}"'
    scenario, config, path = _data_case(tmp_path, "", line + "\n")
    error = _only_error(typecheck_scenario(scenario, config))
    assert "has no attribute '__format__'" in error.message, error.format()
    assert "int(flag)" in error.message, error.format()
    assert error.line == _line_of(path, line)


def test_zip_with_strict_compiles(tmp_path: Path) -> None:
    setup = """
        xs = [1, 2]
        ys = [3.0, 4.0]
        total = 0.0
        for a, b in zip(xs, ys, strict=True):
            total += a + b
        for a, b in zip(
            xs,
            ys,
            strict=False,
        ):
            total += a * b
        assert total > 0.0
    """
    scenario, config, _ = _data_case(tmp_path, "", setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_zip_with_strict_still_needs_iterables(tmp_path: Path) -> None:
    line = "r = zip(5, strict=True)"
    scenario, config, path = _data_case(tmp_path, "", line + "\n")
    error = _only_error(typecheck_scenario(scenario, config))
    assert "'int' object has no attribute '__iter__'" in error.message, error.format()
    assert error.line == _line_of(path, line)


def test_builtins_codon_lacks_compile(tmp_path: Path) -> None:
    # #439: startswith / endswith with a tuple, format(), ascii().
    setup = """
        s = "abc"
        words = ["ab", "cd"]
        spec = ".2f"
        a = s.startswith(("a", "z"))
        b = s.endswith(("c", "y"), 1)
        c = words[0].startswith((words[1], "a"))
        d = (
            s.upper()
            .startswith(
                ("A", words[0].upper()),
            )
        )
        e = format(3.14159, ".2f") + format(7, "04d") + format(2.5, spec)
        f = format(3) + format(s) + ascii(s) + ascii(5)
        g = format(1 + 2, "x") + format(len(words))
        assert a and not b and c and d and len(e) + len(f) + len(g) > 0
    """
    scenario, config, _ = _data_case(tmp_path, "", setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("r = s.startswith((1, 2))", "'int' does not match expected type 'str'"),
        ("r = s.endswith(('a', 2))", "'int' does not match expected type 'str'"),
        ("r = s.startswith(('a', 'b'), 'x')", "'str' does not match expected"),
        ("r = format(s, 5)", "'int' does not match expected type 'str'"),
        ("r: int = format(5)", "r is assigned a str, but is declared int"),
        ("r: int = ascii(5)", "r is assigned a str, but is declared int"),
    ],
    ids=[
        "startswith-ints",
        "endswith-mixed",
        "startswith-start",
        "format-spec",
        "format-type",
        "ascii-type",
    ],
)
def test_the_rewritten_builtins_are_still_type_checked(
    tmp_path: Path, line: str, message: str
) -> None:
    scenario, config, path = _data_case(tmp_path, "", f's = "abc"\n{line}\n')
    error = _only_error(typecheck_scenario(scenario, config))
    assert message in error.message, error.format()
    assert error.line == _line_of(path, line)


@pytest.mark.parametrize(
    ("setup", "last", "hint"),
    [
        (
            'prefixes = ("a", "b")\nr = "abc".startswith(prefixes)',
            'r = "abc".startswith(prefixes)',
            "any(s.startswith(p) for p in prefixes)",
        ),
        (
            'd = {"a": 1}\nd.update(c=3)',
            "d.update(c=3)",
            'd.update({"k": v})',
        ),
        (
            "r = int.from_bytes(b'a', 'big')",
            "r = int.from_bytes(b'a', 'big')",
            "shifts and masks",
        ),
        (
            'r = "{a}".format_map({"a": 1})',
            'r = "{a}".format_map({"a": 1})',
            "write an f-string",
        ),
    ],
    ids=["startswith-tuple-name", "dict-update-keywords", "from-bytes", "format-map"],
)
def test_a_builtin_codon_lacks_is_refused_with_a_hint(
    tmp_path: Path, setup: str, last: str, hint: str
) -> None:
    scenario, config, path = _data_case(tmp_path, "", setup + "\n")
    error = _only_error(typecheck_scenario(scenario, config))
    assert hint in error.message, error.format()
    assert error.line == _line_of(path, last)


def test_one_argument_round_is_an_int(tmp_path: Path) -> None:
    # #585: Codon's round(x, n=0) always returns a float, Python's an int
    # without ndigits; the check writes int(round(x)).
    definitions = """
def pick(xs: list[int], x: float) -> int:
    return xs[round(x)]


def steps(x: float) -> int:
    n: int = round(x)
    total = 0
    for i in range(round(x, None)):
        total += i
    return n + total + round(number=x, ndigits=None)


def tenths(x: float) -> float:
    y: float = round(x)
    ws: list[float] = [float(round(x)), 1.5]
    return y + ws[0] + round(x, 1) + round(x, ndigits=2)
"""
    setup = 'logger.info("%d %d %f", pick([1, 2], 0.6), steps(2.4), tenths(1.25))\n'
    scenario, config, _ = _data_case(tmp_path, definitions, setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_rewritten_round_is_still_type_checked(tmp_path: Path) -> None:
    line = "r: str = round(2.5)"
    scenario, config, path = _data_case(tmp_path, "", line + "\n")
    error = _only_error(typecheck_scenario(scenario, config))
    assert "r is assigned an int, but is declared str" in error.message, error.format()
    assert error.line == _line_of(path, line)


def test_dict_update_with_a_dict_or_pairs_compiles(tmp_path: Path) -> None:
    setup = """
        d = {"a": 1}
        d.update({"c": 3})
        d.update([("e", 5)])
        assert len(d) == 3
    """
    scenario, config, _ = _data_case(tmp_path, "", setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_except_star_is_refused_at_its_line(tmp_path: Path) -> None:
    if sys.version_info < (3, 11):
        pytest.skip("except* needs Python 3.11")
    setup = """
        try:
            raise ValueError("x")
        except* ValueError as eg:
            n = len(eg.exceptions)
    """
    scenario, config, path = _data_case(tmp_path, "", setup)
    result = typecheck_scenario(scenario, config)
    assert not result.ok, result.format()
    error = result.diagnostics[0]
    assert "exception groups" in error.message, result.format()
    assert "use `except`" in error.message, result.format()
    assert error.line == _line_of(path, "except* ValueError")


def test_the_439_rewrites_compose_with_fstring_specs_and_class_constants(
    tmp_path: Path,
) -> None:
    definitions = """
class Names:
    A = "ab"
    B = "cd"
    HIT = "abc".startswith((A, B))
    LABEL = f"{len(A):+d}"

    def check(self, s: str = "abx") -> bool:
        return s.startswith((Names.A, Names.B))
"""
    setup = """
        n = 5
        s = "abc"
        signed = format(n, "+d") + f"{n:+d}" + format(2.5, "+.1f")
        inside = f"{s.startswith(('a', 'z'))}-{format(n, '>4')}-{n:+d}"
        names = Names()
        assert names.check() and Names.HIT and len(signed) + len(inside) > 0
        logger.info("%s %s %s", signed, inside, Names.LABEL)
    """
    scenario, config, _ = _data_case(tmp_path, definitions, setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_a_bytes_method_codon_lacks_is_refused_with_a_hint(tmp_path: Path) -> None:
    setup = "r = b'ab'.hex()\n"
    scenario, config, path = _data_case(tmp_path, "", setup)
    error = _only_error(typecheck_scenario(scenario, config))
    assert "already text" in error.message, error.format()
    assert error.line == _line_of(path, "r = b'ab'.hex()")


def test_a_class_with_only_lt_sorts(tmp_path: Path) -> None:
    setup = """
        items = [ByLt(2), ByLt(1), ByLt(3)]
        ordered = sorted(items)
        items.sort()
        items.sort(reverse=True)
        keyed = sorted([ByLt(2), ByLt(1)], key=lambda v: v)
        assert ordered[0].x == 1 and min(items).x == 1 and max(items).x == 3
        assert ByLt(2) > ByLt(1) and keyed[0].x == 1
    """
    scenario, config, _ = _data_case(tmp_path, _REWRITE_CLASSES, setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_total_ordering_gives_every_comparison(tmp_path: Path) -> None:
    setup = """
        les = sorted([ByLe(2), ByLe(1)])
        lts = sorted([ByLtAndEq(2), ByLtAndEq(1)])
        assert ByLe(1) < ByLe(2) and ByLe(2) > ByLe(1) and ByLe(1) >= ByLe(1)
        assert ByLtAndEq(1) <= ByLtAndEq(1) and ByLtAndEq(2) >= ByLtAndEq(1)
        assert les[0].x == 1 and lts[0].x == 1
    """
    scenario, config, _ = _data_case(tmp_path, _REWRITE_CLASSES, setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_eq_and_hash_class_in_a_set_and_dict_compiles(tmp_path: Path) -> None:
    definitions = """
class Key:
    k: int

    def __init__(self, k: int) -> None:
        self.k = k

    def __eq__(self, o: object) -> bool:
        return isinstance(o, Key) and self.k == o.k

    def __hash__(self) -> int:
        return hash(self.k)

    def __lt__(self, o: "Key") -> bool:
        return self.k < o.k
"""
    setup = """
        seen = {Key(1), Key(1), Key(2)}
        by_key = {Key(3): 1}
        by_key[Key(3)] = 2
        ordered = sorted([Key(2), Key(1)])
        assert len(seen) == 2 and by_key[Key(3)] == 2 and Key(1) != Key(2)
        assert ordered[0].k == 1 and Key(1) in seen
    """
    scenario, config, _ = _data_case(tmp_path, definitions, setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_namedtuple_with_a_field_string_compiles(tmp_path: Path) -> None:
    definitions = """
import collections
from collections import namedtuple

Pt = namedtuple("Pt", "x y")
Pc = collections.namedtuple("Pc", "a, b")
"""
    setup = """
        p = Pt(x=1, y=2)
        q = Pc(3, 4)
        assert p.x + p[1] + q.a + q.b == 10
    """
    scenario, config, _ = _data_case(tmp_path, definitions, setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_namedtuple_with_non_literal_fields_is_refused_with_a_hint(
    tmp_path: Path,
) -> None:
    definitions = """
from collections import namedtuple

FIELDS = ["x", "y"]
Pt = namedtuple("Pt", FIELDS)
"""
    scenario, config, path = _data_case(tmp_path, definitions, "r = 1\n")
    error = _only_error(typecheck_scenario(scenario, config))
    assert "written in the call" in error.message, error.format()
    assert error.line == _line_of(path, 'Pt = namedtuple("Pt", FIELDS)')


def test_sorting_a_class_with_no_ordering_is_refused_with_a_hint(
    tmp_path: Path,
) -> None:
    line = "r = sorted([Unordered(1), Unordered(0)])"
    scenario, config, path = _data_case(tmp_path, _REWRITE_CLASSES, line + "\n")
    error = _only_error(typecheck_scenario(scenario, config))
    assert "define __lt__" in error.message, error.format()
    assert error.line == _line_of(path, line)


def test_total_ordering_on_a_nested_class_is_refused(tmp_path: Path) -> None:
    definitions = """
from functools import total_ordering


def make():
    @total_ordering
    class Inner:
        def __lt__(self, other: "Inner") -> bool:
            return True

    return Inner
"""
    scenario, config, path = _data_case(tmp_path, definitions, "r = make()\n")
    error = _only_error(typecheck_scenario(scenario, config))
    assert "not at module level" in error.message, error.format()
    assert error.line == _line_of(path, "@total_ordering")


def test_a_lambda_capturing_a_default_in_a_comprehension_compiles(
    tmp_path: Path,
) -> None:
    setup = """
        fs = [lambda x, k=k: x + k for k in range(3)]
        gs = {j: (lambda x, k=j, m=len([1]): x + k + m) for j in range(2)}
        plain = lambda a, b=2: a + b
        assert fs[1](1) == 2 and gs[1](1) == 3 and plain(1) == 3
    """
    scenario, config, _ = _data_case(tmp_path, "", setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_a_lambda_capturing_a_default_is_still_typed(tmp_path: Path) -> None:
    setup = """
        fs = [lambda x, k=k: x + k for k in range(3)]
        r = fs[0]("x")
    """
    scenario, config, path = _data_case(tmp_path, "", setup)
    error = _only_error(typecheck_scenario(scenario, config))
    assert "for +: 'str' and 'int'" in error.message, error.format()
    call = _line_of(path, 'r = fs[0]("x")')
    assert any(f"scenario.py:{call}]" in frame for frame in error.trace), error.format()


def test_a_nested_def_default_reading_an_enclosing_local_compiles(
    tmp_path: Path,
) -> None:
    # Codon refuses a bare local of the enclosing function as a default
    # ("name 'name' cannot be captured"); the check writes `(name,)[0]` (#612).
    setup = """
        names = ["a", "b"]
        scale = 2
        for name in names:
            def log_it(world: carla.World, n: str = name) -> None:
                logger.info("%s", n)
            self.register_post_tick(log_it)
        def f(k: int = scale, *, label: str = cfg.name, tag=name) -> int:
            def g(m: int = k) -> int:
                return m * 2
            return g() + len(label) + len(tag)
        logger.info("%d %d", f(), f(3, label="x"))
    """
    scenario, config, _ = _write_case(tmp_path, setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_a_nested_def_default_is_still_typed(tmp_path: Path) -> None:
    setup = """
        name = "a"
        def f(k: int = name) -> int:
            return k * 2
        logger.info("%d", f())
    """
    scenario, config, _ = _write_case(tmp_path, setup)
    error = _only_error(typecheck_scenario(scenario, config))
    assert "'str' does not match expected type 'int'" in error.message, error.format()


def test_an_fstring_in_a_decorator_compiles(tmp_path: Path) -> None:
    # Codon reads it as "'str' object has no attribute '__prefix_f__'" (#616).
    extra = """


N = 3


def tag(label: str, extra: str = ""):
    def deco(f):
        return f
    return deco


@tag(f"lane-{N}", extra=f"x{N!r:>4}")
@tag(f"plain")
def plain() -> int:
    return 1


def helper(k: int) -> int:
    @tag(f"inner-{k}")
    def inner() -> int:
        return k
    return inner()
"""
    scenario, config, _ = _write_case(
        tmp_path, 'logger.info("%d %d", plain(), helper(2))\n', extra=extra
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


@pytest.mark.parametrize("call", ["fs[0](1, k=5)", "fs[0](1, 5)"])
def test_passing_a_captured_lambda_default_is_refused_with_a_hint(
    tmp_path: Path, call: str
) -> None:
    setup = f"""
        fs = [lambda x, k=k: x + k for k in range(3)]
        r = {call}
    """
    scenario, config, path = _data_case(tmp_path, "", setup)
    error = _only_error(typecheck_scenario(scenario, config))
    assert "binds that default once" in error.message, error.format()
    assert error.line == _line_of(path, f"r = {call}")


# ---------------------------------------------------------------------------
# Codon-only refusals of valid Python: a hint and the line (#379)
# ---------------------------------------------------------------------------

_POINT = "@dataclass\nclass Pt:\n    x: int\n    y: int\n"

_PROTOCOL = """
class HasArea(typing.Protocol):
    def area(self) -> float: ...


class Sq:
    side: float

    def __init__(self, side: float) -> None:
        self.side = side

    def area(self) -> float:
        return self.side * self.side


class SqBased(HasArea):
    side: float

    def __init__(self, side: float) -> None:
        self.side = side

    def area(self) -> float:
        return self.side * self.side


def show(h: HasArea) -> float:
    return h.area()


def total(hs: list[HasArea]) -> float:
    return sum([h.area() for h in hs])


def total_of_sq(hs: list[SqBased]) -> float:
    return sum([h.area() for h in hs])


def two(a: int, b: int) -> int:
    return a + b
"""


@pytest.mark.parametrize(
    ("definitions", "subject", "pattern", "hint"),
    [
        ("", "x: int | str = 3", "int(n)", "isinstance(v, Pt)"),
        (_POINT, "x = Pt(1, 2)", "Pt()", "isinstance(v, Pt)"),
        (_POINT, "x = Pt(1, 2)", "Pt(x=a, y=b)", "isinstance(v, Pt)"),
        ("", 'x = {"k": 1}', '{"k": v}', "test `'k' in d`"),
        ("", 'x = {"k": 1}', "{**r}", "test `'k' in d`"),
        ("", 'x = {"k": [1, 2]}', '{"k": [1, b]}', "test `'k' in d`"),
        ("", "x = [1, 2]", "[a, *rest]", "test len(xs)"),
        ("", "x = [1, 2]", "[a, b] as whole", "match the subject unnamed"),
    ],
    ids=[
        "class",
        "own-class",
        "dataclass",
        "mapping",
        "mapping-rest",
        "mapping-nested",
        "star",
        "as",
    ],
)
def test_a_match_pattern_codon_cannot_bind_is_refused_with_a_hint(
    tmp_path: Path, definitions: str, subject: str, pattern: str, hint: str
) -> None:
    setup = f"""
        {subject}
        match x:
            case {pattern}:
                logger.info("%s", "hit")
            case _:
                pass
    """
    scenario, config, path = _data_case(tmp_path, definitions, setup)
    error = _only_error(typecheck_scenario(scenario, config))
    assert hint in error.message, error.format()
    assert error.line == _line_of(path, f"case {pattern}:")


def test_the_match_patterns_codon_compiles_still_compile(tmp_path: Path) -> None:
    setup = """
        n = 3
        d = {"k": 1}
        pair = [1, 2]
        match n:
            case 1 | 2:
                logger.info("%s", "small")
            case _ if n > 5:
                logger.info("%s", "big")
            case _:
                pass
        match d:
            case {"k": 1}:
                logger.info("%s", "one")
            case _:
                pass
        match d:
            case {"k": 1 | 2}:
                logger.info("%s", "one or two")
            case {"k": [1, 2]}:
                logger.info("%s", "pair")
            case _:
                pass
        match pair:
            case [a, b]:
                logger.info("%d", a + b)
            case _:
                pass
        match n:
            case int():
                logger.info("%d", n)
            case _:
                pass
    """
    scenario, config, _ = _data_case(tmp_path, _POINT, setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_isinstance_and_attribute_reads_replace_a_capture_pattern(
    tmp_path: Path,
) -> None:
    setup = """
        v: int | str = 3
        if isinstance(v, int):
            logger.info("%d", v)
        d = {"k": 1}
        if "k" in d:
            logger.info("%d", d["k"])
        xs = [1, 2, 3]
        if len(xs) > 0:
            logger.info("%d %d", xs[0], len(xs[1:]))
        p = Pt(1, 2)
        if isinstance(p, Pt) and p.x < p.y:
            logger.info("%d", p.y)
    """
    scenario, config, _ = _data_case(tmp_path, _POINT, setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


@pytest.mark.parametrize(
    ("call", "hint"),
    [
        ("show(Sq(1.0))", "is a typing.Protocol"),
        ("total([Sq(1.0)])", "is a typing.Protocol"),
        ("two(*[1, 2])", "pass the elements"),
        ('two(**{"a": 1, "b": 2})', "pass the keywords"),
    ],
    ids=["protocol", "protocol-list", "star-list", "star-star-dict"],
)
def test_protocol_and_unpacking_refusals_say_what_to_write(
    tmp_path: Path, call: str, hint: str
) -> None:
    scenario, config, path = _data_case(
        tmp_path, _PROTOCOL, f'logger.info("%s", {call})\n'
    )
    error = _only_error(typecheck_scenario(scenario, config))
    assert hint in error.message, error.format()
    line = _line_of(path, call)
    assert error.line == line or any(
        f"scenario.py:{line}]" in frame for frame in error.trace
    ), error.format()


def test_the_documented_protocol_and_unpacking_spellings_compile(
    tmp_path: Path,
) -> None:
    setup = """
        logger.info("%f", show(SqBased(1.0)))
        logger.info("%f", total_of_sq([SqBased(2.0)]))
        logger.info("%d", two(*(1, 2)))
        xs = [1, 2]
        logger.info("%d", two(xs[0], xs[1]))
    """
    scenario, config, _ = _data_case(tmp_path, _PROTOCOL, setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_a_class_mismatch_that_is_no_protocol_gets_no_protocol_hint(
    tmp_path: Path,
) -> None:
    definitions = (
        _PROTOCOL
        + """

class Other:
    pass


def take(o: Other) -> int:
    return 1
"""
    )
    scenario, config, _ = _data_case(
        tmp_path, definitions, 'logger.info("%d", take(Sq(1.0)))\n'
    )
    error = _only_error(typecheck_scenario(scenario, config))
    assert "does not match expected type" in error.message, error.format()
    assert "Protocol" not in error.message, error.format()


def test_total_ordering_composes_with_the_stubbed_functools_names(
    tmp_path: Path,
) -> None:
    imports = (
        "from functools import total_ordering, reduce, lru_cache\nimport operator\n"
    )
    helper = (
        "@total_ordering\nclass Rank:\n    n: int\n\n"
        "    def __init__(self, n: int):\n        self.n = n\n\n"
        "    def __lt__(self, other: Rank) -> bool:\n        return self.n < other.n\n\n"
        "    def __eq__(self, other: Rank) -> bool:\n        return self.n == other.n\n\n\n"
        "@lru_cache(maxsize=4)\ndef total(xs: list[int]) -> int:\n"
        "    return reduce(operator.add, xs)\n"
    )
    call = "logger.info('%d %d', total([1, 2]), 1 if Rank(1) <= Rank(2) else 0)\n"
    scenario, config, _ = _write_case(
        tmp_path,
        _TIMEOUT_SETUP + call,
        "\n\n" + helper,
        header=_stdlib_header(imports),
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


# ---------------------------------------------------------------------------
# A context manager's __exit__ and a table of functions (#435, #436)
# ---------------------------------------------------------------------------

_CONTEXT_MANAGER = """
class Ctx:
    n: int

    def __init__(self) -> None:
        self.n = 1

    def __enter__(self) -> "Ctx":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object | None,
    ) -> bool | None:
        if exc_type is not None:
            logger.info("failed: %s", exc_type.__name__)
        self.n = 2
        return None
"""


def test_a_context_manager_with_the_usual_exit_compiles(tmp_path: Path) -> None:
    setup = """
        with Ctx() as c:
            n = c.n
        logger.info("%d %d", n, c.n)
    """
    scenario, config, _ = _data_case(tmp_path, _CONTEXT_MANAGER, setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


# -- a class constant read in the class scope (#433) ---------------------------

_CLASS_CONSTANTS = """
import functools

SCALE = 100


class Helper:
    LIMIT = 5
    STEP: typing.Final = 2
    DOUBLE = LIMIT * 2
    TOTAL = DOUBLE + STEP

    def f(self, n: int = LIMIT, *, m: int = STEP) -> int:
        return n + m

    def g(self, n: int = SCALE) -> int:
        return n + Helper.LIMIT


class Color(enum.Enum):
    RED = 1
    BLUE = 2


class Painter:
    DEFAULT = Color.RED
    ALSO = DEFAULT

    def paint(self, color: Color = DEFAULT) -> Color:
        return color


@dataclass(order=True, frozen=True)
class Box:
    SIZE = 5
    width: int = SIZE
    height: int = SIZE + 1

    def grown(self, by: int = SIZE) -> int:
        return self.width + by


class Pt(typing.NamedTuple):
    x: int = 1
    ORIGIN = 7
    FAR = ORIGIN * 2

    def shift(self, by: int = ORIGIN) -> int:
        return self.x + by


class Outer:
    class Inner:
        A = 1
        B = A + 1


class Mode(enum.IntEnum):
    LOW = 1
    HIGH = LOW
    _BASE = 3
    START = _BASE

    def plus(self, n: int = _BASE) -> int:
        return n + 1


@functools.total_ordering
class Rank:
    n: int
    STEP = 2

    def __init__(self, n: int = STEP):
        self.n = n

    def __lt__(self, other: Rank) -> bool:
        return self.n < other.n

    def __eq__(self, other: Rank) -> bool:
        return self.n == other.n
"""


def test_a_class_constant_read_in_the_class_scope_compiles(tmp_path: Path) -> None:
    setup = """
        h = Helper()
        total = Helper.DOUBLE + Helper.TOTAL + h.f() + h.g()
        painted = Painter().paint() == Color.RED and Painter.ALSO == Color.RED
        box = Box()
        bigger = sorted([Box(7), box])
        logger.info("%d %d %d", total, box.grown(), bigger[0].width)
        pt = Pt()
        logger.info("%d %d", pt.shift(), Pt.FAR + Outer.Inner.B)
        logger.info("%d %d", Mode.HIGH.plus(), Mode.START.value)
        ranks = sorted([Rank(), Rank(1)])
        logger.info("%d %s", ranks[0].n, Rank() >= Rank(3))
        assert painted
    """
    scenario, config, _ = _data_case(tmp_path, _CLASS_CONSTANTS, setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_exit_parameters_unannotated_or_defaulted_compile(tmp_path: Path) -> None:
    definitions = """
class Plain:
    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        return False


class Half:
    def __enter__(self):
        return self

    def __exit__(self, et: object, ev: object = None, tb: object | None = None):
        pass


class Star:
    def __enter__(self):
        return self

    def __exit__(self, *args: object):
        pass
"""
    setup = """
        with Plain() as p, Half() as h, Star() as s:
            logger.info("%s", "in")
    """
    scenario, config, _ = _data_case(tmp_path, definitions, setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_the_body_of_an_exit_is_still_typed(tmp_path: Path) -> None:
    definitions = _CONTEXT_MANAGER.replace("self.n = 2", 'self.n = "two"')
    setup = """
        with Ctx() as c:
            n = c.n
    """
    scenario, config, path = _data_case(tmp_path, definitions, setup)
    error = _only_error(typecheck_scenario(scenario, config))
    assert "str" in error.message, error.format()
    assert error.line == _line_of(path, 'self.n = "two"')


def test_an_exception_branch_of_an_exit_is_still_typed(tmp_path: Path) -> None:
    # `exc` keeps its BaseException | None annotation, so the branch is live.
    definitions = _CONTEXT_MANAGER.replace(
        "exc_type.__name__", "exc.no_such_attribute"
    ).replace("if exc_type is not None:", "if exc is not None:")
    setup = """
        with Ctx() as c:
            n = c.n
    """
    scenario, config, path = _data_case(tmp_path, definitions, setup)
    error = _only_error(typecheck_scenario(scenario, config))
    assert "no_such_attribute" in error.message, error.format()


_TABLE = """
def inc(x: int) -> int:
    return x + 1


def dbl(x: int) -> int:
    return x * 2
"""


def test_a_table_of_functions_annotated_callable_compiles(tmp_path: Path) -> None:
    definitions = (
        _TABLE
        + """

def table(k: int) -> dict[str, typing.Callable[[int], int]]:
    return {"inc": inc, "dbl": dbl, "neg": lambda x: -x}


def nested(k: int) -> int:
    ops: dict[str, list[typing.Callable[[int], int]]] = {
        "a": [inc, lambda x: x * 3],
        "b": [dbl],
    }
    one: list[typing.Callable[[int], int]] = [lambda x: x + k]
    return ops["a"][1](2) + ops["b"][0](1) + one[0](1)
"""
    )
    setup = """
        fs: list[typing.Callable[[int], int]] = [inc, dbl]
        d: dict[str, typing.Callable[[int], int]] = {"a": inc, "b": lambda x: x - 1}
        fns: list[typing.Callable[[float, str], None]] = [
            lambda v, s: None,
            lambda v, s: logger.info("%s", s),
        ]
        fns[1](1.0, "x")
        logger.info("%d %d %d", fs[1](3), d["b"](3), table(1)["neg"](2) + nested(1))
    """
    scenario, config, _ = _data_case(tmp_path, definitions, setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_a_table_of_functions_is_still_typed(tmp_path: Path) -> None:
    setup = """
        fs: list[typing.Callable[[int], int]] = [inc, dbl]
        r = fs[1]("x")
    """
    scenario, config, path = _data_case(tmp_path, _TABLE, setup)
    error = _only_error(typecheck_scenario(scenario, config))
    assert error.line == _line_of(path, 'r = fs[1]("x")') or any(
        f"scenario.py:{_line_of(path, 'r = fs[1](')}]" in f for f in error.trace
    ), error.format()


def test_a_table_of_functions_with_the_wrong_signature_is_refused(
    tmp_path: Path,
) -> None:
    setup = """
        fs: list[typing.Callable[[str], int]] = [inc, dbl]
    """
    scenario, config, _ = _data_case(tmp_path, _TABLE, setup)
    assert not typecheck_scenario(scenario, config).ok


@pytest.mark.parametrize(
    "line",
    [
        "fs = [inc, dbl]",
        'fs = {"a": inc, "b": dbl}',
        "fs = [lambda x: x + 1, lambda x: x * 2]",
    ],
)
def test_an_unannotated_table_of_functions_is_refused_with_a_hint(
    tmp_path: Path, line: str
) -> None:
    scenario, config, path = _data_case(
        tmp_path, _TABLE, f"        {line}\n        logger.info('%d', len(fs))\n"
    )
    error = _only_error(typecheck_scenario(scenario, config))
    assert "dispatch table" in error.message, error.format()
    assert "if/elif" in error.message, error.format()
    assert error.line == _line_of(path, line)


def test_a_table_of_bound_methods_is_refused_with_a_hint(tmp_path: Path) -> None:
    definitions = """
class Ops:
    def inc(self, x: int) -> int:
        return x + 1

    def dbl(self, x: int) -> int:
        return x * 2
"""
    setup = """
        ops = Ops()
        fs: list[typing.Callable[[int], int]] = [ops.inc, ops.dbl]
    """
    scenario, config, path = _data_case(tmp_path, definitions, setup)
    error = _only_error(typecheck_scenario(scenario, config))
    assert "dispatch table" in error.message and "bound method" in error.message
    assert error.line == _line_of(path, "fs: list")


def test_a_table_mixing_a_bound_method_and_a_function_gets_the_hint(
    tmp_path: Path,
) -> None:
    # The bound method is not wrapped: Codon's raw text names a Callable.
    definitions = (
        _TABLE + "\nclass K:\n    def m(self, x: int) -> int:\n        return x\n"
    )
    setup = """
        k = K()
        fs: list[typing.Callable[[int], int]] = [k.m, inc]
    """
    scenario, config, _ = _data_case(tmp_path, definitions, setup)
    error = _only_error(typecheck_scenario(scenario, config))
    assert "dispatch table" in error.message, error.format()


def test_a_function_of_another_signature_is_refused_with_a_hint(
    tmp_path: Path,
) -> None:
    definitions = _TABLE + "\ndef two(x: int, y: int) -> int:\n    return x + y\n"
    setup = """
        fs: list[typing.Callable[[int], int]] = [inc, two]
    """
    scenario, config, path = _data_case(tmp_path, definitions, setup)
    error = _only_error(typecheck_scenario(scenario, config))
    assert "signature" in error.message, error.format()
    assert error.line == _line_of(path, "fs: list")


_REGISTRY = """
class Registry:
    handlers: list[typing.Callable[[float], float]]
    named: dict[str, typing.Callable[[float, str], float]]
    latest: typing.Callable[[float], float]
    ticker: typing.Callable[[], None]

    def __init__(self) -> None:
        self.handlers = []
        self.named = {}
        self.latest = lambda v: v
        self.ticker = noop

    def register(self, fn: typing.Callable[[float], float]) -> None:
        self.handlers.append(fn)

    def register_named(
        self, name: str, fn: typing.Callable[[float, str], float]
    ) -> None:
        self.named[name] = fn

    def keep(self, fn: typing.Callable[[float], float]) -> None:
        self.latest = fn

    def on_tick(self, fn: typing.Callable[[], None]) -> None:
        self.ticker = fn

    def call(self, v: float) -> float:
        total = self.latest(v) + self.named["a"](v, "x")
        for handler in self.handlers:
            total += handler(v)
        self.ticker()
        return total


def noop() -> None:
    pass


def inc(v: float) -> float:
    return v + 1.0


def tag(v: float, s: str) -> float:
    return v * 2.0
"""


def test_a_callback_registry_compiles(tmp_path: Path) -> None:
    # A Callable parameter stored in a Callable attribute used to abort Codon
    # with std::bad_alloc (#488): the parameter's annotation was dropped, the
    # attribute's kept.
    setup = """
        reg = Registry()
        reg.register(inc)
        reg.register(lambda v: v - 1.0)
        reg.register_named("a", tag)
        reg.register_named("b", lambda v, s: v)
        reg.keep(lambda v: v + 3.0)
        reg.on_tick(noop)
        self.register_fail_condition(TimeoutCondition(reg.call(1.0), label="t"))
    """
    scenario, config, _ = _data_case(tmp_path, _REGISTRY, setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


@pytest.mark.parametrize(
    "call",
    [
        "reg.register(bad)",
        "reg.register(lambda v, w: v)",
        "reg.register_named('a', inc)",
        "reg.keep(bad)",
        "reg.on_tick(inc)",
    ],
)
def test_a_callback_of_the_wrong_signature_is_refused(
    tmp_path: Path, call: str
) -> None:
    definitions = _REGISTRY + "\ndef bad(v: str) -> str:\n    return v\n"
    setup = f"""
        reg = Registry()
        {call}
        self.register_fail_condition(TimeoutCondition(1.0, label="t"))
    """
    scenario, config, path = _data_case(tmp_path, definitions, setup)
    error = _only_error(typecheck_scenario(scenario, config))
    # At the call, not "failed without a diagnostic".
    assert error.line == _line_of(path, call.split("(")[0]), error.format()
    assert "without a diagnostic" not in error.message, error.format()


_OPTIONAL_CALLBACKS = """
class Hooks:
    on_value: typing.Callable[[int], int] | None = None
    on_text: typing.Optional[typing.Callable[[str], str]]
    on_done: typing.Callable[[], None] | None

    def __init__(
        self,
        on_text: typing.Optional[typing.Callable[[str], str]] = None,
        on_done: typing.Callable[[], None] | None = None,
    ) -> None:
        self.on_text = on_text
        self.on_done = on_done

    def set_value(self, cb: typing.Callable[[int], int] | None) -> None:
        self.on_value = cb

    def apply(self, v: int) -> int:
        if self.on_value is not None:
            return self.on_value(v)
        return v

    def text(self, v: str) -> str:
        if self.on_text is not None:
            return self.on_text(v)
        return v

    def done(self) -> None:
        if self.on_done is not None:
            self.on_done()


def noop() -> None:
    pass


def inc(v: int) -> int:
    return v + 1


def shout(v: str) -> str:
    return v + "!"


def bad(v: str) -> str:
    return v
"""


def test_an_optional_callback_attribute_compiles(tmp_path: Path) -> None:
    # `cb: Callable[...] | None` failed at import, and an Optional attribute
    # filled from a parameter refused a lambda with "%_lambda_N" (#495).
    setup = """
        plain = Hooks()
        plain.set_value(None)
        named = Hooks(shout, noop)
        named.set_value(inc)
        lam = Hooks(on_text=lambda s: s + "?")
        lam.set_value(lambda v: v * 2)
        none = Hooks(None, None)
        total = plain.apply(1) + named.apply(1) + lam.apply(1) + none.apply(1)
        self.register_fail_condition(TimeoutCondition(float(total), label="t"))
        logger.info("%s %s", named.text("a"), lam.text("b"))
        named.done()
        lam.done()
    """
    scenario, config, _ = _data_case(tmp_path, _OPTIONAL_CALLBACKS, setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


@pytest.mark.parametrize(
    "call",
    [
        "Hooks().set_value(bad)",
        "Hooks().set_value(lambda v, w: v)",
        "Hooks(on_text=inc)",
        "Hooks(on_done=inc)",
    ],
)
def test_an_optional_callback_of_the_wrong_signature_is_refused(
    tmp_path: Path, call: str
) -> None:
    setup = f"""
        {call}
        self.register_fail_condition(TimeoutCondition(1.0, label="t"))
    """
    scenario, config, path = _data_case(tmp_path, _OPTIONAL_CALLBACKS, setup)
    error = _only_error(typecheck_scenario(scenario, config))
    assert error.line == _line_of(path, call.split("(")[0]), error.format()
    assert "without a diagnostic" not in error.message, error.format()
    assert "%_lambda" not in error.message, error.format()


def test_none_for_a_required_callback_is_still_refused(tmp_path: Path) -> None:
    # The stored optional parameter is the plain `Callable` for Codon, which
    # converts `None` to it; a required `Callable` parameter stays refused for
    # `None` by the mypy pass (#495).
    definitions = """
class Required:
    def __init__(self, cb: typing.Callable[[int], int]) -> None:
        self.cb = cb
"""
    setup = """
        Required(None)
        self.register_fail_condition(TimeoutCondition(1.0, label="t"))
    """
    scenario, config, path = _data_case(tmp_path, definitions, setup)
    error = _only_error(typecheck_scenario(scenario, config))
    assert error.line == _line_of(path, "Required(None)"), error.format()


def test_an_optional_callback_called_without_a_none_test_compiles(
    tmp_path: Path,
) -> None:
    # Codon only unwraps the Optional when the call runs; nothing to refuse.
    definitions = """
class Bare:
    cb: typing.Callable[[int], int] | None = None

    def go(self, v: int) -> int:
        return self.cb(v)
"""
    scenario, config, _ = _data_case(tmp_path, definitions, "x = Bare().go(1)\n")
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_a_method_body_reading_a_name_of_the_class_scope_is_not_rewritten(
    tmp_path: Path,
) -> None:
    # In Python the body reads the module's LIMIT: it must keep compiling
    # against it (a str here), not against the class attribute (an int).
    definitions = """
LIMIT = "module"


class Helper:
    LIMIT = 5
    DOUBLE = LIMIT * 2

    def label(self) -> str:
        return LIMIT + "!"
"""
    scenario, config, _ = _data_case(
        tmp_path, definitions, "text = Helper().label() + str(Helper.DOUBLE)\n"
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_an_annotated_class_attribute_read_in_the_class_scope_compiles(
    tmp_path: Path,
) -> None:
    definitions = """
@dataclass(order=True)
class D:
    X: int = 5
    NEG: typing.Final[int] = -2
    y: int = X
    z: int = NEG

    def f(self, n: int = X, m: int = NEG**2) -> int:
        return n + m


class Plain:
    LIMIT: int = 3
    NAME: str = "a"

    def g(self, n: int = LIMIT, s: str = NAME) -> int:
        return n
"""
    setup = """
        d = D()
        logger.info("%d %d %d %d", d.y, d.z, d.f(), Plain().g())
    """
    scenario, config, _ = _data_case(tmp_path, definitions, setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_an_annotated_computed_class_attribute_read_in_the_class_scope_is_refused(
    tmp_path: Path,
) -> None:
    definitions = """
class Helper:
    LIMIT: int = 2 * 3

    def f(self, n: int = LIMIT) -> int:
        return n
"""
    scenario, config, path = _data_case(tmp_path, definitions, "x = Helper().f()\n")
    error = _only_error(typecheck_scenario(scenario, config))
    assert "`LIMIT: Final = ...`" in error.message, error.format()
    assert error.line == _line_of(path, "def f(self, n: int = LIMIT)")


def test_an_enum_alias_of_a_negative_member_is_skipped_by_iteration(
    tmp_path: Path,
) -> None:
    definitions = """
class Level(enum.IntEnum):
    LOW = -1
    ALIAS = LOW
    HIGH = 2
"""
    setup = """
        total = 0
        for level in Level:
            total += level.value
        logger.info("%d", total)
    """
    scenario, config, _ = _data_case(tmp_path, definitions, setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_a_namedtuple_field_default_reading_a_class_constant_is_refused(
    tmp_path: Path,
) -> None:
    definitions = """
class Pt(typing.NamedTuple):
    LIM = 5
    x: int = LIM
"""
    scenario, config, path = _data_case(tmp_path, definitions, "p = Pt()\n")
    error = _only_error(typecheck_scenario(scenario, config))
    assert "NamedTuple" in error.message, error.format()
    assert error.line == _line_of(path, "x: int = LIM")


def test_a_class_constant_assigned_again_is_refused(tmp_path: Path) -> None:
    definitions = """
class Helper:
    X = 1
    Y = X
    X = 2
"""
    scenario, config, path = _data_case(tmp_path, definitions, "y = Helper.Y\n")
    error = _only_error(typecheck_scenario(scenario, config))
    assert "assigned again later" in error.message, error.format()
    assert error.line == _line_of(path, "Y = X")


def test_a_class_constant_in_a_class_in_a_function_is_refused(tmp_path: Path) -> None:
    definitions = """
def make():
    class Local:
        Q = 1
        R = Q * 2

    return Local
"""
    scenario, config, path = _data_case(tmp_path, definitions, "k = make()\n")
    error = _only_error(typecheck_scenario(scenario, config))
    assert "defined in a function" in error.message, error.format()
    assert error.line == _line_of(path, "R = Q * 2")


def test_fstrings_in_lambdas_that_capture_compile(tmp_path: Path) -> None:
    """Every form of #470: Codon cannot see a captured name in a lambda's f-string."""
    setup = """
        p = cfg.name
        x = 3.14159
        a = lambda v: f"{v}{p}"
        b = sorted(["b", "a"], key=lambda e: f"{e}{p}")
        c = lambda e: "{}{}".format(e, p)
        d = lambda v, lab=cfg.name: f"{v}{lab}"
        e = lambda v, lab=f"{x}": f"{v}{lab}"
        f = lambda: f"{x=:.2f}"
        g = (lambda lab: lambda v: f"{lab}")("a")
        h = lambda v: f"{v:+.2f}|{x:>10.3f}|{p!r}|{x=}|{{lit}}"
        i = lambda v: [f"{k}{p}" for k in range(v)]

        def inner(name):
            return (lambda v: f"{v}{name}")(1)

        r: str = a(1) + c(2) + d(3) + e(4) + f() + g(5) + h(6.5) + inner("n")
        assert len(r) > 0 and len(b) == 2 and len(i(2)) == 2
    """
    scenario, config, _ = _data_case(tmp_path, "", setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ('r: int = (lambda v: f"{v}{p}")(1)', "int"),
        ('r = (lambda v: f"{v}{p.nope}")(1)', "nope"),
        ('r = (lambda v: f"{v.nope}{p}")(1)', "nope"),
        ('r = (lambda v: "{}{}".format(v, p) + 1)(1)', "str"),
    ],
    ids=["result", "capture-attribute", "param-attribute", "format"],
)
def test_a_lowered_lambda_fstring_is_still_type_checked(
    tmp_path: Path, line: str, message: str
) -> None:
    scenario, config, path = _data_case(tmp_path, "", 'p = "a"\n' + line + "\n")
    error = _only_error(typecheck_scenario(scenario, config))
    assert message in error.message, error.format()
    assert error.line == _line_of(path, line)


_LAMBDA_FSTRING_TYPES = (
    _FLAGS
    + """
class Color(enum.Enum):
    RED = 1
    BLUE = 2


class Shown:
    n: int

    def __init__(self, n: int) -> None:
        self.n = n

    def __str__(self) -> str:
        return "shown"

    def __repr__(self) -> str:
        return "Shown()"


class Holder:
    prefix = "h"


MULTILINE = lambda v, p: f\"\"\"a
{v}
{p}{Holder.prefix}\"\"\"
"""
)


def test_fstrings_in_lambdas_accept_the_types_a_scenario_uses(tmp_path: Path) -> None:
    """``str(x)`` / ``repr(x)`` exist for enums, flags, framework and user types."""
    setup = """
        p = cfg.name
        loc = carla.Location(x=1.0, y=2.0, z=3.0)
        col = Color.RED
        sh = Shown(1)
        a = lambda v: f"{v}{loc}{col}{Perm.R | Perm.W}{Bits.A}{sh}{p}"
        b = lambda v: f"{loc!r}{col!r}{sh!r}{sh!s}{sh!a}{p!s:>5}{p!a:>5}"
        c = lambda v: f"{p:>{v}}|{v:{len(p)}}|{v:>{v}d}"
        d = lambda v: (y := v, f"{y}{p}")[1]
        e = lambda v: [f"{k}{v}{p}" for k in range(v)]
        f = lambda v: (lambda u: [lambda w: f"{u}{v}{w}{p}" for _ in range(2)])(v)
        g = lambda v, h=lambda w: f"{w}{p}": h(v)
        i = MULTILINE
        r: str = a(1) + b(2) + c(3) + d(4) + e(2)[0] + f(5)[0](6) + g(7) + i(8, p)
        assert len(r) > 0
    """
    scenario, config, _ = _data_case(tmp_path, _LAMBDA_FSTRING_TYPES, setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


# -- a comprehension in a class body or a default_factory (#469) ---------------

_CLASS_BODY_COMPREHENSIONS = """
class Color(enum.Enum):
    RED = 1
    BLUE = 2


class Q:
    SQ = [i * i for i in range(3)]
    NAMES = [n.upper() for n in ["a", "b"]]
    TABLE = {k: len(k) for k in ["a", "bb"]}
    COLORS = [c.name for c in Color]
    DIRS = [d.name for d in TurnDirection]


@dataclass
class D:
    names: list[str] = dataclasses.field(
        default_factory=lambda: [str(i) for i in range(3)]
    )
    ds: list[TurnDirection] = dataclasses.field(
        default_factory=lambda: list(TurnDirection)
    )
    dnames: list[str] = dataclasses.field(
        default_factory=lambda: [d.name for d in TurnDirection]
    )
    cnames: list[str] = dataclasses.field(
        default_factory=lambda: [c.name for c in Color]
    )
    squares: tuple[int, ...] = dataclasses.field(
        default_factory=lambda: tuple(i for i in range(3))
    )
    opts: dict[str, int] = dataclasses.field(default_factory=lambda: dict(a=1))
"""


def test_class_body_comprehensions_and_factories_compile(tmp_path: Path) -> None:
    setup = """
        d = D()
        logger.info("%s %s %s", Q.SQ, Q.NAMES, Q.TABLE)
        logger.info("%s %s", Q.COLORS, Q.DIRS)
        logger.info("%s %s %s %s", d.names, d.ds, d.dnames, d.cnames)
        logger.info("%s %s", d.squares, d.opts)
    """
    scenario, config, _ = _data_case(tmp_path, _CLASS_BODY_COMPREHENSIONS, setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_a_hoisted_class_body_comprehension_is_still_type_checked(
    tmp_path: Path,
) -> None:
    definitions = """
class Q:
    BAD = [i.nope for i in range(0)]
"""
    scenario, config, path = _data_case(tmp_path, definitions, "x = Q.BAD\n")
    error = _only_error(typecheck_scenario(scenario, config))
    assert error.line == _line_of(path, "BAD = ")


def test_a_hoisted_default_factory_is_still_type_checked(tmp_path: Path) -> None:
    definitions = """
@dataclass
class D:
    names: list[str] = dataclasses.field(
        default_factory=lambda: [i + "a" for i in range(3)]
    )
"""
    scenario, config, path = _data_case(tmp_path, definitions, "x = D().names\n")
    error = _only_error(typecheck_scenario(scenario, config))
    assert error.line == _line_of(path, "default_factory=lambda"), error.format()


def test_a_class_body_comprehension_reading_its_class_is_refused(
    tmp_path: Path,
) -> None:
    definitions = """
class Q:
    BASE = 3
    SQ = [i * 2 for i in range(BASE)]
"""
    scenario, config, path = _data_case(tmp_path, definitions, "x = Q.SQ\n")
    error = _only_error(typecheck_scenario(scenario, config))
    assert "module-level function" in error.message, error.format()
    assert error.line == _line_of(path, "SQ = ")


def test_format_with_a_method_or_parameter_of_that_name_compiles(
    tmp_path: Path,
) -> None:
    # #471: only a binding in scope shadows the builtin.
    definitions = """
class Rep:
    def format(self) -> str:
        return "r"

    def ascii(self, x: int) -> str:
        return str(x)


def render(x: int, format: str) -> str:
    return format + str(x)
"""
    setup = """
        logger.info("%s", Rep().format() + Rep().ascii(1) + render(1, "n"))
        logger.info("%s %s", format(3, "05d"), ascii(7))
    """
    scenario, config, _ = _data_case(tmp_path, definitions, setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_format_in_the_function_that_binds_it_is_still_the_locals(
    tmp_path: Path,
) -> None:
    definitions = """
def render(x: int, format: int) -> int:
    return format(x, 2)
"""
    scenario, config, path = _data_case(tmp_path, definitions, "render(1, 2)\n")
    error = _only_error(typecheck_scenario(scenario, config))
    assert error.line == _line_of(path, "return format(x, 2)")


def test_a_module_variable_named_like_a_model_enum_compiles(tmp_path: Path) -> None:
    # #471: not rewritten into `_acs_members()`.
    definitions = """
SpeedDirection = ["a", "b"]
"""
    setup = """
        n = 0
        for d in SpeedDirection:
            n += len(d)
        logger.info("%d", n)
    """
    scenario, config, _ = _data_case(tmp_path, definitions, setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


_FUNCTIONS = """
def inc(x: int) -> int:
    return x + 1


def dbl(x: int) -> int:
    return x * 2
"""


@pytest.mark.parametrize(
    "line", ["fs = {inc, dbl}", "fs = {inc: 1}", "fs: set = {inc, dbl}"]
)
def test_a_set_or_dict_key_of_functions_gets_a_hint(tmp_path: Path, line: str) -> None:
    scenario, config, path = _data_case(tmp_path, _FUNCTIONS, line + "\n")
    error = _only_error(typecheck_scenario(scenario, config))
    assert "cannot hash a function" in error.message, error.format()
    assert error.line == _line_of(path, line)


def test_a_dict_of_functions_by_name_compiles(tmp_path: Path) -> None:
    scenario, config, _ = _data_case(
        tmp_path, _FUNCTIONS, 'fs = {"inc": inc}\nlogger.info("%d", fs["inc"](1))\n'
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_an_annotated_class_attribute_read_from_outside_gets_a_hint(
    tmp_path: Path,
) -> None:
    definitions = """
class Q:
    LIMIT: int = 5
"""
    scenario, config, path = _data_case(
        tmp_path, definitions, 'logger.info("%d", Q.LIMIT)\n'
    )
    error = _only_error(typecheck_scenario(scenario, config))
    assert "LIMIT: Final = ..." in error.message, error.format()
    assert error.line == _line_of(path, "Q.LIMIT")


def test_a_final_class_attribute_read_from_outside_compiles(tmp_path: Path) -> None:
    definitions = """
class Q:
    LIMIT: typing.Final = 5
    OTHER = 6
"""
    scenario, config, _ = _data_case(
        tmp_path, definitions, 'logger.info("%d", Q.LIMIT + Q.OTHER)\n'
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


# -- class-body lambdas, method lambda defaults, bare collections (#482) -------

_CLASS_BODY_LAMBDAS = """
class Outer:
    class Inner:
        k = 2
        f = lambda v: v * 2
        key = lambda v: str(v) + "!"
        XS = [i * 2 for i in range(3)]


class T:
    scale = 3
    f = lambda self, v: v * self.scale
    s = staticmethod(lambda a, b=2: a + b)


class P2:
    def sort(self, xs: list[int], key=lambda v: -v) -> list[int]:
        return sorted(xs, key=key)

    def scaled(self, xs: list[int], *, by=lambda v: v * 2) -> list[int]:
        return [by(x) for x in xs]


class Guard:
    def __enter__(self) -> Guard:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: object
    ) -> bool:
        return False
"""


def test_class_body_lambdas_and_method_lambda_defaults_compile(
    tmp_path: Path,
) -> None:
    setup = """
        logger.info("%s %s", Outer.Inner.f(3), Outer.Inner.key(4))
        logger.info("%s %s", Outer.Inner.XS, T().f(2))
        logger.info("%s %s", T.s(1), T.s(1, 5))
        logger.info("%s %s", P2().sort([1, 2]), P2().scaled([1, 2]))
        with Guard():
            pass
    """
    scenario, config, _ = _data_case(tmp_path, _CLASS_BODY_LAMBDAS, setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()
    # The idiomatic __exit__ signature is no warning (it contradicted the docs).
    assert not any("__exit__" in str(w) for w in result.warnings), result.format()


def test_a_class_body_lambda_is_still_type_checked(tmp_path: Path) -> None:
    definitions = """
class Q:
    f = lambda v: v.nope
"""
    scenario, config, path = _data_case(tmp_path, definitions, "x = Q.f(1)\n")
    error = _only_error(typecheck_scenario(scenario, config))
    assert error.line == _line_of(path, "f = lambda"), error.format()


def test_a_method_lambda_default_is_still_type_checked(tmp_path: Path) -> None:
    definitions = """
class Q:
    def run(self, xs: list[int], key=lambda v: v.nope) -> list[int]:
        return sorted(xs, key=key)
"""
    scenario, config, path = _data_case(tmp_path, definitions, "x = Q().run([1])\n")
    error = _only_error(typecheck_scenario(scenario, config))
    assert error.line == _line_of(path, "def run"), error.format()


def test_a_method_lambda_default_in_a_function_is_refused(tmp_path: Path) -> None:
    definitions = """
def build() -> int:
    class Q:
        def run(self, key=lambda v: v) -> int:
            return key(1)

    return Q().run()
"""
    scenario, config, path = _data_case(tmp_path, definitions, "x = build()\n")
    error = _only_error(typecheck_scenario(scenario, config))
    assert "module-level `def`" in error.message, error.format()
    assert error.line == _line_of(path, "key=lambda")


def test_a_comprehension_in_a_nested_class_compiles(tmp_path: Path) -> None:
    definitions = """
class Outer:
    class Inner:
        XS = [i * i for i in range(3)]
        TABLE = {k: len(k) for k in ["a", "bb"]}

    class Other(Inner):
        pass
"""
    scenario, config, _ = _data_case(
        tmp_path,
        definitions,
        'logger.info("%s %s", Outer.Inner.XS, Outer.Inner.TABLE)\n',
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


@pytest.mark.parametrize("annotation", ["list", "dict", "set", "tuple"])
def test_a_bare_collection_field_of_a_dataclass_is_refused_at_its_line(
    tmp_path: Path, annotation: str
) -> None:
    definitions = f"""
@dataclass
class Cfg:
    xs: {annotation}
    name: str = "a"
"""
    scenario, config, path = _data_case(tmp_path, definitions, "x = Cfg([])\n")
    error = _only_error(typecheck_scenario(scenario, config))
    assert error.line == _line_of(path, f"xs: {annotation}"), error.format()
    assert f"bare `{annotation}`" in error.message


def test_a_bare_collection_in_a_plain_class_is_refused_at_its_line(
    tmp_path: Path,
) -> None:
    definitions = """
class Cfg:
    name = "a"
    xs: list = []
"""
    scenario, config, path = _data_case(tmp_path, definitions, "x = Cfg()\n")
    error = _only_error(typecheck_scenario(scenario, config))
    assert error.line == _line_of(path, "xs: list"), error.format()
    assert "bare `list`" in error.message


def test_a_bare_tuple_return_gets_a_hint(tmp_path: Path) -> None:
    definitions = """
def pt() -> tuple:
    return (1.0, 2.0)
"""
    scenario, config, _ = _data_case(tmp_path, definitions, "x = pt()\n")
    error = _only_error(typecheck_scenario(scenario, config))
    assert "tuple[float, float]" in error.message, error.format()


def test_a_codon_abort_names_the_callable_attributes(tmp_path: Path) -> None:
    # Codon 0.19 aborts with std::bad_alloc and no location (#488): the
    # report names the `Callable` attributes of the checked modules.
    import ast

    from autoware_carla_scenario.typecheck import check

    text = "import typing\n\n\nclass R:\n    cbs: list[typing.Callable[[int], int]]\n"
    path = tmp_path / "m.py"
    path.write_text(text)
    sources = check._Sources({"m": check._Module(path, False, text, ast.parse(text))})
    found = check._bad_alloc_hints(sources)
    assert [(d.path, d.line) for d in found] == [(str(path), 5)]
    assert "std::bad_alloc" in found[0].message
    assert "Callable" in found[0].message
    assert "std::bad_alloc" in check._bad_alloc_hints(check._Sources())[0].message


def test_a_callback_stored_through_a_helper_method_compiles(tmp_path: Path) -> None:
    # `register` hands the callback to `_add`, which stores it (#488).
    definitions = """
class Registry:
    handlers: list[typing.Callable[[float], float]]

    def __init__(self) -> None:
        self.handlers = []

    def _add(self, fn: typing.Callable[[float], float]) -> None:
        self.handlers.append(fn)

    def register(self, fn: typing.Callable[[float], float]) -> None:
        self._add(fn)

    def call(self, v: float) -> float:
        return self.handlers[0](v)


def inc(v: float) -> float:
    return v + 1.0
"""
    setup = """
        reg = Registry()
        reg.register(inc)
        self.register_fail_condition(TimeoutCondition(reg.call(1.0), label="t"))
    """
    scenario, config, _ = _data_case(tmp_path, definitions, setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


# ---------------------------------------------------------------------------
# dataclasses.replace is lowered to typed copies (#487)
# ---------------------------------------------------------------------------

_REPLACE_CONFIG = """
from dataclasses import dataclass, field
import enum


class Kind(enum.Enum):
    SLOW = "slow"
    FAST = "fast"


@dataclass
class Waypoint:
    lane: int = 0
    speed: float = 5.0
    kind: Kind = Kind.SLOW


@dataclass(frozen=True)
class Frozen:
    a: int = 1
    note: str | None = "x"


@dataclass
class Plan:
    points: list[Waypoint] = field(default_factory=list)
    best: Waypoint | None = None
    cache: int = field(default=0, init=False)


@dataclass
class Base2:
    a: int = 0


@dataclass
class Derived(Base2):
    b: float = 1.0


@dataclass
class Leaf(Derived):
    c: str = "c"
    a: int = 9


@dataclass(frozen=True)
class FrozenBase:
    x: int = 1


@dataclass(frozen=True)
class FrozenChild(FrozenBase):
    y: float | None = None


@dataclass
class CaseConfig:
    name: str = "case"
    goal_lanelet_ids: list[int] = field(default_factory=lambda: [460, 265])
    timeout_seconds: float = 10.0
    min_speed_kmh: float | None = None
    plan: Plan = field(default_factory=lambda: Plan([Waypoint(1, 3.0, Kind.FAST)]))
"""

_REPLACE_IMPORTS = "\n\nfrom dataclasses import replace\n\nfrom .configs import (\n    Base2,\n    Derived,\n    Frozen,\n    FrozenBase,\n    FrozenChild,\n    Kind,\n    Leaf,\n    Plan,\n    Waypoint,\n)\n\n\n"


def _replace_case(tmp_path: Path, setup: str) -> tuple[Any, Any, Path]:
    return _write_case(
        tmp_path,
        _TIMEOUT_SETUP + textwrap.dedent(setup),
        _REPLACE_IMPORTS,
        config=_REPLACE_CONFIG,
    )


def test_replace_of_a_dataclass_is_checked_and_usable(tmp_path: Path) -> None:
    setup = """
        p = cfg.plan
        q = replace(p, best=None)
        fast = [w for w in q.points]
        assert len(fast) == 1 and q.best is None
        w = Waypoint()
        w2 = replace(w, speed=2.5)
        w3 = replace(w2, lane=3, kind=Kind.FAST, speed=1.0)
        assert w3.lane + 1 == 4 and w3.speed < 2.0 and w.lane == 0
        r = replace(p, best=w3, points=[w, w2])
        assert r.best is not None and r.best.lane == 3 and len(r.points) == 2
        z = replace(
            w,
            lane=7,
        )
        f = replace(Frozen(), note=None)
        g = replace(f, a=2, note="y")
        assert g.a == 2 and f.note is None
        assert replace(g).a == 2
    """
    scenario, config, _ = _replace_case(tmp_path, setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_replace_of_a_field_a_base_dataclass_declares_is_usable(
    tmp_path: Path,
) -> None:
    # #494: a method of a base and of its subclass is virtual, and Codon
    # cannot pass a keyword through the dispatch thunk.
    setup = """
        d = Derived(a=1, b=2.0)
        d2 = replace(d, a=5)
        d3 = replace(d, b=5.0)
        d4 = replace(d, a=3, b=4.0)
        assert d2.a + d3.a + d4.a > 0 and d2.b < d3.b and d4.b < d3.b
        base = replace(Base2(), a=2)
        assert base.a == 2
        leaf = Leaf()
        l1 = replace(leaf, a=1)
        l2 = replace(leaf, b=2.5, c="z")
        l3 = replace(replace(leaf, c="y"), a=4, b=1.5)
        assert l1.c != l2.c and l3.a == 4 and l3.b < 2.0
        fc = FrozenChild(x=1, y=None)
        fc2 = replace(fc, x=2)
        fc3 = replace(fc2, y=1.0, x=3)
        assert fc3.x == 3 and replace(FrozenBase(), x=2).x == 2
        z = replace(
            d,
            a=7,
            b=1.0,
        )
    """
    scenario, config, _ = _replace_case(tmp_path, setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


@pytest.mark.parametrize(
    "call",
    [
        "replace(Derived(), a=None)",
        'replace(Derived(), a="x")',
        "replace(Derived(), nonexistent=1)",
        "replace(Leaf(), c=3, a=1)",
        "replace(FrozenChild(), x=None)",
        'replace(Leaf(), a=1, b="s")',
    ],
    ids=["base-none", "base-str", "unknown", "own-wrong", "frozen", "mixed"],
)
def test_a_wrong_replace_of_a_derived_dataclass_is_refused_at_the_call(
    tmp_path: Path, call: str
) -> None:
    scenario, config, path = _replace_case(tmp_path, f"x = {call}\n")
    error = _only_error(typecheck_scenario(scenario, config))
    assert error.path == str(path), error.format()
    assert error.line == _line_of(path, f"x = {call}"), error.format()
    line = path.read_text().splitlines()[error.line - 1]
    assert error.column is None or error.column <= len(line), error.format()
    assert "failed without a diagnostic" not in error.format()
    assert "class_thunk_dispatch" not in error.message, error.format()


def test_a_wrong_value_for_an_inherited_field_names_the_field(
    tmp_path: Path,
) -> None:
    scenario, config, path = _replace_case(tmp_path, 'x = replace(Leaf(), a="s")\n')
    error = _only_error(typecheck_scenario(scenario, config))
    assert "field 'a' of Leaf is declared int, not str" in error.message
    assert error.trace == (), error.format()
    assert error.line == _line_of(path, "x = replace(Leaf()")
    # Codon locates the call at `replace`, which precedes the rewrite
    # (`._acs_with__a(...)` is appended after it), so the column is the
    # author's: the 1-based column of `replace` in the user's own line.
    written = path.read_text().splitlines()[error.line - 1]
    assert error.column == written.index("replace(") + 1, error.format()


@pytest.mark.parametrize(
    ("argument", "local"),
    [("lambda: None", False), ("noop_str", True)],
)
def test_a_zero_argument_lambda_or_a_local_function_of_another_signature_is_located(
    tmp_path: Path, argument: str, local: bool
) -> None:
    # Codon 0.19 fails in code generation on both (#488); the report names
    # the line that passes them, not "failed without a diagnostic".
    definitions = """
class Ticks:
    cbs: list[typing.Callable[[], None]]

    def __init__(self) -> None:
        self.cbs = []

    def register(self, fn: typing.Callable[[], None]) -> None:
        self.cbs.append(fn)
"""
    prelude = "def noop_str(v: str) -> str:\n            return v\n" if local else ""
    setup = f"""
        {prelude}
        ticks = Ticks()
        ticks.register({argument})
        self.register_fail_condition(TimeoutCondition(1.0, label="t"))
    """
    scenario, config, path = _data_case(tmp_path, definitions, setup)
    error = _only_error(typecheck_scenario(scenario, config))
    assert error.line == _line_of(path, "ticks.register("), error.format()
    assert Path(error.path or "") == path, error.format()
    assert "without a diagnostic" not in error.message, error.format()


def test_replace_as_dataclasses_attribute_is_checked(tmp_path: Path) -> None:
    setup = """
        w = dataclasses.replace(Waypoint(), lane=2)
        assert w.lane == 2
        dataclasses.replace(w, nonexistent=1)
    """
    scenario, config, path = _write_case(
        tmp_path,
        _TIMEOUT_SETUP + textwrap.dedent(setup),
        _REPLACE_IMPORTS + "import dataclasses\n",
        config=_REPLACE_CONFIG,
    )
    error = _only_error(typecheck_scenario(scenario, config))
    assert error.line == _line_of(path, "dataclasses.replace(w, nonexistent=1)")


@pytest.mark.parametrize(
    ("call", "message"),
    [
        ("replace(Waypoint(), nonexistent=1)", "nonexistent"),
        ("replace(Waypoint(), lane=1, nonexistent=1)", "nonexistent"),
        ('replace(Waypoint(), speed="fast")', "str"),
        ("replace(Frozen(), a=None)", "None"),
        ("replace(Plan(), cache=3)", "init=False"),
    ],
    ids=["unknown", "unknown-among-known", "wrong-type", "wrong-none", "init-false"],
)
def test_a_wrong_replace_is_refused_at_its_line(
    tmp_path: Path, call: str, message: str
) -> None:
    scenario, config, path = _replace_case(tmp_path, f"x = {call}\n")
    error = _only_error(typecheck_scenario(scenario, config))
    assert message in error.format(), error.format()
    assert error.path == str(path)
    assert error.line == _line_of(path, f"x = {call}")
    assert "failed without a diagnostic" not in error.format()


@pytest.mark.parametrize(
    "call",
    [
        "replace(Waypoint(), **{'lane': 1})",
        "replace(Waypoint(), 3)",
        "replace(obj=Waypoint(), lane=1)",
    ],
    ids=["double-star", "positional", "obj-keyword"],
)
def test_a_replace_call_that_cannot_be_lowered_is_refused(
    tmp_path: Path, call: str
) -> None:
    scenario, config, path = _replace_case(tmp_path, f"x = {call}\n")
    error = _only_error(typecheck_scenario(scenario, config))
    assert "dataclasses.replace" in error.format(), error.format()
    assert error.line == _line_of(path, f"x = {call}")


def test_replace_of_a_class_the_check_has_no_fields_of_is_refused(
    tmp_path: Path,
) -> None:
    scenario, config, path = _replace_case(
        tmp_path, "x = replace(carla.Location(), x=1.0)\n"
    )
    error = _only_error(typecheck_scenario(scenario, config))
    assert error.line == _line_of(path, "x = replace(carla.Location()")
    assert "failed without a diagnostic" not in error.format()


def test_a_column_after_a_rewrite_is_dropped_and_one_before_it_kept() -> None:
    # #494: a rewritten line keeps the columns that precede the rewrite.
    from autoware_carla_scenario.typecheck.check import _column_survives

    written = "x = replace(d, a=5)"
    compiled = "x = replace(d)._acs_with__a(5)"
    assert _column_survives(5, compiled, written)
    assert not _column_survives(20, compiled, written)
    assert not _column_survives(len(written) + 5, written, written[:5])


def test_codons_lambda_type_names_are_not_shown() -> None:
    from autoware_carla_scenario.typecheck import check

    text = "'Optional[%_lambda_981(...)]' does not match; cb: %_lambda_88[T1]"
    assert check._plain_lambda_names(text) == (
        "'Optional[a lambda]' does not match; cb: a lambda"
    )


# ---------------------------------------------------------------------------
# A dropped decorator stacked under a kept one (#595)
# ---------------------------------------------------------------------------

_STACK_IMPORTS = """

import functools
from abc import ABC, abstractmethod

from typing_extensions import override
"""

#: Each case: the base class's member and the subclass's (#595).
_STACKED_DECORATORS = {
    "property_abstractmethod": (
        "    @property\n    @abstractmethod\n    def area(self) -> float: ...\n",
        "    @property\n    def area(self) -> float:\n        return 4.0\n",
    ),
    "property_override": (
        "    @property\n    def area(self) -> float:\n        return 0.0\n",
        "    @property\n    @override\n    def area(self) -> float:\n"
        "        return 4.0\n",
    ),
    "staticmethod_abstractmethod": (
        "    @staticmethod\n    @abstractmethod\n    def area() -> float: ...\n",
        "    @staticmethod\n    def area() -> float:\n        return 4.0\n",
    ),
    "property_cache": (
        "    @property\n    def area(self) -> float:\n        return 0.0\n",
        "    @property\n    @functools.cache\n    def area(self) -> float:\n"
        "        return 4.0\n",
    ),
    "staticmethod_lru_cache": (
        "    @staticmethod\n    def area() -> float:\n        return 0.0\n",
        "    @staticmethod\n    @functools.lru_cache(maxsize=None)\n"
        "    def area() -> float:\n        return 4.0\n",
    ),
    "multi_line_with_comments": (
        "    @property\n    @abstractmethod\n    def area(self) -> float: ...\n",
        "    @property  # read as an attribute\n"
        "    # cached, and an override\n"
        "    @functools.lru_cache(\n        maxsize=8,\n    )\n"
        "    @override\n"
        "    def area(self) -> float:\n        return 4.0\n",
    ),
}


@pytest.mark.parametrize("case", sorted(_STACKED_DECORATORS))
def test_a_dropped_decorator_under_a_kept_one_compiles(
    tmp_path: Path, case: str
) -> None:
    # `@property` / `@staticmethod` over `@abstractmethod`, `@override`,
    # `@functools.cache` or `@functools.lru_cache(...)`: the dropped one left
    # a blank line in the decorator list, a Codon syntax error (#595).
    base, sub = _STACKED_DECORATORS[case]
    read = "s.area()" if "staticmethod" in case else "s.area"
    setup = f's: Shape = Sq()\nlogger.info("%f", {read})\n'
    extra = f"{_STACK_IMPORTS}\n\nclass Shape(ABC):\n{base}\n\nclass Sq(Shape):\n{sub}"
    scenario, config, _ = _write_case(tmp_path, setup, extra)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_an_error_on_a_moved_decorator_is_at_its_line(tmp_path: Path) -> None:
    # The kept decorator moves down over the dropped `@abstractmethod`
    # (#595): its error names the line it is written on.
    extra = (
        "\n\nfrom abc import ABC, abstractmethod\n\n\n"
        "def tag(n: int):\n    def wrap(f):\n        return f\n\n    return wrap\n\n\n"
        "class Shape(ABC):\n"
        '    @tag("a")\n    @abstractmethod\n    def area(self) -> float: ...\n\n\n'
        "class Sq(Shape):\n    def area(self) -> float:\n        return 4.0\n"
    )
    setup = 's: Shape = Sq()\nlogger.info("%f", s.area())\n'
    scenario, config, path = _write_case(tmp_path, setup, extra)
    error = _only_error(typecheck_scenario(scenario, config))
    assert error.line == _line_of(path, '@tag("a")'), error.format()


# ---------------------------------------------------------------------------
# typing_extensions (#514)
# ---------------------------------------------------------------------------


def test_typing_extensions_names_compile_as_typing_ones(tmp_path: Path) -> None:
    pytest.importorskip("typing_extensions")
    header = _HEADER.replace(
        "from .configs import CaseConfig\n",
        "from typing_extensions import (Self, TypeAlias, TypeGuard, assert_never,\n"
        "    override)\n\nfrom .configs import CaseConfig\n",
        1,
    )
    helper = """

Pose: TypeAlias = tuple[float, float]


class Counter:
    n: int

    def __init__(self, n: int) -> None:
        self.n = n

    def inc(self) -> Self:
        self.n += 1
        return self


class Twice(Counter):
    @override
    def inc(self) -> Self:
        self.n += 2
        return self


def is_pair(v: tuple[float, ...]) -> TypeGuard[Pose]:
    return len(v) == 2


def label(k: int) -> str:
    if k == 0:
        return "a"
    else:
        assert_never(k)
"""
    setup = _TIMEOUT_SETUP + (
        "logger.info('%d %d %s %s', Counter(1).inc().n, Twice(1).inc().n, "
        "is_pair((1.0, 2.0)), label(0))\n"
    )
    scenario, config, _ = _write_case(tmp_path, setup, helper, header=header)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_a_typing_extensions_name_typing_lacks_is_refused_with_a_hint(
    tmp_path: Path,
) -> None:
    header = _HEADER.replace(
        "from .configs import CaseConfig\n",
        "from typing_extensions import deprecated\n\nfrom .configs import CaseConfig\n",
        1,
    )
    scenario, config, path = _write_case(tmp_path, _TIMEOUT_SETUP, header=header)
    error = _only_error(typecheck_scenario(scenario, config))
    assert "cannot import name 'deprecated' from 'typing_extensions'" in error.message
    assert "docs/typecheck.md" in error.message, error.format()


# ---------------------------------------------------------------------------
# collections.abc (#586)
# ---------------------------------------------------------------------------


def test_collections_abc_names_compile_as_typing_ones(tmp_path: Path) -> None:
    header = _HEADER.replace(
        "from .configs import CaseConfig\n",
        "from collections.abc import (\n    Callable,\n    Iterator,\n    Mapping,\n"
        "    Sequence,\n    Set,\n)\n\nfrom .configs import CaseConfig\n",
        1,
    )
    helper = """


def total(xs: Sequence[float], f: Callable[[float], float]) -> float:
    return sum(f(v) for v in xs)


def count(n: int) -> Iterator[int]:
    for i in range(n):
        yield i


def has(s: Set[int], k: int) -> bool:
    return k in s


def get(m: Mapping[str, int]) -> int:
    return m["a"]
"""
    setup = _TIMEOUT_SETUP + (
        'logger.info("%f %d %s %d", total([1.0, 2.0], lambda v: v * 2.0), '
        'sum(count(3)), has({1, 2}, 1), get({"a": 1}))\n'
    )
    scenario, config, _ = _write_case(tmp_path, setup, helper, header=header)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


@pytest.mark.parametrize(
    ("statement", "needle"),
    [
        (
            "from collections.abc import KeysView",
            "collections.abc.KeysView is not modelled by the check",
        ),
        ("import collections.abc", "`import collections.abc` is not supported"),
    ],
    ids=["unmodelled-name", "module-import"],
)
def test_collections_abc_the_check_cannot_read_is_refused_with_a_hint(
    tmp_path: Path, statement: str, needle: str
) -> None:
    header = _HEADER.replace(
        "from .configs import CaseConfig\n",
        f"{statement}\n\nfrom .configs import CaseConfig\n",
        1,
    )
    scenario, config, path = _write_case(tmp_path, _TIMEOUT_SETUP, header=header)
    error = _only_error(typecheck_scenario(scenario, config))
    assert needle in error.message, error.format()
    assert "from typing" in error.message, error.format()
    assert error.line == _line_of(path, statement)


# ---------------------------------------------------------------------------
# A caught exception kept in a variable, attribute or list (#512)
# ---------------------------------------------------------------------------

_EXCEPTIONS = """
class MyErr(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class Holder:
    last: Exception | None
    errs: list[Exception]

    def __init__(self) -> None:
        self.last = None
        self.errs = []

    def run(self) -> None:
        try:
            raise MyErr("boom")
        except MyErr as e:
            self.last = e
            self.errs.append(e)
            self.keep(e)

    def keep(self, err: Exception) -> None:
        self.errs.append(err)


def attempt(text: str) -> Exception | None:
    try:
        int(text)
    except ValueError as e:
        return e
    return None
"""


def test_a_caught_exception_is_kept_in_an_exception_typed_name(
    tmp_path: Path,
) -> None:
    setup = """
        last: Exception = Exception("none")
        last2: Exception | None = None
        errs: list[Exception] = []
        by_key: dict[str, Exception] = {}
        base: BaseException = BaseException("BaseException", "none")
        try:
            int("x")
        except ValueError as e:
            last = e
            last2 = e
            errs.append(e)
            by_key["k"] = e
            base = e
        try:
            raise MyErr("zz")
        except MyErr as e2:
            errs.append(e2)
            errs = [e2]
        h = Holder()
        h.run()
        found = attempt("x")
        logger.info("%s %s %d %d", last, last2 is None, len(errs), len(h.errs))
        logger.info("%s %s %s", found is None, base, len(by_key))
    """
    scenario, config, _ = _data_case(tmp_path, _EXCEPTIONS, setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_an_exceptions_args_are_its_message(tmp_path: Path) -> None:
    setup = """
        try:
            raise MyErr("zz")
        except MyErr as e:
            logger.info("%s %s", e.args, e.args[0])
        try:
            int("x")
        except ValueError as e2:
            logger.info("%d", len(e2.args))
        except KeyError as e3:
            logger.info("%s", e3.args)
    """
    scenario, config, _ = _data_case(tmp_path, _EXCEPTIONS, setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_an_exception_that_is_not_a_caught_one_is_refused_with_a_hint(
    tmp_path: Path,
) -> None:
    scenario, config, path = _data_case(
        tmp_path, "", 'last: Exception = ValueError("x")\nlogger.info("%s", last)\n'
    )
    error = _only_error(typecheck_scenario(scenario, config))
    assert "does not match expected type" in error.message
    assert "str(e)" in error.message, error.format()


def test_assert_never_in_an_expression_is_refused_at_its_line(
    tmp_path: Path,
) -> None:
    # typing.assert_never is Python 3.11+; the import runs on 3.10 too.
    pytest.importorskip("typing_extensions")
    definitions = """
from typing_extensions import assert_never


def pick(x: int) -> int:
    return 1 if x else assert_never(x)
"""
    scenario, config, path = _data_case(
        tmp_path, definitions, 'logger.info("%d", pick(1))\n'
    )
    error = _only_error(typecheck_scenario(scenario, config))
    assert "line of its own" in error.message, error.format()
    assert error.line == _line_of(path, "return 1 if x else assert_never(x)")


# ---------------------------------------------------------------------------
# operator.itemgetter / attrgetter, copy.copy / deepcopy (#517)
# ---------------------------------------------------------------------------

_GETTER_CLASSES = """
import operator
from operator import attrgetter, itemgetter


class Item:
    n: int
    tag: str

    def __init__(self, n: int, tag: str) -> None:
        self.n = n
        self.tag = tag
"""


def test_literal_getters_are_sort_keys(tmp_path: Path) -> None:
    setup = """
        xs = [(1, 2.0), (0, 3.0)]
        a = sorted(xs, key=operator.itemgetter(1))
        b = max(xs, key=itemgetter(0))
        c = sorted(xs, key=itemgetter(1, 0))
        items = [Item(2, "b"), Item(1, "a")]
        d = sorted(items, key=operator.attrgetter("n"))
        e = sorted(items, key=attrgetter("tag", "n"))
        logger.info("%s %s %s %d %d", a, b, c, d[0].n, len(e))
    """
    scenario, config, _ = _data_case(tmp_path, _GETTER_CLASSES, setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_a_getter_of_a_variable_is_refused_with_a_hint(tmp_path: Path) -> None:
    setup = """
        which = 1
        xs = [(1, 2.0), (0, 3.0)]
        a = sorted(xs, key=operator.itemgetter(which))
    """
    scenario, config, path = _data_case(tmp_path, _GETTER_CLASSES, setup)
    error = _only_error(typecheck_scenario(scenario, config))
    assert "write a lambda" in error.message, error.format()
    assert error.line == _line_of(path, "operator.itemgetter(which)")


_COPY_CLASSES = """
import copy


@dataclass
class Point:
    x: float
    y: float


@dataclass
class Path3(Point):
    tags: list[str]


class Box:
    items: list[Point]

    def __init__(self) -> None:
        self.items = [Point(1.0, 2.0)]
"""


def test_copy_and_deepcopy_of_a_class_of_the_module(tmp_path: Path) -> None:
    setup = """
        p = Path3(1.0, 2.0, ["a"])
        shallow = copy.copy(p)
        deep = copy.deepcopy(p)
        deep.tags.append("b")
        shallow.x = 5.0
        box = Box()
        other = copy.deepcopy(box)
        other.items[0].x = 9.0
        assert len(p.tags) == 1 and p.x == 1.0 and box.items[0].x == 1.0
        logger.info("%d %f %f", len(deep.tags), shallow.x, other.items[0].x)
    """
    scenario, config, _ = _data_case(tmp_path, _COPY_CLASSES, setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_copy_of_a_class_from_a_module_without_copy_is_refused_with_a_hint(
    tmp_path: Path,
) -> None:
    helpers = (
        "class Thing:\n    n: int\n\n    def __init__(self, n: int) -> None:\n"
        "        self.n = n\n"
    )
    header = _HEADER.replace(
        "from .configs import CaseConfig\n",
        "import copy\n\nfrom .configs import CaseConfig\nfrom .helpers import Thing\n",
        1,
    )
    setup = _TIMEOUT_SETUP + "t = copy.copy(Thing(1))\nlogger.info('%d', t.n)\n"
    scenario, config, path = _write_case(
        tmp_path, setup, header=header, modules={"helpers.py": helpers}
    )
    result = typecheck_scenario(scenario, config)
    error = _only_error(result)
    assert "__copy__" in error.message and "define them on this class" in error.message


# ---------------------------------------------------------------------------
# A bound or constrained TypeVar (#513)
# ---------------------------------------------------------------------------


def test_a_bound_and_a_constrained_type_var_compile(tmp_path: Path) -> None:
    definitions = """
from typing import TypeVar


class Base:
    def v(self) -> int:
        return 1


B = TypeVar("B", bound=Base)
N = TypeVar("N", int, float)


def pick(a: B, b: B) -> B:
    return a if a.v() > b.v() else b


def clamp(x: N, lo: N, hi: N) -> N:
    return max(lo, min(x, hi))
"""
    setup = """
        logger.info("%d %f %d", pick(Base(), Base()).v(), clamp(1.5, 0.0, 1.0), clamp(5, 0, 3))
    """
    scenario, config, _ = _data_case(tmp_path, definitions, setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


_RAISING_CONFIG = """
from dataclasses import dataclass


@dataclass
class CaseConfig:
    name: str = "p"
    timeout_seconds: float = 5.0
    label: str = "a"

    def __post_init__(self) -> None:
        if self.label == "bad":
            raise ValueError("label must not be bad")
"""


def test_a_config_that_raises_when_built_is_reported(tmp_path: Path) -> None:
    # The runner builds config_cls(**scenario_dict): a __post_init__ that
    # raises, or an Enum name that does not convert, stops the run, and the
    # check says so with the exception (#531).
    scenario, config, _ = _write_case(tmp_path, "pass\n", config=_RAISING_CONFIG)
    assert typecheck_scenario(scenario, config, {"label": "ok"}).ok
    result = typecheck_scenario(scenario, config, {"label": "bad"})
    assert not result.ok
    assert "ValueError: label must not be bad" in result.format()

    setup = "logger.info('%s', cfg.mode.value)\n"
    scenario, config, _ = _write_case(tmp_path, setup, config=_CONVERTING_CHOICE_CONFIG)
    result = typecheck_scenario(scenario, config, {"mode": "nope"})
    assert not result.ok
    assert "ValueError" in result.format()


# ---------------------------------------------------------------------------
# Round 29: to_lanelet2, Optional actions, `or` readers, Optional subclasses
# ---------------------------------------------------------------------------

_R29_HEADER = _HEADER.replace(
    "    TurnAction,\n",
    "    TurnAction,\n    EntityRole,\n    SetSpeedAction,\n    to_lanelet2,\n    to_carla_world,\n    to_opendrive,\n",
    1,
).replace("from typing", "from typing", 1)
_R29_HEADER = _R29_HEADER.replace(
    "import logging\n", "import logging\nfrom typing import Optional\n", 1
)

_R29_EXTRA = """
    def _gap(self, world: carla.World) -> Optional[float]:
        return None
"""


@pytest.mark.parametrize(
    "setup",
    [
        "p = to_lanelet2(to_opendrive(Lanelet2Pose(lanelet_id=1, s=0.0)))\n"
        'logger.info("%d", p.lanelet_id)\n',
        'p = to_lanelet2(OpenDrivePose(road_id="1", lane_id=-1, s=0.0))\n'
        'logger.info("%d", p.lanelet_id)\n',
        "p = to_lanelet2(to_carla_world(Lanelet2Pose(lanelet_id=1, s=0.0)))\n"
        'logger.info("%d", p.lanelet_id)\n',
    ],
    ids=["opendrive", "opendrive-literal", "carla-world"],
)
def test_to_lanelet2_takes_both_documented_poses(tmp_path: Path, setup: str) -> None:
    scenario, config, _ = _write_case(tmp_path, setup, header=_R29_HEADER)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_to_lanelet2_refuses_another_argument(tmp_path: Path) -> None:
    wrong = "to_lanelet2(Lanelet2Pose(lanelet_id=1, s=0.0))"
    scenario, config, path = _write_case(tmp_path, wrong + "\n", header=_R29_HEADER)
    result = typecheck_scenario(scenario, config)
    assert any(
        d.line == _line_of(path, wrong) and "to_lanelet2 takes" in d.message
        for d in result.diagnostics
    ), result.format()


_OPTIONAL_ACTION_SETUPS = {
    "local": """
        a: Optional[SetSpeedAction] = None
        if cfg.timeout_seconds > 5.0:
            a = SetSpeedAction(EntityRole.npc(1), 10.0, label="s")
        if a is not None:
            self.REG(a)
        """,
    "attribute": """
        self._a = None
        if cfg.timeout_seconds > 5.0:
            self._a = SetSpeedAction(EntityRole.npc(1), 10.0, label="s")
        if self._a is not None:
            self.REG(self._a)
        """,
    "conditional": """
        a = SetSpeedAction(EntityRole.npc(1), 10.0, label="s") if cfg.timeout_seconds > 1.0 else None
        if a:
            self.REG(a)
        """,
}


@pytest.mark.parametrize("register", ["register_pre_tick", "register_post_tick"])
@pytest.mark.parametrize("kind", sorted(_OPTIONAL_ACTION_SETUPS))
def test_an_optional_action_registered_after_a_none_test_compiles(
    tmp_path: Path, kind: str, register: str
) -> None:
    setup = _OPTIONAL_ACTION_SETUPS[kind].replace("REG", register)
    scenario, config, _ = _write_case(
        tmp_path,
        setup,
        extra="\n    _a: Optional[SetSpeedAction]\n",
        header=_R29_HEADER,
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


@pytest.mark.parametrize(
    "read",
    ["lambda w: self._gap(w) or 0.0", "lambda w: 0.0 or self._gap(w)"],
    ids=["optional-first", "optional-last"],
)
def test_a_measure_read_that_defaults_an_optional_with_or_compiles(
    tmp_path: Path, read: str
) -> None:
    setup = f'self.register_measure("a", {read})\n'
    scenario, config, _ = _write_case(
        tmp_path, setup, extra=_R29_EXTRA, header=_R29_HEADER
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


_OPTIONAL_BASE_SETUPS = {
    "local": """
        c: Optional[BaseCondition] = ElapsedTimeCondition(1.0, label="e")
        if c is not None:
            self.register_fail_condition(c)
        """,
    "branches": """
        c: Optional[BaseCondition] = None
        if cfg.timeout_seconds > 5.0:
            c = ElapsedTimeCondition(1.0, label="e")
        else:
            c = TimeoutCondition(3.0, label="t")
        if c is not None:
            self.register_fail_condition(c)
        """,
    "action": """
        a: Optional[BaseAction] = None
        if cfg.timeout_seconds > 5.0:
            a = SetSpeedAction(EntityRole.npc(1), 10.0, label="s")
        if a is not None:
            self.register_pre_tick(a)
        """,
}


@pytest.mark.parametrize("kind", sorted(_OPTIONAL_BASE_SETUPS))
def test_an_optional_base_class_holding_a_subclass_is_refused_with_a_hint(
    tmp_path: Path, kind: str
) -> None:
    header = _R29_HEADER.replace(
        "    BaseCondition,\n", "    BaseAction,\n    BaseCondition,\n", 1
    )
    scenario, config, _ = _write_case(
        tmp_path, _OPTIONAL_BASE_SETUPS[kind], header=header
    )
    result = typecheck_scenario(scenario, config)
    # Codon 0.19 converts no Optional of a subclass, in the scenario's own
    # code, which the model cannot change (#541): refused with the hint.
    assert not result.ok
    assert (
        "does not convert an Optional of a subclass" in result.format()
    ), result.format()


@pytest.mark.parametrize("argument", ["5", "'text'", "None"])
def test_register_tick_still_refuses_a_non_action_non_callable(
    tmp_path: Path, argument: str
) -> None:
    line = f"self.register_pre_tick({argument})"
    scenario, config, path = _write_case(tmp_path, line + "\n", header=_R29_HEADER)
    result = typecheck_scenario(scenario, config)
    assert any(
        d.line == _line_of(path, line) for d in result.diagnostics
    ), result.format()


@pytest.mark.parametrize(
    "read",
    [
        "lambda w: self._gap(w) or 0",
        "lambda w: self._gap(w) or self._gap(w) or 0.0",
        "lambda w: self._count(w) or 0",
    ],
    ids=["int-default", "nested", "optional-int"],
)
def test_more_measure_reads_with_or_compile(tmp_path: Path, read: str) -> None:
    extra = (
        _R29_EXTRA
        + """
    def _count(self, world: carla.World) -> Optional[int]:
        return None
"""
    )
    setup = f'self.register_measure("a", {read})\n'
    scenario, config, _ = _write_case(tmp_path, setup, extra=extra, header=_R29_HEADER)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_a_measure_read_of_str_or_float_is_refused(tmp_path: Path) -> None:
    extra = """
    def _name(self, world: carla.World) -> Optional[str]:
        return None
"""
    read = "lambda w: self._name(w) or 0.0"
    setup = f'self.register_measure("a", {read})\n'
    scenario, config, _ = _write_case(tmp_path, setup, extra=extra, header=_R29_HEADER)
    result = typecheck_scenario(scenario, config)
    assert not result.ok


def test_the_optional_hint_is_not_given_for_unrelated_optionals(
    tmp_path: Path,
) -> None:
    setup = """
        n: Optional[int] = 3
        self._s = self._take(n)
        """
    extra = """
    _s: int

    def _take(self, s: Optional[str]) -> int:
        return 0
"""
    scenario, config, _ = _write_case(tmp_path, setup, extra=extra, header=_R29_HEADER)
    result = typecheck_scenario(scenario, config)
    assert "does not convert an Optional of a subclass" not in result.format()


@pytest.mark.parametrize(
    "line",
    [
        'logger.info("%s", base.parts[-1])',
        'logger.info("%s", "a" in base.parts)',
        'logger.info("%d", len(base.parts))',
        'logger.info("%s", base.parts[1:])',
        'logger.info("%s", [p.upper() for p in base.parts])',
        'logger.info("%s", "/".join(base.parts[1:]))',
        'logger.info("%s", base.parts[0] == "/")',
    ],
)
def test_a_scenario_may_index_slice_and_iterate_path_parts(
    tmp_path: Path, line: str
) -> None:
    scenario, config, _ = _write_case(
        tmp_path, f'from pathlib import Path\nbase = Path("/a/b.txt")\n{line}\n'
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


# ---------------------------------------------------------------------------
# A field of a scenario's own frozen dataclass cannot be assigned (#567)
# ---------------------------------------------------------------------------

_FROZEN_EXTRA = """

@dataclass(frozen=True)
class Pt:
    x: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", float(self.x))

    def moved(self, dx: float) -> Pt:
        return replace(self, x=self.x + dx)

    def bump(self) -> None:
        BUMP


@dataclass(frozen=True)
class Sub(Pt):
    y: float = 0.0


@dataclass
class Free:
    x: float = 0.0

"""
_FROZEN_HEADER = _HEADER.replace(
    "import logging\n",
    "import logging\nfrom dataclasses import dataclass, replace\n",
    1,
)


def _frozen_case(
    tmp_path: Path, setup: str, bump: str = "pass"
) -> tuple[Any, Any, Path]:
    return _write_case(
        tmp_path,
        setup,
        extra=_FROZEN_EXTRA.replace("BUMP", bump),
        header=_FROZEN_HEADER,
    )


def test_frozen_dataclass_idioms_compile(tmp_path: Path) -> None:
    # Reading a field, replace(), object.__setattr__ in __post_init__ and
    # assigning a field of a dataclass that is not frozen are all valid.
    scenario, config, _ = _frozen_case(
        tmp_path,
        """
        p = Pt(1.0)
        q = p.moved(2.0)
        s = replace(Sub(1.0), y=2.0)
        f = Free(1.0)
        f.x = 2.0
        assert q.x + s.y + f.x > 0.0
        """,
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


@pytest.mark.parametrize(
    "statement",
    [
        "p.x = 2.0",
        "p.x += 1.0",
        "for it in [Pt(1.0)]:\n    it.x = 2.0",
        "s = Sub(1.0)\ns.y = 5.0",
    ],
)
def test_assigning_a_field_of_a_frozen_dataclass_is_refused(
    tmp_path: Path, statement: str
) -> None:
    scenario, config, path = _frozen_case(tmp_path, f"p = Pt(1.0)\n{statement}\n")
    error = _only_error(typecheck_scenario(scenario, config))
    assert "is a frozen dataclass; use dataclasses.replace(x," in error.message
    assert error.path == str(path)
    assert error.line == _line_of(path, statement.splitlines()[-1].strip())


def test_assigning_self_field_in_a_frozen_dataclass_method_is_refused(
    tmp_path: Path,
) -> None:
    scenario, config, path = _frozen_case(tmp_path, "pass\n", bump="self.x = 3.0")
    error = _only_error(typecheck_scenario(scenario, config))
    assert "Pt is a frozen dataclass" in error.message
    assert error.line == _line_of(path, "self.x = 3.0")


# ---------------------------------------------------------------------------
# bytes and str mixed, an Enum compared with a raw value (#574, #575)
# ---------------------------------------------------------------------------

_MIX_EXTRA = """

class Color(enum.Enum):
    RED = "red"
    BLUE = "blue"


class Mode(enum.Enum):
    A = 1
    B = 2


class Level(enum.IntEnum):
    LOW = 1


class Phase(str, enum.Enum):
    GO = "go"


def label(name: str) -> str:
    payload = name.encode()
    return LABEL


def up(s: str) -> str:
    return s.upper()

"""
_MIX_HEADER = _HEADER.replace("import logging\n", "import enum\nimport logging\n", 1)


def _mix_case(tmp_path: Path, setup: str, label: str = '"sent " + name') -> Any:
    return _write_case(
        tmp_path, setup, extra=_MIX_EXTRA.replace("LABEL", label), header=_MIX_HEADER
    )


@pytest.mark.parametrize(
    "statement",
    [
        's = b"x" + "y"',
        'raw = "ab".encode()\ns = "x" in raw',
        'raw = "ab".encode()\nok = raw == "ab"',
        'logger.info("%s", up("ego".encode("utf-8")))',
        'x: str = "ego".encode()',
    ],
    ids=["operator", "in", "equality", "argument", "assignment"],
)
def test_mixing_bytes_and_str_is_refused(tmp_path: Path, statement: str) -> None:
    scenario, config, path = _mix_case(tmp_path, f"{statement}\n")
    error = _only_error(typecheck_scenario(scenario, config))
    assert "bytes and str do not mix in Python" in error.message, error.format()
    assert ".decode()" in error.message and ".encode()" in error.message
    assert error.path == str(path)
    assert error.line == _line_of(path, statement.splitlines()[-1])


def test_a_bytes_returned_for_a_str_is_refused(tmp_path: Path) -> None:
    scenario, config, path = _mix_case(tmp_path, "pass\n", label="payload")
    error = _only_error(typecheck_scenario(scenario, config))
    assert "bytes and str do not mix in Python" in error.message, error.format()
    assert error.line == _line_of(path, "return payload")


@pytest.mark.parametrize(
    "statement",
    [
        'c = Color.RED\nok = c == "red"',
        "m = Mode.A\nok = m == 1",
        "m = Mode.A\nok = m in (1, 2)",
        "ok = Color.RED == Mode.A",
    ],
    ids=["str", "int", "container", "other-enum"],
)
def test_an_enum_compared_with_its_raw_value_is_refused(
    tmp_path: Path, statement: str
) -> None:
    scenario, config, path = _mix_case(tmp_path, f"{statement}\n")
    error = _only_error(typecheck_scenario(scenario, config))
    assert "an Enum member never equals its raw value" in error.message, error.format()
    assert error.path == str(path)
    assert error.line == _line_of(path, statement.splitlines()[-1])


def test_bytes_str_and_enum_look_alikes_compile(tmp_path: Path) -> None:
    # bytes with bytes, str with str, .encode() used as bytes, a member with
    # a member or its .value, an IntEnum with an int and a (str, Enum) with a
    # str (both overlap their raw value in Python as well).
    scenario, config, _ = _mix_case(
        tmp_path,
        """
        raw = "ab".encode()
        joined = raw + raw
        both = "x" + "y"
        same = raw == b"ab"
        c = Color.RED
        logger.info(
            "%s %s %s %s %s %s %s %s",
            len(joined), both, same, c == Color.RED, c.value == "red",
            Level.LOW == 1, Phase.GO == "go", label("a"),
        )
        """,
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


# ---------------------------------------------------------------------------
# A string formatting Python fails on (#589)
# ---------------------------------------------------------------------------

_FORMAT_EXTRA = """


def label(speed: float, name: str) -> str:
    return LABEL
"""


@pytest.mark.parametrize(
    "label",
    ['"{:d} km/h".format(speed)', '"%d km/h" % name', '"%s at %s" % (name,)'],
    ids=["format-spec", "percent-type", "percent-count"],
)
def test_a_string_formatting_python_fails_on_is_refused(
    tmp_path: Path, label: str
) -> None:
    scenario, config, path = _write_case(
        tmp_path,
        'logger.info("%s", label(3.5, "ego"))\n',
        extra=_FORMAT_EXTRA.replace("LABEL", label),
    )
    error = _only_error(typecheck_scenario(scenario, config))
    assert "Python raises ValueError, TypeError" in error.message, error.format()
    assert "(mypy: str-format)" in error.message
    assert error.path == str(path)
    assert error.line == _line_of(path, label)


def test_a_valid_string_formatting_compiles(tmp_path: Path) -> None:
    # A float for "%d", an int for "{:.2f}", "{:05}" of a str (mypy's
    # "numeric flags" message, which Python runs) and an f-string spec.
    scenario, config, _ = _write_case(
        tmp_path,
        'logger.info("%s", label(3.5, "ego"))\n',
        extra=_FORMAT_EXTRA.replace(
            "LABEL",
            '"%d km/h" % speed + " {:.2f} {:05}".format(int(speed), name)'
            ' + f" {speed:.1f}"',
        ),
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


# ---------------------------------------------------------------------------
# dict.pop and getattr with a default (#590)
# ---------------------------------------------------------------------------

_DEFAULTS_EXTRA = """


class HasV:
    v: float

    def __init__(self) -> None:
        self.v = 1.0


class NoV:
    pass


def take(d: dict[str, float], names: dict[str, str]) -> float:
    a = d.pop("a", 0.0)
    b = d.pop("zz", None)
    c = names.pop("k", "default")
    return a + (b if b is not None else 1.0) + len(c)


def read(a: HasV, b: NoV) -> float:
    return getattr(a, "v", 0.0) + getattr(b, "v", 2.0) + READ
"""


def test_dict_pop_and_getattr_with_a_default_compile(tmp_path: Path) -> None:
    scenario, config, _ = _write_case(
        tmp_path,
        'logger.info("%s %s", take({"a": 1.0}, {"k": "v"}), read(HasV(), NoV()))\n',
        extra=_DEFAULTS_EXTRA.replace("READ", "0.0"),
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_getattr_with_a_name_built_at_run_time_is_refused_with_a_hint(
    tmp_path: Path,
) -> None:
    scenario, config, path = _write_case(
        tmp_path,
        'logger.info("%s", read(HasV(), NoV()))\n',
        extra=_DEFAULTS_EXTRA.replace("READ", 'getattr(a, "v" + "", 0.0)'),
    )
    error = _only_error(typecheck_scenario(scenario, config))
    assert "getattr() takes 2 arguments (3 given)" in error.message, error.format()
    assert 'x.name if hasattr(x, "name") else default' in error.message
    assert error.line == _line_of(path, '"v" + ""')


def test_getattr_default_is_checked_where_the_attribute_exists(
    tmp_path: Path,
) -> None:
    # Python evaluates the default even when the attribute exists, so a typo
    # in it fails every call: the rewrite keeps it compiled (#590).
    scenario, config, path = _write_case(
        tmp_path,
        'logger.info("%s", read(HasV(), NoV()))\n',
        extra=_DEFAULTS_EXTRA.replace("READ", 'getattr(a, "v", float(a.vv))'),
    )
    error = _only_error(typecheck_scenario(scenario, config))
    assert "has no attribute 'vv'" in error.message, error.format()
    assert error.line == _line_of(path, "a.vv")


# ---------------------------------------------------------------------------
# Typing contracts broken: Final, ClassVar, override, read-only ABCs (#577)
# ---------------------------------------------------------------------------

_CONTRACT_EXTRA = """

MAX_SPEED: Final = 30.0


class Counter:
    total: ClassVar[int] = 0
    free: int = 0
    LIMIT: Final = 5.0

    def bump(self) -> None:
        BUMP


class Sub(CaseScenario):
    SUB_BODY


def total_of(items: Sequence[float], table: Mapping[str, int]) -> float:
    ordered = sorted(items)
    own = dict(table)
    own["k"] = 1
    return ordered[0] + own["k"]

"""
_CONTRACT_HEADER = _HEADER.replace(
    "import logging\n",
    "import logging\nfrom typing import ClassVar, Final, Mapping, Sequence\n",
    1,
)


def _contract_case(
    tmp_path: Path,
    setup: str = "pass\n",
    bump: str = "pass",
    sub: str = "pass",
    extra: str = "",
) -> tuple[Any, Any, Path]:
    return _write_case(
        tmp_path,
        setup,
        extra=_CONTRACT_EXTRA.replace("BUMP", bump).replace("SUB_BODY", sub) + extra,
        header=_CONTRACT_HEADER,
    )


def test_final_classvar_override_and_mapping_idioms_compile(tmp_path: Path) -> None:
    # Reading a Final, a ClassVar assigned through the class, a hook
    # overridden with a compatible signature (an extra parameter with a
    # default, a wider argument), and a copy of a Mapping that is mutated.
    scenario, config, _ = _contract_case(
        tmp_path,
        """
        Counter.total += 1
        Counter().free += 1
        logger.info("%s %s", MAX_SPEED, total_of([2.0, 1.0], {"a": 1}))
        """,
        bump="Counter.total += 1\n        self.free += 1",
        sub=(
            "def is_done(self, tolerance: float = 0.0) -> bool:\n"
            "        return False\n\n"
            "    @property\n"
            "    def importance_sampling_weight(self) -> float:\n"
            "        return 1.0"
        ),
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


@pytest.mark.parametrize(
    "statement",
    [
        "global MAX_SPEED\nMAX_SPEED = 40.0",
        "speed: Final = 10.0\nspeed = 20.0",
        "Counter.LIMIT = 1.0",
        "Counter().LIMIT += 1.0",
    ],
    ids=["module", "local", "attribute-of-class", "attribute-augmented"],
)
def test_a_final_assigned_again_is_refused(tmp_path: Path, statement: str) -> None:
    scenario, config, path = _contract_case(tmp_path, f"{statement}\n")
    error = _only_error(typecheck_scenario(scenario, config))
    assert "a name declared Final is bound once" in error.message, error.format()
    assert error.path == str(path)
    assert error.line == _line_of(path, statement.splitlines()[-1])


def test_a_final_assigned_through_self_is_refused(tmp_path: Path) -> None:
    scenario, config, path = _contract_case(tmp_path, bump="self.LIMIT = 40.0")
    error = _only_error(typecheck_scenario(scenario, config))
    assert "a name declared Final is bound once" in error.message, error.format()
    assert error.line == _line_of(path, "self.LIMIT = 40.0")


@pytest.mark.parametrize(
    "statement",
    ["Counter().total = 5", "c = Counter()\nc.total += 1"],
    ids=["assign", "augmented"],
)
def test_a_classvar_assigned_through_an_instance_is_refused(
    tmp_path: Path, statement: str
) -> None:
    scenario, config, path = _contract_case(tmp_path, f"{statement}\n")
    error = _only_error(typecheck_scenario(scenario, config))
    assert "a ClassVar is shared by the class" in error.message, error.format()
    assert error.line == _line_of(path, statement.splitlines()[-1])


@pytest.mark.parametrize(
    "sub",
    [
        "def is_done(self, strict: bool) -> bool:\n        return strict",
        "def is_done(self) -> str:\n        return ''",
        "def setup(self, mode: str) -> None:\n        pass",
    ],
    ids=["required-parameter", "return-type", "setup-parameter"],
)
def test_a_framework_hook_overridden_incompatibly_is_refused(
    tmp_path: Path, sub: str
) -> None:
    scenario, config, path = _contract_case(tmp_path, sub=sub)
    result = typecheck_scenario(scenario, config)
    errors = [d for d in result.diagnostics if "the override must accept" in d.message]
    assert len(errors) == 1, result.format()
    assert "# type: ignore[override]" in errors[0].message
    assert errors[0].path == str(path)
    assert errors[0].line == _line_of(path, sub.splitlines()[0])


def test_an_override_with_type_ignore_compiles(tmp_path: Path) -> None:
    scenario, config, _ = _contract_case(
        tmp_path,
        sub=(
            "def is_done(self, strict: bool) -> bool:  # type: ignore[override]\n"
            "        return strict"
        ),
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


@pytest.mark.parametrize(
    "extra",
    [
        "def f(m: Mapping[str, int]) -> None:\n    m['k'] = 1",
        "def f(s: Sequence[int]) -> None:\n    s.append(3)",
        "def f(s: Sequence[float]) -> None:\n    s.sort()",
        "def f(m: Mapping[str, int]) -> None:\n    m.update({'a': 2})",
    ],
    ids=["index", "append", "sort", "update"],
)
def test_mutating_a_mapping_or_sequence_parameter_is_refused(
    tmp_path: Path, extra: str
) -> None:
    scenario, config, path = _contract_case(tmp_path, extra="\n\n" + extra + "\n")
    error = _only_error(typecheck_scenario(scenario, config))
    assert "is read-only by its annotation" in error.message, error.format()
    assert error.line == _line_of(path, extra.splitlines()[-1].strip())


# ---------------------------------------------------------------------------
# An isinstance branch the check does not compile warns (#565)
# ---------------------------------------------------------------------------


_DEAD_HELPER = """
    def helper(self, r: float) -> None:
        if isinstance(r, int):
            q: int = 1
"""


@pytest.mark.parametrize(
    ("setup", "extra"),
    [
        ('x = 5\nif isinstance(x, str):\n    q: int = "wrong"\n', ""),
        ("pass\n", _DEAD_HELPER),
    ],
)
def test_a_statically_false_isinstance_warns_that_its_branch_is_not_checked(
    tmp_path: Path, setup: str, extra: str
) -> None:
    scenario, config, path = _write_case(tmp_path, setup, extra=extra)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()
    warning = next(w for w in result.warnings if "isinstance(" in w.message)
    assert warning.severity == "warning"
    assert "not checked by the static check" in warning.message
    assert warning.line == _line_of(path, "if isinstance(")


def test_a_valid_isinstance_does_not_warn(tmp_path: Path) -> None:
    scenario, config, _ = _write_case(
        tmp_path,
        "n = 5\nif isinstance(n, int):\n    q: int = n\n",
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()
    assert not [w for w in result.warnings if "isinstance(" in w.message]


# ---------------------------------------------------------------------------
# A local typed by its first assignment: an int literal, None (#583, #584)
# ---------------------------------------------------------------------------


_INT_ACCUMULATOR = """


def mean_speed(speeds: list[float]) -> float:
    total = {start}
    for v in speeds:
        total += v
    return total / max(len(speeds), 1)
"""


@pytest.mark.parametrize(("start", "ok"), [("0", False), ("0.0", True)])
def test_a_float_stored_in_a_local_first_bound_to_an_int_literal_names_it(
    tmp_path: Path, start: str, ok: bool
) -> None:
    scenario, config, path = _write_case(
        tmp_path, "pass\n", extra=_INT_ACCUMULATOR.format(start=start)
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok == ok, result.format()
    if ok:
        return
    error = _only_error(result)
    assert error.line == _line_of(path, "total += v")
    first = _line_of(path, "total = 0")
    assert f"`total = 0` on line {first}" in error.message, error.format()
    assert "`total = 0.0`" in error.message


def test_an_int_literal_branch_of_a_conditional_expression_is_named(
    tmp_path: Path,
) -> None:
    extra = (
        "\n\ndef floor_speed(v: float | None) -> float:\n"
        "    s = v if v is not None else 0\n"
        "    return s * 2.0\n"
    )
    scenario, config, path = _write_case(tmp_path, "pass\n", extra=extra)
    error = _only_error(typecheck_scenario(scenario, config))
    assert error.line == _line_of(path, "s = v if")
    assert "Optional[int]" in error.message
    assert "int literal branch `0`" in error.message, error.format()
    assert "`0.0`" in error.message


_RUNNING_MIN = """


def seen_low(xs: list[float]) -> {result}:
    best{annotation} = None
    for x in xs:
        if best is None or x < best:
            best = x
{tail}"""


@pytest.mark.parametrize(
    ("annotation", "result", "tail"),
    [
        ("", "bool", "    return best is not None\n"),
        ("", "None", ""),
        (": float | None", "bool", "    return best is not None\n"),
    ],
    ids=["crash", "optional-nonetype", "annotated"],
)
def test_a_local_first_bound_to_none_and_rebound_names_its_annotation(
    tmp_path: Path, annotation: str, result: str, tail: str
) -> None:
    extra = _RUNNING_MIN.format(annotation=annotation, result=result, tail=tail)
    scenario, config, path = _write_case(tmp_path, "pass\n", extra=extra)
    checked = typecheck_scenario(scenario, config)
    if annotation:
        assert checked.ok, checked.format()
        return
    assert not checked.ok, checked.format()
    hinted = [
        d
        for d in checked.diagnostics
        if "annotate it: `best: float | None = None`" in d.message
    ]
    assert hinted, checked.format()
    first = _line_of(path, "best = None")
    assert f"`best = None` on line {first}" in hinted[0].message
    if checked.crashed:
        # Codon 0.19 segfaults: the crash is kept, the cause is located.
        assert "Codon failed without a diagnostic" in checked.diagnostics[0].message
        assert hinted[0].line == first
    else:
        assert "Optional[NoneType]" in hinted[0].message
        assert hinted[0].line == _line_of(path, "best = x")


# ---------------------------------------------------------------------------
# Identity == and hash of a plain class (#596)
# ---------------------------------------------------------------------------

_IDENTITY_CLASSES = """


from abc import ABC
from dataclasses import dataclass


class Npc:
    name: str

    def __init__(self, name: str) -> None:
        self.name = name


class Car(Npc):
    def describe(self) -> str:
        return self.name


class Hashed:
    def __hash__(self) -> int:
        return 3


@dataclass(eq=False)
class Loose:
    x: float


class Root(ABC):
    pass


class Leaf(Root):
    pass


class Mine(BaseCondition):
    def __init__(self, label: str) -> None:
        super().__init__(label=label)

    def check(self, world: carla.World, elapsed: float) -> ScenarioResult | None:
        return None


class Keyed(BaseCondition):
    def __init__(self, label: str) -> None:
        super().__init__(label=label)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Keyed) and other.label == self.label

    # Called by name, over the model's identity __eq__ it overrides.
    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash(self.label)

    def check(self, world: carla.World, elapsed: float) -> ScenarioResult | None:
        return None


@dataclass
class Fielded(Loose):
    y: float

    # Over Loose's identity __eq__: the generated __ne__ calls it.
    def __eq__(self, other: object) -> bool:
        return isinstance(other, Fielded) and other.y == self.y

    def __hash__(self) -> int:
        return hash(self.y)


KNOWN = {Npc("known")}
"""


def test_a_plain_class_compares_and_hashes_by_identity(tmp_path: Path) -> None:
    setup = """
        a = Npc("a")
        c = Car("c")
        npcs: list[Npc] = [a, c]
        if a in npcs:
            npcs.remove(a)
        gaps: dict[Npc, float] = {}
        gaps[c] = 1.0
        seen: set[Npc] = set()
        seen.add(a)
        index = {a: 1, c: 2}
        r = Root()
        logger.info(
            "%s %s %d %d %d %d %d %d %d",
            a == c, a != c, npcs.index(c), len(gaps), len(seen), index[c],
            len({Hashed()}), len({Loose(1.0)}) + int(Loose(1.0) == Loose(1.0)),
            len(KNOWN) + len({r, Leaf()}),
        )
        t = TimeoutCondition(1.0, label="a")
        turn = TurnAction(EGO_ROLE_NAME, TurnDirection.LEFT, timing=TickTiming.PRE_TICK)
        conds = {t, TimeoutCondition(2.0, label="b")}
        by_cond = {t: "a"}
        mine = Mine("m")
        listed = [mine, t]
        logger.info(
            "%d %s %s %d %s %d %d",
            len(conds), by_cond[t], mine in listed, len({turn}), turn == turn,
            len({Keyed("k"), Keyed("k")}), len({mine: 1}),
        )
        f = Fielded(1.0, 2.0)
        logger.info(
            "%s %s %s", Keyed("k") != Keyed("j"), Keyed("k").__eq__(t), f != f
        )
        if self.ego_entity is not None:
            logger.info("%d", len({self.ego_entity}))
    """
    scenario, config, _ = _write_case(
        tmp_path, _TIMEOUT_SETUP + textwrap.dedent(setup), _IDENTITY_CLASSES
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


@pytest.mark.parametrize(
    ("extra", "use", "hint"),
    [
        (
            # Python sets __hash__ = None on a class defining __eq__ alone.
            "\n\nclass Pt:\n    x: int\n\n    def __init__(self, x: int) -> None:\n"
            "        self.x = x\n\n    def __eq__(self, other: object) -> bool:\n"
            "        return isinstance(other, Pt) and other.x == self.x\n",
            "seen = {Pt(1), Pt(1)}",
            "Pt defines __eq__ and not __hash__, so Python makes it unhashable",
        ),
        (
            # A plain @dataclass: eq=True and not frozen, unhashable too.
            "\n\nfrom dataclasses import dataclass\n\n\n@dataclass\nclass Goal:\n"
            "    x: float\n",
            "seen = {Goal(1.0)}",
            "the check generates hash() for a @dataclass(frozen=True)",
        ),
    ],
    ids=["eq-only", "dataclass"],
)
def test_a_class_python_cannot_hash_is_refused_as_a_key_with_a_hint(
    tmp_path: Path, extra: str, use: str, hint: str
) -> None:
    scenario, config, path = _write_case(
        tmp_path, _TIMEOUT_SETUP + use + '\nlogger.info("%d", len(seen))\n', extra
    )
    error = _only_error(typecheck_scenario(scenario, config))
    assert "has no attribute '__hash__'" in error.message, error.format()
    assert hint in error.message, error.format()
    assert error.line == _line_of(path, use), error.format()


# ---------------------------------------------------------------------------
# re.Match.group() is a str (#597)
# ---------------------------------------------------------------------------


def test_a_regex_group_is_a_str_and_a_narrowed_branch_unifies_with_none(
    tmp_path: Path,
) -> None:
    setup = r"""
        m = re.match(r"npc(\d+)", cfg.name)
        a = m.group(1) if m else ""
        b = m.group(1) if m is not None else ""
        c = "" if m is None else m.group(1)
        d = m[1] if m else ""
        e: str = m.group() if m else ""
        f = m.group(1).upper() if m else ""
        named = re.fullmatch(r"(?P<word>\w+)-(\d+)?", cfg.name)
        g = ""
        if named:
            first, second = named.groups()
            h: str = named.groupdict()["word"]
            g = first + second + h + named.group("word") + named["word"]
            whole, word = named.group(0, 1)
            g += whole + word
        parts = re.split(r"(-)", cfg.name)
        logger.info(
            "%s %s %s %s %s %s %s %d %s %s",
            a, b, c, d, e, f, g, len(parts), group_or_none(cfg.name),
            name_or_none(None),
        )
    """
    extra = r"""


class Named:
    name: str

    def __init__(self, name: str) -> None:
        self.name = name


def group_or_none(text: str) -> str | None:
    m = re.search(r"(\d+)", text)
    return m.group(1) if m else None


def name_or_none(n: Named | None) -> str | None:
    return n.name if n is not None else None
"""
    scenario, config, _ = _write_case(
        tmp_path,
        _TIMEOUT_SETUP + textwrap.dedent(setup),
        extra,
        header=_stdlib_header("import re\n"),
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


# ---------------------------------------------------------------------------
# What Python raises for, decided from the syntax (#602, #603)
# ---------------------------------------------------------------------------

_LITERAL_HEADER = _HEADER.replace(
    "from .configs import CaseConfig\n",
    "from autoware_carla_scenario import (\n"
    "    MetricCondition,\n"
    "    RouteCrossingPose,\n"
    "    RouteRoadsidePose,\n"
    ")\n\nfrom .configs import CaseConfig\n",
    1,
)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            'MetricCondition("no_such_metric", label="m")',
            "MetricCondition: 'no_such_metric' is not a metric; known: ",
        ),
        (
            'ElapsedTimeCondition(1.0, label="")',
            "ElapsedTimeCondition: label must not be empty",
        ),
        (
            'RouteCrossingPose(approach="Left")',
            "RouteCrossingPose.approach must be one of ['left', 'right', "
            "'opposite'], got 'Left'",
        ),
        (
            'RouteRoadsidePose(side="kerb")',
            "RouteRoadsidePose.side must be one of ['left', 'right'], got 'kerb'",
        ),
    ],
)
def test_a_literal_argument_the_framework_refuses_is_refused_at_the_call(
    tmp_path: Path, call: str, message: str
) -> None:
    setup = _TIMEOUT_SETUP + f"logger.info('%s', {call})\n"
    scenario, config, path = _write_case(tmp_path, setup, header=_LITERAL_HEADER)
    error = _only_error(typecheck_scenario(scenario, config))
    assert "raises ValueError when the scenario runs" in error.message
    assert message in error.message, error.format()
    assert error.path == str(path)
    assert error.line == _line_of(path, call)
    line = path.read_text().splitlines()[error.line - 1]
    assert line[error.column - 1 :].startswith(call)


def test_literal_arguments_the_framework_accepts_pass(tmp_path: Path) -> None:
    setup = _TIMEOUT_SETUP + textwrap.dedent(
        """
        self.register_fail_condition(MetricCondition("offroad", label="m"))
        traj = Trajectory(
            "t",
            [
                TrajectoryVertex(RouteCrossingPose(approach="left", distance=5.0)),
                TrajectoryVertex(RouteRoadsidePose(ds=3.0, side="right")),
            ],
        )
        self.register_pre_tick(FollowTrajectoryAction("npc", traj))
        """
    )
    scenario, config, _ = _write_case(tmp_path, setup, header=_LITERAL_HEADER)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


@pytest.mark.parametrize(
    ("test", "generic"),
    [
        ("isinstance(v, list[int])", "list[int]"),
        ("isinstance(v, (str, dict[str, int]))", "dict[str, int]"),
        ("issubclass(type(v), dict[str, int])", "dict[str, int]"),
    ],
)
def test_isinstance_with_a_subscripted_generic_is_refused_at_the_call(
    tmp_path: Path, test: str, generic: str
) -> None:
    setup = _TIMEOUT_SETUP + textwrap.dedent(
        f"""
        v: object = cfg.goal_lanelet_ids
        if {test}:
            logger.info("list")
        """
    )
    scenario, config, path = _write_case(tmp_path, setup)
    error = _only_error(typecheck_scenario(scenario, config))
    assert f"cannot take the subscripted generic `{generic}`" in error.message
    assert "Python raises TypeError" in error.message
    assert error.path == str(path)
    assert error.line == _line_of(path, test)


def test_isinstance_with_a_union_of_classes_passes(tmp_path: Path) -> None:
    setup = _TIMEOUT_SETUP + textwrap.dedent(
        """
        n = cfg.timeout_seconds
        if isinstance(n, int | float) and isinstance(n, (int, float)):
            logger.info("%f", n)
        """
    )
    scenario, config, _ = _write_case(tmp_path, setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_a_class_decorator_is_refused_with_a_hint(tmp_path: Path) -> None:
    extra = """


class Base:
    def speed(self) -> float:
        return 0.0


def register(cls):
    return cls


@register
class Fast(Base):
    def speed(self) -> float:
        return 10.0
"""
    setup = _TIMEOUT_SETUP + 'logger.info("%f", Fast().speed())\n'
    scenario, config, path = _write_case(tmp_path, setup, extra)
    error = _only_error(typecheck_scenario(scenario, config))
    assert error.message.startswith("unsupported class decorator: "), error.format()
    assert "choose the class with an if/elif chain" in error.message
    assert error.line == _line_of(path, "@register")


def test_an_async_def_is_refused_with_a_hint(tmp_path: Path) -> None:
    extra = """


async def fetch() -> int:
    return 1
"""
    scenario, config, path = _write_case(tmp_path, _TIMEOUT_SETUP, extra)
    error = _only_error(typecheck_scenario(scenario, config))
    assert error.message.startswith("async not yet supported: "), error.format()
    assert "make it a plain `def`" in error.message
    assert error.line == _line_of(path, "async def fetch")


@pytest.mark.parametrize(
    ("members", "shown"),
    [
        (
            "    def __len__(self) -> int:\n        return len(self.items)\n",
            "'Bag[List[int]]'",
        ),
        ("", "'Bag[List[int]]'"),
    ],
)
def test_the_truth_of_an_object_without_bool_is_refused_with_a_hint(
    tmp_path: Path, members: str, shown: str
) -> None:
    extra = (
        "\n\nclass Bag:\n"
        "    def __init__(self) -> None:\n"
        "        self.items: list[int] = []\n\n" + members
    )
    setup = _TIMEOUT_SETUP + 'b = Bag()\nif b:\n    logger.info("nonempty")\n'
    scenario, config, path = _write_case(tmp_path, setup, extra)
    error = _only_error(typecheck_scenario(scenario, config))
    assert error.message.startswith(
        f"{shown} object has no attribute '__bool__': "
    ), error.format()
    assert "`x is not None`" in error.message
    assert "`len(x) > 0`" in error.message
    assert error.line == _line_of(path, "if b:")


def test_a_conditional_expression_of_two_types_is_refused_with_a_hint(
    tmp_path: Path,
) -> None:
    setup = _TIMEOUT_SETUP + textwrap.dedent(
        """
        label = cfg.name if cfg.min_speed_kmh is None else 3
        logger.info("%s", label)
        """
    )
    scenario, config, path = _write_case(tmp_path, setup)
    error = _only_error(typecheck_scenario(scenario, config))
    assert "does not match expected type" in error.message, error.format()
    assert "a conditional expression (`a if c else b`) has one type" in error.message
    assert error.line == _line_of(path, "label = cfg.name if")


# ---------------------------------------------------------------------------
# Hints for common slips (#599)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("setup", "needle", "codon", "hint"),
    [
        (
            "for i in range(len(cfg.goal_lanelet_ids) / 2):\n"
            '    logger.info("%d", i)\n',
            "for i in range(",
            "range() cannot be called with (float)",
            "range() takes ints, and `/` always gives a float",
        ),
        (
            "ids = cfg.goal_lanelet_ids\n"
            "mid = ids[len(ids) / 2]\n"
            'logger.info("%d", mid)\n',
            "mid = ids[",
            "__getitem__() cannot be called with (float)",
            "an index must be an int, and `/` always gives a float",
        ),
        (
            'logger.info("ids: %s", ",".join(cfg.goal_lanelet_ids))\n',
            '",".join(',
            "'Generator[int]' does not match expected type 'Generator[str]'",
            "`sep.join(map(str, values))`",
        ),
        (
            "self.register_pre_tick(lambda: None)\n",
            "register_pre_tick(",
            "takes 0 arguments (1 given)",
            "register_pre_tick() is called with the world (a carla.World): "
            "take it as a parameter, `lambda world: ...`",
        ),
        (
            'self.register_measure("flag", lambda w: True, unit="n")\n',
            'register_measure("flag"',
            "expected a float (or an int)",
            "the read function (the second argument) of measure 'flag' returns "
            "a bool, and a measure is a float (or an int), or None",
        ),
    ],
    ids=["range-of-float", "float-index", "join-of-ints", "tick-arity", "measure-bool"],
)
def test_a_common_slip_is_refused_with_a_hint(
    tmp_path: Path, setup: str, needle: str, codon: str, hint: str
) -> None:
    scenario, config, path = _write_case(tmp_path, setup)
    error = _only_error(typecheck_scenario(scenario, config))
    assert codon in error.message, error.format()
    assert hint in error.message, error.format()
    assert error.line == _line_of(path, needle), error.format()
    # Only the hint that applies: not the one about lambda defaults (`k=k`).
    assert "k=k" not in error.message, error.format()


# ---------------------------------------------------------------------------
# Enum values the check cannot type (#600)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("members", "why"),
    [
        (
            '    LEFT = (-3.5, "left")\n'
            '    RIGHT = (3.5, "right")\n\n'
            "    def __init__(self, offset: float, label: str) -> None:\n"
            "        self.offset = offset\n"
            "        self.label = label\n",
            "its member `LEFT` has a tuple value",
        ),
        (
            '    LEFT = 1\n    RIGHT = "r"\n',
            "its members' values are of different types (int, str)",
        ),
        (
            "    LEFT = WIDTH\n    RIGHT = 2 * WIDTH\n",
            "the value of its member `LEFT` is not an int",
        ),
    ],
    ids=["tuple", "mixed", "not-literal"],
)
def test_an_enum_whose_value_type_cannot_be_told_is_refused_at_its_class(
    tmp_path: Path, members: str, why: str
) -> None:
    extra = (
        "\n\nfrom enum import Enum\n\nWIDTH = 3.5\n\n\nclass Lane(Enum):\n" + members
    )
    scenario, config, path = _write_case(
        tmp_path, 'logger.info("%s", Lane.LEFT.name)\n', extra=extra
    )
    error = _only_error(typecheck_scenario(scenario, config))
    assert f"Enum Lane cannot be checked: {why}" in error.message, error.format()
    assert "literals of one type" in error.message
    assert error.line == _line_of(path, "class Lane(Enum):")


def test_an_enum_with_negative_values_compiles(tmp_path: Path) -> None:
    extra = (
        "\n\nfrom enum import Enum\n\n\nclass Side(Enum):\n"
        "    LEFT = -1\n    CENTER = 0\n    RIGHT = +1\n"
    )
    scenario, config, _ = _write_case(
        tmp_path,
        'logger.info("%s %d %s", Side.LEFT.name, Side.LEFT.value, Side(-1))\n'
        "assert [s.value for s in Side] == [-1, 0, 1]\n",
        extra=extra,
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


# ---------------------------------------------------------------------------
# Builtin gaps (#601)
# ---------------------------------------------------------------------------

_BUILTIN_GAPS = """


import itertools
from typing import Optional


class Base:
    k: float

    def __init__(self, k: float) -> None:
        self.k = k


class Derived(Base):
    def __init__(self, k: float) -> None:
        super(Derived, self).__init__(k)


class Counted:
    hits: list[int]

    def __init__(self) -> None:
        self.hits = [1, 2, 2]

    def count(self, x: int) -> int:
        return self.hits.count(x)


def gaps(d: dict[str, int], e: dict[str, int], pair: tuple[int, str]) -> str:
    xs = [1, 2, 3, 2]
    i: int = xs.index(2, 2) + xs.index(2, 0, 2) + xs.index(3, -2) + xs.index(1)
    t = (1, 2, 2)
    n: int = t.count(2) + t.index(2) + t.index(2, 2) + (4, 5, 4).count(4)
    m: int = pair.count(1) + pair.index("a") + "banana".count("a") + xs.count(2)
    c: int = Counted().count(2) + next(itertools.count(5))
    xs += (4, 5)
    xs += [6]
    xs += []
    grown = []
    grown += [1]
    maybe: list[Optional[int]] = []
    maybe += [1]
    maybe += [None]
    floats: list[float] = [1.0]
    floats += [2, 3]
    bases: list[Base] = []
    bases += [Derived(1.0)]
    common: set[str] = d.keys() & e.keys()
    only: set[str] = d.keys() - {"a"}
    either: set[str] = d.keys() | ["z"]
    pairs: set[tuple[str, int]] = d.items() ^ e.items()
    sub: bool = d.items() <= e.items()
    same: bool = d.keys() == e.keys()
    back: list[str] = list(reversed(d))
    table = str.maketrans("lo", "01")
    a = "hello".translate(table)
    b = "hello".translate(str.maketrans("l", "L", "o"))
    f = "hello".translate(str.maketrans({"h": "H"}))
    return f"{i} {n} {m} {c} {xs} {common} {only} {either} {pairs} {sub} {same} {back} {a} {b} {f} {Derived(2.0).k} {grown} {len(maybe)} {floats} {len(bases)}"
"""


def test_the_builtin_gaps_of_601_compile(tmp_path: Path) -> None:
    scenario, config, _ = _write_case(
        tmp_path,
        'logger.info("%s", gaps({"a": 1}, {"a": 2}, (1, "a")))\n',
        extra=_BUILTIN_GAPS,
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_count_of_a_value_that_has_none_is_still_refused(tmp_path: Path) -> None:
    # The `hasattr` branch keeps a tuple working; anything else without the
    # method is still refused, at the author's line (#601).
    scenario, config, path = _write_case(
        tmp_path,
        "v = 2.5\n" 'logger.info("%d", v.count(2))\n',
    )
    error = _only_error(typecheck_scenario(scenario, config))
    assert "count" in error.message, error.format()
    assert error.line == _line_of(path, "v.count(2)"), error.format()


def test_sys_version_info_is_refused_with_a_hint(tmp_path: Path) -> None:
    scenario, config, path = _write_case(
        tmp_path,
        "import sys\n" "if sys.version_info >= (3, 11):\n" '    logger.info("new")\n',
    )
    error = _only_error(typecheck_scenario(scenario, config))
    assert "cannot import name 'version_info' from 'sys'" in error.message
    assert "sys.version_info (and sys.version, sys.hexversion) is not modelled" in (
        error.message
    ), error.format()
    assert error.line == _line_of(path, "sys.version_info")


_REWRITES_IN_FSTRINGS = """


class Obj:
    a: int = 1


class Hits:
    hits: tuple[int, int] | None = (1, 2)
"""


def test_rewrites_inside_fstrings_and_on_optional_tuples_compile(
    tmp_path: Path,
) -> None:
    # A rewrite's string in a single-quoted f-string ended it ("unexpected
    # dedent"), `dict(a=1)` became an escaped brace, and `count` / `index` of
    # a tuple narrowed from an Optional was refused (#609).
    scenario, config, _ = _write_case(
        tmp_path,
        "o = Obj()\n"
        "xs = [1, 2, 1]\n"
        "t = (1, 2)\n"
        'name = "lane_a_b"\n'
        "logger.info(f'{xs.count(1)} {name.count(\"_\")} {xs.index(2)} {t.count(1)}')\n"
        'logger.info(f\'{getattr(o, "a", 1)} {name.startswith(("a", "b"))}\')\n'
        'logger.info(f"{dict(a=1)} {dict(b=2)!r}")\n'
        "tup: tuple[int, int] | None = (1, 1) if cfg.name else None\n"
        "c = tup.count(1) if tup is not None else 0\n"
        "i = tup.index(1) if tup is not None else 0\n"
        "h = Hits()\n"
        "n = h.hits.count(1) if h.hits is not None else 0\n"
        'logger.info("%d %d %d", c, i, n)\n',
        extra=_REWRITES_IN_FSTRINGS,
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


_OPTIONAL_OWN_COUNT = """


class Tally:
    hits: dict[str, float]

    def __init__(self) -> None:
        self.hits = {}

    def count(self, kind: str) -> float:
        return self.hits.get(kind, 0.0)


class Route:
    def index(self, name: str) -> str:
        return name.upper()


class Holder:
    route: Route | None
    tally: Tally | None
    pair: tuple[int, int] | None

    def __init__(self, on: bool) -> None:
        self.route = Route() if on else None
        self.tally = Tally() if on else None
        self.pair = (1, 2) if on else None

    def label(self) -> str:
        if self.route is not None and self.tally is not None:
            return self.route.index("a") + "!" + str(self.tally.count("npc") / 2.0)
        return ""

    def positions(self) -> int:
        if self.pair is not None:
            n: int = self.pair.count(1) + self.pair.index(2) + self.pair.index(2, 1)
            return n
        return 0
"""


def test_count_and_index_of_an_optional_class_of_yours_keep_their_types(
    tmp_path: Path,
) -> None:
    # Unwrapping an Optional, the model's `_acs_count` / `_acs_index` call
    # the class's own method: declared `-> int`, they refused a count() of a
    # float and an index() of a str (#620). An Optional tuple's still type
    # as int.
    scenario, config, _ = _write_case(
        tmp_path,
        "h = Holder(cfg.timeout_seconds > 1.0)\n"
        'logger.info("%s %d", h.label(), h.positions())\n',
        extra=_OPTIONAL_OWN_COUNT,
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_a_mixin_attribute_its_subclass_framework_base_assigns_compiles(
    tmp_path: Path,
) -> None:
    # BaseScenario.__init__ assigns ego_config on the mixin's subclass, so
    # the declared attribute is assigned (#620).
    header = _HEADER.replace(
        "class CaseScenario(BaseScenario):",
        "class VehicleInfoMixin:\n"
        "    ego_config: EgoConfig\n\n"
        "    def vehicle_label(self) -> str:\n"
        '        return "ego:" + self.ego_config.vehicle_type\n\n\n'
        "class CaseScenario(BaseScenario, VehicleInfoMixin):",
    )
    scenario, config, _ = _write_case(
        tmp_path,
        'logger.info("%s", self.vehicle_label())\n',
        header=header,
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_a_crash_of_the_tuple_helpers_is_compiled_again_for_a_diagnostic(
    tmp_path: Path,
) -> None:
    # Codon crashed (exit -11) instantiating `_acs_index` with the type the
    # list display's error left; compiled again without the rewrite, the
    # display's error is reported (#609). The display is unannotated: an
    # annotated one is written from its annotation and compiles (#611).
    scenario, config, path = _write_case(
        tmp_path,
        "a = Npc()\n" "lst = [None, a]\n" 'logger.info("%d", lst.index(a))\n',
        extra="\n\nclass Npc:\n    pass\n",
    )
    result = typecheck_scenario(scenario, config)
    error = _only_error(result)
    assert not result.crashed, result.format()
    assert "Optional[NoneType]" in error.message, error.format()
    assert error.line == _line_of(path, "lst = [None, a]"), error.format()


@pytest.mark.parametrize(
    ("setup", "extra", "needle"),
    [
        ("o = Obj()\ns = f\"{o.a} {o.a + 'x'}\"\n", _REWRITES_IN_FSTRINGS, "s = f"),
        ('n = 1\ns = "a" + f"{n + \'x\'}"\nt = 2\n', "", "s = "),
        # On the last line of the file: Codon put it past its end.
        (
            'logger.info("%s", label(1))\n',
            "\n\ndef label(n: int) -> str:\n    return f\"{n + 'x'}\"\n",
            "return f",
        ),
        # Implicitly concatenated, on one line and on several (#616).
        ('n = 1\ns = "a" f"b{n + \'x\'}"\nt = 2\n', "", "s = "),
        (
            'n = 1\ns = ("aaa"\n     "bb"\n     f"b{n} {n + \'x\'}")\nt = 2\n',
            "",
            "f\"b{n} {n + 'x'}\"",
        ),
        # An error of a decorator's f-string, which the check lowers (#616).
        (
            'logger.info("%d", helper(2))\n',
            "\n\ndef tag(label: str):\n    def deco(f):\n        return f\n"
            "    return deco\n\n\ndef helper(k: int) -> int:\n"
            "    @tag(f\"lane-{k + 'x'}\")\n    def inner() -> int:\n"
            "        return k\n    return inner()\n",
            "@tag(f",
        ),
    ],
)
def test_an_error_in_an_fstring_field_is_reported_at_its_line(
    tmp_path: Path, setup: str, extra: str, needle: str
) -> None:
    # Codon reports it on the next line, at no column of it (#610).
    scenario, config, path = _write_case(tmp_path, setup, extra=extra)
    error = _only_error(typecheck_scenario(scenario, config))
    assert "unsupported operand" in error.message, error.format()
    assert (error.path, error.line, error.column) == (
        str(path),
        _line_of(path, needle),
        None,
    ), error.format()


@pytest.mark.parametrize(
    ("setup", "needle"),
    [
        ("n = 1\ns = f\"{n}\"\nt = n + 'x'\n", "t = n + 'x'"),
        # Codon locates a comprehension's error from its `for`, which no node
        # of Python's tree starts at: the f-string's field of the same width
        # did not take it (#610).
        (
            'abcd = 1\nnnnn = 5\ns = f"{abcd + 123456}"\n'
            'x = [str(c) + "-suffix-" for c in nnnn]\n',
            "x = [str(c)",
        ),
    ],
)
def test_an_error_after_an_fstring_stays_on_its_line(
    tmp_path: Path, setup: str, needle: str
) -> None:
    scenario, config, path = _write_case(tmp_path, setup)
    error = _only_error(typecheck_scenario(scenario, config))
    assert error.line == _line_of(path, needle), error.format()
    assert error.column is not None, error.format()


def test_a_crash_with_another_cause_is_not_hidden_by_the_rerun(
    tmp_path: Path,
) -> None:
    # Compiled again without the count / index rewrite, a valid
    # `t.count(1)` of a tuple is refused: that is no diagnostic of the crash
    # (a local first bound to None, #584), which stays reported (#609).
    scenario, config, path = _write_case(
        tmp_path,
        't = (1, 2)\nlogger.info("%d", t.count(1))\n',
        extra=_RUNNING_MIN.format(
            annotation="", result="bool", tail="    return best is not None\n"
        ),
    )
    result = typecheck_scenario(scenario, config)
    assert result.crashed, result.format()
    assert not [d for d in result.diagnostics if "count" in d.message], result.format()
    assert any(
        "annotate it: `best: float | None = None`" in d.message
        for d in result.diagnostics
    ), result.format()


_ANNOTATED_DISPLAYS = """


class Npc:
    pass


class Bag:
    items: list[int] = []
    tags: dict[str, int] = {}
    seen: set[str] = set()

    def add(self, v: int) -> None:
        self.items.append(v)
        self.tags[str(v)] = v
        self.seen.add(str(v))


class Node:
    # Its own class is not defined in its body: left `[]` (Codon crashes on
    # `List[Node]()` there), and the `__init__` rebinds it.
    children: list[Node] = []

    def __init__(self) -> None:
        self.children = []


def displays(a: Npc) -> str:
    lst: list[Npc | None] = [None, a]
    d: dict[str, Npc | None] = {"x": None, "y": a}
    e: list[int] = []
    e.append(1)
    return f"{lst.index(a)} {len(d)} {e}"
"""


def test_annotated_empty_and_none_first_displays_compile(tmp_path: Path) -> None:
    # #611: an empty display, or one whose first element is None, is typed
    # from the annotation of its target, also as a class-level default of a
    # class with no __init__ (which Codon's constructor takes).
    scenario, config, _ = _write_case(
        tmp_path,
        "b = Bag()\n"
        "b.add(1)\n"
        "n = Node()\n"
        "n.children.append(Node())\n"
        'logger.info("%d %d %s", len(b.items), len(n.children), displays(Npc()))\n',
        extra=_ANNOTATED_DISPLAYS,
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_an_unannotated_empty_class_attribute_gets_a_hint(tmp_path: Path) -> None:
    scenario, config, path = _write_case(
        tmp_path,
        'Registry.items.append("x")\nlogger.info("%d", len(Registry.items))\n',
        extra="\n\nclass Registry:\n    items = []\n",
    )
    error = _only_error(typecheck_scenario(scenario, config))
    assert "expected type 'NoneType'" in error.message, error.format()
    assert "`items: list[str] = []`" in error.message, error.format()
    assert error.line == _line_of(path, 'Registry.items.append("x")')


def test_the_builtin_idioms_of_613_compile(tmp_path: Path) -> None:
    # dict(base, a=1) is `(dict(base) | {"a": 1})`, filter(None, xs) a
    # generator of the truthy items.
    scenario, config, _ = _write_case(
        tmp_path,
        'base = {"x": 1}\n'
        "merged = dict(base, a=1, b=2)\n"
        "xs = [0, 1, 2]\n"
        "nz = list(filter(None, xs))\n"
        'logger.info("%d %s", merged["a"], nz)\n'
        'logger.info(f"{dict(base, c=3)}")\n',
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_a_class_deriving_from_a_builtin_is_refused_at_its_line(
    tmp_path: Path,
) -> None:
    scenario, config, path = _write_case(
        tmp_path,
        'r = Registry()\nr["a"] = 1\nlogger.info("%d", len(r))\n',
        extra="\n\nclass Registry(dict[str, int]):\n    pass\n",
    )
    error = _only_error(typecheck_scenario(scenario, config))
    assert "derives from the builtin `dict[str, int]`" in error.message
    assert "`items: dict[K, V]`" in error.message
    assert error.line == _line_of(path, "class Registry(")


_LANES = """


@dataclass(frozen=True)
class Plan:
    lanes: tuple[int, ...] = ()

    def wider(self) -> Plan:
        return replace(self, lanes=(1, 1, 1, 1))
"""


def test_a_tuple_display_for_a_variable_length_field_compiles(
    tmp_path: Path,
) -> None:
    # #614: a `tuple[X, ...]` field is checked as a `List[X]`; a tuple display
    # given to it, by the constructor or `replace()` of a told class, is one.
    scenario, config, _ = _write_case(
        tmp_path,
        "p = Plan(lanes=(1, 2, 1))\n"
        "q = replace(Plan(), lanes=(1, 2))\n"
        'logger.info("%d %d %d", len(p.lanes), len(q.lanes), len(p.wider().lanes))\n',
        extra=_LANES,
        header=_FROZEN_HEADER,
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_a_tuple_display_replace_it_cannot_type_is_refused_at_the_call(
    tmp_path: Path,
) -> None:
    # The object's class is not told: refused with a hint, at the `replace`
    # call rather than at the class line of the method it is lowered to.
    scenario, config, path = _write_case(
        tmp_path,
        "p = Plan()\n"
        "q = replace(p, lanes=(1, 2, 1))\n"
        'logger.info("%d", len(q.lanes))\n',
        extra=_LANES,
        header=_FROZEN_HEADER,
    )
    error = _only_error(typecheck_scenario(scenario, config))
    assert "'Tuple[int,int,int]' does not match expected type 'List[int]'" in (
        error.message
    ), error.format()
    assert "`tuple([1, 2, 1])`" in error.message
    assert error.line == _line_of(path, "q = replace(p, lanes=")


# ---------------------------------------------------------------------------
# replace() of a class of another module, of a nested dataclass (#623)
# ---------------------------------------------------------------------------

_LANES_CONFIG = """
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Lanes:
    ids: tuple[int, ...] = (460, 265)


@dataclass
class CaseConfig:
    name: str = "case"
    goal_lanelet_ids: list[int] = field(default_factory=lambda: [460, 265])
    timeout_seconds: float = 10.0
    min_speed_kmh: float | None = None
    lanes: Lanes = field(default_factory=Lanes)
"""


def test_a_wrong_replace_of_a_class_of_another_module_is_refused_at_the_call(
    tmp_path: Path,
) -> None:
    # The method `replace` is lowered to is in configs.py, the call in
    # scenario.py: the error is at the call, not at the class.
    scenario, config, path = _write_case(
        tmp_path,
        "lanes = replace(cfg.lanes, ids=(1, 2, 3))\n"
        'logger.info("%d", len(lanes.ids))\n',
        config=_LANES_CONFIG,
        header=_FROZEN_HEADER,
    )
    error = _only_error(typecheck_scenario(scenario, config))
    assert "'Tuple[int,int,int]' does not match expected type 'List[int]'" in (
        error.message
    ), error.format()
    assert error.path == str(path), error.format()
    assert error.line == _line_of(path, "lanes = replace(cfg.lanes")


_NESTED_DATACLASSES = """


class Plans:
    @dataclass(frozen=True)
    class Leg:
        lanelet: int = 0
        speed: float = 1.0

    def faster(self, leg: Plans.Leg) -> Plans.Leg:
        return replace(leg, speed=leg.speed * 2)


def segments() -> int:
    @dataclass(frozen=True)
    class Seg:
        a: int = 0

        @dataclass(order=True)
        class Part:
            b: int = 0

    kept = {replace(Seg(1), a=2), Seg(2)}
    lower = Seg.Part(1) < replace(Seg.Part(1), b=3)
    return len(kept) + replace(Seg(1), a=2).a + int(lower)
"""


def test_replace_and_hash_of_a_nested_dataclass_compile(tmp_path: Path) -> None:
    # A dataclass in a class or in a function gets its `replace` methods and
    # its ==, hash and ordering in its body, as `@extend` cannot name it.
    scenario, config, _ = _write_case(
        tmp_path,
        "leg = Plans().faster(Plans.Leg(3, 2.0))\n"
        "legs = {leg, Plans.Leg(3, 4.0)}\n"
        'logger.info("%f %d %d", leg.speed, len(legs), segments())\n',
        extra=_NESTED_DATACLASSES,
        header=_FROZEN_HEADER,
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_a_wrong_replace_of_a_nested_dataclass_is_refused_at_the_call(
    tmp_path: Path,
) -> None:
    scenario, config, path = _write_case(
        tmp_path,
        'leg = replace(Plans.Leg(), speed="fast")\n' 'logger.info("%f", leg.speed)\n',
        extra=_NESTED_DATACLASSES,
        header=_FROZEN_HEADER,
    )
    error = _only_error(typecheck_scenario(scenario, config))
    assert "'str' does not match expected type 'float'" in error.message, error.format()
    assert error.line == _line_of(path, "leg = replace(Plans.Leg()")


# ---------------------------------------------------------------------------
# Python 3.12+ syntax (#624): kept in strings, as the suite runs on 3.10
# ---------------------------------------------------------------------------

_PY312 = pytest.mark.skipif(
    sys.version_info < (3, 12), reason="Python 3.12+ syntax (PEP 695, PEP 701)"
)

_PY312_DEFINITIONS = """


type LaneIds = list[int]


def total(ids: LaneIds) -> int:
    return sum(ids)


def first[T](xs: list[T]) -> T:
    return xs[0]


class Base:
    def v(self) -> int:
        return 1


def pick[B: Base](a: B, b: B) -> B:
    return a if a.v() >= b.v() else b


class Box[T]:
    item: T

    def __init__(self, item: T) -> None:
        self.item = item

    def get(self) -> T:
        return self.item


@dataclass(frozen=True)
class Pair[K, V]:
    k: K
    v: V
"""


@_PY312
def test_python_3_12_syntax_compiles(tmp_path: Path) -> None:
    # Same-quote f-string fields are requoted, `type X = T` is an alias, and
    # type parameters are Codon's generics; a generic function or class is
    # compiled where it is used, with a warning each.
    scenario, config, path = _write_case(
        tmp_path,
        'd = {"k": 1}\n'
        'p = replace(Pair(1, "a"), v="b")\n'
        'logger.info(f"{cfg.name.replace("_", "-")} {d["k"]} '
        '{total(cfg.goal_lanelet_ids)} {first(cfg.goal_lanelet_ids)}")\n'
        'logger.info(f"{pick(Base(), Base()).v()} {Box(1).get()} {Box("x").get()} '
        '{p.v} {hash(p) == hash(Pair(1, "b"))}")\n',
        extra=_PY312_DEFINITIONS,
        header=_FROZEN_HEADER,
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()
    lines = {w.line for w in result.warnings}
    assert lines == {
        _line_of(path, "def first["),
        _line_of(path, "def pick["),
        _line_of(path, "class Box["),
        _line_of(path, "class Pair["),
    }, result.format()


@_PY312
def test_a_generic_class_is_checked_where_it_is_used(tmp_path: Path) -> None:
    scenario, config, path = _write_case(
        tmp_path,
        'n = Box("a").get() + 1\nlogger.info("%d", n)\n',
        extra=_PY312_DEFINITIONS,
        header=_FROZEN_HEADER,
    )
    error = _only_error(typecheck_scenario(scenario, config))
    assert "unsupported operand type(s) for +: 'str'" in error.message, error.format()
    assert error.line == _line_of(path, 'n = Box("a")')


@_PY312
@pytest.mark.parametrize(
    ("code", "needle", "hint"),
    [
        (
            'd = {"it\'s": 1}\nlogger.info(f"{d["it\'s"]}")\n',
            "logger.info(f",
            "cannot quote it with the other quote",
        ),
        (
            'd = {"a\\n": 1}\nlogger.info(f"{d["a\\n"]}")\n',
            "logger.info(f",
            "holds a backslash",
        ),
        (
            'x: Pair2[int] = (1, 2)\nlogger.info("%d", x[0])\n',
            "type Pair2[",
            "no generic alias",
        ),
    ],
)
def test_python_3_12_syntax_codon_cannot_parse_is_refused_at_its_line(
    tmp_path: Path, code: str, needle: str, hint: str
) -> None:
    scenario, config, path = _write_case(
        tmp_path,
        code,
        extra="\n\ntype Pair2[T] = tuple[T, T]\n",
        header=_FROZEN_HEADER,
    )
    errors = typecheck_scenario(scenario, config).diagnostics
    found = [e for e in errors if hint in e.message]
    assert found, "\n".join(e.format() for e in errors)
    assert found[0].line == _line_of(path, needle)


# ---------------------------------------------------------------------------
# A conditional expression narrowing an attribute chain (#621)
# ---------------------------------------------------------------------------

_NARROWED_NPC = r"""


class Npc:
    speed: float
    name: str

    def __init__(self, speed: float, name: str) -> None:
        self.speed = speed
        self.name = name

    def label(self) -> str:
        return self.name.upper()


class Holder:
    npc: Optional[Npc]

    def __init__(self) -> None:
        self.npc = None


def npc_name(npc: Optional[Npc]) -> str | None:
    return npc.name if npc is not None else None


def npc_label(h: Holder) -> Optional[str]:
    return None if h.npc is None else h.npc.label()


def lane_number(name: str) -> str | None:
    m = re.match(r"lane_(\d+)", name)
    return m.group(1) if m else None


def describe(h: Holder) -> str:
    speed = h.npc.speed if h.npc is not None else None
    rounded = round(speed) if speed is not None else None
    doubled = 0.0
    if speed is not None:
        doubled = speed * 2.0 + 1.0
    text = f"{speed:.1f}" if speed is not None else "n/a"
    shown = str(speed) if speed is not None else "n/a"
    count = rounded + 1 if rounded is not None else 0
    label = h.npc.label() if h.npc else None
    upper = label.upper() if label is not None else "?"
    return text + shown + upper + str(doubled) + str(count)


def npc_speed(npc: Optional[Npc]) -> str | None:
    return npc.speed if npc is not None else None
"""


def test_a_narrowed_attribute_chain_is_an_optional_of_its_type(
    tmp_path: Path,
) -> None:
    # The value of `npc.speed if npc is not None else None` was an unresolved
    # `Optional[T]`: `str()` of it, a format spec or a `round()` of it were
    # refused, or crashed Codon (exit -11).  Only `npc_speed()`, which returns
    # the float as a str, is wrong.
    scenario, config, path = _write_case(
        tmp_path,
        "h = Holder()\n"
        'h.npc = Npc(3.0, "a")\n'
        'logger.info("%s", describe(h))\n'
        "name = npc_name(h.npc)\n"
        'logger.info("%s", name.upper() if name is not None else "-")\n'
        'logger.info("%s", npc_label(h))\n'
        'number = lane_number("lane_7")\n'
        'logger.info("%s", f"{number}" if number else "?")\n'
        'logger.info("%s", npc_speed(h.npc))\n',
        _NARROWED_NPC,
        header=_stdlib_header("import re\nfrom typing import Optional\n"),
    )
    result = typecheck_scenario(scenario, config)
    error = _only_error(result)
    assert len(result.diagnostics) == 1, result.format()
    assert (
        "npc_speed() returns float | None, but the function is declared -> str | None"
        in error.message
    ), error.format()
    assert error.line == _line_of(path, "return npc.speed if npc is not None")


def test_a_narrowed_branch_is_compiled_only_where_its_value_can_be_something(
    tmp_path: Path,
) -> None:
    # The branch of a conditional expression narrowed at module level, in a
    # comprehension, or of a parameter that is always None (never compiled,
    # as Codon compiles neither branch of a test it decides); a wrong one is
    # reported at its line.
    extra = r"""


class Npc:
    speed: float

    def __init__(self, speed: float) -> None:
        self.speed = speed


DEFAULT: Optional[Npc] = Npc(1.0)
DEFAULT_SPEED = DEFAULT.speed if DEFAULT is not None else None


class Ref:
    name: Optional[str]

    def __init__(self, name=None) -> None:
        self.name = str(name) if name is not None else None


def table(env=None):
    return None if env is None else dict(env)


def speeds(npcs: list[Optional[Npc]]) -> list[Optional[float]]:
    return [n.speed if n else None for n in npcs]


def broken(npc: Optional[Npc]) -> Optional[float]:
    return npc.nope if npc is not None else None
"""
    scenario, config, path = _write_case(
        tmp_path,
        'logger.info("%s %s %s", DEFAULT_SPEED, Ref().name, speeds([Npc(1.0), None]))\n'
        'logger.info("%s %s", table(), broken(None))\n',
        extra,
        header=_stdlib_header("from typing import Optional\n"),
    )
    result = typecheck_scenario(scenario, config)
    error = _only_error(result)
    assert len(result.diagnostics) == 1, result.format()
    assert "'Npc' object has no attribute 'nope'" in error.message, error.format()
    assert error.line == _line_of(path, "return npc.nope if")
    assert not any(".codon" in frame for frame in error.trace), error.format()


# ---------------------------------------------------------------------------
# A positional argument for a keyword-only parameter (#622)
# ---------------------------------------------------------------------------

_KEYWORD_ONLY = r"""


def gap(a: float, *, scale: float = 1.0) -> float:
    return a * scale


def total(*args: float, k: float = 0.0) -> float:
    return sum(args) + k


class Box:
    v: float

    def __init__(self, v: float, *, w: float = 0.0) -> None:
        self.v = v + w

    def m(self, a: float, *, s: float = 1.0, t: float = 2.0) -> float:
        return a * s * t
"""


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            "gap(1.0, 2.0)",
            "gap() takes 1 positional argument but more were given: scale is "
            "keyword-only (pass it by name: `scale=...`)",
        ),
        (
            "b.m(1.0, 2.0)",
            "Box.m() takes 2 positional arguments but more were given: s and t "
            "are keyword-only (pass them by name: `s=...`)",
        ),
        (
            "Box(1.0, 2.0).v",
            "Box.__init__() takes 2 positional arguments but more were given: w "
            "is keyword-only (pass it by name: `w=...`)",
        ),
    ],
)
def test_a_positional_argument_for_a_keyword_only_parameter_is_refused_at_the_call(
    tmp_path: Path, call: str, message: str
) -> None:
    # The `*_acs_kw` Codon needs in place of the bare `*` took it silently;
    # what passes keywords by name, `*args`, and `functools.partial` with a
    # keyword-only argument still compile.
    scenario, config, path = _write_case(
        tmp_path,
        'logger.info("%s", gap(1.0, scale=2.0))\n'
        'logger.info("%s", total(1.0, 2.0, k=1.0))\n'
        "p = functools.partial(gap, scale=3.0)\n"
        'logger.info("%s", p(1.0))\n'
        "b = Box(1.0, w=2.0)\n"
        'logger.info("%s", b.m(1.0, s=2.0) + Box(v=1.0).v)\n'
        f'logger.info("%s", {call})\n',
        _KEYWORD_ONLY,
        header=_stdlib_header("import functools\n"),
    )
    result = typecheck_scenario(scenario, config)
    error = _only_error(result)
    assert len(result.diagnostics) == 1, result.format()
    assert error.message == message, error.format()
    assert error.line == _line_of(path, call)


# ---------------------------------------------------------------------------
# sum() of strings, dict.fromkeys(xs), list[tuple[X, ...]] (#625)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("call", ['sum(["a", "b"], "")', 'sum(["a"], start="")'])
def test_sum_with_a_str_start_is_refused_as_python_refuses_it(
    tmp_path: Path, call: str
) -> None:
    # CPython raises TypeError before it adds anything; Codon concatenates.
    scenario, config, path = _write_case(tmp_path, f'logger.info("%s", {call})\n')
    result = typecheck_scenario(scenario, config)
    error = _only_error(result)
    assert "sum() can't sum strings [use ''.join(seq) instead]" in error.message
    assert error.line == _line_of(path, call)


def test_dict_fromkeys_of_one_argument_and_a_list_of_variable_length_tuples_pass(
    tmp_path: Path,
) -> None:
    scenario, config, _ = _write_case(
        tmp_path,
        """
        d = dict.fromkeys(["a", "b", "a"])
        logger.info("%s %s", list(d), d["a"] is None)
        e = dict.fromkeys(["a"], 0)
        logger.info("%s", e["a"] + 1)
        pairs: list[tuple[int, ...]] = []
        pairs.append((1, 2))
        pairs.append((3,))
        pairs.insert(0, ())
        for p in pairs:
            logger.info("%d", len(p))
        """,
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_variable_length_tuples_of_one_length_pass_as_before(
    tmp_path: Path,
) -> None:
    # Given tuples of one length, the annotation is dropped as before (#625
    # review): the list of fixed-length tuples Codon types takes a tuple held
    # in a name, `extend` and `in`, and so do a dict, a set and an Optional.
    scenario, config, _ = _write_case(
        tmp_path,
        """
        pairs: list[tuple[int, ...]] = []
        t = (3, 4)
        pairs.append(t)
        pairs.extend([(5, 6)])
        logger.info("%s", (3, 4) in pairs)
        d: dict[str, tuple[int, ...]] = {}
        d["a"] = (1, 2)
        s: set[tuple[int, ...]] = set()
        s.add((1, 2))
        o: list[tuple[int, ...] | None] = []
        o.append((1, 2))
        logger.info("%d %d %d %d", len(pairs), len(d), len(s), len(o))
        """,
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_a_list_of_variable_length_tuples_still_checks_its_elements(
    tmp_path: Path,
) -> None:
    # The annotation is kept, not dropped: a tuple of str is no tuple[int, ...].
    scenario, config, path = _write_case(
        tmp_path,
        """
        pairs: list[tuple[int, ...]] = []
        pairs.append((1, 2))
        pairs.append(("a",))
        """,
    )
    result = typecheck_scenario(scenario, config)
    error = _only_error(result)
    assert error.line == _line_of(path, 'pairs.append(("a",))'), result.format()


# ---------------------------------------------------------------------------
# A mixin listed before the framework base (#626)
# ---------------------------------------------------------------------------


def _mixin_first_header(init_call: str) -> str:
    return _HEADER.replace(
        "class CaseScenario(BaseScenario):",
        "class VehicleInfoMixin:\n"
        "    ego_config: EgoConfig\n\n"
        "    def vehicle_label(self) -> str:\n"
        '        return "ego:" + self.ego_config.vehicle_type\n\n\n'
        "class CaseScenario(VehicleInfoMixin, BaseScenario):",
        1,
    ).replace(
        "super().__init__(ego_config, spawn_pose=spawn_pose, "
        "ground_projection=ground_projection)",
        init_call,
        1,
    )


def test_super_init_passes_over_a_mixin_listed_before_the_framework_base(
    tmp_path: Path,
) -> None:
    # Codon's super() is the first base's; Python's follows the MRO past the
    # mixin, which defines no __init__, to BaseScenario.__init__.
    scenario, config, _ = _write_case(
        tmp_path,
        _VALID_SETUP + '        logger.info("%s", self.vehicle_label())\n',
        header=_mixin_first_header(
            "super().__init__(ego_config, spawn_pose=spawn_pose, "
            "ground_projection=ground_projection)"
        ),
    )
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


def test_super_init_past_a_mixin_still_checks_the_base_arguments(
    tmp_path: Path,
) -> None:
    scenario, config, path = _write_case(
        tmp_path,
        _VALID_SETUP,
        header=_mixin_first_header(
            "super().__init__(ego_config, spawn_pose=spawn_pose, no_such=1)"
        ),
    )
    result = typecheck_scenario(scenario, config)
    error = _only_error(result)
    assert "no_such" in error.message, result.format()
    assert error.line == _line_of(path, "no_such=1")


# ---------------------------------------------------------------------------
# Keyword-only overrides, kw-only lambdas and None-first locals (#629)
# ---------------------------------------------------------------------------

_OVERRIDES = r"""


class Base:
    def m(self, a: float, *, s: float = 1.0) -> float:
        return a * s


class Sub(Base):
    def m(self, a: float, *, s: float = 2.0) -> float:
        return a + s


def use(b: Base) -> float:
    return b.m(1.0)


def key_of(x: int, *, rev: bool = False) -> int:
    return -x if rev else x


def dbl(x: int) -> int:
    return 2 * x


def counter(v: Optional[int]) -> Optional[int]:
    y = v
    out: Optional[int] = None

    def bump() -> None:
        nonlocal y, out
        out = y + 1 if y is not None else None
        y = 3

    bump()
    return out


def deltas(xs: list[float]) -> list[Optional[float]]:
    out: list[Optional[float]] = []
    prev = None
    for v in xs:
        d = v - prev if prev is not None else None
        out.append(d)
        prev = v
    return out


def none_only() -> Optional[int]:
    z = None
    return z + 1 if z is not None else None
"""


@pytest.mark.parametrize(
    ("setup", "needle", "hint"),
    [
        # A call of an overridden method with a keyword-only parameter: through
        # the base, then by keyword.
        ('logger.info("%s", use(Sub()))\n', "return b.m(1.0)", "dispatch thunk"),
        (
            'logger.info("%s", Sub().m(1.0, s=3.0))\n',
            "Sub().m(1.0, s=3.0)",
            "dispatch thunk",
        ),
        (
            "fs: list[Callable[[int], int]] = [key_of, dbl]\n"
            'logger.info("%s", [f(1) for f in fs])\n',
            "fs: list",
            "put a lambda that calls it in the table instead (`lambda x: key_of(x)`)",
        ),
        (
            'logger.info("%s", counter(1))\n',
            "out = y + 1",
            "copy it into an annotated local",
        ),
        (
            'logger.info("%s", deltas([1.0, 2.0]))\n',
            "prev = None",
            "`prev: float | None = None`",
        ),
        (
            'logger.info("%s", none_only())\n',
            "return z + 1",
            "the local `z` is only ever None",
        ),
    ],
)
def test_keyword_only_overrides_and_none_first_locals_say_why(
    tmp_path: Path, setup: str, needle: str, hint: str
) -> None:
    scenario, config, path = _write_case(
        tmp_path,
        setup,
        _OVERRIDES,
        header=_stdlib_header("from typing import Callable, Optional\n"),
    )
    result = typecheck_scenario(scenario, config)
    assert not result.ok, result.format()
    hinted = [d for d in result.diagnostics if hint in d.message]
    assert hinted, result.format()
    assert hinted[0].line == _line_of(path, needle), result.format()


def test_a_lambda_with_keyword_only_parameters_is_refused_with_a_hint(
    tmp_path: Path,
) -> None:
    scenario, config, path = _write_case(
        tmp_path,
        "f = lambda a, *, k=1: a * k\n" 'logger.info("%s", f(2, k=3))\n',
    )
    result = typecheck_scenario(scenario, config)
    error = _only_error(result)
    assert "write a nested def instead" in error.message, error.format()
    assert error.line == _line_of(path, "f = lambda")
