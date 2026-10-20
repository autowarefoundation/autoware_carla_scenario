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
    assert not [d for d in result.diagnostics if "None" in d.message], result.format()


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
        (_DATACLASSES, "ByIdentity(1) == ByIdentity(1)", "a is b"),
    ],
    ids=["unordered", "unhashable", "eq-false"],
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
        assert not (Plain.A == 1) and Plain.A != 1
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
}


@pytest.mark.parametrize("name", sorted(_TYPING_CASES))
def test_a_supported_typing_name_compiles_in_a_scenario(
    tmp_path: Path, name: str
) -> None:
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

    assert documented <= set(dir(typing))
    # Every name the shim adds beyond Codon is compiled by a test above.
    assert set(_TYPING_CASES) <= documented


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
