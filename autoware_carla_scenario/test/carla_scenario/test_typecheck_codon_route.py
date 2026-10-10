"""The static check for a logical scenario's API: route poses and progress.

The same check as :mod:`.test_typecheck_codon`, on a scenario written with the
route poses, ``RouteProgressCondition`` and ``appear_on_start``.  Skipped
where no Codon compiler is installed.
"""

from __future__ import annotations

import importlib
import sys
import textwrap
from pathlib import Path

import pytest

from autoware_carla_scenario.typecheck import available_toolchain, typecheck_scenario

from .test_typecheck_codon import _CONFIG, _HEADER, _line_of, _only_error

pytestmark = pytest.mark.skipif(
    available_toolchain() is None, reason="no Codon compiler"
)

_ROUTE_HEADER = _HEADER.replace(
    "    RelativeLanePose,\n",
    "    RelativeLanePose,\n"
    "    RouteCrossingPose,\n"
    "    RouteCrosswalkPose,\n"
    "    RouteLanePose,\n"
    "    RouteOppositePose,\n"
    "    RouteProgressCondition,\n"
    "    RouteRoadsidePose,\n",
)
_counter = 0


def _write_case(tmp_path: Path, setup: str) -> tuple[type, type, Path]:
    global _counter
    _counter += 1
    package = f"acs_typecheck_route_case_{_counter}"
    root = tmp_path / package
    root.mkdir()
    (root / "__init__.py").write_text("")
    (root / "configs.py").write_text(_CONFIG)
    scenario = root / "scenario.py"
    scenario.write_text(
        _ROUTE_HEADER + textwrap.indent(textwrap.dedent(setup), " " * 8)
    )
    sys.path.insert(0, str(tmp_path))
    try:
        module = importlib.import_module(f"{package}.scenario")
        configs = importlib.import_module(f"{package}.configs")
    finally:
        sys.path.remove(str(tmp_path))
    return module.CaseScenario, configs.CaseConfig, scenario


def test_a_logical_scenario_compiles(tmp_path: Path) -> None:
    setup = """
        self._setup_ego_spawn()
        path = Trajectory(
            "route_path",
            [
                TrajectoryVertex(RouteLanePose(10.0, d_lane=1)),
                TrajectoryVertex(
                    RouteLanePose(ds=-5, offset=0.5, yaw=0.1, anchor="junction:0:entry"),
                    RouteProgressCondition(value=-10.0, anchor="junction:0:entry", label="near"),
                ),
                TrajectoryVertex(RouteOppositePose(30.0, lane=2)),
                TrajectoryVertex(RouteCrossingPose(0, "left", -20.0, turn="straight")),
                TrajectoryVertex(RouteCrosswalkPose(junction=0, leg="exit", side="right", along=-1.5)),
                TrajectoryVertex(RouteRoadsidePose(5.0, side="left", kerb_distance=1.0, anchor="start")),
            ],
        )
        self.register_pre_tick(
            FollowTrajectoryAction(
                "npc1",
                path,
                following_mode=TrajectoryFollowingMode.FOLLOW,
                condition=RouteProgressCondition(
                    EGO_ROLE_NAME, 20.0, ComparisonRule.GREATER_THAN, label="go"
                ),
                label="npc1_route",
                speed=8.0,
                appear_on_start=True,
            )
        )
        arrived = RouteProgressCondition(value=0.0, anchor="end", label="arrived")
        self.register_pass_condition(arrived)
        progress = arrived.progress
        if progress is not None:
            logger.info("at %f", progress)
        """
    scenario, config, _ = _write_case(tmp_path, setup)
    result = typecheck_scenario(scenario, config)
    assert result.ok, result.format()


@pytest.mark.parametrize(
    ("setup", "needle", "message"),
    [
        (
            "self.register_pass_condition(RouteProgressCondition(value=1.0))\n",
            "RouteProgressCondition(value=1.0)",
            "RouteProgressCondition(): label is a required keyword argument",
        ),
        (
            'self.register_pass_condition(RouteProgressCondition(value="near", label="x"))\n',
            'value="near"',
            "expected a float",
        ),
        (
            "TrajectoryVertex(RouteLanePose(10.0, anchor=3))\n",
            "anchor=3",
            "expected a str",
        ),
        (
            'FollowTrajectoryAction("npc1", Trajectory("t", [TrajectoryVertex(RouteLanePose()), '
            'TrajectoryVertex(RouteLanePose(5.0))]), appear_on_start="yes")\n',
            'FollowTrajectoryAction("npc1"',
            "no function 'FollowTrajectoryAction.__init__'",
        ),
    ],
    ids=["missing-label", "str-for-float", "int-anchor", "str-appear-on-start"],
)
def test_a_wrong_logical_scenario_is_refused_at_its_line(
    tmp_path: Path, setup: str, needle: str, message: str
) -> None:
    scenario, config, path = _write_case(tmp_path, setup)
    error = _only_error(typecheck_scenario(scenario, config))
    assert message in error.message, error.format()
    assert error.path == str(path)
    assert error.line == _line_of(path, needle)
