"""A production pose reaches the snap as the Lanelet2 pose it was written as.

``snap_to_carla_road`` places a :class:`Lanelet2Pose` on the centreline of the
lanelet it names, and an :class:`OpenDrivePose` through CARLA's own XODR
projection. Those are different answers -- that is the whole point of the
former -- so which one a scenario gets is decided by *what it hands over*, not
by anything inside the snap.

Every production flow used to convert its ``Lanelet2Pose`` to an
``OpenDrivePose`` first and snap that, so the lanelet path was reachable only
from tests: ego spawns, Autoware goals and NPC spawns all kept the round-trip
placement error the lanelet path exists to remove. Unit tests of the snap
cannot see that, because the bug is in the wiring and not in the snap, which is
why these tests watch the wiring instead.

The ``OpenDrivePose`` is still derived where its lane metadata is genuinely
wanted -- the lane-change target lane, a route condition's road id -- and for
the ego it is read off the snapped position, so "derives one" is not the thing
under test here. "Snaps one" is.
"""

from __future__ import annotations

import ast
import pathlib
from typing import Any, List
from unittest.mock import MagicMock

import carla
import pytest

from autoware_carla_scenario import BaseScenario, EgoConfig, SpawnTransform
from autoware_carla_scenario.coordinate import (
    CarlaWorldPose,
    Lanelet2Pose,
    OpenDrivePose,
)

#: Whatever the snap returns; the tests care about its argument, not this.
_SNAPPED = CarlaWorldPose(x=11.0, y=22.0, z=3.0, yaw=45.0)


class _Scenario(BaseScenario):
    """The smallest thing that can run ``_setup_ego_spawn``."""

    def setup(self) -> None:  # pragma: no cover - not exercised
        pass

    def is_done(self) -> bool:  # pragma: no cover - not exercised
        return False


def _ego_config() -> EgoConfig:
    return EgoConfig(
        spawn_location=SpawnTransform(
            carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0))
        ),
        vehicle_type="vehicle.mini.cooper",
    )


@pytest.fixture
def _snapped_poses(monkeypatch: pytest.MonkeyPatch) -> List[Any]:
    """Record every pose handed to the snap, from every module that calls it."""
    seen: List[Any] = []

    def _snap(pose, _world, **_kwargs):
        seen.append(pose)
        return _SNAPPED

    for target in (
        "autoware_carla_scenario.scenario_base.snap_to_carla_road",
        "autoware_carla_scenario.coordinate.snap_to_carla_road",
    ):
        monkeypatch.setattr(target, _snap)

    # The real ``to_map_frame`` wants the projector offsets a loaded map
    # supplies; what the goal is expressed in afterwards is not this file's
    # subject, so it is stubbed rather than set up.
    monkeypatch.setattr(
        "autoware_carla_scenario.entity.autoware_entity.to_map_frame",
        lambda pose: pose,
    )
    return seen


class TestTheEgoSpawn:
    """``_setup_ego_spawn`` is how every packaged scenario places its ego."""

    def _run(self) -> tuple[OpenDrivePose, _Scenario]:
        scenario = _Scenario(_ego_config())
        scenario.set_client(MagicMock())
        scenario._spawn_pose = Lanelet2Pose(lanelet_id=179280, s=8.0)
        return scenario._setup_ego_spawn(), scenario

    def test_the_spawn_is_snapped_as_a_lanelet_pose(
        self, _snapped_poses: List[Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "autoware_carla_scenario.scenario_base.to_opendrive",
            lambda pose: OpenDrivePose(road_id="7", lane_id=-1, s=8.0, t=0.0),
        )

        self._run()

        assert [type(p) for p in _snapped_poses] == [Lanelet2Pose]

    def test_the_opendrive_pose_is_still_derived_for_its_lane_metadata(
        self, _snapped_poses: List[Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Snapping the lanelet must not cost the callers that need the lane."""
        od = OpenDrivePose(road_id="7", lane_id=-1, s=8.0, t=0.0)
        monkeypatch.setattr(
            "autoware_carla_scenario.scenario_base.to_opendrive", lambda pose: od
        )

        returned, scenario = self._run()

        assert returned == od
        assert scenario.ego_config.od_pose == od

    def test_the_lane_metadata_is_read_off_where_the_ego_was_placed(
        self, _snapped_poses: List[Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Converted from the Lanelet2 pose instead, the lane could be the one
        across the reference line, while EntityLanePositionCondition reads the
        lane off the actor's CARLA position -- and a lane-change target derived
        from it could then never match."""
        converted: List[Any] = []

        def _to_opendrive(pose):
            converted.append(pose)
            return OpenDrivePose(road_id="7", lane_id=-1, s=8.0, t=0.0)

        monkeypatch.setattr(
            "autoware_carla_scenario.scenario_base.to_opendrive", _to_opendrive
        )

        self._run()

        assert converted == [_SNAPPED]


class TestTheAutowareGoal:
    """A goal Autoware routes to is a Lanelet2 pose on Autoware's own map."""

    def test_the_goal_is_snapped_as_a_lanelet_pose(
        self, _snapped_poses: List[Any]
    ) -> None:
        from autoware_carla_scenario.autoware_bridge import FakeAutowareBridge
        from autoware_carla_scenario.entity import AutowareEgoEntity

        entity = AutowareEgoEntity(bridge=FakeAutowareBridge())

        entity.route_to(
            MagicMock(),
            Lanelet2Pose(lanelet_id=176640, s=2.0),
            initial_pose=CarlaWorldPose(x=1.0, y=2.0, z=0.5, yaw=0.0),
        )

        assert [type(p) for p in _snapped_poses] == [Lanelet2Pose]


class TestEveryCallSiteInTheShippedPackage:
    """The two flows above are drivable here; the other call sites are not.

    NPC spawns come out of ``DeclarativeScenario._build_npc`` and the packaged
    examples, which need a compiled document or a live world to reach. They are
    the same defect and regress the same way, so they are pinned by reading the
    source: a shipped call to ``snap_to_carla_road`` must not be handed a pose
    that came out of ``to_opendrive``.
    """

    @staticmethod
    def _offenders() -> List[str]:
        src = pathlib.Path(__file__).resolve().parents[2] / "src"
        offenders: List[str] = []

        for path in sorted(src.rglob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))

            # Names bound from a to_opendrive(...) call, per enclosing scope.
            for scope in ast.walk(tree):
                if not isinstance(
                    scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)
                ):
                    continue
                from_opendrive = {
                    target.id
                    for node in ast.walk(scope)
                    if isinstance(node, ast.Assign)
                    and isinstance(node.value, ast.Call)
                    and _called_name(node.value) == "to_opendrive"
                    for target in node.targets
                    if isinstance(target, ast.Name)
                }
                for node in ast.walk(scope):
                    if not isinstance(node, ast.Call):
                        continue
                    if _called_name(node) != "snap_to_carla_road" or not node.args:
                        continue
                    first = node.args[0]
                    bad = (
                        isinstance(first, ast.Call)
                        and _called_name(first) == "to_opendrive"
                    ) or (isinstance(first, ast.Name) and first.id in from_opendrive)
                    if bad:
                        offenders.append(
                            f"{path.relative_to(src)}:{node.lineno} "
                            f"snaps {ast.unparse(first)}"
                        )
        return sorted(set(offenders))

    def test_no_shipped_call_snaps_a_pose_that_came_from_to_opendrive(self) -> None:
        offenders = self._offenders()

        assert not offenders, (
            "these calls convert a Lanelet2 pose to OpenDRIVE and snap that, "
            "which puts the round-trip placement error back: " + "; ".join(offenders)
        )


def _called_name(call: ast.Call) -> str:
    """The bare name of what *call* calls, for both ``f()`` and ``mod.f()``."""
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


class TestTheSpawnRetry:
    """An occupied spawn is retried around where the first attempt was made."""

    def test_the_retries_offset_the_resolved_transform_not_a_re_snap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Re-snapping the OpenDRIVE pose would move the retries back to the
        OpenDRIVE lane -- on a map where it disagrees with the Lanelet2 one,
        possibly the opposing lane -- before offsetting them."""
        from autoware_carla_scenario.entity._spawn import spawn_vehicle_actor

        def _no_snap(*_args, **_kwargs):
            raise AssertionError("the retry must not re-snap od_pose")

        monkeypatch.setattr(
            "autoware_carla_scenario.coordinate.snap.snap_to_carla_road", _no_snap
        )
        tried: List[Any] = []
        actor = MagicMock()

        def _try_spawn(_bp, transform):
            tried.append(transform)
            return actor if len(tried) > 1 else None  # the first spot is taken

        world = MagicMock()
        blueprint = MagicMock(id="vehicle.mini.cooper")
        world.get_blueprint_library.return_value.filter.return_value = [blueprint]
        world.try_spawn_actor.side_effect = _try_spawn
        placed = carla.Transform(
            carla.Location(x=100.0, y=200.0, z=1.0), carla.Rotation(yaw=0.0)
        )

        spawned = spawn_vehicle_actor(
            world,
            "vehicle.mini.cooper",
            "npc",
            SpawnTransform(placed),
            od_pose=OpenDrivePose(road_id="7", lane_id=1, s=8.0, t=0.0),
            spawn_retry_max_count=1,
            spawn_retry_t_step=0.5,
            spawn_retry_z_step=0.0,
        )

        assert spawned is actor
        retry = tried[1].location
        # Half a metre across the placed transform's heading, nowhere else.
        assert (retry.x, retry.y) == pytest.approx((100.0, 200.5))
