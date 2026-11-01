"""A pedestrian moved by a motion generator (SmartPedestrianControllerAction).

The generator is a fake served in-process over the real contract: a body
standing upright that walks along +x of the contract frame at 1 m/s, from
where its session opened.  CARLA is faked as in test_play_motion.
"""

from __future__ import annotations

import json
import math
from concurrent import futures
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator
from unittest.mock import MagicMock

import grpc
import pytest
import typesafe_carla.carla as carla

from autoware_carla_scenario import (
    PedestrianEntity,
    PedestrianEntityConfig,
    SmartPedestrianControllerAction,
    SpawnTransform,
    TimedPrompt,
    WalkWaypoint,
)
from autoware_carla_scenario.entity.pedestrian_entity import walker_control_speed
from autoware_carla_scenario.entity.registry import register_entity, unregister_entity
from autoware_carla_scenario.pedestrian._proto import pedestrian_motion_pb2 as pb
from autoware_carla_scenario.pedestrian._proto import (
    pedestrian_motion_pb2_grpc as pb_grpc,
)
from autoware_carla_scenario.pedestrian.motion_client import (
    CONTRACT_VERSION,
    PedestrianMotionClient,
    to_carla,
    to_contract,
)
from autoware_carla_scenario.pedestrian.poser import WalkerPoser
from autoware_carla_scenario.pedestrian.skeleton import directions_from_joints
from autoware_carla_scenario.prepare import Resources
from autoware_carla_scenario.prepare import pool as pool_module
from autoware_carla_scenario.prepare.pool import BackendPool

from .test_play_motion import _bones

FPS = 30.0
HORIZON = 30
DT = 0.05

#: An upright body, relative to its root: x forward, y left, z up.
_BODY: dict[str, tuple[float, float, float]] = {
    "Hips": (0.0, 0.0, 1.0),
    "Spine1": (0.0, 0.0, 1.1),
    "Spine2": (0.0, 0.0, 1.2),
    "Chest": (0.0, 0.0, 1.3),
    "Neck1": (0.0, 0.0, 1.5),
    "Neck2": (0.0, 0.0, 1.55),
    "Head": (0.0, 0.0, 1.6),
    "LeftShoulder": (0.0, 0.05, 1.45),
    "LeftArm": (0.0, 0.2, 1.45),
    "LeftForeArm": (0.0, 0.2, 1.15),
    "LeftHand": (0.0, 0.2, 0.9),
    "LeftHandMiddleEnd": (0.0, 0.2, 0.8),
    "RightShoulder": (0.0, -0.05, 1.45),
    "RightArm": (0.0, -0.2, 1.45),
    "RightForeArm": (0.0, -0.2, 1.15),
    "RightHand": (0.0, -0.2, 0.9),
    "RightHandMiddleEnd": (0.0, -0.2, 0.8),
    "LeftLeg": (0.0, 0.1, 0.95),
    "LeftShin": (0.0, 0.1, 0.5),
    "LeftFoot": (0.0, 0.1, 0.1),
    "LeftToeBase": (0.15, 0.1, 0.0),
    "RightLeg": (0.0, -0.1, 0.95),
    "RightShin": (0.0, -0.1, 0.5),
    "RightFoot": (0.0, -0.1, 0.1),
    "RightToeBase": (0.15, -0.1, 0.0),
}
_NAMES = tuple(_BODY)


class _Generator(pb_grpc.PedestrianMotionServicer):
    """Walks straight on along the way the session opened facing."""

    def __init__(self) -> None:
        self.sessions: dict[str, pb.OpenSessionRequest] = {}
        self.steps: list[int] = []
        self.closed: list[str] = []

    def Negotiate(self, request: Any, context: Any) -> Any:
        return pb.NegotiateResponse(
            contract_version=CONTRACT_VERSION,
            backend="fake",
            backend_version="0.0.0",
            model="straight",
            fps=FPS,
            horizon_frames=HORIZON,
            joint_names=_NAMES,
        )

    def OpenSession(self, request: Any, context: Any) -> Any:
        self.sessions[request.session_id] = request
        return pb.OpenSessionResponse()

    def Step(self, request: Any, context: Any) -> Any:
        session = self.sessions.get(request.session_id)
        if session is None:
            context.abort(grpc.StatusCode.NOT_FOUND, "no such session")
        assert session is not None
        self.steps.append(request.keep_until_frame)
        reply = pb.StepResponse()
        heading = session.heading
        cos, sin = math.cos(heading), math.sin(heading)
        for frame in range(
            request.keep_until_frame + 1, request.keep_until_frame + 1 + HORIZON
        ):
            along = frame / FPS
            x = session.root.x + along * cos
            y = session.root.y + along * sin
            out = reply.frames.add(frame=frame, heading=heading)
            out.root.x, out.root.y = x, y
            for name in _NAMES:
                f, left, up = _BODY[name]
                out.joints.add(
                    x=x + f * cos - left * sin, y=y + f * sin + left * cos, z=up
                )
        return reply

    def CloseSession(self, request: Any, context: Any) -> Any:
        self.closed.append(request.session_id)
        return pb.CloseSessionResponse()


class _Backend:
    backend_name = "ardy"

    def __init__(self, address: str) -> None:
        self._address = address
        self.timeout_s = 5.0

    @property
    def name(self) -> str:
        return "ardy"

    def resources(self) -> Resources:
        return Resources()

    def start(self, log_dir: Any = None, gpu: Any = None) -> None:
        return

    def address(self) -> str:
        return self._address

    def close(self) -> None:
        return


class _Walker:
    """A walker that keeps its custom pose and goes where it is put."""

    def __init__(self) -> None:
        self.id = 4242
        self.type_id = "walker.pedestrian.0001"
        self.transform = carla.Transform(
            carla.Location(x=10.0, y=5.0, z=1.0), carla.Rotation(yaw=90.0)
        )
        self.blend = 0.0
        self.poses: list[dict[str, carla.Transform]] = []
        self.controls: list[Any] = []
        self.bounding_box = SimpleNamespace(extent=SimpleNamespace(z=0.9))

    def get_bones(self) -> Any:
        return SimpleNamespace(bone_transforms=_bones())

    def get_transform(self) -> carla.Transform:
        return self.transform

    def get_location(self) -> carla.Location:
        return self.transform.location

    def get_velocity(self) -> carla.Vector3D:
        return carla.Vector3D(0.0, 0.0, 0.0)

    def set_transform(self, transform: carla.Transform) -> None:
        self.transform = transform

    def set_location(self, location: carla.Location) -> None:
        self.transform = carla.Transform(location, self.transform.rotation)

    def set_simulate_physics(self, enabled: bool) -> None:
        self.simulated = enabled

    def apply_control(self, control: Any) -> None:
        self.controls.append(control)

    def get_pose_from_animation(self) -> None:
        return

    def set_bones(self, control: Any) -> None:
        self.poses.append({b.name: b.transform for b in control.bone_transforms})

    def blend_pose(self, blend: float) -> None:
        self.blend = blend


class _Scene:
    def __init__(self) -> None:
        self.walker = _Walker()
        self.entity = PedestrianEntity(
            PedestrianEntityConfig(
                role_name="walker1", spawn_location=SpawnTransform(carla.Transform())
            )
        )
        self.entity._walker = self.walker  # type: ignore[assignment]
        self.world = MagicMock()
        self.elapsed = 0.0
        self.world.get_snapshot.side_effect = lambda: SimpleNamespace(
            timestamp=SimpleNamespace(elapsed_seconds=self.elapsed)
        )

    def run(self, action: SmartPedestrianControllerAction, seconds: float) -> None:
        for _ in range(round(seconds / DT)):
            action.tick(self.world, self.elapsed)
            self.elapsed += DT


@pytest.fixture
def generator() -> Iterator[_Generator]:
    servicer = _Generator()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    pb_grpc.add_PedestrianMotionServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    pool = BackendPool(check_resources=False)
    pool._running["ardy"] = _Backend(f"127.0.0.1:{port}")  # type: ignore[assignment]
    pool_module._ACTIVE = pool
    yield servicer
    pool_module._ACTIVE = None
    server.stop(None)


@pytest.fixture
def scene() -> Iterator[_Scene]:
    built = _Scene()
    register_entity("walker1", built.entity)
    yield built
    unregister_entity("walker1")


class TestFrames:
    def test_the_contract_frame_is_carlas_mirrored(self) -> None:
        x, y, heading = to_contract(3.0, 4.0, 90.0)
        assert (x, y) == (3.0, -4.0)
        assert heading == pytest.approx(-math.pi / 2)
        assert to_carla(x, y, heading) == pytest.approx((3.0, 4.0, 90.0))

    def test_bone_directions_are_in_the_body_frame(self) -> None:
        joints = {"LeftShoulder": (0.0, 0.0, 0.0), "LeftArm": (0.0, 1.0, 0.0)}
        # A bone along +y: ahead of a body facing +y ...
        frame = directions_from_joints(joints, math.pi / 2)
        assert frame["LeftShoulder"] == pytest.approx((1.0, 0.0, 0.0))
        # ... and to the left (-right) of one facing +x.
        frame = directions_from_joints(joints, 0.0)
        assert frame["LeftShoulder"] == pytest.approx((0.0, -1.0, 0.0))

    def test_an_upright_spine_points_up(self) -> None:
        frame = directions_from_joints(dict(_BODY), 1.0)
        assert frame["Spine1"] == pytest.approx((0.0, 0.0, 1.0))
        assert frame["LeftShin"] == pytest.approx((0.0, 0.0, -1.0))


class TestSmartPedestrianControllerAction:
    def test_the_walker_goes_where_the_generator_takes_it(
        self, scene: _Scene, generator: _Generator
    ) -> None:
        action = SmartPedestrianControllerAction("walker1", "walks on", duration_s=2.0)
        scene.run(action, 1.0)
        assert action.moving
        assert scene.walker.blend == 1.0
        # Facing +y in CARLA (yaw 90): it walks along +y, about 1 m in 1 s.
        where = scene.walker.transform.location
        assert where.x == pytest.approx(10.0, abs=1e-6)
        assert where.y == pytest.approx(5.0 + 0.95, abs=0.05)
        assert where.z == pytest.approx(1.0)
        assert scene.walker.transform.rotation.yaw == pytest.approx(90.0)
        # Commanded as every walk is, in CARLA's walker speed units.
        assert scene.walker.controls[-1].speed == pytest.approx(
            walker_control_speed(1.0), rel=1e-3
        )
        assert scene.walker.poses

    def test_the_session_carries_the_conditions(
        self, scene: _Scene, generator: _Generator, tmp_path: Path
    ) -> None:
        keyframes = tmp_path / "poses.json"
        keyframes.write_text(
            json.dumps(
                {
                    "keyframes": [
                        {
                            "time_s": 1.0,
                            "joints": {n: list(p) for n, p in _BODY.items()},
                        }
                    ]
                }
            )
        )
        action = SmartPedestrianControllerAction(
            "walker1",
            "walks",
            prompts=[TimedPrompt(2.0, "waves")],
            waypoints=[WalkWaypoint(10.0, 8.0)],
            speed_ms=1.5,
            keyframes=str(keyframes),
            seed=7,
        )
        scene.run(action, 0.05)
        (session,) = generator.sessions.values()
        assert session.seed == 7
        assert [(t.start_frame, t.text) for t in session.texts] == [
            (0, "walks"),
            (60, "waves"),
        ]
        (waypoint,) = session.waypoints
        # 3 m at 1.5 m/s: 2 s.
        assert (waypoint.frame, waypoint.x, waypoint.y) == (60, 10.0, -8.0)
        (key,) = session.keyframes
        assert key.frame == 30
        assert len(key.joints) == len(_NAMES)
        hips = key.joints[_NAMES.index("Hips")]
        assert (hips.x, hips.y, hips.z) == pytest.approx((10.0, -5.0, 1.0))

    def test_more_motion_is_asked_for_as_it_runs_out(
        self, scene: _Scene, generator: _Generator
    ) -> None:
        action = SmartPedestrianControllerAction("walker1", duration_s=3.0)
        scene.run(action, 2.0)
        assert generator.steps[0] == -1
        assert len(generator.steps) > 2
        # Each asks after the frame the run is at, keeping what was shown.
        assert all(b > a for a, b in zip(generator.steps, generator.steps[1:]))

    def test_it_ends_at_its_last_waypoint_and_hands_the_walker_back(
        self, scene: _Scene, generator: _Generator
    ) -> None:
        action = SmartPedestrianControllerAction(
            "walker1", waypoints=[WalkWaypoint(10.0, 6.0)], speed_ms=1.0
        )
        scene.run(action, 1.5)
        assert not action.moving
        assert scene.walker.blend == 0.0
        assert len(generator.closed) == 1

    def test_it_runs_on_to_a_keyframe_after_its_last_waypoint(
        self, scene: _Scene, generator: _Generator, tmp_path: Path
    ) -> None:
        keyframes = tmp_path / "poses.json"
        keyframes.write_text(
            json.dumps(
                {
                    "keyframes": [
                        {
                            "time_s": 2.0,
                            "joints": {n: list(p) for n, p in _BODY.items()},
                        }
                    ]
                }
            )
        )
        action = SmartPedestrianControllerAction(
            "walker1",
            waypoints=[WalkWaypoint(10.0, 6.0)],
            speed_ms=1.0,
            keyframes=str(keyframes),
        )
        scene.run(action, 1.5)
        assert action.moving
        scene.run(action, 0.6)
        assert not action.moving

    def test_only_the_plan_ahead_is_kept(
        self, scene: _Scene, generator: _Generator
    ) -> None:
        action = SmartPedestrianControllerAction("walker1", duration_s=30.0)
        scene.run(action, 10.0)
        assert len(action._frames) <= 2 * HORIZON

    def test_a_generator_that_fails_to_open_hands_the_walker_back(
        self, scene: _Scene, generator: _Generator, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def refuse(*_: Any, **__: Any) -> None:
            raise RuntimeError("out of memory")

        monkeypatch.setattr(PedestrianMotionClient, "open_session", refuse)
        action = SmartPedestrianControllerAction("walker1", duration_s=2.0)
        with pytest.raises(RuntimeError, match="out of memory"):
            scene.run(action, 0.05)
        assert not action.moving
        assert action._client is None
        assert scene.walker.blend == 0.0

    def test_a_still_walker_stands_facing_where_it_faces(self, scene: _Scene) -> None:
        poser = WalkerPoser("walker1", "test")
        assert poser.begin(upper_body_only=False)
        # A root drifting by a millimetre is not a way to turn to.
        poser.place(
            carla.Location(x=10.0, y=5.0, z=1.0),
            90.0,
            carla.Vector3D(-0.001, -0.002, 0.0),
        )
        control = scene.walker.controls[-1]
        assert control.speed == 0.0
        assert (control.direction.x, control.direction.y) == pytest.approx(
            (0.0, 1.0), abs=1e-6
        )

    def test_a_missing_pedestrian_moves_nothing(self, generator: _Generator) -> None:
        action = SmartPedestrianControllerAction("nobody", duration_s=1.0)
        scene = _Scene()
        scene.run(action, 0.1)
        assert not action.moving
        assert generator.sessions == {}

    def test_a_generator_of_another_contract_is_refused(
        self, generator: _Generator, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "autoware_carla_scenario.pedestrian.motion_client.CONTRACT_VERSION", "9.9.9"
        )
        backend = pool_module._ACTIVE.backend("ardy")  # type: ignore[union-attr]
        client = PedestrianMotionClient(backend.address())  # type: ignore[union-attr]
        with pytest.raises(RuntimeError, match="speaks contract"):
            client.negotiate()
        client.close()

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"speed_ms": 0.0}, "speed_ms"),
            ({"duration_s": -1.0}, "duration_s"),
            ({"prompts": [TimedPrompt(-1.0, "x")]}, "negative"),
        ],
    )
    def test_bad_arguments_are_refused(
        self, kwargs: dict[str, Any], match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            SmartPedestrianControllerAction("walker1", **kwargs)

    def test_a_malformed_keyframes_file_is_refused(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text('{"keyframes": [{"joints": {}}]}')
        with pytest.raises(ValueError, match="not a keyframes file"):
            SmartPedestrianControllerAction("walker1", keyframes=str(bad))
        bad.write_text('{"keyframes": [{"time_s": 1.0, "joints": [[0, 0, 1]]}]}')
        with pytest.raises(ValueError, match="not a keyframes file"):
            SmartPedestrianControllerAction("walker1", keyframes=str(bad))
        with pytest.raises(FileNotFoundError):
            SmartPedestrianControllerAction(
                "walker1", keyframes=str(tmp_path / "x.json")
            )
