"""metric_frames.FrameRecorder keeps a frame's road users compactly (#2338):
a shared record of what does not change about each actor, and its motion in
one ``array('d')``. The frames it hands out read the same ActorFrame values,
bit for bit, as the ActorFrame objects it used to keep, at a fraction of the
memory."""

from __future__ import annotations

import gc
import math
import random
import struct
import sys
import tracemalloc
from dataclasses import fields
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from autoware_carla_scenario.constants import EGO_ROLE_NAME
from autoware_carla_scenario.entity_role import canonical_role_name
from autoware_carla_scenario.kinematics.walker_motion import WalkerVelocities
from autoware_carla_scenario.metric_frames import ActorFrame, Frame, FrameRecorder

# ---------------------------------------------------------------------------
# A randomized fake world
# ---------------------------------------------------------------------------

#: Floats a reading may be, beyond uniform ones: signed zeros, the smallest
#: subnormal, the largest double, infinities and a NaN.
_SPECIAL = [0.0, -0.0, 5e-324, -1.7976931348623157e308, math.inf, -math.inf, math.nan]


def _vector(x: float, y: float, z: float) -> Any:
    return SimpleNamespace(x=x, y=y, z=z)


class _Actor:
    """A vehicle, walker or other actor whose readings change every tick;
    *raises* names the readings it cannot give."""

    def __init__(
        self,
        rng: random.Random,
        actor_id: int,
        type_id: str,
        role: str,
        raises: frozenset[str] = frozenset(),
    ) -> None:
        self._rng = rng
        self.id = actor_id
        self.type_id = type_id
        self.attributes = {"role_name": role, "base_type": rng.choice(["car", ""])}
        self.bounding_box = SimpleNamespace(
            extent=_vector(self._float(), self._float(), self._float()),
            location=_vector(self._float(), self._float(), self._float()),
            rotation=SimpleNamespace(yaw=self._float()),
        )
        self._raises = raises
        self._place = [rng.uniform(-100.0, 100.0) for _ in range(3)]

    def _float(self) -> float:
        if self._rng.random() < 0.1:
            return self._rng.choice(_SPECIAL)
        return self._rng.uniform(-1e3, 1e3)

    def tick(self) -> None:
        # A walker walks (its velocity is measured from where it stood); now
        # and then one is teleported, which reads what CARLA reports.
        step = 50.0 if self._rng.random() < 0.05 else 0.05
        self._place = [p + self._rng.uniform(-step, step) for p in self._place]

    def _read(self, name: str) -> None:
        if name in self._raises:
            raise RuntimeError(name)

    def get_transform(self) -> Any:
        x, y, z = self._place
        if self.type_id.startswith("vehicle.") and self._rng.random() < 0.1:
            x = self._rng.choice(_SPECIAL)
        return SimpleNamespace(
            location=_vector(x, y, z),
            rotation=SimpleNamespace(
                roll=self._float(), pitch=self._float(), yaw=self._float()
            ),
        )

    def get_location(self) -> Any:
        return self.get_transform().location

    def get_velocity(self) -> Any:
        return _vector(self._float(), self._float(), self._float())

    def get_acceleration(self) -> Any:
        self._read("acceleration")
        return _vector(self._float(), self._float(), self._float())

    def get_angular_velocity(self) -> Any:
        self._read("angular")
        return _vector(self._float(), self._float(), self._float())

    def get_control(self) -> Any:
        self._read("control")
        return SimpleNamespace(steer=self._float())

    def get_light_state(self) -> int:
        self._read("light")
        return self._rng.randrange(0, 1 << 12)


class _World:
    def __init__(self, actors: list[_Actor], frame_fails: bool = False) -> None:
        self.actors = actors
        self.frame = 100
        self._frame_fails = frame_fails

    def get_actors(self) -> list[_Actor]:
        return self.actors

    def get_snapshot(self) -> Any:
        if self._frame_fails:
            raise RuntimeError("no snapshot")
        return SimpleNamespace(frame=self.frame)

    def tick(self) -> None:
        self.frame += 1
        for actor in self.actors:
            actor.tick()


def _world(seed: int, count: int = 40, frame_fails: bool = False) -> _World:
    rng = random.Random(seed)
    actors = [
        # Egos: one that reads both, one whose light state raises (its steer
        # alone is read), one that reads neither, one spawned for CARLA's
        # native interface; and a walker with the ego's role, whose are not read.
        _Actor(rng, 1, "vehicle.tesla.model3", "Ego"),
        _Actor(rng, 2, "vehicle.tesla.model3", "ego", frozenset({"light"})),
        _Actor(rng, 3, "vehicle.tesla.model3", "hero", frozenset({"control"})),
        _Actor(rng, 4, "walker.pedestrian.0001", "Ego"),
        # Actors that cannot give their acceleration or angular velocity.
        _Actor(rng, 5, "vehicle.audi.tt", "npc", frozenset({"acceleration"})),
        _Actor(rng, 6, "walker.pedestrian.0002", "", frozenset({"angular"})),
        # Neither a vehicle nor a walker: not recorded.
        _Actor(rng, 7, "traffic.traffic_light", ""),
        _Actor(rng, 8, "sensor.camera.rgb", "Ego"),
    ]
    for actor_id in range(100, 100 + count):
        kind = rng.choice(["vehicle.lincoln.mkz", "walker.pedestrian.0003"])
        actors.append(_Actor(rng, actor_id, kind, f"npc{actor_id}"))
    rng.shuffle(actors)
    return _World(actors, frame_fails)


# ---------------------------------------------------------------------------
# The recorder as it was: one ActorFrame object per actor and frame
# ---------------------------------------------------------------------------


def _xyz(read: Any) -> tuple[float, float, float]:
    try:
        v = read()
        return float(v.x), float(v.y), float(v.z)
    except Exception:
        return 0.0, 0.0, 0.0


class _ObjectRecorder:
    """FrameRecorder.record before #2338, verbatim but for the frame it
    keeps (``(frame, t, weight, actors)``, the actors ActorFrame objects)."""

    def __init__(self) -> None:
        self.frames: list[
            tuple[int, float, Optional[float], tuple[ActorFrame, ...]]
        ] = []
        self._walkers = WalkerVelocities()
        self._static: dict[int, Any] = {}

    def record(self, world: Any, elapsed: float, weight: Optional[float]) -> None:
        actors: list[ActorFrame] = []
        for actor in world.get_actors():
            type_id = str(actor.type_id)
            walker = type_id.startswith("walker.")
            if not (walker or type_id.startswith("vehicle.")):
                continue
            transform = actor.get_transform()
            location, rotation = transform.location, transform.rotation
            if walker:
                velocity = self._walkers.walker(actor, elapsed, location)
                v = (velocity.x, velocity.y, velocity.z)
            else:
                reported = actor.get_velocity()
                v = (float(reported.x), float(reported.y), float(reported.z))
            acceleration = _xyz(actor.get_acceleration)
            angular = _xyz(actor.get_angular_velocity)
            key = int(actor.id)
            if key not in self._static:
                box = actor.bounding_box
                self._static[key] = (
                    str(actor.type_id),
                    canonical_role_name(str(actor.attributes.get("role_name", ""))),
                    str(actor.attributes.get("base_type", "")),
                    (float(box.extent.x), float(box.extent.y), float(box.extent.z)),
                    (
                        float(box.location.x),
                        float(box.location.y),
                        float(box.location.z),
                    ),
                    float(box.rotation.yaw),
                )
            kind, role, base, extent, centre, box_yaw = self._static[key]
            steer: Optional[float] = None
            light: Optional[int] = None
            if role == str(EGO_ROLE_NAME) and not walker:
                try:
                    steer = float(actor.get_control().steer)
                    light = int(actor.get_light_state())
                except Exception:
                    pass
            actors.append(
                ActorFrame(
                    id=int(actor.id),
                    type_id=kind,
                    role=role,
                    base_type=base,
                    x=float(location.x),
                    y=float(location.y),
                    z=float(location.z),
                    roll=float(rotation.roll),
                    pitch=float(rotation.pitch),
                    yaw=float(rotation.yaw),
                    vx=v[0],
                    vy=v[1],
                    vz=v[2],
                    ax=acceleration[0],
                    ay=acceleration[1],
                    az=acceleration[2],
                    wx=angular[0],
                    wy=angular[1],
                    wz=angular[2],
                    extent=extent,
                    centre=centre,
                    box_yaw=box_yaw,
                    steer=steer,
                    light_state=light,
                )
            )
        try:
            frame = int(world.get_snapshot().frame)
        except Exception:
            frame = len(self.frames)
        self.frames.append((frame, float(elapsed), weight, tuple(actors)))


def _bits(value: Any) -> Any:
    """*value* with its type, each float as its IEEE bits (NaN equals itself,
    -0.0 differs from 0.0)."""
    if isinstance(value, tuple):
        return tuple(_bits(v) for v in value)
    if isinstance(value, float):
        return (float, struct.pack("<d", value))
    return (type(value), value)


def _actor_bits(actor: ActorFrame) -> Any:
    return tuple((f.name, _bits(getattr(actor, f.name))) for f in fields(actor))


def _record_side_by_side(
    world: _World, ticks: int
) -> tuple[FrameRecorder, _ObjectRecorder]:
    """Record *world* for *ticks* ticks with both recorders, each tick read
    alike: the actors' generators are rewound before the second reads."""
    recorder, reference = FrameRecorder(), _ObjectRecorder()
    rng = random.Random(len(world.actors))
    for k in range(ticks):
        weight = None if k % 3 == 0 else rng.uniform(0.0, 2.0)
        states = [actor._rng.getstate() for actor in world.actors]
        recorder.record(world, 0.05 * k, weight)  # type: ignore[arg-type]
        for actor, state in zip(world.actors, states):
            actor._rng.setstate(state)
        reference.record(world, 0.05 * k, weight)
        world.tick()
    return recorder, reference


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("frame_fails", [False, True])
def test_the_compact_frames_read_what_the_actor_frames_did(
    seed: int, frame_fails: bool
) -> None:
    recorder, reference = _record_side_by_side(
        _world(seed, frame_fails=frame_fails), 30
    )

    assert len(recorder.frames) == len(reference.frames) == 30
    for frame, (number, t, weight, actors) in zip(recorder.frames, reference.frames):
        assert _bits((frame.frame, frame.t, frame.weight)) == _bits((number, t, weight))
        assert [_actor_bits(a) for a in frame.actors] == [
            _actor_bits(a) for a in actors
        ]
        # Read again, the same (a frame makes its actors each time).
        assert [_actor_bits(a) for a in frame.actors] == [
            _actor_bits(a) for a in actors
        ]


def test_the_optional_readings_are_covered() -> None:
    # The fake world gives what the equivalence above has to hold for: an ego
    # with both its steer and light state, one with its steer alone, one with
    # neither, NPCs with neither, readings that raise and special floats.
    recorder, _reference = _record_side_by_side(_world(0), 30)
    actors = [a for frame in recorder.frames for a in frame.actors]
    egos = {a.id: a for a in actors if a.role == "Ego"}
    assert set(egos) == {1, 2, 3, 4}
    assert egos[1].steer is not None and egos[1].light_state is not None
    assert egos[2].steer is not None and egos[2].light_state is None
    assert egos[3].steer is None and egos[3].light_state is None
    assert egos[4].steer is None and egos[4].light_state is None
    assert {7, 8}.isdisjoint(a.id for a in actors)
    values = [getattr(a, f.name) for a in actors for f in fields(a)]
    assert any(isinstance(v, float) and math.isnan(v) for v in values)
    assert any(isinstance(v, float) and str(v) == "-0.0" for v in values)


def test_a_frame_made_with_its_actors_holds_them() -> None:
    actor = ActorFrame(
        1,
        "vehicle.x",
        "Ego",
        "car",
        1.0,
        2.0,
        3.0,
        0.0,
        0.0,
        90.0,
        0.5,
        0.0,
        0.0,
        steer=0.1,
    )
    frame = Frame(frame=3, t=0.15, weight=None, actors=(actor,))
    assert frame.actors == (actor,)
    assert frame == Frame(3, 0.15, None, (actor,))
    assert frame != Frame(3, 0.15, None, ())
    assert hash(frame) == hash(Frame(3, 0.15, None, (actor,)))


def test_a_recorded_frame_equals_one_made_with_its_actors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No NaN, which equals nothing, not even itself.
    monkeypatch.setattr(sys.modules[__name__], "_SPECIAL", [0.0, -0.0, math.inf])
    recorder, reference = _record_side_by_side(_world(1, count=5), 3)
    for frame, (number, t, weight, actors) in zip(recorder.frames, reference.frames):
        made = Frame(number, t, weight, actors)
        assert frame == made and hash(frame) == hash(made)


def test_a_recorded_frame_equals_itself_with_a_nan() -> None:
    # As the frozen dataclass did, though its actors (made afresh each read)
    # hold a NaN, which equals nothing, not even itself.
    recorder, _reference = _record_side_by_side(_world(0), 5)
    nan = [
        frame
        for frame in recorder.frames
        if any(
            isinstance(v, float) and math.isnan(v)
            for a in frame.actors
            for v in (getattr(a, f.name) for f in fields(a))
        )
    ]
    assert nan
    for frame in nan:
        assert frame == frame and not (frame != frame)


def _retained_bytes_per_actor_frame(recorder: Any, world: _World, ticks: int) -> float:
    count = sum(
        1 for a in world.actors if a.type_id.startswith(("vehicle.", "walker."))
    )
    # The first tick reads each actor's static record; leave it out.
    recorder.record(world, 0.0, None)
    world.tick()
    gc.collect()
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        for k in range(1, ticks + 1):
            recorder.record(world, 0.05 * k, 0.5)
            world.tick()
        gc.collect()
        after = tracemalloc.get_traced_memory()[0]
    finally:
        tracemalloc.stop()
    return (after - before) / (count * ticks)


def test_a_recorded_actor_frame_takes_a_fraction_of_the_memory() -> None:
    # An ActorFrame object took ~550 B; the compact frame keeps 15 doubles
    # (120 B) and a reference (8 B) per actor, and a few objects per frame.
    compact = _retained_bytes_per_actor_frame(FrameRecorder(), _world(2, 200), 40)
    objects = _retained_bytes_per_actor_frame(_ObjectRecorder(), _world(2, 200), 40)
    assert compact < 200, compact
    assert objects > 3 * compact, (objects, compact)
