"""The SUMO backend's helpers moved into checked modules (#2386).

Each ``_old_*`` below is the code as it stood in ``traffic/sumo/backend.py``
before it moved out, verbatim but for reading what it read off TraCI,
sumolib or the actor from arguments: ``SumoTrafficBackend._nearest()`` (the
distances handed in), the turn of ``_runs_along()``, the tail of
``_person_angle()`` (the crossing's run handed in) and ``_sumo_class()`` (the
actor's ``base_type`` and whether it is an emergency vehicle handed in).
The new functions must answer the same for generated inputs.
"""

from __future__ import annotations

import math
import random
from typing import Optional

import pytest

from autoware_carla_scenario.traffic.sumo import backend, geometry, options
from autoware_carla_scenario.traffic.sumo.geometry import Pose2D

_OLD_SUMO_CLASSES = {
    "bicycle": ("bicycle", "DEFAULT_BIKETYPE"),
    "motorcycle": ("motorcycle", "acs_motorcycle"),
}
_OLD_PASSENGER = ("passenger", "DEFAULT_VEHTYPE")
_OLD_EMERGENCY = ("emergency", "acs_emergency")


def _old_nearest(
    ids: list[str],
    ego: Optional[Pose2D],
    radius: Optional[float],
    limit: Optional[int],
    positions: dict[str, tuple[float, float]],
) -> list[str]:
    if limit is not None and limit <= 0:
        return []
    if radius is not None and radius <= 0.0:
        return []
    if ego is None:
        # Nothing to measure from: the first ones SUMO lists, up to the
        # limit (the radius cannot be measured yet).
        return ids if limit is None else ids[:limit]
    if not ids or (radius is None and limit is None):
        return ids
    dist = {i: math.dist((ego.x, ego.y), positions[i]) for i in ids}
    ids = sorted(ids, key=dist.__getitem__)
    if radius is not None:
        ids = [i for i in ids if dist[i] <= radius]
    if limit is not None:
        ids = ids[:limit]
    return ids


def _new_nearest(
    ids: list[str],
    ego: Optional[Pose2D],
    radius: Optional[float],
    limit: Optional[int],
    positions: dict[str, tuple[float, float]],
    asked: list[str],
) -> list[str]:
    """As SumoTrafficBackend._nearest() now is, noting what TraCI is asked."""
    cut = geometry.nearest_cut(ids, ego is not None, radius, limit)
    if cut is not None:
        return cut
    assert ego is not None
    dist = {}
    for i in ids:
        asked.append(i)
        dist[i] = math.dist((ego.x, ego.y), positions[i])
    return geometry.nearest_by_distance(ids, dist, radius, limit)


def _old_turn(rotation_deg: float, heading: float) -> float:
    return (rotation_deg - heading + 180.0) % 360.0 - 180.0


def _old_back_angle(dx: float, dy: float) -> Optional[float]:
    if dx == 0.0 and dy == 0.0:
        return None
    # SUMO's angle: clockwise from north; backwards along the crossing.
    return (90.0 - math.degrees(math.atan2(-dy, -dx))) % 360.0


def _old_sumo_class(base_type: str, emergency: bool) -> tuple[str, str]:
    if emergency:
        return _OLD_EMERGENCY
    return _OLD_SUMO_CLASSES.get(base_type, _OLD_PASSENGER)


def _same(a: float, b: float) -> bool:
    return (math.isnan(a) and math.isnan(b)) or (
        a == b and math.copysign(1.0, a) == math.copysign(1.0, b)
    )


def test_the_nearest_are_the_same() -> None:
    rng = random.Random(2386)
    radii: list[Optional[float]] = [None, 0.0, -1.0, -0.0, 5.0, 25.0, 1e9]
    limits: list[Optional[int]] = [None, 0, -1, 1, 2, 3, 10]
    for _ in range(3000):
        count = rng.randint(0, 8)
        ids = [f"car{k}" for k in rng.sample(range(20), count)]
        # Ties too: some share a position.
        spots = [(rng.uniform(-40, 40), rng.uniform(-40, 40)) for _ in range(3)]
        positions = {
            i: rng.choice(spots)
            if rng.random() < 0.3
            else (rng.uniform(-40, 40), rng.uniform(-40, 40))
            for i in ids
        }
        ego = (
            None
            if rng.random() < 0.2
            else Pose2D(rng.uniform(-10, 10), rng.uniform(-10, 10), 0.0, 0.0)
        )
        radius = rng.choice(radii + [rng.uniform(-5, 60)])
        limit = rng.choice(limits)
        old = _old_nearest(list(ids), ego, radius, limit, positions)
        asked: list[str] = []
        given = list(ids)
        new = _new_nearest(given, ego, radius, limit, positions, asked)
        assert new == old
        # TraCI is asked only when the old code measured.
        measured = not (
            (limit is not None and limit <= 0)
            or (radius is not None and radius <= 0.0)
            or ego is None
            or not ids
            or (radius is None and limit is None)
        )
        assert asked == (ids if measured else [])
        # The list itself comes back where the old code handed it back.
        if not measured and (ego is not None or limit is None) and new:
            assert new is given


def test_the_backend_s_nearest_still_reads_traci() -> None:
    class _Vehicle:
        def __init__(self) -> None:
            self.asked: list[str] = []

        def getPosition(self, i: str) -> tuple[float, float]:  # noqa: N802
            self.asked.append(i)
            return {"a": (3.0, 0.0), "b": (1.0, 0.0), "c": (2.0, 0.0)}[i]

    class _Traci:
        vehicle = _Vehicle()

    sumo = backend.SumoTrafficBackend.__new__(backend.SumoTrafficBackend)
    sumo._traci = _Traci()
    ego = Pose2D(0.0, 0.0, 0.0, 0.0)
    assert sumo._nearest(["a", "b", "c"], ego, 2.5, None) == ["b", "c"]
    assert sumo._nearest(["a", "b", "c"], ego, None, 0) == []
    assert _Traci.vehicle.asked == ["a", "b", "c"]


_ANGLES = [
    -720.0,
    -540.0,
    -360.0,
    -180.0,
    -179.999,
    -0.0,
    0.0,
    1e-300,
    -1e-300,
    90.0,
    179.999,
    180.0,
    180.0000001,
    360.0,
    540.0,
    1e17,
    -1e17,
    math.inf,
    -math.inf,
    math.nan,
]


def test_the_turn_is_the_same() -> None:
    rng = random.Random(2386)
    pairs = [(a, b) for a in _ANGLES for b in _ANGLES]
    pairs += [(rng.uniform(-1000, 1000), rng.uniform(-1000, 1000)) for _ in range(5000)]
    pairs += [(rng.choice(_ANGLES), rng.uniform(-400, 400)) for _ in range(2000)]
    for rotation, heading in pairs:
        assert _same(backend._wrap(rotation - heading), _old_turn(rotation, heading))
    assert backend._wrap is geometry._wrap


_RUNS = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    1e-300,
    -1e-300,
    0.1,
    1e300,
    math.inf,
    -math.inf,
    math.nan,
]


def test_the_crossing_back_angle_is_the_same() -> None:
    rng = random.Random(2386)
    pairs = [(dx, dy) for dx in _RUNS for dy in _RUNS]
    pairs += [(rng.uniform(-5, 5), rng.uniform(-5, 5)) for _ in range(5000)]
    pairs += [(rng.choice(_RUNS), rng.uniform(-1, 1)) for _ in range(500)]
    for dx, dy in pairs:
        old = _old_back_angle(dx, dy)
        new = geometry.crossing_back_angle(dx, dy)
        if old is None:
            assert new is None
        else:
            assert new is not None and _same(new, old)


@pytest.mark.parametrize("emergency", [False, True])
@pytest.mark.parametrize(
    "base_type", ["", "car", "bicycle", "motorcycle", "truck", "van", "Bicycle"]
)
def test_the_sumo_class_is_the_same(base_type: str, emergency: bool) -> None:
    assert options.sumo_class_of(base_type, emergency) == _old_sumo_class(
        base_type, emergency
    )
    assert options._SUMO_CLASSES == _OLD_SUMO_CLASSES
    assert options._PASSENGER == _OLD_PASSENGER
    assert options._EMERGENCY == _OLD_EMERGENCY


@pytest.mark.parametrize("emergency", [False, True])
@pytest.mark.parametrize("base_type", [None, "car", "bicycle", "MOTORCYCLE", "van"])
def test_the_backend_s_sumo_class_reads_the_actor(
    base_type: Optional[str], emergency: bool
) -> None:
    class _Actor:
        type_id = "vehicle.test.car"
        attributes = {
            **({} if base_type is None else {"base_type": base_type}),
            **({"special_type": "emergency"} if emergency else {}),
        }

    actor = _Actor()
    want = _old_sumo_class(
        "" if base_type is None else base_type.lower(), backend.is_emergency(actor)
    )
    assert backend._sumo_class(actor) == want
