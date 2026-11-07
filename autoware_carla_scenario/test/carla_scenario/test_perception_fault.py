"""PerceptionFaultInjectionAction: one road user reported to the ego's driver
as something it is not, every other one as it is."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import numpy as np
import pytest

from autoware_carla_scenario import (
    EGO_ROLE_NAME,
    PerceivedCategory,
    PerceptionFaultInjectionAction,
)
from autoware_carla_scenario.entity.autoware_entity import AutowareEgoEntity
from autoware_carla_scenario.entity.carla_driver_entity import CarlaDriverEntity
from autoware_carla_scenario.entity.registry import (
    register_vehicle_entity,
    unregister_entity,
)
from autoware_carla_scenario.perception_fault import (
    PerceptionFault,
    perception_fault,
    renderer_type_id,
    reset_perception_faults,
    set_perception_fault,
    t4_label,
)

from .test_driver_renderer import _build, _ego, _vehicle, _walker, _world
from .test_t4_live import _numeric, _tick, _with_a_car


@pytest.fixture(autouse=True)
def _no_faults() -> Iterator[None]:
    reset_perception_faults()
    yield
    reset_perception_faults()
    unregister_entity(EGO_ROLE_NAME)


# ---------------------------------------------------------------------------
# The fault
# ---------------------------------------------------------------------------


def test_a_fault_keeps_what_it_does_not_set() -> None:
    fault = PerceptionFault(length_m=0.5)
    assert fault.size(4.8, 2.0, 1.5) == (0.5, 2.0, 1.5)
    assert not fault.is_none
    assert PerceptionFault().is_none


@pytest.mark.parametrize("size", [-1.0, float("nan"), float("inf")])
def test_a_size_must_be_finite_and_not_negative(size: float) -> None:
    with pytest.raises(ValueError, match="width_m"):
        PerceptionFault(width_m=size)


def test_a_fault_that_changes_nothing_clears_the_one_set() -> None:
    set_perception_fault(2, PerceptionFault(visible=False))
    assert perception_fault(2) == PerceptionFault(visible=False)
    set_perception_fault(2, PerceptionFault())
    assert perception_fault(2) is None


def test_categories_map_to_the_t4_labels_and_type_id_prefixes() -> None:
    assert t4_label(PerceivedCategory.CAR) == 2
    assert t4_label(PerceivedCategory.PEDESTRIAN) == 7
    assert renderer_type_id(PerceivedCategory.PEDESTRIAN).startswith("walker.")
    assert renderer_type_id(PerceivedCategory.BICYCLE).startswith("vehicle.")
    assert renderer_type_id(PerceivedCategory.HAZARD).startswith("static.prop.")
    with pytest.raises(ValueError):
        t4_label(PerceivedCategory.ACTUAL)


# ---------------------------------------------------------------------------
# The action
# ---------------------------------------------------------------------------


class _Actors(list):
    pass


def _world_with(*actors: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(get_actors=lambda: _Actors(actors))


def _actor(actor_id: int, role: str) -> SimpleNamespace:
    return SimpleNamespace(id=actor_id, attributes={"role_name": role})


def test_the_action_sets_the_fault_of_its_road_user_only() -> None:
    action = PerceptionFaultInjectionAction(
        "npc1", PerceivedCategory.PEDESTRIAN, length_m=0.5
    )
    action.execute(_world_with(_actor(2, "npc1"), _actor(3, "npc2")))
    assert perception_fault(2) == PerceptionFault(
        category=PerceivedCategory.PEDESTRIAN, length_m=0.5
    )
    assert perception_fault(3) is None


def test_the_ego_cannot_be_the_road_user() -> None:
    with pytest.raises(ValueError, match="ego"):
        PerceptionFaultInjectionAction(EGO_ROLE_NAME, visible=False)


def test_an_autoware_ego_is_not_implemented_yet() -> None:
    register_vehicle_entity(EGO_ROLE_NAME, object.__new__(AutowareEgoEntity))
    action = PerceptionFaultInjectionAction("npc1", visible=False)
    with pytest.raises(NotImplementedError, match="Autoware"):
        action.execute(_world_with(_actor(2, "npc1")))
    assert perception_fault(2) is None


def test_a_driver_policy_ego_takes_the_fault() -> None:
    register_vehicle_entity(EGO_ROLE_NAME, object.__new__(CarlaDriverEntity))
    action = PerceptionFaultInjectionAction("npc1", visible=False)
    action.execute(_world_with(_actor(2, "npc1")))
    assert perception_fault(2) == PerceptionFault(visible=False)


# ---------------------------------------------------------------------------
# renderer_data
# ---------------------------------------------------------------------------


def test_renderer_data_reports_the_faulted_road_user_as_the_fault_says() -> None:
    set_perception_fault(
        2,
        PerceptionFault(
            category=PerceivedCategory.PEDESTRIAN, length_m=0.6, width_m=0.6
        ),
    )
    message = _build(_world(actors=[_vehicle(2, x=10.0), _vehicle(3, x=20.0)]), _ego())
    faulted, other = message.actors
    assert faulted.type_id == "walker.perception_fault.pedestrian"
    assert (faulted.aabb.size_x, faulted.aabb.size_y, faulted.aabb.size_z) == (
        pytest.approx((0.6, 0.6, 1.5))
    )
    assert other.type_id == "vehicle.tesla.model3"
    assert (other.aabb.size_x, other.aabb.size_y) == pytest.approx((4.8, 2.0))


def test_renderer_data_leaves_an_unseen_road_user_out() -> None:
    set_perception_fault(2, PerceptionFault(visible=False))
    message = _build(_world(actors=[_walker(2, x=10.0), _vehicle(3, x=20.0)]), _ego())
    assert [a.track_id for a in message.actors] == ["3"]


# ---------------------------------------------------------------------------
# The live T4Scene
# ---------------------------------------------------------------------------


def _boxes(ticks: range, fault_from: int, fault: PerceptionFault):
    world, live, _ = _with_a_car(history_rows=3)
    for tick in ticks:
        if tick == fault_from:
            set_perception_fault(2, fault)
        _tick(world, live, tick)
    scene = live.scene("run")
    assert scene is not None
    return scene.annotations


def test_a_scene_reports_the_fault_from_the_row_it_was_set_on() -> None:
    fault = PerceptionFault(category=PerceivedCategory.PEDESTRIAN, length_m=0.5)
    annotations = _boxes(range(10), fault_from=8, fault=fault)
    boxes = annotations.values.boxes
    # Rows 105, 107, 109: the last one only was reported with the fault.
    assert list(_numeric(boxes.labels)) == [1, 1, 7]
    dimensions = _numeric(boxes.dimensions)  # (width, length, height)
    np.testing.assert_allclose(dimensions[:, 1], [4.9, 4.9, 0.5])
    np.testing.assert_allclose(dimensions[:, 0], [2.14, 2.14, 2.14])


def test_a_scene_leaves_an_unseen_road_user_out_of_its_rows() -> None:
    annotations = _boxes(range(10), fault_from=6, fault=PerceptionFault(visible=False))
    # Rows 105 (before the fault), 107 and 109 (after): only the first has it.
    assert list(_numeric(annotations.box_offsets)) == [0, 1, 1, 1]
