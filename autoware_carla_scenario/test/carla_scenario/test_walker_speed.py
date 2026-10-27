"""A walker's speed, as SpeedCondition reads it, from where it stands."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from autoware_carla_scenario.conditions.composition import speed as speed_module


def _walker(actor_id: int, where: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        id=actor_id,
        type_id="walker.pedestrian.0001",
        get_location=lambda: where,
        get_velocity=lambda: SimpleNamespace(x=0.0, y=0.0, z=0.0),
    )


def test_a_walker_on_its_controller_is_measured_by_where_it_stands() -> None:
    """CARLA's walker controller places its walker: get_velocity() reads 0."""
    where = SimpleNamespace(x=0.0, y=0.0, z=0.0)
    walker = _walker(9101, where)
    assert speed_module.walker_velocity(walker, 1.0).magnitude() == 0.0
    where.x = 0.6
    assert speed_module.walker_velocity(walker, 1.5).magnitude() == pytest.approx(1.2)
    # Asked again the same tick: the same.
    assert speed_module.walker_velocity(walker, 1.5).magnitude() == pytest.approx(1.2)
    assert speed_module.walker_velocity(walker, 2.0).magnitude() == 0.0  # stood
    # The clock gone back (a new run): what CARLA says.
    where.x = 50.0
    assert speed_module.walker_velocity(walker, 0.1).magnitude() == 0.0


def test_a_walkers_standstill_reads_its_steps() -> None:
    from autoware_carla_scenario.conditions.comparison import ComparisonRule

    where = SimpleNamespace(x=0.0, y=0.0, z=0.0)
    walker = _walker(9102, where)
    condition = speed_module.SpeedCondition(
        entity_name="w", value=0.1, rule=ComparisonRule.LESS_THAN_OR_EQUAL, label="s"
    )
    assert condition._extract_speed_component(walker, None, 1.0) == 0.0  # type: ignore[arg-type]
    where.y = 1.0
    moving = condition._extract_speed_component(walker, None, 2.0)  # type: ignore[arg-type]
    assert moving == pytest.approx(1.0)
