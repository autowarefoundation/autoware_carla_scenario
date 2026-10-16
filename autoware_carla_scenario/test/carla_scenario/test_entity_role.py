"""Unit tests for EntityRole."""

from __future__ import annotations

import pytest

from autoware_carla_scenario import EntityRole


class TestEntityRoleConstruction:
    """Tests for EntityRole construction and validation."""

    def test_ego_factory(self) -> None:
        role = EntityRole.ego()
        assert str(role) == "Ego"

    def test_npc_factory(self) -> None:
        role = EntityRole.npc(1)
        assert str(role) == "npc1"

    def test_npc_factory_large_number(self) -> None:
        role = EntityRole.npc(42)
        assert str(role) == "npc42"

    def test_npc_factory_rejects_zero(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            EntityRole.npc(0)

    def test_npc_factory_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            EntityRole.npc(-1)

    def test_direct_construction_ego(self) -> None:
        role = EntityRole("Ego")
        assert str(role) == "Ego"

    def test_direct_construction_npc(self) -> None:
        role = EntityRole("npc1")
        assert str(role) == "npc1"

    def test_direct_construction_custom_category(self) -> None:
        role = EntityRole("vehicle2")
        assert str(role) == "vehicle2"

    def test_direct_construction_pedestrian(self) -> None:
        role = EntityRole("pedestrian3")
        assert str(role) == "pedestrian3"

    def test_rejects_lowercase_ego(self) -> None:
        with pytest.raises(ValueError, match="Invalid entity role name"):
            EntityRole("ego")

    def test_rejects_uppercase_category(self) -> None:
        with pytest.raises(ValueError, match="Invalid entity role name"):
            EntityRole("NPC1")

    def test_rejects_underscore(self) -> None:
        with pytest.raises(ValueError, match="Invalid entity role name"):
            EntityRole("npc_1")

    def test_rejects_leading_zero(self) -> None:
        with pytest.raises(ValueError, match="Invalid entity role name"):
            EntityRole("npc01")

    def test_rejects_zero_suffix(self) -> None:
        with pytest.raises(ValueError, match="Invalid entity role name"):
            EntityRole("npc0")

    def test_rejects_empty_string(self) -> None:
        with pytest.raises(ValueError, match="Invalid entity role name"):
            EntityRole("")

    def test_rejects_number_only(self) -> None:
        with pytest.raises(ValueError, match="Invalid entity role name"):
            EntityRole("123")

    def test_rejects_category_only(self) -> None:
        with pytest.raises(ValueError, match="Invalid entity role name"):
            EntityRole("npc")

    def test_rejects_spaces(self) -> None:
        with pytest.raises(ValueError, match="Invalid entity role name"):
            EntityRole("npc 1")


class TestEntityRoleEquality:
    """Tests for equality and hashing."""

    def test_equal_roles(self) -> None:
        assert EntityRole("npc1") == EntityRole("npc1")

    def test_unequal_roles(self) -> None:
        assert EntityRole("npc1") != EntityRole("npc2")

    def test_ego_equality(self) -> None:
        assert EntityRole.ego() == EntityRole("Ego")

    def test_npc_factory_equality(self) -> None:
        assert EntityRole.npc(1) == EntityRole("npc1")

    def test_hash_consistency(self) -> None:
        a = EntityRole("npc1")
        b = EntityRole("npc1")
        assert hash(a) == hash(b)

    def test_usable_as_dict_key(self) -> None:
        d = {EntityRole.npc(1): "first", EntityRole.ego(): "ego"}
        assert d[EntityRole("npc1")] == "first"
        assert d[EntityRole("Ego")] == "ego"

    def test_usable_in_set(self) -> None:
        s = {EntityRole.npc(1), EntityRole.npc(1), EntityRole.ego()}
        assert len(s) == 2

    def test_not_equal_to_string(self) -> None:
        assert EntityRole("npc1") != "npc1"


class TestEntityRoleStringConversion:
    """Tests for __str__ and __repr__."""

    def test_str(self) -> None:
        assert str(EntityRole.npc(1)) == "npc1"
        assert str(EntityRole.ego()) == "Ego"

    def test_repr(self) -> None:
        assert repr(EntityRole.npc(1)) == "EntityRole('npc1')"
        assert repr(EntityRole.ego()) == "EntityRole('Ego')"


class _FakeActor:
    def __init__(self, actor_id: int, role_name: str | None) -> None:
        self.id = actor_id
        self.attributes = {} if role_name is None else {"role_name": role_name}
        self.is_alive = True


class _FakeWorld:
    """A world that counts full scans (``get_actors()``) and id lookups."""

    def __init__(self, world_id: int, actors: list[_FakeActor]) -> None:
        self.id = world_id
        self.actors = actors
        self.scans = 0
        self.lookups = 0
        # LibCarla's get_actor() serves destroyed actors from its cache too.
        self.known: dict[int, _FakeActor] = {a.id: a for a in actors}

    def get_actors(self) -> list[_FakeActor]:
        self.scans += 1
        for actor in self.actors:
            self.known[actor.id] = actor
        return list(self.actors)

    def get_actor(self, actor_id: int) -> _FakeActor | None:
        self.lookups += 1
        return self.known.get(actor_id)

    def destroy(self, actor: _FakeActor) -> None:
        actor.is_alive = False
        self.actors.remove(actor)


@pytest.fixture
def clean_cache():
    from autoware_carla_scenario import entity_role

    entity_role._ACTOR_IDS.clear()
    yield entity_role
    entity_role._ACTOR_IDS.clear()


class TestFindActorInWorld:
    """find_actor_in_world() scans once, then looks the actor up by id."""

    def test_scans_once_then_looks_up_by_id(self, clean_cache) -> None:
        ego = _FakeActor(7, "Ego")
        world = _FakeWorld(1, [_FakeActor(3, None), _FakeActor(5, "npc1"), ego])
        for _ in range(5):
            assert clean_cache.find_actor_in_world(world, EntityRole.ego()) is ego
        assert world.scans == 1
        assert world.lookups == 4

    def test_absent_role_is_none_and_not_cached(self, clean_cache) -> None:
        world = _FakeWorld(1, [_FakeActor(5, "npc1")])
        assert clean_cache.find_actor_in_world(world, "Ego") is None
        assert clean_cache.find_actor_in_world(world, "Ego") is None
        assert world.scans == 2
        assert clean_cache._ACTOR_IDS == {}

    def test_destroyed_actor_is_never_returned(self, clean_cache) -> None:
        ego = _FakeActor(7, "Ego")
        world = _FakeWorld(1, [ego])
        assert clean_cache.find_actor_in_world(world, "Ego") is ego
        world.destroy(ego)
        assert clean_cache.find_actor_in_world(world, "Ego") is None

    def test_respawned_actor_is_found_again(self, clean_cache) -> None:
        old = _FakeActor(7, "Ego")
        world = _FakeWorld(1, [old])
        assert clean_cache.find_actor_in_world(world, "Ego") is old
        world.destroy(old)
        new = _FakeActor(9, "Ego")
        world.actors.append(new)
        assert clean_cache.find_actor_in_world(world, "Ego") is new
        assert clean_cache.find_actor_in_world(world, "Ego") is new
        assert world.scans == 2

    def test_actor_without_that_role_any_more_rescans(self, clean_cache) -> None:
        actor = _FakeActor(7, "npc1")
        world = _FakeWorld(1, [actor])
        assert clean_cache.find_actor_in_world(world, "npc1") is actor
        actor.attributes = {"role_name": "npc2"}
        assert clean_cache.find_actor_in_world(world, "npc1") is None

    def test_cache_is_per_world(self, clean_cache) -> None:
        # Actor ids start over in a new episode: an id found in one world
        # must not be trusted in another.
        first = _FakeWorld(1, [_FakeActor(7, "Ego")])
        second_ego = _FakeActor(9, "Ego")
        second = _FakeWorld(2, [_FakeActor(7, "npc1"), second_ego])
        clean_cache.find_actor_in_world(first, "Ego")
        assert clean_cache.find_actor_in_world(second, "Ego") is second_ego
        assert second.lookups == 0

    def test_matches_a_full_scan(self, clean_cache) -> None:
        from autoware_carla_scenario.conditions.base import (
            find_actor_by_role_name,
            find_actor_in_list,
        )

        actors = [_FakeActor(i, f"npc{i}" if i % 2 else None) for i in range(1, 9)]
        world = _FakeWorld(1, actors)
        for name in ("npc1", "npc3", "npc4", "Ego", "npc7", "npc3"):
            expected = find_actor_in_list(list(actors), name)
            assert find_actor_by_role_name(world, name) is expected
