"""Reset the CARLA world between scenarios by loading its map again."""

from __future__ import annotations

from typing import Optional

import typesafe_carla.carla as carla


def short_map_name(name: str) -> str:
    """Return the last part of a CARLA map name (``Carla/Maps/X`` -> ``X``).

    That is the part ``FindMapPath`` matches, and the part two names of the
    same map share however the server spells them.
    """
    return str(name).split("/")[-1]


def reload_map(client: carla.Client, map_name: str) -> carla.World:
    """Load *map_name* as a fresh episode, and return its world.

    Loading by the short name rather than with ``client.reload_world()``: see
    :func:`reload_current_map` for why.  An empty name is all
    ``reload_world()`` can be asked for, so that is the call it falls back to.
    """
    name = short_map_name(map_name)
    if not name:
        # Nothing to name it by: the old call is all there is.
        return client.reload_world()
    return client.load_world(name)


def reload_current_map(client: carla.Client) -> carla.World:
    """Load the map the server runs again, as a fresh episode, and return its world.

    This is what ``client.reload_world()`` is for, but CARLA UE5 cannot do it:
    the RPC asks for the current map by an empty name, and
    ``UCarlaEpisode::LoadNewEpisode`` hands that to ``FindMapPath`` unchanged
    (since carla-simulator/carla#8862; ue4-dev was fixed by #8931), so it
    fails at once with a bare ``std::exception`` on every call.  Loading the
    map by its name -- the last part of ``Map.name``, which is what
    ``FindMapPath`` matches; the full ``Carla/Maps/...`` path is not found --
    resets the world the same way on every server, fixed or not.

    The current map is whatever the server runs *now*: when another client may
    have loaded a world of its own, reload the map you mean with
    :func:`reload_map` instead.
    """
    return reload_map(client, str(client.get_world().get_map().name))


def episode_of(world: carla.World) -> Optional[int]:
    """Return the id of the episode *world* belongs to, or ``None`` if unknown.

    Every load of a world -- by this client or by any other on the same
    server -- starts a new episode, so a changed id is how a client sees that
    the world was replaced behind its back.  Reading it costs nothing: the
    client keeps it, unlike ``get_map()``, which builds the whole road network.

    The id is a random 64-bit number, and a client that hands it over signed
    (typesafe-carla does) gives a negative one about every other load: any
    number is an id, so only ``None`` says there is none.
    """
    try:
        return int(world.id)
    except Exception:
        return None


def same_episode(seen: Optional[int], loaded: Optional[int]) -> bool:
    """Whether two ids from :func:`episode_of` name one episode, known to both."""
    if seen is None or loaded is None:
        return False
    # Annotated, so Codon unwraps the Optionals (docs/typecheck.md).
    seen_id: int = seen
    loaded_id: int = loaded
    return seen_id == loaded_id
