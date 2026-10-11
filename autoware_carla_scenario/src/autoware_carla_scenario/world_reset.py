"""Reset the CARLA world between scenarios by loading its map again."""

from __future__ import annotations

import typesafe_carla.carla as carla


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
    """
    name = str(client.get_world().get_map().name).split("/")[-1]
    if not name:
        # Nothing to name it by: the old call is all there is.
        return client.reload_world()
    return client.load_world(name)
