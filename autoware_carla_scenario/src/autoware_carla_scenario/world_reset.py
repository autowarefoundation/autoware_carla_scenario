"""Reset the CARLA world between scenarios.

Two ways (``ScenarioRunner(world_reset=...)``, ``server.world_reset``,
docs/usage.md "Resetting the world between cases"):

- ``reload`` (the default) loads the map again: a new episode, as fresh as the
  server makes one, at the cost of a map load (~5 s on CARLA UE5).
- ``soft`` keeps the episode and puts back what a run changes in it: it
  destroys every actor the freshly loaded world did not have, and restores the
  weather, the episode settings, the traffic lights' timing and frozen state
  and the spectator, as :func:`capture_baseline` read them off the world right
  after it was loaded.  Anything it cannot put back, or finds left over,
  raises, and the runner loads the map instead.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import typesafe_carla.carla as carla

logger = logging.getLogger(__name__)

#: Load the map again between cases (the default).
RESET_RELOAD = "reload"
#: Put the world back to how it was loaded, without loading it again.
RESET_SOFT = "soft"
#: The values ``world_reset`` takes.
RESET_STRATEGIES = (RESET_RELOAD, RESET_SOFT)


def check_world_reset(world_reset: str, reload_every: int) -> None:
    """Refuse a reset strategy that is not one, or a negative *reload_every*.

    Raises:
        ValueError: naming the value and what it may be.
    """
    if world_reset not in RESET_STRATEGIES:
        raise ValueError(
            f"world_reset must be one of {list(RESET_STRATEGIES)}, "
            f"got {world_reset!r}."
        )
    if reload_every < 0:
        raise ValueError(f"reload_every must be >= 0 (0: never), got {reload_every}.")


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
    (typesafe-carla 0.8.3 and earlier did) gives a negative one about every
    other load: any number is an id, so only ``None`` says there is none.
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


class TrafficLightBaseline:
    """One traffic light as the freshly loaded world had it."""

    actor_id: int
    green_time: float
    yellow_time: float
    red_time: float
    frozen: bool

    def __init__(
        self,
        actor_id: int,
        green_time: float,
        yellow_time: float,
        red_time: float,
        frozen: bool,
    ) -> None:
        self.actor_id = actor_id
        self.green_time = green_time
        self.yellow_time = yellow_time
        self.red_time = red_time
        self.frozen = frozen


class WorldBaseline:
    """What a soft reset puts a world back to: the world right after its load.

    Read by :func:`capture_baseline` off a world nothing has run on yet.
    """

    #: The episode it was read in: a soft reset applies to that one only.
    episode: Optional[int]
    settings: carla.WorldSettings
    weather: carla.WeatherParameters
    #: Every actor the world had: the map's own (traffic lights and signs, the
    #: spectator).  Anything else is a run's, and goes -- except a traffic
    #: light, a sign or the spectator, which are the map's whatever this says.
    actor_ids: List[int]
    traffic_lights: List[TrafficLightBaseline]
    spectator: carla.Transform

    def __init__(
        self,
        episode: Optional[int],
        settings: carla.WorldSettings,
        weather: carla.WeatherParameters,
        actor_ids: List[int],
        traffic_lights: List[TrafficLightBaseline],
        spectator: carla.Transform,
    ) -> None:
        self.episode = episode
        self.settings = settings
        self.weather = weather
        self.actor_ids = actor_ids
        self.traffic_lights = traffic_lights
        self.spectator = spectator


def capture_baseline(world: carla.World) -> WorldBaseline:
    """Read what a soft reset restores off *world*, freshly loaded."""
    actor_ids: List[int] = []
    for actor in world.get_actors():
        actor_ids.append(int(actor.id))
    lights: List[TrafficLightBaseline] = []
    for light in world.get_actors().filter("traffic.traffic_light*"):
        lights.append(
            TrafficLightBaseline(
                int(light.id),
                float(light.get_green_time()),
                float(light.get_yellow_time()),
                float(light.get_red_time()),
                bool(light.is_frozen()),
            )
        )
    return WorldBaseline(
        episode_of(world),
        world.get_settings(),
        world.get_weather(),
        actor_ids,
        lights,
        world.get_spectator().get_transform(),
    )


def _belongs_to_the_map(type_id: str) -> bool:
    """Whether an actor of *type_id* is the map's, whatever the baseline lists.

    Traffic lights and signs, and the spectator, are never a run's: one the
    baseline missed (read before the server listed it) is kept, not destroyed.
    """
    return type_id.startswith("traffic.") or type_id == "spectator"


def _destroy_order(type_id: str) -> int:
    """Sensors and controllers before what they are attached to or drive."""
    if type_id.startswith("sensor."):
        return 0
    if type_id.startswith("controller."):
        return 1
    if type_id.startswith("walker."):
        return 2
    if type_id.startswith("vehicle."):
        return 3
    return 4


def soft_reset_world(
    client: carla.Client,
    world: carla.World,
    baseline: WorldBaseline,
    scenario_name: str = "",
) -> int:
    """Put *world* back to *baseline* without loading the map again.

    In this order: the recorder is stopped; every actor the baseline does not
    list (other than a traffic light, a sign or the spectator) is destroyed
    (sensors and controllers first), then the world is
    ticked; the traffic lights get their timing back, are reset to the start
    of their cycle and frozen as they were; the weather and the spectator are
    put back, the world is ticked once more, and the episode settings are
    applied last (a loaded world is asynchronous: the next run makes it
    synchronous again).

    Returns:
        How many actors were destroyed.

    Raises:
        RuntimeError: the world is not in the baseline's episode, a traffic
            light has no state in the baseline (whatever a case did to it could
            not be put back), or an actor survived (or a map actor is gone):
            the caller loads the map instead.
            Whatever CARLA raises along the way is passed on, for the same.
    """
    if not same_episode(episode_of(world), baseline.episode):
        raise RuntimeError("the world is not the episode the baseline was read in")
    try:
        client.stop_recorder()
    except Exception:  # noqa: BLE001 - nothing may be recording
        pass
    kept = set(baseline.actor_ids)
    # A traffic light the baseline does not list (read before the server
    # listed it) is kept, but its timing and frozen state are not known: a
    # case may have frozen it, and nothing here could put that back.
    known_lights = set([saved.actor_id for saved in baseline.traffic_lights])
    unknown_lights: List[int] = []
    for light in world.get_actors().filter("traffic.traffic_light*"):
        if int(light.id) not in known_lights:
            unknown_lights.append(int(light.id))
    if unknown_lights:
        raise RuntimeError(
            f"traffic light(s) {sorted(unknown_lights)} have no baseline state"
        )
    doomed: List[carla.Actor] = []
    for actor in world.get_actors():
        if int(actor.id) not in kept and not _belongs_to_the_map(str(actor.type_id)):
            doomed.append(actor)
    doomed.sort(key=lambda a: _destroy_order(str(a.type_id)))
    destroyed = 0
    for actor in doomed:
        if actor.destroy():
            destroyed += 1
    world.tick()

    for light in world.get_actors().filter("traffic.traffic_light*"):
        for saved in baseline.traffic_lights:
            if saved.actor_id == int(light.id):
                light.set_green_time(saved.green_time)
                light.set_yellow_time(saved.yellow_time)
                light.set_red_time(saved.red_time)
    world.reset_all_traffic_lights()
    for light in world.get_actors().filter("traffic.traffic_light*"):
        for saved in baseline.traffic_lights:
            if saved.actor_id == int(light.id):
                light.freeze(saved.frozen)
    world.set_weather(baseline.weather)
    world.get_spectator().set_transform(baseline.spectator)
    world.tick()

    left: List[int] = []
    present: List[int] = []
    for actor in world.get_actors():
        actor_id = int(actor.id)
        present.append(actor_id)
        if actor_id not in kept and not _belongs_to_the_map(str(actor.type_id)):
            left.append(actor_id)
    if left:
        raise RuntimeError(f"actor(s) {sorted(left)} survived the soft reset")
    missing = sorted(kept - set(present))
    if missing:
        raise RuntimeError(f"map actor(s) {missing} are gone from the world")
    world.apply_settings(baseline.settings)
    logger.info(
        "[%s] World soft-reset: %d actor(s) destroyed", scenario_name, destroyed
    )
    return destroyed
