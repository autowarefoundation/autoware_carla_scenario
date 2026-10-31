"""Environment action: set the world's weather and sun position, the road's grip
and how far drivers see.

A wet road holds a tyre less well than a dry one, and CARLA does not know it:
its weather is drawn, never driven on.  So the action scales every vehicle's
wheel friction (``friction_force_multiplier``) to the grip the road's water
leaves it (:func:`tire_friction_range`, by the weather rules of
:mod:`autoware_carla_scenario.weather_rules`), and
:func:`follow_tire_friction` gives a vehicle spawned later the same.

Nor is CARLA's weather ever seen through: the action records how far drivers
see in the weather it leaves (:mod:`autoware_carla_scenario.visibility`), for
a traffic backend that models their sight.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from ..conditions.base import BaseCondition
from ..utils.vehicles import as_vehicle
from ..visibility import at_quantile, set_world_visibility
from ..entity_role import _role_name
from ..weather_rules import vehicle_range, vehicle_rules_in_use, weather_range
from .base import BaseAction, TickTiming

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

logger = logging.getLogger(__name__)

__all__ = [
    "EnvironmentAction",
    "apply_tire_friction",
    "apply_weather_grip",
    "follow_tire_friction",
    "rain_weather",
    "tire_friction",
    "tire_friction_range",
    "world_tire_friction",
]

#: CARLA's rain presets (``SoftRain``, ``MidRainy``, ``HardRain``; LibCarla's
#: ``WeatherParameters.cpp``), by precipitation: the cloudiness, standing water
#: and wetness that come with it.  The wetness is the night presets', the ones
#: that set it.  Precipitation alone barely shows in a camera image -- a few
#: drops -- on a dry road under a clear sky: it is these that make it rain.
_RAIN_PRESETS: tuple[tuple[float, float, float, float], ...] = (
    # precipitation, cloudiness, precipitation_deposits, wetness
    (0.0, 0.0, 0.0, 0.0),
    (30.0, 60.0, 50.0, 30.0),
    (60.0, 80.0, 60.0, 60.0),
    (100.0, 100.0, 90.0, 100.0),
)


def rain_weather(rain: float) -> dict[str, float]:
    """The weather fields of rain at intensity *rain* (0-100), as CARLA's presets set them.

    ``precipitation`` is *rain*; ``cloudiness``, ``precipitation_deposits`` and
    ``wetness`` are interpolated linearly between CARLA's rain presets.
    """
    level = min(max(float(rain), 0.0), 100.0)
    for (p0, c0, d0, w0), (p1, c1, d1, w1) in zip(_RAIN_PRESETS, _RAIN_PRESETS[1:]):
        if level <= p1:
            t = (level - p0) / (p1 - p0)
            return {
                "precipitation": level,
                "cloudiness": c0 + t * (c1 - c0),
                "precipitation_deposits": d0 + t * (d1 - d0),
                "wetness": w0 + t * (w1 - w0),
            }
    raise AssertionError("unreachable: the presets end at 100")


def tire_friction_range(
    wetness: float, precipitation_deposits: float = 0.0
) -> tuple[float, float]:
    """The ``(low, high)`` share of the dry road's grip a road this wet leaves.

    The weather rules' ``tire_friction``
    (:mod:`autoware_carla_scenario.weather_rules`): by default, by how wet the
    road is -- the larger of the two, ``water`` -- a dry road's around its own
    grip, a wet one's about three quarters of it, a flooded one's about half.

    Args:
        wetness: The weather's ``wetness``, 0-100.
        precipitation_deposits: Its standing water, 0-100.
    """
    return weather_range(
        "tire_friction",
        {
            "wetness": float(wetness),
            "precipitation_deposits": float(precipitation_deposits),
        },
    )


def tire_friction(
    wetness: float, precipitation_deposits: float = 0.0, quantile: float = 0.5
) -> float:
    """The share of the dry road's grip at *quantile* (0-1) of :func:`tire_friction_range`.

    0.5, the default, is the middle of the range: a dry road's is 1.0.
    """
    low, high = tire_friction_range(wetness, precipitation_deposits)
    return at_quantile(low, high, quantile)


#: World (episode) id -> the grip share last applied in it.
_TIRE_FACTORS: dict[int, float] = {}
#: World (episode) id -> vehicle id -> its wheels' friction before any was
#: applied: a later factor scales these, not what an earlier one left.
_TIRE_BASES: dict[int, dict[int, list[float]]] = {}
#: World (episode) id -> the ids of every actor already looked at, vehicle or
#: not, so following the world reads only the actors new since.
_TIRE_SEEN: dict[int, set[int]] = {}
#: World (episode) id -> the weather the grip was read off and the quantile,
#: when it was (:func:`apply_weather_grip`): each vehicle's grip is then its
#: own rules' (:func:`~autoware_carla_scenario.weather_rules.vehicle_range`).
_TIRE_WEATHER: dict[int, tuple[dict[str, float], float]] = {}
#: World (episode) id -> the grip share the ego was given, where it differs
#: by vehicle: what the ODD's ``tire_friction`` probe reads.
_TIRE_EGO: dict[int, float] = {}


def _tire_state(world_id: int) -> tuple[dict[int, list[float]], set[int]]:
    """The world's base frictions and seen actors; another world's are dropped
    (an episode once left is not come back to)."""
    for old in [w for w in _TIRE_BASES if w != world_id]:
        # `in` and `del` rather than pop() with a default, which Codon lacks.
        del _TIRE_BASES[old]
        if old in _TIRE_SEEN:
            del _TIRE_SEEN[old]
        if old in _TIRE_FACTORS:
            del _TIRE_FACTORS[old]
        if old in _TIRE_WEATHER:
            del _TIRE_WEATHER[old]
        if old in _TIRE_EGO:
            del _TIRE_EGO[old]
    return _TIRE_BASES.setdefault(world_id, {}), _TIRE_SEEN.setdefault(world_id, set())


def _scale_wheels(
    bases: dict[int, list[float]], actor: "carla.Actor", factor: float
) -> None:
    """Set *actor*'s wheel friction to *factor* times what it was spawned with."""
    vehicle = as_vehicle(actor)
    physics = vehicle.get_physics_control()
    wheels = list(physics.wheels)
    if actor.id not in bases:
        bases[actor.id] = [float(w.friction_force_multiplier) for w in wheels]
    changed = False
    for wheel, spawned in zip(wheels, bases[actor.id]):
        wanted = spawned * factor
        if abs(float(wheel.friction_force_multiplier) - wanted) > 1e-6:
            wheel.friction_force_multiplier = wanted
            changed = True
    if changed:
        physics.wheels = wheels
        vehicle.apply_physics_control(physics)


def _vehicle_factor(world_id: int, actor: "carla.Actor", factor: float) -> float:
    """*actor*'s grip share: its own rules' at the world's weather and
    quantile where the rules differ by vehicle, else the world's *factor*."""
    if world_id not in _TIRE_WEATHER or not vehicle_rules_in_use():
        return factor
    inputs, quantile = _TIRE_WEATHER[world_id]
    role = _role_name(actor)
    low, high = vehicle_range(
        "tire_friction", inputs, str(actor.type_id), "" if role is None else role
    )
    return at_quantile(low, high, quantile)


def _scale_vehicles(
    world: "carla.World", actors: "carla.ActorList", factor: float
) -> int:
    """Scale the wheels of the vehicles among *actors*, each by its own grip
    share (:func:`_vehicle_factor`); return how many."""
    world_id = world.id
    bases, seen = _tire_state(world_id)
    count = 0
    for actor in actors.filter("vehicle.*"):
        # Looked at once: a vehicle gone since the list was read, or one
        # without wheels, is left as it is rather than tried every tick.
        seen.add(actor.id)
        # One `except` per exception: Codon takes no tuple (docs/typecheck.md).
        try:
            share = _vehicle_factor(world_id, actor, factor)
            _scale_wheels(bases, actor, share)
            if share != factor and _role_name(actor) == "Ego":
                _TIRE_EGO[world_id] = share
        except RuntimeError:
            continue
        except AttributeError:
            continue
        count += 1
    return count


def apply_tire_friction(world: "carla.World", factor: float) -> None:
    """Give every vehicle in *world* *factor* times the grip it was spawned with.

    The factor is remembered for the world, so :func:`follow_tire_friction`
    gives the vehicles spawned later the same.  Applied again, a factor
    replaces the last one rather than scaling what it left; the same factor
    again only reaches the vehicles new since.

    Raises:
        ValueError: If *factor* is not positive.
    """
    if not factor > 0.0:
        raise ValueError(f"tire friction must be positive, got {factor}")
    _set_grip(world, factor, None)


def apply_weather_grip(
    world: "carla.World", inputs: dict[str, float], quantile: float
) -> None:
    """Give every vehicle in *world* the grip the weather *inputs* leave, at
    *quantile* (0-1) of its range: the run's rules' range, or, where they
    differ by vehicle (``environment.rules.vehicles``), each vehicle's own.

    Remembered for the world, as :func:`apply_tire_friction`'s factor is, so
    :func:`follow_tire_friction` gives a vehicle spawned later its own.
    """
    low, high = weather_range("tire_friction", inputs)
    _set_grip(world, at_quantile(low, high, quantile), (dict(inputs), quantile))


def _set_grip(
    world: "carla.World",
    factor: float,
    weather: Optional[tuple[dict[str, float], float]],
) -> None:
    """Record the world's grip -- *factor*, and the *weather* each vehicle's
    is read off, if any -- and scale its vehicles; the same again only
    reaches the vehicles new since."""
    world_id = world.id
    _tire_state(world_id)
    same_weather = (weather is None and world_id not in _TIRE_WEATHER) or (
        weather is not None
        and world_id in _TIRE_WEATHER
        and _TIRE_WEATHER[world_id] == weather
    )
    if world_id in _TIRE_FACTORS and _TIRE_FACTORS[world_id] == factor and same_weather:
        follow_tire_friction(world)
        return
    _TIRE_FACTORS[world_id] = factor
    if world_id in _TIRE_EGO:
        del _TIRE_EGO[world_id]
    if weather is None:
        if world_id in _TIRE_WEATHER:
            del _TIRE_WEATHER[world_id]
    else:
        _TIRE_WEATHER[world_id] = weather
    count = _scale_vehicles(world, world.get_actors(), factor)
    if weather is not None and vehicle_rules_in_use():
        logger.info(
            "Tire friction set by each vehicle's rules on %d vehicle(s) "
            "(%.3f of the dry road's by the default rules)",
            count,
            factor,
        )
    else:
        logger.info(
            "Tire friction set to %.3f of the dry road's on %d vehicle(s)",
            factor,
            count,
        )


def world_tire_friction(world_id: Optional[int]) -> Optional[float]:
    """The share of the dry road's grip :func:`apply_tire_friction` (or
    :func:`apply_weather_grip`) last gave the vehicles of the world
    *world_id* -- the ego's, where it differs by vehicle -- or ``None`` where
    it gave none.

    What the ODD's ``tire_friction`` probe reads.  The vehicles' wheels are
    CARLA's physics alone: no traffic backend reads it.
    """
    if world_id is None:
        return None
    # Where it differs by vehicle, the ego's.
    if world_id in _TIRE_EGO:
        return _TIRE_EGO[world_id]
    if world_id in _TIRE_FACTORS:
        return _TIRE_FACTORS[world_id]
    return None


def follow_tire_friction(world: "carla.World") -> None:
    """Give the vehicles spawned since the last call the world's tire friction.

    A tick callback, which :class:`~autoware_carla_scenario.BaseScenario`
    registers for a scenario with an action that sets the grip: background
    traffic spawns during the run, after the weather was set.  It reads the
    actors' ids off the world's snapshot and asks the server only for those
    it has not looked at; nothing happens before :func:`apply_tire_friction`
    set a factor for *world*.
    """
    world_id = world.id
    if world_id not in _TIRE_FACTORS:
        return
    _bases, seen = _tire_state(world_id)
    new = [snapshot.id for snapshot in world.get_snapshot() if snapshot.id not in seen]
    if len(new) == 0:
        return
    seen.update(new)
    _scale_vehicles(world, world.get_actors(new), _TIRE_FACTORS[world_id])


def _set_weather_field(
    weather: "carla.WeatherParameters", name: str, value: float
) -> None:
    """Set the ``carla.WeatherParameters`` attribute constructor parameter *name* sets.

    Every parameter sets the attribute of its own name.  Spelled out rather
    than ``setattr(weather, name, value)``, which the static check (Codon)
    cannot compile: a new knob is a parameter, an entry in
    ``EnvironmentAction``'s settings and a branch here.
    """
    if name == "cloudiness":
        weather.cloudiness = value
    elif name == "precipitation":
        weather.precipitation = value
    elif name == "precipitation_deposits":
        weather.precipitation_deposits = value
    elif name == "wetness":
        weather.wetness = value
    elif name == "wind_intensity":
        weather.wind_intensity = value
    elif name == "fog_density":
        weather.fog_density = value
    elif name == "fog_distance":
        weather.fog_distance = value
    elif name == "sun_altitude_angle":
        weather.sun_altitude_angle = value
    elif name == "sun_azimuth_angle":
        weather.sun_azimuth_angle = value
    else:
        raise KeyError(name)


def _weather_inputs(weather: "carla.WeatherParameters") -> dict[str, float]:
    """What the weather rules read of *weather*: every field a setting sets."""
    return {
        "cloudiness": float(weather.cloudiness),
        "precipitation": float(weather.precipitation),
        "precipitation_deposits": float(weather.precipitation_deposits),
        "wetness": float(weather.wetness),
        "wind_intensity": float(weather.wind_intensity),
        "fog_density": float(weather.fog_density),
        "fog_distance": float(weather.fog_distance),
        "sun_altitude_angle": float(weather.sun_altitude_angle),
        "sun_azimuth_angle": float(weather.sun_azimuth_angle),
    }


class EnvironmentAction(BaseAction):
    """Change the weather and the position of the sun.

    OpenSCENARIO's ``EnvironmentAction``, as far as CARLA models it.  The one
    action here that changes what the **sensors** see rather than how a vehicle
    drives, which is what makes it worth having beyond scenario coverage: a
    perception scenario that cannot set the weather is not testing perception.

    **Every field is optional and means "leave as is".**  The action reads the
    world's current weather and overwrites only what it was given, so a
    scenario that wants rain does not have to restate the sun's position and
    silently reset it to a default.

    ``TimeOfDay`` maps onto :attr:`sun_altitude_angle` and
    :attr:`sun_azimuth_angle` rather than onto a wall-clock time.  CARLA has no
    clock, and offering one would put a field in the document that the runtime
    could not honour.

    The default phase is ``init``: most scenarios set the weather once, before
    anything moves.  Registering it on the tick loop instead is what makes the
    weather change *during* a run.

    **How far drivers see follows the weather.**  An action that sets the fog,
    the rain or the sun, or names a visibility, records the distance drivers
    see in the resulting weather
    (:func:`~autoware_carla_scenario.visibility.visibility`): *visibility_m*
    if given, else the point *visibility_quantile* of
    :func:`~autoware_carla_scenario.visibility.visibility_range` -- its middle
    by default.  A traffic backend that models drivers' sight (``sumo``)
    holds its traffic to it, and the TrafficManager with
    ``follow_visibility: true``.

    **The road's grip follows the water on it.**  An action that sets the
    ``wetness`` or the standing water (``rain`` sets both), or names a tire
    friction, scales every vehicle's wheel friction to the share of the dry
    road's grip the resulting weather leaves (:func:`tire_friction`):
    *tire_friction* if given, else the point *tire_friction_quantile* of
    :func:`tire_friction_range` -- its middle by default, so ``rain=0`` keeps
    the dry grip.  Vehicles spawned later get it from
    :func:`follow_tire_friction`.

    Args:
        cloudiness: 0-100.
        precipitation: Rain intensity, 0-100.
        precipitation_deposits: Standing water on the road, 0-100.
        wetness: Surface wetness, 0-100.
        wind_intensity: 0-100.
        fog_density: 0-100.
        fog_distance: Metres before fog starts.
        sun_altitude_angle: Degrees above the horizon; negative is night.
        sun_azimuth_angle: Degrees.
        rain: Rain as a camera sees it, 0-100: ``precipitation``, and the
            ``cloudiness``, ``precipitation_deposits`` and ``wetness`` CARLA's
            rain presets pair with it (:func:`rain_weather`).  Cloudiness,
            standing water or wetness given as well wins over the one *rain*
            derives; precipitation may not be given with it.  ``rain=0`` is a
            dry road under a clear sky, whatever the map's own weather.
            Keyword-only.
        tire_friction: The vehicles' grip, as a share of the dry road's
            (1.0): fixed, whatever the weather.  Keyword-only.
        tire_friction_quantile: Where in the range of grip the weather leaves
            (:func:`tire_friction_range`) the vehicles' is, 0 (its low end) to
            1 (its high end); 0.5 when the action sets the road's water and
            this is not given.  *tire_friction* wins over it.  An ODD sweep
            draws it per case.  Keyword-only.
        visibility_m: How far drivers see, metres: fixed, whatever the
            weather.  Keyword-only.
        visibility_quantile: Where in the range of visibility the weather
            leaves (:func:`~autoware_carla_scenario.visibility.visibility_range`)
            the drivers' is, 0 (its low end) to 1 (its high end); 0.5 when the
            action sets the fog, the rain or the sun and this is not given.
            *visibility_m* wins over it.  An ODD sweep draws it per case.
            Keyword-only.
        condition: Trigger condition (see :class:`BaseCondition`).
        timing: Tick phase.
        label: Human-readable identifier.
        once: If ``True`` (default) the action fires at most once.
    """

    _settings: dict[str, float]
    _tire_friction: Optional[float]
    _tire_friction_quantile: Optional[float]
    _visibility_m: Optional[float]
    _visibility_quantile: Optional[float]

    def __init__(
        self,
        cloudiness: Optional[float] = None,
        precipitation: Optional[float] = None,
        precipitation_deposits: Optional[float] = None,
        wetness: Optional[float] = None,
        wind_intensity: Optional[float] = None,
        fog_density: Optional[float] = None,
        fog_distance: Optional[float] = None,
        sun_altitude_angle: Optional[float] = None,
        sun_azimuth_angle: Optional[float] = None,
        condition: Optional[BaseCondition] = None,
        timing: TickTiming = TickTiming.PRE_TICK,
        *,
        rain: Optional[float] = None,
        tire_friction: Optional[float] = None,
        tire_friction_quantile: Optional[float] = None,
        visibility_m: Optional[float] = None,
        visibility_quantile: Optional[float] = None,
        label: str = "environment",
        once: bool = True,
    ) -> None:
        super().__init__(label=label, condition=condition, timing=timing, once=once)
        if rain is not None and precipitation is not None:
            # rain sets the precipitation: both would leave the probe reading a
            # different rain than the one asked for.
            raise ValueError(
                "EnvironmentAction: give rain or precipitation, not both "
                "(rain sets the precipitation)"
            )
        if tire_friction is not None and not tire_friction > 0.0:
            raise ValueError(
                f"EnvironmentAction: tire_friction must be positive, got {tire_friction}"
            )
        if tire_friction_quantile is not None and not (
            0.0 <= tire_friction_quantile <= 1.0
        ):
            raise ValueError(
                "EnvironmentAction: tire_friction_quantile must be within [0, 1], "
                f"got {tire_friction_quantile}"
            )
        if visibility_m is not None and not visibility_m > 0.0:
            raise ValueError(
                f"EnvironmentAction: visibility_m must be positive, got {visibility_m}"
            )
        if visibility_quantile is not None and not (0.0 <= visibility_quantile <= 1.0):
            raise ValueError(
                "EnvironmentAction: visibility_quantile must be within [0, 1], "
                f"got {visibility_quantile}"
            )
        self._tire_friction = tire_friction
        self._tire_friction_quantile = tire_friction_quantile
        self._visibility_m = visibility_m
        self._visibility_quantile = visibility_quantile
        settings: dict[str, float] = {}
        if rain is not None:
            settings.update(rain_weather(rain))
        for name, value in (
            ("cloudiness", cloudiness),
            ("precipitation", precipitation),
            ("precipitation_deposits", precipitation_deposits),
            ("wetness", wetness),
            ("wind_intensity", wind_intensity),
            ("fog_density", fog_density),
            ("fog_distance", fog_distance),
            ("sun_altitude_angle", sun_altitude_angle),
            ("sun_azimuth_angle", sun_azimuth_angle),
        ):
            if value is not None:
                settings[name] = value
        self._settings = settings

    @property
    def settings(self) -> dict[str, float]:
        """The values this action will apply, by attribute name."""
        return dict(self._settings)

    @property
    def sets_tire_friction(self) -> bool:
        """Whether the action sets the vehicles' grip: it names a tire
        friction or sets the road's water."""
        return (
            self._tire_friction is not None
            or self._tire_friction_quantile is not None
            or "wetness" in self._settings
            or "precipitation_deposits" in self._settings
        )

    @property
    def sets_visibility(self) -> bool:
        """Whether the action sets how far drivers see: it names a visibility
        or sets the fog, the rain or the sun."""
        return (
            self._visibility_m is not None
            or self._visibility_quantile is not None
            or "fog_density" in self._settings
            or "precipitation" in self._settings
            or "sun_altitude_angle" in self._settings
        )

    def _visibility(self, weather: "carla.WeatherParameters") -> float:
        """How far drivers see in *weather*, the weather the action leaves, m."""
        if self._visibility_m is not None:
            return self._visibility_m
        quantile = self._visibility_quantile
        low, high = weather_range("visibility", _weather_inputs(weather))
        return at_quantile(low, high, 0.5 if quantile is None else quantile)

    def execute(self, world: "carla.World") -> None:
        """Apply the settings on top of the world's current weather."""
        if (
            len(self._settings) == 0
            and not self.sets_tire_friction
            and not self.sets_visibility
        ):
            logger.warning(
                "EnvironmentAction '%s' sets nothing; the weather is unchanged",
                self.label,
            )
            return

        weather = world.get_weather()
        if len(self._settings) > 0:
            for name, value in self._settings.items():
                _set_weather_field(weather, name, value)
            world.set_weather(weather)
            logger.info(
                "EnvironmentAction '%s' applied %s",
                self.label,
                ", ".join(
                    f"{name}={value:g}" for name, value in self._settings.items()
                ),
            )
        if self.sets_tire_friction:
            if self._tire_friction is not None:
                apply_tire_friction(world, self._tire_friction)
            else:
                quantile = self._tire_friction_quantile
                apply_weather_grip(
                    world,
                    _weather_inputs(weather),
                    0.5 if quantile is None else quantile,
                )
        if self.sets_visibility:
            seen = self._visibility(weather)
            set_world_visibility(world.id, seen)
            logger.info("Drivers see %.0f m", seen)
