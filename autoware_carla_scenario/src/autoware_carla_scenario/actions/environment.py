"""Environment action: set the world's weather and sun position, the road's grip
and how far drivers see.

A wet road holds a tyre less well than a dry one, and CARLA does not know it:
its weather is drawn, never driven on.  So the action scales every vehicle's
wheel friction (``friction_force_multiplier``) to the grip the road's water
leaves it (:func:`tire_friction_range`, by the weather rules of
:mod:`autoware_carla_scenario.weather_rules`), and
:func:`follow_tire_friction` gives a vehicle spawned later the same.

Nor is CARLA's weather ever seen through: the action records how far drivers
see (:mod:`autoware_carla_scenario.visibility`), for a traffic backend that
models their sight.

The action applies numbers: the CARLA fields, the visibility and the
conditions they render (the rainfall rate, the visibility through the fog,
the wind speed, the cloud cover), which it records for the ODD's probes
(:func:`world_rainfall_rate`, :func:`world_fog_visibility_range`,
:func:`world_wind_speed`, :func:`world_cloud_cover`).  What follows from the conditions is
worked out before the run, once
(:func:`~autoware_carla_scenario.weather_rules.resolve_environment`); only
the grip, which may differ by vehicle, is read off the rules here, from the
CARLA weather the action leaves and the conditions the run was set to.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Optional

from ..conditions.base import BaseCondition
from ..utils.vehicles import as_vehicle
from ..visibility import set_world_visibility, world_visibility
from ..entity_role import _role_name
from ..weather_inputs import cloud_cover_level, time_of_day_level
from ..weather_rules import (
    custom_rules_in_use,
    vehicle_at,
    vehicle_rules_in_use,
    weather_at,
    weather_range,
)
from .base import BaseAction, TickTiming

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

logger = logging.getLogger(__name__)

__all__ = [
    "EnvironmentAction",
    "apply_tire_friction",
    "apply_weather_grip",
    "follow_tire_friction",
    "forget_tire_friction",
    "forget_weather_conditions",
    "tire_friction",
    "tire_friction_range",
    "world_cloud_cover",
    "world_fog_visibility_range",
    "world_rainfall_rate",
    "world_wind_speed",
    "world_tire_friction",
]

#: The weather fields given as a share, 0-100: CARLA's own range for them.
_PERCENT_FIELDS: tuple[str, ...] = (
    "cloudiness",
    "precipitation",
    "precipitation_deposits",
    "wetness",
    "wind_intensity",
    "fog_density",
)
#: The arguments that must be greater than 0.
_POSITIVE_ARGUMENTS: tuple[str, ...] = (
    "tire_friction",
    "visibility_m",
    "fog_visibility_range_m",
)
#: The arguments that must not be negative.
_NON_NEGATIVE_ARGUMENTS: tuple[str, ...] = ("rainfall_rate_mm_h", "wind_speed_kmh")
#: The arguments that are quantiles, 0-1.
_QUANTILE_ARGUMENTS: tuple[str, ...] = ("tire_friction_quantile", "visibility_quantile")
#: The cloud covers *cloud_cover* takes: the taxonomy's cloudiness
#: (``odd.probes.CLOUD_COVERS``, which a test holds this to; not imported, so
#: the action does not load the ODD package).
_CLOUD_COVERS: tuple[str, ...] = ("clear", "partly_cloudy", "overcast")


def _refusal(name: str, value: float) -> str:
    """Why :class:`EnvironmentAction` refuses *value* for its argument *name*,
    or ``""`` when it takes it.

    Every value must be finite; the weather fields of :data:`_PERCENT_FIELDS`
    within [0, 100], *tire_friction*, *visibility_m* and
    *fog_visibility_range_m* positive, *rainfall_rate_mm_h* and
    *wind_speed_kmh* not negative, the quantiles within [0, 1].
    """
    if not math.isfinite(value):
        return f"must be finite, got {value}"
    if name in _PERCENT_FIELDS and not (0.0 <= value <= 100.0):
        return f"must be within [0, 100], got {value}"
    if name in _POSITIVE_ARGUMENTS and not value > 0.0:
        return f"must be positive, got {value}"
    if name in _NON_NEGATIVE_ARGUMENTS and not value >= 0.0:
        return f"must not be negative, got {value}"
    if name in _QUANTILE_ARGUMENTS and not (0.0 <= value <= 1.0):
        return f"must be within [0, 1], got {value}"
    return ""


def tire_friction_range(
    wetness: float, precipitation_deposits: float = 0.0
) -> tuple[float, float]:
    """The ``(low, high)`` share of the dry road's grip a road this wet leaves.

    The weather rules' ``tire_friction``
    (:mod:`autoware_carla_scenario.weather_rules`): by default, by how wet the
    road is -- the wetter of the two holds -- a dry road's around its own
    grip, a wet one's about three quarters of it, a flooded one's about half.

    Args:
        wetness: The weather's ``wetness``, 0-100.
        precipitation_deposits: Its standing water, 0-100.
    """
    return weather_range("tire_friction", _water(wetness, precipitation_deposits))


def tire_friction(
    wetness: float, precipitation_deposits: float = 0.0, quantile: float = 0.5
) -> float:
    """The share of the dry road's grip at *quantile* (0-1) of its
    distribution, which spans :func:`tire_friction_range`.

    0.5, the default, is its median: a dry road's is 1.0.
    """
    return weather_at(
        "tire_friction", _water(wetness, precipitation_deposits), quantile
    )


def _water(wetness: float, precipitation_deposits: float) -> dict[str, float]:
    """What the rules' grip reads: CARLA's wetness and standing water."""
    return {
        "carla_wetness": float(wetness),
        "carla_precipitation_deposits": float(precipitation_deposits),
    }


#: World (episode) id -> the rainfall rate (mm/h), the visibility through the
#: fog (m), the wind speed (km/h) and the cloud cover (a label) the run was
#: set to, where an action named them.
_RAINFALL: dict[int, float] = {}
_FOG_VISIBILITY: dict[int, float] = {}
_WIND_SPEED: dict[int, float] = {}
_CLOUD_COVER: dict[int, str] = {}


def _record_condition(store: dict[int, float], world_id: int, value: float) -> None:
    """Record *value* for the world; another world's is dropped."""
    for old in [w for w in store if w != world_id]:
        del store[old]
    store[world_id] = value


def _record_label(store: dict[int, str], world_id: int, label: str) -> None:
    """Record *label* for the world; another world's is dropped."""
    for old in [w for w in store if w != world_id]:
        del store[old]
    store[world_id] = label


def world_rainfall_rate(world_id: Optional[int]) -> Optional[float]:
    """The rainfall rate, mm/h, an action set the world *world_id* to, or
    ``None``: what the ODD's ``rainfall_rate_mm_h`` probe reads.  A run that
    sets CARLA's precipitation alone names no rainfall rate."""
    if world_id is None or world_id not in _RAINFALL:
        return None
    return _RAINFALL[world_id]


def world_fog_visibility_range(world_id: Optional[int]) -> Optional[float]:
    """The visibility through the fog, m, an action set the world *world_id*
    to, or ``None``: what the ODD's ``fog_visibility_range_m`` probe reads."""
    if world_id is None or world_id not in _FOG_VISIBILITY:
        return None
    return _FOG_VISIBILITY[world_id]


def world_wind_speed(world_id: Optional[int]) -> Optional[float]:
    """The wind speed, km/h, an action set the world *world_id* to, or
    ``None``: what the ODD's ``wind_speed_kmh`` probe reads."""
    if world_id is None or world_id not in _WIND_SPEED:
        return None
    return _WIND_SPEED[world_id]


def world_cloud_cover(world_id: Optional[int]) -> Optional[str]:
    """The cloud cover (``clear``, ``partly_cloudy``, ``overcast``) an action
    set the world *world_id* to, or ``None``: what the ODD's ``cloud_cover``
    probe reads.  The cloudiness CARLA renders may be more than it -- rain
    clouds the sky (the weather rules' ``carla_cloudiness``) -- but the
    condition the run was set to is the sky's.  A run that sets CARLA's
    cloudiness alone names no cloud cover."""
    if world_id is None or world_id not in _CLOUD_COVER:
        return None
    return _CLOUD_COVER[world_id]


def forget_weather_conditions() -> None:
    """Forget every world's conditions; the runner calls this for every run,
    so a run that names none does not read the last run's."""
    _RAINFALL.clear()
    _FOG_VISIBILITY.clear()
    _WIND_SPEED.clear()
    _CLOUD_COVER.clear()


#: World (episode) id -> the grip share last applied in it.
_TIRE_FACTORS: dict[int, float] = {}
#: World (episode) id -> vehicle id -> its wheels' friction before any was
#: applied: a later factor scales these, not what an earlier one left.
_TIRE_BASES: dict[int, dict[int, list[float]]] = {}
#: World (episode) id -> the ids of every actor already looked at, vehicle or
#: not, so following the world reads only the actors new since.
_TIRE_SEEN: dict[int, set[int]] = {}
#: World (episode) id -> the weather the grip was read off -- its numbers and
#: its labels -- and the quantile, when it was (:func:`apply_weather_grip`):
#: each vehicle's grip is then its own rules'
#: (:func:`~autoware_carla_scenario.weather_rules.vehicle_at`).
_TIRE_WEATHER: dict[int, tuple[dict[str, float], dict[str, str], float]] = {}
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
    inputs, labels, quantile = _TIRE_WEATHER[world_id]
    role = _role_name(actor)
    return vehicle_at(
        "tire_friction",
        inputs,
        quantile,
        str(actor.type_id),
        "" if role is None else role,
        labels,
    )


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
    world: "carla.World",
    inputs: dict[str, float],
    quantile: float,
    labels: Optional[dict[str, str]] = None,
) -> None:
    """Give every vehicle in *world* the grip the weather *inputs* (and
    *labels*, its categorical inputs: the time of day, the cloud cover)
    leave, at *quantile* (0-1) of its distribution: the run's rules', or,
    where they differ by vehicle (``environment.rules.vehicles``), each
    vehicle's own.

    Remembered for the world, as :func:`apply_tire_friction`'s factor is, so
    :func:`follow_tire_friction` gives a vehicle spawned later its own.
    """
    named: dict[str, str] = {}
    if labels is not None:
        named = dict(labels)
    share = weather_at("tire_friction", inputs, quantile, named)
    _set_grip(world, share, (dict(inputs), named, quantile))


def _set_grip(
    world: "carla.World",
    factor: float,
    weather: Optional[tuple[dict[str, float], dict[str, str], float]],
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


def forget_tire_friction(world: "carla.World") -> None:
    """Put back the grip the vehicles of *world* were spawned with, and forget
    every world's tire friction; the runner calls this for every run.

    A soft world reset keeps the episode, so without this a run that sets no
    grip would read (:func:`world_tire_friction`) and give the vehicles it
    spawns (:func:`follow_tire_friction`) the grip the last run set.  The
    vehicles that grip was applied to mostly went with that run, but not
    every one: an ego that attaches to an actor someone else spawned outlives
    the run, and a retried run gets no reset at all.  Such a vehicle gets its
    spawned grip back before its base is forgotten, so the next run's grip
    scales that rather than what the last one left.  Nothing is asked of the
    server when no grip was ever applied.
    """
    world_id = world.id
    if world_id in _TIRE_BASES and len(_TIRE_BASES[world_id]) > 0:
        bases = _TIRE_BASES[world_id]
        for actor in world.get_actors().filter("vehicle.*"):
            if actor.id not in bases:
                continue
            # One `except` per exception: Codon takes no tuple (docs/typecheck.md).
            try:
                _scale_wheels(bases, actor, 1.0)
            except RuntimeError:
                continue
            except AttributeError:
                continue
    _TIRE_FACTORS.clear()
    _TIRE_BASES.clear()
    _TIRE_SEEN.clear()
    # The ego's own grip and the weather each vehicle's was read off too:
    # world_tire_friction() reads the ego's first.
    _TIRE_WEATHER.clear()
    _TIRE_EGO.clear()


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


def _weather_inputs(
    world_id: int, weather: "carla.WeatherParameters"
) -> dict[str, float]:
    """The numbers the rules' grip reads in the world *world_id*, its weather
    *weather*: every CARLA field (``weather_rules.CARLA_QUANTITIES``), and
    the conditions the run was set to -- the rainfall rate, the visibility
    through the fog, the wind speed -- and how far its drivers see, where
    they were named (a cause not named reads the rules' default, as it does
    before the run, ``resolve_environment``).  So a ``tire_friction`` part
    reading a cause or another quantity reads the run's (#1938)."""
    inputs: dict[str, float] = {
        "carla_precipitation": float(weather.precipitation),
        "carla_cloudiness": float(weather.cloudiness),
        "carla_precipitation_deposits": float(weather.precipitation_deposits),
        "carla_wetness": float(weather.wetness),
        "carla_wind_intensity": float(weather.wind_intensity),
        "carla_fog_density": float(weather.fog_density),
        "carla_sun_altitude": float(weather.sun_altitude_angle),
        "carla_sun_azimuth": float(weather.sun_azimuth_angle),
    }
    if world_id in _RAINFALL:
        inputs["rainfall_rate"] = _RAINFALL[world_id]
    if world_id in _FOG_VISIBILITY:
        inputs["fog_visibility_range"] = _FOG_VISIBILITY[world_id]
    if world_id in _WIND_SPEED:
        inputs["wind_speed"] = _WIND_SPEED[world_id]
    seen = world_visibility(world_id)
    if seen is not None:
        inputs["visibility"] = seen
    return inputs


def _weather_labels(
    world_id: int, weather: "carla.WeatherParameters"
) -> dict[str, str]:
    """The labels the rules' grip reads in the world *world_id*: the time of
    day CARLA's sun renders and the cloud cover the run was set to, else the
    one CARLA's clouds render -- what the ODD's ``time_of_day`` and
    ``cloud_cover`` probes read."""
    sky = (
        _CLOUD_COVER[world_id]
        if world_id in _CLOUD_COVER
        else cloud_cover_level(float(weather.cloudiness))
    )
    return {
        "time_of_day": time_of_day_level(
            float(weather.sun_altitude_angle), float(weather.sun_azimuth_angle)
        ),
        "cloudiness": sky,
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

    **It applies numbers.**  The conditions a run is given -- the rainfall
    rate, the visibility through the fog, the time of day -- are rendered by
    CARLA's fields, and what those are follows from the conditions by the
    weather rules, worked out once before the run
    (:func:`~autoware_carla_scenario.weather_rules.resolve_environment`; the
    runner does it for a config's ``environment``, the sampler for the cases
    it draws)::

        EnvironmentAction(**resolve_environment({"rainfall_rate_mm_h": 5.0}))

    *rainfall_rate_mm_h*, *fog_visibility_range_m*, *wind_speed_kmh* and
    *cloud_cover* are recorded as what the run was set to, which the ODD's
    probes read (:func:`world_rainfall_rate`,
    :func:`world_fog_visibility_range`, :func:`world_wind_speed`,
    :func:`world_cloud_cover`); the time of day is read off CARLA's sun, and
    the cloud cover off CARLA's clouds when no action named one.

    The default phase is ``init``: most scenarios set the weather once, before
    anything moves.  Registering it on the tick loop instead is what makes the
    weather change *during* a run.

    **How far drivers see** is *visibility_m*, when given: the action records
    it (:func:`~autoware_carla_scenario.visibility.set_world_visibility`), and
    a traffic backend that models drivers' sight (``sumo``) holds its traffic
    to it, and the TrafficManager with ``follow_visibility: true``.

    **The road's grip follows the water on it.**  An action that sets the
    ``wetness`` or the standing water, or names a tire friction, scales every
    vehicle's wheel friction to the share of the dry road's grip the
    resulting weather leaves (:func:`tire_friction`): *tire_friction* if
    given, else the value at *tire_friction_quantile* of its distribution,
    which spans :func:`tire_friction_range` -- by default the quantile an
    earlier action in the run read the grip at, else its median, so a dry
    road keeps the dry grip -- read off each vehicle's rules where they
    differ by vehicle.  Under rules other than the default ones (a rule file
    of the run's, or rules by vehicle) every action does, whatever it sets,
    since their grip may differ from CARLA's on any road (#1937) -- but one
    that names no grip, quantile or water keeps a grip named outright
    earlier in the run.  The rules
    read every CARLA field of the weather it leaves and the conditions the
    run was set to -- the rainfall rate, the visibility through the fog, the
    wind speed, the cloud cover, the time of day CARLA's sun renders, how far
    drivers see -- where named (#1938).  Vehicles spawned later get it from
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
        rainfall_rate_mm_h: The rainfall rate the run is set to, mm/h:
            recorded, not applied.  Keyword-only.
        fog_visibility_range_m: The visibility through the fog the run is set
            to, metres: recorded, not applied.  Keyword-only.
        wind_speed_kmh: The wind speed the run is set to, km/h: recorded, not
            applied.  Keyword-only.
        cloud_cover: The cloud cover the run is set to, ``clear``,
            ``partly_cloudy`` or ``overcast``: recorded, not applied (the
            *cloudiness* that renders it may be more: rain clouds the sky).
            Keyword-only.
        tire_friction: The vehicles' grip, as a share of the dry road's
            (1.0): fixed, whatever the weather.  Keyword-only.
        tire_friction_quantile: The quantile of the distribution of grip the
            weather leaves (:func:`tire_friction_range`) the vehicles' is at,
            0 (its low end) to 1 (its high end); when the action sets
            the grip (:attr:`sets_tire_friction`) and this is not given,
            the one an earlier action in the run read it at, else 0.5.  *tire_friction* wins over it.  An ODD sweep
            draws it per case.  Keyword-only.
        visibility_m: How far drivers see, metres.  Keyword-only.
        condition: Trigger condition (see :class:`BaseCondition`).
        timing: Tick phase.
        label: Human-readable identifier.
        once: If ``True`` (default) the action fires at most once.

    Raises:
        ValueError: A numeric argument is not finite (NaN or infinite); a
            weather field given as 0-100 (*cloudiness*, *precipitation*,
            *precipitation_deposits*, *wetness*, *wind_intensity*,
            *fog_density*) is outside it; *tire_friction*, *visibility_m* or
            *fog_visibility_range_m* is not positive; *rainfall_rate_mm_h* or
            *wind_speed_kmh* is negative; *tire_friction_quantile* is outside
            [0, 1]; or *cloud_cover* is none of its labels.  The message names
            the argument.
    """

    _settings: dict[str, float]
    _rainfall_rate_mm_h: Optional[float]
    _fog_visibility_range_m: Optional[float]
    _wind_speed_kmh: Optional[float]
    _cloud_cover: Optional[str]
    _tire_friction: Optional[float]
    _tire_friction_quantile: Optional[float]
    _visibility_m: Optional[float]

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
        rainfall_rate_mm_h: Optional[float] = None,
        fog_visibility_range_m: Optional[float] = None,
        wind_speed_kmh: Optional[float] = None,
        cloud_cover: Optional[str] = None,
        tire_friction: Optional[float] = None,
        tire_friction_quantile: Optional[float] = None,
        visibility_m: Optional[float] = None,
        label: str = "environment",
        once: bool = True,
    ) -> None:
        super().__init__(label=label, condition=condition, timing=timing, once=once)
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
            ("rainfall_rate_mm_h", rainfall_rate_mm_h),
            ("fog_visibility_range_m", fog_visibility_range_m),
            ("wind_speed_kmh", wind_speed_kmh),
            ("tire_friction", tire_friction),
            ("tire_friction_quantile", tire_friction_quantile),
            ("visibility_m", visibility_m),
        ):
            if value is not None:
                refusal = _refusal(name, value)
                if refusal != "":
                    raise ValueError(f"EnvironmentAction: {name} {refusal}")
        self._rainfall_rate_mm_h = rainfall_rate_mm_h
        self._fog_visibility_range_m = fog_visibility_range_m
        # str(): Codon's Optional[str] `in` a tuple is always False.
        if cloud_cover is not None and str(cloud_cover) not in _CLOUD_COVERS:
            raise ValueError(
                f"EnvironmentAction: cloud_cover must be one of "
                f"{list(_CLOUD_COVERS)}, got {cloud_cover!r}"
            )
        self._wind_speed_kmh = wind_speed_kmh
        self._cloud_cover = cloud_cover
        self._tire_friction = tire_friction
        self._tire_friction_quantile = tire_friction_quantile
        self._visibility_m = visibility_m
        settings: dict[str, float] = {}
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

    def _names_grip(self) -> bool:
        """Whether the action names a tire friction or its quantile, or sets
        the road's water."""
        return (
            self._tire_friction is not None
            or self._tire_friction_quantile is not None
            or "wetness" in self._settings
            or "precipitation_deposits" in self._settings
        )

    @property
    def sets_tire_friction(self) -> bool:
        """Whether the action sets the vehicles' grip: it names a tire
        friction or sets the road's water, or the run follows rules other
        than the default ones
        (:func:`~autoware_carla_scenario.weather_rules.custom_rules_in_use`),
        whose grip may differ from CARLA's whatever the action sets."""
        return self._names_grip() or custom_rules_in_use()

    @property
    def sets_visibility(self) -> bool:
        """Whether the action sets how far drivers see: it names a visibility."""
        return self._visibility_m is not None

    @property
    def sets_conditions(self) -> bool:
        """Whether the action names a condition the run is set to."""
        return (
            self._rainfall_rate_mm_h is not None
            or self._fog_visibility_range_m is not None
            or self._wind_speed_kmh is not None
            or self._cloud_cover is not None
        )

    def execute(self, world: "carla.World") -> None:
        """Apply the settings on top of the world's current weather."""
        if (
            len(self._settings) == 0
            and not self.sets_tire_friction
            and not self.sets_visibility
            and not self.sets_conditions
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
        rainfall = self._rainfall_rate_mm_h
        if rainfall is not None:
            _record_condition(_RAINFALL, world.id, rainfall)
        fog = self._fog_visibility_range_m
        if fog is not None:
            _record_condition(_FOG_VISIBILITY, world.id, fog)
        wind = self._wind_speed_kmh
        if wind is not None:
            _record_condition(_WIND_SPEED, world.id, wind)
        sky = self._cloud_cover
        if sky is not None:
            _record_label(_CLOUD_COVER, world.id, sky)
        # Before the grip, which may read it.
        seen = self._visibility_m
        if seen is not None:
            set_world_visibility(world.id, seen)
            logger.info("Drivers see %.0f m", seen)
        if self.sets_tire_friction:
            world_id = world.id
            quantile = self._tire_friction_quantile
            if quantile is None and world_id in _TIRE_WEATHER:
                # The quantile the run's grip is at holds: a later action
                # reads the grip off its weather at it (a case draws one).
                quantile = _TIRE_WEATHER[world_id][2]
            if self._tire_friction is not None:
                apply_tire_friction(world, self._tire_friction)
            elif (
                quantile is None
                and world_id in _TIRE_FACTORS
                and not self._names_grip()
            ):
                # A grip named outright holds against an action that sets
                # the grip only because the run's rules are its own.
                pass
            else:
                apply_weather_grip(
                    world,
                    _weather_inputs(world_id, weather),
                    0.5 if quantile is None else quantile,
                    _weather_labels(world_id, weather),
                )
