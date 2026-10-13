"""Environment action: set the world's weather and sun position."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from ..conditions.base import BaseCondition
from .base import BaseAction, TickTiming

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

logger = logging.getLogger(__name__)

__all__ = ["EnvironmentAction", "rain_weather"]

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
        condition: Trigger condition (see :class:`BaseCondition`).
        timing: Tick phase.
        label: Human-readable identifier.
        once: If ``True`` (default) the action fires at most once.
    """

    _settings: dict[str, float]

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

    def execute(self, world: "carla.World") -> None:
        """Apply the settings on top of the world's current weather."""
        if len(self._settings) == 0:
            logger.warning(
                "EnvironmentAction '%s' sets nothing; the weather is unchanged",
                self.label,
            )
            return

        weather = world.get_weather()
        for name, value in self._settings.items():
            _set_weather_field(weather, name, value)
        world.set_weather(weather)
        logger.info(
            "EnvironmentAction '%s' applied %s",
            self.label,
            ", ".join(f"{name}={value:g}" for name, value in self._settings.items()),
        )
