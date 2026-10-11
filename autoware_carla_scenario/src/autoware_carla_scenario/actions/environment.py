"""Environment action: set the world's weather and sun position."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from ..conditions.base import BaseCondition
from .base import BaseAction, TickTiming

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

logger = logging.getLogger(__name__)

__all__ = ["EnvironmentAction"]


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
        label: str = "environment",
        once: bool = True,
    ) -> None:
        super().__init__(label=label, condition=condition, timing=timing, once=once)
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
