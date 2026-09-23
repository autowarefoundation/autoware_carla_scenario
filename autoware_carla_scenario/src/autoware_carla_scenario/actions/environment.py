"""Environment action: set the world's weather and sun position."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from ..conditions import BaseCondition
from .base import BaseAction, TickTiming

if TYPE_CHECKING:
    import carla

logger = logging.getLogger(__name__)

__all__ = ["EnvironmentAction"]

#: Constructor parameter -> the ``carla.WeatherParameters`` attribute it sets.
#: A mapping rather than a chain of ``if`` statements so that adding a knob is
#: one line and cannot forget either half.
_WEATHER_FIELDS: dict[str, str] = {
    "cloudiness": "cloudiness",
    "precipitation": "precipitation",
    "precipitation_deposits": "precipitation_deposits",
    "wetness": "wetness",
    "wind_intensity": "wind_intensity",
    "fog_density": "fog_density",
    "fog_distance": "fog_distance",
    "sun_altitude_angle": "sun_altitude_angle",
    "sun_azimuth_angle": "sun_azimuth_angle",
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
        self._settings: dict[str, float] = {
            name: value
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
            )
            if value is not None
        }

    @property
    def settings(self) -> dict[str, float]:
        """The values this action will apply, by attribute name."""
        return dict(self._settings)

    def execute(self, world: "carla.World") -> None:
        """Apply the settings on top of the world's current weather."""
        if not self._settings:
            logger.warning(
                "EnvironmentAction '%s' sets nothing; the weather is unchanged",
                self.label,
            )
            return

        weather = world.get_weather()
        for name, value in self._settings.items():
            setattr(weather, _WEATHER_FIELDS[name], value)
        world.set_weather(weather)
        logger.info(
            "EnvironmentAction '%s' applied %s",
            self.label,
            ", ".join(f"{name}={value:g}" for name, value in self._settings.items()),
        )
