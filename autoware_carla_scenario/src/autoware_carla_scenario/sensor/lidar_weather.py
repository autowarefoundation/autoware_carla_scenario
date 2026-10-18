"""What rain and fog do to a LiDAR sweep: CARLA's ray-cast LiDAR ignores the weather.

The server casts every ray through clear air whatever ``carla.WeatherParameters``
says, so a sweep in heavy rain or thick fog is a dry day's.  This adds to a raw
sweep, from the weather's ``precipitation`` and ``fog_density`` (0-100, what the
ODD's ``rain`` and ``fog`` knobs, ``environment.rain`` and
``environment.fog_density``, set), the three things a real sensor suffers:

* **attenuation** -- the beam is scattered both ways, so a return at range ``r``
  keeps ``T = exp(-2 * alpha * r)`` of its power: its intensity is scaled by
  ``T`` and it survives with probability ``T``, so far returns thin out first.
  Fog only scatters beyond ``fog_distance``, where CARLA starts drawing it;
* **range noise** -- each surviving return is moved along its ray by a normal
  error growing with the distance it travelled through the weather;
* **early returns** -- a beam stopped by a drop, or back-scattered by the fog,
  returns from a few metres out with little intensity: clutter around the
  sensor, along the real beam directions.

Each grows linearly with its weather intensity and is scaled by the sensor's
``weather_noise`` (0 turns it off, 1 is this model as tuned here).  Clear
weather leaves a sweep untouched.

How thick a ``fog_density`` looks is the server's: CARLA multiplies it by the
console variable ``carla.Weather.FogDensityScale`` (:data:`CARLA_FOG_DENSITY_SCALE`
unless the server was started with another).  The fog effects scale with it
too, so the LiDAR sees the fog the cameras do: as tuned here at
:data:`FOG_REFERENCE_SCALE`, the scale that draws fog a camera sees, and a tenth
as thick at CARLA's default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "CARLA_FOG_DENSITY_SCALE",
    "FOG_REFERENCE_SCALE",
    "LidarWeather",
    "apply_weather",
]

#: ``carla.Weather.FogDensityScale`` on a server started without it.
CARLA_FOG_DENSITY_SCALE: float = 0.001
#: The ``carla.Weather.FogDensityScale`` the fog effects below are tuned at.
FOG_REFERENCE_SCALE: float = 0.01

#: Two-way extinction coefficients, per metre, at intensity 100.  Heavy rain
#: keeps about a seventh of the power of a return from 100 m; thick fog (at
#: :data:`FOG_REFERENCE_SCALE`), a few percent from 30 m beyond where it starts.
_RAIN_EXTINCTION_PER_M: float = 0.01
_FOG_EXTINCTION_PER_M: float = 0.06
#: Standard deviation of the range error, as a fraction of the distance
#: travelled through the weather, at 100.
_RAIN_RANGE_NOISE: float = 0.01
_FOG_RANGE_NOISE: float = 0.005
#: Probability of a beam returning early, at 100.
_RAIN_EARLY_RETURN: float = 0.02
_FOG_EARLY_RETURN: float = 0.05
#: Mean distance of an early return beyond where its weather starts, in metres.
_EARLY_RETURN_MEAN_M: float = 4.0
#: Nearest early return, in metres: closer, a real sensor's own blind zone.
_EARLY_RETURN_MIN_M: float = 0.5
#: An early return's intensity, at most this fraction of the beam's own.
_EARLY_RETURN_INTENSITY: float = 0.1


@dataclass(frozen=True)
class LidarWeather:
    """The weather a sweep is degraded by, in CARLA's units.

    Attributes:
        precipitation: Rain intensity, 0-100.
        fog_density: Fog intensity, 0-100.
        fog_distance: Metres from the sensor before the fog starts.
        fog_density_scale: The server's ``carla.Weather.FogDensityScale``.
    """

    precipitation: float = 0.0
    fog_density: float = 0.0
    fog_distance: float = 0.0
    fog_density_scale: float = CARLA_FOG_DENSITY_SCALE

    @property
    def clear(self) -> bool:
        """Whether neither rain nor fog falls."""
        return self.precipitation <= 0.0 and self.fog_density <= 0.0


def _fraction(intensity: float, gain: float) -> float:
    return min(max(intensity, 0.0), 100.0) / 100.0 * gain


def apply_weather(
    raw: np.ndarray[tuple[int, int], np.dtype[np.float32]],
    weather: LidarWeather,
    gain: float,
    seed: int,
    frame: int,
) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
    """*raw* (``[N, 4]`` x, y, z, intensity, in the sensor frame) as seen through *weather*.

    A new array: fewer points (lost to attenuation), the rest moved along their
    rays, some returned early.  *raw* itself when *gain* is 0 or the weather
    clear.  The noise is drawn from a generator seeded with *seed* and the
    sweep's *frame* (both non-negative), so a sweep's noise repeats.
    """
    count = raw.shape[0]
    if gain <= 0.0 or weather.clear or count == 0:
        return raw
    rain = _fraction(weather.precipitation, gain)
    fog = (
        _fraction(weather.fog_density, gain)
        * weather.fog_density_scale
        / FOG_REFERENCE_SCALE
    )
    fog_start = max(weather.fog_distance, 0.0)
    rng = np.random.default_rng([seed, frame])

    xyz = raw[:, :3]
    ranges = np.sqrt(np.sum(xyz * xyz, axis=1)).astype(np.float64)
    extinction = ranges * (rain * _RAIN_EXTINCTION_PER_M)
    sigma = ranges * (rain * _RAIN_RANGE_NOISE)
    if fog > 0.0:
        in_fog = np.maximum(ranges - fog_start, 0.0)
        extinction = extinction + in_fog * (fog * _FOG_EXTINCTION_PER_M)
        sigma = sigma + in_fog * (fog * _FOG_RANGE_NOISE)
    transmittance = np.exp(extinction * -2.0)

    # A rain drop can stop a beam anywhere; fog only where it is drawn.  Only
    # the few beams drawn early get an early range.
    p_rain = rain * _RAIN_EARLY_RETURN
    p_early = p_rain + fog * _FOG_EARLY_RETURN
    draw = rng.random(count)
    candidates = np.flatnonzero(draw < p_early)
    early_at = (
        np.where(draw[candidates] >= p_rain, fog_start, 0.0)
        + _EARLY_RETURN_MIN_M
        + rng.exponential(_EARLY_RETURN_MEAN_M, candidates.shape[0])
    )
    hit = early_at < ranges[candidates]
    early = np.zeros(count, dtype=np.bool_)
    early[candidates[hit]] = True
    early_range = np.zeros(count, dtype=np.float64)
    early_range[candidates[hit]] = early_at[hit]

    kept = np.flatnonzero(early | (rng.random(count) < transmittance))
    kept_early = early[kept]
    kept_ranges = ranges[kept]
    noisy = np.where(
        kept_early,
        early_range[kept],
        np.maximum(kept_ranges + rng.standard_normal(kept.shape[0]) * sigma[kept], 0.0),
    )
    intensity_scale = np.where(
        kept_early,
        rng.random(kept.shape[0]) * _EARLY_RETURN_INTENSITY,
        transmittance[kept],
    )
    # A return at the sensor itself has neither noise nor an early range: it
    # stays at the origin whatever the divisor.
    scale = noisy / np.maximum(kept_ranges, 1e-9)
    out = raw[kept]
    out[:, :3] = (out[:, :3] * scale[:, None]).astype(np.float32)
    out[:, 3] = (out[:, 3] * intensity_scale).astype(np.float32)
    return out
