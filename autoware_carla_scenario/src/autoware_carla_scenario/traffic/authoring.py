"""Traffic parameters out of a scenario document's flat fields.

The scenario editor writes a :class:`~autoware_carla_scenario.TrafficParametersAction`
as one field per number -- ``speed_factor_mean``, ``speed_factor_std``, ... --
which is what a form can show.  These assemble the parameter objects the
action takes from them; the editor's spec names them as the action's built
arguments (:mod:`autoware_carla_scenario.authoring.registry`).
"""

from __future__ import annotations

from typing import Optional

from .parameters import PedestrianParameters, Spread, VehicleParameters

__all__ = ["authored_pedestrian_parameters", "authored_vehicle_parameters"]


def _spread(mean: Optional[float], std: Optional[float], name: str) -> Optional[Spread]:
    """``mean`` and ``std`` as a spread; ``None`` when no mean is given.

    Raises:
        ValueError: If a std is given without a mean.
    """
    if mean is None:
        if std is not None:
            raise ValueError(f"{name}: a std needs a mean")
        return None
    return Spread(float(mean), 0.0 if std is None else float(std))


def authored_vehicle_parameters(
    speed_factor_mean: Optional[float] = None,
    speed_factor_std: Optional[float] = None,
    min_gap_m_mean: Optional[float] = None,
    min_gap_m_std: Optional[float] = None,
    lane_changes: Optional[bool] = None,
    ignore_lights_mean: Optional[float] = None,
    ignore_lights_std: Optional[float] = None,
    ignore_signs_mean: Optional[float] = None,
    ignore_signs_std: Optional[float] = None,
    ignore_vehicles_mean: Optional[float] = None,
    ignore_vehicles_std: Optional[float] = None,
    ignore_walkers_mean: Optional[float] = None,
    ignore_walkers_std: Optional[float] = None,
    random_lane_changes_mean: Optional[float] = None,
    random_lane_changes_std: Optional[float] = None,
    accel_mean: Optional[float] = None,
    accel_std: Optional[float] = None,
    decel_mean: Optional[float] = None,
    decel_std: Optional[float] = None,
    reaction_time_s_mean: Optional[float] = None,
    reaction_time_s_std: Optional[float] = None,
    imperfection_mean: Optional[float] = None,
    imperfection_std: Optional[float] = None,
    max_lateral_speed_mps_mean: Optional[float] = None,
    max_lateral_speed_mps_std: Optional[float] = None,
    max_lateral_accel_mean: Optional[float] = None,
    max_lateral_accel_std: Optional[float] = None,
    follow_visibility: Optional[bool] = None,
) -> Optional[VehicleParameters]:
    """The vehicle parameters the fields set; ``None`` when they set none."""
    parameters = VehicleParameters(
        speed_factor=_spread(speed_factor_mean, speed_factor_std, "speed_factor"),
        min_gap_m=_spread(min_gap_m_mean, min_gap_m_std, "min_gap_m"),
        lane_changes=lane_changes,
        ignore_lights=_spread(ignore_lights_mean, ignore_lights_std, "ignore_lights"),
        ignore_signs=_spread(ignore_signs_mean, ignore_signs_std, "ignore_signs"),
        ignore_vehicles=_spread(
            ignore_vehicles_mean, ignore_vehicles_std, "ignore_vehicles"
        ),
        ignore_walkers=_spread(
            ignore_walkers_mean, ignore_walkers_std, "ignore_walkers"
        ),
        random_lane_changes=_spread(
            random_lane_changes_mean, random_lane_changes_std, "random_lane_changes"
        ),
        accel=_spread(accel_mean, accel_std, "accel"),
        decel=_spread(decel_mean, decel_std, "decel"),
        reaction_time_s=_spread(
            reaction_time_s_mean, reaction_time_s_std, "reaction_time_s"
        ),
        imperfection=_spread(imperfection_mean, imperfection_std, "imperfection"),
        max_lateral_speed_mps=_spread(
            max_lateral_speed_mps_mean,
            max_lateral_speed_mps_std,
            "max_lateral_speed_mps",
        ),
        max_lateral_accel=_spread(
            max_lateral_accel_mean, max_lateral_accel_std, "max_lateral_accel"
        ),
        follow_visibility=follow_visibility,
    )
    if not parameters.set_fields():
        return None
    return parameters


def authored_pedestrian_parameters(
    walking_speed_mps_mean: Optional[float] = None,
    walking_speed_mps_std: Optional[float] = None,
    pedestrian_min_gap_m_mean: Optional[float] = None,
    pedestrian_min_gap_m_std: Optional[float] = None,
    red_light_violation_s_mean: Optional[float] = None,
    red_light_violation_s_std: Optional[float] = None,
) -> Optional[PedestrianParameters]:
    """The pedestrian parameters the fields set; ``None`` when they set none.

    The gap's fields are ``pedestrian_min_gap_m_*``, apart from the vehicles'
    ``min_gap_m_*`` on the same form.
    """
    parameters = PedestrianParameters(
        walking_speed_mps=_spread(
            walking_speed_mps_mean, walking_speed_mps_std, "walking_speed_mps"
        ),
        min_gap_m=_spread(
            pedestrian_min_gap_m_mean, pedestrian_min_gap_m_std, "min_gap_m"
        ),
        red_light_violation_s=_spread(
            red_light_violation_s_mean,
            red_light_violation_s_std,
            "red_light_violation_s",
        ),
    )
    if not parameters.set_fields():
        return None
    return parameters
