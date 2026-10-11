"""Traffic light lookups in an OpenDRIVE road network.

Pure XML reading: the caller supplies the ``RoadNetwork``.  Reaching
:class:`~..coordinate.map_manager.MapManager` for it instead would put this
module on top of ``coordinate``, which is the package that owns the singleton
and the CARLA conversions -- and ``coordinate`` imports this one.

:mod:`..coordinate.traffic_light` wraps these for the ambient run, where the
network is the singleton's.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import typesafe_carla.carla as carla
    from pyxodr.road_objects.network import RoadNetwork


def traffic_light_state_name(state: "carla.TrafficLightState") -> str:
    """Return the member name ("Red", "Green", ...) of a traffic light state.

    typesafe_carla's ``TrafficLight.get_state()`` and ``.state`` give a plain
    int, whose ``str()`` is its value ("2"), so the name is looked up from the
    value; an enum member (``carla.TrafficLightState.Green``) works too.
    Anything else is ``"Unknown"``.
    """
    import typesafe_carla.carla as carla  # noqa: PLC0415 -- this helper is CARLA-side by definition

    try:
        value = int(state)
    except TypeError:
        return "Unknown"
    except ValueError:
        return "Unknown"
    # The member of that value, by name: ``TrafficLightState(value).name``
    # without calling the enumeration, which typesafe_carla's Codon library
    # has as a value, not a type (docs/typecheck.md).
    for name, member in carla.TrafficLightState.names.items():
        if int(member) == value:
            return name
    return "Unknown"


def lanelet2_traffic_light_id_to_opendrive_controller_id(
    road_network: "RoadNetwork",
    lanelet2_tl_id: int,
) -> Optional[int]:
    """Return the OpenDRIVE controller ID for a Lanelet2 traffic light ID.

    The mapping is derived from the ``<controller name="Controller_TL_{lanelet2_id}">``
    naming convention used during Lanelet2-to-OpenDRIVE conversion.

    Args:
        road_network: The loaded OpenDRIVE road network to read.
        lanelet2_tl_id: Lanelet2 regulatory element ID of the traffic light.

    Returns:
        OpenDRIVE controller ID, or ``None`` if no matching controller is found.
    """
    root = road_network.root
    expected_name = f"Controller_TL_{lanelet2_tl_id}"

    for ctrl_elem in root.iter("controller"):
        if ctrl_elem.get("name") == expected_name:
            return int(ctrl_elem.get("id"))
    return None


def get_signal_ids_for_controller(
    road_network: "RoadNetwork", controller_id: int
) -> list[str]:
    """Return the signal IDs controlled by an OpenDRIVE controller.

    Parses ``<control signalId="...">`` children of the ``<controller>``
    element whose ``id`` matches *controller_id*.

    Args:
        road_network: The loaded OpenDRIVE road network to read.
        controller_id: The OpenDRIVE controller ID to look up.
    """
    root = road_network.root
    for ctrl_elem in root.iter("controller"):
        if ctrl_elem.get("id") == str(controller_id):
            signal_ids: list[str] = []
            for c in ctrl_elem.iter("control"):
                signal_id = c.get("signalId")
                if signal_id is not None:
                    signal_ids.append(signal_id)
            return signal_ids
    return []
