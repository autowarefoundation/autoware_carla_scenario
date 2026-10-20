"""A mesoscopic traffic model a backend can run its far traffic on.

The model is the queue model of SUMO's own mesoscopic simulation (Eissfeldt's,
as in ``sumo --mesosim``), cut down to what far traffic needs: a road is a
queue, a vehicle that enters it may leave it once it has driven its length at
the road's speed (or its own, or the speed it was told to keep), and then only

* first in, first out,
* one per ``headway_s / lanes`` seconds -- the road's capacity,
* when the road it goes on to is not full -- ``lanes * length / jam_spacing_m``
  vehicles -- so a queue spills back road by road, and
* when the junction lets it through (an optional *gate*: a red light).

What is left out is everything about lanes: no lane changes, no overtaking
within a road, no car-following.  Nobody sees a mesoscopic vehicle, so none of
that shows; what does show, when it comes back, is when it arrives and how
long the queue it arrives in is, and those the model keeps.

A step costs a look at the vehicle at the head of every road that has one,
not a look at every vehicle: a road whose head is still driving costs one
comparison.  Nothing here knows a simulator.  A backend builds the roads from
its own network (:class:`MesoLink`), moves a vehicle in when it becomes
mesoscopic (:meth:`MesoscopicTraffic.add`), reads where it is to decide when to
bring it back (:meth:`MesoscopicTraffic.position`), and takes it out again
(:meth:`MesoscopicTraffic.remove`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterator, Mapping, Optional, Sequence

__all__ = ["MesoLink", "MesoVehicle", "MesoscopicTraffic"]

#: A speed slower than this is a vehicle told to stand, m/s.
_STANDING_MS = 0.01


@dataclass(frozen=True)
class MesoLink:
    """A road of the mesoscopic model: one direction of travel.

    Attributes:
        id: The backend's name for it (a SUMO edge id, say).
        length: Metres.
        lanes: Lanes the model's vehicles may use; at least one.
        speed: The speed limit, m/s.
    """

    id: str
    length: float
    lanes: int
    speed: float


@dataclass
class MesoVehicle:
    """A vehicle the mesoscopic model holds.

    Attributes:
        id: The backend's name for it.
        route: The roads it drives, in order.
        index: Which of *route* it is on.
        entered_s: When it entered the road (or was put on it, or last told
            a new speed), simulated seconds.
        start_m: How far along the road it was then.
        exit_s: The earliest it may leave the road: when it reaches its end.
        max_speed: Its own top speed, m/s.
        length: Metres, for a backend bringing it back.
        type_id: The backend's vehicle type, for the same.
        speed_cap: The speed it was told to keep, m/s; ``None`` for none.
    """

    id: str
    route: list[str]
    index: int
    entered_s: float
    start_m: float
    exit_s: float
    max_speed: float
    length: float = 5.0
    type_id: str = ""
    speed_cap: Optional[float] = None

    @property
    def link(self) -> str:
        """The road it is on."""
        return self.route[self.index]

    @property
    def remaining_route(self) -> list[str]:
        """The roads it has still to drive, the one it is on first."""
        return self.route[self.index :]


@dataclass
class _Queue:
    """The vehicles on one road, the one nearest its end first."""

    vehicles: list[str] = field(default_factory=list)
    #: When the next vehicle may leave: the road's capacity.
    next_exit_s: float = -math.inf


class MesoscopicTraffic:
    """Vehicles as queues on roads; see the module docstring.

    Args:
        links: Every road a vehicle may drive, by id.  A vehicle whose route
            goes on to a road not here arrives where that road starts.
        headway_s: Time gap between two vehicles leaving a road, per lane.
        jam_spacing_m: Road length one vehicle takes up in a queue.
        gate: ``gate(from_link, to_link)`` -- whether a vehicle may cross from
            one road to the next now; ``None`` lets every one through.
    """

    # Declared for the static check (docs/typecheck.md): Codon types an
    # attribute from its declaration; to Python it is a bare annotation.
    _links: dict[str, MesoLink]
    _headway_s: float
    _jam_spacing_m: float
    _gate: Optional[Callable[[str, str], bool]]
    _vehicles: dict[str, MesoVehicle]
    _queues: dict[str, _Queue]

    def __init__(
        self,
        links: Mapping[str, MesoLink],
        *,
        headway_s: float = 1.5,
        jam_spacing_m: float = 7.5,
        gate: Optional[Callable[[str, str], bool]] = None,
    ) -> None:
        self._links = dict(links)
        self._headway_s = headway_s
        self._jam_spacing_m = jam_spacing_m
        self._gate = gate
        self._vehicles: dict[str, MesoVehicle] = {}
        self._queues: dict[str, _Queue] = {}

    # ------------------------------------------------------------------
    # The vehicles
    # ------------------------------------------------------------------

    def __contains__(self, vehicle_id: str) -> bool:
        return vehicle_id in self._vehicles

    def __len__(self) -> int:
        return len(self._vehicles)

    def __iter__(self) -> Iterator[str]:
        return iter(list(self._vehicles))

    def has_link(self, link_id: str) -> bool:
        """Whether *link_id* is a road of the model."""
        return link_id in self._links

    def link(self, link_id: str) -> MesoLink:
        """The road *link_id*."""
        return self._links[link_id]

    def vehicle(self, vehicle_id: str) -> MesoVehicle:
        """The vehicle *vehicle_id*; ``KeyError`` if the model has none."""
        return self._vehicles[vehicle_id]

    def add(
        self,
        vehicle_id: str,
        route: Sequence[str],
        *,
        now: float,
        offset_m: float = 0.0,
        max_speed: float = math.inf,
        length: float = 5.0,
        type_id: str = "",
        speed_cap: Optional[float] = None,
    ) -> MesoVehicle:
        """Put a vehicle on the first road of *route*, *offset_m* along it.

        It goes to the back of that road's queue whatever its offset: the
        vehicles already on the road are ahead of it in the model, which is
        right for the one that left a microscopic simulation last and close
        enough for the rest.

        Raises:
            ValueError: The route is empty, or starts on a road the model does
                not have, or the vehicle is in the model already.
        """
        if vehicle_id in self._vehicles:
            raise ValueError(f"{vehicle_id} is in the mesoscopic model already")
        if not route or route[0] not in self._links:
            raise ValueError(f"{vehicle_id}: no mesoscopic road {list(route)[:1]}")
        link = self._links[route[0]]
        vehicle = MesoVehicle(
            id=vehicle_id,
            route=list(route),
            index=0,
            entered_s=now,
            start_m=min(max(offset_m, 0.0), link.length),
            exit_s=now,
            max_speed=max_speed,
            length=length,
            type_id=type_id,
            speed_cap=speed_cap,
        )
        vehicle.exit_s = self._exit_time(vehicle, link, now)
        self._vehicles[vehicle_id] = vehicle
        self._queue(link.id).vehicles.append(vehicle_id)
        return vehicle

    def remove(self, vehicle_id: str) -> Optional[MesoVehicle]:
        """Take *vehicle_id* out of the model; ``None`` if it was not in it."""
        if vehicle_id not in self._vehicles:
            return None
        vehicle = self._vehicles.pop(vehicle_id)
        queue = self._queues.get(vehicle.link)
        if queue is not None and vehicle_id in queue.vehicles:
            queue.vehicles.remove(vehicle_id)
        return vehicle

    def clear(self) -> None:
        """Take every vehicle out."""
        self._vehicles = {}
        self._queues = {}

    # ------------------------------------------------------------------
    # Telling a vehicle what to do
    # ------------------------------------------------------------------

    def set_speed_cap(
        self, vehicle_id: str, speed: Optional[float], now: float
    ) -> None:
        """Hold *vehicle_id* to *speed* (m/s) from where it is now; ``None``
        lets it drive at the road's speed again."""
        vehicle = self._vehicles[vehicle_id]
        link = self._links[vehicle.link]
        vehicle.start_m = self.position(vehicle_id, now)[1]
        vehicle.entered_s = now
        vehicle.speed_cap = speed
        vehicle.exit_s = self._exit_time(vehicle, link, now)

    def set_route(self, vehicle_id: str, route: Sequence[str]) -> None:
        """Drive *route* from now on; it starts with the road the vehicle is on."""
        vehicle = self._vehicles[vehicle_id]
        if not route or route[0] != vehicle.link:
            raise ValueError(
                f"{vehicle_id}: a new route must start on {vehicle.link} "
                f"(got {list(route)[:1]})"
            )
        vehicle.route = list(route)
        vehicle.index = 0

    # ------------------------------------------------------------------
    # Where a vehicle is
    # ------------------------------------------------------------------

    def position(self, vehicle_id: str, now: float) -> tuple[str, float]:
        """The road *vehicle_id* is on and how far along it, metres.

        Where it would be driving freely, but no further forward than its
        place in the queue: the n-th vehicle of a road's queue is at most
        ``n // lanes`` jam spacings short of the road's end.
        """
        vehicle = self._vehicles[vehicle_id]
        link = self._links[vehicle.link]
        speed = self._free_speed(vehicle, link)
        free = vehicle.start_m + speed * max(now - vehicle.entered_s, 0.0)
        place = self._queue(link.id).vehicles.index(vehicle_id) // max(link.lanes, 1)
        limit = link.length - place * self._jam_spacing_m
        return link.id, max(min(free, limit, link.length), 0.0)

    def speed(self, vehicle_id: str, now: float) -> float:
        """How fast *vehicle_id* is going now, m/s: its free speed while it
        drives, nothing once it waits in a queue or at the road's end."""
        vehicle = self._vehicles[vehicle_id]
        link = self._links[vehicle.link]
        if now >= vehicle.exit_s:
            return 0.0
        offset = self.position(vehicle_id, now)[1]
        free = vehicle.start_m + self._free_speed(vehicle, link) * (
            now - vehicle.entered_s
        )
        return self._free_speed(vehicle, link) if offset >= free - 1e-6 else 0.0

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def step(self, now: float) -> list[MesoVehicle]:
        """Move every vehicle that may leave its road by *now* on to the next.

        Returns:
            The vehicles that reached the end of their route, which are out of
            the model now.
        """
        arrived: list[MesoVehicle] = []
        for link_id in [k for k, q in self._queues.items() if q.vehicles]:
            link = self._links[link_id]
            queue = self._queues[link_id]
            while queue.vehicles:
                vehicle = self._vehicles[queue.vehicles[0]]
                if now < vehicle.exit_s or now < queue.next_exit_s:
                    break
                if vehicle.index + 1 >= len(vehicle.route) or not self.has_link(
                    vehicle.route[vehicle.index + 1]
                ):
                    queue.vehicles.pop(0)
                    del self._vehicles[vehicle.id]
                    arrived.append(vehicle)
                    continue
                to_link = self._links[vehicle.route[vehicle.index + 1]]
                if self._gate is not None and not self._gate(link_id, to_link.id):
                    break
                if len(self._queue(to_link.id).vehicles) >= self._capacity(to_link):
                    break
                queue.vehicles.pop(0)
                queue.next_exit_s = now + self._headway_s / max(link.lanes, 1)
                vehicle.index += 1
                vehicle.entered_s = now
                vehicle.start_m = 0.0
                vehicle.exit_s = self._exit_time(vehicle, to_link, now)
                self._queue(to_link.id).vehicles.append(vehicle.id)
        return arrived

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _queue(self, link_id: str) -> _Queue:
        queue = self._queues.get(link_id)
        if queue is None:
            queue = _Queue()
            self._queues[link_id] = queue
        return queue

    def _capacity(self, link: MesoLink) -> int:
        """How many vehicles *link* holds; at least one, so none is closed."""
        return max(int(link.lanes * link.length / self._jam_spacing_m), 1)

    @staticmethod
    def _free_speed(vehicle: MesoVehicle, link: MesoLink) -> float:
        speed = min(link.speed, vehicle.max_speed)
        if vehicle.speed_cap is not None:
            speed = min(speed, vehicle.speed_cap)
        return max(speed, 0.0)

    def _exit_time(self, vehicle: MesoVehicle, link: MesoLink, now: float) -> float:
        speed = self._free_speed(vehicle, link)
        if speed < _STANDING_MS:
            return math.inf
        return now + max(link.length - vehicle.start_m, 0.0) / speed
