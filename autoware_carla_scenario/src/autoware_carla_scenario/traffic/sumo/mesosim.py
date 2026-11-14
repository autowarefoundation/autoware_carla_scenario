"""Far traffic in SUMO's own mesoscopic simulation (``sumo --mesosim``).

The ``sumo`` backend's far traffic runs in a mesoscopic model
(:mod:`..fidelity`); this is the model it runs in by default
(``mesoscopic.engine: sumo``): a second SUMO, on the same network, started
with ``--mesosim``, whose vehicles are the ones the backend took out of its
microscopic SUMO.  It has the queue model's interface
(:class:`~..mesoscopic.MesoscopicTraffic`), so the backend moves a vehicle
between the two levels the same way whichever model holds the far traffic;
what changes is what drives it there -- Eissfeldt's model as SUMO has it,
with SUMO's junction model and its signals, rather than the cut-down queue
model of :mod:`..mesoscopic`.

libsumo runs one simulation per process, and the microscopic SUMO is that one
when libsumo is installed, so this one is always a TraCI connection to a SUMO
process of its own, kept apart from TraCI's module-level connection (never
switched to), which the microscopic SUMO uses when libsumo is missing.

What SUMO's mesoscopic model does not do, this module does around it:

* **Where a vehicle is.** SUMO puts a mesoscopic vehicle at the start of the
  segment it is in (``--meso-edgelength`` long); the model here moves it on
  from there at its speed, up to the segment's end, so a vehicle that becomes
  microscopic comes back where it would be rather than up to a segment behind.
* **A vehicle told to stand.** SUMO refuses a top speed of 0, and a speed
  change takes effect only at the next segment, so a vehicle held to a stop
  would crawl on through the segment it is in or never leave it.  One told to
  stand is taken out of SUMO and held where it is, and goes back in, from
  there, once it may drive again.  A speed it is told to keep (above
  standing) takes it out and puts it back in at once, so the new speed holds
  from where it is rather than from the next segment.
* **Signals.** The microscopic SUMO's signals are the run's (after CARLA's,
  when CARLA's are the authority); each step copies the ones that changed
  into this one, whose junction control (``--meso-junction-control``) then
  holds a vehicle at red as SUMO's mesoscopic model does.
"""

from __future__ import annotations

import logging
import math
import os
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

from ..mesoscopic import MesoLink, MesoVehicle

__all__ = ["SumoMesoscopicTraffic", "segment_count"]

logger = logging.getLogger(__name__)

#: A speed slower than this is a vehicle told to stand, m/s (as the queue model's).
_STANDING_MS = 0.01

#: How long SUMO may take to listen for TraCI, s: a large network loads first.
_CONNECT_TIMEOUT_S = 60.0

# TraCI's variable ids (traci.constants), named here so the module does not
# import traci until it starts SUMO.
_VAR_ROAD_ID = 0x50
_VAR_LANEPOSITION = 0x56
_VAR_SPEED = 0x40
_VAR_ROUTE_INDEX = 0x69


def segment_count(length: float, segment_length: float) -> int:
    """How many segments SUMO's mesoscopic model cuts a road of *length* into.

    As ``MELoop::numSegmentsFor``: the road's length over the segment length,
    rounded half up, and at least one.
    """
    if segment_length <= 0.0:
        return 1
    return max(int(math.floor(length / segment_length + 0.5)), 1)


@dataclass
class _Running:
    """Where SUMO last had a vehicle that has departed."""

    road: str
    segment_start_m: float
    #: When it was first seen at the start of the segment, simulated seconds.
    segment_entered_s: float
    speed: float


class SumoMesoscopicTraffic:
    """Vehicles in SUMO's mesoscopic simulation; see the module docstring.

    The interface is :class:`~..mesoscopic.MesoscopicTraffic`'s.

    Args:
        links: Every road the backend's vehicles may become mesoscopic on, by
            id; what :meth:`has_link` and :meth:`link` answer.  SUMO itself
            simulates the whole network.
        net_file: The network the microscopic SUMO runs.
        binary: The ``sumo`` binary.
        step_length: SUMO's step, s: the microscopic SUMO's.
        seed: SUMO's random seed.
        segment_length_m: ``--meso-edgelength``: the length of the segments
            SUMO's mesoscopic model cuts each road into.
        junction_control: Hold a vehicle at a red signal
            (``--meso-junction-control``) and copy the signals from
            *signal_states* every step.
        signal_states: The run's signal states, ``{tls id: state}``, read
            every step under *junction_control*.
        additional_files: Vehicle type files, so a vehicle keeps its type.
        log_file: Where SUMO writes its log.
        extra_args: More SUMO options.
    """

    def __init__(
        self,
        links: Mapping[str, MesoLink],
        *,
        net_file: Path,
        binary: str,
        step_length: float,
        seed: int,
        segment_length_m: float = 20.0,
        junction_control: bool = True,
        signal_states: Optional[Callable[[], Mapping[str, str]]] = None,
        additional_files: Sequence[str] = (),
        log_file: Optional[Path] = None,
        extra_args: Sequence[str] = (),
    ) -> None:
        self._links = dict(links)
        self._segment_length_m = float(segment_length_m)
        self._junction_control = junction_control
        self._signal_states = signal_states
        self._vehicles: dict[str, MesoVehicle] = {}
        #: Vehicles in SUMO that have not departed yet: their record says
        #: where they are.
        self._pending: set[str] = set()
        self._running: dict[str, _Running] = {}
        #: Vehicles told to stand: out of SUMO, where their record says.
        self._held: set[str] = set()
        #: A vehicle's id in SUMO, by its own: a new one each time it goes
        #: in, since SUMO keeps a removed vehicle's own type a while and
        #: refuses it to a vehicle of the same id meanwhile.
        self._sumo_ids: dict[str, str] = {}
        self._routes = 0
        self._pushed_signals: dict[str, str] = {}
        cmd = [
            binary,
            "-n",
            str(net_file),
            "--mesosim",
            "true",
            "--meso-edgelength",
            f"{self._segment_length_m:g}",
            "--meso-junction-control",
            "true" if junction_control else "false",
            "--step-length",
            f"{step_length:g}",
            "--seed",
            str(seed),
            # A vehicle the backend put here waits as long as its road is full.
            "--time-to-teleport",
            "-1",
            "--no-step-log",
            "true",
            "--no-warnings",
            "true",
            "--xml-validation",
            "never",
        ]
        if additional_files:
            cmd += ["--additional-files", ",".join(additional_files)]
        if log_file is not None:
            cmd += ["--log", str(log_file)]
        cmd += list(extra_args)
        self._process: Optional[subprocess.Popen[bytes]] = None
        self._conn: Any = self._start(cmd)
        self._types = set(self._conn.vehicletype.getIDList())

    # ------------------------------------------------------------------
    # SUMO
    # ------------------------------------------------------------------

    def _start(self, cmd: list[str]) -> Any:
        """Start SUMO on a free port and connect to it, without switching
        TraCI's module-level connection over to it."""
        import traci  # noqa: PLC0415
        from traci.exceptions import FatalTraCIError, TraCIException  # noqa: PLC0415

        with socket.socket() as probe:
            probe.bind(("localhost", 0))
            port = int(probe.getsockname()[1])
        process = subprocess.Popen(  # noqa: S603 - SUMO's own binary
            [*cmd, "--remote-port", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env={**os.environ},
        )
        self._process = process
        deadline = time.monotonic() + _CONNECT_TIMEOUT_S
        while True:
            if process.poll() is not None:
                err = (
                    process.stderr.read().decode(errors="replace")
                    if process.stderr
                    else ""
                )
                raise RuntimeError(f"SUMO (mesoscopic) did not start: {err.strip()}")
            try:
                # One try each: traci.connect sleeps a second between its own.
                return traci.connect(port, numRetries=0, proc=process)
            except (FatalTraCIError, TraCIException, OSError):
                if time.monotonic() > deadline:
                    process.kill()
                    raise RuntimeError(
                        "SUMO (mesoscopic) did not listen for TraCI "
                        f"within {_CONNECT_TIMEOUT_S:g} s"
                    ) from None
                time.sleep(0.02)

    def close(self) -> None:
        """Stop SUMO."""
        conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.close(False)
            except Exception as exc:  # noqa: BLE001 - teardown must go on
                logger.debug("SUMO (mesoscopic) close: %s", exc)
        process, self._process = self._process, None
        if process is not None:
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
        self._vehicles = {}
        self._pending = set()
        self._running = {}
        self._held = set()
        self._sumo_ids = {}

    @property
    def time(self) -> float:
        """SUMO's clock, s."""
        return float(self._conn.simulation.getTime())

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
        """Whether *link_id* is a road a vehicle may become mesoscopic on."""
        return link_id in self._links

    def link(self, link_id: str) -> MesoLink:
        """The road *link_id*."""
        return self._links[link_id]

    def vehicle(self, vehicle_id: str) -> MesoVehicle:
        """The vehicle *vehicle_id*; ``KeyError`` if the model has none."""
        return self._vehicles[vehicle_id]

    def held(self, vehicle_id: str) -> bool:
        """Whether *vehicle_id* is told to stand, and so out of SUMO."""
        return vehicle_id in self._held

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

        Raises:
            ValueError: The route is empty, or starts on a road the model does
                not have, or the vehicle is in the model already, or SUMO
                refused it.
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
            exit_s=math.inf,
            max_speed=max_speed,
            length=length,
            type_id=type_id,
            speed_cap=speed_cap,
        )
        if self._stands(vehicle):
            self._held.add(vehicle_id)
        else:
            self._insert(vehicle)
        self._vehicles[vehicle_id] = vehicle
        return vehicle

    def remove(self, vehicle_id: str) -> Optional[MesoVehicle]:
        """Take *vehicle_id* out of the model; ``None`` if it was not in it."""
        vehicle = self._vehicles.pop(vehicle_id, None)
        if vehicle is None:
            return None
        self._take_out(vehicle_id)
        self._held.discard(vehicle_id)
        return vehicle

    def clear(self) -> None:
        """Take every vehicle out."""
        for vehicle_id in list(self._vehicles):
            self.remove(vehicle_id)

    # ------------------------------------------------------------------
    # Telling a vehicle what to do
    # ------------------------------------------------------------------

    def set_speed_cap(
        self, vehicle_id: str, speed: Optional[float], now: float
    ) -> None:
        """Hold *vehicle_id* to *speed* (m/s) from where it is now; ``None``
        lets it drive at the road's speed again."""
        vehicle = self._vehicles[vehicle_id]
        before = self._cap(vehicle)
        self._settle(vehicle, now)
        vehicle.speed_cap = speed
        if self._cap(vehicle) != before:
            self._reinsert(vehicle)

    def set_max_speed(self, vehicle_id: str, speed: float, now: float) -> None:
        """Give *vehicle_id* the top speed *speed* (m/s) from where it is now."""
        vehicle = self._vehicles[vehicle_id]
        before = self._cap(vehicle)
        self._settle(vehicle, now)
        vehicle.max_speed = speed
        if self._cap(vehicle) != before:
            self._reinsert(vehicle)

    def set_route(self, vehicle_id: str, route: Sequence[str]) -> None:
        """Drive *route* from now on; it starts with the road the vehicle is on."""
        vehicle = self._vehicles[vehicle_id]
        if not route or route[0] != vehicle.link:
            raise ValueError(
                f"{vehicle_id}: a new route must start on {vehicle.link} "
                f"(got {list(route)[:1]})"
            )
        if vehicle_id not in self._held:
            from traci.exceptions import TraCIException  # noqa: PLC0415

            try:
                self._conn.vehicle.setRoute(self._sumo_ids[vehicle_id], list(route))
            except TraCIException as exc:
                raise ValueError(
                    f"{vehicle_id}: SUMO refused the route: {exc}"
                ) from exc
        vehicle.route = list(route)
        vehicle.index = 0

    # ------------------------------------------------------------------
    # Where a vehicle is
    # ------------------------------------------------------------------

    def position(self, vehicle_id: str, now: float) -> tuple[str, float]:
        """The road *vehicle_id* is on and how far along it, metres.

        SUMO's segment start, moved on at the vehicle's speed since it got
        there, up to the segment's end.
        """
        vehicle = self._vehicles[vehicle_id]
        running = self._running.get(vehicle_id)
        if running is None:
            return vehicle.link, vehicle.start_m
        link = self._links.get(running.road)
        length = link.length if link is not None else math.inf
        segment = length / segment_count(length, self._segment_length_m)
        moved = running.speed * max(now - running.segment_entered_s, 0.0)
        end = min(running.segment_start_m + segment, length)
        return running.road, max(min(running.segment_start_m + moved, end), 0.0)

    def speed(self, vehicle_id: str, now: float) -> float:  # noqa: ARG002 - the queue model's signature
        """How fast *vehicle_id* is going now, m/s, as SUMO has it."""
        running = self._running.get(vehicle_id)
        return running.speed if running is not None else 0.0

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def step(self, now: float) -> list[MesoVehicle]:
        """Run SUMO up to *now*.

        Returns:
            The vehicles that reached the end of their route, which are out of
            the model now.
        """
        conn = self._conn
        if self._junction_control and self._signal_states is not None:
            self._copy_signals(self._signal_states())
        if conn.simulation.getTime() < now - 1e-9:
            conn.simulationStep(now)
        ids = self._sumo_ids
        in_sumo = set(conn.vehicle.getIDList())
        departed = [i for i in self._pending if ids[i] in in_sumo]
        for vehicle_id in departed:
            self._pending.discard(vehicle_id)
            conn.vehicle.subscribe(
                ids[vehicle_id],
                (_VAR_ROAD_ID, _VAR_ROUTE_INDEX, _VAR_LANEPOSITION, _VAR_SPEED),
            )
        arrived: list[MesoVehicle] = []
        gone = [i for i in self._running if ids[i] not in in_sumo]
        if self._pending:
            waiting = set(conn.simulation.getPendingVehicles())
            gone += [i for i in self._pending if ids[i] not in waiting]
        for vehicle_id in gone:
            ids.pop(vehicle_id, None)
            self._pending.discard(vehicle_id)
            self._running.pop(vehicle_id, None)
            vehicle = self._vehicles.pop(vehicle_id, None)
            if vehicle is not None:
                vehicle.index = len(vehicle.route) - 1
                arrived.append(vehicle)
        results = conn.vehicle.getAllSubscriptionResults()
        for vehicle_id in [*self._running, *departed]:
            values = results.get(ids.get(vehicle_id, ""))
            vehicle = self._vehicles.get(vehicle_id)
            if not values or vehicle is None:
                continue
            road = str(values[_VAR_ROAD_ID])
            offset = float(values[_VAR_LANEPOSITION])
            speed = float(values[_VAR_SPEED])
            index = int(values[_VAR_ROUTE_INDEX])
            if 0 <= index < len(vehicle.route) and vehicle.route[index] == road:
                vehicle.index = index
            before = self._running.get(vehicle_id)
            if before is None:
                # SUMO puts a vehicle in at the start of the segment its
                # offset is in: it goes on from the offset it was given.
                entered = now
                if road == vehicle.link and offset <= vehicle.start_m and speed > 0.0:
                    entered = now - (vehicle.start_m - offset) / speed
                self._running[vehicle_id] = _Running(road, offset, entered, speed)
            elif before.road != road or abs(before.segment_start_m - offset) > 1e-6:
                self._running[vehicle_id] = _Running(road, offset, now, speed)
            else:
                before.speed = speed
        return arrived

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _cap(vehicle: MesoVehicle) -> float:
        speed = vehicle.max_speed
        if vehicle.speed_cap is not None:
            speed = min(speed, vehicle.speed_cap)
        return speed

    def _stands(self, vehicle: MesoVehicle) -> bool:
        return self._cap(vehicle) < _STANDING_MS

    def _insert(self, vehicle: MesoVehicle) -> None:
        """Add *vehicle* to SUMO, from its record's road and offset."""
        from traci.exceptions import TraCIException  # noqa: PLC0415

        conn = self._conn
        route = vehicle.remaining_route
        link = self._links.get(route[0])
        offset = vehicle.start_m
        if link is not None:
            offset = min(offset, max(link.length - 0.1, 0.0))
        self._routes += 1
        sumo_id = f"{vehicle.id}@meso{self._routes}"
        try:
            type_id = self._type_of(vehicle)
            conn.route.add(sumo_id, route)
            conn.vehicle.add(
                sumo_id,
                sumo_id,
                typeID=type_id,
                departPos=f"{offset:.2f}",
                departSpeed="max",
            )
            cap = self._cap(vehicle)
            if math.isfinite(cap):
                conn.vehicle.setMaxSpeed(sumo_id, cap)
        except TraCIException as exc:
            try:
                conn.vehicle.remove(sumo_id)
            except TraCIException:
                pass
            raise ValueError(
                f"{vehicle.id}: SUMO (mesoscopic) refused it: {exc}"
            ) from exc
        vehicle.route = list(route)
        vehicle.index = 0
        self._sumo_ids[vehicle.id] = sumo_id
        self._pending.add(vehicle.id)

    def _type_of(self, vehicle: MesoVehicle) -> str:
        """A type of *vehicle*'s own type and length, shared by every vehicle
        of both: SUMO need not make each vehicle a type of its own."""
        base = vehicle.type_id if vehicle.type_id in self._types else "DEFAULT_VEHTYPE"
        type_id = f"{base}@{vehicle.length:.2f}m"
        if type_id not in self._types:
            self._conn.vehicletype.copy(base, type_id)
            self._conn.vehicletype.setLength(type_id, vehicle.length)
            self._types.add(type_id)
        return type_id

    def _take_out(self, vehicle_id: str) -> None:
        """Remove *vehicle_id* from SUMO, if it is there."""
        sumo_id = self._sumo_ids.pop(vehicle_id, None)
        self._pending.discard(vehicle_id)
        subscribed = self._running.pop(vehicle_id, None) is not None
        if sumo_id is None:
            return
        try:
            if subscribed:
                # Else SUMO keeps answering the subscription of a vehicle it
                # no longer has, with an error on its console each step.
                self._conn.vehicle.unsubscribe(sumo_id)
            self._conn.vehicle.remove(sumo_id)
        except Exception as exc:  # noqa: BLE001 - it may have arrived this step
            logger.debug("SUMO (mesoscopic): removing %s: %s", vehicle_id, exc)

    def _settle(self, vehicle: MesoVehicle, now: float) -> None:
        """Write where *vehicle* is now into its record."""
        if vehicle.id in self._held or vehicle.id not in self._running:
            return
        road, offset = self.position(vehicle.id, now)
        if road in vehicle.route[vehicle.index :]:
            vehicle.index = vehicle.route.index(road, vehicle.index)
        elif road in self._links:
            vehicle.route = [road, *vehicle.remaining_route]
            vehicle.index = 0
        vehicle.start_m = offset
        vehicle.entered_s = now

    def _reinsert(self, vehicle: MesoVehicle) -> None:
        """Put *vehicle* back into SUMO with its new speed, or hold it."""
        self._take_out(vehicle.id)
        if self._stands(vehicle):
            self._held.add(vehicle.id)
            return
        self._held.discard(vehicle.id)
        try:
            self._insert(vehicle)
        except ValueError as exc:
            # Not lost: held where it is until it is told something new.
            logger.debug("%s; held", exc)
            self._held.add(vehicle.id)

    def _copy_signals(self, states: Mapping[str, str]) -> None:
        """Set this SUMO's signals to *states* where they changed."""
        for tls, state in states.items():
            if self._pushed_signals.get(tls) == state:
                continue
            try:
                self._conn.trafficlight.setRedYellowGreenState(tls, state)
            except Exception as exc:  # noqa: BLE001 - one signal must not end the run
                logger.debug("SUMO (mesoscopic): signal %s: %s", tls, exc)
                continue
            self._pushed_signals[tls] = state
