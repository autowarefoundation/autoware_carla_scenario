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
* **A vehicle told to stand.** SUMO's mesoscopic model refuses a top speed
  of 0, implements neither ``setSpeed`` nor ``resume``, and takes a speed
  change only at the next segment.  One told to stand is taken out of SUMO
  and held where it is, and goes back in, from there, once it may drive
  again.  It does not hold up those behind it meanwhile: a stop that never
  ends would, but SUMO does not wake the vehicles queued behind a vehicle
  taken out of it, so they would stand on after it drove off.  A speed it
  is told to keep (above standing) takes it out and puts it back in at once,
  so the new speed holds from where it is rather than from the next segment.
* **Signals.** The microscopic SUMO's signals are the run's (after CARLA's,
  when CARLA's are the authority); each step copies the ones that changed
  into this one, whose junction control (``--meso-junction-control``) then
  holds a vehicle at red as SUMO's mesoscopic model does.
"""

from __future__ import annotations

import logging
import math
import random
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

from ...kinematics.angle import normalize_angle_deg
from ..base import TrafficBackendError
from ..fidelity import MesoscopicConfig
from ..mesoscopic import _STANDING_MS, MesoLink, MesoscopicTraffic, MesoVehicle
from .geometry import Pose2D

__all__ = [
    "MesoRoads",
    "SumoMesoscopicTraffic",
    "build_mesoscopic_model",
    "segment_count",
]

logger = logging.getLogger(__name__)

#: Roads a route is given ahead of a vehicle, and given again as it nears its
#: end: the TrafficManager plans no further than this either.
ROUTE_EDGES = 6
#: How far from a road's centre line a vehicle may be and still be on it, m:
#: half of a wide lane, and some.
LOCATE_RADIUS_M = 3.5
#: How far a vehicle's heading may be from its road's to be on it, degrees:
#: a vehicle changing lanes is still on its road, one on the other
#: carriageway is not.
LOCATE_HEADING_DEG = 60.0
#: The side of a cell of the index of lanes by place, m.
_CELL_M = 50.0

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
        #: The time SUMO has been stepped to: kept here, not asked each step.
        self._stepped_s = 0.0
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
        # The callback is the backend's: a closed model keeps it no longer.
        self._signal_states = None

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
        if vehicle_id in self._sumo_ids:
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
        if not self._vehicles:
            # Nothing to follow: SUMO is only kept in time, and the signals
            # catch up (their diff, _copy_signals) when a vehicle is back.
            self._step_to(now)
            return []
        if self._junction_control and self._signal_states is not None:
            self._copy_signals(self._signal_states())
        self._step_to(now)
        from traci.exceptions import TraCIException  # noqa: PLC0415

        ids = self._sumo_ids
        # No list of every vehicle: one in SUMO is subscribed to from the
        # step it departs, and one that has left is missing from the results.
        departed: list[str] = []
        gone: list[str] = []
        if self._pending:
            waiting = set(conn.simulation.getPendingVehicles())
            for vehicle_id in [i for i in self._pending if ids[i] not in waiting]:
                self._pending.discard(vehicle_id)
                try:
                    conn.vehicle.subscribe(
                        ids[vehicle_id],
                        (_VAR_ROAD_ID, _VAR_ROUTE_INDEX, _VAR_LANEPOSITION, _VAR_SPEED),
                    )
                except TraCIException:
                    gone.append(vehicle_id)  # in and out within the step
                    continue
                departed.append(vehicle_id)
        results = conn.vehicle.getAllSubscriptionResults()
        gone += [i for i in self._running if ids[i] not in results]
        arrived: list[MesoVehicle] = []
        for vehicle_id in gone:
            ids.pop(vehicle_id, None)
            self._pending.discard(vehicle_id)
            self._running.pop(vehicle_id, None)
            vehicle = self._vehicles.pop(vehicle_id, None)
            if vehicle is not None:
                vehicle.index = len(vehicle.route) - 1
                arrived.append(vehicle)
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

    def _step_to(self, now: float) -> None:
        if self._stepped_s < now - 1e-9:
            self._conn.simulationStep(now)
            self._stepped_s = now

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


def build_mesoscopic_model(
    config: MesoscopicConfig,
    links: Mapping[str, MesoLink],
    *,
    net_file: Path,
    binary: str,
    step_length: float,
    seed: int,
    log_file: Path,
    signal_states: Optional[Callable[[], Mapping[str, str]]] = None,
    gate: Optional[Callable[[str, str], bool]] = None,
    additional_files: Sequence[str] = (),
) -> "MesoscopicTraffic | SumoMesoscopicTraffic":
    """The model far traffic runs in on *links*, as *config* chooses:
    SUMO's own mesoscopic simulation (``engine: sumo``; *signal_states*
    copied into it under ``junction_control``) or the queue model
    (``engine: queue``; *gate* holding it at junctions under
    ``junction_control``).

    Raises:
        TrafficBackendError: SUMO's mesoscopic simulation did not start.
    """
    if config.engine == "sumo":
        try:
            return SumoMesoscopicTraffic(
                links,
                net_file=net_file,
                binary=binary,
                step_length=step_length,
                seed=seed,
                segment_length_m=config.segment_length_m,
                junction_control=config.junction_control,
                signal_states=signal_states,
                additional_files=additional_files,
                log_file=log_file,
            )
        except (OSError, RuntimeError) as exc:
            raise TrafficBackendError(
                f"SUMO's mesoscopic simulation did not start: {exc}"
            ) from exc
    return MesoscopicTraffic(
        links,
        headway_s=config.headway_s,
        jam_spacing_m=config.jam_spacing_m,
        gate=gate if config.junction_control else None,
    )


class MesoRoads:
    """The roads of a SUMO network a car may drive, for a backend whose
    vehicles are not in SUMO: the mesoscopic model's links, where a point is
    on them, where a place on them is, and routes on.

    Positions are in SUMO's frame (:mod:`.geometry` turns CARLA's into it):
    a vehicle's front bumper, as SUMO places a vehicle, and the edge offset
    along its road's rightmost lane a car may use -- the mesoscopic model has
    no lanes.
    """

    def __init__(self, net: Any) -> None:
        """Index *net* (a ``sumolib.net.Net``)."""
        self._net = net
        self.links: dict[str, MesoLink] = {}
        #: Each road's rightmost lane a car may use: where a vehicle on it is.
        self._shapes: dict[str, list[tuple[float, float]]] = {}
        #: Every lane a car may use, as ``(edge, shape)``, and those near each
        #: cell of the plane, so a point is matched against a few.
        self._lanes: list[tuple[str, list[tuple[float, float]]]] = []
        self._cells: dict[tuple[int, int], list[int]] = {}
        for edge in net.getEdges():
            if edge.getFunction() != "" or not edge.allows("passenger"):
                continue
            lanes = [lane for lane in edge.getLanes() if lane.allows("passenger")]
            if not lanes:
                continue
            edge_id = str(edge.getID())
            self.links[edge_id] = MesoLink(
                edge_id, float(edge.getLength()), len(lanes), float(edge.getSpeed())
            )
            self._shapes[edge_id] = _shape(lanes[0])
            for lane in lanes:
                self._index(edge_id, _shape(lane))

    def _index(self, edge_id: str, shape: list[tuple[float, float]]) -> None:
        index = len(self._lanes)
        self._lanes.append((edge_id, shape))
        xs = [p[0] for p in shape]
        ys = [p[1] for p in shape]
        r = LOCATE_RADIUS_M
        for i in range(_cell(min(xs) - r), _cell(max(xs) + r) + 1):
            for j in range(_cell(min(ys) - r), _cell(max(ys) + r) + 1):
                self._cells.setdefault((i, j), []).append(index)

    def locate(
        self, x: float, y: float, heading_deg: float
    ) -> Optional[tuple[str, float]]:
        """The road a vehicle at SUMO's ``(x, y)``, heading *heading_deg*
        (SUMO's angle: clockwise from north), is on, and how far along it;
        ``None`` in a junction or off every road, or facing against it."""
        import sumolib  # noqa: PLC0415

        geom = sumolib.geomhelper
        best: Optional[tuple[float, str, float]] = None
        for index in self._cells.get((_cell(x), _cell(y)), ()):
            edge_id, shape = self._lanes[index]
            distance = float(geom.distancePointToPolygon((x, y), shape))
            if distance > LOCATE_RADIUS_M or (best is not None and distance >= best[0]):
                continue
            offset = float(
                geom.polygonOffsetWithMinimumDistanceToPoint((x, y), shape, False)
            )
            angle = 90.0 - math.degrees(
                geom.rotationAtShapeOffset(shape, offset) or 0.0
            )
            if _angle_between(angle, heading_deg) > LOCATE_HEADING_DEG:
                continue
            best = (distance, edge_id, offset)
        if best is None:
            return None
        _distance, edge_id, offset = best
        length = self.links[edge_id].length
        return edge_id, min(max(offset, 0.0), max(length - 0.1, 0.0))

    def pose(self, edge_id: str, offset: float) -> Pose2D:
        """Where *offset* metres along *edge_id* is, in SUMO's frame, facing
        along the road (SUMO's angle)."""
        import sumolib  # noqa: PLC0415

        geom = sumolib.geomhelper
        shape = self._shapes[edge_id]
        # On the shape, whose length is not quite the edge's.
        offset = min(max(offset, 0.0), max(geom.polyLength(shape) - 0.01, 0.0))
        x, y = geom.positionAtShapeOffset(shape, offset)
        rotation = geom.rotationAtShapeOffset(shape, offset) or 0.0
        return Pose2D(float(x), float(y), 0.0, (90.0 - math.degrees(rotation)) % 360.0)

    def route_on(
        self, route: list[str], rng: random.Random, edges: int = ROUTE_EDGES
    ) -> list[str]:
        """*route* extended by up to *edges* roads a car may drive, each taken
        at random where a road branches; shorter at a dead end."""
        route = list(route)
        for _ in range(edges):
            outgoing = sorted(
                str(to_edge.getID())
                for to_edge in self._net.getEdge(route[-1]).getOutgoing()
                if str(to_edge.getID()) in self.links
            )
            if not outgoing:
                break
            route.append(rng.choice(outgoing))
        return route


def _shape(lane: Any) -> list[tuple[float, float]]:
    return [(float(p[0]), float(p[1])) for p in lane.getShape()]


def _cell(value: float) -> int:
    return int(math.floor(value / _CELL_M))


def _angle_between(a: float, b: float) -> float:
    """How far apart two headings are, degrees, 0 to 180."""
    return abs(normalize_angle_deg(a - b))
