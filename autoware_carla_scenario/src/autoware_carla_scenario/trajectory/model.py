"""The trajectory an entity is told to follow, after OpenSCENARIO's.

OpenSCENARIO 1.x describes following a trajectory with four parts, and this
module models each one under the same name:

* a :class:`Trajectory` -- a named, possibly closed, shape.  Only the
  ``Polyline`` shape is offered: a recorded drive is a sequence of samples, and
  a clothoid or NURBS fitted to one would only add an approximation between the
  data and the vehicle;
* its :class:`TrajectoryVertex` list -- a position, and the condition the
  entity departs it on (:attr:`TrajectoryVertex.advance`): the time it is
  there (OpenSCENARIO's ``Vertex.time``, here a ``TrajectoryTimeCondition``)
  or -- an extension, a *waypoint condition* -- any other condition;
* a time reference -- ``None`` (OpenSCENARIO's ``<None/>``: the vertex times are
  ignored and the entity keeps its own speed) or a :class:`TrajectoryTiming`;
* a :class:`TrajectoryFollowingMode` -- ``position`` (be exactly on the
  trajectory) or ``follow`` (a controller tracks it).

A vertex's position may be written in any frame the framework addresses: a
:class:`~autoware_carla_scenario.coordinate.poses.Lanelet2Pose`,
:class:`~autoware_carla_scenario.coordinate.poses.OpenDrivePose` or
:class:`~autoware_carla_scenario.coordinate.poses.CarlaWorldPose`, a
:class:`MapPose` -- an absolute pose in Autoware's ``map`` frame, which is what a
recording (a T4 scene, a rosbag) states -- or a :class:`RelativeLanePose`, an
offset in lanes and metres from where an entity is when the action starts
(OpenSCENARIO's ``RelativeLanePosition``).  Positions are resolved to CARLA world
coordinates only when the action runs, because that needs the loaded map; the
geometry here (:class:`ResolvedTrajectory`) works on the resolved points and
imports nothing from CARLA, so it can be tested without a simulator.
"""

from __future__ import annotations

import bisect
import enum
import math
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

if TYPE_CHECKING:
    from ..conditions.base import BaseCondition
    from ..coordinate.poses import CarlaWorldPose, Lanelet2Pose, OpenDrivePose
    from ..entity_role import EntityRole

__all__ = [
    "MapPose",
    "ROUTE_POSITIONS",
    "ReferenceContext",
    "RelativeLanePose",
    "RouteCrossingPose",
    "RouteCrosswalkPose",
    "RouteLanePose",
    "RouteOppositePose",
    "RouteRoadsidePose",
    "ResolvedTrajectory",
    "Trajectory",
    "TrajectoryFollowingMode",
    "TrajectoryPosition",
    "TrajectorySample",
    "TrajectoryTiming",
    "TrajectoryVertex",
    "placed_against_entity",
]

#: Segments shorter than this (m) have no direction of their own.
_EPSILON_M = 1e-6
#: Durations shorter than this (s) are treated as an instant.
_EPSILON_S = 1e-9

#: How far (m) either side of a vertex without a yaw of its own the entity
#: turns from the segment arriving there to the one leaving: at most this, and
#: at most half of either segment, so a sparse path is driven along each leg
#: and a dense one (a recording) turns all the way along, at its own rate.
_CORNER_HALF_WIDTH_M = 1.0

#: How long (s) a timed trajectory has to stay on one point, between two
#: segments that move, for that to be a **stop** rather than part of a corner.
#: A shorter pause is a sample written twice -- a recording resampled onto a
#: faster tick, ``0.05`` to ``0.15`` s on CARLA's -- and the corner turns
#: through it as through the window either side; a stop keeps the heading it
#: arrived with and turns on the spot.
_STOP_MIN_S = 0.5


@dataclass(frozen=True)
class MapPose:
    """An absolute pose in Autoware's ``map`` frame.

    The frame Autoware's ``map_loader`` produces from the Lanelet2 file: the
    projected (MGRS or local) frame this package calls "Lanelet2 absolute",
    right-handed, x East, y North, z Up.  It is the frame a recording of
    Autoware states its poses in, so a recorded trajectory can be written down
    as it was recorded and placed in CARLA through the map's own offset.

    Args:
        x: East, in metres.
        y: North, in metres.
        yaw: Heading in radians, counter-clockwise from East.  ``None`` takes
            the direction of the trajectory at this vertex instead.
        z: Absolute elevation in metres.  ``None`` places the entity on the road
            surface under it, which is what a 2-D recording needs.

    Raises:
        ValueError: A field is not finite (``nan``, ``inf``).
    """

    x: float
    y: float
    yaw: Optional[float] = None
    z: Optional[float] = None

    def __post_init__(self) -> None:
        _finite("MapPose", "x", self.x)
        _finite("MapPose", "y", self.y)
        _finite_or_none("MapPose", "yaw", self.yaw)
        _finite_or_none("MapPose", "z", self.z)


def _finite(owner: str, name: str, value: float) -> None:
    if not math.isfinite(float(value)):
        raise ValueError(f"{owner}.{name} must be finite")


def _finite_or_none(owner: str, name: str, value: Optional[float]) -> None:
    """:func:`_finite` for a field that ``None`` leaves unset."""
    if value is not None:
        _finite(owner, name, value)


def _role_name(role: Union["EntityRole", str, None]) -> Optional[str]:
    """*role* as the ``role_name`` string it stands for (``None`` stays ``None``)."""
    if role is None:
        return None
    return str(role)


@dataclass(frozen=True)
class RelativeLanePose:
    """A pose relative to an entity, in lane coordinates.

    OpenSCENARIO's ``RelativeLanePosition`` (``entityRef``, ``dLane``, ``ds``,
    ``offset``): start from the lanelet the reference entity is on and the
    distance it has come along it, go *ds* metres along the lane, then
    *d_lane* lanes across, and stand *offset* metres from that lane's
    centreline.  It is resolved against where the reference entity is when the
    action starts, and stays where it was resolved to afterwards: the vertex
    does not move with the entity (see ``docs/trajectory.md``).

    * Along the lane, a lanelet's end continues into the lanelet following it
      in the routing graph (a negative *ds* into the one before it).  Where
      there are several, the one that turns least is taken, then the lowest
      id: deterministic, and the "straight on" a route would most often take.
    * Across, *d_lane* counts lanes of the same direction of travel, through
      the routing graph's left and right neighbours, lane-changeable or not.
      The lane change is made at the point *ds* reached, onto the neighbour of
      the lanelet found there.

    Args:
        ds: Metres along the reference entity's lane, from where it is;
            negative goes back.
        offset: Metres from the target lane's centreline, positive to the left
            of its direction of travel -- the sign of
            :attr:`~autoware_carla_scenario.coordinate.poses.Lanelet2Pose.t`.
            Measured from the centreline, not from the reference entity's own
            lateral position, as OpenSCENARIO does.
        d_lane: Lanes across: ``+1`` one lane to the left, ``-1`` one to the
            right, ``0`` the reference entity's own lane.
        yaw: Heading in radians relative to the target lane's direction,
            counter-clockwise (left) positive -- the sense of
            :attr:`~autoware_carla_scenario.coordinate.poses.Lanelet2Pose.heading`.
            ``None`` takes the direction of the trajectory at this vertex.
        entity_ref: ``role_name`` of the entity the pose is relative to.
            ``None`` means the entity the action moves.
    """

    ds: float = 0.0
    offset: float = 0.0
    d_lane: int = 0
    yaw: Optional[float] = None
    #: The reference entity's ``role_name``.  An :class:`EntityRole` is
    #: accepted and stored as its string, so the pose holds one type whatever
    #: it was built with (the static check, ``docs/typecheck.md``, needs an
    #: attribute of one type).
    entity_ref: Optional[str] = None

    def __init__(
        self,
        ds: float = 0.0,
        offset: float = 0.0,
        d_lane: int = 0,
        yaw: Optional[float] = None,
        entity_ref: Union["EntityRole", str, None] = None,
    ) -> None:
        # Written out, rather than generated, so that the constructor still
        # takes an EntityRole while the field holds its string.
        object.__setattr__(self, "ds", ds)
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "d_lane", d_lane)
        object.__setattr__(self, "yaw", yaw)
        object.__setattr__(self, "entity_ref", _role_name(entity_ref))
        self.__post_init__()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "d_lane", _whole("RelativeLanePose", "d_lane", self.d_lane)
        )
        _finite("RelativeLanePose", "ds", self.ds)
        _finite("RelativeLanePose", "offset", self.offset)
        _finite_or_none("RelativeLanePose", "yaw", self.yaw)


def _one_of(owner: str, name: str, value: str, allowed: Sequence[str]) -> None:
    if value not in allowed:
        raise ValueError(
            f"{owner}.{name} must be one of {list(allowed)}, got {value!r}"
        )


def _whole(owner: str, name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{owner}.{name} must be a whole number, got {value!r}")
    if TYPE_CHECKING:
        pass
    elif getattr(getattr(value, "dtype", None), "kind", None) == "b" or (
        # numpy's ndim is an int; a stand-in's (a mock's) is not (#1906).
        isinstance(getattr(value, "ndim", 0), int) and getattr(value, "ndim", 0) != 0
    ):
        # numpy's bool_, which int() takes as 0 or 1, and an array of one
        # element, which the numpy on Python 3.10 (2.2) still converts
        # (#1906), as seed.check_whole_number refuses them.
        raise ValueError(f"{owner}.{name} must be a whole number, got {value!r}")
    if int(value) != value:
        raise ValueError(f"{owner}.{name} must be a whole number, got {value!r}")
    return int(value)


def _check_anchor(owner: str, anchor: Optional[str]) -> None:
    if anchor is None:
        return
    from ..route.model import parse_anchor  # noqa: PLC0415 -- pure, no map

    try:
        parse_anchor(anchor)
    except ValueError as exc:
        raise ValueError(f"{owner}.anchor: {exc}") from exc


@dataclass(frozen=True)
class RouteLanePose:
    """A pose on the ego's route, in lane coordinates (route frame).

    The route counterpart of :class:`RelativeLanePose`, for a *logical*
    scenario (``docs/logical_scenarios.md``): ``ds`` is measured along the
    ego's route -- the lanelets of the scenario's route match, across lanelet
    ends and through its junctions -- rather than along whichever lane an
    entity happens to be on.  From route s ``base + ds``, go ``d_lane`` lanes
    of the same direction across and stand ``offset`` metres from that lane's
    centreline.

    ``base`` is the route s of *anchor* (``junction:0:entry``, see
    :func:`~autoware_carla_scenario.route.model.parse_anchor`) or, with no
    anchor, of the ego when the action starts: the ego's position projected
    onto the route.  Before the route's start or past its end the walk goes on
    along the lane (the straightest way, as :class:`RelativeLanePose` does).

    Args:
        ds: Metres along the route from the base; negative goes back.
        offset: Metres from the target lane's centreline, positive left.
        d_lane: Lanes across: ``+1`` one to the left, ``-1`` one to the right.
        yaw: Heading in radians relative to the lane's direction; ``None``
            faces along the trajectory.
        anchor: The named point ``ds`` counts from; ``None`` is the ego.
    """

    ds: float = 0.0
    offset: float = 0.0
    d_lane: int = 0
    yaw: Optional[float] = None
    anchor: Optional[str] = None

    def __post_init__(self) -> None:
        _finite("RouteLanePose", "ds", self.ds)
        _finite("RouteLanePose", "offset", self.offset)
        _finite_or_none("RouteLanePose", "yaw", self.yaw)
        object.__setattr__(
            self, "d_lane", _whole("RouteLanePose", "d_lane", self.d_lane)
        )
        _check_anchor("RouteLanePose", self.anchor)


@dataclass(frozen=True)
class RouteOppositePose:
    """A pose on the opposite-direction road abreast of a point of the route.

    From route s ``base + ds`` (``base`` as for :class:`RouteLanePose`), cross
    to the lanes that run the other way beside the route -- on whichever side
    they are, so the same document works for left- and right-hand traffic --
    and take the ``lane``-th of them counted from the centre line (``1`` is the
    one next to it).  The pose faces that lane's own direction of travel.

    Args:
        ds: Metres along the route from the base.
        lane: Which opposite lane, from the centre line outwards, from 1.
        offset: Metres from that lane's centreline, positive to its left.
        yaw: Heading relative to that lane's direction; ``None`` faces along
            the trajectory.
        anchor: The named point ``ds`` counts from; ``None`` is the ego.
    """

    ds: float = 0.0
    lane: int = 1
    offset: float = 0.0
    yaw: Optional[float] = None
    anchor: Optional[str] = None

    def __post_init__(self) -> None:
        _finite("RouteOppositePose", "ds", self.ds)
        _finite("RouteOppositePose", "offset", self.offset)
        _finite_or_none("RouteOppositePose", "yaw", self.yaw)
        lane = _whole("RouteOppositePose", "lane", self.lane)
        if lane < 1:
            raise ValueError("RouteOppositePose.lane counts from 1")
        object.__setattr__(self, "lane", lane)
        _check_anchor("RouteOppositePose", self.anchor)


@dataclass(frozen=True)
class RouteCrossingPose:
    """A pose on a lanelet that enters one of the route's junctions from the side.

    Junction ``junction`` of the route (its junction segments counted from 0)
    is entered by other lanelets as well; the *approach* says which, seen from
    the ego coming in: from its ``left``, its ``right`` or ``opposite`` it, or
    ``same`` -- another lane of the ego's own approach (a car turning beside
    it).
    ``turn`` narrows them to those that turn that way.  The pose is
    ``distance`` metres along that lanelet's path from where it enters the
    junction: negative is before it, on the approach road; positive is inside
    the junction and beyond.

    Args:
        junction: Which junction of the route, from 0.
        approach: ``same``, ``left``, ``right`` or ``opposite``.
        distance: Metres along the crossing path from its junction entry.
        turn: ``left``, ``right``, ``straight`` or ``None`` (any).
        offset: Metres from that path's centreline, positive to its left.
        yaw: Heading relative to that path's direction; ``None`` faces along
            the trajectory.
        d_lane: Lanes across from the path's own lane where the pose is,
            positive to its left -- another lane of the road it is on.
        path: Which of the paths entering from *approach* and turning *turn*
            -- counted from 1, from the left as they enter the junction
            (:func:`~autoware_carla_scenario.route.positions.paths_from_left`)
            -- where two lanes turn the same way, say.  ``None``: the one
            crossing the ego's way first, then the lowest lanelet id.
    """

    junction: int = 0
    approach: str = "left"
    distance: float = 0.0
    turn: Optional[str] = None
    offset: float = 0.0
    yaw: Optional[float] = None
    d_lane: int = 0
    path: Optional[int] = None

    def __post_init__(self) -> None:
        from ..route.model import PATH_APPROACHES  # noqa: PLC0415

        _finite("RouteCrossingPose", "distance", self.distance)
        _finite("RouteCrossingPose", "offset", self.offset)
        _finite_or_none("RouteCrossingPose", "yaw", self.yaw)
        junction = _whole("RouteCrossingPose", "junction", self.junction)
        if junction < 0:
            raise ValueError("RouteCrossingPose.junction counts from 0")
        object.__setattr__(self, "junction", junction)
        _one_of("RouteCrossingPose", "approach", self.approach, PATH_APPROACHES)
        object.__setattr__(
            self, "d_lane", _whole("RouteCrossingPose", "d_lane", self.d_lane)
        )
        if self.path is not None:
            path = _whole("RouteCrossingPose", "path", self.path)
            if path < 1:
                raise ValueError("RouteCrossingPose.path counts from 1")
            if self.turn is None:
                # The search counts the paths of one turn; any turn would
                # count others.
                raise ValueError("RouteCrossingPose.path needs a turn")
            object.__setattr__(self, "path", path)
        if self.turn is not None:
            _one_of(
                "RouteCrossingPose", "turn", self.turn, ("left", "right", "straight")
            )


@dataclass(frozen=True)
class RouteCrosswalkPose:
    """A pose on (or just off) a crosswalk at one of the route's junctions.

    The crosswalk that crosses the ego's path on the way into (``leg="entry"``)
    or out of (``"exit"``) junction ``junction``.  ``side`` picks its end on
    the ego's left or right; ``along`` is metres along the crosswalk from that
    end towards the other -- negative stands that far back from it, on the
    pavement.  The pose faces across, towards the other end.

    Args:
        junction: Which junction of the route, from 0.
        leg: ``entry`` or ``exit``.
        side: ``left`` or ``right`` of the ego's path.
        along: Metres along the crosswalk from that end; negative is behind it.
        yaw: Heading relative to facing across; ``None`` faces along the
            trajectory.
    """

    junction: int = 0
    leg: str = "entry"
    side: str = "right"
    along: float = 0.0
    yaw: Optional[float] = None

    def __post_init__(self) -> None:
        from ..route.model import CROSSWALK_LEGS, SIDES  # noqa: PLC0415

        _finite("RouteCrosswalkPose", "along", self.along)
        _finite_or_none("RouteCrosswalkPose", "yaw", self.yaw)
        junction = _whole("RouteCrosswalkPose", "junction", self.junction)
        if junction < 0:
            raise ValueError("RouteCrosswalkPose.junction counts from 0")
        object.__setattr__(self, "junction", junction)
        _one_of("RouteCrosswalkPose", "leg", self.leg, CROSSWALK_LEGS)
        _one_of("RouteCrosswalkPose", "side", self.side, SIDES)


@dataclass(frozen=True)
class RouteRoadsidePose:
    """A pose at the edge of the road abreast of a point of the route.

    From route s ``base + ds`` (``base`` as for :class:`RouteLanePose`), go out
    to the road's edge on ``side`` -- past every lane on that side, the
    opposite road's included -- and stand ``kerb_distance`` metres beyond it
    (negative: on the road, that far in).  Facing along the route.

    Args:
        ds: Metres along the route from the base.
        side: ``left`` or ``right`` of the route's direction.
        kerb_distance: Metres beyond the road's edge.
        yaw: Heading relative to the route's direction; ``None`` faces along
            the trajectory.
        anchor: The named point ``ds`` counts from; ``None`` is the ego.
    """

    ds: float = 0.0
    side: str = "left"
    kerb_distance: float = 0.5
    yaw: Optional[float] = None
    anchor: Optional[str] = None

    def __post_init__(self) -> None:
        from ..route.model import SIDES  # noqa: PLC0415

        _finite("RouteRoadsidePose", "ds", self.ds)
        _finite("RouteRoadsidePose", "kerb_distance", self.kerb_distance)
        _finite_or_none("RouteRoadsidePose", "yaw", self.yaw)
        _one_of("RouteRoadsidePose", "side", self.side, SIDES)
        _check_anchor("RouteRoadsidePose", self.anchor)


#: The positions placed against the scenario's route when the action starts.
ROUTE_POSITIONS: Tuple[type, ...] = (
    RouteLanePose,
    RouteOppositePose,
    RouteCrossingPose,
    RouteCrosswalkPose,
    RouteRoadsidePose,
)

#: Any position a vertex may be written in.
TrajectoryPosition = Union[
    "CarlaWorldPose",
    "Lanelet2Pose",
    "OpenDrivePose",
    MapPose,
    RelativeLanePose,
    RouteLanePose,
    RouteOppositePose,
    RouteCrossingPose,
    RouteCrosswalkPose,
    RouteRoadsidePose,
]


def placed_against_entity(
    position: Union[
        CarlaWorldPose,
        Lanelet2Pose,
        OpenDrivePose,
        MapPose,
        RelativeLanePose,
        RouteLanePose,
        RouteOppositePose,
        RouteCrossingPose,
        RouteCrosswalkPose,
        RouteRoadsidePose,
    ],
) -> bool:
    """Whether *position* is placed against where an entity is now, and so
    moves with it: a :class:`RelativeLanePose` (its ``entity_ref``), or a
    route pose counted from the ego (a :class:`RouteLanePose`,
    :class:`RouteOppositePose` or :class:`RouteRoadsidePose` without an
    ``anchor``).  Any other is a fixed place."""
    if isinstance(position, RelativeLanePose):
        return True
    # Each class on its own, named as the class: Codon reads no attribute of
    # a union (docs/typecheck.md).
    if isinstance(position, RouteLanePose):
        on_lane: RouteLanePose = position
        return on_lane.anchor is None
    if isinstance(position, RouteOppositePose):
        opposite: RouteOppositePose = position
        return opposite.anchor is None
    if isinstance(position, RouteRoadsidePose):
        roadside: RouteRoadsidePose = position
        return roadside.anchor is None
    return False


#: How a vertex's time is written now, for the errors below.
_HOW = "TrajectoryVertex(position, advance=TrajectoryTimeCondition({time!r}))"


#: ``TrajectoryVertex.__init__`` as written, which the wrapper below keeps for
#: mypy: a type checker sees ``(position, advance=None)``, typed.
_Init = TypeVar("_Init", bound="Callable[..., None]")


def _refusing_vertex_times(init: _Init) -> _Init:
    """``TrajectoryVertex.__init__``, telling an old call how it is written now.

    Every call written for the vertex that had a time -- a time second, a time
    and a condition, ``time=`` -- is told how it is written now instead of
    getting Python's arity error, and the signature introspection sees stays
    ``(position, advance=None)``.  A run-time wrapper only: the static check
    (``docs/typecheck.md``) drops the decorator and compiles ``__init__`` as
    written, so ``inspect`` is imported here, where it is never compiled.
    """
    import functools  # noqa: PLC0415 -- run time only, see above
    import inspect  # noqa: PLC0415

    @functools.wraps(init)
    def __init__(self: Any, position: Any, *args: Any, **kwargs: Any) -> None:
        init(self, position, _vertex_advance(args, kwargs))

    __init__.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        [
            inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("position", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter(
                "advance", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None
            ),
        ]
    )
    return __init__  # type: ignore[return-value]


@dataclass(frozen=True)
class TrajectoryVertex:
    """One vertex of a polyline: where, and what the entity departs it on.

    **Departing.**  Each vertex has at most one departure condition,
    :attr:`advance`: when it holds, the entity leaves the vertex for the next
    one.  It is

    * a **time** -- ``advance=TrajectoryTimeCondition(t)``: depart when the
      trajectory's clock reaches ``t`` (see :class:`TrajectoryTiming` for
      that clock).  This is what a vertex time has always meant; :attr:`time`
      reads it back;
    * any other **condition**: depart once it holds.  Any
      :class:`~autoware_carla_scenario.conditions.BaseCondition` will do --
      an elapsed time, a distance to another entity, a speed, a traffic
      signal, an ``AndCondition`` / ``OrCondition`` of them, or one written
      later -- because the action only asks what every condition answers,
      ``check(world, elapsed)``: anything but ``None`` is satisfied.
      ``elapsed`` is the scenario's elapsed time, as for an action's trigger;
    * ``None`` -- the vertex is passed through on arrival.

    See ``docs/trajectory.md`` (*Waypoint conditions*) for how the entity
    moves between vertices departed in these different ways.

    Args:
        position: Where the entity is at this vertex.
        advance: What the entity departs this vertex on.

    Raises:
        TypeError: If *advance* is not a condition -- in particular a number,
            or a ``time=`` keyword: a vertex no longer takes a bare time.
    """

    position: TrajectoryPosition
    advance: Optional["BaseCondition"] = None

    @_refusing_vertex_times
    def __init__(
        self,
        # Spelled out rather than named through TrajectoryPosition: the
        # library check (docs/typecheck.md) calls the constructor with each
        # member of a union written whole, and with the alias's union value
        # otherwise, which catches less.
        position: Union[
            CarlaWorldPose,
            Lanelet2Pose,
            OpenDrivePose,
            MapPose,
            RelativeLanePose,
            RouteLanePose,
            RouteOppositePose,
            RouteCrossingPose,
            RouteCrosswalkPose,
            RouteRoadsidePose,
        ],
        advance: Optional["BaseCondition"] = None,
    ) -> None:
        # Duck-typed rather than an isinstance check: importing the condition
        # package here would pull its CARLA-facing modules into the editor
        # process, which builds trajectories without a simulator.
        if advance is not None and not _has_check(advance):
            raise TypeError(
                "TrajectoryVertex.advance must be a condition (a BaseCondition), "
                f"got {type(advance).__name__}"
            )
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "advance", advance)

    @property
    def time(self) -> Optional[float]:
        """The trajectory-clock time this vertex departs at, if that is its condition."""
        if self.advance is None:
            return None
        return _trajectory_time(self.advance)

    @property
    def gate(self) -> Optional["BaseCondition"]:
        """Its departure condition when that is not a time, else ``None``."""
        if self.advance is None or self.time is not None:
            return None
        return self.advance


def _has_check(condition: "BaseCondition") -> bool:
    """Whether *condition* has a callable ``check``, as every condition has."""
    try:
        check = condition.check
    except AttributeError:
        return False
    return callable(check)


def _vertex_advance(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """The ``advance`` a :class:`TrajectoryVertex` call gives, or why it is wrong."""
    unknown = sorted(set(kwargs) - {"advance", "time"})
    if unknown:
        raise TypeError(f"TrajectoryVertex got unexpected arguments {unknown}")
    given = list(args)
    if "advance" in kwargs:
        given.append(kwargs["advance"])
    if "time" in kwargs:
        time = kwargs["time"]
        if time is None:
            raise TypeError(
                "TrajectoryVertex takes no time any more: drop time=None (a "
                "vertex's time is its departure condition, advance=...)."
            )
        raise TypeError(
            "TrajectoryVertex takes no time any more: a vertex's time is its "
            "departure condition. Write " + _HOW.format(time=time) + "."
        )
    if len(given) > 1:
        first = given[0]
        how = (
            " A time is written " + _HOW.format(time=first) + "."
            if isinstance(first, (int, float)) and not isinstance(first, bool)
            else ""
        )
        raise TypeError(
            "TrajectoryVertex takes one departure condition: a vertex no longer "
            "has a time beside its condition, so give it one advance -- its "
            "time, or the condition (an AndCondition of an "
            "ElapsedTimeCondition and the other, for both)." + how
        )
    advance = given[0] if given else None
    if isinstance(advance, (int, float)) and not isinstance(advance, bool):
        raise TypeError(
            "TrajectoryVertex takes no time any more: a vertex's time is its "
            "departure condition. Write " + _HOW.format(time=advance) + "."
        )
    return advance


def _trajectory_time(condition: "BaseCondition") -> Optional[float]:
    """*condition*'s time if it is a ``TrajectoryTimeCondition``, else ``None``.

    Recognised by the marker that class carries rather than by importing it:
    the condition package's import pulls CARLA in, and the editor process,
    which builds and reads trajectories, has no simulator.  ``hasattr`` rather
    than ``getattr`` with a default, which the static check cannot compile.
    """
    if hasattr(type(condition), "IS_TRAJECTORY_TIME"):
        if type(condition).IS_TRAJECTORY_TIME:  # type: ignore[attr-defined]
            return float(condition.time)  # type: ignore[attr-defined]
    return None


@dataclass(frozen=True)
class Trajectory:
    """A named polyline an entity can be told to follow.

    Args:
        name: What the trajectory is called, for logs.
        vertices: At least two vertices, in the order they are driven.  If
            any vertex departs at a time (a ``TrajectoryTimeCondition``),
            every vertex has to depart on some condition -- either every
            vertex has a time or none has, counting a vertex with another
            condition as having one -- and the times must not decrease along
            the path.
        closed: Whether the last vertex joins back to the first.  Only
            meaningful without timing -- a timed trajectory ends at its last
            vertex's time -- and refused with it, as OpenSCENARIO does.  Its
            other conditions apply on every lap.

    Raises:
        ValueError: On fewer than two vertices, on times given to some
            vertices and no condition to others, on decreasing times, or on a
            closed trajectory that carries times.
    """

    name: str
    vertices: Tuple[TrajectoryVertex, ...]
    closed: bool = False

    def __init__(
        self,
        name: str,
        vertices: Sequence[TrajectoryVertex],
        closed: bool = False,
    ) -> None:
        vertices = tuple(vertices)
        if len(vertices) < 2:
            raise ValueError(
                f"trajectory {name!r} needs at least two vertices, got {len(vertices)}"
            )
        times = [float(t) for t in (v.time for v in vertices) if t is not None]
        if times:
            if any(v.advance is None for v in vertices):
                raise ValueError(
                    f"trajectory {name!r}: either every vertex has a time or none "
                    "has (a vertex departed by another condition counts as "
                    "having one)"
                )
            if any(later < earlier for earlier, later in zip(times, times[1:])):
                raise ValueError(f"trajectory {name!r}: vertex times decrease")
            if closed:
                raise ValueError(
                    f"trajectory {name!r}: a closed trajectory cannot carry times"
                )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "closed", closed)

    @property
    def is_timed(self) -> bool:
        """Whether every vertex carries a time."""
        return all(vertex.time is not None for vertex in self.vertices)

    @property
    def has_times(self) -> bool:
        """Whether any vertex carries a time (a time reference applies to)."""
        return any(vertex.time is not None for vertex in self.vertices)

    @property
    def is_relative(self) -> bool:
        """Whether any vertex is a :class:`RelativeLanePose` or a route pose.

        Such a trajectory is placed anew each time the action starts, since
        where it lies depends on where the reference entity -- or, for a route
        pose (:data:`ROUTE_POSITIONS`), the ego on its route -- then is.
        """
        # The classes spelled out, rather than unpacked from ROUTE_POSITIONS:
        # the static check (docs/typecheck.md) cannot unpack a tuple of
        # classes into isinstance().
        return any(
            isinstance(
                vertex.position,
                (
                    RelativeLanePose,
                    RouteLanePose,
                    RouteOppositePose,
                    RouteCrossingPose,
                    RouteCrosswalkPose,
                    RouteRoadsidePose,
                ),
            )
            for vertex in self.vertices
        )

    @property
    def is_gated(self) -> bool:
        """Whether any vertex is departed by a condition other than a time.

        A trajectory that is not gated moves by its times (or speed) alone,
        exactly as a trajectory always has.
        """
        return any(vertex.gate is not None for vertex in self.vertices)

    def gated(self, advance: Mapping[int, "BaseCondition"]) -> "Trajectory":
        """This trajectory with the vertices named departing on new conditions.

        A convenience for a trajectory built from somewhere else -- a
        recording, a lanelet route -- whose vertices are not written one by
        one: ``path.gated({3: ElapsedTimeCondition(20.0, label="go")})`` makes
        the fourth vertex wait until the scenario is 20 s old.  Each vertex
        named has its departure condition **replaced** -- a time included,
        since a vertex departs on one condition; the others keep theirs.

        Args:
            advance: Vertex index (0-based, negative counting from the end as
                for a list) to its new departure condition.

        Raises:
            IndexError: On an index the trajectory has no vertex for.
            ValueError: On a result the trajectory rules refuse.
        """
        vertices = list(self.vertices)
        count = len(vertices)
        for index, condition in advance.items():
            if not -count <= int(index) < count:
                raise IndexError(
                    f"trajectory {self.name!r} has no vertex {index} (it has {count})"
                )
            vertices[int(index)] = TrajectoryVertex(
                vertices[int(index)].position, condition
            )
        return Trajectory(self.name, vertices, self.closed)


class ReferenceContext(enum.Enum):
    """What a vertex time is measured from (OpenSCENARIO ``ReferenceContext``)."""

    #: From the start of the scenario: the clock pass/fail conditions report.
    ABSOLUTE = "absolute"
    #: From the moment the action starts.
    RELATIVE = "relative"


@dataclass(frozen=True)
class TrajectoryTiming:
    """How vertex times map onto the scenario's clock (OpenSCENARIO ``Timing``).

    A vertex at time ``τ`` is reached at scenario time ``τ * scale + offset``,
    measured from the start of the scenario (:attr:`ReferenceContext.ABSOLUTE`)
    or of the action (:attr:`ReferenceContext.RELATIVE`).

    Args:
        domain: What the times are measured from.
        scale: Factor applied to every vertex time; ``2.0`` plays a recording at
            half speed.  Must be finite and positive.
        offset: Seconds added after scaling.  Must be finite.

    Raises:
        ValueError: If *scale* is not finite and positive, or *offset* is
            not finite.
    """

    domain: ReferenceContext = ReferenceContext.RELATIVE
    scale: float = 1.0
    offset: float = 0.0

    def __post_init__(self) -> None:
        # NaN fails `> 0.0`, but infinity passes it and would freeze a timed
        # trajectory at its first vertex; a non-finite offset makes every
        # trajectory time NaN.
        if not math.isfinite(self.scale) or not self.scale > 0.0:
            raise ValueError(
                f"TrajectoryTiming.scale must be finite and positive, "
                f"not {self.scale!r}"
            )
        if not math.isfinite(self.offset):
            raise ValueError(
                f"TrajectoryTiming.offset must be finite, not {self.offset!r}"
            )

    def trajectory_time(self, scenario_time: float, action_start: float) -> float:
        """The vertex time that corresponds to *scenario_time*."""
        origin = 0.0 if self.domain is ReferenceContext.ABSOLUTE else action_start
        return (scenario_time - origin - self.offset) / self.scale


class TrajectoryFollowingMode(enum.Enum):
    """How closely the entity is held to the trajectory (OpenSCENARIO)."""

    #: The entity is placed on the trajectory every tick: a kinematic replay
    #: that reproduces the recording exactly and ignores vehicle dynamics.
    POSITION = "position"
    #: A controller steers and drives the entity along it, so the motion is
    #: the vehicle's own and may lag or cut a corner.
    FOLLOW = "follow"


@dataclass(frozen=True)
class TrajectorySample:
    """Where a resolved trajectory has the entity, in CARLA world coordinates.

    ``yaw`` is in degrees (CARLA's convention), the velocity in m/s along the
    CARLA world axes.
    """

    x: float
    y: float
    z: float
    yaw: float
    vx: float = 0.0
    vy: float = 0.0
    yaw_rate: float = 0.0

    @property
    def speed(self) -> float:
        """Ground speed in m/s."""
        return math.hypot(self.vx, self.vy)


def _finite_column(name: str, values: Sequence[float]) -> None:
    for index, value in enumerate(values):
        if not math.isfinite(value):
            raise ValueError(
                f"ResolvedTrajectory.{name}[{index}] must be finite, got {value}"
            )


@dataclass
class ResolvedTrajectory:
    """A trajectory whose vertices are CARLA world points.

    Built by the action from a :class:`Trajectory` once the map is loaded.
    Every query is plain arithmetic on the vertices, so this is where the
    interpolation is tested.

    Args:
        xs, ys, zs: Vertex positions in CARLA world coordinates (m).
        yaws: Vertex headings in CARLA degrees; ``None`` for a vertex whose
            heading is taken from the path.  Such a vertex is a corner: the
            entity turns from the arriving segment's direction to the leaving
            one's across a window around it, reaching
            :data:`_CORNER_HALF_WIDTH_M` along each segment or half of it if
            that is shorter, and the vertex itself faces half way round.
            Elsewhere along a segment it faces the segment's own direction,
            with a yaw rate of ``0`` -- a sparse path is driven along its
            legs, a dense one turns smoothly.  Given yaws interpolate between
            each other; between a given yaw and a vertex without one the yaw
            turns from the given heading to the segment's direction (or back)
            outside the corner window.  Segments that do not move (the same
            point written again) are looked past: the corner is between the
            nearest segments that move either side, its vertices all facing
            half way round (:meth:`_find_corners`) -- so a sample written
            twice, or a closed path that repeats its first vertex, turns as
            smoothly as one written once.  The first and last vertices of an
            open path, a vertex with a given yaw among them, and a **stop**
            (a timed trajectory staying on one point for
            :data:`_STOP_MIN_S` or longer) are not corners: they face along
            the path, a stop keeping the heading it arrived with, and a
            segment that does not move turns from one vertex's heading to the
            next's -- a stop turns on the spot.
        times: Vertex times, or ``None`` for an untimed trajectory.  On a
            closed trajectory they are the written vertices' times: the
            closing segment has none, so the clock starts at the first vertex
            facing along the first segment, and ends at the last written
            vertex facing along the segment arriving there, as an open
            trajectory's does (:meth:`at_time`); only along the path
            (:meth:`at_distance`, which wraps) does it turn at the seam.
        closed: Whether the last vertex joins back to the first.

    Raises:
        ValueError: The columns differ in length, there are fewer than two
            vertices, or a value (a position, a given yaw, a time) is not
            finite.
    """

    xs: Sequence[float]
    ys: Sequence[float]
    zs: Sequence[float]
    yaws: Sequence[Optional[float]]
    times: Optional[Sequence[float]] = None
    closed: bool = False
    _arc: list[float] = field(init=False, repr=False)
    _headings: list[float] = field(init=False, repr=False)
    _given: list[bool] = field(init=False, repr=False)
    _corner_before: list[float] = field(init=False, repr=False)
    _corner_after: list[float] = field(init=False, repr=False)
    _corner_turn: list[float] = field(init=False, repr=False)
    _corner_share: list[float] = field(init=False, repr=False)
    _clock_headings: list[float] = field(init=False, repr=False)
    _last: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        count = len(self.xs)
        if not count == len(self.ys) == len(self.zs) == len(self.yaws):
            raise ValueError("ResolvedTrajectory: vertex columns differ in length")
        if count < 2:
            raise ValueError("ResolvedTrajectory needs at least two vertices")
        if self.times is not None and len(self.times) != count:
            raise ValueError("ResolvedTrajectory: times and vertices differ in length")
        self.xs = [float(value) for value in self.xs]
        self.ys = [float(value) for value in self.ys]
        self.zs = [float(value) for value in self.zs]
        if self.times is not None:
            self.times = [float(value) for value in self.times]
        # One NaN vertex would make the length, and so every sample, NaN --
        # the stretches between finite vertices too.  The poses refuse
        # non-finite fields where they are written; this catches whatever a
        # resolution or a computed path lets through.
        _finite_column("xs", self.xs)
        _finite_column("ys", self.ys)
        _finite_column("zs", self.zs)
        for index, yaw in enumerate(self.yaws):
            if yaw is not None:
                if not math.isfinite(float(yaw)):
                    raise ValueError(
                        f"ResolvedTrajectory.yaws[{index}] must be finite, got {yaw}"
                    )
        times = self.times
        if times is not None:
            _finite_column("times", times)
        # The last vertex as written: where the clock of a timed trajectory
        # ends, a closed one's too, whose closing segment has no time.
        self._last = count - 1
        if self.closed:
            # The closing segment is an ordinary one, so the distance queries
            # need no special case for it.
            self.xs.append(self.xs[0])
            self.ys.append(self.ys[0])
            self.zs.append(self.zs[0])
            self.yaws = [*self.yaws, self.yaws[0]]
        arc = [0.0]
        for index in range(1, len(self.xs)):
            arc.append(
                arc[-1]
                + math.hypot(
                    self.xs[index] - self.xs[index - 1],
                    self.ys[index] - self.ys[index - 1],
                )
            )
        self._arc = arc
        self._given = [yaw is not None for yaw in self.yaws]
        self._headings = self._fill_headings()
        self._find_corners()

    # ------------------------------------------------------------------
    # Shape
    # ------------------------------------------------------------------

    @property
    def length(self) -> float:
        """Path length in metres, the closing segment included."""
        return self._arc[-1]

    def at_vertex(self, index: int) -> TrajectorySample:
        """Vertex *index* itself, at rest, with the heading it resolved to."""
        return self._vertex(index)

    def vertex_distance(self, index: int) -> float:
        """How far along the path (m) vertex *index* is.

        The index is the vertex's in the trajectory as written; on a closed
        trajectory the first vertex is at ``0`` (and again at :attr:`length`,
        a lap later).
        """
        return self._arc[index]

    @property
    def start_time(self) -> float:
        """The first vertex's time (``0.0`` for an untimed trajectory)."""
        return float(self.times[0]) if self.times is not None else 0.0

    @property
    def end_time(self) -> float:
        """The last vertex's time (``0.0`` for an untimed trajectory)."""
        return float(self.times[-1]) if self.times is not None else 0.0

    def start(self) -> TrajectorySample:
        """The first vertex, at rest.

        On a closed timed trajectory, whose clock starts there and never
        drives the closing segment, it faces along the first segment; untimed,
        half way round the seam.
        """
        return self._at_rest(0)

    def end(self) -> TrajectorySample:
        """Where the trajectory leaves the entity, at rest.

        The last vertex -- on a closed untimed trajectory the first one again,
        where the closing segment ends.  A timed trajectory's clock ends at the
        last vertex written, so on a closed one that is where it stays, facing
        along the segment arriving there: the closing segment has no time
        (:meth:`at_time`).
        """
        if self.times is not None:
            return self._at_rest(self._last)
        return self._at_rest(len(self.xs) - 1)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def at_time(self, time: float) -> TrajectorySample:
        """Where the entity is at vertex time *time*.

        Before the first vertex it waits there, after the last it stays there,
        at rest in both cases.  The velocity is the segment's own, so it is
        what moving from one vertex to the next at the recorded times takes.
        On a closed trajectory the times are the written vertices': the
        closing segment has none, so at the last written vertex's time the
        entity is there, and stays there -- the clock does not lap (a closed
        path is driven round by :meth:`at_distance`, which wraps).

        Raises:
            ValueError: On an untimed trajectory.
        """
        if self.times is None:
            raise ValueError("at_time needs a timed trajectory")
        times = self.times
        last = self._last
        if time <= times[0]:
            return self._at_rest(0)
        if time >= times[last]:
            return self._at_rest(last)
        index = bisect.bisect_right(times, time) - 1
        duration = float(times[index + 1]) - float(times[index])
        if duration < _EPSILON_S:
            return self._at_rest(index + 1)
        fraction = (time - float(times[index])) / duration
        return self._between(index, fraction, duration)

    def at_distance(self, distance: float, speed: float = 0.0) -> TrajectorySample:
        """Where the entity is *distance* metres along the path, moving at *speed*.

        A closed trajectory wraps around; an open one clamps to its ends.
        """
        if self.closed and self.length > _EPSILON_M:
            distance = distance % self.length
        if distance <= 0.0:
            index, fraction = 0, 0.0
        elif distance >= self.length:
            index, fraction = len(self._arc) - 2, 1.0
        else:
            index = bisect.bisect_right(self._arc, distance) - 1
            span = self._arc[index + 1] - self._arc[index]
            fraction = (
                0.0 if span < _EPSILON_M else (distance - self._arc[index]) / span
            )
        sample = self._between(index, fraction, None)
        heading = math.radians(sample.yaw)
        direction = self._segment_heading(index)
        if direction is not None:
            heading = direction
        return TrajectorySample(
            sample.x,
            sample.y,
            sample.z,
            sample.yaw,
            vx=speed * math.cos(heading),
            vy=speed * math.sin(heading),
        )

    def distance_at_time(self, time: float) -> float:
        """How far along the path the entity is at vertex time *time*.

        From the last vertex's time on, the last written vertex's distance --
        the length, except on a closed trajectory, whose closing segment has no
        time (:meth:`at_time`).
        """
        if self.times is None:
            raise ValueError("distance_at_time needs a timed trajectory")
        times = self.times
        last = self._last
        if time <= times[0]:
            return 0.0
        if time >= times[last]:
            return self._arc[last]
        index = bisect.bisect_right(times, time) - 1
        duration = times[index + 1] - times[index]
        fraction = 0.0 if duration < _EPSILON_S else (time - times[index]) / duration
        return self._arc[index] + fraction * (self._arc[index + 1] - self._arc[index])

    def time_at_distance(self, distance: float) -> float:
        """The vertex time at which the entity is *distance* metres along.

        At or past the last written vertex, its time: on a closed trajectory
        the closing segment has none, and the distance does not wrap.
        """
        if self.times is None:
            raise ValueError("time_at_distance needs a timed trajectory")
        last = self._last
        if distance <= 0.0:
            return float(self.times[0])
        if distance >= self._arc[last]:
            return float(self.times[last])
        index = bisect.bisect_right(self._arc, distance) - 1
        span = self._arc[index + 1] - self._arc[index]
        fraction = 0.0 if span < _EPSILON_M else (distance - self._arc[index]) / span
        return float(self.times[index]) + fraction * (
            float(self.times[index + 1]) - float(self.times[index])
        )

    def project(
        self, x: float, y: float, near: Optional[float] = None, window: float = 30.0
    ) -> float:
        """The distance along the path of the point nearest ``(x, y)``.

        Args:
            x, y: The point, in CARLA world coordinates.
            near: Where along the path to look, so that a path which passes the
                same place twice (a loop, a U-turn) answers with the pass the
                entity is on.  ``None`` searches the whole path.
            window: How far either side of *near* to look (m).  On a closed
                path the window wraps: past the end it goes on from the start,
                and before the start from the end, so an entity crossing the
                first vertex onto its next lap is found there.  The answer is
                in ``[0, length]`` either way.
        """
        length = self.length
        loop = self.closed and length > _EPSILON_M
        if near is None or (loop and 2.0 * window >= length):
            # The whole path; on a loop, also when the window covers it.
            return self._nearest(x, y, 0, len(self._arc) - 1)[1]
        if not loop:
            first, stop = self._window(near - window, near + window)
            return self._nearest(x, y, first, stop)[1]
        # A closed path: the window, its centre taken modulo the length, and
        # whatever of it falls before the start or past the end searched on
        # the other side of the first vertex.
        centre = near % length
        if centre < 0.0:
            centre += length
        low, high = centre - window, centre + window
        first, stop = self._window(low, high)
        best = self._nearest(x, y, first, stop)
        if low < 0.0:
            first, stop = self._window(low + length, length)
            wrapped = self._nearest(x, y, first, stop)
            if wrapped[0] < best[0]:
                best = wrapped
        if high > length:
            first, stop = self._window(0.0, high - length)
            wrapped = self._nearest(x, y, first, stop)
            if wrapped[0] < best[0]:
                best = wrapped
        return best[1]

    def _window(self, low: float, high: float) -> tuple[int, int]:
        """The segments ``[first, stop)`` that reach into ``[low, high]`` (m)
        along the path: from the one ending at or past *low* to the last one
        starting at or before *high* (_arc does not decrease)."""
        first = max(bisect.bisect_left(self._arc, low) - 1, 0)
        stop = min(bisect.bisect_right(self._arc, high), len(self._arc) - 1)
        return first, stop

    def _nearest(
        self, x: float, y: float, first: int, stop: int
    ) -> tuple[float, float]:
        """``(gap, along)``: the point of segments ``[first, stop)`` nearest
        ``(x, y)``, how far it is, and how far along the path."""
        best_distance = math.inf
        best_along = 0.0
        for index in range(first, stop):
            x0, y0 = self.xs[index], self.ys[index]
            dx, dy = self.xs[index + 1] - x0, self.ys[index + 1] - y0
            squared = dx * dx + dy * dy
            fraction = (
                0.0
                if squared < _EPSILON_M
                else min(max(((x - x0) * dx + (y - y0) * dy) / squared, 0.0), 1.0)
            )
            gap = math.hypot(x0 + fraction * dx - x, y0 + fraction * dy - y)
            if gap < best_distance:
                best_distance = gap
                best_along = self._arc[index] + fraction * math.sqrt(squared)
        return best_distance, best_along

    def vertices(self) -> list[TrajectorySample]:
        """Every vertex, with the heading it resolved to and its times' velocity.

        On a timed trajectory each is what :meth:`at_time` gives at the
        vertex's time, leaving it; a vertex with no timed segment after it
        (the last written one, and on a closed trajectory the closing vertex)
        at rest.
        """
        samples: list[TrajectorySample] = []
        for index in range(len(self.xs)):
            times = self.times
            if times is not None and index + 1 < len(times):
                duration = float(times[index + 1]) - float(times[index])
                if duration >= _EPSILON_S:
                    samples.append(self._between(index, 0.0, duration))
                    continue
            samples.append(self._at_rest(index))
        return samples

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _vertex(self, index: int) -> TrajectorySample:
        return TrajectorySample(
            self.xs[index], self.ys[index], self.zs[index], self._headings[index]
        )

    def _at_rest(self, index: int) -> TrajectorySample:
        """Vertex *index* at rest, facing as the clock has it there.

        That is its resolved heading, but within a corner that turns through
        a pause (:meth:`_find_corners`), where the clock spends part of the
        turn on the spot; untimed, the two are the same.
        """
        return TrajectorySample(
            self.xs[index], self.ys[index], self.zs[index], self._clock_headings[index]
        )

    def _between(
        self, index: int, fraction: float, duration: Optional[float]
    ) -> TrajectorySample:
        """The point *fraction* of the way along segment *index*.

        Along a segment that moves the yaw is its direction, but for three
        stretches: within a corner's window at either end (a vertex without a
        yaw of its own, :meth:`_find_corners`) it turns half the corner's turn,
        from the vertex's heading to the segment's direction or back; and
        between the windows, where an end has a given yaw, it turns from that
        yaw to the segment's direction or back (between both, if both ends have
        one).  It never uses a derived vertex heading outside a window, which
        would be the next segment's direction and turn the entity across the
        whole segment.  A segment that does not move has no direction, and
        turns from one vertex's heading to the next's (a turn on the spot, or
        nothing, where a stop kept its heading).

        With a *duration* the velocity is the segment's, and the yaw rate the
        rate of that turn at the segment's speed.  The *duration* also says
        the sample is the clock's (:meth:`at_time`): within a corner that turns
        through a pause the clock spends a share of the turn on the spot, the
        windows the rest (``_corner_share`` each, ``_clock_headings`` at the
        vertices), where a distance along the path, which has no time to spend
        there, turns half in each window.
        """
        x0, y0, z0 = self.xs[index], self.ys[index], self.zs[index]
        x1, y1, z1 = self.xs[index + 1], self.ys[index + 1], self.zs[index + 1]
        span = self._arc[index + 1] - self._arc[index]
        direction = self._segment_heading(index)
        # The yaw at *fraction*, and its rate of change per unit of fraction.
        yaw = 0.0
        slope = 0.0
        clock = duration is not None
        if direction is None:
            headings = self._clock_headings if clock else self._headings
            slope = _wrap_degrees(headings[index + 1] - headings[index])
            yaw = headings[index] + fraction * slope
        else:
            along = math.degrees(direction)
            # Each at most half the segment, so the two never overlap.
            start_window = self._corner_after[index] / span
            end_window = self._corner_before[index + 1] / span
            if fraction < start_window:
                # Leaving a corner: from its share short of the leaving
                # segment's way (half way round, untimed) to that way.
                share = self._corner_share[index] if clock else 0.5
                slope = share * self._corner_turn[index] / start_window
                yaw = along + (fraction - start_window) * slope
            elif end_window > 0.0 and fraction >= 1.0 - end_window:
                # Arriving at a corner: from the segment's way to its share
                # round (half way, untimed).
                share = self._corner_share[index + 1] if clock else 0.5
                slope = share * self._corner_turn[index + 1] / end_window
                yaw = along + (fraction - (1.0 - end_window)) * slope
            else:
                first = self._headings[index] if self._given[index] else along
                last = self._headings[index + 1] if self._given[index + 1] else along
                # A turn here means a given end, which has no window, and the
                # other end's window is at most half: the stretch is at least
                # half the segment.  With neither end given there is no turn.
                stretch = 1.0 - start_window - end_window
                turn = _wrap_degrees(last - first)
                if turn != 0.0 and stretch > 0.0:
                    slope = turn / stretch
                    yaw = first + (fraction - start_window) * slope
                else:
                    yaw = first
        yaw = _wrap_degrees(yaw)
        if duration is None:
            vx = vy = yaw_rate = 0.0
        else:
            vx, vy = (x1 - x0) / duration, (y1 - y0) / duration
            yaw_rate = slope / duration
        return TrajectorySample(
            x0 + fraction * (x1 - x0),
            y0 + fraction * (y1 - y0),
            z0 + fraction * (z1 - z0),
            yaw,
            vx=vx,
            vy=vy,
            yaw_rate=yaw_rate,
        )

    def _segment_heading(self, index: int) -> Optional[float]:
        """Segment *index*'s direction in radians, or ``None`` if it has none."""
        dx = self.xs[index + 1] - self.xs[index]
        dy = self.ys[index + 1] - self.ys[index]
        if math.hypot(dx, dy) < _EPSILON_M:
            return None
        return math.atan2(dy, dx)

    def _find_corners(self) -> None:
        """Every vertex's corner window, and the vertex headings at corners.

        A corner is where the path turns from one segment that moves to the
        next that moves, looking past any segments between them that do not
        (the same point written again): a vertex without a yaw of its own
        between two moving segments, or a run of such vertices on one point.
        On a closed path the first vertex is one too, between the closing
        segment and the first -- the same point as the closing vertex, which
        mirrors it.  A run with a given yaw is not a corner, nor, on a timed
        trajectory, one that stays put for :data:`_STOP_MIN_S` or longer: a
        stop, which keeps the heading it arrived with and turns on the spot
        (:meth:`_fill_headings`).

        The window reaches back along the arriving segment
        :data:`_CORNER_HALF_WIDTH_M` or half of it, whichever is shorter, and
        on along the leaving one the same way, so two windows never overlap
        and on a dense path they cover it all.  The entity turns from the
        arriving segment's direction to the leaving one's across the window,
        and the corner's vertices face half way round: half the turn before
        them, half after.  On a path sampled at even times (a recording) each
        half then takes half a sample's time, so the yaw rate is the turn over
        the time across the window, as steady as the curve -- unlike a split
        in proportion to each half's length, which would swing with uneven
        spacing.

        A corner that pauses (a timed run shorter than a stop: a sample
        written twice) turns on the clock through the pause as well: the
        moving windows keep their half-and-half split of the time they take
        between them, and the pause turns the rest, so the turn is spread over
        the window's time evenly -- a recording resampled onto a faster tick
        turns at its own rate, not in bursts with none between.  Along the
        path the pause takes no distance, so there the windows still turn half
        each; at a vertex of such a corner :meth:`at_time` and
        :meth:`at_distance` differ by the clock's share of the turn spent on
        the spot.  The pause is a difference of the run's times, and one
        within :data:`_EPSILON_S` of :data:`_STOP_MIN_S` is a stop.

        A timed closed path's clock never drives its closing segment, so on
        the clock a corner that segment is part of -- at the first vertex, at
        the last one written, or a run across the seam -- does not turn: its
        windows face along their own segments (``_corner_share`` ``0``), and
        its vertices along the timed segment on their side, the first vertex
        along the first segment and the last written one along the segment
        arriving there, as an open path's ends.  Along the path, which wraps,
        it is a corner like any other.

        ``_corner_before`` and ``_corner_after`` hold each vertex's reach (m)
        back and on (``0`` for none), ``_corner_turn`` the turn (degrees,
        positive towards +y: clockwise seen from above, CARLA's sense of yaw),
        ``_corner_share`` the share of it each window turns on the clock, and
        ``_clock_headings`` the vertex headings on the clock.
        """
        count = len(self.xs)
        segments = count - 1
        corner_before = [0.0] * count
        corner_after = [0.0] * count
        corner_turn = [0.0] * count
        corner_share = [0.5] * count
        clock = list(self._headings)
        for arriving in range(segments):
            before = self._segment_heading(arriving)
            if before is None:
                continue
            # The next segment that moves, past any that do not; on a closed
            # path round the seam, but never back to this one.
            limit = segments - 1 if self.closed else segments - 1 - arriving
            leaving = -1
            reach = 0
            for offset in range(1, limit + 1):
                candidate = (arriving + offset) % segments
                if self._segment_heading(candidate) is not None:
                    leaving = candidate
                    reach = offset
                    break
            if leaving < 0:
                continue
            after = self._segment_heading(leaving)
            if after is None:
                continue
            # The corner's vertices, from the arriving segment's end to the
            # leaving one's start; the segments between them do not move.
            run: list[int] = []
            given = False
            for offset in range(1, reach + 1):
                vertex = (arriving + offset) % segments
                run.append(vertex)
                if self._given[vertex]:
                    given = True
            if given:
                continue
            # How long the run stays put, each stretch of it as one difference
            # of times, so a stop written as several pauses is not a hair
            # short of _STOP_MIN_S.  On a closed path the closing segment,
            # which has no time, splits a run across the seam in two.
            seam = segments - 1
            crosses = self.closed and seam in run[:-1]
            if crosses:
                pause = self._elapsed(run[0], seam) + self._elapsed(0, run[-1])
            else:
                pause = self._elapsed(run[0], run[-1])
            if pause >= _STOP_MIN_S - _EPSILON_S:
                continue
            # A timed closed path's clock never drives the closing segment:
            # it starts at the first vertex and stops at the last one written,
            # so on the clock a corner the closing segment is part of is no
            # corner -- each side faces along its own timed segment, as an
            # open path's ends do.
            untimed = (
                self.closed
                and self.times is not None
                and (arriving == seam or leaving == seam or crosses)
            )
            turn = _wrap_degrees(math.degrees(after) - math.degrees(before))
            span_in = self._arc[arriving + 1] - self._arc[arriving]
            span_out = self._arc[leaving + 1] - self._arc[leaving]
            reach_in = min(_CORNER_HALF_WIDTH_M, 0.5 * span_in)
            reach_out = min(_CORNER_HALF_WIDTH_M, 0.5 * span_out)
            # The time each window takes, and the whole corner's.
            moving = (reach_in / span_in) * self._segment_duration(arriving) + (
                reach_out / span_out
            ) * self._segment_duration(leaving)
            total = moving + pause
            share = 0.5
            if untimed:
                share = 0.0
            elif total >= _EPSILON_S:
                share = 0.5 * moving / total
            middle = _wrap_degrees(math.degrees(before) + 0.5 * turn)
            # On the clock, past the seam yet (untimed corners only).
            past = arriving == seam
            for position in range(len(run)):
                vertex = run[position]
                if position > 0 and run[position - 1] == seam:
                    past = True
                heading = middle
                if untimed:
                    heading = math.degrees(after) if past else math.degrees(before)
                elif total >= _EPSILON_S:
                    done = share + self._elapsed(run[0], vertex) / total
                    heading = math.degrees(before) + done * turn
                indices = [vertex]
                if self.closed and vertex == 0:
                    indices.append(count - 1)
                for index in indices:
                    corner_turn[index] = turn
                    corner_share[index] = share
                    self._headings[index] = middle
                    clock[index] = _wrap_degrees(heading)
                    if position == 0:
                        corner_before[index] = reach_in
                    if position == len(run) - 1:
                        corner_after[index] = reach_out
        self._corner_before = corner_before
        self._corner_after = corner_after
        self._corner_turn = corner_turn
        self._corner_share = corner_share
        self._clock_headings = clock

    def _elapsed(self, first: int, last: int) -> float:
        """The time (s) from vertex *first* to vertex *last*; ``0`` untimed,
        or where either has no time (a closed trajectory's closing vertex)."""
        times = self.times
        if times is None or first >= len(times) or last >= len(times):
            return 0.0
        return float(times[last]) - float(times[first])

    def _segment_duration(self, index: int) -> float:
        """Segment *index*'s duration (s); ``0`` untimed, and for a closed
        trajectory's closing segment, which has no time."""
        times = self.times
        if times is None or index + 1 >= len(times):
            return 0.0
        return float(times[index + 1]) - float(times[index])

    def _fill_headings(self) -> list[float]:
        """Every vertex's heading in CARLA degrees.

        A vertex's own heading wins.  One without takes the direction of the
        path there: the segment leaving it, or the one arriving at the last
        vertex.  A vertex where the path does not move (a stop) takes the
        heading of the nearest vertex before it that has one, so a vehicle that
        stopped does not spin to face East.
        """
        count = len(self.xs)
        headings: list[Optional[float]] = []
        for index in range(count):
            given = self.yaws[index]
            if given is not None:
                headings.append(float(given))
                continue
            direction = self._segment_heading(index) if index + 1 < count else None
            if direction is None and index > 0:
                direction = self._segment_heading(index - 1)
            headings.append(None if direction is None else math.degrees(direction))
        filled: list[float] = []
        previous: Optional[float] = None
        for heading in headings:
            if heading is None:
                heading = previous
            filled.append(heading if heading is not None else math.nan)
            previous = heading if heading is not None else previous
        # A path that does not move at all before its first heading: backfill.
        first = next((value for value in filled if not math.isnan(value)), 0.0)
        return [first if math.isnan(value) else value for value in filled]


def _wrap_degrees(angle: float) -> float:
    """*angle* wrapped into ``[-180, 180)``."""
    wrapped = (angle + 180.0) % 360.0 - 180.0
    # Rounding can land a value just below -180 on +180 itself.
    return -180.0 if wrapped >= 180.0 else wrapped
