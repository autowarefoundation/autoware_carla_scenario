"""What drives the traffic of a run, and the vocabulary it answers.

A scenario has three kinds of vehicle in it, and only the first two are the
scenario's own:

* the **ego**, driven by its entity (TrafficManager, Autoware, a driver policy),
* the **authored NPCs** a scenario spawns in :meth:`BaseScenario.setup`,
* the **ambient traffic** that fills the rest of the road network.

A *traffic backend* owns the third kind outright and answers manoeuvre intents
asked of the second.  CARLA's TrafficManager is one such backend, and until this
module existed it was the only one -- reached directly from
:class:`~autoware_carla_scenario.ScenarioRunner` and from the entity mixin, which
is what made a microscopic traffic simulator (SUMO and the like) impossible to
put in its place.

The seam is the same one the package already uses for scenarios
(:mod:`autoware_carla_scenario.registry`) and for the ego
(``ego.entity``): a name, a factory, and a small interface whose methods the
runner calls at fixed points of the run.  Nothing here imports CARLA or any
traffic simulator, so the config layer and the editor can import it as cheaply as
they import :mod:`autoware_carla_scenario.scenario_config`.

Two rules keep the design honest, and both are enforced rather than hoped for:

**One authority per vehicle.**  A vehicle a backend owns is not autopiloted by
another, which is why :meth:`TrafficBackend.start` takes the actors it must keep
its hands off.

**An intent is not a mechanism.**  ``change_lane`` is what the *scenario* means;
``tm.force_lane_change`` and ``traci.vehicle.changeLane`` are what two backends
do about it.  Entities ask for the first and never name the second.
"""

from __future__ import annotations

import enum
import logging
import math
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Collection,
    List,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

from ..constants import MIN_BACKGROUND_SPEED_KMH
from .context import TrafficContext
from .parameters import (
    BICYCLE_PARAMETER_BACKENDS,
    PEDESTRIAN_PARAMETER_BACKENDS,
    VEHICLE_PARAMETER_BACKENDS,
    BicycleParameters,
    PedestrianParameters,
    VehicleParameters,
)

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

    from ..coordinate.poses import Lanelet2Pose

logger = logging.getLogger(__name__)

#: The parameters of a backend that was given none: every field unset.
_NO_VEHICLE_PARAMETERS = VehicleParameters()
_NO_PEDESTRIAN_PARAMETERS = PedestrianParameters()
_NO_BICYCLE_PARAMETERS = BicycleParameters()

__all__ = [
    "LaneChangeDirection",
    "SimulationFidelity",
    "TurnDirection",
    "LaneChanging",
    "SettingSpeed",
    "TurningAtJunctions",
    "TrafficBackend",
    "warn_if_mesoscopic",
    "TrafficBackendError",
    "TrafficBackendUnavailable",
    "TrafficContext",
    "DrivenEntity",
    "WalkedEntity",
    "NullTrafficBackend",
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TrafficBackendError(RuntimeError):
    """A traffic backend could not do what the run needs of it."""


class TrafficBackendUnavailable(TrafficBackendError):
    """A backend was selected that this installation cannot provide.

    Raised when the simulator behind a backend is not installed, so the message
    can name the extra that installs it.  Distinct from
    :class:`TrafficBackendError` because it is a setup problem, not a run
    problem, and it is raised before anything is spawned.
    """


# ---------------------------------------------------------------------------
# Background traffic: what every backend's spawn_background agrees on (#967)
# ---------------------------------------------------------------------------


def check_background_speed(speed_kmh: float) -> float:
    """*speed_kmh* as a background vehicle's speed cap, in km/h.

    Raises:
        ValueError: If it is not finite or below
            :data:`~autoware_carla_scenario.constants.MIN_BACKGROUND_SPEED_KMH`.
    """
    speed = float(speed_kmh)
    if not math.isfinite(speed) or speed < MIN_BACKGROUND_SPEED_KMH:
        raise ValueError(
            f"speed_kmh must be a finite speed of at least "
            f"{MIN_BACKGROUND_SPEED_KMH:g} km/h, not {speed_kmh!r}: it is the "
            "most a background vehicle drives at"
        )
    return speed


# ---------------------------------------------------------------------------
# The vocabulary
# ---------------------------------------------------------------------------


class LaneChangeDirection(enum.Enum):
    """Direction of a lane change.

    Defined with the traffic seam rather than in the action that asks for one:
    the action names an intent, and a backend is what knows what the intent
    means to whatever is driving.  The ``traffic`` package re-exports it, which is
    where the actions, scenario authors and the editor reach it.
    """

    LEFT = "left"
    RIGHT = "right"

    def to_carla_bool(self) -> bool:
        """Convert to the boolean expected by ``TrafficManager.force_lane_change``.

        CARLA convention: ``True`` -> right, ``False`` -> left.
        """
        return self is LaneChangeDirection.RIGHT


class TurnDirection(enum.Enum):
    """Direction of a turn at a junction."""

    LEFT = "left"
    RIGHT = "right"
    #: Straight on through the junction: what a vehicle left to choose for
    #: itself (the TrafficManager's autopilot) may not do.
    STRAIGHT = "straight"


class SimulationFidelity(enum.Enum):
    """How finely a traffic backend simulates a vehicle at the moment.

    A backend may simulate the traffic far from anything that matters at a
    coarser level than the traffic near it, because nobody there can see the
    difference and the run is cheaper for it.  Two levels, named as traffic
    engineering names them:

    * **Microscopic** -- each vehicle on its own lane, following its leader and
      changing lanes, and present in CARLA as an actor.  Every vehicle is this
      under a backend with one level, which is all of them but one.
    * **Mesoscopic** -- the vehicle is a place in a queue on a road: when it
      reaches the end of the road is modelled, where on it is not, and it is
      *not* in the CARLA world.  An entity that is mesoscopic has no actor
      (``entity.actor`` is ``None``) until its backend makes it microscopic
      again.
    """

    MICROSCOPIC = "microscopic"
    MESOSCOPIC = "mesoscopic"


#: Role names of the entities a traffic backend simulates mesoscopically now
#: (kept by :meth:`TrafficBackend._mark_fidelity`), and those a lookup of their
#: actor has been warned about since they became mesoscopic.
_MESOSCOPIC_ROLES: set[str] = set()
_WARNED_ROLES: set[str] = set()


def warn_if_mesoscopic(role_name: str) -> bool:
    """Whether *role_name* names an entity a backend simulates mesoscopically.

    Called by a lookup that found no actor for *role_name*: such an entity has
    none while it is far, which is not the entity being gone, and a condition
    that reads it reads nothing.  Says so in a warning, the first time after
    the entity became mesoscopic.
    """
    if role_name not in _MESOSCOPIC_ROLES:
        return False
    if role_name not in _WARNED_ROLES:
        _WARNED_ROLES.add(role_name)
        logger.warning(
            "%s has no actor: its traffic backend simulates it mesoscopically "
            "while it is far, out of CARLA. A condition reading it reads "
            "nothing meanwhile; check entity.simulation_fidelity, or call "
            "entity.require_microscopic(world) to bring it back.",
            role_name,
        )
    return True


@runtime_checkable
class LaneChanging(Protocol):
    """What :class:`~autoware_carla_scenario.actions.lane_change.LaneChangeAction`
    needs of an entity.

    Stated as a protocol rather than a base class because the action does not
    care what performs the manoeuvre -- only that something can be asked for one
    and asked whether it is done.
    """

    def change_lane(
        self, world: "carla.World", direction: "LaneChangeDirection"
    ) -> None:
        """Move one lane in *direction*."""
        ...

    def lane_change_finished(self, world: "carla.World") -> bool:
        """Whether the manoeuvre has settled."""
        ...


@runtime_checkable
class SettingSpeed(Protocol):
    """What :class:`~autoware_carla_scenario.actions.set_speed.SetSpeedAction`
    needs of an entity.

    One call, and a target rather than a transition.  A rate-limited change is
    the action walking that target towards its goal a tick at a time, so a
    traffic model which is not the TrafficManager needs to understand only
    "drive at this speed" and never a transition model of its own.

    The corollary is that this seam carries no rate, so a backend cannot honour
    one natively however it drives: a rate is always walked by the action, and
    therefore always reissued.  What *is* this backend's business is whether an
    immediate target survives being sent once -- ``set_desired_speed`` holds the
    last value it was given, and a backend that keeps nothing says so through
    the action's *reissue* argument rather than here.
    """

    def set_desired_speed(self, world: "carla.World", speed_kmh: float) -> None:
        """Aim for *speed_kmh* from now on."""
        ...


@runtime_checkable
class TurningAtJunctions(Protocol):
    """What :class:`~autoware_carla_scenario.actions.turn.TurnAction` needs."""

    def turn_at_junction(
        self, world: "carla.World", direction: "TurnDirection", **kwargs: object
    ) -> None:
        """Go *direction* at the next junction ahead."""
        ...


class DrivenEntity:
    """What a backend's manoeuvres take as the *entity* they steer.

    The base of :class:`~autoware_carla_scenario.traffic.driven.BackendDriven`,
    which the entities mix in, and defined here rather than there because the
    interface below names it: it is the type of every *entity* parameter of
    :class:`TrafficBackend`.  It holds the one thing a backend keeps on a
    vehicle: the state of that vehicle's lane change.

    A backend reads the vehicle's ``actor`` and ``role_name`` where the entity
    has them (the entities define both) and tests for them with ``hasattr``,
    so a stand-in that lacks one is answered as a vehicle with no actor yet.
    """

    #: The lane a backend aimed this vehicle at, and the map it was read from.
    #: Written by whichever backend performs the lane change and read by
    #: whichever one judges it finished -- state of *this vehicle's* manoeuvre,
    #: so it lives on the vehicle: a backend is shared by the whole run, and the
    #: entity is the one thing there is exactly one of per manoeuvre.
    _lane_change_target: Optional[Tuple[int, int]] = None
    _lane_change_map: Optional["carla.Map"] = None
    #: The waypoint of that lane beside where the change was asked for, and
    #: the ``(road_id, lane_id)`` of every lane its successors reach within
    #: ``_lane_change_horizon_m``: lane ids are scoped to a road, so a vehicle
    #: that settles after the target lane has run onto the next road is on
    #: one of these, not on ``_lane_change_target``.  The keys are walked
    #: when first needed and again only when the vehicle is further on.
    _lane_change_waypoint: Optional["carla.Waypoint"] = None
    _lane_change_ahead: Optional[set[Tuple[int, int]]] = None
    _lane_change_horizon_m: float = 0.0
    #: The same for the lane the vehicle was on when the change was asked
    #: for: a lane both lead to (where they merge, or past a junction) is no
    #: sign the vehicle changed lane, so it counts only once the vehicle has
    #: been seen on a lane only the target leads to (``_lane_change_crossed``).
    _lane_change_origin: Optional["carla.Waypoint"] = None
    _lane_change_origin_ahead: Optional[set[Tuple[int, int]]] = None
    _lane_change_crossed: bool = False

    #: How the backend driving this vehicle simulates it now
    #: (:class:`SimulationFidelity`).  Written by that backend alone, through
    #: :meth:`TrafficBackend._mark_fidelity`, and only once a change of level
    #: is complete: a vehicle reading ``MESOSCOPIC`` has no actor in the world,
    #: one reading ``MICROSCOPIC`` has one.  Read by anyone -- a condition about
    #: a vehicle that may leave the world asks this before its actor.
    simulation_fidelity: SimulationFidelity = SimulationFidelity.MICROSCOPIC

    def forget_lane_change(self) -> None:
        """Drop the lane change state, and the map it holds: a finished run's
        scenarios stay in the queue, and their entities with them."""
        self._lane_change_target = None
        self._lane_change_map = None
        self._lane_change_waypoint = None
        self._lane_change_ahead = None
        self._lane_change_horizon_m = 0.0
        self._lane_change_origin = None
        self._lane_change_origin_ahead = None
        self._lane_change_crossed = False

    def _replace_actor(self, actor: Optional["carla.Actor"]) -> bool:
        """Let a backend take this vehicle's actor out of the world, or give it
        a new one: *actor* is ``None`` when the vehicle becomes mesoscopic.

        Answers whether the entity accepted.  The base does not -- an entity
        that owns its actor in a way a backend cannot see (an ego with a
        bridge, a vehicle under a driver policy) must stay in the world -- so a
        backend keeps a vehicle microscopic unless its entity says otherwise.
        The entity only swaps the reference; spawning and destroying the actor
        is the backend's, which knows where the vehicle is.
        """
        return False


class WalkedEntity:
    """What a backend's pedestrian calls take as the *entity* they walk.

    The base of :class:`~autoware_carla_scenario.entity.pedestrian_entity.PedestrianEntity`,
    as :class:`DrivenEntity` is of the vehicles: the type of every *entity*
    parameter of :class:`TrafficBackend`'s pedestrian calls
    (:meth:`TrafficBackend.adopt_pedestrian`, :meth:`TrafficBackend.walk_straight`,
    :meth:`TrafficBackend.release_pedestrian`).  A backend keeps nothing on a
    pedestrian, so it holds nothing; it reads the walker's ``actor`` and
    ``role_name`` where the entity has them, as it reads a vehicle's.
    """


# ---------------------------------------------------------------------------
# The interface
# ---------------------------------------------------------------------------


class TrafficBackend:
    """Base class for whatever drives a run's traffic.

    Every method has a working default, so a backend implements only what it
    actually does: the lifecycle hooks are no-ops, and the manoeuvres report
    that they are not available.  That is deliberate -- a backend which cannot
    force a lane change should say so once, in a log line naming itself, rather
    than force every subclass to write the same refusal.

    The lifecycle mirrors the ego entity's hooks
    (:class:`~autoware_carla_scenario.entity.ego.EgoVehicle`), because the runner
    calls them side by side and one reading order is easier to follow than two:

    ===================== ==================================================
    ``prepare(context)``  Before ``setup()``; nothing is spawned yet.
    ``adopt(entity)``     As the ego and each authored NPC join the run.
    ``start(world)``      After warm-up, before the scenario clock starts.
    ``tick(world, t)``    Once per ``world.tick()``, beside ``ego.on_tick``.
    ``close()``           During teardown, while the world is still alive.
    ===================== ==================================================

    Implementations must not raise out of :meth:`tick` or out of any manoeuvre:
    a traffic model that cannot do something is a warning, while an exception
    there would end a scenario that is otherwise perfectly valid.  :meth:`prepare`
    is the opposite -- it is the place to refuse loudly, before the run has cost
    anything.

    **One backend serves a whole queue.**  :class:`~autoware_carla_scenario.ScenarioQueue`
    builds one runner, so the real sequence over a batch is ``prepare ... close,
    prepare ... close``, with a different scenario -- and possibly a different
    map -- each time.  :meth:`close` is therefore the end of a *run*, not the end
    of the object: it must forget every vehicle the backend created and every
    entity it adopted, and leave the backend able to :meth:`prepare` again.

    **Level of detail is the backend's.**  A backend may simulate the vehicles
    far from what matters mesoscopically (:class:`SimulationFidelity`) and
    only those near it microscopically, as actors in CARLA.  The policy --
    which vehicles, how far -- is
    :class:`~autoware_carla_scenario.traffic.fidelity.MesoscopicConfig`, and a
    queue model any backend can run its far traffic on is
    :class:`~autoware_carla_scenario.traffic.mesoscopic.MesoscopicTraffic`;
    what the seam fixes is the entity's side: its flag says which level it is
    at, only the backend writes it (:meth:`_mark_fidelity`), and an entity
    leaves the world only if it agrees to (:meth:`DrivenEntity._replace_actor`).
    """

    #: The name this backend is selected by, and how it names itself in logs and
    #: in a run's result record.  Subclasses must set it.
    name: ClassVar[str] = "unnamed"
    #: Whether :meth:`close_lane` has said this backend cannot close a lane.
    _warned_lane_closure: bool = False
    #: What :meth:`set_vehicle_parameters` was given; ``None``: every default.
    _vehicle_parameters: Optional[VehicleParameters] = None
    #: What :meth:`set_pedestrian_parameters` was given; ``None``: every default.
    _pedestrian_parameters: Optional[PedestrianParameters] = None
    #: What :meth:`set_bicycle_parameters` was given; ``None``: every default.
    _bicycle_parameters: Optional[BicycleParameters] = None
    #: The parameters the run started with, while an override changed them;
    #: :meth:`_restore_parameters` puts them back when the run closes.
    _configured_vehicles: Optional[VehicleParameters] = None
    _configured_pedestrians: Optional[PedestrianParameters] = None
    _configured_bicycles: Optional[BicycleParameters] = None
    _overridden: bool = False

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def supported_vehicle_parameters(self) -> set[str]:
        """The :class:`~autoware_carla_scenario.traffic.parameters.VehicleParameters`
        fields this backend honours.

        What :data:`~autoware_carla_scenario.traffic.parameters.VEHICLE_PARAMETER_BACKENDS`
        says of this backend's :attr:`name`: none for a backend it does not
        name, which refuses every one set rather than run as if it had.  A
        backend from another package overrides this with the fields it applies.
        """
        return {
            field
            for field, backends in VEHICLE_PARAMETER_BACKENDS.items()
            if self.name in backends
        }

    def supported_pedestrian_parameters(self) -> set[str]:
        """The :class:`~autoware_carla_scenario.traffic.parameters.PedestrianParameters`
        fields this backend honours, as :meth:`supported_vehicle_parameters`
        (:data:`~autoware_carla_scenario.traffic.parameters.PEDESTRIAN_PARAMETER_BACKENDS`).
        """
        return {
            field
            for field, backends in PEDESTRIAN_PARAMETER_BACKENDS.items()
            if self.name in backends
        }

    def supported_bicycle_parameters(self) -> set[str]:
        """The :class:`~autoware_carla_scenario.traffic.parameters.BicycleParameters`
        fields this backend honours, as :meth:`supported_vehicle_parameters`
        (:data:`~autoware_carla_scenario.traffic.parameters.BICYCLE_PARAMETER_BACKENDS`).
        """
        return {
            field
            for field, backends in BICYCLE_PARAMETER_BACKENDS.items()
            if self.name in backends
        }

    def parameter_requirements(self) -> dict[str, str]:
        """The vehicle parameters this backend leaves out of
        :meth:`supported_vehicle_parameters` only because one of its options
        is not set, each with what to set: what an error names instead of the
        other backends.  None by default."""
        return {}

    def set_vehicle_parameters(self, parameters: VehicleParameters) -> None:
        """Drive the vehicles with *parameters* from the next :meth:`prepare` on.

        :func:`~autoware_carla_scenario.traffic.registry.build_backend` calls
        this with the ``traffic.vehicles`` node; a backend built in Python
        takes them in its constructor or here.

        Raises:
            TrafficParameterError: If a field is set that this backend does not
                support (:meth:`supported_vehicle_parameters`), naming the
                backends that do.  Raised before the run, as :meth:`prepare`
                refuses.
        """
        parameters.check_supported(
            self.name,
            self.supported_vehicle_parameters(),
            self.parameter_requirements(),
        )
        self._vehicle_parameters = parameters

    def set_pedestrian_parameters(self, parameters: PedestrianParameters) -> None:
        """Have the pedestrians behave as *parameters* say, from the next
        :meth:`prepare` on (the ``traffic.pedestrians`` node).

        Raises:
            TrafficParameterError: As :meth:`set_vehicle_parameters`, for
                :meth:`supported_pedestrian_parameters`.
        """
        parameters.check_supported(self.name, self.supported_pedestrian_parameters())
        self._pedestrian_parameters = parameters

    def set_bicycle_parameters(self, parameters: BicycleParameters) -> None:
        """Have the bicycles ride as *parameters* say, from the next
        :meth:`prepare` on (the ``traffic.bicycles`` node).

        Raises:
            TrafficParameterError: As :meth:`set_vehicle_parameters`, for
                :meth:`supported_bicycle_parameters`.
        """
        parameters.check_supported(self.name, self.supported_bicycle_parameters())
        self._bicycle_parameters = parameters

    @property
    def vehicle_parameters(self) -> VehicleParameters:
        """The parameters this backend drives vehicles with (every field
        ``None`` when none were set)."""
        if self._vehicle_parameters is None:
            return _NO_VEHICLE_PARAMETERS
        return self._vehicle_parameters

    @property
    def pedestrian_parameters(self) -> PedestrianParameters:
        """The parameters of this backend's pedestrians (every field ``None``
        when none were set)."""
        if self._pedestrian_parameters is None:
            return _NO_PEDESTRIAN_PARAMETERS
        return self._pedestrian_parameters

    @property
    def bicycle_parameters(self) -> BicycleParameters:
        """The parameters of this backend's bicycles (every field ``None``
        when none were set)."""
        if self._bicycle_parameters is None:
            return _NO_BICYCLE_PARAMETERS
        return self._bicycle_parameters

    # ------------------------------------------------------------------
    # Parameters changed during a run
    # ------------------------------------------------------------------

    def _remember_parameters(self) -> None:
        """Keep the run's own parameters before an override changes them: one
        backend serves a whole queue, and the next run starts from its config,
        not from where this one's actions left it."""
        if not self._overridden:
            self._overridden = True
            self._configured_vehicles = self._vehicle_parameters
            self._configured_pedestrians = self._pedestrian_parameters
            self._configured_bicycles = self._bicycle_parameters

    def _restore_parameters(self) -> None:
        """Put back what :meth:`_remember_parameters` kept; :meth:`close` calls it."""
        if self._overridden:
            self._overridden = False
            self._vehicle_parameters = self._configured_vehicles
            self._pedestrian_parameters = self._configured_pedestrians
            self._bicycle_parameters = self._configured_bicycles

    def supported_vehicle_overrides(self) -> set[str]:
        """The vehicle parameters this backend can change during a run
        (:meth:`override_vehicle_parameters`).

        None by default: a backend that cannot refuses the
        :class:`~autoware_carla_scenario.TrafficParametersAction` that asks,
        when it is registered, rather than leave the run as it was.
        """
        return set()

    def supported_pedestrian_overrides(self) -> set[str]:
        """The pedestrian parameters this backend can change during a run
        (:meth:`override_pedestrian_parameters`); none by default."""
        return set()

    def supported_bicycle_overrides(self) -> set[str]:
        """The bicycle parameters this backend can change during a run
        (:meth:`override_bicycle_parameters`); none by default."""
        return set()

    def override_vehicle_parameters(
        self, world: "carla.World", overrides: VehicleParameters
    ) -> None:
        """Change the vehicles' parameters from now on: the fields *overrides*
        sets replace the ones the run had, and every vehicle draws its value
        of them again.

        Only the fields of :meth:`supported_vehicle_overrides` reach here; the
        action checked them when it was registered.
        """
        logger.warning("%s: cannot change vehicle parameters during a run", self.name)

    def override_entity_parameters(
        self, world: "carla.World", overrides: VehicleParameters, entity: DrivenEntity
    ) -> None:
        """Give *entity*'s vehicle alone a new draw of the fields *overrides*
        sets; the run's parameters stay as they were."""
        logger.warning(
            "%s: cannot change a vehicle's parameters during a run", self.name
        )

    def override_pedestrian_parameters(
        self, world: "carla.World", overrides: PedestrianParameters
    ) -> None:
        """Change the pedestrians' parameters from now on, as
        :meth:`override_vehicle_parameters` does the vehicles'."""
        logger.warning(
            "%s: cannot change pedestrian parameters during a run", self.name
        )

    def override_bicycle_parameters(
        self, world: "carla.World", overrides: BicycleParameters
    ) -> None:
        """Change the bicycles' parameters from now on, as
        :meth:`override_vehicle_parameters` does the vehicles'."""
        logger.warning("%s: cannot change bicycle parameters during a run", self.name)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def prepare(self, context: TrafficContext) -> None:
        """Get ready for the run described by *context*.

        Called before the scenario's ``setup()``, so a backend that cannot run
        -- a missing simulator, an unusable map -- raises
        :class:`TrafficBackendUnavailable` here and costs the run nothing.

        Args:
            context: The run this backend is joining.
        """

    def adopt(self, entity: DrivenEntity) -> None:
        """Take note of an entity the scenario owns.

        Called for the ego and for every entity registered with
        :meth:`~autoware_carla_scenario.scenario_base.BaseScenario.register_entity`.
        The vehicle stays the scenario's -- the backend does not spawn, drive or
        destroy it -- but a backend that simulates traffic elsewhere has to know
        it exists, because its own vehicles must react to it.

        Args:
            entity: The entity joining the run.
        """

    def start(
        self, world: "carla.World", *, skip_actor_ids: Collection[int] = ()
    ) -> None:
        """Begin driving traffic.

        Called after the warm-up ticks and the init phase, immediately before
        the scenario clock starts -- the point at which a vehicle is allowed to
        move for the first time.

        Args:
            world: The CARLA world.
            skip_actor_ids: Actors this backend must not touch, because
                something else drives them (an Autoware ego, a driver policy).
                This is the one-authority-per-vehicle rule, passed as data.
        """

    def tick(self, world: "carla.World", elapsed: float) -> None:
        """Advance the backend by one simulation step.

        Called once per ``world.tick()``, right after it. A backend that rides
        the CARLA tick (the TrafficManager does) has nothing to do here; one
        that drives a second simulator steps it exactly once, so the two clocks
        cannot drift.

        Args:
            world: The CARLA world.
            elapsed: Simulated seconds since the run began.
        """

    def close(self) -> None:
        """Release everything this backend created.

        Called during teardown while the world is still alive, and called even
        when the run failed, so it must be safe to call twice and safe to call
        after a failed :meth:`prepare`.
        """

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        """Return what drove this run, for the result record.

        Subclasses add their seed, their process version, whatever makes a run
        reproducible from its record alone.  The parameters that were set
        are there under ``vehicles``, ``pedestrians`` and ``bicycles``.
        """
        out: dict[str, Any] = {"backend": self.name}
        vehicles = self.vehicle_parameters.describe()
        if vehicles:
            out["vehicles"] = vehicles
        pedestrians = self.pedestrian_parameters.describe()
        if pedestrians:
            out["pedestrians"] = pedestrians
        bicycles = self.bicycle_parameters.describe()
        if bicycles:
            out["bicycles"] = bicycles
        return out

    # ------------------------------------------------------------------
    # Manoeuvres -- the intents entities delegate here
    # ------------------------------------------------------------------

    def change_lane(
        self, entity: DrivenEntity, world: "carla.World", direction: LaneChangeDirection
    ) -> None:
        """Move *entity* one lane in *direction*.

        Args:
            entity: The entity asking for the manoeuvre.
            world: The CARLA world.
            direction: Which way to go.
        """
        self._unavailable("change_lane", entity)

    def lane_change_finished(self, entity: DrivenEntity, world: "carla.World") -> bool:
        """Whether *entity*'s lane change has settled.

        ``False`` from a backend that never starts one is the honest answer, and
        it is also OpenSCENARIO's behaviour: a manoeuvre that never completes
        never fires its reaction, and ending the run on a timer is the scenario
        timeout's job.
        """
        return False

    def set_desired_speed(
        self, entity: DrivenEntity, world: "carla.World", speed_kmh: float
    ) -> None:
        """Drive *entity* at *speed_kmh* from now on.

        The target only; a rate limit has already been applied by the caller,
        which walks the target from the vehicle's own speed and re-sends it,
        so a backend implements one call rather than a transition model of its
        own.

        Args:
            entity: The entity asking for the new speed.
            world: The CARLA world.
            speed_kmh: The speed to hold, in km/h.  Never negative.
        """
        self._unavailable("set_desired_speed", entity)

    def turn_at_junction(
        self,
        entity: DrivenEntity,
        world: "carla.World",
        direction: TurnDirection,
        **kwargs: Any,
    ) -> None:
        """Send *entity* *direction* at the next junction ahead."""
        self._unavailable("turn_at_junction", entity)

    def follow_route(
        self, entity: DrivenEntity, world: "carla.World", stops: List["Lanelet2Pose"]
    ) -> None:
        """Drive *entity* along the route through *stops* -- where it is, its
        waypoints, its goal, on the Lanelet2 map -- through every junction as
        planned; see ``docs/traffic_backends.md``."""
        self._unavailable("follow_route", entity)

    def finish_at(
        self,
        entity: DrivenEntity,
        world: "carla.World",
        points: List["carla.Location"],
    ) -> bool:
        """Drive *entity*'s vehicle along *points* -- the rest of its route, its
        last the exact place it is to stop -- itself, and stop it there.
        ``False`` (the default) when the backend cannot: the caller asks for
        no speed instead, and it stops where its driver puts it."""
        return False

    def release(self, entity: DrivenEntity, world: "carla.World") -> None:
        """Stop driving *entity*: something else has taken it over.

        Called when a scenario action drives the vehicle itself -- a
        :class:`~autoware_carla_scenario.actions.RoutingAction` puts it
        on a trajectory every tick, and a backend still steering it would undo
        that on every one.  The one-authority-per-vehicle rule that
        ``skip_actor_ids`` states at :meth:`start`, applied during the run.

        Must hold whether it arrives before or after :meth:`start`, and must be
        safe to repeat.  The base does nothing, which is right for a backend
        that never drives the scenario's vehicles.

        Args:
            entity: The entity being taken over.
            world: The CARLA world.
        """

    def resume(self, entity: DrivenEntity, world: "carla.World") -> None:
        """Drive *entity* again: what took it over (:meth:`release`) is done.

        Called when a scenario action that drove the vehicle itself hands it
        back -- an adversarial manoeuvre over, the vehicle rejoins traffic
        from wherever the action left it.  Undoes :meth:`release`, and like
        it must hold before or after :meth:`start` and be safe to repeat; an
        entity never released is left as it is.  The base does nothing, which
        is right for a backend that never drives the scenario's vehicles.

        Args:
            entity: The entity handed back.
            world: The CARLA world.
        """

    # ------------------------------------------------------------------
    # The scenario's pedestrians
    # ------------------------------------------------------------------

    def adopt_pedestrian(self, entity: WalkedEntity) -> None:
        """Take note of a pedestrian the scenario owns.

        Called for every pedestrian registered with
        :meth:`~autoware_carla_scenario.scenario_base.BaseScenario.register_pedestrian`,
        before :meth:`start`.  As with :meth:`adopt`, the walker stays the
        scenario's: the backend does not spawn or destroy it.  A backend that
        walks pedestrians (:meth:`walk_straight`) takes it on at :meth:`start`.

        Args:
            entity: The pedestrian joining the run.
        """

    def walk_straight(
        self, entity: WalkedEntity, speed_ms: float, yaw_deg: Optional[float] = None
    ) -> bool:
        """Walk *entity* on at *speed_ms* the way it faces -- or along
        *yaw_deg*, a CARLA yaw in degrees, when given -- if this backend walks
        it; *speed_ms* zero stops it where it is.

        Answers whether it did.  ``False`` leaves the walk to the entity, which
        walks its walker straight ahead itself (``carla.WalkerControl``): what
        the base answers, since a backend that walks no pedestrians -- the
        TrafficManager, which has none -- has nothing to send the walk to.

        Args:
            entity: The pedestrian a
                :class:`~autoware_carla_scenario.actions.WalkStraightAction`
                names.
            speed_ms: Walking speed in metres per second.
            yaw_deg: The way to walk; ``None`` is the way the walker faces.
        """
        del yaw_deg
        return False

    def release_pedestrian(self, entity: WalkedEntity, world: "carla.World") -> None:
        """Stop walking *entity*: a scenario action moves it itself.

        :meth:`release` for a pedestrian: a
        :class:`~autoware_carla_scenario.actions.RoutingAction` (an
        adversity among them) puts the walker on its path every tick.  Must
        hold before or after :meth:`start` and be safe to repeat.  The base does
        nothing, which is right for a backend that walks no pedestrians.

        Args:
            entity: The pedestrian being taken over.
            world: The CARLA world.
        """

    # ------------------------------------------------------------------
    # Level of detail -- microscopic near what matters, mesoscopic elsewhere
    # ------------------------------------------------------------------

    def fidelity_of(self, entity: DrivenEntity) -> SimulationFidelity:
        """How this backend simulates *entity* now.

        The entity's own flag, which this backend keeps current
        (:meth:`_mark_fidelity`).  A backend with one level never changes it,
        so every vehicle it drives is microscopic.
        """
        return entity.simulation_fidelity

    def require_microscopic(self, entity: DrivenEntity, world: "carla.World") -> bool:
        """Make *entity* microscopic now, for something that needs its actor.

        For an action that is about to drive the vehicle itself or read its
        actor: a mesoscopic vehicle has none.  Answers whether *entity* is
        microscopic afterwards.  The base has one level, so the answer is the
        entity's flag; a backend with two brings the vehicle back into the
        world here, where its mesoscopic model has it, and keeps it there for
        the rest of the run -- whatever needed the actor may hold on to it.

        Args:
            entity: The entity that must be in the world.
            world: The CARLA world.
        """
        return entity.simulation_fidelity is SimulationFidelity.MICROSCOPIC

    def fidelity_census(self) -> dict[str, int]:
        """How many vehicles this backend simulates at each level now.

        Keyed by :class:`SimulationFidelity` value, and counting every vehicle
        the backend simulates -- its own traffic as well as the scenario's.
        Empty from a backend that does not keep the count.
        """
        return {}

    def _mark_fidelity(
        self, entity: DrivenEntity, fidelity: SimulationFidelity
    ) -> None:
        """Record that *entity* is simulated at *fidelity* from now on.

        The one writer of :attr:`DrivenEntity.simulation_fidelity`.  Called by
        a backend once a change of level is complete -- the actor destroyed
        and the entity's reference dropped (:meth:`DrivenEntity._replace_actor`),
        or a new actor spawned and handed over -- never before, so the flag
        and ``entity.actor`` always agree.
        """
        entity.simulation_fidelity = fidelity
        # What a lookup of the entity's actor by role name asks
        # (warn_if_mesoscopic): kept here, the one writer of the flag.
        if hasattr(entity, "role_name"):
            role_name = entity.role_name
            if role_name is not None:
                name = str(role_name)
                if fidelity is SimulationFidelity.MESOSCOPIC:
                    _MESOSCOPIC_ROLES.add(name)
                else:
                    _MESOSCOPIC_ROLES.discard(name)
                    _WARNED_ROLES.discard(name)

    # ------------------------------------------------------------------
    # Background traffic -- vehicles no scenario entity stands for
    # ------------------------------------------------------------------

    def spawn_background(
        self,
        world: "carla.World",
        transform: "carla.Transform",
        *,
        speed_kmh: float,
        blueprint: Optional[str] = None,
    ) -> Optional[str]:
        """Put one background vehicle on the road and let this backend drive it.

        Background traffic is what :class:`~autoware_carla_scenario.TrafficSourceAction`
        asks for: vehicles no scenario entity stands for, driven by the traffic
        model like any other traffic.  May be called before :meth:`start` --
        during the scenario's initialization -- as well as during the run.

        Args:
            world: The CARLA world.
            transform: Where, as a ``carla.Transform`` on a driving lane.
            speed_kmh: The speed it drives at, at most, in km/h: the cap is
                exactly this on every backend.  It also enters at no more than
                this.  Finite and at least
                :data:`~autoware_carla_scenario.constants.MIN_BACKGROUND_SPEED_KMH`
                (:func:`check_background_speed`).
            blueprint: A CARLA vehicle blueprint, or ``None`` (or ``""``) for
                the backend's own.

        Returns:
            A handle naming the vehicle for :meth:`remove_background`, or
            ``None`` when it could not be placed (no driving lane within
            :data:`~autoware_carla_scenario.constants.BACKGROUND_MAX_LANE_OFFSET_M`
            of *transform*, the spot is taken, the backend has no traffic
            model).

        Raises:
            ValueError: If *speed_kmh* is not a speed a vehicle can be capped
                at (:func:`check_background_speed`), on every backend alike.
        """
        check_background_speed(speed_kmh)
        logger.warning(
            "%s: %r cannot create background traffic", type(self).__name__, self.name
        )
        return None

    def background_vehicles(
        self, world: "carla.World"
    ) -> dict[str, tuple[float, float]]:
        """Every live background vehicle, by handle, at its CARLA ``(x, y)``."""
        return {}

    def remove_background(self, world: "carla.World", handle: str) -> None:
        """Take the background vehicle *handle* off the road."""

    def background_actors(self, world: "carla.World") -> dict[str, "carla.Actor"]:
        """The CARLA actor of every background vehicle this backend drives in
        the world now, by its ``role_name``: what an adversity may take over
        for a while, by its actor (:meth:`release_actor`, #417).  None by
        default."""
        return {}

    def release_actor(self, world: "carla.World", actor: "carla.Actor") -> None:
        """Stop driving the background vehicle *actor*; the caller drives it
        now (:meth:`release` for a vehicle no entity stands for)."""

    def resume_actor(self, world: "carla.World", actor: "carla.Actor") -> None:
        """Drive the background vehicle *actor* again, from where it is now."""

    # ------------------------------------------------------------------
    # Lane closures -- a construction zone (#402)
    # ------------------------------------------------------------------

    def close_lane(
        self, world: "carla.World", transform: "carla.Transform"
    ) -> Optional[str]:
        """Keep this backend's traffic off the lane at *transform* until
        :meth:`reopen_lane`; a handle naming the closure, or ``None`` when
        this backend cannot close a lane (the default, with a warning once)."""
        if not self._warned_lane_closure:
            self._warned_lane_closure = True
            logger.warning(
                "%s: %r cannot close a lane; its traffic may drive into the "
                "zone (a TrafficSinkAction there takes it off the road)",
                type(self).__name__,
                self.name,
            )
        return None

    def reopen_lane(self, world: Optional["carla.World"], handle: str) -> None:
        """Open the lane :meth:`close_lane` closed as *handle* again."""

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _unavailable(self, what: str, entity: DrivenEntity) -> None:
        """Report an intent this backend cannot carry out, without raising."""
        logger.warning(
            "%s: %r cannot %s for %s; the vehicle keeps doing what it was doing",
            type(self).__name__,
            self.name,
            what,
            _entity_name(entity),
        )


def _entity_name(entity: DrivenEntity) -> str:
    """Return the most identifying name an entity has, for a log line."""
    # hasattr rather than getattr with a default: Codon decides it when it
    # compiles (docs/typecheck.md), and DrivenEntity itself has no role_name.
    if hasattr(entity, "role_name"):
        role_name = entity.role_name
        if role_name is not None:
            return str(role_name)
    return type(entity).__name__


class NullTrafficBackend(TrafficBackend):
    """A run with no traffic model at all.

    Nothing is driven and no ambient vehicle is created: the only vehicles that
    move are the ones something else drives -- an Autoware ego, an ego under a
    driver policy, a vehicle a scenario steers itself.  It is the honest option
    for a test about one vehicle on an empty road, and the smallest possible
    proof that the seam is really a seam: selecting it changes the run without
    changing a line of the runner.

    An ego that expects to be *driven for* (``ego.entity=autopilot``, the
    default) will not move under this backend, so :meth:`start` says as much
    rather than leaving the run to die on its timeout with nothing in the log.
    """

    name: ClassVar[str] = "none"

    def start(
        self, world: "carla.World", *, skip_actor_ids: Collection[int] = ()
    ) -> None:
        """Drive nothing, and name what is being left undriven.

        A vehicle nobody drives is the point of this backend for an NPC and a
        trap for an ego that opted into being driven, and the two are
        indistinguishable from here -- so the honest thing is to report the
        count and let the log say why nothing moved.
        """
        skip = set(skip_actor_ids)
        undriven = [
            actor
            for actor in world.get_actors().filter("vehicle.*")
            if actor.id not in skip
        ]
        if undriven:
            logger.warning(
                "traffic backend %r drives nothing: %d vehicle(s) are left "
                "standing. An ego with ego.entity=autopilot is driven by the "
                "traffic backend and will not move under this one.",
                self.name,
                len(undriven),
            )
