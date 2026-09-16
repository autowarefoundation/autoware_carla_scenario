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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Collection, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

__all__ = [
    "LaneChangeDirection",
    "TurnDirection",
    "LaneChanging",
    "TurningAtJunctions",
    "TrafficBackend",
    "TrafficBackendError",
    "TrafficBackendUnavailable",
    "TrafficContext",
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
# The vocabulary
# ---------------------------------------------------------------------------


class LaneChangeDirection(enum.Enum):
    """Direction of a lane change.

    Defined with the traffic seam rather than in the action that asks for one:
    the action names an intent, and a backend is what knows what the intent
    means to whatever is driving.  ``entity.tm_driving`` and ``actions.lane_change``
    re-export it, which is where scenario authors and the editor reach it.
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


@runtime_checkable
class LaneChanging(Protocol):
    """What :class:`~autoware_carla_scenario.actions.lane_change.LaneChangeAction`
    needs of an entity.

    Stated as a protocol rather than a base class because the action does not
    care what performs the manoeuvre -- only that something can be asked for one
    and asked whether it is done.
    """

    def change_lane(self, world: Any, direction: "LaneChangeDirection") -> None:
        """Move one lane in *direction*."""
        ...

    def lane_change_finished(self, world: Any) -> bool:
        """Whether the manoeuvre has settled."""
        ...


@runtime_checkable
class TurningAtJunctions(Protocol):
    """What :class:`~autoware_carla_scenario.actions.turn.TurnAction` needs."""

    def turn_at_junction(
        self, world: Any, direction: "TurnDirection", **kwargs: object
    ) -> None:
        """Go *direction* at the next junction ahead."""
        ...


# ---------------------------------------------------------------------------
# The run a backend joins
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrafficContext:
    """Everything a backend needs to know about the run it is joining.

    Built once per scenario by :class:`~autoware_carla_scenario.ScenarioRunner`
    and handed to :meth:`TrafficBackend.prepare` before anything is spawned, so
    that a backend which has to start a process, derive a road network or refuse
    the run entirely does so while failing is still cheap.

    Attributes:
        client: The CARLA client (``carla.Client``).  Typed loosely so this
            module stays importable without CARLA.
        world: The CARLA world the scenario runs in.
        map_name: Name of the loaded map, as CARLA reports it.
        xodr_path: The OpenDRIVE the world is running, when the run installed
            one.  ``None`` when the map's roads came from CARLA's own assets and
            nothing has read them back yet.
        fixed_delta_seconds: The simulation step the runner applies.  A backend
            that steps a second simulator matches its step length to this.
        random_seed: The scenario's seed.  A backend with any randomness of its
            own derives it from this, so a repeated run repeats.
        output_dir: Where this run's artefacts go; a backend writes its own logs
            under it rather than into the working directory.
    """

    client: Any = None
    world: Any = None
    map_name: str = ""
    xodr_path: Optional[Path] = None
    fixed_delta_seconds: float = 0.05
    random_seed: int = 0
    output_dir: Path = field(default_factory=lambda: Path("scenario_outputs"))


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
    """

    #: The name this backend is selected by, and how it names itself in logs and
    #: in a run's result record.  Subclasses must set it.
    name: ClassVar[str] = "unnamed"

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

    def adopt(self, entity: Any) -> None:
        """Take note of an entity the scenario owns.

        Called for the ego and for every entity registered with
        :meth:`~autoware_carla_scenario.scenario_base.BaseScenario.register_entity`.
        The vehicle stays the scenario's -- the backend does not spawn, drive or
        destroy it -- but a backend that simulates traffic elsewhere has to know
        it exists, because its own vehicles must react to it.

        Args:
            entity: The entity joining the run.
        """

    def start(self, world: Any, *, skip_actor_ids: Collection[int] = ()) -> None:
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

    def tick(self, world: Any, elapsed: float) -> None:
        """Advance the backend by one simulation step.

        Called once per ``world.tick()``, right after it. A backend that rides
        the CARLA tick (the TrafficManager does) has nothing to do here; one
        that drives a second simulator steps it exactly once, so the two clocks
        cannot drift.

        Args:
            world: The CARLA world.
            elapsed: Seconds since the scenario clock started.
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
        reproducible from its record alone.
        """
        return {"backend": self.name}

    # ------------------------------------------------------------------
    # Manoeuvres -- the intents entities delegate here
    # ------------------------------------------------------------------

    def change_lane(
        self, entity: Any, world: Any, direction: LaneChangeDirection
    ) -> None:
        """Move *entity* one lane in *direction*.

        Args:
            entity: The entity asking for the manoeuvre.
            world: The CARLA world.
            direction: Which way to go.
        """
        self._unavailable("change_lane", entity)

    def lane_change_finished(self, entity: Any, world: Any) -> bool:
        """Whether *entity*'s lane change has settled.

        ``False`` from a backend that never starts one is the honest answer, and
        it is also OpenSCENARIO's behaviour: a manoeuvre that never completes
        never fires its reaction, and ending the run on a timer is the scenario
        timeout's job.
        """
        del entity, world
        return False

    def turn_at_junction(
        self, entity: Any, world: Any, direction: TurnDirection, **kwargs: Any
    ) -> None:
        """Send *entity* *direction* at the next junction ahead."""
        del kwargs
        self._unavailable("turn_at_junction", entity)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _unavailable(self, what: str, entity: Any) -> None:
        """Report an intent this backend cannot carry out, without raising."""
        logger.warning(
            "%s: %r cannot %s for %s; the vehicle keeps doing what it was doing",
            type(self).__name__,
            self.name,
            what,
            _entity_name(entity),
        )


def _entity_name(entity: Any) -> str:
    """Return the most identifying name an entity has, for a log line."""
    role_name = getattr(entity, "role_name", None)
    return str(role_name) if role_name is not None else type(entity).__name__


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

    def start(self, world: Any, *, skip_actor_ids: Collection[int] = ()) -> None:
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
