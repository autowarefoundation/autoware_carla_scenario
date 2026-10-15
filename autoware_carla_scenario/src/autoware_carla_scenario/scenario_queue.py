"""Scenario queue bound to a CarlaServerManager."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Generator, List, Optional, Union

from .scenario_runner import ScenarioRunner
from .conditions import ScenarioResult
from .constants import DEFAULT_TM_PORT
from .coordinate.map_manager import MapManager
from .maps import capture_opendrive
from .odd import OddCase, OddDefinition, resolve_odd
from .scenario_base import BaseScenario
from .server import CarlaServerManager
from .traffic.base import TrafficBackend

if TYPE_CHECKING:
    from .autoware_stack.launcher import AutowareLauncher
else:
    # tqdm has no Codon model: the check (docs/typecheck.md), which reads
    # TYPE_CHECKING as true, waits without the progress bar.
    from tqdm import tqdm

logger = logging.getLogger(__name__)


class ScenarioQueue:
    """Owns a :class:`CarlaServerManager` and runs scenarios sequentially.

    The queue acts as a context manager that starts the CARLA server on
    entry and stops it on exit.  All scenarios must be :meth:`add`-ed
    *before* calling :meth:`run_all`.

    When used together with :class:`~autoware_carla_scenario.CarlaScenarioFixture`,
    scenarios are registered at module import time so the full list is known
    before the session fixture starts the server.


    Example – minimal usage::

        queue = ScenarioQueue(map_name="Town10HD_Opt")
        queue.add(MyScenario(ego_config))
        queue.add(AnotherScenario(ego_config))

        with queue:
            results = queue.run_all()

    Example – with an externally-managed server::

        with CarlaServerManager() as server:
            queue = ScenarioQueue(server=server, map_name="Town10HD_Opt")
            queue.add(MyScenario(ego_config))
            results = queue.run_all()
    """

    _server: CarlaServerManager
    _owns_server: bool
    _xodr_path: Optional[Path]
    _overwrite_xodr: bool
    _opendrive_path: Optional[Path]
    _lanelet2_path: Optional[Path]
    _map_name: Optional[str]
    _host: str
    _port: int
    _tm_port: int
    _timeout_seconds: float
    _output_dir: Path
    _cooldown_seconds: float
    _cooldown_max_retries: int
    _max_tick_rate_hz: Optional[float]
    _projector_type: Optional[str]
    _traffic_backend: Optional[TrafficBackend]
    _odd: OddDefinition
    _autoware_launcher: Optional["AutowareLauncher"]
    _scenarios: List[BaseScenario]
    #: id(scenario) -> its own timeout, in place of ``timeout_seconds``.
    _timeouts: dict[int, float]
    #: id(scenario) -> the ODD case it runs, to record.
    _cases: dict[int, OddCase]
    _results: List[ScenarioResult]
    _scenario_results: dict[int, ScenarioResult]
    #: Indices into the last run_all()'s results of the cases that raised on
    #: every attempt (``continue_on_error``).
    _errored: List[int]
    _runner: Optional[ScenarioRunner]
    #: Whether run_all() left its last scenario's world as it was, unreset.
    _world_left_as_is: bool

    def __init__(
        self,
        server: Optional[CarlaServerManager] = None,
        *,
        xodr_path: Optional[Path] = None,
        overwrite_xodr: bool = False,
        opendrive_path: Optional[Path] = None,
        lanelet2_path: Optional[Path] = None,
        map_name: Optional[str] = None,
        host: str = "localhost",
        port: int = 2000,
        tm_port: int = DEFAULT_TM_PORT,
        timeout_seconds: float = 60.0,
        output_dir: Path = Path("scenario_outputs"),
        server_extra_args: Optional[List[str]] = None,
        cooldown_seconds: float = 0.0,
        cooldown_max_retries: int = 0,
        max_tick_rate_hz: Optional[float] = None,
        projector_type: Optional[str] = None,
        traffic_backend: Optional[TrafficBackend] = None,
        odd: Union[str, OddDefinition, None] = None,
        autoware_launcher: Optional["AutowareLauncher"] = None,
    ) -> None:
        """Create a scenario queue.

        Args:
            server: An externally-managed :class:`CarlaServerManager`.  When
                provided the queue borrows it (does not start/stop it).  When
                *None* a new manager is created and owned by this queue.
            xodr_path: OpenDRIVE map file path (optional).  Read for the
                coordinate transforms; it is written into the CARLA
                installation only when *overwrite_xodr* is set.
            overwrite_xodr: Whether to replace the simulator's own road network
                with *xodr_path* before the world is loaded.  Off by default:
                a map cooked for CARLA already carries the OpenDRIVE its assets
                were built from, and overwriting that silently puts the geometry
                and the meshes out of step.  Turn it on only for a map whose
                roads CARLA does not ship, and then *map_name* is required too:
                the install target is derived from it via the
                ``<MAP_NAME_PATH>`` env var.
            opendrive_path: Where to read OpenDRIVE from *without* installing
                it into CARLA -- for a map whose roads the simulator already
                ships, such as an Autoware map published for a CARLA town.  The
                file is written from the loaded world when it is not there yet,
                so the first run of such a scenario fills it and later ones read
                it.  Ignored when *xodr_path* is given.
            lanelet2_path: Lanelet2 map file path (optional).  When provided
                together with an OpenDRIVE file, :class:`MapManager` is
                initialised so that coordinate transforms (Lanelet2 ↔
                OpenDRIVE) work.
            map_name: Built-in CARLA map name to load on start (optional).
                Used alone or together with *xodr_path*.
            host: CARLA RPC host.
            port: CARLA RPC port.
            tm_port: CARLA TrafficManager port.
            timeout_seconds: Default per-scenario timeout.
            output_dir: Directory for MP4 recordings.
            server_extra_args: Extra CLI arguments for CarlaUnreal.sh (only used
                when the queue creates its own server).
            cooldown_seconds: Wait time (seconds) between consecutive scenario
                runs.  Gives the CARLA server time to finish cleanup before the
                next scenario connects.  0 disables the cooldown.
            projector_type: Overrides the projection the Lanelet2 map is read
                with; the map's ``map_projector_info.yaml`` decides when unset.
            max_tick_rate_hz: Upper bound on how fast the runner steps the
                world, or *None* to step as fast as the server allows.  Cap it
                to the rate of the slowest client reading the simulation.
            cooldown_max_retries: Maximum number of retries when a scenario run
                fails (e.g. due to CARLA communication errors).  After each
                failed attempt a cooldown wait is inserted before the next
                retry.  0 means no retries.
            traffic_backend: What drives the vehicles the scenarios did not
                author.  *None* selects CARLA's TrafficManager on *tm_port*.
                One backend serves every scenario in the queue, the same way
                one CARLA server does.
            odd: The ODD every run is measured against: an
                :class:`OddDefinition`, or what :func:`resolve_odd` reads (a
                registered name, an OpenODD YAML path, ``module:function``).
                *None* selects the built-in ``default`` ODD.
            autoware_launcher: The launcher the queue's Autoware egos start a
                fresh Autoware with, scenario after scenario.  The queue
                prepares it when it starts -- building the workspace if it has
                to, and refusing one that cannot run a scenario -- before the
                CARLA server is started, and its stacks' logs go to
                *output_dir*.  The egos are handed the same launcher; *None*
                when Autoware is not the framework's to start.
        """
        if server is not None:
            self._server = server
            self._owns_server = False
        else:
            self._server = CarlaServerManager(
                host=host,
                port=port,
                extra_args=server_extra_args,
            )
            self._owns_server = True

        self._xodr_path = xodr_path
        self._overwrite_xodr = overwrite_xodr
        self._opendrive_path = opendrive_path
        self._lanelet2_path = lanelet2_path
        self._map_name = map_name
        self._host = host
        self._port = port
        self._tm_port = tm_port
        self._timeout_seconds = timeout_seconds
        self._output_dir = output_dir
        self._cooldown_seconds = cooldown_seconds
        self._cooldown_max_retries = cooldown_max_retries
        self._max_tick_rate_hz = max_tick_rate_hz
        self._projector_type = projector_type
        self._traffic_backend = traffic_backend
        # Resolved now, so a misspelt ODD fails before CARLA starts.
        self._odd = resolve_odd(odd)
        self._autoware_launcher = autoware_launcher

        self._scenarios = []
        self._timeouts = {}
        self._cases = {}
        self._results = []
        self._scenario_results = {}
        self._errored = []
        self._runner = None
        self._world_left_as_is = False

    # ------------------------------------------------------------------
    # Scenario registration
    # ------------------------------------------------------------------

    def add(
        self,
        scenario: BaseScenario,
        *,
        timeout_seconds: Optional[float] = None,
        case: Optional[OddCase] = None,
    ) -> None:
        """Append a scenario to the queue.

        Args:
            scenario: A :class:`BaseScenario` instance to enqueue.
            timeout_seconds: This scenario's timeout; the queue's
                ``timeout_seconds`` when ``None``.
            case: The case drawn from the ODD it runs, and the space it was
                drawn in, to record in its coverage file
                (:meth:`ScenarioRunner.run_scenario`).
        """
        self._scenarios.append(scenario)
        if timeout_seconds is not None:
            self._timeouts[id(scenario)] = timeout_seconds
        if case is not None:
            self._cases[id(scenario)] = case

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run_all(
        self, *, release: bool = False, continue_on_error: bool = False
    ) -> List[ScenarioResult]:
        """Execute every registered scenario in order.

        Must be called while the queue is started (inside the context manager
        or after :meth:`start`).

        Args:
            release: Let go of each scenario once it has run, keeping only its
                result.  A finished scenario still holds everything its run
                built -- its entities, conditions, the ego's driver session and
                the trajectory it was following -- so a long batch that keeps
                them all grows with every case.  The queue is empty afterwards:
                :meth:`result_for` does not know the released scenarios, and a
                later :meth:`run_all` runs only what was added since.  Leave it
                off when the caller asks :meth:`result_for` or runs the same
                scenarios again.
            continue_on_error: Go on with the next scenario when one raises on
                every attempt (a spawn point without ground, CARLA going away),
                rather than end the run.  Its result is a failed
                :class:`ScenarioResult` whose message names the error, and
                :meth:`errored_indices` tells it from a scenario that ran and
                failed.  The world is reloaded before the next scenario, as
                after one that ran.  Off by default: a run that raises ends
                there.  A :class:`KeyboardInterrupt` ends the run either way.

        Returns:
            A list of :class:`ScenarioResult` objects in registration order.

        Raises:
            RuntimeError: If called before :meth:`start` / outside the context
                manager, or, without *continue_on_error*, when a scenario
                raises on every attempt.
        """
        if self._runner is None:
            raise RuntimeError(
                "ScenarioQueue must be started before run_all(). "
                "Use the context manager ('with queue:') or call start() first."
            )
        if self._world_left_as_is:
            # The last run_all() left its last scenario's world as it was.
            self._runner.reset_world()
            self._world_left_as_is = False
        results: List[ScenarioResult] = []
        self._errored = []
        max_attempts = 1 + self._cooldown_max_retries
        total = len(self._scenarios)
        # Taken from the end of a reversed copy, so that once the loop has
        # moved on nothing here refers to a scenario that has run.
        pending = list(reversed(self._scenarios))
        if release:
            self._scenarios = []
        for i in range(total):
            scenario = pending.pop()
            key = id(scenario)
            timeout_seconds = self._timeouts.get(key)
            case = self._cases.get(key)
            if release:
                # Keyed by id(): a released scenario's id may be the next
                # scenario's, so nothing is left keyed by it.
                if key in self._timeouts:
                    del self._timeouts[key]
                if key in self._cases:
                    del self._cases[key]
            last_error: Optional[Exception] = None
            last_message = ""
            for attempt in range(max_attempts):
                try:
                    # Cooldown: between consecutive scenarios on the first
                    # attempt, or as a recovery wait on retry attempts.
                    # Placed inside try/except so that crashes during the
                    # cooldown phase (e.g. CARLA communication errors) are
                    # caught and trigger a retry.
                    if (i > 0 or attempt > 0) and self._cooldown_seconds > 0:
                        _wait_with_progress(
                            self._cooldown_seconds,
                            desc="CARLA cooldown"
                            if attempt == 0
                            else "CARLA cooldown (retry)",
                        )
                    result = self._runner.run_scenario(
                        scenario,
                        timeout_seconds=timeout_seconds,
                        case=case,
                        # The map is reloaded for the next scenario; after the
                        # last there is none, and a later run_all() or a new
                        # queue loads it first.
                        reset_world=i < total - 1,
                    )
                    if not release:
                        self._scenario_results[key] = result
                    results.append(result)
                    break
                except Exception as exc:
                    last_error = exc
                    last_message = f"{type(exc).__name__}: {exc}"
                    # With its traceback: a batch that goes on keeps only a
                    # line of it in the result.
                    logger.warning(
                        "[%d/%d] Scenario attempt %d/%d failed: %s",
                        i + 1,
                        total,
                        attempt + 1,
                        max_attempts,
                        exc,
                        exc_info=True,
                    )
            else:
                if not continue_on_error:
                    if release:
                        # The cases not run yet are out of the queue already:
                        # leave nothing keyed by their ids either.
                        for rest in pending:
                            rest_key = id(rest)
                            if rest_key in self._timeouts:
                                del self._timeouts[rest_key]
                            if rest_key in self._cases:
                                del self._cases[rest_key]
                    # All attempts exhausted — re-raise the last error.
                    raise RuntimeError(
                        f"Scenario {type(scenario).__name__} failed after "
                        f"{max_attempts} attempt(s)"
                    ) from last_error
                # All attempts exhausted: the case is recorded as errored and
                # the run goes on.
                result = ScenarioResult(
                    passed=False,
                    message=f"Errored after {max_attempts} attempt(s): {last_message}",
                    elapsed_seconds=0.0,
                )
                logger.error(
                    "[%d/%d] Scenario %s errored after %d attempt(s), "
                    "going on with the next: %s",
                    i + 1,
                    total,
                    type(scenario).__name__,
                    max_attempts,
                    last_message,
                )
                if not release:
                    self._scenario_results[key] = result
                results.append(result)
                self._errored.append(i)
                if i < total - 1:
                    # A run that raises never reaches its own reload, so what
                    # it spawned is still in the world.
                    self._runner.reset_world()

        self._world_left_as_is = total > 0
        self._results = results
        return results

    def errored_indices(self) -> List[int]:
        """Return where, in the last :meth:`run_all`'s results, the scenarios
        are that raised on every attempt (``continue_on_error``), rather than
        ran and failed."""
        return list(self._errored)

    def result_for(self, scenario: BaseScenario) -> ScenarioResult:
        """Return the :class:`ScenarioResult` for a specific scenario instance.

        Args:
            scenario: The exact scenario object that was passed to :meth:`add`.

        Returns:
            The scenario's execution result.

        Raises:
            KeyError: If no result exists for *scenario* (run_all not called yet).
        """
        key = id(scenario)
        if key not in self._scenario_results:
            raise KeyError(
                f"No result found for {type(scenario).__name__}. "
                "Has run_all() been called?"
            )
        return self._scenario_results[key]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Ensure the CARLA server is running, then initialise the runner.

        :meth:`CarlaServerManager.start` is called unconditionally so that
        the server is always reachable by the time the map is loaded.
        When the server is already running (``reuse_if_running=True``, the
        default) the call is effectively a no-op — it just records that the
        server was reused and returns immediately.  Only the queue that
        *owns* its server will stop it in :meth:`stop`.
        """
        if self._autoware_launcher is not None:
            # First: a workspace that has to be built takes far longer than
            # CARLA to start, and one that cannot run a scenario at all is
            # better refused before anything else starts.
            self._autoware_launcher.prepare(log_dir=self._output_dir)
        self._server.start()
        self._world_left_as_is = False
        self._runner = ScenarioRunner(
            self._server,
            host=self._host,
            port=self._port,
            tm_port=self._tm_port,
            timeout_seconds=self._timeout_seconds,
            output_dir=self._output_dir,
            max_tick_rate_hz=self._max_tick_rate_hz,
            traffic_backend=self._traffic_backend,
            odd=self._odd,
        )
        if self._overwrite_xodr:
            if self._xodr_path is None or self._map_name is None:
                raise ValueError(
                    "overwrite_xodr requires both xodr_path and map_name: the "
                    "source is xodr_path and the install target is derived from "
                    "map_name. Leave overwrite_xodr off to load the map by name "
                    "and keep the OpenDRIVE CARLA ships."
                )
            self._runner.load_map_by_overwriting_xodr(self._xodr_path, self._map_name)
        elif self._map_name is not None:
            self._runner.load_map_by_name(self._map_name)

        # Initialise MapManager when both map files are available.
        # Pass the CARLA world so that spawn-point-based z_offset averaging
        # is used instead of single-point sampling.
        # A Path is always true, so this is `self._xodr_path or ...`.
        xodr_path = self._xodr_path
        if xodr_path is None:
            xodr_path = self._capture_opendrive()
        # The runner passes it on to the traffic backend, which is how a backend
        # with a road-network format of its own derives one from the map in play.
        self._runner.xodr_path = xodr_path
        # Always: the runs read the map the MapManager holds (run_carla_map),
        # so one an earlier queue of this process left there would stand in
        # for this queue's world.
        MapManager.reset()
        if xodr_path is not None and self._lanelet2_path is not None:
            MapManager.get_instance().initialize(
                xodr_path=xodr_path,
                lanelet2_path=self._lanelet2_path,
                carla_world=self._runner._world,
                projector_type=self._projector_type,
            )

    def _capture_opendrive(self) -> Optional[Path]:
        """Return the OpenDRIVE for the loaded world, writing it out once.

        A map published for Autoware carries no OpenDRIVE, because the roads
        belong to the CARLA town it was recorded on.  The town is loaded by the
        time this runs, so the road network is right there in the world -- and
        writing it to *opendrive_path* means the sweeper and the editor, which
        run before any world exists, find it on disk from then on.
        """
        destination = self._opendrive_path
        if destination is None or self._runner is None:
            return None
        if destination.is_file():
            return destination
        # Annotated, so Codon unwraps the Optional (docs/typecheck.md).
        runner: ScenarioRunner = self._runner
        world = runner._world
        if world is None:
            return None
        written = capture_opendrive(world, destination)
        logger.info("Wrote the loaded world's OpenDRIVE to %s", written)
        return written

    def stop(self) -> None:
        """Stop any Autoware stack left running, and the server if owned."""
        self._runner = None
        # The map loaded for this queue's runs is not the next queue's: one
        # that loads no lanelet map would read this one's (run_carla_map).
        MapManager.reset()
        if self._autoware_launcher is not None:
            self._autoware_launcher.close()
        if self._owns_server:
            self._server.stop()

    # ------------------------------------------------------------------
    # pytest fixture factory
    # ------------------------------------------------------------------

    def as_fixture(
        self, fixture_name: str = "carla_queue"
    ) -> Callable[[], Generator["ScenarioQueue", None, None]]:
        """Return a session-scoped pytest fixture that manages this queue.

        The generated fixture automatically skips when there is no CARLA to
        launch (``CARLA_EXECUTABLE`` unset and nothing installed), so callers do not need to
        add a manual ``pytest.skip`` guard.

        Args:
            fixture_name: Name under which the fixture is registered.
                Defaults to ``"carla_queue"``.

        Returns:
            A ``@pytest.fixture(scope="session")``-decorated function.
            Assign it to a module-level name in ``conftest.py`` or a test
            file so that pytest discovers it.

        Example::

            # conftest.py
            from autoware_carla_scenario import (
                CarlaScenarioFixture, EgoConfig, ScenarioQueue, SpawnTransform,
            )
            import typesafe_carla.carla as carla

            ego = EgoConfig(
                spawn_location=SpawnTransform(carla.Transform(carla.Location(x=0.0, y=0.0, z=0.5))),
                vehicle_type="vehicle.mini.cooper",
            )
            _queue = ScenarioQueue(map_name="Town10HD_Opt")
            carla_queue  = _queue.as_fixture()              # auto-skip baked in
            my_result    = CarlaScenarioFixture(MyScenario, ego, queue=_queue).as_fixture()

            # test_my.py
            def test_passes(my_result):
                assert my_result.passed
        """
        import pytest

        queue = self

        @pytest.fixture(scope="session", name=fixture_name)
        def _queue_fixture() -> Generator["ScenarioQueue", None, None]:
            if CarlaServerManager.executable() is None:
                pytest.skip(
                    f"Environment variable '{CarlaServerManager.ENV_VAR}' is not set "
                    "and no CARLA is installed. Skipping CARLA integration tests."
                )
            try:
                with queue:
                    queue.run_all()
                    yield queue
            except Exception as exc:
                pytest.skip(f"CARLA session failed: {exc}")

        return _queue_fixture

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "ScenarioQueue":
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()

    def __len__(self) -> int:
        return len(self._scenarios)


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _wait_with_progress(seconds: float, *, desc: str = "Waiting") -> None:
    """Sleep for *seconds* with a tqdm progress bar (0.1 s resolution)."""
    steps = max(1, int(seconds * 10))
    interval = seconds / steps
    if TYPE_CHECKING:
        for _ in range(steps):
            time.sleep(interval)
    else:
        for _ in tqdm(
            range(steps), desc=desc, bar_format="{desc}: {bar}| {elapsed}<{remaining}"
        ):
            time.sleep(interval)
