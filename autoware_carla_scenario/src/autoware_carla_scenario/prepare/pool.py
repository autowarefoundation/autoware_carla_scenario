"""The preparer: one backend of each name, started once, stopped at the end.

:class:`BackendPool` is the
:class:`~autoware_carla_scenario.prepare.BackendPreparer` a queue is given.
It builds each backend a scenario or an action declares
(:func:`~autoware_carla_scenario.prepare.declare.need_prepare`) from its
settings (``backends.<name>`` of the run's config), checks the host has room
for every one it is about to start (:func:`~.host.assign_gpus`), starts them,
and keeps them for the rest of the queue: the next scenario that needs one
finds it running.

An action reaches the backend it needs through :func:`running_backend`: the
pool that started it is the process's active one while it runs.
"""

from __future__ import annotations

import atexit
import logging
import threading
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from . import Backend, BackendError
from .declare import action_backends, scenario_backends
from .host import assign_gpus, probe_host

logger = logging.getLogger(__name__)

__all__ = ["BackendPool", "running_backend"]


class BackendPool:
    """Starts the backends a queue's scenarios need, and stops them.

    Args:
        settings: Each backend's settings, by backend name (``backends`` of
            the run's config); a backend with none takes its defaults.
        check_resources: Refuse a host without room for the backends before
            starting them (off only for tests).
    """

    _settings: dict[str, Mapping[str, Any]]
    _check_resources: bool
    _running: dict[str, Backend]
    _lock: threading.Lock

    def __init__(
        self,
        settings: Optional[Mapping[str, Mapping[str, Any]]] = None,
        *,
        check_resources: bool = True,
    ) -> None:
        self._settings = {str(k): dict(v or {}) for k, v in (settings or {}).items()}
        self._check_resources = check_resources
        self._running = {}
        self._lock = threading.Lock()

    def prepare_scenarios(
        self, scenarios: Sequence[object], log_dir: Optional[Path] = None
    ) -> None:
        """Start what *scenarios* declare they need (before ``setup()``)."""
        needed: dict[type, None] = {}
        for scenario in scenarios:
            needed.update(dict.fromkeys(scenario_backends(scenario)))
        self._prepare(tuple(needed), log_dir)

    def prepare_actions(
        self, actions: Sequence[object], log_dir: Optional[Path] = None
    ) -> None:
        """Start what *actions* need and is not running yet."""
        self._prepare(action_backends(actions), log_dir)

    def backend(self, name: str) -> Optional[Backend]:
        """The running backend called *name*, if one is."""
        return self._running.get(name)

    def close(self) -> None:
        """Stop every backend, the last started first.  Never raises."""
        with self._lock:
            running = list(self._running.values())
            self._running.clear()
        for backend in reversed(running):
            try:
                backend.close()
            except Exception:  # noqa: BLE001 - close() must not raise
                logger.exception("Stopping backend '%s' failed", backend.name)
        _deactivate(self)

    def _prepare(self, classes: Sequence[type], log_dir: Optional[Path]) -> None:
        with self._lock:
            _activate(self)
            wanted: list[Backend] = []
            for cls in classes:
                name = str(getattr(cls, "backend_name"))
                if name in self._running or any(b.name == name for b in wanted):
                    continue
                settings = self._settings.get(name, {})
                wanted.append(getattr(cls, "from_settings")(settings))
            if not wanted:
                return
            if self._check_resources:
                gpus = assign_gpus(
                    [(backend.name, backend.resources()) for backend in wanted],
                    probe_host(),
                )
            else:
                gpus = {backend.name: None for backend in wanted}
            for backend in wanted:
                gpu = gpus.get(backend.name)
                logger.info(
                    "Starting backend '%s'%s",
                    backend.name,
                    "" if gpu is None else f" on GPU {gpu.index}",
                )
                try:
                    backend.start(log_dir, None if gpu is None else gpu.index)
                except BaseException:
                    backend.close()
                    raise
                self._running[backend.name] = backend
                logger.info(
                    "Backend '%s' answers on %s", backend.name, backend.address()
                )


#: The pool whose backends the running actions use.
_ACTIVE: Optional[BackendPool] = None
#: The pool an action starts its own backend in when no queue prepared one
#: (a scenario run without a queue); stopped when the process exits.
_FALLBACK: Optional[BackendPool] = None


def _activate(pool: BackendPool) -> None:
    global _ACTIVE
    _ACTIVE = pool


def _deactivate(pool: BackendPool) -> None:
    global _ACTIVE
    if _ACTIVE is pool:
        _ACTIVE = None


def running_backend(cls: type) -> Backend:
    """The running backend of class *cls*, started now if none is.

    The active pool's (the queue's) when one is; otherwise one this process
    starts on its own and stops at exit -- with the defaults of the backend
    (no ``backends`` settings reach it there).

    Raises:
        InsufficientResourcesError: If it has to be started and the host
            cannot run it.
        BackendError: If it could not be started.
    """
    global _FALLBACK
    name = str(getattr(cls, "backend_name"))
    pool = _ACTIVE
    if pool is None:
        if _FALLBACK is None:
            _FALLBACK = BackendPool()
            atexit.register(_FALLBACK.close)
        pool = _FALLBACK
    found = pool.backend(name)
    if found is None:
        logger.warning(
            "Backend '%s' was not prepared before the run; starting it now", name
        )
        pool._prepare((cls,), None)
        found = pool.backend(name)
    if found is None:  # pragma: no cover - _prepare raises instead
        raise BackendError(f"backend '{name}' did not start")
    return found
