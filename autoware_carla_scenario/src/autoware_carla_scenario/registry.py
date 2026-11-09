"""Pluggable registry for scenarios and their Hydra config directories.

This module is the extension point that lets scenarios live **outside** the
``autoware_carla_scenario`` package.  An external "scenario package" only needs
to:

1. Define its scenario class (a :class:`BaseScenario` subclass) and a config
   dataclass.
2. Ship its concrete YAML configs under a ``conf/`` directory.
3. Register both at import time::

       from pathlib import Path
       from autoware_carla_scenario import register_conf_dir, register_scenario

       from .my_scenario import MyScenario
       from .configs import MyScenarioConfig


       def register() -> None:
           register_scenario("my_scenario", MyScenario, MyScenarioConfig)
           register_conf_dir(Path(__file__).parent / "conf")

4. Advertise ``register`` via a ``autoware_carla_scenario.scenarios`` entry
   point in ``pyproject.toml`` so the ``scenario`` CLI discovers it
   automatically::

       [project.entry-points."autoware_carla_scenario.scenarios"]
       my_scenario = "my_scenario_package:register"

The module is deliberately free of heavy imports (no CARLA, no lanelet2) so it
can be imported cheaply from anywhere -- the scenario/config classes are only
referenced lazily inside the builder closures.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .authoring.models import ScenarioDocument
    from .coordinate import GroundProjectionConfig, Lanelet2Pose
    from .scenario_base import BaseScenario, EgoConfig
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)

__all__ = [
    "ScenarioBuilder",
    "BuildScenarioFn",
    "register_scenario",
    "register_scenario_builder",
    "register_scenario_document",
    "unregister_scenario",
    "get_scenario_registry",
    "get_scenario_builder",
    "get_scenario_classes",
    "get_scenario_document",
    "register_conf_dir",
    "unregister_conf_dir",
    "get_conf_dirs",
    "load_scenario_plugins",
    "load_entry_point_plugins",
    "SCENARIO_ENTRY_POINT_GROUP",
]

#: Entry-point group that external scenario packages advertise in their
#: ``pyproject.toml`` to be auto-discovered by :func:`load_scenario_plugins`.
SCENARIO_ENTRY_POINT_GROUP = "autoware_carla_scenario.scenarios"

#: Callable that creates a :class:`BaseScenario` from its decomposed parts.
#: Signature: ``(ego, scenario_dict, spawn_pose, ground_projection) -> scenario``
ScenarioBuilder = Callable[
    ["EgoConfig", dict[str, Any], "Lanelet2Pose", "GroundProjectionConfig"],
    "BaseScenario",
]

#: Callable that loads the Scenario IR document a declarative scenario runs.
ScenarioDocumentLoader = Callable[[], "ScenarioDocument"]

#: Callable that replaces the entire ``build_scenario`` logic.
#: Signature: ``(cfg) -> (ego, scenario)``
BuildScenarioFn = Callable[["DictConfig"], "tuple[EgoConfig, BaseScenario]"]


# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------

_SCENARIO_REGISTRY: dict[str, ScenarioBuilder] = {}

#: The classes given to :func:`register_scenario`, for the static check
#: (:mod:`autoware_carla_scenario.typecheck`), which compiles the scenario the
#: way its builder constructs it.  A custom builder has no entry.
_SCENARIO_CLASSES: dict[str, tuple[type[BaseScenario], type]] = {}

#: The documents of declarative scenarios (an exported scenario package's), by
#: name: what a run checks its composed config against before anything starts
#: (``examples.run.check_scenario_maps``).  A scenario written in Python has
#: none.
_SCENARIO_DOCUMENTS: dict[str, ScenarioDocumentLoader] = {}


def register_scenario(
    name: str,
    scenario_cls: type[BaseScenario],
    config_cls: type,
) -> None:
    """Register a scenario class and its config class under *name*.

    This creates a standard builder that follows the convention used by all
    built-in scenarios::

        scenario_cls(ego, config=config_cls(**scenario_dict),
                     spawn_pose=spawn_pose, ground_projection=ground_projection)

    Downstream projects can call this at import time to make their custom
    scenarios available to the CLI runner::

        from autoware_carla_scenario import register_scenario
        register_scenario("my_scenario", MyScenario, MyScenarioConfig)
    """

    def _builder(
        ego: EgoConfig,
        scenario_dict: dict[str, Any],
        spawn_pose: Lanelet2Pose,
        ground_projection: GroundProjectionConfig,
    ) -> BaseScenario:
        config = config_cls(**scenario_dict)
        return scenario_cls(  # type: ignore[call-arg]
            ego,
            config=config,
            spawn_pose=spawn_pose,
            ground_projection=ground_projection,
        )

    _SCENARIO_REGISTRY[name] = _builder
    _SCENARIO_CLASSES[name] = (scenario_cls, config_cls)
    _SCENARIO_DOCUMENTS.pop(name, None)


def register_scenario_builder(name: str, builder: ScenarioBuilder) -> None:
    """Register a custom builder function under *name*.

    Use this instead of :func:`register_scenario` when the scenario
    constructor does not follow the standard ``(ego, config=..., ...)``
    pattern and you need full control over instantiation.
    """
    _SCENARIO_REGISTRY[name] = builder
    _SCENARIO_CLASSES.pop(name, None)
    _SCENARIO_DOCUMENTS.pop(name, None)


def register_scenario_document(name: str, loader: ScenarioDocumentLoader) -> None:
    """Register how to load the document the scenario *name* runs.

    For a declarative scenario -- one whose builder runs a Scenario IR
    document, as an exported scenario package's does -- called after its
    builder is registered (registering a builder drops the document of the
    one it replaces).  The run loads it before anything starts to check the
    composed config against it: a document naming its own map refuses a run
    on another (``examples.run.check_scenario_maps``).

    The loader is called on every :func:`get_scenario_document`, with no
    caching here: each lookup reads the document as it is then.
    """
    _SCENARIO_DOCUMENTS[name] = loader


def get_scenario_document(name: str) -> ScenarioDocument | None:
    """Return the document registered for the scenario *name*, loaded.

    ``None`` when the scenario registered none (:func:`register_scenario_document`).

    Raises:
        Whatever the loader raises: a document that cannot be loaded, which
        the scenario's builder would refuse as well.
    """
    loader = _SCENARIO_DOCUMENTS.get(name)
    return None if loader is None else loader()


def get_scenario_registry() -> dict[str, ScenarioBuilder]:
    """Return a **copy** of the current scenario registry.

    Useful for introspection (e.g. listing available scenarios).
    """
    return dict(_SCENARIO_REGISTRY)


def get_scenario_builder(name: str) -> ScenarioBuilder | None:
    """Return the builder registered under *name*, or ``None`` if absent.

    Public lookup helper so callers don't reach into the internal registry
    dict.  Use :func:`get_scenario_registry` when you need the full mapping
    (e.g. to list the registered names).
    """
    return _SCENARIO_REGISTRY.get(name)


def get_scenario_classes(name: str) -> tuple[type[BaseScenario], type] | None:
    """Return the ``(scenario_cls, config_cls)`` registered under *name*.

    ``None`` when *name* is unknown or was registered with a custom builder
    (:func:`register_scenario_builder`), whose construction only the builder
    knows.
    """
    return _SCENARIO_CLASSES.get(name)


def unregister_scenario(name: str) -> None:
    """Remove the scenario registered under *name*, if present.

    Inverse of :func:`register_scenario` / :func:`register_scenario_builder`.
    Unknown names are ignored, so this is safe to call unconditionally (e.g.
    for test cleanup or to swap a scenario at runtime).
    """
    _SCENARIO_REGISTRY.pop(name, None)
    _SCENARIO_CLASSES.pop(name, None)
    _SCENARIO_DOCUMENTS.pop(name, None)


# ---------------------------------------------------------------------------
# Config directory registry
# ---------------------------------------------------------------------------
#
# Every scenario package may contribute a ``conf/`` directory containing Hydra
# config groups (``scenario/``, ``map/``, ``ego/`` ...).  The CLI runner adds
# every registered directory to Hydra's search path so that
# ``scenario=<group>/<name>`` resolves regardless of which package owns it.

_CONF_DIRS: list[Path] = []

#: The framework's own config directory, the ``scenario`` CLI's primary Hydra
#: config directory: whatever a scenario package ships at the same path is
#: shadowed by it, whether or not it was registered (yet).
_FRAMEWORK_CONF_DIR = Path(__file__).resolve().parent / "examples" / "conf"

#: How many shadowed config paths a collision warning lists by name.
_SHOWN_COLLISIONS = 10

#: Each config path (``scenario/lane_change/left.yaml``) and the first
#: directory on the search path that provides it -- the one Hydra composes it
#: from.  Kept so that registering a directory scans only that directory: the
#: editor exports a package per scenario, so hundreds of registered
#: directories are plausible, and every CLI start registers them all.
_CONFIG_INDEX: dict[str, Path] = {}

#: The directories :data:`_CONFIG_INDEX` was built from, in search-path order.
#: When it no longer matches the registered ones (a directory was unregistered,
#: or a test replaced :data:`_CONF_DIRS`) the index is rebuilt from
#: :data:`_DIR_CONFIG_PATHS`, without rescanning.
_INDEXED_DIRS: list[Path] = []

#: The config paths each indexed directory provides, as scanned when it was
#: registered.
_DIR_CONFIG_PATHS: dict[Path, frozenset[str]] = {}


def register_conf_dir(path: str | Path) -> None:
    """Register a Hydra config directory contributed by a scenario package.

    The directory is expected to contain config groups such as
    ``scenario/``, ``map/``, ``ego/`` (mirroring the built-in layout).  It is
    appended to Hydra's search path so its config groups become selectable from
    the ``scenario`` CLI.  Registering the same directory twice is a no-op.

    Hydra composes a config path (``scenario/lane_change/left.yaml``) from the
    first directory on its search path that has it, so a directory registered
    later cannot replace a config an earlier one -- the framework's own, or
    another package's -- already provides.  Every such path is logged as a
    warning naming both directories and the one Hydra uses (#2034).  Only the
    new directory is scanned: what the earlier ones provide is indexed.
    """
    resolved = Path(path).resolve()
    if resolved in _CONF_DIRS:
        return
    if resolved != _FRAMEWORK_CONF_DIR:
        _sync_config_index()
        provided = _config_paths(resolved)
        _DIR_CONFIG_PATHS[resolved] = provided
        _warn_about_shadowed_configs(resolved, provided)
        for config in provided:
            _CONFIG_INDEX.setdefault(config, resolved)
        _INDEXED_DIRS.append(resolved)
    _CONF_DIRS.append(resolved)


def _indexed_order() -> list[Path]:
    """Return the directories the index covers, in search-path order.

    The framework's own comes first whether or not it is registered: it is
    the ``scenario`` CLI's primary config directory.
    """
    return [_FRAMEWORK_CONF_DIR] + [
        conf_dir for conf_dir in _CONF_DIRS if conf_dir != _FRAMEWORK_CONF_DIR
    ]


def _sync_config_index() -> None:
    """Rebuild :data:`_CONFIG_INDEX` if the registered directories changed."""
    order = _indexed_order()
    if _INDEXED_DIRS == order:
        return
    _CONFIG_INDEX.clear()
    for conf_dir in order:
        if conf_dir not in _DIR_CONFIG_PATHS:
            _DIR_CONFIG_PATHS[conf_dir] = _config_paths(conf_dir)
        for config in _DIR_CONFIG_PATHS[conf_dir]:
            _CONFIG_INDEX.setdefault(config, conf_dir)
    _INDEXED_DIRS[:] = order


def _config_paths(directory: Path) -> frozenset[str]:
    """Return the config paths *directory* provides, relative and POSIX."""
    if not directory.is_dir():
        return frozenset()
    return frozenset(
        config.relative_to(directory).as_posix()
        for config in directory.rglob("*.yaml")
        if config.is_file()
    )


def _warn_about_shadowed_configs(directory: Path, provided: frozenset[str]) -> None:
    """Warn about every config path of *directory* an earlier one provides.

    One warning per earlier directory, naming each directory once.
    """
    shadowed: dict[Path, list[str]] = {}
    for config in sorted(provided):
        earlier_dir = _CONFIG_INDEX.get(config)
        if earlier_dir is not None:
            shadowed.setdefault(earlier_dir, []).append(config)
    for earlier_dir, configs in shadowed.items():
        shown = ", ".join(configs[:_SHOWN_COLLISIONS])
        if len(configs) > _SHOWN_COLLISIONS:
            shown += f" and {len(configs) - _SHOWN_COLLISIONS} more"
        logger.warning(
            "The config directory %s ships %d config path(s) already provided "
            "by %s, which comes earlier on Hydra's search path: %s. Hydra "
            "composes these from the earlier directory and never reads the "
            "copies here; rename them (or the scenario) in the package that "
            "registered this directory.",
            directory,
            len(configs),
            earlier_dir,
            shown,
        )


def unregister_conf_dir(path: str | Path) -> None:
    """Remove a previously registered config directory, if present.

    Inverse of :func:`register_conf_dir`.  Unregistered paths are ignored.
    """
    resolved = Path(path).resolve()
    try:
        _CONF_DIRS.remove(resolved)
    except ValueError:
        return
    if resolved != _FRAMEWORK_CONF_DIR:
        # Rescanned if it is registered again: its files may have changed.
        _DIR_CONFIG_PATHS.pop(resolved, None)


def get_conf_dirs() -> list[Path]:
    """Return a copy of the ordered list of registered config directories.

    The first entry is the built-in directory (registered by the runner); any
    directories contributed by external packages follow in registration order.
    """
    return list(_CONF_DIRS)


# ---------------------------------------------------------------------------
# Entry-point based plugin discovery
# ---------------------------------------------------------------------------

_plugins_loaded = False


def load_scenario_plugins() -> None:
    """Discover and load external scenario packages via entry points.

    Every installed distribution that advertises an entry point in the
    :data:`SCENARIO_ENTRY_POINT_GROUP` group is imported and its target called
    with no arguments.  The target is expected to register scenarios and/or
    config directories (see the module docstring).

    The entry points load in a fixed order, by distribution name and then
    entry-point name (:func:`load_entry_point_plugins`), so their config
    directories join Hydra's search path in that order.

    A plugin that registers a scenario name an earlier plugin of the walk
    registered (its builder, classes or document) is refused as a whole: every
    scenario registration it made and every config directory it registered
    are undone, with a warning naming both plugins (#2034).  So the first
    package keeps the name, and a run uses one package's class and YAML: no
    config the refused package ships is on the search path.  What the refused
    plugin did to other registries at import -- traffic backends, ODD
    builders, authoring specs, ... -- is not undone.

    A plugin that replaces a scenario registered before the walk -- a
    built-in, or a package registered explicitly (the tutorial's "No entry
    point?" tip) -- still replaces it, with a warning (#1802).

    Loading is idempotent: subsequent calls are no-ops within a single process.
    A failing plugin is logged and skipped so that one broken package cannot
    prevent the others (or the built-in scenarios) from running.
    """
    global _plugins_loaded
    if _plugins_loaded:
        return
    _plugins_loaded = True

    load_entry_point_plugins(
        SCENARIO_ENTRY_POINT_GROUP,
        "scenario",
        snapshot=_scenario_snapshot,
        checkpoint=_scenario_checkpoint,
    )


def load_entry_point_plugins(
    group: str,
    what: str,
    snapshot: Callable[[], Mapping[str, object]] | None = None,
    *,
    checkpoint: Callable[[], Callable[[], None]] | None = None,
    override_intended: bool = False,
) -> None:
    """Import every distribution advertising *group* and call what it names.

    Each entry point resolves to a zero-argument callable that registers the
    package's contribution.  A plugin that fails to import is logged and
    skipped: one broken third-party package must not stop a run that does not
    use it.

    The entry points are called sorted by distribution name (normalised as
    PEP 503 does) and then by entry-point name and value, not in the order the
    environment lists them (the directory order of site-packages, which
    differs from one machine to the next), so which plugin wins a collision
    does not depend on the environment (#2034).

    Shared with :func:`autoware_carla_scenario.traffic.registry.load_traffic_backend_plugins`,
    which discovers traffic backends the same way -- the walk is the same, only
    the group and the word in the log line differ.  Callers own their own
    idempotence latch, because "loaded already" is a fact about a group.

    Args:
        group: The entry-point group to walk.
        what: What the plugins are, for the log lines (e.g. ``"scenario"``).
        snapshot: Returns what the registry the plugins add to holds, by
            name, if the caller wants to hear about collisions.  Only entries
            whose value changed (``!=``) count; a plugin that adds new names,
            removes one or registers the same thing again is quiet.  A plugin
            that replaces a name registered before the walk (a built-in, or
            one registered explicitly) is warned about, naming the entry
            point and every name it replaced (#1802); one that replaces a name
            an earlier plugin of the walk registered is warned about naming
            both entry points (#2034).
        checkpoint: Requires *snapshot*.  Returns a callable that undoes
            everything the plugin registered since ``checkpoint`` was called.
            With it, a plugin that replaces a name an earlier plugin of the
            walk registered is refused as a whole, so the first plugin keeps
            the name.  Without it, the last plugin wins.
        override_intended: Whether replacing a name registered before the
            walk is something a plugin may mean to do (a traffic backend
            overriding a built-in one), which is reported in softer words.
    """
    import importlib.metadata as importlib_metadata  # noqa: PLC0415

    # The selectable ``group=`` API exists on all supported interpreters
    # (requires-python >= 3.10).
    entry_points = sorted(
        importlib_metadata.entry_points(group=group), key=_entry_point_sort_key
    )
    # The entry point that registered each name during this walk.
    owners: dict[str, Any] = {}
    for entry_point in entry_points:
        before = dict(snapshot()) if snapshot is not None else {}
        undo = checkpoint() if checkpoint is not None else None
        try:
            entry_point.load()()
            logger.info(
                "Loaded %s plugin %r from %s",
                what,
                entry_point.name,
                getattr(entry_point, "value", entry_point),
            )
        except Exception:  # noqa: BLE001 -- one bad plugin must not kill the CLI
            logger.exception(
                "Failed to load %s plugin %r; skipping.", what, entry_point.name
            )
        if snapshot is None:
            continue
        # Checked even for a plugin that then failed: what it registered
        # before the failure stays registered (unless it is refused).
        after = snapshot()
        changed = sorted(
            name
            for name, entry in after.items()
            if name not in before or before[name] != entry
        )
        replaced_earlier = [name for name in changed if name in before]
        replaced_plugins = [name for name in replaced_earlier if name in owners]
        replaced_builtins = [name for name in replaced_earlier if name not in owners]
        if undo is not None and replaced_plugins:
            undo()
            _warn_about_refusal(entry_point, what, replaced_plugins, owners)
            continue
        if replaced_builtins:
            _warn_about_replacements(
                entry_point, what, replaced_builtins, override_intended
            )
        for name in replaced_plugins:
            _warn_about_collision(owners[name], entry_point, what, name)
        for name in changed:
            owners[name] = entry_point


def _entry_point_sort_key(entry_point: Any) -> tuple[str, str, str]:
    """Order entry points by distribution name, then name, then value."""
    distribution = getattr(getattr(entry_point, "dist", None), "name", None)
    if not isinstance(distribution, str):
        distribution = ""
    return (
        re.sub(r"[-_.]+", "-", distribution).lower(),
        str(entry_point.name),
        str(getattr(entry_point, "value", "")),
    )


def _describe(entry_point: Any) -> str:
    """Name *entry_point* and what it points at, for a log line."""
    return f"{entry_point.name!r} ({getattr(entry_point, 'value', entry_point)})"


def _scenario_snapshot() -> dict[str, object]:
    """Return what each registered scenario name stands for, to compare.

    The ``(scenario_cls, config_cls)`` pair for a scenario
    :func:`register_scenario` registered, the builder itself for one
    :func:`register_scenario_builder` did.  Not the builder in both cases:
    :func:`register_scenario` makes a new closure on every call, so a package
    whose ``register()`` ran explicitly before its entry point did (the
    tutorial's "No entry point?" tip), or a distribution on the path twice,
    re-registers the same pair, which replaces nothing.

    With the document loader, if there is one, by where it is defined (its
    module and qualified name), not by identity: a ``register()`` that runs
    twice makes a new closure each time, but it is the same document, while a
    plugin that registers a document for another package's scenario changes
    what that scenario runs.
    """
    names = list(_SCENARIO_REGISTRY)
    names += [name for name in _SCENARIO_DOCUMENTS if name not in _SCENARIO_REGISTRY]
    snapshot: dict[str, object] = {}
    for name in names:
        builder = _SCENARIO_REGISTRY.get(name)
        what = _SCENARIO_CLASSES.get(name, builder)
        loader = _SCENARIO_DOCUMENTS.get(name)
        if loader is None:
            snapshot[name] = what
        else:
            snapshot[name] = (
                what,
                getattr(loader, "__module__", None),
                getattr(loader, "__qualname__", loader),
            )
    return snapshot


def _scenario_checkpoint() -> Callable[[], None]:
    """Save the scenario and config-directory registries, to undo a plugin.

    The callable it returns puts the scenario tables back as they are now --
    every name the plugin added, replaced or removed -- and unregisters every
    config directory registered since, so nothing a refused plugin registered
    here stays.  Other registries are not saved.
    """
    tables: list[dict[str, Any]] = [
        _SCENARIO_REGISTRY,
        _SCENARIO_CLASSES,
        _SCENARIO_DOCUMENTS,
    ]
    saved = [dict(table) for table in tables]
    conf_dirs = list(_CONF_DIRS)

    def undo() -> None:
        for table, copy in zip(tables, saved):
            table.clear()
            table.update(copy)
        for conf_dir in [d for d in _CONF_DIRS if d not in conf_dirs]:
            unregister_conf_dir(conf_dir)

    return undo


def _warn_about_replacements(
    entry_point: Any,
    what: str,
    replaced: list[str],
    override_intended: bool,
) -> None:
    """Warn that *entry_point* replaced names registered before the walk."""
    names = ", ".join(repr(name) for name in replaced)
    if override_intended:
        logger.warning(
            "The %s plugin %s overrides %s, registered before the plugins "
            "loaded (a built-in %s, or one registered explicitly): that name "
            "now selects the plugin's %s.",
            what,
            _describe(entry_point),
            names,
            what,
            what,
        )
        return
    logger.warning(
        "The %s plugin %s replaced %s, registered before the plugins loaded (a "
        "built-in %s, or one registered explicitly): a config that names it "
        "now builds the plugin's %s, but Hydra still composes every config "
        "path the framework's conf directory (or an earlier one) ships from "
        "there, not from the plugin's -- a built-in config such as "
        "scenario=<name>/<config> hands its keys to the plugin's class. "
        "Rename the plugin's %s.",
        what,
        _describe(entry_point),
        names,
        what,
        what,
        what,
    )


def _warn_about_refusal(
    refused: Any, what: str, names: list[str], owners: Mapping[str, Any]
) -> None:
    """Warn that *refused* was not loaded: it took names earlier plugins own."""
    taken = "; ".join(f"{name!r} from {_describe(owners[name])}" for name in names)
    logger.warning(
        "The %s plugin %s was not loaded: it registers %s(s) an earlier plugin "
        "of the walk already registered (%s). Plugins load sorted by "
        "distribution and entry-point name and the first keeps the name, so "
        "every %s registration of %s and every config directory it registered "
        "were undone, and a run uses one package's class and YAML. What it "
        "registered elsewhere at import (traffic backends, ODD builders, ...) "
        "stays. Rename the %s in one of the packages.",
        what,
        _describe(refused),
        what,
        taken,
        what,
        repr(refused.name),
        what,
    )


def _warn_about_collision(first: Any, later: Any, what: str, name: str) -> None:
    """Warn that two plugins of the same walk registered *name*; the last won."""
    logger.warning(
        "The %s plugins %s and %s both register the %s %r. %s loaded last "
        "(plugins load sorted by distribution and entry-point name), so %r "
        "selects its %s and that of %s is gone. Rename the %s in one of the "
        "packages.",
        what,
        _describe(first),
        _describe(later),
        what,
        name,
        repr(later.name),
        name,
        what,
        repr(first.name),
        what,
    )
