"""The digest of a concrete scenario: what a run is, its seed aside (docs/wosac.md).

Whether two runs are rollouts of one scenario -- one initial state, futures
that differ only by chance -- cannot be told from what they did: a future
that differs may be chance, or a parameter that differs.  It is told from
what they were asked to run: :func:`scenario_hash` digests the part of a run's
config that says what is simulated, its seed aside, so that runs of one hash
and different seeds are rollouts of one scenario.  The ``scenario`` command
gives each scenario it builds the hash of its config
(:attr:`~autoware_carla_scenario.BaseScenario.scenario_hash`), and the runner
writes it, with the seed, into the run's trajectory log, where
``scenario-wosac`` reads them.

What it digests is the scenario's class and :data:`SCENARIO_SECTIONS` of the
config -- the scenario and its parameters, the map, the ego's spawn and goal,
the environment, the controls, the background traffic, the traffic backend,
the ODD and the case drawn from it -- with the settings that say where a run
connects (:func:`_is_endpoint`: hosts, addresses, ports and ROS domains,
which a SLURM task moves to its own slot) left out.  What drives the ego (its
stack's settings, ``driver``, ``autoware``, ``ros2``) is not the scenario
and is left out too; how a run is recorded or reported (``rerun``,
``metrics``, ...) does not change what is simulated.  A component's own seed
(``background_traffic.seed``) is part of it: one pinned, so that every
rollout places the same vehicles, is what the scenario is.

Where the map's files sit is not the scenario either: the ``map`` section's
paths (:func:`_is_map_path`: ``xodr_path``, ``lanelet2_path``,
``directory``, any ``*_path`` or ``*_dir``) are left out, so a run reading
the map from an absolute path, another checkout or a
``NISHISHINJUKU_XODR_PATH`` digests as one reading the default relative
path (#2481).  The map is told by ``map.name``, and by ``map.source`` and
``map.pack`` (which name a revision) where they are set.  The files' content
is not read here; the run's trajectory log records each map file's SHA-256
(its ``map`` record, docs/trajectory_log.md).

Nor are where the traffic backend keeps its files and how it is watched or
recorded: of ``traffic.options``, :data:`TRAFFIC_LOCAL_OPTIONS` --
``cache_dir`` (where converted SUMO networks are cached on this machine),
``binary`` (the SUMO executable, ``sumo-gui`` to watch) and ``fcd_output``
(whether SUMO's floating car data is recorded) -- are left out, so a SLURM
node with a cache of its own runs rollouts of the same scenario (#2620).
The options that say what is simulated stay in, a ``net_path`` or a
``route_path`` among them: the network and demand they name are the
scenario.

The ``odd`` section is digested as written: a registered name
(``default``), a ``package.module:function``, or an OpenODD YAML file's
path, which counts as written -- the same file under another path digests
as another scenario, and the file's content is not read.  Give the runs one
spelling of it (a path relative to where they run, or a registered name).

A number is digested by its value, not its YAML spelling: a whole float is
digested as the integer it equals, so ``15`` and ``15.0`` are one
(``scenario.timeout_seconds=15`` against a YAML's ``15.0``, #2481).

Of the other sections, the settings in :data:`SCENARIO_SETTINGS` are part of
it too: ``server.fog_density_scale`` scales how thick ``environment.fog_density``
renders, to the cameras and to the LiDARs' fog alike, so it is the world's
fog as much as the environment's own number is (#2300).  It is digested only
when it is set: ``null``, the server's own scale, leaves the digest what it
was before it was digested at all.

This module imports nothing but the standard library.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

__all__ = [
    "OTHER_SECTIONS",
    "SCENARIO_SECTIONS",
    "SCENARIO_SETTINGS",
    "TRAFFIC_LOCAL_OPTIONS",
    "scenario_hash",
]

#: The top-level sections of a run's config that say what is simulated.
SCENARIO_SECTIONS: tuple[str, ...] = (
    "scenario",
    "map",
    "ego",
    "entity",
    "environment",
    "controls",
    "scenario_rules",
    "background_traffic",
    "traffic",
    "odd",
    "odd_case",
)

#: The top-level sections of a run's config that do not: the seed, what
#: drives the ego, where a run connects and how it is checked, recorded and
#: run.  Every section the ``scenario`` command's config has is one or the
#: other (``test_scenario_hash.py`` checks), so a new section has to be put
#: on one side rather than be left out of the hash unseen.
OTHER_SECTIONS: tuple[str, ...] = (
    "seed",
    "server",
    "traffic_manager",
    "driver",
    "autoware",
    "ros2",
    "hils",
    "launcher",
    "backends",
    "typecheck",
    "rerun",
    "metrics",
    "canvas",
    "video",
    "sweep",
    "t4",
    "t4scene",
)

#: The settings of :data:`OTHER_SECTIONS` that say what is simulated, as
#: ``(section, key)``: digested when set (not ``None``), the rest of their
#: section left out.  ``server.fog_density_scale`` is
#: ``carla.Weather.FogDensityScale``, which scales how thick the
#: environment's ``fog_density`` renders, LiDAR fog included.
SCENARIO_SETTINGS: tuple[tuple[str, str], ...] = (("server", "fog_density_scale"),)

#: The ``traffic.options`` that say where a run's traffic backend keeps its
#: files on this machine, or how it is watched or recorded, not what it
#: simulates: the SUMO network cache (``cache_dir``, the ``sumo`` and
#: ``hybrid`` backends'), the SUMO executable (``binary``) and whether its
#: floating car data is recorded (``fcd_output``).  Left out of the digest.
TRAFFIC_LOCAL_OPTIONS: tuple[str, ...] = ("cache_dir", "binary", "fcd_output")

#: Settings named so say where a run connects, not what it simulates.
_ENDPOINTS = frozenset(("host", "address", "port"))


def _is_endpoint(key: str) -> bool:
    """Whether a setting named *key* says where a run connects: a host, an
    address, a port (``port``, ``tm_port``, ...) or a ROS domain
    (``domain_id``, ``ros_domain_ids``, ``ROS_DOMAIN_ID``)."""
    name = key.lower()
    return name in _ENDPOINTS or name.endswith("_port") or "domain_id" in name


def _is_map_path(key: str) -> bool:
    """Whether a ``map`` setting named *key* says where the map's files sit
    on this machine (``xodr_path``, ``lanelet2_path``, ``directory``, any
    ``*_path`` or ``*_dir``) rather than which map it is."""
    name = key.lower()
    return name in ("path", "directory") or name.endswith(("_path", "_dir"))


def _canonical(value: Any) -> Any:
    """*value* as it is digested: every mapping's endpoint settings
    (:func:`_is_endpoint`) left out, at any depth, and every whole float
    (``15.0``) the integer it equals, so that equal numbers digest alike."""
    if isinstance(value, Mapping):
        named = ((str(key), item) for key, item in value.items())
        return {key: _canonical(item) for key, item in named if not _is_endpoint(key)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def scenario_hash(config: Mapping[str, Any], scenario_class: str) -> str:
    """The SHA-256 (hex) of the concrete scenario *config* runs.

    Args:
        config: A run's config, resolved, as plain mappings and lists (what
            ``OmegaConf.to_container(cfg, resolve=True)`` returns).  Only its
            :data:`SCENARIO_SECTIONS` are read, without their endpoint
            settings, the ``map`` section's paths and the
            :data:`TRAFFIC_LOCAL_OPTIONS` of ``traffic.options``, and the
            :data:`SCENARIO_SETTINGS` set in it; ``seed`` is not one of them.
            A number is digested by its value: ``15`` and ``15.0`` are one,
            as are ``0.01`` and ``1e-2``.
        scenario_class: The scenario's class, by its qualified name
            (``module.QualName``): the code a run of the config runs.

    The digest is of the canonical JSON of both (keys sorted, a value JSON
    has no type for written as its ``str()``), so it does not depend on the
    order the config holds its keys in.
    """
    sections = {
        key: _canonical(config[key]) for key in SCENARIO_SECTIONS if key in config
    }
    map_section = sections.get("map")
    if isinstance(map_section, dict):
        sections["map"] = {
            key: item for key, item in map_section.items() if not _is_map_path(key)
        }
    traffic = sections.get("traffic")
    if isinstance(traffic, dict):
        options = traffic.get("options")
        if isinstance(options, dict):
            traffic["options"] = {
                key: item
                for key, item in options.items()
                if key not in TRAFFIC_LOCAL_OPTIONS
            }
    for section, key in SCENARIO_SETTINGS:
        settings = config.get(section)
        value = settings.get(key) if isinstance(settings, Mapping) else None
        if value is None:
            continue
        # Dotted, so it names no section of its own.
        sections[f"{section}.{key}"] = _canonical(value)
    text = json.dumps(
        {"class": scenario_class, "config": sections},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(text.encode()).hexdigest()
