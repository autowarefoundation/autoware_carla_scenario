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

#: Settings named so say where a run connects, not what it simulates.
_ENDPOINTS = frozenset(("host", "address", "port"))


def _is_endpoint(key: str) -> bool:
    """Whether a setting named *key* says where a run connects: a host, an
    address, a port (``port``, ``tm_port``, ...) or a ROS domain
    (``domain_id``, ``ros_domain_ids``, ``ROS_DOMAIN_ID``)."""
    name = key.lower()
    return name in _ENDPOINTS or name.endswith("_port") or "domain_id" in name


def _without_endpoints(value: Any) -> Any:
    """*value* with every mapping's endpoint settings (:func:`_is_endpoint`)
    left out, at any depth."""
    if isinstance(value, Mapping):
        named = ((str(key), item) for key, item in value.items())
        return {
            key: _without_endpoints(item)
            for key, item in named
            if not _is_endpoint(key)
        }
    if isinstance(value, (list, tuple)):
        return [_without_endpoints(item) for item in value]
    return value


def scenario_hash(config: Mapping[str, Any], scenario_class: str) -> str:
    """The SHA-256 (hex) of the concrete scenario *config* runs.

    Args:
        config: A run's config, resolved, as plain mappings and lists (what
            ``OmegaConf.to_container(cfg, resolve=True)`` returns).  Only its
            :data:`SCENARIO_SECTIONS` are read, without their endpoint
            settings; ``seed`` is not one of them.
        scenario_class: The scenario's class, by its qualified name
            (``module.QualName``): the code a run of the config runs.

    The digest is of the canonical JSON of both (keys sorted, a value JSON
    has no type for written as its ``str()``), so it does not depend on the
    order the config holds its keys in.
    """
    sections = {
        key: _without_endpoints(config[key])
        for key in SCENARIO_SECTIONS
        if key in config
    }
    text = json.dumps(
        {"class": scenario_class, "config": sections},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(text.encode()).hexdigest()
