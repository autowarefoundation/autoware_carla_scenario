"""The CARLA release the framework targets, and the client build that speaks to it.

The framework targets CARLA 1.0 (Unreal Engine 5.8): ``scenario-setup``
installs it (:data:`~autoware_carla_scenario.carla_install.RELEASE_URL`) and
the client speaks its RPC.

typesafe_carla's wheel carries one LibCarla per CARLA branch it supports
(``ue5-dev``, ``ue58-dev``, ``1.0``), picked when ``typesafe_carla.carla`` is
first imported, by ``TYPESAFE_CARLA_CARLA_REF``; unset, it is ``ue5-dev``, a
client for CARLA 0.10. The RPC messages differ between the branches (CARLA 1.0's
``EpisodeSettings`` has fields 0.10's has not), so a client of the wrong branch
fails on the first such call -- ``World.apply_settings``, which every run makes
-- with a msgpack error that names neither. :func:`select_default_client`
makes the 1.0 client the default before anything imports typesafe_carla (the
package's ``__init__`` calls it), and :func:`client_server_mismatch` turns a
client of another branch than the server's into an error that says what to
set, checked as soon as a server is reached
(:meth:`~autoware_carla_scenario.server.CarlaServerManager.start`).

Another server wants its own client, chosen in the environment of the process
that imports the framework (a ``.env`` file is read too late: the client is
loaded by then)::

    TYPESAFE_CARLA_CARLA_REF=ue58-dev scenario ...  # CARLA's nightly (scenario-setup --nightly)
    TYPESAFE_CARLA_CARLA_REF=ue5-dev scenario ...   # a CARLA 0.10 / ue5-dev server
"""

from __future__ import annotations

import os
from typing import Optional

__all__ = [
    "CARLA_REF_ENV",
    "CARLA_RELEASE",
    "DEFAULT_CARLA_REF",
    "client_server_mismatch",
    "default_max_steer_angle_deg",
    "default_steer_exponent",
    "select_default_client",
    "server_family",
    "version_family",
    "walker_max_speed_mps",
    "walker_speed_scale",
]

CARLA_RELEASE = "1.0"
"""The CARLA release the framework targets."""

CARLA_REF_ENV = "TYPESAFE_CARLA_CARLA_REF"
"""The variable typesafe_carla picks its LibCarla (CARLA branch) by."""

DEFAULT_CARLA_REF = "1.0"
"""The LibCarla the framework's client uses unless :data:`CARLA_REF_ENV` says
otherwise: the CARLA 1.0 release's."""

#: The client to name for each server family, in :func:`client_server_mismatch`.
_REF_FOR_FAMILY = {
    "1": "1.0 (or ue58-dev for CARLA's nightly)",
    "0.10": "ue5-dev",
}


def select_default_client() -> None:
    """Make the CARLA 1.0 client typesafe_carla's default, unless one is chosen.

    Takes effect only before ``typesafe_carla.carla`` is first imported in the
    process; it is set in the environment, so every child process (a SLURM
    worker, a Codon build) inherits it.
    """
    os.environ.setdefault(CARLA_REF_ENV, DEFAULT_CARLA_REF)


def version_family(version: str) -> str:
    """The CARLA branch family a ``carla::version()`` string belongs to.

    ``"1"`` for CARLA 1.x (``1.0``; a nightly or source build reports its
    commit too, ``1.0-82c52a7c2``), ``"0.10"`` for 0.10 and its ``ue5-dev``
    builds (``0.10.0``, ``0.10.0-185-g0a5ce0d5b``). Clients and servers of one
    family speak the same RPC.
    """
    head = version.strip().split("-", 1)[0]
    parts = head.split(".")
    if parts[0] == "0" and len(parts) > 1:
        return f"0.{parts[1]}"
    return parts[0]


def server_family() -> str:
    """The family (:func:`version_family`) of the server the runs speak to.

    Read off the client selected (:data:`CARLA_REF_ENV`, unset meaning
    typesafe_carla's ``ue5-dev``): :func:`client_server_mismatch` refuses a
    server of another family once one is reached, so the client's family is
    the server's. ``"0.10"`` for ``ue5-dev`` and ``0.*`` refs, ``"1"`` for
    any other (``1.0``, ``ue58-dev``, a newer branch or a commit).
    """
    ref = os.environ.get(CARLA_REF_ENV, "") or "ue5-dev"
    if ref == "ue5-dev" or ref.startswith("0."):
        return "0.10"
    return "1"


# What the vehicles and walkers of each family do with the controls they are
# given, measured on the Lincoln MKZ and walker.pedestrian.0001 in
# Town10HD_Opt (blackwell2, 2026-11-24; CARLA 0.10's first on 2026-10-04):
#
#   steer 0.1 / 0.2 / 0.3 / 0.5 at 4.8 m/s turns the front wheels
#     CARLA 0.10:  0.54 / 2.13 / 4.79 / 13.0 deg (bicycle-equivalent;
#                  56 * steer**2, whatever max_steer_angle the wheels report)
#     CARLA 1.0:   5.45 / 10.8 / 16.2 / 26.9 deg (53-54 * steer, linear;
#                  51.8-52.2 at 8.6 m/s)
#   WalkerControl speed 2 / 20 / 40 walks the walker at
#     CARLA 0.10:  0.098 / 0.977 / 1.95 m/s (speed / 20.5, at most 2 m/s)
#     CARLA 1.0:   2.0 / 20.0 / 0 m/s (the speed given)
#
# The throttle that pulls a standing MKZ away is the same on both (0.245).
_STEER_EXPONENT = {"0.10": 2.0, "1": 1.0}
_MAX_STEER_ANGLE_DEG = {"0.10": 56.0, "1": 53.0}
_WALKER_SPEED_SCALE = {"0.10": 20.5, "1": 1.0}
_WALKER_MAX_SPEED_MPS = {"0.10": 2.0, "1": 10.0}


def default_steer_exponent() -> float:
    """How the server's vehicles turn the wheels for a ``[-1, 1]`` steer:
    ``angle = max_steer_angle * |steer| ** exponent`` (``ControlConfig``'s
    default): 2 on CARLA 0.10, 1 (linear) on CARLA 1.0."""
    return _STEER_EXPONENT[server_family()]


def default_max_steer_angle_deg() -> float:
    """The wheel angle at full steer in the server's steering response
    (``ControlConfig``'s default): 56 degrees on CARLA 0.10, 53 on 1.0."""
    return _MAX_STEER_ANGLE_DEG[server_family()]


def walker_speed_scale() -> float:
    """The ``WalkerControl`` speed per m/s a walker walks on the server:
    20.5 on CARLA 0.10, 1 on CARLA 1.0."""
    return _WALKER_SPEED_SCALE[server_family()]


def walker_max_speed_mps() -> float:
    """The fastest a walker is walked on the server, in m/s: 2 on CARLA 0.10,
    which walks none faster; 10 on CARLA 1.0, a running person."""
    return _WALKER_MAX_SPEED_MPS[server_family()]


def client_server_mismatch(client: str, server: str) -> Optional[str]:
    """Say what is wrong when *client* and *server* speak different RPC.

    Both are ``carla::version()`` strings (``Client.get_client_version()``,
    ``Client.get_server_version()``). Returns ``None`` when they are of one
    family (:func:`version_family`), else a message naming the client to
    select with :data:`CARLA_REF_ENV`.
    """
    client_family = version_family(client)
    server_family = version_family(server)
    if client_family == server_family:
        return None
    selected = os.environ.get(CARLA_REF_ENV) or "ue5-dev (typesafe_carla's default)"
    wanted = _REF_FOR_FAMILY.get(server_family)
    advice = (
        f"set {CARLA_REF_ENV}={wanted} in the environment that runs the scenarios"
        if wanted is not None
        else f"set {CARLA_REF_ENV} to the CARLA branch the server was built from"
    )
    return (
        f"The CARLA server is version {server}, but the client "
        f"({CARLA_REF_ENV}={selected}) is built from CARLA {client}: their RPC "
        f"messages differ, so the run would fail on its first world settings. "
        f"The framework targets CARLA {CARLA_RELEASE} (scenario-setup installs it); "
        f"for this server, {advice}."
    )
