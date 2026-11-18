"""CARLA maps from content packs shared through carla-pack-registry.

A CARLA map that the server does not ship -- a town cooked as a content pack
-- can be named by the pack that carries it (``map.pack``)::

    map.name=NishishinjukuMap
    map.pack=tier4/carla-packs/NishishinjukuPack@1.2.0
    # or: search a pack repository for the pack providing map.name
    map.pack=tier4/carla-packs

Two things then follow, both through `carla-pack-registry
<https://github.com/autowarefoundation/carla_content_pack_registry>`_ (the
``pack`` extra), whose GitHub token is the access check:

* **Loading the map's files** (:meth:`MapPack.files`, before CARLA starts):
  the pack is downloaded into carla-pack's cache, verified, and its map data
  read by format -- the Autoware map folder (``autoware``) or a bare Lanelet2
  map (``lanelet2``) becomes the scenario's Lanelet2 map, read the way
  ``map.directory`` reads one, and the OpenDRIVE CARLA cooks the map from
  (``opendrive``) the road network the run reads.
* **Running on it** (:meth:`MapPack.deploy`, once the server is up): the pack
  is installed into :func:`packs_dir` and mounted on the server, so
  ``load_world(map.name)`` finds it.  A pack already mounted is left as it is,
  and nothing is downloaded twice: the install is made from the same cached,
  verified tarball.

Nothing here is imported unless a config names a pack, and carla-pack-registry
and the CARLA client are only imported when they are used.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "PACKS_DIR_ENV",
    "MapPack",
    "MapPackError",
    "PackFiles",
    "packs_dir",
]

PACKS_DIR_ENV = "AUTOWARE_CARLA_SCENARIO_PACKS_DIR"
"""Overrides where packs are installed for the server (:func:`packs_dir`)."""

_HOME_ENV = "AUTOWARE_CARLA_SCENARIO_HOME"
"""The framework's home directory (``carla_install_paths.HOME_ENV``)."""

_NAME = r"[A-Za-z0-9_.-]+"
_REFERENCE = re.compile(rf"^{_NAME}/{_NAME}(?:/{_NAME}(?:@[A-Za-z0-9_.+-]+)?)?$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

#: The formats a pack's Lanelet2 map is read from, in order of preference: the
#: Autoware map folder (Lanelet2, projector info, point cloud) over a bare
#: Lanelet2 file.
LANELET2_FORMATS = ("autoware", "lanelet2")

_INSTALL_HINT = (
    "Install carla-pack-registry to use map.pack: "
    "`pip install 'autoware-carla-scenario[pack]'` "
    "(or `uv add carla-pack-registry`)."
)
_ACCESS_HINT = (
    "carla-pack reads the pack repository with your GitHub token: run "
    "`gh auth login` (or `carla-pack login`), or set CARLA_PACK_TOKEN to a "
    "token that can read it; `carla-pack whoami` and `carla-pack ls "
    "OWNER/REPO` show what it can see."
)


class MapPackError(RuntimeError):
    """A pack could not be found, fetched, read or mounted."""


def packs_dir(configured: Optional[str] = None) -> Path:
    """Where packs are installed for the CARLA server to mount.

    *configured* (``server.packs_dir``) first, then the
    ``AUTOWARE_CARLA_SCENARIO_PACKS_DIR`` environment variable, else
    ``<home>/packs`` (``~/.autoware_carla_scenario/packs``).  The server reads
    the pack at the path it is installed at, so a server on another machine
    needs a directory both see at the same path.
    """
    if configured is not None and configured.strip():
        return Path(configured.strip()).expanduser()
    from_env = os.environ.get(PACKS_DIR_ENV, "").strip()
    if from_env:
        return Path(from_env).expanduser()
    home = os.environ.get(_HOME_ENV, "").strip() or "~/.autoware_carla_scenario"
    return Path(home).expanduser() / "packs"


def _short(name: str) -> str:
    return name.rsplit("/", 1)[-1]


def _registry() -> Any:
    """The carla_pack_registry module, or a :class:`MapPackError` saying how
    to install it."""
    try:
        import carla_pack_registry  # noqa: PLC0415 -- optional (the `pack` extra)
    except ImportError as exc:
        raise MapPackError(_INSTALL_HINT) from exc
    return carla_pack_registry


def _failure(action: str, exc: BaseException) -> MapPackError:
    """A :class:`MapPackError` for carla-pack's *exc*, with the access hint
    when it reads as GitHub refusing (or not finding) the repository."""
    code = exc.args[1] if len(exc.args) > 1 else ""
    message = exc.args[0] if exc.args else str(exc)
    hint = ""
    if code in ("no_access", "unauthenticated", "not_found"):
        hint = f" {_ACCESS_HINT}"
    return MapPackError(f"{action}: {message}{hint}")


@dataclass(frozen=True)
class PackFiles:
    """The map files a pack carries, unpacked in carla-pack's cache.

    Attributes:
        map_name: The CARLA map the files are of (its short name).
        directory: The Lanelet2 map: the Autoware map folder, or the ``.osm``
            itself -- what ``map.directory`` takes.  ``None`` when the pack has
            none for the map.
        xodr_path: The OpenDRIVE CARLA cooks the map from, or ``None``.
        reference: The pack, ``OWNER/REPO/PACK@VERSION``.
        sha256: The pack's artifact, to deploy exactly the pack these came from.
    """

    map_name: str
    directory: Optional[Path]
    xodr_path: Optional[Path]
    reference: str
    sha256: str


@dataclass(frozen=True)
class MapPack:
    """A CARLA map's content pack (``map.pack``).

    Attributes:
        reference: ``OWNER/REPO/PACK[@VERSION]`` -- one pack, its newest
            version without one -- or ``OWNER/REPO``, a pack repository
            searched for the pack providing the map.
        sha256: Pins the pack to one artifact (``map.pack_sha256``), as a lock
            file would.
        packs_dir: Where the pack is installed for the server; ``None`` is
            :func:`packs_dir`'s default.
    """

    reference: str
    sha256: Optional[str] = None
    packs_dir: Optional[Path] = None

    def __post_init__(self) -> None:
        if not _REFERENCE.match(self.reference):
            raise ValueError(
                f"map.pack must be OWNER/REPO/PACK[@VERSION] or OWNER/REPO, "
                f"got {self.reference!r}"
            )
        if self.sha256 is not None and not _SHA256.match(self.sha256):
            raise ValueError(
                "map.pack_sha256 must be a SHA-256 in lowercase hex (64 "
                f"characters), got {self.sha256!r}"
            )

    @property
    def is_repository(self) -> bool:
        """Whether :attr:`reference` names a pack repository to search
        rather than one pack."""
        return self.reference.count("/") == 1

    def release(self, map_name: str, client: Any = None) -> dict[str, Any]:
        """The release :attr:`reference` names, for *map_name*.

        Raises:
            MapPackError: carla-pack is not installed, the repository cannot
                be read, or nothing in it provides the map.
        """
        registry = _registry()
        client = client if client is not None else registry.Client()
        try:
            if self.is_repository:
                if not map_name:
                    raise MapPackError(
                        f"map.pack={self.reference} names a pack repository, "
                        "which is searched by map.name: set map.name too"
                    )
                release: dict[str, Any] = client.find_map(
                    map_name, [self.reference], sha256=self.sha256
                )
            else:
                release = client.resolve(self.reference, sha256=self.sha256)
        except registry.CarlaPackError as exc:
            raise _failure(f"Could not find the pack {self.reference}", exc) from exc
        return release

    def files(self, map_name: str = "", client: Any = None) -> PackFiles:
        """Fetch the pack and return the map files it carries for *map_name*.

        The pack is downloaded once into carla-pack's cache (named by its
        SHA-256, verified against the repository manifest and the pack's own
        ``carla-pack.json``) and unpacked there; :meth:`deploy` installs it
        from the same cached tarball.  With *map_name* empty the pack must
        carry a single map, which is the one read.

        Raises:
            MapPackError: The pack cannot be found or fetched, or does not
                carry *map_name*.
        """
        registry = _registry()
        client = client if client is not None else registry.Client()
        release = self.release(map_name, client)
        reference = str(release.get("ref") or self.reference)
        try:
            formats: dict[str, dict[str, Any]] = client.formats(release)
        except registry.CarlaPackError as exc:
            raise _failure(f"Could not fetch the pack {reference}", exc) from exc
        cooked = [str(m) for m in release.get("maps") or []]
        name = self._map_of(map_name, reference, cooked, formats)
        of_map = formats.get(name, {})

        def data(fmt: str) -> Path:
            try:
                return Path(client.map_data(release, name, fmt))
            except registry.CarlaPackError as exc:
                raise _failure(
                    f"Could not read {fmt} data of {name} from {reference}", exc
                ) from exc

        directory: Optional[Path] = None
        for fmt in LANELET2_FORMATS:
            if fmt in of_map:
                directory = data(fmt)
                break
        if directory is None:
            logger.warning(
                "The pack %s carries no Lanelet2 map for %s (formats: %s); set "
                "map.directory, map.source or map.lanelet2_path for one",
                reference,
                name,
                ", ".join(sorted(of_map)) or "none",
            )
        xodr = data("opendrive") if "opendrive" in of_map else None
        return PackFiles(
            map_name=name,
            directory=directory,
            xodr_path=xodr,
            reference=reference,
            sha256=str(release.get("sha256") or self.sha256 or ""),
        )

    def _map_of(
        self,
        map_name: str,
        reference: str,
        cooked: list[str],
        formats: dict[str, dict[str, Any]],
    ) -> str:
        """The key of *map_name* among the pack's maps (by short name), or its
        only map when *map_name* is empty."""
        known = sorted({_short(m) for m in cooked} | set(formats))
        if map_name:
            wanted = _short(map_name)
            for candidate in known:
                if _short(candidate) == wanted:
                    return candidate
            raise MapPackError(
                f"The pack {reference} does not carry the map {wanted} "
                f"(maps: {', '.join(known) or 'none'})"
            )
        if len(cooked) == 1:
            return _short(cooked[0])
        raise MapPackError(
            f"The pack {reference} carries {len(cooked)} maps "
            f"({', '.join(known) or 'none'}): set map.name to the one to run"
        )

    def deploy(
        self, host: str, port: int, map_name: str, timeout: float = 60.0
    ) -> None:
        """Install the pack and mount it on the server at *host*:*port*, so
        that it can load *map_name*.

        For a pack repository (:attr:`is_repository`) nothing is done when
        the server already lists *map_name*; otherwise the newest readable
        pack providing it is installed and mounted.  A named pack is
        installed and mounted unless it is already; another version of it
        mounted is refused (unmount it, or restart the server).

        Raises:
            MapPackError: carla-pack or the CARLA client is not installed,
                the pack cannot be fetched or installed, or the server does
                not mount it.
        """
        registry = _registry()
        try:
            import typesafe_carla.carla as carla  # noqa: PLC0415 -- heavy
        except ImportError as exc:  # pragma: no cover -- depends on the platform
            raise MapPackError(
                "The CARLA Python client (typesafe-carla) is not installed, so "
                f"the pack {self.reference} cannot be mounted"
            ) from exc
        target = self.packs_dir if self.packs_dir is not None else packs_dir()
        target.mkdir(parents=True, exist_ok=True)
        carla_client = carla.Client(host, port)
        carla_client.set_timeout(timeout)
        try:
            if self.is_repository:
                pack = registry.ensure_map(
                    carla_client,
                    map_name,
                    os.fspath(target),
                    repos=[self.reference],
                    sha256=self.sha256,
                )
                if pack is None:
                    logger.info(
                        "The CARLA server at %s:%d already has %s: no pack "
                        "mounted from %s",
                        host,
                        port,
                        map_name,
                        self.reference,
                    )
                    return
            else:
                pack = registry.deploy_pack(
                    self.reference,
                    os.fspath(target),
                    carla_client=carla_client,
                    sha256=self.sha256,
                )
        except registry.CarlaPackError as exc:
            raise _failure(
                f"Could not deploy the pack {self.reference} to {host}:{port}", exc
            ) from exc
        if not pack.get("mounted"):
            raise MapPackError(
                f"The CARLA server at {host}:{port} did not mount the pack "
                f"{pack.get('name')} at {pack.get('path')}: it must read that "
                "path as this machine does (server.packs_dir)"
            )
        logger.info(
            "Mounted the pack %s %s (%s) on %s:%d from %s",
            pack.get("name"),
            pack.get("version"),
            ", ".join(str(m) for m in pack.get("maps") or []),
            host,
            port,
            pack.get("path"),
        )
