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

from ..carla_install_paths import home_dir

logger = logging.getLogger(__name__)

__all__ = [
    "PACKS_DIR_ENV",
    "MapPack",
    "MapPackError",
    "PackAccessError",
    "PackFiles",
    "packs_dir",
]

PACKS_DIR_ENV = "AUTOWARE_CARLA_SCENARIO_PACKS_DIR"
"""Overrides where packs are installed for the server (:func:`packs_dir`)."""

_NAME = r"[A-Za-z0-9_.-]+"
_REFERENCE = re.compile(rf"{_NAME}/{_NAME}(?:/{_NAME}(?:@[A-Za-z0-9_.+-]+)?)?")
_SHA256 = re.compile(r"[0-9a-f]{64}")

#: The formats a pack's Lanelet2 map is read from, in order of preference: the
#: Autoware map folder (Lanelet2, projector info, point cloud) over a bare
#: Lanelet2 file.
LANELET2_FORMATS = ("autoware", "lanelet2")

_INSTALL_HINT = (
    "Install carla-pack-registry to use map.pack: "
    "`pip install 'autoware-carla-scenario[pack]'` "
    "(or `uv add carla-pack-registry`)."
)


class MapPackError(RuntimeError):
    """A pack could not be found, fetched, read or mounted."""


class PackAccessError(MapPackError):
    """GitHub or the pack's storage refused access to a pack: no token, a
    token GitHub rejects (expired, revoked), a user without Read access to
    the pack repository, or storage credentials that expired.

    The message says which, where the token was taken from, and how to fix
    it; :attr:`code` is carla-pack's error code (``unauthenticated``,
    ``no_access``, ``credentials_unavailable``, ``storage_error``, and for a
    pack repository on Google Drive, read through rclone with the user's own
    Google account: ``rclone_unavailable``, ``storage_unauthenticated``,
    ``storage_no_access``).
    """

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


#: carla-pack's codes for a refusal by a pack repository's Google Drive
#: storage (read through rclone with the user's own Google account): no
#: rclone, no valid Google credentials, a drive not shared with the account.
STORAGE_REFUSALS = (
    "rclone_unavailable",
    "storage_unauthenticated",
    "storage_no_access",
)

#: Where carla-pack looks for a GitHub token, in order.
_TOKEN_SOURCES = (
    "CARLA_PACK_TOKEN, GH_TOKEN, GITHUB_TOKEN, the token `carla-pack login` "
    "saved, then `gh auth token` (the GitHub CLI)"
)

#: How to replace a token GitHub rejects, by where carla-pack took it from
#: (carla_pack_registry.Client.token_source).
_REFRESH = {
    "gh auth token": "run `gh auth login` again (or `gh auth refresh`)",
    "carla-pack login": "run `carla-pack login` again",
    "CARLA_PACK_TOKEN": "set CARLA_PACK_TOKEN to a valid token",
    "GH_TOKEN": "set GH_TOKEN to a valid token (or unset it to use `gh auth login`)",
    "GITHUB_TOKEN": (
        "set GITHUB_TOKEN to a valid token (or unset it to use `gh auth login`); "
        "a GitHub Actions workflow's own GITHUB_TOKEN cannot read another "
        "repository: pass a token that can in CARLA_PACK_TOKEN"
    ),
}


def packs_dir(configured: Optional[str] = None) -> Path:
    """Where packs are installed for the CARLA server to mount.

    *configured* (``server.packs_dir``) first, then the
    ``AUTOWARE_CARLA_SCENARIO_PACKS_DIR`` environment variable, else
    ``<home>/packs`` (``~/.autoware_carla_scenario/packs``,
    :func:`~autoware_carla_scenario.carla_install_paths.home_dir`).  The server
    reads the pack at the path it is installed at, so a server on another
    machine needs a directory both see at the same path; a relative one is
    taken from this process's working directory and made absolute here, as
    the server, with a working directory of its own, could not read it
    (#2198).  Symlinks are kept, not resolved (``Path.absolute()``, not
    ``resolve()``): the server, or a container it runs in, may see the path
    as written -- a bind mount of ``~/.autoware_carla_scenario`` -- but not
    the directory a symlink on this machine points at.

    Raises:
        ValueError: *configured* is blank: ``server.packs_dir`` is a path, or
            null for the default (#2143, #2198).
    """
    if configured is not None:
        if not configured.strip():
            raise ValueError(f"server.packs_dir must be a path, got {configured!r}")
        return Path(configured.strip()).expanduser().absolute()
    from_env = os.environ.get(PACKS_DIR_ENV, "").strip()
    if from_env:
        return Path(from_env).expanduser().absolute()
    return (home_dir() / "packs").absolute()


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


def _token_source(client: Any) -> Optional[str]:
    """Where *client* took its GitHub token from, or ``None`` for none."""
    source = getattr(client, "token_source", None)
    return source if isinstance(source, str) and source else None


def _login(client: Any) -> Optional[str]:
    """The GitHub user *client*'s token belongs to, when GitHub says so."""
    try:
        user = client.whoami()
    except Exception:  # noqa: BLE001 -- only to name the user in a message
        return None
    login = user.get("login") if isinstance(user, dict) else None
    return str(login) if login else None


def _repository(reference: str) -> str:
    return "/".join(reference.split("/")[:2])


def _failure(
    action: str, exc: BaseException, client: Any, reference: str
) -> MapPackError:
    """A :class:`MapPackError` for carla-pack's *exc* -- a
    :class:`PackAccessError` saying what to do when access was refused."""
    code = str(exc.args[1]) if len(exc.args) > 1 else ""
    message = str(exc.args[0]) if exc.args else str(exc)
    repository = _repository(reference)
    source = _token_source(client)
    if code == "unauthenticated":
        if "rate limit" in message:
            fix = (
                "GitHub's limit on requests without a token was reached. Give "
                f"carla-pack a GitHub token (it looks in {_TOKEN_SOURCES}): run "
                "`gh auth login`, or set CARLA_PACK_TOKEN."
            )
        elif source is None:
            fix = (
                f"No GitHub token was found, and {repository} is not public. "
                f"carla-pack looks in {_TOKEN_SOURCES}: run `gh auth login`, "
                "or set CARLA_PACK_TOKEN to a token that can read "
                f"{repository}."
            )
        else:
            refresh = _REFRESH.get(source, "replace it with a valid token")
            fix = (
                f"GitHub rejected the token from {source}: it has expired or "
                f"been revoked. To fix it, {refresh}; `carla-pack whoami` "
                "checks it."
            )
        return PackAccessError(
            f"{action}: GitHub authentication failed. {fix} ({message})", code
        )
    # GitHub answers a private repository the token may not read with 404,
    # which carla-pack reports as no_access; not_found is a pack, version or
    # map missing from a manifest it read -- or, from a search, one missing
    # from the repositories it could read, which it names as not readable.
    if code == "no_access" or (code == "not_found" and "not readable" in message):
        login = _login(client) if source is not None else None
        who = (
            f"the GitHub user {login} (token from {source})"
            if login
            else f"the token from {source}"
            if source
            else "an anonymous request (no GitHub token found)"
        )
        fix = (
            f"{repository} does not exist, or {who} cannot read it (GitHub "
            "answers a private repository it may not read with 'not found'). "
            f"Ask the owner of {repository} for Read access"
            + (
                ", or use a token of a user who has it (CARLA_PACK_TOKEN)"
                if source
                else f"; carla-pack looks for a token in {_TOKEN_SOURCES}: "
                "run `gh auth login`"
            )
            + f". `carla-pack ls {repository}` shows what can be read."
        )
        return PackAccessError(f"{action}: access denied. {fix} ({message})", code)
    if code == "credentials_unavailable":
        return PackAccessError(
            f"{action}: the storage credentials of {repository} could not be "
            "read. Your GitHub access is fine; the pack repository's credential "
            f"workflow has to run (ask the owner of {repository}). ({message})",
            code,
        )
    # The pack's storage refused the user after GitHub let them in: a pack
    # repository on Google Drive, read through rclone with their own Google
    # account. carla-pack's message names the cause and the fix.
    if code in STORAGE_REFUSALS:
        return PackAccessError(
            f"{action}: the pack's storage refused access. Your GitHub access "
            f"to {repository} is fine. {message}",
            code,
        )
    if code == "storage_error" and "refused" in message:
        return PackAccessError(
            f"{action}: the pack's storage refused the download: its "
            f"credentials were rotated or revoked. Retry, or ask the owner of "
            f"{repository}. ({message})",
            code,
        )
    return MapPackError(f"{action}: {message}")


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
        # GitHub allows neither "." nor ".." as an owner, repository or
        # pack name, and a path built of one leaves the packs directory
        # (#2233).
        parts = self.reference.replace("@", "/").split("/")
        if not _REFERENCE.fullmatch(self.reference) or any(
            part in (".", "..") for part in parts
        ):
            raise ValueError(
                f"map.pack must be OWNER/REPO/PACK[@VERSION] or OWNER/REPO, "
                f"got {self.reference!r}"
            )
        if self.sha256 is not None and not _SHA256.fullmatch(self.sha256):
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
            PackAccessError: (a MapPackError) GitHub or the pack's storage
                refused access: no token, an expired or revoked one, no Read
                access to the pack repository, expired storage credentials.
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
            raise _failure(
                f"Could not find the pack {self.reference}", exc, client, self.reference
            ) from exc
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
            PackAccessError: (a MapPackError) GitHub or the pack's storage
                refused access: no token, an expired or revoked one, no Read
                access to the pack repository, expired storage credentials.
        """
        registry = _registry()
        client = client if client is not None else registry.Client()
        release = self.release(map_name, client)
        reference = str(release.get("ref") or self.reference)
        try:
            formats: dict[str, dict[str, Any]] = client.formats(release)
        except registry.CarlaPackError as exc:
            raise _failure(
                f"Could not fetch the pack {reference}", exc, client, reference
            ) from exc
        cooked = [str(m) for m in release.get("maps") or []]
        name = self._map_of(map_name, reference, cooked, formats)
        of_map = formats.get(name, {})

        def data(fmt: str) -> Path:
            try:
                return Path(client.map_data(release, name, fmt))
            except registry.CarlaPackError as exc:
                raise _failure(
                    f"Could not read {fmt} data of {name} from {reference}",
                    exc,
                    client,
                    reference,
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
            PackAccessError: (a MapPackError) GitHub or the pack's storage
                refused access: no token, an expired or revoked one, no Read
                access to the pack repository, expired storage credentials.
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
        client = registry.Client()
        try:
            if self.is_repository:
                pack = registry.ensure_map(
                    carla_client,
                    map_name,
                    os.fspath(target),
                    repos=[self.reference],
                    sha256=self.sha256,
                    client=client,
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
                    client=client,
                )
        except registry.CarlaPackError as exc:
            raise _failure(
                f"Could not deploy the pack {self.reference} to {host}:{port}",
                exc,
                client,
                self.reference,
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
