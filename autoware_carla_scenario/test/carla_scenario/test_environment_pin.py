"""A distribution's source and its pin (``environment_pin``, #825), against
what ``environment`` computed before they were split out, and that the
environment file's records keep their shape."""

from __future__ import annotations

import itertools
import json
import shutil
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote, urlparse

import pytest

from autoware_carla_scenario import environment, environment_pin
from autoware_carla_scenario.environment import (
    _package_pin,
    _pin_source,
    _source,
    _source_record,
    capture,
)
from autoware_carla_scenario.environment_pin import (
    HASH_PREFERENCE,
    PinSource,
    fragment,
    normalize,
    pin,
)

# -- What environment computed before #825, verbatim -------------------------


def _master_normalize(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def _master_git(path: Path, *args: str) -> Optional[str]:
    try:
        done = subprocess.run(
            ["git", "-C", str(path), *args],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip()


_MASTER_HASH_PREFERENCE = ("sha512", "sha384", "sha256", "sha224", "sha1", "md5")


def _master_archive_hashes(info: dict[str, Any]) -> dict[str, str]:
    hashes = {
        str(k).lower(): str(v)
        for k, v in (info.get("hashes") or {}).items()
        if isinstance(v, str) and v
    }
    legacy = info.get("hash")
    if isinstance(legacy, str) and "=" in legacy:
        algorithm, _, value = legacy.partition("=")
        hashes.setdefault(algorithm.lower(), value)
    return hashes


def _master_source(dist: Any) -> dict[str, Any]:
    try:
        raw = dist.read_text("direct_url.json")
    except Exception:  # noqa: BLE001
        raw = None
    if not raw:
        return {"kind": "index"}
    try:
        direct = json.loads(raw)
    except ValueError:
        return {"kind": "index"}
    if not isinstance(direct, dict):
        return {"kind": "index"}
    url = str(direct.get("url", ""))
    subdirectory = direct.get("subdirectory") or None
    if "vcs_info" in direct:
        vcs = direct["vcs_info"] or {}
        source: dict[str, Any] = {
            "kind": "vcs",
            "url": url,
            "vcs": vcs.get("vcs", "git"),
            "commit": vcs.get("commit_id", ""),
            "requested": vcs.get("requested_revision"),
        }
        if subdirectory:
            source["subdirectory"] = subdirectory
        return source
    if "archive_info" in direct:
        source = {"kind": "archive", "url": url}
        hashes = _master_archive_hashes(direct["archive_info"] or {})
        if hashes:
            source["hashes"] = hashes
        if subdirectory:
            source["subdirectory"] = subdirectory
        return source
    path = Path(unquote(urlparse(url).path)) if url.startswith("file:") else None
    if path is not None and subdirectory:
        path = path / subdirectory
    out: dict[str, Any] = {
        "kind": "editable"
        if (direct.get("dir_info") or {}).get("editable")
        else "local",
        "path": str(path) if path else url,
    }
    if path is None and subdirectory:
        out["subdirectory"] = subdirectory
    if path is not None and path.is_dir():
        commit = _master_git(path, "rev-parse", "HEAD")
        if commit:
            out["commit"] = commit
            status = _master_git(path, "status", "--porcelain", "--untracked-files=no")
            out["dirty"] = bool(status)
    return out


def _master_fragment(source: dict[str, Any]) -> str:
    parts: list[str] = []
    hashes = source.get("hashes") or {}
    for algorithm in _MASTER_HASH_PREFERENCE:
        if hashes.get(algorithm):
            parts.append(f"{algorithm}={hashes[algorithm]}")
            break
    if source.get("subdirectory"):
        parts.append(f"subdirectory={source['subdirectory']}")
    return "#" + "&".join(parts) if parts else ""


def _master_pin(name: str, version: str, source: dict[str, Any]) -> str:
    kind = source.get("kind")
    if kind == "vcs":
        revision = source.get("commit") or source.get("requested")
        if revision:
            pin = (
                f"{name} @ {source['vcs']}+{source['url']}@{revision}"
                f"{_master_fragment(source)}"
            )
            if not source.get("commit"):
                pin += f"  # requested {revision}, no commit recorded: may move"
            return pin
    if kind == "archive":
        return f"{name} @ {source['url']}{_master_fragment(source)}"
    if kind in ("editable", "local"):
        state = ""
        if source.get("commit"):
            state = f", git {source['commit']}" + (
                " with uncommitted changes" if source.get("dirty") else ""
            )
        return f"{name}=={version}  # {kind} from {source.get('path')}{state}"
    return f"{name}=={version}"


# -- The comparisons ----------------------------------------------------------


class _Dist:
    """A distribution whose ``direct_url.json`` is *raw* (``None``: none)."""

    def __init__(self, raw: Optional[str]) -> None:
        self._raw = raw

    def read_text(self, name: str) -> Optional[str]:
        return self._raw if name == "direct_url.json" else None


class _Broken:
    def read_text(self, name: str) -> str:
        raise RuntimeError(name)


def test_environment_re_exports_them() -> None:
    assert environment._normalize is normalize
    assert environment._fragment is fragment
    assert environment._pin is pin
    assert environment.PinSource is PinSource
    assert environment._HASH_PREFERENCE == HASH_PREFERENCE == _MASTER_HASH_PREFERENCE
    assert environment_pin.__all__ == [
        "HASH_PREFERENCE",
        "PinSource",
        "fragment",
        "normalize",
        "pin",
    ]


@pytest.mark.parametrize(
    "name",
    ["", "a", "A_b.C-d", "Typesafe_Carla", "zope.interface", "x__y..z", "Ünïcode_X"],
)
def test_normalize_is_masters(name: str) -> None:
    assert normalize(name) == _master_normalize(name)


_COMMIT = "9e15ebae" + "0" * 32
_HASHES: list[Any] = [
    None,
    {},
    {"sha256": "ab" * 32},
    {"sha256": "ab" * 32, "md5": "cd" * 16, "SHA512": "ef" * 64},
    {"md5": "", "sha1": "01" * 20},
    {"sha256": 3},
]
_SUBDIRECTORIES: list[Any] = [None, "", "pkg", "a/b"]


def _direct_urls(local: Path) -> list[Any]:
    found: list[Any] = [
        None,
        "",
        "not json",
        "[1, 2]",
        "3",
        "{}",
        {"url": "https://x/a.whl"},
    ]
    for subdirectory in _SUBDIRECTORIES:
        extra = {} if subdirectory is None else {"subdirectory": subdirectory}
        for commit, requested in itertools.product(
            [None, "", _COMMIT, "_missing_"], [None, "", "master", "_missing_"]
        ):
            vcs: dict[str, Any] = {"vcs": "git"}
            if commit != "_missing_":
                vcs["commit_id"] = commit
            if requested != "_missing_":
                vcs["requested_revision"] = requested
            found.append({"url": "https://x/a.git", "vcs_info": vcs, **extra})
        found.append({"url": "https://x/a.git", "vcs_info": None, **extra})
        found.append({"url": "https://x/a", "vcs_info": {"vcs": "hg"}, **extra})
        for hashes in _HASHES:
            info: dict[str, Any] = {} if hashes is None else {"hashes": hashes}
            found.append({"url": "https://x/a.tgz", "archive_info": info, **extra})
        found.append(
            {
                "url": "https://x/a.tgz",
                "archive_info": {"hash": "SHA256=" + "12" * 32},
                **extra,
            }
        )
        found.append({"url": "https://x/a.tgz", "archive_info": None, **extra})
        for editable in [None, False, True]:
            info = {} if editable is None else {"editable": editable}
            for url in [
                local.as_uri(),
                (local / "missing").as_uri(),
                "https://x/dir",
                "",
            ]:
                found.append({"url": url, "dir_info": info, **extra})
        found.append({"url": local.as_uri(), "dir_info": None, **extra})
    return found


def _raw(direct_url: Any) -> Optional[str]:
    return (
        direct_url
        if direct_url is None or isinstance(direct_url, str)
        else (json.dumps(direct_url))
    )


@pytest.fixture(params=["plain", "clean", "dirty"])
def local(request: pytest.FixtureRequest, tmp_path: Path) -> Path:
    """A local project directory: not a checkout, or a clean or dirty one."""
    for directory in (tmp_path, tmp_path / "pkg", tmp_path / "a" / "b"):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "f.txt").write_text("one")
    if request.param != "plain":
        if shutil.which("git") is None:
            pytest.skip("git is not installed")

        def git(*args: str) -> None:
            subprocess.run(
                ["git", "-C", str(tmp_path), *args], check=True, capture_output=True
            )

        git("init", "-q")
        git("add", ".")
        git(
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.com",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-q",
            "-m",
            "init",
        )
        if request.param == "dirty":
            (tmp_path / "pkg" / "f.txt").write_text("two")
    return tmp_path


def test_a_source_s_record_and_pin_are_masters(local: Path) -> None:
    for direct_url in _direct_urls(local):
        dist = _Dist(_raw(direct_url))
        master = _master_source(dist)
        source = _source(dist)  # type: ignore[arg-type]
        record = _source_record(source)
        # The same keys in the same order: the environment file is unchanged.
        assert json.dumps(record) == json.dumps(master), direct_url
        expected = _master_pin("a", "1.0", master)
        assert pin("a", "1.0", source) == expected, direct_url
        assert _package_pin({"name": "a", "version": "1.0", "source": record}) == (
            expected
        ), direct_url
        assert fragment(_pin_source(record)) == _master_fragment(master), direct_url


def test_an_unreadable_source_is_an_index() -> None:
    assert _source(_Broken()) == PinSource(kind="index")  # type: ignore[arg-type]
    assert _source_record(_source(_Broken())) == _master_source(_Broken())  # type: ignore[arg-type]


_RECORDS: list[dict[str, Any]] = [
    {},
    {"kind": "index"},
    {"kind": "other", "url": "u"},
    {"kind": "vcs", "vcs": "git", "url": "https://x/a.git", "commit": "abc"},
    {"kind": "vcs", "vcs": "git", "url": "u", "commit": "", "requested": "v1"},
    {"kind": "vcs", "vcs": "git", "url": "u", "commit": "", "requested": None},
    {"kind": "vcs", "vcs": "git", "url": "u", "commit": "c", "subdirectory": "s"},
    {"kind": "archive", "url": "https://x/a.whl"},
    {"kind": "archive", "url": "u", "hashes": {"md5": "m", "sha1": "s"}},
    {"kind": "archive", "url": "u", "hashes": {}, "subdirectory": "pkg"},
    {"kind": "editable", "path": "/src/a", "commit": "abc", "dirty": True},
    {"kind": "editable", "path": "/src/a", "commit": "abc", "dirty": False},
    {"kind": "local", "path": "/src/a"},
    {"kind": "local", "path": "https://x/d", "subdirectory": "pkg"},
    {"kind": "local"},
]


@pytest.mark.parametrize("record", _RECORDS)
@pytest.mark.parametrize(("name", "version"), [("a", "1.0"), ("b-c", "0.0.0")])
def test_a_recorded_source_pins_as_on_master(
    record: dict[str, Any], name: str, version: str
) -> None:
    assert _package_pin({"name": name, "version": version, "source": record}) == (
        _master_pin(name, version, record)
    )


def test_this_environment_is_recorded_and_pinned_as_on_master() -> None:
    env = capture()
    masters: dict[str, dict[str, Any]] = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if not name:
            continue
        key = _master_normalize(name)
        if key not in masters:
            masters[key] = _master_source(dist)
    assert len(env.packages) == len(masters)
    for package in env.packages:
        assert json.dumps(package["source"]) == json.dumps(masters[package["name"]])
    assert env.pins == [
        _master_pin(p["name"], p["version"], masters[p["name"]]) for p in env.packages
    ]
