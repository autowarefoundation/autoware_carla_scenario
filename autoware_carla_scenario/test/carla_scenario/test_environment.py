"""What a run ran with: the environment and the lock beside its result."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from packaging.requirements import Requirement

from autoware_carla_scenario import environment
from autoware_carla_scenario.environment import (
    PinSource,
    RunEnvironment,
    _pin,
    _source,
    capture,
    write,
)


def test_the_framework_and_its_dependencies_are_found() -> None:
    env = capture()
    framework = env.by_name["autoware-carla-scenario"]
    assert framework["version"]
    assert "openodd-py" in env.by_name and env.summary()["packages"]["openodd-py"]
    assert len(env.id) == 64 and env.id == capture().id


def test_a_lock_pins_every_distribution(tmp_path: Path) -> None:
    summary = write(tmp_path, "Demo")
    assert summary is not None
    lock = (tmp_path / "Demo_requirements.lock").read_text().splitlines()
    assert lock[1] == f"# environment id: {summary['id']}"
    pins = [line for line in lock if line and not line.startswith("#")]
    assert len(pins) == len(capture().packages)
    # From an index (``openodd-py==``) or a git source (``openodd-py @ git+``).
    assert any(line.startswith(("openodd-py==", "openodd-py @ ")) for line in pins)
    doc = json.loads((tmp_path / "Demo_environment.json").read_text())
    assert doc["schema"] == "autoware_carla_scenario.environment/1"
    assert doc["id"] == summary["id"]
    assert {d["name"] for d in doc["distributions"]} >= {
        "autoware-carla-scenario",
        "openodd-py",
    }


def test_how_each_source_is_pinned() -> None:
    assert _pin("a", "1.0", PinSource(kind="index")) == "a==1.0"
    assert _pin(
        "a",
        "1.0",
        PinSource(kind="vcs", vcs="git", url="https://x/a.git", commit="abc"),
    ) == ("a @ git+https://x/a.git@abc")
    assert (
        _pin("a", "1.0", PinSource(kind="archive", url="https://x/a.whl"))
        == "a @ https://x/a.whl"
    )
    assert _pin(
        "a",
        "1.0",
        PinSource(kind="editable", path="/src/a", commit="abc", dirty=True),
    ) == ("a==1.0  # editable from /src/a, git abc with uncommitted changes")


class _Scenario:
    pass


def test_the_scenario_s_package_is_named(monkeypatch: Any) -> None:
    env = RunEnvironment(
        [
            {
                "name": "autoware-carla-scenario",
                "version": "1.0",
                "source": {"kind": "index"},
            },
            {
                "name": "town10-scenarios",
                "version": "0.3.0",
                "source": {"kind": "index"},
            },
        ]
    )
    _Scenario.__module__ = "town10_scenarios.cut_in"
    monkeypatch.setattr(
        environment.metadata,
        "packages_distributions",
        lambda: {"town10_scenarios": ["town10_scenarios"]},
    )
    assert env.summary(_Scenario())["scenario"]["name"] == "town10-scenarios"
    assert "# scenario: town10-scenarios==0.3.0" in env.lock(_Scenario())


def test_a_failure_is_not_raised(tmp_path: Path) -> None:
    blocked = tmp_path / "file"
    blocked.write_text("not a directory")
    assert write(blocked / "out", "Demo") is None


class _Dist:
    """A distribution that only has a ``direct_url.json``."""

    def __init__(self, direct_url: dict[str, Any]) -> None:
        self._raw = json.dumps(direct_url)

    def read_text(self, name: str) -> str | None:
        return self._raw if name == "direct_url.json" else None


REPOSITORY = "https://github.com/autowarefoundation/autoware_carla_scenario.git"
COMMIT = "9e15ebae" + "0" * 32


def _pin_of(direct_url: dict[str, Any]) -> tuple[PinSource, str]:
    source = _source(_Dist(direct_url))  # type: ignore[arg-type]
    pin = _pin("autoware-carla-scenario", "2.63.0", source)
    # Whatever the lock writes has to be a requirement pip and uv accept (a
    # requirements file drops a comment after whitespace).
    assert Requirement(pin.split("  #")[0]).name == "autoware-carla-scenario"
    return source, pin


def test_a_git_pin_keeps_its_subdirectory() -> None:
    # Issue #682: the framework lives in a subdirectory of its repository, and
    # a lock line without it names the workspace root, which has no project.
    source, pin = _pin_of(
        {
            "url": REPOSITORY,
            "vcs_info": {
                "vcs": "git",
                "commit_id": COMMIT,
                "requested_revision": "master",
            },
            "subdirectory": "autoware_carla_scenario",
        }
    )
    assert source.subdirectory == "autoware_carla_scenario"
    assert source.requested == "master"
    assert pin == (
        f"autoware-carla-scenario @ git+{REPOSITORY}@{COMMIT}"
        "#subdirectory=autoware_carla_scenario"
    )


def test_a_git_pin_without_a_commit_falls_back_to_the_requested_revision() -> None:
    _, pin = _pin_of(
        {"url": REPOSITORY, "vcs_info": {"vcs": "git", "requested_revision": "v1"}}
    )
    # A branch or tag can move, so the lock says the pin is not a commit.
    assert pin == (
        f"autoware-carla-scenario @ git+{REPOSITORY}@v1"
        "  # requested v1, no commit recorded: may move"
    )


@pytest.mark.parametrize(
    ("archive_info", "fragment"),
    [
        ({}, "#subdirectory=pkg"),
        (
            {"hashes": {"sha256": "ab" * 32, "md5": "cd" * 16}},
            f"#sha256={'ab' * 32}&subdirectory=pkg",
        ),
        ({"hash": "sha256=" + "ef" * 32}, f"#sha256={'ef' * 32}&subdirectory=pkg"),
    ],
    ids=["no-hash", "hashes", "legacy-hash"],
)
def test_an_archive_pin_keeps_its_hash_and_subdirectory(
    archive_info: dict[str, Any], fragment: str
) -> None:
    source, pin = _pin_of(
        {
            "url": "https://x/repo.tar.gz",
            "archive_info": archive_info,
            "subdirectory": "pkg",
        }
    )
    assert source.subdirectory == "pkg"
    assert pin == f"autoware-carla-scenario @ https://x/repo.tar.gz{fragment}"


def test_a_local_directory_names_its_subdirectory(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    source = _source(
        _Dist(  # type: ignore[arg-type]
            {
                "url": tmp_path.as_uri(),
                "dir_info": {"editable": True},
                "subdirectory": "pkg",
            }
        )
    )
    assert source.kind == "editable"
    assert source.path == str(tmp_path / "pkg")


# -- What capture() reads and write() writes, pinned (#2112) ------------------
#
# Fabricated distributions, as importlib.metadata.distributions() yields them,
# and the environment capture() records of them, the lock and the environment
# file write() writes, byte for byte: the source each direct_url.json gives
# (malformed ones too: values of any JSON type, repeated keys, NaN), the
# distributions shadowed or without a name, and what a failure raises.


class _Fake:
    """A distribution: its metadata's ``Name``, its version, its
    ``direct_url.json`` (``None``: none; an exception: raised reading it)."""

    def __init__(
        self, name: str | None, version: str | None, direct_url: Any = None
    ) -> None:
        self.metadata = {} if name is None else {"Name": name}
        self.version = version
        self._direct_url = direct_url

    def read_text(self, filename: str) -> str | None:
        if filename != "direct_url.json":
            return None
        if isinstance(self._direct_url, BaseException):
            raise self._direct_url
        if self._direct_url is None or isinstance(self._direct_url, str):
            return self._direct_url
        return json.dumps(self._direct_url)


def _git_checkout(path: Path, *, dirty: bool) -> str:
    """Make *path* a git checkout of one commit; return the commit."""
    import shutil
    import subprocess

    if shutil.which("git") is None:
        pytest.skip("git is not installed")

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(path), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    (path / "autoware_carla_scenario").mkdir(parents=True)
    (path / "autoware_carla_scenario" / "f.txt").write_text("one")
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
    if dirty:
        (path / "autoware_carla_scenario" / "f.txt").write_text("two")
    return git("rev-parse", "HEAD")


def _fabricated(root: Path) -> list[_Fake]:
    """The distributions of the pinned environment, in the order
    distributions() yields them (the first of a name is the one recorded)."""
    checkout = root / "checkout"
    plain = root / "d irü"
    plain.mkdir()
    return [
        _Fake("Zeta_Pkg", "1.0"),
        _Fake("zeta.pkg", "2.0", {"url": "https://x/zeta.git", "vcs_info": {}}),
        _Fake(None, "1.0"),
        _Fake("", "1.0"),
        _Fake("alpha", None),
        _Fake(
            "autoware-carla-scenario",
            "26.8.40",
            {
                "url": checkout.as_uri(),
                "dir_info": {"editable": True},
                "subdirectory": "autoware_carla_scenario",
            },
        ),
        _Fake("numpy", "2.1.0"),
        _Fake("openodd-py", "0.18.0", {"url": "https://x/o.whl", "archive_info": {}}),
        _Fake(
            "vcs-odd",
            "0.1",
            {
                "url": "https://x/odd.git",
                "vcs_info": {
                    "vcs": None,
                    "commit_id": 0,
                    "requested_revision": [1, {"a": None}],
                },
                "subdirectory": 5,
            },
        ),
        _Fake(
            "vcs-float",
            "1",
            {
                "url": 123,
                "vcs_info": {"commit_id": "", "requested_revision": 1.5e300},
                "subdirectory": "sub dir/ü",
            },
        ),
        _Fake("vcs-null-info", "1", {"url": "https://x/n.git", "vcs_info": None}),
        _Fake(
            "vcs-falsy",
            "1",
            {
                "url": "https://x/f.git",
                "vcs_info": {"vcs": "hg", "commit_id": "", "requested_revision": ""},
                "subdirectory": "",
            },
        ),
        _Fake(
            "vcs-nan",
            "1",
            '{"url": "u", "vcs_info": {"vcs": Infinity, "commit_id": -0.0,'
            ' "requested_revision": NaN}, "subdirectory": 12345678901234567890}',
        ),
        _Fake(
            "dup-keys",
            "9",
            '{"url": "a", "url": "https://b", "vcs_info":'
            ' {"commit_id": "c1", "commit_id": "c2", "vcs": "git"}}',
        ),
        _Fake(
            "arch",
            "1",
            {
                "url": "https://x/a.tgz",
                "archive_info": {
                    "hashes": {"SHA256": "ab", "md5": 3, "sha1": ""},
                    "hash": "MD5=xx",
                },
            },
        ),
        _Fake(
            "arch-legacy",
            "1",
            {
                "url": "https://x/l.tgz",
                "archive_info": {"hash": "sha512=ff"},
                "subdirectory": [1],
            },
        ),
        _Fake("arch-null", "1", {"url": "https://x/n.tgz", "archive_info": None}),
        _Fake("arch-zero", "1", {"url": "https://x/z.tgz", "archive_info": 0}),
        _Fake(
            "arch-nohash", "1", {"url": "https://x/h.tgz", "archive_info": {"hash": 7}}
        ),
        _Fake(
            "local-pct",
            "4",
            {
                "url": "file://" + str(root).replace(" ", "%20") + "/d%20ir%C3%BC",
                "dir_info": {},
            },
        ),
        _Fake(
            "local-sub",
            "4",
            {"url": plain.as_uri(), "subdirectory": "missing", "dir_info": None},
        ),
        _Fake(
            "local-http",
            "5",
            {"url": "https://x/dir", "dir_info": {"editable": 1}, "subdirectory": "s"},
        ),
        _Fake("local-nourl", "6", {}),
        _Fake("local-file-only", "6", {"url": "file:relative/missing"}),
        _Fake("broken-read", "7", RuntimeError("unreadable")),
        _Fake("bad-json", "8", "{not json"),
        _Fake("list-json", "8", "[1]"),
        _Fake("empty-json", "8", ""),
        _Fake("Ünï_Code.x", "1.0é"),
    ]


@pytest.fixture
def fabricated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """capture() over the fabricated distributions, on a pinned interpreter
    and platform; capture()'s cache is cleared before and after."""
    commit = _git_checkout(tmp_path / "checkout", dirty=True)
    dists = _fabricated(tmp_path)
    monkeypatch.setattr(environment.metadata, "distributions", lambda: iter(dists))
    monkeypatch.setattr(
        environment.metadata,
        "packages_distributions",
        lambda: {"zeta_pkg": ["Zeta_Pkg"], "numpy": ["numpy"], "empty": []},
    )
    monkeypatch.setattr(environment.platform, "python_version", lambda: "3.99.1")
    monkeypatch.setattr(environment.platform, "python_implementation", lambda: "PyPy")
    monkeypatch.setattr(environment.platform, "platform", lambda: "Linux-é-x")
    monkeypatch.setattr(environment.platform, "machine", lambda: "riscv64")
    monkeypatch.setattr(environment.sys, "prefix", '/opt/py ü"x\\y')
    capture.cache_clear()
    yield SimpleNamespace(root=tmp_path, commit=commit)
    capture.cache_clear()


#: The environment file write() writes of the fabricated distributions:
#: ``{ROOT}`` is the fixture's directory, ``{COMMIT}`` the checkout's commit
#: and ``{ID}`` the environment's id.
_PINNED_ENVIRONMENT = r"""{
  "schema": "autoware_carla_scenario.environment/1",
  "id": "{ID}",
  "python": "3.99.1",
  "platform": "Linux-\u00e9-x",
  "framework": {
    "name": "autoware-carla-scenario",
    "version": "26.8.40",
    "source": {
      "kind": "editable",
      "path": "{ROOT}/checkout/autoware_carla_scenario",
      "commit": "{COMMIT}",
      "dirty": true
    }
  },
  "scenario": {
    "name": "zeta-pkg",
    "version": "1.0",
    "source": {
      "kind": "index"
    }
  },
  "packages": {
    "openodd-py": "0.18.0",
    "numpy": "2.1.0"
  },
  "implementation": "PyPy",
  "machine": "riscv64",
  "executable_prefix": "/opt/py \u00fc\"x\\y",
  "distributions": [
    {
      "name": "alpha",
      "version": null,
      "source": {
        "kind": "index"
      }
    },
    {
      "name": "arch",
      "version": "1",
      "source": {
        "kind": "archive",
        "url": "https://x/a.tgz",
        "hashes": {
          "sha256": "ab",
          "md5": "xx"
        }
      }
    },
    {
      "name": "arch-legacy",
      "version": "1",
      "source": {
        "kind": "archive",
        "url": "https://x/l.tgz",
        "hashes": {
          "sha512": "ff"
        },
        "subdirectory": [
          1
        ]
      }
    },
    {
      "name": "arch-nohash",
      "version": "1",
      "source": {
        "kind": "archive",
        "url": "https://x/h.tgz"
      }
    },
    {
      "name": "arch-null",
      "version": "1",
      "source": {
        "kind": "archive",
        "url": "https://x/n.tgz"
      }
    },
    {
      "name": "arch-zero",
      "version": "1",
      "source": {
        "kind": "archive",
        "url": "https://x/z.tgz"
      }
    },
    {
      "name": "autoware-carla-scenario",
      "version": "26.8.40",
      "source": {
        "kind": "editable",
        "path": "{ROOT}/checkout/autoware_carla_scenario",
        "commit": "{COMMIT}",
        "dirty": true
      }
    },
    {
      "name": "bad-json",
      "version": "8",
      "source": {
        "kind": "index"
      }
    },
    {
      "name": "broken-read",
      "version": "7",
      "source": {
        "kind": "index"
      }
    },
    {
      "name": "dup-keys",
      "version": "9",
      "source": {
        "kind": "vcs",
        "url": "https://b",
        "vcs": "git",
        "commit": "c2",
        "requested": null
      }
    },
    {
      "name": "empty-json",
      "version": "8",
      "source": {
        "kind": "index"
      }
    },
    {
      "name": "list-json",
      "version": "8",
      "source": {
        "kind": "index"
      }
    },
    {
      "name": "local-file-only",
      "version": "6",
      "source": {
        "kind": "local",
        "path": "relative/missing"
      }
    },
    {
      "name": "local-http",
      "version": "5",
      "source": {
        "kind": "editable",
        "path": "https://x/dir",
        "subdirectory": "s"
      }
    },
    {
      "name": "local-nourl",
      "version": "6",
      "source": {
        "kind": "local",
        "path": ""
      }
    },
    {
      "name": "local-pct",
      "version": "4",
      "source": {
        "kind": "local",
        "path": "{ROOT}/d ir\u00fc"
      }
    },
    {
      "name": "local-sub",
      "version": "4",
      "source": {
        "kind": "local",
        "path": "{ROOT}/d ir\u00fc/missing"
      }
    },
    {
      "name": "numpy",
      "version": "2.1.0",
      "source": {
        "kind": "index"
      }
    },
    {
      "name": "openodd-py",
      "version": "0.18.0",
      "source": {
        "kind": "archive",
        "url": "https://x/o.whl"
      }
    },
    {
      "name": "vcs-falsy",
      "version": "1",
      "source": {
        "kind": "vcs",
        "url": "https://x/f.git",
        "vcs": "hg",
        "commit": "",
        "requested": ""
      }
    },
    {
      "name": "vcs-float",
      "version": "1",
      "source": {
        "kind": "vcs",
        "url": "123",
        "vcs": "git",
        "commit": "",
        "requested": 1.5e+300,
        "subdirectory": "sub dir/\u00fc"
      }
    },
    {
      "name": "vcs-nan",
      "version": "1",
      "source": {
        "kind": "vcs",
        "url": "u",
        "vcs": Infinity,
        "commit": -0.0,
        "requested": NaN,
        "subdirectory": 12345678901234567890
      }
    },
    {
      "name": "vcs-null-info",
      "version": "1",
      "source": {
        "kind": "vcs",
        "url": "https://x/n.git",
        "vcs": "git",
        "commit": "",
        "requested": null
      }
    },
    {
      "name": "vcs-odd",
      "version": "0.1",
      "source": {
        "kind": "vcs",
        "url": "https://x/odd.git",
        "vcs": null,
        "commit": 0,
        "requested": [
          1,
          {
            "a": null
          }
        ],
        "subdirectory": 5
      }
    },
    {
      "name": "zeta-pkg",
      "version": "1.0",
      "source": {
        "kind": "index"
      }
    },
    {
      "name": "\u00fcn\u00ef-code-x",
      "version": "1.0\u00e9",
      "source": {
        "kind": "index"
      }
    }
  ]
}"""

#: The lines of the lock write() writes beside it.
_PINNED_LOCK = (
    "# The environment a scenario run ran in (autoware_carla_scenario).",
    "# environment id: {ID}",
    "# python PyPy 3.99.1 on Linux-é-x",
    "# framework: autoware-carla-scenario==26.8.40  # editable from {ROOT}/checkout/autoware_carla_scenario, git {COMMIT} with uncommitted changes",
    "# scenario: zeta-pkg==1.0",
    "alpha==None",
    "arch @ https://x/a.tgz#sha256=ab",
    "arch-legacy @ https://x/l.tgz#sha512=ff&subdirectory=[1]",
    "arch-nohash @ https://x/h.tgz",
    "arch-null @ https://x/n.tgz",
    "arch-zero @ https://x/z.tgz",
    "autoware-carla-scenario==26.8.40  # editable from {ROOT}/checkout/autoware_carla_scenario, git {COMMIT} with uncommitted changes",
    "bad-json==8",
    "broken-read==7",
    "dup-keys @ git+https://b@c2",
    "empty-json==8",
    "list-json==8",
    "local-file-only==6  # local from relative/missing",
    "local-http==5  # editable from https://x/dir",
    "local-nourl==6  # local from ",
    "local-pct==4  # local from {ROOT}/d irü",
    "local-sub==4  # local from {ROOT}/d irü/missing",
    "numpy==2.1.0",
    "openodd-py @ https://x/o.whl",
    "vcs-falsy==1",
    "vcs-float @ git+123@1.5e+300#subdirectory=sub dir/ü  # requested 1.5e+300, no commit recorded: may move",
    "vcs-nan @ inf+u@nan#subdirectory=12345678901234567890  # requested nan, no commit recorded: may move",
    "vcs-null-info==1",
    "vcs-odd @ None+https://x/odd.git@[1, {'a': None}]#subdirectory=5  # requested [1, {'a': None}], no commit recorded: may move",
    "zeta-pkg==1.0",
    "ünï-code-x==1.0é",
)


class _ZetaScenario:
    """A scenario of the fabricated ``zeta_pkg`` package."""


_ZetaScenario.__module__ = "zeta_pkg.scenarios.cut_in"


def _expected(template: str, fabricated: Any, env_id: str) -> str:
    return (
        template.replace("{ROOT}", json.dumps(str(fabricated.root))[1:-1])
        .replace("{COMMIT}", fabricated.commit)
        .replace("{ID}", env_id)
    )


def test_the_fabricated_environment_is_written_byte_for_byte(
    fabricated: Any, tmp_path: Path
) -> None:
    import hashlib

    summary = write(tmp_path / "out", "Demo", _ZetaScenario())
    assert summary is not None
    lock = "\n".join(
        line.replace("{ROOT}", str(fabricated.root)).replace(
            "{COMMIT}", fabricated.commit
        )
        for line in _PINNED_LOCK
    )
    pins = lock.splitlines()[5:]
    env_id = hashlib.sha256(
        "\n".join(["python PyPy 3.99.1", *pins]).encode()
    ).hexdigest()
    assert summary["id"] == env_id
    written = (tmp_path / "out" / "Demo_requirements.lock").read_bytes()
    assert written == (lock.replace("{ID}", env_id) + "\n").encode("utf-8")
    text = (tmp_path / "out" / "Demo_environment.json").read_bytes()
    assert text == _expected(_PINNED_ENVIRONMENT, fabricated, env_id).encode("ascii")
    document = json.loads(text)
    assert summary == {
        k: document[k]
        for k in ("id", "python", "platform", "framework", "scenario", "packages")
    }
    assert list(summary) == [
        "id",
        "python",
        "platform",
        "framework",
        "scenario",
        "packages",
    ]
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [
        "Demo_environment.json",
        "Demo_requirements.lock",
    ]


def test_the_fabricated_environment_s_values_and_types(fabricated: Any) -> None:
    import math

    env = capture()
    assert capture() is env
    assert type(env.packages) is list
    assert all(
        type(p) is dict and list(p) == ["name", "version", "source"]
        for p in env.packages
    )
    assert [p["name"] for p in env.packages] == sorted(p["name"] for p in env.packages)
    assert env.by_name == {p["name"]: p for p in env.packages}
    assert all(env.by_name[p["name"]] is p for p in env.packages)
    assert (env.python, env.implementation, env.platform, env.machine) == (
        "3.99.1",
        "PyPy",
        "Linux-é-x",
        "riscv64",
    )
    assert env.by_name["alpha"]["version"] is None
    # The first of two names that normalise alike is the one recorded.
    assert env.by_name["zeta-pkg"]["version"] == "1.0"
    odd = env.by_name["vcs-odd"]["source"]
    assert odd["vcs"] is None and type(odd["commit"]) is int and odd["commit"] == 0
    assert odd["requested"] == [1, {"a": None}] and odd["subdirectory"] == 5
    nan = env.by_name["vcs-nan"]["source"]
    assert nan["vcs"] == math.inf and math.isnan(nan["requested"])
    assert type(nan["commit"]) is float and math.copysign(1.0, nan["commit"]) == -1.0
    assert nan["subdirectory"] == 12345678901234567890
    assert env.by_name["vcs-float"]["source"]["url"] == "123"
    assert env.by_name["arch"]["source"]["hashes"] == {"sha256": "ab", "md5": "xx"}
    assert env.by_name["arch-legacy"]["source"]["subdirectory"] == [1]
    assert env.by_name["dup-keys"]["source"] == {
        "kind": "vcs",
        "url": "https://b",
        "vcs": "git",
        "commit": "c2",
        "requested": None,
    }
    framework = env.by_name["autoware-carla-scenario"]["source"]
    assert framework["commit"] == fabricated.commit and framework["dirty"] is True
    document = env.to_dict(_ZetaScenario())
    assert list(document) == [
        "schema",
        "id",
        "python",
        "platform",
        "framework",
        "scenario",
        "packages",
        "implementation",
        "machine",
        "executable_prefix",
        "distributions",
    ]
    assert document["distributions"] is env.packages
    assert document["executable_prefix"] == '/opt/py ü"x\\y'
    summary = env.summary(_ZetaScenario())
    assert summary["framework"]["source"] is framework
    assert summary["scenario"]["source"] is env.by_name["zeta-pkg"]["source"]
    assert summary["packages"] == {"openodd-py": "0.18.0", "numpy": "2.1.0"}
    assert env.pins == [line for line in env.lock().splitlines() if line[0] != "#"]
    assert env.scenario_package(_ZetaScenario()) is env.by_name["zeta-pkg"]


class _Elsewhere:
    pass


@pytest.mark.parametrize(
    ("module", "found"),
    [
        ("zeta_pkg", "zeta-pkg"),
        ("numpy.linalg", "numpy"),
        ("empty.x", None),
        ("unknown.x", None),
        ("", None),
        (".zeta_pkg", None),
    ],
)
def test_which_distribution_a_scenario_comes_from(
    fabricated: Any, module: str, found: str | None
) -> None:
    _Elsewhere.__module__ = module
    env = capture()
    package = env.scenario_package(_Elsewhere())
    assert (None if package is None else package["name"]) == found
    assert env.scenario_package(None) is None
    lock = env.lock(_Elsewhere())
    assert ("# scenario: " in lock) == (found is not None)


def test_the_framework_s_own_scenario_is_not_named_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    framework = {
        "name": "autoware-carla-scenario",
        "version": "1",
        "source": {"kind": "index"},
    }
    env = RunEnvironment([framework])
    monkeypatch.setattr(
        environment.metadata,
        "packages_distributions",
        lambda: {"zeta_pkg": ["Autoware_Carla.Scenario"]},
    )
    assert env.scenario_package(_ZetaScenario()) is framework
    lock = env.lock(_ZetaScenario()).splitlines()
    assert lock[3] == "# framework: autoware-carla-scenario==1"
    assert not any(line.startswith("# scenario:") for line in lock)
    bare = RunEnvironment([])
    assert bare.summary()["framework"] is None and bare.summary()["packages"] == {}
    assert bare.lock().splitlines()[3:] == []
    assert bare.pins == []


def test_a_run_environment_reads_its_packages_when_asked() -> None:
    packages: list[dict[str, Any]] = [
        {"name": "a", "version": "1", "source": {"kind": "index"}}
    ]
    env = RunEnvironment(packages)
    assert env.packages is packages
    before = env.id
    packages.append({"name": "b", "version": 2, "source": {"kind": "local", "path": 3}})
    assert env.pins == ["a==1", "b==2  # local from 3"]
    assert env.id != before
    assert "b" not in env.by_name
    # A package without a name is refused at once, one without a version or a
    # source only when it is pinned.
    with pytest.raises(KeyError, match="name"):
        RunEnvironment([{"version": "1"}])
    partial = RunEnvironment([{"name": "c"}])
    assert partial.by_name == {"c": {"name": "c"}}
    with pytest.raises(KeyError, match="version"):
        partial.pins
    with pytest.raises(KeyError, match="version"):
        partial.summary()


@pytest.mark.parametrize(
    ("direct_url", "error", "message"),
    [
        (
            {"url": "u", "vcs_info": [1]},
            AttributeError,
            "'list' object has no attribute 'get'",
        ),
        (
            {"url": "u", "vcs_info": "git"},
            AttributeError,
            "'str' object has no attribute 'get'",
        ),
        (
            {"url": "u", "archive_info": [1]},
            AttributeError,
            "'list' object has no attribute 'get'",
        ),
        (
            {"url": "u", "archive_info": {"hashes": [1]}},
            AttributeError,
            "'list' object has no attribute 'items'",
        ),
        (
            {"url": "u", "dir_info": "x"},
            AttributeError,
            "'str' object has no attribute 'get'",
        ),
        ({"url": "file:///tmp", "subdirectory": 3}, TypeError, None),
        ({"url": "file://[x/a"}, ValueError, "Invalid IPv6 URL"),
    ],
    ids=[
        "vcs-info-list",
        "vcs-info-str",
        "archive-info-list",
        "archive-hashes-list",
        "dir-info-str",
        "file-subdirectory-int",
        "file-url-invalid",
    ],
)
def test_a_source_that_cannot_be_read_fails_the_capture(
    fabricated: Any,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
    direct_url: Any,
    error: type[Exception],
    message: str | None,
) -> None:
    dists = [_Fake("first", "1"), _Fake("odd", "1", direct_url)]
    monkeypatch.setattr(environment.metadata, "distributions", lambda: iter(dists))
    with pytest.raises(error) as raised:
        _source(dists[1])  # type: ignore[arg-type]
    if message is not None:
        assert str(raised.value) == message
    with pytest.raises(error):
        capture()
    with caplog.at_level("WARNING", logger=environment.__name__):
        assert write(tmp_path / "out", "Demo") is None
    assert [r.getMessage() for r in caplog.records] == [
        "could not write the run's environment"
    ]
    assert caplog.records[0].exc_info is not None
    assert caplog.records[0].exc_info[0] is error
    assert not (tmp_path / "out").exists()


def test_a_source_that_is_no_utf_8_fails_the_id(
    fabricated: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A lone surrogate is JSON, but no UTF-8: the source is recorded and
    # pinned, but the id (a hash of the pins' UTF-8) fails, and nothing is
    # written in the directory made for it.
    dists = [_Fake("s", "1", '{"url": "u", "vcs_info": {"commit_id": "\\ud800"}}')]
    monkeypatch.setattr(environment.metadata, "distributions", lambda: iter(dists))
    env = capture()
    assert env.packages[0]["source"]["commit"] == "\ud800"
    assert env.pins == ["s @ git+u@\ud800"]
    with pytest.raises(UnicodeEncodeError):
        env.id
    assert write(tmp_path / "out", "Demo") is None
    assert list((tmp_path / "out").iterdir()) == []


def test_a_deeply_nested_source_value_is_kept(
    fabricated: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    deep = "[" * 200 + "]" * 200
    dists = [_Fake("d", "1", '{"url": "u", "vcs_info": {"commit_id": ' + deep + "}}")]
    monkeypatch.setattr(environment.metadata, "distributions", lambda: iter(dists))
    env = capture()
    assert json.dumps(env.packages[0]["source"]["commit"]) == deep.replace("][", "], [")
    assert env.pins == ["d @ git+u@" + deep.replace("][", "], [")]
    assert write(tmp_path, "Demo") is not None
