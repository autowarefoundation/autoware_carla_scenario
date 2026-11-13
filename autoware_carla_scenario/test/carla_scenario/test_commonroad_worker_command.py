"""How commonroad.worker finds the command that starts the CommonRoad worker.

``_old_worker_command``, ``_old_checkout_project`` and ``_old_excerpt`` are
``commonroad.worker``'s ``worker_command()``, ``_checkout_project()`` and
``_excerpt()`` as they were before they moved into the library check
(#2116), copied verbatim but for the file ``_checkout_project()`` starts
from, which is handed in (``commonroad/worker.py``'s, where it was).  What
the module exports now must answer the same, or raise the same, for every
way the command is chosen: ``$AUTOWARE_CARLA_SCENARIO_COMMONROAD_WORKER``
(blank, quoted, malformed), ``commonroad-metrics-worker`` on the ``PATH``,
``uv`` on the ``PATH`` in a checkout, or none of them.
"""

from __future__ import annotations

import os
import random
import shlex
import shutil
from pathlib import Path
from typing import Callable, Optional

import pytest

import autoware_carla_scenario.commonroad as commonroad
from autoware_carla_scenario.commonroad import worker as cr_worker

#: The environment variable that names the command starting the worker.
WORKER_ENV = "AUTOWARE_CARLA_SCENARIO_COMMONROAD_WORKER"
#: The worker's console script.
WORKER_SCRIPT = "commonroad-metrics-worker"

#: How much of an answer out of protocol an error quotes.
_EXCERPT_CHARS = 200

#: The file the old _checkout_project() started from.
_WORKER_FILE = cr_worker.__file__


def _old_excerpt(line: str) -> str:
    """*line*, stripped and cut to :data:`_EXCERPT_CHARS`, quoted."""
    text = line.strip()
    if len(text) > _EXCERPT_CHARS:
        text = text[:_EXCERPT_CHARS] + "..."
    return repr(text)


def _old_checkout_project() -> Optional[Path]:
    """``commonroad_metrics/`` of the repository this package was installed
    from in place (``uv sync`` installs the workspace editable), if any."""
    # src/autoware_carla_scenario/commonroad/worker.py -> the repository root.
    root = Path(_WORKER_FILE).resolve().parents[4]
    project = root / "commonroad_metrics"
    return project if (project / "pyproject.toml").is_file() else None


def _old_worker_command() -> list[str]:
    """The command that starts the worker (the module's docstring says how it
    is chosen).

    Raises:
        CommonRoadWorkerError: No way to start it was found.
    """
    configured = os.environ.get(WORKER_ENV, "").strip()
    if configured:
        return shlex.split(configured)
    script = shutil.which(WORKER_SCRIPT)
    if script:
        return [script]
    project = _old_checkout_project()
    uv = shutil.which("uv")
    if project is not None and uv:
        return [uv, "run", "--quiet", "--project", str(project), WORKER_SCRIPT]
    raise cr_worker.CommonRoadWorkerError(
        f"the CommonRoad metrics need their worker ({WORKER_SCRIPT}): install the "
        f"commonroad_metrics project (Python 3.10 or 3.11) and put it on the PATH, "
        f"or name the command in ${WORKER_ENV}"
    )


def _outcome(call: Callable[[], object]) -> tuple[str, object]:
    try:
        return ("returned", call())
    except Exception as exc:  # noqa: BLE001 - compared, whatever it is
        return ("raised", (type(exc), str(exc)))


def test_the_names_are_the_same() -> None:
    assert cr_worker.WORKER_ENV == WORKER_ENV
    assert cr_worker.WORKER_SCRIPT == WORKER_SCRIPT
    assert cr_worker._EXCERPT_CHARS == _EXCERPT_CHARS
    assert cr_worker.__all__ == [
        "WORKER_ENV",
        "CommonRoadWorker",
        "CommonRoadWorkerError",
        "worker_command",
    ]
    # The package re-exports the module's objects themselves: an error the
    # worker raises is caught as the package's.
    assert commonroad.CommonRoadWorkerError is cr_worker.CommonRoadWorkerError
    assert commonroad.worker_command is cr_worker.worker_command
    assert commonroad.WORKER_ENV is cr_worker.WORKER_ENV
    assert commonroad.CommonRoadWorker is cr_worker.CommonRoadWorker
    assert cr_worker.CommonRoadWorkerError.__mro__[1:] == RuntimeError.__mro__


def test_the_checkout_s_project_is_the_same() -> None:
    assert cr_worker._checkout_project() == _old_checkout_project()
    assert cr_worker._checkout_project() is not None


def _executable(directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)
    return path


@pytest.mark.parametrize(
    "configured",
    [
        None,
        "",
        "   ",
        "\t\n",
        "my-python -m commonroad_metrics --x 'a b'",
        '  uv run --project "/a dir/commonroad_metrics" commonroad-metrics-worker  ',
        "worker # a comment",
        r"worker a\ b 'c'\''d' \"e",
        "worker 'unterminated",
        'worker "unterminated',
        "worker trailing\\",
        "wörker ünïcode",
    ],
)
@pytest.mark.parametrize("on_path", ["script", "uv", "both", "none", "unset"])
def test_the_worker_command_is_the_same(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    configured: Optional[str],
    on_path: str,
) -> None:
    if configured is None:
        monkeypatch.delenv(WORKER_ENV, raising=False)
    else:
        monkeypatch.setenv(WORKER_ENV, configured)
    directories = []
    if on_path in ("script", "both"):
        _executable(tmp_path / "scripts", WORKER_SCRIPT)
        directories.append(str(tmp_path / "scripts"))
    if on_path in ("uv", "both"):
        _executable(tmp_path / "uv", "uv")
        directories.append(str(tmp_path / "uv"))
    if on_path == "unset":
        monkeypatch.delenv("PATH", raising=False)
    else:
        monkeypatch.setenv("PATH", os.pathsep.join(directories or [str(tmp_path)]))
    new = _outcome(cr_worker.worker_command)
    assert new == _outcome(_old_worker_command)
    assert new == _outcome(commonroad.worker_command)


def test_a_worker_command_error_is_the_module_s(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(WORKER_ENV, raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(cr_worker.CommonRoadWorkerError) as raised:
        cr_worker.worker_command()
    assert type(raised.value) is commonroad.CommonRoadWorkerError
    assert str(raised.value) == (
        "the CommonRoad metrics need their worker (commonroad-metrics-worker): "
        "install the commonroad_metrics project (Python 3.10 or 3.11) and put it "
        "on the PATH, or name the command in "
        "$AUTOWARE_CARLA_SCENARIO_COMMONROAD_WORKER"
    )


def test_an_excerpt_is_the_same() -> None:
    rng = random.Random(3)
    alphabet = "ab x\t\n\r\x0b\x0c'\"\\é漢\u2028\u00a0\x00"
    lines = ["", " ", "x" * 200, "x" * 201, " " + "y" * 200 + " ", "z" * 5000]
    lines += [
        "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 420)))
        for _ in range(2000)
    ]
    for line in lines:
        assert cr_worker._excerpt(line) == _old_excerpt(line)
