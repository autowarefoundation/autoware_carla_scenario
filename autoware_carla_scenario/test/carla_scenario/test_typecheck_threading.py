"""The threading shim (``typecheck/codon/threading.codon``, #132).

It replaces the threading module Codon ships, so it keeps all of that one,
and adds Thread with CPython 3.10's signature: the check takes what CPython
takes and refuses what CPython refuses, each refusal confirmed here on the
running CPython too.
"""

from __future__ import annotations

import ast
import textwrap
import threading
from pathlib import Path
from typing import Any

import pytest

from autoware_carla_scenario.typecheck import (
    TypeCheckResult,
    available_toolchain,
    check,
    typecheck_library,
)
from autoware_carla_scenario.typecheck import model_dir
from autoware_carla_scenario.typecheck.library import package_modules

_PACKAGE = "autoware_carla_scenario"
_FAKE = f"{_PACKAGE}.zz_threading_check_case"

# The checker's own tests: a pull request runs them only when the checker may
# have changed (.github/actions/codon-selftests).
pytestmark = [pytest.mark.codon, pytest.mark.codon_selftest]

needs_codon = pytest.mark.skipif(
    available_toolchain() is None, reason="no Codon compiler"
)


def _check_case(
    source: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> TypeCheckResult:
    path = tmp_path / "zz_threading_check_case.py"
    path.write_text(textwrap.dedent(source))
    modules = package_modules()
    modules[_FAKE] = path
    monkeypatch.setattr(check, "package_modules", lambda: modules)
    return typecheck_library([_FAKE])


def _codon_top_level_names(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


@needs_codon
def test_the_threading_shim_keeps_all_of_codons_threading() -> None:
    # The shim replaces the threading module Codon ships, which Codon's own
    # numpy (`from threading import Lock`) and typesafe_carla's Codon library
    # (`threading.Lock()`) import: it declares every name Codon's does, and
    # Lock and RLock as Codon declares them.
    tc = available_toolchain()
    assert tc is not None
    codon_threading = tc.codon_dir / "lib" / "codon" / "stdlib" / "threading.codon"
    if not codon_threading.is_file():
        pytest.skip("this Codon ships no stdlib/threading.codon")
    shim_path = model_dir() / "threading.codon"
    shim = _codon_top_level_names(shim_path)
    assert _codon_top_level_names(codon_threading) <= shim
    assert "Thread" in shim

    def classes(path: Path) -> dict[str, str]:
        return {
            node.name: ast.unparse(node)
            for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.ClassDef) and node.name in ("Lock", "RLock")
        }

    assert classes(shim_path) == classes(codon_threading)


_THREADS = """
from __future__ import annotations

import threading


def work(name: str, count: int) -> None:
    pass


def idle() -> None:
    pass


class Worker:
    _lock: threading.Lock
    _reentrant: threading.RLock

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reentrant = threading.RLock()

    def run(self, name: str) -> bool:
        thread = threading.Thread(target=work, args=(name, 1), daemon=True)
        thread.start()
        thread.join(timeout=1.0)
        other = threading.Thread(None, work, "worker", (name, 2))
        other.start()
        other.join()
        bare = threading.Thread(target=idle, name=None, kwargs=None, daemon=None)
        bare.start()
        bare.join(1)
        with self._lock:
            alive = thread.is_alive()
        with self._reentrant:
            return alive or other.is_alive() or bare.is_alive()
"""

#: (right, wrong): each wrong one fails in CPython 3.10 too, a TypeError,
#: AssertionError or AttributeError, raised by the call or by the thread.
_REFUSED = [
    ("args=(name, 1)", "args=(name,)"),
    ("(name, 2))", "(name, 2), None, True)"),
    ("Thread(target=work,", "Thread(work, target=work,"),
    ("daemon=True)", "daemonic=True)"),
    ("thread.join(timeout=1.0)", 'thread.join(timeout="1")'),
    ("thread.start()", "thread.start(True)"),
    ("alive = thread.is_alive()", "alive = thread.isAlive()"),
]
_REFUSED_IDS = [
    "target-arguments",
    "daemon-keyword-only",
    "group-is-none",
    "unknown-keyword",
    "join-timeout-number",
    "start-arguments",
    "no-method",
]


def _run_in_cpython(source: str) -> list[type[BaseException]]:
    """Run Worker().run() of *source*: what it raised, in the call or a thread."""
    raised: list[type[BaseException]] = []
    namespace: dict[str, Any] = {}
    exec(compile(textwrap.dedent(source), "<threads>", "exec"), namespace)
    hook = threading.excepthook
    threading.excepthook = lambda args: raised.append(args.exc_type)
    # Join only the threads *source* starts: one an earlier test left running
    # (a gRPC server's, an executor's) may never let a join return (#228).
    before = set(threading.enumerate())
    try:
        namespace["Worker"]().run("a")
    except (TypeError, AssertionError, AttributeError) as exc:
        raised.append(type(exc))
    finally:
        threading.excepthook = hook
    for thread in threading.enumerate():
        if thread not in before:
            thread.join(timeout=5.0)
    return raised


def test_what_the_threading_shim_takes_runs_in_cpython() -> None:
    assert _run_in_cpython(_THREADS) == []


@pytest.mark.parametrize(("right", "wrong"), _REFUSED, ids=_REFUSED_IDS)
def test_what_the_threading_shim_refuses_fails_in_cpython(
    right: str, wrong: str
) -> None:
    source = _THREADS.replace(right, wrong, 1)
    assert source != _THREADS
    assert _run_in_cpython(source)


@needs_codon
def test_the_threading_shim_takes_what_cpython_takes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _check_case(_THREADS, tmp_path, monkeypatch)
    assert result.ok, result.format()


@needs_codon
@pytest.mark.parametrize(("right", "wrong"), _REFUSED, ids=_REFUSED_IDS)
def test_the_threading_shim_refuses_what_cpython_refuses(
    right: str, wrong: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _THREADS.replace(right, wrong, 1)
    result = _check_case(source, tmp_path, monkeypatch)
    assert not result.ok
    line = next(
        i
        for i, text in enumerate(textwrap.dedent(source).splitlines(), 1)
        if wrong in text
    )
    located = [d for d in result.diagnostics if d.path and d.path.endswith(".py")]
    assert located, result.format()
    assert located[0].line == line, result.format()
