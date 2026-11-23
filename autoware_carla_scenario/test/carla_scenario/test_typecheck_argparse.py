"""The argparse shim (``typecheck/codon/argparse.codon``, #2374).

It declares what the checked serve_cli uses of argparse, with CPython 3.10's
signatures: ArgumentTypeError, ArgumentParser and its add_argument(). The
check takes what CPython takes of it and refuses what CPython refuses, each
refusal confirmed here on the running CPython too.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import textwrap
from pathlib import Path
from typing import Any

import pytest

from autoware_carla_scenario.typecheck import (
    TypeCheckResult,
    available_toolchain,
    check,
    typecheck_library,
)
from autoware_carla_scenario.typecheck.library import package_modules

_PACKAGE = "autoware_carla_scenario"
_FAKE = f"{_PACKAGE}.zz_argparse_check_case"

# The checker's own tests: a pull request runs them only when the checker may
# have changed (.github/actions/codon-selftests).
pytestmark = [pytest.mark.codon, pytest.mark.codon_selftest]

needs_codon = pytest.mark.skipif(
    available_toolchain() is None, reason="no Codon compiler"
)


def _check_case(
    source: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> TypeCheckResult:
    path = tmp_path / "zz_argparse_check_case.py"
    path.write_text(textwrap.dedent(source))
    modules = package_modules()
    modules[_FAKE] = path
    monkeypatch.setattr(check, "package_modules", lambda: modules)
    return typecheck_library([_FAKE])


_PARSERS = """
from __future__ import annotations

import argparse


def number(text: str) -> int:
    try:
        return int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None


def half(value: int) -> int:
    return value // 2


def build() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tool", add_help=True, exit_on_error=False)
    parser.add_argument("--name", default="a", help="the name")
    parser.add_argument("--count", type=number, default="1", help="how many")
    parser.add_argument("--limit", "-l", type=number, default=None)
    parser.add_argument("--scale", type=half, default=4)
    parser.add_argument("path")
    other = argparse.ArgumentParser("other", None, "another tool", None, allow_abbrev=False)
    other.add_argument("--flag", help=None)
    return parser


def refuse() -> str:
    try:
        raise argparse.ArgumentTypeError("bad")
    except argparse.ArgumentTypeError as exc:
        return str(exc) + str(argparse.ArgumentTypeError())
"""

#: (right, wrong): each wrong one fails in CPython 3.10 too, a TypeError or
#: AttributeError raised by the call, or an argparse error when the string
#: default goes through `type` in parse_args().
_REFUSED = [
    ('prog="tool"', 'progname="tool"'),
    ("None, allow_abbrev=False)", "None, False)"),
    ('type=number, default="1"', 'typ=number, default="1"'),
    ('type=number, default="1"', 'type=half, default="1"'),
    ('add_argument("--name"', 'add_argument(1, "--name"'),
    ('add_argument("--flag", help=None)', "add_argument(help=None)"),
    ('parser.add_argument("path")', 'parser.add_arg("path")'),
]
_REFUSED_IDS = [
    "unknown-parser-keyword",
    "parents-not-declared",
    "unknown-argument-keyword",
    "type-takes-the-str-default",
    "flag-is-a-str",
    "a-name-or-flag",
    "no-method",
]


def _run_in_cpython(source: str) -> list[type[BaseException]]:
    """Build *source*'s parser and parse a command line: what it raised."""
    raised: list[type[BaseException]] = []
    namespace: dict[str, Any] = {}
    exec(compile(textwrap.dedent(source), "<parsers>", "exec"), namespace)
    try:
        # A usage error prints the usage on stderr before it exits.
        with contextlib.redirect_stderr(io.StringIO()):
            parsed = namespace["build"]().parse_args(["p"])
        assert (parsed.name, parsed.count, parsed.limit, parsed.path) == (
            "a",
            1,
            None,
            "p",
        )
        assert namespace["refuse"]() == "bad"
    except (TypeError, AttributeError, argparse.ArgumentError, SystemExit) as exc:
        raised.append(type(exc))
    return raised


def test_what_the_argparse_shim_takes_runs_in_cpython() -> None:
    assert _run_in_cpython(_PARSERS) == []


@pytest.mark.parametrize(("right", "wrong"), _REFUSED, ids=_REFUSED_IDS)
def test_what_the_argparse_shim_refuses_fails_in_cpython(
    right: str, wrong: str
) -> None:
    source = _PARSERS.replace(right, wrong, 1)
    assert source != _PARSERS
    assert _run_in_cpython(source)


@needs_codon
def test_the_argparse_shim_takes_what_cpython_takes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _check_case(_PARSERS, tmp_path, monkeypatch)
    assert result.ok, result.format()


@needs_codon
@pytest.mark.parametrize(("right", "wrong"), _REFUSED, ids=_REFUSED_IDS)
def test_the_argparse_shim_refuses_what_cpython_refuses(
    right: str, wrong: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _PARSERS.replace(right, wrong, 1)
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
