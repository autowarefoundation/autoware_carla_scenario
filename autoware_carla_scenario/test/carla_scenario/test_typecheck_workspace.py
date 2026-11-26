"""The checker's workspace: built for the toolchain it is given (#2091).

None of these runs Codon: the toolchain is a directory laid out like one.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from autoware_carla_scenario.typecheck import check
from autoware_carla_scenario.typecheck.toolchain import Toolchain


def _fake_toolchain(root: Path, marker: str) -> Toolchain:
    """A toolchain whose standard library has each module the model extends."""
    stdlib = root / "lib" / "codon" / "stdlib"
    stdlib.mkdir(parents=True)
    for extension in (check.model_dir() / check._STDLIB_EXTENSIONS).glob("*.codon"):
        (stdlib / extension.name).write_text(
            f"# {marker} {extension.name}\n", encoding="utf-8"
        )
    return Toolchain(root / "bin" / "codon", root, source="test")


def _no_lookup() -> Toolchain:
    raise AssertionError("the workspace looked up a Codon of its own")


def test_the_workspace_extends_the_given_toolchains_stdlib(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(check, "find_codon", _no_lookup)
    tc = _fake_toolchain(tmp_path / "codon", "given")
    codon_path = tmp_path / "codon_path"
    codon_path.mkdir()
    (codon_path / "carla_lib.codon").write_text("", encoding="utf-8")
    root = tmp_path / "ws"
    check._base_workspace(root, None, codon_path, tc)
    extensions = sorted((check.model_dir() / check._STDLIB_EXTENSIONS).glob("*.codon"))
    assert extensions
    for extension in extensions:
        text = (root / extension.name).read_text(encoding="utf-8")
        assert text.startswith(f"# given {extension.name}\n"), extension.name
        assert text.endswith(extension.read_text(encoding="utf-8"))
    assert not (root / check._STDLIB_EXTENSIONS).exists()
    assert (root / "carla_lib.codon").is_symlink()


class _Built(Exception):
    pass


def test_both_builders_pass_their_toolchain_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Toolchain] = []

    def base(root: Path, driver: object, codon_path: Path, tc: Toolchain) -> None:
        seen.append(tc)
        raise _Built

    monkeypatch.setattr(check, "find_codon", _no_lookup)
    monkeypatch.setattr(check, "_base_workspace", base)
    tc = Toolchain(tmp_path / "codon", tmp_path, source="test")
    sources = SimpleNamespace(modules={}, problems=[])
    with pytest.raises(_Built):
        check._build_workspace(tmp_path, sources, None, tmp_path, tc)  # type: ignore[arg-type]
    with pytest.raises(_Built):
        check._build_library_workspace(tmp_path, sources, [], None, tmp_path, tc)  # type: ignore[arg-type]
    assert seen == [tc, tc]


def test_the_compile_builds_for_its_toolchain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A problem found while building returns before Codon would run.
    seen: list[Toolchain] = []

    def build(
        root: Path, sources: object, driver: object, codon_path: Path, tc: Toolchain
    ) -> tuple[check._Workspace, list[check.Diagnostic]]:
        seen.append(tc)
        return check._Workspace(root, driver), [check.Diagnostic("built")]

    monkeypatch.setattr(check, "find_codon", _no_lookup)
    monkeypatch.setattr(check, "_build_workspace", build)
    tc = Toolchain(tmp_path / "codon", tmp_path, source="test")
    sources = SimpleNamespace(modules={}, problems=[])
    result = check._compile("s", sources, None, tc, tmp_path, 1.0)  # type: ignore[arg-type]
    assert result is not None
    assert not result.ok
    assert [d.message for d in result.diagnostics] == ["built"]
    assert seen == [tc]


@pytest.mark.skipif(sys.platform != "linux", reason="the C locale's encoding")
def test_the_workspace_is_utf8_whatever_the_locale(tmp_path: Path) -> None:
    # Under the C locale with UTF-8 mode off, Python's default text encoding
    # is ASCII: a non-ASCII character of the standard library failed the
    # workspace build.
    tc = _fake_toolchain(tmp_path / "codon", "naïve")
    codon_path = tmp_path / "codon_path"
    codon_path.mkdir()
    program = textwrap.dedent(
        f"""
        import locale
        from pathlib import Path
        from autoware_carla_scenario.typecheck import check
        from autoware_carla_scenario.typecheck.toolchain import Toolchain
        assert locale.getpreferredencoding(False).lower() in (
            "ascii", "ansi_x3.4-1968"
        ), locale.getpreferredencoding(False)
        tc = Toolchain(Path({str(tc.executable)!r}), Path({str(tc.codon_dir)!r}), "t")
        root = Path({str(tmp_path / "ws")!r})
        check._base_workspace(root, None, Path({str(codon_path)!r}), tc)
        print((root / "math.codon").read_text(encoding="utf-8").splitlines()[0])
        """
    )
    env = {
        **os.environ,
        "LC_ALL": "C",
        "LANG": "C",
        "PYTHONUTF8": "0",
        "PYTHONCOERCECLOCALE": "0",
        # Only the standard streams: the result is printed.
        "PYTHONIOENCODING": "utf-8",
    }
    proc = subprocess.run(  # noqa: S603 - this Python, a fixed program
        [sys.executable, "-c", program],
        env=env,
        capture_output=True,
        encoding="utf-8",
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "# naïve math.codon"


def _checked_workspace(tmp_path: Path) -> check._Workspace:
    """A workspace whose checked files are a package's ``__init__.py`` and
    ``config.py``, unchanged copies of the author's."""
    source = "a = 1\nb = 2\nc = 3\n"
    ws = check._Workspace(tmp_path, None)
    for rel in ("my_pkg/__init__.py", "my_pkg/config.py"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(check.PRELUDE + source, encoding="utf-8")
        ws.files[rel] = f"/user/{rel}"
        ws.origins[rel] = [1, 2, 3]
    return ws


@pytest.mark.parametrize(
    "file",
    [
        "autoware_carla_scenario/__init__.codon",
        "autoware_carla_scenario/config.py",
        "typesafe_carla/__init__.py",
        "/opt/codon/lib/codon/stdlib/config.py",
        "../elsewhere/my_pkg/__init__.py",
    ],
)
def test_a_frame_outside_the_checked_files_is_not_theirs(
    tmp_path: Path, file: str
) -> None:
    # Same basename as a checked file, another directory: the model's,
    # typesafe_carla's or Codon's, never the author's (#2249).
    ws = _checked_workspace(tmp_path)
    line = check._PRELUDE_LINES + 2
    (diagnostic,) = check._parse_output(f"{file}:{line} (1): error: bad\n", ws)
    assert diagnostic.path is None
    assert diagnostic.message == "bad"


@pytest.mark.parametrize(
    "file",
    ["my_pkg/config.py", "./my_pkg/config.py", "my_pkg/../my_pkg/config.py"],
)
def test_a_workspace_relative_frame_is_its_checked_file(
    tmp_path: Path, file: str
) -> None:
    ws = _checked_workspace(tmp_path)
    line = check._PRELUDE_LINES + 2
    (diagnostic,) = check._parse_output(f"{file}:{line} (1): error: bad\n", ws)
    assert diagnostic.path == "/user/my_pkg/config.py"
    assert diagnostic.line == 2


def test_an_absolute_frame_in_the_workspace_is_its_checked_file(
    tmp_path: Path,
) -> None:
    ws = _checked_workspace(tmp_path)
    line = check._PRELUDE_LINES + 3
    file = tmp_path.resolve() / "my_pkg" / "__init__.py"
    (diagnostic,) = check._parse_output(f"{file}:{line} (1): error: bad\n", ws)
    assert diagnostic.path == "/user/my_pkg/__init__.py"
    assert diagnostic.line == 3


def test_a_frame_in_the_model_is_not_the_users_init(tmp_path: Path) -> None:
    # The reproduction of #2249: an error in the model's __init__.codon,
    # reached from the author's config.py, is reported at config.py.
    ws = _checked_workspace(tmp_path)
    line = check._PRELUDE_LINES + 2
    output = (
        f"my_pkg/config.py:{line} (1): error: during the realization of f\n"
        f"╰─ autoware_carla_scenario/__init__.codon:30 (5): error: bad\n"
    )
    (diagnostic,) = check._parse_output(output, ws)
    assert diagnostic.path == "/user/my_pkg/config.py"
    assert diagnostic.line == 2


def test_a_bare_basename_falls_back_to_the_one_checked_file_of_that_name(
    tmp_path: Path,
) -> None:
    ws = _checked_workspace(tmp_path)
    line = check._PRELUDE_LINES + 1
    (diagnostic,) = check._parse_output(f"config.py:{line} (1): error: bad\n", ws)
    assert diagnostic.path == "/user/my_pkg/config.py"
    assert diagnostic.line == 1


def _shared_basename_workspace(tmp_path: Path) -> check._Workspace:
    """The checked files of the audit's reproduction (#2288): ``a/util.py``
    and ``b/util.py``, both with a function across their line 3, and the
    ``odds.py`` that calls ``bad()`` of ``b/util.py``."""
    sources = {
        "acs_dupbase/a/util.py": "def good(x: int) -> int:\n    y = 1\n    return x\n",
        "acs_dupbase/b/util.py": (
            'def bad(x: int) -> int:\n    y = 1\n    return x + "oops"\n'
        ),
        "acs_dupbase/odds.py": "from .b.util import bad\n\n\ndef build():\n    bad(2)\n",
    }
    ws = check._Workspace(tmp_path, None)
    for rel, source in sources.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(check.PRELUDE + source, encoding="utf-8")
        ws.files[rel] = f"/user/{rel}"
        ws.origins[rel] = list(range(1, source.count("\n") + 1))
    return ws


def test_the_realized_function_tells_files_of_one_basename_apart(
    tmp_path: Path,
) -> None:
    # Codon names b/util.py "util.py", as it names a/util.py: the function
    # the next frame realizes, bad(), is b/util.py's alone (#2288).
    ws = _shared_basename_workspace(tmp_path)
    p = check._PRELUDE_LINES
    output = (
        f"util.py:{p + 3} (12-22): error: unsupported operand type(s) for +: "
        "'int' and 'str'\n"
        f"├─ odds.py:{p + 5} (5-8): error: during the realization of bad(x: int)\n"
        f"╰─ odds.py:{p + 4} (1-4): error: during the realization of build()\n"
    )
    (diagnostic,) = check._parse_output(output, ws)
    assert diagnostic.path == "/user/acs_dupbase/b/util.py"
    assert diagnostic.line == 3
    assert diagnostic.message == "unsupported operand type(s) for +: 'int' and 'str'"
    assert diagnostic.trace[0] == "during the realization of bad(x: int) [odds.py:5]"


def test_a_method_realization_names_its_function(tmp_path: Path) -> None:
    assert check._realized_name("during the realization of bad(x: int)") == "bad"
    assert (
        check._realized_name("during the realization of Foo.__init__:0(self: Foo)")
        == "__init__"
    )
    assert check._realized_name("during the realization of f[T](x: T)") == "f"
    assert check._realized_name("unsupported operand type(s)") is None


def test_files_of_one_basename_nothing_tells_apart_are_all_named(
    tmp_path: Path,
) -> None:
    # No frame encloses the error: rather than dropping it for the call
    # site, the diagnostic names every checked util.py and its line (#2288).
    ws = _shared_basename_workspace(tmp_path)
    p = check._PRELUDE_LINES
    (diagnostic,) = check._parse_output(f"util.py:{p + 2} (5): error: bad\n", ws)
    assert diagnostic.path == "/user/acs_dupbase/a/util.py"
    assert diagnostic.line == 2
    assert diagnostic.column is None
    assert diagnostic.message == (
        "bad (Codon names the file util.py only, which more than one checked "
        "file is called: this is one of /user/acs_dupbase/a/util.py:2, "
        "/user/acs_dupbase/b/util.py:2)"
    )


def test_a_line_only_one_file_of_the_basename_has_is_that_files(
    tmp_path: Path,
) -> None:
    ws = _shared_basename_workspace(tmp_path)
    (tmp_path / "acs_dupbase/b/util.py").write_text(
        check.PRELUDE + "def bad(x: int) -> int:\n    y = 1\n    z = 2\n    return x\n",
        encoding="utf-8",
    )
    ws.origins["acs_dupbase/b/util.py"] = [1, 2, 3, 4]
    p = check._PRELUDE_LINES
    (diagnostic,) = check._parse_output(f"util.py:{p + 4} (5): error: bad\n", ws)
    assert diagnostic.path == "/user/acs_dupbase/b/util.py"
    assert diagnostic.line == 4
    assert diagnostic.message == "bad"


def test_an_ambiguous_frame_of_the_trace_names_every_candidate(
    tmp_path: Path,
) -> None:
    ws = _shared_basename_workspace(tmp_path)
    p = check._PRELUDE_LINES
    output = (
        f"odds.py:{p + 5} (5): error: bad\n"
        f"╰─ util.py:{p + 2} (1): error: during the realization of h()\n"
    )
    (diagnostic,) = check._parse_output(output, ws)
    assert diagnostic.path == "/user/acs_dupbase/odds.py"
    assert diagnostic.trace == (
        "during the realization of h() [one of /user/acs_dupbase/a/util.py:2, "
        "/user/acs_dupbase/b/util.py:2]",
    )


def test_the_scenario_check_leaves_imports_only_python_runs_to_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Codon compiles the body of `if TYPE_CHECKING:` (a compile-time true
    # there) and the `else` of `if not TYPE_CHECKING:`, nothing else of
    # them: an import only Python runs needs no Codon model and is not
    # compiled, though it must still find in Python what it imports (#2288).
    package = tmp_path / "src" / "acs_tc2288"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "helper.py").write_text("X = 1\n", encoding="utf-8")
    (package / "scn.py").write_text(
        textwrap.dedent(
            """\
            from typing import TYPE_CHECKING

            if TYPE_CHECKING:
                import acs_tc2288_compiled
            else:
                import acs_tc2288_run_time
            if not TYPE_CHECKING:
                import acs_tc2288_run_time_too
                from autoware_carla_scenario import AcsNoSuchName2288
                from . import helper
            else:
                import acs_tc2288_compiled_too
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path / "src"))
    codon_path = tmp_path / "codon_path"
    codon_path.mkdir()
    tc = _fake_toolchain(tmp_path / "codon", "fake")
    sources = check._collect(["acs_tc2288.scn"], tc, codon_path)
    problems = {(d.line, d.message.split(" ")[0]) for d in sources.problems}
    assert problems == {
        (4, "acs_tc2288_compiled"),
        (12, "acs_tc2288_compiled_too"),
        (9, "autoware_carla_scenario"),
    }
    (missing,) = [d for d in sources.problems if d.line == 9]
    assert "has no AcsNoSuchName2288 in Python" in missing.message
    assert "acs_tc2288.helper" not in sources.modules


def test_type_checking_branches_are_codons_and_pythons() -> None:
    tree = ast.parse(
        "if TYPE_CHECKING:\n    import a\nelse:\n    import b\n"
        "if not TYPE_CHECKING:\n    import c\nelse:\n    import d\n"
    )
    check_time, run_time = check._type_checking_branches(tree)

    def names(stmts: list[ast.stmt]) -> set[str]:
        return {s.names[0].name for s in stmts if isinstance(s, ast.Import)}

    assert names(check_time) == {"a", "d"}
    assert names(run_time) == {"b", "c"}
    assert {
        n.names[0].name
        for n in check._run_time_only_nodes(tree)
        if isinstance(n, ast.Import)
    } == {"b", "c"}


_EXCLUDED_MODULE = "autoware_carla_scenario.wosac.pack"


def test_the_library_check_refuses_an_excluded_module_before_any_codon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Compiling it would only fail and advise excluding it (#2288).
    monkeypatch.setattr(check, "find_supported_codon", _no_lookup)
    result = check.typecheck_library([_EXCLUDED_MODULE, "autoware_carla_scenario.nope"])
    assert not result.ok
    assert result.skipped is None
    assert [d.message for d in result.diagnostics] == [
        "autoware_carla_scenario.nope is not a module of autoware_carla_scenario "
        "the check can compile",
        f"{_EXCLUDED_MODULE} is EXCLUDED in typecheck/library.py: "
        f"{check.EXCLUDED[_EXCLUDED_MODULE]}",
    ]


def test_scenario_check_library_fails_on_an_excluded_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from autoware_carla_scenario.typecheck import cli

    tc = _fake_toolchain(tmp_path / "codon", "fake")
    monkeypatch.setattr(cli, "find_supported_codon", lambda: tc)
    monkeypatch.setattr(check, "_compile", _no_lookup)
    assert cli.main(["--library", _EXCLUDED_MODULE]) == 1
    out = capsys.readouterr().out
    assert out.startswith("[FAILED] ")
    assert f"{_EXCLUDED_MODULE} is EXCLUDED in typecheck/library.py" in out


def test_an_ambiguous_frame_of_one_error_leaves_another_error_alone(
    tmp_path: Path,
) -> None:
    # util.py:3 is ambiguous in the first error, b/util.py's bad() in the
    # second: the second names one file, with no list of candidates (#2288).
    ws = _shared_basename_workspace(tmp_path)
    p = check._PRELUDE_LINES
    output = (
        f"util.py:{p + 3} (5): error: first\n"
        f"util.py:{p + 3} (5): error: second\n"
        f"╰─ odds.py:{p + 5} (5-8): error: during the realization of bad(x: int)\n"
    )
    first, second = check._parse_output(output, ws)
    assert first.message.startswith("first (Codon names the file util.py only")
    assert second.path == "/user/acs_dupbase/b/util.py"
    assert second.message == "second"


@pytest.mark.parametrize(
    "output",
    [
        # Codon's own failed assertion (seqassert), then its abort().
        "Assert failed:\tunknown type\nExpression:\tt\nSource:\t\ttypecheck.cpp:12\n",
        "\x1b[1mAssert failed:\x1b[0m\tbad\n",
        # A shell's report of the signal, alone on its line.
        "Aborted (core dumped)\n",
        "partial output\nAborted\n",
        "Segmentation fault (core dumped)\n",
        # bash's report, after the wrapper script's line and the process id.
        "codon.sh: line 3:  4242 Segmentation fault      (core dumped) codon build\n",
        "/opt/codon.sh: line 12: 77 Aborted                 codon build -llvm\n",
    ],
)
def test_a_crash_codon_reports_is_one(output: str) -> None:
    assert check._crash_reported(output)


@pytest.mark.parametrize(
    "output",
    [
        "scen.py:12 (5): error: 'Aborted' object has no attribute 'missing'\n",
        "scen.py:12 (5): error: Assert failed: no such thing\n",
        "error: 'Status.Aborted' does not match expected type 'int'\n",
        "╰─ scen.py:3 (1): error: during the realization of Aborted()\n",
        "Aborted is not defined here\n",
        "  Segmentation fault\n",
    ],
)
def test_an_error_naming_a_crash_marker_is_no_crash(output: str) -> None:
    # A class or an enum member called Aborted is the program's (#2519).
    assert not check._crash_reported(output)


def _compile_with_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, returncode: int, output: str
) -> check.TypeCheckResult:
    """:func:`check._compile` of a one-module scenario, Codon answering
    *returncode* and *output*."""
    source = "class Aborted:\n    pass\n\nx = Aborted().missing\n"
    author = tmp_path / "scen.py"
    author.write_text(source, encoding="utf-8")
    sources = check._Sources()
    sources.modules["scen"] = check._Module(author, False, source, ast.parse(source))

    def build(root: Path) -> tuple[check._Workspace, list[check.Diagnostic]]:
        ws = check._Workspace(root, None)
        (root / "scen.py").write_text(check.PRELUDE + source, encoding="utf-8")
        ws.files["scen.py"] = str(author)
        ws.origins["scen.py"] = [1, 2, 3, 4]
        return ws, []

    def run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, returncode, "", output)

    monkeypatch.setattr(check.subprocess, "run", run)
    tc = Toolchain(tmp_path / "bin" / "codon", tmp_path, source="test")
    result = check._compile("s", sources, None, tc, tmp_path, 5.0, build)
    assert result is not None
    return result


def test_a_type_error_of_a_class_called_aborted_is_no_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # #2519: "Aborted" in the error's text took it for a crash, so the
    # verdict was never kept and the report said Codon crashed.
    line = check._PRELUDE_LINES + 4
    output = f"scen.py:{line} (5): error: 'Aborted' object has no attribute 'missing'\n"
    result = _compile_with_output(tmp_path, monkeypatch, 1, output)
    assert not result.ok
    assert not result.crashed
    assert [d.message for d in result.diagnostics] == [
        "'Aborted' object has no attribute 'missing'"
    ]
    assert result.diagnostics[0].line == 4


@pytest.mark.parametrize(
    ("returncode", "output"),
    [
        (1, "Assert failed:\tbad\nExpression:\tt\nSource:\t\tx.cpp:1\n"),
        (-6, "Assert failed:\tbad\n"),
        (-11, ""),
        (134, "Aborted (core dumped)\n"),
    ],
)
def test_a_codon_crash_is_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, returncode: int, output: str
) -> None:
    assert _compile_with_output(tmp_path, monkeypatch, returncode, output).crashed


def test_a_column_is_checked_against_the_file_its_error_is_in(
    tmp_path: Path,
) -> None:
    # #2519: the first error leaves util.py:3 ambiguous, placed in both
    # files; the second is good()'s, of a/util.py, whose line 3 is as
    # written. b/util.py's rewritten line 3 must not drop its column.
    ws = check._Workspace(tmp_path / "ws", None)
    good = "def good(x):\n    y = 1\n    return foo(x, 1)\n"
    bad = "def bad(x):\n    y = 1\n    return foo(x, 1)\n"
    for rel, written, compiled in (
        ("p/a/util.py", good, good),
        ("p/b/util.py", bad, bad.replace("foo", "zzz")),
    ):
        checked = tmp_path / "ws" / rel
        checked.parent.mkdir(parents=True, exist_ok=True)
        checked.write_text(check.PRELUDE + compiled, encoding="utf-8")
        author = tmp_path / "author" / rel
        author.parent.mkdir(parents=True, exist_ok=True)
        author.write_text(written, encoding="utf-8")
        ws.files[rel] = str(author)
        ws.origins[rel] = [1, 2, 3]
    line = check._PRELUDE_LINES + 3
    second = (
        f"util.py:{line} (16): error: second\n"
        "╰─ odds.py:5 (5): error: during the realization of good(x: int)\n"
    )
    (alone,) = check._parse_output(second, ws)
    first, after = check._parse_output(
        f"util.py:{line} (16): error: first\n" + second, ws
    )
    assert first.column is None  # ambiguous
    assert after.path == alone.path == str(tmp_path / "author" / "p/a/util.py")
    assert after.column == alone.column == 16


@pytest.mark.parametrize("separator", ["\x0c", " ", "\x1c"])
def test_a_character_splitlines_splits_at_is_no_new_line(
    tmp_path: Path, separator: str
) -> None:
    # #2519: splitlines() splits at a form feed or U+2028, which end no
    # line of a source, so every line after one was looked up a line late.
    author = tmp_path / "author.py"
    author.write_text(f"x = 1  #{separator}\ny = foo(1)\n", encoding="utf-8")
    (tmp_path / "ws").mkdir()
    checked = tmp_path / "ws" / "m.py"
    checked.write_text(
        check.PRELUDE + f"x = 1  #{separator}\ny = foo(1) or 0\n", encoding="utf-8"
    )
    ws = check._Workspace(tmp_path / "ws", None)
    ws.files["m.py"] = str(author)
    ws.origins["m.py"] = [1, 2]
    line = check._PRELUDE_LINES + 2
    (diagnostic,) = check._parse_output(f"m.py:{line} (5): error: bad\n", ws)
    assert diagnostic.line == 2
    assert diagnostic.column == 5  # "y = " is as written: the column stands
    assert check._source_line(diagnostic) == "y = foo(1)"
    assert check._rewritten_lines(checked, line, str(author), 2) == (
        "y = foo(1) or 0",
        "y = foo(1)",
    )
