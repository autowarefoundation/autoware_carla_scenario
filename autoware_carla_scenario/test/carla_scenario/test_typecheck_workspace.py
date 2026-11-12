"""The checker's workspace: built for the toolchain it is given (#2091).

None of these runs Codon: the toolchain is a directory laid out like one.
"""

from __future__ import annotations

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
