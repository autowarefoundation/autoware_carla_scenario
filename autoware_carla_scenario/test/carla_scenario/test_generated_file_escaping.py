"""User text in generated packages: every generated file still parses.

``scenario-new --description`` and the Scenario Editor's ``description`` (and
its map group) are free text that ends up inside ``pyproject.toml`` strings and
``__init__.py`` docstrings.  A quote, a backslash, three double quotes or
``\\N`` used to produce a package that did not parse (issue #680).  These tests
render both generators with hostile values and parse every TOML, Python and
YAML file they write.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest
import yaml

# tomllib landed in 3.11; tomli is the framework's own dependency on 3.10.
if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - only taken on 3.10
    import tomli as tomllib

from autoware_carla_scenario.authoring import package_export
from autoware_carla_scenario.authoring.framework_pin import Pin
from autoware_carla_scenario.authoring.starter import new_document
from autoware_carla_scenario.scaffold import create_scenario_package
from autoware_carla_scenario.templating import py_docstring, toml_string

#: Text that broke a TOML basic string or a Python docstring, one way or another.
HOSTILE = [
    'Ego handles a "cut-in"',
    r"Map at C:\maps\town",
    'The ego says """stop"""',
    r"Map lives in C:\Nishishinjuku",
    r"Escapes \x \u12 \N{BULLET} \0 \\",
    'ends with a quote"',
    "ends with a backslash\\",
    '""""" five quotes',
    "single '''triple''' quotes",
    "two\nlines\twith a tab\r\nand CRLF",
    "control \x00\x07\x1b\x7f characters",
    "{{ jinja }} {% raw %} and unicode: 西新宿 \u2028",
]


def _parse_tree(root: Path) -> None:
    """Parse every generated TOML, Python and YAML file under *root*."""
    seen = {".toml": 0, ".py": 0, ".yaml": 0}
    for path in sorted(root.rglob("*")):
        if path.suffix not in seen or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".toml":
            tomllib.loads(text)
        elif path.suffix == ".py":
            compile(text, str(path), "exec")
        else:
            yaml.safe_load(text)
        seen[path.suffix] += 1
    assert all(seen.values()), seen


@pytest.mark.parametrize("value", HOSTILE)
def test_toml_string_round_trips(value: str) -> None:
    assert tomllib.loads(f"v = {toml_string(value)}")["v"] == value


@pytest.mark.parametrize("value", HOSTILE)
def test_py_docstring_round_trips(value: str) -> None:
    source = f'"""{py_docstring(value)}"""\n'
    tree = ast.parse(source)
    assert ast.get_docstring(tree, clean=False) == value


def test_py_docstring_leaves_prose_readable() -> None:
    assert py_docstring('A "cut-in" scenario.') == 'A "cut-in" scenario.'


@pytest.mark.parametrize("description", HOSTILE)
def test_scaffolded_package_parses(tmp_path: Path, description: str) -> None:
    result = create_scenario_package(
        "my_pkg", output_dir=tmp_path, description=description
    )
    _parse_tree(result.root)
    pyproject = tomllib.loads((result.root / "pyproject.toml").read_text("utf-8"))
    assert pyproject["project"]["description"] == description
    init = (result.root / "src" / "my_pkg" / "__init__.py").read_text("utf-8")
    docstring = ast.get_docstring(ast.parse(init), clean=False)
    assert docstring is not None and docstring.startswith(description + "\n")


@pytest.mark.parametrize("scenario_name", ["null", "true", "on", "no"])
def test_a_scenario_named_like_a_yaml_literal_stays_a_string(
    tmp_path: Path, scenario_name: str
) -> None:
    result = create_scenario_package(
        "my_pkg", scenario_name=scenario_name, output_dir=tmp_path
    )
    (default,) = result.root.rglob("default.yaml")
    assert yaml.safe_load(default.read_text("utf-8"))["scenario"]["name"] == (
        scenario_name
    )


@pytest.mark.parametrize(
    "pin",
    [
        Pin(kind="version", version="1.0.0"),
        Pin(
            kind="git",
            repository='https://example.invalid/"r"\\x.git',
            commit="a" * 40,
            subdirectory="autoware_carla_scenario",
        ),
        Pin(kind="path", path='C:\\Users\\"me"\\autoware_carla_scenario'),
    ],
    ids=["version", "git", "path"],
)
@pytest.mark.parametrize("description", HOSTILE)
def test_exported_package_parses(tmp_path: Path, description: str, pin: Pin) -> None:
    document = new_document()
    document.description = description
    document.map.group = 'odd """group""" \\N'
    names = package_export.package_names(document)
    package_export._write_package_tree(tmp_path, document, names, pin, "0.9.7", [])
    _parse_tree(tmp_path)
    pyproject = tomllib.loads((tmp_path / "pyproject.toml").read_text("utf-8"))
    assert pyproject["project"]["description"] == description
    init = tmp_path / "src" / names["package_name"] / "__init__.py"
    docstring = ast.get_docstring(ast.parse(init.read_text("utf-8")), clean=False)
    assert docstring is not None and docstring.startswith(description + "\n")
    assert f"map={document.map.group}\n" in docstring
    if pin.kind == "git":
        source = pyproject["tool"]["uv"]["sources"][pin.distribution]
        assert source["git"] == pin.repository
        assert source["subdirectory"] == "autoware_carla_scenario"
    if pin.kind == "path":
        assert pyproject["tool"]["uv"]["sources"][pin.distribution]["path"] == pin.path
