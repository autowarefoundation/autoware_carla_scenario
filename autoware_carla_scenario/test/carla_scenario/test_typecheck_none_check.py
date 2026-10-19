"""The None check of the static check: mypy's strict Optional, narrowed (#239).

No Codon needed: these run mypy alone on a module written for the test.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest

from autoware_carla_scenario.typecheck import carla_stubs, none_check
from autoware_carla_scenario.typecheck.none_check import (
    NoneCheckSkipped,
    find_none_misuses,
    mypy_available,
    run_none_check,
)

pytestmark = pytest.mark.skipif(not mypy_available(), reason="no mypy")

_MODULE = """
from __future__ import annotations


def maybe() -> float | None:
    return None


class Pose:
    lanelet_id: int = 0


def pose() -> Pose | None:
    return None


def half(x: float) -> float:
    return x / 2


def misuses() -> None:
    a = maybe() / 3.6
    b = pose().lanelet_id
    c = half(maybe())
    d: float = maybe()


def narrowed() -> float:
    x = maybe()
    if x is None:
        return 0.0
    p = pose()
    assert p is not None
    return x / 3.6 + p.lanelet_id + half(maybe() or 0.0)


def other_errors() -> int:
    y: int = "not None at all"
    return half("text")
"""


def test_only_a_value_that_may_be_none_misused_is_reported(tmp_path: Path) -> None:
    package = tmp_path / "acs_none_case"
    package.mkdir()
    (package / "__init__.py").write_text("")
    module = package / "probe.py"
    module.write_text(textwrap.dedent(_MODULE))

    findings = find_none_misuses([module])

    assert findings is not None
    lines = module.read_text().splitlines()
    found = [(lines[f.line - 1].strip(), f.code) for f in findings]
    assert found == [
        ("a = maybe() / 3.6", "operator"),
        ("b = pose().lanelet_id", "union-attr"),
        ("c = half(maybe())", "arg-type"),
        ("d: float = maybe()", "assignment"),
    ]
    # Also when mypy replays them from its cache, cached at another path.
    assert all(f.path == str(module) for f in findings), findings
    assert findings[0].notes == ('Left operand is of type "float | None"',)


def test_a_finding_mypy_replays_from_its_cache_is_at_the_module_given(
    tmp_path: Path,
) -> None:
    # mypy replays the errors of a module its cache holds unchanged with the
    # path it was cached at: the same package in two places is reported at
    # the second place when it is checked there.
    modules = []
    for place in ("first", "second"):
        package = tmp_path / place / "acs_none_replayed"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("")
        module = package / "probe.py"
        module.write_text(textwrap.dedent(_MODULE))
        modules.append(module)
    for module in modules:
        findings = find_none_misuses([module])
        assert findings, findings
        assert {f.path for f in findings} == {str(module)}


def test_a_module_of_an_installed_package_is_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An installed (not editable) package is under site-packages, whose
    # top-level modules mypy took for user modules shadowing the library
    # ones when the module was given by its path there, and exited with
    # status 2 before checking anything (#281).
    site = tmp_path / "lib" / "python3" / "site-packages"
    site.mkdir(parents=True)
    (site / "typing_extensions.py").write_text("")
    (site / "mypy_extensions.py").write_text("")
    package = site / "acs_none_installed"
    (package / "scenarios").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "scenarios" / "__init__.py").write_text("")
    module = package / "scenarios" / "probe.py"
    module.write_text(textwrap.dedent(_MODULE))
    loose = site / "acs_none_loose.py"
    loose.write_text(textwrap.dedent(_MODULE))

    # Given relative to the working directory, and checked twice: the second
    # time mypy replays the findings from its cache, at the link the first
    # check gave it.
    monkeypatch.chdir(tmp_path)
    relative = module.relative_to(tmp_path)
    for _ in range(2):
        findings = run_none_check([relative, loose])

        assert sorted(f.path for f in findings) == sorted(
            [str(relative)] * 4 + [str(loose)] * 4
        )
        first = next(f for f in findings if f.path == str(relative))
        lines = module.read_text().splitlines()
        assert lines[first.line - 1].strip() == "a = maybe() / 3.6"
        assert first.notes == ('Left operand is of type "float | None"',)


def test_a_link_that_cannot_be_made_gives_no_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "site-packages" / "acs_none_unlinked"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    module = package / "probe.py"
    module.write_text("x = 1\n")

    def refuse(self: Path, target: Path) -> None:
        raise PermissionError("symlinks not permitted")

    monkeypatch.setattr(Path, "symlink_to", refuse)
    with pytest.raises(NoneCheckSkipped, match="could not link .*not permitted"):
        run_none_check([module])


@pytest.mark.parametrize("variable", ["MYPYPATH", "PYTHONPATH"])
def test_a_relative_search_path_entry_is_searched(
    variable: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mypy runs in a directory of its own, so a relative ``MYPYPATH`` or
    ``PYTHONPATH`` entry is made absolute against the one the check runs
    from; otherwise mypy would find nothing there and report nothing."""
    if variable == "MYPYPATH":
        (tmp_path / "found").mkdir()
        (tmp_path / "found" / "helperlib.pyi").write_text(
            "def get() -> int | None: ...\n", encoding="utf-8"
        )
    else:
        package = tmp_path / "found" / "helperlib"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(
            "def get() -> int | None:\n    return None\n", encoding="utf-8"
        )
        (package / "py.typed").write_text("", encoding="utf-8")
    module = tmp_path / "scenario.py"
    module.write_text(
        "import helperlib\n\n\ndef value() -> int:\n    return helperlib.get()\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MYPYPATH", raising=False)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.setenv(variable, "found")
    findings = run_none_check([Path("scenario.py")])
    assert [f.line for f in findings] == [5], findings


def test_no_module_gives_no_verdict() -> None:
    assert find_none_misuses([]) is None


_DECLARATIONS = """
from __future__ import annotations


class Plain:
    def __init__(self) -> None:
        self.count = None

    def later(self) -> None:
        self.count = 5


class Holder:
    value: int = None


def default(speed: float = None) -> float:
    return 1.0


def maybe() -> float | None:
    return None


def misuse() -> float:
    return maybe() * 2
"""


def test_an_assignment_that_only_declares_is_no_misuse(tmp_path: Path) -> None:
    # A None default or value for a non-Optional annotation, and a value for
    # an attribute mypy inferred as None, use no value that may be None.
    package = tmp_path / "acs_none_declarations"
    package.mkdir()
    (package / "__init__.py").write_text("")
    module = package / "probe.py"
    module.write_text(textwrap.dedent(_DECLARATIONS))

    findings = find_none_misuses([module])

    assert findings is not None
    lines = module.read_text().splitlines()
    assert [lines[f.line - 1].strip() for f in findings] == ["return maybe() * 2"]


def test_an_unwritable_cache_directory_runs_mypy_without_a_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # mypy fails outright on a cache directory it cannot create (a
    # read-only home in a container); the check runs it without one.
    blocker = tmp_path / "a-file"
    blocker.write_text("")
    monkeypatch.setenv("XDG_CACHE_HOME", str(blocker / "cache"))
    assert none_check._cache_dir() == os.devnull
    package = tmp_path / "acs_none_no_cache"
    package.mkdir()
    (package / "__init__.py").write_text("")
    module = package / "probe.py"
    module.write_text(textwrap.dedent(_MODULE))

    assert len(run_none_check([module])) == 4


def test_no_verdict_from_mypy_says_why(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = tmp_path / "probe.py"
    module.write_text("x = 1\n")
    broken = tmp_path / "broken.py"
    broken.write_text("def f(:\n")
    with pytest.raises(NoneCheckSkipped, match=r"exit status 2\): .*Invalid syntax"):
        run_none_check([broken])
    with pytest.raises(NoneCheckSkipped, match="no module"):
        run_none_check([])
    monkeypatch.setattr(none_check, "mypy_available", lambda: False)
    with pytest.raises(NoneCheckSkipped, match="not installed"):
        run_none_check([module])


@pytest.mark.parametrize(
    ("code", "message", "misuse"),
    [
        ("union-attr", 'Item "None" of "Pose | None" has no attribute "x"', True),
        (
            "operator",
            'Unsupported operand types for / ("None" and "float")',
            True,
        ),
        (
            "arg-type",
            'Argument 1 to "half" has incompatible type "float | None"; '
            'expected "float"',
            True,
        ),
        (
            "assignment",
            'Incompatible types in assignment (expression has type "float | None", '
            'variable has type "float")',
            True,
        ),
        (
            # An item assigned: d["k"] = maybe().
            "assignment",
            'Incompatible types in assignment (expression has type "float | None", '
            'target has type "float")',
            True,
        ),
        (
            "return-value",
            'Incompatible return value type (got "Optional[int]", expected "int")',
            True,
        ),
        ("index", 'Value of type "list[int] | None" is not indexable', True),
        (
            "call-overload",
            'No overload variant of "round" matches argument type "float | None"',
            True,
        ),
        # None only inside another type, or in the type expected (#267).
        (
            "arg-type",
            'Argument 2 to "register_measure" of "BaseScenario" has incompatible '
            'type "Callable[[Any], str | None]"; expected '
            '"Callable[[Any], float | None]"',
            False,
        ),
        (
            "arg-type",
            'Argument 1 to "f" has incompatible type "str | None"; '
            'expected "float | None"',
            False,
        ),
        (
            "arg-type",
            'Argument "role" to "f" has incompatible type "int"; '
            'expected "EntityRole | str | None"',
            False,
        ),
        (
            "assignment",
            'Incompatible types in assignment (expression has type "list[float]", '
            'variable has type "list[float | None]")',
            False,
        ),
        (
            "return-value",
            'Incompatible return value type (got "dict[str, None]", '
            'expected "dict[str, int]")',
            False,
        ),
        ("attr-defined", '"None" has no attribute "x"', False),
    ],
)
def test_only_none_as_the_offending_type_is_a_misuse(
    code: str, message: str, misuse: bool
) -> None:
    assert none_check.is_none_misuse(code, message) is misuse


_MENTIONS = """
from __future__ import annotations

from typing import Callable, Optional


def register(read: Callable[[int], float | None]) -> None:
    pass


def name(world: int) -> Optional[str]:
    return "left"


def labelled(label: str | None = None) -> None:
    pass


def mentions() -> None:
    register(name)
    labelled(3)
    x: list[float | None] = [1.0]
    y: list[float] = [1.0]
    x = y


def misuse() -> float:
    return name(1) + 1.0


def reading() -> float | None:
    return None


def item(speeds: dict[str, float]) -> None:
    speeds["ego"] = reading()
"""


def test_an_error_merely_mentioning_none_is_no_misuse(tmp_path: Path) -> None:
    # A str where a float is expected, an int for an Optional[str], a list
    # invariant in its items: mypy's messages name None, the value used is
    # never None (#267).
    package = tmp_path / "acs_none_mentions"
    package.mkdir()
    (package / "__init__.py").write_text("")
    module = package / "probe.py"
    module.write_text(textwrap.dedent(_MENTIONS))

    findings = find_none_misuses([module])

    assert findings is not None
    lines = module.read_text().splitlines()
    assert [lines[f.line - 1].strip() for f in findings] == [
        "return name(1) + 1.0",
        'speeds["ego"] = reading()',
    ], findings


_CARLA = """
from __future__ import annotations

import typesafe_carla.carla as carla


def lane_yaw(world: carla.World, location: carla.Location) -> float:
    wp = world.get_map().get_waypoint(location, project_to_road=False)
    return wp.transform.rotation.yaw


def actor_x(world: carla.World, actor_id: int) -> float:
    return world.get_actors().find(actor_id).get_location().x


def parent_id(actor: carla.Actor) -> int:
    return actor.parent.id


def projected(world: carla.World, location: carla.Location, flag: bool) -> float:
    # Projected onto the road (the default), it is None only on a map with
    # no driving lane at all: not a misuse; a flag mypy cannot read is one.
    m = world.get_map()
    x = m.get_waypoint(location).transform.location.x
    x += m.get_waypoint(location, True, carla.LaneType.Driving).s
    return x + m.get_waypoint(location, project_to_road=flag).s


def guarded(world: carla.World, location: carla.Location) -> float:
    wp = world.get_map().get_waypoint(location)
    if wp is None:
        return 0.0
    loc = wp.transform.location + carla.Location(x=1.0)
    for actor in world.get_actors():
        loc.x += actor.get_location().x
    vehicles = world.get_actors().filter("vehicle.*")
    settings = world.get_settings()
    settings.synchronous_mode = True
    world.set_weather(carla.WeatherParameters.ClearNoon)
    state = carla.TrafficLightState.Red
    return loc.x + vehicles[0].get_velocity().length() + float(state == state)
"""


def _carla_module(tmp_path: Path, name: str) -> Path:
    package = tmp_path / name
    package.mkdir()
    (package / "__init__.py").write_text("")
    module = package / "probe.py"
    module.write_text(textwrap.dedent(_CARLA))
    return module


def test_an_optional_carla_result_used_unguarded_is_a_misuse(tmp_path: Path) -> None:
    # typesafe_carla's CPython package has no type information: the check
    # gives mypy stubs of its Codon library, so a CARLA call that returns an
    # Optional returns one to mypy too, and nothing else is reported (#267).
    module = _carla_module(tmp_path, "acs_none_carla")

    findings = find_none_misuses([module])

    assert findings is not None
    lines = module.read_text().splitlines()
    assert [(lines[f.line - 1].strip(), f.message) for f in findings] == [
        (
            "return wp.transform.rotation.yaw",
            'Item "None" of "Waypoint | None" has no attribute "transform"',
        ),
        (
            "return world.get_actors().find(actor_id).get_location().x",
            'Item "None" of "Actor | None" has no attribute "get_location"',
        ),
        (
            "return actor.parent.id",
            'Item "None" of "Actor | None" has no attribute "id"',
        ),
        (
            "return x + m.get_waypoint(location, project_to_road=flag).s",
            'Item "None" of "Waypoint | None" has no attribute "s"',
        ),
    ]


def test_the_carla_stubs_are_written_where_the_cache_cannot_be(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocker = tmp_path / "a-file"
    blocker.write_text("")
    monkeypatch.setenv("XDG_CACHE_HOME", str(blocker / "cache"))
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    assert carla_stubs.carla_stub_root(scratch) == scratch
    stub = scratch / "typesafe_carla" / "carla" / "__init__.pyi"
    assert "def get_actor(self, *args: Any, **kwargs: Any) -> Optional[Actor]" in (
        stub.read_text()
    )
    module = _carla_module(tmp_path, "acs_none_carla_uncached")
    assert len(run_none_check([module])) == 4


def test_without_typesafe_carlas_library_the_check_runs_untyped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing() -> Path:
        raise RuntimeError("typesafe_carla Codon sources not found")

    monkeypatch.setattr(carla_stubs, "_library_dir", missing)
    assert carla_stubs.carla_stub_root(tmp_path) is None
    module = _carla_module(tmp_path, "acs_none_carla_untyped")
    assert run_none_check([module]) == []


def test_the_checked_framework_modules_misuse_no_none(tmp_path: Path) -> None:
    # The library check runs the None check over the modules it compiles, as
    # the scenario check does (#277); without Codon it does not run, so the
    # framework's own modules are checked here too.  A CARLA Optional read
    # without a test (CollisionEvent.other_actor) was missed by both.
    from autoware_carla_scenario.typecheck.library import CHECKED, package_modules

    # Typed CARLA, or CollisionEvent.other_actor would be Any to mypy.
    assert carla_stubs.carla_stub_root(tmp_path) is not None
    every = package_modules()
    findings = run_none_check(sorted(every[name] for name in CHECKED), timeout=1200)

    assert findings == [], "\n".join(
        f"{f.path}:{f.line}:{f.column}: {f.message} [{f.code}]" for f in findings
    )
