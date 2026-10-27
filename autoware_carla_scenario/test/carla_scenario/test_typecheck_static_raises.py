"""What Python raises for that Codon compiles, decided from the syntax (#602, #603)."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest

from autoware_carla_scenario.typecheck.check import Diagnostic, _variadic_hints
from autoware_carla_scenario.typecheck.static_raises import (
    LITERAL_CHECKED,
    literal_argument_refusals,
    subscripted_isinstances,
)

_IMPORTS = """\
import typing
import typing as t
import collections.abc
from typing import List, Optional, Sequence, Union
import autoware_carla_scenario as acs
from autoware_carla_scenario import (
    ElapsedTimeCondition,
    MetricCondition,
    RouteCrossingPose,
    RouteRoadsidePose,
    TimeoutCondition,
)
TYPES = (int, str)
"""


def _tree(body: str) -> ast.Module:
    return ast.parse(_IMPORTS + "def f(v, name):\n" + textwrap.indent(body, "    "))


# ---------------------------------------------------------------------------
# isinstance / issubclass with a subscripted generic (#603)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("call", "generic"),
    [
        ("isinstance(v, list[int])", "list[int]"),
        ("isinstance(v, (int, dict[str, int]))", "dict[str, int]"),
        ("isinstance(v, (int, (str, set[int])))", "set[int]"),
        ("isinstance(v, int | list[int])", "list[int]"),
        ("isinstance(v, List[int])", "List[int]"),
        ("isinstance(v, t.Sequence[int])", "t.Sequence[int]"),
        ("isinstance(v, typing.Dict[str, int])", "typing.Dict[str, int]"),
        ("isinstance(v, collections.abc.Mapping[str, int])", None),
        ("isinstance(v, Sequence[int])", "Sequence[int]"),
        ("isinstance(v, Optional[list[int]])", "list[int]"),
        ("isinstance(v, Union[int, tuple[int, int]])", "tuple[int, int]"),
        ("issubclass(v, dict[str, int])", "dict[str, int]"),
        ("issubclass(v, type[int])", "type[int]"),
    ],
)
def test_a_subscripted_generic_given_to_isinstance_is_refused(
    call: str, generic: str | None
) -> None:
    (found,) = subscripted_isinstances(_tree(f"{call}\n"))
    assert found.line == _IMPORTS.count("\n") + 2
    assert found.column == 5
    assert "Python raises TypeError" in found.message
    function = call.split("(")[0]
    typing_alias = found.message.split("`")[1].split("[")[0] in (
        "List",
        "t.Sequence",
        "typing.Dict",
        "Sequence",
    )
    assert (
        "Subscripted generics cannot be used with class and instance checks"
        if typing_alias
        else f"{function}() argument 2 cannot be a parameterized generic"
    ) in found.message
    if generic is not None:
        assert f"`{generic}`" in found.message
    assert found.message.startswith(f"{function}() cannot take")


def test_the_refusal_names_the_bare_class_to_test() -> None:
    (found,) = subscripted_isinstances(_tree("isinstance(v, list[int])\n"))
    assert "test against the bare class (`list`)" in found.message
    (found,) = subscripted_isinstances(_tree("isinstance(v, typing.Literal[1])\n"))
    assert "compare the value itself" in found.message


@pytest.mark.parametrize(
    "call",
    [
        "isinstance(v, int)",
        "isinstance(v, (int, float))",
        "isinstance(v, int | float)",
        "isinstance(v, int | None)",
        "isinstance(v, List)",
        "isinstance(v, typing.Sequence)",
        "isinstance(v, Optional[int])",
        "isinstance(v, Union[int, str])",
        "isinstance(v, TYPES[0])",
        "isinstance(v, name[0])",
        "issubclass(v, dict)",
        "len(list[int].__args__)",
    ],
)
def test_classes_python_accepts_are_left_alone(call: str) -> None:
    assert subscripted_isinstances(_tree(f"{call}\n")) == []


def test_a_rebound_name_is_not_taken_for_the_builtin() -> None:
    tree = ast.parse(
        "list = {0: int}\n" "def f(v):\n" "    return isinstance(v, list[0])\n"
    )
    assert subscripted_isinstances(tree) == []
    tree = ast.parse(
        "def isinstance(a, b):\n" "    return True\n" "isinstance(1, list[int])\n"
    )
    assert subscripted_isinstances(tree) == []


# ---------------------------------------------------------------------------
# Framework classes built with literal arguments (#602)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            'MetricCondition("no_such_metric", label="m")',
            "MetricCondition: 'no_such_metric' is not a metric",
        ),
        ('ElapsedTimeCondition(1.0, label="")', "label must not be empty"),
        ('ElapsedTimeCondition(-1.0, label="x")', "duration_seconds must be positive"),
        (
            'RouteCrossingPose(approach="Left")',
            "RouteCrossingPose.approach must be one of ['same', 'left', "
            "'right', 'opposite'], got 'Left'",
        ),
        (
            'RouteRoadsidePose(side="kerb")',
            "RouteRoadsidePose.side must be one of ['left', 'right'], got 'kerb'",
        ),
        ('acs.RouteRoadsidePose(side="kerb")', "got 'kerb'"),
        ('acs.RouteLanePose(anchor="")', "RouteLanePose.anchor"),
        ("acs.RouteOppositePose(lane=0)", "RouteOppositePose.lane counts from 1"),
        # An int for a float parameter fits, as Python takes it.
        ('acs.RouteLanePose(ds=1, anchor="nowhere")', "RouteLanePose.anchor"),
        ('acs.RouteCrosswalkPose(leg="middle")', "RouteCrosswalkPose.leg"),
        ('acs.OffRoadFractionCondition(1.5, label="x")', "fraction must be in"),
        ('acs.RouteProgressCondition(tolerance=-1.0, label="x")', "tolerance"),
        ('TimeoutCondition(10.0, label="")', "label must not be empty"),
        ('acs.AlwaysTrueCondition(label="")', "label must not be empty"),
    ],
)
def test_a_literal_call_the_framework_refuses_is_refused(
    call: str, message: str
) -> None:
    (found,) = literal_argument_refusals(_tree(f"x = {call}\n"))
    assert found.line == _IMPORTS.count("\n") + 2
    assert found.column == 9
    assert "raises ValueError when the scenario runs" in found.message
    assert message in found.message, found.message


@pytest.mark.parametrize(
    "call",
    [
        'MetricCondition("offroad", label="m")',
        'ElapsedTimeCondition(1.0, label="t")',
        'RouteCrossingPose(junction=1, approach="right", turn="left")',
        'RouteRoadsidePose(ds=-3.0, side="left")',
        # Not all literals: left to the run.
        "MetricCondition(name, label='m')",
        "ElapsedTimeCondition(1.0, label=name)",
        "RouteRoadsidePose(*name)",
        "RouteRoadsidePose(**name)",
        "ElapsedTimeCondition(-1.0, acs.ComparisonRule.LESS_THAN, label='x')",
        # A type error is Codon's to report, not this pass's.
        "ElapsedTimeCondition('soon', label='x')",
        "ElapsedTimeCondition(1.0, 2.0, 3.0, 4.0, label='x')",
        # A value of the wrong type raises ValueError in the constructor, but
        # Codon reports the mismatch better ("expected a float").
        "acs.RouteProgressCondition('near', label='x')",
        "acs.RouteLanePose(anchor=3)",
        "RouteRoadsidePose(side=1)",
    ],
)
def test_a_call_the_framework_accepts_or_cannot_be_decided_passes(call: str) -> None:
    assert literal_argument_refusals(_tree(f"x = {call}\n")) == []


def test_a_class_not_listed_or_not_the_frameworks_is_not_built() -> None:
    tree = ast.parse(
        "from autoware_carla_scenario import SpeedCondition\n"
        "class RouteRoadsidePose:\n"
        "    def __init__(self, side: str) -> None:\n"
        "        pass\n"
        "RouteRoadsidePose(side='kerb')\n"
        "SpeedCondition(label='')\n"
    )
    assert literal_argument_refusals(tree) == []


def test_a_name_two_imports_bind_is_not_resolved() -> None:
    # Either order of a try/except ImportError fallback: the name is neither.
    for first, second in (
        ("autoware_carla_scenario", "mylib"),
        ("mylib", "autoware_carla_scenario"),
    ):
        tree = ast.parse(
            "try:\n"
            f"    from {first} import RouteRoadsidePose\n"
            "except ImportError:\n"
            f"    from {second} import RouteRoadsidePose\n"
            "RouteRoadsidePose(side='kerb')\n"
        )
        assert literal_argument_refusals(tree) == []
    # Nor is a builtin name an import binds twice taken for the builtin.
    tree = ast.parse(
        "from a import list\nfrom b import list\nisinstance(1, list[int])\n"
    )
    assert subscripted_isinstances(tree) == []


def test_a_framework_call_not_listed_imports_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from autoware_carla_scenario.typecheck import static_raises

    def refuse(*_: object) -> None:
        raise AssertionError("imported a framework module")

    monkeypatch.setattr(static_raises, "_resolve", refuse)
    monkeypatch.setattr(static_raises, "_resolve_unbuilt", refuse)
    monkeypatch.setattr(static_raises, "_checked_classes", refuse)
    tree = ast.parse(
        "import autoware_carla_scenario as acs\n"
        "from autoware_carla_scenario.ros2.bridge import Bridge\n"
        "Bridge('x')\n"
        "acs.ScenarioRunner('x')\n"
        "acs.autoware_bridge.Something(1)\n"
    )
    assert literal_argument_refusals(tree) == []


def test_typesafe_carla_is_not_built_for_the_check(tmp_path: Path) -> None:
    """Where importing typesafe_carla.carla would build it (no prebuilt
    package applies), the conditions are left to the run: the check builds
    nothing.  The poses, which do not import CARLA, are still built."""
    script = textwrap.dedent(
        """
        import ast, os, sys
        from typesafe_carla import carla_build
        carla_build.prebuilt_is_current = lambda *a: False
        builds = []
        def no_build(*_a, **_k):  # stands in for the 15-50 min build
            builds.append(_a)
            raise ImportError("no build in this test")
        carla_build._build = no_build
        from autoware_carla_scenario.typecheck.static_raises import (
            literal_argument_refusals,
        )
        tree = ast.parse(
            "from autoware_carla_scenario import MetricCondition, RouteRoadsidePose\\n"
            "MetricCondition('nope', label='m')\\n"
            "RouteRoadsidePose(side='kerb')\\n"
        )
        found = literal_argument_refusals(tree)
        assert [f.line for f in found] == [3], found
        assert builds == [], "the check started a pycarla build"
        assert "typesafe_carla.carla" not in sys.modules
        assert "TYPESAFE_CARLA_PYCARLA_BUILD" not in os.environ
        print("ok")
        """
    )
    import os
    import subprocess
    import sys

    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("TYPESAFE_CARLA_PYCARLA")
    }
    env["TYPESAFE_CARLA_PYCARLA_DIR"] = str(tmp_path / "pycarla")
    done = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ok"
    assert not (tmp_path / "pycarla").exists()


def test_every_listed_class_is_importable() -> None:
    import importlib

    for dotted in LITERAL_CHECKED:
        module, name = dotted.rsplit(".", 1)
        assert isinstance(getattr(importlib.import_module(module), name), type)


# ---------------------------------------------------------------------------
# Hints for Codon's bare messages (#603)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("message", "hint"),
    [
        ("unsupported class decorator", "choose the class with an if/elif chain"),
        ("async not yet supported", "make it a plain `def`"),
        (
            "'Bag[List[int]]' object has no attribute '__bool__'",
            "`len(x) > 0` for a container",
        ),
        ("'Helper' object has no attribute '__bool__'", "`x is not None`"),
        (
            "'Tuple[int,int]' object has no attribute '__bool__'",
            "Codon gives no tuple a truth value",
        ),
    ],
)
def test_a_bare_codon_message_gets_its_hint(message: str, hint: str) -> None:
    (out,) = _variadic_hints([Diagnostic(message)])
    assert out.message.startswith(message + ": ")
    assert hint in out.message


def test_a_bound_method_truth_is_left_to_its_own_hint() -> None:
    message = "'flag(XScenario)' object has no attribute '__bool__'"
    (out,) = _variadic_hints([Diagnostic(message)])
    assert out.message == message


def test_a_mismatch_at_a_conditional_expression_gets_its_hint(tmp_path: Path) -> None:
    path = tmp_path / "m.py"
    path.write_text("x = 1\nlabel = name if flag else 3\nother: int = name\n")
    message = "'str' does not match expected type 'int'"
    at_ternary, elsewhere, inside = _variadic_hints(
        [
            Diagnostic(message, str(path), 2, 9),
            Diagnostic(message, str(path), 3, 14),
            Diagnostic(message, str(path), 2, 25),
        ]
    )
    assert "a conditional expression (`a if c else b`)" in at_ternary.message
    assert elsewhere.message == message
    assert inside.message == message
