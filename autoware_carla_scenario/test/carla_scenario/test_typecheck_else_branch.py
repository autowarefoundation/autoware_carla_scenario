"""The `else` branch of `if TYPE_CHECKING:` repeats the call its then branch checks.

Codon reads ``TYPE_CHECKING`` as a compile-time true, so it compiles only the
then branch. Where that branch calls a method through its class because of
the dispatch caveat (``Class.method(obj, ...)``, docs/typecheck.md) and the
``else`` branch, which only Python runs, makes the same call through the
instance (``obj.method(...)``), a wrong argument in the ``else`` copy would
pass the check. This test pins the two copies to the same arguments, the
receiver aside. No Codon needed: it reads the source.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

import pytest

from autoware_carla_scenario.typecheck.library import CHECKED, package_modules

#: Every then/else pair the test finds today, by (module, method). ``traffic.driven``'s
#: ``turn_at_junction()`` is not one: its then branch calls no method through
#: a class (it stops at ``_resolve_backend()``, since a dispatch thunk cannot
#: forward ``**kwargs``), so there is no checked copy to compare with.
EXPECTED_SITES: frozenset[tuple[str, str]] = frozenset(
    {
        ("autoware_carla_scenario.actions.background_traffic", "spawn_background"),
        ("autoware_carla_scenario.actions.turn", "turn_at_junction"),
        ("autoware_carla_scenario.scenario_runner", "start"),
    }
)


@dataclass(frozen=True)
class _Pair:
    module: str
    line: int
    method: str
    then_call: ast.Call
    else_call: ast.Call

    @property
    def site(self) -> tuple[str, str]:
        return (self.module, self.method)

    def __str__(self) -> str:
        return f"{self.module}:{self.line} {self.method}()"


def _is_type_checking(test: ast.expr) -> bool:
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return (
        isinstance(test, ast.Attribute)
        and test.attr == "TYPE_CHECKING"
        and isinstance(test.value, ast.Name)
        and test.value.id == "typing"
    )


def _method_calls(stmts: list[ast.stmt]) -> list[ast.Call]:
    return [
        node
        for stmt in stmts
        for node in ast.walk(stmt)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]


def _through_class(call: ast.Call) -> bool:
    """``Class.method(...)`` or ``module.Class.method(...)``: the receiver is
    a capitalised name, bare or the last part of a dotted one."""
    assert isinstance(call.func, ast.Attribute)
    receiver = call.func.value
    if isinstance(receiver, ast.Name):
        return receiver.id[:1].isupper()
    return isinstance(receiver, ast.Attribute) and receiver.attr[:1].isupper()


def _keywords(call: ast.Call) -> tuple[dict[str, str], list[str]]:
    """The named keyword arguments, and the ``**`` unpackings in order."""
    named = {kw.arg: ast.dump(kw.value) for kw in call.keywords if kw.arg is not None}
    unpacked = [ast.dump(kw.value) for kw in call.keywords if kw.arg is None]
    return named, unpacked


def _pairs() -> list[_Pair]:
    modules = package_modules()
    pairs: list[_Pair] = []
    for name in CHECKED:
        tree = ast.parse(modules[name].read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.If) and _is_type_checking(node.test)):
                continue
            through_class = [
                call
                for call in _method_calls(node.body)
                if _through_class(call) and call.args
            ]
            through_instance = [
                call for call in _method_calls(node.orelse) if not _through_class(call)
            ]
            for then_call in through_class:
                assert isinstance(then_call.func, ast.Attribute)
                method = then_call.func.attr
                for else_call in through_instance:
                    assert isinstance(else_call.func, ast.Attribute)
                    if else_call.func.attr == method:
                        pairs.append(
                            _Pair(name, node.lineno, method, then_call, else_call)
                        )
    return pairs


PAIRS = _pairs()


def test_the_pairs_are_the_expected_ones() -> None:
    found = {pair.site for pair in PAIRS}
    assert found == EXPECTED_SITES, (
        "The `if TYPE_CHECKING:` then/else pairs that call one method through "
        "its class and through an instance changed: "
        f"new {sorted(found - EXPECTED_SITES)}, "
        f"gone {sorted(EXPECTED_SITES - found)}. "
        "Update EXPECTED_SITES in this test if that was intended."
    )
    assert len(PAIRS) == len(EXPECTED_SITES), [str(pair) for pair in PAIRS]


@pytest.mark.parametrize("pair", PAIRS, ids=str)
def test_else_branch_passes_the_checked_arguments(pair: _Pair) -> None:
    then_args = [ast.dump(arg) for arg in pair.then_call.args[1:]]
    else_args = [ast.dump(arg) for arg in pair.else_call.args]
    assert else_args == then_args, (
        f"{pair}: positional arguments differ: checked "
        f"{ast.unparse(pair.then_call)!r}, run {ast.unparse(pair.else_call)!r}"
    )
    assert _keywords(pair.else_call) == _keywords(pair.then_call), (
        f"{pair}: keyword arguments differ: checked "
        f"{ast.unparse(pair.then_call)!r}, run {ast.unparse(pair.else_call)!r}"
    )
