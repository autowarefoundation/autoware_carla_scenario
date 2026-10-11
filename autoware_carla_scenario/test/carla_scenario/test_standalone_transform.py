"""The rewrite that lets Codon compile the framework's own modules.

None of these need Codon: they pin down the source that reaches the compiler,
for the Codon behaviours docs/standalone.md lists.
"""

from __future__ import annotations

import ast
import textwrap

from autoware_carla_scenario.standalone.hierarchy import Hierarchy, ModuleContext
from autoware_carla_scenario.standalone.transform import (
    INSERTED,
    RUNTIME_PRELUDE,
    original_line,
    rewrite_for_runtime,
    rewrite_or,
    transform_for_runtime,
)


def _src(text: str) -> str:
    return textwrap.dedent(text).lstrip("\n")


def _rewrite(text: str, context: ModuleContext | None = None) -> str:
    out = rewrite_for_runtime(_src(text), None, context)
    ast.parse(out)  # still Python
    return out


def _context(modules: dict[str, str], module: str) -> ModuleContext:
    sources = {m: _src(s) for m, s in modules.items()}
    hierarchy = Hierarchy(sources)
    first = {
        m: transform_for_runtime(s, None, hierarchy.base_context(m))
        for m, s in sources.items()
    }
    return hierarchy.context(module, first)


class TestOr:
    def test_an_empty_literal_fallback_takes_the_left_operands_type(self) -> None:
        assert rewrite_or("x = a or []\n") == "x = (_acs_or_empty(a))\n"
        assert rewrite_or("x = a or {}\n") == "x = (_acs_or_empty(a))\n"

    def test_a_condition_spanning_lines_keeps_its_colon_and_line_count(self) -> None:
        source = "if a or (\n    b and c\n):\n    pass\n"
        out = rewrite_or(source)
        compile(out, "<rewritten>", "exec")
        assert out.count("\n") == source.count("\n")

    def test_any_other_fallback_is_evaluated_lazily(self) -> None:
        assert rewrite_or("x = a or f()\n") == "x = (_acs_or(a, lambda: f()))\n"


class TestClassBody:
    def test_a_class_constant_in_a_method_default_is_inlined(self) -> None:
        out = _rewrite(
            """
            class C:
                LIMIT: int = 3

                def f(self, x: int = LIMIT) -> int:
                    return x
            """
        )
        assert "def f(self, x: int = 3)" in out

    def test_a_class_parameter_defaulted_with_or_becomes_a_class_value(self) -> None:
        out = _rewrite(
            """
            class S:
                def __init__(self, kind: type[Base] | None = None) -> None:
                    self.kind = kind or Base
            """
        )
        assert "self.kind = _acs_class(kind, Base)" in out
        assert "kind: _AcsClass[Base]" in out

    def test_an_exception_inherits_statically(self) -> None:
        out = _rewrite(
            """
            class BackendError(RuntimeError):
                pass
            """,
            ModuleContext(exceptions=frozenset({"BackendError"})),
        )
        assert "class BackendError(Static[RuntimeError]):" in out
        assert 'super().__init__("BackendError", message)' in out


class TestAnnotations:
    def test_a_carla_enum_is_an_int(self) -> None:
        out = _rewrite(
            """
            import typesafe_carla.carla as carla

            def f(state: "carla.TrafficLightState", light: carla.TrafficLightState) -> None:
                pass
            """
        )
        assert "def f(state: int, light: int)" in out

    def test_declarations_spell_types_as_codon_does(self) -> None:
        out = _rewrite(
            """
            from typing import IO, Callable, List, Optional, Union

            class C:
                def __init__(self, name: Union[Role, str]) -> None:
                    self._name = name
                    self._out: IO[str] | None = None
                    self._hooks: List[Callable[["carla.World"], None]] = []
            """
        )
        declared = next(line for line in out.splitlines() if line.endswith(INSERTED))
        assert "_name: Union[Role, str]" in declared
        assert "_out: Optional[File]" in declared
        assert "_hooks: List[Callable[[carla.World], None]]" in declared


class TestHierarchy:
    def test_a_class_between_the_definer_and_an_override_forwards(self) -> None:
        modules = {
            "pkg.m": """
                class Base:
                    def get(self, k: int) -> str:
                        return ""

                class Middle(Base):
                    pass

                class Leaf(Middle):
                    def get(self, k: int) -> str:
                        return "leaf"
            """
        }
        ctx = _context(modules, "pkg.m")
        assert ctx.class_lines["Middle"] == [
            "def get(self, k: int) -> str: return super().get(k)"
        ]
        assert "Base" not in ctx.class_lines

    def test_a_class_attribute_is_read_through_a_method(self) -> None:
        modules = {
            "pkg.m": """
                class Ego:
                    use_autopilot: bool = True

                    def __init__(self) -> None:
                        pass

                class Autoware(Ego):
                    use_autopilot: bool = False

                def drive(ego: Ego) -> bool:
                    return ego.use_autopilot
            """
        }
        ctx = _context(modules, "pkg.m")
        assert ctx.defines == {
            "Ego": {"use_autopilot": True},
            "Autoware": {"use_autopilot": False},
        }
        out = _rewrite(modules["pkg.m"], ctx)
        assert "use_autopilot: ClassVar[bool] = True; _acs_use_autopilot: bool" in out
        assert "use_autopilot: ClassVar[bool] = False" in out
        assert "else Autoware.use_autopilot" in out
        assert "return ego._acs_get_use_autopilot()" in out

    def test_a_zero_class_attribute_stays_a_field(self) -> None:
        modules = {
            "pkg.m": """
                class Driven:
                    _backend: object = None
                    _port: int = 0
            """
        }
        assert _context(modules, "pkg.m").attributes == frozenset()

    def test_a_type_checking_import_that_closes_a_cycle_is_dropped(self) -> None:
        modules = {
            "pkg.a": """
                from typing import TYPE_CHECKING
                if TYPE_CHECKING:
                    from .b import B
                class A:
                    pass
            """,
            "pkg.b": """
                from .a import A
                class B(A):
                    pass
            """,
            "pkg.c": """
                from typing import TYPE_CHECKING
                if TYPE_CHECKING:
                    from .a import A
                class C:
                    pass
            """,
        }
        assert _context(modules, "pkg.a").kept == frozenset()
        assert _context(modules, "pkg.c").kept == frozenset({3})


def test_inserted_lines_map_back_to_the_authors_lines() -> None:
    out = transform_for_runtime(
        _src(
            """
            class C:
                def __init__(self, x: int) -> None:
                    self.x = x

                def f(self) -> int:
                    return self.x
            """
        )
    )
    assert out.startswith(RUNTIME_PRELUDE)
    lines = out.splitlines()
    ret = next(i for i, line in enumerate(lines, 1) if "return self.x" in line)
    assert original_line(out, ret) == 6
