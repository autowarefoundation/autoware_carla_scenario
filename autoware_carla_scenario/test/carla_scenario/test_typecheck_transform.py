"""The source rewrite and the checks the static check makes before Codon runs.

None of these need Codon: they pin down what reaches the compiler.
"""

from __future__ import annotations

import ast
import dataclasses
import enum
import textwrap
from typing import Any, Optional

import pytest

from autoware_carla_scenario.typecheck.driver import render_driver
from autoware_carla_scenario.typecheck.transform import (
    PRELUDE,
    codon_annotation,
    transform_source,
    undeclared_attributes,
)


def _transform(source: str) -> str:
    out = transform_source(textwrap.dedent(source)).source
    assert out.startswith(PRELUDE)
    return out[len(PRELUDE) :]


def _annotation(text: str, *, class_level: bool = False) -> str | None:
    return codon_annotation(ast.parse(text, mode="eval").body, class_level=class_level)


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("python", "codon"),
    [
        ("float | None", "Optional[float]"),
        ("None | Lanelet2Pose", "Optional[Lanelet2Pose]"),
        ("Optional[int]", "Optional[int]"),
        ("Union[str, None]", "Optional[str]"),
        ("list[int]", "list[int]"),
        ("dict[str, list[float | None]]", "dict[str, list[Optional[float]]]"),
        ("'carla.World'", "carla.World"),
        ("tuple[int, str]", "tuple[int, str]"),
        ("ClassVar[int]", "ClassVar[int]"),
    ],
)
def test_an_annotation_codon_can_express_is_kept(python: str, codon: str) -> None:
    assert _annotation(python) == codon


@pytest.mark.parametrize(
    "python",
    [
        "Union[EntityRole, str]",  # crashes Codon 0.19
        "EntityRole | str",
        "Any",
        "Callable[[int], None]",
        "type[BaseScenario]",
        "tuple[int, ...]",
        "Sequence[int]",  # a caller may pass a tuple
        "NDArray[np.uint8]",
    ],
)
def test_an_annotation_codon_cannot_express_is_dropped(python: str) -> None:
    assert _annotation(python) is None


def test_a_class_level_declaration_keeps_an_abstract_collection() -> None:
    # The typing shim maps it onto List; a declaration cannot be dropped.
    assert _annotation("Sequence[int]", class_level=True) == "Sequence[int]"


def test_parameters_and_returns_are_rewritten_in_place() -> None:
    out = _transform(
        """
        def f(a: float | None, b: Union[int, str], *, c: Any = 1) -> int | str:
            x: Any = 3
            return a
        """
    )
    assert "def f(a: Optional[float], b, *_acs_kw, c = 1):" in out
    assert "    x = 3" in out


def test_every_line_stays_where_it_was() -> None:
    source = textwrap.dedent(
        """
        from dataclasses import dataclass, field

        @dataclass(
            frozen=True,
        )
        class Config:
            ids: list[int] = field(
                default_factory=lambda: [460, 265]
            )
            npcs: list[NpcVehicleConfig] = field(default_factory=list)
            ground: GroundProjectionConfig = field(default_factory=GroundProjectionConfig)
            limit: float | None = field(default=None)
            later: int = 0  # marker
        """
    )
    out = _transform(source)
    assert out.count("\n") == source.count("\n")
    lines = out.splitlines()
    assert lines[source.splitlines().index("    later: int = 0  # marker")] == (
        "    later: int = 0  # marker"
    )
    assert "dataclass(" not in out.replace("import dataclass", "")
    assert "    ids: list[int] = ([460, 265])" in out
    assert "    npcs: list[NpcVehicleConfig] = []" in out
    assert "    ground: GroundProjectionConfig = GroundProjectionConfig()" in out
    assert "    limit: Optional[float] = None" in out


def test_a_list_of_calls_or_names_becomes_an_acs_list() -> None:
    out = _transform(
        """
        a = AndCondition([TimeoutCondition(1.0, label="t"), SpeedCondition(x, label="s")])
        b = AndCondition([stopped, restarted])
        c = [460, 265]
        d = [only(1)]
        """
    )
    assert (
        'AndCondition(_acs_list(TimeoutCondition(1.0, label="t"), SpeedCondition(x, label="s")))'
        in out
    )
    assert "AndCondition(_acs_list(stopped, restarted))" in out
    assert "c = [460, 265]" in out
    assert "d = [only(1)]" in out
    assert transform_source("x = [f(), g()]\n").uses_acs_list


# ---------------------------------------------------------------------------
# Attribute declarations
# ---------------------------------------------------------------------------


_SCENARIO = """
class MyScenario(BaseScenario):
    _declared: int

    def __init__(self, config: Cfg | None = None) -> None:
        super().__init__()
        self._config = config or Cfg()
        self._declared = 1
        self.ego_config = None
        self._typed: float = 2.0
"""


def test_an_undeclared_attribute_of_a_subclass_is_reported_with_a_type() -> None:
    found = undeclared_attributes(
        ast.parse(_SCENARIO), {"BaseScenario": {"ego_config"}}
    )
    assert [(f.name, f.suggested_type) for f in found] == [
        ("_config", "Cfg"),
        ("_typed", "float"),
    ]
    assert found[0].lineno == 7
    assert "`_config: Cfg`" in found[0].message()


def test_a_class_outside_any_hierarchy_may_infer_its_attributes() -> None:
    source = "class Helper:\n    def __init__(self):\n        self.x = 1\n"
    assert undeclared_attributes(ast.parse(source), {}) == []


def test_a_declaration_on_a_base_in_the_same_module_counts() -> None:
    source = textwrap.dedent(
        """
        class Base:
            x: int
        class Child(Base):
            def __init__(self):
                self.x = 1
        """
    )
    assert undeclared_attributes(ast.parse(source), {}) == []


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _Npc:
    spawn_lanelet_id: int = 0
    spawn_s: float = 0.0


class _Turn(enum.Enum):
    LEFT = "left"


@dataclasses.dataclass
class _Config:
    timeout_seconds: float = 5.0
    min_speed_kmh: Optional[float] = None
    ids: list[int] = dataclasses.field(default_factory=list)
    npcs: list[_Npc] = dataclasses.field(default_factory=list)
    turn: _Turn = _Turn.LEFT


class _Scenario:
    pass


def test_the_driver_renders_the_yaml_values_with_the_config_types() -> None:
    driver = render_driver(
        _Scenario,
        _Config,
        {
            "timeout_seconds": 5,
            "min_speed_kmh": 3,
            "ids": [460, 265],
            "npcs": [{"spawn_lanelet_id": 1, "spawn_s": 2}],
            "turn": _Turn.LEFT,
        },
    )
    lines = driver.source.splitlines()
    by_key = {key: lines[line - 1].strip() for line, key in driver.config_lines.items()}
    assert by_key == {
        "scenario.timeout_seconds": "config.timeout_seconds = 5.0",
        "scenario.min_speed_kmh": "config.min_speed_kmh = 3.0",
        "scenario.ids": "config.ids = [460, 265]",
        "scenario.npcs": "config.npcs = [_Npc(spawn_lanelet_id=1, spawn_s=2.0)]",
        "scenario.turn": "config.turn = _Turn.LEFT",
    }
    assert f"from {__name__} import _Npc" in lines
    assert "    scenario.setup()" in lines


def test_a_config_with_a_mandatory_field_is_built_by_its_constructor() -> None:
    @dataclasses.dataclass
    class Mandatory:
        lanelet_id: int

    driver = render_driver(_Scenario, Mandatory, {"lanelet_id": 3})
    assert "        lanelet_id=3," in driver.source.splitlines()


# ---------------------------------------------------------------------------
# When the runner checks: the typecheck mode
# ---------------------------------------------------------------------------


class _ModeScenario:
    pass


def test_the_mode_comes_from_the_config_then_defaults_to_auto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from autoware_carla_scenario.typecheck.mode import TYPECHECK_ENV, typecheck_mode

    monkeypatch.delenv(TYPECHECK_ENV, raising=False)
    assert typecheck_mode({}) == "auto"
    assert typecheck_mode({"typecheck": "required"}) == "required"
    assert typecheck_mode({"typecheck": False}) == "off"
    monkeypatch.setenv(TYPECHECK_ENV, "off")
    assert typecheck_mode({"typecheck": "required"}) == "off"
    monkeypatch.setenv(TYPECHECK_ENV, "sometimes")
    with pytest.raises(ValueError, match="typecheck must be one of"):
        typecheck_mode({})


def test_without_codon_auto_warns_and_required_refuses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, caplog: pytest.LogCaptureFixture
) -> None:
    from autoware_carla_scenario.registry import register_scenario, unregister_scenario
    from autoware_carla_scenario.typecheck import ENV_CODON, ScenarioTypeError
    from autoware_carla_scenario.typecheck.mode import check_registered_scenario

    monkeypatch.setenv(ENV_CODON, str(tmp_path / "no-such-codon"))
    register_scenario("typecheck_mode_case", _ModeScenario, _Config)  # type: ignore[arg-type]
    try:
        assert check_registered_scenario("typecheck_mode_case", {}, "off") is None
        result = check_registered_scenario("typecheck_mode_case", {}, "auto")
        assert result is not None and result.skipped is not None
        assert "static check skipped" in caplog.text
        with pytest.raises(ScenarioTypeError, match="no Codon compiler"):
            check_registered_scenario("typecheck_mode_case", {}, "required")
    finally:
        unregister_scenario("typecheck_mode_case")


def test_a_scenario_with_a_custom_builder_is_not_checked() -> None:
    from autoware_carla_scenario.registry import (
        get_scenario_classes,
        register_scenario,
        register_scenario_builder,
        unregister_scenario,
    )
    from autoware_carla_scenario.typecheck.mode import check_registered_scenario

    register_scenario("typecheck_builder_case", _ModeScenario, _Config)  # type: ignore[arg-type]
    assert get_scenario_classes("typecheck_builder_case") == (_ModeScenario, _Config)
    register_scenario_builder("typecheck_builder_case", lambda *args: None)  # type: ignore[arg-type,return-value]
    try:
        assert get_scenario_classes("typecheck_builder_case") is None
        assert (
            check_registered_scenario("typecheck_builder_case", {}, "required") is None
        )
    finally:
        unregister_scenario("typecheck_builder_case")
    assert get_scenario_classes("typecheck_builder_case") is None
