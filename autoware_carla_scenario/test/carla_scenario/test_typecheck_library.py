"""The library check: the framework's own modules, compiled with Codon.

The manifest (``typecheck/library.py``) and the source the check generates
are tested without Codon; the compiles are skipped where no Codon compiler is
installed (typesafe-carla's toolchain, a dependency on Linux x86_64 and
aarch64).
"""

from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from autoware_carla_scenario.typecheck import (
    TypeCheckResult,
    available_toolchain,
    check,
    typecheck_library,
)
from autoware_carla_scenario.typecheck import model_dir
from autoware_carla_scenario.typecheck.check import (
    _model_module,
    _uncalled,
    replaced_models,
)
from autoware_carla_scenario.typecheck.cli import main as scenario_check
from autoware_carla_scenario.typecheck.library import (
    CHECKED,
    EXCLUDED,
    MODELLED_IMPORTS,
    NOT_YET_CHECKED,
    REPLACED_MODELS,
    UNCALLED,
    package_modules,
)
from autoware_carla_scenario.typecheck.library_driver import (
    call_sites,
    render_library_checks,
    render_library_driver,
    workspace_module,
)
from autoware_carla_scenario.typecheck.transform import redirect_imports

_PACKAGE = "autoware_carla_scenario"

pytestmark = pytest.mark.codon

needs_codon = pytest.mark.skipif(
    available_toolchain() is None, reason="no Codon compiler"
)


# ---------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------


def test_every_module_is_either_checked_or_excluded() -> None:
    modules = set(package_modules())
    assert len(CHECKED) == len(set(CHECKED)), "a module is listed twice in CHECKED"
    both = set(CHECKED) & set(EXCLUDED)
    assert not both, f"in both CHECKED and EXCLUDED: {sorted(both)}"
    missing = modules - set(CHECKED) - set(EXCLUDED)
    assert not missing, (
        f"not in typecheck/library.py: {sorted(missing)}; add each to EXCLUDED "
        f"({NOT_YET_CHECKED!r}) or, if it compiles, to CHECKED"
    )
    unknown = (set(CHECKED) | set(EXCLUDED)) - modules
    assert (
        not unknown
    ), f"listed in typecheck/library.py but not modules: {sorted(unknown)}"


def test_every_excluded_module_has_a_reason() -> None:
    assert all(reason.strip() for reason in EXCLUDED.values())


def _shim(module: str) -> str | None:
    """The source of the standard module's shim in the model, if it has one.

    A dotted module is a submodule's shim (``importlib.metadata`` is
    ``importlib/metadata.codon``).
    """
    stem = module.replace(".", "/")
    for path in (
        model_dir() / f"{stem}.codon",
        model_dir() / stem / "__init__.codon",
    ):
        if path.is_file():
            return path.read_text(encoding="utf-8")
    return None


def _declares(shim: str, name: str) -> bool:
    """Whether the shim declares ``name`` at module level.

    A method (``Logger.exception()``) is not the module's function of that
    name (``logging.exception()``).
    """
    return (
        re.search(
            rf"^(?:def|class)\s+{re.escape(name)}\b|^{re.escape(name)}\s*[:=]",
            shim,
            re.MULTILINE,
        )
        is not None
    )


def _dotted(node: ast.expr) -> str | None:
    """``a.b.c`` for a chain of attributes of a name, else ``None``."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        owner = _dotted(node.value)
        return None if owner is None else f"{owner}.{node.attr}"
    return None


def _shimmed_uses(source: str) -> dict[str, set[str]]:
    """The names a module takes from each standard module with a shim.

    ``import X`` counts the ``X.name`` it reads, ``from X import name`` the
    names, a submodule's (``import X.Y``, ``from X.Y import name``) included;
    a method of a class (``Path.rename()``) is not seen.
    """
    tree = ast.parse(source)
    aliases: dict[str, str] = {}
    uses: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _shim(alias.name) is not None:
                    aliases[alias.asname or alias.name] = alias.name
                    uses.setdefault(alias.name, set())
        elif (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module
            and _shim(node.module) is not None
        ):
            names = uses.setdefault(node.module, set())
            for alias in node.names:
                submodule = f"{node.module}.{alias.name}"
                if _shim(submodule) is None:
                    names.add(alias.name)
                else:
                    # `from lxml import etree`: what is read of etree is
                    # taken from lxml.etree.
                    aliases[alias.asname or alias.name] = submodule
                    uses.setdefault(submodule, set())
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            owner = _dotted(node.value)
            if owner in aliases:
                uses[aliases[owner]].add(node.attr)
    return uses


def test_an_excluded_reason_names_no_api_its_shim_declares() -> None:
    # docs/typecheck.md: "A reason names `pathlib` only where the module uses
    # more of it than the shim declares, and the other shimmed modules
    # likewise" (#300). A reason names neither a shimmed module the module
    # uses only what the shim declares of, nor (`os.kill()`) a name the shim
    # declares.
    modules = package_modules()
    wrong = []
    for module, reason in EXCLUDED.items():
        for name, attr in re.findall(r"`([\w.]+)\.(\w+)(?:\(\))?`", reason):
            shim = _shim(name)
            if shim is not None and _declares(shim, attr):
                wrong.append(f"{module}: `{name}.{attr}` is declared by its shim")
        # "more of X than its shim declares" names X for what the shim lacks.
        bare = re.sub(r"\bof [\w., ]+? than (?:its|their) shims? declares?", "", reason)
        # The reason itself names the module's own imports; "slurm.manifest
        # (json)" names those of the package module it gives them for.
        named = [(module, re.sub(r"\([^()]*\)", "", bare))]
        for other, imports in re.findall(
            r"(?<![\w.])([a-z_][\w.]*) \(([^()]*)\)", bare
        ):
            if f"autoware_carla_scenario.{other}" in modules:
                named.append((f"autoware_carla_scenario.{other}", imports))
        for importer, text in named:
            source = modules[importer].read_text(encoding="utf-8")
            for name, names in _shimmed_uses(source).items():
                shim = _shim(name)
                assert shim is not None
                if all(_declares(shim, n) for n in names) and re.search(
                    rf"(?<![\w.`]){re.escape(name)}(?![\w.(`])", text
                ):
                    wrong.append(
                        f"{module}: names {name} for {importer}, but its shim "
                        f"declares all {importer} uses of it "
                        f"({', '.join(sorted(names))})"
                    )
    assert not wrong, "\n".join(wrong)


_DOCS = Path(__file__).resolve().parents[2] / "docs" / "typecheck.md"


def test_the_docs_count_the_checked_modules_and_the_uncalled_functions() -> None:
    # docs/typecheck.md's "What is checked of the framework" counts what this
    # manifest lists; computed here, so they cannot drift again (#240).
    text = " ".join(_DOCS.read_text(encoding="utf-8").split())
    modules = len(package_modules())
    percent = round(100 * len(CHECKED) / modules)
    assert (
        f"{len(CHECKED)} of the package's {modules} modules ({percent}%;" in text
    ), "update the module counts in docs/typecheck.md"
    protected = [
        name
        for name in UNCALLED
        if name.rsplit(".", 1)[-1].startswith("_")
        and not name.rsplit(".", 1)[-1].startswith("__")
    ]
    assert (
        f"{len(UNCALLED)} functions and methods of those modules are `UNCALLED` "
        f"({len(protected)} of them protected helpers" in text
    ), "update the UNCALLED counts in docs/typecheck.md"
    # The breakdown of EXCLUDED: its total, and its groups adding up to it (#269).
    total = f"The {len(EXCLUDED)} `EXCLUDED` modules fall in"
    assert total in text, "update the EXCLUDED count in docs/typecheck.md"
    breakdown = text[
        text.index(total) : text.index("The scenario check is not limited")
    ]
    groups = [int(n) for n in re.findall(r"- \*\*(\d+) ", breakdown)]
    assert len(groups) == 3 and sum(groups) == len(EXCLUDED), (
        f"the EXCLUDED groups in docs/typecheck.md add up to {sum(groups)}, "
        f"not {len(EXCLUDED)}"
    )


def test_every_uncalled_function_is_one_of_a_checked_module() -> None:
    modules = package_modules()
    for name, reason in UNCALLED.items():
        assert reason.strip(), f"{name}: no reason"
        # The deepest one: traffic.base.X is in traffic.base, not
        # in traffic, and trajectory.model.X in trajectory.model.
        owners = [m for m in CHECKED if name.startswith(f"{m}.")]
        assert owners, f"{name}: not in a checked module"
        module = max(owners, key=len)
        qualname = name.removeprefix(f"{module}.")
        tree = ast.parse(modules[module].read_text(encoding="utf-8"))
        owner, _, function = qualname.rpartition(".")
        body = tree.body
        if owner:
            classes = [
                n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner
            ]
            assert classes, f"{name}: no class {owner}"
            body = classes[0].body
        if owner and function == "__init__":
            continue  # a generated constructor (a dataclass) has no definition
        assert [
            n
            for n in body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name == function
        ], f"{name}: no such function"


def test_every_modelled_import_is_between_checked_modules() -> None:
    for name, imported in MODELLED_IMPORTS.items():
        assert name in CHECKED, name
        assert imported and all(m in CHECKED for m in imported), name


def test_the_checker_itself_is_not_a_module_to_check() -> None:
    assert not [m for m in package_modules() if m.startswith(f"{_PACKAGE}.typecheck")]


# ---------------------------------------------------------------------------
# Imports of the framework, pointed into the workspace
# ---------------------------------------------------------------------------


_CHECKED = {f"{_PACKAGE}.coordinate.frames", f"{_PACKAGE}.kinematics.vector"}
_MODULES = _CHECKED | {
    _PACKAGE,
    f"{_PACKAGE}.coordinate",
    f"{_PACKAGE}.coordinate.poses",
    f"{_PACKAGE}.kinematics",
    f"{_PACKAGE}.utils",
    f"{_PACKAGE}.utils.config",
}


def _target(name: str) -> str | None:
    if name in _CHECKED:
        return workspace_module(name)
    if name.startswith(f"{_PACKAGE}.coordinate"):
        return f"{_PACKAGE}.coordinate"  # the model of the package
    return None


def _redirect(source: str, module: str = f"{_PACKAGE}.kinematics.velocity"):
    return redirect_imports(
        textwrap.dedent(source), module, False, _target, _MODULES.__contains__
    )


def test_an_import_of_a_checked_module_names_its_workspace_module() -> None:
    out, problems = _redirect(
        """
        from autoware_carla_scenario.coordinate.frames import (
            CoordinateFrame,
            frame_of as of,
        )
        from .vector import Vector3
        import math
        """
    )
    assert problems == []
    lines = out.splitlines()
    assert lines[1] == (
        "from _acs_lib.autoware_carla_scenario__coordinate__frames "
        "import CoordinateFrame, frame_of as of"
    )
    assert (
        lines[5]
        == "from _acs_lib.autoware_carla_scenario__kinematics__vector import Vector3"
    )
    assert lines[6] == "import math"
    assert len(lines) == 7  # every statement kept its line


def test_an_import_of_an_unchecked_module_names_its_model() -> None:
    out, problems = _redirect(
        """
        if TYPE_CHECKING:
            from ..coordinate.poses import Lanelet2Pose
        from ..coordinate import frames, poses as p
        """
    )
    assert problems == []
    assert "    from autoware_carla_scenario.coordinate import Lanelet2Pose" in out
    assert (
        "from _acs_lib import autoware_carla_scenario__coordinate__frames as frames; "
        "from autoware_carla_scenario import coordinate as p"
    ) in out


def test_an_import_with_nothing_to_stand_in_is_reported() -> None:
    out, problems = _redirect(
        """
        from autoware_carla_scenario.utils.config import load
        import autoware_carla_scenario.coordinate.frames
        from .vector import *
        """
    )
    assert [line for line, _ in problems] == [2, 3, 4]
    assert "neither checked" in problems[0][1]
    assert "from autoware_carla_scenario.utils.config import load" in out


def test_a_module_of_a_checked_package_can_have_a_model_of_its_own() -> None:
    # utils is checked, and utils/traffic_light.codon models one module of it
    # that is not; the other modules of utils have nothing to stand in.
    assert _model_module(f"{_PACKAGE}.utils.traffic_light") == (
        f"{_PACKAGE}.utils.traffic_light"
    )
    assert _model_module(f"{_PACKAGE}.utils.config") is None
    assert _model_module(f"{_PACKAGE}.coordinate.transform") == f"{_PACKAGE}.coordinate"


@pytest.mark.parametrize(
    ("module", "sibling"),
    [
        ("autoware_bridge.grpc_server", "autoware_bridge.ros_bridge.client"),
        ("autoware_stack.launcher", "autoware_stack.docker"),
        ("driver.egodriver_client", "driver.renderer"),
    ],
)
def test_a_re_exported_module_has_a_model_of_its_own(module: str, sibling: str) -> None:
    # The checked autoware_bridge and driver packages re-export a class of a
    # module that is not checked (grpc), and a scenario's AutowareEgoEntity
    # takes an AutowareLauncher (whose module the library check compiles from
    # its source): each such module has a model of its own, under a package
    # directory of the model, and its package and the package's other modules
    # have none.
    assert _model_module(f"{_PACKAGE}.{module}") == f"{_PACKAGE}.{module}"
    assert _model_module(f"{_PACKAGE}.{module.split('.')[0]}") is None
    assert _model_module(f"{_PACKAGE}.{sibling}") is None


def test_a_modelled_import_names_the_model_though_the_module_is_checked() -> None:
    # measures and scenario_base import each other: measures has scenario_base
    # through its model, scenario_base has measures from its source.
    measures, scenario_base = f"{_PACKAGE}.measures", f"{_PACKAGE}.scenario_base"
    assert scenario_base in MODELLED_IMPORTS[measures]
    every = package_modules()
    sources = check._Sources()
    for name in (measures, scenario_base):
        text = every[name].read_text(encoding="utf-8")
        sources.modules[name] = check._Module(every[name], False, text, ast.parse(text))
    files = {
        f.module: f.text
        for f in check._library_files([measures, scenario_base], sources, every)
    }
    assert f"from {_PACKAGE}.scenario_base import BaseScenario" in files[measures]
    assert f"from {workspace_module(measures)} import " in files[scenario_base]


# ---------------------------------------------------------------------------
# Model modules compiled from the checked source instead
# ---------------------------------------------------------------------------


def test_every_replaced_model_names_a_model_file_and_package_modules() -> None:
    modules = package_modules()
    for stem, sources in REPLACED_MODELS.items():
        assert (model_dir() / _PACKAGE / f"{stem}.codon").is_file(), stem
        assert sources and all(m in modules for m in sources), stem


def _trees(*names: str) -> dict[str, SimpleNamespace]:
    every = package_modules()
    return {n: SimpleNamespace(tree=ast.parse(every[n].read_text())) for n in names}


def test_a_replaced_model_re_exports_the_checked_definitions() -> None:
    frames, poses = REPLACED_MODELS["_poses"]
    problems: list = []
    out = replaced_models(_trees(frames, poses), problems)
    assert problems == []
    assert out["_poses"].splitlines() == [
        f"from {workspace_module(frames)} import CoordinateFrame, FrameMismatchError",
        f"from {workspace_module(poses)} import CarlaWorldPose, Lanelet2Pose, "
        "OpenDrivePose",
    ]


def test_the_vehicle_entity_model_re_exports_the_checked_spawn_and_vehicle() -> None:
    # So the model's EgoConfig derives from the checked VehicleEntityConfig and
    # the checked ego hands the checked spawn_vehicle_actor its own SpawnLocation.
    spawn, vehicle = REPLACED_MODELS["_vehicle_entity"]
    problems: list = []
    out = replaced_models(_trees(spawn, vehicle), problems)
    assert problems == []
    assert out["_vehicle_entity"].splitlines() == [
        f"from {workspace_module(spawn)} import SpawnLocation, SpawnPointIndex, "
        "SpawnTransform",
        f"from {workspace_module(vehicle)} import VehicleEntity, VehicleEntityConfig",
    ]


def test_the_ground_projection_model_re_exports_the_checked_config() -> None:
    # So a scenario's GroundProjectionConfig, imported from the top-level
    # package (the model), is the one the checked VehicleEntityConfig holds.
    (snap,) = REPLACED_MODELS["_ground_projection"]
    problems: list = []
    out = replaced_models(_trees(snap), problems)
    assert problems == []
    assert out["_ground_projection"].splitlines() == [
        f"from {workspace_module(snap)} import GroundProjectionConfig",
    ]


def test_the_bridge_pose_model_re_exports_the_checked_pose() -> None:
    # So the pose the model's to_map_frame returns is the one the checked
    # AutowareBridge.configure() takes, which AutowareEgoEntity hands it.
    (base,) = REPLACED_MODELS["_bridge_pose"]
    problems: list = []
    out = replaced_models(_trees(base), problems)
    assert problems == []
    assert out["_bridge_pose"].splitlines() == [
        f"from {workspace_module(base)} import BridgePose, Quaternion, Vector3",
    ]


def test_the_map_source_model_re_exports_the_checked_source() -> None:
    # So the model GitMapCache (maps/cache.codon) takes the MapSource the
    # checked maps.resolver and maps.catalogue hand it.
    (source,) = REPLACED_MODELS["_map_source"]
    problems: list = []
    out = replaced_models(_trees(source), problems)
    assert problems == []
    assert out["_map_source"].splitlines() == [
        f"from {workspace_module(source)} import MapSource",
    ]


def test_the_ros2_model_re_exports_the_checked_config_and_transport() -> None:
    # So the model CycloneRos2Transport (ros2/cyclone.codon) derives from the
    # checked Ros2EgoTransport and takes the checked Ros2EgoConfig, which the
    # checked entity.ros2_entity._cyclone_transport() hands it.
    config, transport = REPLACED_MODELS["_ros2"]
    problems: list = []
    out = replaced_models(_trees(config, transport), problems)
    assert problems == []
    assert out["_ros2"].splitlines() == [
        f"from {workspace_module(config)} import Ros2EgoConfig",
        f"from {workspace_module(transport)} import Ros2EgoTransport",
    ]


def test_the_cyclone_model_stands_in_for_the_unchecked_module() -> None:
    # ros2.cyclone imports cyclonedds: the checked ros2_entity imports it
    # through its model, a module of the checked ros2 package's own.
    assert f"{_PACKAGE}.ros2.cyclone" in EXCLUDED
    assert f"{_PACKAGE}.ros2" in CHECKED
    assert _model_module(f"{_PACKAGE}.ros2.cyclone") == f"{_PACKAGE}.ros2.cyclone"
    assert _model_module(f"{_PACKAGE}.ros2") is None
    assert _model_module(f"{_PACKAGE}.ros2.messages") is None


def test_a_model_is_replaced_only_once_all_its_modules_are_checked() -> None:
    frames, _poses = REPLACED_MODELS["_poses"]
    problems: list = []
    assert "_poses" not in replaced_models(_trees(frames), problems)
    assert problems == []


def test_a_name_the_checked_source_lacks_is_a_problem() -> None:
    frames, poses = REPLACED_MODELS["_poses"]
    trees = _trees(frames, poses)
    trees[poses].tree.body = [
        node
        for node in trees[poses].tree.body
        if getattr(node, "name", None) != "OpenDrivePose"
    ]
    problems: list = []
    out = replaced_models(trees, problems)
    assert [p.message.split(",")[0] for p in problems] == ["OpenDrivePose"]
    assert "OpenDrivePose" not in out["_poses"]


# ---------------------------------------------------------------------------
# The calls appended to a checked module
# ---------------------------------------------------------------------------


_MODULE = """
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


def angle(a: float, *, wrap: bool = True) -> float:
    return a


def where(pose: Lanelet2Pose | OpenDrivePose | None) -> float:
    return 0.0


def total(xs: Sequence[float]) -> float:
    return 0.0


def _private(x: float) -> None:
    pass


def __getattr__(name: str) -> object:
    return name


class Vector:
    def __init__(self, x: float) -> None:
        self.x = x

    def __eq__(self, other: object) -> bool:
        return True

    @property
    def norm(self) -> float:
        return self.x

    @staticmethod
    def zero() -> Vector:
        return Vector(0.0)

    @classmethod
    def unit(cls, scale: float) -> Vector:
        return cls(scale)

    def _helper(self, scale: float) -> None:
        pass


@dataclass
class _Walk:
    steps: int
    seen: list[int] = field(init=False)

    def __post_init__(self) -> None:
        self.seen = []

    def step(self, n: int) -> None:
        pass


@dataclass
class Config:
    name: str
    ids: list[int] = field(default_factory=list)
    hidden: int = field(default=0, init=False)


class Level(Enum):
    LOW = 1

    def up(self) -> Level:
        return self


def generic(x: Any, y) -> None:
    pass
"""


def test_every_public_function_and_method_is_called_with_typed_values() -> None:
    checks = render_library_checks(ast.parse(_MODULE))
    calls = checks.source.splitlines()
    assert "    angle(_acs_value(float), wrap=_acs_value(bool))" in calls
    assert "    where(_acs_value(Lanelet2Pose))" in calls
    assert "    where(_acs_value(OpenDrivePose))" in calls
    assert "    total(_acs_value(Sequence[float]))" in calls
    assert "    Vector(_acs_value(float))" in calls
    # `==`, as Python calls it (an explicit __eq__ override call fails, #596).
    assert "    (_acs_value(Vector) == _acs_value(Vector))" in calls
    assert "    _acs_value(Vector).norm" in calls
    assert "    Vector.zero()" in calls
    assert "    Vector.unit(_acs_value(float))" in calls
    assert "    Config(_acs_value(str), _acs_value(list[int]))" in calls
    assert "    _acs_value(Level).up()" in calls
    assert not [c for c in calls if "Level(" in c]
    label = checks.labels[calls.index("    where(_acs_value(OpenDrivePose))") + 1]
    assert label == "where(pose: OpenDrivePose)"


_DEFAULTS_RENDERED = """
def nearest(x: float, max_distance: float = 150.0) -> float:
    return x


def run(cwd: str, *args: str, timeout: int, env: str = "") -> None:
    pass


def plain(x: float) -> None:
    pass


class Bridge:
    def configure(self, goal: float, waypoints: Optional[list[float]] = None) -> None:
        pass

    def start(self, world: float, *, skip: Collection[int] = ()) -> None:
        pass

    @staticmethod
    def make(size: float = 1.0) -> None:
        pass
"""


def test_a_function_with_defaults_is_called_without_them_too() -> None:
    # Every call passes every argument, so no default would be compiled: one
    # call more leaves out each argument that has one (#183).
    checks = render_library_checks(ast.parse(_DEFAULTS_RENDERED))
    calls = checks.source.splitlines()
    assert "    nearest(_acs_value(float), _acs_value(float))" in calls
    assert "    nearest(_acs_value(float))" in calls
    label = checks.labels[calls.index("    nearest(_acs_value(float))") + 1]
    assert label == "nearest(x: float), with the default of `max_distance`"
    # A `*args: T` is still given one T, so the positional parameters before
    # it are passed; the keyword-only one without a default is too.
    assert "    run(_acs_value(str), _acs_value(str), timeout=_acs_value(int))" in calls
    # A function without a default is called once.
    assert [c for c in calls if c.startswith("    plain(")] == [
        "    plain(_acs_value(float))"
    ]
    # A method through its class: its dispatch thunk fills in no default.
    assert "    Bridge.configure(_acs_value(Bridge), _acs_value(float))" in calls
    assert "    Bridge.start(_acs_value(Bridge), _acs_value(float))" in calls
    assert "    Bridge.make()" in calls


_LEFT_TO_DEFAULTS_RENDERED = """
from typing import Any, Callable, Optional


class Holder:
    def __init__(
        self, size: float, kind: type[Base] | None = None, *, tag: Any = None
    ) -> None:
        pass


def build(x: float, factory: Optional[Callable[..., float]] = None) -> None:
    pass


def needs(x: float, kind: type[Base]) -> None:
    pass


def spread(first: Any = None, *rest: int) -> None:
    pass
"""


def test_a_defaulted_parameter_codon_cannot_express_is_left_to_its_default() -> None:
    # Each parameter Codon cannot express has a default: the function is
    # called leaving them out, and only so, instead of being a problem (#199).
    checks = render_library_checks(ast.parse(_LEFT_TO_DEFAULTS_RENDERED))
    calls = checks.source.splitlines()
    assert [c for c in calls if c.startswith("    Holder(")] == [
        "    Holder(_acs_value(float))"
    ]
    label = checks.labels[calls.index("    Holder(_acs_value(float))") + 1]
    assert label == "Holder.__init__(size: float), with the default of `kind`, `tag`"
    assert [c for c in calls if c.startswith("    build(")] == [
        "    build(_acs_value(float))"
    ]
    # A required one is still a problem, as is a defaulted one the call
    # would pass (before a `*args: T`, which is given one T).
    assert not [c for c in calls if c.startswith(("    needs(", "    spread("))]
    lines = _LEFT_TO_DEFAULTS_RENDERED.splitlines()
    assert sorted(
        (lines[line - 1].split("(")[0], m.split(":")[0]) for line, m in checks.problems
    ) == [("def needs", "needs"), ("def spread", "spread")]
    assert all("Codon cannot express" in m for _, m in checks.problems)


_SETTERS_RENDERED = """
from typing import Any, Optional


class Gauge:
    @property
    def level(self) -> float:
        return 0.0

    @level.setter
    def level(self, value: float | str) -> None:
        pass

    @level.deleter
    def level(self) -> None:
        pass

    @property
    def note(self) -> Optional[str]:
        return None

    @note.setter
    def note(self, value: Optional[str]) -> None:
        pass

    @property
    def anything(self) -> int:
        return 0

    @anything.setter
    def anything(self, value: Any) -> None:
        pass
"""


def test_a_property_setter_is_called_by_assigning_the_property() -> None:
    # Codon compiles a setter only where the property is assigned: the check
    # assigns it, once per member of the value's annotation (#200).
    checks = render_library_checks(ast.parse(_SETTERS_RENDERED))
    calls = checks.source.splitlines()
    assert [c for c in calls if " = " in c] == [
        "    _acs_value(Gauge).level = _acs_value(float)",
        "    _acs_value(Gauge).level = _acs_value(str)",
        "    _acs_value(Gauge).note = _acs_value(Optional[str])",
    ]
    for call in [c for c in calls if " = " in c]:
        name = call.split(".")[1].split(" ")[0]
        assert checks.labels[calls.index(call) + 1] == f"Gauge.{name} (setter)"
    # The getter is read as before; the deleter is not called.
    assert calls.count("    _acs_value(Gauge).level") == 1
    lines = _SETTERS_RENDERED.splitlines()
    [(line, message)] = checks.problems
    assert lines[line - 1].strip() == "def anything(self, value: Any) -> None:"
    assert message.startswith("Gauge.anything (setter): Codon cannot express `Any`")


_OPTIONAL_RETURNS_RENDERED = """
from typing import Optional

import typesafe_carla.carla as carla


def find(world: carla.World) -> Optional[carla.Actor]:
    return None


def count() -> int:
    return 0


class Gauge:
    def where(self) -> carla.Transform | None:
        return None

    @property
    def note(self) -> Optional[str]:
        return None
"""


def test_the_type_an_optional_function_returns_is_realized_before_the_call() -> None:
    # Its `return None` becomes `return Optional[X]()`, which Codon 0.19 fails
    # to generate code for when nothing else realizes X (#202).
    checks = render_library_checks(ast.parse(_OPTIONAL_RETURNS_RENDERED))
    calls = checks.source.splitlines()
    start = calls.index("    _acs_value(carla.Actor)")
    assert calls[start : start + 2] == [
        "    _acs_value(carla.Actor)",
        "    find(_acs_value(carla.World))",
    ]
    assert checks.labels[start + 1] == "find, realizing its return type"
    assert "    count()" in calls
    assert not any("_acs_value(int)" in c for c in calls)
    for value, call in [
        ("carla.Transform", "_acs_value(Gauge).where()"),
        ("str", "_acs_value(Gauge).note"),
    ]:
        at = calls.index(f"    {call}")
        assert calls[at - 1] == f"    _acs_value({value})"


_FIELD_DEFAULTS_RENDERED = """
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class Docker:
    workspace: str
    launch: Sequence[str]
    colcon_args: Sequence[str] = ("--symlink-install",)
    env: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    timeout: float = 10.0
    required: int = field(kw_only=True)


@dataclass
class Point:
    x: float
    y: float


@dataclass
class Defaults:
    k: float = 0.0
"""


def test_a_dataclass_is_constructed_without_its_field_defaults_too() -> None:
    # Every constructor call passes every field, so no field default would
    # be compiled: one call more passes only the fields without one (#188).
    checks = render_library_checks(ast.parse(_FIELD_DEFAULTS_RENDERED))
    calls = checks.source.splitlines()
    assert (
        "    Docker(_acs_value(str), _acs_value(Sequence[str]), "
        "_acs_value(Sequence[str]), _acs_value(dict[str, str]), "
        "timeout=_acs_value(float), required=_acs_value(int))" in calls
    )
    # A field after a defaulted one is passed by keyword; a `field()` with no
    # default (here keyword-only) is required.  The `Any` field is left out
    # of both calls.
    required = (
        "    Docker(_acs_value(str), _acs_value(Sequence[str]), "
        "required=_acs_value(int))"
    )
    assert required in calls
    label = checks.labels[calls.index(required) + 1]
    assert label == (
        "Docker.__init__(workspace: str, launch: Sequence[str], required: int), "
        "with the default of `colcon_args`, `env`, `timeout`"
    )
    # A dataclass without a default is constructed once; one with only
    # defaulted fields is constructed with none too.
    assert [c for c in calls if c.startswith("    Point(")] == [
        "    Point(_acs_value(float), _acs_value(float))"
    ]
    assert "    Defaults()" in calls


_CLASS_DEFAULTS_RENDERED = """
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, List, Optional, TypeVar

T = TypeVar("T")
DEFAULT_PORT = 8000
LOW = 0.0


class Action:
    REISSUES: ClassVar[bool] = False
    KIND: ClassVar = "action"
    port: int = DEFAULT_PORT
    label: Optional[str] = None
    ready: bool = True
    LIMIT: float = 2.0
    TWICE: float = LIMIT * 2
    LOW, HIGH = 0.0, 1.0
    MID: float = (LOW + HIGH) / 2
    SQUARES: List[int] = [i * i for i in range(3)]
    held: T = None
    __slots__ = ("_value",)

    def __init__(self) -> None:
        pass


class Follow(Action):
    REISSUES = True
    port = 9000
    KIND = "follow"
    undeclared = 1


class Mixin:
    backend: Optional[str] = None


class _Memo:
    generation: int = -1


_MEMO = _Memo()


@dataclass
class Config:
    timeout: float = 1.0


class Color(Enum):
    RED: str = "red"
"""


def test_a_class_level_default_is_assigned_to_a_typed_local() -> None:
    # Codon makes `x: T = v` of a class with a constructor a field and drops
    # `v`: the check assigns it to a local of type T, so it is compiled (#190).
    checks = render_library_checks(ast.parse(_CLASS_DEFAULTS_RENDERED))
    lines = checks.source.splitlines()
    defaults = {
        checks.labels[n]: line.split(": ", 1)[1]
        for n, line in enumerate(lines, 1)
        if "_acs_cls_default_" in line
    }
    assert defaults == {
        "class-level default of Action.port": "int = DEFAULT_PORT",
        "class-level default of Action.label": "Optional[str] = None",
        # Codon converts any value assigned to a `bool`; a call does not.
        "class-level default of Action.ready": "bool = _expect_bool(True)",
        "class-level default of Action.LIMIT": "float = 2.0",
        # A comprehension's variable is not a member of the class.
        "class-level default of Action.SQUARES": "List[int] = [i * i for i in range(3)]",
        # An override without an annotation takes its base's, a ClassVar's T.
        "class-level default of Follow.REISSUES": "bool = _expect_bool(True)",
        "class-level default of Follow.port": "int = 9000",
        # A class with no constructor is checked too: what derives from it
        # may have one.
        "class-level default of Mixin.backend": "Optional[str] = None",
    }
    # Each local is named after its line, so no two share a name.
    for n, line in enumerate(lines, 1):
        if "_acs_cls_default_" in line:
            assert line.startswith(f"    _acs_cls_default_{n}: ")
    # Left out: a ClassVar (Codon keeps its value) and an override of a bare
    # one, a value naming another member (`LIMIT * 2`, or `LOW`, which an
    # unpacking binds and the module binds too), a TypeVar, a dunder,
    # a name no base declares, the defaults of a class the module constructs
    # with no argument and no constructor of its own (Codon's generated one
    # takes them), and a dataclass's and an enum's.
    for absent in (
        "Action.REISSUES",
        "Action.KIND",
        "Follow.KIND",
        "Action.TWICE",
        "Action.MID",
        "Action.held",
        "__slots__",
        "Follow.undeclared",
        "_Memo.generation",
        "Config.timeout",
        "Color.RED",
    ):
        assert not [label for label in defaults if label.endswith(absent)]
    assert "from autoware_carla_scenario._tc import _acs_value, _expect_bool" in lines


def test_the_scenario_check_binds_each_method_result_as_a_hook() -> None:
    # With `hook_returns` (the scenario check, #238), each call of a method
    # or a property passes its result to the model's `_acs_hook_return`,
    # which checks it against the framework method it overrides; a function,
    # a static method, the constructor and a dunder are no hooks.
    source = textwrap.dedent(
        """
        import typesafe_carla.carla as carla


        def helper(x: float) -> float:
            return x


        class Gap(BaseCondition):
            def __init__(self, label: str) -> None:
                super().__init__(label)

            def check(self, world: carla.World, elapsed: float = 0.0) -> bool:
                return True

            def renamed(self, world: carla.World, *, scale: float = 1.0) -> bool:
                return True

            @property
            def ready(self) -> bool:
                return True

            @staticmethod
            def build() -> int:
                return 0

            def __eq__(self, other: object) -> bool:
                return True
        """
    )
    tree = ast.parse(source)
    plain = render_library_checks(tree, skip_any_returns=True)
    assert "_acs_hook_return" not in plain.source
    hooked = render_library_checks(tree, skip_any_returns=True, hook_returns=True)
    lines = [line.strip() for line in hooked.source.splitlines()]
    assert "from autoware_carla_scenario._tc import _acs_hook_return" in lines
    assert [line for line in lines if "_acs_hook_return(" in line] == [
        '_acs_hook_return(Gap, "check", '
        "_acs_value(Gap).check(_acs_value(carla.World), _acs_value(float)))",
        '_acs_hook_return(Gap, "check", '
        "Gap.check(_acs_value(Gap), _acs_value(carla.World)))",
        '_acs_hook_return(Gap, "renamed", '
        "Gap.renamed(_acs_value(Gap), _acs_value(carla.World), "
        "scale=_acs_value(float)))",
        '_acs_hook_return(Gap, "renamed", '
        "Gap.renamed(_acs_value(Gap), _acs_value(carla.World)))",
        '_acs_hook_return(Gap, "ready", _acs_value(Gap).ready)',
    ]
    assert "helper(_acs_value(float))" in lines
    assert "Gap.build()" in lines
    assert "(_acs_value(Gap) == _acs_value(Gap))" in lines


def test_an_abstract_or_raising_method_is_called_but_not_bound_as_a_hook() -> None:
    # Such a method returns no value, which Codon types None whatever its
    # annotation says, so binding it as a hook refused a valid override.
    source = textwrap.dedent(
        """
        from abc import abstractmethod


        class Base(BaseScenario):
            @abstractmethod
            def is_done(self) -> bool:
                ...

            def create_ego(self) -> EgoVehicle:
                raise NotImplementedError()

            @property
            def ego_requires_goal(self) -> bool:
                raise NotImplementedError()

            def measure(self, key: str) -> float:
                if key:
                    return 1.0
                raise KeyError(key)
        """
    )
    hooked = render_library_checks(
        ast.parse(source), skip_any_returns=True, hook_returns=True
    )
    lines = [line.strip() for line in hooked.source.splitlines()]
    assert "_acs_value(Base).is_done()" in lines
    assert "_acs_value(Base).create_ego()" in lines
    assert "_acs_value(Base).ego_requires_goal" in lines
    assert [line for line in lines if "_acs_hook_return(" in line] == [
        '_acs_hook_return(Base, "measure", '
        "_acs_value(Base).measure(_acs_value(str)))",
    ]


def test_a_generic_function_and_one_returning_any_are_problems() -> None:
    # `_acs_value(list[T])` does not compile: a TypeVar parameter is a
    # problem.  A function returning `Any` builds what Codon cannot express
    # (`get_details() -> dict[str, Any]`): left out with `skip_any_returns`,
    # which the scenario check passes for want of UNCALLED (#192).
    source = textwrap.dedent(
        """
        from typing import Any, TypeVar

        T = TypeVar("T")


        def first(xs: list[T]) -> T:
            return xs[0]


        class Probe:
            def get_details(self) -> dict[str, Any]:
                return {}

            @property
            def raw(self) -> "Any":
                return 0

            def speed(self) -> float:
                return 0.0
        """
    )
    tree = ast.parse(source)
    checks = render_library_checks(tree)
    assert [line for line, _ in checks.problems] == [7]
    assert "parameter `xs` is annotated with a TypeVar" in checks.problems[0][1]
    assert "first(" not in checks.source
    assert "_acs_value(Probe).get_details()" in checks.source

    checks = render_library_checks(tree, skip_any_returns=True)
    assert [line for line, _ in checks.problems] == [7, 12, 16]
    assert "Probe.get_details: returns `dict[str, Any]`" in checks.problems[1][1]
    assert "Probe.raw: returns `Any`" in checks.problems[2][1]
    assert "get_details" not in checks.source
    assert ".raw" not in checks.source
    assert "_acs_value(Probe).speed()" in checks.source


def test_each_appended_line_names_the_line_it_compiles() -> None:
    # The scenario check reports an error on an appended line at the line of
    # what it compiles (#192): a function's `def`, a class-level default.
    source = _CLASS_DEFAULTS_RENDERED
    checks = render_library_checks(ast.parse(source))
    module = source.splitlines()
    assert set(checks.lines) == set(checks.labels)
    for n, label in checks.labels.items():
        line = module[checks.lines[n] - 1]
        if label.startswith("class-level default of "):
            member = label.rsplit(".", 1)[1]
            assert line.lstrip().startswith(member), (label, line)
        else:
            owner, _, name = label.split("(", 1)[0].rpartition(".")
            # A dataclass's generated constructor is at its `class` line.
            assert f"def {name}(" in line or (
                name == "__init__" and f"class {owner}" in line
            ), (label, line)


def test_protected_functions_and_methods_are_called_too() -> None:
    # A `_` function or method is a hook a subclass or another module uses
    # (BaseScenario._setup_ego_spawn): Codon compiles its body only when
    # something calls it, so the check calls it as it calls a public one.
    calls = render_library_checks(ast.parse(_MODULE)).source.splitlines()
    assert "    _private(_acs_value(float))" in calls
    assert "    _acs_value(Vector)._helper(_acs_value(float))" in calls
    # So are the methods of a `_` class: what uses such a helper may not
    # compile them (an UNCALLED constructor, a call behind `hasattr`), and
    # Codon's constructor does not call a dataclass's __post_init__ (#182).
    assert "    _Walk(_acs_value(int))" in calls
    assert "    _acs_value(_Walk).__post_init__()" in calls
    assert "    _acs_value(_Walk).step(_acs_value(int))" in calls
    # A dunder of the module (PEP 562's lazy `__getattr__`) is Python's hook,
    # not the module's code: it is not called.
    assert not [c for c in calls if "__getattr__" in c]
    # One the check cannot call with is a problem of the module, as a public
    # one is, unless UNCALLED names it.
    tree = ast.parse("def _anything(x: Any) -> None:\n    pass\n")
    assert render_library_checks(tree).problems
    assert not render_library_checks(tree, ["_anything"]).problems


def test_an_optional_union_parameter_is_called_with_each_member() -> None:
    source = """
def lookup(ref: Optional[Union[EntityRole, str]]) -> None:
    pass


def override(value: Optional[float | str]) -> None:
    pass
"""
    checks = render_library_checks(ast.parse(source))
    calls = checks.source.splitlines()
    assert checks.problems == []
    assert "    lookup(_acs_value(EntityRole))" in calls
    assert "    lookup(_acs_value(str))" in calls
    assert "    override(_acs_value(float))" in calls
    assert "    override(_acs_value(str))" in calls


def test_a_parameter_the_check_cannot_call_with_is_a_problem_of_the_module() -> None:
    checks = render_library_checks(ast.parse(_MODULE))
    lines = _MODULE.splitlines()
    assert len(checks.problems) == 2
    for line, message in checks.problems:
        assert lines[line - 1].startswith("def generic(")
    assert any("`x`" in m and "Any" in m for _, m in checks.problems)
    assert any("`y` has no annotation" in m for _, m in checks.problems)


_CALLABLES = """
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Union


def apply(f: Callable[[float, str], Optional[float]]) -> None:
    pass


@dataclass
class Reader:
    read: Callable[["carla.World"], float]


def anything(f: Callable[[Any], Any]) -> None:
    pass


def variadic(f: Callable[..., float]) -> None:
    pass


def register(cb: Union[float, Callable[[float], None]]) -> None:
    pass


def maybe(f: Optional[Callable[[float], int]] = None) -> None:
    pass


def place(f: Callable[[Union[Straight, Turn]], float]) -> None:
    pass


def refer(f: Optional[Callable[[Optional[Union[EntityRole, str]]], Pose]]) -> None:
    pass


def unknown(f: Callable[[Union[Straight, Any]], float]) -> None:
    pass
"""


def test_a_callable_is_called_with_a_value_of_its_codon_type() -> None:
    checks = render_library_checks(ast.parse(_CALLABLES))
    calls = checks.source.splitlines()
    assert "    apply(_acs_value(Callable[[float, str], Optional[float]]))" in calls
    assert "    Reader(_acs_value(Callable[[carla.World], float]))" in calls
    # A union with a callable in it: called once with each member.
    assert "    register(_acs_value(float))" in calls
    assert "    register(_acs_value(Callable[[float], None]))" in calls
    # An Optional one: called with the callable.
    assert "    maybe(_acs_value(Callable[[float], int]))" in calls
    # A union among the arguments, a Callable type Codon 0.19 crashes on: a
    # callable of as many arguments, of any type, returning the result's type.
    assert "    place(_acs_value(_acs_Returning[float, 1]))" in calls
    assert "    refer(_acs_value(_acs_Returning[Pose, 1]))" in calls
    lines = _CALLABLES.splitlines()
    assert sorted(lines[line - 1].split("(")[0] for line, _ in checks.problems) == [
        "def anything",
        "def unknown",
        "def variadic",
    ]


_DATACLASS_FIELDS = """
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Union


@dataclass
class Run:
    name: str
    scenario: Straight | Turn = field(default_factory=Straight)
    constraints: dict[str, Any] = field(default_factory=dict)
    npc: Union[Straight, Turn, None] = None
    timeout: float = 5.0


@dataclass
class Required:
    options: dict[str, Any]
"""


def test_a_dataclass_is_constructed_with_each_member_of_a_union_field() -> None:
    checks = render_library_checks(ast.parse(_DATACLASS_FIELDS))
    calls = [c.strip() for c in checks.source.splitlines() if "Run(" in c]
    # A dict[str, Any] field with a default is left out of the call, and the
    # fields after it are passed by keyword.  The last call passes only the
    # field without a default, so the defaults are compiled (#188).
    assert calls == [
        *(
            f"Run(_acs_value(str), _acs_value({scenario}), "
            f"npc=_acs_value({npc}), timeout=_acs_value(float))"
            for scenario in ("Straight", "Turn")
            for npc in ("Straight", "Turn")
        ),
        "Run(_acs_value(str))",
    ]
    label = checks.labels[checks.source.splitlines().index("    " + calls[-2]) + 1]
    assert label == "Run.__init__(name: str, scenario: Turn, npc: Turn, timeout: float)"
    # Without a default, the field cannot be left out: a problem, as before.
    lines = _DATACLASS_FIELDS.splitlines()
    assert [lines[line - 1].strip() for line, _ in checks.problems] == [
        "options: dict[str, Any]"
    ]


_DERIVED_DATACLASS = """
from dataclasses import dataclass
from typing import Optional

from .base import CameraConfig


@dataclass(frozen=True)
class Lens:
    k: float = 0.0


@dataclass(frozen=True)
class Camera(CameraConfig):
    sensor_type: str = "sensor.camera.rgb"
    iso: Optional[float] = None
    width: int = 640  # redefined: keeps the base's place


@dataclass
class Fisheye(Lens):
    fov: float = 180.0


class Plain:
    label: str = ""


@dataclass
class Labelled(Plain):
    size: int = 0
"""

_DATACLASS_BASE = """
from dataclasses import dataclass


@dataclass(frozen=True)
class CameraConfig:
    width: int = 1920
    fov: float = 90.0
"""


def test_a_dataclass_is_constructed_with_the_fields_of_its_bases_first() -> None:
    base = ast.parse(_DATACLASS_BASE)
    classes = {n.name: n for n in base.body if isinstance(n, ast.ClassDef)}
    checks = render_library_checks(ast.parse(_DERIVED_DATACLASS), (), classes)
    calls = [c.strip() for c in checks.source.splitlines() if "_acs_value(" in c]
    # A base of another checked module, a base of the same module, and a base
    # that is no dataclass, whose annotations are no fields.
    assert calls == [
        "Lens(_acs_value(float))",
        "Camera(_acs_value(int), _acs_value(float), _acs_value(str), "
        "_acs_value(Optional[float]))",
        "Fisheye(_acs_value(float), _acs_value(float))",
        "Labelled(_acs_value(int))",
    ]
    # Without the classes of the other modules, the base's fields are unknown.
    alone = render_library_checks(ast.parse(_DERIVED_DATACLASS))
    assert (
        "    Camera(_acs_value(str), _acs_value(Optional[float]), _acs_value(int))"
        in alone.source.splitlines()
    )


_MIXED_DATACLASSES = """
from dataclasses import dataclass, field


@dataclass
class Size:
    width: int = 0


@dataclass
class Name:
    label: str = ""


class Plain(Size):
    pass


@dataclass
class Through(Plain):
    depth: float = 0.0


@dataclass
class Hidden(Size):
    width: int = field(default=1, init=False)
    depth: float = 0.0


@dataclass
class Both(Size, Name):
    depth: float = 0.0
"""


def test_a_dataclass_takes_its_base_fields_as_python_does() -> None:
    import dataclasses

    namespace: dict[str, object] = {}
    exec(_MIXED_DATACLASSES, namespace)  # noqa: S102 - the source above
    checks = render_library_checks(ast.parse(_MIXED_DATACLASSES))
    calls = [c.strip() for c in checks.source.splitlines() if "_acs_value(" in c]
    # Through a plain class derived from a dataclass, and without a field
    # redefined with init=False, as Python's constructor takes them.
    assert "Through(_acs_value(int), _acs_value(float))" in calls
    assert "Hidden(_acs_value(float))" in calls
    for name, count in [("Through", 2), ("Hidden", 1)]:
        cls = namespace[name]
        assert dataclasses.is_dataclass(cls)
        assert len([f for f in dataclasses.fields(cls) if f.init]) == count
    # Python takes Name's field first, Codon 0.19 Size's: no call, a problem.
    assert not any(c.startswith("Both(") for c in calls)
    line = _MIXED_DATACLASSES.splitlines().index("class Both(Size, Name):") + 1
    assert [n for n, message in checks.problems if "Both" in message] == [line]


def test_an_uncalled_function_is_left_out() -> None:
    checks = render_library_checks(
        ast.parse(_MODULE), ["angle", "Vector.norm", "Config.__init__", "generic"]
    )
    calls = checks.source.splitlines()
    assert not [c for c in calls if "angle(" in c or ".norm" in c or "Config(" in c]
    assert "    Vector.zero()" in calls
    # generic() is not called, so its parameters are no problem.
    assert checks.problems == []


_VARIADIC = """
def opts(**kw: float) -> float:
    return kw["speed"]


def pair(*xs: float) -> float:
    return xs[0] + xs[1]


def unused(**kw: float) -> float:
    return kw["speed"]


def unused_get(**kw: float) -> float:
    return kw.get("speed", 1.0)


def unused_args(*xs: float) -> float:
    return xs[0]


def recursive(*xs: float) -> float:
    return recursive(*xs[1:]) if len(xs) > 1 else 0.0


class Tuner:
    def __init__(self, **gains: float) -> None:
        self.p = gains["p"]

    def scale(self, *xs: float) -> float:
        return xs[0] * xs[1]


def setup() -> None:
    print(opts(speed=2.0), pair(1.0, 2.0), Tuner(p=1.0).scale(1.0, 2.0))
    gains: dict[str, float] = {}
    gains.update(p=1.0)


def update(**kw: float) -> float:
    return kw.get("p", 1.0)
"""


def test_a_variadic_function_its_module_calls_is_left_to_those_calls() -> None:
    # The appended calls pass a `**kwargs` no keyword and a `*args: T` one T:
    # Codon would refuse kw["speed"] and xs[1] for what every caller passes
    # (#266), so a function the scenario calls is compiled through its calls.
    tree = ast.parse(_VARIADIC)
    checks = render_library_checks(tree, callers=call_sites([tree]))
    calls = checks.source.splitlines()
    for called in ("opts(", "pair(", "Tuner(", "Tuner.scale("):
        assert not [c for c in calls if c.lstrip().startswith(called)], called
    # One nobody calls is still called, a call of its own not counting...
    assert "    unused_get()" in calls
    assert "    unused_args(_acs_value(float))" in calls
    assert "    recursive(_acs_value(float))" in calls
    # ...and so is one whose body does not read what its **kwargs holds: a
    # call is known by its name only, and gains.update() is dict's, not the
    # helper's, which nothing would compile if it were left to it.
    assert "    update()" in calls
    # ...unless it reads its **kwargs by key: no call without one compiles.
    assert not [c for c in calls if c.lstrip().startswith("unused(")]
    [(line, message)] = checks.problems
    assert line == 10
    assert message.startswith("unused: reads `**kw` by key"), message
    # The library check, which has no callers, calls them all as before.
    plain = render_library_checks(tree).source.splitlines()
    assert "    opts()" in plain
    assert "    pair(_acs_value(float))" in plain


def test_an_uncalled_name_belongs_to_the_longest_checked_module() -> None:
    # trajectory.model's wrapper is of trajectory.model, not of the
    # trajectory package that is checked too.
    package = "autoware_carla_scenario.trajectory"
    assert "_refusing_vertex_times" in _uncalled(f"{package}.model")
    assert not [n for n in _uncalled(package) if "_refusing_vertex_times" in n]


def test_the_driver_runs_the_calls_of_each_module() -> None:
    driver = render_library_driver([f"{_PACKAGE}.kinematics.angle"])
    assert (
        "from _acs_lib.autoware_carla_scenario__kinematics__angle import "
        "_acs_library_check as _acs_check_0"
    ) in driver.source
    assert "_acs_check_0()" in driver.source


# ---------------------------------------------------------------------------
# Compiling
# ---------------------------------------------------------------------------


@needs_codon
def test_the_checked_modules_compile() -> None:
    result = typecheck_library()
    assert result.skipped is None
    assert result.ok, result.format()


@needs_codon
def test_scenario_check_checks_the_library(capsys: pytest.CaptureFixture[str]) -> None:
    assert scenario_check(["--library", f"{_PACKAGE}.kinematics.angle"]) == 0
    assert "[ok]" in capsys.readouterr().out


def _dependencies(*names: str) -> list[str]:
    every = package_modules()
    sources = check._Sources()
    for name in names:
        sources.modules[name] = check._read_module(every[name])
    return check._library_dependencies(list(names), sources, every)


def test_a_named_module_brings_the_checked_modules_it_imports() -> None:
    # actions.turn imports find_pedestrian_entity, which only the checked
    # entity.registry defines (the model's entity.codon does not).
    deps = _dependencies(f"{_PACKAGE}.actions.turn")
    assert f"{_PACKAGE}.entity.registry" in deps
    # ... and, transitively, what entity.registry imports.
    assert f"{_PACKAGE}.entity.pedestrian_entity" in deps
    assert set(deps) <= set(CHECKED)


def test_a_modelled_import_is_not_followed() -> None:
    deps = _dependencies(f"{_PACKAGE}.entity.ego")
    assert f"{_PACKAGE}.scenario_base" not in deps  # MODELLED_IMPORTS
    # entity._spawn brings the rest of the vehicle entity's replaced model,
    # so the model re-exports the checked definitions, as in the full run.
    assert {f"{_PACKAGE}.entity._spawn", f"{_PACKAGE}.entity.vehicle_entity"} <= set(
        deps
    )


def test_the_full_run_compiles_only_the_checked_modules() -> None:
    assert _dependencies(*CHECKED) == []


@needs_codon
@pytest.mark.parametrize(
    "module",
    [
        "scenario_base",
        "entity.ego",
        "actions.turn",
        "actions.lane_change",
        # BridgedEgoEntity derives from EgoVehicle, which imports
        # scenario_base through the model, and AutowareEgoEntity from it.
        "entity.bridged_ego",
        "entity.autoware_entity",
        # The run starts at coordinate.frames, which the model's _poses.codon
        # re-exports: the driver must read the model first.
        "coordinate.poses",
    ],
)
def test_a_module_importing_a_checked_only_name_compiles_alone(module: str) -> None:
    result = typecheck_library([f"{_PACKAGE}.{module}"])
    assert result.ok, result.format()
    assert "more they import" in result.scenario


_FAKE = f"{_PACKAGE}.zz_library_check_case"

_GOOD = """
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Optional

from autoware_carla_scenario.coordinate.poses import Lanelet2Pose, OpenDrivePose
from .entity_role import EntityRole
from .kinematics.vector import Vector3


class Turn(str, Enum):
    LEFT = "left"
    RIGHT = "right"


class Level(Enum):
    LOW = auto()
    HIGH = auto()


class LevelError(ValueError):
    level: Level

    def __init__(self, level: Level) -> None:
        super().__init__(f"bad level {level.name}")
        self.level = level


class Thing:
    size: float

    def __init__(self, size: float) -> None:
        self.size = size

    @classmethod
    def unit(cls) -> Thing:
        return cls(1.0)

    @property
    def doubled(self) -> float:
        return self.size * 2

    def scaled(self, v: Vector3) -> Vector3:
        return v * self.size


def lane_s(pose: Lanelet2Pose | OpenDrivePose) -> float:
    return pose.s


@dataclass(frozen=True)
class Reader:
    read: Callable[[float], Optional[float]]


def read_twice(reader: Reader, f: Callable[[float], float]) -> Optional[float]:
    return reader.read(f(1.0))


def describe(turn: Turn, level: Level, role: EntityRole) -> str:
    if level == Level.HIGH:
        raise LevelError(level)
    return turn.value + str(role)
"""


def _check_case(
    source: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> TypeCheckResult:
    path = tmp_path / "zz_library_check_case.py"
    path.write_text(textwrap.dedent(source))
    modules = package_modules()
    modules[_FAKE] = path
    monkeypatch.setattr(check, "package_modules", lambda: modules)
    return typecheck_library(
        [_FAKE, f"{_PACKAGE}.entity_role", f"{_PACKAGE}.kinematics.vector"]
    )


@needs_codon
def test_enums_exceptions_classmethods_and_unions_compile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _check_case(_GOOD, tmp_path, monkeypatch)
    assert result.ok, result.format()


@needs_codon
@pytest.mark.parametrize(
    ("wrong", "right"),
    [
        # A body of the wrong type, in each kind of callable.
        ("return turn.value + str(role)", "return level.value"),
        ("return cls(1.0)", "return cls('one')"),
        ("return self.size * 2", "return self.size.real()"),
        # Only one member of the union has the attribute.
        ("return pose.s", "return pose.road_id"),
    ],
    ids=["function", "classmethod", "property", "union-member"],
)
def test_a_wrong_module_is_refused_at_its_line(
    wrong: str, right: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _GOOD.replace(wrong, right)
    assert source != _GOOD
    result = _check_case(source, tmp_path, monkeypatch)
    assert not result.ok
    stripped = [text.strip() for text in textwrap.dedent(source).splitlines()]
    line = stripped.index(right) + 1
    located = [d for d in result.diagnostics if d.path and d.path.endswith(".py")]
    assert located, result.format()
    assert located[0].path == str(tmp_path / "zz_library_check_case.py")
    assert located[0].line == line, result.format()


_PROTECTED = """
from __future__ import annotations


class Gauge:
    size: float

    def __init__(self, size: float) -> None:
        self.size = size

    def _scaled(self, k: float) -> float:
        return self.size * k


def _doubled(x: float) -> float:
    return x * 2.0


class _Meter:
    reading: float

    def __init__(self, reading: float) -> None:
        self.reading = reading

    def offset(self, k: float) -> float:
        return self.reading - k
"""


@needs_codon
def test_protected_functions_and_methods_compile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _check_case(_PROTECTED, tmp_path, monkeypatch)
    assert result.ok, result.format()


@needs_codon
@pytest.mark.parametrize(
    ("right", "wrong", "realized"),
    [
        ("return self.size * k", "return self.size + str(k)", "_scaled"),
        ("return x * 2.0", "return x + 'two'", "_doubled"),
        ("return self.reading - k", "return self.reading - str(k)", "offset"),
    ],
    ids=["method", "function", "private-class-method"],
)
def test_an_error_in_a_protected_function_or_method_is_reported(
    right: str,
    wrong: str,
    realized: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Nothing public calls _scaled(), _doubled() or _Meter.offset(): only the
    # check's own call compiles them, so an error in any is found only
    # because of it.
    source = _PROTECTED.replace(right, wrong)
    result = _check_case(source, tmp_path, monkeypatch)
    assert not result.ok
    stripped = [text.strip() for text in textwrap.dedent(source).splitlines()]
    located = [d for d in result.diagnostics if d.path and d.path.endswith(".py")]
    assert located, result.format()
    assert located[0].path == str(tmp_path / "zz_library_check_case.py")
    assert located[0].line == stripped.index(wrong) + 1, result.format()
    assert realized in result.format()


_DEFAULTS = """
from __future__ import annotations

from typing import Callable, Collection, Dict, List, Optional, Sequence

from autoware_carla_scenario.kinematics.vector import Vector3


def nearest(x: float, max_distance: float = 150.0) -> float:
    return x + max_distance


class Route:
    _waypoints: List[Vector3]

    def __init__(self, goal: Vector3, waypoints: Sequence[Vector3] = ()) -> None:
        self._waypoints = list(waypoints)


class Launch:
    _args: Sequence[str]

    def __init__(self, args: Sequence[str] = ("--symlink-install", "-v")) -> None:
        self._args = args


def hold(ids: List[int], spare: Collection[int] = ()) -> int:
    return len([i for i in ids if i not in spare])


def first(poses: Optional[Sequence[Vector3]] = None) -> int:
    kept: List[Vector3] = []
    if poses:
        kept = list(poses)
    return len(kept)


def tally(ids: List[int] = [], names: Dict[str, Vector3] = {}) -> float:
    return sum(ids) + sum(v.x for v in names.values())


class Box:
    def add(self, more: List[Vector3] = [], *, by: Dict[str, float] = {}) -> int:
        return len(more) + len(by)


def height(ground: Optional[Callable[[float], Optional[float]]] = None) -> float:
    z: Optional[float] = None
    if ground is not None:
        z = ground(1.0)
    if z is None:
        return 0.0
    return z
"""


@needs_codon
def test_defaults_compile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The calls without the defaults compile a `()` of a Sequence[T] (an empty
    # List[T]), a `None` of an Optional the transform drops, and an empty
    # `[]` / `{}`, which Codon would make a list of None (#183).  A tuple
    # display of a Sequence[T] is a List[T] too, so it can be kept in a
    # Sequence[T] attribute (#188).
    result = _check_case(_DEFAULTS, tmp_path, monkeypatch)
    assert result.ok, result.format()


@needs_codon
def test_a_default_of_the_wrong_type_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every other call passes `max_distance`: only the one leaving it out
    # compiles the default.
    source = _DEFAULTS.replace(
        "max_distance: float = 150.0", 'max_distance: float = "far"'
    )
    assert source != _DEFAULTS
    result = _check_case(source, tmp_path, monkeypatch)
    assert not result.ok
    assert "with the default of `max_distance`" in result.format(), result.format()


_NONE_MISUSED = """
from __future__ import annotations

from typing import Optional


def reading(x: float) -> Optional[float]:
    return x if x > 0.0 else None


def halved(x: float) -> float:
    value: float = reading(x)
    return value / 2.0
"""


@needs_codon
def test_a_none_misused_in_a_checked_module_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Codon unwraps the Optional, so the compile accepts it; the None check
    # the scenario check runs is run over the checked modules too (#277).
    result = _check_case(_NONE_MISUSED, tmp_path, monkeypatch)
    assert not result.ok
    located = [d for d in result.diagnostics if d.path and d.path.endswith(".py")]
    assert [(d.path, d.line) for d in located] == [
        (str(tmp_path / "zz_library_check_case.py"), 12)
    ], result.format()
    assert "test the value for None" in located[0].message


def test_a_none_check_that_did_not_run_warns_and_is_not_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Without Codon: the compile passes, mypy has no verdict.  The library
    # check keeps Codon's verdict with a warning, and does not keep it beyond
    # the process, so the next run gives the None check another go (#277).
    from autoware_carla_scenario.typecheck.none_check import (  # noqa: PLC0415
        NoneCheckSkipped,
    )

    monkeypatch.setenv(check.CACHE_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(check, "_CACHE", {})
    monkeypatch.setattr(check, "codon_path_dir", lambda: tmp_path)
    monkeypatch.setattr(check, "_toolchain_digest", lambda tc: "tc")
    monkeypatch.setattr(check, "_framework_digest", lambda codon_path: "fw")
    monkeypatch.setattr(check, "_library_imports", lambda *args: None)

    def compile_(name: str, *args: object, **kwargs: object) -> TypeCheckResult:
        return TypeCheckResult(name, ok=True)

    def no_mypy(paths: object, timeout: float) -> object:
        raise NoneCheckSkipped("mypy is not installed")

    monkeypatch.setattr(check, "_compile", compile_)
    monkeypatch.setattr(check, "run_none_check", no_mypy)
    result = typecheck_library(
        [f"{_PACKAGE}.kinematics.vector"],
        toolchain=SimpleNamespace(),  # type: ignore[arg-type]
    )

    assert result.ok, result.format()
    assert [w.message for w in result.warnings] == [
        "the None check did not run (mypy is not installed), so an Optional "
        "used without a test for None is not reported"
    ]
    assert not [p for p in tmp_path.rglob("*") if p.is_file()]


_INIT_AND_SETTER = """
from __future__ import annotations

from typing import TYPE_CHECKING

from autoware_carla_scenario.kinematics.vector import Vector3


class Holder:
    kind: type[Vector3]
    _size: float
    _scale: float

    def __init__(self, size: float, kind: type[Vector3] | None = None) -> None:
        self._size = size
        if not TYPE_CHECKING:
            self.kind = kind or Vector3
        self._scale = 2.0


class Gauge:
    _level: float

    def __init__(self) -> None:
        self._level = 0.0

    @property
    def level(self) -> float:
        return self._level

    @level.setter
    def level(self, value: float | int) -> None:
        self._level = float(value)
"""


@needs_codon
def test_a_constructor_left_to_its_default_and_a_setter_compile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _check_case(_INIT_AND_SETTER, tmp_path, monkeypatch)
    assert result.ok, result.format()


@needs_codon
@pytest.mark.parametrize(
    ("right", "wrong", "label"),
    [
        (
            "self._scale = 2.0",
            'self._scale = "two"',
            "Holder.__init__(size: float), with the default of `kind`",
        ),
        (
            "self._level = float(value)",
            "self._level = str(value)",
            "Gauge.level (setter)",
        ),
    ],
    ids=["constructor-left-to-default", "setter"],
)
def test_an_error_in_a_constructor_left_to_its_default_or_a_setter_is_reported(
    right: str,
    wrong: str,
    label: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Nothing calls Holder() or assigns Gauge.level: only the check's own
    # calls compile them (#199, #200).
    source = _INIT_AND_SETTER.replace(right, wrong)
    assert source != _INIT_AND_SETTER
    result = _check_case(source, tmp_path, monkeypatch)
    assert not result.ok
    stripped = [text.strip() for text in textwrap.dedent(source).splitlines()]
    located = [d for d in result.diagnostics if d.path and d.path.endswith(".py")]
    assert located, result.format()
    assert located[0].path == str(tmp_path / "zz_library_check_case.py")
    assert located[0].line == stripped.index(wrong) + 1, result.format()
    assert f"<library check: {label}>" in result.format(), result.format()


_FIELD_DEFAULTS = """
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from autoware_carla_scenario.kinematics.vector import Vector3


@dataclass
class Launch:
    workspace: str
    colcon_args: Sequence[str] = ("--symlink-install", "--merge-install")
    one: Sequence[str] = ("a",)
    sizes: tuple[float, ...] = field(default=(1.0, 2.0))
    origin: Vector3 = field(default_factory=lambda: Vector3(0.0, 0.0, 0.0))
    env: Dict[str, str] = field(default_factory=dict)
    max_distance: float = 150.0

    def arguments(self) -> List[str]:
        return [self.workspace] + list(self.colcon_args) + list(self.one)
"""


@needs_codon
def test_field_defaults_compile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The constructor call passing only the required fields compiles each
    # field default against its field's type: a tuple display of a
    # `Sequence[str]` or `tuple[float, ...]` field is a list of it (#188).
    result = _check_case(_FIELD_DEFAULTS, tmp_path, monkeypatch)
    assert result.ok, result.format()


@needs_codon
def test_a_field_default_of_the_wrong_type_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every other constructor call passes `max_distance`: only the one
    # passing the required fields alone compiles the default.
    source = _FIELD_DEFAULTS.replace(
        "max_distance: float = 150.0", 'max_distance: float = "far"'
    )
    assert source != _FIELD_DEFAULTS
    result = _check_case(source, tmp_path, monkeypatch)
    assert not result.ok
    assert "Launch.__init__(workspace: str), with the default of" in (
        result.format()
    ), result.format()


_CLASS_DEFAULTS = """
from __future__ import annotations

from typing import ClassVar, List, Optional

from autoware_carla_scenario.kinematics.vector import Vector3

DEFAULT_PORT = 8000


class Action:
    REISSUES: ClassVar[bool] = False
    port: int = DEFAULT_PORT
    scale: float = 1
    origin: Optional[Vector3] = None
    names: List[str] = []
    ready: bool = True

    def __init__(self) -> None:
        pass

    def describe(self) -> str:
        return str(self.port) + str(self.scale) + str(self.ready)


class Follow(Action):
    REISSUES = True
    port = 9000

    def __init__(self) -> None:
        super().__init__()


def tuples(xs: List[int]) -> tuple[int, ...]:
    return tuple(xs)
"""


@needs_codon
def test_class_level_defaults_and_tuple_returns_compile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Each class-level default is compiled against its annotation (#190), an
    # override without one against its base's, and a `tuple[X, ...]` return
    # is checked as a List[X] (#191).
    result = _check_case(_CLASS_DEFAULTS, tmp_path, monkeypatch)
    assert result.ok, result.format()


@pytest.mark.parametrize(
    ("right", "wrong", "reported"),
    [
        (
            "port: int = DEFAULT_PORT",
            'port: int = "x"',
            "class-level default of Action.port",
        ),
        (
            "origin: Optional[Vector3] = None",
            "origin: Optional[Vector3] = 1.0",
            "class-level default of Action.origin",
        ),
        # Codon converts any value assigned to a `bool`: the check calls.
        ("ready: bool = True", 'ready: bool = "x"', "Action.ready"),
        ("    REISSUES = True", '    REISSUES = "x"', "Follow.REISSUES"),
        ("    port = 9000", '    port = "x"', "Follow.port"),
        # Nothing uses what `tuples` returns: only its annotation checks it.
        (
            "    return tuple(xs)",
            "    return 1",
            "'List[int]' does not match expected type 'int'",
        ),
    ],
)
@needs_codon
def test_a_wrong_class_level_default_or_tuple_return_is_reported(
    right: str,
    wrong: str,
    reported: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Before #190 and #191, each of these compiled: Codon dropped the default
    # of a class with a constructor, and the check the tuple return.
    assert _CLASS_DEFAULTS.count(right) == 1
    result = _check_case(_CLASS_DEFAULTS.replace(right, wrong), tmp_path, monkeypatch)
    assert not result.ok
    assert reported in result.format(), result.format()


@needs_codon
def test_an_abstract_base_class_and_its_subclass_compile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = """
from __future__ import annotations

from abc import ABC, abstractmethod


class Bridge(ABC):
    @abstractmethod
    def is_ready(self) -> bool:
        \"\"\"Whether it is ready.\"\"\"

    def close(self) -> None:
        pass


class Fake(Bridge):
    polls: int = 0

    def is_ready(self) -> bool:
        self.polls += 1
        return self.polls > 1
"""
    result = _check_case(source, tmp_path, monkeypatch)
    assert result.ok, result.format()


@needs_codon
def test_pyxodrs_road_network_is_the_one_map_manager_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # utils/traffic_light.py annotates with pyxodr's RoadNetwork, since utils
    # may not import coordinate; the pyxodr shim names the model's class, so
    # the network MapManager returns goes into it.
    source = """
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from autoware_carla_scenario.coordinate.map_manager import MapManager

if TYPE_CHECKING:
    from pyxodr.road_objects.network import RoadNetwork


def first_controller_name(road_network: RoadNetwork) -> Optional[str]:
    for element in road_network.root.iter("controller"):
        return element.get("name")
    return None


def ambient_controller_name() -> Optional[str]:
    return first_controller_name(MapManager.get_instance().road_network)
"""
    result = _check_case(source, tmp_path, monkeypatch)
    assert result.ok, result.format()


_SINGLETON_AND_UNION_CALLABLE = """
from __future__ import annotations

from typing import Callable, ClassVar, Optional, Union

from autoware_carla_scenario.entity_role import EntityRole
from autoware_carla_scenario.kinematics.vector import Vector3

#: A callable alias, which Codon 0.19 refuses as an expression.
Lookup = Callable[[Optional[Union[EntityRole, str]]], Vector3]


class Single:
    _instance: ClassVar[Optional["Single"]] = None
    size: Optional[float]

    def __new__(cls) -> "Single":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.size = None
        return cls._instance

    def __init__(self) -> None:
        pass


def length(
    lookup: Callable[[Optional[Union[EntityRole, str]]], Vector3],
) -> float:
    return lookup(EntityRole.ego()).dot(lookup(None)) + (Single().size or 0.0)
"""


@needs_codon
def test_a_singleton_and_a_callable_of_a_union_compile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # super().__new__(cls) allocates the class, and a callable with a union
    # argument is called with one of any arguments returning its result type.
    right = _SINGLETON_AND_UNION_CALLABLE
    result = _check_case(right, tmp_path, monkeypatch)
    assert result.ok, result.format()
    # What the callable returns is typed: a Vector3 has no norm_of().
    wrong = right.replace(".dot(lookup(None))", ".norm_of(lookup(None))")
    assert wrong != right
    assert not _check_case(wrong, tmp_path, monkeypatch).ok
    # Nor does a call with another number of arguments than the Callable's.
    arity = right.replace(".dot(lookup(None))", ".dot(lookup(None, None))")
    assert arity != right
    assert not _check_case(arity, tmp_path, monkeypatch).ok


@needs_codon
def test_a_module_importing_what_codon_cannot_compile_is_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _check_case(
        "import yaml\n\n\ndef f() -> int:\n    return 1\n", tmp_path, monkeypatch
    )
    assert not result.ok
    assert result.diagnostics[0].line == 1
    assert "yaml has no Codon model" in result.diagnostics[0].message


@needs_codon
def test_an_unknown_module_is_refused() -> None:
    result = typecheck_library([f"{_PACKAGE}.no_such_module"])
    assert not result.ok
    assert "not a module" in result.diagnostics[0].message


_ABSTRACT = """
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Shape(ABC):
    @abstractmethod
    def area(self) -> float: ...

    def summary(self) -> dict[str, Any]:
        import textwrap  # noqa: PLC0415

        return {"area": self.area(), "kind": textwrap.dedent("shape")}


class Square(Shape):
    side: float

    def __init__(self, side: float) -> None:
        self.side = side

    def area(self) -> float:
        return self.side * self.side


def total(shape: Shape) -> float:
    return shape.area()
"""


@needs_codon
def test_abc_and_an_uncalled_method_compile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Without the entry, summary() is called: textwrap is reported.
    result = _check_case(_ABSTRACT, tmp_path, monkeypatch)
    assert not result.ok
    assert "textwrap has no Codon model" in result.diagnostics[0].message
    # With it, neither its import nor its mixed dict reaches Codon.
    monkeypatch.setitem(check.UNCALLED, f"{_FAKE}.Shape.summary", "uses textwrap")
    result = _check_case(_ABSTRACT, tmp_path, monkeypatch)
    assert result.ok, result.format()


_ELSE_IMPORT = """
from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass
else:
    from tqdm import tqdm


def wait(steps: int) -> None:
    if TYPE_CHECKING:
        for _ in range(steps):
            time.sleep(0.1)
    else:
        import textwrap  # noqa: PLC0415

        for _ in tqdm(range(steps), desc=textwrap.dedent("wait")):
            time.sleep(0.1)
"""


@needs_codon
def test_an_import_in_the_else_branch_of_type_checking_is_not_compiled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Codon reads TYPE_CHECKING as a compile-time true and compiles only the
    # first branch, so what the `else` branch imports needs no model (#126).
    result = _check_case(_ELSE_IMPORT, tmp_path, monkeypatch)
    assert result.ok, result.format()


def test_an_import_in_the_first_branch_of_type_checking_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    swapped = _ELSE_IMPORT.replace(
        "    pass\nelse:\n    from tqdm import tqdm",
        "    from tqdm import tqdm\nelse:\n    pass",
    )
    assert swapped != _ELSE_IMPORT
    result = _check_case(swapped, tmp_path, monkeypatch)
    assert not result.ok
    assert "tqdm has no Codon model" in result.diagnostics[0].message


_NOT_TYPE_CHECKING_IMPORT = """
from __future__ import annotations

import time
from typing import TYPE_CHECKING

if not TYPE_CHECKING:
    from tqdm import tqdm


def wait(steps: int) -> None:
    if not TYPE_CHECKING:
        import textwrap  # noqa: PLC0415

        for _ in tqdm(range(steps), desc=textwrap.dedent("wait")):
            time.sleep(0.1)
    else:
        for _ in range(steps):
            time.sleep(0.1)
"""


def test_the_body_of_not_type_checking_is_run_time_only() -> None:
    # `if not TYPE_CHECKING:` is the `else` branch of `if TYPE_CHECKING:`
    # written the other way round: Codon compiles its `else`, not its body.
    tree = ast.parse(_NOT_TYPE_CHECKING_IMPORT)
    skipped = {
        node.names[0].name
        for node in check._run_time_only_nodes(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    }
    assert skipped == {"tqdm", "textwrap"}
    compiled = {
        id(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "sleep"
    }
    run_time_only = {id(node) for node in check._run_time_only_nodes(tree)}
    # The loop of the body is run-time-only; the one of the `else` is compiled.
    assert len(compiled - run_time_only) == 1


@pytest.mark.parametrize(
    ("source", "run_time_only"),
    [
        # An `elif not TYPE_CHECKING:` is an `if` of its own.
        (
            "if x:\n    import b\nelif not TYPE_CHECKING:\n    import a\n"
            "else:\n    import b2\n",
            {"a"},
        ),
        ("if not (TYPE_CHECKING):\n    import a\nelse:\n    import b\n", {"a"}),
        (
            "import typing\nif not typing.TYPE_CHECKING:\n    import a\n"
            "else:\n    import b\n",
            {"a"},
        ),
        (
            "if not TYPE_CHECKING:\n    if x:\n        import a\n    else:\n"
            "        import a2\nelse:\n    if not TYPE_CHECKING:\n"
            "        import a3\n    import b\n",
            {"a", "a2", "a3"},
        ),
        # What the check cannot read as TYPE_CHECKING alone hides nothing.
        ("if TYPE_CHECKING or x:\n    import b\nelse:\n    import b2\n", set()),
        ("if not TYPE_CHECKING and x:\n    import b\nelse:\n    import b2\n", set()),
        ("if not not TYPE_CHECKING:\n    import b\nelse:\n    import b2\n", set()),
        (
            "import typing as t\nif not t.TYPE_CHECKING:\n    import b\n"
            "else:\n    import b2\n",
            set(),
        ),
    ],
)
def test_only_code_codon_never_compiles_is_run_time_only(
    source: str, run_time_only: set[str]
) -> None:
    # Reading check-time code as run-time-only would hide its imports: where
    # the check cannot tell, it reports.
    skipped = {
        node.names[0].name
        for node in check._run_time_only_nodes(ast.parse(source))
        if isinstance(node, ast.Import)
    }
    assert skipped == run_time_only


@needs_codon
def test_an_import_in_the_body_of_not_type_checking_is_not_compiled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _check_case(_NOT_TYPE_CHECKING_IMPORT, tmp_path, monkeypatch)
    assert result.ok, result.format()


def test_an_import_in_the_else_branch_of_not_type_checking_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    swapped = _NOT_TYPE_CHECKING_IMPORT.replace(
        "if not TYPE_CHECKING:\n    from tqdm import tqdm",
        "if not TYPE_CHECKING:\n    pass\nelse:\n    from tqdm import tqdm",
    )
    assert swapped != _NOT_TYPE_CHECKING_IMPORT
    result = _check_case(swapped, tmp_path, monkeypatch)
    assert not result.ok
    assert "tqdm has no Codon model" in result.diagnostics[0].message


# ---------------------------------------------------------------------------
# What the traffic backends need (#79)
# ---------------------------------------------------------------------------


_OVERRIDDEN = """
from __future__ import annotations

from typing import Any


class Backend:
    def start(self, world: int, *, skip: list[int]) -> None:
        pass

    def turn(self, world: int, **kwargs: Any) -> None:
        pass

    def tick(self, world: int) -> None:
        pass
"""


def test_a_method_with_keyword_only_parameters_is_called_through_its_class() -> None:
    # Codon 0.19 calls a method a subclass may override through a dispatch
    # thunk, which takes neither a keyword argument nor **kwargs.
    calls = render_library_checks(ast.parse(_OVERRIDDEN)).source.splitlines()
    assert (
        "    Backend.start(_acs_value(Backend), _acs_value(int), "
        "skip=_acs_value(list[int]))"
    ) in calls
    assert "    Backend.turn(_acs_value(Backend), _acs_value(int))" in calls
    assert "    _acs_value(Backend).tick(_acs_value(int))" in calls


def test_an_import_inside_an_uncalled_function_is_left_as_written() -> None:
    source = """
        def build(options: dict) -> None:
            from autoware_carla_scenario.utils.config import load
        """
    out, problems = redirect_imports(
        textwrap.dedent(source),
        f"{_PACKAGE}.kinematics.velocity",
        False,
        _target,
        _MODULES.__contains__,
        uncalled=["build"],
    )
    assert problems == []
    assert "from autoware_carla_scenario.utils.config import load" in out


_BACKENDS = """
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import typesafe_carla.carla as carla

logger = logging.getLogger(__name__)


@runtime_checkable
class Steering(Protocol):
    def steer(self, world: carla.World) -> None:
        ...


class Backend:
    def start(self, world: carla.World, *, skip_actor_ids: list[int] = []) -> None:
        pass

    @classmethod
    def from_options(cls, options: dict[str, int]) -> Backend:
        from autoware_carla_scenario.utils.config import checked_options

        checked_options(cls, options)
        return cls()


class Manager(Backend):
    def start(self, world: carla.World, *, skip_actor_ids: list[int] = []) -> None:
        try:
            world.tick()
        except RuntimeError:
            logger.warning("tick failed", exc_info=True)


def vehicle(actor: carla.Actor) -> carla.Vehicle:
    if TYPE_CHECKING:
        return actor.as_vehicle()
    else:
        return actor
"""


@needs_codon
def test_overridden_methods_protocols_and_type_checking_compile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Each override is called by keyword, a framework module with no model is
    # imported only in an uncalled method, a protocol is a declaration, and
    # only the TYPE_CHECKING branch of vehicle() is compiled.
    monkeypatch.setitem(
        check.UNCALLED, f"{_FAKE}.Backend.from_options", "uses utils.config"
    )
    result = _check_case(_BACKENDS, tmp_path, monkeypatch)
    assert result.ok, result.format()


# ---------------------------------------------------------------------------
# The queue shim (the sensors' frame queues)
# ---------------------------------------------------------------------------


_QUEUES = """
from __future__ import annotations

import queue
from typing import Optional


class Frames:
    _frames: queue.Queue[int]
    _unbounded: queue.Queue[tuple[int, str]]

    def __init__(self) -> None:
        self._frames = queue.Queue(maxsize=2)
        self._unbounded = queue.Queue()

    def receive(self, frame: int) -> None:
        self._frames.put(frame)
        self._frames.put(frame, True, 1.0)
        self._frames.put(frame, block=False, timeout=None)
        self._frames.put(frame, True, 1)
        self._frames.put(frame, timeout=2)
        try:
            self._frames.put_nowait(frame)
        except queue.Full:
            pass
        self._unbounded.put((frame, "frame"))

    def newest(self, timeout: float) -> Optional[int]:
        try:
            frame = self._frames.get(timeout=timeout)
        except queue.Empty:
            return None
        while True:
            try:
                frame = self._frames.get_nowait()
            except queue.Empty:
                break
        self._frames.get(True, timeout)
        self._frames.get(block=False)
        self._frames.get(timeout=1)
        return frame

    def size(self) -> int:
        if self._frames.empty() or self._frames.full():
            return 0
        return self._frames.qsize() + queue.Queue[int](0).qsize()
"""


@needs_codon
def test_the_queue_shim_takes_what_cpython_takes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _check_case(_QUEUES, tmp_path, monkeypatch)
    assert result.ok, result.format()


@needs_codon
@pytest.mark.parametrize(
    ("right", "wrong"),
    [
        # Each is a TypeError or an AttributeError in CPython too.
        ("queue.Queue(maxsize=2)", "queue.Queue(size=2)"),
        ("self._frames.get_nowait()", "self._frames.get_nowait(True)"),
        ("self._frames.get(True, timeout)", "self._frames.get(True, timeout, 1)"),
        ("self._frames.qsize() + ", "self._frames.peek() + "),
        ("self._frames.get(timeout=1)", 'self._frames.get(timeout="1")'),
    ],
    ids=[
        "unknown-keyword",
        "get_nowait-argument",
        "get-arguments",
        "no-method",
        "timeout-type",
    ],
)
def test_the_queue_shim_refuses_what_cpython_refuses(
    right: str, wrong: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _QUEUES.replace(right, wrong, 1)
    assert source != _QUEUES
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


# ---------------------------------------------------------------------------
# The pathlib and urllib.parse shims (#97)
# ---------------------------------------------------------------------------


_PATHLIB = """
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Run:
    xodr_path: Optional[Path] = None
    output_dir: Path = field(default_factory=lambda: Path("outputs"))


def cache_dir(configured: str) -> Path:
    if configured:
        return Path(configured).expanduser().resolve(strict=False)
    return Path.home() / ".cache" / "sumo"


def log_file(run: Run) -> str:
    return str(run.output_dir / "sumo" / "fcd.xml")


def log_text(run: Run) -> str:
    return (run.output_dir / "fcd.xml").read_text(encoding="utf-8")


def is_default(run: Run, configured: str) -> bool:
    # As in Python: a str on the left of `/`, a Path of no parts, and a Path
    # compared with a str (never equal).
    here = "outputs" / Path()
    return run.output_dir == here and run.output_dir != configured
"""


@needs_codon
def test_the_pathlib_shim_builds_joins_and_prints_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert (model_dir() / "pathlib.codon").is_file()
    result = _check_case(_PATHLIB, tmp_path, monkeypatch)
    assert result.ok, result.format()
    # A Path is not a str: the shim is typed, not a pass-through.
    wrong = _PATHLIB.replace(
        'return str(run.output_dir / "sumo" / "fcd.xml")',
        'return run.output_dir / "sumo" / "fcd.xml"',
    )
    assert wrong != _PATHLIB
    assert not _check_case(wrong, tmp_path, monkeypatch).ok
    # Python builds and joins a Path with a str or an os.PathLike only.
    for part in ("Path(configured, 1)", "(Path(configured) / 1)"):
        wrong = _PATHLIB.replace("Path(configured)", part, 1)
        assert wrong != _PATHLIB
        result = _check_case(wrong, tmp_path, monkeypatch)
        assert not result.ok
        assert "a str or an os.PathLike only" in result.format(), result.format()


_URLLIB = """
from __future__ import annotations

from typing import Optional
from urllib.parse import urlsplit, urlunsplit


def repository(url: str) -> str:
    parts = urlsplit(url)
    if not parts.scheme or not parts.netloc:
        return url
    segments = [s for s in parts.path.split("/") if s]
    return urlunsplit((parts.scheme, parts.netloc, "/" + "/".join(segments), "", ""))


def host(url: str) -> Optional[str]:
    # Python's SplitResult: a 5-tuple of str with the netloc's parts.
    parts = urlsplit(url, scheme="https")
    scheme, netloc, path, query, fragment = parts
    if parts.port is None and parts[0] == scheme:
        return parts.hostname
    return urlunsplit(parts) + parts.geturl()
"""


@needs_codon
def test_the_urllib_parse_shim_splits_and_joins_urls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert (model_dir() / "urllib" / "parse.codon").is_file()
    result = _check_case(_URLLIB, tmp_path, monkeypatch)
    assert result.ok, result.format()
    # SplitResult's parts are typed: a part Python's lacks is refused.
    wrong = _URLLIB.replace("parts.netloc:", "parts.host:")
    assert wrong != _URLLIB
    assert not _check_case(wrong, tmp_path, monkeypatch).ok


# ---------------------------------------------------------------------------
# The file-system pathlib calls, hashlib, and str.encode (#116)
# ---------------------------------------------------------------------------


_FILE_SYSTEM = """
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional


def maps_in(directory: Path) -> list[Path]:
    return sorted(p for p in directory.glob("*.osm") if p.is_file())


def first_xodr(directory: Path) -> Optional[Path]:
    for candidate in sorted(directory.glob(pattern="*.xodr")):
        if candidate.exists() and not candidate.is_dir():
            found: Optional[Path] = candidate
            return found
    return None


def inside(root: Path, candidate: Path) -> bool:
    real = candidate.absolute().resolve()
    return real.parents[0] <= root or real == root or root in real.parents


def describe(path: Path) -> str:
    names = [parent.name for parent in path.parents]
    suffix = path.suffix
    return f"{path.name}{suffix} in {path.parent} ({len(path.parents)} {names})"


def digest(text: str) -> str:
    key = hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]
    h = hashlib.sha256(usedforsecurity=False)
    h.update(text.encode(errors="strict"))
    h.update(b"-")
    other = hashlib.sha256(string=b"x").copy()
    size = other.digest_size + other.block_size
    return key + h.hexdigest() + other.name + str(size + len(h.digest()))
"""


@needs_codon
def test_the_file_system_shims_take_what_cpython_takes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert (model_dir() / "hashlib.codon").is_file()
    result = _check_case(_FILE_SYSTEM, tmp_path, monkeypatch)
    assert result.ok, result.format()


@needs_codon
@pytest.mark.parametrize(
    ("right", "wrong"),
    [
        # Each is a TypeError or an AttributeError in CPython 3.10, the oldest
        # Python the package supports (glob() takes `case_sensitive` from 3.12
        # on, sha256() `data=` from 3.13 on), except chmod(): it writes, and
        # the shim declares nothing that writes.
        ('directory.glob("*.osm")', 'directory.glob("*.osm", case_sensitive=True)'),
        ("candidate.exists()", "candidate.exists(True)"),
        ("suffix = path.suffix", "suffix = path.suffix()"),
        ("real.parents[0] <= root", 'real.parents[0] <= "/"'),
        ("candidate.absolute()", "candidate.chmod(420)"),
        (".hexdigest()[:10]", ".hexdigest(10)[:10]"),
        ("hashlib.sha256(usedforsecurity=False)", 'hashlib.sha256(b"", False)'),
        ('h.update(b"-")', 'h.update(data=b"-")'),
        ('hashlib.sha256(string=b"x")', 'hashlib.sha256(data=b"x")'),
        ('text.encode("utf-8")', 'text.encode("utf-8", "strict", 1)'),
        ('text.encode(errors="strict")', "text.encode(encoding=1)"),
        ('text.encode("utf-8")', 'text.decode("utf-8")'),
    ],
    ids=[
        "glob-keyword",
        "exists-argument",
        "suffix-is-a-property",
        "path-ordered-with-str",
        "no-method",
        "hexdigest-argument",
        "usedforsecurity-keyword-only",
        "update-positional-only",
        "sha256-keyword",
        "encode-arguments",
        "encode-encoding-type",
        "str-has-no-decode",
    ],
)
def test_the_file_system_shims_refuse_what_cpython_refuses(
    right: str, wrong: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _FILE_SYSTEM.replace(right, wrong, 1)
    assert source != _FILE_SYSTEM
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


_STEM = """
from __future__ import annotations

from pathlib import Path


def numbered(path: Path, counter: int) -> Path:
    return path.with_name(f"{path.stem}_{counter}{path.suffix}")
"""


@needs_codon
@pytest.mark.parametrize(
    ("right", "wrong"),
    [
        ("", ""),
        # A property in CPython, and with_name() takes the name: each is a
        # TypeError in CPython 3.10.
        ("{path.stem}", "{path.stem()}"),
        ("path.with_name(f", "path.with_name(1, f"),
    ],
    ids=["right", "stem-is-a-property", "with-name-takes-one-name"],
)
def test_the_pathlib_shim_has_the_stem_and_with_name(
    right: str, wrong: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # scenario_runner numbers a run's video after the file it would replace (#126).
    source = _STEM.replace(right, wrong, 1) if right else _STEM
    result = _check_case(source, tmp_path, monkeypatch)
    assert result.ok is (not right), result.format()


# ---------------------------------------------------------------------------
# The os and shutil shims, dict.get(key), and the missing exceptions (#117)
# ---------------------------------------------------------------------------


_OS_SHUTIL = """
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional


class MissingTool(FileNotFoundError):
    def __init__(self, message: str) -> None:
        super().__init__(message)


def configured(name: str) -> Optional[Path]:
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    return Path(value).expanduser()


def environment(name: str) -> str:
    unset: Optional[str] = os.environ.get(name)
    fallback = os.getenv(name, "fallback")
    maybe: Optional[str] = os.getenv(key=name)
    os.environ[name] = fallback
    env = os.environ.copy()
    names = [key for key in os.environ if key.startswith("ACS_")]
    return f"{unset} {maybe} {len(env)} {os.pathsep.join(names)}"


def lookup(table: dict[str, int], key: str) -> Optional[int]:
    return table.get(key)


def processes(pgid: int) -> int:
    os.killpg(os.getpgid(os.getpid()), 15)
    os.kill(pgid, 9)
    return os.system("true")


def install(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("x", encoding="utf-8")
    shutil.copy2(source, destination)
    copied: str = shutil.copy2(str(source), "target.xodr", follow_symlinks=False)
    moved = shutil.move(src=source, dst=destination)
    shutil.rmtree(destination.parent, ignore_errors=True)
    found: Optional[str] = shutil.which("git")
    joined = os.path.join(destination, "a")
    return f"{copied} {Path(moved)} {found} {joined}"


def guarded(path: Path) -> str:
    try:
        return path.name
    except ModuleNotFoundError as exc:
        return str(exc)
    except ImportError:
        raise MissingTool("no tool")
    except PermissionError:
        raise TimeoutError("slow")
    except BrokenPipeError:
        raise ProcessLookupError(path)
    except ConnectionError as exc:
        raise OSError(str(exc))
"""


@needs_codon
def test_the_os_and_shutil_shims_take_what_cpython_takes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert (model_dir() / "os" / "__init__.codon").is_file()
    assert (model_dir() / "shutil.codon").is_file()
    result = _check_case(_OS_SHUTIL, tmp_path, monkeypatch)
    assert result.ok, result.format()


@needs_codon
@pytest.mark.parametrize(
    ("right", "wrong"),
    [
        # Each is a TypeError or an AttributeError in CPython 3.10, the oldest
        # Python the package supports (rmtree() takes `onexc` from 3.12 on).
        ('os.environ.get(name, "")', 'os.environ.get(name, "", "")'),
        ("os.environ.copy()", 'os.environ.fromkeys(["a"])'),
        ("os.environ[name] = fallback", "os.environ[name] = len(fallback)"),
        ("os.getenv(key=name)", "os.getenv(name=name)"),
        ("table.get(key)", "table.get(key=key)"),
        ("os.kill(pgid, 9)", "os.kill(pid=pgid, signal=9)"),
        ("os.getpgid(os.getpid())", "os.getpgid(os.getpid(), 0)"),
        ('os.system("true")', 'os.system(cmd="true")'),
        ("mkdir(parents=True, exist_ok=True)", "mkdir(exist_ok=True, missing_ok=True)"),
        ('write_text("x", encoding="utf-8")', "write_text(1)"),
        ("follow_symlinks=False)", "False)"),
        ("ignore_errors=True)", "ignore_errors=True, onexc=None)"),
        ('shutil.which("git")', 'shutil.which("git", 1, None, True)'),
        ("dst=destination)", "dest=destination)"),
        ('os.path.join(destination, "a")', "os.path.join(destination, 1)"),
        ('raise TimeoutError("slow")', 'raise TimeoutError(message="slow")'),
        ('raise MissingTool("no tool")', 'raise ImportError(message="no tool")'),
    ],
    ids=[
        "environ-get-arguments",
        "environ-is-no-dict",
        "environ-value-str",
        "getenv-keyword",
        "dict-get-positional-only",
        "kill-positional-only",
        "getpgid-arguments",
        "system-keyword",
        "mkdir-keyword",
        "write-text-str",
        "copy2-follow-symlinks-keyword-only",
        "rmtree-onexc",
        "which-arguments",
        "move-keyword",
        "join-path",
        "oserror-no-keyword",
        "importerror-keyword",
    ],
)
def test_the_os_and_shutil_shims_refuse_what_cpython_refuses(
    right: str, wrong: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _OS_SHUTIL.replace(right, wrong, 1)
    assert source != _OS_SHUTIL
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


@needs_codon
def test_what_shutil_returns_for_a_path_is_only_a_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # copy2() and move() return their Path destination, or a str when it names
    # a directory: the check takes str() and Path() of it, and nothing else.
    wrong = _OS_SHUTIL.replace("{Path(moved)}", "{moved.name}", 1)
    assert wrong != _OS_SHUTIL
    result = _check_case(wrong, tmp_path, monkeypatch)
    assert not result.ok
    assert "name" in result.format()


# ---------------------------------------------------------------------------
# The file helpers of pathlib, os, os.path and shutil (#568, #569)
# ---------------------------------------------------------------------------


_FILE_HELPERS = """
from __future__ import annotations

import os
import os.path
import shutil
from pathlib import Path


def names(directory: Path) -> list[str]:
    found: list[str] = []
    for entry in sorted(directory.iterdir()):
        found.append(entry.with_suffix(".csv").name)
        found.append(entry.with_stem("x").name)
    found += [str(directory.joinpath("a", "b.txt")), directory.parts[0]]
    found += list(directory.parts[1:]) + entry_suffixes(directory)
    found += [directory.root, directory.drive, directory.anchor, directory.as_uri()]
    return found


def entry_suffixes(path: Path) -> list[str]:
    first: str = path.suffixes[0] if len(path.suffixes) > 0 else ""
    return [first, str(len(path.parts)), "a" if "a" in path.parts else ""]


def touch(path: Path, other: Path) -> int:
    path.touch()
    path.touch(exist_ok=False)
    path.touch(384, True)
    moved: Path = path.rename(other)
    moved.rename(target="x")
    other.rmdir()
    path.chmod(420, follow_symlinks=False)
    path.lstat()
    link: Path = path.readlink()
    same = path.samefile(other) or path.match("*.txt") or link == path
    return path.write_bytes(b"x") + len(path.owner()) + len(path.group()) + int(same)


def listing(directory: str) -> int:
    count = len(os.listdir(directory)) + len(os.listdir()) + len(os.listdir(Path(directory)))
    os.makedirs(directory, exist_ok=True)
    os.makedirs(Path(directory) / "x", 493, True)
    os.remove(directory)
    os.unlink(Path(directory))
    os.rename(directory, "other")
    os.replace("other", Path(directory))
    os.rmdir(directory)
    for root, dirs, files in os.walk(directory):
        count += len(dirs) + len(files) + len(root)
    for root, dirs, files in os.walk(Path(directory), topdown=False, followlinks=True):
        count += len(dirs) + len(files) + len(root)
    return count + len(os.getcwd())


def measuring(path: Path, other: str) -> str:
    size: int = os.path.getsize(path)
    mtime: float = os.path.getmtime(other)
    relative: str = os.path.relpath(path, other)
    relative = os.path.relpath(other)
    real: str = os.path.realpath(path)
    real = os.path.realpath(other, strict=True)
    head, tail = os.path.split(other)
    flat: str = os.path.normpath(path) + os.path.expandvars(other)
    return f"{size} {mtime} {relative} {real} {head} {tail} {flat}"


def copying(source: Path, destination: Path) -> str:
    shutil.copy(source, destination)
    a: str = shutil.copy("a", "b", follow_symlinks=False)
    shutil.copyfile(source, destination)
    b: str = shutil.copyfile("a", "b")
    shutil.copytree(source, destination, dirs_exist_ok=True)
    c: str = shutil.copytree("a", "b", True, None)
    return f"{a} {b} {c} {Path(shutil.copy(source, destination))}"
"""


@needs_codon
def test_the_file_helpers_of_pathlib_os_and_shutil_take_what_cpython_takes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _check_case(_FILE_HELPERS, tmp_path, monkeypatch)
    assert result.ok, result.format()


@needs_codon
@pytest.mark.parametrize(
    ("right", "wrong"),
    [
        # Each is a TypeError in CPython 3.10, the oldest Python the package
        # supports (`Path.walk()` and `follow_symlinks` of copy() are newer or
        # keyword-only).
        ('with_suffix(".csv")', "with_suffix(1)"),
        ('with_stem("x")', "with_stem(1)"),
        ('joinpath("a", "b.txt")', "joinpath(1)"),
        ("directory.parts[0]", "directory.parts[0] + 1"),
        ("path.touch(exist_ok=False)", "path.touch(missing_ok=False)"),
        ("path.touch(384, True)", "path.touch(384, True, 1)"),
        ('moved.rename(target="x")', "moved.rename()"),
        ("other.rmdir()", "other.rmdir(True)"),
        ("follow_symlinks=False)\n    path.lstat", "False)\n    path.lstat"),
        ('write_bytes(b"x")', "write_bytes(1)"),
        ("os.listdir(directory)", "os.listdir(directory, 1)"),
        ("exist_ok=True)\n    os.makedirs(Path", "exist_ok=1)\n    os.makedirs(Path"),
        ("os.remove(directory)", "os.remove(directory, 1)"),
        ("os.walk(Path(directory), topdown=False", "os.walk(Path(directory), 1.5"),
        ("os.walk(directory)", "os.walk(directory, True, print)"),
        ("os.path.getsize(path)", "os.path.getsize(1)"),
        ("os.path.relpath(path, other)", "os.path.relpath(path, 1)"),
        ("os.path.realpath(other, strict=True)", "os.path.realpath(other, True)"),
        ("os.path.split(other)", "os.path.split(other, other)"),
        (
            'shutil.copy("a", "b", follow_symlinks=False)',
            'shutil.copy("a", "b", False)',
        ),
        ('shutil.copyfile("a", "b")', 'shutil.copyfile("a")'),
        ("dirs_exist_ok=True)", "dirs_exist_ok=True, ignore=print)"),
        ('copytree("a", "b", True, None)', 'copytree("a", "b", True, print)'),
    ],
    ids=[
        "with-suffix-str",
        "with-stem-str",
        "joinpath-parts",
        "parts-are-str",
        "touch-keyword",
        "touch-arguments",
        "rename-target",
        "rmdir-arguments",
        "chmod-keyword-only",
        "write-bytes-data",
        "listdir-arguments",
        "makedirs-exist-ok-bool",
        "remove-keyword-only",
        "walk-topdown-bool",
        "walk-onerror-callable",
        "getsize-path",
        "relpath-start",
        "realpath-strict-keyword-only",
        "split-arguments",
        "copy-follow-symlinks-keyword-only",
        "copyfile-arguments",
        "copytree-ignore-callable",
        "copytree-ignore-positional",
    ],
)
def test_the_file_helpers_refuse_what_cpython_refuses(
    right: str, wrong: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _FILE_HELPERS.replace(right, wrong, 1)
    assert source != _FILE_HELPERS
    result = _check_case(source, tmp_path, monkeypatch)
    assert not result.ok, wrong


def test_the_file_helpers_are_declared_with_the_parameters_cpython_3_10_has() -> None:
    # The shims' parameter names, in order, are CPython's: the positional ones
    # first, then (after `*_kw`) the keyword-only ones. A name that CPython
    # takes by position only, or as a callable, is not compared.
    import inspect
    import os
    import os.path
    import pathlib
    import shutil

    def declared(
        path: Path, owner: str | None
    ) -> dict[str, tuple[list[str], list[str]]]:
        tree = ast.parse(path.read_text())
        body = tree.body
        if owner is not None:
            body = next(
                n for n in body if isinstance(n, ast.ClassDef) and n.name == owner
            ).body
        out = {}
        for node in body:
            if isinstance(node, ast.FunctionDef):
                args = node.args
                positional = [a.arg for a in args.args if a.arg != "self"]
                keyword = [a.arg for a in args.kwonlyargs]
                out[node.name] = (positional, keyword)
        return out

    cases = [
        (
            model_dir() / "os" / "__init__.codon",
            None,
            os,
            "getcwd listdir makedirs remove unlink rmdir rename replace walk",
        ),
        (
            model_dir() / "os" / "path.codon",
            None,
            os.path,
            "getsize getmtime getatime getctime islink lexists realpath relpath "
            "expandvars split normpath",
        ),
        (
            model_dir() / "shutil.codon",
            None,
            shutil,
            "copy copyfile copytree",
        ),
        (
            model_dir() / "pathlib.codon",
            "Path",
            pathlib.Path,
            "with_suffix with_stem joinpath iterdir touch rename rmdir chmod "
            "write_bytes samefile lstat readlink owner group as_uri",
        ),
    ]
    # Keyword-only parameters CPython added after 3.10, which the 3.10 shims
    # do not declare (Path.owner/group(follow_symlinks=) is 3.13).
    newer = {"owner": {"follow_symlinks"}, "group": {"follow_symlinks"}}
    for path, owner, python, names in cases:
        shim = declared(path, owner)
        for name in names.split():
            signature = inspect.signature(getattr(python, name))
            positional = [
                p.name
                for p in signature.parameters.values()
                if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
            ]
            keyword = [
                p.name
                for p in signature.parameters.values()
                if p.kind is inspect.Parameter.KEYWORD_ONLY
                and not (owner is not None and p.name in newer.get(name, set()))
            ]
            if owner is not None:
                positional = (
                    positional[1:] if positional[:1] == ["self"] else positional
                )
            got_positional, got_keyword = shim[name]
            assert (
                got_positional == positional
            ), f"{getattr(python, '__name__', 'Path')}.{name}"
            assert (
                got_keyword == keyword
            ), f"{getattr(python, '__name__', 'Path')}.{name}"


# ---------------------------------------------------------------------------
# The subprocess and importlib.metadata shims, str.removesuffix(), a typed
# *args and except (A, B) (#118)
# ---------------------------------------------------------------------------


_SUBPROCESS = """
from __future__ import annotations

import subprocess
from importlib import metadata
from pathlib import Path
from typing import Optional


def installed(name: str) -> Optional[str]:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError as exc:
        return str(exc)


def git(repo: Path, *args: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            cwd=repo,
            env={"LANG": "C"},
            check=False,
        )
        done = subprocess.run(["git"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or done.stderr is None:
        return None
    return f"{result.stdout} {done.args[0]}".strip()


def url(text: str, repo: Path, base: Path) -> str:
    inside = repo.relative_to(base).as_posix()
    return text.removesuffix(".git").removeprefix("git+") + inside + str(Path.cwd())


def spawn(repo: Path) -> int:
    process: subprocess.Popen[bytes] = subprocess.Popen(
        ["sleep", "1"],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        cwd=str(repo),
        start_new_session=True,
    )
    if process.poll() is None:
        process.terminate()
    return process.wait(timeout=5.0)


def checked() -> subprocess.CompletedProcess[str]:
    done = subprocess.CompletedProcess[str](["true"], 0, "out", None)
    try:
        done.check_returncode()
    except subprocess.CalledProcessError as exc:
        raise subprocess.TimeoutExpired(exc.cmd, 1, output=exc.stdout)
    return done
"""


@needs_codon
def test_the_subprocess_and_metadata_shims_take_what_cpython_takes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert (model_dir() / "subprocess.codon").is_file()
    assert (model_dir() / "importlib" / "metadata.codon").is_file()
    result = _check_case(_SUBPROCESS, tmp_path, monkeypatch)
    assert result.ok, result.format()


@needs_codon
@pytest.mark.parametrize(
    ("right", "wrong"),
    [
        # Each is a TypeError or an AttributeError in CPython 3.10, the oldest
        # Python the package supports (process_group is 3.11's, walk_up
        # 3.12's), but relative-to-parts, which 3.10 takes and 3.14 refuses,
        # and split-handler-body, a body of the wrong type in the
        # handler `except (A, B):` becomes two of. The ones after that handler
        # are located past the lines its copy adds.
        ("metadata.version(name)", "metadata.version(distribution=name)"),
        ("metadata.version(name)", 'metadata.version(name, "x")'),
        ('["git"], capture_output=True', '["git"], capture_outputs=True'),
        (
            '["git"], capture_output=True',
            '["git"], process_group=0, capture_output=True',
        ),
        ("        return None\n    if result", "        return 1\n    if result"),
        ("done.stderr is None", "done.output is None"),
        ('text.removesuffix(".git")', 'text.removesuffix(suffix=".git")'),
        ("repo.relative_to(base)", "repo.relative_to(base, walk_up=True)"),
        ("repo.relative_to(base)", 'repo.relative_to(base, "x")'),
        ("repo.relative_to(base)", "repo.relative_to(other=base)"),
        ("str(Path.cwd())", "str(Path.cwd(repo))"),
        ("process.wait(timeout=5.0)", "process.wait(5.0, 1)"),
        ("subprocess.TimeoutExpired(exc.cmd, 1,", "subprocess.TimeoutExpired(exc.cmd,"),
    ],
    ids=[
        "version-keyword",
        "version-arguments",
        "run-unknown-keyword",
        "run-process-group",
        "split-handler-body",
        "completed-has-no-output",
        "removesuffix-positional-only",
        "relative-to-walk-up",
        "relative-to-parts",
        "relative-to-keyword",
        "cwd-arguments",
        "wait-arguments",
        "timeout-expired-arguments",
    ],
)
def test_the_subprocess_and_metadata_shims_refuse_what_cpython_refuses(
    right: str, wrong: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _SUBPROCESS.replace(right, wrong, 1)
    assert source != _SUBPROCESS
    result = _check_case(source, tmp_path, monkeypatch)
    assert not result.ok
    first = wrong.splitlines()[0]
    line = next(
        i
        for i, text in enumerate(textwrap.dedent(source).splitlines(), 1)
        if first in text
    )
    located = [d for d in result.diagnostics if d.path and d.path.endswith(".py")]
    assert located, result.format()
    assert located[0].line == line, result.format()


def test_a_typed_star_args_is_called_with_one_value() -> None:
    checks = render_library_checks(
        ast.parse(
            "def git(repo: Path, *args: str, timeout: int) -> None: ...\n"
            "def log(*values: object) -> None: ...\n"
            "def anything(*values) -> None: ...\n"
            "class Context:\n"
            "    def __exit__(self, *args: object) -> None: ...\n"
        )
    )
    calls = checks.source.splitlines()
    assert (
        "    git(_acs_value(Path), _acs_value(str), timeout=_acs_value(int))" in calls
    )
    assert "    log()" in calls
    assert "    anything()" in calls
    assert "    _acs_value(Context).__exit__()" in calls
    assert not checks.problems


# ---------------------------------------------------------------------------
# The signal and atexit shims, a Popen's pipes, and the dotenv and
# carla_install models (#119)
# ---------------------------------------------------------------------------


_PROCESS_SIGNALS = """
from __future__ import annotations

import atexit
import os
import signal
import subprocess
from pathlib import Path
from typing import Any, Optional

import numpy as np
from dotenv import load_dotenv

from .carla_install import installed_executable


class Recorder:
    _process: Optional[subprocess.Popen[bytes]]

    def __init__(self) -> None:
        self._process = None

    def start(self, size: str) -> None:
        load_dotenv(override=False)
        load_dotenv(Path(".env"), None, True)
        self._process = subprocess.Popen(
            ["encoder", "-s", size],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        atexit.register(self.stop)
        atexit.register(self.finish, 3, reason="exit")
        signal.signal(signal.SIGTERM, self._on_signal)
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    def _on_signal(self, signum: int, frame: Any) -> None:
        self.stop()

    def finish(self, code: int, reason: str = "") -> None:
        self.stop()

    def write(self) -> int:
        bgr = np.zeros((2, 2, 3), dtype=np.uint8)
        if self._process is None or self._process.stdin is None:
            return 0
        return self._process.stdin.write(bgr.data) + self._process.stdin.write(b"raw")

    def stop(self) -> None:
        if self._process is not None:
            if self._process.stdin is not None:
                self._process.stdin.close()
            self._process.wait()
            if self._process.stderr is not None:
                logged: bytes = self._process.stderr.read()
            os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
            atexit.unregister(self.stop)
            self._process = None


def launcher() -> Optional[Path]:
    return installed_executable()
"""


def _check_signals_case(
    source: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> TypeCheckResult:
    # A signal handler's frame is CPython's frame object, which Codon has no
    # type for (`Any`): the handler is compiled through signal.signal(), which
    # calls it as CPython does, not called with values of its own.
    monkeypatch.setitem(
        check.UNCALLED,
        f"{_FAKE}.Recorder._on_signal",
        "a signal handler, whose frame Codon has no type for",
    )
    return _check_case(source, tmp_path, monkeypatch)


@needs_codon
def test_the_signal_atexit_and_pipe_shims_and_their_models_take_what_cpython_takes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for shim in ("signal.codon", "atexit.codon", "dotenv", "subprocess.codon"):
        assert (model_dir() / shim).exists(), shim
    result = _check_signals_case(_PROCESS_SIGNALS, tmp_path, monkeypatch)
    assert result.ok, result.format()


@needs_codon
@pytest.mark.parametrize(
    ("right", "wrong", "at"),
    [
        # Each fails in CPython 3.10 (the oldest Python the package supports)
        # with the installed python-dotenv: a TypeError or an AttributeError
        # when it is called, but handler-arity and register-arguments, which
        # CPython takes and fails on later, with a TypeError when the signal
        # arrives and one it reports at exit. Codon reports an error in the
        # Popen call at its first line (`at`).
        ("signal.SIG_IGN)", "5)", "signal.SIGINT, 5)"),
        ("signal.SIGKILL)", "signal.SIGKILLED)", None),
        ("signal.SIGTERM, self._on_signal)", "signal.SIGTERM, self.stop)", None),
        ("atexit.register(self.stop)", "atexit.register(func=self.stop)", None),
        ('3, reason="exit")', '3, why="exit")', None),
        ("atexit.unregister(self.stop)", "atexit.unregister(func=self.stop)", None),
        ("load_dotenv(override=False)", "load_dotenv(overwrite=False)", None),
        ("stdin=subprocess.PIPE,", "stdinput=subprocess.PIPE,", "subprocess.Popen("),
        (
            "start_new_session=True,",
            "start_new_session=True, shell=1, x=1,",
            "subprocess.Popen(",
        ),
        ("stdin.write(bgr.data)", "stdin.write(bgr.size)", None),
        ("installed_executable()", "installed_executable(True)", None),
    ],
    ids=[
        "signal-int-handler",
        "signal-unknown-number",
        "handler-arity",
        "register-keyword",
        "register-arguments",
        "unregister-keyword",
        "dotenv-keyword",
        "popen-keyword",
        "popen-unknown-keyword",
        "write-bytes-like",
        "installed-executable-arguments",
    ],
)
def test_the_signal_atexit_and_pipe_shims_and_their_models_refuse_what_cpython_refuses(
    right: str,
    wrong: str,
    at: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _PROCESS_SIGNALS.replace(right, wrong, 1)
    assert source != _PROCESS_SIGNALS
    result = _check_signals_case(source, tmp_path, monkeypatch)
    assert not result.ok
    needle = at if at is not None else wrong
    line = next(
        i
        for i, text in enumerate(textwrap.dedent(source).splitlines(), 1)
        if needle in text
    )
    located = [d for d in result.diagnostics if d.path and d.path.endswith(".py")]
    assert located, result.format()
    assert located[0].line == line, result.format()


# ---------------------------------------------------------------------------
# The shlex and sys shims, signal.Signals, a file object (Path.open(), a
# Popen stream) and the symlink queries (#127)
# ---------------------------------------------------------------------------


_STACK = """
from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path
from typing import IO, Optional


def command(root: str, args: list[str]) -> str:
    argv = ["python3", "-m", "x y", *args]
    return shlex.quote(root) + " " + shlex.join(argv)


def interpreter() -> list[str]:
    if sys.platform != "linux":
        return list(sys.argv)
    return [sys.executable, "-c", "pass"]


def stop(pid: int, sig: signal.Signals) -> str:
    try:
        os.killpg(pid, sig)
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    signals = [sig, signal.SIGTERM, signal.Signals(9)]
    name = signals[0].name
    return f"{name} {int(signal.SIGINT)} {signal.SIGINT == 2}"


def link(directory: Path, target: Path) -> Path:
    path = directory / "link"
    if path.is_symlink() or path.exists():
        path.unlink()
    path.symlink_to(target, target_is_directory=True)
    return path


def opened(directory: Path) -> IO[bytes]:
    return (directory / "x.log").open("ab")


def logged(directory: Path) -> Optional[int]:
    log = opened(directory)
    log.write(b"start\\n")
    process: subprocess.Popen[bytes] = subprocess.Popen(
        ["true"], stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL
    )
    code = process.wait()
    process.send_signal(signal.SIGTERM)
    log.close()
    return code
"""


@needs_codon
def test_a_checked_module_has_its_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Python sets a module's __file__, which Codon does not: the library check
    # gives each checked module its source's path.
    source = """
    from __future__ import annotations

    from pathlib import Path

    from .entity_role import __file__ as role_file


    def directory() -> Path:
        return Path(role_file).resolve().parent
    """
    result = _check_case(source, tmp_path, monkeypatch)
    assert result.ok, result.format()


def _top_level_names(path: Path) -> set[str]:
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
def test_the_sys_shim_keeps_all_of_codons_sys() -> None:
    # The shim replaces the sys module Codon ships, which Codon's own modules
    # (time, random, itertools) and typesafe_carla's library import: it
    # declares every name Codon's does.
    tc = available_toolchain()
    assert tc is not None
    codon_sys = tc.codon_dir / "lib" / "codon" / "stdlib" / "sys.codon"
    if not codon_sys.is_file():
        pytest.skip("this Codon ships no stdlib/sys.codon")
    shim = _top_level_names(model_dir() / "sys.codon")
    assert _top_level_names(codon_sys) <= shim
    assert {"platform", "executable"} <= shim


@needs_codon
def test_the_shlex_sys_signal_and_file_shims_take_what_cpython_takes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for shim in ("shlex.codon", "sys.codon", "signal.codon", "typing.codon"):
        assert (model_dir() / shim).is_file(), shim
    result = _check_case(_STACK, tmp_path, monkeypatch)
    assert result.ok, result.format()


@needs_codon
@pytest.mark.parametrize(
    ("right", "wrong", "at"),
    [
        # Each fails in CPython 3.10 (the oldest Python the package supports)
        # and 3.13 when it is called: a TypeError, an AttributeError, or the
        # ValueError of a Signals looked up by anything but its number. Codon
        # reports a bad argument of a call written over several lines at the
        # call's first line (`at`).
        ("shlex.quote(root)", "shlex.quote(len(root))", None),
        ("shlex.join(argv)", "shlex.join([len(root)])", None),
        ('sys.platform != "linux"', 'sys.platfrom != "linux"', None),
        ("signals[0].name", "signals[0].label", None),
        ("os.killpg(pid, sig)", "os.killpg(pid, sig.name)", None),
        ("signal.Signals(9)", 'signal.Signals("SIGKILL")', None),
        ("target_is_directory=True)", "target_is_dir=True)", None),
        ("path.is_symlink()", "path.is_symlink(True)", None),
        ('.open("ab")', '.open("ab", buffer=0)', None),
        ('log.write(b"start\\n")', 'log.write(len(b"start\\n"))', None),
        ("stdout=log,", 'stdout="x.log",', "subprocess.Popen("),
    ],
    ids=[
        "quote-str",
        "join-str",
        "sys-unknown-name",
        "signal-unknown-attribute",
        "killpg-signal-number",
        "signals-by-number",
        "symlink-to-keyword",
        "is-symlink-arguments",
        "open-keyword",
        "write-bytes",
        "popen-stream-file",
    ],
)
def test_the_shlex_sys_signal_and_file_shims_refuse_what_cpython_refuses(
    right: str,
    wrong: str,
    at: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _STACK.replace(right, wrong, 1)
    assert source != _STACK
    result = _check_case(source, tmp_path, monkeypatch)
    assert not result.ok
    needle = at if at is not None else wrong
    line = next(
        i
        for i, text in enumerate(textwrap.dedent(source).splitlines(), 1)
        if needle in text
    )
    located = [d for d in result.diagnostics if d.path and d.path.endswith(".py")]
    assert located, result.format()
    assert located[0].line == line, result.format()


# ---------------------------------------------------------------------------
# The json and contextlib shims, Path.stat() and Path.replace() (#557)
# ---------------------------------------------------------------------------


_JSON = """
from __future__ import annotations

import contextlib
import json
from pathlib import Path


def entry(value: float, names: dict[str, str]) -> str:
    return '{"z_offset": ' + json.dumps(value) + ", " + json.dumps(names)[1:]


def read(path: Path) -> float:
    document = json.loads(path.read_text(encoding="utf-8"))
    for key, item in document.items():
        print(key, str(item))
    return float(document["z_offset"]) + float(document.get("x")) + len(document)


def store(path: Path, other: Path) -> int:
    path.write_text(entry(1.5, {"a": "b"}), encoding="utf-8")
    with path.open("r") as stream:
        document = json.load(stream)
    with other.open("w") as stream:
        json.dump([1, 2], stream, indent=2)
    stat = path.stat()
    path.replace(other)
    with contextlib.suppress(OSError, ValueError):
        other.unlink(missing_ok=True)
    try:
        json.loads("{")
    except json.JSONDecodeError as error:
        return error.pos
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return stat.st_size + stat.st_mtime_ns + stat.st_dev + stat.st_ino + int(stat.st_mtime) + int(document)


def used() -> float:
    return read(Path("x")) + store(Path("x"), Path("y"))
"""


@needs_codon
def test_the_json_contextlib_and_path_stat_shims_take_what_cpython_takes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for shim in ("json.codon", "contextlib.codon"):
        assert (model_dir() / shim).is_file(), shim
    result = _check_case(_JSON, tmp_path, monkeypatch)
    assert result.ok, result.format()


@needs_codon
@pytest.mark.parametrize(
    ("right", "wrong"),
    [
        # dumps() takes any one type, but the caller's own parameter is typed.
        ('entry(1.5, {"a": "b"})', 'entry(1.5, {"a": 1})'),
        ('document["z_offset"]', "document[1.5]"),
        ('document.get("x")', "document.get(1)"),
        ('json.loads(path.read_text(encoding="utf-8"))', "json.loads(1)"),
        ("stat.st_size +", "stat.st_name +"),
        ("path.replace(other)", "path.replace()"),
        ("json.JSONDecodeError as error", "json.JSONEncoder as error"),
    ],
    ids=[
        "dumps-mixed-dict",
        "value-index-float",
        "value-get-int",
        "loads-non-str",
        "stat-unknown-field",
        "replace-target",
        "json-unknown-name",
    ],
)
def test_the_json_contextlib_and_path_stat_shims_refuse_what_cpython_refuses(
    right: str,
    wrong: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert right in _JSON, right
    result = _check_case(_JSON.replace(right, wrong, 1), tmp_path, monkeypatch)
    assert not result.ok


# ---------------------------------------------------------------------------
# What autoware_stack.docker reads: the ids, a link, a file's text and bytes,
# a recursive glob, the path queries and a Popen's output by line (#128)
# ---------------------------------------------------------------------------


_DOCKER = """
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path


def label() -> str:
    uid = os.getuid()
    gid = os.getgid()
    return f"uid-{uid}:{gid}"


def followed(path: Path, mount: str) -> Path:
    target = Path(os.readlink(path))
    if not target.is_absolute():
        target = path.parent / target
    elif target.is_relative_to(mount):
        target = path / target.relative_to(mount)
    return target


def digest(source: Path) -> str:
    sha = hashlib.sha256()
    for path in sorted(source.rglob("package.xml")):
        sha.update(path.read_bytes())
    return sha.hexdigest()


def supported(path: Path) -> bool:
    return 'name="x"' in path.read_text()


def streamed(argv: list[str]) -> int:
    process: subprocess.Popen[str] = subprocess.Popen(
        argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    assert process.stdout is not None
    for line in process.stdout:
        sys.stdout.write(line)
    return process.wait()
"""


@needs_codon
def test_the_os_pathlib_and_popen_shims_take_what_docker_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _check_case(_DOCKER, tmp_path, monkeypatch)
    assert result.ok, result.format()


@needs_codon
@pytest.mark.parametrize(
    ("right", "wrong"),
    [
        # Each fails in CPython 3.10 (the oldest Python the package supports)
        # and 3.14 when it is called: a TypeError or an AttributeError; but
        # is-relative-to-parts, which 3.10 takes and 3.14 refuses, and
        # read-text-newline, which 3.13 takes and 3.10 refuses.
        ("os.getuid()", "os.getuid(0)"),
        ("os.getgid()", "os.getgid(0)"),
        ("os.readlink(path)", "os.readlink(path, None)"),
        ("target.is_absolute()", "target.is_absolute(True)"),
        ("target.is_relative_to(mount)", "target.is_relative_to(len(mount))"),
        ("target.is_relative_to(mount)", 'target.is_relative_to(mount, "x")'),
        ("target.is_relative_to(mount)", "target.is_relative_to(other=mount)"),
        ('source.rglob("package.xml")', "source.rglob()"),
        ("path.read_bytes()", 'path.read_bytes("rb")'),
        ("path.read_text()", 'path.read_text(encodig="utf-8")'),
        ("path.read_text()", 'path.read_text(newline="")'),
        ("sys.stdout.write(line)", "sys.stdout.write(line.decode())"),
    ],
    ids=[
        "getuid-arguments",
        "getgid-arguments",
        "readlink-dir-fd-keyword",
        "is-absolute-arguments",
        "is-relative-to-path",
        "is-relative-to-parts",
        "is-relative-to-keyword",
        "rglob-pattern",
        "read-bytes-arguments",
        "read-text-keyword",
        "read-text-newline",
        "popen-line-is-str",
    ],
)
def test_the_os_pathlib_and_popen_shims_refuse_what_cpython_refuses(
    right: str,
    wrong: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _DOCKER.replace(right, wrong, 1)
    assert source != _DOCKER
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


# ---------------------------------------------------------------------------
# The process calls of a forking parent and its child: os.fork(), waitpid()
# and the wait-status macros, signal.alarm(), getsignal() and
# pthread_sigmask(), KeyboardInterrupt, ChildProcessError and SystemExit.code
# (#719)
# ---------------------------------------------------------------------------


_FORKING = """
from __future__ import annotations

import os
import signal
from typing import Any, Callable


def on_alarm(signum: int, frame: Any) -> None:
    pass


def launch(run: Callable[[], int], timeout: int) -> tuple[bool, int]:
    mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
    pid = os.fork()
    if pid == 0:
        code = 1
        try:
            os.setpgrp()
            signal.pthread_sigmask(signal.SIG_SETMASK, mask)
            code = run()
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
        except KeyboardInterrupt:
            code = 128 + int(signal.SIGINT)
        finally:
            os._exit(code)
    previous = signal.getsignal(signal.SIGALRM)
    os.setpgid(pid, pid)
    signal.signal(signal.SIGALRM, on_alarm)
    left: int = signal.alarm(timeout)
    try:
        done, status = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        done, status = os.waitpid(pid, 0)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    if os.WIFSIGNALED(status):
        return False, -os.WTERMSIG(status)
    code = os.WEXITSTATUS(status=status) if os.WIFEXITED(status) else 1
    return code == 0, code + left + done + len(mask)
"""


def _check_forking_case(
    source: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> TypeCheckResult:
    # The handler's frame is `Any`: it is compiled through signal.signal().
    monkeypatch.setitem(
        check.UNCALLED,
        f"{_FAKE}.on_alarm",
        "a signal handler, whose frame Codon has no type for",
    )
    return _check_case(source, tmp_path, monkeypatch)


@needs_codon
def test_the_os_and_signal_shims_take_what_a_forking_parent_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _check_forking_case(_FORKING, tmp_path, monkeypatch)
    assert result.ok, result.format()


@needs_codon
@pytest.mark.parametrize(
    ("right", "wrong"),
    [
        # Each is a TypeError in CPython 3.10 (the oldest Python the package
        # supports) and 3.13 when it is called.
        ("os.fork()", "os.fork(1)"),
        ("os.setpgid(pid, pid)", "os.setpgid(pid=pid, pgrp=pid)"),
        ("os.waitpid(pid, 0)", "os.waitpid(pid)"),
        ("os._exit(code)", "os._exit(str(code))"),
        ("os.WIFSIGNALED(status)", "os.WIFSIGNALED(str(status))"),
        ("os.WEXITSTATUS(status=status)", "os.WEXITSTATUS(code=status)"),
        ("signal.alarm(timeout)", "signal.alarm(seconds=timeout)"),
        ("signal.alarm(0)", "signal.alarm(0.5)"),
        ("signal.getsignal(signal.SIGALRM)", 'signal.getsignal("SIGALRM")'),
        ("{signal.SIGINT, signal.SIGTERM}", '{"SIGINT"}'),
        ("signal.SIG_SETMASK, mask)", "signal.SIG_SETMASK, 2)"),
        ("signal.SIGALRM, previous)", "signal.SIGALRM, mask)"),
    ],
    ids=[
        "fork-arguments",
        "setpgid-positional-only",
        "waitpid-arguments",
        "exit-status-int",
        "wifsignaled-status-int",
        "wexitstatus-keyword",
        "alarm-positional-only",
        "alarm-seconds-int",
        "getsignal-number",
        "sigmask-signal-numbers",
        "sigmask-iterable",
        "signal-handler",
    ],
)
def test_the_os_and_signal_shims_refuse_what_cpython_refuses_of_a_fork(
    right: str, wrong: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _FORKING.replace(right, wrong, 1)
    assert source != _FORKING
    result = _check_forking_case(source, tmp_path, monkeypatch)
    assert not result.ok, wrong
    line = next(
        i
        for i, text in enumerate(textwrap.dedent(source).splitlines(), 1)
        if wrong in text
    )
    located = [d for d in result.diagnostics if d.path and d.path.endswith(".py")]
    assert located, result.format()
    assert located[0].line == line, result.format()


def test_the_process_shims_are_declared_with_the_parameters_cpython_has() -> None:
    # As for the file helpers: the parameter names CPython takes by keyword
    # (`os._exit(status)`, `signal.pthread_sigmask(how, mask)`) are the
    # shims'; a positional-only one (`waitpid(pid, options, /)`) is a
    # `*args` there.
    import inspect
    import os
    import signal

    cases = [
        (
            model_dir() / "os" / "__init__.codon",
            os,
            "fork setpgrp setpgid waitpid _exit WIFEXITED WEXITSTATUS "
            "WIFSIGNALED WTERMSIG",
        ),
        (model_dir() / "signal.codon", signal, "getsignal alarm pthread_sigmask"),
    ]
    for path, python, names in cases:
        shim = {
            node.name: node.args
            for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.FunctionDef)
        }
        for name in names.split():
            parameters = inspect.signature(getattr(python, name)).parameters.values()
            by_keyword = [
                p.name
                for p in parameters
                if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
            ]
            positional_only = [
                p.name
                for p in parameters
                if p.kind is inspect.Parameter.POSITIONAL_ONLY
            ]
            args = shim[name]
            assert [a.arg for a in args.args] == by_keyword, f"{name}"
            assert (args.vararg is not None) == bool(positional_only), f"{name}"
    assert os.WNOHANG == 1
    assert (signal.SIG_BLOCK, signal.SIG_UNBLOCK, signal.SIG_SETMASK) == (0, 1, 2)
    shim_os = (model_dir() / "os" / "__init__.codon").read_text()
    shim_signal = (model_dir() / "signal.codon").read_text()
    assert re.search(r"^WNOHANG = 1$", shim_os, re.MULTILINE)
    for number, mask in enumerate(("SIG_BLOCK", "SIG_UNBLOCK", "SIG_SETMASK")):
        assert re.search(rf"^{mask} = {number}$", shim_signal, re.MULTILINE)


# ---------------------------------------------------------------------------
# The fnmatch shim: fnmatch() and fnmatchcase() (#728)
# ---------------------------------------------------------------------------


_FNMATCH = """
from __future__ import annotations

import fnmatch
from fnmatch import fnmatchcase
from pathlib import Path

EXCLUDED = ("*firetruck*", "*ambulance*", "*cop*", "*police*")


def candidate(name: str) -> bool:
    if any(fnmatch.fnmatch(name, pattern) for pattern in EXCLUDED):
        return False
    if fnmatch.fnmatch(Path(name), "*.osm"):
        return False
    return fnmatchcase(name=name, pat="vehicle.*") or fnmatch.fnmatchcase(name, "?")
"""


@needs_codon
def test_the_fnmatch_shim_takes_what_cpython_takes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert (model_dir() / "fnmatch.codon").is_file()
    result = _check_case(_FNMATCH, tmp_path, monkeypatch)
    assert result.ok, result.format()


@needs_codon
@pytest.mark.parametrize(
    ("right", "wrong"),
    [
        # Each fails in CPython 3.10 (the oldest Python the package supports)
        # and 3.13 when it is called: a TypeError or an AttributeError.
        ("fnmatch.fnmatch(name, pattern)", "fnmatch.fnmatch(name, len(pattern))"),
        ("fnmatch.fnmatch(name, pattern)", "fnmatch.fnmatch(len(name), pattern)"),
        ("fnmatch.fnmatch(name, pattern)", "fnmatch.fnmatch(name, pattern=pattern)"),
        ("fnmatch.fnmatch(name, pattern)", "fnmatch.fnmatch(name, pattern, True)"),
        ('fnmatch.fnmatchcase(name, "?")', 'fnmatch.fnmatchcase(name, "?", 1)'),
        ('fnmatch.fnmatchcase(name, "?")', 'fnmatch.fnmatches(name, "?")'),
        (
            'fnmatch.fnmatch(Path(name), "*.osm")',
            'fnmatch.fnmatchcase(Path(name), "*.osm")',
        ),
    ],
    ids=[
        "pattern-str",
        "name-str",
        "pattern-keyword",
        "fnmatch-arguments",
        "fnmatchcase-arguments",
        "unknown-name",
        "fnmatchcase-path",
    ],
)
def test_the_fnmatch_shim_refuses_what_cpython_refuses(
    right: str, wrong: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _FNMATCH.replace(right, wrong, 1)
    assert source != _FNMATCH
    result = _check_case(source, tmp_path, monkeypatch)
    assert not result.ok, wrong
    line = next(
        i
        for i, text in enumerate(textwrap.dedent(source).splitlines(), 1)
        if wrong in text
    )
    located = [d for d in result.diagnostics if d.path and d.path.endswith(".py")]
    assert located, result.format()
    assert located[0].line == line, result.format()


def test_the_fnmatch_shim_matches_as_cpython_does() -> None:
    # The shim is plain Python too: run under CPython, its match (written
    # without a regular expression) gives what fnmatch's does, for its
    # parameter names, the patterns entity.fit excludes, and every set, range
    # and wildcard of a sample of patterns.
    import fnmatch
    import inspect
    import random

    from autoware_carla_scenario.entity.fit import FIT_EXCLUDED

    source = (model_dir() / "fnmatch.codon").read_text()
    shim: dict[str, Any] = {}
    exec(source, shim)  # noqa: S102 - the shim, which is valid Python
    for name in ("fnmatch", "fnmatchcase"):
        python = list(inspect.signature(getattr(fnmatch, name)).parameters)
        assert list(inspect.signature(shim[name]).parameters) == python, name
    blueprints = [
        "vehicle.carlamotors.firetruck",
        "vehicle.ford.ambulance",
        "vehicle.dodge.charger_police_2020",
        "vehicle.mini.cooper",
        "vehicle.tesla.model3",
    ]
    for blueprint in blueprints:
        for pattern in (*FIT_EXCLUDED, "vehicle.*", "vehicle.?ini.*"):
            want = fnmatch.fnmatch(blueprint, pattern)
            assert shim["fnmatch"](blueprint, pattern) == want, (blueprint, pattern)
    rng = random.Random(728)
    for _ in range(20000):
        pattern = "".join(rng.choice("ab]![-*?^\\z&") for _ in range(rng.randint(0, 8)))
        name = "".join(rng.choice("ab]![-^\\z&\nc") for _ in range(rng.randint(0, 6)))
        want = fnmatch.fnmatchcase(name, pattern)
        assert shim["fnmatchcase"](name, pattern) == want, (name, pattern)
        assert shim["fnmatch"](name, pattern) == want, (name, pattern)


# Directories of codon/ that model a third-party package, not the standard
# library; the rest of codon/ (bar _stdlib) is a shim the docs must list.
_THIRD_PARTY_MODELS = {
    "autoware_carla_scenario",
    "autoware_lanelet2_extension_python",
    "carla_canvas",
    "carla_driver_interface",
    "dotenv",
    "lanelet2",
    "lxml",
    "pyxodr",
}


def _shim_modules() -> set[str]:
    root = model_dir()
    names = set()
    for path in root.rglob("*.codon"):
        parts = path.relative_to(root).with_suffix("").parts
        if parts[0] in _THIRD_PARTY_MODELS or parts[0] == "_stdlib":
            continue
        if parts[-1] == "__init__":
            parts = parts[:-1]
            body = ast.parse(path.read_text()).body
            # A package whose __init__ only imports or is empty names no module.
            imports = (ast.Import, ast.ImportFrom)
            only_imports = all(isinstance(n, imports) for n in body)
            # ... unless there is no submodule to stand for it: then it is a shim.
            has_submodule = any(p != path for p in path.parent.rglob("*.codon"))
            if only_imports and has_submodule:
                continue
        names.add(".".join(parts))
    return names


def test_shim_list() -> None:
    # The docs list every standard-library shim of codon/ (#472), so a new
    # one cannot go undocumented, nor a removed one stay listed.
    docs = (Path(__file__).resolve().parents[2] / "docs" / "typecheck.md").read_text()
    block = docs.split("<!-- shims:start -->")[1].split("<!-- shims:end -->")[0]
    listed = set(re.findall(r"`([\w.]+)`", block))
    assert listed == _shim_modules()
    assert "tempfile" in listed
    # The shim of tempfile declares mkstemp() and mkdtemp() and nothing else.
    tempfile = ast.parse((model_dir() / "tempfile.codon").read_text())
    assert [n.name for n in tempfile.body if isinstance(n, ast.FunctionDef)] == [
        "mkstemp",
        "mkdtemp",
    ]
    # The standard modules the docs call missing are not shimmed.
    unavailable = {"inspect", "concurrent.futures"}
    assert not unavailable & listed
