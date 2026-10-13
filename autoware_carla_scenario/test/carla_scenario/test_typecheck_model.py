"""The Codon model declares the public API as the Python package defines it.

The static check compiles scenarios against ``typecheck/codon/`` instead of
the Python package, so a parameter renamed in Python and not in the model
would reject a correct scenario (or accept a wrong one).  These tests read
the model as Python syntax -- it is written to parse as Python -- and compare
every class and function ``autoware_carla_scenario/__init__.codon`` exports
with the object of the same name: its parameters (names, order, which are
keyword-only, which have defaults), the methods and properties it declares,
its enum members.  They need no Codon.
"""

from __future__ import annotations

import ast
import dataclasses
import enum
import inspect
import typing
from pathlib import Path
from typing import Any

import dotenv
import ffmpeg
import ffmpeg.nodes
import pytest
from pyxodr.road_objects.network import RoadNetwork as PyxodrRoadNetwork
from pyxodr.road_objects.road import Road as PyxodrRoad

import autoware_carla_scenario as acs
from autoware_carla_scenario import carla_install, environment, trajectory_recorder
from autoware_carla_scenario.autoware_bridge import base as bridge_base
from autoware_carla_scenario.autoware_bridge import grpc_server as bridge_grpc_server
from autoware_carla_scenario.autoware_stack import devcontainer as stack_devcontainer
from autoware_carla_scenario.autoware_stack import launcher as stack_launcher
from autoware_carla_scenario.coordinate import map_manager as coordinate_map_manager
from autoware_carla_scenario.coordinate import projection as coordinate_projection
from autoware_carla_scenario.coordinate import transform as coordinate_transform
from autoware_carla_scenario.coverage import collector as coverage_collector
from autoware_carla_scenario.coverage import items as coverage_items
from autoware_carla_scenario.driver import egodriver_client
from autoware_carla_scenario.entity import carla_driver_entity
from autoware_carla_scenario.maps import cache as maps_cache
from autoware_carla_scenario.maps import source as maps_source
from autoware_carla_scenario.odd import probes as odd_probes
from autoware_carla_scenario.odd import registry as odd_registry
from autoware_carla_scenario.odd import route as odd_route
from autoware_carla_scenario.route import frame as route_frame
from autoware_carla_scenario.traffic import context as traffic_context
from autoware_carla_scenario.traffic import registry as traffic_registry
from autoware_carla_scenario.trajectory import model as trajectory_model
from autoware_carla_scenario.typecheck import model_dir
from autoware_carla_scenario.ui import models as ui_models
from carla_driver_interface import geometry as cdi_geometry

_MODEL = model_dir() / "autoware_carla_scenario"

#: Python members the model leaves out on purpose, by class.  Each is a
#: free-form mapping (dict[str, Any]), which has no Codon type.
_OMITTED_PARAMETERS: dict[str, set[str]] = {
    "SweepConfig": {"constraints", "bindings", "odd_sample"},
}

#: (kind, name, has default): kind is "pos", "kw", "*args" or "**kwargs".
Param = tuple[str, str, bool]


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _exports() -> dict[str, str]:
    """Name -> model module (stem) for every name ``__init__.codon`` imports."""
    out: dict[str, str] = {}
    for node in _parse(_MODEL / "__init__.codon").body:
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            for alias in node.names:
                out[alias.asname or alias.name] = node.module
    return out


def _definitions(module: str, root: Path = _MODEL) -> dict[str, list[ast.stmt]]:
    """Top-level definitions of a model module (a function may be overloaded)."""
    out: dict[str, list[ast.stmt]] = {}
    for node in _parse(root / f"{module}.codon").body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            out.setdefault(node.name, []).append(node)
        elif isinstance(node, (ast.AnnAssign, ast.Assign)):
            targets = [node.target] if isinstance(node, ast.AnnAssign) else node.targets
            for target in targets:
                if isinstance(target, ast.Name):
                    out.setdefault(target.id, []).append(node)
        elif isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            for alias in node.names:
                out.setdefault(alias.asname or alias.name, []).extend(
                    _definitions(node.module, root).get(alias.name, [])
                )
    return out


def _is_required_marker(node: ast.expr | None) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_Required"
    )


def _model_params(fn: ast.FunctionDef, *, drop_first: bool) -> list[Param]:
    args = fn.args
    positional = [*args.posonlyargs, *args.args]
    defaults: list[ast.expr | None] = [None] * (
        len(positional) - len(args.defaults)
    ) + list(args.defaults)
    out: list[Param] = [
        ("pos", a.arg, d is not None and not _is_required_marker(d))
        for a, d in zip(positional, defaults)
    ]
    if drop_first:
        out = out[1:]
    if args.vararg is not None and args.vararg.arg != "_kw":
        out.append(("*args", args.vararg.arg, False))
    for a, d in zip(args.kwonlyargs, args.kw_defaults):
        out.append(("kw", a.arg, d is not None and not _is_required_marker(d)))
    if args.kwarg is not None:
        out.append(("**kwargs", args.kwarg.arg, False))
    return out


def _model_fields(cls: ast.ClassDef) -> list[Param]:
    """The generated __init__ of a model class without one: its fields."""
    out: list[Param] = []
    for stmt in cls.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            if "ClassVar" in ast.unparse(stmt.annotation):
                continue
            out.append(("pos", stmt.target.id, stmt.value is not None))
    return out


_KINDS = {
    inspect.Parameter.POSITIONAL_ONLY: "pos",
    inspect.Parameter.POSITIONAL_OR_KEYWORD: "pos",
    inspect.Parameter.KEYWORD_ONLY: "kw",
    inspect.Parameter.VAR_POSITIONAL: "*args",
    inspect.Parameter.VAR_KEYWORD: "**kwargs",
}


def _python_params(obj: Any, *, drop_first: bool) -> list[Param]:
    params = list(inspect.signature(obj).parameters.values())
    if drop_first:
        params = params[1:]
    return [
        (_KINDS[p.kind], p.name, p.default is not inspect.Parameter.empty)
        for p in params
    ]


def _methods(cls: ast.ClassDef) -> dict[str, ast.FunctionDef]:
    out: dict[str, ast.FunctionDef] = {}
    for stmt in cls.body:
        if isinstance(stmt, ast.FunctionDef):
            out.setdefault(stmt.name, stmt)  # a property's getter, not its setter
    return out


def _decorators(fn: ast.FunctionDef) -> set[str]:
    return {ast.unparse(d) for d in fn.decorator_list}


def _class_vars(cls: ast.ClassDef) -> set[str]:
    return {
        stmt.target.id
        for stmt in cls.body
        if isinstance(stmt, ast.AnnAssign)
        and isinstance(stmt.target, ast.Name)
        and "ClassVar" in ast.unparse(stmt.annotation)
    }


_EXPORTS = _exports()


def test_the_model_exports_only_names_the_package_exports() -> None:
    unknown = sorted(set(_EXPORTS) - set(acs.__all__) - set(acs._LAZY_IMPORTS))
    assert not unknown, f"the model exports names the package does not: {unknown}"


@pytest.mark.parametrize("name", sorted(_EXPORTS))
def test_an_exported_name_is_declared_as_in_python(name: str) -> None:
    _assert_declared_as_in_python(name, _EXPORTS[name], getattr(acs, name))


#: Names the model declares for the modules the library check compiles
#: (typecheck/library.py), though the package does not export them: name ->
#: (model module, the Python object).
_BOUNDARY: dict[str, tuple[str, Any]] = {
    "CarlaDriverEntity": ("entity", carla_driver_entity.CarlaDriverEntity),
    "CoverItem": ("coverage", coverage_items.CoverItem),
    "CrossItem": ("coverage", coverage_items.CrossItem),
    "CoverageCollector": ("coverage", coverage_collector.CoverageCollector),
    "MapManager": ("coordinate", coordinate_map_manager.MapManager),
    # What map_manager and sweeper.map_loader import of coordinate.projection,
    # which the library check now compiles from its source too.
    **{
        name: ("coordinate", getattr(coordinate_projection, name))
        for name in ("_parse_geo_reference", "map_origin", "resolve_projector")
    },
    "TrafficContext": ("traffic", traffic_context.TrafficContext),
    **{
        name: ("traffic", getattr(traffic_registry, name))
        for name in (
            "TRAFFIC_BACKEND_ENTRY_POINT_GROUP",
            "available_backends",
            "build_backend",
            "get_backend_factory",
            "load_traffic_backend_plugins",
            "register_backend",
            "unregister_backend",
        )
    },
    "ResolvedTrajectory": ("trajectory", trajectory_model.ResolvedTrajectory),
    "RouteFrame": ("route", route_frame.RouteFrame),
    "TrajectorySample": ("trajectory", trajectory_model.TrajectorySample),
    "to_map_frame": ("coordinate", coordinate_transform.to_map_frame),
    "reset_probes": ("odd", odd_probes.reset_probes),
    "resolve_odd": ("odd", odd_registry.resolve_odd),
    # What scenario_runner's _warn_if_route_leaves_odd uses of odd.route.
    **{
        name: ("odd", getattr(odd_route, name))
        for name in ("RouteCoverage", "RouteLanelet", "plan_route_coverage")
    },
}


@pytest.mark.parametrize("name", sorted(_BOUNDARY))
def test_a_boundary_name_is_declared_as_in_python(name: str) -> None:
    module, python = _BOUNDARY[name]
    _assert_declared_as_in_python(name, module, python)


def _assert_declared_as_in_python(
    name: str, module: str, python: Any, root: Path = _MODEL
) -> None:
    definitions = _definitions(module, root).get(name)
    assert definitions, f"{name} is not defined in {module}.codon"
    node = definitions[0]

    if isinstance(node, ast.FunctionDef):
        assert inspect.isfunction(python), f"{name} is a function in the model only"
        expected = _python_params(python, drop_first=False)
        for overload in definitions:
            assert isinstance(overload, ast.FunctionDef)
            assert _model_params(overload, drop_first=False) == expected, name
        return

    if not isinstance(node, ast.ClassDef):
        return  # a constant (EGO_ROLE_NAME): its type is the class it is built from

    if typing.get_origin(python) is typing.Union:
        # A Union alias (SpawnLocation): the base class of its members here.
        for member in typing.get_args(python):
            classes = _definitions(_EXPORTS[member.__name__]).get(member.__name__, [])
            assert classes and isinstance(classes[0], ast.ClassDef)
            assert name in {
                ast.unparse(b) for b in classes[0].bases
            }, f"{member.__name__} does not derive from {name} in the model"
        return

    assert inspect.isclass(python), f"{name} is a class in the model only"
    if issubclass(python, enum.Enum):
        assert _class_vars(node) == {
            m.name for m in python
        }, f"{name}: the model's members differ from the enum's"
        return

    methods = _methods(node)
    if "__init__" in methods:
        model_init = _model_params(methods["__init__"], drop_first=True)
    else:
        model_init = _model_fields(node)
    omitted = _OMITTED_PARAMETERS.get(name, set())
    python_init = [
        p for p in _python_params(python, drop_first=False) if p[1] not in omitted
    ]
    assert model_init == python_init, f"{name}(): the model's parameters differ"

    if dataclasses.is_dataclass(python) and "__init__" not in methods:
        python_fields = {f.name for f in dataclasses.fields(python)}
        declared = {p[1] for p in _model_fields(node)}
        assert declared <= python_fields, f"{name}: fields the dataclass does not have"

    for attr in _class_vars(node):
        assert hasattr(python, attr), f"{name}.{attr} is in the model only"

    for method_name, method in methods.items():
        if method_name.startswith("__") or method_name.startswith("_acs_"):
            continue
        assert hasattr(
            python, method_name
        ), f"{name}.{method_name} is in the model only"
        python_attr = inspect.getattr_static(python, method_name)
        if "property" in _decorators(method):
            assert isinstance(
                python_attr, property
            ), f"{name}.{method_name} is a property"
            continue
        if "staticmethod" in _decorators(method):
            # A classmethod in Python (EntityRole.ego): bound, it takes the same.
            assert isinstance(
                python_attr, (staticmethod, classmethod)
            ), f"{name}.{method_name} is not a static or class method in Python"
            assert _model_params(method, drop_first=False) == _python_params(
                getattr(python, method_name), drop_first=False
            ), f"{name}.{method_name}(): the model's parameters differ"
            continue
        assert _model_params(method, drop_first=True) == _python_params(
            getattr(python, method_name), drop_first=True
        ), f"{name}.{method_name}(): the model's parameters differ"


#: Model modules of one framework module each, which the library check
#: compiles a checked module against (typecheck/library.py): model module ->
#: the Python module it models and the names it declares, every one compared
#: with the Python object below.
_MODULE_MODELS: dict[str, tuple[Any, tuple[str, ...]]] = {
    "_bridge_pose": (bridge_base, ("BridgePose", "Quaternion", "Vector3")),
    "carla_install": (carla_install, ("installed_executable",)),
    "trajectory_recorder": (trajectory_recorder, ("TrajectoryRecorder",)),
    # What scenario_runner's _record_run reads of the unchecked environment.
    "environment": (environment, ("RunEnvironment", "capture")),
    "autoware_bridge/grpc_server": (bridge_grpc_server, ("GrpcAutowareBridgeServer",)),
    "autoware_stack/launcher": (
        stack_launcher,
        ("AutowareEpisode", "AutowareLauncher"),
    ),
    # What the checked autoware_stack.docker and the package take of the
    # unchecked devcontainer (yaml, json).
    "autoware_stack/devcontainer": (
        stack_devcontainer,
        (
            "DEFAULT_WORKSPACE_MOUNT",
            "DevContainer",
            "DevContainerError",
            "read_devcontainer",
        ),
    ),
    "driver/egodriver_client": (egodriver_client, ("EgoDriverGrpcClient",)),
    # MapSource is shared with the checked maps.source (REPLACED_MODELS).
    "_map_source": (maps_source, ("MapSource",)),
    "maps/cache": (
        maps_cache,
        (
            "CachedRepo",
            "DERIVED_SUBDIR",
            "GitMapCache",
            "MapCacheError",
            "is_lfs_pointer",
            "map_root",
        ),
    ),
    # A pydantic model: its fields by keyword only, as Python's signature.
    "ui/models": (ui_models, ("RunProgress", "Status")),
}

#: Classes the model declares only so a checked package can re-export them:
#: their constructors are a compile error, so a scenario cannot build one.
_UNBUILDABLE: dict[str, str] = {
    "AutowareLauncher": "autoware_stack/launcher",
    "CarlaDriverEntity": "entity",
    "EgoDriverGrpcClient": "driver/egodriver_client",
    "GrpcAutowareBridgeServer": "autoware_bridge/grpc_server",
}


def test_a_module_model_declares_only_the_compared_names() -> None:
    # A name the module lacks would pass a scenario's check and fail its run.
    for module, (_python, names) in _MODULE_MODELS.items():
        assert sorted(_definitions(module)) == sorted(names), module


_MODULE_MODEL_NAMES = [
    (module, name)
    for module, (_python, names) in _MODULE_MODELS.items()
    for name in names
    if name != "AutowareLauncher"  # a Protocol: compared on its own below
]


@pytest.mark.parametrize(("module", "name"), _MODULE_MODEL_NAMES)
def test_a_module_model_is_declared_as_in_python(module: str, name: str) -> None:
    python_module, _names = _MODULE_MODELS[module]
    # A cached function (environment.capture(), `@lru_cache`) is compared as
    # the function it wraps: the cache takes the same parameters.
    python = inspect.unwrap(getattr(python_module, name))
    _assert_declared_as_in_python(name, module, python)


def test_the_devcontainer_model_holds_python_s_values() -> None:
    # A constant is compared by its value, and the frozen dataclass is a
    # @tuple (an assignment to a field fails the check as it fails the run).
    definitions = _definitions("autoware_stack/devcontainer")
    constant = definitions["DEFAULT_WORKSPACE_MOUNT"][0]
    assert isinstance(constant, ast.Assign)
    assert (
        ast.literal_eval(constant.value) == stack_devcontainer.DEFAULT_WORKSPACE_MOUNT
    )
    node = definitions["DevContainer"][0]
    assert isinstance(node, ast.ClassDef)
    assert [ast.unparse(d) for d in node.decorator_list] == ["tuple"]
    params = stack_devcontainer.DevContainer.__dataclass_params__  # type: ignore[attr-defined]
    assert params.frozen
    assert issubclass(stack_devcontainer.DevContainerError, ValueError)


def test_the_launcher_model_is_the_protocol_s_methods() -> None:
    # Python cannot instantiate a Protocol, so the model's constructor is a
    # compile error (below); what AutowareEgoEntity and ScenarioQueue call is
    # declared as Python's.
    python = stack_launcher.AutowareLauncher
    assert getattr(python, "_is_protocol", False)
    node = _definitions("autoware_stack/launcher")["AutowareLauncher"][0]
    assert isinstance(node, ast.ClassDef)
    methods = {n: m for n, m in _methods(node).items() if n != "__init__"}
    assert sorted(methods) == ["close", "poll", "prepare", "start", "stop"]
    for method_name, method in methods.items():
        assert _model_params(method, drop_first=True) == _python_params(
            getattr(python, method_name), drop_first=True
        ), f"AutowareLauncher.{method_name}(): the model's parameters differ"


@pytest.mark.parametrize("name", sorted(_UNBUILDABLE))
def test_a_class_only_re_exported_cannot_be_built_in_a_scenario(name: str) -> None:
    node = _definitions(_UNBUILDABLE[name])[name][0]
    assert isinstance(node, ast.ClassDef)
    assert not node.bases, f"{name}: a base would bring methods to compare"
    methods = _methods(node)
    assert set(methods) <= {
        "__init__",
        "close",
        "poll",
        "prepare",
        "start",
        "stop",
    }, name
    body = methods["__init__"].body
    assert len(body) == 1 and isinstance(body[0], ast.Expr), name
    call = body[0].value
    assert isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    assert call.func.id == "compile_error", name


#: The model of the third-party carla_driver_interface (its geometry): what the
#: driver modules use of it, compared with the installed package.
_CDI_MODEL = model_dir() / "carla_driver_interface"
_CDI_NAMES = ("Pose", "Trajectory", "waypoints_to_proto")

#: (class, field) -> (the package's annotation, the model's): an array is
#: Codon's of its dtype and number of dimensions (the docstrings give them).
_CDI_FIELDS = {
    ("Pose", "position"): ("NDArray[np.float64]", "np.ndarray[np.float64, 1]"),
    ("Pose", "quat_xyzw"): ("NDArray[np.float64]", "np.ndarray[np.float64, 1]"),
    ("Trajectory", "timestamps_us"): ("List[int]", "List[int]"),
    ("Trajectory", "poses"): ("List[Pose]", "List[Pose]"),
}


def test_the_carla_driver_interface_model_declares_only_the_compared_names() -> None:
    # A name the package lacks would pass a scenario's check and fail its run:
    # whatever the model declares is compared below, dunder methods included.
    definitions = _definitions("geometry", _CDI_MODEL)
    assert sorted(definitions) == sorted(_CDI_NAMES)
    for name in _CDI_NAMES:
        node = definitions[name][0]
        if isinstance(node, ast.ClassDef):
            python = getattr(cdi_geometry, name)
            for method in _methods(node):
                assert hasattr(python, method), f"{name}.{method} is in the model only"


@pytest.mark.parametrize("name", _CDI_NAMES)
def test_the_carla_driver_interface_model_is_declared_as_in_the_package(
    name: str,
) -> None:
    _assert_declared_as_in_python(
        name, "geometry", getattr(cdi_geometry, name), root=_CDI_MODEL
    )


@pytest.mark.parametrize("name", ["Pose", "Trajectory"])
def test_the_carla_driver_interface_fields_have_the_packages_types(name: str) -> None:
    node = _definitions("geometry", _CDI_MODEL)[name][0]
    assert isinstance(node, ast.ClassDef)
    model = {
        stmt.target.id: ast.unparse(stmt.annotation)
        for stmt in node.body
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
    }
    python = getattr(cdi_geometry, name)
    init_fields = [f.name for f in dataclasses.fields(python) if f.init]
    assert list(model) == init_fields, f"{name}: the model's fields differ"
    for field_name, annotation in model.items():
        expected_python, expected_model = _CDI_FIELDS[(name, field_name)]
        assert (
            python.__annotations__[field_name] == expected_python
        ), f"{name}.{field_name} changed type in carla_driver_interface"
        assert annotation == expected_model, f"{name}.{field_name}"


#: The models of the third-party python-dotenv and ffmpeg-python: what the
#: checked server and camera recorder use of them, compared with the installed
#: packages.
_DOTENV_MODEL = model_dir() / "dotenv"
_FFMPEG_MODEL = model_dir() / "ffmpeg"

#: The ffmpeg-python stream classes the model declares, and their methods.
_FFMPEG_STREAMS = {
    "FilterableStream": ("output",),
    "OutputStream": ("overwrite_output", "run_async"),
}


def _public(definitions: dict[str, list[ast.stmt]]) -> list[str]:
    return sorted(name for name in definitions if not name.startswith("_"))


def test_the_dotenv_model_is_load_dotenv_as_in_the_package() -> None:
    assert _public(_definitions("__init__", _DOTENV_MODEL)) == ["load_dotenv"]
    _assert_declared_as_in_python(
        "load_dotenv", "__init__", dotenv.load_dotenv, root=_DOTENV_MODEL
    )


def test_the_ffmpeg_model_declares_only_the_compared_names() -> None:
    # A name the package lacks would pass a scenario's check and fail its run.
    assert _public(_definitions("__init__", _FFMPEG_MODEL)) == ["input"]
    nodes = _definitions("nodes", _FFMPEG_MODEL)
    assert _public(nodes) == sorted(_FFMPEG_STREAMS)
    for name, methods in _FFMPEG_STREAMS.items():
        node = nodes[name][0]
        assert isinstance(node, ast.ClassDef)
        assert sorted(_methods(node)) == sorted(methods), name


def test_the_ffmpeg_model_is_declared_as_in_the_package() -> None:
    _assert_declared_as_in_python("input", "__init__", ffmpeg.input, root=_FFMPEG_MODEL)
    nodes = _definitions("nodes", _FFMPEG_MODEL)
    for name, methods in _FFMPEG_STREAMS.items():
        node = nodes[name][0]
        assert isinstance(node, ast.ClassDef)
        python = getattr(ffmpeg.nodes, name)
        for method_name, method in _methods(node).items():
            function = getattr(python, method_name)
            # ffmpeg-python binds a module function to the stream classes: the
            # stream is its first parameter, or the first of its *args
            # (output(*streams_and_filename, **kwargs)).
            first = next(iter(inspect.signature(function).parameters.values()))
            bound = first.kind is not inspect.Parameter.VAR_POSITIONAL
            assert _model_params(method, drop_first=True) == _python_params(
                function, drop_first=bound
            ), f"{name}.{method_name}(): the model's parameters differ"


def test_the_ffmpeg_model_returns_the_packages_streams() -> None:
    # What each call returns, as the model's annotations say (nothing runs).
    source = ffmpeg.input("pipe:")
    assert type(source) is ffmpeg.nodes.FilterableStream
    output = source.output("out.mp4")
    assert type(output) is ffmpeg.nodes.OutputStream
    assert type(output.overwrite_output()) is ffmpeg.nodes.OutputStream
    assert not hasattr(source, "overwrite_output")
    assert not hasattr(source, "run_async")


def test_every_model_file_parses_as_python() -> None:
    # The tests above read the model with ast; keep it valid Python syntax.
    for path in sorted(model_dir().rglob("*.codon")):
        _parse(path)


# ---------------------------------------------------------------------------
# pyxodr's RoadNetwork, as coordinate.codon declares how it is loaded
# ---------------------------------------------------------------------------

_XODR_PATH = Path(__file__).resolve().parents[3] / "data" / "nishishinjuku_carla.xodr"


def _road_network_model() -> ast.ClassDef:
    node = _definitions("coordinate")["RoadNetwork"][0]
    assert isinstance(node, ast.ClassDef)
    return node


@pytest.mark.parametrize("method", ["__init__", "get_roads"])
def test_the_road_network_model_takes_pyxodr_s_parameters(method: str) -> None:
    # pyxodr's own signature, but for the trailing parameters with a default
    # the framework leaves at pyxodr's (the lane types to ignore).
    model = _model_params(_methods(_road_network_model())[method], drop_first=True)
    python = _python_params(getattr(PyxodrRoadNetwork, method), drop_first=True)
    assert model == python[: len(model)], method
    assert all(has_default for _kind, _name, has_default in python[len(model) :])


def test_the_road_network_model_loads_the_fixture_map() -> None:
    # What MapManager.initialize() and road_network do with it.
    network = PyxodrRoadNetwork(str(_XODR_PATH))
    roads = network.get_roads()
    assert isinstance(roads, list) and roads
    assert all(isinstance(road, PyxodrRoad) for road in roads)
    assert isinstance(network.road_ids_to_object, dict)
    assert network.road_ids_to_object
    for road_id, road in network.road_ids_to_object.items():
        assert isinstance(road_id, str)
        assert isinstance(road, PyxodrRoad)
