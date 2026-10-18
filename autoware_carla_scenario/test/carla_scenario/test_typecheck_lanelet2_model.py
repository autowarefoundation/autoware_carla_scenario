"""The Codon model of lanelet2 declares what the installed lanelet2 has.

``typecheck/codon/lanelet2/`` and ``typecheck/codon/autoware_lanelet2_extension_python/``
model the Lanelet2 API the checked framework modules read (docs/typecheck.md).
lanelet2 is a compiled extension, so its signatures cannot be introspected:
these tests read the model as Python syntax and check that

* every class, member and function it declares exists in the installed
  package, and every base class it names is a base in Python too (or the
  Const twin of a mutable class, which the model derives from it, or
  lanelet2's ``Projector`` of a projector, as C++ derives it);
* the parameter names and defaults agree with the stubs the package ships
  (``lanelet2/*.pyi``), where those name them;
* each declared call, made on the fixture map with the model's arity,
  returns what the model says it returns.

They need no Codon.
"""

from __future__ import annotations

import ast
import importlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

# autoware_lanelet2_extension_python must be imported before lanelet2: its
# regulatory elements are registered by the import, and the fixture map has them.
import autoware_lanelet2_extension_python.regulatory_elements as autoware_res
from autoware_lanelet2_extension_python.projection import (
    MGRSProjector,
    TransverseMercatorProjector,
)
import lanelet2
import lanelet2.core
import lanelet2.geometry
import lanelet2.io
import lanelet2.projection
import lanelet2.routing
import lanelet2.traffic_rules

from autoware_carla_scenario.typecheck import model_dir

_PACKAGES = ("lanelet2", "autoware_lanelet2_extension_python")
_OSM_PATH = Path(__file__).resolve().parents[3] / "data" / "nishishinjuku.osm"

#: Mutable class -> its Const twin, which the model derives it from although
#: Python does not (lanelet2 converts the one into the other).
_TWINS = {
    "Lanelet": "ConstLanelet",
    "LineString2d": "ConstLineString2d",
    "LineString3d": "ConstLineString3d",
    "Point3d": "ConstPoint3d",
}

#: Projectors the model derives from lanelet2's Projector, as C++ does
#: (lanelet2.io.load() takes any of them), though Python registers none as
#: derived from it.
_PROJECTORS = {
    "LocalCartesianProjector",
    "MGRSProjector",
    "TransverseMercatorProjector",
    "UtmProjector",
}


def _model_modules() -> dict[str, ast.Module]:
    """Dotted module name -> its model, for both packages."""
    out: dict[str, ast.Module] = {}
    for package in _PACKAGES:
        root = model_dir() / package
        for path in sorted(root.glob("*.codon")):
            name = package if path.stem == "__init__" else f"{package}.{path.stem}"
            out[name] = ast.parse(path.read_text(), filename=str(path))
    return out


_MODELS = _model_modules()


def _public(name: str) -> bool:
    return not name.startswith("_") or (name.startswith("__") and name.endswith("__"))


def _classes() -> list[tuple[str, ast.ClassDef]]:
    return [
        (module, node)
        for module, tree in _MODELS.items()
        for node in tree.body
        if isinstance(node, ast.ClassDef) and _public(node.name)
    ]


def _functions() -> list[tuple[str, ast.FunctionDef]]:
    return [
        (module, node)
        for module, tree in _MODELS.items()
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    ]


def _members(cls: ast.ClassDef) -> list[str]:
    names: list[str] = []
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name != "__init__":
            names.append(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.append(node.target.id)
    return [n for n in names if _public(n)]


def _stub_params(
    module: str, owner: str | None, name: str
) -> list[tuple[str, str | None]] | None:
    """``(name, default)`` of each parameter the package's stub names for a
    function (``None``: no default), or ``None`` when there is no stub or it
    takes ``*args``."""
    # The submodules are the extension's, with no file of their own: the
    # stubs lie next to the package's __init__.py.
    package, _dot, rest = module.partition(".")
    root = Path(importlib.import_module(package).__file__ or "").parent
    stub = root / f"{rest or '__init__'}.pyi"
    if not stub.is_file():
        return None
    body = ast.parse(stub.read_text()).body
    if owner is not None:
        classes = [n for n in body if isinstance(n, ast.ClassDef) and n.name == owner]
        if not classes:
            return None
        body = classes[0].body
    for node in body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            if node.args.vararg is not None:
                return None
            return _with_defaults(node.args)
    return None


def _with_defaults(args: ast.arguments) -> list[tuple[str, str | None]]:
    params = args.posonlyargs + args.args
    defaults: list[str | None] = [None] * (len(params) - len(args.defaults))
    defaults += [ast.unparse(d) for d in args.defaults]
    return [(p.arg, d) for p, d in zip(params, defaults)]


def test_the_model_covers_both_packages() -> None:
    assert set(_MODELS) >= {
        "lanelet2",
        "lanelet2.core",
        "lanelet2.geometry",
        "lanelet2.io",
        "lanelet2.projection",
        "lanelet2.routing",
        "lanelet2.traffic_rules",
        "autoware_lanelet2_extension_python.projection",
        "autoware_lanelet2_extension_python.regulatory_elements",
    }


@pytest.mark.parametrize(
    ("module", "cls"), _classes(), ids=lambda v: getattr(v, "name", v)
)
def test_a_modelled_class_and_its_members_exist(module: str, cls: ast.ClassDef) -> None:
    runtime = getattr(importlib.import_module(module), cls.name)
    for member in _members(cls):
        assert hasattr(runtime, member), f"{module}.{cls.name}.{member}"
    mro = [c.__name__ for c in runtime.__mro__]
    for base in cls.bases:
        assert isinstance(base, ast.Name)
        assert (
            base.id in mro
            or _TWINS.get(cls.name) == base.id
            or (cls.name in _PROJECTORS and base.id == "Projector")
        ), (
            f"{module}.{cls.name} derives from {base.id} in the model, "
            f"not in Python ({mro})"
        )


@pytest.mark.parametrize(("mutable", "const"), sorted(_TWINS.items()))
def test_a_mutable_class_has_every_member_of_its_const_twin(
    mutable: str, const: str
) -> None:
    const_model = dict((c.name, c) for _m, c in _classes())[const]
    for member in _members(const_model):
        assert hasattr(getattr(lanelet2.core, mutable), member), f"{mutable}.{member}"


@pytest.mark.parametrize(
    ("module", "fn"), _functions(), ids=lambda v: getattr(v, "name", v)
)
def test_a_modelled_function_exists_with_the_stub_s_parameters(
    module: str, fn: ast.FunctionDef
) -> None:
    assert hasattr(importlib.import_module(module), fn.name), f"{module}.{fn.name}"
    _assert_stub_parameters(module, None, fn)


def _assert_stub_parameters(
    module: str, owner: str | None, fn: ast.FunctionDef
) -> None:
    stub = _stub_params(module, owner, fn.name)
    if stub is None:
        return
    model = _with_defaults(fn.args)
    if owner is not None:
        stub, model = stub[1:], model[1:]
    where = f"{module}.{owner + '.' if owner else ''}{fn.name}"
    assert [n for n, _d in model] == [
        n for n, _d in stub[: len(model)]
    ], f"{where}: model {model}, stub {stub}"
    for (name, default), (_n, stub_default) in zip(model, stub):
        # Where the model gives a default, it is Python's (False, 0, 0.0...).
        if default is not None and stub_default is not None:
            assert ast.literal_eval(default) == ast.literal_eval(
                stub_default
            ), f"{where}({name})"
    # A parameter the model leaves out has a default in Python.
    assert all(d is not None for _n, d in stub[len(model) :]), where


@pytest.mark.parametrize(
    ("module", "cls"), _classes(), ids=lambda v: getattr(v, "name", v)
)
def test_a_modelled_method_has_the_stub_s_parameters(
    module: str, cls: ast.ClassDef
) -> None:
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and not any(
            isinstance(d, ast.Name) and d.id == "property" for d in node.decorator_list
        ):
            _assert_stub_parameters(module, cls.name, node)


def test_the_traffic_rules_constants_are_python_s() -> None:
    tree = _MODELS["lanelet2.traffic_rules"]
    for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
        for node in cls.body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.Constant):
                assert isinstance(node.target, ast.Name)
                runtime = getattr(
                    getattr(lanelet2.traffic_rules, cls.name), node.target.id
                )
                assert runtime == node.value.value, f"{cls.name}.{node.target.id}"


# ---------------------------------------------------------------------------
# The declared calls, on the fixture map
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lanelet_map() -> Iterator[Any]:
    yield lanelet2.io.load(
        str(_OSM_PATH), MGRSProjector(lanelet2.io.Origin(0.0, 0.0, 0.0))
    )


@pytest.fixture(scope="module")
def graph(lanelet_map: Any) -> Any:
    rules = lanelet2.traffic_rules.create(
        lanelet2.traffic_rules.Locations.Germany,
        lanelet2.traffic_rules.Participants.Vehicle,
    )
    assert isinstance(rules, lanelet2.traffic_rules.TrafficRules)
    return lanelet2.routing.RoutingGraph(lanelet_map, rules)


def test_the_map_and_its_lanelets(lanelet_map: Any) -> None:
    core = lanelet2.core
    assert isinstance(lanelet_map, core.LaneletMap)
    layer = lanelet_map.laneletLayer
    assert isinstance(layer, core.LaneletLayer)
    assert isinstance(len(layer), int)
    lanelet = next(iter(layer))
    assert isinstance(lanelet, core.Lanelet)
    assert isinstance(lanelet.id, int)
    assert isinstance(layer[lanelet.id], core.Lanelet)
    assert layer.exists(lanelet.id) is True
    with pytest.raises(RuntimeError):
        layer[-1]
    for bound in (lanelet.leftBound, lanelet.rightBound):
        usages = layer.findUsages(bound)
        assert isinstance(usages, list)
        assert all(isinstance(u, core.Lanelet) for u in usages)
        assert lanelet.id in {u.id for u in usages}
    assert isinstance(lanelet.attributes, core.AttributeMap)
    assert isinstance("subtype" in lanelet.attributes, bool)
    assert isinstance(lanelet.attributes["subtype"], str)
    assert isinstance(len(lanelet.attributes), int)
    # items(): a list of (key, value) pairs of strings, in key order.
    items = lanelet.attributes.items()
    assert isinstance(items, list) and items
    assert all(
        isinstance(i, tuple) and [type(x) for x in i] == [str, str] for i in items
    )
    assert [k for k, _v in items] == sorted(k for k, _v in items)
    assert dict(items)["subtype"] == lanelet.attributes["subtype"]
    centerline = lanelet.centerline
    assert isinstance(centerline, core.ConstLineString3d)
    # A Lanelet's bounds are its mutable twin's; the model reads them as Const.
    assert isinstance(lanelet.leftBound, (core.ConstLineString3d, core.LineString3d))
    assert isinstance(lanelet.rightBound, (core.ConstLineString3d, core.LineString3d))
    assert isinstance(centerline.id, int)
    assert isinstance(centerline.attributes, core.AttributeMap)
    assert isinstance(len(centerline), int)
    assert isinstance(centerline[0], core.ConstPoint3d)
    for point in centerline:
        assert isinstance(point, core.ConstPoint3d)
        assert isinstance(point.id, int)
        assert isinstance(point.attributes, core.AttributeMap)
        assert all(isinstance(v, float) for v in (point.x, point.y, point.z))
    assert isinstance(lanelet.regulatoryElements, list)
    point = core.BasicPoint2d(1.0, 2.0)
    assert (point.x, point.y) == (1.0, 2.0)


def _model_roles() -> dict[str, str]:
    """Role -> the class the model's ``ConstRuleParameterMap.__getitem__``
    says its parameters are, read off its ``if key == ...`` branches."""
    core = dict((c.name, c) for _m, c in _classes())["ConstRuleParameterMap"]
    getitem = next(
        n
        for n in core.body
        if isinstance(n, ast.FunctionDef) and n.name == "__getitem__"
    )
    roles: dict[str, str] = {}
    branch: ast.stmt | None = getitem.body[0]
    while isinstance(branch, ast.If):
        keys = [
            n.comparators[0].value
            for n in ast.walk(branch.test)
            if isinstance(n, ast.Compare) and isinstance(n.comparators[0], ast.Constant)
        ]
        returned = branch.body[0]
        assert isinstance(returned, ast.Return) and isinstance(returned.value, ast.Call)
        assert isinstance(returned.value.func, ast.Subscript)
        cls = returned.value.func.slice
        assert isinstance(cls, ast.Name)
        roles.update({str(key): cls.id for key in keys})
        branch = branch.orelse[0] if branch.orelse else None
    return roles


#: A model class that is no Python class: a parameter of a role that holds
#: any of lanelet2's primitives, of which the model declares the members
#: every primitive has.
_ANY_PRIMITIVE = "_RuleParameter"


def test_the_model_declares_the_roles_the_framework_reads() -> None:
    assert _model_roles() == {
        "ref_line": "ConstLineString3d",
        "cancel_line": "ConstLineString3d",
        "light_bulbs": "ConstLineString3d",
        "right_of_way": "ConstLanelet",
        "yield": "ConstLanelet",
        "refers": _ANY_PRIMITIVE,
    }


def test_the_regulatory_elements(lanelet_map: Any) -> None:
    core = lanelet2.core
    roles = _model_roles()
    any_primitive = next(
        c
        for c in ast.walk(_MODELS["lanelet2.core"])
        if isinstance(c, ast.ClassDef) and c.name == _ANY_PRIMITIVE
    )
    seen: set[str] = set()
    held: dict[str, set[str]] = {}
    for lanelet in lanelet_map.laneletLayer:
        for element in lanelet.regulatoryElements:
            assert isinstance(element, core.RegulatoryElement)
            assert isinstance(element.id, int)
            assert isinstance(element.attributes, core.AttributeMap)
            params = element.parameters
            assert isinstance(params, core.ConstRuleParameterMap)
            assert isinstance(len(params), int)
            for role, cls in roles.items():
                assert isinstance(role in params, bool)
                if role not in params:
                    continue
                for p in params[role]:
                    held.setdefault(role, set()).add(type(p).__name__)
                    if cls == _ANY_PRIMITIVE:
                        for member in _members(any_primitive):
                            assert hasattr(p, member), f"{role}: {type(p)}.{member}"
                        assert isinstance(p.id, int)
                        assert isinstance(p.attributes, core.AttributeMap)
                    else:
                        assert type(p).__name__ == cls, f"{role}: {type(p)}"
            if isinstance(element, (core.TrafficLight, core.RightOfWay)):
                stop_line = element.stopLine
                assert stop_line is None or isinstance(stop_line, core.LineString3d)
            seen.add(type(element).__name__)
    # The fixture map has every class the model declares a branch for.
    assert {"AutowareTrafficLight", "RightOfWay", "TrafficSign", "RoadMarking"} <= seen
    assert issubclass(autoware_res.AutowareTrafficLight, core.TrafficLight)
    # "refers" is declared as any primitive because it holds more than one
    # class: polygons (the detection areas) besides line strings.
    assert len(held["refers"]) > 1, held["refers"]


def test_the_geometry_functions(lanelet_map: Any) -> None:
    core, geometry = lanelet2.core, lanelet2.geometry
    lanelet = next(iter(lanelet_map.laneletLayer))
    centerline_2d = geometry.to2D(lanelet.centerline)
    assert isinstance(centerline_2d, core.ConstLineString2d)
    assert isinstance(centerline_2d.id, int)
    assert isinstance(len(centerline_2d), int)
    mutable = core.LineString3d(
        lanelet2.core.getId(),
        [
            core.Point3d(lanelet2.core.getId(), 0.0, 0.0, 0.0),
            core.Point3d(lanelet2.core.getId(), 3.0, 4.0, 0.0),
        ],
    )
    assert isinstance(geometry.to2D(mutable), core.LineString2d)
    for line in (lanelet.centerline, centerline_2d, mutable, geometry.to2D(mutable)):
        assert isinstance(geometry.length(line), float)
    first = lanelet.centerline[0]
    point = core.BasicPoint2d(first.x, first.y)
    arc = geometry.toArcCoordinates(centerline_2d, point)
    assert isinstance(arc, geometry.ArcCoordinates)
    assert isinstance(arc.length, float) and isinstance(arc.distance, float)
    assert isinstance(
        geometry.toArcCoordinates(geometry.to2D(mutable), point),
        geometry.ArcCoordinates,
    )
    found = geometry.findNearest(lanelet_map.laneletLayer, point, 2)
    assert isinstance(found, list) and len(found) == 2
    for distance, nearest in found:
        assert isinstance(distance, float)
        assert isinstance(nearest, core.Lanelet)
    assert isinstance(geometry.inside(lanelet, point), bool)
    assert isinstance(geometry.length2d(lanelet), float)


def test_the_routing_graph(lanelet_map: Any, graph: Any) -> None:
    core = lanelet2.core
    lanelets = list(lanelet_map.laneletLayer)
    neighbours = 0
    conflicts = 0
    for lanelet in lanelets:
        for walk in (graph.following, graph.previous):
            for result in (
                walk(lanelet),
                walk(lanelet, False),
                walk(lanelet, withLaneChanges=True),
            ):
                assert isinstance(result, list)
                assert all(isinstance(r, core.ConstLanelet) for r in result)
        for side in (graph.left, graph.right, graph.adjacentLeft, graph.adjacentRight):
            for result in (
                side(lanelet),
                side(lanelet, 0),
                side(lanelet, routingCostId=0),
            ):
                assert result is None or isinstance(result, core.ConstLanelet)
                neighbours += result is not None
        following = graph.following(lanelet)
        if following:
            # A ConstLanelet goes back into the graph and the geometry.
            assert isinstance(graph.following(following[0]), list)
            assert isinstance(following[0].centerline, core.ConstLineString3d)
            assert isinstance(following[0].regulatoryElements, list)
            assert isinstance(
                lanelet2.geometry.inside(following[0], core.BasicPoint2d(0.0, 0.0)),
                bool,
            )
            assert isinstance(lanelet2.geometry.length2d(following[0]), float)
        for result in (graph.conflicting(lanelet), graph.conflicting(lanelet, 0)):
            # simple-lanelet2 hands out the lanelets, not ConstLaneletOrAreas.
            assert isinstance(result, list)
            assert all(type(r) is core.ConstLanelet for r in result)
            conflicts += len(result)
    assert neighbours > 0
    assert conflicts > 0


def test_the_shortest_path(lanelet_map: Any, graph: Any) -> None:
    core = lanelet2.core
    lanelets = list(lanelet_map.laneletLayer)
    paths = 0
    unreachable = 0
    for lanelet in lanelets:
        following = graph.following(lanelet)
        if not following:
            continue  # not in the vehicles' graph (a crosswalk), or a dead end
        # To itself: a path of that one lanelet.
        alone = graph.shortestPath(lanelet, lanelet)
        assert isinstance(alone, lanelet2.routing.LaneletPath)
        assert [ll.id for ll in alone] == [lanelet.id]
        # A ConstLanelet (what the graph hands out) goes back in, as the
        # model's ConstLanelet parameters say.
        for path in (
            graph.shortestPath(lanelet, following[0]),
            graph.shortestPath(lanelet, following[0], 0),
            graph.shortestPath(following[0], following[0], 0, True),
            graph.shortestPath(
                lanelet, following[0], routingCostId=0, withLaneChanges=False
            ),
        ):
            assert isinstance(path, lanelet2.routing.LaneletPath)
            assert isinstance(len(path), int)
            assert all(type(ll) is core.ConstLanelet for ll in path)
            assert type(path[0]) is core.ConstLanelet
            assert path[len(path) - 1].id == following[0].id
            paths += 1
        # No path back against the direction of travel: None.
        back = graph.shortestPath(following[0], lanelet)
        assert back is None or isinstance(back, lanelet2.routing.LaneletPath)
        unreachable += back is None
    assert paths > 0
    assert unreachable > 0


def test_the_projector_and_origin() -> None:
    assert isinstance(lanelet2.io.Origin(35.0, 139.0), lanelet2.io.Origin)
    origin = lanelet2.io.Origin(lat=35.0, lon=139.0, alt=0.0)
    assert isinstance(MGRSProjector(origin), MGRSProjector)


_PROJECTOR_CLASSES = [
    MGRSProjector,
    TransverseMercatorProjector,
    lanelet2.projection.UtmProjector,
    lanelet2.projection.LocalCartesianProjector,
]


def test_the_model_declares_each_projector() -> None:
    declared = {cls.name for _module, cls in _classes() if cls.name in _PROJECTORS}
    assert declared == {p.__name__ for p in _PROJECTOR_CLASSES}
    assert "Projector" in {cls.name for _module, cls in _classes()}


@pytest.mark.parametrize("projector", _PROJECTOR_CLASSES, ids=lambda p: p.__name__)
def test_a_projector_projects_a_gps_point_and_loads_the_map(projector: Any) -> None:
    # What coordinate.map_manager and sweeper.map_loader do with one.
    core = lanelet2.core
    built = projector(origin=lanelet2.io.Origin(35.68, 139.69))
    point = core.GPSPoint(35.68, 139.69, 1.0)
    assert (point.lat, point.lon, point.alt) == (35.68, 139.69, 1.0)
    assert isinstance(core.GPSPoint(35.68, 139.69), core.GPSPoint)
    projected = built.forward(point)
    assert isinstance(projected, core.BasicPoint3d)
    assert all(isinstance(v, float) for v in (projected.x, projected.y, projected.z))
    for loaded in (
        lanelet2.io.load(str(_OSM_PATH), built),
        lanelet2.io.load(filename=str(_OSM_PATH), projector=built),
    ):
        assert isinstance(loaded, core.LaneletMap)
        assert len(loaded.laneletLayer) > 0
