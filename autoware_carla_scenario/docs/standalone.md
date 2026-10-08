# Standalone binaries (work in progress)

A scenario that runs against Autoware can be compiled, with the framework's
own runner, into a native binary that needs no Python environment: the
binary is launched beside ROS and Autoware and talks to an already-running
CARLA. It is not a second implementation of the framework. Codon compiles the
framework's real modules, after a rewrite that keeps every line where its
author wrote it (`standalone/transform.py`).

!!! warning "Not usable yet"
    The build reaches the scenario's construction, the ego entity and the
    traffic backend for `intersection_passing` with `ego.entity=autoware`.
    The Autoware gRPC transport is a stand-in that refuses to start.

## What a binary contains

* **The framework modules the run reaches** (`standalone/collect.py`). Imports
  are followed from the scenario and the runner. Each package on the way gets
  a generated `__init__` that re-exports only the names the compiled modules
  use. Codon runs a package's `__init__` whenever one of its modules is
  imported, and the real ones re-export the editor, Hydra and the sweeper.
* **Boundary modules** the runtime provides instead of the source:
    * `coordinate.map_manager`, `projection` and `transform` work on map
      geometry baked at build time, instead of Lanelet2 and pyxodr.
    * `autoware_bridge.grpc_server` is the gRPC bridge.
    * `server` connects to a running CARLA and never launches one.
* **Stand-ins for the standard library Codon lacks** (`standalone/codon/*.codon`):
  `enum`, `abc`, `pathlib`, `logging` and the others.
* **An empty stub for every other external package** that a compiled module
  imports. A module-level import resolves. A function that uses one on the
  run path is a build error that names it.
* **typesafe_carla's Codon library**, the CARLA client.

## What the rewrite does, and why

Most of the rewrites are for Codon 0.19 behaviour that differs from Python's.
The rewrite either:

* stays on the line it rewrites, or
* inserts whole lines marked `# acs:inserted`.

A build error is therefore reported at the author's file and line.

| Python | Codon 0.19 | Rewrite |
| --- | --- | --- |
| `a or b` | typed as a Union; an `Optional[T] or T` fails at run time | `_acs_or(a, lambda: b)`; `a or []` is `_acs_or_empty(a)` |
| `x if x is not None else d` | both branches need one type | `_acs_default(x, lambda: d)` |
| Required keyword-only parameter after defaulted ones | refused | A sentinel default, checked on entry |
| `Optional[C]` parameter | takes neither a subclass of C nor an `Optional` of one | Generic, converted on entry |
| `name: T = v` in a plain class body | An instance field, left at zero by a hand-written `__init__` | A class variable read through a dispatched accessor (below) |
| A method default naming a class constant | Looked up in the module | The constant inlined |
| `type[C]` as a value | No type for it | `_AcsClass[C]`, which constructs a C when called |
| `class E(RuntimeError)` | Makes the built-in polymorphic, which then cannot be raised | `class E(Static[RuntimeError])` with Codon's two constructors |
| `Enum` | No `enum` module | A class with singleton members |
| `@abstractmethod`, `ABC`, `Protocol`, `@runtime_checkable`, `__slots__` | Refused or miscompiled | Dropped |
| `@classmethod` | None | `@staticmethod`, with `cls` meaning the class |
| `carla.TrafficLightState` and other CARLA enums | Values, whose members are ints | `int` |
| `IO[str]` | `File` | `File` |

### Rewrites that need the whole program

Some rewrites need more than one module (`standalone/hierarchy.py`):

* **Methods overridden two levels down.** Take a method that one class defines and a class further down overrides. Codon miscompiles a call to it through a class in between that only inherits it. Every such class gets a forwarding definition of its own. The forwarder spells out the parameter types, because Codon cannot dispatch a method whose parameters are untyped.
* **Class attributes.** `use_autopilot: bool = True` on `EgoVehicle` and `False` on `AutowareEgoEntity` stays a class variable on each class. Every `obj.use_autopilot` becomes `obj._acs_get_use_autopilot()`, which is dispatched as Python's attribute lookup is. An attribute that is zero wherever it is given keeps its field, because zero is where a Codon field starts.
* **Imports under `if TYPE_CHECKING:`.** Codon runs them, because the typing stand-in sets `TYPE_CHECKING`. It compiles no import cycle at all. Each one that closes no cycle is kept; the others are dropped, together with the annotations that name them.

## Limits found so far

These are limits of Codon 0.19 that the rewrite cannot remove:

* **No import cycles.** A name that two modules need from each other has to
  live in a third module. `EgoConfig` moved to `entity/ego_config.py` for this
  reason, and `scenario_base` re-exports it.
* **No dispatch of a method whose parameters are untyped.** A method that a
  subclass overrides and that is called through the base class needs concrete
  parameter types. `Any` and a dropped annotation do not count.
* **No keyword arguments and no omitted defaults through a dispatched call.**
  Every argument has to be passed by position.
* **An exception is caught only as its own class**, not as one of its bases.
