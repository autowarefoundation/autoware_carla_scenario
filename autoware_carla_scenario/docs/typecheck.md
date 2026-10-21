# Static Check (Codon)

Before a scenario runs, the runner compiles it with
[Codon](https://github.com/exaloop/codon), a statically typed compiler for
Python syntax, and refuses it when it does not compile. A misspelled method, a
`str` where a `float` belongs, a condition missing its `label`, a turn direction
passed where a lane-change direction is expected, or a YAML value of the wrong
type is reported with the file and line it is on, before CARLA is started:

```text
$ uv run scenario scenario=intersection_passing/straight scenario.timeout_seconds=fast
autoware_carla_scenario.examples.intersection_passing.IntersectionPassingScenario: static check failed:
  scenario.timeout_seconds: error: 'str' does not match expected type 'float'
```

The scenario still runs as Python, exactly as before: the check compiles the
same source, never runs what it compiled, and changes nothing at run time.
The CARLA API a scenario calls is
[typesafe_carla](https://github.com/hakuturu583/typesafe_carla), the statically
typed CARLA client this framework runs on. A scenario imports it as
`import typesafe_carla.carla as carla`: at run time that is typesafe_carla's
CPython package, and for the check it is typesafe_carla's Codon library, so
the same names are checked that run.

## Running it

The runner checks every scenario registered with `register_scenario()` when
it builds it, so a run, a batch (`scenario='lane_change/*'`) and every job of a
sweep are checked before anything starts. To check without running anything:

```bash
uv run scenario-check                                   # every scenario config
uv run scenario-check scenario=intersection_passing/straight
uv run scenario-check scenario='lane_change/*' scenario.timeout_seconds=20
```

`scenario-check` composes each config exactly as the runner does, so the same
overrides apply. It exits with 0 when every scenario passed, 1 when one
failed, and 2 when there is no Codon compiler. A scenario package can make
the same check in its own tests, with no simulator:

```python
from autoware_carla_scenario.typecheck import typecheck_scenario

def test_my_scenario_compiles() -> None:
    result = typecheck_scenario(MyScenario, MyScenarioConfig, {"timeout_seconds": 10.0})
    assert result.ok, result.format()
```

### When it runs: the `typecheck` key

| Value | Effect |
|---|---|
| `auto` (default) | Check when a Codon compiler is installed; warn and run unchecked when none is. |
| `required` | Refuse to run without a Codon compiler. |
| `off` | Do not check. |

Set it like any other key (`uv run scenario ... typecheck=required`) or with
`AUTOWARE_CARLA_SCENARIO_TYPECHECK`, which takes precedence. A scenario
registered with `register_scenario_builder()` (a custom builder, such as the
packages the Scenario Editor exports) is not checked: only its builder knows
how it is constructed.

### Keeping verdicts between runs

A process compiles a program once and keeps the verdict for as long as it
runs. The program is keyed by its shape: the driver with each config value
replaced by its type (a bool, a float, an int, a string, an enum's class),
since Codon types it statically and reads no value at compile time. The
cases of a sweep that differ only in their values are therefore one program,
compiled for the first case and reused, verdict and diagnostics, for the
others (logged as "reusing the static check made for the config with the
same types"); a failure refuses each of them. `scenario-check` checks
configs side by side, and one whose program is being compiled when it
starts waits for that compile rather than starting its own. What changes the shape
compiles again: another key or order of keys, a value of another type (an
int where the first case had a float in a field not typed `float`, `None`
where it had a value), another length of a list, a mapping the config
converts in one case and not in another. A value whose literal Codon could
refuse on its own (an int beyond 64 bits, a string with an escape) is kept
as it is.

Verdicts are kept beyond the process, one file per program, in
`$XDG_CACHE_HOME/autoware_carla_scenario/typecheck` (`~/.cache/...` without
an absolute `XDG_CACHE_HOME`): a later process with the same inputs reads the
verdict instead of compiling. Set `AUTOWARE_CARLA_SCENARIO_TYPECHECK_CACHE`
to use another directory, or to an empty string to keep each verdict for its
process only (as happens anyway without a home directory, or with one that
cannot be written). Nothing prunes the directory; delete it, or its files
not accessed for a while, to reclaim the space. The key covers
everything the verdict depends on. For the compile, that is the scenario's
(or the framework's) modules, the generated driver's shape, the model,
typesafe_carla's Codon library, and the compiler's version and standard
library. For the None check, which the scenario, ODD and library checks all
run, it is every Python source and stub of the framework, the checker
included (mypy reads them for the types of what a scenario calls, and the
CARLA stubs are written by its `carla_stubs.py` from typesafe_carla's Codon
library, already in the key), the name and version of every other installed distribution
(mypy follows a scenario's imports into numpy, lanelet2 or typesafe_carla),
the mypy version, and the Python version. Where a version says nothing of
what mypy reads, the files are in it by path, size and modification time:
the `.py`, `.pyi` and `py.typed` files of each distribution installed
editable and of each directory on `MYPYPATH` and `PYTHONPATH`, which mypy
searches (a relative entry is taken from the directory the check runs in;
symlinked directories are followed). Of the packages being checked, whose
modules the key has by their text already, only their stubs count there, so
an edit to one scenario of an editable or `PYTHONPATH` package does not
stale the verdicts on the others. And it includes what the checks made in
Python found before the compile: an import the framework's Python lacks, or
an abstract class that is built. These digests are taken once a process
(about 100 ms, more with a large `PYTHONPATH` such as a ROS workspace's).
mypy runs with `--config-file=` and in an empty directory of its own, so
neither a configuration file nor a module of the directory the check runs
from reaches the verdict. A scenario package is therefore found as an
installed or editable distribution, or through `MYPYPATH`: give a namespace
package (one without `__init__.py`) an `__init__.py`, as the scaffolded
packages have. What the key still leaves out: a distribution
installed in development mode without PEP 610's `direct_url.json` (an old
`setup.py develop`, a `.pth` file written by hand); clear the directory after
editing one.
A crash is never kept, and an entry
that cannot be read counts as a miss. A verdict read from the directory has
its file's times touched, so the directory can be pruned by access time.

The test suite shares such a directory between its xdist workers. CI keeps
one from run to run, so a run compiles only the programs whose inputs changed.

### Installing Codon and typesafe_carla

The check needs typesafe_carla (the `typesafe-carla` package, from PyPI) and
the Codon compiler it pins (`typesafe-carla-toolchain`, 0.19, which
typesafe-carla depends on). Both ship for Linux x86_64 and aarch64, and both are
run-time dependencies of the framework anyway: typesafe_carla is its CARLA
client (see [installation](installation.md)).

The checker finds Codon the way typesafe_carla's `typesafe-codon` launcher
does (it calls typesafe_carla's own lookup), so one setup serves both:
`$TYPESAFE_CODON` (a `codon` executable), the `typesafe-carla-toolchain`
package, `$CODON_DIR/bin/codon`, `~/.codon/bin/codon`, then `codon` on `PATH`.
It runs it with the launcher's environment (`CODON_DIR`, and `LD_LIBRARY_PATH`
for the bundled runtime).
The check is written for Codon 0.19: a Codon of another release series counts
as no Codon at all, so `auto` warns and `required` refuses. So does a missing
typesafe-carla.

## What is checked

The checker copies the scenario's own modules (the module defining the
scenario class, its config class, and every module of the same package they
import) into a scratch directory, next to a typed model of the framework, and
compiles a small program that does what the runner does:

```python
config = MyScenarioConfig()          # then one line per YAML value:
config.timeout_seconds = 10.0        #   scenario.timeout_seconds
scenario = MyScenario(ego, config=config, spawn_pose=..., ground_projection=...)
scenario.setup()
scenario.is_done()
```

Codon checks a function only when something calls it. The program checks
everything `setup()` and `is_done()` reach: the calls into the framework, the
scenario's own helpers, and, through `register_pass_condition()` and the
other `register_*` methods, the `check()` of a custom condition and the
`execute()` of a custom action.

**Then it compiles the rest of the scenario's own modules too** (#192). Each
module the checker copies gets the library check's function appended
([*What is compiled*](#what-is-compiled)), and the program calls it after the
lines above:

```python
def _acs_library_check():                      # appended to my_scenario.py
    MyScenario(_acs_value(EgoConfig), _acs_value(Lanelet2Pose), ...)
    _acs_value(MyScenario)._goal_ids(_acs_value(int))    # a helper setup() never calls
    _acs_value(MyScenario).initial_speeds_mps(_acs_value(Optional[carla.Actor]))
    unused_probe(_acs_value(carla.World))                # a module-level function
    _acs_cls_default_9: int = "three"                    # MyScenario.RETRIES: int = "three"
```

So an error is reported wherever it is in those modules, not only on the path
`setup()` takes: in a private helper or a public method nothing calls yet, in
a hook override the runner calls later (`initial_speeds_mps()`), in a
module-level function, in a class-level default, or in the config class's
methods. Each call passes a value of each parameter's annotated type, and a
defaulted parameter's default is compiled by one more call that leaves it out.

**The result of a hook override is checked against the framework's type**
(#238). Codon compiles an override against its own annotation only, and the
framework code that calls it is a stub of the model, so `check() -> bool`,
`initial_speeds_mps() -> list[float]` or `is_done() -> str` compiled, and
failed or misbehaved when the scenario ran. So each appended call of a method
passes its result to the model, which checks it strictly against the return
type of the method it overrides, when the class derives from `BaseScenario`,
`BaseCondition`, `BaseAction` or `EgoVehicle` (directly, or through a built-in
condition or another class of the scenario's modules):

```python
    _acs_hook_return(MyScenario, "is_done", _acs_value(MyScenario).is_done())
    _acs_hook_return(GapCondition, "check",
                     _acs_value(GapCondition).check(_acs_value(carla.World), _acs_value(float)))
```

Every method and property each of the four classes declares is such a hook,
with the return type the model declares for it (`test_typecheck_model.py`
keeps the two in step): `setup()` returns `None`, `is_done()` a `bool`,
`initial_speeds_mps()` a `Dict[int, float]`, `check()` an
`Optional[ScenarioResult]`, `execute()` `None`, `debug_markers()` a
`DebugMarkers`, and so on. The check is strict where an annotated assignment
is not (Codon 0.19 converts any value with `__bool__` in `done: bool = x`):
an override must return the type itself or what Python's typing takes for
it -- a subclass of a class, an `int` for a `float`, a `bool` for an `int`,
and, for an `Optional[X]`, an `X`, an `Optional` of one, or only `None` (a
condition that never decides). A `str`, an `int` or an `Optional[bool]` for a
`bool` is refused, at the override's `def` line:

```text
scenario.py:62: error: BaseScenario.is_done() returns bool, and this override returns another type
    reached by <scenario check: MyScenario.is_done()>
    during the realization of _acs_expect_return(r: str, T: bool, ...) [scenario_base.codon:230]
```

A method of a class deriving from none of them, or one its base does not
declare (a helper of the scenario), is not checked this way, and neither is
an abstract one (`@abstractmethod`) or one that only raises
(`raise NotImplementedError()`): it returns no value, which Codon types
`None` whatever its annotation says. The program
also checks what the scenario's own `setup()` and `is_done()` return, after
the module checks, for one the appended calls leave out.

**Then mypy checks that an `Optional` is tested for `None` before it is
used** (#239). Codon unwraps an `Optional[T]` implicitly wherever a `T` is
needed, so `cfg.min_speed_kmh / 3.6` of a `float | None` field compiled, and
raised `TypeError` in Python when the field was `None`. After the compile,
the checker runs mypy (a dependency of the framework) over the scenario's own
modules, following imports silently, checking the bodies of unannotated
functions too, and with no configuration file of the package's (so the
verdict does not depend on how strict the package's own mypy settings are),
and keeps only what it reports of a value that may be `None` used where it
may not be: an attribute of it, an operator, an argument, an assignment, a
return, an index, or an item of a list or dict display (mypy's
`union-attr`, `operator`, `arg-type`, `assignment`, `return-value`, `index`,
`call-overload`, `list-item` and `dict-item`; a set display's item is an
`arg-type` to mypy), each only where
`None` is the offending type (#267): the `None` item of a union an attribute
is missing on (`Item "None" of "Waypoint | None"`), an operand, the value
indexed or an argument that is `None` or a union holding it, or a value whose
type holds `None` given where the type expected does not (`incompatible type
"float | None"; expected "float"`), the key or the value of a dict entry
included (`Dict entry 0 has incompatible type "str": "float | None"; expected
"str": "float"`). An item matters because Codon unwraps it where the model
iterates the sequence (#308): `register_cover(..., buckets=[0.0, 10.0,
cfg.min_speed_kmh])` and `register_cross("c", ["speed", maybe_name])`
compiled, and failed in Python on the `None`; `[0.0, x or 0.0]` passes. A message that only mentions `None`
inside another type or in the type expected is not one: a reader returning
`Optional[str]` given to `register_measure()`, which takes one returning
`Optional[float]`, is a `str` for a `float`, Codon's error to report, and a
`list[float]` for a `list[float | None]` is mypy's invariance, which Codon
accepts. An assignment that only declares is not one either: a `None` default for
a non-Optional annotation (`speed: float = None`), or a value assigned to an
attribute mypy inferred as `None` from its first assignment. Each finding
fails the check at its line, after Codon's errors:

```text
scenario.py:67:28: error: Unsupported operand types for / ("None" and "float"): Codon unwraps an Optional implicitly, so test the value for None before this use, or Python fails here when it is None (mypy: operator)
    Left operand is of type "float | None"
```

**The same pass refuses a framework union used as one of its members**
(#309). Codon holds `TrajectoryVertex.position` (a `TrajectoryPosition`, the
union of the ten poses) as a `Union`, and converts it to whichever member a
use needs at run time (*No attribute of a union*, below), so
`v.position.lanelet_id` of a `MapPose` vertex, or `v.position` passed to a
parameter typed `Lanelet2Pose`, compiled, and raised `AttributeError` in
Python. mypy reports both (`union-attr`, `arg-type`), and the pass keeps
them where the union mypy names is made of the members of one of the unions
the model holds: each module-level `Name = Union[...]` alias of
`typecheck/codon/autoware_carla_scenario/` (`TrajectoryPosition`, `AnyPose`)
and each property it declares to return a `Union[...]`
(`TrajectoryVertex.position`, `CoverItem.event`), read from the model, so
the list follows it (`none_check.framework_unions()`;
`test_typecheck_none_check.py` checks each against the union Python
declares). mypy reports a missing attribute once per member lacking it; the
check reports it once, naming them all, with its own advice:

```text
scenario.py:71:27: error: Items "CarlaWorldPose", "OpenDrivePose", "MapPose", ... of "CarlaWorldPose | Lanelet2Pose | ... | RouteRoadsidePose" have no attribute "lanelet_id": Codon holds this union of the framework's (TrajectoryPosition) and converts it to the member this use needs at run time, so narrow it with isinstance first, or Python fails here on any other member (mypy: union-attr)
```

Narrowed code passes (`if isinstance(p, Lanelet2Pose): p.lanelet_id`), and
so does the union passed on where the union is taken
(`TrajectoryVertex(v.position, advance=v.advance)`, a `TrajectoryPosition`
parameter).

**The CARLA API's `Optional` results are checked too** (#267).
typesafe_carla's CPython package is a prebuilt extension with no `.pyi` and
no `py.typed`, so mypy alone reads `typesafe_carla.carla` as `Any`. The
checker gives mypy stubs of it instead, written from typesafe_carla's own
inventory of its Codon library (`typesafe_carla.pycarla.Inventory`, which the
CPython package is generated from; `typecheck/carla_stubs.py`) and put on
`MYPYPATH`: every public class with its bases, and every method and property
with the return type the library declares. So `Map.get_waypoint()`,
`ActorList.find()`, `World.get_actor()`, `World.try_spawn_actor()`,
`Actor.parent`, `Vehicle.get_traffic_light()`, `Waypoint.get_left_lane()`,
`Sensor.poll()` and the library's other `Optional` results are `Optional` to
mypy, and a use without a test for `None` is refused like a `float | None`
config field:

```text
scenario.py:76:16: error: Item "None" of "Waypoint | None" has no attribute "transform": Codon unwraps an Optional implicitly, so test the value for None before this use, or Python fails here when it is None (mypy: union-attr)
```

`get_waypoint()` is the one exception to the library's declarations: it
returns `None` when no lane of `lane_type` matches, and projected onto the
road (`project_to_road=True`, its default) that is only on a map with no
such lane at all, so the stubs give that call a `Waypoint` (the framework
uses it unguarded too). With `project_to_road=False`, or a flag mypy cannot
read as `True`, it is `Optional`. The stubs type only what the None check needs: parameters are
`*args: Any, **kwargs: Any` (the arguments are Codon's to check), and a
return type mypy cannot be given (a pointer, a type parameter, overloads
returning different types), an enumeration, `carla.command` and any name not
written are `Any`, so they add no error of their own. They are written once
per library, under `~/.cache/autoware_carla_scenario/carla-stubs/` (named by
a digest of the library's Codon sources), or into a temporary directory where
that cannot be written; where typesafe_carla's library cannot be read, mypy
runs with CARLA untyped, as before.

Narrow the value first, as mypy reads it: `if x is not None:`, `assert x is
not None`, an early `return` when it is `None`, or a default (`x or 0.0`).
mypy keeps a cache in `~/.cache/autoware_carla_scenario/mypy` (under
`$XDG_CACHE_HOME` when it is set), so only the first check reads the
framework (about ten seconds); where that directory cannot be written, as in
a container with a read-only home, mypy runs without a cache. A scenario
package installed as a wheel (not editable) is checked too: mypy would take
`site-packages` for its source root, so the check gives it the package
through a link in a directory of its own. When mypy
gives no verdict (it ran out of time or crashed), the check keeps Codon's
verdict and says so in a warning:

```text
warning: the None check did not run (mypy did not finish within 600s), so an Optional used without a test for None is not reported
```

The error is reported at the scenario file's own line. One in a function's
body is at that line, with the call that reached it in the trace:

```text
scenario.py:88:16: error: 'MyConfig' object has no attribute 'timeout_secs'
    during the realization of _helper(self: MyScenario) [<scenario check: MyScenario._helper()>, scenario.py:87]
```

One in what the appended line compiles itself is at the line of the
class-level default, or, for a parameter's default, at the `def` of its
function, and the trace names the call
(`reached by <scenario check: class-level default of MyScenario.RETRIES>`).

A function the check cannot call, because a parameter has no annotation, one
Codon cannot express (`Any`, `object`, `Callable[..., R]`) or one naming a
`TypeVar` (a generic function), is **not refused**: it is left out of the
appended calls with a **warning**, and compiled where `setup()` and
`is_done()` reach it, generically, as before. So is one whose return
annotation names `Any`, such as a custom condition's
`get_details() -> dict[str, Any]` of mixed value types, which Codon cannot
build (the library check lists the framework's own as uncalled):

```text
my_scenario.py:40: warning: MyScenario._scaled: parameter `x` has no annotation; the check does not call it on its own, so it is compiled only where the scenario's constructor, setup() and is_done() reach it
```

**A helper with a `**kwargs` or a `*args` that the scenario calls is
compiled through those calls** (#266). The appended calls give a `*args: T`
one `T` and a `**kwargs` no keyword at all, and Codon types each from the
call: a `Tuple[T]`, an empty `NamedTuple`. So `def opts(**kw: float)` reading
`kw["speed"]` was refused (`'NamedTuple' object has no attribute 'speed'`,
reached by `<scenario check: opts()>`) although `setup()` calls
`opts(speed=2.0)`, and `xs[1]` of a `*xs: float` likewise. Such a function
whose body reads what they hold (`kw["speed"]`, `xs[1]`, `a, b = xs`,
`g(*xs)`, `g(**kw)`) and that a function, a class body or the module's top
level of the scenario's modules calls (by its name: `opts(...)`, `self.scale(...)`,
`helpers.opts(...)`, `Tuner(...)` for its `__init__`; its own recursive call
does not count) is not called again: its callers' calls compile it with what
they pass. One whose body does not read them (`kw.get("speed", 1.0)`,
`sum(xs)`) is called as before: a call is known by its name only, so a
`gains.update(p=1.0)` would otherwise leave a helper `update` to a call of
dict's, and nothing would compile it. One nobody calls is still called, so
an error in it is found,
unless it subscripts its `**kwargs`, which no call without a keyword
compiles: that is left out with the warning above (`MyScenario._opts: reads
`**kw` by key, and nothing in the scenario calls it: the check has no keyword
to call it with`).

A warning does not fail the check (`result.ok` stays true; they are in
`result.warnings`), and `scenario-check` and the runner print them. Annotate
the parameter to have the function checked on its own. Refusing such a
function would have refused a scenario that passed before, for a helper that
compiles wherever it is called. No built-in example, the scenario package
template or a scaffolded package has one.

An ODD written in Python ([ODD](odd.md)) is checked the same way: the program
calls its builder (`odd: OddDefinition = urban_odd()`). Every attribute's
probe is then called with a world, and every condition must come from an
attribute. The ODD's own modules have every function compiled as a
scenario's do, with the same warning for one the check cannot call (compiled
where the builder reaches it). The runner checks the ODD named by the `odd` key under the same
`typecheck` mode, and `scenario-odd check` does it without running anything.

### The model

`autoware_carla_scenario/typecheck/codon/` holds Codon declarations of
everything a scenario imports from `autoware_carla_scenario`: the conditions,
actions, poses and coordinate functions, entities, `BaseScenario`, the shared
config dataclasses, with their static types. The CARLA API is not modelled:
the checker puts typesafe_carla's Codon library next to the model, the way
`typesafe-codon` puts it on `CODON_PATH`, and `import typesafe_carla.carla as
carla` (the import the runtime uses) resolves to it, so a scenario is checked
against the whole typed CARLA API.
typesafe_carla's Python-API compatibility shortcuts (such as calling a
`Vehicle` method on a plain `Actor`) compile, as they do outside strict mode.
Nothing in the model runs; `test_typecheck_model.py` keeps each declaration
it compares in step with the Python definition of the same name (parameter
names, order, defaults, keyword-only parameters, methods, enum members): every
name `__init__.codon` exports, the names the checked modules or a scenario
import from the model beside those (`_BOUNDARY`), and every name of the model
modules that stand for one framework module each (`_MODULE_MODELS`), which
must declare exactly the names compared, beside model-only helpers named
`_acs_*` (the list below). `metrics.codon` and `measures.codon` are two of
those: each declares the whole of its module's
`__all__`, and the test also compares by value their constants (the keys,
`VEHICLE_AHEAD_VERSION`, the distances and speeds the vehicle-ahead measures
read with), each built-in entry of `BUILT_IN_METRICS` (title, unit, text,
aggregation) and of `BUILT_IN_MEASURES` (unit, text, version), the field types
and defaults of `Metric`, `Measure` and `Bumpers` (frozen dataclasses, so
`@tuple`s: assigning to a field fails the check, as it fails the run), and
the parameters of `Metric`'s and `Measure`'s `__new__`. Only a built-in's
`read` differs: Python's is a private function of the module, which no
scenario can import, and the model's a stub (`_acs_unread`), since nothing in
the model is read. `Metric()`, `Measure()` and
`BaseScenario.register_measure()` take any `read` a world can be passed to
that returns a `float`, an `int`, `None`, or an `Optional` of a `float` or an
`int` -- a function returning `float`, `lambda world: None` -- as Python's
do, and refuse any other return type (a `str`, say) at the line that passes
`read`, with "expected a float (or an int)". The three share that check
(`_reader.codon`, outside `measures.codon`, which imports `scenario_base`),
and `Measure` is defined there too, so `BaseScenario.measures` can return a
`Dict[str, Measure]`; `measures.codon` re-exports it.
`Metric()` and `Measure()` store the stub: a Codon `Callable` field takes a
function of its exact type only, and Codon 0.19 crashes on any other. `set_measured_scenario()` takes a scenario of any
`BaseScenario` subclass or `None`, which Python annotates
`Optional[BaseScenario]`: Codon's `Optional` takes no subclass.

An actor is checked by kind for the same reason (#278).
`follow_with_spectator()` and `log_actor_position()` take a getter Python
types `Callable[[], Optional[carla.Actor]]`, and `initial_speeds_mps()` an
`Optional[carla.Actor]`. Each takes a `carla.Actor`, any subclass of it
(`carla.Vehicle`, `carla.Walker`, `carla.Sensor`, ...), an `Optional` of
one, or `None`. A getter returning a `carla.Vehicle` is the natural one to
pass, because typesafe_carla keeps a vehicle's controls on `carla.Vehicle`.
Codon 0.19 refused an `Optional[carla.Vehicle]` for an
`Optional[carla.Actor]`, so these parameters are generic in the model. The
model's `_expect_optional_actor` unwraps the `Optional` and upcasts a class
of a hierarchy to `carla.Actor`, the way a hook result is checked. Anything
else is refused at the call, the framework's `EgoVehicle` (`lambda:
self.ego_entity`) included:

```text
scenario.py:70:9: error: BaseScenario.follow_with_spectator(): the getter must return a carla.Actor (or a subclass of it, such as carla.Vehicle), or None
```

`register_cover()`'s `values=` and `ignore=`, and `OddAttribute(values=)`,
are checked as openodd uses them (#279). `values=` must be `None`, an
iterable (a list, a tuple, an empty one too, a set, a str, a range) or an
enum class, the framework's (`values=TurnDirection`) or the scenario's own.
Nothing is iterated: the enum class is read as a value of the type, not
looped over (see [Iterating an enum](#iterating-an-enum)).
`ignore=` must be `None` or a callable that takes one value of the type
`expression` returns, unwrapped from its `Optional`, since the collector never
passes `None`. The model calls it with such a value, so its body is compiled
too. A wrong `ignore` was the worse gap. The collector calls it inside a
`try`, and counts an exception as "cannot bucket ... counted as no sample",
so a wrong `ignore` emptied the item for the whole run, with one warning
line. What `ignore` returns is tested for truth, which Python takes of any
value, so the result is not checked:

```text
scenario.py:68:9: error: BaseScenario.register_cover(): ignore must take one argument, a value the expression returns
scenario.py:68:9: error: BaseScenario.register_cover(): values must be an iterable of the bucket values (a list, a tuple or an enum class), or None
```

What `test_typecheck_model.py` compares, model by model:

- **Every name `__init__.codon` exports**, with the object of the same name
  in the package.
- **`coordinate.codon`**: beside its exports, `lanelet_length`, `on_lanelet`,
  `project_onto_road`, `project_onto_lanelet`, `get_stop_line_linestrings`,
  `get_stop_line_linestrings_with_following`,
  `find_traffic_lights_for_lanelet2_id`, `get_signal_ids_for_controller`,
  `to_map_frame`, `MapManager`, `map_origin`, `resolve_projector`,
  `_parse_geo_reference`, and pyxodr's `RoadNetwork` and lxml's element
  (below).
- **`conditions.codon`**: beside its exports, `ActionStateCondition`,
  `LaneChangeSettledCondition`, `compare`, `beside_lane_id`, and
  `ConditionStatus`, all but its `details` (a `dict[str, Any]`, which Codon
  cannot express: `_OMITTED_PARAMETERS`), and no other name it defines
  (`find_actor_in_list` it imports from `entity_role.codon`, as Python's
  `conditions.base` does). `ActionStateCondition`'s `state` is an
  `ActionState` or its OpenSCENARIO name, and its `actions` any dict of
  action ids to actions: `actions.codon` imports `conditions.codon`, so an
  action is checked by what `BaseAction` declares.
- **`entity_role.codon`**: beside `EntityRole`, `find_actor_in_list` (a
  `carla.ActorList` or a list of actors, as Python's) and
  `find_actor_in_world`.
- **`kinematics.codon`**: beside its exports, `normalize_angle_deg`, and
  `frame_of` and `FrameMismatchError` as `coordinate.codon` declares them.
- **`coordinate.codon`**: `AnyPose`, the Union alias, by its members' names,
  in order, for a scenario's own annotations (no model function takes it: a
  Union parameter crashes Codon 0.19, so each is generic or overloaded).
- **`constants.codon`**: every constant, by value (`EGO_ROLE_NAME` as
  `EntityRole.ego()`), and no name `constants.py` lacks.
- **`route.codon`**: every name it declares -- `parse_anchor`,
  `scenario_route`, `scenario_route_frame`, `RouteFrame`, `RouteMatch`, and
  `APPROACHES`, `CROSSWALK_LEGS` and `SIDES` by value. `RouteMatch`'s
  constructor is generic: Python's is a frozen dataclass of variadic tuples
  (`tuple[int, ...]`), which Codon cannot express, but the model's takes the
  same parameters, and an overload of none is a `compile_error` (Codon builds
  a class without fields with no arguments otherwise; see *What Codon builds
  by the fields* below), so a call Python refuses fails the check too.
- **`trajectory.codon`**: beside its exports, `ResolvedTrajectory`,
  `TrajectorySample` and the `TrajectoryPosition` Union, by its members'
  names, in order, and no other name.
- **`traffic.codon`**: every name it declares -- `TurnDirection`,
  `LaneChangeDirection`, `TrafficContext` and traffic.registry's functions
  and `TRAFFIC_BACKEND_ENTRY_POINT_GROUP` (by value). `TrafficBackendFactory`
  differs on purpose (`_DIFFERING`): a `Callable` alias in Python of a mapping
  of any values (Hydra's options), which Codon has no type for, and an empty
  class in the model, there only so the checked `traffic` package's re-export
  resolves; a test checks that it stays empty.
- **`utils/traffic_light.codon`**: its three functions, and no other name.
- **`coverage.codon`, `entity.codon`, `odd.codon`**: the names the checked
  modules import (`_BOUNDARY`), `COVERAGE_SCHEMA` and `CoverGroup` by value.
- **The models of one framework module each** (`_MODULE_MODELS`):
  `_bridge_pose`, `action_state`, `carla_install`, `debug_markers`,
  `trajectory_recorder`, `environment`, `autoware_bridge/grpc_server`,
  `autoware_stack/launcher`, `autoware_stack/devcontainer` (and its values),
  `driver/egodriver_client`, `_map_source`, `maps/cache`,
  `maps/z_offset_cache`, `maps/road_network_cache`, `ui/models`, `_ros2`, `metrics`, `measures` and
  `utils/traffic_light`, each declaring exactly the names compared;
  `ros2/cyclone` where cyclonedds is installed.
- **Every public name of a modelled module**: each name a module the model
  has exports in Python (its `__all__`, or the public names it defines when
  it has none) is declared by the model or listed, with the reason, in
  `typecheck/not_modelled.py` (`NOT_MODELLED`), and a name listed there must
  be neither declared nor gone from Python. Each declared one is compared as
  above, re-exports included (`kinematics.frame_of`). `OddSample` and
  `OddCase` differ on purpose (`_DIFFERING`): an opaque, unbuildable stand-in
  for openodd's sample, and the `@tuple` of the pair Python's alias names.
- **Every public member of a compared class** (#265): the comparisons above
  check that what a model class declares exists in Python; this checks the
  converse. For every class they compare, enums included, each public
  method, property, class attribute, dataclass field and instance attribute
  the package's Python, or openodd's, defines on the class or a class it
  derives from
  (`VehicleEntity.set_desired_speed()` is `BackendDriven`'s) is declared by
  the model class or one of its model bases, or listed, with the reason, in
  `NOT_MODELLED_MEMBERS` (`typecheck/not_modelled.py`), under the class or
  the Python base that defines it (`BaseCondition.get_details` for every
  condition). openodd's classes, which the package re-exports
  (`OddAttribute`, `OddModule`, `OddDefinition`, `OddCondition`, `CoverItem`,
  `CrossItem`, `CoverageReport`), are compared as the package's are (#298),
  each listed under the name the package gives it (openodd's `Condition` is
  `OddCondition`). A listed member must still be public in Python and absent
  from the model. A dataclass's public field is a member too, beside the
  constructor's parameter it is compared as (#298): the model declares it as
  a field or, where a field would change the constructor Codon generates
  (*What Codon builds by the fields*, below), as a read-only property
  (`RouteMatch.lanelet_ids`, a `List[int]` for Python's `tuple[int, ...]`;
  `TrajectoryVertex.position`, the union of the ten poses, which narrows
  with `isinstance` and passes on to a new `TrajectoryVertex`, and which
  the None check's pass refuses un-narrowed where one pose is expected,
  #309; `CoverItem`'s
  fields). A frozen dataclass's field is read-only in Python too; writing a
  `CoverItem`'s is refused, which Python allows, and a scenario, which
  registers its items with `register_cover()`, has none to write.
  An attribute a method sets on the instance (`self.<name> = ...` or
  `setattr(self, "<name>", ...)`, read from the class's source,
  `__init__`'s and the others', openodd's included; one set under a name
  computed at run time is not found) is a member too (#307), as Python
  reads it like any other: the model declares it as a read-only property,
  since a field would change the constructor Codon generates
  (`TimeoutCondition.timeout_seconds`, `OddDefinition.attributes`,
  `OddAttribute.target` and `item`, `TrafficSinkAction.removed`,
  `RouteFrame`'s `match`, `map`, `graph` and `lanelets`). Writing one is
  refused, which Python allows (`'TimeoutCondition' object has no attribute
  'timeout_seconds'`), except in the `__init__` of a scenario's own
  subclass, where the assignment declares the subclass's own attribute. An enum is
  compared beyond its members (#307): a model enum is a `@tuple` of its
  value, so Codon gives it no `Enum`, and it declares Enum's `name` as a
  property, a chain of `if self == Rule.MEMBER: return "MEMBER"` that
  `test_typecheck_model.py` compares with each member's Python name, in
  order (it also compares each member's value), and its own methods and
  properties (`ComparisonRule.text`, `CollisionTargetType.matches()`,
  `LaneChangeDirection.to_carla_bool()`). A scenario's own enum gets its
  `name` from the check's rewrite (below). The scenario check adds the
  reason to Codon's error:

  ```text
  scenario.py:68:19: error: 'CaseScenario' object has no attribute 'traffic_backend': BaseScenario.traffic_backend is not part of the Codon model, on purpose: takes or returns the run's TrafficBackend, ...
  ```

  Left out are what returns or takes a value Codon has no type for (a
  `dict[str, Any]`: a condition's `get_details()` and `to_summary_dict()`,
  `to_dict()`, `describe()`, `RouteMatch.to_config()` and `from_config()`),
  the run's traffic backend (`set_traffic_backend()`, `traffic_backend`),
  `RouteMatch.junction()`, `junctions` and `segments` (a
  `RouteSegmentMatch`), a field of `Any` (`ConditionStatus.details`,
  `RouteLanelet.values`, `OddModule.metadata`, `SweepConfig`'s Hydra
  mappings, `CoverItem`'s `expression`, `ignore` and `values`,
  `OddAttribute.probe`), a class held as a value (`BaseScenario.ego_type`:
  a scenario reads `ego_requires_goal` and calls `create_ego()`),
  `RouteFrame.features` (`route.geometry`'s queries, which the library check
  compiles from their source), openodd's
  evaluation, sampling and archive of an ODD (`OddDefinition.evaluate()`,
  `sample()`, `archive()`, ..., `OddCondition.evaluate()`) and its report's
  rows (`CoverageReport.entries`, ...), and the members of the classes the
  model declares for the library check alone (`CarlaDriverEntity`, the
  egodriver and bridge servers, the ROS 2 transport and its config, the map
  cache's `MapSource`, `RunEnvironment`). An ODD's members compile as they
  run:

  ```python
  roads = OddModule("roads", include_and=[location.is_in(["urban"])], labels=["fair"])
  roads.as_situation(cover_by="meters", target=100.0)
  text = roads.conditions()[0].describe() + roads.labels[0] + location.cover_by
  for attribute in odd.attributes:
      text += attribute.name + str(attribute.target)
  ```

  So do an instance attribute and an enum's name and methods:

  ```python
  timeout = TimeoutCondition(cfg.timeout_seconds, label="timeout")
  logger.info("limit %f, %s", timeout.timeout_seconds, ComparisonRule.LESS_THAN.name)
  logger.info("%s %s", ComparisonRule.LESS_THAN.text, LaneChangeDirection.LEFT.to_carla_bool())
  ```
- **Every model file**: each `.codon` of the package's model is in one of
  the tables above or exempt with the reason (`_UNCOMPARED_MODEL_FILES`: the
  model's helpers, the modules `coordinate.codon` and `entity.codon`
  re-export, and the `__init__` of a checked subpackage, which models
  nothing), so a model file no test compares fails the suite.
- **Third-party models**: carla_driver_interface's geometry, python-dotenv's
  `load_dotenv`, pyxodr's `RoadNetwork` and lxml's element.

A constant whose model value is a literal (a `str`, `int`, `float` or a tuple
of them) is compared by its value and type (a tuple's items' too: `1` is no
`1.0` or `True`) wherever it is compared, and a `Literal` alias
(`ui.models.Status`) as the type of its values. Any other constant fails the
comparison unless a test of its own compares it (`EGO_ROLE_NAME`,
`BUILT_IN_METRICS`, `BUILT_IN_MEASURES`).

**What Codon builds by the fields.** Codon 0.19 gives every class a
constructor of its fields, its bases' first, by position or by name, beside
the `__init__` it declares: a class without fields is built with no arguments
(`VehicleEntity()`, #225), a condition or an action with the label it
inherits alone (`TimeoutCondition("10")`, `TurnAction("npc1")`, #236), and
any class by fields its constructor does not take (`DebugMarkers([])`,
`CoverItem("speed")`, `RunProgress(1, 2, "case", "running")`, #244), where
Python refuses each. A class that derives from one of Codon's exceptions
(`Static[Exception]`) also takes its bases' constructors, and its fields
start with `BaseException`'s eight: `FrameMismatchError("x")` passed by
`Exception(typename, msg="")` (#244). A class that derives from a model class
plainly does not take its base's `__init__`.

Codon tries the `__init__` overloads a class declares from the last one up,
and those it generates or inherits only after them, so the model declares,
ahead of the constructor, an `__init__` of as many arguments whose body is a
`compile_error`:

```python
class TimeoutCondition(BaseCondition):
    def __init__(self, label):
        compile_error("TimeoutCondition(): no constructor takes these arguments "
                      "(required: keyword label)")

    def __init__(self, timeout_seconds: float = 60.0, *_kw, label = _Required()):
        ...
```

The constructor, declared last, still wins every call it takes
(`TimeoutCondition(label="t")`, `TimeoutCondition(10.0, label="t")`); the
refusing overload only takes what would reach Codon's own. The one of as many
arguments as the class has fields takes the fields' names, so a call by them
(`CoverItem(name="speed")`) is refused too.

`test_typecheck_model.py` compares the last `__init__` with Python's and
finds the overloads needed. For every compared class that declares an
`__init__`, it counts each number of positional arguments the constructor
Codon generates (the fields, `_codon_fields`) or an exception base's
(`_inherited_inits`, Codon's own read from `_CODON_EXCEPTIONS`, which a test
compares with Codon's standard library) would take, and asks Python whether
its constructor binds as many placeholders (`inspect.signature().bind`).
Where Python refuses and the model's constructor does not take them either,
the model declares the refusing overload of that many; where the model's
constructor takes them through `*_kw`, its `_keyword_only(_kw, ...)` refuses
them (`OddModule`, `OddAttribute`, `BaseScenario`). The classes it finds:

| Arguments | Classes |
|---|---|
| none | `RouteMatch`, `RouteFrame`, `VehicleEntity`, `PedestrianEntity`, `CoverageCollector`, `CoverageReport`, `EgoDriverGrpcClient`, `Ros2EgoTransport` (and `CycloneRos2Transport`, compared apart) |
| the label | the conditions and actions whose constructor Python refuses one argument (`SpeedCondition`, `TurnAction`) |
| the fields | `DebugMarkers`, `CoverItem`, `OddCondition` (one), `RunProgress` (four) |
| an exception's | `FrameMismatchError` (one, two, ten), `DevContainerError`, `MapCacheError` (none, two, eight) |

`RouteCoverage` declares Python's fields in Python's order (#298), so the
constructor Codon generates of them takes no call Python's refuses, and the
one it declares refuses each.

Every condition and action declares the overload of one, its label, even
where Python takes one argument by position: a call the constructor refuses
by its type (`TimeoutCondition("10")`) would still reach the one Codon
generates.

Those constructors take their parameters by keyword too, and a call by
keyword Python refuses passes where no guard takes those names: an
exception's base parameters (`DevContainerError(typename="x")`,
`FrameMismatchError("x", message="y")`), or a class's field names
(`OddAttribute(name="speed", unit="m", text="t")`, `EntityRole(_value="x")`).
So the test also enumerates every call, by position and by keyword, the
constructor Codon generates or an exception base's would bind, and requires,
for each one Python and the model's constructor refuse, a refusing overload
that binds it too: an `__init__` of the fields' names (`OddAttribute`,
`OddDefinition`, `EntityRole`, `TrajectoryVertex`, `EgoConfig`,
`BaseScenario`, `RunEnvironment`), and of `typename` and
`typename, message` (`DevContainerError`, `MapCacheError`;
`FrameMismatchError` takes one positional `typename` already). A class
whose `__init__` takes its fields by position and by name needs none: the
constructor Codon generates is its own.

A scenario that imports a module with no model (lanelet2's `matching`, say,
or a framework module outside the public API) is refused with a message
naming the module. For a framework module, the message says where the model
declares the names imported as Python exports them, and only where Python's
module has the same object, so the import it suggests runs too; a name the
model leaves out on purpose is refused with its reason, whichever module it
is imported from (#237):

```text
scenario.py:3: error: autoware_carla_scenario.actions.base is not part of the Codon model; import ActionState from autoware_carla_scenario.action_state instead
scenario.py:4: error: autoware_carla_scenario.route.parse_route_search is not part of the Codon model, on purpose: the route search, parsed from a document's YAML ...
```

Third-party packages the framework reads are modelled as
far as it reads them (lanelet2, pyxodr, carla_driver_interface, python-dotenv;
see *Making a module compile*), and a scenario may use what
those models declare.

## Writing a scenario that type-checks

A scenario is ordinary typed Python. Codon is stricter than Python in a few
places, and the checker smooths over most of them (it rewrites `X | None` into
`Optional[X]`, drops `@dataclass` (and appends the `==`, ordering and hash it
generates) and resolves `field(...)` defaults, accepts
keyword-only parameters, required ones after a default included, gives a list of different conditions the type
`list[BaseCondition]`, checks a `tuple[X, ...]` return annotation as a
`List[X]`, drops a parameter or return annotation naming a
class the module defines further down, which Codon cannot name yet, gives a
subclass that keeps its base's `__init__` that `__init__`, and types the
`return None` of a function returning `Optional[X]`; see below). What a
scenario has to do itself:

- **Import its own modules from its own package.** A module of the
  scenario's package imported as a module works (#310): Codon 0.19 resolves
  `from X import name` only as a name `X/__init__` defines, and an absolute
  `import a.b` only from the importing file's directory, so the checker
  rewrites, on the same line, `from . import helpers`, `from pkg import
  helpers` and `import pkg.helpers as helpers` in `pkg/scenario.py` into
  `import helpers as helpers`, and `from .sub import mod` into `import
  sub.mod as mod`. A module outside the importing module's package (`from
  .. import helpers` in `pkg/sub/scenario.py`, or `import pkg as p`) has no
  such form, and neither has one imported without a name (`import
  pkg.helpers`, which binds `pkg`): each is refused at its line, naming the
  form that works, which is to import the names (`from ..helpers import
  speed_kph`) or to name the module (`import pkg.helpers as helpers`):

  ```text
  scenario.py:3: error: `from .. import helpers` imports the module pkg.helpers as a module, which Codon 0.19 resolves only inside the importing module's own package (pkg.sub): import the names instead (`from ..helpers import name`)
  ```

- **Import a name once.** Python accepts `from autoware_carla_scenario import
  ComparisonRule` in the import header and again further down, and so does
  the check (#354): Codon 0.19 leaves a module-level name imported twice
  undefined in every module but the one it compiles first, so a use of it
  failed the whole function with a bare "cannot typecheck". The checker drops
  a module-level import that repeats an identical earlier one (the same
  module and name, under the same alias), on its line: the repeat becomes
  `pass`, or loses just the repeated names. An import in a function or an
  `if` is left as written. A name imported from two different places
  (`from a import X`, then `from b import X`) means the later one in Python:
  the checker drops the earlier import when nothing reads the name in
  between, and refuses the later import, naming the first, when something
  does:

  ```text
  scenario.py:12: error: `ComparisonRule` is imported again from somewhere else (first on line 5) after it was used, which Codon 0.19 cannot compile: import each name once, under its own name (`import ... as other` to tell two apart)
  ```

- **Declare the attributes it assigns on `self`**, at class level, with their
  type:

  ```python
  class MyScenario(BaseScenario):
      _config: MyScenarioConfig

      def __init__(self, ego_config, spawn_pose, config=None, ground_projection=None):
          super().__init__(ego_config, spawn_pose=spawn_pose, ground_projection=ground_projection)
          self._config = config or MyScenarioConfig()
  ```

  A bare annotation is ignored by Python, so this changes nothing when the
  scenario runs; Codon needs it to type an attribute of a subclass. The check
  names each undeclared attribute and suggests its type.

  **Declare them in every class of the scenario's modules**, helper classes
  included (#218). The check refuses an undeclared attribute of a class with
  a base, a subclass, or at least one class-level annotation: once a class
  declares one attribute, Codon gives it only those it declares. A plain
  class that declares none has its attributes inferred from its `__init__`,
  which Codon cannot always do: not for `self._samples = []` (a list of
  nothing yet), and not in a method the scenario check calls on its own
  ([*What is checked*](#what-is-checked)), on a value its `__init__` did not
  build (`'int' does not match expected type 'NoneType'`, during the
  realization of `get(self: Counter[NoneType])`). Codon's error is then at
  a line of the helper, or names the attribute holding the helper, not the
  helper (`type of attribute '_tracker' of object 'GapCondition' cannot be
  inferred`), so the check adds, for that helper, each attribute it assigns
  without declaring it, at its line, with the type to declare:

  ```text
  scenario.py:34: error: GapTracker._samples is assigned without a type: declare it in the class body, e.g. `_samples: list[...]` ...
  ```
- **Use only the public API** from `autoware_carla_scenario` and its listed
  subpackages, and the `BaseScenario` attributes meant for subclasses
  (`ego_config`, `ego_entity`, `world`, `_ground_projection`, ...).
  An import from a modelled framework module must also work in Python
  (#214): the model declares a few names the Python module does not have
  (`coordinate.RoadNetwork`, `entity.EgoConfig`, `actions.ActionState`:
  import it from `autoware_carla_scenario.action_state`, as Python does) and
  model-only helper modules (`autoware_carla_scenario._tc`, `_lists`), and
  an import of one inside a function would compile and raise `ImportError`
  when the function first runs. So the check imports each modelled module
  a scenario imports from, in Python, and refuses each name it lacks, as an
  attribute and as a submodule, at the import's line:

  ```text
  scenario.py:118: error: autoware_carla_scenario.coordinate has no RoadNetwork in Python, whatever its Codon model declares: `from autoware_carla_scenario.coordinate import RoadNetwork` raises ImportError when the scenario runs
  ```

  An import under `if TYPE_CHECKING:` is exempt: Python never runs it.
- **Define every abstract method of what you build** (#215). Codon has no
  `@abstractmethod`, and the model gives `BaseScenario.is_done()`,
  `BaseCondition.check()` and `BaseAction.execute()` bodies (its built-in
  conditions and actions inherit them), so a scenario whose `is_done` was
  renamed, or a condition that defines only `_check`, compiles, and Python
  refuses to build it (`TypeError: Can't instantiate abstract class`). The
  check asks Python (`inspect.isabstract`) about each class of the
  scenario's modules that is built, the scenario class (the runner builds
  it) or a class the modules call by name (`GapCondition(...)`), and
  refuses one still abstract at its `class` line:

  ```text
  scenario.py:46: error: GapCondition is built (at scenario.py:99) but leaves abstract check undefined: define it in GapCondition, or Python refuses to build it (TypeError: Can't instantiate abstract class)
  ```

  A class only derived from, such as an abstract base of the scenario's own
  conditions with an `@abstractmethod` of its own, is never called and may
  stay abstract.
- **Pass numbers of the declared type.** An `int` is accepted for a `float`,
  but not inside a container: a `list[float]` parameter wants `[1.0, 2.0]`.
  YAML values are converted for you.
- **An empty `[]` or `()` is any sequence** (#264). Codon gives an empty
  display no element type (`[]` is a `List[NoneType]`, `()` a tuple of no
  items), so the model takes one wherever Python takes a `Sequence[X]`
  (`RoutingAction(EGO_ROLE_NAME, goal)`, whose `waypoints` default to `()`,
  `waypoints=[]`, `self.waypoint_poses = []`, `derive_goal_from_route(())`,
  `AndCondition([], ...)`, an ODD's `all_of([])`) through `_acs_sequence`
  (`_tc.codon`), and checks the elements of any other: `waypoints=[460, 265]`
  and `waypoints=[None]` are refused, in `EgoVehicle.route_to()` too.
- **A sequence of names is a list or a tuple, not a `str`** (#299). Python's
  typing takes a `str` as a `Sequence[str]`, and the framework then reads
  each of its characters as a name (`roots="roads"` fails at run time with
  "no root module ['r', 'o', 'a', 'd', 's']"). Here the check is stricter
  than mypy, on purpose: `_expect_str_sequence` (`_tc.codon`) refuses a `str`
  (or an `Optional[str]`) where a sequence of names is taken, the items of
  `BaseScenario.register_cross()`, `OddModule`'s `labels` and
  `OddDefinition`'s `roots`, and takes a list or a tuple of `str`, an empty
  one too:

  ```text
  scenario.py:41: error: BaseScenario.register_cross(): items: expected a sequence of names (a list or a tuple of str), not a str: wrap it in a list
  ```

  `OddAttribute.is_in("urban")` still passes: it takes an `Iterable[Any]` of
  values, and CPython takes a `str`'s characters as values.
- **Call a manoeuvre your own entity class overrides with every argument, by
  position.** `VehicleEntity` and `EgoVehicle` declare the manoeuvres they
  take from `traffic.driven.BackendDriven` (`set_desired_speed()`,
  `change_lane()`, `lane_change_finished()`, `turn_at_junction()`,
  `release_from_traffic()`, ...), and an `EgoVehicle` subclass's override of
  one is checked as a hook. Codon 0.19 calls a method a subclass overrides
  through a dispatch thunk, which takes no keyword argument, fills in no
  default (`set_client(client)` needs its `tm_port` then), and cannot call
  `turn_at_junction(world, direction, **kwargs)` at all: once a class of the
  scenario overrides it, a call of it does not compile ("cannot
  typecheck"), on the subclass as on its base; turn with a `TurnAction`
  instead. A class that only defines the override, and never calls it,
  compiles. (Declaring Python's `**kwargs` as `TurnAction`'s three keywords
  would not help: the thunk fills in none of their defaults, so even
  `turn_at_junction(world, direction)` would fail, and a backend's own
  keywords would be refused.)
- **A dict of different actions (or conditions) is a dict of their base.**
  Codon types a dict display by its first item, so the check rewrites one
  whose values are not evidently of one type, as it does a list display:
  `{"turn": turn, "lane_change": lc}` (a `TurnAction` and a
  `LaneChangeAction`) becomes `{"turn": _acs_base(turn), "lane_change":
  _acs_base(lc)}`, which makes each action a `BaseAction` (each condition a
  `BaseCondition`): a `dict[str, BaseAction]` as in Python, which passes as
  `ActionStateCondition`'s `actions`. Assigned to an annotated variable,
  `actions: dict[str, BaseAction] = {...}`, it becomes a dict of the
  annotated type (`_acs_dict_into`, when the key and value types are plain
  names). A value of another type is still refused at the display's line:

  ```text
  scenario.py:73: error: 'int' does not match expected type 'BaseAction'
  ```

  Any other value is left to Codon's display, which types it as before
  (`{"a": 1, "b": speed}` is a `dict[str, float]`, a value with `None` is
  optional). Values evidently of one type (all literals, all calls of the
  same name, all members of one enum) are left as written. Building the dict
  one item at a time into a `dict[str, BaseAction]` stays valid.
- **Convert nested mappings in the config.** The runner passes a YAML mapping
  to the config class as a `dict`, so a field annotated with a dataclass
  (`goal: Goal`) holds one at run time unless the config's `__post_init__`
  turns it into a `Goal`, as the built-in configs do for their NPC lists.
  The check builds the config the same way and refuses an unconverted
  mapping at its `scenario.<key>`.

- **Annotate every parameter of every function and method**, helpers and
  hook overrides included. Every function of the scenario's modules is
  compiled, whether `setup()` reaches it or not, with a value of each
  parameter's annotated type ([*What is checked*](#what-is-checked)). A
  function with a parameter that has no annotation, one Codon cannot
  express or one naming a `TypeVar`, is only compiled where `setup()` and
  `is_done()` call it, and the check warns about it, at the parameter's
  line; so is one returning `Any`, with the warning at its `def`.
- **Give each class-level default a value of its annotated type**
  (`RETRIES: int = 3`): it is compiled against the annotation.
- **Return what the framework reads from a hook you override**: a `bool`
  from `is_done()`, a `ScenarioResult` or `None` from a condition's
  `check()`, a `dict[int, float]` from `initial_speeds_mps()`, `None` from
  `setup()` and an action's `execute()` ([*What is checked*](#what-is-checked)).
- **Test an `Optional` for `None` before using it**: a `float | None` config
  field, `goal_pose`, `find_actor_by_role_name(...)`, and CARLA's `Optional`
  results (`get_waypoint(...)`, `get_actors().find(...)`,
  `world.get_actor(...)`, `actor.parent`).
- **Read `**kwargs` by name with `kw.get()`, and test `*args`, like any
  tuple, with `len()`** (#266). Codon makes `**kwargs` a `NamedTuple` of the
  keywords passed, not a dict, and `*args` a `Tuple`, and gives no tuple a
  truth value (#278). Codon refuses this valid Python, and the message says
  what to write instead:

  | Python | Codon's error | Write instead |
  | --- | --- | --- |
  | `for k, v in kw.items():` (`.keys()`, `.values()`, `for k in kw`, `len(kw)`) | `'NamedTuple[a=float]' object has no attribute 'items'` | `kw.get("a", default)` for each name, or keyword parameters (`def f(*, a: float = 1.0)`) |
  | `if args:` / `x if args else y`, or `if window:` of any tuple | `'Tuple[float,float]' object has no attribute '__bool__'` | `if len(args) > 0:` (`if len(window) > 0:`) |

  `kw["speed"]` and `"speed" in kw` compile for a keyword the call passes;
  `kw.get("speed", 1.0)` compiles whether it does or not. A helper taking
  `**kwargs` or `*args` that the scenario calls and that reads them by key
  or position is compiled through its callers' calls, so it is checked with
  the keywords and values they pass
  ([*What is checked*](#what-is-checked)).
- **Give your own helper an `Optional` of the class it gets, not of its
  base.** Codon 0.19 does not convert an `Optional` of a subclass to an
  `Optional` of its base (#278). A helper `def gap(actor: carla.Actor | None)`
  refuses an `Optional[carla.Vehicle]`, and a plain `carla.Vehicle` too, which
  Codon wraps in an `Optional` first (`'Optional[Vehicle]' does not match
  expected type 'Optional[Actor]'`). Python accepts both. A parameter of
  `carla.Actor` alone takes a `carla.Vehicle`, and the framework's own
  `Optional[carla.Actor]` parameters take either
  ([*What is checked*](#what-is-checked)). Annotate the helper with the
  subclass (`carla.Vehicle | None`), or test for `None` before the call and
  pass the unwrapped value to a `carla.Actor` parameter.
- **Build an object of a class you hold in an `Optional`, when the class has
  subclasses and nothing else builds one.** Codon 0.19 checks
  `x: X | None = None` (or a parameter defaulting to `None`) of such a
  class, then generates code that LLVM refuses ("LLVM: error: value doesn't
  match function result type 'ptr'"), when the program builds no `X` nor
  any subclass of it, or defines a subclass after Codon compiled `X` (#263).
  The model builds its own such classes once, after their subclasses:
  `BaseCondition`, `BaseAction` and lanelet2's `ConstLanelet`,
  `ConstLineString2d`, `ConstLineString3d`, `ConstPoint3d` and
  `RegulatoryElement` (the others are built by the scenario check anyway).
  So a scenario that builds actions but no condition compiles, though each
  action's `condition = None` is an `Optional[BaseCondition]`, whether it
  registers the actions or keeps them in a `list` or `dict` of `BaseAction`.
  A class of your own, or your subclass of a framework class that nothing
  builds (`class MyResult(ScenarioResult)`), can still meet it; the check
  then names the class and what to do, at the `class` line of it or of its
  subclass, and keeps no verdict, as for any crash:

  ```text
  scenario.py:73: error: Codon 0.19 generated code it cannot compile for autoware_carla_scenario.conditions.ScenarioResult, a class with subclasses: a bug of Codon's rather than an error of the program (#263). It does so for an Optional of such a class (`x: ScenarioResult | None = None`, a parameter that defaults to None) when the program builds no object of the class or of a subclass, or defines a subclass after Codon compiled the class. Build a MyResult where setup() reaches it, or hold the value without Optional
      LLVM: error: '%obj' defined with type 'ptr' but expected '{ ptr, ptr }'
  ```

Annotations Codon cannot express (`Union` of two types, `Any`, `Callable`,
`Sequence`, `type[...]`) are dropped from parameters, which become generic:
Codon then checks each call with the arguments it is given. The appended
calls give a union parameter (`Lanelet2Pose | OpenDrivePose`) each of its
members in turn, and a `Callable[[A], R]` of types Codon can express a value
of that signature; `Any`, `object` and `Callable[..., R]` give them nothing to
call with, hence the warning.

The checker also rewrites what Codon spells differently: an `Enum` class (its
members become instances with a `name` and a typed `value`; a lookup by value,
`ActionState("completeState")`, takes a value of that type and is typed but
not resolved to the member; looking a member up by name and iterating over the
class work for a scenario's own enum, see
[Iterating an enum](#iterating-an-enum)), a class deriving from an exception, a `@classmethod`
(a static method of its class: `cls` names the class it is defined in), and an
annotation naming one of CARLA's integer enumerations
(`carla.TrafficLightState`), which typesafe_carla's Codon library has as a
value whose members are `int`, not as a type: it becomes `int`.

Two things Python does that Codon 0.19 does not are filled in for you:

- **A subclass that inherits `__init__`.** Codon 0.19 inherits no
  `__init__`: it gives a class without one a constructor taking each of its
  fields, which refused a scenario deriving from another scenario and keeping
  its constructor ("'config' is an invalid keyword argument for
  `__init__()`", #201). The check appends one to such a class, after the
  module, so no line moves:

  ```python
  @extend
  class MyDerivedScenario:
      def __init__(self, *args, **kwargs):
          super().__init__(*args, **kwargs)
  ```

  It is added only where a base really defines an `__init__` (a class of the
  scenario's modules, its own or one it inherits, or one of the model's, such
  as `BaseScenario`'s), and never to a dataclass, an `Enum` or an exception,
  which are built as before. When the `__init__` is a later base's
  (`class MyDerivedScenario(MyMixin, MyScenario)`, the mixin having none),
  it is called through that base, `MyScenario.__init__(self, *args,
  **kwargs)`, as Python does: Codon's `super()` is the first base's. The
  base's `__init__` still checks every argument: a wrong one is reported at
  the `class` line of the subclass, with the call that passed it in the
  trace:

  ```text
  scenario.py:52:9: error: no function 'MyScenario.__init__' with arguments (Super[MyScenario], EgoConfig, Lanelet2Pose, str)
      during the realization of __init__(self: MyDerivedScenario, ...) [scenario.py:58]
  ```

  A literal `None` passed through it is refused, though Python takes it
  (`MyDerivedScenario(ego, pose, None)`, or `config=None`): forwarded
  through `*args` or `**kwargs`, it reaches the base's `__init__` as an
  `Optional[NoneType]`, which no `Optional[X]` parameter takes. Leave the
  argument to its default, pass a variable declared `Optional[X]`, or give
  the subclass an `__init__` of its own.

- **`return None` from a function returning `Optional[X]`.** A bare `None`
  is Codon's `NoneType`, so a function whose only return is `return None`
  could not be typed ("cannot typecheck 'None'"), and, when nothing else in
  the program used `X` (`carla.Actor` in a scenario), `Optional[X]` failed
  in code generation ("value doesn't match function result type 'ptr'",
  #202). The check makes each `return None` (and bare `return`) of the
  function itself `return Optional[X]()`, the `None` of its type (a nested
  function is rewritten for its own annotation), and the appended calls
  realize `X` (`_acs_value(X)`) before they call it. So

  ```python
  def _find_npc(self) -> Optional[carla.Actor]:
      return None
  ```

  compiles as written, and so does a local `found: Optional[carla.Actor] =
  None` returned at the end.

- **Your own dataclasses, enums and named tuples** keep what Python gives
  them (#365), with these limits:
  - A `@dataclass` compares, orders and hashes as Python's does: the check
    appends `==` and `!=` (`eq=True`, the default), `<`, `<=`, `>` and `>=`
    (`order=True`) and `hash()` (`frozen=True`, or `unsafe_hash=True`) after
    the module, comparing the fields in order (a dataclass base's first;
    `field(compare=False)`, `ClassVar` and `InitVar` are left out). A method
    the class defines itself is kept. This holds for a dataclass whose bases
    are dataclasses of the same module (or none): one deriving from a class
    the check cannot see the fields of, such as a framework dataclass, gets
    none, and `==` on it is refused. `eq=False` (identity comparison) is not
    modelled either: write `a is b`. Python refuses `<` without `order=True`
    and `hash()` of a non-frozen dataclass, and so does the check, with a
    hint naming the flag.
  - An `IntEnum` (also `IntFlag`, `class M(int, Enum)`) member compares with
    its value (`M.A == 1`), orders against a member or an `int`
    (`M.A < M.B`) and converts with `int()` and `float()`. A `StrEnum` or
    `class M(str, Enum)` member compares with its string (`M.A == "a"`) and
    does nothing more: ordering and `int()` are refused, as in Python for a
    plain `Enum`, a `Flag` and a `(str, Enum)`. A plain `Enum` member equals
    only members (`M.A == 1` is `False`, as in Python). An enum member is
    **not** an `int` or a `str` otherwise: passing it to an `int` / `str`
    parameter or doing arithmetic on it (`M.A + 1`) is refused with a hint.
    Write `M.A.value`.
  - A `class Pt(typing.NamedTuple)` is checked as a Codon `@tuple` class: its
    fields, defaults and methods, `==`, ordering, `hash()`, iteration and
    `len()` work. Codon indexes and unpacks it (`p[0]`, `x, y = p`) only when
    all its fields have one type: otherwise read `p.x`, `p.y`. It has no
    `_replace()`, `_asdict()`, `_fields` or `_make()`: build a new `Pt(...)`.
    Only the class syntax is covered (not `NamedTuple("Pt", [...])` or
    `collections.namedtuple`).

### Python the check rewrites for Codon (#376)

Each of these is valid Python that Codon 0.19 refuses or crashes on. The check
rewrites the checked copy of your module (on the same lines; what runs is your
code as written), so you write the Python you would anyway. A form the rewrite
cannot express is refused at its line, with what to write instead.

| You write | The check compiles | Refused when |
| --- | --- | --- |
| `class MyError(Exception): ...` | the same with `pass` for `...` | |
| `{**a, "k": v, **b}` | `(a \| {"k": v} \| b)`; `{**a}` is `dict(a)` | the values of the parts have other types (`Dict[str, int]` and `Dict[str, str]`) |
| `"{:>5}\|{a}".format(1, a=x)` on a string literal | the f-string `f"{1:>5}\|{x}"` (a complex argument is bound once by a lambda) | `*args` / `**kwargs`, a field inside a format spec (`{:>{}}`), a missing argument, mixed auto and manual numbering, a template that is not a literal (write an f-string) |
| `zip(a, b, strict=True)` | `zip(a, b)`: the check types the loop, it does not run it | |
| `sorted(xs)`, `xs.sort()`, `min` / `max` of a class that defines `__lt__` only | the class gets `__gt__` (`other < self`) and `__ge__` (`not self < other`), which Codon's sort calls | the class defines no ordering (use `@dataclass(order=True)` or `key=`) |
| `@functools.total_ordering` | the decorator and its import are dropped, and the class gets the comparisons it would add from its `__lt__`, `__le__`, `__gt__` or `__ge__` (and `==`, or identity without `__eq__`) | the class is not at module level, or defines none of the four |
| `lambda x, k=k: x + k` (a default that is not a literal) | `(lambda k: lambda x: x + k)(k)`: the default is bound by a call, once, as in Python | |
| `def __exit__(self, exc_type, exc, tb)` (annotated `type[BaseException] \| None`, `BaseException \| None`, `object`, ... or not) | each parameter without a default gets `=None` and loses its annotation: Codon calls `__exit__()` with no arguments, even when the block raised (#435) | |
| `fs: list[Callable[[int], int]] = [inc, lambda x: x * 2]`, `table: dict[str, Callable[[int], int]] = {"a": inc, "b": dbl}`, or a `return` of such a display from a function annotated so (a dispatch table) | each function, module function (`mod.f`) or lambda element is wrapped in Codon's `Callable[Tuple[int], int](...)`, so they share one type (#436) | the element is a bound method, a lambda using a local variable of its function, or a call (not wrapped, then Codon's own refusal and a hint); the table is an attribute of a class, is unannotated, or is built another way than a display |
| a class constant read in its own class body (#433): `DOUBLE = LIMIT * 2`, `x: int = SIZE` (a dataclass default), `def f(self, n=LIMIT)`, `@register(LIMIT)` | `Class.LIMIT` (`Outer.Inner.LIMIT` in a nested class) | the constant is annotated and not a single-line literal (`LIMIT: int = 2 * 3`), it is assigned again later in the body, the class is defined in a function, the read is a `NamedTuple` field default, or it is an Enum member with a non-literal value |

Notes:

- `__exit__` is `None` in all its parameters in the check, so
  `if exc_type is not None:` is not compiled and anything in it is not checked
  (Python runs it as you wrote it). `from types import TracebackType` is
  refused (`types` has no model): annotate `tb` as `object`. `__enter__`
  returning `self` (`-> "Ctx"`, or no annotation) compiles as it is.
  `__exit__(self, *args: object)` needs no rewrite.
- Codon gives every function and lambda a type of its own, so a list or dict
  of different ones does not compile without the annotation, even if you
  annotate nothing else. Annotate where the table is assigned or returned. An
  unannotated `fs = [inc, dbl]` or a class attribute is
  refused with a hint: write an `if`/`elif` chain or a `match` on the key
  instead (`if key == "a": return inc(x)`). A bound method (`self.run`) or a
  lambda using a local variable cannot be converted: use a module-level
  function that takes the object, or the chain.
- `__ge__` of a class that defines `__lt__` only is more than Python gives it
  (`a >= b` is a `TypeError` there): the check accepts it because Codon's sort
  needs it. It does not change what runs.
- A rewritten lambda no longer takes the captured parameter: `fs[0](1, 5)`,
  which passes `k`, is refused (with a hint naming this). A default after a non-literal one is bound the
  same way, so positional calls keep their meaning. Use `functools.partial`
  where you pass the captured value.
- Codon's f-strings have no `{{` / `}}`: a literal brace becomes a string
  joined with `+`.
- A class constant read in the class body (#433): Python resolves the name in
  the class scope, Codon in the module's (`name 'LIMIT' is not defined`), so
  the check writes `Class.LIMIT`. It rewrites exactly what Python resolves
  there: a name an earlier assignment of the same body bound, read by another
  statement of the body, a dataclass field default, a method's parameter
  default or a decorator. A method body, a lambda body and a comprehension
  (but for its first iterable) do not see the class scope, so a bare `LIMIT`
  there is the module's and stays; a nested class sees only its own body; a
  parameter or comprehension variable named `LIMIT` shadows nothing at the
  default, which is read in the class scope. In an Enum a member read by a
  later statement is its value in Python (`B = A` makes an alias), so the check
  writes the literal; any other name of the body is qualified as above.
- The constant has to be a class attribute in Codon. A name annotated at class
  level (`LIMIT: int = 5`, `Final[int]`, `ClassVar[int]`) is a typed field of
  the instance, not an attribute of the class, so a read of it cannot be
  qualified: when its value is a single-line literal the check writes that
  literal where the name is (`n: int = 5`; Python reads the same value), and
  otherwise the read is refused with a hint: write `LIMIT: Final = 5` (a bare
  `Final` is dropped, leaving a class attribute) or no annotation, or use a
  module-level constant.
  A `NamedTuple` field default cannot read its own class constant either
  (`x: int = LIM` is "no module named"): its other statements and methods can.

### The names of `typing` a scenario may import

Codon has no `typing` module of its own beyond a few names, so the check
brings a shim (`typecheck/codon/typing.codon`). A scenario may import these
names from `typing` (#364); `test_typing_names` keeps this list and the shim
in step:

<!-- typing-names:start -->
`TYPE_CHECKING`, `Sequence`, `MutableSequence`, `Iterable`, `Collection`,
`Mapping`, `MutableMapping`, `AbstractSet`, `FrozenSet`, `Any`, `TypeVar`,
`Protocol`, `runtime_checkable`, `Literal`, `IO`, `Iterator`, `Deque`,
`DefaultDict`, `Counter`, `Final`, `Annotated`, `NoReturn`, `Type`, `Sized`,
`Hashable`, `cast`, `Optional`, `List`, `Dict`, `Set`, `Tuple`, `Callable`,
`Union`, `ClassVar`, `Generator`, `overload`
<!-- typing-names:end -->

- `Final` and `Annotated` are read as what they wrap (`Final[int]` and
  `Annotated[int, "m"]` are `int`; a bare `Final` is dropped). `NoReturn`,
  `Type`, `Sized`, `Hashable`, `Any` and `Callable` annotations are dropped,
  so they are not checked.
- `cast(T, x)` is `x` typed as `T` (as in Python, nothing checks that `x` is
  a `T`); when `T` cannot be named in Codon (a `Callable`, a class defined
  further down) it is `x` as it is.
- A class-level `Y: Final[int] = 4` stays a typed field, not a class
  attribute, as any annotated class-level name does; write `Y: Final = 4` for
  a class attribute read as `C.Y`.
- `Iterator` is Codon's generator type, and `Deque`, `DefaultDict` and
  `Counter` are `collections.deque`, `defaultdict` and `Counter`.
- Any other name is refused at its import line. `Generic` and `TypedDict`
  cannot be modelled: leave the type parameters of a class or function
  unannotated (Codon makes it generic over them) instead of subclassing
  `Generic`, and write a `@dataclass` (or a `Dict[str, X]`) instead of a
  `TypedDict`.

### Python constructs Codon refuses (#379)

Each of these is valid, mypy-clean Python that Codon 0.19 cannot compile and
that the check cannot cheaply rewrite. The check refuses it at its line with a
hint, and the working spelling is below. What runs is your code as written; only
the check needs the other spelling.

| You write | Codon says | Write instead |
| --- | --- | --- |
| `case int(n):`, `case Pt(x=x, y=y):`, `case Pt(a, b):` (a class pattern with arguments), or `case Pt():` of a class of your own | `name 'n' is not defined`, or it reads the pattern as a call of `Pt` (`__init__() missing 1 required positional argument`) | `if isinstance(v, Pt):`, then read the attributes (`v.x`, `v.y`); `case int():`, `case str():` and the other builtin types compile |
| `case {"k": v}:`, `case {"k": _}:`, `case {**rest}:` (any name or `_` inside the mapping) | `name 'v' is not defined` | `if "k" in d:` and read `d["k"]`; a mapping pattern whose values are literals, values or or-patterns (`case {"k": 1 \| 2}:`) compiles |
| `case [a, *rest]:`, `case [*_]:` | `unexpected star expression` | a fixed length (`case [a, b]:`), or test `len(xs)` and slice |
| `case [a, b] as whole:` | `unexpected indentation` | match without `as` and use the name the subject already has |
| `def show(h: HasArea)` called with a class that has `area()` but does not derive from the `Protocol` `HasArea` | `'Sq[float]' does not match expected type 'HasArea'` (also `List[Sq]` against `List[HasArea]`) | derive from it (`class Sq(HasArea):`), give the classes a common base class, or annotate `Union[Sq, Circle]`; Codon lists are invariant: annotate a list with the class it holds (`list[Sq]`) |
| `f(*xs)` with a list, `f(**d)` with a dict | `argument after * must be a tuple, not 'List[int]'` / `argument after ** must be a named tuple, not 'Dict[str,int]'` | pass the elements (`f(xs[0], xs[1])`, `f(a=d["a"], b=d["b"])`), or make the parameter a list or dict; a tuple (`f(*(1, 2))`) is accepted |

Plain value, or-pattern, guard, sequence (without `*`) and enum-member
(`case Mode.A:`) patterns compile, as does `case int():` of a builtin type.

A `Protocol` is not structural in the check: Codon compares class names, so a
class is a `HasArea` only if it lists `HasArea` among its bases. Python runs
the same code either way, because a `Protocol` base is only a marker there.

### Standard library names Codon lacks

Codon ships its own `functools` (only `partial`), `itertools` and
`collections`, and a name it lacks fails with a bare "cannot import name".
The check supplies these (#378); `test_stdlib_names` keeps this list, the
check's additions and the tests in step:

<!-- stdlib-names:start -->
`functools.reduce`, `functools.lru_cache`, `functools.cache`,
`functools.cached_property`, `itertools.pairwise`, `collections.OrderedDict`
<!-- stdlib-names:end -->

- `functools.reduce` (with or without an initial value) is a function of the
  model, which the check puts in place of the name.
- `lru_cache` (with or without arguments), `cache` and `cached_property` are
  decorators the check drops, as caching changes no type; `cached_property`
  becomes a `property`. Nothing is cached, so `f.cache_clear()` and
  `f.cache_info()` are refused, and so is assigning to a `cached_property`
  attribute (`self.area = 3` is legal in Python, not on a `property`). A name
  the module binds again (a parameter `reduce`, a `def cache`) is left alone.
  The names are handled where the module imports them, not in a function.
- The initial value of `reduce` must have the type of the accumulator the
  function returns: `reduce(add, [1.5, 2.5], 0)` is refused (write `0.0`).
- `itertools.pairwise` is added to Codon's `itertools`, and
  `collections.OrderedDict` is a `dict`, which keeps its insertion order;
  `move_to_end()` and `popitem(last=...)` are not declared, and are
  refused when called.

These are refused at the import line, with a hint saying what to write
instead:

<!-- stdlib-refused:start -->
`dataclasses.asdict`, `functools.wraps`, `contextlib`, `warnings`, `json`,
`types`, `textwrap`, `traceback`, `io`
<!-- stdlib-refused:end -->

- `dataclasses.asdict` returns a `dict[str, Any]`: build the dict from the
  fields you need. The `dataclasses` shim declares only `dataclass`, `field`
  and `replace`.
- `functools.wraps` (and any other name of `functools`, `itertools` or
  `collections` not listed above) is not modelled.
- `contextlib`: write a class with `__enter__` and
  `__exit__(self, exc_type, exc, tb)` for `contextmanager` (the check rewrites
  the signature, see "Python the check rewrites for Codon"), and `try/except X: pass` for `suppress`. `warnings`: log
  with `logger.warning`. `json`: read into a `@dataclass`. `types`: name the
  type itself. `textwrap`: build the string with `str` methods.
  `traceback`: log with `logger.exception`. `io`: use a `str`, or a
  `pathlib.Path` opened with `open()`.

### Iterating an enum

For a scenario's own `Enum` class (any of `Enum`, `IntEnum`, `StrEnum`,
`Flag`, `IntFlag`, `(str, Enum)`), the check rewrites, as Python reads it:

- `for m in Mode` and a comprehension over `Mode`, and `*Mode`;
- `list(Mode)`, `tuple`, `set`, `frozenset`, `sorted`, `len`, `reversed`,
  `enumerate`, `iter`, `min`, `max`, `zip`, `map` and `filter` (with a
  function) of `Mode`;
- `x in Mode` and `x not in Mode`;
- `Mode["A"]`, a member looked up by name.

Iteration yields the members in definition order. A member whose value repeats
an earlier one is an alias, and is skipped, as in Python. Only the bare name of
the class is rewritten, so a parameter or local called like the class is left
alone, and the enum may be defined in another module of the package.

What does not work (Flag operators and the model enums are tracked in #437):

- the framework's own model enums (`TurnDirection`, ...) are not iterable;
- the class passed as a value, `f(Mode)`, or `Mode.__members__`: both stay as
  written, which Codon does not resolve (`no function 'List.__init__' with
  arguments (List[T], type[Mode])` for `list(e)` in `f`, and `'Mode' object
  has no attribute '__members__'`);
- `sum`, `any` and `all` of the class, and `filter(None, Mode)`: Codon needs
  `__bool__` or `+` on the member (`sum(int(m) for m in Mode)`, `any(m == x
  for m in Mode)`);
- the operators of a `Flag` (`A | B`: `unsupported operand type(s) for |`);
- an enum with no member (`'Empty' object has no attribute '_acs_members'`).

## What the check does not catch

The check compiles the scenario and never runs it, so it says nothing of
what a value is at run time (a division by zero, a missing key, an actor
that is not spawned yet), and it accepts some code that fails in Python.
The known gaps:

- **An `Optional` unwrapped implicitly, where mypy cannot see it.** Codon
  unwraps an `Optional[T]` wherever a `T` is needed (arithmetic, an
  attribute, an argument), and mypy's pass (#239,
  [*What is checked*](#what-is-checked)) catches what mypy can type,
  CARLA's results included (from stubs of typesafe_carla's Codon library,
  #267). It does not see a value mypy reads as `Any`: one from a package
  with no type information (an import it cannot follow, which it ignores,
  such as `lanelet2`), a CARLA result whose type the stubs leave `Any` (an
  enumeration, `carla.command`, a pointer, a method whose overloads return
  different types), or the result of a function with no return annotation.
  Neither does it check the arguments of a CARLA call: an `Optional` passed
  to one is Codon's to unwrap (`world.get_actor(maybe_id)` compiles).
- **A union of the scenario's own classes** used as one of its members.
  The None check's pass refuses only a union made of the members of one of
  the framework's (#309, [*What is checked*](#what-is-checked)): a union of
  classes the scenario defines (a class-level `tuple[A | B, ...]`, held as
  `List[Union[A, B]]`) is converted at run time as Codon converts any union,
  and an un-narrowed use of one member's attribute compiles.
- **A helper with `**kwargs` or `*args` that nothing calls and that reads
  `**kwargs` by key** is not compiled, with a warning: the check has no
  keyword to call it with (#266). One the scenario calls that reads them is
  compiled only with the keywords and the number of values its callers
  pass, and, as a call is known by its name only, not at all when the only
  calls of its name are another's (`d.update(...)` for a helper `update`)
  or are in a function the check does not compile.
- **A value converted by an annotated assignment.** Codon 0.19 converts any
  value with `__bool__` in an annotated assignment: `done: bool = "no"`
  compiles, and is `True`. The check binds hook results strictly (#238) and
  class-level `bool` defaults through `_expect_bool` (#190), but an annotated
  local of the scenario's own code is converted as written. A local
  `x: bool = some_str` is no `bool` in Python.
- **An `int` where a `float` is expected** is accepted, in a call and a hook
  result alike, as Python's typing takes it; not inside a container
  (`[1, 2]` for a `list[float]` is refused).
- **A function the check cannot call** (a parameter with no annotation, or
  one of `Any`, `object`, `Callable[..., R]` or a `TypeVar`, or a return
  annotation naming `Any`) is compiled only where `setup()` and `is_done()`
  reach it, with a warning ([*What is checked*](#what-is-checked)).
- **A hook a class takes from a mixin listed before its framework base**
  (`class Gap(CheckMixin, BaseCondition)`, with `check()` defined in
  `CheckMixin`) is compiled as the mixin's own method, not checked against
  the hook's return type: define the override in the class itself.
- **A `None` default** is not checked against its annotation:
  `speed: float = None` is an `Optional[float]` to Codon (mypy, in the
  scenario package's own pre-commit, reports it; the None check reports only
  its uses).
- **What only a tuple does**: a `tuple[X, ...]` is checked as a `List[X]`,
  so its hash, its equality with another tuple and its immutability are not
  checked.
- **`bytes` is read as `str`**: `path.read_bytes() + "text"` compiles, and
  CPython refuses it.
- **A `values=` that only Codon iterates** (`register_cover()`,
  `OddAttribute`, #279). Codon 0.19 cannot tell a class passed as an argument
  from one of its objects, so an enum member passes where an enum class is
  meant (`values=TurnDirection.LEFT`). A frozen dataclass of the framework
  (a `@tuple`, so iterable to Codon) passes too, and so does an element type
  `expression` never returns.
- **An enum member looked up by value** (`ActionState("completeState")`) is
  typed, but not resolved to the member. Lookup by name and iteration work
  for a scenario's own enum, within the limits of
  [Iterating an enum](#iterating-an-enum).
- **typesafe_carla's Python-API compatibility shortcuts** (a `Vehicle`
  method called on a plain `Actor`) compile, as they do outside its strict
  mode.
- **A scenario registered with `register_scenario_builder()`** is not
  checked at all: only its builder knows how it is built.

## Checking the framework itself

The scenario check trusts the model: it never compiles the framework's own
source. The library check does, a module at a time, as each is made to
compile:

```bash
uv run scenario-check --library                                   # every checked module
uv run scenario-check --library autoware_carla_scenario.kinematics.vector
```

```python
from autoware_carla_scenario.typecheck import typecheck_library

result = typecheck_library()  # the modules typecheck/library.py lists as checked
assert result.ok, result.format()
```

`test_typecheck_library.py` makes the same check in the test suite.

**The checked modules get the None check too** (#277), as a scenario's
modules do: after the compile, mypy runs over every module the run compiled,
with the CARLA stubs, and an `Optional` used without a test for `None` fails
the check at its line. Codon unwraps an `Optional` in the framework as it does
in a scenario, so the compile alone accepted `CollisionCondition` reading
`CollisionEvent.other_actor` (`None` for a collision with map geometry) as a
`carla.Actor`, and the sensor callback raised in Python:

```text
conditions/collision.py:158:32: error: Incompatible types in assignment (expression has type "Actor | None", variable has type "Actor"): Codon unwraps an Optional implicitly, so test the value for None before this use, or Python fails here when it is None (mypy: assignment)
```

It adds about 11 seconds to a run with mypy's cache cold, and under a second
with it warm, to a compile of about a minute. A verdict kept between runs
(below) holds the None check's too: the key already covers what it reads.
`test_typecheck_none_check.py` runs the same None check over `CHECKED`
without Codon, so a machine without a Codon compiler catches it as well.

A run that names modules is a slice of the full run. With the named
modules, it compiles from their source the `CHECKED` modules they import,
transitively. It does not follow an import that `MODELLED_IMPORTS` routes
through the model, or one inside an `UNCALLED` function. It also brings the
other modules of a `REPLACED_MODELS` entry once one of them is in, so that the
model re-exports the checked definitions, as it does in the full run.
`scenario_base`, for example, imports `register_vehicle_entity` from the
checked `entity.registry`, which the model does not declare, and
`entity._spawn` holds the checked `GroundProjectionConfig`, which the model
`VehicleEntityConfig` must share. The modules it imports are checked as
fully as the named ones: their functions and methods are called too, and **errors
are reported for every compiled module**, not only the named ones. An error
in a module the named one imports is an error the full run reports as well,
and it would stop the named module from compiling anyway. Hiding it would
only make the run fail without a reason. The result's title gives both
counts (`1 checked modules, 44 more they import` for `scenario_base`, at the
time of writing). Run without names, the check
compiles exactly `CHECKED`, as before. Every checked module passes on its own.

The check's driver imports the model before any checked module, whichever
module it names first. The model reaches the checked `coordinate.frames`
through `_poses.codon`, and a run that started at `coordinate.frames` would
reach the model, and come back to `coordinate.frames`, before that module
defines anything.

### The manifest: `typecheck/library.py`

`CHECKED` lists the modules compiled from their source; `EXCLUDED` maps every
other module to the reason it is not. `"not yet checked (#45)"` marks a module
nobody has made compile yet; any other reason names what keeps a module out
(`"imports omegaconf, hydra"`: a package or a standard module Codon cannot
compile). numpy is not one: Codon ships its own, and a module's numpy calls
compile. Every module of the package except `typecheck/` is in exactly one of
the two, which the test suite checks, so **a new module needs an entry**.

`UNCALLED` names the few functions and methods of a checked module the
check leaves out (`module.Class.method`, or `module._helper` for a protected
one), each with its reason: one whose job Codon cannot express, in a module
that otherwise compiles. Today that is
what builds a condition's details (`get_details()`, `to_summary_dict()`,
`to_dict()`: a `dict[str, Any]` of mixed value types) and what serialises
them (`ScenarioResult.to_json()`, with `json`), and what reads the
authoring package's pydantic specs, typed `Any`
(`signals.build_controllers()`). Codon compiles a function
only when something calls it, so neither its body nor an import made inside
it is compiled: a module that needs a standard module Codon lacks for one
function imports it inside that function (`import json` in `to_json()`) and
lists the function here. A test checks that every entry names a function or
method of a checked module.

To check one more module:

1. Move its entry from `EXCLUDED` to `CHECKED`.
2. Run `uv run scenario-check --library` and fix what it reports, in the
   module. Only annotations and Codon-friendly rewrites: the module must
   behave exactly as it did.
3. Run `uv run pytest autoware_carla_scenario/test/carla_scenario/test_typecheck_library.py`.

### What is checked of the framework

Every module now has an entry other than `"not yet checked (#45)"`. 195 of
the package's 287 modules (68%; about 47,300 of 84,200 lines) are `CHECKED`:
the kinematics, the coordinate frames and poses, the conversions between
Lanelet2, OpenDRIVE and CARLA world poses (`coordinate.transform`), the stop
lines read off a Lanelet2 map (`utils.stop_line`, `coordinate.stop_line`)
and the lane arithmetic of a relative lane pose (`trajectory.relative_lane`),
the map manager that loads the Lanelet2 map and the OpenDRIVE road network
(`coordinate.map_manager`) and the sweeper's lighter loader of the Lanelet2
map (`sweeper.map_loader`), with the projection and origin both read a map
with (`coordinate.projection`; not how it reads `map_projector_info.yaml`,
whose YAML values are of any type, nor how it moves a `Local` map's points to
their `local_x`/`local_y` tags), what places a trajectory's vertices in CARLA
world coordinates (`trajectory.resolve`),
the sweeper's lanelet constraints and parameter bindings
(`sweeper.constraints`, `sweeper.bindings`) and the route's map questions,
frame, poses and search (`route.geometry`, `route.frame`, `route.positions`,
`route.search`), all against the lanelet2 model, the condition base and
every condition, the action base with the environment, follow-trajectory,
traffic-signal, signal-controller, lane-change, routing, set-speed, turn and
walk-straight actions, the stalled object (`actions.stalled_object`) and the background-traffic source and sink
(`actions.background_traffic`), the logical scenario's route search and matches (`route.model`) and
the scenario's route (`route.active`),
the `signals` package (the signal controllers and their registry), the ego,
NPC vehicle and pedestrian entities, the base of the externally driven egos
(`entity.externally_driven`), the ego that takes its mission over the bridge
(`entity.bridged_ego`) and the Autoware ego on it (`entity.autoware_entity`), the ego driven by ROS 2 nodes over DDS
(`entity.ros2_entity`) with its configs, commands, arbitration and transport
interface (`ros2`, `ros2.config`, `ros2.commands`, `ros2.transport`), the entity registry and the policy
warm-up with the Chaos powertrain model, `BaseScenario`, the scenario
runner's run loop and the queue that runs scenarios one after another
(`scenario_runner`, `scenario_queue`), the UI's runner, which runs them in
a subprocess on a thread (`ui.runner`, over a model of the pydantic
`ui.models`),
the traffic backend interface and its run context with the TrafficManager
backend and the SUMO backend's options, the map source URIs and what
resolves them to files on disk (`maps.resolver`, with the repository listing
of `maps.catalogue`, the `map` config's paths of `maps.config` and the
OpenDRIVE taken from CARLA by `maps.opendrive`), what an exported scenario
package pins the framework to and how it runs `uv` (`authoring.framework_pin`,
`authoring.uv_tool`), the CARLA server manager (`server`), the markers an ego
entity hands the runner for the trajectory log (`debug_markers`), the Autoware
launcher that runs a stack as a process group (`autoware_stack.launcher`),
the `scenario_bridge` node's command and Python root beside it
(`autoware_stack.bridge_node`) and the one that runs a workspace in its dev
container with Docker (`autoware_stack.docker`, over a model of
`autoware_stack.devcontainer`, which reads JSON and YAML), the Autoware
bridge interface and its fake, the trajectory model, the adversity math, location gate, observation, commands and triggers
(`adversity`, NADE and its avoidance included, the actions' `_adversarial_execution`, `adversarial` and `adversarial_traffic`, `conditions.adversity_feasible`), the measures and the metrics, the ODD
unit names (`odd.unit_names`), the ODD's probes, which read its attributes off
the world and, for those the Lanelet2 map decides, off a lanelet
(`odd.probes`), the placement of a road user so a scenario
measure reads a given value (`placement`), the planning of a scenario's route on the
Lanelet2 map that the ODD's route check starts from (`odd.route_plan`), the CARLA-only helpers, the traffic-light
lookups included, and the scenario configs (`scenario_config`, `examples.configs`), the camera
and LiDAR sensors (`sensor`), the six example scenarios, and the external
driver's configs and client interface, trajectory follower and observations
(`driver.base`, `driver.control`, `driver.observation`, over
`driver.geometry`), and the `autoware_bridge`, `autoware_stack`, `entity`,
`driver`, `maps` and `coverage` packages themselves, whose `__init__` re-exports what is not checked
through models of it (see *A class a checked package only re-exports*
below). 93
functions and methods of those modules are `UNCALLED` (21 of them
protected helpers, see *What is compiled* below): most of them a
condition's `get_details()` or a backend's `describe()`, what takes a traffic
backend's options node (`Mapping[str, Any]`), what reads a route search, a
match, a constraint or a binding from YAML values of any type
(`Range.parse()`, `parse_route_search()`, `RouteMatch.from_config()`,
`parse_constraint()`, `parse_binding()`), the `[tool.uv.sources]` entry of a
framework pin, a dict of str and bool values (`Pin.uv_source()`), the two members of
`BaseScenario` built on the ego's class (`create_ego()`, which instantiates
`ego_type`, a `type[EgoVehicle]`) or on cover expressions of any value (`register_cover()`),
the run-time wrapper of `TrajectoryVertex.__init__`
(`_refusing_vertex_times`), the road-marking pattern of the stop lines, which returns a `refers`
parameter (any of lanelet2's primitives) as a line string
(`utils.stop_line._road_marking_stop_lines()`), what moves a `Local` map's
points to their `local_x`/`local_y` tags and rebuilds the map, which the
lanelet2 model has no setters for (`coordinate.projection._read_local_tags()`),
`signals.build_controllers()`, which reads
the authoring package's pydantic specs, what builds the `scenario_bridge`
node's command from its ROS parameters (`bridge_ros_args()`,
`bridge_command()`: a `Mapping[str, Any]` of bool, int, float and str
values; Codon 0.19 fails on a `Union` in a parameter annotation, `union
already sealed`, so the checker drops one and the check has nothing to call
with), what the runner does with `json` or a condition's details (writing
the result and its environment, `_write_result()`; collecting the condition
statuses, `_collect_condition_statuses()`), the queue's pytest fixture
(`ScenarioQueue.as_fixture()`), `maps.config.resolve_map_paths()`,
which reads the `map` config node of any shape (`Any`), the driver and
ROS 2 configs' `from_mapping()` (a Hydra node, `Mapping[str, Any]`), what
builds the ROS 2 ego's published cameras (`entity.ros2_entity._publishing_camera()`,
which imports `ros2.camera` inside it),
`encode_frame_jpeg()` (cv2), and protected helpers of the same kinds: what
reads a `map` setting, a route search's segment, a signal group or phase, or
a `TrajectoryVertex` call's arguments, each of any type (`Any`), walks or
binds a sweeper constraint of any kind, builds a condition's status (its
details), or takes the list a plain tick callback goes into
(`BaseScenario._register_tick()`, a `List[Callable[[carla.World], None]]`,
which the check has no value of). The 92 `EXCLUDED` modules fall in
four groups. Many fit more than one (about 69 import what Codon cannot
compile, 29 build on an excluded module and 20 hold values Codon has no type
for), so each is counted once, in the first group that fits it:

- **69 import what Codon cannot compile**: pyxodr,
  pydantic, pickle, lxml, grpc, omegaconf and hydra, yaml, openodd (a Rust extension),
  cyclonedds (`ros2.cyclone` and `ros2.messages`, the ROS 2 ego's DDS
  transport and its message types), or
  a standard module Codon lacks (`json`, `inspect`, `concurrent.futures`,
  `importlib`
  beyond `importlib.metadata`, or more of `os`, `pathlib`, `shutil`, `signal`,
  `subprocess`, `tempfile`, `threading`, `dataclasses`, `importlib.metadata`
  or `urllib.parse` than their shims declare), at module level or inside a function the check calls. This is
  the tooling around a run (the editor, the UI, authoring, the sweeper's
  expansion, Hydra sweeper and map loader, the map caches (the map
  repositories, the Lanelet2 z offset and the pyxodr road network), the Autoware
  stack's dev container, the
  record of a run's environment, the SLURM launcher's submitting side,
  its worker and its manifest, the batch summary they share, and the Rerun
  recordings of a run's logs), the ODD and coverage modules that wrap
  openodd, and the rest of what reads Lanelet2. A reason names `pathlib`
  only where the module uses more of it than the shim declares, and the
  other shimmed modules likewise; `test_typecheck_library.py` holds every
  reason to that.
- **19 build on an excluded module**, and import nothing Codon lacks
  themselves: they import a module that is neither checked nor modelled, or
  names that its package's model does not declare. Among them are the
  packages that re-export one (`authoring`, `odd`, `sweeper`,
  `examples`, `scaffold`, `slurm`), `slurm.collect`, the
  authoring compiler, builders and validator, `odd.route`, the coverage
  half of the route check (names of `odd.model` and `coverage.items` that
  the odd and coverage models do not declare, such as `OddDefinition.evaluate()`
  and `value_label()`; it also holds each lanelet's attribute values,
  `dict[str, Any]`, which it hands to openodd, and finds a probe's
  `on_lanelet` reader on any callable) and
  `traffic.registry` (the `registry`, in its plugin loader); 4 of the 19
  also hold values Codon has no type for.
- **3 hold values Codon has no type for**, and nothing else keeps them out:
  `Any` (`trajectory.authoring`, `authoring.registry`'s
  field defaults), or a class as a value (`utils.config`, which `traffic.config`,
  `traffic.fidelity` and `traffic.sumo.config` import only in their uncalled
  `from_mapping()`).
- **1 has a callback the check cannot type**: `ros2.camera`, whose camera
  hands each frame back to the ROS 2 ego from a CARLA sensor thread through
  `CarlaCameraSensor`'s listen callback.

The scenario check is not limited by this: it compiles every scenario against
the model. The model does not declare the whole public API: what
`test_typecheck_model.py` guarantees is that every public name of a modelled
module, and every public method, property and class attribute of a class the
model declares, is either declared or listed with the reason it is left out
in `typecheck/not_modelled.py` (`NOT_MODELLED`, `NOT_MODELLED_MEMBERS`), and
the check refuses a use of a listed one with that reason (#249, #265).
Attributes a class only annotates (set on the instance) are outside that
guarantee.

### What is compiled

Codon checks a function only when something calls it, so the check appends a
function to each checked module that calls every function and every
method of every class (constructors, static and class methods,
properties and their setters, dunder methods, protected `_` ones and those
of `_` classes included) with a value of
each parameter's annotated type:

```python
def _acs_library_check():
    normalize_angle_deg(_acs_value(float))
    Vector3(_acs_value(float), _acs_value(float), _acs_value(float))
    _acs_value(Vector3).dot(_acs_value(Vector3))
```

`_acs_value(T)` is a `T` Codon cannot tell from a real one; nothing compiled
is ever run. A parameter annotated with a union (`Lanelet2Pose |
OpenDrivePose`, `Union[BaseAction, Callable[[carla.World], None]]`, or an
`Optional` of one, `Optional[Union[EntityRole, str]]`) is
called once with each member, so the body is checked for every type it
accepts. A parameter with no annotation, or one Codon cannot
express (`Any`, `object`, `Callable[..., R]`, `Callable[[Any], Any]`, ...),
gives the check nothing to call with and is reported as a problem of the
module, at its line: annotate it with the type its callers pass. The
exceptions are `other: object` in `__eq__` and `__ne__`, which Python
requires, called with the class itself, and a parameter Codon cannot express
that has a default: a function whose every such parameter has one is called
only leaving the defaults out (*Defaults are compiled too*, below), so its body
is compiled with them (#199). A `Callable` whose argument and result
types Codon can express (`Callable[[carla.World], Optional[float]]`, a
dataclass field too, or an `Optional` of one) is called with a value of
Codon's `Callable` type, which a function of that signature converts to. One
with a union among its argument types (`Callable[[Optional[Union[EntityRole,
str]]], CarlaWorldPose]`, the reference lookup of `trajectory.resolve`) is no
type Codon 0.19 can hold: it crashes on it ("union already sealed"). Such a
parameter is called with `_acs_Returning[R, N]` (`_tc.codon`), a callable
that takes `N` arguments (as many as the `Callable` does) of any type and
returns an `R`, so the body is compiled with the number of arguments it passes
checked, their types unchecked, and with what it gets back checked.

A `*args: T` is given one `T` (`run_uv(cwd, *args: str, ...)` is called
with one `str`), so the body is compiled for it; one with no annotation, or
of `object` or `Any` (`__exit__(self, *args: object)`), is given nothing.

A method with keyword-only parameters or `**kwargs` is called through its
class (`TrafficBackend.start(_acs_value(TrafficBackend), ...,
skip_actor_ids=...)`): Codon 0.19 calls a method a subclass may override
through a dispatch thunk, which takes neither a keyword argument nor
`**kwargs`, and the call through the class checks the same body.

**Defaults are compiled too** (#183). The calls above pass every
argument, so no default was ever compiled: `find_nearest_traffic_light(...,
max_distance: float = "far")` passed the check. A function or method with a
defaulted parameter is now called once more, with only the arguments that
have no default, each of its first type:

```python
    find_nearest_traffic_light(_acs_value(carla.World), _acs_value(Lanelet2Pose))
    AutowareBridge.configure(_acs_value(AutowareBridge), _acs_value(BridgePose), _acs_value(BridgePose))
```

A method is called through its class (`AutowareBridge.configure(...)`, not
`_acs_value(AutowareBridge).configure(...)`): a dispatch thunk fills in no
default, and through it `AutowareBridge.configure()` and
`FakeAutowareBridge.configure()` do not compile. A `*args: T` is still given
one `T` (`run_uv(cwd, *args: str, timeout: int, env=None)` with one `str`
and `timeout`), so the positional parameters before it are passed, defaulted
or not: without it `run_uv()` does not compile. A function with no default gets no
such call. An error in a default is reported at the call, which names it
(`<library check: find_nearest_traffic_light(world: carla.World, location:
Lanelet2Pose), with the default of `max_distance`>: error: 'str' does not
match expected type 'float'`).

That was 169 calls more, beside the 1525 others (1694 in all), when it came
in. Measured before any fix, 13 of them failed, and all 13 are fixed by
the transform, with no change to a module:

- 11 had a `()` default on a `Sequence[X]` or `Collection[X]` parameter
  (`RoutingAction.__init__`'s `waypoints`, `set_mission()` / `route_to()` of
  `BridgeHandshake` and `BridgedEgoEntity`, `EgoVehicle.route_to()`, the
  `skip_actor_ids` of the three traffic backends' `start()`,
  `utils.vehicles.hold_vehicles_still()` and
  `scenario_runner._hold_vehicles_still()`). The parameter's annotation is
  dropped, so the `()` made it Codon's empty tuple ("'Tuple' does not match
  expected type 'List[Lanelet2Pose]'"). The transform makes such a default
  `List[X]()`, as it does a dataclass's `field(default=())`.
- 2 had a `None` default on an `Optional` parameter whose annotation the
  transform drops: `EgoConfig.__init__`'s `waypoint_poses:
  Optional[Sequence[Lanelet2Pose]]`, and the `Callable` parameters of
  `trajectory.resolve.resolve_trajectory()`. `None` alone is Codon's
  `NoneType`, which has no elements and no call, and a body that tests it
  for `None` first is compiled with it all the same. The transform makes
  such a default the `None` of the `Optional` the library check calls with:
  `Optional[List[Lanelet2Pose]]()`, `Optional[Callable[[float, float],
  Optional[float]]]()`, and for a `Callable` with a union among its
  arguments `Optional[_acs_Returning[CarlaWorldPose, 1]]()` (the transform's
  prelude imports `_acs_Returning`; the check's own function does not import
  it again, which Codon 0.19 cannot type: "cannot typecheck").

**So are the field defaults of a dataclass** (#188). The check drops
`@dataclass`, and Codon gives the class a constructor that takes each field,
with its default; every constructor call above passes every field, so no
field default was compiled: `stop_timeout_s: float = "far"` on
`CommandAutowareLauncher` passed the check. A dataclass with a defaulted
field is now constructed once more, with only the fields that have no
default, each of its first type:

```python
    DockerAutowareConfig(_acs_value(Path), _acs_value(Sequence[str]))
    DriverClientConfig()
```

A field without a default after one with (a `field(kw_only=True)`) is passed
by keyword. A field with no default of its own (`field()`, or
`field(init=False)`, which the constructor leaves out anyway) is not one
with a default, and neither is a field "defaulted" to OmegaConf's
mandatory-value marker (`spawn_lanelet_id: int = MISSING`, the string
`"???"`): the marker is no value of the field but a placeholder for one the
config must supply, so the call passes the field, and the marker is not
compiled against the field's type. Nothing enforces that in Python: reading
the field from Hydra's config raises `MissingMandatoryValue`, but
`OmegaConf.to_container()` returns `"???"` for it, and a dataclass built
without it (`NpcVehicleConfig()`) holds the string, which the check does not
see. A field the call leaves out because its annotation names `Any` (*A
dataclass field may hold a union or a free-form mapping*, below) is left out
of this call too. An error is
reported at the call, which names the defaults it leaves out
(`<library check: CommandAutowareLauncher.__init__(argv: Sequence[str]),
with the default of `env`, `log_dir`, `stop_timeout_s`>: error: 'str' does
not match expected type 'float'`).

That is 96 calls more, beside the 1694 others (1790 in all, 265 of them
leaving defaults out), at the time of writing. Measured before any fix, 5
of the 96 failed, at the first error of each. Those 5 classes have 10
defaults Codon refused, of four causes; all are fixed in the check, with no
change to a module and nothing `UNCALLED` (9 by the transform, whose output
changes at those 9 defaults alone, and 1 by the call):

- 3 had a tuple display default on a `Sequence[X]` or `tuple[X, ...]`
  field (`DockerAutowareConfig.colcon_args`, `DriverClientConfig.cameras`
  and `map_formats`). The field is declared as a `List[X]` (the `typing`
  shim's `Sequence` is `List`), and the display is a fixed-length Codon
  tuple ("'Tuple[str,str,str]' does not match expected type 'List[str]'").
  The transform makes it `List[X]([...])`, a list of the field's element
  type, replacing only the brackets, so the elements are rewritten as
  anywhere else.
- 4 had a plain `()` default on such a field
  (`DockerAutowareConfig.ros_domain_ids` and `docker_args`,
  `DriverClientConfig.lidars`, `PlannedRoute.via`), which only a
  `field(default=())` of a `tuple[X, ...]` was typed for. The transform
  makes it `List[X]()`, written directly or through `field(default=())`,
  for a `Sequence[X]` as for a `tuple[X, ...]`.
- 2 had `field(default_factory=dict)` on a `Mapping[str, str]` field
  (`env` of `DockerAutowareConfig` and of `CommandAutowareLauncher`): the
  factory is typed only for a `dict[K, V]` field, so it was a bare `{}`,
  a dict of `None` ("'Dict[NoneType,NoneType]' does not match expected type
  'Dict[str,str]'"). The transform makes a `list`, `dict` or `set` factory
  of an abstract collection the concrete one the `typing` shim maps it
  onto, of the field's types (`Dict[str, str]()`).
- 1 was the OmegaConf marker on an `int` field
  (`NpcVehicleConfig.spawn_lanelet_id: int = MISSING`, "'str' does not
  match expected type 'int'"), which the call now passes, as above.

**So are class-level defaults** (#190). On a class with a constructor, its
own or a base's, Codon makes a class-level `x: T = v` a field and drops `v`:
no call compiles it, and `BackendDriven._tm_port: int = "x"`,
`BaseScenario.STABILIZE_TICKS: int = "a"` and `AutowareEgoEntity.stack_name:
str = 1` passed the check. The check's function now assigns each such default
to a local of its type, which compiles the value against the annotation:

```python
    _acs_cls_default_10: int = DEFAULT_TM_PORT
    _acs_cls_default_13: bool = _expect_bool(True)
```

Each local is named after its line in the function, and an error is reported
at it, labelled with the member (`<library check: class-level default of
BackendDriven._tm_port>: error: 'int' does not match expected type 'str'`;
Codon names the two types the other way round in an annotated assignment). A
`bool` default goes through `_expect_bool()` (`_tc.codon`): Codon converts
any value assigned to a `bool` local with its `__bool__`, so `ready: bool =
"x"` would compile, and a call converts nothing. A default written without
an annotation, overriding one a base declares (`REISSUES_BY_DEFAULT = True`
on `FollowTrajectoryAction`, whose base declares `REISSUES_BY_DEFAULT:
ClassVar[bool] = False`), is checked against the base's annotation, a
`ClassVar[T]`'s `T`; the base is found by name among the classes of its
module and the other checked modules, depth first. Left out:

- the defaults of a dataclass, compiled by the constructor call that leaves
  them out (above), and the members of an enum, which are its values;
- a `ClassVar`, which Codon keeps as a class variable, with its value;
- every default of a class that has no `__init__`, its own or a base's, and
  that its module's top level constructs with no argument (`_CACHE =
  _Cache()` in `measures`): Codon's generated constructor takes each field
  with its default, and that call compiles them already (an error is
  reported at the call);
- a default without an annotation that no base declares (a base outside the
  checked modules), a dunder (`__slots__`), and one over a bare `ClassVar`;
- an annotation Codon cannot express or naming a `TypeVar`, and a value or
  annotation naming anything else the class body binds (`TWICE: float =
  LIMIT * 2`; a member unpacked, `LOW, HIGH = 0.0, 1.0`; a method; a nested
  class), which at the function's scope names something else, or nothing.

That is 22 defaults, at the time of writing: 20 annotated ones, in 10
classes (`BaseScenario`, the five ego entities, `TurnAction`,
`CollisionCondition`, `DrivenEntity` and `BackendDriven`), and the 2
unannotated overrides of `BaseAction.REISSUES_BY_DEFAULT`
(`FollowTrajectoryAction`, `_BackendAction`). Of the 34 non-`ClassVar`
class-level defaults outside dataclasses and enums, the other 14 are those
of `measures`' `_Cache`, `_Lines` and `_FrameMemo`, which the module
constructs; the 7 `ClassVar`s are left out. None failed when it came in.

**So are the values a function returns as a `tuple[X, ...]`** (#191). Codon's
tuples have a fixed length, and the transform dropped such a return
annotation, so nothing checked what the function returns:
`signals.registry.registered_signal_controllers()` returning `1` passed the
check. A class-level `tuple[X, ...]` is a `List[X]`, and `tuple(x)` of a list
is one, so the transform now writes the return annotation as `List[X]`
(`Optional[List[X]]`, `dict[str, List[X]]` inside another type), and the
error is reported at the `return`. It stays dropped when it names an abstract
collection (a parameter's is dropped, so the function may return it as it
got it), a class defined further down the module or a `TypeVar`, and so does
a `Union`, `Any` or `Callable` return (a `Callable` return kept fails for a
nested function). That is 10 return annotations, in 6 modules, at the time
of writing: `actions.routing`, `maps.resolver`, `route.model`,
`signals.registry`, 3 in `ros2.config` and 3 in `signals.controller`
(one of them `dict[str, tuple[int, ...]]`). None failed when it came in.

**Protected functions and methods are called too** (#175). A `_` method is
often a hook a subclass or another module uses: six example scenarios call
`BaseScenario._setup_ego_spawn()`, and the model's `BaseScenario` is what a
scenario compiles against, so before the driver called it, nothing compiled
its real body, and an error injected into it passed the check. The check now
calls every `_` function of a checked module and every `_` method of its
public classes, as it calls a public one: about 340 of them, beside about
920 public ones. Measured before any fix, 25 of them failed: 15 had a
parameter the check cannot call with, 8 a bare `np.ndarray` parameter, and
`_setup_ego_spawn()` and `scenario_runner._build_condition_status()` did not
compile. 12 were fixed in the module (shape-typed arrays, an annotated
local, an lxml element type for `coordinate.transform._cubic()`, and the
driver taking `Optional[Union[A, B]]`), and 13 are `UNCALLED`, each with the
reason, most of them because they take `Any` and only an `UNCALLED` function
calls them.

**The methods of `_` classes are called too** (#182). A `_` class is a
module's own helper, and was left to be compiled through what uses it, but
what uses it does not always compile it. Of the 42 methods of the 20 `_`
classes of the checked modules, an error injected into each found 4 that
nothing compiled: `_Never.__init__` and `_BackendAction.__init__`
(`actions.background_traffic`), reached only from constructors that are
`UNCALLED`; `_BackendAction.set_traffic_backend()`, which
`BaseScenario._hand_backend_to()` calls behind a `hasattr()` on a
`BaseAction`, which Codon decides statically as false; and
`route.search._Search.__post_init__()`, which Codon's constructor does not
call. The check now calls them as it calls a public class's: 53 calls
(for the 48 methods of the 21 `_` classes there are since #180), beside
about 1470 others, none with a parameter the check cannot call with.
Measured before any fix, 3 of the first 47 failed, and all 3 were fixed in the module:
`_ScenarioClock` declares the `_world` and `_simulated_start` it assigns on
`self`, and `_BackendAction.__init__` branches into a `BaseCondition` local
instead of choosing between `until` and `_Never()` with a conditional
expression, which Codon 0.19 cannot type. Each of the 4 now reports an
injected error. Two things are left out:

- a module's dunders (PEP 562's `__getattr__`), which are Python's hooks, not
  the module's code; a class's (`__eq__`, `__enter__`) are called, as before;
- what `UNCALLED` names, as for a public function.

A protected member that calls an `UNCALLED` one calls it in the `else`
branch of `if TYPE_CHECKING:`, as a public one does (see *A call into a
module Codon cannot compile, from a function it must* below).
`_setup_ego_spawn()` needs no such branch: the `register_route_to_goal()` it
ends with is checked (#174).

**A constructor is compiled with the defaults of what Codon cannot
express** (#199). `BaseScenario.__init__` takes the ego's class (`ego_type:
type[EgoVehicle] | None = None`), and Codon cannot hold a class as a value, so
the whole constructor was `UNCALLED`, though only one line stores it: its
other ~25 lines were compiled nowhere (the example scenarios compile against
the model's `BaseScenario`), and an injected `self._tm_port = "x"` passed
every run. A function some of whose parameters Codon cannot express, each
with a default, is now called once, leaving out every defaulted argument,
instead of being a problem:

```python
    BaseScenario(_acs_value(EgoConfig))
```

and the one line that stores the class is under `if not TYPE_CHECKING:`,
which Python runs and Codon does not (`ego_type` stays declared at class
level, so `ego_requires_goal` still reads it). The error is reported at its
line, through the call that leaves the defaults out (`<library check:
BaseScenario.__init__(ego_config: EgoConfig), with the default of
`spawn_pose`, `ground_projection`, `random_seed`, `ego_type`,
`ego_entity`>`). A parameter Codon cannot express with no default is a
problem as before, and so is a defaulted one that call would still pass (one
before a `*args: T`, which is given one `T`). `ConditionStatus.__init__`
(its `details: dict[str, Any]`) and `TrafficConfig.__init__` (a backend's
options, `dict[str, Any]`) are compiled the same way, with no `UNCALLED`
entry, so a wrong default of one of their other parameters is reported
(#217).

**Property setters are called too** (#200). Codon compiles an `@x.setter`
only where something assigns the property, so an error injected into the
setter of `ScenarioRunner.xodr_path` passed a run of `scenario_runner` alone:
only the full run found it, through `ScenarioQueue.start()`. The check now
assigns each property that has a setter, once with each member of the
annotation of the value it takes:

```python
    _acs_value(BaseScenario).goal_pose = _acs_value(Optional[Lanelet2Pose])
    _acs_value(ScenarioRunner).xodr_path = _acs_value(Optional[Path])
```

An error is reported at its line, through `<library check:
ScenarioRunner.xodr_path (setter)>`. That is 3 assignments, for the 3
setters of 2 modules (`scenario_base` and `scenario_runner`), at the time of
writing; none failed when it came in. A deleter is not called. The scenario
check appends the same calls to a scenario's own modules (#197), so a setter
of a scenario nothing assigns is compiled too.

### Checked modules and the model

The checked modules are compiled next to the same model the scenario check
uses. Every import of the framework in a checked module is pointed at what
stands in for it in the check's workspace:

- another **checked** module: its own (rewritten) source;
- any other module: its **model**, the `codon/autoware_carla_scenario/` module
  of the public package it belongs to (`coordinate.poses` is modelled by
  `coordinate.codon`), or, for a module of a package that is itself checked,
  a model of that module alone (`utils/traffic_light.codon`,
  `autoware_bridge/grpc_server.codon`, `driver/egodriver_client.codon`,
  `maps/cache.codon`, `autoware_stack/devcontainer.codon`,
  `ros2/cyclone.codon`), or, for a module of
  the top-level package, a model of that module (`carla_install.codon`, for
  `server`, declares `installed_executable()`; `trajectory_recorder.codon`,
  for `scenario_runner`, the `TrajectoryRecorder` it starts, records and
  closes, whose JSON lines hold values of mixed types; `environment.codon`,
  for `scenario_runner`, `capture()` and the id of the `RunEnvironment` it
  returns). A module with no
  model (`utils.config`) cannot be imported by a checked module until it is
  checked itself.

Each checked module gets the `__file__` Python sets on it (its source's path,
a `str`), which Codon 0.19 does not, appended after the calls of the check
so no line moves: `autoware_stack.bridge_node` finds the installed
`autoware_bridge` package with `from ..autoware_bridge import __file__`.
Codon has no module object, so `from .. import autoware_bridge` and
`autoware_bridge.__file__` do not compile.

Two checked modules that import each other can each need the other's types
the moment Codon reads it: `BaseScenario` declares `_measures: dict[str,
Measure]` at class level, and `measures` declares the running scenario
(`_MEASURED: dict[str, Optional[BaseScenario]]`) as a global, so neither can
be read first. `MODELLED_IMPORTS` in `typecheck/library.py` breaks such a
cycle: the module it names (`measures`, `entity.ego`) imports the other
(`scenario_base`) through its model, as it did before that one was checked.
Its types are the model's, which never meet the checked module's.
The egos derived from `EgoVehicle` (`entity.bridged_ego`,
`entity.autoware_entity`) do not override `EgoVehicle.spawn()`, so they never
name `EgoConfig` and need no entry: Codon 0.19 cannot call `super().spawn()`
through an intermediate class ("'EgoVehicle' does not match expected type"
in its dispatch thunk), and an override that repeated the spawn instead would
take `scenario_base` through the model as `entity.ego` does. Through the
checked `scenario_base`, a run with an example scenario in it failed in Codon
0.19's code generation (`LLVM: error: value doesn't match function result
type '{ ptr, ptr }'`, in `opt_ref_new` of the model's `BaseScenario`).

A model module that stands in for unchecked code declares only what the
checked modules and scenarios use of it. `coordinate.codon` still declares,
of `coordinate.projection`, the `map_origin()`, `resolve_projector()` and
`_parse_geo_reference()` that `coordinate.map_manager` and
`sweeper.map_loader` import, though the library check now compiles all three
from their source. Its `MapManager` is now the scenario check's
alone: the library check compiles `map_manager.py` from its source, which
every checked module imports, and nothing of the model takes or returns a
`MapManager`, so it needs no `REPLACED_MODELS` entry.

A checked module is compiled under a name of its own
(`_acs_lib.autoware_carla_scenario__coordinate__frames`), since Codon reads
the workspace's packages only from `.codon` files and names a file in its
errors by its base name alone; errors are reported at the line of the real
source file. A type a checked module defines is not the type the model
declares under the same name, so a checked module cannot hand its own value to
a model function that wants the model's: check modules bottom-up, before the
modules that use them.

Where the model's functions take the types a checked module defines (the
poses, which `to_opendrive` takes and returns), the model's own declarations
of them live in a model module of their own (`_poses.codon`), which
`REPLACED_MODELS` in `typecheck/library.py` names with the modules it
models. Once those are all checked, the library check compiles that model
module as a re-export of their definitions, so the model and the checked
modules share one `Lanelet2Pose`. A name the model declares that none of the
checked modules defines is reported.

### Conditions: the checked base and the model

`conditions.base` is checked, so the checked conditions derive from the real
`BaseCondition` (`_acs_lib.autoware_carla_scenario__conditions__base`), not
from the model's. The two are different types that never meet:

- the **scenario check** compiles a scenario against the model only; no
  checked module is in its workspace, and the model's `BaseCondition`,
  `_expect_condition` and `_acs_list` (which recognises a condition by the
  model-only `_acs_condition` marker) are unchanged;
- the **library check** compiles the checked conditions against the checked
  base. A checked module that hands a condition to a model function (an
  unchecked action) cannot do so until that module is checked too
  (`scenario_base` is checked, and imports `BaseCondition` from
  `conditions.base` for that);
  `_acs_list` gives a list display of checked conditions the type of its
  first element (`_acs_base` leaves a checked condition in a dict display as
  it is), so a display mixing two checked condition classes waits for the
  same.

`BaseCondition.__init_subclass__`, which wraps every subclass's `check()` to
record its last result for the UI, is in `UNCALLED`: the check sees each
`check()` as written.

`actions.base` is checked the same way: a checked action takes the checked
`BaseCondition`, so it imports it from `conditions.base` rather than from the
`conditions` package, which the model stands in for until it is checked. Its
`BaseAction` and `ActionState` are again not the model's, so
`conditions.action_state` watches checked actions. An action that hands its
base a condition of its own making imports that condition from its checked
module too (`SetSpeedAction`'s default `SpeedCondition` from
`conditions.composition.speed`); one whose module is not checked waits for it.

### Making a module compile

What Codon 0.19 needs that Python does not, beyond
[the rewrites above](#writing-a-scenario-that-type-checks):

- **Annotate what the check calls**: every parameter of a function or method
  it calls, protected (`_`) ones included, with a type Codon can express. Spell a union out (`Lanelet2Pose | OpenDrivePose`) rather
  than through an alias (`AnyPose`); a union with `object` or `Any` in it
  gives nothing to call with.
- **A default must have the parameter's type** (#183): the check compiles
  each function once with its defaults. A `()` default of a sequence
  parameter is checked as an empty `List[X]`, so `X` must be a type Codon can
  express and name there: a class defined further down the module is not
  (the `()` is left as it is, and fails). A `None` default of a dropped
  `Optional` is typed only for a sequence or a `Callable` the check can call
  with; `Optional[Callable[..., R]]` and `Optional[Mapping[K, V]]` stay a
  bare `None`, so a body that uses them past an `is None` test fails. A
  tuple display default (`= ("a", "b")`) of a sequence parameter is checked
  as a list of its elements, `List[X](["a", "b"])`, as a dataclass field's
  is (#188); a fixed-length Codon tuple is no `List[X]`.
  An empty `[]` / `{}` default of a `list[X]` / `dict[K, V]` parameter (or
  `[]` of a sequence) is checked as an empty one of that type; Codon would
  make it a list of `None`. A `None` default is not checked against its
  annotation: Codon makes `speed: float = None` an `Optional[float]`, which
  passes (mypy, in pre-commit, reports it).
- **Declare attributes** on any class in a hierarchy (exceptions included)
  and on a class with `__slots__`, at class level (`_value: str`).
- **A class-level default is compiled as written** (#190): the library check
  assigns it to a local of its annotation in its own function, which the
  transform does not rewrite. So a default the transform would rewrite in a
  dataclass (a tuple display or `()` of a `tuple[X, ...]` member,
  `frozenset()`) is reported: assign such a value in `__init__` instead. A
  default naming another member of the class
  (`TWICE: float = LIMIT * 2`), or anything else its body binds, is not checked, as at the function's scope
  the name is not the member; neither is a `None` default, which Codon
  makes an `Optional` of any annotation (`x: float = None` compiles). An
  override without an annotation (`REISSUES_BY_DEFAULT = True`) is checked
  against its base's annotation only when the base is a class of its module
  or of a checked one.
- **`hasattr()` instead of `getattr(x, name, default)`**: Codon decides
  `hasattr` when it compiles, so the branch for a type without the attribute
  is never compiled.
- An exception's `__init__` passes one message string to `super().__init__`.
  An exception class with nothing in its body but a docstring needs that
  `__init__` written out (`def __init__(self, message: str) -> None:`).
- **`d.get(key, default)` takes a default of the value's type**, as Codon's
  `dict.get` does; `d.get(key)` is an `Optional` of it, as in Python (the
  model's `_tc.codon` declares it). Likewise `if not x: continue` instead
  of `(x or {})`, `s.startswith(a) or s.startswith(b)` instead of
  `s.startswith((a, b))`, and `if x is None: return None` / `return x.attr`
  instead of `None if x is None else x.attr`.
- **A dict of values of different types** (`{"at": None, "around": None}`
  read back as a tuple and a list) becomes a small class with one declared
  attribute per key. A cache that told "not read yet" from a read that found
  nothing with an `object()` sentinel (`cache.get(key, _UNREAD)`) keeps the
  names it has read in a `set[str]` beside them: `odd.probes._Frame` holds
  the frame's waypoint, weather, lanelet tags, ... each `Optional`, and
  `read`, so a read that failed is still kept as `None` for the frame, as
  before.
- Standard modules Codon does not have (`json`, `inspect`, `tempfile`,
  ...) are not available to a checked module; the `typing`,
  `dataclasses`, `enum`, `logging`, `abc`, `queue`, `pathlib`, `hashlib`,
  `os`, `shutil`, `subprocess`, `signal`, `atexit`, `shlex`, `sys`,
  `threading`, `importlib.metadata`, `urllib.parse` and `__future__` shims
  of `codon/` are. Each takes the
  arguments CPython takes, so the check accepts what CPython accepts and
  refuses the rest; where Python versions differ, it takes what Python 3.10,
  the oldest the package supports, takes. `pathlib` declares `Path(*parts)`,
  `/`, `str()`, `==`, `hash()`, `expanduser()`, `resolve()` and
  `Path.home()` and `Path.cwd()`; like Python's, it builds and joins a `Path` with a `str` or
  an `os.PathLike` only (a `Path`, or what `shutil.copy2()` returns for one;
  an `Optional` of one is unwrapped, as Codon unwraps one passed anywhere). It declares the file-system queries too: `exists()`,
  `is_file()`, `is_dir()`, `glob(pattern)`, `rglob(pattern)` and
  `absolute()`, with Python
  3.10's signatures (3.12 and 3.13 added the keyword-only `follow_symlinks`,
  `case_sensitive` and `recurse_symlinks`, which are refused), and `name`,
  `suffix`, `parent`, `parents` (indexed, iterated, `len()` and `in`; not
  sliced), `stem`, `with_name(name)` (of a `str`), `as_posix()`, `relative_to(other)` (one path, by position:
  Python 3.10 also joins more parts, but 3.14 refuses them, so the check
  does too; 3.12's `walk_up` is refused), `is_relative_to(other)` (one path,
  by position, likewise), `is_absolute()` and the ordering of two `Path`s, so
  `sorted()` takes them. Nothing
  touches the file system: a query is `False` and a glob yields nothing.
  `is_symlink()` takes no argument, as in every supported Python. Of what
  reads, `read_text(encoding=None, errors=None)` (3.13's `newline` is
  refused) and `read_bytes()` are declared (`autoware_stack.docker` reads a
  launch file and hashes the `package.xml` files, `coordinate.map_manager`
  an OpenDRIVE file); both return `""`, and
  `read_bytes()` returns a `str`, since the check reads `bytes` as `str`
  (so it takes `path.read_bytes() + "text"`, which CPython refuses, and
  refuses `path.read_bytes().decode()`, which CPython takes). Of what
  writes, `mkdir(mode=0o777, parents=False, exist_ok=False)` and
  `write_text(data, encoding=None, errors=None, newline=None)` are declared
  (`maps.opendrive` caches what it fetches), and so are `unlink(missing_ok=False)`,
  `symlink_to(target, target_is_directory=False)` (a target that is a `str` or
  an `os.PathLike`) and `open(mode="r", buffering=-1, encoding=None,
  errors=None, newline=None)`, which returns a file object, `typing.IO[str]`
  whatever the mode, since the check reads `bytes` as `str` (so `IO[bytes]`
  and `IO[str]` are one type). `typing.IO` declares `name`, `mode`,
  `closed`, `close()`, `fileno()`, `flush()`, `read()`, `readline()`,
  `write()` (of a `str`, read as bytes too), iteration over the lines and
  `with`. `chmod()`, `write_bytes()` and the rest are not. `os` replaces Codon's own (whose
  `environ` has no `get()` and iterates over pairs), keeping its `getenv()`,
  which Codon's runtime and typesafe_carla's library call: `environ` is a
  mapping of `str` to `str` (`[key]`, assignment, `del`, `in`, iteration
  over the keys, `len()`, `get(key)`, `get(key, default)`, `pop()`,
  `setdefault()`, `keys()`, `values()`, `items()` and `copy()`, a `dict`, as
  CPython's returns), `getenv(key, default=None)` (a `str` default gives a
  `str`, `None` an `Optional[str]`; a default of another type, which CPython
  takes, is refused), `fspath()`, `getpid()`, `getppid()`, `getpgid(pid)`,
  `getuid()`, `getgid()`, `readlink(path, *, dir_fd=None)` (of a `str` or an
  `os.PathLike`, a `str`; a bytes path is not declared),
  `kill(pid, signal, /)`, `killpg(pgid, signal, /)` (of an `int` or a
  `signal.Signals`), `system(command)`,
  `mkdir()` (Codon's had one), the separators (`sep`, `pathsep`, `linesep`,
  `devnull`, ...) and the `access()` modes. `os.path` declares `join()`,
  `basename()`, `dirname()`, `splitext()`, `isabs()`, `exists()`, `isfile()`,
  `isdir()`, `expanduser()` and `abspath()`, of a `str` or a `Path`, with
  CPython's parameter names. `shutil` declares `copy2(src, dst, *,
  follow_symlinks=True)`, `move(src, dst)`, `rmtree(path,
  ignore_errors=False)` and `which(cmd, mode=1, path=None)` of a `str`
  command; the callables CPython also takes (`onerror`, `copy_function`)
  are not declared. `subprocess` declares `run(args, *, input=None,
  capture_output=False, timeout=None, check=False, ...)` and `Popen(args,
  *, ...)` with the keywords of CPython's `Popen` the framework passes
  (`stdin`, `stdout`, `stderr`, `cwd`, `env`, `text`, `encoding`, `errors`,
  `start_new_session`), a standard stream being `None`, one of the constants,
  a file descriptor or a file object (`typing.IO`, what `Path.open()`
  returns), `CompletedProcess(args, returncode, stdout=None,
  stderr=None)` and its `check_returncode()`, `Popen`'s `poll()`, `wait()`,
  `communicate()`, `send_signal()` (of an `int` or a `signal.Signals`), `terminate()`, `kill()`, `pid`,
  `returncode` and standard streams, `PIPE`, `STDOUT`, `DEVNULL`, and
  `SubprocessError`, `CalledProcessError(returncode, cmd, output=None,
  stderr=None)` and `TimeoutExpired(cmd, timeout, output=None,
  stderr=None)`. The command is a list of `str`, a working directory a
  `str` or an `os.PathLike`, a stream `None`, one of the constants, a file
  descriptor or a file object, an environment a `dict` of `str`, a timeout
  a number of seconds or `None`. What CPython also takes is refused: a `str`
  command, `shell=True` and the other keywords, `os.environ` itself as the environment (pass `os.environ.copy()`, which
  is the `dict` CPython's returns, as `authoring.uv_tool` does), and
  arguments after the command by position (`run(args, -1)`, which run()
  hands to `Popen` as `bufsize`). Since the check reads `bytes` as `str`, a
  process's output is a `str` with or without `text=True`, and
  `CompletedProcess[bytes]` and `Popen[bytes]` are `[str]`; it is `None`
  when it was not captured, as in CPython, so `stdout` and `stderr` are an
  `Optional`. A stream iterates over its lines, each a `str`
  (`for line in process.stdout:`, after `assert process.stdout is not
  None`). A stream's `write()` takes what a binary stream takes in
  CPython, bytes (a `str`) or a buffer such as an array's `data` (a `Ptr` in
  Codon's numpy), and since the check
  cannot tell a text stream from a binary one, it takes either for a text
  one too, where CPython refuses a buffer. `importlib.metadata` declares
  `version(distribution_name)` and `PackageNotFoundError`, a `ModuleNotFoundError` built from the
  distribution's name; `from importlib import metadata` and `import
  importlib.metadata` compile, and nothing else of `importlib` does (the
  package imports its `metadata`, as Codon 0.19 finds a submodule no other
  way, so the check also takes `importlib.metadata` after a bare `import
  importlib`, which CPython takes only once something has imported the
  submodule). `signal` declares Linux's signal numbers (`SIGHUP`, `SIGINT`,
  `SIGQUIT`, `SIGKILL`, `SIGUSR1`, `SIGUSR2`, `SIGALRM`, `SIGTERM`,
  `SIGCHLD`) as members of `Signals`, CPython's `IntEnum` of them, here a
  class of its own: `name` (`"SIGTERM"`), `value` and `int()` (the number),
  `==`, `!=` and `hash()` with a `Signals` or an `int`, `str()` and `repr()`
  as Python 3.10 writes them, and `Signals(number)`, which looks a member up.
  What takes a signal number (`os.kill()`, `os.killpg()`, `signal()`,
  `Popen.send_signal()`) takes an `int` or a `Signals`, so
  `os.killpg(pgid, signal.SIGTERM)` compiles; the rest of an `int`'s
  operations (arithmetic, ordering) are not declared for a `Signals`, which
  CPython takes: take `int(sig)` first. `SIG_DFL` and `SIG_IGN`, which
  are no `int`s here, since CPython takes no other `int` as a handler; and
  `signal(signalnum, handler)`, whose handler is one of those two or a
  function the check calls as CPython calls a handler, with the signal's
  number (an `int`) and the current frame (`None` in the check, which has no frame
  type: annotate that parameter `Any`), so a function that cannot take them
  is refused where CPython fails only when the signal arrives. What it
  returns, the previous handler, is a value nothing is declared of (a
  `Handlers` member, a function or `None` in CPython, which takes no `None`
  as a handler), so it cannot be handed back to `signal()`; `getsignal()`,
  `alarm()` and the rest of the module are not declared. `shlex` declares
  `quote(s)` and `join(split_command)`, CPython 3.10's on Codon's `re`; a
  `quote()` of a false value that is no `str` (`None`, which CPython quotes
  as `''`) is refused. `sys` replaces Codon's, keeping all of it (`argv`,
  the standard streams, `exit()`, `maxsize`), and adds `platform`
  (`"linux"`) and `executable` (a `str`). `atexit`
  declares `register(func, *args, **kwargs)`, which takes the function by
  position only and returns it, and which calls it with the arguments it is
  given, so a function that cannot take them is refused where CPython
  reports it only at exit, and `unregister(func)`, of one value by position.
  `copy2()` and `move()` return a `str` for a `str`
  destination, as CPython does; for a `Path` one CPython returns the `Path`,
  or a `str` when it names a directory, so the check returns a path that
  `str()`, `os.fspath()` and `Path()` take and nothing else does. Python's
  built-in exceptions that Codon lacks (`ImportError`,
  `ModuleNotFoundError`, `FileNotFoundError`, `PermissionError`,
  `ProcessLookupError`, `TimeoutError`, `ConnectionError`,
  `BrokenPipeError`) are declared in the model's `_exceptions.codon`, each
  derived from its CPython base, and the prelude of every rewritten module
  imports them, so `except ImportError:` compiles as Python reads it. Each
  takes no message or one value by position, as CPython does; a keyword
  (`TimeoutError(message=...)`) and `OSError`'s `(errno, strerror)` and
  their attributes (`errno`, `filename`) are refused. Nothing in these
  shims runs: no environment is read, no file is copied, no process is
  signalled, no signal handler is installed and no exit function is
  registered. `hashlib` declares `sha256(string=b"", *,
  usedforsecurity=True)` and its hash object (`update(data, /)`, `digest()`,
  `hexdigest()`, `copy()`, `name`, `digest_size`, `block_size`); no other
  hash. `str.encode()` takes `(encoding="utf-8", errors="strict")`. Since
  the check reads `bytes` as `str` (below), it cannot tell the two apart:
  it takes `hashlib.sha256("text")`, which CPython refuses (a `str` must be
  encoded first). `bytes.decode()` is not declared, since it would be a
  `str`'s too and pass `"text".decode()`; the check refuses
  `data.decode()`, which CPython takes. `str.removeprefix(prefix, /)` and
  `str.removesuffix(suffix, /)` (Python 3.9) are declared too. `urllib.parse`
  declares `urlsplit()`, `urlunsplit()` and `SplitResult` (its five parts,
  as attributes and as a tuple, `hostname`, `port`, `username`, `password`
  and `geturl()`). A scenario sees the same shims: it may import `pathlib`,
  `hashlib`, `os`, `shutil`, `subprocess`, `importlib.metadata`,
  `urllib.parse`, `signal`, `atexit`, `shlex` and `sys` and call what they
  declare,
  file-system queries and the two writes included (`Path(...).exists()`, `Path(...).write_text(...)`), read
  and set the environment (`os.environ.get()`), copy, move and remove files
  with `shutil`, start and wait for a process with `subprocess.run()` or
  `subprocess.Popen`, signal one with `os.kill()`, install a signal handler
  with `signal.signal()`, register a function to run at exit with
  `atexit.register()`, read an installed
  distribution's version with `importlib.metadata.version()`, and catch the
  exceptions above. Any other name of them (`Path.chmod()`, `hashlib.md5()`,
  `shutil.copytree()`, `subprocess.check_output()`, `os.listdir()`,
  `importlib.import_module()`, `urllib.request`, `signal.getsignal()`) fails
  its check wherever Codon compiles it. The model's top-level package declares `__file__` (a
  `str`, as CPython sets it on a regular package), which
  `authoring.framework_pin` imports (`from .. import __file__`).
  `collections.abc` is not: import `Sequence` and the other
  abstract collections from `typing`. `@abstractmethod` is dropped, since
  Codon 0.19 cannot decorate a method; `ABC` is an empty base, and so is
  `typing.Protocol`, whose `@runtime_checkable` is dropped too.
  `typing.Any` imports, and every annotation naming it is dropped.
  `typing.TYPE_CHECKING` is a compile-time true, so Codon compiles only the
  first branch of `if TYPE_CHECKING: ... else: ...`, and the check reports
  no import in the `else` branch, nor in the body of `if not TYPE_CHECKING:`
  (the same branch, written the other way round): `scenario_queue` imports
  tqdm in an `else` branch, at module level, and waits without the progress
  bar in the first branch. The `logging` shim's
  methods take `exc_info` as a `bool`.
- **pyxodr's road network is the model's**: `codon/pyxodr/` names the
  `RoadNetwork`, `Road`, `LaneSection` and `Lane` that `coordinate.codon`
  declares (what the framework reads of them, down to the XML elements, and
  how `MapManager` loads one: `RoadNetwork(xodr_file_path, resolution)` and
  `get_roads()`), so a checked module annotates with pyxodr's own classes
  under `TYPE_CHECKING` (`utils/traffic_light.py`, which may not import
  `coordinate`; `coordinate/transform.py`) and takes the network
  `MapManager` returns. Nothing else of pyxodr is there. `codon/lxml/` names
  the XML element `coordinate.codon` declares likewise, as
  `lxml.etree._Element`, the type of pyxodr's `lane_xml` and `road_xml`
  (`coordinate.transform._cubic()`; `import lxml.etree` under
  `TYPE_CHECKING`, since Codon does not find `etree` with
  `from lxml import etree`).
  `test_typecheck_model.py` compares the constructor and `get_roads()` with
  pyxodr's and loads the fixture map's OpenDRIVE with them.
- **lanelet2 is modelled**: `codon/lanelet2/` and
  `codon/autoware_lanelet2_extension_python/` declare the classes and
  functions of simple-lanelet2's Python API the checked modules use, with
  its parameter names: `lanelet2.core`'s map, lanelet layer, lanelets, line
  strings, points, attribute maps and regulatory elements (their ids,
  attributes, centrelines, bounds, rule parameters and stop lines),
  `BasicPoint2d`, `BasicPoint3d`, `GPSPoint`, and the lanelets whose bound a
  line string is
  (`LaneletLayer.findUsages()`, of a line string only); `lanelet2.geometry`'s
  `to2D()`, `length()`, `length2d()` (of a lanelet), `toArcCoordinates()`
  (an `ArcCoordinates`), `findNearest()` (of the lanelet layer) and
  `inside()`; `lanelet2.routing.RoutingGraph(laneletMap, trafficRules)` with
  `following()`, `previous()`, `left()`, `right()`, `adjacentLeft()`,
  `adjacentRight()` and `conflicting()` (a list of `ConstLanelet`s, as
  simple-lanelet2 hands them out: lanelet2's own binding wraps each in a
  `ConstLaneletOrArea`, which `route.geometry` still unwraps at run time);
  `lanelet2.traffic_rules.create()`
  with its `Locations` and `Participants`; `lanelet2.io.Origin` and
  `lanelet2.io.load(filename, projector)`; `lanelet2.projection`'s
  `UtmProjector` and `LocalCartesianProjector` (with `forward()` of a
  `GPSPoint`); and Autoware's `MGRSProjector`, `TransverseMercatorProjector`,
  `AutowareTrafficLight` and `RoadMarking`. Each module's header says what
  it leaves out; parts of the API the framework does not use yet (writing a
  map, routes, matching, editing) are not declared, and the check refuses
  them. `MapManager` (`coordinate.map_manager`, checked) and
  `sweeper.map_loader` load the map with them and hand it out as these
  classes, and `MapManager` its `routing_graph` too. Python's
  lanelet2 has no signatures to introspect (it is a compiled extension), so
  `test_typecheck_lanelet2_model.py` checks that every declared name exists in
  the installed package, that its parameters are the ones the package's
  stubs (`lanelet2/*.pyi`) name, and that each declared call, made on the
  fixture map, returns what the model says. What the model has to say about
  lanelet2 that Python does not:
  - **A mutable class derives from its Const twin** (`Lanelet` from
    `ConstLanelet`, `LineString3d` from `ConstLineString3d`, `Point3d` from
    `ConstPoint3d`): in Python they are unrelated classes, but every function
    that takes the Const one takes the mutable one, which has all its
    members. So a layer's `Lanelet` goes where the routing graph takes a
    `ConstLanelet`, and a variable that holds either is annotated with the
    Const class (`lanelet: ConstLanelet = lanelet_map.laneletLayer[...]`, as
    `trajectory/relative_lane.py` walks from a lanelet to the ones that
    follow it). lanelet2's stubs, which mypy reads, keep the two unrelated,
    so a function that takes either spells the union out
    (`Union[ConstLanelet, Lanelet]`), which the check drops, as it drops any
    union parameter. A member the mutable class narrows (`Lanelet.leftBound`
    and `TrafficLight.stopLine` are `LineString3d`s) is declared with the
    Const type, so the check reads less than Python hands out, never more.
    The projectors likewise derive from lanelet2's `Projector`, as they do in
    C++: Python registers none of them as derived from it, but
    `lanelet2.io.load()` takes any, so the projector
    `coordinate.projection.resolve_projector()` hands out (one of four) is a
    `Projector` to the check, which `load()` takes.
    (Codon 0.19 does not convert an `Optional` of a subclass to its base in
    one step, `Optional[LineString3d]` into a `list[ConstLineString3d]`:
    that would take an annotated local of the subclass first.)
  - **An overloaded function dispatches on its argument**: Codon 0.19 cannot
    overload a module-level function, so `to2D()` and `length()` are generic
    and declare the overloads the framework calls (of a line string); any
    other argument is a compile error. Codon decides `isinstance` by the
    static type, and `isinstance(Lanelet, ConstLanelet)` is false there as
    in Python, so each overload names both twins.
  - **A regulatory element is its own class**: lanelet2 hands each one out as
    the class of what it is (`TrafficLight`, `RightOfWay`, `TrafficSign`, or
    Autoware's), all derived from `RegulatoryElement`, as the model declares
    them, and a lanelet's `regulatoryElements` is a list of the base. Codon
    decides `hasattr(element, "stopLine")` by that static type, so the
    stop-line branch of code that duck-types the element would never be
    compiled. `utils/stop_line.py` therefore reads one element in a public
    function whose parameter is annotated with the union of the classes it
    reads (`Union[TrafficLight, AutowareTrafficLight, RightOfWay,
    TrafficSign, RoadMarking, RegulatoryElement]`), which the library check
    calls once per member, so each branch is compiled for each class that
    takes it. At run time it still asks the element, as before.
  - **A rule parameter is declared by its role**: lanelet2 holds a point, a
    line string, a polygon, a lanelet or an area in a regulatory element's
    `parameters[role]`, and Codon has no type for "one of five classes".
    `ref_line`, `cancel_line` and `light_bulbs` hold line strings
    (`ConstLineString3d`), `right_of_way` and `yield` lanelets
    (`ConstLanelet`); any other role is a compile error. `refers` holds a
    different class per element (line strings or polygons for a traffic
    light, polygons for Autoware's detection areas, lanelets for its
    crosswalks), so the model declares it as a class of its own with only
    what all five have, `id` and `attributes`. The stop line a road marking
    refers to is therefore not checked: `utils/stop_line.py` reads it in a
    function called in the `else` branch of `if TYPE_CHECKING:` (`UNCALLED`).
    `test_typecheck_lanelet2_model.py` reads the roles off the model and
    checks each against what the fixture map's elements hold.
- **carla_driver_interface's geometry is modelled**:
  `codon/carla_driver_interface/geometry.codon` declares the `Pose` and
  `Trajectory` the driver modules use (and `waypoints_to_proto`, which
  `driver.geometry` re-exports, with its protobuf result left untyped), with
  the package's fields, signatures and types, the arrays as Codon's
  (`position: np.ndarray[np.float64, 1]`). Its scipy-typed constructor
  (`from_rotation`) and the protobuf conversions (`from_proto`, `to_proto`,
  `to_proto_at_time`) are not there. `test_typecheck_model.py` compares the
  model with the installed package. Like every top-level directory of
  `codon/` (`pyxodr`, `pathlib`, ...), it is a model of a module that is not
  the framework's: `_shims()` lists it, so the library check and the scenario
  check let `import carla_driver_interface...` through, and Codon compiles
  the import against the model. `MODELLED_IMPORTS`, which redirects one
  framework module's import of another, does not concern it. A scenario may
  therefore use `Pose` and `Trajectory` of `carla_driver_interface.geometry`
  now; a name the model lacks fails its check (`Pose.to_proto()` has no
  attribute), and so does any other module of the package (`no module named
  'carla_driver_interface.protocol'`).
- **python-dotenv is modelled** as far as `server` uses it, with the
  installed package's signatures, which `test_typecheck_model.py` compares:
  `codon/dotenv/` declares
  `load_dotenv(dotenv_path=None, stream=None, verbose=False, override=False,
  interpolate=True, encoding="utf-8")`, of a path that is `None`, a `str` or
  an `os.PathLike` (a stream, an `IO[str]` Codon has no type for, is not
  taken). Nothing else of the package is declared (`dotenv_values()`), and
  a scenario may use what is, as it may the shims.
- **No `bytes`**: Codon 0.19 has none, and reads a `b"..."` literal as a
  `str`, so the check reads an annotation naming `bytes` as `str`
  (`DriveOutcome.debug_info: bytes = b""`). What a process writes is a `str`
  in the check, which has no `decode()`: decode it in the `else` branch of
  `if TYPE_CHECKING:` and take the `str` in the first.
- **`dataclasses.fields()` is not in the shim**: a function that reads a
  dataclass's fields from its class takes a free-form mapping anyway, so it
  is `UNCALLED` and imports `fields` inside it (`ControlConfig.from_mapping`).
- **Codon's numpy appends an array of the same dimensions**:
  `np.append(yaws, yaws[-1:])`, not `np.append(yaws, yaws[-1])`, which numpy
  answers the same. Nor does Codon unpack an array into arguments
  (`f(*row)`): pass `row[0], row[1], row[2]`.
- **numpy is Codon's own**, so `import numpy as np` compiles
  (`utils/powertrain.py`). Codon compares no `np.float32` with a Python
  `float`: convert the float (`x <= np.float32(max_rpm)`), which is what
  numpy 2 does anyway (NEP 50), so nothing changes at run time. Codon's
  `ndarray` needs its dtype and dimension count, so a parameter annotated
  with a bare `np.ndarray` or numpy.typing's `NDArray[...]` does not
  translate; numpy's shape-typed array does (see *An array is typed by its
  shape* below). A bare `np.ndarray` parameter compiles while only its
  callers reach it, but the check has no value of it to call with
  ("cannot typecheck '_reach_m(_acs_value(np.ndarray))'"): the protected
  geometry helpers of `coordinate.transform` and `driver.control` take
  `np.ndarray[tuple[int, int], np.dtype[np.float64]]` and their points
  `np.ndarray[tuple[int], np.dtype[np.float64]]`.
- **An overridden method takes concrete types**: Codon 0.19 cannot call a
  method a subclass overrides when one of its parameters is generic, which an
  abstract collection (`Sequence[X]`) or an unannotated parameter is.
  `AutowareBridge.configure()` takes `waypoints: Optional[list[BridgePose]]
  = None` in the base class and in the checked `FakeAutowareBridge`, and its
  caller passes a list. (`None`, not `[]`: a mutable default is shared by
  every call.)
- **A call to an overridden method passes every argument, by position, of
  the parameter's own type**: Codon 0.19 calls it through a dispatch thunk,
  which fills in no default, takes no keyword argument and no `**kwargs`,
  and does not upcast. Hand a subclass over through a local of the
  parameter's type (`entity: DrivenEntity = self`, as `traffic/driven.py`
  does). A thunk cannot call a method that takes `**kwargs` at all:
  `BackendDriven.turn_at_junction()`, which `actions.turn` calls, hands the
  backend its `**kwargs` only in the `else` branch of `if TYPE_CHECKING:`
  (see below), and the check calls each backend's `turn_at_junction()`
  through its class instead.
- **A list of a subclass is not a list of its base**: Codon 0.19 refuses a
  `list[EntityLanePositionCondition]` where a `list[BaseCondition]` is
  expected, and `extend()` does not upcast either. Build the list in a local
  declared `list[BaseCondition]`, appending each element
  (`composition/temporary_stop.py`). An element of a `Sequence[AnyPose]` is
  a `Union` of poses; the model's `_expect_pose` takes one, reading its
  members with `static.vars_types(pose, with_index=True)` (without
  `with_index`, Codon 0.19 does not bind the loop variable).
- **A class cannot be named before it is defined or imported**, in a
  signature or a class-level declaration, and an import cycle resolves only
  with the names already defined. Two classes whose signatures name each
  other across modules (a backend's methods take the entity, the entity holds
  the backend) need a base class the first module defines: the backends take
  a `traffic.base.DrivenEntity`, which `traffic.driven.BackendDriven` derives
  from.
- **A part of a module that needs what Codon lacks moves out**: the
  backend interface of `traffic/base.py` is checked, and `TrafficContext`,
  which holds `pathlib.Path`s, lives in `traffic/context.py` (re-exported by
  `base`), which is checked on its own with the `pathlib` shim.
- **A `carla.Actor` where the CARLA API takes a `carla.Vehicle`**
  (`TrafficManager.set_desired_speed()`): typesafe_carla's Python API takes
  the actor as it is, and its Codon library wants the checked conversion.
  Convert under `if TYPE_CHECKING:`, which Codon reads as true and Python as
  false, so nothing changes at run time (`traffic_manager._vehicle()`):

  ```python
  if TYPE_CHECKING:
      return actor.as_vehicle()
  else:
      return actor
  ```
- **An import of the framework inside an uncalled function** is left as
  written: Codon does not compile it, so a checked module may import a
  module with no model there (`traffic.config` imports `utils.config` in
  `from_mapping()`).
- **A CARLA container has no truth value in Codon**: `if not found:` on a
  `carla.BlueprintLibrary` is `if len(found) == 0:`.
- **No `**` in a dict display** (`{"a": 1, **other}`): Codon's parser fails on
  it, even in a function it never compiles. Build the dict, then
  `update()` it.
- **`dict.get(key, default)` takes a default of the value type** in Codon.
  Where "absent" has no value of that type, take `d.get(k)`, an `Optional`
  of the value type, as the check declares it with Python's meaning.
- **CARLA's containers are not lists.** `world.get_actors()` is a
  `carla.ActorList`; a parameter that takes it or a list is
  `Union[carla.ActorList, list[carla.Actor]]`, and the check calls it with
  each.
- **Codon cannot hash a class**: `hash((MyClass, x))` names the class by its
  name instead.
- **An attribute holds one type**: one assigned a union (`EntityRole | str`)
  is stored converted where every use converts it anyway
  (`self._entity_name = str(entity_name)`). Codon 0.19 can assign nothing
  to an attribute declared with an optional union (`Optional[Union[EntityRole,
  str]]`): neither a member nor `None` matches it ("'Optional[EntityRole]'
  does not match expected type 'Optional[Union[EntityRole | str]]'"), and
  `Union[EntityRole, str, None]` takes no `None`. A public dataclass field
  is converted in `__post_init__` (`VehicleEntityConfig.role_name: str`,
  `self.role_name = str(self.role_name)`), which still takes an `EntityRole`
  at run time; or
  by a constructor written out, which keeps the union on the parameter for
  mypy (`RelativeLanePose(entity_ref: Union[EntityRole, str, None])`, stored
  as `entity_ref: Optional[str]`). Without `None` the union is held as it
  is: `RoutingAction._entity_name: Union[EntityRole, str]` keeps what it was
  given, which its public `entity_name` returns.
- **A union of classes without `None` is an attribute type**: Codon 0.19
  holds `Union[Lanelet2Pose, MapPose, ...]` (`TrajectoryVertex.position`, any
  of ten poses), declared whole or through an alias (`position:
  TrajectoryPosition`), and decides `isinstance` on it when the program
  runs. A union is no parameter type: Codon 0.19 crashes on one ("union
  already sealed"), so the check drops it and the library check calls the
  function once per member. Spell the parameter's union out, not through an
  alias, which the library check would call with the union itself and so
  catch less. A module-level alias whose members are strings
  (`Union["Lanelet2Pose", MapPose]`, for names imported under
  `TYPE_CHECKING`) has them unquoted by the check.
- **Codon does not run `__post_init__`**: its constructor of a class with
  declared fields only assigns them. The check calls `__post_init__` itself,
  as a dunder method, so its body is checked. A `field(init=False)` set there
  is given an empty value of its type (`list[float]()`) by the check, so the
  constructor leaves it out as Python's does.
- **A dataclass field may hold a union or a free-form mapping**: the library
  check constructs a dataclass once per member of a union field
  (`ScenarioRunConfig.scenario`, one of six scenario configs), as it calls a
  function once per member of a union parameter. A field with a default whose
  annotation names `Any` (`SweepConfig.constraints: dict[str, Any]`, a Hydra
  mapping of any values) keeps its declaration, its empty default is built as
  that type (`dict[str, Any]()`), and the check's constructor call leaves it
  out, passing the fields after it by keyword. A `default_factory=list` on a
  `List[int]` field is built as a `List[int]` as on a `list[int]` one, and a
  `list`, `dict` or `set` factory of an abstract collection as the one the
  `typing` shim maps it onto (`env: Mapping[str, str] =
  field(default_factory=dict)` as a `Dict[str, str]`, #188).
- **`object.__setattr__(self, "name", value)`**, as a frozen dataclass sets
  its fields, is checked as `self.name = value`: Codon 0.19 has no
  `object.__setattr__`.
- **No `**kwargs` iterated as a dict**: Codon's `**kwargs` is a named tuple,
  with no `items()`. Pass the names and values as parameters
  (`_finite(owner, "ds", self.ds)`), one call each.
- **A run-time-only wrapper of a method**: Codon 0.19 cannot decorate a
  method, and a decorator `UNCALLED` names (a function of the same module) is
  dropped by the library check, which compiles the method as written.
  `TrajectoryVertex.__init__` is typed `(position, advance=None)`;
  `_refusing_vertex_times` wraps it at run time to refuse the calls written
  for a vertex with a time, with `inspect` and `functools` imported inside
  it, where Codon never compiles them. The wrapper is typed with a
  `TypeVar` (`(init: _Init) -> _Init`), so mypy still sees the typed
  `__init__`; the check drops an annotation that names a module-level
  `TypeVar`.
- **A class chosen by a conditional expression** (`type(entity) if entity is
  not None else self.ego_type`) does not compile: Codon cannot hold a class
  as a value ("'type[Optional[EgoVehicle]]' does not match expected type
  'EgoVehicle'"). Read a class attribute off each in its own branch, the
  instance or the class (`BaseScenario.ego_requires_goal`, which reads
  `requires_goal`, a class attribute no instance assigns).
- **A call to a `register_*()` method that takes a `Union`** (`register_init`,
  `Union[BaseAction, Callable[[carla.World], None]]`) compiles both
  `isinstance` branches of its dispatch with the caller's type. Where the
  caller has an action, do in the `if TYPE_CHECKING:` branch what the method
  does for one, and call it in the `else` branch
  (`BaseScenario.register_route_to_goal`).
- **`isinstance(x, (A, *OTHERS))`** does not compile: Codon 0.19 cannot
  unpack a tuple of classes into `isinstance`. Spell the classes out
  (`Trajectory.is_relative`).
- **`callable(x)`** is checked as `hasattr(x, "__call__")`, which Codon
  decides when it compiles: Codon 0.19 has no `callable`.
- **Annotate a local that starts as `None`** (`spawn_points:
  Optional[list[carla.Transform]] = None`): Codon types it by its first
  assignment, and on a branch it then compiles alone it can crash without a
  diagnostic. A function that imports a name locally (`import
  typesafe_carla.carla as carla`) makes it local to the whole function, so an
  annotation naming it needs that import above it.
- **Operators take the operand types they accept**, not `object`. Annotate
  `other` with the class (`def __add__(self, other: FrenetVelocity)`) and a
  scalar with `float`, and keep the `isinstance` guard that returns
  `NotImplemented`: Python still runs it, and Codon, which decides
  `isinstance` when it compiles, drops it.
- **An operator whose result depends on the operand** (`Absolute - Absolute
  -> Relative`, `Absolute - Relative -> Absolute`) keeps its `@overload`
  stubs, for mypy, and spells the union out on the implementation:

  ```python
  @overload
  def __sub__(self, other: AbsoluteVelocity) -> RelativeVelocity: ...
  @overload
  def __sub__(self, other: RelativeVelocity) -> AbsoluteVelocity: ...
  def __sub__(
      self, other: AbsoluteVelocity | RelativeVelocity
  ) -> Union[RelativeVelocity, AbsoluteVelocity]:
      if isinstance(other, AbsoluteVelocity):
          return RelativeVelocity(...)
      if isinstance(other, RelativeVelocity):
          return AbsoluteVelocity(...)
      return NotImplemented
  ```

  The check calls the implementation once per member, and Codon compiles
  only the `isinstance` branch of that member, so each call has one result
  type; the `Union` return annotation is dropped. Two classes whose methods
  take each other are fine: an annotation naming the later one is dropped
  from the earlier one's signatures, and the check still calls with it.
- **Two checked modules that import each other**: Codon reads
  `TYPE_CHECKING` as true, so an import made only for annotations is still
  an import cycle, and Codon resolves a cycle only with the names already
  defined. Put that `if TYPE_CHECKING:` import at the end of its module
  (`# noqa: E402`), after the names the other module imports, as
  `coordinate/frames.py` does for `poses.py`. Set class attributes in the
  class body (`FRAME: ClassVar[CoordinateFrame] = CoordinateFrame.LANELET2`)
  rather than from a function run after the class.
- **Codon does not narrow an `Optional`** after `if x is None: return`. Where
  the narrowed value goes into something typed (a tuple the function
  returns), assign it to an annotated local first (`od: OpenDrivePose = x`),
  which Codon unwraps. Likewise, a function returning
  `tuple[Optional[Path], bool]` returns a `Path` it found through an
  `Optional[Path]` local (`shipped: Optional[Path] = candidate`, as
  `maps.resolver._find_xodr()` does): Codon types a tuple by its members,
  and a bare annotation without a value (`found: Optional[Path]`) does not
  change the type of what is assigned to it later. An attribute is not
  narrowed either: after `if self._spawn_pose is None: raise ...`,
  `self._spawn_pose` is still an `Optional[Lanelet2Pose]`, and
  `snap_to_carla_road()`'s `isinstance` dispatch takes its CARLA branch for
  it ("'Lanelet2Pose' does not match expected type 'CarlaWorldPose'").
  `BaseScenario._setup_ego_spawn()` reads it into
  `ll2_pose: Lanelet2Pose = self._spawn_pose` first.
- **Codon does not narrow a union either**, with `is` or `isinstance`, and an
  attribute holding a union (`Union[Sequence[int], TrafficLightTarget,
  None]`) reaches every use as the union. Store it split by kind
  (`_all_lights: bool`, `_lanelet2_ids: Optional[Sequence[int]]`), deciding
  the kind once in `__init__`, where `isinstance` on the parameter is decided
  when Codon compiles.
- **A default of a subclass for an `Optional` of its base** goes through an
  `if` statement, not a conditional expression: `x if x is not None else
  Sub()` is refused (`Optional[Base]` is not `Optional[Sub]`).
- **A conditional expression of two types** (`log if log is not None else
  subprocess.DEVNULL`, a file or an `int`) is typed by its first branch.
  Neither can an `if` statement assign the two to one name: call a function
  once with each (`autoware_stack.launcher._start_session()`, whose
  `Union[IO[bytes], int]` parameter the checker drops, so Codon compiles it
  for each).
- **A dict has no truth value** in Codon: `if not d:` is `if len(d) == 0:`.
- **No `setattr(obj, name, value)` with a name known only at run time**:
  spell the assignments out (`actions/environment.py`).
- **No attribute on a function**: a Codon function has none, so neither
  `setattr(probe, "on_lanelet", read)` in a decorator nor
  `probe.on_lanelet = read` compiles. `odd.probes` defines each lanelet
  reader as a plain function and attaches it at the end of the module, in
  the body of `if not TYPE_CHECKING:`, which Python alone runs: the
  probes have the same `on_lanelet` attribute as before, and the check
  compiles each reader on its own, as it does every function of the module.
- **A union a function returns goes through a module-level alias**: the
  check drops a `Union` written out in any annotation (Codon 0.19 crashes on
  one in a parameter), a local's and a return's included, and Codon then
  types the function by its first `return`. An alias is kept, and Codon
  converts each `return` to it: `odd.probes._speed_limit_on_lanelet()`
  returns `_SpeedLimitOnLanelet = Union[float, "Undecided"]`, a number or
  `UNDECIDED`. Codon 0.19 converts a value of any other type too, failing
  only when the program runs (`internal.new_union`), so the check compiles
  the body but does not check which member each `return` is.
- **An exception class spells out its constructor**: Codon 0.19 cannot
  raise a subclass of a built-in exception that has none of its own
  (`'MapSourceError' object has no attribute '_pytype'`), so it takes the
  message and passes it on: `def __init__(self, message: str) -> None:
  super().__init__(message)` (`maps/source.py`).
- **`except (A, B):` is split by the check**: Codon 0.19 takes one class
  per `except` clause ("expected type expression"), so the check rewrites a
  tuple written out into one clause per class, in the tuple's order, each
  with the same body (and the same `as` name), where the tuple's clause was:
  Python tries a tuple's classes in order, so the same exceptions reach the
  same body and the others the clauses after it. The copies are lines the
  source does not have; an error in one is reported at the line it copies,
  and the lines after them where they are. A tuple that is not written out
  (`except errors:`, `except (*errors,):`) is left as it is, and Codon
  refuses it. `raise X from None` is rewritten to `raise X` by the check;
  `raise X from exc` compiles as it is. A builtin exception class raised
  as it is (`raise NotImplementedError`, an abstract stub) is rewritten to
  an instance (`raise NotImplementedError()`), which Python raises anyway:
  Codon 0.19 crashes on the class.
- **`frozenset` is checked as a `set`**: Codon 0.19 has none, so the check
  rewrites `frozenset(...)` to `set(...)`, `frozenset[X](...)` to
  `Set[X](...)` and `frozenset[X]` / `FrozenSet[X]` to `Set[X]`; the source
  keeps its `frozenset`. An empty one is `frozenset[str]()`: a bare
  `frozenset()` is a set of no element type in Codon
  (`scenario_runner._destroy_all_dynamic_actors()`).
- **A variable-length tuple attribute is checked as a list**: a class-level
  `tuple[X, ...]` / `Tuple[X, ...]` becomes `List[X]`, and `tuple(x)` of a
  list or a generator becomes a list (a tuple passed to `tuple()` stays
  itself), so `self.vertices = tuple(vertices)` compiles; the source keeps
  its tuple. A dataclass field's `()` default, written directly or as
  `field(default=())`, is an empty `List[X]` (`Pin.warnings`,
  `PlannedRoute.via`), for a `Sequence[X]` field as for a `tuple[X, ...]`
  one, and its tuple display default (`("a", "b")`) a list of its elements,
  `List[X](["a", "b"])` (`DockerAutowareConfig.colcon_args`, #188). Any
  other tuple display stays a fixed-length Codon tuple, which is no
  `List[X]`: pass `tuple([...])` of a list instead (`ros2.config`'s
  `_default_sources()`; `CommandSourceConfig.topics`, which returns a one-
  or a two-element tuple, likewise, so its two returns have one type). A
  parameter `tuple[X, ...]` is dropped (a caller may pass a tuple), and the
  library check calls it with a `List[X]`. Its `()` default, and that of a
  `Sequence[X]`, `MutableSequence[X]`, `Iterable[X]` or `Collection[X]`
  parameter, is an empty `List[X]` too (`RoutingAction(..., waypoints:
  Sequence[Lanelet2Pose] = ())`), and its tuple display default a list of
  its elements, as a field's; see *Defaults are compiled too* above.
  `Phase.states` and `SignalController.phases` (`signals.controller`) are
  checked this way. What only a tuple does is not checked: its hash (a
  frozen dataclass's `__hash__`), its equality with another tuple and its
  immutability; nothing compiled is run, so nothing can mutate the list.
  Of a union of classes (`tuple[A | B, ...]`, `tuple["A | B", ...]`) it
  becomes `List[Union[A, B]]`, which Codon 0.19 holds as it holds an
  attribute of the union (above): `route.model`'s
  `RouteSearchSpec.segments`, each a lane or a junction segment. A union with
  `None` in it is not translated.
- **No star-unpacking into a tuple display** (`(*a, *b)`): Codon wants a
  tuple after `*`. Concatenate the lists (`a + b`).
- **No `*` of a list in a call** (`f("rm", *ids)` with `ids` a list):
  Codon 0.19 unpacks only a tuple into arguments ("argument after * must be
  a tuple"). Give the function a twin that takes the list
  (`DockerAutowareLauncher._run_docker(["rm", "--force"] + ids, ...)`, which
  `_docker(*args)` calls with `list(args)`).
- **An empty dict display for a `Mapping` parameter** (`extra_env={}`) is a
  dict of no element type: the check drops the abstract `Mapping[str, str]`
  from a parameter, so nothing gives it one. A private function that is
  handed dicts takes a `dict[str, str]` (`DockerAutowareLauncher._run_argv()`).
- **A default does not name a class attribute** (`random_seed: int =
  DEFAULT_RANDOM_SEED` in a method of the class defining it): Python
  evaluates it once, when the class is created, and Codon cannot name it
  there. Write the value, with a comment naming the attribute.
- **No `del` of a parameter** (`del world, goal` to mark it unused): Codon
  reads it as a call on the value. Leave the parameter unused.
- **An `Optional` attribute is assigned without an annotation**: with one
  declared at class level (`_lane_id: Optional[int]`), Codon refuses
  `self._lane_id: Optional[int] = address.lane_id` (an `int` "does not match"
  `Optional[int]`) but takes `self._lane_id = address.lane_id`, and
  `X(...) if c else None` likewise only without the annotation.
- **`x or []` with `x: Optional[list[T]]`** has no one type in Codon: start
  from an annotated empty list and assign `x` under `if x:`. Likewise
  `stated or DEFAULT` with `stated: Optional[str]`: branch on the cases
  into an annotated `str` local (`coordinate.projection.resolve_projector`).
- **A regex group used as a `str`** (`m.group(1)` handed to `re.search`, or
  to a `str` local): Codon's `Match.group()` is an `Optional[str]` whose
  type Codon 0.19 cannot settle from such a use. Take `str(m.group(1))`,
  the same `str` in Python (`coordinate.projection._parse_geo_reference`).
- **Projectors of different classes return through one `-> Projector`**
  (`coordinate.projection.make_projector`): the lanelet2 model derives each
  from `Projector`, and Codon 0.19 converts a subclass to it at `return`.
- **A subclass goes into an `Optional` base parameter from a base-typed
  variable**: Codon does not upcast and wrap in one step, so
  `super().__init__(child=PersistentCondition(...))` against
  `child: BaseCondition | None` fails, and
  `child: BaseCondition = PersistentCondition(...)` passed on compiles.
- **`getattr(x, name, None)`** for an attribute a value may lack at run time
  (a mock in a test) becomes `try: v = x.name` / `except AttributeError:`,
  which reads the attribute once, as `getattr` did.
- **`hasattr` on an `Optional` is decided on the `Optional`**, which has
  none of its value's attributes: assign the value to an annotated local
  after the `None` test (`walker: PedestrianEntity = entity`) and ask that.
- **An `Optional` local tested in a loop** (`assert x is not None`, or `if x
  is None: return`) is left the unwrapped type for the next lap, whose
  assignment of the `Optional` then fails ("'BaseCondition' does not match
  expected type 'Optional[BaseCondition]'"). Read the value through a function
  that returns it unwrapped (`follow_trajectory._gate_at()`), and test the
  `Optional` expression itself where the loop needs the test.
- **A call into a module Codon cannot compile, from a function it must**: a
  function `UNCALLED` names is still compiled when a checked function calls
  it. Call it in the `else` branch of `if TYPE_CHECKING:` and check in the
  first branch what it can (`ScenarioRunner` calls `create_ego()`, built
  on the ego's class, in the `else` branch). A
  private helper that imports such a module inside it needs an `UNCALLED`
  entry of its own: its import is rewritten, and refused, even when nothing
  compiled calls it.
- **A type alias imported from another module, inside an alias of this one**
  (`Received = Tuple[str, Command]`, with `Command` a `Union` alias of
  `ros2.commands`) is "not defined" to Codon 0.19, though the imported alias
  works as an annotation. Spell the union out in the alias
  (`ros2.transport.Received`; `test_ros2_commands.py` checks the two are
  equal).
- **An entity looked up by role is looked up by kind**:
  `find_entity_by_role_name` returns an entity of any kind (`Optional[Any]`),
  so a checked module asks `entity.registry` for the kind it acts on,
  `find_vehicle_entity` (a `BackendDriven`: an NPC vehicle or the ego) or
  `find_pedestrian_entity`. An object of neither kind (a test double) is
  registered as both, so at run time every lookup still finds it; keep a
  `hasattr` test where the code relied on `getattr(entity, name, None)`.
  Where the code called a method the other kind lacks, look the other kind
  up when the lookup misses and raise the `AttributeError` the call raised
  (`LaneChangeAction`, `TurnAction` and `SetSpeedAction` on a pedestrian),
  so a wrong kind still fails rather than reading as an entity not found.
- **An entity is registered by kind, too**: `register_entity` picks the kind
  with `isinstance(entity, BackendDriven)`, which Codon decides by the
  static type and does not match for a subclass (a `VehicleEntity`), so a
  checked caller calls `register_vehicle_entity` (through a `BackendDriven`
  local, which upcasts) or `register_pedestrian_entity`. `BaseScenario`
  calls them under an `isinstance` test on its parameter's own class, which
  Codon decides as true, and hands anything else (a test double) to
  `register_entity`, so at run time it is registered as before.
- **No sign in an f-string format spec**: Codon's parser rejects
  `f"{x:+.1f}"`. Format the value first (`x.__format__("+.1f")`) and put the
  string in the f-string.
- **`carla.TrafficLightState` is a value in the check**, not a type: its
  members are ints, as `get_state()` returns. The check reads an annotation
  naming one of typesafe_carla's integer enumerations
  (`carla.TrafficLightState`, `carla.LaneType`, ...) as `int`, so keep the
  CARLA name (the authoring code generator reads it), and compare with
  `get_state()` rather than look a member up by value. A member's name is
  found through the enumeration's `names` (`for name, member in
  carla.TrafficLightState.names.items()`), which both APIs have, as
  `utils/traffic_light.py` does.
- **A sensor callback gets a `carla.SensorData` in the check**, where the
  CPython API hands it the measurement itself (a `carla.CollisionEvent`).
  Listen through a method that converts only when it has to
  (`if hasattr(data, "as_collision"): data = data.as_collision()`, as
  `conditions/collision.py` does): Codon decides `hasattr` when it compiles,
  and the CPython measurement has no `as_<kind>()`.
  The camera and the LiDAR (`sensor/`) listen the same way, with
  `as_image()` and `as_lidar()`. A bound method of another object (a
  queue's `put`) is no listener: wrap it in a method of the sensor.
- **An array is typed by its shape**: Codon's numpy types an array by its
  dtype and its number of dimensions (`np.ndarray[np.float32, 2]`), which
  numpy.typing's `NDArray[np.float32]` does not say, and Codon has no
  `numpy.typing`. Annotate with numpy's shape-typed array instead,
  `np.ndarray[tuple[int, int], np.dtype[np.float32]]` (two dimensions),
  which mypy reads as precisely and the check translates; a shape of
  `tuple[int, ...]` or `Any`, and `NDArray`, are dropped. An annotation
  broken over lines keeps its line breaks inside its brackets, so
  `newest: Optional[\n    ...\n] = None` still compiles.
- **A measurement's `raw_data` is a view** in typesafe_carla's Codon library,
  and Codon's `np.frombuffer` takes the bytes: read
  `np.frombuffer(image.raw_data.tobytes(), ...)` under `if TYPE_CHECKING:`
  and the CPython `raw_data` in the `else` branch, which copies nothing.
- **`queue.Queue[T]`** is the `queue` shim's, with CPython's signatures
  (`Queue(maxsize=0)`, `put(item, block=True, timeout=None)`, `get(block=True,
  timeout=None)`, `put_nowait`, `get_nowait`, `empty`, `full`, `qsize`, and
  the `Empty` and `Full` exceptions). Declare the queue's item type at class
  level (`_frame_queue: queue.Queue[carla.Image]`); `queue.Queue()` in
  `__init__` takes it from there.
- **`threading.Thread`** is the `threading` shim's, with CPython 3.10's
  signature (`Thread(group=None, target=None, name=None, args=(),
  kwargs=None, *, daemon=None)`, `start()`, `join(timeout=None)`,
  `is_alive()`); the check compiles the call the thread makes,
  `target(*args)`, so a target that does not take its arguments fails it.
  `args` is a tuple, `kwargs` None and `group` None. The shim replaces the
  `threading` module Codon ships, which Codon's numpy and typesafe_carla's
  Codon library import for `Lock`, so it keeps all of that module
  (`Lock`, `RLock`, `active_count()`, `get_native_id()`, `get_ident()`), as
  Codon has it; a test compares the two.
- **A datetime is formatted from its fields**: Codon's `datetime` has no
  `__format__`, so `f"{now:%Y-%m-%d}"` is
  `f"{now.year:04d}-{now.month:02d}-{now.day:02d}"`, the same string for
  every year from 1000 on (`ui.runner._multirun_dir()`).
- **A dataclass derived from a dataclass** is constructed with its base's
  fields first, as Python's constructor and Codon's take them, wherever the
  base is checked (`CarlaCameraSensorConfig(CameraSensorConfig)`). Of two
  bases that both bring fields, Python takes the last one's first and Codon
  0.19 the first one's, so such a constructor is a problem: list it as
  `UNCALLED`.
- **No variable-length tuples beyond attributes**: Codon's tuples have a
  fixed length. A class-level `tuple[X, ...]` and `tuple(x)` are checked as
  lists (above), but a tuple built otherwise (a display grown in a loop,
  `a + (b,)`) is not, and turning the source's tuple into a list changes what
  the class does (a frozen dataclass's hash, its equality with a tuple). A
  function that returns one is checked as returning a `List[X]` (#191):
  return `tuple(x)` of a list (`signals.registry.registered_signal_controllers()`),
  not a tuple display, which is a fixed-length Codon tuple, nor `()`.
- **No `**` of a dict in a call** (`f(**extra)` with `extra` a `dict`):
  Codon unpacks only a named tuple into keyword arguments. Write one call
  per set of keywords (`RoutingAction` passes `waypoints=` only when there
  are some, in a call of its own).
- **No escaped brace in an f-string** (`f"{{min, max}} of {name}"`): Codon's
  parser rejects it, even in a function it never compiles. Concatenate the
  literal part (`"{min, max}" + f" of {name}"`, as `Range.parse` does).
- **`Literal["a", "b"]` is checked as the type of its values** (`str`), an
  annotation and a module-level alias alike (`route.model`'s `Tristate`):
  Codon's own `Literal` types a compile-time value of `int`, `bool` or `str`
  only, and the `typing` shim's `Literal` is `str` (its `TYPE_CHECKING` is
  declared with `Static[bool]`, Codon's other name for its `Literal`). A
  `Literal` inside a `Union`/`Optional` alias is translated the same way. A
  `Literal` of values of several types is dropped as an annotation; as an
  alias it is left as written, which the check rejects.
- **No `getattr(obj, name)` with a name built at run time**: spell the
  attributes out (`JunctionSegmentSpec.crossing()`), and keep the `getattr`
  for any other name in the `else` branch of `if TYPE_CHECKING:`, so Python
  still raises the `AttributeError` it raised.
- **A method only some subclasses have** (`EgoVehicle.route_to()`, which an
  entity found as a `BackendDriven` may or may not have): ask
  `hasattr(entity, "route_to")` and call it under that, and keep the call
  that raises the `AttributeError` for an entity without it in the `else`
  branch of `if TYPE_CHECKING:`. Codon decides `hasattr` by the static type,
  which lacks the method, so it compiles none of the calls: the check covers
  the lookup around them, not the calls (`RoutingAction.execute()`).
- **lanelet2's map at the model boundary**: the checked `MapManager` hands
  out its `lanelet_map` and `routing_graph` as the lanelet2 model's classes
  (above), which the checked `RouteFrame` (`route.frame`) takes, so the
  checked `scenario_route_frame()` (`route.active`) builds it from them.
- **A model shared with checked modules is split out**: the model's
  `EgoConfig` derives from `VehicleEntityConfig`, and the checked
  `EgoVehicle.spawn()` hands its spawn location to the checked
  `spawn_vehicle_actor()`. So the spawn locations and the NPC vehicle live
  in a model module of their own (`_vehicle_entity.codon`), which
  `REPLACED_MODELS` replaces with `entity._spawn` and
  `entity.vehicle_entity`: the model's `EgoConfig` then holds the checked
  `SpawnLocation`. The model of `maps.cache` (`maps/cache.codon`,
  subprocess and json) takes and returns a `MapSource` from a model module
  of its own (`_map_source.codon`), which `REPLACED_MODELS` replaces with the
  checked `maps.source`, so the checked `maps.resolver` hands the model's
  `GitMapCache` its own `MapSource`. Likewise `GroundProjectionConfig` (`_ground_projection.codon`,
  replaced with `coordinate.snap`): the example scenarios import it from the
  top-level package, which is the model, and hand it both to the model's
  `BaseScenario` and to the checked `VehicleEntityConfig` (#99). The
  top-level package model itself is not replaced: the examples derive from
  the model's `BaseScenario`, so their constructors' calls to
  `super().__init__()` compile the model's constructor. The checked
  `BaseScenario.__init__` is compiled by the library check's own call, with
  its defaults; the one line that stores the ego's class (`ego_type`) is
  under `if not TYPE_CHECKING:` (#199). The model of
  `ros2.cyclone` (`ros2/cyclone.codon`, cyclonedds) declares the
  `CycloneRos2Transport` constructor that the checked
  `entity.ros2_entity._cyclone_transport()` calls; its base `Ros2EgoTransport`
  and its `Ros2EgoConfig` come from a model module of their own
  (`_ros2.codon`), which `REPLACED_MODELS` replaces with the checked
  `ros2.transport` and `ros2.config`, so the transport `_cyclone_transport()`
  returns is the checked `Ros2EgoTransport`, and its body is compiled. For the
  scenario check, which has no ROS 2 package, the two are stand-ins whose
  constructors are a `compile_error`, as for a class a checked package only
  re-exports (below).
- **A class a checked package only re-exports** is declared in the model by
  name and constructor: `GrpcAutowareBridgeServer`
  (`autoware_bridge/grpc_server.codon`, grpc), `EgoDriverGrpcClient`
  (`driver/egodriver_client.codon`, grpc) and `CarlaDriverEntity`
  (`entity.codon`, grpc and carla_driver_interface's protocol), which the
  checked `autoware_bridge`, `driver` and `entity` packages re-export and no
  checked module builds or uses, and `CoverageReport` (`coverage.codon`,
  openodd's), which the checked `coverage` package re-exports with
  `merge_coverage(documents)`, the `COVERAGE_SCHEMA` string and the
  `CoverGroup` enum (`ODD`, `SITUATION`, `SCENARIO`). Each constructor takes
  Python's parameters and is a `compile_error`: its config
  (`AutowareBridgeConfig`, `DriverClientConfig`) is a checked class the model
  does not have, or the class is only ever built by openodd's merge, so a
  scenario that builds one fails its check rather than passing it with
  arguments the run would refuse. Such a class derives from nothing in the
  model: a subclass of the model's `EgoVehicle` that nothing builds breaks
  Codon 0.19's code generation for every scenario (`LLVM: error: value
  doesn't match function result type`). `test_typecheck_model.py` compares
  each with the Python class, and checks that its constructor is the
  `compile_error`. Of `CoverageReport`, which `merge_coverage()` returns, the
  model also declares what Codon has types for (`runs`, `scenarios`,
  `groups()`, `group_grade()`, `to_markdown()`, #298).
- **A Protocol a scenario may take** is modelled with what is called of it:
  the library check compiles `autoware_stack.launcher` from its source, and
  for the scenario check `autoware_stack/launcher.codon` declares the `AutowareLauncher` that
  `AutowareEgoEntity` starts, polls and stops (`start(*, name,
  bridge_address)`, `poll()`, `stop()`), with the `prepare()` and `close()`
  that `ScenarioQueue` calls (below), and the `AutowareEpisode` that
  `start()` returns. Python cannot instantiate a Protocol, so the model's
  constructor is a `compile_error`.
- **A model function returning a checked class**: `to_map_frame`
  (`coordinate.codon`, which the scenario check and the checked modules that
  import the `coordinate` package compile against) returns
  autoware_bridge's `BridgePose`, which `_bridge_pose.codon` models
  (`BridgePose`, its `Vector3` and `Quaternion`, as `@tuple`s, since they are
  frozen, and `moved_forward()`); `REPLACED_MODELS` replaces it with the
  checked `autoware_bridge.base`, so `AutowareEgoEntity` hands the pose
  `to_map_frame` returns to the checked `AutowareBridge.configure()`
  (it imports `coordinate.transform` itself, which is checked, so the library
  check compiles the real `to_map_frame` there).
- **No `math.dist`** in Codon's `math`: `math.hypot` of the differences,
  which is the computation CPython's `math.dist` makes (`AutowareEgoEntity`,
  `actions.background_traffic`, where `test_background_traffic.py` pins the
  two to the same number).
- **A function whose only return is `return None`**: with an `Optional[X]`
  return annotation, the check returns `Optional[X]()` and realizes `X`
  before calling it ([*Writing a scenario that
  type-checks*](#writing-a-scenario-that-type-checks), #202), so it compiles
  as written, as does a local declared `Optional[X] = None` and returned
  (`_Never.check` in `actions.background_traffic`). Without an `Optional`
  annotation a bare `None` is still Codon's `NoneType` ("cannot typecheck
  'None'"), and `del` of a parameter is refused.
- **A method that takes `**kwargs`, called on a value whose class a checked
  subclass overrides it in**, is called through the base class under `if
  TYPE_CHECKING:`, and as written in the `else` branch: `actions.turn` calls
  `BackendDriven.turn_at_junction(entity, ...)` in the check, since
  `AutowareEgoEntity` overrides it and Codon 0.19 calls an overridden method
  through a dispatch thunk, which takes no keyword argument. The check calls
  each override through its class.
  Codon never compiles the `else` copy, so `test_typecheck_else_branch.py`
  guards it: wherever the then branch calls a method through its class and
  the `else` branch calls it through an instance, both must pass the same
  arguments, the receiver aside (#193).
- **A name the model lacks at the boundary**: when a checked module imports a
  name of a modelled package that the model does not declare
  (`trajectory/__init__.py` re-exports `ResolvedTrajectory`), declare it in
  the model module. When the package does not export it, add it to
  `_BOUNDARY` in `test_typecheck_model.py`, so it is compared with the Python
  class as an exported name is.
- **A Protocol that checked code dispatches through becomes a base class**:
  Codon has no structural typing, and decides `isinstance` and `hasattr` by
  the static type. `sweeper.constraints.Constraint` was a `Protocol`; it is
  an `ABC` whose `evaluate()` is abstract (so, as before, it cannot be
  instantiated itself), and every constraint of the module derives from it.
  What walks a constraint tree asks each class what to bind
  (`_bind_set_level()`, which the composite and set-level constraints
  override) instead of testing the value with `isinstance` and `hasattr`. A
  constraint of someone else's that does not derive from it still works at
  run time: the walk takes the duck-typed path for it, in the `else` branch
  of `isinstance(constraint, Constraint)`, which Codon decides as true. What
  a set-level constraint caches (`_cached_ids`) is a `field(init=False,
  repr=False, compare=False)`: declared, as Codon needs, but neither a
  constructor parameter nor part of the repr or the comparisons, and unset
  until the constraint is bound, as before (`dataclasses.fields()` now lists
  it, so `dataclasses.asdict()` of a set-level constraint includes it, and
  raises `AttributeError` before the constraint is bound; nothing in the
  framework calls either on a constraint).
- **A registry of classes at module level** (`sweeper.constraints`'
  `_LEAF_REGISTRY`, `sweeper.bindings`' `_BINDING_REGISTRY`): Codon holds no
  class as a value, and cannot type `dict[str, type]`. Declare it
  `dict[str, Any]` and assign it in the `else` branch of `if TYPE_CHECKING:`;
  only the uncalled parser reads it.
- **An enum member looked up by name, or an enum iterated over**
  (`ComparisonRule[name]`, `list(ComparisonRule)`): write it as in Python.
  The check rewrites both for an enum of the package being checked
  ([Iterating an enum](#iterating-an-enum)), so `sweeper.constraints` has no
  spelled-out member lists left. The framework's own model enums are the
  exception: they are not iterable.
- **An exception class raised with no arguments** (`raise _Exhausted()` in
  `route.search`): Codon 0.19 raises no exception class without its own
  `__init__`, and its `Exception` takes a message where Python's takes none.
  The constructor calls `super().__init__("")` under `if TYPE_CHECKING:` and
  `super().__init__()` in the `else` branch.
- **An Optional chosen by a conditional expression, whose value goes to a
  union field** (`beside = graph.left(l) if left else graph.right(l)`, then
  `BindingResult(value=beside.id)`, a `float | int | list[int]`): Codon 0.19
  generated invalid LLVM IR and crashed without a diagnostic ("Call
  parameter type does not match function signature", `optimize.cpp:1070`).
  Choose with an `if` statement into a local annotated `Optional[...]`.
- **A function that returns a tuple with an `Optional` member**, `None` in
  one return and a value in another (`route.frame.ego_placement()`):
  annotate the local in each return's own assignment (`no_goal:
  Optional[tuple[int, float]] = None`, `goal_at: Optional[tuple[int, float]]
  = frame.locate(s)`): one local annotated `Optional` where it is assigned
  `None`, and assigned the value later, was not enough.
- **A loop variable assigned a subclass's value is retyped**: in a function
  that already declared `other: ConstLanelet`, `other = near` (a `Lanelet`)
  makes `other` a `Lanelet`, and a list of `(float, ConstLanelet)` then
  refuses it. Give the upcast a name of its own (`nearby: ConstLanelet =
  near`, `route.geometry`).
- **A value of a type the model holds**: a checked module that hands a checked
  value to a checked module that imports it through the model
  (`MODELLED_IMPORTS`) hands the wrong type. `scenario_runner` hands the
  checked `BaseScenario` to `measures.set_measured_scenario()`, and its
  `EgoConfig` to `EgoVehicle.spawn()`, which take the model's: it makes those
  calls in the `else` branch of `if TYPE_CHECKING:`, and takes the spawned
  actor from `ego.actor` in the first.
- **No module object to assign a global through**
  (`vehicle_entity._warmup_done = True`): Codon 0.19 imports no module as a
  value. The module sets its global in a function of its own
  (`entity.vehicle_entity._set_warmup_done()`).
- **The runner's boundary is modelled**: `coverage.codon` declares the
  `CoverageCollector` (its constructor, `start()`, `tick()`, `end()`,
  `write(path, scenario)`, and the run's case and outcome,
  `record_case(case, space=None)` and
  `record_outcome(passed, *, failed_at=None, environment=None)`; openodd's
  `backend` and `generator` are refused), `odd.codon` `resolve_odd(spec)`,
  whose union parameter (`str`, `Path`, `OddDefinition` or `None`) is
  generic and checked in its body, `reset_probes()`, the ODD's `modules`,
  the case a run records (`OddCase`, a `@tuple` of openodd's `OddSample` and
  `ParameterSpace`, which only the sampler builds) and what the runner reads
  of odd.route's verdict on the ego's planned route (`plan_route_coverage()`,
  `RouteCoverage` and `RouteLanelet`, which only it builds), and openodd's
  `UNDECIDED`, which the checked `odd.probes` returns: the one instance of a
  class openodd does not export (odd.model names it `Undecided`, for
  annotations), whose constructor is a compile error, as Python cannot build
  another, so the model allocates its one with `Undecided.__new__()`;
  `environment.codon` the id of the run's environment (`capture()` and
  `RunEnvironment.id`), and `autoware_stack/launcher.codon` the launcher's
  `prepare()` and `close()`, which the queue calls.
  `test_typecheck_model.py` compares each with the Python one.
- **No attribute of a union**: Codon 0.19 reads `pose.yaw` of a value typed
  as a union (a vertex's `position`) as no type, even when every member has
  it with one type. After `isinstance`, name the value as the class
  (`relative: RelativeLanePose = position`) and read that; a union reaches a
  parameter typed with one of its members the same way. A generic function
  handed the union compiles every branch for it, so one that returns a
  different class per branch (`route.positions.resolve_route_pose()`) is
  called with each class on its own (`trajectory.resolve._place_on_route()`).
- **A callable of a union**: `Callable[[Union[A, B]], R]` crashes Codon 0.19
  ("union already sealed"), whether a parameter's, a variable's or a
  return's. Spell the callable out in the annotation (the checker drops a
  `Callable` parameter or return annotation, so the function is generic,
  and calls it with `_acs_Returning[R, N]`); a module-level alias
  (`PositionResolver = Callable[...]`) stays for Python and is `None` to the
  check, which refuses the alias as an expression ("expected type
  expression"). A function that takes either a resolver it is given or one
  it builds hands each to a helper of its own
  (`trajectory.resolve._resolve_vertices()`): one variable holds one type.
- **A closure calling another closure** does not see that one's captures
  ("name 'x' is not defined"): make the inner one a module-level function
  taking what it read (`trajectory.resolve._ego_route_s()`).
- **A tuple display does not make a `float` the `Optional[float]` its
  annotation wants** (`return world.x, world.y, world.z, world.yaw` from a
  function returning `Tuple[float, float, Optional[float], Optional[float]]`):
  a call does (`trajectory.resolve._resolved()`).
- **A singleton's `__new__`**: Codon's constructor allocates and calls
  `__init__`, never a class's `__new__`, and its `object.__new__` takes no
  class. The check rewrites `super().__new__(cls)` to `C.__new__()`, which
  allocates a `C`, and a class without `__init__` is built by Codon from
  every attribute it declares: give it an `__init__` that does nothing
  (`MapManager`; Python calls it after `__new__`).
- **`Path.read_text()`** is in the pathlib shim (CPython 3.10's
  `encoding` and `errors`; 3.13's `newline` is refused); it reads an empty
  text.
