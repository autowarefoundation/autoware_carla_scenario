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
done: bool = scenario.is_done()
```

Codon checks a function only when something calls it, so what is checked is
everything `setup()` and `is_done()` reach: the calls into the framework, the
scenario's own helpers, and, through `register_pass_condition()` and the
other `register_*` methods, the `check()` of a custom condition and the
`execute()` of a custom action.

An ODD written in Python ([ODD](odd.md)) is checked the same way: the program
calls its builder (`odd: OddDefinition = urban_odd()`). Every attribute's
probe is then called with a world, and every condition must come from an
attribute. The runner checks the ODD named by the `odd` key under the same
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
Nothing in the model runs; `test_typecheck_model.py` keeps every declaration
in step with the Python definition of the same name (parameter names, order,
defaults, keyword-only parameters, methods, enum members).

A scenario that imports a module with no model (lanelet2's `matching`, say,
or a framework module outside the public API) is refused with a message
naming the module. Third-party packages the framework reads are modelled as
far as it reads them (lanelet2, pyxodr, carla_driver_interface, python-dotenv,
ffmpeg-python; see *Making a module compile*), and a scenario may use what
those models declare.

## Writing a scenario that type-checks

A scenario is ordinary typed Python. Codon is stricter than Python in a few
places, and the checker smooths over most of them (it rewrites `X | None` into
`Optional[X]`, drops `@dataclass` and resolves `field(...)` defaults, accepts
keyword-only parameters, required ones after a default included, gives a list of different conditions the type
`list[BaseCondition]`, and drops a parameter or return annotation naming a
class the module defines further down, which Codon cannot name yet). What a
scenario has to do itself:

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
- **Use only the public API** from `autoware_carla_scenario` and its listed
  subpackages, and the `BaseScenario` attributes meant for subclasses
  (`ego_config`, `ego_entity`, `world`, `_ground_projection`, ...).
- **Pass numbers of the declared type.** An `int` is accepted for a `float`,
  but not inside a container: a `list[float]` parameter wants `[1.0, 2.0]`.
  YAML values are converted for you.
- **Convert nested mappings in the config.** The runner passes a YAML mapping
  to the config class as a `dict`, so a field annotated with a dataclass
  (`goal: Goal`) holds one at run time unless the config's `__post_init__`
  turns it into a `Goal`, as the built-in configs do for their NPC lists.
  The check builds the config the same way and refuses an unconverted
  mapping at its `scenario.<key>`.

Annotations Codon cannot express (`Union` of two types, `Any`, `Callable`,
`Sequence`, `type[...]`) are dropped from parameters, which become generic:
Codon then checks each call with the arguments it is given.

The checker also rewrites what Codon spells differently: an `Enum` class (its
members become instances with a `name` and a typed `value`; a lookup by value,
`ActionState("completeState")`, takes a value of that type and is typed but
not resolved to the member; looking a member up by name and iterating over the
class are not modelled), a class deriving from an exception, a `@classmethod`
(a static method of its class: `cls` names the class it is defined in), and an
annotation naming one of CARLA's integer enumerations
(`carla.TrafficLightState`), which typesafe_carla's Codon library has as a
value whose members are `int`, not as a type: it becomes `int`.

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
fully as the named ones: their public functions are called too, and **errors
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
(`"imports lanelet2, json"`: a package or a standard module Codon cannot
compile). numpy is not one: Codon ships its own, and a module's numpy calls
compile. Every module of the package except `typecheck/` is in exactly one of
the two, which the test suite checks, so **a new module needs an entry**.

`UNCALLED` names the few public functions and methods of a checked module the
check leaves out (`module.Class.method`), each with its reason: one whose job
Codon cannot express, in a module that otherwise compiles. Today that is
what builds a condition's details (`get_details()`, `to_summary_dict()`,
`to_dict()`: a `dict[str, Any]` of mixed value types) and what serialises
them (`ScenarioResult.to_json()`, with `json`), and what reads the
authoring package's pydantic specs, typed `Any`
(`signals.build_controllers()`). Codon compiles a function
only when something calls it, so neither its body nor an import made inside
it is compiled: a module that needs a standard module Codon lacks for one
function imports it inside that function (`import json` in `to_json()`) and
lists the function here. A test checks that every entry names a public
function or method of a checked module.

To check one more module:

1. Move its entry from `EXCLUDED` to `CHECKED`.
2. Run `uv run scenario-check --library` and fix what it reports, in the
   module. Only annotations and Codon-friendly rewrites: the module must
   behave exactly as it did.
3. Run `uv run pytest autoware_carla_scenario/test/carla_scenario/test_typecheck_library.py`.

### What is checked of the framework

Every module now has an entry other than `"not yet checked (#45)"`. 138 of
the package's 228 modules (61%; about 29,800 of 65,000 lines) are `CHECKED`:
the kinematics, the coordinate frames and poses, the conversions between
Lanelet2, OpenDRIVE and CARLA world poses (`coordinate.transform`), the stop
lines read off a Lanelet2 map (`utils.stop_line`, `coordinate.stop_line`)
and the lane arithmetic of a relative lane pose (`trajectory.relative_lane`),
all against the lanelet2 model, the condition base and
every condition, the action base with the environment, follow-trajectory,
traffic-signal, signal-controller, lane-change, routing, set-speed, turn and
walk-straight actions, the logical scenario's route search and matches (`route.model`) and
the scenario's route (`route.active`),
the `signals` package (the signal controllers and their registry), the ego,
NPC vehicle and pedestrian entities, the Autoware ego
(`entity.autoware_entity`), the entity registry and the policy
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
`authoring.uv_tool`), the CARLA server manager (`server`) and the camera
recorder that writes a run's video (`camera_recorder`), the Autoware
bridge interface and its fake, the trajectory model, the measures, the ODD
unit names (`odd.unit_names`), the CARLA-only helpers, the traffic-light
lookups included, and the scenario configs (`scenario_config`, `examples.configs`), the camera
and LiDAR sensors (`sensor`), the six example scenarios, and the external
driver's configs and client interface, trajectory follower and observations
(`driver.base`, `driver.control`, `driver.observation`, over
`driver.geometry`), and the `autoware_bridge`, `entity`, `driver` and `maps`
packages themselves, whose `__init__` re-exports what is not checked
through models of it (see *A class a checked package only re-exports*
below). 69
functions and methods of those modules are `UNCALLED`: most of them a
condition's `get_details()` or a backend's `describe()`, what takes a traffic
backend's options node (`Mapping[str, Any]`), what reads a route search or a
match from YAML values of any type (`Range.parse()`, `parse_route_search()`,
`RouteMatch.from_config()`) and what lays a route on a lanelet2 map handed
over as `Any` (`set_scenario_route()`), the `[tool.uv.sources]` entry of a
framework pin, a dict of str and bool values (`Pin.uv_source()`), the six members of
`BaseScenario` built on the ego's class (`ego_type`, a `type[EgoVehicle]`:
the constructor, `create_ego()`, `ego_requires_goal`, `require_goal()` and
`register_route_to_goal()`) or on cover expressions of any value (`register_cover()`),
the run-time wrapper of `TrajectoryVertex.__init__`
(`_refusing_vertex_times`), what resolves a followed trajectory with the
unchecked `trajectory.resolve` (`follow_trajectory._resolve_in_world()`),
the road-marking pattern of the stop lines, which returns a `refers`
parameter (any of lanelet2's primitives) as a line string
(`utils.stop_line._road_marking_stop_lines()`),
`signals.build_controllers()`, which reads
the authoring package's pydantic specs, what the runner does with `json`,
lanelet2 or a condition's details (writing the result and its environment,
`_write_result()`; checking the planned route against the ODD,
`ScenarioRunner._warn_if_route_leaves_odd()`; collecting the condition
statuses, `_collect_condition_statuses()`), the queue's pytest fixture
(`ScenarioQueue.as_fixture()`), `maps.config.resolve_map_paths()`,
which reads the `map` config node of any shape (`Any`), the driver configs'
`from_mapping()` (a Hydra node, `Mapping[str, Any]`) and
`encode_frame_jpeg()` (cv2). The 90 `EXCLUDED` modules fall in
three groups. Many fit more than one (66 import what Codon cannot compile,
39 build on an excluded module and 27 hold values Codon has no type for), so
each is counted once, in the first group that fits it:

- **66 import what Codon cannot compile**: lanelet2 beyond what its model
  declares (most of these walk its objects typed `Any`), pyxodr,
  pydantic, grpc, omegaconf and hydra, yaml, openodd (a Rust extension), or
  a standard module Codon lacks (`json`, `inspect`, `tempfile`, `importlib`
  beyond `importlib.metadata`, or more of `pathlib`, `shutil`, `signal`,
  `subprocess`, `importlib.metadata` or `urllib.parse` than their shims
  declare), at module level or inside a function the check calls. This is
  the tooling around a run (the editor, the UI, authoring, the sweeper, the
  map cache, the Autoware launcher, the record of a run's environment), the
  ODD and coverage modules that wrap openodd, and the rest of what reads
  Lanelet2. A reason names `pathlib`
  only where the module uses more of it than the shim declares, and the
  other shimmed modules likewise.
- **21 build on an excluded module**, and import nothing Codon lacks
  themselves: they import a module that is neither checked nor modelled, or
  names that its package's model does not declare. Among them are the
  packages that re-export one (`authoring`, `odd`, `sweeper`,
  `coverage`, `autoware_stack`, `examples`, `scaffold`), the
  authoring compiler, builders and validator, the route frame, poses and search
  (`route.frame`, `route.positions`, `route.search`, through lanelet2),
  `trajectory.resolve`
  and `traffic.registry` (the `registry`, in its plugin loader); 9 of the 21
  also hold values Codon has no type for.
- **3 hold values Codon has no type for**, and nothing else keeps them out:
  `Any` (`trajectory.authoring`, `authoring.registry`'s
  field defaults) or a class as a value (`utils.config`, which `traffic.config`
  and `traffic.sumo.config` import only in their uncalled `from_mapping()`).

The scenario check is not limited by this: it compiles every scenario against
the model, which declares the whole public API.

### What is compiled

Codon checks a function only when something calls it, so the check appends a
function to each checked module that calls every public function and every
method of every public class (constructors, static and class methods,
properties and dunder methods included) with a value of each parameter's
annotated type:

```python
def _acs_library_check():
    normalize_angle_deg(_acs_value(float))
    Vector3(_acs_value(float), _acs_value(float), _acs_value(float))
    _acs_value(Vector3).dot(_acs_value(Vector3))
```

`_acs_value(T)` is a `T` Codon cannot tell from a real one; nothing compiled
is ever run. A parameter annotated with a union (`Lanelet2Pose |
OpenDrivePose`, or `Union[BaseAction, Callable[[carla.World], None]]`) is
called once with each member, so the body is checked for every type it
accepts. A parameter with no annotation, or one Codon cannot
express (`Any`, `object`, `Callable[..., R]`, `Callable[[Any], Any]`, ...),
gives the check nothing to call with and is reported as a problem of the
module, at its line: annotate it with the type its callers pass. The one
exception is `other: object` in `__eq__` and `__ne__`, which Python requires;
it is called with the class itself. A `Callable` whose argument and result
types Codon can express (`Callable[[carla.World], Optional[float]]`, a
dataclass field too) is called with a value of Codon's `Callable` type, which
a function of that signature converts to.

A `*args: T` is given one `T` (`run_uv(cwd, *args: str, ...)` is called
with one `str`), so the body is compiled for it; one with no annotation, or
of `object` or `Any` (`__exit__(self, *args: object)`), is given nothing.

A method with keyword-only parameters or `**kwargs` is called through its
class (`TrafficBackend.start(_acs_value(TrafficBackend), ...,
skip_actor_ids=...)`): Codon 0.19 calls a method a subclass may override
through a dispatch thunk, which takes neither a keyword argument nor
`**kwargs`, and the call through the class checks the same body.

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
  `autoware_stack/launcher.codon`, `maps/cache.codon`), or, for a module of
  the top-level package, a model of that module (`carla_install.codon`, for
  `server`, declares `installed_executable()`; `trajectory_recorder.codon`,
  for `scenario_runner`, the `TrajectoryRecorder` it starts, records and
  closes, whose JSON lines hold values of mixed types). A module with no
  model (`utils.config`) cannot be imported by a checked module until it is
  checked itself.

Two checked modules that import each other can each need the other's types
the moment Codon reads it: `BaseScenario` declares `_measures: dict[str,
Measure]` at class level, and `measures` declares the running scenario
(`_MEASURED: dict[str, Optional[BaseScenario]]`) as a global, so neither can
be read first. `MODELLED_IMPORTS` in `typecheck/library.py` breaks such a
cycle: the module it names (`measures`, `entity.ego`) imports the other
(`scenario_base`) through its model, as it did before that one was checked.
Its types are the model's, which never meet the checked module's.
`entity.autoware_entity` imports `scenario_base` through the model too: its
`AutowareEgoEntity.spawn()` overrides `EgoVehicle.spawn()`, so it takes the
same, the model's, `EgoConfig`. Through the checked `scenario_base`, a run
with an example scenario in it failed in Codon 0.19's code generation
(`LLVM: error: value doesn't match function result type '{ ptr, ptr }'`, in
`opt_ref_new` of the model's `BaseScenario`).

A model module that stands in for unchecked code (`coordinate.codon` for
`map_manager.py`) declares only what the checked modules
and scenarios use of it: `MapManager` there is `get_instance()` and the
`road_network` it reads, not the class's whole API.

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
  first element, so a display mixing two checked condition classes waits for
  the same.

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

- **Annotate what the check calls**: every public parameter, with a type
  Codon can express. Spell a union out (`Lanelet2Pose | OpenDrivePose`) rather
  than through an alias (`AnyPose`); a union with `object` or `Any` in it
  gives nothing to call with.
- **Declare attributes** on any class in a hierarchy (exceptions included)
  and on a class with `__slots__`, at class level (`_value: str`).
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
  attribute per key.
- Standard modules Codon does not have (`json`, `inspect`, `tempfile`,
  ...) are not available to a checked module; the `typing`,
  `dataclasses`, `enum`, `logging`, `abc`, `queue`, `pathlib`, `hashlib`,
  `os`, `shutil`, `subprocess`, `signal`, `atexit`, `threading`,
  `importlib.metadata`, `urllib.parse` and `__future__` shims of `codon/` are. Each takes the
  arguments CPython takes, so the check accepts what CPython accepts and
  refuses the rest; where Python versions differ, it takes what Python 3.10,
  the oldest the package supports, takes. `pathlib` declares `Path(*parts)`,
  `/`, `str()`, `==`, `hash()`, `expanduser()`, `resolve()` and
  `Path.home()` and `Path.cwd()`; like Python's, it builds and joins a `Path` with a `str` or
  an `os.PathLike` only (a `Path`, or what `shutil.copy2()` returns for one;
  an `Optional` of one is unwrapped, as Codon unwraps one passed anywhere). It declares the file-system queries too: `exists()`,
  `is_file()`, `is_dir()`, `glob(pattern)` and `absolute()`, with Python
  3.10's signatures (3.12 and 3.13 added the keyword-only `follow_symlinks`,
  `case_sensitive` and `recurse_symlinks`, which are refused), and `name`,
  `suffix`, `parent`, `parents` (indexed, iterated, `len()` and `in`; not
  sliced), `stem`, `with_name(name)` (of a `str`), `as_posix()`, `relative_to(other)` (one path, by position:
  Python 3.10 also joins more parts, but 3.14 refuses them, so the check
  does too; 3.12's `walk_up` is refused) and the ordering of two `Path`s, so
  `sorted()` takes them. Nothing
  touches the file system: a query is `False` and a glob yields nothing.
  Of what writes, `mkdir(mode=0o777, parents=False, exist_ok=False)` and
  `write_text(data, encoding=None, errors=None, newline=None)` are declared
  (`maps.opendrive` caches what it fetches); `open()`, `chmod()`,
  `read_text()` and the rest are not. `os` replaces Codon's own (whose
  `environ` has no `get()` and iterates over pairs), keeping its `getenv()`,
  which Codon's runtime and typesafe_carla's library call: `environ` is a
  mapping of `str` to `str` (`[key]`, assignment, `del`, `in`, iteration
  over the keys, `len()`, `get(key)`, `get(key, default)`, `pop()`,
  `setdefault()`, `keys()`, `values()`, `items()` and `copy()`, a `dict`, as
  CPython's returns), `getenv(key, default=None)` (a `str` default gives a
  `str`, `None` an `Optional[str]`; a default of another type, which CPython
  takes, is refused), `fspath()`, `getpid()`, `getppid()`, `getpgid(pid)`,
  `kill(pid, signal, /)`, `killpg(pgid, signal, /)`, `system(command)`,
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
  `start_new_session`), `CompletedProcess(args, returncode, stdout=None,
  stderr=None)` and its `check_returncode()`, `Popen`'s `poll()`, `wait()`,
  `communicate()`, `send_signal()`, `terminate()`, `kill()`, `pid`,
  `returncode` and standard streams, `PIPE`, `STDOUT`, `DEVNULL`, and
  `SubprocessError`, `CalledProcessError(returncode, cmd, output=None,
  stderr=None)` and `TimeoutExpired(cmd, timeout, output=None,
  stderr=None)`. The command is a list of `str`, a working directory a
  `str` or an `os.PathLike`, a stream `None`, one of the constants or a file
  descriptor, an environment a `dict` of `str`, a timeout a number of
  seconds or `None`. What CPython also takes is refused: a `str` command,
  `shell=True` and the other keywords, a file object for a stream,
  `os.environ` itself as the environment (pass `os.environ.copy()`, which
  is the `dict` CPython's returns, as `authoring.uv_tool` does), and
  arguments after the command by position (`run(args, -1)`, which run()
  hands to `Popen` as `bufsize`). Since the check reads `bytes` as `str`, a
  process's output is a `str` with or without `text=True`, and
  `CompletedProcess[bytes]` and `Popen[bytes]` are `[str]`; it is `None`
  when it was not captured, as in CPython, so `stdout` and `stderr` are an
  `Optional`. A stream's `write()` takes what a binary stream takes in
  CPython, bytes (a `str`) or a buffer such as an array's `data` (a `Ptr` in
  Codon's numpy; `camera_recorder` writes `bgr.data`), and since the check
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
  `SIGCHLD`) as `int`s, which CPython's `Signals` members are, so
  `os.killpg(pgid, signal.SIGTERM)` compiles; `SIG_DFL` and `SIG_IGN`, which
  are no `int`s here, since CPython takes no other `int` as a handler; and
  `signal(signalnum, handler)`, whose handler is one of those two or a
  function the check calls as CPython calls a handler, with the signal's
  number and the current frame (`None` in the check, which has no frame
  type: annotate that parameter `Any`), so a function that cannot take them
  is refused where CPython fails only when the signal arrives. What it
  returns, the previous handler, is a value nothing is declared of (a
  `Handlers` member, a function or `None` in CPython, which takes no `None`
  as a handler), so it cannot be handed back to `signal()`; `getsignal()`,
  `Signals`, `alarm()` and the rest of the module are not declared. `atexit`
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
  `urllib.parse`, `signal` and `atexit` and call what they declare,
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
  no import in the `else` branch: `scenario_queue` imports tqdm there, at
  module level, and waits without the progress bar in the first branch. The `logging` shim's
  methods take `exc_info` as a `bool`.
- **pyxodr's road network is the model's**: `codon/pyxodr/` names the
  `RoadNetwork`, `Road`, `LaneSection` and `Lane` that `coordinate.codon`
  declares (what the framework reads of them, down to the XML elements), so
  a checked module annotates with pyxodr's own classes under `TYPE_CHECKING`
  (`utils/traffic_light.py`, which may not import `coordinate`;
  `coordinate/transform.py`) and takes the network `MapManager` returns.
  Nothing else of pyxodr is there.
- **lanelet2 is modelled**: `codon/lanelet2/` and
  `codon/autoware_lanelet2_extension_python/` declare the classes and
  functions of simple-lanelet2's Python API the checked modules use, with
  its parameter names: `lanelet2.core`'s map, lanelet layer, lanelets, line
  strings, points, attribute maps and regulatory elements (their ids,
  attributes, centrelines, bounds, rule parameters and stop lines),
  `BasicPoint2d`; `lanelet2.geometry`'s `to2D()`, `length()`,
  `toArcCoordinates()` (an `ArcCoordinates`), `findNearest()` (of the
  lanelet layer) and `inside()`; `lanelet2.routing.RoutingGraph(laneletMap,
  trafficRules)` with `following()`, `previous()`, `left()`, `right()`,
  `adjacentLeft()` and `adjacentRight()`; `lanelet2.traffic_rules.create()`
  with its `Locations` and `Participants`; `lanelet2.io.Origin`; and
  Autoware's `MGRSProjector`, `AutowareTrafficLight` and `RoadMarking`.
  Each module's header says what it leaves out; parts of the API the
  framework does not use yet (loading a map, routes, matching, editing)
  are not declared, and the check refuses them. The model's `MapManager`
  hands out its `lanelet_map` and `routing_graph` as these classes. Python's
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
- **python-dotenv and ffmpeg-python are modelled** as far as `server` and
  `camera_recorder` use them, with the installed packages' signatures, which
  `test_typecheck_model.py` compares: `codon/dotenv/` declares
  `load_dotenv(dotenv_path=None, stream=None, verbose=False, override=False,
  interpolate=True, encoding="utf-8")`, of a path that is `None`, a `str` or
  an `os.PathLike` (a stream, an `IO[str]` Codon has no type for, is not
  taken); `codon/ffmpeg/` declares the fluent pipeline the camera recorder
  builds: `ffmpeg.input(filename, **kwargs)` returns a `FilterableStream`
  (`ffmpeg.nodes`), whose `output(*streams_and_filename, **kwargs)` takes
  the file name last, as a `str` (ffmpeg-python raises a `ValueError` for a
  `Path`), and returns an `OutputStream`, which alone has
  `overwrite_output()` and `run_async(cmd="ffmpeg", pipe_stdin=False,
  pipe_stdout=False, pipe_stderr=False, quiet=False,
  overwrite_output=False)`, a `subprocess.Popen[bytes]`. An ffmpeg option is
  any keyword of any value, as ffmpeg-python hands it to ffmpeg as text.
  Nothing else of either package is declared (`dotenv_values()`, the
  filters, `run()`, the module-level `ffmpeg.output()`), and a scenario may
  use what is, as it may the shims.
- **No `bytes`**: Codon 0.19 has none, and reads a `b"..."` literal as a
  `str`, so the check reads an annotation naming `bytes` as `str`
  (`DriveOutcome.debug_info: bytes = b""`). What a process writes is a `str`
  in the check, which has no `decode()`: decode it in the `else` branch of
  `if TYPE_CHECKING:` and take the `str` in the first
  (`CameraRecorder.stop()`, ffmpeg's error output).
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
  shape* below).
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
  `List[int]` field is built as a `List[int]` as on a `list[int]` one.
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
  change the type of what is assigned to it later.
- **Codon does not narrow a union either**, with `is` or `isinstance`, and an
  attribute holding a union (`Union[Sequence[int], TrafficLightTarget,
  None]`) reaches every use as the union. Store it split by kind
  (`_all_lights: bool`, `_lanelet2_ids: Optional[Sequence[int]]`), deciding
  the kind once in `__init__`, where `isinstance` on the parameter is decided
  when Codon compiles.
- **A default of a subclass for an `Optional` of its base** goes through an
  `if` statement, not a conditional expression: `x if x is not None else
  Sub()` is refused (`Optional[Base]` is not `Optional[Sub]`).
- **A dict has no truth value** in Codon: `if not d:` is `if len(d) == 0:`.
- **No `setattr(obj, name, value)` with a name known only at run time**:
  spell the assignments out (`actions/environment.py`).
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
  `raise X from exc` compiles as it is.
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
  its tuple. A dataclass field's `field(default=())` is an empty `List[X]`
  (`Pin.warnings`); a tuple display (`("a",)`) stays a fixed-length Codon
  tuple, which is no `List[X]`: pass `tuple([...])` of a list instead. A parameter `tuple[X, ...]` is dropped (a caller may pass a
  tuple), and the library check calls it with a `List[X]`.
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
  from an annotated empty list and assign `x` under `if x:`.
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
  first branch what it is handed (`FollowTrajectoryAction._resolve()` hands
  `_resolve_in_world()`, which imports `trajectory.resolve`, its reference
  lookup, which the first branch calls instead).
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
  function that only *returns* one compiles: the check drops the
  `tuple[X, ...]` return annotation
  (`signals.registry.registered_signal_controllers()`).
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
- **lanelet2's map at the model boundary**: the model's `MapManager` hands
  out its `lanelet_map` and `routing_graph` as the lanelet2 model's classes
  (above), which the model's `RouteFrame` takes, so the checked
  `scenario_route_frame()` (`route.active`) compiles; `route.frame` itself,
  which reads them, is not checked.
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
  top-level package model itself is not replaced: the examples would then
  derive from the checked `BaseScenario`, and their constructors' calls to
  `super().__init__()` would compile `BaseScenario.__init__`, which is
  `UNCALLED` (it stores the ego's class, `ego_type`).
- **A class a checked package only re-exports** is declared in the model by
  name and constructor: `GrpcAutowareBridgeServer`
  (`autoware_bridge/grpc_server.codon`, grpc), `EgoDriverGrpcClient`
  (`driver/egodriver_client.codon`, grpc) and `CarlaDriverEntity`
  (`entity.codon`, grpc and carla_driver_interface's protocol), which the
  checked `autoware_bridge`, `driver` and `entity` packages re-export and no
  checked module builds or uses. Each constructor takes Python's parameters
  and is a `compile_error`: its config (`AutowareBridgeConfig`,
  `DriverClientConfig`) is a checked class the model does not have, so a
  scenario that builds one fails its check rather than passing it with
  arguments the run would refuse. Such a class derives from nothing in the
  model: a subclass of the model's `EgoVehicle` that nothing builds breaks
  Codon 0.19's code generation for every scenario (`LLVM: error: value
  doesn't match function result type`). `test_typecheck_model.py` compares
  each with the Python class, and checks that its constructor is the
  `compile_error`.
- **A Protocol a checked module takes** is modelled with what it calls:
  `autoware_stack/launcher.codon` declares the `AutowareLauncher` that
  `AutowareEgoEntity` starts, polls and stops (`start(*, name,
  bridge_address)`, `poll()`, `stop()`) and the `AutowareEpisode` that
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
  which is the computation CPython's `math.dist` makes (`AutowareEgoEntity`).
- **A method that takes `**kwargs`, called on a value whose class a checked
  subclass overrides it in**, is called through the base class under `if
  TYPE_CHECKING:`, and as written in the `else` branch: `actions.turn` calls
  `BackendDriven.turn_at_junction(entity, ...)` in the check, since
  `AutowareEgoEntity` overrides it and Codon 0.19 calls an overridden method
  through a dispatch thunk, which takes no keyword argument. The check calls
  each override through its class.
- **A name the model lacks at the boundary**: when a checked module imports a
  name of a modelled package that the model does not declare
  (`trajectory/__init__.py` re-exports `ResolvedTrajectory`), declare it in
  the model module. When the package does not export it, add it to
  `_BOUNDARY` in `test_typecheck_model.py`, so it is compared with the Python
  class as an exported name is.
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
  `CoverageCollector` (its constructor, `start()`, `tick()`, `end()` and
  `write(path, scenario)`; openodd's `backend` and `generator` are refused),
  `odd.codon` `resolve_odd(spec)`, whose union parameter (`str`, `Path`,
  `OddDefinition` or `None`) is generic and checked in its body, and
  `reset_probes()`, `coordinate.codon` `MapManager.reset()` and
  `initialize()`, and `autoware_stack/launcher.codon` the launcher's
  `prepare()` and `close()`, which the queue calls. `test_typecheck_model.py`
  compares each with the Python one.
