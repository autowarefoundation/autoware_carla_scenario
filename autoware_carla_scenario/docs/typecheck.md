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

A scenario that imports a module with no model (numpy, or a framework module
outside the public API) is refused with a message naming the module.

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
members become instances with a `name` and a typed `value`; looking a member
up by value or name, and iterating over the class, are not modelled), a class
deriving from an exception, and a `@classmethod` (a static method of its
class: `cls` names the class it is defined in).

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

### The manifest: `typecheck/library.py`

`CHECKED` lists the modules compiled from their source; `EXCLUDED` maps every
other module to the reason it is not. `"not yet checked (#45)"` marks a module
nobody has made compile yet; any other reason names what keeps a module out
(`"imports numpy, lanelet2"`: a package or a standard module Codon cannot
compile). Every module of the package except `typecheck/` is in exactly one of
the two, which the test suite checks, so **a new module needs an entry**.

`UNCALLED` names the few public functions and methods of a checked module the
check leaves out (`module.Class.method`), each with its reason: one whose job
Codon cannot express, in a module that otherwise compiles. Today that is
what builds a condition's details (`get_details()`, `to_summary_dict()`,
`to_dict()`: a `dict[str, Any]` of mixed value types) and what serialises
them (`ScenarioResult.to_json()`, with `json`). Codon compiles a function
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
OpenDrivePose`) is called once with each member, so the body is checked for
every type it accepts. A parameter with no annotation, or one Codon cannot
express (`Any`, `object`, `Callable`, ...), gives the check nothing to call
with and is reported as a problem of the module, at its line: annotate it with
the type its callers pass. The one exception is `other: object` in `__eq__`
and `__ne__`, which Python requires; it is called with the class itself.

### Checked modules and the model

The checked modules are compiled next to the same model the scenario check
uses. Every import of the framework in a checked module is pointed at what
stands in for it in the check's workspace:

- another **checked** module: its own (rewritten) source;
- any other module: its **model**, the `codon/autoware_carla_scenario/` module
  of the public package it belongs to (`coordinate.poses` is modelled by
  `coordinate.codon`). A module with no model (`utils.config`) cannot be
  imported by a checked module until it is checked itself.

A checked module is compiled under a name of its own
(`_acs_lib.autoware_carla_scenario__coordinate__frames`), since Codon reads
the workspace's packages only from `.codon` files and names a file in its
errors by its base name alone; errors are reported at the line of the real
source file. A type a checked module defines is not the type the model
declares under the same name, so a checked module cannot hand its own value to
a model function that wants the model's: check modules bottom-up, before the
modules that use them.

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
  action, `scenario_base`) cannot do so until that module is checked too;
  `_acs_list` gives a list display of checked conditions the type of its
  first element, so a display mixing two checked condition classes waits for
  the same.

`BaseCondition.__init_subclass__`, which wraps every subclass's `check()` to
record its last result for the UI, is in `UNCALLED`: the check sees each
`check()` as written.

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
- Standard modules Codon does not have (`json`, `inspect`, `importlib`,
  `pathlib`, ...) are not available to a checked module; the `typing`,
  `dataclasses`, `enum`, `logging`, `abc` and `__future__` shims of
  `codon/` are. `collections.abc` is not: import `Sequence` and the other
  abstract collections from `typing`. `@abstractmethod` is dropped, since
  Codon 0.19 cannot decorate a method; `ABC` is an empty base.
  `typing.Any` imports, and every annotation naming it is dropped.
- **An overridden method takes concrete types**: Codon 0.19 cannot call a
  method a subclass overrides when one of its parameters is generic, which an
  abstract collection (`Sequence[X]`) or an unannotated parameter is.
- **No `**` in a dict display** (`{"a": 1, **other}`): Codon's parser fails on
  it, even in a function it never compiles. Build the dict, then
  `update()` it.
- **`dict.get()` takes a default** in Codon, of the value type. Where
  "absent" has no value of that type, test membership:
  `d[k] if k in d else None`.
- **CARLA's containers are not lists.** `world.get_actors()` is a
  `carla.ActorList`; a parameter that takes it or a list is
  `Union[carla.ActorList, list[carla.Actor]]`, and the check calls it with
  each.
- **Codon cannot hash a class**: `hash((MyClass, x))` names the class by its
  name instead.
- **An attribute holds one type**: one assigned a union (`EntityRole | str`)
  is stored converted where every use converts it anyway
  (`self._entity_name = str(entity_name)`).
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
- **No variable-length tuples**: Codon's tuples have a fixed length, so a
  `tuple[int, ...]` parameter or field gives the check nothing to call with.
  Turning it into a list changes what the class does (a frozen dataclass's
  hash, its equality with a tuple), so such a module stays in `EXCLUDED`
  (`route.model`).
- **A name the model lacks at the boundary**: when a checked module imports a
  name of a modelled package that the model does not declare
  (`trajectory/__init__.py` re-exports `ResolvedTrajectory`), declare it in
  the model module. When the package does not export it, add it to
  `_BOUNDARY` in `test_typecheck_model.py`, so it is compared with the Python
  class as an exported name is.
