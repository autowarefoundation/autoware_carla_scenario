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
keyword-only parameters, and gives a list of different conditions the type
`list[BaseCondition]`). What a scenario has to do itself:

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
