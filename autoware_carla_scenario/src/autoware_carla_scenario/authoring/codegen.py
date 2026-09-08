"""Render :mod:`autoware_carla_scenario.authoring._builders_generated`.

A builder used to be written out by hand for every primitive, and nearly all of
that writing was the same three moves: rename a field to the constructor's
keyword, lift a select's string into an enum, and pass the pieces the build
context holds.  The constructor already states all three -- in its signature --
so this module reads the signature and writes the builder.

It runs at development time, not at import time, and that is the point.  The
editor process imports :mod:`.registry` without CARLA installed, so nothing
there may introspect a runtime class; a generator that runs where CARLA *is*
installed can, and the code it emits is ordinary Python that mypy still checks
and a reader can still follow.  Generating also moves the failure: a spec that
names a keyword no constructor takes now breaks the build instead of breaking a
scenario against a live simulator.

Usage::

    uv run python -m autoware_carla_scenario.authoring.codegen          # write
    uv run python -m autoware_carla_scenario.authoring.codegen --check  # verify
"""

from __future__ import annotations

import argparse
import difflib
import json
import enum
import importlib
import inspect
import sys
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..templating import code_environment
from .registry import (
    ActionSpec,
    ArgumentChoice,
    ConditionSpec,
    FieldSpec,
    action_specs,
    condition_specs,
)

__all__ = ["GenerationError", "diff_against_disk", "generate", "main"]

#: Where the rendered module lands, and the template that renders it.
_HERE = Path(__file__).parent
TEMPLATES_DIR = _HERE / "templates"
OUTPUT_PATH = _HERE / "_builders_generated.py"
TEMPLATE_NAME = "builders.py.jinja"

#: Constructor parameters filled from the :class:`~.compiler.BuildContext`
#: rather than from a field.  A runtime class that grows one of these needs no
#: spec change: the generator sees it in the signature and wires it up.
CTX_PARAMS: dict[str, str] = {
    "client": "ctx.client",
    "tm_port": "ctx.tm_port",
    # The live mapping, not a copy: an ``action_completed`` condition may name
    # an action that is instantiated after it.
    "actions": "ctx.actions",
}

#: Field kinds holding text the document supplies, as opposed to a number or a
#: list ``coerce_params`` has already converted.  A parameter that accepts these
#: *and* something else is handed a ``str``, so that whatever a hand-edited
#: document put there reaches the runtime as the kind of value the editor means.
TEXT_KINDS: frozenset = frozenset({"text", "select", "entity", "action"})

#: Action constructor parameters the dispatcher passes positionally.
ACTION_PARAMS: dict[str, str] = {
    "condition": "condition",
    "timing": "timing",
    "once": "compiled.node.once",
}


class GenerationError(Exception):
    """Raised when a spec and the constructor it targets disagree."""


# ---------------------------------------------------------------------------
# Runtime introspection
# ---------------------------------------------------------------------------


def _import_localns() -> dict[str, Any]:
    """Return the namespace needed to resolve the runtime's own annotations.

    Several constructors annotate a type they only import under
    ``TYPE_CHECKING`` -- ``carla.TrafficLightState``, ``BaseAction`` -- so
    :func:`typing.get_type_hints` cannot resolve them from module globals
    alone.  Handing it the framework's public names, and CARLA, closes that gap
    without asking the runtime to import anything it does not need.
    """
    namespace: dict[str, Any] = {}
    for module_name in ("carla",):
        namespace[module_name] = importlib.import_module(module_name)
    for module_name in (
        "autoware_carla_scenario.actions",
        "autoware_carla_scenario.conditions",
        "autoware_carla_scenario.coordinate",
    ):
        module = importlib.import_module(module_name)
        namespace.update(
            {name: getattr(module, name) for name in dir(module) if name[0] != "_"}
        )
    return namespace


def _resolve_target(target: str) -> tuple[Any, str, str]:
    """Return ``(class, module path, class name)`` for a ``"..mod:Name"`` target."""
    try:
        relative, name = target.split(":", 1)
    except ValueError as exc:
        raise GenerationError(
            f"target {target!r} must be written '<relative module>:<class name>'"
        ) from exc
    absolute = "autoware_carla_scenario" + relative[1:].replace("..", ".")
    try:
        module = importlib.import_module(absolute)
        return getattr(module, name), relative, name
    except (ImportError, AttributeError) as exc:
        raise GenerationError(f"target {target!r} does not resolve: {exc}") from exc


@dataclass(frozen=True)
class _Parameter:
    """One constructor parameter, with its annotation resolved."""

    name: str
    annotation: Any
    has_default: bool
    keyword_only: bool


def _parameters(cls: Any, localns: dict[str, Any]) -> "list[_Parameter]":
    """Return *cls*'s constructor parameters, ``self`` excluded."""
    try:
        hints = typing.get_type_hints(cls.__init__, localns=localns)
    except NameError as exc:  # pragma: no cover -- guarded by _import_localns
        raise GenerationError(
            f"cannot resolve the annotations of {cls.__name__}.__init__: {exc}. "
            f"Add the missing name to codegen._import_localns."
        ) from exc
    signature = inspect.signature(cls.__init__)
    return [
        _Parameter(
            name=name,
            annotation=hints.get(name, parameter.annotation),
            has_default=parameter.default is not inspect.Parameter.empty,
            keyword_only=parameter.kind is inspect.Parameter.KEYWORD_ONLY,
        )
        for name, parameter in list(signature.parameters.items())[1:]
    ]


# ---------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------


def _import_line(obj: Any, name: str, preferred: str) -> str:
    """Return the import statement that brings *name* into a builder.

    The package is preferred over the defining module when it re-exports the
    name, so generated code reads like the hand-written builders it replaces
    (``from ..conditions import SpeedCondition``, not the module it lives in).
    """
    absolute = "autoware_carla_scenario" + preferred[1:].replace("..", ".")
    package = importlib.import_module(absolute)
    if getattr(package, name, None) is obj:
        return f"from {preferred} import {name}"
    defining = getattr(obj, "__module__", "")
    if not defining.startswith("autoware_carla_scenario."):
        raise GenerationError(f"cannot place an import for {name!r} ({defining})")
    return "from .." + defining[len("autoware_carla_scenario.") :] + f" import {name}"


def _enum_expression(
    enum_type: Any,
    source: str,
    options: "tuple[str, ...]",
    where: str,
    package: str,
) -> tuple[str, Optional[str]]:
    """Return the expression lifting *source* into *enum_type*, and its import.

    A select's option values are matched against the enum's member *names* and
    its member *values*, so the registry may spell them either way -- which it
    does: ``ComparisonRule`` is chosen by name, ``TurnDirection`` by value.
    Deciding here rather than per builder is what makes the option lists and
    the enums impossible to drift apart: an option matching neither is an error
    now, where before it only surfaced as a ``KeyError`` at build time.
    """
    if not (isinstance(enum_type, type) and issubclass(enum_type, enum.Enum)):
        # Boost.Python enums (``carla.TrafficLightState``) are plain classes
        # whose members are attributes, so they are read with ``getattr`` and
        # checked with ``hasattr``.  Checking matters most here: this is the
        # mirror the registry cannot import, and an option that no longer names
        # a member would otherwise raise only against a live simulator.
        unknown = [option for option in options if not hasattr(enum_type, option)]
        if unknown:
            raise GenerationError(
                f"{where}: options {unknown} are not members of "
                f"carla.{enum_type.__name__}; the select has drifted from CARLA"
            )
        return (
            f"getattr(carla.{enum_type.__name__}, str({source}))",
            "import carla",
        )

    name = enum_type.__name__
    names = {member.name for member in enum_type}
    values = {str(member.value) for member in enum_type}
    if not options:
        raise GenerationError(f"{where}: enum field has no options to check")
    by_name = all(option in names for option in options)
    by_upper = all(option.upper() in names for option in options)
    by_value = all(option in values for option in options)
    if not (by_name or by_upper or by_value):
        unknown = [
            option
            for option in options
            if option not in names
            and option.upper() not in names
            and option not in values
        ]
        raise GenerationError(
            f"{where}: options {unknown} name neither a member nor a value of "
            f"{name}; the select and the enum have drifted apart"
        )
    _check_unambiguous(enum_type, options, by_name, by_upper, by_value, where)
    if by_name:
        expression = f"{name}[str({source})]"
    elif by_upper:
        expression = f"{name}[str({source}).upper()]"
    else:
        expression = f"{name}(str({source}))"
    return expression, _import_line(enum_type, name, package)


def _check_unambiguous(
    enum_type: Any,
    options: "tuple[str, ...]",
    by_name: bool,
    by_upper: bool,
    by_value: bool,
    where: str,
) -> None:
    """Fail when reading an option as a name and as a value disagree.

    Both readings are accepted because the registry uses both, so an enum whose
    member names and values cross over -- ``LEFT = "right"`` -- would let the
    order the readings are tried in decide the meaning.  Nothing in the tree is
    shaped that way today; this makes sure nothing quietly becomes so.
    """
    readings = []
    if by_name:
        readings.append({option: enum_type[option] for option in options})
    if by_upper:
        readings.append({option: enum_type[option.upper()] for option in options})
    if by_value:
        readings.append({option: enum_type(option) for option in options})
    first = readings[0]
    for other in readings[1:]:
        disagree = [option for option in options if first[option] is not other[option]]
        if disagree:
            raise GenerationError(
                f"{where}: options {disagree} mean different members read as a "
                f"name and read as a value of {enum_type.__name__}"
            )


def _constant_expression(dotted: str) -> tuple[str, str]:
    """Return the expression and import for a ``"..mod:Name.MEMBER"`` constant."""
    holder, _, member = dotted.partition(":")
    name, _, attribute = member.partition(".")
    obj, relative, _ = _resolve_target(f"{holder}:{name}")
    if attribute and not hasattr(obj, attribute):
        raise GenerationError(f"constant {dotted!r} has no attribute {attribute!r}")
    return member, _import_line(obj, name, relative)


# ---------------------------------------------------------------------------
# Rendering model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RenderedCase:
    """One branch of a discriminated argument, as source text."""

    when: str
    expr: str


#: Line width ``ruff-format`` wraps at.  The generator has to render what the
#: formatter would write, or the committed file and the drift check disagree
#: forever: regenerating produces a line the formatter then rewraps, and the
#: check fails again on the same spec with no way to satisfy it.
LINE_LENGTH = 88


@dataclass(frozen=True)
class RenderedChoice:
    """A discriminated argument, as source text."""

    kwarg: str
    discriminator: str
    cases: "tuple[RenderedCase, ...]"
    raise_line: str


@dataclass(frozen=True)
class RenderedBuilder:
    """Everything the template needs to write one builder."""

    name: str
    type_id: str
    node: str
    cls: str
    article: str
    imports: "tuple[str, ...]"
    choices: "tuple[RenderedChoice, ...]"
    args: "tuple[tuple[str, str], ...]"
    uses_params: bool
    needs_actor_assert: bool


def _field_expression(
    field: FieldSpec, parameter: _Parameter, where: str, package: str
) -> tuple[str, Optional[str]]:
    """Return the expression feeding *parameter* from *field*.

    No ``float()`` or ``int()`` appears here on purpose: ``coerce_params`` has
    already converted the value to the type ``FieldSpec.kind`` declares.  The
    hand-written builders converted a second time, and the ``or default``
    idiom they used to do it turned a deliberate ``0.0`` back into the default.
    """
    source = f'params["{field.name}"]'
    annotation = parameter.annotation
    if isinstance(annotation, type) and (
        issubclass(annotation, enum.Enum) or annotation.__module__ == "carla"
    ):
        options = tuple(option.value for option in field.options)
        return _enum_expression(annotation, source, options, where, package)
    if annotation is str or (
        field.kind in TEXT_KINDS and str in typing.get_args(annotation)
    ):
        # ``coerce_params`` converts every numeric kind and leaves the text-like
        # ones as the document wrote them, so this is where a text field's type
        # is settled.  It covers the parameters that accept a union as well as
        # the ones annotated plainly: ``validate_document`` compares a select
        # against its options as ``str(value)``, so a value that is not a string
        # but reads as a valid option passes validation and arrives here as it
        # was written.
        #
        # For an ``entity`` field this is defence rather than conversion --
        # ``_compile_condition`` has already replaced the id with a role name --
        # and it is kept uniform on purpose: a rule that holds for every text
        # kind needs no per-kind exception to stay true when one of those
        # guarantees moves.
        return f"str({source})", None
    return source, None


def _build_one(
    spec: "ConditionSpec | ActionSpec", localns: dict[str, Any]
) -> RenderedBuilder:
    """Turn one spec and its target constructor into a rendered builder."""
    cls, relative, class_name = _resolve_target(spec.target)
    where = f"{spec.type_id} -> {class_name}"
    is_action = isinstance(spec, ActionSpec)

    fields = {field.name: field for field in spec.fields}
    argmap = dict(spec.argmap)
    unknown = set(argmap) - set(fields)
    if unknown:
        raise GenerationError(f"{where}: argmap names absent fields {sorted(unknown)}")
    by_kwarg = {argmap.get(name, name): field for name, field in fields.items()}

    imports = [_import_line(cls, class_name, relative)]
    choices, consumed = _render_choices(spec, fields, where, imports)
    parameters = {parameter.name: parameter for parameter in _parameters(cls, localns)}

    args: list[tuple[str, str]] = []
    uses_params = bool(choices)
    for name, parameter in parameters.items():
        expression = _argument(
            spec, name, parameter, by_kwarg, consumed, choices, where, relative, imports
        )
        if expression is None:
            continue
        if "params[" in expression:
            uses_params = True
        args.append((name, expression))

    _check_required(parameters, args, where)
    _check_all_fields_used(by_kwarg, parameters, consumed, where)

    return RenderedBuilder(
        name=spec.builder,
        type_id=spec.type_id,
        node="Action" if is_action else "Condition",
        cls=class_name,
        article="an" if class_name[0] in "AEIOU" else "a",
        imports=tuple(dict.fromkeys(imports)),
        choices=choices,
        args=tuple(args),
        uses_params=uses_params,
        needs_actor_assert=isinstance(spec, ActionSpec) and spec.actor_required,
    )


def _render_choices(
    spec: "ConditionSpec | ActionSpec",
    fields: dict[str, FieldSpec],
    where: str,
    imports: list[str],
) -> "tuple[tuple[RenderedChoice, ...], set[str]]":
    """Render every :class:`ArgumentChoice`, returning the fields they consume."""
    rendered: list[RenderedChoice] = []
    consumed: set[str] = set()
    for choice in spec.choices:
        discriminator = fields.get(choice.discriminator)
        if discriminator is None or discriminator.kind != "select":
            raise GenerationError(
                f"{where}: choice for {choice.kwarg!r} discriminates on "
                f"{choice.discriminator!r}, which is not a select field"
            )
        consumed.add(choice.discriminator)
        _check_exhaustive(choice, discriminator, where)

        cases: list[RenderedCase] = []
        for case in choice.cases:
            if bool(case.constant) == bool(case.field):
                raise GenerationError(
                    f"{where}: case {case.when!r} must set exactly one of "
                    f"constant and field"
                )
            if case.constant:
                expression, import_line = _constant_expression(case.constant)
                imports.append(import_line)
            else:
                if case.field not in fields:
                    raise GenerationError(
                        f"{where}: case {case.when!r} names absent field "
                        f"{case.field!r}"
                    )
                consumed.add(case.field)
                expression = f'params["{case.field}"]'
            cases.append(RenderedCase(when=json.dumps(case.when), expr=expression))
        rendered.append(
            RenderedChoice(
                kwarg=choice.kwarg,
                discriminator=choice.discriminator,
                cases=tuple(cases),
                raise_line=_raise_line(spec.type_id, choice.discriminator, where),
            )
        )
    return tuple(rendered), consumed


def _raise_line(type_id: str, discriminator: str, where: str) -> str:
    """Render the unreachable ``else`` of a discriminated argument, on one line.

    It has to be one line because that is what ``ruff-format`` leaves a short
    single-argument call as.  A message long enough to be wrapped would be
    rendered one way here and stored another way on disk, so the length is
    checked rather than hoped for.
    """
    line = (
        f'raise ValueError(f"{type_id}: unknown {discriminator} '
        f"{{params['{discriminator}']!r}}\")"
    )
    if len(line) + 8 > LINE_LENGTH:
        raise GenerationError(
            f"{where}: the error message for {discriminator!r} does not fit on "
            f"one line, which the drift check needs. Shorten the type id or the "
            f"field name."
        )
    return line


def _check_exhaustive(
    choice: ArgumentChoice, discriminator: FieldSpec, where: str
) -> None:
    """Fail unless every option of *discriminator* has exactly one case."""
    options = [option.value for option in discriminator.options]
    cased = [case.when for case in choice.cases]
    missing = [option for option in options if option not in cased]
    extra = [when for when in cased if when not in options]
    if missing:
        raise GenerationError(
            f"{where}: {choice.discriminator!r} offers {missing} with no case "
            f"for {choice.kwarg!r}. Every option needs one -- an option that "
            f"falls through is the bug this check exists to catch."
        )
    if extra:
        raise GenerationError(
            f"{where}: cases for {extra} that {choice.discriminator!r} never offers"
        )
    if len(set(cased)) != len(cased):
        raise GenerationError(f"{where}: duplicate cases for {choice.kwarg!r}")


def _argument(
    spec: "ConditionSpec | ActionSpec",
    name: str,
    parameter: _Parameter,
    by_kwarg: dict[str, FieldSpec],
    consumed: set[str],
    choices: "tuple[RenderedChoice, ...]",
    where: str,
    package: str,
    imports: list[str],
) -> Optional[str]:
    """Return the expression for constructor parameter *name*, or ``None``.

    ``None`` means "leave it to its default": a parameter no field feeds and no
    context supplies is one the registry has chosen not to expose, and the
    constructor's own default is the single place its value should live.
    """
    if any(choice.kwarg == name for choice in choices):
        return name
    if name == "label":
        return "compiled.label"
    if name in CTX_PARAMS:
        return CTX_PARAMS[name]
    if isinstance(spec, ActionSpec):
        if name in ACTION_PARAMS:
            return ACTION_PARAMS[name]
        if name == "entity_name" and spec.actor_required:
            return "compiled.actor_role"
    elif isinstance(spec, ConditionSpec):
        child = _children_expression(spec, name, parameter)
        if child is not None:
            return child
    field = by_kwarg.get(name)
    if field is None or field.name in consumed:
        return None
    expression, import_line = _field_expression(field, parameter, where, package)
    if import_line:
        imports.append(import_line)
    return expression


def _children_expression(
    spec: ConditionSpec, name: str, parameter: _Parameter
) -> Optional[str]:
    """Return how a composite or wrapper hands its children to the constructor.

    Which parameter that is comes from the signature rather than a convention:
    ``AndCondition`` calls it ``conditions`` and ``NotCondition`` calls it
    ``condition``, and the shape -- many or exactly one -- is what the spec's
    ``kind`` already says.
    """
    if spec.kind == "composite" and typing.get_origin(parameter.annotation) in (
        list,
        tuple,
        typing.Sequence,
        __import__("collections.abc", fromlist=["Sequence"]).Sequence,
    ):
        return "children"
    if spec.kind == "wrapper" and name == "condition":
        return "children[0]"
    return None


def _check_required(
    parameters: dict[str, _Parameter],
    args: "list[tuple[str, str]]",
    where: str,
) -> None:
    """Fail when a parameter with no default is left unfilled."""
    filled = {name for name, _ in args}
    missing = [
        name
        for name, parameter in parameters.items()
        if not parameter.has_default and name not in filled
    ]
    if missing:
        raise GenerationError(
            f"{where}: nothing supplies the required argument(s) {missing}. "
            f"Add a field, an argmap entry, or hand-write the builder."
        )


def _check_all_fields_used(
    by_kwarg: dict[str, FieldSpec],
    parameters: dict[str, _Parameter],
    consumed: set[str],
    where: str,
) -> None:
    """Fail when a field reaches no constructor argument.

    An editable field the runtime never receives is a control that does
    nothing, which is worse than a missing one: the author sets it and the
    scenario ignores them.
    """
    stranded = [
        field.name
        for kwarg, field in by_kwarg.items()
        if kwarg not in parameters and field.name not in consumed
    ]
    if stranded:
        raise GenerationError(
            f"{where}: field(s) {stranded} reach no constructor argument. "
            f"Fix the argmap, or hand-write the builder."
        )


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def generate() -> str:
    """Return the rendered source of the generated builders module."""
    localns = _import_localns()
    specs: "list[ConditionSpec | ActionSpec]" = [
        *(spec for spec in condition_specs() if spec.target),
        *(spec for spec in action_specs() if spec.target),
    ]
    builders = [_build_one(spec, localns) for spec in specs]
    environment = code_environment(TEMPLATES_DIR)
    return environment.get_template(TEMPLATE_NAME).render(builders=builders)


def diff_against_disk(rendered: str) -> str:
    """Return a unified diff of the committed module against *rendered*.

    An empty string means they agree.  The diff itself is the useful part of a
    drift failure: it says which spec moved, which is what the reader needs.
    """
    current = OUTPUT_PATH.read_text(encoding="utf-8") if OUTPUT_PATH.exists() else ""
    if current == rendered:
        return ""
    return "".join(
        difflib.unified_diff(
            current.splitlines(keepends=True),
            rendered.splitlines(keepends=True),
            fromfile=f"committed/{OUTPUT_PATH.name}",
            tofile=f"generated/{OUTPUT_PATH.name}",
            n=1,
        )
    )


def main(argv: "list[str] | None" = None) -> int:
    """Write the generated module, or check that the file on disk matches."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if the file on disk is not what would be generated",
    )
    args = parser.parse_args(argv)

    rendered = generate()
    if args.check:
        stale = diff_against_disk(rendered)
        if stale:
            print(stale, file=sys.stderr)
            print(
                "Regenerate with:\n"
                "    uv run python -m autoware_carla_scenario.authoring.codegen",
                file=sys.stderr,
            )
            return 1
        return 0
    OUTPUT_PATH.write_text(rendered, encoding="utf-8")
    print(f"wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
