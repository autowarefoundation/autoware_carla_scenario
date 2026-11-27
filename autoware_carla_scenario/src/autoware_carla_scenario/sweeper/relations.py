"""Relations between a sweep's parameters, solved with Z3.

A distribution in ``sweep.constraints`` draws one parameter on its own.  A
scenario often needs its parameters to stand in a relation to each other --
the ego faster than the car it follows, a gap long enough for the speed --
which no distribution of one parameter can say.  ``sweep.relations`` lists
them, each a Python-like expression over the config's keys::

    sweep:
      constraints:
        ego.initial_speed_kmh:
          - {type: uniform, low: 20.0, high: 60.0}
        scenario.lead_speed_kmh:
          - {type: uniform, low: 10.0, high: 50.0}
        scenario.gap_m:
          - {type: normal, mean: 30.0, std: 10.0, low: 5.0}
      relations:
        - ego.initial_speed_kmh > scenario.lead_speed_kmh + 5
        - scenario.gap_m >= (ego.initial_speed_kmh - scenario.lead_speed_kmh) / 3.6 * 2
        - scenario.npc_lanelet_id != ego.spawn_lanelet_id
        - scenario.weather in ["rain", "fog"] or ego.initial_speed_kmh > 30
      samples: 10

The relations apply **after** the sweep is expanded: the map's part -- which
lanelets, what is adjacent, where the stop line is -- is the constraints' and
bindings' (:mod:`.constraints`, :mod:`.bindings`), and each case they expand
to has its values.  A key a relation names is either

* a parameter drawn from a distribution: a real number the relations solve
  for, or
* a value the case holds: one the case sets (a binding's, a route match's,
  an ODD sample's, an override's), else the scenario config's own: a
  constant of that case, a number (an int such as a lanelet ID, or a float),
  a string (an enum's name too) or a bool.

An expression is made of keys (``a.b.c``), numbers, strings, ``True`` and
``False``, ``+ - * /``, ``abs()``, ``min()``, ``max()``, the comparisons
``< <= > >= == !=`` (chained too: ``0 < a < b``), ``in [...]`` and ``not in
[...]`` (a list of literals), and ``and``, ``or``, ``not``; a bool key is a
condition of its own.  ``< <= > >=`` and arithmetic take numbers; ``==``,
``!=`` and ``in`` take two values of one type.  It is parsed, never run.

For each case the relations are handed to the SMT solver Z3
(:class:`RelationSolver`):

1. With the case's values and the distributions' bounds, the relations must
   be satisfiable.  A case for which they are not is dropped (the log says
   how many, and for the first the relations that conflict: Z3's unsat
   core); a sweep with no case left is an error.  Relations over values alone are so a filter of cases.
2. Each drawn parameter's range is narrowed to the values the relations
   leave it (Z3's ``Optimize``, minimising and maximising it over the
   relations' closure).  Narrowing never removes a value a solution has, so
   the draws below are distributed exactly as before, only drawn faster.
3. Values are drawn from the distributions, truncated to the narrowed ranges,
   until a draw meets every relation: the values are the distributions'
   *conditioned on the relations*.  A parameter in no relation is drawn on
   its own as before, so adding a relation moves only the values it relates.
4. If no draw meets them in :data:`MAX_REJECTIONS` tries (an equality, say,
   which a continuous draw meets with probability 0), the parameters are
   drawn one after another, in key order, each from its distribution
   truncated to the range the relations leave it given the values drawn
   before it -- every draw still meets the relations, but the values are no
   longer exactly the conditioned distribution, which a warning says.

Every step is deterministic: the draws come from the same seeds as the
distributions' (:func:`~.distributions.fnv1a`), and Z3 is only asked for
bounds and satisfiability, which do not depend on its search.
"""

from __future__ import annotations

import ast
import enum
import logging
import math
import operator
import random
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable, Mapping, Sequence

from .constraints import _real_number
from .distributions import NormalDistribution

__all__ = [
    "MAX_REJECTIONS",
    "Relation",
    "RelationSolver",
    "RelationsUnsatisfiable",
    "constant_value",
    "parse_relation",
    "parse_relations",
]

logger = logging.getLogger(__name__)

#: How many draws a case tries against the relations before it draws its
#: parameters one after another (step 4 of the module's description).
MAX_REJECTIONS = 1000

#: How many times a parameter drawn on its own (step 4) is drawn again before
#: it takes the value Z3's model gives it.
_SEQUENTIAL_RETRIES = 32

#: The types of a relation's values.
NUM, STR, BOOL = "number", "string", "bool"


class RelationsUnsatisfiable(ValueError):
    """The relations have no solution for a case."""


def constant_value(value: Any) -> Any:
    """*value* (a config value) as a relation's constant: a float for a
    finite number, the string for a string or an enum (its name), the bool;
    ``None`` for anything else (null, a list, a mapping, an infinity), which
    a relation cannot name."""
    if isinstance(value, (bool, str)):
        return value
    if isinstance(value, enum.Enum):
        return value.name
    number = _real_number(value)
    return number if math.isfinite(number) else None


def _type_of(value: Any) -> str:
    if isinstance(value, bool):
        return BOOL
    if isinstance(value, str):
        return STR
    return NUM


# -- Expressions -----------------------------------------------------------------
#
# An expression is a small tree, built from Python's own parser (``ast``, mode
# "eval") and accepted only in the shapes below; nothing of it is ever run by
# Python.  A key's type is only known for a case (a drawn key is a number; a
# constant has its value's type), so the tree is typed twice: once as parsed,
# with what its literals and operators say, and again for each case.  Unary
# minus is ``0 - x``.


@dataclass(frozen=True)
class _Lit:
    value: Any  # Fraction, str or bool


@dataclass(frozen=True)
class _Key:
    key: str


@dataclass(frozen=True)
class _Arith:
    op: str  # "+", "-", "*", "/"
    left: Any
    right: Any


@dataclass(frozen=True)
class _Call:
    name: str  # "abs", "min", "max"
    args: tuple[Any, ...]


@dataclass(frozen=True)
class _Compare:
    op: str  # "<", "<=", ">", ">=", "==", "!="
    left: Any
    right: Any


@dataclass(frozen=True)
class _In:
    item: Any
    options: tuple[_Lit, ...]
    negated: bool


@dataclass(frozen=True)
class _BoolOp:
    op: str  # "and", "or"
    operands: tuple[Any, ...]


@dataclass(frozen=True)
class _Not:
    operand: Any


#: The arithmetic and comparison operators: on floats and on Z3 terms alike.
_ARITH: dict[str, Callable[[Any, Any], Any]] = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
}
_COMPARE: dict[str, Callable[[Any, Any], Any]] = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}
_AST_ARITH: dict[type, str] = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
_AST_COMPARE: dict[type, str] = {
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
    ast.Eq: "==",
    ast.NotEq: "!=",
}
_ORDERING = frozenset(["<", "<=", ">", ">="])
#: A comparison negated: ``not (a < b)`` is ``a >= b``.
_NEGATED = {"<": ">=", "<=": ">", ">": "<=", ">=": "<", "==": "!=", "!=": "=="}
#: A comparison's closure: ``a < b`` relaxed to ``a <= b``; ``!=`` to true.
_RELAXED = {"<": "<=", ">": ">=", "<=": "<=", ">=": ">=", "==": "==", "!=": None}
_FUNCTIONS = {"abs": (1, 1), "min": (2, None), "max": (2, None)}


def _show(tree: Any) -> str:
    """*tree* written back as an expression, for errors."""
    if isinstance(tree, _Lit):
        if isinstance(tree.value, Fraction):
            return f"{float(tree.value):g}"
        return repr(tree.value)
    if isinstance(tree, _Key):
        return tree.key
    if isinstance(tree, (_Arith, _Compare)):
        return f"({_show(tree.left)} {tree.op} {_show(tree.right)})"
    if isinstance(tree, _Call):
        return f"{tree.name}({', '.join(_show(a) for a in tree.args)})"
    if isinstance(tree, _In):
        options = ", ".join(_show(o) for o in tree.options)
        return f"{_show(tree.item)} {'not in' if tree.negated else 'in'} [{options}]"
    if isinstance(tree, _BoolOp):
        return "(" + f" {tree.op} ".join(_show(o) for o in tree.operands) + ")"
    if isinstance(tree, _Not):
        return f"not {_show(tree.operand)}"
    return repr(tree)


def _dotted(node: ast.AST) -> str | None:
    """The key ``a.b.c`` an ``ast.Name``/``ast.Attribute`` chain spells."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _node(node: ast.AST, label: str) -> Any:
    """*node* of an expression, as a tree; refused, naming *label*, if it is
    anything but the shapes the module's description lists."""
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, (bool, str)):
            return _Lit(value)
        if isinstance(value, (int, float)):
            if not math.isfinite(value):
                raise ValueError(f"{label}: {value!r} is not a finite number")
            return _Lit(Fraction(value))
        raise ValueError(f"{label}: {value!r} is not a number, string or bool")
    if isinstance(node, (ast.Name, ast.Attribute)):
        key = _dotted(node)
        if key is None:
            raise ValueError(f"{label}: {ast.unparse(node)!r} is not a key")
        return _Key(key)
    if isinstance(node, ast.UnaryOp):
        operand = _node(node.operand, label)
        if isinstance(node.op, ast.Not):
            return _Not(operand)
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            if isinstance(operand, _Lit) and isinstance(operand.value, Fraction):
                return _Lit(-operand.value)
            return _Arith("-", _Lit(Fraction(0)), operand)
    if isinstance(node, ast.BinOp) and type(node.op) in _AST_ARITH:
        return _Arith(
            _AST_ARITH[type(node.op)], _node(node.left, label), _node(node.right, label)
        )
    if isinstance(node, ast.BoolOp):
        return _BoolOp(
            "and" if isinstance(node.op, ast.And) else "or",
            tuple(_node(v, label) for v in node.values),
        )
    if isinstance(node, ast.Compare):
        return _compare(node, label)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
        if name not in _FUNCTIONS or node.keywords:
            raise ValueError(
                f"{label}: {name}() is not a function a relation takes "
                "(abs, min, max)"
            )
        least, most = _FUNCTIONS[name]
        if len(node.args) < least or (most is not None and len(node.args) > most):
            wanted = (
                f"{least} argument" if least == most else f"{least} or more arguments"
            )
            raise ValueError(f"{label}: {name}() takes {wanted}, got {len(node.args)}")
        return _Call(name, tuple(_node(a, label) for a in node.args))
    raise ValueError(
        f"{label}: {ast.unparse(node)!r} is not something a relation takes "
        "(keys, literals, + - * /, abs/min/max, comparisons, in [...], "
        "and/or/not)"
    )


def _compare(node: ast.Compare, label: str) -> Any:
    """A comparison, chained ones split into an ``and`` of pairs."""
    operands = [node.left, *node.comparators]
    pairs: list[Any] = []
    for i, op in enumerate(node.ops):
        left = _node(operands[i], label)
        right_node = operands[i + 1]
        if isinstance(op, (ast.In, ast.NotIn)):
            if not isinstance(right_node, (ast.List, ast.Tuple, ast.Set)):
                raise ValueError(
                    f"{label}: 'in' takes a list of literals "
                    f'([1, 2] or ["rain", "fog"]), got {ast.unparse(right_node)!r}'
                )
            options = tuple(_node(e, label) for e in right_node.elts)
            for option, element in zip(options, right_node.elts):
                if not isinstance(option, _Lit):
                    raise ValueError(
                        f"{label}: {ast.unparse(element)!r} is not a literal "
                        "(a number, a string, True or False)"
                    )
            if not options:
                raise ValueError(f"{label}: 'in' takes a list of at least one value")
            pairs.append(_In(left, options, isinstance(op, ast.NotIn)))
            continue
        if type(op) not in _AST_COMPARE:
            raise ValueError(
                f"{label}: {type(op).__name__} is no comparison a relation "
                f"takes (use {' '.join(_COMPARE)}, in, not in)"
            )
        pairs.append(_Compare(_AST_COMPARE[type(op)], left, _node(right_node, label)))
    return pairs[0] if len(pairs) == 1 else _BoolOp("and", tuple(pairs))


def _keys(tree: Any) -> frozenset[str]:
    """The keys *tree* names."""
    if isinstance(tree, _Key):
        return frozenset([tree.key])
    if isinstance(tree, _Not):
        return _keys(tree.operand)
    if isinstance(tree, (_Arith, _Compare)):
        return _keys(tree.left) | _keys(tree.right)
    if isinstance(tree, _In):
        return _keys(tree.item)
    if isinstance(tree, _Call):
        return frozenset().union(*(_keys(a) for a in tree.args))
    if isinstance(tree, _BoolOp):
        return frozenset().union(*(_keys(o) for o in tree.operands))
    return frozenset()


def _type(tree: Any, types: Mapping[str, str], label: str) -> str | None:
    """The type of *tree* (:data:`NUM`, :data:`STR`, :data:`BOOL`; ``None``
    for a key whose type *types* does not know yet), refusing, naming
    *label*, an operand of the wrong type."""

    def need(sub: Any, wanted: str, where: str) -> None:
        got = _type(sub, types, label)
        if got is not None and got != wanted:
            whose = (
                f" (the case sets {sub.key} to a {got})"
                if isinstance(sub, _Key)
                else ""
            )
            raise ValueError(
                f"{label}: {where} takes a {wanted}, but {_show(sub)} is a "
                f"{got}{whose}"
            )

    if isinstance(tree, _Lit):
        return _type_of(tree.value)
    if isinstance(tree, _Key):
        return types.get(tree.key)
    if isinstance(tree, _Arith):
        need(tree.left, NUM, f"'{tree.op}'")
        need(tree.right, NUM, f"'{tree.op}'")
        return NUM
    if isinstance(tree, _Call):
        for arg in tree.args:
            need(arg, NUM, f"{tree.name}()")
        return NUM
    if isinstance(tree, _Compare):
        if tree.op in _ORDERING:
            need(tree.left, NUM, f"'{tree.op}'")
            need(tree.right, NUM, f"'{tree.op}'")
        else:
            left = _type(tree.left, types, label)
            right = _type(tree.right, types, label)
            if left is not None and right is not None and left != right:
                raise ValueError(
                    f"{label}: '{tree.op}' compares values of one type, but "
                    f"{_show(tree.left)} is a {left} and {_show(tree.right)} "
                    f"a {right}"
                )
        return BOOL
    if isinstance(tree, _In):
        kinds = {_type_of(o.value) for o in tree.options}
        if len(kinds) > 1:
            raise ValueError(
                f"{label}: the list of 'in' mixes "
                + " and ".join(f"{kind}s" for kind in sorted(kinds))
            )
        need(tree.item, next(iter(kinds)), "'in' over that list")
        return BOOL
    if isinstance(tree, _BoolOp):
        for operand in tree.operands:
            need(operand, BOOL, f"'{tree.op}'")
        return BOOL
    if isinstance(tree, _Not):
        need(tree.operand, BOOL, "'not'")
        return BOOL
    raise TypeError(f"not an expression: {tree!r}")


def _evaluate(tree: Any, values: Mapping[str, Any]) -> Any:
    """*tree* on plain values: a float, a string or a bool."""
    if isinstance(tree, _Lit):
        return float(tree.value) if isinstance(tree.value, Fraction) else tree.value
    if isinstance(tree, _Key):
        return values[tree.key]
    if isinstance(tree, _Arith):
        return _ARITH[tree.op](
            _evaluate(tree.left, values), _evaluate(tree.right, values)
        )
    if isinstance(tree, _Call):
        args = [_evaluate(a, values) for a in tree.args]
        if tree.name == "abs":
            return abs(args[0])
        return min(args) if tree.name == "min" else max(args)
    if isinstance(tree, _Compare):
        return _COMPARE[tree.op](
            _evaluate(tree.left, values), _evaluate(tree.right, values)
        )
    if isinstance(tree, _In):
        found = _evaluate(tree.item, values) in [
            _evaluate(o, values) for o in tree.options
        ]
        return found != tree.negated
    if isinstance(tree, _BoolOp):
        if tree.op == "and":
            return all(_evaluate(o, values) for o in tree.operands)
        return any(_evaluate(o, values) for o in tree.operands)
    if isinstance(tree, _Not):
        return not _evaluate(tree.operand, values)
    raise TypeError(f"not an expression: {tree!r}")


@dataclass(frozen=True)
class Relation:
    """One relation of ``sweep.relations``: its text, where it is, and its
    parsed expression (a condition)."""

    text: str
    label: str
    formula: Any = field(repr=False)
    #: The keys it names.
    keys: frozenset[str] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "keys", _keys(self.formula))

    def check_types(self, types: Mapping[str, str]) -> None:
        """Refuse the relation if, with the keys' *types*, an operand has the
        wrong type or the whole is not a condition.

        Raises:
            ValueError: Naming the relation and the operand.
        """
        kind = _type(self.formula, types, self.label)
        if kind not in (None, BOOL):
            raise ValueError(
                f"{self.label}: {self.text!r} is a {kind}, where a condition "
                "(a comparison, and/or/not, a bool key) is expected"
            )

    def holds(self, values: Mapping[str, Any]) -> bool:
        """Whether the relation holds for *values* (key -> value), every key
        of it among them.  A division by zero makes it fail."""
        try:
            return bool(_evaluate(self.formula, values))
        except ZeroDivisionError:
            return False


def parse_relation(text: Any, label: str) -> Relation:
    """The relation *text* spells (*label* names it in errors, such as
    ``sweep.relations[0]``).

    Raises:
        ValueError: Naming *label*, if *text* is not a string, does not
            parse, or is not a condition made of the shapes the module's
            description lists.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError(
            f"{label} must be a relation such as 'a.x > b.y + 1', got {text!r}"
        )
    try:
        parsed = ast.parse(text.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"{label}: {text!r} does not parse: {exc.msg}") from exc
    relation = Relation(
        text=text.strip(), label=label, formula=_node(parsed.body, label)
    )
    relation.check_types({})
    return relation


def parse_relations(value: Any) -> list[Relation]:
    """``sweep.relations`` (*value*: a list of strings, or ``None``), parsed.

    Raises:
        ValueError: Naming the section or the item, if it is not a list of
            relations.
    """
    if value is None:
        return []
    # A list, or OmegaConf's ListConfig; not a string or a mapping.
    if isinstance(value, (str, bytes, Mapping)) or not hasattr(value, "__iter__"):
        raise ValueError(
            "sweep.relations must be a list of relations "
            f"(- a.x > b.y), got {type(value).__name__}"
        )
    return [
        parse_relation(text, f"sweep.relations[{i}]") for i, text in enumerate(value)
    ]


# -- Z3 ----------------------------------------------------------------------------


def _z3() -> Any:
    """The z3 module, imported on first use: a sweep without relations never
    loads the solver."""
    import z3  # noqa: PLC0415

    return z3


def _exact(value: float | Fraction) -> str:
    """*value* as the exact rational Z3 reads (a float is a dyadic
    rational): no decimal rounding between the draw and the check."""
    fraction = Fraction(value)
    return f"{fraction.numerator}/{fraction.denominator}"


def _z3_value(z3: Any, value: Any) -> Any:
    """A constant as a Z3 value of its type."""
    if isinstance(value, bool):
        return z3.BoolVal(value)
    if isinstance(value, str):
        return z3.StringVal(value)
    return z3.RealVal(_exact(value))


def _z3_term(z3: Any, tree: Any, env: Mapping[str, Any]) -> Any:
    """*tree* as a Z3 expression of its type; *env* maps a key to its Z3
    variable or constant."""
    if isinstance(tree, _Lit):
        return _z3_value(z3, tree.value)
    if isinstance(tree, _Key):
        return env[tree.key]
    if isinstance(tree, _Arith):
        return _ARITH[tree.op](
            _z3_term(z3, tree.left, env), _z3_term(z3, tree.right, env)
        )
    if isinstance(tree, _Call):
        args = [_z3_term(z3, a, env) for a in tree.args]
        if tree.name == "abs":
            return z3.If(args[0] >= 0, args[0], -args[0])
        result = args[0]
        for arg in args[1:]:
            better = arg < result if tree.name == "min" else arg > result
            result = z3.If(better, arg, result)
        return result
    # A condition where a value is compared (a bool key == True).
    return _z3_formula(z3, tree, env)


def _z3_formula(
    z3: Any,
    tree: Any,
    env: Mapping[str, Any],
    *,
    negate: bool = False,
    relax: bool = False,
) -> Any:
    """Condition *tree* as a Z3 boolean, negated if *negate* (pushed down to
    the comparisons), and with every comparison replaced by its closure if
    *relax* (``<`` by ``<=``, ``!=`` dropped): the set it describes then
    holds the original's, and its bounds are the original's bounds, which
    Z3's ``Optimize`` finds without infinitesimals."""
    if isinstance(tree, _Not):
        return _z3_formula(z3, tree.operand, env, negate=not negate, relax=relax)
    if isinstance(tree, _BoolOp):
        parts = [
            _z3_formula(z3, o, env, negate=negate, relax=relax) for o in tree.operands
        ]
        conjunction = (tree.op == "and") != negate
        return z3.And(*parts) if conjunction else z3.Or(*parts)
    if isinstance(tree, _In):
        item = _z3_term(z3, tree.item, env)
        options = [_z3_value(z3, o.value) for o in tree.options]
        if tree.negated == negate:
            return z3.Or(*[item == o for o in options])
        # Only a real's "not one of" is relaxed: a string's or bool's is exact.
        if relax and z3.is_arith(item):
            return z3.BoolVal(True)
        return z3.And(*[item != o for o in options])
    if isinstance(tree, _Compare):
        op = _NEGATED[tree.op] if negate else tree.op
        left = _z3_term(z3, tree.left, env)
        right = _z3_term(z3, tree.right, env)
        # Only a comparison of reals has a closure to relax to; one of
        # strings or bools stays exact.
        if relax and z3.is_arith(left):
            relaxed = _RELAXED[op]
            if relaxed is None:
                return z3.BoolVal(True)
            op = relaxed
        return _COMPARE[op](left, right)
    if isinstance(tree, (_Key, _Lit)):
        # A bool key or literal.
        value = _z3_term(z3, tree, env)
        return z3.Not(value) if negate else value
    raise TypeError(f"not a condition: {tree!r}")


def _as_float(z3: Any, value: Any) -> float | None:
    """A Z3 numeral as a float; ``None`` for anything else (an infinity, an
    infinitesimal, an algebraic number Z3 did not reduce)."""
    if z3.is_int_value(value):
        # An objective's optimum comes back as an integer when it is one.
        return float(value.as_long())
    if z3.is_rational_value(value):
        return float(Fraction(value.numerator_as_long(), value.denominator_as_long()))
    if z3.is_algebraic_value(value):
        return float(Fraction(value.approx(20).as_fraction()))
    return None


def _format_value(value: Any) -> str:
    """A constant as an override writes it, for errors."""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:g}"
    return repr(value)


# -- Solving and drawing -------------------------------------------------------------


@dataclass
class _Case:
    """What the solver keeps for the constants of a case (a signature):
    the relations in Z3, the ranges the relations leave each related key,
    and whether drawing until a draw meets them has failed once."""

    constants: dict[str, Any]
    env: dict[str, Any]
    formulas: list[Any]
    relaxed: list[Any]
    bounds: list[Any]
    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    sequential: bool = False


def _draw_in(distribution: Any, low: float, high: float, rng: random.Random) -> float:
    """A value of *distribution* truncated to ``[low, high]``: *low* if that
    is one point."""
    if not low < high:
        return low
    truncated = distribution.truncated(low, high)
    if (
        isinstance(truncated, NormalDistribution)
        and not truncated.mass() > 1e-300
        and math.isfinite(high - low)
    ):
        # The normal has no probability left there (the interval is far in
        # its tail): uniform over it keeps the draw inside, not at a bound.
        return rng.uniform(low, high)
    return float(truncated.draw(rng))


class RelationSolver:
    """The relations of a sweep over the distributions it draws from.

    *distributions* maps each drawn key to its distribution
    (:mod:`.distributions`); *relations* are ``sweep.relations``, parsed.
    A relation naming no drawn key is a filter of cases.

    Raises:
        ValueError: If a relation compares a drawn key (a number) with a
            literal of another type.
    """

    def __init__(
        self, relations: Sequence[Relation], distributions: Mapping[str, Any]
    ) -> None:
        self.relations = list(relations)
        self.distributions = dict(distributions)
        keys = frozenset().union(*(r.keys for r in self.relations))
        #: The drawn keys some relation names, in key order.
        self.related = sorted(keys & set(self.distributions))
        #: The keys the relations name that no distribution draws: each case
        #: must hold a value for them.
        self.constant_keys = sorted(keys - set(self.distributions))
        drawn = {key: NUM for key in self.distributions}
        for relation in self.relations:
            relation.check_types(drawn)
        self._cases: dict[tuple[Any, ...], _Case] = {}

    def _case(self, values: Mapping[str, Any]) -> _Case:
        """What the solver keeps for *values* (a case's: key -> a float, a
        string or a bool), built and type-checked on first use.

        Raises:
            ValueError: Naming the relation, if a key it names is neither
                drawn nor held by the case, or the case's values give an
                operand the wrong type.
        """
        constants: dict[str, Any] = {}
        for key in self.constant_keys:
            if values.get(key) is None:
                relation = next(r for r in self.relations if key in r.keys)
                raise ValueError(
                    f"{relation.label}: {relation.text!r} names {key}, which "
                    "is neither drawn from a distribution in sweep.constraints "
                    "nor set to a number, a string or a bool by the case or "
                    "the scenario's config"
                )
            constants[key] = values[key]
        signature = tuple((k, _type_of(v), v) for k, v in constants.items())
        case = self._cases.get(signature)
        if case is not None:
            return case
        types = {key: _type_of(value) for key, value in constants.items()}
        types.update({key: NUM for key in self.related})
        for relation in self.relations:
            relation.check_types(types)
        z3 = _z3()
        env = {key: z3.Real(key) for key in self.related}
        env.update({key: _z3_value(z3, value) for key, value in constants.items()})
        bounds: list[Any] = []
        for key in self.related:
            distribution = self.distributions[key]
            if math.isfinite(distribution.low):
                bounds.append(env[key] >= z3.RealVal(_exact(distribution.low)))
            if math.isfinite(distribution.high):
                bounds.append(env[key] <= z3.RealVal(_exact(distribution.high)))
        case = _Case(
            constants=constants,
            env=env,
            formulas=[_z3_formula(z3, r.formula, env) for r in self.relations],
            relaxed=[
                _z3_formula(z3, r.formula, env, relax=True) for r in self.relations
            ],
            bounds=bounds,
        )
        self._check(case)
        case.ranges = self._ranges(case, self.related, {})
        self._cases[signature] = case
        return case

    def check(self, values: Mapping[str, Any] | None = None) -> None:
        """Check the relations have a solution for a case of *values*.

        Raises:
            RelationsUnsatisfiable: Naming the relations that conflict (with
                the case's values and the distributions' bounds), if they
                have none.
            ValueError: If a key a relation names is neither drawn nor held
                by the case, has the wrong type, or Z3 cannot decide.
        """
        self._case(values or {})

    def _check(self, case: _Case) -> None:
        """Refuse *case* when its relations have no solution, naming the
        relations that conflict (Z3's unsat core)."""
        z3 = _z3()
        solver = z3.Solver()
        for i, bound in enumerate(case.bounds):
            solver.assert_and_track(bound, f"bound{i}")
        for i, formula in enumerate(case.formulas):
            solver.assert_and_track(formula, f"relation{i}")
        result = solver.check()
        if result == z3.sat:
            return
        if result == z3.unsat:
            core = {str(name) for name in solver.unsat_core()}
            named = [
                f"{r.label} ({r.text})"
                for i, r in enumerate(self.relations)
                if f"relation{i}" in core
            ]
            given = ", ".join(
                f"{k}={_format_value(v)}" for k, v in sorted(case.constants.items())
            )
            raise RelationsUnsatisfiable(
                "sweep.relations cannot all hold"
                + (" within the distributions' bounds" if self.related else "")
                + (f" (with {given})" if given else "")
                + ": "
                + ("; ".join(named) if named else "the bounds alone conflict")
                + (" conflict" if len(named) > 1 else " cannot hold" if named else "")
            )
        raise ValueError(
            "Z3 could not decide whether sweep.relations can hold "
            f"({solver.reason_unknown()}); simplify them (a product of two "
            "drawn parameters, say, is non-linear)"
        )

    def _ranges(
        self, case: _Case, keys: Sequence[str], fixed: Mapping[str, float]
    ) -> dict[str, tuple[float, float]]:
        """Each of *keys*: the interval the relations leave it, with the
        *fixed* values and within its distribution's bounds.  One
        ``Optimize`` holds the relations; each bound is one objective of its
        own (``push``/``pop``).  Not Z3's box mode, which several objectives
        could share: z3 4.15 answers it with wrong bounds."""
        z3 = _z3()
        opt = z3.Optimize()
        opt.set("timeout", 10_000)
        opt.add(*case.relaxed, *case.bounds)
        for key, value in fixed.items():
            opt.add(case.env[key] == z3.RealVal(_exact(value)))

        def bound(key: str, sense: str) -> float | None:
            opt.push()
            try:
                handle = (
                    opt.minimize(case.env[key])
                    if sense == "min"
                    else opt.maximize(case.env[key])
                )
                # Non-linear, or timed out: no bound of the relations'.
                if opt.check() != z3.sat:
                    return None
                return _as_float(z3, handle.value())
            finally:
                opt.pop()

        ranges: dict[str, tuple[float, float]] = {}
        for key in keys:
            least = bound(key, "min")
            most = bound(key, "max")
            low = self.distributions[key].low
            high = self.distributions[key].high
            low = low if least is None else max(low, least)
            high = high if most is None else min(high, most)
            # Rounding can leave a pinned key's bounds a hair apart, crossed.
            ranges[key] = (low, max(high, low))
        return ranges

    def draw(self, values: Mapping[str, Any], rng: random.Random) -> dict[str, float]:
        """A value of every related key for one draw of a case of *values*
        (the constants), from *rng*: the distributions conditioned on the
        relations (steps 2-4 of the module's description).

        Raises:
            RelationsUnsatisfiable: If the relations have no solution for
                the case.
        """
        case = self._case(values)
        if not self.related:
            return {}
        if not case.sequential:
            for _ in range(MAX_REJECTIONS):
                drawn = {
                    key: _draw_in(self.distributions[key], *case.ranges[key], rng)
                    for key in self.related
                }
                scope = {**case.constants, **drawn}
                if all(r.holds(scope) for r in self.relations):
                    return drawn
            # It fails again for the case's next draws: go straight on.
            case.sequential = True
            logger.warning(
                "No draw met sweep.relations in %d tries (an equality, or a "
                "narrow region); drawing %s one after another within what the "
                "relations leave each, so their values are not exactly the "
                "distributions conditioned on the relations",
                MAX_REJECTIONS,
                ", ".join(self.related),
            )
        return self._draw_sequentially(case, rng)

    def _draw_sequentially(self, case: _Case, rng: random.Random) -> dict[str, float]:
        """Step 4: each related key in turn, drawn within the range the
        relations leave it given the keys drawn before it; one solver, a
        value tried at a time (``push``/``pop``)."""
        z3 = _z3()
        solver = z3.Solver()
        solver.add(*case.formulas, *case.bounds)
        fixed: dict[str, float] = {}
        for key in self.related:
            low, high = (
                case.ranges[key] if not fixed else self._ranges(case, [key], fixed)[key]
            )
            for _ in range(_SEQUENTIAL_RETRIES):
                candidate = _draw_in(self.distributions[key], low, high, rng)
                solver.push()
                solver.add(case.env[key] == z3.RealVal(_exact(candidate)))
                fits = solver.check() == z3.sat
                solver.pop()
                if fits:
                    value = candidate
                    break
            else:
                solver.check()
                found = _as_float(
                    z3, solver.model().eval(case.env[key], model_completion=True)
                )
                value = low if found is None else found
            fixed[key] = value
            solver.add(case.env[key] == z3.RealVal(_exact(value)))
        return fixed
