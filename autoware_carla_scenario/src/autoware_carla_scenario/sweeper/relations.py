"""Relations between the parameters a sweep draws, solved with Z3.

A distribution in ``sweep.constraints`` draws one parameter on its own.  A
scenario often needs its parameters to stand in a relation to each other --
the ego faster than the car it follows, a gap long enough for the speed --
which no distribution of one parameter can say.  ``sweep.relations`` lists
them, each a Python-like expression over the parameters' keys::

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
      samples: 10

An expression is made of the keys (``a.b.c``), numbers, ``+ - * /``,
``abs()``, ``min()``, ``max()``, the comparisons ``< <= > >= == !=`` (chained
too: ``0 < a < b``), and ``and``, ``or``, ``not``.  A key is a parameter drawn
from a distribution, or one the case sets to a number (a binding's value, a
route match's, an override's): that one is a constant of the case.

For each case the relations are handed to the SMT solver Z3
(:class:`RelationSolver`):

1. With the distributions' bounds, the relations must be satisfiable.  A case
   for which they are not is dropped, with a warning naming the relations
   that conflict (Z3's unsat core); a sweep with no case left is an error.
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
import logging
import math
import operator
import random
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable, Mapping, Sequence

from .distributions import NormalDistribution, UniformDistribution

__all__ = [
    "MAX_REJECTIONS",
    "Relation",
    "RelationSolver",
    "RelationsUnsatisfiable",
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


class RelationsUnsatisfiable(ValueError):
    """The relations have no solution within the distributions' bounds."""


# -- Expressions -----------------------------------------------------------------
#
# An expression is a small tree, built from Python's own parser (``ast``, mode
# "eval") and accepted only in the shapes below; nothing of it is ever run by
# Python.  Each node evaluates to a float (a term) or a bool (a formula), both
# on plain floats -- to check a draw -- and as a Z3 expression.


@dataclass(frozen=True)
class _Num:
    value: Fraction


@dataclass(frozen=True)
class _Key:
    key: str


@dataclass(frozen=True)
class _Neg:
    operand: Any


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
class _BoolOp:
    op: str  # "and", "or"
    operands: tuple[Any, ...]


@dataclass(frozen=True)
class _Not:
    operand: Any


_TERMS = (_Num, _Key, _Neg, _Arith, _Call)
_FORMULAS = (_Compare, _BoolOp, _Not)

_ARITH_OPS: dict[type, str] = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
}
_COMPARE_OPS: dict[type, str] = {
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
    ast.Eq: "==",
    ast.NotEq: "!=",
}
#: A comparison negated: ``not (a < b)`` is ``a >= b``.
_NEGATED = {"<": ">=", "<=": ">", ">": "<=", ">=": "<", "==": "!=", "!=": "=="}
#: A comparison's closure: ``a < b`` relaxed to ``a <= b``; ``!=`` to true.
_RELAXED = {"<": "<=", ">": ">=", "<=": "<=", ">=": ">=", "==": "==", "!=": None}
_PY_COMPARE: dict[str, Callable[[float, float], bool]] = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}
_FUNCTIONS = {"abs": (1, 1), "min": (2, None), "max": (2, None)}


@dataclass(frozen=True)
class Relation:
    """One relation of ``sweep.relations``: its text, where it is, and its
    parsed expression (a formula)."""

    text: str
    label: str
    formula: Any = field(repr=False)
    keys: frozenset[str] = frozenset()

    def holds(self, values: Mapping[str, float]) -> bool:
        """Whether the relation holds for *values* (key -> value), every key
        of it among them.  A division by zero makes it fail."""
        try:
            return bool(_evaluate(self.formula, values))
        except ZeroDivisionError:
            return False


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


def _term(node: ast.AST, label: str) -> Any:
    """The term (a number-valued expression) *node* is."""
    tree = _node(node, label)
    if not isinstance(tree, _TERMS):
        raise ValueError(
            f"{label}: {ast.unparse(node)!r} is a condition where a number is "
            "expected"
        )
    return tree


def _formula(node: ast.AST, label: str) -> Any:
    """The formula (a true-or-false expression) *node* is."""
    tree = _node(node, label)
    if not isinstance(tree, _FORMULAS):
        raise ValueError(
            f"{label}: {ast.unparse(node)!r} is a number where a condition "
            "(a comparison, and/or/not) is expected"
        )
    return tree


def _node(node: ast.AST, label: str) -> Any:
    """*node* of an expression, as a tree; refused, naming *label*, if it is
    anything but the shapes the module's description lists."""
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label}: {value!r} is not a number")
        if not math.isfinite(value):
            raise ValueError(f"{label}: {value!r} is not a finite number")
        return _Num(Fraction(value))
    if isinstance(node, (ast.Name, ast.Attribute)):
        key = _dotted(node)
        if key is None:
            raise ValueError(f"{label}: {ast.unparse(node)!r} is not a key")
        if key in ("True", "False", "None"):
            raise ValueError(f"{label}: {key} is not a number")
        return _Key(key)
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return _Neg(_term(node.operand, label))
        if isinstance(node.op, ast.UAdd):
            return _term(node.operand, label)
        if isinstance(node.op, ast.Not):
            return _Not(_formula(node.operand, label))
    if isinstance(node, ast.BinOp) and type(node.op) in _ARITH_OPS:
        return _Arith(
            _ARITH_OPS[type(node.op)],
            _term(node.left, label),
            _term(node.right, label),
        )
    if isinstance(node, ast.BoolOp):
        return _BoolOp(
            "and" if isinstance(node.op, ast.And) else "or",
            tuple(_formula(v, label) for v in node.values),
        )
    if isinstance(node, ast.Compare):
        terms = [_term(node.left, label)] + [_term(c, label) for c in node.comparators]
        pairs: list[Any] = []
        for i, op in enumerate(node.ops):
            if type(op) not in _COMPARE_OPS:
                raise ValueError(
                    f"{label}: {type(op).__name__} is no comparison a relation "
                    f"takes (use {' '.join(_PY_COMPARE)})"
                )
            pairs.append(_Compare(_COMPARE_OPS[type(op)], terms[i], terms[i + 1]))
        return pairs[0] if len(pairs) == 1 else _BoolOp("and", tuple(pairs))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
        if name not in _FUNCTIONS or node.keywords:
            raise ValueError(
                f"{label}: {name}() is not a function a relation takes "
                "(abs, min, max)"
            )
        least, most = _FUNCTIONS[name]
        if len(node.args) < least or (most is not None and len(node.args) > most):
            raise ValueError(
                f"{label}: {name}() takes "
                + (f"{least} argument" if least == most else f"{least} or more arguments")
                + f", got {len(node.args)}"
            )
        return _Call(name, tuple(_term(a, label) for a in node.args))
    raise ValueError(
        f"{label}: {ast.unparse(node)!r} is not something a relation takes "
        "(keys, numbers, + - * /, abs/min/max, comparisons, and/or/not)"
    )


def _keys(tree: Any) -> frozenset[str]:
    """The keys *tree* names."""
    if isinstance(tree, _Key):
        return frozenset([tree.key])
    if isinstance(tree, (_Neg, _Not)):
        return _keys(tree.operand)
    if isinstance(tree, (_Arith, _Compare)):
        return _keys(tree.left) | _keys(tree.right)
    if isinstance(tree, _Call):
        return frozenset().union(*(_keys(a) for a in tree.args))
    if isinstance(tree, _BoolOp):
        return frozenset().union(*(_keys(o) for o in tree.operands))
    return frozenset()


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
    formula = _formula(parsed.body, label)
    return Relation(
        text=text.strip(), label=label, formula=formula, keys=_keys(formula)
    )


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


def _evaluate(tree: Any, values: Mapping[str, float]) -> Any:
    """*tree* on plain floats: a float for a term, a bool for a formula."""
    if isinstance(tree, _Num):
        return float(tree.value)
    if isinstance(tree, _Key):
        return values[tree.key]
    if isinstance(tree, _Neg):
        return -_evaluate(tree.operand, values)
    if isinstance(tree, _Arith):
        left = _evaluate(tree.left, values)
        right = _evaluate(tree.right, values)
        if tree.op == "+":
            return left + right
        if tree.op == "-":
            return left - right
        if tree.op == "*":
            return left * right
        return left / right
    if isinstance(tree, _Call):
        args = [_evaluate(a, values) for a in tree.args]
        if tree.name == "abs":
            return abs(args[0])
        return min(args) if tree.name == "min" else max(args)
    if isinstance(tree, _Compare):
        return _PY_COMPARE[tree.op](
            _evaluate(tree.left, values), _evaluate(tree.right, values)
        )
    if isinstance(tree, _BoolOp):
        if tree.op == "and":
            return all(_evaluate(o, values) for o in tree.operands)
        return any(_evaluate(o, values) for o in tree.operands)
    if isinstance(tree, _Not):
        return not _evaluate(tree.operand, values)
    raise TypeError(f"not an expression: {tree!r}")


# -- Z3 ----------------------------------------------------------------------------


def _z3() -> Any:
    """The z3 module, imported on first use: a sweep without relations never
    loads the solver."""
    import z3  # noqa: PLC0415

    return z3


def _z3_term(tree: Any, env: Mapping[str, Any]) -> Any:
    """Term *tree* as a Z3 real expression; *env* maps a key to its Z3
    variable or constant."""
    z3 = _z3()
    if isinstance(tree, _Num):
        return z3.RealVal(f"{tree.value.numerator}/{tree.value.denominator}")
    if isinstance(tree, _Key):
        return env[tree.key]
    if isinstance(tree, _Neg):
        return -_z3_term(tree.operand, env)
    if isinstance(tree, _Arith):
        left = _z3_term(tree.left, env)
        right = _z3_term(tree.right, env)
        if tree.op == "+":
            return left + right
        if tree.op == "-":
            return left - right
        if tree.op == "*":
            return left * right
        return left / right
    if isinstance(tree, _Call):
        args = [_z3_term(a, env) for a in tree.args]
        if tree.name == "abs":
            return z3.If(args[0] >= 0, args[0], -args[0])
        result = args[0]
        for arg in args[1:]:
            if tree.name == "min":
                result = z3.If(arg < result, arg, result)
            else:
                result = z3.If(arg > result, arg, result)
        return result
    raise TypeError(f"not a term: {tree!r}")


def _z3_compare(op: str, left: Any, right: Any) -> Any:
    if op == "<":
        return left < right
    if op == "<=":
        return left <= right
    if op == ">":
        return left > right
    if op == ">=":
        return left >= right
    if op == "==":
        return left == right
    return left != right


def _z3_formula(
    tree: Any,
    env: Mapping[str, Any],
    *,
    negate: bool = False,
    relax: bool = False,
) -> Any:
    """Formula *tree* as a Z3 boolean, negated if *negate* (pushed down to
    the comparisons), and with every comparison replaced by its closure if
    *relax* (``<`` by ``<=``, ``!=`` dropped): the set it describes then
    holds the original's, and its bounds are the original's bounds, which
    Z3's ``Optimize`` finds without infinitesimals."""
    z3 = _z3()
    if isinstance(tree, _Not):
        return _z3_formula(tree.operand, env, negate=not negate, relax=relax)
    if isinstance(tree, _BoolOp):
        parts = [
            _z3_formula(o, env, negate=negate, relax=relax) for o in tree.operands
        ]
        conjunction = (tree.op == "and") != negate
        return z3.And(*parts) if conjunction else z3.Or(*parts)
    if isinstance(tree, _Compare):
        op = _NEGATED[tree.op] if negate else tree.op
        if relax:
            relaxed = _RELAXED[op]
            if relaxed is None:
                return z3.BoolVal(True)
            op = relaxed
        return _z3_compare(op, _z3_term(tree.left, env), _z3_term(tree.right, env))
    raise TypeError(f"not a formula: {tree!r}")


def _as_float(value: Any) -> float | None:
    """A Z3 numeral as a float; ``None`` for anything else (an infinity, an
    infinitesimal, an algebraic number Z3 did not reduce)."""
    z3 = _z3()
    if z3.is_int_value(value):
        # An objective's optimum comes back as an integer when it is one.
        return float(value.as_long())
    if z3.is_rational_value(value):
        return float(Fraction(value.numerator_as_long(), value.denominator_as_long()))
    if z3.is_algebraic_value(value):
        return float(Fraction(value.approx(20).as_fraction()))
    return None


# -- Drawing -----------------------------------------------------------------------


def _bounds(distribution: Any) -> tuple[float, float]:
    """The interval *distribution* draws from (infinite for an untruncated
    normal)."""
    return distribution.low, distribution.high


def _narrowed(distribution: Any, low: float, high: float) -> Any:
    """*distribution* truncated to ``[low, high]``; ``None`` if that is a
    single point (or nothing)."""
    if not low < high:
        return None
    if isinstance(distribution, UniformDistribution):
        return UniformDistribution(low=low, high=high)
    if isinstance(distribution, NormalDistribution):
        return NormalDistribution(
            mean=distribution.mean, std=distribution.std, low=low, high=high
        )
    raise TypeError(f"not a distribution: {distribution!r}")


def _draw_in(distribution: Any, low: float, high: float, rng: random.Random) -> float:
    """A value of *distribution* truncated to ``[low, high]``: *low* if that
    is one point."""
    narrowed = _narrowed(distribution, low, high)
    if narrowed is None:
        return low
    if (
        isinstance(narrowed, NormalDistribution)
        and not narrowed.mass() > 1e-300
        and math.isfinite(high - low)
    ):
        # The normal has no mass left there (the interval is far in its
        # tail): uniform over it keeps the draw inside, not at a bound.
        return rng.uniform(low, high)
    return narrowed.draw(rng)


class RelationSolver:
    """The relations of a sweep over the distributions it draws from.

    *distributions* maps each drawn key to its distribution
    (:mod:`.distributions`); *relations* are ``sweep.relations``, parsed.
    """

    def __init__(
        self, relations: Sequence[Relation], distributions: Mapping[str, Any]
    ) -> None:
        self.relations = list(relations)
        self.distributions = dict(distributions)
        #: The drawn keys some relation names, in key order.
        self.related = sorted(
            set().union(*(r.keys for r in self.relations)) & set(self.distributions)
        )
        for relation in self.relations:
            if not relation.keys & set(self.distributions):
                raise ValueError(
                    f"{relation.label}: {relation.text!r} names no parameter "
                    "drawn from a distribution in sweep.constraints; a relation "
                    "relates drawn parameters (to each other, or to values the "
                    "case sets)"
                )
        self._ranges: dict[tuple[tuple[str, float], ...], dict[str, tuple[float, float]]] = {}
        self._warned_sequential = False

    def constants(self) -> frozenset[str]:
        """The keys the relations name that no distribution draws: each case
        must set them to a number."""
        names = set().union(*(r.keys for r in self.relations))
        return frozenset(names - set(self.distributions))

    def _constants_of(self, case: Mapping[str, float]) -> dict[str, float]:
        """The constants the relations need, from *case* (key -> number).

        Raises:
            ValueError: Naming the relation, if a key it names is neither
                drawn nor set to a number by the case.
        """
        values: dict[str, float] = {}
        for relation in self.relations:
            for key in sorted(relation.keys - set(self.distributions)):
                if key not in case:
                    raise ValueError(
                        f"{relation.label}: {relation.text!r} names {key}, which "
                        "is neither drawn from a distribution in sweep.constraints "
                        "nor set to a number by the case (a binding, the route "
                        "or an override); draw it, or set it"
                    )
                values[key] = case[key]
        return values

    def _env(self, constants: Mapping[str, float], fixed: Mapping[str, float]) -> dict[str, Any]:
        z3 = _z3()
        env: dict[str, Any] = {}
        for key in self.related:
            env[key] = (
                z3.RealVal(_exact(fixed[key])) if key in fixed else z3.Real(key)
            )
        for key, value in constants.items():
            env[key] = z3.RealVal(_exact(value))
        return env

    def _bound_constraints(self, env: Mapping[str, Any], fixed: Mapping[str, float]) -> list[Any]:
        z3 = _z3()
        out: list[Any] = []
        for key in self.related:
            if key in fixed:
                continue
            low, high = _bounds(self.distributions[key])
            if math.isfinite(low):
                out.append(env[key] >= z3.RealVal(_exact(low)))
            if math.isfinite(high):
                out.append(env[key] <= z3.RealVal(_exact(high)))
        return out

    def check(self, case: Mapping[str, float] | None = None) -> None:
        """Check the relations have a solution for *case* (its constants).

        Raises:
            RelationsUnsatisfiable: Naming the relations that conflict (with
                the distributions' bounds), if they have none.
            ValueError: If a key a relation names is neither drawn nor set
                by *case*, or Z3 cannot decide (``unknown``).
        """
        constants = self._constants_of(case or {})
        self._solve(constants, {})

    def _solve(self, constants: Mapping[str, float], fixed: Mapping[str, float]) -> Any:
        """A Z3 model of the relations with *constants* and *fixed* values.

        Raises:
            RelationsUnsatisfiable: If there is none, naming the unsat core.
        """
        z3 = _z3()
        env = self._env(constants, fixed)
        solver = z3.Solver()
        for i, bound in enumerate(self._bound_constraints(env, fixed)):
            solver.assert_and_track(bound, f"bound{i}")
        for i, relation in enumerate(self.relations):
            solver.assert_and_track(_z3_formula(relation.formula, env), f"relation{i}")
        result = solver.check()
        if result == z3.sat:
            return solver.model()
        if result == z3.unsat:
            core = {str(name) for name in solver.unsat_core()}
            named = [
                f"{r.label} ({r.text})"
                for i, r in enumerate(self.relations)
                if f"relation{i}" in core
            ]
            given = ", ".join(f"{k}={v:g}" for k, v in sorted({**constants, **fixed}.items()))
            raise RelationsUnsatisfiable(
                "sweep.relations cannot all hold within the distributions' "
                "bounds"
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

    def _ranges_for(
        self, constants: Mapping[str, float], fixed: Mapping[str, float]
    ) -> dict[str, tuple[float, float]]:
        """Each related key not in *fixed*: the interval the relations leave
        it, within its distribution's bounds."""
        z3 = _z3()
        env = self._env(constants, fixed)
        relaxed = [
            _z3_formula(r.formula, env, relax=True) for r in self.relations
        ] + self._bound_constraints(env, fixed)
        ranges: dict[str, tuple[float, float]] = {}
        for key in self.related:
            if key in fixed:
                continue
            low, high = _bounds(self.distributions[key])
            for sense in ("min", "max"):
                opt = z3.Optimize()
                opt.set("timeout", 10_000)
                opt.add(*relaxed)
                handle = opt.minimize(env[key]) if sense == "min" else opt.maximize(env[key])
                if opt.check() != z3.sat:
                    # Non-linear, or timed out: the distribution's own bound.
                    continue
                bound = _as_float(handle.value())
                if bound is None:
                    continue
                if sense == "min":
                    low = max(low, bound)
                else:
                    high = min(high, bound)
            # Rounding can leave a pinned key's bounds a hair apart, crossed.
            ranges[key] = (low, max(high, low))
        return ranges

    def draw(
        self,
        case: Mapping[str, float],
        rng: random.Random,
    ) -> dict[str, float]:
        """A value of every related key for one draw of *case* (its numeric
        values: the constants), from *rng*: the distributions conditioned on
        the relations (steps 2-4 of the module's description).

        Raises:
            RelationsUnsatisfiable: If the relations have no solution for
                *case*.
        """
        constants = self._constants_of(case)
        signature = tuple(sorted(constants.items()))
        ranges = self._ranges.get(signature)
        if ranges is None:
            self._solve(constants, {})
            ranges = self._ranges_for(constants, {})
            self._ranges[signature] = ranges
        for _ in range(MAX_REJECTIONS):
            values = {
                key: _draw_in(self.distributions[key], *ranges[key], rng)
                for key in self.related
            }
            scope = {**constants, **values}
            if all(r.holds(scope) for r in self.relations):
                return values
        if not self._warned_sequential:
            self._warned_sequential = True
            logger.warning(
                "No draw met sweep.relations in %d tries (an equality, or a "
                "narrow region); drawing %s one after another within what the "
                "relations leave each, so their values are not exactly the "
                "distributions conditioned on the relations",
                MAX_REJECTIONS,
                ", ".join(self.related),
            )
        return self._draw_sequentially(constants, rng)

    def _draw_sequentially(
        self, constants: Mapping[str, float], rng: random.Random
    ) -> dict[str, float]:
        """Step 4: each related key in turn, drawn within the range the
        relations leave it given the keys drawn before it."""
        fixed: dict[str, float] = {}
        for key in self.related:
            low, high = self._ranges_for(constants, fixed)[key]
            value: float | None = None
            for _ in range(_SEQUENTIAL_RETRIES):
                candidate = _draw_in(self.distributions[key], low, high, rng)
                try:
                    self._solve(constants, {**fixed, key: candidate})
                except RelationsUnsatisfiable:
                    continue
                value = candidate
                break
            if value is None:
                model = self._solve(constants, fixed)
                z3 = _z3()
                found = _as_float(model.eval(z3.Real(key), model_completion=True))
                value = low if found is None else found
            fixed[key] = value
        return fixed


def _exact(value: float) -> str:
    """*value* as the exact rational Z3 reads (a float is a dyadic
    rational): no decimal rounding between the draw and the check."""
    fraction = Fraction(value)
    return f"{fraction.numerator}/{fraction.denominator}"
