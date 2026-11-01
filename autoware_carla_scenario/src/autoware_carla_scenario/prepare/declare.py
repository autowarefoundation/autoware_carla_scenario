"""``@need_prepare``: a class says which backends must run before it does.

Any class can carry it: an action, a scenario, a renderer.  The preparer
(:class:`~autoware_carla_scenario.prepare.pool.BackendPool`) reads it off the
scenarios of a queue before CARLA starts (:func:`scenario_backends`) and off
the actions a scenario's ``setup()`` registered before its run ticks
(:func:`action_backends`).
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, ClassVar, Iterable, Mapping, Protocol, TypeVar

from . import Backend

__all__ = [
    "BackendFactory",
    "action_backends",
    "need_prepare",
    "required_backends",
    "scenario_backends",
]

#: The class attribute the declaration is kept in.
_ATTRIBUTE = "__need_prepare__"

_T = TypeVar("_T", bound=type)


class BackendFactory(Protocol):
    """A backend class: named, and built from its settings."""

    #: The backend's name, which its settings are found under
    #: (``backends.<name>`` of the run's config).
    backend_name: ClassVar[str]

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any]) -> Backend:
        """The backend, configured by *settings* (missing keys: defaults)."""
        ...


def need_prepare(*backends: type) -> Callable[[_T], _T]:
    """Declare the backends the decorated class needs running before a run.

    A subclass needs what its bases do, and what it declares itself.

    Args:
        backends: Backend classes (:class:`BackendFactory`).

    Raises:
        TypeError: If one is not a backend class.
    """
    for backend in backends:
        name = getattr(backend, "backend_name", None)
        if not isinstance(backend, type) or not isinstance(name, str) or not name:
            raise TypeError(
                f"need_prepare() takes backend classes with a backend_name, "
                f"not {backend!r}"
            )
        if not callable(getattr(backend, "from_settings", None)):
            raise TypeError(f"{backend.__name__} has no from_settings()")

    def decorate(cls: _T) -> _T:
        merged = tuple(dict.fromkeys((*required_backends(cls), *backends)))
        setattr(cls, _ATTRIBUTE, merged)
        return cls

    return decorate


def required_backends(item: object) -> tuple[type, ...]:
    """The backend classes *item* (a class, or an instance of one) needs."""
    cls = item if isinstance(item, type) else type(item)
    declared: tuple[type, ...] = ()
    # Bases first, so a declaration on a subclass adds to theirs.
    for klass in reversed(cls.__mro__):
        own = klass.__dict__.get(_ATTRIBUTE, ())
        declared = tuple(dict.fromkeys((*declared, *own)))
    return declared


def action_backends(actions: Iterable[object]) -> tuple[type, ...]:
    """The backend classes the *actions* need, each once, in order."""
    found: dict[type, None] = {}
    for action in actions:
        found.update(dict.fromkeys(required_backends(action)))
    return tuple(found)


def scenario_backends(scenario: object) -> tuple[type, ...]:
    """What *scenario* needs, as far as it is known before ``setup()``.

    Its class's declaration, and, for a declarative scenario, the
    declarations of the classes its document's actions build.
    """
    found: dict[type, None] = dict.fromkeys(required_backends(scenario))
    compiled = getattr(scenario, "compiled", None)
    for action in getattr(compiled, "actions", ()):
        target = getattr(getattr(action, "spec", None), "target", None)
        if isinstance(target, str):
            found.update(dict.fromkeys(required_backends(_resolve(target))))
    return tuple(found)


def _resolve(target: str) -> type:
    """The class an editor card's ``"..module:Name"`` target names."""
    relative, name = target.split(":", 1)
    module = importlib.import_module(
        "autoware_carla_scenario" + relative[1:].replace("..", ".")
    )
    resolved = getattr(module, name)
    if not isinstance(resolved, type):
        raise TypeError(f"{target!r} is not a class")
    return resolved
