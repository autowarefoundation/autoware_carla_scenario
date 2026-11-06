"""A config's true/false switch, read strictly: a boolean or its spelling.

A Hydra override (``rerun.enabled=false``, quoted or not) or an
``${oc.env:VAR,false}`` may hand a switch as text, so ``bool()`` of it read
``"false"`` as true, and any typo (``"ture"``) as true as well (#1753).
:func:`check_flag` takes a boolean, the spellings of :data:`TRUE_TEXT` and
:data:`FALSE_TEXT` in any case and with blanks around them, and the integers
1 and 0, and refuses anything else (a float, a list, null, other text) with a
one-line ``ValueError`` naming the key.  The ``launcher.*`` switches of a
SLURM run (``slurm.settings``) and the run's ``rerun``, ``t4``, ``t4scene``
and ``background_traffic`` switches, ``autoware.launcher.rosdep`` and
``metrics.heavy`` (``examples.run``, #1761) are read with it, as
are ``sweep.odd_sample``'s ``tire_friction`` and ``visibility``
(``odd.sampler``), a route search's ``ego_goal`` (``route.model``, #1756),
the traffic options' (``traffic.sumo.config``, ``traffic.fidelity``),
the bool vehicle parameters (``traffic.parameters``) and the ``driver``
switches (``send_ground_truth``, ``send_renderer_data``,
``send_actor_ground_truth``, ``t4_scene``; ``driver.base``, #1760).
"""

from __future__ import annotations

from .value_text import huge_integer

__all__ = ["FALSE_TEXT", "TRUE_TEXT", "check_flag"]

#: The spellings of a true switch in a string, after ``strip().lower()``, as
#: authoring.registry's TRUTHY_VALUES (repeated here, not imported, so that
#: this module stays free of the authoring package).
TRUE_TEXT = frozenset({"1", "true", "on", "yes"})
#: The spellings of a false switch, as authoring.registry's FALSY_VALUES.
FALSE_TEXT = frozenset({"0", "false", "off", "no"})


def check_flag(key: str, value: bool) -> bool:
    """*value*, a config's true/false switch as it holds it, as a ``bool``.

    A boolean is itself; a string is true or false when, stripped and in
    lower case, it is one of :data:`TRUE_TEXT` or :data:`FALSE_TEXT`
    (``"false"``, ``" Yes "``, ``"0"``); an integer when it is 1 or 0.

    Raises:
        ValueError: naming *key*, for anything else (``"ture"``, ``2``,
            ``0.0``, null, a list).
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in TRUE_TEXT:
            return True
        if text in FALSE_TEXT:
            return False
    elif isinstance(value, int):
        # Compared as a number: str() of an int of more digits than
        # sys.get_int_max_str_digits() raises.
        if value == 1:
            return True
        if value == 0:
            return False
        try:
            text = repr(value)
        except ValueError:
            text = huge_integer(value)
        raise ValueError(f"{key} must be true or false, got {text}")
    raise ValueError(f"{key} must be true or false, got {value!r}")
