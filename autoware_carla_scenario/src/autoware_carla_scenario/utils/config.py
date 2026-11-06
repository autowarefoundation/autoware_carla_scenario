"""Validation shared by the configuration dataclasses.

Every config group in this package is built from a plain mapping -- a Hydra
node, usually -- and every one of them has to answer the same question about a
key it does not define.  The answer is always "refuse it", and this is where
that is written once.

Lives under :mod:`..utils` because both callers have to reach it and neither can
reach the other: :mod:`..driver.base` pulls in numpy and a CARLA sensor config,
while :mod:`..traffic.config` must stay importable by the editor and the config
layer.  This module imports nothing but the standard library.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Mapping, Optional, Sequence

__all__ = ["checked_options", "checked_str_list"]


def checked_options(
    config_cls: type, mapping: Mapping[str, Any], ignore: Optional[set] = None
) -> dict:
    """Return *mapping* as a dict, rejecting keys *config_cls* does not define.

    A silently dropped key is the failure mode this guards against: a typo in a
    YAML override would otherwise leave the default in place with no indication
    that the override did nothing.

    Args:
        config_cls: The dataclass the mapping is meant to build.
        mapping: The values to check.
        ignore: Keys that are neither fields nor errors -- a config that takes
            an alias it converts itself names it here, and it is dropped from
            the result rather than passed on.

    Returns:
        The mapping as a plain dict, without the ignored keys.

    Raises:
        ValueError: If *mapping* holds a key the config does not define.
    """
    known = {field.name for field in fields(config_cls)}
    skip = ignore or set()
    unknown = sorted(set(mapping) - known - skip)
    if unknown:
        raise ValueError(
            f"Unknown {config_cls.__name__} key(s): {unknown}. "
            f"Known keys: {sorted(known)}"
        )
    return {key: value for key, value in mapping.items() if key not in skip}


def checked_str_list(key: str, value: Any) -> list[str]:
    """*value*, a list setting such as ``traffic.options.sumo_args``, checked:
    a list (a tuple, an OmegaConf ``ListConfig``) of strings or numbers, each
    made a ``str``; null is empty.

    A string is refused rather than taken as a list of one or split: Python
    would iterate it into characters (``sumo_args=--verbose`` became
    ``['-', '-', 'v', ...]``), and only the author knows whether
    ``'--seed 42'`` is one item or two.  An item that is a list, a mapping,
    null or a boolean is refused too: its ``str()`` (``"['a']"``, ``"None"``,
    ``"True"``) is never the item that was meant.

    Args:
        key: The setting's full name, for the error.
        value: The value to check.

    Returns:
        The items as a new list of strings.

    Raises:
        ValueError: naming *key*.
    """
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(
            f"{key} must be a list, not {type(value).__name__} {value!r}: "
            f"write '{key}=[a,b,c]'"
        )
    items = list(value)
    for index, item in enumerate(items):
        if isinstance(item, bool) or not isinstance(item, (str, int, float)):
            raise ValueError(
                f"{key}[{index}] must be a string or a number, not "
                f"{type(item).__name__} {item!r}"
            )
    return [str(item) for item in items]
