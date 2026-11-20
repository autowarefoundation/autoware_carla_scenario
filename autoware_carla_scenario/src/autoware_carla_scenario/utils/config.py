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

from dataclasses import Field, fields
from typing import Any, Mapping, Optional, Sequence

__all__ = [
    "checked_mapping_list",
    "checked_options",
    "checked_section",
    "checked_str_list",
    "entry_id",
    "null_as_default",
]


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


def checked_section(key: str, value: Any) -> Optional[Mapping[str, Any]]:
    """*value*, a nested options section such as ``traffic.options.ambient``,
    checked: a mapping is itself, null (an absent section, or
    ``${oc.select:...,null}``) is ``None`` -- the caller's default.

    ``false``, ``0``, ``""`` or a list is refused rather than read as the
    default (#2143) or passed on to fail later, naming no key, when the
    section is read as the options it stands for (#2174).

    Args:
        key: The section's full name, for the error.
        value: The value to check.

    Returns:
        The mapping, or ``None`` for a null section.

    Raises:
        ValueError: naming *key* and the value, for anything else.
    """
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping, got {value!r}")
    return value


def checked_mapping_list(key: str, value: Any) -> list[Mapping[str, Any]]:
    """*value*, a list of sections such as ``driver.lidars``, checked: a list
    (a tuple, an OmegaConf ``ListConfig``) of mappings.

    ``false`` or a string would fail iterating, and an entry that is no
    mapping as it is read, naming no key (#2217, #2228).  A null list is the
    caller's to read (its default): this is not handed one.

    Args:
        key: The setting's full name, for the error.
        value: The value to check.

    Returns:
        The entries as a new list.

    Raises:
        ValueError: naming *key*, or the entry ``key[i]`` that is no mapping.
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{key} must be a list of mappings, got {value!r}")
    items = list(value)
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise ValueError(f"{key}[{index}] must be a mapping, got {item!r}")
    return items


def entry_id(entry: Mapping[str, Any], default: str) -> Any:
    """The ``logical_id`` an entry of a sensor list is given, *default* when
    it gives none (or null): what names the entry in an error."""
    logical_id = entry.get("logical_id")
    return default if logical_id is None else logical_id


def _takes_null(field: Field) -> bool:
    """Whether *field* takes ``None`` as a value of its own: its default is
    ``None``, or its type says ``Optional`` (``curve_lateral_acceleration:
    Optional[float] = 3.0``, where null means something else than 3.0)."""
    if field.default is None:
        return True
    text = field.type if isinstance(field.type, str) else repr(field.type)
    return "Optional[" in text or "None" in text


def null_as_default(config_cls: type, checked: Mapping[str, Any]) -> dict:
    """*checked* -- a mapping :func:`checked_options` has already checked --
    without the null values of the fields that do not take null: an absent
    or null setting is its default (#2143), so ``blueprint: null`` is the
    default blueprint rather than refused.

    A field that takes null -- an ``Optional`` one -- keeps it, as its
    default or as a value of its own.  The dataclass's constructor still
    refuses ``None`` for the others: only a mapping, where null stands for
    "not set" (``${oc.select:...,null}``), reads it as the default.

    Args:
        config_cls: The dataclass the mapping is meant to build.
        checked: The values to read.

    Returns:
        The values as a new dict, without the nulls that mean the default.
    """
    defaults = {field.name for field in fields(config_cls) if not _takes_null(field)}
    return {
        key: value
        for key, value in checked.items()
        if value is not None or key not in defaults
    }
