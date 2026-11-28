"""The checks of a config's number settings: a number, a finite number and a
TCP port, each refused with a ``ValueError`` naming its key.

One copy of each, which the validators of the run's config, the declarative
scenarios, the authored cards, the driver's sensors, the example scenarios,
the traffic backends and the HILS settings call (#2398): their copies had
drifted, so ``ego.initial_speed_kmh=true`` ran at 1 km/h on one path and was
refused on another, and an integer too big for a float (``10**400``) raised
an ``OverflowError`` naming no key on some.  Each error shows the value
through :func:`~autoware_carla_scenario.value_text.shown`, which never raises,
not even for an integer too long for ``repr()``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Union

from .value_text import shown

__all__ = ["check_port", "finite_number_setting", "is_port", "number_setting"]


def number_setting(key: str, value: Union[float, str], *, text: bool = False) -> float:
    """*value* of the setting *key* as a ``float``.

    A config dataclass or node is not coerced, so a setting typed ``float``
    holds whatever its mapping did: Hydra reads ``=true`` as a bool, which
    ``float()`` would take as 1.0, and a quoted ``='1.5'`` as a string.  An
    int or a float is a number, and so is anything else ``float()`` reads
    but a bool or a string; a bool never is (``true`` is not 1), numpy's
    ``bool_`` and a 0-d bool array included, nor is an array of one or more
    dimensions (``np.array([1.5])``), whatever numpy's version.  A string is
    read as the number it spells only with *text* -- the run's ``ego.*`` and
    ``entity.*`` numbers, a declarative scenario's spawn override and an
    authored card's field, which read ``'1.5'`` as 1.5 -- and refused
    otherwise, as is a blank one either way.

    An int too big for a float (``10**400``) is refused as not finite, rather
    than with ``float()``'s ``OverflowError``.  A NaN or an infinity is
    returned: :func:`finite_number_setting` refuses those too.

    Raises:
        ValueError: naming *key*.
    """
    if isinstance(value, bool) or (isinstance(value, str) and not text):
        raise ValueError(f"{key} must be a number, got {shown(value)}")
    if TYPE_CHECKING:
        pass
    elif getattr(getattr(value, "dtype", None), "kind", None) == "b" or (
        # numpy's ndim is an int; a stand-in's (a mock's) is not (#1906).
        isinstance(getattr(value, "ndim", 0), int) and getattr(value, "ndim", 0) != 0
    ):
        # numpy's bool_ (or a 0-d bool array), which float() takes as 0 or 1,
        # as seed.check_whole_number refuses it (#1900, #2507); and an array
        # of one element, which the numpy on Python 3.10 (2.2) still converts,
        # as check_whole_number refuses it too (#1904, #2620).
        raise ValueError(f"{key} must be a number, got {shown(value)}")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a number, got {shown(value)}") from None
    except OverflowError:
        raise ValueError(f"{key} must be finite, got {shown(value)}") from None


def finite_number_setting(
    key: str, value: Union[float, str], *, text: bool = False
) -> float:
    """*value* of the setting *key* as a finite ``float``: a number as
    :func:`number_setting` reads it, and neither a NaN nor an infinity.

    Raises:
        ValueError: naming *key*.
    """
    number = number_setting(key, value, text=text)
    if not math.isfinite(number):
        raise ValueError(f"{key} must be finite, got {shown(value)}")
    return number


def is_port(port: int) -> bool:
    """Whether *port* is a TCP port: an integer within 1-65535, and not a
    bool, which is an int to Python."""
    return not isinstance(port, bool) and isinstance(port, int) and 1 <= port <= 65535


def check_port(key: str, port: int) -> None:
    """Refuse a TCP port that is not an integer within 1-65535
    (:func:`is_port`).

    Refused where a run's config is checked, rather than by
    ``client.get_trafficmanager()`` or the CARLA client once CARLA is up: a
    bool is an int to Python, and 0 or 70000 is no TCP port.  The one check
    of a port setting: ``server.port``, ``traffic_manager.port``, the
    ``sumo`` backend's ``tm_port`` and ``hils.port`` alike (``hils.config``'s
    ``carla.port`` takes :func:`is_port`, with its own wording).

    Args:
        key: The setting's name, for the error.
        port: The value to check.

    Raises:
        ValueError: naming *key* and showing *port*, a huge integer included.
    """
    if not is_port(port):
        raise ValueError(f"{key} must be an integer within 1-65535 (got {shown(port)})")
