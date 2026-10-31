"""What YAML 1.1 makes of a plain scalar that does not mean what it looks like.

The parser-free half of :mod:`.config`'s refusal of unquoted YAML 1.1 forms:
given the tag a plain scalar resolved to and its text, whether it is one of
the forms YAML 1.2 reads differently -- octal ``0102``, sexagesimal ``1:30``,
``yes``/``no``/``on``/``off`` -- and, if so, what YAML 1.1 reads and how to
write it instead.  Nothing here imports yaml or walks a document, so the
library check compiles it (docs/typecheck.md); :mod:`.config` uses it and
re-exports it.
"""

from __future__ import annotations

from typing import Optional

__all__ = [
    "YAML_BOOL",
    "YAML_FLOAT",
    "YAML_INT",
    "yaml11_reading",
]

#: The tags YAML's resolver gives a plain scalar read as an integer, a float
#: or a boolean.
YAML_INT = "tag:yaml.org,2002:int"
YAML_FLOAT = "tag:yaml.org,2002:float"
YAML_BOOL = "tag:yaml.org,2002:bool"


def yaml11_reading(tag: str, text: str) -> Optional[str]:
    """What YAML 1.1 makes of the plain scalar *text* (resolved to *tag*) when
    that is not what it looks like, and how to write it instead; else None.

    These are the forms YAML 1.2 reads differently: ``0102`` (octal 66),
    ``1:30`` (sexagesimal 90) and ``yes``/``no``/``on``/``off`` (booleans).
    ``0x7DF``, ``1_000``, ``1.5`` and ``true`` mean what they say.
    """
    digits = text.lstrip("+-").replace("_", "")
    if tag == YAML_BOOL and text.lower() in ("yes", "no", "on", "off"):
        truth = text.lower() in ("yes", "on")
        return f"the boolean {truth}; write {str(truth).lower()}"
    if tag in (YAML_INT, YAML_FLOAT) and ":" in digits:
        return "a sexagesimal (base 60) number; write it in decimal"
    if (
        tag == YAML_INT
        and len(digits) > 1
        and digits[0] == "0"
        and digits[1] in "01234567"
    ):
        value = int(digits, 8) * (-1 if text.startswith("-") else 1)
        return f"the octal number {value}; write {value} or {value:#x}"
    return None
