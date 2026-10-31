"""The ``launcher.*`` keys the viewer sets on a SLURM batch (docs/slurm.md):
what each may hold, the Hydra overrides they become, and the ``scenario``
command line that submits the batch with them.

Split out of ``ui.slurm_launch`` so the library check compiles it (#742).
What reads the run panel's fields, values of any type off a JSON request
(``SlurmOptions.from_request()``), and what starts ``scenario`` (the command
``VIEWER_SCENARIO_COMMAND`` names, split with ``shlex``) stay there:
``from_request()`` turns each field into a ``str`` or a number and checks it
with the ``check_*`` functions here.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

__all__ = [
    "SlurmOptions",
    "build_command",
    "check_at_least",
    "check_env",
    "check_node_path",
    "check_sbatch_value",
    "check_setup_line",
    "check_time",
]

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TIME = re.compile(r"^(\d+-)?\d{1,2}(:\d{2}){0,2}$|^\d+$")
_SIMPLE = re.compile(r"^[A-Za-z0-9_.:/@+-]+$")
#: A path on the nodes (``launcher.remote_cwd``, ``launcher.python``).
_NODE_PATH = re.compile(r"^~?[A-Za-z0-9_.:/@+-]*$")

#: An unquoted value Hydra reads as something other than that string: its
#: grammar's INT, FLOAT, BOOL and NULL tokens (``hydra/grammar/OverrideLexer.g4``).
_NOT_A_STRING = re.compile(
    r"[+-]?(?:"
    r"(?:0|[1-9](?:_?\d)*)"  # INT
    r"|(?:(?:0|[1-9](?:_?\d)*)\.|(?:0|[1-9](?:_?\d)*)?\.\d(?:_?\d)*)"  # POINT_FLOAT
    r"(?:[eE][+-]?\d(?:_?\d)*)?"
    r"|(?:0|[1-9](?:_?\d)*)[eE][+-]?\d(?:_?\d)*"  # EXPONENT_FLOAT
    r"|inf|nan"
    r")|true|false|null",
    re.IGNORECASE,
)
#: A run of backslashes a quote ends.
_BEFORE_QUOTE = re.compile(r"(\\*)'")


def _quote(value: str) -> str:
    """*value* as a Hydra override's value that Hydra reads back as exactly
    this string: as it is when plain, else single-quoted.

    In a quoted value Hydra un-escapes only what the grammar's
    ``QUOTED_VALUE`` needs (``_unescape_quoted_string``): a quote is ``\\'``,
    and a run of backslashes is doubled when a quote or the end of the value
    follows it; every other backslash is itself (``a\\b`` is ``a\\b``).
    """
    if _SIMPLE.fullmatch(value) and not _NOT_A_STRING.fullmatch(value):
        return value
    return _quote_always(value)


def _quote_always(value: str) -> str:
    """*value* single-quoted, as :func:`_quote` quotes it."""
    body = _BEFORE_QUOTE.sub(lambda m: m[1] * 2 + "\\'", value)
    # The run of backslashes that ends the value, doubled: what
    # `re.sub(r"(\\+)\Z", ...)` did, which Codon's re (RE2) has no `\Z` for.
    trailing = len(body) - len(body.rstrip("\\"))
    return "'" + body + "\\" * trailing + "'"


# ── What each field may hold ───────────────────────────────────────────────


def check_env(name: str, value: str, shown: str) -> None:
    """An ``env`` entry: *name* a variable's name, *value* the text it is
    exported as, without an interpolation; *shown* is the value as the
    request gave it, as the message shows it (its ``repr()``).

    Raises:
        ValueError: It is not.
    """
    if not _ENV_NAME.fullmatch(name):
        raise ValueError(f"{name!r} is not an environment variable name")
    # The value becomes an OmegaConf string: `${` would be an
    # interpolation, not the text the job is to see.
    if "${" in value:
        raise ValueError(
            f"env {name}={shown}: no ${{...}} -- the value is exported "
            "as written, not expanded (set it in a setup line, e.g. "
            "export NAME=$HOME/x, or set launcher.env in a config)"
        )


def check_time(time: str | None) -> None:
    """``time``: sbatch's ``--time``, if set.

    Raises:
        ValueError: It is not.
    """
    if time is not None and not _TIME.match(time):
        raise ValueError(
            f"time is sbatch's --time ([D-]HH:MM:SS or minutes), not {time!r}"
        )


def check_sbatch_value(key: str, value: str | None) -> None:
    """``partition``, ``mem`` or ``gres``: a word sbatch takes, if set.

    Raises:
        ValueError: It is not.
    """
    if value is not None and not _SIMPLE.match(value):
        raise ValueError(f"{key} {value!r} has characters sbatch does not take")


def check_node_path(key: str, value: str | None) -> None:
    """``remote_cwd`` or ``python``: a path the launcher takes, if set.

    Raises:
        ValueError: It is not.
    """
    if value is not None and (
        not _NODE_PATH.match(value)
        or (value.startswith("~") and value != "~" and value[1] != "/")
    ):
        raise ValueError(
            f"{key} {value!r} is not a path the launcher takes (letters, "
            "digits, _ . : / @ + -, and ~ or ~/ first)"
        )


def check_setup_line(line: str) -> None:
    """A ``setup`` line, as the job script is to run it.

    Raises:
        ValueError: It holds an interpolation or ends in a backslash.
    """
    # A Hydra override's value, then an OmegaConf string: an
    # interpolation would not reach the job as written.
    if "${" in line:
        raise ValueError(
            f"setup line {line!r}: write $NAME rather than ${{NAME}} "
            "(or set launcher.setup in a config)"
        )
    # The job script holds each line as written: a line ending in an
    # unescaped backslash would run on into the next line the script
    # has (`set -u`, say), not end where the field's line does.
    if (len(line) - len(line.rstrip("\\"))) % 2:
        raise ValueError(
            f"setup line {line!r} ends in a backslash: write each command "
            "on one line"
        )


def check_at_least(key: str, value: float, minimum: int) -> None:
    """A number field: finite and at least *minimum* (1 for a count, 0 for
    seconds).

    Raises:
        ValueError: It is not finite (``nan``, ``inf``, ``1e400``) or less.
    """
    if not math.isfinite(value):
        raise ValueError(f"{key} must be a finite number")
    if value < minimum:
        raise ValueError(f"{key} must be at least {minimum}")


# ── The overrides ──────────────────────────────────────────────────────────


def _text_override(key: str, value: str | None) -> str | None:
    """``launcher.<key>=<value>`` for a text key, or ``None`` when unset."""
    # A number set on a text key is written as it is (as a number key is).
    if isinstance(value, (int, float)):
        return f"launcher.{key}={value}"
    if value is None:
        return None
    return f"launcher.{key}={_quote(value)}"


def _count_override(key: str, value: int | None) -> str | None:
    """``launcher.<key>=<value>`` for a count key, or ``None`` when unset."""
    if isinstance(value, (int, float)):
        return f"launcher.{key}={value}"
    if value is None:
        return None
    # Text set on a number key is quoted as a text key's is; anything else
    # is refused, as _quote() refuses it.
    if not isinstance(value, str):
        raise TypeError(f"launcher.{key} must be a number or text")
    text = str(value)
    return f"launcher.{key}={_quote(text)}"


def _seconds_override(key: str, value: float | None) -> str | None:
    """``launcher.<key>=<value>`` for a seconds key, or ``None`` when unset."""
    if isinstance(value, (int, float)):
        return f"launcher.{key}={value}"
    if value is None:
        return None
    # Text set on a number key is quoted as a text key's is; anything else
    # is refused, as _quote() refuses it.
    if not isinstance(value, str):
        raise TypeError(f"launcher.{key} must be a number or text")
    text = str(value)
    return f"launcher.{key}={_quote(text)}"


@dataclass
class SlurmOptions:
    """The ``launcher.*`` keys the viewer sets (the rest are the config's)."""

    partition: str | None = None
    array_parallelism: int | None = None
    cases_per_task: int | None = None
    time: str | None = None
    mem: str | None = None
    gres: str | None = None
    start_stagger_seconds: float | None = None
    #: Where the jobs work on the nodes, for a viewer whose host sees their
    #: file system at another path (docs/slurm.md).
    remote_cwd: str | None = None
    python: str | None = None
    #: ``launcher.setup``: shell lines a job runs first; empty: the config's.
    setup: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)

    def overrides(self) -> list[str]:
        """``launcher=slurm`` and the keys set, as Hydra overrides."""
        out = ["launcher=slurm"]
        for override in [
            _text_override("partition", self.partition),
            _count_override("array_parallelism", self.array_parallelism),
            _count_override("cases_per_task", self.cases_per_task),
            _text_override("time", self.time),
            _text_override("mem", self.mem),
            _text_override("gres", self.gres),
            _seconds_override("start_stagger_seconds", self.start_stagger_seconds),
            _text_override("remote_cwd", self.remote_cwd),
            _text_override("python", self.python),
        ]:
            if override is not None:
                out.append(override)
        if self.setup:
            lines = ",".join(_quote_always(line) for line in self.setup)
            out.append(f"launcher.setup=[{lines}]")
        # `env` is an empty mapping in the config: a new key is added (++).
        out.extend(
            f"++launcher.env.{k}={_quote(v)}" for k, v in sorted(self.env.items())
        )
        return out


def build_command(
    command: list[str],
    scenario: str,
    extra_overrides: list[str],
    sweeper: str,
    options: SlurmOptions,
) -> list[str]:
    """The ``scenario`` command that submits the batch and exits
    (``--no-wait``), *command* being how ``scenario`` is started.

    With a sweeper the batch runs every case of each scenario's sweep
    (``--multirun``, expanded by the batch itself); without one each scenario
    the pattern matches runs once.
    """
    cmd = list(command)
    if sweeper:
        cmd.append("--multirun")
    cmd.append(f"scenario={scenario}")
    cmd.extend(extra_overrides)
    cmd.extend(options.overrides())
    cmd.append("--no-wait")
    return cmd
