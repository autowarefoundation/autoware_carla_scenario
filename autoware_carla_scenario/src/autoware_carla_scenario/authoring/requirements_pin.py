"""Pin a wheelhouse's direct-reference requirements to the wheels built for them.

The string half of :mod:`.wheelhouse`, which runs uv, builds wheels and
writes the directory: the PEP 503 normalised form of a distribution name, the
distribution a bare direct reference (a path or URL with no name on it)
points at, and the rewrite of ``uv export``'s requirements that names each
git or path source by the version of the wheel built for it.  Nothing here
touches a file, a process or the network, so the library check compiles it
(docs/typecheck.md); :mod:`.wheelhouse` uses it.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

__all__ = [
    "DIRECT_SCHEMES",
    "canonical_name",
    "pin_direct_references",
    "referenced_distribution",
]


def canonical_name(name: str) -> str:
    """Return *name* in PEP 503 normalised form, for comparing distributions."""
    return re.sub(r"[-_.]+", "-", name).lower()


#: What a requirement line looks like when uv writes the source instead of a
#: version. A git source comes through as ``name @ <url>``; a path source comes
#: through as the bare URL, with no name on it at all.
DIRECT_SCHEMES: Tuple[str, str, str, str, str, str, str] = (
    "file:",
    "git+",
    "http:",
    "https:",
    "./",
    "../",
    "/",
)


def _is_direct(line: str) -> bool:
    """Whether *line* starts with one of :data:`DIRECT_SCHEMES`."""
    return any(line.startswith(scheme) for scheme in DIRECT_SCHEMES)


def _without_suffix(text: str, suffix: str) -> str:
    """``text.removesuffix(suffix)``, spelled out."""
    if suffix and text.endswith(suffix):
        return text[: len(text) - len(suffix)]
    return text


def referenced_distribution(reference: str) -> str:
    """Return the distribution a bare direct reference names.

    There is no name in the line to read, so it comes from the location: the
    subdirectory when the reference has one, otherwise the last path segment.
    That is a guess, and the caller only acts on it when it matches a wheel it
    actually built.
    """
    head = reference.split(";", 1)[0].strip()
    parts = head.partition("#")
    head = parts[0]
    fragment = parts[2]
    subdirectory = re.search(r"subdirectory=([^&]+)", fragment)
    if subdirectory is not None:
        found: Optional[str] = subdirectory.group(1)
        if found is not None:
            return found.rstrip("/").split("/")[-1]
    if head.startswith("git+"):
        head = re.sub(r"@[^/@]+$", "", head)
    return _without_suffix(head.rstrip("/").split("/")[-1], ".git")


def pin_direct_references(requirements: str, wheels: Tuple[str, ...]) -> str:
    """Rewrite requirements that name a source to the version built for them.

    ``uv export`` keeps a git or path source as a *direct reference*, and pip
    honours a direct reference however many ``--find-links`` it was given: it
    clones the repository, or reads a directory on the exporting machine.  Both
    are exactly what a wheelhouse exists to avoid, and neither is there on the
    target.  The wheel is already in the directory, so naming it by version is
    what makes ``-r requirements.txt`` an offline install.

    Two shapes, because uv writes two: ``name @ <url>`` for a git source, and
    the bare URL for a path one.  A requirement whose wheel is not in the
    directory is left alone rather than guessed at.
    """
    versions: Dict[str, str] = {}
    for wheel in wheels:
        fields = wheel.split("-")
        if len(fields) < 2:
            # What unpacking `wheel.split("-")[:2]` into two names raised.
            raise ValueError(
                f"not enough values to unpack (expected 2, got {len(fields)})"
            )
        versions[canonical_name(fields[0])] = fields[1]

    lines: List[str] = []
    for line in requirements.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            lines.append(line)
            continue

        name = ""
        rest = ""
        if " @ " in stripped:
            split = stripped.partition(" @ ")
            name = split[0]
            rest = split[2]
        elif _is_direct(stripped):
            # Nothing in the line is a name, so write the canonical one rather
            # than the directory spelling the location happened to use.
            name = canonical_name(referenced_distribution(stripped))
            rest = stripped
        else:
            lines.append(line)
            continue

        pinned = versions.get(canonical_name(name.split("[")[0]), None)
        if pinned is None:
            lines.append(line)
            continue
        # Markers travel with the requirement; the source does not.
        marker = f" ;{rest.split(';', 1)[1]}" if ";" in rest else ""
        lines.append(f"{name}=={pinned}{marker}")
    return "\n".join(lines)
