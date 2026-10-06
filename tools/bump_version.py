#!/usr/bin/env python3
"""Bumps the release version in every place that has to agree on it.

The version lives in `autoware_carla_scenario/pyproject.toml` (what PyPI sees), and
`uv.lock` records it again in the editable workspace entry. Bumping the first without
re-locking leaves the lock stale, and every later non-frozen uv command rewrites it --
so this script does both, and the release job calls it rather than editing by hand.

    tools/bump_version.py patch          # 3.1.2 -> 3.1.3
    tools/bump_version.py minor          # 3.1.2 -> 3.2.0
    tools/bump_version.py major          # 3.1.2 -> 4.0.0
    tools/bump_version.py 3.4.0          # set exactly
    tools/bump_version.py --current      # print the version, change nothing

The single line printed to stdout is the resulting version, which the workflow reads
back to form the tag. uv's own output goes to stderr, so it can be captured directly.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "autoware_carla_scenario" / "pyproject.toml"

_VERSION_LINE = re.compile(r'(?m)(^\[project\][^\[]*?^version = )"([^"]+)"')


def current_version() -> str:
    match = _VERSION_LINE.search(PYPROJECT.read_text())
    if match is None:
        raise SystemExit(f"could not find [project] version in {PYPROJECT}")
    return match.group(2)


def next_version(current: str, spec: str) -> str:
    if spec not in {"major", "minor", "patch"}:
        if not re.fullmatch(r"\d+\.\d+\.\d+", spec):
            raise SystemExit(
                f"expected major|minor|patch or an X.Y.Z version, got {spec!r}"
            )
        return spec
    major, minor, patch = (int(part) for part in current.split("."))
    if spec == "major":
        return f"{major + 1}.0.0"
    if spec == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def set_pyproject(version: str) -> None:
    text, count = _VERSION_LINE.subn(
        rf'\g<1>"{version}"', PYPROJECT.read_text(), count=1
    )
    if count != 1:
        raise SystemExit(f"failed to rewrite the version in {PYPROJECT}")
    PYPROJECT.write_text(text)


def relock() -> None:
    # `uv lock` keeps every other pin as it is; only the workspace entry moves.
    subprocess.run(["uv", "lock"], cwd=ROOT, check=True, stdout=sys.stderr)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    if sys.argv[1] == "--current":
        print(current_version())
        return
    version = next_version(current_version(), sys.argv[1])
    set_pyproject(version)
    relock()
    print(version)


if __name__ == "__main__":
    main()
