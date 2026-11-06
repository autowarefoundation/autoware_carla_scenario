"""stl_driver must not require an older carla-driver-interface than the framework.

``stl_driver`` is a project of its own with its own ``uv.lock``, but its policy
also loads in the scenario's own process (``driver.transport=direct``), and its
out-of-process server talks to the framework over the same contract.  Its
``carla-driver-interface`` range is meant to be the framework's; this test keeps
the lower bound from falling behind when the framework's is raised.
"""

from __future__ import annotations

import sys
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

# tomllib landed in 3.11, and 3.10 is still the floor of the supported range.
if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - only taken on 3.10
    import tomli as tomllib

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_WORKSPACE_ROOT = _PACKAGE_ROOT.parent
_STL_DRIVER_ROOT = _WORKSPACE_ROOT / "stl_driver"

_DISTRIBUTION = "carla-driver-interface"


def _requirement(directory: Path) -> Requirement:
    """The project's own requirement on carla-driver-interface."""
    manifest = tomllib.loads((directory / "pyproject.toml").read_text())
    matches = [
        requirement
        for requirement in map(Requirement, manifest["project"]["dependencies"])
        if requirement.name.lower().replace("_", "-") == _DISTRIBUTION
    ]
    assert (
        len(matches) == 1
    ), f"{directory.name} declares {_DISTRIBUTION} {len(matches)} times"
    return matches[0]


def _lower_bound(requirement: Requirement) -> Version:
    bounds = [
        Version(specifier.version)
        for specifier in requirement.specifier
        if specifier.operator in (">=", "==", "~=")
    ]
    assert bounds, f"{requirement} has no lower bound"
    return max(bounds)


def test_stl_driver_lower_bound_is_at_least_the_frameworks() -> None:
    framework = _requirement(_PACKAGE_ROOT)
    stl_driver = _requirement(_STL_DRIVER_ROOT)
    assert _lower_bound(stl_driver) >= _lower_bound(framework), (
        f"stl_driver requires {stl_driver}, older than the framework's {framework}; "
        "raise it in stl_driver/pyproject.toml and re-lock with `cd stl_driver && uv lock`"
    )


def test_stl_driver_lower_bound_is_inside_the_frameworks_range() -> None:
    """So a version satisfying both ranges exists."""
    framework = _requirement(_PACKAGE_ROOT)
    stl_driver = _requirement(_STL_DRIVER_ROOT)
    assert stl_driver.specifier.contains(_lower_bound(stl_driver)), stl_driver
    assert framework.specifier.contains(
        _lower_bound(stl_driver)
    ), f"stl_driver's {stl_driver} lower bound is outside the framework's {framework}"
