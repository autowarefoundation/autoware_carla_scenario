"""Read an Autoware workspace's dev container: which image, mounted where.

The workspace is a clone of ``autowarefoundation/autoware`` -- the repository
that carries ``.devcontainer/<variant>/devcontainer.json`` -- with Autoware's
sources under ``src/``.  Each variant (``universe-devel-cuda``,
``universe-devel``, ``core-devel``) names a Compose file and a service in it,
and that service names the image and where the workspace is mounted.  The
stack is built and run in that same image at that same path, so a workspace
built from VS Code and one built here are the same build: colcon writes
absolute paths into ``install/``, and they hold on both sides only when the
workspace is mounted where it was built.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

__all__ = ["DevContainer", "DevContainerError", "read_devcontainer"]

#: Where Autoware's dev containers mount the workspace.
DEFAULT_WORKSPACE_MOUNT: str = "/home/aw/autoware"


class DevContainerError(ValueError):
    """The workspace's dev container configuration could not be read."""


@dataclass(frozen=True)
class DevContainer:
    """The parts of a dev container the stack is run with.

    Attributes:
        image: The image the dev container runs.
        workspace_mount: Where the workspace is mounted in it.
        environment: The service's environment, as Compose resolves it.
    """

    image: str
    workspace_mount: str = DEFAULT_WORKSPACE_MOUNT
    environment: Mapping[str, str] = field(default_factory=dict)


def _strip_jsonc(text: str) -> str:
    """Drop ``//`` and ``/* */`` comments and trailing commas from JSONC."""
    out: list[str] = []
    i, n = 0, len(text)
    in_string = False
    while i < n:
        c = text[i]
        if in_string:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if c == '"':
                in_string = False
            i += 1
        elif c == '"':
            in_string = True
            out.append(c)
            i += 1
        elif text.startswith("//", i):
            end = text.find("\n", i)
            i = n if end == -1 else end
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = n if end == -1 else end + 2
        else:
            out.append(c)
            i += 1
    return re.sub(r",(\s*[}\]])", r"\1", "".join(out))


_VARIABLE = re.compile(
    r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?:(?P<op>:?-)(?P<default>[^}]*))?\}"
)


def _interpolate(value: str, env: Mapping[str, str]) -> str:
    """Resolve Compose's ``${VAR}``, ``${VAR:-default}`` and ``${VAR-default}``."""

    def _replace(match: re.Match[str]) -> str:
        name, op, default = match.group("name", "op", "default")
        if op == ":-":
            return env.get(name) or default
        if op == "-":
            return env[name] if name in env else default
        return env.get(name, "")

    return _VARIABLE.sub(_replace, value)


def _service_environment(
    service: Mapping[str, Any], env: Mapping[str, str]
) -> dict[str, str]:
    raw = service.get("environment") or {}
    pairs: dict[str, str] = {}
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            pairs[str(key)] = "" if value is None else _interpolate(str(value), env)
    else:
        for item in raw:
            key, _, value = str(item).partition("=")
            pairs[key] = _interpolate(value, env)
    return pairs


def _workspace_mount(
    service: Mapping[str, Any],
    compose_dir: Path,
    workspace: Path,
    env: Mapping[str, str],
) -> Optional[str]:
    """Where *service* mounts *workspace*, if one of its volumes does."""
    for volume in service.get("volumes") or ():
        if isinstance(volume, Mapping):
            source, target = volume.get("source"), volume.get("target")
        else:
            parts = _interpolate(str(volume), env).split(":")
            if len(parts) < 2:
                continue
            source, target = parts[0], parts[1]
        if not source or not target:
            continue
        source_path = Path(_interpolate(str(source), env)).expanduser()
        if not source_path.is_absolute():
            source_path = compose_dir / source_path
        if source_path.resolve() == workspace.resolve():
            return _interpolate(str(target), env)
    return None


def read_devcontainer(
    workspace: Path,
    variant: str,
    *,
    env: Optional[Mapping[str, str]] = None,
) -> DevContainer:
    """Read ``<workspace>/.devcontainer/<variant>/devcontainer.json``.

    Its ``image`` is taken as is; otherwise its ``dockerComposeFile`` and
    ``service`` are followed to the service's ``image``.  Compose variables
    resolve against *env* (this process's environment by default), as
    ``docker compose`` would resolve them.

    Raises:
        DevContainerError: If the variant does not exist or names no image --
            one built by Compose from a Dockerfile, say, which has to be named
            as ``image`` in the configuration instead.
    """
    env = dict(os.environ if env is None else env)
    config_dir = workspace / ".devcontainer" / variant
    config_path = config_dir / "devcontainer.json"
    if not config_path.is_file():
        available = sorted(
            p.parent.name
            for p in (workspace / ".devcontainer").glob("*/devcontainer.json")
        )
        raise DevContainerError(
            f"{config_path} does not exist. The workspace must be a clone of "
            f"autowarefoundation/autoware; dev containers it has: {available or 'none'}."
        )
    try:
        config = json.loads(_strip_jsonc(config_path.read_text()))
    except json.JSONDecodeError as exc:
        raise DevContainerError(f"Could not parse {config_path}: {exc}") from exc

    folder = config.get("workspaceFolder")
    if config.get("image"):
        return DevContainer(
            image=_interpolate(str(config["image"]), env),
            workspace_mount=str(folder or DEFAULT_WORKSPACE_MOUNT),
        )

    compose_files = config.get("dockerComposeFile")
    service_name = config.get("service")
    if not compose_files or not service_name:
        raise DevContainerError(
            f"{config_path} names neither an image nor a Compose service; "
            "set the image explicitly."
        )
    if isinstance(compose_files, str):
        compose_files = [compose_files]

    # Later files override earlier ones, as `docker compose -f a -f b` does.
    service: dict[str, Any] = {}
    compose_dir = config_dir
    for name in compose_files:
        compose_path = (config_dir / name).resolve()
        try:
            document = yaml.safe_load(compose_path.read_text()) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise DevContainerError(f"Could not read {compose_path}: {exc}") from exc
        found = (document.get("services") or {}).get(service_name)
        if found:
            service.update(found)
            compose_dir = compose_path.parent
    if not service:
        raise DevContainerError(
            f"No service {service_name!r} in {compose_files} (from {config_path})."
        )
    image = service.get("image")
    if not image:
        raise DevContainerError(
            f"Service {service_name!r} is built rather than pulled; build it and "
            "set the image explicitly."
        )
    mount = _workspace_mount(service, compose_dir, workspace, env)
    return DevContainer(
        image=_interpolate(str(image), env),
        workspace_mount=str(folder or mount or DEFAULT_WORKSPACE_MOUNT),
        environment=_service_environment(service, env),
    )
