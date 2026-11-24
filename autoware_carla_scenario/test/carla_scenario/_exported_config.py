"""An exported scenario config as a run reads it.

An export's ``ego.*`` keys read the package's document through
``${scenario_document:<id>,<path>}`` (``registry.SCENARIO_DOCUMENT_RESOLVER``),
which needs the document registered, as a package's ``register()`` does.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from omegaconf import OmegaConf

from autoware_carla_scenario import registry
from autoware_carla_scenario.authoring.hydra_config import build_scenario_config
from autoware_carla_scenario.authoring.models import ScenarioDocument


@contextmanager
def registered_document(document: ScenarioDocument) -> Iterator[None]:
    """*document* registered under its id for the block, as a package's
    ``register()`` registers its own; whatever was registered before is put
    back."""
    documents = registry._SCENARIO_DOCUMENTS
    before = documents.get(document.id)
    registry.register_scenario_document(document.id, lambda: document)
    try:
        yield
    finally:
        if before is None:
            documents.pop(document.id, None)
        else:
            documents[document.id] = before


def resolved_config(
    document: ScenarioDocument, config: dict[str, Any] | None = None
) -> dict[str, Any]:
    """*config* (*document*'s export when ``None``) with its ``ego`` keys
    read from *document*, as a plain dict.  Only those: the other sections'
    references (``${map...}``) are the composed run's to resolve."""
    exported = build_scenario_config(document) if config is None else config
    if "ego" not in exported:
        return dict(exported)
    with registered_document(document):
        ego = OmegaConf.to_container(OmegaConf.create(exported["ego"]), resolve=True)
    return {**exported, "ego": ego}
