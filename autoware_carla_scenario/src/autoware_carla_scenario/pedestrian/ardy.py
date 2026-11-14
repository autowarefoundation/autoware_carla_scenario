"""The ARDY backend: nv-tlabs/ardy serving the pedestrian motion contract.

`ARDY <https://github.com/nv-tlabs/ardy>`_ generates human motion
autoregressively, conditioned online by text, root paths and full-body
keyframes.  hakuturu583/carla_ardy_backend packages it as a docker image
(``ghcr.io/hakuturu583/carla_ardy_backend``) whose server speaks
``proto/pedestrian_motion``; :class:`ArdyBackend` runs that image on this
host for the runs that need it (``@need_prepare(ArdyBackend)``).

Its text encoder is Llama-3-8B (gated on Hugging Face): the host's Hugging
Face cache, where the token and the downloaded weights live, is mounted into
the container, and ``HF_TOKEN`` is passed on when it is set.

Settings (``backends.ardy`` of the run's config), each optional:

==================== ========================================================
``image``            The image (:data:`DEFAULT_IMAGE`).
``model``            The ARDY checkpoint (``soma``: ARDY-SOMA-RP-30FPS-Horizon60).
``acceleration``     ``none``, ``compile`` (torch.compile) or ``trt``.
``gpus``             ``auto`` (the GPU the preparer chose), ``all``, ``device=N``.
``hf_home``          The host's Hugging Face cache (``~/.cache/huggingface``).
``gpu_memory_mb``    GPU memory it takes (24000: the text encoder alone ~14 GB).
``memory_mb``        Host memory it takes (16000).
``startup_timeout_s`` How long it may take to answer (1800: a first start
                     downloads ~20 GB of weights).
``timeout_s``        How long one Step may take (120).
``pull``             ``missing``, ``always`` or ``never``.
``docker_args``      More ``docker run`` arguments: a list.
==================== ========================================================

Every key absent or null is its default.  ``model`` and ``hf_home`` must be
non-empty text, ``acceleration`` one of :data:`ACCELERATIONS`, the timeouts
finite and above 0, the memories whole numbers, not negative; a value refused
is a ``ValueError`` naming its key (``backends.ardy.<key>``), raised by :meth:`ArdyBackend.from_settings`
before any image is pulled or container started
(:func:`~autoware_carla_scenario.prepare.docker.read_docker_settings`).
"""

from __future__ import annotations

from typing import Any, ClassVar, Mapping

import grpc

from ..prepare.docker import (
    DockerBackend,
    DockerSettings,
    positive_seconds,
    read_docker_settings,
    setting_key,
)
from .motion_client import PedestrianMotionClient

__all__ = ["ACCELERATIONS", "DEFAULT_IMAGE", "ArdyBackend"]

#: The image of the backend release this package speaks the contract of.
DEFAULT_IMAGE = "ghcr.io/hakuturu583/carla_ardy_backend:0.1.0"

#: The port the server listens on in the container.
CONTAINER_PORT = 50061

#: The ARDY checkpoint run when ``model`` is absent or null.
DEFAULT_MODEL = "soma"

#: The host's Hugging Face cache mounted when ``hf_home`` is absent or null.
DEFAULT_HF_HOME = "~/.cache/huggingface"

#: What ``acceleration`` takes: none, ``torch.compile``, or TensorRT.
ACCELERATIONS = ("none", "compile", "trt")


def _text(backend: str, key: str, settings: Mapping[str, Any], default: str) -> str:
    """*settings*' *key*, text that is not blank; absent or null is *default*.

    ``false`` or a number is refused rather than handed to the container as
    its ``str()`` (``ARDY_MODEL='None'``, an ``hf_home`` of ``None`` that
    docker mounts as a named volume, #2158).

    Raises:
        ValueError: naming the key (``backends.<backend>.<key>``).
    """
    value = settings.get(key)
    if value is None:
        return default
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{setting_key(backend, key)} must be non-empty text, got {value!r}"
        )
    return value


class ArdyBackend(DockerBackend):
    """ARDY in its container (:mod:`~autoware_carla_scenario.pedestrian.ardy`)."""

    backend_name: ClassVar[str] = "ardy"

    _timeout_s: float

    def __init__(self, settings: DockerSettings, timeout_s: float = 120.0) -> None:
        super().__init__(settings)
        self._timeout_s = timeout_s

    @property
    def timeout_s(self) -> float:
        """How long one call of a session may take."""
        return self._timeout_s

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any]) -> "ArdyBackend":
        """ARDY configured by *settings* (``backends.ardy``), checked.

        Raises:
            ValueError: naming the key (``backends.ardy.<key>``) of the first
                value refused, before any image is pulled or container
                started.
        """
        name = cls.backend_name
        hf_home = _text(name, "hf_home", settings, DEFAULT_HF_HOME)
        model = _text(name, "model", settings, DEFAULT_MODEL)
        acceleration = settings.get("acceleration")
        if acceleration is None:
            acceleration = "none"
        if not isinstance(acceleration, str) or acceleration not in ACCELERATIONS:
            raise ValueError(
                f"{setting_key(name, 'acceleration')} must be one of "
                f"{', '.join(ACCELERATIONS)}, got {acceleration!r}"
            )
        env = {
            "ARDY_MODEL": model,
            "ARDY_ACCELERATION": acceleration,
            "HF_HOME": "/hf",
        }
        docker = read_docker_settings(
            name,
            settings,
            image=DEFAULT_IMAGE,
            container_port=CONTAINER_PORT,
            env=env,
            pass_env=("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"),
            mounts=(f"{hf_home}:/hf",),
            startup_timeout_s=1800.0,
            gpu_memory_mb=24000,
            memory_mb=16000,
        )
        timeout_value = settings.get("timeout_s")
        if timeout_value is None:
            timeout_value = 120.0
        timeout_s = positive_seconds(setting_key(name, "timeout_s"), timeout_value)
        return cls(docker, timeout_s=timeout_s)

    def ready(self, timeout_s: float) -> bool:
        client = PedestrianMotionClient(self.address(), timeout_s=timeout_s)
        try:
            client.negotiate()
        except grpc.RpcError:
            return False
        finally:
            client.close()
        return True
