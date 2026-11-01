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
``docker_args``      More ``docker run`` arguments.
==================== ========================================================
"""

from __future__ import annotations

from typing import Any, ClassVar, Mapping

import grpc

from ..prepare import Resources
from ..prepare.docker import DockerBackend, DockerSettings
from .motion_client import PedestrianMotionClient

__all__ = ["DEFAULT_IMAGE", "ArdyBackend"]

#: The image of the backend release this package speaks the contract of.
DEFAULT_IMAGE = "ghcr.io/hakuturu583/carla_ardy_backend:0.1.0"

#: The port the server listens on in the container.
CONTAINER_PORT = 50061


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
        hf_home = str(settings.get("hf_home", "~/.cache/huggingface"))
        env = {
            "ARDY_MODEL": str(settings.get("model", "soma")),
            "ARDY_ACCELERATION": str(settings.get("acceleration", "none")),
            "HF_HOME": "/hf",
        }
        return cls(
            DockerSettings(
                image=str(settings.get("image", DEFAULT_IMAGE)),
                container_port=CONTAINER_PORT,
                gpus=str(settings.get("gpus", "auto")),
                env=env,
                pass_env=("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"),
                mounts=(f"{hf_home}:/hf",),
                docker_args=tuple(
                    str(a) for a in settings.get("docker_args", ()) or ()
                ),
                pull=str(settings.get("pull", "missing")),
                startup_timeout_s=float(settings.get("startup_timeout_s", 1800.0)),
                resources=Resources(
                    gpu_memory_mb=int(settings.get("gpu_memory_mb", 24000)),
                    memory_mb=int(settings.get("memory_mb", 16000)),
                ),
            ),
            timeout_s=float(settings.get("timeout_s", 120.0)),
        )

    def ready(self, timeout_s: float) -> bool:
        client = PedestrianMotionClient(self.address(), timeout_s=timeout_s)
        try:
            client.negotiate()
        except grpc.RpcError:
            return False
        finally:
            client.close()
        return True
