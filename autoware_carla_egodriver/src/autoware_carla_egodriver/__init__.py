"""Both ends of alpasim's ``egodriver.EgodriverService`` gRPC contract, without CARLA.

A *policy* -- anything that plans a trajectory from camera frames, ego motion and a
route -- implements :class:`~autoware_carla_egodriver.driver.BaseDriver` and is served
with :func:`~autoware_carla_egodriver.server.run_server` (or
``autoware-carla-egodriver serve``). A *runtime* drives it: the scenario framework
(``autoware_carla_scenario``, ``ego.entity=carla_driver``) against CARLA, an upstream
alpasim runtime, or :mod:`autoware_carla_egodriver.testing` without a simulator.

The package is deliberately light -- numpy, grpcio, protobuf and Pillow -- so a policy
depends on it without pulling in the scenario framework, and it runs on Python 3.10.
The wire contract is vendored from NVlabs/alpasim (see ``proto/README.md``); field
numbers and service names are unchanged, so it interoperates with upstream.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("autoware-carla-egodriver")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0"

__all__ = ["__version__"]
