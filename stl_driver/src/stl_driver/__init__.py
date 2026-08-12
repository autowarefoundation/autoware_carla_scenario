# SPDX-License-Identifier: Apache-2.0
"""An ego policy that drives by maximising the robustness of an STL specification.

The behaviour this package implements is not spread across the planner -- it is
written down, as temporal logic, in :mod:`stl_driver.rules`.  Everything else
exists to serve that specification: :mod:`stl_driver.rollout` turns plan
parameters into the signal the formulas read, :mod:`stl_driver.optimizer`
searches for the plan that satisfies them best, and :mod:`stl_driver.policy`
wires the result into ``carla_driver_interface``.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["EgoPolicy", "STLDriverConfig", "__version__"]


def __getattr__(name: str) -> object:
    # Imported lazily so that `import stl_driver` stays cheap: pulling in the
    # policy drags in torch and the gRPC stack, which the CLI's --help and the
    # rule-only tests have no use for.
    if name in ("EgoPolicy", "STLDriverConfig"):
        from stl_driver.config import STLDriverConfig
        from stl_driver.policy import EgoPolicy

        return {"EgoPolicy": EgoPolicy, "STLDriverConfig": STLDriverConfig}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
