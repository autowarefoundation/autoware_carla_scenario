"""Reference policies.

Deliberately dependency-free (numpy only), so the closed loop can be exercised
without a model checkpoint or a GPU.
"""

from __future__ import annotations

from typing import Dict, Type

from autoware_carla_egodriver.driver import BaseDriver
from autoware_carla_egodriver.policies.constant_speed import ConstantSpeedPolicy
from autoware_carla_egodriver.policies.route_follower import RouteFollowerPolicy

#: Name -> class, for ``autoware-carla-egodriver serve --policy``.
POLICY_REGISTRY: Dict[str, Type[BaseDriver]] = {
    ConstantSpeedPolicy.name: ConstantSpeedPolicy,
    RouteFollowerPolicy.name: RouteFollowerPolicy,
}

__all__ = ["POLICY_REGISTRY", "ConstantSpeedPolicy", "RouteFollowerPolicy"]
