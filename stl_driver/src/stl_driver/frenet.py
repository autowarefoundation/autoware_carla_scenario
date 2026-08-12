# SPDX-License-Identifier: Apache-2.0
"""The route as a coordinate system: station along it, offset across it.

Planning in ``(s, d)`` rather than ``(x, y)`` is what makes the specification
expressible.  "Stay in the lane" is a bound on ``d``; "stop before the line" is
a bound on ``s``.  In Cartesian coordinates both are awkward functions of the
road's shape; here they are constants.

Everything in this module is numpy and happens once per ``drive()`` call --
the route does not depend on which plan we are considering, so resampling it
per candidate would be wasted work.  The torch side (:mod:`stl_driver.rollout`)
only ever indexes into the table this module builds.

The arc-length primitives come from ``carla_driver_interface.polyline``.  They
are already tested and already used by the runtime to lay out the route we are
handed, so reimplementing them here would risk the two disagreeing about where
the road is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from carla_driver_interface import polyline

__all__ = ["RouteTable", "build_route_table", "project_to_frenet"]


@dataclass(frozen=True)
class RouteTable:
    """A route resampled onto a uniform station grid.

    Uniform spacing is the entire trick.  It reduces the differentiable
    station lookup in :mod:`stl_driver.rollout` to index arithmetic -- no
    search, no sort -- and gives an exact piecewise-linear derivative, which a
    non-uniform grid would not.
    """

    #: Station spacing, metres. Constant by construction.
    ds: float
    #: ``(M, 2)`` centreline positions in the rig frame.
    xy: np.ndarray
    #: ``(M,)`` heights, carried through but never used for distance.
    z: np.ndarray
    #: ``(M, 2)`` unit tangents.
    tangent: np.ndarray
    #: Arc length of the route as published, before extrapolation. The table
    #: itself runs past this so that plan lookups never clamp; `stay_on_route`
    #: is what discourages planning beyond it.
    published_length_m: float

    def __len__(self) -> int:
        return len(self.xy)

    @property
    def table_length_m(self) -> float:
        return float((len(self.xy) - 1) * self.ds)

    def position_at(self, station_m: float) -> np.ndarray:
        """Centreline point at one station, as ``(x, y)``. For tests and debug."""
        index = np.clip(station_m / self.ds, 0.0, len(self.xy) - 1.0)
        low = int(np.floor(index))
        high = min(low + 1, len(self.xy) - 1)
        weight = index - low
        return self.xy[low] * (1.0 - weight) + self.xy[high] * weight

    def frenet_to_xy(self, station_m: float, offset_m: float) -> np.ndarray:
        """Inverse of :func:`project_to_frenet`, for round-trip tests."""
        index = np.clip(station_m / self.ds, 0.0, len(self.xy) - 1.0)
        low = int(np.floor(index))
        high = min(low + 1, len(self.xy) - 1)
        weight = index - low
        point = self.xy[low] * (1.0 - weight) + self.xy[high] * weight
        tangent = self.tangent[low] * (1.0 - weight) + self.tangent[high] * weight
        tangent = tangent / max(float(np.linalg.norm(tangent)), 1e-9)
        normal = np.array([-tangent[1], tangent[0]])
        return point + offset_m * normal


def build_route_table(
    route_in_rig: np.ndarray,
    *,
    ds: float,
    length_m: float,
) -> RouteTable:
    """Resample a route polyline onto a uniform station grid.

    Args:
        route_in_rig: ``(N, 3)`` route waypoints in the rig frame, as the
            runtime publishes them.
        ds: Station spacing for the resampled table.
        length_m: How far the table must reach.  Should exceed anything the
            fastest plan could travel, so that the lookup in
            :mod:`stl_driver.rollout` never hits its clamp and loses gradient.

    Raises:
        ValueError: if the route has fewer than two distinct points.
    """
    route = np.asarray(route_in_rig, dtype=np.float64).reshape(-1, 3)
    if len(route) < 2:
        raise ValueError(f"a route needs at least two points, got {len(route)}")

    arc = polyline.arc_lengths(route)
    published_length = float(arc[-1])
    if published_length < 1e-6:
        raise ValueError("route has zero length")

    # Extrapolating past the published end is required, not optional: the
    # runtime publishes an 80 m horizon and four seconds at 20 m/s is exactly
    # 80 m, so a fast plan runs off the end on an ordinary step. `polyline`
    # continues along the final segment's heading, which keeps the table
    # smooth where a clamp would put a kink -- and a kink in the table is a
    # discontinuity in the gradient.
    stations = np.arange(0.0, length_m + ds, ds)
    sampled = polyline.sample(route, arc, stations, extrapolate=True)

    xy = sampled[:, :2]
    z = sampled[:, 2]

    # Central differences give a tangent that is continuous across vertices,
    # unlike per-segment headings which jump at every one.
    deltas = np.gradient(xy, axis=0)
    norms = np.linalg.norm(deltas, axis=1, keepdims=True)
    tangent = deltas / np.maximum(norms, 1e-9)

    return RouteTable(
        ds=float(ds),
        xy=np.ascontiguousarray(xy),
        z=np.ascontiguousarray(z),
        tangent=np.ascontiguousarray(tangent),
        published_length_m=published_length,
    )


def project_to_frenet(table: RouteTable, points_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project points onto the route, returning station and signed offset.

    The offset is positive to the **left** of the direction of travel, which
    is the rig frame's ``+y``.  Getting that sign backwards is the single
    easiest mistake here and produces a policy that steers confidently the
    wrong way, so :mod:`tests.test_frenet` pins it explicitly.

    Args:
        table: the route to project onto.
        points_xy: ``(N, 2)`` points in the rig frame.

    Returns:
        ``(station, offset)``, each ``(N,)``.
    """
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)

    # Nearest grid station first, then an exact projection onto the segment it
    # belongs to. The two-step approach costs one distance matrix and is exact
    # to within the segment, which a nearest-vertex answer would not be.
    deltas = points[:, None, :] - table.xy[None, :, :]
    nearest = np.argmin(np.einsum("nmk,nmk->nm", deltas, deltas), axis=1)
    nearest = np.clip(nearest, 0, len(table.xy) - 2)

    base = table.xy[nearest]
    segment = table.xy[nearest + 1] - base
    segment_length_sq = np.einsum("nk,nk->n", segment, segment)

    offset_vector = points - base
    # Fraction along the segment, clamped so a point beyond either end
    # projects onto the endpoint rather than off the segment.
    fraction = np.where(
        segment_length_sq > 1e-12,
        np.einsum("nk,nk->n", offset_vector, segment) / np.maximum(segment_length_sq, 1e-12),
        0.0,
    )
    fraction = np.clip(fraction, 0.0, 1.0)

    station = (nearest + fraction) * table.ds

    foot = base + fraction[:, None] * segment
    residual = points - foot
    tangent = table.tangent[nearest]
    # 2-D cross product: positive when the point lies to the left of the
    # tangent, matching the rig frame's +y.
    offset = tangent[:, 0] * residual[:, 1] - tangent[:, 1] * residual[:, 0]

    return station, offset
