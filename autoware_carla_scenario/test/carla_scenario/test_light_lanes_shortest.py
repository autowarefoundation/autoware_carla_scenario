"""``light_lanes.elements_ahead`` reaches each lanelet by its shortest path (#2266).

The walk used to mark a lanelet seen on its first visit, so a long branch
reaching a lanelet first hid a shorter one reaching it later: whether a light
just past a merge was found hung on the order the map listed successors in.
"""

from __future__ import annotations

import numpy as np
import pytest

from autoware_carla_scenario.utils.light_lanes import (
    LIGHT_SEARCH_M,
    LightLanes,
    elements_ahead,
)


def _square(x0: float) -> np.ndarray:
    return np.array([[x0, 0], [x0 + 1, 0], [x0 + 1, 1], [x0, 1]], dtype=np.float64)


def _lanes(
    successors: dict[int, tuple[int, ...]],
    lengths: dict[int, float],
    elements: dict[int, tuple[int, ...]],
) -> LightLanes:
    ids = tuple(sorted(lengths))
    outlines = tuple(_square(100.0 * k) for k in range(len(ids)))
    boxes = np.array([[*o.min(0), *o.max(0)] for o in outlines], dtype=np.float64)
    return LightLanes(ids, outlines, boxes, lengths, elements, successors)


# 1 (the point's) -> 2 (40 m) and 3 (5 m) -> 4 (30 m) -> 5 (light 99): the
# short path to 5 is 35 m, inside LIGHT_SEARCH_M; the long one, 70 m, is not.
_LENGTHS = {1: 10.0, 2: 40.0, 3: 5.0, 4: 30.0, 5: 10.0}
_ELEMENTS: dict[int, tuple[int, ...]] = {1: (), 2: (), 3: (), 4: (), 5: (99,)}


@pytest.mark.parametrize("first", [(2, 3), (3, 2)])
def test_a_light_past_a_merge_is_found_whatever_the_successor_order(
    first: tuple[int, int],
) -> None:
    assert 5.0 + 30.0 <= LIGHT_SEARCH_M < 40.0 + 30.0
    lanes = _lanes({1: first, 2: (4,), 3: (4,), 4: (5,), 5: ()}, _LENGTHS, _ELEMENTS)
    assert elements_ahead(lanes, np.array([0.5, 0.5])) == [99]


def test_a_light_beyond_reach_on_every_path_is_not_found() -> None:
    lengths = {**_LENGTHS, 3: 30.0}
    lanes = _lanes({1: (2, 3), 2: (4,), 3: (4,), 4: (5,), 5: ()}, lengths, _ELEMENTS)
    assert elements_ahead(lanes, np.array([0.5, 0.5])) == []


def test_elements_come_nearest_lanelet_first() -> None:
    # 1 -> 2 (20 m, then light 7 on 4) and 3 (light 8): 3 is nearer than 4.
    lanes = _lanes(
        {1: (2, 3), 2: (4,), 3: (), 4: ()},
        {1: 10.0, 2: 20.0, 3: 5.0, 4: 5.0},
        {1: (), 2: (), 3: (8,), 4: (7,)},
    )
    assert elements_ahead(lanes, np.array([0.5, 0.5])) == [8, 7]
