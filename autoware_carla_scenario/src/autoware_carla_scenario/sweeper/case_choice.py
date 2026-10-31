"""Which base case of a sweep each case drawn from the ODD is put on.

The pure half of :mod:`.expand`'s ODD sampling over a sweep with cases of its
own (constraints or a route search): the lanelet a base case is on, read off
its overrides; whether a drawn case's buckets for the attributes the map
decides fit a base case's; and, for each drawn case, the base case on a
lanelet in the most of the buckets it drew, the one run least so far among
equals.  A drawn case is given here by its buckets (``sample.buckets``), so
nothing here reads the sampler, the map or a config, and the library check
compiles it (docs/typecheck.md); :mod:`.expand` uses it.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "case_lanelet",
    "choose_cases",
    "fits",
]


def case_lanelet(case: Sequence[str]) -> Optional[int]:
    """The lanelet a base case is on: its first ``*lanelet_id`` override (the
    constraints' pick, or the route's spawn)."""
    for item in case:
        key, _, value = item.lstrip("+").partition("=")
        if key.endswith("lanelet_id"):
            try:
                return int(value)
            except ValueError:
                return None
    return None


def fits(
    drawn: Dict[str, str], buckets: Dict[str, str], attributes: Sequence[str]
) -> bool:
    """Whether a base case in *buckets* is in every bucket a drawn case drew
    (*drawn*, its ``buckets``) for those of *attributes* it drew."""
    for a in attributes:
        if a in drawn and buckets.get(a) != drawn[a]:
            return False
    return True


def choose_cases(
    drawn: Sequence[Dict[str, str]],
    buckets: List[Dict[str, str]],
    attributes: Sequence[str],
) -> Tuple[List[int], int]:
    """The base case for each drawn case (*drawn*, their ``buckets``), by
    index: one on a lanelet in every bucket the case drew for *attributes*,
    else in as many of them as any; among those, the one run least so far
    (the first of equals), so the cases are spread over as many lanelets as
    the buckets allow.  Returns the indices and how many matched fully.

    Raises ValueError if there is a drawn case but no base case."""
    chosen: List[int] = []
    uses: List[int] = [0] * len(buckets)
    full = 0
    for sample in drawn:
        wanted: Dict[str, str] = {}
        for a in attributes:
            if a in sample:
                wanted[a] = sample[a]
        scores: List[int] = []
        for b in buckets:
            score = 0
            for a, label in wanted.items():
                if b.get(a) == label:
                    score += 1
            scores.append(score)
        best = max(scores) if scores else 0
        if best == len(wanted):
            full += 1
        # The first of the best-scoring cases that is run least so far.
        index = -1
        for i in range(len(scores)):
            if scores[i] == best and (index < 0 or uses[i] < uses[index]):
                index = i
        if index < 0:
            raise ValueError("min() arg is an empty sequence")
        uses[index] += 1
        chosen.append(index)
    return chosen, full
