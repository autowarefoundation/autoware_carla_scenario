"""Communication loss action: stop data between the HILS bridge and its ECUs."""

from __future__ import annotations

import itertools
import logging
from typing import TYPE_CHECKING, Optional, Sequence, Union

from ..conditions.base import BaseCondition, ScenarioResult
from ..constants import EGO_ROLE_NAME
from ..entity.hils_entity import HilsEgoEntity
from ..entity.registry import find_vehicle_entity
from ..entity_role import EntityRole
from ..hils.protocol import LINK_DIRECTIONS
from .base import BaseAction, TickTiming

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

logger = logging.getLogger(__name__)

__all__ = ["CommunicationLossAction"]

#: Tells the cuts of one run apart: several actions may cut the same device.
_CUT_IDS = itertools.count(1)


def _names(messages: Union[str, Sequence[str], None]) -> Optional[tuple[str, ...]]:
    """*messages* as names: a sequence, or one string of comma-separated names
    (an editor's text field); ``None`` or nothing named for every message."""
    if messages is None:
        return None
    parts = messages.split(",") if isinstance(messages, str) else list(messages)
    names = tuple(dict.fromkeys(str(p).strip() for p in parts if str(p).strip()))
    return names or None


class CommunicationLossAction(BaseAction):
    """Stop data on the way to or from an ECU under test, while the action runs.

    For an ego driven by ECUs through the HILS bridge (``ego.entity=hils``,
    docs/hils.md): it reproduces a break in the communication path -- a CAN
    frame that stops arriving, a LiDAR or camera whose packets stop, a GNSS
    receiver that falls silent.  The bridge stops carrying the data named
    (*messages* of *device*, or everything the device carries) through its
    hilspy channel (``hilspy.Channel.disconnect``), while the bus or socket
    stays open and the device's other traffic carries on.  Periodic messages
    keep being encoded meanwhile (an E2E counter moves on) and only their
    frames are dropped, as on a cut wire.

    The data stays stopped **for as long as the action runs**: from the tick
    its *condition* fires until *until* fires or *duration* seconds of the
    scenario have passed, whichever comes first, or until the scenario ends
    when neither is given.  Then the bridge carries it again.  On any other
    ego the action only logs a warning, so one scenario serves HILS and
    simulation runs alike.

    Args:
        device: The bridge configuration's device, by name; ``None`` for every
            device.
        messages: The messages to stop, by the names the device's codec gives
            them (``"VehicleSpeed"`` of a DBC, ``"scan"`` of a LiDAR): a
            sequence, or one string of comma-separated names.  ``None`` stops
            everything the device carries (camera frames, which have no
            name, included).
        direction: ``"to_ecu"`` (what the simulation sends the ECU),
            ``"from_ecu"`` (what the ECU sends back) or ``"both"``.
        duration: Seconds of scenario time the data stays stopped; ``None``
            leaves the end to *until*, or to the end of the scenario.
        entity_name: ``role_name`` of the HILS ego.
        condition: Trigger condition (see :class:`BaseCondition`).
        timing: Tick phase (``PRE_TICK`` or ``POST_TICK``).
        label: Human-readable identifier.
        once: If ``True`` (default) the action fires at most once.
        until: A condition that ends the run, and with it the loss.

    Raises:
        ValueError: *direction* is not one of the three, or *duration* is not
            positive.
    """

    # Declared for the static check (docs/typecheck.md).
    _entity_name: str
    _device: Optional[str]
    _messages: Optional[tuple[str, ...]]
    _duration: Optional[float]
    _direction: str
    _end: Optional[BaseCondition]
    _now: float
    _started_at: float
    _cut: Optional[tuple[HilsEgoEntity, str]]

    def __init__(
        self,
        device: Optional[str] = None,
        messages: Union[str, Sequence[str], None] = None,
        direction: str = "both",
        duration: Optional[float] = None,
        entity_name: Union[EntityRole, str] = EGO_ROLE_NAME,
        condition: Optional[BaseCondition] = None,
        timing: TickTiming = TickTiming.PRE_TICK,
        *,
        label: str = "communication_loss",
        once: bool = True,
        until: Optional[BaseCondition] = None,
    ) -> None:
        if direction not in LINK_DIRECTIONS:
            raise ValueError(
                f"CommunicationLossAction: direction must be one of "
                f"{', '.join(LINK_DIRECTIONS)}, got {direction!r}"
            )
        if duration is not None and not duration > 0.0:
            raise ValueError(
                f"CommunicationLossAction: duration must be positive, got {duration}"
            )
        # The run lasts until the loss ends, so it is never instantaneous.
        restored: BaseCondition = _Restored(self, label)
        super().__init__(
            label=label, condition=condition, timing=timing, once=once, until=restored
        )
        self._entity_name = str(entity_name)
        # An editor's empty field means every device too.
        self._device = device or None
        self._messages = _names(messages)
        self._direction = direction
        self._duration = None if duration is None else float(duration)
        self._end = until
        self._now = 0.0
        self._started_at = 0.0
        self._cut = None

    @property
    def device(self) -> Optional[str]:
        """The device whose data is stopped, or ``None`` for every device."""
        return self._device

    @property
    def messages(self) -> Optional[tuple[str, ...]]:
        """The names of the messages stopped, or ``None`` for everything."""
        return self._messages

    @property
    def duration(self) -> Optional[float]:
        """Seconds the data stays stopped, or ``None``."""
        return self._duration

    @property
    def direction(self) -> str:
        """``"to_ecu"``, ``"from_ecu"`` or ``"both"``."""
        return self._direction

    @property
    def cutting(self) -> bool:
        """Whether the bridge has been asked to stop the data and not yet to
        carry it again."""
        return self._cut is not None

    def tick(self, world: "carla.World", elapsed: float) -> None:
        """Evaluate as :meth:`BaseAction.tick`; *elapsed* times *duration*."""
        self._now = elapsed
        super().tick(world, elapsed)

    def execute(self, world: "carla.World") -> None:
        """Have the HILS bridge stop the data."""
        self._started_at = self._now
        entity = find_vehicle_entity(self._entity_name)
        if not isinstance(entity, HilsEgoEntity):
            logger.warning(
                "CommunicationLossAction: '%s' is not driven through the HILS "
                "bridge (ego.entity=hils); there is no communication to stop",
                self._entity_name,
            )
            return
        cut_id = f"{self.label}-{next(_CUT_IDS)}"
        messages = None if self._messages is None else list(self._messages)
        if entity.disconnect(cut_id, self._device, messages, self._direction):
            self._cut = (entity, cut_id)
            logger.info(
                "CommunicationLossAction: %s of %s stopped (%s)",
                "every message" if messages is None else ", ".join(messages),
                "every device" if self._device is None else repr(self._device),
                self._direction,
            )

    def restore(self) -> None:
        """Have the bridge carry the data again (idempotent)."""
        cut, self._cut = self._cut, None
        if cut is None:
            return
        entity, cut_id = cut
        if entity.reconnect(cut_id):
            logger.info("CommunicationLossAction: '%s' restored", self.label)

    def close(self) -> None:
        """Restore the data, and close the conditions, at the run's end."""
        self.restore()
        super().close()


class _Restored(BaseCondition):
    """Ends a :class:`CommunicationLossAction`'s run -- once its *until* fires
    or its *duration* has passed -- and restores the data as it does."""

    _action: CommunicationLossAction

    def __init__(self, action: CommunicationLossAction, label: str) -> None:
        super().__init__(label=f"{label}_restored")
        self._action = action

    def check(self, world: "carla.World", elapsed: float) -> Optional[ScenarioResult]:
        action = self._action
        result: Optional[ScenarioResult] = None
        end = action._end
        if end is not None:
            result = end.check(world, elapsed)
        duration = action.duration
        if (
            result is None
            and duration is not None
            and elapsed - action._started_at >= duration - 1e-9
        ):
            result = ScenarioResult(
                passed=True,
                message=f"{action.label}: {duration:g} s passed",
                elapsed_seconds=elapsed,
            )
        if result is not None:
            action.restore()
        return result

    def close(self) -> None:
        end = self._action._end
        if end is not None:
            end.close()
