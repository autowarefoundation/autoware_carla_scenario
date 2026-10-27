"""Ego vehicle driven by ECUs under test, through the resident HILS bridge.

``scenario-hils`` stays up on the HILS PC (docs/hils.md); a run with
``ego.entity=hils`` connects to it for each scenario
(:mod:`~autoware_carla_scenario.hils.protocol`):

* :meth:`HilsEgoEntity.on_scenario_start` tells the bridge the scenario starts
  and which actor the ego is; the bridge attaches its sensors, starts the
  ECUs' periodic inputs and runs its ``scenario_start`` steps.  The runner
  ticks the world meanwhile, holding every vehicle still, until the bridge
  answers ``ready`` (:attr:`HilsEgoEntity.is_initialized`) -- or fails the
  run when it reports an error or does not answer within ``ready_timeout_s``;
* every tick, :meth:`HilsEgoEntity.on_tick` applies the ECUs' latest command
  (CARLA's ``VehicleControl``, or ``VehicleAckermannControl``), which the
  bridge forwards whenever an ECU sends one.  A command older than
  ``stale_s`` (wall clock: the ECUs run in real time) is stale: the ego brakes,
  or holds the last command;
* :meth:`HilsEgoEntity.on_scenario_end` tells the bridge the scenario ends and
  waits for its ``scenario_end`` steps (``end_timeout_s``).

The link needs nothing beyond the standard library: hilspy, the ECUs and their
buses are the bridge's.
"""

from __future__ import annotations

import logging
import socket
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple

from ..hils.protocol import (
    COMMAND_FIELDS,
    DEFAULT_PORT,
    PROTOCOL_VERSION,
    LineReader,
    encode,
)
from .externally_driven import ExternallyDrivenEgo

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

logger = logging.getLogger(__name__)

__all__ = ["HilsEgoConfig", "HilsEgoEntity"]

#: What the ego does once the ECUs' command is stale.
ON_STALE_ACTIONS: Tuple[str, ...] = ("brake", "hold")


@dataclass(frozen=True)
class HilsEgoConfig:
    """Where the bridge is, and how long to wait for it (the ``hils`` group).

    Attributes:
        host, port: The bridge's ``runner.listen`` address.
        connect_timeout_s: For the connection and the bridge's greeting.
        ready_timeout_s: For the ``scenario_start`` steps (wall clock).
        end_timeout_s: For the ``scenario_end`` steps (wall clock).
        stale_s: A command older than this (wall clock) is stale.
        on_stale: ``"brake"`` or ``"hold"`` the last command once stale; before
            any command the ego brakes either way.
    """

    host: str = "localhost"
    port: int = DEFAULT_PORT
    connect_timeout_s: float = 10.0
    ready_timeout_s: float = 60.0
    end_timeout_s: float = 30.0
    stale_s: float = 0.5
    on_stale: str = "brake"

    def __post_init__(self) -> None:
        if self.on_stale not in ON_STALE_ACTIONS:
            raise ValueError(
                f"hils.on_stale must be one of {', '.join(ON_STALE_ACTIONS)}, "
                f"got {self.on_stale!r}"
            )
        if min(self.connect_timeout_s, self.ready_timeout_s, self.end_timeout_s) <= 0:
            raise ValueError("hils timeouts must be positive")
        if self.stale_s <= 0.0:
            raise ValueError("hils.stale_s must be positive")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "HilsEgoConfig":
        """The configuration the ``hils`` group's *values* give.

        Raises:
            ValueError: A key is not one of the configuration's, or a value is
                out of range.
        """
        known = cls.__dataclass_fields__
        unknown = sorted(set(values) - set(known))
        if unknown:
            raise ValueError(
                f"unknown hils setting(s) {', '.join(unknown)}; "
                f"expected {', '.join(known)}"
            )
        return cls(**dict(values))


class HilsEgoEntity(ExternallyDrivenEgo):
    """Ego vehicle whose command comes from ECUs, through the HILS bridge.

    Args:
        config: Where the bridge is.  ``None`` uses the defaults.
        episode_name: The scenario's name, which the bridge logs.
    """

    stack_name: str = "HILS"

    _config: HilsEgoConfig
    _episode_name: str
    _sock: Optional[socket.socket]
    _reader: Optional[LineReader]
    _ready: bool
    _start_sent: float
    _command: Optional[Tuple[str, Dict[str, float]]]
    _command_at: float
    _stale: bool

    def __init__(
        self, config: Optional[HilsEgoConfig] = None, episode_name: str = ""
    ) -> None:
        super().__init__()
        self._config = config if config is not None else HilsEgoConfig()
        self._episode_name = episode_name
        self._sock = None
        self._reader = None
        self._ready = False
        self._start_sent = 0.0
        self._command = None
        self._command_at = 0.0
        self._stale = False

    @property
    def config(self) -> HilsEgoConfig:
        """Return the settings."""
        return self._config

    @property
    def is_initialized(self) -> bool:
        """Whether the bridge has initialized the ECUs (``ready``)."""
        return self._ready

    @property
    def command(self) -> Optional[Tuple[str, Dict[str, float]]]:
        """The ECUs' latest command: ``(mode, fields)``, or ``None`` before any."""
        return self._command

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_scenario_start(self, world: "carla.World") -> None:
        """Reach the bridge and tell it the scenario starts.

        Readiness is awaited in ticks (:attr:`is_initialized`); a bridge that
        cannot be reached ends the run (:attr:`termination_requested`).

        Raises:
            RuntimeError: If the ego actor has not been spawned yet.
        """
        actor = self.actor
        if actor is None:
            raise RuntimeError("HilsEgoEntity.on_scenario_start called before spawn()")
        self._reset_termination()
        self._ready = False
        self._command = None
        self._stale = False
        cfg = self._config
        try:
            sock = socket.create_connection(
                (cfg.host, cfg.port), timeout=cfg.connect_timeout_s
            )
        except OSError as exc:
            self._terminate(
                f"The HILS bridge at {cfg.host}:{cfg.port} cannot be reached ({exc}); "
                "is scenario-hils running?"
            )
            return
        self._sock = sock
        self._reader = LineReader(sock)
        try:
            hello = self._reader.read(timeout=cfg.connect_timeout_s)
        except (OSError, ValueError) as exc:
            self._fail(f"The HILS bridge's greeting was broken ({exc})")
            return
        if not hello or hello[0].get("type") != "hello":
            self._fail(
                f"The HILS bridge at {cfg.host}:{cfg.port} did not greet within "
                f"{cfg.connect_timeout_s:.0f} s (another run connected?)"
            )
            return
        protocol = hello[0].get("protocol")
        if protocol != PROTOCOL_VERSION:
            self._fail(
                f"The HILS bridge speaks protocol {protocol}, this run "
                f"{PROTOCOL_VERSION}: update one of them"
            )
            return
        self._handle(hello[1:])
        if not self._send(
            {"type": "start", "scenario": self._episode_name, "ego_id": int(actor.id)}
        ):
            return
        self._start_sent = time.monotonic()
        logger.info(
            "Waiting for the HILS bridge at %s:%d to initialize the ECUs",
            cfg.host,
            cfg.port,
        )

    def on_tick(self, world: "carla.World", elapsed: float) -> None:
        """Take what the bridge sent, and apply the ECUs' command once ready."""
        actor = self.actor
        reader = self._reader
        if actor is None or reader is None or self.termination_requested:
            return
        try:
            self._handle(reader.read(timeout=0.0))
        except (OSError, ValueError) as exc:
            self._fail(f"The HILS bridge's link failed ({exc})")
            return
        if self.termination_requested:
            return
        if reader.closed:
            self._fail("The HILS bridge closed the link")
            return
        if not self._ready:
            waited = time.monotonic() - self._start_sent
            if waited > self._config.ready_timeout_s:
                self._fail(
                    "The HILS bridge did not initialize the ECUs within "
                    f"{self._config.ready_timeout_s:.0f} s"
                )
            return
        self._apply(actor)

    def on_scenario_end(self, world: "carla.World") -> None:
        """Tell the bridge the scenario ends and wait for its ``scenario_end``
        steps.  A failure is logged rather than raised."""
        reader = self._reader
        if reader is not None and not reader.closed and self._send({"type": "end"}):
            deadline = time.monotonic() + self._config.end_timeout_s
            ended = False
            while not ended and not reader.closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    logger.warning(
                        "The HILS bridge did not finalize the ECUs within %.0f s",
                        self._config.end_timeout_s,
                    )
                    break
                try:
                    messages = reader.read(timeout=remaining)
                except (OSError, ValueError):
                    break
                ended = any(m.get("type") == "ended" for m in messages)
            if ended:
                logger.info("The HILS bridge finalized the ECUs")
        self._close()

    def disconnect(
        self,
        cut_id: str,
        device: Optional[str] = None,
        messages: Optional[List[str]] = None,
        direction: str = "both",
    ) -> bool:
        """Have the bridge stop data between it and the ECUs (fault injection)
        until :meth:`reconnect` with the same *cut_id*, or the scenario's end.

        :class:`~autoware_carla_scenario.actions.CommunicationLossAction` is
        the way a scenario says so.

        Args:
            cut_id: Names this cut, for :meth:`reconnect`; cuts with other ids
                stay as they are.
            device: The bridge's device, by name; ``None`` for every device.
            messages: The names of the messages to stop; ``None`` for
                everything the device carries.
            direction: One of :data:`~autoware_carla_scenario.hils.protocol.LINK_DIRECTIONS`.

        Returns:
            Whether the request went to the bridge: not before it is reached,
            nor once the link to it failed.  A device the bridge does not have
            ends the run, as its other errors do.
        """
        if self._sock is None or self.termination_requested:
            logger.warning(
                "The HILS bridge is not connected; the data of %s is not stopped",
                "every device" if device is None else repr(device),
            )
            return False
        return self._send(
            {
                "type": "disconnect",
                "id": cut_id,
                "device": device,
                "messages": messages,
                "direction": direction,
            }
        )

    def reconnect(self, cut_id: str) -> bool:
        """Have the bridge carry the data the cut *cut_id* stopped again.

        Returns:
            Whether the request went to the bridge (when it did not, the
            bridge restores everything at the scenario's end).
        """
        if self._sock is None or self.termination_requested:
            return False
        return self._send({"type": "reconnect", "id": cut_id})

    # ------------------------------------------------------------------
    # The link
    # ------------------------------------------------------------------

    def _handle(self, messages: Any) -> None:
        for message in messages:
            kind = message.get("type")
            if kind == "ready":
                self._ready = True
                logger.info(
                    "The HILS bridge initialized the ECUs; scenario may proceed"
                )
            elif kind == "error":
                self._fail(f"The HILS bridge: {message.get('message')}")
            elif kind == "command":
                mode = str(message.get("mode"))
                fields = COMMAND_FIELDS.get(mode)
                if fields is None:
                    logger.warning("The HILS bridge sent a %r command; ignored", mode)
                    continue
                self._command = (
                    mode,
                    {
                        key: float(value)
                        for key, value in message.items()
                        if key in fields and isinstance(value, (int, float))
                    },
                )
                self._command_at = time.monotonic()
            elif kind == "ended":
                pass
            else:
                logger.warning("The HILS bridge sent an unknown message %r", kind)

    def _send(self, message: Mapping[str, Any]) -> bool:
        sock = self._sock
        if sock is None:
            return False
        try:
            sock.sendall(encode(message))
        except OSError as exc:
            self._fail(f"The HILS bridge's link failed ({exc})")
            return False
        return True

    def _fail(self, reason: str) -> None:
        self._terminate(reason)
        self._close()

    def _close(self) -> None:
        sock, self._sock, self._reader = self._sock, None, None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def _apply(self, actor: "carla.Actor") -> None:
        import typesafe_carla.carla as _carla  # noqa: PLC0415

        command = self._command
        fresh = (
            command is not None
            and time.monotonic() - self._command_at <= self._config.stale_s
        )
        if command is not None and fresh != (not self._stale):
            self._stale = not fresh
            if self._stale:
                logger.warning(
                    "The ECUs' command is stale; %s",
                    "holding it" if self._config.on_stale == "hold" else "braking",
                )
            else:
                logger.info("The ECUs' command is fresh again")
        if command is None or (not fresh and self._config.on_stale == "brake"):
            actor.apply_control(_carla.VehicleControl(brake=1.0))
            return
        mode, fields = command
        if mode == "ackermann":
            actor.apply_ackermann_control(
                _carla.VehicleAckermannControl(
                    steer=fields.get("steer", 0.0),
                    steer_speed=fields.get("steer_speed", 0.0),
                    speed=max(0.0, fields.get("speed", 0.0)),
                    acceleration=fields.get("acceleration", 0.0),
                    jerk=fields.get("jerk", 0.0),
                )
            )
            return
        actor.apply_control(
            _carla.VehicleControl(
                throttle=min(max(fields.get("throttle", 0.0), 0.0), 1.0),
                steer=min(max(fields.get("steer", 0.0), -1.0), 1.0),
                brake=min(max(fields.get("brake", 0.0), 0.0), 1.0),
                hand_brake=fields.get("hand_brake", 0.0) >= 0.5,
                reverse=fields.get("reverse", 0.0) >= 0.5,
                manual_gear_shift=fields.get("manual_gear_shift", 0.0) >= 0.5,
                gear=int(fields.get("gear", 0.0)),
            )
        )
