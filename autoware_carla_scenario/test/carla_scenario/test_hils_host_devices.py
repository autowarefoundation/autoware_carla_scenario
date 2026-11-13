"""The virtual devices ``scenario-hils --setup`` derives, pinned.

Pins, against verbatim copies of them as they were in ``hils.host_setup``
(the ``_old_*`` oracles below), what ``host_setup()`` derives from a
configuration's devices -- the vcan interfaces and whether each is CAN FD,
the v4l2loopback devices and their labels, the devices left alone -- or
raises, message for message, in the order the devices are validated (every
device's ``fd`` flag, #2127); what ``HostSetup.describe()`` and
``HostSetup.playbooks()`` make of it; and how ``_flag()`` reads a boolean
query option and ``_label()`` cuts a card label.

Devices are random: schemes of every case, interface names at and past
Linux's 15 characters, video devices of any digits, queries naming ``fd``
once, twice or as a blank, spellings of every case and percent-escaped, and
names with quotes, commas and multi-byte characters past 31 bytes.
"""

from __future__ import annotations

import random
import re
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qsl, urlsplit

import pytest

from autoware_carla_scenario.hils import host_setup as host_setup_module
from autoware_carla_scenario.hils import parse_config
from autoware_carla_scenario.hils.host_setup import (
    HostSetup,
    HostSetupError,
    LoopbackCamera,
    VcanInterface,
)

VCAN_PREFIX = "vcan"
_LABEL_BYTES = 31
_INTERFACE = re.compile(r"[A-Za-z0-9_.-]{1,15}")
_VIDEO_DEVICE = re.compile(r"/dev/video([0-9]+)")
_TRUE = ("", "1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")


# ---------------------------------------------------------------------------
# The oracles: hils.host_setup's device planning as it was, verbatim but for
# the names of the helpers it calls.
# ---------------------------------------------------------------------------


def _old_playbooks(
    self: HostSetup, user: Optional[str] = None
) -> List[Tuple[str, Dict[str, Any]]]:
    runs: List[Tuple[str, Dict[str, Any]]] = []
    if self.vcan:
        runs.append(
            (
                "vcan.yml",
                {"vcan_interfaces": [{"name": i.name, "fd": i.fd} for i in self.vcan]},
            )
        )
    if self.cameras:
        runs.append(
            (
                "v4l2loopback.yml",
                {
                    "v4l2loopback_numbers": [c.number for c in self.cameras],
                    "v4l2loopback_labels": [c.label for c in self.cameras],
                    "v4l2loopback_exclusive_caps": True,
                    "hilspy_user": user if user is not None else _old_user(),
                    # The wheel hilspy is installed from has no v4l2
                    # feature to build; the bridge only needs the devices.
                    "hilspy_build": False,
                },
            )
        )
    return runs


def _old_describe(self: HostSetup) -> str:
    """What the setup does, one device a line."""
    lines = [f"  vcan {i.name}" + (" (CAN FD)" if i.fd else "") for i in self.vcan]
    lines += [
        f"  v4l2loopback /dev/video{c.number} ({c.label!r})" for c in self.cameras
    ]
    lines += [f"  left alone: {device}: {why}" for device, why in self.untouched]
    return "\n".join(lines)


def _old_user() -> str:
    import getpass
    import os

    return os.environ.get("SUDO_USER") or getpass.getuser()


def _old_flag(query: str, key: str, where: str) -> bool:
    values = [v for k, v in parse_qsl(query, keep_blank_values=True) if k == key]
    if not values:
        return False
    value = values[-1].lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise HostSetupError(f"{where}: {key}={values[-1]!r} is not a boolean")


def _old_label(name: str) -> str:
    label = re.sub(r'[",]', "_", name)
    while len(label.encode("utf-8")) > _LABEL_BYTES:
        label = label[:-1]
    return label


def _old_host_setup(config: Any) -> HostSetup:
    vcan: Dict[str, bool] = {}
    cameras: Dict[int, LoopbackCamera] = {}
    untouched: List[Tuple[str, str]] = []
    for device in config.devices:
        where = f"devices.{device.name}.url"
        url = urlsplit(device.url)
        if url.scheme == "can":
            name = url.netloc or url.path
            if not name.startswith(VCAN_PREFIX):
                untouched.append(
                    (device.name, f"{name} is not a vcan interface (a CAN adapter)")
                )
                continue
            if not _INTERFACE.fullmatch(name):
                raise HostSetupError(
                    f"{where}: {name!r} is not an interface name (at most 15 "
                    "letters, digits, '_', '.' or '-')"
                )
            # Read the flag first: every device's ``fd`` is validated, even
            # when an earlier one on the interface already made it FD.
            fd = _old_flag(url.query, "fd", where)
            vcan[name] = vcan.get(name, False) or fd
        elif url.scheme == "v4l2":
            match = _VIDEO_DEVICE.fullmatch(url.netloc + url.path)
            if match is None:
                raise HostSetupError(
                    f"{where}: v4l2loopback creates /dev/videoN devices, not "
                    f"{url.netloc + url.path!r}"
                )
            number = int(match.group(1))
            if number in cameras:
                raise HostSetupError(
                    f"{where}: /dev/video{number} is already "
                    f"{cameras[number].label!r}'s; give each camera its own device"
                )
            cameras[number] = LoopbackCamera(number, _old_label(device.name))
        else:
            untouched.append((device.name, f"{url.scheme}:// needs no virtual device"))
    return HostSetup(
        vcan=tuple(VcanInterface(name, fd) for name, fd in vcan.items()),
        cameras=tuple(cameras[n] for n in sorted(cameras)),
        untouched=tuple(untouched),
    )


# ---------------------------------------------------------------------------
# Comparing
# ---------------------------------------------------------------------------


def _outcome(function: Callable[..., Any], *args: Any) -> Tuple[Any, ...]:
    try:
        value = function(*args)
    except Exception as exc:  # noqa: BLE001 - the oracle's exceptions are compared
        return ("raised", type(exc), exc.args)
    return ("returned", type(value), value)


def _setup_outcome(function: Callable[[Any], HostSetup], config: Any) -> Any:
    found = _outcome(function, config)
    if found[0] == "raised":
        return found
    setup: HostSetup = found[2]
    return (
        type(setup),
        [(type(i), i.name, i.fd) for i in setup.vcan],
        [(type(c), c.number, c.label) for c in setup.cameras],
        list(setup.untouched),
        type(setup.vcan),
        type(setup.cameras),
        type(setup.untouched),
        setup,
        _outcome(setup.describe),
        _outcome(_old_describe, setup),
        _outcome(setup.playbooks, "someone"),
        _outcome(_old_playbooks, setup, "someone"),
    )


def _config(devices: List[Tuple[str, str]]) -> Any:
    return SimpleNamespace(
        devices=tuple(SimpleNamespace(name=n, url=u) for n, u in devices)
    )


def _same_setup(config: Any) -> None:
    new = _setup_outcome(host_setup_module.host_setup, config)
    old = _setup_outcome(_old_host_setup, config)
    assert new == old, config
    if new[0] != "raised":
        # describe() and playbooks() as they were, on the setup derived.
        assert new[8] == new[9] and new[10] == new[11]


_SCHEMES = ["can", "CAN", "v4l2", "V4L2", "udp", "tcp", "serial", "", "can+fd", "v4l"]
_INTERFACES = [
    "vcan0",
    "vcan1",
    "vcan0123456789a",  # 15
    "vcan0123456789ab",  # 16
    "vcan",
    "vcan_x.y-z",
    "vcan 0",
    "vcan/0",
    "vcanü",
    "Vcan0",
    "can0",
    "",
    "vcan0:1",
]
_FD = [
    "fd",
    "fd=",
    "fd=1",
    "fd=0",
    "fd=TRUE",
    "fd=Yes",
    "fd=on",
    "fd=OFF",
    "fd=no",
    "fd=false",
    "fd=maybe",
    "fd=2",
    "fd=%31",
    "fd=+",
    "fd=%20",
    "fd=tru%C3%A9",
    "fd=%ff",
    "FD=1",
    "fd=1&fd=maybe",
    "fd=maybe&fd=1",
    "fd=1&fd=0",
    "fd=0&fd=1",
    "&fd=on&",
    "x=1",
    "fd=1;fd=x",
    "f%64=1",
    "",
]
_VIDEOS = [
    "/dev/video10",
    "/dev/video0",
    "/dev/video010",
    "/dev/video",
    "/dev/videoX",
    "/dev/video١",
    "/dev/camera",
    "/dev/video10/",
    "/dev/video99999999999999999999",
    "",
]
_NAMES = [
    "cam",
    "front_camera",
    'a"b',
    "a,b",
    "x" * 40,
    "ü" * 20,
    "日本語のカメラの名前はとても長い",
    "",
    '",' * 20,
    "bus.can",
]


def _random_url(rng: random.Random) -> str:
    scheme = rng.choice(_SCHEMES)
    roll = rng.random()
    if scheme.lower() == "v4l2" or roll < 0.2:
        video = rng.choice(_VIDEOS)
        form = rng.random()
        if form < 0.7:
            url = f"{scheme}://{video}"
        elif form < 0.85:
            url = f"{scheme}://dev{video[4:]}"
        else:
            url = f"{scheme}:{video}"
    else:
        interface = rng.choice(_INTERFACES)
        url = (
            f"{scheme}://{interface}" if rng.random() < 0.8 else f"{scheme}:{interface}"
        )
    if rng.random() < 0.7:
        url += "?" + rng.choice(_FD)
    if rng.random() < 0.05:
        url += "#fd=1"
    if rng.random() < 0.02:
        url = rng.choice(["can://[vcan0", "v4l2://[::1", "can://vcan0]"])
    return url


@pytest.mark.parametrize("seed", range(300))
def test_random_devices_are_planned_as_before(seed: int) -> None:
    rng = random.Random(seed)
    devices = [
        (rng.choice(_NAMES) + str(i), _random_url(rng))
        for i in range(rng.choice([0, 1, 2, 3, 5, 8]))
    ]
    _same_setup(_config(devices))


@pytest.mark.parametrize("first", _FD)
@pytest.mark.parametrize("second", _FD)
def test_two_devices_on_one_interface_are_planned_as_before(
    first: str, second: str
) -> None:
    # Every pair of flags on a shared interface, each validated in order,
    # with a device on another interface and a camera between them.
    _same_setup(
        _config(
            [
                ("a", f"can://vcan0?{first}"),
                ("cam", "v4l2:///dev/video3"),
                ("b", f"can://vcan0?{second}"),
                ("c", "can://vcan1"),
            ]
        )
    )


@pytest.mark.parametrize("url", ["can://vcan0?fd=maybe", "v4l2:///dev/camera"])
def test_the_first_bad_device_is_reported(url: str) -> None:
    # Two bad devices: the first one's message, the second never read.
    config = _config(
        [("ok", "can://vcan0?fd"), ("bad1", url), ("bad2", "can://vcan_far_too_long0")]
    )
    _same_setup(config)
    with pytest.raises(HostSetupError, match="bad1"):
        host_setup_module.host_setup(config)


def test_a_parsed_configuration_is_planned_as_before() -> None:
    config = parse_config(
        {
            "devices": {
                "body_can": {"url": "can://vcan0?fd"},
                "chassis": {"url": "can://vcan1"},
                "rear": {"url": "v4l2:///dev/video12?width=640"},
                "front": {"url": "v4l2:///dev/video10"},
                "lidar": {"url": "udp://192.168.1.201:2368"},
                "adapter": {"url": "can://can0"},
            }
        }
    )
    _same_setup(config)


@pytest.mark.parametrize("key", ["fd", "x", "", "FD"])
def test_a_flag_is_read_as_before(key: str) -> None:
    for query in _FD + ["fd=1&x=yes", "x", "x=", "=1", "fd=1&=x&x=off"]:
        assert _outcome(host_setup_module._flag, query, key, "w") == _outcome(
            _old_flag, query, key, "w"
        )


def test_a_label_is_cut_as_before() -> None:
    rng = random.Random(7)
    names = list(_NAMES)
    for _ in range(300):
        names.append(
            "".join(rng.choice('ab",ü日😀\0\n ') for _ in range(rng.randrange(0, 45)))
        )
    names.append("a\udcff" + "b" * 40)  # a lone surrogate: UTF-8 cannot hold it
    for name in names:
        assert _outcome(host_setup_module._label, name) == _outcome(_old_label, name)


def test_the_default_user_is_as_before(monkeypatch: pytest.MonkeyPatch) -> None:
    setup = HostSetup(cameras=(LoopbackCamera(1, "c"),))
    for user in ("someone", "", None):
        if user is None:
            monkeypatch.delenv("SUDO_USER", raising=False)
        else:
            monkeypatch.setenv("SUDO_USER", user)
        assert _outcome(setup.playbooks) == _outcome(_old_playbooks, setup)


def test_the_constants_are_the_oracles() -> None:
    assert host_setup_module.VCAN_PREFIX == VCAN_PREFIX
    assert host_setup_module._LABEL_BYTES == _LABEL_BYTES
    assert host_setup_module._INTERFACE.pattern == _INTERFACE.pattern
    assert host_setup_module._VIDEO_DEVICE.pattern == _VIDEO_DEVICE.pattern
    assert host_setup_module._TRUE == _TRUE
    assert host_setup_module._FALSE == _FALSE


class _OldHostSetupError(Exception):
    """The setup cannot run (a device it cannot create, no ansible-runner, a
    hilspy without the playbooks)."""


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [((), {}), (("one",), {}), (("a", "b"), {}), ((1, None), {}), ((), {"x": 1})],
)
def test_the_error_is_built_as_before(
    args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> None:
    def build(cls: Any) -> Any:
        found = _outcome(lambda: cls(*args, **kwargs))
        if found[0] == "raised":
            return (found[0], found[1], str(found[2]).replace(cls.__name__, "E"))
        error = found[2]
        return (
            error.args,
            str(error),
            repr(error).replace(cls.__name__, "E"),
            type(error).__mro__[1:],
        )

    assert build(HostSetupError) == build(_OldHostSetupError)
