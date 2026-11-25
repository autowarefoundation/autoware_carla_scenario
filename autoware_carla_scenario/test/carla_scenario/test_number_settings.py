"""One finite-number check, one port check and one angle wrap (#2398).

The copies of each had drifted: ``ego.initial_speed_kmh=true`` ran at 1 km/h
on one path and was refused on another, an integer too big for a float
(``10**400``) raised an ``OverflowError`` naming no key on some, a port
check formatted a huge integer with ``repr()`` (which raises), and the SUMO
backend's angle wrap put ``-180.00000000000003`` on ``+180``.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable

import numpy as np
import pytest

from autoware_carla_scenario.number_settings import (
    check_port,
    finite_number_setting,
    is_port,
    number_setting,
)

HUGE = 10**400
#: Too long for ``repr()`` (``sys.get_int_max_str_digits()``, 4300).
HUGER = 10**5000


# ---------------------------------------------------------------------------
# The canonical checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [True, False])
@pytest.mark.parametrize("text", [False, True])
def test_a_boolean_is_no_number(value: bool, text: bool) -> None:
    for check in (number_setting, finite_number_setting):
        with pytest.raises(ValueError, match=rf"^k must be a number, got {value}$"):
            check("k", value, text=text)


@pytest.mark.parametrize(
    "value",
    [np.True_, np.False_, np.array(True), np.array(False)],
    ids=["True_", "False_", "0d-true", "0d-false"],
)
@pytest.mark.parametrize("text", [False, True])
def test_a_numpy_boolean_is_no_number(value: Any, text: bool) -> None:
    """``float(np.True_)`` is 1.0, as ``seed.check_whole_number`` refuses
    it (#1900, #2507)."""
    for check in (number_setting, finite_number_setting):
        with pytest.raises(ValueError, match=r"^k must be a number, got "):
            check("k", value, text=text)


def test_numpy_numbers_are_still_numbers() -> None:
    assert number_setting("k", np.float64(1.5)) == 1.5
    assert number_setting("k", np.int32(2)) == 2.0  # type: ignore[arg-type]
    assert number_setting("k", np.array(2.5)) == 2.5  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value", [HUGE, -HUGE, HUGER], ids=["1e400", "-1e400", "1e5000"]
)
def test_an_int_too_big_for_a_float_is_refused_naming_the_key(value: int) -> None:
    for check in (number_setting, finite_number_setting):
        with pytest.raises(ValueError, match=r"^a\.key must be finite, got "):
            check("a.key", value)


def test_a_huge_int_is_shown_without_repr() -> None:
    with pytest.raises(ValueError, match=r"got an integer of about 10\*\*5000$"):
        finite_number_setting("k", HUGER)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_finite_refuses_nan_and_infinities(value: float) -> None:
    assert math.isnan(number_setting("k", value)) or math.isinf(value)
    with pytest.raises(ValueError, match=r"^k must be finite, got "):
        finite_number_setting("k", value)


@pytest.mark.parametrize("value", [0, -3, 2.5, 1e308, -0.0, 10**300])
def test_a_number_is_read_as_its_float(value: float) -> None:
    for check in (number_setting, finite_number_setting):
        number = check("k", value)
        assert number == float(value) and type(number) is float


def test_text_is_read_only_where_a_caller_asks() -> None:
    with pytest.raises(ValueError, match=r"^k must be a number, got '1\.5'$"):
        number_setting("k", "1.5")
    assert number_setting("k", "1.5", text=True) == 1.5
    assert finite_number_setting("k", " -2 ", text=True) == -2.0
    for blank in ("", " ", "far"):
        with pytest.raises(ValueError, match=r"^k must be a number, got "):
            number_setting("k", blank, text=True)
    with pytest.raises(ValueError, match=r"^k must be finite, got 'nan'$"):
        finite_number_setting("k", "nan", text=True)


@pytest.mark.parametrize("value", [None, [1.0], {}])
def test_what_float_refuses_is_no_number(value: Any) -> None:
    with pytest.raises(ValueError, match=r"^k must be a number, got "):
        number_setting("k", value)


@pytest.mark.parametrize("port", [1, 2000, 65535])
def test_a_port_is_taken(port: int) -> None:
    assert is_port(port)
    check_port("k", port)


@pytest.mark.parametrize("port", [0, -1, 65536, True, False, 8100.0, "8100", None])
def test_what_is_no_port_is_refused(port: Any) -> None:
    assert not is_port(port)
    with pytest.raises(ValueError, match=r"^k must be an integer within 1-65535"):
        check_port("k", port)


def test_a_huge_port_is_refused_naming_the_key() -> None:
    with pytest.raises(
        ValueError,
        match=r"^a\.port must be an integer within 1-65535 "
        r"\(got an integer of about 10\*\*5000\)$",
    ):
        check_port("a.port", HUGER)


# ---------------------------------------------------------------------------
# The callers
# ---------------------------------------------------------------------------


def _run_ego(value: Any) -> float:
    from autoware_carla_scenario.examples import run

    return run._config_finite("ego.spawn_t", value)


def _run_speed(value: Any) -> float:
    from autoware_carla_scenario.examples import run

    return run._config_finite("ego.initial_speed_kmh", value, minimum=0.0)


def _declarative(value: Any) -> float:
    from .test_declarative_scenario import DeclarativeScenarioConfig, _scenario

    config = DeclarativeScenarioConfig(
        name="cut_in", spawn_overrides={"npc1": {"s": value}}
    )
    npc = _scenario(config=config).document.entity("npc1")
    assert npc is not None
    return float(npc.spawn.s.value)


def _authored(value: Any) -> float:
    from autoware_carla_scenario.trajectory.authoring import authored_place

    return float(authored_place(lanelet_id=1, s=value).s)


def _lens(value: Any) -> float:
    from autoware_carla_scenario.driver.base import DriverCameraLens

    return DriverCameraLens(
        principal_point=[1.0, 2.0],
        pixeldist_to_angle_poly=[0.0, 1.0],
        angle_to_pixeldist_poly=[0.0, 1.0],
        max_angle=value,
    ).max_angle


def _camera(value: Any) -> float:
    from autoware_carla_scenario.driver.base import DriverCameraConfig

    return DriverCameraConfig(position_x=value).position_x


def _example(value: Any) -> float:
    from autoware_carla_scenario.examples.configs import PedestrianDartOutConfig

    return PedestrianDartOutConfig(pedestrian_ahead_m=value).pedestrian_ahead_m


def _example_number(value: Any) -> float:
    from autoware_carla_scenario.examples.configs import AdversityConfig

    return AdversityConfig(nade_probability=value).nade_probability


#: Each caller, the key its errors name, and whether it refuses a NaN.
CALLERS: list[tuple[Callable[[Any], float], str, bool]] = [
    (_run_ego, r"ego\.spawn_t", True),
    (_run_speed, r"ego\.initial_speed_kmh", True),
    (_declarative, r"scenario\.spawn_overrides\.npc1\.s", True),
    (_authored, r"s", False),
    (_lens, r"lens\.max_angle", True),
    (_camera, r"camera '[a-z_0-9]+': position_x", True),
    (_example, r"scenario\.pedestrian_ahead_m", True),
    (_example_number, r"scenario\.nade_probability", True),
]
_IDS = [caller.__name__.lstrip("_") for caller, _, _ in CALLERS]


@pytest.mark.parametrize(("read", "key", "finite"), CALLERS, ids=_IDS)
def test_every_caller_refuses_true(
    read: Callable[[Any], float], key: str, finite: bool
) -> None:
    """``true`` used to be 1.0 to examples.run and trajectory.authoring."""
    with pytest.raises(ValueError, match=rf"^{key} must be a number, got True$"):
        read(True)


@pytest.mark.parametrize(("read", "key", "finite"), CALLERS, ids=_IDS)
def test_every_caller_names_the_key_of_a_huge_int(
    read: Callable[[Any], float], key: str, finite: bool
) -> None:
    """driver.base and examples.configs used to raise ``OverflowError``."""
    with pytest.raises(ValueError, match=rf"^{key} must be finite, got "):
        read(HUGE)
    with pytest.raises(ValueError, match=rf"^{key} must be finite, got an integer of"):
        read(HUGER)


@pytest.mark.parametrize(("read", "key", "finite"), CALLERS, ids=_IDS)
@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_a_caller_that_needs_a_finite_number_refuses_nan_and_inf(
    read: Callable[[Any], float], key: str, finite: bool, value: float
) -> None:
    if not finite:
        result = read(value)
        assert math.isnan(result) if math.isnan(value) else result == value
        return
    with pytest.raises(ValueError, match=rf"^{key} must be "):
        read(value)


@pytest.mark.parametrize(("read", "key", "finite"), CALLERS, ids=_IDS)
def test_every_caller_reads_a_number_as_before(
    read: Callable[[Any], float], key: str, finite: bool
) -> None:
    assert read(0.5) == 0.5
    assert read(1) == 1.0


@pytest.mark.parametrize(
    ("read", "takes_text"),
    [
        (_run_ego, True),
        (_declarative, True),
        (_authored, True),
        (_lens, False),
        (_camera, False),
        (_example, False),
    ],
    ids=["run", "declarative", "authored", "lens", "camera", "example"],
)
def test_numeric_text_is_read_where_it_was(
    read: Callable[[Any], float], takes_text: bool
) -> None:
    """A quoted number was read by the run's ``ego.*`` settings, a spawn
    override and an authored card, and refused by the driver's and the
    example scenarios' dataclasses; each keeps doing so."""
    if takes_text:
        assert read("0.5") == 0.5
    else:
        with pytest.raises(ValueError, match=r"must be a number, got '0\.5'$"):
            read("0.5")


def test_the_run_port_check_names_the_key_of_a_huge_int() -> None:
    from omegaconf import OmegaConf

    from autoware_carla_scenario.examples import run

    for section in ("server", "traffic_manager"):
        cfg = OmegaConf.create({section: {"port": 8100}})
        cfg[section].port = HUGER
        with pytest.raises(
            ValueError,
            match=rf"^{section}\.port must be an integer within 1-65535 "
            r"\(got an integer of about 10\*\*5000\)$",
        ):
            run._check_port_setting(cfg, section)
    cfg = OmegaConf.create({"server": {"port": None}})
    with pytest.raises(ValueError, match=r"^server\.port .* \(got None\)$"):
        run._check_port_setting(cfg, "server", null_is_default=False)
    run._check_port_setting(
        OmegaConf.create({"traffic_manager": {"port": None}}), "traffic_manager"
    )


def test_the_hils_port_check_names_the_key_of_a_huge_int() -> None:
    from autoware_carla_scenario.entity.hils_entity import HilsEgoConfig

    with pytest.raises(
        ValueError, match=r"^hils\.port must be an integer within 1-65535 \(got an"
    ):
        HilsEgoConfig(port=HUGER)
    for port in (True, 0, 70000, "2000"):
        with pytest.raises(ValueError, match=r"^hils\.port must be an integer"):
            HilsEgoConfig(port=port)  # type: ignore[arg-type]
    assert HilsEgoConfig(port=2000).port == 2000


def test_the_traffic_config_re_exports_the_port_check() -> None:
    from autoware_carla_scenario.traffic import config

    assert config.check_port is check_port


# ---------------------------------------------------------------------------
# The angle wrap
# ---------------------------------------------------------------------------

_EDGES = [
    -180.00000000000003,
    -180.0,
    -179.99999999999997,
    179.99999999999997,
    180.0,
    180.00000000000003,
    -540.0000000000001,
    539.9999999999999,
    -0.0,
    0.0,
    1e-300,
    -1e-300,
    360.0,
    -360.0,
    1e17,
    -1e17,
    math.inf,
    -math.inf,
    math.nan,
]


def _same(a: float, b: float) -> bool:
    return (math.isnan(a) and math.isnan(b)) or a == b


def test_the_sumo_wrap_keeps_the_edge_inside_its_range() -> None:
    from autoware_carla_scenario.kinematics.angle import normalize_angle_deg
    from autoware_carla_scenario.traffic.sumo import backend, geometry

    assert geometry._wrap(-180.00000000000003) == -180.0
    assert backend._wrap is geometry._wrap is normalize_angle_deg
    for angle in _EDGES:
        wrapped = geometry._wrap(angle)
        assert math.isnan(wrapped) or -180.0 <= wrapped < 180.0, angle


def test_the_trajectory_wrap_is_normalize_angle_deg() -> None:
    """``trajectory.model._wrap_degrees`` copies ``normalize_angle_deg`` on
    purpose (the trajectory model imports nothing of CARLA); the two must
    answer alike."""
    from autoware_carla_scenario.kinematics.angle import normalize_angle_deg
    from autoware_carla_scenario.trajectory.model import _wrap_degrees

    rng = random.Random(2398)
    angles = list(_EDGES)
    angles += [rng.uniform(-1e4, 1e4) for _ in range(20000)]
    angles += [k * 180.0 + d for k in range(-8, 9) for d in (-1e-13, 0.0, 1e-13)]
    angles += [math.nextafter(k * 180.0, s) for k in range(-8, 9) for s in (-1e9, 1e9)]
    for angle in angles:
        assert _same(_wrap_degrees(angle), normalize_angle_deg(angle)), angle
