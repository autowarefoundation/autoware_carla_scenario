"""The ``scenario`` CLI caps glibc's malloc arenas, unless the user said otherwise."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from autoware_carla_scenario.examples import run


@pytest.fixture
def libc() -> MagicMock:
    """A libc whose mallopt() succeeds, standing in for glibc's."""
    fake = MagicMock(name="libc")
    fake.mallopt.return_value = 1
    return fake


def _glibc_linux(libc: MagicMock, environ: dict[str, str]):
    """Patch the process into glibc Linux with *environ*, and *libc* loaded."""
    return (
        patch.object(run.sys, "platform", "linux"),
        patch.dict(run.os.environ, environ, clear=True),
        patch("platform.libc_ver", return_value=("glibc", "2.39")),
        patch("ctypes.util.find_library", return_value="libc.so.6"),
        patch("ctypes.CDLL", return_value=libc),
    )


class TestCapMallocArenas:
    def test_it_caps_the_arenas_at_two_on_glibc_linux(self, libc: MagicMock) -> None:
        a, b, c, d, e = _glibc_linux(libc, {})
        with a, b, c, d, e:
            assert run._cap_malloc_arenas() is True

        # M_ARENA_MAX is -8 in glibc's malloc.h.
        libc.mallopt.assert_called_once_with(-8, 2)

    @pytest.mark.parametrize(
        "environ",
        [
            {"MALLOC_ARENA_MAX": "4"},
            {"GLIBC_TUNABLES": "glibc.malloc.arena_max=4"},
            {"GLIBC_TUNABLES": "glibc.malloc.tcache_count=0:glibc.malloc.arena_max=1"},
        ],
    )
    def test_the_users_setting_stands(
        self, libc: MagicMock, environ: dict[str, str]
    ) -> None:
        a, b, c, d, e = _glibc_linux(libc, environ)
        with a, b, c, d, e:
            assert run._cap_malloc_arenas() is False

        libc.mallopt.assert_not_called()

    def test_other_tunables_do_not_stop_it(self, libc: MagicMock) -> None:
        a, b, c, d, e = _glibc_linux(
            libc, {"GLIBC_TUNABLES": "glibc.malloc.tcache_count=0"}
        )
        with a, b, c, d, e:
            assert run._cap_malloc_arenas() is True

        libc.mallopt.assert_called_once_with(-8, 2)

    def test_not_on_another_platform(self, libc: MagicMock) -> None:
        a, b, c, d, e = _glibc_linux(libc, {})
        with a, b, c, d, e, patch.object(run.sys, "platform", "darwin"):
            assert run._cap_malloc_arenas() is False

        libc.mallopt.assert_not_called()

    def test_not_on_another_libc(self, libc: MagicMock) -> None:
        a, b, c, d, e = _glibc_linux(libc, {})
        with a, b, c, d, e, patch("platform.libc_ver", return_value=("", "")):
            assert run._cap_malloc_arenas() is False

        libc.mallopt.assert_not_called()

    def test_not_without_a_libc_to_load(self, libc: MagicMock) -> None:
        a, b, c, d, e = _glibc_linux(libc, {})
        with a, b, c, d, e, patch("ctypes.util.find_library", return_value=None):
            assert run._cap_malloc_arenas() is False

        libc.mallopt.assert_not_called()

    def test_a_failing_libc_is_ignored(self, libc: MagicMock) -> None:
        a, b, c, d, e = _glibc_linux(libc, {})
        with a, b, c, d, e, patch("ctypes.CDLL", side_effect=OSError("no libc")):
            assert run._cap_malloc_arenas() is False

    def test_a_refused_mallopt_is_reported(self, libc: MagicMock) -> None:
        libc.mallopt.return_value = 0
        a, b, c, d, e = _glibc_linux(libc, {})
        with a, b, c, d, e:
            assert run._cap_malloc_arenas() is False


def test_the_cli_caps_the_arenas_before_anything_else() -> None:
    """main() caps them first: arenas are made for threads, so before any starts."""
    order: list[str] = []

    def load_plugins() -> None:
        order.append("plugins")
        raise SystemExit(0)

    with (
        patch.object(
            run, "_cap_malloc_arenas", side_effect=lambda: order.append("cap")
        ),
        patch.object(run, "load_scenario_plugins", side_effect=load_plugins),
        pytest.raises(SystemExit),
    ):
        run.main()

    assert order == ["cap", "plugins"]
