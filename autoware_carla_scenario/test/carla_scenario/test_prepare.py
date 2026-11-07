"""Backends a run needs started before it (autoware_carla_scenario.prepare).

No docker runs here: the backends are fakes, and docker's command line is
recorded rather than run.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, Mapping, Optional

import pytest

from autoware_carla_scenario import SmartPedestrianControllerAction
from autoware_carla_scenario.pedestrian.ardy import DEFAULT_IMAGE, ArdyBackend
from autoware_carla_scenario.prepare import (
    Backend,
    BackendPreparer,
    InsufficientResourcesError,
    Resources,
)
from autoware_carla_scenario.prepare import docker as docker_module
from autoware_carla_scenario.prepare import host as host_module
from autoware_carla_scenario.prepare import pool as pool_module
from autoware_carla_scenario.prepare.declare import (
    action_backends,
    need_prepare,
    required_backends,
    scenario_backends,
)
from autoware_carla_scenario.prepare.host import Gpu, HostResources, assign_gpus
from autoware_carla_scenario.prepare.pool import BackendPool, running_backend


class _Fake:
    """A backend that records what was done to it."""

    backend_name: ClassVar[str] = "fake"
    built: ClassVar[list["_Fake"]] = []

    def __init__(self, settings: Mapping[str, Any]) -> None:
        self.settings = dict(settings)
        self.started: Optional[tuple[Optional[Path], Optional[int]]] = None
        self.closed = 0
        _Fake.built.append(self)

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any]) -> "_Fake":
        return cls(settings)

    @property
    def name(self) -> str:
        return self.backend_name

    def resources(self) -> Resources:
        return Resources(
            gpu_memory_mb=int(self.settings.get("gpu", 0)),
            memory_mb=int(self.settings.get("ram", 0)),
        )

    def start(self, log_dir: Optional[Path] = None, gpu: Optional[int] = None) -> None:
        if self.settings.get("fail"):
            raise RuntimeError("would not start")
        self.started = (log_dir, gpu)

    def address(self) -> str:
        return "127.0.0.1:1"

    def close(self) -> None:
        self.closed += 1


class _Other(_Fake):
    backend_name: ClassVar[str] = "other"


@pytest.fixture(autouse=True)
def _clean() -> Any:
    _Fake.built.clear()
    yield
    pool_module._ACTIVE = None


def test_the_contract_is_what_the_pool_and_the_backends_are() -> None:
    assert isinstance(BackendPool(), BackendPreparer)
    assert isinstance(_Fake({}), Backend)
    assert isinstance(ArdyBackend.from_settings({}), Backend)


class TestNeedPrepare:
    def test_a_class_declares_the_backends_it_needs(self) -> None:
        @need_prepare(_Fake)
        class Action:
            pass

        assert required_backends(Action) == (_Fake,)
        assert required_backends(Action()) == (_Fake,)

    def test_a_subclass_needs_its_bases_backends_and_its_own(self) -> None:
        @need_prepare(_Fake)
        class Base:
            pass

        @need_prepare(_Other, _Fake)
        class Derived(Base):
            pass

        assert required_backends(Derived) == (_Fake, _Other)
        assert required_backends(Base) == (_Fake,)

    def test_an_undecorated_class_needs_nothing(self) -> None:
        assert required_backends(object()) == ()

    def test_only_a_backend_class_is_taken(self) -> None:
        with pytest.raises(TypeError, match="backend_name"):
            need_prepare(object)

    def test_the_smart_pedestrian_needs_ardy(self) -> None:
        assert required_backends(SmartPedestrianControllerAction) == (ArdyBackend,)

    def test_actions_are_scanned_each_backend_once(self) -> None:
        @need_prepare(_Fake)
        class A:
            pass

        @need_prepare(_Other, _Fake)
        class B:
            pass

        assert action_backends([A(), B(), A()]) == (_Fake, _Other)

    def test_a_declarative_scenarios_actions_are_read_before_setup(self) -> None:
        scenario = SimpleNamespace(
            compiled=SimpleNamespace(
                actions=[
                    SimpleNamespace(
                        spec=SimpleNamespace(target="..actions:WalkStraightAction")
                    ),
                    SimpleNamespace(
                        spec=SimpleNamespace(
                            target="..actions:SmartPedestrianControllerAction"
                        )
                    ),
                ]
            )
        )
        assert scenario_backends(scenario) == (ArdyBackend,)


class TestBackendPool:
    def test_each_backend_starts_once_with_its_settings(self, tmp_path: Path) -> None:
        @need_prepare(_Fake)
        class Scenario:
            pass

        pool = BackendPool({"fake": {"x": 1}}, check_resources=False)
        pool.prepare_scenarios([Scenario(), Scenario()], log_dir=tmp_path)
        pool.prepare_actions([Scenario()], log_dir=tmp_path)
        assert len(_Fake.built) == 1
        assert _Fake.built[0].settings == {"x": 1}
        assert _Fake.built[0].started == (tmp_path, None)
        assert pool.backend("fake") is _Fake.built[0]

    def test_close_stops_every_backend(self) -> None:
        @need_prepare(_Fake, _Other)
        class Scenario:
            pass

        pool = BackendPool(check_resources=False)
        pool.prepare_scenarios([Scenario()])
        pool.close()
        pool.close()
        assert [b.closed for b in _Fake.built] == [1, 1]
        assert pool.backend("fake") is None

    def test_a_host_without_room_refuses_before_anything_starts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        @need_prepare(_Fake)
        class Scenario:
            pass

        monkeypatch.setattr(
            pool_module,
            "probe_host",
            lambda: HostResources(gpus=(Gpu(0, "GPU-a", 8000),), memory_mb=64000),
        )
        pool = BackendPool({"fake": {"gpu": 24000, "ram": 1000}})
        with pytest.raises(InsufficientResourcesError, match="24000 MiB of GPU memory"):
            pool.prepare_scenarios([Scenario()])
        assert _Fake.built[0].started is None

    def test_a_backend_is_started_on_the_gpu_chosen_for_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        @need_prepare(_Fake)
        class Scenario:
            pass

        monkeypatch.setattr(
            pool_module,
            "probe_host",
            lambda: HostResources(
                gpus=(Gpu(0, "GPU-a", 8000), Gpu(1, "GPU-b", 40000)), memory_mb=64000
            ),
        )
        BackendPool({"fake": {"gpu": 24000}}).prepare_scenarios([Scenario()])
        assert _Fake.built[0].started == (None, 1)

    def test_a_backend_that_fails_to_start_is_closed(self) -> None:
        @need_prepare(_Fake)
        class Scenario:
            pass

        pool = BackendPool({"fake": {"fail": True}}, check_resources=False)
        with pytest.raises(RuntimeError, match="would not start"):
            pool.prepare_scenarios([Scenario()])
        assert _Fake.built[0].closed == 1
        assert pool.backend("fake") is None

    def test_an_action_finds_the_backend_its_queue_started(self) -> None:
        @need_prepare(_Fake)
        class Scenario:
            pass

        pool = BackendPool(check_resources=False)
        pool.prepare_scenarios([Scenario()])
        assert running_backend(_Fake) is _Fake.built[0]
        pool.close()
        assert pool_module._ACTIVE is None

    def test_an_action_starts_a_backend_no_queue_prepared(self) -> None:
        pool = BackendPool(check_resources=False)
        pool_module._ACTIVE = pool
        backend = running_backend(_Fake)
        assert backend is _Fake.built[0]
        assert pool.backend("fake") is backend


class TestAssignGpus:
    def test_host_memory_is_added_up(self) -> None:
        host = HostResources(gpus=(), memory_mb=10000)
        with pytest.raises(InsufficientResourcesError, match="12000 MiB needed"):
            assign_gpus(
                [("a", Resources(memory_mb=6000)), ("b", Resources(memory_mb=6000))],
                host,
            )

    def test_a_gpu_is_needed_when_none_is_visible(self) -> None:
        with pytest.raises(InsufficientResourcesError, match="none is visible"):
            assign_gpus([("a", Resources(gpu_memory_mb=1))], HostResources((), 1000))

    def test_two_backends_share_a_gpu_with_room_for_both(self) -> None:
        host = HostResources(gpus=(Gpu(0, "GPU-a", 50000),), memory_mb=1000)
        placed = assign_gpus(
            [
                ("a", Resources(gpu_memory_mb=24000)),
                ("b", Resources(gpu_memory_mb=24000)),
            ],
            host,
        )
        assert placed == {"a": host.gpus[0], "b": host.gpus[0]}

    def test_two_backends_that_do_not_fit_one_gpu_are_refused(self) -> None:
        host = HostResources(gpus=(Gpu(0, "GPU-a", 40000),), memory_mb=1000)
        with pytest.raises(InsufficientResourcesError, match="b: 24000"):
            assign_gpus(
                [
                    ("a", Resources(gpu_memory_mb=24000)),
                    ("b", Resources(gpu_memory_mb=24000)),
                ],
                host,
            )

    def test_a_backend_set_to_a_gpu_is_checked_on_it(self) -> None:
        host = HostResources(
            gpus=(Gpu(0, "GPU-a", 50000), Gpu(1, "GPU-b", 1000)), memory_mb=1000
        )
        with pytest.raises(InsufficientResourcesError, match="set to .1."):
            assign_gpus([("a", Resources(gpu_memory_mb=24000, gpu=1))], host)
        with pytest.raises(InsufficientResourcesError, match="GPU 2, which is not"):
            assign_gpus([("a", Resources(gpu_memory_mb=1, gpu=2))], host)
        placed = assign_gpus([("a", Resources(gpu_memory_mb=24000, gpu=0))], host)
        assert placed == {"a": host.gpus[0]}

    def test_a_backend_set_to_a_gpu_is_placed_before_the_others(self) -> None:
        # The larger unpinned backend would take GPU 0's room first (#1121).
        host = HostResources(
            gpus=(Gpu(0, "GPU-a", 10000), Gpu(1, "GPU-b", 9000)), memory_mb=1000
        )
        placed = assign_gpus(
            [
                ("policy", Resources(gpu_memory_mb=8000)),
                ("perception", Resources(gpu_memory_mb=5000, gpu=0)),
            ],
            host,
        )
        assert placed == {"policy": host.gpus[1], "perception": host.gpus[0]}

    def test_a_backend_set_to_a_gpu_too_small_is_still_refused(self) -> None:
        host = HostResources(
            gpus=(Gpu(0, "GPU-a", 4000), Gpu(1, "GPU-b", 9000)), memory_mb=1000
        )
        with pytest.raises(
            InsufficientResourcesError,
            match="perception: 5000 MiB of GPU memory needed, 4000 MiB free "
            r"on the GPU it is set to \(0\)",
        ):
            assign_gpus(
                [
                    ("policy", Resources(gpu_memory_mb=8000)),
                    ("perception", Resources(gpu_memory_mb=5000, gpu=0)),
                ],
                host,
            )

    def test_the_backend_that_fits_nowhere_is_the_one_named(self) -> None:
        # The pinned backend fits its GPU; the unpinned one fits no GPU.
        host = HostResources(
            gpus=(Gpu(0, "GPU-a", 10000), Gpu(1, "GPU-b", 3000)), memory_mb=1000
        )
        with pytest.raises(InsufficientResourcesError) as raised:
            assign_gpus(
                [
                    ("policy", Resources(gpu_memory_mb=8000)),
                    ("perception", Resources(gpu_memory_mb=5000, gpu=0)),
                ],
                host,
            )
        message = str(raised.value)
        assert (
            "policy: 8000 MiB of GPU memory needed, 5000 MiB free on the "
            "roomiest GPU (0)" in message
        )
        assert "perception: 5000 MiB of GPU memory needed" not in message

    def test_a_placement_the_greedy_one_misses_is_found(self) -> None:
        # Pinned first, the roomiest GPU each time leaves "small" out:
        # perception and policy on GPU 0, "large" on GPU 1, no room for
        # "small". Placing policy beside perception instead fits them all.
        host = HostResources(
            gpus=(Gpu(0, "GPU-a", 11000), Gpu(1, "GPU-b", 11000)), memory_mb=1000
        )
        placed = assign_gpus(
            [
                ("policy", Resources(gpu_memory_mb=9000)),
                ("large", Resources(gpu_memory_mb=6000)),
                ("small", Resources(gpu_memory_mb=5000)),
                ("perception", Resources(gpu_memory_mb=1000, gpu=0)),
            ],
            host,
        )
        assert placed == {
            "perception": host.gpus[0],
            "policy": host.gpus[0],
            "large": host.gpus[1],
            "small": host.gpus[1],
        }

    def test_the_greedy_placement_is_kept_when_it_fits(self) -> None:
        host = HostResources(
            gpus=(Gpu(0, "GPU-a", 8000), Gpu(1, "GPU-b", 8000)), memory_mb=1000
        )
        placed = assign_gpus(
            [
                ("a", Resources(gpu_memory_mb=3000)),
                ("b", Resources(gpu_memory_mb=2000)),
                ("c", Resources(gpu_memory_mb=1000)),
            ],
            host,
        )
        assert placed == {"a": host.gpus[0], "b": host.gpus[1], "c": host.gpus[1]}

    def test_only_the_visible_gpus_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        gpus = (Gpu(0, "GPU-aaaa", 1), Gpu(1, "GPU-bbbb", 2))
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
        assert host_module._visible(gpus) == (gpus[1],)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-aa")
        assert host_module._visible(gpus) == (gpus[0],)
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
        assert host_module._visible(gpus) == gpus


class _Docker:
    """``subprocess.run`` of docker, recorded."""

    def __init__(self, listing: str = "", image_present: bool = True) -> None:
        self.calls: list[list[str]] = []
        self.listing = listing
        self.image_present = image_present

    def __call__(self, argv: list[str], **_: Any) -> Any:
        self.calls.append(list(argv))
        out, code = "", 0
        if argv[1] == "info":
            out = '{"nvidia":{},"runc":{}}'
        elif argv[1] == "ps":
            out = self.listing
        elif argv[1:3] == ["image", "inspect"]:
            code = 0 if self.image_present else 1
        elif argv[1] == "run":
            out = "c0ffee\n"
        elif argv[1] == "inspect":
            out = "true\n"
        return subprocess.CompletedProcess(argv, code, out, "")


class _Answering(ArdyBackend):
    def ready(self, timeout_s: float) -> bool:
        return True


class TestDockerBackend:
    def test_the_container_runs_on_its_gpu_with_its_port_cache_and_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorded = _Docker()
        monkeypatch.setattr(docker_module.subprocess, "run", recorded)
        monkeypatch.setenv("HF_TOKEN", "secret")
        backend = _Answering.from_settings({"hf_home": "~/hf", "model": "core"})
        backend.start(None, gpu=1)
        (run,) = [c for c in recorded.calls if c[1] == "run"]
        assert run[run.index("--gpus") + 1] == "device=1"
        published = run[run.index("--publish") + 1]
        assert published == f"127.0.0.1:{backend.address().split(':')[1]}:50061"
        assert f"{os.path.expanduser('~/hf')}:/hf" in run
        assert "ARDY_MODEL=core" in run
        # Passed on by name: the token never shows in the command line.
        assert "HF_TOKEN" in run and "secret" not in " ".join(run)
        assert run[-1] == DEFAULT_IMAGE
        assert "--rm" not in run
        backend.close()
        assert ["docker", "stop", "--time", "10", "c0ffee"] in recorded.calls
        assert ["docker", "rm", "--force", "c0ffee"] in recorded.calls

    @pytest.mark.parametrize(
        "gpus, expected",
        [
            ("none", Resources(memory_mb=16000)),
            ("device=1", Resources(gpu_memory_mb=24000, memory_mb=16000, gpu=1)),
            ("auto", Resources(gpu_memory_mb=24000, memory_mb=16000)),
        ],
    )
    def test_what_a_container_takes_follows_its_gpus(
        self, gpus: str, expected: Resources
    ) -> None:
        assert ArdyBackend.from_settings({"gpus": gpus}).resources() == expected

    def test_a_missing_image_is_pulled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorded = _Docker(image_present=False)
        monkeypatch.setattr(docker_module.subprocess, "run", recorded)
        _Answering.from_settings({}).start()
        assert ["docker", "pull", DEFAULT_IMAGE] in recorded.calls

    def test_only_a_stopped_runs_leftover_is_removed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        host = docker_module.socket.gethostname()
        recorded = _Docker(
            listing=(
                f"dead {host}:999999999\n"
                f"alive {host}:{os.getpid()}\n"
                f"elsewhere other-host:999999999\n"
            )
        )
        monkeypatch.setattr(docker_module.subprocess, "run", recorded)
        _Answering.from_settings({}).start()
        removed = [c[-1] for c in recorded.calls if c[1] == "rm"]
        assert removed == ["dead"]


class TestDockerBackendLeavesNoContainer:
    """A container docker created is removed whatever went wrong (#1960)."""

    @staticmethod
    def _failing(recorded: _Docker, command: str, failure: Any) -> Any:
        def run(argv: list[str], **kwargs: Any) -> Any:
            if argv[1] == command:
                recorded.calls.append(list(argv))
                if isinstance(failure, BaseException):
                    raise failure
                return subprocess.CompletedProcess(argv, 125, "", failure)
            return recorded(argv, **kwargs)

        return run

    @pytest.mark.parametrize(
        "failure",
        [
            "docker: Error response from daemon: Bind for 127.0.0.1:5 failed: "
            "port is already allocated.",
            subprocess.TimeoutExpired(["docker", "run"], 120.0),
            KeyboardInterrupt(),
        ],
        ids=["port-allocated", "timeout", "ctrl-c"],
    )
    def test_a_run_that_fails_removes_the_container_by_name(
        self, monkeypatch: pytest.MonkeyPatch, failure: Any
    ) -> None:
        recorded = _Docker()
        monkeypatch.setattr(
            docker_module.subprocess,
            "run",
            self._failing(recorded, "run", failure),
        )
        backend = _Answering.from_settings({})
        expected: type[BaseException] = (
            KeyboardInterrupt
            if isinstance(failure, KeyboardInterrupt)
            else docker_module.BackendError
        )
        with pytest.raises(expected):
            backend.start()
        (run,) = [c for c in recorded.calls if c[1] == "run"]
        name = run[run.index("--name") + 1]
        assert ["docker", "rm", "--force", name] in recorded.calls
        assert backend._container is None
        backend.close()

    def test_a_failed_removal_does_not_hide_why_the_run_failed(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        recorded = _Docker()
        failing = self._failing(recorded, "run", "port is already allocated")

        def run(argv: list[str], **kwargs: Any) -> Any:
            if argv[1] == "rm":
                raise subprocess.TimeoutExpired(argv, 60.0)
            return failing(argv, **kwargs)

        monkeypatch.setattr(docker_module.subprocess, "run", run)
        with pytest.raises(docker_module.BackendError, match="already allocated"):
            _Answering.from_settings({}).start()
        assert "Could not remove backend container acs-ardy-" in caplog.text

    def test_a_stop_that_times_out_still_removes_the_container(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorded = _Docker()
        monkeypatch.setattr(
            docker_module.subprocess,
            "run",
            self._failing(
                recorded, "stop", subprocess.TimeoutExpired(["docker", "stop"], 60)
            ),
        )
        backend = _Answering.from_settings({})
        backend.start()
        backend.close()
        assert ["docker", "stop", "--time", "10", "c0ffee"] in recorded.calls
        assert ["docker", "rm", "--force", "c0ffee"] in recorded.calls

    @pytest.mark.parametrize("pull", ["never", "missing", "always"])
    def test_docker_run_never_pulls_itself(
        self, monkeypatch: pytest.MonkeyPatch, pull: str
    ) -> None:
        recorded = _Docker(image_present=False)
        monkeypatch.setattr(docker_module.subprocess, "run", recorded)
        _Answering.from_settings({"pull": pull}).start()
        (run,) = [c for c in recorded.calls if c[1] == "run"]
        assert run[run.index("--pull") + 1] == "never"
        assert run.index("--pull") < run.index(DEFAULT_IMAGE)
        pulled = ["docker", "pull", DEFAULT_IMAGE] in recorded.calls
        assert pulled == (pull != "never")


class TestDockerSettingsAreChecked:
    """A backend's settings are refused, naming their key, before any image
    is pulled or container started (#1486)."""

    @pytest.mark.parametrize(
        ("key", "value", "message"),
        [
            ("pull", "sometimes", "must be one of missing, always, never"),
            ("pull", True, "must be one of"),
            ("docker_args", "--shm-size=8g", "must be a list"),
            ("docker_args", {"a": 1}, "must be a list"),
            ("docker_args", [["--shm-size", "8g"]], "must be a list of strings"),
            ("startup_timeout_s", -1, "must be finite and above 0"),
            ("startup_timeout_s", 0, "must be finite and above 0"),
            ("startup_timeout_s", float("nan"), "must be finite and above 0"),
            ("startup_timeout_s", float("inf"), "must be finite and above 0"),
            ("startup_timeout_s", "600", "must be a number of seconds"),
            ("startup_timeout_s", True, "must be a number of seconds"),
            pytest.param(  # too big for a float (#1652)
                "startup_timeout_s",
                10**400,
                "must be finite and above 0",
                id="startup_timeout_s-huge",
            ),
            ("timeout_s", float("nan"), "must be finite and above 0"),
            ("timeout_s", -5.0, "must be finite and above 0"),
            ("timeout_s", None, "must be a number of seconds"),
            ("gpu_memory_mb", -1, "must not be negative"),
            ("gpu_memory_mb", 1.5, "must be a whole number"),
            ("memory_mb", "16000", "must be a whole number"),
            ("memory_mb", float("inf"), "must be finite"),
        ],
    )
    def test_a_bad_ardy_setting_is_refused_naming_its_key(
        self, key: str, value: object, message: str
    ) -> None:
        with pytest.raises(ValueError, match=rf"^backends\.ardy\.{key} {message}"):
            ArdyBackend.from_settings({key: value})

    def test_the_shipped_and_documented_settings_are_taken(self) -> None:
        backend = ArdyBackend.from_settings(
            {
                "image": DEFAULT_IMAGE,
                "gpu_memory_mb": 24000.0,
                "memory_mb": 16000,
                "startup_timeout_s": 1800,
                "timeout_s": 120,
                "pull": "never",
                "docker_args": ["--shm-size=8g", 1],
            }
        )
        settings = backend._settings
        assert settings.pull == "never"
        assert settings.docker_args == ("--shm-size=8g", "1")
        assert settings.startup_timeout_s == 1800.0
        assert backend.timeout_s == 120.0
        assert backend.resources() == Resources(gpu_memory_mb=24000, memory_mb=16000)
        assert ArdyBackend.from_settings({"docker_args": None})._settings == (
            ArdyBackend.from_settings({})._settings
        )

    def test_the_shared_reader_names_any_backend(self) -> None:
        with pytest.raises(ValueError, match=r"^backends\.soma\.pull must"):
            docker_module.read_docker_settings(
                "soma", {"pull": "always!"}, image="img:1", container_port=1
            )
        settings = docker_module.read_docker_settings(
            "soma", {"pull": "always"}, image="img:1", container_port=1
        )
        assert settings.pull == "always" and settings.image == "img:1"

    def test_nothing_is_pulled_or_started_for_a_bad_setting(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorded = _Docker(image_present=False)
        monkeypatch.setattr(docker_module.subprocess, "run", recorded)

        @need_prepare(_Answering)
        class Scenario:
            pass

        pool = BackendPool({"ardy": {"startup_timeout_s": -1}}, check_resources=False)
        with pytest.raises(ValueError, match=r"backends\.ardy\.startup_timeout_s"):
            pool.prepare_scenarios([Scenario()])
        assert recorded.calls == []
        assert pool.backend("ardy") is None


class TestBackendSettingsInTheBatchCheck:
    """``check_batch_configs`` refuses a bad ``backends.ardy`` setting before
    anything starts (#1486)."""

    @pytest.mark.parametrize(
        ("override", "key"),
        [
            ("backends.ardy.pull=sometimes", "pull"),
            ("backends.ardy.startup_timeout_s=-1", "startup_timeout_s"),
            ("backends.ardy.timeout_s=nan", "timeout_s"),
            ("backends.ardy.docker_args='--shm-size=8g'", "docker_args"),
            ("backends.ardy.memory_mb=1.5", "memory_mb"),
        ],
    )
    def test_a_bad_setting_is_refused(self, override: str, key: str) -> None:
        from autoware_carla_scenario.examples import run

        cfg = run._compose_config("adversity/nade", ["map=town10hd_opt", override])
        with pytest.raises(run.RunConfigError, match=rf"backends\.ardy\.{key} must"):
            run.check_batch_configs([cfg])

    def test_a_backend_block_that_is_no_mapping_is_refused(self) -> None:
        from autoware_carla_scenario.examples import run

        cfg = run._compose_config(
            "adversity/nade", ["map=town10hd_opt", "backends.ardy=5"]
        )
        with pytest.raises(run.RunConfigError, match=r"backends\.ardy must be"):
            run.check_batch_configs([cfg])

    def test_the_shipped_config_passes(self) -> None:
        from autoware_carla_scenario.examples import run

        cfg = run._compose_config(
            "adversity/nade",
            ["map=town10hd_opt", "backends.ardy.docker_args=['--shm-size=8g']"],
        )
        run.check_batch_configs([cfg])


#: A docker whose ``logs`` (and every other command) print bytes that are not
#: UTF-8, as a backend's own output may (#1346).
_UNDECODABLE_DOCKER = """\
import sys
sys.stdout.buffer.write(b"out \\xff\\n")
sys.stderr.buffer.write(b"err \\xfe\\n")
"""


class _Silent(docker_module.DockerBackend):
    backend_name = "silent"

    def ready(self, timeout_s: float) -> bool:
        return False


def _undecodable_docker(tmp_path: Path) -> str:
    script = tmp_path / "docker"
    script.write_text(f"#!{sys.executable}\n{_UNDECODABLE_DOCKER}")
    script.chmod(0o755)
    return str(script)


def _silent_backend(docker: str, startup_timeout_s: float) -> _Silent:
    backend = _Silent(
        docker_module.DockerSettings(
            image="img:1",
            container_port=1,
            startup_timeout_s=startup_timeout_s,
            docker=docker,
        )
    )
    backend._container = "c0ffee"
    return backend


class TestDockerBackendOutput:
    """The backend's output is any bytes: it is read, not refused (#1346)."""

    def test_a_log_that_is_not_utf8_is_read_with_replacements(
        self, tmp_path: Path
    ) -> None:
        backend = _silent_backend(_undecodable_docker(tmp_path), 600.0)
        tail = backend._tail()
        assert "out \ufffd" in tail
        assert "err \ufffd" in tail

    def test_a_backend_that_exits_reports_a_log_that_is_not_utf8(
        self, tmp_path: Path
    ) -> None:
        # ``inspect`` does not print "true": the container is not running.
        backend = _silent_backend(_undecodable_docker(tmp_path), 600.0)
        with pytest.raises(docker_module.BackendError, match="exited while starting"):
            backend._wait_ready()

    def test_a_timeout_reports_a_log_that_is_not_utf8_and_closes(
        self, tmp_path: Path
    ) -> None:
        backend = _silent_backend(_undecodable_docker(tmp_path), 0.0)
        with pytest.raises(docker_module.BackendError, match="did not answer") as err:
            backend._wait_ready()
        assert "out \ufffd" in str(err.value)
        assert backend._container is None

    def test_a_timeout_closes_even_when_the_log_cannot_be_read(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorded = _Docker()
        monkeypatch.setattr(docker_module.subprocess, "run", recorded)
        backend = _silent_backend("docker", 0.0)

        def unreadable() -> str:
            raise OSError("no log")

        monkeypatch.setattr(backend, "_tail", unreadable)
        with pytest.raises(OSError, match="no log"):
            backend._wait_ready()
        assert ["docker", "stop", "--time", "10", "c0ffee"] in recorded.calls
        assert ["docker", "rm", "--force", "c0ffee"] in recorded.calls
        assert backend._container is None

    def test_docker_output_that_is_not_utf8_is_read_with_replacements(
        self, tmp_path: Path
    ) -> None:
        out = docker_module._docker([_undecodable_docker(tmp_path), "ps"], "list")
        assert out == "out \ufffd\n"


#: A docker that records each command line in ``RECORD`` and stops answering
#: (sleeps, for a bounded time) for the commands that start with one of
#: ``HANGS``, as a daemon on an overloaded host does (#1785).
_HANGING_DOCKER = """\
import sys, time
args = sys.argv[1:]
line = " ".join(args)
with open(RECORD, "a") as record:
    record.write(line + "\\n")
if any(line.startswith(hang) for hang in HANGS):
    time.sleep(30)
if args[0] == "info":
    print('{"runc":{}}')
elif args[0] == "ps":
    print(LISTING)
elif args[0] == "run":
    print("c0ffee")
elif args[0] == "inspect":
    print("true")
"""


def _hanging_docker(
    tmp_path: Path, hangs: tuple[str, ...], listing: str = ""
) -> tuple[str, Path]:
    """The fake docker's path, and the file it records its commands in."""
    record = tmp_path / "calls.txt"
    script = tmp_path / "docker"
    script.write_text(
        f"#!{sys.executable}\n"
        f"RECORD = {str(record)!r}\nHANGS = {hangs!r}\nLISTING = {listing!r}\n"
        f"{_HANGING_DOCKER}"
    )
    script.chmod(0o755)
    return str(script), record


def _calls(record: Path) -> list[str]:
    return record.read_text().splitlines() if record.exists() else []


class _Ready(_Silent):
    def ready(self, timeout_s: float) -> bool:
        return True


def _settings(docker: str, startup_timeout_s: float) -> docker_module.DockerSettings:
    return docker_module.DockerSettings(
        image="img:1",
        container_port=1,
        gpus="none",
        startup_timeout_s=startup_timeout_s,
        docker=docker,
    )


class TestDockerThatStopsAnswering:
    """``startup_timeout_s`` bounds startup even when docker hangs (#1785).

    The fake docker sleeps 30 s at most, so a test that failed to bound a
    call still ends; each call it makes is a child of ``subprocess.run``,
    which kills it on its timeout.
    """

    def test_a_hanging_inspect_ends_in_the_deadline_and_removes_the_container(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(docker_module, "_TAIL_TIMEOUT_S", 1.0)
        docker, record = _hanging_docker(tmp_path, ("inspect", "logs --tail"))
        backend = _Silent(_settings(docker, 2.0))
        began = time.monotonic()
        with pytest.raises(docker_module.BackendError, match="did not answer") as err:
            backend.start()
        assert time.monotonic() - began < 2.0 + 15.0
        assert "no log" in str(err.value)
        calls = _calls(record)
        assert any(c.startswith("inspect") for c in calls)
        assert "rm --force c0ffee" in calls
        assert backend._container is None

    def test_a_hanging_image_inspect_fails_the_start_in_bounded_time(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(docker_module, "_CALL_TIMEOUT_S", 1.0)
        docker, record = _hanging_docker(tmp_path, ("image inspect",))
        backend = _Ready(_settings(docker, 600.0))
        began = time.monotonic()
        with pytest.raises(docker_module.BackendError, match="timed out"):
            backend.start()
        assert time.monotonic() - began < 15.0
        # Nothing was started, and nothing pulled.
        assert not any(c.split()[0] in ("run", "pull") for c in _calls(record))

    def test_a_leftover_whose_rm_hangs_is_given_up_on_in_bounded_time(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(docker_module, "_CALL_TIMEOUT_S", 1.0)
        host = docker_module.socket.gethostname()
        docker, record = _hanging_docker(
            tmp_path, ("rm --force dead",), listing=f"dead {host}:999999999"
        )
        backend = _Ready(_settings(docker, 600.0))
        began = time.monotonic()
        try:
            backend.start()
        finally:
            backend.close()
        assert time.monotonic() - began < 15.0
        assert "rm --force dead" in _calls(record)
        assert "remove container dead" in caplog.text

    def test_a_docker_that_cannot_be_run_for_a_leftover_is_logged(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        host = docker_module.socket.gethostname()
        recorded = _Docker(listing=f"dead {host}:999999999\n")

        def run(argv: list[str], **kwargs: Any) -> Any:
            assert kwargs.get("timeout") is not None, argv
            if argv[1] == "rm":
                raise PermissionError("docker is not executable")
            return recorded(argv, **kwargs)

        monkeypatch.setattr(docker_module.subprocess, "run", run)
        backend = _Answering.from_settings({})
        backend.start()
        assert "docker is not executable" in caplog.text
        assert backend._container == "c0ffee"


class TestDockerThatCannotBeRun:
    """An OSError from docker (a removed binary, a fork that fails with
    EMFILE or ENOMEM) is a :class:`BackendError`, and the container started
    is removed (#1790)."""

    @staticmethod
    def _failing_run(recorded: _Docker, command: str) -> Any:
        def run(argv: list[str], **kwargs: Any) -> Any:
            if argv[1] == command:
                raise OSError(24, "Too many open files")
            return recorded(argv, **kwargs)

        return run

    def test_an_inspect_that_cannot_run_fails_the_start_and_closes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorded = _Docker()
        monkeypatch.setattr(
            docker_module.subprocess, "run", self._failing_run(recorded, "inspect")
        )
        closed: list[str] = []
        close = _Answering.close

        def counted(backend: _Answering) -> None:
            closed.append(backend.name)
            close(backend)

        monkeypatch.setattr(_Answering, "close", counted)

        @need_prepare(_Answering)
        class Scenario:
            pass

        pool = BackendPool({}, check_resources=False)
        with pytest.raises(docker_module.BackendError, match="Too many open files"):
            pool.prepare_scenarios([Scenario()])
        assert closed == ["ardy"]
        assert ["docker", "rm", "--force", "c0ffee"] in recorded.calls
        assert pool.backend("ardy") is None

    def test_a_log_that_cannot_be_read_is_a_note(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorded = _Docker()
        monkeypatch.setattr(
            docker_module.subprocess, "run", self._failing_run(recorded, "logs")
        )
        backend = _silent_backend("docker", 0.0)
        assert backend._tail() == (
            "(no log: could not run docker: [Errno 24] Too many open files)"
        )
        with pytest.raises(docker_module.BackendError, match="did not answer") as err:
            backend._wait_ready()
        assert "(no log: could not run docker" in str(err.value)
        assert backend._container is None

    def test_a_log_follower_that_cannot_start_fails_the_start_and_closes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorded = _Docker()
        monkeypatch.setattr(docker_module.subprocess, "run", recorded)

        def popen(*args: Any, **kwargs: Any) -> Any:
            raise OSError(12, "Cannot allocate memory")

        monkeypatch.setattr(docker_module.subprocess, "Popen", popen)
        started: list[_Answering] = []
        start = _Answering.start

        def recorded_start(backend: _Answering, *args: Any) -> None:
            started.append(backend)
            start(backend, *args)

        monkeypatch.setattr(_Answering, "start", recorded_start)

        @need_prepare(_Answering)
        class Scenario:
            pass

        pool = BackendPool({}, check_resources=False)
        with pytest.raises(docker_module.BackendError, match="follow the log"):
            pool.prepare_scenarios([Scenario()], log_dir=tmp_path)
        assert ["docker", "rm", "--force", "c0ffee"] in recorded.calls
        (backend,) = started
        assert backend._container is None
        assert backend._log is None
        assert pool.backend("ardy") is None


class TestQueue:
    def test_a_scenario_added_after_the_start_is_prepared_then(
        self, tmp_path: Path
    ) -> None:
        from unittest.mock import MagicMock

        from autoware_carla_scenario.scenario_queue import ScenarioQueue

        prepared: list[list[object]] = []
        preparer = SimpleNamespace(
            prepare_scenarios=lambda scenarios, log_dir=None: prepared.append(
                list(scenarios)
            )
        )
        queue = ScenarioQueue(
            server=MagicMock(),
            output_dir=tmp_path,
            backends=preparer,  # type: ignore[arg-type]
        )
        before, after = MagicMock(), MagicMock()
        queue.add(before)
        assert prepared == []  # the start prepares it, with the others
        queue._runner = MagicMock()
        queue.add(after)
        assert prepared == [[after]]
