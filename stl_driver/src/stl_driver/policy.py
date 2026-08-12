# SPDX-License-Identifier: Apache-2.0
"""The policy: an STL specification, wired into the driver interface.

Everything interesting happens elsewhere -- the specification is in
:mod:`stl_driver.rules`, the search in :mod:`stl_driver.optimizer`.  This
module's job is to turn one ``DriveContext`` into the inputs those need, and
to turn the result back into a trajectory the runtime can drive.

The per-rule robustness values it reports in ``debug_scalars`` are, in
practice, the most useful thing here.  When the car does something surprising,
the answer is in ``stl_rule_collision_free`` going negative three steps
earlier -- the behaviour is explained in the same language it was specified
in, which is the point of writing the specification down in the first place.
"""

from __future__ import annotations

import threading
from dataclasses import replace

import numpy as np
import torch
from carla_driver_interface.driver import BaseDriver, DriveContext, DriveResult, SessionState
from carla_driver_interface.geometry import Pose, Trajectory

from stl_driver.config import STLDriverConfig
from stl_driver.fallback import emergency_brake_plan, hold_plan
from stl_driver.frenet import build_route_table, project_to_frenet
from stl_driver.optimizer import PlanSolution, RobustnessOptimizer
from stl_driver.perception import forecast_actors, read_traffic_light
from stl_driver.rollout import RolloutInputs, plan_xy
from stl_driver.rules import RuleSet, build_rules

__all__ = ["EgoPolicy"]


class EgoPolicy(BaseDriver):
    """Drive by maximising the robustness of an STL specification.

    Every step: read the world, search plan space for the trajectory that
    satisfies the specification with the most room to spare, and drive it.  If
    no plan satisfies the rules marked ``hard``, brake instead.
    """

    name = "stl_ego"
    #: No camera is used -- the specification is written over kinematics and
    #: the runtime's ground truth, not over pixels. Keeping this at 1 stops
    #: the service retaining frames nothing will read.
    frame_history_length = 1

    def __init__(
        self,
        cruise_speed_mps: float | None = None,
        config: STLDriverConfig | None = None,
    ) -> None:
        # `cruise_speed_mps` is the first keyword and named to match, so the
        # upstream CLI -- which calls `policy_cls(cruise_speed_mps=...)` -- can
        # serve this policy through POLICY_REGISTRY without knowing anything
        # about it. It defaults to None rather than a speed because upstream
        # omits the argument entirely when the flag is unset, and that has to
        # mean "whatever the config says" rather than "8 m/s".
        base = config or STLDriverConfig()
        if cruise_speed_mps is None:
            self.config = base
        else:
            self.config = replace(base, cruise_speed_mps=float(cruise_speed_mps))

        torch.set_num_threads(self.config.optimizer.torch_threads)
        self.rules = RuleSet(build_rules(self.config))
        self.optimizer = RobustnessOptimizer(self.rules, self.config)

        self._warm_start: torch.Tensor | None = None
        self._warm_start_lock = threading.Lock()

    # -- session hooks -----------------------------------------------------

    def on_session_start(self, session: SessionState) -> None:
        """Reset per-rollout state and pay torch's start-up cost up front.

        The service calls this before any observation arrives, so the warm-up
        happens in time the runtime is not waiting on -- whereas paying it
        inside the first ``drive`` would show up as a a step that misses its
        deadline.
        """
        with self._warm_start_lock:
            self._warm_start = None
        self._warm_up()

    def on_session_close(self, session: SessionState) -> None:
        with self._warm_start_lock:
            self._warm_start = None

    def _warm_up(self) -> None:
        """Solve one synthetic problem so the real first step is not the first."""
        straight = np.stack(
            [
                np.arange(0.0, 260.0, 2.0),
                np.zeros(130),
                np.zeros(130),
            ],
            axis=1,
        )
        try:
            inputs = self._build_inputs(
                route_in_rig=straight,
                anchor=Pose.identity(),
                renderer_data=None,
                speed_mps=5.0,
                accel_mps2=0.0,
            )
            self.optimizer.solve(inputs, warm_start=None)
        except Exception:  # noqa: BLE001
            # A failed warm-up costs a slower first step and nothing else; it
            # must never stop a session from starting.
            pass

    # -- the step ----------------------------------------------------------

    def drive(self, ctx: DriveContext) -> DriveResult:
        session = ctx.session
        # Hold the lock only for the snapshot: gRPC handler threads write to
        # the session concurrently, but the search must not block them.
        with session.lock:
            route = np.asarray(session.route_waypoints_in_rig, dtype=np.float64)
            anchor = session.latest_pose()
            speed_mps = session.speed_mps()
            state = session.latest_dynamic_state()
            accel_mps2 = float(state.linear_acceleration.x) if state is not None else 0.0
        accel_mps2 = self._plausible_acceleration(accel_mps2)

        if len(route) < 2 or anchor is None:
            # Not an error: the route has not arrived yet.
            return DriveResult(
                trajectory_in_rig=hold_plan(ctx.time_now_us, speed_mps, self.config),
                debug_scalars={
                    "stl_no_route": 1.0,
                    "current_speed_mps": speed_mps,
                },
            )

        try:
            inputs = self._build_inputs(
                route_in_rig=route,
                anchor=anchor,
                renderer_data=ctx.renderer_data,
                speed_mps=speed_mps,
                accel_mps2=accel_mps2,
            )
        except ValueError:
            # A degenerate route -- all waypoints coincident, say. Coasting is
            # the right answer, and it is not worth failing the rollout over.
            return DriveResult(
                trajectory_in_rig=hold_plan(ctx.time_now_us, speed_mps, self.config),
                debug_scalars={"stl_bad_route": 1.0, "current_speed_mps": speed_mps},
            )

        with self._warm_start_lock:
            warm_start = self._warm_start
        solution = self.optimizer.solve(inputs, warm_start=warm_start)

        scalars = self._debug_scalars(solution, inputs, speed_mps)

        if solution.needs_fallback:
            # Deliberately not the optimizer's best plan: it violates a rule
            # we said we would never violate, and driving it anyway would make
            # `hard` meaningless.
            with self._warm_start_lock:
                self._warm_start = None
            return DriveResult(
                trajectory_in_rig=emergency_brake_plan(
                    ctx.time_now_us,
                    speed_mps,
                    self.config,
                    centreline_xy=inputs.route_xy.numpy(),
                ),
                debug_scalars=scalars,
            )

        with self._warm_start_lock:
            self._warm_start = solution.params.detach().clone()

        return DriveResult(
            trajectory_in_rig=self._to_trajectory(solution.params, inputs, ctx.time_now_us),
            debug_scalars=scalars,
            sampled_trajectories_in_rig=[
                self._to_trajectory(params, inputs, ctx.time_now_us)
                for params in solution.alternatives
            ],
        )

    # -- plumbing ----------------------------------------------------------

    def _plausible_acceleration(self, measured_mps2: float) -> float:
        """Confine the measured acceleration to the envelope the plan respects.

        The plan's acceleration profile is anchored at this value so that it
        starts from the vehicle's actual state rather than stepping to a new
        one.  That makes it load-bearing: the opening jerk is the difference
        between the first knot and this number, so an implausible reading
        becomes an implausible jerk and puts `comfort` out of reach of every
        candidate.

        CARLA derives acceleration by finite-differencing velocity and it is
        correspondingly noisy -- transients well outside anything the vehicle
        could physically produce are routine. Clamping to the same envelope
        the plan is held to keeps a sensor artefact from dictating the plan.
        """
        limits = self.config.limits
        if not np.isfinite(measured_mps2):
            return 0.0
        return float(np.clip(measured_mps2, -limits.emergency_decel_mps2, limits.max_accel_mps2))

    def _build_inputs(
        self,
        route_in_rig: np.ndarray,
        anchor: Pose,
        renderer_data,
        speed_mps: float,
        accel_mps2: float,
    ) -> RolloutInputs:
        """Everything the rollout needs that does not depend on the plan."""
        config = self.config
        table = build_route_table(
            route_in_rig,
            ds=config.route_table_ds_m,
            length_m=config.route_table_length_m(),
        )

        # The runtime publishes the route from roughly the ego's position, but
        # projecting rather than assuming keeps this correct when it does not.
        station, offset = project_to_frenet(table, np.zeros((1, 2)))
        station0, offset0 = float(station[0]), float(offset[0])

        light = read_traffic_light(renderer_data, station0, config)
        forecast = forecast_actors(renderer_data, anchor, table, station0, config)

        speed_limit = 0.0
        if renderer_data is not None:
            speed_limit = float(renderer_data.speed_limit_mps)
        target_speed = config.target_speed_mps(speed_limit)

        return RolloutInputs(
            route_xy=torch.from_numpy(table.xy).to(torch.float32),
            route_tangent=torch.from_numpy(table.tangent).to(torch.float32),
            route_ds=table.ds,
            published_length_m=table.published_length_m,
            station0_m=station0,
            offset0_m=offset0,
            speed0_mps=float(speed_mps),
            accel0_mps2=float(accel_mps2),
            actor_centres=torch.from_numpy(forecast.centres),
            actor_radii=torch.from_numpy(forecast.radii),
            lead_station_m=torch.from_numpy(forecast.lead_station_m),
            stop_station_m=light.stop_station_m,
            light_gate=(
                config.limits.gate_magnitude_m if light.active else -config.limits.gate_magnitude_m
            ),
            speed_limit_mps=target_speed,
            progress_target_m=config.progress_target_m(target_speed, speed_mps),
        )

    def _to_trajectory(
        self,
        params: torch.Tensor,
        inputs: RolloutInputs,
        time_now_us: int,
    ) -> Trajectory:
        """Lay a parameter vector out as a rig-frame trajectory."""
        with torch.no_grad():
            positions, headings = plan_xy(params, inputs, self.config)
        points = positions.numpy()
        yaws = headings.numpy()

        step_us = int(round(self.config.horizon.step_s * 1e6))
        plan = Trajectory.empty()
        for index in range(len(points)):
            plan.append(
                time_now_us + (index + 1) * step_us,
                Pose.from_xyz_yaw(
                    float(points[index, 0]),
                    float(points[index, 1]),
                    0.0,
                    float(yaws[index]),
                ),
            )
        return plan

    def _debug_scalars(
        self,
        solution: PlanSolution,
        inputs: RolloutInputs,
        speed_mps: float,
    ) -> dict[str, float]:
        """The step, explained in the specification's own terms."""
        scalars = {
            "stl_robustness": solution.robustness,
            "stl_fallback": 1.0 if solution.needs_fallback else 0.0,
            "stl_solve_seconds": solution.solve_seconds,
            "stl_refine_steps": float(solution.n_refine_steps),
            "stl_vmap": 1.0 if solution.used_vmap else 0.0,
            "current_speed_mps": speed_mps,
            "route_length_m": inputs.published_length_m,
            "n_actor_discs": float(inputs.n_actor_discs),
            "light_active": 1.0 if inputs.light_gate > 0.0 else 0.0,
        }
        for name, value in solution.per_rule.items():
            scalars[f"stl_rule_{name}"] = value

        with torch.no_grad():
            from stl_driver.signals import Col

            scalars["target_speed_mps"] = float(solution.signal[:, Col.SPEED].mean())
            scalars["min_clearance_m"] = float(solution.signal[:, Col.CLEARANCE].min())
            scalars["min_headway_slack_m"] = float(solution.signal[:, Col.HEADWAY_SLACK].min())
        return scalars


def register_with_upstream_cli() -> None:
    """Make this policy selectable as ``--policy stl_ego`` upstream.

    Only effective if called before the upstream CLI builds its parser, since
    the ``--policy`` choices are read out of the registry dict at that point.
    ``stl-driver serve`` is the supported entry point; this is a convenience
    for anyone already driving the upstream CLI.
    """
    from carla_driver_interface.driver.policies import POLICY_REGISTRY

    POLICY_REGISTRY[EgoPolicy.name] = EgoPolicy
