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
from typing import Any

import numpy as np
import torch
from carla_driver_interface.driver import BaseDriver, DriveContext, DriveResult, SessionState
from carla_driver_interface.geometry import Pose, Trajectory

from stl_driver.config import STLDriverConfig
from stl_driver.fallback import emergency_brake_plan, hold_plan
from stl_driver.frenet import build_route_table, project_to_frenet
from stl_driver.optimizer import PlanSolution, RobustnessOptimizer
from stl_driver.perception import forecast_actors, pad_actor_discs, read_traffic_light
from stl_driver.rollout import RolloutInputs, plan_xy
from stl_driver.rules import build_rules
from stl_driver.ruleset import RuleSet

__all__ = ["EgoPolicy", "warmed_up_policy"]


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
        rules: RuleSet | None = None,
    ) -> None:
        # Every argument has a default, so `load_policy` can build the policy
        # from its spec alone (`driver.policy=stl_driver.policy:EgoPolicy`).
        # `cruise_speed_mps` defaults to None rather than a speed because a
        # loader passes it only when asked to, and its absence has to mean
        # "whatever the config says" rather than "8 m/s".
        base = config or STLDriverConfig()
        if cruise_speed_mps is None:
            self.config = base
        else:
            self.config = replace(base, cruise_speed_mps=float(cruise_speed_mps))

        torch.set_num_threads(self.config.optimizer.torch_threads)
        # A caller can supply an entirely different specification without
        # editing anything here; see `stl_driver.rules` for the shape of one.
        self.rules = rules or RuleSet(build_rules(self.config))
        self.optimizer = RobustnessOptimizer(self.rules, self.config)
        self.device = self.optimizer.device

        self._warm_start: torch.Tensor | None = None
        # The acceleration the last plan was anchored at, so this one can be
        # anchored somewhere the vehicle could actually have got to. Under the
        # same lock as the warm start, and for the same reason: gRPC handler
        # threads and the search both touch it.
        self._anchor_accel_mps2: float | None = None
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
            self._anchor_accel_mps2 = None
        self.warm_up()

    def on_session_close(self, session: SessionState) -> None:
        with self._warm_start_lock:
            self._warm_start = None
            self._anchor_accel_mps2 = None

    def warm_up(self) -> None:
        """Pay the one-off costs before the runtime starts waiting on us.

        Torch's lazy initialisation, and compilation when it is enabled. The
        service calls `on_session_start` before any observation arrives, so
        this is time nobody is waiting on -- whereas the same work inside the
        first `drive` shows up as a step that missed its deadline.

        The first call compiles, which on a cold torch cache takes about a
        minute: longer than a runtime's deadline for ``start_session`` (60 s
        in autoware_carla_scenario and in carla_driver_interface's fake loop).
        Call it before serving, as :func:`warmed_up_policy` and the CLI do;
        later calls cost one solve.
        """
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
            self.optimizer.prepare(inputs)
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
                    centreline_xy=inputs.route_xy.cpu().numpy(),
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

        Bounding the magnitude is not enough, because what the ride feels is
        the *change*. Each plan is jerk-limited inside itself, so nothing
        within one is ever harsh; what the vehicle actually gets is the first
        step of a new plan every 100 ms, and consecutive plans agreed only as
        well as their anchors did. Measured over a drive, the reading moved by
        up to 5.9 m/s^2 between one step and the next -- 59 m/s^3, fifteen
        times the comfort bound -- and the commanded acceleration followed it
        with a correlation of 0.99, exceeding the bound on 42% of steps while
        every rule read as satisfied. `comfort` cannot see this: it judges one
        plan, and this is a disagreement between two.

        So the anchor moves no faster than the vehicle itself could, which is
        the jerk limit over one policy step -- about 0.39 m/s^2. Noise faster
        than that is not acceleration, and a real change is followed at the
        rate a real vehicle would take to make it.
        """
        if not np.isfinite(measured_mps2):
            return 0.0
        # Bounded by what a *plan* may command, not by what the emergency
        # stop may. The measured value anchors the acceleration profile, so
        # an extreme one ties the whole plan to it: after a 7 m/s^2 stop,
        # every way of pulling away again is a jerk violation and the
        # vehicle sits there. The emergency plan brakes that hard without
        # going through here.
        #
        # The envelope comes from `Limits` rather than from the comfort bounds,
        # so that this and the first knot agree exactly. They used to differ by
        # the inset, and that 2% arrived in the opening jerk ten times over --
        # see `Limits.plan_accel_envelope`.
        limits = self.config.limits
        lowest, highest = limits.plan_accel_envelope()
        plausible = float(np.clip(measured_mps2, lowest, highest))

        with self._warm_start_lock:
            previous = self._anchor_accel_mps2
            if previous is not None:
                step = limits.plan_jerk_limit() * self.config.horizon.step_s
                plausible = float(np.clip(plausible, previous - step, previous + step))
            self._anchor_accel_mps2 = plausible
        return plausible

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

        ego_offsets, ego_radius = config.ego_footprint()
        device = self.device

        def tensor(values, dtype=torch.float32):
            return torch.as_tensor(values, dtype=dtype, device=device)

        padded_centres, padded_radii = pad_actor_discs(
            forecast.centres, forecast.radii, config.limits.max_actor_discs
        )

        def scalar(value: float) -> torch.Tensor:
            """A 0-d tensor, so `torch.compile` traces it instead of baking it in."""
            return torch.as_tensor(float(value), dtype=torch.float32, device=device)

        return RolloutInputs(
            route_xy=tensor(table.xy),
            route_tangent=tensor(table.tangent),
            route_curvature=tensor(table.curvature),
            route_ds=table.ds,
            published_length_m=scalar(table.published_length_m),
            station0_m=scalar(station0),
            offset0_m=scalar(offset0),
            speed0_mps=scalar(speed_mps),
            accel0_mps2=scalar(accel_mps2),
            actor_centres=tensor(padded_centres),
            actor_radii=tensor(padded_radii),
            n_actors=forecast.n_actors,
            n_forecast_discs=int(forecast.radii.shape[0]),
            ego_disc_offsets=tensor(ego_offsets),
            ego_disc_radius=ego_radius,
            lead_station_m=tensor(forecast.lead_station_m),
            lead_speed_mps=tensor(forecast.lead_speed_mps),
            stop_station_m=scalar(light.stop_station_m),
            light_gate=scalar(
                config.limits.gate_magnitude_m if light.active else -config.limits.gate_magnitude_m
            ),
            speed_limit_mps=scalar(target_speed),
            progress_target_m=scalar(config.progress_target_m(target_speed, speed_mps)),
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
        points = positions.cpu().numpy()
        yaws = headings.cpu().numpy()

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
            "route_length_m": float(inputs.published_length_m),
            # The forecast's own count, not the padded tensor's width -- that
            # width is a constant, and reporting it would hide the one case
            # worth seeing, a scene with more discs than the rollout has room
            # for. What is dropped is the farthest, which no minimum over
            # discs could have selected, but a cap that never says so is the
            # kind of thing this specification exists not to have.
            "n_actor_discs": float(inputs.n_forecast_discs),
            "n_actor_discs_dropped": float(max(0, inputs.n_forecast_discs - inputs.n_actor_discs)),
            "n_actors": float(inputs.n_actors),
            "light_active": 1.0 if float(inputs.light_gate) > 0.0 else 0.0,
        }
        for name, value in solution.per_rule.items():
            scalars[f"stl_rule_{name}"] = value

        with torch.no_grad():
            from stl_driver.signals import Col

            scalars["target_speed_mps"] = float(solution.signal[:, Col.SPEED].mean())
            scalars["min_clearance_m"] = float(solution.signal[:, Col.CLEARANCE].min())
            scalars["min_headway_slack_m"] = float(solution.signal[:, Col.HEADWAY_SLACK].min())
        return scalars


def warmed_up_policy(**kwargs: Any) -> EgoPolicy:
    """An :class:`EgoPolicy` built from ``kwargs``, compiled before it is returned.

    The spec to load it by in a runtime's own process
    (``driver.policy=stl_driver.policy:warmed_up_policy`` in
    autoware_carla_scenario): the compile happens while the policy loads,
    rather than inside the first session's ``start_session`` call and its
    deadline.
    """
    policy = EgoPolicy(**kwargs)
    policy.warm_up()
    return policy
