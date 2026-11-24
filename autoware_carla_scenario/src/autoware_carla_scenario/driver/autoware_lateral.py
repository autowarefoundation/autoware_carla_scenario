"""Autoware's lateral control chain behind an end-to-end planner, for the driver policy's plan.

:class:`~.control.TrajectoryFollower` steers on pure pursuit.  The X2 steers its plans
through a longer chain, which tier4/OnePlanner's closed-loop replay emulates
(``projects/resworld/closed_loop_control_chain``: ``replay_mpc.py``, ``mpc_emu.py``,
``tso_opt.py``); this module is that chain's software half, so a policy driven in
CARLA sees the controller it sees in the replay and on the car::

    the policy's plan, stamped when it was asked for
      -> TrajectoryTimeSequenceRawOptimizer (TSO): the plan re-solved as an OCP from the
         ego's state at the stamp (:class:`TimeSequenceRawOptimizer`)
      -> published ``publish_delay_s`` after its stamp
      -> autoware_mpc_lateral_controller in temporal mode, every ``ctrl_period``
         (:class:`MpcLateralController`), with MpcLateralController's stop handling,
         reading the localization ``localization_lag_s`` behind the vehicle
      -> the tire angle CARLA is asked for

The steering actuator and the vehicle's yaw response are CARLA's own vehicle physics
here, not the replay's identified X2 plant.  The longitudinal controller is
:class:`~.control.TrajectoryFollower`'s port of ``autoware_pid_longitudinal_controller``,
which follows the published trajectory (the TSO's output, with its speeds and
accelerations), as Autoware's does.

The numerical code mirrors OnePlanner's line for line -- the same discretisation, cost
terms, QP constraints, least-squares OCP and tolerances -- so the two can be compared
tick for tick; where it differs (the CARLA tick of 0.05 s, the route goal) the
difference is named where it is made.  ``driver.lateral`` (:class:`LateralChainConfig`)
turns it on; off, the follower steers as before.

All geometry is in the run's local frame (right-handed, x east, y north, yaw
counter-clockwise), and steering angles are tire angles, positive to the left.
"""

from __future__ import annotations

import logging
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from scipy.special import expit

from .control import ControlConfig, plan_velocities
from .geometry import Pose, Trajectory

__all__ = [
    "LATERAL_CONTROLLERS",
    "TRAJECTORY_OPTIMIZERS",
    "AutowareLateralChain",
    "LateralChainConfig",
    "MpcLateralController",
    "MpcParams",
    "MpcReference",
    "PublishedPlan",
    "TimeSequenceRawOptimizer",
    "TsoParams",
    "load_mpc_params",
    "load_tso_params",
]

logger = logging.getLogger(__name__)

#: The lateral controllers :attr:`LateralChainConfig.controller` names.
LATERAL_CONTROLLERS: Tuple[str, ...] = ("pure_pursuit", "mpc")
#: The optimizers :attr:`LateralChainConfig.trajectory_optimizer` names.
TRAJECTORY_OPTIMIZERS: Tuple[str, ...] = ("none", "tso")

#: The parameter files bundled with the package: the X2's, as OnePlanner's chain reads them.
_PARAMS_DIR: Path = Path(__file__).resolve().parent / "params"
DEFAULT_MPC_PARAM_FILE: Path = _PARAMS_DIR / "mpc.param.yaml"
DEFAULT_TSO_PARAM_FILE: Path = _PARAMS_DIR / "tso.param.yaml"

_US_PER_S: float = 1e6


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LateralChainConfig:
    """``driver.lateral``: which lateral chain steers the driver policy's plan.

    The defaults past :attr:`controller` are the X2's, as OnePlanner's closed-loop
    replay runs them (its README's command line): ``OPT=tso TSO_V0=ego OPT_T0=1
    TSO_SEMANTICS=upstream TSO_BOUNDS=1 MPC_COMPANION=1 PLANT_PUB_S=0.078
    PLANT_EKF_LAG_S=0.03``, the MPC on ``params/mpc.param.yaml`` with a 2.75 m
    wheelbase and the TSO on ``params/tso.param.yaml``.

    Raises:
        ValueError: If a field is out of its domain.
    """

    controller: str = "pure_pursuit"
    """``"pure_pursuit"``: :class:`~.control.TrajectoryFollower`'s own steering, and
    nothing else of this config is read.  ``"mpc"``: this module's chain."""

    trajectory_optimizer: str = "tso"
    """``"tso"``: the plan goes through the TrajectoryTimeSequenceRawOptimizer before it
    is published (``OPT=tso``).  ``"none"``: the plan is published as the e2e node
    publishes it, an ego point at its stamp first."""

    mpc_param_file: str = ""
    """The MPC's ROS parameter file; empty: the bundled X2 file."""

    tso_param_file: str = ""
    """The TSO's ROS parameter file; empty: the bundled X2 file (``TSO_PARAM``)."""

    wheelbase_m: float = 2.75
    """The wheelbase the MPC and the TSO model the vehicle with (the JPNTAXI
    vehicle_info the node checks), not CARLA's vehicle's."""

    publish_delay_s: float = 0.078
    """From the plan's stamp to the MPC taking it (``PLANT_PUB_S``; 78 ms on the bags)."""

    localization_lag_s: float = 0.03
    """How far the pose the MPC reads trails the vehicle (``PLANT_EKF_LAG_S``; the EKF's
    30 ms).  The policy is told the vehicle's own pose, as before."""

    ctrl_period_s: float = 0.03
    """The MPC's period.  It ticks on its own clock; CARLA steps every 0.05 s, so a
    step runs the ticks that fell due within it, on the pose at the step."""

    mpc_companion: bool = True
    """The MPC stop handling the TSO's parameter file says must go with enabling it
    (``MPC_COMPANION=1``: no steer hold on stop, no confidence slew limit)."""

    tso_v0_from_ego: bool = True
    """The OCP's initial speed is the ego's measured speed (``TSO_V0=ego``); off, the
    mean chord speed of the plan's first stages."""

    tso_upstream_semantics: bool = True
    """A short plan's last stage solved as upstream #3641 solves it (``TSO_SEMANTICS``)."""

    tso_bounds: bool = True
    """The OCP's lateral-acceleration soft bound and state bounds (``TSO_BOUNDS=1``)."""

    tso_t0_point: bool = True
    """The TSO's output gets the ego's state at the stamp as a ``t = 0`` point
    (``OPT_T0=1``)."""

    def __post_init__(self) -> None:
        from ..value_text import shown  # noqa: PLC0415

        if self.controller not in LATERAL_CONTROLLERS:
            raise ValueError(
                f"controller must be one of {list(LATERAL_CONTROLLERS)}, "
                f"got {shown(self.controller)}"
            )
        if self.trajectory_optimizer not in TRAJECTORY_OPTIMIZERS:
            raise ValueError(
                f"trajectory_optimizer must be one of {list(TRAJECTORY_OPTIMIZERS)}, "
                f"got {shown(self.trajectory_optimizer)}"
            )
        # wheelbase_m and ctrl_period_s divide; the delays only shift a clock.
        for name, positive in (
            ("wheelbase_m", True),
            ("ctrl_period_s", True),
            ("publish_delay_s", False),
            ("localization_lag_s", False),
        ):
            value = getattr(self, name)
            if not (value > 0.0 if positive else value >= 0.0):
                domain = "positive" if positive else "zero or more"
                raise ValueError(f"{name} must be {domain}, got {value}")

    @property
    def enabled(self) -> bool:
        """Whether this chain steers (``controller: mpc``)."""
        return self.controller == "mpc"

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "LateralChainConfig":
        """The config of a ``driver.lateral`` mapping, as
        :meth:`~.control.ControlConfig.from_mapping` reads ``driver.control``: a
        null value is the key's default, and each value is checked as its
        field's default says (a number, true or false, a path).

        Raises:
            ValueError: naming the key, for a key this config does not define or a
                value outside its domain.
        """
        from dataclasses import fields  # noqa: PLC0415

        from ..utils.config import null_as_default  # noqa: PLC0415
        from ..value_text import shown  # noqa: PLC0415
        from .control import _config_number  # noqa: PLC0415

        known = {f.name for f in fields(cls)}
        unknown = sorted(set(mapping) - known)
        if unknown:
            raise ValueError(
                f"{unknown[0]}: Unknown LateralChainConfig key(s) {unknown}. "
                f"Known keys: {sorted(known)}"
            )
        values = null_as_default(cls, mapping)
        for name, value in values.items():
            field_default = getattr(cls, name)
            if isinstance(field_default, bool):
                if not isinstance(value, bool):
                    raise ValueError(
                        f"{name} must be true or false, got {shown(value)}"
                    )
            elif isinstance(field_default, float):
                values[name] = _config_number(name, value)
            elif not isinstance(value, str):
                raise ValueError(f"{name} must be a string, got {shown(value)}")
        return cls(**values)


def _ros_parameters(path: Path) -> Dict[str, Any]:
    """The ``ros__parameters`` of a ROS 2 parameter file's single node."""
    import yaml  # noqa: PLC0415

    with open(path, encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    (node,) = document.values()
    return dict(node["ros__parameters"])


# ---------------------------------------------------------------------------
# MPC: autoware_mpc_lateral_controller in temporal mode (OnePlanner's mpc_emu.py)
# ---------------------------------------------------------------------------


@dataclass
class MpcParams:
    """autoware_mpc_lateral_controller's parameters the emulation reads."""

    wheelbase: float = 2.75
    N: int = 50
    dt: float = 0.1
    ctrl_period: float = 0.03
    input_delay: float = 0.25
    steer_tau: float = 0.25
    w_lat: float = 0.2
    w_head: float = 0.0
    w_head_v2: float = 0.03
    w_steer: float = 1.0
    w_steer_v2: float = 0.25
    w_lat_jerk: float = 0.1
    w_steer_rate: float = 0.0
    w_steer_acc: float = 1e-6
    w_term_lat: float = 1.0
    w_term_head: float = 0.1
    zero_ff_deg: float = 0.5
    curv_num: int = 15
    resample: float = 0.1
    lpf_hz: float = 10.0
    steer_lim: float = 0.64
    #: getWeight's low-curvature weights (|k| < low_k): lat, head, head_v2, steer,
    #: steer_v2, lat_jerk.
    low_k: float = 0.0
    low: Tuple[float, ...] = (0.1, 0.0, 0.3, 1.0, 0.25, 0.0)
    #: Steering rate limits (deg/s) by curvature (signed, as the code reads it) and
    #: by velocity.
    rate_k: Tuple[Tuple[float, float], ...] = (
        (0.001, 40.0),
        (0.002, 50.0),
        (0.01, 60.0),
    )
    rate_v: Tuple[Tuple[float, float], ...] = ((10.0, 60.0), (15.0, 50.0), (20.0, 40.0))
    constrained: bool = True
    #: generateMPCMatrix: with a temporal trajectory that carries
    #: front_wheel_angle_rad, Uref is that steering (resampled in time) instead of
    #: atan(L * smoothed geometric curvature).
    ff_from_traj_steer: bool = True


@dataclass(frozen=True)
class MpcStopParams:
    """MpcLateralController's stop handling (updateStopStateHoldTimer,
    isStoppedState, applyConfidenceSteerSlewLimit)."""

    entry_ego: float = 0.2
    entry_target: float = 0.5
    hold: float = 2.0
    slew: bool = True
    l_min: float = 0.5
    l_ref: float = 1.0
    rate_min: float = 0.02


def load_mpc_params(
    path: Path, wheelbase: float = 2.75, ctrl_period: float = 0.03
) -> Tuple[MpcParams, MpcStopParams]:
    """The MPC's parameters and its stop handling's, from a parameter file.

    Raises:
        ValueError: If the file asks for ``keep_steer_control_until_converged``,
            which is not mirrored (it needs isTrajectoryShapeChanged).
    """
    q = _ros_parameters(path)

    def g(key: str, default: Any) -> Any:
        return q.get(key, default)

    params = MpcParams(
        wheelbase=wheelbase,
        N=g("mpc_prediction_horizon", 50),
        dt=g("mpc_prediction_dt", 0.1),
        ctrl_period=ctrl_period,
        input_delay=g("input_delay", 0.25),
        steer_tau=g("vehicle_model_steer_tau", 0.25),
        w_lat=g("mpc_weight_lat_error", 0.2),
        w_head=g("mpc_weight_heading_error", 0.0),
        w_head_v2=g("mpc_weight_heading_error_squared_vel", 0.03),
        w_steer=g("mpc_weight_steering_input", 1.0),
        w_steer_v2=g("mpc_weight_steering_input_squared_vel", 0.25),
        w_lat_jerk=g("mpc_weight_lat_jerk", 0.1),
        w_steer_rate=g("mpc_weight_steer_rate", 0.0),
        w_steer_acc=g("mpc_weight_steer_acc", 1e-6),
        w_term_lat=g("mpc_weight_terminal_lat_error", 1.0),
        w_term_head=g("mpc_weight_terminal_heading_error", 0.1),
        zero_ff_deg=g("mpc_zero_ff_steer_deg", 0.5),
        curv_num=g("curvature_smoothing_num_traj", 15),
        resample=g("traj_resample_dist", 0.1),
        lpf_hz=g("steering_lpf_cutoff_hz", 10.0),
        low_k=g("mpc_low_curvature_thresh_curvature", 0.0),
        ff_from_traj_steer=g("use_trajectory_steering_for_feedforward", True),
        low=tuple(
            g("mpc_low_curvature_" + key, default)
            for key, default in (
                ("weight_lat_error", 0.1),
                ("weight_heading_error", 0.0),
                ("weight_heading_error_squared_vel", 0.3),
                ("weight_steering_input", 1.0),
                ("weight_steering_input_squared_vel", 0.25),
                ("weight_lat_jerk", 0.0),
            )
        ),
        rate_k=tuple(
            zip(
                g("curvature_list_for_steer_rate_lim", [0.001, 0.002, 0.01]),
                g("steer_rate_lim_dps_list_by_curvature", [40.0, 50.0, 60.0]),
            )
        ),
        rate_v=tuple(
            zip(
                g("velocity_list_for_steer_rate_lim", [10.0, 15.0, 20.0]),
                g("steer_rate_lim_dps_list_by_velocity", [60.0, 50.0, 40.0]),
            )
        ),
    )
    if g("keep_steer_control_until_converged", False):
        raise ValueError(
            f"{path}: keep_steer_control_until_converged is not mirrored "
            "(it needs isTrajectoryShapeChanged); set it false"
        )
    stop = MpcStopParams(
        entry_ego=g("stop_state_entry_ego_speed", 0.2),
        entry_target=g("stop_state_entry_target_speed", 0.5),
        hold=g("stop_state_steer_hold_duration", 2.0),
        slew=g("enable_confidence_steer_slew_limit", True),
        l_min=g("reference_confidence_L_ahead_min", 0.5),
        l_ref=g("reference_confidence_L_ahead_ref", 1.0),
        rate_min=g("steer_slew_rate_min_rad_s", 0.02),
    )
    return params, stop


def _zoh_interp(table: Tuple[Tuple[float, float], ...], x: float) -> float:
    ref = np.array([a for a, _ in table])
    lim = np.array([b for _, b in table])
    return float(
        np.interp(x, ref, lim)
    )  # np.interp holds the end values, as the code does


def _weights(p: MpcParams, k: float) -> Tuple[float, ...]:
    if abs(k) < p.low_k:
        return p.low
    return (p.w_lat, p.w_head, p.w_head_v2, p.w_steer, p.w_steer_v2, p.w_lat_jerk)


def _curvature(x: np.ndarray, y: np.ndarray, num: int) -> np.ndarray:
    n = len(x)
    k = np.zeros(n)
    m = min(num, (n - 1) // 2)
    if m < 1:
        return k
    for i in range(m, n - m):
        p1 = np.array([x[i - m], y[i - m]])
        p2 = np.array([x[i], y[i]])
        p3 = np.array([x[i + m], y[i + m]])
        den = max(
            np.linalg.norm(p1 - p2) * np.linalg.norm(p2 - p3) * np.linalg.norm(p3 - p1),
            1e-4,
        )
        cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
        k[i] = 2.0 * cross / den
    k[:m] = k[m]
    k[n - m :] = k[n - m - 1]
    return k


def _short_segment(ds: float, dt: float, vx: float) -> bool:
    """isTemporalShortSegment."""
    return (ds < 2e-2 and abs(vx) < 0.1) or ds < 0.5 * max(abs(vx), 0.1) * max(dt, 1e-3)


class MpcReference:
    """setReferenceTrajectory of a temporal trajectory.

    Args:
        pts: ``[N, >= 6]``: x, y, yaw, v, a, t (time from the stamp), and optionally
            front_wheel_angle_rad.
        stamp: The trajectory's stamp, s.
        p: The MPC's parameters.
    """

    def __init__(self, pts: np.ndarray, stamp: float, p: MpcParams) -> None:
        self.stamp = stamp
        x = pts[:, 0].copy()
        y = pts[:, 1].copy()
        yaw = pts[:, 2].copy()
        v = pts[:, 3].copy()
        t = pts[:, 5].copy()
        s = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
        yaw_xy = yaw.copy()
        last = len(x) - 1
        for i in range(len(x)):
            ds_prev = s[i] - s[i - 1] if i > 0 else np.inf
            ds_next = s[i + 1] - s[i] if i < last else np.inf
            dt_prev = t[i] - t[i - 1] if i > 0 else 1.0
            dt_next = t[i + 1] - t[i] if i < last else 1.0
            v_prev = 0.5 * (v[i - 1] + v[i]) if i > 0 else 1.0
            v_next = 0.5 * (v[i] + v[i + 1]) if i < last else 1.0
            if (i > 0 and _short_segment(ds_prev, dt_prev, v_prev)) or (
                i < last and _short_segment(ds_next, dt_next, v_next)
            ):
                continue  # isTemporalShortSegment: keep the input yaw
            # findYawDifferentiationWindow, 0.5 m baseline
            lo, hi = max(i - 1, 0), min(i + 1, last)
            while s[hi] - s[lo] < 0.5 and not (lo == 0 and hi == last):
                lo, hi = max(lo - 1, 0), min(hi + 1, last)
            if s[hi] - s[lo] >= 0.5 and np.hypot(x[hi] - x[lo], y[hi] - y[lo]) >= 1e-3:
                yaw_xy[i] = np.arctan2(y[hi] - y[lo], x[hi] - x[lo])
        yaw = np.unwrap(yaw_xy)
        k = np.zeros(len(x))
        keep = np.r_[True, np.diff(s) > 1e-3]
        if s[-1] > 1e-6 and keep.sum() >= 3:
            ss = np.arange(0.0, s[-1], p.resample)
            if ss[-1] < s[-1] - 1e-6:
                ss = np.r_[ss, s[-1]]
            if len(ss) >= 3:
                sx = CubicSpline(s[keep], x[keep])(ss)
                sy = CubicSpline(s[keep], y[keep])(ss)
                k = np.interp(s, ss, _curvature(sx, sy, p.curv_num))
        self.x, self.y, self.yaw, self.v, self.t, self.k = x, y, yaw, v, t, k
        self.steer = pts[:, 6].copy() if pts.shape[1] > 6 else np.zeros(len(x))
        self.has_steer = bool(np.any(np.abs(self.steer) > 1e-6))

    def at(self, tau: Any) -> Tuple[Any, Any, Any, Any, Any]:
        """x, y, yaw, v and curvature at time(s) *tau* from the stamp."""
        return (
            np.interp(tau, self.t, self.x),
            np.interp(tau, self.t, self.y),
            np.interp(tau, self.t, self.yaw),
            np.interp(tau, self.t, self.v),
            np.interp(tau, self.t, self.k),
        )


class MpcLateralController:
    """autoware_mpc_lateral_controller's QP, kinematic bicycle model with first-order
    steer lag (OnePlanner's ``mpc_emu.MPC``).

    Mirrors mpc.cpp: setReferenceTrajectory (yaw from XY over a 0.5 m baseline,
    curvature by 0.1 m spatial resample with +-15 points), getData (reference at now
    - stamp), delay compensation over the command buffer, time resample from
    input_delay, generateMPCMatrix, the QP cost and the steering rate constraints.
    Not mirrored (inactive or second order there): the steering limit constraint,
    the velocity dynamics filter, steer offset removal, path smoothing (off by
    default).  The Butterworth LPF is a first-order stand-in.
    """

    def __init__(self, p: MpcParams) -> None:
        self.p = p
        self.nbuf = int(round(p.input_delay / p.ctrl_period))
        self.buf: List[float] = [0.0] * self.nbuf
        self.prev = 0.0
        self.pprev = 0.0
        self.lpf: Optional[float] = None
        self.last: Dict[str, float] = {}

    def reset(self, steer: float) -> None:
        """syncMpcSteerStateToCommand: every memory of the commands at *steer*."""
        self.buf = [steer] * self.nbuf
        self.prev = self.pprev = steer
        self.lpf = steer

    def _discrete(
        self, v: float, k: float, dt: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        p = self.p
        v = v if abs(v) >= 1e-4 else 1e-4
        dr = np.clip(np.arctan(p.wheelbase * k), -p.steer_lim, p.steer_lim)
        c2 = 1.0 / np.cos(dr) ** 2
        A = np.array(
            [[0, v, 0], [0, 0, v / p.wheelbase * c2], [0, 0, -1 / p.steer_tau]]
        )
        B = np.array([[0], [0], [1 / p.steer_tau]])
        W = np.array([[0], [-v * k + v / p.wheelbase * (np.tan(dr) - dr * c2)], [0]])
        Iinv = np.linalg.inv(np.eye(3) - dt * 0.5 * A)
        return Iinv @ (np.eye(3) + dt * 0.5 * A), Iinv @ B * dt, Iinv @ W * dt

    def step(
        self,
        ref: MpcReference,
        now: float,
        pose: Tuple[float, float, float],
        steer: float,
        record_buffer: bool = True,
    ) -> float:
        """One control tick.

        Args:
            ref: The trajectory followed.
            now: The time, s, on the trajectory's stamp's clock.
            pose: The localization's ``(x, y, yaw)``.
            steer: The measured tire angle, rad.
            record_buffer: Whether the command goes into the delay compensation's
                buffer.

        Returns:
            The filtered steering command, rad.
        """
        p = self.p
        tau = float(np.clip(now - ref.stamp, ref.t[0], ref.t[-1]))
        rx, ry, ryaw, _, _ = ref.at(tau)
        dx, dy = pose[0] - rx, pose[1] - ry
        lat = -np.sin(ryaw) * dx + np.cos(ryaw) * dy
        yerr = (pose[2] - ryaw + np.pi) % (2 * np.pi) - np.pi
        x0 = np.array([[lat], [yerr], [steer]])
        # delay compensation over the buffered commands, from the nearest time
        tc = tau
        for u in self.buf:
            _, _, _, v, k = ref.at(tc)
            Ad, Bd, Wd = self._discrete(v, k, p.ctrl_period)
            x0 = Ad @ x0 + Bd * u + Wd
            tc += p.ctrl_period
        ts = tau + p.input_delay + p.dt * np.arange(p.N)
        _, _, _, vs, ks = ref.at(ts)
        N = p.N
        Aex = np.zeros((3 * N, 3))
        Bex = np.zeros((3 * N, N))
        Wex = np.zeros((3 * N, 1))
        Cex = np.zeros((2 * N, 3 * N))
        Qex = np.zeros((2 * N, 2 * N))
        R1 = np.zeros((N, N))
        R2 = np.zeros((N, N))
        Uref = np.zeros((N, 1))
        for i in range(N):
            Ad, Bd, Wd = self._discrete(vs[i], ks[i], p.dt)
            if i == 0:
                Aex[0:3] = Ad
                Wex[0:3] = Wd
            else:
                Aex[3 * i : 3 * i + 3] = Ad @ Aex[3 * i - 3 : 3 * i]
                Bex[3 * i : 3 * i + 3, :i] = Ad @ Bex[3 * i - 3 : 3 * i, :i]
                Wex[3 * i : 3 * i + 3] = Ad @ Wex[3 * i - 3 : 3 * i] + Wd
            Bex[3 * i : 3 * i + 3, i : i + 1] = Bd
            Cex[2 * i, 3 * i] = 1.0
            Cex[2 * i + 1, 3 * i + 1] = 1.0
            wl, wh, wh2, ws, ws2, _ = _weights(p, ks[i])
            ql, qh = (p.w_term_lat, p.w_term_head) if i == N - 1 else (wl, wh)
            Qex[2 * i, 2 * i] = ql
            Qex[2 * i + 1, 2 * i + 1] = qh + vs[i] ** 2 * wh2
            R1[i, i] = ws + vs[i] ** 2 * ws2
            if p.ff_from_traj_steer and ref.has_steer:
                uff = float(
                    np.clip(
                        np.interp(ts[i], ref.t, ref.steer), -p.steer_lim, p.steer_lim
                    )
                )
            else:
                uff = float(np.arctan(p.wheelbase * ks[i]))
            Uref[i] = 0.0 if abs(uff) < np.radians(p.zero_ff_deg) else uff
        for i in range(N - 1):
            j = vs[i] ** 2 * _weights(p, ks[i])[5] / p.dt**2
            R2[i : i + 2, i : i + 2] += j * np.array([[1, -1], [-1, 1]])
        # steer rate / acceleration weights (addSteerWeightR / F)
        cp = p.ctrl_period
        D = p.w_steer_rate / p.dt**2 * np.array([[1, -1], [-1, 1]])
        for i in range(N - 1):
            R1[i : i + 2, i : i + 2] += D
        R1[0, 0] += p.w_steer_rate / cp**2
        w = p.w_steer_acc
        a4, c1, c2, c4 = (
            w / p.dt**4,
            w / (p.dt**3 * cp),
            w / (p.dt**2 * cp**2),
            w / cp**4,
        )
        D3 = a4 * np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]])
        for i in range(1, N - 1):
            R1[i - 1 : i + 2, i - 1 : i + 2] += D3
        R1[0, 0] += a4 + c2 + 2 * c1
        R1[1, 0] += -a4 - c1
        R1[0, 1] += -a4 - c1
        R1[1, 1] += a4
        R1[0, 0] += c4
        CB = Cex @ Bex
        QCB = Qex @ CB
        H = CB.T @ QCB + R1 + R2
        f = (Cex @ (Aex @ x0 + Wex)).T @ QCB - Uref.T @ R1
        f[0, 0] += -2.0 * p.w_steer_rate / p.dt**2 * 0.5
        f[0, 0] += ((-2.0 * self.prev + self.pprev) * c4) * 0.5
        f[0, 0] += (-2.0 * self.prev * (c1 + c2)) * 0.5
        f[0, 1] += (2.0 * self.prev * c1) * 0.5
        U = -np.linalg.solve(H, f.T)
        if p.constrained:
            # executeOptimization's rate constraints: |u0 - prev| <= r0 ctrl_period,
            # |u_i - u_i-1| <= r_i dt.  In difference variables z (u = T z) they are a
            # box; the steering limit is not binding here.
            rate = np.radians(
                [
                    min(_zoh_interp(p.rate_k, kk), _zoh_interp(p.rate_v, vv))
                    for kk, vv in zip(ks, vs)
                ]
            )
            if (
                abs(vs[0]) < 0.01
            ):  # the code tests the measured speed; the reference's stands in
                rate[:] = np.radians(100.0)
            hi = rate * p.dt
            lo = -hi
            hi[0], lo[0] = self.prev + rate[0] * cp, self.prev - rate[0] * cp
            z0 = np.r_[U[0, 0], np.diff(U[:, 0])]
            if np.any(z0 > hi + 1e-12) or np.any(z0 < lo - 1e-12):
                T = np.tril(np.ones((N, N)))
                Hz = T.T @ H @ T
                fz = (f @ T).ravel()
                z = np.clip(z0, lo, hi)
                step = 1.0 / np.linalg.eigvalsh(Hz)[-1]
                for _ in range(
                    3000
                ):  # projected gradient (box QP), exact to ~1e-9 rad here
                    z_new = np.clip(z - step * (Hz @ z + fz), lo, hi)
                    if np.max(np.abs(z_new - z)) < 1e-10:
                        z = z_new
                        break
                    z = z_new
                U = (T @ z)[:, None]
        u0 = float(np.clip(U[0, 0], -p.steer_lim, p.steer_lim))
        if self.lpf is None:
            self.lpf = u0
        a = 1.0 - np.exp(
            -2 * np.pi * p.lpf_hz * cp
        )  # first-order stand-in for the Butterworth LPF
        self.lpf += a * (u0 - self.lpf)
        self.pprev, self.prev = self.prev, float(U[0, 0])
        if record_buffer:
            self.buf = self.buf[1:] + [self.lpf]
        self.last = {
            "tau": tau,
            "lat": float(lat),
            "yaw_err": float(yerr),
            "ff": float(Uref[0, 0]),
            "raw": float(U[0, 0]),
        }
        return float(self.lpf)


# ---------------------------------------------------------------------------
# TSO: TrajectoryTimeSequenceRawOptimizer (OnePlanner's tso_opt.py)
# ---------------------------------------------------------------------------


@dataclass
class TsoParams:
    """The OCP's weights and bounds (the package defaults; :func:`load_tso_params`
    reads a parameter file's)."""

    dt: float = 0.1
    horizon: int = 80
    w_lon: float = 0.5
    w_lat: float = 0.5
    w_yaw: float = 0.05
    w_jerk: float = 0.1
    w_dr: float = 10.0
    term: float = 2.5
    j_max: float = 5.0
    dr_max: float = 1.0
    a_min: float = -4.0
    a_max: float = 3.0
    a_lat_max: float = 3.0
    v_min: float = 0.0
    v_max: float = 30.0
    delta_max: float = 0.7
    #: Temporal consistency: each stage also tracks the previous solution at the same
    #: absolute time, weight ``far + (1 - far) exp(-t / tau)`` times these; ``None``: off.
    tc: Optional[Dict[str, float]] = None
    #: x0's speed: the ego's measured speed (True) or the plan's first chord speeds.
    v0_from_ego: bool = False
    #: A short input's last stage as upstream #3641 solves it.
    upstream_short: bool = False
    #: The soft lateral-acceleration bound and the hard v / delta / a state bounds.
    bounds: bool = False
    z_lin: float = 0.0
    z_slack: float = 1.0e3
    w_hard: float = 1.0e3
    #: 1 / (m/s^2): softplus width of the slack (acados' slack is exact; the kink
    #: stalls least_squares).
    beta: float = 50.0
    #: TrajectoryTimeSequenceRawOptimizer::resolve_steer_stop_mode.
    stopped_velocity: float = 0.15
    stopped_length: float = 1.5


def load_tso_params(path: Path) -> TsoParams:
    """A ``time_sequence_raw_optimizer`` parameter file's weights and bounds."""
    q = _ros_parameters(path)["time_sequence_raw_optimizer"]
    params = TsoParams(
        w_lon=q["weight_longitudinal"],
        w_lat=q["weight_lateral"],
        w_yaw=q["weight_yaw"],
        w_jerk=q["weight_jerk"],
        w_dr=q["weight_steering_rate"],
        term=q["terminal_weight_scale"],
        j_max=q["max_jerk_mps3"],
        dr_max=q["max_steering_rate_rps"],
        a_min=q["min_acceleration_mps2"],
        a_max=q["max_acceleration_mps2"],
        a_lat_max=q["max_lateral_acceleration_mps2"],
        v_min=q["min_velocity_mps"],
        v_max=q["max_velocity_mps"],
        stopped_velocity=q.get("stopped_velocity_threshold_mps", 0.15),
        stopped_length=q.get("stopped_trajectory_max_length_m", 1.5),
    )
    tc = q.get("temporal_consistency")
    if tc and tc.get("enable"):
        params.tc = dict(
            lon=tc["weight_longitudinal"],
            lat=tc["weight_lateral"],
            yaw=tc["weight_yaw"],
            vel=tc["weight_velocity"],
            tau=tc["decay_time_constant_s"],
            far=tc["far_weight_ratio"],
        )
    return params


def _tso_f(x: np.ndarray, u: np.ndarray, L: float) -> np.ndarray:
    return np.array(
        [
            x[3] * np.cos(x[2]),
            x[3] * np.sin(x[2]),
            x[3] * np.tan(x[4]) / L,
            x[5],
            u[1],
            u[0],
        ]
    )


def _tso_rk4(x: np.ndarray, u: np.ndarray, L: float, dt: float) -> np.ndarray:
    k1 = _tso_f(x, u, L)
    k2 = _tso_f(x + 0.5 * dt * k1, u, L)
    k3 = _tso_f(x + 0.5 * dt * k2, u, L)
    k4 = _tso_f(x + dt * k3, u, L)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def _tso_step_jac(
    x: np.ndarray, u: np.ndarray, L: float, dt: float, eps: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    f0 = _tso_rk4(x, u, L, dt)
    A = np.empty((6, 6))
    B = np.empty((6, 2))
    for i in range(6):
        d = np.zeros(6)
        d[i] = eps
        A[:, i] = (_tso_rk4(x + d, u, L, dt) - f0) / eps
    for i in range(2):
        d = np.zeros(2)
        d[i] = eps
        B[:, i] = (_tso_rk4(x, u + d, L, dt) - f0) / eps
    return f0, A, B


def tso_optimize(
    q: TsoParams,
    ref: np.ndarray,
    x0: Tuple[float, ...],
    L: float,
    u_init: Optional[np.ndarray] = None,
    prev: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """The OCP over the reference stages *ref* ``[N, 3]`` (x, y, psi) from *x0*.

    OCP: states x, y, psi, v, delta, a; inputs jerk, delta_rate; ERK4.  Stage
    weights scaled as acados scales them; stage 0 tracks x0, stage s >= 1 tracks
    ``ref[s - 1]``, the last the terminal weight (unless upstream short semantics).

    Args:
        prev: ``[N, 4]`` the previous solution's (x, y, psi, v) at these stages'
            absolute times, or ``None``.

    Returns:
        ``(X [N + 1, 6], U [N, 2], status)``, status least_squares' (0: unconverged).
    """
    N = len(ref)
    DT = q.dt
    short_upstream = q.upstream_short and N < q.horizon
    n_tc = N if short_upstream else N - 1
    tc = q.tc
    use_tc = tc is not None and prev is not None and n_tc > 0
    unscale = 1.0 / DT
    c, s = np.cos(ref[:, 2]), np.sin(ref[:, 2])
    # acados scales a stage cost by its time step (cost_scaling default) and the terminal cost by 1
    w_stage = np.sqrt(DT * unscale * np.array([q.w_lon, q.w_lat, q.w_yaw]))
    w_term = np.sqrt(q.term / unscale * np.array([q.w_lon, q.w_lat, q.w_yaw]))
    w_u = np.sqrt(DT * unscale * np.array([q.w_jerk, q.w_dr]))
    x0a = np.asarray(x0, float)
    # The lateral-acceleration bound on the stages before the compiled terminal; only
    # the slack's quadratic part is mirrored (Gauss-Newton cannot model the linear
    # part), as a softplus of width 1 / beta.  The hard state bounds are stiff
    # quadratic penalties.
    n_h = (N if short_upstream else N - 1) if q.bounds else 0
    w_slack = np.sqrt(DT * q.z_slack)
    b_slack = DT * q.z_lin / np.sqrt(DT * q.z_slack)
    w_hard = np.sqrt(q.w_hard)
    lb_x = np.array([q.v_min, -q.delta_max, q.a_min])
    ub_x = np.array([q.v_max, q.delta_max, q.a_max])

    def roll(
        Z: np.ndarray, jac: bool
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        U = Z.reshape(N, 2)
        X = np.empty((N + 1, 6))
        X[0] = x0a
        S = np.zeros((N + 1, 6, 2 * N)) if jac else None
        for k in range(N):
            if S is not None:
                X[k + 1], A, B = _tso_step_jac(X[k], U[k], L, DT)
                S[k + 1] = A @ S[k]
                S[k + 1][:, 2 * k : 2 * k + 2] += B
            else:
                X[k + 1] = _tso_rk4(X[k], U[k], L, DT)
        return U, X, S

    w_tc = np.zeros((0, 4))
    if use_tc:
        assert tc is not None  # noqa: S101 - use_tc says so
        t_st = DT * np.arange(1, n_tc + 1)
        alpha = (
            tc["far"] + (1 - tc["far"]) * np.exp(-t_st / tc["tau"])
            if tc["tau"] > 0
            else np.ones(n_tc)
        )
        w_tc = np.sqrt(
            DT
            * unscale
            * alpha[:, None]
            * np.array([tc["lon"], tc["lat"], tc["yaw"], tc["vel"]])
        )

    def over(
        X: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """h at stages 1 .. n_h, its soft slack, the slack's d/dh, and the
        state-bound excess [N, 3] below / above."""
        Xh = X[1 : n_h + 1]
        h = Xh[:, 3] ** 2 * np.tan(Xh[:, 4]) / L
        m = np.sqrt(h * h + 1e-12) - q.a_lat_max
        sl = np.logaddexp(0.0, q.beta * m) / q.beta
        dsl = expit(q.beta * m) * h / np.sqrt(h * h + 1e-12)
        xs = X[1:, 3:6]
        return h, sl, dsl, np.maximum(0.0, lb_x - xs), np.maximum(0.0, xs - ub_x)

    def resid(z: np.ndarray) -> np.ndarray:
        U, X, _ = roll(z[: 2 * N], False)
        dx, dy, dp = X[1:, 0] - ref[:, 0], X[1:, 1] - ref[:, 1], X[1:, 2] - ref[:, 2]
        e = np.c_[c * dx + s * dy, -s * dx + c * dy, dp]
        w = np.tile(w_stage, (N, 1))
        if not short_upstream:
            w[-1] = w_term
        r = [(w_u * U).ravel(), (w * e).ravel()]
        if use_tc:
            assert prev is not None  # noqa: S101 - use_tc says so
            m = n_tc
            qx, qy = X[1 : m + 1, 0] - prev[:m, 0], X[1 : m + 1, 1] - prev[:m, 1]
            cc, ss = c[:m], s[:m]
            et = np.c_[
                cc * qx + ss * qy,
                -ss * qx + cc * qy,
                X[1 : m + 1, 2] - prev[:m, 2],
                X[1 : m + 1, 3] - prev[:m, 3],
            ]
            r.append((w_tc * et).ravel())
        if q.bounds:
            _, sl, _, below, above = over(X)
            r += [
                w_slack * sl + b_slack,
                w_hard * below.ravel(),
                w_hard * above.ravel(),
            ]
        return np.concatenate(r)

    def jac(z: np.ndarray) -> np.ndarray:
        _, X, S = roll(z[: 2 * N], True)
        assert S is not None  # noqa: S101 - asked for
        rows = 5 * N + (4 * n_tc if use_tc else 0) + (n_h + 6 * N if q.bounds else 0)
        J = np.zeros((rows, 2 * N))
        J[: 2 * N, : 2 * N] = np.diag(np.tile(w_u, N))
        for k in range(N):
            w = w_term if k == N - 1 and not short_upstream else w_stage
            Sx = S[k + 1]
            J[2 * N + 3 * k, : 2 * N] = w[0] * (c[k] * Sx[0] + s[k] * Sx[1])
            J[2 * N + 3 * k + 1, : 2 * N] = w[1] * (-s[k] * Sx[0] + c[k] * Sx[1])
            J[2 * N + 3 * k + 2, : 2 * N] = w[2] * Sx[2]
        base = 5 * N
        if use_tc:
            for k in range(n_tc):
                Sx = S[k + 1]
                wt = w_tc[k]
                J[base + 4 * k, : 2 * N] = wt[0] * (c[k] * Sx[0] + s[k] * Sx[1])
                J[base + 4 * k + 1, : 2 * N] = wt[1] * (-s[k] * Sx[0] + c[k] * Sx[1])
                J[base + 4 * k + 2, : 2 * N] = wt[2] * Sx[2]
                J[base + 4 * k + 3, : 2 * N] = wt[3] * Sx[3]
            base += 4 * n_tc
        if q.bounds:
            _, _, dsl, below, above = over(X)
            for k in map(int, np.flatnonzero(dsl != 0.0)):
                v, d = X[k + 1, 3], X[k + 1, 4]
                dh = (2 * v * np.tan(d) / L) * S[k + 1][3] + (
                    v * v / (L * np.cos(d) ** 2)
                ) * S[k + 1][4]
                J[base + k, : 2 * N] = w_slack * dsl[k] * dh
            base += n_h
            for k, i in zip(*np.nonzero(below)):
                J[base + 3 * k + i, : 2 * N] = -w_hard * S[k + 1][3 + i]
            base += 3 * N
            for k, i in zip(*np.nonzero(above)):
                J[base + 3 * k + i, : 2 * N] = w_hard * S[k + 1][3 + i]
        return J

    lo = np.tile([-q.j_max, -q.dr_max], N)
    hi = np.tile([q.j_max, q.dr_max], N)
    z0 = (
        np.zeros(2 * N)
        if u_init is None
        else np.clip(np.asarray(u_init, float).ravel(), lo, hi)
    )
    sol = least_squares(
        resid,
        z0,
        jac=jac,
        bounds=(lo, hi),
        method="trf",
        xtol=1e-10,
        ftol=1e-10,
        max_nfev=100 if q.bounds else 60,
    )
    U, X, _ = roll(sol.x, False)
    return X, U, int(sol.status)


def _stage_samples(pts: np.ndarray, q: TsoParams) -> Tuple[np.ndarray, np.ndarray]:
    """The trajectory's poses at stages t = dt k (k = 1 .. horizon), as far as it
    reaches (sampled by time_from_start, a t = 0 point is the start, a shorter
    trajectory is tracked to its own end)."""
    t = pts[:, 5]
    yaw = np.unwrap(pts[:, 2])
    ts = q.dt * np.arange(1, q.horizon + 1)
    ts = ts[ts <= t[-1] + 1e-6]
    return (
        np.c_[
            np.interp(ts, t, pts[:, 0]),
            np.interp(ts, t, pts[:, 1]),
            np.interp(ts, t, yaw),
        ],
        ts,
    )


def _previous_at(prev_X: np.ndarray, age_s: float, N: int, dt: float) -> np.ndarray:
    """The previous solution (states [M + 1, 6], stage 0 = its x0) at this cycle's
    stages 1 .. N: index k + age/dt, clamped to its horizon end."""
    idx = np.clip(np.arange(1, N + 1) + age_s / dt, 0, len(prev_X) - 1)
    out = np.empty((N, 4))
    for j in range(4):
        out[:, j] = np.interp(idx, np.arange(len(prev_X)), prev_X[:, j])
    return out


def tso_optimized_points(
    q: TsoParams,
    pts: np.ndarray,
    v_meas: float,
    delta0: float,
    accel0: float,
    L: float,
    u_init: Optional[np.ndarray] = None,
    prev: Optional[Tuple[np.ndarray, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """The TSO's output for a node trajectory (a t = 0 ego point first) under the
    time-indexed horizon.

    Returns:
        ``(points [N, 8], U, X, status)``: points at t = dt .. as x, y, yaw, v, a, t,
        front_wheel_angle, yaw rate.

    Raises:
        ValueError: If the trajectory reaches no stage.
    """
    ref, ts = _stage_samples(pts, q)
    if len(ref) == 0:
        raise ValueError("the trajectory reaches no TSO stage")
    chord = np.hypot(np.diff(ref[:3, 0]), np.diff(ref[:3, 1])) / q.dt
    v0 = float(
        np.clip(
            v_meas if q.v0_from_ego or len(chord) == 0 else np.mean(chord), 0.0, 30.0
        )
    )
    if u_init is not None and len(u_init) != len(ref):
        u_init = None
    p = None
    if prev is not None:
        p = _previous_at(prev[0], prev[1], len(ref), q.dt)
        p[:, 2] = ref[:, 2] + (p[:, 2] - ref[:, 2] + np.pi) % (2 * np.pi) - np.pi
    X, U, status = tso_optimize(
        q,
        ref,
        (
            pts[0, 0],
            pts[0, 1],
            pts[0, 2],
            v0,
            delta0,
            float(np.clip(accel0, q.a_min, q.a_max)),
        ),
        L,
        u_init,
        p,
    )
    out = np.c_[
        X[1:, 0],
        X[1:, 1],
        X[1:, 2],
        X[1:, 3],
        X[1:, 5],
        ts,
        X[1:, 4],
        X[1:, 3] * np.tan(X[1:, 4]) / L,
    ]
    return out, U, X, status


class TimeSequenceRawOptimizer:
    """The TSO node: warm start, the previous solution for temporal consistency, the
    previous trajectory on a failed solve, and the steer stop mode.

    Args:
        params: The OCP's parameters.
        wheelbase: The vehicle model's wheelbase, m.
        t0_point: Whether the output gets the ego's state at the stamp as a t = 0 point.
    """

    def __init__(self, params: TsoParams, wheelbase: float, t0_point: bool) -> None:
        self.params = params
        self.wheelbase = wheelbase
        self.t0_point = t0_point
        self.reset()

    def reset(self) -> None:
        """Forget the previous solutions."""
        self._u_prev: Optional[np.ndarray] = None
        self._prev: Optional[Tuple[np.ndarray, float]] = None
        self._ok: Optional[np.ndarray] = None
        self._steer_hold = False
        self._latched = 0.0

    def _stopped_reference(self, pts: np.ndarray) -> bool:
        """is_stopped_reference_trajectory."""
        length = float(np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1])).sum())
        return (
            length <= self.params.stopped_length
            and float(np.max(np.abs(pts[:, 3]))) <= self.params.stopped_velocity
        )

    def optimize(
        self, raw: np.ndarray, now: float, speed: float, steer: float, accel: float
    ) -> np.ndarray:
        """The trajectory the node publishes for *raw* (``[N, 8]``, a t = 0 ego point
        first), made at *now* from the ego's measured *speed*, tire angle *steer* and
        *accel*."""
        warm = self._u_prev
        warm = None if warm is None else np.r_[warm[1:], warm[-1:]]
        ego = raw[0].copy()
        prev = None if self._prev is None else (self._prev[0], now - self._prev[1])
        pts = raw
        try:
            pts, self._u_prev, X, status = tso_optimized_points(
                self.params, raw, speed, steer, accel, self.wheelbase, warm, prev
            )
            self._prev = (X, now)
            self._ok = pts
        except (np.linalg.LinAlgError, ValueError) as error:
            # The node on a failed solve (universe #3638): the previous successful
            # trajectory, else the input unchanged; warm start and previous solution kept.
            pts = self._ok if self._ok is not None else raw
            logger.warning(
                "TSO solve failed at t=%.2f s (%s): %s",
                now,
                type(error).__name__,
                "previous trajectory" if pts is not raw else "input unchanged",
            )
        # resolve_steer_stop_mode: with the ego stopped on a stopped reference the OCP
        # still runs, and front_wheel_angle_rad is the tire angle latched at entry.
        # The goal's Zero mode needs the route goal, which the chain is not given.
        if abs(speed) <= self.params.stopped_velocity and self._stopped_reference(raw):
            if not self._steer_hold:
                self._latched = steer
            self._steer_hold = True
            pts = pts.copy()
            pts[:, 6] = self._latched
        else:
            self._steer_hold = False
        if self.t0_point:
            ego[3], ego[5], ego[6] = pts[0, 3], 0.0, steer
            pts = np.r_[ego[None, : pts.shape[1]], pts]
        return pts


# ---------------------------------------------------------------------------
# The chain
# ---------------------------------------------------------------------------


@dataclass
class PublishedPlan:
    """A trajectory as the planning stack publishes it.

    Attributes:
        visible_us: When the controllers take it (the stamp plus the publish delay).
        points: ``[N, >= 6]`` x, y, yaw, v, a, t (time from the stamp), and with the
            TSO, the front wheel angle; in the local frame.
        reference: The MPC's reference made of it.
        trajectory: The same, for the longitudinal controller.
        velocities: Its speeds, when it carries its own (the TSO's), else ``None``.
        accelerations: With *velocities*, its accelerations.
    """

    visible_us: int
    points: np.ndarray
    reference: MpcReference
    trajectory: Trajectory
    velocities: Optional[List[float]] = None
    accelerations: Optional[List[float]] = None


@dataclass
class _PoseSample:
    t: float
    x: float
    y: float
    yaw: float


class AutowareLateralChain:
    """The TSO, the publish delay and the MPC between the policy's plan and the steering.

    Call :meth:`submit` with each plan the policy returns, and :meth:`step` every
    simulation tick; :meth:`published` is the trajectory the controllers follow now,
    for the longitudinal controller.

    Args:
        config: The chain's settings.
        control: The trajectory follower's settings, for the speeds the e2e node puts
            on a plan (``velocity_smoothing_window``, ``stopping_threshold``).
    """

    def __init__(
        self, config: LateralChainConfig, control: Optional[ControlConfig] = None
    ) -> None:
        self.config = config
        self.control = control if control is not None else ControlConfig()
        mpc_file = (
            Path(config.mpc_param_file)
            if config.mpc_param_file
            else DEFAULT_MPC_PARAM_FILE
        )
        self.mpc_params, stop = load_mpc_params(
            mpc_file, wheelbase=config.wheelbase_m, ctrl_period=config.ctrl_period_s
        )
        if config.mpc_companion:
            stop = dataclasses.replace(stop, hold=9999.0, slew=False)
        self.stop_params = stop
        self.tso: Optional[TimeSequenceRawOptimizer] = None
        if config.trajectory_optimizer == "tso":
            tso_file = (
                Path(config.tso_param_file)
                if config.tso_param_file
                else DEFAULT_TSO_PARAM_FILE
            )
            tso_params = load_tso_params(tso_file)
            tso_params.v0_from_ego = config.tso_v0_from_ego
            tso_params.upstream_short = config.tso_upstream_semantics
            tso_params.bounds = config.tso_bounds
            self.tso = TimeSequenceRawOptimizer(
                tso_params, config.wheelbase_m, config.tso_t0_point
            )
        self.reset()

    def reset(self) -> None:
        """Forget every plan, the MPC's memory and the pose history."""
        self.mpc = MpcLateralController(self.mpc_params)
        self._mpc_started = False
        self._pending: List[PublishedPlan] = []
        self._active: Optional[PublishedPlan] = None
        self._next_tick = 0.0
        self._cmd: Optional[float] = None
        self._cmd_prev = 0.0
        self._hold_since: Optional[float] = None
        self._conf_since: Optional[float] = None
        self._poses: List[_PoseSample] = []
        #: How many MPC ticks ran.
        self.tick_count = 0
        if self.tso is not None:
            self.tso.reset()

    # -- plans ------------------------------------------------------------------

    def submit(
        self,
        plan: Trajectory,
        stamp_us: int,
        pose: Pose,
        speed_mps: float,
        steer_rad: float,
        accel_mps2: float,
    ) -> None:
        """Take a plan the policy returned.

        Args:
            plan: The plan, in the local frame.
            stamp_us: When it was asked for: the ego was at *pose* then.
            pose: Where the ego was (the rig, in the local frame).
            speed_mps: The ego's measured speed then.
            steer_rad: Its measured tire angle then, positive left.
            accel_mps2: Its measured longitudinal acceleration then.
        """
        stamp_s = stamp_us / _US_PER_S
        times = (np.asarray(plan.timestamps_us, dtype=np.int64) - stamp_us) / _US_PER_S
        keep = times > 1e-3
        if not keep.any():
            return
        positions = plan.positions[keep]
        yaws = np.array([p.yaw for p in plan.poses])[keep]
        times = times[keep]
        stamps = np.asarray(plan.timestamps_us, dtype=np.int64)[keep].tolist()
        base = np.asarray(pose.position, dtype=np.float64)
        speeds, _, pose_index = plan_velocities(
            positions,
            stamps,
            base,
            stamp_us,
            self.control.velocity_smoothing_window,
            speed_mps > 0.0,
            self.control.stopping_threshold,
        )
        positions = positions[pose_index]
        yaws = yaws[pose_index]
        # The node's trajectory: an ego point at the stamp, then the plan.
        n = len(times) + 1
        pts = np.zeros((n, 8))
        pts[:, 0] = np.r_[base[0], positions[:, 0]]
        pts[:, 1] = np.r_[base[1], positions[:, 1]]
        pts[:, 2] = np.unwrap(np.r_[pose.yaw, yaws])
        pts[:, 3] = np.r_[speeds[0], speeds]
        pts[:, 5] = np.r_[0.0, times]
        velocities: Optional[List[float]] = None
        accelerations: Optional[List[float]] = None
        if self.tso is not None:
            pts = self.tso.optimize(pts, stamp_s, speed_mps, steer_rad, accel_mps2)
            velocities = [float(v) for v in pts[:, 3]]
            accelerations = [float(a) for a in pts[:, 4]]
        trajectory = Trajectory.empty()
        z = float(base[2])
        for row in pts:
            trajectory.append(
                stamp_us + int(round(float(row[5]) * _US_PER_S)),
                Pose.from_xyz_yaw(float(row[0]), float(row[1]), z, float(row[2])),
            )
        self._pending.append(
            PublishedPlan(
                visible_us=stamp_us
                + int(round(self.config.publish_delay_s * _US_PER_S)),
                points=pts,
                reference=MpcReference(pts, stamp_s, self.mpc_params),
                trajectory=trajectory,
                velocities=velocities,
                accelerations=accelerations,
            )
        )

    def published(self, now_us: int) -> Optional[PublishedPlan]:
        """The trajectory the controllers follow at *now_us*; ``None`` before the first
        is published."""
        self._publish(now_us)
        return self._active

    def _publish(self, now_us: int) -> None:
        while self._pending and self._pending[0].visible_us <= now_us:
            self._active = self._pending.pop(0)
            if not self._mpc_started:
                self._next_tick = self._active.visible_us / _US_PER_S

    # -- localization -------------------------------------------------------------

    def _localization(self, now: float, pose: Pose) -> Tuple[float, float, float]:
        """The pose the localization reports at *now*: the vehicle's
        ``localization_lag_s`` ago, interpolated between the ticks."""
        position = pose.position
        sample = _PoseSample(
            now, float(position[0]), float(position[1]), float(pose.yaw)
        )
        if self._poses and self._poses[-1].t >= now:
            self._poses[-1] = sample
        else:
            self._poses.append(sample)
        lag = self.config.localization_lag_s
        # The samples from the last one at or before now - lag on are all it reads.
        while len(self._poses) > 2 and self._poses[1].t <= now - lag:
            self._poses.pop(0)
        if lag <= 0.0 or len(self._poses) < 2:
            return sample.x, sample.y, sample.yaw
        t = np.array([s.t for s in self._poses])
        yaw = np.unwrap([s.yaw for s in self._poses])
        at = now - lag
        return (
            float(np.interp(at, t, [s.x for s in self._poses])),
            float(np.interp(at, t, [s.y for s in self._poses])),
            float(np.interp(at, t, yaw)),
        )

    # -- control ------------------------------------------------------------------

    def step(
        self, now_us: int, pose: Pose, speed_mps: float, steer_rad: float
    ) -> Optional[float]:
        """The tire angle, rad (positive left), the MPC asks for at *now_us*.

        Runs every MPC tick that fell due since the last call, on the pose at
        *now_us*.  ``None`` before the first trajectory is published.

        Args:
            now_us: The simulation time.
            pose: The ego's pose (the rig, in the local frame), true.
            speed_mps: Its measured speed.
            steer_rad: Its measured tire angle, positive left.
        """
        now = now_us / _US_PER_S
        seen = self._localization(now, pose)
        self._publish(now_us)
        active = self._active
        if active is None:
            return None
        if not self._mpc_started:
            self.mpc.reset(steer_rad)
            self._cmd_prev = steer_rad
            self._mpc_started = True
        period = self.mpc_params.ctrl_period
        while now >= self._next_tick - 1e-9:
            raw = self.mpc.step(active.reference, now, seen, steer_rad)
            self._cmd = self._lateral_stop(raw, seen, now, speed_mps, active)
            self.tick_count += 1
            self._next_tick += period
        return self._cmd

    def _lateral_stop(
        self,
        cmd: float,
        pose: Tuple[float, float, float],
        now: float,
        speed: float,
        active: PublishedPlan,
    ) -> float:
        """The MPC node's command after its stop handling, given the solved command."""
        S = self.stop_params
        min_v, ahead = _remaining_after_nearest(active.points, pose)
        eligible = abs(speed) <= S.entry_ego and abs(min_v) < S.entry_target
        if not eligible:
            self._hold_since = None
        else:
            if self._hold_since is None:
                self._hold_since = now
            if now - self._hold_since > S.hold:
                self.mpc.reset(self._cmd_prev)  # syncMpcSteerStateToCommand
                return self._cmd_prev
        if S.slew:
            conf = float(
                np.clip((ahead - S.l_min) / max(S.l_ref - S.l_min, 1e-6), 0.0, 1.0)
            )
            if conf >= 1.0 - 1e-6:
                self._conf_since = None
            else:
                if self._conf_since is None:
                    self._conf_since = now
                if now - self._conf_since <= S.hold:
                    ds = cmd - self._cmd_prev
                    ds_max = (
                        1 - conf
                    ) * S.rate_min * self.mpc_params.ctrl_period + conf * abs(ds)
                    if abs(ds) > ds_max:
                        cmd = self._cmd_prev + float(np.clip(ds, -ds_max, ds_max))
                        self.mpc.reset(cmd)
        self._cmd_prev = cmd
        return cmd


def _remaining_after_nearest(
    pts: np.ndarray, pose: Tuple[float, float, float]
) -> Tuple[float, float]:
    """(min target speed within 1 m after the nearest point, arc length from it to the
    end): getMinTargetVelocityAhead, getReferenceRemainingArcLength (nearest without
    the soft yaw constraint)."""
    d = np.hypot(pts[:, 0] - pose[0], pts[:, 1] - pose[1])
    i = int(np.argmin(d))
    seg = np.hypot(np.diff(pts[i:, 0]), np.diff(pts[i:, 1]))
    covered = np.cumsum(seg)
    j = i + 1 + int(np.searchsorted(covered, 1.0, side="right")) if len(seg) else i
    min_v = float(np.min(pts[i : min(j + 1, len(pts)), 3]))
    return min_v, float(seg.sum())
