#!/usr/bin/env python3
"""Adamo ↔ Unitree teleop bridge.

Runs on the Unitree robot PC. Does two things:

  1. Publishes one Adamo video track per camera flag — hardware-encoded
     H.264, picked up automatically by the Adamo frontend's video player.

  2. Subscribes to the control streams the Adamo frontend publishes:

       adamo/{org}/{robot}/control/cdr/xr_tracking
           CDR ROS envelopes multiplexed on one Zenoh topic. Inner topics:
             /head_pose                       geometry_msgs/PoseStamped
             /controller/{left|right}         geometry_msgs/PoseStamped
             /controller/{left|right}/joy     sensor_msgs/Joy

       adamo/{org}/{robot}/control/{joy_channel}
           Plain-gamepad sensor_msgs/Joy from @adamo/teleop's JoypadManager
           (CDR-with-envelope *or* JSON {"type":"Joy","axes","buttons"}).

     Joy from either source is forwarded into the Unitree locomotion
     client. Wrist poses are buffered; arm IK is opt-in via --enable-ik
     (requires the teleop pin/dex-retargeting stack to be importable).

Launch:

    ADAMO_API_KEY=ak_...  python teleop/adamo_bridge.py \\
        --robot-name g1-01 \\
        --head /dev/video0 \\
        --left-wrist /dev/video2 --right-wrist /dev/video4
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import signal
import struct
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

log = logging.getLogger("adamo_bridge")


# ---------------------------------------------------------------------------
# CDR + ROS-envelope decoding (minimal, enough for Joy + PoseStamped)
# ---------------------------------------------------------------------------
#
# Envelope: 4B BE topic_len | topic | 4B BE type_len | type | CDR payload.
# CDR payload: 4B encapsulation header (we only accept LE PLAIN_CDR), then
# the message fields with standard CDR alignment.

@dataclass
class Joy:
    axes: list[float]
    buttons: list[int]


@dataclass
class Pose:
    """6DoF pose. Rotation is a right-handed quaternion (x, y, z, w)."""
    pos: np.ndarray     # shape (3,)
    quat: np.ndarray    # shape (4,) — (x, y, z, w)


def _decode_envelope(data: bytes) -> tuple[str, str, bytes] | None:
    if len(data) < 8:
        return None
    try:
        topic_len = struct.unpack_from(">I", data, 0)[0]
        type_off = 4 + topic_len
        if type_off + 4 > len(data):
            return None
        type_len = struct.unpack_from(">I", data, type_off)[0]
        payload_off = type_off + 4 + type_len
        if payload_off > len(data):
            return None
        topic = data[4 : 4 + topic_len].decode("utf-8", errors="replace")
        mtype = data[type_off + 4 : type_off + 4 + type_len].decode("utf-8", errors="replace")
        return topic, mtype, data[payload_off:]
    except Exception:
        return None


class _Cdr:
    def __init__(self, buf: bytes):
        self.buf = buf
        self.pos = 0

    def _align(self, n: int) -> None:
        self.pos += (-self.pos) % n

    def u32(self) -> int:
        self._align(4); v = struct.unpack_from("<I", self.buf, self.pos)[0]; self.pos += 4; return v

    def i32(self) -> int:
        self._align(4); v = struct.unpack_from("<i", self.buf, self.pos)[0]; self.pos += 4; return v

    def f32(self) -> float:
        self._align(4); v = struct.unpack_from("<f", self.buf, self.pos)[0]; self.pos += 4; return v

    def f64(self) -> float:
        self._align(8); v = struct.unpack_from("<d", self.buf, self.pos)[0]; self.pos += 8; return v

    def string(self) -> str:
        n = self.u32()
        raw = self.buf[self.pos : self.pos + max(0, n - 1)]
        self.pos += n
        return raw.decode("utf-8", errors="replace")

    def skip_header(self) -> None:
        # std_msgs/Header: builtin_interfaces/Time (sec int32 + nanosec uint32) + frame_id string
        self.i32(); self.u32(); self.string()


def _cdr_from_payload(payload: bytes) -> Optional[_Cdr]:
    if len(payload) < 4:
        return None
    if payload[1] != 1:   # accept LE PLAIN_CDR only — that's what the frontend emits
        return None
    return _Cdr(payload[4:])


def _decode_joy_cdr(payload: bytes) -> Optional[Joy]:
    r = _cdr_from_payload(payload)
    if r is None:
        return None
    r.skip_header()
    n = r.u32(); axes = [r.f32() for _ in range(n)]
    n = r.u32(); buttons = [r.i32() for _ in range(n)]
    return Joy(axes=axes, buttons=buttons)


def _decode_posestamped_cdr(payload: bytes) -> Optional[Pose]:
    r = _cdr_from_payload(payload)
    if r is None:
        return None
    r.skip_header()
    # geometry_msgs/Pose: Point (3x f64) + Quaternion (4x f64)
    px = r.f64(); py = r.f64(); pz = r.f64()
    qx = r.f64(); qy = r.f64(); qz = r.f64(); qw = r.f64()
    return Pose(pos=np.array([px, py, pz]), quat=np.array([qx, qy, qz, qw]))


def _decode_joy_json(payload: bytes) -> Optional[Joy]:
    if payload[:1] != b"{":
        return None
    try:
        obj = json.loads(payload)
    except Exception:
        return None
    if obj.get("type") not in ("Joy", "JoystickCommand"):
        return None
    return Joy(
        axes=[float(x) for x in obj.get("axes", [])],
        buttons=[int(x) for x in obj.get("buttons", [])],
    )


# ---------------------------------------------------------------------------
# Pose transforms
# ---------------------------------------------------------------------------
#
# WebXR xr_origin frame (what the frontend emits): +x right, +y up, -z forward.
# Robot torso IK frame (ROS convention):            +x forward, +y left, +z up.
#
# The mapping below is an identity calibration: robot_x = -xr_z, robot_y = -xr_x,
# robot_z = xr_y. It gets you in-the-ballpark poses without a per-user recenter —
# refine with a per-operator calibration step when needed.

_XR_TO_ROBOT_R = np.array([
    [ 0.0,  0.0, -1.0],
    [-1.0,  0.0,  0.0],
    [ 0.0,  1.0,  0.0],
])


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy)],
        [2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy)],
    ])


def pose_to_robot_tf(p: Pose, origin_offset: np.ndarray) -> np.ndarray:
    """WebXR PoseStamped → 4x4 homogeneous transform in the robot torso frame."""
    R_local = _quat_to_rot(p.quat)
    R = _XR_TO_ROBOT_R @ R_local @ _XR_TO_ROBOT_R.T
    t = _XR_TO_ROBOT_R @ p.pos + origin_offset
    tf = np.eye(4)
    tf[:3, :3] = R
    tf[:3, 3] = t
    return tf


# ---------------------------------------------------------------------------
# Joy → locomotion
# ---------------------------------------------------------------------------
#
# Matches teleop_hand_and_arm.py's controller path (lines ~316-321):
#   axes[0] = left stick X  → vy (lateral)        (sign flipped)
#   axes[1] = left stick Y  → vx (forward/back)   (sign flipped)
#   axes[2] = right stick X → vyaw                (sign flipped)
# Both thumbsticks pressed → damp (soft e-stop).

class LocoDriver:
    def __init__(self, speed_scale: float, timeout_s: float):
        from teleop.utils.motion_switcher import LocoClientWrapper
        self._wrapper = LocoClientWrapper()
        self._scale = speed_scale
        self._timeout = timeout_s
        self._last_rx = 0.0
        self._stop = threading.Event()
        threading.Thread(target=self._watchdog, name="loco-watchdog", daemon=True).start()

    def on_joy(self, joy: Joy) -> None:
        self._last_rx = time.monotonic()
        # Both thumbsticks pressed (ROS mapping: button 9 = L stick, 10 = R stick)
        if len(joy.buttons) > 10 and joy.buttons[9] and joy.buttons[10]:
            try:
                self._wrapper.Enter_Damp_Mode()
            except Exception as e:
                log.warning("Damp failed: %s", e)
            return
        if len(joy.axes) < 4:
            return
        vx = -joy.axes[1] * self._scale
        vy = -joy.axes[0] * self._scale
        vyaw = -joy.axes[2] * self._scale
        try:
            self._wrapper.Move(vx, vy, vyaw)
        except Exception as e:
            log.warning("Move failed: %s", e)

    def _watchdog(self) -> None:
        while not self._stop.is_set():
            time.sleep(0.05)
            if self._last_rx and time.monotonic() - self._last_rx > self._timeout:
                try:
                    self._wrapper.Move(0.0, 0.0, 0.0)
                except Exception:
                    pass
                self._last_rx = 0.0

    def close(self) -> None:
        self._stop.set()


# ---------------------------------------------------------------------------
# XR wrist poses → arm IK (opt-in)
# ---------------------------------------------------------------------------


@dataclass
class PoseBuffer:
    left: Optional[Pose] = None
    right: Optional[Pose] = None
    head: Optional[Pose] = None
    left_t: float = 0.0
    right_t: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)


class ArmDriver:
    """Owns IK + arm controller. Runs the IK loop at a fixed frequency."""

    def __init__(self, arm: str, motion: bool, sim: bool, frequency: float,
                 pose_max_age_s: float, origin_offset: np.ndarray, buf: PoseBuffer):
        from teleop.robot_control.robot_arm import (
            G1_29_ArmController, G1_23_ArmController,
            H1_2_ArmController, H1_ArmController,
        )
        from teleop.robot_control.robot_arm_ik import (
            G1_29_ArmIK, G1_23_ArmIK, H1_2_ArmIK, H1_ArmIK,
        )
        by_name = {
            "G1_29": (G1_29_ArmIK, G1_29_ArmController),
            "G1_23": (G1_23_ArmIK, G1_23_ArmController),
            "H1_2":  (H1_2_ArmIK,  H1_2_ArmController),
            "H1":    (H1_ArmIK,    H1_ArmController),
        }
        ik_cls, ctrl_cls = by_name[arm]
        self.ik = ik_cls()
        self.ctrl = ctrl_cls(simulation_mode=sim) if arm == "H1" else ctrl_cls(motion_mode=motion, simulation_mode=sim)
        self.ctrl.speed_gradual_max()
        self._freq = frequency
        self._max_age = pose_max_age_s
        self._origin = origin_offset
        self._buf = buf
        self._stop = threading.Event()
        threading.Thread(target=self._run, name="ik-loop", daemon=True).start()

    def _run(self) -> None:
        period = 1.0 / max(1e-3, self._freq)
        log.info("arm IK loop running @ %.1f Hz", self._freq)
        while not self._stop.is_set():
            t0 = time.monotonic()
            with self._buf.lock:
                lp, rp = self._buf.left, self._buf.right
                lt, rt = self._buf.left_t, self._buf.right_t
            if lp is not None and rp is not None:
                now = time.monotonic()
                if now - lt < self._max_age and now - rt < self._max_age:
                    try:
                        L = pose_to_robot_tf(lp, self._origin)
                        R = pose_to_robot_tf(rp, self._origin)
                        q = self.ctrl.get_current_dual_arm_q()
                        dq = self.ctrl.get_current_dual_arm_dq()
                        sol_q, sol_tauff = self.ik.solve_ik(L, R, q, dq)
                        self.ctrl.ctrl_dual_arm(sol_q, sol_tauff)
                    except Exception as e:
                        log.warning("IK step failed: %s", e)
            dt = time.monotonic() - t0
            if dt < period:
                time.sleep(period - dt)

    def close(self) -> None:
        self._stop.set()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s: %(message)s")

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--robot-name", default=os.environ.get("ADAMO_ROBOT"))
    p.add_argument("--api-key", default=os.environ.get("ADAMO_API_KEY"))
    p.add_argument("--protocol", default="udp", choices=["udp", "quic", "tcp"])
    p.add_argument("--network-interface", default=None,
                   help="Interface for Unitree DDS (e.g. eth0)")
    p.add_argument("--sim", action="store_true",
                   help="Use DDS domain 1 (Isaac sim) instead of 0")

    # cameras
    p.add_argument("--head", default=None, help="head camera device path or index")
    p.add_argument("--left-wrist", default=None)
    p.add_argument("--right-wrist", default=None)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--bitrate-kbps", type=int, default=4000)

    # control — XR tracking is the primary channel; gamepad joy channel optional
    p.add_argument("--xr-channel", default="cdr/xr_tracking",
                   help="Adamo control sub-key for XR tracking (matches the Adamo frontend)")
    p.add_argument("--joy-channel", default="joy",
                   help="Adamo control sub-key for plain gamepad Joy (JoypadManager default)")
    p.add_argument("--locomotion-source", default="gamepad",
                   choices=["gamepad", "right", "left", "off"],
                   help="Which Joy stream drives loco: gamepad / XR right controller / XR left / off")
    p.add_argument("--speed-scale", type=float, default=0.3)
    p.add_argument("--control-timeout", type=float, default=0.5,
                   help="Stop walking if no Joy for this many seconds")

    # arm IK (opt-in)
    p.add_argument("--enable-ik", action="store_true",
                   help="Run arm IK on XR wrist poses and drive the arm controller")
    p.add_argument("--arm", choices=["G1_29", "G1_23", "H1_2", "H1"], default="G1_29")
    p.add_argument("--motion", action="store_true",
                   help="Arm controller motion-mode flag (ignored for H1)")
    p.add_argument("--frequency", type=float, default=30.0,
                   help="IK / arm control loop rate")
    p.add_argument("--pose-max-age-s", type=float, default=0.2,
                   help="Skip IK step if either wrist pose is older than this")
    p.add_argument("--origin-offset", type=float, nargs=3, default=[0.25, 0.0, 0.1],
                   metavar=("X", "Y", "Z"),
                   help="Additive offset (metres, robot frame) applied to XR poses "
                        "— shifts the operator 'neutral' point into the robot's arm workspace")

    args = p.parse_args()

    if not args.robot_name:
        print("error: --robot-name (or $ADAMO_ROBOT) is required", file=sys.stderr); return 2
    if not args.api_key:
        print("error: --api-key (or $ADAMO_API_KEY) is required", file=sys.stderr); return 2

    # DDS must be initialised before any Unitree SDK client is constructed.
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    ChannelFactoryInitialize(1 if args.sim else 0, networkInterface=args.network_interface)

    import adamo

    robot = adamo.Robot(api_key=args.api_key, name=args.robot_name, protocol=args.protocol)

    # -- video --
    attached_cams: list[str] = []
    for track, device in [("head", args.head),
                          ("left_wrist", args.left_wrist),
                          ("right_wrist", args.right_wrist)]:
        if not device:
            continue
        robot.attach_video(
            track, device=device,
            width=args.width, height=args.height,
            fps=args.fps, bitrate_kbps=args.bitrate_kbps,
        )
        attached_cams.append(track)
        log.info("video track '%s' attached @ %dx%d/%dfps from %s",
                 track, args.width, args.height, args.fps, device)

    # -- locomotion --
    loco = LocoDriver(args.speed_scale, args.control_timeout) if args.locomotion_source != "off" else None

    # -- arm IK --
    pose_buf = PoseBuffer()
    arm = None
    if args.enable_ik:
        arm = ArmDriver(
            arm=args.arm, motion=args.motion, sim=args.sim,
            frequency=args.frequency, pose_max_age_s=args.pose_max_age_s,
            origin_offset=np.array(args.origin_offset, dtype=float),
            buf=pose_buf,
        )

    # -- control subscribers --
    seen_xr = {"count": 0, "topics": set()}

    def on_xr(payload: bytes) -> None:
        env = _decode_envelope(payload)
        if env is None:
            return
        inner_topic, mtype, body = env
        seen_xr["count"] += 1
        if inner_topic not in seen_xr["topics"]:
            seen_xr["topics"].add(inner_topic)
            log.info("XR topic seen: %s (%s)", inner_topic, mtype)

        if mtype == "geometry_msgs/msg/PoseStamped":
            pose = _decode_posestamped_cdr(body)
            if pose is None:
                return
            now = time.monotonic()
            with pose_buf.lock:
                if inner_topic.endswith("/left") or inner_topic == "/controller/left":
                    pose_buf.left = pose; pose_buf.left_t = now
                elif inner_topic.endswith("/right") or inner_topic == "/controller/right":
                    pose_buf.right = pose; pose_buf.right_t = now
                elif "head" in inner_topic:
                    pose_buf.head = pose
            return

        if mtype == "sensor_msgs/msg/Joy":
            joy = _decode_joy_cdr(body)
            if joy is None:
                return
            if loco is None:
                return
            # Route based on which XR controller drives locomotion
            src = args.locomotion_source
            if src == "right" and "right" in inner_topic:
                loco.on_joy(joy)
            elif src == "left" and "left" in inner_topic:
                loco.on_joy(joy)

    def on_gamepad(payload: bytes) -> None:
        # @adamo/teleop JoypadManager: CDR-with-envelope by default, JSON optional.
        joy = _decode_joy_json(payload)
        if joy is None:
            env = _decode_envelope(payload)
            if env is None:
                return
            _, mtype, body = env
            if "Joy" not in mtype:
                return
            joy = _decode_joy_cdr(body)
        if joy is None or loco is None:
            return
        if args.locomotion_source == "gamepad":
            loco.on_joy(joy)

    robot.session.subscribe(
        f"{args.robot_name}/control/{args.xr_channel}", callback=on_xr,
    )
    log.info("subscribed: %s/control/%s", args.robot_name, args.xr_channel)

    if args.joy_channel:
        robot.session.subscribe(
            f"{args.robot_name}/control/{args.joy_channel}", callback=on_gamepad,
        )
        log.info("subscribed: %s/control/%s", args.robot_name, args.joy_channel)

    # Liveliness — frontend uses this to discover the robot.
    alive = robot.session.alive(f"{args.robot_name}/alive")

    # Run the Rust video pipeline if any camera was attached.
    if attached_cams:
        threading.Thread(target=robot.run, name="adamo-pipeline", daemon=True).start()

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    log.info("bridge up: robot=%s cameras=%s loco=%s ik=%s",
             args.robot_name, attached_cams, args.locomotion_source, args.enable_ik)

    try:
        while not stop.is_set():
            time.sleep(0.5)
    finally:
        if arm: arm.close()
        if loco: loco.close()
        try: alive.undeclare()
        except Exception: pass
        robot.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
