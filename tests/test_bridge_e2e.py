#!/usr/bin/env python3
"""End-to-end test for teleop/adamo_bridge.py.

No robot, no network, no Unitree SDK install required. We:

  1. Stub the Unitree SDK modules the bridge imports (``LocoClientWrapper``,
     ``G1_29_ArmController``, ``G1_29_ArmIK``) so they record calls instead
     of talking to hardware.
  2. Encode synthetic ``sensor_msgs/Joy`` and ``geometry_msgs/PoseStamped``
     messages in the exact CDR + ROS-envelope bytes the Adamo frontend
     emits (LE encapsulation 00 01 00 00, per adamo-ts/packages/ros/src/cdr.ts).
  3. Drive the bridge's decoders + driver classes with those bytes and
     assert we get the right Unitree SDK calls out the other end.

Run:  python3 tests/test_bridge_e2e.py
"""
from __future__ import annotations

import os
import struct
import sys
import time
import types
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


# ---------------------------------------------------------------------------
# Stub Unitree SDK modules  (injected before we import the bridge)
# ---------------------------------------------------------------------------

class FakeLoco:
    def __init__(self):
        self.move_calls: list[tuple[float, float, float]] = []
        self.damp_calls: int = 0

    def Move(self, vx, vy, vyaw):
        self.move_calls.append((vx, vy, vyaw))

    def Enter_Damp_Mode(self):
        self.damp_calls += 1


class FakeArmController:
    def __init__(self, *_, **__):
        self.q = np.zeros(14)
        self.dq = np.zeros(14)
        self.ctrl_calls: list[tuple[np.ndarray, np.ndarray]] = []

    def speed_gradual_max(self):
        pass

    def get_current_dual_arm_q(self):
        return self.q.copy()

    def get_current_dual_arm_dq(self):
        return self.dq.copy()

    def ctrl_dual_arm(self, sol_q, sol_tauff):
        self.ctrl_calls.append((np.asarray(sol_q).copy(), np.asarray(sol_tauff).copy()))


class FakeArmIK:
    def __init__(self, *_, **__):
        self.solve_calls: list[tuple[np.ndarray, np.ndarray]] = []

    def solve_ik(self, L, R, q=None, dq=None):
        self.solve_calls.append((np.asarray(L).copy(), np.asarray(R).copy()))
        # Return dummy joint targets + feedforward torques (14 dof for G1_29 dual arm)
        return np.zeros(14), np.zeros(14)


def _install_stubs():
    ms = types.ModuleType("teleop.utils.motion_switcher")
    ms.LocoClientWrapper = FakeLoco
    sys.modules["teleop.utils.motion_switcher"] = ms

    ra = types.ModuleType("teleop.robot_control.robot_arm")
    for name in ("G1_29_ArmController", "G1_23_ArmController",
                 "H1_2_ArmController", "H1_ArmController"):
        setattr(ra, name, FakeArmController)
    sys.modules["teleop.robot_control.robot_arm"] = ra

    rik = types.ModuleType("teleop.robot_control.robot_arm_ik")
    for name in ("G1_29_ArmIK", "G1_23_ArmIK", "H1_2_ArmIK", "H1_ArmIK"):
        setattr(rik, name, FakeArmIK)
    sys.modules["teleop.robot_control.robot_arm_ik"] = rik


_install_stubs()

# Bridge imports the fake modules lazily inside driver ctors, so importing here
# is safe (no DDS, no hardware touched).
from teleop import adamo_bridge as bridge  # noqa: E402


# ---------------------------------------------------------------------------
# Encoders — mirror the TS side byte-for-byte
# ---------------------------------------------------------------------------

def _align(buf: bytearray, n: int) -> None:
    while len(buf) % n:
        buf.append(0)


def _encap(body: bytes) -> bytes:
    # OMG CDR encapsulation: {reserved, endian, options_hi, options_lo}
    return bytes([0x00, 0x01, 0x00, 0x00]) + body


def _write_string(buf: bytearray, s: str) -> None:
    _align(buf, 4)
    data = s.encode("utf-8") + b"\x00"
    buf += struct.pack("<I", len(data))
    buf += data


def _write_header(buf: bytearray, sec: int, nsec: int, frame_id: str) -> None:
    _align(buf, 4)
    buf += struct.pack("<iI", sec, nsec)
    _write_string(buf, frame_id)


def encode_joy(axes: list[float], buttons: list[int],
               sec: int = 0, nsec: int = 0, frame_id: str = "") -> bytes:
    body = bytearray()
    _write_header(body, sec, nsec, frame_id)
    _align(body, 4)
    body += struct.pack("<I", len(axes))
    for a in axes:
        _align(body, 4)
        body += struct.pack("<f", a)
    _align(body, 4)
    body += struct.pack("<I", len(buttons))
    for b in buttons:
        _align(body, 4)
        body += struct.pack("<i", b)
    return _encap(bytes(body))


def encode_posestamped(pos: tuple[float, float, float],
                       quat: tuple[float, float, float, float],
                       sec: int = 0, nsec: int = 0,
                       frame_id: str = "xr_origin") -> bytes:
    body = bytearray()
    _write_header(body, sec, nsec, frame_id)
    _align(body, 8)
    body += struct.pack("<ddd", *pos)
    _align(body, 8)
    body += struct.pack("<dddd", *quat)
    return _encap(bytes(body))


def ros_envelope(topic: str, mtype: str, payload: bytes) -> bytes:
    # Big-endian lengths (matches encodeRosEnvelope in adamo-ts/packages/ros/src/index.ts)
    tb = topic.encode(); yb = mtype.encode()
    return struct.pack(">I", len(tb)) + tb + struct.pack(">I", len(yb)) + yb + payload


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = ""):
    if cond:
        print(f"  ✓ {name}")
    else:
        print(f"  ✗ {name}  {detail}")
        FAILED.append(name)


# 1) Wire-format round-trip
print("\n[1] CDR + envelope round-trip")

joy_bytes = encode_joy(axes=[0.25, -0.5, 0.75, 0.1, 0.0, 0.0],
                      buttons=[0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0])
joy_env = ros_envelope("/controller/right/joy", "sensor_msgs/msg/Joy", joy_bytes)
env = bridge._decode_envelope(joy_env)
check("envelope topic", env and env[0] == "/controller/right/joy")
check("envelope type",  env and env[1] == "sensor_msgs/msg/Joy")

joy = bridge._decode_joy_cdr(env[2])
check("joy axes",     np.allclose(joy.axes, [0.25, -0.5, 0.75, 0.1, 0.0, 0.0], atol=1e-6))
check("joy buttons",  joy.buttons == [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0])

pose_bytes = encode_posestamped(pos=(0.1, 1.2, -0.3),
                                quat=(0.0, 0.0, 0.0, 1.0))
pose_env = ros_envelope("/controller/left", "geometry_msgs/msg/PoseStamped", pose_bytes)
env = bridge._decode_envelope(pose_env)
pose = bridge._decode_posestamped_cdr(env[2])
check("pose position",    np.allclose(pose.pos, [0.1, 1.2, -0.3]))
check("pose quat",        np.allclose(pose.quat, [0.0, 0.0, 0.0, 1.0]))

# JSON JoystickCommand path (JoypadManager with serializationFormat="json")
import json
json_msg = json.dumps({"type": "JoystickCommand", "axes": [0.1, 0.2, 0.3, 0.4], "buttons": [0, 1]})
joy_json = bridge._decode_joy_json(json_msg.encode())
check("json joy decode", joy_json and joy_json.axes == [0.1, 0.2, 0.3, 0.4])


# 2) Joy → LocoClient.Move routing
print("\n[2] LocoDriver: Joy axes → LocoClient.Move")

loco = bridge.LocoDriver(speed_scale=0.3, timeout_s=5.0)
try:
    # Left stick forward (axes[1] = -1.0 means stick pushed forward in WebXR)
    loco.on_joy(bridge.Joy(axes=[0.0, -1.0, 0.0, 0.0], buttons=[0] * 12))
    check("Move called once", len(loco._wrapper.move_calls) == 1)
    vx, vy, vyaw = loco._wrapper.move_calls[-1]
    check("vx forward (positive)", vx > 0.0 and abs(vx - 0.3) < 1e-6,
          f"got vx={vx}")
    check("vy zero",  abs(vy) < 1e-6,   f"got vy={vy}")
    check("vyaw zero", abs(vyaw) < 1e-6, f"got vyaw={vyaw}")

    # Both thumbsticks pressed → Damp
    loco.on_joy(bridge.Joy(axes=[0.0, 0.0, 0.0, 0.0],
                           buttons=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]))
    check("Damp triggered", loco._wrapper.damp_calls == 1)
finally:
    loco.close()


# 3) Bridge on_xr dispatch (envelope → LocoDriver / PoseBuffer)
print("\n[3] on_xr dispatch by inner topic")

# Rebuild the handler the way main() does, but with our fakes
pose_buf = bridge.PoseBuffer()
loco2 = bridge.LocoDriver(speed_scale=0.3, timeout_s=5.0)
try:
    LOCOMOTION_SOURCE = "right"   # drive walking from the right XR controller

    def on_xr(payload: bytes) -> None:
        env = bridge._decode_envelope(payload)
        if env is None: return
        inner, mtype, body = env
        if mtype == "geometry_msgs/msg/PoseStamped":
            p = bridge._decode_posestamped_cdr(body)
            if p is None: return
            now = time.monotonic()
            with pose_buf.lock:
                if inner.endswith("/left"):  pose_buf.left = p;  pose_buf.left_t = now
                elif inner.endswith("/right"): pose_buf.right = p; pose_buf.right_t = now
                elif "head" in inner: pose_buf.head = p
        elif mtype == "sensor_msgs/msg/Joy":
            j = bridge._decode_joy_cdr(body)
            if j and LOCOMOTION_SOURCE == "right" and "right" in inner:
                loco2.on_joy(j)

    # Feed a left wrist pose
    on_xr(ros_envelope("/controller/left",
                       "geometry_msgs/msg/PoseStamped",
                       encode_posestamped((0.2, 1.0, -0.4), (0, 0, 0, 1))))
    # Right wrist pose
    on_xr(ros_envelope("/controller/right",
                       "geometry_msgs/msg/PoseStamped",
                       encode_posestamped((-0.2, 1.0, -0.4), (0, 0, 0, 1))))
    # Head pose (should land in pose_buf.head, no SDK call)
    on_xr(ros_envelope("/head_pose",
                       "geometry_msgs/msg/PoseStamped",
                       encode_posestamped((0.0, 1.7, 0.0), (0, 0, 0, 1))))
    # Right-controller Joy → locomotion (left stick Y forward = -1.0)
    on_xr(ros_envelope("/controller/right/joy",
                       "sensor_msgs/msg/Joy",
                       encode_joy(axes=[0.0, -1.0, 0.0, 0.0], buttons=[0] * 12)))

    check("pose_buf.left set",  pose_buf.left is not None)
    check("pose_buf.right set", pose_buf.right is not None)
    check("pose_buf.head set",  pose_buf.head is not None)
    check("loco got Move from right-controller Joy",
          len(loco2._wrapper.move_calls) == 1)
finally:
    loco2.close()


# 4) ArmDriver: pose_buf → IK → ctrl_dual_arm
print("\n[4] ArmDriver: wrist poses → arm_ik.solve_ik → arm_ctrl.ctrl_dual_arm")

pose_buf2 = bridge.PoseBuffer()
pose_buf2.left  = bridge.Pose(pos=np.array([0.2, 1.0, -0.4]), quat=np.array([0, 0, 0, 1.0]))
pose_buf2.right = bridge.Pose(pos=np.array([-0.2, 1.0, -0.4]), quat=np.array([0, 0, 0, 1.0]))
pose_buf2.left_t = pose_buf2.right_t = time.monotonic()

arm = bridge.ArmDriver(
    arm="G1_29", motion=False, sim=False,
    frequency=30.0, pose_max_age_s=1.0,
    origin_offset=np.array([0.25, 0.0, 0.1]),
    buf=pose_buf2,
)
try:
    time.sleep(0.15)  # several tick periods at 30 Hz
    check("ik.solve_ik called",       len(arm.ik.solve_calls) > 0)
    check("ctrl.ctrl_dual_arm called", len(arm.ctrl.ctrl_calls) > 0)

    # Verify the input transforms to IK have the expected shape + frame mapping.
    # WebXR (0, 1.0, -0.4) with our remap (robot_x = -xr_z, robot_y = -xr_x, robot_z = xr_y)
    # + origin offset (0.25, 0, 0.1) should give right wrist at:
    #   x = -(-0.4) + 0.25 = 0.65
    #   y = -(-0.2)        = 0.20
    #   z =   1.0   + 0.1  = 1.10
    L, R = arm.ik.solve_calls[-1]
    check("IK received 4x4 transforms",   L.shape == (4, 4) and R.shape == (4, 4))
    check("right wrist x mapped", abs(R[0, 3] - 0.65) < 1e-6, f"got R[0,3]={R[0,3]}")
    check("right wrist y mapped", abs(R[1, 3] - 0.20) < 1e-6, f"got R[1,3]={R[1,3]}")
    check("right wrist z mapped", abs(R[2, 3] - 1.10) < 1e-6, f"got R[2,3]={R[2,3]}")
finally:
    arm.close()


# ---------------------------------------------------------------------------
# 5) Full loop through the Adamo routers (real Zenoh transport)
# ---------------------------------------------------------------------------
#
# Publishes on  adamo/{org}/{robot}/control/cdr/xr_tracking  using the real
# adamo.connect() session and verifies the bridge's on_xr subscription
# receives and decodes the envelopes identically. Skipped if adamo isn't
# importable.

print("\n[5] Live round-trip via Adamo routers")

try:
    import adamo as _adamo
    _adamo_ok = True
except Exception as e:
    _adamo_ok = False
    print(f"  (skipped — adamo not importable: {e})")

if _adamo_ok:
    import queue
    import uuid

    api_key = os.environ.get("ADAMO_API_KEY") or "ak_2M3T7rqPYGubJO2gBsxRoWswKn83z0L6"
    protocol = os.environ.get("ADAMO_PROTOCOL", "udp")
    robot_name = f"bridge-test-{uuid.uuid4().hex[:8]}"
    key = f"{robot_name}/control/cdr/xr_tracking"

    print(f"  connecting protocol={protocol} robot={robot_name} …")
    sub_sess = _adamo.connect(api_key=api_key, protocol=protocol)
    pub_sess = _adamo.connect(api_key=api_key, protocol=protocol)

    received: "queue.Queue[bytes]" = queue.Queue()
    sub_handle = sub_sess.subscribe(key, callback=lambda s: received.put(bytes(s.payload)))
    time.sleep(0.8)  # let the subscription propagate to the router

    pub = pub_sess.publisher(key, express=True)

    # Pose: left wrist at (0.42, 1.23, -0.77), unit quaternion
    pose_env = ros_envelope(
        "/controller/left",
        "geometry_msgs/msg/PoseStamped",
        encode_posestamped((0.42, 1.23, -0.77), (0.0, 0.0, 0.0, 1.0)),
    )
    joy_env = ros_envelope(
        "/controller/right/joy",
        "sensor_msgs/msg/Joy",
        encode_joy(axes=[0.0, -0.5, 0.25, 0.0], buttons=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    )

    pub.put(pose_env)
    pub.put(joy_env)

    collected: list[bytes] = []
    deadline = time.monotonic() + 6.0
    while time.monotonic() < deadline and len(collected) < 2:
        try:
            collected.append(received.get(timeout=max(0.1, deadline - time.monotonic())))
        except queue.Empty:
            break

    check("received both envelopes via router", len(collected) >= 2,
          f"got {len(collected)}/2 in 6s")

    if len(collected) >= 2:
        # Match them up by type after decoding (order isn't guaranteed).
        decoded = []
        for raw in collected:
            env = bridge._decode_envelope(raw)
            if env is None:
                continue
            inner, mtype, body = env
            if mtype == "geometry_msgs/msg/PoseStamped":
                decoded.append(("pose", inner, bridge._decode_posestamped_cdr(body)))
            elif mtype == "sensor_msgs/msg/Joy":
                decoded.append(("joy", inner, bridge._decode_joy_cdr(body)))

        pose_hit = next((d for d in decoded if d[0] == "pose"), None)
        joy_hit  = next((d for d in decoded if d[0] == "joy"),  None)

        check("pose envelope topic",     pose_hit and pose_hit[1] == "/controller/left")
        check("pose position survived",  pose_hit and np.allclose(pose_hit[2].pos,  [0.42, 1.23, -0.77]))
        check("pose quat survived",      pose_hit and np.allclose(pose_hit[2].quat, [0.0, 0.0, 0.0, 1.0]))
        check("joy envelope topic",      joy_hit  and joy_hit[1] == "/controller/right/joy")
        check("joy axes survived",       joy_hit  and np.allclose(joy_hit[2].axes, [0.0, -0.5, 0.25, 0.0], atol=1e-6))
        check("joy buttons survived",    joy_hit  and joy_hit[2].buttons == [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    try: pub.close()
    except Exception: pass
    try: sub_handle.close()
    except Exception: pass
    try: pub_sess.close(); sub_sess.close()
    except Exception: pass


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print()
if FAILED:
    print(f"FAILED ({len(FAILED)}): {FAILED}")
    sys.exit(1)
print("OK — all bridge paths verified end-to-end")
