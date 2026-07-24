#!/usr/bin/env python3
"""
visualize_real_robot_ik.py — Live IK control of the S2full robot via PyBullet.

Combines three scripts:
  • visualize_robot.py     — init_pos.json loading and URDF setup
  • visualize_robot_ik.py  — EE-target sliders and IK solver with coloured box
  • visualize_real_robot.py — real-robot mirroring via ROS 2 motor-status topics

Workflow each frame:
  1. PyBullet joints are set to the real robot's current measured positions
  2. IK is solved from that configuration to the slider-defined EE target
  3. If sending is enabled (press X to toggle), the IK-solved angles are
     published as position commands to the real robot's right arm + waist
  4. As the robot moves the status callbacks drive PyBullet — the model
     always reflects actual hardware state

Keyboard (click PyBullet window first):
    x   — toggle continuous IK command sending ON / OFF  (starts OFF — safe)
    1   — open right hand
    2   — close right hand
    3   — open left hand
    4   — close left hand
    r   — save current pose to saved_positions.csv
    7   — list, select and execute a saved position from saved_positions.csv
    t   — print real joint values and live EE pose
    9   — record current body pose to trajectory buffer
    0   — save trajectory buffer to trajectories.csv (prompts for name)
    8   — list, select and execute a saved trajectory on the robot
    q   — quit

⚠  WARNING — Sending IK commands (X key) moves physical hardware.
   Ensure the robot is safely supported before enabling.
"""

import csv
import datetime
import json
import math
import os
import threading
import time

import numpy as np
import pybullet as p
import pybullet_data
import rclpy
from rclpy.node import Node
from bodyctrl_msgs.msg import CmdSetMotorPosition, SetMotorPosition, MotorStatusMsg
from sensor_msgs.msg import JointState


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROBOT_KEY = 'S2full'

POSITION_THRESHOLD    = 0.05   # metres — IK accuracy colouring
ORIENTATION_THRESHOLD = 0.10   # radians

# IK commands are sent only for these motor IDs (right arm + waist).
# Legs are intentionally excluded for safety.
IK_MOTOR_IDS   = set(range(21, 28)) | {31}   # right arm + waist
LEFT_IK_MOTOR_IDS      = set(range(11, 18))   # left arm
LEFT_LEG_IK_MOTOR_IDS  = set(range(51, 57))   # left leg
RIGHT_LEG_IK_MOTOR_IDS = set(range(61, 67))   # right leg
HEAD_IK_MOTOR_IDS      = {1, 2, 3}            # head

# motor_id → URDF joint name
MOTOR_TO_URDF = {
    # Head
    1: 'head_roll_joint',
    2: 'head_pitch_joint',
    3: 'head_yaw_joint',
    # Left arm
    11: 'shoulder_pitch_l_joint', 12: 'shoulder_roll_l_joint',
    13: 'shoulder_yaw_l_joint',   14: 'elbow_pitch_l_joint',
    15: 'wrist_yaw_l_joint',      16: 'wrist_pitch_l_joint',  17: 'wrist_roll_l_joint',
    # Right arm
    21: 'shoulder_pitch_r_rjoint', 22: 'shoulder_roll_r_rjoint',
    23: 'shoulder_yaw_r_rjoint',   24: 'elbow_pitch_r_rjoint',
    25: 'wrist_yaw_r_rjoint',      26: 'wrist_pitch_r_rjoint', 27: 'wrist_roll_r_rjoint',
    # Waist
    31: 'body_yaw_rjoint',
    # Left leg
    51: 'hip_roll_l_joint',    52: 'hip_pitch_l_joint',   53: 'hip_yaw_l_joint',
    54: 'knee_pitch_l_joint',  55: 'ankle_pitch_l_joint', 56: 'ankle_roll_l_joint',
    # Right leg
    61: 'hip_roll_r_joint',    62: 'hip_pitch_r_joint',   63: 'hip_yaw_r_joint',
    64: 'knee_pitch_r_joint',  65: 'ankle_pitch_r_joint', 66: 'ankle_roll_r_joint',
}

URDF_TO_MOTOR = {v: k for k, v in MOTOR_TO_URDF.items()}

# Publishing route per motor group
def _motor_route(motor_id: int):
    """Return (topic_key, profile_spd, current_A) for motor ID."""
    if 11 <= motor_id <= 27:
        return 'arm',   0.5, 8.0
    if 51 <= motor_id <= 66:
        return 'leg',   0.5, 8.0
    if motor_id == 31:
        return 'waist', 0.2, 8.0
    if 1 <= motor_id <= 3:
        return 'head',  0.2, 2.0
    return None, 0.5, 8.0

# URDF → GUI position array mapping (for save-to-CSV feature)
URDF_TO_GUI = {
    'hip_roll_l_joint':    ('left_leg_pos',  0),
    'hip_pitch_l_joint':   ('left_leg_pos',  1),
    'hip_yaw_l_joint':     ('left_leg_pos',  2),
    'knee_pitch_l_joint':  ('left_leg_pos',  3),
    'ankle_pitch_l_joint': ('left_leg_pos',  4),
    'ankle_roll_l_joint':  ('left_leg_pos',  5),
    'hip_roll_r_joint':    ('right_leg_pos', 0),
    'hip_pitch_r_joint':   ('right_leg_pos', 1),
    'hip_yaw_r_joint':     ('right_leg_pos', 2),
    'knee_pitch_r_joint':  ('right_leg_pos', 3),
    'ankle_pitch_r_joint': ('right_leg_pos', 4),
    'ankle_roll_r_joint':  ('right_leg_pos', 5),
    'body_yaw_rjoint':          ('waist_pos',     0),
    'head_roll_joint':          ('head_pos',      0),
    'head_pitch_joint':         ('head_pos',      1),
    'head_yaw_joint':           ('head_pos',      2),
    'shoulder_pitch_l_joint':   ('left_arm_pos',  0),
    'shoulder_roll_l_joint':    ('left_arm_pos',  1),
    'shoulder_yaw_l_joint':     ('left_arm_pos',  2),
    'elbow_pitch_l_joint':      ('left_arm_pos',  3),
    'wrist_yaw_l_joint':        ('left_arm_pos',  4),
    'wrist_pitch_l_joint':      ('left_arm_pos',  5),
    'wrist_roll_l_joint':       ('left_arm_pos',  6),
    'shoulder_pitch_r_rjoint':  ('right_arm_pos', 0),
    'shoulder_roll_r_rjoint':   ('right_arm_pos', 1),
    'shoulder_yaw_r_rjoint':    ('right_arm_pos', 2),
    'elbow_pitch_r_rjoint':     ('right_arm_pos', 3),
    'wrist_yaw_r_rjoint':       ('right_arm_pos', 4),
    'wrist_pitch_r_rjoint':     ('right_arm_pos', 5),
    'wrist_roll_r_rjoint':      ('right_arm_pos', 6),
}


# ---------------------------------------------------------------------------
# ROS 2 node
# ---------------------------------------------------------------------------

class RobotMirrorNode(Node):
    """Subscribes to motor-status topics; publishes position commands."""

    def __init__(self):
        super().__init__('visualize_real_robot_ik')
        self._lock = threading.Lock()
        self._motor_pos: dict[int, float] = {}   # motor_id → rad

        for topic in ('/arm/status', '/leg/status', '/waist/status', '/head/status'):
            self.create_subscription(MotorStatusMsg, topic, self._status_cb, 10)

        self._pubs = {
            'arm':   self.create_publisher(CmdSetMotorPosition, '/arm/cmd_pos',   10),
            'leg':   self.create_publisher(CmdSetMotorPosition, '/leg/cmd_pos',   10),
            'waist': self.create_publisher(CmdSetMotorPosition, '/waist/cmd_pos', 10),
            'head':  self.create_publisher(CmdSetMotorPosition, '/head/cmd_pos',  10),
        }
        self._hand_right_pub = self.create_publisher(
            JointState, '/inspire_hand/ctrl/right_hand', 10
        )
        self._hand_left_pub = self.create_publisher(
            JointState, '/inspire_hand/ctrl/left_hand', 10
        )

    def _status_cb(self, msg: MotorStatusMsg):
        with self._lock:
            for st in msg.status:
                self._motor_pos[st.name] = st.pos

    def get_positions(self) -> dict[int, float]:
        with self._lock:
            return dict(self._motor_pos)

    def send_positions(self, commands: dict[int, float]):
        """Send a {motor_id: rad} dict to the robot. Groups by topic automatically."""
        groups: dict[str, list] = {}
        for motor_id, pos_rad in commands.items():
            topic_key, spd, cur = _motor_route(motor_id)
            if topic_key is None:
                continue
            c = SetMotorPosition()
            c.name = motor_id
            c.pos  = float(pos_rad)
            c.spd  = spd
            c.cur  = cur
            groups.setdefault(topic_key, []).append(c)

        for topic_key, cmds in groups.items():
            msg = CmdSetMotorPosition()
            msg.cmds = cmds
            self._pubs[topic_key].publish(msg)

    def send_gripper(self, motor_id_values: list[tuple[int, float]]):
        """Send gripper open/close commands (uses 'arm' or 'hand' publisher)."""
        msg = CmdSetMotorPosition()
        msg.cmds = []
        for motor_id, val in motor_id_values:
            c = SetMotorPosition()
            c.name = motor_id
            c.pos  = float(val)
            c.spd  = 0.5
            c.cur  = 2.0
            msg.cmds.append(c)
        if msg.cmds:
            self._pubs['arm'].publish(msg)

    def send_right_hand(self, positions: list[float], velocities: list[float] | None = None):
        """Publish a JointState command to the right Inspire hand."""
        if velocities is None:
            velocities = [1.0] * 6
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name     = ['1', '2', '3', '4', '5', '6']
        msg.position = list(positions)
        msg.velocity = list(velocities)
        msg.effort   = [1.0] * 6
        self._hand_right_pub.publish(msg)

    def send_left_hand(self, positions: list[float], velocities: list[float] | None = None):
        """Publish a JointState command to the left Inspire hand."""
        if velocities is None:
            velocities = [1.0] * 6
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name     = ['1', '2', '3', '4', '5', '6']
        msg.position = list(positions)
        msg.velocity = list(velocities)
        msg.effort   = [1.0] * 6
        self._hand_left_pub.publish(msg)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_look_at_quat(eye_pos, target_pos):
    """Return a quaternion orienting eye_pos to look at target_pos.
    Yaw and pitch derived from direction vector; roll kept zero.
    Mirrors grasper.py Grasper.get_look_at_quaternion().
    """
    direction = np.asarray(target_pos) - np.asarray(eye_pos)
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return [0.0, 0.0, 0.0, 1.0]
    yaw    = math.atan2(direction[1], direction[0])
    h_dist = math.sqrt(direction[0] ** 2 + direction[1] ** 2)
    pitch  = math.atan2(-direction[2], h_dist)
    return list(p.getQuaternionFromEuler([0.0, pitch, yaw]))


def _prepare_urdf_for_pybullet(src_urdf: str, examples_dir: str) -> str:
    with open(src_urdf, 'r', encoding='utf-8') as f:
        text = f.read()
    text = text.replace('package://ubtech/', '')
    out = os.path.join(examples_dir, f'_{os.path.splitext(os.path.basename(src_urdf))[0]}_pybullet.urdf')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(text)
    return out


def _load_init_pos(path: str) -> dict:
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'r_dict': {}, 'g_dict': {}}


def _save_to_gui_positions(robot_id: int, joint_idxs: list[int],
                           joint_names: list[str], positions_file: str) -> str:
    """Read current joint states (rad) and append a named position to saved_positions.csv."""
    limb_sizes = {
        'left_leg_pos': 6, 'right_leg_pos': 6,
        'left_arm_pos': 7, 'right_arm_pos': 7,
        'waist_pos': 1, 'head_pos': 3,
    }
    positions = {k: [0.0] * n for k, n in limb_sizes.items()}
    for ji, jname in zip(joint_idxs, joint_names):
        if jname not in URDF_TO_GUI:
            continue
        limb_key, idx = URDF_TO_GUI[jname]
        positions[limb_key][idx] = round(math.degrees(p.getJointState(robot_id, ji)[0]), 2)

    payload = {
        'leg_mode': 'Position', 'arm_mode': 'Position',
        'left_leg_pos': positions['left_leg_pos'],
        'right_leg_pos': positions['right_leg_pos'],
        'left_arm_pos': positions['left_arm_pos'],
        'right_arm_pos': positions['right_arm_pos'],
        'leg_profile_speed': 0.5,  'leg_position_current': 8.0, 'leg_speed_current': 8.0,
        'arm_profile_speed': 0.5,  'arm_position_current': 8.0, 'arm_speed_current': 8.0,
        'waist_pos': positions['waist_pos'], 'waist_speed': [0.2],
        'head_pos': positions['head_pos'],   'head_speed': [0.2],
        'left_finger_pos': [0.0] * 6, 'right_finger_pos': [0.0] * 6,
        'left_finger_vel': [1.0] * 6, 'right_finger_vel': [1.0] * 6,
        'hand_effort': [1.0],
    }
    name = datetime.datetime.now().strftime('ik_real_%Y%m%d_%H%M%S')
    saved = {}
    if os.path.exists(positions_file):
        try:
            with open(positions_file, 'r', newline='', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    n, pl = (row.get('name') or '').strip(), row.get('payload')
                    if n and pl:
                        try:
                            saved[n] = json.loads(pl)
                        except json.JSONDecodeError:
                            pass
        except OSError:
            pass
    saved[name] = payload
    with open(positions_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'payload'])
        writer.writeheader()
        for n in sorted(saved):
            writer.writerow({'name': n, 'payload': json.dumps(saved[n])})
    return name


def _get_current_pose_payload(robot_id: int, joint_idxs: list[int], joint_names: list[str]) -> dict:
    """Return a payload dict of current joint angles (degrees) from PyBullet."""
    limb_sizes = {
        'left_leg_pos': 6, 'right_leg_pos': 6,
        'left_arm_pos': 7, 'right_arm_pos': 7,
        'waist_pos': 1, 'head_pos': 3,
    }
    positions = {k: [0.0] * n for k, n in limb_sizes.items()}
    for ji, jname in zip(joint_idxs, joint_names):
        if jname not in URDF_TO_GUI:
            continue
        limb_key, idx = URDF_TO_GUI[jname]
        positions[limb_key][idx] = round(math.degrees(p.getJointState(robot_id, ji)[0]), 2)
    return {
        'leg_mode': 'Position', 'arm_mode': 'Position',
        'left_leg_pos':  positions['left_leg_pos'],
        'right_leg_pos': positions['right_leg_pos'],
        'left_arm_pos':  positions['left_arm_pos'],
        'right_arm_pos': positions['right_arm_pos'],
        'leg_profile_speed': 0.5,  'leg_position_current': 8.0, 'leg_speed_current': 8.0,
        'arm_profile_speed': 0.5,  'arm_position_current': 8.0, 'arm_speed_current': 8.0,
        'waist_pos': positions['waist_pos'], 'waist_speed': [0.2],
        'head_pos':  positions['head_pos'],  'head_speed':  [0.2],
        'left_finger_pos':  [0.0] * 6, 'right_finger_pos': [0.0] * 6,
        'left_finger_vel':  [1.0] * 6, 'right_finger_vel': [1.0] * 6,
        'hand_effort': [1.0],
    }


def _save_trajectory(traj_name: str, waypoints: list, traj_file: str):
    """Append / overwrite a named trajectory in trajectories.csv."""
    trajectories: dict = {}
    if os.path.exists(traj_file):
        try:
            with open(traj_file, 'r', newline='', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    n = (row.get('name') or '').strip()
                    wps = row.get('waypoints')
                    if n and wps:
                        try:
                            trajectories[n] = json.loads(wps)
                        except json.JSONDecodeError:
                            pass
        except OSError:
            pass
    trajectories[traj_name] = waypoints
    with open(traj_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'waypoints'])
        writer.writeheader()
        for n, wps in trajectories.items():
            writer.writerow({'name': n, 'waypoints': json.dumps(wps)})


def _load_saved_positions(positions_file: str) -> dict:
    """Load all saved positions from saved_positions.csv; returns {name: payload_dict}."""
    positions: dict = {}
    if not os.path.exists(positions_file):
        return positions
    try:
        with open(positions_file, 'r', newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                n = (row.get('name') or '').strip()
                pl = row.get('payload')
                if n and pl:
                    try:
                        positions[n] = json.loads(pl)
                    except json.JSONDecodeError:
                        pass
    except OSError:
        pass
    return positions


def _execute_saved_position(payload: dict, ros_node):
    """Send a single saved-position payload (degrees) to the robot."""
    _LIMB_TO_MOTORS = {
        'head_pos':      [1,  2,  3],
        'left_arm_pos':  [11, 12, 13, 14, 15, 16, 17],
        'right_arm_pos': [21, 22, 23, 24, 25, 26, 27],
        'waist_pos':     [31],
        'left_leg_pos':  [51, 52, 53, 54, 55, 56],
        'right_leg_pos': [61, 62, 63, 64, 65, 66],
    }
    commands: dict[int, float] = {}
    for limb_key, motor_ids in _LIMB_TO_MOTORS.items():
        angles = payload.get(limb_key, [])
        for idx, motor_id in enumerate(motor_ids):
            if idx < len(angles):
                commands[motor_id] = math.radians(angles[idx])
    ros_node.send_positions(commands)
    print('[7] Position sent to robot.')


def _load_trajectories(traj_file: str) -> dict:
    """Load all trajectories from trajectories.csv; returns {name: [waypoints]}."""
    trajectories: dict = {}
    if not os.path.exists(traj_file):
        return trajectories
    try:
        with open(traj_file, 'r', newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                n = (row.get('name') or '').strip()
                wps = row.get('waypoints')
                if n and wps:
                    try:
                        trajectories[n] = json.loads(wps)
                    except json.JSONDecodeError:
                        pass
    except OSError:
        pass
    return trajectories


def _execute_trajectory(waypoints: list, ros_node, step_delay: float = 2.0,
                        interp_hz: float = 20.0, traj_active: dict | None = None):
    """Send trajectory waypoints with continuous interpolated commands at interp_hz.

    Instead of one burst per waypoint, positions are streamed continuously so
    the motor controllers always have a fresh setpoint — matching the behaviour
    of the IK-sending loop.  IK sending is suppressed via traj_active while
    this function runs.
    """
    _LIMB_TO_MOTORS = {
        'head_pos':      [1,  2,  3],
        'left_arm_pos':  [11, 12, 13, 14, 15, 16, 17],
        'right_arm_pos': [21, 22, 23, 24, 25, 26, 27],
        'waist_pos':     [31],
        'left_leg_pos':  [51, 52, 53, 54, 55, 56],
        'right_leg_pos': [61, 62, 63, 64, 65, 66],
    }

    def _interp_commands(wp_a: dict, wp_b: dict, alpha: float) -> dict[int, float]:
        cmds: dict[int, float] = {}
        for limb_key, motor_ids in _LIMB_TO_MOTORS.items():
            a_ang = wp_a.get(limb_key, [])
            b_ang = wp_b.get(limb_key, [])
            for idx, motor_id in enumerate(motor_ids):
                a = float(a_ang[idx]) if idx < len(a_ang) else 0.0
                b = float(b_ang[idx]) if idx < len(b_ang) else 0.0
                cmds[motor_id] = math.radians(a + alpha * (b - a))
        return cmds

    if traj_active is not None:
        traj_active['value'] = True
    try:
        n_interp  = max(1, int(step_delay * interp_hz))
        sleep_dt  = 1.0 / interp_hz
        print(f'[traj] Executing {len(waypoints)} waypoints '
              f'({step_delay}s × {interp_hz:.0f}Hz per step) …')

        # Hold first waypoint position
        ros_node.send_positions(_interp_commands(waypoints[0], waypoints[0], 1.0))
        print(f'[traj] Waypoint 1/{len(waypoints)}')

        # Stream interpolated commands between consecutive waypoints
        for i in range(1, len(waypoints)):
            for step in range(1, n_interp + 1):
                alpha = step / n_interp
                ros_node.send_positions(
                    _interp_commands(waypoints[i - 1], waypoints[i], alpha)
                )
                time.sleep(sleep_dt)
            print(f'[traj] Waypoint {i + 1}/{len(waypoints)}')

        print('[traj] Done.')
    finally:
        if traj_active is not None:
            traj_active['value'] = False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    examples_dir = os.path.dirname(os.path.abspath(__file__))
    src_urdf     = os.path.join(examples_dir, 'envs', 'robots', 'ubtech', 'S2full.urdf')

    # ── Load init_pos.json ────────────────────────────────────────────────────
    init_pos_path = os.path.join(examples_dir, 'init_pos.json')
    init_pos = _load_init_pos(init_pos_path)
    rd = init_pos.get('r_dict', {})
    gd = init_pos.get('g_dict', {})
    robot_info        = rd.get(ROBOT_KEY, {})
    default_joint_ori = robot_info.get('default_joint_ori', [])
    gripper_open      = gd.get(ROBOT_KEY, {}).get('open',  [])
    gripper_close     = gd.get(ROBOT_KEY, {}).get('close', [])
    print(f'init_pos.json: body={len(default_joint_ori)} joints, '
          f'gripper open={len(gripper_open)}, close={len(gripper_close)}')

    # ── ROS 2 ─────────────────────────────────────────────────────────────────
    rclpy.init()
    ros_node = RobotMirrorNode()
    spin_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    spin_thread.start()

    # Wait up to 3 s for first status messages
    print('Waiting for robot status …', end='', flush=True)
    deadline = time.time() + 3.0
    while time.time() < deadline and not ros_node.get_positions():
        time.sleep(0.05)
    initial_pos = ros_node.get_positions()  # motor_id → rad
    if initial_pos:
        print(f' received {len(initial_pos)} motor positions.')
    else:
        print(' timed out — starting at init_pos.json values.')

    # ── PyBullet ──────────────────────────────────────────────────────────────
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF('plane.urdf', basePosition=[0, 0, -1.02])

    #urdf = _prepare_urdf_for_pybullet(src_urdf, examples_dir)
    urdf = src_urdf
    try:
        robot_id = p.loadURDF(
            urdf,
            useFixedBase=True,
            basePosition=[0.0, 0.0, 0.0],
            baseOrientation=[0, 0, 0, 1],
            flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
        )
        print(f'Loaded URDF: {src_urdf}')
    except Exception as ex:
        print(f'Error loading URDF: {ex}')
        ros_node.destroy_node(); rclpy.shutdown()
        return

    num_joints = p.getNumJoints(robot_id)

    # Build name → joint-index map
    name_to_joint: dict[str, int] = {}
    for ji in range(num_joints):
        info = p.getJointInfo(robot_id, ji)
        name_to_joint[info[1].decode('utf-8')] = ji

    # Separate gripper joints and body joints
    non_fixed_idxs:  list[int] = []
    non_fixed_names: list[str] = []
    gjoint_idxs:     list[int] = []

    for ji in range(num_joints):
        info = p.getJointInfo(robot_id, ji)
        if info[2] == p.JOINT_FIXED:
            continue
        jname = info[1].decode('utf-8')
        non_fixed_idxs.append(ji)
        non_fixed_names.append(jname)
        if 'gjoint' in jname:
            gjoint_idxs.append(ji)

    gjoint_set = set(gjoint_idxs)

    # Initialise joints: prefer live robot positions, fall back to init_pos.json
    for i, (ji, jname) in enumerate(zip(non_fixed_idxs, non_fixed_names)):
        motor_id = URDF_TO_MOTOR.get(jname)
        if motor_id is not None and motor_id in initial_pos:
            p.resetJointState(robot_id, ji, initial_pos[motor_id])
        elif i < len(default_joint_ori):
            p.resetJointState(robot_id, ji, default_joint_ori[i])

    # Override gripper with stored open values
    for i, ji in enumerate(gjoint_idxs):
        if i < len(gripper_open):
            p.resetJointState(robot_id, ji, gripper_open[i])

    # Find end-effector link
    ee_index = -1
    for ji in range(num_joints):
        if p.getJointInfo(robot_id, ji)[12].decode('utf-8') == 'endeffector':
            ee_index = ji
            print(f'End effector at joint index {ee_index}')
            break
    if ee_index == -1:
        # Fall back to last non-gripper joint
        ee_index = [ji for ji in non_fixed_idxs if ji not in gjoint_set][-1]
        print(f'Warning: "endeffector" link not found; using joint {ee_index}')

    # IK joint list — non-fixed, non-gripper, in joint-index order
    ik_joint_idxs  = [ji for ji in non_fixed_idxs if ji not in gjoint_set]
    ik_joint_names = [non_fixed_names[non_fixed_idxs.index(ji)] for ji in ik_joint_idxs]

    # Extract joint limits for constrained IK (used by head look-at IK)
    joints_lower: list[float] = []
    joints_upper: list[float] = []
    joints_ranges: list[float] = []
    joints_rest: list[float] = []
    for ji in ik_joint_idxs:
        info = p.getJointInfo(robot_id, ji)
        lower, upper = float(info[8]), float(info[9])
        if lower >= upper:          # unlimited joint — use ±π fallback
            lower, upper = -math.pi, math.pi
        joints_lower.append(lower)
        joints_upper.append(upper)
        joints_ranges.append(upper - lower)
        joints_rest.append(0.0)

    # Find left end-effector link (named 'endeffectol' in the URDF)
    ee_index_l = -1
    for ji in range(num_joints):
        if p.getJointInfo(robot_id, ji)[12].decode('utf-8') == 'endeffectol':
            ee_index_l = ji
            print(f'Left end effector at joint index {ee_index_l}')
            break
    if ee_index_l == -1:
        print('Warning: "endeffectol" link not found; left IK disabled')

    # Find left leg end-effector link
    ee_index_ll = -1
    for ji in range(num_joints):
        if p.getJointInfo(robot_id, ji)[12].decode('utf-8') == 'endeffectoll':
            ee_index_ll = ji
            print(f'Left leg end effector at joint index {ee_index_ll}')
            break
    if ee_index_ll == -1:
        print('Warning: "endeffectoll" link not found; left leg IK disabled')

    # Find right leg end-effector link
    ee_index_rl = -1
    for ji in range(num_joints):
        if p.getJointInfo(robot_id, ji)[12].decode('utf-8') == 'endeffectorl':
            ee_index_rl = ji
            print(f'Right leg end effector at joint index {ee_index_rl}')
            break
    if ee_index_rl == -1:
        print('Warning: "endeffectorl" link not found; right leg IK disabled')

    # Find head end-effector link
    ee_index_h = -1
    for ji in range(num_joints):
        if p.getJointInfo(robot_id, ji)[12].decode('utf-8') == 'endeffectoh':
            ee_index_h = ji
            print(f'Head end effector at joint index {ee_index_h}')
            break
    if ee_index_h == -1:
        print('Warning: "endeffectoh" link not found; head IK disabled')

    # Initial EE pose from current FK (after joints are initialised)
    ls = p.getLinkState(robot_id, ee_index)
    ee_init_pos   = list(ls[0])
    ee_init_euler = list(p.getEulerFromQuaternion(ls[1]))
    print(f'Initial R EE pos:   {[round(v,3) for v in ee_init_pos]}')
    print(f'Initial R EE euler: {[round(v,3) for v in ee_init_euler]}')

    # Left EE initial pose
    if ee_index_l != -1:
        ls_l = p.getLinkState(robot_id, ee_index_l)
        ee_init_pos_l   = list(ls_l[0])
        ee_init_euler_l = list(p.getEulerFromQuaternion(ls_l[1]))
    else:
        ee_init_pos_l   = [0.0, 0.2, 0.3]
        ee_init_euler_l = [0.0, 0.0, 0.0]
    print(f'Initial L EE pos:   {[round(v,3) for v in ee_init_pos_l]}')
    print(f'Initial L EE euler: {[round(v,3) for v in ee_init_euler_l]}')

    # Left leg EE initial pose
    if ee_index_ll != -1:
        ls_ll = p.getLinkState(robot_id, ee_index_ll)
        ee_init_pos_ll   = list(ls_ll[0])
        ee_init_euler_ll = list(p.getEulerFromQuaternion(ls_ll[1]))
    else:
        ee_init_pos_ll   = [0.0, 0.15, -0.8]
        ee_init_euler_ll = [0.0, 0.0, 0.0]
    print(f'Initial LL EE pos:   {[round(v,3) for v in ee_init_pos_ll]}')

    # Right leg EE initial pose
    if ee_index_rl != -1:
        ls_rl = p.getLinkState(robot_id, ee_index_rl)
        ee_init_pos_rl   = list(ls_rl[0])
        ee_init_euler_rl = list(p.getEulerFromQuaternion(ls_rl[1]))
    else:
        ee_init_pos_rl   = [0.0, -0.15, -0.8]
        ee_init_euler_rl = [0.0, 0.0, 0.0]
    print(f'Initial RL EE pos:   {[round(v,3) for v in ee_init_pos_rl]}')

    # Head look-at target: cube spawns 1 m ahead along the head EE forward axis.
    # Each frame the direction (endeffectoh → cube) drives the look-at IK.
    if ee_index_h != -1:
        _ls_h_init = p.getLinkState(robot_id, ee_index_h)
        _h_pos     = np.array(_ls_h_init[0])
        _h_quat    = _ls_h_init[1]
        _rot_h     = np.array(p.getMatrixFromQuaternion(_h_quat)).reshape(3, 3)
        _h_fwd     = _rot_h[:, 0]            # local +X axis in world (head forward)
        ee_init_pos_h  = list(_h_pos + 1.0 * _h_fwd)
        ee_init_quat_h = list(_h_quat)
    else:
        ee_init_pos_h  = [1.2, 0.0, 0.7]
        ee_init_quat_h = [0.0, 0.0, 0.0, 1.0]
    print(f'Initial H look-at target: {[round(v,3) for v in ee_init_pos_h]}')

    # Precompute correct IK solution indices for legs + head
    # calculateInverseKinematics returns one value per non_fixed joint (incl. grippers);
    # use non_fixed_idxs.index(ji) for unambiguously correct solution indexing.
    ll_sol_idxs:      list[int] = []
    ll_jt_motor_ids:  list[int] = []
    rl_sol_idxs:      list[int] = []
    rl_jt_motor_ids:  list[int] = []
    hd_sol_idxs:      list[int] = []
    hd_jt_motor_ids:  list[int] = []
    for ji, jname in zip(ik_joint_idxs, ik_joint_names):
        motor_id = URDF_TO_MOTOR.get(jname)
        if motor_id is None:
            continue
        sol_idx = non_fixed_idxs.index(ji)
        if motor_id in LEFT_LEG_IK_MOTOR_IDS:
            ll_sol_idxs.append(sol_idx)
            ll_jt_motor_ids.append(motor_id)
        elif motor_id in RIGHT_LEG_IK_MOTOR_IDS:
            rl_sol_idxs.append(sol_idx)
            rl_jt_motor_ids.append(motor_id)
        elif motor_id in HEAD_IK_MOTOR_IDS:
            hd_sol_idxs.append(sol_idx)
            hd_jt_motor_ids.append(motor_id)

    # Visual target boxes  (red = right, cyan = left)
    box_vis_r = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[0.02] * 3, rgbaColor=[1, 0, 0, 0.6]
    )
    box_id = p.createMultiBody(
        baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=box_vis_r,
        basePosition=ee_init_pos,
        baseOrientation=p.getQuaternionFromEuler(ee_init_euler),
    )

    box_vis_l = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[0.02] * 3, rgbaColor=[0, 0.8, 1, 0.6]
    )
    box_id_l = p.createMultiBody(
        baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=box_vis_l,
        basePosition=ee_init_pos_l,
        baseOrientation=p.getQuaternionFromEuler(ee_init_euler_l),
    )

    # Visual target box — LEFT LEG (orange)
    box_vis_ll = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[0.02] * 3, rgbaColor=[1, 0.5, 0, 0.6]
    )
    box_id_ll = p.createMultiBody(
        baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=box_vis_ll,
        basePosition=ee_init_pos_ll,
        baseOrientation=p.getQuaternionFromEuler(ee_init_euler_ll),
    )

    # Visual target box — RIGHT LEG (magenta)
    box_vis_rl = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[0.02] * 3, rgbaColor=[1, 0, 0.8, 0.6]
    )
    box_id_rl = p.createMultiBody(
        baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=box_vis_rl,
        basePosition=ee_init_pos_rl,
        baseOrientation=p.getQuaternionFromEuler(ee_init_euler_rl),
    )

    # Visual target box — HEAD (yellow)
    box_vis_h = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[0.02] * 3, rgbaColor=[1, 1, 0, 0.6]
    )
    box_id_h = p.createMultiBody(
        baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=box_vis_h,
        basePosition=ee_init_pos_h,
        baseOrientation=ee_init_quat_h,
    )

    # IK target sliders — RIGHT arm
    x_sl     = p.addUserDebugParameter('R EE Target X',     0.2,    0.7,  ee_init_pos[0])
    y_sl     = p.addUserDebugParameter('R EE Target Y',    -1.0,    0.2,  ee_init_pos[1])
    z_sl     = p.addUserDebugParameter('R EE Target Z',     0.1,    0.5,  ee_init_pos[2])
    roll_sl  = p.addUserDebugParameter('R EE Roll',  -math.pi, math.pi, ee_init_euler[0])
    pitch_sl = p.addUserDebugParameter('R EE Pitch', -math.pi, math.pi, ee_init_euler[1])
    yaw_sl   = p.addUserDebugParameter('R EE Yaw',   -math.pi, math.pi, ee_init_euler[2])

    # IK target sliders — LEFT arm
    lx_sl     = p.addUserDebugParameter('L EE Target X',    0.2,    0.7,  ee_init_pos_l[0])
    ly_sl     = p.addUserDebugParameter('L EE Target Y',    -0.2,    1.0,  ee_init_pos_l[1])
    lz_sl     = p.addUserDebugParameter('L EE Target Z',     0.1,    0.5,  ee_init_pos_l[2])
    lroll_sl  = p.addUserDebugParameter('L EE Roll',  -math.pi, math.pi, ee_init_euler_l[0])
    lpitch_sl = p.addUserDebugParameter('L EE Pitch', -math.pi, math.pi, ee_init_euler_l[1])
    lyaw_sl   = p.addUserDebugParameter('L EE Yaw',   -math.pi, math.pi, ee_init_euler_l[2])

    # IK target sliders — LEFT LEG
    llx_sl     = p.addUserDebugParameter('LL EE Target X',     -0.5,   0.5,  ee_init_pos_ll[0])
    lly_sl     = p.addUserDebugParameter('LL EE Target Y',     -0.5,   0.5,  ee_init_pos_ll[1])
    llz_sl     = p.addUserDebugParameter('LL EE Target Z',     -1.2,   0.0,  ee_init_pos_ll[2])
    llroll_sl  = p.addUserDebugParameter('LL EE Roll',  -math.pi, math.pi, ee_init_euler_ll[0])
    llpitch_sl = p.addUserDebugParameter('LL EE Pitch', -math.pi, math.pi, ee_init_euler_ll[1])
    llyaw_sl   = p.addUserDebugParameter('LL EE Yaw',   -math.pi, math.pi, ee_init_euler_ll[2])

    # IK target sliders — RIGHT LEG
    rlx_sl     = p.addUserDebugParameter('RL EE Target X',     -0.5,   0.5,  ee_init_pos_rl[0])
    rly_sl     = p.addUserDebugParameter('RL EE Target Y',     -0.5,   0.5,  ee_init_pos_rl[1])
    rlz_sl     = p.addUserDebugParameter('RL EE Target Z',     -1.2,   0.0,  ee_init_pos_rl[2])
    rlroll_sl  = p.addUserDebugParameter('RL EE Roll',  -math.pi, math.pi, ee_init_euler_rl[0])
    rlpitch_sl = p.addUserDebugParameter('RL EE Pitch', -math.pi, math.pi, ee_init_euler_rl[1])
    rlyaw_sl   = p.addUserDebugParameter('RL EE Yaw',   -math.pi, math.pi, ee_init_euler_rl[2])

    # IK target sliders — HEAD (look-at: 3 sliders, orientation auto-computed)
    hx_sl = p.addUserDebugParameter('H Look-At X', -1.5, 1.5, ee_init_pos_h[0])
    hy_sl = p.addUserDebugParameter('H Look-At Y', -1.5, 1.5, ee_init_pos_h[1])
    hz_sl = p.addUserDebugParameter('H Look-At Z',  0.0, 2.5, ee_init_pos_h[2])

    p.resetDebugVisualizerCamera(
        cameraDistance=2.0, cameraYaw=45, cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0.5],
    )

    # Status overlay text
    send_label_id = p.addUserDebugText(
        'IK SEND: OFF  (X=toggle | 1/2=R hand | 3/4=L hand | r=save | 7=pos | 9=rec | 0=traj | 8=exec | t=print | q=quit)',

        textPosition=[0, 0, 1.3],
        textColorRGB=[0.9, 0.4, 0.1],
        textSize=1.0,
    )

    ik_sending = False       # toggled by 'x'
    trajectory_waypoints: list = []   # accumulated by '9'
    _input_active = {'value': False}  # guard against concurrent input prompts
    _traj_active  = {'value': False}  # True while a trajectory is streaming

    print('\nControls: X=toggle IK send  |  1/2=R hand  |  3/4=L hand  |  r=save  |  7=exec pos  |  9=rec wp  |  0=save traj  |  8=exec traj  |  t=print  |  q=quit')
    print('⚠  IK sending starts OFF — press X to enable after ensuring robot is safe.')

    try:
        while True:
            keys = p.getKeyboardEvents()

            # While an input() prompt is active, skip all key commands so
            # stray keypresses are not mis-interpreted as robot commands.
            if _input_active['value']:
                p.stepSimulation()
                time.sleep(0.02)
                continue

            # 'q' — quit
            if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
                print('Quitting.')
                break

            # 'x' — toggle IK sending
            if ord('x') in keys and keys[ord('x')] & p.KEY_WAS_TRIGGERED:
                ik_sending = not ik_sending
                state_str = 'ON  ⚡' if ik_sending else 'OFF'
                color = [0.2, 0.9, 0.2] if ik_sending else [0.9, 0.4, 0.1]
                p.addUserDebugText(
                    f'IK SEND: {state_str}  (X=toggle | 1/2=R hand | 3/4=L hand | r=save | 7=pos | 9=rec | 0=traj | 8=exec | t=print | q=quit)',

                    textPosition=[0, 0, 1.3],
                    textColorRGB=color, textSize=1.0,
                    replaceItemUniqueId=send_label_id,
                )
                print(f'[x] IK sending {"ENABLED" if ik_sending else "DISABLED"}')

            # '1' — open right hand
            if ord('1') in keys and keys[ord('1')] & p.KEY_WAS_TRIGGERED:
                open_pos = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
                threading.Thread(
                    target=ros_node.send_right_hand, args=(open_pos,), daemon=True
                ).start()
                print(f'[1] Right hand open')

            # '2' — close right hand
            if ord('2') in keys and keys[ord('2')] & p.KEY_WAS_TRIGGERED:
                close_pos = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
                threading.Thread(
                    target=ros_node.send_right_hand, args=(close_pos,), daemon=True
                ).start()
                print(f'[2] Right hand close')

            # '3' — open left hand
            if ord('3') in keys and keys[ord('3')] & p.KEY_WAS_TRIGGERED:
                open_pos = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
                threading.Thread(
                    target=ros_node.send_left_hand, args=(open_pos,), daemon=True
                ).start()
                print(f'[3] Left hand open')

            # '4' — close left hand
            if ord('4') in keys and keys[ord('4')] & p.KEY_WAS_TRIGGERED:
                close_pos = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
                threading.Thread(
                    target=ros_node.send_left_hand, args=(close_pos,), daemon=True
                ).start()
                print(f'[4] Left hand close')

            # 'r' — save current real pose to saved_positions.csv
            if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
                positions_file = os.path.join(examples_dir, 'saved_positions.csv')
                name = _save_to_gui_positions(
                    robot_id, ik_joint_idxs, ik_joint_names, positions_file
                )
                print(f"[r] Saved pose '{name}' → {positions_file}")

            # 't' — print live state
            if ord('t') in keys and keys[ord('t')] & p.KEY_WAS_TRIGGERED:
                real_pos = ros_node.get_positions()
                print(f'[t] Real robot: {len(real_pos)} motors')
                for mid in sorted(real_pos):
                    print(f'    motor {mid:2d}: {math.degrees(real_pos[mid]):7.2f}°')
                ls_now = p.getLinkState(robot_id, ee_index)
                ee_now = list(ls_now[0])
                ee_euler_now = list(p.getEulerFromQuaternion(ls_now[1]))
                print(f'[t] Real EE pos:   {[round(v,4) for v in ee_now]}')
                print(f'[t] Real EE euler: {[round(v,4) for v in ee_euler_now]}')

            # '9' — record current pose as trajectory waypoint
            if ord('9') in keys and keys[ord('9')] & p.KEY_WAS_TRIGGERED:
                wp = _get_current_pose_payload(robot_id, ik_joint_idxs, ik_joint_names)
                trajectory_waypoints.append(wp)
                print(f'[9] Waypoint {len(trajectory_waypoints)} recorded to trajectory buffer.')

            # '0' — save trajectory buffer to trajectories.csv
            if ord('0') in keys and keys[ord('0')] & p.KEY_WAS_TRIGGERED:
                if not trajectory_waypoints:
                    print('[0] Trajectory buffer is empty — press 9 to record waypoints first.')
                elif _input_active['value']:
                    print('[0] Another input prompt is already active.')
                else:
                    wps_snapshot = list(trajectory_waypoints)
                    def _save_traj_thread(wps=wps_snapshot):
                        _input_active['value'] = True
                        try:
                            traj_name = input('[0] Enter trajectory name: ').strip()
                            if traj_name:
                                traj_file = os.path.join(examples_dir, 'trajectories.csv')
                                _save_trajectory(traj_name, wps, traj_file)
                                trajectory_waypoints.clear()
                                print(f"[0] Trajectory '{traj_name}' saved "
                                      f'({len(wps)} waypoints) → {traj_file}')
                            else:
                                print('[0] Empty name — trajectory not saved.')
                        finally:
                            _input_active['value'] = False
                    threading.Thread(target=_save_traj_thread, daemon=True).start()

            # '7' — list and execute a saved position
            if ord('7') in keys and keys[ord('7')] & p.KEY_WAS_TRIGGERED:
                if _input_active['value']:
                    print('[7] Another input prompt is already active.')
                else:
                    def _exec_pos_thread():
                        _input_active['value'] = True
                        try:
                            positions_file = os.path.join(examples_dir, 'saved_positions.csv')
                            saved_pos = _load_saved_positions(positions_file)
                            if not saved_pos:
                                print('[7] No saved positions found in saved_positions.csv')
                                return
                            names = sorted(saved_pos.keys())
                            print('[7] Saved positions:')
                            for i, n in enumerate(names):
                                print(f'    {i + 1}. {n}')
                            sel = input('[7] Select position (number or name): ').strip()
                            chosen = None
                            try:
                                idx = int(sel) - 1
                                if 0 <= idx < len(names):
                                    chosen = names[idx]
                            except ValueError:
                                if sel in saved_pos:
                                    chosen = sel
                            if chosen is None:
                                print('[7] Invalid selection.')
                                return
                            print(f"[7] Moving to position '{chosen}' …")
                        finally:
                            _input_active['value'] = False
                        if chosen:
                            threading.Thread(
                                target=_execute_saved_position,
                                args=(saved_pos[chosen], ros_node),
                                daemon=True,
                            ).start()
                    threading.Thread(target=_exec_pos_thread, daemon=True).start()

            # '8' — list and execute a saved trajectory
            if ord('8') in keys and keys[ord('8')] & p.KEY_WAS_TRIGGERED:
                if _input_active['value']:
                    print('[8] Another input prompt is already active.')
                else:
                    def _exec_traj_thread(traj_active=_traj_active):
                        _input_active['value'] = True
                        chosen = None
                        chosen_delay = 2.0
                        try:
                            traj_file = os.path.join(examples_dir, 'trajectories.csv')
                            trajs = _load_trajectories(traj_file)
                            if not trajs:
                                print('[8] No trajectories found in trajectories.csv')
                                return
                            names = sorted(trajs.keys())
                            print('[8] Available trajectories:')
                            for i, n in enumerate(names):
                                print(f'    {i + 1}. {n}  ({len(trajs[n])} waypoints)')
                            sel = input('[8] Select trajectory (number or name): ').strip()
                            try:
                                idx = int(sel) - 1
                                if 0 <= idx < len(names):
                                    chosen = names[idx]
                            except ValueError:
                                if sel in trajs:
                                    chosen = sel
                            if chosen is None:
                                print('[8] Invalid selection.')
                                return
                            # Ask for speed (step delay)
                            spd_str = input('[8] Step delay in seconds per waypoint [2.0]: ').strip()
                            try:
                                chosen_delay = float(spd_str) if spd_str else 2.0
                                if chosen_delay <= 0:
                                    raise ValueError
                            except ValueError:
                                print('[8] Invalid delay — using 2.0 s.')
                                chosen_delay = 2.0
                            print(f"[8] Starting trajectory '{chosen}' at {chosen_delay} s/waypoint …")
                        finally:
                            _input_active['value'] = False
                        if chosen:
                            threading.Thread(
                                target=_execute_trajectory,
                                args=(trajs[chosen], ros_node),
                                kwargs={'step_delay': chosen_delay, 'traj_active': traj_active},
                                daemon=True,
                            ).start()
                    threading.Thread(target=_exec_traj_thread, daemon=True).start()

            # ── Mirror real robot → PyBullet ────────────────────────────────
            real_pos = ros_node.get_positions()
            for motor_id, urdf_name in MOTOR_TO_URDF.items():
                ji = name_to_joint.get(urdf_name)
                if ji is not None and motor_id in real_pos:
                    p.resetJointState(robot_id, ji, real_pos[motor_id])

            # ── Read EE targets from sliders ────────────────────────────────
            target_pos   = [p.readUserDebugParameter(x_sl),
                            p.readUserDebugParameter(y_sl),
                            p.readUserDebugParameter(z_sl)]
            target_euler = [p.readUserDebugParameter(roll_sl),
                            p.readUserDebugParameter(pitch_sl),
                            p.readUserDebugParameter(yaw_sl)]
            target_quat  = p.getQuaternionFromEuler(target_euler)

            target_pos_l   = [p.readUserDebugParameter(lx_sl),
                              p.readUserDebugParameter(ly_sl),
                              p.readUserDebugParameter(lz_sl)]
            target_euler_l = [p.readUserDebugParameter(lroll_sl),
                              p.readUserDebugParameter(lpitch_sl),
                              p.readUserDebugParameter(lyaw_sl)]
            target_quat_l  = p.getQuaternionFromEuler(target_euler_l)

            target_pos_ll   = [p.readUserDebugParameter(llx_sl),
                               p.readUserDebugParameter(lly_sl),
                               p.readUserDebugParameter(llz_sl)]
            target_euler_ll = [p.readUserDebugParameter(llroll_sl),
                               p.readUserDebugParameter(llpitch_sl),
                               p.readUserDebugParameter(llyaw_sl)]
            target_quat_ll  = p.getQuaternionFromEuler(target_euler_ll)

            target_pos_rl   = [p.readUserDebugParameter(rlx_sl),
                               p.readUserDebugParameter(rly_sl),
                               p.readUserDebugParameter(rlz_sl)]
            target_euler_rl = [p.readUserDebugParameter(rlroll_sl),
                               p.readUserDebugParameter(rlpitch_sl),
                               p.readUserDebugParameter(rlyaw_sl)]
            target_quat_rl  = p.getQuaternionFromEuler(target_euler_rl)

            target_pos_h = [p.readUserDebugParameter(hx_sl),
                            p.readUserDebugParameter(hy_sl),
                            p.readUserDebugParameter(hz_sl)]
            if ee_index_h == -1:
                target_quat_h = [0.0, 0.0, 0.0, 1.0]

            # ── Solve IK — RIGHT arm ────────────────────────────────────────
            ik_solution = p.calculateInverseKinematics(
                robot_id, ee_index, target_pos, target_quat
            )
            ik_commands: dict[int, float] = {}
            for sol_idx, (ji, jname) in enumerate(zip(ik_joint_idxs, ik_joint_names)):
                motor_id = URDF_TO_MOTOR.get(jname)
                if motor_id is not None and motor_id in IK_MOTOR_IDS:
                    ik_commands[motor_id] = ik_solution[sol_idx]

            # ── Solve IK — LEFT arm ─────────────────────────────────────────
            ik_commands_l: dict[int, float] = {}
            if ee_index_l != -1:
                ik_solution_l = p.calculateInverseKinematics(
                    robot_id, ee_index_l, target_pos_l, target_quat_l
                )
                for sol_idx, (ji, jname) in enumerate(zip(ik_joint_idxs, ik_joint_names)):
                    motor_id = URDF_TO_MOTOR.get(jname)
                    if motor_id is not None and motor_id in LEFT_IK_MOTOR_IDS:
                        ik_commands_l[motor_id] = ik_solution_l[sol_idx]

            # ── Solve IK — LEFT LEG ─────────────────────────────────────────
            ik_commands_ll: dict[int, float] = {}
            if ee_index_ll != -1:
                ik_solution_ll = p.calculateInverseKinematics(
                    robot_id, ee_index_ll, target_pos_ll, target_quat_ll
                )
                for sol_idx, motor_id in zip(ll_sol_idxs, ll_jt_motor_ids):
                    ik_commands_ll[motor_id] = ik_solution_ll[sol_idx]

            # ── Solve IK — RIGHT LEG ────────────────────────────────────────
            ik_commands_rl: dict[int, float] = {}
            if ee_index_rl != -1:
                ik_solution_rl = p.calculateInverseKinematics(
                    robot_id, ee_index_rl, target_pos_rl, target_quat_rl
                )
                for sol_idx, motor_id in zip(rl_sol_idxs, rl_jt_motor_ids):
                    ik_commands_rl[motor_id] = ik_solution_rl[sol_idx]

            # ── Solve IK — HEAD (look-at orientation) ───────────────────────
            ik_commands_h: dict[int, float] = {}
            if ee_index_h != -1:
                _ls_h_ik = p.getLinkState(robot_id, ee_index_h)
                target_quat_h = _get_look_at_quat(_ls_h_ik[0], target_pos_h)
                ik_solution_h = p.calculateInverseKinematics(
                    robot_id, ee_index_h, target_pos_h, target_quat_h,
                    lowerLimits=joints_lower,
                    upperLimits=joints_upper,
                    jointRanges=joints_ranges,
                    restPoses=joints_rest,
                    maxNumIterations=300,
                    residualThreshold=0.0001,
                )
                for sol_idx, motor_id in zip(hd_sol_idxs, hd_jt_motor_ids):
                    ik_commands_h[motor_id] = ik_solution_h[sol_idx]

            # ── Send IK commands to real robot (if enabled) ─────────────────
            # IK is suppressed while a trajectory is streaming so its continuous
            # setpoints are not overridden by the IK loop.
            if ik_sending and ik_commands and not _traj_active['value']:
                threading.Thread(
                    target=ros_node.send_positions,
                    args=(ik_commands,),
                    daemon=True,
                ).start()
            if ik_sending and ik_commands_l and not _traj_active['value']:
                threading.Thread(
                    target=ros_node.send_positions,
                    args=(ik_commands_l,),
                    daemon=True,
                ).start()
            if ik_sending and ik_commands_ll and not _traj_active['value']:
                threading.Thread(
                    target=ros_node.send_positions,
                    args=(ik_commands_ll,),
                    daemon=True,
                ).start()
            if ik_sending and ik_commands_rl and not _traj_active['value']:
                threading.Thread(
                    target=ros_node.send_positions,
                    args=(ik_commands_rl,),
                    daemon=True,
                ).start()
            if ik_sending and ik_commands_h and not _traj_active['value']:
                threading.Thread(
                    target=ros_node.send_positions,
                    args=(ik_commands_h,),
                    daemon=True,
                ).start()

            # ── Update target boxes ─────────────────────────────────────────
            p.resetBasePositionAndOrientation(box_id,    target_pos,    target_quat)
            p.resetBasePositionAndOrientation(box_id_l,  target_pos_l,  target_quat_l)
            p.resetBasePositionAndOrientation(box_id_ll, target_pos_ll, target_quat_ll)
            p.resetBasePositionAndOrientation(box_id_rl, target_pos_rl, target_quat_rl)
            p.resetBasePositionAndOrientation(box_id_h,  target_pos_h,  target_quat_h)

            # ── Colour boxes by IK accuracy ─────────────────────────────────
            ls = p.getLinkState(robot_id, ee_index)
            ee_pos_now   = np.array(ls[0])
            ee_euler_now = np.array(p.getEulerFromQuaternion(ls[1]))
            pos_ok = np.linalg.norm(ee_pos_now - np.array(target_pos)) < POSITION_THRESHOLD
            ori_ok = np.linalg.norm(ee_euler_now - np.array(target_euler)) < ORIENTATION_THRESHOLD
            if pos_ok and ori_ok:
                colour = [0, 1, 0, 0.6]
            elif pos_ok or ori_ok:
                colour = [0, 0.5, 1, 0.6]
            else:
                colour = [1, 0, 0, 0.6]
            p.changeVisualShape(box_id, -1, rgbaColor=colour)

            if ee_index_l != -1:
                ls_l = p.getLinkState(robot_id, ee_index_l)
                ee_pos_now_l   = np.array(ls_l[0])
                ee_euler_now_l = np.array(p.getEulerFromQuaternion(ls_l[1]))
                pos_ok_l = np.linalg.norm(ee_pos_now_l - np.array(target_pos_l)) < POSITION_THRESHOLD
                ori_ok_l = np.linalg.norm(ee_euler_now_l - np.array(target_euler_l)) < ORIENTATION_THRESHOLD
                if pos_ok_l and ori_ok_l:
                    colour_l = [0, 1, 0, 0.6]
                elif pos_ok_l or ori_ok_l:
                    colour_l = [0, 0.5, 1, 0.6]
                else:
                    colour_l = [0, 0.8, 1, 0.6]   # default cyan when both off
                p.changeVisualShape(box_id_l, -1, rgbaColor=colour_l)

            if ee_index_ll != -1:
                ls_ll = p.getLinkState(robot_id, ee_index_ll)
                ee_pos_now_ll   = np.array(ls_ll[0])
                ee_euler_now_ll = np.array(p.getEulerFromQuaternion(ls_ll[1]))
                pos_ok_ll = np.linalg.norm(ee_pos_now_ll - np.array(target_pos_ll)) < POSITION_THRESHOLD
                ori_ok_ll = np.linalg.norm(ee_euler_now_ll - np.array(target_euler_ll)) < ORIENTATION_THRESHOLD
                if pos_ok_ll and ori_ok_ll:
                    colour_ll = [0, 1, 0, 0.6]
                elif pos_ok_ll or ori_ok_ll:
                    colour_ll = [1, 0.8, 0, 0.6]
                else:
                    colour_ll = [1, 0.5, 0, 0.6]   # default orange
                p.changeVisualShape(box_id_ll, -1, rgbaColor=colour_ll)

            if ee_index_rl != -1:
                ls_rl = p.getLinkState(robot_id, ee_index_rl)
                ee_pos_now_rl   = np.array(ls_rl[0])
                ee_euler_now_rl = np.array(p.getEulerFromQuaternion(ls_rl[1]))
                pos_ok_rl = np.linalg.norm(ee_pos_now_rl - np.array(target_pos_rl)) < POSITION_THRESHOLD
                ori_ok_rl = np.linalg.norm(ee_euler_now_rl - np.array(target_euler_rl)) < ORIENTATION_THRESHOLD
                if pos_ok_rl and ori_ok_rl:
                    colour_rl = [0, 1, 0, 0.6]
                elif pos_ok_rl or ori_ok_rl:
                    colour_rl = [0.8, 0, 1, 0.6]
                else:
                    colour_rl = [1, 0, 0.8, 0.6]   # default magenta
                p.changeVisualShape(box_id_rl, -1, rgbaColor=colour_rl)

            if ee_index_h != -1:
                ls_h_col = p.getLinkState(robot_id, ee_index_h)
                actual_quat_h  = np.array(ls_h_col[1])
                desired_quat_h = np.array(target_quat_h)
                dot_h       = float(np.clip(abs(np.dot(actual_quat_h, desired_quat_h)), 0.0, 1.0))
                angle_err_h = 2.0 * np.arccos(dot_h)
                if angle_err_h < 0.1:
                    colour_h = [0, 1, 0, 0.6]       # green
                elif angle_err_h < 0.35:
                    colour_h = [1, 1, 0.3, 0.6]     # light yellow
                else:
                    colour_h = [1, 1, 0, 0.6]       # yellow (default)
                p.changeVisualShape(box_id_h, -1, rgbaColor=colour_h)

            p.stepSimulation()
            time.sleep(0.02)   # ~50 Hz

    except Exception as ex:
        print(f'Simulation error: {ex}')
    finally:
        ros_node.destroy_node()
        rclpy.shutdown()
        p.disconnect()


if __name__ == '__main__':
    main()
