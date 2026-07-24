#!/usr/bin/env python3
"""
sim2real.py — Evaluate a trained myGym model in PyBullet simulation, then
              offer to save or directly execute successful trajectories on the
              real S2 robot.

Workflow:
  1. Load a pretrained model + PyBullet environment (identical to test.py model path)
  2. Run evaluation episodes one at a time
  3. During each episode, record joint snapshots every --sample_every steps
  4. When an episode terminates with success, prompt:
       [1] Save trajectory to trajectories.csv  (same format as visualize_real_robot_ik.py)
       [2] Execute trajectory on real S2 robot  (requires ROS 2 / real robot)
       [s] Skip and continue to next episode
  5. After all eval episodes, print summary

Usage:
    python sim2real.py --config configs/AGM.json \\
                       --pretrained_model trained_models/.../best_model.zip \\
                       [--eval_episodes 10] [--sample_every 5]
"""

import csv
import json
import math
import os
import sys
import time
import threading
from typing import Any, Dict, Optional

import numpy as np

from myGym.train import (
    get_parser,
    get_arguments,
    configure_env,
    configure_implemented_combos,
    automatic_argument_assignment,
)

# ── ROS 2 / real-robot support (optional) ───────────────────────────────────
try:
    import rclpy
    from rclpy.node import Node
    from bodyctrl_msgs.msg import CmdSetMotorPosition, SetMotorPosition, MotorStatusMsg
    from sensor_msgs.msg import JointState
    _ROS2_AVAILABLE = True
except ImportError:
    _ROS2_AVAILABLE = False


# ── Joint mapping (mirrors visualize_real_robot_ik.py) ──────────────────────

MOTOR_TO_URDF: Dict[int, str] = {
    # Head
    1: 'head_roll_joint',
    2: 'head_pitch_joint',
    3: 'head_yaw_joint',
    # Left arm
    11: 'shoulder_pitch_l_joint',
    12: 'shoulder_roll_l_joint',
    13: 'shoulder_yaw_l_joint',
    14: 'elbow_pitch_l_joint',
    15: 'wrist_yaw_l_joint',
    16: 'wrist_pitch_l_joint',
    17: 'wrist_roll_l_joint',
    # Right arm
    21: 'shoulder_pitch_r_rjoint',
    22: 'shoulder_roll_r_rjoint',
    23: 'shoulder_yaw_r_rjoint',
    24: 'elbow_pitch_r_rjoint',
    25: 'wrist_yaw_r_rjoint',
    26: 'wrist_pitch_r_rjoint',
    27: 'wrist_roll_r_rjoint',
    # Waist
    31: 'body_yaw_rjoint',
    # Left leg
    51: 'hip_roll_l_joint',
    52: 'hip_pitch_l_joint',
    53: 'hip_yaw_l_joint',
    54: 'knee_pitch_l_joint',
    55: 'ankle_pitch_l_joint',
    56: 'ankle_roll_l_joint',
    # Right leg
    61: 'hip_roll_r_joint',
    62: 'hip_pitch_r_joint',
    63: 'hip_yaw_r_joint',
    64: 'knee_pitch_r_joint',
    65: 'ankle_pitch_r_joint',
    66: 'ankle_roll_r_joint',
}

# Mapping from URDF joint name → (position-array key, index-within-array)
# Used to build the waypoint payload from PyBullet joint states.
URDF_TO_GUI: Dict[str, tuple] = {
    # Left leg
    'hip_roll_l_joint':    ('left_leg_pos',  0),
    'hip_pitch_l_joint':   ('left_leg_pos',  1),
    'hip_yaw_l_joint':     ('left_leg_pos',  2),
    'knee_pitch_l_joint':  ('left_leg_pos',  3),
    'ankle_pitch_l_joint': ('left_leg_pos',  4),
    'ankle_roll_l_joint':  ('left_leg_pos',  5),
    # Right leg
    'hip_roll_r_joint':    ('right_leg_pos', 0),
    'hip_pitch_r_joint':   ('right_leg_pos', 1),
    'hip_yaw_r_joint':     ('right_leg_pos', 2),
    'knee_pitch_r_joint':  ('right_leg_pos', 3),
    'ankle_pitch_r_joint': ('right_leg_pos', 4),
    'ankle_roll_r_joint':  ('right_leg_pos', 5),
    # Waist
    'body_yaw_rjoint':         ('waist_pos',     0),
    # Head
    'head_roll_joint':         ('head_pos',      0),
    'head_pitch_joint':        ('head_pos',      1),
    'head_yaw_joint':          ('head_pos',      2),
    # Left arm
    'shoulder_pitch_l_joint':  ('left_arm_pos',  0),
    'shoulder_roll_l_joint':   ('left_arm_pos',  1),
    'shoulder_yaw_l_joint':    ('left_arm_pos',  2),
    'elbow_pitch_l_joint':     ('left_arm_pos',  3),
    'wrist_yaw_l_joint':       ('left_arm_pos',  4),
    'wrist_pitch_l_joint':     ('left_arm_pos',  5),
    'wrist_roll_l_joint':      ('left_arm_pos',  6),
    # Right arm — S2full URDF names (used by real robot)
    'shoulder_pitch_r_rjoint': ('right_arm_pos', 0),
    'shoulder_roll_r_rjoint':  ('right_arm_pos', 1),
    'shoulder_yaw_r_rjoint':   ('right_arm_pos', 2),
    'elbow_pitch_r_rjoint':    ('right_arm_pos', 3),
    'wrist_yaw_r_rjoint':      ('right_arm_pos', 4),
    # S2 simulation URDF may use elbow_yaw instead of wrist_yaw (slot 4)
    'elbow_yaw_r_rjoint':      ('right_arm_pos', 4),
    'wrist_pitch_r_rjoint':    ('right_arm_pos', 5),
    'wrist_roll_r_rjoint':     ('right_arm_pos', 6),
}

_LIMB_TO_MOTORS: Dict[str, list] = {
    'head_pos':      [1,  2,  3],
    'left_arm_pos':  [11, 12, 13, 14, 15, 16, 17],
    'right_arm_pos': [21, 22, 23, 24, 25, 26, 27],
    'waist_pos':     [31],
    'left_leg_pos':  [51, 52, 53, 54, 55, 56],
    'right_leg_pos': [61, 62, 63, 64, 65, 66],
}


# ── ROS 2 node (only instantiated when option 2 is selected) ────────────────

if _ROS2_AVAILABLE:
    def _motor_route(motor_id: int):
        if 11 <= motor_id <= 27:
            return 'arm',   0.5, 8.0
        if 51 <= motor_id <= 66:
            return 'leg',   0.5, 8.0
        if motor_id == 31:
            return 'waist', 0.2, 8.0
        if 1 <= motor_id <= 3:
            return 'head',  0.2, 2.0
        return None, 0.5, 8.0

    class _RobotNode(Node):
        def __init__(self):
            super().__init__('sim2real_node')
            self._pubs = {
                'arm':   self.create_publisher(CmdSetMotorPosition, '/arm/cmd_pos',   10),
                'leg':   self.create_publisher(CmdSetMotorPosition, '/leg/cmd_pos',   10),
                'waist': self.create_publisher(CmdSetMotorPosition, '/waist/cmd_pos', 10),
                'head':  self.create_publisher(CmdSetMotorPosition, '/head/cmd_pos',  10),
            }

        def send_positions(self, commands: Dict[int, float]):
            groups: Dict[str, list] = {}
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


# ── Trajectory utilities (same format as visualize_real_robot_ik.py) ─────────

def _snapshot_from_env(env) -> dict:
    """Read current joint positions from the myGym env's PyBullet robot and
    return a waypoint payload dict (degrees) in the trajectories.csv format.

    All limbs not controlled by the simulation stay at 0.0 degrees, which
    means the real robot will hold its current/default position for those joints.
    """
    robot = env.unwrapped.robot
    pb    = robot.p
    uid   = robot.robot_uid

    limb_sizes = {
        'left_leg_pos': 6, 'right_leg_pos': 6,
        'left_arm_pos': 7, 'right_arm_pos': 7,
        'waist_pos':    1, 'head_pos':      3,
    }
    positions = {k: [0.0] * n for k, n in limb_sizes.items()}

    num_joints = pb.getNumJoints(uid)
    for ji in range(num_joints):
        info  = pb.getJointInfo(uid, ji)
        jtype = info[2]
        if jtype == pb.JOINT_FIXED:
            continue
        jname = info[1].decode('utf-8')
        if jname not in URDF_TO_GUI:
            continue
        limb_key, idx = URDF_TO_GUI[jname]
        angle_rad = pb.getJointState(uid, ji)[0]
        positions[limb_key][idx] = round(math.degrees(angle_rad), 2)

    return {
        'leg_mode': 'Position', 'arm_mode': 'Position',
        'left_leg_pos':  positions['left_leg_pos'],
        'right_leg_pos': positions['right_leg_pos'],
        'left_arm_pos':  positions['left_arm_pos'],
        'right_arm_pos': positions['right_arm_pos'],
        'leg_profile_speed': 0.5, 'leg_position_current': 8.0, 'leg_speed_current': 8.0,
        'arm_profile_speed': 0.5, 'arm_position_current': 8.0, 'arm_speed_current': 8.0,
        'waist_pos': positions['waist_pos'], 'waist_speed': [0.2],
        'head_pos':  positions['head_pos'],  'head_speed':  [0.2],
        'left_finger_pos':  [0.0] * 6, 'right_finger_pos': [0.0] * 6,
        'left_finger_vel':  [1.0] * 6, 'right_finger_vel': [1.0] * 6,
        'hand_effort': [1.0],
    }


def _save_trajectory(traj_name: str, waypoints: list, traj_file: str) -> None:
    """Append / overwrite a named trajectory in trajectories.csv."""
    trajectories: dict = {}
    if os.path.exists(traj_file):
        try:
            with open(traj_file, 'r', newline='', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    n  = (row.get('name') or '').strip()
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


def _execute_trajectory_ros(waypoints: list, ros_node, step_delay: float = 2.0,
                             interp_hz: float = 20.0) -> None:
    """Stream trajectory waypoints to the real S2 robot via ROS 2 (blocking),
    then return to the start by replaying waypoints in reverse order."""

    def _interp_commands(wp_a: dict, wp_b: dict, alpha: float) -> Dict[int, float]:
        cmds: Dict[int, float] = {}
        for limb_key, motor_ids in _LIMB_TO_MOTORS.items():
            a_ang = wp_a.get(limb_key, [])
            b_ang = wp_b.get(limb_key, [])
            for idx, motor_id in enumerate(motor_ids):
                a = float(a_ang[idx]) if idx < len(a_ang) else 0.0
                b = float(b_ang[idx]) if idx < len(b_ang) else 0.0
                cmds[motor_id] = math.radians(a + alpha * (b - a))
        return cmds

    def _stream(sequence: list, label: str) -> None:
        """Interpolate and stream a list of waypoints."""
        n_interp = max(1, int(step_delay * interp_hz))
        sleep_dt = 1.0 / interp_hz
        ros_node.send_positions(_interp_commands(sequence[0], sequence[0], 1.0))
        print(f'[real] {label} waypoint 1 / {len(sequence)}')
        for i in range(1, len(sequence)):
            for step in range(1, n_interp + 1):
                alpha = step / n_interp
                ros_node.send_positions(
                    _interp_commands(sequence[i - 1], sequence[i], alpha)
                )
                time.sleep(sleep_dt)
            print(f'[real] {label} waypoint {i + 1} / {len(sequence)}')

    print(f'[real] Executing {len(waypoints)} waypoints '
          f'({step_delay} s × {interp_hz:.0f} Hz per step) …')
    _stream(waypoints, 'Forward')

    print(f'[real] Returning to start — replaying {len(waypoints)} waypoints in reverse …')
    _stream(list(reversed(waypoints)), 'Return')

    print('[real] Done.')


# ── Post-success menu ────────────────────────────────────────────────────────

def _handle_success_menu(waypoints: list, episode: int,
                          traj_file: str, step_delay: float) -> None:
    """Prompt the user what to do with the recorded trajectory."""
    print()
    print('=' * 60)
    print(f'  SUCCESS — Episode {episode}  ({len(waypoints)} waypoints recorded)')
    print('=' * 60)
    print('  [1] Save trajectory to trajectories.csv')
    print('  [2] Execute trajectory on real S2 robot now')
    print('  [s] Skip')
    print('=' * 60)

    while True:
        choice = input('Choice: ').strip().lower()

        if choice == '1':
            default_name = f'sim2real_ep{episode}'
            name = input(f'Trajectory name [{default_name}]: ').strip()
            if not name:
                name = default_name
            _save_trajectory(name, waypoints, traj_file)
            print(f"[1] Saved '{name}' ({len(waypoints)} waypoints) → {traj_file}")
            break

        elif choice == '2':
            if not _ROS2_AVAILABLE:
                print('[2] ROS 2 is not available in this environment. '
                      'Install rclpy and robot message packages to use this option.')
                continue
            spd_str = input('[2] Step delay in seconds per waypoint [0.05]: ').strip()
            try:
                exec_delay = float(spd_str) if spd_str else 0.05
                if exec_delay <= 0:
                    raise ValueError
            except ValueError:
                print('[2] Invalid delay — using 0.05 s.')
                exec_delay = 0.05
            print('[2] Initialising ROS 2 …')
            try:
                rclpy.init()
                ros_node = _RobotNode()
                spin_thread = threading.Thread(
                    target=rclpy.spin, args=(ros_node,), daemon=True
                )
                spin_thread.start()
                print('[2] ⚠  WARNING — This will move physical hardware!')
                print(f'[2] Speed: {exec_delay} s/waypoint  |  will return to start after completion.')
                input('[2] Press Enter to confirm execution (Ctrl+C to abort): ')
                _execute_trajectory_ros(waypoints, ros_node, step_delay=exec_delay)
            except Exception as exc:
                print(f'[2] Error during real-robot execution: {exc}')
            finally:
                try:
                    if 'ros_node' in dir():
                        ros_node.destroy_node()
                    rclpy.shutdown()
                except Exception:
                    pass
            break

        elif choice in ('s', 'skip', ''):
            print('Skipping.')
            break

        else:
            print('Invalid choice. Enter 1, 2, or s.')


# ── Main evaluation loop ─────────────────────────────────────────────────────

def evaluate(env: Any, model, arg_dict: Dict[str, Any],
             model_logdir: str, deterministic: bool = False) -> None:
    """Evaluate model in PyBullet and offer sim→real transfer after each success."""
    import pybullet as p_module

    sample_every  = arg_dict.get('sample_every', 5)
    step_delay    = arg_dict.get('step_delay',   0.05)
    eval_episodes = arg_dict.get('eval_episodes', 10)
    script_dir    = os.path.dirname(os.path.abspath(__file__))
    traj_file     = os.path.join(script_dir, 'trajectories.csv')

    success_count    = 0
    distance_err_sum = 0.0
    steps_sum        = 0

    print(f'\n[sim2real] Running {eval_episodes} episode(s), '
          f'sampling joints every {sample_every} step(s).')
    print(f'[sim2real] Trajectory file: {traj_file}')
    print()

    for ep in range(1, eval_episodes + 1):
        obs, info = env.reset()
        done      = False
        step      = 0
        waypoints: list = []
        is_successful   = False
        distance_error  = 0.0

        print(f'--- Episode {ep} / {eval_episodes} ---')

        while not done:
            step      += 1
            steps_sum += 1

            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Print subgoal progress
            rewarder = env.unwrapped.reward if hasattr(env.unwrapped, 'reward') else None
            if rewarder is not None and hasattr(rewarder, 'last_result'):
                result = rewarder.last_result
                print(
                    f"  Subgoal: {rewarder.network_name} "
                    f"({rewarder.owner + 1}/{rewarder.num_networks}) | "
                    f"Dist: {result['absolute_distance']:.4f} | "
                    f"Arm: {result['arm_progress']:.1f}% (solved={result['arm_solved']}) | "
                    f"Gripper: {result['gripper_progress']:.1f}% "
                    f"(solved={result['gripper_solved']}) | "
                    f"Reward: {reward:.4f}",
                    end='\r', flush=True,
                )

            # Record joint snapshot every sample_every steps
            if step % sample_every == 0 or done:
                try:
                    wp = _snapshot_from_env(env)
                    waypoints.append(wp)
                except Exception as exc:
                    print(f'\n[warn] Could not snapshot joints at step {step}: {exc}')

            is_successful  = not info.get('f', True)
            distance_error = info.get('d', 0.0)

        print()  # newline after \r progress
        print(f'  Episode {ep}: {"SUCCESS" if is_successful else "FAIL"} '
              f'after {step} steps  (dist={distance_error:.4f})')

        success_count    += int(is_successful)
        distance_err_sum += distance_error

        #if is_successful and waypoints:
        if waypoints:
            _handle_success_menu(waypoints, ep, traj_file, step_delay)

    # Summary
    print()
    print('#─────────────── Evaluation Summary ───────────────#')
    print(f'  {success_count} / {eval_episodes} episodes successful '
          f'({success_count / eval_episodes * 100:.1f} %)')
    print(f'  Mean distance error: {distance_err_sum / eval_episodes * 100:.2f} %')
    print(f'  Mean steps per episode: {steps_sum // eval_episodes}')
    print('#──────────────────────────────────────────────────#')

    # Write summary to log file (matches test.py behaviour)
    model_name = arg_dict.get('algo', 'model') + '_' + str(arg_dict.get('steps', ''))
    log_path   = os.path.join(model_logdir, f'sim2real_{model_name}.txt')
    try:
        with open(log_path, 'a') as f:
            f.write('\n#sim2real evaluation results:\n')
            f.write(f'#{success_count} / {eval_episodes} episodes successful\n')
            f.write(f'#Mean distance error: {distance_err_sum / eval_episodes * 100:.2f}%\n')
            f.write(f'#Mean steps: {steps_sum // eval_episodes}\n')
        print(f'[sim2real] Log written to {log_path}')
    except OSError as exc:
        print(f'[sim2real] Could not write log: {exc}')


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = get_parser()
    # Extra sim2real-specific arguments
    parser.add_argument(
        '--sample_every', type=int, default=3,
        help='Record a joint snapshot every N simulation steps (default: 5).',
    )
    parser.add_argument(
        '--step_delay', type=float, default=0.05,
        help='Seconds per waypoint when executing on the real robot (default: 0.05).',
    )
    parser.add_argument(
        '--deterministic', action='store_true', default=False,
        help='Use deterministic model predictions (default: False).',
    )

    # Convenience: if a .zip file was passed as --config, treat it as --pretrained_model
    for flag in ('-cfg', '--config'):
        if flag in sys.argv:
            idx = sys.argv.index(flag)
            val = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else ''
            if val.endswith('.zip'):
                print(f'[sim2real] Detected model zip passed as {flag}; '
                      f'treating as --pretrained_model.')
                sys.argv[idx] = '--pretrained_model'
                break

    # Auto-discover train.json from the model directory when no --config is given
    config_flags = {'-cfg', '--config'}
    has_config = any(f in sys.argv for f in config_flags)
    if not has_config:
        # Find the model path from sys.argv
        model_zip = None
        for flag in ('-ptm', '--pretrained_model'):
            if flag in sys.argv:
                idx = sys.argv.index(flag)
                if idx + 1 < len(sys.argv):
                    model_zip = sys.argv[idx + 1]
                    break
        if model_zip:
            candidate = os.path.join(os.path.dirname(os.path.abspath(model_zip)), 'train.json')
            if os.path.exists(candidate):
                print(f'[sim2real] Auto-loading config: {candidate}')
                sys.argv.extend(['--config', candidate])
            else:
                print(f'[sim2real] Warning: no train.json found in {os.path.dirname(model_zip)}')

    arg_dict, _ = get_arguments(parser)

    if arg_dict.get('pretrained_model') is None:
        print('[sim2real] --pretrained_model is required.')
        print('           Specify the path to a trained model directory or zip file.')
        parser.print_help()
        sys.exit(1)

    ptm = arg_dict['pretrained_model']
    # The custom PPO.load() appends "/best_model" internally, so model_logdir
    # must be the directory containing best_model.zip.  If the user passed the
    # zip file itself, take its parent directory.
    if ptm.endswith('.zip'):
        model_logdir = os.path.dirname(os.path.abspath(ptm))
    else:
        model_logdir = os.path.abspath(ptm)

    print('[sim2real] Configuring environment …')
    env = configure_env(arg_dict, model_logdir, for_train=0)

    print('[sim2real] Loading model …')
    implemented_combos = configure_implemented_combos(env, model_logdir, arg_dict)

    # The custom PPO.load() appends "/best_model" internally, so it expects
    # a directory path, not a .zip file path.
    load_path = model_logdir

    try:
        if 'multi' in arg_dict.get('algo', ''):
            model = implemented_combos[arg_dict['algo']][arg_dict['train_framework']][0].load(
                load_path, env=env
            )
        else:
            model = implemented_combos[arg_dict['algo']][arg_dict['train_framework']][0].load(
                load_path, env=env
            )
    except Exception as exc:
        algo = arg_dict.get('algo', '?')
        fw   = arg_dict.get('train_framework', '?')
        if algo in implemented_combos and fw not in implemented_combos[algo]:
            print(f'[sim2real] {algo} is only implemented for '
                  f'{list(implemented_combos[algo].keys())[0]}')
        elif algo not in implemented_combos:
            print(f'[sim2real] Algorithm "{algo}" is not implemented.')
        else:
            print(f'[sim2real] Failed to load model: {exc}')
        sys.exit(1)

    print(f'[sim2real] Model loaded: {arg_dict["pretrained_model"]}')
    print(f'[sim2real] Eval episodes: {arg_dict.get("eval_episodes", 10)}')
    print(f'[sim2real] Sample every:  {arg_dict.get("sample_every", 5)} steps')

    evaluate(
        env,
        model,
        arg_dict,
        model_logdir,
        deterministic=arg_dict.get('deterministic', False),
    )


if __name__ == '__main__':
    main()
