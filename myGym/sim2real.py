import multiprocessing
import os
import subprocess
import time
from typing import Dict, Any

import cv2
import imageio
import numpy as np
import pybullet as p
import pybullet_data
from numpy import matrix
from sklearn.model_selection import ParameterGrid
import pandas as pd

from utils.grasper import Grasper
from utils.sim_height_calculation import calculate_z



from myGym import oraculum
from myGym.train import get_parser, get_arguments, configure_implemented_combos, configure_env, automatic_argument_assignment
from myGym.envs.env_object import EnvObject

clear = lambda: os.system('clear')

AVAILABLE_SIMULATION_ENGINES = ["mujoco", "pybullet"]
AVAILABLE_TRAINING_FRAMEWORKS = ["tensorflow", "pytorch"]

print("Initializing Grasper...")
try:
    grasper = Grasper(
        urdf_path="./envs/robots/nico/nico_grasper_real.urdf",
        motor_config="./utils/nico_humanoid_upper_rh7d_ukba.json",
        connect_robot=True,     # Connect to the real robot hardware
    )
    print("Grasper initialized successfully for real robot.")
except Exception as e:
    print(f"Error initializing Grasper for real robot: {e}")

def visualize_sampling_area(arg_dict: dict) -> None:
    task_object = arg_dict["task_objects"][0]
    goal_area = task_object["goal"]["sampling_area"]

    # Calculate the half-extents (rx, ry, rz)
    rx = (goal_area[0] - goal_area[1]) / 2
    ry = (goal_area[2] - goal_area[3]) / 2
    rz = (goal_area[4] - goal_area[5]) / 2

    # Create a visual shape and multi-body for the sampling area
    visual = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[rx, ry, rz], rgbaColor=[1, 0, 0, .2])
    collision = -1

    p.createMultiBody(
        baseVisualShapeIndex=visual,
        baseCollisionShapeIndex=collision,
        baseMass=0,
        basePosition=[goal_area[0] - rx, goal_area[2] - ry, goal_area[4] - rz],
    )


def visualize_trajectories(info: dict, action: list) -> None:
    visual_actual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[0, 0, 1, .3])
    collision = -1
    p.createMultiBody(
        baseVisualShapeIndex=visual_actual,
        baseCollisionShapeIndex=collision,
        baseMass=0,
        basePosition=info['o']['actual_state'],
    )

    visual_action = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[1, 0, 0, .3])
    p.createMultiBody(
        baseVisualShapeIndex=visual_action,
        baseCollisionShapeIndex=collision,
        baseMass=0,
        basePosition=action[:3],
    )


def visualize_goal(info: dict) -> None:
    visual_goal = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[1, 0, 0, .5])
    collision = -1
    p.createMultiBody(
        baseVisualShapeIndex=visual_goal,
        baseCollisionShapeIndex=collision,
        baseMass=0,
        basePosition=info['o']['goal_state'],
    )


def change_dynamics(cubex: int, lfriction: int, rfriction: int, ldamping: int, adamping: int) -> None:
    p.changeDynamics(cubex, -1, lateralFriction=p.readUserDebugParameter(lfriction))
    p.changeDynamics(cubex, -1, rollingFriction=p.readUserDebugParameter(rfriction))
    p.changeDynamics(cubex, -1, linearDamping=p.readUserDebugParameter(ldamping))
    p.changeDynamics(cubex, -1, angularDamping=p.readUserDebugParameter(adamping))


def visualize_infotext(action: list, env: object, info: dict) -> None:
    debug_params = [
        (f"Episode: {env.env.episode_number}", [0.65, 1., 0.45], [0.4, 0.2, .3]),
        (f"Step: {env.env.episode_steps}", [0.67, 1., .40], [0.2, 0.8, 1]),
        (f"Subtask: {env.env.task.current_task}", [0.69, 1., 0.35], [0.4, 0.2, 1]),
        (f"Network: {env.env.unwrapped.reward.current_network}", [0.71, 1., 0.3], [0.0, 0.0, 1]),
        (f"Action (Gripper): {matrix(np.around(np.array(action), 2))}", [0.73, 1., 0.25], [1, 0, 0]),
        (f"Actual State: {matrix(np.around(np.array(env.env.observation['task_objects']['actual_state'][:3]), 2))}",
         [0.75, 1., 0.2], [0.0, 1., 0.0]),
        (f"End Effector: {matrix(np.around(np.array(env.env.robot.end_effector_pos), 2))}", [0.77, 1., 0.15],
         [0.0, 1., 0.0]),
        (f"Object: {matrix(np.around(np.array(info['o']['actual_state']), 2))}", [0.8, 1., 0.10], [0.0, 0.0, 1]),
        (f"Velocity: {env.env.max_velocity}", [0.79, 1., 0.05], [0.6, 0.8, .3]),
        (f"Force: {env.env.max_force}", [0.81, 1., 0.00], [0.3, 0.2, .4]),
    ]

    for text, pos, color in debug_params:
        p.addUserDebugText(text, pos, textSize=1.0, lifeTime=0.5, textColorRGB=color)


# Function to detect key presses and update action accordingly
def detect_key(keypress: dict, arg_dict: dict, action: list) -> list:
    key_action_mapping = {
        97: (2, 0.03),  # A
        122: (2, -0.03),  # Z/Y
        65297: (1, -0.03),  # ARROW UP
        65298: (1, 0.03),  # ARROW DOWN
        65295: (0, 0.03),  # ARROW LEFT
        65296: (0, -0.03),  # ARROW RIGHT
        120: [(3, -0.03), (4, -0.03)],  # X
        99: [(3, 0.03), (4, 0.03)],  # C
        113: [(0, 0)],  # Q
        110: [(0, 0)]  # N - next task (useful when the robot gets stuck, no need to wait for episode end)
    }

    for key, value in key_action_mapping.items():
        if key in keypress.keys() and keypress[key] == 1:
            if key == 113:
                print("'Q' pressed, quitting")
                quit()
            elif (key == 99 or key == 120) and "gripper" not in arg_dict["robot_action"]:
                print("Gripper is not present, cannot perform actions 'C' or 'X'")
            elif isinstance(value, tuple):
                action[value[0]] += value[1]
            elif isinstance(value, list):
                for v in value:
                    action[v[0]] += v[1]

    if "step" in arg_dict["robot_action"]:
        action[:3] = np.multiply(action[:3], 10)
    elif "joints" in arg_dict["robot_action"]:
        print("Robot action: Joints - KEYBOARD CONTROL UNDER DEVELOPMENT, quitting")
        quit()

    return action


def n_pressed(last_call_time):
    """Function which detects, whether the key n was pressed and new episode should be launched"""
    keypress = p.getKeyboardEvents()
    now = time.time()
    if now - last_call_time > 0.5:
        for key in keypress.keys():
            if key == 110:
                print("N pressed, switching to next subtask")
                return True, now
        return False, last_call_time
    else:
        return False, last_call_time


def apply_goal_position_override(env: Any, pos_xyz) -> bool:
    """Teleport the goal object to pos_xyz if available. Returns True on success."""
    try:
        goal = getattr(env.unwrapped, 'task_objects', {}).get('goal_state', None)
        if not isinstance(goal, EnvObject):
            return False
        uid = getattr(goal, 'uid', None)
        if uid is None:
            return False
        p_client = getattr(env.unwrapped, 'p', p)
        _, orn = p_client.getBasePositionAndOrientation(uid)
        target = [float(pos_xyz[0]), float(pos_xyz[1]), float(pos_xyz[2])]
        p_client.resetBasePositionAndOrientation(uid, target, orn)
        return True
    except Exception as ex:
        print("[WARN] Failed to override goal position:", ex)
        return False



def sync_hardware_pose_from_env(env: Any, grasper: Grasper) -> bool:
    """Map sim joint states to hardware joint names and command the real robot.
    Returns True if a command was sent, False otherwise.
    """
    try:
        robot = getattr(env.unwrapped, 'robot', None)
        if robot is None or not getattr(grasper, 'is_robot_connected', False):
            return False
        indices = getattr(robot, 'motor_indices', [])
        mapped_names, rad_values = [], []
        for jid in indices:
            ji = robot.p.getJointInfo(robot.robot_uid, jid)
            nm = ji[1]
            nm = nm.decode('utf-8') if isinstance(nm, (bytes, bytearray)) else str(nm)
            base = nm.replace('_rjoint', '').replace('_pjoint', '').replace('_gjoint', '')
            q = robot.p.getJointState(robot.robot_uid, jid)[0]
            mapped_names.append(base)
            rad_values.append(float(q))
        nico_deg = grasper.rad2nicodeg(mapped_names, rad_values)
        hw_keys = set(grasper.INIT_POS.keys())
        target_angles_deg = {k: v for k, v in nico_deg.items() if k in hw_keys}
        if target_angles_deg:
            grasper.perform_move(target_angles_deg)
            return True
    except Exception as ex:
        print("[WARN] Could not sync hardware pose:", ex)
    return False

def test_model(
        env: Any,
        model=None,
        implemented_combos: Dict[str, Any] = None,
        arg_dict: Dict[str, Any] = None,
        model_logdir: str = None,
        deterministic: bool = False
) -> None:
    env.reset()
    try:
        #TODO: maybe this if else is unnecessary?
        if "multi" in arg_dict["algo"]:
            model_args = implemented_combos[arg_dict["algo"]][arg_dict["train_framework"]][1]
            model = implemented_combos[arg_dict["algo"]][arg_dict["train_framework"]][0].load(arg_dict["pretrained_model"], env = env)
            model.env = model_args[1].env
        else:
            model = implemented_combos[arg_dict["algo"]][arg_dict["train_framework"]][0].load(arg_dict["pretrained_model"], env = env)
    except:
        if (arg_dict["algo"] in implemented_combos.keys()) and (
                arg_dict["train_framework"] not in list(implemented_combos[arg_dict["algo"]].keys())):
            err = "{} is only implemented with {}".format(arg_dict["algo"],
                                                          list(implemented_combos[arg_dict["algo"]].keys())[0])
        elif arg_dict["algo"] not in implemented_combos.keys():
            err = "{} algorithm is not implemented.".format(arg_dict["algo"])
        else:
            err = "invalid model_path argument"
        raise Exception(err)

    images = []  # Empty list for GIF images
    video_path = None
    success_episodes_num = 0
    distance_error_sum = 0
    steps_sum = 0
    global done

    p.resetDebugVisualizerCamera(1.2, 180, -30, [0.0, 0.5, 0.05])
    model_name = arg_dict["algo"] + '_' + str(arg_dict["steps"])
    for e in range(arg_dict["eval_episodes"]):
        done = False
        obs, info = env.reset()
        # Optionally reposition goal object from CLI -pos
        if arg_dict.get("pos") is not None:
            apply_goal_position_override(env, arg_dict["pos"])
            # refresh observation after moving goal
        # Sync hardware joints with the environment's reset pose
        sync_hardware_pose_from_env(env, grasper)
        is_successful = 0
        distance_error = 0

        while not done:
            steps_sum += 1
            action, _state = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            is_successful = not info['f']
            distance_error = info['d']
            if arg_dict["vinfo"]:
                visualize_infotext(action, env, info)

            if arg_dict["record"] > 0 and len(images) < 8000:
                if video_path is None:
                    if arg_dict["record"] == 1:
                        video_path = make_path(arg_dict, ".gif", True)
                    elif arg_dict["record"] == 2:
                        video_path = make_path(arg_dict, ".webm", True)
                record_video(images, arg_dict, env, video_path, finalize=False)

        # After episode ends, if successful, print joint names controlled by action and their last values
        if is_successful:
            try:
                robot = getattr(env.unwrapped, 'robot', None)
                if robot is not None:
                    indices = getattr(robot, 'motor_indices', [])
                    names = []
                    values_print = []
                    for jid in indices:
                        ji = robot.p.getJointInfo(robot.robot_uid, jid)
                        nm = ji[1]
                        nm = nm.decode('utf-8') if isinstance(nm, (bytes, bytearray)) else str(nm)
                        jtype = ji[2]
                        q = robot.p.getJointState(robot.robot_uid, jid)[0]
                        if jtype == robot.p.JOINT_REVOLUTE:
                            values_print.append(round(float(q) * 57.29577951308232, 5))  # degrees
                            names.append(f"{nm} (deg)")
                        else:
                            values_print.append(round(float(q), 5))  # meters for prismatic or others
                            names.append(f"{nm} (m)")
                    print(f"Successful episode {e+1}:")
                    #print("Controlled joints:", names)
                    #print("Last joint values:", values_print)
                    try:
                        input("Press to execute on real robot...")
                        # Also push the final sim pose to hardware
                        if not sync_hardware_pose_from_env(env, grasper):
                            print("[WARN] No matching hardware joints found to command.")
                    except Exception:
                        pass
            except Exception as ex:
                print("[WARN] Could not print joint info:", ex)

        
        success_episodes_num += is_successful
        distance_error_sum += distance_error

    mean_distance_error = distance_error_sum / arg_dict["eval_episodes"]
    mean_steps_num = steps_sum // arg_dict["eval_episodes"]

    print("#---------Evaluation-Summary---------#")
    print("{} of {} episodes ({} %) were successful".format(success_episodes_num, arg_dict["eval_episodes"],
                                                            success_episodes_num / arg_dict["eval_episodes"] * 100))
    print("Mean distance error is {:.2f}%".format(mean_distance_error * 100))
    print("Mean number of steps {}".format(mean_steps_num))
    print("#------------------------------------#")

    file = open(os.path.join(model_logdir, "train_" + model_name + ".txt"), 'a')
    file.write("\n")
    file.write("#Evaluation results: \n")
    file.write("#{} of {} episodes were successful \n".format(success_episodes_num, arg_dict["eval_episodes"]))
    file.write("#Mean distance error is {:.2f}% \n".format(mean_distance_error * 100))
    file.write("#Mean number of steps {}\n".format(mean_steps_num))
    file.close()
    if arg_dict.get("record",0) > 0 and video_path is not None and len(images)>0:
        record_video(images, arg_dict, env, video_path, finalize=True)

def print_init_info(arg_dict):
    control = arg_dict.get("control")
    print("Path to the model using --model_path argument not specified. ")
    if control == "keyboard":
        print("Testing robot using keyboard control in selected environment.")
    elif control == "oraculum":
        print("Testing scenario feasibility using oraculum control in selected environment.")
    elif control == "slider":
        print("Testing robot joint control in selected environment using slider.")
    elif control == "observation":
        print("Testing robot control in selected environment using observation.")
    else:
        print("Testing random actions in selected environment.")


def main() -> None:
    """Main entry point for the testing script."""
    parser = get_parser()
    parser.add_argument("-ct", "--control", default="oraculum",
                        help="How to control robot during testing. Valid arguments: keyboard, observation, random, oraculum, slider")
    parser.add_argument("-vs", "--vsampling", action="store_true", help="Visualize sampling area.")
    parser.add_argument("-vt", "--vtrajectory", action="store_true", help="Visualize gripper trajectory.")
    parser.add_argument("-vn", "--vinfo", action="store_true", help="Visualize info. Valid arguments: True, False")
    parser.add_argument("-ns", "--network_switcher", default="gt", help="How does a robot switch to next network (gt or keyboard)")
    parser.add_argument("-rr", "--results_report", default = False, help="Used only with oraculum - shows report of task feasibility at the end.")
    parser.add_argument("-tp", "--top_grasp",  default = True, help="Use top grasp when reaching objects with oraculum.")
    # new: override goal position in env (x y z)
    parser.add_argument("-pos", "--pos", nargs=3, type=float, metavar=("X","Y","Z"), default=[0.3, 0.3, 0.2], help="Override goal object position (meters)")
    # parser.add_argument("-nl", "--natural_language", default=False, help="NL Valid arguments: True, False")
    arg_dict, commands = get_arguments(parser)
    parameters = {}
    args = parser.parse_args()
        
    for key, arg in arg_dict.items():
        if type(arg_dict[key]) == list:
            if len(arg_dict[key]) > 1 and key != "robot_init" and key != "end_effector_orn":
                if key != "task_objects":
                    parameters[key] = arg
                    if key in commands:
                        commands.pop(key)
    

    # Check if we chose one of the existing engines
    if arg_dict["engine"] not in AVAILABLE_SIMULATION_ENGINES:
        print(f"Invalid simulation engine. Valid arguments: --engine {AVAILABLE_SIMULATION_ENGINES}.")
        return
    else:
        if arg_dict["results_report"]:
            print("Results report cannot be used without oraculum.")
            arg_dict["results_report"] = False
    if arg_dict.get("pretrained_model") is None:
        print_init_info(arg_dict)
        arg_dict["gui"] = 1
        arg_dict = automatic_argument_assignment(arg_dict)
        arg_dict["robot_action"] = "absolute_gripper"
        env = configure_env(arg_dict, model_logdir=None, for_train=0)
        test_env(env, arg_dict)
    else:
        #arg_dict["robot_action"] = "joints_gripper" #Model has to be tested with this action type
        model_logdir = os.path.dirname(arg_dict.get("pretrained_model", ""))
        env = configure_env(arg_dict, model_logdir, for_train=0)
        implemented_combos = configure_implemented_combos(env, model_logdir, arg_dict)
        print(model_logdir)
        test_model(env, None, implemented_combos, arg_dict, model_logdir, deterministic=False)


if __name__ == "__main__":
    main()
