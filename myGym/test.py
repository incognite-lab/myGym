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

from myGym import oraculum
from myGym.train import get_parser, get_arguments, configure_implemented_combos, configure_env, automatic_argument_assignment

clear = lambda: os.system('clear')

AVAILABLE_SIMULATION_ENGINES = ["mujoco", "pybullet"]
AVAILABLE_TRAINING_FRAMEWORKS = ["tensorflow", "pytorch"]


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


def test_env(env: object, arg_dict: dict) -> None:
    obs, info = env.reset()
    results = pd.DataFrame(columns = ["Task type", "Workspace", "Robot", "Gripper init", "Object init", "Object goal", "Success"])
    current_result = None
    env.render()
    global done
    # Prepare names for sliders
    joints = [f"Joint{i}" for i in range(1, 20)]
    jointparams = [f"Jnt{i}" for i in range(1, 20)]

    images = []
    video_path = None
    action = None
    info = None
    grip_limits = env.unwrapped.robot.gjoints_limits
    oraculum_obj = oraculum.Oraculum(env, info, arg_dict["robot_action"], grip_limits)

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(1.2, 180, -30, [0.0, 0.5, 0.05])
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    last_call_time = time.time()
    if arg_dict["control"] == "slider":
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        if "joints" in arg_dict["robot_action"]:
            if 'gripper' in arg_dict["robot_action"]:
                print("gripper is present")
                for i in range(env.unwrapped.action_space.shape[0]):
                    if i < (env.unwrapped.action_space.shape[0] - len(env.unwrapped.robot.gjoints_rest_poses)):
                        joints[i] = p.addUserDebugParameter(joints[i], env.unwrapped.action_space.low[i],
                                                            env.unwrapped.action_space.high[i], env.unwrapped.robot.init_joint_poses[i])
                    else:
                        joints[i] = p.addUserDebugParameter(joints[i], env.unwrapped.action_space.low[i],
                                                            env.unwrapped.action_space.high[i], .02)
            else:
                for i in range(env.unwrapped.action_space.shape[0]):
                    joints[i] = p.addUserDebugParameter(joints[i], env.unwrapped.action_space.low[i], env.unwrapped.action_space.high[i],
                                                        env.unwrapped.robot.init_joint_poses[i])
        elif "absolute" in arg_dict["robot_action"]:
            if 'gripper' in arg_dict["robot_action"]:
                print("gripper is present")
                for i in range(env.action_space.shape[0]):
                    if i < (env.action_space.shape[0] - len(env.unwrapped.robot.gjoints_rest_poses)):
                        joints[i] = p.addUserDebugParameter(joints[i], -1, 1, arg_dict["robot_init"][i])
                    else:
                        joints[i] = p.addUserDebugParameter(joints[i], -1, 1, .02)
            else:
                for i in range(env.action_space.shape[0]):
                    joints[i] = p.addUserDebugParameter(joints[i], -1, 1, arg_dict["robot_init"][i])
        elif "step" in arg_dict["robot_action"]:
            if 'gripper' in arg_dict["robot_action"]:
                print("gripper is present")
                for i in range(env.action_space.shape[0]):
                    if i < (env.action_space.shape[0] - len(env.unwrapped.robot.gjoints_rest_poses)):
                        joints[i] = p.addUserDebugParameter(joints[i], -1, 1, 0)
                    else:
                        joints[i] = p.addUserDebugParameter(joints[i], -1, 1, .02)
            else:
                for i in range(env.action_space.shape[0]):
                    joints[i] = p.addUserDebugParameter(joints[i], -1, 1, 0)

    p.addUserDebugParameter("Lateral Friction", 0, 100, 0)
    p.addUserDebugParameter("Spinning Friction", 0, 100, 0)
    p.addUserDebugParameter("Linear Damping", 0, 100, 0)
    p.addUserDebugParameter("Angular Damping", 0, 100, 0)

    if arg_dict["vsampling"]:
        visualize_sampling_area(arg_dict)
    if arg_dict["control"] == "random":
        action = env.action_space.sample()
    if arg_dict["control"] == "keyboard":
        action = arg_dict["robot_init"]
        if "gripper" in arg_dict["robot_action"]:
            action.append(.1)
            action.append(.1)
    if arg_dict["control"] == "slider":
        action = []
        for i in range(env.action_space.shape[0]):
            jointparams[i] = p.readUserDebugParameter(joints[i])
            action.append(jointparams[i])

    eval_episodes = arg_dict.get("eval_episodes", 50)
    for e in range(eval_episodes):
        obs, info = env.reset()
        # Store oraculum results if selected:
        if arg_dict["results_report"]:
            if len(arg_dict["task_type"]) <=2: #A or AG task
                positions = [info['o']['actual_state'],  None,
                              info['o']['goal_state']]
            else:
                positions = [info['o']["additional_obs"]["endeff_xyz"],
                             info['o']['actual_state'],
                             info['o']['goal_state']]
            current_result = [arg_dict["task_type"], arg_dict["workspace"], arg_dict["robot"],
                            np.round(np.array(positions[0]), 2) if positions[0] is not None else None,
                            np.round(np.array(positions[1]), 2) if positions[1] is not None else None,
                            np.round(np.array(positions[2]), 2) if positions[2] is not None else None]
        for t in range(arg_dict["max_episode_steps"]):
            if arg_dict["control"] == "slider":
                action = []
                for i in range(env.action_space.shape[0]):
                    jointparams[i] = p.readUserDebugParameter(joints[i])
                    action.append(jointparams[i])

            if arg_dict["control"] == "observation":
                if t == 0:
                    action = env.unwrapped.robot.init_joint_poses
                    last_action = action
                else:
                    if "joints" in arg_dict["robot_action"]:
                        action = env.unwrapped.robot.get_joints_states()
                    elif "absolute" in arg_dict["robot_action"]:
                        action = info['o']["actual_state"]
                    else:
                        action = [0, 0, 0]

                    # Compute element-wise difference safely using numpy
                    try:
                        diff = np.array(last_action, dtype=float) - np.array(action, dtype=float)
                        print(f"\rLast action difference: {np.round(diff, 4)}", end='', flush=True)
                    except Exception:
                        # Fallback if shapes mismatch
                        print("\rLast action difference: (shape mismatch)", end='', flush=True)

                    last_action = action

            if arg_dict["control"] == "oraculum":
                action = oraculum_obj.perform_oraculum_task(t, env, action, info)
            elif arg_dict["control"] == "keyboard":
                keypress = p.getKeyboardEvents()
                action = detect_key(keypress, arg_dict, action)
            elif arg_dict["control"] == "random":
                action = env.action_space.sample()

            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            #print("observation shape:", len(obs))

            n_p, last_call_time = n_pressed(last_call_time)
            if n_p:  # If key 'n' is pressed, switch to next task - useful if robot gets stuck
                env.unwrapped.task.end_episode_fail("manual_switch")
                done = True

            if arg_dict["results_report"] and done:
                if terminated:
                    current_result.append(True)
                elif truncated:
                    current_result.append(False)
                else:
                    current_result.append(False)
                results.loc[len(results)] = current_result #Append result to pd dataframe

            if arg_dict["vtrajectory"]:
                visualize_trajectories(info, action)
            if arg_dict["vinfo"]:
                visualize_infotext(action, env, info)

            if "step" in arg_dict["robot_action"]:
                action[:3] = [0, 0, 0]

            if arg_dict["visualize"]:
                visualizations = [[], []]
                env.render()
                for camera_id in range(arg_dict["camera"]):
                    camera_render = env.render()
                    image = cv2.cvtColor(camera_render[camera_id]["image"], cv2.COLOR_RGB2BGR)
                    depth = camera_render[camera_id]["depth"]
                    image = cv2.copyMakeBorder(image, 30, 10, 10, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                    cv2.putText(image, 'Camera {}'.format(camera_id), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .5,
                                (0, 0, 0), 1, 0)
                    visualizations[0].append(image)
                    depth = cv2.copyMakeBorder(depth, 30, 10, 10, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                    cv2.putText(depth, 'Camera {}'.format(camera_id), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .5,
                                (0, 0, 0), 1, 0)
                    visualizations[1].append(depth)

                if len(visualizations[0]) % 2 != 0:
                    visualizations[0].append(255 * np.ones(visualizations[0][0].shape, dtype=np.uint8))
                    visualizations[1].append(255 * np.ones(visualizations[1][0].shape, dtype=np.float32))
                fig_rgb = np.vstack((np.hstack((visualizations[0][0::2])), np.hstack((visualizations[0][1::2]))))
                fig_depth = np.vstack((np.hstack((visualizations[1][0::2])), np.hstack((visualizations[1][1::2]))))
                cv2.imshow('Camera RGB renders', fig_rgb)
                cv2.imshow('Camera depth renders', fig_depth)
                cv2.waitKey(1)

            if arg_dict["record"] > 0 and len(images) < 80000:
                if video_path is None:
                    if arg_dict["record"] == 1:
                        video_path = make_path(arg_dict, ".gif", False)
                    elif arg_dict["record"] == 2:
                        video_path = make_path(arg_dict, ".webm", False)
                record_video(images, arg_dict, env, video_path, finalize=False)

            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    if arg_dict["results_report"]:
        # results = results.round(2)
        i=1
        print(results.dtypes)
        print(results)
        print(type(results))
        while True:
            filename = f"./oraculum_results/results{i}.csv"
            if not(os.path.exists(filename)):
                break
            i+=1
        results.to_csv(filename, index = False)
    if arg_dict.get("record",0) > 0 and video_path is not None and len(images)>0:
        record_video(images, arg_dict, env, video_path, finalize=True)


def record_video(images: list, arg_dict: dict, env: object, path: str, finalize: bool=False) -> None:
    if arg_dict["camera"] < 1:
        raise ValueError("Camera parameter must be set to > 0 to record!")

    render_info = env.render()
    image = render_info[arg_dict["camera"] - 1]["image"]
    images.append(image)
    if not finalize:
        print(f"\rRecording frame: {len(images)} - ", end='', flush=True)
        return
    # Finalize and write video
    print()  # newline after progress line
    print(f"Finalizing video with {len(images)} frames -> {path}")
    if ".gif" in path:
        imageio.mimsave(path, [np.array(img) for i, img in enumerate(images) if i % 2 == 0], duration=65)
        os.system('./utils/gifopt -O3 --lossy=5 --colors 256 -o {dest} {source}'.format(source=path, dest=path))
        print("Record saved to " + path)
        return
    # Ensure webm extension
    if not path.endswith('.webm'):
        base = os.path.splitext(path)[0]
        path = base + '.webm'
    height, width, _ = images[0].shape
    codec_chain = [
        ['-c:v', 'libvpx-vp9', '-b:v', '2M'],
        ['-c:v', 'libvpx', '-b:v', '2M']
    ]
    for codec_args in codec_chain:
        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-s', f'{width}x{height}',
            '-r', '30',
            '-i', '-',
            '-an',
        ] + codec_args + ['-pix_fmt', 'yuv420p', path]
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            print('[ERROR] ffmpeg not found in PATH. Cannot save video.')
            return
        try:
            for frame in images:
                f = frame
                if f.dtype != np.uint8:
                    f = np.clip(f, 0, 255).astype(np.uint8)
                proc.stdin.write(f.tobytes())
        finally:
            if proc.stdin:
                proc.stdin.close()
            proc.wait()
        if proc.returncode == 0:
            print(f'Record saved to {path} ({codec_args[1]})')
            break
        else:
            print(f'[WARN] ffmpeg failed with codec {codec_args[1]}, trying fallback...')
    else:
        print('[ERROR] All webm codecs failed; no video saved.')


def make_path(arg_dict: dict, record_format: str, model: bool):
    counter = 0
    if model:
        model_logdir = os.path.dirname(arg_dict.get("logdir", ""))
        model_name = str(arg_dict["robot"]) + '_' + str(arg_dict["task_type"]) + '_' + str(arg_dict["robot_action"]) + '_' + str(arg_dict["algo"])
        logdir = os.path.join(model_logdir,model_name)
        print("Saving to " + logdir)
    else:
        if not os.path.exists(arg_dict["logdir"]):
            os.makedirs(arg_dict["logdir"])
        logdir = os.path.join(arg_dict["logdir"], "train_record_" + arg_dict["task_type"])

    video_path = logdir + "_" + str(counter) + record_format
    while os.path.exists(video_path):
        counter += 1
        video_path = logdir + "_" + str(counter) + record_format

    return video_path


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
    parser.add_argument("-ct", "--control",
                        help="How to control robot during testing. Valid arguments: keyboard, observation, random, oraculum, slider")
    parser.add_argument("-vs", "--vsampling", action="store_true", help="Visualize sampling area.")
    parser.add_argument("-vt", "--vtrajectory", action="store_true", help="Visualize gripper trajectory.")
    parser.add_argument("-vn", "--vinfo", action="store_true", help="Visualize info. Valid arguments: True, False")
    parser.add_argument("-ns", "--network_switcher", default="gt", help="How does a robot switch to next network (gt or keyboard)")
    parser.add_argument("-rr", "--results_report", default = False, help="Used only with oraculum - shows report of task feasibility at the end.")
    parser.add_argument("-tp", "--top_grasp",  default = False, help="Use top grasp when reaching objects with oraculum.")
    # parser.add_argument("-nl", "--natural_language", default=False, help="NL Valid arguments: True, False")
    arg_dict, commands = get_arguments(parser)
    parameters = {}
    args = parser.parse_args()

    from myGym.envs.robot import ROSRobot, Robot
    Robot.shim_to(ROSRobot)  # swap Robot for ROSRobot class

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
