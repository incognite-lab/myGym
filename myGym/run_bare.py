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

from myGym import oraculum
from myGym.envs.robot import Robot
from myGym.trainSB3 import get_parser, get_arguments, configure_implemented_combos, configure_env

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
        (f"Network: {env.env.reward.current_network}", [0.71, 1., 0.3], [0.0, 0.0, 1]),
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
        113: [(0, 0)]  # Q
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


def test_env(env: object, arg_dict: dict) -> None:
    env.reset()
    env.render()
    # Prepare names for sliders
    joints = [f"Joint{i}" for i in range(1, 20)]
    jointparams = [f"Jnt{i}" for i in range(1, 20)]

    action = None
    info = None

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(1.2, 180, -30, [0.0, 0.5, 0.05])
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    if arg_dict["control"] == "slider":
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        if "joints" in arg_dict["robot_action"]:
            if 'gripper' in arg_dict["robot_action"]:
                print("gripper is present")
                for i in range(env.action_space.shape[0]):
                    if i < (env.action_space.shape[0] - len(env.env.robot.gjoints_rest_poses)):
                        joints[i] = p.addUserDebugParameter(joints[i], env.action_space.low[i],
                                                            env.action_space.high[i], env.env.robot.init_joint_poses[i])
                    else:
                        joints[i] = p.addUserDebugParameter(joints[i], env.action_space.low[i],
                                                            env.action_space.high[i], .02)
            else:
                for i in range(env.action_space.shape[0]):
                    joints[i] = p.addUserDebugParameter(joints[i], env.action_space.low[i], env.action_space.high[i],
                                                        env.env.robot.init_joint_poses[i])
        elif "absolute" in arg_dict["robot_action"]:
            if 'gripper' in arg_dict["robot_action"]:
                print("gripper is present")
                for i in range(env.action_space.shape[0]):
                    if i < (env.action_space.shape[0] - len(env.env.robot.gjoints_rest_poses)):
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
                    if i < (env.action_space.shape[0] - len(env.env.robot.gjoints_rest_poses)):
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
        env.reset()
        for t in range(arg_dict["max_episode_steps"]):
            if arg_dict["control"] == "slider":
                action = []
                for i in range(env.action_space.shape[0]):
                    jointparams[i] = p.readUserDebugParameter(joints[i])
                    action.append(jointparams[i])

            if arg_dict["control"] == "observation":
                if t == 0:
                    action = env.action_space.sample()
                else:
                    if "joints" in arg_dict["robot_action"]:
                        action = info['o']["additional_obs"]["joints_angles"]
                    elif "absolute" in arg_dict["robot_action"]:
                        action = info['o']["actual_state"]
                    else:
                        action = [0, 0, 0]

            if arg_dict["control"] == "oraculum":
                action = oraculum.perform_oraculum_task(t, env, arg_dict, action, info)
            elif arg_dict["control"] == "keyboard":
                keypress = p.getKeyboardEvents()
                action = detect_key(keypress, arg_dict, action)
            elif arg_dict["control"] == "random":
                action = env.action_space.sample()

            observation, reward, done, _, info = env.step(action)
            if arg_dict["vtrajectory"]:
                visualize_trajectories(info, action)
            if arg_dict["vinfo"]:
                visualize_infotext(action, env, info)

            if "step" in arg_dict["robot_action"]:
                action[:3] = [0, 0, 0]

            if arg_dict["visualize"]:
                visualizations = [[], []]
                env.render()
                for camera_id in range(len(env.env.cameras)):
                    # cannot set render mode?
                    camera_render = env.render(mode="rgb_array", camera_id=camera_id)
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

            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break


def test_model(
        env: Any,
        model=None,
        implemented_combos: Dict[str, Any] = None,
        arg_dict: Dict[str, Any] = None,
        model_logdir: str = None,
        deterministic: bool = False
) -> None:
    try:
        if "multi" in arg_dict["algo"]:
            model_args = implemented_combos[arg_dict["algo"]][arg_dict["train_framework"]][1]
            #print("arg_dict model path:", arg_dict["model_path"])
            model = implemented_combos[arg_dict["algo"]][arg_dict["train_framework"]][0].load(arg_dict["model_path"])
            model.env = model_args[1].env
        else:
            model = implemented_combos[arg_dict["algo"]][arg_dict["train_framework"]][0].load(arg_dict["model_path"])
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
    success_episodes_num = 0
    distance_error_sum = 0
    steps_sum = 0

    p.resetDebugVisualizerCamera(1.2, 180, -30, [0.0, 0.5, 0.05])

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
            if arg_dict["vinfo"] == True:
                visualize_infotext(action, env, info)

            if (arg_dict["record"] > 0) and (len(images) < 8000):
                render_info = env.render(mode="rgb_array", camera_id=arg_dict["camera"])
                image = render_info[arg_dict["camera"]]["image"]
                images.append(image)
                print(f"appending image: total size: {len(images)}]")

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
    model_name = arg_dict["algo"] + '_' + str(arg_dict["steps"])
    file = open(os.path.join(model_logdir, "train_" + model_name + ".txt"), 'a')
    file.write("\n")
    file.write("#Evaluation results: \n")
    file.write("#{} of {} episodes were successful \n".format(success_episodes_num, arg_dict["eval_episodes"]))
    file.write("#Mean distance error is {:.2f}% \n".format(mean_distance_error * 100))
    file.write("#Mean number of steps {}\n".format(mean_steps_num))
    file.close()

    if arg_dict["record"] == 1:
        gif_path = os.path.join(model_logdir, "train_" + model_name + ".gif")
        imageio.mimsave(gif_path, [np.array(img) for i, img in enumerate(images) if i % 2 == 0], fps=15)
        os.system('./utils/gifopt -O3 --lossy=5 -o {dest} {source}'.format(source=gif_path, dest=gif_path))
        print("Record saved to " + gif_path)
    elif arg_dict["record"] == 2:
        video_path = os.path.join(model_logdir, "train_" + model_name + ".avi")
        height, width, layers = image.shape
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))
        for img in images:
            out.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        out.release()
        print("Record saved to " + video_path)


def multi_test(params: Dict[str, Any], arg_dict: Dict[str, Any], configfile: str, commands: Dict[str, Any]) -> None:
    """Execute a multi-test command with the specified parameters.
        Args:
            params (Dict[str, Any]): Parameters for the test.
            arg_dict (Dict[str, Any]): Argument dictionary for configuration.
            configfile (str): Path to the configuration file.
            commands (Dict[str, Any]): Additional command options.
    """
    logdirfile = arg_dict["logdir"]
    print((" ".join(f"--{key} {value}" for key, value in params.items())).split())
    command = (
            f"python testSB3.py --config {configfile} --logdir {logdirfile} "
            + " ".join(f"--{key} {value}" for key, value in params.items()) + " "
            + " ".join(
        f"--{key} {' '.join(map(str, value)) if isinstance(value, list) else value}" for key, value in commands.items())
    )
    print(command)

    # use this if you want all the prints in terminal + file
    # with open("test.log", "wb") as f:
    #     process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    #     for c in iter(lambda: process.stdout.read(1), b''):
    #         sys.stdout.buffer.write(c)
    #         f.write(c)

    # use this if you don't want prints from threads
    subprocess.check_output(command.split())


def multi_main(arg_dict: Dict[str, Any], parameters: Dict[str, Any], configfile: str, commands: Dict[str, Any]) -> None:
    """Manage the execution of multi-test commands, either in threads or sequentially.

    Args:
        arg_dict (Dict[str, Any]): Argument dictionary for configuration.
        parameters (Dict[str, Any]): Parameter grid for testing.
        configfile (str): Path to the configuration file.
        commands (Dict[str, Any]): Additional command options.
    """
    parameter_grid = ParameterGrid(parameters)

    threaded = arg_dict["threaded"]
    threads = []

    start_time = time.time()
    for i, params in enumerate(parameter_grid):
        if threaded:
            print(f"Thread {i + 1} starting")
            thread = multiprocessing.Process(target=multi_test, args=(params, arg_dict, configfile, commands))
            thread.start()
            threads.append(thread)
        else:
            multi_test(params.copy(), arg_dict, configfile, commands)
    if threaded:
        for i, thread in enumerate(threads):
            thread.join()
            print(f"Thread {i + 1} finishing")

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")


def main() -> None:
    from myGym.envs.robot import RobotClassShim, ROSRobot
    RobotClassShim.shim_to(ROSRobot)

    """Main entry point for the testing script."""
    parser = get_parser()
    parser.add_argument("-ct", "--control", default="random",
                        help="How to control robot during testing. Valid arguments: keyboard, observation, random, oraculum, slider")
    parser.add_argument("-vs", "--vsampling", action="store_true", help="Visualize sampling area.")
    parser.add_argument("-vt", "--vtrajectory", action="store_true", help="Visualize gripper trajectory.")
    parser.add_argument("-vn", "--vinfo", action="store_true", help="Visualize info. Valid arguments: True, False")
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

    model_logdir = os.path.dirname(arg_dict.get("model_path", ""))
    # Check if we chose one of the existing engines
    if arg_dict["engine"] not in AVAILABLE_SIMULATION_ENGINES:
        print(f"Invalid simulation engine. Valid arguments: --engine {AVAILABLE_SIMULATION_ENGINES}.")
        return

    if parameters:
        print("THREADING")
        multi_main(arg_dict, parameters, args.config, commands)

    arg_dict["gui"] = 1
    env = configure_env(arg_dict, model_logdir, for_train=0)
    test_env(env, arg_dict)


if __name__ == "__main__":
    main()
