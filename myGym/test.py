import multiprocessing
import os
import subprocess
import sys
import time

import cv2
import imageio
import numpy as np
import pybullet as p
import pybullet_data
from numpy import matrix
from sklearn.model_selection import ParameterGrid

from myGym.train import get_parser, get_arguments, configure_implemented_combos, configure_env

clear = lambda: os.system('clear')

AVAILABLE_SIMULATION_ENGINES = ["pybullet"]
AVAILABLE_TRAINING_FRAMEWORKS = ["tensorflow"]


def visualize_sampling_area(arg_dict):
    rx = (arg_dict["task_objects"][0]["goal"]["sampling_area"][0] -
          arg_dict["task_objects"][0]["goal"]["sampling_area"][1]) / 2
    ry = (arg_dict["task_objects"][0]["goal"]["sampling_area"][2] -
          arg_dict["task_objects"][0]["goal"]["sampling_area"][3]) / 2
    rz = (arg_dict["task_objects"][0]["goal"]["sampling_area"][4] -
          arg_dict["task_objects"][0]["goal"]["sampling_area"][5]) / 2

    visual = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[rx, ry, rz], rgbaColor=[1, 0, 0, .2])
    collision = -1

    sampling = p.createMultiBody(
        baseVisualShapeIndex=visual,
        baseCollisionShapeIndex=collision,
        baseMass=0,
        basePosition=[arg_dict["task_objects"][0]["goal"]["sampling_area"][0] - rx,
                      arg_dict["task_objects"][0]["goal"]["sampling_area"][2] - ry,
                      arg_dict["task_objects"][0]["goal"]["sampling_area"][4] - rz],
    )


def visualize_trajectories(info, action):
    visualo = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[0, 0, 1, .3])
    collision = -1
    p.createMultiBody(
        baseVisualShapeIndex=visualo,
        baseCollisionShapeIndex=collision,
        baseMass=0,
        basePosition=info['o']['actual_state'],
    )

    visuala = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[1, 0, 0, .3])
    p.createMultiBody(
        baseVisualShapeIndex=visuala,
        baseCollisionShapeIndex=collision,
        baseMass=0,
        basePosition=action[:3],
    )


def visualize_goal(info):
    visualg = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[1, 0, 0, .5])
    collision = -1
    p.createMultiBody(
        baseVisualShapeIndex=visualg,
        baseCollisionShapeIndex=collision,
        baseMass=0,
        basePosition=info['o']['goal_state'],
    )


def change_dynamics(cubex, lfriction, rfriction, ldamping, adamping):
    p.changeDynamics(cubex, -1, lateralFriction=p.readUserDebugParameter(lfriction))
    p.changeDynamics(cubex, -1, rollingFriction=p.readUserDebugParameter(rfriction))
    p.changeDynamics(cubex, -1, linearDamping=p.readUserDebugParameter(ldamping))
    p.changeDynamics(cubex, -1, angularDamping=p.readUserDebugParameter(adamping))


def visualize_infotext(action, env, info):
    p.addUserDebugText(f"Episode:{env.env.episode_number}",
                       [.65, 1., 0.45], textSize=1.0, lifeTime=0.5, textColorRGB=[0.4, 0.2, .3])
    p.addUserDebugText(f"Step:{env.env.episode_steps}",
                       [.67, 1, .40], textSize=1.0, lifeTime=0.5, textColorRGB=[0.2, 0.8, 1])
    p.addUserDebugText(f"Subtask:{env.env.task.current_task}",
                       [.69, 1, 0.35], textSize=1.0, lifeTime=0.5, textColorRGB=[0.4, 0.2, 1])
    p.addUserDebugText(f"Network:{env.env.reward.current_network}",
                       [.71, 1, 0.3], textSize=1.0, lifeTime=0.5, textColorRGB=[0.0, 0.0, 1])
    p.addUserDebugText(f"Action (Gripper):{matrix(np.around(np.array(action), 2))}",
                       [.73, 1, 0.25], textSize=1.0, lifeTime=0.5, textColorRGB=[1, 0, 0])
    p.addUserDebugText(
        f"Actual_state:{matrix(np.around(np.array(env.env.observation['task_objects']['actual_state'][:3]), 2))}",
        [.75, 1, 0.2], textSize=1.0, lifeTime=0.5, textColorRGB=[0.0, 1, 0.0])
    p.addUserDebugText(f"End_effector:{matrix(np.around(np.array(env.env.robot.end_effector_pos), 2))}",
                       [.77, 1, 0.15], textSize=1.0, lifeTime=0.5, textColorRGB=[0.0, 1, 0.0])
    p.addUserDebugText(f"        Object:{matrix(np.around(np.array(info['o']['actual_state']), 2))}",
                       [.8, 1, 0.10], textSize=1.0, lifeTime=0.5, textColorRGB=[0.0, 0.0, 1])
    p.addUserDebugText(f"Velocity:{env.env.max_velocity}",
                       [.79, 1, 0.05], textSize=1.0, lifeTime=0.5, textColorRGB=[0.6, 0.8, .3])
    p.addUserDebugText(f"Force:{env.env.max_force}",
                       [.81, 1, 0.00], textSize=1.0, lifeTime=0.5, textColorRGB=[0.3, 0.2, .4])


def detect_key(keypress, arg_dict, action):
    if 97 in keypress.keys() and keypress[97] == 1:  # A
        action[2] += .03
    if 122 in keypress.keys() and keypress[122] == 1:  # Z/Y
        action[2] -= .03
    if 65297 in keypress.keys() and keypress[65297] == 1:  # ARROW UP
        action[1] -= .03
    if 65298 in keypress.keys() and keypress[65298] == 1:  # ARROW DOWN
        action[1] += .03
    if 65295 in keypress.keys() and keypress[65295] == 1:  # ARROW LEFT
        action[0] += .03
    if 65296 in keypress.keys() and keypress[65296] == 1:  # ARROW RIGHT
        action[0] -= .03
    if 120 in keypress.keys() and keypress[120] == 1:  # X
        action[3] -= .03
        action[4] -= .03
    if 99 in keypress.keys() and keypress[99] == 1:  # C
        action[3] += .03
        action[4] += .03

    if "step" in arg_dict["robot_action"]:
        action[:3] = np.multiply(action[:3], 10)
    elif "joints" in arg_dict["robot_action"]:
        print("Robot action: Joints - KEYBOARD CONTROL UNDER DEVELOPMENT")
        quit()
    return action


def test_env(env, arg_dict):
    spawn_objects = False
    env.render("human")
    # Prepare names for sliders
    joints = ['Joint1', 'Joint2', 'Joint3', 'Joint4', 'Joint5', 'Joint6', 'Joint7', 'Joint 8', 'Joint 9', 'Joint10',
              'Joint11', 'Joint12', 'Joint13', 'Joint14', 'Joint15', 'Joint16', 'Joint17', 'Joint 18', 'Joint 19']
    jointparams = ['Jnt1', 'Jnt2', 'Jnt3', 'Jnt4', 'Jnt5', 'Jnt6', 'Jnt7', 'Jnt 8', 'Jnt 9', 'Jnt10', 'Jnt11', 'Jnt12',
                   'Jnt13', 'Jnt14', 'Jnt15', 'Jnt16', 'Jnt17', 'Jnt 18', 'Jnt 19']
    cube = ['Cube1', 'Cube2', 'Cube3', 'Cube4', 'Cube5', 'Cube6', 'Cube7', 'Cube8', 'Cube9', 'Cube10', 'Cube11',
            'Cube12', 'Cube13', 'Cube14', 'Cube15', 'Cube16', 'Cube17', 'Cube18', 'Cube19']
    cubecount = 0

    if arg_dict["gui"] == 0:
        print("Add --gui 1 parameter to visualize environment")
        quit()

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

    for e in range(50):
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
                if t == 0:
                    action = env.action_space.sample()
                else:
                    if "absolute" in arg_dict["robot_action"]:
                        if env.env.reward.reward_name == "approach":
                            try:
                                if env.env.reward.rewards_num <= 2:
                                    action[:3] = info['o']["goal_state"][:3]
                                else:
                                    action[:3] = info['o']["actual_state"][:3]
                            except:
                                try:
                                    action[:3] = info['o']["actual_state"][:3]
                                except:
                                    action[:3] = info['o']["goal_state"][:3]
                            if "gripper" in arg_dict["robot_action"]:
                                action[3] = 1
                                action[4] = 1
                        elif env.env.reward.reward_name == "grasp":
                            if "gripper" in arg_dict["robot_action"]:
                                action[3] = 0
                                action[4] = 0
                        elif env.env.reward.reward_name == "move":
                            action[:3] = info['o']["goal_state"][:3]
                            if "gripper" in arg_dict["robot_action"]:
                                action[3] = 0
                                action[4] = 0
                        elif env.env.reward.reward_name == "drop":
                            if "gripper" in arg_dict["robot_action"]:
                                action[3] = 1
                                action[4] = 1
                        elif env.env.reward.reward_name == "withdraw":
                            action[:3] = np.array(info['o']["actual_state"][:3]) + np.array([0, 0, 0.4])
                            if "gripper" in arg_dict["robot_action"]:
                                action[3] = 1
                                action[4] = 1
                        else:
                            #Old rewards -> reward name is None
                            action[:3] = info['o']["goal_state"][:3]
                    else:
                        print("ERROR - Oraculum mode only works for absolute actions")
                        quit()

            elif arg_dict["control"] == "keyboard":
                keypress = p.getKeyboardEvents()
                action = detect_key(keypress, arg_dict, action)

            elif arg_dict["control"] == "random":
                action = env.action_space.sample()

            deg = np.rad2deg(action)
            observation, reward, done, info = env.step(action)

            if arg_dict["vtrajectory"] == True:
                visualize_trajectories(info, action)
            if arg_dict["vinfo"] == True:
                visualize_infotext(action, env, info)
            print(
                "Reward: {}  \n Observation: {} \n EnvObservation: {}".format(reward, observation, env.env.observation))

            if "step" in arg_dict["robot_action"]:
                action[:3] = [0, 0, 0]

            if arg_dict["visualize"]:
                visualizations = [[], []]
                env.render("human")
                for camera_id in range(len(env.cameras)):
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
                cv2.imshow('Camera depthrenders', fig_depth)
                cv2.waitKey(1)

            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break


def test_model(env, model=None, implemented_combos=None, arg_dict=None, model_logdir=None, deterministic=False):
    try:
        if "multi" in arg_dict["algo"]:
            model_args = implemented_combos[arg_dict["algo"]][arg_dict["train_framework"]][1]
            model = implemented_combos[arg_dict["algo"]][arg_dict["train_framework"]][0].load(arg_dict["model_path"],
                                                                                              env=model_args[1].env)
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

    images = []  # Empty list for gif images
    success_episodes_num = 0
    distance_error_sum = 0
    vel = arg_dict["max_velocity"]
    force = arg_dict["max_force"]
    steps_sum = 0
    p.resetDebugVisualizerCamera(1.2, 180, -30, [0.0, 0.5, 0.05])

    for e in range(arg_dict["eval_episodes"]):
        done = False
        obs = env.reset()
        is_successful = 0
        distance_error = 0
        while not done:
            steps_sum += 1
            action, _state = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            is_successful = not info['f']
            distance_error = info['d']
            if arg_dict["vinfo"]:
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
    sys.stdout.flush()
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


def multi_test(params, arg_dict, configfile, commands):
    logdirfile = arg_dict["logdir"]
    print((" ".join(f"--{key} {value}" for key, value in params.items())).split())
    command = (
            f"python test.py --config {configfile} --logdir {logdirfile} "
            + " ".join(f"--{key} {value}" for key, value in params.items()) + " "
            + " ".join(f"--{key} {' '.join(map(str, value)) if isinstance(value, list) else value}" for key, value in commands.items())
    )
    print(command)
    # use this if you want all the prints in terminal + file
    with open("test.log", "wb") as f:
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        for c in iter(lambda: process.stdout.read(1), b''):
            sys.stdout.buffer.write(c)
            f.write(c)

    # use this if you don't want prints from threads
    # subprocess.check_output(command.split())


def multi_main(arg_dict, parameters, configfile, commands):
    parameter_grid = ParameterGrid(parameters)

    threaded = arg_dict["threaded"]
    threads = []

    start_time = time.time()
    for i, params in enumerate(parameter_grid):
        if threaded:
            print("Thread ", i + 1, " starting")
            thread = multiprocessing.Process(target=multi_test, args=(params, arg_dict, configfile, commands))
            thread.start()
            threads.append(thread)
        else:
            multi_test(params.copy(), arg_dict, configfile, commands)
    if threaded:
        i = 0
        for thread in threads:
            thread.join()
            print("Thread ", i + 1, " finishing")
            i += 1

    end_time = time.time()
    print(end_time - start_time)


def main():
    parser = get_parser()
    parser.add_argument("-ct", "--control", default="slider",
                        help="How to control robot during testing. Valid arguments: keyboard, observation, random, oraculum, slider")
    parser.add_argument("-vs", "--vsampling", action="store_true", help="Visualize sampling area.")
    parser.add_argument("-vt", "--vtrajectory", action="store_true", help="Visualize gripper trajectgory.")
    parser.add_argument("-vn", "--vinfo", action="store_true", help="Visualize info. Valid arguments: True, False")
    parser.add_argument("-nl", "--natural_language", default=False, help="NL Valid arguments: True, False")

    arg_dict, commands = get_arguments(parser)
    parameters = {}
    args = parser.parse_args()

    for key, arg in arg_dict.items():
        if type(arg_dict[key]) == list:
            if len(arg_dict[key]) > 1 and key != "robot_init":
                if key != "task_objects":
                    parameters[key] = arg
                    if key in commands:
                        commands.pop(key)

    # # debug info
    # with open("arg_dict_test", "w") as f:
    #     f.write("ARG DICT: ")
    #     f.write(str(arg_dict))
    #     f.write("\n")
    #     f.write("PARAMETERS: ")
    #     f.write(str(parameters))
    #     f.write("\n")
    #     f.write("COMMANDS: ")
    #     f.write(str(commands))

    model_logdir = os.path.dirname(arg_dict.get("model_path", ""))
    # Check if we chose one of the existing engines
    if arg_dict["engine"] not in AVAILABLE_SIMULATION_ENGINES:
        print(f"Invalid simulation engine. Valid arguments: --engine {AVAILABLE_SIMULATION_ENGINES}.")
        return

    if len(parameters) != 0:
        print("THREADING")
        multi_main(arg_dict, parameters, args.config, commands)

    if arg_dict.get("model_path") is None:
        print(
            "Path to the model using --model_path argument not specified. Testing random actions in selected "
            "environment."
        )
        arg_dict["gui"] = 1
        env = configure_env(arg_dict, model_logdir, for_train=0)
        test_env(env, arg_dict)
    else:
        env = configure_env(arg_dict, model_logdir, for_train=0)
        implemented_combos = configure_implemented_combos(env, model_logdir, arg_dict)
        test_model(env, None, implemented_combos, arg_dict, model_logdir)


if __name__ == "__main__":
    main()
