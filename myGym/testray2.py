import gym
import ray
import argparse
import json, commentjson
import time
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac.sac import SACConfig
from ray.rllib.algorithms.appo import APPOConfig
from ray.rllib.algorithms.ddpg import DDPGConfig
from ray.rllib.algorithms.marwil import MARWILConfig
from ray.rllib.algorithms.impala import ImpalaConfig
from gymnasium.wrappers import EnvCompatibility
from ray.tune.registry import register_env
from myGym.envs.gym_env import GymEnv
from ray.rllib.algorithms.algorithm import Algorithm
import pybullet as p
import pybullet_data
import sys
import os
import numpy as np


from ray.rllib.policy.policy import Policy


NUM_WORKERS = 1


def get_parser():
    parser = argparse.ArgumentParser()
    #Envinronment
    parser.add_argument("-cfg", "--config", default="./configs/train_A_RDDL.json", help="Can be passed instead of all arguments")
    parser.add_argument("-rt", "--ray_tune", action='store_true', help="Whether to train with ray grid search")
    parser.add_argument("-n", "--env_name", type=str, help="The name of environment")
    parser.add_argument("-ws", "--workspace", type=str, help="The name of workspace")
    parser.add_argument("-p", "--engine", type=str,  help="Name of the simulation engine you want to use")
    parser.add_argument("-sd", "--seed", type=int, help="Seed number")
    parser.add_argument("-d", "--render", type=str,  help="Type of rendering: opengl, opencv")
    parser.add_argument("-c", "--camera", type=int, help="The number of camera used to render and record")
    parser.add_argument("-vi", "--visualize", type=int,  help="Whether visualize camera render and vision in/out or not: 1 or 0")
    parser.add_argument("-vg", "--visgym", type=int,  help="Whether visualize gym background: 1 or 0")
    parser.add_argument("-g", "--gui", type=int, help="Wether the GUI of the simulation should be used or not: 1 or 0")
    #Robot
    parser.add_argument("-b", "--robot", type=str, help="Robot to train: kuka, panda, jaco ...")
    parser.add_argument("-bi", "--robot_init", nargs="*", type=float, help="Initial robot's end-effector position")
    parser.add_argument("-ba", "--robot_action", type=str, help="Robot's action control: step - end-effector relative position, absolute - end-effector absolute position, joints - joints' coordinates")
    parser.add_argument("-mv", "--max_velocity", type=float, help="Maximum velocity of robotic arm")
    parser.add_argument("-mf", "--max_force", type=float, help="Maximum force of robotic arm")
    parser.add_argument("-ar", "--action_repeat", type=int, help="Substeps of simulation without action from env")
    #Task
    parser.add_argument("-tt", "--task_type", type=str,  help="Type of task to learn: reach, push, throw, pick_and_place")
    parser.add_argument("-to", "--task_objects", nargs="*", type=str, help="Object (for reach) or a pair of objects (for other tasks) to manipulate with")
    parser.add_argument("-u", "--used_objects", nargs="*", type=str, help="List of extra objects to randomly appear in the scene")
    #Distractors
    parser.add_argument("-di", "--distractors", type=str, help="Object (for reach) to evade")
    parser.add_argument("-dm", "--distractor_moveable", type=int, help="can distractor move (0/1)")
    parser.add_argument("-ds", "--distractor_constant_speed", type=int, help="is speed of distractor constant (0/1)")
    parser.add_argument("-dd", "--distractor_movement_dimensions", type=int, help="in how many directions can the distractor move (1/2/3)")
    parser.add_argument("-de", "--distractor_movement_endpoints", nargs="*", type=float, help="2 coordinates (starting point and ending point)")
    parser.add_argument("-no", "--observed_links_num", type=int, help="number of robot links in observation space")
    #Reward
    parser.add_argument("-re", "--reward", type=str,  help="Defines how to compute the reward")
    parser.add_argument("-dt", "--distance_type", type=str, help="Type of distance metrics: euclidean, manhattan")
    #Train
    parser.add_argument("-w", "--train_framework", type=str,  help="Name of the training framework you want to use: {tensorflow, pytorch}")
    parser.add_argument("-a", "--algo", type=str,  help="The learning algorithm to be used (ppo2 or her)")
    parser.add_argument("-s", "--steps", type=int, help="The number of steps to train")
    parser.add_argument("-ms", "--max_episode_steps", type=int,  help="The maximum number of steps per episode")
    parser.add_argument("-ma", "--algo_steps", type=int,  help="The number of steps per for algo training (PPO2,A2C)")
    #Evaluation
    parser.add_argument("-ef", "--eval_freq", type=int,  help="Evaluate the agent every eval_freq steps")
    parser.add_argument("-e", "--eval_episodes", type=int,  help="Number of episodes to evaluate performance of the robot")
    #Saving and Logging
    parser.add_argument("-l", "--logdir", type=str,  help="Where to save results of training and trained models")
    parser.add_argument("-r", "--record", type=int, help="1: make a gif of model perfomance, 2: make a video of model performance, 0: don't record")
    #Mujoco
    parser.add_argument("-i", "--multiprocessing", type=int,  help="True: multiprocessing on (specify also the number of vectorized environemnts), False: multiprocessing off")
    parser.add_argument("-v", "--vectorized_envs", type=int,  help="The number of vectorized environments to run at once (mujoco multiprocessing only)")
    #Paths
    parser.add_argument("-m", "--model_path", type=str, help="Path to the the trained model to test")
    parser.add_argument("-vp", "--vae_path", type=str, help="Path to a trained VAE in 2dvu reward type")
    parser.add_argument("-yp", "--yolact_path", type=str, help="Path to a trained Yolact in 3dvu reward type")
    parser.add_argument("-yc", "--yolact_config", type=str, help="Path to saved config obj or name of an existing one in the data/Config script (e.g. 'yolact_base_config') or None for autodetection")
    parser.add_argument('-ptm', "--pretrained_model", type=str, help="Path to a model that you want to continue training")
    return parser


def configure_implemented_combos(arg_dict):
    implemented_combos = {"ppo": PPOConfig, "sac": SACConfig, "marwil": MARWILConfig, "appo":APPOConfig, "ddpg":DDPGConfig}
    return implemented_combos[arg_dict["algo"]]


def visualize_infotext(action, env, info):
    p.addUserDebugText(f"Episode:{env.env.episode_number}",
        [.65, 1., 0.45], textSize=1.0, lifeTime=0.5, textColorRGB=[0.4, 0.2, .3])
    p.addUserDebugText(f"Step:{env.env.episode_steps}",
        [.67, 1, .40], textSize=1.0, lifeTime=0.5, textColorRGB=[0.2, 0.8, 1])
    p.addUserDebugText(f"Subtask:{env.env.task.current_task}",
        [.69, 1, 0.35], textSize=1.0, lifeTime=0.5, textColorRGB=[0.4, 0.2, 1])
    p.addUserDebugText(f"Network:{env.env.reward.current_network}",
        [.71, 1, 0.3], textSize=1.0, lifeTime=0.5, textColorRGB=[0.0, 0.0, 1])
    p.addUserDebugText(f"Action (Gripper):{matrix(np.around(np.array(action),2))}",
        [.73, 1, 0.25], textSize=1.0, lifeTime=0.5, textColorRGB=[1, 0, 0])
    p.addUserDebugText(f"Actual_state:{matrix(np.around(np.array(env.env.observation['task_objects']['actual_state'][:3]),2))}",
        [.75, 1, 0.2], textSize=1.0, lifeTime=0.5, textColorRGB=[0.0, 1, 0.0])
    p.addUserDebugText(f"End_effector:{matrix(np.around(np.array(env.env.robot.end_effector_pos),2))}",
        [.77, 1, 0.15], textSize=1.0, lifeTime=0.5, textColorRGB=[0.0, 1, 0.0])
    p.addUserDebugText(f"        Object:{matrix(np.around(np.array(info['o']['actual_state']),2))}",
        [.8, 1, 0.10], textSize=1.0, lifeTime=0.5, textColorRGB=[0.0, 0.0, 1])
    p.addUserDebugText(f"Velocity:{env.env.max_velocity}",
        [.79, 1, 0.05], textSize=1.0, lifeTime=0.5, textColorRGB=[0.6, 0.8, .3])
    p.addUserDebugText(f"Force:{env.env.max_force}",
        [.81, 1, 0.00], textSize=1.0, lifeTime=0.5, textColorRGB=[0.3, 0.2, .4])


def build_env(arg_dict, model_logdir=None, for_train=True):
    env_arguments = {"render_on": True, "visualize": arg_dict["visualize"], "workspace": arg_dict["workspace"],
                     "robot": arg_dict["robot"], "robot_init_joint_poses": arg_dict["robot_init"],
                     "robot_action": arg_dict["robot_action"],"max_velocity": arg_dict["max_velocity"],
                     "max_force": arg_dict["max_force"],"task_type": arg_dict["task_type"],
                     "action_repeat": arg_dict["action_repeat"],
                     "task_objects":arg_dict["task_objects"], "observation":arg_dict["observation"], "distractors":arg_dict["distractors"],
                     "num_networks":arg_dict.get("num_networks", 1), "network_switcher":arg_dict.get("network_switcher", "gt"),
                     "distance_type": arg_dict["distance_type"], "used_objects": arg_dict["used_objects"],
                     "active_cameras": arg_dict["camera"], "color_dict":arg_dict.get("color_dict", {}),
                     "max_episode_steps": arg_dict["max_episode_steps"], "visgym":arg_dict["visgym"],
                     "reward": arg_dict["reward"], "logdir": arg_dict["logdir"], "vae_path": arg_dict["vae_path"],
                     "yolact_path": arg_dict["yolact_path"], "yolact_config": arg_dict["yolact_config"],
                     "natural_language": bool(arg_dict["natural_language"]),
                     }
    if for_train:
        env_arguments["gui_on"] = arg_dict["gui"]
    else:
        env_arguments["gui_on"] = arg_dict["gui"]

    if arg_dict["algo"] == "her":
        env = gym.make(arg_dict["env_name"], **env_arguments, obs_space="dict")  # her needs obs as a dict
    else:
        env = gym.make(arg_dict["env_name"], **env_arguments)
    if for_train:
        if arg_dict["engine"] == "mujoco":
            env = VecMonitor(env, model_logdir) if arg_dict["multiprocessing"] else Monitor(env, model_logdir)
        elif arg_dict["engine"] == "pybullet":
           try:
                env = Monitor(env, model_logdir, info_keywords=tuple('d'))
           except:
                pass

    if arg_dict["algo"] == "her":
        env = HERGoalEnvWrapper(env)
    return env


def detect_key(keypress,arg_dict,action):

    if 97 in keypress.keys() and keypress[97] == 1: # A
        action[2] += .03
        print(action)
    if 122 in keypress.keys() and keypress[122] == 1: # Z/Y
        action[2] -= .03
        print(action)
    if 65297 in keypress.keys() and keypress[65297] == 1: # ARROW UP
        action[1] -= .03
        print(action)
    if 65298 in keypress.keys() and keypress[65298] == 1: # ARROW DOWN
        action[1] += .03
        print(action)
    if 65295 in keypress.keys() and keypress[65295] == 1: # ARROW LEFT
        action[0] += .03
        print(action)
    if 65296 in keypress.keys() and keypress[65296] == 1: # ARROW RIGHT
        action[0] -= .03
        print(action)
    if 120 in keypress.keys() and keypress[120] == 1: # X
        action[3] -= .005
        action[4] -= .005
        print(action)
    if 99 in keypress.keys() and keypress[99] == 1: # C
        action[3] += .005
        action[4] += .005
        print(action)
    # if 100 in keypress.keys() and keypress[100] == 1:
    #     cube[cubecount] = p.loadURDF(pkg_resources.resource_filename("myGym", os.path.join("envs", "objects/assembly/urdf/cube_holes.urdf")), [action[0], action[1],action[2]-0.2 ])
    #     change_dynamics(cube[cubecount],lfriction,rfriction,ldamping,adamping)
    #     cubecount +=1
    if "step" in arg_dict["robot_action"]:
        action[:3] = np.multiply(action [:3],10)
    elif "joints" in arg_dict["robot_action"]:
        print("Robot action: Joints - KEYBOARD CONTROL UNDER DEVELOPMENT")
        quit()
    #for i in range (env.action_space.shape[0]):
    #    env.env.robot.joints_max_velo[i] = p.readUserDebugParameter(maxvelo)
    #    env.env.robot.joints_max_force[i] = p.readUserDebugParameter(maxforce)
    return action



def get_arguments(parser):
    args = parser.parse_args()
    with open(args.config, "r") as f:
            arg_dict = commentjson.load(f)
    for key, value in vars(args).items():
        if value is not None and key != "config":
            if key in ["robot_init"]:
                arg_dict[key] = [float(arg_dict[key][i]) for i in range(len(arg_dict[key]))]
            else:
                arg_dict[key] = value
    return arg_dict, args


def configure_env(arg_dict, for_train=True):
    env_arguments = {"render_on": True, "visualize": arg_dict["visualize"], "workspace": arg_dict["workspace"],
                     "robot": arg_dict["robot"], "robot_init_joint_poses": arg_dict["robot_init"],
                     "robot_action": arg_dict["robot_action"],"max_velocity": arg_dict["max_velocity"],
                     "max_force": arg_dict["max_force"],"task_type": arg_dict["task_type"],
                     "action_repeat": arg_dict["action_repeat"],
                     "task_objects":arg_dict["task_objects"], "observation":arg_dict["observation"], "distractors":arg_dict["distractors"],
                     "num_networks":arg_dict.get("num_networks", 1), "network_switcher":arg_dict.get("network_switcher", "gt"),
                     "distance_type": arg_dict["distance_type"], "used_objects": arg_dict["used_objects"],
                     "active_cameras": arg_dict["camera"], "color_dict":arg_dict.get("color_dict", {}),
                     "max_episode_steps": arg_dict["max_episode_steps"], "visgym":arg_dict["visgym"],
                     "reward": arg_dict["reward"], "logdir": arg_dict["logdir"], "vae_path": arg_dict["vae_path"],
                     "yolact_path": arg_dict["yolact_path"], "yolact_config": arg_dict["yolact_config"], "algo":arg_dict["algo"]}
    if for_train:
        env_arguments["gui_on"] = arg_dict["gui"]
    else:
        env_arguments["gui_on"] = arg_dict["gui"]
    return env_arguments


def env_creator(env_config):
        env = EnvCompatibility(GymEnv(**env_config))
        env.spec.max_episode_steps = 512
        return env


def test_env(arg_dict):
    arg_dict["vsampling"] = 0
    arg_dict["vinfo"] = 0
    register_env('GymEnv-v0', env_creator)
    env_args = configure_env(arg_dict, for_train=False)

    # Create environment
    env_args.pop("algo")
    env = env_creator(env_args)

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(1.2, 180, -30, [0.0, 0.5, 0.05])
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    joints = ['Joint1', 'Joint2', 'Joint3', 'Joint4', 'Joint5', 'Joint6', 'Joint7', 'Joint 8', 'Joint 9', 'Joint10',
              'Joint11', 'Joint12', 'Joint13', 'Joint14', 'Joint15', 'Joint16', 'Joint17', 'Joint 18', 'Joint 19']
    jointparams = ['Jnt1', 'Jnt2', 'Jnt3', 'Jnt4', 'Jnt5', 'Jnt6', 'Jnt7', 'Jnt 8', 'Jnt 9', 'Jnt10', 'Jnt11', 'Jnt12',
                   'Jnt13', 'Jnt14', 'Jnt15', 'Jnt16', 'Jnt17', 'Jnt 18', 'Jnt 19']

    if arg_dict["gui"] == 0:
        print ("Add --gui 1 parameter to visualize environment")
        quit()

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(1.2, 180, -30, [0.0, 0.5, 0.05])
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    if arg_dict["control"] == "slider":
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        if "joints" in arg_dict["robot_action"]:
            if 'gripper' in arg_dict["robot_action"]:
                print ("gripper is present")
                for i in range (env.action_space.shape[0]):
                    if i < (env.action_space.shape[0] - len(env.env.robot.gjoints_rest_poses)):
                        joints[i] = p.addUserDebugParameter(joints[i], env.action_space.low[i], env.action_space.high[i], env.env.robot.init_joint_poses[i])
                    else:
                        joints[i] = p.addUserDebugParameter(joints[i], env.action_space.low[i], env.action_space.high[i], .02)
            else:
                for i in range (env.action_space.shape[0]):
                    joints[i] = p.addUserDebugParameter(joints[i],env.action_space.low[i], env.action_space.high[i], env.env.robot.init_joint_poses[i])
        elif "absolute" in arg_dict["robot_action"]:
            if 'gripper' in arg_dict["robot_action"]:
                print ("gripper is present")
                for i in range (env.action_space.shape[0]):
                    if i < (env.action_space.shape[0] - len(env.env.robot.gjoints_rest_poses)):
                        joints[i] = p.addUserDebugParameter(joints[i], -1, 1, arg_dict["robot_init"][i])
                    else:
                        joints[i] = p.addUserDebugParameter(joints[i], -1, 1, .02)
            else:
                for i in range (env.action_space.shape[0]):
                    joints[i] = p.addUserDebugParameter(joints[i], -1, 1, arg_dict["robot_init"][i])
        elif "step" in arg_dict["robot_action"]:
            if 'gripper' in arg_dict["robot_action"]:
                print ("gripper is present")
                for i in range (env.action_space.shape[0]):
                    if i < (env.action_space.shape[0] - len(env.env.robot.gjoints_rest_poses)):
                        joints[i] = p.addUserDebugParameter(joints[i], -1, 1, 0)
                    else:
                        joints[i] = p.addUserDebugParameter(joints[i], -1, 1, .02)
            else:
                for i in range (env.action_space.shape[0]):
                    joints[i] = p.addUserDebugParameter(joints[i], -1, 1, 0)


    if arg_dict["control"] == "keyboard":
        action = arg_dict["robot_init"]
        if "gripper" in arg_dict["robot_action"]:
            action.append(.1)
            action.append(.1)

    if arg_dict["control"] == "random":
        action = env.action_space.sample()
    if arg_dict["control"] == "keyboard":
        action = arg_dict["robot_init"]
        # if "gripper" in arg_dict["robot_action"]:
        #     action.append(.1)
        #     action.append(.1)
    if arg_dict["control"] == "slider":
        action = []
        for i in range(env.action_space.shape[0]):
            jointparams[i] = p.readUserDebugParameter(joints[i])
            action.append(jointparams[i])


    for e in range(50):
        observation = env.reset()[0]
        for t in range(arg_dict["max_episode_steps"]):

            if arg_dict["control"] == "slider":
                action = []
                for i in range(env.action_space.shape[0]):
                    jointparams[i] = p.readUserDebugParameter(joints[i])
                    action.append(jointparams[i])
                    # env.env.robot.joints_max_velo[i] = p.readUserDebugParameter(maxvelo)
                    # env.env.robot.joints_max_force[i] = p.readUserDebugParameter(maxforce)

            if arg_dict["control"] == "observation":
                if t == 0:
                    action = env.action_space.sample()
                else:
                    if "joints" in arg_dict["robot_action"]:
                        action = info['o']["additional_obs"]["joints_angles"] #n
                    elif "absolute" in arg_dict["robot_action"]:
                        action = info['o']["actual_state"]
                    else:
                        action = [0,0,0]

            elif arg_dict["control"] == "oraculum":
                if t == 0:
                    action = env.action_space.sample()
                else:
                    if "absolute" in arg_dict["robot_action"]:
                        if env.env.reward.reward_name == "approach":
                            if env.env.reward.rewards_num <= 2:
                                action[:3] = info['o']["goal_state"][:3]
                            else:
                                action[:3] = info['o']["actual_state"][:3]
                            if "gripper" in arg_dict["robot_action"]:
                                action[3] = 1
                                action[4] = 1
                        if env.env.reward.reward_name == "grasp":
                            #print(arg_dict["robot_action"])
                            if "gripper" in arg_dict["robot_action"]:
                                action[3] = 0
                                action[4] = 0
                        if env.env.reward.reward_name == "move":
                            action[:3] = info['o']["goal_state"][:3]
                            if "gripper" in arg_dict["robot_action"]:
                                action[3] = 0
                                action[4] = 0
                        if env.env.reward.reward_name == "drop":
                            if "gripper" in arg_dict["robot_action"]:
                                action[3] = 1
                                action[4] = 1
                        if env.env.reward.reward_name == "withdraw":
                            action[:3] = np.array(info['o']["actual_state"][:3]) + np.array([0,0,0.4])
                            if "gripper" in arg_dict["robot_action"]:
                                action[3] = 1
                                action[4] = 1
                    else:
                        print("ERROR - Oraculum mode only works for absolute actions")
                        quit()

            elif arg_dict["control"] == "keyboard":
                keypress = p.getKeyboardEvents()
                # print(action)
                # if env.env.reward.reward_name == "grasp":
                #     #print("grasping")
                # else:
                #     print("reward name:", env.env.reward.reward_name)
                action = detect_key(keypress, arg_dict, action)
            elif arg_dict["control"] == "random":
                    action = env.action_space.sample()


            observation, reward, done, _, info = env.step(action)[:5]

            #print("info:", info)
            if arg_dict["control"] == "oraculum":
                gripper_pos = info['o']['additional_obs']['endeff_xyz']
                action[:3] = gripper_pos
                env.step(action)
            elif "step" in arg_dict["robot_action"]:
                action[:3] = [0,0,0]
                gripper_pos = np.array(info['o']['additional_obs']['endeff_xyz'])
                object_pos = np.array(info['o']['actual_state'])[:3]
                dist = np.linalg.norm(object_pos - gripper_pos)
                #print("object distance:", dist)

            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break





def test_model(arg_dict):
    # Define the configuration for the trainer
    register_env('GymEnv-v0', env_creator)
    env_args = configure_env(arg_dict, for_train=False)

    # Create environment
    env_args.pop("algo")
    env = env_creator(env_args)




    #variables for evaluation
    success_episodes_num = 0
    distance_error_sum = 0
    steps_sum = 0


    # learned policy from checkpoint
    model_path = arg_dict["model_path"] + "policies/default_policy/"
    #model_path =  "./trained_models/A/A_table_tiago_tiago_dual_absolute_gripper_ppo_2/"
    testing_policy = Policy.from_checkpoint(model_path) #Specify model policy path name

    #Evaluation loop
    for e in range(arg_dict["eval_episodes"]):
        # Initial state
        state = env.reset()[0]
        is_successful = 0
        distance_error = 0
        done = False
        while not done:
            steps_sum += 1
            action = testing_policy.compute_single_action(state)[0]
            state, reward, done, _, info = env.step(action)[:5]
            is_successful = not info['f']
            distance_error = info['d']
            # Render the environment
            env.render()
        success_episodes_num += is_successful
        distance_error_sum += distance_error

    mean_distance_error = distance_error_sum / arg_dict["eval_episodes"]
    mean_steps_num = steps_sum // arg_dict["eval_episodes"]
    env.close()
    #Evaluation summary printout
    print("#---------Evaluation-Summary---------#")
    print("{} of {} episodes ({} %) were successful".format(success_episodes_num, arg_dict["eval_episodes"],
                                                            success_episodes_num / arg_dict["eval_episodes"] * 100))
    print("Mean distance error is {:.2f}%".format(mean_distance_error * 100))
    print("Mean number of steps {}".format(mean_steps_num))
    print("#------------------------------------#")
    model_name = arg_dict["algo"] + '_' + str(arg_dict["steps"])
    file = open(os.path.join(arg_dict['logdir'], "train_" + model_name + ".txt"), 'a')
    file.write("\n")
    file.write("#Evaluation results: \n")
    file.write("#{} of {} episodes were successful \n".format(success_episodes_num, arg_dict["eval_episodes"]))
    file.write("#Mean distance error is {:.2f}% \n".format(mean_distance_error * 100))
    file.write("#Mean number of steps {}\n".format(mean_steps_num))
    file.close()

    ray.shutdown()

if __name__ == "__main__":
    ray.init(num_gpus=1, num_cpus=5)
    parser = get_parser()
    parser.add_argument("-ct", "--control", default="slider",
                        help="How to control robot during testing. Valid arguments: keyboard, observation, random, oraculum, slider")
    arg_dict, args = get_arguments(parser)
    if arg_dict.get("model_path") is None:
        print("Path to the model using --model_path argument not specified. Testing actions in environment.")
        test_env(arg_dict)
    else:
        test_model(arg_dict)
