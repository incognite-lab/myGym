import argparse
import copy
import json
import multiprocessing
import os
import random
import subprocess
import sys
import time

import commentjson
import gym
import numpy as np
import pkg_resources
from sklearn.model_selection import ParameterGrid

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.bench import Monitor
from stable_baselines.her import HERGoalEnvWrapper
# Importing with slightly modified names: T-TensorFlow
from stable_baselines import PPO1 as PPO1_T, PPO2 as PPO2_T, HER as HER_T, SAC as SAC_T, DDPG as DDPG_T
from stable_baselines import TD3 as TD3_T, A2C as A2C_T, ACKTR as ACKTR_T, TRPO as TRPO_T, GAIL as GAIL_T

from myGym.stable_baselines_mygym.algo import MyAlgo
from myGym.stable_baselines_mygym.reference import REFER
from myGym.stable_baselines_mygym.multi_ppo2 import MultiPPO2
from myGym.stable_baselines_mygym.multi_acktr import MultiACKTR
from myGym.stable_baselines_mygym.policies import MyMlpPolicy

from stable_baselines.gail import ExpertDataset, generate_expert_traj
from stable_baselines.sac.policies import MlpPolicy as MlpPolicySAC
from stable_baselines.ddpg.policies import MlpPolicy as MlpPolicyDDPG
from stable_baselines.td3.policies import MlpPolicy as MlpPolicyTD3

# Import helper classes and functions for monitoring
from myGym.utils.callbacks import SaveOnBestTrainingRewardCallback, CustomEvalCallback
from myGym.envs.natural_language import NaturalLanguage

# This is global variable for the type of engine we are working with
AVAILABLE_SIMULATION_ENGINES = ["pybullet"]
AVAILABLE_TRAINING_FRAMEWORKS = ["tensorflow"]

def save_results(arg_dict, model_name, env, model_logdir=None, show=False):
    if model_logdir is None:
        model_logdir = arg_dict["logdir"]
    print(f"model_logdir: {model_logdir}")
    print("Congratulations! Training with {} timesteps succeed!".format(arg_dict["steps"]))


def configure_env(arg_dict, model_logdir=None, for_train=True):
    env_arguments = {"render_on": True, "visualize": arg_dict["visualize"], "workspace": arg_dict["workspace"],
                     "robot": arg_dict["robot"], "robot_init_joint_poses": arg_dict["robot_init"],
                     "robot_action": arg_dict["robot_action"], "max_velocity": arg_dict["max_velocity"],
                     "max_force": arg_dict["max_force"], "task_type": arg_dict["task_type"],
                     "action_repeat": arg_dict["action_repeat"],
                     "task_objects": arg_dict["task_objects"], "observation": arg_dict["observation"],
                     "distractors": arg_dict["distractors"],
                     "num_networks": arg_dict.get("num_networks", 1),
                     "network_switcher": arg_dict.get("network_switcher", "gt"),
                     "distance_type": arg_dict["distance_type"], "used_objects": arg_dict["used_objects"],
                     "max_episode_steps": arg_dict["max_episode_steps"], "visgym": arg_dict["visgym"],
                     "active_cameras": arg_dict["camera"], "color_dict": arg_dict.get("color_dict", {}),
                     "reward": arg_dict["reward"], "logdir": arg_dict["logdir"], "vae_path": arg_dict["vae_path"],
                     "yolact_path": arg_dict["yolact_path"], "yolact_config": arg_dict["yolact_config"],
                     "natural_language": bool(arg_dict["natural_language"]),
                     "training": bool(for_train),
                     "gui_on": arg_dict["gui"]
                     }

    if arg_dict["algo"] == "her":
        env = gym.make(arg_dict["env_name"], **env_arguments, obs_space="dict")  # her needs obs as a dict
    else:
        env = gym.make(arg_dict["env_name"], **env_arguments)

    if for_train:
        if arg_dict["engine"] == "pybullet":
            env = Monitor(env, model_logdir, info_keywords=tuple('d'))
    if arg_dict["algo"] == "her":
        env = HERGoalEnvWrapper(env)
    return env


def configure_implemented_combos(env, model_logdir, arg_dict):
    implemented_combos = {"ppo2": {"tensorflow": [PPO2_T, (MlpPolicy, env),
                                                  {"n_steps": arg_dict["algo_steps"], "verbose": 1,
                                                   "tensorboard_log": model_logdir}]},
                          "ppo": {"tensorflow": [PPO1_T, (MlpPolicy, env),
                                                 {"verbose": 1, "tensorboard_log": model_logdir}], },
                          "her": {"tensorflow": [HER_T, (MlpPolicyDDPG, env, DDPG_T),
                                                 {"goal_selection_strategy": 'final', "verbose": 1,
                                                  "tensorboard_log": model_logdir, "n_sampled_goal": 1}]},
                          "sac": {"tensorflow": [SAC_T, (MlpPolicySAC, env),
                                                 {"verbose": 1, "tensorboard_log": model_logdir}], },
                          "ddpg": {"tensorflow": [DDPG_T, (MlpPolicyDDPG, env),
                                                  {"verbose": 1, "tensorboard_log": model_logdir}]},
                          "td3": {"tensorflow": [TD3_T, (MlpPolicyTD3, env),
                                                 {"verbose": 1, "tensorboard_log": model_logdir}], },
                          "acktr": {"tensorflow": [ACKTR_T, (MlpPolicy, env),
                                                   {"n_steps": arg_dict["algo_steps"], "verbose": 1,
                                                    "tensorboard_log": model_logdir}]},
                          "trpo": {"tensorflow": [TRPO_T, (MlpPolicy, env),
                                                  {"verbose": 1, "tensorboard_log": model_logdir}]},
                          "gail": {"tensorflow": [GAIL_T, (MlpPolicy, env),
                                                  {"verbose": 1, "tensorboard_log": model_logdir}]},
                          "a2c": {"tensorflow": [A2C_T, (MlpPolicy, env),
                                                 {"n_steps": arg_dict["algo_steps"], "verbose": 1,
                                                  "tensorboard_log": model_logdir}], },
                          "myalgo": {"tensorflow": [MyAlgo, (MyMlpPolicy, env),
                                                    {"n_steps": arg_dict["algo_steps"], "verbose": 1,
                                                     "tensorboard_log": model_logdir}]},
                          "ref": {"tensorflow": [REFER, (MlpPolicy, env),
                                                 {"n_steps": arg_dict["algo_steps"], "verbose": 1,
                                                  "tensorboard_log": model_logdir}]},
                          "multippo2": {"tensorflow": [MultiPPO2, (MlpPolicy, env), {"n_steps": arg_dict["algo_steps"],
                                                                                     "n_models": arg_dict[
                                                                                         "num_networks"], "verbose": 1,
                                                                                     "tensorboard_log": model_logdir}]},
                          "multiacktr": {"tensorflow": [MultiACKTR, (MlpPolicy, env),
                                                        {"n_steps": arg_dict["algo_steps"],
                                                         "n_models": arg_dict["num_networks"], "verbose": 1,
                                                         "tensorboard_log": model_logdir}]}}

    return implemented_combos


def train(env, implemented_combos, model_logdir, arg_dict, pretrained_model=None):
    model_name = arg_dict["algo"] + '_' + str(arg_dict["steps"])
    conf_pth = os.path.join(model_logdir, "train.json")
    model_path = os.path.join(model_logdir, "best_model.zip")
    arg_dict["model_path"] = model_path
    seed = arg_dict.get("seed", None)
    with open(conf_pth, "w") as f:
        json.dump(arg_dict, f, indent=4)

    print("WWWWWWWWWWWWWWWWWWWWWW")
    print(env)
    print("WWWWWWWWWWWWWWWWWWWWWW")
    model_args = implemented_combos[arg_dict["algo"]][arg_dict["train_framework"]][1]
    model_kwargs = implemented_combos[arg_dict["algo"]][arg_dict["train_framework"]][2]
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        model_kwargs["seed"] = seed
    if pretrained_model:
        if not os.path.isabs(pretrained_model):
            pretrained_model = pkg_resources.resource_filename("myGym", pretrained_model)
        env = model_args[1]
        vec_env = DummyVecEnv([lambda: env])
        model = implemented_combos[arg_dict["algo"]][arg_dict["train_framework"]][0].load(pretrained_model, vec_env)
    else:
        model = implemented_combos[arg_dict["algo"]][arg_dict["train_framework"]][0](*model_args, **model_kwargs)

    if arg_dict["algo"] == "gail":
        # Multi processing: (using MPI)
        if arg_dict["train_framework"] == 'tensorflow':
            # Generate expert trajectories (train expert)
            generate_expert_traj(model, model_name, n_timesteps=3000, n_episodes=100)
            # Load the expert dataset
            dataset = ExpertDataset(expert_path=model_name + '.npz', traj_limitation=10, verbose=1)
            kwargs = {"verbose": 1}
            if seed is not None:
                kwargs["seed"] = seed
            model = GAIL_T('MlpPolicy', model_name, dataset, **kwargs)

    start_time = time.time()
    callbacks_list = []
    if pretrained_model:
        model_logdir = pretrained_model.split('/')
        model_logdir = model_logdir[:-1]
        model_logdir = "/".join(model_logdir)
        auto_save_callback = SaveOnBestTrainingRewardCallback(check_freq=1024, logdir=model_logdir, env=env,
                                                              engine=arg_dict["engine"],
                                                              multiprocessing=arg_dict["multiprocessing"])
    else:
        auto_save_callback = SaveOnBestTrainingRewardCallback(check_freq=1024, logdir=model_logdir, env=env,
                                                              engine=arg_dict["engine"],
                                                              multiprocessing=arg_dict["multiprocessing"])
    callbacks_list.append(auto_save_callback)
    if arg_dict["eval_freq"]:
        eval_env = env
        eval_callback = CustomEvalCallback(eval_env, log_path=model_logdir,
                                           eval_freq=arg_dict["eval_freq"],
                                           algo_steps=arg_dict["algo_steps"],
                                           n_eval_episodes=arg_dict["eval_episodes"],
                                           record=arg_dict["record"],
                                           camera_id=arg_dict["camera"])
        callbacks_list.append(eval_callback)

    print("learn started")
    model.learn(total_timesteps=arg_dict["steps"], callback=callbacks_list)
    print("learn ended")
    model.save(os.path.join(model_logdir, model_name))
    print("Training time: {:.2f} s".format(time.time() - start_time))
    print("Training steps: {:} s".format(model.num_timesteps))

    # info_keywords in monitor class above is necessary for pybullet to save_results
    # when using the info_keywords for mujoco we get an error
    if arg_dict["engine"] == "pybullet":
        save_results(arg_dict, model_name, env, model_logdir)
    return model


def get_parser():
    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument("-cfg", "--config", type=str, default="./configs/train_FM_nico.json", help="Config file path")
    parser.add_argument("-n", "--env_name", type=str, help="Environment name")
    parser.add_argument("-ws", "--workspace", type=str, help="Workspace name")
    parser.add_argument("-p", "--engine", type=str, help="Simulation engine name")
    parser.add_argument("-sd", "--seed", type=int, default=1, help="Seed number")
    parser.add_argument("-d", "--render", type=str, help="Rendering type: opengl, opencv")
    parser.add_argument("-c", "--camera", type=int, help="Number of cameras for rendering and recording")
    parser.add_argument("-vi", "--visualize", type=int, help="Visualize camera render and vision: 1 or 0")
    parser.add_argument("-vg", "--visgym", type=int, help="Visualize gym background: 1 or 0")
    parser.add_argument("-g", "--gui", type=int, help="Use GUI: 1 or 0")

    # Robot
    parser.add_argument("-b", "--robot", default=["kuka", "panda"], nargs='*', help="Robot to train")
    parser.add_argument("-bi", "--robot_init", nargs="*", type=float, help="Initial robot's end-effector position")
    parser.add_argument("-ba", "--robot_action", default=["joints"], nargs='*', help="Robot's action control")
    parser.add_argument("-mv", "--max_velocity", default=[3], nargs='*', help="Arm speed")
    parser.add_argument("-mf", "--max_force", default=[100], nargs='*', help="Arm force")
    parser.add_argument("-ar", "--action_repeat", default=[1], nargs='*', help="Simulation substeps without action")

    # Task
    parser.add_argument("-tt", "--task_type", default=["reach"], nargs='*', help="Task type to learn")
    parser.add_argument("-to", "--task_objects", nargs="*", type=str, help="Objects to manipulate")
    parser.add_argument("-u", "--used_objects", nargs="*", type=str, help="Extra objects to appear in the scene")

    # Distractors
    parser.add_argument("-di", "--distractors", type=str, help="Object to evade")
    parser.add_argument("-dm", "--distractor_moveable", type=int, help="Can distractor move: 0 or 1")
    parser.add_argument("-ds", "--distractor_constant_speed", type=int, help="Is speed of distractor constant: 0 or 1")
    parser.add_argument("-dd", "--distractor_movement_dimensions", type=int, help="Movement directions: 1, 2, or 3")
    parser.add_argument("-de", "--distractor_movement_endpoints", nargs="*", type=float, help="Movement endpoints")
    parser.add_argument("-no", "--observed_links_num", type=int, help="Number of robot links in observation space")

    # Reward
    parser.add_argument("-re", "--reward", type=str, help="Reward computation method")
    parser.add_argument("-dt", "--distance_type", type=str, help="Distance metrics type: euclidean, manhattan")

    # Train
    parser.add_argument("-w", "--train_framework", default=["tensorflow"], nargs='*', help="Training framework")
    parser.add_argument("-a", "--algo", default=["ppo2"], nargs='*', help="Algorithms to test")
    parser.add_argument("-s", "--steps", type=int, help="Number of training steps")
    parser.add_argument("-ms", "--max_episode_steps", type=int, help="Maximum steps per episode")
    parser.add_argument("-ma", "--algo_steps", type=int, help="Steps per algorithm training")

    # Evaluation
    parser.add_argument("-ef", "--eval_freq", type=int, help="Evaluation frequency in steps")
    parser.add_argument("-e", "--eval_episodes", type=int, help="Number of evaluation episodes")

    # Saving and Logging
    parser.add_argument("-l", "--logdir", type=str, default="./trained_models/reach", help="Directory to save results")
    parser.add_argument("-r", "--record", type=int, help="Record performance: 1 for gif, 2 for video, 0 for none")

    # Paths
    parser.add_argument("-m", "--model_path", type=str, help="Path to the trained model")
    parser.add_argument("-vp", "--vae_path", type=str, help="Path to a trained VAE")
    parser.add_argument("-yp", "--yolact_path", type=str, help="Path to a trained Yolact")
    parser.add_argument("-yc", "--yolact_config", type=str, help="Path to Yolact config or name in data/Config script")
    parser.add_argument('-ptm', "--pretrained_model", type=str, help="Path to a model for continued training")
    parser.add_argument("-thread", "--threaded", type=bool, default=True, help="Run in threads")
    parser.add_argument("-out", "--output", type=str, default="./trained_models/multitester.json", help="Output file")

    return parser


def get_arguments(parser):
    args = parser.parse_args()
    commands = {}
    with open(args.config, "r") as f:
        arg_dict = commentjson.load(f)
    for key, value in arg_dict.items():
        if value is not None and key != "config":
            if key in ["robot_init"]:
                arg_dict[key] = [float(arg_dict[key][i]) for i in range(len(arg_dict[key]))]
            elif type(value) is list and len(value) <= 1 and key != "task_objects":
                arg_dict[key] = value[0]
    for key, value in vars(args).items():
        if value is not None and key != "config":
            if key not in arg_dict or arg_dict[key] is None:
                arg_dict[key] = value
            if value != parser.get_default(key):
                commands[key] = value
                if key in ["task_objects"]:
                    arg_dict[key] = task_objects_replacement(value, arg_dict[key], arg_dict["task_type"])
                    if len(value) == 1:
                        commands[key] = value[0]
                elif type(value) is list and len(value) <= 1:
                    arg_dict[key] = value[0]
                else:
                    arg_dict[key] = value
    return arg_dict, commands


def task_objects_replacement(task_objects_new, task_objects_old, task_type):
    """
    If task_objects is given as a parameter, this method converts string into a proper format depending on task_type
    (null init for task_type reach)

    [{"init":{"obj_name":"null"}, "goal":{"obj_name":"cube_holes","fixed":1,"rand_rot":0, "sampling_area":[-0.5, 0.2,
    0.3, 0.6, 0.1, 0.4]}}]
    """
    ret = copy.deepcopy(task_objects_old)
    if len(task_objects_new) > len(task_objects_old):
        msg = "More objects given than there are subtasks."
        raise Exception(msg)
    if task_type == "reach":
        dest = "goal"
    else:
        dest = "init"
    for i in range(len(task_objects_new)):
        ret[i][dest]["obj_name"] = task_objects_new[i]

    return ret


def process_natural_language_command(cmd, env,
                                     output_relative_path=os.path.join("envs", "examples", "natural_language.txt")):
    env.reset()
    nl = NaturalLanguage(env)

    if cmd in ["description", "new_tasks"]:
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), output_relative_path), "w") as file:
            file.write(nl.generate_task_description() if cmd == "description" else "\n".join(nl.generate_new_tasks()))
    else:
        msg = f"Unknown natural language command: {cmd}"
        raise Exception(msg)


def multi_train(params, arg_dict, configfile, commands):
    logdirfile = arg_dict["logdir"]
    print((" ".join(f"--{key} {value}" for key, value in params.items())).split())
    # WE WANT TO ALSO SEND PARAMS FROM COMMAND LINE
    command = (
            f"python train.py --config {configfile} --logdir {logdirfile} "
            + " ".join(f"--{key} {value}" for key, value in params.items()) + " "
            + " ".join(f"--{key} {' '.join(map(str, value)) if isinstance(value, list) else value}" for key, value in commands.items())
    )
    print(command)
    with open("train.log", "wb") as f:
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        for c in iter(lambda: process.stdout.read(1), b''):
            sys.stdout.buffer.write(c)
            f.write(c)


def multi_main(arg_dict, parameters, configfile, commands):
    parameter_grid = ParameterGrid(parameters)

    threaded = arg_dict["threaded"]
    threads = []

    start_time = time.time()
    for i, params in enumerate(parameter_grid):
        if threaded:
            print("Thread ", i + 1, " starting")
            thread = multiprocessing.Process(target=multi_train, args=(params.copy(), arg_dict, configfile, commands))
            thread.start()
            threads.append(thread)
        else:
            multi_train(params.copy(), arg_dict, configfile, commands)

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
    # with open("arg_dict_train", "w") as f:
    #     f.write("ARG DICT: ")
    #     f.write(str(arg_dict))
    #     f.write("\n")
    #     f.write("PARAMETERS: ")
    #     f.write(str(parameters))
    #     f.write("\n")
    #     f.write("COMMANDS: ")
    #     f.write(str(commands))

    # Check if we chose one of the existing engines
    if arg_dict["engine"] not in AVAILABLE_SIMULATION_ENGINES:
        print(f"Invalid simulation engine. Valid arguments: --engine {AVAILABLE_SIMULATION_ENGINES}.")
        return

    if len(parameters) != 0:
        print("THREADING")
        multi_main(arg_dict, parameters, args.config, commands)

    if not os.path.isabs(arg_dict["logdir"]):
        arg_dict["logdir"] = os.path.join("./", arg_dict["logdir"])
    os.makedirs(arg_dict["logdir"], exist_ok=True)
    model_logdir_ori = os.path.join(arg_dict["logdir"], "_".join(
        (arg_dict["task_type"], arg_dict["workspace"], arg_dict["robot"], arg_dict["robot_action"], arg_dict["algo"])))

    model_logdir = model_logdir_ori
    add = 2
    while True:
        try:
            os.makedirs(model_logdir, exist_ok=False)
            break
        except:
            model_logdir = "_".join((model_logdir_ori, str(add)))
            add += 1

    env = configure_env(arg_dict, model_logdir, for_train=True)
    implemented_combos = configure_implemented_combos(env, model_logdir, arg_dict)
    train(env, implemented_combos, model_logdir, arg_dict, arg_dict["pretrained_model"])


if __name__ == "__main__":
    main()
