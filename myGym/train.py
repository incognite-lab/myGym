import multiprocessing
import subprocess
import sys
import warnings

import commentjson
import copy
import json
import os
import random
import time
from typing import Callable


import numpy as np
import importlib.resources as pkg_resources
import os, sys, time, yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
import json, commentjson
import gymnasium as gym
from sklearn.model_selection import ParameterGrid
from myGym.envs.gym_env import GymEnv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6

try:
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.utils import set_random_seed
except Exception as e:
    print(e)

try:
    from stable_baselines3 import A2C as A2C_P, SAC as SAC_P, TD3 as TD3_P, PPO
except:
    print("Torch isn't probably installed correctly")

# Import helper classes and functions for monitoring
from myGym.utils.callbacksSB3 import SaveOnBestTrainingRewardCallback, MultiPPOEvalCallback, PPOEvalCallback
from myGym.envs.natural_language import NaturalLanguage
from myGym.stable_baselines_mygym.multi_ppo_SB3 import MultiPPOSB3
from myGym.stable_baselines_mygym.ppoSB3 import PPO as PPO_P
from myGym.stable_baselines_mygym.Subproc_vec_envSB3 import SubprocVecEnv

# This is a global variable for the type of engine we are working with
AVAILABLE_SIMULATION_ENGINES = ["pybullet"]
AVAILABLE_TRAINING_FRAMEWORKS = ["pytorch"]




def save_results(arg_dict, model_name, env, model_logdir=None, show=False):
    if model_logdir is None:
        model_logdir = arg_dict["logdir"]
    print(f"model_logdir: {model_logdir}")
    print("Congratulations! Training with {} timesteps succeed!".format(arg_dict["steps"]))


def configure_env(arg_dict, model_logdir=None, for_train=True):
    env_arguments = {"render_on": True, "visualize": arg_dict["visualize"], "workspace": arg_dict["workspace"], "framework":"stable_baselines",
                     "robot": arg_dict["robot"], "robot_init_joint_poses": arg_dict["robot_init"],
                     "robot_action": arg_dict["robot_action"],"max_velocity": arg_dict["max_velocity"],
                     "max_force": arg_dict["max_force"],
                     "action_repeat": arg_dict["action_repeat"], "rddl": arg_dict["rddl"],
                     "observation":arg_dict["observation"], "distractors":arg_dict["distractors"],
                     "num_networks":arg_dict.get("num_networks", 1), "network_switcher":arg_dict.get("network_switcher", "gt"),
                     "active_cameras": arg_dict["camera"], "color_dict":arg_dict.get("color_dict", {}),
                     "max_steps": arg_dict["max_episode_steps"], "visgym":arg_dict["visgym"],
                     "logdir": arg_dict["logdir"], "vae_path": arg_dict["vae_path"],
                     "yolact_path": arg_dict["yolact_path"], "yolact_config": arg_dict["yolact_config"],
                     "natural_language": bool(arg_dict["natural_language"]),
                     "training": bool(for_train)
                     }

    if "network_switcher" in arg_dict.keys():
        env_arguments["network_switcher"] = arg_dict["network_switcher"]
    if arg_dict["algo"] == "her":
        env = gym.make(arg_dict["env_name"], **env_arguments, obs_space="dict")  # her needs obs as a dict
    else:
        env = gym.make(arg_dict["env_name"], **env_arguments)
        env.spec.max_episode_steps = 512

    if for_train:
        if arg_dict["engine"] == "mujoco":
            env = VecMonitor(env, model_logdir) if arg_dict["multiprocessing"] else Monitor(env, model_logdir)
        elif arg_dict["engine"] == "pybullet" and not arg_dict["multiprocessing"]:
            env = Monitor(env, filename=model_logdir, info_keywords=tuple('d'))

    if arg_dict["algo"] == "her":
        env = HERGoalEnvWrapper(env)
    return env


def make_env(arg_dict: dict, rank: int, seed: int = 0, model_logdir=None) -> Callable:
    """
        Utility function for multiprocessed env.

        :param arg_dict: (dict) the environment ID
        :param seed: (int) the initial seed for RNG
        :param rank: (int) index of the subprocess
        :return: (Callable)
        """

    def _init():
        if "Gym-v0" not in gym.registry:
            gym.register("Gym-v0", GymEnv)
        arg_dict["seed"] = seed + rank
        env = configure_env(arg_dict, for_train=True, model_logdir=model_logdir)
        env.reset()
        return env

    set_random_seed(seed)
    return _init


def configure_implemented_combos(env, model_logdir, arg_dict):
    implemented_combos = {"ppo": {}, "sac": {}, "td3": {}, "a2c": {}, "multippo": {}}

    implemented_combos["ppo"]["pytorch"] = [PPO_P, ('MlpPolicy', env),
                                            {"n_steps": 1024, "verbose": 1, "tensorboard_log": model_logdir,
                                             "device": "cpu"}]
    implemented_combos["sac"]["pytorch"] = [SAC_P, ('MlpPolicy', env), {"verbose": 1, "tensorboard_log": model_logdir}]
    implemented_combos["td3"]["pytorch"] = [TD3_P, ('MlpPolicy', env), {"verbose": 1, "tensorboard_log": model_logdir}]
    implemented_combos["a2c"]["pytorch"] = [A2C_P, ('MlpPolicy', env), {"n_steps": arg_dict["algo_steps"], "verbose": 1,
                                                                        "tensorboard_log": model_logdir}]
    implemented_combos["multippo"]["pytorch"] = [MultiPPOSB3, ("MlpPolicy", env),
                                                 {"n_steps": 1024, "verbose": 1, "tensorboard_log": model_logdir,
                                                  "device": "cpu", "n_models": arg_dict["num_networks"]}]
    return implemented_combos


def train(env, implemented_combos, model_logdir, arg_dict, pretrained_model=None):
    model_name = arg_dict["algo"] + '_' + str(arg_dict["steps"])
    conf_pth = os.path.join(model_logdir, "train.json")
    arg_dict["logdir"] = model_logdir #TODO: figure out whether logdir is needed (as there is a duplicity with pretrained model)
    arg_dict["pretrained_model"] = model_logdir
    seed = arg_dict.get("seed", None)
    steps = 0
    if not pretrained_model:
        #creating train.json when training from scratch
        with open(conf_pth, "w") as f:
            json.dump(arg_dict, f, indent=4)
        with open(os.path.join(model_logdir,"trained_steps.txt"), "a+") as f:
            f.write(f"model {model_name} has been saved at steps:" + "\n")
    else:
        #when loading pretrained model, figure out how many steps have already been trained
        with open(os.path.join(pretrained_model,"trained_steps.txt"), "r") as f:
            lines = f.readlines()
            line = lines[-1]
            steps = int(line)
        model_logdir = pretrained_model
    model_args = implemented_combos[arg_dict["algo"]][arg_dict["train_framework"]][1]
    model_kwargs = implemented_combos[arg_dict["algo"]][arg_dict["train_framework"]][2]
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        model_kwargs["seed"] = seed
    if pretrained_model:
        if not os.path.isabs(pretrained_model):
            pretrained_model = os.path.join(pkg_resources.files("myGym"), pretrained_model)
        env = model_args[1]
        if not arg_dict["multiprocessing"]:
            vec_env = DummyVecEnv([lambda: env])
        else:
            vec_env = env
        model = implemented_combos[arg_dict["algo"]][arg_dict["train_framework"]][0].load(pretrained_model, vec_env, device = "cpu")
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
    auto_save_callback = SaveOnBestTrainingRewardCallback(check_freq=1024, logdir=model_logdir, env=env,
                                                              engine=arg_dict["engine"],
                                                              multiprocessing=arg_dict["multiprocessing"],
                                                              save_model_every_steps=arg_dict["eval_freq"], starting_steps = steps,
                                                          algo = arg_dict["algo"])
    callbacks_list.append(auto_save_callback)
    if arg_dict["eval_freq"]:
        eval_env = env
        if arg_dict["multiprocessing"] is not None:
            NUM_CPU = int(arg_dict["multiprocessing"])
        else:
            NUM_CPU = 1
        if arg_dict["algo"] == "multippo":
            eval_callback = MultiPPOEvalCallback(eval_env, log_path=model_logdir,
                                               eval_freq=arg_dict["eval_freq"],
                                               algo_steps=arg_dict["algo_steps"],
                                               n_eval_episodes=arg_dict["eval_episodes"],
                                               record=arg_dict["record"],
                                               camera_id=arg_dict["camera"], num_cpu=NUM_CPU, starting_steps = steps)
        else:
            eval_callback = PPOEvalCallback(eval_env, log_path=model_logdir,
                                           eval_freq=arg_dict["eval_freq"],
                                           algo_steps=arg_dict["algo_steps"],
                                           n_eval_episodes=arg_dict["eval_episodes"],
                                           record=arg_dict["record"],
                                           camera_id=arg_dict["camera"], num_cpu=NUM_CPU, starting_steps = steps)
        callbacks_list.append(eval_callback)
    print("learn started")
    model.learn(total_timesteps=arg_dict["steps"], callback=callbacks_list)
    print("learn ended")
    model.save(model_logdir, steps = steps + model.num_timesteps)
    env.close()
    print("Training time: {:.2f} s".format(time.time() - start_time))
    print("Training steps: {:} s".format(model.num_timesteps))

    # info_keywords in monitor class above is necessary for pybullet to save_results
    # when using the info_keywords for mujoco we get an error
    if arg_dict["engine"] == "pybullet":
        save_results(arg_dict, model_name, env, model_logdir)
    return model


def get_parser():
    parser = argparse.ArgumentParser()
    # Environmentr
    parser.add_argument("-cfg", "--config", type=str, default = "./configs/prag_reach_kuka.json", help="Config file path") #./trained_models/AG/AG_table_tiago_tiago_dual_joints_gripper_multippo/train.json
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
    parser.add_argument("-b", "--robot", default=["kuka", "panda"], nargs='*',
                        help="Robot to train: kuka, panda, jaco ...")
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
    parser.add_argument("-i", "--multiprocessing", type=int, help="True: multiprocessing on (specify also the number of vectorized environemnts), False: multiprocessing off")
    parser.add_argument("-v", "--vectorized_envs", type=int,  help="The number of vectorized environments to run at once (mujoco multiprocessing only)")
    #Paths
    parser.add_argument("-m", "--model_path", type=str, help="Path to the the trained model to test")
    parser.add_argument("-vp", "--vae_path", type=str, help="Path to a trained VAE in 2dvu reward type")
    parser.add_argument("-yp", "--yolact_path", type=str, help="Path to a trained Yolact in 3dvu reward type")
    parser.add_argument("-yc", "--yolact_config", type=str, help="Path to saved config obj or name of an existing one in the data/Config script (e.g. 'yolact_base_config') or None for autodetection")
    parser.add_argument('-ptm', "--pretrained_model", type=str, help="Path to a model that you want to continue training")
    #Language
    parser.add_argument("-nl", "--natural_language", type=str, default="",
                        help="If passed, instead of training the script will produce a natural language output "
                             "of the given type, save it to the predefined file (for communication with other scripts) "
                             "and exit the program (without the actual training taking place). Expected values are \"description\" "
                             "(generate a task description) or \"new_tasks\" (generate new tasks)")
    return parser


def get_arguments(parser):
    args = parser.parse_args()
    commands = {}
    with open(args.config, "r") as f:
        arg_dict = commentjson.load(f)
    for key, value in arg_dict.items():
        if value is not None and key != "config":
            if key in ["robot_init"] or key in ["end_effector_orn"]:
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


def main():
    parser = get_parser()
    arg_dict, commands = get_arguments(parser)
    args = parser.parse_args()

    # for key, arg in arg_dict.items():
    #     if type(arg_dict[key]) == list:
    #         if len(arg_dict[key]) > 1 and key != "robot_init":
    #             if key != "task_objects":
    #                 parameters[key] = arg
    #                 if key in commands:
    #                     commands.pop(key)

    # Check if we chose one of the existing engines
    if arg_dict["engine"] not in AVAILABLE_SIMULATION_ENGINES:
        print(f"Invalid simulation engine. Valid arguments: --engine {AVAILABLE_SIMULATION_ENGINES}.")
        return


    if not os.path.isabs(arg_dict["logdir"]):
        arg_dict["logdir"] = os.path.join("./", arg_dict["logdir"])
    os.makedirs(arg_dict["logdir"], exist_ok=True)
    model_logdir_ori = os.path.join(arg_dict["logdir"], "_".join(
        (arg_dict["workspace"], arg_dict["robot"], arg_dict["robot_action"], arg_dict["algo"])))

    model_logdir = model_logdir_ori
    add = 1
    if not arg_dict["pretrained_model"]:
        #If training from scratch, make a new logdir for the model
        #logdir includes train.json file and monitor.csv file
        while True:
            try:
                os.makedirs(model_logdir, exist_ok=False)
                break
            except:
                model_logdir = "_".join((model_logdir_ori, str(add)))
                add += 1
    else:
        #In case of renewing training from a checkpoint, logdir with monitor.csv
        #and train.json are located in the directory where pretrained model is stored
        model_logdir = os.path.dirname(os.path.dirname(arg_dict["pretrained_model"]))
    if arg_dict["multiprocessing"]:
        NUM_CPU = int(arg_dict["multiprocessing"])
        env = SubprocVecEnv([make_env(arg_dict, i, model_logdir=model_logdir) for i in range(NUM_CPU)])
        env = VecMonitor(env, model_logdir)
    else:
        env = configure_env(arg_dict, model_logdir, for_train=1)

    implemented_combos = configure_implemented_combos(env, model_logdir, arg_dict)
    train(env, implemented_combos, model_logdir, arg_dict, arg_dict["pretrained_model"])



if __name__ == "__main__":
    main()
