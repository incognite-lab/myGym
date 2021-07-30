import pkg_resources
import os, sys, time, yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
import json, commentjson
import gym
from myGym import envs
import myGym.utils.cfg_comparator as cfg
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.bench import Monitor
from stable_baselines import results_plotter
from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper
# For now I am importing both with slightly modified names P-PyTorch T-TensorFlow
from stable_baselines import PPO2 as PPO2_T, HER as HER_T, SAC as SAC_T, DDPG as DDPG_T
from stable_baselines import TD3 as TD3_T, A2C as A2C_T, ACKTR as ACKTR_T, TRPO as TRPO_T, GAIL as GAIL_T
try:
    from stable_baselines3 import PPO as PPO_P, A2C as A2C_P, SAC as SAC_P, TD3 as TD3_P
except:
    print("Torch isn't probably installed correctly")

from myGym.stable_baselines_mygym.algo import MyAlgo
from myGym.stable_baselines_mygym.policies import MyMlpPolicy

from stable_baselines.gail import ExpertDataset, generate_expert_traj
from stable_baselines.sac.policies import MlpPolicy as MlpPolicySAC
from stable_baselines.ddpg.policies import MlpPolicy as MlpPolicyDDPG
from stable_baselines.td3.policies import MlpPolicy as MlpPolicyTD3

# Import helper classes and functions for monitoring
from myGym.utils.callbacks import ProgressBarManager, SaveOnBestTrainingRewardCallback,  PlottingCallback, CustomEvalCallback

# This is global variable for the type of engine we are working with
AVAILABLE_SIMULATION_ENGINES = ["mujoco", "pybullet"]
AVAILABLE_TRAINING_FRAMEWORKS = ["tensorflow", "pytorch"]


def save_results(arg_dict, model_name, env, model_logdir=None, show=False):
    if model_logdir is None:
        model_logdir = arg_dict["logdir"]
    print(f"model_logdir: {model_logdir}")

    results_plotter.EPISODES_WINDOW = 100
    results_plotter.plot_results([model_logdir], arg_dict["steps"], results_plotter.X_TIMESTEPS, arg_dict["algo"] + " " + arg_dict["env_name"] + " reward")
    plt.gcf().set_size_inches(8, 6)
    plt.savefig(os.path.join(model_logdir, model_name) + '_reward_results.png')
    #plot_extended_results(model_logdir, 'd', results_plotter.X_TIMESTEPS, arg_dict["algo"] + " " + arg_dict["env_name"] + " distance", "Episode Distances")
    plt.gcf().set_size_inches(8, 6)
    plt.savefig(os.path.join(model_logdir, model_name) + '_distance_results.png')
    plt.close()
    plt.close()
    results_plotter.plot_curves([(np.arange(len(env.unwrapped.episode_final_distance)),np.asarray(env.unwrapped.episode_final_distance))],'episodes',arg_dict["algo"] + " " + arg_dict["env_name"] + ' final step distance')
    plt.gcf().set_size_inches(8, 6)
    plt.ylabel("Step Distances")
    plt.savefig(os.path.join(model_logdir, model_name) + "_final_distance_results.png")
    plt.close()
    print("Congratulations! Training with {} timesteps succeed!".format(arg_dict["steps"]))
    if show:
        plt.show()

def configure_env(arg_dict, model_logdir=None, for_train=True):
    if arg_dict["engine"] == "pybullet":
        env_arguments = {"render_on": True, "visualize": arg_dict["visualize"], "workspace": arg_dict["workspace"],
                         "robot": arg_dict["robot"], "robot_init_joint_poses": arg_dict["robot_init"],
                         "robot_action": arg_dict["robot_action"], "task_type": arg_dict["task_type"], "num_subgoals": arg_dict["num_subgoals"],
                         "task_objects":arg_dict["task_objects"], "distractors":arg_dict["distractors"], 
                         "distractor_moveable":arg_dict["distractor_moveable"],
                         "distractor_constant_speed":arg_dict["distractor_constant_speed"],
                         "distractor_movement_dimensions":arg_dict["distractor_movement_dimensions"],
                         "distractor_movement_endpoints":arg_dict["distractor_movement_endpoints"],
                         "observed_links_num":arg_dict["observed_links_num"],
                         "reward_type": arg_dict["reward_type"],
                         "distance_type": arg_dict["distance_type"], "used_objects": arg_dict["used_objects"],
                         "object_sampling_area": arg_dict["object_sampling_area"], "active_cameras": arg_dict["camera"],
                         "max_steps": arg_dict["max_episode_steps"], "visgym":arg_dict["visgym"],
                         "reward": arg_dict["reward"], "logdir": arg_dict["logdir"], "vae_path": arg_dict["vae_path"],
                         "yolact_path": arg_dict["yolact_path"], "yolact_config": arg_dict["yolact_config"]}
        if for_train:
            env_arguments["gui_on"] = False
        else:
            env_arguments["gui_on"] = arg_dict["gui"]

        if arg_dict["algo"] == "her":
            env = gym.make(arg_dict["env_name"], **env_arguments, obs_space="dict")  # her needs obs as a dict
        else:
            env = gym.make(arg_dict["env_name"], **env_arguments)
    elif arg_dict["engine"] == "mujoco":
        if arg_dict["multiprocessing"]:
            # ACKTR, PPO2, A2C, DDPG can use vectorized environments, but the only way to display the results (for me) is using CV2 imshow. -(TensorFlow comment)
            env = make_vec_env(arg_dict["env_name"], n_envs=arg_dict["vectorized_envs"])
        else:
            env = gym.make(arg_dict["env_name"])
    if for_train:
        if arg_dict["engine"] == "mujoco":
            env = VecMonitor(env, model_logdir) if arg_dict["multiprocessing"] else Monitor(env, model_logdir)
        elif arg_dict["engine"] == "pybullet":
            env = Monitor(env, model_logdir, info_keywords=tuple('d'))

    if arg_dict["algo"] == "her":
        env = HERGoalEnvWrapper(env)

    return env


def configure_implemented_combos(env, model_logdir, arg_dict):
    implemented_combos = {"ppo2": {"tensorflow": [PPO2_T, (MlpPolicy, env), {"n_steps": arg_dict["algo_steps"], "verbose": 1, "tensorboard_log": model_logdir}]},
                          "ppo": {"tensorflow": [PPO2_T, (MlpPolicy, env),  {"verbose": 1, "tensorboard_log": model_logdir}],},
                          "her": {"tensorflow": [HER_T, (MlpPolicyDDPG, env, DDPG_T), {"goal_selection_strategy": 'future', "verbose": 1,"tensorboard_log": model_logdir}]},
                          "sac": {"tensorflow": [SAC_T, (MlpPolicySAC, env), {"verbose": 1, "tensorboard_log": model_logdir}],},
                          "ddpg": {"tensorflow": [DDPG_T, (MlpPolicyDDPG, env),{"verbose": 1, "tensorboard_log": model_logdir}]},
                          "td3": {"tensorflow": [TD3_T, (MlpPolicyTD3, env), {"verbose": 1, "tensorboard_log": model_logdir}],},
                          "acktr": {"tensorflow": [ACKTR_T, (MlpPolicy, env), {"n_steps": arg_dict["algo_steps"], "verbose": 1, "tensorboard_log": model_logdir}]},
                          "trpo": {"tensorflow": [TRPO_T, (MlpPolicy, env), {"verbose": 1, "tensorboard_log": model_logdir}]},
                          "gail": {"tensorflow": [SAC_T, ('MlpPolicy', env), {"verbose": 1, "tensorboard_log": model_logdir}]},
                          "a2c":    {"tensorflow": [A2C_T, (MlpPolicy, env), {"n_steps": arg_dict["algo_steps"], "verbose": 1, "tensorboard_log": model_logdir}],},
                          "myalgo": {"tensorflow": [MyAlgo, (MyMlpPolicy, env), {"n_steps": arg_dict["algo_steps"], "verbose": 1, "tensorboard_log": model_logdir}]},
                          "dual":   {"tensorflow": [PPO2_T, (MlpPolicy, env), {"n_steps": arg_dict["algo_steps"], "verbose": 1, "tensorboard_log": model_logdir}]}}

    if "PPO_P" in sys.modules:
        implemented_combos["ppo"]["pytorch"] = [PPO_P, ('MlpPolicy', env), {"n_steps": 1024, "verbose": 1, "tensorboard_log": model_logdir}]
        implemented_combos["sac"]["pytorch"] = [SAC_P, ('MlpPolicy', env), {"verbose": 1, "tensorboard_log": model_logdir}]
        implemented_combos["td3"]["pytorch"] = [TD3_P, ('MlpPolicy', env), {"verbose": 1, "tensorboard_log": model_logdir}]
        implemented_combos["a2c"]["pytorch"] = [A2C_P, ('MlpPolicy', env), {"n_steps": arg_dict["algo_steps"], "verbose": 1, "tensorboard_log": model_logdir}]

    return implemented_combos


def train(env, implemented_combos, model_logdir, arg_dict, pretrained_model=None):
    model_name = arg_dict["algo"] + '_' + str(arg_dict["steps"])
    conf_pth   = os.path.join(model_logdir, "train.json")
    model_path = os.path.join(model_logdir, "best_model.zip")
    arg_dict["model_path"] = model_path
    with open(conf_pth, "w") as f:
        json.dump(arg_dict, f)

    try:
        model_args = implemented_combos[arg_dict["algo"]][arg_dict["train_framework"]][1]
        model_kwargs = implemented_combos[arg_dict["algo"]][arg_dict["train_framework"]][2]
        if pretrained_model:
            if not os.path.isabs(pretrained_model):
                pretrained_model = pkg_resources.resource_filename("myGym", pretrained_model)
            env = model_args[1]
            vec_env = DummyVecEnv([lambda: env])
            model = implemented_combos[arg_dict["algo"]][arg_dict["train_framework"]][0].load(pretrained_model, vec_env)
        else:
            model = implemented_combos[arg_dict["algo"]][arg_dict["train_framework"]][0](*model_args, **model_kwargs)
    except:
        if arg_dict["algo"] in implemented_combos.keys():
            err = "{} is only implemented with {}".format(arg_dict["algo"], list(implemented_combos[arg_dict["algo"]].keys())[0])
        else:
            err = "{} algorithm is not implemented.".format(arg_dict["algo"])
        raise Exception(err)

    if arg_dict["algo"] == "gail":
        # Multi processing: (using MPI)
        if arg_dict["train_framework"] == 'tensorflow':
            # Generate expert trajectories (train expert)
            generate_expert_traj(model, model_name, n_timesteps=3000, n_episodes=100)
            # Load the expert dataset
            dataset = ExpertDataset(expert_path=model_name+'.npz', traj_limitation=10, verbose=1)
            model = GAIL_T('MlpPolicy', model_name, dataset, verbose=1)
            # Note: in practice, you need to train for 1M steps to have a working policy

    start_time = time.time()
    callbacks_list = []
    auto_save_callback = SaveOnBestTrainingRewardCallback(check_freq=1024, logdir=model_logdir, env=env, engine=arg_dict["engine"], multiprocessing=arg_dict["multiprocessing"])
    callbacks_list.append(auto_save_callback)
    if arg_dict["eval_freq"]:
        eval_env = configure_env(arg_dict, model_logdir, for_train=False)
        eval_callback = CustomEvalCallback(eval_env, log_path=model_logdir,
                                           eval_freq=arg_dict["eval_freq"],
                                           n_eval_episodes=arg_dict["eval_episodes"],
                                           record=arg_dict["record"],
                                           camera_id=arg_dict["camera"])
        callbacks_list.append(eval_callback)
    # plotting_callback = PlottingCallback(model_logdir)
    with ProgressBarManager(total_timesteps=arg_dict["steps"]) as progress_callback:
        callbacks_list.append(progress_callback)
        model.learn(total_timesteps=arg_dict["steps"], callback=callbacks_list)
    model.save(os.path.join(model_logdir, model_name))
    print("Training time: {:.2f} s".format(time.time() - start_time))

    # info_keywords in monitor class above is neccessary for pybullet to save_results
    # when using the info_keywords for mujoco we get an error
    if arg_dict["engine"] == "pybullet":
        save_results(arg_dict, model_name, env, model_logdir)
    return model

def get_parser():
    parser = argparse.ArgumentParser()
    #Envinronment
    parser.add_argument("-cfg", "--config", default="configs/train.json", help="Can be passed instead of all arguments")
    parser.add_argument("-n", "--env_name", type=str, help="The name of environment")
    parser.add_argument("-ws", "--workspace", type=str, help="The name of workspace")
    parser.add_argument("-p", "--engine", type=str,  help="Name of the simulation engine you want to use")
    parser.add_argument("-d", "--render", type=str,  help="Type of rendering: opengl, opencv")
    parser.add_argument("-c", "--camera", type=int, help="The number of camera used to render and record")
    parser.add_argument("-vi", "--visualize", type=int,  help="Whether visualize camera render and vision in/out or not: 1 or 0")
    parser.add_argument("-vg", "--visgym", type=int,  help="Whether visualize gym background: 1 or 0")
    parser.add_argument("-g", "--gui", type=int, help="Wether the GUI of the simulation should be used or not: 1 or 0")
    #Robot
    parser.add_argument("-b", "--robot", type=str, help="Robot to train: kuka, panda, jaco ...")
    parser.add_argument("-bi", "--robot_init", nargs="*", type=float, help="Initial robot's end-effector position")
    parser.add_argument("-ba", "--robot_action", type=str, help="Robot's action control: step - end-effector relative position, absolute - end-effector absolute position, joints - joints' coordinates")
    #Task
    parser.add_argument("-tt", "--task_type", type=str,  help="Type of task to learn: reach, push, throw, pick_and_place")
    parser.add_argument("-ns", "--num_subgoals", type=int, help="Number of subgoals in task")
    parser.add_argument("-to", "--task_objects", nargs="*", type=str, help="Object (for reach) or a pair of objects (for other tasks) to manipulate with")
    parser.add_argument("-u", "--used_objects", nargs="*", type=str, help="List of extra objects to randomly appear in the scene")
    parser.add_argument("-oa", "--object_sampling_area", nargs="*", type=float, help="Area in the scene where objects can appear")
    #Distractors
    parser.add_argument("-di", "--distractors", type=str, help="Object (for reach) to evade")
    parser.add_argument("-dm", "--distractor_moveable", type=int, help="can distractor move (0/1)")
    parser.add_argument("-ds", "--distractor_constant_speed", type=int, help="is speed of distractor constant (0/1)")
    parser.add_argument("-dd", "--distractor_movement_dimensions", type=int, help="in how many directions can the distractor move (1/2/3)")
    parser.add_argument("-de", "--distractor_movement_endpoints", nargs="*", type=float, help="2 coordinates (starting point and ending point)")
    parser.add_argument("-no", "--observed_links_num", type=int, help="number of robot links in observation space")
    #Reward
    parser.add_argument("-rt", "--reward_type", type=str, help="Type of reward: gt(ground truth), 3dvs(3D vision supervised), 2dvu(2D vision unsupervised), 6dvs(6D vision supervised)")
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

def get_arguments(parser):
    args = parser.parse_args()
    with open(args.config, "r") as f:
            arg_dict = commentjson.load(f)
    for key, value in vars(args).items():
        if value is not None and key is not "config":
            if key in ["robot_init", "object_sampling_area"]:
                arg_dict[key] = [float(arg_dict[key][i]) for i in range(len(arg_dict[key]))]
            else:
                arg_dict[key] = value
    return arg_dict


def main():
    parser = get_parser()
    arg_dict = get_arguments(parser)

    # Check if we chose one of the existing engines
    if arg_dict["engine"] not in AVAILABLE_SIMULATION_ENGINES:
        print(f"Invalid simulation engine. Valid arguments: --engine {AVAILABLE_SIMULATION_ENGINES}.")
        return
    if not os.path.isabs(arg_dict["logdir"]):
        arg_dict["logdir"] = pkg_resources.resource_filename("myGym", arg_dict["logdir"])
    os.makedirs(arg_dict["logdir"], exist_ok=True)
    model_logdir_ori = os.path.join(arg_dict["logdir"], "_".join((arg_dict["task_type"],arg_dict["workspace"],arg_dict["robot"],arg_dict["robot_action"],arg_dict["reward_type"],arg_dict["algo"])))
    model_logdir = model_logdir_ori
    add = 2
    while True:
        try:
            os.makedirs(model_logdir, exist_ok=False)
            break
        except:
            model_logdir = "_".join((model_logdir_ori, str(add)))
            add += 1

    env = configure_env(arg_dict, model_logdir, for_train=1)
    implemented_combos = configure_implemented_combos(env, model_logdir, arg_dict)
    train(env, implemented_combos, model_logdir, arg_dict, arg_dict["pretrained_model"])
    print(model_logdir)

if __name__ == "__main__":
    main()
