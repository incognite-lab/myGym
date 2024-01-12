import random
import pkg_resources
import os, sys, time, yaml
import argparse
import numpy as np
import json, commentjson
from myGym import envs
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac.sac import SACConfig
from ray.rllib.algorithms.appo import APPOConfig
from ray.rllib.algorithms.marwil import MARWILConfig
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.env.base_env import BaseEnv
from ray.tune.logger import pretty_print
from train import get_parser, get_arguments, AVAILABLE_SIMULATION_ENGINES
from gymnasium.wrappers import EnvCompatibility
from ray.tune.registry import register_env
from myGym.envs.gym_env import GymEnv

def save_results(arg_dict, model_name, env, model_logdir=None, show=False):
    if model_logdir is None:
        model_logdir = arg_dict["logdir"]
    print(f"model_logdir: {model_logdir}")

    #results_plotter.EPISODES_WINDOW = 100
    #results_plotter.plot_results([model_logdir], arg_dict["steps"], results_plotter.X_TIMESTEPS, arg_dict["algo"] + " " + arg_dict["env_name"] + " reward")
    #plt.gcf().set_size_inches(8, 6)
    #plt.savefig(os.path.join(model_logdir, model_name) + '_reward_results.png')
    #plot_extended_results(model_logdir, 'd', results_plotter.X_TIMESTEPS, arg_dict["algo"] + " " + arg_dict["env_name"] + " distance", "Episode Distances")
    #plt.gcf().set_size_inches(8, 6)
    #plt.savefig(os.path.join(model_logdir, model_name) + '_distance_results.png')
    #plt.close()
    #plt.close()
    #if isinstance(env, HERGoalEnvWrapper):
    #    results_plotter.plot_curves([(np.arange(len(env.env.episode_final_distance)),np.asarray(env.env.episode_final_distance))],'episodes',arg_dict["algo"] + " " + arg_dict["env_name"] + ' final step distance')
    #else:
    #    results_plotter.plot_curves([(np.arange(len(env.unwrapped.episode_final_distance)),np.asarray(env.unwrapped.episode_final_distance))],'episodes',arg_dict["algo"] + " " + arg_dict["env_name"] + ' final step distance')
    #plt.gcf().set_size_inches(8, 6)
    #plt.ylabel("Step Distances")
    #plt.savefig(os.path.join(model_logdir, model_name) + "_final_distance_results.png")
    #plt.close()
    print("Congratulations! Training with {} timesteps succeed!".format(arg_dict["steps"]))
    #if show:
    #    plt.show()

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
                     "max_steps": arg_dict["max_episode_steps"], "visgym":arg_dict["visgym"],
                     "reward": arg_dict["reward"], "logdir": arg_dict["logdir"], "vae_path": arg_dict["vae_path"],
                     "yolact_path": arg_dict["yolact_path"], "yolact_config": arg_dict["yolact_config"], "algo":arg_dict["algo"]}
    if for_train:
        env_arguments["gui_on"] = arg_dict["gui"]
    else:
        env_arguments["gui_on"] = arg_dict["gui"]
    return env_arguments

def env_creator(env_config):
        return EnvCompatibility(GymEnv(**env_config))    

def configure_implemented_combos(arg_dict):
    implemented_combos = {"ppo": PPOConfig, "sac": SACConfig, "marwil": MARWILConfig, "appo":APPOConfig}
    return implemented_combos[arg_dict["algo"]]

def get_parser():
    parser = argparse.ArgumentParser()
    #Envinronment
    parser.add_argument("-cfg", "--config", default="./configs/rllib_debug.json", help="Can be passed instead of all arguments")
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

def get_arguments(parser):
    args = parser.parse_args()
    with open(args.config, "r") as f:
            arg_dict = commentjson.load(f)
    for key, value in vars(args).items():
        if value is not None and key is not "config":
            if key in ["robot_init"]:
                arg_dict[key] = [float(arg_dict[key][i]) for i in range(len(arg_dict[key]))]
            else:
                arg_dict[key] = value
    return arg_dict, args

def train(args, arg_dict, algorithm, num_steps, algo_steps):
    if not args.ray_tune:
        # manual training with train loop using PPO and fixed learning rate
        print("Running manual train loop without Ray Tune.")
        # use fixed learning rate instead of grid search (needs tune)
        algo = (
            algorithm()
            .rollouts(num_rollout_workers=12) # You can try to increase or decrease based on your systems specs
            .resources(num_gpus=1) # You can try to increase or decrease based on your systems specs
            .environment(env='GymEnv-v0', env_config=arg_dict)
            .build()
        )
        # run manual training loop and print results after each iteration
        for _ in range(int(num_steps/algo_steps)):
            result = algo.train() # Runs one logical iteration of training.
            print(pretty_print(result))
            # stop training of the target train steps or reward are reached
            if result["timesteps_total"] >= num_steps:
                break
        algo.stop()
    else:
        print("Not Implemented, TODO")
        # automated run with Tune and grid search and TensorBoard
        # tuner = tune.Tuner(
        #     args.run,
        #     param_space=config.to_dict(),
        #     run_config=air.RunConfig(stop=stop),
        # )
        # results = tuner.fit()

        # if args.as_test:
        #     print("Checking if learning goals were achieved")
        #     check_learning_achieved(results, args.stop_reward)


def main():
    parser = get_parser()
    arg_dict, args = get_arguments(parser)
    ray.init(local_mode=True)
    
    # Check if we chose one of the existing engines
    if arg_dict["engine"] not in AVAILABLE_SIMULATION_ENGINES:
        print(f"Invalid simulation engine. Valid arguments: --engine {AVAILABLE_SIMULATION_ENGINES}.")
        return
    if not os.path.isabs(arg_dict["logdir"]):
        arg_dict["logdir"] = os.path.join("./", arg_dict["logdir"])
    os.makedirs(arg_dict["logdir"], exist_ok=True)
    model_logdir_ori = os.path.join(arg_dict["logdir"], "_".join((arg_dict["task_type"],arg_dict["workspace"],arg_dict["robot"],arg_dict["robot_action"],arg_dict["algo"])))
    model_logdir = model_logdir_ori
    add = 2
    while True:
        try:
            os.makedirs(model_logdir, exist_ok=False)
            break
        except:
            model_logdir = "_".join((model_logdir_ori, str(add)))
            add += 1

    num_steps = arg_dict["steps"]
    algo_steps = arg_dict["algo_steps"]
    arg_dict = configure_env(arg_dict, for_train=1)
    register_env('GymEnv-v0', env_creator)
    if not args.ray_tune:
        assert arg_dict["algo"] == "ppo", "Training without ray tune only works with PPO (rllib limitation)"
    algorithm = configure_implemented_combos(arg_dict)
    arg_dict.pop("algo")
    train(args, arg_dict, algorithm, num_steps, algo_steps)


if __name__ == "__main__":
    main()
