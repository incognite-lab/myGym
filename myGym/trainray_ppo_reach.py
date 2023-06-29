from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from train import get_parser, get_arguments, AVAILABLE_SIMULATION_ENGINES
import os
import gymnasium as gym
from gymnasium.wrappers import EnvCompatibility
from ray.tune.registry import register_env
from myGym.envs.gym_env import GymEnv
import numpy as np


def main():
    parser = get_parser()
    arg_dict = get_arguments(parser)

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

    # Set env arguments 
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
                     "yolact_path": arg_dict["yolact_path"], "yolact_config": arg_dict["yolact_config"]}

    env_arguments["gui_on"] = True

    # Register OpenAI gym env in gymnasium
    def env_creator(env_config):
        return EnvCompatibility(GymEnv(**env_config))    
    register_env('GymEnv-v0', env_creator)

    # Set up algo
    algo = (
        PPOConfig()
        .rollouts(num_rollout_workers=4) # You can try to increase or decrease based on your systems specs
        .resources(num_gpus=1) # You can try to increase or decrease based on your systems specs
        .environment(env='GymEnv-v0', env_config=env_arguments)
        .build()
    )

    #Train
    for i in range(10):
        result = algo.train()
        print(pretty_print(result))

        if i % 5 == 0:
            checkpoint_dir = algo.save()
            print(f"Checkpoint saved in directory {checkpoint_dir}")

if __name__ == "__main__":
    main()
