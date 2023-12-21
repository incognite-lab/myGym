from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.env.base_env import BaseEnv
from ray.tune.logger import pretty_print
from train import get_parser, get_arguments, AVAILABLE_SIMULATION_ENGINES
import os
import gymnasium as gym
from gymnasium.wrappers import EnvCompatibility
from ray.tune.registry import register_env
from myGym.envs.gym_env import GymEnv
import numpy as np
import time

from typing import Dict, Tuple
import argparse
import gymnasium as gym


class MyCallbacks(DefaultCallbacks):
    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs,
    ):
        # Make sure this episode has just been started (only initial obs
        # logged so far).
        assert episode.length == 0, (
            "ERROR: `on_episode_start()` callback should be called right "
            "after env reset!"
        )
        # Create lists to store angles in
        episode.user_data["pole_angles"] = []
        episode.hist_data["pole_angles"] = []

    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs,
    ):
        # Make sure this episode is ongoing.
        assert episode.length > 0, (
            "ERROR: `on_episode_step()` callback should not be called right "
            "after env reset!"
        )
        pole_angle = abs(episode.last_observation_for()[2])
        raw_angle = abs(episode.last_raw_obs_for()[2])
        assert pole_angle == raw_angle
        episode.user_data["pole_angles"].append(pole_angle)

        # Sometimes our pole is moving fast. We can look at the latest velocity
        # estimate from our environment and log high velocities.
        if np.abs(episode.last_info_for()["pole_angle_vel"]) > 0.25:
            print("This is a fast pole!")

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs,
    ):
        # Check if there are multiple episodes in a batch, i.e.
        # "batch_mode": "truncate_episodes".
        if worker.config.batch_mode == "truncate_episodes":
            # Make sure this episode is really done.
            assert episode.batch_builder.policy_collectors["default_policy"].batches[
                -1
            ]["dones"][-1], (
                "ERROR: `on_episode_end()` should only be called "
                "after episode is done!"
            )
        pole_angle = np.mean(episode.user_data["pole_angles"])
        episode.custom_metrics["pole_angle"] = pole_angle
        episode.hist_data["pole_angles"] = episode.user_data["pole_angles"]

    def on_sample_end(self, *, worker: RolloutWorker, samples: SampleBatch, **kwargs):
        # We can also do our own sanity checks here.
        assert (
            samples.count == 2000
        ), f"I was expecting 2000 here, but got {samples.count}!"

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        # you can mutate the result dict to add new fields to return
        result["callback_ok"] = True

        # Normally, RLlib would aggregate any custom metric into a mean, max and min
        # of the given metric.
        # For the sake of this example, we will instead compute the variance and mean
        # of the pole angle over the evaluation episodes.
        pole_angle = result["custom_metrics"]["pole_angle"]
        var = np.var(pole_angle)
        mean = np.mean(pole_angle)
        result["custom_metrics"]["pole_angle_var"] = var
        result["custom_metrics"]["pole_angle_mean"] = mean
        # We are not interested in these original values
        del result["custom_metrics"]["pole_angle"]
        del result["custom_metrics"]["num_batches"]

    def on_learn_on_batch(
        self, *, policy: Policy, train_batch: SampleBatch, result: dict, **kwargs
    ) -> None:
        result["sum_actions_in_train_batch"] = train_batch["actions"].sum()
        # Log the sum of actions in the train batch.
        print(
            "policy.learn_on_batch() result: {} -> sum actions: {}".format(
                policy, result["sum_actions_in_train_batch"]
            )
        )

    def on_postprocess_trajectory(
        self,
        *,
        worker: RolloutWorker,
        episode: Episode,
        agent_id: str,
        policy_id: str,
        policies: Dict[str, Policy],
        postprocessed_batch: SampleBatch,
        original_batches: Dict[str, Tuple[Policy, SampleBatch]],
        **kwargs,
    ):
        if "num_batches" not in episode.custom_metrics:
            episode.custom_metrics["num_batches"] = 0
        episode.custom_metrics["num_batches"] += 1


def main():
    #Start time counter and print it
    start_time = time.time()
    print(start_time)
    parser = get_parser()
    arg_dict = get_arguments(parser)

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
    env_arguments = {"render_on": False, "visualize": arg_dict["visualize"], "workspace": arg_dict["workspace"],
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

    env_arguments["gui_on"] = False

    # Register OpenAI gym env in gymnasium
    def env_creator(env_config):
        return EnvCompatibility(GymEnv(**env_config))    
    register_env('GymEnv-v0', env_creator)

    # Set up algo
    algo = (
        PPOConfig()
        .rollouts(num_rollout_workers=12) # You can try to increase or decrease based on your systems specs
        .resources(num_gpus=1) # You can try to increase or decrease based on your systems specs
        .environment(env='GymEnv-v0', env_config=env_arguments)
        .build()
    )

    #Train
    for i in range(200):
        result = algo.train()
        print(pretty_print(result))

        if i % 100 == 0:
            #action = algo.compute_single_action(obs)
            #obs, reward, terminated, truncated, info = env.step(action)
            checkpoint_dir = algo.save()
            print(f"Checkpoint saved in directory {checkpoint_dir}")
    #Print start time minus end time
    print("--- %s seconds ---" % (time.time() - start_time))
if __name__ == "__main__":
    main()
