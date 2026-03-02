import json
import os
import warnings
from typing import Union, Optional

import gymnasium as gym
import imageio
import matplotlib.pyplot as plt
import numpy as np
from numpy import matrix
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecEnv, sync_envs_normalization
from tqdm.auto import tqdm
from myGym.utils.helpers import PrintEveryNCalls
import time

np.set_printoptions(suppress = True)

#TODO: CustomEvalCallback might not be used - maybe delete



class CustomEvalCallback(EvalCallback):
    """
    Callback for evaluating an agent.
    :param eval_env: (Union[gym.Env, VecEnv]) The environment used for initialization
    :param callback_on_new_best: (Optional[BaseCallback]) Callback to trigger
        when there is a new best model according to the `mean_reward`
    :param n_eval_episodes: (int) The number of episodes to test the agent
    :param eval_freq: (int) Evaluate the agent every eval_freq call of the callback.
    :param log_path: (str) Path to a folder where the evaluations (`evaluations.npz`)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: (str) Path to a folder where the best model,
        according to performance on the eval env, will be saved.
    :param deterministic: (bool) Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: (bool) Whether to render or not the environment during evaluation
    :param verbose: (int)
    """

    def __init__(self, eval_env: Union[gym.Env, SubprocVecEnv],
                 callback_on_new_best: Optional[BaseCallback] = None,
                 n_eval_episodes: int = 10,
                 algo_steps: int = 256,
                 eval_freq: int = 10000,
                 log_path: str = None,
                 best_model_save_path: str = None,
                 deterministic: bool = False,
                 render: bool = False,
                 verbose: int = 1,
                 physics_engine="pybullet",
                 gui_on=True,
                 record=False,
                 camera_id=0,
                 record_steps_limit=256,
                 num_cpu=1):  # pybullet or mujoco
        super(EvalCallback, self).__init__(callback_on_new_best, verbose=verbose)
        self.n_eval_episodes = n_eval_episodes
        self.algo_steps = algo_steps
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.physics_engine = physics_engine
        self.gui_on = gui_on
        self.record = record
        self.camera_id = camera_id
        self.record_steps_limit = record_steps_limit
        self.is_tb_set = False
        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path
        self.evaluations_results = {}
        self.evaluations_timesteps = []
        self.evaluations_length = []
        self.num_cpu = num_cpu
        self.num_evals = 0
        self.printer = None


    def evaluate_policy(
            self,
            model: None,
            n_eval_episodes: int = 10,
            deterministic: bool = False,
    ):
        debug = False
        start_timer = time.time()
        success_episodes_num = 0
        distance_error_sum = 0
        steps_sum = 0

        episode_rewards = []
        images = []
        subrewards = []
        subrewsteps = []
        subrewsuccess = []
        print("---Evaluation----")
        for e in range(n_eval_episodes):
            # Avoid double reset, as VecEnv are reset automatically
            if not isinstance(self.eval_env, VecEnv) or e ==0:
                if isinstance(self.eval_env, VecMonitor):
                    obs = self.eval_env.reset()
                else:
                    obs, info = self.eval_env.reset()
            done, state = False, None
            is_successful = 0
            distance_error = 0
            episode_reward = 0.0
            steps = 0
            last_network = 0
            last_steps = 0
            if isinstance(self.eval_env, VecMonitor):
                # During multiprocess training, evaluation environment needs to be accessed differently
                evaluation_env = self.eval_env.get_attr("env")[0]
            else:
                evaluation_env = self.eval_env.env.env
                evaluation_env.reset()

            srewardsteps = np.zeros(evaluation_env.unwrapped.reward.num_networks)
            srewardsuccess = np.zeros(evaluation_env.unwrapped.reward.num_networks)
            while not done:
                steps_sum += 1
                action, state = model.predict(obs, deterministic=deterministic)
                if isinstance(self.eval_env, VecMonitor):
                    obs, reward, done, info = self.eval_env.step(action)
                else:
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    done = terminated or truncated

                if debug:
                    evaluation_env.p.addUserDebugText(
                        f"Endeff:{matrix(np.around(np.array(info['o']['additional_obs']['endeff_xyz']), 5))}",
                        [.8, .5, 0.1], textSize=1.0, lifeTime=0.5, textColorRGB=[0.0, 1, 0.0])
                    evaluation_env.p.addUserDebugText(
                        f"Object:{matrix(np.around(np.array(info['o']['actual_state']), 5))}",
                        [.8, .5, 0.15], textSize=1.0, lifeTime=0.5, textColorRGB=[0.0, 0.0, 1])
                    evaluation_env.p.addUserDebugText(f"Network:{evaluation_env.unwrapped.reward.current_network}",
                                                      [.8, .5, 0.25], textSize=1.0, lifeTime=0.5,
                                                      textColorRGB=[0.0, 0.0, 1])
                    evaluation_env.p.addUserDebugText(f"Subtask:{evaluation_env.task.current_task}",
                                                      [.8, .5, 0.35], textSize=1.0, lifeTime=0.5,
                                                      textColorRGB=[0.4, 0.2, 1])
                    evaluation_env.p.addUserDebugText(f"Episode:{e}",
                                                      [.8, .5, 0.45], textSize=1.0, lifeTime=0.5,
                                                      textColorRGB=[0.4, 0.2, .3])
                    evaluation_env.p.addUserDebugText(f"Step:{steps}",
                                                      [.8, .5, 0.55], textSize=1.0, lifeTime=0.5,
                                                      textColorRGB=[0.2, 0.8, 1])
                episode_reward += reward
                is_successful = not info['f']

                if evaluation_env.unwrapped.reward.current_network != last_network:
                    srewardsteps.put([last_network], steps - last_steps)
                    srewardsuccess.put([last_network], 1)
                    last_network = evaluation_env.unwrapped.reward.current_network
                    last_steps = steps
                #distance_error = self.eval_env.env.unwrapped.reward.get_distance_error(info['o'])

                #print("distance_error", distance_error)

                if self.physics_engine == "pybullet":
                    if self.record and e == n_eval_episodes - 1 and len(images) < self.record_steps_limit:
                        render_info = evaluation_env.render(mode="rgb_array", camera_id=self.camera_id)
                        image = render_info[self.camera_id]["image"]
                        images.append(image)
                        print(f"appending image: total size: {len(images)}]")

                if self.physics_engine == "mujoco" and self.gui_on:  # Rendering for mujoco engine
                    evaluation_env.render()
                steps += 1
            srewardsteps.put([last_network], steps - last_steps)
            if is_successful:
                srewardsuccess.put([last_network], 1)
            subrewards.append(evaluation_env.unwrapped.reward.network_rewards)
            subrewsteps.append(srewardsteps)
            subrewsuccess.append(srewardsuccess)
            episode_rewards.append(episode_reward)
            success_episodes_num += is_successful
            distance_error_sum += distance_error

        if self.record:
            gif_path = os.path.join(self.log_path, "last_eval_episode_after_{}_steps.gif".format(self.n_calls))
            imageio.mimsave(gif_path, [np.array(img) for i, img in enumerate(images) if i % 2 == 0], fps=15)
            os.system('./utils/gifopt -O3 --lossy=5 -o {dest} {source}'.format(source=gif_path, dest=gif_path))
            print("Record saved to " + gif_path)

        meansr = np.mean(subrewards, axis=0)
        meansrs = np.mean(subrewsteps, axis=0)
        srsu = np.array(subrewsuccess)
        meansgoals = np.count_nonzero(srsu) / evaluation_env.unwrapped.reward.num_networks / n_eval_episodes * 100
        print("n_eval_episodes:", n_eval_episodes)
        results = {
            "episode": "{}".format(self.n_calls*self.num_cpu),
            "n_eval_episodes": "{}".format(n_eval_episodes),
            "success_episodes_num": "{}".format(success_episodes_num),
            "success_rate": "{}".format(success_episodes_num / n_eval_episodes * 100),
            "mean_distance_error": "{:.2f}".format(distance_error_sum / n_eval_episodes),
            "mean_steps_num": "{}".format(steps_sum // n_eval_episodes),
            "mean_reward": "{:.2f}".format(np.mean(episode_rewards)),
            "std_reward": "{:.2f}".format(np.std(episode_rewards)),
            "number of tasks": "{}".format(evaluation_env.task.number_tasks),
            "number of networks": "{}".format(evaluation_env.unwrapped.reward.num_networks),
            "mean subgoals finished": "{}".format(str(meansgoals)),
            "mean subgoal reward": "{}".format(str(meansr)),
            "mean subgoal steps": "{}".format(str(meansrs)),
        }

        for k, v in results.items():
            print(k, ':', v)

        self.eval_env.reset()
        end_timer = time.time()
        print("evaluation time:", end_timer - start_timer)
        print("Evaluation finished successfully")
        return results

    def _init_callback(self):
        # Does not work in some corner cases, where the wrapper is different
        if not type(self.training_env) is type(self.eval_env):
            warnings.warn("Training and eval env are not of the same type"
                          "{} != {}".format(self.training_env, self.eval_env))

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def _on_step(self) -> bool:
        actual_calls = self.n_calls * self.num_cpu
        if (self.eval_freq > 0 and actual_calls > self.eval_freq*self.num_evals):
            # Sync training and eval env if there is VecNormalize
            self.num_evals += 1
            sync_envs_normalization(self.training_env, self.eval_env)

            results = self.evaluate_policy(self.model,
                                           n_eval_episodes=self.n_eval_episodes,
                                           deterministic=self.deterministic)

            if self.log_path is not None:
                self.evaluations_results["evaluation_after_{}_steps".format(actual_calls)] = results
                filename = "evaluation_results.json"
                with open(os.path.join(self.log_path, filename), 'w') as f:
                    json.dump(self.evaluations_results, f, indent=4)
                print("Evaluation stored after {} calls.".format(actual_calls))
        return True


class MultiPPOEvalCallback(EvalCallback):
    """
    Callback for evaluating an agent.
    This method is called for evaluating multippo algorithm. Works both for single process or multi process
    training.
    Evaluation is carried using only the first env of vecenv (that's why for example observation
    used is obs[0])- special methods had to be implemented for this to work.
    Also, this callback is used for multi-policy algorithms (unlike PPOEvalCallback)

    :param eval_env: (Union[gym.Env, VecEnv]) The environment used for initialization
    :param callback_on_new_best: (Optional[BaseCallback]) Callback to trigger
        when there is a new best model according to the `mean_reward`
    :param n_eval_episodes: (int) The number of episodes to test the agent
    :param eval_freq: (int) Evaluate the agent every eval_freq call of the callback.
    :param log_path: (str) Path to a folder where the evaluations (`evaluations.npz`)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: (str) Path to a folder where the best model,
        according to performance on the eval env, will be saved.
    :param deterministic: (bool) Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: (bool) Whether to render or not the environment during evaluation
    :param verbose: (int)
    """

    def __init__(self, eval_env: Union[gym.Env, SubprocVecEnv],
                 callback_on_new_best: Optional[BaseCallback] = None,
                 n_eval_episodes: int = 10,
                 algo_steps: int = 256,
                 eval_freq: int = 10000,
                 log_path: str = None,
                 best_model_save_path: str = None,
                 deterministic: bool = False,
                 render: bool = False,
                 verbose: int = 1,
                 physics_engine="pybullet",
                 gui_on=True,
                 record=False,
                 camera_id=0,
                 record_steps_limit=256,
                 num_cpu=1, starting_steps = 0):  # pybullet or mujoco
        super(EvalCallback, self).__init__(callback_on_new_best, verbose=verbose)
        self.n_eval_episodes = n_eval_episodes
        self.algo_steps = algo_steps
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.physics_engine = physics_engine
        self.gui_on = gui_on
        self.record = record
        self.camera_id = camera_id
        self.record_steps_limit = record_steps_limit
        self.is_tb_set = False
        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path
        self.evaluations_results = {}
        self.evaluations_timesteps = []
        self.evaluations_length = []
        self.num_cpu = num_cpu
        self.num_evals = 0
        self.starting_steps = starting_steps

    def evaluate_policy(
            self,
            model: None,
            n_eval_episodes: int = 10,
            deterministic: bool = False,
    ):
        start_timer = time.time()
        success_episodes_num = 0
        distance_error_sum = 0
        steps_sum = 0

        episode_rewards = []
        images = []
        subrewards = []
        subrewsteps = []
        subrewsuccess = []
        print("---Evaluation----")
        if isinstance(self.eval_env, VecEnv):
            env_reward = self.eval_env.get_attr("reward")[0]
            env_p = self.eval_env.get_attr("p")[0]
            env_task = self.eval_env.get_attr("task")[0]
        else:
            env_reward = self.eval_env.unwrapped.reward
            env_p = self.eval_env.unwrapped.p
            env_task = self.eval_env.unwrapped.task

        for e in range(n_eval_episodes): #Iterate through eval episodes
            debug = False
            obs = self.eval_env.reset()
            obs = obs[0] #Use only first env for evaluation
            done, state = False, None
            is_successful = 0
            distance_error = 0
            episode_reward = 0.0
            steps = 0
            last_network = 0
            last_steps = 0
            srewardsteps = np.zeros(env_reward.num_networks)
            srewardsuccess = np.zeros(env_reward.num_networks)
            print("Episode:", e)
            while not done: #Carry out episode steps until the episode is done
                steps_sum += 1
                if isinstance(self.eval_env, VecEnv):
                    action, state = model.eval_predict(obs, deterministic=deterministic) #Predict action in first environment
                    obs, reward, done, info, current_network = self.eval_env.eval_step(action)
                else:
                    action, state = model.predict(obs, deterministic = deterministic)
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    done = terminated or truncated
                    current_network = self.eval_env.unwrapped.reward.current_network
                 #Special eval step (uses first env of vec env only)
                if debug:
                    env_p.addUserDebugText(
                        f"Endeff:{matrix(np.around(np.array(info['o']['additional_obs']['endeff_xyz']), 5))}",
                        [.8, .5, 0.1], textSize=1.0, lifeTime=0.5, textColorRGB=[0.0, 1, 0.0])
                    env_p.addUserDebugText(
                        f"Object:{matrix(np.around(np.array(info['o']['actual_state']), 5))}",
                        [.8, .5, 0.15], textSize=1.0, lifeTime=0.5, textColorRGB=[0.0, 0.0, 1])
                    env_p.addUserDebugText(f"Network:{env_reward.current_network}",
                                                      [.8, .5, 0.25], textSize=1.0, lifeTime=0.05,
                                                      textColorRGB=[0.0, 0.0, 1])
                    env_p.addUserDebugText(f"Subtask:{env_task.current_task}",
                                                      [.8, .5, 0.35], textSize=1.0, lifeTime=0.05,
                                                      textColorRGB=[0.4, 0.2, 1])
                    env_p.addUserDebugText(f"Episode:{e}",
                                                      [.8, .5, 0.45], textSize=1.0, lifeTime=0.05,
                                                      textColorRGB=[0.4, 0.2, .3])
                    env_p.addUserDebugText(f"Step:{steps}",
                                                      [.8, .5, 0.55], textSize=1.0, lifeTime=0.5,
                                                      textColorRGB=[0.2, 0.8, 1])

                episode_reward += reward
                is_successful = not info['f']

                if current_network != last_network:
                    if not done:
                        srewardsteps.put([last_network], steps - last_steps)
                        srewardsuccess.put([last_network], 1)
                        last_network = current_network
                        last_steps = steps

                distance_error = env_reward.get_distance_error(info['o']) #Compute how far from goal is the gripper/object
                if self.physics_engine == "pybullet":
                    if self.record and e == n_eval_episodes - 1 and len(images) < self.record_steps_limit:
                        render_info = self.eval_env.render(mode="rgb_array", camera_id=self.camera_id)
                        image = render_info[self.camera_id]["image"]
                        images.append(image)
                        print(f"appending image: total size: {len(images)}]")

                if self.physics_engine == "mujoco" and self.gui_on:  # Rendering for mujoco engine
                    self.eval_env.render()
                steps += 1

            #Save all gathered eval episode values
            srewardsteps.put([last_network], steps - last_steps)
            if is_successful:
                srewardsuccess.put([last_network], 1)
            if isinstance(self.eval_env, VecEnv):
                env_reward = self.eval_env.get_attr("reward")[0]
            else:
                env_reward = self.eval_env.unwrapped.reward
            subrewards.append(env_reward.eval_network_rewards)
            subrewsteps.append(srewardsteps)
            subrewsuccess.append(srewardsuccess)
            episode_rewards.append(episode_reward)
            success_episodes_num += is_successful
            distance_error_sum += distance_error

        if self.record:
            gif_path = os.path.join(self.log_path, "last_eval_episode_after_{}_steps.gif".format(self.n_calls))
            imageio.mimsave(gif_path, [np.array(img) for i, img in enumerate(images) if i % 2 == 0], fps=15)
            os.system('./utils/gifopt -O3 --lossy=5 -o {dest} {source}'.format(source=gif_path, dest=gif_path))
            print("Record saved to " + gif_path)

        meansr = np.mean(subrewards, axis=0)
        meansrs = np.mean(subrewsteps, axis=0)
        srsu = np.array(subrewsuccess)
        meansgoals = np.count_nonzero(srsu) / env_reward.num_networks / n_eval_episodes * 100
        if isinstance(self.eval_env, VecEnv):
            env_task = self.eval_env.get_attr("task")[0]
        else:
            env_task = self.eval_env.unwrapped.task
        results = {
            "episode": "{}".format(self.n_calls*self.num_cpu),
            "n_eval_episodes": "{}".format(n_eval_episodes),
            "success_episodes_num": "{}".format(success_episodes_num),
            "success_rate": "{}".format(success_episodes_num / n_eval_episodes * 100),
            "mean_distance_error": "{:.2f}".format(distance_error_sum / n_eval_episodes),
            "mean_steps_num": "{}".format(steps_sum // n_eval_episodes),
            "mean_reward": "{:.2f}".format(np.mean(episode_rewards)),
            "std_reward": "{:.2f}".format(np.std(episode_rewards)),
            "number of tasks": "{}".format(env_task.number_tasks),
            "number of networks": "{}".format(env_reward.num_networks),
            "mean subgoals finished": "{}".format(str(meansgoals)),
            "mean subgoal reward": "{}".format(str(meansr)),
            "mean subgoal steps": "{}".format(str(meansrs)),
        }

        for k, v in results.items():
            print(k, ':', v)

        self.eval_env.reset()
        end_timer = time.time()
        print("evaluation time:", end_timer - start_timer)
        print("Evaluation finished successfully")
        return results


    def _init_callback(self):
        # Does not work in some corner cases, where the wrapper is different
        if not type(self.training_env) is type(self.eval_env):
            warnings.warn("Training and eval env are not of the same type"
                          "{} != {}".format(self.training_env, self.eval_env))

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def _on_step(self) -> bool:
        actual_calls = self.n_calls * self.num_cpu

        if (self.eval_freq > 0 and actual_calls >= self.eval_freq*self.num_evals):
            # Sync training and eval env if there is VecNormalize
            self.num_evals += 1
            sync_envs_normalization(self.training_env, self.eval_env)
            results = self.evaluate_policy(self.model,
                                           n_eval_episodes=self.n_eval_episodes,
                                           deterministic=self.deterministic)
            if self.log_path is not None:
                self.evaluations_results["evaluation_after_{}_steps".format(actual_calls+self.starting_steps)] = results
                filename = "evaluation_results.json"
                with open(os.path.join(self.log_path, filename), 'w') as f:
                    json.dump(self.evaluations_results, f, indent=4)
                print("Evaluation stored after {} calls.".format(actual_calls + self.starting_steps))
        return True



class PPOEvalCallback(EvalCallback):
    """
    Callback for evaluating an agent.
    This callback is used for training single-policy PPO.
    :param eval_env: (Union[gym.Env, VecEnv]) The environment used for initialization
    :param callback_on_new_best: (Optional[BaseCallback]) Callback to trigger
        when there is a new best model according to the `mean_reward`
    :param n_eval_episodes: (int) The number of episodes to test the agent
    :param eval_freq: (int) Evaluate the agent every eval_freq call of the callback.
    :param log_path: (str) Path to a folder where the evaluations (`evaluations.npz`)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: (str) Path to a folder where the best model,
        according to performance on the eval env, will be saved.
    :param deterministic: (bool) Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: (bool) Whether to render or not the environment during evaluation
    :param verbose: (int)
    """

    def __init__(self, eval_env: Union[gym.Env, SubprocVecEnv],
                 callback_on_new_best: Optional[BaseCallback] = None,
                 n_eval_episodes: int = 10,
                 algo_steps: int = 256,
                 eval_freq: int = 10000,
                 log_path: str = None,
                 best_model_save_path: str = None,
                 deterministic: bool = False,
                 render: bool = False,
                 verbose: int = 1,
                 physics_engine="pybullet",
                 gui_on=True,
                 record=False,
                 camera_id=0,
                 record_steps_limit=256,
                 num_cpu=1, starting_steps = 0):  # pybullet or mujoco
        super(EvalCallback, self).__init__(callback_on_new_best, verbose=verbose)
        self.n_eval_episodes = n_eval_episodes
        self.algo_steps = algo_steps
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.physics_engine = physics_engine
        self.gui_on = gui_on
        self.record = record
        self.camera_id = camera_id
        self.record_steps_limit = record_steps_limit
        self.is_tb_set = False
        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path
        self.evaluations_results = {}
        self.evaluations_timesteps = []
        self.evaluations_length = []
        self.num_cpu = num_cpu
        self.starting_steps = starting_steps
        self.num_evals = 0

    def evaluate_policy(
            self,
            model: None,
            n_eval_episodes: int = 10,
            deterministic: bool = False,
    ):
        debug = False
        success_episodes_num = 0
        distance_error_sum = 0
        steps_sum = 0

        episode_rewards = []
        images = []
        subrewards = []
        subrewsteps = []
        subrewsuccess = []
        print("---Evaluation----")
        start_timer = time.time()

        if isinstance(self.eval_env, VecEnv):
            env_reward = self.eval_env.get_attr("reward")[0]
            env_p = self.eval_env.get_attr("p")[0]
            env_task = self.eval_env.get_attr("task")[0]
        else:
            env_reward = self.eval_env.unwrapped.reward
            env_p = self.eval_env.unwrapped.p
            env_task = self.eval_env.unwrapped.task

        for e in range(n_eval_episodes):
            obs = self.eval_env.reset()
            obs = obs[0]
            done, state = False, None
            is_successful = 0
            distance_error = 0
            episode_reward = 0.0
            steps = 0
            last_network = 0
            last_steps = 0

            srewardsteps = np.zeros(env_reward.num_networks)
            srewardsuccess = np.zeros(env_reward.num_networks)
            while not done:
                steps_sum += 1
                action, state = model.predict(obs, deterministic=deterministic)

                if isinstance(self.eval_env, VecMonitor):
                    obs, reward, done, info, current_network = self.eval_env.eval_step(action)
                else:
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    done = terminated or truncated
                    current_network = self.eval_env.unwrapped.reward.current_network

                if debug:
                    env_p.addUserDebugText(
                        f"Endeff:{matrix(np.around(np.array(info['o']['additional_obs']['endeff_xyz']), 5))}",
                        [.8, .5, 0.1], textSize=1.0, lifeTime=0.5, textColorRGB=[0.0, 1, 0.0])
                    env_p.addUserDebugText(
                        f"Object:{matrix(np.around(np.array(info['o']['actual_state']), 5))}",
                        [.8, .5, 0.15], textSize=1.0, lifeTime=0.5, textColorRGB=[0.0, 0.0, 1])
                    env_p.addUserDebugText(f"Network:{current_network}",
                                                      [.8, .5, 0.25], textSize=1.0, lifeTime=0.5,
                                                      textColorRGB=[0.0, 0.0, 1])
                    env_p.addUserDebugText(f"Subtask:{env_task.current_task}",
                                                      [.8, .5, 0.35], textSize=1.0, lifeTime=0.5,
                                                      textColorRGB=[0.4, 0.2, 1])
                    env_p.addUserDebugText(f"Episode:{e}",
                                                      [.8, .5, 0.45], textSize=1.0, lifeTime=0.5,
                                                      textColorRGB=[0.4, 0.2, .3])
                    env_p.addUserDebugText(f"Step:{steps}",
                                                      [.8, .5, 0.55], textSize=1.0, lifeTime=0.5,
                                                      textColorRGB=[0.2, 0.8, 1])
                episode_reward += reward
                is_successful = not info['f']
                if current_network != last_network:
                    if not done:
                        srewardsteps.put([last_network], steps - last_steps)
                        srewardsuccess.put([last_network], 1)
                        last_network = current_network
                        last_steps = steps
                distance_error = env_reward.get_distance_error(info['o'])

                if self.physics_engine == "pybullet":
                    if self.record and e == n_eval_episodes - 1 and len(images) < self.record_steps_limit:
                        render_info = self.eval_env.render(mode="rgb_array", camera_id=self.camera_id)
                        image = render_info[self.camera_id]["image"]
                        images.append(image)
                        print(f"appending image: total size: {len(images)}]")

                if self.physics_engine == "mujoco" and self.gui_on:  # Rendering for mujoco engine
                    self.eval_env.render()
                steps += 1
            if isinstance(self.eval_env, VecEnv):
                env_reward = self.eval_env.get_attr("reward")[0]
            else:
                env_reward = self.eval_env.unwrapped.reward
            srewardsteps.put([last_network], steps - last_steps)
            if is_successful:
                srewardsuccess.put([last_network], 1)

            subrewards.append(env_reward.eval_network_rewards)
            subrewsteps.append(srewardsteps)
            subrewsuccess.append(srewardsuccess)
            episode_rewards.append(episode_reward)
            success_episodes_num += is_successful
            distance_error_sum += distance_error

        if self.record:
            gif_path = os.path.join(self.log_path, "last_eval_episode_after_{}_steps.gif".format(self.n_calls))
            imageio.mimsave(gif_path, [np.array(img) for i, img in enumerate(images) if i % 2 == 0], fps=15)
            os.system('./utils/gifopt -O3 --lossy=5 -o {dest} {source}'.format(source=gif_path, dest=gif_path))
            print("Record saved to " + gif_path)

        meansr = np.mean(subrewards, axis=0)
        meansrs = np.mean(subrewsteps, axis=0)
        srsu = np.array(subrewsuccess)
        meansgoals = np.count_nonzero(srsu) / env_reward.num_networks / n_eval_episodes * 100
        results = {
            "episode": "{}".format(self.n_calls * self.num_cpu),
            "n_eval_episodes": "{}".format(n_eval_episodes),
            "success_episodes_num": "{}".format(success_episodes_num),
            "success_rate": "{}".format(success_episodes_num / n_eval_episodes * 100),
            "mean_distance_error": "{:.2f}".format(distance_error_sum / n_eval_episodes),
            "mean_steps_num": "{}".format(steps_sum // n_eval_episodes),
            "mean_reward": "{:.2f}".format(np.mean(episode_rewards)),
            "std_reward": "{:.2f}".format(np.std(episode_rewards)),
            "number of tasks": "{}".format(env_task.number_tasks),
            "number of networks": "{}".format(env_reward.num_networks),
            "mean subgoals finished": "{}".format(str(meansgoals)),
            "mean subgoal reward": "{}".format(str(meansr)),
            "mean subgoal steps": "{}".format(str(meansrs)),
        }

        for k, v in results.items():
            print(k, ':', v)

        self.eval_env.reset()
        print("Evaluation finished successfully")
        end_timer = time.time()
        print("evaluation time:", end_timer - start_timer)
        return results

    def _init_callback(self):
        # Does not work in some corner cases, where the wrapper is different
        if not type(self.training_env) is type(self.eval_env):
            warnings.warn("Training and eval env are not of the same type"
                          "{} != {}".format(self.training_env, self.eval_env))

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def _on_step(self) -> bool:

        actual_calls = self.n_calls * self.num_cpu
        if not hasattr(self, "printer"):
            self.printer = PrintEveryNCalls("Actual_calls: ", 20)
        #self.printer(actual_calls)
        if not hasattr(self, "printer2"):
            self.printer2 = PrintEveryNCalls("self.num_cpu: ", 20)
        if not hasattr(self, "printer3"):
            self.printer3 = PrintEveryNCalls("self.n_calls: ", 20)
        #self.printer2(self.num_cpu)
        #self.printer3(self.n_calls)

        if (self.eval_freq > 0 and actual_calls >= self.eval_freq*self.num_evals):
            # Sync training and eval env if there is VecNormalize
            self.num_evals += 1
            sync_envs_normalization(self.training_env, self.eval_env)

            results = self.evaluate_policy(self.model,
                                           n_eval_episodes=self.n_eval_episodes,
                                           deterministic=self.deterministic)

            if self.log_path is not None:
                self.evaluations_results["evaluation_after_{}_steps".format(actual_calls)] = results
                filename = "evaluation_results.json"
                with open(os.path.join(self.log_path, filename), 'w') as f:
                    json.dump(self.evaluations_results, f, indent=4)
                print("Evaluation stored after {} calls.".format(actual_calls))
        return True


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param args.logdir: (str) Path to the folder where the model will be saved.
      It must contain the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    :args.engine (str) Name of our current simulation engine we are using
    :param env: (gym environment)
    :stats_every (int) How often to create new datapoints for mujoco graph in episodes
    :save_success_graph_every_steps (int) How often to save graph plotting
    successful episodes (mujoco)
    :save_model_every_steps (int) How often in steps to save our model
    :success_graph_mean_past_episodes (int) How many past episodes will be
    taken into account when calculating average success rate (mujoco)
    """

    def __init__(self, check_freq: int, logdir: str, verbose=1,
                 engine="pybullet",
                 env="None",
                 stats_every=50,
                 save_success_graph_every_steps=40_000,
                 save_model_every_steps=500_000,
                 success_graph_mean_past_episodes=30,
                 multiprocessing=0, starting_steps=0, algo = "ppo"):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.logdir = logdir
        self.save_path = logdir
        self.best_average_reward = -np.inf
        self.engine = engine
        self.save_model_every_steps = save_model_every_steps
        # MUJOCO PART FOR NOW
        self.periodical_save_path = logdir
        self.env = env  # Our access to running environment
        self.STATS_EVERY = stats_every
        self.save_success_graph_every_steps = save_success_graph_every_steps
        self.success_graph_mean_past_episodes = success_graph_mean_past_episodes
        self.starting_steps = starting_steps
        self.algo = algo
        if multiprocessing is not None:
            self.num_cpu = multiprocessing
        else:
            self.num_cpu = 1
        self.num_evals = 0 #Number of undergone evaluations


    def _on_step(self) -> bool:
        actual_calls = self.n_calls * self.num_cpu #self.n_calls doesn't calculate with multiproc
        # DOESNT WORK WITH MULTIPROCESSING (?)
        if actual_calls >= self.save_model_every_steps*self.num_evals: #Saving model just before evaluation
            print("Saving model to {}".format(self.periodical_save_path))
            self.model.save(self.periodical_save_path, steps = actual_calls + self.starting_steps)
            self.num_evals += 1
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.logdir), 'timesteps')

            episode = len(y)  # Current episode
            if episode:
                # Average training reward over the last 'self.STATS_EVERY' episodes
                average_reward = np.mean(y[-self.STATS_EVERY:])
                # Save the new best model (with the best average reward)
                if average_reward > self.best_average_reward:
                    self.best_average_reward = average_reward
                    self.model.save(self.save_path, steps = self.starting_steps + actual_calls, best = True)
                if self.engine == "mujoco":  # Mujoco has additional prints
                    # Temporal workaround multiprocessing
                    if not self.num_cpu==1 and self.verbose > 0:
                        # Current success rate over the last 'self.STATS_EVERY' episodes
                        current_success_rate = np.mean(self.env.successfull_failed_episodes[-self.STATS_EVERY:])
                        print(f"Best average reward: {self.best_average_reward:.2f} \
                              - Last average reward: {average_reward:.2f} \
                              and current success rate: {current_success_rate:.2f} \
                              over the last {self.STATS_EVERY} episodes \
                              - Current touch threshold: {self.env.sensor_goal_threshold:.2f} - \
                              Current episode: {episode}")

        if self.engine == "mujoco":
            # Save graph of success rate every 'self.save_success_graph_every_steps'.
            if actual_calls % self.save_success_graph_every_steps == 0 and self.num_cpu==1:
                # Save graph
                generate_and_save_mean_graph_from_1_or_2arrays(
                    data_array_1=self.env.successfull_failed_episodes,
                    data_array_2=None,
                    save_dir=self.logdir,
                    average_x_axis_span=self.success_graph_mean_past_episodes,
                    axis_x_name="episodes",
                    axis_y_names=["success_rate"])
        return True


class ProgressBarCallback(BaseCallback):
    """
    :param pbar: (tqdm.pbar) Progress bar object
    """

    def __init__(self, pbar):
        super(ProgressBarCallback, self).__init__()
        self._pbar = pbar

    def _on_step(self):
        # Update the progress bar:
        self._pbar.n = self.num_timesteps
        self._pbar.update(0)


# This callback uses the 'with' block, allowing for correct initialization and destruction
class ProgressBarManager(object):
    def __init__(self, total_timesteps):  # Init object with total timesteps
        self.pbar = None
        self.total_timesteps = total_timesteps

    def __enter__(self):  # Create the progress bar and callback, return the callback
        self.pbar = tqdm(total=self.total_timesteps)

        return ProgressBarCallback(self.pbar)

    def __exit__(self, exc_type, exc_val, exc_tb):  # Close the callback
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()


class PlottingCallback(BaseCallback):
    """
    Callback for plotting the performance in realtime.

    :param verbose: (int)
    """

    def __init__(self, logdir: str, verbose=1):
        super(PlottingCallback, self).__init__(verbose)
        self._plot = None
        self.logdir = logdir

    def _on_step(self):
        # get the monitor's data
        x, y = ts2xy(load_results(self.logdir), 'timesteps')
        if self._plot is None:  # Make the plot
            plt.ion()
            fig = plt.figure(figsize=(6, 3))
            ax = fig.add_subplot(111)
            line, = ax.plot(x, y)
            self._plot = (line, ax, fig)
            plt.title('Realtime reward')
            plt.show()
        else:  # Update and rescale the plot
            self._plot[0].set_data(x, y)
            self._plot[-2].relim()
            self._plot[-2].set_xlim([self.locals["total_timesteps"] * -0.02,
                                     self.locals["total_timesteps"] * 1.02])
            self._plot[-2].autoscale_view(True, True, True)
            self._plot[-1].canvas.draw()


# This function is imported and called from another file. We can plot data along
# 2 axes. This function plots 1 or 2 types of data on y-axis
# x-axis is usually 'episodes' but it represents the total amount of datapoints in our array,
# y-axis is usually 'success_rate' and/or 'XXX' and represents the mean value of our data on some index
# in our array over some amount of past values ->  'average_x_axis_span'. We will calculate
# average from the last 'average_x_axis_span' and plot it to y-axis
def generate_and_save_mean_graph_from_1_or_2arrays(data_array_1=[], data_array_2=None,
                                                   save_dir="", average_x_axis_span=20,
                                                   axis_x_name="episodes", axis_y_names=["success_rate"]):
    # Initialize a storage array
    data_to_plot = {axis_x_name: [], axis_y_names[0]: []}
    if data_array_2 is not None:
        data_to_plot = {axis_x_name: [], axis_y_names[0]: [], axis_y_names[1]: []}

    x_axis_length = len(data_array_1)  # Total episode count taken from array 1
    # x_axis_counter <==> f.e. episodes
    for x_axis_counter in range(x_axis_length):  # Iterate through every 'episode'/datapoint
        # Add data to plotter
        if (x_axis_counter % average_x_axis_span) == 0:
            # Average data over last 'average_episodes_span'
            mean_y_value_1 = np.mean(data_array_1[x_axis_counter: (x_axis_counter + average_x_axis_span)])

            if data_array_2 is not None:
                mean_y_value_2 = np.mean(data_array_2[x_axis_counter: (x_axis_counter + average_x_axis_span)])
            # Append the data
            data_to_plot[axis_x_name].append(x_axis_counter)
            data_to_plot[axis_y_names[0]].append(mean_y_value_1)

            if data_array_2 is not None:
                data_to_plot[axis_y_names[1]].append(mean_y_value_2)

    # Save data as an image
    print(f"len: data_to_plot[axis_y_names[0]: {len(data_to_plot[axis_y_names[0]])}")
    print(f"data_to_plot[axis_y_names[0]: {data_to_plot[axis_y_names[0]]}")
    plt.plot(data_to_plot[axis_x_name], data_to_plot[axis_y_names[0]], label=axis_y_names[0])

    if data_array_2 is not None:
        plt.plot(data_to_plot[axis_x_name], data_to_plot[axis_y_names[1]], label=axis_y_names[1])

    plt.legend(loc=2)

    if save_dir == "":
        print("Save directory for image not specified! -> 'generate_and_save_mean_graph_from_array()'")
    else:
        if data_array_2 is not None:
            image_path = save_dir + f"/graph_{axis_y_names[0]}_{axis_y_names[1]}_over_{axis_x_name}" + '.png'
        else:
            image_path = save_dir + f"/graph_{axis_y_names[0]}_over_{axis_x_name}" + '.png'

        plt.savefig(image_path)

        if data_array_2 is not None:
            print(f"\n Graph {axis_y_names[0]}_{axis_y_names[1]}_over_{axis_x_name} succesfully saved to {image_path}")
        else:
            print(f"\n Graph {axis_y_names[0]}_over_{axis_x_name} succesfully saved to {image_path}")

        plt.close()
    return


class SaveOnTopRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param args.logdir: (str) Path to the folder where the model will be saved.
      It must contain the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, logdir: str, verbose=1, models_num=2):
        super(SaveOnTopRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.logdir = logdir
        self.save_path = os.path.join(logdir, 'highest_reward')
        self.top_rewards = [-np.inf] * models_num

    def _on_rollout_end(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            x, y = ts2xy(load_results(self.logdir), 'timesteps')
            episode = len(y)  # Current episode
            if episode:
                i = 0
                for submodel in self.model.models:
                    actual_reward = submodel.episode_reward
                    if actual_reward > self.top_rewards[i]:
                        self.top_rewards[i] = actual_reward
                        if self.verbose > 0:
                            print("Saving new most rewarded model to {}".format(self.save_path))
                        submodel.save(self.save_path, i)
                    i += 1
        return True




