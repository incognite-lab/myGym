import json
import os
import warnings
from typing import Union, Optional

import gym
import imageio
import matplotlib.pyplot as plt
import numpy as np
from numpy import matrix
from stable_baselines.common.callbacks import BaseCallback, EvalCallback
from stable_baselines.common.vec_env import VecEnv, sync_envs_normalization
from stable_baselines.results_plotter import load_results, ts2xy
from tqdm.auto import tqdm


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

    def __init__(self, eval_env: Union[gym.Env, VecEnv],
                 callback_on_new_best: Optional[BaseCallback] = None,
                 n_eval_episodes: int = 10,
                 algo_steps: int = 256,
                 eval_freq: int = 10000,
                 log_path: str = None,
                 best_model_save_path: str = None,
                 deterministic: bool = False,
                 render: bool = False,
                 verbose: int = 1,
                 physics_engine = "pybullet",
                 gui_on = True,
                 record = False,
                 camera_id = 0,
                 record_steps_limit = 256):  # pybullet or mujoco
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

    def evaluate_policy(
            self,
            model: None,
            n_eval_episodes: int = 10,
            deterministic: bool = False,
    ):

        if isinstance(self.eval_env, VecEnv):
            assert self.eval_env.num_envs == 1, "You must pass only one environment when using this function"

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
            if not isinstance(self.eval_env, VecEnv) or e == 0:
                obs = self.eval_env.reset()
            done, state = False, None
            is_successful = 0
            distance_error = 0
            episode_reward = 0.0
            steps = 0
            last_network = 0
            last_steps = 0

            evaluation_env = self.eval_env.env

            srewardsteps = np.zeros(evaluation_env.reward.num_networks)
            srewardsuccess = np.zeros(evaluation_env.reward.num_networks)
            while not done:
                steps_sum += 1
                action, state = model.predict(obs, deterministic=deterministic)
                obs, reward, done, info = self.eval_env.step(action)
                if len(np.shape(action)) == 2:
                    info = info[0]
                    reward = reward[0]
                    done = done[0]

                if evaluation_env.p.getConnectionInfo()["isConnected"] != 0:
                    evaluation_env.p.addUserDebugText(
                        f"Endeff:{matrix(np.around(np.array(info['o']['additional_obs']['endeff_xyz']), 5))}",
                        [.8, .5, 0.1], textSize=1.0, lifeTime=0.5, textColorRGB=[0.0, 1, 0.0])
                    evaluation_env.p.addUserDebugText(
                        f"Object:{matrix(np.around(np.array(info['o']['actual_state']), 5))}",
                        [.8, .5, 0.15], textSize=1.0, lifeTime=0.5, textColorRGB=[0.0, 0.0, 1])
                    evaluation_env.p.addUserDebugText(f"Network:{evaluation_env.reward.current_network}",
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
                if evaluation_env.reward.current_network != last_network:
                    srewardsteps.put([last_network], steps - last_steps)
                    srewardsuccess.put([last_network], 1)
                    last_network = self.eval_env.env.reward.current_network
                    last_steps = steps
                distance_error = info['d']

                if self.physics_engine == "pybullet":
                    if self.record and e == n_eval_episodes - 1 and len(images) < self.record_steps_limit:
                        render_info = self.eval_env.render(mode="rgb_array", camera_id=self.camera_id)
                        image = render_info[self.camera_id]["image"]
                        images.append(image)
                        print(f"appending image: total size: {len(images)}]")

                if self.physics_engine == "mujoco" and self.gui_on:  # Rendering for mujoco engine
                    self.eval_env.render()
                steps += 1
            srewardsteps.put([last_network], steps - last_steps)
            if is_successful:
                srewardsuccess.put([last_network], 1)
            subrewards.append(evaluation_env.reward.network_rewards)
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
        meansgoals = np.count_nonzero(srsu) / evaluation_env.reward.num_networks / n_eval_episodes * 100

        results = {
            "episode": "{}".format(self.n_calls),
            "n_eval_episodes": "{}".format(n_eval_episodes),
            "success_episodes_num": "{}".format(success_episodes_num),
            "success_rate": "{}".format(success_episodes_num / n_eval_episodes * 100),
            "mean_distance_error": "{:.2f}".format(distance_error_sum / n_eval_episodes),
            "mean_steps_num": "{}".format(steps_sum // n_eval_episodes),
            "mean_reward": "{:.2f}".format(np.mean(episode_rewards)),
            "std_reward": "{:.2f}".format(np.std(episode_rewards)),
            "number of tasks": "{}".format(evaluation_env.task.number_tasks),
            "number of networks": "{}".format(evaluation_env.reward.num_networks),
            "mean subgoals finished": "{}".format(str(meansgoals)),
            "mean subgoal reward": "{}".format(str(meansr)),
            "mean subgoal steps": "{}".format(str(meansrs)),
        }

        for k, v in results.items():
            print(k, ':', v)

        # from HERE
        '''summary = tf.Summary(value=[tf.Summary.Value(tag='Evaluation/1.Episode_success', simple_value=(success_episodes_num/n_eval_episodes*100))])
        self.locals['writer'].add_summary(summary, self.num_timesteps)
        summary = tf.Summary(value=[tf.Summary.Value(tag='Evaluation/3.Mean_distance_error', simple_value=(distance_error_sum / n_eval_episodes))])
        self.locals['writer'].add_summary(summary, self.num_timesteps)
        summary = tf.Summary(value=[tf.Summary.Value(tag='Evaluation/4.Mean_step_num', simple_value=(steps_sum // n_eval_episodes))])
        self.locals['writer'].add_summary(summary, self.num_timesteps)
        summary = tf.Summary(value=[tf.Summary.Value(tag='Evaluation/2.Mean_reward', simple_value=np.mean(episode_rewards))])
        self.locals['writer'].add_summary(summary, self.num_timesteps)
        summary = tf.Summary(value=[tf.Summary.Value(tag='Evaluation/2.Mean_sgoals', simple_value=meansgoals)])
        self.locals['writer'].add_summary(summary, self.num_timesteps)

        for i in range (self.eval_env.env.task.number_tasks):
            #print("Task: {}".format(i)) 
            for j, (k,l) in enumerate(zip(meansr,meansrs)):  
                m = np.count_nonzero(srsu[:,j])/n_eval_episodes*100
                #print("Reward {}: {} , steps: {} , Success: {}".format(j, k, l, m )) 
                summary = tf.Summary(value=[tf.Summary.Value(tag='Task{}/Subgoal{}/Reward'.format(i,j),
                                                                              simple_value=k)])
                self.locals['writer'].add_summary(summary, self.num_timesteps)
                summary = tf.Summary(value=[tf.Summary.Value(tag='Task{}/Subgoal{}/Steps'.format(i,j),
                                                                              simple_value=l)])
                self.locals['writer'].add_summary(summary, self.num_timesteps)
                summary = tf.Summary(value=[tf.Summary.Value(tag='Task{}/Subgoal{}/Success'.format(i,j),
                                                                              simple_value=m)])
                self.locals['writer'].add_summary(summary, self.num_timesteps)
'''
        # to HERE
        self.eval_env.reset()
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
        if (self.eval_freq > 0 and self.n_calls % self.eval_freq == 0) or self.n_calls == 1:
            # Sync training and eval env if there is VecNormalize
            sync_envs_normalization(self.training_env, self.eval_env)

            results = self.evaluate_policy(self.model,
                                           n_eval_episodes=self.n_eval_episodes,
                                           deterministic=self.deterministic)

            if self.log_path is not None:
                self.evaluations_results["evaluation_after_{}_steps".format(self.n_calls)] = results
                filename = "evaluation_results.json"
                with open(os.path.join(self.log_path, filename), 'w') as f:
                    json.dump(self.evaluations_results, f, indent=4)
                print("Evaluation stored after {} calls.".format(self.n_calls))
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
                 multiprocessing=0):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.logdir = logdir
        self.save_path = os.path.join(logdir, 'best_model')
        self.best_average_reward = -np.inf
        self.engine = engine
        self.save_model_every_steps = save_model_every_steps
        # MUJOCO PART FOR NOW
        self.periodical_save_path = os.path.join(logdir, 'steps_')
        self.env = env  # Our access to running environment
        self.STATS_EVERY = stats_every
        self.save_success_graph_every_steps = save_success_graph_every_steps
        self.success_graph_mean_past_episodes = success_graph_mean_past_episodes
        self.multiprocessing = multiprocessing

    def _on_step(self) -> bool:
        # DOESNT WORK WITH MULTIPROCESSING
        if self.n_calls % self.save_model_every_steps == 0:
            print("Saving model to {}".format(self.periodical_save_path))
            self.model.save(self.periodical_save_path + f"{self.n_calls}")
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
                    self.model.save(self.save_path)

                if self.engine == "mujoco":  # Mujoco has additional prints
                    # Temporal workaround multiprocessing
                    if not self.multiprocessing and self.verbose > 0:
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
            if self.n_calls % self.save_success_graph_every_steps == 0 and not self.multiprocessing:
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


# This function is imported and called from another file.
# We can plot data along 2 axes.
# This function plots 1 or 2 types of data on y-axis
# x-axis is usually 'episodes' but it represents the total amount of datapoints in our array,
# y-axis is usually 'success_rate' and/or 'XXX' and represents the mean value of our data on some index
# in our array over some amount of past values ->  'average_x_axis_span'.
# We will calculate average from the last 'average_x_axis_span' and plot it to y-axis
def generate_and_save_mean_graph_from_1_or_2arrays(data_array_1=None, data_array_2=None,
                                                   save_dir="", average_x_axis_span=20,
                                                   axis_x_name="episodes", axis_y_names=None):
    # Initialize a storage array
    if axis_y_names is None:
        axis_y_names = ["success_rate"]
    if data_array_1 is None:
        data_array_1 = []
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
            print(f"\n Graph {axis_y_names[0]}_{axis_y_names[1]}_over_{axis_x_name} successfully saved to {image_path}")
        else:
            print(f"\n Graph {axis_y_names[0]}_over_{axis_x_name} successfully saved to {image_path}")

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
