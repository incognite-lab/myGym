#from stable_baselines.results_plotter import load_results, ts2xy
#from stable_baselines.common.callbacks import BaseCallback, EvalCallback
#from stable_baselines import results_plotter

#from pygifsicle import optimize

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from typing import Union, List, Dict, Any, Optional
from tqdm.auto import tqdm
import numpy as np
from ray.tune.logger import pretty_print

#from utils.callbacks import MyCallbacks
#from myGym.trainray import delete_unnecessary_logs
#from stable_baselines.common.vec_env import VecEnv, sync_envs_normalization, DummyVecEnv
#from stable_baselines.common.evaluation import evaluate_policy

#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)



def delete_unnecessary_logs(res):
    """Deletes unnecessary logged metrics after each episode to get a more readable log"""
    result = res.copy()
    unnecessary_keys = {
                        "info": {
                                "learner":
                                       {
                                           "default_policy" :{
                                                "curr_entropy_coeff": None, "curr_kl_coeff": None, "curr_lr": None,
                                                "default_optimizer_lr":None
                                            },
                                           "__all__": None
                                       }
                                ,"num_agent_steps_trained": None, "num_env_steps_sampled": None,
                            "num_env_steps_trained": None
                        },
                        "sampler_perf":{
                                "mean_env_render_ms": None
                        },
                        "sampler_results":{
                                "connector_metrics": None, "custom_metrics" : None, "episode_media": None, "num_faulty_episodes": None,
                                "policy_reward_max":None, "policy_reward_mean": None, "policy_reward_min": None, "sampler_perf": None,
                                "episode_len_mean": None, "episode_reward_mean": None, "episode_reward_max": None, "episode_reward_min": None,
                                "episodes_this_iter": None
                        },
                        "connector_metrics": None, "counters": None, "date": None, "done": None, "episode_media": None,
                        "episodes_total": None, "hostname": None, "node_ip": None, "num_agent_steps_sampled": None,
                        "num_agent_steps_trained": None, "num_env_steps_sampled": None, "num_env_steps_trained": None,
                        "num_env_steps_trained_this_iter": None, "num_env_steps_trained_throughput_per_sec": None,
                        "num_faulty_episodes": None,"num_in_flight_async_reqs": None, "num_remote_worker_restarts": None,
                        "num_steps_trained_this_iter": None, "pid": None, "time_total_s": None, "timers": None,
                        "trial_id": None, "custom_metrics": None, "policy_reward_max": None, "policy_reward_min": None,
                        "policy_reward_mean": None, "agent_timesteps_total": None
                    }
    result = remove_keys_in_dict(result, unnecessary_keys)
    return result

def remove_keys_in_dict(change_dict, rm_dict):
    """Recursive function to delete specific keys from a nested dictionary"""
    ch_copy = change_dict.copy()
    rm_copy = rm_dict.copy()
    for key in rm_copy.keys():

        val = rm_copy[key]
        if val == None:
            #Remove dictionary value

            del ch_copy[key]
        elif isinstance(val, dict):
            ch_copy[key] = remove_keys_in_dict(ch_copy[key], rm_copy[key]).copy()
    return ch_copy

class EvalCallbackRay(DefaultCallbacks):
    def on_evaluate_start(
            self,
            *,
            algorithm: "Algorithm",
            **kwargs,
    ) -> None:
        """Callback before evaluation starts.

        This method gets called at the beginning of Algorithm.evaluate().

        Args:
            algorithm: Reference to the algorithm instance.
            metrics_logger: The MetricsLogger object inside the `Algorithm`. Can be
                used to log custom metrics before running the next round of evaluation.
            kwargs: Forward compatibility placeholder.
        """
        print("#---------Starting evaluation--------------# ")
    def on_evaluate_end(
        self,
        *,
        algorithm: "Algorithm",
        evaluation_metrics: dict,
        **kwargs,
    ) -> None:
        """Runs when the evaluation is done.

        Runs at the end of Algorithm.evaluate().

        Args:
            algorithm: Reference to the algorithm instance.
            metrics_logger: The MetricsLogger object inside the `Algorithm`. Can be
                used to log custom metrics after the most recent evaluation round.
            evaluation_metrics: Results dict to be returned from algorithm.evaluate().
                You can mutate this object to add additional metrics.
            kwargs: Forward compatibility placeholder.
        """

        cropped_metrics = delete_unnecessary_logs(evaluation_metrics)
        ep_lengths = cropped_metrics["sampler_results"]["hist_stats"]["episode_lengths"]
        success_count = 0.
        ep_len_sum = 0
        episodes = len(ep_lengths)
        for ep_len in ep_lengths:
            if ep_len != 512:
                success_count += 1.
            ep_len_sum += ep_len
        success_rate =  success_count/episodes
        mean_steps_num = ep_len_sum/episodes
        print("#---------Evaluation-Summary---------#")
        print("{} of {} episodes ({} %) were successful".format(success_count, episodes,
                                                                success_rate * 100))
        print("Mean number of steps {}".format(mean_steps_num))
        print("#------------------------------------#")
        model_name = arg_dict["algo"] + '_' + str(arg_dict["steps"])

        file = open(os.path.join(model_logdir, "train_" + model_name + ".txt"), 'a')
        file.write("\n")
        file.write("#Evaluation results: \n")
        file.write("#{} of {} episodes were successful \n".format(success_episodes_num, arg_dict["eval_episodes"]))
        file.write("#Mean number of steps {}\n".format(mean_steps_num))
        file.close()
        #print(pretty_print(cropped_metrics))


# class MyCallbacks(DefaultCallbacks):
#     def on_episode_start(
#         self,
#         *,
#         worker: RolloutWorker,
#         base_env: BaseEnv,
#         policies: Dict[str, Policy],
#         episode: Episode,
#         env_index: int,
#         **kwargs,
#     ):
#         # Make sure this episode has just been started (only initial obs
#         # logged so far).
#         assert episode.length == 0, (
#             "ERROR: `on_episode_start()` callback should be called right "
#             "after env reset!"
#         )
#         # Create lists to store angles in
#         episode.user_data["pole_angles"] = []
#         episode.hist_data["pole_angles"] = []
#
#     def on_episode_step(
#         self,
#         *,
#         worker: RolloutWorker,
#         base_env: BaseEnv,
#         policies: Dict[str, Policy],
#         episode: Episode,
#         env_index: int,
#         **kwargs,
#     ):
#         # Make sure this episode is ongoing.
#         assert episode.length > 0, (
#             "ERROR: `on_episode_step()` callback should not be called right "
#             "after env reset!"
#         )
#         pole_angle = abs(episode.last_observation_for()[2])
#         raw_angle = abs(episode.last_raw_obs_for()[2])
#         assert pole_angle == raw_angle
#         episode.user_data["pole_angles"].append(pole_angle)
#
#         # Sometimes our pole is moving fast. We can look at the latest velocity
#         # estimate from our environment and log high velocities.
#         if np.abs(episode.last_info_for()["pole_angle_vel"]) > 0.25:
#             print("This is a fast pole!")
#
#     def on_episode_end(
#         self,
#         *,
#         worker: RolloutWorker,
#         base_env: BaseEnv,
#         policies: Dict[str, Policy],
#         episode: Episode,
#         env_index: int,
#         **kwargs,
#     ):
#         # Check if there are multiple episodes in a batch, i.e.
#         # "batch_mode": "truncate_episodes".
#         if worker.config.batch_mode == "truncate_episodes":
#             # Make sure this episode is really done.
#             assert episode.batch_builder.policy_collectors["default_policy"].batches[
#                 -1
#             ]["dones"][-1], (
#                 "ERROR: `on_episode_end()` should only be called "
#                 "after episode is done!"
#             )
#         pole_angle = np.mean(episode.user_data["pole_angles"])
#         episode.custom_metrics["pole_angle"] = pole_angle
#         episode.hist_data["pole_angles"] = episode.user_data["pole_angles"]
#
#     def on_sample_end(self, *, worker: RolloutWorker, samples: SampleBatch, **kwargs):
#         # We can also do our own sanity checks here.
#         assert (
#             samples.count == 2000
#         ), f"I was expecting 2000 here, but got {samples.count}!"
#
#     def on_train_result(self, *, algorithm, result: dict, **kwargs):
#         # you can mutate the result dict to add new fields to return
#         result["callback_ok"] = True
#
#         # Normally, RLlib would aggregate any custom metric into a mean, max and min
#         # of the given metric.
#         # For the sake of this example, we will instead compute the variance and mean
#         # of the pole angle over the evaluation episodes.
#         pole_angle = result["custom_metrics"]["pole_angle"]
#         var = np.var(pole_angle)
#         mean = np.mean(pole_angle)
#         result["custom_metrics"]["pole_angle_var"] = var
#         result["custom_metrics"]["pole_angle_mean"] = mean
#         # We are not interested in these original values
#         del result["custom_metrics"]["pole_angle"]
#         del result["custom_metrics"]["num_batches"]
#
#     def on_learn_on_batch(
#         self, *, policy: Policy, train_batch: SampleBatch, result: dict, **kwargs
#     ) -> None:
#         result["sum_actions_in_train_batch"] = train_batch["actions"].sum()
#         # Log the sum of actions in the train batch.
#         print(
#             "policy.learn_on_batch() result: {} -> sum actions: {}".format(
#                 policy, result["sum_actions_in_train_batch"]
#             )
#         )
#
#     def on_postprocess_trajectory(
#         self,
#         *,
#         worker: RolloutWorker,
#         episode: Episode,
#         agent_id: str,
#         policy_id: str,
#         policies: Dict[str, Policy],
#         postprocessed_batch: SampleBatch,
#         original_batches: Dict[str, Tuple[Policy, SampleBatch]],
#         **kwargs,
#     ):
#         if "num_batches" not in episode.custom_metrics:
#             episode.custom_metrics["num_batches"] = 0
#         episode.custom_metrics["num_batches"] += 1
#
#




# class CustomEvalCallback(EvalCallback):
#     """
#     Callback for evaluating an agent.
#     :param eval_env: (Union[gym.Env, VecEnv]) The environment used for initialization
#     :param callback_on_new_best: (Optional[BaseCallback]) Callback to trigger
#         when there is a new best model according to the `mean_reward`
#     :param n_eval_episodes: (int) The number of episodes to test the agent
#     :param eval_freq: (int) Evaluate the agent every eval_freq call of the callback.
#     :param log_path: (str) Path to a folder where the evaluations (`evaluations.npz`)
#         will be saved. It will be updated at each evaluation.
#     :param best_model_save_path: (str) Path to a folder where the best model
#         according to performance on the eval env will be saved.
#     :param deterministic: (bool) Whether the evaluation should
#         use a stochastic or deterministic actions.
#     :param render: (bool) Whether to render or not the environment during evaluation
#     :param verbose: (int)
#     """
#     def __init__(self, eval_env: Union[gym.Env, VecEnv],
#                  callback_on_new_best: Optional[BaseCallback] = None,
#                  n_eval_episodes: int = 10,
#                  algo_steps: int = 256,
#                  eval_freq: int = 10000,
#                  log_path: str = None,
#                  best_model_save_path: str = None,
#                  deterministic: bool = False,
#                  render: bool = False,
#                  verbose: int = 1,
#                  physics_engine = "pybullet",
#                  gui_on = True,
#                  record=False,
#                  camera_id=0,
#                  record_steps_limit=256): # pybullet or mujoco
#         super(EvalCallback, self).__init__(callback_on_new_best, verbose=verbose)
#         self.n_eval_episodes = n_eval_episodes
#         self.algo_steps = algo_steps
#         self.eval_freq = eval_freq
#         self.best_mean_reward = -np.inf
#         self.last_mean_reward = -np.inf
#         self.deterministic = deterministic
#         self.render = render
#         self.physics_engine = physics_engine
#         self.gui_on = gui_on
#         self.record = record
#         self.camera_id = camera_id
#         self.record_steps_limit = record_steps_limit
#         self.is_tb_set = False

#         # Convert to VecEnv for consistency
#         # if not isinstance(eval_env, VecEnv):
#         #     eval_env = DummyVecEnv([lambda: eval_env])

#         #assert eval_env.num_envs == 1, "You must pass only one environment for evaluation"

#         self.eval_env = eval_env
#         self.best_model_save_path = best_model_save_path
#         # # Logs will be written in `evaluations.npz`
#         # if log_path is not None:
#         #     log_path = os.path.join(log_path, 'evaluations')
#         self.log_path = log_path
#         self.evaluations_results = {}
#         self.evaluations_timesteps = []
#         self.evaluations_length = []

#     def evaluate_policy(
#         self,
#         model: None,
#         #env: Union[gym.Env, VecEnv],
#         n_eval_episodes: int = 10,
#         deterministic: bool = False,
#     ):

#         if isinstance(self.eval_env, VecEnv):
#             assert self.eval_env.num_envs == 1, "You must pass only one environment when using this function"

#         success_episodes_num = 0
#         distance_error_sum = 0
#         steps_sum = 0

#         episode_rewards = []
#         images = []
#         subrewards = []
#         subrewsteps = []
#         subrewsuccess = []
#         print("---Evaluation----")
#         for e in range(n_eval_episodes):

#             # Avoid double reset, as VecEnv are reset automatically
#             if not isinstance(self.eval_env, VecEnv) or e == 0:
#                 obs = self.eval_env.reset()
#             done, state = False, None
#             is_successful = 0
#             distance_error = 0
#             episode_reward = 0.0
#             steps=0
#             last_network = 0
#             last_steps = 0
#             srewardsteps = np.zeros(self.eval_env.env.reward.num_networks)
#             srewardsuccess = np.zeros(self.eval_env.env.reward.num_networks)

#             while not done:
#                 steps_sum += 1
#                 action, state = model.predict(obs, deterministic=deterministic)
#                 obs, reward, done, info = self.eval_env.step(action)
#                 p.addUserDebugText(f"Action (Gripper):{matrix(np.around(np.array(action),5))}",
#                     [.8, .5, 0.2], textSize=1.0, lifeTime=0.5, textColorRGB=[1, 0, 0])
#                 #p.addUserDebugText(f"Endeff:{matrix(np.around(np.array(info['o']['additional_obs']['endeff_xyz']),5))}",
#                 #    [.8, .5, 0.1], textSize=1.0, lifeTime=0.5, textColorRGB=[0.0, 1, 0.0])
#                 p.addUserDebugText(f"Object:{matrix(np.around(np.array(info['o']['actual_state']),5))}",
#                     [.8, .5, 0.15], textSize=1.0, lifeTime=0.5, textColorRGB=[0.0, 0.0, 1])
#                 p.addUserDebugText(f"Network:{self.eval_env.env.reward.current_network}",
#                     [.8, .5, 0.25], textSize=1.0, lifeTime=0.5, textColorRGB=[0.0, 0.0, 1])
#                 p.addUserDebugText(f"Subtask:{self.eval_env.env.task.current_task}",
#                     [.8, .5, 0.35], textSize=1.0, lifeTime=0.5, textColorRGB=[0.4, 0.2, 1])
#                 p.addUserDebugText(f"Episode:{e}",
#                     [.8, .5, 0.45], textSize=1.0, lifeTime=0.5, textColorRGB=[0.4, 0.2, .3])
#                 p.addUserDebugText(f"Step:{steps}",
#                     [.8, .5, 0.55], textSize=1.0, lifeTime=0.5, textColorRGB=[0.2, 0.8, 1])
#                 #print(f"Network:{self.eval_env.env.reward.current_network})
#                 episode_reward += reward
#                 # Info is list with dict inside
#                 #info = info[0]
#                 is_successful = not info['f']
#                 if self.eval_env.env.reward.current_network != last_network:
#                     srewardsteps.put([last_network], steps-last_steps)
#                     srewardsuccess.put([last_network], 1)
#                     last_network = self.eval_env.env.reward.current_network
#                     last_steps = steps
#                 distance_error = info['d']


#                 if self.physics_engine == "pybullet":
#                     if self.record and e == n_eval_episodes - 1 and len(images) < self.record_steps_limit:
#                         render_info = self.eval_env.render(mode="rgb_array", camera_id = self.camera_id)
#                         image = render_info[self.camera_id]["image"]
#                         images.append(image)
#                         print(f"appending image: total size: {len(images)}]")

#                 if self.physics_engine == "mujoco" and self.gui_on: # Rendering for mujoco engine
#                     self.eval_env.render()
#                 steps +=1      
#             srewardsteps.put([last_network], steps-last_steps)
#             if is_successful:
#                 srewardsuccess.put([last_network], 1)
#             subrewards.append(self.eval_env.env.reward.network_rewards)            
#             subrewsteps.append(srewardsteps)
#             subrewsuccess.append(srewardsuccess)
#             episode_rewards.append(episode_reward)
#             success_episodes_num += is_successful
#             distance_error_sum += distance_error

#         if self.record:
#             gif_path = os.path.join(self.log_path, "last_eval_episode_after_{}_steps.gif".format(self.n_calls))
#             imageio.mimsave(gif_path, [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=15)
#             #optimize(gif_path)
#             os.system('./utils/gifopt -O3 --lossy=5 -o {dest} {source}'.format(source=gif_path, dest=gif_path)) 
#             print("Record saved to " + gif_path)

        

#         meansr = np.mean(subrewards, axis=0)
#         meansrs = np.mean(subrewsteps, axis=0)
#         srsu = np.array(subrewsuccess)
#         meansgoals=np.count_nonzero(srsu)/self.eval_env.env.reward.num_networks/n_eval_episodes*100

#         results = {
#             "episode": "{}".format(self.n_calls),
#             "n_eval_episodes": "{}".format(n_eval_episodes),
#             "success_episodes_num": "{}".format(success_episodes_num),
#             "success_rate": "{}".format(success_episodes_num/n_eval_episodes*100),
#             "mean_distance_error": "{:.2f}".format(distance_error_sum / n_eval_episodes),
#             "mean_steps_num": "{}".format(steps_sum // n_eval_episodes),
#             "mean_reward": "{:.2f}".format(np.mean(episode_rewards)),
#             "std_reward": "{:.2f}".format(np.std(episode_rewards)),
#             "number of tasks":"{}".format(self.eval_env.env.task.number_tasks),
#             "number of networks":"{}".format(self.eval_env.env.reward.num_networks),
#             "mean subgoals finished":"{}".format(str(meansgoals)),
#             "mean subgoal reward":"{}".format(str(meansr)),
#             "mean subgoal steps":"{}".format(str(meansrs)),
#         }
        
#         for k, v in results.items():
#             print (k, ':', v)

        

#         #print ("mean subtasks finished:{}".format(meansgoals))

#         #if not self.is_tb_set:
#         #    with self.model.graph.as_default():
#         #        tf.summary.scalar('value_target', tf.reduce_mean(self.model.value_target))
#         #        self.model.summary = tf.summary.merge_all()
#         #    self.is_tb_set = True
#         summary = tf.Summary(value=[tf.Summary.Value(tag='Evaluation/1.Episode_success', simple_value=(success_episodes_num/n_eval_episodes*100))])
#         self.locals['writer'].add_summary(summary, self.num_timesteps)
#         summary = tf.Summary(value=[tf.Summary.Value(tag='Evaluation/3.Mean_distance_error', simple_value=(distance_error_sum / n_eval_episodes))])
#         self.locals['writer'].add_summary(summary, self.num_timesteps)
#         summary = tf.Summary(value=[tf.Summary.Value(tag='Evaluation/4.Mean_step_num', simple_value=(steps_sum // n_eval_episodes))])
#         self.locals['writer'].add_summary(summary, self.num_timesteps)
#         summary = tf.Summary(value=[tf.Summary.Value(tag='Evaluation/2.Mean_reward', simple_value=np.mean(episode_rewards))])
#         self.locals['writer'].add_summary(summary, self.num_timesteps)
#         summary = tf.Summary(value=[tf.Summary.Value(tag='Evaluation/2.Mean_sgoals', simple_value=meansgoals)])
#         self.locals['writer'].add_summary(summary, self.num_timesteps)
        
#         for i in range (self.eval_env.env.task.number_tasks):
#             #print("Task: {}".format(i)) 
#             for j, (k,l) in enumerate(zip(meansr,meansrs)):  
#                 m = np.count_nonzero(srsu[:,j])/n_eval_episodes*100
#                 #print("Reward {}: {} , steps: {} , Success: {}".format(j, k, l, m )) 
#                 summary = tf.Summary(value=[tf.Summary.Value(tag='Task{}/Subgoal{}/Reward'.format(i,j),
#                                                                               simple_value=k)])
#                 self.locals['writer'].add_summary(summary, self.num_timesteps)
#                 summary = tf.Summary(value=[tf.Summary.Value(tag='Task{}/Subgoal{}/Steps'.format(i,j),
#                                                                               simple_value=l)])
#                 self.locals['writer'].add_summary(summary, self.num_timesteps)
#                 summary = tf.Summary(value=[tf.Summary.Value(tag='Task{}/Subgoal{}/Success'.format(i,j),
#                                                                               simple_value=m)])
#                 self.locals['writer'].add_summary(summary, self.num_timesteps)

#         self.eval_env.reset()
#         print ("Evaluation finished successfully")
#         return results


#     def _init_callback(self):
#         # Does not work in some corner cases, where the wrapper is not the same
#         if not type(self.training_env) is type(self.eval_env):
#             warnings.warn("Training and eval env are not of the same type"
#                           "{} != {}".format(self.training_env, self.eval_env))

#         # Create folders if needed
#         if self.best_model_save_path is not None:
#             os.makedirs(self.best_model_save_path, exist_ok=True)
#         if self.log_path is not None:
#             os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

#     def _on_step(self) -> bool:
#         if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
#             # Sync training and eval env if there is VecNormalize
#             sync_envs_normalization(self.training_env, self.eval_env)

#             results = self.evaluate_policy(self.model,
#                                       n_eval_episodes=self.n_eval_episodes,
#                                       deterministic=self.deterministic)

#             if self.log_path is not None:
#                 self.evaluations_results["evaluation_after_{}_steps".format(self.n_calls)] = results
#                 #print(self.n_calls)
#                 #print(self.num_timesteps)
#                 #print (self.evaluations_results)
#                 filename = "evaluation_results.json"
#                 with open(os.path.join(self.log_path, filename), 'w') as f:
#                     json.dump(self.evaluations_results, f, indent=4)
#                 print("Evaluation stored after {} calls.".format(self.n_calls))    
#         return True


# class SaveOnBestTrainingRewardCallback(BaseCallback):
#     """
#     Callback for saving a model (the check is done every ``check_freq`` steps)
#     based on the training reward (in practice, we recommend using ``EvalCallback``).

#     :param check_freq: (int)
#     :param args.logdir: (str) Path to the folder where the model will be saved.
#       It must contains the file created by the ``Monitor`` wrapper.
#     :param verbose: (int)
#     :args.engine (str) Name of our current simulation engine we are using
#     :param env: (gym environment)
#     :stats_every (int) How often to create new datapoints for mujoco graph in episodes
#     :save_success_graph_every_steps (int) How often to save graph plotting
#     successfull episodes (mujoco)
#     :save_model_every_steps (int) How often in steps to save our model
#     :success_graph_mean_past_episodes (int) How many past episodes will be
#     taken into account when calculating average success rate (mujoco)
#     """

#     def __init__(self, check_freq: int, logdir: str, verbose=1,
#                  engine="pybullet",
#                  env="None",
#                  stats_every=50,
#                  save_success_graph_every_steps=40_000,
#                  save_model_every_steps=500_000,
#                  success_graph_mean_past_episodes=30,
#                  multiprocessing=0):
#         super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
#         self.check_freq = check_freq
#         self.logdir = logdir
#         self.save_path = os.path.join(logdir, 'best_model')
#         self.best_average_reward = -np.inf
#         self.engine = engine
#         self.save_model_every_steps = save_model_every_steps
#         # MUJOCO PART FOR NOW
#         self.periodical_save_path = os.path.join(logdir,'steps_')
#         self.env = env  # Our access to running environment
#         self.STATS_EVERY = stats_every
#         self.save_success_graph_every_steps = save_success_graph_every_steps
#         self.success_graph_mean_past_episodes = success_graph_mean_past_episodes
#         self.multiprocessing = multiprocessing

#     def _on_step(self) -> bool:
#         # DOESNT WORK WITH MULTIPROCESSING
#         if self.n_calls % self.save_model_every_steps == 0:
#             print("Saving model to {}".format(self.periodical_save_path))
#             self.model.save(self.periodical_save_path + f"{self.n_calls}")
#         if self.n_calls % self.check_freq == 0:
#             # Retrieve training reward
#             x, y = ts2xy(load_results(self.logdir), 'timesteps')

#             episode = len(y)  # Current episode
#             if episode:
#                 # Average training reward over the last 'self.STATS_EVERY' episodes
#                 average_reward = np.mean(y[-self.STATS_EVERY:])
#                 # Save the new best model (with best average reward)
#                 if average_reward > self.best_average_reward:
#                     self.best_average_reward = average_reward
#                     #if self.verbose > 0:
#                         #print("Saving new best model to {}".format(self.save_path))
#                     self.model.save(self.save_path)

#                 if self.engine == "mujoco":  # Mujoco has additional prints
#                     # Temporal workaround multiprocessing
#                     if not self.multiprocessing and self.verbose > 0:
#                         # Current success rate over the last 'self.STATS_EVERY' episodes
#                         current_success_rate = np.mean(self.env.successfull_failed_episodes[-self.STATS_EVERY:])
#                         print(f"Best average reward: {self.best_average_reward:.2f} \
#                               - Last average reward: {average_reward:.2f} \
#                               and current success rate: {current_success_rate:.2f} \
#                               over the last {self.STATS_EVERY} episodes \
#                               - Current touch threshold: {self.env.sensor_goal_threshold:.2f} - \
#                               Current episode: {episode}")

#         if self.engine == "mujoco":
#             # Save graph of success rate every 'self.save_success_graph_every_steps'.
#             if self.n_calls % self.save_success_graph_every_steps == 0 and not self.multiprocessing:
#                 # Save graph
#                 generate_and_save_mean_graph_from_1_or_2arrays(
#                     data_array_1=self.env.successfull_failed_episodes,
#                     data_array_2=None,
#                     save_dir=self.logdir,
#                     average_x_axis_span=self.success_graph_mean_past_episodes ,
#                     axis_x_name="episodes",
#                     axis_y_names=["success_rate"])
#         return True


# class ProgressBarCallback(BaseCallback):
#     """
#     :param pbar: (tqdm.pbar) Progress bar object
#     """

#     def __init__(self, pbar):
#         super(ProgressBarCallback, self).__init__()
#         self._pbar = pbar

#     def _on_step(self):
#         # Update the progress bar:
#         self._pbar.n = self.num_timesteps
#         self._pbar.update(0)


# # This callback uses the 'with' block, allowing for correct initialisation and destruction
# class ProgressBarManager(object):
#     def __init__(self, total_timesteps):  # Init object with total timesteps
#         self.pbar = None
#         self.total_timesteps = total_timesteps

#     def __enter__(self):  # Create the progress bar and callback, return the callback
#         self.pbar = tqdm(total=self.total_timesteps)

#         return ProgressBarCallback(self.pbar)

#     def __exit__(self, exc_type, exc_val, exc_tb):  # Close the callback
#         self.pbar.n = self.total_timesteps
#         self.pbar.update(0)
#         self.pbar.close()


# class PlottingCallback(BaseCallback):
#     """
#     Callback for plotting the performance in realtime.

#     :param verbose: (int)
#     """

#     def __init__(self, logdir: str, verbose=1):
#         super(PlottingCallback, self).__init__(verbose)
#         self._plot = None
#         self.logdir = logdir

#     def _on_step(self) -> bool:
#         # get the monitor's data
#         x, y = ts2xy(load_results(self.logdir), 'timesteps')
#         if self._plot is None:  # Make the plot
#             plt.ion()
#             fig = plt.figure(figsize=(6,3))
#             ax = fig.add_subplot(111)
#             line, = ax.plot(x, y)
#             self._plot = (line, ax, fig)
#             plt.title('Realtime reward')
#             plt.show()
#         else:  # Update and rescale the plot
#             self._plot[0].set_data(x, y)
#             self._plot[-2].relim()
#             self._plot[-2].set_xlim([self.locals["total_timesteps"] * -0.02,
#                                     self.locals["total_timesteps"] * 1.02])
#             self._plot[-2].autoscale_view(True, True, True)
#             self._plot[-1].canvas.draw()


# # This function is imported and called from another file. We can plot data along
# # 2 axis. This function plots 1 or 2 types of data on y axis
# # x-axis is usually 'episodes' but it represents the total amount of datapoints in our array,
# # y-axis is usually 'success_rate' and/or 'XXX' and represents mean value of our data on some index
# # in our array over some amount of past values ->  'average_x_axis_span'. We will calculate
# # average from the last 'average_x_axis_span' and plot it to y-axis
# def generate_and_save_mean_graph_from_1_or_2arrays(data_array_1=[], data_array_2= None,
#                                                    save_dir="", average_x_axis_span=20,
#                                                    axis_x_name="episodes", axis_y_names=["success_rate"]):
#     # Initialize storage array

#     data_to_plot = {axis_x_name: [], axis_y_names[0]: []}
#     if data_array_2 is not None:
#         data_to_plot = {axis_x_name: [], axis_y_names[0]: [], axis_y_names[1]: []}

#     x_axis_length = len(data_array_1)  # Total episode count takenn from array 1
#     # x_axis_counter <==> f.e. episodes
#     for x_axis_counter in range(x_axis_length):  # Iterate through every 'episode'/datapoint

#         # Add data to plotter
#         if (x_axis_counter % average_x_axis_span) == 0:
#             # Average data over last 'average_episodes_span'
#             mean_y_value_1 = np.mean(data_array_1[x_axis_counter : (x_axis_counter+average_x_axis_span)])

#             if data_array_2 is not None:
#                 mean_y_value_2 = np.mean(data_array_2[x_axis_counter : (x_axis_counter+average_x_axis_span)])
#             # Append the data
#             data_to_plot[axis_x_name].append(x_axis_counter)
#             data_to_plot[axis_y_names[0]].append(mean_y_value_1)

#             if data_array_2 is not None:
#                 data_to_plot[axis_y_names[1]].append(mean_y_value_2)

#     # Save data as an image
#     print(f"len: data_to_plot[axis_y_names[0]: {len(data_to_plot[axis_y_names[0]])}")
#     print(f"data_to_plot[axis_y_names[0]: {data_to_plot[axis_y_names[0]]}")
#     plt.plot(data_to_plot[axis_x_name], data_to_plot[axis_y_names[0]], label=axis_y_names[0])

#     if data_array_2 is not None:
#         plt.plot(data_to_plot[axis_x_name], data_to_plot[axis_y_names[1]], label=axis_y_names[1])

#     plt.legend(loc=2)

#     if save_dir == "":
#         print("Save directory for image not specified! -> 'generate_and_save_mean_graph_from_array()'")
#     else:
#         if data_array_2 is not None:
#             image_path = save_dir + f"/graph_{axis_y_names[0]}_{axis_y_names[1]}_over_{axis_x_name}" + '.png'
#         else:
#             image_path = save_dir + f"/graph_{axis_y_names[0]}_over_{axis_x_name}" + '.png'

#         plt.savefig(image_path)

#         if data_array_2 is not None:
#             print(f"\n Graph {axis_y_names[0]}_{axis_y_names[1]}_over_{axis_x_name} succesfully saved to {image_path}")
#         else:
#             print(f"\n Graph {axis_y_names[0]}_over_{axis_x_name} succesfully saved to {image_path}")

#         plt.close()
#     return

# class SaveOnTopRewardCallback(BaseCallback):
#     """
#     Callback for saving a model (the check is done every ``check_freq`` steps)
#     based on the training reward (in practice, we recommend using ``EvalCallback``).

#     :param check_freq: (int)
#     :param args.logdir: (str) Path to the folder where the model will be saved.
#       It must contains the file created by the ``Monitor`` wrapper.
#     :param verbose: (int)
#     """

#     def __init__(self, check_freq: int, logdir: str, verbose=1, models_num=2):
#         super(SaveOnTopRewardCallback, self).__init__(verbose)
#         self.check_freq = check_freq
#         self.logdir = logdir
#         self.save_path = os.path.join(logdir, 'highest_reward')
#         self.top_rewards = [-np.inf]*models_num

#     def _on_rollout_end(self) -> bool:
#         if self.n_calls % self.check_freq == 0:
#             x, y = ts2xy(load_results(self.logdir), 'timesteps')

#             episode = len(y)  # Current episode
#             if episode:
#                 i = 0
#                 for submodel in self.model.models:
#                     actual_reward = submodel.episode_reward
#                     if actual_reward > self.top_rewards[i]:
#                         self.top_rewards[i] = actual_reward
#                         if self.verbose > 0:
#                             print("Saving new most rewarded model to {}".format(self.save_path))
#                         submodel.save(self.save_path, i)

#                     i += 1
#         return True
