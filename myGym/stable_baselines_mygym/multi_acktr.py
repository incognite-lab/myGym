import time
import warnings

import tensorflow as tf
from gym.spaces import Box, Discrete

from stable_baselines import logger
from stable_baselines.a2c.a2c import A2CRunner
from myGym.stable_baselines_mygym.multi_ppo2 import Runner as PPO2Runner
from myGym.utils.callbacks import SaveOnTopRewardCallback
from stable_baselines.common.tf_util import mse, total_episode_reward_logger
from stable_baselines.acktr import kfac
from stable_baselines.common.schedules import Scheduler
from stable_baselines.common import explained_variance, ActorCriticRLModel, tf_util, SetVerbosity, TensorboardWriter
from stable_baselines.common.policies import ActorCriticPolicy, RecurrentActorCriticPolicy
from stable_baselines.common.math_util import safe_mean

import commentjson
import numpy as np
import gym
import os
from collections import deque

class MultiACKTR(ActorCriticRLModel):
    """
    The ACKTR (Actor Critic using Kronecker-Factored Trust Region) model class, https://arxiv.org/abs/1708.05144

    :param policy: (ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) Discount factor
    :param nprocs: (int) The number of threads for TensorFlow operations

        .. deprecated:: 2.9.0
            Use `n_cpu_tf_sess` instead.

    :param n_steps: (int) The number of steps to run for each environment
    :param ent_coef: (float) The weight for the entropy loss
    :param vf_coef: (float) The weight for the loss on the value function
    :param vf_fisher_coef: (float) The weight for the fisher loss on the value function
    :param learning_rate: (float) The initial learning rate for the RMS prop optimizer
    :param max_grad_norm: (float) The clipping value for the maximum gradient
    :param kfac_clip: (float) gradient clipping for Kullback-Leibler
    :param lr_schedule: (str) The type of scheduler for the learning rate update ('linear', 'constant',
                        'double_linear_con', 'middle_drop' or 'double_middle_drop')
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param async_eigen_decomp: (bool) Use async eigen decomposition
    :param kfac_update: (int) update kfac after kfac_update steps
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param gae_lambda: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        If None (default), then the classic advantage will be used instead of GAE
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        WARNING: this logging can take a lot of space quickly
    :param seed: (int) Seed for the pseudo-random generators (python, numpy, tensorflow).
        If None (default), use random seed. Note that if you want completely deterministic
        results, you must set `n_cpu_tf_sess` to 1.
    :param n_cpu_tf_sess: (int) The number of threads for TensorFlow operations
        If None, the number of cpu of the current machine will be used.
    """

    def __init__(self, policy, env, gamma=0.99, nprocs=None, n_steps=20, n_models=None, ent_coef=0.01, vf_coef=0.25, vf_fisher_coef=1.0,
                 learning_rate=0.25, max_grad_norm=0.5, kfac_clip=0.001, lr_schedule='linear', verbose=0,
                 tensorboard_log=None, _init_setup_model=True, async_eigen_decomp=False, kfac_update=1,
                 gae_lambda=0.95, policy_kwargs=None, full_tensorboard_log=False, seed=None, n_cpu_tf_sess=1):

        if nprocs is not None:
            warnings.warn("nprocs will be removed in a future version (v3.x.x) "
                          "use n_cpu_tf_sess instead", DeprecationWarning)
            n_cpu_tf_sess = nprocs

        self.n_steps = n_steps
        self.models_num = n_models
        self.gamma = gamma
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.vf_fisher_coef = vf_fisher_coef
        self.kfac_clip = kfac_clip
        self.max_grad_norm = max_grad_norm
        self.learning_rate = learning_rate
        self.lr_schedule = lr_schedule

        self.tensorboard_log = tensorboard_log
        self.async_eigen_decomp = async_eigen_decomp
        self.full_tensorboard_log = full_tensorboard_log
        self.kfac_update = kfac_update
        self.gae_lambda = gae_lambda

        self.actions_ph = None
        self.advs_ph = None
        self.rewards_ph = None
        self.learning_rate_ph = None
        self.step_model = None
        self.train_model = None
        self.entropy = None
        self.pg_loss = None
        self.vf_loss = None
        self.pg_fisher = None
        self.vf_fisher = None
        self.joint_fisher = None
        self.grads_check = None
        self.optim = None
        self.train_op = None
        self.q_runner = None
        self.learning_rate_schedule = None
        self.proba_step = None
        self.value = None
        self.initial_state = None
        self.n_batch = None
        self.summary = None
        self.trained = False
        self.continuous_actions = False

        super(MultiACKTR, self).__init__(policy=policy, env=env, verbose=verbose, requires_vec_env=True,
                                    _init_setup_model=_init_setup_model, policy_kwargs=policy_kwargs,
                                    seed=seed, n_cpu_tf_sess=n_cpu_tf_sess)

        if _init_setup_model:
            self.setup_model()

    def _make_runner(self):
        if self.gae_lambda is not None:
            return PPO2Runner(
                env=self.env, model=self, models=self.models, n_steps=self.n_steps, gamma=self.gamma, lam=self.gae_lambda)
        else:
            # TODO: modify A2CRunner
            return A2CRunner(
                self.env, self, n_steps=self.n_steps, gamma=self.gamma)

    def _get_pretrain_placeholders(self):
        policy = self.train_model
        if isinstance(self.action_space, Discrete):
            return policy.obs_ph, self.actions_ph, policy.policy
        return policy.obs_ph, self.actions_ph, policy.deterministic_action

    def setup_model(self):
        self.models = []
        for i in range(self.models_num):
            self.models.append(SubModel(self, i))
        
        # with SetVerbosity(self.verbose):

        #     assert issubclass(self.policy, ActorCriticPolicy), "Error: the input policy for the ACKTR model must be " \
        #                                                        "an instance of common.policies.ActorCriticPolicy."

        #     # Enable continuous actions tricks (normalized advantage)
        #     self.continuous_actions = isinstance(self.action_space, Box)

        #     self.graph = tf.Graph()
        #     with self.graph.as_default():
        #         self.set_random_seed(self.seed)
        #         self.sess = tf_util.make_session(num_cpu=self.n_cpu_tf_sess, graph=self.graph)

        #         n_batch_step = None
        #         n_batch_train = None
        #         if issubclass(self.policy, RecurrentActorCriticPolicy):
        #             n_batch_step = self.n_envs
        #             n_batch_train = self.n_envs * self.n_steps

        #         step_model = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs,
        #                                  1, n_batch_step, reuse=False, **self.policy_kwargs)

        #         self.params = params = tf_util.get_trainable_vars("model")

        #         with tf.variable_scope("train_model", reuse=True,
        #                                custom_getter=tf_util.outer_scope_getter("train_model")):
        #             train_model = self.policy(self.sess, self.observation_space, self.action_space,
        #                                       self.n_envs, self.n_steps, n_batch_train,
        #                                       reuse=True, **self.policy_kwargs)

        #         with tf.variable_scope("loss", reuse=False, custom_getter=tf_util.outer_scope_getter("loss")):
        #             self.advs_ph = advs_ph = tf.placeholder(tf.float32, [None])
        #             self.rewards_ph = rewards_ph = tf.placeholder(tf.float32, [None])
        #             self.learning_rate_ph = learning_rate_ph = tf.placeholder(tf.float32, [])
        #             self.actions_ph = train_model.pdtype.sample_placeholder([None])

        #             neg_log_prob = train_model.proba_distribution.neglogp(self.actions_ph)

        #             # training loss
        #             pg_loss = tf.reduce_mean(advs_ph * neg_log_prob)
        #             self.entropy = entropy = tf.reduce_mean(train_model.proba_distribution.entropy())
        #             self.pg_loss = pg_loss = pg_loss - self.ent_coef * entropy
        #             self.vf_loss = vf_loss = mse(tf.squeeze(train_model.value_fn), rewards_ph)
        #             train_loss = pg_loss + self.vf_coef * vf_loss

        #             # Fisher loss construction
        #             self.pg_fisher = pg_fisher_loss = -tf.reduce_mean(neg_log_prob)
        #             sample_net = train_model.value_fn + tf.random_normal(tf.shape(train_model.value_fn))
        #             self.vf_fisher = vf_fisher_loss = - self.vf_fisher_coef * tf.reduce_mean(
        #                 tf.pow(train_model.value_fn - tf.stop_gradient(sample_net), 2))
        #             self.joint_fisher = pg_fisher_loss + vf_fisher_loss

        #             tf.summary.scalar('entropy_loss', self.entropy)
        #             tf.summary.scalar('policy_gradient_loss', pg_loss)
        #             tf.summary.scalar('policy_gradient_fisher_loss', pg_fisher_loss)
        #             tf.summary.scalar('value_function_loss', self.vf_loss)
        #             tf.summary.scalar('value_function_fisher_loss', vf_fisher_loss)
        #             tf.summary.scalar('loss', train_loss)

        #             self.grads_check = tf.gradients(train_loss, params)

        #         with tf.variable_scope("input_info", reuse=False):
        #             tf.summary.scalar('discounted_rewards', tf.reduce_mean(self.rewards_ph))
        #             tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate_ph))
        #             tf.summary.scalar('advantage', tf.reduce_mean(self.advs_ph))

        #             if self.full_tensorboard_log:
        #                 tf.summary.histogram('discounted_rewards', self.rewards_ph)
        #                 tf.summary.histogram('learning_rate', self.learning_rate_ph)
        #                 tf.summary.histogram('advantage', self.advs_ph)
        #                 if tf_util.is_image(self.observation_space):
        #                     tf.summary.image('observation', train_model.obs_ph)
        #                 else:
        #                     tf.summary.histogram('observation', train_model.obs_ph)

        #         with tf.variable_scope("kfac", reuse=False, custom_getter=tf_util.outer_scope_getter("kfac")):
        #             with tf.device('/gpu:0'):
        #                 self.optim = optim = kfac.KfacOptimizer(learning_rate=learning_rate_ph, clip_kl=self.kfac_clip,
        #                                                         momentum=0.9, kfac_update=self.kfac_update,
        #                                                         epsilon=0.01, stats_decay=0.99,
        #                                                         async_eigen_decomp=self.async_eigen_decomp,
        #                                                         cold_iter=10,
        #                                                         max_grad_norm=self.max_grad_norm, verbose=self.verbose)

        #                 optim.compute_and_apply_stats(self.joint_fisher, var_list=params)

        #         self.train_model = train_model
        #         self.step_model = step_model
        #         self.step = step_model.step
        #         self.proba_step = step_model.proba_step
        #         self.value = step_model.value
        #         self.initial_state = step_model.initial_state
        #         tf.global_variables_initializer().run(session=self.sess)

        #         self.summary = tf.summary.merge_all()


    def _train_step(self, obs, states, rewards, masks, actions, values, update, model, writer):
        """
        applies a training step to the model

        :param obs: ([float]) The input observations
        :param states: ([float]) The states (used for recurrent policies)
        :param rewards: ([float]) The rewards from the environment
        :param masks: ([bool]) Whether or not the episode is over (used for recurrent policies)
        :param actions: ([float]) The actions taken
        :param values: ([float]) The logits values
        :param update: (int) the current step iteration
        :param writer: (TensorFlow Summary.writer) the writer for tensorboard
        :return: (float, float, float) policy loss, value loss, policy entropy
        """
        advs = rewards - values
        # Normalize advantage (used in the original continuous version)
        if model.continuous_actions:
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        current_lr = None

        assert len(obs) > 0, "Error: the observation input array cannot be empty"

        # Note: in the original continuous version,
        # the stepsize was automatically tuned computing the kl div
        # and comparing it to the desired one
        for _ in range(len(obs)):
            current_lr = self.learning_rate_schedule.value()

        td_map = {
            model.train_model.obs_ph: obs,
            model.actions_ph: actions,
            model.advs_ph: advs,
            model.rewards_ph: rewards,
            model.learning_rate_ph: current_lr
        }

        if states is not None:
            td_map[model.train_model.states_ph] = states
            td_map[model.train_model.dones_ph] = masks

        if writer is not None:
            # run loss backprop with summary, but once every 10 runs save the metadata (memory, compute time, ...)
            if self.full_tensorboard_log and (1 + update) % 10 == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, policy_loss, value_loss, policy_entropy, _ = model.sess.run(
                    [model.summary, model.pg_loss, model.vf_loss, model.entropy, model.train_op],
                    td_map, options=run_options, run_metadata=run_metadata)
                writer.add_run_metadata(run_metadata, 'step%d' % (update * (model.n_batch + 1)))
            else:
                summary, policy_loss, value_loss, policy_entropy, _ = model.sess.run(
                    [model.summary, model.pg_loss, model.vf_loss, model.entropy, model.train_op], td_map)
            writer.add_summary(summary, update * (model.n_batch + 1))
        else:
            policy_loss, value_loss, policy_entropy, _ = model.sess.run(
                [model.pg_loss, model.vf_loss, model.entropy, model.train_op], td_map)

        return policy_loss, value_loss, policy_entropy

    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="DUAL_ACKTR",
              reset_num_timesteps=True):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        top_callback = SaveOnTopRewardCallback(check_freq=self.n_steps, logdir=self.tensorboard_log, models_num=self.models_num)
        callback.append(top_callback)
        callback     = self._init_callback(callback)

        with SetVerbosity(self.verbose), TensorboardWriter(self.models[0].graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            for model in self.models:
                model._setup_learn(self)
            self.n_batch = self.n_envs * self.n_steps

            self.learning_rate_schedule = Scheduler(initial_value=self.learning_rate, n_values=total_timesteps,
                                                    schedule=self.lr_schedule)

            # TODO: make this multi
            # FIFO queue of the q_runner thread is closed at the end of the learn function.
            # As a result, it needs to be redefinied at every call
            for model in self.models:
                with model.graph.as_default():
                    with tf.variable_scope("kfac_apply", reuse=self.trained,
                                        custom_getter=tf_util.outer_scope_getter("kfac_apply")):
                        # Some of the variables are not in a scope when they are create
                        # so we make a note of any previously uninitialized variables
                        tf_vars = tf.global_variables()
                        is_uninitialized = model.sess.run([tf.is_variable_initialized(var) for var in tf_vars])
                        old_uninitialized_vars = [v for (v, f) in zip(tf_vars, is_uninitialized) if not f]

                        model.train_op, model.q_runner = model.optim.apply_gradients(list(zip(model.grads_check, model.params)))

                        # then we check for new uninitialized variables and initialize them
                        tf_vars = tf.global_variables()
                        is_uninitialized = model.sess.run([tf.is_variable_initialized(var) for var in tf_vars])
                        new_uninitialized_vars = [v for (v, f) in zip(tf_vars, is_uninitialized)
                                                if not f and v not in old_uninitialized_vars]

                        if len(new_uninitialized_vars) != 0:
                            model.sess.run(tf.variables_initializer(new_uninitialized_vars))

            self.trained = True

            t_start = time.time()
            coord = tf.train.Coordinator()
            if self.q_runner is not None:
                enqueue_threads = self.q_runner.create_threads(self.sess, coord=coord, start=True)
            else:
                enqueue_threads = []

            callback.on_training_start(locals(), globals())

            for update in range(1, total_timesteps // self.n_batch + 1):

                callback.on_rollout_start()

                # pytype:disable=bad-unpacking
                # true_reward is the reward without discount
                if isinstance(self.runner, PPO2Runner):
                    # We are using GAE
                    rollouts = self.runner.run(callback)
                    #obs, returns, masks, actions, values, _, states, ep_infos, true_reward = rollout
                else:
                    rollout = self.runner.run(callback)
                    obs, states, returns, masks, actions, values, ep_infos, true_reward = rollout
                # pytype:enable=bad-unpacking
                callback.update_locals(locals())
                callback.on_rollout_end()

                # Early stopping due to the callback
                if not self.runner.continue_training:
                    break

                # Unpack

                i = 0
                steps_used = rollouts[-1]
                for rollout in rollouts[0]:
                    obs, returns, masks, actions, values, neglogpacs, states, ep_infos, true_reward, success_stages = rollout
                    model = self.models[i]
                    # calc = len(true_reward)
                    # model.n_batch = calc

                    if steps_used[i] == 0:
                    #if model.n_batch == 0:
                        b = 0
                    else:

                        self.ep_info_buf.extend(ep_infos)
                        policy_loss, value_loss, policy_entropy = self._train_step(obs, states, returns, masks, actions, values,
                                                                                self.num_timesteps // (self.n_batch + 1),
                                                                                self.models[i], writer)
                        n_seconds = time.time() - t_start
                        fps = int((update * self.n_batch) / n_seconds)

                        if writer is not None:
                            n_steps = model.n_batch
                            try:
                                print("true reward:{}".format(true_reward.shape))
                                total_episode_reward_logger(self.episode_reward,
                                                        true_reward.reshape((self.n_envs, n_steps)),
                                                        masks.reshape((self.n_envs, n_steps)),
                                                        writer, self.num_timesteps)

                            except:
                               print("Failed to log episode reward of shape {}".format(true_reward.shape))
                            summary = tf.Summary(value=[tf.Summary.Value(tag='episode_reward/Successful stages',
                                                                         simple_value=success_stages)])
                            writer.add_summary(summary, self.num_timesteps)
                            #@TODO plot in one graph:
                            for j, val in enumerate(steps_used):
                                summary = tf.Summary(value=[tf.Summary.Value(tag='episode_reward/Used steps net {}'.format(j),
                                                                              simple_value=val)])
                                writer.add_summary(summary, self.num_timesteps)

                        if self.verbose >= 1 and (update % log_interval == 0 or update == 1):
                            explained_var = explained_variance(values, returns)
                            logger.record_tabular("nupdates", update)
                            logger.record_tabular("total_timesteps", self.num_timesteps)
                            logger.record_tabular("fps", fps)
                            logger.record_tabular("Steps", steps_used)
                            logger.record_tabular("policy_entropy", float(policy_entropy))
                            logger.record_tabular("policy_loss", float(policy_loss))
                            logger.record_tabular("value_loss", float(value_loss))
                            logger.record_tabular("explained_variance", float(explained_var))
                            if len(self.ep_info_buf) > 0 and len(self.ep_info_buf[0]) > 0:
                                logger.logkv('ep_reward_mean', safe_mean([ep_info['r'] for ep_info in self.ep_info_buf]))
                                logger.logkv('ep_len_mean', safe_mean([ep_info['l'] for ep_info in self.ep_info_buf]))
                            logger.dump_tabular()
                    
                    i += 1

            coord.request_stop()
            coord.join(enqueue_threads)

        callback.on_training_end()
        return self

    def save(self, parent_path, cloudpickle=False):
        data = {
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "n_steps": self.n_steps,
            "vf_coef": self.vf_coef,
            "ent_coef": self.ent_coef,
            "vf_fisher_coef": self.vf_fisher_coef,
            "max_grad_norm": self.max_grad_norm,
            "learning_rate": self.learning_rate,
            "kfac_clip": self.kfac_clip,
            "lr_schedule": self.lr_schedule,
            "verbose": self.verbose,
            "policy": self.policy,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "n_envs": self.n_envs,
            "n_cpu_tf_sess": self.n_cpu_tf_sess,
            "seed": self.seed,
            "kfac_update": self.kfac_update,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs
        }

        parent_path = parent_path.split("/")
        start       = parent_path[:-1]
        end         = parent_path[-1]
        start       = "/".join(start)

        for i in range(self.models_num):
            submodel_path = start + "/submodel_" + str(i) + "/" + end
            params_to_save = self.models[i].get_parameters()
            self._save_to_file(submodel_path, data=data, params=params_to_save, cloudpickle=cloudpickle)

    def submodel_path(self, i):
        return self.tensorboard_log + "/submodel_" + str(i)

    @classmethod
    def load(cls, load_path, env=None, custom_objects=None, **kwargs):
        """
        Load the model from file

        :param load_path: (str or file-like) the saved parameter location
        :param env: (Gym Environment) the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model)
        :param custom_objects: (dict) Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            `keras.models.load_model`. Useful when you have an object in
            file that can not be deserialized.
        :param kwargs: extra arguments to change the model when loading
        """
        load_path = load_path.split("/")
        load_path = load_path[:-1]
        path = "/".join(load_path)


        with open(path + "/train.json", "r") as f:
            json = commentjson.load(f)

        models = json["num_networks"]
        load = [] # data, params
        for i in range(models):
            # load_path = "/home/jonas/myGym/myGym/trained_models/dual/poke_table_kuka_joints_gt_dual_13"
            load_path = path + "/submodel_" + str(i) + "/best_model.zip"
            load.append(cls._load_from_file(load_path, custom_objects=custom_objects))

        data = load[0][0]

        if 'policy_kwargs' in kwargs and kwargs['policy_kwargs'] != data['policy_kwargs']:
            raise ValueError("The specified policy kwargs do not equal the stored policy kwargs. "
                             "Stored kwargs: {}, specified kwargs: {}".format(data['policy_kwargs'],
                                                                              kwargs['policy_kwargs']))

        model = cls(policy=data["policy"], env=None, _init_setup_model=False)  # pytype: disable=not-instantiable
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        if env:
            model.env = env
        # model.set_env(env)
        model.tensorboard_log = "/home/jonas/myGym/myGym/trained_models/dual/poke_table_kuka_joints_gt_dual_13"
        model.models_num = models
        model.setup_model()

        i = 0
        for submodel in model.models:
            submodel.load_parameters(load[i][1])
            i += 1

        return model

    def predict(self, observation, state=None, mask=None, deterministic=False):
        if state is None:
            state = self.initial_state
        if mask is None:
            mask = [False for _ in range(self.n_envs)]
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        model = self.models[self.approved(observation)]
        actions, _, states, _ = model.step(observation, state, mask, deterministic=deterministic)

        clipped_actions = actions
        # Clip the actions to avoid out of bound error
        if isinstance(self.action_space, gym.spaces.Box):
            clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

        if not vectorized_env:
            if state is not None:
                raise ValueError("Error: The environment must be vectorized when using recurrent policies.")
            clipped_actions = clipped_actions[0]

        return clipped_actions, states

    def approved(self, observation):
        # based on obs, decide which model should be used
        if hasattr(self.env, 'envs'):
            submodel_id = self.env.envs[0].env.env.reward.network_switch_control(self.env.envs[0].env.env.observation["task_objects"])
        else:
            submodel_id = self.env.reward.network_switch_control(self.env.observation["task_objects"])
        return submodel_id


class SubModel(MultiACKTR):
    def __init__(self, parent, i):
        self.episode_reward  = 0
        self.model_num = i
        self._param_load_ops = None
        self.path = parent.tensorboard_log + "/submodel_" + str(i)
        try:
            os.makedirs(self.path)
        except:
            pass

        self.env = parent.env
        self.observation_space = parent.observation_space

        with SetVerbosity(parent.verbose):

            assert issubclass(parent.policy, ActorCriticPolicy), "Error: the input policy for the ACKTR model must be " \
                                                               "an instance of common.policies.ActorCriticPolicy."

            # Enable continuous actions tricks (normalized advantage)
            self.continuous_actions = isinstance(parent.action_space, Box)

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.set_random_seed(parent.seed)
                self.sess = tf_util.make_session(num_cpu=parent.n_cpu_tf_sess, graph=self.graph)

                n_batch_step = None
                n_batch_train = None
                if issubclass(parent.policy, RecurrentActorCriticPolicy):
                    n_batch_step = parent.n_envs
                    n_batch_train = parent.n_envs * parent.n_steps

                step_model = parent.policy(self.sess, self.observation_space, parent.action_space, parent.n_envs,
                                         1, n_batch_step, reuse=False, **parent.policy_kwargs)

                self.params = params = tf_util.get_trainable_vars("model")

                with tf.variable_scope("train_model", reuse=True,
                                       custom_getter=tf_util.outer_scope_getter("train_model")):
                    train_model = parent.policy(self.sess, self.observation_space, parent.action_space,
                                              parent.n_envs, parent.n_steps, n_batch_train,
                                              reuse=True, **parent.policy_kwargs)

                with tf.variable_scope("loss_network_{}".format(self.model_num), reuse=False, custom_getter=tf_util.outer_scope_getter("loss")):
                    self.advs_ph = advs_ph = tf.placeholder(tf.float32, [None])
                    self.rewards_ph = rewards_ph = tf.placeholder(tf.float32, [None])
                    self.learning_rate_ph = learning_rate_ph = tf.placeholder(tf.float32, [])
                    self.actions_ph = train_model.pdtype.sample_placeholder([None])

                    neg_log_prob = train_model.proba_distribution.neglogp(self.actions_ph)

                    # training loss
                    pg_loss = tf.reduce_mean(advs_ph * neg_log_prob)
                    self.entropy = entropy = tf.reduce_mean(train_model.proba_distribution.entropy())
                    self.pg_loss = pg_loss = pg_loss - parent.ent_coef * entropy
                    self.vf_loss = vf_loss = mse(tf.squeeze(train_model.value_fn), rewards_ph)
                    train_loss = pg_loss + parent.vf_coef * vf_loss

                    # Fisher loss construction
                    self.pg_fisher = pg_fisher_loss = -tf.reduce_mean(neg_log_prob)
                    sample_net = train_model.value_fn + tf.random_normal(tf.shape(train_model.value_fn))
                    self.vf_fisher = vf_fisher_loss = - parent.vf_fisher_coef * tf.reduce_mean(
                        tf.pow(train_model.value_fn - tf.stop_gradient(sample_net), 2))
                    self.joint_fisher = pg_fisher_loss + vf_fisher_loss

                    tf.summary.scalar('entropy_loss', self.entropy)
                    tf.summary.scalar('policy_gradient_loss', pg_loss)
                    tf.summary.scalar('policy_gradient_fisher_loss', pg_fisher_loss)
                    tf.summary.scalar('value_function_loss', self.vf_loss)
                    tf.summary.scalar('value_function_fisher_loss', vf_fisher_loss)
                    tf.summary.scalar('loss', train_loss)

                    self.grads_check = tf.gradients(train_loss, params)

                with tf.variable_scope("input_info", reuse=False):
                    tf.summary.scalar('discounted_rewards', tf.reduce_mean(self.rewards_ph))
                    tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate_ph))
                    tf.summary.scalar('advantage', tf.reduce_mean(self.advs_ph))

                    if parent.full_tensorboard_log:
                        tf.summary.histogram('discounted_rewards', self.rewards_ph)
                        tf.summary.histogram('learning_rate', self.learning_rate_ph)
                        tf.summary.histogram('advantage', self.advs_ph)
                        if tf_util.is_image(self.observation_space):
                            tf.summary.image('observation', train_model.obs_ph)
                        else:
                            tf.summary.histogram('observation', train_model.obs_ph)

                with tf.variable_scope("kfac", reuse=False, custom_getter=tf_util.outer_scope_getter("kfac")):
                    with tf.device('/gpu:0'):
                        self.optim = optim = kfac.KfacOptimizer(learning_rate=learning_rate_ph, clip_kl=parent.kfac_clip,
                                                                momentum=0.9, kfac_update=parent.kfac_update,
                                                                epsilon=0.01, stats_decay=0.99,
                                                                async_eigen_decomp=parent.async_eigen_decomp,
                                                                cold_iter=10,
                                                                max_grad_norm=parent.max_grad_norm, verbose=parent.verbose)

                        optim.compute_and_apply_stats(self.joint_fisher, var_list=params)

                self.train_model = train_model
                self.step_model = step_model
                self.step = step_model.step
                self.proba_step = step_model.proba_step
                self.value = step_model.value
                self.initial_state = step_model.initial_state
                tf.global_variables_initializer().run(session=self.sess)

                self.summary = tf.summary.merge_all()

                self.parent = parent

    def _setup_learn(self, parent): 
        """
        Check the environment.
        """
        if parent.env is None:
            raise ValueError("Error: cannot train the model without a valid environment, please set an environment with set_env(self, env) method.")
        if parent.episode_reward is None:
            parent.episode_reward = np.zeros((parent.n_envs,))
        if parent.ep_info_buf is None:
            parent.ep_info_buf = deque(maxlen=100)


    def save(self, parent_path, i, cloudpickle=False):
        # save only one model
        data = {
            "gamma": self.parent.gamma,
            "gae_lambda": self.parent.gae_lambda,
            "n_steps": self.parent.n_steps,
            "vf_coef": self.parent.vf_coef,
            "ent_coef": self.parent.ent_coef,
            "vf_fisher_coef": self.parent.vf_fisher_coef,
            "max_grad_norm": self.parent.max_grad_norm,
            "learning_rate": self.parent.learning_rate,
            "kfac_clip": self.parent.kfac_clip,
            "lr_schedule": self.parent.lr_schedule,
            "verbose": self.parent.verbose,
            "policy": self.parent.policy,
            "observation_space": self.parent.observation_space,
            "action_space": self.parent.action_space,
            "n_envs": self.parent.n_envs,
            "n_cpu_tf_sess": self.parent.n_cpu_tf_sess,
            "seed": self.parent.seed,
            "kfac_update": self.parent.kfac_update,
            "_vectorize_action": self.parent._vectorize_action,
            "policy_kwargs": self.parent.policy_kwargs
        }
        
        parent_path = parent_path.split("/")
        start       = parent_path[:-1]
        end         = parent_path[-1]
        start       = "/".join(start)

        submodel_path = start + "/submodel_" + str(i) + "/" + end
        params_to_save = self.get_parameters()
        self._save_to_file(submodel_path, data=data, params=params_to_save, cloudpickle=cloudpickle)
