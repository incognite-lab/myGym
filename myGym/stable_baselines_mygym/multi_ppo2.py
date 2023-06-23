from pickle import NONE
import time
import os
import gym
import numpy as np
from numpy.lib.function_base import append
from myGym.utils.callbacks import SaveOnTopRewardCallback
import tensorflow as tf
#tf.get_logger().setLevel('ERROR')

from collections import OrderedDict, deque

from stable_baselines import logger
from stable_baselines.common import explained_variance, ActorCriticRLModel, tf_util, SetVerbosity, TensorboardWriter
from stable_baselines.common.runners import AbstractEnvRunner
from stable_baselines.common.policies import ActorCriticPolicy, RecurrentActorCriticPolicy
from stable_baselines.common.schedules import get_schedule_fn
from stable_baselines.common.tf_util import total_episode_reward_logger
from stable_baselines.common.math_util import safe_mean
from stable_baselines.common.misc_util import set_global_seeds

class MultiPPO2(ActorCriticRLModel):
    """
    Proximal Policy Optimization algorithm (GPU version).
    Paper: https://arxiv.org/abs/1707.06347

    :param policy: (ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) Discount factor
    :param n_steps: (int) The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param ent_coef: (float) Entropy coefficient for the loss calculation
    :param learning_rate: (float or callable) The learning rate, it can be a function
    :param vf_coef: (float) Value function coefficient for the loss calculation
    :param max_grad_norm: (float) The maximum value for the gradient clipping
    :param lam: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param nminibatches: (int) Number of training minibatches per update. For recurrent policies,
        the number of environments run in parallel should be a multiple of nminibatches.
    :param noptepochs: (int) Number of epoch when optimizing the surrogate
    :param cliprange: (float or callable) Clipping parameter, it can be a function
    :param cliprange_vf: (float or callable) Clipping parameter for the value function, it can be a function.
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        then `cliprange` (that is used for the policy) will be used.
        IMPORTANT: this clipping depends on the reward scaling.
        To deactivate value function clipping (and recover the original PPO implementation),
        you have to pass a negative value (e.g. -1).
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        WARNING: this logging can take a lot of space quickly
    :param seed: (int) Seed for the pseudo-random generators (python, numpy, tensorflow).
        If None (default), use random seed. Note that if you want completely deterministic
        results, you must set `n_cpu_tf_sess` to 1.
    :param n_cpu_tf_sess: (int) The number of threads for TensorFlow operations
        If None, the number of cpu of the current machine will be used.
    """
    def __init__(self, policy, env, gamma=0.99, n_steps=128,n_models=None, ent_coef=0.01, learning_rate=2.5e-4, vf_coef=0.5,
                 max_grad_norm=0.5, lam=0.95, nminibatches=4, noptepochs=4, cliprange=0.2, cliprange_vf=None,
                 verbose=0, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None,
                 full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None):

        self.learning_rate          = learning_rate
        self.cliprange              = cliprange
        self.cliprange_vf           = cliprange_vf
        self.n_steps                = n_steps
        self.ent_coef               = ent_coef
        self.vf_coef                = vf_coef
        self.max_grad_norm          = max_grad_norm
        self.gamma                  = gamma
        self.lam                    = lam
        self.nminibatches           = nminibatches
        self.noptepochs             = noptepochs
        self.tensorboard_log        = tensorboard_log
        self.full_tensorboard_log   = full_tensorboard_log
        self.models_num             = n_models

        self.action_ph          = None
        self.advs_ph            = None
        self.rewards_ph         = None
        self.old_neglog_pac_ph  = None
        self.old_vpred_ph       = None
        self.learning_rate_ph   = None
        self.clip_range_ph      = None
        self.entropy            = None
        self.vf_loss            = None
        self.pg_loss            = None
        self.approxkl           = None
        self.clipfrac           = None
        self._train             = None
        self.loss_names         = None
        self.train_model        = None
        self.act_model          = None
        self.value              = None
        self.n_batch            = None
        self.summary            = None

        super().__init__(policy=policy, env=env, verbose=verbose, requires_vec_env=True,
                         _init_setup_model=_init_setup_model, policy_kwargs=policy_kwargs,
                         seed=seed, n_cpu_tf_sess=n_cpu_tf_sess)

        if _init_setup_model:
            self.setup_model()

    def _make_runner(self):
        return Runner(env=self.env, model=self, models=self.models, n_steps=self.n_steps, gamma=self.gamma, lam=self.lam)

    def _get_pretrain_placeholders(self):
        policy = self.act_model
        if isinstance(self.action_space, gym.spaces.Discrete):
            return policy.obs_ph, self.action_ph, policy.policy
        return policy.obs_ph, self.action_ph, policy.deterministic_action

    def setup_model(self):
        self.models = []
        for i in range(self.models_num):
            self.models.append(SubModel(self, i))

    def _train_step(self, learning_rate, cliprange, obs, returns, masks, actions, values, neglogpacs, update, model,
                    writer, states=None, cliprange_vf=None):
        """
        Training of PPO2 Algorithm

        :param learning_rate: (float) learning rate
        :param cliprange: (float) Clipping factor
        :param obs: (np.ndarray) The current observation of the environment
        :param returns: (np.ndarray) the rewards
        :param masks: (np.ndarray) The last masks for done episodes (used in recurent policies)
        :param actions: (np.ndarray) the actions
        :param values: (np.ndarray) the values
        :param neglogpacs: (np.ndarray) Negative Log-likelihood probability of Actions
        :param update: (int) the current step iteration
        :param writer: (TensorFlow Summary.writer) the writer for tensorboard
        :param states: (np.ndarray) For recurrent policies, the internal state of the recurrent model
        :return: policy gradient loss, value function loss, policy entropy,
                approximation of kl divergence, updated clipping range, training update operation
        :param cliprange_vf: (float) Clipping factor for the value function
        """

        advs = returns - values
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        td_map = {model.train_model.obs_ph: obs, model.action_ph: actions,
                  model.advs_ph: advs, model.rewards_ph: returns,
                  model.learning_rate_ph: learning_rate, model.clip_range_ph: cliprange,
                  model.old_neglog_pac_ph: neglogpacs, model.old_vpred_ph: values}
        if states is not None:
            td_map[model.train_model.states_ph] = states
            td_map[model.train_model.dones_ph] = masks

        if cliprange_vf is not None and cliprange_vf >= 0:
            td_map[model.clip_range_vf_ph] = cliprange_vf

        if states is None:
            update_fac = max(model.n_batch // self.nminibatches // self.noptepochs, 1)
        else:
            update_fac = max(model.n_batch // self.nminibatches // self.noptepochs // self.n_steps, 1)

        if writer is not None:
            # run loss backprop with summary, but once every 10 runs save the metadata (memory, compute time, ...)
            if self.full_tensorboard_log and (1 + update) % 10 == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ = model.sess.run([self.summary, self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self._train], td_map, options=run_options, run_metadata=run_metadata)
                writer.add_run_metadata(run_metadata, 'step%d' % (update * update_fac))
            else:
                summary, policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ = model.sess.run([model.summary, model.pg_loss, model.vf_loss, model.entropy, model.approxkl, model.clipfrac, model._train],td_map)
            writer.add_summary(summary, (update * update_fac))
        else:
            policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ = model.sess.run([self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self._train], td_map)

        return policy_loss, value_loss, policy_entropy, approxkl, clipfrac

    def learn(self, total_timesteps, callback=None, log_interval=1, tb_log_name="Dual",
              reset_num_timesteps=True):
        # Transform to callable if needed
        self.learning_rate  = get_schedule_fn(self.learning_rate)
        self.cliprange      = get_schedule_fn(self.cliprange)
        cliprange_vf        = get_schedule_fn(self.cliprange_vf)

        new_tb_log   = self._init_num_timesteps(reset_num_timesteps)
        top_callback = SaveOnTopRewardCallback(check_freq=self.n_steps, logdir=self.tensorboard_log, models_num=self.models_num)
        callback.append(top_callback)
        callback     = self._init_callback(callback)

        with SetVerbosity(self.verbose), TensorboardWriter(self.models[0].graph, self.tensorboard_log, tb_log_name, new_tb_log) as writer:

            for model in self.models:
                model._setup_learn(self)

            t_first_start = time.time()
            n_updates     = total_timesteps // (self.n_envs * self.n_steps)
            update = 0
            done = False

            callback.on_training_start(locals(), globals())


            while not done:
                assert (self.n_envs * self.n_steps) % self.nminibatches == 0, ("The number of minibatches (`nminibatches`) is not a factor of the total number of samples collected per rollout (`n_batch`), some samples won't be used.")

                batch_size       = (self.n_envs * self.n_steps) // self.nminibatches
                t_start          = time.time()
                frac             = 1.0 - (update - 1.0) / n_updates
                lr_now           = self.learning_rate(frac)
                cliprange_now    = self.cliprange(frac)
                cliprange_vf_now = cliprange_vf(frac)

                callback.on_rollout_start()

                rollouts = self.runner.run(callback) #execute episode

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
                        mb_loss_vals = []
                        if states is None:  # nonrecurrent version
                            update_fac = max(model.n_batch // self.nminibatches // self.noptepochs, 1)
                            inds = np.arange(len(obs))#np.arange(model.n_batch)
                            for epoch_num in range(self.noptepochs):
                                np.random.shuffle(inds)
                                for start in range(0, model.n_batch, batch_size):
                                    timestep = self.num_timesteps // update_fac + ((epoch_num * model.n_batch + start) // batch_size)
                                    end      = start + batch_size
                                    mbinds   = inds[start:end]
                                    if len(obs) > 1:
                                        slices   = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                                        mb_loss_vals.append(self._train_step(lr_now, cliprange_now, *slices, model=self.models[i], writer=writer, update=timestep, cliprange_vf=cliprange_vf_now))
                                    else:
                                        mb_loss_vals.append((0,0,0,0,0))
                            
                        else:
                            exit("does not support recurrent version")

                        loss_vals = np.mean(mb_loss_vals, axis=0)
                        t_now     = time.time()
                        fps       = int(model.n_batch / (t_now - t_start))

                        if writer is not None:
                            n_steps = model.n_batch
                            try:
                                #print("true reward:{}".format(true_reward.shape))
                                total_episode_reward_logger(self.episode_reward, true_reward.reshape((self.n_envs, n_steps)), masks.reshape((self.n_envs, n_steps)), writer, self.num_timesteps)

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
                            logger.logkv("Steps", steps_used)
                            logger.dumpkvs()
                    
                    i+=1
                print("Steps: " + str(self.num_timesteps) + "/" + str(total_timesteps) + " - (" + str(round(self.num_timesteps/total_timesteps*100)) + "%)")
                print("Episodes: " + str(update+1) + "/" + str(n_updates) + " - (" + str(round((update+1)/n_updates*100)) + "%)")
                if self.num_timesteps >= total_timesteps:
                    done = True
                update +=1
            callback.on_training_end()
            return self

# dual

    def save(self, parent_path, cloudpickle=False):
        data = {
            "gamma": self.gamma,
            "n_steps": self.n_steps,
            "vf_coef": self.vf_coef,
            "ent_coef": self.ent_coef,
            "max_grad_norm": self.max_grad_norm,
            "learning_rate": self.learning_rate,
            "lam": self.lam,
            "nminibatches": self.nminibatches,
            "noptepochs": self.noptepochs,
            "cliprange": self.cliprange,
            "cliprange_vf": self.cliprange_vf,
            "verbose": self.verbose,
            "policy": self.policy,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "n_envs": self.n_envs,
            "n_cpu_tf_sess": self.n_cpu_tf_sess,
            "seed": self.seed,
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


        import commentjson

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

class SubModel(MultiPPO2):
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

            assert issubclass(parent.policy, ActorCriticPolicy), "Error: the input policy for the PPO2 model must be " \
                                                               "an instance of common.policies.ActorCriticPolicy."

            self.n_batch = parent.n_envs * parent.n_steps

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.set_random_seed(parent.seed)
                self.sess = tf_util.make_session(num_cpu=parent.n_cpu_tf_sess, graph=self.graph)

                n_batch_step  = None
                n_batch_train = None
                
                if issubclass(parent.policy, RecurrentActorCriticPolicy):
                    assert parent.n_envs % parent.nminibatches == 0, "For recurrent policies, the number of environments run in parallel should be a multiple of nminibatches."
                    n_batch_step  = parent.n_envs
                    n_batch_train = parent.n_batch // parent.nminibatches

                act_model = parent.policy(self.sess, self.observation_space, parent.action_space, parent.n_envs, 1, n_batch_step, reuse=False, **parent.policy_kwargs)

                with tf.variable_scope("train_model", reuse=True, custom_getter=tf_util.outer_scope_getter("train_model")):
                    train_model = parent.policy(self.sess, self.observation_space, parent.action_space, parent.n_envs // parent.nminibatches, parent.n_steps, n_batch_train, reuse=True, **parent.policy_kwargs)

                with tf.variable_scope("loss_network_{}".format(self.model_num), reuse=False):
                    self.action_ph          = train_model.pdtype.sample_placeholder([None], name="action_ph")
                    self.advs_ph            = tf.placeholder(tf.float32, [None], name="advs_ph")
                    self.rewards_ph         = tf.placeholder(tf.float32, [None], name="rewards_ph")
                    self.old_neglog_pac_ph  = tf.placeholder(tf.float32, [None], name="old_neglog_pac_ph")
                    self.old_vpred_ph       = tf.placeholder(tf.float32, [None], name="old_vpred_ph")
                    self.learning_rate_ph   = tf.placeholder(tf.float32, [],     name="learning_rate_ph")
                    self.clip_range_ph      = tf.placeholder(tf.float32, [],     name="clip_range_ph")

                    neglogpac = train_model.proba_distribution.neglogp(self.action_ph)
                    self.entropy = tf.reduce_mean(train_model.proba_distribution.entropy())

                    vpred = train_model.value_flat

                    # Value function clipping: not present in the original PPO
                    if parent.cliprange_vf is None:
                        # Default behavior (legacy from OpenAI baselines):
                        # use the same clipping as for the policy
                        self.clip_range_vf_ph = self.clip_range_ph
                        parent.cliprange_vf = parent.cliprange
                    elif isinstance(parent.cliprange_vf, (float, int)) and parent.cliprange_vf < 0:
                        # Original PPO implementation: no value function clipping
                        self.clip_range_vf_ph = None
                    else:
                        # Last possible behavior: clipping range
                        # specific to the value function
                        self.clip_range_vf_ph = tf.placeholder(tf.float32, [], name="clip_range_vf_ph")

                    if self.clip_range_vf_ph is None:
                        # No clipping
                        vpred_clipped = train_model.value_flat
                    else:
                        # Clip the different between old and new value
                        # NOTE: this depends on the reward scaling
                        vpred_clipped = self.old_vpred_ph + tf.clip_by_value(train_model.value_flat - self.old_vpred_ph, - self.clip_range_vf_ph, self.clip_range_vf_ph)

                    vf_losses1    = tf.square(vpred - self.rewards_ph)
                    vf_losses2    = tf.square(vpred_clipped - self.rewards_ph)
                    self.vf_loss  = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

                    ratio         = tf.exp(self.old_neglog_pac_ph - neglogpac)
                    pg_losses     = -self.advs_ph * ratio
                    pg_losses2    = -self.advs_ph * tf.clip_by_value(ratio, 1.0 - self.clip_range_ph, 1.0 + self.clip_range_ph)
                    self.pg_loss  = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
                    self.approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - self.old_neglog_pac_ph))
                    self.clipfrac = tf.reduce_mean(tf.cast(tf.greater(tf.abs(ratio - 1.0), self.clip_range_ph), tf.float32))
                    loss          = self.pg_loss - self.entropy * parent.ent_coef + self.vf_loss * parent.vf_coef

                    tf.summary.scalar('entropy_loss'                , self.entropy)
                    tf.summary.scalar('policy_gradient_loss'        , self.pg_loss)
                    tf.summary.scalar('value_function_loss'         , self.vf_loss)
                    tf.summary.scalar('approximate_kullback-leibler', self.approxkl)
                    tf.summary.scalar('clip_factor'                 , self.clipfrac)
                    if hasattr(self.env, 'envs'):
                        tf.summary.scalar('network_reward'          , self.env.envs[0].env.reward.network_rewards[self.model_num])
                    else:
                        tf.summary.scalar('network_reward'          , self.env.reward.network_rewards[self.model_num])
                    tf.summary.scalar('loss', loss)

                    with tf.variable_scope('model'):
                        self.params = tf.trainable_variables()
                        if parent.full_tensorboard_log:
                            for var in self.params:
                                tf.summary.histogram(var.name, var)

                    grads = tf.gradients(loss, self.params)
                    if parent.max_grad_norm is not None:
                        grads, _grad_norm = tf.clip_by_global_norm(grads, parent.max_grad_norm)
                    grads = list(zip(grads, self.params))

                trainer         = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph, epsilon=1e-5)
                self._train     = trainer.apply_gradients(grads)

                self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

                with tf.variable_scope("input_info", reuse=False):
                    tf.summary.scalar('discounted_rewards'  , tf.reduce_mean(self.rewards_ph))
                    tf.summary.scalar('learning_rate'       , tf.reduce_mean(self.learning_rate_ph))
                    tf.summary.scalar('advantage'           , tf.reduce_mean(self.advs_ph))
                    tf.summary.scalar('clip_range'          , tf.reduce_mean(self.clip_range_ph))

                    if self.clip_range_vf_ph is not None:
                        tf.summary.scalar('clip_range_vf'   , tf.reduce_mean(self.clip_range_vf_ph))

                    tf.summary.scalar('old_neglog_action_probability'   , tf.reduce_mean(self.old_neglog_pac_ph))
                    tf.summary.scalar('old_value_pred'                  , tf.reduce_mean(self.old_vpred_ph))

                    if parent.full_tensorboard_log:
                        tf.summary.histogram('discounted_rewards'           , self.rewards_ph)
                        tf.summary.histogram('learning_rate'                , self.learning_rate_ph)
                        tf.summary.histogram('advantage'                    , self.advs_ph)
                        tf.summary.histogram('clip_range'                   , self.clip_range_ph)
                        tf.summary.histogram('old_neglog_action_probability', self.old_neglog_pac_ph)
                        tf.summary.histogram('old_value_pred'               , self.old_vpred_ph)

                        if tf_util.is_image(self.observation_space):
                            tf.summary.image('observation'      , train_model.obs_ph)
                        else:
                            tf.summary.histogram('observation'  , train_model.obs_ph)

                self.train_model    = train_model
                self.act_model      = act_model
                self.step           = act_model.step
                self.proba_step     = act_model.proba_step
                self.value          = act_model.value
                self.initial_state  = act_model.initial_state

                tf.global_variables_initializer().run(session=self.sess)  # pylint: disable=E1101

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

    def set_random_seed(self, seed: int) -> None:
        """
        :param seed: (Optional[int]) Seed for the pseudo-random generators. If None,
            do not change the seeds.
        """
        # Ignore if the seed is None
        if seed is None:
            return
        # Seed python, numpy and tf random generator
        set_global_seeds(seed)
        if self.env is not None:
            self.env.seed(seed)
            # Seed the action space
            # useful when selecting random actions
            self.env.action_space.seed(seed)

    def save(self, parent_path, i, cloudpickle=False):
        # save only one model
        data = {
            "gamma": self.parent.gamma,
            "n_steps": self.parent.n_steps,
            "vf_coef": self.parent.vf_coef,
            "ent_coef": self.parent.ent_coef,
            "max_grad_norm": self.parent.max_grad_norm,
            "learning_rate": self.parent.learning_rate,
            "lam": self.parent.lam,
            "nminibatches": self.parent.nminibatches,
            "noptepochs": self.parent.noptepochs,
            "cliprange": self.parent.cliprange,
            "cliprange_vf": self.parent.cliprange_vf,
            "verbose": self.parent.verbose,
            "policy": self.parent.policy,
            "observation_space": self.observation_space,
            "action_space": self.parent.action_space,
            "n_envs": self.parent.n_envs,
            "n_cpu_tf_sess": self.parent.n_cpu_tf_sess,
            "seed": self.parent.seed,
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


class Runner(AbstractEnvRunner):
    def __init__(self, *, env, model, models, n_steps, gamma, lam):
        """
        A runner to learn the policy of an environment for a model

        :param env: (Gym environment) The environment to learn from
        :param model: (Model) The model to learn
        :param n_steps: (int) The number of steps to run for each environment
        :param gamma: (float) Discount factor
        :param lam: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        """
        super().__init__(env=env, model=model, n_steps=n_steps)
        self.models = models
        self.lam    = lam
        self.gamma  = gamma

    def _run(self):
        """
        Run a learning step of the model

        :return:
            - observations: (np.ndarray) the observations
            - rewards: (np.ndarray) the rewards
            - masks: (numpy bool) whether an episode is over or not
            - actions: (np.ndarray) the actions
            - values: (np.ndarray) the value function output
            - negative log probabilities: (np.ndarray)
            - states: (np.ndarray) the internal states of the recurrent policies
            - infos: (dict) the extra information of the model
        """
        # mb stands for minibatch

        minibatches = []
        for model in self.models:
            mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [], [], [], [], [], []
            minibatch = mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs
            minibatches.append(minibatch)

        mb_states = self.states
        ep_infos = []        
        for model in self.models:
            model.n_batch = 0
        last_success = 0
        last_owner = -1
        for _ in range(self.n_steps):

            owner = self.model.approved(self.obs)

            model = self.models[owner]
            actions, values, self.states, neglogpacs = model.step(self.obs, self.states, self.dones)
            successful_stages = owner
            if self.states:
                successful_stages += 1
                print ("Episode successfully finished at step {}".format(_))
            if owner > last_owner: 
                #print("Changed to Net {} at step {}".format(owner,_))
                last_owner = owner
            #print(_)
            #if last_success != successful_stages:
            #    print("Changed to Net {} at step {}".format(successful_stages,_))
            #    last_success = successful_stages
            minibatch = minibatches[owner]
            mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = minibatch

            mb_obs.append       (self.obs.copy())
            mb_actions.append   (actions)
            mb_values.append    (values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append     (self.dones)

            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.env.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)

            self.obs[:], rewards, self.dones, infos = self.env.step(clipped_actions)
            # self.model.num_timesteps += self.n_envs
            self.model.num_timesteps += self.n_envs

            if self.callback is not None:
                # Abort training early
                self.callback.update_locals(locals())
                if self.callback.on_step() is False:
                    self.continue_training = False
                    # Return dummy values
                    return [None] * 9

            for info in infos:
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    ep_infos.append(maybe_ep_info)
            mb_rewards.append(rewards)
            model.n_batch += 1
            if self.dones:
                #print('Finished batch training')
                break
        # batch of steps to batch of rollouts
        last_values          = model.value(self.obs, self.states, self.dones) # last observation, last state, last done
        finished_minibatches = []
        # discount/bootstrap off value fn
        #print("Minibatches: {}".format(len(minibatches)))
        for i, minibatch in enumerate(minibatches):
            #print("Step:{}".format(i))
            mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = minibatch
            
            mb_obs        = np.asarray(mb_obs,        dtype=self.obs.dtype)
            mb_rewards    = np.asarray(mb_rewards,    dtype=np.float32)
            mb_actions    = np.asarray(mb_actions)
            mb_values     = np.asarray(mb_values,     dtype=np.float32)
            mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
            mb_dones      = np.asarray(mb_dones,      dtype=np.bool)

            mb_advs      = np.zeros_like(mb_rewards) # 0s long as rewards
            true_reward  = np.copy(mb_rewards)   # true rewards list
            last_gae_lam = 0
            count = len(mb_rewards) # number of steps in this minibatch
            self.models[i].episode_reward = sum(true_reward)
            for step in reversed(range(count)):
                if step == count - 1:
                    nextnonterminal = 1.0 - self.dones
                    nextvalues = last_values
                else:
                    nextnonterminal = 1.0 - mb_dones[step + 1]
                    nextvalues = mb_values[step + 1]
                delta = mb_rewards[step] + self.gamma * nextvalues * nextnonterminal - mb_values[step]
                mb_advs[step] = last_gae_lam = delta + self.gamma * self.lam * nextnonterminal * last_gae_lam
            mb_returns = mb_advs + mb_values
            try:
                mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward = map(swap_and_flatten, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward))
            except:
              #print("model {} had no data".format(i))
              pass  
            finished_minibatch = mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, mb_states, ep_infos, true_reward, successful_stages
            finished_minibatches.append(finished_minibatch)
        steps_taken = [finished_minibatches[x][0].shape[0] for x in range(len(finished_minibatches))]
        return finished_minibatches, steps_taken

        #return mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, mb_states, ep_infos, true_reward, successful_stages

def swap_and_flatten(arr):
    """
    swap and then flatten axes 0 and 1

    :param arr: (np.ndarray)
    :return: (np.ndarray)
    """
    shape = arr.shape
    return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])
