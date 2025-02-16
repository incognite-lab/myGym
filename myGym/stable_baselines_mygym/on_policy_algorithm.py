import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.base_class import BaseAlgorithm
from myGym.stable_baselines_mygym.buffersSB3 import RolloutBuffer, DictRolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv

SelfOnPolicyAlgorithm = TypeVar("SelfOnPolicyAlgorithm", bound="OnPolicyAlgorithm")


class OnPolicyAlgorithm(BaseAlgorithm):
    """
    The base for On-Policy algorithms (ex: A2C/PPO).

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param supported_action_spaces: The action spaces supported by the algorithm.
    """

    rollout_buffer: RolloutBuffer
    policy: ActorCriticPolicy

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[Tuple[Type[spaces.Space], ...]] = None,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            support_multi_env=True,
            monitor_wrapper=monitor_wrapper,
            seed=seed,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=supported_action_spaces,
        )

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer_class = rollout_buffer_class
        self.rollout_buffer_kwargs = rollout_buffer_kwargs or {}

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        if self.rollout_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                self.rollout_buffer_class = DictRolloutBuffer
            else:
                self.rollout_buffer_class = RolloutBuffer

        self.rollout_buffer = self.rollout_buffer_class(
            self.n_steps,
            self.observation_space,  # type: ignore[arg-type]
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            num_models = self.models_num,
            **self.rollout_buffer_kwargs,
        )

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        start_time = time.time()
        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            for i in range(self.num_models):
                self.models[i].policy.reset_noise(env.num_envs)
                self.models[i].policy.set_training_mode(False)
        start_time = time.time()

        callback.on_rollout_start()
        #arrays for one iteration
        actions = np.zeros((self.n_envs, self.action_space.shape[0]))
        rewards = np.zeros(self.n_envs)
        values = np.zeros(self.n_envs)
        log_probs = np.zeros(self.n_envs)
        dones = np.zeros(self.n_envs)
        new_obs = np.zeros((self.n_envs, self.observation_space.shape[0]))
        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix for every model
                for model in self.models:
                    model.policy.reset_noise(env.num_envs)
            # Choosing model based on observation
            owner = self.approved(self._last_obs)
            #owner = [0, 0, 0, 0]
            if isinstance(owner, list):

                #OLDEr version of retrieving actions
                """
                for i in range(len(self.models)):
                    model = self.models[i]
                    if i in owner:
                        owner = np.array(owner)
                        # Indexes of current model got from observation
                        indexes = np.where(owner == i)
                        with th.no_grad():
                            # Convert to pytorch tensor or to TensorDict
                            obs_tensor = obs_as_tensor(self._last_obs, self.device)
                            obs_i = obs_tensor[indexes]
                            actions_i, values_i, log_probs_i = model.policy(obs_i)
                        actions[indexes] = actions_i
                        values[indexes] = values_i
                        log_probs[indexes] = log_probs_i
                        # actions_i = actions.cpu().numpy()
                        # Rescale and perform action
                """
                owner = np.array(owner)
                for i in range(np.max(owner)+1):
                    model = self.models[i]
                    indices = np.where(owner == i)

                    with th.no_grad():
                        obs_tensor = obs_as_tensor(self._last_obs, self.device)
                        actions_i, values_i, log_probs_i = model.policy(obs_tensor)

                    actions[indices] = actions_i[indices].cpu().numpy()
                    values[indices] = values_i[indices].cpu().numpy().squeeze()
                    log_probs[indices] = log_probs_i[indices].cpu().numpy()

                actions = actions_i.cpu().numpy()
                values = values_i.cpu().numpy()
                log_probs = log_probs_i.cpu().numpy()

                # with th.no_grad():
                #     obs_tensor = obs_as_tensor(self._last_obs, self.device)
                #actions, values, log_probs = self.env.get_actions(owner, self._last_obs)

                #print("action returned from get_actions:", actions)
                model = self.models[owner[0]]
                # actions = actions.cpu().numpy()
                # actions = th.from_numpy(actions)
                # values = th.from_numpy(values)
                # log_probs = th.from_numpy(log_probs)


                # with th.no_grad():
                #     # Convert to pytorch tensor or to TensorDict
                #     obs_tensor = obs_as_tensor(self._last_obs, self.device)
                #     actions, values, log_probs = self.models[owner[0]].policy(obs_tensor)

                #actions = actions.cpu().numpy()
                model = self.models[owner[0]]
                values = np.squeeze(values)

            else:
                with th.no_grad():
                    model = self.models[owner]
                    # Convert to pytorch tensor or to TensorDict
                    obs_tensor = obs_as_tensor(self._last_obs, self.device)
                    actions, values, log_probs = model.policy(obs_tensor)
            #actions = actions.cpu().numpy()
            if isinstance(self.action_space, spaces.Box):
                if model.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = model.policy.unscale_action(actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
            #print("clipped actions:", clipped_actions)
            new_obs, rewards, dones, infos = env.step(clipped_actions)

            if isinstance(owner, list):
                self.num_timesteps += env.num_envs
            else:
                self.num_timesteps += 1
            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False
            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = model.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = model.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value
            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
                owner #array (different owner for each env)
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            owner = self.approved(self._last_obs)
            if isinstance(owner, list):
                owner = np.array(owner)
                for i in range(np.max(owner)+1):
                    indices = np.where(owner == i)
                    model = self.models[i]
                    values_i = model.policy.predict_values(obs_as_tensor(new_obs, self.device))
                    print("values_i:", values_i)# type: ignore[arg-type]
                    values[indices] = values_i[indices].squeeze()
                    print("values:", values)
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.update_locals(locals())
        callback.on_rollout_end()
        end_time = time.time()
        total_time = end_time - start_time
        print("rollout_time:", total_time)
        return True

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def _dump_logs(self, iteration: int) -> None:
        """
        Write log.

        :param iteration: Current logging iteration
        """
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.logger.record("time/iterations", iteration, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        self.logger.dump(step=self.num_timesteps)

    def learn(
        self: SelfOnPolicyAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfOnPolicyAlgorithm:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None
        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)
            if not continue_training:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                self._dump_logs(iteration)

            self.train()

        callback.on_training_end()

        return self

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []
