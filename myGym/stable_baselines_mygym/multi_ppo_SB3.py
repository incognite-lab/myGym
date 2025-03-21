import warnings
from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union, Iterable, Tuple

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines_mygym.Subproc_vec_envSB3 import SubprocVecEnv
from torch.nn import functional as F
import os
import sys
import pathlib
import io

from stable_baselines3.common.save_util import save_to_zip_file, recursive_getattr, load_from_zip_file
from stable_baselines3.common.buffers import RolloutBuffer
#from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from myGym.stable_baselines_mygym.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn

from stable_baselines3.common.vec_env.patch_gym import _convert_space, _patch_env

SelfPPO = TypeVar("SelfPPO", bound="PPO")
SelfBaseAlgorithm = TypeVar("SelfBaseAlgorithm", bound="BaseAlgorithm")

"""
MultiPPO algorithm based on stable_baselines3's PPO algorithm. Implements switching of networks during training when a
subtask is finished.
"""

class MultiPPOSB3(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        n_models: int = 1,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"


        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.models_num = n_models
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        self.models = []
        if _init_setup_model:
            self._setup_model()


    def _setup_model(self) -> None:
        super()._setup_model()
        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"
            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)
        for i in range(self.models_num):
            self.models.append(SubModel(self, i))


    def set_env(self, env) -> None:
        super().set_env(env)
        self.n_batch = self.n_envs * self.n_steps


    def save(
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase], steps = None,
        best = False,
        exclude: Optional[Iterable[str]] = None,
        include: Optional[Iterable[str]] = None,
    ) -> None:
        """
        Save all the attributes of the object and the model parameters in a zip-file.

        :param path: path to the file where the rl agent should be saved
        :param exclude: name of parameters that should be excluded in addition to the default ones
        :param include: name of parameters that might be excluded but should be included anyway
        """
        # Copy parameter list so we don't mutate the original dict
        data = self.__dict__.copy()
        with open(os.path.join(path, "trained_steps.txt"), "a") as f:
            f.write(f"{steps}\n")

        # Exclude is union of specified parameters (if any) and standard exclusions
        if exclude is None:
            exclude = []
        exclude = set(exclude).union(self._excluded_save_params())

        # Do not exclude params if they are specifically included
        if include is not None:
            exclude = exclude.difference(include)

        state_dicts_names, torch_variable_names = self._get_torch_save_params()
        all_pytorch_variables = state_dicts_names + torch_variable_names
        for torch_var in all_pytorch_variables:
            # We need to get only the name of the top most module as we'll remove that
            var_name = torch_var.split(".")[0]
            # Any params that are in the save vars must not be saved by data
            exclude.add(var_name)

        # Remove parameter entries of parameters which are to be excluded
        for param_name in exclude:
            data.pop(param_name, None)

        # Build dict of torch variables
        pytorch_variables = None
        if torch_variable_names is not None:
            pytorch_variables = {}
            for name in torch_variable_names:
                attr = recursive_getattr(self, name)
                pytorch_variables[name] = attr
        # Build dict of state_dicts
        data.pop("models")
        # with open()
        if isinstance(self.env, VecMonitor):
            reward_names = self.env.get_attr("reward")[0].network_names
        else:
            reward_names = self.env.reward.network_names
        # path_steps = os.path.join(path, f"steps_{steps}/" )
        for i in range(len(self.models)):
            model = self.models[i]
            params_to_save = model.get_parameters()
            if not best:
                save_path = os.path.join(path, reward_names[i] + f"/steps_{steps}")
            else:
                save_path = os.path.join(path, reward_names[i] + "/best_model")
            save_to_zip_file(save_path, data=data, params=params_to_save, pytorch_variables=pytorch_variables)


    def approved(self, observation):
        # based on obs, decide which model should be used
        if hasattr(self.env, 'envs'):
            submodel_id = self.env.envs[0].env.env.reward.network_switch_control(self.env.envs[0].env.env.observation["task_objects"])
        elif isinstance(self.env, VecMonitor):
            submodel_id = self.env.network_control()
        else:
            submodel_id = self.env.reward.network_switch_control(self.env.observation["task_objects"])
        return submodel_id


    def predict(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        owner = self.approved(observation)
        #MUltiprocessing
        if isinstance(owner, list):
            owner = np.array(owner)
            actions = np.zeros((self.n_envs, self.action_space.shape[0]))

            for i in range(np.max(owner) + 1):
                model = self.models[i]
                indices = np.where(owner == i)
                action_i, state_i = model.policy.predict(observation, state, episode_start, deterministic)
                actions[indices] = action_i[indices]
        #Single process
        else:
            model = self.models[owner]
            actions, state =model.policy.predict(observation, state, episode_start, deterministic)
        return actions, state


    def eval_predict(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        owner = self.approved(observation)
        #print("owner=", owner)
        #print("observation", observation)
        owner = owner[0]
        model = self.models[owner]
        action, state = model.policy.predict(observation, state, episode_start, deterministic)
        return action, state


    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        for model in self.models:
            policy = model.policy
            policy.set_training_mode(True)
            # Update optimizer learning rate
            self._update_learning_rate(policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data, owner in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                # Choose appropriate submodel
                model = self.models[owner]

                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()
                # print("actions", actions)
                # print("observations", rollout_data.observations)
                values, log_prob, entropy = model.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                model.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(model.policy.parameters(), self.max_grad_norm)
                model.policy.optimizer.step()
            self._n_updates += 1
            if not continue_training:
                break
        explained_vars = []
        owner_sizes = self.rollout_buffer.owner_sizes
        for i in range(self.models_num):
            val_arr = self.rollout_buffer.values[sum(owner_sizes[:i]):sum(owner_sizes[:i+1])]
            ret_arr = self.rollout_buffer.returns[sum(owner_sizes[:i]):sum(owner_sizes[:i+1])]
            if val_arr.shape != ret_arr.shape:
                print("Value and return arr shapes are not equal:")
                print("val_arr shape:", val_arr.shape)
                print("ret_arr shape:", ret_arr.shape)
            if owner_sizes[i] != 0:
                flat_val = np.array(val_arr).flatten()
                flat_ret = np.array(ret_arr).flatten()
                if flat_val.shape != flat_ret.shape:
                    print("flattened value and return arr shapes are not equal")
                try:
                    explained_vars.append(explained_variance(flat_val, np.array(ret_arr).flatten()))
                except Exception as e:
                    print("Exception", e)
                    print("happened at line line 423 in multi_ppo_SB3.py")
                    explained_vars.append(np.nan)
            else:
                explained_vars.append(np.nan)
        #explained_var = explained_variance(np.array(self.rollout_buffer.value_arrs).flatten(), np.array(self.rollout_buffer.return_arrs).flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variances", np.around(explained_vars, 5))
        for i in range(self.models_num):
            model = self.models[i]
            if hasattr(model.policy, "log_std"):
                self.logger.record("train/std" + str(i), th.exp(model.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
        self: SelfPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfPPO:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    @classmethod
    def load(  # noqa: C901
            cls: Type[SelfBaseAlgorithm],
            load_path: Union[str, pathlib.Path, io.BufferedIOBase],
            env: Optional[GymEnv] = None,
            device: Union[th.device, str] = "auto",
            custom_objects: Optional[Dict[str, Any]] = None,
            print_system_info: bool = False,
            force_reset: bool = True,
            **kwargs,
    ) -> SelfBaseAlgorithm:
        """
        Load the model from a zip-file.
        Warning: ``load`` re-creates the model from scratch, it does not update it in-place!
        For an in-place load use ``set_parameters`` instead.

        :param path: path to the file (or a file-like) where to
            load the agent from
        :param env: the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model) has priority over any saved environment
        :param device: Device on which the code should run.
        :param custom_objects: Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            ``keras.models.load_model``. Useful when you have an object in
            file that can not be deserialized.
        :param print_system_info: Whether to print system info from the saved model
            and the current system info (useful to debug loading issues)
        :param force_reset: Force call to ``reset()`` before training
            to avoid unexpected behavior.
            See https://github.com/DLR-RM/stable-baselines3/issues/597
        :param kwargs: extra arguments to change the model when loading
        :return: new model instance with loaded parameters
        """
        if print_system_info:
            print("== CURRENT SYSTEM INFO ==")
            get_system_info()

        load_path = load_path.split("/")
        #load_path = load_path[:-1]
        path = "/".join(load_path)
        dir_path = os.path.dirname(path)

        import commentjson
        with open(path + "/train.json", "r") as f:
            json = commentjson.load(f)
        num_models = json["num_networks"]
        load = [] #data, params, pytorch_variables
        if isinstance(env, VecMonitor):
            reward_names = env.get_attr("reward")[0].network_names
        else:
            reward_names = env.reward.network_names
        for i in range(num_models):
            load_path = path + "/" + reward_names[i] + "/best_model"
            data, params, pytorch_variables = load_from_zip_file(
                load_path,
                device=device,
                custom_objects=custom_objects,
                print_system_info=print_system_info,
            )
            assert data is not None, "No data found in the saved file"
            assert params is not None, "No params found in the saved file"
            load.append((data, params, pytorch_variables))
        # Remove stored device information and replace with ours

        data = load[0][0]
        if "policy_kwargs" in data:
            if "device" in data["policy_kwargs"]:
                del data["policy_kwargs"]["device"]
            # backward compatibility, convert to new format
            saved_net_arch = data["policy_kwargs"].get("net_arch")
            if saved_net_arch and isinstance(saved_net_arch, list) and isinstance(saved_net_arch[0], dict):
                data["policy_kwargs"]["net_arch"] = saved_net_arch[0]

        if "policy_kwargs" in kwargs and kwargs["policy_kwargs"] != data["policy_kwargs"]:
            raise ValueError(
                f"The specified policy kwargs do not equal the stored policy kwargs."
                f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
            )

        if "observation_space" not in data or "action_space" not in data:
            raise KeyError("The observation_space and action_space were not given, can't verify new environments")

        # Gym -> Gymnasium space conversion
        for key in {"observation_space", "action_space"}:
            data[key] = _convert_space(data[key])


        # if env is not None:
        #     # Wrap first if needed
        #     #env = cls._wrap_env(env, data["verbose"])
        #     # Check if given env is valid
        #     check_for_correct_spaces(env, data["observation_space"], data["action_space"])
        #     # Discard `_last_obs`, this will force the env to reset before training
        #     # See issue https://github.com/DLR-RM/stable-baselines3/issues/597
        #     if force_reset and data is not None:
        #         data["_last_obs"] = None
        #     # `n_envs` must be updated. See issue https://github.com/DLR-RM/stable-baselines3/issues/1018
        #     if data is not None:
        #         data["n_envs"] = env.num_envs
        # else:
        #     # Use stored env, if one exists. If not, continue as is (can be used for predict)
        #     if "env" in data:
        #         env = data["env"]
        model = cls(
            policy=data["policy_class"],
            env=env,
            device=device,
            _init_setup_model=False,  # type: ignore[call-arg]
        )

        # load parameters

        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        if env is not None:
            model.env = env
        model.models_num = num_models
        model._setup_model()
        i = 0
        for submodel in model.models:
            try:
                # put state_dicts back in place
                submodel.set_parameters(load[i][1], exact_match=True, device=device)
                print("successfully set parameters of model", i)
            except RuntimeError as e:
                # Patch to load policies saved using SB3 < 1.7.0
                # the error is probably due to old policy being loaded
                # See https://github.com/DLR-RM/stable-baselines3/issues/1233
                if "pi_features_extractor" in str(e) and "Missing key(s) in state_dict" in str(e):
                    submodel.set_parameters(params, exact_match=False, device=device)
                    warnings.warn(
                        "You are probably loading a A2C/PPO model saved with SB3 < 1.7.0, "
                        "we deactivated exact_match so you can save the model "
                        "again to avoid issues in the future "
                        "(see https://github.com/DLR-RM/stable-baselines3/issues/1233 for more info). "
                        f"Original error: {e} \n"
                        "Note: the model should still work fine, this only a warning."
                    )
                else:
                    raise e
            except ValueError as e:
                # Patch to load DQN policies saved using SB3 < 2.4.0
                # The target network params are no longer in the optimizer
                # See https://github.com/DLR-RM/stable-baselines3/pull/1963
                saved_optim_params = params["policy.optimizer"]["param_groups"][0]["params"]  # type: ignore[index]
                n_params_saved = len(saved_optim_params)
                n_params_saved = len(saved_optim_params)
                n_params = len(model.policy.optimizer.param_groups[0]["params"])
                if n_params_saved == 2 * n_params:
                    # Truncate to include only online network params
                    params["policy.optimizer"]["param_groups"][0]["params"] = saved_optim_params[
                                                                              :n_params]  # type: ignore[index]

                    submodel.set_parameters(params, exact_match=True, device=device)
                    warnings.warn(
                        "You are probably loading a DQN model saved with SB3 < 2.4.0, "
                        "we truncated the optimizer state so you can save the model "
                        "again to avoid issues in the future "
                        "(see https://github.com/DLR-RM/stable-baselines3/pull/1963 for more info). "
                        f"Original error: {e} \n"
                        "Note: the model should still work fine, this only a warning."
                    )
                else:
                    raise e

            # put other pytorch variables back in place
            if pytorch_variables is not None:
                for name in pytorch_variables:
                    # Skip if PyTorch variable was not defined (to ensure backward compatibility).
                    # This happens when using SAC/TQC.
                    # SAC has an entropy coefficient which can be fixed or optimized.
                    # If it is optimized, an additional PyTorch variable `log_ent_coef` is defined,
                    # otherwise it is initialized to `None`.
                    if pytorch_variables[name] is None:
                        continue
                    # Set the data attribute directly to avoid issue when using optimizers
                    # See https://github.com/DLR-RM/stable-baselines3/issues/391
                    recursive_setattr(submodel, f"{name}.data", pytorch_variables[name].data)
            i += 1

        # Sample gSDE exploration matrix, so it uses the right device
        # see issue #44
        if model.use_sde:
            for i in range(num_models):
                submodel = model.models[i]
                submodel.policy.reset_noise()  # type: ignore[operator]
        return model



class SubModel(MultiPPOSB3):
    def __init__(self, parent, i):
        self.model_num = i
        if isinstance(parent.env, VecMonitor):
            reward_names = parent.env.get_attr("reward")[0].network_names
        else:
            reward_names = parent.env.reward.network_names
        self.path = os.path.join(parent.tensorboard_log, reward_names[i])
        try:
            os.makedirs(self.path)
        except:
            pass
        #self.env = parent.env
        self.policy = parent.policy_class(
            parent.observation_space, parent.action_space, parent.lr_schedule, use_sde = parent.use_sde, **parent.policy_kwargs
        )
        self.policy = self.policy.to(parent.device)
        #self.parent = parent


