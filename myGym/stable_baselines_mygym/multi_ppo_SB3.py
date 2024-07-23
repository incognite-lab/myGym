from pickle import NONE

import warnings
from typing import Any, Dict, Optional, Type, Union

import numpy as np
import torch as th
from gym import spaces
from torch.nn import functional as F

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn

class MultiPPOSB3(OnPolicyAlgorithm):
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

    def __init__(
            self,
            policy: Union[str, Type[ActorCriticPolicy]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Schedule] = 3e-4,
            n_steps: int = 2048,
            n_models=None,
            batch_size: int = 64,
            n_epochs: int = 10,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_range: Union[float, Schedule] = 0.2,
            clip_range_vf: Union[None, float, Schedule] = None,
            ent_coef: float = 0.0,
            vf_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            use_sde: bool = False,
            sde_sample_freq: int = -1,
            target_kl: Optional[float] = None,
            tensorboard_log: Optional[str] = None,
            create_eval_env: bool = False,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
    ):
        super(PPO, self).__init__(
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
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
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
        assert (
                batch_size > 1
        ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert (
                    buffer_size > 1
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
        self.target_kl = target_kl

        if _init_setup_model:
            self._setup_model()


    def _setup_model(self) -> None:
        self.models = []
        for i in range(self.models_num):
            self.models.append(SubModel(self, i))
            #TODO: create submodel class


