
# This file is here just to define MlpPolicy/CnnPolicy
# that work for PPO
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, register_policy

TorchMlpPolicy = ActorCriticPolicy
CnnPolicy = ActorCriticCnnPolicy

register_policy("TorchMlpPolicy", ActorCriticPolicy)
register_policy("CnnPolicy", ActorCriticCnnPolicy)