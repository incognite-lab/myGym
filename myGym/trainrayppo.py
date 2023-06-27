from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print


algo = (
    PPOConfig()
    .rollouts(num_rollout_workers=10)
    .resources(num_gpus=1)
    .environment(env="CartPole-v1")
    .build()
)

for i in range(10):
    result = algo.train()
    print(pretty_print(result))

    if i % 5 == 0:
        checkpoint_dir = algo.save()
        print(f"Checkpoint saved in directory {checkpoint_dir}")