from gym.envs.registration import register

register(
    id="CrowWorkspaceEnv-v0",
    entry_point="myGym.envs.crow_workspace_env:CrowWorkspaceEnv",
    max_episode_steps=8192,
)

register(
    id="Gym-v0",
    entry_point="myGym.envs.gym_env:GymEnv",
    max_episode_steps=8192,
)


register(
    id="ObjectTestEnv-v0",
   entry_point="myGym.envs.object_test_env:ObjectTestEnv",
    max_episode_steps=25600,
)
