import gymnasium as gym

from sbx import DDPG, DQN, PPO, SAC, TD3, TQC, CrossQ

env = gym.make("Pendulum-v1", render_mode="human")

model = TQC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000, progress_bar=True)

vec_env = model.get_env()
obs = vec_env.reset()
for _ in range(1000):
    vec_env.render()
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)

vec_env.close()


