import random
from collections import deque

import matplotlib.pyplot as plt
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def plot_training_vs_evaluation(training_rewards, evaluation_rewards, save_path=None):
    """
    Plot training vs evaluation rewards over time.

    Args:
        training_rewards (list): List of training rewards collected over time.
        evaluation_rewards (list): List of evaluation rewards collected over time.
        save_path (str): Optional path to save the figure instead of displaying it.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(training_rewards, label='Training Reward', color='blue')
    plt.plot(evaluation_rewards, label='Evaluation Reward', color='green')
    plt.xlabel('Training steps / episodes')
    plt.ylabel('Average Reward')
    plt.title('Training vs Evaluation Reward Over Time')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def _flatten_observation(obs):
    if isinstance(obs, dict):
        flat = []
        if 'actual_state' in obs:
            flat += list(obs['actual_state'])
        if 'additional_obs' in obs:
            for val in obs['additional_obs'].values():
                flat += list(val)
        if 'goal_state' in obs:
            flat += list(obs['goal_state'])
        return flat
    return obs


class DeciderPolicy(nn.Module):
    def __init__(self, obs_dim, num_networks, hidden_dim=128, lr=1e-3, buffer_size=10000, batch_size=64,
                 entropy_coef=0.01, baseline_beta=0.9):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_networks)
        )
        self.optimizer = th.optim.Adam(self.model.parameters(), lr=lr)

        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.num_networks = num_networks
        self.entropy_coef = entropy_coef
        self.baseline_beta = baseline_beta
        self.register_buffer('running_reward_mean', th.tensor(0.0))

    def predict(self, obs, deterministic=False):
        with th.no_grad():
            obs = _flatten_observation(
                obs)  # maybe remove goal state if we want Decider to act blindly just based on the current state
            logits = self.model(th.tensor(obs, dtype=th.float32))
            probs = F.softmax(logits,
                              dim=0)  # decider can be trained using cross-entropy loss, comparing the predicted distribution with the "good choices"
            if deterministic:
                return th.argmax(probs).item()
            dist = Categorical(probs)
            return dist.sample().item()

    def store(self, obs, action, reward):
        # Handle dictionary observations by extracting task_objects
        if isinstance(obs, dict):
            obs = obs["task_objects"]
        self.buffer.append((obs, action, reward))

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None  # not enough data yet

        batch = random.sample(self.buffer, self.batch_size)
        obs_batch, action_batch, reward_batch = zip(*batch)

        obs_tensor = th.tensor(obs_batch, dtype=th.float32)
        action_tensor = th.tensor(action_batch, dtype=th.long)
        reward_tensor = th.tensor(reward_batch, dtype=th.float32)

        # Update running reward mean baseline
        batch_mean = reward_tensor.mean().detach()
        self.running_reward_mean = self.baseline_beta * self.running_reward_mean + (1 - self.baseline_beta) * batch_mean
        advantages = reward_tensor - self.running_reward_mean

        logits = self.model(obs_tensor)
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs[range(len(action_tensor)), action_tensor]

        # Compute losses
        policy_loss = -(advantages.detach() * selected_log_probs).mean()
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=1).mean()
        entropy_loss = -self.entropy_coef * entropy

        total_loss = policy_loss + entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            'loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'entropy': entropy.item(),
            'baseline': self.running_reward_mean.item()
        }

    def get_probabilities(self, obs):
        with th.no_grad():
            # Handle dictionary observations by extracting task_objects
            if isinstance(obs, dict):
                obs = obs["task_objects"]
            logits = self.model(th.tensor(obs, dtype=th.float32))
            return F.softmax(logits, dim=0).numpy()
