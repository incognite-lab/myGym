import random
from collections import deque

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def _flatten_observation(obs):
    if isinstance(obs, dict):
        flat = []
        if "actual_state" in obs:
            flat.append(np.array(obs["actual_state"], dtype=np.float32))
        if "goal_state" in obs:
            flat.append(np.array(obs["goal_state"], dtype=np.float32))
        if "additional_obs" in obs and "endeff_xyz" in obs["additional_obs"]:
            flat.append(np.array(obs["additional_obs"]["endeff_xyz"], dtype=np.float32))
        obs = np.concatenate(flat)
    return obs


class DeciderPolicy(nn.Module):
    def __init__(self, obs_dim, num_networks, lstm_hidden_dim=64, hidden_dim=128, lr=1e-3, buffer_size=10000,
                 batch_size=64, entropy_coef=0.03, baseline_beta=0.9, temperature=2):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_networks = num_networks

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, lstm_hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(lstm_hidden_dim, num_networks)

        self.optimizer = th.optim.Adam(self.parameters(), lr=lr)
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.baseline_beta = baseline_beta
        self.temperature = temperature
        self.register_buffer('running_reward_mean', th.tensor(0.0))

        self.hidden = None  # LSTM hidden state

    def reset_hidden(self):
        h = (th.zeros(1, 1, self.lstm_hidden_dim),
             th.zeros(1, 1, self.lstm_hidden_dim))
        self.hidden = h
        return h

    def predict(self, obs, deterministic=False):
        obs = _flatten_observation(obs)
        obs_tensor = th.tensor(obs, dtype=th.float32).unsqueeze(0).unsqueeze(0)  # shape: (1, 1, obs_dim)
        x = F.relu(self.fc1(obs_tensor))

        if self.hidden is None:
            self.reset_hidden()

        lstm_out, self.hidden = self.lstm(x, self.hidden)
        logits = self.output_layer(lstm_out.squeeze(0))  # shape: (1, num_networks)

        if not deterministic:
            self.temperature = max(self.temperature * 0.995, 0.7)

        probs = F.softmax(logits / self.temperature, dim=-1)
        if deterministic:
            return th.argmax(probs).item()
        return Categorical(probs).sample().item()

    def store(self, obs, action, reward):
        obs = _flatten_observation(obs)
        self.buffer.append((obs, action, reward))

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None

        batch = random.sample(self.buffer, self.batch_size)
        obs_batch, action_batch, reward_batch = zip(*batch)

        obs_tensor = th.tensor(np.stack(obs_batch), dtype=th.float32).unsqueeze(1)  # (batch, 1, obs_dim)
        action_tensor = th.tensor(action_batch, dtype=th.long)
        reward_tensor = th.tensor(reward_batch, dtype=th.float32)

        # Advantage calculation with running baseline
        batch_mean = reward_tensor.mean().detach()
        self.running_reward_mean = (
                self.baseline_beta * self.running_reward_mean +
                (1 - self.baseline_beta) * batch_mean
        )
        advantages = reward_tensor - self.running_reward_mean

        x = F.relu(self.fc1(obs_tensor))
        lstm_out, _ = self.lstm(x)
        logits = self.output_layer(lstm_out.squeeze(1))  # shape: (batch, num_networks)

        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs[range(len(action_tensor)), action_tensor]

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
            obs = _flatten_observation(obs)
            obs_tensor = th.tensor(obs, dtype=th.float32).unsqueeze(0).unsqueeze(0)
            x = F.relu(self.fc1(obs_tensor))
            h = self.hidden if self.hidden is not None else self.reset_hidden()
            lstm_out, _ = self.lstm(x, self.hidden if self.hidden is not None else self.reset_hidden())
            logits = self.output_layer(lstm_out.squeeze(0))
            return F.softmax(logits, dim=-1).squeeze().numpy()
