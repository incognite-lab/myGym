import random
from collections import deque
import numpy as np
import torch as th
import torch.nn.functional as F
from torch.distributions import Categorical
import os
import json

EPS = 1e-8


def flatten_obs_any(obs):
    """
    Recursively flatten an arbitrary nested myGym observation into 1D float32 numpy array.
    Deterministic ordering for dicts: keys sorted alphabetically.
    """
    if obs is None:
        return np.zeros(0, dtype=np.float32)

    if isinstance(obs, np.ndarray):
        # Handle object arrays (nested stuff)
        if obs.dtype == object:
            parts = [flatten_obs_any(x) for x in obs]
            return np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, dtype=np.float32)
        return obs.astype(np.float32).ravel()

    if isinstance(obs, dict):
        parts = [flatten_obs_any(obs[k]) for k in sorted(obs.keys())]
        return np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, dtype=np.float32)

    if isinstance(obs, (list, tuple)):
        parts = [flatten_obs_any(x) for x in obs]
        return np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, dtype=np.float32)

    # scalar
    try:
        return np.array([obs], dtype=np.float32)
    except Exception as e:
        raise TypeError(f"Unsupported observation element: {obs} ({type(obs)})") from e


class DeciderPolicy(th.nn.Module):
    """
    High-level decider for subpolicy selection.

    Interface:
      - predict(obs, deterministic=False) -> int
      - store(obs, selected_action_idx, subpolicy_idx, return_, steps_spent, success, switched, grasp_from_far)
      - update() -> metrics dict or None
      - get_probabilities(obs) -> np.ndarray
      - decay_temperature(...)
    """

    def __init__(
        self,
        obs_dim: int = None,
        num_networks: int = 3,
        hidden_sizes=(128, 64),
        lr=1e-4,
        buffer_size=4096,
        batch_size=128,
        gamma=0.9,                 # kept for API compatibility (not used here)
        baseline_beta=0.9,
        success_bonus=0.3,
        time_penalty=0.0,
        switch_penalty=0.0,
        entropy_coef=0.2,
        min_batch_for_update=128,
        max_grad_norm=2.0,
        temperature=1.5,
        grasp_far_penalty=0.07,
        eps_start=0.25,
        eps_end=0.04,
        eps_decay_steps=200_000,
        normalize_reward_by_time=True,
        log_path=None,
    ):
        super().__init__()

        self._obs_dim_given = obs_dim
        self.obs_dim = obs_dim
        self.num_networks = int(num_networks)
        self.hidden_sizes = tuple(hidden_sizes)

        # exploration schedule
        self.eps_start = float(eps_start)
        self.eps_end = float(eps_end)
        self.eps_decay_steps = float(eps_decay_steps)

        self.normalize_reward_by_time = bool(normalize_reward_by_time)

        # initial tiny model; will rebuild when obs_dim known
        self._build_model(obs_dim if obs_dim is not None else 1, self.hidden_sizes)
        self.optimizer = th.optim.Adam(self.model.parameters(), lr=lr)

        # REINFORCE buffer with replay-style sampling
        self.buffer = deque(maxlen=int(buffer_size))
        self.batch_size = int(batch_size)
        self.gamma = float(gamma)
        self.baseline_beta = float(baseline_beta)
        self.success_bonus = float(success_bonus)
        self.time_penalty = float(time_penalty)
        self.switch_penalty = float(switch_penalty)
        self.entropy_coef = float(entropy_coef)
        self.min_batch_for_update = int(min_batch_for_update)
        self.max_grad_norm = float(max_grad_norm)
        self.temperature = float(temperature)
        self.grasp_far_penalty = float(grasp_far_penalty)

        # per-network running baseline and std
        self.register_buffer("running_baselines", th.zeros(self.num_networks))
        self.register_buffer("running_stds", th.ones(self.num_networks))

        # diagnostics
        self.steps = 0                 # number of optimizer updates
        self.action_counts = np.zeros(self.num_networks, dtype=np.int64)
        self.total_actions = 0         # number of decider decisions

        # logging-related
        self.log_path = log_path  # path to .txt / .tsv file
        self.update_counter = 0  # how many times update() was called
        self.global_step = 0

    # --------------------------------------------------------------------- #
    #  Internal helpers
    # --------------------------------------------------------------------- #

    def _build_model(self, obs_dim, hidden_sizes):
        layers = []
        last = int(obs_dim)
        for h in hidden_sizes:
            h = int(h)
            layers.append(th.nn.Linear(last, h))
            layers.append(th.nn.ReLU())
            last = h
        layers.append(th.nn.Linear(last, self.num_networks))
        self.model = th.nn.Sequential(*layers)

    def _ensure_model_for_obs(self, flat_obs: np.ndarray):
        """
        When obs_dim not provided, initialize model to match flattened obs length.
        """
        if self.obs_dim is None:
            self.obs_dim = int(flat_obs.shape[0])
            self._build_model(self.obs_dim, self.hidden_sizes)
            # re-create optimizer for new params
            lr = self.optimizer.defaults.get("lr", 1e-4)
            self.optimizer = th.optim.Adam(self.model.parameters(), lr=lr)

    @staticmethod
    def _flatten_any(obs):
        if isinstance(obs, np.ndarray) and obs.dtype != object:
            return obs.astype(np.float32).ravel()
        return flatten_obs_any(obs)

    # --------------------------------------------------------------------- #
    #  Public API
    # --------------------------------------------------------------------- #

    def predict(self, obs, deterministic: bool = False) -> int:
        """
        Return chosen subpolicy index (int).
        Accepts dict, list/tuple, or numpy obs.
        """
        flat = self._flatten_any(obs)
        self._ensure_model_for_obs(flat)

        obs_t = th.from_numpy(flat).unsqueeze(0)  # (1, D)

        with th.no_grad():
            logits = self.model(obs_t) / max(self.temperature, EPS)
            probs = F.softmax(logits, dim=-1).squeeze(0)  # (num_networks,)

        if deterministic:
            action = int(th.argmax(probs).item())
        else:
            # ε-greedy on top of softmax for robust exploration
            progress = min(1.0, self.total_actions / max(1.0, self.eps_decay_steps))
            eps = self.eps_start - progress * (self.eps_start - self.eps_end)

            if np.random.rand() < eps:
                action = int(np.random.randint(self.num_networks))
            else:
                dist = Categorical(probs)
                action = int(dist.sample().item())

        if 0 <= action < self.num_networks:
            self.action_counts[action] += 1
            self.total_actions += 1

        return action

    def decay_temperature(self, factor=0.996, min_temp=0.8):
        """
        Optional: call periodically from training loop if you want annealing.
        """
        self.temperature = max(min_temp, self.temperature * float(factor))

    def get_probabilities(self, obs):
        """
        Return current action probabilities as numpy array (num_networks,).
        """
        flat = self._flatten_any(obs)
        self._ensure_model_for_obs(flat)
        with th.no_grad():
            obs_t = th.from_numpy(flat).unsqueeze(0)
            logits = self.model(obs_t) / max(self.temperature, EPS)
            probs = F.softmax(logits, dim=-1).squeeze(0)
        return probs.cpu().numpy()

    def store(
        self,
        obs,
        selected_action_idx,
        subpolicy_idx,
        return_=0.0,
        steps_spent=1,
        success=False,
        switched=False,
        grasp_from_far=False,
    ):
        """
        Store one segment:
          - obs: observation at the time decider chose this subpolicy
          - selected_action_idx: index decider actually chose (usually == subpolicy_idx)
          - subpolicy_idx: which subpolicy executed this segment
          - return_: cumulative reward collected while this subpolicy was active
          - steps_spent: number of env steps in this segment
          - success: 1 if segment ended in success
          - switched: 1 if this segment ended because decider switched to another subpolicy
          - grasp_from_far: 1 if grasp was triggered from too far, to penalize
        """
        flat = self._flatten_any(obs)
        self._ensure_model_for_obs(flat)

        self.buffer.append(
            {
                "obs": flat.astype(np.float32),
                "action": int(selected_action_idx),
                "subidx": int(subpolicy_idx),
                "return": float(return_),
                "steps": int(steps_spent),
                "success": int(bool(success)),
                "switched": int(bool(switched)),
                "grasp_from_far": int(bool(grasp_from_far)),
            }
        )

    def update(self):
        """
        Sample minibatch from buffer and perform one gradient step.
        Returns metrics dict or None if not enough data.
        """
        if len(self.buffer) < max(self.min_batch_for_update, 4):
            return None

        batch_size = min(self.batch_size, len(self.buffer))
        batch = random.sample(list(self.buffer), batch_size)

        obs_batch = th.tensor(np.stack([b["obs"] for b in batch]), dtype=th.float32)  # (B, D)
        action_batch = th.tensor([b["action"] for b in batch], dtype=th.long)         # (B,)
        subidx_batch = th.tensor([b["subidx"] for b in batch], dtype=th.long)         # (B,)
        return_batch = th.tensor([b["return"] for b in batch], dtype=th.float32)      # (B,)
        steps_batch = th.tensor([b["steps"] for b in batch], dtype=th.float32)        # (B,)
        success_batch = th.tensor([b["success"] for b in batch], dtype=th.float32)    # (B,)
        switched_batch = th.tensor([b["switched"] for b in batch], dtype=th.float32)  # (B,)
        grasp_far_batch = th.tensor(
            [b["grasp_from_far"] for b in batch], dtype=th.float32
        )                                                                             # (B,)

        # reward metric: per-step by default
        if self.normalize_reward_by_time:
            metric = return_batch / (steps_batch + EPS)
        else:
            metric = return_batch

        # ----------------- update running baselines per subpolicy ----------------- #
        with th.no_grad():
            for i in range(self.num_networks):
                mask = (subidx_batch == i)
                if mask.any():
                    vals = metric[mask]
                    mean_i = vals.mean()
                    std_i = vals.std(unbiased=False)
                    std_i = th.clamp(std_i, min=1e-3)

                    self.running_baselines[i] = (
                        self.baseline_beta * self.running_baselines[i]
                        + (1.0 - self.baseline_beta) * mean_i
                    )
                    self.running_stds[i] = (
                        self.baseline_beta * self.running_stds[i]
                        + (1.0 - self.baseline_beta) * std_i
                    )

        baselines = self.running_baselines[subidx_batch]
        stds = self.running_stds[subidx_batch]

        # advantage-like z-score
        z_scores = (metric - baselines) / (stds + 1e-3)

        # decider reward
        decider_rewards = (
            z_scores
            + self.success_bonus * success_batch
            - self.time_penalty * steps_batch
            - self.switch_penalty * switched_batch
            - self.grasp_far_penalty * grasp_far_batch
        )
        decider_rewards = th.clamp(decider_rewards, -5.0, 5.0)

        # ---------------------------- policy loss --------------------------------- #
        logits = self.model(obs_batch) / max(self.temperature, EPS)
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs[th.arange(len(action_batch)), action_batch]

        policy_loss = -(decider_rewards.detach() * selected_log_probs).mean()

        # entropy regularization
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=1).mean()
        entropy_loss = -self.entropy_coef * entropy

        total_loss = policy_loss + entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        th.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.steps += 1

        # normalized action frequencies
        if self.total_actions > 0:
            action_freqs = (self.action_counts / max(1, self.total_actions)).astype(np.float32)
        else:
            action_freqs = np.zeros(self.num_networks, dtype=np.float32)

        # build stats dict
        stats = {
            "batch_size": int(batch_size),
            "mean_metric": float(metric.mean().item()),  # your per-step or total metric
            "loss": float(total_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "entropy": float(entropy.item()),
            "baseline_means": self.running_baselines.detach().cpu().numpy().tolist(),
            "action_freqs": action_freqs.tolist(),
        }

        # log to file (if enabled)
        self._log_stats(stats)
        return stats

    def set_log_path(self, log_path: str) -> None:
        """
        Set or change the decider log path. If None, logging is disabled.
        """
        self.log_path = log_path

    def _log_stats(self, stats: dict) -> None:
        """
        Append one line with stats to log file
        Format: TSV with a few key fields + JSON for the complex ones
        """
        if not self.log_path:
            return

        line = {
            "global_step": getattr(self, "global_step", 0),
            "update_idx": self.update_counter,
            "batch_size": stats.get("batch_size", 0),
            "mean_return": stats.get("mean_return", 0.0),
            "loss": stats.get("loss", 0.0),
            "entropy": stats.get("entropy", 0.0),
            "mean_advantage": stats.get("mean_advantage", 0.0),
            "std_advantage": stats.get("std_advantage", 0.0),
            # complex ones we dump as JSON strings
            "action_counts": stats.get("action_counts", None),
            "baseline_means": stats.get("baseline_means", None),
        }

        # Serialize dict into a TSV line: key1=value1<TAB>key2=value2...
        parts = []
        for k, v in line.items():
            if isinstance(v, (list, dict)):
                v_str = json.dumps(v)
            else:
                v_str = str(v)
            parts.append(f"{k}={v_str}")
        text_line = "\t".join(parts) + "\n"

        # Append to file
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(text_line)
