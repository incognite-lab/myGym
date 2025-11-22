import random
from collections import deque
import numpy as np
import torch as th
import torch.nn.functional as F
from torch.distributions import Categorical

EPS = 1e-8

def flatten_obs_any(obs):
    """
    Recursively flatten an arbitrary nested myGym observation into 1D float32 numpy array
    Deterministic ordering: dict keys sorted alphabetically
    """
    if obs is None:
        return np.zeros(0, dtype=np.float32)

    if isinstance(obs, np.ndarray):
        if obs.dtype == object:
            parts = []
            for x in obs:
                parts.append(flatten_obs_any(x))
            if len(parts) == 0:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(parts).astype(np.float32)
        return obs.astype(np.float32).ravel()

    if isinstance(obs, dict):
        parts = []
        for k in sorted(obs.keys()):
            parts.append(flatten_obs_any(obs[k]))
        if len(parts) == 0:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(parts).astype(np.float32)

    if isinstance(obs, (list, tuple)):
        parts = []
        for x in obs:
            parts.append(flatten_obs_any(x))
        if len(parts) == 0:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(parts).astype(np.float32)

    # scalar
    try:
        return np.array([obs], dtype=np.float32)
    except Exception as e:
        raise TypeError(f"Unsupported observation element: {obs} ({type(obs)})") from e

class DeciderPolicy(th.nn.Module):
    """
    Decider meta-controller (REINFORCE-style) with per-network baselines and robust observation flattening
    API compatibility: predict(obs[, deterministic]), store(...), update(), get_probabilities(obs)
    """
    def __init__(self,
                 obs_dim: int = None, # if None, auto-detect on first predict/store
                 num_networks: int = 2,
                 hidden_sizes=(128,64),
                 lr=3e-4,
                 buffer_size=4096,
                 batch_size=128,
                 gamma=0.99,
                 baseline_beta=0.995,
                 success_bonus=5.0,
                 time_penalty=0.002,
                 switch_penalty=0.05,
                 entropy_coef=0.01,
                 min_batch_for_update=32,
                 max_grad_norm=5.0,
                 temperature=1.0):
        super().__init__()
        self._obs_dim_given = obs_dim
        self.obs_dim = obs_dim
        self.num_networks = int(num_networks)
        self.hidden_sizes = hidden_sizes

        # create a placeholder network, will re-init when obs_dim known
        self._build_model(obs_dim if obs_dim is not None else 1, hidden_sizes)

        self.optimizer = th.optim.Adam(self.model.parameters(), lr=lr)
        self.buffer = deque(maxlen=buffer_size)
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

        # per-network baseline (running mean), created now if obs_dim known, else on first update
        self.register_buffer("running_baselines", th.zeros(self.num_networks))
        # diagnostics
        self.steps = 0

    def _build_model(self, obs_dim, hidden_sizes):
        layers = []
        last = int(obs_dim)
        for h in hidden_sizes:
            layers.append(th.nn.Linear(last, int(h)))
            layers.append(th.nn.ReLU())
            last = int(h)
        layers.append(th.nn.Linear(last, self.num_networks))
        self.model = th.nn.Sequential(*layers)

    def _ensure_model_for_obs(self, flat_obs):
        """ When obs_dim not provided, initialize model to match flattened obs length"""
        if self.obs_dim is None:
            self.obs_dim = int(flat_obs.shape[0])
            # rebuild model with correct input dim
            self._build_model(self.obs_dim, self.hidden_sizes)
            # move optimizer to new params
            self.optimizer = th.optim.Adam(self.model.parameters(), lr=self.optimizer.defaults['lr'])

    def predict(self, obs, deterministic=False):
        """
        Return chosen subpolicy index (int)
        Accepts dict obs or flattened numpy array
        """
        if isinstance(obs, dict) or (isinstance(obs, (list, tuple, np.ndarray)) and getattr(obs, "dtype", None) == object):
            flat = flatten_obs_any(obs)
        elif isinstance(obs, np.ndarray):
            flat = obs.astype(np.float32).ravel()
        else:
            # fallback: try to build from scalar/other
            flat = flatten_obs_any(obs)

        self._ensure_model_for_obs(flat)
        obs_t = th.from_numpy(flat.astype(np.float32)).unsqueeze(0)  # (1, D)
        with th.no_grad():
            logits = self.model(obs_t) / max(self.temperature, EPS)   # (1, num)
            probs = F.softmax(logits, dim=-1).squeeze(0)              # (num,)
            if deterministic:
                return int(th.argmax(probs).item())
            return int(Categorical(probs).sample().item())

    def get_probabilities(self, obs):
        flat = flatten_obs_any(obs) if isinstance(obs, (dict, list, tuple)) else np.asarray(obs, dtype=np.float32).ravel()
        self._ensure_model_for_obs(flat)
        with th.no_grad():
            obs_t = th.from_numpy(flat.astype(np.float32)).unsqueeze(0)
            logits = self.model(obs_t) / max(self.temperature, EPS)
            return F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    def store(self, obs, selected_action_idx, subpolicy_idx, return_=0.0, steps_spent=1, success=False, switched=False):
        """
        store one sample that corresponds to a contiguous block where chosen subpolicy acted.
        return_ should be cumulative (optionally discounted) reward collected while the subpolicy was active.
        """
        if isinstance(obs, dict) or isinstance(obs, (list, tuple, np.ndarray)) and getattr(obs, "dtype", None) == object:
            flat = flatten_obs_any(obs)
        elif isinstance(obs, np.ndarray):
            flat = obs.astype(np.float32).ravel()
        else:
            flat = flatten_obs_any(obs)

        self._ensure_model_for_obs(flat)

        self.buffer.append({
            "obs": np.asarray(flat, dtype=np.float32),
            "action": int(selected_action_idx),
            "subidx": int(subpolicy_idx),
            "return": float(return_),
            "steps": int(steps_spent),
            "success": int(bool(success)),
            "switched": int(bool(switched))
        })

    def update(self):
        """
        Sample minibatch from buffer and perform one gradient step on the decider policy.
        Returns metrics dict or None.
        """
        if len(self.buffer) < max(self.min_batch_for_update, 4):
            return None

        batch_size = min(self.batch_size, len(self.buffer))
        batch = random.sample(list(self.buffer), batch_size)

        obs_batch = th.tensor(np.stack([b["obs"] for b in batch]), dtype=th.float32)   # (B, D)
        action_batch = th.tensor([b["action"] for b in batch], dtype=th.long)          # (B,)
        subidx_batch = np.array([b["subidx"] for b in batch], dtype=np.int64)          # (B,)
        return_batch = th.tensor([b["return"] for b in batch], dtype=th.float32)       # (B,)
        steps_batch = th.tensor([b["steps"] for b in batch], dtype=th.float32)
        success_batch = th.tensor([b["success"] for b in batch], dtype=th.float32)
        switched_batch = th.tensor([b["switched"] for b in batch], dtype=th.float32)

        # compute per-sample advantage = (return - baseline[subpolicy])
        # baselines: pick baseline values per-sample from running_baselines buffer
        baselines = th.stack([self.running_baselines[i] for i in subidx_batch]).to(return_batch.device)
        advantages = return_batch - baselines

        # update running baselines per network using EMA of observed returns in this batch
        for net_idx in range(self.num_networks):
            mask = (subidx_batch == net_idx)
            if mask.sum() > 0:
                mean_ret = return_batch[mask].mean().detach()
                self.running_baselines[net_idx] = self.baseline_beta * self.running_baselines[net_idx] + (1.0 - self.baseline_beta) * mean_ret

        # compose decider reward = advantage (normalized) + success_bonus - time_penalty*steps - switch_penalty*switched
        adv_mean = advantages.mean()
        adv_std = advantages.std(unbiased=False) + EPS
        norm_adv = (advantages - adv_mean) / adv_std

        decider_rewards = norm_adv + self.success_bonus * success_batch - self.time_penalty * steps_batch - self.switch_penalty * switched_batch

        # policy loss (REINFORCE using decider_rewards)
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
        # gradient clipping for stability
        th.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.steps += 1

        return {
            "loss": float(total_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "entropy": float(entropy.item()),
            "baseline_means": self.running_baselines.clone().cpu().numpy().tolist()
        }
