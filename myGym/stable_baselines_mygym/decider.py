import json
import os
import random
from collections import deque

import numpy as np
import torch as th
import torch.nn.functional as F
from torch.distributions import Categorical

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
      - store(obs, selected_action_idx, subpolicy_idx, return_, steps_spent, success, switched)
      - update() -> metrics dict or None
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
            replay_window=512,
            baseline_beta=0.98,
            success_bonus=0.5,
            switch_correct_bonus=0.5,
            entropy_coef=0.05,
            min_batch_for_update=32,
            max_grad_norm=1.0,
            temperature=1.0,
            switch_penalty = 0.1, # only for switching to wrong subpolicy
            log_path=None,
            _gt_labels_dropped=False
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.num_networks = int(num_networks)
        self.hidden_sizes = tuple(hidden_sizes)

        # initial tiny model; will rebuild when obs_dim known
        self._build_model(obs_dim if obs_dim is not None else 1, self.hidden_sizes)
        self.optimizer = th.optim.Adam(self.model.parameters(), lr=lr)

        # REINFORCE buffer with replay-style sampling
        self.buffer = deque(maxlen=int(buffer_size))
        self.replay_window = int(replay_window) if replay_window else 0
        if self.replay_window and self.replay_window > int(buffer_size):
            self.replay_window = int(buffer_size)
        self.batch_size = int(batch_size)
        self.baseline_beta = float(baseline_beta)
        self.success_bonus = float(success_bonus)
        self.switch_correct_bonus = float(switch_correct_bonus)
        self.entropy_coef = float(entropy_coef)
        self.min_batch_for_update = int(min_batch_for_update)
        self.max_grad_norm = float(max_grad_norm)
        self.temperature = float(temperature)
        self.switch_penalty = float(switch_penalty)
        self._gt_labels_dropped = bool(_gt_labels_dropped)
        # per-network running baseline and std
        self.register_buffer("running_baselines", th.zeros(self.num_networks))
        self.register_buffer("running_stds", th.ones(self.num_networks))

        # diagnostics
        self.steps = 0  # number of optimizer updates
        self.action_counts = np.zeros(self.num_networks, dtype=np.int64)
        self.total_actions = 0  # number of decider decisions

        # logging-related
        self.log_path = log_path  # path to .txt / .tsv file
        self.update_counter = 0  # how many times update() was called
        self.global_step = 0

    #  Internal helpers
    def _check_obs_dim(self, flat: np.ndarray):
        if self.obs_dim is not None and int(flat.shape[0]) != int(self.obs_dim):
            raise ValueError(
                f"Decider obs dim changed: got {flat.shape[0]}, expected {self.obs_dim}. "
                "Your observation structure/length is not stable."
            )

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
            device = self.running_baselines.device
            self.obs_dim = int(flat_obs.shape[0])
            self._build_model(self.obs_dim, self.hidden_sizes)
            self.model.to(device)
            # re-create optimizer for new params
            lr = self.optimizer.defaults.get("lr", 1e-4)
            self.optimizer = th.optim.Adam(self.model.parameters(), lr=lr)

    @staticmethod
    def _flatten_any(obs):
        if isinstance(obs, np.ndarray) and obs.dtype != object:
            return obs.astype(np.float32).ravel()
        return flatten_obs_any(obs)

    #  Public API
    def predict(self, obs, deterministic: bool = False, stage_idx=None, return_logp: bool = False) -> tuple[int, float]:
        """
        Return chosen subpolicy index (int).
        Accepts dict, list/tuple, or numpy obs.
        """
        if obs is None:
            raise ValueError("DeciderPolicy got obs=None. Pass the full observation dict.")

        obs = self._with_stage_feature(obs, stage_idx=stage_idx)

        flat = self._flatten_any(obs)
        self._ensure_model_for_obs(flat)
        self._check_obs_dim(flat)

        device = next(self.model.parameters()).device
        obs_t = th.from_numpy(flat).unsqueeze(0).to(device)  # (1, D)

        with th.no_grad():
            logits = self.model(obs_t) / max(self.temperature, EPS)
            probs = F.softmax(logits, dim=-1).squeeze(0)  # (num_networks,)

        if deterministic:
            action = int(th.argmax(probs).item())
        else:
            dist = Categorical(probs)
            action = int(dist.sample().item())

        logp = float(th.log(probs[action] + EPS).item())

        if 0 <= action < self.num_networks:
            self.action_counts[action] += 1
            self.total_actions += 1

        return (action, logp) if return_logp else action

    def decay_temperature(self, factor=0.996, min_temp=0.8):
        """
        Optional: call periodically from training loop if you want annealing.
        """
        self.temperature = max(min_temp, self.temperature * float(factor))

    def peek_action(self, obs, deterministic: bool = False, stage_idx=None) -> int:
        """
        Like predict(), but with NO side effects:
          - does NOT update action_counts / total_actions
          - does NOT store anything
          - does NOT backprop

        Intended for logging/debug (e.g., during warmup).
        """
        if obs is None:
            raise ValueError("DeciderPolicy got obs=None. Pass the full observation dict.")

        obs = self._with_stage_feature(obs, stage_idx=stage_idx)

        flat = self._flatten_any(obs)
        self._ensure_model_for_obs(flat)
        self._check_obs_dim(flat)

        device = next(self.model.parameters()).device
        with th.no_grad():
            obs_t = th.from_numpy(flat).unsqueeze(0).to(device)  # (1, D)
            logits = self.model(obs_t) / max(self.temperature, EPS)
            probs = F.softmax(logits, dim=-1).squeeze(0)  # (num_networks,)

        if deterministic:
            return int(th.argmax(probs).item())

        dist = Categorical(probs)
        return int(dist.sample().item())

    def store(
            self,
            obs,
            selected_action_idx,
            subpolicy_idx,
            return_=0.0,
            steps_spent=1,
            success=False,
            switch_correct=False,
            switched=False,
            gt_action=None,
            stage_idx=None,
            logp_old=None,
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
        """
        if obs is None:
            raise ValueError("DeciderPolicy got obs=None. Pass the full observation dict.")

        obs = self._with_stage_feature(obs, stage_idx=stage_idx)

        flat = self._flatten_any(obs)
        self._ensure_model_for_obs(flat)
        self._check_obs_dim(flat)

        selected_action_idx = int(selected_action_idx)
        subpolicy_idx = int(subpolicy_idx)

        override = int(selected_action_idx != subpolicy_idx)

        self.buffer.append(
            {
                "obs": flat.astype(np.float32),
                "action": int(selected_action_idx),
                "chosen_action": int(subpolicy_idx),
                "override": override,
                "return": float(return_),
                "steps": int(steps_spent),
                "success": int(bool(success)),
                "switch_correct": int(bool(switch_correct)),
                "switched": int(bool(switched)),
                "gt_action": None if gt_action is None else int(gt_action),
                "logp_old": None if logp_old is None else float(logp_old),
            }
        )

    def update(self):
        """
        Sample minibatch from buffer and perform one gradient step.
        Returns metrics dict or None if not enough data.
        """
        buf = list(self.buffer)
        if self.replay_window and len(buf) > self.replay_window:
            buf = buf[-self.replay_window:]

        candidates = [b for b in buf if b.get("override", 0) == 0]
        if len(candidates) < max(self.min_batch_for_update, 4):
            return None

        batch_size = min(self.batch_size, len(candidates))
        batch = random.sample(candidates, batch_size)

        device = next(self.model.parameters()).device

        obs_all = th.tensor(np.stack([b["obs"] for b in batch]), dtype=th.float32, device=device)
        logits_all = self.model(obs_all) / max(self.temperature, EPS)

        gt_idx = [i for i, b in enumerate(batch) if b.get("gt_action") is not None]
        ce_loss = th.zeros((), device=device)

        if gt_idx:
            gt_actions = th.tensor([batch[i]["gt_action"] for i in gt_idx], dtype=th.long, device=device)
            counts = th.bincount(gt_actions, minlength=self.num_networks).float()
            weights = counts.sum() / (counts + 1e-6)
            weights = th.where(counts > 0, weights, th.ones_like(weights))

            present = (counts > 0).float()
            denom = (weights * present).sum() / present.sum().clamp(min=1.0)
            weights = weights / denom

            weights = th.clamp(weights, max=10.0)  # safety clamp
            ce_loss = F.cross_entropy(logits_all[gt_idx], gt_actions, weight=weights)

        action_batch = th.tensor([b["action"] for b in batch], dtype=th.long, device=device)  # (B,)
        return_batch = th.tensor([b["return"] for b in batch], dtype=th.float32, device=device)  # (B,)
        steps_batch = th.tensor([max(1, b.get("steps", 1)) for b in batch], dtype=th.float32, device=device)
        success_batch = th.tensor([b["success"] for b in batch], dtype=th.float32, device=device)  # (B,)
        switch_correct_batch = th.tensor([b.get("switch_correct", 0) for b in batch], dtype=th.float32, device=device)

        metric = return_batch / steps_batch

        baselines_old = self.running_baselines.detach().clone()
        stds_old = self.running_stds.detach().clone()

        baselines = baselines_old[action_batch]
        stds = stds_old[action_batch]

        # advantage-like z-score (using OLD stats)
        z_scores = (metric - baselines) / (stds + 1e-3)

        # decider reward
        decider_rewards = (
                z_scores
                + self.success_bonus * success_batch  # successful segment in the right stage
                + self.switch_correct_bonus * switch_correct_batch # switched to the expected subpolicy
                - self.switch_penalty * (1.0 - switch_correct_batch)# switched to wrong policy
        )

        decider_rewards = th.clamp(decider_rewards, -5.0, 5.0)

        # ---------------------------- policy loss ---------------------------- #
        log_probs = F.log_softmax(logits_all, dim=-1)
        logp_new = log_probs[th.arange(len(action_batch)), action_batch]

        logp_old = th.tensor(
            [float("nan") if (b.get("logp_old") is None) else float(b["logp_old"]) for b in batch],
            dtype=th.float32,
            device=device,
        )

        missing = ~th.isfinite(logp_old)
        logp_old = th.where(missing, logp_new.detach(), logp_old)

        ratio = th.exp(logp_new - logp_old)
        ratio = th.clamp(ratio, 0.0, 10.0)  # safety

        clip_eps = 0.2
        ratio_clip = th.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)

        adv = decider_rewards.detach()
        sur = th.minimum(ratio * adv, ratio_clip * adv)

        # Do NOT apply policy-gradient/entropy to GT-labeled samples (off-policy)
        rl_mask = th.tensor(
            [1.0 if b.get("gt_action") is None else 0.0 for b in batch],
            dtype=th.float32,
            device=device,
        )
        rl_mask = rl_mask * (~missing).float()
        rl_denom = rl_mask.sum().clamp(min=1.0)

        policy_loss = -((sur * rl_mask).sum() / rl_denom)

        # entropy regularization
        probs = F.softmax(logits_all, dim=-1)
        entropy_per = -(probs * log_probs).sum(dim=1)
        entropy = (entropy_per * rl_mask).sum() / rl_denom
        entropy_loss = -self.entropy_coef * entropy

        bc_coef = max(0.0, 1.0 * (0.999 ** self.steps)) if not self._gt_labels_dropped else 0.0
        total_loss = policy_loss + entropy_loss + bc_coef * ce_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        th.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.steps += 1

        # ----------------- update running baselines per subpolicy ----------------- #
        with th.no_grad():
            for i in range(self.num_networks):
                mask = (action_batch == i)
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

        # normalized action frequencies
        if self.total_actions > 0:
            action_freqs = (self.action_counts / max(1, self.total_actions)).astype(np.float32)
        else:
            action_freqs = np.zeros(self.num_networks, dtype=np.float32)

        # build stats dict
        stats = {
            "batch_size": len(batch),
            "mean_metric": float(metric.mean().item()),  # your per-step or total metric
            "loss": float(total_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "entropy": float(entropy.item()),
            "mean_advantage": float(decider_rewards.mean().item()),
            "std_advantage": float(decider_rewards.std(unbiased=False).item()),
            "action_counts": self.action_counts.tolist(),
            "baseline_means": self.running_baselines.detach().cpu().numpy().tolist(),
            "action_freqs": action_freqs.tolist(),
        }

        # log to file (if enabled)
        self._log_stats(stats)
        self.update_counter += 1
        return stats

    def drop_gt_labels(self):
        self.buffer = deque([b for b in self.buffer if b.get("gt_action") is None],
                            maxlen=self.buffer.maxlen)
        self._gt_labels_dropped = True

    def _with_stage_feature(self, obs, stage_idx=None):
        """
        Ensure a small "stage" feature exists in the obs:
          _decider_stage_oh: one-hot of length (num_networks + 1)
            - index [0..num_networks-1] = stage
            - last index = unknown/start
        If stage_idx is None, we mark unknown.
        """
        if stage_idx is None and isinstance(obs, dict) and "_decider_stage_oh" in obs:
            return obs
        oh = np.zeros(self.num_networks + 1, dtype=np.float32)
        if stage_idx is None:
            oh[-1] = 1.0
        else:
            si = int(stage_idx)
            if 0 <= si < self.num_networks:
                oh[si] = 1.0
            else:
                oh[-1] = 1.0

        if isinstance(obs, dict):
            out = dict(obs)
            out["_decider_stage_oh"] = oh
            return out

        return {"obs": obs, "_decider_stage_oh": oh}

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
            "mean_metric": stats.get("mean_metric", 0.0),
            "loss": stats.get("loss", 0.0),
            "entropy": stats.get("entropy", 0.0),
            "mean_advantage": stats.get("mean_advantage", 0.0),
            "std_advantage": stats.get("std_advantage", 0.0),
            # complex ones dumped as JSON strings
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
        dirpath = os.path.dirname(self.log_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(text_line)
