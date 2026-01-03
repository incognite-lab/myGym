import copy
import os
import time

import matplotlib.pyplot as plt
import numpy as np

GREEN = [0, 125, 0]
RED = [125, 0, 0]

INITIAL_LOCK = 64
MIN_LOCK = 8
LOCK_DECAY_STEPS = 1_000_000  # after this many global steps, MIN_LOCK is reached
WARMUP_STEPS = 500_000  # cap for warmup


class Reward:
    """
    Reward base class for reward signal calculation and visualization

    Parameters:
        :param env: (object) Environment, where the training takes place
        :param task: (object) Task that is being trained, instance of a class TaskModule
    """

    def __init__(self, env, task=None):
        self.env = env
        self.task = task
        self.rewards_history = []

        if getattr(self.env, "network_switcher", "gt") == "decider":
            self.current_network = None
        else:
            self.current_network = 0

        self.num_networks = env.num_networks
        self.global_step = 0
        self.network_rewards = [0] * self.num_networks

        self._last_decider_choice = None
        self._current_segment_logp_old = None
        # GT-warmup -> Decider handoff (coverage by successful segments)
        self._decider_warmup_done = False
        self._decider_warmup_success_counts = np.zeros(self.num_networks, dtype=np.int32)

    def network_switch_control(self, observation):
        self.global_step += 1

        # Helper: encode "unknown stage" as -1 (so _decider_obs can map it to the "unknown" slot)
        def _stage_for_obs(stage_idx):
            return int(stage_idx) if stage_idx is not None else -1

        # Initialize segment-start obs (must be decider-formatted, not raw env obs)
        if getattr(self, "_current_decider_obs", None) is None:
            stage0 = _stage_for_obs(self.current_network)
            self._current_decider_obs = copy.deepcopy(self._decider_obs(observation, stage0))
            self._current_decider_stage_idx = stage0
            self._current_segment_logp_old = None

        if getattr(self.env, "num_networks", 1) <= 1:
            self.current_network = 0
            self._last_decider_choice = 0
            return self.current_network

        # Always compute decider prediction (even during warmup) for logging/debug
        if getattr(self.env, "network_switcher", "gt") == "decider":
            decider = getattr(self, "decider_model", None)
            if decider is not None:
                try:
                    stage_dbg = _stage_for_obs(self.current_network)
                    obs_d = self._decider_obs(observation, stage_dbg)
                    self._last_decider_choice = int(decider.peek_action(obs_d, deterministic=True, stage_idx=stage_dbg))
                except Exception:
                    self._last_decider_choice = int(self.current_network) if self.current_network is not None else 0

        # GT warmup until each subpolicy produced >=5 successful segment (cap by WARMUP_STEPS)
        if self.env.network_switcher == "decider" and not getattr(self, "_decider_warmup_done", False):
            need_coverage = np.any(self._decider_warmup_success_counts < 5) and (self.global_step < WARMUP_STEPS)
            if need_coverage:
                old_idx = int(self.current_network) if self.current_network is not None else 0

                # Use GT logic for switching, but keep env.network_switcher="decider" outside this call
                orig = self.env.network_switcher
                try:
                    self.env.network_switcher = "gt"
                    self.current_network = old_idx  # ensure GT logic starts from correct state
                    tmp = self.decide(observation)  # may mutate current_network internally
                    new_idx = old_idx if tmp is None else int(tmp)
                finally:
                    self.env.network_switcher = orig
                    self.current_network = old_idx

                # If GT caused a switch, close the previous segment so it gets recorded/counts
                if new_idx != old_idx and hasattr(self, "on_subpolicy_end"):
                    allowed, expected = self._allowed_actions_from_state(observation)
                    switch_correct = float(new_idx in allowed)

                    if getattr(self, "_current_subpolicy_steps", 0) > 0:
                        self.on_subpolicy_end(switched=True)

                    # New segment starts here: store segment-start obs with "stage = old_idx"
                    self._current_decider_obs = copy.deepcopy(self._decider_obs(observation, old_idx))
                    self._current_decider_stage_idx = old_idx
                    self._current_segment_logp_old = None
                    self._current_switch_correct = float(switch_correct)

                self.current_network = new_idx
                return int(self.current_network)

            # Coverage achieved OR cap reached -> allow decider logic from now on
            if getattr(self, "_current_subpolicy_steps", 0) > 0:
                self.on_subpolicy_end(switched=True)

            self._decider_warmup_done = True
            self.current_network = None
            self.decider_lock_until_step = 0

            # segment-start obs for first decider pick
            self._current_decider_obs = copy.deepcopy(self._decider_obs(observation, 0))
            self._current_decider_stage_idx = 0
            print(f"Decider warmup is finished at step {self.global_step}")

        if self.env.network_switcher == "gt":
            # ground-truth switching: use internal rules
            self.current_network = self.decide(observation)

        elif self.env.network_switcher == "decider":
            decider = getattr(self, "decider_model", None)
            if decider is None:
                print("NO DECIDER (NETWORK SWITCH CONTROL)")
                self.current_network = self.decide(observation)
                if self.current_network is None:
                    self.current_network = 0
            else:
                step = getattr(self.env, "episode_steps", 0)

                progress = min(1.0, self.global_step / LOCK_DECAY_STEPS)
                lock_len = max(MIN_LOCK, int(INITIAL_LOCK - progress * (INITIAL_LOCK - MIN_LOCK)))

                # first-time selection
                if self.current_network is None:
                    obs_d = self._decider_obs(observation, 0)
                    new_idx, logp = decider.predict(obs_d, return_logp=True,
                                                    stage_idx=0)
                    self._current_decider_stage_idx = 0
                    self._progress_stage = 0
                    allowed, expected = self._allowed_actions_from_state(observation)
                    self._current_switch_correct = float(new_idx in allowed)

                    self.current_network = new_idx
                    self._last_decider_choice = new_idx
                    self.decider_lock_until_step = step + lock_len

                    self._current_decider_obs = copy.deepcopy(obs_d)
                    self._current_decider_stage_idx = 0
                    self._current_segment_logp_old = float(logp)

                    print(f"[Reward/Decider] first lock-in: net={new_idx} until step {self.decider_lock_until_step}")

                # allowed to reconsider after lock expires
                elif step >= getattr(self, "decider_lock_until_step", 0):
                    old_idx = int(self.current_network)
                    # stage = old_idx (the policy we are currently in, before switching)
                    obs_d = self._decider_obs(observation, old_idx)
                    new_idx, logp = decider.predict(obs_d, return_logp=True, stage_idx=old_idx)
                    new_idx = int(new_idx)

                    allowed, expected = self._allowed_actions_from_state(observation)
                    switch_correct = float(new_idx in allowed)

                    self._last_decider_choice = new_idx

                    self.on_subpolicy_end(switched=(new_idx != old_idx))

                    # New segment starts here: store segment-start obs with stage=old_idx
                    self._current_decider_obs = copy.deepcopy(obs_d)
                    self._current_decider_stage_idx = old_idx
                    self._current_segment_logp_old = float(logp)
                    self._current_switch_correct = float(switch_correct)

                    self.current_network = new_idx
                    self.decider_lock_until_step = step + lock_len
                    print(f"[Reward/Decider] lock-in: net={new_idx} until step {self.decider_lock_until_step}")
            try:
                self.env.p.addUserDebugText(
                    text=f"DeciderNetwork: {self.current_network}",
                    textPosition=[0.7, 0.55, 0.3],
                    textSize=1.2,
                    lifeTime=0.3,
                    textColorRGB=[1, 1, 0]
                )
            except Exception as e:
                print(f"failed due {e}")

        elif self.env.network_switcher == "cycle_fixed":
            step = int(getattr(self.env, "episode_steps", 0))

            max_ep = int(getattr(self.env, "max_episode_steps", 0) or 0)
            if max_ep <= 0:
                max_ep = 512

            if step == 0 or not hasattr(self, "cycle_budgets") or len(self.cycle_budgets) != self.num_networks:
                base = max_ep // max(1, self.num_networks)
                rem = max_ep % max(1, self.num_networks)
                self.cycle_budgets = [base + (1 if i < rem else 0) for i in range(self.num_networks)]
                self.current_network = 0
                self.cycle_next_switch_step = self.cycle_budgets[0]
                self.cycle_idx = 0
            while step >= self.cycle_next_switch_step and self.cycle_idx < self.num_networks - 1:
                self.cycle_idx += 1
                self.current_network = self.cycle_idx
                self.cycle_next_switch_step += self.cycle_budgets[self.cycle_idx]

        elif self.env.network_switcher == "keyboard":
            keypress = self.env.p.getKeyboardEvents()
            if 107 in keypress.keys() and keypress[107] == 1:  # K
                if self.current_network < self.num_networks - 1:
                    self.current_network += 1
            elif 106 in keypress.keys() and keypress[106] == 1:  # J
                if self.current_network > 0:
                    self.current_network -= 1
        else:
            raise NotImplementedError(
                "Currently only implemented ground truth ('gt'), decider and cycle_fixed network switchers")

        if self.current_network is None:
            self.current_network = 0
        if self._last_decider_choice is None:
            self._last_decider_choice = int(self.current_network)

        return int(self.current_network)

    def _decider_obs(self, observation, stage_idx):
        """
        Return an obs object safe to feed into Decider:
          adds _decider_stage_oh (one-hot length num_networks+1; last=unknown)
        Does NOT mutate the original observation.
        """
        if observation is None:
            return None

        oh = np.zeros(self.num_networks + 1, dtype=np.float32)
        if stage_idx is None:
            oh[-1] = 1.0
        else:
            si = int(stage_idx)
            if 0 <= si < self.num_networks:
                oh[si] = 1.0
            else:
                oh[-1] = 1.0

        if isinstance(observation, dict):
            out = dict(observation)  # shallow copy
            out["_decider_stage_oh"] = oh
            return out

        return {"obs": observation, "_decider_stage_oh": oh}

    def compute(self, observation=None):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def visualize_reward_over_steps(self):
        """
        Plot and save a graph of reward values assigned to individual steps during an episode. Call this method after the end of the episode.
        """
        save_dir = os.path.join(self.env.logdir, "rewards")
        os.makedirs(save_dir, exist_ok=True)
        if self.env.episode_steps > 0:
            plt.ylabel("reward")
            plt.gcf().set_size_inches(8, 6)
            plt.savefig(save_dir + "/reward_over_steps_episode{}.png".format(self.env.episode_number))
            plt.close()

    def visualize_reward_over_episodes(self):
        """
        Plot and save a graph of cumulative reward values assigned to individual episodes. Call this method to plot data from the current and all previous episodes.
        """
        save_dir = os.path.join(self.env.logdir, "rewards")
        os.makedirs(save_dir, exist_ok=True)
        if self.env.episode_number > 0:
            plt.ylabel("reward")
            plt.gcf().set_size_inches(8, 6)
            plt.savefig(save_dir + "/reward_over_episodes_episode{}.png".format(self.env.episode_number))
            plt.close()

    def get_magnetization_status(self):
        return self.env.robot.use_magnet


# PROTOREWARDS

class Protorewards(Reward):

    def reset(self):
        self.last_find_dist = None
        self.last_approach_dist = None
        self.last_grip_dist = None
        self.last_lift_dist = None
        self.last_move_dist = None
        self.last_place_dist = None
        self.last_rot_dist = None
        self.subgoaloffset_dist = None
        self.last_leave_dist = None
        self.prev_object_position = None
        self.was_near = False

        if getattr(self.env, "network_switcher", "gt") == "decider":
            # during GT warmup we must start from a valid index so GT predicates can advance
            if getattr(self, "_decider_warmup_done", False):
                self.current_network = None
            else:
                self.current_network = 0
        else:
            self.current_network = 0

        self.eval_network_rewards = self.network_rewards
        self.network_rewards = [0] * self.num_networks
        self.has_left = False
        self.last_traj_idx = 0
        self.last_traj_dist = 0
        self.offset = [0.3, 0.0, 0.0]
        self.offsetleft = [0.2, 0.0, -0.1]
        self.offsetright = [-0.2, 0.0, -0.1]
        self.offsetcenter = [0.0, 0.0, -0.1]
        self.grip_threshold = 0.1
        self.approached_threshold = 0.045
        self.withdraw_threshold = 0.3
        self.opengr_threshold = self.env.robot.opengr_threshold
        self.closegr_threshold = self.env.robot.closegr_threshold
        self.near_threshold = 0.07
        self.lift_threshold = 0.1
        self.above_offset = 0.02
        self.reward_name = None
        self.iter = 1

        # decider integration state
        self._progress_stage = 0           # expected next subpolicy to complete
        self._current_subpolicy_return = 0.0
        self._current_subpolicy_steps = 0
        self._current_subpolicy_success = False
        self.finished_segments = []  # list[(ret, steps, success, idx, switched)]
        self.decider_lock_until_step = 0  # env.episode_steps threshold for next possible switch
        self._current_decider_obs = None
        self._gt_log = []
        self._last_gt_log_step = -1
        self._last_decider_choice = None
        self._current_switch_correct = 0.0
        self._current_decider_stage_idx = None

        # cycle_fixed baseline state (per-episode)
        if getattr(self.env, "network_switcher", "gt") == "cycle_fixed":
            k = int(getattr(self.env, "cycle_fixed_steps", 0) or 0)
            if k <= 0:
                max_ep = int(getattr(self.env, "max_episode_steps", 0) or 0)
                if max_ep <= 0:
                    max_ep = 512
                k = max(1, max_ep // max(1, self.num_networks))

            self.current_network = 0
            self.cycle_next_switch_step = k

    def _gt_enabled(self) -> bool:
        return (self.env.network_switcher == "gt") or (
                self.env.network_switcher == "decider" and not getattr(self, "_decider_warmup_done", False)
        )

    def compute(self, observation=None):
        # inherit and define your sequence of protoactions here
        pass

    def decide(self, observation=None):
        # inherit and define subgoals checking and network switching here
        pass

    def show_every_n_iters(self, text, value, n):
        """
        Every n iterations (steps) show given text and value.
        """
        if self.iter % n == 0:
            print(text, value)
        self.iter += 1

    def disp_reward(self, reward, owner):
        """Display reward in green if it's positive or in red if it's negative"""
        if reward > 0:
            color = GREEN
        else:
            color = RED
        if self.network_rewards[owner] > 0:
            color_sum = GREEN
        else:
            color_sum = RED
        self.env.p.addUserDebugText(f"Reward: {reward}", [0.63, 0.8, 0.55], lifeTime=0.5, textColorRGB=color)
        self.env.p.addUserDebugText(f"Reward sum for network {owner}, : {self.network_rewards[owner]}",
                                    [0.65, 0.6, 0.7],
                                    lifeTime=0.5,
                                    textColorRGB=color_sum)

    def change_network_based_on_key(self):
        keypress = self.env.p.getKeyboardEvents()
        if 107 in keypress.keys() and keypress[107] == 1:  # K
            if self.current_network < self.num_networks - 1:
                self.current_network += 1
                time.sleep(0.1)
        elif 106 in keypress.keys() and keypress[106] == 1:  # J
            if self.current_network > 0:
                self.current_network -= 1
                time.sleep(0.1)

    def get_distance_error(self, observation):
        gripper = observation["additional_obs"]["endeff_xyz"]
        object = observation["actual_state"]
        goal = observation["goal_state"]
        object_goal_distance = self.task.calc_distance(object, goal)
        if self.current_network == 0:
            gripper_object_distance = self.task.calc_distance(gripper, object)
            final_distance = object_goal_distance + gripper_object_distance
        else:
            final_distance = object_goal_distance
        return final_distance

    def get_positions(self, observation):
        goal_position = np.array(observation["goal_state"], dtype=np.float32, copy=True)
        object_position = observation["actual_state"]
        gripper_position = observation["additional_obs"]["endeff_xyz"]
        gripper_states = self.env.robot.get_gjoints_states()

        if self.prev_object_position is None:
            self.prev_object_position = object_position
        if self.__class__.__name__ in ["AaGaM", "AaGaMaD", "AaGaMaDaW"]:
            goal_position[2] += self.above_offset
        return goal_position, object_position, gripper_position, gripper_states

    #### PROTOREWARDS DEFINITIONS  ####

    def approach_compute(self, gripper, object, gripper_states):
        self.env.robot.set_magnetization(False)
        dist = self.task.calc_distance(gripper[:3], object[:3])
        gripdist = sum(gripper_states)
        if self.last_approach_dist is None:
            self.last_approach_dist = dist
        if self.last_grip_dist is None:
            self.last_grip_dist = gripdist
        reward = (self.last_approach_dist - dist) + (gripdist - self.last_grip_dist) * 0.2
        self.last_approach_dist = dist
        self.last_grip_dist = gripdist
        self.network_rewards[self.current_network] += reward
        self.reward_name = "approach"
        return reward

    def grasp_compute(self, gripper, object, gripper_states):
        self.env.robot.set_magnetization(True)
        dist = self.task.calc_distance(gripper[:3], object[:3])
        gripdist = sum(gripper_states)
        if self.last_approach_dist is None:
            self.last_approach_dist = dist
        if self.last_grip_dist is None:
            self.last_grip_dist = gripdist
        reward = (self.last_approach_dist - dist) * 0.2 + (self.last_grip_dist - gripdist) * 10
        self.last_approach_dist = dist
        self.last_grip_dist = gripdist
        self.network_rewards[self.current_network] += reward
        self.reward_name = "grasp"
        return reward

    def move_compute(self, object, goal, gripper_states):
        self.env.robot.set_magnetization(False)
        object_XY = object[:3]
        goal_XY = goal[:3]
        gripdist = sum(gripper_states)
        dist = self.task.calc_distance(object_XY, goal_XY)
        if self.last_move_dist is None:
            self.last_move_dist = dist
        if self.last_grip_dist is None:
            self.last_grip_dist = gripdist
        reward = (self.last_move_dist - dist) + (self.last_grip_dist - gripdist) * 0.2
        self.last_move_dist = dist
        self.last_grip_dist = gripdist
        self.network_rewards[self.current_network] += reward
        self.reward_name = "move"
        return reward

    def drop_compute(self, gripper, object, gripper_states):
        self.env.robot.set_magnetization(True)
        dist = self.task.calc_distance(gripper[:3], object[:3])
        gripdist = sum(gripper_states)
        if self.last_approach_dist is None:
            self.last_approach_dist = dist
        if self.last_grip_dist is None:
            self.last_grip_dist = gripdist
        reward = (self.last_approach_dist - dist) * 0.2 + (gripdist - self.last_grip_dist) * 10.0
        self.last_approach_dist = dist
        self.last_grip_dist = gripdist
        self.network_rewards[self.current_network] += reward
        self.reward_name = "drop"
        return reward

    def withdraw_compute(self, gripper, object, gripper_states):
        self.env.robot.set_magnetization(False)
        dist = self.task.calc_distance(gripper[:3], object[:3])
        gripdist = sum(gripper_states)
        if self.last_approach_dist is None:
            self.last_approach_dist = dist
        if self.last_grip_dist is None:
            self.last_grip_dist = gripdist
        if dist >= self.withdraw_threshold:  # rewarding only up to distance which should finish the task
            reward = 0  # This is done so that robot doesn't learn to drop object out of goal and then get the biggest
            # reward by withdrawing without finishing the task
        else:
            reward = (dist - self.last_approach_dist) + (gripdist - self.last_grip_dist) * 0.2
        self.last_approach_dist = dist
        self.last_grip_dist = gripdist
        self.network_rewards[self.current_network] += reward
        self.reward_name = "withdraw"
        return reward

    def rotate_compute(self, object, goal, gripper_states):
        self.env.robot.set_magnetization(False)
        dist = self.task.calc_distance(object, goal)
        if self.last_place_dist is None:
            self.last_place_dist = dist
        gripdist = sum(gripper_states)
        gripper_rew = (self.last_grip_dist - gripdist) * 0.1
        reward = self.last_place_dist - dist
        rot = self.task.calc_rot_quat(object, goal)
        if self.last_rot_dist is None:
            self.last_rot_dist = rot
        rewardrot = self.last_rot_dist - rot
        reward = reward + rewardrot
        self.last_place_dist = dist
        self.last_rot_dist = rot
        self.network_rewards[self.current_network] += reward
        self.reward_name = "rotate"
        return reward

    def transform_compute(self, object, goal, trajectory, magnetization=True):
        """Calculate reward based on following a trajectory
        params: object: self-explanatory
                goal: self-explanatory
                trajectory: (np.array) 3D trajectory, lists of points x, y, z
                magnetization: (boolean) sets magnetization on or off
        Reward is calculated based on distance of object from goal and square distance of object from trajectory.
        That way, object tries to approach goal while trying to stay on trajectory path.
        """
        self.env.robot.set_magnetization(magnetization)
        dist_g = self.task.calc_distance(object, goal)
        if self.last_place_dist is None:
            self.last_place_dist = dist_g
        reward_g_dist = self.last_place_dist - dist_g  # distance from goal
        pos = object[:3]
        dist_t, self.last_traj_idx = self.task.trajectory_distance(trajectory, pos, self.last_traj_idx, 10)
        if self.last_traj_dist is None:
            self.last_traj_dist = dist_t
        reward_t_dist = self.last_traj_dist - dist_t  # distance from trajectory
        reward = reward_g_dist + 4 * reward_t_dist
        self.last_place_dist = dist_g
        self.last_traj_dist = dist_t
        self.network_rewards[self.current_network] += reward
        self.reward_name = "transform"
        return reward

    def follow_compute(self, object, goal, trajectory, magnetization=True):
        """Calculate reward based on following a trajectory
        params: object: self-explanatory
                goal: self-explanatory
                trajectory: (np.array) 3D trajectory, lists of points x, y, z
                magnetization: (boolean) sets magnetization on or off
        Reward is calculated based on distance of object from goal and square distance of object from trajectory.
        That way, object tries to approach goal while trying to stay on trajectory path.
        """
        self.env.robot.set_magnetization(magnetization)
        dist_g = self.task.calc_distance(object, goal)
        if self.last_place_dist is None:
            self.last_place_dist = dist_g
        reward_g_dist = self.last_place_dist - dist_g  # distance from goal
        pos = object[:3]
        dist_t, self.last_traj_idx = self.task.trajectory_distance(trajectory, pos, self.last_traj_idx, 10)
        if self.last_traj_dist is None:
            self.last_traj_dist = dist_t
        reward_t_dist = self.last_traj_dist - dist_t  # distance from trajectory
        reward = reward_g_dist + 4 * reward_t_dist
        self.last_place_dist = dist_g
        self.last_traj_dist = dist_t
        self.network_rewards[self.current_network] += reward
        self.reward_name = "transform"
        return reward

    # PREDICATES

    def gripper_approached_object(self, gripper, object):
        if self.task.calc_distance(gripper, object) <= self.approached_threshold:
            return True
        return False

    def gripper_withdraw_object(self, gripper, object):
        if self.task.calc_distance(gripper, object) >= self.withdraw_threshold:
            return True
        return False

    def gripper_opened(self, gripper_states):
        if sum(gripper_states) >= self.opengr_threshold:
            self.env.robot.release_object(self.env.env_objects["actual_state"])
            self.env.robot.set_magnetization(False)
            return True
        return False

    def gripper_closed(self, gripper_states):
        if sum(gripper_states) <= self.closegr_threshold:
            try:
                # self.env.robot.magnetize_object(self.env.env_objects["actual_state"])
                self.env.robot.set_magnetization(True)
                return True
            except:
                return True
        return False

    def object_near_goal(self, object, goal):
        goal_local = goal.copy()
        goal_local[2] += self.above_offset
        distance = self.task.calc_distance(goal_local, object)
        if distance < self.near_threshold:
            return True
        return False

    def _allowed_actions_from_state(self, observation):
        """
        State-based allowed action set for switching correctness.
        Designed for sequences starting with: approach (0), grasp (1), move/rotate/follow/transform (2).
        Returns (allowed_set, expected_idx) or (None, None) if not applicable.
        """
        # Extract state
        _, object_position, gripper_position, gripper_states = self.get_positions(observation)

        # Predicates (use existing thresholds set in reset())
        approached = bool(self.gripper_approached_object(gripper_position, object_position))
        grip_sum = float(np.sum(gripper_states))
        is_closed = grip_sum <= float(getattr(self, "closegr_threshold", 0.0))

        # State-based "what should we do now?"
        if not approached:
            return {0}, 0

        if not is_closed:
            # close but allow micro-approach corrections
            return {0, 1}, 1

        # For AGM: keep doing stage2 unless done
        allowed = {2}
        expected = 2

        return allowed, expected

    # when network switch or subtask end happens (crucial)
    def on_subpolicy_end(self, switched: bool = False):
        """
        Called whenever a subpolicy segment ends (either because we switched
        to another subpolicy or the episode ended).

        Accumulates a segment into finished_segments and also updates the
        last_subpolicy_* fields for backward compatibility.
        """
        idx = int(self.current_network) if self.current_network is not None else 0
        obs_copy = None if self._current_decider_obs is None else copy.deepcopy(self._current_decider_obs)

        stage_idx = getattr(self, "_current_decider_stage_idx", None)
        logp_old = getattr(self, "_current_segment_logp_old", None)

        # Progress-gated success: only count "success" when it advances the episode.
        ps = getattr(self, "_progress_stage", None)
        try:
            ps_i = int(ps) if ps is not None else None
        except Exception:
            ps_i = None

        order_correct = bool(self._current_subpolicy_success) and (ps_i is None or idx == ps_i)
        switch_correct = float(getattr(self, "_current_switch_correct", 0.0))

        # append full segment to list for Decider
        seg = (
            self._current_subpolicy_return,
            self._current_subpolicy_steps,
            order_correct,
            switch_correct,
            idx,
            switched,
            obs_copy,
            stage_idx,
            None if logp_old is None else float(logp_old),
        )

        print(
            f"[SEG END] idx={idx} switched={switched} "
            f"ret={self._current_subpolicy_return:.3f} "
            f"steps={self._current_subpolicy_steps} "
            f"order_correct={order_correct} "
            f"switch_correct={switch_correct} "
            f"start_dist={getattr(self, '_current_segment_start_dist', 0.0):.4f}"
        )

        if hasattr(self, "finished_segments"):
            self.finished_segments.append(seg)

        # warmup coverage accounting: count successful segments per policy
        if not getattr(self, "_decider_warmup_done", False):
            try:
                self._decider_warmup_success_counts[idx] += int(bool(self._current_subpolicy_success))
                print(f"SUCCESS COUNTS: {self._decider_warmup_success_counts}")
            except Exception:
                pass

        # Advance expected progress stage when the correct subpolicy succeeds.
        if ps_i is not None and order_correct and ps_i < self.num_networks:
            self._progress_stage = min(ps_i + 1, self.num_networks - 1)

        # reset accumulators for the next segment
        self._current_subpolicy_return = 0.0
        self._current_subpolicy_steps = 0
        self._current_subpolicy_success = False
        self._current_segment_start_dist = 0.0
        self._current_decider_obs = None
        self._current_decider_stage_idx = None
        self._current_segment_logp_old = None
        self._current_switch_correct = 0.0


# ATOMIC ACTIONS - Examples of 1-5 protorewards

class A(Protorewards):
    def __init__(self, env, task=None):
        super().__init__(env, task)
        self.network_names = ["approach"]

    def reset(self):
        super().reset()
        self.network_names = ["approach"]

    def compute(self, observation=None):
        goal_position, object_position, gripper_position, gripper_states = self.get_positions(observation)
        owner = self.decide(observation)
        target = [[object_position, goal_position, gripper_states]][owner]
        reward = [self.approach_compute][owner](*target)
        if self.env.episode_terminated:
            reward += 0.2  # Adding reward for successful finish of episode
        self.rewards_history.append(reward)
        self.rewards_num = 1
        return reward

    def decide(self, observation=None):
        goal_position, object_position, gripper_position, gripper_states = self.get_positions(observation)
        if self.gripper_approached_object(object_position, goal_position):
            if self.gripper_opened(gripper_states):
                self.task.check_goal()
        self.task.check_episode_steps()
        return self.current_network


class AaG(Protorewards):
    def __init__(self, env, task=None):
        super().__init__(env, task)
        self.network_names = ["approach", "grasp"]
        self._current_segment_start_dist = 0.0

    def reset(self):
        super().reset()
        self.network_names = ["approach", "grasp"]

    def compute(self, observation=None):
        goal_position, object_position, gripper_position, gripper_states = self.get_positions(observation)
        owner = self.decide(observation)

        if self._current_subpolicy_steps == 0:
            # distance between gripper and object at segment start
            g = np.asarray(gripper_position, dtype=np.float32)
            o = np.asarray(object_position, dtype=np.float32)
            dist = np.linalg.norm(g[:3] - o[:3])
            self._current_segment_start_dist = float(dist)

        target = [[object_position, goal_position, gripper_states], [object_position, goal_position, gripper_states]][
            owner]
        reward = [self.approach_compute, self.grasp_compute][owner](*target)

        # self.disp_reward(reward, owner)
        self.rewards_history.append(reward)
        self.rewards_num = 2

        # decider
        if self.env.network_switcher == "decider":
            self._current_subpolicy_steps += 1
            self._current_subpolicy_return += reward

            if self.current_network == 0:
                if self.gripper_approached_object(gripper_position, object_position):
                    if self.gripper_opened(gripper_states):
                        self._current_subpolicy_success = True
            elif self.current_network == 1:
                if self.gripper_approached_object(gripper_position, object_position):
                    if self.gripper_closed(gripper_states):
                        self._current_subpolicy_success = True

            if self.env.episode_terminated:
                if hasattr(self, "on_subpolicy_end"):
                    self.on_subpolicy_end(switched=False)
        return reward

    def decide(self, observation=None):
        goal_position, object_position, gripper_position, gripper_states = self.get_positions(observation)
        if self.env.network_switcher == "keyboard":
            self.change_network_based_on_key()
        elif self.env.network_switcher == "gt":
            if self.current_network == 0:
                if self.gripper_approached_object(gripper_position, object_position):
                    if self.gripper_opened(gripper_states):
                        self.current_network = 1
        if self.current_network == 1:
            if self.gripper_closed(gripper_states):
                self.task.check_goal()
        self.task.check_episode_steps()
        return self.current_network


class AaGaM(Protorewards):
    def __init__(self, env, task=None):
        super().__init__(env, task)
        self.network_names = ["approach", "grasp", "move"]
        self._current_segment_start_dist = 0.0

    def reset(self):
        super().reset()
        self.network_names = ["approach", "grasp", "move"]

    def compute(self, observation=None):
        goal_position, object_position, gripper_position, gripper_states = self.get_positions(observation)
        owner = self.decide(observation)

        if self.env.network_switcher == "decider":
            if self.current_network == 0:
                if self.gripper_approached_object(gripper_position, object_position):
                    if self.gripper_opened(gripper_states):
                        self._current_subpolicy_success = True

            elif self.current_network == 1:
                if self.gripper_approached_object(gripper_position, object_position):
                    if self.gripper_closed(gripper_states):
                        self._current_subpolicy_success = True

            elif self.current_network == 2:
                if self.object_near_goal(object_position, goal_position):
                    self._current_subpolicy_success = True

            # log GT vs Decider once per env step
            step = int(getattr(self.env, "episode_steps", 0))
            if step != int(getattr(self, "_last_gt_log_step", -1)):
                # hypothetical GT transition from *current_network* (does not mutate)
                is_open = (sum(gripper_states) >= self.opengr_threshold)
                is_closed = (sum(gripper_states) <= self.closegr_threshold)

                gt_choice = int(self.current_network if self.current_network is not None else 0)

                if gt_choice == 0:
                    if self.gripper_approached_object(gripper_position, object_position) and is_open:
                        gt_choice = 1
                elif gt_choice == 1:
                    if self.gripper_approached_object(gripper_position, object_position) and is_closed:
                        gt_choice = 2
                dec_choice = int(getattr(self, "_last_decider_choice",
                                         self.current_network if self.current_network is not None else 0))

                self._gt_log.append((step, gt_choice, dec_choice))
                self._last_gt_log_step = step

        if self._current_subpolicy_steps == 0:
            # distance between gripper and object at segment start
            g = np.asarray(gripper_position, dtype=np.float32)
            o = np.asarray(object_position, dtype=np.float32)
            dist = np.linalg.norm(g[:3] - o[:3])
            self._current_segment_start_dist = float(dist)

        target = \
            [[gripper_position, object_position, gripper_states], [gripper_position, object_position, gripper_states],
             [object_position, goal_position, gripper_states]][owner]
        reward = [self.approach_compute, self.grasp_compute, self.move_compute][owner](*target)

        self.rewards_history.append(reward)
        self.rewards_num = 3

        # decider
        if self.env.network_switcher == "decider":
            self._current_subpolicy_steps += 1
            self._current_subpolicy_return += reward

            if self._current_subpolicy_success:
                # next env step network_switch_control() will be allowed to switch
                self.decider_lock_until_step = int(getattr(self.env, "episode_steps", 0))

            if self.env.episode_terminated or getattr(self.env, "episode_truncated", False):
                os.makedirs(self.env.logdir, exist_ok=True)
                save_path = os.path.join(self.env.logdir, "gt_vs_decider.txt")
                with open(save_path, "a") as f:
                    for step, gt, dec in self._gt_log:
                        f.write(f"{self.env.episode_number},{step},{gt},{dec}\n")
                self._gt_log = []

                if hasattr(self, "on_subpolicy_end"):
                    self.on_subpolicy_end(switched=False)
        return reward

    def decide(self, observation=None):
        goal_position, object_position, gripper_position, gripper_states = self.get_positions(observation)

        if self.env.network_switcher == "keyboard":
            self.change_network_based_on_key()
        elif self.env.network_switcher == "gt":
        #elif self._gt_enabled():
            if self.current_network == 0:
                if self.gripper_approached_object(gripper_position, object_position):
                    if self.gripper_opened(gripper_states):
                        self.current_network = 1
            if self.current_network == 1:
                if self.gripper_approached_object(gripper_position, object_position):
                    if self.gripper_closed(gripper_states):
                        self.current_network = 2
        if self.current_network == 2:
            if self.object_near_goal(object_position, goal_position):
                self.task.check_goal()
        self.task.check_episode_steps()
        return self.current_network


class AaGaR(Protorewards):
    """
    Reward sequence: Approach -> Grasp -> Rotate
    Uses the 'rotate_compute' protoreward for the third step.
    """

    def __init__(self, env, task=None):
        super().__init__(env, task)
        self.network_names = ["approach", "grasp", "rotate"]
        self._current_segment_start_dist = 0.0

    def reset(self):
        super().reset()
        # Updated network names to include "rotate"
        self.network_names = ["approach", "grasp", "rotate"]

    def compute(self, observation=None):
        goal_position, object_position, gripper_position, gripper_states = self.get_positions(observation)
        owner = self.decide(observation)

        if self._current_subpolicy_steps == 0:
            # first step of the current subpolicy segment
            # distance between gripper and object at segment start
            g = np.asarray(gripper_position, dtype=np.float32)
            o = np.asarray(object_position, dtype=np.float32)
            dist = np.linalg.norm(g[:3] - o[:3])
            self._current_segment_start_dist = float(dist)

        # Define targets for each protoreward function
        # Note: rotate_compute needs object_position and goal_position
        target = [
            [gripper_position, object_position, gripper_states],  # approach
            [gripper_position, object_position, gripper_states],  # grasp
            [object_position, goal_position, gripper_states]  # rotate
        ][owner]

        # List of protoreward functions corresponding to network_names
        reward_func_list = [
            self.approach_compute,
            self.grasp_compute,
            self.rotate_compute  # Using rotate_compute here
        ]

        # Calculate reward using the function for the current network owner
        reward = reward_func_list[owner](*target)

        # Display, log, and update history
        # self.disp_reward(reward, owner)
        self.rewards_history.append(reward)
        self.rewards_num = 3  # Total number of networks is 3

        # decider
        if self.env.network_switcher == "decider":
            self._current_subpolicy_steps += 1
            self._current_subpolicy_return += reward

            if self.current_network == 0:
                if self.gripper_approached_object(gripper_position, object_position):
                    if self.gripper_opened(gripper_states):
                        self._current_subpolicy_success = True

            elif self.current_network == 1:
                if self.gripper_approached_object(gripper_position, object_position):
                    if self.gripper_closed(gripper_states):
                        self._current_subpolicy_success = True

            elif self.current_network == 2:
                if self.gripper_approached_object(gripper_position, object_position):
                    if self.gripper_closed(gripper_states):
                        if self.object_near_goal(object_position, goal_position):
                            self._current_subpolicy_success = True
        if self.env.episode_terminated:
            if hasattr(self, "on_subpolicy_end"):
                self.on_subpolicy_end(switched=False)
        return reward

    def decide(self, observation=None):
        goal_position, object_position, gripper_position, gripper_states = self.get_positions(observation)

        if self.env.network_switcher == "keyboard":
            self.change_network_based_on_key()
        elif self.env.network_switcher == "gt":
            # Ground truth logic for switching networks based on predicates
            if self.current_network == 0:  # approach
                # Switch to grasp if gripper is near object and open
                if self.gripper_approached_object(gripper_position, object_position) and self.gripper_opened(
                        gripper_states):
                    self.current_network = 1
            elif self.current_network == 1:  # grasp
                # Switch to rotate if gripper is near object and closed (grasped)
                if self.gripper_approached_object(gripper_position, object_position) and self.gripper_closed(
                        gripper_states):
                    self.current_network = 2
            # Network 2 (rotate) is the final stage

        # Check for task completion in the final network stage (rotate)
        if self.current_network == 2:
            # Check if the object is near the goal (rotation might also affect position)
            if self.object_near_goal(object_position, goal_position):
                # Additionally, you might want to check rotation alignment if applicable
                # if self.task.check_rotation_alignment(object_position, goal_position):
                self.task.check_goal()  # Check if the overall task goal is met

        # Check for episode step limits
        self.task.check_episode_steps()
        return self.current_network


class AaGaMaD(Protorewards):
    def __init__(self, env, task=None):
        super().__init__(env, task)
        self.network_names = ["approach", "grasp", "move", "drop"]
        self._current_segment_start_dist = 0.0

    def reset(self):
        super().reset()
        self.network_names = ["approach", "grasp", "move", "drop"]

    def compute(self, observation=None):
        goal_position, object_position, gripper_position, gripper_states = self.get_positions(observation)
        owner = self.decide(observation)

        if self._current_subpolicy_steps == 0:
            # first step of the current subpolicy segment
            # distance between gripper and object at segment start
            g = np.asarray(gripper_position, dtype=np.float32)
            o = np.asarray(object_position, dtype=np.float32)
            dist = np.linalg.norm(g[:3] - o[:3])
            self._current_segment_start_dist = float(dist)

        target = [[gripper_position, object_position, gripper_states],
                  [gripper_position, object_position, gripper_states],
                  [object_position, goal_position, gripper_states],
                  [gripper_position, goal_position, gripper_states]][owner]
        reward = [self.approach_compute, self.grasp_compute, self.move_compute, self.drop_compute][owner](*target)

        # if self.env.episode_terminated:
        #     reward += 0.2  # Adding reward for succesful finish of episode
        self.rewards_history.append(reward)
        self.rewards_num = 4

        # decider
        if self.env.network_switcher == "decider":
            self._current_subpolicy_steps += 1
            self._current_subpolicy_return += reward

            if self.current_network == 0:
                if self.gripper_approached_object(gripper_position, object_position):
                    if self.gripper_opened(gripper_states):
                        self._current_subpolicy_success = True

            elif self.current_network == 1:
                if self.gripper_approached_object(gripper_position, object_position):
                    if self.gripper_closed(gripper_states):
                        self._current_subpolicy_success = True

            elif self.current_network == 2:
                if self.gripper_approached_object(gripper_position, object_position):
                    if self.gripper_closed(gripper_states):
                        if self.object_near_goal(object_position, goal_position):
                            self._current_subpolicy_success = True
            elif self.current_network == 3:
                if self.object_near_goal(object_position, goal_position):
                    if self.gripper_opened(gripper_states):
                        self._current_subpolicy_success = True
        if self.env.episode_terminated:
            if hasattr(self, "on_subpolicy_end"):
                self.on_subpolicy_end(switched=False)
        return reward

    def decide(self, observation=None):
        goal_position, object_position, gripper_position, gripper_states = self.get_positions(observation)

        if self.env.network_switcher == "keyboard":
            self.change_network_based_on_key()
        elif self.env.network_switcher == "gt":
            if self.current_network == 0:
                if self.gripper_approached_object(gripper_position, object_position):
                    if self.gripper_opened(gripper_states):
                        self.current_network = 1
            if self.current_network == 1:
                if self.gripper_approached_object(gripper_position, object_position):
                    if self.gripper_closed(gripper_states):
                        self.current_network = 2
            if self.current_network == 2:
                if self.object_near_goal(object_position, goal_position):
                    self.current_network = 3
        if self.current_network == 3:
            if self.gripper_opened(gripper_states):
                self.task.check_goal()
        self.task.check_episode_steps()
        return self.current_network


class AaGaMaDaW(Protorewards):
    def __init__(self, env, task=None):
        super().__init__(env, task)
        self.network_names = ["approach", "grasp", "move", "drop", "withdraw"]
        self._current_segment_start_dist = 0.0

    def reset(self):
        super().reset()
        self.network_names = ["approach", "grasp", "move", "drop", "withdraw"]

    def compute(self, observation=None):
        goal_position, object_position, gripper_position, gripper_states = self.get_positions(observation)
        owner = getattr(self, "current_network", self.decide(observation))

        if self._current_subpolicy_steps == 0:
            # first step of the current subpolicy segment
            # distance between gripper and object at segment start
            g = np.asarray(gripper_position, dtype=np.float32)
            o = np.asarray(object_position, dtype=np.float32)
            dist = np.linalg.norm(g[:3] - o[:3])
            self._current_segment_start_dist = float(dist)

        target = [[gripper_position, object_position, gripper_states],
                  [gripper_position, object_position, gripper_states],
                  [object_position, goal_position, gripper_states],
                  [gripper_position, object_position, gripper_states],
                  [gripper_position, goal_position, gripper_states]][owner]

        reward = \
            [self.approach_compute, self.grasp_compute, self.move_compute, self.drop_compute, self.withdraw_compute][
                owner](
                *target)

        # if self.env.episode_terminated:
        #     reward += 1.0  # Adding reward for successful finish of episode

        self.rewards_history.append(reward)
        self.rewards_num = 5
        self.last_reward = reward

        # decider
        if self.env.network_switcher == "decider":
            self._current_subpolicy_steps += 1
            self._current_subpolicy_return += reward

            if self.current_network == 0:
                if self.gripper_approached_object(gripper_position, object_position):
                    if self.gripper_opened(gripper_states):
                        self._current_subpolicy_success = True

            elif self.current_network == 1:
                if self.gripper_approached_object(gripper_position, object_position):
                    if self.gripper_closed(gripper_states):
                        self._current_subpolicy_success = True

            elif self.current_network == 2:
                if self.gripper_approached_object(gripper_position, object_position):
                    if self.gripper_closed(gripper_states):
                        if self.object_near_goal(object_position, goal_position):
                            self._current_subpolicy_success = True
                            self.task.check_goal()
            elif self.current_network == 3:
                if self.object_near_goal(object_position, goal_position):
                    if self.gripper_opened(gripper_states):
                        self._current_subpolicy_success = True
            elif self.current_network == 4:
                if self.object_near_goal(object_position, goal_position):
                    if self.gripper_withdraw_object(gripper_position, object_position):
                        if self.gripper_opened(gripper_states):
                            self._current_subpolicy_success = True
        if self.env.episode_terminated:
            if hasattr(self, "on_subpolicy_end"):
                self.on_subpolicy_end(switched=False)
        return reward

    def decide(self, observation=None):
        goal_position, object_position, gripper_position, gripper_states = self.get_positions(observation)

        if self.env.network_switcher == "keyboard":
            self.change_network_based_on_key()
        elif self.env.network_switcher == "gt":
            if self.current_network == 0:
                if self.gripper_approached_object(gripper_position, object_position):
                    if self.gripper_opened(gripper_states):
                        self.current_network = 1
            if self.current_network == 1:
                if self.gripper_approached_object(gripper_position, object_position):
                    if self.gripper_closed(gripper_states):
                        self.current_network = 2
            if self.current_network == 2:
                if self.object_near_goal(object_position, goal_position):
                    self.current_network = 3
            if self.current_network == 3:
                if self.gripper_approached_object(gripper_position, object_position):
                    if self.gripper_opened(gripper_states):
                        self.current_network = 4
        if self.current_network == 4:
            if self.gripper_withdraw_object(gripper_position, object_position):
                if self.gripper_opened(gripper_states):
                    self.task.check_goal()

        self.task.check_episode_steps()
        return self.current_network


class AaGaRaDaW(Protorewards):
    def __init__(self, env, task=None):
        super().__init__(env, task)
        self.network_names = ["approach", "grasp", "rotate", "drop", "withdraw"]
        self._current_segment_start_dist = 0.0

    def reset(self):
        super().reset()
        # Updated network names to include "rotate" instead of "move"
        self.network_names = ["approach", "grasp", "rotate", "drop", "withdraw"]

    def compute(self, observation=None):
        goal_position, object_position, gripper_position, gripper_states = self.get_positions(observation)
        owner = self.decide(observation)

        if self._current_subpolicy_steps == 0:
            # first step of the current subpolicy segment
            # distance between gripper and object at segment start
            g = np.asarray(gripper_position, dtype=np.float32)
            o = np.asarray(object_position, dtype=np.float32)
            dist = np.linalg.norm(g[:3] - o[:3])
            self._current_segment_start_dist = float(dist)

        # Updated target list for the new sequence of protorewards
        target = [[gripper_position, object_position, gripper_states],  # approach
                  [gripper_position, object_position, gripper_states],  # grasp
                  [object_position, goal_position, gripper_states],  # rotate (takes object, goal)
                  [gripper_position, object_position, gripper_states],  # drop
                  [gripper_position, goal_position, gripper_states]][owner]  # withdraw
        # Updated list of protoreward functions to call
        reward = \
            [self.approach_compute, self.grasp_compute, self.rotate_compute, self.drop_compute, self.withdraw_compute][
                owner](
                *target)
        # if self.env.episode_terminated:
        #     reward += 0.2  # Adding reward for succesful finish of episode
        # self.disp_reward(reward, owner)
        self.rewards_history.append(reward)
        self.rewards_num = 5  # Still 5 networks
        return reward

    def decide(self, observation=None):
        goal_position, object_position, gripper_position, gripper_states = self.get_positions(observation)

        if self.env.network_switcher == "keyboard":
            self.change_network_based_on_key()
        elif self.env.network_switcher == "gt":
            # Logic for switching networks based on predicates
            if self.current_network == 0:  # approach
                if self.gripper_approached_object(gripper_position, object_position):
                    if self.gripper_opened(gripper_states):
                        self.current_network = 1
            elif self.current_network == 1:  # grasp
                if self.gripper_approached_object(gripper_position, object_position):
                    if self.gripper_closed(gripper_states):
                        self.current_network = 2
            elif self.current_network == 2:  # rotate
                # Switch from rotate to drop when the object is near the goal
                # (Rotation might also affect position, or this is a proxy)
                if self.object_near_goal(object_position, goal_position):
                    self.current_network = 3
            elif self.current_network == 3:  # drop
                if self.gripper_approached_object(gripper_position,
                                                  object_position):  # Check approach to object again for dropping
                    if self.gripper_opened(gripper_states):
                        self.current_network = 4
            # Network 4 (withdraw) is the last step before checking goal

        # Check for task completion in the final network stage
        if self.current_network == 4:  # withdraw
            if self.gripper_withdraw_object(gripper_position, object_position):  # Check withdraw distance
                if self.gripper_opened(gripper_states):  # Ensure gripper is open after withdrawing
                    self.task.check_goal()

        self.task.check_episode_steps()
        return self.current_network


class AaGaFaDaW(Protorewards):
    """
    Reward sequence: Approach -> Grasp -> Follow Trajectory -> Drop -> Withdraw
    Uses the 'follow_compute' protoreward for the third step.
    """

    def __init__(self, env, task=None):
        super().__init__(env, task)
        self.network_names = ["approach", "grasp", "follow", "drop", "withdraw"]
        self._current_segment_start_dist = 0.0

    def reset(self):
        super().reset()
        # Updated network names to include "follow"
        self.network_names = ["approach", "grasp", "follow", "drop", "withdraw"]

    def compute(self, observation=None):
        goal_position, object_position, gripper_position, gripper_states = self.get_positions(observation)
        owner = self.decide(observation)

        if self._current_subpolicy_steps == 0:
            # first step of the current subpolicy segment
            # distance between gripper and object at segment start
            g = np.asarray(gripper_position, dtype=np.float32)
            o = np.asarray(object_position, dtype=np.float32)
            dist = np.linalg.norm(g[:3] - o[:3])
            self._current_segment_start_dist = float(dist)

        # --- Get Trajectory for Follow ---
        # NOTE: Assumes the task object has a method or attribute to get the relevant trajectory
        # You might need to adjust this based on your specific Task implementation
        try:
            # Example: Get trajectory from the task object
            trajectory = self.task.get_current_trajectory(observation)
            if trajectory is None:
                print("Warning: get_current_trajectory returned None. Follow reward might be incorrect.")
                # Provide a default or handle appropriately if trajectory isn't available
                trajectory = np.array([])  # Example default
        except AttributeError:
            print(
                "Warning: Task object does not have 'get_current_trajectory' method. Using empty trajectory for 'follow'.")
            trajectory = np.array([])  # Default empty trajectory if method doesn't exist
        except Exception as e:
            print(f"Warning: Error getting trajectory: {e}. Using empty trajectory.")
            trajectory = np.array([])

        # Define targets for each protoreward function
        # Note: follow_compute needs object_position, goal_position, and the trajectory
        # Note: drop_compute target uses object_position like in AaGaMaDaW
        target = [
            [gripper_position, object_position, gripper_states],  # approach
            [gripper_position, object_position, gripper_states],  # grasp
            [object_position, goal_position, trajectory],  # follow
            [gripper_position, object_position, gripper_states],  # drop
            [gripper_position, goal_position, gripper_states]  # withdraw (away from goal)
        ][owner]

        # List of protoreward functions corresponding to network_names
        reward_func_list = [
            self.approach_compute,
            self.grasp_compute,
            self.follow_compute,  # Using follow_compute here
            self.drop_compute,
            self.withdraw_compute
        ]

        # Calculate reward using the function for the current network owner
        reward = reward_func_list[owner](*target)

        # Add bonus for successful episode termination
        if self.env.episode_terminated:
            reward += 0.2

        # Display, log, and update history
        # self.disp_reward(reward, owner)
        self.rewards_history.append(reward)
        self.rewards_num = 5  # Total number of networks remains 5
        return reward

    def decide(self, observation=None):
        goal_position, object_position, gripper_position, gripper_states = self.get_positions(observation)

        if self.env.network_switcher == "keyboard":
            self.change_network_based_on_key()
        elif self.env.network_switcher == "gt":
            # Ground truth logic for switching networks based on predicates
            if self.current_network == 0:  # approach
                # Switch to grasp if gripper is near object and open
                if self.gripper_approached_object(gripper_position, object_position) and self.gripper_opened(
                        gripper_states):
                    self.current_network = 1
            elif self.current_network == 1:  # grasp
                # Switch to follow if gripper is near object and closed (grasped)
                if self.gripper_approached_object(gripper_position, object_position) and self.gripper_closed(
                        gripper_states):
                    self.current_network = 2
            elif self.current_network == 2:  # follow
                # Switch to drop when the object following the trajectory is near the goal
                if self.object_near_goal(object_position, goal_position):
                    self.current_network = 3
            elif self.current_network == 3:  # drop
                # Switch to withdraw if gripper is near the object (or goal) and opened
                # Using object position check consistent with AaGaMaDaW's drop->withdraw transition
                if self.gripper_approached_object(gripper_position, object_position) and self.gripper_opened(
                        gripper_states):
                    self.current_network = 4
            # Network 4 (withdraw) is the final stage before checking goal completion

        # Check for task completion in the final network stage (withdraw)
        if self.current_network == 4:
            # Check if gripper has withdrawn sufficiently from the object and is open
            if self.gripper_withdraw_object(gripper_position, object_position) and self.gripper_opened(gripper_states):
                self.task.check_goal()  # Check if the overall task goal is met

        # Check for episode step limits
        self.task.check_episode_steps()
        return self.current_network


class AaGaTaDaW(Protorewards):
    """
    Reward sequence: Approach -> Grasp -> Transform Trajectory -> Drop -> Withdraw
    Uses the 'transform_compute' protoreward for the third step.
    """

    def __init__(self, env, task=None):
        super().__init__(env, task)
        self.network_names = ["approach", "grasp", "transform", "drop", "withdraw"]
        self._current_segment_start_dist = 0.0

    def reset(self):
        super().reset()
        # Updated network names to include "transform"
        self.network_names = ["approach", "grasp", "transform", "drop", "withdraw"]

    def compute(self, observation=None):
        goal_position, object_position, gripper_position, gripper_states = self.get_positions(observation)
        owner = self.decide(observation)

        if self._current_subpolicy_steps == 0:
            # first step of the current subpolicy segment
            # distance between gripper and object at segment start
            g = np.asarray(gripper_position, dtype=np.float32)
            o = np.asarray(object_position, dtype=np.float32)
            dist = np.linalg.norm(g[:3] - o[:3])
            self._current_segment_start_dist = float(dist)

        # --- Get Trajectory for Transform ---
        # NOTE: Assumes the task object has a method or attribute to get the relevant trajectory
        # This logic is similar to AaGaFaDaW, adjust if needed for transform's specific trajectory requirements.
        try:
            # Example: Get trajectory from the task object
            trajectory = self.task.get_current_trajectory(observation)
            if trajectory is None:
                print("Warning: get_current_trajectory returned None. Transform reward might be incorrect.")
                trajectory = np.array([])  # Example default
        except AttributeError:
            print(
                "Warning: Task object does not have 'get_current_trajectory' method. Using empty trajectory for 'transform'.")
            trajectory = np.array([])  # Default empty trajectory
        except Exception as e:
            print(f"Warning: Error getting trajectory: {e}. Using empty trajectory.")
            trajectory = np.array([])

        # Define targets for each protoreward function
        # Note: transform_compute needs object_position, goal_position, and the trajectory
        target = [
            [gripper_position, object_position, gripper_states],  # approach
            [gripper_position, object_position, gripper_states],  # grasp
            [object_position, goal_position, trajectory],  # transform
            [gripper_position, object_position, gripper_states],  # drop
            [gripper_position, goal_position, gripper_states]  # withdraw (away from goal)
        ][owner]

        # List of protoreward functions corresponding to network_names
        reward_func_list = [
            self.approach_compute,
            self.grasp_compute,
            self.transform_compute,  # Using transform_compute here
            self.drop_compute,
            self.withdraw_compute
        ]

        # Calculate reward using the function for the current network owner
        # Pass magnetization=True explicitly if needed by transform_compute, otherwise default is used
        if owner == 2:  # transform network
            reward = reward_func_list[owner](*target)  # Uses default magnetization=True
            # Or explicitly: reward = reward_func_list[owner](*target, magnetization=True)
        else:
            reward = reward_func_list[owner](*target)

        # Add bonus for successful episode termination
        if self.env.episode_terminated:
            reward += 0.2

        # Display, log, and update history
        # self.disp_reward(reward, owner)
        self.rewards_history.append(reward)
        self.rewards_num = 5  # Total number of networks remains 5
        return reward

    def decide(self, observation=None):
        goal_position, object_position, gripper_position, gripper_states = self.get_positions(observation)

        if self.env.network_switcher == "keyboard":
            self.change_network_based_on_key()
        elif self.env.network_switcher == "gt":
            # Ground truth logic for switching networks based on predicates
            if self.current_network == 0:  # approach
                # Switch to grasp if gripper is near object and open
                if self.gripper_approached_object(gripper_position, object_position) and self.gripper_opened(
                        gripper_states):
                    self.current_network = 1
            elif self.current_network == 1:  # grasp
                # Switch to transform if gripper is near object and closed (grasped)
                if self.gripper_approached_object(gripper_position, object_position) and self.gripper_closed(
                        gripper_states):
                    self.current_network = 2
            elif self.current_network == 2:  # transform
                # Switch to drop when the object being transformed is near the goal
                if self.object_near_goal(object_position, goal_position):
                    self.current_network = 3
            elif self.current_network == 3:  # drop
                # Switch to withdraw if gripper is near the object (or goal) and opened
                if self.gripper_approached_object(gripper_position, object_position) and self.gripper_opened(
                        gripper_states):
                    self.current_network = 4
            # Network 4 (withdraw) is the final stage

        # Check for task completion in the final network stage (withdraw)
        if self.current_network == 4:
            # Check if gripper has withdrawn sufficiently from the object and is open
            if self.gripper_withdraw_object(gripper_position, object_position) and self.gripper_opened(gripper_states):
                self.task.check_goal()  # Check if the overall task goal is met

        # Check for episode step limits
        self.task.check_episode_steps()
        return self.current_network
