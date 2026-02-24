"""
Tests for the UniversalReward class in rewards.py.
Validates absolute, relative, temporal reward computation,
progress tracking, and solved status.
"""

import sys
import os
import numpy as np

# Add parent paths so we can import directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


class MockTask:
    """Mock task with calc_distance and calc_rot_quat methods."""
    def __init__(self):
        self.distance_type = "euclidean"

    def calc_distance(self, obj1, obj2):
        return np.linalg.norm(np.asarray(obj1[:3]) - np.asarray(obj2[:3]))

    def calc_rot_quat(self, obj1, obj2):
        # Simple quaternion distance approximation for testing
        q1 = np.asarray(obj1[3:])
        q2 = np.asarray(obj2[3:])
        dot = np.clip(np.abs(np.dot(q1, q2)), 0.0, 1.0)
        return 2.0 * np.arccos(dot)


class MockRobot:
    """Mock robot with check_gripper_status, close_gripper, open_gripper."""
    def __init__(self):
        self.close_gripper = [0.0]
        self.open_gripper = [1.0]

    def check_gripper_status(self, gripper_values):
        close_vec = np.array(self.close_gripper)
        open_vec = np.array(self.open_gripper)
        current_vec = np.array(gripper_values)
        range_vec = open_vec - close_vec
        if np.allclose(range_vec, 0):
            metric = 0.0
        else:
            normalized = (current_vec - close_vec) / range_vec
            metric = float(np.mean(np.clip(normalized, 0.0, 1.0)))
        if metric <= 0.1:
            status = "close"
        elif metric >= 0.9:
            status = "open"
        else:
            status = "neutral"
        return status, metric


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'envs'))
from rewards import UniversalReward


def test_first_step_returns_zero_rewards():
    """First step should record max/min distances and return all zeros."""
    task = MockTask()
    robot = MockRobot()
    ur = UniversalReward(task, robot)

    actual = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    goal = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    gripper = [0.5]

    result = ur.compute(actual, goal, gripper)

    assert result["task_absolute_reward"] == 0.0, f"Expected 0.0, got {result['task_absolute_reward']}"
    assert result["task_relative_reward"] == 0.0, f"Expected 0.0, got {result['task_relative_reward']}"
    assert result["task_temporal_reward"] == 0.0
    assert result["task_progress"] == 0.0
    assert result["task_solved"] == False
    assert result["gripper_absolute_reward"] == 0.0
    assert result["gripper_relative_reward"] == 0.0
    assert result["gripper_temporal_reward"] == 0.0
    assert result["total_reward"] == 0.0
    print("PASS: test_first_step_returns_zero_rewards")


def test_positive_relative_reward_when_distance_decreases():
    """When distance decreases, relative reward should be positive."""
    task = MockTask()
    robot = MockRobot()
    ur = UniversalReward(task, robot)

    goal = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    gripper = [0.5]

    # Step 0: far away
    actual_0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    ur.compute(actual_0, goal, gripper)

    # Step 1: closer
    actual_1 = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    result = ur.compute(actual_1, goal, gripper)

    assert result["task_relative_reward"] > 0, f"Expected positive, got {result['task_relative_reward']}"
    print("PASS: test_positive_relative_reward_when_distance_decreases")


def test_negative_relative_reward_when_distance_increases():
    """When distance increases, relative reward should be negative."""
    task = MockTask()
    robot = MockRobot()
    ur = UniversalReward(task, robot)

    goal = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    gripper = [0.5]

    # Step 0: close
    actual_0 = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    ur.compute(actual_0, goal, gripper)

    # Step 1: further away
    actual_1 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    result = ur.compute(actual_1, goal, gripper)

    assert result["task_relative_reward"] < 0, f"Expected negative, got {result['task_relative_reward']}"
    print("PASS: test_negative_relative_reward_when_distance_increases")


def test_absolute_reward_scaling():
    """Absolute reward should be 0 at max distance and 1 at min distance."""
    task = MockTask()
    robot = MockRobot()
    ur = UniversalReward(task, robot)

    goal = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    gripper = [0.5]

    # Step 0: initial distance = 1.0 (max)
    actual_0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    ur.compute(actual_0, goal, gripper)

    # Step 1: move to distance = 0.5 (now min)
    actual_1 = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    result = ur.compute(actual_1, goal, gripper)

    # At distance 0.5, with max=1.0 and min=0.5:
    # normalized = (0.5 - 0.5) / (1.0 - 0.5) = 0
    # absolute = 1.0 - 0 = 1.0 (for translation)
    assert result["task_absolute_reward"] > 0.0, f"Expected > 0, got {result['task_absolute_reward']}"

    # Step 2: move back to max distance
    actual_2 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    result_2 = ur.compute(actual_2, goal, gripper)

    # At max distance, absolute reward for translation should be 0
    # normalized = (1.0 - 0.5) / (1.0 - 0.5) = 1.0
    # absolute = 1.0 - 1.0 = 0.0
    # But rotation dist hasn't changed so rot part = 0
    # Task absolute = (0.0 + 0.0) / 2.0 = 0.0
    assert result_2["task_absolute_reward"] == 0.0, f"Expected 0.0 at max distance, got {result_2['task_absolute_reward']}"
    print("PASS: test_absolute_reward_scaling")


def test_progress_and_solved():
    """Progress should reach 90%+ and solved should become True."""
    task = MockTask()
    robot = MockRobot()
    ur = UniversalReward(task, robot)

    goal = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    gripper = [0.5]

    # Step 0: distance = 1.0
    actual_0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    ur.compute(actual_0, goal, gripper)

    # Step 1: distance = 0.05 (95% progress in translation, rot = 0)
    actual_1 = [0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    result = ur.compute(actual_1, goal, gripper)

    assert result["task_progress"] >= 90.0, f"Expected >= 90%, got {result['task_progress']}"
    assert result["task_solved"] == True, f"Expected True, got {result['task_solved']}"
    print("PASS: test_progress_and_solved")


def test_progress_not_solved_below_threshold():
    """Progress below 90% should not mark task as solved."""
    task = MockTask()
    robot = MockRobot()
    ur = UniversalReward(task, robot)

    goal = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    gripper = [0.5]

    # Step 0: distance = 1.0
    actual_0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    ur.compute(actual_0, goal, gripper)

    # Step 1: distance = 0.5 (50% progress in translation)
    actual_1 = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    result = ur.compute(actual_1, goal, gripper)

    assert result["task_progress"] < 90.0, f"Expected < 90%, got {result['task_progress']}"
    assert result["task_solved"] == False, f"Expected False, got {result['task_solved']}"
    print("PASS: test_progress_not_solved_below_threshold")


def test_temporal_reward_sliding_window():
    """Temporal reward should reflect sliding window mean behavior."""
    task = MockTask()
    robot = MockRobot()
    ur = UniversalReward(task, robot, window_size=3)

    goal = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    gripper = [0.5]

    # Step 0: initial
    actual = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    ur.compute(actual, goal, gripper)

    # Steps 1-3: progressively closer (positive trend)
    for x in [0.8, 0.6, 0.4]:
        actual = [x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        result = ur.compute(actual, goal, gripper)

    # All relative rewards were positive (distance decreasing)
    # Temporal should be positive
    assert result["task_temporal_reward"] > 0, f"Expected positive temporal, got {result['task_temporal_reward']}"
    print("PASS: test_temporal_reward_sliding_window")


def test_gripper_reward():
    """Gripper reward should track gripper distance changes."""
    task = MockTask()
    robot = MockRobot()
    ur = UniversalReward(task, robot)

    goal = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    actual = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    # Step 0: gripper at 0.5
    ur.compute(actual, goal, [0.5])

    # Step 1: gripper closer to closed (decreasing distance metric)
    result = ur.compute(actual, goal, [0.3])
    assert result["gripper_relative_reward"] > 0, f"Expected positive gripper relative, got {result['gripper_relative_reward']}"

    # Step 2: gripper opens further (increasing distance metric)
    result2 = ur.compute(actual, goal, [0.8])
    assert result2["gripper_relative_reward"] < 0, f"Expected negative gripper relative, got {result2['gripper_relative_reward']}"
    print("PASS: test_gripper_reward")


def test_reset():
    """After reset, the reward calculator should behave as if starting fresh."""
    task = MockTask()
    robot = MockRobot()
    ur = UniversalReward(task, robot)

    goal = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    actual = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    # Run a few steps
    ur.compute(actual, goal, [0.5])
    ur.compute([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], goal, [0.3])

    # Reset
    ur.reset()

    # After reset, first step should return zeros again
    result = ur.compute(actual, goal, [0.5])
    assert result["total_reward"] == 0.0, f"Expected 0.0 after reset, got {result['total_reward']}"
    assert ur.step == 1
    print("PASS: test_reset")


def test_zero_distance_same_position():
    """When actual equals goal, progress should be 100% (solved)."""
    task = MockTask()
    robot = MockRobot()
    ur = UniversalReward(task, robot)

    goal = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    gripper = [0.5]

    # Step 0: some distance
    actual_0 = [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    ur.compute(actual_0, goal, gripper)

    # Step 1: exactly at goal
    result = ur.compute(goal, goal, gripper)
    assert result["task_progress"] == 100.0, f"Expected 100%, got {result['task_progress']}"
    assert result["task_solved"] == True
    print("PASS: test_zero_distance_same_position")


def test_rot_false_ignores_rotation():
    """When rot=False, rotation should not affect task rewards or progress."""
    task = MockTask()
    robot = MockRobot()
    ur = UniversalReward(task, robot)

    # Identical translation, different rotation
    actual = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    goal = [0.0, 0.0, 0.0, 0.707, 0.0, 0.707, 0.0]
    gripper = [0.5]

    # Step 0
    ur.compute(actual, goal, gripper, rot=False)

    # Step 1: move closer in translation only
    actual_1 = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    result = ur.compute(actual_1, goal, gripper, rot=False)

    # With rot=False, task_absolute_reward should equal translation-only reward
    # Translation: trans_dist=0.5, max=1.0, min=0.5 → abs = 1.0
    assert result["task_absolute_reward"] == 1.0, f"Expected 1.0, got {result['task_absolute_reward']}"

    # Progress should be translation-only: (1 - 0.5/1.0)*100 = 50%
    assert result["task_progress"] == 50.0, f"Expected 50.0, got {result['task_progress']}"
    print("PASS: test_rot_false_ignores_rotation")


def test_rot_true_includes_rotation():
    """When rot=True, rotation should affect task rewards and progress."""
    task = MockTask()
    robot = MockRobot()
    ur = UniversalReward(task, robot)

    # Different rotation between actual and goal
    actual = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    goal = [0.0, 0.0, 0.0, 0.707, 0.0, 0.707, 0.0]
    gripper = [0.5]

    # Step 0
    ur.compute(actual, goal, gripper, rot=True)

    # Step 1: move closer in translation, rotation unchanged
    actual_1 = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    result = ur.compute(actual_1, goal, gripper, rot=True)

    # With rot=True, task_absolute_reward = (trans_abs + rot_abs) / 2
    # Translation abs = 1.0, rotation abs = 0.0 (unchanged from max)
    # So task_absolute_reward = (1.0 + 0.0) / 2.0 = 0.5
    assert result["task_absolute_reward"] == 0.5, f"Expected 0.5, got {result['task_absolute_reward']}"

    # Progress should average translation and rotation
    # trans_progress = 50%, rot_progress = 0% → average = 25%
    assert result["task_progress"] == 25.0, f"Expected 25.0, got {result['task_progress']}"
    print("PASS: test_rot_true_includes_rotation")


def test_gripper_open_rewards_opening():
    """When gripper='Open', reward should increase as gripper opens (metric increases)."""
    task = MockTask()
    robot = MockRobot()
    ur = UniversalReward(task, robot)

    actual = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    goal = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    # Step 0: gripper at 0.5
    ur.compute(actual, goal, [0.5], gripper="Open")

    # Step 1: gripper opens to 0.8 (metric increases)
    result = ur.compute(actual, goal, [0.8], gripper="Open")

    # With gripper="Open", relative reward should be positive when metric increases
    assert result["gripper_relative_reward"] > 0, \
        f"Expected positive gripper relative for Open, got {result['gripper_relative_reward']}"
    print("PASS: test_gripper_open_rewards_opening")


def test_gripper_open_penalizes_closing():
    """When gripper='Open', reward should decrease as gripper closes (metric decreases)."""
    task = MockTask()
    robot = MockRobot()
    ur = UniversalReward(task, robot)

    actual = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    goal = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    # Step 0: gripper at 0.5
    ur.compute(actual, goal, [0.5], gripper="Open")

    # Step 1: gripper closes to 0.2 (metric decreases)
    result = ur.compute(actual, goal, [0.2], gripper="Open")

    # With gripper="Open", relative reward should be negative when metric decreases
    assert result["gripper_relative_reward"] < 0, \
        f"Expected negative gripper relative for Open, got {result['gripper_relative_reward']}"
    print("PASS: test_gripper_open_penalizes_closing")


def test_gripper_open_absolute_reward_at_max():
    """When gripper='Open', absolute reward should be 1.0 at maximal gripper value."""
    task = MockTask()
    robot = MockRobot()
    ur = UniversalReward(task, robot)

    actual = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    goal = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    # Step 0: gripper at 0.3
    ur.compute(actual, goal, [0.3], gripper="Open")

    # Step 1: gripper at 0.9 (near max)
    ur.compute(actual, goal, [0.9], gripper="Open")

    # Step 2: gripper back to 0.3 (at min)
    result = ur.compute(actual, goal, [0.3], gripper="Open")

    # At min value (0.3), with Open mode: absolute = (0.3 - 0.3)/(0.9 - 0.3) = 0.0
    assert result["gripper_absolute_reward"] == 0.0, \
        f"Expected 0.0 at min for Open mode, got {result['gripper_absolute_reward']}"

    # Step 3: gripper back to max (0.9)
    result2 = ur.compute(actual, goal, [0.9], gripper="Open")

    # At max value (0.9), with Open mode: absolute = (0.9 - 0.3)/(0.9 - 0.3) = 1.0
    assert result2["gripper_absolute_reward"] == 1.0, \
        f"Expected 1.0 at max for Open mode, got {result2['gripper_absolute_reward']}"
    print("PASS: test_gripper_open_absolute_reward_at_max")


def test_gripper_open_progress_at_max():
    """When gripper='Open', progress should be high when gripper is at maximal value."""
    task = MockTask()
    robot = MockRobot()
    ur = UniversalReward(task, robot)

    actual = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    goal = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    # Step 0: gripper at 0.2
    ur.compute(actual, goal, [0.2], gripper="Open")

    # Step 1: gripper opens to 0.95 (95% of max=0.95)
    result = ur.compute(actual, goal, [0.95], gripper="Open")

    # Progress for Open = (grip_dist / max_grip_dist) * 100 = (0.95/0.95)*100 = 100%
    assert result["gripper_progress"] >= 90.0, \
        f"Expected >= 90%, got {result['gripper_progress']}"
    assert result["gripper_solved"] == True, \
        f"Expected True, got {result['gripper_solved']}"
    print("PASS: test_gripper_open_progress_at_max")


def test_gripper_close_progress_at_min():
    """When gripper='Close', progress should be high when gripper is at minimal value."""
    task = MockTask()
    robot = MockRobot()
    ur = UniversalReward(task, robot)

    actual = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    goal = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    # Step 0: gripper at 0.8
    ur.compute(actual, goal, [0.8], gripper="Close")

    # Step 1: gripper closes to 0.05 (close to 0 = closed)
    result = ur.compute(actual, goal, [0.05], gripper="Close")

    # For Close mode, progress = (1 - 0.05/0.8)*100 = 93.75%
    assert result["gripper_progress"] >= 90.0, \
        f"Expected >= 90%, got {result['gripper_progress']}"
    assert result["gripper_solved"] == True, \
        f"Expected True, got {result['gripper_solved']}"
    print("PASS: test_gripper_close_progress_at_min")


def test_default_params_backward_compatible():
    """Default parameters (rot=True, gripper='Close') should match original behavior."""
    task = MockTask()
    robot = MockRobot()

    ur_default = UniversalReward(task, robot)
    ur_explicit = UniversalReward(task, robot)

    goal = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    actual_0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    actual_1 = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    gripper = [0.5]

    # Call without explicit params (defaults)
    ur_default.compute(actual_0, goal, gripper)
    result_default = ur_default.compute(actual_1, goal, gripper)

    # Call with explicit defaults
    ur_explicit.compute(actual_0, goal, gripper, rot=True, gripper="Close")
    result_explicit = ur_explicit.compute(actual_1, goal, gripper, rot=True, gripper="Close")

    for key in result_default:
        assert result_default[key] == result_explicit[key], \
            f"Key {key}: default={result_default[key]} vs explicit={result_explicit[key]}"
    print("PASS: test_default_params_backward_compatible")


if __name__ == "__main__":
    test_first_step_returns_zero_rewards()
    test_positive_relative_reward_when_distance_decreases()
    test_negative_relative_reward_when_distance_increases()
    test_absolute_reward_scaling()
    test_progress_and_solved()
    test_progress_not_solved_below_threshold()
    test_temporal_reward_sliding_window()
    test_gripper_reward()
    test_reset()
    test_zero_distance_same_position()
    test_rot_false_ignores_rotation()
    test_rot_true_includes_rotation()
    test_gripper_open_rewards_opening()
    test_gripper_open_penalizes_closing()
    test_gripper_open_absolute_reward_at_max()
    test_gripper_open_progress_at_max()
    test_gripper_close_progress_at_min()
    test_default_params_backward_compatible()
    print("\nAll tests passed!")
