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
    print("\nAll tests passed!")
