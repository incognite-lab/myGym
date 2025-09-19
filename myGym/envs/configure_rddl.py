from functools import partial
from rddl import Operand, Variable, Entity
from rddl.entities import Location
import numpy as np
from pyquaternion import Quaternion


NEAR_THRESHOLD = 0.1
GRIPPER_FACTOR = 1
DIST_FACTOR = 0.2


def euclidean_distance(point_A: np.ndarray, point_B: np.ndarray) -> float:
    return np.linalg.norm(np.array(point_A) - np.array(point_B))


def distance_reward(point_A, point_B, increase_distance=False) -> float:
    """
    Implements the computation of RDDL's rewards based on distance between two points.
    Args:
        point_A (np.array): Point A: can be object's or goal's position (without orientation)
        point_B (np.array): Point B: can be object's or goal's position (without orientation)
        increase_distance: Whether larger or smaller distance should be rewarded (approach or withdraw)

    Returns: Computed reward for rl algorithm.
    """
    #TODO: This needs to be refactored after discussion
    sgn = 1 if increase_distance else -1
    dist = euclidean_distance(point_A, point_B)
    return sgn * dist


def gripper_reward(gripper, obj=None, open=True) -> float:
    """
    Implements the computation of RDDL's rewards based on the states of the gripper.
    Args:
        gripper (TiagoGripper): Gripper object from which its state can be retrieved. (can also be any other robot's gripper)
        obj (Apple): Object to be manipulated by gripper - used to compute whether the gripper is close enough.
            (can also be any other type of object defined below)
        open (bool): Whether the gripper should open or close.

    Returns: Computed reward for rl algorithm.
    """
    #TODO: This needs to be refactored after discussion
    sgn = 1 if open else -1
    gripper_states = gripper.get_gjoints_states()
    # distance = euclidean_distance(gripper.get_position(), obj.get_position())
    # reward = sgn*sum(gripper_states)*GRIPPER_FACTOR - DIST_FACTOR*distance
    reward = sgn * sum(gripper_states) * GRIPPER_FACTOR
    return reward


def rotate_reward(object, goal):
    """
    Implements the computation of RDDL's rotation reward based on position and orientation of objects.
    Args:
        object (Apple): Current object with both its position and orientation.
        goal (Apple): Dummy object, used to store and get the desired position and orientation.

    Returns: Computed reward for rl algorithm.
    """
    # TODO: This needs to be refactored after discussion
    distance = euclidean_distance(object.get_position(), goal.get_position())
    rot_distance = Quaternion.distance(Quaternion(object.get_orientation()), Quaternion(goal.get_orientation()))
    return -distance - rot_distance


def gripper_at(gripper, object):
    norm = np.linalg.norm(gripper.location - object.location)
    # print(norm)
    return norm < NEAR_THRESHOLD


mapping = {
    "euclidean_distance": euclidean_distance,
    "near_threshold": NEAR_THRESHOLD,
    "distance_reward": partial(distance_reward, increase_distance=True),
    "gripper_open_reward": partial(gripper_reward, open=True),
    "gripper_close_reward": partial(gripper_reward, open=False),
    "rotate_reward": rotate_reward,
    "gripper_at": gripper_at,
    # "gripper_at": lambda g, o: all(g.location == o.location),
    "gripper_open": lambda g: np.random.random() < 0.5,
    "object_at": lambda g, o: g.location == o.location,
    "exists": lambda e: True,
}
Operand.set_mapping(mapping)


from rddl import Predicate


def is_on_top(obj1, obj2) -> bool:
    return False  # FIXME


def is_holding(gripper, obj) -> bool:
    return bool((np.linalg.norm(gripper.location - obj.location) < NEAR_THRESHOLD))


def is_reachable(gripper, obj) -> bool:
    return True  # FIXME


predicate_mapping = {
    "on_top": is_on_top,
    "is_holding": is_holding,
    "is_reachable": is_reachable,

}
Predicate.set_mapping(predicate_mapping)


Entity.set_observation_getter(lambda self: self)
Location.monkey_patch(Location._get_location, lambda self: self.get_position())

from rddl.actions import Approach
from rddl.rewards import ApproachReward

Approach.REWARD_CLASS = ApproachReward.RELATIVE_REWARD
