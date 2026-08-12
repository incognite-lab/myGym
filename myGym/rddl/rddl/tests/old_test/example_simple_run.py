import os
import sys
from typing import Optional

import numpy as np

SCRIPT_DIR = os.path.dirname(__file__)
MODULE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
sys.path.append(MODULE_DIR)

from rddl import Entity, Operand, Predicate, Reward, Variable
from rddl.action import AtomicAction
from rddl.entity import Gripper, Location, LocationType, ObjectEntity
from rddl.predicate import IsHolding, Near, Not
from rddl.reward import GraspReward, NearReward

"""
 The following function and variables should be defined for a specific environment
"""


def euclidean_distance(point_A: np.ndarray, point_B: np.ndarray) -> float:
    return np.linalg.norm(point_A - point_B)


def is_holding(gripper, obj) -> bool:
    return np.linalg.norm(gripper.location - obj.location) < NEAR_THRESHOLD


NEAR_THRESHOLD = 0.1

mapping = {
    "is_holding": is_holding,
    "euclidean_distance": euclidean_distance,
    "near_threshold": NEAR_THRESHOLD,
}

# CONSTANTS

# END OF CUSTOM FUNCTIONS & VARIABLES


class EnvObjectProxy:  # This should be replaced by EnvObject from myGym

    def __init__(self, name: str):
        self.name = name
        self._position = np.random.randn(3)
        self._orientation = np.random.randn(3)

    def set_position(self, position: np.ndarray):
        self._position = position

    def set_orientation(self, orientation: np.ndarray):
        self._orientation = orientation

    def get_position(self) -> np.ndarray:
        return self._position

    def get_orientation(self) -> np.ndarray:
        return self._orientation

    def get_position_and_orientation(self) -> tuple[np.ndarray, np.ndarray]:
        return self.get_position(), self.get_orientation()

    def get_name(self) -> str:
        return self.name

    def get_uuid(self) -> str:
        return self.name + "_uuid"


class EnvSimulator:  # This should be replaced by actual env from myGym
    """Simulates the simulation environment"""

    def __init__(self, list_of_objects: list[str]):
        self._objects = [EnvObjectProxy(obj) for obj in list_of_objects]

    def reset(self):
        pass

    def step(self, action):
        # generate observation
        obs = {}
        for obj in self._objects:
            obj.set_position(np.random.randn(3))  # randomly jiggle the object about
            obj.set_orientation(np.random.randn(3))
            obs[obj.get_name()] = obj
        return obs


class Observer:  # This class serves merely as a container & memory for current observation

    def __init__(self):
        self.obs = None

    def set_observation(self, obs):
        self.obs = obs

    def get_observation(self):
        return self.obs


Location.monkey_patch(Location._get_location, lambda self: self().get_position())


class Apple(ObjectEntity):

    def __init__(self, reference: Optional[str] = None, kind: str = "RedDelicious"):
        super().__init__(self._get_generic_reference() if reference is None else reference, "apple")
        self._kind = kind


class TiagoGripper(Gripper):

    def __init__(self, reference: Optional[str] = None):
        super().__init__("gripper_tiago" if reference is None else reference)
        self._is_holding_predicate = IsHolding(self)


class Find(AtomicAction):

    # def __init__(self, g: Gripper, o: ObjectEntity):
    def __init__(self, g: Variable[Gripper], o: Variable[ObjectEntity]):
        self._g: Variable[Gripper] = g
        self._o: Variable[ObjectEntity] = o
        self._initial = Not(Near(self._g, self._o))
        self._goal = Near(self._g, self._o)
        self._reward = NearReward(self._g, self._o)


class Grasp(AtomicAction):

    def __init__(self, gripper: Gripper, obj: ObjectEntity) -> None:
        self._gripper = gripper
        self._obj = obj
        self._holding_predicate = IsHolding(self._gripper)
        self._initial = Not(self._holding_predicate(self._obj))
        self._goal = self._holding_predicate(self._obj)
        self._reward = GraspReward(self._gripper, self._obj)


if __name__ == "__main__":
    Operand.set_mapping(mapping)

    gripper_name = "tiago_gripper"
    apple_name = "apple_01"
    g1 = Variable("gripper", Gripper)
    o1 = Variable("apple", ObjectEntity)
    print(f"gripper: {g1}")

    r = Find(g1, o1)

    env = EnvSimulator([gripper_name, apple_name])
    observer = Observer()
    Entity.set_observation_getter(observer.get_observation)

    t_gripper = TiagoGripper(gripper_name)
    apple = Apple(apple_name)

    g1.bind(t_gripper)
    o1.bind(apple)
    print(f"gripper: {g1}")

    env.reset()
    for i in range(10):
        print(f"Step {i}")
        obs = env.step([0])
        observer.set_observation(obs)
        print(f"Apple position: {apple.location}")

        print(f"gripper.location: {g1.location}")
        print(f"holding(gripper, apple): {g1.is_holding(o1)}")

        print(f"Decision for Reach action: {r.decide()}")
        print(f"Reward: {r.evaluate()}")
        print("==========")
