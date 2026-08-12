from typing import Optional

import numpy as np

from rddl import Entity, Operand, Variable
from rddl.actions import AtomicAction
from rddl.entities import Gripper, Location, ObjectEntity
from rddl.predicates import IsHolding, Near

NEAR_THRESHOLD = 0.1


def euclidean_distance(point_A: np.ndarray, point_B: np.ndarray) -> float:
    return np.linalg.norm(point_A - point_B)


def is_holding(gripper, obj) -> bool:
    return np.linalg.norm(gripper.location - obj.location) < NEAR_THRESHOLD


mapping = {
    "is_holding": is_holding,
    "euclidean_distance": euclidean_distance,
    "near_threshold": NEAR_THRESHOLD,
}


Operand.set_mapping(mapping)


class Approach(AtomicAction):
    _VARIABLES = {
        "gripper": Gripper,
        "object": ObjectEntity
    }

    # def __init__(self, g: Gripper, o: ObjectEntity):
    def __init__(self):
        super().__init__()
        self._predicate = Near(self.get_argument("gripper"), self.get_argument("object"))
        self._initial = Not(self._predicate)


Location.monkey_patch(Location._get_location, lambda self: np.random.randn(3))


class Apple(ObjectEntity):

    def __init__(self, reference: Optional[str] = None, kind: str = "RedDelicious"):
        super().__init__(self._get_generic_reference() if reference is None else reference, "apple")
        self._kind = kind


class TiagoGripper(Gripper):

    def __init__(self, reference: Optional[str] = None):
        super().__init__("gripper_tiago" if reference is None else reference)
        self._is_holding_predicate = IsHolding(self)


if __name__ == "__main__":
    Entity.set_observation_getter(lambda: None)
    a = Approach()

    gripper_name = "tiago_gripper"
    apple_name = "apple_01"

    t_gripper = TiagoGripper(gripper_name)
    apple = Apple(apple_name)
    objects_for_approach = {
        "gripper": t_gripper,
        "object": apple
    }

    print(a)
    a.bind(objects_for_approach)
    print(a.gripper.location)
    print(a())
