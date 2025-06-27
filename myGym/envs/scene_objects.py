from time import time
from typing import Callable, Optional, TypeVar

import numpy as np
#import pytest
import os

from rddl import Operand, Variable
from myGym.envs.env_object import EnvObject
from myGym.envs.robot import Robot
import importlib.resources as resources
currentdir = resources.files("myGym").joinpath("envs")

def time_function(f: Callable, *args, **kwargs):
    start = time()
    result = f(*args, **kwargs)
    end = time()
    return end - start, result


def euclidean_distance(point_A: np.ndarray, point_B: np.ndarray) -> float:
    return np.linalg.norm(np.array(point_A) - np.array(point_B))


def is_holding(gripper, obj) -> bool:
    return bool(np.linalg.norm(gripper.location - obj.location) < NEAR_THRESHOLD)


NEAR_THRESHOLD = 0.1

mapping = {
    "is_holding": is_holding,
    "euclidean_distance": euclidean_distance,
    "is_reachable": lambda g, o: True,
    "near_threshold": NEAR_THRESHOLD,
    "gripper_at": lambda g, o: all(g.location == o.location),
    "gripper_open": lambda g: np.random.random() < 0.5,
    "object_at": lambda g, o: g.location == o.location,
    "exists": lambda e: True,
    "on_top" : lambda e: True
}

Operand.set_mapping(mapping)

from rddl import AtomicAction, Entity, Reward
from rddl.actions import Approach, Grasp, Drop, Move, Follow
from rddl.entities import GraspableObject, Gripper, Location, ObjectEntity
from rddl.operators import NotOp
from rddl.predicates import IsHolding, Near, OnTop

# CONSTANTS

# END OF CUSTOM FUNCTIONS & VARIABLES


class EnvObjectProxy:  # This should be replaced by EnvObject from myGym

    def __init__(self, **kwds):
        super().__init__()
        self._name = self.reference
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
        return self._name

    def get_uuid(self) -> str:
        return self.name + "_uuid"


class EnvSimulator:  # This should be replaced by actual env from myGym
    """Simulates the simulation environment"""

    def __init__(self, list_of_objects: list[EnvObjectProxy]):
        self._objects = list_of_objects

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


Entity.set_observation_getter(lambda self: self)
Location.monkey_patch(Location._get_location, lambda self: self.get_position())


class Apple(GraspableObject, EnvObject):

    def __init__(self, reference: Optional[str] = None, kind: str = "RedDelicious", **kw):
        self.urdf = "objects/household/urdf/apple.urdf"
        self.rgba = (0.54, 0.0, 0.0, 1.0)
        kw["rgba"] = self.rgba
        kw["urdf_path"] = resources.files("myGym").joinpath("envs",self.urdf)
        super().__init__(self._get_generic_reference() if reference is None else reference, "apple", **kw)
        self._kind = kind


#class Banana(GraspableObject, EnvObject):

#    def __init__(self, reference: Optional[str] = None, kind: str = "", **kw):
#        self.urdf = "objects/household/urdf/banana.urdf"
#        self.rgba = (1.0, 0.94, 0.0, 1.0)
#        kw["urdf_path"] = resources.files("myGym").joinpath("envs",self.urdf)
#        kw["rgba"] = self.rgba
#        super().__init__(self._get_generic_reference() if reference is None else reference, "banana", **kw)
#        self._kind = kind



class Tuna(GraspableObject, EnvObject):

    def __init__(self, reference: Optional[str] = None, kind: str = "", **kw):
        self.urdf = "objects/household/urdf/tuna_can.urdf"
        self.rgba = None
        kw["urdf_path"] = resources.files("myGym").joinpath("envs",self.urdf)
        kw["rgba"] = self.rgba
        super().__init__(self._get_generic_reference() if reference is None else reference, "tuna_can", **kw)
        self._kind = kind

class Applered(GraspableObject, EnvObject):

    def __init__(self, reference: Optional[str] = None, kind: str = "", **kw):
        self.urdf = "objects/ycb/00_apple_red/00_apple_red.urdf"
        self.rgba = None
        kw["urdf_path"] = resources.files("myGym").joinpath("envs",self.urdf)
        kw["rgba"] = self.rgba
        super().__init__(self._get_generic_reference() if reference is None else reference, "apple_red", **kw)
        self._kind = kind


class Crackerbox(GraspableObject, EnvObject):

    def __init__(self, reference: Optional[str] = None, kind: str = "", **kw):
        self.urdf = "objects/ycb/003_cracker_box/003_cracker_box.urdf"
        self.rgba = None
        kw["urdf_path"] = resources.files("myGym").joinpath("envs",self.urdf)
        kw["rgba"] = self.rgba
        super().__init__(self._get_generic_reference() if reference is None else reference, "crackerbox", **kw)
        self._kind = kind

class Mustard(GraspableObject, EnvObject):

    def __init__(self, reference: Optional[str] = None, kind: str = "", **kw):
        self.urdf = "objects/ycb/006_mustard_bottle/006_mustard_bottle.urdf"
        self.rgba = None
        kw["urdf_path"] = resources.files("myGym").joinpath("envs",self.urdf)
        kw["rgba"] = self.rgba
        super().__init__(self._get_generic_reference() if reference is None else reference, "mustard", **kw)
        self._kind = kind
        
class Meatcan(GraspableObject, EnvObject):

    def __init__(self, reference: Optional[str] = None, kind: str = "", **kw):
        self.urdf = "objects/ycb/010_potted_meat_can/010_potted_meat_can.urdf"
        self.rgba = None
        kw["urdf_path"] = resources.files("myGym").joinpath("envs",self.urdf)
        kw["rgba"] = self.rgba
        super().__init__(self._get_generic_reference() if reference is None else reference, "potted_meat_can", **kw)
        self._kind = kind


class Banana(GraspableObject, EnvObject):

    def __init__(self, reference: Optional[str] = None, kind: str = "", **kw):
        self.urdf = "objects/ycb/011_banana/011_banana.urdf"
        self.rgba = None
        kw["urdf_path"] = resources.files("myGym").joinpath("envs",self.urdf)
        kw["rgba"] = self.rgba
        super().__init__(self._get_generic_reference() if reference is None else reference, "banana", **kw)
        self._kind = kind

class Strawberry(GraspableObject, EnvObject):

    def __init__(self, reference: Optional[str] = None, kind: str = "", **kw):
        self.urdf = "objects/ycb/012_strawberry/012_strawberry.urdf"
        self.rgba = None
        kw["urdf_path"] = resources.files("myGym").joinpath("envs",self.urdf)
        kw["rgba"] = self.rgba
        super().__init__(self._get_generic_reference() if reference is None else reference, "strawberry", **kw)
        self._kind = kind

class Lemon(GraspableObject, EnvObject):

    def __init__(self, reference: Optional[str] = None, kind: str = "", **kw):
        self.urdf = "objects/ycb/014_lemon/014_lemon.urdf"
        self.rgba = None
        kw["urdf_path"] = resources.files("myGym").joinpath("envs",self.urdf)
        kw["rgba"] = self.rgba
        super().__init__(self._get_generic_reference() if reference is None else reference, "lemon", **kw)
        self._kind = kind

class Peach(GraspableObject, EnvObject):

    def __init__(self, reference: Optional[str] = None, kind: str = "", **kw):
        self.urdf = "objects/ycb/015_peach/015_peach.urdf"
        self.rgba = None
        kw["urdf_path"] = resources.files("myGym").joinpath("envs",self.urdf)
        kw["rgba"] = self.rgba
        super().__init__(self._get_generic_reference() if reference is None else reference, "peach", **kw)
        self._kind = kind

class Orange(GraspableObject, EnvObject):

    def __init__(self, reference: Optional[str] = None, kind: str = "", **kw):
        self.urdf = "objects/ycb/017_orange/017_orange.urdf"
        self.rgba = None
        kw["urdf_path"] = resources.files("myGym").joinpath("envs",self.urdf)
        kw["rgba"] = self.rgba
        super().__init__(self._get_generic_reference() if reference is None else reference, "orange", **kw)
        self._kind = kind

class Bowl(GraspableObject, EnvObject):

    def __init__(self, reference: Optional[str] = None, kind: str = "", **kw):
        self.urdf = "objects/ycb/024_bowl/024_bowl.urdf"
        self.rgba = None
        kw["urdf_path"] = resources.files("myGym").joinpath("envs",self.urdf)
        kw["rgba"] = self.rgba
        super().__init__(self._get_generic_reference() if reference is None else reference, "bowl", **kw)
        self._kind = kind

class Mug(GraspableObject, EnvObject):

    def __init__(self, reference: Optional[str] = None, kind: str = "", **kw):
        self.urdf = "objects/ycb/025_mug/025_mug.urdf"
        self.rgba = None
        kw["urdf_path"] = resources.files("myGym").joinpath("envs",self.urdf)
        kw["rgba"] = self.rgba
        super().__init__(self._get_generic_reference() if reference is None else reference, "mug", **kw)
        self._kind = kind

class Plate(GraspableObject, EnvObject):

    def __init__(self, reference: Optional[str] = None, kind: str = "", **kw):
        self.urdf = "objects/ycb/029_plate/029_plate.urdf"
        self.rgba = None
        kw["urdf_path"] = resources.files("myGym").joinpath("envs",self.urdf)
        kw["rgba"] = self.rgba
        super().__init__(self._get_generic_reference() if reference is None else reference, "plate", **kw)
        self._kind = kind

class Fork(GraspableObject, EnvObject):

    def __init__(self, reference: Optional[str] = None, kind: str = "", **kw):
        self.urdf = "objects/ycb/030_fork/030_fork.urdf"
        self.rgba = None
        kw["urdf_path"] = resources.files("myGym").joinpath("envs",self.urdf)
        kw["rgba"] = self.rgba
        super().__init__(self._get_generic_reference() if reference is None else reference, "fork", **kw)
        self._kind = kind

class Spoon(GraspableObject, EnvObject):

    def __init__(self, reference: Optional[str] = None, kind: str = "", **kw):
        self.urdf = "objects/ycb/031_spoon/031_spoon.urdf"
        self.rgba = None
        kw["urdf_path"] = resources.files("myGym").joinpath("envs",self.urdf)
        kw["rgba"] = self.rgba
        super().__init__(self._get_generic_reference() if reference is None else reference, "spoon", **kw)
        self._kind = kind

class Knife(GraspableObject, EnvObject):

    def __init__(self, reference: Optional[str] = None, kind: str = "", **kw):
        self.urdf = "objects/ycb/032_knife/032_knife.urdf"
        self.rgba = None
        kw["urdf_path"] = resources.files("myGym").joinpath("envs",self.urdf)
        kw["rgba"] = self.rgba
        super().__init__(self._get_generic_reference() if reference is None else reference, "knife", **kw)
        self._kind = kind







class TiagoGripper(Gripper, Robot):

    def __init__(self, reference: Optional[str] = None, **kw):
        super().__init__("gripper_tiago" if reference is None else reference, **kw)
        # self._is_holding_predicate = IsHolding(self)



# @pytest.fixture
def create_approach_action() -> AtomicAction:
    a = Approach()
    return a


# @pytest.fixture
def create_gripper_and_apple() -> dict[str, Entity]:
    gripper_name = "tiago_gripper"
    apple_name = "apple_01"

    t_gripper = TiagoGripper(gripper_name)
    apple = Apple(apple_name)
    objects_for_approach = {
        "gripper": t_gripper,
        "object": apple
    }

    return objects_for_approach