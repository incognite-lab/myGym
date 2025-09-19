from time import time
from typing import Callable, Optional, TypeVar

import numpy as np
#import pytest
import os

from myGym.envs.env_object import EnvObject
from myGym.envs.robot import Robot
import importlib.resources as resources
from myGym.envs.predicates import *
from pyquaternion import Quaternion
currentdir = resources.files("myGym").joinpath("envs")


from rddl import AtomicAction, Entity, Reward
from rddl.actions import Approach, Grasp, Drop, Move, Follow, Rotate, Withdraw
from rddl.entities import GraspableObject, Gripper, Location, ObjectEntity, AbstractRotation
from rddl.operators import NotOp
from rddl.predicates import IsHolding, Near, OnTop


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


### YCB OBJECTS ###

class Apple(GraspableObject, EnvObject):

    def __init__(self, reference: Optional[str] = None, kind: str = "", **kw):
        self.urdf = "objects/ycb/00_apple_red/00_apple_red.urdf"
        self.rgba = None
        kw["urdf_path"] = resources.files("myGym").joinpath("envs",self.urdf)
        kw["rgba"] = self.rgba
        super().__init__(self._get_generic_reference() if reference is None else reference, "apple", **kw)
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





class RobotGripper(Gripper, Robot):

    def __init__(self, reference: Optional[str] = None, **kw):
        super().__init__("gripper_robot" if reference is None else reference, **kw)

