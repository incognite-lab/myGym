"""Reward classes for the atomic actions of the RDDL task generator.

Each reward subclass declares the externally-defined functions it needs via ``_0_*``
class attributes (resolved through the ``_0_*`` mapping mechanism), a ``_VARIABLES``
dict describing the entities it observes, an ``__init__`` whose signature matches those
variables, and a ``__call__ -> float`` producing the scalar reward for the current step.

Two families exist: *absolute* rewards score the current state directly, while *relative*
rewards are stateful and shape on the change in distance between successive calls. The
``Enum`` wrappers at the bottom map a variant name (e.g. RELATIVE/ABSOLUTE) to a concrete
reward class; an action selects one as its ``REWARD_CLASS``.
"""

from typing import Callable

from rddl import Reward, Variable
from rddl.entities import AbstractRotation, GraspableObject, Gripper, Location, LocationType, ObjectEntity
from rddl.predicates import IsHolding
from enum import Enum


""" HOW TO DEFINE A REWARD

class <Some>Reward(Reward):

    _0_<FUNCTION_THIS_REWARD_NEEDS> = "name_of_function"  # "name_of_function" is a function defined externally
    ...  # possibly more functions

    _VARIABLES = {  # Variables that will be part of the observation; not necessarily all variables used by the reward
        "<variable_name>": <variable_type>,
        ...
    }

    def __init__(self, <args>, <kwargs>) -> None:  # eventually, takes arguments from _VARIABLES
        ...

    def __call__(self) -> float:
        return <self._0_<FUNCTION_THIS_REWARD_NEEDS>(<args>, <kwargs>)>  # possibly some other computations
"""

##############################################
# SPECIFIC REWARDS
##############################################

class AbsoluteApproachReward(Reward):
    """Absolute reward for approaching: scores the current gripper-to-object distance.

    Higher reward the closer the gripper is to the object (``increase_distance=False``).
    """

    _0_DIST_REWARD: Callable = "distance_reward"  # type: ignore

    _VARIABLES = {
        "gripper": Gripper,
        "object": GraspableObject
    }

    def __init__(self, gripper: Variable[Gripper], object: Variable[GraspableObject]) -> None:
        super().__init__(gripper=gripper, object=object)
        self._gripper = gripper
        self._obj = object

    def __call__(self) -> float:
        # increase_distance=False -> reward grows as the gripper gets closer to the object
        return self.__class__._0_DIST_REWARD(self._gripper.location, self._obj.location, increase_distance=False)


class RelativeApproachReward(Reward):
    """Stateful relative reward for approaching, shaping on the change in distance.

    Rewards reducing the gripper-to-object distance since the previous call, plus a small
    bonus (weight 0.2) for keeping the gripper open. Formula per step::

        reward = (last_distance - current_distance) + 0.2 * gripper_open_reward(gripper)

    Positive when the gripper moved closer. The first call merely primes ``_last_distance``
    and returns 0.
    """

    _0_EUCLIDIAN_DIST_REWARD: Callable = "euclidean_distance"  # type: ignore
    _0_GRIPPER_OPEN_REWARD: Callable = "gripper_open_reward"  # type: ignore

    _VARIABLES = {
        "gripper": Gripper,
        "object": GraspableObject
    }

    def __init__(self, gripper: Variable[Gripper], object: Variable[GraspableObject]) -> None:
        super().__init__(gripper=gripper, object=object)
        self._gripper = gripper
        self._obj = object
        self._last_distance = None  # previous-step distance; None until first call (priming)

    def __call__(self) -> float:
        dist = self.__class__._0_EUCLIDIAN_DIST_REWARD(self._gripper.location, self._obj.location)
        if self._last_distance is None:
            # prime step: record baseline distance, no reward yet
            self._last_distance = dist
            return 0
        # delta-distance shaping (positive = moved closer) + keep-gripper-open bonus
        reward = self._last_distance - dist + self.__class__._0_GRIPPER_OPEN_REWARD(self._gripper) * 0.2
        self._last_distance = dist
        return reward


class AbsoluteWithdrawReward(Reward):
    """Absolute reward for withdrawing: scores current gripper-to-object distance.

    Mirror of :class:`AbsoluteApproachReward` but with ``increase_distance=True`` so the
    reward grows as the gripper moves away from the object.
    """

    _0_EUCLIDIAN_DIST_REWARD: Callable = "distance_reward"  # type: ignore

    _VARIABLES = {
        "gripper": Gripper,
        "object": GraspableObject
    }

    def __init__(self, gripper: Variable[Gripper], object: Variable[GraspableObject]) -> None:
        super().__init__(gripper=gripper, object=object)
        self._gripper = gripper
        self._obj = object

    def __call__(self) -> float:
        # increase_distance=True -> reward grows as the gripper gets farther from the object
        return self.__class__._0_EUCLIDIAN_DIST_REWARD(self._gripper.location, self._obj.location, increase_distance=True)


class RelativeWithdrawReward(Reward):
    """Stateful relative reward for withdrawing, shaping on the change in distance.

    Same shape as :class:`RelativeApproachReward` but with no gripper-open term. Formula
    per step::

        reward = current_distance - last_distance

    The first call primes ``_last_distance`` and returns 0.
    """

    _0_EUCLIDIAN_DIST_REWARD: Callable = "euclidean_distance"  # type: ignore

    _VARIABLES = {
        "gripper": Gripper,
        "object": GraspableObject
    }

    def __init__(self, gripper: Variable[Gripper], object: Variable[GraspableObject]) -> None:
        super().__init__(gripper=gripper, object=object)
        self._gripper = gripper
        self._obj = object
        self._last_distance = None  # previous-step distance; None until first call (priming)

    def __call__(self) -> float:
        dist = self.__class__._0_EUCLIDIAN_DIST_REWARD(self._gripper.location, self._obj.location)
        if self._last_distance is None:
            # prime step: record baseline distance, no reward yet
            self._last_distance = dist
            return 0
        reward = dist - self._last_distance  # positive when the gripper moves away from the object
        self._last_distance = dist
        return reward


class SimpleGraspReward(Reward):
    """Reward for grasping: rewards closing the gripper on the object.

    Delegates to ``gripper_close_reward(gripper, object, open=False)``.
    """

    _0_GRIPPER_REWARD: Callable = "gripper_close_reward"  # type: ignore
    _VARIABLES = {
        "gripper": Gripper,
        "object": GraspableObject
    }

    def __init__(self, gripper: Variable[Gripper], object: Variable[GraspableObject]) -> None:
        super().__init__(gripper=gripper, object=object)
        self._gripper = gripper
        self._obj = object

    def __call__(self) -> float:
        # open=False -> reward for the gripper being closed around the object
        return self.__class__._0_GRIPPER_REWARD(self._gripper, self._obj, open=False)


class SimpleDropReward(Reward):
    """Reward for dropping: rewards opening the gripper to release the object.

    Delegates to ``gripper_open_reward(gripper, object, open=True)``.
    """

    _0_GRIPPER_REWARD: Callable = "gripper_open_reward"  # type: ignore
    _VARIABLES = {
        "gripper": Gripper,
        "object": GraspableObject
    }

    def __init__(self, gripper: Variable[Gripper], object: Variable[GraspableObject]) -> None:
        super().__init__(gripper=gripper, object=object)
        self._gripper = gripper
        self._obj = object

    def __call__(self) -> float:
        # open=True -> reward for the gripper being open (object released)
        return self.__class__._0_GRIPPER_REWARD(self._gripper, self._obj, open=True)


class AbsoluteMoveReward(Reward):
    """Absolute reward for moving an object to a target location.

    Scores the object-to-target-location distance with ``increase_distance=False``, so the
    reward grows as the object approaches the goal location.
    """

    _0_EUCLIDIAN_DIST_REWARD: Callable = "distance_reward"  # type: ignore
    _VARIABLES = {
        "gripper": Gripper,
        "object": GraspableObject,
        "location": Location,
    }

    def __init__(self, gripper: Variable[Gripper], object: Variable[GraspableObject], location: Variable[Location]) -> None:
        super().__init__(gripper=gripper, object=object, location=location)
        self._gripper = gripper
        self._obj = object
        self._location = location

    def __call__(self) -> float:
        # distance measured object->target location; increase_distance=False -> closer is better
        return self.__class__._0_EUCLIDIAN_DIST_REWARD(self._obj.location, self._location.location, increase_distance = False)


class AbsoluteRotateReward(Reward):
    """Absolute reward for rotating an object towards a target orientation.

    Delegates to ``rotate_reward(object, angle)``, scoring the object's current orientation
    against the target ``angle``.
    """

    _0_ROTATE_REWARD: Callable = "rotate_reward"  # type: ignore
    _VARIABLES = {
        "gripper": Gripper,
        "object": GraspableObject,
        "angle": AbstractRotation
    }

    def __init__(self, gripper: Variable[Gripper], object: Variable[GraspableObject], angle: Variable[AbstractRotation]) -> None:
        super().__init__(gripper=gripper, object=object, angle=angle)
        self._gripper = gripper
        self._obj = object
        self._angle = angle

    def __call__(self) -> float:
        return self.__class__._0_ROTATE_REWARD(self._obj, self._angle)


class SimpleTransformReward(Reward):
    """Placeholder reward for the Transform action.

    No mapped function; ``__call__`` returns a constant ``1`` (WIP stub).
    """

    _VARIABLES = {
        "gripper": Gripper,
        "object": GraspableObject,
        "location": Location
    }

    def __init__(self, gripper: Variable[Gripper], object: Variable[GraspableObject], location: Variable[Location]) -> None:
        super().__init__(gripper=gripper, object=object, location=location)
        self._gripper = gripper
        self._obj = object
        self._location = location

    def __call__(self) -> float:
        return 1  # placeholder constant reward


class SimpleFollowReward(Reward):
    """Placeholder reward for the Follow action.

    No mapped function; ``__call__`` returns a constant ``1`` (WIP stub).
    """

    _VARIABLES = {
        "gripper": Gripper,
        "object": GraspableObject,
        "location": Location
    }

    def __init__(self, gripper: Variable[Gripper], object: Variable[GraspableObject], location: Variable[Location]) -> None:
        super().__init__(gripper=gripper, object=object, location=location)
        self._gripper = gripper
        self._obj = object
        self._location = location

    def __call__(self) -> float:
        return 1  # placeholder constant reward


##############################################
# GENERIC REWARDS
##############################################

class NearReward(Reward):
    """Generic proximity reward between two locations.

    Returns ``euclidean_distance(A, B) - near_threshold``: negative once the two are within
    the near threshold, positive while still apart.
    """

    _0_EDISTANCE_PREDICATE: Callable = "euclidean_distance"  # type: ignore
    _0_NEAR_THRESHOLD: Callable = "near_threshold"  # type: ignore

    _VARIABLES = {
        "object_A": Location,
        "object_B": Location
    }

    def __init__(self, object_A: Variable[Location], object_B: Variable[Location]) -> None:
        super().__init__(object_A=object_A, object_B=object_B)
        self._object_A = object_A
        self._object_B = object_B

    def __call__(self):
        # distance minus the near threshold (negative => within near range)
        return self.__class__._0_EDISTANCE_PREDICATE(self._object_A.location, self._object_B.location) - self.__class__._0_NEAR_THRESHOLD


##############################################
# REWARD ENUMs
##############################################


class ApproachReward(Enum):
    """Variant selector for the Approach action's reward (relative or absolute)."""

    RELATIVE_REWARD = RelativeApproachReward
    ABSOLUTE_REWARD = AbsoluteApproachReward


class WithdrawReward(Enum):
    """Variant selector for the Withdraw action's reward (relative or absolute)."""

    RELATIVE_REWARD = RelativeWithdrawReward
    ABSOLUTE_REWARD = AbsoluteWithdrawReward


class GraspReward(Enum):
    """Variant selector for the Grasp action's reward."""

    SIMPLE_REWARD = SimpleGraspReward


class DropReward(Enum):
    """Variant selector for the Drop action's reward."""

    SIMPLE_REWARD = SimpleDropReward


class MoveReward(Enum):
    """Variant selector for the Move action's reward."""

    ABSOLUTE_REWARD = AbsoluteMoveReward


class RotateReward(Enum):
    """Variant selector for the Rotate action's reward."""

    ABSOLUTE_REWARD = AbsoluteRotateReward


class TransformReward(Enum):
    """Variant selector for the Transform action's reward."""

    SIMPLE_REWARD = SimpleTransformReward


class FollowReward(Enum):
    """Variant selector for the Follow action's reward."""

    SIMPLE_REWARD = SimpleFollowReward
