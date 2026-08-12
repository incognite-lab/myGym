"""Atomic actions of the RDDL task generator.

Each :class:`AtomicAction` is itself a predicate that can be decided. In ``__init__`` it
populates three instance fields and then calls ``setup_reward()``:

- ``self._initial`` — the **precondition** (a ``LogicalOperand``); ``can_be_executed()`` is
  ``_initial.decide()``.
- ``self._predicate`` — the **goal/effect** (a ``LogicalOperand``); used as the action's
  decidable goal.
- ``self._reward`` — built by ``setup_reward()`` as ``REWARD_CLASS.value(**self.variables)``,
  i.e. the chosen reward class instantiated with the action's variables splatted as kwargs.
  The reward's ``_VARIABLES`` keys must therefore match the action's variable names.

Move, Transform and Follow are identical at the logic layer (same variables, precondition
and ``ObjectAt`` goal); they differ only in their ``REWARD_CLASS``.
"""

import dis
from enum import Enum
from typing import Callable

from rddl import AtomicAction, Reward, Variable
from rddl.entities import AbstractRotation, GraspableObject, Gripper, Location
from rddl.operators import NotOp, ParallelAndOp
from rddl.predicates import (Exists, GripperAt, GripperOpen, IsHolding, IsReachable,
                             Near, ObjectAt, ObjectAtPose)
from rddl.rewards import (ApproachReward, DropReward, FollowReward, GraspReward, MoveReward,
                          RotateReward, TransformReward, WithdrawReward)


""" HOW TO DEFINE ATOMIC ACTION

class <Some>AtomicAction(AtomicAction):

    _VARIABLES = {
        "<variable_name>": <variable_type>,
        ...
    }

    def __init__(self, **kwds) -> None:  # do not take _VARIABLES in __init__
        super().__init__(**kwds)
        self._predicate: LogicalOperand  # goal condition
        self._initial: LogicalOperand  # initial condition
        self._reward: Operand/Reward  # associated reward

"""


class Approach(AtomicAction):
    """Move an open gripper to a reachable object.

    Precondition: ``IsReachable(g, obj) AND GripperOpen(g) AND NOT GripperAt(g, obj)``.
    Effect (goal): ``GripperAt(g, obj)``.
    Reward: ``ApproachReward.RELATIVE_REWARD`` (delta-distance shaping with open bonus).
    """

    _VARIABLES = {
        "gripper": Gripper,
        "object": GraspableObject
    }
    REWARD_CLASS: Enum = ApproachReward.RELATIVE_REWARD

    def __init__(self, **kwds) -> None:
        super().__init__(**kwds)
        gripper = self.get_argument("gripper")
        obj = self.get_argument("object")
        self._predicate = GripperAt(gripper=gripper, object=obj)  # goal: gripper reaches the object
        # precondition: object reachable, gripper open, and not already at the object
        self._initial = ParallelAndOp(
            left=IsReachable(gripper=gripper, location=obj),
            right=ParallelAndOp(left=GripperOpen(gripper=gripper), right=NotOp(operand=self._predicate))
        )
        self.setup_reward()
        # self._reward = self.REWARD_CLASS.value(gripper, obj)


class Withdraw(AtomicAction):
    """Move an open gripper away from an object it is currently at.

    Precondition: ``GripperAt(g, obj) AND GripperOpen(g)``.
    Effect (goal): ``NOT GripperAt(g, obj)``.
    Reward: ``WithdrawReward.RELATIVE_REWARD`` (delta-distance shaping, rewards moving away).
    """

    _VARIABLES = {
        "gripper": Gripper,
        "object": GraspableObject
    }
    REWARD_CLASS: Enum = WithdrawReward.RELATIVE_REWARD

    def __init__(self, **kwds) -> None:
        super().__init__(**kwds)
        gripper = self.get_argument("gripper")
        obj = self.get_argument("object")
        gripper_at = GripperAt(gripper=gripper, object=obj)
        # precondition: gripper is at the object and open
        self._initial = ParallelAndOp(
            left=gripper_at,
            right=GripperOpen(gripper=gripper)
        )
        self._predicate = NotOp(operand=gripper_at)  # goal: gripper no longer at the object
        self.setup_reward()


class Grasp(AtomicAction):
    """Close an open gripper around an object it is at, to hold it.

    Precondition: ``GripperAt(g, obj) AND GripperOpen(g)``.
    Effect (goal): ``IsHolding(g, obj) AND NOT GripperOpen(g)``.
    Reward: ``GraspReward.SIMPLE_REWARD`` (rewards closing the gripper on the object).
    """

    _VARIABLES = {
        "gripper": Gripper,
        "object": GraspableObject
    }
    REWARD_CLASS: Enum = GraspReward.SIMPLE_REWARD

    def __init__(self, **kwds) -> None:
        super().__init__(**kwds)
        gripper = self.get_argument("gripper")
        obj = self.get_argument("object")
        gripper_open = GripperOpen(gripper=gripper)
        # precondition: gripper at the object and open
        self._initial = ParallelAndOp(
            left=GripperAt(gripper=gripper, object=obj),
            right=gripper_open
        )
        # goal: holding the object with the gripper now closed
        self._predicate = ParallelAndOp(
            left=IsHolding(gripper=gripper, object=obj),
            right=NotOp(operand=gripper_open)
        )
        self.setup_reward()


class Drop(AtomicAction):
    """Open the gripper to release a held object.

    Precondition: ``GripperAt(g, obj) AND NOT (NOT IsHolding(g, obj) AND GripperOpen(g))``
    (i.e. at the object and not already in the released-and-open state).
    Effect (goal): ``NOT IsHolding(g, obj) AND GripperOpen(g)``.
    Reward: ``DropReward.SIMPLE_REWARD`` (rewards opening the gripper to release).
    """

    _VARIABLES = {
        "gripper": Gripper,
        "object": GraspableObject
    }
    REWARD_CLASS: Enum = DropReward.SIMPLE_REWARD

    def __init__(self, **kwds) -> None:
        super().__init__(**kwds)
        gripper = self.get_argument("gripper")
        obj = self.get_argument("object")
        # released-and-open state: object no longer held and gripper open
        gripper_open_and_released = ParallelAndOp(
            left=NotOp(operand=IsHolding(gripper=gripper, object=obj)),
            right=GripperOpen(gripper=gripper)
        )
        # precondition: at the object and not yet in the released-and-open state
        self._initial = ParallelAndOp(
            left=GripperAt(gripper=gripper, object=obj),
            right=NotOp(operand=gripper_open_and_released)
        )
        self._predicate = gripper_open_and_released  # goal: object released, gripper open
        self.setup_reward()


class Move(AtomicAction):
    """Carry a held object to a reachable target location.

    Precondition: ``IsHolding(g, obj) AND IsReachable(g, loc)``.
    Effect (goal): ``ObjectAt(obj, loc)``.
    Reward: ``MoveReward.ABSOLUTE_REWARD`` (object-to-location distance).
    """

    _VARIABLES = {
        "gripper": Gripper,
        "object": GraspableObject,
        "location": Location
    }

    REWARD_CLASS: Enum = MoveReward.ABSOLUTE_REWARD

    def __init__(self, **kwds) -> None:
        super().__init__(**kwds)
        gripper = self.get_argument("gripper")
        obj = self.get_argument("object")
        loc = self.get_argument("location")

        # precondition: holding the object and the target location is reachable
        self._initial = ParallelAndOp(
            left=IsHolding(gripper=gripper, object=obj),
            right=IsReachable(gripper=gripper, location=loc)
        )
        self._predicate = ObjectAt(object=obj, location=loc)  # goal: object at the location
        self.setup_reward()


class Rotate(AtomicAction):
    """Rotate a held object to a target orientation.

    Precondition: ``IsHolding(g, obj) AND Exists(angle)``.
    Effect (goal): ``ObjectAtPose(obj, angle)``.
    Reward: ``RotateReward.ABSOLUTE_REWARD`` (orientation-vs-target via ``rotate_reward``).
    """

    _VARIABLES = {
        "gripper": Gripper,
        "object": GraspableObject,
        "angle": AbstractRotation
    }

    REWARD_CLASS: Enum = RotateReward.ABSOLUTE_REWARD

    def __init__(self, **kwds) -> None:
        super().__init__(**kwds)
        gripper = self.get_argument("gripper")
        obj = self.get_argument("object")
        angle = self.get_argument("angle")

        # precondition: holding the object and the target angle exists
        self._initial = ParallelAndOp(
            left=IsHolding(gripper=gripper, object=obj),
            right=Exists(entity=angle)
        )
        self._predicate = ObjectAtPose(object=obj, angle=angle)  # goal: object at target orientation
        self.setup_reward()  #TODO: fix this, second obj should be dummy obj or angle should be used instead


class Transform(AtomicAction):
    """Bring a held object to a reachable target location.

    Logically identical to :class:`Move` (same variables, precondition and goal); it differs
    only in the reward used.
    Precondition: ``IsHolding(g, obj) AND IsReachable(g, loc)``.
    Effect (goal): ``ObjectAt(obj, loc)``.
    Reward: ``TransformReward.SIMPLE_REWARD`` (placeholder constant 1).
    """

    _VARIABLES = {
        "gripper": Gripper,
        "object": GraspableObject,
        "location": Location
    }

    REWARD_CLASS: Enum = TransformReward.SIMPLE_REWARD

    def __init__(self, **kwds) -> None:
        super().__init__(**kwds)
        gripper = self.get_argument("gripper")
        obj = self.get_argument("object")
        loc = self.get_argument("location")

        # precondition: holding the object and the target location is reachable
        self._initial = ParallelAndOp(
            left=IsHolding(gripper=gripper, object=obj),
            right=IsReachable(gripper=gripper, location=loc)
        )
        self._predicate = ObjectAt(object=obj, location=loc)  # goal: object at the location
        self.setup_reward()


class Follow(AtomicAction):
    """Bring a held object to a reachable target location, following a path.

    Logically identical to :class:`Move`/:class:`Transform` (same variables, precondition and
    goal); it differs only in the reward used.
    Precondition: ``IsHolding(g, obj) AND IsReachable(g, loc)``.
    Effect (goal): ``ObjectAt(obj, loc)``.
    Reward: ``FollowReward.SIMPLE_REWARD`` (placeholder constant 1).
    """

    _VARIABLES = {
        "gripper": Gripper,
        "object": GraspableObject,
        "location": Location
    }

    REWARD_CLASS: Enum = FollowReward.SIMPLE_REWARD

    def __init__(self, **kwds) -> None:
        super().__init__(**kwds)
        gripper = self.get_argument("gripper")
        obj = self.get_argument("object")
        loc = self.get_argument("location")

        # precondition: holding the object and the target location is reachable
        self._initial = ParallelAndOp(
            left=IsHolding(gripper=gripper, object=obj),
            right=IsReachable(gripper=gripper, location=loc)
        )
        self._predicate = ObjectAt(object=obj, location=loc)  # goal: object at the location
        self.setup_reward()
