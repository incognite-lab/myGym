"""Concrete predicates over domain entities.

Each predicate declares its typed arguments in `_VARIABLES`, registers one or
more simulator capabilities via `_0_*` string keys (resolved to callables at
class definition; see SPECIFICATION section 3.5), and implements `__call__`
returning a bool. Only `decide()` (the cached `__call__`) is a legal evaluation
channel for predicates; `evaluate()` is not used here.

Some predicates pass resolved `.location` coordinates to their external
function (`Near`, `ObjectAt`); others pass the entity objects themselves
(`GripperAt`, `IsHolding`, ...) and rely on the simulator function to resolve
them.
"""
from typing import Any, Callable, ClassVar, Union

from rddl import Predicate, Variable
from rddl.core import Entity
from rddl.entities import (AbstractRotation, GraspableObject, Gripper, Location, LocationType,
                           ObjectEntity)


class SpatialPredicate(Predicate):
    """Semantic marker base for predicates about spatial relations (adds no behavior)."""

    def __init__(self, **kwds) -> None:
        """Forward to `Predicate`."""
        super().__init__(**kwds)


class Near(SpatialPredicate):
    """True when two locations are within `near_threshold` Euclidean distance."""

    _0_EDISTANCE_DISTANCE: ClassVar[Union[Callable, str]] = "euclidean_distance"
    _0_NEAR_THRESHOLD: ClassVar[Union[float, str]] = "near_threshold"
    _VARIABLES = {"object_A": Location, "object_B": Location}

    # def __init__(self, object_A: Variable[LocationType], object_B: Variable[LocationType]) -> None:
    def __init__(self, **kwds) -> None:
        """Bind `object_A`, `object_B` via `Predicate`."""
        super().__init__(**kwds)

    def __call__(self):
        """Return ``dist(object_A.location, object_B.location) < near_threshold``."""
        return Near._0_EDISTANCE_DISTANCE(self.object_A.location, self.object_B.location) < Near._0_NEAR_THRESHOLD


class GripperAt(SpatialPredicate):
    """True when the gripper is positioned at the given graspable object."""

    _0_GRIPPER_AT_CHECK: ClassVar[Union[Callable, str]] = "gripper_at"
    _VARIABLES = {"gripper": Gripper, "object": GraspableObject}

    def __init__(self, **kwds) -> None:
        """Bind `gripper`, `object` via `Predicate`."""
        super().__init__(**kwds)

    def __call__(self):
        """Return ``gripper_at(gripper, object)`` (entities passed through)."""
        return GripperAt._0_GRIPPER_AT_CHECK(self.gripper, self.object)


class GripperOpen(Predicate):
    """True when the gripper is open."""

    _0_GRIPPER_OPEN_CHECK: ClassVar[Union[Callable, str]] = "gripper_open"
    _VARIABLES = {"gripper": Gripper}

    def __init__(self, **kwds) -> None:
        """Bind `gripper` via `Predicate`."""
        super().__init__(**kwds)

    def __call__(self):
        """Return ``gripper_open(gripper)``."""
        return GripperOpen._0_GRIPPER_OPEN_CHECK(self.gripper)


class IsReachable(SpatialPredicate):
    """True when the location is within the gripper's reach."""

    _0_REACHABLE_TEST: ClassVar[Union[Callable, str]] = "is_reachable"
    _VARIABLES = {"gripper": Gripper, "location": Location}

    def __init__(self, **kwds) -> None:
        """Bind `gripper`, `location` via `Predicate`."""
        super().__init__(**kwds)

    def __call__(self):
        """Return ``is_reachable(gripper, location)`` (entities passed through)."""
        return IsReachable._0_REACHABLE_TEST(self.gripper, self.location)


class IsHolding(Predicate):
    """True when the gripper is holding the object."""

    _0_IS_HOLDING_FUNCTION: ClassVar[Union[Callable, str]] = "is_holding"
    _VARIABLES = {"gripper": Gripper, "object": ObjectEntity}

    def __init__(self, **kwds) -> None:
        """Bind `gripper`, `object` via `Predicate`."""
        super().__init__(**kwds)

    def __call__(self) -> Any:
        """Return ``is_holding(gripper, object)`` (entities passed through)."""
        return IsHolding._0_IS_HOLDING_FUNCTION(self.gripper, self.object)


class ObjectAt(SpatialPredicate):
    """True when the object is at the given location."""

    _0_OBJECT_AT_CHECK: ClassVar[Union[Callable, str]] = "object_at"
    _VARIABLES = {"object": ObjectEntity, "location": Location}

    def __init__(self, **kwds) -> None:
        """Bind `object`, `location` via `Predicate`."""
        super().__init__(**kwds)

    def __call__(self):
        """Return ``object_at(object, location)`` (entities passed through)."""
        return ObjectAt._0_OBJECT_AT_CHECK(self.object, self.location)


class ObjectAtPose(SpatialPredicate):
    """True when the object is at the given rotation/pose."""

    _0_OBJECT_AT_POSE_CHECK: ClassVar[Union[Callable, str]] = "object_at_pose"
    _VARIABLES = {"object": ObjectEntity, "angle": AbstractRotation}

    def __init__(self, **kwds) -> None:
        """Bind `object`, `angle` via `Predicate`."""
        super().__init__(**kwds)

    def __call__(self):
        """Return ``object_at_pose(object, angle)`` (entities passed through)."""
        return ObjectAtPose._0_OBJECT_AT_POSE_CHECK(self.object, self.angle)


class Exists(Predicate):
    """True when the given entity exists in the world."""

    _0_EXISTS_CHECK: ClassVar[Union[Callable, str]] = "exists"
    _VARIABLES = {"entity": Entity}

    def __init__(self, **kwds) -> None:
        """Bind `entity` via `Predicate`."""
        super().__init__(**kwds)

    def __call__(self):
        """Return ``exists(self.entity)``."""
        return Exists._0_EXISTS_CHECK(self.entity)


class OnTop(SpatialPredicate):
    """True when `object_A` rests on top of `object_B`."""

    _0_ON_TOP_CHECK: ClassVar[Union[Callable, str]] = "on_top"
    _VARIABLES = {"object_A": ObjectEntity, "object_B": ObjectEntity}

    def __init__(self, **kwds) -> None:
        """Bind `object_A`, `object_B` via `Predicate`."""
        super().__init__(**kwds)

    def __call__(self):
        """Return ``on_top(object_A, object_B)`` (entities passed through)."""
        return OnTop._0_ON_TOP_CHECK(self.object_A, self.object_B)
