"""Domain entity types for the RDDL task generator.

Entities are the typed nouns that predicates and operators bind to (grippers,
graspable objects, locations, rotations). Most concrete, simulator-backed
entities are created by multiple inheritance of one of these RDDL bases with a
geometry-providing mixin (see SPECIFICATION sections 8 and 11), which supplies
the concrete `_get_location`. Bases that can stand alone auto-generate a unique
`reference` string (``loc_N``, ``angle_N``) when none is given.
"""
from abc import abstractmethod
from typing import Iterable, Optional, TypeVar

import numpy as np

from rddl import Entity, Predicate


class Location(Entity):
    """Entity that has a spatial position, exposed via the `location` property.

    `_get_location` is abstract; concrete subclasses (or a geometry mixin)
    supply it. The base `Location` itself is instantiable and auto-names
    unbound instances ``loc_N``.
    """

    def __init__(self, reference: Optional[str] = None, **kw):
        """Auto-assign a ``loc_N`` reference for a bare `Location` when none is given."""
        # Only the base class auto-names; subclasses are expected to manage their own reference.
        if reference is None and self.__class__ is Location:
            reference = f"loc_{Location._get_reference_count()}"
        super().__init__(reference, **kw)

    @abstractmethod
    def _get_location(self):
        """Return this entity's position. Must be implemented by a subclass/mixin."""
        raise NotImplementedError(f"_get_location is not implemented for {self.__class__}")

    @property
    def location(self) -> Iterable[float]:
        """The entity's spatial coordinates (delegates to `_get_location`)."""
        return self._get_location()


class Angle(Entity):
    """A scalar orientation. Its `location` accessor returns the stored angle.

    Reuses the `location` property name so angle entities are interchangeable
    with positional ones in predicates expecting a `.location`.
    """

    def __init__(self, reference: Optional[str] = None, angle: float=0.0, **kw):
        """Store `angle` and auto-name a bare `Angle` instance ``angle_N``."""
        if reference is None and self.__class__ is Angle:
            reference = f"angle_{Angle._get_reference_count()}"
        super().__init__(reference, **kw)
        self._angle = angle

    @property
    def location(self) -> float:
        """The stored scalar angle (named `location` for predicate compatibility)."""
        return self._angle


class ObjectEntity(Location):
    """A positioned object that additionally carries a human-readable `name`.

    `_get_location` remains abstract here; a concrete implementation comes from
    a simulator geometry mixin.
    """

    def __init__(self, reference: str, name: Optional[str] = None, **kw):
        """Bind `reference`; default `name` to `reference` when not supplied."""
        super().__init__(reference, **kw)
        self._name = name if name is not None else reference

    @property
    def name(self):
        """The object's display name (falls back to its reference)."""
        return self._name


class GraspableObject(ObjectEntity):
    """An object a gripper can pick up (apples, boxes, ...)."""

    def __init__(self, reference: str, name: Optional[str] = None, **kw):
        """Construct a graspable object; see `ObjectEntity`."""
        super().__init__(reference, name, **kw)


class Table(ObjectEntity):
    """A fixed, non-graspable surface that other objects can rest on top of.

    Provides its own concrete `_get_location` (a fixed origin) since it has no
    simulator geometry mixin of its own.
    """

    def __init__(self, reference: Optional[str] = None, name: Optional[str] = None, **kw):
        """Auto-name a bare `Table` instance ``table_N`` when no reference is given."""
        super().__init__(self._get_generic_reference() if reference is None else reference,
                         name if name is not None else "table", **kw)

    def _get_location(self):
        """Fixed at the origin; tables aren't expected to move."""
        return np.zeros(3)


class Gripper(Location):
    """An end-effector entity that can report whether it is holding an object."""

    def __init__(self, reference: Optional[str] = None, **kw):
        """Construct a gripper; `_is_holding_predicate` is bound externally."""
        super().__init__(reference, **kw)
        # Annotation only: the actual IsHolding predicate is injected elsewhere.
        self._is_holding_predicate: Predicate

    def is_holding(self, obj: ObjectEntity) -> bool:
        """Return whether this gripper currently holds `obj`, via its bound predicate."""
        return self._is_holding_predicate(obj)


class AbstractLocation(Location):
    """A `Location` whose `value` is undefined until a concrete subclass supplies it."""

    def __init__(self, reference: Optional[str] = None):
        """Construct an abstract location with an optional reference."""
        super().__init__(reference)

    @property
    def value(self):
        """Placeholder; raises until overridden by a concrete subclass."""
        raise NotImplementedError


class AbstractRotation(Entity):
    """A rotation entity with a `value` to be provided by a subclass. `reference` is required."""

    def __init__(self, reference: str):
        """Construct an abstract rotation with a required reference."""
        super().__init__(reference)

    @property
    def value(self):
        """Placeholder; raises until overridden by a concrete subclass."""
        raise NotImplementedError


class RandomRotation(AbstractRotation):
    """An `AbstractRotation` whose `value` is a fixed angle drawn uniformly from (-pi, pi)."""

    def __init__(self, reference: str):
        """Sample and store a random angle in (-pi, pi)."""
        super().__init__(reference)
        self._angle = np.random.uniform(-np.pi, np.pi)

    @property
    def value(self):
        """The sampled rotation angle in radians."""
        return self._angle


# Bound type var for generic `Variable[LocationType]` annotations across the library.
LocationType = TypeVar("LocationType", bound=Location)
