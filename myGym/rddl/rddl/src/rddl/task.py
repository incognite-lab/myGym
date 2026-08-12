"""Legacy/abstract task skeleton.

Defines an interface-only ``RDDLTask`` (actions, objects, initial/goal state) whose
accessors raise ``NotImplementedError``. This is *not* the task produced by the sampler;
see ``rddl.rddl_task.RDDLTask`` for the concrete implementation.
"""
from rddl import Entity, AtomicAction, LogicalOperand


class RDDLTask:
    """Abstract task skeleton: an ordered action list with objects and initial/goal states."""

    def __init__(self, actions: list[AtomicAction], objects: list[Entity], initial_state: LogicalOperand, goal_state: LogicalOperand) -> None:
        """Store the action sequence, objects, and the initial and goal logical states."""
        self._actions = actions
        self._objects = objects
        self._initial_state = initial_state
        self._goal_state = goal_state

    def current_action(self) -> AtomicAction:
        """Return the currently active action. Unimplemented."""
        raise NotImplementedError

    def next_action(self) -> AtomicAction:
        """Advance to and return the next action. Unimplemented."""
        raise NotImplementedError

    def current_reward(self) -> float:
        """Return the reward for the current action. Unimplemented."""
        raise NotImplementedError
