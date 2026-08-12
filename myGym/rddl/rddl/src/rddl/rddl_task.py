"""Concrete task data structure produced by the sampler.

``RDDLTask`` bundles a sampled action sequence with its per-step symbolic states and the
initial state, plus an optional back-reference to the generating ``RDDLWorld`` (enabling
object re-sampling). Objects are *derived* from the final state's variables, not stored
separately. Provides goal/reward accessors and a playback generator over the sequence.
"""
import stat
from typing import Generator, Iterable, Iterator, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from rddl.rddl_sampler import RDDLWorld

import numpy as np
from rddl import Entity, AtomicAction, LogicalOperand
from rddl.core import CacheMode, Operand, Variable
from rddl import rddl_sampler
from rddl.sampling_utils import StrictSymbolicCacheContainer, SymbolicCacheContainer


class RDDLTask:
    """A sampled task: an ordered action sequence with its symbolic states and cursors.

    Holds the action list, the per-step post-states, the initial state, and an optional
    world generator used to re-sample concrete objects. Tracks the current action/state.
    """

    def __init__(self, actions: list[AtomicAction], states: list[StrictSymbolicCacheContainer], initial_state: StrictSymbolicCacheContainer, world_generator: 'Optional[RDDLWorld]' = None) -> None:
        """Store actions/states and initialise the current-action/state cursors.

        ``_final_state`` is the last per-step state; the cursors start at the first action
        and the initial state. ``world_generator`` is kept for later object re-sampling.
        """
        self._actions = actions
        self._states = states
        self._world_generator = world_generator

        self._initial_state: StrictSymbolicCacheContainer = initial_state
        self._final_state: StrictSymbolicCacheContainer = states[-1]
        self._current_action: Optional[AtomicAction] = actions[0]
        self._current_state: Optional[StrictSymbolicCacheContainer] = initial_state

    def gather_objects(self) -> list[Variable]:
        """Return the task's objects, derived from the final state's bound variables."""
        return list(self._final_state.variables.values())

    @property
    def current_action(self) -> Optional[AtomicAction]:
        """The action at the current cursor position."""
        # FIXME: should probably return the last executable action?
        # for action in self._actions:
        #     if action.can_be_executed():
        #         return action
        return self._current_action

    def regenerate_objects(self) -> None:
        """Re-sample concrete objects against a fresh symbolic container.

        Requires a world generator. Temporarily swaps in a new symbolic cache, relinks each
        action's variables (sampling new object subclasses), re-asserts initial/predicate
        values, sets the result as ``_final_state``, then restores the previous cache.
        """
        if self._world_generator is None:
            raise RuntimeError("Cannot regenerate objects without a world generator!")

        # Preserve the global cache state so this re-sampling is non-destructive.
        previous_cache_mode = Operand.get_cache_mode()
        previous_cache = Operand.get_cache()

        new_state = StrictSymbolicCacheContainer()
        Operand.set_cache_symbolic(new_state)
        for a_idx, (action, state) in enumerate(zip(self._actions, self._states)):
            a_vars = action.gather_variables()
            added_vars = new_state.lookup_and_link_variables(a_vars, self._world_generator.sample_object_subclass)
            if a_idx > 0:
                # Only newly added variables get a fresh initial assertion after the first action.
                action.initial.set_symbolic_value(True, set(added_vars))
            else:
                action.initial.set_symbolic_value(True)
            # print(f"regenerating {action} with {new_state}")
            action.predicate.set_symbolic_value(True)

        self._final_state = new_state

        if previous_cache_mode == CacheMode.SYMBOLIC and isinstance(previous_cache, SymbolicCacheContainer):
            Operand.set_cache_symbolic(previous_cache)
        else:
            Operand.set_cache_normal(previous_cache)

    def get_actions(self) -> Iterator[AtomicAction]:
        """Return an iterator over the task's actions (no cursor side effects)."""
        return iter(self._actions)

    def get_generator(self) -> Generator[AtomicAction, bool, bool]:
        """Yield each action in order for playback, advancing the current-action/state cursors."""
        for action, state in zip(self._actions, self._states):
            self._current_action = action
            self._current_state = state
            yield action
        return True

    # def next_action(self) -> AtomicAction:
    #     raise NotImplementedError

    @property
    def global_goal(self) -> LogicalOperand:
        """The task's overall goal: the predicate of the last action."""
        return self._actions[-1].predicate

    @property
    def current_goal(self) -> LogicalOperand:
        """The predicate of the current action."""
        if self._current_action is None:
            raise RuntimeError("No current action")
        return self._current_action.predicate

    @property
    def current_reward(self) -> float:
        """Reward of the current action (``NaN`` if there is none)."""
        ca = self.current_action
        if ca is None:
            return np.nan
        return ca.compute_reward()

    def show_current_state(self) -> None:
        """Print the current symbolic state as a table."""
        if self._current_state is None:
            raise RuntimeError("No current state")
        print(self._current_state.show_table())

    @property
    def initial_state(self) -> StrictSymbolicCacheContainer:
        """The task's initial symbolic state."""
        return self._initial_state

    @property
    def final_state(self) -> StrictSymbolicCacheContainer:
        """The task's final symbolic state (post-state of the last action)."""
        return self._final_state
