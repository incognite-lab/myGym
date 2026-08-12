"""RDDL generation engine.

This module is the constructive forward sampler that turns a declarative domain
(entities, predicates, STRIPS-like actions) into concrete action sequences.

The central idea: **the world state is exactly the symbolic decide-cache**. RDDL is
not a classical planner searching over a propositional state; instead it rides on the
``StrictSymbolicCacheContainer`` truth table (a map ``predicate(args) -> bool``, closed
world so anything absent decides ``False``). An action's precondition is not *matched*
against the state -- it is **made true** by writing it into the cache via
``set_symbolic_value(True)``; calling ``decide()`` afterwards acts as a consistency
check (``And``/``Not`` decide over their children's cached values, so the root returns
``False`` iff some child was already forced to a contradicting value by an earlier
action's effect). The effect is applied by mutating the cache so the next action
observes the post-state. A ``deque`` of cache clones is the backtracking stack.

Public surface:
    - ``RDDLWorld`` -- the sampler (``sample_generator`` / ``sample_world`` for a single
      weighted-random sequence, ``recurse_generator`` for DFS enumeration of many tasks).
    - ``RDDL`` -- a thin facade wiring up the textual DSL parser and action definitions.
"""

from collections import defaultdict, deque
from copy import deepcopy
from functools import lru_cache
from html import entities
from queue import Empty, Queue
from threading import Lock, Thread
from typing import Callable, ClassVar, Generator, Iterable, Optional, Type, Union
from warnings import warn

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike, NDArray
#from testing_utils import Apple

from rddl import AtomicAction, Operand, Variable
from rddl.actions import Approach, Drop, Follow, Grasp, Move, Rotate, Withdraw, Transform
from rddl.core import CacheMode, Entity, LogicalOperand, Predicate, SymbolicCacheContainer
from rddl.entities import GraspableObject, Gripper, ObjectEntity
from rddl.predicates import IsReachable, Near
from rddl.rddl_parser import (EntityType, OperatorType, PredicateType,
                              RDDLParser)

from rddl.rule_book import RuleBook
from rddl.sampling_utils import StrictSymbolicCacheContainer, SymbolicEntity, Weighter
from rddl.rddl_task import RDDLTask


SEED = np.random.randint(0, 2**32)


class RDDLWorld:
    """Constructive forward sampler of consistent action sequences.

    Holds a stack of symbolic-cache clones (``_symbolic_table_stack``; the top is the
    current world state), a ``RuleBook`` for cross-predicate consistency, a ``Weighter``
    for weighted/sequence-aware action ordering, the allowed entity/action pools, and
    ``__real_cache`` which saves the normal compute-cache while symbolic mode is active.

    By default ``Withdraw`` is excluded from the initial action set (it requires a prior
    ``Approach``). Generation proceeds step-by-step: branch the cache, bind the action's
    free variables, force/check the precondition, apply the effect, and either keep the
    branch (accepted) or pop it and re-weight (rejected).
    """

    ROBOTS: NDArray = np.asanyarray([Gripper])
    OBJECTS: NDArray = np.asanyarray([ObjectEntity])
    # VALID_INITIAL_ACTIONS: NDArray = np.asanyarray([Approach])
    VALID_INITIAL_ACTIONS: NDArray = np.asanyarray([Approach, Grasp, Drop, Move, Rotate, Follow, Transform])
    VALID_ACTIONS: NDArray = np.asanyarray([Approach, Withdraw, Grasp, Drop, Move, Rotate, Follow, Transform])
    RNG = np.random.default_rng(SEED)

    STATE_IDLE = 1
    STATE_GENERATING = 2

    def __init__(self,
                 allowed_entities: Optional[Iterable[Type[Entity]]] = None,
                 allowed_actions: Optional[Iterable[Type[AtomicAction]]] = None,
                 allowed_initial_actions: Optional[Iterable[Type[AtomicAction]]] = None,
                 sample_single_object_per_class: bool = False,
                 action_weights: Optional[list[float]] = None,
                 object_weights: Optional[dict[Type[Entity], float]] = None
                 ) -> None:
        """Configure the sampler's entity/action pools and weighting.

        Args:
            allowed_entities: concrete entity classes the sampler may instantiate; ``None``
                means all subclasses are allowed.
            allowed_actions: action classes usable at any step; defaults to ``VALID_ACTIONS``.
            allowed_initial_actions: action classes legal as the first step; defaults to
                ``VALID_INITIAL_ACTIONS`` (excludes ``Withdraw``).
            sample_single_object_per_class: if true, an object class's weight drops to 0
                once used (one object per class); else it merely decays.
            action_weights: per-action base weights; must match ``allowed_actions`` length.
            object_weights: per-class sampling weights (default 1.0 for any class).
        """
        self._reset_symbolic_table_stack()

        self.__real_cache = None

        self._rule_book = RuleBook()

        Weighter.set_rng(self.RNG)

        self._allowed_entities: Optional[NDArray]
        self._allowed_actions: NDArray
        self._allowed_initial_actions: NDArray
        if allowed_entities is not None:
            self._allowed_entities = np.asanyarray(allowed_entities)
        else:
            self._allowed_entities = None
        if allowed_actions is None:
            self._allowed_actions = self.VALID_ACTIONS
        else:
            self._allowed_actions = np.asanyarray(allowed_actions)
        if allowed_initial_actions is None:
            self._allowed_initial_actions = self.VALID_INITIAL_ACTIONS
        else:
            self._allowed_initial_actions = np.asanyarray(allowed_initial_actions)
        self._sample_single_object_per_class = sample_single_object_per_class

        self._object_weights = defaultdict(lambda: 1.0)
        if object_weights is not None:
            self._object_weights.update(object_weights)

        if action_weights is None:
            action_weights = [1.0] * len(self._allowed_actions)
        else:
            if len(action_weights) != len(self._allowed_actions):
                raise ValueError("Length of action weights must be equal to the number of allowed actions!")

        self._weighter = Weighter(
            items=self._allowed_actions,
            weights=action_weights,
            initial_weights=[1.0 if a in self._allowed_initial_actions else 0.0 for a in self._allowed_actions],
        )
        self.full_reset()

    @property
    def rule_book(self) -> RuleBook:
        """The ``RuleBook`` enforcing cross-predicate consistency constraints."""
        return self._rule_book

    def sample_generator(self,
                         sequence_length: int = 2,
                         add_robots: bool = True,
                         retry_ad_infinitum: bool = True) -> Generator[tuple[AtomicAction, StrictSymbolicCacheContainer], bool, bool]:
        """Yield ``(action, post_state_snapshot)`` for each accepted step of one sequence.

        A Python generator that constructs a single consistent action sequence of length
        ``sequence_length`` on the symbolic cache. Per step it branches the cache, picks a
        weighted-random candidate action, binds its free variables, forces/checks the
        precondition, applies the effect, and yields a clone of the resulting world state.

        Args:
            sequence_length: number of actions to produce.
            add_robots: if true, seed the world with one ``Gripper`` variable.
            retry_ad_infinitum: if true (the ``Weighter`` flag), the action stream cycles
                forever so a step never hard-fails; if false, each choice list is traversed
                once.

        Yields:
            ``(action, world_state_clone)`` after the action's effect has been applied.

        Returns:
            ``True`` once ``sequence_length`` steps were produced.

        Raises:
            RuntimeError: if a generator is already running, or the initial action space is
                empty.
        """
        # re-entrancy guard: only one generation may run on this world at a time
        if self.__state != self.STATE_IDLE:
            raise RuntimeError("Generator is already running!")

        self._initialize_world(add_robots=add_robots)  # fresh symbolic world + gripper
        action: AtomicAction
        added_vars: list[Variable] = []

        sample_idx = 0
        self._initial_world_state: StrictSymbolicCacheContainer
        self._goal_world_state: StrictSymbolicCacheContainer

        Weighter.RETRY_AD_INFINITUM = retry_ad_infinitum

        # sample random action
        while sample_idx < sequence_length:
            # step 0 draws only from the legal initial actions; later steps use the
            # weighted, sequence-aware stream that learns to avoid recent/failed transitions
            action_generator = self._weighter.get_random_generator() if sample_idx > 0 else self._weighter.get_initial_generator()
            yield_accepted = False
            while not yield_accepted:
                if sample_idx == 0:
                    # --- establish a consistent INITIAL state (free world, no prior facts) ---
                    while True:
                        try:
                            action = next(action_generator)
                        except StopIteration:
                            # empty initial action space is the only fatal generation error
                            raise RuntimeError("Failed to sample initial action! This should never happen. Check if the initial action space is not empty.")
                        action_variables = action.gather_variables()  # the action's empty slots
                        self._symbolic_cache_duplicate_and_stack()    # branch off current world
                        # bind each slot to a reused or freshly created world entity
                        added_vars = self._symbolic_table.lookup_and_link_variables(action_variables, self.sample_object_subclass)
                        action.initial.set_symbolic_value(True)       # force the WHOLE precondition true
                        if action.initial.decide():                   # consistency check
                            # FIXME: hot fix to make things reachable
                            # the only step-0 safety net: graspables must be reachable or
                            # later grasp/move preconditions can never be satisfied
                            for v in added_vars:
                                if issubclass(v.base_type, GraspableObject):
                                    IsReachable(gripper=self._gripper, location=v).set_symbolic_value(True)
                            if not action.initial.decide():
                                raise RuntimeError(f"Making objects reachable somehow broke initial condition of {action}!!!")
                            self._initial_world_state = self._symbolic_table.clone()  # snapshot the initial state
                            break
                        # rejected: roll back bindings + branch and downweight this initial action
                        self._remove_variables(added_vars)
                        self._weighter.penalize_initial(action)
                        self._symbolic_cache_pop()
                else:
                    # --- step > 0: precondition must agree with accumulated prior effects ---
                    while True:
                        action = next(action_generator)
                        action_variables = action.gather_variables()
                        self._symbolic_cache_duplicate_and_stack()    # branch off current world
                        added_vars = self._symbolic_table.lookup_and_link_variables(action_variables, self.sample_object_subclass)
                        if added_vars:
                            # only freely assert facts about NEWLY created objects; facts about
                            # pre-existing objects must already hold from earlier effects
                            action.initial.set_symbolic_value(True, set(added_vars))
                        if action.initial.decide():                   # precondition vs. prior state
                            break
                        # TODO: cleanup if initial still not true (clone sym. table and delete)
                        # rejected: downweight the transition and roll back the branch
                        self._weighter.penalize_bad(action)
                        self._remove_variables(added_vars)
                        self._symbolic_cache_pop()

                # tentatively apply the effect, then reject the step if it would violate a
                # rule-book exclusivity constraint (a no-op when no such rules are registered)
                action.predicate.set_symbolic_value(True)
                if self._rule_book.check_if_breaks_consistency():
                    if sample_idx > 0:
                        self._weighter.penalize_bad(action)
                    else:
                        self._weighter.penalize_initial(action)
                    self._remove_variables(added_vars)
                    self._symbolic_cache_pop()
                    continue   # re-draw a candidate for this same step
                yield_accepted = True

            # emit outside symbolic mode so the consumer can inspect/evaluate normally
            self.deactivate_symbolic_mode()
            yield action, self._symbolic_table.clone()   # (action, post-effect state snapshot)
            self.activate_symbolic_mode()

            # learn transition weights from the accepted action; branch is kept (not popped)
            # so it becomes the base world for the next step
            if sample_idx > 0:
                self._weighter.add_and_penalize(action)
            else:
                self._weighter.add_and_penalize_initial(action)
            sample_idx += 1

        # self._symbolic_table.show_table()
        self._goal_world_state = self._symbolic_table.clone()  # final cache = goal state
        self.deactivate_symbolic_mode()
        self._reset_symbolic_table_stack()
        return sample_idx == sequence_length

    def sample_world(self, sequence_length: int = 2, add_robots: bool = True, retry_ad_infinitum: bool = True) -> RDDLTask:
        """Drain ``sample_generator`` into a single :class:`RDDLTask`.

        Collects the full action sequence and per-step post-state snapshots, then wraps
        them with the captured initial state into an ``RDDLTask``.
        """
        generator = self.sample_generator(sequence_length=sequence_length, add_robots=add_robots, retry_ad_infinitum=retry_ad_infinitum)
        action_sequence: list[AtomicAction] = []
        state_sequence: list[StrictSymbolicCacheContainer] = []

        for action, state in generator:
            action_sequence.append(action)
            state_sequence.append(state)

        start_state = self._initial_world_state
        task = RDDLTask(action_sequence, state_sequence, start_state, world_generator=self)
        # check action variables
        # add missing variables
        # make init = true
        # make goal = true

        return task

    def _recursive_sampling(self, sample_idx, action_sequence, state_sequence, start_state=None):
        """DFS producer: enumerate all consistent sequences of ``__recurse_max_samples`` actions.

        Same per-step logic as ``sample_generator`` (branch, bind, force/check precondition,
        apply effect) but recursive: each candidate that survives the precondition check
        recurses one level deeper with immutable ``seq + [action]`` extensions, so the
        Python call stack is the backtracking mechanism. Completed sequences are pushed
        onto ``__action_state_stack`` for the consumer in ``recurse_generator``.
        """
        if sample_idx == self.__recurse_max_samples:
            # reached requested depth: emit the completed (actions, states, start) tuple
            # store sequences
            self.__action_state_stack.put((action_sequence, state_sequence, start_state))
            return

        action_generator = self._weighter.get_random_generator() if sample_idx > 0 else self._weighter.get_initial_generator()
        while True:
            try:
                action = next(action_generator)
            except StopIteration:
                break  # this choice list is exhausted (RETRY_AD_INFINITUM is off here)
            except BaseException as e:
                print(e)
            action_variables = action.gather_variables()
            current_world = self._symbolic_cache_duplicate_and_stack()  # branch off current world
            added_vars = self._symbolic_table.lookup_and_link_variables(action_variables, self.sample_object_subclass)
            if added_vars:
                if sample_idx > 0:
                    # later steps: only assert facts about newly created objects
                    action.initial.set_symbolic_value(True, set(added_vars))
                    start_state = self._symbolic_table.clone()
                else:
                    action.initial.set_symbolic_value(True)  # step 0: force the whole precondition
            if not action.initial.decide():
                # precondition inconsistent: downweight, roll back this branch, try next candidate
                self._weighter.penalize_initial(action)
                self._remove_variables(added_vars)
                self._symbolic_cache_pop()
                continue
            action.predicate.set_symbolic_value(True)  # apply effect, then descend
            self._recursive_sampling(sample_idx + 1, action_sequence + [action], state_sequence + [current_world], start_state)

    def _recursive_sampling_entry(self, *args) -> None:
        """Thread entry point for the DFS producer.

        Runs ``_recursive_sampling`` and -- in a ``finally`` -- always pushes a ``None``
        sentinel so the consumer (``recurse_generator``) is reliably told the tree is
        exhausted, even if the DFS raised.
        """
        # run the DFS producer and always signal termination to the consumer, even on error,
        # so recurse_generator never blocks forever on an exhausted/failed tree
        try:
            self._recursive_sampling(*args)
        finally:
            self.__action_state_stack.put(None)

    def recurse_generator(self,
                          sequence_length: int = 2,
                          add_robots: bool = True,
                          n_samples_requested: int = np.iinfo(np.int32).max,
                          ) -> Generator['RDDLTask', bool, bool]:
        """Enumerate many distinct tasks via DFS instead of one weighted-random sequence.

        Runs ``_recursive_sampling`` on a daemon thread (``RETRY_AD_INFINITUM`` forced off
        so each choice list is traversed once and the tree is finite) feeding a bounded
        ``Queue``. This generator consumes that queue, wrapping each completed sequence in
        an :class:`RDDLTask` and yielding it, until the tree is exhausted (``None``
        sentinel) or ``n_samples_requested`` tasks have been produced.

        Args:
            sequence_length: number of actions per enumerated task.
            add_robots: if true, seed the world with one ``Gripper`` variable.
            n_samples_requested: stop after yielding this many tasks.

        Yields:
            One :class:`RDDLTask` per distinct enumerated sequence.

        Raises:
            RuntimeError: if a generator is already running.
        """
        if self.__state != self.STATE_IDLE:
            raise RuntimeError("Generator is already running!")

        self._initialize_world(add_robots=add_robots)

        self._initial_world_state: StrictSymbolicCacheContainer
        self._goal_world_state: StrictSymbolicCacheContainer

        Weighter.RETRY_AD_INFINITUM = False  # finite tree: traverse each choice list once

        self.__action_state_stack = Queue(maxsize=10)  # bounded hand-off; producer blocks when full
        self.__recurse_max_samples = sequence_length

        # the DFS runs on a background thread; this generator is the consumer
        generator_thread = Thread(target=self._recursive_sampling_entry, args=(0, [], []), daemon=True, name="rddl_generator")
        generator_thread.start()

        # self._recursive_sampling(0, [], [])
        n_samples_out = 0

        while True:
            try:
                # output = self.__action_state_stack.get(block=True, timeout=1 * self.__recurse_max_samples)
                output = self.__action_state_stack.get(block=True)
            except Empty:
                warn("Generator timed out after {n_samples_out} samples!")
                break
            if output is None:
                break  # sentinel from _recursive_sampling_entry: producer finished
            else:
                actions, states, start_state = output
                n_samples_out += 1

            # self._goal_world_state = self._symbolic_table.clone()
            # self.__action_state_stack.task_done()
            task = RDDLTask(actions=actions, states=states, initial_state=start_state, world_generator=self)

            # c = Operand.get_cache()
            # c.show_table()

            yield task
            if n_samples_out >= n_samples_requested:
                break

        self.deactivate_symbolic_mode()
        return True

    def get_created_variables(self) -> list[Variable]:
        """Return all variables (entity bindings) registered in the current world."""
        return list(self._variables.values())

    def show_world_state(self) -> None:
        """Print the current symbolic-cache truth table (the live world state)."""
        self._symbolic_table.show_table()

    def show_initial_world_state(self) -> None:
        """Print the snapshot of the initial world state captured at step 0."""
        self._initial_world_state.show_table()

    def show_goal_world_state(self) -> None:
        """Print the snapshot of the final/goal world state."""
        self._goal_world_state.show_table()

    # def _lookup_and_link_variables(self, variables: list[Variable]) -> list[Variable]:
    #     missing_vars = []
    #     linked_vars = []
    #     for v in variables:
    #         like_gen = self._symbolic_table.find_variable_like_this(v)
    #         try:
    #             while (like_v := next(like_gen)) is not None:
    #                 if like_v not in linked_vars:
    #                     break
    #         except StopIteration:
    #             like_v = None

    #         if like_v is None:
    #             like_v = self._get_random_variable(v.type)
    #             missing_vars.append(like_v)
    #             self._add_variable(v.name, like_v)  # FIXME: name should be somehow estimated
    #         v.link_to(like_v)
    #         linked_vars.append(like_v)
    #     return missing_vars

    def _initialize_world(self, add_robots: bool = True):
        """Enter symbolic mode, reset the world, and optionally seed a gripper.

        Switches the global ``Operand`` cache to the symbolic table, clears it, and (when
        ``add_robots``) creates and registers a single ``Gripper`` variable as ``self._gripper``.
        """
        self.activate_symbolic_mode()
        self.reset_world()
        if add_robots:
            self._gripper = self._symbolic_table._get_random_variable(Gripper, self.sample_object_subclass)
            self._add_variable("gripper", self._gripper)

    def _add_variable(self, name: str, variable: SymbolicEntity) -> None:
        """Register ``variable`` in the current symbolic table (``name`` currently unused)."""
        # self._variables[name] = variable
        self._symbolic_table.register_variable(variable)

    def _remove_variables(self, variables: list[Variable]) -> None:
        """Unregister the given variables from the current symbolic table (rollback)."""
        self._symbolic_table.remove_variables(variables)
        # for variable in variables:
        #     for local_name, local_var in self._variables.items():
        #         if local_var == variable:
        #             del self._variables[local_name]
        #             break

    @lru_cache(maxsize=128)
    def _find_allowed_subclasses(self, base_type: type[Entity]) -> NDArray:
        """Concrete subclasses of ``base_type`` permitted by ``_allowed_entities`` (cached)."""
        if self._allowed_entities is not None:
            options = np.asanyarray([sc for sc in base_type.list_subclasses() if sc in self._allowed_entities])
        else:
            options = np.asanyarray([sc for sc in base_type.list_subclasses()])
        return options

    def _weight_object_classes(self, subclasses: NDArray, object_weights: dict[type[Entity], float]) -> NDArray:
        """Return a normalized probability vector over ``subclasses`` from ``object_weights``."""
        weights = np.array([object_weights[sc] for sc in subclasses])
        weights /= np.sum(weights)
        return weights

    def _reduce_object_class_weight(self, object_type: type[Entity], object_weights: dict[type[Entity], float]) -> None:
        """Decay a class's sampling weight after use: to 0 if single-object-per-class, else x0.9."""
        if self._sample_single_object_per_class:
            object_weights[object_type] = 0
        else:
            object_weights[object_type] *= 0.9

    def sample_object_subclass(self, base_type: type[Entity], object_weights: Optional[dict[type[Entity], float]] = None) -> type[Entity]:
        """Pick a concrete subclass of ``base_type`` at random, weighted and weight-decaying.

        Chooses among the allowed subclasses via ``RNG.choice`` (using ``object_weights`` as
        probabilities when given) and then decays the chosen class's weight in place. Used as
        the ``sampler`` callback for entity binding.
        """
        options = self._find_allowed_subclasses(base_type)
        if object_weights is not None:
            probs = self._weight_object_classes(options, object_weights)
        else:
            probs = None
        choice = self.RNG.choice(options, p=probs)
        if object_weights is not None:
            self._reduce_object_class_weight(choice, object_weights)
        return choice

    # def _get_random_variable(self, typ: type[Entity]) -> SymbolicEntity:
    #     return SymbolicEntity(self.sample_object_subclass(typ))

    @property
    def _symbolic_table(self) -> StrictSymbolicCacheContainer:
        """The current world state: the top clone on the symbolic-table stack."""
        return self._symbolic_table_stack[-1]

    @property
    def _variables(self) -> dict[str, Variable]:
        """The entity bindings registered in the current world state."""
        return self._symbolic_table.variables

    def _reset_symbolic_table_stack(self) -> None:
        """Reset the backtracking stack to a single fresh, empty symbolic table."""
        self._symbolic_table_stack = deque()
        self._symbolic_table_stack.append(StrictSymbolicCacheContainer())

    def _symbolic_cache_duplicate_and_stack(self) -> StrictSymbolicCacheContainer:
        """
        Duplicate the current symbolic table, set it as the current world state
        (in Operand cache) and push it onto the stack.
        This creates a new "branch" in the search tree.
        """
        current_world = self._symbolic_table.clone()
        self._symbolic_table_stack.append(current_world)
        Operand.set_cache_symbolic(current_world)
        return current_world

    def _symbolic_cache_pop(self) -> None:
        """Discard the top cache clone (backtrack a branch); refuses to pop the last one."""
        if len(self._symbolic_table_stack) == 1:
            raise RuntimeError("Cannot pop last cache!")
        self._symbolic_table_stack.pop()

    def activate_symbolic_mode(self):
        """Enter symbolic mode: save the normal compute-cache and point ``Operand`` at the symbolic table."""
        self.__real_cache = Operand.get_cache()
        Operand.set_cache_symbolic(self._symbolic_table)

    def deactivate_symbolic_mode(self):
        """Leave symbolic mode: restore the saved normal compute-cache (no-op if none saved)."""
        if self.__real_cache is None:
            return
        Operand.set_cache_normal(self.__real_cache)

    def reset_world(self):
        """Mark the world idle and clear the current symbolic table."""
        self.__state = self.STATE_IDLE
        # self._variables = {}
        self._symbolic_table.reset()

    def reset_weights(self, mode: Optional[int] = None):
        """Reset the ``Weighter``'s learned weights, optionally setting its sampling mode."""
        self._weighter.reset_weights()
        if mode is not None:
            self._weighter.set_mode(mode)

    def full_reset(self):
        """Reset both the world state and the action weights to their initial condition."""
        self.reset_world()
        self.reset_weights()

    @property
    def weighter(self) -> Weighter:
        """The ``Weighter`` driving weighted, sequence-aware action ordering."""
        return self._weighter

    @classmethod
    def set_seed(cls, seed: int) -> None:
        """Re-seed the shared class-level RNG (and thus the ``Weighter``'s stream)."""
        BitGen = type(cls.RNG.bit_generator)
        cls.RNG.bit_generator.state = BitGen(seed).state


class RDDL:
    """Facade tying together the textual DSL parser and action definitions.

    Wraps an ``RDDLParser`` and offers a small entry point to register the operator/
    predicate/type vocabularies and to compile textual action definitions into
    ``AtomicAction`` subclasses.
    """

    def __init__(self):
        """Create an uninitialized facade; ``initialize_parser`` must be called first."""
        self._parser = None

    def initialize_parser(self, combinator_mapping: dict[str, type[OperatorType]], predicate_mapping: dict[str, type[PredicateType]], type_definitions: dict[str, type[EntityType]]):
        """Build the ``RDDLParser`` from the operator, predicate, and entity-type vocabularies."""
        self._parser = RDDLParser(combinator_mapping, predicate_mapping, type_definitions)

    def load_definitions(self, aa_definitions: dict[str, dict]):
        """Compile textual action definitions into ``AtomicAction`` classes and register them.

        Each entry's predicate string is parsed into an action class and added to the
        parser's predicate mapping under its name.

        Raises:
            ValueError: if the parser is uninitialized, or a name collides with an existing
                function/predicate.
        """
        if self._parser is None:
            raise ValueError("Please initialize the parser first!")

        for action_name, aa_def in aa_definitions.items():
            if action_name in self._parser.predicate_mapping:
                raise ValueError(f"The name {action_name} is already defined as a function!")
            action = self._extract_action(action_name, aa_def)
            self._parser.predicate_mapping[action_name] = action

    def step(self):
        """Reset the global ``Operand`` cache (advance to a fresh evaluation step)."""
        Operand.reset_cache()

    def _extract_action(self, action_name: str, action_def: dict[str, dict[str, str]]) -> type[AtomicAction]:
        """Parse one action definition's predicate into a new ``AtomicAction`` subclass.

        Raises:
            ValueError: if the parser is uninitialized or the definition lacks a predicate.
        """
        if self._parser is None:
            raise ValueError("Please initialize the parser first!")

        if "predicate" not in action_def:
            raise ValueError(f"Action {action_name} does not have a predicate!")
        predicate = self._parser.parse_action_predicate(action_def["predicate"])

        return type(action_name, (AtomicAction,), {
            "_predicate": predicate,
        })
