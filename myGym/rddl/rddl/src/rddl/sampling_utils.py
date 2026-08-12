"""Entity binding and action-ordering helpers for the RDDL generation engine.

This module backs §6.4 of the specification. It provides:

- ``SymbolicEntity`` — a ``Variable`` standing in for a concrete entity during
  symbolic world generation.
- ``StrictSymbolicCacheContainer`` — the world-state container. The world state
  *is* its decide-cache (a ``predicate(args) -> bool`` map, closed-world). It adds
  a variable registry that refuses cache keys over unregistered variables, plus
  ``clone()`` for branch/snapshot and ``get_predicates()`` for reading the state out.
- ``Weighter`` — a stochastic action-ordering oracle driving which action the
  generator tries next, using per-action weights, an initial-action mask, and a
  graph of learned bigram/trigram transition penalties.
"""
from collections import defaultdict, deque
from copy import deepcopy
from typing import Callable, ClassVar, Generator, Iterable, Optional, Type, Union
from rddl import AtomicAction, Operand, Variable
from rddl.core import Entity, LogicalOperand, Predicate, SymbolicCacheContainer
import numpy as np
import networkx as nx
from numpy.typing import ArrayLike, NDArray


class SymbolicEntity(Variable):
    """A ``Variable`` that represents a concrete entity type during generation.

    The base name encodes the entity type so the symbolic id stays human-readable.
    """

    def __init__(self, typ: type[Entity]):
        """Create a symbolic variable bound to entity type ``typ``."""
        super().__init__(typ, base_name=f"entity_{typ.__name__}")


class StrictSymbolicCacheContainer(SymbolicCacheContainer):
    """Symbolic world-state container guarded by a variable registry.

    The world state is the decide-cache inherited from ``SymbolicCacheContainer``:
    a ``predicate(args) -> bool`` map, closed-world (absent => False). On top of the
    base cache this subclass keeps:

    - ``_variable_registry``: ``{symbolic_id -> Variable}``; every variable
      referenced by a cache key must be registered first (see ``_make_key``), so the
      cache never holds dangling variables.
    - ``_object_weights``: per-type sampling weights for new variables (defaults 1.0).

    ``clone()`` is the snapshot/branch primitive used by the backtracking stack.
    """

    def __init__(self):
        """Initialize an empty cache, variable registry, and object weights."""
        super().__init__()
        self._variable_registry = {}
        self._object_weights = defaultdict(lambda: 1.0)

    def _make_key(self, func, internal_args, *args, **kwds):
        """Build a cache key, refusing any key over an unregistered variable.

        Guards the cache against dangling variables: every internal arg (a symbolic
        id) must already be in the registry, else a ``ValueError`` is raised.
        """
        # registry guard: reject keys mentioning variables not registered here
        for arg in internal_args:
            if arg not in self._variable_registry:
                raise ValueError(f"Variable with id '{arg}' is not registered in the SymbolicCacheContainer! Register a variable with the correct id (global name) first.")
        return super()._make_key(func, internal_args, *args, **kwds)

    def register_variable(self, variable: Variable) -> None:
        """Register ``variable`` by its symbolic id so it may appear in cache keys."""
        self._variable_registry[variable.symbolic_id] = variable

    def remove_variables(self, variables: list[Variable]) -> None:
        """Unregister the given variables (used to roll back created variables)."""
        for variable in variables:
            del self._variable_registry[variable.symbolic_id]

    def show_table(self):
        """Pretty-print the cached predicates as ``Predicate(args) -> value`` lines."""
        c = self._cache_decide
        for key, value in c.items():
            hashed_internal_args = key.arg_list
            try:
                args = ', '.join([self._variable_registry[id].name for id in hashed_internal_args])
            except KeyError:
                continue  # skip cache entries referencing variables no longer registered
            if key.other:
                args += ', ' + ', '.join(key.other)
            print(f"{key.class_name.__name__}({args}) -> {value}")

    def get_predicates(self) -> list[tuple[Predicate, list[Variable], bool]]:
        """Materialize the decide-cache into a list of resolved predicate facts.

        Returns:
            One ``(PredicateClass, [arg names], bool)`` triple per cached decision;
            this is the human/consumer read-out of the world state.
        """
        c = self._cache_decide
        result = []
        for key, value in c.items():
            hashed_internal_args = key.arg_list
            # resolve symbolic ids back to variable names via the registry
            try:
                args = [self._variable_registry[id].name for id in hashed_internal_args]
            except KeyError:
                continue  # skip entries whose variables are no longer registered
            if key.other:
                args += key.other  # append any non-variable ("other") args
            result.append((key.class_name, args, value))
        return result

    @property
    def variables(self):
        """The variable registry: ``{symbolic_id -> Variable}``."""
        return self._variable_registry

    def __contains__(self, variable: Variable) -> bool:
        """True iff ``variable`` (by symbolic id) is registered in this container."""
        return variable.symbolic_id in self._variable_registry

    def clone(self):
        """Deep-copy this container into an independent snapshot/branch.

        Used by the backtracking stack: the clone shares the immutable sentinel but
        gets its own registry, decide-cache, and object weights so mutations on one
        branch do not leak into another.
        """
        s = StrictSymbolicCacheContainer()
        s._cache_sentinel = self._cache_sentinel  # immutable; safe to share
        s._variable_registry = deepcopy(self._variable_registry)
        s._cache_decide = self._cache_decide.clone()
        s._object_weights = deepcopy(self._object_weights)
        return s

    def reset(self):
        """Clear the cache and unregister every variable, returning to an empty world."""
        super().reset()
        self.remove_variables(list(self._variable_registry.values()))

    def find_variable_like_this(self, variable: Variable) -> Generator[Variable, None, Optional[Variable]]:
        """Yields variables from the current world that are of the same type as the given variable.

        Args:
            variable: The variable to find similar variables for.

        Yields:
            Variable: A variable that is of the same type as the given variable.

        Returns:
            None: If no variable of the same type is found.
        """
        for v in self.variables.values():
            if issubclass(v.type, variable.type):
                yield v
        return None

    def _get_random_variable(self, typ: type[Entity], object_sampling_function: Callable[[type[Entity], Optional[dict[type, float]]], type[Entity]]) -> SymbolicEntity:
        """Create a new ``SymbolicEntity`` of a concrete subclass of ``typ``.

        ``object_sampling_function`` picks the concrete subclass, weighted by this
        container's per-type ``_object_weights``.
        """
        return SymbolicEntity(object_sampling_function(typ, self._object_weights))

    def lookup_and_link_variables(self, variables: list[Variable], object_sampling_function: Callable[[type[Entity], Optional[dict[type, float]]], type[Entity]]) -> list[Variable]:
        """Bind each action slot to a world entity, reusing one where possible.

        For each free slot, reuse the first registered variable whose type is
        compatible and not already linked in this call; otherwise create, register,
        and record a new one. Reuse is how the sequence chains the same objects (the
        same gripper/apple) across actions — the implicit symmetry/dedup mechanism.

        Args:
            variables: The action's free slots to bind.
            object_sampling_function: Picks a concrete subclass for new variables.

        Returns:
            Only the newly created variables (so the caller can roll them back).
        """
        missing_vars = []
        linked_vars = []
        for v in variables:
            like_gen = self.find_variable_like_this(v)
            # reuse: take the first type-compatible registered var not yet linked here
            try:
                while (like_v := next(like_gen)) is not None:
                    if like_v not in linked_vars:
                        break
            except StopIteration:
                like_v = None

            # create: no reusable candidate -> sample, register, and mark as new
            if like_v is None:
                like_v = self._get_random_variable(v.type, object_sampling_function)
                missing_vars.append(like_v)
                self.register_variable(like_v)
            v.link_to(like_v)  # alias slot -> chosen entity (shared global name)
            linked_vars.append(like_v)
        return missing_vars


class Weighter:
    """Stochastic action-ordering oracle for the generation loop.

    Decides which action to try next. It (a) honours per-action ``_weights``,
    (b) restricts the first action to the legal initial set via ``_initial_weights``,
    (c) learns to avoid recently-seen / failed action pairs and triples through an
    ``nx.MultiDiGraph`` of bigram/trigram transition penalties keyed off a 3-deep
    history queue, and (d) either gives up after a single pass over the choices or
    retries forever (``RETRY_AD_INFINITUM``).

    Behaviour is toggled by the ``MODE_*`` bit-flags (``mode`` is an OR of them).
    """

    # --- penalty coefficients: weights are multiplied by these on selection ---
    INITIAL_WEIGHT_PENALTY_COEFF = 0.9  # coefficient to multiply the initial weight when an action is selected
    WEIGHT_PENALTY_COEFF = 0.9  # coefficient to multiply the weight when an action is selected
    SEQUENCE_PENALTY_COEFF = 0.95  # coefficient to multiply the sequence weight when an action is selected, given previous actions
    RETRY_AD_INFINITUM: ClassVar[bool] = True  # whether to supply actions (from randomly shuffled sequence) for ever or terminate after single loop through the sequence
    EPS = np.finfo(float).min

    RNG: ClassVar[np.random.Generator]

    # --- mode bit-flags (OR together; tested via self._mode & MODE_*) ---
    MODE_NONE: ClassVar[int] = 0  # no weighting (uniform)
    MODE_INITIAL: ClassVar[int] = 1  # use initial weights
    MODE_WEIGHT: ClassVar[int] = 2  # use weights for individual actions
    MODE_SEQUENCE: ClassVar[int] = 4  # use sequence weights
    MODE_RANDOM: ClassVar[int] = 8  # randomize the total weight to add noise to the weights
    MODE_MAX_NOISE: ClassVar[int] = 16  # break ties among max-weight items with small noise

    BASE_MODE: ClassVar[int] = MODE_WEIGHT | MODE_INITIAL | MODE_SEQUENCE | MODE_RANDOM | MODE_MAX_NOISE  # all flags on

    def __init__(self, items: Iterable[AtomicAction], weights: Optional[list[float]] = None, initial_weights: Optional[list[float]] = None) -> None:
        """Build a weighter over ``items`` (action classes).

        Args:
            items: Candidate action classes to order.
            weights: Optional per-action base weights (default 1.0 each).
            initial_weights: Optional mask/weights for legal first actions
                (default 1.0 each); a 0.0 entry excludes the action from step 0.
        """
        self.set_mode(self.BASE_MODE)

        self._items = np.asarray(items)
        if weights is None:
            self._weights = {item.__name__: 1.0 for item in items}
        else:
            self._weights = {item.__name__: weight for item, weight in zip(items, weights)}
        self._initial_weights = {item.__name__: 1.0 for item in items}
        if initial_weights is None:
            self._initial_weights = {item.__name__: 1.0 for item in items}
        else:
            self._initial_weights = {item.__name__: weight for item, weight in zip(items, initial_weights)}

        self._bkp_weights = deepcopy(self._weights)
        self._bkp_initial_weights = deepcopy(self._initial_weights)

        # if self._mode & self.MODE_SEQUENCE:
        self._previous_item_queue = deque(maxlen=3)
        self._weight_graph = nx.MultiDiGraph()

    @staticmethod
    def _get_random_item(choices: NDArray, condition: Callable) -> Generator[tuple[AtomicAction], None, None]:
        # def _get_random_item(choices: NDArray, condition: Callable) -> Generator[tuple[AtomicAction, list[Variable]], None, None]:
        """Yields items from the list of choices.

        Args:
            choices (NDArray): list of items to loop through.
            condition (Callable): function that returns True if the loop should continue.

        Yields:
            Generator[tuple[AtomicAction, list[Variable]], None, None]: Returns (action, [list of variables]).
        """
        n_choices = len(choices)
        choice_idx = 0

        while condition(choice_idx, n_choices):
            choice_idx %= n_choices
            action = choices[choice_idx]()  # get and instantiate next action
            # a_variables = action.gather_variables()
            yield action  # , a_variables
            choice_idx += 1

    def set_mode(self, mode: int) -> None:
        """Sets the mode of the weighter.

        Args:
            mode (int): Mode to set.
        """
        self._mode = mode
        # choose the sort key for _weighted_shuffle based on the RANDOM flag
        if self._mode & self.MODE_RANDOM:
            # random key: higher weight -> exponent near 0 -> key near 1 -> sorts later
            self._sampling_key_function = lambda weights: lambda i: self.RNG.random() ** np.exp(1.0 / (weights[i] + self.EPS))
        else:
            # deterministic key: higher weight -> lower key -> sorts earlier
            self._sampling_key_function = lambda weights: lambda i: 1 - weights[i]

    def _get_seq_weights(self) -> list[float]:
        """Computes weights based on the sequence of previous actions and individual action weights.


        Returns:
            list[float]: List of weights.
        """
        coefs = deepcopy(self._weights)  # get a copy of the individual weights
        for item, weight in coefs.items():  # for each action / weight
            # bigram edge: previous action -> this candidate (None if never penalized)
            w_data = self._weight_graph.get_edge_data(self._previous_item_queue[-1], item)
            if w_data is None:
                continue  # no learned transition penalty for this candidate
            preceding = self._previous_item_queue[-2]  # find the action preceding the previous action
            triple_w = w_data[preceding]['weight'] if preceding in w_data else 1
            coefs[item] = weight * w_data[0]['weight'] * triple_w  # individual weight * previous->current weight * preceding->previous->current weight
        return list(coefs.values())

    def _add_max_noise(self, weights: NDArray) -> NDArray:
        """Jitter the maximum-weight entries to randomly break ties among them.

        Adds small uniform noise only to the entries equal to the current max, so
        that ties at the top do not always sort in the same (index) order.

        Args:
            weights: Weight array, modified in place.

        Returns:
            The same array with the max entries perturbed.
        """
        m = weights.max()
        where_max = weights == m  # mask of entries tied for the maximum
        noise = self.RNG.uniform(0, 0.01, np.count_nonzero(where_max))
        weights[where_max] += noise
        return weights

    def get_random_generator(self) -> Generator:
        """Yields items from the list of choices. The items are randomly shuffled based on their weights.

        Returns:
            Generator: Generator yielding (action, [list of variables]).

        Yields:
            action, list[Variable]: Returns (action, [list of variables]).
        """
        # use sequence-aware weights once the 3-deep history is full, else base weights
        if self._mode & self.MODE_SEQUENCE and len(self._previous_item_queue) == 3:
            weighted_weights = np.array(self._get_seq_weights())
        else:
            weighted_weights = np.array(list(self._weights.values()))

        weighted_weights = self._add_max_noise(weighted_weights)

        choices = self._weighted_shuffle(weighted_weights)  # shuffle action classes (so they are in varying order each time)
        # condition controls how long _get_random_item keeps yielding: forever if
        # RETRY_AD_INFINITUM, else stop after one pass over the choices
        condition = lambda ci, nc, ad_inf=self.RETRY_AD_INFINITUM: ad_inf or ci < nc  # noqa
        return self._get_random_item(choices, condition)

    def get_initial_generator(self) -> Generator:
        """Yield candidate first actions, weighted by ``_initial_weights``.

        Like ``get_random_generator`` but draws on the initial-action mask and drops
        actions with weight 0.0 (illegal as a first action). Always finite (one pass).

        Returns:
            Generator yielding instantiated candidate actions.
        """
        weights = self._add_max_noise(np.asarray(list(self._initial_weights.values())))
        choices = self._weighted_shuffle(weights)  # shuffle action classes (so they are in varying order each time)
        choices = np.asarray([c for c in choices if self._initial_weights[c.__name__] > 0.0])  # cleanup improbable choices
        condition = lambda ci, nc: ci < nc  # noqa
        return self._get_random_item(choices, condition)

    def _weighted_shuffle(self, weights) -> NDArray:
        """Order the items by the mode-selected sampling key (weighted shuffle).

        Sorts item indices by ``_sampling_key_function(weights)`` (random or
        deterministic per the RANDOM flag) and returns the items in that order.
        """
        order = sorted(range(len(self._items)), key=self._sampling_key_function(weights))
        return np.asarray([self._items[i] for i in order])

    def _get_weight(self, from_item: str, to_item: str, preceded_by: Union[str, int] = 0) -> float:
        """Return the learned transition weight for an edge, or 1.0 if none.

        ``preceded_by`` selects the multigraph edge key: ``0`` for the bigram
        (``from_item -> to_item``) edge, or an action name for the trigram edge
        (preceded by that action).
        """
        attrs = self._weight_graph.get_edge_data(from_item, to_item, preceded_by)
        return 1.0 if attrs is None else attrs['weight']

    def penalize(self, action: AtomicAction) -> None:
        """Down-weight ``action``'s per-action weight (used after a rejected step)."""
        cls_name = action.__class__.__name__
        if self._mode & self.MODE_WEIGHT:
            self._weights[cls_name] = self.WEIGHT_PENALTY_COEFF * self._weights[cls_name]

    def penalize_initial(self, action: AtomicAction) -> None:
        """Down-weight ``action`` in the initial-action mask."""
        cls_name = action.__class__.__name__
        if self._mode & self.MODE_INITIAL:
            self._initial_weights[cls_name] = self.INITIAL_WEIGHT_PENALTY_COEFF * self._initial_weights[cls_name]

    def penalize_bad(self, action: AtomicAction) -> None:
        """Record ``action`` as a bad transition, strengthening the avoid-penalty.

        Pushes ``action`` onto the history queue and, once it holds three actions,
        writes (extra ``0.98`` factor) bigram and trigram penalty edges for the
        prev->current transition so it is avoided more strongly than a normal step.
        """
        if self._mode & self.MODE_SEQUENCE:
            cls_name = action.__class__.__name__
            self._previous_item_queue.append(cls_name)
            if len(self._previous_item_queue) == 3:
                # trigram penalty: edge keyed by the action two steps back (-3)
                triple_weight = 0.98 * self.SEQUENCE_PENALTY_COEFF * self._get_weight(self._previous_item_queue[-2], self._previous_item_queue[-1], self._previous_item_queue[-3])
                # bigram penalty: edge key 0 (prev -> current)
                double_weight = 0.98 * self.SEQUENCE_PENALTY_COEFF * self._get_weight(self._previous_item_queue[-2], self._previous_item_queue[-1])
                self._weight_graph.add_edge(self._previous_item_queue[-2], self._previous_item_queue[-1], self._previous_item_queue[-3], weight=triple_weight)
                self._weight_graph.add_edge(self._previous_item_queue[-2], self._previous_item_queue[-1], 0, weight=double_weight)

    def add_and_penalize(self, action: AtomicAction) -> None:
        """Accept ``action`` as a step: down-weight it and learn its transition.

        Penalizes the action's weight, then (sequence mode, full history) decays the
        bigram and trigram edges for the prev->current transition so that recently
        taken pairs/triples are progressively avoided.
        """
        self.penalize(action)
        if self._mode & self.MODE_SEQUENCE:
            cls_name = action.__class__.__name__
            if len(self._previous_item_queue) == 3:
                # trigram edge keyed by the action before the previous one (-2)
                triple_weight = self.SEQUENCE_PENALTY_COEFF * self._get_weight(self._previous_item_queue[-1], cls_name, self._previous_item_queue[-2])
                # bigram edge key 0 (previous -> this action)
                double_weight = self.SEQUENCE_PENALTY_COEFF * self._get_weight(self._previous_item_queue[-1], cls_name)
                self._weight_graph.add_edge(self._previous_item_queue[-1], cls_name, self._previous_item_queue[-2], weight=triple_weight)
                self._weight_graph.add_edge(self._previous_item_queue[-1], cls_name, 0, weight=double_weight)

    def add_and_penalize_initial(self, action: AtomicAction) -> None:
        """Accept the first action: down-weight its initial mask and seed the history."""
        self.penalize_initial(action)
        if self._mode & self.MODE_SEQUENCE:
            cls_name = action.__class__.__name__
            self._previous_item_queue.append(cls_name)

    def reset_weights(self) -> None:
        """Restore weights from backups and clear the history queue and penalty graph."""
        self._weights = deepcopy(self._bkp_weights)
        self._initial_weights = deepcopy(self._bkp_initial_weights)
        self._previous_item_queue = deque(maxlen=3)
        self._weight_graph = nx.MultiDiGraph()

    @classmethod
    def set_rng(cls, rng: np.random.Generator) -> None:
        """Set the shared class-level RNG used by all weighters."""
        cls.RNG = rng

    @property
    def mode(self) -> int:
        """The current mode bit-flags (OR of ``MODE_*``)."""
        return self._mode
