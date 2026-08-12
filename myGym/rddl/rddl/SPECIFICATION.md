# RDDL — Software Specification & Manual

> **RDDL** — *Reward Domain Description Language* — also branded **PRAG** (*Procedural
> action‑sequence symbolic Generator for Robotic manipulation tasks*).
>
> A library that **procedurally generates valid sequences of robotic‑manipulation
> actions** (with their initial state, goal state, and per‑action reward) from a
> declarative description of actions, predicates, and object types. It is intended as a
> task/curriculum generator for reinforcement learning in robotics.

---

## Table of contents

1. [Overview & purpose](#1-overview--purpose)
2. [System architecture](#2-system-architecture)
3. [Core type system (`core.py`)](#3-core-type-system-corepy)
4. [Domain layer (entities, predicates, operators, rewards)](#4-domain-layer)
5. [Actions (`actions.py`)](#5-actions-actionspy)
6. [The generation engine](#6-the-generation-engine)
7. [Task data structures](#7-task-data-structures)
8. [The simulator bridge (the `_0_*` mapping)](#8-the-simulator-bridge)
9. [Rule book (consistency constraints)](#9-rule-book)
10. [The textual DSL parser](#10-the-textual-dsl-parser)
11. [User manual](#11-user-manual)
12. [API quick reference](#12-api-quick-reference)
13. [Known issues & rough edges](#13-known-issues--rough-edges)
14. [Glossary](#14-glossary)

---

## 1. Overview & purpose

Given a **declarative domain** — object types (entities), boolean state functions
(predicates), and STRIPS‑like actions (precondition → effect, plus a reward) — RDDL
**samples** consistent action sequences of a requested length. Each generated **task**
carries:

- an ordered **action sequence** (e.g. `Approach → Grasp → Move → Drop`),
- an **initial symbolic state** (predicates true at the start),
- a **goal/final symbolic state** (predicates true at the end),
- the **objects** (entity bindings) involved, and
- a **reward** per action.

The abstract layer never touches physics. All concrete numeric/boolean primitives
(distance, reachability, gripper‑open, reward values) are supplied at runtime by an
external **simulator** through a string‑keyed function registry (§8). This makes the same
domain reusable across simulators (the reference test harness stubs them, mostly with
random values, to exercise the *structural* sampling logic).

**Entry point:** `python tests/test_world.py` generates tasks of length 2–10, 10 each,
and dumps them to `test_world.yaml`.

---

## 2. System architecture

```
                       ┌──────────────────────────────────────────┐
        user / sim ──▶ │  Operand.set_mapping(...)   (§8 bridge)   │ ◀── string→callable
                       │  Entity.set_observation_getter(...)       │
                       └──────────────────────────────────────────┘
                                          │  (must run BEFORE classes are defined)
                                          ▼
 ┌───────────────────────────── core.py (foundation) ──────────────────────────────┐
 │  Entity ── Variable ──┐                                                           │
 │                       │   Operand (decide()->bool / evaluate()->float, cached)    │
 │                       │     ├─ LogicalOperand ─ Operator    (AND/NOT/SEQ ...)     │
 │                       │     │                 └ Predicate    (boolean state fn)   │
 │                       │     │                     └ AtomicAction (pre/effect/rew) │
 │                       │     └─ Reward            (float reward fn)                 │
 │  Cache subsystem: NORMAL (compute) | SYMBOLIC (settable truth table = WORLD STATE)│
 └───────────────────────────────────────────────────────────────────────────────┘
        │ entities.py     │ predicates.py   │ operators.py   │ rewards.py   │ actions.py
        ▼                 ▼                 ▼                ▼              ▼
   Gripper, Graspable  Near, GripperAt   ParallelAndOp   AbsoluteMove   Approach, Grasp
   Location, Angle,    IsHolding, ...     NotOp, Seq...   Relative...    Move, Rotate ...
   AbstractRotation                                                       │
                                                                          ▼
 ┌──────────────────── generation engine ────────────────────┐   ┌──── task model ────┐
 │ rddl_sampler.py : RDDLWorld.sample_generator / sample_world │──▶│ rddl_task.RDDLTask │
 │ sampling_utils.py: StrictSymbolicCacheContainer, Weighter   │   │ (actions+states)   │
 │ rddl_parser.py  : RDDLParser (text DSL → AtomicAction)      │   └────────────────────┘
 └─────────────────────────────────────────────────────────────┘
```

**Layer responsibilities**

| Layer | Files | Responsibility |
|---|---|---|
| Foundation | `core.py` | Type system, variable binding, the dual‑mode evaluation cache, the function‑mapping registry. |
| Domain | `entities.py`, `predicates.py`, `operators.py`, `rewards.py` | Concrete object types, state predicates, logical connectives, reward formulas. |
| Actions | `actions.py` | STRIPS‑style operators bundling precondition + effect + reward class. |
| Engine | `rddl_sampler.py`, `sampling_utils.py`, `rddl_parser.py` | The constructive sampler, weighted action ordering, entity binding, text DSL. |
| Task | `task.py`, `rddl_task.py` | The generated‑task container & playback. |
| Constraints | `rule_book.py` | Cross‑predicate consistency rules. |
| Harness | `tests/test_world.py`, `tests/testing_utils.py` | Entry point + simulator stub + concrete entity classes. |

---

## 3. Core type system (`core.py`)

The single root abstraction is **`Operand`** — anything computable. Two orthogonal
non‑operand trees exist alongside it: **`Entity`** (world things) and **`Variable`**
(typed placeholders).

### 3.1 `Entity` (`core.py:12`)

ABC for everything that exists in the world.

- A **global** `_observation_getter` callback (set via `Entity.set_observation_getter`)
  returns the observation for an instance; `entity()` calls it. Instantiation **fails**
  if the getter is unset.
- Per‑subclass instance counter (reset in `__init_subclass__`, bumped in `__new__`) drives
  auto‑generated reference names: `_get_generic_reference()` → `"<classname_lower>_<n>"`.
- `monkey_patch(cls, method, alternative)` (§3.6) replaces an **existing** method on a
  class (and thus all subclasses).

### 3.2 `Variable` (`core.py:114`) — the binding model

A `Variable(typ, *, arg_name, global_name, base_name)` is a **typed placeholder** that is
later *bound* to a concrete `Entity`.

- **Identity is the global name.** `__hash__ == hash(self._name)` and `__eq__` compares
  hashes, so **two `Variable`s with the same global name are the same key** — this is how
  linking/sharing a binding works.
- **Storage is global.** Bindings live in a class‑level table
  `_VAR_VAL_TABLE: {hash(global_name) → _VarRecord(type, value) | None}`. `bind(value)`
  asserts `issubclass(type(value), declared_type)` and writes the record; `unbind()` sets
  it to `None`; `is_bound()` checks it.
- **Transparent proxy.** `__getattr__` forwards non‑underscore attribute reads to the
  bound entity, so `var.location` ≡ `bound_entity.location`, and `var()` returns the entity.
- **Naming.** `base_name="apple"` → `global_name="apple_<count>"` (deduped by appending
  `_`); `arg_name` is the *local* name inside one predicate (the `_VARIABLES` key), while
  `name`/global name is the world‑unique identity used across predicates.
- **Linking.** `link_to(other)` aliases this variable's name to `other` (requires matching
  types, unbound destination) → they share a binding. `unlink` / `global_rename` break or
  rename the link group.
- **IDs for cache keys.** `id` = `hash(value)` if bound else `hash(self)`; `symbolic_id` =
  always `hash(self)`. (Normal mode keys on concrete value; symbolic mode keys on symbol.)

### 3.3 `Operand` (`core.py:582`) & the dual‑mode cache

`Operand` declares abstract `__decide__() → bool` and `__evaluate__() → float`. The public,
`final` `decide()` / `evaluate()` route through a **class‑global cache** when
`_USE_CACHE` is true (Operators and AtomicActions set it false).

Two cache modes (`CacheMode`):

| Mode | Container | `decide()` on cache miss | `evaluate()` |
|---|---|---|---|
| **NORMAL** | `_CacheContainer` | **computes** `__decide__` (calls the real external fn), memoizes | computes `__evaluate__` |
| **SYMBOLIC** | `SymbolicCacheContainer` | **does not compute** — returns stored truth value, default **`False`** (closed world) | raises `NotImplementedError` |

`set_symbolic_value(value, only_if_contains=None)` writes the symbolic truth table for an
operand, optionally gated on whether the operand's args contain a given variable set. **In
symbolic mode the decide‑cache *is* the world state** — this is the cornerstone of the
generator (§6).

Cache keys are `_CacheKey(class, internal_args, *args, **kwds)` where `internal_args`
comes from `_prepare_args_for_key()` — `symbolic_id`s in symbolic mode, `id`s in normal
mode. The same predicate instance therefore keys differently per mode, letting a symbolic
truth table coexist with concrete evaluation.

### 3.4 The class hierarchy under `Operand`

```
Operand
├─ LogicalOperand                      (adds gather_variables())
│   ├─ Operator        (_USE_CACHE=False; requires _ARITY, _SYMBOL)
│   └─ Predicate       (boolean; declares _VARIABLES; evaluate() forbidden)
│       └─ AtomicAction (_USE_CACHE=False; adds _initial, _predicate, _reward, REWARD_CLASS)
└─ Reward              (float; declares _VARIABLES; decide() forbidden)
```

- **`Predicate`** (`core.py:803`): declares `_VARIABLES: {name → EntityType}`. Its
  `__init_subclass__` installs, per variable, a **property** returning the bound entity
  (`self.object_A` ≡ `self.__variables['object_A']()`). `__init__` accepts each variable
  from kwargs (type‑checked `Variable`) or auto‑creates one named
  `"<Class>_<counter>_<arg>"`. `decide()` (cached) ⇒ `__decide__` ⇒ `self(...)`; subclasses
  implement the external check in `__call__`. `evaluate()` warns and returns `None`.
- **`Reward`** (`core.py:942`): declares `_VARIABLES`; `__init_subclass__` *validates the
  constructor signature* — every `_VARIABLES` key must be a typed parameter whose
  annotation (unwrapping `Variable[T]` generics) is a subclass of the declared type — and
  installs the same property proxies. `evaluate()` ⇒ `__evaluate__` ⇒ `self(...)`;
  `decide()` is forbidden.
- **`AtomicAction`** (`core.py:1008`): see §5. Inherits `Predicate` (so it owns typed
  variables) but disables caching.

### 3.5 The `_0_*` mapping mechanism (function registration)

Domain classes reference simulator capabilities by **string key**, declared as
`_0_`‑prefixed class attributes, e.g. `_0_GRIPPER_OPEN_CHECK = "gripper_open"`.

- **Registration:** `Operand.set_mapping(dict)` merges `{key → callable}` into the global
  `Operand.__MAPPING`.
- **Resolution (at class definition):** `__init_subclass__ → __register_attributes` scans
  the class's own `_0_*` attributes, looks up each **string value** in `__MAPPING`, and
  **replaces the attribute in place with the resolved callable**. A missing key raises
  `ValueError` *immediately*.
- **Ordering constraint:** therefore `set_mapping(...)` **must run before** any operand
  subclass that needs those keys is imported/defined.
- **Discovery:** `Operand.list_required_mappings()` returns every `_0_*` string declared
  across all subclasses — the authoritative "what must I register" list.

```
# pseudocode — at subclass creation
for attr, value in vars(cls):
    if attr.startswith("_0_"):
        if value in Operand.__MAPPING: setattr(cls, attr, __MAPPING[value])  # string → callable
        else: raise ValueError(f"required mapping {value} for {attr} not defined")
```

### 3.6 Monkey patching

`Entity.monkey_patch(cls, method, alternative)` replaces an existing method (guarded by
`hasattr`; refuses to *add* new methods) and emits a warning, because the swap changes
behavior for **all** subclasses. The reference harness uses it to redirect
`Location._get_location` to the simulator's `get_position()` (§8).

---

## 4. Domain layer

### 4.1 Entities (`entities.py`)

| Class | Base | Notes |
|---|---|---|
| `Location` | `Entity` | abstract `_get_location()`; `location` property returns it. Base `Location` auto‑names `loc_N`. |
| `Angle` | `Entity` | stores scalar `_angle`; `location` returns the scalar. |
| `ObjectEntity` | `Location` | adds `name`; `_get_location` still abstract (concrete impl from a mixin). |
| `GraspableObject` | `ObjectEntity` | graspable things (apples, boxes…). |
| `Gripper` | `Location` | adds `is_holding(obj)` → `_is_holding_predicate(obj)`. |
| `AbstractLocation` | `Location` | `value` raises until implemented. |
| `AbstractRotation` | `Entity` | reference required; `value` abstract. |
| `RandomRotation` | `AbstractRotation` | `_angle = U(-π, π)`; `value` returns it. |

`LocationType = TypeVar(bound=Location)` is used in generic `Variable[LocationType]`
annotations. Concrete simulator‑backed entities are created by **multiple inheritance** of
an RDDL base with a geometry‑providing mixin (see §8 / §11).

### 4.2 Predicates (`predicates.py`)

Each predicate declares `_VARIABLES`, one or more `_0_*` mapped functions, and a `__call__`
returning a bool. `SpatialPredicate` is a semantic marker subclass (no added behavior).

| Predicate | `_VARIABLES` | Mapped fn(s) | `__call__` |
|---|---|---|---|
| `Near` | object_A, object_B : Location | `euclidean_distance`, `near_threshold` | `dist(A.loc, B.loc) < near_threshold` |
| `GripperAt` | gripper, object | `gripper_at` | `gripper_at(gripper, object)` |
| `GripperOpen` | gripper | `gripper_open` | `gripper_open(gripper)` |
| `IsReachable` | gripper, location | `is_reachable` | `is_reachable(gripper, location)` |
| `IsHolding` | gripper, object:ObjectEntity | `is_holding` | `is_holding(gripper, object)` |
| `ObjectAt` | object, location | `object_at` | `object_at(object, location)` |
| `ObjectAtPose` | object, angle:AbstractRotation | `object_at_pose` | `object_at_pose(object, angle)` |
| `Exists` | entity:Entity | `exists` | `exists(entity)` |
| `OnTop` | object_A, object_B | `on_top` | `on_top(object_A, object_B)` |

**Evaluation contract:** only `decide()` is legal (cached `__call__`); `evaluate()` is
forbidden. Some predicates pass resolved `.location` coordinates to the external fn
(`Near`, `ObjectAt`), others pass the entity objects themselves (`GripperAt`, `IsHolding`,
…) and expect the simulator fn to resolve them. (`ObjectAtPose` and `Exists` previously
carried defects — see resolved KI‑2 / KI‑1 in §13 — and now use their own `object_at_pose`
hook and the correct `entity` slot respectively.)

### 4.3 Operators (`operators.py`)

Operators compose operands into a logic/scoring tree. `_USE_CACHE=False` (children are
cached). Two parallel channels: `decide()` (boolean satisfaction) and `evaluate()`
(additive reward shaping).

| Operator | Symbol | `decide()` | `evaluate()` | `set_symbolic_value(v)` |
|---|---|---|---|---|
| `ParallelAndOp` | `&` | `L.decide ∧ R.decide` | `L.eval + R.eval` | push to both children |
| `SequentialOp` | `->` | `R.decide` (after computing L) | `L.eval + (R.eval if L.decide else 0)` | push to both |
| `NotOp` | `~` | `¬operand.decide` | `−operand.eval` | push `¬v` (De Morgan) |
| `NAryAndOp` | `&` | `⋀ op.decide` | `Σ op.eval` | push `v` to all |

`gather_variables()` walks the tree, collecting `Variable`s only from `LogicalOperand`
leaves (predicates) — how an expression discovers its full variable set. `NotOp`'s symbolic
push of `¬v` lets the sampler flip truth assignments through negations.

```
ParallelAndOp(L,R): decide = L∧R ;  eval = L+R
NAryAndOp(ops):     decide = ⋀ops; eval = Σops
NotOp(X):           decide = ¬X ;   eval = −X ;  set_symbolic(v) → X.set_symbolic(¬v)
SequentialOp(L,R):  decide = R ;    eval = L + (R if L else 0)      # "do L, then R"
```

### 4.4 Rewards (`rewards.py`)

A reward subclass declares `_0_*` functions, `_VARIABLES`, a matching‑signature `__init__`
(validated at class creation), and a `__call__ → float`. Two **stateful relative** rewards
shape on the *change* in distance between calls:

```
RelativeApproachReward.__call__():
    dist = euclidean_distance(gripper.location, obj.location)
    if last is None: last = dist; return 0                  # prime step
    reward = (last − dist) + 0.2 · gripper_open_reward(gripper)   # closer + keep-open bonus
    last = dist; return reward

RelativeWithdrawReward.__call__():    # same shape, no gripper term
    reward = dist − last  (positive when the gripper moves away; see resolved KI‑3)
```

| Reward | mapped fn | formula |
|---|---|---|
| `AbsoluteApproachReward` | `distance_reward` | `distance_reward(grip.loc, obj.loc, increase=False)` |
| `AbsoluteWithdrawReward` | `distance_reward` | `… increase=True` (reward moving away) |
| `AbsoluteMoveReward` | `distance_reward` | `… (obj.loc, location.loc, increase=False)` |
| `AbsoluteRotateReward` | `rotate_reward` | `rotate_reward(obj, angle)` |
| `SimpleGraspReward` | `gripper_close_reward` | `…(grip, obj, open=False)` |
| `SimpleDropReward` | `gripper_open_reward` | `…(grip, obj, open=True)` |
| `SimpleTransformReward` / `SimpleFollowReward` | — | constant `1` (placeholder) |
| `NearReward` | `euclidean_distance`, `near_threshold` | `dist(A,B) − near_threshold` |

**Enum wrappers** map a *variant name* to a concrete reward **class**, and an action picks
one as its `REWARD_CLASS` (§5):

```
ApproachReward.{RELATIVE,ABSOLUTE}  WithdrawReward.{RELATIVE,ABSOLUTE}
GraspReward.SIMPLE  DropReward.SIMPLE  MoveReward.ABSOLUTE  RotateReward.ABSOLUTE
TransformReward.SIMPLE  FollowReward.SIMPLE
```

### 4.5 `operands.py`

A vestigial stub (only an `abc` import). The real `Operand`/`LogicalOperand` live in
`core.py`. No functional content.

---

## 5. Actions (`actions.py`)

Each `AtomicAction` is itself a `Predicate`, so it can be decided. It bundles three
instance fields populated in `__init__`, then calls `setup_reward()`:

- `self._initial` — **precondition** (`LogicalOperand`); `can_be_executed()` = `_initial.decide()`.
- `self._predicate` — **goal/effect** (`LogicalOperand`); action‑as‑operand and `goal`/`predicate` props = `_predicate.decide()`.
- `self._reward` — built by `setup_reward()` = `self.REWARD_CLASS.value(**self.variables)` — i.e. the chosen reward class instantiated **with the action's variables splatted as kwargs**. (Hence the reward's `_VARIABLES` keys must match the action's variable names — the source of the bug fixed earlier for `Rotate`/`Move`.)

STRIPS view (preconditions are right‑nested `ParallelAndOp`; `¬` = `NotOp`):

| Action | Variables | Precondition | Goal/effect | Reward |
|---|---|---|---|---|
| **Approach** | gripper, object | `IsReachable(g,obj) ∧ (GripperOpen(g) ∧ ¬GripperAt(g,obj))` | `GripperAt(g,obj)` | ApproachReward.RELATIVE |
| **Withdraw** | gripper, object | `GripperAt(g,obj) ∧ GripperOpen(g)` | `¬GripperAt(g,obj)` | WithdrawReward.RELATIVE |
| **Grasp** | gripper, object | `GripperAt(g,obj) ∧ GripperOpen(g)` | `IsHolding(g,obj) ∧ ¬GripperOpen(g)` | GraspReward.SIMPLE |
| **Drop** | gripper, object | `GripperAt(g,obj) ∧ ¬(¬IsHolding(g,obj) ∧ GripperOpen(g))` | `¬IsHolding(g,obj) ∧ GripperOpen(g)` | DropReward.SIMPLE |
| **Move** | gripper, object, location | `IsHolding(g,obj) ∧ IsReachable(g,loc)` | `ObjectAt(obj,loc)` | MoveReward.ABSOLUTE |
| **Rotate** | gripper, object, angle | `IsHolding(g,obj) ∧ Exists(angle)` | `ObjectAtPose(obj,angle)` | RotateReward.ABSOLUTE |
| **Transform** | gripper, object, location | `IsHolding(g,obj) ∧ IsReachable(g,loc)` | `ObjectAt(obj,loc)` | TransformReward.SIMPLE |
| **Follow** | gripper, object, location | `IsHolding(g,obj) ∧ IsReachable(g,loc)` | `ObjectAt(obj,loc)` | FollowReward.SIMPLE |

> **Move / Transform / Follow are identical at the logic layer** (same variables,
> precondition, and `ObjectAt` goal); they differ only in `REWARD_CLASS`. Differentiation
> is purely in the reward.

---

## 6. The generation engine

This is the heart of the system and the part most worth understanding.

### 6.1 The central idea — "world state = the symbolic cache"

RDDL is **not** a classical planner that searches over a propositional state. It is a
**constructive forward sampler riding on the symbolic decide‑cache**:

- The world state is exactly the contents of a `StrictSymbolicCacheContainer`'s
  decide‑cache: a map `predicate(args) → bool`. Closed world: anything not in the cache
  decides **`False`**.
- An action's precondition is not *matched* against the state — it is **made true** by
  writing it into the cache (`set_symbolic_value(True)`), then `decide()` is called as a
  **consistency check**: because `And`/`Not` decide over their children's cached values, the
  root returns `False` iff some child was already forced to a contradicting value by an
  earlier action's effect. That catches cross‑step inconsistency.
- The effect is applied by `predicate.set_symbolic_value(True)`, mutating the cache so the
  *next* action observes the post‑state.
- A `deque` of cache **clones** is the backtracking stack: duplicate‑and‑push opens a
  speculative branch; pop discards it; an accepted step keeps its branch as the base for the
  next step.

### 6.2 `RDDLWorld` (`rddl_sampler.py:31`)

Key state: `_symbolic_table_stack` (deque of `StrictSymbolicCacheContainer` clones; top =
current world), `_rule_book`, `_weighter` (weighted action ordering, §6.4), allowed
entity/action pools, `_object_weights`, and `__real_cache` (saves the normal cache while
symbolic mode is active). Default pools exclude `Withdraw` from the **initial** action set
(it needs a prior `Approach`).

Cache/branch primitives:
- `activate/deactivate_symbolic_mode` — swap the global `Operand` cache to/from the symbolic table.
- `_symbolic_cache_duplicate_and_stack` — `clone()` top, push, point cache at it (**new branch**).
- `_symbolic_cache_pop` — discard top clone (**backtrack**; refuses to pop the last).
- `sample_object_subclass(type)` — `RNG.choice` over allowed concrete subclasses, weighted; then decays that class's weight (to 0 if single‑object‑per‑class, else ×0.9).

### 6.3 The generation loop — `sample_generator` (`rddl_sampler.py:95`)

A Python generator yielding `(action, post_state_snapshot)` per accepted step.
`sample_world(L)` drains it into lists and wraps them in an `RDDLTask`.

```
generator sample_generator(L, add_robots=True, retry_ad_infinitum=True):
    assert state == IDLE                       # re-entrancy guard
    _initialize_world(add_robots)              # fresh symbolic world + one Gripper variable
    Weighter.RETRY_AD_INFINITUM = retry_ad_infinitum

    for idx in 0 .. L-1:
        action_gen = (idx==0) ? weighter.get_initial_generator()   # legal first actions only
                              : weighter.get_random_generator()     # weighted, sequence-aware

        accepted = False
        while not accepted:
            action = next(action_gen)                    # weighted-random candidate (instantiated)
            free   = action.gather_variables()           # the action's empty slots
            _symbolic_cache_duplicate_and_stack()        # branch off current world
            added  = table.lookup_and_link_variables(free, sample_object_subclass)  # BIND slots

            if idx == 0:                                 # establish a consistent INITIAL state
                action.initial.set_symbolic_value(True)  # force WHOLE precondition true
                ok = action.initial.decide()
                if ok:
                    for v in added:                      # hot-fix: make graspables reachable
                        if v is GraspableObject: IsReachable(gripper, v).set_symbolic_value(True)
                    assert action.initial.decide()
                    initial_state = table.clone()        # SNAPSHOT initial state
            else:                                        # idx > 0
                if added:                                # only force facts about NEW objects;
                    action.initial.set_symbolic_value(True, only_if_contains=set(added))
                ok = action.initial.decide()             # facts about OLD objects must already hold

            if ok: accepted = True
            else:                                        # reject → roll back
                weighter.penalize(action)                # downweight action / transition
                _remove_variables(added); _symbolic_cache_pop()

        action.predicate.set_symbolic_value(True)        # APPLY EFFECT → mutates current cache
        yield action, table.clone()                      # emit (action, post-state)
        weighter.add_and_penalize(action)                # learn transition weights
        # branch is NOT popped → it is the base for idx+1

    goal_state = table.clone()                           # final cache = goal state
    deactivate_symbolic_mode(); reset_stack()
    return idx == L
```

**Why step 0 differs from step > 0:** step 0 forces the *entire* precondition true (a free
world, no prior facts). For later steps, only sub‑predicates that mention **newly created**
variables are freely asserted (`only_if_contains=set(added)`); preconditions about
**pre‑existing** objects must already be satisfied by accumulated effects. This is precisely
the mechanism that ties an action's precondition to the prior state.

### 6.4 Entity binding & action ordering (`sampling_utils.py`)

- **`StrictSymbolicCacheContainer`** — the world‑state container. Adds a
  `_variable_registry: {symbolic_id → Variable}`; `_make_key` refuses keys over unregistered
  variables (no dangling vars). `get_predicates()` materializes the cache into
  `[(PredicateClass, [arg names], bool)]` (the human/consumer state read‑out). `clone()` is
  the snapshot/branch primitive.

- **`lookup_and_link_variables(free_vars, sampler)`** — binds each action slot, *reusing an
  existing compatible world entity when possible*, else creating & registering a new one;
  returns only the **newly created** vars (for rollback). Reuse is how the sequence chains
  objects across actions (the same gripper/apple recurs) and is the implicit symmetry/dedup
  mechanism.

```
function lookup_and_link_variables(free, sample):
    new = []; linked = []
    for slot in free:
        cand = first registered var whose type ⊑ slot.type and not already in linked
        if cand is None:
            cand = SymbolicEntity(sample(slot.type)); register(cand); new.append(cand)
        slot.link_to(cand)          # alias slot → chosen entity (shared global name)
        linked.append(cand)
    return new
```

- **`Weighter`** — a stochastic action‑ordering oracle. Holds per‑action `_weights`, an
  `_initial_weights` mask (legal first actions), a 3‑deep history queue, and an
  `nx.MultiDiGraph` of learned **bigram/trigram transition penalties**. It (a) honours
  per‑action weights, (b) restricts the first action to the legal initial set, (c) learns to
  avoid recently‑seen or failed action pairs/triples (`_get_seq_weights` multiplies base
  weight by learned edge weights), and (d) either gives up after one pass (`recurse`, finite)
  or retries forever (`sample_generator`). Modes are bit‑flags
  (`MODE_INITIAL|WEIGHT|SEQUENCE|RANDOM|MAX_NOISE`). The candidate stream `_get_random_item`
  instantiates `choices[i % n]()` and yields it.

### 6.5 Alternate engine — `recurse_generator` (`rddl_sampler.py:194`)

A **DFS enumerator** producing many distinct tasks rather than one. Same per‑step logic, but
`RETRY_AD_INFINITUM=False` so each choice list is traversed once (finite tree), backtracking
via the Python call stack and immutable `seq + [action]` extensions. Runs on a daemon
`Thread` feeding a bounded `Queue` consumed by the generator. The producer is wrapped by
`_recursive_sampling_entry`, whose `finally` always enqueues a `None` sentinel so the
consumer terminates cleanly once the tree is exhausted (resolved KI‑6).

### 6.6 Randomness, termination, backtracking

- **Seeding:** one shared `np.random.Generator` `RDDLWorld.RNG` (seeded once at import);
  `set_seed(seed)` re‑seeds; the `Weighter` shares the same stream. Randomness enters at
  action ordering, object‑subclass choice, and reuse‑vs‑create.
- **Termination:** `sample_generator` ends at `idx == L`; `recurse_generator` ends when the
  DFS tree is exhausted or the requested sample count is reached.
- **Backtracking (3 layers):** (1) per‑candidate rejection rolls back bindings + cache clone
  and downweights; (2) `RETRY_AD_INFINITUM` cycles the shuffled action list so a step never
  hard‑fails (the only fatal error is an empty initial action space); (3) the cache‑clone
  stack is the explicit branch frontier. There is no global restart; consistency is
  maintained incrementally. The `IsReachable` hot‑fix is the only safety net for step 0.

---

## 7. Task data structures

There are **two** `RDDLTask` definitions:

- **`task.py`** — an abstract/legacy skeleton (`actions, objects, initial_state,
  goal_state`); its `current_action`/`next_action`/`current_reward` raise
  `NotImplementedError`. Interface only — not what the sampler produces.

- **`rddl_task.py`** — the concrete task the sampler builds:
  `RDDLTask(actions, states, initial_state, world_generator=None)`.
  - `_actions` — ordered action sequence.
  - `_states` — per‑step symbolic states (post‑state of each action); `_final_state = states[-1]`.
  - `_initial_state`; `_current_action/_current_state` cursors.
  - `_world_generator` — back‑reference enabling object re‑sampling.

  Key behavior: `global_goal` = last action's `predicate`; `current_reward` =
  `current_action.compute_reward()` (→ `reward.evaluate()`); `gather_objects()` =
  `list(_final_state.variables.values())` (objects are **derived**, not stored);
  `get_generator()` walks `zip(actions, states)` for playback; `regenerate_objects()`
  re‑samples concrete objects against a fresh symbolic container; `show_current_state()` /
  `initial_state.show_table()` / `final_state.show_table()` render tabular state. **There is
  no built‑in YAML/`__str__` serializer** in the task classes — `test_world.yaml` is built
  ad hoc by the entry point (§11).

---

## 8. The simulator bridge

The abstract layer is bound to a concrete "simulator" through the `_0_*` registry. The
reference harness (`tests/testing_utils.py`) installs it **at import time**, and the order
is load‑bearing: mapping + observation getter + monkey‑patch must precede any sampling.

```
Operand.set_mapping({
    "euclidean_distance": lambda A,B: np.linalg.norm(A-B),
    "near_threshold":     0.1,                       # a CONSTANT, not a callable
    "is_holding":         lambda g,o: dist(g,o) < 0.1,
    "is_reachable":       lambda g,o: True,          # stub
    "gripper_at":         lambda g,o: g.location == o.location,
    "gripper_open":       lambda g: random() < 0.5,  # stub
    "object_at":          lambda g,o: g.location == o.location,
    "exists":             lambda e: True,            # stub
    "on_top":             lambda a,b: random() < 0.5,# stub
    "distance_reward":    lambda g,o: random(),      # stub reward
    "gripper_open_reward":lambda g: random(),        # stub reward
    "gripper_close_reward":lambda g: random(),       # stub reward
    "rotate_reward":      lambda g,o: random(),      # stub reward
})
Entity.set_observation_getter(lambda self: self)                       # each entity is its own observation
Location.monkey_patch(Location._get_location, lambda self: self.get_position())  # entity → coords
```

**Required keys** (union over predicates + rewards): `euclidean_distance`, `near_threshold`,
`gripper_at`, `gripper_open`, `is_reachable`, `is_holding`, `object_at`, `object_at_pose`,
`exists`, `on_top`, `distance_reward`, `gripper_open_reward`, `gripper_close_reward`,
`rotate_reward`.
`Operand.list_required_mappings()` enumerates them authoritatively. **Note:** in the test
harness most gating predicates and **all** rewards are random stubs, so the harness exercises
the *structural sampling* logic, not real physics. A production user replaces these callables
(and the `Location` getter / `Entity` observation getter) with real simulator hooks; the
inert `EnvObjectProxy` / `EnvSimulator` / `Observer` classes in `testing_utils.py` document
the intended seams.

Concrete entities are defined by **multiple inheritance** (`Apple`, `Bowl`, `Orange`, `Can`,
`CerealBox` ⟵ `GraspableObject, EnvObjectProxy`; `TiagoGripper` ⟵ `Gripper, EnvObjectProxy`),
so each gains both RDDL semantics and a (random) position from the proxy.

---

## 9. Rule book

`RuleBook` (`rule_book.py`) is a **constraint registry** keeping symbolically‑sampled states
logically consistent. It indexes *predicate classes* (membership tested by `__class__`):

- **`ExclusivityRule(*predicates)`** — at most one may be true: `check()` returns `False` if
  any pair both `decide()` true.
- **`Consequent(exemplar ⇒ consequences)`** — `apply()`: if `exemplar.decide()`, force each
  consequence true via `set_symbolic_value(True)`.
- **`RuleBook`** — buckets rules into `_exclusivity_rules` / `_consequential_rules`;
  `check_consistency()` = `all(rule.check())`; `apply_rules()` forward‑chains consequents.
  It reads the live state via `Operand.get_cache().get_predicates()`.

The book is **built** externally (`rule_book.add_rule(ExclusivityRule(OnTop, Near))`) — no
rules ship by default. `_construct_exclusivity_predicates` and `check_if_breaks_consistency`
are stubs (see §13).

---

## 10. The textual DSL parser

`RDDLParser` (`rddl_parser.py`) is a small recursive‑descent / shunting parser for **textual
predicate expressions** (action condition specs like `GripperAt(gripper, object) and
not(...)`). It is constructed with three dictionaries (`combinator_mapping` op→Operator,
`predicate_mapping` name→Predicate, `type_definitions` name→Entity). `parse()` alternately
matches a predicate then an operator, pushing onto operand/operator stacks, then folds
operators by their `ARITY` into a nested `Operator(...)` tree.

It is the **action‑definition front‑end**, not a serializer and not on the sampling path: the
`RDDL` facade (`rddl_sampler.py:421`) uses it in `load_definitions(...)` to dynamically build
`AtomicAction` subclasses from a `{action: {"predicate": "<expr>"}}` spec via
`type(name, (AtomicAction,), {...})`. Argument binding is now implemented (resolved KI‑7):
`create_predicate` resolves each `name:Type` token to a `Variable` (reused by name within an
expression) and maps them to the predicate's `_VARIABLES`, and `parse_action_predicate`
delegates to `parse()`. (Note the parser's tokeniser still has rough edges — see open KI‑18.)

---

## 11. User manual

### 11.1 Install & run

```bash
pip install .            # or: pdm install   (editable; points rddl.pth at this src)
python tests/test_world.py
```

`tests/test_world.py` runs `test_world(sequence_lengths=range(2,11), n_repeats=10,
output_path="test_world.yaml")`, which for each length samples `n_repeats` tasks and dumps a
nested dict to YAML. The per‑repeat record schema:

```yaml
<sequence_length>:
  - action_list: [Approach, Grasp, Move, ...]        # ordered class names
    sequence:
      - {action: "Approach(gripper: Gripper, object: GraspableObject)",
         action_objects: {gripper: <var name>, object: <var name>}}
    all_objects: {<var name>: <EntityClassName>, ...}
```

### 11.2 Using the library in your own code

The order below is mandatory — the `_0_*` resolution and entity instantiation both fail if
their prerequisites aren't installed first.

**(a) Register simulator functions & observation hooks (before importing domain classes):**

```python
from rddl import Operand, Entity
from rddl.entities import Location
Operand.set_mapping({ ... })                  # every _0_* key you will use (see §8)
Entity.set_observation_getter(lambda self: self)            # or a real Observer
Location.monkey_patch(Location._get_location, lambda self: self.get_position())
```
Use `Operand.list_required_mappings()` to discover exactly which keys are needed.

**(b) Define entities** by mixing an RDDL base with a geometry‑providing backend:

```python
class Apple(GraspableObject, EnvObjectProxy):
    def __init__(self, reference=None):
        super().__init__(self._get_generic_reference() if reference is None else reference, "apple")

class TiagoGripper(Gripper, EnvObjectProxy):
    def __init__(self, reference=None):
        super().__init__("gripper_tiago" if reference is None else reference)
```

**(c) Generate tasks:**

```python
from rddl_sampler import RDDLWorld          # adjust import to your layout

world = RDDLWorld()                          # optional: RDDLWorld(sample_single_object_per_class=True)
# optional constraint: world.rule_book.add_rule(ExclusivityRule(OnTop, Near))
# optional scoping:    world.set_allowed_entities([...]); world.set_allowed_actions([...])
task = world.sample_world(sequence_length=5)

for a in task.get_actions():
    print(a.__class__.__name__, {role: v.name for role, v in a.variables.items()})
print(task.gather_objects())                 # participating Variables / objects
```

**Alternative APIs:**
- Incremental: `gen = world.sample_generator(L)`, `next(gen)` per step; reject the last step
  with `gen.send(False)`.
- Deterministic multi‑task: `gen = world.recurse_generator(L)`, `task = next(gen)`; inspect
  with `task.initial_state.show_table()` / `task.final_state.show_table()`; re‑roll objects
  with `task.regenerate_objects()`.
- Reproducibility: `RDDLWorld.set_seed(seed)`; diversity tuning: `world.reset_weights(mode)`.

### 11.3 Adding a new action

1. Pick/author a `Reward` subclass and wrap it in an `Enum` (e.g.
   `class FooReward(Enum): SIMPLE_REWARD = SimpleFooReward`). The reward's `_VARIABLES` keys
   **must equal** the action's variable names (this matching is enforced and is the bug class
   fixed for `Rotate`/`Move`).
2. Define the action:

```python
class Foo(AtomicAction):
    _VARIABLES = {"gripper": Gripper, "object": GraspableObject}
    REWARD_CLASS: Enum = FooReward.SIMPLE_REWARD
    def __init__(self, **kwds):
        super().__init__(**kwds)
        g, o = self.get_argument("gripper"), self.get_argument("object")
        self._initial   = ParallelAndOp(left=GripperAt(gripper=g, object=o), right=...)
        self._predicate = IsHolding(gripper=g, object=o)
        self.setup_reward()                  # MUST be last
```
3. Add it to the world's action pool (and the initial pool only if it can legally start a task).

---

## 12. API quick reference

| Symbol | Where | Purpose |
|---|---|---|
| `Operand.set_mapping(dict)` | core | Register simulator callables for `_0_*` keys. |
| `Operand.list_required_mappings()` | core | All `_0_*` keys the domain needs. |
| `Entity.set_observation_getter(fn)` | core | Required before any entity is created. |
| `Entity.monkey_patch(method, alt)` | core | Swap an existing method on a class tree. |
| `Variable.bind/unbind/link_to/global_rename` | core | Manage entity bindings. |
| `predicate.decide()` | core | Evaluate a predicate/operator (bool). |
| `reward.evaluate()` | core | Evaluate a reward (float). |
| `operand.set_symbolic_value(v, only_if_contains=)` | core | Assert a symbolic truth value. |
| `RDDLWorld()` | sampler | The world/task generator. |
| `.sample_world(sequence_length=)` | sampler | One task. |
| `.sample_generator(L)` / `.recurse_generator(L)` | sampler | Incremental / DFS multi‑task generators. |
| `.set_allowed_{entities,actions,predicates}([...])` | sampler | Scope the universe. |
| `RDDLWorld.set_seed(seed)` | sampler | Reproducibility. |
| `task.get_actions()` / `.gather_objects()` | task | Sequence / participating objects. |
| `task.initial_state` / `.final_state` (`.show_table()`) | task | Symbolic states. |
| `task.regenerate_objects()` | task | Re‑sample concrete objects. |
| `rule_book.add_rule(ExclusivityRule|Consequent)` | rule_book | Add consistency constraints. |

---

## 13. Known issues & rough edges

Each issue has a stable ID (**KI‑n**) for cross‑reference, a severity, the precise location,
the symptom, the root cause, and a **proposed solution that fixes or extends the current
behavior without removing functionality** and stays consistent with the established
patterns (string‑keyed `_0_*` hooks, `_VARIABLES`‑declared operands, symbolic cache as
state, tree‑recursive `gather_variables`/`set_symbolic_value`). Severities reflect impact
once the affected path is exercised.

**Severity legend:** 🔴 correctness bug (wrong/crashing result on a live path) · 🟠
robustness (hang/latent crash under specific use) · 🟡 incomplete feature (WIP, currently
inert) · ⚪ maintainability (no behavioral impact).

**Status legend:** ✅ resolved (implemented in the source) · ⬜ open.

> **How to read this register.** Each heading is tagged ✅ or ⬜. For a **✅ resolved**
> entry, the Location/Symptom/Root‑cause describe the *original* problem and the **Fix**
> shows what was applied — the current source already matches the fix (and the factual
> sections above, e.g. §4.2 and §8, reflect the post‑fix code). For a **⬜ open** entry the
> code is still as described. Resolved so far: KI‑1 … KI‑9, KI‑14 … KI‑16 (verified — default
> `test_world.py` clean, targeted smoke tests pass). Open: KI‑10 … KI‑13 and the newly
> recorded KI‑17 … KI‑21. The cross‑reference table at the end records per‑issue status.

### Correctness bugs

#### KI‑1 ✅ 🔴 `Exists.__call__` reads a non‑existent attribute
- **Location:** `predicates.py:96,102` — `_VARIABLES = {"entity": Entity}` but
  `return Exists._0_EXISTS_CHECK(self.object)`.
- **Symptom:** any `decide()` of an `Exists` predicate raises `AttributeError` (`self.object`
  is never created; the property generated from `_VARIABLES` is `self.entity`). `Rotate`'s
  precondition uses `Exists(entity=angle)`, so this fires the moment a `Rotate` precondition
  is decided in NORMAL mode. It is currently masked only because the sampler asserts
  preconditions symbolically (the truth table short‑circuits `__call__`).
- **Root cause:** attribute name drift between `_VARIABLES` key and `__call__` body.
- **Fix:** rename the access to the declared variable.
  ```python
  def __call__(self):
      return Exists._0_EXISTS_CHECK(self.entity)
  ```

#### KI‑2 ✅ 🔴 `ObjectAtPose` borrows another predicate's hook and misuses it
- **Location:** `predicates.py:84,91` — declares its own `_0_OBJECT_AT_CHECK = "object_at"`
  yet calls `ObjectAt._0_OBJECT_AT_CHECK(self.object, self.angle)`.
- **Symptom:** it invokes **`ObjectAt`'s** resolved callable (works only because both happen
  to map to `"object_at"`) and feeds an `AbstractRotation` into a slot the simulator's
  `object_at(object, location)` expects to be a `Location` — semantically wrong, and a hidden
  coupling that breaks if either mapping changes.
- **Root cause:** copy‑paste from `ObjectAt`; pose‑checking was never given its own hook.
- **Fix (extends the hook vocabulary, the idiomatic move):** give the predicate a dedicated
  mapped function and call its own attribute.
  ```python
  class ObjectAtPose(SpatialPredicate):
      _0_OBJECT_AT_POSE_CHECK: ClassVar[Union[Callable, str]] = "object_at_pose"
      _VARIABLES = {"object": ObjectEntity, "angle": AbstractRotation}
      def __call__(self):
          return ObjectAtPose._0_OBJECT_AT_POSE_CHECK(self.object, self.angle)
  ```
  Register `"object_at_pose"` in the simulator `mapping` (e.g. compares object orientation to
  the target rotation). This adds a capability rather than overloading `object_at`.

#### KI‑3 ✅ 🔴 `RelativeWithdrawReward` rewards the wrong direction
- **Location:** `rewards.py:111` — `reward = self._last_distance - dist`.
- **Symptom:** the delta is **positive when the gripper moves *closer***, identical in sign to
  `RelativeApproachReward`; withdrawing (distance increasing) yields a negative reward. The
  reward contradicts the action's intent.
- **Root cause:** the approach formula was reused verbatim; only `AbsoluteWithdrawReward`
  encodes direction (via `increase_distance=True`).
- **Fix:** invert the delta so growing distance is rewarded, mirroring the approach reward's
  structure (and optionally add the symmetric gripper term the approach reward has):
  ```python
  reward = dist - self._last_distance        # positive when moving away
  ```

#### KI‑4 ✅ 🔴 `NAryAndOp` cannot gather its variables
- **Location:** `operators.py:133` — overrides `__decide__`/`__evaluate__`/`set_symbolic_value`
  but not `gather_variables`; `NAryOperator` (`operators.py:56`) doesn't define it either, so
  it inherits the abstract `LogicalOperand.gather_variables` (`core.py:759`).
- **Symptom:** calling `gather_variables()` on any n‑ary AND raises `NotImplementedError` —
  which the sampler does for every action precondition. Only the binary `ParallelAndOp` is
  used today, so it is latent, but it blocks adopting the n‑ary operator.
- **Root cause:** missing override on the n‑ary branch.
- **Fix (put it on the base so every future n‑ary operator inherits it, matching
  `BinaryOperator`'s gating):**
  ```python
  class NAryOperator(Operator):
      def gather_variables(self) -> list[Variable]:
          out = []
          for op in self._operands:
              if isinstance(op, LogicalOperand):
                  out += op.gather_variables()
          return out
  ```

#### KI‑5 ✅ 🔴 `NearReward` bypasses the `Reward` machinery
- **Location:** `rewards.py:222‑233` — no `_VARIABLES`; `__init__` sets `self.object_A/​_B`
  directly and **does not call `super().__init__()`**.
- **Symptom:** its variables are never registered in `Reward.__variables`, so the inherited
  property proxies, `get_argument`, `get_relevant_entities`, and `setup_reward`‑style
  instantiation don't work; it also can't be selected via an `Enum`/`REWARD_CLASS` like every
  other reward. A stray `print` fires on every call (`rewards.py:231`).
- **Root cause:** written before the `_VARIABLES` + constructor‑validation contract was
  standardized (§4.4).
- **Fix:** bring it onto the standard contract. Use **concrete** annotations (the
  `__init_subclass__` validator checks `issubclass(annotation_arg, _VARIABLES_type)`, which
  fails for the `LocationType` `TypeVar`):
  ```python
  class NearReward(Reward):
      _0_EDISTANCE_PREDICATE: Callable = "euclidean_distance"
      _0_NEAR_THRESHOLD: Callable = "near_threshold"
      _VARIABLES = {"object_A": Location, "object_B": Location}
      def __init__(self, object_A: Variable[Location], object_B: Variable[Location]) -> None:
          super().__init__(object_A=object_A, object_B=object_B)
          self._a, self._b = object_A, object_B
      def __call__(self):
          return self._0_EDISTANCE_PREDICATE(self._a.location, self._b.location) - self._0_NEAR_THRESHOLD
  ```
  Optionally add a `class NearReward(Enum): SIMPLE_REWARD = NearReward`‑style wrapper if it is
  ever to be attached to an action.

### Robustness

#### KI‑6 ✅ 🟠 `recurse_generator` can block forever after the DFS tree is exhausted
- **Location:** `rddl_sampler.py:243‑257`. The worker thread runs `_recursive_sampling` and
  returns when the tree is exhausted, but **never enqueues a sentinel**; the consumer loops on
  `self.__action_state_stack.get(block=True)` (no timeout), so the `if output is None: break`
  branch (`:256`) and the `except Empty` branch (`:253`, unreachable — no timeout is passed)
  never trigger.
- **Symptom:** once the finite tree is drained (and fewer than `n_samples_requested` tasks were
  produced), `recurse_generator` hangs on an empty queue.
- **Root cause:** producer/consumer termination handshake is incomplete.
- **Fix (additive — finalize the producer; keep the `None` sentinel the consumer already
  checks for):** wrap the thread target so it always posts a sentinel, even on error.
  ```python
  def _recursive_sampling_entry(self, *args):
      try:
          self._recursive_sampling(*args)
      finally:
          self.__action_state_stack.put(None)      # consumer's `if output is None: break` now fires
  # ...
  generator_thread = Thread(target=self._recursive_sampling_entry, args=(0, [], []),
                            daemon=True, name="rddl_generator")
  ```
  (Keeps the unbounded `get(block=True)`; the sentinel guarantees termination. If a timeout is
  also wanted, pass one to `get` so the existing `except Empty` path becomes live.)

### Incomplete features (currently inert WIP)

#### KI‑7 ✅ 🟡 Text DSL → action path is stubbed
- **Location:** `rddl_parser.py:62‑75`. `create_predicate` hardcodes `true_args = None` then
  splats it (`self._get_operand(predicate)(*None)` would raise); `parse_action_predicate`
  parses predicates but never folds operators or returns an expression. The `RDDL` facade's
  `load_definitions`/`_extract_action` therefore can't build action classes.
- **Symptom:** defining actions from text (the intended front‑end, §10) does not work.
- **Root cause:** argument‑binding and operator‑folding were left unimplemented.
- **Fix (complete it, reusing the working `parse()` folding):**
  1. Resolve args to `Variable`s using `type_definitions`. The DSL token grammar already has
     `var_ex = name:type`; bind each arg to `Variable(self.type_definitions[type], base_name=name)`,
     caching by name within one expression so repeated names share a binding (the same
     `link_to` identity model as the sampler, §6.4):
     ```python
     def create_predicate(self, predicate, args):
         true_args = [self._resolve_arg(a) for a in args]   # a == "name:Type" or a known name
         return self._get_operand(predicate)(*true_args)
     ```
  2. Make `parse_action_predicate` delegate to `parse()` (which already folds by `ARITY`,
     `rddl_parser.py:95‑101`) and return the root operand, so `_extract_action` receives a real
     `_predicate`.

#### KI‑8 ✅ 🟡 `RuleBook` predictive consistency check is unimplemented
- **Location:** `rule_book.py:81‑96`. `_construct_exclusivity_predicates` and
  `check_if_breaks_consistency` fetch the current predicate set then fall through with empty
  bodies / no return.
- **Symptom:** the sampler can only *retroactively* validate (`check_consistency()` over already
  asserted facts) and forward‑chain (`apply_rules()`); it cannot ask "would asserting these
  candidate predicates violate an exclusivity rule?" before committing — so exclusivity
  constraints aren't enforced during generation.
- **Root cause:** the look‑ahead method bodies were never written.
- **Fix (implement the predictive check on top of the existing `get_predicates()` read‑out and
  `Rule.__contains__`):**
  ```python
  def check_if_breaks_consistency(self, *candidates: LogicalOperand) -> bool:
      true_now = {type(p) for p, _vars, val in self._get_current_predicate_set() if val}
      for cand in candidates:
          for rule in self._exclusivity_rules:
              if cand in rule and any(other in rule and other is not type(cand) for other in true_now):
                  return True            # candidate conflicts with a currently-true exclusive sibling
      return False
  ```
  Then call it in `sample_generator`/`_recursive_sampling` right before
  `action.predicate.set_symbolic_value(True)` and treat a `True` result as a rejection
  (reusing the existing penalize/rollback path). This extends generation with real constraint
  enforcement.

#### KI‑9 ✅ 🟡 `ExclusivityRule.list_breaking_predicates` is empty
- **Location:** `rule_book.py:37‑41` — loops over `_exclusive_predicates` with a bare `pass`,
  returns `None`.
- **Symptom:** callers can't learn *which* predicates conflict (useful for targeted
  backtracking / diagnostics).
- **Root cause:** unimplemented.
- **Fix:** return the currently‑true members of the rule other than the exemplar.
  ```python
  def list_breaking_predicates(self, exemplar, variables=None):
      return [p for p in self._exclusive_predicates
              if p is not type(exemplar) and p.decide()]
  ```

### Maintainability (no behavioral impact)

#### KI‑10 ⬜ ⚪ Shared mutable `Operand.__slots__` leaks slots across subclasses
- **Location:** `core.py:590` (`__slots__ = []`) with in‑place appends at `core.py:813‑814`
  (Predicate) and `965‑966` (Reward).
- **Symptom:** every Predicate/Reward subclass appends its variable names to the **one** list
  object shared by `Operand`, so slot membership accumulates module‑wide; the memory‑saving
  intent of `__slots__` is defeated and unrelated classes share slot names.
- **Root cause:** mutating the inherited class attribute instead of creating a per‑class one.
- **Fix (preserve the proxy mechanism, give each subclass its own slot list before appending):**
  ```python
  # in Predicate.__init_subclass__ / Reward.__init_subclass__, before the append loop:
  if "__slots__" not in cls.__dict__:
      cls.__slots__ = []
  # then: if name not in cls.__slots__: cls.__slots__.append(name)
  ```

#### KI‑11 ⬜ ⚪ `_CacheKey` field labels / `__init__` params are swapped vs `__new__`
- **Location:** `core.py:455‑485`. `key_tuple` fields are `['arg_list', 'class_name']` but
  `__new__` builds `key_tuple(class_name, tuple(arg_list))`, and `__init__`'s signature is
  `(self, arg_list, class_name, ...)` — reversed from `__new__`'s `(cls, class_name, arg_list,...)`.
- **Symptom:** none functionally — keys are constructed and hashed positionally, and the
  `class_name`/`arg_list` *properties* (which index `self[0]`/`self[1]`) happen to return the
  right slots. But the names are actively misleading to readers/maintainers.
- **Root cause:** namedtuple field order doesn't match construction order.
- **Fix:** align names for clarity — reorder the namedtuple to `['class_name', 'arg_list']` and
  make `__init__`'s parameter names match `__new__`. Pure rename; behavior unchanged.

#### KI‑12 ⬜ ⚪ Diagnostic `print`s on hot/definition paths
- **Location:** e.g. `Reward.__init_subclass__` argspec print (`core.py:947`), the `_0_*`
  registration print (`core.py:670`‑area), `NearReward.__call__` (`rewards.py:231`), and
  `decide()` debug output gated by `Operand.DEBUG_MODE`.
- **Symptom:** noisy stdout at import and during sampling (visible in every `test_world.py`
  run).
- **Root cause:** `print` used for debugging instead of a logger.
- **Fix (additive, keeps the information):** route through the `logging` module at `DEBUG`
  level (`logger = logging.getLogger("rddl")`), so output is opt‑in via log configuration and
  the existing `DEBUG_MODE` flag can set the logger level.

#### KI‑13 ⬜ ⚪ Two divergent `RDDLTask` definitions
- **Location:** `task.py:1‑19` (skeleton whose `current_action`/`next_action`/`current_reward`
  raise `NotImplementedError`) vs the concrete `rddl_task.py:13` used by the sampler.
- **Symptom:** import confusion; the `task.py` interface advertises methods the real task
  implements under different names.
- **Root cause:** an early interface sketch left in place.
- **Fix (turn it into the shared abstract base instead of a dead duplicate — additive):** make
  `task.RDDLTask` an ABC (the documented interface) and have `rddl_task.RDDLTask` subclass it,
  implementing the abstract methods. Keeps both files, removes the divergence, and gives the
  codebase a single typed contract.

#### KI‑14 ✅ ⚪ `operands.py` is an empty placeholder
- **Location:** `operands.py:1` (only `from abc import ABCMeta, abstractmethod`).
- **Symptom:** a module named after a central concept contains nothing; readers expect
  `Operand` here.
- **Root cause:** `Operand` ended up in `core.py`; the placeholder was never filled or wired.
- **Fix (additive, improves API discoverability without moving code):** re‑export the public
  operand ABCs for a stable import path —
  ```python
  from rddl.core import Operand, LogicalOperand   # public re-export
  ```
  (Leaves `core.py` as the source of truth; nothing is removed.)

#### KI‑15 ✅ ⚪ `IsHolding` hook declared without the `ClassVar` annotation
- **Location:** `predicates.py:62` — `_0_IS_HOLDING_FUNCTION = "is_holding"` (plain assignment)
  vs every sibling's `_0_… : ClassVar[Union[Callable, str]] = …`.
- **Symptom:** none (the `_0_` resolver keys on the name prefix, not the annotation); purely an
  inconsistency that can mislead tooling/readers.
- **Fix:** annotate it like the others — `_0_IS_HOLDING_FUNCTION: ClassVar[Union[Callable, str]] = "is_holding"`.

#### KI‑16 ✅ ⚪ `AbsoluteWithdrawReward` carries an unused field
- **Location:** `rewards.py:86` — `self.last_distance = None` is set but never read (the class
  is stateless/absolute).
- **Symptom:** dead state hinting at an intended (never‑implemented) relative behavior.
- **Fix:** either remove the assignment, or — consistent with "extend, don't remove" — actually
  use it to add optional delta shaping (guarded), matching `RelativeWithdrawReward` once KI‑3
  is fixed.

### Newly recorded (open)

Surfaced during the repo‑wide docstring pass; recorded for completeness, not yet fixed.

#### KI‑17 ⬜ 🟠 `_recursive_sampling` swallows errors then uses a stale `action`
- **Location:** `rddl_sampler.py` — `_recursive_sampling`'s candidate loop:
  `try: action = next(action_generator) except StopIteration: break except BaseException as e: print(e)`.
- **Symptom:** a non‑`StopIteration` error from `next(...)` is merely printed; control falls
  through and the code then calls `action.gather_variables()` on a possibly‑undefined or
  stale `action`, raising `NameError`/`UnboundLocalError` or silently reusing the previous
  candidate. Only the one‑shot `sample_generator` got the rule‑book/robustness attention; the
  DFS path retains this hole.
- **Root cause:** an over‑broad `except` that neither re‑raises nor `continue`s/`break`s.
- **Fix:** narrow the handler and skip the bad candidate, mirroring the one‑shot loop —
  ```python
  try:
      action = next(action_generator)
  except StopIteration:
      break
  except Exception as exc:          # log and move on to the next candidate
      warn(f"action sampling error: {exc}")
      continue
  ```

#### KI‑18 ⬜ 🟠 `match_predicate` can loop forever on malformed DSL input
- **Location:** `rddl_parser.py` — `match_predicate`'s argument loop: `while True: match, text =
  match_and_trim(arg_ex, text); if match: ... if ")" in end: break`.
- **Symptom:** the loop only breaks when an arg match's `end` group contains `)`. If `arg_ex`
  *fails* (malformed input — empty arg list, stray whitespace, missing close paren) the body
  neither appends nor breaks, spinning forever on unchanged `text`.
- **Root cause:** no `else: break` for the no‑match case; the close‑paren is the sole exit.
- **Fix (additive — make the parser fail fast on bad input):**
  ```python
  while True:
      match, text = RDDLParser.match_and_trim(RDDLParser.arg_ex, text)
      if not match:
          raise ValueError(f"malformed predicate arguments near: {text!r}")
      args.append(match.group('args'))
      if ")" in match.group('end'):
          break
  ```

#### KI‑19 ⬜ 🟠 Abstract operand bodies `return NotImplementedError(...)` instead of raising
- **Location:** `core.py` — e.g. `Reward.__call__` (and similar abstract stubs) do
  `return NotImplementedError(f"...")` rather than `raise NotImplementedError(...)`.
- **Symptom:** calling an unimplemented abstract returns the *exception instance* (a truthy
  object) instead of raising. A caller doing `if reward(): ...` or summing rewards would treat
  it as a value, masking the "not implemented" condition instead of failing loudly.
- **Root cause:** `return` where `raise` was intended.
- **Fix:** `raise NotImplementedError(...)` (these are `@abstractmethod` anyway, so subclasses
  are required to override — this only hardens the safety net).

#### KI‑20 ⬜ ⚪ `Weighter.__init__` re‑consumes its `items` iterable (and a redundant assignment)
- **Location:** `sampling_utils.py` — `Weighter.__init__` calls `np.asarray(items)` and then
  several dict comprehensions / `zip`s over `items`; it also assigns `self._initial_weights`
  once unconditionally and immediately overwrites it in the `if initial_weights is None` branch.
- **Symptom:** if a caller passes a one‑shot iterator/generator for `items`, only the first
  consumption sees data and the later weight dicts come out empty. Works for the list inputs
  used today. The first `_initial_weights` assignment is dead.
- **Root cause:** treating a generic `Iterable` as re‑iterable; a leftover assignment.
- **Fix:** materialise once at the top (`items = list(items)`) and use that list everywhere;
  drop the redundant first `_initial_weights = {...}`.

#### KI‑21 ⬜ ⚪ Minor dead code: ignored param, unused imports/vars, unreachable branch
- **Locations / symptoms:**
  - `rddl_sampler.py` `_add_variable(name, variable)` ignores `name` entirely (the dict write
    is commented out; only `register_variable` runs) — the parameter is misleading.
  - `rddl_sampler.py` `recurse_generator`'s `except Empty` branch is unreachable because the
    `get(block=True)` call passes no timeout (the timeout variant is commented out).
  - `rddl_task.py` imports `stat` and `Iterable` that are unused; the `state` loop variable in
    `regenerate_objects`/`get_generator` is bound but unused.
- **Fix:** drop the unused `name` parameter (or use it), remove the unused imports/loop vars,
  and either pass a timeout to `get` so the `Empty` branch becomes live or delete it. Pure
  cleanup; no behavioral change.

### Cross‑reference summary

| ID | Sev | Status | Area | One‑line fix |
|---|---|---|---|---|
| KI‑1 | 🔴 | ✅ done | predicates | `self.object` → `self.entity` in `Exists.__call__` |
| KI‑2 | 🔴 | ✅ done | predicates | give `ObjectAtPose` its own `object_at_pose` hook (+ registered the stub) |
| KI‑3 | 🔴 | ✅ done | rewards | invert delta: `dist - last` in `RelativeWithdrawReward` |
| KI‑4 | 🔴 | ✅ done | operators | add `gather_variables` to `NAryOperator` |
| KI‑5 | 🔴 | ✅ done | rewards | give `NearReward` `_VARIABLES` + `super().__init__()` (concrete annots) |
| KI‑6 | 🟠 | ✅ done | engine | enqueue `None` sentinel via `_recursive_sampling_entry` `finally` |
| KI‑7 | 🟡 | ✅ done | parser | `_resolve_arg` binding + `parse_action_predicate` delegates to `parse()` |
| KI‑8 | 🟡 | ✅ done | rule_book | implemented `check_if_breaks_consistency` + wired into `sample_generator` |
| KI‑9 | 🟡 | ✅ done | rule_book | implemented `list_breaking_predicates` |
| KI‑10 | ⚪ | ⬜ open | core | per‑class `__slots__` before appending |
| KI‑11 | ⚪ | ⬜ open | core | align `_CacheKey` field/param names |
| KI‑12 | ⚪ | ⬜ open | core/rewards | replace `print` with `logging` at DEBUG |
| KI‑13 | ⚪ | ⬜ open | task | make `task.RDDLTask` the ABC base of `rddl_task.RDDLTask` |
| KI‑14 | ⚪ | ✅ done | operands | re‑export `Operand`/`LogicalOperand` |
| KI‑15 | ⚪ | ✅ done | predicates | add `ClassVar` annotation to `IsHolding` hook |
| KI‑16 | ⚪ | ✅ done | rewards | dropped dead `AbsoluteWithdrawReward.last_distance` assignment |
| KI‑17 | 🟠 | ⬜ open | engine | narrow `_recursive_sampling`'s `except`; `continue` on a bad candidate |
| KI‑18 | 🟠 | ⬜ open | parser | `match_predicate`: raise on a non‑matching arg instead of looping |
| KI‑19 | 🟠 | ⬜ open | core | `raise` (not `return`) `NotImplementedError` in abstract bodies |
| KI‑20 | ⚪ | ⬜ open | sampling_utils | materialise `Weighter` `items` once; drop redundant assignment |
| KI‑21 | ⚪ | ⬜ open | engine/task | remove ignored param, unused imports/vars, unreachable branch |

> **Suggested order of attack:** KI‑1, KI‑3, KI‑4 were one‑line correctness fixes with
> immediate payoff. KI‑5 and KI‑2 unblocked the `Near`/pose features. KI‑6 then made
> `recurse_generator` safe to use, after which KI‑7/KI‑8/KI‑9 rounded out the WIP feature set
> (all ✅). Of the remaining open items, **KI‑17 / KI‑18 / KI‑19** are the worthwhile
> robustness fixes (they harden the DFS path, the DSL parser, and the abstract‑method safety
> net); the maintainability items (KI‑10 … KI‑13, KI‑20, KI‑21) can land opportunistically.

---

## 14. Glossary

| Term | Meaning |
|---|---|
| **Entity** | A thing in the world (gripper, object, location, rotation). |
| **Variable** | A typed placeholder bound to an entity; identity = its global name. |
| **Operand** | Anything that can `decide()` (bool) or `evaluate()` (float). |
| **Predicate** | A boolean state function over typed variables (external check). |
| **Operator** | A logical/scoring connective over operands (`And`, `Not`, `Seq`). |
| **Reward** | A float scoring function over typed variables. |
| **AtomicAction** | A STRIPS‑like operator: precondition + effect + reward. |
| **Symbolic mode** | Cache mode where the decide‑cache is a settable truth table = world state. |
| **`_0_*` mapping** | String‑keyed registry binding domain hooks to simulator callables. |
| **`StrictSymbolicCacheContainer`** | The concrete world‑state object (cache + variable registry). |
| **Weighter** | Stochastic, sequence‑aware oracle ordering candidate actions. |
| **Task** | A generated `{action sequence, initial state, goal state, objects, rewards}`. |

---

*Generated from a full read of the `rddl-agented` source tree
(`src/rddl/*.py`, `tests/test_world.py`, `tests/testing_utils.py`) and kept in sync with the
post‑fix source: the factual sections reflect the current code (after KI‑1 … KI‑9, KI‑14 …
KI‑16 and the repo‑wide docstring pass), while §13 retains every issue as a tracked record
(✅ resolved / ⬜ open). File:line references may drift by a few lines as docstrings were
added; the symbol names remain accurate.*
