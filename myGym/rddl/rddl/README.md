# Procedural Action Sequence Symbolic Generator for Robotic Manipulation Tasks (PRAG)

Generator of tasks in the form of action sequences for reinforcement learning. Based on a specification of the task environment it generates sequences of atomic actions, alongside of initial and goal state and objects to be used.
The specification consists of class definitions of actions, predicates (that are used to specify the state of the environment) and objects (entities in general).

## Installation  

`pip install .`


## Testing

`python tests/test_world.py` — a showcase/smoke test of the generation functionality (not an end-user tool).

## Generating tasks (CLI)

`generate.py` is the end-user generator. It exposes every generation option and every
`RDDLWorld` parameter as a CLI flag (with short forms), and can read an optional YAML config
whose values are overridden by explicit CLI options (defaults fill the rest).

```bash
# explicit options
python generate.py -l 2 -l 3 -l 4 -n 10 -s 42 -o tasks.yaml

# from a config file (see example_generation_config.yaml), overriding just the seed
python generate.py -c example_generation_config.yaml -s 123 -o tasks.yaml

# inspect what the domain provides, or validate a config without generating
python generate.py --inspect
python generate.py -c example_generation_config.yaml --dry-run

python generate.py --help        # full option list
```

Class names (actions, entities, predicates) are matched **case-insensitively**, so
`-i approach` resolves to the `Approach` class. Output detail is controlled by `--detail/-D`:
`concise` (default — action sequence + bound objects), `full` (adds the task's initial- and
final-state predicates), and `detailed` (adds the predicate state after each individual
action, like `test_world.yaml`).

Configuration precedence (low → high): built-in defaults → YAML config (`-c`) → CLI options.
A *domain* module (`-d`, default `testing_utils`) must register the simulator function
mapping and entity classes; it is imported before the rest of `rddl`. YAML keys are the long
option names with underscores (see `example_generation_config.yaml`).

## Input  

### Action specification


```Py3
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
```

Example:

```Py3
class Grasp(AtomicAction):
    _VARIABLES = {
        "gripper": Gripper,
        "object": GraspableObject
    }

    def __init__(self, **kwds) -> None:
        super().__init__(**kwds)
        gripper = self.get_argument("gripper")
        obj = self.get_argument("object")
        gripper_open = GripperOpen(gripper=gripper)
        self._initial = ParallelAndOp(
            left=GripperAt(gripper=gripper, object=obj),
            right=gripper_open
        )
        self._predicate = ParallelAndOp(
            left=IsHolding(gripper=gripper, object=obj),
            right=NotOp(operand=gripper_open)
        )
        self._reward = GraspReward()
```

### Predicates

```Py3
class GripperOpen(Predicate):
    _0_GRIPPER_OPEN_CHECK: ClassVar[Union[Callable, str]] = "gripper_open"  # name of function to call in the simulator
    _VARIABLES = {"gripper": Gripper}  # specify applicable variables

    def __init__(self, **kwds) -> None:
        super().__init__(**kwds)

    def __call__(self):
        return GripperOpen._0_GRIPPER_OPEN_CHECK(self.gripper)  # calling the function implemented by the simulator
```

### Objects

Objects are simply defined as Python classes derived from rddl.Entity or any of its subclasses. Multiple inheritance can be used to make some simulator object class into class usable by the generator.

The following class inherits from GraspableObject, which is a subclass of rddl.Entity. It also inherits from EnvObject, which is a gym simulator object.

```Py3
class Bowl(GraspableObject, EnvObject):

    def __init__(self, reference: Optional[str] = None):
        super().__init__(self._get_generic_reference() if reference is None else reference, "bowl")

    # some other functionality specific to the simulator
```

## Output

Example generator output. Action sequence shows on each line an action with it's (generic) parameters. Initial state shows predicates defining world-state at the start of the task. Final (goal) state shows predicates defining world-state at the end of the task. Objects show possible entity binding.

```Terminal
Action sequence:
        Approach(gripper: Gripper, object: GraspableObject)
        Grasp(gripper: Gripper, object: GraspableObject)
        Move(gripper: Gripper, object: GraspableObject, location: Location)
        Drop(gripper: Gripper, object: GraspableObject)
        Withdraw(gripper: Gripper, object: GraspableObject)
        Approach(gripper: Gripper, object: GraspableObject)
        Grasp(gripper: Gripper, object: GraspableObject)
        Follow(gripper: Gripper, object: GraspableObject, location: Location)
        Rotate(gripper: Gripper, object: GraspableObject, angle: AbstractRotation)
        Drop(gripper: Gripper, object: GraspableObject)
Initial state:
GripperOpen(entity_TiagoGripper_1) -> False
GripperAt(entity_TiagoGripper_1, entity_CerealBox_3) -> True
IsHolding(entity_TiagoGripper_1, entity_CerealBox_3) -> True
IsReachable(entity_TiagoGripper_1, entity_Bowl_8) -> True
ObjectAt(entity_CerealBox_3, entity_Bowl_8) -> True
Exists(entity_AbstractRotation_57) -> True
Final state:
GripperOpen(entity_TiagoGripper_1) -> True
GripperAt(entity_TiagoGripper_1, entity_CerealBox_3) -> True
IsHolding(entity_TiagoGripper_1, entity_CerealBox_3) -> False
IsReachable(entity_TiagoGripper_1, entity_Bowl_8) -> True
ObjectAt(entity_CerealBox_3, entity_Bowl_8) -> True
Exists(entity_AbstractRotation_57) -> True
ObjectAtPose(entity_CerealBox_3, entity_AbstractRotation_57) -> True
Object set 0:
         - TiagoGripper
         - CerealBox
         - Bowl
         - AbstractRotation
```
