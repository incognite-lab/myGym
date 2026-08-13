"""
Translation layer between rddl task definitions (generated_tasks.yaml) and myGym
training configs (the "*_predicates.json" structure used by configs/AG_predicates.json).
"""

import json
import os
import re

RDDL_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(RDDL_DIR, "rddl_config.json")
TASKS_PATH = os.path.join(RDDL_DIR, "generated_tasks.yaml")

# Maps rddl action names to the single-letter codes used in myGym task_type strings
ACTION_TO_CODE = {
    "approach": "A",
    "withdraw": "W",
    "grasp": "G",
    "drop": "D",
    "move": "M",
    "rotate": "R",
    "transform": "T",
    "follow": "F",
}

_PREDICATE_RE = re.compile(r"^(\w+)\(([^)]*)\)\s*->\s*(True|False)$")

# Generator keys not passed to train.py
_GENERATION_ONLY_KEYS = {
    "sequence_lengths", "n_repeats", "allowed_actions", "allowed_entities", "allowed_predicates",
    "method", "sample_single_object_per_class", "action_weights", "object_weights",
    "weight_mode", "add_robots", "retry_ad_infinitum",
}


def _convert_predicate(pred_str: str, name_map: dict) -> str:
    """Convert a single 'Name(args) -> True/False' rddl predicate string to myGym format.

    Args missing from name_map are gripper entities (excluded by _build_name_map) and are dropped.
    """
    match = _PREDICATE_RE.match(pred_str.strip())
    if not match:
        raise ValueError(f"Unrecognized predicate format: {pred_str}")
    name, raw_args, value = match.groups()
    args = [a.strip() for a in raw_args.split(",") if a.strip()]
    mapped_args = [name_map[a] for a in args if a in name_map]  # drop gripper args
    suffix = ": True" if value == "True" else ": False"
    return f"{name}({','.join(mapped_args)}){suffix}"


def _convert_predicate_list(predicates: list, name_map: dict, exclude: set = frozenset()) -> list:
    """Convert a list of rddl predicate strings to myGym format, dropping excluded names."""
    converted = []
    for pred in predicates:
        result = _convert_predicate(pred, name_map)
        predicate_name = result.split("(", 1)[0]
        if predicate_name in exclude:
            continue
        converted.append(result)
    return converted


def _pick_init_goal_objects(objects: list) -> tuple:
    """
    Pick the real init/goal objects — the first and second entity, in order (regardless of
    whether they share a type). goal_urdf/goal_obj_name are None when there's only one.

    objects: [(obj_name, urdf), ...] as returned by _build_name_map, one entry per real entity.

    Returns (init_urdf, goal_urdf, init_obj_name, goal_obj_name).
    """
    init_obj_name, init_urdf = objects[0]
    if len(objects) > 1:
        goal_obj_name, goal_urdf = objects[1]
        return init_urdf, goal_urdf, init_obj_name, goal_obj_name
    return init_urdf, None, init_obj_name, None


def _pad_missing_goal_object(init_urdf: str, goal_urdf, init_obj_name: str, goal_obj_name) -> tuple:
    """
    TEMPORARY WORKAROUND for a myGym bug: task_objects currently always requires both an init
    and a goal object, even for tasks that genuinely have only one (e.g. Approach). Until myGym
    supports single-object tasks, fabricate a placeholder goal object by duplicating init_urdf
    (e.g. kostka1 + kostka2). Remove this once that's fixed.
    """
    if goal_urdf is not None:
        return init_urdf, goal_urdf, init_obj_name, goal_obj_name
    return init_urdf, init_urdf, init_obj_name, f"{init_urdf}2"


def _build_name_map(all_objects: dict) -> tuple:
    """
    Map each non-gripper entity to a numbered myGym name (kostka1, kostka2, …).

    Returns (name_map, objects); objects is [(obj_name, urdf), ...], one entry per real
    entity in order, used by _pick_init_goal_objects.
    """
    type_counters: dict = {}
    name_map = {}
    objects = []
    for entity, obj_type in all_objects.items():
        if "gripper" in obj_type.lower():
            continue
        obj_type = obj_type.lower()
        type_counters[obj_type] = type_counters.get(obj_type, 0) + 1
        obj_name = f"{obj_type}{type_counters[obj_type]}"
        name_map[entity] = obj_name
        objects.append((obj_name, obj_type))
    return name_map, objects


def _inject_on_top_predicates(init_predicates: list) -> list:
    """
    Add OnTop(obj, table) for every reachable init object (mutates init_predicates in place).

    RDDL doesn't emit OnTop; it's derived here since a reachable object is assumed to start on the table.
    """
    for pred in list(init_predicates):
        if pred.startswith("IsReachable("):
            obj = pred[len("IsReachable("):pred.index(")")]
            on_top = f"OnTop({obj},table): True"
            if on_top not in init_predicates:
                init_predicates.append(on_top)
    return init_predicates


def _build_subgoal_predicates(sequence: list, name_map: dict) -> dict:
    """
    Convert the RDDL plan's step postconditions into myGym's subgoal/goal predicates.
    """
    predicates = {}
    for i, step in enumerate(sequence, start=1):
        key = "goal" if i == len(sequence) else f"subgoal{i}"
        predicates[key] = _convert_predicate_list(step["predicates"], name_map, exclude=set())
    return predicates


def _assemble_config(config_path: str, task_type: str, init_obj: dict, goal_obj: dict, predicates: dict) -> dict:
    """Load the base config, strip generator-only keys, and stamp in task's computed fields."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    for key in _GENERATION_ONLY_KEYS:
        config.pop(key, None)

    config["task_type"] = task_type
    config["task_objects"] = [{"init": init_obj, "goal": goal_obj}]
    config["predicates"] = [predicates]
    config["color_dict"] = {init_obj["obj_name"]: ["green"], "target": ["gray"]}
    config["used_objects"] = {"num_range": [0, 0], "obj_list": []}

    return config


def build_config_from_task(task: dict, config_path: str = CONFIG_PATH) -> dict:
    """Build a myGym training config dict from a single task dict (as yielded by iter_tasks)."""
    task_type = "".join(ACTION_TO_CODE[action.lower()] for action in task["action_list"])

    name_map, objects = _build_name_map(task["all_objects"])
    init_urdf, goal_urdf, init_obj_name, goal_obj_name = _pad_missing_goal_object(
        *_pick_init_goal_objects(objects)
    )

    # OnTop is excluded here; _inject_on_top_predicates derives it.
    predicates = {"init": _inject_on_top_predicates(
        _convert_predicate_list(task["initial_state"], name_map, exclude={"OnTop"})
    )}
    predicates.update(_build_subgoal_predicates(task["sequence"], name_map))

    init_obj = {"obj_name": init_obj_name, "fixed": 0, "rand_rot": 0, "urdf_name": init_urdf}
    goal_obj = {"obj_name": goal_obj_name, "fixed": 1, "rand_rot": 0, "urdf_name": goal_urdf}

    return _assemble_config(config_path, task_type, init_obj, goal_obj, predicates)
