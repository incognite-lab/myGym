"""
Translation layer between rddl task definitions (generated_tasks.yaml) and myGym
training configs (the "*_predicates.json" structure used by configs/AG_predicates.json).
"""

import json
import os
import re

import yaml

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

# Keys in config_body that are only for the RDDL generator and must not be passed to train.py
_GENERATION_ONLY_KEYS = {
    "sequence_lengths", "n_repeats", "allowed_actions", "allowed_entities", "allowed_predicates",
    "method", "sample_single_object_per_class", "action_weights", "object_weights",
    "weight_mode", "add_robots", "retry_ad_infinitum",
}


def _load_first_task(tasks_yaml_path: str) -> dict:
    with open(tasks_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    first_length = next(iter(data["tasks"]))
    return data["tasks"][first_length][0]


def _entity_name_map(all_objects: dict) -> dict:
    """Map rddl entity variables (e.g. entity_Apple_3) to myGym object names, skipping the gripper entities."""
    return {
        entity: obj_type.lower()
        for entity, obj_type in all_objects.items()
        if "gripper" not in obj_type.lower()
    }


def _convert_predicate(pred_str: str, name_map: dict):
    """Convert a single 'Name(args) -> True/False' rddl predicate string to myGym format."""
    match = _PREDICATE_RE.match(pred_str.strip())
    if not match:
        raise ValueError(f"Unrecognized predicate format: {pred_str}")
    name, raw_args, value = match.groups()
    args = [a.strip() for a in raw_args.split(",") if a.strip()]
    mapped_args = [name_map[a] for a in args if a in name_map]
    suffix = ": True" if value == "True" else ": False"
    return f"{name}({','.join(mapped_args)}){suffix}"


def _convert_predicate_list(predicates: list, name_map: dict, exclude: set = frozenset()) -> list:
    converted = []
    for pred in predicates:
        result = _convert_predicate(pred, name_map)
        if result is None:
            continue
        predicate_name = result.split("(", 1)[0]
        if predicate_name in exclude or result in converted:
            continue
        converted.append(result)
    return converted


def build_config_from_task(task: dict, config_path: str = CONFIG_PATH) -> dict:
    """
    Build a myGym training config dict from a single task dict (as yielded by iter_tasks),
    producing a config in the same "*_predicates.json" structure as configs/AG_predicates.json.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    for key in _GENERATION_ONLY_KEYS:
        config.pop(key, None)

    task_type = "".join(ACTION_TO_CODE[action.lower()] for action in task["action_list"])

    type_map = _entity_name_map(task["all_objects"])
    # Exclude location types from init/goal assignment — only graspable objects go here
    object_types = list(dict.fromkeys(v for v in type_map.values() if v != "table"))

    # Object names always include a number; urdf_name is the bare type without the number.
    # Move with two distinct types: apple1 + banana1; all other tasks: kostka1 + kostka2 (placeholder).
    if "M" in task_type and len(object_types) > 1:
        init_urdf, goal_urdf = object_types[0], object_types[1]
        init_obj_name, goal_obj_name = f"{init_urdf}1", f"{goal_urdf}1"
    else:
        init_urdf = goal_urdf = object_types[0]
        init_obj_name, goal_obj_name = f"{init_urdf}1", f"{init_urdf}2"

    # Number entities sequentially per type (kostka1, kostka2, …); table stays unnumbered.
    type_counters: dict = {}
    name_map = {}
    for entity, obj_type in type_map.items():
        if obj_type == "table":
            name_map[entity] = "table"
        else:
            type_counters[obj_type] = type_counters.get(obj_type, 0) + 1
            name_map[entity] = f"{obj_type}{type_counters[obj_type]}"

    # OnTop(...) is excluded: RDDL doesn't output it; we inject it manually below based on Reachable.
    predicates = {"init": _convert_predicate_list(task["initial_state"], name_map, exclude={"OnTop"})}

    # Inject OnTop(obj, table) for every reachable init object
    for pred in list(predicates["init"]):
        if pred.startswith("IsReachable("):
            obj = pred[len("IsReachable("):pred.index(")")]
            on_top = f"OnTop({obj},table): True"
            if on_top not in predicates["init"]:
                predicates["init"].append(on_top)

    sequence = task["sequence"]
    for i, step in enumerate(sequence, start=1):
        key = "goal" if i == len(sequence) else f"subgoal{i}"
        predicates[key] = _convert_predicate_list(step["predicates"], name_map, exclude=set())

    init_obj = {"obj_name": init_obj_name, "fixed": 0, "rand_rot": 0, "urdf_name": init_urdf}
    goal_obj = {"obj_name": goal_obj_name, "fixed": 1, "rand_rot": 0, "urdf_name": goal_urdf}

    config["task_type"] = task_type
    config["task_objects"] = [{"init": init_obj, "goal": goal_obj}]
    config["predicates"] = [predicates]
    config["color_dict"] = {init_obj_name: ["green"], "target": ["gray"]}
    config["used_objects"] = {"num_range": [0, 0], "obj_list": []}

    return config


def build_config(config_path: str = CONFIG_PATH, tasks_yaml_path: str = TASKS_PATH) -> dict:
    """Build a myGym config from the first task in tasks_yaml_path. Convenience wrapper around build_config_from_task."""
    return build_config_from_task(_load_first_task(tasks_yaml_path), config_path)
