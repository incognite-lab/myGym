import csv
import glob
import json
import os
import re
import subprocess
import sys

import yaml

RDDL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(RDDL_DIR)
CONFIG_BODY_PATH = os.path.join(RDDL_DIR, "config_body.json")
EXAMPLE_TASKS_PATH = os.path.join(RDDL_DIR, "example_out.yaml")
TEST_SCRIPT = os.path.join(PROJECT_ROOT, "test.py")
ORACULUM_RESULTS_DIR = os.path.join(PROJECT_ROOT, "oraculum_results")

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

# myGym predicate names differ slightly from the rddl domain's predicate names
PREDICATE_RENAME = {
    "IsReachable": "Reachable",
}

_PREDICATE_RE = re.compile(r"^(\w+)\(([^)]*)\)\s*->\s*(True|False)$")


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
    """Convert a single 'Name(args) -> True/False' rddl predicate string to myGym format, or None if it's False."""
    match = _PREDICATE_RE.match(pred_str.strip())
    if not match:
        raise ValueError(f"Unrecognized predicate format: {pred_str}")
    name, raw_args, value = match.groups()
    if value == "False":
        return None
    name = PREDICATE_RENAME.get(name, name)
    args = [a.strip() for a in raw_args.split(",") if a.strip()]
    mapped_args = [name_map[a] for a in args if a in name_map]
    return f"{name}({','.join(mapped_args)})"


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


def build_config(config_body_path: str = CONFIG_BODY_PATH, tasks_yaml_path: str = EXAMPLE_TASKS_PATH) -> dict:
    """
    Build a myGym training config dict from config_body.json, filled in with the
    first task found in tasks_yaml_path (a generate.py output file), producing a
    config in the same "*_predicates.json" structure as configs/AG_predicates.json.
    """
    with open(config_body_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    task = _load_first_task(tasks_yaml_path)
    type_map = _entity_name_map(task["all_objects"])
    object_types = list(dict.fromkeys(type_map.values()))

    # task_objects requires both "init" and "goal" to be real, non-null objects. When the rddl
    # task only involves one object type, reuse it for both slots (like configs/AGM_predicates.json
    # does with apple1/apple2) instead of leaving goal as a "null" placeholder.
    if len(object_types) > 1:
        init_urdf, goal_urdf = object_types[0], object_types[1]
        init_obj_name, goal_obj_name = init_urdf, goal_urdf
    else:
        init_urdf = goal_urdf = object_types[0]
        init_obj_name, goal_obj_name = f"{init_urdf}1", f"{goal_urdf}2"

    # Predicates only ever reference the manipulated (init) object, never the goal placeholder
    name_map = {entity: init_obj_name for entity, obj_type in type_map.items() if obj_type == init_urdf}

    task_type = "".join(ACTION_TO_CODE[action.lower()] for action in task["action_list"])

    # Reachable(...) is a precondition established in "init" and is not restated in later subgoals/goal.
    # GripperOpen(...) is excluded from "init" too: the gripper always starts closed, so its rddl-reported
    # initial state doesn't matter.
    predicates = {"init": _convert_predicate_list(task["initial_state"], name_map, exclude={"GripperOpen"})}
    sequence = task["sequence"]
    for i, step in enumerate(sequence, start=1):
        key = "goal" if i == len(sequence) else f"subgoal{i}"
        predicates[key] = _convert_predicate_list(step["predicates"], name_map, exclude={"Reachable"})

    init_obj = {"obj_name": init_obj_name, "fixed": 0, "rand_rot": 0}
    goal_obj = {"obj_name": goal_obj_name, "fixed": 1, "rand_rot": 0}
    if init_obj_name != init_urdf:
        init_obj["urdf_name"] = init_urdf
    if goal_obj_name != goal_urdf:
        goal_obj["urdf_name"] = goal_urdf

    config["task_type"] = task_type
    config["task_objects"] = [{"init": init_obj, "goal": goal_obj}]
    config["predicates"] = [predicates]
    config["color_dict"] = {init_obj_name: ["green"], "target": ["gray"]}
    config["used_objects"] = {"num_range": [0, 0], "obj_list": []}

    return config


# Maps keyword argument names to their CLI flags in train.py / test.py (shared)
_ARG_FLAGS = {
    "config": "-cfg",
    "env_name": "-n",
    "workspace": "-ws",
    "engine": "-p",
    "seed": "-sd",
    "render": "-d",
    "camera": "-c",
    "visualize": "-vi",
    "visgym": "-vg",
    "gui": "-g",
    "robot": "-b",
    "robot_init": "-bi",
    "robot_action": "-ba",
    "max_velocity": "-mv",
    "max_force": "-mf",
    "action_repeat": "-ar",
    "task_type": "-tt",
    "task_objects": "-to",
    "used_objects": "-u",
    "distractors": "-di",
    "distractor_moveable": "-dm",
    "distractor_constant_speed": "-ds",
    "distractor_movement_dimensions": "-dd",
    "distractor_movement_endpoints": "-de",
    "observed_links_num": "-no",
    "reward": "-re",
    "distance_type": "-dt",
    "train_framework": "-w",
    "algo": "-a",
    "steps": "-s",
    "max_episode_steps": "-ms",
    "algo_steps": "-ma",
    "eval_freq": "-ef",
    "eval_episodes": "-e",
    "logdir": "-l",
    "record": "-r",
    "multiprocessing": "-i",
    "vectorized_envs": "-v",
    "model_path": "-m",
    "vae_path": "-vp",
    "yolact_path": "-yp",
    "yolact_config": "-yc",
    "pretrained_model": "-ptm",
    "natural_language": "-nl",
}


# test.py-only flags (in addition to all _ARG_FLAGS)
_TEST_ARG_FLAGS = {
    "control": "-ct",
    "network_switcher": "-ns",
    "results_report": "-rr",
    "top_grasp": "-tp",
}

# store_true flags: value=True appends the flag, value=False omits it
_TEST_STORE_TRUE_FLAGS = {
    "vsampling": "-vs",
    "vtrajectory": "-vt",
    "vinfo": "-vn",
}

GENERATED_CONFIG_PATH = os.path.join(RDDL_DIR, "generated_config.json")


def write_config(config: dict, output_path: str) -> str:
    # TODO: train.py only accepts a config file path (it does open(args.config) internally),
    # so this write-to-disk is a temporary handoff. Revisit to pass the config in-memory instead.
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    return output_path


def main():
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-o", "--output", default=GENERATED_CONFIG_PATH,
                         help="Where to write the config generated from config_body.json + example_out.yaml")
    args, remaining = parser.parse_known_args()
    sys.exit(run_generated_config(args.output, _extra=remaining))


def _parse_latest_oraculum_results(results_dir: str):
    """Read the most recent oraculum results*.csv, returning (success_count, failed_subtask_sequences, error)."""
    result_files = glob.glob(os.path.join(results_dir, "results*.csv"))
    if not result_files:
        return 0, [], "No results CSV file was created"

    latest_file = max(result_files, key=os.path.getctime)
    success_count = 0
    failed_subtask_sequences = []
    try:
        with open(latest_file, "r") as f:
            reader = csv.DictReader(f)
            for run_idx, row in enumerate(reader, 1):
                if row["Success"].strip() == "True":
                    success_count += 1
                else:
                    failed_subtask_sequences.append((run_idx, row["Subtasks"].strip()))
    except Exception as e:
        return 0, [], f"Failed to parse results file: {e}"

    return success_count, failed_subtask_sequences, None


def check_feasibility(config_path: str, trials: int = 3, min_success: int = 3, timeout: int = 60):
    """
    Probe whether config_path is solvable at all, by running test.py's oraculum (oracle) controller
    against it before spending time on real training. Same technique as
    unittest/test_oraculum_configs.py, with a lighter trial count/timeout since this only gates a
    single generated config rather than sweeping the whole configs folder.

    Returns (feasible: bool, success_count: int, error: str or None).
    """
    os.makedirs(ORACULUM_RESULTS_DIR, exist_ok=True)
    for stale_file in glob.glob(os.path.join(ORACULUM_RESULTS_DIR, "results*.csv")):
        os.remove(stale_file)

    cmd = [
        sys.executable, TEST_SCRIPT,
        "--config", config_path,
        "-ct", "oraculum",
        "-ba", "absolute_gripper",
        "-g", "0",  # No GUI
        "--eval_episodes", str(trials),
        "-rr", "True",  # Enable results report
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=PROJECT_ROOT)
    except subprocess.TimeoutExpired:
        return False, 0, f"Feasibility check timed out after {timeout} seconds"

    success_count, _failed_subtasks, parse_error = _parse_latest_oraculum_results(ORACULUM_RESULTS_DIR)
    if parse_error:
        error_msg = result.stderr.strip() if result.stderr else parse_error
        return False, 0, error_msg

    if success_count < min_success:
        return False, success_count, f"Only {success_count}/{trials} oraculum trials succeeded (minimum required: {min_success})"

    return True, success_count, None


def run_generated_config(output_path: str = GENERATED_CONFIG_PATH, _extra: list = None, **kwargs) -> int:
    config = build_config()
    write_config(config, output_path)

    feasible, success_count, error = check_feasibility(output_path)
    if not feasible:
        print(f"Feasibility check failed ({success_count} successful oraculum trials): {error}")
        return 1

    print(f"Feasibility check passed ({success_count} successful oraculum trials), starting training.")
    return run_config(output_path, _extra=_extra, **kwargs)


def run_test(config_path: str, _extra: list = None, **kwargs) -> int:
    test_script = os.path.join(PROJECT_ROOT, "test.py")
    cmd = [sys.executable, test_script, "-cfg", config_path]
    all_flags = {**_ARG_FLAGS, **_TEST_ARG_FLAGS}
    for key, value in kwargs.items():
        if key in _TEST_STORE_TRUE_FLAGS:
            if value:
                cmd.append(_TEST_STORE_TRUE_FLAGS[key])
        else:
            flag = all_flags[key]
            if isinstance(value, list):
                cmd += [flag] + [str(v) for v in value]
            else:
                cmd += [flag, str(value)]
    if _extra:
        cmd += _extra
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode


def run_config(config_path: str, _extra: list = None, **kwargs) -> int:
    train_script = os.path.join(os.path.dirname(__file__), "..", "train.py")
    cmd = [sys.executable, train_script, "-cfg", config_path]
    for key, value in kwargs.items():
        flag = _ARG_FLAGS[key]
        if isinstance(value, list):
            cmd += [flag] + [str(v) for v in value]
        else:
            cmd += [flag, str(value)]
    if _extra:
        cmd += _extra
    result = subprocess.run(cmd, cwd=os.path.dirname(train_script))
    return result.returncode


if __name__ == "__main__":
    main()
