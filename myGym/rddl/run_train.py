import csv
import glob
import json
import os
import re
import subprocess
import sys
import tempfile

import yaml

RDDL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(RDDL_DIR)
CONFIG_PATH = os.path.join(RDDL_DIR, "rddl_config.json")
TASKS_PATH = os.path.join(RDDL_DIR, "generated_tasks.yaml")
GENERATOR_SCRIPT = os.path.join(RDDL_DIR, "rddl", "generate.py")
TEST_SCRIPT = os.path.join(PROJECT_ROOT, "test.py")
ORACULUM_RESULTS_DIR = os.path.join(PROJECT_ROOT, "oraculum_results")
GENERATED_CONFIGS_DIR = os.path.join(RDDL_DIR, "generated_configs")

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


def generate_tasks(config_path: str = CONFIG_PATH, output: str = TASKS_PATH) -> int:
    """Run the RDDL generator using params from config_body, writing results to output."""
    with open(config_path) as f:
        cfg = json.load(f)

    def flags(flag, values): return sum([[flag, str(v)] for v in values], [])
    def bool_flag(true_flag, false_flag, value): return [true_flag] if value else [false_flag]

    cmd = (
        [sys.executable, GENERATOR_SCRIPT]
        + flags("-l", cfg.get("sequence_lengths", [4]))
        + ["-n", str(cfg.get("n_repeats", 1))]
        + ["-m", cfg.get("method", "one-shot")]
        + flags("-a", cfg.get("allowed_actions", []))
        + flags("-e", cfg.get("allowed_entities", []))
        + flags("-W", cfg.get("action_weights") or [])
        + flags("-O", [f"{n}={w}" for n, w in cfg.get("object_weights", {}).items()])
        + flags("-w", cfg.get("weight_mode", []))
        + bool_flag("--single-object", "--multi-object", cfg.get("sample_single_object_per_class", False))
        + bool_flag("--robots", "--no-robots", cfg.get("add_robots", True))
        + bool_flag("--retry", "--no-retry", cfg.get("retry_ad_infinitum", True))
        + ["-i", "Approach", "-D", "detailed", "--output", output]
    )
    result = subprocess.run(cmd, cwd=os.path.dirname(GENERATOR_SCRIPT), capture_output=True, text=True)
    if result.returncode != 0:
        tail = "\n".join((result.stdout + result.stderr).splitlines()[-50:])
        print(f"Generator failed:\n{tail}")
    return result.returncode


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


# Keys in config_body that are only for the RDDL generator and must not be passed to train.py
_GENERATION_ONLY_KEYS = {
    "sequence_lengths", "n_repeats", "allowed_actions", "allowed_entities", "allowed_predicates",
    "method", "sample_single_object_per_class", "action_weights", "object_weights",
    "weight_mode", "add_robots", "retry_ad_infinitum",
}


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

    # Two distinct objects only when Move is involved (needs a target) and RDDL provided two types.
    # All other tasks (AG, AW, AGW, …) manipulate a single object, so goal is a duplicate placeholder.
    if "M" in task_type and len(object_types) > 1:
        init_urdf, goal_urdf = object_types[0], object_types[1]
        init_obj_name, goal_obj_name = init_urdf, goal_urdf
    else:
        init_urdf = goal_urdf = object_types[0]
        init_obj_name, goal_obj_name = f"{init_urdf}1", f"{init_urdf}2"

    # Init-type entities → named init object; all others (table etc.) keep their type name for predicate args
    name_map = {
        entity: (init_obj_name if obj_type == init_urdf else obj_type)
        for entity, obj_type in type_map.items()
    }

    # Reachable(...) is excluded from "init": it is not restated in later subgoals/goal.
    # GripperOpen(...) is excluded: the gripper always starts open, rddl-reported state is irrelevant.
    # OnTop(...) is excluded: RDDL outputs it but the table arg would be mangled; we inject it correctly below.
    predicates = {"init": _convert_predicate_list(task["initial_state"], name_map, exclude={"GripperOpen", "OnTop"})}

    # Inject OnTop(obj, table) for every reachable init object
    for pred in list(predicates["init"]):
        if pred.startswith("Reachable("):
            obj = pred[len("Reachable("):pred.index(")")]
            on_top = f"OnTop({obj},table): True"
            if on_top not in predicates["init"]:
                predicates["init"].append(on_top)

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


def build_config(config_path: str = CONFIG_PATH, tasks_yaml_path: str = TASKS_PATH) -> dict:
    """Build a myGym config from the first task in tasks_yaml_path. Convenience wrapper around build_config_from_task."""
    return build_config_from_task(_load_first_task(tasks_yaml_path), config_path)


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




def mark_task(tasks_yaml_path: str, length, idx: int, feasible: bool) -> None:
    """Set the feasible flag on a single task in the YAML (true/false; absent = untested)."""
    with open(tasks_yaml_path) as f:
        data = yaml.safe_load(f)
    data["tasks"][length][idx]["feasible"] = feasible
    with open(tasks_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)


def iter_tasks(tasks_yaml_path: str, status: str = "all"):
    """Yield (length, idx, task) for tasks in the YAML.

    status: 'all' | 'untested' (feasible field absent) | 'feasible' (feasible == True)
    """
    with open(tasks_yaml_path) as f:
        data = yaml.safe_load(f)
    for length, task_list in data["tasks"].items():
        for idx, task in enumerate(task_list):
            f = task.get("feasible")
            if status == "untested" and f is not None:
                continue
            if status == "feasible" and f is not True:
                continue
            yield length, idx, task


def write_config(config: dict, output_path: str) -> str:
    # TODO: train.py only accepts a config file path (it does open(args.config) internally),
    # so this write-to-disk is a temporary handoff. Revisit to pass the config in-memory instead.
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    return output_path


def main():
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--no-generate", action="store_true",
                        help="Skip task generation and use existing generated_tasks.yaml")
    parser.add_argument("--mode", default="untested", choices=["untested", "feasible", "both"],
                        help="untested: only new tasks; feasible: retrain passed tasks; both: all except failed")
    parser.add_argument("--save-configs", action="store_true",
                        help=f"Save each generated config to {GENERATED_CONFIGS_DIR}/")
    parser.add_argument("-g", "--gui", type=int, default=0,
                        help="Enable GUI for both feasibility check and training (0/1)")
    args, remaining = parser.parse_known_args()

    if not args.no_generate:
        if generate_tasks() != 0:
            print("Task generation failed.")
            sys.exit(1)

    sys.exit(run_generated_config(mode=args.mode, save_configs=args.save_configs,
                                  gui=bool(args.gui), _extra=remaining))


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


def check_feasibility(config_path: str, trials: int = 3, min_success: int = 3, timeout: int = 60, gui: bool = False):
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
        "-g", "1" if gui else "0",
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


def run_generated_config(mode: str = "untested", save_configs: bool = False,
                         gui: bool = False, _extra: list = None, **kwargs) -> int:
    """Run the full pipeline over tasks in generated_tasks.yaml.

    mode: 'untested'  — test and train only tasks not yet checked
          'feasible'  — retrain only tasks already marked feasible (skip feasibility check)
          'both'      — untested + feasible
    save_configs: write each task's config to generated_configs/ with a descriptive filename
    """
    if save_configs:
        os.makedirs(GENERATED_CONFIGS_DIR, exist_ok=True)

    for length, idx, task in iter_tasks(TASKS_PATH, status="all"):
        already_feasible = task.get("feasible") is True
        already_failed = task.get("feasible") is False
        is_untested = task.get("feasible") is None

        if already_failed:
            continue
        if mode == "untested" and not is_untested:
            continue
        if mode == "feasible" and not already_feasible:
            continue

        config = build_config_from_task(task)
        if save_configs:
            config_path = os.path.join(GENERATED_CONFIGS_DIR, f"{config['task_type']}_len{length}_idx{idx}.json")
            write_config(config, config_path)
        else:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
                json.dump(config, tmp, indent=4)
                config_path = tmp.name

        try:
            if already_feasible:
                print(f"Task [{length}][{idx}] already marked feasible, starting training.")
                #run_config(config_path, _extra=_extra, gui=int(gui), **kwargs)
            else:
                feasible, n, err = check_feasibility(config_path, gui=gui)
                mark_task(TASKS_PATH, length, idx, feasible)
                if feasible:
                    print(f"Task [{length}][{idx}] feasibility check passed ({n} trials), starting training.")
                    #run_config(config_path, _extra=_extra, gui=int(gui), **kwargs)
                else:
                    print(f"Task [{length}][{idx}] feasibility check failed: {err}")
        finally:
            if not save_configs:
                os.unlink(config_path)

    return 0


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