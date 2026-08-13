import argparse
import contextlib
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
TRAIN_SCRIPT = os.path.join(PROJECT_ROOT, "train.py")
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


def run_test(config_path: str, extra: list = None, capture_output: bool = False,
             timeout: float = None) -> subprocess.CompletedProcess:
    """Run test.py against config_path. extra: raw CLI args forwarded as-is (e.g. ["-g", "1"]) —
    test.py's own argparse (myGym.train.get_parser plus its test-only flags) interprets them.

    capture_output: capture stdout/stderr as text (result.stdout/.stderr) instead of letting the
    subprocess print straight to the terminal — needed by callers that inspect the output.
    timeout: forwarded to subprocess.run; raises subprocess.TimeoutExpired if exceeded.
    """
    cmd = [sys.executable, TEST_SCRIPT, "-cfg", config_path] + (extra or [])
    return subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=capture_output,
                          text=capture_output, timeout=timeout)


def run_train(config_path: str, extra: list = None) -> int:
    """Run train.py against config_path. extra: raw CLI args forwarded as-is (e.g. ["-g", "1"]) —
    train.py's own argparse (myGym.train.get_parser) interprets them."""
    cmd = [sys.executable, TRAIN_SCRIPT, "-cfg", config_path] + (extra or [])
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode


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

    extra = [
        "-ct", "oraculum",
        "-ba", "absolute_gripper",
        "-g", "1" if gui else "0",
        "--eval_episodes", str(trials),
        "-rr", "True",  # Enable results report
    ]

    try:
        result = run_test(config_path, extra, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, 0, f"Feasibility check timed out after {timeout} seconds"

    success_count, _failed_subtasks, parse_error = _parse_latest_oraculum_results(ORACULUM_RESULTS_DIR)
    if parse_error:
        error_msg = result.stderr.strip() if result.stderr else parse_error
        return False, 0, error_msg

    if success_count < min_success:
        return False, success_count, f"Only {success_count}/{trials} oraculum trials succeeded (minimum required: {min_success})"

    return True, success_count, None


def _task_selected(task: dict, mode: str) -> bool:
    """Whether `mode` selects this task for testing/training.

    mode: 'untested' — only tasks not yet checked
          'feasible' — only tasks already marked feasible (retrain, skip feasibility check)
          'both'     — untested + feasible
    Tasks already marked infeasible are never selected.
    """
    feasible = task.get("feasible")
    if feasible is False:
        return False
    if mode == "untested":
        return feasible is None
    if mode == "feasible":
        return feasible is True
    return True  # mode == "both"


@contextlib.contextmanager
def _task_config_path(task: dict, length, idx: int, save_configs: bool):
    """Build task's myGym config, write it to disk, and yield its path.

    save_configs: write to generated_configs/ with a descriptive filename and keep it;
    otherwise write to a temp file that's deleted on exit.
    """
    config = build_config_from_task(task)
    if save_configs:
        os.makedirs(GENERATED_CONFIGS_DIR, exist_ok=True)
        config_path = os.path.join(GENERATED_CONFIGS_DIR, f"{config['task_type']}_len{length}_idx{idx}.json")
        write_config(config, config_path)
    else:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(config, tmp, indent=4)
            config_path = tmp.name

    try:
        yield config_path
    finally:
        if not save_configs:
            os.unlink(config_path)


def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--no-generate", action="store_true",
                        help="Skip task generation and use existing generated_tasks.yaml")
    parser.add_argument("--mode", default="untested", choices=["untested", "feasible", "both"],
                        help="untested: only new tasks; feasible: retrain passed tasks; both: all except failed")
    parser.add_argument("--save-configs", action="store_true",
                        help=f"Save each generated config to {GENERATED_CONFIGS_DIR}/")
    parser.add_argument("-g", "--gui", type=int, default=0,
                        help="Enable GUI for both feasibility check and training (0/1)")
    return parser.parse_known_args()


def main():
    args, remaining = _parse_args()
    gui = bool(args.gui)

    if not args.no_generate and generate_tasks() != 0:
        print("Task generation failed.")
        sys.exit(1)

    extra = ["-g", "1" if gui else "0"] + remaining

    for length, idx, task in iter_tasks(TASKS_PATH, status="all"):
        if not _task_selected(task, args.mode):
            continue

        with _task_config_path(task, length, idx, args.save_configs) as config_path:
            if task.get("feasible") is True:
                print(f"Task [{length}][{idx}] already marked feasible, starting training.")
                run_train(config_path, extra)
                continue

            feasible, n, err = check_feasibility(config_path, gui=gui)
            mark_task(TASKS_PATH, length, idx, feasible)
            if feasible:
                print(f"Task [{length}][{idx}] feasibility check passed ({n} trials), starting training.")
                run_train(config_path, extra)
            else:
                print(f"Task [{length}][{idx}] feasibility check failed: {err}")

    sys.exit(0)


if __name__ == "__main__":
    main()