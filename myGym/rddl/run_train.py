import argparse
import contextlib
import csv
import glob
import json
import os
import subprocess
import sys
import tempfile
import yaml

from myGym.rddl.rddl_to_mygym import RDDL_DIR, CONFIG_PATH, TASKS_PATH, build_config_from_task

PROJECT_ROOT = os.path.dirname(RDDL_DIR)
GENERATOR_SCRIPT = os.path.join(RDDL_DIR, "rddl", "generate.py")
TEST_SCRIPT = os.path.join(PROJECT_ROOT, "test.py")
TRAIN_SCRIPT = os.path.join(PROJECT_ROOT, "train.py")
ORACULUM_RESULTS_DIR = os.path.join(PROJECT_ROOT, "oraculum_results")
GENERATED_CONFIGS_DIR = os.path.join(RDDL_DIR, "generated_configs")


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