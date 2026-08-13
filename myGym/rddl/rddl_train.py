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
    # test.py/train.py currently only accept a config file path
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


def run_test(config_path: str, args: list = None, capture_output: bool = False,
             timeout: float = None) -> subprocess.CompletedProcess:
    """Run test.py against config_path.

    args: raw CLI args (test.py's own argparse interprets them).
    capture_output: capture stdout/stderr as text instead of printing to the terminal.
    timeout: forwarded to subprocess.run.
    """
    cmd = [sys.executable, TEST_SCRIPT, "-cfg", config_path] + (args or [])
    return subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=capture_output,
                          text=capture_output, timeout=timeout)


def run_train(config_path: str, args: list = None) -> int:
    """Run train.py against config_path."""
    cmd = [sys.executable, TRAIN_SCRIPT, "-cfg", config_path] + (args or [])
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode


def check_feasibility(config_path: str, args: list = None, trials: int = 3, min_success: int = 3, timeout: int = 60):
    """
    Check config feasibility with oraculum before training.

    Returns (feasible: bool, success_count: int, error: str or None).
    """
    os.makedirs(ORACULUM_RESULTS_DIR, exist_ok=True)
    for stale_file in glob.glob(os.path.join(ORACULUM_RESULTS_DIR, "results*.csv")):
        os.remove(stale_file)

    oraculum_args = (args or []) + [
        "-ct", "oraculum",
        "-ba", "absolute_gripper",
        "--eval_episodes", str(trials),
        "-rr", "True",  # Enable results report
    ]

    try:
        result = run_test(config_path, oraculum_args, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, 0, f"Feasibility check timed out after {timeout} seconds"

    success_count, _failed_subtasks, parse_error = _parse_latest_oraculum_results(ORACULUM_RESULTS_DIR)
    if parse_error:
        error_msg = result.stderr.strip() if result.stderr else parse_error
        return False, 0, error_msg

    if success_count < min_success:
        return False, success_count, f"Only {success_count}/{trials} oraculum trials succeeded (minimum required: {min_success})"

    return True, success_count, None


def _task_selected(task: dict, select: str) -> bool:
    """Whether `select` ('untested' | 'feasible' | 'both') selects this task.
    Tasks already marked infeasible are never selected.
    """
    feasible = task.get("feasible")
    if feasible is False:
        return False
    if select == "untested":
        return feasible is None
    if select == "feasible":
        return feasible is True
    return True  # select == "both"


@contextlib.contextmanager
def _task_config_path(task: dict):
    """Build task's config, write it to a temp file, and yield (config, config_path)."""
    config = build_config_from_task(task)
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield config, write_config(config, os.path.join(tmp_dir, "config.json"))


def _maybe_save_config(config: dict, length, idx: int, save_mode: str, feasible: bool) -> None:
    """Persist a copy of config to generated_configs/, per save_mode:

    'none' (default) -> never; 'all' -> always; 'feasible' -> only if feasible is True.
    """
    if save_mode == "none" or (save_mode == "feasible" and not feasible):
        return
    os.makedirs(GENERATED_CONFIGS_DIR, exist_ok=True)
    config_path = os.path.join(GENERATED_CONFIGS_DIR, f"{config['task_type']}_len{length}_idx{idx}.json")
    write_config(config, config_path)


def _should_generate(reuse: bool, tasks_yaml_path: str) -> bool:
    if reuse and os.path.exists(tasks_yaml_path):
        return False
    return True


def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--reuse-tasks", dest="select", default=None, choices=["untested", "feasible", "both"],
                        help="Reuse generated_tasks.yaml (skip regeneration unless missing); act on: "
                             "untested (new only), feasible (retrain), or both.")
    parser.add_argument("--save-configs", default="none", choices=["none", "all", "feasible"],
                        help=f"Persist configs to {GENERATED_CONFIGS_DIR}/: none (default), all, "
                             "or only those that pass the feasibility check.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run only feasibility checks without training.")
    # Unrecognized flags (e.g. -e, -ct) fall through to `remaining` and get forwarded as-is.
    return parser.parse_known_args()


def main():
    args, remaining = _parse_args()
    select = args.select or "untested"

    if _should_generate(args.select is not None, TASKS_PATH) and generate_tasks() != 0:
        print("Task generation failed.")
        sys.exit(1)

    for length, idx, task in iter_tasks(TASKS_PATH, status="all"):
        if not _task_selected(task, select):
            continue

        with _task_config_path(task) as (config, config_path):
            if task.get("feasible") is True:
                feasible, status = True, "already marked feasible"
            else:
                feasible, n, err = check_feasibility(config_path, remaining)
                mark_task(TASKS_PATH, length, idx, feasible)
                status = f"feasibility check passed ({n} trials)" if feasible else f"feasibility check failed: {err}"

            _maybe_save_config(config, length, idx, args.save_configs, feasible)

            if not feasible:
                print(f"Task [{length}][{idx}] {status}.")
                continue

            if args.dry_run:
                print(f"Task [{length}][{idx}] {status}, dry run: skipping training.")
                continue

            print(f"Task [{length}][{idx}] {status}, starting training.")
            run_train(config_path, remaining)

    sys.exit(0)


if __name__ == "__main__":
    main()