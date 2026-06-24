#!/usr/bin/env python3
"""
Unit test that runs test.py with all configs in ./configs folder with -ct oraculum.
Tests all configs with oraculum method. If there is task success for at least 5 trials 
then marks as OK and continues to next config.

Requirements:
    - All dependencies from pyproject.toml must be installed
    - Run: pip install -e . (from repository root)
    
Usage:
    # Test all configs with oraculum method (5 trials per config by default)
    python3 myGym/unittest/test_oraculum_configs.py
    
    # Test with custom number of trials
    python3 myGym/unittest/test_oraculum_configs.py --trials 10
    
    # Test a specific config
    python3 myGym/unittest/test_oraculum_configs.py --config train_A.json
    
    # Custom timeout per config
    python3 myGym/unittest/test_oraculum_configs.py --timeout 300
    
    # Minimum successful trials required (default: 5)
    python3 myGym/unittest/test_oraculum_configs.py --min-success 3
"""
import os
import sys
import subprocess
import glob
import argparse
import csv

# ANSI colors for output marks
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS_DIR = os.path.join(PROJECT_ROOT, 'configs')
GEN_CONFIGS_DIR = os.path.join(PROJECT_ROOT, 'configs_gen')
TEST_SCRIPT = os.path.join(PROJECT_ROOT, 'test.py')


def clean_oraculum_results(oraculum_results_dir: str) -> None:
    """
    Clean up old results files before/after running tests.
    """
    if os.path.exists(oraculum_results_dir):
        old_files = glob.glob(os.path.join(oraculum_results_dir, 'results*.csv'))
        for f in old_files:
            try:
                os.remove(f)
            except:
                pass

def parse_latest_results_file(oraculum_results_dir: str) -> tuple[int, list[tuple[int, str]], str | None]:
    """
    Parse the latest results CSV file.

    Returns:
        success_count: Number of successful runs
        failed_subtask_sequences: List of (run_idx, subtask_sequence)
        error: Error message, or None
    """

    # Find the most recently created results file
    result_files = glob.glob(os.path.join(oraculum_results_dir, "results*.csv"))

    if not result_files:
        return 0, [], "No results CSV file was created"

    latest_file = max(result_files, key=os.path.getctime)

    # Read the file
    success_count = 0
    failed_subtask_sequences = []

    try:
        with open(latest_file, "r") as f:
            reader = csv.DictReader(f)

            for run_idx, row in enumerate(reader, 1):
                is_success = row["Success"].strip() == "True"

                if is_success:
                    success_count += 1
                else:
                    failed_subtask_sequences.append(
                        (run_idx, row["Subtasks"].strip())
                    )

    except Exception as e:
        return 0, [], f"Failed to parse results file: {str(e)}"

    return success_count, failed_subtask_sequences, None


def test_config_with_oraculum(
    config_path: str,
    oraculum_results_dir: str,
    trials: int = 5,
    timeout: int = 300,
    min_success: int = 5,
    ) -> tuple[bool, int, str | None, list[tuple[int, str]]]:
    """
    Test a single config by running test.py with oraculum control.
    
    Args:
        config_path: Path to the config file
        trials: Number of trials (eval_episodes) to run (default: 5)
        timeout: Timeout in seconds (default: 300 = 5 minutes)
        min_success: Minimum number of successful trials required (default: 5)
    
    Returns:
        tuple: (success: bool, success_count: int, error_message: str or None)
    """
    
    try:        
        # Run test.py with oraculum control
        cmd = [
            sys.executable,
            TEST_SCRIPT,
            '--config', config_path,
            '-ct', 'oraculum',
            '-ba', 'absolute_gripper',
            '-g', '0',  # No GUI
            '--eval_episodes', str(trials),
            '-rr', 'True'  # Enable results report
        ]
        
        # Run the command and capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=PROJECT_ROOT
        )
        
        # Parse the output to count successful episodes
        # The test.py script should write results to a CSV file
        success_count = 0
        failed_subtask_sequences = []
        
        # Try to find results in oraculum_results directory
        if os.path.exists(oraculum_results_dir):

            success_count, failed_subtask_sequences, parse_error = parse_latest_results_file(
                oraculum_results_dir
            )

            if parse_error:
                return False, 0, parse_error, []
            
        
        # If we didn't get results from file, check if there was an error
        if success_count == 0 and result.returncode != 0:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            if not error_msg and result.stdout:
                # Get last few lines of stdout for error context
                stdout_lines = result.stdout.strip().split('\n')
                error_msg = '\n'.join(stdout_lines[-5:])
            return False, 0, error_msg, []
        
        # Check if we met the minimum success criteria
        if success_count >= min_success:
            return True, success_count, None, failed_subtask_sequences
        else:
            return (
                False,
                success_count,
                f"Only {success_count}/{trials} trials succeeded (minimum required: {min_success})",
                failed_subtask_sequences,
    )
  
    except subprocess.TimeoutExpired:
        return False, 0, f"Testing timed out after {timeout} seconds", []
    except Exception as e:
        return False, 0, str(e), []


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    
    parser = argparse.ArgumentParser(
        description='Test oraculum method with all configs in ./configs folder'
    )
    parser.add_argument(
        '--trials',
        type=int,
        default=5,
        help='Number of trials (eval_episodes) to run per config (default: 5)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=300,
        help='Timeout in seconds for each config test (default: 300)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Test only a specific config file (provide filename or path)'
    )
    parser.add_argument(
        '--min-success',
        type=int,
        default=5,
        help='Minimum number of successful trials required (default: 5)'
    )
    parser.add_argument(
        "--gen",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use generated config files from configs_gen. Use --no-gen for configs."
    )
    
    args = parser.parse_args()
    return args

    
def return_configs_path(args: argparse.Namespace) -> list[str]:
    """
    Return paths to config files selected by command-line arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        List of config file paths.
    """

    configs_dir = GEN_CONFIGS_DIR if args.gen else CONFIGS_DIR

    if args.config:
        # Test only the specified config
        if os.path.isabs(args.config):
            config_path = args.config
        else:
            config_path = os.path.join(configs_dir, args.config)

        return [config_path] if os.path.exists(config_path) else []

    else:
        # Find all JSON config files in the configs directory
        return sorted(glob.glob(os.path.join(configs_dir, "*.json")))

def get_unique_filepath(directory: str, filename: str) -> str:
    """
    Generate unique filepath by appending number if the file already exists.
    """
    
    summary_path = os.path.join(directory, f"{filename}.txt")

    if not os.path.exists(summary_path):
        return summary_path
    
    index = 1

    while True:
        summary_path = os.path.join(directory, f"{filename}{index}.txt")

        if not os.path.exists(summary_path):
            return summary_path

        index += 1

def save_result_summary(
    oraculum_results_dir: str,
    config_files: list[str],
    successful_configs: list[tuple[str, int]],
    failed_configs: list[tuple[str, int, str | None, list[tuple[int, str]]]],
    trials: int,
    ) -> None:
    """
    Save the final test summary to a text file.

    Args:
        oraculum_results_dir: Directory where the summary file will be saved.
        config_files: All tested config file paths.
        successful_configs: Successfully tested configs with success counts.
        failed_configs: Failed configs with errors and failed subtask sequences.
        trials: Number of trials run per config.
    """

    summary_path = get_unique_filepath(oraculum_results_dir, "results_summary")
    with open(summary_path, "w") as f:
        f.write("SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total configs tested: {len(config_files)}\n")
        f.write(f"Successful: {len(successful_configs)}\n")
        f.write(f"Failed: {len(failed_configs)}\n")

        if successful_configs:
            f.write("\nSUCCESSFULLY TESTED CONFIGS (ORACULUM)\n")
            f.write("=" * 80 + "\n")
            for i, (config_name, success_count) in enumerate(successful_configs, 1):
                f.write(f"{i:2d}. OK {config_name} ({success_count}/{trials} successful)\n")

        if failed_configs:
            f.write("\nFAILED CONFIGS\n")
            f.write("=" * 80 + "\n")
            for config_name, success_count, error, failed_subtasks in failed_configs:
                f.write(f"FAIL {config_name} ({success_count}/{trials} successful)\n")

                if failed_subtasks:
                    f.write("  Failed subtask sequences:\n")
                    for run_idx, subtasks in failed_subtasks:
                        f.write(f"    Run {run_idx}: {subtasks}\n")

                if error:
                    f.write(f"  Error: {error}\n")
                    
def print_init_info(config_files: list[str], args: argparse.Namespace) -> None:
    """
    Print initial information about the test run.
    """

    print(f"Testing {len(config_files)} config file(s) with oraculum control")
    print(f"Trials per config: {args.trials}")
    print(f"Minimum successful trials required: {args.min_success}")
    print(f"Timeout per config: {args.timeout} seconds")
    print("="*80 + "\n")

def print_summary(
    config_files: list[str],
    successful_configs: list[tuple[str, int]],
    failed_configs: list[tuple[str, int, str | None, list[tuple[int, str]]]],
    trials: int,
    ) -> None:
    """
    Print the final summary of the tests success.
    """

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total configs tested: {len(config_files)}")
    print(f"Successful: {GREEN}{len(successful_configs)}{RESET}")
    print(f"Failed: {RED}{len(failed_configs)}{RESET}")
    
    # Table of successfully tested configs
    if successful_configs:
        print("\n" + "="*80)
        print("SUCCESSFULLY TESTED CONFIGS (ORACULUM)")
        print("="*80)
        for i, (config_name, success_count) in enumerate(successful_configs, 1):
            print(f"  {i:2d}. {GREEN}✔{RESET} {config_name} ({success_count}/{trials} successful)")
    
    # Table of failed configs with error messages
    if failed_configs:
        print("\n" + "="*80)
        print("FAILED CONFIGS")
        print("="*80)
        for config_name, success_count, error, failed_subtasks in failed_configs:
            print(f"  {RED}✖{RESET} {config_name} ({success_count}/{trials} successful)")
            # Print error message for brevity
            if error:
                error_lines = error.split('\n')[:3]
                for line in error_lines:
                    if line.strip():
                        print(f"     {line[:100]}")
    
    print("\n")



def main() -> int:
    """
    Run oraculum tests for selected config files and save the final summary.

    Returns:
        Exit code: 0 if all configs passed, otherwise 1.
    """

    args = parse_args()

    # Validate that min_success doesn't exceed trials
    if args.min_success > args.trials:
        print(f"{RED}Error: --min-success ({args.min_success}) cannot exceed --trials ({args.trials}){RESET}")
        return 1
    
    config_files = return_configs_path(args)

    if not config_files:
        if args.config:
            print(f"{RED}Config file not found: {args.config}{RESET}")
        else:
            print(f"No config files found in {CONFIGS_DIR}")
        return 1
    
    # Create oraculum_results directory if it doesn't exist
    oraculum_results_dir = os.path.join(PROJECT_ROOT, 'oraculum_results')
    os.makedirs(oraculum_results_dir, exist_ok=True)
    clean_oraculum_results(oraculum_results_dir)
    
    print_init_info(config_files, args)
    
    successful_configs = []
    failed_configs = []
    
    # Test each config file
    for idx, config_path in enumerate(config_files, 1):
        config_name = os.path.basename(config_path)
        print(f"[{idx}/{len(config_files)}] Testing: {config_name}...", end=" ", flush=True)
        
        success, success_count, error, failed_subtasks = test_config_with_oraculum(
            config_path, 
            oraculum_results_dir,
            trials=args.trials, 
            timeout=args.timeout,
            min_success=args.min_success
        )
        
        if success:
            print(f"{GREEN}✔ OK{RESET} ({success_count}/{args.trials} successful)")
            successful_configs.append((config_name, success_count))
        else:
            print(f"{RED}✖ FAIL{RESET} ({success_count}/{args.trials} successful)")
            failed_configs.append((config_name, success_count, error, failed_subtasks))
    
    print_summary(config_files, successful_configs, failed_configs, args.trials)
    save_result_summary(
    oraculum_results_dir,
    config_files,
    successful_configs,
    failed_configs,
    args.trials
    )
    clean_oraculum_results(oraculum_results_dir)
    
    # Return exit code based on results
    return 0 if len(failed_configs) == 0 else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code if exit_code is not None else 0)
