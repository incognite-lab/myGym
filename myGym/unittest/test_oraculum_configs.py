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
import re
import tempfile

# ANSI colors for output marks
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS_DIR = os.path.join(PROJECT_ROOT, 'configs')
TEST_SCRIPT = os.path.join(PROJECT_ROOT, 'test.py')


def test_config_with_oraculum(config_path: str, trials: int = 5, timeout: int = 300, min_success: int = 5) -> tuple:
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
    config_name = os.path.basename(config_path)
    
    try:
        # Clean up old results files before running
        oraculum_results_dir = os.path.join(PROJECT_ROOT, 'oraculum_results')
        if os.path.exists(oraculum_results_dir):
            old_files = glob.glob(os.path.join(oraculum_results_dir, 'results*.csv'))
            for f in old_files:
                try:
                    os.remove(f)
                except:
                    pass
        
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
        
        # Try to find results in oraculum_results directory
        if os.path.exists(oraculum_results_dir):
            # Find the most recently created results file
            result_files = glob.glob(os.path.join(oraculum_results_dir, 'results*.csv'))
            if result_files:
                # Get the most recent file
                latest_file = max(result_files, key=os.path.getctime)
                # Parse CSV to count successes
                try:
                    with open(latest_file, 'r') as f:
                        lines = f.readlines()
                        # Skip header line
                        for line in lines[1:]:
                            if line.strip():
                                # The last column should be Success (True/False)
                                if 'True' in line:
                                    success_count += 1
                    # Clean up the results file after reading
                    try:
                        os.remove(latest_file)
                    except:
                        pass
                except Exception as e:
                    # If we can't parse the file, return error
                    return False, 0, f"Failed to parse results file: {str(e)}"
        
        # If we didn't get results from file, check if there was an error
        if success_count == 0 and result.returncode != 0:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            if not error_msg and result.stdout:
                # Get last few lines of stdout for error context
                stdout_lines = result.stdout.strip().split('\n')
                error_msg = '\n'.join(stdout_lines[-5:])
            return False, 0, error_msg
        
        # Check if we met the minimum success criteria
        if success_count >= min_success:
            return True, success_count, None
        else:
            return False, success_count, f"Only {success_count}/{trials} trials succeeded (minimum required: {min_success})"
            
    except subprocess.TimeoutExpired:
        return False, 0, f"Testing timed out after {timeout} seconds"
    except Exception as e:
        return False, 0, str(e)


def main():
    """
    Main function to test all configs with oraculum control.
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
    
    args = parser.parse_args()
    
    # Validate that min_success doesn't exceed trials
    if args.min_success > args.trials:
        print(f"{RED}Error: --min-success ({args.min_success}) cannot exceed --trials ({args.trials}){RESET}")
        return 1
    
    # Find all JSON config files in the configs directory
    if args.config:
        # Test only the specified config
        if os.path.isabs(args.config):
            config_files = [args.config] if os.path.exists(args.config) else []
        else:
            config_path = os.path.join(CONFIGS_DIR, args.config)
            config_files = [config_path] if os.path.exists(config_path) else []
        
        if not config_files:
            print(f"{RED}Config file not found: {args.config}{RESET}")
            return 1
    else:
        config_files = sorted(glob.glob(os.path.join(CONFIGS_DIR, '*.json')))
    
    if not config_files:
        print(f"No config files found in {CONFIGS_DIR}")
        return 1
    
    # Create oraculum_results directory if it doesn't exist
    oraculum_results_dir = os.path.join(PROJECT_ROOT, 'oraculum_results')
    os.makedirs(oraculum_results_dir, exist_ok=True)
    
    print(f"Testing {len(config_files)} config file(s) with oraculum control")
    print(f"Trials per config: {args.trials}")
    print(f"Minimum successful trials required: {args.min_success}")
    print(f"Timeout per config: {args.timeout} seconds")
    print("="*80 + "\n")
    
    successful_configs = []
    failed_configs = []
    
    # Test each config file
    for idx, config_path in enumerate(config_files, 1):
        config_name = os.path.basename(config_path)
        print(f"[{idx}/{len(config_files)}] Testing: {config_name}...", end=" ", flush=True)
        
        success, success_count, error = test_config_with_oraculum(
            config_path, 
            trials=args.trials, 
            timeout=args.timeout,
            min_success=args.min_success
        )
        
        if success:
            print(f"{GREEN}✔ OK{RESET} ({success_count}/{args.trials} successful)")
            successful_configs.append((config_name, success_count))
        else:
            print(f"{RED}✖ FAIL{RESET} ({success_count}/{args.trials} successful)")
            failed_configs.append((config_name, success_count, error))
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total configs tested: {len(config_files)}")
    print(f"Successful: {GREEN}{len(successful_configs)}{RESET}")
    print(f"Failed: {RED}{len(failed_configs)}{RESET}")
    
    # Print table of successfully tested configs
    if successful_configs:
        print("\n" + "="*80)
        print("SUCCESSFULLY TESTED CONFIGS (ORACULUM)")
        print("="*80)
        for i, (config_name, success_count) in enumerate(successful_configs, 1):
            print(f"  {i:2d}. {GREEN}✔{RESET} {config_name} ({success_count}/{args.trials} successful)")
    
    # Print failed configs with error messages
    if failed_configs:
        print("\n" + "="*80)
        print("FAILED CONFIGS")
        print("="*80)
        for config_name, success_count, error in failed_configs:
            print(f"  {RED}✖{RESET} {config_name} ({success_count}/{args.trials} successful)")
            # Print error message for brevity
            if error:
                error_lines = error.split('\n')[:3]
                for line in error_lines:
                    if line.strip():
                        print(f"     {line[:100]}")
    
    print("\n")
    
    # Return exit code based on results
    return 0 if len(failed_configs) == 0 else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code if exit_code is not None else 0)
