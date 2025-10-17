#!/usr/bin/env python3
"""
Unit test that runs train.py with all configs in ./configs folder with steps=10000.
Prints the name of each config with OK mark if there's no error during execution.
At the end, prints a table with successfully trained configs.

Requirements:
    - All dependencies from pyproject.toml must be installed
    - Run: pip install -e . (from repository root)
    
Usage:
    # Test all configs with default settings (10000 steps)
    python3 myGym/unittest/test_train_configs.py
    
    # Test with custom step count
    python3 myGym/unittest/test_train_configs.py --steps 5000
    
    # Test a specific config
    python3 myGym/unittest/test_train_configs.py --config train_A_nico.json
    
    # Custom timeout per config
    python3 myGym/unittest/test_train_configs.py --timeout 600
"""
import os
import sys
import subprocess
import glob
import argparse

# ANSI colors for output marks
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS_DIR = os.path.join(PROJECT_ROOT, 'configs')
TRAIN_SCRIPT = os.path.join(PROJECT_ROOT, 'train.py')


def test_config(config_path: str, steps: int = 10000, timeout: int = 1200) -> tuple:
    """
    Test a single config by running train.py with the specified number of steps.
    
    Args:
        config_path: Path to the config file
        steps: Number of training steps (default: 10000)
        timeout: Timeout in seconds (default: 1200 = 20 minutes)
    
    Returns:
        tuple: (success: bool, error_message: str or None)
    """
    config_name = os.path.basename(config_path)
    
    try:
        # Run train.py with the config and steps parameter
        cmd = [
            sys.executable,
            TRAIN_SCRIPT,
            '--config', config_path,
            '--steps', str(steps)
        ]
        
        # Run the command and capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        # Check if the command succeeded
        if result.returncode == 0:
            return True, None
        else:
            # Extract error message from stderr
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            # Also check stdout for errors
            if not error_msg and result.stdout:
                error_msg = result.stdout.strip()
            return False, error_msg
            
    except subprocess.TimeoutExpired:
        return False, f"Training timed out after {timeout} seconds"
    except Exception as e:
        return False, str(e)


def main():
    """
    Main function to test all configs in the configs folder.
    """
    parser = argparse.ArgumentParser(
        description='Test train.py with all configs in ./configs folder'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=10000,
        help='Number of training steps to run (default: 10000)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=1200,
        help='Timeout in seconds for each config test (default: 1200)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Test only a specific config file (provide filename or path)'
    )
    
    args = parser.parse_args()
    
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
            return
    else:
        config_files = sorted(glob.glob(os.path.join(CONFIGS_DIR, '*.json')))
    
    if not config_files:
        print(f"No config files found in {CONFIGS_DIR}")
        return
    
    print(f"Testing {len(config_files)} config file(s) with steps={args.steps}...")
    print(f"Timeout per config: {args.timeout} seconds")
    print("="*80 + "\n")
    
    successful_configs = []
    failed_configs = []
    
    # Test each config file
    for idx, config_path in enumerate(config_files, 1):
        config_name = os.path.basename(config_path)
        print(f"[{idx}/{len(config_files)}] Testing: {config_name}...", end=" ", flush=True)
        
        success, error = test_config(config_path, steps=args.steps, timeout=args.timeout)
        
        if success:
            print(f"{GREEN}✔ OK{RESET}")
            successful_configs.append(config_name)
        else:
            print(f"{RED}✖ FAIL{RESET}")
            failed_configs.append((config_name, error))
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total configs tested: {len(config_files)}")
    print(f"Successful: {GREEN}{len(successful_configs)}{RESET}")
    print(f"Failed: {RED}{len(failed_configs)}{RESET}")
    
    # Print table of successfully trained configs
    if successful_configs:
        print("\n" + "="*80)
        print("SUCCESSFULLY TRAINED CONFIGS")
        print("="*80)
        for i, config_name in enumerate(successful_configs, 1):
            print(f"  {i:2d}. {GREEN}✔{RESET} {config_name}")
    
    # Print failed configs with error messages
    if failed_configs:
        print("\n" + "="*80)
        print("FAILED CONFIGS")
        print("="*80)
        for config_name, error in failed_configs:
            print(f"  {RED}✖{RESET} {config_name}")
            # Print first few lines of error for brevity
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
