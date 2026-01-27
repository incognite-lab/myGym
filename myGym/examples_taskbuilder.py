#!/usr/bin/env python3
"""
Example usage of TaskBuilder CLI

This script demonstrates various ways to use the TaskBuilder CLI
to create task configurations.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and display output"""
    print("\n" + "=" * 70)
    print(f"Example: {description}")
    print("=" * 70)
    print(f"Command: {' '.join(cmd)}")
    print("-" * 70)
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    return result.returncode == 0

def main():
    """Run example commands"""
    print("TaskBuilder CLI Examples")
    print("=" * 70)
    
    mygym_dir = Path(__file__).parent.parent
    cli_script = mygym_dir / "myGym" / "taskbuilder_cli.py"
    
    examples = [
        {
            "description": "Create a simple config with Panda robot",
            "cmd": [sys.executable, str(cli_script), 
                   "-o", "/tmp/example1_panda.json",
                   "--set", "robot=panda", "task_type=AG", "algo=ppo"]
        },
        {
            "description": "Create a config for Tiago with SAC algorithm",
            "cmd": [sys.executable, str(cli_script),
                   "-o", "/tmp/example2_tiago.json",
                   "--set", "robot=tiago_dual", "workspace=table_tiago", 
                   "task_type=AGM", "algo=sac", "steps=10000000"]
        },
        {
            "description": "Create a config with multiple parameters",
            "cmd": [sys.executable, str(cli_script),
                   "-o", "/tmp/example3_custom.json",
                   "--set", "robot=nico", "workspace=table_nico",
                   "task_type=reach", "algo=td3", "steps=5000000",
                   "max_episode_steps=256", "eval_freq=100000"]
        },
        {
            "description": "Show config before saving",
            "cmd": [sys.executable, str(cli_script),
                   "-o", "/tmp/example4_show.json",
                   "--set", "robot=g1", "task_type=push",
                   "--show"]
        },
    ]
    
    success_count = 0
    for example in examples:
        if run_command(example["cmd"], example["description"]):
            success_count += 1
    
    print("\n" + "=" * 70)
    print(f"Summary: {success_count}/{len(examples)} examples completed successfully")
    print("=" * 70)
    
    # Show how to use the generated configs
    print("\nTo use these configs:")
    print("  python myGym/test.py --config /tmp/example1_panda.json")
    print("  python myGym/train.py --config /tmp/example2_tiago.json")

if __name__ == "__main__":
    main()
