#!/usr/bin/env python3
"""
TaskBuilder CLI - Command-line interface for creating task configurations

This provides a simplified command-line interface for users who prefer CLI
or are working in headless environments.
"""

import sys
import json
import argparse
import commentjson
from pathlib import Path


class TaskBuilderCLI:
    """Command-line interface for task configuration"""
    
    def __init__(self):
        self.mygym_dir = Path(__file__).parent
        self.config_dir = self.mygym_dir / "configs"
        self.template_path = self.config_dir / "AG.json"
    
    def load_template(self):
        """Load the AG.json template"""
        with open(self.template_path, 'r') as f:
            return commentjson.load(f)
    
    def interactive_config(self):
        """Interactively create a configuration"""
        print("=" * 60)
        print("myGym TaskBuilder - Interactive Configuration")
        print("=" * 60)
        print("\nPress Enter to keep default values in [brackets]\n")
        
        config = self.load_template()
        
        # Key parameters to configure interactively
        key_params = [
            ('workspace', ['table', 'table_nico', 'table_complex', 'table_tiago']),
            ('robot', ['g1', 'panda', 'tiago_dual', 'nico', 'kuka']),
            ('robot_action', ['joints_gripper', 'step_gripper', 'absolute_gripper']),
            ('task_type', ['AG', 'A', 'AGM', 'reach', 'push']),
            ('algo', ['ppo', 'sac', 'td3', 'a2c', 'multippo']),
            ('steps', None),  # Numeric
            ('max_episode_steps', None),
        ]
        
        for param, options in key_params:
            current = config.get(param, '')
            if options:
                print(f"\n{param} (options: {', '.join(options)})")
                value = input(f"  [{current}]: ").strip()
                if value:
                    config[param] = value
            else:
                value = input(f"{param} [{current}]: ").strip()
                if value:
                    try:
                        # Try to convert to int
                        config[param] = int(value)
                    except ValueError:
                        config[param] = value
        
        return config
    
    def modify_config(self, template_path, modifications):
        """Load a config and apply modifications"""
        with open(template_path, 'r') as f:
            config = commentjson.load(f)
        
        for key, value in modifications.items():
            # Try to convert value to appropriate type
            try:
                value_lower = value.lower()
                if value_lower == 'null' or value_lower == 'none':
                    config[key] = None
                elif value_lower == 'true':
                    config[key] = True
                elif value_lower == 'false':
                    config[key] = False
                elif value.isdigit():
                    config[key] = int(value)
                elif value.replace('.', '').isdigit():
                    config[key] = float(value)
                else:
                    config[key] = value
            except:
                config[key] = value
        
        return config


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='TaskBuilder CLI - Create and modify myGym task configurations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python taskbuilder_cli.py -i -o my_config.json
  
  # Create config from template with modifications
  python taskbuilder_cli.py -t configs/AG.json -o my_config.json \\
      --set robot=panda task_type=AGM algo=sac
  
  # Quick config creation
  python taskbuilder_cli.py -o quick_config.json \\
      --set robot=tiago_dual workspace=table_complex
        """
    )
    
    parser.add_argument('-i', '--interactive', action='store_true',
                       help='Interactive mode - prompt for key parameters')
    parser.add_argument('-t', '--template', type=str, default='configs/AG.json',
                       help='Template config file to use (default: configs/AG.json)')
    parser.add_argument('-o', '--output', type=str, required=True,
                       help='Output configuration file path')
    parser.add_argument('--set', nargs='+', metavar='KEY=VALUE',
                       help='Set configuration values (e.g., robot=panda task_type=AG)')
    parser.add_argument('--show', action='store_true',
                       help='Display the generated config before saving')
    
    args = parser.parse_args()
    
    cli = TaskBuilderCLI()
    
    # Determine template path
    template_path = Path(args.template)
    if not template_path.is_absolute():
        template_path = cli.mygym_dir / args.template
    
    if not template_path.exists():
        print(f"Error: Template file not found: {template_path}")
        return 1
    
    # Create config
    if args.interactive:
        config = cli.interactive_config()
    else:
        # Parse modifications
        modifications = {}
        if args.set:
            for item in args.set:
                if '=' in item:
                    key, value = item.split('=', 1)
                    modifications[key] = value
                else:
                    print(f"Warning: Ignoring invalid --set argument: {item}")
        
        config = cli.modify_config(template_path, modifications)
    
    # Show config if requested
    if args.show:
        print("\n" + "=" * 60)
        print("Generated Configuration:")
        print("=" * 60)
        print(json.dumps(config, indent=2))
        print("=" * 60 + "\n")
    
    # Determine output path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = cli.mygym_dir / args.output
    
    # Create directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"✓ Configuration saved to: {output_path}")
    
    # Optionally display command to test/train
    print("\nTo test this configuration:")
    print(f"  python myGym/test.py --config {output_path}")
    print("\nTo train with this configuration:")
    print(f"  python myGym/train.py --config {output_path}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
