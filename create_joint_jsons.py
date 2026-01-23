#!/usr/bin/env python3
"""
Script to extract joint information from URDF files and create JSON configuration files.
For each URDF file in get_robot_dict(), this script:
1. Parses the URDF to find joints ending with "rjoint" (arm joints) and "gjoint" (gripper joints)
2. Creates a JSON file in the same directory with the same name but .json extension
3. Stores the extracted joints in format: {"arm": [rjoints], "gripper": [gjoints]}
"""

import os
import json
import re
import ast


def get_robot_dict_from_file(helpers_path):
    """
    Parse helpers.py to extract the robot dictionary without importing it.
    
    Args:
        helpers_path: Path to the helpers.py file
        
    Returns:
        dict: Robot dictionary
    """
    with open(helpers_path, 'r') as f:
        content = f.read()
    
    # Find the get_robot_dict function
    # Extract the dictionary definition
    pattern = r'def get_robot_dict\(\):\s*r_dict\s*=\s*(\{.*?\})\s*return r_dict'
    match = re.search(pattern, content, re.DOTALL)
    
    if not match:
        raise ValueError("Could not find get_robot_dict() in helpers.py")
    
    dict_str = match.group(1)
    
    # Parse the dictionary - need to handle numpy arrays
    # Replace numpy.array with list for parsing
    dict_str = re.sub(r'np\.array\((.*?)\)', r'\1', dict_str)
    
    # Safely evaluate the dictionary
    robot_dict = ast.literal_eval(dict_str)
    
    return robot_dict


def extract_joints_from_urdf(urdf_path):
    """
    Extract joints ending with 'rjoint' and 'gjoint' from a URDF file.
    
    Args:
        urdf_path: Path to the URDF file
        
    Returns:
        tuple: (list of rjoints, list of gjoints)
    """
    rjoints = []
    gjoints = []
    
    if not os.path.exists(urdf_path):
        print(f"Warning: URDF file not found: {urdf_path}")
        return rjoints, gjoints
    
    try:
        with open(urdf_path, 'r') as f:
            content = f.read()
        
        # Find all joint names using regex
        # Looking for: <joint name="joint_name_here" ...>
        joint_pattern = r'<joint\s+name="([^"]+)"'
        matches = re.findall(joint_pattern, content)
        
        # Use sets to track unique joints (order is preserved by list append operations)
        seen_rjoints = set()
        seen_gjoints = set()
        
        for joint_name in matches:
            if joint_name.endswith('rjoint') and joint_name not in seen_rjoints:
                rjoints.append(joint_name)
                seen_rjoints.add(joint_name)
            elif joint_name.endswith('gjoint') and joint_name not in seen_gjoints:
                gjoints.append(joint_name)
                seen_gjoints.add(joint_name)
    
    except Exception as e:
        print(f"Error reading {urdf_path}: {e}")
    
    return rjoints, gjoints


def create_json_for_urdf(urdf_path, robot_name):
    """
    Create a JSON file with joint information for a URDF file.
    
    Args:
        urdf_path: Path to the URDF file
        robot_name: Name of the robot (for logging)
    """
    # Extract joints from URDF
    rjoints, gjoints = extract_joints_from_urdf(urdf_path)
    
    # Create JSON data
    json_data = {
        "arm": rjoints,
        "gripper": gjoints
    }
    
    # Create JSON file path (same directory and name as URDF, but .json extension)
    json_path = os.path.splitext(urdf_path)[0] + '.json'
    
    # Write JSON file
    try:
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"✓ Created {json_path}")
        print(f"  - Arm joints ({len(rjoints)}): {rjoints}")
        print(f"  - Gripper joints ({len(gjoints)}): {gjoints}")
    except Exception as e:
        print(f"✗ Error creating {json_path}: {e}")


def main():
    """Main function to process all robots in get_robot_dict()."""
    # Get the base directory (myGym/myGym)
    base_dir = os.path.join(os.path.dirname(__file__), 'myGym')
    
    # Get robot dictionary from helpers.py
    helpers_path = os.path.join(base_dir, 'utils', 'helpers.py')
    robot_dict = get_robot_dict_from_file(helpers_path)
    
    print(f"Processing {len(robot_dict)} robots...\n")
    
    # Process each robot
    for robot_name, robot_info in robot_dict.items():
        urdf_relative_path = robot_info['path']
        
        # Convert relative path to absolute path
        # The path in the dict starts with /envs/robots/...
        urdf_path = os.path.join(base_dir, urdf_relative_path.lstrip('/'))
        
        print(f"\nProcessing {robot_name}:")
        print(f"  URDF: {urdf_path}")
        
        create_json_for_urdf(urdf_path, robot_name)
    
    print("\n" + "="*60)
    print("Processing complete!")


if __name__ == '__main__':
    main()
