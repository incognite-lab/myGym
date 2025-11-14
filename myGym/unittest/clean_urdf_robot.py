#!/usr/bin/env python3
"""
Script to scan all URDF files in env/robots subfolders recursively
and compare them with the r_dict urdf list in helpers.py.
Prints all URDFs that are not in the r_dict list.
"""
import os
import sys
import re

# Determine project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_all_urdfs(robots_dir):
    """
    Recursively find all URDF files in the robots directory.
    
    Args:
        robots_dir: Path to the robots directory
        
    Returns:
        List of relative paths (relative to myGym root) to all URDF files
    """
    urdf_files = []
    for root, dirs, files in os.walk(robots_dir):
        for file in files:
            if file.endswith('.urdf'):
                full_path = os.path.join(root, file)
                # Convert to relative path from PROJECT_ROOT
                rel_path = os.path.relpath(full_path, PROJECT_ROOT)
                urdf_files.append(rel_path)
    return sorted(urdf_files)


def get_r_dict_paths():
    """
    Extract all URDF paths from r_dict in helpers.py by parsing the file.
    
    Returns:
        Set of paths from r_dict (converted to relative format)
    """
    helpers_path = os.path.join(PROJECT_ROOT, 'utils', 'helpers.py')
    
    if not os.path.exists(helpers_path):
        print(f"Error: helpers.py not found at {helpers_path}")
        sys.exit(1)
    
    paths = set()
    
    # Read and parse helpers.py to extract paths
    with open(helpers_path, 'r') as f:
        content = f.read()
    
    # Find all 'path': '/envs/robots/...' patterns
    pattern = r"'path':\s*'(/envs/robots/[^']+\.urdf)'"
    matches = re.findall(pattern, content)
    
    for path in matches:
        # Convert from '/envs/robots/...' to 'envs/robots/...'
        if path.startswith('/'):
            path = path[1:]
        paths.add(path)
    
    return paths


def main():
    """Main function to compare URDFs and print missing ones."""
    # Define robots directory relative to PROJECT_ROOT
    robots_dir = os.path.join(PROJECT_ROOT, 'envs', 'robots')
    
    if not os.path.exists(robots_dir):
        print(f"Error: Robots directory not found at {robots_dir}")
        sys.exit(1)
    
    print("Scanning for URDF files in envs/robots...")
    all_urdfs = find_all_urdfs(robots_dir)
    print(f"Found {len(all_urdfs)} URDF files total.\n")
    
    print("Extracting URDF paths from r_dict in helpers.py...")
    r_dict_paths = get_r_dict_paths()
    print(f"Found {len(r_dict_paths)} URDF files in r_dict.\n")
    
    # Find URDFs not in r_dict
    missing_urdfs = []
    for urdf in all_urdfs:
        if urdf not in r_dict_paths:
            missing_urdfs.append(urdf)
    
    # Print results
    if missing_urdfs:
        print(f"URDFs not in r_dict ({len(missing_urdfs)}):")
        print("-" * 60)
        for urdf in missing_urdfs:
            print(f"  {urdf}")
    else:
        print("All URDF files are present in r_dict!")
    
    return 0 if not missing_urdfs else 1


if __name__ == "__main__":
    sys.exit(main())
