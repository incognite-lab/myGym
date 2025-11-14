#!/usr/bin/env python3
"""
Script to scan all URDF files in env/robots subfolders recursively
and compare them with the r_dict urdf list in helpers.py.
Prints all URDFs that are not in the r_dict list.
Offers interactive deletion of unused URDFs and mesh files.
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


def find_all_meshes(robots_dir):
    """
    Recursively find all mesh files (obj, stl, dae) in the robots directory.
    
    Args:
        robots_dir: Path to the robots directory
        
    Returns:
        List of relative paths to all mesh files
    """
    mesh_extensions = ('.obj', '.stl', '.dae')
    mesh_files = []
    
    for root, dirs, files in os.walk(robots_dir):
        for file in files:
            if file.lower().endswith(mesh_extensions):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, PROJECT_ROOT)
                mesh_files.append(rel_path)
    
    return sorted(mesh_files)


def extract_meshes_from_urdf(urdf_path):
    """
    Extract all mesh file references from a URDF file.
    
    Args:
        urdf_path: Path to the URDF file
        
    Returns:
        Set of mesh file paths referenced in the URDF
    """
    full_path = os.path.join(PROJECT_ROOT, urdf_path)
    
    if not os.path.exists(full_path):
        return set()
    
    meshes = set()
    
    try:
        with open(full_path, 'r') as f:
            content = f.read()
        
        # Find all mesh filename references
        # Pattern matches: filename="path/to/mesh.obj" or filename='path/to/mesh.obj'
        pattern = r'filename=["\']([^"\']+\.(?:obj|stl|dae))["\']'
        matches = re.findall(pattern, content, re.IGNORECASE)
        
        urdf_dir = os.path.dirname(full_path)
        
        for mesh_ref in matches:
            # Handle relative paths
            if mesh_ref.startswith('./') or mesh_ref.startswith('../'):
                mesh_full_path = os.path.normpath(os.path.join(urdf_dir, mesh_ref))
                mesh_rel_path = os.path.relpath(mesh_full_path, PROJECT_ROOT)
            else:
                # Assume it's relative to URDF location
                mesh_full_path = os.path.normpath(os.path.join(urdf_dir, mesh_ref))
                mesh_rel_path = os.path.relpath(mesh_full_path, PROJECT_ROOT)
            
            meshes.add(mesh_rel_path)
    
    except Exception as e:
        print(f"Warning: Error reading {urdf_path}: {e}")
    
    return meshes


def check_unused_meshes(robots_dir):
    """
    Find all mesh files that are not referenced in any URDF.
    
    Args:
        robots_dir: Path to the robots directory
        
    Returns:
        List of unused mesh file paths
    """
    print("\n" + "=" * 60)
    print("Scanning for mesh files...")
    print("=" * 60)
    
    # Find all mesh files
    all_meshes = find_all_meshes(robots_dir)
    print(f"Found {len(all_meshes)} mesh files total.")
    
    # Find all URDFs
    all_urdfs = find_all_urdfs(robots_dir)
    print(f"Scanning {len(all_urdfs)} URDF files for mesh references...")
    
    # Extract all mesh references from URDFs
    used_meshes = set()
    for urdf in all_urdfs:
        meshes_in_urdf = extract_meshes_from_urdf(urdf)
        used_meshes.update(meshes_in_urdf)
    
    print(f"Found {len(used_meshes)} mesh references in URDFs.")
    
    # Find unused meshes
    unused_meshes = []
    for mesh in all_meshes:
        if mesh not in used_meshes:
            unused_meshes.append(mesh)
    
    return sorted(unused_meshes)


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


def delete_file_interactive(file_path, file_type="file"):
    """
    Interactively ask user if they want to delete a file.
    
    Args:
        file_path: Relative path to the file
        file_type: Type of file (for display purposes)
        
    Returns:
        True if file was deleted, False otherwise, None to quit
    """
    full_path = os.path.join(PROJECT_ROOT, file_path)
    
    if not os.path.exists(full_path):
        print(f"  Warning: File not found: {full_path}")
        return False
    
    # Get file size for display
    file_size = os.path.getsize(full_path)
    size_kb = file_size / 1024
    
    print(f"\n{file_type}: {file_path}")
    print(f"Size: {size_kb:.2f} KB")
    print(f"Full path: {full_path}")
    
    while True:
        response = input("Delete this file? [y/n/q]: ").strip().lower()
        
        if response == 'q':
            return None  # Signal to quit
        elif response == 'y':
            try:
                os.remove(full_path)
                print(f"  ✓ Deleted: {file_path}")
                return True
            except Exception as e:
                print(f"  ✗ Error deleting file: {e}")
                return False
        elif response == 'n':
            print(f"  ⊘ Skipped: {file_path}")
            return False
        else:
            print("  Invalid input. Please enter 'y' (yes), 'n' (no), or 'q' (quit).")


def interactive_cleanup(files_list, file_type="file"):
    """
    Interactively delete files from a list.
    
    Args:
        files_list: List of file paths to potentially delete
        file_type: Type of files (for display)
        
    Returns:
        Tuple of (deleted_count, skipped_count)
    """
    deleted_count = 0
    skipped_count = 0
    
    for file_path in files_list:
        result = delete_file_interactive(file_path, file_type)
        
        if result is None:  # User chose to quit
            print("\nDeletion process terminated by user.")
            break
        elif result:
            deleted_count += 1
        else:
            skipped_count += 1
    
    return deleted_count, skipped_count


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
        
        # Offer deletion or mesh check
        print("\n" + "=" * 60)
        print("Cleanup options")
        print("=" * 60)
        print("  [d] Delete unused URDFs")
        print("  [c] Check for unused mesh files")
        print("  [q] Quit")
        
        while True:
            response = input("\nChoose an option [d/c/q]: ").strip().lower()
            
            if response == 'q':
                print("Exiting.")
                return 0
            elif response == 'd':
                # Delete URDFs
                confirm = input(f"\nDelete {len(missing_urdfs)} unused URDFs one by one? [y/n]: ").strip().lower()
                
                if confirm == 'y':
                    deleted, skipped = interactive_cleanup(missing_urdfs, "URDF")
                    
                    print("\n" + "=" * 60)
                    print(f"URDF Cleanup Summary:")
                    print(f"  Deleted: {deleted}")
                    print(f"  Skipped: {skipped}")
                    print(f"  Remaining: {len(missing_urdfs) - deleted - skipped}")
                    print("=" * 60)
                else:
                    print("\nNo URDFs were deleted.")
                break
            elif response == 'c':
                # Check meshes
                unused_meshes = check_unused_meshes(robots_dir)
                
                if unused_meshes:
                    print(f"\n" + "=" * 60)
                    print(f"Unused mesh files ({len(unused_meshes)}):")
                    print("-" * 60)
                    for mesh in unused_meshes:
                        print(f"  {mesh}")
                    
                    confirm = input(f"\nDelete {len(unused_meshes)} unused mesh files one by one? [y/n]: ").strip().lower()
                    
                    if confirm == 'y':
                        deleted, skipped = interactive_cleanup(unused_meshes, "Mesh")
                        
                        print("\n" + "=" * 60)
                        print(f"Mesh Cleanup Summary:")
                        print(f"  Deleted: {deleted}")
                        print(f"  Skipped: {skipped}")
                        print(f"  Remaining: {len(unused_meshes) - deleted - skipped}")
                        print("=" * 60)
                    else:
                        print("\nNo mesh files were deleted.")
                else:
                    print("\nAll mesh files are referenced in URDFs!")
                break
            else:
                print("Invalid option. Please enter 'd', 'c', or 'q'.")
    else:
        print("All URDF files are present in r_dict!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
