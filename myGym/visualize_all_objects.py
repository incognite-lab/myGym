#!/usr/bin/env python3
"""
Visualize all URDF objects from the myGym/envs/objects directory.
This script recursively finds all URDF files and displays them in a square grid on a plane.
"""

import pybullet as p
import pybullet_data
import argparse
import time
import os
import math
import importlib.resources as pkg_resources


def find_urdf_files(directory):
    """
    Recursively find all URDF files in a directory.
    
    Args:
        directory (str): Root directory to search for URDF files
        
    Returns:
        list: List of paths to URDF files
    """
    urdf_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.urdf'):
                urdf_files.append(os.path.join(root, file))
    return sorted(urdf_files)


def calculate_grid_dimensions(num_objects):
    """
    Calculate grid dimensions for a square grid.
    
    Args:
        num_objects (int): Number of objects to place in grid
        
    Returns:
        tuple: (rows, cols) dimensions for the grid
    """
    # Calculate square root and round up for rows and cols
    grid_size = math.ceil(math.sqrt(num_objects))
    return grid_size, grid_size


def calculate_grid_position(index, grid_cols, spacing, center_offset):
    """
    Calculate position for an object in the grid.
    
    Args:
        index (int): Index of the object in the list
        grid_cols (int): Number of columns in the grid
        spacing (float): Spacing between objects
        center_offset (tuple): Offset to center the grid
        
    Returns:
        tuple: (x, y, z) position for the object
    """
    row = index // grid_cols
    col = index % grid_cols
    
    x = col * spacing + center_offset[0]
    y = row * spacing + center_offset[1]
    z = 0.1  # Slightly above the ground
    
    return [x, y, z]


def main():
    parser = argparse.ArgumentParser(
        description='Visualize all URDF objects from myGym/envs/objects directory in a grid'
    )
    parser.add_argument(
        '-s', '--spacing',
        type=float,
        default=0.3,
        help='Spacing between objects in the grid (default: 0.3 meters)'
    )
    parser.add_argument(
        '-p', '--plane',
        type=str,
        default='plane.urdf',
        help='Plane URDF file to use - can be a filename (searched in myGym/envs/rooms/) or absolute path (default: plane.urdf)'
    )
    parser.add_argument(
        '--no-gui',
        action='store_true',
        help='Run without GUI (useful for testing)'
    )
    
    args = parser.parse_args()
    
    # Get the path to the objects directory
    try:
        currentdir = os.path.join(pkg_resources.files("myGym"), "envs")
    except (AttributeError, TypeError, ModuleNotFoundError):
        # Fallback for older Python versions or different package structure
        currentdir = os.path.join(os.path.dirname(__file__), "envs")
    
    objects_dir = os.path.join(currentdir, "objects")
    
    if not os.path.exists(objects_dir):
        print(f"Error: Objects directory not found at {objects_dir}")
        return 1
    
    # Find all URDF files
    print(f"Searching for URDF files in {objects_dir}...")
    urdf_files = find_urdf_files(objects_dir)
    
    if not urdf_files:
        print("No URDF files found!")
        return 1
    
    print(f"Found {len(urdf_files)} URDF files")
    
    # Calculate grid dimensions
    grid_rows, grid_cols = calculate_grid_dimensions(len(urdf_files))
    print(f"Grid dimensions: {grid_rows}x{grid_cols}")
    
    # Calculate center offset to center the grid at origin
    center_offset = (
        -(grid_cols - 1) * args.spacing / 2,
        -(grid_rows - 1) * args.spacing / 2
    )
    
    # Initialize PyBullet
    if args.no_gui:
        physicsClient = p.connect(p.DIRECT)
    else:
        physicsClient = p.connect(p.GUI)
    
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    # Load plane
    # Try to find the plane in the rooms directory, or use absolute path if provided
    if os.path.isabs(args.plane):
        plane_path = args.plane
    else:
        plane_path = os.path.join(currentdir, "rooms", args.plane)
    
    if not os.path.exists(plane_path):
        # Try with pybullet_data if not found in myGym
        print(f"Plane not found at {plane_path}, using PyBullet's default plane")
        try:
            plane_id = p.loadURDF("plane.urdf")
            print("Loaded PyBullet's default plane")
        except Exception as e:
            print(f"Error loading plane: {e}")
            return 1
    else:
        plane_id = p.loadURDF(plane_path, useFixedBase=True)
        print(f"Loaded plane from: {plane_path}")
    
    # Load all objects in grid
    loaded_objects = []
    for i, urdf_path in enumerate(urdf_files):
        try:
            position = calculate_grid_position(i, grid_cols, args.spacing, center_offset)
            
            # Load the object
            obj_id = p.loadURDF(
                urdf_path,
                basePosition=position,
                useFixedBase=False
            )
            
            loaded_objects.append({
                'id': obj_id,
                'path': urdf_path,
                'name': os.path.basename(urdf_path)
            })
            
            print(f"[{i+1}/{len(urdf_files)}] Loaded: {os.path.basename(urdf_path)}")
            
        except Exception as e:
            print(f"[{i+1}/{len(urdf_files)}] Failed to load {os.path.basename(urdf_path)}: {e}")
    
    print(f"\nSuccessfully loaded {len(loaded_objects)} out of {len(urdf_files)} objects")
    
    # Set up camera view
    if not args.no_gui:
        # Calculate camera distance based on grid size
        camera_distance = max(3.0, (grid_rows * args.spacing) * 0.8)
        
        p.resetDebugVisualizerCamera(
            cameraDistance=camera_distance,
            cameraYaw=45,
            cameraPitch=-45,
            cameraTargetPosition=[0, 0, 0]
        )
        
        print("\nVisualization running...")
        print("Press Ctrl+C to exit")
        
        # Keep the simulation running
        p.setRealTimeSimulation(1)
        
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nShutting down...")
    
    p.disconnect()
    return 0


if __name__ == "__main__":
    exit(main())
