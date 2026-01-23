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


def generate_color_palette(num_colors=100):
    """
    Generate a palette of distinct colors using HSV color space.
    
    Args:
        num_colors (int): Number of colors to generate
        
    Returns:
        list: List of (R, G, B, A) tuples with values in [0, 1]
    """
    import colorsys
    colors = []
    for i in range(num_colors):
        # Use golden ratio for better distribution
        hue = (i * 0.618033988749895) % 1.0
        saturation = 0.6 + (i % 3) * 0.15  # Vary saturation slightly
        value = 0.7 + (i % 5) * 0.05  # Vary brightness slightly
        
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append((r, g, b, 1.0))  # Alpha = 1.0 for full opacity
    
    return colors


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
    z = -0.3  # Slightly above the ground
    
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
    color_palette = generate_color_palette(100)
    
    for i, urdf_path in enumerate(urdf_files):
        try:
            # Skip files with "target" in the name
            if "target" in os.path.basename(urdf_path).lower():
                print(f"[{i+1}/{len(urdf_files)}] Skipped (target): {os.path.basename(urdf_path)}")
                continue
            
            position = calculate_grid_position(i, grid_cols, args.spacing, center_offset)
            
            # Load the object
            obj_id = p.loadURDF(
                urdf_path,
                basePosition=position,
                useFixedBase=False
            )
            time.sleep(0.05)
            # Change color using the palette (cycle through colors)
            color = color_palette[i % len(color_palette)]
            
            # Get the number of links (including base link which is -1)
            num_links = p.getNumJoints(obj_id)
            
            # Change color of base link
            try:
                p.changeVisualShape(obj_id, -1, rgbaColor=color)
            except Exception as e:
                pass  # Some objects may not support color change
            
            # Change color of all other links
            for link_idx in range(num_links):
                try:
                    p.changeVisualShape(obj_id, link_idx, rgbaColor=color)
                except Exception as e:
                    pass  # Some links may not support color change
            
            loaded_objects.append({
                'id': obj_id,
                'path': urdf_path,
                'name': os.path.basename(urdf_path),
                'initial_pos': position  # Store initial spawn position
            })
            
            print(f"[{i+1}/{len(urdf_files)}] Loaded: {os.path.basename(urdf_path)}")
            
        except Exception as e:
            print(f"[{i+1}/{len(urdf_files)}] Failed to load {os.path.basename(urdf_path)}: {e}")
    
    print(f"\nSuccessfully loaded {len(loaded_objects)} out of {len(urdf_files)} objects")
    
    # Enable realtime simulation
    p.setRealTimeSimulation(1)
    
    # Run simulation steps to let objects settle
    print("\nRunning 200 simulation steps...")
    for step in range(200):
        time.sleep(1./240.)  # Small delay for stability
    
    # Check for objects with problematic mass/gravity
    high_objects = []  # z > 0.2 (floating)
    low_objects = []   # z < -2 (sunken/fallen through)
    drifted_objects = []  # moved more than 0.1 in x or y
    
    for obj in loaded_objects:
        pos, _ = p.getBasePositionAndOrientation(obj['id'])
        initial_pos = obj['initial_pos']
        
        # Check z position
        if pos[2] > 0.2:
            high_objects.append({
                'name': obj['name'],
                'z': pos[2]
            })
        elif pos[2] < -2:
            low_objects.append({
                'name': obj['name'],
                'z': pos[2]
            })
        
        # Check x, y drift from initial position
        x_drift = abs(pos[0] - initial_pos[0])
        y_drift = abs(pos[1] - initial_pos[1])
        if x_drift > 0.2 or y_drift > 0.2:
            drifted_objects.append({
                'name': obj['name'],
                'x_drift': x_drift,
                'y_drift': y_drift
            })
    
    if high_objects:
        print("\n=== Objects with problematic mass and gravity (floating, z > 0.2) ===")
        for obj in high_objects:
            print(f"  - {obj['name']} (z={obj['z']:.3f})")
        print(f"Total: {len(high_objects)} objects")
    
    if low_objects:
        print("\n=== Objects with problematic missing collision mesh (possibly target) (too low, z < 0.02) ===")
        for obj in low_objects:
            print(f"  - {obj['name']} (z={obj['z']:.3f})")
        print(f"Total: {len(low_objects)} objects")
    
    if drifted_objects:
        print("\n=== Objects with problematic origin (drifted > 0.2 in x or y) ===")
        for obj in drifted_objects:
            print(f"  - {obj['name']} (x_drift={obj['x_drift']:.3f}, y_drift={obj['y_drift']:.3f})")
        print(f"Total: {len(drifted_objects)} objects")
    
    if not high_objects and not low_objects and not drifted_objects:
        print("\nAll objects settled properly")
    
    # Add debug text labels above each object
    if not args.no_gui:
        for obj in loaded_objects:
            pos, _ = p.getBasePositionAndOrientation(obj['id'])
            # Remove .urdf extension from name
            display_name = obj['name'].replace('.urdf', '')
            # Add text above the object with better visibility
            p.addUserDebugText(
                text=display_name,
                textPosition=[pos[0], pos[1], pos[2] + 0.2],
                textColorRGB=[1, 1, 1],  # White text
                textSize=1.5,
                lifeTime=0  # Permanent text
            )
    
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
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nShutting down...")
    
    p.disconnect()
    return 0


if __name__ == "__main__":
    exit(main())
