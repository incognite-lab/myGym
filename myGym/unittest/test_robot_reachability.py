#!/usr/bin/env python3
"""
Test robot IK reachability across a 3D volume.
This script tests robots from utils/helpers.py r_dict by spawning an IK target
in a grid from [-1,-1,-1] to [1,1,1] with step 0.1 and checking if the robot
can reach each point within a threshold distance of 0.05 (L-norm).
"""

import argparse
import os
import sys
import numpy as np
import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import get_robot_dict


def get_controllable_arm_joints(robot_id, num_joints):
    """Get joint information for arm control (non-fixed joints), excluding gripper joints."""
    joint_idxs = []
    for joint_idx in range(num_joints):
        joint_info = p.getJointInfo(robot_id, joint_idx)
        joint_name = joint_info[1].decode("utf-8")
        joint_type = joint_info[2]
        
        if joint_type != p.JOINT_FIXED:
            # Exclude gripper joints from the main IK control list
            if 'right_' not in joint_name or 'finger' not in joint_name:
                if 'gjoint' not in joint_name:  # Also exclude joints with 'gjoint' pattern
                    joint_idxs.append(joint_idx)
    
    return joint_idxs


def find_end_effector(robot_id):
    """Find the end effector link index."""
    num_joints = p.getNumJoints(robot_id)
    for joint_idx in range(num_joints):
        joint_info = p.getJointInfo(robot_id, joint_idx)
        link_name = joint_info[12].decode("utf-8")
        if link_name == 'endeffector':
            return joint_idx
    return -1


def test_reachability(robot_id, end_effector_idx, joint_idxs, target_pos, target_orientation=None, threshold=0.05):
    """
    Test if robot can reach target position.
    
    Args:
        robot_id: PyBullet robot ID
        end_effector_idx: End effector link index
        joint_idxs: List of controllable joint indices
        target_pos: Target position [x, y, z]
        target_orientation: Target orientation as quaternion (optional)
        threshold: Distance threshold for considering position reached
        
    Returns:
        bool: True if position is reachable within threshold
    """
    # Calculate IK
    if target_orientation is not None:
        ik_solution = p.calculateInverseKinematics(
            robot_id, end_effector_idx, target_pos, target_orientation
        )
    else:
        ik_solution = p.calculateInverseKinematics(
            robot_id, end_effector_idx, target_pos
        )
    
    # Apply IK solution to joints
    for index, joint_idx in enumerate(joint_idxs):
        if index < len(ik_solution):
            p.setJointMotorControl2(
                bodyIndex=robot_id,
                jointIndex=joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=ik_solution[index],
                force=500
            )
    
    # Step simulation to let robot move
    for _ in range(100):
        p.stepSimulation()
    
    # Get actual end effector position
    link_state = p.getLinkState(robot_id, end_effector_idx)
    ee_pos = np.array(link_state[0])
    
    # Calculate distance
    distance = np.linalg.norm(np.array(target_pos) - ee_pos)
    
    return distance <= threshold


def generate_grid_points(min_coords, max_coords, step):
    """Generate a 3D grid of points."""
    x_points = np.arange(min_coords[0], max_coords[0] + step, step)
    y_points = np.arange(min_coords[1], max_coords[1] + step, step)
    z_points = np.arange(min_coords[2], max_coords[2] + step, step)
    
    points = []
    for x in x_points:
        for y in y_points:
            for z in z_points:
                points.append([x, y, z])
    
    return np.array(points)


def compute_bounding_box(points):
    """Compute 3D bounding box from list of points."""
    if len(points) == 0:
        return None, None
    
    points = np.array(points)
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    
    return min_coords, max_coords


def plot_reachable_volume(reachable_points, bounding_box_min, bounding_box_max, robot_name):
    """Create 3D plot of reachable volume and bounding box."""
    if len(reachable_points) == 0:
        print("No reachable points to plot.")
        return
    
    reachable_points = np.array(reachable_points)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot reachable points
    ax.scatter(reachable_points[:, 0], reachable_points[:, 1], reachable_points[:, 2],
               c='blue', marker='o', s=1, alpha=0.3, label='Reachable points')
    
    # Plot bounding box
    if bounding_box_min is not None and bounding_box_max is not None:
        # Create box edges
        r = [bounding_box_min, bounding_box_max]
        for s, e in [
            ([r[0][0], r[0][1], r[0][2]], [r[1][0], r[0][1], r[0][2]]),
            ([r[0][0], r[1][1], r[0][2]], [r[1][0], r[1][1], r[0][2]]),
            ([r[0][0], r[0][1], r[1][2]], [r[1][0], r[0][1], r[1][2]]),
            ([r[0][0], r[1][1], r[1][2]], [r[1][0], r[1][1], r[1][2]]),
            ([r[0][0], r[0][1], r[0][2]], [r[0][0], r[1][1], r[0][2]]),
            ([r[1][0], r[0][1], r[0][2]], [r[1][0], r[1][1], r[0][2]]),
            ([r[0][0], r[0][1], r[1][2]], [r[0][0], r[1][1], r[1][2]]),
            ([r[1][0], r[0][1], r[1][2]], [r[1][0], r[1][1], r[1][2]]),
            ([r[0][0], r[0][1], r[0][2]], [r[0][0], r[0][1], r[1][2]]),
            ([r[1][0], r[0][1], r[0][2]], [r[1][0], r[0][1], r[1][2]]),
            ([r[0][0], r[1][1], r[0][2]], [r[0][0], r[1][1], r[1][2]]),
            ([r[1][0], r[1][1], r[0][2]], [r[1][0], r[1][1], r[1][2]]),
        ]:
            ax.plot3D(*zip(s, e), color='red', linewidth=2)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Reachable Volume for {robot_name}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'/tmp/reachability_{robot_name}.png')
    print(f"Plot saved to /tmp/reachability_{robot_name}.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Test robot IK reachability across a 3D volume"
    )
    parser.add_argument(
        "--robot",
        type=str,
        help="Robot key from r_dict (if not provided, interactive selection)"
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Run with PyBullet GUI (default: no GUI)"
    )
    parser.add_argument(
        "--with-orientation",
        action="store_true",
        help="Use orientation constraint in IK (default: position only)"
    )
    parser.add_argument(
        "--euler",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        help="Euler angles for orientation [roll, pitch, yaw] in radians (default: 0 0 0)"
    )
    parser.add_argument(
        "--min",
        type=float,
        nargs=3,
        default=[-1.0, -1.0, -1.0],
        help="Minimum coordinates [x, y, z] (default: -1 -1 -1)"
    )
    parser.add_argument(
        "--max",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        help="Maximum coordinates [x, y, z] (default: 1 1 1)"
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.1,
        help="Grid step size (default: 0.1)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Distance threshold for reachability (default: 0.05)"
    )
    
    args = parser.parse_args()
    
    # Get robot dictionary
    r_dict = get_robot_dict()
    
    # Select robot
    if args.robot:
        if args.robot not in r_dict:
            print(f"Error: Robot '{args.robot}' not found in r_dict")
            print(f"Available robots: {', '.join(sorted(r_dict.keys()))}")
            return 1
        selected_key = args.robot
    else:
        # Interactive selection
        robots = sorted(r_dict.keys())
        print("Available robots:")
        for i, robot_key in enumerate(robots):
            print(f"[{i}] {robot_key}")
        
        while True:
            sel = input("Select robot index (q to quit): ").strip().lower()
            if sel in ("q", "quit", "exit"):
                return 0
            if sel.isdigit():
                idx = int(sel)
                if 0 <= idx < len(robots):
                    selected_key = robots[idx]
                    break
            print("Invalid selection.")
    
    print(f"\nSelected robot: {selected_key}")
    
    # Get robot info
    robot_info = r_dict[selected_key]
    rel_path = robot_info['path'].lstrip('/')
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    urdf_path = os.path.join(base_dir, rel_path)
    
    if not os.path.exists(urdf_path):
        print(f"Error: URDF file not found: {urdf_path}")
        return 1
    
    robot_base_pos = list(np.array(robot_info.get('position', [0.0, 0.0, 0.0])).astype(float))
    robot_base_orientation = robot_info.get('orientation', [0.0, 0.0, 0.0])
    robot_base_quat = p.getQuaternionFromEuler(robot_base_orientation)
    
    # Initialize PyBullet
    if args.gui:
        physics_client = p.connect(p.GUI)
        p.resetDebugVisualizerCamera(
            cameraDistance=2.0,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.5]
        )
    else:
        physics_client = p.connect(p.DIRECT)
    
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    # Load robot
    try:
        robot_id = p.loadURDF(
            urdf_path,
            useFixedBase=True,
            basePosition=robot_base_pos,
            baseOrientation=robot_base_quat,
            flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
        )
        print(f"Loaded robot from: {urdf_path}")
    except Exception as ex:
        print(f"Error loading URDF: {ex}")
        p.disconnect()
        return 1
    
    # Find end effector
    num_joints = p.getNumJoints(robot_id)
    end_effector_idx = find_end_effector(robot_id)
    
    if end_effector_idx == -1:
        print("Error: Could not find end effector link named 'endeffector'")
        p.disconnect()
        return 1
    
    print(f"End effector link index: {end_effector_idx}")
    
    # Get controllable arm joints
    joint_idxs = get_controllable_arm_joints(robot_id, num_joints)
    print(f"Controllable arm joints: {joint_idxs}")
    
    # Prepare orientation if needed
    target_orientation = None
    if args.with_orientation:
        target_orientation = p.getQuaternionFromEuler(args.euler)
        print(f"Using orientation constraint: Euler={args.euler}, Quat={target_orientation}")
    else:
        print("Using position-only IK (no orientation constraint)")
    
    # Create IK target visualization object
    box_size = 0.03
    box_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[box_size/2]*3,
            rgbaColor=[1, 0, 0, 0.8]
        ),
        basePosition=[0, 0, 0]
    )
    
    # Generate grid points
    print(f"\nGenerating grid from {args.min} to {args.max} with step {args.step}...")
    grid_points = generate_grid_points(args.min, args.max, args.step)
    total_points = len(grid_points)
    print(f"Testing {total_points} points...")
    
    # Test reachability for each point
    reachable_points = []
    
    for i, point in enumerate(grid_points):
        # Update visualization
        p.resetBasePositionAndOrientation(box_id, point, [0, 0, 0, 1])
        
        # Test reachability
        is_reachable = test_reachability(
            robot_id, end_effector_idx, joint_idxs,
            point, target_orientation, args.threshold
        )
        
        if is_reachable:
            reachable_points.append(point)
        
        # Progress indicator
        if (i + 1) % 100 == 0 or (i + 1) == total_points:
            progress = (i + 1) / total_points * 100
            reachable_count = len(reachable_points)
            print(f"Progress: {i+1}/{total_points} ({progress:.1f}%) - "
                  f"Reachable: {reachable_count} ({reachable_count/(i+1)*100:.1f}%)")
    
    # Compute bounding box
    bbox_min, bbox_max = compute_bounding_box(reachable_points)
    
    # Output results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Robot: {selected_key}")
    print(f"Total points tested: {total_points}")
    print(f"Reachable points: {len(reachable_points)} ({len(reachable_points)/total_points*100:.2f}%)")
    
    if bbox_min is not None and bbox_max is not None:
        print(f"\n3D Bounding Box of Reachable Volume:")
        print(f"  Lower bounds: [{bbox_min[0]:.3f}, {bbox_min[1]:.3f}, {bbox_min[2]:.3f}]")
        print(f"  Upper bounds: [{bbox_max[0]:.3f}, {bbox_max[1]:.3f}, {bbox_max[2]:.3f}]")
        print(f"  Volume dimensions: "
              f"[{bbox_max[0]-bbox_min[0]:.3f}, "
              f"{bbox_max[1]-bbox_min[1]:.3f}, "
              f"{bbox_max[2]-bbox_min[2]:.3f}]")
    else:
        print("\nNo reachable points found - cannot compute bounding box.")
    
    print("="*60)
    
    # Generate plot
    if len(reachable_points) > 0:
        print("\nGenerating 3D plot...")
        plot_reachable_volume(reachable_points, bbox_min, bbox_max, selected_key)
    
    # Cleanup
    p.disconnect()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
