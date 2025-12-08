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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import get_robot_dict, get_workspace_dict
import importlib.resources as pkg_resources


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


def calculate_home_joint_angles(robot_id, end_effector_idx, joint_idxs, home_position=[0.0, 0.0, 3]):
    """Calculate joint angles to move end effector to home position.
    
    Args:
        robot_id: PyBullet robot ID
        end_effector_idx: End effector link index
        joint_idxs: List of controllable joint indices
        home_position: Target home position for end effector [x, y, z]
        
    Returns:
        List of joint angles (clipped to joint limits if they exist)
    """
    # Calculate IK for home position
    ik_solution = p.calculateInverseKinematics(
        robot_id, end_effector_idx, home_position
    )
    
    # Clip IK solution to joint limits
    joint_angles = []
    for index, joint_idx in enumerate(joint_idxs):
        if index < len(ik_solution):
            joint_info = p.getJointInfo(robot_id, joint_idx)
            joint_lower_limit = joint_info[8]
            joint_upper_limit = joint_info[9]
            
            ik_value = ik_solution[index]
            
            # Clip to joint limits if they exist
            #if joint_lower_limit < joint_upper_limit:
            #    ik_value = np.clip(ik_value, joint_lower_limit, joint_upper_limit)
            
            joint_angles.append(ik_value)
    
    return joint_angles


def reset_robot_with_joint_angles(robot_id, joint_idxs, joint_angles):
    """Reset robot joints to specified angles by directly setting joint states.
    
    Args:
        robot_id: PyBullet robot ID
        joint_idxs: List of controllable joint indices
        joint_angles: List of target joint angles
    """
    # Directly reset joint states to target angles
    for joint_idx, angle in zip(joint_idxs, joint_angles):
        p.resetJointState(robot_id, joint_idx, angle)


def reset_robot_to_home(robot_id, end_effector_idx, joint_idxs, robot_info):
    """Reset robot to home position using default_joint_ori from robot info.
    
    Args:
        robot_id: PyBullet robot ID
        end_effector_idx: End effector link index
        joint_idxs: List of controllable joint indices
        robot_info: Robot information dictionary containing default_joint_ori
    """
    # Get default joint orientations from robot info
    default_joint_ori = robot_info.get('default_joint_ori', [0.0] * len(joint_idxs))
    
    # Ensure we have enough values for all joints
    if len(default_joint_ori) < len(joint_idxs):
        print(f"Warning: default_joint_ori has {len(default_joint_ori)} values but need {len(joint_idxs)}, padding with zeros")
        default_joint_ori = list(default_joint_ori) + [0.0] * (len(joint_idxs) - len(default_joint_ori))
    
    joint_angles = default_joint_ori[:len(joint_idxs)]
    reset_robot_with_joint_angles(robot_id, joint_idxs, joint_angles)
    return joint_angles


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
    
    # Apply IK solution to joints with joint limit checking
    joints_within_limits = True
    target_positions = []
    
    for index, joint_idx in enumerate(joint_idxs):
        if index < len(ik_solution):
            joint_info = p.getJointInfo(robot_id, joint_idx)
            joint_lower_limit = joint_info[8]  # Lower joint limit
            joint_upper_limit = joint_info[9]  # Upper joint limit
            
            ik_value = ik_solution[index]
            
            # Check if joint has limits (some joints may have limits set to 0, meaning unlimited)
            if joint_lower_limit < joint_upper_limit:
                # Clip to joint limits
                clipped_value = np.clip(ik_value, joint_lower_limit, joint_upper_limit)
                
                # Track if any joint was outside limits
                if abs(clipped_value - ik_value) > 1e-6:
                    joints_within_limits = False
                
                target_positions.append(clipped_value)
            else:
                # No limits, use IK solution directly
                target_positions.append(ik_value)
    
    # If any joint was outside limits, the IK solution is not valid
    if not joints_within_limits:
        return False
    
    # Set motor commands instead of resetting joint states
    p.setJointMotorControlArray(
        bodyUniqueId=robot_id,
        jointIndices=joint_idxs,
        controlMode=p.POSITION_CONTROL,
        targetPositions=target_positions,
        forces=[50] * len(joint_idxs),
        targetVelocities=[1] * len(joint_idxs)
    )
    
    # Run simulation for 100 steps to let motors reach target
    for _ in range(300):
        p.stepSimulation()
        #time.sleep(0.01)
    
    # Check for collisions with all robot links
    #num_joints = p.getNumJoints(robot_id)
    #contact_points = p.getContactPoints(bodyA=robot_id)
    
    # Filter out self-collisions (contacts where both bodies are the robot)
    #external_contacts = [cp for cp in contact_points if cp[2] != robot_id]
    
    #if len(external_contacts) > 0:
    #    # Collision detected with environment/workspace
    #    return False
    
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


def get_robot_kinematic_tree(robot_id):
    """Extract kinematic tree structure and link positions for non-fixed joints.
    
    Returns:
        links: list of tuples (link_index, link_name, parent_index, position, is_end_effector)
    """
    num_joints = p.getNumJoints(robot_id)
    links = []
    
    # Add base link
    base_pos, _ = p.getBasePositionAndOrientation(robot_id)
    links.append((-1, "base", None, np.array(base_pos), False))
    
    # Add all non-fixed joints
    for joint_idx in range(num_joints):
        joint_info = p.getJointInfo(robot_id, joint_idx)
        joint_type = joint_info[2]
        link_name = joint_info[12].decode("utf-8")
        parent_idx = joint_info[16]
        
        # Only include non-fixed joints
        if joint_type != p.JOINT_FIXED:
            link_state = p.getLinkState(robot_id, joint_idx)
            link_pos = np.array(link_state[0])
            is_end_effector = (link_name == 'endeffector')
            
            links.append((joint_idx, link_name, parent_idx, link_pos, is_end_effector))
    
    return links


def plot_reachable_volume(reachable_points, bounding_box_min, bounding_box_max, robot_name, tested_min=None, tested_max=None, robot_links=None):
    """Create 3D plot of reachable volume and bounding box.
    
    Args:
        reachable_points: list/array of reachable points
        bounding_box_min: min coords of reachable points
        bounding_box_max: max coords of reachable points
        robot_name: name for title and output file
        tested_min: min coords of tested limits (optional)
        tested_max: max coords of tested limits (optional)
        robot_links: kinematic tree links from get_robot_kinematic_tree (optional)
    """
    if len(reachable_points) == 0:
        print("No reachable points to plot.")
        return
    
    reachable_points = np.array(reachable_points)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Compute and plot convex hull first (so it appears behind)
    if len(reachable_points) >= 4:  # Need at least 4 points for 3D convex hull
        try:
            hull = ConvexHull(reachable_points)
            
            # Create mesh from convex hull
            faces = []
            for simplex in hull.simplices:
                faces.append([reachable_points[simplex[0]], 
                             reachable_points[simplex[1]], 
                             reachable_points[simplex[2]]])
            
            # Plot convex hull as transparent volume
            poly3d = Poly3DCollection(faces, alpha=0.3, facecolor='cyan', 
                                     edgecolor='none', label='Reachable volume (convex hull)')
            ax.add_collection3d(poly3d)
            
            # Calculate and print convex hull volume
            hull_volume = hull.volume
            print(f"Convex hull volume: {hull_volume:.6f} cubic units")
            
        except Exception as e:
            print(f"Warning: Could not compute convex hull: {e}")
    else:
        print("Not enough points to compute convex hull (need at least 4)")
    
    # Plot tested limits bounding box (blue)
    if tested_min is not None and tested_max is not None:
        r = [tested_min, tested_max]
        first_tested_edge = True
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
            ax.plot3D(*zip(s, e), color='blue', linewidth=2, alpha=0.5, 
                     label='Tested limits' if first_tested_edge else None)
            first_tested_edge = False
    
    # Plot reachable bounding box (green)
    if bounding_box_min is not None and bounding_box_max is not None:
        r = [bounding_box_min, bounding_box_max]
        first_edge = True
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
            ax.plot3D(*zip(s, e), color='green', linewidth=2, alpha=0.7,
                     label='Reachable bbox' if first_edge else None)
            first_edge = False
    
    # Plot reachable points
    ax.scatter(reachable_points[:, 0], reachable_points[:, 1], reachable_points[:, 2],
               c='blue', marker='o', s=10, alpha=0.3, label='Reachable points')
    
    # Plot coordinate axes at origin (0,0,0) - without labels
    axis_length = 0.5
    ax.quiver(0, 0, 0, axis_length, 0, 0, color='red', linewidth=3, arrow_length_ratio=0.2)
    ax.quiver(0, 0, 0, 0, axis_length, 0, color='green', linewidth=3, arrow_length_ratio=0.2)
    ax.quiver(0, 0, 0, 0, 0, axis_length, color='blue', linewidth=3, arrow_length_ratio=0.2)
    
    # Plot robot kinematic tree (initial position) - plot AFTER volume so it's on top with correct colors
    if robot_links is not None:
        first_link = True
        for link_idx, link_name, parent_idx, link_pos, is_end_effector in robot_links:
            # Draw connection to parent first (lines behind joints)
            if parent_idx is not None and parent_idx >= -1:
                # Find parent position
                parent_pos = None
                for p_idx, p_name, _, p_pos, _ in robot_links:
                    if p_idx == parent_idx:
                        parent_pos = p_pos
                        break
                
                if parent_pos is not None:
                    # Draw line from parent to child
                    ax.plot3D([parent_pos[0], link_pos[0]], 
                             [parent_pos[1], link_pos[1]], 
                             [parent_pos[2], link_pos[2]], 
                             color='red', linewidth=3, alpha=1.0,
                             label='Robot kinematic tree' if first_link else None, zorder=10)
                    first_link = False
        
        # Draw joints on top of lines
        for link_idx, link_name, parent_idx, link_pos, is_end_effector in robot_links:
            if link_idx == -1:
                # Base link - larger red dot
                ax.scatter([link_pos[0]], [link_pos[1]], [link_pos[2]], 
                          c='red', marker='o', s=150, alpha=1.0, edgecolors='darkred', 
                          linewidths=2, zorder=11)
            elif is_end_effector:
                # End effector - big red X
                ax.scatter([link_pos[0]], [link_pos[1]], [link_pos[2]], 
                          c='red', marker='x', s=500, alpha=1.0, linewidths=4, zorder=11)
            else:
                # Regular joint - small red sphere
                ax.scatter([link_pos[0]], [link_pos[1]], [link_pos[2]], 
                          c='red', marker='o', s=80, alpha=1.0, edgecolors='darkred',
                          linewidths=1, zorder=11)
    
    # Add text annotations for bounding box limits
    if bounding_box_min is not None and bounding_box_max is not None:
        bbox_text = f"Reachable bbox:\nMin: [{bounding_box_min[0]:.2f}, {bounding_box_min[1]:.2f}, {bounding_box_min[2]:.2f}]\n"
        bbox_text += f"Max: [{bounding_box_max[0]:.2f}, {bounding_box_max[1]:.2f}, {bounding_box_max[2]:.2f}]"
        ax.text2D(0.02, 0.98, bbox_text, transform=ax.transAxes, 
                 fontsize=9, verticalalignment='top', 
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Set camera view angle
    ax.view_init(elev=30, azim=30)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Reachable Volume for {robot_name}')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'./unittest/reachability_{robot_name}.png', dpi=150)
    print(f"Plot saved to ./unittest/reachability_{robot_name}.png")
    plt.close()


def create_bbox_visual(bbox_min, bbox_max, rgba_color):
    """Create visual representation of a bounding box using line segments.
    
    Args:
        bbox_min: Minimum coordinates [x, y, z]
        bbox_max: Maximum coordinates [x, y, z]
        rgba_color: Color as [r, g, b, a]
        
    Returns:
        list of visual shape IDs
    """
    edges = [
        # Bottom face
        ([bbox_min[0], bbox_min[1], bbox_min[2]], [bbox_max[0], bbox_min[1], bbox_min[2]]),
        ([bbox_max[0], bbox_min[1], bbox_min[2]], [bbox_max[0], bbox_max[1], bbox_min[2]]),
        ([bbox_max[0], bbox_max[1], bbox_min[2]], [bbox_min[0], bbox_max[1], bbox_min[2]]),
        ([bbox_min[0], bbox_max[1], bbox_min[2]], [bbox_min[0], bbox_min[1], bbox_min[2]]),
        # Top face
        ([bbox_min[0], bbox_min[1], bbox_max[2]], [bbox_max[0], bbox_min[1], bbox_max[2]]),
        ([bbox_max[0], bbox_min[1], bbox_max[2]], [bbox_max[0], bbox_max[1], bbox_max[2]]),
        ([bbox_max[0], bbox_max[1], bbox_max[2]], [bbox_min[0], bbox_max[1], bbox_max[2]]),
        ([bbox_min[0], bbox_max[1], bbox_max[2]], [bbox_min[0], bbox_min[1], bbox_max[2]]),
        # Vertical edges
        ([bbox_min[0], bbox_min[1], bbox_min[2]], [bbox_min[0], bbox_min[1], bbox_max[2]]),
        ([bbox_max[0], bbox_min[1], bbox_min[2]], [bbox_max[0], bbox_min[1], bbox_max[2]]),
        ([bbox_max[0], bbox_max[1], bbox_min[2]], [bbox_max[0], bbox_max[1], bbox_max[2]]),
        ([bbox_min[0], bbox_max[1], bbox_min[2]], [bbox_min[0], bbox_max[1], bbox_max[2]]),
    ]
    
    line_ids = []
    for start, end in edges:
        line_id = p.addUserDebugLine(start, end, lineColorRGB=rgba_color[:3], lineWidth=3)
        line_ids.append(line_id)
    
    return line_ids


def visualize_in_pybullet(robot_id, reachable_points, bbox_min, bbox_max):
    """Visualize reachable volume in PyBullet with transparent meshes.
    
    Args:
        robot_id: PyBullet robot ID
        reachable_points: Array of reachable points
        bbox_min: Minimum coordinates of bounding box
        bbox_max: Maximum coordinates of bounding box
    """
    print("\nVisualizing in PyBullet...")
    print("Press 'q' in terminal to quit visualization")
    
    # Create transparent bounding box (green)
    if bbox_min is not None and bbox_max is not None:
        bbox_lines = create_bbox_visual(bbox_min, bbox_max, [0, 1, 0, 0.7])
        print("Created green bounding box visualization")
    
    # Create convex hull visualization (blue)
    hull_visual_id = None
    if len(reachable_points) >= 4:
        try:
            hull = ConvexHull(reachable_points)
            
            # Create mesh from convex hull vertices and simplices
            vertices = reachable_points[hull.vertices]
            
            # Calculate center of hull for positioning
            center = np.mean(vertices, axis=0)
            
            # Create vertices relative to center
            relative_vertices = vertices - center
            
            # Create mesh indices (triangles)
            indices = hull.simplices.flatten().tolist()
            
            # Create collision and visual shapes
            collision_shape = p.createCollisionShape(
                shapeType=p.GEOM_MESH,
                vertices=relative_vertices.tolist(),
                indices=indices
            )
            
            visual_shape = p.createVisualShape(
                shapeType=p.GEOM_MESH,
                vertices=relative_vertices.tolist(),
                indices=indices,
                rgbaColor=[0, 0, 1, 0.3]  # Transparent blue
            )
            
            # Create multibody with the mesh
            hull_visual_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=center.tolist()
            )
            
            print("Created blue convex hull visualization")
            
        except Exception as e:
            print(f"Warning: Could not create convex hull visualization: {e}")
    
    # Visualize reachable points as small spheres
    point_visual_ids = []
    sphere_radius = 0.01
    for i, point in enumerate(reachable_points[::10]):  # Sample every 10th point to avoid overload
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=sphere_radius,
            rgbaColor=[0, 0, 1, 0.3]
        )
        point_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual_shape,
            basePosition=point.tolist()
        )
        point_visual_ids.append(point_id)
    
    print(f"Created {len(point_visual_ids)} point visualizations (sampled)")
    
    # Keep visualization open until user quits
    print("\nVisualization active. Press Ctrl+C or close the PyBullet window to continue...")
    try:
        while True:
            p.stepSimulation()
            import time
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nVisualization closed by user")


def test_robot_reachability(robot_key, r_dict, args):
    """Run reachability test for a single robot.
    
    Args:
        robot_key: Key identifying the robot in r_dict
        r_dict: Robot dictionary
        args: Command line arguments
        
    Returns:
        0 on success, 1 on error
    """
    print(f"\nTesting robot: {robot_key}")
    
    # Get robot info
    robot_info = r_dict[robot_key]
    rel_path = robot_info['path'].lstrip('/')
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    urdf_path = os.path.join(base_dir, rel_path)
    
    if not os.path.exists(urdf_path):
        print(f"Error: URDF file not found: {urdf_path}")
        return 1
    
    # Get workspace dictionary and determine which workspace to use
    ws_dict = get_workspace_dict()
    workspace_key = "table_complex"  # default
    
    # Check if robot name is partially in any workspace key
    #for ws_key in ws_dict.keys():
    #    if robot_key in ws_key:
    #        workspace_key = ws_key
    #        print(f"Using workspace: {workspace_key} (matched with robot name)")
    #        break
    #
    #if workspace_key == "table":
    #    print(f"Using default workspace: table")
    
    workspace_info = ws_dict[workspace_key]
    
    # Get robot position from workspace or use robot_info position

    robot_base_pos = list(np.array(robot_info.get('position', [0.0, 0.0, 0.0])).astype(float))
    robot_base_orientation = robot_info.get('orientation', [0.0, 0.0, 0.0])
    
    robot_base_quat = p.getQuaternionFromEuler(robot_base_orientation)
    
    # Initialize PyBullet
    if args.gui:
        physics_client = p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(
            cameraDistance=2.0,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.0]
        )
    else:
        physics_client = p.connect(p.DIRECT)
    
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    # Load scene similar to _setup_scene in gym_env
    currentdir = os.path.join(pkg_resources.files("myGym"), "envs")
    
    
    
    # Load workspace
    workspace_urdf_path = os.path.join(currentdir, "rooms/collision/" + workspace_info['urdf'])
    if os.path.exists(workspace_urdf_path):
        transform = workspace_info['transform']
        workspace_id = p.loadURDF(
            workspace_urdf_path,
            transform['position'],
            p.getQuaternionFromEuler(transform['orientation']),
            useFixedBase=True
        )
        print(f"Loaded workspace from: {workspace_urdf_path}")
        print(f"Workspace position: {transform['position']}")
        print(f"Workspace orientation (euler): {transform['orientation']}")
    else:
        print(f"Warning: Workspace URDF not found: {workspace_urdf_path}")
    
    # Load floor
    floor_path = os.path.join(currentdir, "rooms/plane.urdf")
    floor_id = p.loadURDF(floor_path, transform['position'], p.getQuaternionFromEuler(transform['orientation']), useFixedBase=True)
    print(f"Loaded floor from: {floor_path}")
    print(f"Floor position: {transform['position']}")
    print(f"Floor orientation (euler): {transform['orientation']}") 

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
        print(f"Robot position: {robot_base_pos}")
        print(f"Robot orientation (euler): {robot_base_orientation}")
    except Exception as ex:
        print(f"Error loading URDF: {ex}")
        p.disconnect()
        return 1
    
    # Get controllable arm joints
    num_joints = p.getNumJoints(robot_id)
    joint_idxs = get_controllable_arm_joints(robot_id, num_joints)
    print(f"Controllable arm joints: {joint_idxs}")
    
    # Store original joint angles from URDF
    original_joint_angles = []
    for joint_idx in joint_idxs:
        joint_state = p.getJointState(robot_id, joint_idx)
        original_joint_angles.append(joint_state[0])  # joint_state[0] is the position
    print(f"Original joint angles from URDF: {original_joint_angles}")
    
    # Get initial robot kinematic tree for visualization
    initial_robot_links = get_robot_kinematic_tree(robot_id)
    
    # Find end effector
    end_effector_idx = find_end_effector(robot_id)
    
    if end_effector_idx == -1:
        print("Error: Could not find end effector link named 'endeffector'")
        p.disconnect()
        return 1
    
    print(f"End effector link index: {end_effector_idx}")
    
    # Reset robot to home position
    print(f"Resetting robot to home position using default_joint_ori")
    init_joint_angles = reset_robot_to_home(robot_id, end_effector_idx, joint_idxs, robot_info)
    print(f"Home joint angles: {init_joint_angles}")
    
    # Prepare orientation if needed
    target_orientation = None
    if args.with_orientation:
        target_orientation = p.getQuaternionFromEuler(args.euler)
        print(f"Using orientation constraint: Euler={args.euler}, Quat={target_orientation}")
    else:
        print("Using position-only IK (no orientation constraint)")
    
    # Create IK target visualization object
    box_size = 0.02
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
    
    # Visual marker lists (only used in GUI mode)
    reachable_visual_ids = []
    unreachable_visual_ids = []
    
    for i, point in enumerate(grid_points):
        # Update visualization
        p.resetBasePositionAndOrientation(box_id, point, [0, 0, 0, 1])
        
        # Test reachability
        is_reachable = test_reachability(
            robot_id, end_effector_idx, joint_idxs,
            point, target_orientation, args.threshold
        )
        
        # Reset robot to home position after each test
        reset_robot_with_joint_angles(robot_id, joint_idxs, init_joint_angles)
        if is_reachable:
            reachable_points.append(point)
            
            # Create green transparent sphere for reachable point (GUI mode only)
            if args.gui:
                visual_shape = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[0.1/2]*3,
                    rgbaColor=[0, 1, 0, 0.2]
                )
                marker_id = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=-1,
                    baseVisualShapeIndex=visual_shape,
                    basePosition=point.tolist()
                )
                reachable_visual_ids.append(marker_id)
        else:
            # Create red transparent dot for unreachable point (GUI mode only)
            if args.gui:
                visual_shape = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[0.005/2]*3,
                    rgbaColor=[1, 0, 0, 0.3]
                )
                marker_id = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=-1,
                    baseVisualShapeIndex=visual_shape,
                    basePosition=point.tolist()
                )
                unreachable_visual_ids.append(marker_id)
        
        # Progress indicator
        if (i + 1) % 100 == 0 or (i + 1) == total_points:
            progress = (i + 1) / total_points * 100
            reachable_count = len(reachable_points)
            print(f"Progress: {i+1}/{total_points} ({progress:.1f}%) - "
                  f"Reachable: {reachable_count} ({reachable_count/(i+1)*100:.1f}%)")
    
    # Compute bounding box
    bbox_min, bbox_max = compute_bounding_box(reachable_points)
    
    # Get robot kinematic tree for visualization
    robot_links = get_robot_kinematic_tree(robot_id)
    
    # Output results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Robot: {robot_key}")
    print(f"Workspace: {workspace_key}")
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
    
    if args.gui:
        print(f"\nGUI visualization markers created:")
        print(f"  Green spheres (reachable): {len(reachable_visual_ids)}")
        print(f"  Red dots (unreachable): {len(unreachable_visual_ids)}")
    
    print("="*60)
    
    # Generate plot
    if len(reachable_points) > 0:
        print("\nGenerating 3D plot...")
        plot_reachable_volume(reachable_points, bbox_min, bbox_max, robot_key, args.min, args.max, initial_robot_links)
    
    # PyBullet visualization if requested
    if args.visualize_pybullet and len(reachable_points) > 0:
        # Reconnect to GUI if not already in GUI mode
        if not args.gui:
            p.disconnect()
            physics_client = p.connect(p.GUI)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.81)
            
            # Reload scene
            floor_id = p.loadURDF(floor_path, useFixedBase=True)
            if os.path.exists(workspace_urdf_path):
                workspace_id = p.loadURDF(
                    workspace_urdf_path,
                    transform['position'],
                    p.getQuaternionFromEuler(transform['orientation']),
                    useFixedBase=True
                )
            
            # Reload robot
            robot_id = p.loadURDF(
                urdf_path,
                useFixedBase=True,
                basePosition=robot_base_pos,
                baseOrientation=robot_base_quat,
                flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
            )
            
            p.resetDebugVisualizerCamera(
                cameraDistance=2.0,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=[0, 0, 0.5]
            )
        
        visualize_in_pybullet(robot_id, np.array(reachable_points), bbox_min, bbox_max)
    
    # Cleanup
    p.disconnect()
    
    return 0


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
        default=0.2,
        help="Grid step size (default: 0.1)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Distance threshold for reachability (default: 0.05)"
    )
    parser.add_argument(
        "--visualize-pybullet",
        action="store_true",
        help="Visualize reachable volume in PyBullet after test (default: off)"
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
        print("[0] all")
        for i, robot_key in enumerate(robots):
            print(f"[{i+1}] {robot_key}")
        
        while True:
            sel = input("Select robot index (q to quit): ").strip().lower()
            if sel in ("q", "quit", "exit"):
                return 0
            if sel.isdigit():
                idx = int(sel)
                if idx == 0:
                    # Run for all robots
                    print("\nRunning reachability test for all robots...")
                    for robot_key in robots:
                        result = test_robot_reachability(robot_key, r_dict, args)
                        if result != 0:
                            print(f"Warning: Test failed for robot {robot_key}")
                    print("\nAll tests completed!")
                    return 0
                elif 1 <= idx <= len(robots):
                    selected_key = robots[idx - 1]
                    break
            print("Invalid selection.")
    
    print(f"\nSelected robot: {selected_key}")
    
    # Run test for single robot
    return test_robot_reachability(selected_key, r_dict, args)


if __name__ == "__main__":
    sys.exit(main())