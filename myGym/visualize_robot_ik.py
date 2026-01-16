import pybullet as p
import pybullet_data
import argparse
import time
import numpy as np
from numpy import rad2deg, deg2rad, set_printoptions, array, linalg, round, any, mean
from myGym.utils.helpers import get_robot_dict
from myGym.utils.helpers import get_workspace_dict
import importlib.resources as pkg_resources
import os

def save_robot_dict_to_helpers(rd, helpers_path):
    """Save updated robot dictionary back to helpers.py file."""
    # Read the current file
    with open(helpers_path, 'r') as f:
        lines = f.readlines()
    
    # Find the start and end of get_robot_dict function
    start_idx = None
    end_idx = None
    brace_count = 0
    in_r_dict = False
    
    for i, line in enumerate(lines):
        if 'def get_robot_dict():' in line:
            start_idx = i
        if start_idx is not None and 'r_dict =' in line and '{' in line:
            in_r_dict = True
            brace_count = line.count('{') - line.count('}')
        elif in_r_dict:
            brace_count += line.count('{') - line.count('}')
            if brace_count == 0:
                end_idx = i
                break
    
    if start_idx is None or end_idx is None:
        print("Error: Could not find r_dict in helpers.py")
        return False
    
    # Generate new r_dict string
    new_r_dict_lines = ["    r_dict =   {"]
    
    # Define the order of keys for consistent output
    key_order = ['path', 'position', 'orientation', 'default_joint_ori', 'ee_pos', 'ee_ori', 'ee_quat_ori']
    
    for key, value in rd.items():
        parts = []
        
        # Sort keys according to defined order, then alphabetically for any remaining
        sorted_keys = sorted(value.keys(), key=lambda k: (key_order.index(k) if k in key_order else len(key_order), k))
        
        for k in sorted_keys:
            v = value[k]
            if k == 'position':
                parts.append(f"'{k}': np.array({list(v)})")
            elif k in ['default_joint_ori', 'ee_pos', 'ee_ori', 'ee_quat_ori']:
                parts.append(f"'{k}': {v}")
            elif k == 'orientation':
                # Format orientation with np.pi if present
                formatted_v = str(v)
                if 1.5707 in v or abs(v[2] - 1.5707963267948966) < 0.0001:
                    # Replace with np.pi/2
                    formatted_list = [v[0], v[1], 'np.pi/2' if abs(v[2] - 1.5707963267948966) < 0.0001 else v[2]]
                    formatted_v = str(formatted_list).replace("'np.pi/2'", "np.pi/2").replace('"np.pi/2"', 'np.pi/2')
                elif 3.14159 in v or abs(v[2] - np.pi) < 0.0001:
                    formatted_list = [v[0], v[1], 'np.pi' if abs(v[2] - np.pi) < 0.0001 else v[2]]
                    formatted_v = str(formatted_list).replace("'np.pi'", "np.pi").replace('"np.pi"', 'np.pi')
                else:
                    # Check for multiplies of pi
                    for mult in [0.5, 0.35, 0.4, 0.2, 1.0]:
                        if abs(v[2] - mult * np.pi) < 0.0001:
                            formatted_list = [v[0], v[1], f'{mult}*np.pi']
                            formatted_v = str(formatted_list).replace(f"'{mult}*np.pi'", f"{mult}*np.pi").replace(f'"{mult}*np.pi"', f'{mult}*np.pi')
                            break
                parts.append(f"'{k}': {formatted_v}")
            else:
                parts.append(f"'{k}': {repr(v)}")
        
        line_str = f"'{key}': {{" + ", ".join(parts) + "}"
        new_r_dict_lines.append("                             " + line_str + ",")
    
    new_r_dict_lines.append("                             }")
    
    # Replace the old r_dict with new one
    new_lines = lines[:start_idx+1] + [line + '\n' for line in new_r_dict_lines] + lines[end_idx+1:]
    
    # Write back to file
    with open(helpers_path, 'w') as f:
        f.writelines(new_lines)
    
    return True

def apply_ik_solution(robot_id, ik_solution, joint_idxs):
    """Apply the IK solution to the robot joints."""
    joint_values = []
    for index, joint_idx in enumerate(joint_idxs):
        joint_pos = ik_solution[index]
        p.setJointMotorControl2(
            bodyIndex=robot_id,
            jointIndex=joint_idx,
            controlMode=p.POSITION_CONTROL,
            targetPosition=joint_pos,
            force= 500
        )
        joint_values.append(f"{p.getJointInfo(robot_id, joint_idx)[1].decode('utf-8')}: {joint_pos:.4f} rad ({joint_pos*57.2958:.2f}°)")
    return joint_values

def find_gripper_joints(robot_id):
    """Find gripper joints and return (indices, lower_limits, upper_limits).

    Prints limits; if invalid, substitutes +/- pi in returned lists."""
    num_joints = p.getNumJoints(robot_id)
    gripper_idxs: list[int] = []
    lower_limits: list[float] = []
    upper_limits: list[float] = []
    print("\n--- Gripper Joint Scan ---")
    for joint_idx in range(num_joints):
        joint_info = p.getJointInfo(robot_id, joint_idx)
        joint_name = joint_info[1].decode("utf-8")
        if 'gjoint' in joint_name:
            lower = float(joint_info[8])
            upper = float(joint_info[9])
            if lower >= upper:  # invalid / fixed
                print(f"  {joint_name} (idx {joint_idx}) limits invalid ({lower:.4f}, {upper:.4f}) -> using [-pi, pi]")
                lower, upper = -np.pi, np.pi
            else:
                print(f"  {joint_name} (idx {joint_idx}) limits: [{lower:.4f}, {upper:.4f}]")
            gripper_idxs.append(joint_idx)
            lower_limits.append(lower)
            upper_limits.append(upper)
    if not gripper_idxs:
        print("  No gripper joints found matching pattern 'gjoint'.")
    else:
        print(f"Found gripper joints: {gripper_idxs}")
    print("---------------------------\n")
    return gripper_idxs, lower_limits, upper_limits

def get_controllable_arm_joints(robot_id, num_joints):
    """Get joint information for arm control (non-fixed joints), excluding gripper joints."""
    joint_idxs = []
    joint_names = []
    print("\n--- Controllable Arm Joints ---")
    for joint_idx in range(num_joints):
        joint_info = p.getJointInfo(robot_id, joint_idx)
        joint_name = joint_info[1].decode("utf-8")
        joint_type = joint_info[2]
        
        # Print all joint names and IDs for inspection
        print(f"  Joint ID: {joint_idx}, Name: {joint_name}, Type: {joint_type}")

        if joint_type != p.JOINT_FIXED:
            # Exclude gripper joints from the main IK control list
            if 'right_' not in joint_name or 'finger' not in joint_name: 
                lower = joint_info[8]
                upper = joint_info[9]
                
                if lower >= upper:
                    # Handle cases with no limits or invalid limits
                    print(f"    Warning: Joint {joint_idx} ('{joint_name}') has invalid limits ({lower}, {upper}). Defaulting to +/- pi.")
                    lower, upper = -np.pi, np.pi # Default to reasonable limits if needed
                    
                joint_idxs.append(joint_idx)
                joint_names.append(joint_name)
                print(f"    -> Selected for IK control.")
            else:
                print(f"    -> Skipped (Gripper Joint).")
        else:
             print(f"    -> Skipped (Fixed Joint).")

    print(f"Selected Arm Joint Indices for IK: {joint_idxs}")
    print("-------------------------------\n")
    return joint_idxs, joint_names


def main():
    parser = argparse.ArgumentParser(description="Nico Robot Grasping Control")
    parser.add_argument("--urdf", type=str, default="./envs/robots/nico/nico_grasper.urdf", help="Path to the robot URDF file.")
    parser.add_argument("--interactive", action="store_true", default=1, help="Interactive selection of robot from registry (overrides --urdf)")
    parser.add_argument("--robot-key", type=str, help="Optional robot key from r_dict to use pose from if not using interactive picker.")
    parser.add_argument("--workspace", action="store_true", help="Workspace mode: select workspace then robot; loads both.")
    args = parser.parse_args()

    selected_key = None
    robot_base_pos = [0.0, 0.0, 0.0]
    robot_base_quat = [0, 0, 0, 1]
    rdict = get_robot_dict()
    selected_workspace = None

    # Workspace mode: first choose workspace then robot
    if args.workspace:
        ws_dict = get_workspace_dict()
        ws_keys = sorted(ws_dict.keys())
        if not ws_keys:
            print("No workspaces available.")
            return
        print("Available workspaces:")
        for i, wk in enumerate(ws_keys, 1):
            print(f"[{i}] {wk}")
        while True:
            wsel = input("Select workspace index (q to quit): ").strip().lower()
            if wsel in ("q", "quit", "exit"):
                return
            if wsel.isdigit():
                widx = int(wsel)
                if 1 <= widx <= len(ws_keys):
                    selected_workspace = ws_keys[widx - 1]
                    break
            print("Invalid selection.")
        ws_info = ws_dict[selected_workspace]
        # Workspace params
        ws_urdf_name = ws_info['urdf']
        base_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_urdf_path = os.path.join(base_dir, 'envs', 'rooms', 'collision', ws_urdf_name)
        ws_transform = ws_info.get('transform', {})
        ws_pos = ws_transform.get('position', [0, 0, 0])
        ws_eul = ws_transform.get('orientation', [0, 0, 0])
        ws_quat = p.getQuaternionFromEuler(ws_eul)
        ws_robot = ws_info.get('robot', {})
        ws_robot_pos = ws_robot.get('position')
        ws_robot_eul = ws_robot.get('orientation')

        # Robot selection after workspace
        robots = sorted(rdict.keys())
        print("Available robots:")
        for i, rk in enumerate(robots, 1):
            print(f"[{i}] {rk}")
        while True:
            rsel = input("Select robot index (q to quit): ").strip().lower()
            if rsel in ("q", "quit", "exit"):
                return
            if rsel.isdigit():
                ridx = int(rsel)
                if 1 <= ridx <= len(robots):
                    selected_key = robots[ridx - 1]
                    rinfo = rdict[selected_key]
                    rel_path = rinfo['path'].lstrip('/')
                    candidate = os.path.join(base_dir, rel_path)
                    args.urdf = candidate if os.path.exists(candidate) else rel_path
                    # Set robot base pose: workspace override > r_dict
                    if ws_robot_pos is not None:
                        robot_base_pos = list(np.array(ws_robot_pos).astype(float))
                    elif 'position' in rinfo:
                        robot_base_pos = list(np.array(rinfo['position']).astype(float))
                    if ws_robot_eul is not None:
                        robot_base_quat = p.getQuaternionFromEuler(ws_robot_eul)
                    elif 'orientation' in rinfo:
                        robot_base_quat = p.getQuaternionFromEuler(rinfo['orientation'])
                    break
            print("Invalid selection.")

    # Robot-only interactive mode (no workspace)
    if (not args.workspace) and args.interactive:
        robots = sorted(rdict.keys())
        if not robots:
            print("No robots available.")
            return
        print("Available robots:")
        for i, rk in enumerate(robots, 1):
            print(f"[{i}] {rk}")
        while True:
            sel = input("Select robot index (q to quit): ").strip().lower()
            if sel in ("q", "quit", "exit"):
                return
            if sel.isdigit():
                idx = int(sel)
                if 1 <= idx <= len(robots):
                    selected_key = robots[idx - 1]
                    info = rdict[selected_key]
                    rel_path = info['path'].lstrip('/')
                    base_dir = os.path.dirname(os.path.abspath(__file__))
                    candidate = os.path.join(base_dir, rel_path)
                    args.urdf = candidate if os.path.exists(candidate) else rel_path
                    if 'position' in info:
                        robot_base_pos = list(np.array(info['position']).astype(float))
                    if 'orientation' in info:
                        robot_base_quat = p.getQuaternionFromEuler(info['orientation'])
                    print(f"Selected robot '{selected_key}' -> {args.urdf}")
                    break
            print("Invalid selection.")
    elif not args.workspace:  # Non-interactive, no workspace
        if args.robot_key and args.robot_key in rdict:
            selected_key = args.robot_key
        else:
            # Try to infer by path ending
            for k, v in rdict.items():
                rel = v.get('path', '').lstrip('/')
                if rel and (args.urdf.endswith(rel) or os.path.basename(args.urdf) == os.path.basename(rel)):
                    selected_key = k
                    break
        if selected_key:
            info = rdict[selected_key]
            if 'position' in info:
                robot_base_pos = list(np.array(info['position']).astype(float))
            if 'orientation' in info:
                robot_base_quat = p.getQuaternionFromEuler(info['orientation'])
            print(f"Using robot key '{selected_key}' pose: pos={robot_base_pos}, euler={info.get('orientation', [0,0,0])}")

    # Initialize PyBullet
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # Load workspace and plane (unless already loaded in workspace mode)
    if not args.workspace:
        ws_dict = get_workspace_dict()
        workspace_key = "table_complex"
        workspace_info = ws_dict[workspace_key]
        
        # Load scene
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
        else:
            print(f"Warning: Workspace URDF not found: {workspace_urdf_path}")
        
        # Load floor
        floor_path = os.path.join(currentdir, "rooms/plane_universal.urdf")
        if os.path.exists(floor_path):
            floor_id = p.loadURDF(floor_path, transform['position'], 
                                 p.getQuaternionFromEuler(transform['orientation']), 
                                 useFixedBase=True)
            print(f"Loaded floor from: {floor_path}")

    # Load workspace first if requested
    if args.workspace and selected_workspace is not None:
        try:
            w_id = p.loadURDF(workspace_urdf_path, basePosition=ws_pos, baseOrientation=ws_quat, useFixedBase=True)
            print(f"Loaded workspace '{selected_workspace}' from {workspace_urdf_path}")
        except Exception as ex:
            print(f"Error loading workspace '{selected_workspace}': {ex}")

    # Load robot
    try:
        robot_id = p.loadURDF(args.urdf, useFixedBase=True, basePosition=robot_base_pos, baseOrientation=robot_base_quat, flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)
    except Exception as ex:
        print(f"Error: Failed to load URDF file '{args.urdf}': {ex}")
        return

    # Reset joint positions to default_joint_ori if available
    if selected_key and selected_key in rdict:
        robot_info = rdict[selected_key]
        if 'default_joint_ori' in robot_info:
            init_pos = robot_info['default_joint_ori']
            print(f"Resetting joints to default_joint_ori: {init_pos}")
            
            # Get list of non-fixed joints
            non_fixed_joints = []
            num_joints = p.getNumJoints(robot_id)
            for joint_idx in range(num_joints):
                joint_info = p.getJointInfo(robot_id, joint_idx)
                joint_type = joint_info[2]
                if joint_type != p.JOINT_FIXED:
                    non_fixed_joints.append(joint_idx)
            
            # Reset joint states
            for i, joint_idx in enumerate(non_fixed_joints):
                if i < len(init_pos):
                    p.resetJointState(robot_id, joint_idx, init_pos[i])
                    print(f"  Joint {joint_idx}: {init_pos[i]}")

    # Find end effector link
    num_joints = p.getNumJoints(robot_id)
    end_effector_index = -1
    for joint_idx in range(num_joints):
        joint_info = p.getJointInfo(robot_id, joint_idx)
        link_name = joint_info[12].decode("utf-8")
        if link_name == 'endeffector':
            end_effector_index = joint_idx

    
    if end_effector_index == -1:
        print("Error: Could not find end effector link in URDF")
        return

    # Get joint information for arm control using the new function
    joint_idxs, joint_names = get_controllable_arm_joints(robot_id, num_joints)

    # Find gripper joint indices
    gripper_idxs, gripper_low_limits, gripper_up_limits = find_gripper_joints(robot_id)

    # Get initial end effector position and orientation
    link_state = p.getLinkState(robot_id, end_effector_index)
    ee_initial_pos = link_state[0]
    ee_initial_quat = link_state[1]
    ee_initial_euler = p.getEulerFromQuaternion(ee_initial_quat)
    
    print(f"Initial end effector position: {ee_initial_pos}")
    print(f"Initial end effector orientation (euler): {ee_initial_euler}")

    # Get target position and orientation from r_dict if available
    target_initial_pos = ee_initial_pos
    target_initial_euler = ee_initial_euler
    
    if selected_key and selected_key in rdict:
        robot_info = rdict[selected_key]
        if 'ee_pos' in robot_info:
            target_initial_pos = robot_info['ee_pos']
            print(f"Using stored ee_pos as target: {target_initial_pos}")
        if 'ee_ori' in robot_info:
            target_initial_euler = robot_info['ee_ori']
            print(f"Using stored ee_ori as target: {target_initial_euler}")
        elif 'ee_quat_ori' in robot_info:
            target_initial_quat = robot_info['ee_quat_ori']
            target_initial_euler = p.getEulerFromQuaternion(target_initial_quat)
            print(f"Using stored ee_quat_ori as target: {target_initial_quat}")

    # Create visual target box
    box_size = 0.02
    box_visual_shape = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[box_size/2, box_size/2, box_size/2],
        rgbaColor=[1, 0, 0, 0.5]  # Red semi-transparent (initial)
    )
    box_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,  # No collision
        baseVisualShapeIndex=box_visual_shape,
        basePosition=target_initial_pos,
        baseOrientation=p.getQuaternionFromEuler(target_initial_euler)
    )
    print(f"Created target box at position: {target_initial_pos}, orientation: {target_initial_euler}")

    # Thresholds for error checking
    POSITION_THRESHOLD = 0.05  # meters
    ORIENTATION_THRESHOLD_QUAT_DOT = 0.99905  # corresponds to ~5 degrees

    # Create sliders for box position control - initialized with target pose
    x_slider = p.addUserDebugParameter("Target X", -1, 1.0, target_initial_pos[0])
    y_slider = p.addUserDebugParameter("Target Y", -1, 1, target_initial_pos[1])
    z_slider = p.addUserDebugParameter("Target Z", -1, 1, target_initial_pos[2])

    # Create additional init sliders for Yaw, Pitch, Roll - initialized with target orientation
    roll_slider = p.addUserDebugParameter("Target Roll", -np.pi, np.pi, target_initial_euler[0])
    pitch_slider = p.addUserDebugParameter("Target Pitch", -np.pi, np.pi, target_initial_euler[1])
    yaw_slider = p.addUserDebugParameter("Target Yaw", -np.pi, np.pi, target_initial_euler[2])

    # Main simulation loop
    p.setRealTimeSimulation(1)
    p.resetDebugVisualizerCamera(
        cameraDistance=3,
        cameraYaw=70,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0.0]
    )
 
    orientation_line_id = None # Initialize desired orientation line ID tracker
    actual_orientation_line_id = None # Initialize actual orientation line ID tracker
    orientation_diff_text_id = None # Initialize debug text ID tracker
    position_diff_text_id = None # Initialize position difference text ID tracker
    
    # Get path to helpers.py
    helpers_path = os.path.join(os.path.dirname(__file__), 'utils', 'helpers.py')
    
    try:
        while True:
            # Check keyboard events first to potentially switch targets
            keys = p.getKeyboardEvents()
            
            # Check if 't' key is pressed to save current configuration
            if ord('t') in keys and keys[ord('t')] & p.KEY_WAS_TRIGGERED:
                if selected_key and selected_key in rdict:
                    # Get current joint values
                    joint_values = []
                    for joint_idx in joint_idxs:
                        joint_state = p.getJointState(robot_id, joint_idx)
                        joint_values.append(round(joint_state[0], 2))
                    print(f"Current joint values: {joint_values}")
                    
                    # Add default_joint_ori to robot dict first
                    rdict[selected_key]['default_joint_ori'] = joint_values
                    
                    # Get end effector pose
                    link_state = p.getLinkState(robot_id, end_effector_index)
                    ee_pos = [round(x, 4) for x in link_state[0]]
                    ee_quat = link_state[1]
                    ee_quat_ori = [round(x, 4) for x in ee_quat]
                    ee_ori = [round(x, 4) for x in p.getEulerFromQuaternion(ee_quat)]
                    print(f"End effector position: {ee_pos}")
                    print(f"End effector orientation (euler): {ee_ori}")
                    print(f"End effector orientation (quaternion): {ee_quat_ori}")
                    
                    # Store ee_pos, ee_ori, and ee_quat_ori in robot dict
                    rdict[selected_key]['ee_pos'] = ee_pos
                    rdict[selected_key]['ee_ori'] = ee_ori
                    rdict[selected_key]['ee_quat_ori'] = ee_quat_ori
                    
                    # Save to helpers.py
                    if save_robot_dict_to_helpers(rdict, helpers_path):
                        print(f"Updated helpers.py with default_joint_ori, ee_pos, ee_ori, and ee_quat_ori for {selected_key}")
                    else:
                        print("Failed to update helpers.py")
                else:
                    print("No robot key selected, cannot save to helpers.py")
            
            # Check if 'c' key is pressed to close the gripper (set gripper joints to 0)
            if ord('c') in keys and keys[ord('c')] & p.KEY_WAS_TRIGGERED:
                # Move gripper joints to their individual lower limits
                print("Closing gripper (moving to lower joint limits)")
                apply_ik_solution(robot_id, gripper_low_limits, gripper_idxs)
                print(gripper_low_limits)
                # grasper.perform_grasp()

            if ord('d') in keys and keys[ord('d')] & p.KEY_WAS_TRIGGERED:
                # Move gripper joints to their individual upper limits
                print("Opening gripper (moving to upper joint limits)")
                apply_ik_solution(robot_id, gripper_up_limits, gripper_idxs)
                print(gripper_up_limits)
                # grasper.perform_drop()
            
            if ord('m') in keys and keys[ord('m')] & p.KEY_WAS_TRIGGERED:
                grasper.move_arm([0.35,-0.4,0.2], args.ori, args.side)
                        

            target_pos = [
                p.readUserDebugParameter(x_slider),
                p.readUserDebugParameter(y_slider),
                p.readUserDebugParameter(z_slider)
            ]
            # target_pos =[0.168, -0.084, 0.221]
            # Use main orientation sliders
            roll = p.readUserDebugParameter(roll_slider)
            pitch = p.readUserDebugParameter(pitch_slider)
            yaw = p.readUserDebugParameter(yaw_slider)
            target_orientation_quat = p.getQuaternionFromEuler([roll, pitch, yaw])
            target_orientation_euler = [roll, pitch, yaw]
            
            # --- Get current state of the end effector AFTER applying IK ---
            link_state = p.getLinkState(robot_id, end_effector_index)
            ee_pos = link_state[0] # World position
            actual_orientation_quat = link_state[1] # Actual world orientation (quaternion)
            actual_orientation_euler = p.getEulerFromQuaternion(actual_orientation_quat)

            # --- Calculate Position and Orientation Errors ---
            # Position error (Euclidean distance)
            pos_diff_vec = np.array(target_pos) - np.array(ee_pos)
            pos_distance = np.linalg.norm(pos_diff_vec)
            
            # Orientation error (quaternion dot product)
            # Normalize quaternions
            target_quat_norm = np.array(target_orientation_quat) / np.linalg.norm(target_orientation_quat)
            actual_quat_norm = np.array(actual_orientation_quat) / np.linalg.norm(actual_orientation_quat)
            quat_dot = abs(np.dot(target_quat_norm, actual_quat_norm))
            
            # Check thresholds
            position_within_threshold = pos_distance <= POSITION_THRESHOLD
            orientation_within_threshold = quat_dot >= ORIENTATION_THRESHOLD_QUAT_DOT
            
            # Determine box color based on errors
            if position_within_threshold and orientation_within_threshold:
                box_color = [0, 1, 0, 0.5]  # Green - both within limits
            elif not position_within_threshold and not orientation_within_threshold:
                box_color = [1, 0, 0, 0.5]  # Red - both exceed limits
            else:
                box_color = [0, 0, 1, 0.5]  # Blue - one exceeds limit
            
            # Update box color
            p.changeVisualShape(box_id, -1, rgbaColor=box_color)
            
            # Update the visual target box position and orientation
            p.resetBasePositionAndOrientation(box_id, target_pos, target_orientation_quat)

            # Apply IK first using the determined target

            ik_solution = p.calculateInverseKinematics(robot_id, end_effector_index, target_pos, target_orientation_quat)
            #print(f"IK Solution: {ik_solution}", end="\r")
            apply_ik_solution(robot_id, ik_solution, joint_idxs)

            # Get current state of the end effector AFTER applying IK (update for visualization)
            link_state = p.getLinkState(robot_id, end_effector_index)
            ee_pos = link_state[0] # World position
            actual_orientation_quat = link_state[1] # Actual world orientation (quaternion)
            actual_orientation_euler = p.getEulerFromQuaternion(actual_orientation_quat)

            line_length = 0.05 # Length of visualization lines

            # --- Visualize Desired Orientation (Red Line) ---
            # Remove previous red line
            if orientation_line_id is not None:
                p.removeUserDebugItem(orientation_line_id)

            # Calculate end point for the desired orientation line (originating from current ee_pos)
            rot_matrix_desired = p.getMatrixFromQuaternion(target_orientation_quat) # Use active target orientation
            z_axis_direction_desired = [rot_matrix_desired[2], rot_matrix_desired[5], rot_matrix_desired[8]]
            # Invert the z-axis direction to point upside down
            line_end_desired = [ee_pos[0] - z_axis_direction_desired[0] * line_length,
                                ee_pos[1] - z_axis_direction_desired[1] * line_length,
                                ee_pos[2] - z_axis_direction_desired[2] * line_length]

            # Draw the new red line (Desired Orientation)
            #orientation_line_id = p.addUserDebugLine(ee_pos, line_end_desired, [1, 0, 0], 5) # Red line

            # --- Calculate and Display Orientation Difference ---
            # Quaternion difference
            quat_diff = p.getDifferenceQuaternion(target_orientation_quat, actual_orientation_quat)
            
            # Convert quaternion difference to Euler angles (axis-angle representation essentially)
            axis, angle = p.getAxisAngleFromQuaternion(quat_diff)
            
            
            
            # Euler angle difference (direct subtraction, might wrap around pi)
            euler_diff = [d - a for d, a in zip(target_orientation_euler, actual_orientation_euler)]
            # Normalize Euler differences to [-pi, pi] for better readability (optional)
            euler_diff_norm = [(diff + np.pi) % (2 * np.pi) - np.pi for diff in euler_diff]
            #print (f"Orientation Difference (Euler): {euler_diff_norm}")
            

            # Remove previous orientation debug text
            if orientation_diff_text_id is not None:
                p.removeUserDebugItem(orientation_diff_text_id)

            # Format orientation difference text
            orientation_diff_text = (
                f"  Quat Diff Angle: {angle:.3f} rad\n"
            )
            
            # Add new orientation debug text
            if angle> 0.1:
                color=[1,0,0]
            else:
                color = [0, 0, 1]
            
            #orientation_diff_text_id = p.addUserDebugText(
            #    orientation_diff_text, 
            #    textPosition=[0.05, 0.0, 0.8], # Position in the GUI
            #    textColorRGB=color, 
            #    textSize=1.0
            #)
            #orientation_diff_text_id = p.addUserDebugText(
            #    str(rad2nicodeg(joint_names, ik_solution)), 
            #    textPosition=[0.05, -2.0, 0.5], # Position in the GUI
            #    textColorRGB=color, 
            #    textSize=1.0
            #)

            # --- Calculate and Display Position Difference ---
            # Calculate Euclidean distance using the active target position
            pos_diff_vec = np.array(target_pos) - np.array(ee_pos)
            pos_distance = np.linalg.norm(pos_diff_vec)

            # Remove previous position debug text
            if position_diff_text_id is not None:
                p.removeUserDebugItem(position_diff_text_id)

            # Format position difference text
            position_diff_text = f"Position Distance: {pos_distance:.4f} m"

            if pos_distance> 0.1:
                color2 = [1,0,0]
            else:
                color2 = [0, 0, 1]

            # Add new position debug text
            #position_diff_text_id = p.addUserDebugText(
            #    position_diff_text,
            #    textPosition=[0.05, 0, 0.7], # Position above orientation text
            #    textColorRGB=color2, 
            #    textSize=1.0
            #)

            time.sleep(0.01)
    except KeyboardInterrupt:
        p.disconnect()

if __name__ == "__main__":
    main()