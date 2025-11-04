import pybullet as p
import pybullet_data
import argparse
import time
import numpy as np
from utils.helpers import get_robot_dict, get_workspace_dict
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
    key_order = ['path', 'position', 'orientation', 'default_joint_ori', 'ee_pos', 'ee_ori']
    
    for key, value in rd.items():
        parts = []
        
        # Sort keys according to defined order, then alphabetically for any remaining
        sorted_keys = sorted(value.keys(), key=lambda k: (key_order.index(k) if k in key_order else len(key_order), k))
        
        for k in sorted_keys:
            v = value[k]
            if k == 'position':
                parts.append(f"'{k}': np.array({list(v)})")
            elif k in ['default_joint_ori', 'ee_pos', 'ee_ori']:
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

def main():
    parser = argparse.ArgumentParser(description='URDF Visualizer with Joint Sliders')
    parser.add_argument("-b", "--robot",  type=str, default=None,
                       help='Robot key from r_dict (if not provided, interactive selection)')
    parser.add_argument("-i", "--interactive", action="store_true",
                       help='Enable interactive robot selection')
    args = parser.parse_args()

    # Get robot dictionary
    rd = get_robot_dict()
    
    # Select robot
    if args.interactive or args.robot is None:
        # Interactive selection
        robots = sorted(rd.keys())
        print("Available robots:")
        for i, robot_key in enumerate(robots):
            print(f"[{i+1}] {robot_key}")
        
        while True:
            sel = input("Select robot index (q to quit): ").strip().lower()
            if sel in ("q", "quit", "exit"):
                return
            if sel.isdigit():
                idx = int(sel)
                if 1 <= idx <= len(robots):
                    selected_robot = robots[idx - 1]
                    break
            print("Invalid selection.")
    else:
        if args.robot not in rd:
            print(f"Error: Robot '{args.robot}' not found in r_dict")
            print(f"Available robots: {', '.join(sorted(rd.keys()))}")
            return
        selected_robot = args.robot
    
    print(f"\nSelected robot: {selected_robot}")

    # Initialize PyBullet
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    # Get workspace dictionary
    ws_dict = get_workspace_dict()
    workspace_key = "table_uni"
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
    floor_path = os.path.join(currentdir, "rooms/plane.urdf")
    if os.path.exists(floor_path):
        floor_id = p.loadURDF(floor_path, transform['position'], 
                             p.getQuaternionFromEuler(transform['orientation']), 
                             useFixedBase=True)
        print(f"Loaded floor from: {floor_path}")
    
    # Fixed orientation for IK (quaternion)
    fixed_orientation = p.getQuaternionFromEuler([0, -np.pi/2, 0])  # Fixed downward orientation

    # Load URDF
    try:
        robot_info = rd[selected_robot]
        urdf = robot_info['path']
        robot_base_pos = list(np.array(robot_info.get('position', [0.0, 0.0, 0.0])).astype(float))
        robot_base_orientation = robot_info.get('orientation', [0.0, 0.0, 0.0])
        robot_base_quat = p.getQuaternionFromEuler(robot_base_orientation)
        
        robot_id = p.loadURDF(urdf, useFixedBase=True, 
                              basePosition=robot_base_pos,
                              baseOrientation=robot_base_quat)
        print(f"Loaded robot from: {urdf}")
        print(f"Robot position: {robot_base_pos}")
        print(f"Robot orientation (euler): {robot_base_orientation}")
    except Exception as e:
        print(f"Error: Failed to load URDF file of robot '{selected_robot}': {e}")
        return

    num_joints = p.getNumJoints(robot_id)

    # Find end effector link
    end_effector_index = -1
    for joint_idx in range(num_joints):
        joint_info = p.getJointInfo(robot_id, joint_idx)
        link_name = joint_info[12].decode("utf-8")
        if link_name == 'endeffector':
            end_effector_index = joint_idx
            print(f"Found end effector link at joint index: {end_effector_index}")
            break
    
    if end_effector_index == -1:
        print("Warning: Could not find end effector link named 'endeffector' in URDF")

    # Reset joint positions to default_joint_ori if available
    if 'default_joint_ori' in robot_info:
        default_joint_ori = robot_info['default_joint_ori']
        print(f"Resetting joints to default_joint_ori: {default_joint_ori}")

        # Get list of non-fixed joints
        non_fixed_joints = []
        for joint_idx in range(num_joints):
            joint_info = p.getJointInfo(robot_id, joint_idx)
            joint_type = joint_info[2]
            if joint_type != p.JOINT_FIXED:
                non_fixed_joints.append(joint_idx)
        
        # Reset joint states
        for i, joint_idx in enumerate(non_fixed_joints):
            if i < len(default_joint_ori):
                p.resetJointState(robot_id, joint_idx, default_joint_ori[i])
                print(f"  Joint {joint_idx}: {default_joint_ori[i]}")

    # Get joint information
    sliders = []
    
    for joint_idx in range(num_joints):
        joint_info = p.getJointInfo(robot_id, joint_idx)
        joint_name = joint_info[1].decode("utf-8")
        joint_type = joint_info[2]
        
        # Only create sliders for non-fixed joints
        if joint_type != p.JOINT_FIXED:
            lower = joint_info[8]
            upper = joint_info[9]
            
            # Handle unlimited joints
            if lower >= upper:
                lower, upper = -180, 180  # Default to ±180° for rotation joints
            
            # Get current joint position for slider start value
            joint_state = p.getJointState(robot_id, joint_idx)
            current_pos = joint_state[0]
                
            # Convert joint limits from radians to degrees for display
            lower_deg = lower * 57.2958
            upper_deg = upper * 57.2958
            if joint_type == p.JOINT_REVOLUTE:
                current_pos_deg = current_pos * 57.2958
                slider = p.addUserDebugParameter(
                    paramName=joint_name + " (deg)",
                    rangeMin=lower_deg,
                    rangeMax=upper_deg,
                    startValue=current_pos_deg
                )
                sliders.append((joint_idx, slider))
            else:
                #Prismatic joints
                slider = p.addUserDebugParameter(
                    paramName=joint_name + " (m)",
                    rangeMin=lower,
                    rangeMax=upper,
                    startValue=current_pos
                )
                sliders.append((joint_idx,slider))

    # Box control variables
    #box_size = 0.03  # 5x5x5 cm
    #box_id = p.createMultiBody(
    #    baseMass=0.1,
    #    baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[box_size/2]*3),
    #    basePosition=[0.35, 0, 0.07]  # Initial position
    #)
    
    # Create sliders for box position control
    #x_slider = p.addUserDebugParameter("Box X", 0.05, 0.6, 0.35)
    #y_slider = p.addUserDebugParameter("Box Y", -0.45, 0.45, 0)
    #z_slider = p.addUserDebugParameter("Box Z", 0.07, 0.6, 0.07)

    # Main simulation loop
    p.setRealTimeSimulation(1)
    p.resetDebugVisualizerCamera(
        cameraDistance=1.95,
        cameraYaw=90,
        cameraPitch=-15,
        cameraTargetPosition=[0, 0, 0.3]
    )
    
    # Get path to helpers.py
    helpers_path = os.path.join(os.path.dirname(__file__), 'utils', 'helpers.py')
    
    try:
        while True:
            # Update box position from sliders
            #box_pos = [
            #    p.readUserDebugParameter(x_slider),
            #    p.readUserDebugParameter(y_slider), 
            #    p.readUserDebugParameter(z_slider)
            #]
            #p.resetBasePositionAndOrientation(box_id, box_pos, [0,0,0,1])
            
            # Check keyboard events
            keys = p.getKeyboardEvents()
            if ord('t') in keys and keys[ord('t')] & p.KEY_WAS_TRIGGERED:
                # Print current joint values
                joint_values = []
                for joint_idx, slider_id in sliders:
                    value_deg = p.readUserDebugParameter(slider_id)
                    joint_info = p.getJointInfo(robot_id, joint_idx)
                    joint_type = joint_info[2]
                    if joint_type == p.JOINT_REVOLUTE:
                        value = value_deg * 0.0174533  # Convert degrees to radians
                    else:
                        value = value_deg
                    joint_values.append(round(value, 2))
                print(f"Current joint values: {joint_values}")

                # Add default_joint_ori to robot dict first
                rd[selected_robot]['default_joint_ori'] = joint_values

                # Get end effector pose if available
                if end_effector_index != -1:
                    link_state = p.getLinkState(robot_id, end_effector_index)
                    ee_pos = [round(x, 4) for x in link_state[0]]
                    ee_quat = link_state[1]
                    ee_ori = [round(x, 4) for x in p.getEulerFromQuaternion(ee_quat)]
                    print(f"End effector position: {ee_pos}")
                    print(f"End effector orientation (euler): {ee_ori}")
                    
                    # Store ee_pos and ee_ori in robot dict
                    rd[selected_robot]['ee_pos'] = ee_pos
                    rd[selected_robot]['ee_ori'] = ee_ori

                # Save to helpers.py
                if save_robot_dict_to_helpers(rd, helpers_path):
                    print(f"Updated helpers.py with default_joint_ori, ee_pos, and ee_ori for {selected_robot}")
                else:
                    print("Failed to update helpers.py")
            
            for joint_idx, slider_id in sliders:
                value_deg = p.readUserDebugParameter(slider_id)
                joint_info = p.getJointInfo(robot_id, joint_idx)
                joint_type = joint_info[2]
                if joint_type == p.JOINT_REVOLUTE:
                    value = value_deg * 0.0174533  # Convert degrees back to radians
                else:
                    value = value_deg
                p.setJointMotorControl2(
                    bodyIndex=robot_id,
                    jointIndex=joint_idx,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=value,
                    force=500
                )
            time.sleep(0.01)
    except KeyboardInterrupt:
        p.disconnect()

if __name__ == "__main__":
    main()