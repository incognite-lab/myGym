import pybullet as p
import pybullet_data
import argparse
import time
import numpy as np
from numpy import rad2deg, deg2rad, set_printoptions, array, linalg, round, any, mean

from grasper import Grasper

def get_look_at_quaternion(eye_pos, target_pos):
    """
    Calculates a quaternion to orient an object at eye_pos to look at target_pos.
    This is achieved by calculating the required yaw and pitch from the direction vector.
    
    Args:
        eye_pos (list or np.array): The starting position (e.g., head link).
        target_pos (list or np.array): The position to look at.

    Returns:
        list: A quaternion [x, y, z, w].
    """
    # Ensure inputs are numpy arrays
    eye_pos = np.array(eye_pos)
    target_pos = np.array(target_pos)
    
    # Calculate the direction vector from the eye to the target
    direction = target_pos - eye_pos
    
    # Handle case where the target is at the same position as the eye
    if np.linalg.norm(direction) < 1e-6:
        return [0, 0, 0, 1] # Return identity quaternion (no rotation)
        
    # Calculate Yaw (rotation around Z-axis)
    # This determines the left-right direction
    yaw = np.arctan2(direction[1], direction[0])
    
    # Calculate Pitch (rotation around Y-axis)
    # This determines the up-down direction
    # The horizontal distance is the length of the projection onto the XY plane
    horizontal_dist = np.sqrt(direction[0]**2 + direction[1]**2)
    pitch = np.arctan2(-direction[2], horizontal_dist)
    
    # We assume Roll (rotation around X-axis) is 0 for a simple "look at"
    roll = 0
    
    # Convert the Euler angles (roll, pitch, yaw) to a quaternion
    return p.getQuaternionFromEuler([roll, pitch, yaw])

def rad2nicodeg(nicojoints, rads):
        """Converts radians to Nico-specific degrees. Returns a dictionary."""
        if isinstance(nicojoints, str):
            nicojoints = [nicojoints]
        if isinstance(rads, (int, float)):
            rads = [rads]

        nicodegrees_dict = {}
        for nicojoint, rad in zip(nicojoints, rads):
            # Ensure rad is a number before applying rad2deg
            if isinstance(rad, (int, float)):
                if nicojoint == 'r_wrist_z':
                    nicodegree = rad2deg(rad) * 2
                elif nicojoint == 'r_wrist_x':
                    nicodegree = rad2deg(rad) * 4
                else:
                    nicodegree = rad2deg(rad)
                nicodegrees_dict[nicojoint] = nicodegree
            else:
                 # Handle cases where IK might return non-numeric values or None
                 print(f"Warning: Non-numeric radian value '{rad}' for joint '{nicojoint}'. Skipping conversion.")
                 # Optionally add a placeholder like None to the dict, or just skip
                 # nicodegrees_dict[nicojoint] = None

        return nicodegrees_dict # Return the dictionary

def calculate_ik(robot_id, end_effector_index, box_pos, orientation):   
    """Calculate inverse kinematics solution for the robot arm."""
    ik_solution = p.calculateInverseKinematics(
        robot_id,
        end_effector_index,
        box_pos,
        targetOrientation=orientation,
        maxNumIterations=100,
        residualThreshold=0.001
    )
    return ik_solution

def apply_ik_solution(robot_id, ik_solution, joint_idxs):
    """Apply the IK solution to the robot joints."""
    joint_values = []
    for index, joint_idx in enumerate(joint_idxs):
        joint_pos = ik_solution[index]
        p.setJointMotorControl2(
            bodyIndex=robot_id,
            jointIndex=joint_idx,
            controlMode=p.POSITION_CONTROL,
            targetPosition=joint_pos
        )
        joint_values.append(f"{p.getJointInfo(robot_id, joint_idx)[1].decode('utf-8')}: {joint_pos:.4f} rad ({joint_pos*57.2958:.2f}°)")
    return joint_values

def find_gripper_joints(robot_id):
    """Find all joints related to the right fingers and return their indices."""
    num_joints = p.getNumJoints(robot_id)
    gripper_idxs = []
    for joint_idx in range(num_joints):
        joint_info = p.getJointInfo(robot_id, joint_idx)
        joint_name = joint_info[1].decode("utf-8")
        # Check if the joint name contains 'r_' and 'finger'
        if 'right_' in joint_name and 'finger' in joint_name:
            gripper_idxs.append(joint_idx)
            print(joint_name)
    print(f"Found gripper joints: {gripper_idxs}") # Optional: print found joints
    return gripper_idxs

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
    parser.add_argument("--urdf", type=str, default="./urdf/nico_grasper.urdf", help="Path to the robot URDF file.")
    parser.add_argument("--config", type=str, default="./nico_humanoid_upper_rh7d_ukba.json", help="Path to the motor config JSON.")
    parser.add_argument("--real_robot", action="store_true", help="Execute actions on the real robot (requires hardware connection).")
    parser.add_argument("-i", "--init_pos", nargs=3, type=float, default = [0.0, -0.3, 0.5], help="Target position for the robot end effector as a list of three floats.")
    parser.add_argument("-g", "--goal_pos", nargs=3, type=float, default = [0.3, -0.3, 0.07], help="Target position for the robot end effector as a list of three floats.")
    parser.add_argument("--scene", type=str, help="Path to workspace scene object, such as tiago table", default = "./urdf/table_nico.urdf")
    parser.add_argument("--texture", type=str, help="Path to the texture of scene object", default = "./urdf/textures/table.jpg")
    
    args = parser.parse_args()

    connect_hw = args.real_robot

    #print("Initializing Grasper...")
    #try:
    #    grasper = Grasper(
    #        urdf_path="./nico_grasper.urdf",
    #        motor_config="./nico_humanoid_upper_rh7d_ukba.json",
    #        connect_robot=True,     # Connect to the real robot hardware
    #    )
    #    print("Grasper initialized successfully for real robot.")
    #except Exception as e:
    #    print(f"Error initializing Grasper for real robot: {e}")

    mode = "right"

    # Initialize PyBullet
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Fixed orientation for IK (quaternion)
    #orientation = p.getQuaternionFromEuler([-3.14, 0, 0])  # Fixed downward orientation

    # Create ground plane (90x60x3 cm)

    # p.createMultiBody(baseVisualShapeIndex=p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[.30, .45, 0.025],
    #                                                        rgbaColor=[0.8, 0.7, 0.4, 1]),
    #                   baseCollisionShapeIndex=p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[.30, .45, 0.025]),
    #                   baseMass=0, basePosition=[0.26, 0, 0.029])

    scene_uid = p.loadURDF(args.scene, useFixedBase=True)
    texture_id = p.loadTexture(args.texture)
    p.changeVisualShape(scene_uid, -1, rgbaColor=[1, 1, 1, 1], textureUniqueId=texture_id)
    #Nico table position
    p.resetBasePositionAndOrientation(scene_uid, [0.0, 0.0, -0.73], p.getQuaternionFromEuler([0, 0, -np.pi / 2]))
    #Tiago table position 
    #p.resetBasePositionAndOrientation(scene_uid, [0.225, 0.2, 0.7], p.getQuaternionFromEuler([0, 0, -np.pi / 2]))
    # Create tablet mesh
    # p.createMultiBody(baseVisualShapeIndex=p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[.165, .267, 0.001],
    #                                                        rgbaColor=[0, 0, 0.0, .5]),
    #                   baseCollisionShapeIndex=p.createCollisionShape(shapeType=p.GEOM_BOX,
    #                                                                 halfExtents=[.165, .267, 0.001]), baseMass=0,
    #                   basePosition=[0.395, 0, 1.054])
    time.sleep(1)


    # Load URDF
    try:
        robot_id = p.loadURDF(args.urdf, useFixedBase=True)
    except:
        print(f"Error: Failed to load URDF file '{args.urdf}'")
        return

    # Find end effector link
    num_joints = p.getNumJoints(robot_id)
    end_effector_index = -1
    for joint_idx in range(num_joints):
        joint_info = p.getJointInfo(robot_id, joint_idx)
        link_name = joint_info[12].decode("utf-8")
        if link_name == 'endeffectol':
            end_effector_index_l = joint_idx
        elif link_name == 'endeffector':
            end_effector_index_r = joint_idx
        elif link_name == 'head':
            end_effector_index_h = joint_idx
    
    end_effector_index = end_effector_index_r
    if end_effector_index == -1:
        print("Error: Could not find end effector link in URDF")
        return

    # Get joint information for arm control using the new function
    joint_idxs, joint_names = get_controllable_arm_joints(robot_id, num_joints)

    # Find gripper joint indices
    gripper_idxs = find_gripper_joints(robot_id)


    # Box control variables
    box_size = 0.03
    box_initial_pos = args.goal_pos
    box_id = p.createMultiBody(
        baseMass=0, # Set mass to 0 if it's only visual
        baseCollisionShapeIndex=-1, # No collision shape
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[box_size/2]*3, rgbaColor=[1, 0.0, 0.0, 0.8]), # Visual shape only
        basePosition=box_initial_pos
    )

    # Second box control variables
    box2_size = 0.03
    box2_initial_pos = args.init_pos
    box2_id = p.createMultiBody(
        baseMass=0, # Visual only
        baseCollisionShapeIndex=-1, # No collision
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[box2_size/2]*3, rgbaColor=[0, 0, 1, 0.7]), # Black color
        basePosition=box2_initial_pos
    )

    # Create sliders for  box position control
    x2_slider = p.addUserDebugParameter("Init X", -0.2, 1.0, box2_initial_pos[0])
    y2_slider = p.addUserDebugParameter("Init Y", -0.6, 0.6, box2_initial_pos[1])
    z2_slider = p.addUserDebugParameter("Init Z", 0.3, 1.5, box2_initial_pos[2])

    # Create additional init sliders for Yaw, Pitch, Roll
    init_roll_slider = p.addUserDebugParameter("Init Roll", -np.pi, np.pi, 0)
    init_pitch_slider = p.addUserDebugParameter("Init Pitch", -np.pi, np.pi, -np.pi/2)
    init_yaw_slider = p.addUserDebugParameter("Init Yaw", -np.pi, np.pi, 0) 
    
    
    # Create sliders for goal box position control
    x_slider = p.addUserDebugParameter("Goal X", 0.0, 0.6, box_initial_pos[0])
    y_slider = p.addUserDebugParameter("Goal Y", -0.5, 0.5, box_initial_pos[1])
    z_slider = p.addUserDebugParameter("Goal Z", 0.0, 0.7, box_initial_pos[2])
    roll_slider = p.addUserDebugParameter("Goal_Roll", -np.pi, np.pi, 0)
    pitch_slider = p.addUserDebugParameter("Goal_Pitch", -np.pi, np.pi, 0)
    yaw_slider = p.addUserDebugParameter("Goal_Yaw", -np.pi, np.pi, 0) # Default to downward




    # Main simulation loop
    p.setRealTimeSimulation(1)
    p.resetDebugVisualizerCamera(
        cameraDistance=0.95,
        cameraYaw=90,
        cameraPitch=-15,
        cameraTargetPosition=[0, 0, 0.3]
    )
 
    orientation_line_id = None # Initialize desired orientation line ID tracker
    actual_orientation_line_id = None # Initialize actual orientation line ID tracker
    orientation_diff_text_id = None # Initialize debug text ID tracker
    position_diff_text_id = None # Initialize position difference text ID tracker
    
    use_second_target = True # Flag to switch between target boxes/orientations
    
    try:
        while True:
            # Check keyboard events first to potentially switch targets
            keys = p.getKeyboardEvents()
            if ord('x') in keys and keys[ord('x')] & p.KEY_WAS_TRIGGERED:
                use_second_target = not use_second_target
                print(f"Switched target source: {'Initial state' if use_second_target else 'Goal State'}")
                #grasper.perform_move()
            
            # Check if 'c' key is pressed to close the gripper (set gripper joints to 0)
            if ord('c') in keys and keys[ord('c')] & p.KEY_WAS_TRIGGERED:
                print("Closing gripper (setting finger joints to -2)")
                gripper_zero_solution = [-2.0] * len(gripper_idxs) # Create a list of zeros
                apply_ik_solution(robot_id, gripper_zero_solution, gripper_idxs) # Apply to gripper joints
                #grasper.perform_grasp()

            if ord('d') in keys and keys[ord('d')] & p.KEY_WAS_TRIGGERED:
                print("Opening gripper (setting finger joints to 0)")
                gripper_zero_solution = [0.0] * len(gripper_idxs) # Create a list of zeros
                apply_ik_solution(robot_id, gripper_zero_solution, gripper_idxs) # Apply to gripper joints
                #grasper.perform_drop()
            
            if ord('m') in keys and keys[ord('m')] & p.KEY_WAS_TRIGGERED:
                grasper.move_arm([0.35,-0.4,0.2], args.ori, args.side)
            
            if ord('b') in keys and keys[ord('b')] & p.KEY_WAS_TRIGGERED:
                end_effector_index = end_effector_index_r
                print("Switched to right end effector") 
                mode = "right"

            if ord('h') in keys and keys[ord('h')] & p.KEY_WAS_TRIGGERED:
                end_effector_index = end_effector_index_h
                print("Switched to head end effector") 
                mode = "head"             
            
            if ord('n') in keys and keys[ord('n')] & p.KEY_WAS_TRIGGERED:
                end_effector_index = end_effector_index_l
                print("Switched to left end effector")
                mode = "left"
            

            


            # --- Determine Target Position and Orientation based on flag ---
            if use_second_target:
                # Use second box sliders for position
                target_pos = [
                    p.readUserDebugParameter(x2_slider),
                    p.readUserDebugParameter(y2_slider),
                    p.readUserDebugParameter(z2_slider)
                ]
                # Use initial sliders for orientation
                roll = p.readUserDebugParameter(init_roll_slider)
                pitch = p.readUserDebugParameter(init_pitch_slider)
                yaw = p.readUserDebugParameter(init_yaw_slider)
                target_orientation_quat = p.getQuaternionFromEuler([roll, pitch, yaw])
                target_orientation_euler = [roll, pitch, yaw]
                # Update the visual position of the second box (black)
                p.resetBasePositionAndOrientation(box2_id, target_pos, [0,0,0,1])
                # Keep the first box (red) at its slider position for reference
                box1_pos_ref = [ p.readUserDebugParameter(x_slider), p.readUserDebugParameter(y_slider), p.readUserDebugParameter(z_slider)]
                p.resetBasePositionAndOrientation(box_id, box1_pos_ref, [0,0,0,1])

            else:
                # Use first box sliders for position
                target_pos = [
                    p.readUserDebugParameter(x_slider),
                    p.readUserDebugParameter(y_slider), 
                    p.readUserDebugParameter(z_slider)
                ]
                # Use main orientation sliders
                roll = p.readUserDebugParameter(roll_slider)
                pitch = p.readUserDebugParameter(pitch_slider)
                yaw = p.readUserDebugParameter(yaw_slider)
                target_orientation_quat = p.getQuaternionFromEuler([roll, pitch, yaw])
                target_orientation_euler = [roll, pitch, yaw]
                # Update the visual position of the first box (red)
                p.resetBasePositionAndOrientation(box_id, target_pos, [0,0,0,1])
                # Keep the second box (black) at its slider position for reference
                box2_pos_ref = [ p.readUserDebugParameter(x2_slider), p.readUserDebugParameter(y2_slider), p.readUserDebugParameter(z2_slider)]
                p.resetBasePositionAndOrientation(box2_id, box2_pos_ref, [0,0,0,1])


            # Apply IK first using the determined target
            if mode == "head":
                # Get the current head position to calculate the look-at orientation
                head_link_state = p.getLinkState(robot_id, end_effector_index_h)
                head_pos = head_link_state[0]
                
                # Calculate the look-at quaternion
                target_orientation_quat = get_look_at_quaternion(head_pos, target_pos)
                ik_solution = p.calculateInverseKinematics(robot_id, end_effector_index, target_pos,target_orientation_quat)
            else:
                ik_solution = p.calculateInverseKinematics(robot_id, end_effector_index, target_pos, target_orientation_quat)
            #print(f"IK Solution: {ik_solution}", end="\r")
            apply_ik_solution(robot_id, ik_solution, joint_idxs)

            # Get current state of the end effector AFTER applying IK
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
            orientation_line_id = p.addUserDebugLine(ee_pos, line_end_desired, [1, 0, 0], 5) # Red line

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