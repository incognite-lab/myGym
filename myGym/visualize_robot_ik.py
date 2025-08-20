import pybullet as p
import pybullet_data
import argparse
import time
import numpy as np
from numpy import rad2deg, deg2rad, set_printoptions, array, linalg, round, any, mean


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
        if 'gjoint' in joint_name:
            gripper_idxs.append(joint_idx)
            print(joint_name)
    print(f"Found gripper joints: {gripper_idxs}") # Optional: print found joints
    return gripper_idxs

def get_controllable_arm_joints(robot_id, num_joints):
    """Get joint information for arm control (non-fixed joints), excluding gripper joints."""
    joint_idxs = []
    joint_names = []
    gripper_idxs = []
    gripper_names = []
    print("\n--- Controllable Arm Joints ---")
    for joint_idx in range(num_joints):
        joint_info = p.getJointInfo(robot_id, joint_idx)
        joint_name = joint_info[1].decode("utf-8")
        joint_type = joint_info[2]
        
        # Print all joint names and IDs for inspection
        #print(f"  Joint ID: {joint_idx}, Name: {joint_name}, Type: {joint_type}")
        link_name = joint_info[12].decode("utf-8")
        if link_name == 'endeffector':
            end_effector_index = joint_idx

        if joint_type != p.JOINT_FIXED:
            # Exclude gripper joints from the main IK control list
            if 'gjoint' in joint_name:
                gripper_idxs.append(joint_idx)
                gripper_names.append(joint_name)
                g_lower_limit = joint_info[8]
                g_upper_limit = joint_info[9]
            if 'rjoint' in joint_name:
                joint_idxs.append(joint_idx)
                joint_names.append(joint_name)

    print(f"Selected Arm Joint Indices for IK: {joint_idxs}")
    print(f"Selected Arm Joint Names for IK: {joint_names}")
    print(f"Selected Gripper Joint Indices for IK: {gripper_idxs}")
    print(f"Selected Gripper Joint Names for IK: {gripper_names}")
    print("-------------------------------\n")
    return joint_idxs, joint_names, gripper_idxs, gripper_names, g_lower_limit, g_upper_limit, end_effector_index


def main():
    parser = argparse.ArgumentParser(description="Nico Robot Grasping Control")
    parser.add_argument("--urdf", type=str, default="./envs/robots/tiago/tiago_dual_mygym_rotslide2.urdf", help="Path to the robot URDF file.")
    args = parser.parse_args()


    # Initialize PyBullet
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Load URDF
    try:
        robot_id = p.loadURDF(args.urdf, useFixedBase=True,flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)
    except:
        print(f"Error: Failed to load URDF file '{args.urdf}'")
        return

    # Find end effector link
    num_joints = p.getNumJoints(robot_id)

    # Get joint information for arm control using the new function
    joint_idxs, joint_names, gripper_idxs, gripper_names, g_lower_limit, g_upper_limit, end_effector_index = get_controllable_arm_joints(robot_id, num_joints)

    if end_effector_index == -1:
        print("Error: Could not find end effector link in URDF")
        return


    # Box control variables
    box_size = 0.03
    box_initial_pos = p.getLinkState(robot_id, end_effector_index)

    print(f"Initial end effector position: {box_initial_pos[0]}")
    box_id = p.createMultiBody(
        baseMass=0, # Set mass to 0 if it's only visual
        baseCollisionShapeIndex=-1, # No collision shape
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[box_size/2]*3, rgbaColor=[1, 0.0, 0.0, 0.8]), # Visual shape only
        basePosition=box_initial_pos)

    # Create sliders for  box position control
    x_slider = p.addUserDebugParameter("Box X", -1, 1.0, 0.4)
    y_slider = p.addUserDebugParameter("Box Y", -1, 1, -0.6)
    z_slider = p.addUserDebugParameter("Box Z", 0, 2, 0.4)

    # Create additional init sliders for Yaw, Pitch, Roll
    roll_slider = p.addUserDebugParameter("Box Roll", -np.pi, np.pi, 0)
    pitch_slider = p.addUserDebugParameter("Box Pitch", -np.pi, np.pi, 0)
    yaw_slider = p.addUserDebugParameter("Box Yaw", -np.pi, np.pi, 0)


    # Main simulation loop
    p.setRealTimeSimulation(1)
    p.resetDebugVisualizerCamera(
        cameraDistance=1.95,
        cameraYaw=90,
        cameraPitch=-15,
        cameraTargetPosition=[0, 0, 0.3]
    )
 
    orientation_line_id = None # Initialize desired orientation line ID tracker
    actual_orientation_line_id = None # Initialize actual orientation line ID tracker
    orientation_diff_text_id = None # Initialize debug text ID tracker
    position_diff_text_id = None # Initialize position difference text ID tracker
    
    
    try:
        while True:
            # Check keyboard events first to potentially switch targets
            keys = p.getKeyboardEvents()
            
            # Check if 'c' key is pressed to close the gripper (set gripper joints to 0)
            if ord('c') in keys and keys[ord('c')] & p.KEY_WAS_TRIGGERED:
                print(f"Closing gripper {g_lower_limit})")
                for idx in gripper_idxs:
                    p.setJointMotorControl2(
                        bodyIndex=robot_id,
                        jointIndex=idx,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=g_lower_limit
                    )

            if ord('d') in keys and keys[ord('d')] & p.KEY_WAS_TRIGGERED:
                print(f"Opening gripper to {g_upper_limit})")
                for idx in gripper_idxs:
                    p.setJointMotorControl2(
                        bodyIndex=robot_id,
                        jointIndex=idx,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=g_upper_limit
                    )
                        

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



            # Apply IK first using the determined target

            ik_solution = p.calculateInverseKinematics(robot_id, end_effector_index, target_pos)
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