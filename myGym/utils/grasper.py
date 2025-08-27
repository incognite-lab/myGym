import pybullet as p
import time
from numpy import random, rad2deg, deg2rad, set_printoptions, array, linalg, round, any, mean, arctan2, sqrt
from scipy.interpolate import Rbf
import sys  # Add this import at the top of the file
from utils.sim_height_calculation import calculate_z

# Attempt to import Motion, but handle failure gracefully if not simulating
try:
    from nicomotion.Motion import Motion
    NICOMOTION_AVAILABLE = True
except ImportError:
    Motion = None # Define Motion as None if import fails
    NICOMOTION_AVAILABLE = False
    print("Warning: nicomotion library not found. Hardware control disabled.")

class Grasper:
    # Constants
    SPEED = 0.03
    SPEEDF = 0.03
    DELAY = 2
    REPEAT = 1

    # Predefined poses (consider making these configurable or loading from file)
    RESET_POSE = [0,0,0,90,90,90,0,0,-180,-180,-180,-180,0.22,12.88,11.03,100.97,-24.13,-91.91,-180.0,-180.0,-180.0,-174.81]
    DROP_POSE = [0,0,-20,27,40,90,125,100,180,20,20,20,0,13,11,100,-24,-91,-180.0,-180.0,-180.0,-175]
    GRASP_POSE = [0,0,-2,34,17,122,161,-13,180,-180,-180,-180,0,13,11,100,-24,-91,-180.0,-180.0,-180.0,-175]
    INIT_POS = {  # standard position
        'head_z': 0.0, 'head_y': 0.0, 'r_shoulder_z': 1, 'r_shoulder_y': 87,
        'r_arm_x': 88, 'r_elbow_y': 87, 'r_wrist_z': 2, 'r_wrist_x': -29,
        'r_thumb_z': -1, 'r_thumb_x': 44, 'r_indexfinger_x': -90, 'r_middlefingers_x': 100.0,
        'l_shoulder_z': -24.0, 'l_shoulder_y': 13.0, 'l_arm_x': 0.0, 'l_elbow_y': 104.0,
        'l_wrist_z': -4.0, 'l_wrist_x': -55.0, 'l_thumb_z': -62.0, 'l_thumb_x': -180.0,
        'l_indexfinger_x': -170.0, 'l_middlefingers_x': -180.0
    }

    RIGHTHAND_RBF_POINTS = {
        'x': [0.133, 0.208, 0.272, 0.338, 0.388, 0.420, 0.458, 0.483, 0.496, 0.480, 0.464, 0.433, 0.392, 0.344, 0.291, 0.227, 0.164, 0.152, 0.139, 0.123, 0.104, 0.111, 0.085, 0.069, 0.085, 0.158, 0.177, 0.193, 0.202, 0.218, 0.234, 0.313, 0.300, 0.278, 0.268, 0.240, 0.227, 0.313, 0.344, 0.366, 0.376, 0.385, 0.407, 0.426, 0.426, 0.272, 0.369],
        'y': [-0.395, -0.437, -0.447, -0.411, -0.358, -0.289, -0.216, -0.142, -0.074, -0.005, 0.068, 0.142, 0.200, 0.247, 0.279, 0.305, 0.321, 0.258, 0.184, 0.100, 0.005, -0.084, -0.163, -0.253, -0.326, -0.289, -0.189, -0.089, 0.016, 0.116, 0.205, 0.158, 0.053, -0.053, -0.153, -0.242, -0.342, -0.316, -0.216, -0.121, -0.026, 0.079, -0.179, -0.089, 0.005, -0.384, -0.274],
        # 'yaw': [-0.595, -0.728, -0.794, -0.761, -0.694, -0.463, -0.265, -0.132, -0.033, 0.165, 0.331, 0.529, 0.728, 0.893, 1.058, 1.224, 1.455, 1.389, 1.389, 1.389, 1.488, 0.794, 0.298, -0.132, -0.298, -0.298, -0.099, 0.298, 0.959, 1.058, 1.190, 0.992, 0.761, 0.496, 0.066, -0.265, -0.496, -0.496, -0.198, 0.099, 0.298, 0.562, -0.066, 0.198, 0.331, -0.496, -0.331],                     # old
        'yaw': [-0.595, -0.728, -0.794, -0.761, -0.694, -0.463, -0.265, -0.132, -0.033, 0.165, 0.331, 0.529, 0.728, 0.893, 1.058, 1.224, 1.455, 1.389, 1.389, 1.389, 1.819, 1.257, 0.496, -0.132, -0.298, 0.0, 0.331, 0.661, 1.29, 1.488, 1.29, 1.157, 1.124, 0.827, 0.496, 0.165, -0.298, -0.198, 0.165, 0.331, 0.43, 0.529, -0.066, 0.198, 0.331, -0.496, -0.331],                       # new
        # 'z': [0.115, 0.123, 0.128, 0.130, 0.132, 0.131, 0.128, 0.129, 0.127, 0.127, 0.126, 0.124, 0.120, 0.114, 0.115, 0.107, 0.091, 0.095, 0.115, 0.105, 0.101, 0.095, 0.092, 0.089, 0.102, 0.106, 0.101, 0.094, 0.095, 0.104, 0.112, 0.116, 0.112, 0.109, 0.108, 0.110, 0.120, 0.118, 0.117, 0.122, 0.123],                                                                                     # old
        'z': [0.115, 0.123, 0.128, 0.13, 0.132, 0.131, 0.128, 0.129, 0.127, 0.127, 0.126, 0.124, 0.12, 0.114, 0.115, 0.107, 0.091, 0.095, 0.035, 0.038, 0.063, 0.115, 0.105, 0.101, 0.095, 0.092, 0.089, 0.096, 0.102, 0.106, 0.101, 0.098, 0.101, 0.104, 0.112, 0.118, 0.112, 0.109, 0.108, 0.11, 0.12, 0.118, 0.117, 0.122, 0.123]                                                            # new
    }

    LEFTHAND_RBF_POINTS = {
        'x': [0.133, 0.208, 0.272, 0.338, 0.388, 0.420, 0.458, 0.483, 0.496, 0.480, 0.464, 0.433, 0.392, 0.344, 0.291, 0.227, 0.164, 0.152, 0.139, 0.123, 0.104, 0.111, 0.085, 0.069, 0.085, 0.158, 0.177, 0.193, 0.202, 0.218, 0.234, 0.313, 0.300, 0.278, 0.268, 0.240, 0.227, 0.313, 0.344, 0.366, 0.376, 0.385, 0.407, 0.426, 0.426, 0.272, 0.369],
        'y': [0.395, 0.437, 0.447, 0.411, 0.358, 0.289, 0.216, 0.142, 0.074, 0.005, -0.068, -0.142, -0.2, -0.247, -0.279, -0.305, -0.321, -0.258, -0.184, -0.1, -0.005, 0.084, 0.163, 0.253, 0.326, 0.289, 0.189, 0.089, -0.016, -0.116, -0.205, -0.158, -0.053, 0.053, 0.153, 0.242, 0.342, 0.316, 0.216, 0.121, 0.026, -0.079, 0.179, 0.089, -0.005, 0.384, 0.274],
        'yaw': [0.595, 0.728, 0.794, 0.761, 0.694, 0.463, 0.265, 0.132, 0.033, -0.165, -0.331, -0.529, -0.728, -0.893, -1.058, -1.224, -1.455, -1.389, -1.389, -1.389, -1.819, -1.257, -0.496, 0.132, 0.298, 0.0, -0.331, -0.661, -1.29, -1.488, -1.29, -1.157, -1.124, -0.827, -0.496, -0.165, 0.298, 0.198, -0.165, -0.331, -0.43, -0.529, 0.066, -0.198, -0.331, 0.496, 0.331],
        'z': [0.107, 0.113, 0.114, 0.117, 0.114, 0.116, 0.114, 0.112, 0.113, 0.114, 0.113, 0.111, 0.107, 0.105, 0.102, 0.101, 0.087, 0.097, -0.010, 0.030, 0.046, 0.110, 0.099, 0.098, 0.092, 0.087, 0.086, 0.093, 0.098, 0.102, 0.097, 0.096, 0.098, 0.098, 0.105, 0.109, 0.106, 0.104, 0.101, 0.099, 0.109, 0.108, 0.106, 0.112, 0.111]
    }

    def __init__(self, urdf_path="./envs/robots/nico/nico_grasper.urdf", motor_config='./nico_humanoid_upper_rh7d_ukba.json', connect_robot=True):
        """
        Initializes the Grasper class.

        Args:
            urdf_path (str): Path to the robot URDF file.
            motor_config (str): Path to the motor configuration JSON file for nicomotion.
            connect_robot (bool): Whether to attempt connection to the physical robot..
        """
        set_printoptions(precision=3, suppress=True)

        self.head = ['head_z', 'head_y']    
        self.right_arm = ['r_shoulder_z', 'r_shoulder_y', 'r_arm_x', 'r_elbow_y', 'r_wrist_z', 'r_wrist_x']
        self.right_gripper = ['r_thumb_z', 'r_thumb_x', 'r_indexfinger_x', 'r_middlefingers_x']
        self.left_arm = ['l_shoulder_z', 'l_shoulder_y', 'l_arm_x', 'l_elbow_y', 'l_wrist_z', 'l_wrist_x']
        self.left_gripper = ['l_thumb_z', 'l_thumb_x', 'l_indexfinger_x', 'l_middlefingers_x']
        self.head_actuated = []  
        self.right_arm_actuated = [] # List to store actuated joints for the right arm
        self.right_gripper_actuated = [] # List to store actuated joints for the right gripper
        self.left_arm_actuated = [] # List to store actuated joints for the left arm 
        self.left_gripper_actuated = []
        self.robot_id = None
        self.num_joints = 0
        self.joints_limits_l = []
        self.joints_limits_u = []
        self.joints_ranges = []
        self.joints_rest_poses = []
        self.joint_names = []
        self.link_names = []
        self.joint_indices = [] # List of movable joint indices
        self.joint_name_to_index = {} # Map from joint name to index
        self.end_effector_index_r = -1
        self.end_effector_index_l = -1
        self.end_effector_index = -1
        self.ori = [0,0,0]
        self.robot = None # For nicomotion hardware interface
        self.is_pybullet_connected = False
        self.is_robot_connected = False
        self.closed = 20
        self.open = -170
        self.opposite = 170
        self.speed = self.SPEED
        self.delay = self.DELAY
        self.righthand_rbf_yaw = Rbf(
            self.RIGHTHAND_RBF_POINTS['x'],
            self.RIGHTHAND_RBF_POINTS['y'],
            self.RIGHTHAND_RBF_POINTS['yaw'],
            function='multiquadric'                 # possible options: linear, gaussian
        )
        self.righthand_rbf_z = Rbf(
            self.RIGHTHAND_RBF_POINTS['x'][0:18] + self.RIGHTHAND_RBF_POINTS['x'][20:],
            self.RIGHTHAND_RBF_POINTS['y'][0:18] + self.RIGHTHAND_RBF_POINTS['y'][20:],
            self.RIGHTHAND_RBF_POINTS['z'],
            function='multiquadric'                 # possible options: linear, gaussian
        )
        self.lefthand_rbf_yaw = Rbf(
            self.LEFTHAND_RBF_POINTS['x'],
            self.LEFTHAND_RBF_POINTS['y'],
            self.LEFTHAND_RBF_POINTS['yaw'],
            function='multiquadric'                 # possible options: linear, gaussian
        )
        self.lefthand_rbf_z = Rbf(
            self.LEFTHAND_RBF_POINTS['x'][0:18] + self.LEFTHAND_RBF_POINTS['x'][20:],
            self.LEFTHAND_RBF_POINTS['y'][0:18] + self.LEFTHAND_RBF_POINTS['y'][20:],
            self.LEFTHAND_RBF_POINTS['z'],
            function='multiquadric'                 # possible options: linear, gaussian
        )

        try:
            # Connect to PyBullet (GUI or DIRECT)
            self.physics_client = p.connect(p.DIRECT)
            self.is_pybullet_connected = True
            print(f"Connected to PyBullet")

            # Load robot slightly above ground in GUI mode
            self.robot_id =p.loadURDF(urdf_path, useFixedBase=True)
            self.num_joints = p.getNumJoints(self.robot_id)
            self._get_joints_limits()
            print(f"Loaded URDF: {urdf_path}")
            print(f"Found {self.num_joints} joints.")
            print(f"Found {len(self.joint_names)} movable joints.")
            print (f"Head joints: {self.head_actuated}")
            print (f"Right arm joints: {self.right_arm_actuated}")
            print (f"Left arm joints: {self.left_arm_actuated}")
            print (f"Right gripper joints: {self.right_gripper_actuated}")
            print (f"Left gripper joints: {self.left_gripper_actuated}")
            print(f"Left end effector index: {self.end_effector_index_l}")
            print(f"Right end effector index: {self.end_effector_index_r}")
        except Exception as e:
            print(f"Error connecting to PyBullet or loading URDF: {e}")
            self.is_pybullet_connected = False


        if connect_robot and NICOMOTION_AVAILABLE and Motion is not None:
            try:
                self.robot = Motion(motorConfig=motor_config)
                self.is_robot_connected = True
                print(f"Robot hardware initialized using config: {motor_config}")
            except Exception as e:
                print(f"Could not initialize robot hardware: {e}")
                print("Proceeding without hardware connection.")
                self.is_robot_connected = False
        elif connect_robot and not NICOMOTION_AVAILABLE:
             print("Hardware connection requested, but nicomotion library is not available.")
             self.is_robot_connected = False
        else:
             # Not attempting to connect robot
             self.is_robot_connected = False

    def _get_joints_limits(self):
        """
        Internal method to retrieve joint limits and info from PyBullet model.
        """
        if not self.is_pybullet_connected:
            print("PyBullet not connected. Cannot get joint limits.")
            return

        joints_limits_l, joints_limits_u, joints_ranges, joints_rest_poses = [], [], [], []
        joint_names, link_names, joint_indices, joint_name_to_index = [], [], [], {}
        end_effector_index = -1

        for jid in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, jid)
            q_index = joint_info[3]
            joint_name_bytes = joint_info[1]
            link_name_bytes = joint_info[12]

            if q_index > -1:  # Fixed joints have q_index -1
                # Decode names once
                decoded_joint_name = joint_name_bytes.decode("utf-8")
                decoded_link_name = link_name_bytes.decode("utf-8")
                joint_id = joint_info[0]

                # Append the decoded name *once*
                joint_names.append(decoded_joint_name)
                link_names.append(decoded_link_name)
                joint_indices.append(joint_id)
                joint_name_to_index[decoded_joint_name] = joint_id # Populate map

                joints_limits_l.append(joint_info[8])
                joints_limits_u.append(joint_info[9])
                joints_ranges.append(joint_info[9] - joint_info[8])
                joints_rest_poses.append((joint_info[9] + joint_info[8]) / 2)

                if decoded_joint_name in self.right_arm:
                    self.right_arm_actuated.append(decoded_joint_name)

                if decoded_joint_name in self.left_arm:
                    self.left_arm_actuated.append(decoded_joint_name)
                
                if decoded_joint_name in self.right_gripper:
                    self.right_gripper_actuated.append(decoded_joint_name)

                if decoded_joint_name in self.left_gripper:
                    self.left_gripper_actuated.append(decoded_joint_name)

                if decoded_joint_name in self.head:
                    self.head_actuated.append(decoded_joint_name)

            if link_name_bytes.decode("utf-8") == 'endeffector': # Make sure your URDF defines this link name
                end_effector_index_r = jid
            if link_name_bytes.decode("utf-8") == 'endeffectol': # Make sure your URDF defines this link name
                end_effector_index_l = jid
            if link_name_bytes.decode("utf-8") == 'head': # Make sure your URDF defines this link name
                end_effector_index_h = jid
                

        self.joints_limits_l = joints_limits_l
        self.joints_limits_u = joints_limits_u
        self.joints_ranges = joints_ranges
        self.joints_rest_poses = joints_rest_poses
        self.joint_names = joint_names
        self.link_names = link_names
        self.joint_indices = joint_indices # List of movable joint indices
        self.joint_name_to_index = joint_name_to_index # Map name -> index
        self.end_effector_index_r = end_effector_index_r
        self.end_effector_index_l = end_effector_index_l
        self.end_effector_index_h = end_effector_index_h

    def calculate_ik(self, side, pos, ori_euler):
        """
        Calculates Inverse Kinematics for a given end-effector pose.
        Assumes the URDF defines the correct kinematic chain leading to end_effector_index.

        Args:
            pos (list/tuple): Target position [x, y, z].
            ori_euler (list/tuple): Target orientation in Euler angles [roll, pitch, yaw].

        Returns:
            list: The calculated joint angles (in radians) for the movable joints
                  as returned by PyBullet, or None if calculation fails. The length
                  should match the number of movable joints (DoFs).
        """
        if not self.is_pybullet_connected:
            print("PyBullet not connected. Cannot calculate IK.")
            return None
        
        self.end_effector_index = self.end_effector_index_r if side == 'right' else self.end_effector_index_l
        # self.ori = ori_euler if side == 'right' else [-abs(x) for x in ori_euler]
        self.ori = ori_euler
        if self.end_effector_index < 0:
            print("End effector index not found. Cannot calculate IK.")
            return None
        # Ensure internal lists are populated and consistent
        if not all([self.joints_limits_l, self.joints_limits_u, self.joints_ranges, self.joints_rest_poses, self.joint_indices]):
             print("Joint information not fully available. Cannot calculate IK.")
             return None

        try:
            # calculateInverseKinematics returns a list whose length is the number of
            # movable joints (DoFs) identified by PyBullet for the robot.
            ik_solution = p.calculateInverseKinematics(self.robot_id,
                                                     self.end_effector_index,
                                                     pos,
                                                     targetOrientation=p.getQuaternionFromEuler(self.ori),
                                                     lowerLimits=self.joints_limits_l,
                                                     upperLimits=self.joints_limits_u,
                                                     jointRanges=self.joints_ranges,
                                                     restPoses=self.joints_rest_poses,
                                                     maxNumIterations=300,
                                                     residualThreshold=0.0001)

            # Basic check if solution is valid
            if ik_solution is None:
                 print("IK calculation returned None.")
                 return None

            # Verify the length matches the number of movable joints found earlier
            # This is the crucial check based on the actual DoFs from the URDF
            if len(ik_solution) != len(self.joint_indices):
                 print(f"Error: IK solution length ({len(ik_solution)}) doesn't match number of movable joints ({len(self.joint_indices)}).")
                 return None

            return list(ik_solution) # Ensure it's a list

        except Exception as e:
            import traceback
            print(f"IK calculation failed with exception:")
            traceback.print_exc()
            return None

    @staticmethod
    def nicodeg2rad(nicojoints, nicodegrees):
        """Converts Nico-specific degrees to radians."""
        if isinstance(nicojoints, str):
            nicojoints = [nicojoints]
        if isinstance(nicodegrees, (int, float)):
            nicodegrees = [nicodegrees]

        rads = []
        for nicojoint, nicodegree in zip(nicojoints, nicodegrees):
            if nicojoint == 'r_wrist_z' or nicojoint == 'l_wrist_z':
                rad = deg2rad(nicodegree / 2)
            elif nicojoint == 'r_wrist_x' or nicojoint == 'l_wrist_x':
                rad = deg2rad(nicodegree / 4)
            else:
                rad = deg2rad(nicodegree)
            rads.append(rad)

        return rads[0] if len(rads) == 1 else rads

    @staticmethod
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
                degree = rad2deg(rad)
                scale_factors = {
                'r_wrist_z': 2,
                'l_wrist_z': 2,
                'r_wrist_x': 4,
                'l_wrist_x': 4
                }   
                nicodegree = degree * scale_factors.get(nicojoint, 1)
                nicodegrees_dict[nicojoint] = nicodegree
            else:
                 # Handle cases where IK might return non-numeric values or None
                 print(f"Warning: Non-numeric radian value '{rad}' for joint '{nicojoint}'. Skipping conversion.")
                 # Optionally add a placeholder like None to the dict, or just skip
                 # nicodegrees_dict[nicojoint] = None

        return nicodegrees_dict # Return the dictionary

    ### Non IK related methods ###

    def _set_pose_hardware(self, pose_values):
        """Internal: Sends a pose command to the physical robot."""
        if not self.is_robot_connected:
            print("Robot hardware not connected. Skipping hardware command.")
            return False

        index = 0
        # Ensure pose_values has the correct number of elements corresponding to INIT_POS keys
        keys = list(self.INIT_POS.keys())
        if len(pose_values) != len(keys):
             print(f"Error: Pose values length ({len(pose_values)}) does not match INIT_POS keys length ({len(keys)}).")
             return False

        try:
            for k in keys:
                # Make sure the joint name k exists in the robot's configuration
                # This check might depend on the specifics of the nicomotion library
                # if hasattr(self.robot, 'setAngle'): # Basic check
                self.robot.setAngle(k, float(pose_values[index]), self.SPEED)
                # else:
                #    print(f"Warning: Joint '{k}' not found or setAngle not available.")
                index += 1
            return True
        except Exception as e:
            print(f"Error setting hardware pose: {e}")
            return False

    def _set_hand_angles_hardware(self, thumb_z, thumb_x, index_x, middle_x):
         """Internal: Sets individual finger angles on the hardware."""
         if not self.is_robot_connected:
             print("Robot hardware not connected. Skipping hand command.")
             return False
         try:
             # Assuming right hand joints based on original script
             self.robot.setAngle('r_thumb_z', thumb_z, self.SPEEDF)
             self.robot.setAngle('r_thumb_x', thumb_x, self.SPEEDF)
             self.robot.setAngle('r_indexfinger_x', index_x, self.SPEEDF)
             self.robot.setAngle('r_middlefingers_x', middle_x, self.SPEEDF)
             return True
         except Exception as e:
             print(f"Error setting hand angles: {e}")
             return False


    def close_hand(self):
        """Closes the robot's hand (hardware)."""
        print("Closing hand...")
        if not self.is_robot_connected:
            print("Robot hardware not connected. Cannot close hand.")
            return
        # Original logic used a loop, simplified here for clarity
        # Final closed position angles might need adjustment
        success = self._set_hand_angles_hardware(thumb_z=180, thumb_x=20, index_x=20, middle_x=20)
        if success:
            print("Hand closed.")
        else:
            print("Failed to close hand.")
        time.sleep(0.5) # Short delay after command


    def open_hand(self):
        """Opens the robot's hand (hardware)."""
        print("Opening hand...")
        if not self.is_robot_connected:
            print("Robot hardware not connected. Cannot open hand.")
            return
        # Final open position angles might need adjustment
        success = self._set_hand_angles_hardware(thumb_z=180, thumb_x=-180, index_x=-180, middle_x=-180)
        if success:
            print("Hand opened.")
        else:
            print("Failed to open hand.")
        time.sleep(0.5) # Short delay after command

    def move_to_pose(self, pose_name):
        """Moves the robot to a predefined pose (e.g., 'reset', 'grasp', 'drop')."""
        if not self.is_robot_connected:
             print("Robot hardware not connected. Cannot move to pose.")
             return

        pose_map = {
            'reset': self.RESET_POSE,
            'grasp': self.GRASP_POSE,
            'drop': self.DROP_POSE
        }
        if pose_name.lower() in pose_map:
            print(f"Moving to {pose_name} pose...")
            success = self._set_pose_hardware(pose_map[pose_name.lower()])
            if success:
                print(f"Moved to {pose_name} pose.")
            else:
                print(f"Failed to move to {pose_name} pose.")
            time.sleep(self.DELAY)
        else:
            print(f"Error: Unknown pose name '{pose_name}'. Available: {list(pose_map.keys())}")


    def perform_grasp_sequence(self):
        """Executes the predefined grasp sequence by calling individual steps."""
        if not self.is_robot_connected:
            print("Robot hardware not connected. Cannot perform grasp sequence.")
            return

        print("\n--- Starting Grasp Sequence ---")
        self.perform_move()
        self.perform_grasp()
        # Optional: Lift arm after grasp - moved to perform_grasp
        self.perform_drop()
        print("--- Grasp Sequence Finished ---\n")

    def perform_move(self, target_angles_deg=None):
        """
        Moves the robot arm to the specified target angles (degrees).
        If target_angles_deg is None, moves to the predefined 'grasp' pose.

        Args:
            target_angles_deg (dict, optional): A dictionary mapping joint names
                                                to target angles in degrees.
                                                Defaults to None.
        """
        if not self.is_robot_connected:
            print("Robot hardware not connected. Cannot perform move.")
            return

        if target_angles_deg:
            print(f"Moving arm to IK solution angles...")
            success_count = 0
            total_joints = 0
            try:
                for joint_name, angle_deg in target_angles_deg.items():
                    # Check if the joint exists in the hardware interface (optional but good practice)
                    # This depends on the nicomotion API, assuming setAngle handles unknown joints gracefully or raises an error
                    if angle_deg is not None: # Check if conversion was successful
                        total_joints += 1
                        # print(f"  Setting {joint_name} to {angle_deg:.2f} deg") # Verbose logging
                        self.robot.setAngle(joint_name, float(angle_deg), self.SPEED)
                        success_count += 1
                    else:
                        print(f"  Skipping {joint_name} due to invalid angle.")

                if success_count == total_joints and total_joints > 0:
                    print("Arm moved to target angles.")
                elif total_joints == 0:
                     print("No valid target angles provided for move.")
                else:
                    print(f"Moved {success_count}/{total_joints} joints successfully.")

            except Exception as e:
                print(f"Error setting hardware angles during move: {e}")
        else:
            # Fallback to predefined grasp pose if no specific angles are given
            print("Moving to predefined grasp pose...")
            self.move_to_pose('grasp')
            print("Moved to predefined grasp pose.")

        time.sleep(self.DELAY) # Delay after the move completes

    def perform_grasp(self):
        """Closes the robot's hand and optionally lifts the arm."""
        print("Closing hand...")
        self.close_hand()
        print("Hand closed.")
        # Optional: Lift arm after grasp
        print("Lifting arm slightly...")
        try:
            self.robot.setAngle('r_elbow_y', 90, self.SPEED) # Example joint adjustment
            print("Arm lifted.")
            time.sleep(self.DELAY)
        except Exception as e:
             print(f"Could not lift arm: {e}")

    def perform_drop(self):
        """Moves the robot to the drop pose and opens the hand."""
        print("Moving to drop pose...")
        self.move_to_pose('drop')
        print("Moved to drop pose.")
        self.open_hand()
        self.move_to_pose('reset')
        time.sleep(self.DELAY)

    ## IK related methods ##

    
    def compute_ori_from_pos(self, pos):
        """
        Computes the third value of ori (yaw) as a linear function of pos[1]:
        pos[1] = -0.3 --> ori_yaw = 0
        pos[1] =  0.3 --> ori_yaw = 1.5
        """
        y = pos[1]
        # Clamp y to [-0.3, 0.3] to avoid extrapolation
        y = max(-0.3, min(0.3, y))
        # Linear mapping: ori_yaw = m * y + b
        m = (1.5 - 0) / (0.3 - (-0.3))  # (delta_ori) / (delta_y)
        b = 0 - m * (-0.3)
        ori_yaw = m * y + b
        return ori_yaw
    
    def move_arm(self, pos, ori, side, autozpos=False, autoori=False):
        """
        Moves the robot arm(s) to the specified target angles (degrees) based on the side.
        If side is 'both', calculates IK for left with negated y.

        Args:
            pos (list): Target position [x, y, z].
            ori (list): Target orientation [roll, pitch, yaw].
            side (str): Specify 'left', 'right', or 'both' to move the respective arm(s).
            autozpos (bool): If True, automatically set pos[2] using calculate_z.
            autoori (bool): If True, automatically set ori[2] using compute_ori_from_pos.
        """
        sides = []
        positions = {}

        # Copy to avoid modifying input lists
        pos = list(pos)
        ori = list(ori)

        if autozpos:
            if side.lower() == 'left':
                pos[2] = self.lefthand_rbf_z(pos[0], pos[1]) - 0.002
            else:
                # pos[2] = calculate_z(pos[0], pos[1]) + 0.04
                pos[2] = self.righthand_rbf_z(pos[0], pos[1]) - 0.001

        if autoori:
            if side.lower() == 'left':
                ori[2] = self.lefthand_rbf_yaw(pos[0], pos[1])
            else:
                # ori[2] = self.compute_ori_from_pos(pos)
                ori[2] = self.righthand_rbf_yaw(pos[0], pos[1])

        ik_solution_nico_deg = self.rad2nicodeg(self.joint_names, self.calculate_ik(side, pos, ori))

        if not self.is_robot_connected:
            print("Robot hardware not connected. Cannot move arm.")
            return

        if not ik_solution_nico_deg:
            print("No valid IK solution provided. Cannot move arm.")
            return

        # Determine which arm to filter
        if side.lower() == 'right':
            arm_actuated = self.right_arm_actuated
        elif side.lower() == 'left':
            arm_actuated = self.left_arm_actuated
        else:
            print("Invalid side specified. Use 'left' or 'right'.")
            return

        # Filter the IK solution
        filtered_solution = {
            joint: angle for joint, angle in ik_solution_nico_deg.items()
            if joint in arm_actuated
        }

        if not filtered_solution:
            print(f"No valid joints found for the {side} arm in the IK solution.")
            return

        print(f"Moving {side} arm to filtered IK solution angles...")
        success_count = 0
        total_joints = len(filtered_solution)

        try:
            for joint_name, angle_deg in filtered_solution.items():
                if angle_deg is not None:  # Ensure the angle is valid
                    # Execute the movement command
                    self.robot.setAngle(joint_name, float(angle_deg), self.speed)
                    success_count += 1
                    print(f"  Set {joint_name} to {angle_deg:.2f} degrees.")
                else:
                    print(f"  Skipping {joint_name} due to invalid angle.")
            time.sleep(self.delay)  # Delay after the move completes
            if success_count == total_joints:
                print(f"{side.capitalize()} arm successfully moved to all target angles.")
            else:
                print(f"Moved {success_count}/{total_joints} joints successfully.")
        except Exception as e:
            print(f"Error moving {side} arm: {e}")
    
    def move_both_arms(self, pos, ori):
        """
        Moves both arms to the specified target angles (degrees) simultaneously.
        Calculates IK for left with negated y, and for right as given.

        Args:
            pos (list): Target position [x, y, z].
            ori (list): Target orientation [roll, pitch, yaw].
        """
        # Prepare positions for both arms
        pos_left = [pos[0], -pos[1], pos[2]]
        pos_right = pos

        # Calculate IK and convert to Nico degrees
        ik_left_deg = self.rad2nicodeg(
            self.joint_names,
            self.calculate_ik('left', pos_left, ori)
        )
        ik_right_deg = self.rad2nicodeg(
            self.joint_names,
            self.calculate_ik('right', pos_right, ori)
        )

        if not self.is_robot_connected:
            print("Robot hardware not connected. Cannot move arms.")
            return

        if not ik_left_deg or not ik_right_deg:
            print("No valid IK solution provided for both arms. Cannot move arms.")
            return

        # Filter for actuated joints
        left_filtered = {joint: angle for joint, angle in ik_left_deg.items() if joint in self.left_arm_actuated}
        right_filtered = {joint: angle for joint, angle in ik_right_deg.items() if joint in self.right_arm_actuated}

        # Ensure both have the same number of joints for simultaneous movement
        min_len = min(len(left_filtered), len(right_filtered))
        left_joints = list(left_filtered.items())
        right_joints = list(right_filtered.items())

        print("Moving both arms to filtered IK solution angles simultaneously...")
        success_count = 0

        try:
            for i in range(min_len):
                l_joint, l_angle = left_joints[i]
                r_joint, r_angle = right_joints[i]
                if l_angle is not None:
                    self.robot.setAngle(l_joint, float(l_angle), self.speed)
                    print(f"  Set {l_joint} to {l_angle:.2f} degrees.")
                if r_angle is not None:
                    self.robot.setAngle(r_joint, float(r_angle), self.speed)
                    print(f"  Set {r_joint} to {r_angle:.2f} degrees.")
                success_count += 1
            time.sleep(self.delay)
            print(f"Both arms successfully moved to {success_count} joint positions.")
        except Exception as e:
            print(f"Error moving both arms: {e}")

    def move_both_arms_head(self, pos, ori):
        """
        Moves both arms to the specified target angles (degrees) simultaneously.
        Calculates IK for left with negated y, and for right as given.

        Args:
            pos (list): Target position [x, y, z].
            ori (list): Target orientation [roll, pitch, yaw].
        """
        # Prepare positions for both arms
        pos_left = [pos[0], -pos[1], pos[2]]
        pos_right = pos

        # Calculate IK and convert to Nico degrees
        ik_left_deg = self.rad2nicodeg(
            self.joint_names,
            self.calculate_ik('left', pos_left, ori)
        )
        ik_right_deg = self.rad2nicodeg(
            self.joint_names,
            self.calculate_ik('right', pos_right, ori)
        )

        ik_head_deg = self.looking_ik(pos)

        filtered_solution = {
            joint: angle for joint, angle in ik_head_deg.items()
            if joint in self.head_actuated}

        if not self.is_robot_connected:
            print("Robot hardware not connected. Cannot move arms.")
            return

        if not ik_left_deg or not ik_right_deg:
            print("No valid IK solution provided for both arms. Cannot move arms.")
            return

        # Filter for actuated joints
        left_filtered = {joint: angle for joint, angle in ik_left_deg.items() if joint in self.left_arm_actuated}
        right_filtered = {joint: angle for joint, angle in ik_right_deg.items() if joint in self.right_arm_actuated}
        head_filtered = {joint: angle for joint, angle in ik_head_deg.items() if joint in self.head_actuated}
        # Ensure both have the same number of joints for simultaneous movement
        min_len = min(len(left_filtered), len(right_filtered))
        left_joints = list(left_filtered.items())
        right_joints = list(right_filtered.items())
        head_joints = list(head_filtered.items())

        print("Moving both arms to filtered IK solution angles simultaneously...")
        success_count = 0

        try:
            for i in range(min_len):
                l_joint, l_angle = left_joints[i]
                r_joint, r_angle = right_joints[i]
                if l_angle is not None:
                    self.robot.setAngle(l_joint, float(l_angle), self.speed)
                    print(f"  Set {l_joint} to {l_angle:.2f} degrees.")
                if r_angle is not None:
                    self.robot.setAngle(r_joint, float(r_angle), self.speed)
                    print(f"  Set {r_joint} to {r_angle:.2f} degrees.")
                if i==0:
                    self.robot.setAngle(head_joints[0][0], float(head_joints[0][1]), self.speed)
                    self.robot.setAngle(head_joints[1][0], float(head_joints[1][1]), self.speed)
                    print(f"  Set {head_joints[0][0]} to {head_joints[0][1]:.2f} degrees.")
                    print(f"  Set {head_joints[1][0]} to {head_joints[1][1]:.2f} degrees.")
                success_count += 1
            time.sleep(self.delay)
            print(f"Both arms successfully moved to {success_count} joint positions.")
        except Exception as e:
            print(f"Error moving both arms: {e}")
    def move_gripper(self, side, value):
        """
        Closes the gripper (left or right) by setting all actuated joint angles to 0.

        Args:
            side (str): Specify 'left' or 'right' to close the respective gripper.
        """
        if not self.is_robot_connected:
            print("Robot hardware not connected. Cannot close gripper.")
            return

        if side.lower() == 'right':
            gripper_actuated = self.right_gripper_actuated
        elif side.lower() == 'left':
            gripper_actuated = self.left_gripper_actuated
        else:
            print("Invalid side specified. Use 'left' or 'right'.")
            return

        print(f"Moving {side} gripper...")
        try:
            for joint_name in gripper_actuated:
                if joint_name in 'r_thumb_z' or joint_name in 'l_thumb_z':
                    self.robot.setAngle(joint_name, self.opposite, self.speed)
                    print(f"  Set {joint_name} to opposite position.")    
                else:
                    self.robot.setAngle(joint_name, value, self.speed)
                    print(f"  Set {joint_name} to {value}.")
            time.sleep(self.delay) # Delay after the move completes
            print(f"{side.capitalize()} gripper moved.")
        except Exception as e:
            print(f"Error moving {side} gripper: {e}")
    
    def close_gripper(self, side):
        """
        Closes the gripper (left or right) by setting all actuated joint angles to 0.

        Args:
            side (str): Specify 'left' or 'right' to close the respective gripper.
        """
        if not self.is_robot_connected:
            print("Robot hardware not connected. Cannot close gripper.")
            return

        if side.lower() == 'right':
            gripper_actuated = self.right_gripper_actuated
        elif side.lower() == 'left':
            gripper_actuated = self.left_gripper_actuated
        else:
            print("Invalid side specified. Use 'left' or 'right'.")
            return

        print(f"Closing {side} gripper...")
        try:
            for joint_name in gripper_actuated:
                if joint_name in 'r_thumb_z' or joint_name in 'l_thumb_z':
                    self.robot.setAngle(joint_name, self.opposite, self.speed)
                    print(f"  Set {joint_name} to opposite position.")    
                else:
                    self.robot.setAngle(joint_name, self.closed, self.speed)
                    print(f"  Set {joint_name} to close.")
            time.sleep(self.delay) # Delay after the move completes
            print(f"{side.capitalize()} gripper closed.")
        except Exception as e:
            print(f"Error closing {side} gripper: {e}")
    
    def close_finger(self, name):

        if not self.is_robot_connected:
            print("Robot hardware not connected. Cannot close gripper.")
            return

        print(f"Closing {name} finger...")
        try:
            self.robot.setAngle(name, self.closed , self.speed)
            print(f"  Set {joint_name} to close.")
            time.sleep(self.delay) # Delay after the move completes
            print(f"{name.capitalize()} finger closed.")
        except Exception as e:
            print(f"Error closing {name} name: {e}")


    def open_gripper(self, side):
        """
        Closes the gripper (left or right) by setting all actuated joint angles to 0.

        Args:
            side (str): Specify 'left' or 'right' to close the respective gripper.
        """
        if not self.is_robot_connected:
            print("Robot hardware not connected. Cannot close gripper.")
            return

        if side.lower() == 'right':
            gripper_actuated = self.right_gripper_actuated
        elif side.lower() == 'left':
            gripper_actuated = self.left_gripper_actuated
        else:
            print("Invalid side specified. Use 'left' or 'right'.")
            return

        print(f"Opening {side} gripper...")
        try:
            for joint_name in gripper_actuated:
                if joint_name in 'r_thumb_z' or joint_name in 'l_thumb_z':
                    self.robot.setAngle(joint_name, self.opposite, self.speed)
                    print(f"  Set {joint_name} to opposite postion.")    
                else:
                    self.robot.setAngle(joint_name, self.open, self.speed-0.01)
                    print(f"  Set {joint_name} to open.")
            time.sleep(self.delay) # Delay after the move completes
        except Exception as e:
            print(f"Error opening {side} gripper: {e}")
    
    def point_gripper(self, side):
        """
        Closes the gripper (left or right) by setting all actuated joint angles to 0.

        Args:
            side (str): Specify 'left' or 'right' to close the respective gripper.
        """
        if not self.is_robot_connected:
            print("Robot hardware not connected. Cannot close gripper.")
            return

        if side.lower() == 'right':
            gripper_actuated = self.right_gripper_actuated
        elif side.lower() == 'left':
            gripper_actuated = self.left_gripper_actuated
        else:
            print("Invalid side specified. Use 'left' or 'right'.")
            return

        print(f"Opening {side} gripper...")
        try:
            for joint_name in gripper_actuated:
                if joint_name in 'r_thumb_z' or joint_name in 'l_thumb_z':
                    self.robot.setAngle(joint_name, self.opposite, self.speed)
                    print(f"  Set {joint_name} to opposite postion.")    
                if joint_name in 'r_indexfinger_x' or joint_name in 'l_indexfinger_x':
                    self.robot.setAngle(joint_name, self.open, self.speed)
                    print(f"  Set {joint_name} to opposite postion.")   
                else:
                    self.robot.setAngle(joint_name, self.closed, self.speed-0.01)
                    print(f"  Set {joint_name} to open.")
            time.sleep(self.delay) # Delay after the move completes
        except Exception as e:
            print(f"Error opening {side} gripper: {e}")

    def open_finger(self, name):

        if not self.is_robot_connected:
            print("Robot hardware not connected. Cannot close gripper.")
            return

        print(f"Opening {name} finger...")
        try:
            self.robot.setAngle(name, self.open , self.speed)
            print(f"  Set {joint_name} to close.")
            time.sleep(self.delay) # Delay after the move completes
            print(f"{side.capitalize()} gripper opened.")
        except Exception as e:
            print(f"Error opening {name} finger: {e}")
    
    def get_look_at_quaternion(self, eye_pos, target_pos):
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
        eye_pos = array(eye_pos)
        target_pos = array(target_pos)
        
        # Calculate the direction vector from the eye to the target
        direction = target_pos - eye_pos
        
        # Handle case where the target is at the same position as the eye
        if linalg.norm(direction) < 1e-6:
            return [0, 0, 0, 1] # Return identity quaternion (no rotation)
            
        # Calculate Yaw (rotation around Z-axis)
        # This determines the left-right direction
        yaw = arctan2(direction[1], direction[0])
        
        # Calculate Pitch (rotation around Y-axis)
        # This determines the up-down direction
        # The horizontal distance is the length of the projection onto the XY plane
        horizontal_dist = sqrt(direction[0]**2 + direction[1]**2)
        pitch = arctan2(-direction[2], horizontal_dist)
        
        # We assume Roll (rotation around X-axis) is 0 for a simple "look at"
        roll = 0
        
        # Convert the Euler angles (roll, pitch, yaw) to a quaternion
        return p.getQuaternionFromEuler([roll, pitch, yaw])

    def looking_ik(self, pos):
        head_link_state = p.getLinkState(self.robot_id, self.end_effector_index_h)
        head_pos = head_link_state[0]
                
        # Calculate the look-at quaternion
        target_orientation_quat = self.get_look_at_quaternion(head_pos, pos)
        
        ik_solution = p.calculateInverseKinematics(self.robot_id,
                                                     self.end_effector_index_h,
                                                     pos,
                                                     target_orientation_quat,
                                                     lowerLimits=self.joints_limits_l,
                                                     upperLimits=self.joints_limits_u,
                                                     jointRanges=self.joints_ranges,
                                                     restPoses=self.joints_rest_poses,
                                                     maxNumIterations=300,
                                                     residualThreshold=0.0001)
        
        return self.rad2nicodeg(self.head_actuated,ik_solution)
    
    def look_at(self, pos):

        if not self.is_robot_connected:
            print("Robot hardware not connected. Cannot close gripper.")
            return

        ik_solution_nico_deg = self.looking_ik(pos)

        filtered_solution = {
            joint: angle for joint, angle in ik_solution_nico_deg.items()
            if joint in self.head_actuated}

        
        for joint_name, angle_deg in filtered_solution.items():
            if angle_deg is not None:  # Ensure the angle is valid
                # Execute the movement command
                self.robot.setAngle(joint_name, float(angle_deg), self.speed)
                #success_count += 1
                print(f"  Set {joint_name} to {angle_deg:.2f} degrees.")
            else:
                print(f"  Skipping {joint_name} due to invalid angle.")
        time.sleep(self.delay/2)  # Delay after the move completes

    def pick_object(self, pos, ori, side, autozpos=False, autoori=False):
        if autozpos:
            if side.lower() == 'left':
                pos[2] = self.lefthand_rbf_z(pos[0], pos[1]) - 0.002
            else:
                pos[2] = self.righthand_rbf_z(pos[0], pos[1]) - 0.001
        self.move_arm([pos[0],pos[1],pos[2]+0.12], ori, side, autoori=autoori)
        self.move_arm(pos, ori, side, autoori=autoori)
        self.close_gripper(side)
        self.move_arm([pos[0],pos[1],pos[2]+0.12], ori, side, autoori=autoori) # Close right gripper

    def approach_object(self, pos, ori, side):
        self.move_arm([pos[0],pos[1],pos[2]+0.05], ori, side)
        self.move_arm(pos, ori, side)
        self.close_gripper(side)
        self.move_arm([pos[0],pos[1],pos[2]+0.05], ori, side) # Close right gripper

    def place_object(self, pos, ori, side):
        self.move_arm([pos[0],pos[1],pos[2]+0.15], ori, side)
        self.move_arm(pos, ori, side)
        self.open_gripper(side)
        self.move_arm([pos[0],pos[1],pos[2]+0.15], ori, side)
    
    def init_position(self, pos, ori, side):
        self.move_arm(pos, ori, side)
        self.open_gripper(side)
    
    def filter_arm_ik_solution(self, ik_solution_nico_deg, side):
        """
        Filters the IK solution to include only the joints in right_arm_actuated or left_arm_actuated.

        Args:
            ik_solution_nico_deg (dict): Dictionary of joint names and their angles.
            side (str): Specify 'left' or 'right' to filter for the respective arm.

        Returns:
            dict: Filtered dictionary containing only the joints in the specified arm.
        """
        if not ik_solution_nico_deg:
            print("No IK solution provided.")
            return {}

        if side.lower() == 'right':
            arm_actuated = self.right_arm_actuated
        elif side.lower() == 'left':
            arm_actuated = self.left_arm_actuated
        else:
            print("Invalid side specified. Use 'left' or 'right'.")
            return {}

        filtered_solution = {
            joint: angle for joint, angle in ik_solution_nico_deg.items()
            if joint in arm_actuated
        }

        print(f"Filtered IK Solution ({side.capitalize()} Arm Actuated):")
        for joint, angle in filtered_solution.items():
            print(f"{joint}: {angle}")

        return filtered_solution
    

    def get_real_joint_angles(self):
        """Reads current joint angles from the hardware."""
        if not self.is_robot_connected:
            print("Robot hardware not connected. Cannot read joint angles.")
            return None

        last_position = {}
        try:
            # Assuming getAngle works for reading too, or use a specific read method if available
            for k in self.INIT_POS.keys():
                 # Check if the robot object has the getAngle method and the joint exists
                 # This check depends heavily on the nicomotion library's API
                 if hasattr(self.robot, 'getAngle'): # Basic check
                     actual = self.robot.getAngle(k)
                     last_position[k] = actual
                 # else:
                 #    print(f"Warning: Joint '{k}' not found or getAngle not available.")
            print("Current hardware joint angles:", last_position)
            return last_position
        except Exception as e:
            print(f"Error reading hardware joint angles: {e}")
            return None

    def set_pose_sim(self, target_angles_rad):
        """Sets the robot's pose in the PyBullet simulation using radian values."""
        if not self.is_pybullet_connected:
            print("PyBullet not connected. Cannot set simulation pose.")
            return
        # Check if the number of target angles matches the number of movable joints
        if len(self.joint_indices) != len(target_angles_rad):
             print(f"Error setting sim pose: Mismatch between movable joint indices ({len(self.joint_indices)}) and target angles ({len(target_angles_rad)})")
             return

        print("Setting robot pose in simulation...")
        # Iterate through the movable joint indices and set their state
        for i, joint_index in enumerate(self.joint_indices):
            target_angle = target_angles_rad[i]
            # Use resetJointState for immediate effect in visualization
            p.resetJointState(self.robot_id, joint_index, target_angle)
            # Optional: Print joint name if available
            # joint_name = self.joint_names[i] # Assuming self.joint_names is correctly populated and ordered
            # print(f"  Joint index {joint_index} set to {target_angle:.4f} rad")
        print("Simulation pose set.")

    def disconnect(self):
        """Disconnects from PyBullet and potentially cleans up hardware resources."""
        print("Disconnecting...")
        if self.is_pybullet_connected:
            try:
                p.disconnect(self.physics_client)
                self.is_pybullet_connected = False
                print("Disconnected from PyBullet.")
            except Exception as e:
                print(f"Error disconnecting PyBullet: {e}")

        if self.is_robot_connected and hasattr(self.robot, 'stop'):
             try:
                 # Add any necessary hardware cleanup/shutdown commands here
                 # e.g., self.robot.stop() or similar, depending on nicomotion API
                 print("Stopped robot hardware interface (if applicable).")
             except Exception as e:
                 print(f"Error stopping robot hardware: {e}")
        print("Cleanup complete.")
        sys.exit()  # Terminate the program after cleanup

    
