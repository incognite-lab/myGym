import os, inspect
import pkg_resources
import pybullet
import numpy as np
import math

currentdir = pkg_resources.resource_filename("myGym", "envs")
repodir = pkg_resources.resource_filename("myGym", "")


class Robot:
    """
    Robot class for control of robot environment interaction

    Parameters:
        :param robot: (string) Type of robot to train in the environment (kuka, panda, ur3, ...)
        :param position: (list) Position of the robot's base link in the coordinate frame of the environment ([x,y,z])
        :param orientation: (list) Orientation of the robot's base link in the coordinate frame of the environment (Euler angles [x,y,z])
        :param end_effector_index: (int) Index of the robot's end-effector link. For myGym prepared robots this is assigned automatically.  
        :param gripper_index: (int) Index of the robot's gripper link. For myGym prepared robots this is assigned automatically.
        :param init_joint_poses: (list) Configuration in which robot will be initialized in the environment. Specified either in joint space as list of joint poses or in the end-effector space as [x,y,z] coordinates.
        :param robot_action: (string) Mechanism of robot control (absolute, step, joints)
        :param use_fixed_gripper_orn: (bool) Whether to fix robot's end-effector orientation or not
        :param gripper_orn: (list) Orientation of gripper in Euler angles for the fixed_gripper_orn option
        :param dimension_velocity: (float) Maximum allowed velocity for robot movements in individual x,y,z axis
        :param max_velocity: (float) Maximum allowed velocity for robot movements. Should be adjusted in case of sim2real scenario.
        :param max_force: (float) Maximum allowed force reached by individual joint motor. Should be adjusted in case of sim2real scenario.
        :param pybullet_client: Which pybullet client the environment should refere to in case of parallel existence of multiple instances of this environment
    """
    def __init__(self,
                 robot='kuka',
                 position=[-0.1, 0, 0.07], orientation=[0, 0, 0],
                 end_effector_index=None, gripper_index=6, 
                 init_joint_poses=None,
                 robot_action="step",
                 use_fixed_gripper_orn=False,
                 gripper_orn=[0, -math.pi, 0],
                 dimension_velocity = 0.5,
                 max_velocity = 10.,
                 max_force = 500.,
                 pybullet_client=None):

        self.p = pybullet_client
        self.robot_dict =   {'kuka': {'path': '/envs/robots/kuka_magnetic_gripper_sdf/kuka_magnetic_gripper.sdf', 'position': np.array([0.0, 0.0, -0.041]), 'orientation': [0.0, 0.0, 1*np.pi]},
                             'kuka_push': {'path': '/envs/robots/kuka_magnetic_gripper_sdf/kuka_push_gripper.urdf', 'position': np.array([0.0, 0.0, -0.041]), 'orientation': [0.0, 0.0, 0*np.pi]},
                             'panda': {'path': '/envs/robots/franka_emika/panda/urdf/panda.urdf', 'position': np.array([0.0, 0.0, -0.04])},
                             'jaco': {'path': '/envs/robots/jaco_arm/jaco/urdf/jaco_robotiq.urdf', 'position': np.array([0.0, 0.0, -0.041])},
                             'jaco_fixed': {'path': '/envs/robots/jaco_arm/jaco/urdf/jaco_robotiq_fixed.urdf', 'position': np.array([0.0, 0.0, -0.041])},
                             'reachy': {'path': '/envs/robots/pollen/reachy/urdf/reachy.urdf', 'position': np.array([0.0, 0.0, 0.32]), 'orientation': [0.0, 0.0, 0.0]},
                             'leachy': {'path': '/envs/robots/pollen/reachy/urdf/leachy.urdf', 'position': np.array([0.0, 0.0, 0.32]), 'orientation': [0.0, 0.0, 0.0]},
                             'reachy_and_leachy': {'path': '/envs/robots/pollen/reachy/urdf/reachy_and_leachy.urdf', 'position': np.array([0.0, 0.0, 0.32]), 'orientation': [0.0, 0.0, 0.0]},
                             'gummi': {'path': '/envs/robots/gummi_arm/urdf/gummi.urdf', 'position': np.array([0.0, 0.0, 0.021]), 'orientation': [0.0, 0.0, 0.5*np.pi]},
                             'gummi_fixed': {'path': '/envs/robots/gummi_arm/urdf/gummi_fixed.urdf', 'position': np.array([-0.1, 0.0, 0.021]), 'orientation': [0.0, 0.0, 0.5*np.pi]},
                             'ur3': {'path': '/envs/robots/universal_robots/urdf/ur3.urdf', 'position': np.array([0.0, -0.02, -0.041]), 'orientation': [0.0, 0.0, 0.0]},
                             'ur5': {'path': '/envs/robots/universal_robots/urdf/ur5.urdf', 'position': np.array([0.0, -0.03, -0.041]), 'orientation': [0.0, 0.0, 0.0]},
                             'ur10': {'path': '/envs/robots/universal_robots/urdf/ur10.urdf', 'position': np.array([0.0, -0.04, -0.041]), 'orientation': [0.0, 0.0, 0.0]},
                             'yumi': {'path': '/envs/robots/abb/yumi/urdf/yumi.urdf', 'position': np.array([0.0, 0.15, -0.042]), 'orientation': [0.0, 0.0, 0.0]},
                             'human': {'path': '/envs/robots/real_hands/humanoid_with_hands_fixed.urdf', 'position': np.array([0.0, 1.5, 0.45]), 'orientation': [0.0, 0.0, 1.5*np.pi]}
                            }

        self.robot_path = self.robot_dict[robot]['path']
        self.position = np.array(position) + self.robot_dict[robot].get('position',np.zeros(len(position)))
        self.orientation = np.array(orientation) + self.robot_dict[robot].get('orientation',np.zeros(len(orientation)))
        self.orientation = self.p.getQuaternionFromEuler(self.orientation)
        self.name = robot + '_gripper'

        self.max_velocity = max_velocity
        self.max_force = max_force

        self.end_effector_index = end_effector_index
        self.gripper_index = gripper_index
        self.use_fixed_gripper_orn = use_fixed_gripper_orn
        self.gripper_orn = self.p.getQuaternionFromEuler(gripper_orn)
        self.dimension_velocity = dimension_velocity
        self.motor_names = []
        self.motor_indices = []
        self.robot_action = robot_action
        self.magnetized_objects = {}

        self._load_robot()
        self.num_joints = self.p.getNumJoints(self.robot_uid)
        self._set_motors()
        self.joints_limits, self.joints_ranges, self.joints_rest_poses = self.get_joints_limits()
        if len(init_joint_poses) == 3:
            joint_poses = list(self._calculate_accurate_IK(init_joint_poses))
            self.init_joint_poses = joint_poses
        else:
            self.init_joint_poses = np.zeros((len(self.motor_names)))
        self.reset()

    def _load_robot(self):
        """
        Load SDF or URDF model of specified robot and place it in the environment to specified position and orientation
        """
        if self.robot_path[-3:] == 'sdf':
            objects = self.p.loadSDF(
                pkg_resources.resource_filename("myGym",
                                                self.robot_path))
            self.robot_uid = objects[0]
            self.p.resetBasePositionAndOrientation(self.robot_uid, self.position,
                                              self.orientation)
        else:
            self.robot_uid = self.p.loadURDF(
                pkg_resources.resource_filename("myGym",
                                                self.robot_path),
                self.position, self.orientation, useFixedBase=True, flags=(self.p.URDF_USE_SELF_COLLISION))
        for jid in range(self.p.getNumJoints(self.robot_uid)):
            self.p.changeDynamics(self.robot_uid, jid,  collisionMargin=0., contactProcessingThreshold=0.0, ccdSweptSphereRadius=0)
        # if 'jaco' in self.name: #@TODO jaco gripper has closed loop between finger and finger_tip that is not respected by the simulator
        #     self.p.createConstraint(self.robot_uid, 11, self.robot_uid, 15, self.p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 0])
        #     self.p.createConstraint(self.robot_uid, 13, self.robot_uid, 17, self.p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 0])

    def _set_motors(self):
        """
        Identify motors among all joints (fixed joints aren't motors). Identify index of gripper and end-effector link among all links. Uses data from robot model.
        """
        for i in range(self.num_joints):
            joint_info = self.p.getJointInfo(self.robot_uid, i)
            if joint_info[12].decode("utf-8") == 'gripper':
                self.gripper_index = i
            if joint_info[12].decode("utf-8") == 'end_effector':
                self.end_effector_index = i
            q_index = joint_info[3]
            if q_index > -1: # Fixed joints have q_index -1
                self.motor_names.append(str(joint_info[1]))
                self.motor_indices.append(i)
        if self.end_effector_index == None:
            self.end_effector_index = self.gripper_index
        print("Gripper index is: " + str(self.gripper_index))
        print("End effector index is: " + str(self.end_effector_index))

    def reset(self, random_robot=False):
        """
        Reset joint motors

        Parameters:
            :param random_robot: (bool) Whether the joint positions after reset should be randomized or equal to initial values.
        """
        if random_robot:
            self.reset_random()
        else:
            self.reset_joints(self.init_joint_poses)

    def reset_random(self):
        """
        Reset joint motors to random values within joint ranges
        """
        joint_poses = []
        for jid in range(len(self.motor_indices)):
            joint_poses.append(np.random.uniform(self.joints_limits[0][jid], self.joints_limits[1][jid]))
        self.reset_joints(joint_poses)

    def reset_up(self):
        """
        Reset joint motors to zero values (robot in upright position)
        """
        self.reset_joints(np.zeros((len(self.motor_indices))))

    def reset_joints(self, joint_poses):
        """
        Reset joint motors to requested values

        Parameters:
            :param positions: (list) Values for individual joint motors
        """
        joint_poses = np.clip(joint_poses, self.joints_limits[0], self.joints_limits[1])
        for jid in range(len(self.motor_indices)):
            self.p.resetJointState(self.robot_uid, self.motor_indices[jid], joint_poses[jid])
        self._run_motors(joint_poses)

    def get_joints_limits(self):
        """
        Identify limits, ranges and rest poses of individual robot joints. Uses data from robot model.

        Returns:
            :return [joints_limits_l, joints_limits_u]: (list) Lower and upper limits of all joints
            :return joints_ranges: (list) Ranges of movement of all joints
            :return joints_rest_poses: (list) Rest poses of all joints
        """
        joints_limits_l = []
        joints_limits_u = []
        joints_ranges = []
        joints_rest_poses = []
        for jid in self.motor_indices:
            joint_info = self.p.getJointInfo(self.robot_uid, jid)
            joints_limits_l.append(joint_info[8])
            joints_limits_u.append(joint_info[9])
            joints_ranges.append(joint_info[9] - joint_info[8])
            joints_rest_poses.append((joint_info[9] + joint_info[8])/2)
        return [joints_limits_l, joints_limits_u], joints_ranges, joints_rest_poses

    def get_action_dimension(self):
        """
        Get dimension of action data, based on robot control mechanism

        Returns:
            :return dimension: (int) The dimension of action data
        """
        if self.robot_action == "joints":
            return len(self.motor_indices)
        else:
            return 3

    def get_observation_dimension(self):
        """
        Get dimension of robot part of observation data, based on robot task and rewatd type

        Returns:
            :return dimension: (int) The dimension of observation data
        """
        return len(self.get_observation())

    def get_observation(self):
        """
        Get robot part of observation data

        Returns: 
            :return observation: (list) Position of end-effector link (center of mass)
        """
        observation = []
        state = self.p.getLinkState(self.robot_uid, self.end_effector_index)
        pos = state[0]
        orn = self.p.getEulerFromQuaternion(state[1])

        observation.extend(list(pos))
        return observation

    def get_position(self):
        """
        Get position of robot's end-effector link

        Returns: 
            :return position: (list) Position of end-effector link (center of mass)
        """
        return self.p.getLinkState(self.robot_uid, self.end_effector_index)[0]

    def _run_motors(self, joint_poses):
        """
        Move joint motors towards desired joint poses respecting robot's dynamics

        Parameters:
            :param joint_poses: (list) Desired poses of individual joints
        """
        joint_poses = np.clip(joint_poses, self.joints_limits[0], self.joints_limits[1])
        self.joints_state = []
        for i in range(len(self.motor_indices)):
            self.p.setJointMotorControl2(bodyUniqueId=self.robot_uid,
                                    jointIndex=self.motor_indices[i],
                                    controlMode=self.p.POSITION_CONTROL,
                                    targetPosition=joint_poses[i],
                                    force=self.max_force,
                                    maxVelocity=self.max_velocity,
                                    positionGain=0.7,
                                    velocityGain=0.3)
            #self.joints_state.append(self.p.getJointState(self.robot_uid, self.motor_indices[i])[0])
        #print('poses',joint_poses)
        #print('state',self.joints_state)
        self.end_effector_pos = self.p.getLinkState(self.robot_uid, self.end_effector_index)[0]
        
    def _calculate_joint_poses(self, end_effector_pos):
        """
        Calculate joint poses corresponding to desired position of end-effector. Uses inverse kinematics.

        Parameters:
            :param end_effector_pos: (list) Desired position of end-effector in environment [x,y,z]
        Returns:
            :return joint_poses: (list) Calculated joint poses corresponding to desired end-effector position
        """
        if (self.use_fixed_gripper_orn):
            joint_poses = self.p.calculateInverseKinematics(self.robot_uid,
                                                       self.gripper_index,
                                                       end_effector_pos,
                                                       self.gripper_orn,
                                                       lowerLimits=self.joints_limits[0],
                                                       upperLimits=self.joints_limits[1],
                                                       jointRanges=self.joints_ranges,
                                                       restPoses=self.joints_rest_poses)
        else:
            joint_poses = self.p.calculateInverseKinematics(self.robot_uid,
                                                       self.gripper_index,
                                                       end_effector_pos)
    
        joint_poses = np.clip(joint_poses, self.joints_limits[0], self.joints_limits[1])
        return joint_poses

    def _calculate_accurate_IK(self, end_effector_pos):
        """
        Calculate joint poses corresponding to desired position of end-effector. Uses accurate inverse kinematics (iterative solution).

        Parameters:
            :param end_effector_pos: (list) Desired position of end-effector in environment [x,y,z]
        Returns:
            :return joint_poses: (list) Calculated joint poses corresponding to desired end-effector position
        """
        thresholdPos = 0.01
        thresholdOrn = 0.01
        maxIter = 100
        closeEnough = False
        iter = 0
        while (not closeEnough and iter < maxIter):
            if (self.use_fixed_gripper_orn):
                joint_poses = self.p.calculateInverseKinematics(self.robot_uid,
                                                            self.gripper_index,
                                                            end_effector_pos,
                                                            self.gripper_orn)
            else:
                joint_poses = self.p.calculateInverseKinematics(self.robot_uid,
                                                            self.gripper_index,
                                                            end_effector_pos,
                                                            lowerLimits=self.joints_limits[0],
                                                            upperLimits=self.joints_limits[1],
                                                            jointRanges=self.joints_ranges,
                                                            restPoses=self.joints_rest_poses)
            joint_poses = np.clip(joint_poses, self.joints_limits[0], self.joints_limits[1])
            #reset the joint state (ignoring all dynamics, not recommended to use during simulation)
            for jid in range(len(self.motor_indices)):
                self.p.resetJointState(self.robot_uid, self.motor_indices[jid], joint_poses[jid])

            ls = self.p.getLinkState(self.robot_uid, self.gripper_index)
            newPos = ls[4] #world position of the URDF link frame
            newOrn = ls[5] #world orientation of the URDF link frame
            diffPos = np.linalg.norm(np.asarray(end_effector_pos)-np.asarray(newPos))
            if (self.use_fixed_gripper_orn):
                diffOrn = np.linalg.norm(np.asarray(self.gripper_orn)-np.asarray(newOrn))
            else:
                diffOrn = 0
            closeEnough = ((diffPos < thresholdPos) and (diffOrn < thresholdOrn))
            iter = iter + 1
        return joint_poses

    def apply_action_step(self, action):
        """
        Apply action command to robot using step control mechanism

        Parameters:
            :param action: (list) Desired action data
        """
        action = [i * self.dimension_velocity for i in action]
        des_end_effector_pos = np.add(self.end_effector_pos, action)
        joint_poses = list(self._calculate_joint_poses(des_end_effector_pos))
        self._run_motors(joint_poses)

    def apply_action_absolute(self, action):
        """
        Apply action command to robot using absolute control mechanism

        Parameters:
            :param action: (list) Desired action data
        """
        des_end_effector_pos = action
        joint_poses = self._calculate_joint_poses(des_end_effector_pos)
        self._run_motors(joint_poses)

    def apply_action_joints(self, action):
        """
        Apply action command to robot using joints control mechanism

        Parameters:
            :param action: (list) Desired action data
        """
        self._run_motors(action)
        
    def apply_action(self, action):
        """
        Apply action command to robot in simulated environment

        Parameters:
            :param action: (list) Desired action data
        """
        if self.robot_action == "step":
            self.apply_action_step(action)
        elif self.robot_action == "absolute":
            self.apply_action_absolute(action)
        elif self.robot_action == "joints":
            self.apply_action_joints(action)

    def magnetize_object(self, object):
        # Creates fixed joint between kuka gripper and object
        # TODO: Set constraint position
        # constraint_id = self.p.createConstraint(self.robot_uid, self.end_effector_index, object.uid, -1,
        #                     jointType=self.p.JOINT_FIXED,
        #                     jointAxis=[0, 0, 0],
        #                     parentFramePosition=[ 0.0, 0.082, -0.033], #[ 0.0, 0.0, 0.135]
        #                     childFramePosition=[-0.0, -0.0, -0.0])
        self.p.changeVisualShape(object.uid, -1, rgbaColor=[0, 255, 0, 1])
        #self.magnetized_objects[object] = constraint_id

    def release_object(self, object):
        self.p.removeConstraint(self.magnetized_objects[object])
        self.magnetized_objects.pop(object)

    def get_name(self):
        """
        Get name of robot

        Returns:
            :return name: (string) Name of robot
        """
        return self.name
