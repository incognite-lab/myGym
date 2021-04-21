import os, inspect
import pkg_resources
import pybullet
import numpy as np
import math
import matplotlib.pyplot as plt
from stable_baselines import results_plotter

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
                 max_velocity = None, #1
                 max_force = None, #300
                 timestep = 1/240.,
                 pybullet_client=None):

        self.p = pybullet_client
        self.robot_dict =   {'kuka': {'path': '/envs/robots/kuka_magnetic_gripper_sdf/kuka_magnetic_gripper.sdf', 'position': np.array([0.0, -0.05, -0.041]), 'orientation': [0.0, 0.0, 1*np.pi]},
                             'kuka_push': {'path': '/envs/robots/kuka_magnetic_gripper_sdf/kuka_push_gripper.sdf', 'position': np.array([0.0, 0.0, -0.041]), 'orientation': [0.0, 0.0, 1*np.pi]},
                             #'panda': {'path': '/envs/robots/franka_emika/panda_moveit/urdf/panda.urdf', 'position': np.array([0.0, 0.0, -0.04])},
                             'panda': {'path': '/envs/robots/franka_emika/panda_gazebo/robots/panda_arm_hand.urdf', 'position': np.array([0.0, 0.0, -0.04])},
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
        self.actions_history = []
        self.joints_state_history = []
        self.joints_velocity_history = []
        self.joints_torque_history = []

        self.max_velocity = max_velocity
        self.max_force = max_force
        self.timestep = timestep

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
        self.obsdim = 0 #len(self.motor_indices)*2
        self.action_dim = self.get_action_dimension()
        self.joints_limits, self.joints_ranges, self.joints_rest_poses, self.joints_max_force, self.joints_max_velo = self.get_joints_limits()
        self.last_action = self.joints_max_force
        if len(init_joint_poses) == 3:
            joint_poses = list(self._calculate_accurate_IK(init_joint_poses))
            self.init_joint_poses = joint_poses
        else:
            self.init_joint_poses = np.zeros((len(self.motor_names)))
        self.reset()

    def visualize_actions_over_steps(self, logdir, episode_steps, episode_number, timestep):
        """
        Plot and save a graph of action values assigned to individual steps during an episode. Call this method after the end of the episode.
        """
        save_dir = os.path.join(logdir, "actions")
        os.makedirs(save_dir, exist_ok=True)
        if episode_steps > 0:
            for joint in range(len(self.motor_indices)):
                results_plotter.EPISODES_WINDOW=50
                results_plotter.plot_curves([(np.arange(episode_steps),np.asarray(self.actions_history[-episode_steps:]).transpose()[joint])],'step','Step actions')
                plt.ylabel("action")
                plt.gcf().set_size_inches(8, 6)
                plt.savefig(save_dir + "/actions_over_steps_episode{}_{}.png".format(episode_number, joint))
                plt.close()

                if 'velo' in self.robot_action:
                    action_achieved = self.joints_velocity_history
                elif 'torque' in self.robot_action:
                    action_achieved = self.joints_torque_history
                else:    
                    action_achieved = self.joints_state_history

                joints_state_diff = np.abs(np.asarray(action_achieved) - np.asarray(self.actions_history))
                results_plotter.EPISODES_WINDOW=50
                results_plotter.plot_curves([(np.arange(episode_steps),joints_state_diff[-episode_steps:].transpose()[joint])],'step','Step actions diffs')
                plt.ylabel("action diff")
                plt.gcf().set_size_inches(8, 6)
                plt.savefig(save_dir + "/actions_diff_over_steps_episode{}_{}.png".format(episode_number, joint))
                plt.close()

            for joint in range(len(self.motor_indices)):
                results_plotter.EPISODES_WINDOW=50
                results_plotter.plot_curves([(np.arange(episode_steps),np.asarray(self.joints_velocity_history[-episode_steps:]).transpose()[joint])],'step','Step velocities achieved')
                plt.ylabel("velocities achieved")
                plt.gcf().set_size_inches(8, 6)
                plt.savefig(save_dir + "/velocities_achieved_over_steps_episode{}_{}.png".format(episode_number, joint))
                plt.close()

            joints_acc_history = (np.asarray(self.joints_velocity_history[1:]) - np.asarray(self.joints_velocity_history[:-1]))/timestep
            for joint in range(len(self.motor_indices)):
                results_plotter.EPISODES_WINDOW=50
                results_plotter.plot_curves([(np.arange(episode_steps-1),np.asarray(joints_acc_history[-episode_steps+1:]).transpose()[joint])],'step','Step acc achieved')
                plt.ylabel("acc achieved")
                plt.gcf().set_size_inches(8, 6)
                plt.savefig(save_dir + "/acc_achieved_over_steps_episode{}_{}.png".format(episode_number, joint))
                plt.close()

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
        if 'gripper' not in self.robot_action and 'torque_control' not in self.robot_action:
            self.motor_indices = [x for x in self.motor_indices if x < self.gripper_index]
        print("Gripper index is: " + str(self.gripper_index))
        print("End effector index is: " + str(self.end_effector_index))

    def reset(self, random_robot=False):
        """
        Reset joint motors

        Parameters:
            :param random_robot: (bool) Whether the joint positions after reset should be randomized or equal to initial values.
        """
        self.magnetized_objects = {}
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
        joints_limits_l, joints_limits_u, joints_ranges, joints_rest_poses, joints_max_force, joints_max_velo = [], [], [], [], [], []
        for jid in self.motor_indices:
            joint_info = self.p.getJointInfo(self.robot_uid, jid)
            joints_limits_l.append(joint_info[8])
            joints_limits_u.append(joint_info[9])
            joints_ranges.append(joint_info[9] - joint_info[8])
            joints_rest_poses.append((joint_info[9] + joint_info[8])/2)
            joints_max_force.append(self.max_force if self.max_force else joint_info[10])
            joints_max_velo.append(self.max_velocity if self.max_velocity else joint_info[11])
        return [joints_limits_l, joints_limits_u], joints_ranges, joints_rest_poses, joints_max_force, joints_max_velo

    def get_action_dimension(self):
        """
        Get dimension of action data, based on robot control mechanism

        Returns:
            :return dimension: (int) The dimension of action data
        """
        if self.robot_action in ["joints", "joints_step", "joints_gripper", "velo_step", "torque_step", "pybulletx"]:
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

    def get_orientation(self):
        """
        Get orientation of robot's end-effector link

        Returns:
            :return oriention: (list) Orientation of end-effector link (center of mass)
        """
        return self.p.getLinkState(self.robot_uid, self.end_effector_index)[1]

    def get_lin_velocity(self):
        """
        Get linear velocity of robot's end-effector link

        Returns:
            :return linear velocity: (list) Linear velocity of end-effector link (center of mass)
        """
        return self.p.getLinkState(self.robot_uid, self.end_effector_index, 1)[6]

    def get_ang_velocity(self):
        """
        Get angulat velocity of robot's end-effector link

        Returns:
            :return angular velocity: (list) Linear velocity of end-effector link (center of mass)
        """
        return self.p.getLinkState(self.robot_uid, self.end_effector_index, 1)[7]

    def get_joints_state(self):
        """
        Get position of robot's end-effector link

        Returns:
            :return position: (list) Position of end-effector link (center of mass)
        """
        state_pos = []
        state_velo = []
        for motor in self.motor_indices:
            joint_info = (self.p.getJointState(self.robot_uid, motor))
            state_pos.append(joint_info[0])
            state_velo.append(joint_info[1])
        return state_pos, state_velo

    def _run_motors(self, joint_poses):
        """
        Move joint motors towards desired joint poses respecting robot's dynamics

        Parameters:
            :param joint_poses: (list) Desired poses of individual joints
        """
        joint_poses = np.clip(joint_poses, self.joints_limits[0], self.joints_limits[1])
        self.actions_history.append(joint_poses)
        self.joints_state = []
        self.joints_velo = []
        self.joints_torque = []
        for i in range(len(self.motor_indices)):
            self.p.setJointMotorControl2(bodyUniqueId=self.robot_uid,
                                    jointIndex=self.motor_indices[i],
                                    controlMode=self.p.POSITION_CONTROL,
                                    targetPosition=joint_poses[i],
                                    targetVelocity=0,
                                    force=self.joints_max_force[i],
                                    maxVelocity=self.joints_max_velo[i],
                                    positionGain=0.1,
                                    velocityGain=1)
            joint_info = self.p.getJointState(self.robot_uid, self.motor_indices[i])
            self.joints_state.append(joint_info[0])
            self.joints_velo.append(joint_info[1])
            self.joints_torque.append(joint_info[3])
        self.joints_state_history.append(self.joints_state)
        self.joints_velocity_history.append(self.joints_velo)
        self.joints_torque_history.append(self.joints_torque)
        self.end_effector_pos = self.p.getLinkState(self.robot_uid, self.end_effector_index)[0]

    def _run_motors_velo(self, joint_velos):
        """
        Move joint motors towards desired joint velocities respecting robot's dynamics

        Parameters:
            :param joint_velos: (list) Desired velocities of individual joints
        """
        joint_velos = np.clip(joint_velos, -1*np.asarray(self.joints_max_velo), self.joints_max_velo)
        self.actions_history.append(joint_velos)
        self.joints_state = []
        self.joints_velo = []
        self.joints_torque = []
        for i in range(len(self.motor_indices)):
            self.p.setJointMotorControl2(bodyUniqueId=self.robot_uid,
                                    jointIndex=self.motor_indices[i],
                                    controlMode=self.p.VELOCITY_CONTROL,
                                    targetVelocity=joint_velos[i],
                                    force=self.joints_max_force[i],
                                    maxVelocity=self.joints_max_velo[i])
            joint_info = self.p.getJointState(self.robot_uid, self.motor_indices[i])
            self.joints_state.append(joint_info[0])
            self.joints_velo.append(joint_info[1])
            self.joints_torque.append(joint_info[3])
        self.joints_state_history.append(self.joints_state)
        self.joints_velocity_history.append(self.joints_velo)
        self.joints_torque_history.append(self.joints_torque)
        self.end_effector_pos = self.p.getLinkState(self.robot_uid, self.end_effector_index)[0]

    def _run_motors_torque(self, joint_torques):
        """
        Move joint motors by desired torques respecting robot's dynamics

        Parameters:
            :param joint_torques: (list) Desired torques of individual joints
        """
        joint_torques = np.clip(joint_torques.reshape(len(self.motor_indices),), -1*np.asarray(self.joints_max_force), self.joints_max_force)
        self.actions_history.append(joint_torques)
        self.joints_state = []
        self.joints_velo = []
        self.joints_torque = []

        # The magic that enables torque control
        self.p.setJointMotorControlArray(
            bodyIndex=self.robot_uid,
            jointIndices=self.motor_indices,
            controlMode=self.p.VELOCITY_CONTROL,
            forces=np.zeros(len(self.motor_indices)),
        )

        for i in range(len(self.motor_indices)):
            self.p.setJointMotorControl2(bodyUniqueId=self.robot_uid,
                                    jointIndex=self.motor_indices[i],
                                    controlMode=self.p.TORQUE_CONTROL,
                                    force=joint_torques[i])
            joint_info = self.p.getJointState(self.robot_uid, self.motor_indices[i])
            self.joints_state.append(joint_info[0])
            self.joints_velo.append(joint_info[1])
            self.joints_torque.append(joint_info[3])
        self.joints_state_history.append(self.joints_state)
        self.joints_velocity_history.append(self.joints_velo)
        self.joints_torque_history.append(self.joints_torque)
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
                                                       end_effector_pos,
                                                       self.gripper_orn,
                                                       lowerLimits=self.joints_limits[0],
                                                       upperLimits=self.joints_limits[1],
                                                       jointRanges=self.joints_ranges,
                                                       restPoses=self.joints_rest_poses)
        joint_poses = joint_poses[:len(self.motor_indices)]
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
        itera = 0
        while (not closeEnough and itera < maxIter):
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
            joint_poses = joint_poses[:len(self.motor_indices)]
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
            itera = itera + 1
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

    def apply_action_joints_step(self, action):
        """
        Apply action command to robot using joint-step control mechanism

        Parameters:
            :param action: (list) Desired action data
        """
        action = [i * self.dimension_velocity for i in action]
        joint_poses = np.add(self.joints_state, action)
        self._run_motors(joint_poses)

    def apply_action_velo_step(self, action):
        """
        Apply action command to robot using velocity-step control mechanism

        Parameters:
            :param action: (list) Desired action data
        """
        action = [i * self.dimension_velocity for i in action]
        _, joint_velos = self.get_joints_state()
        new_velo = np.add(joint_velos, action)
        new_velo = np.clip(new_velo, np.asarray(self.joints_max_velo)*-1, self.joints_max_velo)
        #joint_poses = np.add(joint_states, new_velo*self.timestep)
        #self._run_motors(joint_poses)
        self._run_motors_velo(new_velo)

    def apply_action_torque_step(self, action):
        """
        Apply action command to robot using torque-step control mechanism

        Parameters:
            :param action: (list) Desired action data
        """
        # action = [i * self.dimension_velocity * 50 for i in action]
        # new_action = np.add(self.last_action, action)
        # self.last_action = action
        # self._run_motors_torque(new_action)
        actions = []
        for i, ac in enumerate(action):
            actions.append(self.joints_max_force[i] * ac * 0.05)
        new_action = np.add(self.last_action, actions)
        self.last_action = actions
        self._run_motors_torque(new_action)

    def apply_action_pybulletx(self, action):
        """
        Apply action command to robot using pybulletx control mechanism

        Parameters:
            :param action: (list) Desired joint positions data
        """
        P_GAIN = 10
        joint_states, _ = self.get_joints_state()
        error = np.asarray(action) - np.asarray(joint_states)
        torque = error * P_GAIN
        self._run_motors_torque(torque)

    def apply_action_torque_control(self, action):
        """
        Apply action command to robot using torque control mechanism

        Parameters:
            :param action: (list) Desired end_effector positions data
        """
        joint_states, joint_velocities = self.get_joints_state()

        # Task-space controller parameters
        # stiffness gains
        P_pos = 10.
        P_ori = 1.
        # damping gains
        D_pos = 2.
        D_ori = 1.

        curr_pos = np.asarray(self.get_position())
        curr_ori = np.asarray(self.get_orientation())
        curr_vel = np.asarray(self.get_lin_velocity()).reshape([3, 1])
        curr_omg = np.asarray(self.get_ang_velocity()).reshape([3, 1])
        goal_pos = np.asarray(action)
        goal_ori = curr_ori #@TODO: change if control is made wrt orientation as well (not only e-e position)
        delta_pos = (goal_pos - curr_pos).reshape([3, 1]) # +  0.01 * pdot.reshape((-1, 1))
        #delta_ori = quatdiff_in_euler(curr_ori, goal_ori).reshape([3, 1])
        delta_ori = np.zeros((3,1))
        # TODO: limit speed when delta pos or ori are too big. we can scale the damping exponentially to catch to high deltas
        # Desired task-space force using PD law
        F = np.vstack([P_pos * (delta_pos), P_ori * (delta_ori)]) - \
            np.vstack([D_pos * (curr_vel), D_ori * (curr_omg)])
        # F = np.vstack([P_pos * (delta_pos), P_ori * (delta_ori)]) - \
        #     np.vstack([D_pos * (curr_vel-pdot.reshape((3,-1))), D_ori * (curr_omg- qdot.reshape(3,-1))])
        error = np.linalg.norm(delta_pos) # + np.linalg.norm(delta_ori)
        # print(error)
        # panda_robot equivalent: panda.jacobian(angles[optional]) or panda.zero_jacobian()
        Jt, Jr = self.p.calculateJacobian(self.robot_uid, self.end_effector_index, [0,0,0], joint_states, joint_velocities, [10]*(len(joint_states)))
        J = np.array([Jt,Jr]).reshape(-1, len(joint_states))
        # A = robot.joint_inertia_matrix()
        # A_inv = np.linalg.pinv(A)
        # LAMBDA = np.linalg.pinv(np.dot(np.dot(J,A_inv), J.T))
        # J_hash_T = np.dot(np.dot(LAMBDA, J), A_inv)
        # q_d = np.array(robot.q_d)
        # k_v = 5.0
        # k_vq = 1.0
        # D = (k_v - k_vq) * np.dot(J.T, np.dot(LAMBDA, J)) + k_vq * A
        # # Haken_0 = - np.dot(A, q_d)
        # Haken_0 = - np.dot(D, q_d)
        # tau_0 = np.dot(np.eye(7) - np.dot(J.T, J_hash_T), Haken_0)
        # joint torques to be commanded
        # tau = np.dot(J.T, F) + tau_0.reshape((-1,1)) +  robot.coriolis_comp().reshape(7,1)
        tau = np.dot(J.T, F) # + robot.coriolis_comp().reshape(7,1)

        self._run_motors_torque(tau)


    def apply_action(self, action):
        """
        Apply action command to robot in simulated environment

        Parameters:
            :param action: (list) Desired action data
        """
        if self.robot_action == "step":
            self.apply_action_step(action)
        elif self.robot_action == "joints_step":
            self.apply_action_joints_step(action)
        elif self.robot_action == "absolute":
            self.apply_action_absolute(action)
        elif self.robot_action in ["joints", "joints_gripper"]:
            self.apply_action_joints(action)
        elif self.robot_action == "velo_step":
            self.apply_action_velo_step(action)
        elif self.robot_action == "torque_step":
            self.apply_action_torque_step(action)
        elif self.robot_action == "pybulletx":
            self.apply_action_pybulletx(action)
        elif self.robot_action == "torque_control":
            self.apply_action_torque_control(action)
        if len(self.magnetized_objects):
            pos_diff = np.array(self.end_effector_pos) - np.array(self.end_effector_prev_pos)
            for key,val in self.magnetized_objects.items():
                #key.set_position(val+pos_diff)
                self.p.changeConstraint(val, key.get_position()+pos_diff)
                #self.magnetized_objects[key] = key.get_position()
            self.end_effector_prev_pos = self.end_effector_pos

    def magnetize_object(self, object, contacts):
        # Creates fixed joint between kuka gripper and object
        if any(isinstance(i, tuple) for i in contacts):
            contacts = contacts[0]
        self.p.changeVisualShape(object.uid, -1, rgbaColor=[0, 255, 0, 1])

        self.end_effector_prev_pos = self.end_effector_pos
        constraint_id = self.p.createConstraint(object.uid, -1, -1, -1, self.p.JOINT_FIXED, [0, 0, 0], [0, 0, 0],
                              object.get_position())
        self.magnetized_objects[object] = constraint_id

    def release_object(self, object):
        self.p.removeConstraint(self.magnetized_objects[object])
        self.magnetized_objects.pop(object)

    def grasp_panda(self):
        for i in range(2):
            self.p.setJointMotorControl2(bodyUniqueId=self.robot_uid,
                                jointIndex=self.motor_indices[-1-i],
                                controlMode=self.p.POSITION_CONTROL,
                                targetPosition=self.joints_limits[0][-1-i],
                                targetVelocity=0,
                                force=self.joints_max_force[-1-i],
                                maxVelocity=self.joints_max_velo[-1-i],
                                positionGain=0.7,
                                velocityGain=0.3)
        self.p.stepSimulation()

    def get_name(self):
        """
        Get name of robot

        Returns:
            :return name: (string) Name of robot
        """
        return self.name

    def get_uid(self):
        """
        Get robot's unique ID

        Returns:
            :return self.uid: Robot's unique ID
        """
        return self.robot_uid
