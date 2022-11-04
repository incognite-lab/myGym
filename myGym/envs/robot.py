import pkg_resources
from myGym.utils.vector import Vector
import numpy as np
import math
from myGym.utils.helpers import get_robot_dict

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
                 position=[-0.1, 0.0, 0.07], orientation=[0, 0, 0],
                 end_effector_index=None, gripper_index=None, 
                 init_joint_poses=None,
                 robot_action="step",
                 task_type="reach",
                 use_fixed_gripper_orn=True,
                 gripper_orn=[0, -math.pi, 0],
                 dimension_velocity = 0.5,
                 max_velocity = None, #1.,
                 max_force = None, #50.,
                 pybullet_client=None):

        self.p = pybullet_client
        self.robot_dict = get_robot_dict()
        self.robot_path = self.robot_dict[robot]['path']
        self.position = np.array(position) + self.robot_dict[robot].get('position',np.zeros(len(position)))
        self.orientation = self.p.getQuaternionFromEuler(np.array(orientation) +
                                                         self.robot_dict[robot].get('orientation',np.zeros(len(orientation))))
        self.name = robot
        self.max_velocity = max_velocity
        self.max_force = max_force
        self.end_effector_index = end_effector_index
        self.gripper_index = gripper_index
        self.use_fixed_gripper_orn = use_fixed_gripper_orn
        self.gripper_orn = self.p.getQuaternionFromEuler(gripper_orn)
        self.dimension_velocity = dimension_velocity
        self.motor_names = []
        self.motor_indices = []
        self.link_names = []
        self.link_indices = []
        self.gripper_names = []
        self.gripper_indices = []
        self.robot_action = robot_action
        self.task_type = task_type
        self.magnetized_objects = {}
        self.gripper_active = False
        self._load_robot()
        self.num_joints = self.p.getNumJoints(self.robot_uid)
        self._set_motors()
        self.joints_limits, self.joints_ranges, self.joints_rest_poses, self.joints_max_force, self.joints_max_velo = self.get_joints_limits(self.motor_indices)       
        if self.gripper_names:
            self.gjoints_limits, self.gjoints_ranges, self.gjoints_rest_poses, self.gjoints_max_force, self.gjoints_max_velo = self.get_joints_limits(self.gripper_indices)
        joint_poses = list(self._calculate_accurate_IK(init_joint_poses[:3]))
        self.init_joint_poses = joint_poses
        #self.init_joint_poses = np.zeros((len(self.motor_names)))
        #self.reset()

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
            joint_name = joint_info[1]
            q_index = joint_info[3]
            link_name = joint_info[12]
            self.link_names.append(str(joint_info[12]))
            self.link_indices.append(i)
            if link_name.decode("utf-8") == 'gripper':
                self.gripper_index = i
            if link_name.decode("utf-8") == 'endeffector':
                self.end_effector_index = i
            if q_index > -1 and "rjoint" in joint_name.decode("utf-8"): # Fixed joints have q_index -1
                self.motor_names.append(str(joint_name))
                self.motor_indices.append(i)
            if q_index > -1 and "gjoint" in joint_name.decode("utf-8"):
                self.gripper_names.append(str(joint_name))
                self.gripper_indices.append(i)

        print("Robot summary")
        print("--------------")
        print("Links:")
        print("\n".join(map(str,self.link_names)))
        print("Joints:")
        print("\n".join(map(str,self.motor_names)))
        print("Gripper joints:")
        print("\n".join(map(str,self.gripper_names)))
        print("Gripper index is: " + str(self.gripper_index))
        print("End effector index is: " + str(self.end_effector_index))

        self.joints_num = len(self.motor_names)
        self.gjoints_num = len(self.gripper_names)


        if self.end_effector_index == None:
            print("No end effector detected. Please add endeffector joint and link to the URDF file (see panda.urdf for example)")
            exit()
        if self.gripper_index == None:
            print("No gripper detected. Please add gripper joint and link to the URDF file (see panda.urdf for example)")
            exit()
        
        if 'gripper' in self.robot_action and not self.gripper_indices:
            print("Gripper control active but no gripped joints detected. Please add gjoints to the URDF file (see panda.urdf for example)")
            exit()
        
        if 'gripper' not in self.robot_action and self.gripper_indices:
            print("Gripper joints detected but not active gripper control. Setting gripper joints to fixed values")
        
        #    self.motor_indices = [x for x in self.motor_indices if x < self.gripper_index]
        #print(f"Gripper active:{self.gripper_active}")

    def touch_sensors_active(self, target_object):
        contact_points = self.p.getContactPoints(self.robot_uid, target_object.uid)
        if len(contact_points)> 0:
            return True
        return False

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
        #if self.robot_action in ["absolute","step"]:
        #    joint_poses = self._calculate_joint_poses(joint_poses)
        joint_poses = np.clip(joint_poses, self.joints_limits[0], self.joints_limits[1])
        for jid in range(len(self.motor_indices)):
            self.p.resetJointState(self.robot_uid, self.motor_indices[jid], joint_poses[jid])
        self._run_motors(joint_poses)

    def get_joints_limits(self,indices):
        """
        Identify limits, ranges and rest poses of individual robot joints. Uses data from robot model.

        Returns:
            :return [joints_limits_l, joints_limits_u]: (list) Lower and upper limits of all joints
            :return joints_ranges: (list) Ranges of movement of all joints
            :return joints_rest_poses: (list) Rest poses of all joints
        """
        joints_limits_l, joints_limits_u, joints_ranges, joints_rest_poses, joints_max_force, joints_max_velo = [], [], [], [], [], []
        for jid in indices:
            joint_info = self.p.getJointInfo(self.robot_uid, jid)
            joints_limits_l.append(joint_info[8])
            joints_limits_u.append(joint_info[9])
            joints_ranges.append(joint_info[9] - joint_info[8])
            joints_rest_poses.append((joint_info[9] + joint_info[8])/2)
            joints_max_force.append(joint_info[10] if "gjoint" in joint_info[1].decode("utf-8") else self.max_force)
            joints_max_velo.append(joint_info[11] if "gjoint" in joint_info[1].decode("utf-8") else self.max_velocity)
        return [joints_limits_l, joints_limits_u], joints_ranges, joints_rest_poses, joints_max_force, joints_max_velo

    def get_action_dimension(self):
        """
        Get dimension of action data, based on robot control mechanism

        Returns:
            :return dimension: (int) The dimension of action data
        """
        if "absolute" in self.robot_action or "step" in self.robot_action:
            dim = 3
        elif "joints" in self.robot_action:
            dim = len(self.motor_indices)
        return dim

    def observe_all_links(self):
        """
        Returns the cartesian world position of all robot's links
        """
        return [self.p.getLinkState(self.robot_uid, link)[0] for link in range(self.num_joints)]

    def get_joints_states(self):
        """
        Returns the current positions of all robot's joints
        """
        joints = []
        for link in range(self.gripper_index):
           joints.append(self.p.getJointState(self.robot_uid,link)[0])
        return joints

    def get_observation_dimension(self):
        """
        Get dimension of robot part of observation data, based on robot task and rewatd type

        Returns:
            :return dimension: (int) The dimension of observation data
        """
        return len(self.get_observation())

    def get_observation(self):
        """
        Get position and orientation of the robot end effector

        Returns: 
            :return observation: (list) Position of end-effector link (center of mass)
        """
        observation = []
        state = self.p.getLinkState(self.robot_uid, self.end_effector_index)
        pos = state[0]
        orn = self.p.getEulerFromQuaternion(state[1])

        observation.extend(list(pos))
        observation.extend(list(orn))
        return observation

    def get_links_observation(self, num):
        """
        Get robot part of observation data

        Returns: 
            :return observation: (list) Position of all links (center of mass)
        """
        observation = []
        if "kuka" in self.name:
            for link in range(self.gripper_index-num, self.gripper_index):  
            # for link in range(4, self.gripper_index):  
                state = self.p.getLinkState(self.robot_uid, link)
                pos = state[0]
                observation.extend(list(pos))
        else:
            exit("not implemented for other arms than kuka")

        self.observed_links_num = num
        return observation

    def get_position(self):
        """
        Get position of robot's end-effector link

        Returns: 
            :return position: (list) Position of end-effector link (center of mass)
        """
        #return self.get_accurate_gripper_position()
        #print(self.p.getLinkState(self.robot_uid, self.end_effector_index)[0])
        return self.p.getLinkState(self.robot_uid, self.end_effector_index)[0]
        
    def get_orientation(self):
        """
        Get orientation of robot's end-effector link

        Returns:
            :return orientation: (list) Orientation of end-effector link (center of mass)
        """
        return self.p.getLinkState(self.robot_uid, self.end_effector_index)[1]

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
                                    force=self.joints_max_force[i],
                                    maxVelocity=self.joints_max_velo[i],
                                    positionGain=0.7,
                                    velocityGain=0.3)
        
        self.end_effector_pos = self.p.getLinkState(self.robot_uid, self.end_effector_index)[0]
        self.end_effector_ori = self.p.getLinkState(self.robot_uid, self.end_effector_index)[1]
        self.gripper_pos = self.p.getLinkState(self.robot_uid, self.gripper_index)[0]  
        self.gripper_ori = self.p.getLinkState(self.robot_uid, self.gripper_index)[1]  
    
    def _move_gripper(self, action):
        """
        Move gripper motors towards desired joint poses respecting robot's dynamics

        Parameters:
            :param joint_poses: (list) Desired poses of individual joints
        """
        for i in range(len(self.gripper_indices)):
            self.p.setJointMotorControl2(bodyUniqueId=self.robot_uid,
                                    jointIndex=self.gripper_indices[i],
                                    controlMode=self.p.POSITION_CONTROL,
                                    targetPosition=action[i],
                                    force=self.gjoints_max_force[i],
                                    maxVelocity=self.gjoints_max_velo[i],
                                    positionGain=0.7,
                                    velocityGain=0.3)
        


    def _calculate_joint_poses(self, gripper_pos):
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
                                                       gripper_pos,
                                                       self.gripper_orn,
                                                       lowerLimits=self.joints_limits[0],
                                                       upperLimits=self.joints_limits[1],
                                                       jointRanges=self.joints_ranges,
                                                       restPoses=self.joints_rest_poses)
        else:
            joint_poses = self.p.calculateInverseKinematics(self.robot_uid,
                                                       self.gripper_index,
                                                       gripper_pos)
        joint_poses = np.clip(joint_poses[:len(self.motor_indices)], self.joints_limits[0], self.joints_limits[1])
        return joint_poses

    def _calculate_accurate_IK(self, gripper_pos):
        """
        Calculate joint poses corresponding to desired position of end-effector. Uses accurate inverse kinematics (iterative solution).

        Parameters:
            :param end_effector_pos: (list) Desired position of end-effector in environment [x,y,z]
        Returns:
            :return joint_poses: (list) Calculated joint poses corresponding to desired end-effector position
        """
        thresholdPos = 0.01
        thresholdOrn = 0.01
        maxIter = 1000
        closeEnough = False
        iter = 0
        while (not closeEnough and iter < maxIter):
            if (self.use_fixed_gripper_orn):
                joint_poses = self.p.calculateInverseKinematics(self.robot_uid,
                                                            self.gripper_index,
                                                            gripper_pos,
                                                            self.gripper_orn)
            else:
                joint_poses = self.p.calculateInverseKinematics(self.robot_uid,
                                                            self.gripper_index,
                                                            gripper_pos,
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
            diffPos = np.linalg.norm(np.asarray(gripper_pos)-np.asarray(newPos))
            if (self.use_fixed_gripper_orn):
                diffOrn = np.linalg.norm(np.asarray(self.gripper_orn)-np.asarray(newOrn))
            else:
                diffOrn = 0
            closeEnough = ((diffPos < thresholdPos) and (diffOrn < thresholdOrn)) 
            iter = iter + 1
        if not closeEnough:
            print(f"WARNING - Intitalization error: pos:{diffPos}, orr:{diffOrn}")
        return joint_poses


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
        Jt, Jr = self.p.calculateJacobian(self.robot_uid, self.end_effector_index, [0,0,0], joint_states, joint_velocities, [10]*(len(joint_states)))
        J = np.array([Jt,Jr]).reshape(-1, len(joint_states))
        tau = np.dot(J.T, F) # + robot.coriolis_comp().reshape(7,1)

        self._run_motors_torque(tau)

    def apply_action_step(self, action):
        """
        Apply action command to robot using step control mechanism

        Parameters:
            :param action: (list) Desired action data
        """
        action = [i * self.dimension_velocity for i in action[:3]]
        des_gripper_pos = np.add(self.gripper_pos, action)
        joint_poses = list(self._calculate_joint_poses(des_gripper_pos))
        self._run_motors(joint_poses)
    

    def apply_action_absolute(self, action):
        """
        Apply action command to robot using absolute control mechanism

        Parameters:
            :param action: (list) Desired action data
        """
        des_gripper_pos = action[:3]
        joint_poses = self._calculate_joint_poses(des_gripper_pos)
        self._run_motors(joint_poses)
        

    def apply_action_joints(self, action):
        """
        Apply action command to robot using joints control mechanism

        Parameters:
            :param action: (list) Desired action data
        """
        self._run_motors(action[:(self.joints_num)])
        
    def apply_action_joints_step(self, action):
        """
        Apply action command to robot using joint-step control mechanism
        Parameters:
            :param action: (list) Desired action data
        """
        action = [i * self.dimension_velocity for i in action]
        joint_poses = np.add(self.joints_state, action)
        self._run_motors(joint_poses)

    def apply_action(self, action, env_objects=None):
        """
        Apply action command to robot in simulated environment

        Parameters:
            :param action: (list) Desired action data
        """
        if "step" in self.robot_action:
            self.apply_action_step(action)
        elif "absolute" in self.robot_action:
            self.apply_action_absolute(action)
        elif "joints" in self.robot_action:
            self.apply_action_joints(action)
        if "gripper" in self.robot_action:
            self._move_gripper(action[-(self.gjoints_num):])
        else:
            if self.gjoints_num:
                self._move_gripper(self.gjoints_limits[1])
            if "pnp" in self.task_type: 
            #"Need to provide env_objects to use gripper"
            #When gripper is not in robot action it will magnetize objects
                self.gripper_active = True
                self.magnetize_object(env_objects["actual_state"])
            #    else:
            #        self.gripper_active = False
            #        self.release_all_objects()
            #else:
            #    self.apply_action_joints(action)
        
                if len(self.magnetized_objects):
            #pos_diff = np.array(self.end_effector_pos) - np.array(self.end_effector_prev_pos)
            #ori_diff = np.array(self.end_effector_ori) - np.array(self.end_effector_prev_ori)
                    for key,val in self.magnetized_objects.items():
                        self.p.changeConstraint(val, self.get_position(),self.get_orientation())
                #self.p.resetBasePositionAndOrientation(val,self.end_effector_pos,self.end_effector_ori)
            #self.end_effector_prev_pos = self.end_effector_pos
            #self.end_effector_prev_ori = self.end_effector_ori
        #if 'gripper' not in self.robot_action:
        #    for joint_index in range(self.gripper_index, self.end_effector_index + 1):
        #        self.p.resetJointState(self.robot_uid, joint_index, self.p.getJointInfo(self.robot_uid, joint_index)[9])

    def magnetize_object(self, object, distance_threshold=.05):
        if len(self.magnetized_objects) == 0 :
            if np.linalg.norm(np.asarray(self.get_position()) - np.asarray(object.get_position()[:3])) <= distance_threshold:
                self.p.changeVisualShape(object.uid, -1, rgbaColor=[.8, .1 , 0.1, 0.5])
                #self.end_effector_prev_pos = self.end_effector_pos
                #self.end_effector_prev_ori = self.end_effector_ori
                self.p.resetBasePositionAndOrientation(object.uid,self.get_position(),self.get_orientation())
                constraint_id = self.p.createConstraint(object.uid, -1, -1, -1, self.p.JOINT_FIXED, [0, 0, 0], [0, 0, 0],
                                      self.get_position())
                #self.p.resetBasePositionAndOrientation(object.uid,self.get_position(),self.get_orientation())
                self.magnetized_objects[object] = constraint_id
                self.gripper_active = True

    def release_object(self, object):
        if object in self.magnetized_objects.keys():
            self.p.removeConstraint(self.magnetized_objects[object])
            #self.p.resetBasePositionAndOrientation(object.uid,self.get_position(),self.get_orientation())
            self.magnetized_objects.pop(object)
        self.gripper_active = False

    def release_all_objects(self):
        for x in self.magnetized_objects:
            self.p.removeConstraint(self.magnetized_objects[x])
            #self.p.resetBasePositionAndOrientation(object.uid,self.get_position(),self.get_orientation())
            #self.p.changeVisualShape(object.uid, -1, rgbaColor=[255, 0, 0, 1])
        self.magnetized_objects = {}
        self.gripper_active = False

    def get_accurate_gripper_position(self):
        """
        Returns the position of the tip of the pointy gripper. Tested on Kuka only
        """
        gripper_position = self.p.getLinkState(self.robot_uid, self.end_effector_index)[0]
        gripper_orientation = self.p.getLinkState(self.robot_uid, self.end_effector_index)[1]
        gripper_matrix      = self.p.getMatrixFromQuaternion(gripper_orientation)
        direction_vector    = Vector([0,0,0], [0.0, 0, 0.14])
        m = np.array([[gripper_matrix[0], gripper_matrix[1], gripper_matrix[2]], [gripper_matrix[3], gripper_matrix[4], gripper_matrix[5]], [gripper_matrix[6], gripper_matrix[7], gripper_matrix[8]]])
        direction_vector.rotate_with_matrix(m)
        gripper = Vector([0,0,0], gripper_position)
        final = direction_vector.add_vector(gripper)
        return final


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

