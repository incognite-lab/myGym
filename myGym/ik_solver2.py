import pybullet as p
import time
import math
from datetime import datetime
import pybullet_data
import pybullet as p
import pybullet_data
from numpy import random

def get_joints_limits(robot_id, indices):
        """
        Identify limits, ranges and rest poses of individual robot joints. Uses data from robot model.

        Returns:
            :return [joints_limits_l, joints_limits_u]: (list) Lower and upper limits of all joints
            :return joints_ranges: (list) Ranges of movement of all joints
            :return joints_rest_poses: (list) Rest poses of all joints
        """
        joints_limits_l, joints_limits_u, joints_ranges, joints_rest_poses, joints_max_force, joints_max_velo = [], [], [], [], [], []
        for jid in indices:
            joint_info = p.getJointInfo(robot_id, jid)
            link_name = joint_info[12]
            joints_limits_l.append(joint_info[8])
            joints_limits_u.append(joint_info[9])
            joints_ranges.append(joint_info[9] - joint_info[8])
            joints_rest_poses.append((joint_info[9] + joint_info[8])/2)
        end_effector_index = jid
        return [joints_limits_l, joints_limits_u], joints_ranges, joints_rest_poses, end_effector_index

def main():
    # Initialize PyBullet
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    # Load the URDF file
      # Replace "path_to_your_urdf_file.urdf" with the path to your URDF file
    robot_id = p.loadURDF("/home/michal/mygym/myGym/envs/robots/nico/nico_ik.urdf", [0, 0, 0])
    numJoints = p.getNumJoints(robot_id)
    print(numJoints)
    motor_indices = [1,2,3,4,5,6,7]
    joints_limits, joints_ranges, joints_rest_poses,end_effector_index = get_joints_limits(robot_id, motor_indices)
    num_joints = 6
    p.setGravity(0, 0, -9.81)
    #rp = [0, 0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
    while True:
        for i in range(num_joints):
            p.resetJointState(robot_id, i, joints_rest_poses[i])
        # Set the target position (xyz)
        target_position = [3*random.rand(), 3*random.rand(), 3*random.rand()]  # Adjust the desired position as needed

        # Set the IK parameters
        
        
        
        #end_effector_index = 6  # Assuming the end effector is the last joint, adjust if necessary
        max_iterations = 10
        residual_threshold = 0.001

        # Perform IK
        #ik_solution = p.calculateInverseKinematics(robot_id, end_effector_index, target_position,
        #                                        maxNumIterations=max_iterations,
        #                                        residualThreshold=residual_threshold)
        
        ik_solution = p.calculateInverseKinematics(robot_id,
                                                       end_effector_index,
                                                       target_position,
                                                       lowerLimits=joints_limits[0],
                                                       upperLimits=joints_limits[1],
                                                       jointRanges=joints_ranges,
                                                       restPoses=joints_rest_poses)

        # Set joint angles to the IK solution
        for joint_index in range(num_joints):
            p.setJointMotorControl2(robot_id, joint_index, p.POSITION_CONTROL, ik_solution[joint_index])

        # Simulation loop
        for _ in range(10000):
            p.stepSimulation()
        # Disconnect from the physics server
    p.disconnect()

if __name__ == "__main__":
    main()
