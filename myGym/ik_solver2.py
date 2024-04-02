import pybullet as p
import time
import math
from datetime import datetime
import pybullet_data
import pybullet as p
import pybullet_data

def main():
    # Initialize PyBullet
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    # Load the URDF file
      # Replace "path_to_your_urdf_file.urdf" with the path to your URDF file
    robot_id = p.loadURDF("/home/code/myGym/myGym/envs/robots/nico/nico_ik.urdf", [0, 0, 0])
    numJoints = p.getNumJoints(robot_id)
    print(numJoints)
    num_joints = 7
    p.setGravity(0, 0, -9.81)
    rp = [0, 0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
    for i in range(num_joints):
        p.resetJointState(robot_id, i, rp[i])
    # Set the target position (xyz)
    target_position = [3, 3, 3]  # Adjust the desired position as needed

    # Set the IK parameters
    num_joints = 7
    
    
    
    end_effector_index = num_joints - 1  # Assuming the end effector is the last joint, adjust if necessary
    max_iterations = 1000
    residual_threshold = 0.001

    # Perform IK
    ik_solution = p.calculateInverseKinematics(robot_id, end_effector_index, target_position,
                                               maxNumIterations=max_iterations,
                                               residualThreshold=residual_threshold)

    # Set joint angles to the IK solution
    for joint_index in range(num_joints):
        p.setJointMotorControl2(robot_id, joint_index, p.POSITION_CONTROL, ik_solution[joint_index])

    # Simulation loop
    for _ in range(10000):
        p.stepSimulation()
    input("Press Enter to continue...")
    # Disconnect from the physics server
    p.disconnect()

if __name__ == "__main__":
    main()
