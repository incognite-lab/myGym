import pybullet as p
import pybullet_data
import argparse
import time
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='URDF Visualizer with Joint Sliders')
    parser.add_argument('--urdf', type=str, default='./envs/robots/tiago/tiago_dual_mygym_rotslide2.urdf',
                       help='Path to URDF file')
    args = parser.parse_args()

    # Initialize PyBullet
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Fixed orientation for IK (quaternion)
    fixed_orientation = p.getQuaternionFromEuler([0, -np.pi/2, 0])  # Fixed downward orientation

    # Create ground plane (90x60x3 cm)
    #p.createMultiBody(baseVisualShapeIndex=p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[.30, .45, 0.025],
    #                                                           rgbaColor=[0.0, 0.6, 0.6, 1]),
    #                  baseCollisionShapeIndex=p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[.30, .45, 0.025]),
    #                  baseMass=0, basePosition=[0.26, 0, 0.029])
    # Create tablet mesh
    #p.createMultiBody(baseVisualShapeIndex=p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[.165, .267, 0.001],
    #                                                           rgbaColor=[0, 0, 0.0, .5]),
    #                  baseCollisionShapeIndex=p.createCollisionShape(shapeType=p.GEOM_BOX,
    #                                                                halfExtents=[.165, .267, 0.001]), baseMass=0,
    #                  basePosition=[0.395, 0, 0.054])
    
    # Load URDF
    try:
        robot_id = p.loadURDF(args.urdf, useFixedBase=True)
    except:
        print(f"Error: Failed to load URDF file '{args.urdf}'")
        return

    num_joints = p.getNumJoints(robot_id)


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
                
            # Convert joint limits from radians to degrees for display
            lower_deg = lower * 57.2958
            upper_deg = upper * 57.2958
            if joint_type == p.JOINT_REVOLUTE:
                slider = p.addUserDebugParameter(
                    paramName=joint_name + " (deg)",
                    rangeMin=lower_deg,
                    rangeMax=upper_deg,
                    startValue=(lower_deg + upper_deg)/2
                )
                sliders.append((joint_idx, slider))
            else:
                #Prismatic joints
                slider = p.addUserDebugParameter(
                    paramName=joint_name + " (m)",
                    rangeMin=lower,
                    rangeMax=upper,
                    startValue=(lower + upper) / 2
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
                    targetPosition=value
                )
            time.sleep(0.01)
    except KeyboardInterrupt:
        p.disconnect()

if __name__ == "__main__":
    main()