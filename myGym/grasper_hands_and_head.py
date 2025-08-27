import numpy as np
from utils.grasper import Grasper
from utils.sim_height_calculation import calculate_z
import random
print("Initializing Grasper...")
try:
    grasper = Grasper(
        urdf_path="./envs/robots/nico/nico_grasper.urdf",
        motor_config="./nico_humanoid_upper_rh7d_ukba.json",
        connect_robot=True,     # Connect to the real robot hardware
    )
    print("Grasper initialized successfully for real robot.")
except Exception as e:
    print(f"Error initializing Grasper for real robot: {e}")

init_pos_l = [0, 0.3, 0.5]
init_pos_r = [init_pos_l[0], -init_pos_l[1], init_pos_l[2]]
init_ori = [0, -1.57, 0]
grasp_ori = [1.57,0,0.0] # Top grap [0,0,0] or side grasp [1.57,0,0]
goal1= [0.27, 0.0]
goal2 = [0.4, 0.0]
    
print("\n--- Executing Sequence with IK Move ---")
# Initial position
grasper.move_both_arms (init_pos_r, init_ori)
grasper.point_gripper("right")
grasper.point_gripper("left")
while True:
    rand_pos = [
    random.uniform(0.15, 0.45),   # x: 0.15 to 0.35
    random.uniform(-0.3, -0.1),   # y: -0.3 to -0.1
    random.uniform(0.15, 0.35)  ]
    grasper.move_both_arms_head(rand_pos, grasp_ori)
    #object_z2 = calculate_z(goal2[0],goal2[1]) + 0.03
    #grasper.place_object([goal1[0],goal1[1],object_z1], grasp_ori, "right")
    #grasper.move_both_arms (init_pos_r, init_ori)
