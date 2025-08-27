import numpy as np
from utils.grasper import Grasper
from utils.sim_height_calculation import calculate_z

print("Initializing Grasper...")
try:
    grasper = Grasper(
        urdf_path="./envs/robots/nico/nico_grasper_real.urdf",
        motor_config="./utils/nico_humanoid_upper_rh7d_ukba.json",
        connect_robot=True,     # Connect to the real robot hardware
    )
    print("Grasper initialized successfully for real robot.")
except Exception as e:
    print(f"Error initializing Grasper for real robot: {e}")

init_pos_l = [0, 0.3, 0.5]
init_pos_r = [init_pos_l[0], -init_pos_l[1], init_pos_l[2]]
init_ori = [0, -1.57, 0]
grasp_ori = [0,0,0] # Top grap [0,0,0] or side grasp [1.57,0,0]
hand = "right"

# grasper.move_to_pose("reset")
# grasper.move_both_arms(init_pos_r, init_ori)
    
print("\n--- Executing Sequence with IK Move ---")
# Initial position
grasper.init_position([0, -0.3, 0.5], [0,-1.57,0], hand)

# for i in range(5):
#     object_x, object_y, object_z = 0.369, 0.274, 0.112

#     # Calculate z for picking up the object
#     # object_z = 0.125
#     print(f"Calculated z for pick at x={object_x}, y={object_y}: {calculate_z(object_x, object_y) + 0.04}")
#     # Pick object
#     grasper.pick_object([object_x,object_y,object_z], grasp_ori, hand, autozpos=False, autoori=True)

#     # grasper.move_arm([object_x,object_y,object_z], grasp_ori, hand,autozpos = True,autoori = True)

#     # Place object
#     # grasper.place_object([object_x,object_y,object_z], grasp_ori, hand)

#     # Initial position
#     grasper.init_position([0, 0.3, 0.5], [0,-1.57,0], hand)

# right grid
# for x in np.arange(0.25, 0.45, 0.05):
#     for y in np.arange(-0.3, 0.21, 0.05):
#         grasper.pick_object([x,y,0.7], grasp_ori, hand, autozpos=True, autoori=True)
#         grasper.init_position([0, -0.3, 0.5], [0,-1.57,0], hand)

# right diagonal
for i in range(8):
    grasper.pick_object([0.25+i*0.025,-0.3+i*0.0625,0.7], grasp_ori, hand, autozpos=True, autoori=True)
    grasper.init_position([0, -0.3, 0.5], [0,-1.57,0], hand)

# left grid
# for x in np.arange(0.25, 0.45, 0.05):
#     for y in np.arange(0.3, -0.21, -0.05):
#         grasper.pick_object([x,y,0.13], grasp_ori, hand, autozpos=True, autoori=True)
#         grasper.init_position([0, 0.3, 0.5], [0,-1.57,0], hand)

# left diagonal
# for i in range(8):
#     grasper.pick_object([0.25+i*0.025,0.3+i*(-0.0625),0.7], grasp_ori, hand, autozpos=True, autoori=True)
#     grasper.init_position([0, 0.3, 0.5], [0,-1.57,0], hand)


print("--- Sequence Finished ---\n")