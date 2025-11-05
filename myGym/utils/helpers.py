from matplotlib.pyplot import table
import numpy as np

class PrintEveryNCalls:
    def __init__(self, msg, n):
        self.default_msg = msg
        self.n = n
        self.call_count = 0

    def __call__(self, additional_msg = ""):
        self.call_count += 1
        if self.call_count % self.n == 0:
            print(self.default_msg + str(additional_msg))



def get_workspace_dict():
    ws_dict = {'baskets':  {'urdf': 'baskets.urdf', 'texture': 'baskets.jpg',
                                            'transform': {'position':[3.18, -3.49, -1.05], 'orientation':[0.0, 0.0, -0.4*np.pi]},
                                            'robot': {'position': [0.0, 0.0, 0.0], 'orientation': [0.0, 0.0, 0.5*np.pi]},
                                            'camera': {'position': [[0.56, -1.71, 0.6], [-1.3, 3.99, 0.6], [-3.43, 0.67, 1.0], [2.76, 2.68, 1.0], [-0.54, 1.19, 3.4]],
                                                        'target': [[0.53, -1.62, 0.59], [-1.24, 3.8, 0.55], [-2.95, 0.83, 0.8], [2.28, 2.53, 0.8], [-0.53, 1.2, 3.2]]},
                                            'borders':[-0.7, 0.7, 0.3, 1.3, -0.9, -0.9]},
                                'collabtable': {'urdf': 'collabtable.urdf', 'texture': 'collabtable.jpg',
                                            'transform': {'position':[0.45, -5.1, -1.05], 'orientation':[0.0, 0.0, -0.35*np.pi]},
                                            'robot': {'position': [0.0, 0.0, 0.0], 'orientation': [0.0, 0.0, 0.5*np.pi]},
                                            'camera': {'position': [[-0.25, 3.24, 1.2], [-0.44, -1.34, 1.0], [-1.5, 2.6, 1.0], [1.35, -1.0, 1.0], [-0.1, 1.32, 1.4]],
                                                        'target': [[-0.0, 0.56, 0.6], [-0.27, 0.42, 0.7], [-1, 2.21, 0.8], [-0.42, 2.03, 0.2], [-0.1, 1.2, 0.7]]},
                                            'borders':[-0.7, 0.7, 0.5, 1.2, 0.2, 0.2]},
                                'darts':    {'urdf': 'darts.urdf', 'texture': 'darts.jpg',
                                            'transform': {'position':[-1.4, -6.7, -1.05], 'orientation':[0.0, 0.0, -1.0*np.pi]},
                                            'robot': {'position': [0.0, 0.0, 0.0], 'orientation': [0.0, 0.0, 0.5*np.pi]},
                                            'camera': {'position': [[-0.0, 2.1, 1.0], [0.0, -1.5, 1.2], [2.3, 0.5, 1.0], [-2.6, 0.5, 1.0], [-0.0, 1.1, 4.9]],
                                                        'target': [[0.0, 0.0, 0.7], [-0.0, 1.3, 0.6], [1.0, 0.9, 0.9], [-1.6, 0.9, 0.9], [-0.0, 1.2, 3.1]]},
                                            'borders':[-0.7, 0.7, 0.3, 1.3, -0.9, -0.9]},
                                'drawer':   {'urdf': 'drawer.urdf', 'texture': 'drawer.jpg',
                                            'transform': {'position':[-4.81, 1.75, -1.05], 'orientation':[0.0, 0.0, 0.0*np.pi]},
                                            'robot': {'position': [0.0, 0.2, 0.0], 'orientation': [0, 0, 0.5*np.pi]},
                                            'camera': {'position': [[-0.14, -1.63, 1.0], [-0.14, 3.04, 1.0], [-1.56, -0.92, 1.0], [1.2, -1.41, 1.0], [-0.18, 0.88, 2.5]],
                                                        'target': [[-0.14, -0.92, 0.8], [-0.14, 2.33, 0.8], [-0.71, -0.35, 0.7], [0.28, -0.07, 0.6], [-0.18, 0.84, 2.1]]},
                                            'borders':[-0.7, 0.7, 0.4, 1.3, 0.8, 0.1]},
                                'football': {'urdf': 'football.urdf', 'texture': 'football.jpg',
                                            'transform': {'position':[4.2, -5.4, -1.05], 'orientation':[0.0, 0.0, -1.0*np.pi]},
                                            'robot': {'position': [0.0, 0.0, 0.0], 'orientation': [0.0, 0.0, 0.5*np.pi]},
                                            'camera': {'position': [[-0.0, 2.1, 1.0], [0.0, -1.7, 1.2], [3.5, -0.6, 1.0], [-3.5, -0.7, 1.0], [-0.0, 2.0, 4.9]],
                                                        'target': [[0.0, 0.0, 0.7], [-0.0, 1.3, 0.2], [3.05, -0.2, 0.9], [-2.9, -0.2, 0.9], [-0.0, 2.1, 3.6]]},
                                            'borders':[-0.7, 0.7, 0.3, 1.3, -0.9, -0.9]},
                                'fridge':   {'urdf': 'fridge.urdf', 'texture': 'fridge.jpg',
                                            'transform': {'position':[1.6, -5.95, -1.05], 'orientation':[0.0, 0.0, 0*np.pi]},
                                            'robot': {'position': [0.0, 0.0, 0.0], 'orientation': [0.0, 0.0, 0.5*np.pi]},
                                            'camera': {'position': [[0.0, -1.3, 1.0], [0.0, 2.35, 1.2], [-1.5, 0.85, 1.0], [1.4, 0.85, 1.0], [0.0, 0.55, 2.5]],
                                                        'target': [[0.0, 0.9, 0.7], [0.0, 0.9, 0.6], [0.0, 0.55, 0.5], [0.4, 0.55, 0.7], [0.0, 0.45, 1.8]]},
                                            'borders':[-0.7, 0.7, 0.3, 0.5, -0.9, -0.9]},
                                'maze':     {'urdf': 'maze.urdf', 'texture': 'maze.jpg',
                                            'transform': {'position':[6.7, -3.1, 0.0], 'orientation':[0.0, 0.0, -0.5*np.pi]},
                                            'robot': {'position': [0.0, 0.0, 0.1], 'orientation': [0.0, 0.0, 0.5*np.pi]},
                                            'camera': {'position': [[0.0, -1.4, 2.3], [-0.0, 5.9, 1.9], [4.7, 2.7, 2.0], [-3.2, 2.7, 2.0], [-0.0, 3.7, 5.0]],
                                                        'target': [[0.0, -1.0, 1.9], [-0.0, 5.6, 1.7], [3.0, 2.7, 1.5], [-2.9, 2.7, 1.7], [-0.0, 3.65, 4.8]]},
                                            'borders':[-2.5, 2.2, 0.7, 4.7, 0.05, 0.05]},
                                'stairs':   {'urdf': 'stairs.urdf', 'texture': 'stairs.jpg',
                                            'transform': {'position':[-5.5, -0.08, -1.05], 'orientation':[0.0, 0.0, -0.20*np.pi]},
                                            'robot': {'position': [0.0, 0.0, 0.0], 'orientation': [0.0, 0.0, 0.5*np.pi]},
                                            'camera': {'position': [[0.04, -1.64, 1.0], [0.81, 3.49, 1.0], [-2.93, 1.76, 1.0], [4.14, 0.33, 1.0], [2.2, 1.24, 3.2]],
                                                        'target': [[0.18, -1.12, 0.85], [0.81, 2.99, 0.8], [-1.82, 1.57, 0.7], [3.15, 0.43, 0.55], [2.17, 1.25, 3.1]]},
                                            'borders':[-0.5, 2.5, 0.8, 1.6, 0.1, 0.1]},
                                'table':    {'urdf': 'table.urdf', 'texture': 'table.jpg',
                                            'transform': {'position':[-0.0, -0.0, -1.05], 'orientation':[0.0, 0.0, 0*np.pi]},
                                            'robot': {'position': [0.0, 0.0, 0.0], 'orientation': [0.0, 0.0, 0.5*np.pi]},
                                            'camera': {'position': [[0.0, 2.4, 1.0], [-0.0, -1.5, 1.0], [1.8, 0.9, 1.0], [-1.8, 0.9, 1.0], [0., 0.85, 1.4],
                                                                    [0.0, 1.6, 0.8], [-0.0, -0.5, 0.8], [0.8, 0.9, 0.6], [-0.8, 0.9, 0.8], [0.0, 0.9, 1.]],
                                                        'target': [[0.0, 2.1, 0.9], [-0.0, -0.8, 0.9], [1.4, 0.9, 0.88], [-1.4, 0.9, 0.88], [0.0, 0.80, 1.],
                                                                   [0.0, 1.3, 0.5], [-0.0, -0.0, 0.6], [0.6, 0.9, 0.4], [-0.6, 0.9, 0.5], [0.0, 0.898, 0.8]]},
                                            'borders':[-0.7, 0.7, 0.5, 1.3, 0.1, 0.1]},
                                'table_tiago': {'urdf': 'table_tiago.urdf', 'texture': 'table.jpg',
                                            'transform': {'position':[-0.0, -0.0, -1.05], 'orientation':[0.0, 0.0, 0*np.pi]},
                                            'robot': {'position': [0.0, 0.3, -0.85], 'orientation': [0.0, 0.0, 0.5*np.pi]},
                                            'camera': {'position': [[0.0, 2.4, 1.0], [-0.0, -1.5, 1.0], [1.8, 0.9, 1.0], [-1.8, 0.9, 1.0], [0., 0.85, 1.4],
                                                                    [0.0, 1.6, 0.8], [-0.0, -0.5, 0.8], [0.8, 0.9, 0.6], [-0.8, 0.9, 0.8], [0.0, 0.9, 1.]],
                                                        'target': [[0.0, 2.1, 0.9], [-0.0, -0.8, 0.9], [1.4, 0.9, 0.88], [-1.4, 0.9, 0.88], [0.0, 0.80, 1.],
                                                                   [0.0, 1.3, 0.5], [-0.0, -0.0, 0.6], [0.6, 0.9, 0.4], [-0.6, 0.9, 0.5], [0.0, 0.898, 0.8]]},
                                            'borders':[-0.7, 0.7, 0.5, 1.3, 0.1, 0.1]},
                                'table_nico': {'urdf': 'table_nico.urdf', 'texture': 'table.jpg',
                                            'transform': {'position':[-0.0, -0.0, -1.05], 'orientation':[0.0, 0.0, 0*np.pi]},
                                            'robot': {'position': [0.0, 0.0, 0.0], 'orientation': [0.0, 0.0, 0.5*np.pi]},
                                            'camera': {'position': [[0.0, 2.4, 1.0], [-0.0, -1.5, 1.0], [1.8, 0.9, 1.0], [-1.8, 0.9, 1.0], [0., 0.85, 1.4],
                                                                    [0.0, 1.6, 0.8], [-0.0, -0.5, 0.8], [0.8, 0.9, 0.6], [-0.8, 0.9, 0.8], [0.0, 0.9, 1.]],
                                                        'target': [[0.0, 2.1, 0.9], [-0.0, -0.8, 0.9], [1.4, 0.9, 0.88], [-1.4, 0.9, 0.88], [0.0, 0.80, 1.],
                                                                   [0.0, 1.3, 0.5], [-0.0, -0.0, 0.6], [0.6, 0.9, 0.4], [-0.6, 0.9, 0.5], [0.0, 0.898, 0.8]]},
                                            'borders':[-0.7, 0.7, 0.5, 1.3, 0.1, 0.1]},
                                'verticalmaze': {'urdf': 'verticalmaze.urdf', 'texture': 'verticalmaze.jpg',
                                            'transform': {'position':[-5.7, -7.55, -1.05], 'orientation':[0.0, 0.0, 0.5*np.pi]},
                                            'robot': {'position': [0.0, 0.0, 0.0], 'orientation': [0.0, 0.0, 0.]},
                                            'camera': {'position': [[-0.0, -1.25, 1.0], [0.0, 1.35, 1.3], [1.7, -1.25, 1.0], [-1.6, -1.25, 1.0], [0.0, 0.05, 2.5]],
                                                        'target': [[-0.0, -1.05, 1.0], [0.0, 0.55, 1.3], [1.4, -0.75, 0.9], [-1.3, -0.75, 0.9], [0.0, 0.15, 2.1]]},
                                            'borders':[-0.7, 0.8, 0.65, 0.65, 0.7, 1.4]},
                                'modularmaze': {'urdf': 'modularmaze.urdf', 'texture': 'verticalmaze.jpg',
                                            'transform': {'position':[-7, 8, 0.0], 'orientation':[0.0, 0.0, 0.0]},
                                            'robot': {'position': [0.0, -0.5, 0.05], 'orientation': [0.0, 0.0, 0.5*np.pi]},
                                            'camera': {'position': [[-0.0, -1.25, 1.0], [0.0, 1.35, 1.3], [1.7, -1.25, 1.0], [-1.6, -1.25, 1.0], [0.0, 0.7, 2.1], [-0.0, -0.3, 0.2]],
                                                        'target': [[-0.0, -1.05, 0.9], [0.0, 0.55, 1.3], [1.4, -0.75, 0.9], [-1.3, -0.75, 0.9], [0.0, 0.71, 1.8], [-0.0, -0.25, 0.199]]},
                                            'borders':[-0.7, 0.8, 0.65, 0.65, 0.7, 1.4]},
                                'table_uni': {'urdf': 'table_uni.urdf', 'texture': 'table.jpg',
                                            'transform': {'position':[0.0, 0.0, -0.72], 'orientation':[0.0, 0.0, 0.0]},
                                            'robot': {'position': [0.0, 0.0, 0.0], 'orientation': [0.0, 0.0, 0.0]},
                                            'camera': {'position': [[0.0, 2.4, 1.0], [-0.0, -1.5, 1.0], [1.8, 0.9, 1.0], [-1.8, 0.9, 1.0], [0., 0.85, 1.4],
                                                                    [0.0, 1.6, 0.8], [-0.0, -0.5, 0.8], [0.8, 0.9, 0.6], [-0.8, 0.9, 0.8], [0.0, 0.9, 1.]],
                                                        'target': [[0.0, 2.1, 0.9], [-0.0, -0.8, 0.9], [1.4, 0.9, 0.88], [-1.4, 0.9, 0.88], [0.0, 0.80, 1.],
                                                                   [0.0, 1.3, 0.5], [-0.0, -0.0, 0.6], [0.6, 0.9, 0.4], [-0.6, 0.9, 0.5], [0.0, 0.898, 0.8]]},
                                            'borders':[-0.7, 0.7, 0.5, 1.3, 0.1, 0.1]}}
    return ws_dict


def get_robot_dict():
    r_dict =   {
                             'kuka': {'path': '/envs/robots/kuka_magnetic_gripper_sdf/kuka_magnetic.urdf', 'position': np.array([0.0, 0.0, 0.0]), 'orientation': [0.0, 0.0, np.pi/2], 'default_joint_ori': [0.0, 0.0, 0.0, -1.06, 0.0, 2.09, -1.58], 'ee_pos': [0.0, 0.3473, 0.7746], 'ee_ori': [-0.01, 0.0, 0.0046]},
                             'kuka_push': {'path': '/envs/robots/kuka_magnetic_gripper_sdf/kuka_push.urdf', 'position': np.array([0.0, 0.0, 0.0]), 'orientation': [0.0, 0.0, np.pi/2], 'default_joint_ori': [0.0, 0.0, 0.0, -1.06, 0.0, 2.09, -1.58], 'ee_pos': [0.0, 0.3473, 0.7746], 'ee_ori': [-0.01, 0.0, 0.0046]},
                             'kuka_gripper': {'path': '/envs/robots/kuka_gripper/kuka_gripper.urdf', 'position': np.array([0.0, 0.0, 0.0]), 'orientation': [0.0, 0.0, 0], 'default_joint_ori': [0.0, 0.0, 0.0, -1.06, 0.0, 2.09, 0.0, 0.79, 0.0, 0.79, 0.0], 'ee_pos': [-0.0, 0.3468, 0.7246], 'ee_ori': [-0.01, -0.0, -0.0008]},
                             'panda_sgripper': {'path': '/envs/robots/franka_emika/panda/urdf/panda1.urdf', 'position': np.array([0.0, 0.0, 0.0]), 'orientation': [0.0, 0.0, np.pi/2], 'default_joint_ori': [0.0, 0.0, 0.0, -1.57, 1.56, 1.58, 0.69, 0.06, 0.06], 'ee_pos': [0.0033, 0.5546, 0.5049], 'ee_ori': [-0.0018, -0.0108, 0.087]},
                             'panda_boxgripper': {'path': '/envs/robots/franka_emika/panda/urdf/panda_cgripper.urdf', 'position': np.array([0.0, 0.0, 0.0]), 'orientation': [0.0, 0.0, np.pi/2], 'default_joint_ori': [0.0, 0.0, 0.0, -1.57, 1.56, 1.6, 2.28, 0.1, 0.1], 'ee_pos': [0.005, 0.5545, 0.5048], 'ee_ori': [-0.0015, -0.0108, 0.047]},
                             'panda_lgripper': {'path': '/envs/robots/franka_emika/panda_moveit/urdf/panda2.urdf', 'position': np.array([0.0, 0.0, 0.0]), 'orientation': [0.0, 0.0, np.pi/2], 'default_joint_ori': [0.0, 0.0, 0.0, -0.8, 0.0, 0.8, 0.78, 0.04, 0.04], 'ee_pos': [0.0, 0.3885, 0.7486], 'ee_ori': [-0.0021, 0.0, 0.0054]},
                             'panda_gripper': {'path': '/envs/robots/franka_emika/panda_bullet/panda.urdf', 'position': np.array([0.0, 0.0, 0.0]), 'orientation': [0.0, 0.0, np.pi/2], 'default_joint_ori': [0.0, 0.0, 0.0, -0.8, 0.0, 0.8, 0.0, 0.07, 0.07], 'ee_pos': [-0.0, 0.3885, 0.7487], 'ee_ori': [-0.0016, 0.0, 0.0008]},
                             'jaco_gripper': {'path': '/envs/robots/jaco_arm/jaco/urdf/jaco_robotiq_fixed.urdf', 'position': np.array([0.0, 0.0, -0.0]), 'orientation': [0.0, 0.0, 0.0], 'default_joint_ori': [-1.59, 3.14, 0.0, 3.91, 0.0, 0.8, 0.0, 0.0, 0.0], 'ee_pos': [0.0054, 0.2261, 0.6182], 'ee_ori': [0.03, 0.0001, 0.0186]},
                             'nico_grasp': {'path': '/envs/robots/nico/nico_grasper.urdf', 'position': np.array([0.0, 0.0, 0.0]), 'orientation': [0.0, 0.0, 0], 'default_joint_ori': [-0.44, 0.69, 1.22, 1.12, 0.03, 0.79, 0.0, 0.0, 0.0], 'ee_pos': [0.258, 0.1159, 0.3095], 'ee_ori': [-0.0622, 0.0289, 1.6704]},
                             'reachy': {'path': '/envs/robots/pollen/reachy/urdf/reachy.urdf', 'position': np.array([0.0, 0.0, 0.0]), 'orientation': [0.0, 0.0, np.pi/2], 'default_joint_ori': [1.37, -2.65, 1.02, -1.65, -1.48, 0.62, 1.05, 0.0, 0.0], 'ee_pos': [0.4753, 0.0949, 0.3462], 'ee_ori': [0.0001, -0.0229, -0.0002]},
                             'leachy': {'path': '/envs/robots/pollen/reachy/urdf/leachy.urdf', 'position': np.array([0.0, 0.0, 0.32]), 'orientation': [0.0, 0.0, np.pi/2], 'default_joint_ori': [-0.38, 1.08, -1.57, -2.18, 0.19, 1.16, 0.0, 0.0], 'ee_pos': [-0.211, 0.3608, 0.3614], 'ee_ori': [0.1379, -0.0877, 0.0307]},
                             'reachy_and_leachy': {'path': '/envs/robots/pollen/reachy/urdf/reachy_and_leachy.urdf', 'position': np.array([0.0, 0.0, 0.32]), 'orientation': [0.0, 0.0, 0.0]},
                             'gummi': {'path': '/envs/robots/gummi_arm/urdf/gummi.urdf', 'position': np.array([0.0, 0.0, 0.021]), 'orientation': [0.0, 0.0, np.pi/2]},
                             'gummi_fixed': {'path': '/envs/robots/gummi_arm/urdf/gummi_fixed.urdf', 'position': np.array([-0.1, 0.0, 0.021]), 'orientation': [0.0, 0.0, np.pi/2]},
                             'ur3': {'path': '/envs/robots/universal_robots/urdf/ur3.urdf', 'position': np.array([0.0, -0.02, -0.041]), 'orientation': [0.0, 0.0, 0.0]},
                             'ur5': {'path': '/envs/robots/universal_robots/urdf/ur5.urdf', 'position': np.array([0.0, -0.03, -0.041]), 'orientation': [0.0, 0.0, 0.0]},
                             'ur10': {'path': '/envs/robots/universal_robots/urdf/ur10.urdf', 'position': np.array([0.0, -0.04, -0.041]), 'orientation': [0.0, 0.0, 0.0]},
                             'yumi': {'path': '/envs/robots/abb/yumi/urdf/yumi.urdf', 'position': np.array([0.0, 0.15, -0.042]), 'orientation': [0.0, 0.0, 0.0]},
                             'icub': {'path': '/envs/robots/iCub/robots/iCubGenova04_plus/model.urdf', 'position': np.array([0.0, 0.15, -0.042]), 'orientation': [0.0, 0.0, 0.0]},
                             'human': {'path': '/envs/robots/real_hands/humanoid_with_hands.urdf', 'position': np.array([0.0, 2.0, 0.45]), 'orientation': [0.0, 0.0, 0.0]},
                             'tiago': {'path': '/envs/robots/tiago/tiago_pal_gripper.urdf', 'position': np.array([0.0, -0.4, -0.2]), 'orientation': [0.0, 0.0, 0.0]},
                             'tiago_dual': {'path': '/envs/robots/tiago/tiago_dual_mygym.urdf', 'position': np.array([0.0, -0.5, -0.2]), 'orientation': [0.0, 0.0, 0.0]},
                             'tiago_dual_fix': {'path': '/envs/robots/tiago/tiago_dual_mygym_fix.urdf', 'position': np.array([0.0, -0.5, -0.2]), 'orientation': [0.0, 0.0, 0.0]},
                             'tiago_dual_rot': {'path': '/envs/robots/tiago/tiago_dual_mygym_rot.urdf', 'position': np.array([0.0, -0.5, -0.72]), 'orientation': [0.0, 0.0, np.pi/2]},
                             'tiago_dual_rotslide': {'path': '/envs/robots/tiago/tiago_dual_mygym_rotslide.urdf', 'position': np.array([0.0, -0.6, -0.1]), 'orientation': [0.0, 0.0, 0.0]},
                             'tiago_dual_rotslide2': {'path': '/envs/robots/tiago/tiago_dual_mygym_rotslide2.urdf', 'position': np.array([0.2, -0.6, -0.1]), 'orientation': [0.0, 0.0, 0.0]},
                             'tiago_dual_hand': {'path': '/envs/robots/tiago/tiago_dual_hand.urdf', 'position': np.array([0.0, -0.4, -0.2]), 'orientation': [0.0, 0.0, 0.0]},
                             'tiago_omni_single': {'path': '/envs/robots/tiago/tiago_stanford_right_arm.urdf', 'position': np.array([0.0, -0.4, -0.2]), 'orientation': [0.0, 0.0, 0.0]},
                             'tiago_simple': {'path': '/envs/robots/tiago/tiago_simple.urdf', 'position': np.array([0.0, -0.4, -0.2]), 'orientation': [0.0, 0.0, 0.0]},
                             'tiago_dual_fingers': {'path': '/envs/robots/tiago_dualhand/tiago_dual_hey5.urdf', 'position': np.array([0.0, -0.4, -0.2]), 'orientation': [0.0, 0.0, 0.0]},
                             'hsr': {'path': '/envs/robots/hsr/hsrb4s.urdf', 'position': np.array([0.0, -0.15, -0.4]), 'orientation': [0.0, 0.0, 0.0]},
                             'g1': {'path': '/envs/robots/unitree/g1_mygym.urdf', 'position': np.array([0.0, -0.2, 0.6]), 'orientation': [0.0, 0.0, 0.0]},
                             'g1_dual': {'path': '/envs/robots/unitree/g1_mygym_dual.urdf', 'position': np.array([0.0, -0.2, 0.6]), 'orientation': [0.0, 0.0, 0.0]},
                             'g1_whole': {'path': '/envs/robots/unitree/g1_mygym_whole.urdf', 'position': np.array([0.0, -0.2, 0.6]), 'orientation': [0.0, 0.0, 0.0]},
                             'pepper': {'path': '/envs/robots/pepper/pepper.urdf', 'position': np.array([-0.0, -0.18, -0.721]), 'orientation': [0.0, 0.0, 0.0]},
                             }
    return r_dict

def get_gripper_dict():
    """
    Dict with important values for the grippers of each robot - closed/open values and thresholds
    cpen: value for each gripper joint which opens the gripper
    close: value for each gripper joint which closes the gripper
    th_open: value
    """
    g_dict ={"tiago": {"open": [1, 1], "close": [0,0], "th_open": [(0.7, 'g'), (0.7, 'g')], "th_closed": [(0.001, 'l'), (0.001, 'l')]},
             "g1": {"open": [0 ,0, 0], "close": [-0.4, 1.57, 1.57], "th_open": [(0.02, 'g'), (0.5, 'l'), (0.5, 'l')],
                    "th_closed": [(-0.375, 'l'), (1.55, 'g'), (1.55, 'g')]},
             "kuka_gripper": {"open": 0, "close": 0, "th_open": 0, "th_closed": 0},
             "nico_grasp": {"open": 0, "close": 0, "th_open": 0, "th_closed": 0}}
    return g_dict