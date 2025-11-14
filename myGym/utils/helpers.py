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
    ws_dict = {
                                'table':    {'urdf': 'table.urdf', 'texture': 'table.jpg',
                                            'transform': {'position':[0.0, 0.0, 0.0], 'orientation':[0.0, 0.0, 0.0]},
                                            'robot': {'position': [0.0, 0.0, 0.0], 'orientation': [0.0, 0.0, 0.0]},
                                            'camera': {'position': [[0.0, 2.4, 1.0], [-0.0, -1.5, 1.0], [1.8, 0.9, 1.0], [-1.8, 0.9, 1.0], [0., 0.85, 1.4],
                                                                    [0.0, 1.6, 0.8], [-0.0, -0.5, 0.8], [0.8, 0.9, 0.6], [-0.8, 0.9, 0.8], [0.0, 0.9, 1.]],
                                                        'target': [[0.0, 2.1, 0.9], [-0.0, -0.8, 0.9], [1.4, 0.9, 0.88], [-1.4, 0.9, 0.88], [0.0, 0.80, 1.],
                                                                   [0.0, 1.3, 0.5], [-0.0, -0.0, 0.6], [0.6, 0.9, 0.4], [-0.6, 0.9, 0.5], [0.0, 0.898, 0.8]]},
                                            'borders':[-0.7, 0.7, 0.5, 1.3, 0.1, 0.1]},
                                'table_nico': {'urdf': 'table_nico.urdf', 'texture': 'table.jpg',
                                            'transform': {'position':[-0.0, -0.0, 0.0], 'orientation':[0.0, 0.0, 0.0]},
                                            'robot': {'position': [0.0, 0.0, 0.0], 'orientation': [0.0, 0.0, 0.0]},
                                            'camera': {'position': [[0.0, 2.4, 1.0], [-0.0, -1.5, 1.0], [1.8, 0.9, 1.0], [-1.8, 0.9, 1.0], [0., 0.85, 1.4],
                                                                    [0.0, 1.6, 0.8], [-0.0, -0.5, 0.8], [0.8, 0.9, 0.6], [-0.8, 0.9, 0.8], [0.0, 0.9, 1.]],
                                                        'target': [[0.0, 2.1, 0.9], [-0.0, -0.8, 0.9], [1.4, 0.9, 0.88], [-1.4, 0.9, 0.88], [0.0, 0.80, 1.],
                                                                   [0.0, 1.3, 0.5], [-0.0, -0.0, 0.6], [0.6, 0.9, 0.4], [-0.6, 0.9, 0.5], [0.0, 0.898, 0.8]]},
                                            'borders':[-0.7, 0.7, 0.5, 1.3, 0.1, 0.1]},
                                'table_complex': {'urdf': 'table_universal.urdf', 'texture': 'table.jpg',
                                            'transform': {'position':[0.0, 0.0, 0.0], 'orientation':[0.0, 0.0, 0.0]},
                                            'robot': {'position': [0.0, 0.0, 0.0], 'orientation': [0.0, 0.0, 0.0]},
                                            'camera': {'position': [[0.0, 2.4, 1.0], [-0.0, -1.5, 1.0], [1.8, 0.9, 1.0], [-1.8, 0.9, 1.0], [0., 0.85, 1.4],
                                                                    [0.0, 1.6, 0.8], [-0.0, -0.5, 0.8], [0.8, 0.9, 0.6], [-0.8, 0.9, 0.8], [0.0, 0.9, 1.]],
                                                        'target': [[0.0, 2.1, 0.9], [-0.0, -0.8, 0.9], [1.4, 0.9, 0.88], [-1.4, 0.9, 0.88], [0.0, 0.80, 1.],
                                                                   [0.0, 1.3, 0.5], [-0.0, -0.0, 0.6], [0.6, 0.9, 0.4], [-0.6, 0.9, 0.5], [0.0, 0.898, 0.8]]},
                                            'borders':[-0.7, 0.7, 0.5, 1.3, 0.1, 0.1]}}
    return ws_dict


def get_robot_dict():
    r_dict =   {
                             'g1': {'path': '/envs/robots/unitree/g1_mygym.urdf', 'position': np.array([-0.3, 0.0, 0.07]), 'orientation': [0.0, 0.0, 0.0], 'default_joint_ori': [0.33, -0.11, -0.16, 0.69, -1.42, -0.89, -0.17, -0.01, 0.19, 0.39, 0.72, 0.39, 0.68], 'ee_pos': [-0.0011, -0.2737, 0.3977], 'ee_ori': [-0.0166, -0.0262, 0.0471], 'ee_quat_ori': [-0.008, -0.0133, 0.0234, 0.9996]},
                             'g1_full': {'path': '/envs/robots/unitree/g1_mygym_dual.urdf', 'position': np.array([-0.3, 0.0, 0.07]), 'orientation': [0.0, 0.0, 0.0], 'default_joint_ori': [-0.0, -0.0, -0.01, 0.01, 0.0, 0.12, -0.04, 0.01, 0.03, 0.08, -0.06, -0.01, 0.13, 0.09, -0.37, 0.27, -0.05, 0.16, -0.14, 0.01, 0.25, -0.06, 0.02, 0.03, -0.0, -0.0, -0.0, -0.0, 0.0, -0.46, -0.38, 1.04, 0.15, -0.85, 1.06, 0.17, -1.04, -0.01, -0.62, 0.03, 1.33, 0.14, 0.02], 'ee_pos': [0.0097, -0.0823, 0.1612], 'ee_ori': [-0.0011, 0.0256, -0.001], 'ee_quat_ori': [-0.0006, 0.0128, -0.0005, 0.9999]},
                             'gummi': {'path': '/envs/robots/gummi_arm/urdf/gummi.urdf', 'position': np.array([0.0, 0.0, 0.0]), 'orientation': [0.0, 0.0, 0.0], 'default_joint_ori': [-0.34, 0.6, -0.08, 0.03, -0.75, 2.89, -0.83, -0.0, -0.0, -0.0, -0.0, -0.0], 'ee_pos': [0.3773, -0.2003, 0.5334], 'ee_ori': [-0.0061, 0.0168, -0.0064], 'ee_quat_ori': [-0.003, 0.0084, -0.0032, 1.0]},
                             'hsr': {'path': '/envs/robots/hsr/hsrb4s.urdf', 'position': np.array([-0.5, 0.0, -0.72]), 'orientation': [0.0, 0.0, 0.0], 'default_joint_ori': [0.34, 0.01, 0.0, 0.69, -0.66, 0.0, -1.01, -0.01, 0.57, 0.48, 0.26, -0.24, -0.08, 0.29, 0.32, -0.35, -0.2], 'ee_pos': [0.0817, 0.0451, 0.5442], 'ee_ori': [-1.6954, -1.4684, 0.069], 'ee_quat_ori': [-0.541, -0.4623, -0.4851, 0.5083]},
                             'human': {'path': '/envs/robots/real_hands/humanoid_with_hands.urdf', 'position': np.array([1.7, 0.0, 0.77]), 'orientation': [0.0, 0.0, -1.5707963267948966]},
                             'jaco_gripper': {'path': '/envs/robots/jaco_arm/jaco/urdf/jaco_robotiq_fixed.urdf', 'position': np.array([0.0, 0.0, 0.0]), 'orientation': [0.0, 0.0, 0.0], 'default_joint_ori': [-1.5, 3.56, -0.04, 4.13, -0.03, 0.57, 0.01, 0.01, 0.01], 'ee_pos': [0.0, 0.0, 0.6211], 'ee_ori': [-0.0, 0.0001, 0.0], 'ee_quat_ori': [-0.0, 0.0, 0.0, 1.0]},
                             'kuka': {'path': '/envs/robots/kuka_magnetic_gripper_sdf/kuka_magnetic.urdf', 'position': np.array([0.0, 0.0, 0.0]), 'orientation': [0.0, 0.0, 0.0], 'default_joint_ori': [0.41, 0.17, 2.59, 0.86, 0.1, -2.09, 3.05], 'ee_pos': [0.4105, -0.0007, 0.7563], 'ee_ori': [-0.0035, -0.0396, 0.0047], 'ee_quat_ori': [-0.0017, -0.0198, 0.0023, 0.9998]},
                             'kuka_push': {'path': '/envs/robots/kuka_magnetic_gripper_sdf/kuka_push.urdf', 'position': np.array([0.0, 0.0, 0.0]), 'orientation': [0.0, 0.0, 0.0], 'default_joint_ori': [0.36, -0.13, -0.29, -1.22, -0.05, 2.05, 0.05], 'ee_pos': [0.3053, -0.0, 0.7789], 'ee_ori': [-0.01, 0.0, 0.0046], 'ee_quat_ori': [-0.005, -0.0, 0.0023, 1.0]},
                             'kuka_gripper': {'path': '/envs/robots/kuka_gripper/kuka_gripper.urdf', 'position': np.array([0.0, 0.0, 0.0]), 'orientation': [0.0, 0.0, 0.0], 'default_joint_ori': [-0.0, -0.38, 0.0, -1.46, 0.0, 2.05, -0.0, 0.79, -0.14, 0.75, 0.0], 'ee_pos': [0.2, 0.0, 0.7053], 'ee_ori': [0.0, 0.0, 0.0], 'ee_quat_ori': [0.0, 0.0, 0.0, 1.0]},
                             'leachy': {'path': '/envs/robots/pollen/reachy/urdf/leachy.urdf', 'position': np.array([0.0, 0.0, 0.32]), 'orientation': [0.0, 0.0, 0.0], 'default_joint_ori': [-0.46, 1.1, -1.57, -1.11, 1.53, 0.43, 0.0, 0.0], 'ee_pos': [0.457, 0.3629, 0.4495], 'ee_ori': [0.0433, 0.0434, 0.0087], 'ee_quat_ori': [0.0215, 0.0218, 0.0039, 0.9995]},
                             'nico_grasp': {'path': '/envs/robots/nico/nico_grasper.urdf', 'position': np.array([0.0, 0.0, 0.0]), 'orientation': [0.0, 0.0, 0.0], 'default_joint_ori': [-0.29, 0.66, 0.78, 1.32, 0.57, 0.79, -0.0, 0.0, -0.44], 'ee_pos': [0.1929, -0.2672, 0.2728], 'ee_ori': [0.03, -0.0444, 0.0099], 'ee_quat_ori': [0.0151, -0.0221, 0.0053, 0.9996]},
                             'panda_lgripper': {'path': '/envs/robots/franka_emika/panda/urdf/panda1.urdf', 'position': np.array([0.0, 0.0, 0.0]), 'orientation': [0.0, 0.0, 0.0], 'default_joint_ori': [2.97, 0.31, -2.84, -1.86, 1.66, 3.71, 1.91, 0.06, 0.06], 'ee_pos': [0.3266, -0.0004, 0.5331], 'ee_ori': [-0.0001, -0.0013, -0.0011], 'ee_quat_ori': [-0.0001, -0.0006, -0.0005, 1.0]},
                             'panda_boxgripper': {'path': '/envs/robots/franka_emika/panda/urdf/panda_cgripper.urdf', 'position': np.array([0.0, 0.0, 0.0]), 'orientation': [0.0, 0.0, 0.0], 'default_joint_ori': [-1.48, -0.36, 0.12, -1.93, 1.61, 3.48, 2.22, 0.1, 0.1], 'ee_pos': [0.3265, -0.0001, 0.4827], 'ee_ori': [0.0, -0.0012, -0.0003], 'ee_quat_ori': [0.0, -0.0006, -0.0002, 1.0]},
                             'panda_sgripper': {'path': '/envs/robots/franka_emika/panda_moveit/urdf/panda2.urdf', 'position': np.array([0.0, 0.0, 0.0]), 'orientation': [0.0, 0.0, 0.0], 'default_joint_ori': [0.0, -0.77, 0.0, -1.14, 0.0, 0.41, 0.91, 0.04, 0.04], 'ee_pos': [-0.0047, 0.0, 0.8021], 'ee_ori': [-0.005, 0.04, 3.0211], 'ee_quat_ori': [-0.0201, -0.0013, 0.998, 0.0601]},
                             'panda_gripper': {'path': '/envs/robots/franka_emika/panda_bullet/panda.urdf', 'position': np.array([0.0, 0.0, 0.0]), 'orientation': [0.0, 0.0, 0.0], 'default_joint_ori': [-0.28, -0.84, 0.0, -1.3, -0.0, 0.45, 2.86, 0.07, 0.07], 'ee_pos': [0.0, -0.0, 0.7789], 'ee_ori': [0.0, 0.0, -0.0], 'ee_quat_ori': [0.0, 0.0, -0.0, 1.0]},
                             'pepper': {'path': '/envs/robots/pepper/pepper.urdf', 'position': np.array([-0.2, 0.0, -0.72]), 'orientation': [0.0, 0.0, 0.0], 'default_joint_ori': [-0.97, -0.45, -1.31, 1.19, 1.09, 1.57, 1.57, 1.57, 1.57], 'ee_pos': [0.0604, -0.2511, 0.2883], 'ee_ori': [-0.002, 0.0052, -0.0052], 'ee_quat_ori': [-0.001, 0.0026, -0.0026, 1.0]},
                             'reachy': {'path': '/envs/robots/pollen/reachy/urdf/reachy.urdf', 'position': np.array([0.0, 0.0, 0.0]), 'orientation': [0.0, 0.0, 0.0], 'default_joint_ori': [1.51, -2.8, 1.27, -1.44, -0.94, -1.22, 1.0, 0.0, 0.0], 'ee_pos': [0.0941, -0.4512, 0.4093], 'ee_ori': [0.007, 0.0032, 0.0013], 'ee_quat_ori': [0.0035, 0.0016, 0.0006, 1.0]},
                             'tiago_single': {'path': '/envs/robots/tiago/tiago_pal_gripper.urdf', 'position': np.array([-0.3, 0.0, -0.72]), 'orientation': [0.0, 0.0, 0.0], 'default_joint_ori': [0.33, 0.0, 1.09, -0.71, 1.21, -1.86, 1.57, -0.93, 0.05, 0.05], 'ee_pos': [0.0078, -0.5147, 0.374], 'ee_ori': [0.0217, -0.0042, 0.0149], 'ee_quat_ori': [0.0109, -0.002, 0.0075, 0.9999]},
                             'tiago_dual': {'path': '/envs/robots/tiago/tiago_dual_mygym.urdf', 'position': np.array([-0.3, 0.0, -0.72]), 'orientation': [0.0, 0.0, 0.0], 'default_joint_ori': [0.35, 0.11, -1.18, 1.61, 1.72, 1.18, -1.41, 1.7, 0.05, 0.05], 'ee_pos': [0.0009, -0.3985, 0.3127], 'ee_ori': [0.0074, -0.0071, 0.001], 'ee_quat_ori': [0.0037, -0.0036, 0.0005, 1.0]},
                             'tiago_dual_fix': {'path': '/envs/robots/tiago/tiago_dual_mygym_fix.urdf', 'position': np.array([-0.3, 0.0, -0.72]), 'orientation': [0.0, 0.0, 0.0], 'default_joint_ori': [0.18, 0.12, -1.18, 1.62, 1.72, 1.18, -1.41, 1.7], 'ee_pos': [0.0023, -0.3984, 0.1364], 'ee_ori': [0.007, -0.0033, 0.0017], 'ee_quat_ori': [0.0035, -0.0016, 0.0009, 1.0]},
                             'tiago_dual_rot': {'path': '/envs/robots/tiago/tiago_dual_mygym_rot.urdf', 'position': np.array([-0.3, 0.0, -0.72]), 'orientation': [0.0, 0.0, 0.0], 'default_joint_ori': [0.0, 0.35, 0.11, -1.18, 1.61, 1.72, 1.18, -1.41, 1.7, 0.05, 0.05], 'ee_pos': [0.0009, -0.3985, 0.3127], 'ee_ori': [0.0074, -0.0071, 0.001], 'ee_quat_ori': [0.0037, -0.0036, 0.0005, 1.0]},
                             'tiago_dual_rotslide': {'path': '/envs/robots/tiago/tiago_dual_mygym_rotslide.urdf', 'position': np.array([-0.3, 0.0, -0.72]), 'orientation': [0.0, 0.0, 0.0], 'default_joint_ori': [0.0, 0.0, 0.35, 0.11, -1.18, 1.61, 1.72, 1.18, -1.41, 1.7, 0.05, 0.05], 'ee_pos': [0.0009, -0.3985, 0.3127], 'ee_ori': [0.0074, -0.0071, 0.001], 'ee_quat_ori': [0.0037, -0.0036, 0.0005, 1.0]},
                             'tiago_dual_rotslide2': {'path': '/envs/robots/tiago/tiago_dual_mygym_rotslide2.urdf', 'position': np.array([-0.3, 0.0, -0.72]), 'orientation': [0.0, 0.0, 0.0], 'default_joint_ori': [0.0, 0.0, 0.0, 0.35, 0.11, -1.18, 1.61, 1.72, 1.18, -1.41, 1.7, 0.05, 0.05], 'ee_pos': [0.0009, -0.3985, 0.3127], 'ee_ori': [0.0074, -0.0071, 0.001], 'ee_quat_ori': [0.0037, -0.0036, 0.0005, 1.0]},
                             'ur3': {'path': '/envs/robots/universal_robots/urdf/ur3.urdf', 'position': np.array([0.0, -0.02, -0.041]), 'orientation': [0.0, 0.0, 0.0], 'default_joint_ori': [0.6, -1.9, 0.67, -0.34, -1.57, 0.6], 'ee_pos': [-0.0, 0.1158, 0.4104], 'ee_ori': [-0.0, 0.0001, -0.0], 'ee_quat_ori': [-0.0, 0.0, -0.0, 1.0]},
                             'ur10': {'path': '/envs/robots/universal_robots/urdf/ur10.urdf', 'position': np.array([0.0, 0.0, 0.0]), 'orientation': [0.0, 0.0, 0.0], 'default_joint_ori': [-1.34, -2.16, 1.07, -0.48, -1.57, -1.34], 'ee_pos': [0.1686, 0.0, 1.0], 'ee_ori': [-0.0003, 0.0, -0.0], 'ee_quat_ori': [-0.0001, 0.0, -0.0, 1.0]},
                             'yumi': {'path': '/envs/robots/abb/yumi/urdf/yumi.urdf', 'position': np.array([0.0, 0.0, 0.0]), 'orientation': [0.0, 0.0, 0.0], 'default_joint_ori': [-0.41, -2.29, -0.12, 0.19, -0.93, 2.41, -3.99, 0.02, 0.02, 1.09, -1.66, 1.08, -0.1, 0.47, 2.06, -1.5, 0.03, 0.02], 'ee_pos': [-0.0005, 0.3054, 0.7052], 'ee_ori': [-0.001, -0.0033, 0.0047], 'ee_quat_ori': [-0.0005, -0.0017, 0.0024, 1.0]},
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