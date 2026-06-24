import os
import numpy as np
import importlib.resources as pkg_resources

from myGym.train import get_parser, get_arguments, automatic_argument_assignment, configure_env
from myGym.envs import env_object
from myGym.envs.predicates import *
from myGym.utils.helpers import get_workspace_dict



def _parse():
    parser = get_parser()
    parser.add_argument("-ct", "--control",
                        help="How to control robot during testing. Valid arguments: keyboard, observation, random, oraculum, slider")
    parser.add_argument("-vs", "--vsampling", action="store_true", help="Visualize sampling area.")
    parser.add_argument("-vt", "--vtrajectory", action="store_true", help="Visualize gripper trajectory.")
    parser.add_argument("-vn", "--vinfo", action="store_true", help="Visualize info. Valid arguments: True, False")
    parser.add_argument("-ns", "--network_switcher", default="gt", help="How does a robot switch to next network (gt or keyboard)")
    parser.add_argument("-rr", "--results_report", default = False, help="Used only with oraculum - shows report of task feasibility at the end.")
    parser.add_argument("-tp", "--top_grasp",  default = False, help="Use top grasp when reaching objects with oraculum.")
    # parser.add_argument("-nl", "--natural_language", default=False, help="NL Valid arguments: True, False")
    arg_dict, commands = get_arguments(parser)
    parameters = {}
    args = parser.parse_args()
    for key, arg in arg_dict.items():
        if type(arg_dict[key]) == list:
            if len(arg_dict[key]) > 1 and key != "robot_init" and key != "end_effector_orn":
                if key != "task_objects":
                    parameters[key] = arg
                    if key in commands:
                        commands.pop(key)
    
    # Automatically adjust robot_action when oraculum control is selected
    if arg_dict.get("control") == "oraculum":
        if "gripper" in arg_dict.get("robot_action", ""):
            arg_dict["robot_action"] = "absolute_gripper"
        else:
            arg_dict["robot_action"] = "absolute"
        print(f"Oraculum control selected. Robot action automatically set to: {arg_dict['robot_action']}")
    
    if  arg_dict.get("control") == "keyboard":
        if "gripper" in arg_dict.get("robot_action", ""):
            arg_dict["robot_action"] = "step_gripper"
        else:
            arg_dict["robot_action"] = "step"
        print(f"Keyboard control selected. Robot action automatically set to: {arg_dict['robot_action']}")
    return arg_dict

def _read_urdf_scale(urdf_path: str) -> float:
    """
    Read the first mesh scale value from a URDF file,
    return 1 if not specified
    """
    with open(urdf_path) as file:
        lines = file.readlines()

    scale_lines = [line for line in lines if "scale" in line]

    if not scale_lines:
        return 1.0

    return float(scale_lines[0].split('scale="')[1].split(" ")[0])

def _voxel_demo(obj):
    import open3d as o3d
    from myGym.envs.test_volume_class import VolumeMesh

    # load tuna model and check the scale
    obj1_info = obj.p.getVisualShapeData(obj.get_uid())[0]
    obj1_scale = _read_urdf_scale(obj.urdf_path)
    objpth = obj1_info[4].decode("utf-8")

    # voxelize
    o3model = o3d.io.read_triangle_model(objpth)
    mesh = o3model.meshes[0].mesh
    mesh = mesh.scale(obj1_scale, center=mesh.get_center())
    vm = VolumeMesh(mesh)
    orig = vm.duplicate()
    orig.paint(np.array([0, 1, 0]))
    voxel_grid = vm.voxelgrid

    # visualize geometry
    o3d.visualization.draw_geometries([voxel_grid, orig.voxelgrid])

def run_simple_test(env):
    table = env.static_scene_objects[env.workspace]
    on_top = OnTop()
    touching = Touching()

    urdf_tuna = os.path.join(pkg_resources.files("myGym"), "envs/objects/household/urdf/tuna_can.urdf")
    tuna_on_table_area = on_top.compute_area(urdf_tuna, table)
    pos_tuna = env_object.EnvObject.get_random_object_position(tuna_on_table_area)

    obj_tuna = env_object.EnvObject(
        urdf_tuna,
        pos_tuna,
        [0, 0, 0, 1],
        pybullet_client=env.p,
        fixed=False
    )

    pos_tuna2 = [pos_tuna[0], pos_tuna[1], pos_tuna[2]+0.2]
    obj_tuna2 = env_object.EnvObject(
        urdf_tuna,
        pos_tuna2,
        [0, 0, 0, 1],
        pybullet_client=env.p,
        fixed=False
    )

    for _ in range(15):
        # tuna falls on the table
        env.p.stepSimulation()

    print("Tuna reachable:")
    print(IsReachable().check(env.robot, obj_tuna))
    print("Tuna touching table:")
    print(touching.check(obj_tuna, table))
    print("Tuna on top of table:")
    print(on_top.check(obj_tuna, table))
    print("Tunas touching:")
    print(touching.check(obj_tuna2, obj_tuna))

    print("-----------tuna2 falls-----------")
    for _ in range(100):
        # tuna2 falls on tuna
        env.p.stepSimulation()

    print("Tunas touching:")
    print(touching.check(obj_tuna2, obj_tuna))
    print("Tuna2 on top of tuna:")
    print(on_top.check(obj_tuna2, obj_tuna))
    print("Tuna on top of tuna2:")
    print(on_top.check(obj_tuna, obj_tuna2))

    urdf_towertarget = os.path.join(pkg_resources.files("myGym"), "envs/objects/assembly/urdf/towertarget.urdf")
    tuna_on_table_area = on_top.compute_area(urdf_towertarget, table)

def test_init(env):
    predicates = {'init': ["Reachable(apple)", "OnTop(apple,table)", "Reachable(tuna_can)"]}
    urdf_apple = os.path.join(pkg_resources.files("myGym"), "envs/objects/household/urdf/apple.urdf")
    urdf_tuna_can = os.path.join(pkg_resources.files("myGym"), "envs/objects/household/urdf/tuna_can.urdf")
    apple_info = {'urdf': urdf_apple, 'obj_name': "apple"}
    tuna_can_info = {'urdf': urdf_tuna_can, 'obj_name': "tuna_can"}

    table = env.static_scene_objects[env.workspace]
    robot = env.robot

    for i in range(1000000):
        apple_area = InitPredicateResolver().get_area(apple_info, table, robot, predicates)
        apple_pos = env_object.EnvObject.get_random_object_position(apple_area)
        apple = env_object.EnvObject(
            urdf_apple,
            apple_pos,
            [0, 0, 0, 1],
            pybullet_client=env.p,
            fixed=False
        )

        tuna_can_area = InitPredicateResolver().get_area(tuna_can_info, table, robot, predicates)
        tuna_can_pos = env_object.EnvObject.get_random_object_position(tuna_can_area)
        tuna_can = env_object.EnvObject(
            urdf_tuna_can,
            tuna_can_pos,
            [0, 0, 0, 1],
            pybullet_client=env.p,
            fixed=False
        )

        placed_objects = {'init': apple, 'goal': tuna_can}
        for _ in range(100): env.p.stepSimulation()

        check = InitPredicateResolver().check(placed_objects, env, predicates)
        if not check:
            print(apple_pos)
            print(tuna_can_pos)
            raise RuntimeError("Check not satisfied")
        env.p.removeBody(apple.uid)
    
    print("Success")


def main():
    arg_dict = _parse()

    arg_dict["gui"] = 0
    arg_dict = automatic_argument_assignment(arg_dict)
    env = configure_env(arg_dict, model_logdir=None, for_train=0)
    env = env.unwrapped

    test_init(env)
    

    



if __name__ == '__main__':
    main()
    