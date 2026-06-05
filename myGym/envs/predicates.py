#from myGym.envs.igibson_predicates import *
from myGym.envs import env_object
from myGym.envs.test_volume_class import VolumeMesh
import pybullet as p
import open3d as o3d
import numpy as np
        

def get_reachable_range(robot):
        # old helper from PRAG
        # TODO update: get the area from test_robot_reachability.py

        is_humanoid = False
        for humanoid in ["tiago", "nico"]:
            if humanoid in robot.name:
                is_humanoid = True

        if is_humanoid:
            return [[-0.1, 0.5], [0.15, 0.8], [0.6, 1.5]]
        else:
            return [[-0.7, 0.7], [0.1, 0.8], [-0.1, 1.2]]


class IsReachable:
    """
    checking if object lies inside the precomputed gripper's reachable envelope

    ! does not solve IK and does not check collisions
    """

    def compute_area(self, robot, obj=None) -> list[list[float]]:
        """
        Return reachable 3D area in world coordinates:
            [
                [x_min, x_max],
                [y_min, y_max],
                [z_min, z_max],
            ]
        """
        robot_base = list(robot.position) # gripper.get_position()?
        reachable_range = get_reachable_range(robot)

        reachable_area = []
        for axis in range(3):
            axis_min = robot_base[axis] + reachable_range[axis][0]
            axis_max = robot_base[axis] + reachable_range[axis][1]
            reachable_area.append([axis_min, axis_max])

        return reachable_area
    
    def check(self, robot, obj) -> bool:
        """
        Return True if object is inside reachable area
        """
        reachable_area = self.compute_area(robot)
        obj_position, orn = obj.get_position_and_orientation() # obj_position = obj.get_position()

        for axis, coordinate in enumerate(obj_position):
            axis_min, axis_max = reachable_area[axis]

            if coordinate < axis_min or coordinate > axis_max:
                return False

        return True


class Touching():
    def set_value(self, obj1, obj2):
        raise NotImplementedError()

    def get_value(self, obj1, obj2):
        overlap_objs = p.getOverlappingObjects(obj1.get_bounding_box()[0], obj1.get_bounding_box()[4])
        overlapping = list(o[0] for o in overlap_objs)
        return obj2.uid in overlapping


class OnTop():
    def set_value(self, obj1, obj2):
        raise NotImplementedError()

    def get_value(self, obj1, obj2):
        overlap_objs = p.getOverlappingObjects(obj1.get_bounding_box()[0], obj1.get_bounding_box()[4])
        overlapping = list(o[0] for o in overlap_objs)
        base1 = obj1.get_bounding_box()[-1][-1]
        base2 = obj2.get_bounding_box()[-1][-1]
        return obj2.uid in overlapping and base1 > base2
    
    
def get_scale_from_urdf(pth):
    with open(pth) as f:
        lines = f.readlines()
    scale = float([x for x in lines if "scale" in x][0].split("scale=\"")[1].split(" ")[0])
    return scale




if __name__ == '__main__':
    from myGym.train import get_parser, get_arguments, automatic_argument_assignment, configure_env
    from myGym.envs import env_object
    import os
    import importlib.resources as pkg_resources

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
    
    arg_dict["gui"] = 1
    arg_dict = automatic_argument_assignment(arg_dict)
    env = configure_env(arg_dict, model_logdir=None, for_train=0)
    env = env.unwrapped
    table = env.static_scene_objects[env.workspace]

    pth1 = os.path.join(pkg_resources.files("myGym"), "envs/objects/household/urdf/tuna_can.urdf")
    pos = env_object.EnvObject.get_random_object_position([-0.5, 0.5, 0.4, 0.6, 0.07, 0.07])

    obj1 = env_object.EnvObject(
        pth1,
        pos,
        [0, 0, 0, 1],
        pybullet_client=env.p,
        fixed=False
    )

    touching = Touching()
    on_top = OnTop()

    print("Object reachable:")
    print(IsReachable().check(env.robot, obj1))
    print("Tuna touching table:")
    print(touching.get_value(obj1, table))
    print("Tuna on top of table:")
    print(on_top.get_value(obj1, table))
    print("Table on top of tuna:")
    print(on_top.get_value(table, obj1))

    env.reset()
