import numpy as np
import re
from dataclasses import dataclass
from abc import ABC, abstractmethod

from myGym.envs import env_object
from myGym.utils.helpers import get_workspace_dict


Area = list[float]



class Predicate(ABC):
    """
    Base class for predicates
    """

    @abstractmethod
    def check(self, *args) -> bool:
        """
        Return True if the predicate is satisfied
        """
        raise NotImplementedError



class IsReachable(Predicate):
    """
    Check if object lies inside the precomputed gripper's reachable envelope
    ! does not solve IK and does not check collisions
    """

    def compute_area(self, robot) -> Area:
        """
        Return reachable area
        """
        reachable_range = np.array(self._get_reachable_range(robot))
        robot_pos = np.repeat(list(robot.position), 2)
        reachable_area = robot_pos + reachable_range

        return reachable_area.tolist()
        

    def check(self, robot, obj) -> bool:
        """
        Return True if object is inside reachable area
        """
        reachable_area = self.compute_area(robot)
        obj_position = obj.get_position()
        return self._obj_inside_area(obj_position, reachable_area)
    

    @staticmethod
    def _obj_inside_area(coords, area: Area) -> bool:
        """
        Return True if object is inside area
        """
        return (
            area[0] <= coords[0] <= area[1]
            and area[2] <= coords[1] <= area[3]
            and area[4] <= coords[2] <= area[5]
        )
    

    @staticmethod
    def _get_reachable_range(robot) -> Area:
        # old helper from PRAG
        # TODO: get the area from test_robot_reachability.py and save it to workspace_dict

        if robot.name == "g1":
            return [0.2, 0.5, -0.5, 0.5, -0.07, 0.6]

        if "tiago" in robot.name or "nico" in robot.name:
            return [-0.1, 0.5, 0.15, 0.8, 0.6, 1.5]

        return [-0.7, 0.7, 0.1, 0.8, -0.1, 1.2]



class Touching(Predicate):
    """
    Check whether two objects are physically touching
    """

    def compute_area(self, obj1, obj2) -> Area:
        raise NotImplementedError()
        # TODO future base for OnTheLeft predicate?
        """
        Return a target area for the center of obj1 such that obj1 touches obj2.

        The area is computed by expanding obj2's AABB by half of obj1's size.
        If obj1's center is inside this area, their AABBs overlap or touch.
        """
        obj1_min, obj1_max = obj1.get_bounding_box()[0], obj1.get_bounding_box()[4]
        obj2_min, obj2_max = obj2.get_bounding_box()[0], obj2.get_bounding_box()[4]

        obj1_size = np.asarray(obj1_max) - np.asarray(obj1_min)
        obj1_half_size = obj1_size / 2

        target_area = [
            obj2_min[0] - obj1_half_size[0],
            obj2_max[0] + obj1_half_size[0],
            obj2_min[1] - obj1_half_size[1],
            obj2_max[1] + obj1_half_size[1],
            obj2_min[2] - obj1_half_size[2],
            obj2_max[2] + obj1_half_size[2],
        ]

        return target_area


    def check(self, obj1, obj2) -> bool:
        """
        Return True if obj1 and obj2 touch (using PyBullet)
        """
        contact_points = obj1.p.getContactPoints(
            bodyA=obj1.uid,
            bodyB=obj2.uid,
        )
        return len(contact_points) > 0
    


class OnTop(Predicate):
    """
    Check whether object1 is on to of object2
    """

    def compute_area(self, obj2, obj1_urdf=None) -> Area:
        """
        Return the top area of obj2 so obj1 could be placed on top
        """
        # TODO placing based on the object height using _read_urdf_scale()
        if obj2.name == "table_complex":
            desk_area = self._get_desk_area(obj2)
            PLACING_SPACE = [-0., 0., -0., 0., 0.05, 0.05]
            return (np.asarray(desk_area) + np.asarray(PLACING_SPACE)).tolist()

        PLACING_BORDER = 0.02
        obj2_min, obj2_max = obj2.get_bounding_box()[0], obj2.get_bounding_box()[4]

        above_obj2 = [obj2_min[0] + PLACING_BORDER,    obj2_max[0] - PLACING_BORDER,
                      obj2_min[1] + PLACING_BORDER,    obj2_max[1] - PLACING_BORDER,
                      obj2_max[2] + 20*PLACING_BORDER, obj2_max[2] + 20*PLACING_BORDER,]
        
        return above_obj2
    

    def check(self, obj1, obj2) -> bool:
        """
        Return True if obj1 is on top of obj2
        """
        print("OnTop.check:", obj1.name, obj2.name)
        # 1. obj1 and obj2 are touching
        if not Touching().check(obj1, obj2):
            return False
        
        # 2. bottom of obj1 is near top of obj2
        obj1_min = obj1.get_bounding_box()[0][2]
        if obj2.name == "table_complex":
            obj2_max = self._get_desk_area(obj2)[-1]
        else:
            obj2_max = obj2.get_bounding_box()[4][2]

        tolerance = 0.02
        bottom_is_near_top = abs(obj1_min - obj2_max) < tolerance
        print(abs(obj1_min - obj2_max))
        return bottom_is_near_top

        # the overlap is ignored for now
        # 3. their x/y projections overlap
        obj1_min, obj1_max = obj1.get_bounding_box()[0], obj1.get_bounding_box()[4]
        obj2_min, obj2_max = obj2.get_bounding_box()[0], obj2.get_bounding_box()[4]
        x_overlap = obj1_max[0] >= obj2_min[0] and obj1_min[0] <= obj2_max[0]
        y_overlap = obj1_max[1] >= obj2_min[1] and obj1_min[1] <= obj2_max[1]
        return x_overlap and y_overlap


    @staticmethod
    def _get_desk_area(obj2) -> Area:
        """
        Get desk operation area
        ! the table rotation is not applied
        """
        ws_dict = get_workspace_dict()
        desk_dim = np.array(ws_dict[obj2.name]["desk_dim"])
        table_pos = np.repeat(obj2.get_position(), 2)
        desk_area = table_pos + desk_dim
        return desk_area.tolist()
    

    @staticmethod
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



@dataclass
class PredicateCall:
    predicate: str
    args: list[str]

    
class PredicateResolver:
    """
    Base class to check if all predicates are satisfied
    """

    predicate_key: str | None = None

    def check(self, placed_objects: dict, env, predicates: dict) -> bool:
        """
        Return True if all of the selected predicates are satisfied
        """
        if self.predicate_key is None:
            raise NotImplementedError("predicate_key must be defined in child class")

        selected_predicates = predicates.get(self.predicate_key, []) if predicates else []

        if not selected_predicates:
            # no restriction for object placement
            return True

        objects_by_name = self._build_object_lookup(env, placed_objects)

        for predicate in self._parse_predicates(selected_predicates):
            if not self._check_predicate(predicate, objects_by_name, env):
                return False

        return True
    
    
    @staticmethod
    def _get_predicate_map(predicate_calls: list[PredicateCall]
                           ) -> dict[str, list[PredicateCall]]:
        """
        Convert list of PredicateCall objects to dict by predicate name
        [PredicateCall("OnTop", ["apple", "table"]), ...] -> {"OnTop": [PredicateCall(...), ...}
        """
        predicate_map = {}

        for predicate_call in predicate_calls:
            predicate_map.setdefault(predicate_call.predicate, []).append(predicate_call)

        return predicate_map
    

    @staticmethod
    def _parse_predicates(predicates: list[str]) -> list[PredicateCall]:
        """ 
        Convert list of predicates tsrings to list predicates objects
        ['OnTop(apple,table)', ...] -> [PredicateCall(name='OnTop', args=['apple','table']), ...]
        """
        parsed_predicates = []
        for predicate in predicates:
            match = re.fullmatch(r"\s*(\w+)\s*\((.*)\)\s*", predicate)

            if match is None:
                raise ValueError(f"Invalid predicate format: {predicate}")

            name = match.group(1)
            args_text = match.group(2)
            args = [arg.strip() for arg in args_text.split(",") if arg.strip()]

            parsed_predicates.append(PredicateCall(predicate=name, args=args))

        return parsed_predicates
    

    def _filter_obj_predicates(self, predicates, obj_name):
        """
        Select current object predicates
        """
        if not predicates:
            return []

        return [p for p in predicates if self._predicate_contains_obj(p, obj_name)]
    

    @staticmethod
    def _check_predicate(predicate: PredicateCall, objects_by_name: dict, env,
                         )-> bool:
        """predicates
        Check a predicate after objects have already been placed.
        """
        if predicate.predicate == "Reachable":
            obj_name = predicate.args[0]
            return IsReachable().check(env.robot, objects_by_name[obj_name])

        if predicate.predicate == "OnTop":
            obj_name, support_name = predicate.args
            print(OnTop().check(objects_by_name[obj_name], objects_by_name[support_name]))
            return OnTop().check(objects_by_name[obj_name], objects_by_name[support_name])

        """if predicate.predicate == "ObjectAt":
            obj_name, target_name = predicate.args
            return ObjectAt().check(objects_by_name[obj_name], objects_by_name[target_name])"""

        raise ValueError(f"Unknown predicate: {predicate.predicate}")


    @staticmethod
    def _build_object_lookup(env, placed_objects: dict) -> dict:
        """
        Build name -> object lookup for predicates.
        """
        objects_by_name = {}

        # Active task objects.
        for obj in placed_objects.values():
            if hasattr(obj, "name"):
                objects_by_name[obj.name] = obj

        # Static scene aliases.
        objects_by_name["table"] = env.static_scene_objects[env.workspace]
        objects_by_name[env.workspace] = env.static_scene_objects[env.workspace]

        # Robot alias.
        objects_by_name["robot"] = env.robot
        objects_by_name["gripper"] = env.robot

        return objects_by_name
    

    @staticmethod
    def _predicate_contains_obj(predicate: str, obj_name: str) -> bool:
        """
        Return True if obj_name is one of the predicate arguments
        """
        args_start = predicate.find("(")
        args_end = predicate.rfind(")")

        if args_start == -1 or args_end == -1:
            return False

        args_text = predicate[args_start + 1:args_end]
        args = [arg.strip() for arg in args_text.split(",")]

        return obj_name in args


    @staticmethod
    def _get_range_intersection(area_a: Area, area_b: Area) -> Area|None:
        """
        Return intersection of two 3D areas, None if no overlap
        """
        intersection = []

        for i in range(0, 6, 2):
            min_val = max(area_a[i], area_b[i])
            max_val = min(area_a[i + 1], area_b[i + 1])

            if min_val > max_val:
                # No overlap for this axis
                return None

            intersection.extend([min_val, max_val])

        return intersection


    @staticmethod
    def _get_infinite_area() -> Area:
        return [
            -float("inf"), float("inf"),
            -float("inf"), float("inf"),
            -float("inf"), float("inf"),
        ]



class InitPredicateResolver(PredicateResolver):
    """
        Resolve predicates describing the initial state
    """

    predicate_key = "init"
    

    def get_area(self, obj_info, table, robot, predicates) -> Area | None:
        """
        Compute object sampling area from init predicates
        """
        predicates = predicates["init"] if predicates else []
        obj1_urdf = obj_info["urdf"]
        predicates = self._filter_obj_predicates(predicates, obj_info["obj_name"])
        print(obj_info["obj_name"], "predicates:", predicates)

        if not predicates:
            random_table_area = OnTop().compute_area(table, obj1_urdf)
            return random_table_area
        
        area = self._get_infinite_area()
        predicate_calls = self._parse_predicates(predicates)
        predicate_map = self._get_predicate_map(predicate_calls)
        
        for on_top_predicate in predicate_map.get("OnTop", []):
            area = self._apply_on_top_area(
                current_area=area,
                predicate=on_top_predicate,
                table=table,
                obj1_urdf=obj1_urdf,
            )

        for reachable_predicate in predicate_map.get("Reachable", []):
            area = self._apply_reachable_area(
                current_area=area,
                predicate=reachable_predicate,
                robot=robot,
            )
            break  # repetitive input

        # near_predicate = predicate_map.get("Near")
        # if near_predicate is not None:
        #     area = self._apply_near_area(...)
        return area
    

    def _apply_on_top_area(
            self, current_area: Area, predicate: PredicateCall, table, obj1_urdf: str
            ) -> Area | None:
        """
        Apply OnTop(obj1, obj2) as an area constraint
        ! currently supports only obj2 == table
        """
        if len(predicate.args) != 2:
            raise ValueError(f"OnTop expects 2 arguments, got {predicate.args}")

        obj1_name, obj2_name = predicate.args

        if obj2_name != "table":
            print(f"OnTop({obj1_name}, {obj2_name}) cannot be resolved during initial placement yet.")
            return current_area

        on_top_area = OnTop().compute_area(table, obj1_urdf)

        return self._get_range_intersection(current_area, on_top_area)


    def _apply_reachable_area(
        self, current_area: Area, predicate: PredicateCall, robot
        ) -> Area | None:
        """
        Apply Reachable(obj) as an area constraint
        """
        if len(predicate.args) != 1:
            raise ValueError(f"Reachable expects 1 argument, got {predicate.args}")

        reachable_area = IsReachable().compute_area(robot)
        final_area = self._get_range_intersection(current_area, reachable_area)
        return final_area


class GoalPredicateResolver(PredicateResolver):
    """
    Check predicates describing the goal state.
    """
    predicate_key = "goal"





if __name__ == '__main__':
    import os
    import importlib.resources as pkg_resources
    #import open3d as o3d
    from myGym.train import get_parser, get_arguments, automatic_argument_assignment, configure_env
    #from myGym.envs.test_volume_class import VolumeMesh


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
    pos = env_object.EnvObject.get_random_object_position([-0., 0.2, 0.4, 0.5, 0.07, 0.07])

    obj1 = env_object.EnvObject(
        pth1,
        pos,
        [0, 0, 0, 1],
        pybullet_client=env.p,
        fixed=False
    )

    for _ in range(100):
        # tuna falls on the table
        env.p.stepSimulation()

    touching = Touching()
    on_top = OnTop()

    print("Object reachable:")
    print(IsReachable().check(env.robot, obj1))
    print("Tuna touching table:")
    print(touching.check(obj1, table))
    print("Tuna on top of table:")
    print(on_top.check(obj1, table))
    print("Table on top of tuna:")
    print(on_top.check(table, obj1))


    """
    # load tuna model and check the scale
    obj1_info = pybullet.getVisualShapeData(obj1.get_uid())[0]
    obj1_scale = get_scale_from_urdf(pth1)
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
    """

    """
    pth1 = "./envs/objects/geometric/urdf/cube.urdf"
    pos = env_object.EnvObject.get_random_object_position([-0.5, 0.5, 0.4, 0.6, 0.1, 0.1])
    pan1 = env_object.EnvObject(pth1, pos, [0, 0, 0, 1], pybullet_client=p, fixed=False)
    pan1_info = pybullet.getVisualShapeData(pan1.get_uid())[0]
    pan1_scale = get_scale_from_urdf(pth1)
    objpth = pan1_info[4].decode("utf-8")
    o3model = o3d.io.read_triangle_model(objpth)
    mesh = o3model.meshes[0].mesh
    mesh = mesh.scale(pan1_scale, center=mesh.get_center())
    vm = VolumeMesh(mesh)
    orig = vm.duplicate()
    orig.paint(np.array([0, 1, 0]))
    voxel_grid = vm.voxelgrid
    o3d.visualization.draw_geometries([voxel_grid, orig.voxelgrid])
    pan1.paint([0,1,0,1])
    print("Green pan touching table:")
    print(touching.get_value(pan1, table))
    print("Green pan on top of table:")
    print(on_top.get_value(pan1, table))
    print("Table on top of green pan:")
    print(on_top.get_value(table, pan1))

    pos = env_object.EnvObject.get_random_object_position([-0.5, 0.5, 0.4, 0.6, 0.5, 0.55])
    pan2 = env_object.EnvObject("myGym/envs/objects/household/urdf/pan.urdf", pos, [0, 0, 0, 1], pybullet_client=p, fixed=False)
    pan2.set_color([1,0,0,1])
    print("Red pan touching table:")
    print(touching.get_value(pan2, table))
    print("")
    """

