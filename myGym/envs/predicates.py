import numpy as np
import re
from dataclasses import dataclass
from abc import ABC, abstractmethod

from myGym.envs import env_object
from myGym.utils.helpers import get_workspace_dict


# Area format: [x_min, x_max, y_min, y_max, z_min, z_max]
Area = list[float]

# Point/corner point format: [x, y, z] | (x, y, z)
Point3D = list[float] | tuple[float, float, float]

# Bounding box format: ((x_min, y_min, z_min), (x_max, y_max, z_max))
BBox = tuple[Point3D, Point3D]


# ---------- area / geometry helpers ----------

def get_infinite_area() -> Area:
    """
    Return an unconstrained 3D area
    """
    return [
        -float("inf"), float("inf"),
        -float("inf"), float("inf"),
        -float("inf"), float("inf"),
    ]

def get_range_intersection(area_a: Area, area_b: Area) -> Area|None:
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

def aabb_overlap(min_a: Point3D, max_a: Point3D, 
                 min_b: Point3D, max_b: Point3D, dims: int = 3
                 ) -> bool:
    """
    Return True if two AABBs overlap

    dims=2 checks overlap only in x, y
    dims=3 checks overlap in x, y, and z
    """
    return all(
        min_a[i] <= max_b[i] and max_a[i] >= min_b[i]
        for i in range(dims)
    )

def pos_inside_area(pos: Point3D, area: Area) -> bool:
    """
    Return True if a position is inside area
    """
    return (
        area[0] <= pos[0] <= area[1] and
        area[2] <= pos[1] <= area[3] and
        area[4] <= pos[2] <= area[5]
    )

def get_bounding_box_limits(obj) -> BBox:
    """
    Return AABB min and max corners for object

    For table_complex, manually defined desk area is used instead of
    full object bounding box
    """
    if obj.name == "table_complex":
        obj_min, obj_max = get_desk_bounding_box(obj)
    else:
        obj_min, obj_max = obj.get_bounding_box()[0], obj.get_bounding_box()[4]
    return obj_min, obj_max

def get_desk_area(table_obj) -> Area:
    """
    Return manually defined desk operation area
    The table rotation is not applied.
    """
    ws_dict = get_workspace_dict()
    desk_dim = np.array(ws_dict[table_obj.name]["desk_dim"])
    table_pos = np.repeat(table_obj.get_position(), 2)
    desk_area = table_pos + desk_dim
    return desk_area.tolist()

def get_desk_bounding_box(table_obj) -> BBox:
    """
    Return min and max corners of the manually defined desk area
    """
    desk_area = get_desk_area(table_obj)
    obj_min = (desk_area[0], desk_area[2], desk_area[4])
    obj_max = (desk_area[1], desk_area[3], desk_area[5])
    return obj_min, obj_max



# ---------- predicate classes ----------

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

class AreaPredicate(Predicate):
    """
    Base class for predicates that can also produce sampling area
    """

    @abstractmethod
    def compute_area(self, *args) -> Area:
        """
        Compute sampling area for the predicate to be satisfied
        """
        raise NotImplementedError



class IsReachable(AreaPredicate):
    """
    Check if object lies inside the precomputed gripper's reachable envelope
    ! does not solve IK and does not check collisions
    """

    def check(self, robot, obj) -> bool:
        """
        Return True if object is inside reachable area
        """
        reachable_area = self.compute_area(robot)
        obj_position = obj.get_position()
        return pos_inside_area(obj_position, reachable_area)
    
    def compute_area(self, robot) -> Area:
        """
        Return reachable area
        """
        reachable_range = np.array(self._get_reachable_range(robot))
        robot_pos = np.repeat(list(robot.position), 2)
        reachable_area = robot_pos + reachable_range
        return reachable_area.tolist()

    @staticmethod
    def _get_reachable_range(robot) -> Area:
        # old helper from PRAG
        # TODO: get the area from test_robot_reachability.py and save it to workspace_dict

        if robot.name == "g1":
            return [0.2, 0.7, -0.4, 0.4, -0.07, 0.4]

        if "tiago" in robot.name or "nico" in robot.name:
            return [-0.1, 0.5, 0.15, 0.8, 0.6, 1.5]

        return [-0.7, 0.7, 0.1, 0.8, -0.1, 1.2]


class Touching(Predicate):
    """
    Check whether two objects are touching
    """

    def check(self, obj1, obj2) -> bool:
        """
        Uses PyBullet contacts when possible.
        Falls back to AABB overlap, which is useful for fixed objects.
        """

        if not obj1.fixed or not obj2.fixed:
            contact_points = obj1.p.getContactPoints(
                bodyA=obj1.uid,
                bodyB=obj2.uid,
            )
            return len(contact_points) > 0
        
        # Fallback for fixed objects or not-yet-updated contacts.
        obj1_min, obj1_max = get_bounding_box_limits(obj1)
        obj2_min, obj2_max = get_bounding_box_limits(obj2)

        return aabb_overlap(obj1_min, obj1_max, obj2_min, obj2_max)


class OnTop(AreaPredicate):
    """
    Check whether object1 is on to of object2
    """
    # magic numbers
    # TODO find more accurate ones and save it to helpers maybe
    PLACING_BORDER = 0.01   # TODO random number, needs tuning
    PLACING_MARGIN = 0.005  # seems very small but it worked with different objects
    TOLERANCE = 0.02        # might be too forgiving but works for now

    def check(self, obj1, obj2) -> bool:
        """
        Return True if obj1 is on top of obj2
        """
        # 1. obj1 and obj2 are touching
        if not Touching().check(obj1, obj2):
            return False
        
        # 2. bottom of obj1 is near top of obj2
        obj1_min = obj1.get_bounding_box()[0][2]
        if obj2.name == "table_complex":
            obj2_max = get_desk_area(obj2)[-1]
        else:
            obj2_max = obj2.get_bounding_box()[4][2]

        bottom_is_near_top = abs(obj1_min - obj2_max) < self.TOLERANCE
        #print("diff:", abs(obj1_min - obj2_max))
        return bottom_is_near_top

        # the overlap is ignored for now
        # 3. their x/y projections overlap
        obj1_min, obj1_max = get_bounding_box_limits(obj1)
        obj2_min, obj2_max = get_bounding_box_limits(obj2)
        return aabb_overlap(obj1_min, obj1_max, obj2_min, obj2_max, dims=2)

    def compute_area(self, obj1_urdf: str, obj2) -> Area:
        """
        Return sampling area for obj1 origin so that obj1 is placed on top of obj2
        """
        obj2_min, obj2_max = get_bounding_box_limits(obj2)
        obj1_bottom_offset = self._get_bottom_offset_from_urdf(
            obj1_urdf,
            obj2.p,
        )

        placing_height = obj1_bottom_offset + self.PLACING_MARGIN
        placing_border = self.PLACING_BORDER

        sampling_area = [obj2_min[0] + placing_border, obj2_max[0] - placing_border,
                         obj2_min[1] + placing_border, obj2_max[1] - placing_border,
                         obj2_max[2] + placing_height, obj2_max[2] + placing_height,]
        
        return sampling_area

    @staticmethod
    def _get_bottom_offset_from_urdf(obj1_urdf: str, pybullet_client) -> float:
        temp_obj = env_object.EnvObject(
            obj1_urdf,
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            pybullet_client=pybullet_client,
            fixed=True,
        )

        obj_min = temp_obj.get_bounding_box()[0]
        obj_pos = temp_obj.get_position()

        bottom_offset = obj_pos[2] - obj_min[2]

        pybullet_client.removeBody(temp_obj.uid)

        return bottom_offset



# ---------- predicate parsing / resolving ----------

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
            return OnTop().check(objects_by_name[obj_name], objects_by_name[support_name])
        
        if predicate.predicate == "Touching":
            obj_name, support_name = predicate.args
            return Touching().check(objects_by_name[obj_name], objects_by_name[support_name])

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
        #print(obj_info["obj_name"], "predicates:", predicates)

        if not predicates:
            random_table_area = OnTop().compute_area(obj1_urdf, table)
            return random_table_area
        
        area = get_infinite_area()
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

        on_top_area = OnTop().compute_area(obj1_urdf, table)

        return get_range_intersection(current_area, on_top_area)

    def _apply_reachable_area(
        self, current_area: Area, predicate: PredicateCall, robot
        ) -> Area | None:
        """
        Apply Reachable(obj) as an area constraint
        """
        if len(predicate.args) != 1:
            raise ValueError(f"Reachable expects 1 argument, got {predicate.args}")

        reachable_area = IsReachable().compute_area(robot)
        final_area = get_range_intersection(current_area, reachable_area)
        return final_area


class GoalPredicateResolver(PredicateResolver):
    """
    Check predicates describing the goal state.
    """
    predicate_key = "goal"





if __name__ == '__main__':
    import os
    import importlib.resources as pkg_resources
    from myGym.train import get_parser, get_arguments, automatic_argument_assignment, configure_env
    
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
    

    arg_dict = _parse()

    arg_dict["gui"] = 1
    arg_dict = automatic_argument_assignment(arg_dict)
    env = configure_env(arg_dict, model_logdir=None, for_train=0)
    env = env.unwrapped
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

