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

def aabb_overlap(min_a: Point3D, max_a: Point3D, min_b: Point3D, max_b: Point3D,
                 dims: int = 3, tolerance: float = 0.0) -> bool:
    """
    Return True if two AABBs overlap (or are within tolerance)

    dims=2 checks overlap only in x, y
    dims=3 checks overlap in x, y, and z
    """
    return all(
        min_a[i] <= max_b[i] + tolerance and
        max_a[i] >= min_b[i] - tolerance
        for i in range(dims)
    )

def get_aabb_distance(min_a: Point3D, max_a: Point3D,
    min_b: Point3D, max_b: Point3D,) -> float:
    """
    Return min Euclidean distance between two AABBs, 0 if overlap
    """
    dx = max(min_b[0] - max_a[0], min_a[0] - max_b[0], 0.0)
    dy = max(min_b[1] - max_a[1], min_a[1] - max_b[1], 0.0)
    dz = max(min_b[2] - max_a[2], min_a[2] - max_b[2], 0.0)

    return float(np.sqrt(dx * dx + dy * dy + dz * dz))

def get_point_distance(pos1: Point3D, pos2:Point3D) -> float:
    """
    Return Euclidean distance between two 3D points
    """
    pos1 = np.asarray(pos1, dtype=float)
    pos2 = np.asarray(pos2, dtype=float)

    return float(np.linalg.norm(pos1 - pos2))

def pos_inside_area(pos: Point3D, area: Area|BBox) -> bool:
    """
    Return True if a position is inside area
    """
    if isinstance(area[0], (int, float)):
        return (
            area[0] <= pos[0] <= area[1] and
            area[2] <= pos[1] <= area[3] and
            area[4] <= pos[2] <= area[5]
        )
    else:
        return (
            area[0][0] <= pos[0] <= area[1][0] and
            area[0][1] <= pos[1] <= area[1][1] and
            area[0][2] <= pos[2] <= area[1][2]
        )

def get_bounding_box_limits(obj) -> BBox:
    """
    Return AABB min and max corners for object

    For table_complex, manually defined desk area is used instead of
    full object bounding box
    """
    ws_dict = get_workspace_dict()
    if obj.name in ws_dict:
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

def get_desk_sampling_area(table_obj):
    """
    Return safety xy border for placing objects on the table desk
    """
    # TODO border_size needs tuning
    ws_dict = get_workspace_dict()
    sampling_border = ws_dict[table_obj.name]["desk_sampling_border"]
    obj_min, obj_max = get_desk_bounding_box(table_obj)

    obj_min = np.array(obj_min) + np.array(sampling_border)
    obj_max = np.array(obj_max) - np.array(sampling_border)
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
            return [0.2, 0.6, -0.4, 0.4, -0.07, 0.4]

        if "tiago" in robot.name or "nico" in robot.name:
            return [-0.1, 0.5, 0.15, 0.8, 0.6, 1.5]

        return [-0.7, 0.7, 0.1, 0.8, -0.1, 1.2]


class Touching(Predicate):
    """
    Check whether two objects are touching
    """
    TOLERANCE = 0.01 # TODO tune and move somewhere alse

    def check(self, obj1, obj2) -> bool:
        """
        Return True if AABB overlap (with tolerance)
        """
        obj1_min, obj1_max = get_bounding_box_limits(obj1)
        obj2_min, obj2_max = get_bounding_box_limits(obj2)
        t = self.TOLERANCE
        if aabb_overlap(obj1_min, obj1_max, obj2_min, obj2_max, tolerance=t):
            return True

        # NOTE: just for double check: doesnt work for some object comtinations
        # PyBullet contacts
        contact_points = obj1.p.getContactPoints(
            bodyA=obj1.uid,
            bodyB=obj2.uid,
        )
        return len(contact_points) > 0
        

class OnTop(AreaPredicate):
    """
    Check whether object1 is on to of object2
    """
    # magic numbers
    # TODO find more accurate ones and save it to helpers maybe
    PLACING_MARGIN = 0.007  # seems small but worked with different objects
    TOLERANCE = 0.02        # might be too forgiving but works for now

    def check(self, obj1, obj2) -> bool:
        """
        Return True if obj1 is on top of obj2
        """
        obj1_min, obj1_max = get_bounding_box_limits(obj1)
        obj2_min, obj2_max = get_bounding_box_limits(obj2)
        obj1_center_x = (obj1_min[0] + obj1_max[0]) / 2
        obj1_center_y = (obj1_min[1] + obj1_max[1]) / 2

        # 1. obj1 bottom close to obj2 top
        bottom_is_near_top = abs(obj1_min[2] - obj2_max[2]) < self.TOLERANCE

        # 2. obj1 AABB center inside obj2 xy bounds
        center_inside_support_xy = (
            obj2_min[0] <= obj1_center_x <= obj2_max[0] and
            obj2_min[1] <= obj1_center_y <= obj2_max[1]
        )

        return bottom_is_near_top and center_inside_support_xy

    def compute_area(self, obj1_urdf: str, obj2) -> Area:
        """
        Return sampling area for obj1 origin so that obj1 is placed on top of obj2
        """
        obj1_bottom_offset = self._get_bottom_offset_from_urdf(obj1_urdf, obj2.p)
        placing_height = obj1_bottom_offset + self.PLACING_MARGIN
        ws_dict = get_workspace_dict()

        if obj2.name in ws_dict:
            # place at random pos on top of the table
            xy_min, xy_max = get_desk_sampling_area(obj2)
            placing_height += xy_max[2]

        else:
            # place at the xy center of the object
            xy_min = obj2.get_position()
            xy_max = xy_min
            # place on top
            _ , obj2_max = get_bounding_box_limits(obj2)
            placing_height += obj2_max[2]

        sampling_area = [xy_min[0], xy_max[0],
                         xy_min[1], xy_max[1],
                         placing_height, placing_height,]
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
        pybullet_client.removeBody(temp_obj.uid)

        bottom_offset = obj_pos[2] - obj_min[2]
        return bottom_offset


class Inside(Predicate):
    """
    Check whether object1 is inside of object2
    """

    def check(self, obj1, obj2) -> bool:
        """
        Return True if obj1 bottom is inside of obj2 bounding box
        """
        obj1_min, obj1_max = get_bounding_box_limits(obj1)
        obj2_min, obj2_max = get_bounding_box_limits(obj2)
        obj1_center_x = (obj1_min[0] + obj1_max[0]) / 2
        obj1_center_y = (obj1_min[1] + obj1_max[1]) / 2

        obj1_bottom = [obj1_center_x, obj1_center_y, obj1_min[2]]
        obj2_area = [obj2_min, obj2_max]
        return pos_inside_area(obj1_bottom, obj2_area)


class Near(Predicate):
    """
    Check whether obj1 is close to obj2
    """
    MAX_DIST = 0.02  # TODO magic number

    def check(self, obj1, obj2) -> bool:
        """
        Return True if obj1 and obj2 are close
        """
        obj1_min, obj1_max = get_bounding_box_limits(obj1)
        obj2_min, obj2_max = get_bounding_box_limits(obj2)
        distance = get_aabb_distance(obj1_min, obj1_max, obj2_min, obj2_max)
        max_dist = self.MAX_DIST
        #print("distance", distance)
        return distance < max_dist


class ObjectAt(Predicate):
    """
    Check whether obj1/gripper is almost at the same position as obj2
    """
    E = 0.02  # TODO magic number, not tuned

    def check(self, obj, target_obj) -> bool:
        """
        Return True if obj1 and obj2 are almost at the same position
        """
        e = self.E
        obj1_pos = obj.get_position()
        obj2_pos = target_obj.get_position()
        distance = get_point_distance(obj1_pos, obj2_pos)
        # separate dist when placing obj for z based on obj height?
        return distance < e


class GripperStatus(Predicate):
    """
    Check whether the gripper is in the desired state

    Supports predicates GripperOpen() and GripperClosed()

    Note:
        GripperClosed() is not equivalent to not(GripperOpen()),
        the gripper has also a neutral state
    """

    def check(self, gripper, desired_status) -> bool:
        """
        Return True if gripper status == desired status (open/close)
        """
        gripper_states = gripper.get_gjoints_states()
        status, _ = gripper.check_gripper_status(gripper_states)
        return status == desired_status


class IsHolding(Predicate):
    """
    Check whether gripper is holding object
    """
    def check(self, gripper, obj) -> bool:
        return obj in gripper.holding



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

        return [p for p in predicates if self._predicate_has_obj_as_first_arg(p, obj_name)]

    @staticmethod
    def _check_predicate(predicate: PredicateCall, objects_by_name: dict, env,
                         )-> bool:
        """
        Check a predicate after objects have already been placed.
        """
        if predicate.predicate == "Reachable":
            obj_name = predicate.args[0]
            return IsReachable().check(env.robot, objects_by_name[obj_name])

        if predicate.predicate == "OnTop":
            obj_name, support_name = predicate.args
            return OnTop().check(objects_by_name[obj_name], objects_by_name[support_name])
        
        if predicate.predicate == "Touching":
            obj1_name, obj2_name = predicate.args
            return Touching().check(objects_by_name[obj1_name], objects_by_name[obj2_name])
        
        if predicate.predicate == "Near":
            obj1_name, obj2_name = predicate.args
            return Near().check(objects_by_name[obj1_name], objects_by_name[obj2_name])

        if predicate.predicate == "ObjectAt":
            obj_name, target_name = predicate.args
            return ObjectAt().check(objects_by_name[obj_name], objects_by_name[target_name])
        
        if predicate.predicate == "GripperAt":
            target_name = predicate.args[0]
            return ObjectAt().check(objects_by_name["robot"], objects_by_name[target_name])
        
        if predicate.predicate == "GripperClosed":
            return GripperStatus().check(objects_by_name["robot"], "close")
        
        if predicate.predicate == "GripperOpen":
            return GripperStatus().check(objects_by_name["robot"], "open")
        
        if predicate.predicate == "IsHolding":
            target_name = predicate.args[0]
            return IsHolding().check(env.robot, objects_by_name[target_name])

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
    def _predicate_has_obj_as_first_arg(predicate: str, obj_name: str) -> bool:
        """
        Return True if obj_name is the first predicate argument.
        """
        args_start = predicate.find("(")
        args_end = predicate.rfind(")")

        if args_start == -1 or args_end == -1:
            return False

        args_text = predicate[args_start + 1:args_end]
        args = [arg.strip() for arg in args_text.split(",")]

        if not args:
            return False

        return args[0] == obj_name


class InitPredicateResolver(PredicateResolver):
    """
        Resolve predicates describing the initial state
    """

    predicate_key = "init"

    def get_area(self, obj_info, table, robot, predicates, placed_objects=None) -> Area | None:
        """
        Compute object sampling area from init predicates
        """
        predicates = predicates["init"] if predicates else []
        placed_objects = placed_objects or {}
        placed_objects.setdefault("table", table)
        placed_objects.setdefault("workspace", table)
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
                placed_objects=placed_objects,
            )

        for reachable_predicate in predicate_map.get("Reachable", []):
            area = self._apply_reachable_area(
                current_area=area,
                predicate=reachable_predicate,
                robot=robot,
            )
            break  # repetitive input

        return area
    

    def _apply_on_top_area(
            self, current_area: Area, predicate: PredicateCall, table, obj1_urdf: str, placed_objects
            ) -> Area | None:
        """
        Apply OnTop(obj1, obj2) as an area constraint
        The support object has to be already placed
        """
        if len(predicate.args) != 2:
            raise ValueError(f"OnTop expects 2 arguments, got {predicate.args}")

        obj1_name, obj2_name = predicate.args

        if obj2_name == "table":
            support_object = table
        else:
            support_object = placed_objects.get(obj2_name)
            #print(placed_objects)
            if not support_object:
                raise ValueError(
                    f"Cannot compute OnTop area for '{predicate.args[0]}'. "
                    f"Supporting object '{obj2_name}' has not been placed yet."
                )
        
        on_top_area = OnTop().compute_area(obj1_urdf, support_object)
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
    

    def get_placement_order(self, objects_to_place, predicates):
        """
        Sort placement records according to initial predicate dependencies.

        Example:
            OnTop(apple, banana)

        means:
            banana must be placed before apple.

        objects_to_place is a list of placement records:
            {
                "obj_info": {...},
                "role": ...,
                "target": ...,
                "state_name": ...
            }
        """
        init_predicates = predicates["init"] if predicates else []
        predicate_calls = self._parse_predicates(init_predicates)
        predicate_map = self._get_predicate_map(predicate_calls)

        records_by_name = {
            record["obj_info"]["obj_name"]: record
            for record in objects_to_place
            if record["obj_info"]["obj_name"] != "null"
        }

        object_names = set(records_by_name.keys())

        dependencies = {
            obj_name: set()
            for obj_name in object_names
        }

        for predicate in predicate_map.get("OnTop", []):
            upper_obj = predicate.args[0]
            support_obj = predicate.args[1]

            if upper_obj not in object_names:
                continue

            if support_obj in object_names:
                dependencies[upper_obj].add(support_obj)

        return self._topological_sort_records(
            records_by_name=records_by_name,
            dependencies=dependencies,
        )


    def _topological_sort_records(self, records_by_name, dependencies):
        """
        Return placement records sorted so that support objects are placed first.
        """
        sorted_records = []
        temporary_marks = set()
        permanent_marks = set()

        def visit(obj_name):
            if obj_name in permanent_marks:
                return

            if obj_name in temporary_marks:
                raise ValueError(
                    f"Cyclic OnTop dependency detected around object '{obj_name}'."
                )

            temporary_marks.add(obj_name)

            for dependency_name in dependencies[obj_name]:
                visit(dependency_name)

            temporary_marks.remove(obj_name)
            permanent_marks.add(obj_name)
            sorted_records.append(records_by_name[obj_name])

        for obj_name in records_by_name:
            visit(obj_name)

        return sorted_records


class GoalPredicateResolver(PredicateResolver):
    """
    Check predicates describing the goal state.
    """
    predicate_key = "goal"

