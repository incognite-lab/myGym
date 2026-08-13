"""
this file contains
1. geometry helpers
2. predicate classes
3. predicate parsing/resolving


The implemented predicates are:
 unary predicates:
    IsReachable (G)
    Upright
    Empty
    GripperAt
     -see ObjectAt
    GripperStatus
     -GripperClosed, GripperOpen
    IsHolding

 binary predicates:
    Touching
    OnTop     (G)
    Inside    (G)
    Near
    Far
    ObjectAt  (G)
    Above     (G)
    Below     (G)
    LeftOf    (G)
    RightOf   (G)
    InFrontOF (G)
    Behind    (G)
    NextTo    (G)

 - (G) = able to generate sampling area for placing objects
 - gripper predicates are used without gripper parameter in config file - GripperAt(obj)

Predicate checkers:
 PredicateResolver and its child classes:
    InitPredicateResolver
     -able to generate area for placing objects
    GoalPredicateResolver
    SubgoalPredicateResolver
"""

"""
Problems with the current logic:TODO
 1. No negation of predicates generating area at init 
 2. Any area can only be rectangular
    - objs represented as bounding boxes
    - reachable area not accurate
    - position of already sampled objects cannot be excluded form sampling area
        -> hight chance of collision from sampling NextTo(obj1, obj2), it is solved
           by randomly selecting side, but in cost of big limitation (see NextTo)
"""

import os
import numpy as np
import random
import re
from dataclasses import dataclass
from abc import ABC, abstractmethod

from myGym.envs import env_object
from myGym.utils.helpers import get_workspace_dict, get_robot_dict


# Area format: [x_min, x_max, y_min, y_max, z_min, z_max]
Area = list[float]

# Point/corner point format: [x, y, z] | (x, y, z)
Point3D = list[float] | tuple[float, float, float]

# Bounding box format: ((x_min, y_min, z_min), (x_max, y_max, z_max))
BBox = tuple[Point3D, Point3D]

# TODO might need tuning
TOLERANCE = 0.02       # tolerance for touching bounding boxes
TOLERANCE_DEG = 15     # rotation tolerance for upright position
CLOSE = 0.02           # max dist to be close enough
FAR = 2                # min dist to be far enough
MAX_HEIGHT = 1         # max sampling height for above predicate
MIN_HEIGHT = -1        # max sampling height for under predicate


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

def get_unconstrained_working_area() -> Area:
    """
    Return an unconstrained 3D area for sampling objects
    """
    return [-2, 2,
            -2, 2,
            -2, 2]

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

def get_object_extent_from_urdf(obj1_urdf: str, pybullet_client, env=None) -> tuple[Point3D, Point3D, Point3D]:
    """
    Briefly spawn obj1 to measure its own bounding box and origin position.
    Used by predicates that need to know obj1's size before placing it
    (e.g. to offset a sampling boundary by obj1's own extent).
    Returns (obj_min, obj_max, obj_pos).
    """
    if os.path.splitext(os.path.basename(obj1_urdf))[0] == "towertarget":
        obj1_urdf = env._get_urdf_filename("kostka")

    temp_obj = env_object.EnvObject(
        obj1_urdf,
        position=[0.0, 0.0, 1.0],
        orientation=[0.0, 0.0, 0.0, 1.0],
        pybullet_client=pybullet_client,
        fixed=True,
    )

    obj_min, obj_max = get_bounding_box_limits(temp_obj)
    obj_pos = temp_obj.get_position()
    pybullet_client.removeBody(temp_obj.uid)

    return obj_min, obj_max, obj_pos

def get_edge_offset_from_urdf(obj1_urdf: str, pybullet_client, axis: int, edge: str, env=None) -> float:
    """
    Return the distance from obj1's own origin to one edge of its bounding
    box. axis: 0=X, 1=Y, 2=Z. edge: "min" or "max".
    """
    obj_min, obj_max, obj_pos = get_object_extent_from_urdf(obj1_urdf, pybullet_client, env)
    return (obj_pos[axis] - obj_min[axis]) if edge == "min" else (obj_max[axis] - obj_pos[axis])

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

def get_desk_sampling_area(table_obj) -> BBox:
    """
    Return area on top of table desk for sampling
    """
    # TODO sampling border needs tuning
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

    def check(self, robot, obj, grip_type: str | None) -> bool:
        """
        Return True if object can be reached
        """
        reachable_area = self.compute_area(robot, grip_type)
        obj_position = obj.get_position()
        return pos_inside_area(obj_position, reachable_area)

    def compute_area(self, robot, grip_type: str | None) -> Area:
        """
        Return reachable area for default robot position
        """
        reachable_area = np.array(self._get_reachable_range(robot, grip_type))
        return reachable_area.tolist()

    @staticmethod
    def _get_reachable_range(robot, grip_type: str | None) -> Area:
        key = IsReachable._reachable_key(grip_type)
        robot_ws = get_robot_dict().get(robot.name, {})

        if key in robot_ws:
            return robot_ws[key]

        if key != "reachable" and "reachable" in robot_ws:
            print(f"Warning: '{key}' not defined for robot '{robot.name}', falling back to 'reachable'")
            return robot_ws["reachable"]

        print(f"Warning: Reachable area not found for robot '{robot.name}'")
        return [0.2, 0.6, -0.4, 0.4, -0.07, 0.4]
    
    @staticmethod
    def _reachable_key(grip_type: str | None) -> str:
        """
        Translate a grip_type keyword from protorewards.json ("top", "left",
        "right", "back", "front", "bottom", "any") into the matching
        reachable_<grip_type> key used in helpers.py's ROBOTS dict.
        """
        if not grip_type or grip_type in ("any", "reachable"):
            return "reachable"

        return f"reachable_{grip_type}"


class Touching(Predicate):
    """
    Check whether two objects are touching
    """

    def check(self, obj1, obj2) -> bool:
        # NOTE: currently not in use, because the PyBullet contacts didn't work well enough
        """
        Return True if AABB overlap (with tolerance)
        """
        obj1_min, obj1_max = get_bounding_box_limits(obj1)
        obj2_min, obj2_max = get_bounding_box_limits(obj2)
        t = TOLERANCE
        if aabb_overlap(obj1_min, obj1_max, obj2_min, obj2_max, tolerance=t):
            return True

        # NOTE: just for double check: doesnt work for some object combinations
        # PyBullet contacts
        contact_points = obj1.p.getContactPoints(
            bodyA=obj1.uid,
            bodyB=obj2.uid,
        )
        return len(contact_points) > 0
        

class OnTop(AreaPredicate):
    """
    Check whether object1 is on top of object2
    """

    def check(self, obj1, obj2) -> bool:
        """
        Return True if obj1 is on top of obj2
        """
        if self.is_target_obj(obj1):
            return Above().check(obj1, obj2)
        
        obj1_min, obj1_max = get_bounding_box_limits(obj1)
        obj2_min, obj2_max = get_bounding_box_limits(obj2)
        obj1_center_x = (obj1_min[0] + obj1_max[0]) / 2
        obj1_center_y = (obj1_min[1] + obj1_max[1]) / 2

        # 1. obj1 bottom close to obj2 top
        bottom_is_near_top = abs(obj1_min[2] - obj2_max[2]) < TOLERANCE

        # 2. obj1 AABB center inside obj2 xy bounds
        center_inside_support_xy = (
            obj2_min[0] <= obj1_center_x <= obj2_max[0] and
            obj2_min[1] <= obj1_center_y <= obj2_max[1]
        )

        return bottom_is_near_top and center_inside_support_xy

    def compute_area(self, obj1_urdf: str, obj2, env=None) -> Area:
        """
        Return sampling area for obj1 origin so that obj1 is placed on top of obj2
        """
        ws_dict = get_workspace_dict()

        if obj2.name in ws_dict:
            # place at random pos on top of the table
            xy_min, xy_max = get_desk_sampling_area(obj2)
            placing_height = xy_max[2]

        else:
            # place on top of the object at the xy center
            xy_min = xy_max = obj2.get_position()
            placing_height = obj2.get_bounding_box()[4][2]

        placing_height += get_edge_offset_from_urdf(obj1_urdf, obj2.p, axis=2, edge="min", env=env)
        sampling_area = [xy_min[0], xy_max[0],
                         xy_min[1], xy_max[1],
                         placing_height, placing_height]

        return sampling_area

    @staticmethod
    def is_target_obj(obj):
        """
        Return True if object has no collision shape (i.e. is a target marker object, not a solid object)
        """
        return len(obj.p.getCollisionShapeData(obj.uid, -1)) == 0


class Above(AreaPredicate):
    """
    Check whether object1 is above object2
    """

    def check(self, obj1, obj2) -> bool:
        """
        Return True if obj1 is above obj2
        """
        obj1_min, obj1_max = get_bounding_box_limits(obj1)
        obj2_min, obj2_max = get_bounding_box_limits(obj2)

        # 1. obj1 bottom > obj2 top
        if not (obj1_min[2] + TOLERANCE) > obj2_max[2]:
            return False
    
        # 2. obj XY intersect
        return aabb_overlap(obj1_min, obj1_max, obj2_min, obj2_max, 2)

    def compute_area(self, obj1_urdf: str, obj2, env=None) -> Area:
        """
        Return sampling area for obj1 origin so that obj1 is placed above obj2
        """
        sampling_area = OnTop().compute_area(obj1_urdf, obj2, env)

        if sampling_area[-2] < MAX_HEIGHT:
            sampling_area[-1] = MAX_HEIGHT
        else:
            print(f"WARNING: Not able to place {obj1_urdf} above {obj2.name}, max height exceeded.")
            sampling_area = None
        return sampling_area

    
class Below(AreaPredicate):
    """
    Check whether object1 is below object2
    """

    def check(self, obj1, obj2) -> bool:
        """
        Return True if obj1 is below obj2
        """
        return Above().check(obj2, obj1)

    def compute_area(self, obj1_urdf: str, obj2, env=None) -> Area:
        """
        Return sampling area for obj1 origin so that obj1 is placed below obj2
        WARNING: use Above(obj2, obj1) + OnTop(obj1, table) instead if possible,
                 there is no logic to leave room on the table for obj1 when placing obj2
        """
        ws_dict = get_workspace_dict()

        if obj2.name in ws_dict:
            # place at random pos under the table
            xy_min, xy_max = get_desk_sampling_area(obj2)
            placing_height = xy_min[2]

        else:
            # place under the xy center of the object
            xy_min = xy_max = obj2.get_position()
            placing_height = obj2.get_bounding_box()[0][2]

        placing_height -= get_edge_offset_from_urdf(obj1_urdf, obj2.p, axis=2, edge="max", env=env)

        if placing_height < MIN_HEIGHT:
            print(f"WARNING: Not able to place {obj1_urdf} under {obj2.name}, min height exceeded.")
            return None

        sampling_area = [xy_min[0],  xy_max[0],
                         xy_min[1],  xy_max[1],
                         MIN_HEIGHT, placing_height]
        return sampling_area


class Inside(AreaPredicate):
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

    def compute_area(self, obj1_urdf: str, obj2, env=None) -> Area:
        """
        Return sampling area for obj1 so that obj1 is inside obj2,
        obj1 center is at obj2 center
        NOTE: use OnTop() instead for hollow containers (e.g. bowl) to drop obj inside
        """
        # TODO check
        # 1. check if obj1 fits inside obj2
        obj1_min, obj1_max = self._get_bounding_box_from_urdf(obj1_urdf, obj2.get_position(), obj2.p, env)
        obj2_min, obj2_max = get_bounding_box_limits(obj2)
        fits = np.all(np.array([obj2_min, obj1_max]) < np.array([obj1_min, obj2_max]))
        if not fits:
            print(f"WARNING: Not able to place {obj1_urdf} inside {obj2.name}.")
            return None

        # 2. return obj2 pos
        return ObjectAt().compute_area(obj2)


    @staticmethod
    def _get_bounding_box_from_urdf(obj1_urdf: str, obj_pos, pybullet_client, env=None) -> float:
        if os.path.splitext(os.path.basename(obj1_urdf))[0] == "towertarget":
            obj1_urdf = env._get_urdf_filename("kostka")

        temp_obj = env_object.EnvObject(
            obj1_urdf,
            position=obj_pos,
            orientation=[0.0, 0.0, 0.0, 1.0],
            pybullet_client=pybullet_client,
            fixed=True,
        )

        obj_min, obj_max = get_bounding_box_limits(temp_obj)
        pybullet_client.removeBody(temp_obj.uid)
        return obj_min, obj_max


class Empty(Predicate):
    """
    Check if obj's interior is empty (no other object is Inside obj)
    """

    def check(self, obj) -> bool:
        """
        Return True if no other body's bottom-center point lies inside obj's bounding box
        """
        obj_min, obj_max = get_bounding_box_limits(obj)
        obj_area = [obj_min, obj_max]

        # broadphase pre-filter: only bodies whose AABB overlaps obj's at all can be Inside it
        overlapping = obj.p.getOverlappingObjects(obj_min, obj_max)
        if not overlapping:
            return True

        other_uids = {uid for uid, _link in overlapping if uid != obj.uid}
        for uid in other_uids:
            other_min, other_max = obj.p.getAABB(uid)
            other_center_x = (other_min[0] + other_max[0]) / 2
            other_center_y = (other_min[1] + other_max[1]) / 2
            other_bottom = [other_center_x, other_center_y, other_min[2]]

            if pos_inside_area(other_bottom, obj_area):
                return False

        return True


class Near(Predicate):
    """
    Check whether obj1 is close to obj2
    """

    def check(self, obj1, obj2) -> bool:
        """
        Return True if obj1 and obj2 are close (BB almost touching)
        """
        obj1_min, obj1_max = get_bounding_box_limits(obj1)
        obj2_min, obj2_max = get_bounding_box_limits(obj2)
        distance = get_aabb_distance(obj1_min, obj1_max, obj2_min, obj2_max)
        return distance < CLOSE


class Far(Predicate):
    """
    Check whether obj1 is far from obj2
    """
    MIN_DIST = 2

    def check(self, obj1, obj2) -> bool:
        """
        Return True if obj1 and obj2 are far apart
        """
        obj1_min, obj1_max = get_bounding_box_limits(obj1)
        obj2_min, obj2_max = get_bounding_box_limits(obj2)
        distance = get_aabb_distance(obj1_min, obj1_max, obj2_min, obj2_max)
        min_dist = self.MIN_DIST
        return distance > min_dist


class ObjectAt(AreaPredicate):
    """
    Check whether obj1/gripper is almost at the same position as obj2
    """

    def check(self, obj, target_obj) -> bool:
        """
        Return True if obj1 and obj2 are almost at the same position
        """
        obj1_pos = obj.get_position()
        obj2_pos = target_obj.get_position()
        distance = get_point_distance(obj1_pos, obj2_pos)
        # separate dist when placing obj for z based on obj height?
        return distance < CLOSE

    def compute_area(self, obj2) -> Area:
        """
        Return sampling area for obj1 so that obj1 has the same position as obj2
        """
        # TODO: check
        sampling_area = np.repeat(obj2.get_position(), 2)
        return sampling_area.tolist()



class LeftOf(AreaPredicate):
    """
    Check if obj1 is on the left of obj2
    """

    def check(self, obj1, obj2) -> bool:
        """
        Return True if max obj1 Y <= min obj2 Y
        """
        
        _, obj1_max = get_bounding_box_limits(obj1)
        obj2_min, _ = get_bounding_box_limits(obj2)

        # 1. max obj1 Y <= min obj2 Y
        return obj1_max[1] <= (obj2_min[1] + TOLERANCE)

    def compute_area(self, obj1_urdf: str, obj2, env=None) -> Area:
        """
        Return sampling area for obj1 so that obj1 is on the left of obj2
        """
        # TODO: check
        max_y = obj2.get_bounding_box()[0][1]
        max_y -= get_edge_offset_from_urdf(obj1_urdf, obj2.p, axis=1, edge="max", env=env)

        sampling_area = get_unconstrained_working_area()
        if not sampling_area[2] < max_y:
            sampling_area = get_infinite_area()

        sampling_area[3] = max_y
        return sampling_area


class RightOf(AreaPredicate):
    """
    Check if obj1 is on the right of obj2
    """

    def check(self, obj1, obj2) -> bool:
        """
        Return True if min obj1 Y >= max obj2 Y
        """
        return LeftOf().check(obj2, obj1)

    def compute_area(self, obj1_urdf: str, obj2, env=None) -> Area:
        """
        Return sampling area for obj1 so that obj1 is on the right of obj2
        """
        # TODO: check
        min_y = obj2.get_bounding_box()[4][1]
        min_y += get_edge_offset_from_urdf(obj1_urdf, obj2.p, axis=1, edge="min", env=env)

        sampling_area = get_unconstrained_working_area()
        if not min_y < sampling_area[3]:
            sampling_area = get_infinite_area()

        sampling_area[2] = min_y
        return sampling_area


class InFrontOF(AreaPredicate):
    """
    Check if obj1 is in front of obj2
    """

    def check(self, obj1, obj2) -> bool:
        """
        Return True if max obj1 X <= min obj2 X
        """
        
        _, obj1_max = get_bounding_box_limits(obj1)
        obj2_min, _ = get_bounding_box_limits(obj2)

        # 1. max obj1 X <= min obj2 X
        return obj1_max[0] <= (obj2_min[0] + TOLERANCE)

    def compute_area(self, obj1_urdf: str, obj2, env=None) -> Area:
        """
        Return sampling area for obj1 so that obj1 is in front of obj2
        """
        # TODO: check
        max_x = obj2.get_bounding_box()[0][0]
        max_x -= get_edge_offset_from_urdf(obj1_urdf, obj2.p, axis=0, edge="max", env=env)

        sampling_area = get_unconstrained_working_area()
        if not sampling_area[0] < max_x:
            sampling_area = get_infinite_area()

        sampling_area[1] = max_x
        return sampling_area         


class Behind(AreaPredicate):
    """
    Check if obj1 is behind obj2
    """

    def check(self, obj1, obj2) -> bool:
        """
        Return True if min obj1 X >= max obj2 X
        """
        return InFrontOF().check(obj2, obj1)

    def compute_area(self, obj1_urdf: str, obj2, env=None) -> Area:
        """
        Return sampling area for obj1 so that obj1 is behind obj2
        """
        # TODO: check
        min_x = obj2.get_bounding_box()[4][0]
        min_x += get_edge_offset_from_urdf(obj1_urdf, obj2.p, axis=0, edge="min", env=env)

        sampling_area = get_unconstrained_working_area()
        if not min_x < sampling_area[1]:
            sampling_area = get_infinite_area()

        sampling_area[0] = min_x
        return sampling_area


class NextTo(AreaPredicate):
    """
    objects are close at similar height
    """

    def check(self, obj1, obj2) -> bool:
        """
        Check whether obj1 is next to obj2
        """
        if not InSimilarHeight().check(obj1, obj2):
            return False
        
        obj1_min, obj1_max = get_bounding_box_limits(obj1)
        obj2_min, obj2_max = get_bounding_box_limits(obj2)
        distance = get_aabb_distance(obj1_min, obj1_max, obj2_min, obj2_max)
        return distance < 3*CLOSE

    def compute_area(self, obj1_urdf: str, obj2, env=None) -> Area:
        """
        Return base sampling area for obj1 close to obj2, at obj2's own height.
        WARNING: high chance of collision between the objects - meant to be
                 further restricted to one side (LeftOf/RightOf/InFrontOF/Behind,
                 or _apply_on_random_side) before sampling from it
        """
        obj2_min, obj2_max = get_bounding_box_limits(obj2)
        margin = self._get_side_offset_from_urdf(obj1_urdf, obj2.p, env) + 2*TOLERANCE

        return [
            obj2_min[0] - margin,    obj2_max[0] + margin,
            obj2_min[1] - margin,    obj2_max[1] + margin,
            obj2_min[2] - TOLERANCE, obj2_max[2] + TOLERANCE,
        ]

    @staticmethod
    def _get_side_offset_from_urdf(obj1_urdf: str, pybullet_client, env=None) -> float:
        """
        Return the largest of obj1's own front/back/left/right offsets (origin
        to bounding-box edge), used as NextTo's margin so the base area always
        contains whichever precise boundary LeftOf/RightOf/InFrontOF/Behind end
        up computing for obj1 (their per-direction offsets can differ if obj1's
        origin isn't exactly centered).
        Measures obj1 once (unlike calling get_edge_offset_from_urdf 4x, which
        would spawn 4 separate temp objects for the same measurement).
        """
        obj_min, obj_max, obj_pos = get_object_extent_from_urdf(obj1_urdf, pybullet_client, env)

        front_offset = obj_pos[0] - obj_min[0]
        back_offset = obj_max[0] - obj_pos[0]
        left_offset = obj_pos[1] - obj_min[1]
        right_offset = obj_max[1] - obj_pos[1]
        return max(front_offset, back_offset, left_offset, right_offset)


class InSimilarHeight():
    """
    Check if objects are at similar height level
    """

    def check(self, obj1, obj2) -> bool:
        """
        Return True if obj1 and obj2 Z ranges overlap
        """
        obj1_min, obj1_max = get_bounding_box_limits(obj1)
        obj2_min, obj2_max = get_bounding_box_limits(obj2)
        return (obj1_min[2] <= obj2_max[2] + TOLERANCE and
                obj2_min[2] <= obj1_max[2] + TOLERANCE)


class Upright(Predicate):
    """
    Check if object is still in its base upright position (i.e. has not been tipped over)
    """

    def check(self, obj) -> bool:
        """
        Return True if obj's local up axis has not tilted away from its spawn orientation
        beyond TOLERANCE_DEG (ignores rotation about the object's own vertical axis)
        """
        init_matrix = np.array(obj.p.getMatrixFromQuaternion(obj.init_orientation)).reshape(3, 3)
        current_matrix = np.array(obj.p.getMatrixFromQuaternion(obj.get_orientation())).reshape(3, 3)

        init_up = init_matrix @ np.array([0, 0, 1])
        current_up = current_matrix @ np.array([0, 0, 1])

        cos_angle = np.clip(np.dot(init_up, current_up), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_angle))
        return angle_deg <= TOLERANCE_DEG


class GripperStatus(Predicate):
    """
    Check whether the gripper is in the desired state

    Supports predicates GripperOpen() and GripperClosed()

    Note:
        GripperClosed() is not equivalent to not(GripperOpen()),
        because the gripper has also a neutral state
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

    Uses gripper joint status + proximity rather than magnetized_objects, because
    magnetize_object has an independent 0.1m distance gate that can fire later than
    the protoreward's arm_solved (85% progress), causing inconsistent results.
    """
    DISTANCE_THRESHOLD = 0.15

    def check(self, gripper, obj) -> bool:
        closed = GripperStatus().check(gripper, "close")
        if not closed:
            return False
        distance = np.linalg.norm(
            np.asarray(gripper.get_position()) - np.asarray(obj.get_position()[:3])
        )
        return distance <= self.DISTANCE_THRESHOLD



# ---------- predicate parsing / resolving ----------

@dataclass
class PredicateCall:
    predicate: str
    args: list[str]
    negated: bool = False


class PredicateResolver:
    """
    Base class to check if all predicates are satisfied
    """

    predicate_key: str | None = None

    def check(self, placed_objects: dict, env, predicates: dict, grip_type: str | None = None) -> bool:
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
            if not self._check_predicate(predicate, objects_by_name, grip_type):
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
        Convert list of predicates strings to list predicates objects
        ['OnTop(apple,table)', ...] -> [PredicateCall(name='OnTop', args=['apple','table']), ...]
        """
        parsed_predicates = []
        for predicate in predicates:
            match = re.fullmatch(r"\s*(\w+)\s*\((.*)\)\s*(?::\s*(True|False))?\s*", predicate)

            if match is None:
                raise ValueError(f"Invalid predicate format: {predicate}")

            name = match.group(1)
            args_text = match.group(2)
            args = [arg.strip() for arg in args_text.split(",") if arg.strip()]
            negated = match.group(3) == "False" if match.group(3) is not None else False

            parsed_predicates.append(PredicateCall(predicate=name, args=args, negated=negated))

        return parsed_predicates
    
    def _filter_obj_predicates(self, predicates, obj_name):
        """
        Select current object predicates
        """
        if not predicates:
            return []

        return [p for p in predicates if self._predicate_has_obj_as_first_arg(p, obj_name)]

    @staticmethod
    def _check_predicate(predicate: PredicateCall, objects_by_name: dict, grip_type: str | None = None) -> bool:
        """
        Evaluate predicate and apply negation. Returns the final bool.
        """

        gripper = objects_by_name.get("gripper")

        if predicate.predicate == "GripperOpen":
            status = "close" if predicate.negated else "open"  # Open/Closed are not logic complements (neutral state)
            return GripperStatus().check(gripper, status)

        if predicate.predicate == "GripperClosed":
            status = "open" if predicate.negated else "close"
            return GripperStatus().check(gripper, status)


        if predicate.predicate == "IsReachable":
            obj_name = predicate.args[0]
            result = IsReachable().check(objects_by_name["robot"], objects_by_name[obj_name], grip_type)

        elif predicate.predicate == "OnTop":
            obj_name, support_name = predicate.args
            result = OnTop().check(objects_by_name[obj_name], objects_by_name[support_name])

        elif predicate.predicate == "Touching":
            obj1_name, obj2_name = predicate.args
            result = Touching().check(objects_by_name[obj1_name], objects_by_name[obj2_name])

        elif predicate.predicate == "Near":
            obj1_name, obj2_name = predicate.args
            result = Near().check(objects_by_name[obj1_name], objects_by_name[obj2_name])

        elif predicate.predicate == "ObjectAt":
            obj_name, target_name = predicate.args
            result = ObjectAt().check(objects_by_name[obj_name], objects_by_name[target_name])

        elif predicate.predicate == "GripperAt":
            target_name = predicate.args[0]
            result = ObjectAt().check(gripper, objects_by_name[target_name])

        elif predicate.predicate == "IsHolding":
            target_name = predicate.args[0]
            result = IsHolding().check(gripper, objects_by_name[target_name])

        elif predicate.predicate == "Above":
            obj1_name, obj2_name = predicate.args
            result = Above().check(objects_by_name[obj1_name], objects_by_name[obj2_name])

        elif predicate.predicate == "Below":
            obj1_name, obj2_name = predicate.args
            result = Below().check(objects_by_name[obj1_name], objects_by_name[obj2_name])

        elif predicate.predicate == "LeftOf":
            obj1_name, obj2_name = predicate.args
            result = LeftOf().check(objects_by_name[obj1_name], objects_by_name[obj2_name])

        elif predicate.predicate == "RightOf":
            obj1_name, obj2_name = predicate.args
            result = RightOf().check(objects_by_name[obj1_name], objects_by_name[obj2_name])

        elif predicate.predicate == "InFrontOF":
            obj1_name, obj2_name = predicate.args
            result = InFrontOF().check(objects_by_name[obj1_name], objects_by_name[obj2_name])

        elif predicate.predicate == "Behind":
            obj1_name, obj2_name = predicate.args
            result = Behind().check(objects_by_name[obj1_name], objects_by_name[obj2_name])

        elif predicate.predicate == "NextTo":
            obj1_name, obj2_name = predicate.args
            result = NextTo().check(objects_by_name[obj1_name], objects_by_name[obj2_name])

        elif predicate.predicate == "Far":
            obj1_name, obj2_name = predicate.args
            result = Far().check(objects_by_name[obj1_name], objects_by_name[obj2_name])

        elif predicate.predicate == "Upright":
            obj_name = predicate.args[0]
            result = Upright().check(objects_by_name[obj_name])

        elif predicate.predicate == "Empty":
            obj_name = predicate.args[0]
            result = Empty().check(objects_by_name[obj_name])

        else:
            raise ValueError(f"Unknown predicate: {predicate.predicate}")

        return (not result) if predicate.negated else result

    @staticmethod
    def _build_object_lookup(env, placed_objects: dict) -> dict:
        """
        Build {"name": object} lookup for env objects
        """
        objects_by_name = {
            "table": env.static_scene_objects[env.workspace],
            "workspace": env.static_scene_objects[env.workspace],
            "robot": env.robot,
            "gripper": env.robot,
        }

        # Active task objects
        for key in ["actual_state", "goal_state"]:
            obj = placed_objects.get(key)

            if hasattr(obj, "name"):
                objects_by_name[obj.name] = obj
        
        # Other env objects
        for obj in placed_objects.get("env_objects", []):
            if hasattr(obj, "name"):
                objects_by_name[obj.name] = obj

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

    def get_area(self, obj_info, table, robot, predicates, placed_objects=None, env=None, grip_type: str | None = None) -> Area | None:
        """
        Compute object sampling area from init predicates
        """
        predicates = predicates["init"] if predicates else []
        placed_objects = placed_objects or {}
        placed_objects.setdefault("table", table)
        placed_objects.setdefault("workspace", table)
        obj1_urdf = obj_info["urdf"]
        predicates = self._filter_obj_predicates(predicates, obj_info["obj_name"])

        if not predicates:
            # return reachable area on table if not specified
            table_area = OnTop().compute_area(obj1_urdf, table, env)
            reachable_area = IsReachable().compute_area(robot, grip_type)
            reachable_table_area = get_range_intersection(table_area, reachable_area)
            return reachable_table_area if reachable_table_area is not None else table_area

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
                env=env,
            )

        for next_to_predicate in predicate_map.get("NextTo", []):
            # base reference: narrow to close to obj2 first, then pick a side
            area = self._apply_nextto_area(
                current_area=area,
                predicate=next_to_predicate,
                table=table,
                obj1_urdf=obj1_urdf,
                placed_objects=placed_objects,
                env=env,
            )
            has_side = self._has_matching_side(
                predicate_map, ("LeftOf", "RightOf", "InFrontOF", "Behind"), next_to_predicate.args[1]
            )
            if not has_side:
                area = self._apply_on_random_side(
                    current_area=area,
                    predicate=next_to_predicate,
                    table=table,
                    obj1_urdf=obj1_urdf,
                    placed_objects=placed_objects,
                    env=env,
                )

        for leftof_predicate in predicate_map.get("LeftOf", []):
            area = self._apply_side_area(
                LeftOf, "LeftOf", area, leftof_predicate, table, obj1_urdf, placed_objects, env
            )

        for rightof_predicate in predicate_map.get("RightOf", []):
            area = self._apply_side_area(
                RightOf, "RightOf", area, rightof_predicate, table, obj1_urdf, placed_objects, env
            )

        for infrontof_predicate in predicate_map.get("InFrontOF", []):
            area = self._apply_side_area(
                InFrontOF, "InFrontOF", area, infrontof_predicate, table, obj1_urdf, placed_objects, env
            )

        for behind_predicate in predicate_map.get("Behind", []):
            area = self._apply_side_area(
                Behind, "Behind", area, behind_predicate, table, obj1_urdf, placed_objects, env
            )

        for above_predicate in predicate_map.get("Above", []):
            area = self._apply_above_area(
                current_area=area,
                predicate=above_predicate,
                table=table,
                obj1_urdf=obj1_urdf,
                placed_objects=placed_objects,
                env=env,
            )

        for under_predicate in predicate_map.get("Below", []):
            area = self._apply_under_area(
                current_area=area,
                predicate=under_predicate,
                table=table,
                obj1_urdf=obj1_urdf,
                placed_objects=placed_objects,
                env=env,
            )

        for reachable_predicate in predicate_map.get("IsReachable", []):
            area = self._apply_reachable_area(
                current_area=area,
                predicate=reachable_predicate,
                robot=robot,
                grip_type=grip_type,
            )
            break  # repetitive input

        if area is None:
            # Keep reset robust when predicate constraints do not overlap
            # (common with new robot/workspace combinations).
            return OnTop().compute_area(obj1_urdf, table, env)

        return area

    @staticmethod
    def _has_matching_side(predicate_map, side_names, obj2_name) -> bool:
        """
        Return True if any predicate under one of side_names targets the same obj2_name
        as the NextTo predicate (e.g. NextTo(obj1,X) should only skip the random-side
        step for LeftOf(obj1,X), not LeftOf(obj1,Y) referencing a different object)
        """
        return any(
            len(p.args) == 2 and p.args[1] == obj2_name
            for name in side_names
            for p in predicate_map.get(name, [])
        )

    def _apply_on_top_area(
            self, current_area: Area, predicate: PredicateCall, table, obj1_urdf: str, placed_objects, env=None
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
            if not support_object:
                raise ValueError(
                    f"Cannot compute OnTop area for '{predicate.args[0]}'. "
                    f"Supporting object '{obj2_name}' has not been placed yet."
                )
        
        on_top_area = OnTop().compute_area(obj1_urdf, support_object, env)
        return get_range_intersection(current_area, on_top_area)

    def _apply_above_area(
            self, current_area: Area, predicate: PredicateCall, table, obj1_urdf: str, placed_objects, env=None
            ) -> Area | None:
        """
        Apply Above(obj1, obj2) as an area constraint
        The support object has to be already placed
        """
        if len(predicate.args) != 2:
            raise ValueError(f"Above expects 2 arguments, got {predicate.args}")

        obj1_name, obj2_name = predicate.args

        if obj2_name == "table":
            support_object = table
        else:
            support_object = placed_objects.get(obj2_name)
            if not support_object:
                raise ValueError(
                    f"Cannot compute Above area for '{predicate.args[0]}'. "
                    f"Supporting object '{obj2_name}' has not been placed yet."
                )

        above_area = Above().compute_area(obj1_urdf, support_object, env)
        if above_area is None:
            return None

        return get_range_intersection(current_area, above_area)

    def _apply_under_area(
            self, current_area: Area, predicate: PredicateCall, table, obj1_urdf: str, placed_objects, env=None
            ) -> Area | None:
        """
        Apply Below(obj1, obj2) as an area constraint
        The reference object has to be already placed
        """
        if len(predicate.args) != 2:
            raise ValueError(f"Below expects 2 arguments, got {predicate.args}")

        obj1_name, obj2_name = predicate.args

        if obj2_name == "table":
            reference_object = table
        else:
            reference_object = placed_objects.get(obj2_name)
            if not reference_object:
                raise ValueError(
                    f"Cannot compute Below area for '{predicate.args[0]}'. "
                    f"Reference object '{obj2_name}' has not been placed yet."
                )

        under_area = Below().compute_area(obj1_urdf, reference_object, env)
        if under_area is None:
            return None

        return get_range_intersection(current_area, under_area)

    @staticmethod
    def _apply_side_area(
            predicate_cls, predicate_name: str, current_area: Area, predicate: PredicateCall,
            table, obj1_urdf: str, placed_objects, env=None
            ) -> Area | None:
        """
        Shared logic for LeftOf/RightOf/InFrontOF/Behind area constraints:
        resolve the reference object, compute_area() with predicate_cls, intersect.
        """
        if len(predicate.args) != 2:
            raise ValueError(f"{predicate_name} expects 2 arguments, got {predicate.args}")

        obj1_name, obj2_name = predicate.args

        if obj2_name == "table":
            reference_object = table
        else:
            reference_object = placed_objects.get(obj2_name)
            if not reference_object:
                raise ValueError(
                    f"Cannot compute {predicate_name} area for '{predicate.args[0]}'. "
                    f"Reference object '{obj2_name}' has not been placed yet."
                )

        side_area = predicate_cls().compute_area(obj1_urdf, reference_object, env)
        if side_area is None:
            return None

        return get_range_intersection(current_area, side_area)

    def _apply_on_random_side(
            self, current_area: Area, predicate: PredicateCall, table, obj1_urdf: str, placed_objects, env=None
            ) -> Area | None:
        """
        Choose a random side (Left/Right/InFront/Behind) to avoid collision of obj1 and obj2
        """
        side_predicate = random.choice([LeftOf, RightOf, InFrontOF, Behind])
        side_name = side_predicate.__name__
        return self._apply_side_area(side_predicate, side_name, current_area, predicate, table, obj1_urdf, placed_objects, env)

    def _apply_nextto_area(
            self, current_area: Area, predicate: PredicateCall, table, obj1_urdf: str, placed_objects, env=None
            ) -> Area | None:
        """
        Apply NextTo(obj1, obj2) as an area constraint (base reference, narrow
        to close to obj2 before optionally picking a side with _apply_on_random_side)
        """
        return self._apply_side_area(NextTo, "NextTo", current_area, predicate, table, obj1_urdf, placed_objects, env)

    def _apply_reachable_area(
        self, current_area: Area, predicate: PredicateCall, robot, grip_type: str | None = None
        ) -> Area | None:
        """
        Apply IsReachable(obj) as an area constraint
        """
        if len(predicate.args) != 1:
            raise ValueError(f"IsReachable expects 1 argument, got {predicate.args}")

        reachable_area = IsReachable().compute_area(robot, grip_type)
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


class SubgoalPredicateResolver(PredicateResolver):
    """
    Check predicates describing an intermediate subgoal state, e.g. "subgoal1", "subgoal2".
    """

    def __init__(self, subgoal_index: int):
        self.predicate_key = f"subgoal{subgoal_index}"

