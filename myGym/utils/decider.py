import math
import numpy as np

class Shape:
    def __init__(self, shape):
        self.A      = shape[0]
        self.B      = shape[1]
        self.r      = shape[2]
        self.id     = shape[3]

    def get_center(self, p1, p2):
        l = len(p1)
        s = [0]*l
        for i in l:
            s[i] = (p1[i] + p2[i])/2
        return s

    def get_distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

    def triangle_height(self, a, b, c):
        p = a+b+c
        p = p/2
        one = 2/a
        two = p*(p-a)
        three = (p-b)
        four = (p-c)
        five = two*three*four
        six = math.sqrt(five)
        return one*six

    def distance_of_point_from_abscissa(self, A, B, point):
        a = self.task.calc_distance(A, B)
        b = self.task.calc_distance(A, point)
        c = self.task.calc_distance(point, B)

        height_a = self.triangle_height(a, b, c)
        distance = height_a

        if b > a:
            distance = c
        if c > a:
            distance = c

        return distance

    def is_inside(args) -> bool:
        pass

class Sphere(Shape):
    def __init__(self, shape):
        super().__init__(shape)

    def is_inside(self, point):
        # is point in shape?
        S  = self.get_center(self.A, self.B)
        SR = self.r
        SA = self.get_distance(S, self.A)
        SB = self.get_distance(S, self.B)
        SP = self.get_distance(S, point)

        if SP <= SR and SP <= SA:
            return self.id
        elif SP <= SR and SP > SA:
            rp = math.cos(SA-(SR-SA))
            cone = Cone(["cone", self.A, self.B, rp, self.id, self.lock])
            if cone.decide() == self.id:
                return self.id
        else:
            return False
            
class Cone(Shape):
    def __init__(self, shape):
        super().__init__(shape)

    def is_inside(self, point):
        AB = self.get_distance(self.A, self.B)
        TP = self.distance_of_point_from_abscissa(self.A, self.B, point)
        BP = self.get_distance(self.B, point)
        BT = math.sqrt(BP**2 - TP**2)

        s  = AB/BT
        rp = self.r/s

        if TP < rp:
            return self.id
        return False

class Cube(Shape):
    def __init__(self, shape):
        super().__init__(shape)

    def is_inside(self, point):
        p = np.array([10.5, .5, 0.01]) # inspected point

        a0 = np.array([self.A[0]-self.r, self.A[0], self.A[0]+self.r])
        a1 = np.array([self.A[0]-self.r, self.A[0], self.A[0]-self.r])
        a2 = np.array([self.A[0]+self.r, self.A[0], self.A[0]+self.r])
        a3 = np.array([self.A[0]+self.r, self.A[0], self.A[0]-self.r])

        b0 = np.array([self.B[0]-self.r, self.B[0], self.B[0]+self.r])
        b1 = np.array([self.B[0]-self.r, self.B[0], self.B[0]-self.r])
        b2 = np.array([self.B[0]+self.r, self.B[0], self.B[0]+self.r])
        b3 = np.array([self.B[0]+self.r, self.B[0], self.B[0]-self.r])

        u1 = a1 - a0
        u2 = a2 - a0
        u3 = a3 - a0
        up = p - a0

        v1 = b1 - b0
        v2 = b2 - b0
        v3 = b3 - b0
        vp = p - b0

        is_inside = (np.dot(u1, up) > 0 and np.dot(u2, up) > 0 and np.dot(u3, up) > 0 and
                    np.dot(v1, vp) > 0 and np.dot(v2, vp) > 0 and np.dot(v3, vp) > 0)

        if is_inside:
            return self.id
        return is_inside

class Cylinder(Shape):
    def __init__(self, shape):
        super().__init__(shape)

    def is_inside(self, point):
        AB = self.get_distance(self.A, self.B)
        TP = self.distance_of_point_from_abscissa(self.A, self.B, point)
        BP = self.get_distance(self.B, point)
        BT = math.sqrt(BP**2 - TP**2)

        if TP < self.r:
            return self.id
        return False

class Decider():
    def __init__(self, lock=False):
        """
        lock = id of network locked as owner till end of current episode
        """
        self.locked = lock

    def reset(self):
        """
        Call on end of each training episode
        """
        self.lock = False

    def decide(self, shapes, point):
        """
        shapes:         list of definitions of deciding shapes
        shape[0]:       type of shape (sphere, cone, cylinder, cube)
        shape[1,2]:     endpoints of shape
        shape[3]:       half of diameter of shape
        shape[4]:       id of network which should active inside this shape (cannot be 0 it is reserved as default network id)
        shape[5]:       continue using this network for the rest of the episode?        
        
        point:          if is inside defined shape activate chosen network
        
        example of cube:
        ["cube", [0,0,0], [1,1,1], 0.1, 3, False] (in this case r stands for r of inside cylinder) 
        if point gets into this shape, model will switch to network with id 3 (shape[4]) and will swith on another after leaving the shape (shape[5])
        """
        
        if self.locked:
            return self.locked

        return_id = None
        for shape in shapes:
            hap = shape[1:-1]
            if shape[0] == "sphere":
                Shape = Sphere(hap)
                return_id = self.get_return_id(point, shape, Shape)
            elif shape[0] == "cone":
                Shape = Cone(hap)
                return_id = self.get_return_id(point, shape, Shape)
            elif shape[0] == "cube":
                Shape = Cube(hap)
                return_id = self.get_return_id(point, shape, Shape)
            elif shape[0] == "cylinder":
                Shape = Cylinder(hap)
                return_id = self.get_return_id(point, shape, Shape)
            else:
                exit("decision error")

        if return_id:
            return return_id
        return 0 # id reserved for default network (outside of all chosen shapes)

    def get_return_id(self, point, shape, Shape):
        id = Shape.is_inside(point)
        if id:
            if shape[-1]:
                self.locked = shape[-1]
            return id
        return False


