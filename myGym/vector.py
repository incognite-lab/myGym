import numpy as np
import math

class Vector:

    def __init__(self, beginning, end, env):
        self.env = env

        self.beginning = beginning
        self.end = end

        self.vector     = self.move_to_origin([beginning, end])
        self.norm       = self.count_norm()
        self.normalized = self.get_normalized()


    ### self modifiers:

    def set_len(self, len):
        self.multiply(1/self.norm)
        self.multiply(len)

    def multiply(self, multiplier):
        self.vector = np.array(self.vector) * multiplier

    def count_norm(self):
        return math.sqrt(np.dot(self.vector, self.vector))

    def get_normalized(self):
        if self.norm == 0:
            return 0
        return self.vector * (1/self.norm)

    def normalize(self):
        self.vector = self.multiply_vector(self.vector, 1/self.norm)

    def move_to_origin(self, vector):
        a = vector[1][0] - vector[0][0]
        b = vector[1][1] - vector[0][1]
        c = vector[1][2] - vector[0][2]       
        
        return np.array([a, b, c])
    
    def visualize(self, origin=[0,0,0], color=(0,0,0), time=0.1):

        self.env.p.addUserDebugLine(origin, np.add(np.array(origin), np.array(self.vector)), lineColorRGB=color, lineWidth = 1, lifeTime = time)

    def add(self, v2):
        r = []

        for i in range(len(self.vector)):
            r.append(self.vector[i] + v2.vector[i])

        self.vector = r

    def rotate_with_matrix(self, matrix):
        self.vector = matrix.dot(self.vector) 

    ### interactive:

    def add_vector(self, v2):
        r = []

        for i in range(len(self.vector)):
            r.append(self.vector[i] + v2.vector[i])

        return r

    def get_dot_product(self, v2):
        product = 0

        for i in range(len(self.vector)):
            product += self.vector[i]* v2.vector[i]
        return product

    def get_align(self, v2):
        align = 0

        for i in range(len(self.normalized)):
            align += self.normalized[i]* v2.normalized[i]
        return align
