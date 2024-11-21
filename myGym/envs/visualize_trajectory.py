from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
#from stable_baselines import results_plotter
import os
import math
from math import sqrt, fabs, exp, pi, asin
from myGym.utils.vector import Vector
import random


def skew_symmetric(vector) -> np.ndarray:
    return np.asarray([[0, - vector[2], vector[1]], [vector[2], 0, -vector[0]], [-vector[1], vector[0], 0]])


def create_trajectory(fx, fy, fz, t):
    """General trajectory creator. Pass functions fx, fy, fz and parametric vector t to create any 3D trajectory"""
    trajectory = np.asarray([fx(t), fy(t), fz(t)])
    return trajectory


def create_line(point1, point2, step=0.01):
    """Creates line from point1 to point2 -> vectors of length 1/step"""
    t = np.arange(0, 1, step)

    def linemaker_x(t):
        "axis: 0 = x, 1 = y, 2 = z"
        return (point2[0] - point1[0]) * t + point1[0]
    def linemaker_y(t):
        return (point2[1] - point1[1]) * t + point1[1]
    def linemaker_z(t):
        return (point2[2] - point1[2]) * t + point1[2]
    return create_trajectory(fx=linemaker_x, fy=linemaker_y, fz=linemaker_z, t=t)


def create_circular_trajectory(center, radius, rot_vector, arc=np.pi, step=0.01):
    """Creates a 2D circular trajectory in 3D space.
    params: center ([x,y,z]), radius (float): self-explanatory
            rot_vector ([x,y,z]): Axis of rotation. Angle of rotation is norm of rot_vector
            arc (radians): 2pi means full circle, 1 pi means half a circle etc...
    """
    phi = np.arange(0, arc, step)
    v = np.asarray(rot_vector)
    theta = np.linalg.norm(v)

    # creation of unrotated circle of given radius located at [0,0,0]
    base_circle = np.asarray([np.cos(phi) * radius, np.sin(phi) * radius, [0] * len(phi)])
    rotation = np.eye(3)
    print(theta)
    if theta != 0.0:
        normalized_v = v * (1 / theta)
        # Rodrigues' formula:
        rotation = (np.eye(3) + np.sin(theta) * skew_symmetric(normalized_v) +
                    (1 - np.cos(theta)) * np.matmul(skew_symmetric(normalized_v),
                                                    skew_symmetric(normalized_v)))
    rotated = np.asarray([[], [], []])
    print(rotation)
    for i in range(len(phi)):
        rotated_v = np.asarray([np.matmul(rotation, base_circle[:3, i])])
        rotated = np.append(rotated, np.transpose(rotated_v), axis=1)
    # moving circle to its center
    move = np.asarray(center)
    final_circle = np.transpose(np.transpose(rotated) + move)
    return final_circle


ax = plt.figure().add_subplot(projection='3d')
line = create_line([5, -2, 3], [-8,-4, 2])
circle = create_circular_trajectory([0,0,0], 2, rot_vector=np.asarray([1, 1, 0])*sqrt(1)*np.pi/4, arc=np.pi*2)

#ax.plot(line[0], line[1], line[2])
ax.plot(circle[0], circle[1], circle[2])
ax.legend()
plt.show()
print("PLOTTED 3D")


