import random

import pkg_resources
import os, sys, time, yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
import json, commentjson
import quaternion

q = np.array([0,  0.707, 0,  0.707])
q2 = np.quaternion(4.9130e-5, -9.2098e-06, -0.18425, 0.98287)
v = np.array([0.0, 0, -1.0])
def quaternion_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w,x,y,z])

def q_inv(q):
    w, x, y, z = q
    norm = w**2 + x**2 + y**2 + z**2
    print(norm)
    return np.array([w,-x,-y,-z])/norm

def rotate(vec, quat):
    w, x, y, z = quat
    norm = w**2 + x**2 + y**2 + z**2
    quat = quat/norm
    print(norm)
    qcon = q_inv(quat)
    rvect = quaternion_mult(quat, np.concatenate(([0], vec)))
    rvect = quaternion_mult(rvect, qcon)[1:]
    return rvect

print(rotate(v, q))
#print(q2.rotate(v))