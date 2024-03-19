from ikpy.chain import Chain
import matplotlib.pyplot
from mpl_toolkits.mplot3d import Axes3D
import time

my_chain = Chain.from_urdf_file("./envs/robots/nico/nico_ik.urdf")
#my_chain = Chain.from_urdf_file("/home/michal/ikpy/resources/poppy_ergo.URDF")
target_position = [ 0.2 , -0.2, 0.4]
target_joints = [0,0,0,0,0,0,0,0]
real_frame = my_chain.forward_kinematics(target_joints)
initial_position = real_frame[:3, 3]
print(initial_position)


ax = matplotlib.pyplot.figure().add_subplot(111, projection='3d')

my_chain.plot(my_chain.inverse_kinematics(target_position), ax)
matplotlib.pyplot.show()
time.sleep(10)