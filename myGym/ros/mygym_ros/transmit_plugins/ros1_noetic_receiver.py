from calendar import c
import rospy
from geometry_msgs.msg import Pose, Quaternion, Point
from std_msgs.msg import String
import sys
import struct
from threading import Thread
from zmq_comm.srv import ServiceServer


# Each data point is 7 floats = 28 bytes
CHUNK_SIZE = 28  # 7 floats * 4 bytes per float


class RospyTramsiter:
    def __init__(self):
        print("ros1_noetic_receiver.py")
        rospy.init_node('receiver', anonymous=True)
        self.pub = rospy.Publisher('target_pose', Pose, queue_size=10)
        # self._pipe_thread = Thread(target=self.callback, daemon=True)
        # self._pipe_thread.start()
        self.server = ServiceServer(callback=self.data_callback)

    def data_callback(self, data_dict):
        print("data_callback")
        print(f"Received: {data_dict}")
        data = data_dict[b"data"]
        msg = Pose(
            position=Point(x=data[0], y=data[1], z=data[2]),
            orientation=Quaternion(x=data[3], y=data[4], z=data[5], w=data[6])
        )
        self.pub.publish(msg)

    def callback(self):
        try:
            while True:
                # Read exactly 28 bytes from stdin
                binary_data = sys.stdin.buffer.read(CHUNK_SIZE)
                if not binary_data:
                    break  # Exit on EOF

                # Unpack to 7 floats
                data_point = struct.unpack('7f', binary_data)

                # Process the data point (e.g., save to DB, visualize, etc.)
                print(f"Received: {data_point}")
                data = Pose(
                    position=Point(x=data_point[0], y=data_point[1], z=data_point[2]),
                    orientation=Quaternion(x=data_point[3], y=data_point[4], z=data_point[5], w=data_point[6])
                )
                self.pub.publish(data)

        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    spinner = RospyTramsiter()
    rospy.spin()
