import sys
import struct
import subprocess
import time
import os
import traceback as tb
from zmq_comm.srv import ServiceClient

import numpy as np


class RosNoeticTransmitPlugin:
    def __init__(self):
        # Start the receiver script in a subprocess with a pipe
        transmitter_path = os.path.join(os.path.dirname(__file__), 'ros1_noetic_receiver.py')
        self.receiver_proc = subprocess.Popen(
            [
                # 'conda', 'deactivate', '&&'
                '.', '/opt/ros/noetic/setup.bash', '&&',
                'python', '-c "import rospy; print(\"dpy\")"', '&&'
                'python', transmitter_path
            ],
            stdin=subprocess.PIPE,  # Pipe for binary data
            bufsize=0,  # No buffering for real-time transmission
            shell=True,
        )
        self.client = ServiceClient()
        print("pipe open")

    def send_data(self, data):
        # data = np.asarray(data, dtype=np.float16).tolist()
        data_dict = {
            "data": data
        }
        print(f"Sending: {data_dict}")
        ret = self.client.call(data_dict)
        print(f"Response: {ret}")

    # def send_data(self, data):
    #     try:
    #         data = np.asarray(data, dtype=np.float16).tolist()
    #         # Generate a sample data point (7 floats)
    #         print(f"Sending: {data}")

    #         # Pack into binary format (7 floats = 28 bytes)
    #         binary_data = struct.pack('7f', *data)

    #         # Send through the pipe
    #         self.receiver_proc.stdin.write(binary_data)
    #         self.receiver_proc.stdin.flush()  # Force immediate send
    #         # while True:
    #         #     # Generate a sample data point (7 floats)
    #         #     data_point = [random.random() for _ in range(7)]
    #         #     print(f"Sending: {data_point}")

    #         #     # Pack into binary format (7 floats = 28 bytes)
    #         #     binary_data = struct.pack('7f', *data_point)

    #         #     # Send through the pipe
    #         #     self.receiver_proc.stdin.write(binary_data)
    #         #     self.receiver_proc.stdin.flush()  # Force immediate send

    #         #     # Simulate processing time
    #         #     time.sleep(0.01)  # Adjust as needed

    #     except (BrokenPipeError, KeyboardInterrupt):
    #         tb.print_exc()
    #         self.receiver_proc.kill()
