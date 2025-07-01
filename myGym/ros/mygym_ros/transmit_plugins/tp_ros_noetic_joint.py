import sys
import subprocess
import time
import os
import traceback as tb
from zmq_comm.srv import ServiceClient

import numpy as np


class RosNoeticTransmitPlugin:
    def __init__(self):
        # Start the receiver script in a subprocess with a pipe
        transmitter_path = os.path.join(os.path.dirname(__file__), 'ros_noetic_joint_receiver.py')
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
