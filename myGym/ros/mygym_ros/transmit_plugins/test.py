import sys
import struct
import subprocess
import time
import random

# Start the receiver script in a subprocess with a pipe
receiver_proc = subprocess.Popen(
    ['python', 'ros1_noetic_receiver.py'],
    stdin=subprocess.PIPE,  # Pipe for binary data
    bufsize=0  # No buffering for real-time transmission
)

try:
    while True:
        # Generate a sample data point (7 floats)
        data_point = [random.random() for _ in range(7)]
        print(f"Sending: {data_point}")

        # Pack into binary format (7 floats = 28 bytes)
        binary_data = struct.pack('7f', *data_point)

        # Send through the pipe
        receiver_proc.stdin.write(binary_data)
        receiver_proc.stdin.flush()  # Force immediate send

        # Simulate processing time
        time.sleep(0.01)  # Adjust as needed

except (BrokenPipeError, KeyboardInterrupt):
    receiver_proc.kill()
